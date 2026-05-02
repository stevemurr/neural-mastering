// Composite TONE CLAP plugin: 1 dylib that wires
//
//   audio → LufsLeveler → ort(autoeq controller) → SpectralMaskEq
//                       → RationalA (saturator)  → ort(la2a)
//                       → TruePeakCeiling → output trim
//
// The composite has two host-exposed knobs (AMT, TRM) defined in the
// composite_meta. AMT remaps to per-stage params (saturator pre/post + wet/dry,
// LA-2A peak reduction, auto-EQ wet/dry); TRM is a final linear gain.
//
// Block-rate streaming: the auto-EQ controller and the LA-2A LSTM both want
// fixed 128-sample blocks (cond_block_size). The plugin accumulates host
// audio into a 128-sample input ring per channel and flushes the chain block
// by block; output samples come out of an output ring with the same depth.
// Total internal latency = 128 (one block of accumulator) + ceiling lookahead.
//
// v1 limitations:
//   - arm64 macOS only (parent CMakeLists guards against other platforms)
//   - CPU execution provider only
//   - per-block parameter snapshot (no sample-accurate smoothing)
//   - refuses activation if host sample rate != composite_meta.sample_rate

#include <algorithm>
#include <atomic>
#include <cctype>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "tone_gui.h"

#include <dlfcn.h>

#include <clap/clap.h>
#include <onnxruntime_cxx_api.h>

#include "composite_meta.hpp"
#include "dimension_d.hpp"
#include "lufs_leveler.hpp"
#include "meta.hpp"
#include "param_id.hpp"
#include "rational_a.hpp"
#include "spectral_mask_eq.hpp"
#include "true_peak_ceiling.hpp"

namespace nablafx_tone {

namespace fs = std::filesystem;
using nablafx::CompositeMeta;
using nablafx::ControlSpec;
using nablafx::DspBlockSpec;
using nablafx::LufsLeveler;
using nablafx::PluginMeta;
using nablafx::RationalA;
using nablafx::RationalAParams;
using nablafx::SpectralMaskEq;
using nablafx::SpectralMaskEqParams;
using nablafx::TruePeakCeiling;
using nablafx::DimensionD;
using nablafx::load_composite_meta;
using nablafx::load_meta;
using nablafx::param_id_for;

// All ORT sessions in this plugin process audio in fixed kBlockSize chunks,
// which matches both the auto-EQ controller's cond_block_size and the LA-2A
// processor's TVFiLM cond_block_size. Changing this requires re-exporting both
// ONNX bundles at the new block size.
constexpr int kBlockSize  = 128;
constexpr int kNumStages  = 5;
// SSL bus comp accumulator size — must be a multiple of kBlockSize. Larger
// values cut CPU proportionally (1 ORT call per kSslHop samples instead of
// 1 per kBlockSize) at the cost of (kSslHop - kBlockSize) extra latency.
//
// CRITICAL CONSTRAINT: `kSslHop <= trace_len - RF` so every ring shift
// preserves at least RF samples of past context for the model's causal
// convolutions. Asserted at activate.
//
// The wet output is delayed by `kSslHop - kBlockSize` samples relative to
// the dry signal at the blend step. We compensate via a per-channel dry
// delay ring (see ssl_comp_dry_delay) so the wet/dry mix is sample-aligned.
constexpr int kSslHop     = 1024;

enum class StageID : int {
    InputLeveler  = 0,
    AutoEQ        = 1,
    Saturator     = 2,
    OutputLeveler = 3,
    SslComp       = 4,
};

// ---------------------------------------------------------------------------
// Spectrum analyzer — Goertzel-based, runs on main thread
// ---------------------------------------------------------------------------

struct SpectrumAnalyzer {
    static constexpr int   kFFT     = 2048;   // accumulation window
    static constexpr int   kDisp    = 128;    // log-spaced display bins
    static constexpr int   kNumBins = 50;     // eq_bins resolution (log-spaced 20–20k Hz)
    static constexpr float kAlpha   = 0.65f;  // EMA coefficient (~110 ms at ~21 fps)
    static constexpr float kFlo     = 20.f;
    static constexpr float kFhi   = 20000.f;

    // Audio thread: one mono accumulator per chain position.
    struct Accum {
        std::array<float, kFFT> buf{};
        int fill{0};
    };
    std::array<Accum, kNumStages> accum{};

    // Transfer buffer: audio thread fills, main thread processes.
    std::mutex  xfer_mtx;
    bool        xfer_ready{false};
    std::array<std::array<float, kFFT>, kNumStages> xfer_frames{};
    std::array<float, 5>        xfer_eq_gains_snap{};
    std::array<float, kNumBins> xfer_eq_bins_snap{};
    bool                        xfer_has_bins_snap{false};

    // Main-thread state.
    std::array<float, kFFT>  hann{};
    std::array<float, kDisp> disp_hz{};    // Hz for each display bin
    std::array<float, kFFT>  windowed{};   // scratch for Goertzel input

    // EMA magnitude [chain_pos][disp_bin], linear.
    std::array<std::array<float, kDisp>, kNumStages> ema{};

    // Last EQ band gains (dB) and optional 50-point bin gains from the LSTM.
    // Written from the audio thread, snapped under xfer_mtx, read by main thread.
    std::array<float, 5>        xfer_eq_gains{};
    std::array<float, kNumBins> xfer_eq_bins{};
    bool                        xfer_has_bins{false};
    std::array<float, 5>        mt_eq_gains{};
    std::array<float, kNumBins> mt_eq_bins{};
    bool                        mt_has_bins{false};

    // Staging for JSON build (main thread only).
    std::array<std::array<float, kFFT>, kNumStages> mt_frames{};

    void init() {
        for (int i = 0; i < kFFT; ++i)
            hann[i] = 0.5f * (1.f - std::cos(2.f * static_cast<float>(M_PI) * i / kFFT));
        for (int i = 0; i < kDisp; ++i)
            disp_hz[i] = kFlo * std::pow(kFhi / kFlo, float(i) / (kDisp - 1));
        for (auto& row : ema) row.fill(0.f);
    }

    // Audio thread: accumulate n mono (or averaged stereo) samples for chain pos.
    void push(int pos, const float* L, const float* R, uint32_t n_ch, int n) {
        auto& a = accum[pos];
        if (a.fill >= kFFT) return;
        const int take = std::min(n, kFFT - a.fill);
        float* dst = a.buf.data() + a.fill;
        if (n_ch >= 2) {
            for (int i = 0; i < take; ++i) dst[i] = 0.5f * (L[i] + R[i]);
        } else {
            std::copy_n(L, take, dst);
        }
        a.fill += take;
    }

    // Audio thread: latch the 5 LSTM EQ band gains (dB) for the next transfer.
    void set_eq_gains(const float* gains_db_5) {
        std::copy_n(gains_db_5, 5, xfer_eq_gains.data());
    }
    // Audio thread: latch 50-point bin gains (dB) for SpectralMask view.
    void set_eq_bins(const float* gains_db) {
        std::copy_n(gains_db, kNumBins, xfer_eq_bins.data());
        xfer_has_bins = true;
    }
    void clear_eq_bins() { xfer_has_bins = false; }

    // Audio thread: when all accumulators are full, try to hand off to main thread.
    // Returns true when a transfer was attempted (whether or not the lock was acquired).
    bool advance_and_transfer() {
        if (accum[0].fill < kFFT) return false;
        if (xfer_mtx.try_lock()) {
            for (int p = 0; p < kNumStages; ++p) xfer_frames[p] = accum[p].buf;
            xfer_eq_gains_snap   = xfer_eq_gains;
            xfer_eq_bins_snap    = xfer_eq_bins;
            xfer_has_bins_snap   = xfer_has_bins;
            xfer_ready = true;
            xfer_mtx.unlock();
        }
        for (auto& a : accum) a.fill = 0;
        return true;
    }

    // Main thread: process pending transfer; returns true if new data was ready.
    bool process_if_ready(double sample_rate) {
        {
            std::lock_guard<std::mutex> lk(xfer_mtx);
            if (!xfer_ready) return false;
            mt_frames     = xfer_frames;
            mt_eq_gains   = xfer_eq_gains_snap;
            mt_eq_bins    = xfer_eq_bins_snap;
            mt_has_bins   = xfer_has_bins_snap;
            xfer_ready    = false;
        }
        const float sr = static_cast<float>(sample_rate);
        for (int pos = 0; pos < kNumStages; ++pos) {
            for (int i = 0; i < kFFT; ++i)
                windowed[i] = mt_frames[pos][i] * hann[i];
            for (int b = 0; b < kDisp; ++b) {
                const float bin_f = disp_hz[b] * kFFT / sr;
                const float mag   = goertzel(windowed.data(), kFFT, bin_f);
                ema[pos][b] = kAlpha * ema[pos][b] + (1.f - kAlpha) * mag;
            }
        }
        return true;
    }

    // Main thread: build the JS call string for the WebView.
    std::string build_js(const std::array<int, kNumStages>& order) const {
        std::string s;
        s.reserve(8192);
        s = "toneSpectrum({\"order\":[";
        for (int i = 0; i < kNumStages; ++i) { if (i) s += ','; s += std::to_string(order[i]); }
        s += "],\"db\":[";
        char buf[16];
        for (int pos = 0; pos < kNumStages; ++pos) {
            if (pos) s += ',';
            s += '[';
            for (int b = 0; b < kDisp; ++b) {
                if (b) s += ',';
                snprintf(buf, sizeof(buf), "%.1f",
                         20.f * std::log10(std::max(ema[pos][b], 1e-9f)));
                s += buf;
            }
            s += ']';
        }
        // 5 LSTM EQ band gains in dB so JS can draw the filter response curve.
        s += "],\"eq\":[";
        for (int b = 0; b < 5; ++b) {
            if (b) s += ',';
            snprintf(buf, sizeof(buf), "%.2f", mt_eq_gains[b]);
            s += buf;
        }
        // 50-point bin gains (SpectralMask only) or null (PEQ classes).
        s += "],\"eq_bins\":";
        if (mt_has_bins) {
            s += '[';
            for (int b = 0; b < kNumBins; ++b) {
                if (b) s += ',';
                snprintf(buf, sizeof(buf), "%.2f", mt_eq_bins[b]);
                s += buf;
            }
            s += ']';
        } else {
            s += "null";
        }
        s += "});";
        return s;
    }

private:
    // Goertzel algorithm for the magnitude at a single fractional bin.
    static float goertzel(const float* x, int N, float bin_f) {
        const int    k     = static_cast<int>(std::round(bin_f));
        const double w     = 2.0 * M_PI * static_cast<double>(std::clamp(k, 0, N/2));
                           // N samples in denominator:
        const double coeff = 2.0 * std::cos(w / N);
        double s1 = 0.0, s2 = 0.0;
        for (int n = 0; n < N; ++n) {
            const double s0 = x[n] + coeff * s1 - s2;
            s2 = s1; s1 = s0;
        }
        const float power = static_cast<float>(s1*s1 + s2*s2 - coeff*s1*s2);
        return std::sqrt(std::max(power, 0.f)) * (2.f / N);
    }
};

// ---------------------------------------------------------------------------
// Module-global state (loaded once at module init)
// ---------------------------------------------------------------------------

struct ModuleState {
    CompositeMeta              tone_meta;
    // One PluginMeta per auto-EQ class. Indexed by tone_meta.auto_eq.class_order;
    // the lookup map mirrors the same data keyed by class name for convenience.
    std::vector<PluginMeta>                              autoeq_metas;
    std::unordered_map<std::string, std::size_t>         autoeq_class_index;
    PluginMeta                 sat_meta;
    PluginMeta                 ssl_comp_meta;          // optional; loaded if
                                                       // tone_meta.sub_bundles
                                                       // has "ssl_comp"
    bool                       ssl_comp_loaded{false};
    std::string                bundle_dir;            // .../NeuralMastering.clap/Contents
    std::string                resources_dir;         // .../Contents/Resources
    std::string                plugin_id_str;         // "com.nablafx.<model_id>"
    clap_plugin_descriptor_t   descriptor{};
    std::vector<const char*>   feature_ptrs;
    std::array<const char*, 3> feature_storage{};
    std::unique_ptr<Ort::Env>  ort_env;
    // Pulled out of sat_meta once at load.
    RationalAParams            sat_rational;
    // Per-class DSP-block payload, parsed once at load. Every class declares
    // ``spectral_mask_eq`` as its dsp_blocks[0]; held here so the audio
    // thread can read num_control_params without re-parsing meta.
    std::vector<DspBlockSpec>  autoeq_dsp_per_class;
    int                        autoeq_default_idx{0};
};

static ModuleState* g_state = nullptr;

static std::string find_bundle_contents_() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&find_bundle_contents_), &info) == 0
        || !info.dli_fname) {
        return {};
    }
    fs::path dylib = info.dli_fname;
    // .clap/Contents/MacOS/<dylib> → .clap/Contents
    return dylib.parent_path().parent_path().string();
}

static void populate_descriptor_(ModuleState& st) {
    // Build a short plugin ID from effect_name (model_id can be 300+ chars,
    // which overflows fixed-size ID buffers in some CLAP hosts).
    std::string short_name = st.tone_meta.effect_name;
    std::transform(short_name.begin(), short_name.end(), short_name.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    for (auto& c : short_name)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-') c = '_';
    st.plugin_id_str = "com.nablafx." + short_name;

    st.feature_storage[0] = CLAP_PLUGIN_FEATURE_AUDIO_EFFECT;
    st.feature_storage[1] = CLAP_PLUGIN_FEATURE_MASTERING;
    st.feature_storage[2] = nullptr;
    st.feature_ptrs.assign(st.feature_storage.begin(), st.feature_storage.end());

    st.descriptor.clap_version = CLAP_VERSION_INIT;
    st.descriptor.id           = st.plugin_id_str.c_str();
    st.descriptor.name         = st.tone_meta.effect_name.c_str();
    st.descriptor.vendor       = "nablafx";
    st.descriptor.url          = "https://github.com/mcomunita/nablafx";
    st.descriptor.manual_url   = "";
    st.descriptor.support_url  = "";
    st.descriptor.version      = "1.0.0";
    st.descriptor.description  = "NeuralMastering — adaptive mastering chain (auto-EQ + saturator + LA-2A + leveler + ceiling)";
    st.descriptor.features     = st.feature_ptrs.data();
}

// ---------------------------------------------------------------------------
// Per-channel chain state. One of these per audio channel.
// ---------------------------------------------------------------------------

struct StateBuf {
    std::vector<int64_t> shape;
    std::vector<float>   data;
};

class OrtMiniSession {
    // Thin wrapper around an Ort::Session for fixed-shape audio + state I/O.
    // Owns the input/output state buffers; you set audio + controls (if any),
    // then run() reads/writes state, and call swap() to make this run's
    // outputs the next run's inputs.
public:
    OrtMiniSession(Ort::Env& env, const std::string& model_path, const PluginMeta& meta)
        : env_(env), cpu_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
          meta_(meta) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetExecutionMode(ORT_SEQUENTIAL);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
        for (const auto& nm : meta.input_names)  in_names_owned_.push_back(nm);
        for (const auto& nm : meta.output_names) out_names_owned_.push_back(nm);
        for (auto& s : in_names_owned_)  in_names_.push_back(s.c_str());
        for (auto& s : out_names_owned_) out_names_.push_back(s.c_str());

        // Allocate state buffers and pre-build owned "_in"/"_out" name strings
        // so run() never constructs temporaries whose .c_str() would dangle.
        for (const auto& s : meta.state_tensors) {
            int64_t n = 1;
            for (auto d : s.shape) n *= d;
            in_states_[s.name].shape  = s.shape;
            in_states_[s.name].data.assign(n, 0.0f);
            out_states_[s.name].shape = s.shape;
            out_states_[s.name].data.assign(n, 0.0f);
            state_in_names_owned_.push_back(s.name + "_in");
            state_out_names_owned_.push_back(s.name + "_out");
        }
    }

    StateBuf& in_state(const std::string& name) { return in_states_.at(name); }

    void reset_state() {
        for (auto& [_, b] : in_states_)  std::fill(b.data.begin(), b.data.end(), 0.0f);
        for (auto& [_, b] : out_states_) std::fill(b.data.begin(), b.data.end(), 0.0f);
    }

    // Run with caller-owned audio (and optional controls) buffers. Outputs
    // are written into `audio_out` (and the internal state-out buffers).
    void run(const float* audio_in, int audio_in_len,
             float* audio_out, int audio_out_len,
             const float* controls /*nullable*/, int n_controls,
             const std::string& audio_out_name = "audio_out") {
        std::vector<Ort::Value>      inputs;
        std::vector<const char*>     in_names;
        std::vector<const char*>     out_names;
        in_names.reserve(in_names_.size());
        out_names.reserve(out_names_.size());

        std::array<int64_t, 3> aud_shape{1, 1, audio_in_len};
        inputs.push_back(Ort::Value::CreateTensor<float>(
            cpu_, const_cast<float*>(audio_in), audio_in_len,
            aud_shape.data(), aud_shape.size()));
        in_names.push_back("audio_in");

        std::array<int64_t, 2> ctl_shape{1, n_controls};
        if (n_controls > 0) {
            inputs.push_back(Ort::Value::CreateTensor<float>(
                cpu_, const_cast<float*>(controls), n_controls,
                ctl_shape.data(), ctl_shape.size()));
            in_names.push_back("controls");
        }

        // Add state inputs in the order the meta declared them.
        for (std::size_t si = 0; si < meta_.state_tensors.size(); ++si) {
            auto& buf = in_states_[meta_.state_tensors[si].name];
            inputs.push_back(Ort::Value::CreateTensor<float>(
                cpu_, buf.data.data(), static_cast<int64_t>(buf.data.size()),
                buf.shape.data(), buf.shape.size()));
            in_names.push_back(state_in_names_owned_[si].c_str());
        }

        // Output: audio first, then states in declared order.
        for (const auto& nm : out_names_) out_names.push_back(nm);

        auto outs = session_->Run(Ort::RunOptions{nullptr},
                                  in_names.data(), inputs.data(), inputs.size(),
                                  out_names.data(), out_names.size());

        // Copy audio out (must match the requested length).
        // The first output is named audio_out_name; locate it by index.
        std::size_t audio_out_idx = 0;
        for (std::size_t i = 0; i < out_names_owned_.size(); ++i) {
            if (out_names_owned_[i] == audio_out_name) {
                audio_out_idx = i; break;
            }
        }
        const float* aud_out = outs[audio_out_idx].GetTensorData<float>();
        // The actual ORT output length may be SHORTER than audio_in_len —
        // streaming-mode TCN export trims (rf-1) samples (no internal
        // pre-padding; output_len = input_len - (rf-1)). Clamp to the real
        // tensor element count to avoid reading past the end (which would
        // emit garbage memory and produce a hop-rate flutter on the wet).
        const auto out_info = outs[audio_out_idx].GetTensorTypeAndShapeInfo();
        const int64_t actual_len =
            static_cast<int64_t>(out_info.GetElementCount());
        const int64_t copy_len =
            std::min<int64_t>(audio_out_len, actual_len);
        std::copy_n(aud_out, copy_len, audio_out);
        // If the caller asked for more than the model produced, zero-fill the
        // tail so callers that don't size-check at least see silence rather
        // than uninitialised memory.
        if (copy_len < audio_out_len) {
            std::fill(audio_out + copy_len,
                      audio_out + audio_out_len, 0.0f);
        }

        // Read back states by output-name (always "<state>_out").
        for (std::size_t si = 0; si < meta_.state_tensors.size(); ++si) {
            const std::string& out_name = state_out_names_owned_[si];
            std::size_t idx = 0;
            for (std::size_t i = 0; i < out_names_owned_.size(); ++i) {
                if (out_names_owned_[i] == out_name) { idx = i; break; }
            }
            const float* p = outs[idx].GetTensorData<float>();
            const std::string& sname = meta_.state_tensors[si].name;
            std::copy_n(p, out_states_[sname].data.size(), out_states_[sname].data.begin());
        }
    }

    void swap_state() {
        for (const auto& s : meta_.state_tensors) {
            std::swap(in_states_[s.name].data, out_states_[s.name].data);
        }
    }

    // Run-arbitrary variant for the auto-EQ controller, where the audio
    // output channel name is "params_proc_0" and represents [1, 15, T]
    // sigmoid params instead of audio. We expose only the first sample
    // (all samples in a block are identical post-repeat_interleave).
    void run_controller(const float* audio_in, int audio_in_len,
                        float* params_out_first, int params_out_count) {
        std::vector<Ort::Value>  inputs;
        std::vector<const char*> in_names;
        std::vector<const char*> out_names;

        std::array<int64_t, 3> aud_shape{1, 1, audio_in_len};
        inputs.push_back(Ort::Value::CreateTensor<float>(
            cpu_, const_cast<float*>(audio_in), audio_in_len,
            aud_shape.data(), aud_shape.size()));
        in_names.push_back("audio_in");

        for (std::size_t si = 0; si < meta_.state_tensors.size(); ++si) {
            auto& buf = in_states_[meta_.state_tensors[si].name];
            inputs.push_back(Ort::Value::CreateTensor<float>(
                cpu_, buf.data.data(), static_cast<int64_t>(buf.data.size()),
                buf.shape.data(), buf.shape.size()));
            in_names.push_back(state_in_names_owned_[si].c_str());
        }
        for (const auto& nm : out_names_) out_names.push_back(nm);

        auto outs = session_->Run(Ort::RunOptions{nullptr},
                                  in_names.data(), inputs.data(), inputs.size(),
                                  out_names.data(), out_names.size());

        // First output is the params tensor [1, 15, T]. We want the first
        // sample (all are identical within a block per the controller's
        // repeat_interleave structure).
        const float* p = outs[0].GetTensorData<float>();
        // Stride: T = audio_in_len, channel-major contiguous so element
        // [channel c, sample 0] is at offset c * T.
        for (int c = 0; c < params_out_count; ++c) {
            params_out_first[c] = p[c * audio_in_len + 0];
        }

        for (std::size_t si = 0; si < meta_.state_tensors.size(); ++si) {
            const std::string& out_name = state_out_names_owned_[si];
            std::size_t idx = 0;
            for (std::size_t i = 0; i < out_names_owned_.size(); ++i) {
                if (out_names_owned_[i] == out_name) { idx = i; break; }
            }
            const float* sp = outs[idx].GetTensorData<float>();
            const std::string& sname = meta_.state_tensors[si].name;
            std::copy_n(sp, out_states_[sname].data.size(), out_states_[sname].data.begin());
        }
    }

private:
    Ort::Env&                     env_;
    Ort::MemoryInfo               cpu_;
    PluginMeta                    meta_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string>      in_names_owned_;
    std::vector<std::string>      out_names_owned_;
    std::vector<std::string>      state_in_names_owned_;   // "<base>_in" per state tensor
    std::vector<std::string>      state_out_names_owned_;  // "<base>_out" per state tensor
    std::vector<const char*>      in_names_;
    std::vector<const char*>      out_names_;
    std::unordered_map<std::string, StateBuf> in_states_;
    std::unordered_map<std::string, StateBuf> out_states_;
};

struct ChannelChain {
    // Per-channel stage instances. TruePeakCeiling runs per-channel; the
    // LufsLeveler is shared on Plugin so it can apply linked stereo gain.
    // One ORT session per auto-EQ class (bass/drums/vocals/other/full_mix);
    // CLS picks which one is active. Inactive sessions hold zeroed state
    // so a class switch starts the LSTM from a neutral init and avoids
    // bleeding stale activations from a different class's signal.
    std::vector<std::unique_ptr<OrtMiniSession>> autoeq_ort_per_class;
    // Per-class SpectralMaskEq instance. Every class shares the same kind
    // now, but each class still gets its own instance so the per-class
    // mask-smoother state doesn't bleed across class switches.
    std::vector<std::unique_ptr<SpectralMaskEq>>    autoeq_spec_per_class;
    // Peak-hold envelope follower for auto-EQ controller input normalization.
    // Training normalized peak per ~10 s segment; using a per-128-block peak at
    // runtime collapsed the LSTM's input distribution. Attack-instant /
    // decay-slow tracking gives a stable scale across blocks.
    float                                  autoeq_peak_env{0.f};
    RationalA                              saturator;
    // 1st-order HPF for bass-preserved saturation (bilinear transform).
    float sat_hpf_fc{-1.f};               // cached cutoff; -1 = stale
    float sat_hpf_b0{0.f}, sat_hpf_b1{0.f}, sat_hpf_a1{0.f};
    float sat_hpf_x1{0.f}, sat_hpf_y1{0.f};
    // SSL-style bus comp (separate stage from LA-2A). Stateless long-RF
    // causal TCN: needs a trace_len-sized input ring per channel because the
    // ORT call expects all RF samples of context per invocation. Allocated
    // only when the ssl_comp sub-bundle is shipped; otherwise null and the
    // SslComp stage is a passthrough.
    std::unique_ptr<OrtMiniSession>        ssl_comp_ort;
    std::vector<float>                     ssl_comp_in_ring;     // [trace_len]
    std::vector<float>                     ssl_comp_out_buf;     // [trace_len]
    // Hop accumulation: re-running the full TCN forward pass every
    // kBlockSize=128 samples blows the audio thread's deadline at long RF.
    // Accumulate kSslHop input samples, then run ORT once and play out the
    // resulting kSslHop samples over (kSslHop / kBlockSize) host calls. The
    // model still sees trace_len of context per call; we just call it less
    // often. Adds (kSslHop - kBlockSize) samples of latency.
    std::vector<float>                     ssl_comp_in_accum;    // [kSslHop]
    int                                    ssl_comp_in_fill{0};
    std::vector<float>                     ssl_comp_out_queue;   // [kSslHop]
    int                                    ssl_comp_out_avail{0};
    int                                    ssl_comp_out_read{0};
    // Dry delay ring: holds (kSslHop - kBlockSize) samples of dry audio so
    // the wet/dry blend is sample-aligned. Without this, blending a delayed
    // wet with the current dry produces hop-rate comb-filter flutter.
    std::vector<float>                     ssl_comp_dry_delay;
    int                                    ssl_comp_dry_write{0};
    TruePeakCeiling                        ceiling;

    // 128-sample accumulator: kBlockSize input samples in, then a chain pass,
    // then kBlockSize output samples ready. Output ring fills before any reads
    // so the first kBlockSize host samples produce silence (latency reported
    // to the host so DAWs compensate).
    std::array<float, kBlockSize> in_buf{};
    int                           in_fill = 0;

    std::array<float, kBlockSize> out_buf{};
    int                           out_avail = 0;
    int                           out_read  = 0;
};

// ---------------------------------------------------------------------------
// Per-instance state
// ---------------------------------------------------------------------------

struct Plugin {
    clap_plugin_t      plugin{};
    const clap_host_t* host{nullptr};

    const CompositeMeta* meta{nullptr};
    int                  channels{2};
    double               sample_rate{};
    bool                 activated{false};

    std::vector<float>   control_values;

    // Decay coefficient for the per-channel auto-EQ peak envelope follower.
    // Computed at activate from sample_rate so the time constant stays at
    // ~500 ms regardless of host sr. Attack is instantaneous.
    float                autoeq_env_decay{0.f};

    // Processor ordering — driven by GUI drag-and-drop.
    std::array<int, kNumStages> processor_order{0, 1, 2, 3, 4};

    // Active auto-EQ class index (into ModuleState::autoeq_metas /
    // tone_meta.auto_eq.class_order). Updated from the audio thread when the
    // CLS control changes, so the AutoEQ stage routes through
    // chains[ch].autoeq_ort_per_class[active_autoeq_cls].
    int active_autoeq_cls{0};

    // GUI → audio-thread param queue (try_lock on audio thread, never blocks).
    std::mutex                       param_mutex;
    std::vector<std::pair<int,float>> param_queue;

    // GUI → audio-thread order change.
    std::mutex               order_mutex;
    bool                     order_pending{false};
    std::array<int,kNumStages> pending_order{0,1,2,3,4};

    // CLAP GUI handle (main thread only).
    ToneGUIState* gui_state{nullptr};

    // Dynamic latency tracking.
    // current_latency is written by the audio thread and read by latency_get
    // (main thread). latency_needs_notify is set by the audio thread and
    // cleared by on_main_thread after calling host_latency_ext->changed().
    std::atomic<uint32_t>           current_latency{0};
    std::atomic<bool>               latency_needs_notify{false};
    const clap_host_latency_t*      host_latency_ext{nullptr};

    // Spectrum analyzer (audio thread accumulates, main thread computes + renders).
    SpectrumAnalyzer spectrum;

    // Shared levelers — input and output, both apply linked L/R gain.
    LufsLeveler              leveler;
    LufsLeveler              out_leveler;
    std::vector<ChannelChain> chains;
};

// ---------------------------------------------------------------------------
// CLAP extension: audio ports — 1 stereo input, 1 stereo output
// ---------------------------------------------------------------------------

static uint32_t audio_ports_count(const clap_plugin_t*, bool /*is_input*/) { return 1; }

static bool audio_ports_get(const clap_plugin_t*, uint32_t index, bool is_input,
                            clap_audio_port_info_t* info) {
    if (index != 0) return false;
    info->id            = is_input ? 0 : 1;
    std::snprintf(info->name, sizeof(info->name), "%s", is_input ? "in" : "out");
    info->channel_count = 2;
    info->flags         = CLAP_AUDIO_PORT_IS_MAIN;
    info->port_type     = CLAP_PORT_STEREO;
    info->in_place_pair = CLAP_INVALID_ID;
    return true;
}
static const clap_plugin_audio_ports_t s_ext_audio_ports = {audio_ports_count, audio_ports_get};

// ---------------------------------------------------------------------------
// CLAP extension: params (AMT, TRM)
// ---------------------------------------------------------------------------

static uint32_t params_count(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    return static_cast<uint32_t>(plug->meta->controls.size());
}

static bool params_get_info(const clap_plugin_t* p, uint32_t index, clap_param_info_t* info) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    if (index >= plug->meta->controls.size()) return false;
    const auto& c = plug->meta->controls[index];
    info->id        = param_id_for(plug->meta->effect_name, c.id);
    info->flags     = CLAP_PARAM_IS_AUTOMATABLE;
    info->cookie    = nullptr;
    info->min_value = c.min;
    info->max_value = c.max;
    info->default_value = c.def;
    std::snprintf(info->name,   sizeof(info->name),   "%s", c.name.c_str());
    std::snprintf(info->module, sizeof(info->module), "%s", "");
    return true;
}

static bool params_get_value(const clap_plugin_t* p, clap_id id, double* value) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    for (size_t i = 0; i < plug->meta->controls.size(); ++i) {
        if (param_id_for(plug->meta->effect_name, plug->meta->controls[i].id) == id) {
            *value = plug->control_values[i];
            return true;
        }
    }
    return false;
}

static bool params_value_to_text(const clap_plugin_t* p, clap_id id, double value, char* out, uint32_t out_size) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    // CLS displays the class name instead of the integer index.
    for (size_t i = 0; i < plug->meta->controls.size(); ++i) {
        if (param_id_for(plug->meta->effect_name, plug->meta->controls[i].id) == id
            && plug->meta->controls[i].id == "CLS") {
            const auto& classes = g_state->tone_meta.auto_eq.class_order;
            int idx = std::clamp(static_cast<int>(std::lround(value)),
                                 0, static_cast<int>(classes.size()) - 1);
            std::snprintf(out, out_size, "%s", classes[idx].c_str());
            return true;
        }
    }
    std::snprintf(out, out_size, "%.3f", value);
    return true;
}

static bool params_text_to_value(const clap_plugin_t* p, clap_id id, const char* text, double* out) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    // CLS accepts a class name and converts it to the canonical index.
    for (size_t i = 0; i < plug->meta->controls.size(); ++i) {
        if (param_id_for(plug->meta->effect_name, plug->meta->controls[i].id) == id
            && plug->meta->controls[i].id == "CLS") {
            const auto& classes = g_state->tone_meta.auto_eq.class_order;
            for (size_t k = 0; k < classes.size(); ++k) {
                if (classes[k] == text) {
                    *out = static_cast<double>(k);
                    return true;
                }
            }
            // Fall through to numeric parse if the text isn't a class name.
        }
    }
    char* end = nullptr;
    double v = std::strtod(text, &end);
    if (end == text) return false;
    *out = v;
    return true;
}

static void params_flush(const clap_plugin_t*, const clap_input_events_t*, const clap_output_events_t*) {}

static const clap_plugin_params_t s_ext_params = {
    params_count, params_get_info, params_get_value, params_value_to_text,
    params_text_to_value, params_flush,
};

// ---------------------------------------------------------------------------
// CLAP extension: latency
// ---------------------------------------------------------------------------

// Computes the true end-to-end latency for the current parameter state.
// Called from the audio thread (after param drain) and from activate.
// Sources:
//   kBlockSize          — input accumulator always present
//   SpectralMaskEq      — n_fft - hop, only when EQ wet > 0 and class is spectral
//   SSL bus comp        — kSslHop - kBlockSize, only when SSC > 0 and loaded
//   TruePeakCeiling     — lookahead, always present once activated
static uint32_t compute_latency_(const Plugin& plug) {
    if (plug.chains.empty() || !g_state) return 0;

    uint32_t lat = kBlockSize;
    lat += static_cast<uint32_t>(plug.chains[0].ceiling.latency_samples());

    float eq_wet = 0.f, ssc_wet = 0.f;
    int   cls_idx = 0;
    for (size_t i = 0; i < plug.meta->controls.size(); ++i) {
        const auto& c = plug.meta->controls[i];
        const float v = plug.control_values[i];
        if      (c.id == "EQ")  eq_wet  = v;
        else if (c.id == "SSC") ssc_wet = v;
        else if (c.id == "CLS") cls_idx = static_cast<int>(std::lround(v));
    }

    if (eq_wet > 0.f && !g_state->autoeq_dsp_per_class.empty()) {
        const int n_cls = static_cast<int>(g_state->autoeq_dsp_per_class.size());
        cls_idx = std::clamp(cls_idx, 0, n_cls - 1);
        if (g_state->autoeq_dsp_per_class[cls_idx].kind == "spectral_mask_eq") {
            const auto& sp = std::get<SpectralMaskEqParams>(
                g_state->autoeq_dsp_per_class[cls_idx].params);
            lat += static_cast<uint32_t>(sp.n_fft);
        }
    }

    if (g_state->ssl_comp_loaded && ssc_wet > 0.f)
        lat += static_cast<uint32_t>(kSslHop - kBlockSize);

    return lat;
}

static uint32_t latency_get(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    return plug->current_latency.load(std::memory_order_relaxed);
}

static const clap_plugin_latency_t s_ext_latency = {latency_get};

// ---------------------------------------------------------------------------
// CLAP extension: state (save / load)
// ---------------------------------------------------------------------------

// Forward declaration — defined with the GUI extension below.
static void gui_send_full_state_(Plugin* plug);

static bool state_save(const clap_plugin_t* p, const clap_ostream_t* stream) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    nlohmann::json j;
    j["version"] = 2;
    for (size_t i = 0; i < plug->meta->controls.size(); ++i)
        j["controls"][plug->meta->controls[i].id] = plug->control_values[i];
    auto& jo = j["processor_order"];
    for (int i = 0; i < kNumStages; ++i) jo.push_back(plug->processor_order[i]);
    std::string txt = j.dump();
    int64_t written = stream->write(stream, txt.data(), txt.size());
    return written == static_cast<int64_t>(txt.size());
}

static bool state_load(const clap_plugin_t* p, const clap_istream_t* stream) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    std::string txt;
    char buf[4096];
    int64_t n;
    while ((n = stream->read(stream, buf, sizeof(buf))) > 0)
        txt.append(buf, static_cast<size_t>(n));
    if (n < 0) return false;
    try {
        auto j = nlohmann::json::parse(txt);
        auto& jc = j.at("controls");
        for (size_t i = 0; i < plug->meta->controls.size(); ++i) {
            const auto& id = plug->meta->controls[i].id;
            if (jc.contains(id))
                plug->control_values[i] = jc.at(id).get<float>();
        }
        if (j.contains("processor_order")) {
            auto& jo = j.at("processor_order");
            if (jo.is_array() && static_cast<int>(jo.size()) == kNumStages) {
                for (int i = 0; i < kNumStages; ++i)
                    plug->processor_order[i] = jo[i].get<int>();
            }
        }
    } catch (...) { return false; }

    // Tell the host that all parameter values changed.
    const auto* host_params = static_cast<const clap_host_params_t*>(
        plug->host->get_extension(plug->host, CLAP_EXT_PARAMS));
    if (host_params && host_params->rescan)
        host_params->rescan(plug->host, CLAP_PARAM_RESCAN_VALUES);
    // If the GUI is already open, push the restored state to it immediately.
    if (plug->gui_state)
        gui_send_full_state_(plug);
    return true;
}

static const clap_plugin_state_t s_ext_state = {state_save, state_load};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

static bool plugin_init(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    plug->control_values.resize(plug->meta->controls.size());
    for (size_t i = 0; i < plug->meta->controls.size(); ++i)
        plug->control_values[i] = plug->meta->controls[i].def;
    plug->host_latency_ext = static_cast<const clap_host_latency_t*>(
        plug->host->get_extension(plug->host, CLAP_EXT_LATENCY));
    return true;
}
static void plugin_destroy(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    if (plug->gui_state) { tone_gui_destroy(plug->gui_state); plug->gui_state = nullptr; }
    delete plug;
}

static bool plugin_activate(const clap_plugin_t* p, double sample_rate,
                            uint32_t /*min_frames*/, uint32_t /*max_frames*/) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    plug->sample_rate = sample_rate;
    // ~500 ms peak-envelope decay, evaluated once per kBlockSize-sample block.
    {
        constexpr float kEnvTauSeconds = 0.5f;
        const float blocks_per_tau = (static_cast<float>(sample_rate) * kEnvTauSeconds)
                                     / static_cast<float>(kBlockSize);
        plug->autoeq_env_decay = std::exp(-1.0f / std::max(blocks_per_tau, 1.0f));
    }
    plug->leveler = LufsLeveler(LufsLeveler::Config{
        /*target_lufs=*/g_state->tone_meta.leveler.target_lufs,
    });
    plug->leveler.reset(sample_rate, g_state->tone_meta.leveler.target_lufs);
    plug->out_leveler = LufsLeveler(LufsLeveler::Config{
        /*target_lufs=*/g_state->tone_meta.leveler.target_lufs,
    });
    plug->out_leveler.reset(sample_rate, g_state->tone_meta.leveler.target_lufs);

    // Seed active class from CLS control; clamp to a valid class index.
    {
        const auto& classes = g_state->tone_meta.auto_eq.class_order;
        int cls_idx = g_state->autoeq_default_idx;
        for (size_t i = 0; i < plug->meta->controls.size(); ++i) {
            if (plug->meta->controls[i].id == "CLS") {
                cls_idx = static_cast<int>(std::lround(plug->control_values[i]));
                break;
            }
        }
        cls_idx = std::clamp(cls_idx, 0, static_cast<int>(classes.size()) - 1);
        plug->active_autoeq_cls = cls_idx;
    }

    plug->chains.clear();
    plug->chains.resize(plug->channels);
    for (auto& ch : plug->chains) {
        const auto& classes = g_state->tone_meta.auto_eq.class_order;
        ch.autoeq_ort_per_class.clear();
        ch.autoeq_ort_per_class.reserve(classes.size());
        ch.autoeq_spec_per_class.clear();
        ch.autoeq_spec_per_class.resize(classes.size());
        for (size_t i = 0; i < classes.size(); ++i) {
            const std::string& cls = classes[i];
            const std::string& dir = g_state->tone_meta.auto_eq.classes.at(cls);
            ch.autoeq_ort_per_class.push_back(std::make_unique<OrtMiniSession>(
                *g_state->ort_env,
                g_state->resources_dir + "/" + dir + "/model.onnx",
                g_state->autoeq_metas[i]));

            // Every auto-EQ class declares spectral_mask_eq as its
            // dsp_blocks[0]; meta.cpp throws if anything else slips through.
            const auto& dsp = g_state->autoeq_dsp_per_class[i];
            ch.autoeq_spec_per_class[i] = std::make_unique<SpectralMaskEq>();
            ch.autoeq_spec_per_class[i]->reset(
                std::get<SpectralMaskEqParams>(dsp.params));
        }
        ch.saturator.reset(g_state->sat_rational.numerator,
                           g_state->sat_rational.denominator);

        if (g_state->ssl_comp_loaded && g_state->ssl_comp_meta.trace_len > 0) {
            const int N  = g_state->ssl_comp_meta.trace_len;
            const int rf = g_state->ssl_comp_meta.receptive_field;
            // Causal-context safety: each ring shift must preserve at least
            // RF samples of past audio so the model's first hop-output sample
            // sees its full receptive field. Otherwise hop-rate discontinuity.
            if (kSslHop > N - rf) {
                throw std::runtime_error(
                    "ssl_comp: kSslHop=" + std::to_string(kSslHop) +
                    " exceeds trace_len-RF=" + std::to_string(N - rf) +
                    "; would cause hop-rate discontinuity. Either lower "
                    "kSslHop or re-export the bundle with a larger trace_len.");
            }
            ch.ssl_comp_ort = std::make_unique<OrtMiniSession>(
                *g_state->ort_env,
                g_state->resources_dir + "/" +
                    g_state->tone_meta.sub_bundles.at("ssl_comp") + "/model.onnx",
                g_state->ssl_comp_meta);
            ch.ssl_comp_in_ring.assign(N, 0.0f);
            ch.ssl_comp_out_buf.assign(N, 0.0f);
            ch.ssl_comp_in_accum.assign(kSslHop, 0.0f);
            ch.ssl_comp_in_fill = 0;
            ch.ssl_comp_out_queue.assign(kSslHop, 0.0f);
            ch.ssl_comp_out_avail = 0;
            ch.ssl_comp_out_read = 0;
            // Dry delay buffer matches the wet output queue's offset so the
            // blend step sees time-aligned dry. Length = kSslHop - kBlockSize
            // (zero when no accumulation, fits the natural per-block path).
            const int dry_delay_len = kSslHop - kBlockSize;
            ch.ssl_comp_dry_delay.assign(std::max(dry_delay_len, 1), 0.0f);
            ch.ssl_comp_dry_write = 0;
        }

        TruePeakCeiling::Config tcfg{
            /*ceiling_dbtp=*/g_state->tone_meta.ceiling.ceiling_dbtp,
            /*lookahead_ms=*/g_state->tone_meta.ceiling.lookahead_ms,
            /*attack_ms=*/g_state->tone_meta.ceiling.attack_ms,
            /*release_ms=*/g_state->tone_meta.ceiling.release_ms,
        };
        ch.ceiling = TruePeakCeiling(tcfg);
        ch.ceiling.reset(sample_rate);

        ch.in_fill   = 0;
        ch.out_avail = 0;
        ch.out_read  = 0;
        ch.autoeq_peak_env = 0.f;
    }

    plug->spectrum.init();

    plug->current_latency.store(compute_latency_(*plug), std::memory_order_relaxed);
    plug->activated = true;
    return true;
}

static void plugin_deactivate(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    plug->chains.clear();
    plug->activated = false;
}

static bool plugin_start_processing(const clap_plugin_t*) { return true; }
static void plugin_stop_processing(const clap_plugin_t*) {}

static void plugin_reset(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    plug->leveler.reset(plug->sample_rate, g_state->tone_meta.leveler.target_lufs);
    plug->out_leveler.reset(plug->sample_rate, g_state->tone_meta.leveler.target_lufs);
    for (auto& ch : plug->chains) {
        for (auto& s : ch.autoeq_ort_per_class) {
            if (s) s->reset_state();
        }
        ch.ceiling.reset(plug->sample_rate);
        ch.in_fill   = 0;
        ch.out_avail = 0;
        ch.out_read  = 0;
        ch.autoeq_peak_env = 0.f;
        std::fill(ch.in_buf.begin(),  ch.in_buf.end(),  0.0f);
        std::fill(ch.out_buf.begin(), ch.out_buf.end(), 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Block flush — chain one 128-sample block through every stage.
// ---------------------------------------------------------------------------

namespace {

struct AmountSnapshot {
    float lvl_wet, lvl_target_lufs;
    float out_lvl_wet, out_lvl_target_lufs;
    float autoeq_wet_mix;
    int   autoeq_cls_idx;
    float sat_pre_db, sat_post_db, sat_wet_mix, sat_hpf_hz, sat_thresh_lin, sat_bias;
    float trim_lin;
    float eq_range;
    float eq_boost_scale;
    float eq_speed_ms;
    float ssl_comp_wet;
};

AmountSnapshot resolve_amount_(const Plugin& plug) {
    float lvl=1.f,lvt=-14.f,sdr=0.f,svo=0.f,smx=0.5f,shf=20.f,sth=0.f,sbs=0.f;
    float eq=0.5f,trm_db=0.f;
    float olv=1.f,olt=-14.f;
    float eqr=1.f,eqs=100.f,eqb=1.f;
    float ssc=0.f;
    int   cls_idx=plug.active_autoeq_cls;
    for (size_t i=0;i<plug.meta->controls.size();++i) {
        const auto& c=plug.meta->controls[i];
        float v=std::clamp(plug.control_values[i],c.min,c.max);
        if(c.id=="LVL") lvl=v; else if(c.id=="LVT") lvt=v;
        else if(c.id=="SDR") sdr=v; else if(c.id=="SVO") svo=v;
        else if(c.id=="SMX") smx=v; else if(c.id=="SHF") shf=v;
        else if(c.id=="STH") sth=v; else if(c.id=="SBS") sbs=v;
        else if(c.id=="EQ")  eq=v;
        else if(c.id=="CLS") cls_idx=static_cast<int>(std::lround(v));
        else if(c.id=="EQR") eqr=v; else if(c.id=="EQS") eqs=v;
        else if(c.id=="EQB") eqb=v;
        else if(c.id=="OLV") olv=v;
        else if(c.id=="OLT") olt=v;
        else if(c.id=="TRM") trm_db=v;
        else if(c.id=="SSC") ssc=v;
    }
    AmountSnapshot s{};
    s.lvl_wet=lvl; s.lvl_target_lufs=lvt;
    s.out_lvl_wet=olv; s.out_lvl_target_lufs=olt;
    s.sat_pre_db=sdr; s.sat_post_db=svo; s.sat_wet_mix=smx; s.sat_hpf_hz=shf;
    s.sat_thresh_lin=std::pow(10.f, sth/20.f); s.sat_bias=sbs;
    s.autoeq_wet_mix=eq*g_state->tone_meta.amt_autoeq.wet_mix_max;
    const int n_cls = static_cast<int>(g_state->tone_meta.auto_eq.class_order.size());
    s.autoeq_cls_idx = std::clamp(cls_idx, 0, n_cls > 0 ? n_cls - 1 : 0);
    s.trim_lin=std::pow(10.f,trm_db/20.f);
    s.eq_range=eqr;
    s.eq_boost_scale=eqb;
    s.eq_speed_ms=eqs;
    s.ssl_comp_wet = ssc * g_state->tone_meta.amt_ssl_comp.wet_mix_max;
    return s;
}

// Helpers for wet/dry blend into a buffer in-place.
static void blend_(float* buf, const float* dry, const float* wet, float w, int n) {
    for (int i=0;i<n;++i) buf[i]=(1.f-w)*dry[i]+w*wet[i];
}
static void blend_inplace_(float* buf, const float* wet, float w, int n) {
    if (w>=1.f) { std::copy_n(wet,n,buf); return; }
    for (int i=0;i<n;++i) buf[i]+=(wet[i]-buf[i])*w;
}

// Unified stage-dispatch block processor.
// Applies all user-orderable stages to work_l/work_r in plug.processor_order,
// then writes through TruePeakCeiling into each chain's out_buf.
void flush_chain_block_(Plugin& plug,
                        float* work_l, float* work_r,
                        uint32_t n_ch,
                        const AmountSnapshot& amt) {

    std::array<float,kBlockSize> dry{}, wet_a{}, wet_b{};

    for (int pos = 0; pos < kNumStages; ++pos) {
        const int stage_idx = plug.processor_order[pos];
        switch (static_cast<StageID>(stage_idx)) {

        case StageID::InputLeveler: {
            if (amt.lvl_wet <= 0.f) break;
            plug.leveler.set_target(static_cast<double>(amt.lvl_target_lufs));
            std::array<float,kBlockSize> lev_l{}, lev_r{};
            if (n_ch >= 2) {
                plug.leveler.process_linked(work_l, work_r,
                                            lev_l.data(), lev_r.data(), kBlockSize);
                blend_inplace_(work_l, lev_l.data(), amt.lvl_wet, kBlockSize);
                blend_inplace_(work_r, lev_r.data(), amt.lvl_wet, kBlockSize);
            } else {
                plug.leveler.process(work_l, lev_l.data(), kBlockSize);
                blend_inplace_(work_l, lev_l.data(), amt.lvl_wet, kBlockSize);
            }
            break;
        }

        case StageID::AutoEQ: {
            // If CLS changed since the previous block, swap the active class
            // and zero out the new session's LSTM state so we don't carry over
            // hidden activations conditioned on a different signal class.
            if (amt.autoeq_cls_idx != plug.active_autoeq_cls) {
                plug.active_autoeq_cls = amt.autoeq_cls_idx;
                for (auto& chan : plug.chains) {
                    auto& s = chan.autoeq_ort_per_class[plug.active_autoeq_cls];
                    if (s) s->reset_state();
                    chan.autoeq_peak_env = 0.f;
                }
            }
            const int cls = plug.active_autoeq_cls;
            const auto& cls_dsp = g_state->autoeq_dsp_per_class[cls];
            const int   n_params =
                std::get<SpectralMaskEqParams>(cls_dsp.params).num_control_params;
            std::array<float, 64> eq_params_storage{};
            float* eq_params = eq_params_storage.data();
            float* ch_buf[2] = {work_l, work_r};
            for (uint32_t ch=0; ch<n_ch; ++ch) {
                float* blk = ch_buf[ch];
                auto& sess = plug.chains[ch].autoeq_ort_per_class[cls];
                // Peak-hold envelope normalisation to match training distribution.
                std::array<float, kBlockSize> ctrl_buf;
                float blk_peak = 0.f;
                for (int i = 0; i < kBlockSize; ++i)
                    blk_peak = std::max(blk_peak, std::abs(blk[i]));
                auto& env = plug.chains[ch].autoeq_peak_env;
                if (blk_peak > env) env = blk_peak;
                else env = plug.autoeq_env_decay * env
                          + (1.f - plug.autoeq_env_decay) * blk_peak;
                const float ctrl_scale = (env > 1e-6f) ? (0.5f / env) : 1.f;
                for (int i = 0; i < kBlockSize; ++i) ctrl_buf[i] = blk[i] * ctrl_scale;
                sess->run_controller(ctrl_buf.data(), kBlockSize, eq_params, n_params);
                sess->swap_state();

                std::copy_n(blk, kBlockSize, dry.data());
                // Spectral mask: range scales the predicted dB curve toward
                // 0 dB; speed sets the bin-gain smoother time constant. Both
                // applied inside set_params on each tick.
                auto& dsp = plug.chains[ch].autoeq_spec_per_class[cls];
                dsp->set_range_norm(amt.eq_range);
                dsp->set_boost_scale(amt.eq_boost_scale);
                dsp->set_speed_tau_ms(amt.eq_speed_ms);
                dsp->set_params(eq_params, n_params);
                dsp->process(blk, wet_a.data(), kBlockSize);
                if (ch == 0) {
                    // 5-point curve display at the historical PEQ band centres
                    // so the GUI's curve overlay stays where users expect.
                    static constexpr float kDisplayHz[5] =
                        {1010.f, 110.f, 1100.f, 7000.f, 10000.f};
                    float gains5[5];
                    dsp->sample_gains_db(kDisplayHz, gains5, 5);
                    plug.spectrum.set_eq_gains(gains5);
                    // 50-point bin display (log-spaced 20–20k Hz).
                    static const std::array<float, SpectrumAnalyzer::kNumBins> kBinHz = []() {
                        std::array<float, SpectrumAnalyzer::kNumBins> hz;
                        for (int i = 0; i < SpectrumAnalyzer::kNumBins; ++i)
                            hz[i] = 20.f * std::pow(1000.f,
                                float(i) / (SpectrumAnalyzer::kNumBins - 1));
                        return hz;
                    }();
                    float gains50[SpectrumAnalyzer::kNumBins];
                    dsp->sample_gains_db(kBinHz.data(), gains50,
                                         SpectrumAnalyzer::kNumBins);
                    plug.spectrum.set_eq_bins(gains50);
                }
                blend_(blk, dry.data(), wet_a.data(), amt.autoeq_wet_mix, kBlockSize);
            }
            break;
        }

        case StageID::Saturator: {
            const float pre  = std::pow(10.f, amt.sat_pre_db / 20.f);
            const float pst  = std::pow(10.f, amt.sat_post_db / 20.f);
            const float T    = amt.sat_thresh_lin;     // input axis scale (unity = 1.0)
            const float invT = 1.f / T;
            const bool  use_hpf = amt.sat_hpf_hz > 21.f;
            float* ch_buf[2] = {work_l, work_r};
            for (uint32_t ch = 0; ch < n_ch; ++ch) {
                float* blk   = ch_buf[ch];
                auto&  chain = plug.chains[ch];
                // DC offset introduced by bias; subtracted after eval to keep output AC.
                const float dc = static_cast<float>(
                    chain.saturator.eval(static_cast<double>(amt.sat_bias)));
                if (use_hpf) {
                    // Recompute 1st-order bilinear HPF coefficients when fc changes.
                    if (std::abs(amt.sat_hpf_hz - chain.sat_hpf_fc) > 0.5f) {
                        const float K = std::tan(
                            static_cast<float>(M_PI) * amt.sat_hpf_hz
                            / static_cast<float>(plug.sample_rate));
                        const float norm = 1.f / (1.f + K);
                        chain.sat_hpf_b0 =  norm;
                        chain.sat_hpf_b1 = -norm;
                        chain.sat_hpf_a1 = (K - 1.f) * norm;
                        chain.sat_hpf_fc = amt.sat_hpf_hz;
                    }
                    // Filter into wet_a (hi band); lo = blk - hi.
                    for (int i = 0; i < kBlockSize; ++i) {
                        const float x = blk[i];
                        wet_a[i] = chain.sat_hpf_b0 * x
                                 + chain.sat_hpf_b1 * chain.sat_hpf_x1
                                 - chain.sat_hpf_a1 * chain.sat_hpf_y1;
                        chain.sat_hpf_x1 = x;
                        chain.sat_hpf_y1 = wet_a[i];
                    }
                    // Saturate hi band with threshold + bias into wet_b.
                    for (int i = 0; i < kBlockSize; ++i) {
                        const float x_in = wet_a[i] * pre * invT + amt.sat_bias;
                        wet_b[i] = (static_cast<float>(chain.saturator.eval(
                            static_cast<double>(x_in))) - dc) * T * pst;
                    }
                    // Recombine: lo + blend(hi_dry, hi_wet).
                    for (int i = 0; i < kBlockSize; ++i)
                        blk[i] = (blk[i] - wet_a[i])
                                + (1.f - amt.sat_wet_mix) * wet_a[i]
                                +         amt.sat_wet_mix  * wet_b[i];
                } else {
                    std::copy_n(blk, kBlockSize, dry.data());
                    for (int i = 0; i < kBlockSize; ++i) {
                        const float x_in = blk[i] * pre * invT + amt.sat_bias;
                        wet_a[i] = (static_cast<float>(chain.saturator.eval(
                            static_cast<double>(x_in))) - dc) * T * pst;
                    }
                    blend_(blk, dry.data(), wet_a.data(), amt.sat_wet_mix, kBlockSize);
                }
            }
            break;
        }

        case StageID::SslComp: {
            // SSL-style bus comp: stateless long-RF causal TCN with hop
            // accumulation. The wet output queue trails the input by
            // (kSslHop - kBlockSize) samples — we delay the dry signal by the
            // same amount via a per-channel ring so the wet/dry blend stays
            // sample-aligned (otherwise blending current dry with delayed
            // wet produces a hop-rate comb-filter flutter).
            //
            // Streaming-mode TCN export contract: the ONNX takes `trace_len`
            // samples in (the entire ring, including the `rf-1` history
            // prefix) and produces `trace_len - (rf-1)` samples out — the
            // model's predictions for ring positions [rf-1, trace_len-1].
            // Output position i corresponds to ring position (rf-1 + i).
            //
            // Skipped if the SSL bundle wasn't shipped (ssl_comp_ort null) or
            // the wet mix is at zero.
            if (amt.ssl_comp_wet <= 0.f) break;
            if (!plug.chains[0].ssl_comp_ort) break;
            const int N           = g_state->ssl_comp_meta.trace_len;
            const int rf          = g_state->ssl_comp_meta.receptive_field;
            const int actual_olen = N - (rf - 1);  // ORT output length
            const int dry_delay_len = kSslHop - kBlockSize;
            float* ch_buf[2]={work_l,work_r};
            for (uint32_t ch=0;ch<n_ch;++ch) {
                float* blk=ch_buf[ch];
                std::copy_n(blk,kBlockSize,dry.data());

                auto& accum  = plug.chains[ch].ssl_comp_in_accum;
                auto& ring   = plug.chains[ch].ssl_comp_in_ring;
                auto& obuf   = plug.chains[ch].ssl_comp_out_buf;
                auto& outq   = plug.chains[ch].ssl_comp_out_queue;
                auto& dryd   = plug.chains[ch].ssl_comp_dry_delay;
                int&  fill   = plug.chains[ch].ssl_comp_in_fill;
                int&  avail  = plug.chains[ch].ssl_comp_out_avail;
                int&  rd     = plug.chains[ch].ssl_comp_out_read;
                int&  dwr    = plug.chains[ch].ssl_comp_dry_write;

                // Push input into the hop-sized accumulator.
                std::copy_n(blk, kBlockSize, accum.data() + fill);
                fill += kBlockSize;

                // When the accumulator fills, shift the ring by kSslHop, append
                // the accumulator at the tail, run ORT once, and stash the
                // trailing kSslHop samples of output into the playback queue.
                if (fill >= kSslHop) {
                    std::memmove(ring.data(),
                                 ring.data() + kSslHop,
                                 (N - kSslHop) * sizeof(float));
                    std::copy_n(accum.data(), kSslHop,
                                ring.data() + N - kSslHop);
                    fill = 0;

                    // ORT produces `actual_olen` (= N - rf + 1) samples;
                    // pass that as audio_out_len so OrtMiniSession doesn't
                    // read past the tensor end.
                    plug.chains[ch].ssl_comp_ort->run(ring.data(), N,
                        obuf.data(), actual_olen,
                        nullptr, 0, "audio_out");

                    // Output sample i corresponds to ring position (rf-1+i).
                    // We want the kSslHop newest predictions (ring positions
                    // [N-kSslHop, N-1]), which are at output positions
                    // [N-kSslHop-(rf-1), actual_olen-1] = the LAST kSslHop
                    // samples of the actual output.
                    std::copy_n(obuf.data() + actual_olen - kSslHop, kSslHop,
                                outq.data());
                    avail = kSslHop;
                    rd    = 0;
                }

                // Build a per-sample time-aligned dry block. With dry delay
                // length = kSslHop - kBlockSize, the dry sample we need to
                // blend against this call's wet was written into the delay
                // (kSslHop / kBlockSize - 1) host calls ago. Read those
                // samples out, then write the new dry in for future use.
                std::array<float, kBlockSize> dry_aligned{};
                if (dry_delay_len > 0) {
                    const int len = static_cast<int>(dryd.size());
                    for (int i = 0; i < kBlockSize; ++i) {
                        const int read_idx = (dwr + i) % len;
                        dry_aligned[i] = dryd[read_idx];
                        dryd[read_idx] = dry[i];   // overwrite with new dry
                    }
                    dwr = (dwr + kBlockSize) % len;
                } else {
                    std::copy_n(dry.data(), kBlockSize, dry_aligned.data());
                }

                // Pop kBlockSize samples from the output queue. While the
                // queue is short of a full block (the kSslHop-sample warm-up
                // window after activate, before the first ORT call), leave
                // the current dry untouched in blk so the host hears the
                // bypassed signal instead of silence or a comb of stale-zero
                // delayed-dry against silence.
                if (avail >= kBlockSize) {
                    std::copy_n(outq.data() + rd, kBlockSize, wet_a.data());
                    rd    += kBlockSize;
                    avail -= kBlockSize;
                    // Blend the (delayed) wet against the time-aligned dry,
                    // NOT the current dry — they're the same audio in
                    // absolute time, so this is the only mix that doesn't
                    // produce hop-rate comb-filter flutter.
                    blend_inplace_(wet_a.data(), dry_aligned.data(),
                                   1.f - amt.ssl_comp_wet, kBlockSize);
                    std::copy_n(wet_a.data(), kBlockSize, blk);
                }
                // else: leave blk = current dry (warm-up pass-through).
            }
            break;
        }

        case StageID::OutputLeveler: {
            if (amt.out_lvl_wet <= 0.f) break;
            plug.out_leveler.set_target(static_cast<double>(amt.out_lvl_target_lufs));
            std::array<float,kBlockSize> ol_l{},ol_r{};
            if (n_ch >= 2) {
                plug.out_leveler.process_linked(work_l,work_r,
                                                ol_l.data(),ol_r.data(),kBlockSize);
                blend_inplace_(work_l,ol_l.data(),amt.out_lvl_wet,kBlockSize);
                blend_inplace_(work_r,ol_r.data(),amt.out_lvl_wet,kBlockSize);
            } else {
                plug.out_leveler.process(work_l,ol_l.data(),kBlockSize);
                blend_inplace_(work_l,ol_l.data(),amt.out_lvl_wet,kBlockSize);
            }
            break;
        }

        } // switch

        // Capture stage output for the spectrum analyzer.
        plug.spectrum.push(pos, work_l, work_r, n_ch, kBlockSize);
    } // for stage

    // When a full 2048-sample frame has accumulated, hand off to the main thread.
    if (plug.spectrum.advance_and_transfer())
        plug.host->request_callback(plug.host);

    // Trim + TruePeakCeiling — always last, not user-reorderable.
    if (amt.trim_lin != 1.f) {
        for (int i=0;i<kBlockSize;++i) work_l[i]*=amt.trim_lin;
        if (n_ch>=2) for (int i=0;i<kBlockSize;++i) work_r[i]*=amt.trim_lin;
    }
    for (uint32_t ch=0;ch<n_ch;++ch) {
        float* blk=(ch==0)?work_l:work_r;
        plug.chains[ch].ceiling.process(blk,plug.chains[ch].out_buf.data(),kBlockSize);
        plug.chains[ch].out_avail=kBlockSize;
        plug.chains[ch].out_read=0;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// process — accumulator-driven 128-sample block flushes
// ---------------------------------------------------------------------------

static void apply_events_(Plugin* plug, const clap_input_events_t* in_events) {
    if (!in_events) return;
    const uint32_t n = in_events->size(in_events);
    for (uint32_t i = 0; i < n; ++i) {
        const auto* hdr = in_events->get(in_events, i);
        if (!hdr) continue;
        if (hdr->space_id != CLAP_CORE_EVENT_SPACE_ID) continue;
        if (hdr->type != CLAP_EVENT_PARAM_VALUE) continue;
        const auto* pv = reinterpret_cast<const clap_event_param_value_t*>(hdr);
        for (size_t k = 0; k < plug->meta->controls.size(); ++k) {
            if (param_id_for(plug->meta->effect_name, plug->meta->controls[k].id) == pv->param_id) {
                plug->control_values[k] = static_cast<float>(pv->value);
                break;
            }
        }
    }
}

// GUI-thread callbacks — write pending changes into thread-safe queues.
static void tone_on_param_change(void* plug_ptr, const char* param_id, float value) {
    auto* plug = static_cast<Plugin*>(plug_ptr);
    for (size_t i = 0; i < plug->meta->controls.size(); ++i) {
        if (plug->meta->controls[i].id == param_id) {
            std::lock_guard<std::mutex> lk(plug->param_mutex);
            plug->param_queue.emplace_back(static_cast<int>(i), value);
            break;
        }
    }
}
static void tone_on_order_change(void* plug_ptr, const int* order, int count) {
    auto* plug = static_cast<Plugin*>(plug_ptr);
    if (count != kNumStages) return;
    std::lock_guard<std::mutex> lk(plug->order_mutex);
    for (int i = 0; i < kNumStages; ++i) plug->pending_order[i] = order[i];
    plug->order_pending = true;
}

static clap_process_status plugin_process(const clap_plugin_t* p, const clap_process_t* process) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    apply_events_(plug, process->in_events);

    // Drain GUI param queue (try_lock — never blocks audio thread).
    {
        std::unique_lock<std::mutex> lk(plug->param_mutex, std::try_to_lock);
        if (lk.owns_lock() && !plug->param_queue.empty()) {
            for (auto& [idx, val] : plug->param_queue)
                plug->control_values[idx] = val;
            plug->param_queue.clear();
        }
    }
    // Drain GUI order change.
    {
        std::unique_lock<std::mutex> lk(plug->order_mutex, std::try_to_lock);
        if (lk.owns_lock() && plug->order_pending) {
            plug->processor_order = plug->pending_order;
            plug->order_pending = false;
        }
    }

    // Recompute latency after any param updates; notify host on change.
    {
        const uint32_t new_lat = compute_latency_(*plug);
        if (new_lat != plug->current_latency.load(std::memory_order_relaxed)) {
            plug->current_latency.store(new_lat, std::memory_order_relaxed);
            plug->latency_needs_notify.store(true, std::memory_order_relaxed);
            plug->host->request_callback(plug->host);
        }
    }

    const uint32_t n_frames = process->frames_count;
    if (n_frames == 0) return CLAP_PROCESS_CONTINUE;
    if (process->audio_inputs_count == 0 || process->audio_outputs_count == 0)
        return CLAP_PROCESS_ERROR;

    const float* const* in_ch  = process->audio_inputs[0].data32;
    float* const*       out_ch = process->audio_outputs[0].data32;
    const uint32_t n_ch = std::min<uint32_t>(
        static_cast<uint32_t>(plug->chains.size()),
        std::min(process->audio_inputs[0].channel_count,
                 process->audio_outputs[0].channel_count));
    if (n_ch == 0) return CLAP_PROCESS_ERROR;

    AmountSnapshot amt = resolve_amount_(*plug);

    // All channels fill/drain at the same rate, so use one shared in_pos/out_pos.
    uint32_t in_pos = 0, out_pos = 0;

    while (out_pos < n_frames) {
        // Drain all channels' output rings together.
        while (out_pos < n_frames && plug->chains[0].out_read < plug->chains[0].out_avail) {
            for (uint32_t ch = 0; ch < n_ch; ++ch)
                out_ch[ch][out_pos] = plug->chains[ch].out_buf[plug->chains[ch].out_read];
            for (uint32_t ch = 0; ch < n_ch; ++ch)
                ++plug->chains[ch].out_read;
            ++out_pos;
        }
        if (out_pos >= n_frames) break;

        if (in_pos >= n_frames) {
            while (out_pos < n_frames) {
                for (uint32_t ch = 0; ch < n_ch; ++ch) out_ch[ch][out_pos] = 0.0f;
                ++out_pos;
            }
            break;
        }

        // Push input into all channels' accumulators simultaneously.
        const uint32_t take = std::min<uint32_t>(
            n_frames - in_pos,
            static_cast<uint32_t>(kBlockSize - plug->chains[0].in_fill));
        for (uint32_t ch = 0; ch < n_ch; ++ch) {
            std::copy_n(in_ch[ch] + in_pos,
                        take,
                        plug->chains[ch].in_buf.data() + plug->chains[ch].in_fill);
            plug->chains[ch].in_fill += take;
        }
        in_pos += take;

        if (plug->chains[0].in_fill < kBlockSize) {
            // Accumulator not yet full — pad output with zeros.
            while (out_pos < n_frames) {
                for (uint32_t ch = 0; ch < n_ch; ++ch) out_ch[ch][out_pos] = 0.0f;
                ++out_pos;
            }
            break;
        }

        // Blocks are full — run the unified stage chain.
        std::array<float, kBlockSize> work_l{}, work_r{};
        std::copy_n(plug->chains[0].in_buf.data(), kBlockSize, work_l.data());
        if (n_ch >= 2) std::copy_n(plug->chains[1].in_buf.data(), kBlockSize, work_r.data());
        flush_chain_block_(*plug, work_l.data(), work_r.data(), n_ch, amt);
        for (uint32_t ch = 0; ch < n_ch; ++ch)
            plug->chains[ch].in_fill = 0;
    }
    return CLAP_PROCESS_CONTINUE;
}

// ---------------------------------------------------------------------------
// GUI extension  (CLAP_EXT_GUI, CLAP_WINDOW_API_COCOA)
// ---------------------------------------------------------------------------

static void gui_send_full_state_(Plugin* plug) {
    if (!plug->gui_state) return;
    const size_t n = plug->meta->controls.size();
    std::vector<ToneParamInfo> params(n);
    // Cache class-name pointers for the CLS enum picker. The vector itself
    // backs the const char* array we pass in ToneParamInfo::enum_options;
    // both must outlive the tone_gui_send_init() call (it copies the strings
    // into the JS payload synchronously on the main thread or buffers them).
    const auto& classes = g_state->tone_meta.auto_eq.class_order;
    std::vector<const char*> class_ptrs;
    class_ptrs.reserve(classes.size());
    for (const auto& s : classes) class_ptrs.push_back(s.c_str());

    for (size_t i = 0; i < n; ++i) {
        const auto& c = plug->meta->controls[i];
        params[i].id             = c.id.c_str();
        params[i].name           = c.name.c_str();
        params[i].min            = c.min;
        params[i].max            = c.max;
        params[i].def            = c.def;
        params[i].unit           = c.unit.c_str();
        params[i].current_value  = plug->control_values[i];
        params[i].enum_options   = nullptr;
        params[i].n_enum_options = 0;
        if (c.id == "CLS" && !class_ptrs.empty()) {
            params[i].enum_options   = class_ptrs.data();
            params[i].n_enum_options = static_cast<int>(class_ptrs.size());
        }
    }
    tone_gui_send_init(plug->gui_state,
                       params.data(), static_cast<int>(n),
                       plug->processor_order.data(), kNumStages);
}

static bool gui_is_api_supported(const clap_plugin_t*, const char* api, bool is_floating) {
    return !is_floating && std::strcmp(api, CLAP_WINDOW_API_COCOA) == 0;
}
static bool gui_get_preferred_api(const clap_plugin_t*, const char** api, bool* is_floating) {
    *api = CLAP_WINDOW_API_COCOA;
    *is_floating = false;
    return true;
}
static bool gui_create(const clap_plugin_t* p, const char* api, bool is_floating) {
    if (!gui_is_api_supported(p, api, is_floating)) return false;
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    if (plug->gui_state) return true;  // already exists
    plug->gui_state = tone_gui_create(
        plug,
        g_state->resources_dir.c_str(),
        tone_on_param_change,
        tone_on_order_change);
    return plug->gui_state != nullptr;
}
static void gui_destroy_fn(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    tone_gui_destroy(plug->gui_state);
    plug->gui_state = nullptr;
}
static bool gui_set_scale(const clap_plugin_t*, double) { return false; }
static bool gui_get_size(const clap_plugin_t*, uint32_t* w, uint32_t* h) {
    tone_gui_get_size(w, h);
    return true;
}
static bool gui_can_resize(const clap_plugin_t*) { return false; }
static bool gui_get_resize_hints(const clap_plugin_t*, clap_gui_resize_hints_t* hints) {
    hints->can_resize_horizontally = false;
    hints->can_resize_vertically   = false;
    hints->preserve_aspect_ratio   = false;
    hints->aspect_ratio_width  = 700;
    hints->aspect_ratio_height = 460;
    return false;
}
static bool gui_adjust_size(const clap_plugin_t*, uint32_t* w, uint32_t* h) {
    tone_gui_get_size(w, h);
    return true;
}
static bool gui_set_size(const clap_plugin_t*, uint32_t, uint32_t) { return true; }
static bool gui_set_parent(const clap_plugin_t* p, const clap_window_t* window) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    if (!plug->gui_state) return false;
    return tone_gui_set_parent(plug->gui_state, window->cocoa);
}
static bool gui_set_transient(const clap_plugin_t*, const clap_window_t*) { return false; }
static void gui_suggest_title(const clap_plugin_t*, const char*) {}
static bool gui_show(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    if (!plug->gui_state) return false;
    gui_send_full_state_(plug);
    tone_gui_show(plug->gui_state);
    return true;
}
static bool gui_hide(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    tone_gui_hide(plug->gui_state);
    return true;
}

static const clap_plugin_gui_t s_ext_gui = {
    gui_is_api_supported,
    gui_get_preferred_api,
    gui_create,
    gui_destroy_fn,
    gui_set_scale,
    gui_get_size,
    gui_can_resize,
    gui_get_resize_hints,
    gui_adjust_size,
    gui_set_size,
    gui_set_parent,
    gui_set_transient,
    gui_suggest_title,
    gui_show,
    gui_hide,
};

static const void* plugin_get_extension(const clap_plugin_t*, const char* id) {
    if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS) == 0) return &s_ext_audio_ports;
    if (std::strcmp(id, CLAP_EXT_PARAMS)       == 0) return &s_ext_params;
    if (std::strcmp(id, CLAP_EXT_LATENCY)      == 0) return &s_ext_latency;
    if (std::strcmp(id, CLAP_EXT_STATE)        == 0) return &s_ext_state;
    if (std::strcmp(id, CLAP_EXT_GUI)          == 0) return &s_ext_gui;
    return nullptr;
}

static void plugin_on_main_thread(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);

    if (plug->latency_needs_notify.exchange(false, std::memory_order_relaxed)) {
        if (plug->host_latency_ext && plug->host_latency_ext->changed)
            plug->host_latency_ext->changed(plug->host);
    }

    if (!plug->gui_state) return;
    if (plug->spectrum.process_if_ready(plug->sample_rate)) {
        const std::string js = plug->spectrum.build_js(plug->processor_order);
        tone_gui_eval_js(plug->gui_state, js.c_str());
    }
}

// ---------------------------------------------------------------------------
// Factory + entry
// ---------------------------------------------------------------------------

static const clap_plugin_t* factory_create_plugin(const clap_plugin_factory_t*,
                                                  const clap_host_t* host,
                                                  const char* plugin_id) {
    if (!g_state) return nullptr;
    if (std::strcmp(plugin_id, g_state->plugin_id_str.c_str()) != 0) return nullptr;

    auto* plug = new Plugin{};
    plug->host     = host;
    plug->meta     = &g_state->tone_meta;
    plug->channels = 2;

    plug->plugin.desc             = &g_state->descriptor;
    plug->plugin.plugin_data      = plug;
    plug->plugin.init             = plugin_init;
    plug->plugin.destroy          = plugin_destroy;
    plug->plugin.activate         = plugin_activate;
    plug->plugin.deactivate       = plugin_deactivate;
    plug->plugin.start_processing = plugin_start_processing;
    plug->plugin.stop_processing  = plugin_stop_processing;
    plug->plugin.reset            = plugin_reset;
    plug->plugin.process          = plugin_process;
    plug->plugin.get_extension    = plugin_get_extension;
    plug->plugin.on_main_thread   = plugin_on_main_thread;
    return &plug->plugin;
}

static uint32_t factory_get_plugin_count(const clap_plugin_factory_t*) { return g_state ? 1 : 0; }
static const clap_plugin_descriptor_t* factory_get_plugin_descriptor(const clap_plugin_factory_t*,
                                                                     uint32_t index) {
    if (!g_state || index != 0) return nullptr;
    return &g_state->descriptor;
}

static const clap_plugin_factory_t s_factory = {
    factory_get_plugin_count, factory_get_plugin_descriptor, factory_create_plugin,
};

static bool entry_init(const char* /*plugin_path*/) {
    if (g_state) return true;
    try {
        auto st = std::make_unique<ModuleState>();
        st->bundle_dir   = find_bundle_contents_();
        if (st->bundle_dir.empty()) return false;
        st->resources_dir = st->bundle_dir + "/Resources";

        st->tone_meta = load_composite_meta(st->resources_dir + "/tone_meta.json");
        st->sat_meta    = load_meta(st->resources_dir + "/" +
                                    st->tone_meta.sub_bundles.at("saturator")
                                    + "/plugin_meta.json");
        // SSL bus comp is optional — older bundles don't ship it. If present,
        // load its meta so we can size per-channel ring buffers at activate.
        if (st->tone_meta.sub_bundles.count("ssl_comp")) {
            st->ssl_comp_meta = load_meta(
                st->resources_dir + "/" + st->tone_meta.sub_bundles.at("ssl_comp")
                + "/plugin_meta.json");
            st->ssl_comp_loaded = true;
        }

        // Load every auto-EQ class meta in the canonical class_order. All
        // classes must declare spectral_mask_eq with identical geometry
        // (n_fft, hop, n_bands, gain range, frequency range), since the
        // runtime SpectralMaskEq downstream is shared and only the
        // controller ONNX is swapped on a class change.
        st->autoeq_metas.clear();
        st->autoeq_class_index.clear();
        st->autoeq_metas.reserve(st->tone_meta.auto_eq.class_order.size());
        for (const auto& cls : st->tone_meta.auto_eq.class_order) {
            const std::string& dir = st->tone_meta.auto_eq.classes.at(cls);
            PluginMeta m = load_meta(st->resources_dir + "/" + dir + "/plugin_meta.json");
            if (m.dsp_blocks.empty()) {
                throw std::runtime_error(
                    "auto_eq sub-bundle '" + dir + "' has no dsp_blocks");
            }
            st->autoeq_class_index[cls] = st->autoeq_metas.size();
            st->autoeq_metas.push_back(std::move(m));
        }
        if (st->autoeq_metas.empty()) {
            throw std::runtime_error("tone_meta.auto_eq is empty");
        }

        // Pull the DSP block payloads we need at chain construction time.
        if (st->sat_meta.dsp_blocks.empty()) {
            throw std::runtime_error("saturator sub-bundle has no dsp_blocks");
        }
        st->sat_rational = std::get<RationalAParams>(st->sat_meta.dsp_blocks[0].params);
        // Per-class DSP-block payloads (all spectral_mask_eq).
        st->autoeq_dsp_per_class.clear();
        st->autoeq_dsp_per_class.reserve(st->autoeq_metas.size());
        for (const auto& m : st->autoeq_metas) {
            st->autoeq_dsp_per_class.push_back(m.dsp_blocks[0]);
        }
        st->autoeq_default_idx = static_cast<int>(
            st->autoeq_class_index.at(st->tone_meta.auto_eq.default_class));

        st->ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "nablafx-tone");
        populate_descriptor_(*st);
        g_state = st.release();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

static void entry_deinit() { delete g_state; g_state = nullptr; }

static const void* entry_get_factory(const char* factory_id) {
    if (std::strcmp(factory_id, CLAP_PLUGIN_FACTORY_ID) == 0) return &s_factory;
    return nullptr;
}

}  // namespace nablafx_tone

extern "C" {
CLAP_EXPORT const clap_plugin_entry_t clap_entry = {
    CLAP_VERSION_INIT,
    nablafx_tone::entry_init,
    nablafx_tone::entry_deinit,
    nablafx_tone::entry_get_factory,
};
}
