#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace nablafx {

struct ControlSpec {
    std::string id;
    std::string name;
    float       min;
    float       max;
    float       def;
    float       skew;
    std::string unit;
};

struct StateSpec {
    std::string          name;   // ONNX name stem (e.g. "processor_lstm_h")
    std::vector<int64_t> shape;  // (num_layers, 1, hidden_size)
    std::string          dtype;  // always "float32" in v1
};

// ---- DSP block payloads (schema_version >= 2) -----------------------------

// Rational version A nonlinearity: P(x)/Q(x) where
//   P(x) = sum_i numerator[i] * x^i           (length n+1)
//   Q(x) = 1 + sum_j |denominator[j] * x^j|   (length m, j starts at 1)
struct RationalAParams {
    std::vector<float> numerator;
    std::vector<float> denominator;
};

// 5-band parametric EQ: gain/Q come from the controller's ONNX output every
// `block_size` samples (sigmoid in [0, 1]); the C++ side denormalizes via
// per-band ranges and runs a fixed-frequency biquad cascade.
struct ParametricEq5BandParams {
    enum class Kind { LowShelf, Peaking, HighShelf };
    struct Band {
        std::string name;
        Kind        kind;
        float       cutoff_freq;       // Hz — fixed (freeze_freqs=true)
        float       gain_db_min, gain_db_max;
        float       q_min, q_max;
        int         ch_gain;           // index into the [num_control_params] vector
        int         ch_q;
    };
    int               sample_rate;
    int               block_size;      // ONNX call cadence
    int               num_control_params;
    std::vector<Band> bands;
};

// STFT-domain magnitude-mask EQ. Controller emits ``n_bands`` sigmoid values
// per block; the C++ runtime computes the mel filterbank from the geometry
// parameters (sample_rate, n_fft, n_bands, f_min, f_max) and applies the mask
// via overlap-add. Latency is ``n_fft - hop`` samples.
struct SpectralMaskEqParams {
    int   sample_rate;
    int   block_size;          // controller call cadence
    int   num_control_params;  // == n_bands
    int   n_fft;
    int   hop;
    int   n_bands;
    float min_gain_db;
    float max_gain_db;
    float f_min;
    float f_max;
};

using DspBlockParams = std::variant<RationalAParams, ParametricEq5BandParams, SpectralMaskEqParams>;

struct DspBlockSpec {
    std::string    kind;   // "rational_a" | "parametric_eq_5band" | "spectral_mask_eq"
    std::string    name;
    DspBlockParams params;
};

// ---- Stage classification (schema_version >= 2) ---------------------------
//
//   Nn      — single ONNX graph; state_tensors / input_names / output_names
//             populated; no DSP blocks. (BlackBoxModel-derived)
//   Dsp     — pure DSP, no ONNX; only dsp_blocks populated.
//   NnDsp   — controller NN exported to ONNX; downstream DSP runs natively
//             via dsp_blocks.
enum class StageKind { Nn, Dsp, NnDsp };

struct PluginMeta {
    int                       schema_version{};
    std::string               effect_name;
    std::string               model_id;
    std::string               architecture;   // "tcn" | "lstm" | "gcn" | "dsp"
    int                       sample_rate{};
    int                       channels{};
    bool                      causal{};
    int                       receptive_field{};
    int                       latency_samples{};
    int                       num_controls{};
    // Fixed audio_in length the ONNX was traced at. Stateless models that
    // need RF samples of context per call (e.g. long-RF causal TCN) advertise
    // their trace length here; the host must call run() with exactly this
    // many samples and use a ring buffer to feed older context. 0 = legacy
    // bundle predating this field; treat audio_in as variable-length.
    int                       trace_len{0};
    StageKind                 stage_kind{StageKind::Nn};
    std::vector<ControlSpec>  controls;
    std::vector<StateSpec>    state_tensors;
    std::vector<std::string>  input_names;
    std::vector<std::string>  output_names;
    std::vector<DspBlockSpec> dsp_blocks;
};

// Parse a plugin_meta.json file from disk. Throws std::runtime_error on any
// problem (missing file, malformed JSON, unknown schema_version).
PluginMeta load_meta(const std::string& path);

}  // namespace nablafx
