// STFT-domain magnitude-mask EQ with N mel-spaced bands.
//
// Mirrors the Python `SpectralMaskEQ` processor: same n_fft, hop, mel band
// edges (HTK formula, f_min..f_max), Hann analysis+synthesis windows, and
// zero-phase magnitude-only application. Backed by Apple Accelerate vDSP for
// the real-input FFT — macOS-only, which matches build.sh's macOS-only
// constraint.
//
// Streaming contract:
//   - The host calls process(in, out, n) with arbitrary `n` (commonly 128).
//   - Internally we accumulate samples into an n_fft-sized analysis ring; on
//     every `hop` accumulated samples we run an FFT frame, apply the per-band
//     gain mask (set via set_params), inverse-FFT, window, and OLA into an
//     output ring.
//   - process() pulls `n` samples of finished output. Latency = n_fft - hop.
//
// Pure DSP — no CLAP / ORT / std::variant deps so it can be unit-tested
// standalone.

#pragma once

#include <Accelerate/Accelerate.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "meta.hpp"

namespace nablafx {

class SpectralMaskEq {
public:
    SpectralMaskEq() = default;
    ~SpectralMaskEq() {
        if (fft_setup_) vDSP_destroy_fftsetup(fft_setup_);
    }

    SpectralMaskEq(const SpectralMaskEq&)            = delete;
    SpectralMaskEq& operator=(const SpectralMaskEq&) = delete;

    void reset(const SpectralMaskEqParams& cfg) {
        cfg_ = cfg;
        if (cfg_.n_fft <= 0 || (cfg_.n_fft & (cfg_.n_fft - 1)) != 0) {
            throw std::runtime_error(
                "spectral_mask_eq: n_fft must be a power of two, got " +
                std::to_string(cfg_.n_fft));
        }
        if (cfg_.hop <= 0 || cfg_.hop > cfg_.n_fft) {
            throw std::runtime_error("spectral_mask_eq: hop must be in (0, n_fft]");
        }

        n_fft_     = cfg_.n_fft;
        hop_       = cfg_.hop;
        n_bands_   = cfg_.n_bands;
        n_freq_    = n_fft_ / 2 + 1;
        log2_nfft_ = static_cast<vDSP_Length>(std::log2(static_cast<double>(n_fft_)));

        if (fft_setup_) vDSP_destroy_fftsetup(fft_setup_);
        fft_setup_ = vDSP_create_fftsetup(log2_nfft_, kFFTRadix2);
        if (!fft_setup_) {
            throw std::runtime_error("spectral_mask_eq: vDSP_create_fftsetup failed");
        }

        window_.assign(n_fft_, 0.0f);
        for (int n = 0; n < n_fft_; ++n) {
            window_[n] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * n / n_fft_));
        }

        in_ring_.assign(n_fft_, 0.0f);
        out_ring_.assign(n_fft_ + hop_, 0.0f);
        norm_ring_.assign(n_fft_ + hop_, 0.0f);  // OLA window² accumulator
        in_fill_         = 0;
        samples_since_   = 0;
        out_write_       = 0;
        out_read_        = 0;
        out_avail_       = 0;

        // Mel filterbank.
        build_mel_(cfg_.sample_rate, n_fft_, n_bands_, cfg_.f_min, cfg_.f_max);

        // 1/6-octave per-bin smoothing kernel, in dB domain. Kernel sigma
        // scales with frequency (constant fraction of an octave), so LF bins
        // get a near-degenerate kernel (one bin) and HF bins get a wide one.
        build_freq_smoothing_kernel_(cfg_.sample_rate, n_fft_, /*octave_frac=*/1.0f / 6.0f);
        bin_db_buf_.assign(n_freq_, 0.0f);

        // Per-bin gain mask (linear). bin_gain_target_ is the controller's
        // most recent prediction; bin_gain_ is the smoothed value the FFT
        // frame actually applies. set_params updates the target every
        // block_size samples (~2.9 ms at 44.1k); the FFT consumes bin_gain_
        // every hop samples (~11.6 ms). Smoothing the controller→mask
        // transition kills the frame-rate (~86 Hz) modulation that otherwise
        // shows up as graininess on kick/bass content.
        bin_gain_.assign(n_freq_, 1.0f);
        bin_gain_target_.assign(n_freq_, 1.0f);
        // Smoother time constant defaults to 25 ms; runtime can override via
        // set_speed_tau_ms(). Stored as ms so the (re)compute is one std::exp
        // and only fires when the user changes the value.
        speed_tau_ms_      = 25.0f;
        speed_tau_cached_  = -1.0f;  // forces alpha recompute on first set_params
        recompute_alpha_();

        // Range scale: 1.0 = full trained ±max_gain_db, 0.0 = bypass.
        range_norm_ = 1.0f;

        // Scratch buffers for FFT.
        windowed_.assign(n_fft_, 0.0f);
        split_real_.assign(n_fft_ / 2, 0.0f);
        split_imag_.assign(n_fft_ / 2, 0.0f);
        time_out_.assign(n_fft_, 0.0f);

        // Minimum-phase reconstruction scratch + output filter.
        cep_split_real_.assign(n_fft_ / 2, 0.0f);
        cep_split_imag_.assign(n_fft_ / 2, 0.0f);
        cep_time_.assign(n_fft_, 0.0f);
        h_mp_real_.assign(n_freq_, 1.0f);
        h_mp_imag_.assign(n_freq_, 0.0f);

        // vDSP forward+inverse round-trip scale is 2*n_fft (Apple vDSP guide:
        // "divide by 2n to recover original values after inverse"). We apply
        // 1/(2*n_fft) in the OLA write and then divide per-sample by the
        // accumulated sum of window² — matching torch.istft's normalization —
        // so output is correctly scaled regardless of the COLA sum varying
        // between 0.5 and 1.0 over the hop cycle.
        ola_scale_ = 1.0f / (2.0f * static_cast<float>(n_fft_));
    }

    // Runtime control: scale all per-band gains by ``r`` ∈ [0, 1] before
    // applying. r=1 → full predicted EQ; r=0 → flat (bypass). Smooth on the
    // user side; here it just multiplies into band_db every set_params tick.
    void set_range_norm(float r) {
        range_norm_ = std::clamp(r, 0.0f, 1.0f);
    }
    void set_boost_scale(float s) {
        boost_scale_ = std::clamp(s, 0.0f, 1.0f);
    }

    // Runtime control: time constant (ms) for the per-block bin-gain smoother.
    // ~5 ms = snappy / transient-tracking; ~200 ms = slow / mastering-style.
    void set_speed_tau_ms(float ms) {
        if (ms < 0.1f) ms = 0.1f;
        speed_tau_ms_ = ms;
    }

    // Apply latest controller output: ``params`` holds n_bands sigmoid values
    // in [0, 1]. Updates the per-bin linear gain mask.
    void set_params(const float* params, std::size_t n) {
        if (static_cast<int>(n) != cfg_.num_control_params) {
            throw std::runtime_error(
                "spectral_mask_eq::set_params: expected " +
                std::to_string(cfg_.num_control_params) +
                ", got " + std::to_string(n));
        }
        // Per-band gain in dB, then per-bin via mel_band_to_bin_ matrix.
        const float gain_span = cfg_.max_gain_db - cfg_.min_gain_db;
        // Per-band dB. Range knob is applied as a scale around 0 dB so the
        // user's "Range" reduces the predicted EQ's depth proportionally
        // without flipping the curve's sign or biasing the mid bands.
        std::vector<float> band_db(n_bands_, 0.0f);
        for (int b = 0; b < n_bands_; ++b) {
            float g = params[b];
            if (g < 0.0f) g = 0.0f;
            if (g > 1.0f) g = 1.0f;
            float db = (cfg_.min_gain_db + g * gain_span) * range_norm_;
            band_db[b] = db > 0.f ? db * boost_scale_ : db;
        }
        // Refresh the smoother's alpha if the user moved the Speed knob.
        if (speed_tau_ms_ != speed_tau_cached_) recompute_alpha_();
        // Step 1: per-bin gain in dB (linear band→bin mix).
        for (int k = 0; k < n_freq_; ++k) {
            float sum = 0.0f;
            for (int b = 0; b < n_bands_; ++b) {
                sum += band_to_bin_[b * n_freq_ + k] * band_db[b];
            }
            bin_db_buf_[k] = (bin_norm_[k] > 1e-6f) ? (sum / bin_norm_[k]) : 0.0f;
        }
        // Step 2: smooth across frequency in dB using the precomputed
        // 1/6-octave Gaussian kernel. Smooths band-edge interpolation wiggle
        // and the partial-tone-jitter that produces "musical noise" on tonal
        // mid/high content. dB smoothing → geometric mean in linear,
        // perceptually well-behaved when adjacent bins differ by many dB.
        for (int k = 0; k < n_freq_; ++k) {
            const int   start = freq_kernel_start_[k];
            const int   len   = freq_kernel_len_[k];
            const int   wbase = freq_kernel_woff_[k];
            float       acc   = 0.0f;
            for (int j = 0; j < len; ++j) {
                acc += freq_kernel_w_[wbase + j] * bin_db_buf_[start + j];
            }
            bin_gain_target_[k] = std::pow(10.0f, acc / 20.0f);
        }
        // Advance the smoother one step (per set_params tick = once per
        // block_size samples). At a ~25 ms time constant this knocks ~5 dB
        // off any sudden gain step over the first ~10 ms.
        const float a = mask_smooth_alpha_;
        const float ia = 1.0f - a;
        for (int k = 0; k < n_freq_; ++k) {
            bin_gain_[k] = a * bin_gain_[k] + ia * bin_gain_target_[k];
        }
    }

    // In-place safe.
    void process(const float* in, float* out, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            // Push input into analysis ring.
            in_ring_[in_fill_] = in[i];
            in_fill_ = (in_fill_ + 1) % n_fft_;
            ++samples_since_;

            // Run an FFT frame every `hop` samples.
            if (samples_since_ >= hop_) {
                samples_since_ -= hop_;
                run_frame_();
            }

            // Hand back one sample. Divide by accumulated window² to match
            // torch.istft per-sample normalisation; guards against near-zero
            // norm at Hann window edges.
            if (out_avail_ > 0) {
                const int   rd   = out_read_;
                const float norm = norm_ring_[rd];
                out[i] = (norm > 1e-8f) ? (out_ring_[rd] / norm) : 0.0f;
                out_ring_[rd]  = 0.0f;
                norm_ring_[rd] = 0.0f;
                out_read_ = (rd + 1) % static_cast<int>(out_ring_.size());
                --out_avail_;
            } else {
                out[i] = 0.0f;
            }
        }
    }

    // Sample the current per-bin gain mask at n arbitrary frequencies (Hz)
    // and return linear→dB values. Used to populate the 5-band display.
    void sample_gains_db(const float* hz_arr, float* db_arr, int n) const {
        for (int i = 0; i < n; ++i) {
            int bin = static_cast<int>(
                std::round(hz_arr[i] * n_fft_ / static_cast<float>(cfg_.sample_rate)));
            bin = std::max(0, std::min(bin, n_freq_ - 1));
            const float g = bin_gain_[bin];
            db_arr[i] = (g > 1e-8f) ? 20.0f * std::log10(g) : -80.0f;
        }
    }

    int latency_samples() const { return n_fft_ - hop_; }
    int block_size() const { return cfg_.block_size; }
    int num_control_params() const { return cfg_.num_control_params; }

private:
    void run_frame_() {
        // Copy ring (oldest first) into windowed_.
        for (int n = 0; n < n_fft_; ++n) {
            const int src = (in_fill_ + n) % n_fft_;
            windowed_[n] = in_ring_[src] * window_[n];
        }

        // Pack real input into split form for vDSP.
        DSPSplitComplex split{split_real_.data(), split_imag_.data()};
        vDSP_ctoz(reinterpret_cast<DSPComplex*>(windowed_.data()), 2,
                  &split, 1, n_fft_ / 2);

        // Forward FFT (in-place split-complex).
        vDSP_fft_zrip(fft_setup_, &split, 1, log2_nfft_, kFFTDirection_Forward);

        // After zrip-forward: split_real_[0] = DC,
        //                    split_imag_[0] = Nyquist real (packed),
        //                    split_real_[k]+i*split_imag_[k] = bin k for 0 < k < n_fft/2
        //
        // Build a *minimum-phase* per-bin filter H_mp from the magnitude mask
        // and apply it as a complex multiply. Same |H| as before, but the
        // impulse response is causal and asymmetric — no pre-ring on HF cuts.
        // Pre-ring was the dominant cause of the "loss of top-end energy"
        // perception on transient material (kicks/cymbals smeared into the
        // pre-strike silence by the symmetric IR).
        compute_min_phase_(bin_gain_.data());
        // DC and Nyquist are real for any min-phase filter from a real cepstrum.
        const float dc_re      = split.realp[0];
        const float ny_re      = split.imagp[0];
        split.realp[0] = dc_re * h_mp_real_[0];
        split.imagp[0] = ny_re * h_mp_real_[n_freq_ - 1];
        for (int k = 1; k < n_fft_ / 2; ++k) {
            const float xr = split.realp[k];
            const float xi = split.imagp[k];
            const float hr = h_mp_real_[k];
            const float hi = h_mp_imag_[k];
            split.realp[k] = xr * hr - xi * hi;
            split.imagp[k] = xr * hi + xi * hr;
        }

        // Inverse FFT.
        vDSP_fft_zrip(fft_setup_, &split, 1, log2_nfft_, kFFTDirection_Inverse);

        // Unpack split-complex back into time domain (interleaved).
        vDSP_ztoc(&split, 1,
                  reinterpret_cast<DSPComplex*>(time_out_.data()), 2,
                  n_fft_ / 2);

        // Hann²-OLA: accumulate audio (scaled by 1/(2N)) and window² into
        // parallel rings. Per-sample division in process() normalises away the
        // varying Hann² COLA sum (0.5–1.0 for hop=N/2), mirroring torch.istft.
        const int ring_sz = static_cast<int>(out_ring_.size());
        for (int n = 0; n < n_fft_; ++n) {
            const int idx = (out_write_ + n) % ring_sz;
            out_ring_[idx]  += time_out_[n] * window_[n] * ola_scale_;
            norm_ring_[idx] += window_[n] * window_[n];
        }
        out_write_ = (out_write_ + hop_) % ring_sz;
        out_avail_ += hop_;
    }

    // Build a minimum-phase per-bin filter from a magnitude vector
    // ``mag[0..n_freq-1]``. Writes complex coefficients into h_mp_real_ /
    // h_mp_imag_ (length n_freq). DC and Nyquist come out purely real.
    //
    // Recipe (Oppenheim & Schafer, "Real Cepstrum → Min-Phase"):
    //   1. log_mag[k] = log(max(mag[k], floor))
    //   2. Real cepstrum c[n] = IDFT{log_mag} (real-valued via conjugate
    //      symmetry of log_mag).
    //   3. c_min[n] = c[n] · w[n] where
    //         w[0] = 1, w[1..N/2-1] = 2, w[N/2] = 1, w[N/2+1..N-1] = 0
    //      → folds the anti-causal half of the cepstrum onto the causal half.
    //   4. log H_mp[k] = DFT{c_min}.
    //   5. H_mp[k] = exp(log H_mp[k]).
    void compute_min_phase_(const float* mag) {
        DSPSplitComplex cep_split{cep_split_real_.data(), cep_split_imag_.data()};

        // Step 1+2: pack log|H| into rfft-input layout and run inverse rfft.
        //   real[0] = log|H[0]|         (DC)
        //   imag[0] = log|H[N/2]|       (Nyquist, packed)
        //   real[k] = log|H[k]| imag[k] = 0 for k=1..N/2-1
        constexpr float kFloor = 1e-7f;  // -140 dB
        cep_split.realp[0] = std::log(std::max(mag[0],            kFloor));
        cep_split.imagp[0] = std::log(std::max(mag[n_freq_ - 1],  kFloor));
        for (int k = 1; k < n_fft_ / 2; ++k) {
            cep_split.realp[k] = std::log(std::max(mag[k], kFloor));
            cep_split.imagp[k] = 0.0f;
        }
        vDSP_fft_zrip(fft_setup_, &cep_split, 1, log2_nfft_, kFFTDirection_Inverse);
        // Unpack split→interleaved real cepstrum (already real-valued, but vDSP
        // returns it in the same split layout used for forward rfft outputs).
        vDSP_ztoc(&cep_split, 1,
                  reinterpret_cast<DSPComplex*>(cep_time_.data()), 2,
                  n_fft_ / 2);

        // Step 3: scale by 1/(2N) for true IDFT and apply the min-phase fold.
        const float inv_2n = 1.0f / (2.0f * static_cast<float>(n_fft_));
        cep_time_[0]            *= inv_2n * 1.0f;          // w[0] = 1
        cep_time_[n_fft_ / 2]   *= inv_2n * 1.0f;          // w[N/2] = 1
        for (int n = 1; n < n_fft_ / 2; ++n) {
            cep_time_[n]              *= inv_2n * 2.0f;    // w[1..N/2-1] = 2
            cep_time_[n_fft_ - n]      = 0.0f;             // anti-causal half → 0
        }

        // Step 4: forward rfft of c_min → log H_mp[k] (complex).
        //   pack real signal back into split form, run forward FFT.
        vDSP_ctoz(reinterpret_cast<DSPComplex*>(cep_time_.data()), 2,
                  &cep_split, 1, n_fft_ / 2);
        vDSP_fft_zrip(fft_setup_, &cep_split, 1, log2_nfft_, kFFTDirection_Forward);

        // Step 5: H_mp[k] = exp(log_re + j*log_im) per bin.
        // DC and Nyquist are real-only after a real-cepstrum min-phase build.
        h_mp_real_[0]            = std::exp(cep_split.realp[0]);
        h_mp_imag_[0]            = 0.0f;
        h_mp_real_[n_freq_ - 1]  = std::exp(cep_split.imagp[0]);  // Nyquist
        h_mp_imag_[n_freq_ - 1]  = 0.0f;
        for (int k = 1; k < n_fft_ / 2; ++k) {
            const float lr = cep_split.realp[k];
            const float li = cep_split.imagp[k];
            const float em = std::exp(lr);
            h_mp_real_[k] = em * std::cos(li);
            h_mp_imag_[k] = em * std::sin(li);
        }
    }

    void recompute_alpha_() {
        const float blocks_per_tau =
            (static_cast<float>(cfg_.sample_rate) * speed_tau_ms_ * 0.001f)
            / static_cast<float>(cfg_.block_size);
        mask_smooth_alpha_ = std::exp(-1.0f / std::max(blocks_per_tau, 1.0f));
        speed_tau_cached_  = speed_tau_ms_;
    }

    void build_freq_smoothing_kernel_(int sr, int n_fft, float octave_frac) {
        // For each output bin k at frequency f_k, build a Gaussian kernel of
        // sigma proportional to f_k (constant fraction of an octave). Stored
        // as a ragged array via parallel start/len/woff arrays so the apply
        // loop is just three small fetches per bin.
        const int   n_freq      = n_fft / 2 + 1;
        const float bin_hz      = sr / static_cast<float>(n_fft);
        // half_octave_frac → sigma multiplier on f. e.g. octave_frac=1/6
        // means a half-power half-width of 1/12 octave: sigma_hz = f * (2^(1/12) - 1).
        const float sigma_coeff = std::pow(2.0f, 0.5f * octave_frac) - 1.0f;
        const float min_sigma   = 1.0f;  // bins; floors LF kernel to ~1 bin
        const int   max_kernel_half = 32; // hard cap so HF kernels stay sane

        freq_kernel_start_.assign(n_freq, 0);
        freq_kernel_len_.assign(n_freq, 1);
        freq_kernel_woff_.assign(n_freq, 0);
        freq_kernel_w_.clear();
        freq_kernel_w_.reserve(n_freq * 8);

        for (int k = 0; k < n_freq; ++k) {
            const float f_k        = k * bin_hz;
            float       sigma_bins = (f_k * sigma_coeff) / bin_hz;
            if (sigma_bins < min_sigma) sigma_bins = min_sigma;
            int half = static_cast<int>(std::ceil(2.0f * sigma_bins));
            if (half > max_kernel_half) half = max_kernel_half;
            int start = k - half;
            int end   = k + half;
            if (start < 0)         start = 0;
            if (end   > n_freq - 1) end   = n_freq - 1;
            const int len = end - start + 1;

            // Compute and normalize the Gaussian.
            const int   woff = static_cast<int>(freq_kernel_w_.size());
            float       sum  = 0.0f;
            for (int j = 0; j < len; ++j) {
                const int   bin = start + j;
                const float d   = static_cast<float>(bin - k) / sigma_bins;
                const float w   = std::exp(-0.5f * d * d);
                freq_kernel_w_.push_back(w);
                sum += w;
            }
            if (sum > 0.0f) {
                for (int j = 0; j < len; ++j) freq_kernel_w_[woff + j] /= sum;
            }
            freq_kernel_start_[k] = start;
            freq_kernel_len_[k]   = len;
            freq_kernel_woff_[k]  = woff;
        }
    }

    void build_mel_(int sr, int n_fft, int n_bands, float f_min, float f_max) {
        const int n_freq = n_fft / 2 + 1;
        const float mel_min = 2595.0f * std::log10(1.0f + f_min / 700.0f);
        const float mel_max = 2595.0f * std::log10(1.0f + f_max / 700.0f);
        std::vector<float> mel_pts(n_bands + 2, 0.0f);
        std::vector<float> hz_pts(n_bands + 2, 0.0f);
        std::vector<float> bin_pts(n_bands + 2, 0.0f);
        for (int i = 0; i < n_bands + 2; ++i) {
            mel_pts[i] = mel_min + (mel_max - mel_min) * i / (n_bands + 1);
            hz_pts[i]  = 700.0f * (std::pow(10.0f, mel_pts[i] / 2595.0f) - 1.0f);
            bin_pts[i] = hz_pts[i] * (n_fft / static_cast<float>(sr));
            if (bin_pts[i] < 0.0f) bin_pts[i] = 0.0f;
            if (bin_pts[i] > n_freq - 1) bin_pts[i] = n_freq - 1;
        }
        band_to_bin_.assign(n_bands * n_freq, 0.0f);
        for (int b = 0; b < n_bands; ++b) {
            const float left   = bin_pts[b];
            const float center = bin_pts[b + 1];
            const float right  = bin_pts[b + 2];
            const float l_span = std::max(center - left,  1e-6f);
            const float r_span = std::max(right  - center, 1e-6f);
            for (int k = 0; k < n_freq; ++k) {
                const float kf = static_cast<float>(k);
                const float up = (kf - left) / l_span;
                const float dn = (right - kf) / r_span;
                float w = std::min(up, dn);
                if (w < 0.0f) w = 0.0f;
                band_to_bin_[b * n_freq + k] = w;
            }
        }
        bin_norm_.assign(n_freq, 0.0f);
        for (int b = 0; b < n_bands; ++b) {
            for (int k = 0; k < n_freq; ++k) {
                bin_norm_[k] += band_to_bin_[b * n_freq + k];
            }
        }
    }

    SpectralMaskEqParams cfg_{};
    int n_fft_{0}, hop_{0}, n_bands_{0}, n_freq_{0};
    vDSP_Length log2_nfft_{0};
    FFTSetup fft_setup_{nullptr};

    std::vector<float> window_;       // Hann
    std::vector<float> in_ring_;      // n_fft circular
    int                in_fill_{0};
    int                samples_since_{0};

    std::vector<float> out_ring_;     // OLA audio accumulator (n_fft + hop)
    std::vector<float> norm_ring_;   // OLA window² accumulator (same size)
    int                out_write_{0};
    int                out_read_{0};
    int                out_avail_{0};

    std::vector<float> band_to_bin_;     // [n_bands * n_freq]
    std::vector<float> bin_norm_;        // [n_freq]
    std::vector<float> bin_gain_;        // [n_freq] linear, smoothed in time (consumed by FFT)
    std::vector<float> bin_gain_target_; // [n_freq] linear, freq-smoothed controller target
    std::vector<float> bin_db_buf_;      // [n_freq] scratch for per-bin dB before freq smoothing
    float              mask_smooth_alpha_{0.f};  // per-set_params time decay
    // Runtime knobs.
    float              range_norm_{1.0f};        // [0, 1] scale on predicted band_db
    float              boost_scale_{1.0f};       // [0, 1] asymmetric boost attenuation (1 = symmetric)
    float              speed_tau_ms_{25.0f};     // user-set time constant
    float              speed_tau_cached_{-1.f};  // last value alpha was computed at
    // Ragged 1/6-octave Gaussian kernel: per output bin k, the apply loop
    // reads len kernels starting at woff and weights them against
    // bin_db_buf_[start..start+len-1].
    std::vector<int>   freq_kernel_start_;
    std::vector<int>   freq_kernel_len_;
    std::vector<int>   freq_kernel_woff_;
    std::vector<float> freq_kernel_w_;

    // FFT scratch
    std::vector<float> windowed_;
    std::vector<float> split_real_;
    std::vector<float> split_imag_;
    std::vector<float> time_out_;

    // Minimum-phase reconstruction scratch + output complex filter.
    std::vector<float> cep_split_real_;
    std::vector<float> cep_split_imag_;
    std::vector<float> cep_time_;     // real cepstrum (length n_fft)
    std::vector<float> h_mp_real_;    // [n_freq] complex H_mp
    std::vector<float> h_mp_imag_;

    float ola_scale_{1.0f};
};

}  // namespace nablafx
