// True-peak (inter-sample) peak limiter.
//
// Guarantees the output never exceeds a configured ceiling (default -1 dBTP)
// when measured by a 4× oversampled peak detector. Architecture:
//   4× polyphase FIR upsample -> peak detect -> lookahead delay -> gain smooth
//   -> apply gain to delayed sample -> hard-clip safety net at the ceiling.
//
// This is a deterministic backstop for the TONE chain — the LA-2A comp and
// saturator can overshoot peak targets; this stage makes the ceiling
// non-negotiable without coloring the dynamics.
//
// Pure C++; no CLAP/ORT dependencies so it unit-tests standalone.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace nablafx {

class TruePeakCeiling {
public:
    struct Config {
        double ceiling_dbtp      = -1.0;  // hard ceiling measured 4×
        double lookahead_ms      = 1.5;   // at 44.1k ≈ 66 samples
        double attack_ms         = 0.5;   // gain-reduction time constants
        double release_ms        = 50.0;
    };

    TruePeakCeiling() = default;
    explicit TruePeakCeiling(Config cfg) : cfg_(cfg) {}

    void reset(double sample_rate);

    // In-place safe. `n` is the sample count per channel.
    void process(const float* in, float* out, std::size_t n);

    // Latency the plugin should report to the DAW.
    std::size_t latency_samples() const { return lookahead_samples_; }

    // 4× polyphase half-band FIR. Public so internal helpers can reference
    // the sizes; the coefficients themselves stay in a private member.
    static constexpr std::size_t kOvsFactor = 4;
    static constexpr std::size_t kFirTaps   = 32;
    static constexpr std::size_t kFirPhase  = kFirTaps / kOvsFactor;

private:
    std::array<double, kFirTaps> fir_{};

    Config cfg_{};
    double sample_rate_ = 0.0;

    // Delay line for lookahead (aligned to the peak estimate we compute for
    // the future sample).
    std::vector<float> delay_;
    std::size_t delay_idx_   = 0;
    std::size_t lookahead_samples_ = 0;

    // Short FIR history for the upsampler. Only input samples — the FIR has
    // a small footprint.
    std::array<double, kFirPhase> fir_hist_{};
    std::size_t fir_hist_idx_ = 0;

    // Gain-reduction smoothing.
    double gr_lin_         = 1.0;   // smoothed gain
    double attack_coeff_   = 0.0;
    double release_coeff_  = 0.0;
    double ceiling_lin_    = 1.0;   // linear ceiling

};

}  // namespace nablafx
