#include "true_peak_ceiling.hpp"

#include <algorithm>
#include <cmath>

namespace nablafx {

namespace {

// Design a 4× zero-stuffed reconstruction lowpass via windowed sinc.
// Cutoff = Fs_in / 2 (i.e., π/4 at the 4× oversampled rate). Hann window.
// Normalized so DC gain is `oversample_factor` (standard polyphase convention:
// each of the `fac` output phases contributes 1/fac of DC; we sum them via
// the peak-max so the apparent per-phase gain is 1).
void build_half_band_fir(std::array<double, TruePeakCeiling::kFirTaps>& h) {
    constexpr std::size_t N = TruePeakCeiling::kFirTaps;
    // Cutoff at Fs_in/2 = original Nyquist. In the oversampled domain's
    // normalized units (Fs_ovs = 1 ⇒ Nyquist = 0.5), that's 0.5 / fac =
    // 0.125 for fac = 4.
    const double fc_norm = 0.5 / static_cast<double>(TruePeakCeiling::kOvsFactor);
    const double center = 0.5 * (N - 1);
    double sum = 0.0;
    for (std::size_t n = 0; n < N; ++n) {
        double k = static_cast<double>(n) - center;
        double sinc_val = (std::abs(k) < 1e-9)
            ? 2.0 * fc_norm
            : std::sin(2.0 * M_PI * fc_norm * k) / (M_PI * k);
        // Hann window.
        double w = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (N - 1)));
        h[n] = sinc_val * w;
        sum += h[n];
    }
    // Normalize to unity DC gain at the full-rate output (sum of all taps
    // after polyphase recombination = 1). Since each polyphase phase is
    // evaluated separately but we pick the max, DC gain per phase should
    // average to 1/fac; polyphase compensation in the loop multiplies by
    // fac to give unity per-phase average. We want `sum` = `fac` so that
    // average phase-sum = 1.
    double target = static_cast<double>(TruePeakCeiling::kOvsFactor);
    double scale = target / sum;
    for (double& v : h) v *= scale;
}

}  // namespace

void TruePeakCeiling::reset(double sample_rate) {
    sample_rate_ = sample_rate;
    build_half_band_fir(fir_);

    lookahead_samples_ = static_cast<std::size_t>(
        std::round(cfg_.lookahead_ms * 1e-3 * sample_rate));
    if (lookahead_samples_ == 0) lookahead_samples_ = 1;

    delay_.assign(lookahead_samples_, 0.0f);
    delay_idx_ = 0;

    fir_hist_.fill(0.0);
    fir_hist_idx_ = 0;

    ceiling_lin_ = std::pow(10.0, cfg_.ceiling_dbtp / 20.0);
    gr_lin_      = 1.0;

    auto pole = [&](double ms) {
        double tau_s = std::max(ms, 1e-3) * 1e-3;
        return std::exp(-1.0 / (tau_s * sample_rate));
    };
    attack_coeff_  = pole(cfg_.attack_ms);
    release_coeff_ = pole(cfg_.release_ms);
}

void TruePeakCeiling::process(const float* in, float* out, std::size_t n) {
    const std::size_t ph = kFirPhase;
    const std::size_t fac = kOvsFactor;

    for (std::size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(in[i]);

        // Push into FIR history.
        fir_hist_[fir_hist_idx_] = x;
        fir_hist_idx_ = (fir_hist_idx_ + 1) % ph;

        // Compute the 4 oversampled outputs for this input sample.
        // Polyphase index p = 0..3 selects every 4th tap starting at p.
        // Map each tap index (0..31) to a history slot relative to the
        // just-pushed sample so we convolve in the right order.
        double peak_mag = 0.0;
        for (std::size_t p = 0; p < fac; ++p) {
            double acc = 0.0;
            for (std::size_t k = 0; k < ph; ++k) {
                std::size_t tap = p + k * fac;  // 0..31
                // fir_hist_[(idx - 1 - k) mod ph] is the (k+1)-th most-recent input.
                std::size_t h_idx = (fir_hist_idx_ + ph - 1 - k) % ph;
                acc += fir_[tap] * fir_hist_[h_idx];
            }
            double mag = std::abs(acc);
            if (mag > peak_mag) peak_mag = mag;
        }

        // Desired gain reduction for this 4× peak sample. The lookahead is
        // baked into the audio delay line below: at iteration i we read the
        // delayed sample from `lookahead_samples_` iterations ago and apply
        // the GR derived from the *current* peak — which sits ahead of the
        // delayed audio by exactly `lookahead_samples_`. That gives attack
        // time to respond before the peak arrives.
        double target_gr = 1.0;
        if (peak_mag > ceiling_lin_) {
            target_gr = ceiling_lin_ / peak_mag;
        }

        double coeff = (target_gr < gr_lin_) ? attack_coeff_ : release_coeff_;
        gr_lin_ = coeff * gr_lin_ + (1.0 - coeff) * target_gr;

        // Pull delayed sample, apply smoothed GR, hard-clip as safety.
        float delayed = delay_[delay_idx_];
        delay_[delay_idx_] = static_cast<float>(x);
        delay_idx_ = (delay_idx_ + 1) % delay_.size();

        double y = delayed * gr_lin_;
        if (y > ceiling_lin_)  y = ceiling_lin_;
        if (y < -ceiling_lin_) y = -ceiling_lin_;

        out[i] = static_cast<float>(y);
    }
}

}  // namespace nablafx
