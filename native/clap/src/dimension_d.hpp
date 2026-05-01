// Stereo widener inspired by the Roland Dimension D (SDD-320).
//
// Two LFO-modulated delay lines (one per channel) run in quadrature (90°
// phase offset). A fixed cross-feed coefficient mixes a fraction of each
// channel's delayed signal into the other, creating stereo width without
// obvious chorus character.
//
// Header-only — no separate .cpp needed.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace nablafx {

class DimensionD {
public:
    // Reset all state. Call on activate() or when parameters change sample rate.
    void reset(double sample_rate, double rate_hz, double depth_norm);

    // Update rate and depth without clearing delay or LFO state.
    void set_params(double rate_hz, double depth_norm);

    // Process one stereo block. out_l/out_r must NOT alias in_l/in_r.
    void process(const float* in_l, const float* in_r,
                 float* out_l, float* out_r, std::size_t n);

private:
    // Center delay ~7 ms keeps the pitch modulation symmetric around a
    // stable-sounding point. Max depth ±4 ms gives the range of Dimension D
    // modes 1–4. Cross-feed at 0.3 matches the subtle character of mode 1/2.
    static constexpr double kCenterMs   = 7.0;
    static constexpr double kMaxDepthMs = 4.0;
    static constexpr double kCrossFeed  = 0.30;
    static constexpr std::size_t kN     = 2048;  // power-of-2 > max delay samples
    static constexpr std::size_t kMask  = kN - 1;

    struct DelayLine {
        std::array<float, kN> buf{};
        std::size_t write = 0;

        void push(float x) { buf[write++ & kMask] = x; }

        // Returns the sample pushed D iterations ago (linear interpolation).
        // Safe for D in [1, kN-1].
        float read(double D) const {
            std::size_t d0 = static_cast<std::size_t>(D);
            float       fr = static_cast<float>(D - static_cast<double>(d0));
            float s0 = buf[(write - 1 - d0)      & kMask];
            float s1 = buf[(write - 2 - d0)      & kMask];
            return s0 * (1.0f - fr) + s1 * fr;
        }

        void clear() { buf.fill(0.0f); write = 0; }
    };

    double sample_rate_    = 44100.0;
    double phase_          = 0.0;
    double phase_inc_      = 0.0;
    double center_samples_ = 0.0;
    double depth_samples_  = 0.0;

    DelayLine dl_, dr_;
};

inline void DimensionD::reset(double sample_rate, double rate_hz, double depth_norm) {
    sample_rate_    = sample_rate;
    center_samples_ = kCenterMs * sample_rate / 1000.0;
    phase_          = 0.0;
    dl_.clear();
    dr_.clear();
    set_params(rate_hz, depth_norm);
}

inline void DimensionD::set_params(double rate_hz, double depth_norm) {
    phase_inc_     = 2.0 * M_PI * std::max(rate_hz, 0.01) / sample_rate_;
    depth_samples_ = depth_norm * kMaxDepthMs * sample_rate_ / 1000.0;
}

inline void DimensionD::process(const float* in_l, const float* in_r,
                                 float* out_l, float* out_r, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        // Push current input into delay lines.
        dl_.push(in_l[i]);
        dr_.push(in_r[i]);

        // L and R use quadrature LFOs so they pitch in opposite directions.
        double delay_l = center_samples_ + depth_samples_ * std::sin(phase_);
        double delay_r = center_samples_ + depth_samples_ * std::cos(phase_);

        float dl_out = dl_.read(delay_l);
        float dr_out = dr_.read(delay_r);

        // Cross-feed: mix a fraction of the opposite channel's delayed signal.
        out_l[i] = static_cast<float>((1.0 - kCrossFeed) * dl_out + kCrossFeed * dr_out);
        out_r[i] = static_cast<float>(kCrossFeed * dl_out + (1.0 - kCrossFeed) * dr_out);

        phase_ += phase_inc_;
        if (phase_ >= 2.0 * M_PI) phase_ -= 2.0 * M_PI;
    }
}

}  // namespace nablafx
