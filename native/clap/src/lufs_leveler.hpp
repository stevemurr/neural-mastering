// BS.1770-4 short-term LUFS leveler.
//
// Runs K-weighting (high-shelf pre-filter + 2nd-order HPF), measures short-term
// (3 s) loudness, and rides a smoothed gain to pin the signal at a target LUFS.
// Pure DSP — deliberately has no dependency on CLAP, ORT, or libebur128 so it
// can be unit-tested standalone.
//
// Usage (mono):
//   LufsLeveler lvl;
//   lvl.reset(44100.0, /*target_lufs=*/-14.0);
//   lvl.process(audio, audio, num_samples);    // in-place ok
//
// Stereo: call twice, once per channel, but share the *same* leveler if you
// want linked gain — i.e., feed both channels via the `process_linked` helper.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace nablafx {

class LufsLeveler {
public:
    struct Config {
        double target_lufs   = -14.0;  // EBU / streaming-era default
        double short_term_s  = 3.0;    // BS.1770 short-term window
        double attack_ms     = 50.0;   // gain ride time constants
        double release_ms    = 500.0;
        double min_gain_db   = -12.0;  // clamp on how hard we push
        double max_gain_db   = +12.0;
        // Inputs below this level (mean-square across the window) are treated
        // as silence — skip the LUFS update so silent tails don't force a
        // massive boost.
        double silence_floor_dbfs = -70.0;
    };

    // Default config.
    LufsLeveler() = default;
    explicit LufsLeveler(Config cfg) : cfg_(cfg) {}

    // Call on activate(). Safe to call mid-stream to reset state.
    void reset(double sample_rate, double target_lufs);

    // Update the LUFS target without resetting filter or gain state.
    void set_target(double target_lufs) { target_lufs_ = target_lufs; }

    // Process one mono buffer; output may alias input.
    void process(const float* in, float* out, std::size_t n);

    // Stereo linked: applies the *same* gain ride derived from a BS.1770
    // channel-sum to both channels. Channels must have the same length.
    void process_linked(const float* lin, const float* rin,
                        float* lout, float* rout, std::size_t n);

    // Diagnostics.
    double last_measured_lufs() const { return last_lufs_; }
    double current_gain_db()    const;

private:
    struct Biquad {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0;
        double a1 = 0.0, a2 = 0.0;
        double z1_l = 0.0, z2_l = 0.0;
        double z1_r = 0.0, z2_r = 0.0;

        double step_l(double x) {
            double y = b0 * x + z1_l;
            z1_l = b1 * x - a1 * y + z2_l;
            z2_l = b2 * x - a2 * y;
            return y;
        }
        double step_r(double x) {
            double y = b0 * x + z1_r;
            z1_r = b1 * x - a1 * y + z2_r;
            z2_r = b2 * x - a2 * y;
            return y;
        }
        void reset() { z1_l = z2_l = z1_r = z2_r = 0.0; }
    };

    void set_k_weighting_coeffs_(double sr);

    Config cfg_{};
    double sample_rate_ = 0.0;
    double target_lufs_ = -14.0;

    // Two-stage K-weighting: pre-filter (high-shelf) then RLB (2nd-order HPF).
    // Each biquad has separate left/right state registers so a shared leveler
    // can process a linked stereo pair without cross-talk in the filter.
    Biquad pre_{};
    Biquad rlb_{};

    // Short-term mean-square ring: per-100ms sub-blocks of accumulated MS.
    // Total window = short_term_s seconds → `ring_blocks_` sub-blocks.
    std::vector<double> ms_ring_;
    std::size_t ring_blocks_ = 0;
    std::size_t ring_idx_    = 0;
    std::size_t sub_block_samples_ = 0;
    std::size_t sub_block_fill_    = 0;
    double sub_block_sum_sq_       = 0.0;
    // Running sum over the ring (kept incrementally for O(1) LUFS updates).
    double ring_sum_ms_ = 0.0;
    // How many sub-blocks of real data are in the ring (ramps up on reset).
    std::size_t ring_filled_ = 0;

    // Smoothed gain (linear). Target gain updated from LUFS every sub-block.
    double smooth_gain_lin_ = 1.0;
    double target_gain_lin_ = 1.0;
    double attack_coeff_    = 0.0;
    double release_coeff_   = 0.0;

    double last_lufs_ = -120.0;
};

}  // namespace nablafx
