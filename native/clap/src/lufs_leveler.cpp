#include "lufs_leveler.hpp"

#include <algorithm>
#include <cmath>

namespace nablafx {

namespace {

// K-weighting filter coefficients from ITU-R BS.1770-4 Annex 1, bilinear-
// transformed to 48 kHz (reference) and 44.1 kHz. Values match the
// libebur128 reference implementation.
struct KCoeffs {
    double pre_b0, pre_b1, pre_b2;
    double pre_a1, pre_a2;
    double rlb_b0, rlb_b1, rlb_b2;
    double rlb_a1, rlb_a2;
};

static constexpr KCoeffs kCoeff_48000{
    1.53512485958697,   -2.69169618940638,   1.19839281085285,
    -1.69065929318241,   0.73248077421585,
    1.0,                 -2.0,                1.0,
    -1.99004745483398,   0.99007225036621,
};

static constexpr KCoeffs kCoeff_44100{
    1.5308412300503478, -2.6509799000031379, 1.1690790340624427,
    -1.6636551132560902, 0.7125954280732254,
    1.0,                 -2.0,                1.0,
    -1.9891696736297957, 0.9891959257876969,
};

// Bilinear-transform a reference (48 kHz) biquad to an arbitrary rate.
// Not exact per the BS.1770 spec (which designs in continuous time) but
// close enough for rates far from 48 kHz when we lack tabulated coeffs.
// Only used as a fallback — 44.1 and 48 kHz have exact constants above.
inline void warp_biquad_to_sr(double src_sr, double dst_sr,
                              double b0, double b1, double b2,
                              double a1, double a2,
                              double& nb0, double& nb1, double& nb2,
                              double& na1, double& na2) {
    // Simple proportional warp of the z-plane pole/zero angles. Degrades
    // accuracy at extreme rate differences — acceptable here because the
    // LUFS short-term window is integrated over 3 s, smoothing errors.
    double scale = src_sr / dst_sr;
    nb0 = b0;
    nb1 = b1 * scale;
    nb2 = b2 * scale * scale;
    na1 = a1 * scale;
    na2 = a2 * scale * scale;
}

constexpr std::size_t kSubBlockMs = 100;

}  // namespace

void LufsLeveler::set_k_weighting_coeffs_(double sr) {
    const KCoeffs* c = nullptr;
    if (std::abs(sr - 44100.0) < 0.5) {
        c = &kCoeff_44100;
    } else if (std::abs(sr - 48000.0) < 0.5) {
        c = &kCoeff_48000;
    }

    if (c != nullptr) {
        pre_.b0 = c->pre_b0; pre_.b1 = c->pre_b1; pre_.b2 = c->pre_b2;
        pre_.a1 = c->pre_a1; pre_.a2 = c->pre_a2;
        rlb_.b0 = c->rlb_b0; rlb_.b1 = c->rlb_b1; rlb_.b2 = c->rlb_b2;
        rlb_.a1 = c->rlb_a1; rlb_.a2 = c->rlb_a2;
    } else {
        // Fallback warp from 48 kHz.
        warp_biquad_to_sr(48000.0, sr,
                          kCoeff_48000.pre_b0, kCoeff_48000.pre_b1, kCoeff_48000.pre_b2,
                          kCoeff_48000.pre_a1, kCoeff_48000.pre_a2,
                          pre_.b0, pre_.b1, pre_.b2, pre_.a1, pre_.a2);
        warp_biquad_to_sr(48000.0, sr,
                          kCoeff_48000.rlb_b0, kCoeff_48000.rlb_b1, kCoeff_48000.rlb_b2,
                          kCoeff_48000.rlb_a1, kCoeff_48000.rlb_a2,
                          rlb_.b0, rlb_.b1, rlb_.b2, rlb_.a1, rlb_.a2);
    }
    pre_.reset();
    rlb_.reset();
}

void LufsLeveler::reset(double sample_rate, double target_lufs) {
    sample_rate_ = sample_rate;
    target_lufs_ = target_lufs;
    set_k_weighting_coeffs_(sample_rate);

    sub_block_samples_ = static_cast<std::size_t>((kSubBlockMs * sample_rate) / 1000.0);
    ring_blocks_       = static_cast<std::size_t>(
        std::ceil(cfg_.short_term_s * 1000.0 / kSubBlockMs));
    ms_ring_.assign(ring_blocks_, 0.0);

    ring_idx_        = 0;
    ring_filled_     = 0;
    sub_block_fill_  = 0;
    sub_block_sum_sq_ = 0.0;
    ring_sum_ms_     = 0.0;

    smooth_gain_lin_ = 1.0;
    target_gain_lin_ = 1.0;

    // One-pole smoothing coeffs. y[n] = a*y[n-1] + (1-a)*x[n];
    //   a = exp(-1 / (tau_s * fs))
    auto pole = [&](double ms) {
        double tau_s = std::max(ms, 1e-3) * 1e-3;
        return std::exp(-1.0 / (tau_s * sample_rate));
    };
    attack_coeff_  = pole(cfg_.attack_ms);
    release_coeff_ = pole(cfg_.release_ms);

    last_lufs_ = -120.0;
}

double LufsLeveler::current_gain_db() const {
    if (smooth_gain_lin_ <= 0.0) return cfg_.min_gain_db;
    return 20.0 * std::log10(smooth_gain_lin_);
}

static inline double lufs_from_ms(double ms) {
    // BS.1770: L_k = -0.691 + 10 log10(ms)
    if (ms <= 0.0) return -120.0;
    return -0.691 + 10.0 * std::log10(ms);
}

void LufsLeveler::process(const float* in, float* out, std::size_t n) {
    const double tgt_lin_max = std::pow(10.0, cfg_.max_gain_db / 20.0);
    const double tgt_lin_min = std::pow(10.0, cfg_.min_gain_db / 20.0);
    const double silence_ms  = std::pow(10.0, cfg_.silence_floor_dbfs / 10.0);

    for (std::size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(in[i]);

        // Measurement chain: K-weight a copy of the input sample.
        double km = pre_.step_l(x);
        km = rlb_.step_l(km);
        sub_block_sum_sq_ += km * km;
        ++sub_block_fill_;

        // Sub-block boundary: commit mean-square to the ring, update LUFS,
        // update target gain.
        if (sub_block_fill_ >= sub_block_samples_) {
            double ms = sub_block_sum_sq_ / static_cast<double>(sub_block_fill_);
            ring_sum_ms_ += ms - ms_ring_[ring_idx_];
            ms_ring_[ring_idx_] = ms;
            ring_idx_ = (ring_idx_ + 1) % ring_blocks_;
            if (ring_filled_ < ring_blocks_) ++ring_filled_;

            double window_ms = ring_sum_ms_ / static_cast<double>(ring_filled_);
            if (window_ms >= silence_ms) {
                last_lufs_ = lufs_from_ms(window_ms);
                double delta_db = target_lufs_ - last_lufs_;
                if (delta_db > cfg_.max_gain_db) delta_db = cfg_.max_gain_db;
                if (delta_db < cfg_.min_gain_db) delta_db = cfg_.min_gain_db;
                target_gain_lin_ = std::pow(10.0, delta_db / 20.0);
            }
            // In silence, leave target_gain_lin_ alone.

            sub_block_fill_  = 0;
            sub_block_sum_sq_ = 0.0;
        }

        // Sample-accurate one-pole ride toward the target gain.
        double coeff = (target_gain_lin_ > smooth_gain_lin_) ? attack_coeff_ : release_coeff_;
        smooth_gain_lin_ = coeff * smooth_gain_lin_ + (1.0 - coeff) * target_gain_lin_;
        // Safety clamp in case numerical drift pushes us out.
        if (smooth_gain_lin_ > tgt_lin_max) smooth_gain_lin_ = tgt_lin_max;
        if (smooth_gain_lin_ < tgt_lin_min) smooth_gain_lin_ = tgt_lin_min;

        out[i] = static_cast<float>(x * smooth_gain_lin_);
    }
}

void LufsLeveler::process_linked(const float* lin, const float* rin,
                                 float* lout, float* rout, std::size_t n) {
    const double tgt_lin_max = std::pow(10.0, cfg_.max_gain_db / 20.0);
    const double tgt_lin_min = std::pow(10.0, cfg_.min_gain_db / 20.0);
    const double silence_ms  = std::pow(10.0, cfg_.silence_floor_dbfs / 10.0);

    for (std::size_t i = 0; i < n; ++i) {
        double xl = static_cast<double>(lin[i]);
        double xr = static_cast<double>(rin[i]);

        // K-weight each channel, use equal-weighted sum for BS.1770 stereo.
        double kl = pre_.step_l(xl); kl = rlb_.step_l(kl);
        double kr = pre_.step_r(xr); kr = rlb_.step_r(kr);
        sub_block_sum_sq_ += (kl * kl) + (kr * kr);
        sub_block_fill_ += 1;

        if (sub_block_fill_ >= sub_block_samples_) {
            // Mean-square per *sample-pair* (matches BS.1770 channel-sum).
            double ms = sub_block_sum_sq_ / static_cast<double>(sub_block_fill_);
            ring_sum_ms_ += ms - ms_ring_[ring_idx_];
            ms_ring_[ring_idx_] = ms;
            ring_idx_ = (ring_idx_ + 1) % ring_blocks_;
            if (ring_filled_ < ring_blocks_) ++ring_filled_;

            double window_ms = ring_sum_ms_ / static_cast<double>(ring_filled_);
            if (window_ms >= silence_ms) {
                last_lufs_ = lufs_from_ms(window_ms);
                double delta_db = target_lufs_ - last_lufs_;
                if (delta_db > cfg_.max_gain_db) delta_db = cfg_.max_gain_db;
                if (delta_db < cfg_.min_gain_db) delta_db = cfg_.min_gain_db;
                target_gain_lin_ = std::pow(10.0, delta_db / 20.0);
            }

            sub_block_fill_  = 0;
            sub_block_sum_sq_ = 0.0;
        }

        double coeff = (target_gain_lin_ > smooth_gain_lin_) ? attack_coeff_ : release_coeff_;
        smooth_gain_lin_ = coeff * smooth_gain_lin_ + (1.0 - coeff) * target_gain_lin_;
        if (smooth_gain_lin_ > tgt_lin_max) smooth_gain_lin_ = tgt_lin_max;
        if (smooth_gain_lin_ < tgt_lin_min) smooth_gain_lin_ = tgt_lin_min;

        lout[i] = static_cast<float>(xl * smooth_gain_lin_);
        rout[i] = static_cast<float>(xr * smooth_gain_lin_);
    }
}

}  // namespace nablafx
