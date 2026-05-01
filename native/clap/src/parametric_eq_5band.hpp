// 5-band parametric EQ: low-shelf, three peaking bands, high-shelf.
//
// Consumes the controller LSTM's per-block ONNX output (15 sigmoid values in
// [0, 1]; gain_norm + freq_norm + Q_norm per band, freq channels are unused
// because freeze_freqs=true and cutoff is baked into the meta) and runs a
// direct-form-II transposed biquad cascade.
//
// Streaming contract: the controller emits one parameter set per block_size
// samples; call ``set_params(...)`` once per block before ``process(...)``.
// Filter state survives across blocks so the cascade stays continuous even
// when coefficients change every 128 samples.
//
// Pure DSP — no CLAP / ORT / std::variant deps so it can be unit-tested
// standalone with the same recipe as test_dsp.cpp.

#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "meta.hpp"

namespace nablafx {

class ParametricEq5Band {
public:
    static constexpr std::size_t kBands = 5;

    ParametricEq5Band() = default;

    void reset(const ParametricEq5BandParams& cfg) {
        cfg_ = cfg;
        if (cfg_.bands.size() != kBands) {
            // We could relax this but the meta and tests assume 5.
            throw std::runtime_error(
                "parametric_eq_5band: meta has " +
                std::to_string(cfg_.bands.size()) + " bands, expected 5");
        }
        sample_rate_ = static_cast<double>(cfg_.sample_rate);
        for (auto& b : biquads_) {
            b = Biquad{};
        }
    }

    // Apply the latest controller output. ``params`` is the [num_control_params]
    // sigmoid vector for the current block (channels indexed by Band::ch_gain
    // and Band::ch_q in the meta).
    void set_params(const float* params, std::size_t n) {
        if (static_cast<int>(n) != cfg_.num_control_params) {
            throw std::runtime_error("set_params: expected " +
                std::to_string(cfg_.num_control_params) +
                " values, got " + std::to_string(n));
        }
        for (std::size_t i = 0; i < kBands; ++i) {
            const auto& spec = cfg_.bands[i];
            float g_norm = clamp01(params[spec.ch_gain]);
            float q_norm = clamp01(params[spec.ch_q]);
            float gain_db = g_norm * (spec.gain_db_max - spec.gain_db_min) + spec.gain_db_min;
            float q       = q_norm * (spec.q_max       - spec.q_min)       + spec.q_min;
            compute_biquad_(spec.kind, spec.cutoff_freq, gain_db, q, biquads_[i]);
        }
    }

    // In-place safe.
    void process(const float* in, float* out, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            double x = static_cast<double>(in[i]);
            for (auto& b : biquads_) {
                x = b.step(x);
            }
            out[i] = static_cast<float>(x);
        }
    }

    int block_size() const { return cfg_.block_size; }
    int num_control_params() const { return cfg_.num_control_params; }

private:
    struct Biquad {
        // Direct-form-II transposed: stable coefficient updates between blocks.
        double b0 = 1.0, b1 = 0.0, b2 = 0.0;
        double a1 = 0.0, a2 = 0.0;
        double s1 = 0.0, s2 = 0.0;
        double step(double x) {
            double y = b0 * x + s1;
            s1 = b1 * x - a1 * y + s2;
            s2 = b2 * x - a2 * y;
            return y;
        }
    };

    static float clamp01(float v) {
        return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }

    // Mirrors nablafx/processors/dsp.py:biquad — same RBJ cookbook formulas.
    void compute_biquad_(ParametricEq5BandParams::Kind kind,
                         double cutoff_freq, double gain_db, double q,
                         Biquad& bq) const {
        constexpr double kPi = 3.14159265358979323846;
        double A      = std::pow(10.0, gain_db / 40.0);
        double w0     = 2.0 * kPi * cutoff_freq / sample_rate_;
        double cos_w0 = std::cos(w0);
        double sin_w0 = std::sin(w0);
        double alpha  = sin_w0 / (2.0 * q);
        double sqrtA  = std::sqrt(A);

        double b0, b1, b2, a0, a1, a2;
        switch (kind) {
            case ParametricEq5BandParams::Kind::HighShelf:
                b0 =       A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha);
                b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0);
                b2 =       A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha);
                a0 =            (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha;
                a1 =  2.0 *    ((A - 1.0) - (A + 1.0) * cos_w0);
                a2 =            (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha;
                break;
            case ParametricEq5BandParams::Kind::LowShelf:
                b0 =       A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha);
                b1 =  2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0);
                b2 =       A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha);
                a0 =            (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha;
                a1 = -2.0 *    ((A - 1.0) + (A + 1.0) * cos_w0);
                a2 =            (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha;
                break;
            case ParametricEq5BandParams::Kind::Peaking:
            default:
                b0 = 1.0 + alpha * A;
                b1 = -2.0 * cos_w0;
                b2 = 1.0 - alpha * A;
                a0 = 1.0 + (alpha / A);
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - (alpha / A);
                break;
        }
        bq.b0 = b0 / a0; bq.b1 = b1 / a0; bq.b2 = b2 / a0;
        bq.a1 = a1 / a0; bq.a2 = a2 / a0;
        // s1, s2 deliberately preserved across coefficient updates.
    }

    ParametricEq5BandParams cfg_{};
    double                  sample_rate_ = 0.0;
    Biquad                  biquads_[kBands]{};
};

}  // namespace nablafx
