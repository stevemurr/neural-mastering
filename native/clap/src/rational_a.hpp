// Static Rational nonlinearity, version "A".
//
// Mirrors rational.torch.Rational_PYTORCH_A_F:
//   P(x) = a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
//   Q(x) = 1 + |b_1*x| + |b_2*x^2| + ... + |b_m*x^m|
//   y    = P(x) / Q(x)
//
// Pure DSP — no CLAP / ORT / std::variant deps so it can be unit-tested
// standalone with the same recipe as test_dsp.cpp.

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace nablafx {

class RationalA {
public:
    RationalA() = default;

    // numerator: length n+1 (a_0 .. a_n)
    // denominator: length m   (b_1 .. b_m)
    void reset(const std::vector<float>& numerator,
               const std::vector<float>& denominator) {
        num_ = numerator;
        den_ = denominator;
    }

    // Apply the rational nonlinearity sample-wise. In-place safe.
    void process(const float* in, float* out, std::size_t n) const {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = static_cast<float>(eval(static_cast<double>(in[i])));
        }
    }

    double eval(double x) const {
        // Numerator: Horner's method, starting from highest-degree coeff.
        double p = 0.0;
        if (!num_.empty()) {
            p = num_.back();
            for (std::size_t i = num_.size() - 1; i-- > 0;) {
                p = p * x + num_[i];
            }
        }
        // Denominator: 1 + sum_j |b_j * x^j| for j = 1..m
        double q = 1.0;
        double xj = 1.0;
        for (float b : den_) {
            xj *= x;
            q += std::fabs(static_cast<double>(b) * xj);
        }
        return p / q;
    }

    bool empty() const { return num_.empty() && den_.empty(); }

private:
    std::vector<float> num_;
    std::vector<float> den_;
};

}  // namespace nablafx
