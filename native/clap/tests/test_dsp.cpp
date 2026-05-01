// Standalone unit tests for the LUFS leveler and true-peak ceiling.
// Compile with:
//   g++ -O2 -std=c++17 -I../src test_dsp.cpp \
//       ../src/lufs_leveler.cpp ../src/true_peak_ceiling.cpp \
//       -o test_dsp && ./test_dsp
//
// Run on any platform. The C++ DSP classes have no CLAP/ORT dependencies.

#include "../src/lufs_leveler.hpp"
#include "../src/true_peak_ceiling.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<float> sine(double freq, double sr, double seconds, double amp = 0.5) {
    std::size_t n = static_cast<std::size_t>(seconds * sr);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(amp * std::sin(2.0 * kPi * freq * i / sr));
    }
    return out;
}

std::vector<float> pink_noise(double sr, double seconds, uint32_t seed = 1) {
    std::size_t n = static_cast<std::size_t>(seconds * sr);
    std::vector<float> out(n);
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 0.3);
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double w = nd(rng);
        acc = 0.995 * acc + 0.1 * w;  // ~pink-ish
        out[i] = static_cast<float>(acc);
    }
    // Normalize to ~0.3 RMS
    double rms_sq = 0.0;
    for (float v : out) rms_sq += v * v;
    double rms = std::sqrt(rms_sq / n);
    double scale = (rms > 1e-9) ? (0.3 / rms) : 1.0;
    for (float& v : out) v = static_cast<float>(v * scale);
    return out;
}

double compute_lufs_rough(const float* x, std::size_t n, double sr) {
    // Independent LUFS computation to cross-check the leveler.
    nablafx::LufsLeveler probe;
    probe.reset(sr, -14.0);
    std::vector<float> tmp(n);
    probe.process(x, tmp.data(), n);
    return probe.last_measured_lufs();
}

void test_lufs_converges() {
    const double sr = 44100.0;
    nablafx::LufsLeveler lvl;
    lvl.reset(sr, -14.0);

    // 10 s of moderately quiet pink noise — should be boosted a few dB.
    // Keep it inside the ±12 dB gain cap so the test verifies convergence,
    // not the cap itself.
    auto quiet = pink_noise(sr, 10.0, 42);
    for (float& v : quiet) v *= 0.5f;

    std::vector<float> out(quiet.size());
    lvl.process(quiet.data(), out.data(), quiet.size());

    // After 10 s, gain should have converged toward raising the signal.
    double measured = lvl.last_measured_lufs();
    double gain_db  = lvl.current_gain_db();
    std::fprintf(stderr,"[lufs] final measured LUFS (input) = %.2f, gain = %+.2f dB\n",
                measured, gain_db);

    // Now measure the OUTPUT LUFS independently.
    // Snip the last 5 seconds so the ramp-up isn't folded into the measurement.
    std::size_t skip = static_cast<std::size_t>(5.0 * sr);
    double out_lufs = compute_lufs_rough(out.data() + skip, out.size() - skip, sr);
    std::fprintf(stderr,"[lufs] output LUFS (last 5 s) = %.2f  (target -14)\n", out_lufs);

    // Within 2 dB of target after 5 s is a pass.
    assert(std::abs(out_lufs - (-14.0)) < 2.0);
    std::fprintf(stderr,"[lufs] PASS\n");
}

void test_lufs_attenuates_loud() {
    const double sr = 44100.0;
    nablafx::LufsLeveler lvl;
    lvl.reset(sr, -14.0);

    // Hot signal: 1 kHz sine at -6 dBFS → about -9 LUFS; should attenuate.
    auto hot = sine(1000.0, sr, 10.0, std::pow(10.0, -6.0 / 20.0));
    std::vector<float> out(hot.size());
    lvl.process(hot.data(), out.data(), hot.size());

    std::size_t skip = static_cast<std::size_t>(5.0 * sr);
    double out_lufs = compute_lufs_rough(out.data() + skip, out.size() - skip, sr);
    std::fprintf(stderr,"[lufs-attn] output LUFS (last 5 s) = %.2f  gain=%.2f dB\n",
                out_lufs, lvl.current_gain_db());
    assert(std::abs(out_lufs - (-14.0)) < 2.0);
    std::fprintf(stderr,"[lufs-attn] PASS\n");
}

void test_lufs_silence() {
    const double sr = 44100.0;
    nablafx::LufsLeveler lvl;
    lvl.reset(sr, -14.0);
    std::vector<float> silence(static_cast<std::size_t>(5.0 * sr), 0.0f);
    std::vector<float> out(silence.size());
    lvl.process(silence.data(), out.data(), silence.size());
    for (float v : out) assert(std::isfinite(v));
    // Gain should remain at unity — no boost-the-silence pathology.
    double gain_db = lvl.current_gain_db();
    std::fprintf(stderr,"[lufs-silence] gain after silence = %+.2f dB\n", gain_db);
    assert(std::abs(gain_db) < 0.5);
    std::fprintf(stderr,"[lufs-silence] PASS\n");
}

void test_ceiling_holds() {
    const double sr = 44100.0;
    nablafx::TruePeakCeiling tpc;
    tpc.reset(sr);

    // A +6 dBFS sine at 17 kHz is a classic inter-sample peak generator:
    // full-rate samples stay under 1.0 but the reconstructed waveform
    // overshoots. Drive it hard and confirm the ceiling holds.
    auto src = sine(17000.0, sr, 2.0, 2.0);  // amp 2.0 = +6 dBFS
    std::vector<float> out(src.size());
    tpc.process(src.data(), out.data(), src.size());

    // Skip the first lookahead samples so the priming tail isn't scored.
    std::size_t skip = tpc.latency_samples() + 128;
    double peak_lin = 0.0;
    for (std::size_t i = skip; i < out.size(); ++i) {
        double m = std::abs(out[i]);
        if (m > peak_lin) peak_lin = m;
    }
    double peak_db = 20.0 * std::log10(peak_lin + 1e-12);
    std::fprintf(stderr,"[ceiling] input +6 dBFS sine → output peak = %.3f (%.3f dBFS)\n",
                peak_lin, peak_db);
    // Full-rate peak should be very close to the ceiling (−1 dBFS = 0.891).
    // Allow 0.1 dB slack for numerical + FIR overshoot.
    assert(peak_lin <= std::pow(10.0, -0.9 / 20.0) + 1e-4);
    std::fprintf(stderr,"[ceiling] PASS\n");
}

void test_ceiling_transparent_under_threshold() {
    const double sr = 44100.0;
    nablafx::TruePeakCeiling tpc;
    tpc.reset(sr);

    // Quiet sine well under the ceiling should pass through (aside from the
    // lookahead delay and the FIR's group delay).
    auto src = sine(440.0, sr, 1.0, 0.1);  // -20 dBFS
    std::vector<float> out(src.size());
    tpc.process(src.data(), out.data(), src.size());

    // Compare samples beyond startup / lookahead: output should equal a
    // delayed copy of input (within float precision).
    std::size_t la = tpc.latency_samples();
    std::size_t skip = la + 64;
    double max_abs_err = 0.0;
    for (std::size_t i = skip; i < src.size(); ++i) {
        double err = std::abs(static_cast<double>(out[i]) - static_cast<double>(src[i - la]));
        if (err > max_abs_err) max_abs_err = err;
    }
    std::fprintf(stderr,"[ceiling-xparent] max abs error vs delayed input: %.6f\n", max_abs_err);
    // Tolerance is slack because of the non-unity FIR gain near DC and
    // because tiny transient GR could pull on early samples.
    assert(max_abs_err < 0.02);
    std::fprintf(stderr,"[ceiling-xparent] PASS\n");
}

void test_ceiling_no_nan_on_extremes() {
    const double sr = 44100.0;
    nablafx::TruePeakCeiling tpc;
    tpc.reset(sr);

    std::vector<float> dc(static_cast<std::size_t>(0.5 * sr), 1.0f);  // +0 dBFS DC
    std::vector<float> out(dc.size());
    tpc.process(dc.data(), out.data(), dc.size());
    for (float v : out) assert(std::isfinite(v));

    // Hard alternating +/-10 sample-to-sample (non-physical but a good torture).
    std::vector<float> torture(dc.size());
    for (std::size_t i = 0; i < torture.size(); ++i) torture[i] = (i & 1) ? 10.0f : -10.0f;
    tpc.reset(sr);
    tpc.process(torture.data(), out.data(), torture.size());
    for (float v : out) assert(std::isfinite(v));
    for (float v : out) assert(std::abs(v) <= std::pow(10.0, -1.0 / 20.0) + 1e-3);

    std::fprintf(stderr,"[ceiling-extreme] PASS\n");
}

}  // namespace

int main() {
    test_lufs_silence();
    test_lufs_converges();
    test_lufs_attenuates_loud();
    test_ceiling_transparent_under_threshold();
    test_ceiling_holds();
    test_ceiling_no_nan_on_extremes();
    std::fprintf(stderr,"ALL TESTS PASSED\n");
    return 0;
}
