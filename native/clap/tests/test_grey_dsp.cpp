// Standalone unit tests for the grey-box DSP runtime blocks.
//
// Build (Linux/macOS):
//   g++ -O2 -std=c++17 -I../src test_grey_dsp.cpp ../src/meta.cpp \
//       -I<path-to-nlohmann-json/include> -o test_grey_dsp && ./test_grey_dsp
//
// Both DSP classes are header-only and have no CLAP/ORT dependencies, so this
// runs anywhere the meta loader can.

#include "../src/meta.hpp"
#include "../src/parametric_eq_5band.hpp"
#include "../src/rational_a.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

// Coefficients are stored as float (matches the JSON payload), so eval()
// matches the reference to float precision (~1e-7), not double precision.
constexpr double kFloatTol = 1e-6;

void test_rational_a_identity() {
    // P(x) = x, Q(x) = 1 → y = x.
    nablafx::RationalA r;
    r.reset(/*numerator=*/{0.0f, 1.0f}, /*denominator=*/{});
    for (double x : {-1.5, -0.3, 0.0, 0.3, 1.5}) {
        double y = r.eval(x);
        assert(std::fabs(y - x) < kFloatTol);
    }
    std::fprintf(stderr, "[rational_a/identity] PASS\n");
}

void test_rational_a_constant() {
    // P(x) = 0.7, Q(x) = 1 → y = 0.7 for all x (numerator length 1 = constant).
    nablafx::RationalA r;
    r.reset({0.7f}, {});
    for (double x : {-2.0, 0.0, 2.0}) {
        double y = r.eval(x);
        assert(std::fabs(y - 0.7) < kFloatTol);
    }
    std::fprintf(stderr, "[rational_a/constant] PASS\n");
}

void test_rational_a_quadratic_over_linear() {
    // P(x) = x + x^2, Q(x) = 1 + |2x| → analytic form, a few hand-computed checks.
    nablafx::RationalA r;
    r.reset({0.0f, 1.0f, 1.0f}, {2.0f});
    auto ref = [](double x) {
        double p = x + x * x;
        double q = 1.0 + std::fabs(2.0 * x);
        return p / q;
    };
    for (double x : {-1.5, -0.7, -0.1, 0.0, 0.1, 0.7, 1.5}) {
        double got = r.eval(x);
        double exp = ref(x);
        assert(std::fabs(got - exp) < kFloatTol);
    }
    std::fprintf(stderr, "[rational_a/quad_over_lin] PASS\n");
}

void test_rational_a_buffer_processing() {
    nablafx::RationalA r;
    r.reset({0.1f, 1.0f, 0.0f, -0.2f}, {0.5f, 0.3f});
    std::vector<float> in(64), out(64);
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = static_cast<float>(std::sin(0.1 * i));
    }
    r.process(in.data(), out.data(), in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        double y = r.eval(static_cast<double>(in[i]));
        assert(std::fabs(static_cast<double>(out[i]) - y) < 1e-6);
    }
    std::fprintf(stderr, "[rational_a/buffer] PASS\n");
}

// Construct a 5-band ParametricEq spec matching the auto-EQ bundle layout
// (low_shelf, 3 peakings, high_shelf) with the trained ranges.
nablafx::ParametricEq5BandParams make_5band_spec(int sample_rate = 44100) {
    using K = nablafx::ParametricEq5BandParams::Kind;
    nablafx::ParametricEq5BandParams cfg;
    cfg.sample_rate        = sample_rate;
    cfg.block_size         = 128;
    cfg.num_control_params = 15;
    cfg.bands = {
        {"low_shelf",  K::LowShelf,  1010.0f, -9.0f, 9.0f, 0.5f, 2.0f, 0,  2},
        {"band0",      K::Peaking,    110.0f, -9.0f, 9.0f, 0.5f, 2.0f, 3,  5},
        {"band1",      K::Peaking,   1100.0f, -9.0f, 9.0f, 0.5f, 2.0f, 6,  8},
        {"band2",      K::Peaking,   7000.0f, -9.0f, 9.0f, 0.5f, 2.0f, 9, 11},
        {"high_shelf", K::HighShelf,10000.0f, -9.0f, 9.0f, 0.5f, 2.0f, 12, 14},
    };
    return cfg;
}

// Set every band's gain to 0 dB by feeding sigmoid=0.5 (midpoint of ±9 dB
// range) and Q to its midpoint, irrelevant when gain is zero.
std::vector<float> midgain_params() {
    std::vector<float> p(15, 0.5f);
    return p;
}

void test_eq_unity_gain_passthrough() {
    nablafx::ParametricEq5Band eq;
    eq.reset(make_5band_spec());
    auto p = midgain_params();
    eq.set_params(p.data(), p.size());

    // Process a sine and confirm output ≈ input (cascade of zero-dB biquads
    // is mathematically identity, with a steady-state achieved after a few
    // samples through the DF2T state).
    constexpr double sr   = 44100.0;
    constexpr double freq = 1000.0;
    const std::size_t N   = 4096;
    std::vector<float> in(N), out(N);
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = static_cast<float>(0.3 * std::sin(2.0 * kPi * freq * i / sr));
    }
    eq.process(in.data(), out.data(), N);

    // After 256 samples of warmup, output should match input within float noise.
    double max_err = 0.0;
    for (std::size_t i = 256; i < N; ++i) {
        double e = std::fabs(static_cast<double>(out[i]) - static_cast<double>(in[i]));
        if (e > max_err) max_err = e;
    }
    std::fprintf(stderr, "[eq/unity] max abs err vs input = %.3e\n", max_err);
    assert(max_err < 1e-5);
    std::fprintf(stderr, "[eq/unity] PASS\n");
}

void test_eq_peaking_boost_at_band1() {
    // band1 = 1100 Hz peaking. Push its gain channel to 1.0 (sigmoid=1 → +9 dB)
    // and its Q to ~midpoint. All other bands stay at 0 dB.
    nablafx::ParametricEq5Band eq;
    eq.reset(make_5band_spec());
    auto p = midgain_params();
    p[6] = 1.0f;  // band1 gain → +9 dB
    p[8] = 0.5f;  // band1 Q → midpoint
    eq.set_params(p.data(), p.size());

    // Drive a sine at band1's center freq; output magnitude should rise ~9 dB.
    constexpr double sr   = 44100.0;
    constexpr double freq = 1100.0;
    const std::size_t N   = 16384;
    std::vector<float> in(N), out(N);
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = static_cast<float>(std::sin(2.0 * kPi * freq * i / sr));
    }
    eq.process(in.data(), out.data(), N);

    // Measure RMS over the back half (steady state).
    double in_rms = 0.0, out_rms = 0.0;
    std::size_t skip = N / 2;
    for (std::size_t i = skip; i < N; ++i) {
        in_rms  += static_cast<double>(in[i])  * in[i];
        out_rms += static_cast<double>(out[i]) * out[i];
    }
    in_rms  = std::sqrt(in_rms  / (N - skip));
    out_rms = std::sqrt(out_rms / (N - skip));
    double gain_db = 20.0 * std::log10(out_rms / in_rms);
    std::fprintf(stderr, "[eq/peak@1100] in_rms=%.4f  out_rms=%.4f  gain=%.2f dB (expect ≈ +9)\n",
                 in_rms, out_rms, gain_db);
    // Peaking biquad at center frequency should hit the gain to within ~0.2 dB.
    assert(std::fabs(gain_db - 9.0) < 0.3);
    std::fprintf(stderr, "[eq/peak@1100] PASS\n");
}

void test_eq_peaking_off_band_unaffected() {
    // Same boost on band1 (1100 Hz) but probe at 110 Hz (band0 center) — should
    // be much closer to unity since band0 is at 0 dB and the 1100-Hz peaking
    // bell rolls off fast (Q ≈ 1.25).
    nablafx::ParametricEq5Band eq;
    eq.reset(make_5band_spec());
    auto p = midgain_params();
    p[6] = 1.0f; p[8] = 0.5f;  // band1 boost
    eq.set_params(p.data(), p.size());

    constexpr double sr   = 44100.0;
    constexpr double freq = 110.0;
    const std::size_t N   = 16384;
    std::vector<float> in(N), out(N);
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = static_cast<float>(std::sin(2.0 * kPi * freq * i / sr));
    }
    eq.process(in.data(), out.data(), N);

    double in_rms = 0.0, out_rms = 0.0;
    std::size_t skip = N / 2;
    for (std::size_t i = skip; i < N; ++i) {
        in_rms  += static_cast<double>(in[i])  * in[i];
        out_rms += static_cast<double>(out[i]) * out[i];
    }
    in_rms  = std::sqrt(in_rms  / (N - skip));
    out_rms = std::sqrt(out_rms / (N - skip));
    double gain_db = 20.0 * std::log10(out_rms / in_rms);
    std::fprintf(stderr, "[eq/probe@110] gain at 110 Hz with 1100-Hz boost = %.2f dB\n", gain_db);
    // Far enough from band1 (decade away) that response is dominated by the
    // other bands at unity.
    assert(std::fabs(gain_db) < 1.0);
    std::fprintf(stderr, "[eq/probe@110] PASS\n");
}

void test_eq_state_preserved_across_block_param_updates() {
    // Two equivalent ways to process N samples:
    //   (A) one set_params call, one process call covering all N
    //   (B) one set_params call, then chunked process calls
    // Outputs must match — DF2T state must persist across chunk boundaries.
    nablafx::ParametricEq5Band eq_a, eq_b;
    eq_a.reset(make_5band_spec());
    eq_b.reset(make_5band_spec());
    auto p = midgain_params();
    p[6] = 0.8f; p[8] = 0.3f;
    eq_a.set_params(p.data(), p.size());
    eq_b.set_params(p.data(), p.size());

    constexpr std::size_t N = 1024;
    std::vector<float> in(N);
    for (std::size_t i = 0; i < N; ++i) {
        in[i] = static_cast<float>(0.4 * std::sin(2.0 * kPi * 440.0 * i / 44100.0));
    }
    std::vector<float> out_a(N), out_b(N);
    eq_a.process(in.data(), out_a.data(), N);

    constexpr std::size_t kBlock = 128;
    for (std::size_t off = 0; off < N; off += kBlock) {
        eq_b.process(in.data() + off, out_b.data() + off, kBlock);
    }

    double max_err = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double e = std::fabs(static_cast<double>(out_a[i]) - static_cast<double>(out_b[i]));
        if (e > max_err) max_err = e;
    }
    std::fprintf(stderr, "[eq/state-across-blocks] max abs err = %.3e\n", max_err);
    assert(max_err < 1e-6);
    std::fprintf(stderr, "[eq/state-across-blocks] PASS\n");
}

// Round-trip the schema-v2 bundle metas through load_meta and confirm the
// payload comes out looking right. Driven by an env var so we don't fail on
// machines that don't have the bundles handy.
void test_meta_schema_v2_roundtrip() {
    const char* root = std::getenv("NABLAFX_EXPORTS_ROOT");
    if (!root) {
        std::fprintf(stderr, "[meta/v2] SKIP (set NABLAFX_EXPORTS_ROOT to run)\n");
        return;
    }
    {
        auto m = nablafx::load_meta(std::string(root) + "/saturator/plugin_meta.json");
        assert(m.schema_version == 2);
        assert(m.stage_kind == nablafx::StageKind::Dsp);
        assert(m.architecture == "dsp");
        assert(m.dsp_blocks.size() == 1);
        assert(m.dsp_blocks[0].kind == "rational_a");
        const auto& r = std::get<nablafx::RationalAParams>(m.dsp_blocks[0].params);
        assert(r.numerator.size() == 7);
        assert(r.denominator.size() == 5);
        std::fprintf(stderr, "[meta/v2 saturator] PASS\n");
    }
    {
        auto m = nablafx::load_meta(std::string(root) + "/auto_eq/plugin_meta.json");
        assert(m.schema_version == 2);
        assert(m.stage_kind == nablafx::StageKind::NnDsp);
        assert(m.architecture == "lstm");
        assert(m.dsp_blocks.size() == 1);
        assert(m.dsp_blocks[0].kind == "parametric_eq_5band");
        const auto& eq = std::get<nablafx::ParametricEq5BandParams>(m.dsp_blocks[0].params);
        assert(eq.bands.size() == 5);
        assert(eq.bands[0].kind == nablafx::ParametricEq5BandParams::Kind::LowShelf);
        assert(eq.bands[4].kind == nablafx::ParametricEq5BandParams::Kind::HighShelf);
        assert(eq.num_control_params == 15);
        assert(eq.block_size == 128);
        std::fprintf(stderr, "[meta/v2 auto_eq] PASS\n");
    }
    {
        auto m = nablafx::load_meta(std::string(root) + "/la2a/plugin_meta.json");
        // LA-2A stayed at v1 originally but now exports as v2 nn-only;
        // accept either.
        assert(m.schema_version == 1 || m.schema_version == 2);
        assert(m.stage_kind == nablafx::StageKind::Nn);
        assert(m.dsp_blocks.empty());
        assert(!m.input_names.empty());
        std::fprintf(stderr, "[meta/v2 la2a] PASS\n");
    }
}

}  // namespace

int main() {
    test_rational_a_identity();
    test_rational_a_constant();
    test_rational_a_quadratic_over_linear();
    test_rational_a_buffer_processing();
    test_eq_unity_gain_passthrough();
    test_eq_peaking_boost_at_band1();
    test_eq_peaking_off_band_unaffected();
    test_eq_state_preserved_across_block_param_updates();
    test_meta_schema_v2_roundtrip();
    std::fprintf(stderr, "ALL GREY-DSP TESTS PASSED\n");
    return 0;
}
