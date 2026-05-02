// Standalone unit tests for the grey-box DSP runtime blocks.
//
// Build (Linux/macOS):
//   g++ -O2 -std=c++17 -I../src test_grey_dsp.cpp ../src/meta.cpp \
//       -I<path-to-nlohmann-json/include> -o test_grey_dsp && ./test_grey_dsp
//
// Both DSP classes are header-only and have no CLAP/ORT dependencies, so this
// runs anywhere the meta loader can.

#include "../src/meta.hpp"
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
        assert(m.dsp_blocks[0].kind == "spectral_mask_eq");
        const auto& sm = std::get<nablafx::SpectralMaskEqParams>(m.dsp_blocks[0].params);
        assert(sm.n_bands > 0);
        assert(sm.n_fft >= sm.hop * 2);
        assert(sm.num_control_params == sm.n_bands);
        assert(sm.block_size == 128);
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
    test_meta_schema_v2_roundtrip();
    std::fprintf(stderr, "ALL GREY-DSP TESTS PASSED\n");
    return 0;
}
