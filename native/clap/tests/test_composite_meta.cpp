// Standalone sanity test for the composite tone_meta.json loader. Runs on
// Linux because composite_meta.cpp has no CLAP / ORT deps.
//
// Build:
//   g++ -O2 -std=c++17 -I../src -I<json>/single_include \
//       test_composite_meta.cpp ../src/composite_meta.cpp ../src/meta.cpp \
//       -o test_composite_meta
//
// Run (point at the staged composite dir):
//   ./test_composite_meta /tmp/tone-staging/tone_meta.json

#include "../src/composite_meta.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1]
                                   : std::getenv("TONE_META_JSON");
    if (!path) {
        std::fprintf(stderr,
            "usage: test_composite_meta <tone_meta.json>  (or set TONE_META_JSON)\n");
        return 2;
    }
    auto m = nablafx::load_composite_meta(path);

    std::fprintf(stderr, "schema_version: %d\n",   m.schema_version);
    std::fprintf(stderr, "effect_name:    %s\n",   m.effect_name.c_str());
    std::fprintf(stderr, "model_id:       %s\n",   m.model_id.c_str());
    std::fprintf(stderr, "sample_rate:    %d\n",   m.sample_rate);
    std::fprintf(stderr, "channels:       %d\n",   m.channels);
    std::fprintf(stderr, "sub_bundles:\n");
    for (const auto& [k, v] : m.sub_bundles) {
        std::fprintf(stderr, "  %s -> %s\n", k.c_str(), v.c_str());
    }
    std::fprintf(stderr, "controls:\n");
    for (const auto& c : m.controls) {
        std::fprintf(stderr, "  %s (%s) [%g..%g] def=%g unit=%s\n",
                     c.id.c_str(), c.name.c_str(), c.min, c.max, c.def, c.unit.c_str());
    }
    std::fprintf(stderr, "sat:    pre=+%g dB max, post=%g dB, wet_mix_max=%g\n",
                 m.amt_sat.pre_gain_db_max, m.amt_sat.post_gain_db_max, m.amt_sat.wet_mix_max);
    std::fprintf(stderr, "la2a:   PR [%g..%g], C/L=%g\n",
                 m.amt_la2a.peak_reduction_min, m.amt_la2a.peak_reduction_max,
                 m.amt_la2a.comp_or_limit);
    std::fprintf(stderr, "autoeq: wet_mix_max=%g\n", m.amt_autoeq.wet_mix_max);
    std::fprintf(stderr, "leveler: target_lufs=%g\n", m.leveler.target_lufs);
    std::fprintf(stderr, "ceiling: %g dBTP, %g ms LA, atk=%g rel=%g\n",
                 m.ceiling.ceiling_dbtp, m.ceiling.lookahead_ms,
                 m.ceiling.attack_ms, m.ceiling.release_ms);

    assert(m.schema_version == 1);
    assert(m.effect_name == "NeuralMastering");
    assert(m.sample_rate == 44100);
    assert(m.sub_bundles.count("auto_eq") && m.sub_bundles.count("saturator")
                                           && m.sub_bundles.count("la2a"));
    assert(m.controls.size() == 2);
    assert(m.amt_sat.pre_gain_db_max  == 12.0f);
    assert(m.amt_sat.post_gain_db_max == -12.0f);
    assert(m.amt_la2a.peak_reduction_min == 20.0f);
    assert(m.amt_la2a.peak_reduction_max == 70.0f);
    assert(m.leveler.target_lufs == -14.0f);
    assert(m.ceiling.ceiling_dbtp == -1.0f);
    std::fprintf(stderr, "[composite_meta] PASS\n");
    return 0;
}
