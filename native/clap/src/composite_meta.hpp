// tone_meta.json — top-level meta for the composite TONE plugin.
//
// Written on the Python side by ``nablafx.export.composite``; read here on
// module load to wire the host-exposed AMT/TRM/CLS knobs to per-stage
// parameters and to locate the saturator + la2a + per-class auto-EQ bundles.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "meta.hpp"

namespace nablafx {

struct CompositeAmountSat {
    float pre_gain_db_max  = 12.0f;
    float post_gain_db_max = -12.0f;
    float wet_mix_max      = 1.0f;
};

struct CompositeAmountLa2a {
    float peak_reduction_min = 20.0f;
    float peak_reduction_max = 70.0f;
    float comp_or_limit      = 1.0f;
};

struct CompositeAmountAutoEq {
    float wet_mix_max = 1.0f;
};

struct CompositeAmountSslComp {
    float wet_mix_max = 1.0f;
};

struct CompositeLevelerCfg {
    float target_lufs = -14.0f;
};

struct CompositeCeilingCfg {
    float ceiling_dbtp = -1.0f;
    float lookahead_ms = 1.5f;
    float attack_ms    = 0.5f;
    float release_ms   = 50.0f;
};

// Multi-class auto-EQ. Each entry in ``classes`` names the sub-bundle dir
// under .clap/Contents/Resources. ``class_order`` is the canonical index
// order — the integer-valued CLS control selects classes via this index.
struct CompositeAutoEqClasses {
    std::string                                  default_class;
    std::vector<std::string>                     class_order;
    std::unordered_map<std::string, std::string> classes;  // class → bundle dir
};

struct CompositeMeta {
    int                                            schema_version{};
    std::string                                    effect_name;
    std::string                                    model_id;
    int                                            sample_rate{};
    int                                            channels{};
    // Single-instance sub-bundles (saturator, la2a). Role → directory name
    // relative to .clap/Contents/Resources/.
    std::unordered_map<std::string, std::string>   sub_bundles;
    // Multi-class auto-EQ — one bundle per instrument-class preset.
    CompositeAutoEqClasses                         auto_eq;
    // Host-exposed knobs. Stored as the same ControlSpec used by the per-stage
    // plugins so existing param-id helpers apply unchanged.
    std::vector<ControlSpec>                       controls;

    CompositeAmountSat                             amt_sat;
    CompositeAmountLa2a                            amt_la2a;
    CompositeAmountAutoEq                          amt_autoeq;
    CompositeAmountSslComp                         amt_ssl_comp;
    CompositeLevelerCfg                            leveler;
    CompositeCeilingCfg                            ceiling;
};

// Throws std::runtime_error on malformed JSON or unknown schema_version.
CompositeMeta load_composite_meta(const std::string& path);

}  // namespace nablafx
