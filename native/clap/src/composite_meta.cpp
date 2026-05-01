#include "composite_meta.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace nablafx {

using nlohmann::json;

CompositeMeta load_composite_meta(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("failed to open " + path);
    }
    json j;
    f >> j;

    CompositeMeta m;
    m.schema_version = j.value("schema_version", 0);
    if (m.schema_version != 2) {
        std::ostringstream oss;
        oss << "unsupported tone_meta schema_version " << m.schema_version
            << " (this build understands version 2)";
        throw std::runtime_error(oss.str());
    }

    m.effect_name = j.at("effect_name").get<std::string>();
    m.model_id    = j.at("model_id").get<std::string>();
    m.sample_rate = j.at("sample_rate").get<int>();
    m.channels    = j.at("channels").get<int>();

    for (const auto& [role, dir] : j.at("sub_bundles").items()) {
        m.sub_bundles[role] = dir.get<std::string>();
    }

    // Multi-class auto-EQ.
    {
        const auto& aeq = j.at("auto_eq");
        m.auto_eq.default_class = aeq.at("default_class").get<std::string>();
        for (const auto& c : aeq.at("class_order")) {
            m.auto_eq.class_order.push_back(c.get<std::string>());
        }
        for (const auto& [cls, dir] : aeq.at("classes").items()) {
            m.auto_eq.classes[cls] = dir.get<std::string>();
        }
        if (m.auto_eq.class_order.empty()) {
            throw std::runtime_error("tone_meta.auto_eq.class_order is empty");
        }
        if (m.auto_eq.classes.find(m.auto_eq.default_class) == m.auto_eq.classes.end()) {
            throw std::runtime_error(
                "tone_meta.auto_eq.default_class '" + m.auto_eq.default_class
                + "' not present in classes map");
        }
        for (const auto& cls : m.auto_eq.class_order) {
            if (m.auto_eq.classes.find(cls) == m.auto_eq.classes.end()) {
                throw std::runtime_error(
                    "tone_meta.auto_eq.class_order entry '" + cls
                    + "' not present in classes map");
            }
        }
    }

    for (const auto& [id, c] : j.at("controls").items()) {
        m.controls.push_back(ControlSpec{
            c.value("id", id),
            c.at("name").get<std::string>(),
            c.at("min").get<float>(),
            c.at("max").get<float>(),
            c.at("default").get<float>(),
            c.value("skew", 1.0f),
            c.value("unit", std::string{}),
        });
    }

    const auto& am = j.at("amount_mapping");
    const auto& sat = am.at("saturator");
    m.amt_sat.pre_gain_db_max  = sat.at("pre_gain_db_max").get<float>();
    m.amt_sat.post_gain_db_max = sat.at("post_gain_db_max").get<float>();
    m.amt_sat.wet_mix_max      = sat.at("wet_mix_max").get<float>();

    const auto& la2a = am.at("la2a");
    m.amt_la2a.peak_reduction_min = la2a.at("peak_reduction_min").get<float>();
    m.amt_la2a.peak_reduction_max = la2a.at("peak_reduction_max").get<float>();
    m.amt_la2a.comp_or_limit      = la2a.value("comp_or_limit", 1.0f);

    const auto& aeq = am.at("auto_eq");
    m.amt_autoeq.wet_mix_max = aeq.at("wet_mix_max").get<float>();

    // ssl_comp section is optional — older bundles don't include it.
    if (am.contains("ssl_comp")) {
        const auto& sslc = am.at("ssl_comp");
        m.amt_ssl_comp.wet_mix_max = sslc.value("wet_mix_max", 1.0f);
    }

    const auto& lev = j.at("leveler");
    m.leveler.target_lufs = lev.at("target_lufs").get<float>();

    const auto& ceil_j = j.at("ceiling");
    m.ceiling.ceiling_dbtp = ceil_j.at("ceiling_dbtp").get<float>();
    m.ceiling.lookahead_ms = ceil_j.value("lookahead_ms", 1.5f);
    m.ceiling.attack_ms    = ceil_j.value("attack_ms",    0.5f);
    m.ceiling.release_ms   = ceil_j.value("release_ms",   50.0f);

    return m;
}

}  // namespace nablafx
