#include "meta.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace nablafx {

using nlohmann::json;

namespace {

StageKind parse_stage_kind(const std::string& s) {
    if (s == "nn")     return StageKind::Nn;
    if (s == "dsp")    return StageKind::Dsp;
    if (s == "nn+dsp") return StageKind::NnDsp;
    throw std::runtime_error("unknown stage_kind: " + s);
}

ParametricEq5BandParams::Kind parse_band_kind(const std::string& s) {
    if (s == "low_shelf")  return ParametricEq5BandParams::Kind::LowShelf;
    if (s == "peaking")    return ParametricEq5BandParams::Kind::Peaking;
    if (s == "high_shelf") return ParametricEq5BandParams::Kind::HighShelf;
    throw std::runtime_error("unknown EQ band kind: " + s);
}

DspBlockSpec parse_dsp_block(const json& j) {
    DspBlockSpec out;
    out.kind = j.at("kind").get<std::string>();
    out.name = j.at("name").get<std::string>();
    const auto& p = j.at("params");

    if (out.kind == "rational_a") {
        if (p.value("version", "A") != "A") {
            throw std::runtime_error(
                "rational_a block " + out.name +
                ": only Rational version 'A' is supported");
        }
        RationalAParams r;
        r.numerator   = p.at("numerator").get<std::vector<float>>();
        r.denominator = p.at("denominator").get<std::vector<float>>();
        out.params = std::move(r);
    } else if (out.kind == "parametric_eq_5band") {
        ParametricEq5BandParams eq;
        eq.sample_rate        = p.at("sample_rate").get<int>();
        eq.block_size         = p.at("block_size").get<int>();
        eq.num_control_params = p.at("num_control_params").get<int>();
        for (const auto& b : p.at("bands")) {
            ParametricEq5BandParams::Band band;
            band.name        = b.at("name").get<std::string>();
            band.kind        = parse_band_kind(b.at("kind").get<std::string>());
            band.cutoff_freq = b.at("cutoff_freq").get<float>();
            const auto& g    = b.at("gain_db_range");
            band.gain_db_min = g.at(0).get<float>();
            band.gain_db_max = g.at(1).get<float>();
            const auto& q    = b.at("q_range");
            band.q_min       = q.at(0).get<float>();
            band.q_max       = q.at(1).get<float>();
            const auto& ch   = b.at("param_channels");
            band.ch_gain     = ch.at("gain").get<int>();
            band.ch_q        = ch.at("q").get<int>();
            eq.bands.push_back(std::move(band));
        }
        out.params = std::move(eq);
    } else if (out.kind == "spectral_mask_eq") {
        SpectralMaskEqParams sm;
        sm.sample_rate        = p.at("sample_rate").get<int>();
        sm.block_size         = p.at("block_size").get<int>();
        sm.num_control_params = p.at("num_control_params").get<int>();
        sm.n_fft              = p.at("n_fft").get<int>();
        sm.hop                = p.at("hop").get<int>();
        sm.n_bands            = p.at("n_bands").get<int>();
        sm.min_gain_db        = p.at("min_gain_db").get<float>();
        sm.max_gain_db        = p.at("max_gain_db").get<float>();
        sm.f_min              = p.at("f_min").get<float>();
        sm.f_max              = p.at("f_max").get<float>();
        out.params = std::move(sm);
    } else {
        throw std::runtime_error("unsupported dsp_block kind: " + out.kind);
    }
    return out;
}

}  // namespace

PluginMeta load_meta(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("failed to open " + path);
    }
    json j;
    f >> j;

    PluginMeta m;
    m.schema_version = j.value("schema_version", 0);
    if (m.schema_version != 1 && m.schema_version != 2) {
        std::ostringstream oss;
        oss << "unsupported plugin_meta schema_version " << m.schema_version
            << " (this build understands versions 1, 2)";
        throw std::runtime_error(oss.str());
    }

    m.effect_name     = j.at("effect_name").get<std::string>();
    m.model_id        = j.at("model_id").get<std::string>();
    m.architecture    = j.at("architecture").get<std::string>();
    m.sample_rate     = j.at("sample_rate").get<int>();
    m.channels        = j.at("channels").get<int>();
    m.causal          = j.at("causal").get<bool>();
    m.receptive_field = j.at("receptive_field").get<int>();
    m.latency_samples = j.at("latency_samples").get<int>();
    m.num_controls    = j.at("num_controls").get<int>();
    m.trace_len       = j.value("trace_len", 0);   // 0 for legacy bundles

    // schema_v1 had no stage_kind — every v1 bundle is a single-ONNX black-box.
    if (m.schema_version >= 2) {
        m.stage_kind = parse_stage_kind(j.value("stage_kind", "nn"));
    } else {
        m.stage_kind = StageKind::Nn;
    }

    for (const auto& c : j.at("controls")) {
        m.controls.push_back(ControlSpec{
            c.at("id").get<std::string>(),
            c.at("name").get<std::string>(),
            c.at("min").get<float>(),
            c.at("max").get<float>(),
            c.at("default").get<float>(),
            c.value("skew", 1.0f),
            c.value("unit", std::string{}),
        });
    }

    for (const auto& s : j.at("state_tensors")) {
        StateSpec spec;
        spec.name  = s.at("name").get<std::string>();
        spec.shape = s.at("shape").get<std::vector<int64_t>>();
        spec.dtype = s.at("dtype").get<std::string>();
        if (spec.dtype != "float32") {
            throw std::runtime_error("state tensor " + spec.name + ": only float32 is supported");
        }
        m.state_tensors.push_back(std::move(spec));
    }

    m.input_names  = j.at("input_names").get<std::vector<std::string>>();
    m.output_names = j.at("output_names").get<std::vector<std::string>>();

    if (m.schema_version >= 2 && j.contains("dsp_blocks")) {
        for (const auto& b : j.at("dsp_blocks")) {
            m.dsp_blocks.push_back(parse_dsp_block(b));
        }
    }

    // Sanity: stage_kind must agree with what's actually populated.
    switch (m.stage_kind) {
        case StageKind::Nn:
            if (!m.dsp_blocks.empty()) {
                throw std::runtime_error("stage_kind=nn but dsp_blocks is non-empty");
            }
            if (m.input_names.empty()) {
                throw std::runtime_error("stage_kind=nn but input_names is empty");
            }
            break;
        case StageKind::Dsp:
            if (m.dsp_blocks.empty()) {
                throw std::runtime_error("stage_kind=dsp but dsp_blocks is empty");
            }
            if (!m.input_names.empty()) {
                throw std::runtime_error("stage_kind=dsp but input_names is non-empty");
            }
            break;
        case StageKind::NnDsp:
            if (m.dsp_blocks.empty()) {
                throw std::runtime_error("stage_kind=nn+dsp but dsp_blocks is empty");
            }
            if (m.input_names.empty()) {
                throw std::runtime_error("stage_kind=nn+dsp but input_names is empty");
            }
            break;
    }
    return m;
}

}  // namespace nablafx
