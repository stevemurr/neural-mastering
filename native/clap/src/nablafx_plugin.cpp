// Generic nablafx CLAP plugin.
//
// The same compiled dylib is bundled into a per-model .clap (see build.sh);
// effect-specific behavior (model_id, knobs, receptive field, sample rate) is
// read from Contents/Resources/plugin_meta.json at module load time.
//
// Realtime path is:
//   1. parameter events -> control snapshot
//   2. for each channel:
//        prepend (rf-1) history from ring buffer
//        run ORT session (pre-allocated buffers, no allocation)
//        copy output to host; update ring buffer
//        swap state A/B banks
//
// v1 limitations:
//   - arm64 macOS only
//   - CPU execution provider only (no CoreML)
//   - no knob smoothing (per-block snapshots only)
//   - refuses activation if host sample rate != meta.sample_rate

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <dlfcn.h>

#include <clap/clap.h>
#include <onnxruntime_cxx_api.h>

#include "meta.hpp"
#include "ort_session.hpp"
#include "param_id.hpp"

namespace nablafx {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Module-global state (loaded once at module init)
// ---------------------------------------------------------------------------

struct ModuleState {
    PluginMeta                 meta;
    std::string                bundle_dir;           // .../MyEffect.clap/Contents
    std::string                model_onnx_path;
    std::string                plugin_id_str;        // "com.nablafx.<model_id>"
    clap_plugin_descriptor_t   descriptor{};
    std::vector<const char*>   feature_ptrs;         // CLAP wants a null-terminated char**
    std::array<const char*, 3> feature_storage{};
    std::unique_ptr<Ort::Env>  ort_env;
};

static ModuleState* g_state = nullptr;

// Locate the .clap bundle by asking dlfcn for the path of the symbol we're
// executing — the dylib lives at .clap/Contents/MacOS/<name>, so the bundle
// is two directories up.
static std::string find_bundle_contents_() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&find_bundle_contents_), &info) == 0 || !info.dli_fname) {
        return {};
    }
    fs::path dylib = info.dli_fname;
    // .clap/Contents/MacOS/<dylib>  ->  .clap/Contents
    return dylib.parent_path().parent_path().string();
}

static void populate_descriptor_(ModuleState& st) {
    st.plugin_id_str = "com.nablafx." + st.meta.model_id;

    st.feature_storage[0] = CLAP_PLUGIN_FEATURE_AUDIO_EFFECT;
    st.feature_storage[1] = CLAP_PLUGIN_FEATURE_UTILITY;
    st.feature_storage[2] = nullptr;
    st.feature_ptrs.assign(st.feature_storage.begin(), st.feature_storage.end());

    st.descriptor.clap_version = CLAP_VERSION_INIT;
    st.descriptor.id           = st.plugin_id_str.c_str();
    st.descriptor.name         = st.meta.effect_name.c_str();
    st.descriptor.vendor       = "nablafx";
    st.descriptor.url          = "https://github.com/mcomunita/nablafx";
    st.descriptor.manual_url   = "";
    st.descriptor.support_url  = "";
    st.descriptor.version      = "1.0.0";
    st.descriptor.description  = "Neural audio effect exported from a nablafx checkpoint";
    st.descriptor.features     = st.feature_ptrs.data();
}

// ---------------------------------------------------------------------------
// Per-instance state
// ---------------------------------------------------------------------------

struct Plugin {
    clap_plugin_t plugin{};
    const clap_host_t* host{nullptr};

    const PluginMeta*        meta{nullptr};
    std::unique_ptr<OrtSession> sessions[2];   // one per channel (stereo)
    std::vector<std::vector<float>> ring_buffers; // ring_buffers[channel][i]
    std::vector<float>       control_values;   // length == num_controls
    int                      ring_len{};       // rf - 1
    int                      channels{2};
    double                   sample_rate{};
    uint32_t                 max_block_len{};
    bool                     activated{false};
};

// Copy the last `ring_len` samples of `block` (length `block_len`) into the
// ring buffer, preserving the prepended-history contract: ring_buffer always
// holds the most recent `ring_len` samples of the plugin's input stream.
static void update_ring_(std::vector<float>& ring, const float* block_in, int block_len, int ring_len) {
    if (block_len >= ring_len) {
        std::copy_n(block_in + (block_len - ring_len), ring_len, ring.begin());
    } else {
        // shift existing contents left by block_len
        std::copy(ring.begin() + block_len, ring.end(), ring.begin());
        std::copy_n(block_in, block_len, ring.end() - block_len);
    }
}

// ---------------------------------------------------------------------------
// CLAP extension: audio ports — 1 stereo input, 1 stereo output
// ---------------------------------------------------------------------------

static uint32_t audio_ports_count(const clap_plugin_t*, bool /*is_input*/) { return 1; }

static bool audio_ports_get(const clap_plugin_t*, uint32_t index, bool is_input,
                            clap_audio_port_info_t* info) {
    if (index != 0) return false;
    info->id            = is_input ? 0 : 1;
    std::snprintf(info->name, sizeof(info->name), "%s", is_input ? "in" : "out");
    info->channel_count = 2;
    info->flags         = CLAP_AUDIO_PORT_IS_MAIN;
    info->port_type     = CLAP_PORT_STEREO;
    info->in_place_pair = CLAP_INVALID_ID;
    return true;
}

static const clap_plugin_audio_ports_t s_ext_audio_ports = {
    audio_ports_count,
    audio_ports_get,
};

// ---------------------------------------------------------------------------
// CLAP extension: params
// ---------------------------------------------------------------------------

static uint32_t params_count(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    return static_cast<uint32_t>(plug->meta->controls.size());
}

static bool params_get_info(const clap_plugin_t* p, uint32_t index, clap_param_info_t* info) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    if (index >= plug->meta->controls.size()) return false;
    const auto& c = plug->meta->controls[index];
    info->id       = param_id_for(plug->meta->effect_name, c.id);
    info->flags    = CLAP_PARAM_IS_AUTOMATABLE;
    info->cookie   = nullptr;
    info->min_value = c.min;
    info->max_value = c.max;
    info->default_value = c.def;
    std::snprintf(info->name, sizeof(info->name), "%s", c.name.c_str());
    std::snprintf(info->module, sizeof(info->module), "%s", "");
    return true;
}

static bool params_get_value(const clap_plugin_t* p, clap_id id, double* value) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    for (size_t i = 0; i < plug->meta->controls.size(); ++i) {
        if (param_id_for(plug->meta->effect_name, plug->meta->controls[i].id) == id) {
            *value = plug->control_values[i];
            return true;
        }
    }
    return false;
}

static bool params_value_to_text(const clap_plugin_t*, clap_id, double value, char* out,
                                 uint32_t out_size) {
    std::snprintf(out, out_size, "%.3f", value);
    return true;
}

static bool params_text_to_value(const clap_plugin_t*, clap_id, const char* text, double* out) {
    char* end = nullptr;
    double v = std::strtod(text, &end);
    if (end == text) return false;
    *out = v;
    return true;
}

static void params_flush(const clap_plugin_t*, const clap_input_events_t*, const clap_output_events_t*) {
    // parameter changes are processed inside process() — outside the audio
    // thread, the CLAP host may call flush(), but we have no out-of-band
    // state to reconcile.
}

static const clap_plugin_params_t s_ext_params = {
    params_count,
    params_get_info,
    params_get_value,
    params_value_to_text,
    params_text_to_value,
    params_flush,
};

// ---------------------------------------------------------------------------
// CLAP extension: latency
// ---------------------------------------------------------------------------

static uint32_t latency_get(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    return static_cast<uint32_t>(plug->meta->latency_samples);
}

static const clap_plugin_latency_t s_ext_latency = { latency_get };

// ---------------------------------------------------------------------------
// CLAP plugin lifecycle
// ---------------------------------------------------------------------------

static bool plugin_init(const clap_plugin_t* /*p*/) { return true; }

static void plugin_destroy(const clap_plugin_t* p) {
    delete static_cast<Plugin*>(p->plugin_data);
}

static bool plugin_activate(const clap_plugin_t* p, double sample_rate,
                            uint32_t /*min_frames*/, uint32_t max_frames) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    if (static_cast<int>(std::lround(sample_rate)) != plug->meta->sample_rate) {
        // v1 does not resample. Refuse the activation; the host will surface
        // an error to the user.
        return false;
    }

    plug->sample_rate   = sample_rate;
    plug->max_block_len = max_frames;
    plug->ring_len      = plug->meta->receptive_field - 1;

    plug->ring_buffers.assign(plug->channels, std::vector<float>(plug->ring_len, 0.0f));
    plug->control_values.assign(plug->meta->num_controls, 0.0f);
    for (size_t i = 0; i < plug->meta->controls.size(); ++i) {
        plug->control_values[i] = plug->meta->controls[i].def;
    }

    const std::string model_path = g_state->model_onnx_path;
    for (int ch = 0; ch < plug->channels; ++ch) {
        plug->sessions[ch] = std::make_unique<OrtSession>(
            *g_state->ort_env, model_path, *plug->meta, static_cast<int>(max_frames));
    }

    plug->activated = true;
    return true;
}

static void plugin_deactivate(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    for (auto& s : plug->sessions) s.reset();
    plug->ring_buffers.clear();
    plug->activated = false;
}

static bool plugin_start_processing(const clap_plugin_t*) { return true; }
static void plugin_stop_processing(const clap_plugin_t*) {}

static void plugin_reset(const clap_plugin_t* p) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);
    for (auto& ring : plug->ring_buffers) std::fill(ring.begin(), ring.end(), 0.0f);
    for (auto& s : plug->sessions) {
        if (s) s->reset_state();
    }
}

// Pull parameter events from the input-events list and apply them to the
// plugin's control snapshot. v1 takes a per-block snapshot — no sample-accurate
// smoothing yet.
static void apply_events_(Plugin* plug, const clap_input_events_t* in_events) {
    if (!in_events) return;
    const uint32_t n = in_events->size(in_events);
    for (uint32_t i = 0; i < n; ++i) {
        const auto* hdr = in_events->get(in_events, i);
        if (!hdr) continue;
        if (hdr->space_id != CLAP_CORE_EVENT_SPACE_ID) continue;
        if (hdr->type != CLAP_EVENT_PARAM_VALUE) continue;
        const auto* pv = reinterpret_cast<const clap_event_param_value_t*>(hdr);
        for (size_t k = 0; k < plug->meta->controls.size(); ++k) {
            if (param_id_for(plug->meta->effect_name, plug->meta->controls[k].id) == pv->param_id) {
                plug->control_values[k] = static_cast<float>(pv->value);
                break;
            }
        }
    }
}

// Normalize a host-rate control value to the [0, 1] range the model was
// trained on.
static float normalize_ctl_(const ControlSpec& c, float v) {
    const float denom = (c.max - c.min);
    if (denom == 0.0f) return 0.0f;
    return (v - c.min) / denom;
}

static clap_process_status plugin_process(const clap_plugin_t* p, const clap_process_t* process) {
    auto* plug = static_cast<Plugin*>(p->plugin_data);

    apply_events_(plug, process->in_events);

    const uint32_t n_frames = process->frames_count;
    if (n_frames == 0) return CLAP_PROCESS_CONTINUE;

    if (process->audio_inputs_count == 0 || process->audio_outputs_count == 0) {
        return CLAP_PROCESS_ERROR;
    }

    const float* const* in_ch  = process->audio_inputs[0].data32;
    float* const*       out_ch = process->audio_outputs[0].data32;
    const uint32_t      in_channels  = std::min<uint32_t>(plug->channels, process->audio_inputs[0].channel_count);
    const uint32_t      out_channels = std::min<uint32_t>(plug->channels, process->audio_outputs[0].channel_count);

    for (uint32_t ch = 0; ch < in_channels && ch < out_channels; ++ch) {
        auto& session = *plug->sessions[ch];
        auto& ring    = plug->ring_buffers[ch];

        float* in_buf = session.audio_in_buffer();
        // prepend ring buffer history
        std::copy(ring.begin(), ring.end(), in_buf);
        // then the new block
        std::copy_n(in_ch[ch], n_frames, in_buf + plug->ring_len);

        // fill controls snapshot
        float* ctl_buf = session.controls_buffer();
        for (int k = 0; k < plug->meta->num_controls; ++k) {
            ctl_buf[k] = normalize_ctl_(plug->meta->controls[k], plug->control_values[k]);
        }

        const int input_len = plug->ring_len + static_cast<int>(n_frames);
        const float* out = session.run(input_len);
        std::copy_n(out, n_frames, out_ch[ch]);

        update_ring_(ring, in_ch[ch], static_cast<int>(n_frames), plug->ring_len);
        session.swap_state_buffers();
    }

    return CLAP_PROCESS_CONTINUE;
}

static const void* plugin_get_extension(const clap_plugin_t* /*p*/, const char* id) {
    if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS) == 0) return &s_ext_audio_ports;
    if (std::strcmp(id, CLAP_EXT_PARAMS)       == 0) return &s_ext_params;
    if (std::strcmp(id, CLAP_EXT_LATENCY)      == 0) return &s_ext_latency;
    return nullptr;
}

static void plugin_on_main_thread(const clap_plugin_t*) {}

// ---------------------------------------------------------------------------
// Plugin factory
// ---------------------------------------------------------------------------

static const clap_plugin_t* factory_create_plugin(const clap_plugin_factory_t* /*f*/,
                                                  const clap_host_t* host,
                                                  const char* plugin_id) {
    if (!g_state) return nullptr;
    if (std::strcmp(plugin_id, g_state->plugin_id_str.c_str()) != 0) return nullptr;

    auto* plug = new Plugin{};
    plug->host = host;
    plug->meta = &g_state->meta;
    plug->channels = 2;

    plug->plugin.desc            = &g_state->descriptor;
    plug->plugin.plugin_data     = plug;
    plug->plugin.init            = plugin_init;
    plug->plugin.destroy         = plugin_destroy;
    plug->plugin.activate        = plugin_activate;
    plug->plugin.deactivate      = plugin_deactivate;
    plug->plugin.start_processing = plugin_start_processing;
    plug->plugin.stop_processing = plugin_stop_processing;
    plug->plugin.reset           = plugin_reset;
    plug->plugin.process         = plugin_process;
    plug->plugin.get_extension   = plugin_get_extension;
    plug->plugin.on_main_thread  = plugin_on_main_thread;

    return &plug->plugin;
}

static uint32_t factory_get_plugin_count(const clap_plugin_factory_t*) { return g_state ? 1 : 0; }

static const clap_plugin_descriptor_t* factory_get_plugin_descriptor(const clap_plugin_factory_t*,
                                                                     uint32_t index) {
    if (!g_state || index != 0) return nullptr;
    return &g_state->descriptor;
}

static const clap_plugin_factory_t s_factory = {
    factory_get_plugin_count,
    factory_get_plugin_descriptor,
    factory_create_plugin,
};

// ---------------------------------------------------------------------------
// CLAP entry point
// ---------------------------------------------------------------------------

static bool entry_init(const char* /*plugin_path*/) {
    if (g_state) return true;
    try {
        auto st = std::make_unique<ModuleState>();
        st->bundle_dir = find_bundle_contents_();
        if (st->bundle_dir.empty()) return false;

        const std::string meta_path = st->bundle_dir + "/Resources/plugin_meta.json";
        st->meta = load_meta(meta_path);
        st->model_onnx_path = st->bundle_dir + "/Resources/model.onnx";

        st->ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "nablafx");
        populate_descriptor_(*st);
        g_state = st.release();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

static void entry_deinit() {
    delete g_state;
    g_state = nullptr;
}

static const void* entry_get_factory(const char* factory_id) {
    if (std::strcmp(factory_id, CLAP_PLUGIN_FACTORY_ID) == 0) return &s_factory;
    return nullptr;
}

}  // namespace nablafx

extern "C" {
CLAP_EXPORT const clap_plugin_entry_t clap_entry = {
    CLAP_VERSION_INIT,
    nablafx::entry_init,
    nablafx::entry_deinit,
    nablafx::entry_get_factory,
};
}
