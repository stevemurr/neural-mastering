// Headless CLAP host for benchmarking the NeuralMastering composite plugin.
//
// Loads a .clap bundle, drives a fixed input wav through it in fixed-size
// blocks, and reports per-block timing stats (p50/p95/p99/max) plus the
// realtime factor. Designed to be runnable from a script:
//
//   tone_bench --plugin build/NeuralMastering.clap \
//              --in    bench-input.wav \
//              --out   /tmp/out.wav \
//              --buffer 256 \
//              --iters 20 \
//              --params 'EQ=1,EQR=1,SSC=1,CMP=50,CLS=4' \
//              --json
//
// Per-block timing is the meaningful number: each plug->process() call is
// the realtime deadline. We also report per-iteration totals so a wrapper
// can compute the realtime factor across the full file.

#include <clap/clap.h>
#include <sndfile.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// Argument parsing
// -----------------------------------------------------------------------------

struct Args {
    std::string plugin_path;
    std::string in_path;
    std::string out_path;
    int sample_rate = 44100;
    int buffer_size = 256;
    int iters       = 1;
    int warmup      = 1;
    int channels    = 2;
    bool json_stats = false;
    std::vector<std::pair<std::string, double>> params;
};

static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s --plugin <NeuralMastering.clap> --in <in.wav> [options]\n"
        "  --plugin <path>          path to the .clap bundle\n"
        "  --in <wav>               input audio file\n"
        "  --out <wav>              optional output file (writes last iter)\n"
        "  --sr <hz>                sample rate (default: 44100)\n"
        "  --buffer <n>             host buffer size in frames (default: 256)\n"
        "  --iters <n>              process the input N times (default: 1)\n"
        "  --warmup <n>             untimed warmup iters (default: 1)\n"
        "  --channels <1|2>         plugin channel layout (default: 2)\n"
        "  --params 'ID=v,ID=v,...' set short-id params before processing\n"
        "  --json                   emit JSON stats to stdout\n"
        "  -h, --help               this help\n",
        prog);
}

static bool parse_params_csv(const std::string& s, std::vector<std::pair<std::string, double>>& out) {
    size_t i = 0;
    while (i < s.size()) {
        size_t comma = s.find(',', i);
        std::string token = s.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
        size_t eq = token.find('=');
        if (eq == std::string::npos) {
            std::fprintf(stderr, "bad --params token (missing '='): %s\n", token.c_str());
            return false;
        }
        std::string id  = token.substr(0, eq);
        std::string val = token.substr(eq + 1);
        try {
            out.emplace_back(id, std::stod(val));
        } catch (...) {
            std::fprintf(stderr, "bad --params value for %s: %s\n", id.c_str(), val.c_str());
            return false;
        }
        if (comma == std::string::npos) break;
        i = comma + 1;
    }
    return true;
}

static bool parse_args(int argc, char** argv, Args& a) {
    auto need = [&](int i) { if (i+1 >= argc) { usage(argv[0]); std::exit(2); } };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else if (k == "--plugin")   { need(i); a.plugin_path = argv[++i]; }
        else if (k == "--in")       { need(i); a.in_path     = argv[++i]; }
        else if (k == "--out")      { need(i); a.out_path    = argv[++i]; }
        else if (k == "--sr")       { need(i); a.sample_rate = std::atoi(argv[++i]); }
        else if (k == "--buffer")   { need(i); a.buffer_size = std::atoi(argv[++i]); }
        else if (k == "--iters")    { need(i); a.iters       = std::atoi(argv[++i]); }
        else if (k == "--warmup")   { need(i); a.warmup      = std::atoi(argv[++i]); }
        else if (k == "--channels") { need(i); a.channels    = std::atoi(argv[++i]); }
        else if (k == "--json")     { a.json_stats = true; }
        else if (k == "--params")   { need(i); if (!parse_params_csv(argv[++i], a.params)) return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); usage(argv[0]); return false; }
    }
    if (a.plugin_path.empty() || a.in_path.empty()) { usage(argv[0]); return false; }
    if (a.channels != 1 && a.channels != 2) { std::fprintf(stderr, "channels must be 1 or 2\n"); return false; }
    if (a.buffer_size <= 0 || a.iters <= 0 || a.warmup < 0) { std::fprintf(stderr, "bad sizing args\n"); return false; }
    return true;
}

// -----------------------------------------------------------------------------
// FNV-1a — mirrors nablafx::param_id_for so we can map short ID → CLAP param id.
// -----------------------------------------------------------------------------

static uint32_t fnv1a_step(const std::string& s, uint32_t h) {
    constexpr uint32_t P = 0x01000193u;
    for (unsigned char c : s) { h ^= c; h *= P; }
    return h;
}
static uint32_t param_id_for(const std::string& effect, const std::string& ctrl) {
    constexpr uint32_t OFFSET = 0x811C9DC5u;
    uint32_t h = fnv1a_step(effect, OFFSET);
    h = fnv1a_step(":", h);
    h = fnv1a_step(ctrl, h);
    if (h == 0xFFFFFFFFu) h = 0xFFFFFFFEu;
    return h;
}

// -----------------------------------------------------------------------------
// Minimal CLAP host
// -----------------------------------------------------------------------------

static const void* host_get_extension(const clap_host_t*, const char*) { return nullptr; }
static void host_request_restart (const clap_host_t*) {}
static void host_request_process (const clap_host_t*) {}
static void host_request_callback(const clap_host_t*) {}

static clap_host_t make_host() {
    clap_host_t h{};
    h.clap_version    = CLAP_VERSION_INIT;
    h.host_data       = nullptr;
    h.name            = "tone_bench";
    h.vendor          = "nablafx";
    h.url             = "";
    h.version         = "0.1";
    h.get_extension   = host_get_extension;
    h.request_restart = host_request_restart;
    h.request_process = host_request_process;
    h.request_callback= host_request_callback;
    return h;
}

// -----------------------------------------------------------------------------
// Event lists — clap_input_events_t embedded as the first member so the vtable
// callbacks can reinterpret_cast the base pointer back to the wrapper.
// -----------------------------------------------------------------------------

struct ParamEventList {
    clap_input_events_t base{};
    std::vector<clap_event_param_value_t> evs;
};

static uint32_t pel_size(const clap_input_events_t* l) {
    return static_cast<uint32_t>(reinterpret_cast<const ParamEventList*>(l)->evs.size());
}
static const clap_event_header_t* pel_get(const clap_input_events_t* l, uint32_t i) {
    return &reinterpret_cast<const ParamEventList*>(l)->evs[i].header;
}
static uint32_t empty_size(const clap_input_events_t*) { return 0; }
static const clap_event_header_t* empty_get(const clap_input_events_t*, uint32_t) { return nullptr; }
static bool out_try_push(const clap_output_events_t*, const clap_event_header_t*) { return true; }

// -----------------------------------------------------------------------------
// Bundle helpers
// -----------------------------------------------------------------------------

static std::string find_clap_dylib(const std::string& bundle) {
    fs::path mac = fs::path(bundle) / "Contents" / "MacOS";
    if (!fs::is_directory(mac)) return {};
    for (auto& e : fs::directory_iterator(mac)) {
        if (e.is_regular_file()) return e.path().string();
    }
    return {};
}

// -----------------------------------------------------------------------------
// Bench
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 2;

    // 1. dlopen the bundle's executable + grab clap_entry.
    std::string dylib = find_clap_dylib(a.plugin_path);
    if (dylib.empty()) {
        std::fprintf(stderr, "no executable in %s/Contents/MacOS/\n", a.plugin_path.c_str());
        return 1;
    }
    void* handle = dlopen(dylib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto* entry = reinterpret_cast<const clap_plugin_entry_t*>(dlsym(handle, "clap_entry"));
    if (!entry) { std::fprintf(stderr, "missing clap_entry symbol\n"); return 1; }

    if (!entry->init(a.plugin_path.c_str())) {
        std::fprintf(stderr, "clap_entry->init failed\n");
        return 1;
    }
    auto* factory = static_cast<const clap_plugin_factory_t*>(
        entry->get_factory(CLAP_PLUGIN_FACTORY_ID));
    if (!factory || factory->get_plugin_count(factory) == 0) {
        std::fprintf(stderr, "no plugins in factory\n");
        return 1;
    }
    const clap_plugin_descriptor_t* desc = factory->get_plugin_descriptor(factory, 0);
    std::string effect_name = desc->name;

    clap_host_t host = make_host();
    const clap_plugin_t* plug = factory->create_plugin(factory, &host, desc->id);
    if (!plug || !plug->init(plug)) { std::fprintf(stderr, "plugin init failed\n"); return 1; }

    if (!plug->activate(plug, static_cast<double>(a.sample_rate),
                        1, static_cast<uint32_t>(a.buffer_size))) {
        std::fprintf(stderr, "plugin activate failed\n");
        return 1;
    }
    plug->start_processing(plug);

    // 2. Read input WAV.
    SF_INFO sfi{};
    SNDFILE* sf = sf_open(a.in_path.c_str(), SFM_READ, &sfi);
    if (!sf) { std::fprintf(stderr, "sndfile open: %s\n", sf_strerror(nullptr)); return 1; }
    if (sfi.samplerate != a.sample_rate) {
        std::fprintf(stderr, "wav sr=%d != requested sr=%d\n", sfi.samplerate, a.sample_rate);
        return 1;
    }
    int wav_ch = sfi.channels;
    if (wav_ch != 1 && wav_ch != 2) {
        std::fprintf(stderr, "wav channels=%d unsupported\n", wav_ch);
        return 1;
    }
    sf_count_t total = sfi.frames;
    std::vector<float> interleaved(static_cast<size_t>(total) * wav_ch);
    sf_readf_float(sf, interleaved.data(), total);
    sf_close(sf);

    // De-interleave to plug.channels planar buffers (mono input → duplicated).
    int n_ch = a.channels;
    std::vector<std::vector<float>> in_planar(n_ch, std::vector<float>(total));
    for (sf_count_t i = 0; i < total; ++i) {
        float l = interleaved[i * wav_ch];
        float r = (wav_ch == 2) ? interleaved[i * wav_ch + 1] : l;
        in_planar[0][i] = l;
        if (n_ch == 2) in_planar[1][i] = r;
    }
    std::vector<std::vector<float>> out_planar(n_ch, std::vector<float>(total));

    // 3. Build the param-set event list (delivered on block 0 of every iter).
    ParamEventList evlist;
    evlist.base.ctx  = nullptr;
    evlist.base.size = pel_size;
    evlist.base.get  = pel_get;
    for (auto& [short_id, value] : a.params) {
        clap_event_param_value_t e{};
        e.header.size     = sizeof(e);
        e.header.time     = 0;
        e.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        e.header.type     = CLAP_EVENT_PARAM_VALUE;
        e.header.flags    = 0;
        e.param_id        = param_id_for(effect_name, short_id);
        e.cookie          = nullptr;
        e.note_id         = -1;
        e.port_index      = -1;
        e.channel         = -1;
        e.key             = -1;
        e.value           = value;
        evlist.evs.push_back(e);
    }
    clap_input_events_t empty_in{};
    empty_in.ctx  = nullptr;
    empty_in.size = empty_size;
    empty_in.get  = empty_get;
    clap_output_events_t out_events{};
    out_events.ctx      = nullptr;
    out_events.try_push = out_try_push;

    // 4. Process loop. Time each plug->process() call individually and
    //    aggregate per-block stats (the meaningful realtime number) plus
    //    per-iter totals.
    std::vector<double> block_us;     // per-block wall time, microseconds
    std::vector<double> iter_seconds; // per-iter wall time, seconds
    block_us.reserve(static_cast<size_t>((total / a.buffer_size + 1) * a.iters));
    iter_seconds.reserve(a.iters);

    auto run_once = [&](bool record) {
        sf_count_t pos = 0;
        std::vector<float*> in_ptrs(n_ch), out_ptrs(n_ch);

        clap_audio_buffer_t in_buf{};
        clap_audio_buffer_t out_buf{};
        in_buf.channel_count  = static_cast<uint32_t>(n_ch);
        out_buf.channel_count = static_cast<uint32_t>(n_ch);
        in_buf.constant_mask  = 0;
        out_buf.constant_mask = 0;
        in_buf.latency        = 0;
        out_buf.latency       = 0;

        clap_event_transport_t transport{};
        transport.header.size     = sizeof(transport);
        transport.header.time     = 0;
        transport.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        transport.header.type     = CLAP_EVENT_TRANSPORT;
        transport.header.flags    = 0;

        clap_process_t pp{};
        pp.steady_time         = 0;
        pp.transport           = &transport;
        pp.audio_inputs        = &in_buf;
        pp.audio_outputs       = &out_buf;
        pp.audio_inputs_count  = 1;
        pp.audio_outputs_count = 1;
        pp.in_events           = nullptr;
        pp.out_events          = &out_events;

        auto t_iter0 = std::chrono::high_resolution_clock::now();
        bool first = true;
        while (pos < total) {
            uint32_t frames = static_cast<uint32_t>(
                std::min<sf_count_t>(a.buffer_size, total - pos));
            for (int c = 0; c < n_ch; ++c) {
                in_ptrs[c]  = in_planar[c].data()  + pos;
                out_ptrs[c] = out_planar[c].data() + pos;
            }
            in_buf.data32   = in_ptrs.data();
            out_buf.data32  = out_ptrs.data();
            pp.frames_count = frames;
            pp.in_events    = first ? &evlist.base : &empty_in;

            auto t_blk0 = std::chrono::high_resolution_clock::now();
            plug->process(plug, &pp);
            auto t_blk1 = std::chrono::high_resolution_clock::now();
            if (record) {
                double us = std::chrono::duration<double, std::micro>(t_blk1 - t_blk0).count();
                block_us.push_back(us);
            }

            pos += frames;
            pp.steady_time += frames;
            first = false;
        }
        auto t_iter1 = std::chrono::high_resolution_clock::now();
        if (record) {
            iter_seconds.push_back(std::chrono::duration<double>(t_iter1 - t_iter0).count());
        }
    };

    for (int i = 0; i < a.warmup; ++i) run_once(/*record=*/false);
    for (int i = 0; i < a.iters;  ++i) run_once(/*record=*/true);

    plug->stop_processing(plug);
    plug->deactivate(plug);
    plug->destroy(plug);
    entry->deinit();
    if (handle) dlclose(handle);

    // 5. Optional output wav (last iter's output is in out_planar).
    if (!a.out_path.empty()) {
        SF_INFO oi{};
        oi.samplerate = a.sample_rate;
        oi.channels   = n_ch;
        oi.format     = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
        SNDFILE* osf = sf_open(a.out_path.c_str(), SFM_WRITE, &oi);
        if (!osf) {
            std::fprintf(stderr, "sndfile out open: %s\n", sf_strerror(nullptr));
        } else {
            std::vector<float> outi(static_cast<size_t>(total) * n_ch);
            for (sf_count_t i = 0; i < total; ++i) {
                outi[i * n_ch + 0] = out_planar[0][i];
                if (n_ch == 2) outi[i * n_ch + 1] = out_planar[1][i];
            }
            sf_writef_float(osf, outi.data(), total);
            sf_close(osf);
        }
    }

    // 6. Stats.
    auto pct = [](std::vector<double>& v, double p) -> double {
        if (v.empty()) return 0.0;
        size_t idx = std::min<size_t>(static_cast<size_t>(p * v.size()), v.size() - 1);
        return v[idx];
    };
    std::sort(block_us.begin(), block_us.end());
    double blk_mean = block_us.empty() ? 0.0
        : std::accumulate(block_us.begin(), block_us.end(), 0.0) / block_us.size();
    double blk_p50  = pct(block_us, 0.50);
    double blk_p95  = pct(block_us, 0.95);
    double blk_p99  = pct(block_us, 0.99);
    double blk_min  = block_us.empty() ? 0.0 : block_us.front();
    double blk_max  = block_us.empty() ? 0.0 : block_us.back();

    double iter_mean = iter_seconds.empty() ? 0.0
        : std::accumulate(iter_seconds.begin(), iter_seconds.end(), 0.0) / iter_seconds.size();
    double audio_per_iter_s = static_cast<double>(total) / a.sample_rate;
    double rtf_mean = (iter_mean > 0) ? (audio_per_iter_s / iter_mean) : 0.0;

    // Block deadline at this buffer size (us).
    double deadline_us = (a.buffer_size * 1e6) / static_cast<double>(a.sample_rate);
    int    deadline_misses = 0;
    for (double us : block_us) if (us > deadline_us) ++deadline_misses;

    if (a.json_stats) {
        std::printf("{\n");
        std::printf("  \"plugin\": \"%s\",\n", effect_name.c_str());
        std::printf("  \"sample_rate\": %d,\n", a.sample_rate);
        std::printf("  \"buffer_size\": %d,\n", a.buffer_size);
        std::printf("  \"channels\": %d,\n", n_ch);
        std::printf("  \"iters\": %d,\n", a.iters);
        std::printf("  \"warmup\": %d,\n", a.warmup);
        std::printf("  \"frames_per_iter\": %lld,\n", static_cast<long long>(total));
        std::printf("  \"audio_seconds_per_iter\": %.6f,\n", audio_per_iter_s);
        std::printf("  \"per_block_us\": {\"min\": %.3f, \"p50\": %.3f, \"p95\": %.3f, \"p99\": %.3f, \"max\": %.3f, \"mean\": %.3f, \"count\": %zu},\n",
                    blk_min, blk_p50, blk_p95, blk_p99, blk_max, blk_mean, block_us.size());
        std::printf("  \"per_iter_seconds_mean\": %.6f,\n", iter_mean);
        std::printf("  \"realtime_factor_mean\": %.4f,\n", rtf_mean);
        std::printf("  \"block_deadline_us\": %.3f,\n", deadline_us);
        std::printf("  \"deadline_miss_count\": %d,\n", deadline_misses);
        std::printf("  \"deadline_miss_pct\": %.4f\n",
                    block_us.empty() ? 0.0 : 100.0 * deadline_misses / block_us.size());
        std::printf("}\n");
    } else {
        std::printf("plugin       : %s\n", effect_name.c_str());
        std::printf("sr=%d  buffer=%d  ch=%d  iters=%d  warmup=%d\n",
                    a.sample_rate, a.buffer_size, n_ch, a.iters, a.warmup);
        std::printf("audio/iter   : %.3f s  (%lld frames)\n",
                    audio_per_iter_s, static_cast<long long>(total));
        std::printf("per-block us : min=%.2f  p50=%.2f  p95=%.2f  p99=%.2f  max=%.2f  mean=%.2f  n=%zu\n",
                    blk_min, blk_p50, blk_p95, blk_p99, blk_max, blk_mean, block_us.size());
        std::printf("deadline     : %.2f us  misses=%d (%.2f%%)\n",
                    deadline_us, deadline_misses,
                    block_us.empty() ? 0.0 : 100.0 * deadline_misses / block_us.size());
        std::printf("RTF mean     : %.3fx\n", rtf_mean);
    }

    return 0;
}
