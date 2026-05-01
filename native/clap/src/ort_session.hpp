#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "meta.hpp"

namespace nablafx {

// Realtime-safe ONNX Runtime session wrapper.
//
// - single-threaded (intra = inter = 1) to avoid thread-pool stalls on the
//   audio thread
// - uses Ort::IoBinding + pre-allocated CPU tensors so Run() never allocates
// - maintains two state buffers (A/B) per state tensor and swaps between
//   calls, so input and output state never alias within a single Run()
//
// Not thread-safe. Each session is meant to be driven by one audio thread.
class OrtSession {
public:
    OrtSession(Ort::Env& env, const std::string& model_path, const PluginMeta& meta,
               int max_block_len);
    ~OrtSession();

    OrtSession(const OrtSession&) = delete;
    OrtSession& operator=(const OrtSession&) = delete;

    // Input buffer the caller fills before each run. Length in samples is
    // total = receptive_field - 1 + block_len (caller does the ring-buffer
    // prepend).
    float* audio_in_buffer() { return audio_in_.data(); }

    // Controls buffer (length == meta.num_controls). Caller writes the
    // current knob values each block. If the model is non-parametric this is
    // unused.
    float* controls_buffer() { return controls_.data(); }

    // Run one block. ``input_len`` is the total length of data written into
    // audio_in_buffer(). Returns a pointer to the output samples (length
    // input_len - (receptive_field - 1), but the caller already knows its
    // own block_len).
    const float* run(int input_len);

    // Swap A/B state buffers after every successful run(). The plugin calls
    // this explicitly rather than baking it into run() so that aborting a
    // block leaves state unchanged.
    void swap_state_buffers();

    // Zero all state and clear the ring-buffer lookback. Called when the
    // host requests a reset via clap_plugin::reset().
    void reset_state();

private:
    void allocate_io_buffers_(int max_block_len);
    void rebind_(int input_len);
    void warmup_();

    Ort::Env&                    env_;
    Ort::SessionOptions          session_opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo              cpu_memory_;
    Ort::AllocatorWithDefaultOptions allocator_;

    PluginMeta meta_;
    int        max_input_len_{};

    // I/O buffers owned by us, wrapped in Ort::Value per Run()
    std::vector<float> audio_in_;
    std::vector<float> audio_out_;
    std::vector<float> controls_;

    // A/B double-buffered state. state_a_[i] and state_b_[i] each hold the
    // flat float32 buffer for state tensor i. `state_in_ab_` tracks which is
    // the "current input" bank.
    std::vector<std::vector<float>> state_a_;
    std::vector<std::vector<float>> state_b_;
    bool state_in_is_a_{true};

    std::vector<const char*> input_name_cstrs_;
    std::vector<const char*> output_name_cstrs_;
    // String storage backing input_name_cstrs_ / output_name_cstrs_
    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
};

}  // namespace nablafx
