#include "ort_session.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace nablafx {

namespace {

int64_t shape_product(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

}  // namespace

OrtSession::OrtSession(Ort::Env& env, const std::string& model_path, const PluginMeta& meta,
                       int max_block_len)
    : env_(env), cpu_memory_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
      meta_(meta), max_input_len_(meta.receptive_field - 1 + max_block_len) {

    session_opts_.SetIntraOpNumThreads(1);
    session_opts_.SetInterOpNumThreads(1);
    session_opts_.SetExecutionMode(ORT_SEQUENTIAL);
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_opts_.EnableMemPattern();
    session_opts_.EnableCpuMemArena();

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_opts_);

    // Cache input/output name c-strings. ORT's AllocatedStringPtr frees on
    // destruction so we copy into std::strings we own.
    const size_t n_in  = session_->GetInputCount();
    const size_t n_out = session_->GetOutputCount();
    input_name_storage_.reserve(n_in);
    output_name_storage_.reserve(n_out);
    for (size_t i = 0; i < n_in; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        input_name_storage_.emplace_back(name.get());
    }
    for (size_t i = 0; i < n_out; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        output_name_storage_.emplace_back(name.get());
    }
    for (auto& s : input_name_storage_) input_name_cstrs_.push_back(s.c_str());
    for (auto& s : output_name_storage_) output_name_cstrs_.push_back(s.c_str());

    allocate_io_buffers_(max_block_len);
    warmup_();
}

OrtSession::~OrtSession() = default;

void OrtSession::allocate_io_buffers_(int max_block_len) {
    audio_in_.assign(static_cast<size_t>(max_input_len_), 0.0f);
    audio_out_.assign(static_cast<size_t>(max_block_len), 0.0f);
    controls_.assign(static_cast<size_t>(meta_.num_controls), 0.0f);

    state_a_.resize(meta_.state_tensors.size());
    state_b_.resize(meta_.state_tensors.size());
    for (size_t i = 0; i < meta_.state_tensors.size(); ++i) {
        const auto n = static_cast<size_t>(shape_product(meta_.state_tensors[i].shape));
        state_a_[i].assign(n, 0.0f);
        state_b_[i].assign(n, 0.0f);
    }
}

void OrtSession::warmup_() {
    // Run once with zeros so ORT's kernel caches, mem patterns, and arena
    // are primed before the audio thread touches it.
    std::fill(audio_in_.begin(), audio_in_.end(), 0.0f);
    std::fill(controls_.begin(), controls_.end(), 0.0f);
    (void)run(max_input_len_);
    reset_state();  // discard warmup's effect on the state buffers
}

void OrtSession::reset_state() {
    for (auto& s : state_a_) std::fill(s.begin(), s.end(), 0.0f);
    for (auto& s : state_b_) std::fill(s.begin(), s.end(), 0.0f);
    state_in_is_a_ = true;
}

void OrtSession::swap_state_buffers() {
    state_in_is_a_ = !state_in_is_a_;
}

const float* OrtSession::run(int input_len) {
    if (input_len > max_input_len_) {
        throw std::runtime_error("OrtSession::run: input_len exceeds max configured block length");
    }

    // Compose the input tensor list in ONNX declaration order. We built
    // input_names_ from session inspection, so it already matches.
    std::vector<Ort::Value> input_values;
    input_values.reserve(input_name_cstrs_.size());

    const std::vector<int64_t> audio_in_shape = {1, 1, input_len};
    input_values.emplace_back(Ort::Value::CreateTensor<float>(
        cpu_memory_, audio_in_.data(), static_cast<size_t>(input_len),
        audio_in_shape.data(), audio_in_shape.size()));

    if (meta_.num_controls > 0) {
        const std::vector<int64_t> ctl_shape = {1, meta_.num_controls};
        input_values.emplace_back(Ort::Value::CreateTensor<float>(
            cpu_memory_, controls_.data(), static_cast<size_t>(meta_.num_controls),
            ctl_shape.data(), ctl_shape.size()));
    }

    for (size_t i = 0; i < meta_.state_tensors.size(); ++i) {
        auto& in_bank = state_in_is_a_ ? state_a_[i] : state_b_[i];
        const auto& shape = meta_.state_tensors[i].shape;
        input_values.emplace_back(Ort::Value::CreateTensor<float>(
            cpu_memory_, in_bank.data(), in_bank.size(),
            shape.data(), shape.size()));
    }

    // Pre-allocate output values likewise. Audio output length = input_len - (rf - 1).
    std::vector<Ort::Value> output_values;
    output_values.reserve(output_name_cstrs_.size());

    const int audio_out_len = input_len - (meta_.receptive_field - 1);
    const std::vector<int64_t> audio_out_shape = {1, 1, audio_out_len};
    output_values.emplace_back(Ort::Value::CreateTensor<float>(
        cpu_memory_, audio_out_.data(), static_cast<size_t>(audio_out_len),
        audio_out_shape.data(), audio_out_shape.size()));

    for (size_t i = 0; i < meta_.state_tensors.size(); ++i) {
        auto& out_bank = state_in_is_a_ ? state_b_[i] : state_a_[i];
        const auto& shape = meta_.state_tensors[i].shape;
        output_values.emplace_back(Ort::Value::CreateTensor<float>(
            cpu_memory_, out_bank.data(), out_bank.size(),
            shape.data(), shape.size()));
    }

    Ort::RunOptions run_opts;
    session_->Run(run_opts,
                  input_name_cstrs_.data(), input_values.data(), input_values.size(),
                  output_name_cstrs_.data(), output_values.data(), output_values.size());
    return audio_out_.data();
}

}  // namespace nablafx
