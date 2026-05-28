#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <cstdint>

// Phase 0 spike: compare aggregate decode tok/s between the current "separate
// llama_context per sequence" architecture and a prospective "single shared
// llama_context with n_seq_max > 1, batched llama_decode" architecture.
//
// The goal is a yes/no decision: on the M-series + Metal hardware our users
// run, does batched decode actually move the needle (>= 1.4x aggregate tok/s)
// or is the GPU command queue serializing everything regardless?
//
// Two baselines, identical warmup, identical token count. Both report total
// samples produced in the timed section and elapsed wall-clock so the caller
// can compute tok/s without us baking in a definition.

struct BenchConfig {
    std::string model_path;
    std::string scenario;
    int num_sequences = 2;
    int prompt_tokens = 256;
    int sample_tokens = 200;
    int gpu_layers = -1;
    int batch_size = 512;
    int ctx_size = 2048;
    bool verbose = false;
};

struct BenchResult {
    std::string scenario;
    int num_sequences = 0;
    int prompt_tokens = 0;
    int sample_tokens = 0;
    double elapsed_seconds = 0.0;
    int total_tokens_sampled = 0;
    double aggregate_tokens_per_second = 0.0;
    double per_sequence_tokens_per_second = 0.0;
    std::string error;
};

class Timer {
public:
    void start() { t0_ = std::chrono::steady_clock::now(); }
    double elapsed_seconds() const {
        auto dt = std::chrono::steady_clock::now() - t0_;
        return std::chrono::duration<double>(dt).count();
    }
private:
    std::chrono::steady_clock::time_point t0_;
};

// Pads a short English seed string to a target token count using the supplied
// vocab. Returning real tokens (not random IDs) keeps the sampler in a regime
// the model was trained for, so the decode cost matches what real usage hits.
std::vector<int32_t> make_synthetic_prompt(
    const struct llama_vocab* vocab,
    int target_tokens
);

// Emits a single-line JSON record to stdout. One result per invocation; the
// caller is expected to run the binary multiple times and compare lines.
void print_result(const BenchResult& r);

// Baseline A: N separate llama_context instances, each decoded from its own
// thread. This is what `CotabbyInferenceEngine` does today.
BenchResult run_baseline_a_two_contexts(
    struct llama_model* model,
    const BenchConfig& cfg
);

// Baseline B: one shared llama_context with n_seq_max = N, batched
// llama_decode calls carrying tokens for all sequences. This is the candidate
// architecture for Phase 1.
BenchResult run_baseline_b_shared_context(
    struct llama_model* model,
    const BenchConfig& cfg
);
