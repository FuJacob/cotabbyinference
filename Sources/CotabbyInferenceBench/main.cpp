#include "bench_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <llama/llama.h>

namespace {

void silenced_log(ggml_log_level, const char*, void*) {}

bool parse_args(int argc, char** argv, BenchConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        if (a == "--model") {
            const char* v = next();
            if (v) cfg.model_path = v;
        } else if (a == "--scenario") {
            const char* v = next();
            if (v) cfg.scenario = v;
        } else if (a == "--num-sequences") {
            const char* v = next();
            if (v) cfg.num_sequences = std::atoi(v);
        } else if (a == "--prompt-tokens") {
            const char* v = next();
            if (v) cfg.prompt_tokens = std::atoi(v);
        } else if (a == "--sample-tokens") {
            const char* v = next();
            if (v) cfg.sample_tokens = std::atoi(v);
        } else if (a == "--gpu-layers") {
            const char* v = next();
            if (v) cfg.gpu_layers = std::atoi(v);
        } else if (a == "--batch-size") {
            const char* v = next();
            if (v) cfg.batch_size = std::atoi(v);
        } else if (a == "--ctx-size") {
            const char* v = next();
            if (v) cfg.ctx_size = std::atoi(v);
        } else if (a == "--verbose") {
            cfg.verbose = true;
        } else if (a == "--help" || a == "-h") {
            return false;
        }
    }
    return !cfg.model_path.empty() && !cfg.scenario.empty();
}

void usage() {
    std::fprintf(stderr,
        "CotabbyInferenceBench — Phase 0 spike for batched-decode evaluation\n"
        "\n"
        "Usage:\n"
        "  CotabbyInferenceBench --model PATH --scenario SCENARIO [options]\n"
        "\n"
        "Scenarios:\n"
        "  a_two_contexts     N separate llama_context instances, one thread each\n"
        "                     (current CotabbyInferenceEngine architecture)\n"
        "  b_shared_context   One shared llama_context with n_seq_max=N, batched\n"
        "                     llama_decode (candidate Phase 1 architecture)\n"
        "\n"
        "Options:\n"
        "  --num-sequences N   Number of concurrent sequences (default 2)\n"
        "  --prompt-tokens N   Synthetic prompt length per sequence (default 256)\n"
        "  --sample-tokens N   Tokens to generate per sequence (default 200)\n"
        "  --gpu-layers N      llama n_gpu_layers; -1 means all (default -1)\n"
        "  --batch-size N      Decode batch size (default 512)\n"
        "  --ctx-size N        Per-sequence KV slots (default 2048)\n"
        "  --verbose           Don't silence llama's internal logging\n"
        "\n"
        "Output: one line of JSON to stdout with elapsed_seconds,\n"
        "        total_tokens_sampled, aggregate_tokens_per_second.\n"
        "\n"
        "Both scenarios exclude prompt decode and the seed sample from the\n"
        "timed section, so the numerator counts (sample_tokens - 1) *\n"
        "num_sequences steady-state samples and is directly comparable.\n"
    );
}

} // namespace

int main(int argc, char** argv) {
    BenchConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        usage();
        return 1;
    }

    if (!cfg.verbose) {
        llama_log_set(silenced_log, nullptr);
    }
    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.gpu_layers;
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    llama_model* model = llama_model_load_from_file(
        cfg.model_path.c_str(), model_params);
    if (!model) {
        std::fprintf(stderr, "Failed to load model: %s\n", cfg.model_path.c_str());
        llama_backend_free();
        return 2;
    }

    BenchResult r;
    if (cfg.scenario == "a_two_contexts") {
        r = run_baseline_a_two_contexts(model, cfg);
    } else if (cfg.scenario == "b_shared_context") {
        r = run_baseline_b_shared_context(model, cfg);
    } else {
        std::fprintf(stderr, "Unknown scenario: %s\n", cfg.scenario.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 3;
    }

    print_result(r);

    llama_model_free(model);
    llama_backend_free();
    return r.error.empty() ? 0 : 4;
}
