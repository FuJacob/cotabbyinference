#include "bench_common.h"

#include <cstdio>
#include <string>

#include <llama/llama.h>

std::vector<int32_t> make_synthetic_prompt(
    const llama_vocab* vocab,
    int target_tokens
) {
    if (!vocab || target_tokens <= 0) return {};

    // Real text so the tokenizer produces a sensible distribution. The seed is
    // short; we pad until tokenization yields at least `target_tokens` tokens.
    static const char seed[] =
        "The quick brown fox jumps over the lazy dog. ";
    std::string text;
    while (static_cast<int>(text.size()) < target_tokens * 6) {
        text += seed;
    }

    bool add_bos = llama_vocab_get_add_bos(vocab);
    int capacity = target_tokens * 4 + 16;
    std::vector<int32_t> tokens(capacity);
    int n = llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(capacity),
        add_bos,
        false
    );
    if (n <= 0) return {};

    tokens.resize(n);
    if (static_cast<int>(tokens.size()) > target_tokens) {
        tokens.resize(target_tokens);
    }
    return tokens;
}

void print_result(const BenchResult& r) {
    if (!r.error.empty()) {
        std::printf(
            "{\"scenario\":\"%s\",\"error\":\"%s\"}\n",
            r.scenario.c_str(),
            r.error.c_str()
        );
        return;
    }
    std::printf(
        "{"
        "\"scenario\":\"%s\","
        "\"num_sequences\":%d,"
        "\"prompt_tokens\":%d,"
        "\"sample_tokens\":%d,"
        "\"elapsed_seconds\":%.4f,"
        "\"total_tokens_sampled\":%d,"
        "\"aggregate_tokens_per_second\":%.2f,"
        "\"per_sequence_tokens_per_second\":%.2f"
        "}\n",
        r.scenario.c_str(),
        r.num_sequences,
        r.prompt_tokens,
        r.sample_tokens,
        r.elapsed_seconds,
        r.total_tokens_sampled,
        r.aggregate_tokens_per_second,
        r.per_sequence_tokens_per_second
    );
}
