#include "bench_common.h"

#include <algorithm>
#include <thread>
#include <vector>

#include <llama/llama.h>

namespace {

llama_sampler* make_greedy_sampler() {
    auto params = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(params);
    if (!chain) return nullptr;
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    return chain;
}

// Single shared context with room for `num_sequences` distinct seq_ids. The
// KV cache size is multiplied by `num_sequences` so each sequence still gets
// its configured `ctx_size` of slots — Phase 0 also wants to confirm whether
// n_ctx is shared or per-sequence in the b9310 build.
llama_context* create_shared_context(
    llama_model* model,
    const BenchConfig& cfg
) {
    auto p = llama_context_default_params();
    p.n_ctx = static_cast<uint32_t>(cfg.ctx_size * cfg.num_sequences);
    p.n_batch = static_cast<uint32_t>(cfg.batch_size);
    p.n_ubatch = static_cast<uint32_t>(cfg.batch_size);
    p.n_seq_max = static_cast<uint32_t>(cfg.num_sequences);
    int t = static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency())
    );
    p.n_threads = t;
    p.n_threads_batch = t;
    p.offload_kqv = true;
    return llama_init_from_model(model, p);
}

// Decodes one sequence's prompt into the shared context, tagged with `seq_id`.
// Only the final token of the final chunk has logits=1, so the next sampler
// call reads from batch index 0 of that decode.
bool decode_prompt_for_seq(
    llama_context* ctx,
    llama_seq_id seq_id,
    const std::vector<int32_t>& tokens,
    int batch_cap
) {
    llama_batch batch = llama_batch_init(batch_cap, 0, 1);
    int cursor = 0;
    int n = static_cast<int>(tokens.size());
    while (cursor < n) {
        int chunk_end = std::min(cursor + batch_cap, n);
        int chunk_size = chunk_end - cursor;
        batch.n_tokens = chunk_size;
        for (int i = 0; i < chunk_size; ++i) {
            int idx = cursor + i;
            batch.token[i] = tokens[idx];
            batch.pos[i] = idx;
            batch.n_seq_id[i] = 1;
            if (batch.seq_id && batch.seq_id[i]) batch.seq_id[i][0] = seq_id;
            bool is_last = (chunk_end == n && i == chunk_size - 1);
            batch.logits[i] = is_last ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return false;
        }
        cursor = chunk_end;
    }
    llama_batch_free(batch);
    return true;
}

} // namespace

BenchResult run_baseline_b_shared_context(
    llama_model* model,
    const BenchConfig& cfg
) {
    BenchResult r;
    r.scenario = "b_shared_context";
    r.num_sequences = cfg.num_sequences;
    r.prompt_tokens = cfg.prompt_tokens;
    r.sample_tokens = cfg.sample_tokens;

    const llama_vocab* vocab = llama_model_get_vocab(model);
    auto prompt = make_synthetic_prompt(vocab, cfg.prompt_tokens);

    llama_context* ctx = nullptr;
    std::vector<llama_sampler*> samplers(cfg.num_sequences, nullptr);

    auto cleanup = [&]() {
        for (auto* s : samplers) if (s) llama_sampler_free(s);
        if (ctx) llama_free(ctx);
    };

    if (prompt.empty()) {
        r.error = "Failed to tokenize synthetic prompt";
        cleanup();
        return r;
    }

    ctx = create_shared_context(model, cfg);
    if (!ctx) {
        r.error = "Failed to create shared context";
        cleanup();
        return r;
    }

    for (int s = 0; s < cfg.num_sequences; ++s) {
        samplers[s] = make_greedy_sampler();
        if (!samplers[s]) {
            r.error = "Failed to create sampler";
            cleanup();
            return r;
        }
    }

    std::vector<llama_token> last_tokens(cfg.num_sequences, 0);
    std::vector<int> positions(cfg.num_sequences, cfg.prompt_tokens);

    // Warmup: decode each prompt and immediately sample one token while that
    // sequence's prompt logits are still resident. The next prompt decode for
    // sequence s+1 overwrites the logits buffer, so we must sample-before-
    // advance. The sampled token becomes the seed input to the timed loop.
    for (int s = 0; s < cfg.num_sequences; ++s) {
        if (!decode_prompt_for_seq(
                ctx, static_cast<llama_seq_id>(s),
                prompt, cfg.batch_size)) {
            r.error = "Prompt decode failed";
            cleanup();
            return r;
        }
        // -1 reads from the last logits row, which is where this sequence's
        // prompt-final-token logits live until the next decode overwrites
        // them.
        last_tokens[s] = llama_sampler_sample(samplers[s], ctx, -1);
        llama_sampler_accept(samplers[s], last_tokens[s]);
    }

    // Timed: build one batch carrying num_sequences tokens (different seq_ids),
    // decode all of them in a single llama_decode call, then sample one new
    // token per sequence from its respective logit row. Each iteration
    // produces one new token per sequence.
    llama_batch batch = llama_batch_init(cfg.num_sequences + 4, 0, 1);
    Timer timer;
    timer.start();
    bool decode_failed = false;

    for (int step = 1; step < cfg.sample_tokens; ++step) {
        batch.n_tokens = cfg.num_sequences;
        for (int s = 0; s < cfg.num_sequences; ++s) {
            batch.token[s] = last_tokens[s];
            batch.pos[s] = positions[s];
            batch.n_seq_id[s] = 1;
            if (batch.seq_id && batch.seq_id[s]) {
                batch.seq_id[s][0] = static_cast<llama_seq_id>(s);
            }
            batch.logits[s] = 1;
            positions[s]++;
        }
        if (llama_decode(ctx, batch) != 0) {
            decode_failed = true;
            break;
        }
        for (int s = 0; s < cfg.num_sequences; ++s) {
            last_tokens[s] = llama_sampler_sample(samplers[s], ctx, s);
            llama_sampler_accept(samplers[s], last_tokens[s]);
        }
    }

    r.elapsed_seconds = timer.elapsed_seconds();
    llama_batch_free(batch);

    if (decode_failed) {
        r.error = "llama_decode failed mid-loop";
        cleanup();
        return r;
    }

    // Exclude the warmup seed sample from total_tokens_sampled so the
    // numerator is comparable to baseline A's timed-only count.
    r.total_tokens_sampled = (cfg.sample_tokens - 1) * cfg.num_sequences;
    if (r.elapsed_seconds > 0.0) {
        r.aggregate_tokens_per_second =
            r.total_tokens_sampled / r.elapsed_seconds;
        r.per_sequence_tokens_per_second =
            r.aggregate_tokens_per_second / cfg.num_sequences;
    }

    cleanup();
    return r;
}
