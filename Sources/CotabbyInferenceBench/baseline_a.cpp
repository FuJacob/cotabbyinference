#include "bench_common.h"

#include <algorithm>
#include <atomic>
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

llama_context* create_isolated_context(
    llama_model* model,
    const BenchConfig& cfg
) {
    auto p = llama_context_default_params();
    p.n_ctx = static_cast<uint32_t>(cfg.ctx_size);
    p.n_batch = static_cast<uint32_t>(cfg.batch_size);
    p.n_ubatch = static_cast<uint32_t>(cfg.batch_size);
    p.n_seq_max = 1;
    int t = static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency())
    );
    p.n_threads = t;
    p.n_threads_batch = t;
    p.offload_kqv = true;
    return llama_init_from_model(model, p);
}

// Walks the prompt in chunks of `batch_cap`, setting logits=1 only on the
// final token of the final chunk. That leaves a single logits row at batch
// index 0 of the last decode, which is where the first sample reads from.
bool decode_prompt(
    llama_context* ctx,
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
            if (batch.seq_id && batch.seq_id[i]) batch.seq_id[i][0] = 0;
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

// Steady-state sampling loop, run once per sequence in its own thread. Takes
// the seed token from warmup as input. Each iteration feedback-decodes the
// previous sample then takes a new one. Mirrors CotabbyInferenceEngine's
// sampleNext pattern minus the per-call batch allocation.
int run_sampling_loop(
    llama_context* ctx,
    llama_sampler* sampler,
    llama_token seed_token,
    int start_position,
    int sample_count
) {
    if (sample_count <= 0) return 0;

    llama_token next = seed_token;
    int position = start_position;
    int sampled = 0;

    llama_batch batch = llama_batch_init(1, 0, 1);
    for (int i = 0; i < sample_count; ++i) {
        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = position;
        batch.n_seq_id[0] = 1;
        if (batch.seq_id && batch.seq_id[0]) batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return sampled;
        }
        position++;
        next = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, next);
        sampled++;
    }
    llama_batch_free(batch);
    return sampled;
}

} // namespace

BenchResult run_baseline_a_two_contexts(
    llama_model* model,
    const BenchConfig& cfg
) {
    BenchResult r;
    r.scenario = "a_two_contexts";
    r.num_sequences = cfg.num_sequences;
    r.prompt_tokens = cfg.prompt_tokens;
    r.sample_tokens = cfg.sample_tokens;

    const llama_vocab* vocab = llama_model_get_vocab(model);
    auto prompt = make_synthetic_prompt(vocab, cfg.prompt_tokens);

    std::vector<llama_context*> ctxs(cfg.num_sequences, nullptr);
    std::vector<llama_sampler*> samplers(cfg.num_sequences, nullptr);

    auto cleanup = [&]() {
        for (auto* s : samplers) if (s) llama_sampler_free(s);
        for (auto* c : ctxs) if (c) llama_free(c);
    };

    if (prompt.empty()) {
        r.error = "Failed to tokenize synthetic prompt";
        cleanup();
        return r;
    }

    // Allocate all resources upfront. Any failure here aborts before any
    // timing happens.
    for (int i = 0; i < cfg.num_sequences; ++i) {
        ctxs[i] = create_isolated_context(model, cfg);
        if (!ctxs[i]) {
            r.error = "Failed to create context";
            cleanup();
            return r;
        }
        samplers[i] = make_greedy_sampler();
        if (!samplers[i]) {
            r.error = "Failed to create sampler";
            cleanup();
            return r;
        }
    }

    // Warmup: prompt decode + first sample per context. Both untimed so the
    // timed section measures only steady-state decode+sample work, which is
    // what the user actually feels in the real app (prompt is mostly cached
    // across requests via KV reuse). Pulling the seed sample out also makes
    // the count symmetric with baseline B's timed section.
    std::vector<llama_token> seed_tokens(cfg.num_sequences, 0);
    for (int i = 0; i < cfg.num_sequences; ++i) {
        if (!decode_prompt(ctxs[i], prompt, cfg.batch_size)) {
            r.error = "Prompt decode failed";
            cleanup();
            return r;
        }
        // -1 reads from the last logits row, which is where the prompt's
        // final-token logits live (we set logits=1 only on that position
        // during prompt decode).
        seed_tokens[i] = llama_sampler_sample(samplers[i], ctxs[i], -1);
        llama_sampler_accept(samplers[i], seed_tokens[i]);
    }

    // Timed: one thread per sequence, each runs (sample_tokens - 1) feedback
    // decode + sample iterations against its own context.
    Timer timer;
    timer.start();
    std::vector<std::thread> threads;
    std::atomic<int> total{0};
    threads.reserve(cfg.num_sequences);
    for (int i = 0; i < cfg.num_sequences; ++i) {
        threads.emplace_back([&, i]() {
            int n = run_sampling_loop(
                ctxs[i],
                samplers[i],
                seed_tokens[i],
                cfg.prompt_tokens,
                cfg.sample_tokens - 1
            );
            total.fetch_add(n);
        });
    }
    for (auto& t : threads) t.join();
    r.elapsed_seconds = timer.elapsed_seconds();
    r.total_tokens_sampled = total.load();

    if (r.elapsed_seconds > 0.0) {
        r.aggregate_tokens_per_second =
            r.total_tokens_sampled / r.elapsed_seconds;
        r.per_sequence_tokens_per_second =
            r.aggregate_tokens_per_second / cfg.num_sequences;
    }

    cleanup();
    return r;
}
