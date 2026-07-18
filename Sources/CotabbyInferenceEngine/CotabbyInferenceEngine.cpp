#include "CotabbyInferenceEngine.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <llama/llama.h>
#include <llama/ggml.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

static void silenced_log_callback(ggml_log_level, const char*, void*) {}

// Single-token chat/instruct/FIM scaffolding that must never surface in autocomplete text.
// Most GGUFs flag these as control tokens (masked by the base rule); this catches vocabularies
// that ship them as ordinary text tokens. EOG-flagged tokens are exempted by the caller so the
// natural stop check keeps firing. Matching is exact (or prefix for the FIM/repo families)
// against the special-rendered piece, so ordinary text like "<|" fragments is never affected.
static bool isScaffoldingMarkerPiece(const char* piece, int length) {
    if (!piece || length <= 0) return false;
    const std::string_view view(piece, static_cast<size_t>(length));
    static constexpr std::string_view exact_markers[] = {
        "<|im_start|>", "<|im_end|>",
        "<|user|>", "<|assistant|>", "<|system|>",
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
        "<|end|>", "<|endoftext|>",
        "<start_of_turn>", "<end_of_turn>",
        "[INST]", "[/INST]",
    };
    for (const auto marker : exact_markers) {
        if (view == marker) return true;
    }
    static constexpr std::string_view prefix_markers[] = {
        "<|fim_", "<fim_", "<|file_sep", "<|repo_name",
    };
    for (const auto prefix : prefix_markers) {
        if (view.size() >= prefix.size() && view.substr(0, prefix.size()) == prefix) {
            return true;
        }
    }
    return false;
}

// Decode threads should match the *performance* core count, not the logical core count.
// llama.cpp's CPU work is a per-layer parallel matmul with a barrier at each layer: schedule any
// of those threads onto efficiency cores and every P-core finishes early only to stall at the
// barrier waiting for the E-core stragglers — slower AND higher energy. With full Metal offload
// the CPU threads only orchestrate/sample, so oversubscribing all logical cores is pure wasted
// wake-ups. P-cores-only is the standard llama.cpp guidance on Apple Silicon; on Intel the
// analogous rule is physical cores, not hyperthreads.
static int resolveDecodeThreadCount() {
#if defined(__APPLE__)
    const auto readSysctlInt = [](const char* name) -> int {
        int value = 0;
        size_t size = sizeof(value);
        if (sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value > 0) {
            return value;
        }
        return 0;
    };
    // perflevel0 = performance cores on Apple Silicon; absent on Intel, where the
    // physical-core fallback applies.
    if (int performance_cores = readSysctlInt("hw.perflevel0.physicalcpu")) {
        return performance_cores;
    }
    if (int physical_cores = readSysctlInt("hw.physicalcpu")) {
        return physical_cores;
    }
#endif
    return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

// ---------------------------------------------------------------------------
// Sequence state
//
// Cotabby owns one autocomplete stream, so the engine owns at most one live sequence. The public
// ID still changes each time a sequence is created; that prevents a late cancellation from
// targeting a replacement sequence that reuses llama's fixed internal slot 0.
//
// `seed_token` / `has_seed_token` carries the first sample produced by
// `decodePrompt`. We sample it right after the prompt's final decode while
// that sequence's logits are still live in the shared context; the next
// `llama_decode` for a different sequence would overwrite them.
//
// `pending_input_token` / `has_pending_input` carries the token that
// `sampleNext` returned and must be feedback-decoded on the next call so the
// shared context produces fresh logits at the new position.
// ---------------------------------------------------------------------------

struct SequenceState {
    int32_t external_id = -1;
    llama_sampler* sampler = nullptr;
    int kv_position_count = 0;
    std::atomic<bool> cancelled{false};
    std::string last_piece;

    llama_token seed_token = 0;
    bool has_seed_token = false;

    llama_token pending_input_token = 0;
    bool has_pending_input = false;

    // Set by setForceWordContinuation; consumed (and cleared) when the next seed token is sampled.
    bool force_word_continuation = false;

    // Whether computeLogprob runs for this sequence's tokens. Defaults to true (the historical
    // behavior) so existing callers keep getting real log-probabilities; callers whose confidence
    // gate is disabled opt out via setComputeLogprob to skip two O(vocab) passes per token.
    bool compute_logprob = true;

    // Log-probability of the seed token, computed at decodePrompt and returned with the seed.
    float seed_logprob = 0.0f;

    // argmax-is-EOG verdict for the seed token's logits row, captured at decodePrompt while
    // those logits are still resident; the row is gone by the time sampleNext returns the seed.
    bool seed_argmax_is_eog = false;

    ~SequenceState() {
        if (sampler) { llama_sampler_free(sampler); }
    }

};

// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------

struct CotabbyInferenceEngine::Impl {
    static constexpr llama_seq_id SEQUENCE_ID = 0;

    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    bool backend_initialized = false;
    std::string model_path;

    llama_context* shared_ctx = nullptr;
    int context_window_tokens = 0;
    int batch_size = 0;
    int thread_count = 0;
    int gpu_layer_count = 0;

    // Token masks built once per model load (see buildTokenMasks). EOG tokens are deliberately
    // excluded so the stop check still fires; they are never emitted as text. `starts_new_word`
    // flags tokens whose decoded text begins with whitespace.
    std::vector<llama_logit_bias> nonprintable_bias;
    std::vector<llama_logit_bias> linebreak_bias;
    std::vector<bool> starts_new_word;

    // One product sequence with a monotonically changing external identity. The mutex protects
    // create/destroy and lookup; callers still must not destroy the sequence while another method
    // is using the returned state pointer.
    mutable std::mutex sequence_mutex;
    std::unique_ptr<SequenceState> sequence;
    int32_t next_external_id = 1;

    // Serializes every llama context mutation. Cotabby's Swift wrapper already orders generation,
    // prefill, trim, and reset, while cancellation only touches the sequence's atomic flag.
    std::mutex decode_mutex;

    SequenceState* findSequence(int32_t id) {
        std::lock_guard<std::mutex> lock(sequence_mutex);
        return sequence && sequence->external_id == id ? sequence.get() : nullptr;
    }

    const SequenceState* findSequence(int32_t id) const {
        std::lock_guard<std::mutex> lock(sequence_mutex);
        return sequence && sequence->external_id == id ? sequence.get() : nullptr;
    }

    llama_sampler* buildSampler(const SamplingConfig& cfg) const {
        auto params = llama_sampler_chain_default_params();
        llama_sampler* chain = llama_sampler_chain_init(params);
        if (!chain) return nullptr;

        // Quality mask: control/unknown/unused tokens can never be sampled as visible text, and
        // for single-line fields line-break tokens are masked too. Placed first so the -inf bias
        // is absolute regardless of the temperature/top-k stages that follow. EOG is intentionally
        // left sampleable so the stop check in sampleNext still fires.
        std::vector<llama_logit_bias> mask = nonprintable_bias;
        if (cfg.single_line && !linebreak_bias.empty()) {
            mask.insert(mask.end(), linebreak_bias.begin(), linebreak_bias.end());
        }
        if (!mask.empty()) {
            auto* bias = llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                static_cast<int32_t>(mask.size()),
                mask.data()
            );
            if (bias) llama_sampler_chain_add(chain, bias);
        }

        if (cfg.repetition_penalty > 1.0f) {
            auto* pen = llama_sampler_init_penalties(
                64, cfg.repetition_penalty, 0.0f, 0.0f
            );
            if (pen) llama_sampler_chain_add(chain, pen);
        }

        if (cfg.temperature > 0.0f) {
            auto* temp = llama_sampler_init_temp(cfg.temperature);
            if (temp) llama_sampler_chain_add(chain, temp);

            if (cfg.top_k > 0) {
                auto* tk = llama_sampler_init_top_k(cfg.top_k);
                if (tk) llama_sampler_chain_add(chain, tk);
            }

            if (cfg.min_p > 0.0f && cfg.min_p < 1.0f) {
                auto* mp = llama_sampler_init_min_p(cfg.min_p, 1);
                if (mp) llama_sampler_chain_add(chain, mp);
            }

            if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
                auto* tp = llama_sampler_init_top_p(cfg.top_p, 1);
                if (tp) llama_sampler_chain_add(chain, tp);
            }

            uint32_t resolved_seed = cfg.seed;
            if (resolved_seed == 0) {
                std::random_device rd;
                resolved_seed = static_cast<uint32_t>(rd());
            }
            auto* dist = llama_sampler_init_dist(resolved_seed);
            if (dist) llama_sampler_chain_add(chain, dist);
        } else {
            auto* greedy = llama_sampler_init_greedy();
            if (greedy) llama_sampler_chain_add(chain, greedy);
        }

        return chain;
    }

    // Classifies the whole vocabulary once per model load. Populates the logit-bias masks and the
    // whitespace-leading flag used for first-token word continuation. Doing it here keeps the hot
    // sampling path free of any per-token tokenizer calls.
    void buildTokenMasks() {
        nonprintable_bias.clear();
        linebreak_bias.clear();
        starts_new_word.clear();
        if (!vocab) return;

        const int32_t n = llama_vocab_n_tokens(vocab);
        starts_new_word.assign(static_cast<size_t>(n), false);

        // BOS belongs at sequence start only; some vocabularies ship it without the control
        // attribute, which would otherwise let it be sampled mid-text.
        const llama_token bos_token = llama_vocab_bos(vocab);

        char piece[64];
        for (llama_token t = 0; t < n; ++t) {
            const bool is_eog = llama_vocab_is_eog(vocab, t);

            // Nonprintable: control (non-EOG), unknown, and unused tokens must never appear as
            // text. EOG stays sampleable so the stop check can recognize a natural end of output.
            if (!is_eog) {
                const enum llama_token_attr attr = llama_vocab_get_attr(vocab, t);
                const bool junk_attr =
                    (attr & (LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED)) != 0;
                bool masked = llama_vocab_is_control(vocab, t) || junk_attr || t == bos_token;
                if (!masked) {
                    // Probe with special rendering: control-style markers decode to an empty
                    // piece under the plain rendering used below, so the scaffolding rule must
                    // look at the special-rendered text instead.
                    const int special_written =
                        llama_token_to_piece(vocab, t, piece, sizeof(piece), 0, true);
                    if (special_written > 0 &&
                        isScaffoldingMarkerPiece(piece, special_written)) {
                        masked = true;
                    }
                }
                if (masked) {
                    nonprintable_bias.push_back({ t, -INFINITY });
                }
            }

            const int written = llama_token_to_piece(vocab, t, piece, sizeof(piece), 0, false);
            if (written <= 0) {
                continue;
            }
            const char first = piece[0];
            if (first == ' ' || first == '\t' || first == '\n' || first == '\r') {
                starts_new_word[static_cast<size_t>(t)] = true;
            }
            if (!is_eog) {
                for (int i = 0; i < written; ++i) {
                    if (piece[i] == '\n' || piece[i] == '\r') {
                        linebreak_bias.push_back({ t, -INFINITY });
                        break;
                    }
                }
            }
        }
    }

    // Masks every "starts a new word" token (decoded text begins with whitespace) in the logits
    // row so the next sampled token must continue the current word. Used for the first token only.
    void maskNewWordStarts(int logits_row) {
        if (!shared_ctx || !vocab) return;
        float* logits = llama_get_logits_ith(shared_ctx, logits_row);
        if (!logits) return;
        const int32_t n = static_cast<int32_t>(starts_new_word.size());
        for (llama_token t = 0; t < n; ++t) {
            if (starts_new_word[static_cast<size_t>(t)]) {
                logits[t] = -INFINITY;
            }
        }
    }

    // Log-probability of `token` under the raw model distribution at `logits_row`, used as a
    // confidence signal. Two O(vocab) passes; only invoked on the autocomplete path.
    float computeLogprob(int logits_row, llama_token token) const {
        if (!shared_ctx || !vocab) return 0.0f;
        const float* logits = llama_get_logits_ith(shared_ctx, logits_row);
        if (!logits) return 0.0f;
        const int32_t n = llama_vocab_n_tokens(vocab);
        if (token < 0 || token >= n) return 0.0f;
        float maxLogit = -INFINITY;
        for (llama_token t = 0; t < n; ++t) {
            if (logits[t] > maxLogit) { maxLogit = logits[t]; }
        }
        double sumExp = 0.0;
        for (llama_token t = 0; t < n; ++t) {
            sumExp += std::exp(static_cast<double>(logits[t] - maxLogit));
        }
        if (!(sumExp > 0.0)) return 0.0f;
        return static_cast<float>(
            static_cast<double>(logits[token] - maxLogit) - std::log(sumExp)
        );
    }

    // Whether the raw distribution at `logits_row` puts its single highest logit on an
    // end-of-generation token. One O(vocab) pass over the row the caller just sampled from;
    // the sampler chain works on a copied candidate array, so the row is unmutated here.
    bool argmaxIsEOG(int logits_row) const {
        if (!shared_ctx || !vocab) return false;
        const float* logits = llama_get_logits_ith(shared_ctx, logits_row);
        if (!logits) return false;
        const int32_t n = llama_vocab_n_tokens(vocab);
        llama_token argmax = 0;
        float best = -INFINITY;
        for (llama_token t = 0; t < n; ++t) {
            if (logits[t] > best) {
                best = logits[t];
                argmax = t;
            }
        }
        return llama_vocab_is_eog(vocab, argmax) || argmax == llama_vocab_eos(vocab);
    }

    void destroySequenceState() {
        std::lock_guard<std::mutex> lock(sequence_mutex);
        sequence.reset();
    }
};

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

CotabbyInferenceEngine::CotabbyInferenceEngine() : impl_(new Impl) {}

CotabbyInferenceEngine::CotabbyInferenceEngine(CotabbyInferenceEngine&& other) noexcept
    : impl_(other.impl_) {
    other.impl_ = nullptr;
}

CotabbyInferenceEngine::~CotabbyInferenceEngine() {
    if (impl_) {
        unloadModel();
        delete impl_;
    }
}

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

EngineStatus CotabbyInferenceEngine::loadModel(const char* path, int gpu_layers,
                                             int context_window_tokens,
                                             int batch_size) {
    if (!impl_ || !path) return EngineStatus::error;

    if (impl_->model && impl_->model_path == path) {
        return EngineStatus::ok;
    }

    if (impl_->model) {
        unloadModel();
    }

    if (!impl_->backend_initialized) {
        llama_log_set(silenced_log_callback, nullptr);
        llama_backend_init();
        impl_->backend_initialized = true;
    }

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    impl_->model = llama_model_load_from_file(path, model_params);
    if (!impl_->model) {
        return EngineStatus::error;
    }

    impl_->vocab = llama_model_get_vocab(impl_->model);
    if (!impl_->vocab) {
        llama_model_free(impl_->model);
        impl_->model = nullptr;
        return EngineStatus::error;
    }

    impl_->model_path = path;
    impl_->context_window_tokens = context_window_tokens;
    impl_->batch_size = batch_size;
    impl_->gpu_layer_count = gpu_layers;
    // Performance cores only — see resolveDecodeThreadCount. hardware_concurrency() counted
    // every logical core including efficiency cores, which both slows barriered matmuls and
    // burns extra package power for nothing when layers are Metal-offloaded anyway.
    impl_->thread_count = resolveDecodeThreadCount();

    // Cotabby has one autocomplete sequence, so the configured context window is the complete KV
    // budget. Reserving a second unused llama sequence used to double this allocation.
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(context_window_tokens);
    ctx_params.n_batch = static_cast<uint32_t>(batch_size);
    ctx_params.n_ubatch = static_cast<uint32_t>(batch_size);
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = static_cast<int32_t>(impl_->thread_count);
    ctx_params.n_threads_batch = static_cast<int32_t>(impl_->thread_count);
    ctx_params.offload_kqv = true;

    impl_->shared_ctx = llama_init_from_model(impl_->model, ctx_params);
    if (!impl_->shared_ctx) {
        llama_model_free(impl_->model);
        impl_->model = nullptr;
        impl_->vocab = nullptr;
        return EngineStatus::error;
    }

    // Precompute the token masks now that the vocab is available; the sampler chains built in
    // createSequence read these, and the hot path then needs no per-token tokenizer work.
    impl_->buildTokenMasks();

    return EngineStatus::ok;
}

void CotabbyInferenceEngine::unloadModel() {
    if (!impl_) return;

    impl_->destroySequenceState();

    if (impl_->shared_ctx) {
        llama_free(impl_->shared_ctx);
        impl_->shared_ctx = nullptr;
    }

    if (impl_->model) {
        llama_model_free(impl_->model);
        impl_->model = nullptr;
    }
    impl_->vocab = nullptr;
    impl_->model_path.clear();

    if (impl_->backend_initialized) {
        llama_backend_free();
        impl_->backend_initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Sequence lifecycle
// ---------------------------------------------------------------------------

int32_t CotabbyInferenceEngine::createSequence(SamplingConfig config) {
    if (!impl_->model || !impl_->shared_ctx) return -1;

    std::lock_guard<std::mutex> lock(impl_->sequence_mutex);
    if (impl_->sequence) return -1;

    llama_sampler* sampler = impl_->buildSampler(config);
    if (!sampler) return -1;

    int32_t id = impl_->next_external_id++;
    auto state = std::make_unique<SequenceState>();
    state->external_id = id;
    state->sampler = sampler;
    impl_->sequence = std::move(state);
    return id;
}

void CotabbyInferenceEngine::destroySequence(int32_t sequence_id) {
    if (!impl_) return;

    std::lock_guard<std::mutex> sequence_lock(impl_->sequence_mutex);
    if (!impl_->sequence || impl_->sequence->external_id != sequence_id) return;

    // Wipe slot 0 before a replacement sequence can reuse it. The caller must not destroy a
    // sequence while decode/sample is using it; decode_mutex protects the llama context mutation.
    if (impl_->shared_ctx) {
        std::lock_guard<std::mutex> decode_lock(impl_->decode_mutex);
        llama_memory_t memory = llama_get_memory(impl_->shared_ctx);
        if (memory) {
            llama_memory_seq_rm(memory, Impl::SEQUENCE_ID, 0, -1);
        }
    }
    impl_->sequence.reset();
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

std::vector<int32_t> CotabbyInferenceEngine::tokenize(const char* text,
                                                     int text_length) const {
    if (!impl_->vocab || !text || text_length <= 0) {
        return {};
    }

    const bool add_bos = llama_vocab_get_add_bos(impl_->vocab);
    int capacity = text_length + 8;

    while (true) {
        std::vector<int32_t> tokens(capacity);
        int n = llama_tokenize(
            impl_->vocab,
            text,
            static_cast<int32_t>(text_length),
            tokens.data(),
            static_cast<int32_t>(capacity),
            add_bos,
            false
        );

        if (n > 0) {
            tokens.resize(n);
            return tokens;
        }
        if (n == 0) {
            return {};
        }
        capacity = std::max(capacity * 2, -n);
    }
}

// ---------------------------------------------------------------------------
// Prompt decoding
//
// Prompt decode runs synchronously on the calling thread under `decode_mutex`. After the prompt's
// final decode succeeds, we
// immediately sample one "seed" token using this sequence's sampler while
// the prompt's logits are still live in the shared context. That seed is
// handed back via the very next `sampleNext` call without any further
// decode work — subsequent calls feedback-decode this seed (then each
// previous sample) to produce fresh logits for the next sample.
// ---------------------------------------------------------------------------

EngineStatus CotabbyInferenceEngine::decodePrompt(int32_t sequence_id,
                                                 const int32_t* tokens,
                                                 int token_count,
                                                 int start_position) {
    if (!impl_->model || !impl_->shared_ctx) return EngineStatus::not_loaded;
    if (!tokens || token_count <= 0) return EngineStatus::ok;

    SequenceState* seq = impl_->findSequence(sequence_id);
    if (!seq) return EngineStatus::error;

    if (seq->cancelled.load(std::memory_order_acquire)) {
        return EngineStatus::cancelled;
    }

    std::unique_lock<std::mutex> lock(impl_->decode_mutex);

    int batch_cap = impl_->batch_size;
    llama_batch batch = llama_batch_init(static_cast<int32_t>(batch_cap), 0, 1);

    int cursor = 0;
    int end = token_count;
    int total_end_position = start_position + token_count;

    while (cursor < end) {
        if (seq->cancelled.load(std::memory_order_acquire)) {
            llama_batch_free(batch);
            return EngineStatus::cancelled;
        }

        int chunk_end = std::min(cursor + batch_cap, end);
        int chunk_size = chunk_end - cursor;

        batch.n_tokens = static_cast<int32_t>(chunk_size);

        for (int i = 0; i < chunk_size; ++i) {
            int token_index = cursor + i;
            batch.token[i] = tokens[token_index];
            batch.pos[i] = static_cast<llama_pos>(start_position + token_index);
            batch.n_seq_id[i] = 1;
            if (batch.seq_id && batch.seq_id[i]) {
                batch.seq_id[i][0] = Impl::SEQUENCE_ID;
            }
            bool is_last = (chunk_end == end && i == chunk_size - 1);
            batch.logits[i] = is_last ? 1 : 0;
        }

        if (llama_decode(impl_->shared_ctx, batch) != 0) {
            llama_batch_free(batch);
            return EngineStatus::error;
        }

        cursor = chunk_end;
    }

    llama_batch_free(batch);
    seq->kv_position_count = total_end_position;

    // First-token word-continuation constraint: when the caret is mid-word, mask new-word-start
    // tokens for this seed only so the completion continues the current word instead of starting
    // a new one. The flag clears after this single token.
    if (seq->force_word_continuation) {
        impl_->maskNewWordStarts(-1);
        seq->force_word_continuation = false;
    }

    // Seed sample: take one token from the prompt's final logits row. The seed will be returned by
    // the next sampleNext call as-is and feedback-decoded by the call after that.
    llama_token seed = llama_sampler_sample(seq->sampler, impl_->shared_ctx, -1);
    llama_sampler_accept(seq->sampler, seed);
    seq->seed_token = seed;
    seq->seed_logprob = seq->compute_logprob ? impl_->computeLogprob(-1, seed) : 0.0f;
    seq->seed_argmax_is_eog = impl_->argmaxIsEOG(-1);
    seq->has_seed_token = true;
    seq->has_pending_input = false;

    return EngineStatus::ok;
}

// ---------------------------------------------------------------------------
// Sampling
//
// First call after decodePrompt returns the seed token directly. Subsequent calls synchronously
// feedback-decode the previously sampled token, then sample and return the next one.
// ---------------------------------------------------------------------------

SampleResult CotabbyInferenceEngine::sampleNext(int32_t sequence_id) {
    SampleResult result{};
    result.token = 0;
    result.piece = nullptr;
    result.piece_length = 0;
    result.is_eos = false;
    result.was_cancelled = false;

    if (!impl_->model || !impl_->vocab || !impl_->shared_ctx) {
        result.is_eos = true;
        return result;
    }

    SequenceState* seq = impl_->findSequence(sequence_id);
    if (!seq) {
        result.is_eos = true;
        return result;
    }

    if (seq->cancelled.load(std::memory_order_acquire)) {
        result.was_cancelled = true;
        return result;
    }

    // Fast path: deliver the seed token sampled at decodePrompt time. No
    // shared-context work needed because the seed was already computed under
    // decode_mutex while the prompt's logits were resident.
    if (seq->has_seed_token) {
        llama_token next = seq->seed_token;
        seq->has_seed_token = false;
        result.argmax_is_eog = seq->seed_argmax_is_eog;

        if (next == llama_vocab_eos(impl_->vocab) ||
            llama_vocab_is_eog(impl_->vocab, next)) {
            result.token = next;
            result.is_eos = true;
            return result;
        }

        seq->last_piece.resize(64);
        while (true) {
            int written = llama_token_to_piece(
                impl_->vocab, next, seq->last_piece.data(),
                static_cast<int32_t>(seq->last_piece.size()), 0, false
            );
            if (written >= 0) { seq->last_piece.resize(written); break; }
            seq->last_piece.resize(static_cast<size_t>(-written) + 1);
        }

        // The seed has not yet been added to KV. Remember it as the next feedback-decode input so
        // the call after this one can produce fresh logits.
        seq->pending_input_token = next;
        seq->has_pending_input = true;

        result.token = next;
        result.piece = seq->last_piece.c_str();
        result.piece_length = static_cast<int>(seq->last_piece.size());
        result.logprob = seq->seed_logprob;
        return result;
    }

    if (!seq->has_pending_input) {
        // Nothing to decode and no seed — caller forgot to decodePrompt or
        // trimmed the KV down to nothing without re-priming.
        result.is_eos = true;
        return result;
    }

    // Cotabby has no sibling sequence to batch with, so feedback decode happens directly on the
    // calling worker under the same context lock used by prompt decode and KV mutation.
    std::lock_guard<std::mutex> lock(impl_->decode_mutex);
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = seq->pending_input_token;
    batch.pos[0] = static_cast<llama_pos>(seq->kv_position_count);
    batch.n_seq_id[0] = 1;
    if (batch.seq_id && batch.seq_id[0]) {
        batch.seq_id[0][0] = Impl::SEQUENCE_ID;
    }
    batch.logits[0] = 1;

    const int status = llama_decode(impl_->shared_ctx, batch);
    if (status != 0) {
        result.is_eos = true;
    } else if (seq->cancelled.load(std::memory_order_acquire)) {
        result.was_cancelled = true;
    } else {
        const llama_token next = llama_sampler_sample(seq->sampler, impl_->shared_ctx, 0);
        result.argmax_is_eog = impl_->argmaxIsEOG(0);
        result.token = next;

        if (next == llama_vocab_eos(impl_->vocab) ||
            llama_vocab_is_eog(impl_->vocab, next)) {
            result.is_eos = true;
        } else {
            llama_sampler_accept(seq->sampler, next);
            seq->last_piece.resize(64);
            while (true) {
                const int written = llama_token_to_piece(
                    impl_->vocab,
                    next,
                    seq->last_piece.data(),
                    static_cast<int32_t>(seq->last_piece.size()),
                    0,
                    false
                );
                if (written >= 0) {
                    seq->last_piece.resize(written);
                    break;
                }
                seq->last_piece.resize(static_cast<size_t>(-written) + 1);
            }
            result.piece = seq->last_piece.c_str();
            result.piece_length = static_cast<int>(seq->last_piece.size());
            result.logprob = seq->compute_logprob ? impl_->computeLogprob(0, next) : 0.0f;
        }
    }
    llama_batch_free(batch);

    if (result.is_eos || result.was_cancelled) {
        return result;
    }

    // Feedback decode advanced KV by one position; record the just-sampled
    // token as input for the next call.
    seq->kv_position_count++;
    seq->pending_input_token = result.token;
    seq->has_pending_input = true;
    return result;
}

// ---------------------------------------------------------------------------
// KV cache management
// ---------------------------------------------------------------------------

bool CotabbyInferenceEngine::trimKV(int32_t sequence_id, int keep_positions) {
    if (!impl_->shared_ctx) return false;
    SequenceState* seq = impl_->findSequence(sequence_id);
    if (!seq) return false;

    llama_memory_t memory = llama_get_memory(impl_->shared_ctx);
    if (!memory) return false;

    // Serialize with prompt and feedback decode; never remove KV while llama is mutating it.
    std::lock_guard<std::mutex> lock(impl_->decode_mutex);

    bool ok = llama_memory_seq_rm(
        memory,
        Impl::SEQUENCE_ID,
        static_cast<llama_pos>(keep_positions),
        -1
    );

    if (ok) {
        seq->kv_position_count = keep_positions;
        // Any seed/pending input is now stale (it would feedback-decode into
        // a trimmed-away position). Caller must call decodePrompt to re-seed
        // before the next sampleNext.
        seq->has_seed_token = false;
        seq->has_pending_input = false;
    }
    return ok;
}

void CotabbyInferenceEngine::setForceWordContinuation(int32_t sequence_id, bool enabled) {
    if (!impl_) return;
    SequenceState* seq = impl_->findSequence(sequence_id);
    if (seq) {
        seq->force_word_continuation = enabled;
    }
}

void CotabbyInferenceEngine::setComputeLogprob(int32_t sequence_id, bool enabled) {
    if (!impl_) return;
    SequenceState* seq = impl_->findSequence(sequence_id);
    if (seq) {
        seq->compute_logprob = enabled;
    }
}

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

void CotabbyInferenceEngine::cancelSequence(int32_t sequence_id) {
    SequenceState* seq = impl_->findSequence(sequence_id);
    if (seq) {
        seq->cancelled.store(true, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

int CotabbyInferenceEngine::getContextWindowTokens() const {
    return impl_->context_window_tokens;
}

int CotabbyInferenceEngine::getBatchSize() const {
    return impl_->batch_size;
}

int CotabbyInferenceEngine::getThreadCount() const {
    return impl_->thread_count;
}

int CotabbyInferenceEngine::getGPULayerCount() const {
    return impl_->gpu_layer_count;
}
