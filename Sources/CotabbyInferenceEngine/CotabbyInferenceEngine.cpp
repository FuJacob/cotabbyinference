#include "CotabbyInferenceEngine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <llama/llama.h>
#include <llama/ggml.h>

static void silenced_log_callback(ggml_log_level, const char*, void*) {}

// ---------------------------------------------------------------------------
// Per-sequence state
//
// Phase 1 architecture: all sequences share a single `llama_context` allocated
// in `Impl`. Each sequence owns its own sampler chain, KV-cache position
// counter, cancellation flag, and detokenization buffer. The `seq_id` is the
// internal `llama_seq_id` slot (0..MAX_SEQUENCES-1) used to tag this
// sequence's tokens in the shared KV cache.
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
    llama_seq_id seq_id = -1;
    llama_sampler* sampler = nullptr;
    SamplingConfig config{};
    int kv_position_count = 0;
    std::atomic<bool> cancelled{false};
    std::string last_piece;

    llama_token seed_token = 0;
    bool has_seed_token = false;

    llama_token pending_input_token = 0;
    bool has_pending_input = false;

    // Set by setForceWordContinuation; consumed (and cleared) when the next seed token is sampled.
    bool force_word_continuation = false;

    ~SequenceState() {
        if (sampler) { llama_sampler_free(sampler); }
    }

    SequenceState() = default;
    SequenceState(SequenceState&& o) noexcept
        : seq_id(o.seq_id),
          sampler(o.sampler),
          config(o.config),
          kv_position_count(o.kv_position_count),
          cancelled(o.cancelled.load()),
          last_piece(std::move(o.last_piece)),
          seed_token(o.seed_token),
          has_seed_token(o.has_seed_token),
          pending_input_token(o.pending_input_token),
          has_pending_input(o.has_pending_input),
          force_word_continuation(o.force_word_continuation) {
        o.sampler = nullptr;
    }
    SequenceState& operator=(SequenceState&&) = delete;
    SequenceState(const SequenceState&) = delete;
    SequenceState& operator=(const SequenceState&) = delete;
};

// ---------------------------------------------------------------------------
// Pending decode + sample request handed to the decoder thread.
//
// Holds raw pointers into a SequenceState entry. SequenceState entries live in
// a node-based unordered_map (stable addresses across inserts), and the
// public contract forbids destroying a sequence with sampleNext in flight, so
// the pointers stay valid for the request's lifetime.
// ---------------------------------------------------------------------------

struct PendingRequest {
    llama_seq_id seq_id = -1;
    llama_token token = 0;
    int position = 0;
    llama_sampler* sampler = nullptr;
    std::atomic<bool>* cancelled_ptr = nullptr;
    std::string* piece_buffer = nullptr;
    SampleResult* result_out = nullptr;
    std::promise<void> done;
};

// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------

struct CotabbyInferenceEngine::Impl {
    static constexpr int MAX_SEQUENCES = 4;

    // Microseconds the decoder thread waits after the first request arrives
    // before flushing. This is the knob that lets multi-sequence callers pile
    // up tokens for a batched `llama_decode`. Too short → no batching; too
    // long → single-sequence callers feel extra latency per token. 200µs was
    // chosen as a starting point and should be tuned via the bench.
    static constexpr int BATCH_WINDOW_MICROS = 200;

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

    // Public-facing sequence map (external int32_t IDs → state) and the
    // internal `llama_seq_id` slot allocator.
    mutable std::mutex sequences_mutex;
    std::unordered_map<int32_t, SequenceState> sequences;
    int32_t next_external_id = 1;
    bool seq_slot_in_use[MAX_SEQUENCES] = {false};

    // Decoder thread. Owns all `llama_decode` calls on `shared_ctx` after
    // model load, including both sample-step batches (built from
    // `sampleNext` requests) and prompt-decode chunks (forwarded from
    // `decodePrompt` via the same queue path).
    std::mutex decode_mutex;
    std::condition_variable request_cv;
    std::vector<PendingRequest> pending;
    std::thread decoder_thread;
    bool decoder_should_stop = false;
    bool decoder_running = false;

    int allocateSeqSlot() {
        for (int i = 0; i < MAX_SEQUENCES; ++i) {
            if (!seq_slot_in_use[i]) {
                seq_slot_in_use[i] = true;
                return i;
            }
        }
        return -1;
    }

    void releaseSeqSlot(int slot) {
        if (slot >= 0 && slot < MAX_SEQUENCES) {
            seq_slot_in_use[slot] = false;
        }
    }

    SequenceState* findSequence(int32_t id) {
        std::lock_guard<std::mutex> lock(sequences_mutex);
        auto it = sequences.find(id);
        return it != sequences.end() ? &it->second : nullptr;
    }

    const SequenceState* findSequence(int32_t id) const {
        std::lock_guard<std::mutex> lock(sequences_mutex);
        auto it = sequences.find(id);
        return it != sequences.end() ? &it->second : nullptr;
    }

    llama_sampler* buildSampler(const SamplingConfig& cfg) const {
        auto params = llama_sampler_chain_default_params();
        llama_sampler* chain = llama_sampler_chain_init(params);
        if (!chain) return nullptr;

        // Quality mask: control/unknown/unused tokens can never be sampled as visible text, and
        // for single-line fields line-break tokens are masked too. Placed first so the -inf bias
        // is absolute regardless of the temperature/top-k stages that follow. EOG is intentionally
        // left sampleable so the stop check in processBatch/sampleNext still fires.
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

        char piece[64];
        for (llama_token t = 0; t < n; ++t) {
            const bool is_eog = llama_vocab_is_eog(vocab, t);

            // Nonprintable: control (non-EOG), unknown, and unused tokens must never appear as
            // text. EOG stays sampleable so the stop check can recognize a natural end of output.
            if (!is_eog) {
                const enum llama_token_attr attr = llama_vocab_get_attr(vocab, t);
                const bool junk_attr =
                    (attr & (LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED)) != 0;
                if (llama_vocab_is_control(vocab, t) || junk_attr) {
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

    void destroyAllSequences() {
        std::lock_guard<std::mutex> lock(sequences_mutex);
        for (auto& [id, seq] : sequences) {
            releaseSeqSlot(seq.seq_id);
        }
        sequences.clear();
    }

    void startDecoderThread() {
        if (decoder_running) return;
        decoder_should_stop = false;
        decoder_running = true;
        decoder_thread = std::thread([this]() { decoderRun(); });
    }

    void stopDecoderThread() {
        if (!decoder_running) return;
        {
            std::lock_guard<std::mutex> lock(decode_mutex);
            decoder_should_stop = true;
        }
        request_cv.notify_all();
        if (decoder_thread.joinable()) {
            decoder_thread.join();
        }
        decoder_running = false;
    }

    // Decoder loop: collect pending sample-step requests, batch them into one
    // llama_decode, sample each sequence's next token using its own sampler,
    // and resolve every request's promise so the caller threads can return.
    void decoderRun() {
        while (true) {
            std::vector<PendingRequest> batch;
            {
                std::unique_lock<std::mutex> lock(decode_mutex);
                request_cv.wait(lock, [&] {
                    return decoder_should_stop || !pending.empty();
                });
                if (decoder_should_stop && pending.empty()) return;

                // Brief flush window: when only one request has arrived,
                // wait a short period for siblings to pile in so the next
                // llama_decode can batch them together. Multi-sequence
                // workloads naturally fall into lockstep here because each
                // sequence resubmits as soon as its previous sample returns.
                request_cv.wait_for(
                    lock,
                    std::chrono::microseconds(BATCH_WINDOW_MICROS),
                    [&] { return decoder_should_stop; }
                );

                if (pending.empty()) {
                    if (decoder_should_stop) return;
                    continue;
                }

                batch = std::move(pending);
                pending.clear();

                // Process while still holding decode_mutex so prompt-decode
                // calls in `decodePrompt` cannot race with the sample-step
                // llama_decode below. llama_decode + sampling for a small
                // batch is ~10ms; callers blocked on staging will simply
                // queue for the next cycle.
                processBatch(batch);
            }
        }
    }

    void processBatch(std::vector<PendingRequest>& reqs) {
        if (reqs.empty() || !shared_ctx) return;

        llama_batch batch = llama_batch_init(
            static_cast<int32_t>(reqs.size() + 4), 0, 1
        );
        batch.n_tokens = static_cast<int32_t>(reqs.size());
        for (int i = 0; i < static_cast<int>(reqs.size()); ++i) {
            batch.token[i] = reqs[i].token;
            batch.pos[i] = static_cast<llama_pos>(reqs[i].position);
            batch.n_seq_id[i] = 1;
            if (batch.seq_id && batch.seq_id[i]) {
                batch.seq_id[i][0] = reqs[i].seq_id;
            }
            batch.logits[i] = 1;
        }

        int status = llama_decode(shared_ctx, batch);

        for (int i = 0; i < static_cast<int>(reqs.size()); ++i) {
            PendingRequest& req = reqs[i];
            SampleResult r{};
            r.token = 0;
            r.piece = nullptr;
            r.piece_length = 0;
            r.is_eos = false;
            r.was_cancelled = false;

            if (status != 0) {
                r.is_eos = true;
            } else if (req.cancelled_ptr &&
                       req.cancelled_ptr->load(std::memory_order_acquire)) {
                r.was_cancelled = true;
            } else {
                llama_token next = llama_sampler_sample(
                    req.sampler, shared_ctx, i
                );
                if (next == llama_vocab_eos(vocab) ||
                    llama_vocab_is_eog(vocab, next)) {
                    r.token = next;
                    r.is_eos = true;
                } else {
                    llama_sampler_accept(req.sampler, next);

                    std::string& piece = *req.piece_buffer;
                    piece.resize(64);
                    while (true) {
                        int written = llama_token_to_piece(
                            vocab, next, piece.data(),
                            static_cast<int32_t>(piece.size()), 0, false
                        );
                        if (written >= 0) { piece.resize(written); break; }
                        piece.resize(static_cast<size_t>(-written) + 1);
                    }

                    r.token = next;
                    r.piece = piece.c_str();
                    r.piece_length = static_cast<int>(piece.size());
                }
            }

            *req.result_out = r;
            req.done.set_value();
        }

        llama_batch_free(batch);
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
    impl_->thread_count = static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency())
    );

    // Shared context sized to hold MAX_SEQUENCES sequences each with up to
    // `context_window_tokens` KV slots. llama.cpp's `n_ctx` is the total slot
    // budget across all sequences in a context, so we multiply to give each
    // sequence the configured window without sequences stealing slots from
    // each other.
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(
        context_window_tokens * Impl::MAX_SEQUENCES
    );
    ctx_params.n_batch = static_cast<uint32_t>(batch_size);
    ctx_params.n_ubatch = static_cast<uint32_t>(batch_size);
    ctx_params.n_seq_max = static_cast<uint32_t>(Impl::MAX_SEQUENCES);
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

    impl_->startDecoderThread();
    return EngineStatus::ok;
}

void CotabbyInferenceEngine::unloadModel() {
    if (!impl_) return;

    impl_->stopDecoderThread();
    impl_->destroyAllSequences();

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

bool CotabbyInferenceEngine::isModelLoaded() const {
    return impl_ && impl_->model != nullptr && impl_->shared_ctx != nullptr;
}

// ---------------------------------------------------------------------------
// Sequence lifecycle
// ---------------------------------------------------------------------------

int32_t CotabbyInferenceEngine::createSequence(SamplingConfig config) {
    if (!impl_->model || !impl_->shared_ctx) return -1;

    std::lock_guard<std::mutex> lock(impl_->sequences_mutex);
    int slot = impl_->allocateSeqSlot();
    if (slot < 0) return -1;

    llama_sampler* sampler = impl_->buildSampler(config);
    if (!sampler) {
        impl_->releaseSeqSlot(slot);
        return -1;
    }

    SequenceState state;
    state.seq_id = static_cast<llama_seq_id>(slot);
    state.sampler = sampler;
    state.config = config;

    int32_t id = impl_->next_external_id++;
    impl_->sequences.emplace(id, std::move(state));
    return id;
}

void CotabbyInferenceEngine::destroySequence(int32_t sequence_id) {
    if (!impl_) return;

    // Look up the internal slot once. Caller's contract is to not destroy a
    // sequence with sampleNext in flight, so the entry is stable for the
    // duration of this call.
    llama_seq_id slot_to_wipe = -1;
    {
        std::lock_guard<std::mutex> seq_lock(impl_->sequences_mutex);
        auto it = impl_->sequences.find(sequence_id);
        if (it == impl_->sequences.end()) return;
        slot_to_wipe = it->second.seq_id;
    }

    // Wipe this sequence's KV slots in the shared context before releasing
    // the slot, otherwise stale positions linger and reusing the slot later
    // would mix old tokens with new ones. Hold decode_mutex so the wipe does
    // not race with the decoder thread's llama_decode call.
    if (impl_->shared_ctx && slot_to_wipe >= 0) {
        std::lock_guard<std::mutex> decode_lock(impl_->decode_mutex);
        llama_memory_t memory = llama_get_memory(impl_->shared_ctx);
        if (memory) {
            llama_memory_seq_rm(memory, slot_to_wipe, 0, -1);
        }
    }

    std::lock_guard<std::mutex> seq_lock(impl_->sequences_mutex);
    auto it = impl_->sequences.find(sequence_id);
    if (it != impl_->sequences.end()) {
        impl_->releaseSeqSlot(it->second.seq_id);
        impl_->sequences.erase(it);
    }
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

std::vector<int32_t> CotabbyInferenceEngine::tokenize(const char* text,
                                                     int text_length) const {
    // Preserve the historical contract: add BOS per model metadata, treat any
    // special-token text as plaintext (parse_special = false).
    bool add_bos = impl_->vocab ? llama_vocab_get_add_bos(impl_->vocab) : false;
    return tokenizeWithOptions(text, text_length, add_bos, false);
}

std::vector<int32_t> CotabbyInferenceEngine::tokenizeWithOptions(
    const char* text, int text_length,
    bool add_special, bool parse_special) const {
    if (!impl_->vocab || !text || text_length <= 0) {
        return {};
    }

    int capacity = text_length + 8;

    while (true) {
        std::vector<int32_t> tokens(capacity);
        int n = llama_tokenize(
            impl_->vocab,
            text,
            static_cast<int32_t>(text_length),
            tokens.data(),
            static_cast<int32_t>(capacity),
            add_special,
            parse_special
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

bool CotabbyInferenceEngine::hasChatTemplate() const {
    if (!impl_->model) {
        return false;
    }
    return llama_model_chat_template(impl_->model, /*name=*/nullptr) != nullptr;
}

int CotabbyInferenceEngine::applyChatTemplate(
    const char* system_text,
    const char* user_text,
    bool add_assistant,
    char* buffer,
    int buffer_size) const {
    if (!impl_->model || !system_text || !user_text ||
        !buffer || buffer_size <= 0) {
        return 0;
    }

    const char* tmpl = llama_model_chat_template(impl_->model, /*name=*/nullptr);
    if (!tmpl) {
        return 0;
    }

    // Borrowed `const char*` from the caller; valid for this call's duration.
    llama_chat_message chat[2] = {
        { "system", system_text },
        { "user", user_text }
    };

    int32_t n = llama_chat_apply_template(
        tmpl,
        chat,
        2,
        add_assistant,
        buffer,
        static_cast<int32_t>(buffer_size)
    );

    // Contract of llama_chat_apply_template: returns the total byte length of
    // the formatted prompt; negative means the template is unsupported by
    // llama.cpp's predefined list. A positive value larger than the buffer
    // means the output did not fit and the caller must retry with a bigger
    // buffer. Map all three onto this function's documented C-ABI contract.
    if (n < 0) {
        return 0;              // genuine render failure → caller falls back to raw
    }
    if (n > buffer_size) {
        return -n;             // too small → -(required size); caller resizes and retries
    }
    return n;                  // success: n bytes written (n <= buffer_size)
}

int CotabbyInferenceEngine::detokenize(int32_t token, char* buffer,
                                      int buffer_size) const {
    if (!impl_->vocab || !buffer || buffer_size <= 0) return 0;

    int written = llama_token_to_piece(
        impl_->vocab,
        token,
        buffer,
        static_cast<int32_t>(buffer_size),
        0,
        false
    );

    return written;
}

// ---------------------------------------------------------------------------
// Prompt decoding
//
// Prompt decode runs synchronously on the calling thread, but takes
// `decode_mutex` so it serializes with the decoder thread's sample-step
// llama_decode calls. After the prompt's final decode succeeds, we
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
                batch.seq_id[i][0] = seq->seq_id;
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

    // Seed sample: take one token from the prompt's logits row right now,
    // before any other sequence's decode can overwrite the shared logits
    // buffer. The seed will be returned by the next sampleNext call as-is
    // and feedback-decoded by the call after that.
    llama_token seed = llama_sampler_sample(seq->sampler, impl_->shared_ctx, -1);
    llama_sampler_accept(seq->sampler, seed);
    seq->seed_token = seed;
    seq->has_seed_token = true;
    seq->has_pending_input = false;

    return EngineStatus::ok;
}

// ---------------------------------------------------------------------------
// Sampling
//
// First call after decodePrompt: return the seed token directly (no decode
// queued). Subsequent calls: queue the previously sampled token for feedback
// decode via the decoder thread, then return its sampled result.
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

        // The seed has not yet been added to KV. Queue it as the next
        // feedback-decode input so the call after this one has fresh logits.
        seq->pending_input_token = next;
        seq->has_pending_input = true;

        result.token = next;
        result.piece = seq->last_piece.c_str();
        result.piece_length = static_cast<int>(seq->last_piece.size());
        return result;
    }

    if (!seq->has_pending_input) {
        // Nothing to decode and no seed — caller forgot to decodePrompt or
        // trimmed the KV down to nothing without re-priming.
        result.is_eos = true;
        return result;
    }

    // Steady-state path: queue the previously sampled token (or the seed,
    // if this is the call right after the seed was delivered) for feedback
    // decode via the decoder thread, which batches it with any other
    // sequences that happen to be sampling at the same time.
    PendingRequest req;
    req.seq_id = seq->seq_id;
    req.token = seq->pending_input_token;
    req.position = seq->kv_position_count;
    req.sampler = seq->sampler;
    req.cancelled_ptr = &seq->cancelled;
    req.piece_buffer = &seq->last_piece;
    req.result_out = &result;
    auto done_future = req.done.get_future();

    {
        std::lock_guard<std::mutex> lock(impl_->decode_mutex);
        impl_->pending.push_back(std::move(req));
        impl_->request_cv.notify_one();
    }

    done_future.wait();

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

    // Serialize with the decoder thread; we don't want to remove KV slots
    // mid-batch.
    std::lock_guard<std::mutex> lock(impl_->decode_mutex);

    bool ok = llama_memory_seq_rm(
        memory,
        seq->seq_id,
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

int CotabbyInferenceEngine::getKVPositionCount(int32_t sequence_id) const {
    const SequenceState* seq = impl_->findSequence(sequence_id);
    return seq ? seq->kv_position_count : 0;
}

void CotabbyInferenceEngine::setForceWordContinuation(int32_t sequence_id, bool enabled) {
    if (!impl_) return;
    SequenceState* seq = impl_->findSequence(sequence_id);
    if (seq) {
        seq->force_word_continuation = enabled;
    }
}

// ---------------------------------------------------------------------------
// KV state snapshot / restore
//
// Thin wrappers over llama's single-sequence state copy. They serialize with the decoder thread
// via decode_mutex so a snapshot or restore never races an in-flight llama_decode. Callers pair a
// snapshot with the KV position count they observed and pass it back on restore, so this engine's
// own position bookkeeping stays consistent with the restored KV cache.
// ---------------------------------------------------------------------------

size_t CotabbyInferenceEngine::snapshotSize(int32_t sequence_id) const {
    if (!impl_ || !impl_->shared_ctx) return 0;
    const SequenceState* seq = impl_->findSequence(sequence_id);
    if (!seq) return 0;
    return llama_state_seq_get_size(impl_->shared_ctx, seq->seq_id);
}

size_t CotabbyInferenceEngine::snapshotSequence(int32_t sequence_id, uint8_t* dst, size_t capacity) {
    if (!impl_ || !impl_->shared_ctx || !dst) return 0;
    SequenceState* seq = impl_->findSequence(sequence_id);
    if (!seq) return 0;
    std::lock_guard<std::mutex> lock(impl_->decode_mutex);
    return llama_state_seq_get_data(impl_->shared_ctx, dst, capacity, seq->seq_id);
}

bool CotabbyInferenceEngine::restoreSequence(int32_t sequence_id, const uint8_t* src,
                                             size_t size, int position_count) {
    if (!impl_ || !impl_->shared_ctx || !src) return false;
    SequenceState* seq = impl_->findSequence(sequence_id);
    if (!seq) return false;
    std::lock_guard<std::mutex> lock(impl_->decode_mutex);
    const size_t read = llama_state_seq_set_data(impl_->shared_ctx, src, size, seq->seq_id);
    if (read == 0) return false;
    seq->kv_position_count = position_count;
    // The restored blob invalidates any seed/pending token captured before; force the caller to
    // re-prime via decodePrompt before the next sampleNext.
    seq->has_seed_token = false;
    seq->has_pending_input = false;
    return true;
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
