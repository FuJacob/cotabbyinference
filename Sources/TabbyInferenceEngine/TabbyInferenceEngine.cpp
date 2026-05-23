#include "TabbyInferenceEngine.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>

#include <llama/llama.h>
#include <llama/ggml.h>

static void silenced_log_callback(ggml_log_level, const char*, void*) {}

// ---------------------------------------------------------------------------
// Per-sequence state (one llama_context + sampler per sequence)
// ---------------------------------------------------------------------------

struct SequenceState {
    llama_context* context = nullptr;
    llama_sampler* sampler = nullptr;
    SamplingConfig config{};
    int kv_position_count = 0;
    std::atomic<bool> cancelled{false};
    std::string last_piece;

    ~SequenceState() {
        if (sampler) { llama_sampler_free(sampler); }
        if (context) { llama_free(context); }
    }

    SequenceState() = default;
    SequenceState(SequenceState&& o) noexcept
        : context(o.context), sampler(o.sampler), config(o.config),
          kv_position_count(o.kv_position_count),
          cancelled(o.cancelled.load()), last_piece(std::move(o.last_piece)) {
        o.context = nullptr;
        o.sampler = nullptr;
    }
    SequenceState& operator=(SequenceState&&) = delete;
    SequenceState(const SequenceState&) = delete;
    SequenceState& operator=(const SequenceState&) = delete;
};

// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------

struct TabbyInferenceEngine::Impl {
    static constexpr int MAX_SEQUENCES = 4;
    static constexpr llama_seq_id SEQ_ID = 0;

    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    bool backend_initialized = false;
    std::string model_path;

    int context_window_tokens = 0;
    int batch_size = 0;
    int thread_count = 0;
    int gpu_layer_count = 0;

    mutable std::mutex sequences_mutex;
    std::unordered_map<int32_t, SequenceState> sequences;
    int32_t next_sequence_id = 1;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

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

        // 1. Repetition penalty
        if (cfg.repetition_penalty > 1.0f) {
            auto* pen = llama_sampler_init_penalties(
                64,
                cfg.repetition_penalty,
                0.0f,
                0.0f
            );
            if (pen) llama_sampler_chain_add(chain, pen);
        }

        // 2a. Stochastic path
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

        // 2b. Greedy path
        } else {
            auto* greedy = llama_sampler_init_greedy();
            if (greedy) llama_sampler_chain_add(chain, greedy);
        }

        return chain;
    }

    void destroyAllSequences() {
        std::lock_guard<std::mutex> lock(sequences_mutex);
        sequences.clear();
    }
};

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

TabbyInferenceEngine::TabbyInferenceEngine() : impl_(new Impl) {}

TabbyInferenceEngine::TabbyInferenceEngine(TabbyInferenceEngine&& other) noexcept
    : impl_(other.impl_) {
    other.impl_ = nullptr;
}

TabbyInferenceEngine::~TabbyInferenceEngine() {
    if (impl_) {
        unloadModel();
        delete impl_;
    }
}

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

EngineStatus TabbyInferenceEngine::loadModel(const char* path, int gpu_layers,
                                             int context_window_tokens,
                                             int batch_size) {
    if (!impl_ || !path) return EngineStatus::error;

    // Idempotent for same path
    if (impl_->model && impl_->model_path == path) {
        return EngineStatus::ok;
    }

    // Different model already loaded — tear down first
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

    return EngineStatus::ok;
}

void TabbyInferenceEngine::unloadModel() {
    if (!impl_) return;
    impl_->destroyAllSequences();

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

bool TabbyInferenceEngine::isModelLoaded() const {
    return impl_ && impl_->model != nullptr;
}

// ---------------------------------------------------------------------------
// Sequence lifecycle
// ---------------------------------------------------------------------------

int32_t TabbyInferenceEngine::createSequence(SamplingConfig config) {
    if (!impl_->model) return -1;

    {
        std::lock_guard<std::mutex> lock(impl_->sequences_mutex);
        if (static_cast<int>(impl_->sequences.size()) >= Impl::MAX_SEQUENCES) {
            return -1;
        }
    }

    // Build context
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(impl_->context_window_tokens);
    ctx_params.n_batch = static_cast<uint32_t>(impl_->batch_size);
    ctx_params.n_ubatch = static_cast<uint32_t>(impl_->batch_size);
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = static_cast<int32_t>(impl_->thread_count);
    ctx_params.n_threads_batch = static_cast<int32_t>(impl_->thread_count);
    ctx_params.offload_kqv = true;

    llama_context* ctx = llama_init_from_model(impl_->model, ctx_params);
    if (!ctx) return -1;

    // Build sampler
    llama_sampler* sampler = impl_->buildSampler(config);
    if (!sampler) {
        llama_free(ctx);
        return -1;
    }

    SequenceState state;
    state.context = ctx;
    state.sampler = sampler;
    state.config = config;

    std::lock_guard<std::mutex> lock(impl_->sequences_mutex);
    int32_t id = impl_->next_sequence_id++;
    impl_->sequences.emplace(id, std::move(state));
    return id;
}

void TabbyInferenceEngine::destroySequence(int32_t sequence_id) {
    std::lock_guard<std::mutex> lock(impl_->sequences_mutex);
    impl_->sequences.erase(sequence_id);
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

std::vector<int32_t> TabbyInferenceEngine::tokenize(const char* text,
                                                     int text_length) const {
    if (!impl_->vocab || !text || text_length <= 0) {
        return {};
    }

    bool add_bos = llama_vocab_get_add_bos(impl_->vocab);
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
        // n < 0 means buffer too small, -n is the required capacity
        capacity = std::max(capacity * 2, -n);
    }
}

int TabbyInferenceEngine::detokenize(int32_t token, char* buffer,
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
// ---------------------------------------------------------------------------

EngineStatus TabbyInferenceEngine::decodePrompt(int32_t sequence_id,
                                                 const int32_t* tokens,
                                                 int token_count,
                                                 int start_position) {
    if (!impl_->model) return EngineStatus::not_loaded;
    if (!tokens || token_count <= 0) return EngineStatus::ok;

    SequenceState* seq = impl_->findSequence(sequence_id);
    if (!seq) return EngineStatus::error;

    if (seq->cancelled.load(std::memory_order_acquire)) {
        return EngineStatus::cancelled;
    }

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
                batch.seq_id[i][0] = Impl::SEQ_ID;
            }
            // Logits only for the very last token of the entire prompt
            bool is_last = (chunk_end == end && i == chunk_size - 1);
            batch.logits[i] = is_last ? 1 : 0;
        }

        if (llama_decode(seq->context, batch) != 0) {
            llama_batch_free(batch);
            return EngineStatus::error;
        }

        cursor = chunk_end;
    }

    llama_batch_free(batch);
    seq->kv_position_count = total_end_position;
    return EngineStatus::ok;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

SampleResult TabbyInferenceEngine::sampleNext(int32_t sequence_id) {
    SampleResult result{};
    result.token = 0;
    result.piece = nullptr;
    result.piece_length = 0;
    result.is_eos = false;
    result.was_cancelled = false;

    if (!impl_->model || !impl_->vocab) {
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

    // Sample
    llama_token next_token = llama_sampler_sample(seq->sampler, seq->context, -1);

    // Check EOS / EOG
    if (next_token == llama_vocab_eos(impl_->vocab) ||
        llama_vocab_is_eog(impl_->vocab, next_token)) {
        result.token = next_token;
        result.is_eos = true;
        return result;
    }

    // Detokenize into the sequence's persistent buffer
    seq->last_piece.resize(64);
    while (true) {
        int written = llama_token_to_piece(
            impl_->vocab,
            next_token,
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

    llama_sampler_accept(seq->sampler, next_token);

    // Decode the sampled token to advance KV cache
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = next_token;
    batch.pos[0] = static_cast<llama_pos>(seq->kv_position_count);
    batch.n_seq_id[0] = 1;
    if (batch.seq_id && batch.seq_id[0]) {
        batch.seq_id[0][0] = Impl::SEQ_ID;
    }
    batch.logits[0] = 1;

    int decode_status = llama_decode(seq->context, batch);
    llama_batch_free(batch);

    if (decode_status != 0) {
        result.is_eos = true;
        return result;
    }

    seq->kv_position_count++;

    result.token = next_token;
    result.piece = seq->last_piece.c_str();
    result.piece_length = static_cast<int>(seq->last_piece.size());
    result.is_eos = false;
    result.was_cancelled = false;
    return result;
}

// ---------------------------------------------------------------------------
// KV cache management
// ---------------------------------------------------------------------------

bool TabbyInferenceEngine::trimKV(int32_t sequence_id, int keep_positions) {
    SequenceState* seq = impl_->findSequence(sequence_id);
    if (!seq) return false;

    llama_memory_t memory = llama_get_memory(seq->context);
    if (!memory) return false;

    bool ok = llama_memory_seq_rm(
        memory,
        Impl::SEQ_ID,
        static_cast<llama_pos>(keep_positions),
        -1
    );

    if (ok) {
        seq->kv_position_count = keep_positions;
    }
    return ok;
}

int TabbyInferenceEngine::getKVPositionCount(int32_t sequence_id) const {
    const SequenceState* seq = impl_->findSequence(sequence_id);
    return seq ? seq->kv_position_count : 0;
}

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

void TabbyInferenceEngine::cancelSequence(int32_t sequence_id) {
    SequenceState* seq = impl_->findSequence(sequence_id);
    if (seq) {
        seq->cancelled.store(true, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

int TabbyInferenceEngine::getContextWindowTokens() const {
    return impl_->context_window_tokens;
}

int TabbyInferenceEngine::getBatchSize() const {
    return impl_->batch_size;
}

int TabbyInferenceEngine::getThreadCount() const {
    return impl_->thread_count;
}

int TabbyInferenceEngine::getGPULayerCount() const {
    return impl_->gpu_layer_count;
}
