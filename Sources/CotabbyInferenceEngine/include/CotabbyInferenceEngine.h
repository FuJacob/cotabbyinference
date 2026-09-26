#pragma once
#include <cstdint>
#include <vector>
#include <swift/bridging>

struct SamplingConfig {
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    float repetition_penalty;
    uint32_t seed;

    // When true, tokens that introduce a line break are masked from sampling so single-line
    // fields never receive a multi-line completion. Defaults to false to preserve prior behavior.
    bool single_line = false;
};

struct SWIFT_SELF_CONTAINED SampleResult {
    int32_t token;
    const char* piece;
    int piece_length;
    bool is_eos;
    bool was_cancelled;
    // Log-probability of the chosen token under the raw model distribution (<= 0). Used as a
    // confidence signal; 0 for the EOS/cancelled cases where it carries no meaning.
    float logprob;
    // True when the single most-likely token of the raw distribution this token was sampled
    // from is an end-of-generation token. Stochastic sampling can draw past the point where
    // the model wants to stop; this flag lets callers detect that stop intent on the very
    // step it appears, even when the sampled token is something else. Appended after the
    // existing fields so Swift call sites that only read members keep compiling.
    bool argmax_is_eog;
};

enum class EngineStatus : int {
    ok = 0,
    error = 1,
    cancelled = 2,
    not_loaded = 3,
};

// A snapshot for logs/tests; no prompt or generated content crosses this diagnostics boundary.
struct CacheDiagnostics {
    int decoded_token_count = 0;
    int checkpoint_position = 0;
    uint64_t checkpoint_bytes = 0;
    int last_restore_replayed_tokens = 0;
    bool uses_partial_checkpoint = false;
};

class CotabbyInferenceEngine {
public:
    CotabbyInferenceEngine();
    ~CotabbyInferenceEngine();

    // Move-only (owns native resources via PIMPL)
    CotabbyInferenceEngine(CotabbyInferenceEngine&& other) noexcept;
    CotabbyInferenceEngine& operator=(CotabbyInferenceEngine&&) = delete;
    CotabbyInferenceEngine(const CotabbyInferenceEngine&) = delete;
    CotabbyInferenceEngine& operator=(const CotabbyInferenceEngine&) = delete;

    // Model lifecycle
    EngineStatus loadModel(const char* path, int gpu_layers,
                           int context_window_tokens, int batch_size);
    void unloadModel();

    // Sequence lifecycle
    // The engine owns at most one sequence. `createSequence` returns -1 while another sequence is
    // alive; destroy the current sequence before creating its replacement.
    int32_t createSequence(SamplingConfig config);
    void destroySequence(int32_t sequence_id);

    // Tokenization (thread-safe, read-only on vocab)
    std::vector<int32_t> tokenize(const char* text, int text_length) const;
    // Plain decoded bytes for one printable token; special/unknown tokens return empty.
    std::vector<uint8_t> tokenPiece(int32_t token) const;
    // The next continuation must reproduce these already-typed bytes before adding new text.
    // Set before decodePrompt. Matching may span several byte-fallback tokens.
    void setCompletionPrefix(int32_t sequence_id, const uint8_t* bytes, int length);
    // Prompt decoding
    EngineStatus decodePrompt(int32_t sequence_id,
                              const int32_t* tokens, int token_count,
                              int start_position);

    // Sampling
    SampleResult sampleNext(int32_t sequence_id);

    // KV cache management
    bool trimKV(int32_t sequence_id, int keep_positions);

    // Constrains the FIRST token of the next generation on `sequence_id` to continue the current
    // word: tokens whose decoded text begins with whitespace are masked for that one token, then
    // the constraint clears. Set this before `decodePrompt`, which samples the first (seed) token.
    void setForceWordContinuation(int32_t sequence_id, bool enabled);

    // Controls whether `SampleResult.logprob` is computed for this sequence. Defaults to true
    // (the historical behavior). The log-probability costs two O(vocab-size) passes per generated
    // token, so callers whose confidence gating is disabled should pass false to skip it; results
    // then report `logprob == 0`.
    void setComputeLogprob(int32_t sequence_id, bool enabled);

    // Cancellation (thread-safe, non-blocking)
    void cancelSequence(int32_t sequence_id);

    // Diagnostics
    int getContextWindowTokens() const;
    int getBatchSize() const;
    int getThreadCount() const;
    int getGPULayerCount() const;
    CacheDiagnostics getCacheDiagnostics(int32_t sequence_id) const;

private:
    struct Impl;
    Impl* impl_;
};
