#pragma once
#include <cstdint>
#include <vector>
#include <swift/bridging>

struct SamplingConfig {
    int max_prediction_tokens;
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    float repetition_penalty;
    uint32_t seed;
    // CPU threads this sequence's context may use during decode. 0 (or negative) keeps the
    // engine default of all hardware threads. Set a smaller budget on background sequences (e.g.
    // the visual-context summarizer) so they decode concurrently with latency-critical
    // autocomplete instead of oversubscribing every core and starving it.
    int thread_count;
};

struct SWIFT_SELF_CONTAINED SampleResult {
    int32_t token;
    const char* piece;
    int piece_length;
    bool is_eos;
    bool was_cancelled;
};

enum class EngineStatus : int {
    ok = 0,
    error = 1,
    cancelled = 2,
    not_loaded = 3,
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
    bool isModelLoaded() const;

    // Sequence lifecycle
    int32_t createSequence(SamplingConfig config);
    void destroySequence(int32_t sequence_id);

    // Tokenization (thread-safe, read-only on vocab)
    std::vector<int32_t> tokenize(const char* text, int text_length) const;
    int detokenize(int32_t token, char* buffer, int buffer_size) const;

    // Prompt decoding
    EngineStatus decodePrompt(int32_t sequence_id,
                              const int32_t* tokens, int token_count,
                              int start_position);

    // Sampling
    SampleResult sampleNext(int32_t sequence_id);

    // KV cache management
    bool trimKV(int32_t sequence_id, int keep_positions);
    int getKVPositionCount(int32_t sequence_id) const;

    // Cancellation (thread-safe, non-blocking)
    void cancelSequence(int32_t sequence_id);

    // Diagnostics
    int getContextWindowTokens() const;
    int getBatchSize() const;
    int getThreadCount() const;
    int getGPULayerCount() const;

private:
    struct Impl;
    Impl* impl_;
};
