#pragma once
#include <cstddef>
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
    // Like `tokenize`, but the caller controls BOS/EOS injection and whether
    // special/control tokens in the text (e.g. chat-template markers like
    // <|im_start|>) are recognized as their token IDs instead of plain text.
    // The plain `tokenize` keeps `parse_special = false` for backward
    // compatibility; the chat-template path needs `true` so rendered markers
    // tokenize correctly.
    std::vector<int32_t> tokenizeWithOptions(const char* text, int text_length,
                                             bool add_special,
                                             bool parse_special) const;
    int detokenize(int32_t token, char* buffer, int buffer_size) const;

    // Chat templates
    //
    // `hasChatTemplate` reports whether the loaded model ships a chat template
    // in its GGUF metadata. Instruct models (Qwen, Gemma, Llama) do; raw base
    // models do not. Callers use this to decide between the structured
    // chat-template prompt path and the legacy raw-continuation path so a
    // user-supplied base model keeps working.
    bool hasChatTemplate() const;
    // Renders a system + user turn through the model's built-in chat template
    // into `buffer`. `add_assistant` appends the assistant-turn opening marker
    // so the model continues as the assistant. Returns:
    //   > 0 : number of bytes written (<= buffer_size) — the formatted prompt.
    //   < 0 : -(required buffer size); the buffer was too small, retry at that size.
    //   = 0 : no model, no template, or render failure — caller falls back to raw.
    //
    // Autocomplete needs exactly one system turn (rules + context) and one user
    // turn (the text to continue), so the signature takes those two directly
    // rather than a message array. This buffer-based C ABI mirrors `detokenize`
    // and deliberately avoids std::string / struct parameter and return types,
    // so it bridges cleanly into the Swift objcxx interop mode the app target
    // uses (where a std::string return does not bridge).
    int applyChatTemplate(const char* system_text,
                          const char* user_text,
                          bool add_assistant,
                          char* buffer,
                          int buffer_size) const;

    // Prompt decoding
    EngineStatus decodePrompt(int32_t sequence_id,
                              const int32_t* tokens, int token_count,
                              int start_position);

    // Sampling
    SampleResult sampleNext(int32_t sequence_id);

    // KV cache management
    bool trimKV(int32_t sequence_id, int keep_positions);
    int getKVPositionCount(int32_t sequence_id) const;

    // Constrains the FIRST token of the next generation on `sequence_id` to continue the current
    // word: tokens whose decoded text begins with whitespace are masked for that one token, then
    // the constraint clears. Set this before `decodePrompt`, which samples the first (seed) token.
    void setForceWordContinuation(int32_t sequence_id, bool enabled);

    // Single-sequence KV state snapshot/restore. `snapshotSize` reports the buffer size needed,
    // `snapshotSequence` copies the sequence's KV state into `dst` (returns bytes written, 0 on
    // failure), and `restoreSequence` loads a previously captured blob back and sets the KV
    // position to `position_count` (returns false on failure). Blobs are model- and context-
    // specific; never restore one across a model reload.
    size_t snapshotSize(int32_t sequence_id) const;
    size_t snapshotSequence(int32_t sequence_id, uint8_t* dst, size_t capacity);
    bool restoreSequence(int32_t sequence_id, const uint8_t* src, size_t size, int position_count);

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
