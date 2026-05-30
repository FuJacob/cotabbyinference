#pragma once
#include <cstdint>
#include <string>
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
};

/// One message in a chat-template conversation, mirroring `llama_chat_message`.
/// Roles are the usual "system" / "user" / "assistant". Owned by the caller.
struct ChatMessage {
    std::string role;
    std::string content;
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
    // Renders `messages` through the model's built-in chat template and returns
    // the formatted prompt string. `add_assistant` appends the assistant-turn
    // opening marker so the model continues as the assistant. Returns an empty
    // string if no model is loaded, the model has no template, or formatting
    // fails — callers must treat empty as "fall back to the raw path".
    std::string applyChatTemplate(const ChatMessage* messages, int message_count,
                                  bool add_assistant) const;

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
