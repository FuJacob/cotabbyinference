#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cotabby {

/// Read-only index of decoded vocabulary bytes, owned once by the loaded engine.
///
/// A caret can split a token: fixing the existing tokenization of "sched" prevents the model
/// from choosing a token for "schedule". The caller backs up a token and asks this index which
/// replacements can reproduce the exact bytes already typed. This is byte matching, deliberately
/// independent of UTF-8 character boundaries, so byte-fallback tokenizers work as well.
class TokenHealingVocabulary {
public:
    struct Entry {
        int32_t token;
        std::string piece;
        // Derived once at reset, so single-line filtering adds no per-step string scanning.
        bool contains_line_break = false;
    };

    void reset(std::vector<Entry> entries);
    void clear();

    /// Returns tokens covering the complete remaining prefix, including exact replay. Falls back
    /// to shorter matching pieces only when no covering token exists (e.g. byte-fallback text).
    /// Empty/control tokens must not be indexed: they would make no progress through the prefix.
    std::vector<int32_t> matchingTokens(std::string_view prefix, bool single_line = false) const;

private:
    std::vector<Entry> entries_;
};

/// Per-request state for replaying the prompt's backed-up bytes before exposing new text.
/// The engine owns this beside its sampler; resetting a generation also resets this value.
class TokenPrefix {
public:
    void reset(std::string bytes);
    void clear();
    bool empty() const;
    std::string_view remaining() const;

    /// Returns false on a mismatching sample without advancing. Callers fail closed rather than
    /// exposing an altered spelling of text that is already in the user's document.
    bool consume(std::string_view piece);

private:
    std::string bytes_;
    size_t consumed_ = 0;
};

} // namespace cotabby
