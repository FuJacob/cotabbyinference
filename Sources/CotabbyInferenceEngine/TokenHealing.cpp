#include "TokenHealing.h"

#include <algorithm>
#include <utility>

namespace cotabby {

void TokenHealingVocabulary::reset(std::vector<Entry> entries) {
    entries.erase(std::remove_if(entries.begin(), entries.end(), [](const Entry& entry) {
        return entry.piece.empty();
    }), entries.end());
    for (auto& entry : entries) {
        entry.contains_line_break = entry.piece.find_first_of("\r\n") != std::string::npos;
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
        return lhs.piece < rhs.piece;
    });
    entries_ = std::move(entries);
}

void TokenHealingVocabulary::clear() {
    entries_ = {};
}

std::vector<int32_t> TokenHealingVocabulary::matchingTokens(std::string_view prefix, bool single_line) const {
    std::vector<int32_t> matches;
    if (prefix.empty()) return matches;

    const auto lowerBound = [&](std::string_view value) {
        return std::lower_bound(entries_.begin(), entries_.end(), value,
            [](const Entry& entry, std::string_view key) { return entry.piece < key; });
    };

    // Tokens that finish the replay and may also introduce new text form one contiguous range.
    // Binary search avoids rescanning hundreds of thousands of vocabulary strings per keystroke.
    for (auto it = lowerBound(prefix); it != entries_.end(); ++it) {
        if (std::string_view(it->piece).substr(0, prefix.size()) != prefix) break;
        if (!single_line || !it->contains_line_break) matches.push_back(it->token);
    }

    // Prefer a token that accounts for ALL known bytes. A short token such as " i" has high
    // probability because it can start many words that contradict the already-typed " int".
    // Letting it compete with " intelligence", then forcing "nt" afterward, ranks incompatible
    // paths before their likelihood is known. Covering tokens can be compared at the same prefix
    // boundary instead. Exact replay remains allowed, so a completed word can end normally.
    // Ignore line-break tokens before choosing this branch: a covering token that the sampler
    // later forbids must not prevent a valid shorter spelling from being replayed.
    if (!matches.empty()) return matches;

    // Multi-token/byte-fallback text may have no covering token. Only then allow shorter pieces;
    // the next step applies this same rule to the remaining bytes. Include duplicate spellings.
    for (size_t length = 1; length < prefix.size(); ++length) {
        const auto fragment = prefix.substr(0, length);
        for (auto it = lowerBound(fragment); it != entries_.end() && it->piece == fragment; ++it) {
            if (!single_line || !it->contains_line_break) matches.push_back(it->token);
        }
    }
    return matches;
}

void TokenPrefix::reset(std::string bytes) {
    bytes_ = std::move(bytes);
    consumed_ = 0;
}

void TokenPrefix::clear() {
    bytes_.clear();
    consumed_ = 0;
}

bool TokenPrefix::empty() const {
    return consumed_ == bytes_.size();
}

std::string_view TokenPrefix::remaining() const {
    return std::string_view(bytes_).substr(consumed_);
}

bool TokenPrefix::consume(std::string_view piece) {
    if (empty()) return true;
    if (piece.empty()) return false;
    const auto count = std::min(remaining().size(), piece.size());
    if (remaining().substr(0, count) != piece.substr(0, count)) return false;
    consumed_ += count;
    return true;
}

} // namespace cotabby
