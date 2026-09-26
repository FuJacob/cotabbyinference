#include "../Sources/CotabbyInferenceEngine/TokenHealing.h"

#include <algorithm>
#include <cassert>
#include <iostream>

using cotabby::TokenHealingVocabulary;
using cotabby::TokenPrefix;

static std::vector<int32_t> sorted(std::vector<int32_t> tokens) {
    std::sort(tokens.begin(), tokens.end());
    return tokens;
}

int main() {
    TokenHealingVocabulary vocabulary;
    vocabulary.reset({
        {1, "sched"}, {2, "schedule"}, {3, "scheduling"}, {4, "sch"},
        {5, "send"}, {6, " schedule"}, {7, " "}, {8, ""}, {9, "sched"},
        {10, "\n"}, {11, "\n\n"}, {12, std::string("\xC3", 1)}, {13, "éclair"}
    });

    // Unfinished words can choose a whole-word token instead of being trapped in the original
    // tokenization. A completed word still permits replaying itself and predicting a space next.
    assert(sorted(vocabulary.matchingTokens("sched")) == std::vector<int32_t>({1, 2, 3, 9}));
    assert(sorted(vocabulary.matchingTokens("schedule")) == std::vector<int32_t>({2}));
    assert(sorted(vocabulary.matchingTokens(" ")) == std::vector<int32_t>({6, 7}));
    assert(sorted(vocabulary.matchingTokens("\n")) == std::vector<int32_t>({10, 11}));
    assert(sorted(vocabulary.matchingTokens("é")) == std::vector<int32_t>({13}));
    // A fragment must not lose its context just because a high-probability shorter token can
    // start it. Neither " i" nor " in" accounts for the known final 't' of " int".
    TokenHealingVocabulary technical;
    technical.reset({{20, " i"}, {21, " in"}, {22, " int"}, {23, " intelligence"}, {24, " internal"}});
    assert(sorted(technical.matchingTokens(" int")) == std::vector<int32_t>({22, 23, 24}));
    // A multi-token spelling that has no complete covering token still progresses byte by byte.
    assert(sorted(vocabulary.matchingTokens("schedx")) == std::vector<int32_t>({1, 4, 9}));
    assert(sorted(vocabulary.matchingTokens("éx")) == std::vector<int32_t>({12}));
    TokenHealingVocabulary line_breaks;
    line_breaks.reset({{30, "foo\n"}, {31, "f"}, {32, "oo"}, {33, "foo\r\n"}});
    assert(sorted(line_breaks.matchingTokens("foo")) == std::vector<int32_t>({30, 33}));
    // Single-line sampling masks the only covering tokens. Shorter safe pieces must remain
    // available, or all logits become -infinity before the sampler can complete the replay.
    assert(sorted(line_breaks.matchingTokens("foo", true)) == std::vector<int32_t>({31}));
    assert(sorted(line_breaks.matchingTokens("oo", true)) == std::vector<int32_t>({32}));
    assert(vocabulary.matchingTokens("xyz").empty());
    assert(vocabulary.matchingTokens("").empty());

    TokenPrefix prefix;
    prefix.reset("sched");
    assert(!prefix.consume("send"));
    assert(prefix.remaining() == "sched");
    assert(!prefix.consume(""));
    assert(prefix.consume("sch"));
    assert(prefix.remaining() == "ed");
    assert(prefix.consume("edule"));
    assert(prefix.empty());
    assert(prefix.consume(" tomorrow"));

    // Prefix state can end inside a UTF-8 code point. Neither comparison nor consumption may
    // decode an isolated byte into a replacement character or silently drop it.
    prefix.reset("é");
    assert(prefix.consume(std::string("\xC3", 1)));
    assert(prefix.consume(std::string("\xA9", 1)));
    assert(prefix.empty());
    prefix.reset(" ");
    assert(prefix.consume(" schedule"));
    assert(prefix.empty());
    prefix.reset("old");
    prefix.clear();
    assert(prefix.empty());
    vocabulary.clear();
    assert(vocabulary.matchingTokens("sched").empty());

    std::cout << "Token healing tests passed\n";
}
