# CotabbyInference

CotabbyInference is Cotabby's narrow C++ boundary around
[llama.cpp](https://github.com/ggml-org/llama.cpp). It owns the mapped GGUF model, one llama
context, one autocomplete sequence, sampler state, KV-cache mutation, cancellation, and the
token-level signals consumed by the macOS app.

The package deliberately does not own prompts, request identity, streaming order, normalization,
editor focus, overlays, or insertion. Those remain product responsibilities in Cotabby.

## Current Architecture

~~~text
Cotabby LlamaRuntimeCore
  -> CotabbyInferenceEngine
      -> one mapped llama_model
      -> one llama_context / KV allocation
      -> zero or one SequenceState using llama seq_id 0
      -> one sampler chain
~~~

Cotabby serializes generation and prefill through its runtime lock, so the middleware does not
reserve unused secondary sequence capacity or run a batching worker. Prompt decode, feedback decode,
KV trim, and sequence destruction use one native context mutex. Cancellation is the intentional
cross-thread operation and uses an atomic flag, rearmed only after successful cache restoration.

The public sequence ID changes whenever a sequence is recreated even though llama's internal slot
is always zero. That prevents a late cancellation from accidentally targeting a replacement
sequence.

## Responsibilities

- Load and unload one memory-mapped GGUF model.
- Allocate one context with exactly the configured token window.
- Tokenize Cotabby's base-model continuation prompts.
- Build the sampler chain and precompute invalid-token, line-break, and word-boundary masks.
- Decode a prompt and capture its first seed token while the final logits row is live.
- Return one sampled UTF-8 piece at a time.
- Expose sampled EOS, raw-argmax EOG intent, cancellation, and optional log-probability.
- Trim the sequence KV cache for verified prefix reuse.
- Release sampler, context, model, and llama backend resources in order.

## Swift Package

The package contains:

- `llama-cpp`: checksum-pinned binary build `b9310`;
- `CotabbyInferenceEngine`: the C++ library imported by Cotabby through Swift C++ interop;
- `CotabbyInferenceTests`: no-model contract tests and optional GGUF-backed integration tests.

Cotabby currently consumes the package's `main` branch through `project.yml` and records an exact
revision in `Package.resolved`.

## Basic Usage

~~~swift
import CotabbyInference

var engine = CotabbyInferenceEngine()
guard engine.loadModel("/path/to/model.gguf", -1, 2048, 512) == .ok else {
    fatalError("Model load failed")
}
defer { engine.unloadModel() }

let config = SamplingConfig(
    temperature: 0.1,
    top_k: 20,
    top_p: 0.7,
    min_p: 0.08,
    repetition_penalty: 1.05,
    seed: 0x00C0_FFEE,
    single_line: true
)

let sequenceID = engine.createSequence(config)
guard sequenceID >= 0 else {
    fatalError("A sequence is already active or the model is unavailable")
}
defer { engine.destroySequence(sequenceID) }

let prompt = "The quick brown fox"
var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
guard engine.decodePrompt(sequenceID, &tokens, Int32(tokens.count), 0) == .ok else {
    fatalError("Prompt decode failed")
}

// The caller owns the generation budget.
var completionBytes: [UInt8] = []
for _ in 0 ..< 8 {
    let result = engine.sampleNext(sequenceID)
    if result.is_eos || result.was_cancelled { break }

    if let piece = result.piece, result.piece_length > 0 {
        completionBytes += Array(
            UnsafeBufferPointer(
                start: UnsafeRawPointer(piece).assumingMemoryBound(to: UInt8.self),
                count: Int(result.piece_length)
            )
        )
    }
}
if let text = String(bytes: completionBytes, encoding: .utf8) {
    print(text)
}
~~~

`SampleResult.piece` is borrowed sequence storage. Copy it before another sampling call or sequence
destruction. Its bytes can end inside a UTF-8 scalar; streaming clients must accumulate bytes before
converting to text instead of discarding undecodable individual pieces.

## Generation Semantics

`decodePrompt` decodes the prompt and immediately samples one seed token. The first `sampleNext`
returns that saved seed without another decode. Each later call feedback-decodes the previously
returned token into KV and samples the next token.

Cotabby controls the maximum token count in Swift. The engine controls token selection and reports:

- `is_eos`: the sampled token is an end-of-generation token;
- `was_cancelled`: the native sequence cancellation flag was observed;
- `argmax_is_eog`: the raw model distribution most strongly wanted to stop even if stochastic
  sampling selected visible text;
- `logprob`: the selected token's raw-model log-probability when enabled.

## Caret Token Healing

`tokenPiece(token)` returns printable bytes for planning a completion. When the final prompt token
exactly matches the typed suffix, the client can remove that token and pass its bytes to
`setCompletionPrefix(sequence, bytes, length)` before `decodePrompt`. The sampler then admits only
tokens compatible with that prefix, including byte-fallback tokens that cover part of it. This
allows `sched` to participate in the token for `schedule` without altering the writer's text.

`TokenHealingVocabulary` is built once per loaded model; the per-sequence `TokenPrefix` owns only
the unconsumed replay bytes. The caller strips exactly those replayed bytes, preserves incomplete
UTF-8 fragments, and budgets replay separately from visible continuation. Cotabby caps replay at
16 bytes/tokens. While constrained, the legacy whitespace mask is bypassed and `argmax_is_eog`
is false. No EOS/control tokens can complete an unfinished replay. Clear the prefix for requests
that do not heal; sampler settings and cache lifetime do not imply the next request's prefix.

## KV Reuse and Cancellation

`trimKV` restores a prefix in slot zero and invalidates its pending seed/feedback token. Ordinary
attention uses direct suffix removal. Recurrent, hybrid, and sliding-window models use one
`PARTIAL_ONLY | ON_DEVICE` checkpoint near the prompt tail, retaining full attention KV in place.
The checkpoint's tensor memory is capped at 128 MiB and restoration replays at most eight prompt
tokens without sampling. Device storage belongs to the llama context's slot-zero checkpoint and
is released when that context unloads. Checkpoint metadata belongs to `SequenceState`.

A request that edits before the saved checkpoint, a very short prompt without a nonempty checkpoint,
or a model exceeding the cap returns a cache miss; callers rebuild that request. A miss must not
permanently disable reuse for the model. Saving a newer near-caret checkpoint intentionally gives
up deeper backspace history. Gemma's sliding-window cache supports suffix removal, but restoring
its saved window also protects rows evicted during prediction. No model-family-name heuristics
are used: llama model metadata selects the partial-state path.

`decodePrompt` resets the sampler and accepts only committed prompt tokens. Discarded predictions
therefore never pollute penalties or RNG state; `llama_sampler_sample` already accepts its sample,
so the wrapper must not accept it a second time. A sampled seed is not in KV until feedback decode.
Every successful decode advances tracked positions even when the next result is EOS/cancelled.
A failed native decode invalidates cache reuse until the sequence is destroyed; an unchanged
tracked position must never make the equal-position trim shortcut certify uncertain memory.

`getCacheDiagnostics(sequence)` exposes actual token position, checkpoint tensor bytes/position,
restoration replay count, and whether partial checkpoints are required. It contains no text.
Cotabby independently validates field continuity, byte/token prefix, and sampling compatibility.

`cancelSequence` is thread-safe and nonblocking. Prompt decode checks cancellation between chunks;
sample generation checks before work and after feedback decode. An active llama decode is not
preempted mid-call. Successful `trimKV` clears the cancellation flag only after memory is valid;
a failed restoration leaves the sequence cancelled and requires destruction. Cotabby closes its
operation-specific cancellation target before restoring, so a late task cancellation cannot poison
the next request that reuses the same sequence. Destruction and cancellation serialize ownership
through the sequence mutex.

## Testing

Run compile-time and no-model contracts:

~~~bash
swift test
~~~

Run the full native path with a local GGUF:

~~~bash
COTABBY_TEST_MODEL_PATH=/absolute/path/model.gguf swift test
~~~

If the default Xcode-backed SwiftPM runner fails code signing because of local Finder metadata,
`swift test --build-system native` runs the same tests with SwiftPM's native runner.

Run the deterministic vocabulary-prefix tests without downloading a model or linking llama:

~~~bash
clang++ -std=c++17 -I Sources/CotabbyInferenceEngine \
  Sources/CotabbyInferenceEngine/TokenHealing.cpp Tests/TokenHealingTests.cpp \
  -o /tmp/cotabby-token-healing-tests
/tmp/cotabby-token-healing-tests
~~~

The model-backed suite covers single-sequence admission/replacement, prompt decode, sampling, KV
trim, cancellation, mid-word continuation, optional log-probability, scaffolding-token masking, and
argmax-EOG behavior. CI does not currently provide a GGUF, so these tests skip there unless the
environment variable is configured. New coverage compares cold/restored token output, cancellation
rearming, exact replay of unfinished words and trailing whitespace, and checkpoint/replay bounds.
An oversized-prompt regression also verifies failed-decode invalidation and fresh-sequence recovery.

`testWarmPromptDecodeReportsLatency` prints medians for 32-, 214-, and 838-token prompts (the exact
counts vary by tokenizer), excluding the first pass and asserting no wall-clock threshold. On one
local Qwen3.5-0.8B-Base Q6_K run, cold/warm prompt processing measured about 40/26, 89/25, and 284/25
milliseconds with a 20.2 MB checkpoint. This is native prompt work, not keystroke-to-visible-word
latency, and is not a Gemma or cross-hardware speed claim. Host-memory checkpoints were slower in
the same experiment; keeping tensor copies on device was necessary for the measured improvement.

## Requirements

- macOS 14+
- Swift 6.2+
- Xcode 26+

## License

[MIT](LICENSE)
