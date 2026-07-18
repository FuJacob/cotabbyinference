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
cross-thread operation and uses a one-way atomic flag.

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
for _ in 0 ..< 8 {
    let result = engine.sampleNext(sequenceID)
    if result.is_eos || result.was_cancelled { break }

    if let piece = result.piece, result.piece_length > 0 {
        let text = String(
            bytes: UnsafeBufferPointer(
                start: UnsafeRawPointer(piece).assumingMemoryBound(to: UInt8.self),
                count: Int(result.piece_length)
            ),
            encoding: .utf8
        ) ?? ""
        print(text, terminator: "")
    }
}
~~~

`SampleResult.piece` is borrowed sequence storage. Copy it before another sampling call or sequence
destruction.

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

## KV Reuse and Cancellation

`trimKV` removes a suffix from fixed llama sequence slot zero and invalidates any saved seed or
pending feedback token. Cotabby independently validates request continuity, UTF-8 prefix, token
prefix, and sampling compatibility before calling it.

`cancelSequence` is thread-safe and nonblocking. Prompt decode checks cancellation between chunks;
sample generation checks before work and after feedback decode. An active llama decode is not
preempted mid-call. Cotabby destroys a natively cancelled sequence because the flag is intentionally
one-way.

## Testing

Run compile-time and no-model contracts:

~~~bash
swift test
~~~

Run the full native path with a local GGUF:

~~~bash
COTABBY_TEST_MODEL_PATH=/absolute/path/model.gguf swift test
~~~

The model-backed suite covers single-sequence admission/replacement, prompt decode, sampling, KV
trim, cancellation, mid-word continuation, optional log-probability, scaffolding-token masking, and
argmax-EOG behavior. CI does not currently provide a GGUF, so these tests skip there unless the
environment variable is configured.

## Requirements

- macOS 14+
- Swift 6.2+
- Xcode 26+

## License

[MIT](LICENSE)
