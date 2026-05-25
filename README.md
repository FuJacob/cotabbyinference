# CotabbyInference

A C++ middleware layer for running LLM inference on-device, built on top of [llama.cpp](https://github.com/ggml-org/llama.cpp). Designed as the inference backend for [Cotabby](https://github.com/nicktrienenern/Cotabby), a macOS AI assistant.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Why CotabbyInference?

llama.cpp is powerful but low-level. CotabbyInference sits on top of it and provides:

- **Concurrent sequences** -- run up to 4 independent inference streams at once (e.g. autocomplete + summarization), each with its own context, sampler, and sampling config. No shared decode mutex, no contention.
- **Per-sequence isolation** -- every sequence owns its own `llama_context` and sampler chain. One sequence can be cancelled, trimmed, or destroyed without affecting the others.
- **Thread-safe cancellation** -- cancel any running sequence from any thread via an atomic flag. The next decode or sample call returns immediately with a `cancelled` status.
- **KV cache control** -- trim the KV cache per-sequence to reuse prompt prefixes without re-decoding. Useful for autocomplete where the user keeps typing.
- **Clean Swift interop** -- the public header uses PIMPL to hide all llama.cpp internals. Swift consumers import a single `CotabbyInference` module with no transitive C++ dependencies. The engine class is move-only for `~Copyable` compatibility in Swift 6.2.
- **Zero-config GPU** -- pass `-1` for GPU layers and the engine offloads everything it can to Metal. No manual layer counting.

## Requirements

- macOS 14+
- Swift 6.2+
- Xcode 26+

## Installation

Add CotabbyInference to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/FuJacob/cotabbyinference.git", from: "0.2.0"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "CotabbyInference", package: "cotabbyinference"),
        ],
        swiftSettings: [
            .interoperabilityMode(.Cxx),
        ]
    ),
]
```

## Usage

```swift
import CotabbyInference

var engine = CotabbyInferenceEngine()

// Load a GGUF model (-1 for all GPU layers, 2048 context, 512 batch)
let status = engine.loadModel("/path/to/model.gguf", -1, 2048, 512)

// Tokenize
let prompt = "The quick brown fox"
let tokens = engine.tokenize(prompt, Int32(prompt.utf8.count))

// Create a sequence with sampling parameters
let config = SamplingConfig(
    max_prediction_tokens: 64,
    temperature: 0.7,
    top_k: 40,
    top_p: 0.95,
    min_p: 0.05,
    repetition_penalty: 1.1,
    seed: 0
)
let seqId = engine.createSequence(config)

// Decode prompt into KV cache
var tokenArray = Array(tokens)
engine.decodePrompt(seqId, &tokenArray, Int32(tokenArray.count), 0)

// Sample tokens
while true {
    let result = engine.sampleNext(seqId)
    if result.is_eos { break }

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

// Cleanup
engine.destroySequence(seqId)
engine.unloadModel()
```

### Running multiple sequences concurrently

Each sequence is fully independent -- different sampling configs, different prompts, different lifetimes:

```swift
// Autocomplete: low temperature, short output
let autocompleteConfig = SamplingConfig(
    max_prediction_tokens: 8, temperature: 0.1,
    top_k: 20, top_p: 0.7, min_p: 0.08,
    repetition_penalty: 1.05, seed: 42
)
let seqA = engine.createSequence(autocompleteConfig)

// Summary: higher temperature, longer output
let summaryConfig = SamplingConfig(
    max_prediction_tokens: 256, temperature: 0.5,
    top_k: 40, top_p: 0.95, min_p: 0.05,
    repetition_penalty: 1.4, seed: 0
)
let seqB = engine.createSequence(summaryConfig)

// Both run against the same loaded model, with separate contexts.
// Cancel one without affecting the other:
engine.cancelSequence(seqA)
```

## Architecture

CotabbyInference gives each sequence its own `llama_context` and sampler chain. This means:

- No shared decode mutex -- sequences never block each other
- Clean cancellation via per-sequence atomic flags
- Independent KV caches that can be trimmed separately
- Up to 4 concurrent sequences (the memory overhead per context is ~2-4 MB for small models)

The public C++ API uses PIMPL to keep all llama.cpp headers out of the public interface. Swift consumers link against the `CotabbyInference` module only -- no need to deal with `llama.h`, `ggml.h`, or any transitive C dependencies.

## License

[MIT](LICENSE)
