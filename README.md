# TabbyInference

A C++ inference engine wrapping [llama.cpp](https://github.com/ggml-org/llama.cpp) for on-device LLM inference on macOS, designed for [Tabby](https://github.com/nicktrienenern/Tabby).

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Requirements

- macOS 14+
- Swift 6.2+
- Xcode 26+

## Installation

Add TabbyInference to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/FuJacob/tabbyinference.git", from: "0.1.0"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "TabbyInference", package: "tabbyinference"),
        ],
        swiftSettings: [
            .interoperabilityMode(.Cxx),
        ]
    ),
]
```

## Usage

```swift
import TabbyInference

var engine = TabbyInferenceEngine()

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

## Architecture

Each sequence gets its own `llama_context` and sampler chain -- fully independent lifetimes, clean cancellation, no shared decode mutex. The engine supports up to 4 concurrent sequences.

The public C++ API uses PIMPL to keep `llama.h` out of the public header, so consumers only link against the `TabbyInference` module.

## License

[MIT](LICENSE)
