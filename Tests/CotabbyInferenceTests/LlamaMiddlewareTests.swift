import XCTest
import CotabbyInference

final class LlamaMiddlewareTests: XCTestCase {

    func testEngineStartsUnloaded() {
        let engine = CotabbyInferenceEngine()
        XCTAssertFalse(engine.isModelLoaded())
    }

    func testUnloadWhenNothingLoadedIsIdempotent() {
        var engine = CotabbyInferenceEngine()  // var: unloadModel mutates
        engine.unloadModel()
        engine.unloadModel()
        XCTAssertFalse(engine.isModelLoaded())
    }

    func testLoadModelWithBadPathReturnsError() {
        var engine = CotabbyInferenceEngine()
        let status = engine.loadModel("/nonexistent/path.gguf", -1, 2048, 512)
        XCTAssertEqual(status, EngineStatus.error)
        XCTAssertFalse(engine.isModelLoaded())
    }

    func testCreateSequenceWithoutModelReturnsMinus1() {
        var engine = CotabbyInferenceEngine()
        let config = SamplingConfig(
            max_prediction_tokens: 8,
            temperature: 0.1,
            top_k: 20,
            top_p: 0.7,
            min_p: 0.08,
            repetition_penalty: 1.05,
            seed: 0
        )
        let seqId = engine.createSequence(config)
        XCTAssertEqual(seqId, -1)
    }

    func testDestroySequenceWithInvalidIdDoesNotCrash() {
        var engine = CotabbyInferenceEngine()
        engine.destroySequence(999)
        engine.destroySequence(-1)
    }

    func testCancelSequenceWithInvalidIdDoesNotCrash() {
        var engine = CotabbyInferenceEngine()
        engine.cancelSequence(999)
    }

    func testTokenizeWithoutModelReturnsEmpty() {
        let engine = CotabbyInferenceEngine()
        let text = "hello"
        let tokens = engine.tokenize(text, Int32(text.utf8.count))
        XCTAssertTrue(tokens.isEmpty)
    }

    func testDiagnosticsDefaultToZero() {
        let engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.getContextWindowTokens(), 0)
        XCTAssertEqual(engine.getBatchSize(), 0)
        XCTAssertEqual(engine.getGPULayerCount(), 0)
    }

    func testDecodePromptWithoutModelReturnsNotLoaded() {
        var engine = CotabbyInferenceEngine()
        var tokens: [Int32] = [1, 2, 3]
        let status = engine.decodePrompt(1, &tokens, Int32(tokens.count), 0)
        XCTAssertEqual(status, EngineStatus.not_loaded)
    }

    func testEndToEndWithModel() throws {
        let modelPath = "/Users/jacobfu/Library/Application Support/tabby/LlamaRuntime/Qwen3-0.6B-Q4_K_M.gguf"
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw XCTSkip("Test model not found at \(modelPath)")
        }

        var engine = CotabbyInferenceEngine()

        // Load
        let loadStatus = engine.loadModel(modelPath, -1, 2048, 512)
        XCTAssertEqual(loadStatus, EngineStatus.ok)
        XCTAssertTrue(engine.isModelLoaded())
        XCTAssertEqual(engine.getContextWindowTokens(), 2048)
        XCTAssertEqual(engine.getBatchSize(), 512)
        XCTAssertGreaterThan(engine.getThreadCount(), 0)

        // Idempotent re-load
        let reloadStatus = engine.loadModel(modelPath, -1, 2048, 512)
        XCTAssertEqual(reloadStatus, EngineStatus.ok)

        // Tokenize
        let prompt = "The quick brown fox"
        let tokens = engine.tokenize(prompt, Int32(prompt.utf8.count))
        XCTAssertFalse(tokens.isEmpty)

        // Detokenize first token
        var buf = [CChar](repeating: 0, count: 64)
        let written = engine.detokenize(tokens[0], &buf, Int32(buf.count))
        XCTAssertGreaterThan(written, 0)

        // Create autocomplete sequence
        let autoConfig = SamplingConfig(
            max_prediction_tokens: 8,
            temperature: 0.1,
            top_k: 20,
            top_p: 0.7,
            min_p: 0.08,
            repetition_penalty: 1.05,
            seed: 42
        )
        let seqA = engine.createSequence(autoConfig)
        XCTAssertGreaterThan(seqA, 0)

        // Decode prompt
        var tokenArray = Array(tokens)
        let decodeStatus = engine.decodePrompt(
            seqA, &tokenArray, Int32(tokenArray.count), 0
        )
        XCTAssertEqual(decodeStatus, EngineStatus.ok)
        XCTAssertEqual(engine.getKVPositionCount(seqA), Int32(tokenArray.count))

        // Sample a few tokens
        var generated = ""
        for _ in 0..<4 {
            let result = engine.sampleNext(seqA)
            if result.is_eos { break }
            XCTAssertFalse(result.was_cancelled)
            if let piece = result.piece, result.piece_length > 0 {
                generated += String(
                    bytes: UnsafeBufferPointer(
                        start: UnsafeRawPointer(piece)
                            .assumingMemoryBound(to: UInt8.self),
                        count: Int(result.piece_length)
                    ),
                    encoding: .utf8
                ) ?? ""
            }
        }
        XCTAssertFalse(generated.isEmpty, "Expected at least one generated token")

        // Trim KV back to prompt (remove sampled tokens)
        let trimOk = engine.trimKV(seqA, Int32(tokenArray.count))
        XCTAssertTrue(trimOk)
        XCTAssertEqual(engine.getKVPositionCount(seqA), Int32(tokenArray.count))

        // Create a second concurrent sequence (summary config)
        let summaryConfig = SamplingConfig(
            max_prediction_tokens: 60,
            temperature: 0.5,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repetition_penalty: 1.4,
            seed: 0
        )
        let seqB = engine.createSequence(summaryConfig)
        XCTAssertGreaterThan(seqB, 0)
        XCTAssertNotEqual(seqA, seqB)

        // Both sequences exist simultaneously
        XCTAssertGreaterThan(engine.getKVPositionCount(seqA), 0)
        XCTAssertEqual(engine.getKVPositionCount(seqB), 0)

        // Destroy both
        engine.destroySequence(seqB)
        engine.destroySequence(seqA)

        // Double-destroy is safe
        engine.destroySequence(seqA)

        // Unload
        engine.unloadModel()
        XCTAssertFalse(engine.isModelLoaded())
    }
}
