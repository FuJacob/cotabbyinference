import CotabbyInference
import XCTest

final class LlamaMiddlewareTests: XCTestCase {
    func testUnloadWhenNothingLoadedIsIdempotent() {
        var engine = CotabbyInferenceEngine()
        engine.unloadModel()
        engine.unloadModel()
    }

    func testLoadModelWithBadPathReturnsError() {
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(
            engine.loadModel("/nonexistent/path.gguf", -1, 2048, 512),
            EngineStatus.error
        )
    }

    func testCreateSequenceWithoutModelReturnsMinusOne() {
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.createSequence(Self.samplingConfig()), -1)
    }

    func testInvalidSequenceOperationsDoNotCrash() {
        var engine = CotabbyInferenceEngine()
        engine.destroySequence(999)
        engine.destroySequence(-1)
        engine.cancelSequence(999)
        engine.setForceWordContinuation(999, true)
        engine.setComputeLogprob(999, false)
    }

    func testTokenizeWithoutModelReturnsEmpty() {
        let engine = CotabbyInferenceEngine()
        let text = "hello"
        XCTAssertTrue(engine.tokenize(text, Int32(text.utf8.count)).isEmpty)
    }

    func testDiagnosticsDefaultToZero() {
        let engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.getContextWindowTokens(), 0)
        XCTAssertEqual(engine.getBatchSize(), 0)
        XCTAssertEqual(engine.getThreadCount(), 0)
        XCTAssertEqual(engine.getGPULayerCount(), 0)
    }

    func testDecodePromptWithoutModelReturnsNotLoaded() {
        var engine = CotabbyInferenceEngine()
        var tokens: [Int32] = [1, 2, 3]
        XCTAssertEqual(
            engine.decodePrompt(1, &tokens, Int32(tokens.count), 0),
            EngineStatus.not_loaded
        )
    }

    func testEndToEndSingleSequenceLifecycle() throws {
        let modelPath = try Self.modelPath()
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 2048, 512), EngineStatus.ok)
        defer { engine.unloadModel() }

        XCTAssertEqual(engine.getContextWindowTokens(), 2048)
        XCTAssertEqual(engine.getBatchSize(), 512)
        XCTAssertGreaterThan(engine.getThreadCount(), 0)

        // Repeating an identical load remains an idempotent no-op.
        XCTAssertEqual(engine.loadModel(modelPath, -1, 2048, 512), EngineStatus.ok)

        let prompt = "The quick brown fox"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        XCTAssertFalse(tokens.isEmpty)

        let sequence = engine.createSequence(Self.samplingConfig())
        XCTAssertGreaterThan(sequence, 0)
        XCTAssertEqual(
            engine.createSequence(Self.samplingConfig(seed: 99)),
            -1,
            "The engine must reject a second live sequence"
        )

        XCTAssertEqual(
            engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0),
            EngineStatus.ok
        )

        var generated = ""
        for _ in 0 ..< 4 {
            let result = engine.sampleNext(sequence)
            if result.is_eos { break }
            XCTAssertFalse(result.was_cancelled)
            generated += Self.string(from: result)
        }
        XCTAssertFalse(generated.isEmpty, "Expected at least one generated token")

        // Hybrid/recurrent and SWA model caches can reject partial KV removal. Cotabby treats
        // that as a cache-reuse miss and rebuilds the sequence, so lifecycle coverage must not
        // require a model-specific optimization to succeed.
        _ = engine.trimKV(sequence, Int32(tokens.count))

        engine.destroySequence(sequence)
        let replacement = engine.createSequence(Self.samplingConfig(seed: 100))
        XCTAssertGreaterThan(replacement, 0)
        XCTAssertNotEqual(replacement, sequence)
        engine.destroySequence(replacement)

        // Stale and repeated destruction must not affect a later sequence identity.
        engine.destroySequence(sequence)
    }

    func testCancellationStopsSamplingPromptly() throws {
        let modelPath = try Self.modelPath()
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let sequence = engine.createSequence(Self.samplingConfig(temperature: 0))
        let prompt = "Hello"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        XCTAssertEqual(
            engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0),
            EngineStatus.ok
        )

        _ = engine.sampleNext(sequence)
        engine.cancelSequence(sequence)
        XCTAssertTrue(engine.sampleNext(sequence).was_cancelled)
        engine.destroySequence(sequence)
    }

    func testForceWordContinuationConstrainsFirstToken() throws {
        let modelPath = try Self.modelPath()
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let sequence = engine.createSequence(Self.samplingConfig(temperature: 0))
        let prompt = "I am writ"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        engine.setForceWordContinuation(sequence, true)
        XCTAssertEqual(
            engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0),
            EngineStatus.ok
        )

        let result = engine.sampleNext(sequence)
        if !result.is_eos, let first = Self.string(from: result).first {
            XCTAssertFalse(first.isWhitespace)
        }
        engine.destroySequence(sequence)
    }

    func testSampleNextReportsFiniteLogprob() throws {
        let modelPath = try Self.modelPath()
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let sequence = engine.createSequence(Self.samplingConfig(temperature: 0))
        let prompt = "The quick brown fox"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        XCTAssertEqual(
            engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0),
            EngineStatus.ok
        )

        let result = engine.sampleNext(sequence)
        if !result.is_eos {
            XCTAssertTrue(result.logprob.isFinite)
            XCTAssertLessThanOrEqual(result.logprob, 0.0001)
        }
        engine.destroySequence(sequence)
    }

    func testDisablingLogprobSkipsSeedAndSteadyStateWork() throws {
        let modelPath = try Self.modelPath()
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let sequence = engine.createSequence(Self.samplingConfig(temperature: 0))
        engine.setComputeLogprob(sequence, false)
        let prompt = "The quick brown fox"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        XCTAssertEqual(
            engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0),
            EngineStatus.ok
        )

        for _ in 0 ..< 3 {
            let result = engine.sampleNext(sequence)
            if result.is_eos || result.was_cancelled { break }
            XCTAssertEqual(result.logprob, 0)
        }
        engine.destroySequence(sequence)
    }

    func testSamplingNeverEmitsScaffoldingMarkerPieces() throws {
        let modelPath = try Self.modelPath()
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let markers: Set<String> = [
            "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", "<|system|>",
            "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end|>",
            "<|endoftext|>", "<start_of_turn>", "<end_of_turn>", "[INST]", "[/INST]"
        ]
        let sequence = engine.createSequence(Self.samplingConfig(temperature: 1.8, seed: 7))
        let prompt = "<|im_start|>user\nWrite a reply<|im_end|>\n<|im_start|>assistant\n"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        XCTAssertEqual(
            engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0),
            EngineStatus.ok
        )

        for _ in 0 ..< 64 {
            let result = engine.sampleNext(sequence)
            if result.is_eos || result.was_cancelled { break }
            XCTAssertFalse(markers.contains(Self.string(from: result)))
        }
        engine.destroySequence(sequence)
    }

    func testArgmaxIsEOGMatchesGreedySample() throws {
        let modelPath = try Self.modelPath()
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let sequence = engine.createSequence(
            Self.samplingConfig(temperature: 0, repetitionPenalty: 1)
        )
        let prompt = "The capital of France is"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        XCTAssertEqual(
            engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0),
            EngineStatus.ok
        )

        var steps = 0
        for _ in 0 ..< 24 {
            let result = engine.sampleNext(sequence)
            if result.was_cancelled { break }
            XCTAssertEqual(result.argmax_is_eog, result.is_eos)
            steps += 1
            if result.is_eos { break }
        }
        XCTAssertGreaterThan(steps, 0)
        engine.destroySequence(sequence)
    }
}

private extension LlamaMiddlewareTests {
    static func modelPath() throws -> String {
        guard let path = ProcessInfo.processInfo.environment["COTABBY_TEST_MODEL_PATH"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Set COTABBY_TEST_MODEL_PATH to a .gguf file to run model-backed tests")
        }
        return path
    }

    static func samplingConfig(
        temperature: Float = 0.1,
        repetitionPenalty: Float = 1.05,
        seed: UInt32 = 42
    ) -> SamplingConfig {
        SamplingConfig(
            temperature: temperature,
            top_k: 20,
            top_p: 0.7,
            min_p: 0.08,
            repetition_penalty: repetitionPenalty,
            seed: seed,
            single_line: false
        )
    }

    static func string(from result: SampleResult) -> String {
        guard let piece = result.piece, result.piece_length > 0 else { return "" }
        return String(
            bytes: UnsafeBufferPointer(
                start: UnsafeRawPointer(piece).assumingMemoryBound(to: UInt8.self),
                count: Int(result.piece_length)
            ),
            encoding: .utf8
        ) ?? ""
    }
}
