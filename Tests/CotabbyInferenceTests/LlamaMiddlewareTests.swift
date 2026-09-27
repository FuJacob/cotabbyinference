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

        // This short prompt may not have a nonempty partial-state checkpoint. Cotabby treats
        // that as a miss and rebuilds, so lifecycle coverage does not require reuse here.
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
    /// Reuse is correct only when it agrees with a fresh request: abandoned predictions must
    /// neither change the model state nor enter the repetition penalty's token history.
    func testRestoredPromptMatchesColdGenerationAndBoundsReplay() throws {
        var engine = CotabbyInferenceEngine()
        let modelPath = try Self.modelPath()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), .ok)
        defer { engine.unloadModel() }
        let prompt =
            "Hi Alex, thanks for sending the project update. I will review the schedule and send you my"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        let config = Self.samplingConfig(temperature: 0, repetitionPenalty: 1.1)
        let sequence = engine.createSequence(config)
        XCTAssertEqual(engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0), .ok)
        let cold = Self.sampleTokens(engine: &engine, sequence: sequence, count: 16)
        XCTAssertFalse(cold.isEmpty)
        XCTAssertTrue(engine.trimKV(sequence, Int32(tokens.count)))
        let restored = engine.getCacheDiagnostics(sequence)
        XCTAssertEqual(restored.decoded_token_count, Int32(tokens.count))
        XCTAssertLessThanOrEqual(restored.last_restore_replayed_tokens, 8)
        if restored.uses_partial_checkpoint {
            XCTAssertGreaterThan(restored.checkpoint_bytes, 0)
            XCTAssertLessThanOrEqual(restored.checkpoint_bytes, 128 * 1024 * 1024)
        }

        XCTAssertTrue(engine.trimKV(sequence, Int32(tokens.count - 1)))
        var last = [tokens.last!]
        XCTAssertEqual(engine.decodePrompt(sequence, &last, 1, Int32(tokens.count - 1)), .ok)
        XCTAssertEqual(Self.sampleTokens(engine: &engine, sequence: sequence, count: 16), cold)
        engine.destroySequence(sequence)
    }

    /// Greedy equivalence cannot detect a stale random stream. A fixed nonzero seed and positive
    /// temperature make this request exercise the distribution sampler across repeated restores.
    func testRestoredSeededSamplingMatchesColdGeneration() throws {
        var engine = CotabbyInferenceEngine()
        let modelPath = try Self.modelPath()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), .ok)
        defer { engine.unloadModel() }
        let prompt = "Hi Alex, thanks for sending the project update. I will review the schedule and send you my"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        let config = Self.samplingConfig(temperature: 0.7, repetitionPenalty: 1.1, seed: 42)
        let sequence = engine.createSequence(config)
        XCTAssertEqual(engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0), .ok)
        let cold = Self.sampleTokens(engine: &engine, sequence: sequence, count: 24)
        XCTAssertGreaterThan(cold.count, 1, "The test must advance the random stream beyond its seed sample")

        for _ in 0 ..< 2 {
            XCTAssertTrue(engine.trimKV(sequence, Int32(tokens.count - 1)))
            var last = [tokens.last!]
            XCTAssertEqual(engine.decodePrompt(sequence, &last, 1, Int32(tokens.count - 1)), .ok)
            XCTAssertEqual(
                Self.sampleTokens(engine: &engine, sequence: sequence, count: 24),
                cold,
                "Restoration must reset both random state and prompt-only repetition history"
            )
        }
        engine.destroySequence(sequence)
    }

    func testCancellationCanBeRearmedOnlyBySuccessfulRestoration() throws {
        var engine = CotabbyInferenceEngine()
        let modelPath = try Self.modelPath()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), .ok)
        defer { engine.unloadModel() }
        let prompt =
            "Thank you for your thoughtful comments on the document. I have updated the draft to include"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        let sequence = engine.createSequence(Self.samplingConfig(temperature: 0))
        XCTAssertEqual(engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0), .ok)
        _ = Self.sampleTokens(engine: &engine, sequence: sequence, count: 4)
        engine.cancelSequence(sequence)
        XCTAssertTrue(engine.sampleNext(sequence).was_cancelled)
        XCTAssertFalse(engine.trimKV(sequence, Int32(tokens.count + 100)))
        XCTAssertTrue(engine.sampleNext(sequence).was_cancelled)
        XCTAssertTrue(engine.trimKV(sequence, Int32(tokens.count - 1)))
        var last = [tokens.last!]
        XCTAssertEqual(engine.decodePrompt(sequence, &last, 1, Int32(tokens.count - 1)), .ok)
        XCTAssertFalse(engine.sampleNext(sequence).was_cancelled)
        engine.destroySequence(sequence)
    }

    func testFailedPromptDecodeCannotBeReusedAtTrackedPosition() throws {
        var engine = CotabbyInferenceEngine()
        let modelPath = try Self.modelPath()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 64, 32), .ok)
        defer { engine.unloadModel() }
        let sequence = engine.createSequence(Self.samplingConfig(temperature: 0))
        // Exceed a deliberately small attention context without an artificial failure hook.
        // Earlier chunks commit successfully before the first chunk with no free KV cells fails.
        let oversized = String(repeating: "one two three four five six seven eight ", count: 128)
        var tokens = Array(engine.tokenize(oversized, Int32(oversized.utf8.count)))
        let status = engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0)
        if status == .ok {
            engine.destroySequence(sequence)
            throw XCTSkip("This model does not exhaust attention KV for an oversized prompt")
        }
        XCTAssertEqual(status, .error)
        let committed = engine.getCacheDiagnostics(sequence).decoded_token_count
        XCTAssertGreaterThan(committed, 0, "The failure should happen after an earlier successful batch")
        XCTAssertFalse(
            engine.trimKV(sequence, committed),
            "A failed decode must not pass the equal-position cache shortcut"
        )
        engine.destroySequence(sequence)

        let replacement = engine.createSequence(Self.samplingConfig(temperature: 0))
        let shortPrompt = "The quick brown fox"
        var shortTokens = Array(engine.tokenize(shortPrompt, Int32(shortPrompt.utf8.count)))
        XCTAssertEqual(engine.decodePrompt(replacement, &shortTokens, Int32(shortTokens.count), 0), .ok)
        engine.destroySequence(replacement)
    }

    func testHealingReproducesTypedBytesIncludingTrailingWhitespace() throws {
        var engine = CotabbyInferenceEngine()
        let modelPath = try Self.modelPath()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), .ok)
        defer { engine.unloadModel() }
        for prompt in ["Please send me the sched", "Please send me the ", "The café serves "] {
            var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
            XCTAssertGreaterThan(tokens.count, 1)
            let prefix = Array(engine.tokenPiece(tokens.removeLast()))
            XCTAssertFalse(prefix.isEmpty)
            XCTAssertTrue(Array(prompt.utf8).suffix(prefix.count).elementsEqual(prefix))
            let sequence = engine.createSequence(Self.samplingConfig(temperature: 0))
            prefix.withUnsafeBufferPointer {
                engine.setCompletionPrefix(sequence, $0.baseAddress, Int32($0.count))
            }
            XCTAssertEqual(engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0), .ok)
            var generated: [UInt8] = []
            for _ in 0..<prefix.count + 4 {
                let result = engine.sampleNext(sequence)
                if result.is_eos || result.was_cancelled { break }
                generated += Array(engine.tokenPiece(result.token))
                if generated.count > prefix.count { break }
            }
            XCTAssertTrue(generated.starts(with: prefix), "Healing must reproduce the exact typed prefix")
            XCTAssertGreaterThan(generated.count, prefix.count)
            engine.destroySequence(sequence)
        }
    }

    /// This reports native prompt work separately from typing-to-display latency in the app.
    /// No wall-clock threshold is asserted: CI hardware and Metal warmup vary substantially.
    func testWarmPromptDecodeReportsLatency() throws {
        var engine = CotabbyInferenceEngine()
        let modelPath = try Self.modelPath()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), .ok)
        defer { engine.unloadModel() }
        for paragraphCount in [2, 16, 64] {
            let prompt =
                String(
                    repeating:
                        "We are reviewing the project schedule and collecting feedback from the team. ",
                    count: paragraphCount)
                + "Please send the updated schedule by"
            var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
            var coldMilliseconds: [Double] = []
            var warmMilliseconds: [Double] = []
            var checkpointBytes: UInt64 = 0
            for iteration in 0..<4 {
                let sequence = engine.createSequence(Self.samplingConfig(temperature: 0))
                let coldStart = ContinuousClock.now
                XCTAssertEqual(engine.decodePrompt(sequence, &tokens, Int32(tokens.count), 0), .ok)
                let coldDuration = coldStart.duration(to: .now)
                let coldToken = engine.sampleNext(sequence).token
                let warmStart = ContinuousClock.now
                XCTAssertTrue(engine.trimKV(sequence, Int32(tokens.count - 1)))
                var last = [tokens.last!]
                XCTAssertEqual(engine.decodePrompt(sequence, &last, 1, Int32(tokens.count - 1)), .ok)
                let warmDuration = warmStart.duration(to: .now)
                XCTAssertEqual(engine.sampleNext(sequence).token, coldToken)
                checkpointBytes = engine.getCacheDiagnostics(sequence).checkpoint_bytes
                // Exclude the first pass so graph/kernel initialization does not dominate the report.
                if iteration > 0 {
                    coldMilliseconds.append(Self.milliseconds(coldDuration))
                    warmMilliseconds.append(Self.milliseconds(warmDuration))
                }
                engine.destroySequence(sequence)
            }
            print(
                "CACHE_BENCHMARK prompt_tokens=\(tokens.count) cold_median_ms=\(coldMilliseconds.sorted()[1]) warm_median_ms=\(warmMilliseconds.sorted()[1]) checkpoint_bytes=\(checkpointBytes)"
            )
        }
    }
}

private extension LlamaMiddlewareTests {
    static func milliseconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1_000
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000
    }
    static func sampleTokens(
        engine: inout CotabbyInferenceEngine, sequence: Int32, count: Int
    ) -> [Int32] {
        var tokens: [Int32] = []
        for _ in 0..<count {
            let result = engine.sampleNext(sequence)
            if result.is_eos || result.was_cancelled { break }
            tokens.append(result.token)
        }
        return tokens
    }
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
