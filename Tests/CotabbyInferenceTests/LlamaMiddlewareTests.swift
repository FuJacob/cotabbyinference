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
            seed: 0,
            single_line: false
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

    func testTokenizeWithOptionsWithoutModelReturnsEmpty() {
        let engine = CotabbyInferenceEngine()
        let text = "hello"
        let tokens = engine.tokenizeWithOptions(
            text, Int32(text.utf8.count), false, true
        )
        XCTAssertTrue(tokens.isEmpty)
    }

    func testHasChatTemplateWithoutModelIsFalse() {
        let engine = CotabbyInferenceEngine()
        XCTAssertFalse(engine.hasChatTemplate())
    }

    func testApplyChatTemplateWithoutModelReturnsZero() {
        let engine = CotabbyInferenceEngine()
        var buffer = [CChar](repeating: 0, count: 256)
        let written = engine.applyChatTemplate(
            "You complete text.", "The quick brown", true, &buffer, Int32(buffer.count)
        )
        // No model loaded → 0 (caller falls back to the raw path).
        XCTAssertEqual(written, 0)
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
        guard let modelPath = ProcessInfo.processInfo.environment["COTABBY_TEST_MODEL_PATH"],
              FileManager.default.fileExists(atPath: modelPath) else {
            throw XCTSkip("Set COTABBY_TEST_MODEL_PATH to a .gguf file to run this test")
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

        // Chat-template path: instruct models ship a template; if present,
        // rendering a simple conversation must produce a non-empty prompt that
        // tokenizes (with parse_special) to a non-empty token list.
        if engine.hasChatTemplate() {
            // Render system + user through the model's template into a caller buffer.
            var buffer = [CChar](repeating: 0, count: 4096)
            let written = engine.applyChatTemplate(
                "You complete text.", "The quick brown", true, &buffer, Int32(buffer.count)
            )
            XCTAssertGreaterThan(written, 0, "Model reports a template but rendering produced no bytes")

            let rendered = buffer.prefix(Int(written)).withUnsafeBufferPointer { ptr in
                String(
                    bytes: UnsafeRawBufferPointer(ptr),
                    encoding: .utf8
                )
            }
            let renderedSwift = try XCTUnwrap(rendered, "Rendered template was not valid UTF-8")
            XCTAssertFalse(renderedSwift.isEmpty)

            let templated = engine.tokenizeWithOptions(
                renderedSwift, Int32(renderedSwift.utf8.count), false, true
            )
            XCTAssertFalse(templated.isEmpty)
        }

        // Detokenize a content token (the prompt's last token). Index 0 can be BOS, a control
        // token that renders to zero bytes with special=false, so we avoid it here.
        var buf = [CChar](repeating: 0, count: 64)
        let written = engine.detokenize(tokens[tokens.count - 1], &buf, Int32(buf.count))
        XCTAssertGreaterThan(written, 0)

        // Create autocomplete sequence
        let autoConfig = SamplingConfig(
            max_prediction_tokens: 8,
            temperature: 0.1,
            top_k: 20,
            top_p: 0.7,
            min_p: 0.08,
            repetition_penalty: 1.05,
            seed: 42,
            single_line: false
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
            seed: 0,
            single_line: false
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

    // Multi-sequence sequential test. Drives the new shared-context decoder
    // thread through two interleaved sampling loops to verify the seed-token
    // / feedback-decode handoff produces valid tokens for both sequences
    // when their sampleNext calls alternate.
    func testInterleavedMultiSequenceSampling() throws {
        guard let modelPath = ProcessInfo.processInfo.environment["COTABBY_TEST_MODEL_PATH"] else {
            try XCTSkipIf(true, "Set COTABBY_TEST_MODEL_PATH to a .gguf file to run this test")
            return
        }

        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let prompt = "The quick brown fox jumps over the lazy dog."
        let tokens = engine.tokenize(prompt, Int32(prompt.utf8.count))
        XCTAssertGreaterThan(tokens.size(), 0)

        let configA = SamplingConfig(
            max_prediction_tokens: 16, temperature: 0,
            top_k: 0, top_p: 0, min_p: 0,
            repetition_penalty: 0, seed: 1,
            single_line: false
        )
        let configB = SamplingConfig(
            max_prediction_tokens: 16, temperature: 0,
            top_k: 0, top_p: 0, min_p: 0,
            repetition_penalty: 0, seed: 2,
            single_line: false
        )

        let seqA = engine.createSequence(configA)
        let seqB = engine.createSequence(configB)
        XCTAssertGreaterThan(seqA, 0)
        XCTAssertGreaterThan(seqB, 0)

        var tokenArr = Array(tokens)
        XCTAssertEqual(
            engine.decodePrompt(seqA, &tokenArr, Int32(tokenArr.count), 0),
            EngineStatus.ok
        )
        XCTAssertEqual(
            engine.decodePrompt(seqB, &tokenArr, Int32(tokenArr.count), 0),
            EngineStatus.ok
        )

        // Alternate sampleNext between the two sequences. With greedy
        // sampling and identical prompts, the first sampled tokens for both
        // sequences should be identical (different samplers reading the
        // same logits row at separate decodePrompt times).
        var sampledA: [Int32] = []
        var sampledB: [Int32] = []
        for _ in 0..<8 {
            let rA = engine.sampleNext(seqA)
            let rB = engine.sampleNext(seqB)
            XCTAssertFalse(rA.was_cancelled)
            XCTAssertFalse(rB.was_cancelled)
            if rA.is_eos || rB.is_eos { break }
            sampledA.append(rA.token)
            sampledB.append(rB.token)
        }
        XCTAssertEqual(sampledA.count, sampledB.count)
        XCTAssertGreaterThan(sampledA.count, 0)
        XCTAssertEqual(sampledA, sampledB,
            "Greedy sampling with identical prompts should match across sequences")

        engine.destroySequence(seqA)
        engine.destroySequence(seqB)
    }

    // Cancellation regression: setting cancelled on a sequence mid-loop must
    // cause subsequent sampleNext calls to return was_cancelled=true so
    // callers can break out without waiting for the full prediction budget.
    func testCancellationStopsSamplingPromptly() throws {
        guard let modelPath = ProcessInfo.processInfo.environment["COTABBY_TEST_MODEL_PATH"] else {
            try XCTSkipIf(true, "Set COTABBY_TEST_MODEL_PATH to a .gguf file to run this test")
            return
        }

        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let prompt = "Hello"
        let tokens = engine.tokenize(prompt, Int32(prompt.utf8.count))
        let config = SamplingConfig(
            max_prediction_tokens: 32, temperature: 0,
            top_k: 0, top_p: 0, min_p: 0,
            repetition_penalty: 0, seed: 0,
            single_line: false
        )

        let seq = engine.createSequence(config)
        var tokenArr = Array(tokens)
        XCTAssertEqual(
            engine.decodePrompt(seq, &tokenArr, Int32(tokenArr.count), 0),
            EngineStatus.ok
        )

        // Sample a couple of tokens first.
        for _ in 0..<2 {
            let r = engine.sampleNext(seq)
            XCTAssertFalse(r.was_cancelled)
        }

        // Cancel. The next sampleNext should return was_cancelled=true
        // without doing any further model work.
        engine.cancelSequence(seq)
        let cancelled = engine.sampleNext(seq)
        XCTAssertTrue(cancelled.was_cancelled,
            "sampleNext after cancelSequence must return was_cancelled=true")

        engine.destroySequence(seq)
    }

    func testSetForceWordContinuationWithoutModelDoesNotCrash() {
        var engine = CotabbyInferenceEngine()
        engine.setForceWordContinuation(999, true)
        engine.setForceWordContinuation(-1, false)
        XCTAssertFalse(engine.isModelLoaded())
    }

    func testSnapshotSizeWithoutModelIsZero() {
        let engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.snapshotSize(1), 0)
    }

    // With the first-token word-continuation constraint set, the seed token must not start a new
    // word, i.e. its decoded text must not begin with whitespace.
    func testForceWordContinuationConstrainsFirstToken() throws {
        guard let modelPath = ProcessInfo.processInfo.environment["COTABBY_TEST_MODEL_PATH"] else {
            try XCTSkipIf(true, "Set COTABBY_TEST_MODEL_PATH to a .gguf file to run this test")
            return
        }
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let config = SamplingConfig(
            max_prediction_tokens: 8, temperature: 0,
            top_k: 0, top_p: 0, min_p: 0,
            repetition_penalty: 0, seed: 0,
            single_line: false
        )
        let seq = engine.createSequence(config)
        XCTAssertGreaterThan(seq, 0)

        // Prompt ends mid-word ("writ"); the forced continuation must finish the word.
        let prompt = "I am writ"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        XCTAssertGreaterThan(tokens.count, 0)

        engine.setForceWordContinuation(seq, true)
        XCTAssertEqual(engine.decodePrompt(seq, &tokens, Int32(tokens.count), 0), EngineStatus.ok)

        let result = engine.sampleNext(seq)
        if !result.is_eos, let piece = result.piece, result.piece_length > 0 {
            let text = String(
                bytes: UnsafeBufferPointer(
                    start: UnsafeRawPointer(piece).assumingMemoryBound(to: UInt8.self),
                    count: Int(result.piece_length)
                ),
                encoding: .utf8
            ) ?? ""
            if let firstChar = text.first {
                XCTAssertFalse(
                    firstChar.isWhitespace,
                    "Forced word continuation must not begin the first token with whitespace"
                )
            }
        }
        engine.destroySequence(seq)
    }

    // Snapshotting a sequence then restoring it must return the engine's KV position bookkeeping
    // to the captured value.
    func testSnapshotRestorePreservesPosition() throws {
        guard let modelPath = ProcessInfo.processInfo.environment["COTABBY_TEST_MODEL_PATH"] else {
            try XCTSkipIf(true, "Set COTABBY_TEST_MODEL_PATH to a .gguf file to run this test")
            return
        }
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let config = SamplingConfig(
            max_prediction_tokens: 8, temperature: 0,
            top_k: 0, top_p: 0, min_p: 0,
            repetition_penalty: 0, seed: 0,
            single_line: false
        )
        let seq = engine.createSequence(config)
        let prompt = "The quick brown fox"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        XCTAssertEqual(engine.decodePrompt(seq, &tokens, Int32(tokens.count), 0), EngineStatus.ok)

        let position = engine.getKVPositionCount(seq)
        XCTAssertGreaterThan(position, 0)

        let size = engine.snapshotSize(seq)
        XCTAssertGreaterThan(size, 0)
        var buffer = [UInt8](repeating: 0, count: Int(size))
        let written = engine.snapshotSequence(seq, &buffer, size)
        XCTAssertGreaterThan(written, 0)

        // Advance past the snapshot point, then restore back to it.
        _ = engine.sampleNext(seq)
        XCTAssertTrue(engine.restoreSequence(seq, buffer, written, position))
        XCTAssertEqual(engine.getKVPositionCount(seq), position)

        engine.destroySequence(seq)
    }

    // A sampled (non-EOS) token must carry a finite log-probability that is <= 0, so the app can use
    // it as a confidence signal.
    func testSampleNextReportsFiniteLogprob() throws {
        guard let modelPath = ProcessInfo.processInfo.environment["COTABBY_TEST_MODEL_PATH"] else {
            try XCTSkipIf(true, "Set COTABBY_TEST_MODEL_PATH to a .gguf file to run this test")
            return
        }
        var engine = CotabbyInferenceEngine()
        XCTAssertEqual(engine.loadModel(modelPath, -1, 1024, 256), EngineStatus.ok)
        defer { engine.unloadModel() }

        let config = SamplingConfig(
            max_prediction_tokens: 4, temperature: 0,
            top_k: 0, top_p: 0, min_p: 0,
            repetition_penalty: 0, seed: 0,
            single_line: false
        )
        let seq = engine.createSequence(config)
        let prompt = "The quick brown fox"
        var tokens = Array(engine.tokenize(prompt, Int32(prompt.utf8.count)))
        XCTAssertEqual(engine.decodePrompt(seq, &tokens, Int32(tokens.count), 0), EngineStatus.ok)

        let result = engine.sampleNext(seq)
        if !result.is_eos {
            XCTAssertTrue(result.logprob.isFinite, "logprob must be finite")
            XCTAssertLessThanOrEqual(result.logprob, 0.0001, "a log-probability must be <= 0")
        }
        engine.destroySequence(seq)
    }
}
