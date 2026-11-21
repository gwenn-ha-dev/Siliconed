import Darwin
import Foundation
import XCTest
@testable import Siliconed

/// **The engine lives in its own process** (`silicore-worker`): the app asks it, never plans a session itself, and what
/// crosses the pipe comes back as it left — a render's images bit for bit.
final class EngineWorkerTests: XCTestCase {
    private static let sources = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../../Sources").standardizedFileURL

    /// The app's code runs no engine: no session planned, no render or writer run in its process, no ledger read, no
    /// guard armed — all of it is the worker's (`EngineWorker`). Comments may name them.
    func testTheAppRunsNoEngine() throws {
        let pattern = #"\bimport\s+SiliCore\b|\bSessionPlan\b|\bCoreAI\.(plan|once|under)\b|\bEngine\(\)|\bLedger\b|\bStageLedger\b"#
            + #"|\bPromptWriter\(|\.renderBatch\(|\.renderBatchOnQueue\(|\.traits\.render\b|MemoryBudget\.machineCeiling|Engine\.measuringThisMac"#
            + #"|\bQwen3VLGenerator\b|\bRenderCache\b"#
        let regex = try NSRegularExpression(pattern: pattern)
        let root = Self.sources.appendingPathComponent("SiliconedApp")
        let files = (FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)?.compactMap { $0 as? URL } ?? [])
            .filter { $0.pathExtension == "swift" }
        XCTAssertGreaterThan(files.count, 50)
        var hits: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for (i, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let code = String(line.range(of: "//").map { line[..<$0.lowerBound] } ?? line[...])
                if let m = regex.firstMatch(in: code, range: NSRange(code.startIndex..., in: code)) {
                    hits.append("\(file.lastPathComponent):\(i + 1): \((code as NSString).substring(with: m.range))")
                }
            }
        }
        XCTAssertTrue(hits.isEmpty, hits.joined(separator: "\n"))
    }

    private func image(_ w: Int, _ h: Int, alpha: Bool, seed: Float) -> ImageRGB {
        let pixels = (0..<(3 * w * h)).map { Float($0 % 97) / 48 - 1 + seed * 1e-7 }
        let rgba: [UInt8]? = alpha ? (0..<(4 * w * h)).map { UInt8($0 % 251) } : nil
        return ImageRGB(pixels: pixels, height: h, width: w, rgba: rgba)
    }

    /// **Calls and replies through a real pipe**: written by `WorkerPipe`, read back by `WorkerInbox` in pieces of any
    /// size, each message with the blobs it names, whole.
    private func through<M: Encodable & Sendable, R: Decodable>(_ message: M, blobs: [(UInt32, Data)], as: R.Type,
                                                     names: (R) -> [UInt32]) throws -> (R, [UInt32: Data]) {
        var fds: [Int32] = [0, 0]
        XCTAssertEqual(pipe(&fds), 0)
        defer { close(fds[0]) }
        let (writeEnd, writer) = (fds[1], WorkerPipe(fd: fds[1]))
        let thread = Thread { writer.send(message, blobs: blobs); close(writeEnd) }
        thread.start()
        var inbox = WorkerInbox()
        var buffer = [UInt8](repeating: 0, count: 7919)   // a prime: frames cut anywhere
        var messages: [Data] = []
        while true {
            let n = read(fds[0], &buffer, buffer.count)
            guard n > 0 else { break }
            messages += try inbox.receive(Data(buffer[0..<n]))
        }
        XCTAssertEqual(messages.count, 1)
        let reply = try JSONDecoder().decode(R.self, from: messages[0])
        return (reply, inbox.take(names(reply)))
    }

    func testARenderOrderComesBackAsItLeft() throws {
        var request = Request("a lighthouse, {dawn|dusk}", width: 64, height: 48, seed: 7, steps: 6,
                              loras: [LoRAEntry("/a/flat.safetensors", strength: 0.8)],
                              image: image(64, 48, alpha: false, seed: 1), strength: 0.55, previews: true)
        request.references = [image(32, 40, alpha: true, seed: 2), image(16, 16, alpha: false, seed: 3)]
        request.detail = .more
        request.solver = .dpmpp2m
        request.variations = [Variation(seed: 9, strength: 0.25)]
        request.precision = .exact
        let library = Library(root: URL(fileURLWithPath: "/tmp/models"), local: URL(fileURLWithPath: "/tmp/mac"))
        let (order, blobs) = RenderOrder.of(request, seeds: [7, 8], model: "some-model", library: library,
                                            processPrecision: .neuralEngine, firstBlob: 40)
        XCTAssertEqual(blobs.map(\.0), [40, 41, 42])
        let (call, received) = try through(WorkerCall.render(id: 3, order), blobs: blobs, as: WorkerCall.self) {
            if case .render(_, let o) = $0 { return o.blobs }
            return []
        }
        guard case let .render(id, back) = call else { return XCTFail("\(call)") }
        XCTAssertEqual(id, 3)
        XCTAssertEqual(back, order)
        XCTAssertEqual(back.seeds, [7, 8])
        XCTAssertEqual(back.library.library.root, library.root)
        XCTAssertEqual(back.library.library.local, library.local)
        XCTAssertEqual(back.processPrecision, .neuralEngine)
        let again = try back.request(received)
        XCTAssertEqual(again, request)
    }

    func testEveryEventComesBackAsItLeft() throws {
        let rendered = Engine.Render(
            model: "m", prompt: "p", seed: 42, steps: 8, loras: [LoRAEntry("x.safetensors", strength: 0.5)],
            image: image(24, 16, alpha: false, seed: 4), evaluations: 8, strength: 0.6, startStep: 2, startSigma: 0.7,
            timings: { var t = Engine.Timings(); t.text = 1.25; t.denoising = 3.5; t.total = 5; return t }(),
            footprints: { var f = Engine.Footprints(); var h = Engine.Held(stage: .denoising); h.held = 7; h.ring = 2; f.stages = [h]; f.neuralEngine = 0.25; return f }(),
            tokens: 12, detail: .more, solver: .dpmpp2m, variations: [Variation(seed: 1, strength: 0.5)], modelName: "M",
            isImportedModel: true, loraOrigins: [.init(name: "x", sha256: "ab")])
        var held = Engine.Held(stage: .text)
        held.held = 1 << 30; held.ring = 2
        let events: [Engine.Event] = [
            .start(Engine.Plan(denoising: DenoisingPlan(startStep: 2, startSigma: 0.7, evaluations: 6), seeds: [42, 43], stages: [.text, .image, .denoising, .decoding])),
            .stage(.denoising, image: 1),
            .text(tokens: 12, tokenizer: 0.01, encoder: 0.99, cached: true),
            .encoding(seconds: 0.3),
            .step(image: 1, index: 3, total: 6, sigma: 0.123456789, seconds: 1.5, latentGridHeight: 2, latentGridWidth: 3),
            .preview(Engine.Preview(image: 0, index: 1, total: 6, height: 2, width: 3, rgba: (0..<24).map { UInt8($0 * 10) })),
            .decoding(seconds: 0.4),
            .image(index: 1, rendered),
            .warning("a fallback"),
            .held([held]),
            .memory(Engine.MemorySeries(buckets: [.init(index: 2, low: 1 << 30, high: 2 << 30)], marks: [.init(stage: .text, at: 0.25)])),
        ]
        for (n, event) in events.enumerated() {
            let (wire, blob) = WorkerEvent.of(event, blob: UInt32(100 + n))
            let (reply, blobs) = try through(WorkerReply.event(id: 9, wire), blobs: blob.map { [(wire.blob!, $0)] } ?? [],
                                             as: WorkerReply.self) {
                if case .event(_, let e) = $0 { return e.blob.map { [$0] } ?? [] }
                return []
            }
            guard case let .event(id, back) = reply else { return XCTFail("\(reply)") }
            XCTAssertEqual(id, 9)
            XCTAssertEqual("\(try back.event(blobs))", "\(event)", "event \(n)")
            if case .image(_, let r) = try back.event(blobs) {
                // Bit for bit: the app makes the PNG from these floats.
                XCTAssertEqual(r.image, rendered.image)
                XCTAssertEqual(try r.png(), try rendered.png())
            }
        }
    }

    func testFailuresAndTheWritersEventsCross() throws {
        let failures: [WorkerFailure] = [
            .engine(.insufficientMemory(needed: 9 << 30, budget: 4 << 30)),
            .engine(.engineStopped(budget: 4 << 30, detail: "held")),
            .writer(.promptTooLong(tokens: 3000, limit: 2048)),
            .writer(.notInstalled),
        ]
        for f in failures {
            let back = try WireJSON.decode(WorkerReply.self, from: WireJSON.encode(WorkerReply.failed(id: 1, f)))
            guard case .failed(1, let g) = back else { return XCTFail("\(back)") }
            XCTAssertEqual(g, f)
        }
        XCTAssertEqual(WorkerFailure(PromptWriter.Failure.notInstalled), .writer(.notInstalled))
        XCTAssertEqual(WorkerFailure(CancellationError()), .engine(.cancelled))
        for e: PromptWriter.Event in [.loading(0.5), .text("so far"), .finished(text: "all", tokens: 3), .cancelled] {
            let back = try WireJSON.decode(WorkerReply.self, from: WireJSON.encode(WorkerReply.written(id: 2, e)))
            guard case .written(2, let f) = back else { return XCTFail("\(back)") }
            XCTAssertEqual(f, e)
        }
    }

    /// A float that is not finite crosses the wire as itself (the default encoder refused it, and the message was
    /// dropped); a message without one is the default encoder's JSON. Compared as JSON values, not bytes: two encodings
    /// of one value may order its keys differently (failed 1 run in 3 as bytes, 21/11).
    func testWireCarriesFloatsThatAreNotFinite() throws {
        let back = try WireJSON.decode([Double].self, from: WireJSON.encode([Double.nan, .infinity, -.infinity, 1.5]))
        XCTAssertTrue(back[0].isNaN)
        XCTAssertEqual(Array(back.dropFirst()), [.infinity, -.infinity, 1.5])
        let reply = WorkerReply.written(id: 3, .loading(0.25))
        func value(_ data: Data) throws -> NSObject { try JSONSerialization.jsonObject(with: data) as! NSObject }
        XCTAssertEqual(try value(WireJSON.encode(reply)), try value(JSONEncoder().encode(reply)))
    }

    // MARK: - A worker that does not answer as it should

    private func folder() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("engine-worker-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    /// **A stand-in for `silicore-worker`**: `/bin/sh` running `script`, `$0` a file holding a hello of `version` as the
    /// worker writes it.
    private func standIn(hello version: Int, _ script: String, silence: TimeInterval = EngineWorker.silence) throws -> EngineWorker {
        let hello = try folder().appendingPathComponent("hello")
        try Frame(kind: .message, payload: JSONEncoder().encode(WorkerHello(protocolVersion: version))).bytes.write(to: hello)
        return EngineWorker(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", script, hello.path], silence: silence)
    }

    /// What a call ends with: its error, `nil` when it finished.
    private func ending(_ worker: EngineWorker) async throws -> Error? {
        let library = Library(root: try folder())
        do {
            for try await _ in worker.write(rewrite: "a lighthouse", instruction: "shorter", library: library) {}
            return nil
        } catch { return error }
    }

    /// A worker of another protocol is refused before any call: the call fails with the mismatch, both versions said.
    func testAWorkerOfAnotherBuildIsRefused() async throws {
        let worker = try standIn(hello: WorkerHello.version + 1, #"cat "$0"; exec cat >/dev/null"#)
        let error = try await ending(worker)
        XCTAssertEqual(error as? EngineError, .engineMismatch(app: WorkerHello.version, engine: WorkerHello.version + 1))
        // A worker from before the hello: its first message is no hello.
        let older = EngineWorker(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", #"printf '\000\000\000\000\000\002{}'; exec cat >/dev/null"#])
        let olderError = try await ending(older)
        XCTAssertEqual(olderError as? EngineError, .engineMismatch(app: WorkerHello.version, engine: nil))
    }

    /// The worker refuses an app of another protocol: it says its own version first, then exits (status 66) on the
    /// app's hello. The real `silicore-worker`, built beside the tests.
    func testTheWorkerRefusesAnAppOfAnotherBuild() throws {
        let url = Bundle(for: Self.self).bundleURL.deletingLastPathComponent().appendingPathComponent("silicore-worker")
        guard FileManager.default.isExecutableFile(atPath: url.path) else { throw XCTSkip("no silicore-worker at \(url.path)") }
        let process = Process(), input = Pipe(), output = Pipe(), errors = Pipe()
        process.executableURL = url
        process.standardInput = input; process.standardOutput = output; process.standardError = errors
        try process.run()
        let theirs = Frame(kind: .message, payload: try JSONEncoder().encode(WorkerHello(protocolVersion: WorkerHello.version + 1)))
        try input.fileHandleForWriting.write(contentsOf: theirs.bytes)
        let out = try output.fileHandleForReading.readToEnd() ?? Data()
        let said = String(decoding: try errors.fileHandleForReading.readToEnd() ?? Data(), as: UTF8.self)
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 66, said)
        XCTAssertTrue(said.contains("refused"), said)
        var reader = FrameReader()
        reader.append(out)
        let first = try XCTUnwrap(try reader.next())
        XCTAssertEqual(first.kind, .message)
        XCTAssertEqual(WorkerHello.version(of: first.payload), WorkerHello.version)
    }

    /// The app and its worker of the same build talk: a writer's gesture in an empty library fails as the writer says.
    /// Then, idle past its silence, the worker lives on: its clock runs only with a call in hand.
    func testTheWorkerOfThisBuildAnswers() async throws {
        let url = Bundle(for: Self.self).bundleURL.deletingLastPathComponent().appendingPathComponent("silicore-worker")
        guard FileManager.default.isExecutableFile(atPath: url.path) else { throw XCTSkip("no silicore-worker at \(url.path)") }
        let worker = EngineWorker(executable: url, silence: 1)
        defer { worker.stop() }
        let error = try await ending(worker)
        XCTAssertEqual(error as? PromptWriter.Failure, .notInstalled, "\(String(describing: error))")
        let pid = try XCTUnwrap(worker.pid)
        try await Task.sleep(for: .seconds(2.5))
        XCTAssertEqual(worker.pid, pid)
    }

    /// **A worker that says nothing**, a call in hand, is killed past its silence: the call fails with it, and the next
    /// call starts another worker.
    func testASilentWorkerIsStopped() async throws {
        let worker = try standIn(hello: WorkerHello.version, #"cat "$0"; exec cat >/dev/null"#, silence: 0.4)
        let start = Date()
        let error = try await ending(worker)
        XCTAssertEqual(error as? EngineError, .engineSilent(seconds: 0))
        XCTAssertLessThan(Date().timeIntervalSince(start), 5)
        XCTAssertNil(worker.pid, "killed")
        // Another worker for the next call, watched the same.
        let again = try await ending(worker)
        XCTAssertEqual(again as? EngineError, .engineSilent(seconds: 0))
    }

    /// **The fuse's last line reaches the app**: still on its way on the standard error when the worker's end is seen,
    /// it says the stop was past B, not a crash. Here it comes 0.3 s late, written by a child of the stand-in that
    /// holds its standard error (its standard output closed): the app reads that pipe to its end before saying why.
    func testTheFusesLastLineIsDelivered() async throws {
        let script = #"cat "$0"; (exec 1>&-; sleep 0.3; echo "SiliCore: the footprint 9.0 MiB past 1.0 MiB: stopped" >&2) & exit 3"#
        let worker = try standIn(hello: WorkerHello.version, script)
        let error = try await ending(worker)
        guard case let .engineStopped(budget?, detail)? = error as? EngineError else { return XCTFail("\(String(describing: error))") }
        XCTAssertEqual(budget, MemoryBudget.budget)
        XCTAssertTrue(detail.contains("past 1.0 MiB"), detail)
    }
}
