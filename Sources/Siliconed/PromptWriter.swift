import Foundation
import Synchronization

/// **The lab's writer**: a prompt in, a prompt out, written by the installed text encoder made to write
/// (Qwen-Image-2.1's Qwen3-VL-8B, `Qwen3VLGenerator`). Rewrite turns a text into another following an
/// instruction (the publisher's guidance, as the system turn); Describe turns an image into a prompt.
///
///     write(request) ─▶ AsyncThrowingStream<Event>
///         prompt (template, tokenizer, image) ─ need from shapes ≤ B? ─ no: `Failure.unavailable(Mac size)`
///         weights resident? ─ no: `.loading(0…1)` while the layers and the head are read into pages
///         greedy decoding ─ each token's bytes ─ `UTF8Stream` ─ `.text(so far)` ─ `.finished(text, tokens)`
///     cancel()    the run in flight stops between two layers: `.cancelled`
///     release()   cancels, then gives the weights back (a render is about to start)
///
/// What the lab's use asks of it, and what a caller would get wrong:
///
///   - **It never runs during a render, and the engine sees to it.** A run is a job of the engine's queue (`Engine.file`) and holds
///     the machine's render lock (`RenderLock.machine`): the exclusion with a render, in this process and in any other, is the
///     engine's, not a convention of the app. The writer's resident weights (8.2 GB at the Compact's width) are not in any
///     render's need, which assumes the whole of B: a render's first act, on that same queue, is `unloadResident()`.
///   - **One thing at a time**: a run waits for the one before it on the engine's queue; two never share the pages. A run still
///     waiting is cancelled by `cancel()` and `release()` as well as the one in flight.
///   - **The plan is chosen before the first token**: the need is computed from the prompt's actual size
///     (`PromptWriterNeed`) and compared with B (`MemoryBudget.machineCeiling`), after the process's base, read
///     once, before the writer holds anything of its own. Above B, the writer is unavailable, with the Mac
///     that would hold it; a prompt over `Limits.promptTokens` is refused with its count.
///   - **The base is read at every run**, never kept: the app's footprint changes between two runs; what the writer itself holds
///     resident is taken out of it (it is in the need).
///   - **A cancelled stream ends with `.cancelled`, not an error**; a consumer that stops reading cancels the run.
///   - **The text is cumulative**, and never ends inside a character (`UTF8Stream`).
///   - **Greedy**, whatever the Instruct's `generation_config.json` recommends: ties to the lowest id.
package final class PromptWriter: @unchecked Sendable {
    package enum Request: Sendable {
        /// `instruction` is the system turn (empty: none); `text` the user turn.
        case rewrite(text: String, instruction: String)
        /// `instruction` is the system turn (empty: none); the user turn is the image, then `ask` if any.
        case describe(image: Qwen3VLImages.RGB8, instruction: String, ask: String = "")
    }

    package enum Event: Sendable, Equatable {
        /// The weights being read into memory, 0…1 (once per load).
        case loading(Double)
        /// The text so far (cumulative).
        case text(String)
        /// The whole answer, and the tokens it took (the stop token not counted).
        case finished(text: String, tokens: Int)
        case cancelled
    }

    package enum Failure: Error, CustomStringConvertible, Equatable {
        case notInstalled
        /// The need is above B: the writer cannot run on this Mac.
        case unavailable(PromptWriterNeed.Availability)
        case promptTooLong(tokens: Int, limit: Int)
        case failed(String)

        package var description: String {
            switch self {
            case .notInstalled: return "the writer is not installed"
            case .unavailable(.needsMac(let gigabytes, _)): return "the writer needs a Mac with \(gigabytes) GB of memory"
            case .unavailable(.available): return "the writer is available"
            case let .promptTooLong(tokens, limit): return "the prompt is \(tokens) tokens, the writer reads at most \(limit)"
            case .failed(let why): return why
            }
        }
    }

    private let maps: PromptWriterMaps
    private let encoderPath: String, headPath: String, tokenizerPath: String
    private let limits: PromptWriterNeed.Limits
    private let ceiling: () -> Int
    private let lock = NSLock()
    /// Every run not yet finished, queued or in flight: the tokens `cancel()` reaches.
    private var pending: [ObjectIdentifier: Cancellation] = [:]

    // Owned by the engine's queue (`Engine.file`).
    private var generator: Qwen3VLGenerator?
    private var tokenizer: Tokenizer?
    private var tokenBytes: TokenBytes?

    /// The writers that may hold weights: a render unloads them on the engine's queue before its first computation.
    private static let live = Mutex<[ObjectIdentifier: Weak]>([:])
    private struct Weak { weak var writer: PromptWriter? }

    package init(maps: PromptWriterMaps, encoderPath: String, headPath: String, tokenizerPath: String,
                 limits: PromptWriterNeed.Limits = .init(), ceiling: @escaping () -> Int = { MemoryBudget.machineCeiling() }) {
        self.maps = maps; self.encoderPath = encoderPath; self.headPath = headPath; self.tokenizerPath = tokenizerPath
        self.limits = limits; self.ceiling = ceiling
        PromptWriter.live.withLock { $0[ObjectIdentifier(self)] = Weak(writer: self) }
    }

    deinit { let id = ObjectIdentifier(self); PromptWriter.live.withLock { _ = $0.removeValue(forKey: id) } }

    /// **A render's first act**, on the engine's queue: every writer gives its weights back. The resident layers and head are
    /// not in any render's need.
    static func unloadResident() {
        dispatchPrecondition(condition: .onQueue(Engine.file))
        let writers = live.withLock { $0.values.compactMap(\.writer) }
        for writer in writers { writer.unload() }
    }

    private func unload() {
        generator?.unload()
        generator = nil; tokenizer = nil; tokenBytes = nil
    }

    /// The writer of Qwen-Image-2.1 in `library`; `nil` while its encoder, head or tokenizer is not installed.
    package convenience init?(library: Library = .processWide, limits: PromptWriterNeed.Limits = .init()) {
        guard let maps = PromptWriterMaps(library: library), let head = library.writerHeadMap(.qwenImage21),
              let processor = try? library.component(.qwenImage21, "processor") else { return nil }
        self.init(maps: maps, encoderPath: library.map(library.encoderMap(.qwenImage21)), headPath: head,
                  tokenizerPath: processor + "/tokenizer.json", limits: limits)
    }

    /// **Whether the feature can run on this Mac at its limits**, an image included — before any request, from
    /// shapes alone. `base`: the process's footprint (`OperatorFootprint.processBase()`).
    package func availability(base: Int = OperatorFootprint.processBase()) -> PromptWriterNeed.Availability {
        PromptWriterNeed.atLimits(maps: maps, limits: limits, base: base).availability(ceiling: ceiling())
    }

    // MARK: - Running

    /// Starts a run (after the one in flight, if any) and streams its events.
    package func write(_ request: Request) -> AsyncThrowingStream<Event, Error> {
        AsyncThrowingStream { continuation in
            let token = Cancellation()
            // Registered now, not when the run starts: a run still queued is cancelled too.
            let id = ObjectIdentifier(token)
            lock.withLock { pending[id] = token }
            continuation.onTermination = { _ in token.cancel() }
            Engine.file.async { [self] in
                defer { lock.withLock { _ = pending.removeValue(forKey: id) } }
                do {
                    // The machine's lock, as a render takes it: another process rendering refuses this run by name.
                    let machine = try RenderLock.acquire(RenderLock.machine)
                    defer { withExtendedLifetime(machine) {} }
                    try run(request, cancellation: token) { continuation.yield($0) }
                    continuation.finish()
                } catch EngineError.cancelled {
                    continuation.yield(.cancelled)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Stops the run in flight, between two layers, and the runs still waiting.
    package func cancel() { lock.withLock { Array(pending.values) }.forEach { $0.cancel() } }

    /// **Cancels, then gives the weights back** once the run has stopped: the resident layers and head, the
    /// tokenizer. Awaited before a render starts.
    package func release() async {
        cancel()
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            Engine.file.async { [self] in
                unload()
                done.resume()
            }
        }
    }

    private func run(_ request: Request, cancellation: Cancellation, emit: @escaping (Event) -> Void) throws {
        try cancellation.check()
        let tokenizer = try tokenizer ?? Tokenizer(file: tokenizerPath)
        self.tokenizer = tokenizer

        // ── the prompt, and the need it makes ─────────────────────────────────────────────────────
        let (system, user, image): (String?, String, Qwen3VLImages.RGB8?)
        switch request {
        case let .rewrite(text, instruction): (system, user, image) = (Self.system(instruction), text, nil)
        case let .describe(picture, instruction, ask): (system, user, image) = (Self.system(instruction), ask, picture)
        }
        let pixels = try image.map { try Qwen3VLChat.pixels($0).pixels }
        let tokens = try Qwen3VLPrompt.Tokens(header: Artifact.header(encoderPath), tokenizer: tokenizer)
        let prepared = try Qwen3VLChat.prepare(system: system, user: user, grids: pixels.map { [($0.gridHeight, $0.gridWidth)] } ?? [],
                                               tokenizer: tokenizer, tokens: tokens)
        guard prepared.ids.count <= limits.promptTokens else {
            throw Failure.promptTooLong(tokens: prepared.ids.count, limit: limits.promptTokens)
        }
        var need = PromptWriterNeed(maps: maps, promptTokens: prepared.ids.count, newTokens: limits.newTokens, patches: pixels?.patches)
        // The process's footprint now, less what this writer holds resident (that is in the need, not in the base).
        let measured = OperatorFootprint.processBase()
        need.base = self.generator == nil ? measured : max(0, measured - need.residentBytes)
        let availability = need.availability(ceiling: ceiling())
        guard case .available = availability else { throw Failure.unavailable(availability) }

        // ── the weights ─────────────────────────────────────────────────────────────────────────
        let generator: Qwen3VLGenerator
        if let kept = self.generator { generator = kept } else {
            let encoder = try Artifact(path: encoderPath), head = try Artifact(path: headPath)
            generator = try Qwen3VLGenerator(encoder: encoder, head: head, tokenizer: tokenizer)
            generator.cancellation = cancellation
            emit(.loading(0))
            try generator.load { emit(.loading($0)) }
            self.generator = generator
        }
        generator.cancellation = cancellation
        let table = tokenBytes ?? tokenizer.tokenBytes(rows: generator.vocabulary)
        tokenBytes = table

        // ── the decoding, streamed ──────────────────────────────────────────────────────────────
        var stream = UTF8Stream(), text = ""
        let output = try generator.generate(.init(ids: prepared.ids, positions: prepared.positions, images: pixels.map { [$0] } ?? []),
                                            maxNewTokens: limits.newTokens) { id in
            guard !generator.stops.contains(id), !tokenizer.specialIdentifiers.contains(id) else { return }
            let more = stream.push(Array(table.bytes(of: id)))
            if !more.isEmpty { text += more; emit(.text(text)) }
        }
        let rest = stream.finish()
        if !rest.isEmpty { text += rest; emit(.text(text)) }
        emit(.finished(text: text, tokens: output.tokens.count - (output.stopped ? 1 : 0)))
    }

    /// The system turn of an instruction: none for an empty one (the template writes a system turn when one is given).
    package static func system(_ instruction: String) -> String? {
        instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : instruction
    }
}
