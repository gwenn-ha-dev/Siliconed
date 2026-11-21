import Foundation

/// **The prompt writer**: a prompt in, a prompt out, written by an installed text encoder made to write
/// (`Qwen3VLGenerator`). Rewrite turns a text into another following an instruction (the system turn); Describe turns
/// a picture into a prompt.
///
///     write(request) ─▶ AsyncThrowingStream<Event>
///         prompt (template, tokenizer, picture) ─ over the limit? `Failure.promptTooLong`
///         `.loading` ─ the session planned under B (no plan fits: `EngineError.insufficientMemory`)
///         greedy decoding ─ each token's bytes ─ `UTF8Stream` ─ `.text(so far)` ─ `.finished(text, tokens)`
///     cancel()    the run in flight stops at its next token, the runs waiting before they start: `.cancelled`
///
/// What its use asks of it, and what a caller would get wrong:
///
///   - **It never runs during a render, and the engine sees to it.** A run is a job of the engine's queue
///     (`Engine.file`) and holds the machine's render lock (`RenderLock.machine`): the exclusion with a render, in
///     this process and in any other, is not a convention of the app.
///   - **One thing at a time**: a run waits for the one before it on the engine's queue.
///   - **The weights go back at the end of every run**: the session is the run's, let go with it. The next gesture
///     pays its pages again, some seconds before its first word.
///   - **A cancelled stream ends with `.cancelled`, not an error**; a consumer that stops reading cancels the run.
///   - **The text is cumulative**, and never ends inside a character (`UTF8Stream`).
///   - **A loop is cut** (`RepetitionGuard`): an exact cycle of the last tokens ends the run, the text keeping one
///     turn of it.
///   - **Greedy**, whatever the model's `generation_config.json` recommends: the same text for the same request.
package final class PromptWriter: @unchecked Sendable {
    /// **What a family's encoder needs to write**, declared in the family's folder (`FamilyTraits.writer`): the
    /// encoder's file (its head and final norm in it), the tokenizer's folder, and the model's shape.
    package struct Source: Sendable {
        package let map: String
        package let tokenizer: String
        package let config: Qwen3Encoder.Config
        package let tower: Qwen3VLVision.Config
        package init(map: String, tokenizer: String, config: Qwen3Encoder.Config, tower: Qwen3VLVision.Config = .eightB) {
            (self.map, self.tokenizer, self.config, self.tower) = (map, tokenizer, config, tower)
        }
    }

    package enum Request: Sendable {
        /// `instruction` is the system turn (empty: none); `text` the user turn.
        case rewrite(text: String, instruction: String)
        /// `instruction` is the system turn (empty: none); the user turn is the picture, then `ask` if any.
        case describe(image: Qwen3VLImages.RGB8, instruction: String, ask: String = "")
    }

    package enum Event: Sendable, Equatable, Codable {
        /// The weights being brought in, 0…1 (once a run).
        case loading(Double)
        /// The text so far (cumulative).
        case text(String)
        /// The whole answer, and the tokens it took (the stop token not counted).
        case finished(text: String, tokens: Int)
        case cancelled
    }

    package enum Failure: Error, CustomStringConvertible, Equatable, Codable {
        case notInstalled
        case promptTooLong(tokens: Int, limit: Int)

        package var description: String {
            switch self {
            case .notInstalled: return "the writer is not installed"
            case let .promptTooLong(tokens, limit): return "the prompt is \(tokens) tokens, the writer reads at most \(limit)"
            }
        }
    }

    package struct Limits: Sendable, Equatable {
        /// The prompt's tokens, template, instruction and picture tokens included.
        package var promptTokens = 2048
        /// The tokens the answer may take.
        package var newTokens = 768
        package init() {}
        package init(promptTokens: Int, newTokens: Int) { (self.promptTokens, self.newTokens) = (promptTokens, newTokens) }
    }

    private let source: Source
    private let limits: Limits
    private let lock = NSLock()
    /// Every run not yet finished, queued or in flight: the tokens `cancel()` reaches.
    private var pending: [ObjectIdentifier: Cancellation] = [:]

    package init(source: Source, limits: Limits = .init()) {
        (self.source, self.limits) = (source, limits)
    }

    /// **The family whose installed encoder writes**, in `library`: the first that declares a writer and has its
    /// files there.
    package static func family(in library: Library = .processWide) -> Family? {
        Family.allCases.first { (try? $0.traits.writer?(library)) != nil }
    }

    /// The writer of `library`; `nil` while no family's writer is installed.
    package convenience init?(library: Library = .processWide, limits: Limits = .init()) {
        guard let source = Family.allCases.lazy.compactMap({ try? $0.traits.writer?(library) }).first else { return nil }
        self.init(source: source, limits: limits)
    }

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
                    // The machine's lock, as a render takes it: another process rendering refuses this run by name. A
                    // run has no journal: one the system leaves unguarded runs without a word.
                    let machine = try RenderLock.acquire(RenderLock.machine) { _ in }
                    defer { withExtendedLifetime(machine) {} }
                    try run(request, cancellation: token) { continuation.yield($0) }
                    continuation.finish()
                } catch EngineError.cancelled {
                    continuation.yield(.cancelled)
                    continuation.finish()
                } catch let failure as Failure {
                    continuation.finish(throwing: failure)
                } catch {
                    continuation.finish(throwing: EngineError(error))
                }
            }
        }
    }

    /// Stops the run in flight at its next token, and the runs still waiting.
    package func cancel() { lock.withLock { Array(pending.values) }.forEach { $0.cancel() } }

    /// **The request's chat**, tokenized: what `write` continues, and what says whether a request is within the limits
    /// before anything is loaded.
    package func prompt(_ request: Request, tokenizer: Tokenizer) throws -> Qwen3VLGenerator.Prompt {
        let (system, user, image): (String?, String, Qwen3VLImages.RGB8?)
        switch request {
        case let .rewrite(text, instruction): (system, user, image) = (Self.system(instruction), text, nil)
        case let .describe(picture, instruction, ask): (system, user, image) = (Self.system(instruction), ask, picture)
        }
        let pictures = try image.map { [try Qwen3VLChat.pixels($0)] } ?? []
        let prepared = try Qwen3VLChat.prepare(system: system, user: user, grids: pictures.map { (height: $0.gridHeight, width: $0.gridWidth) },
                                               tokenizer: tokenizer, merge: source.tower.merge)
        guard prepared.ids.count <= limits.promptTokens else {
            throw Failure.promptTooLong(tokens: prepared.ids.count, limit: limits.promptTokens)
        }
        return .init(ids: prepared.ids, slots: prepared.slots, pictures: pictures)
    }

    private func run(_ request: Request, cancellation: Cancellation, emit: @escaping (Event) -> Void) throws {
        try cancellation.check()
        let tokenizer = try Tokenizer(directory: source.tokenizer)
        let prompt = try prompt(request, tokenizer: tokenizer)
        let generator = try Qwen3VLGenerator(map: source.map, config: source.config, tower: source.tower)
        guard generator.writes else { throw Failure.notInstalled }
        let table = tokenizer.tokenBytes(rows: source.config.vocabulary)
        let stops = Qwen3VLChat.stops(tokenizer)
        emit(.loading(0))

        // ── the decoding, streamed ──
        var stream = UTF8Stream(), text = ""
        // The identifiers written (the stop and special tokens apart), for the loop guard.
        var written: [Int] = [], loop: Int?, first = true
        let output = try generator.generate(prompt, limit: limits.newTokens, stops: stops) { id in
            if first { first = false; emit(.loading(1)) }
            guard !cancellation.isCancelled else { return false }
            guard !tokenizer.specialIdentifiers.contains(id) else { return true }
            written.append(id)
            if let keep = RepetitionGuard.cut(written) { loop = keep; return false }
            let more = stream.push(Array(table.bytes(of: id)))
            if !more.isEmpty { text += more; emit(.text(text)) }
            return true
        }
        try cancellation.check()
        if let loop {
            // The model went round in a circle: what was shown so far shrinks to the text before the second turn.
            written.removeSubrange(loop...)
            var again = UTF8Stream()
            text = written.map { again.push(Array(table.bytes(of: $0))) }.joined() + again.finish()
            emit(.text(text))
            emit(.finished(text: text, tokens: written.count))
            return
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
