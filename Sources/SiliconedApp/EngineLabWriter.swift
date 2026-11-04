// **The lab's writer, over the engine** (`PromptWriter`): Qwen-Image-2.1's Qwen3-VL-8B text encoder, made to write.
//
//     window ─ write(input, target) ─▶ instruction (ours, per family and gesture: `PromptWriterInstructions`)
//                                      image Data ─ ImageIO, EXIF, alpha over white ─▶ RGB8 (`PromptWriterImage`)
//                                      PromptWriter.write(Request) ─ events, one to one ─▶ LabEvent
//
// - **Nothing is read at launch.** The engine writer is built at the first `availability` (headers and the tokenizer's tables,
//   no weight); each gesture reads the weights and gives them back when it ends (`PromptWriter`); `release()` (a render starting,
//   the window closing) stops the gesture in flight and drops the rest.
// - **Availability is honest** (`PromptWriterStatus`): the encoder missing is « not installed »; the encoder there and the head
//   not is its own state, with the download's size (the lab offers it); an engine that does not fit says the Mac that would.

import Foundation
import Siliconed

final class EngineLabWriter: LabWriter, @unchecked Sendable {
    private let lock = NSLock()
    private var _library: Library
    private var engine: PromptWriter?

    init(library: Library) { _library = library }

    /// The library the window follows (it can move): the engine writer of the old one is dropped.
    var library: Library {
        get { lock.withLock { _library } }
        set {
            let old: PromptWriter? = lock.withLock {
                guard newValue.root != _library.root else { return nil }
                _library = newValue
                defer { engine = nil }
                return engine
            }
            if let old { Task { await old.release() } }
        }
    }

    /// The engine writer, built once everything it reads is installed; `nil` before that.
    private func current() -> PromptWriter? {
        lock.withLock {
            if engine == nil { engine = PromptWriter(library: _library) }
            return engine
        }
    }

    var availability: LabAvailability {
        let status = library.writerStatus { self.current() }
        // A removed model leaves no stale writer behind.
        switch status {
        case .notInstalled, .needsHead:
            if let old = lock.withLock({ defer { engine = nil }; return engine }) { Task { await old.release() } }
        default: break
        }
        switch status {
        case .notInstalled: return .unavailable(.notInstalled)
        case .needsHead(let bytes): return .unavailable(.needsHead(bytes: Int64(bytes)))
        case .needsMac(let gigabytes): return .unavailable(.needsMemory(gigabytes: gigabytes))
        case .ready(let bytes): return .ready(name: "Qwen3-VL-8B", size: Int64(bytes))
        }
    }

    func write(_ input: LabInput, instruction: String) -> AsyncThrowingStream<LabEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [self] in
                do {
                    guard let engine = current() else { throw PromptWriter.Failure.notInstalled }
                    let request: PromptWriter.Request
                    switch input {
                    case .text(let text):
                        request = .rewrite(text: text, instruction: instruction)
                    case .image(let data):
                        request = .describe(image: try PromptWriterImage.rgb8(from: data), instruction: instruction)
                    }
                    for try await event in engine.write(request) {
                        switch event {
                        case .loading(let fraction): continuation.yield(.loading(fraction))
                        case .text(let text): continuation.yield(.text(text))
                        case .finished(let text, let tokens): continuation.yield(.finished(text, tokens: tokens))
                        case .cancelled: continuation.yield(.cancelled)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: Self.failure(error))
                }
            }
            continuation.onTermination = { [self] _ in
                task.cancel()
                lock.withLock { engine }?.cancel()
            }
        }
    }

    func release() async {
        let held = lock.withLock { engine }
        await held?.release()
    }

    /// One sentence, in the user's language, for what can go wrong — and the stable code a command says it by.
    private static func failure(_ error: Error) -> LabWriterFailure {
        switch error {
        case PromptWriter.Failure.notInstalled:
            return LabWriterFailure(String(localized: "No writer is installed yet. Install a model to have one."), code: "writer_absent")
        case PromptWriter.Failure.unavailable(.needsMac(let n, _)):
            return LabWriterFailure(String(localized: "The writer needs a Mac with \(n) GB of memory."), code: "writer_memory")
        case let PromptWriter.Failure.promptTooLong(tokens, limit):
            return LabWriterFailure(String(localized: "This text is too long for the writer: \(tokens) tokens, at most \(limit)."), code: "prompt_too_long")
        case EngineError.imageUnreadable:
            return LabWriterFailure(String(localized: "This image could not be read."), code: "ref")
        default:
            let problem = Problem(error)
            return LabWriterFailure(problem.title, code: problem.code)
        }
    }
}
