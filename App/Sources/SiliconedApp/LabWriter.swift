// **The Prompt Lab's writer, over the engine** (`PromptWriter`): the installed text encoder that writes, as the window
// and `silicontrol` ask it (`Lab.swift`, `LabRemote.swift`).
//
//     write(input, instruction) ─▶ the engine's process (`EngineWorker`) ─ image Data ─ ImageIO, EXIF, alpha over white
//                                  ─▶ RGB8 (`PromptWriterImage`) ─ PromptWriter.write(Request) ─ its events as they are:
//                                  loading · text · finished · cancelled
//                                  a failure ─▶ one sentence in the user's language, and its stable code
//
// - **Nothing is read before a gesture**: `isInstalled` looks for the files of a family that declares a writer, no more.
//   Each gesture reads the weights in the engine's process and gives them back when it ends; the engine plans the memory
//   and refuses a run no plan fits (`insufficient_memory`), which comes back as that gesture's failure.
// - **`text` carries the whole text so far, not a delta**: the window shows what it is given.
// - **Cancelling is the consumer's**: the window cancels the task reading the stream, and the stream's end stops the
//   run at its next token.

import Foundation
import Siliconed

/// What the window gives the writer.
enum LabInput: Sendable, Equatable {
    /// A prompt to improve.
    case text(String)
    /// An image to describe: the file's bytes as they were dropped or opened (any format the system reads).
    case image(Data)

    /// The gesture it is: Rewrite reads a text, Describe an image.
    var gesture: PromptWriterInstructions.Gesture {
        switch self {
        case .text: .rewrite
        case .image: .describe
        }
    }
}

/// What went wrong while writing: one sentence in the user's language, and the engine's stable `code` when it has one
/// (`render_already_running`…) — the window shows the sentence, `silicontrol` the two.
struct LabWriterFailure: LocalizedError {
    let errorDescription: String?
    let code: String?
    init(_ sentence: String, code: String? = nil) { errorDescription = sentence; self.code = code }
}

struct LabWriter: Sendable {
    let library: Library

    /// Some installed family's encoder writes.
    var isInstalled: Bool { PromptWriter.family(in: library) != nil }

    /// `instruction` is the system turn: ours for the family written for and the gesture, or the user's
    /// (`LabModel.instruction`). The stream ends after `finished` or `cancelled`, or throws a `LabWriterFailure`.
    func write(_ input: LabInput, instruction: String) -> AsyncThrowingStream<PromptWriter.Event, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let events = switch input {
                    case .text(let text): EngineWorker.shared.write(rewrite: text, instruction: instruction, library: library)
                    case .image(let data): EngineWorker.shared.write(describe: data, instruction: instruction, library: library)
                    }
                    // Leaving this loop on cancellation ends the engine's stream, which stops its run.
                    for try await event in events { continuation.yield(event) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: failure(error))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// One sentence, in the user's language, for what can go wrong — and the stable code a command says it by.
    private func failure(_ error: Error) -> LabWriterFailure {
        switch error {
        // The writer's own sentence: the render's speaks of a size and of images. Where the Standard version's encoder
        // is the one that writes, its Compact version is the way under this Mac's budget (it writes under 10 GiB).
        case let EngineError.insufficientMemory(needed, budget):
            // Within the limit: every form stopped the fuse (`Stage.choose`), the number known is a socle, not a need.
            let need = needed <= budget
                ? String(localized: "The writer went past this Mac’s memory limit of \(Figures.memory(budget)) every time it was tried: it is refused.")
                : String(localized: "The writer needs \(Figures.memory(needed)) of memory, more than this Mac’s limit of \(Figures.memory(budget)).")
            guard let family = PromptWriter.family(in: library), library.variant(of: family) == .standard else {
                return LabWriterFailure(need, code: "insufficient_memory")
            }
            return LabWriterFailure(need + " " + String(localized: "Install the Compact version of \(family.name) to write on this Mac."), code: "insufficient_memory")
        case PromptWriter.Failure.notInstalled:
            return LabWriterFailure(String(localized: "No writer is installed yet. Install a model to have one."), code: WriterPause.notInstalled.code)
        case let PromptWriter.Failure.promptTooLong(tokens, limit):
            return LabWriterFailure(String(localized: "This text is too long for the writer: \(tokens) tokens, at most \(limit)."), code: "prompt_too_long")
        case let EngineError.engineStopped(budget?, _):
            return LabWriterFailure(String(localized: "The writer went past this Mac’s memory limit of \(Figures.memory(budget)): the engine was stopped. Try again: it starts afresh."),
                                    code: EngineError.engineStopped(budget: nil, detail: "").code)
        case EngineError.engineStopped:
            return LabWriterFailure(String(localized: "The engine stopped unexpectedly while writing. Try again: it starts afresh."),
                                    code: EngineError.engineStopped(budget: nil, detail: "").code)
        case EngineError.imageUnreadable:
            return LabWriterFailure(String(localized: "This image could not be read."), code: "ref")
        default:
            let problem = Problem(error)
            return LabWriterFailure(problem.title, code: problem.code)
        }
    }
}
