// **The Prompt Lab's two gestures, asked from the terminal** (`silicontrol rewrite`, `silicontrol describe`).
//
//     command ──▶ LabModel.pause / target / rewriteInput      the lab's own refusals, target and preparation
//             ──▶ LabModel.writeForRemote ──▶ writer.write    the lab's writer (`EngineLabWriter`), the lab's instructions
//             ◀── loading · text (cumulative) · written / failed / stopped
//
// - **Same road as the window, minus the field.** The writer, its instructions per family and gesture, the protection of the
//   `@names` and the image's reading are the lab's (`Lab.swift`, `EngineLabWriter.swift`); only the destination differs: the
//   text goes back to the command, never into a field, and nothing is kept — no file, no history (incognito).
// - **One writer, one text at a time.** A command holds the writer as a gesture of the window does (`LabModel.isWriting`): the
//   window's buttons wait, and a second command is refused (`writer_busy`).
// - **The render has priority**, here as in the window: one that starts stops the command's gesture (`stopped`).
// - **A client that hangs up cancels the gesture** (`cancelRemote`): the stream's termination stops the sampling
//   (`EngineLabWriter.write`, `PromptWriter.cancel`) and the weights go back.

import Foundation
import Siliconed

extension LabModel {
    enum RemoteEvent: Sendable {
        case loading(Double?)
        /// Everything written so far, the mentions already put back.
        case text(String)
    }

    struct RemoteWritten: Sendable {
        let text: String
        let tokens: Int
        /// The mentions the writer lost or deformed, and the sentence the window would show (`nil`: all kept).
        let lost: [String]
        let notice: String?
    }

    enum RemoteOutcome: Sendable {
        case written(RemoteWritten)
        case failed(code: String, message: String)
        /// A render started and took the writer's memory.
        case stoppedByRender
    }

    /// Starts a gesture for a command. `finish` is called once, unless `cancelRemote` took the gesture away first.
    /// The caller has already judged `pause` and `isWriting`.
    func writeForRemote(_ input: LabInput, for family: Family, guarded: LabInputs.MentionGuard?,
                        progress: @escaping @MainActor (RemoteEvent) -> Void,
                        finish: @escaping @MainActor (RemoteOutcome) -> Void) {
        let ended = Box()
        func end(_ outcome: RemoteOutcome) {
            guard !ended.done else { return }
            ended.done = true
            remote = nil
            finish(outcome)
        }
        let task = Task { [writer] in
            do {
                var last = "", tokens = 0
                for try await event in writer.write(input, instruction: instruction(for: family, input)) {
                    if Task.isCancelled { return }
                    switch event {
                    case .loading(let fraction): progress(.loading(fraction))
                    case .text(let t): last = t; progress(.text(Self.rewritten(t, guarded: guarded).text))
                    case .finished(let t, let n): last = t; tokens = n
                    case .cancelled:
                        // The writer stopped by itself: for a render (the window's `renderChanged`) or because it was released.
                        end(.stoppedByRender)
                        return
                    }
                }
                if Task.isCancelled { return }
                let clean = last.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !clean.isEmpty else {
                    end(.failed(code: "writer_failed", message: String(localized: "The writer wrote nothing.")))
                    return
                }
                let back = Self.rewritten(clean, guarded: guarded)
                end(.written(RemoteWritten(text: back.text, tokens: tokens, lost: back.lost, notice: back.notice)))
            } catch {
                if Task.isCancelled { return }
                let failure = error as? LabWriterFailure
                end(.failed(code: failure?.code ?? "writer_failed", message: error.localizedDescription))
            }
        }
        remote = (task, stop: { task.cancel(); end(.stoppedByRender) })
    }

    /// The client is gone: the gesture ends without an answer, the writer gives its memory back.
    func cancelRemote() {
        guard let remote else { return }
        self.remote = nil
        remote.task.cancel()
        Task { await writer.release() }
    }

    private final class Box: @unchecked Sendable { var done = false }
}
