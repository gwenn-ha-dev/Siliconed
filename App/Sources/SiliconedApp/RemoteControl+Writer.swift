// `rewrite` and `describe`: the Prompt Lab's two gestures, from the terminal;
// `instructions`, Settings › Prompt Lab: what the writer is told, the user's to rewrite; and `settings`, the window that shows it.
// They go through the lab's own functions (`LabRemote.swift`, `Lab.swift`), so a command and a click write the same text.
//
// - **The refusals are the lab's** (`LabModel.pause`): the window follows another Mac, an image renders, no writer installed.
//   Each has a stable code (`WriterPause.code`); the sentence is the one the lab shows.
// - **Incognito**: the text goes out on the socket only. A client that hangs up cancels the gesture.

import Foundation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    // ── rewrite, describe ──

    /// `rewrite "<text>" [--for ID]` and `describe <image|i7> [--for ID]`.
    func writing(_ gesture: WriterOrder.Gesture, _ arguments: [String], cwd: String, _ c: Connection) throws {
        let order: WriterOrder
        switch WriterOrder.parse(gesture, arguments) {
        case .success(let parsed): order = parsed
        case .failure(let failure): throw Self.refusal(failure, help: gesture.rawValue)
        }
        let lab = LabModel.shared
        lab.watchRender()
        lab.refresh()
        // The queue is read here, not the lab's copy of it: a job added a moment ago counts.
        if let pause = lab.pause(rendering: local.busy) { throw Self.refusal(pause) }
        guard !lab.isWriting else {
            throw Refusal("writer_busy", String(localized: "Another Rewrite or Describe is under way."), help: "silicontrol wait, then retry")
        }
        let family = try writerTarget(order.target, lab)
        let input: LabInput, guarded: LabInputs.MentionGuard?
        var subject: String?
        switch gesture {
        case .rewrite:
            (input, guarded) = LabModel.rewriteInput(order.subject.trimmingCharacters(in: .whitespacesAndNewlines),
                                                     capacity: lab.capacity(for: family))
        case .describe:
            let (data, name) = try describedImage(order.subject, cwd: cwd)
            input = .image(data); guarded = nil; subject = name
        }

        emit(c, ["type": "started", "gesture": gesture.rawValue, "for": family.rawValue])
        let finished = Finished()
        lab.writeForRemote(input, for: family, guarded: guarded, progress: { [self] event in
            switch event {
            case .loading(let fraction): emit(c, ["type": "loading", "fraction": fraction.map { Response.number($0, 3) } ?? NSNull()])
            case .text(let text): emit(c, ["type": "text", "text": text])
            }
        }, finish: { [self] outcome in
            finished.done = true
            switch outcome {
            case .written(let w):
                var o: [String: Any] = ["ok": true, "gesture": gesture.rawValue, "for": family.rawValue, "text": w.text, "tokens": w.tokens]
                if let subject { o["image"] = subject } else { o["lost_mentions"] = w.lost }
                if let notice = w.notice { o["notice"] = notice }
                finish(c, o)
            case .failed(let code, let message):
                finish(c, Response.failure(code, message))
            case .stoppedByRender:
                finish(c, Response.failure("stopped", String(localized: "Stopped: an image started rendering."), help: "silicontrol wait, then retry"))
            }
        })
        // A client that hangs up before the end takes the gesture with it.
        Task { @MainActor in
            while !finished.done {
                try? await Task.sleep(for: .milliseconds(250))
                if !finished.done, c.isGone { lab.cancelRemote(); return }
            }
        }
    }

    private final class Finished: @unchecked Sendable { var done = false }

    /// The family a gesture writes for: `--for`, among the installed ones — else the lab's own choice.
    private func writerTarget(_ word: String?, _ lab: LabModel) throws -> Family {
        if let word {
            let cards = local.cards
            guard let family = WriterTarget.named(word, installed: lab.families, modelFamily: { id in cards.first { $0.id == id }?.family }) else {
                let installed = lab.families.map(\.rawValue).joined(separator: ", ")
                throw Refusal("model", String(localized: "unknown model: \"\(word)\""), help: String(localized: "models: \(installed)"))
            }
            return family
        }
        guard let family = lab.target else { throw Self.refusal(.notInstalled) }
        return family
    }

    /// The image to describe, as the lab holds one: its bytes as they are, read upright (`SubjectPictures.upright`) — a history
    /// image (`i7`), else a file.
    private func describedImage(_ word: String, cwd: String) throws -> (Data, String) {
        if let entry = image(word) { return (entry.png, entry.name) }
        let path = absolute(word, cwd)
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)), SubjectPictures.upright(data) != nil else {
            if word.first == "i", Int(word.dropFirst()) != nil {
                throw Refusal("unknown_image", String(localized: "no image \"\(word)\" in the history"), help: "silicontrol history")
            }
            throw Refusal("ref", String(localized: "image unreadable or missing: \(path)"))
        }
        return (data, path)
    }

    // ── The refusals ──

    static func refusal(_ pause: WriterPause) -> Refusal {
        let hint: String? = switch pause {
        case .following: String(localized: "Render › Compute On › This Mac, then retry")
        case .rendering: "silicontrol wait, then retry"
        // The model whose encoder writes, by the id `install` takes.
        case .notInstalled: Family.allCases.first { $0.traits.writer != nil }.map { "silicontrol install \($0.rawValue)" }
        }
        return Refusal(pause.code, LabModel.presentation(pause).sentence, help: hint)
    }

    /// `help`: the command whose page the refusal points to (`rewrite`, `describe`, `instructions`).
    static func refusal(_ failure: WriterOrder.Failure, help command: String) -> Refusal {
        let help = "silicontrol help \(command)"
        switch failure {
        case .usage(let line): return Refusal("usage", line, help: help)
        // `rewrite` takes a text, which may begin with `--`.
        case .unknownFlag(let flag): return command == "rewrite" ? unknownFlag(flag, text: true)
                                                                 : Refusal("usage", String(localized: "unknown flag: \(flag)"), help: help)
        case .flagWithoutValue(let flag): return Refusal("usage", String(localized: "\(flag) expects a value"), help: help)
        case .emptyText: return Refusal("usage", String(localized: "empty text"), help: help)
        }
    }

    // ── instructions ──

    /// **`instructions [rewrite|describe] [--for ID] [--set "<text>" | --file PATH | --restore]`**: Settings › Prompt Lab, by the
    /// same `LabModel.instructions` the pane's field writes. The text is the field's (theirs as typed, or ours); `"sent"` says
    /// whose text goes to the writer — a blank one of theirs sends ours. Allowed while the window follows another Mac: the
    /// instructions are this Mac's, as its writer is.
    func instructions(_ arguments: [String], cwd: String) throws -> [String: Any] {
        let order: InstructionOrder
        switch InstructionOrder.parse(arguments) {
        case .success(let parsed): order = parsed
        case .failure(let failure): throw Self.refusal(failure, help: "instructions")
        }
        let lab = LabModel.shared
        let family = try order.target.map(instructionFamily)
        func row(_ f: Family, _ g: PromptWriterInstructions.Gesture) -> [String: Any] {
            ["for": f.rawValue, "gesture": g.rawValue, "edited": lab.instructions.isEdited(f, g),
             "sent": lab.instructions.sent(f, g) == nil ? "ours" : "yours"]
        }
        guard let gesture = order.gesture else {
            let families = family.map { [$0] } ?? Family.allCases
            return ["ok": true, "instructions": families.flatMap { f in PromptWriterInstructions.Gesture.allCases.map { row(f, $0) } }]
        }
        guard let target = family ?? lab.target else {
            throw Refusal("usage", String(localized: "no model installed: name one with --for"), help: "silicontrol help instructions")
        }
        switch order.change {
        case .set(let text): lab.instructions.set(text, for: target, gesture)
        case .file(let path):
            let full = absolute(path, cwd)
            guard let text = try? String(contentsOfFile: full, encoding: .utf8) else {
                throw Refusal("unreadable", String(localized: "cannot read \(full) as UTF-8 text"))
            }
            lab.instructions.set(text, for: target, gesture)
        case .restore: lab.instructions.set(nil, for: target, gesture)
        case nil: break
        }
        return row(target, gesture).merging(["ok": true, "text": lab.instructions.text(target, gesture),
                                            "instruction": lab.instructions.instruction(for: target, gesture)]) { a, _ in a }
    }

    /// What `--for` names for an instruction: any family, installed or not — the instructions are the app's, not the library's.
    private func instructionFamily(_ word: String) throws -> Family {
        let cards = local.cards
        guard let f = WriterTarget.named(word, installed: Family.allCases, modelFamily: { id in cards.first { $0.id == id }?.family }) else {
            throw Refusal("model", String(localized: "unknown model: \"\(word)\""),
                          help: String(localized: "models: \(Family.allCases.map(\.rawValue).joined(separator: ", "))"))
        }
        return f
    }

    // ── settings ──

    /// **`settings [PANE]`**: Siliconed › Settings… (⌘,), on that pane (`SettingsWindow.show`, the tabs' `select`); without one,
    /// on the pane last shown. **`settings lab [rewrite|describe] [--for ID]`**: Settings › Prompt Lab on one instruction, as a
    /// click on the Prompt Lab's « Your Instruction » opens it (`showInstruction`) — without `--for`, the model the Lab writes
    /// for. The window is this Mac's: allowed while the window follows another one.
    func settings(_ action: GestureOrder.SettingsAction) throws -> [String: Any] {
        let window = SettingsWindow.shared
        switch action {
        case .show(let pane):
            window.show(pane)
        case .instruction(let gesture, let word):
            let family = try word.map(instructionFamily) ?? LabModel.shared.target
            window.showInstruction(family, gesture)
        }
        var o: [String: Any] = ["pane": window.shown.pane.rawValue]
        if window.shown.pane == .lab {
            o["for"] = window.shown.family.rawValue
            o["gesture"] = window.shown.gesture.rawValue
        }
        return ["ok": true, "settings": o]
    }

    /// What `status` says of the writer: `ready`, or the stable code of `rewrite`'s refusal (the render, which passes, apart).
    func writerState() -> [String: Any] {
        let lab = LabModel.shared
        lab.refresh()
        return ["state": lab.installed ? "ready" : WriterPause.notInstalled.code]
    }
}
