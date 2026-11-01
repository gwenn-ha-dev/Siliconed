// `rewrite`, `describe` and `install writer`: the Prompt Lab's two gestures and its one button, from the terminal.
// They go through the lab's own functions (`LabRemote.swift`, `Lab.swift`), so a command and a click write the same text.
//
// - **The refusals are the lab's** (`LabModel.pause`): the window follows another Mac, an image renders, no writer installed,
//   the head to download, a Mac too small. Each has a stable code (`WriterPause.code`); the sentence is the one the lab shows.
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
        case .failure(let failure): throw Self.refusal(failure, gesture)
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
        // A client that hangs up before the end takes the gesture with it: the sampling stops, the weights go back.
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
        case .notInstalled: "silicontrol models"
        case .needsHead: "silicontrol install writer"
        case .needsMemory: nil
        case .preparing: "silicontrol status"
        }
        return Refusal(pause.code, LabModel.presentation(pause).sentence, help: hint)
    }

    static func refusal(_ failure: WriterOrder.Failure, _ gesture: WriterOrder.Gesture) -> Refusal {
        let help = "silicontrol help \(gesture.rawValue)"
        switch failure {
        case .usage(let line): return Refusal("usage", line, help: help)
        case .unknownFlag(let flag): return Refusal("usage", String(localized: "unknown flag: \(flag)"), help: help)
        case .flagWithoutValue(let flag): return Refusal("usage", String(localized: "\(flag) expects a value"), help: help)
        case .emptyText: return Refusal("usage", String(localized: "empty text"), help: help)
        }
    }

    // ── install writer ──

    /// **`install writer`**: the lab's « Install the Writer » button (`LabModel.installHead`), the 1.25 GB head of Qwen-Image-2.1's
    /// text encoder. No sheet and no acceptance of its own: it is the licence of Qwen-Image-2.1, which the encoder's installation
    /// already had accepted (`Installer.installWriterHead` refuses otherwise, and the command says so first). Answered at once;
    /// `status` tells the progress (`"installing"`) and the end (`"writer"`).
    func installWriter(_ arguments: [String]) throws -> [String: Any] {
        if let flag = arguments.first(where: { $0.hasPrefix("--") }) {
            throw Refusal("usage", String(localized: "unknown flag: \(flag)"), help: "silicontrol help install")
        }
        let lab = LabModel.shared
        lab.refresh()
        if lab.installingHead {
            return ["ok": true, "installed": false, "installing": true,
                    "message": String(localized: "the writer is being installed: silicontrol status says when it is done")]
        }
        switch lab.availability {
        case .ready:
            return ["ok": true, "installed": true, "message": String(localized: "already installed")]
        case .unavailable(.notInstalled):
            throw Refusal(WriterPause.notInstalled.code, LabModel.presentation(.notInstalled).sentence,
                          help: "silicontrol install qwen-image-2.1")
        case .unavailable(.needsMemory(let n)):
            throw Refusal(WriterPause.needsMemory(gigabytes: n).code, LabModel.presentation(.needsMemory(gigabytes: n)).sentence)
        case .unavailable(.preparing):
            throw Refusal(WriterPause.preparing.code, LabModel.presentation(.preparing).sentence, help: "silicontrol status")
        case .unavailable(.needsHead(let bytes)):
            guard local.managementPossible else {
                throw Refusal("busy", String(localized: "a render, an installation or the diagnostic is running"), help: "silicontrol wait, then retry")
            }
            guard Licenses.implicit || local.library.isLicenseAccepted(ModelCard.of(Library.writerFamily)) else {
                throw Refusal("license_not_accepted", String(localized: "The writer’s head comes with Qwen-Image-2.1, whose license is not accepted yet."),
                              help: "silicontrol install qwen-image-2.1")
            }
            lab.installHead(local)
            return ["ok": true, "installed": false, "installing": true, "download_bytes": NSNumber(value: bytes),
                    "message": String(localized: "the writer is being installed: silicontrol status says when it is done")]
        }
    }

    /// What `status` and `models` say of the writer: `ready`, or why not (the stable code of `rewrite`'s refusal, without the
    /// render, which passes), and the head's install under way or failed.
    func writerState() -> [String: Any] {
        let lab = LabModel.shared
        lab.refresh()
        var o: [String: Any] = [:]
        switch lab.availability {
        case .ready(_, let size): o["state"] = "ready"; o["bytes"] = NSNumber(value: size)
        case .unavailable(.notInstalled): o["state"] = WriterPause.notInstalled.code
        case .unavailable(.needsMemory(let n)): o["state"] = WriterPause.needsMemory(gigabytes: n).code; o["gigabytes"] = n
        case .unavailable(.preparing): o["state"] = WriterPause.preparing.code
        case .unavailable(.needsHead(let bytes)): o["state"] = WriterPause.needsHead(bytes: bytes).code; o["download_bytes"] = NSNumber(value: bytes)
        }
        o["installing"] = lab.installingHead
        o["failure"] = lab.headFailure ?? NSNull()
        return o
    }
}
