// **`lab`: the Prompt Lab's bar, from the terminal** — what it holds, folded or unfolded, an image in, a role, an image out, and its
// Rewrite and Describe written in its text. Each calls what the bar calls (`LabModel`, `AppState+Inputs.swift`), on the rack the
// window shows; the rules are `LabInputs`'.
//
// - **An image comes in as a drop brings it**: a file as the file, an image of the history as « Send to the Prompt Lab » does,
//   with the role a newcomer takes (`LabInputs.defaultRole`). All are read before any comes in: one unreadable file refuses the
//   command whole.
// - **A role the rack's model lacks is refused**, as the bar's menu does not offer it; the images keep theirs across a change of
//   model, without effect where the model lacks them (`"used": false`).
// - **`lab rewrite` and `lab describe N` press the bar's buttons** (`LabModel.rewrite`, `describe`): the text is written in the
//   field, token by token with its marks, as the click writes it, ⌘Z included — unlike `rewrite` and `describe`, which answer on
//   the socket and leave the field alone. The bar unfolds first: the buttons are the unfolded plate's. The gesture is then the
//   window's: it answers at once (`lab` says where it stands, `"writing"`), or with `--wait` once written; a client that hangs
//   up leaves it writing, and the bar's Stop or `lab close` stops it.

import AppKit
import Foundation
import Siliconed

extension RemoteControl {
    func lab(_ action: GestureOrder.LabAction, cwd: String, _ c: Connection) throws {
        let lab = LabModel.shared
        let rack = lab.rack
        func picture(_ n: Int) throws -> InputImage {
            guard rack.inputs.indices.contains(n - 1) else {
                throw Refusal("image", String(localized: "no image \(String(n)) in the Prompt Lab"), help: "silicontrol lab")
            }
            return rack.inputs[n - 1]
        }
        switch action {
        case .status:
            break
        case .unfold:
            lab.unfold()
        case .fold:
            lab.fold()
        case .detach:
            lab.detached = true
        case .attach:
            lab.detached = false
        case .add(let words):
            var images: [(ReferenceImage, Data?)] = []
            for word in words {
                if let entry = image(word) {
                    guard let r = rack.fromHistory(entry) else {
                        throw Refusal("ref", String(localized: "image unreadable or missing: \(word)"))
                    }
                    images.append((r, entry.png))
                } else {
                    let url = URL(fileURLWithPath: absolute(word, cwd))
                    guard let r = rack.read(url) else {
                        let path = url.path
                        throw Refusal("ref", String(localized: "image unreadable or missing: \(path)"))
                    }
                    images.append((r, nil))
                }
            }
            for (image, bytes) in images { rack.addInput(image, bytes: bytes) }
        case .role(let n, let role):
            let p = try picture(n)
            guard LabInputs.roles(capacity: rack.imageCapacity, startsFromImage: rack.startsFromImage).contains(role) else {
                let name = rack.card?.name ?? rack.identifier
                throw Refusal("role", String(localized: "\(name) has no role “\(role.label)”"), help: "silicontrol lab")
            }
            rack.setRole(role, of: p.id)
        case .remove(let n):
            lab.remove(try picture(n))
        case .rewrite(let wait):
            try writeInText(nil, wait: wait, c)
            return
        case .describe(let n, let wait):
            try writeInText(try picture(n), wait: wait, c)
            return
        }
        finish(c, ["ok": true, "lab": labJSON()])
    }

    // MARK: Rewrite and Describe, in the text

    /// The refusals first, in the order `rewrite` and `describe` give them (`WriterPause.decide`, then a gesture under way), and
    /// those of the button itself: greyed on an empty text, an image the lab cannot read.
    private func writeInText(_ picture: InputImage?, wait: Bool, _ c: Connection) throws {
        let lab = LabModel.shared
        lab.watchRender()
        lab.refresh()
        // The queue is read here, as `rewrite` reads it: a job added a moment ago counts.
        if let pause = lab.pause(rendering: local.busy) { throw Self.refusal(pause) }
        guard !lab.isWriting else {
            throw Refusal("writer_busy", String(localized: "Another Rewrite or Describe is under way."), help: "silicontrol wait, then retry")
        }
        if picture == nil, lab.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw Refusal("empty_text", String(localized: "The Prompt Lab’s text is empty: there is nothing to rewrite."),
                          help: "silicontrol reuse <i7>, or lab describe <n>")
        }
        if let picture, LabModel.bytes(of: picture) == nil {
            throw Refusal("ref", String(localized: "This image could not be read."))
        }
        guard let family = lab.target else { throw Self.refusal(.notInstalled) }
        let gesture = picture == nil ? "rewrite" : "describe"

        lab.unfold()
        let started = picture.map { lab.describe($0) } ?? lab.rewrite()
        // The bar's own guard, a moment behind the queue (`LabModel.rendering` follows it by observation).
        guard started else { throw Self.refusal(lab.pause(rendering: true) ?? .rendering) }
        let mine = lab.generation
        var head: [String: Any] = ["gesture": gesture, "for": family.rawValue]
        if let picture, let n = lab.rack.inputs.firstIndex(where: { $0.id == picture.id }) { head["image"] = n + 1 }

        guard wait else {
            finish(c, head.merging(["ok": true, "writing": true, "lab": labJSON()]) { a, _ in a })
            return
        }
        emit(c, head.merging(["type": "started"]) { a, _ in a })
        Task { @MainActor in
            var loading: Double?? = .none, text: String?
            // The bar's state, read as the window draws it: the stream is the gesture's, not the command's.
            while lab.generation == mine, lab.running != nil {
                if c.isGone { return }
                switch lab.phase {
                case .loading(let f) where loading != .some(f):
                    loading = .some(f)
                    emit(c, ["type": "loading", "fraction": f.map { Response.number($0, 3) } ?? NSNull()])
                case .writing:
                    if let now = lab.changes?.after, now != text {
                        text = now
                        emit(c, ["type": "text", "text": now])
                    }
                default: break
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
            if c.isGone { return }
            finish(c, writtenInText(head, mine))
        }
    }

    /// How the bar's gesture ended: written (the field's text, its marks over it), failed, or stopped — by a render, or in the bar.
    private func writtenInText(_ head: [String: Any], _ mine: Int) -> [String: Any] {
        let lab = LabModel.shared
        guard lab.generation == mine else {
            if lab.stopped {
                return Response.failure("stopped", String(localized: "Stopped: an image started rendering."), help: "silicontrol wait, then retry")
            }
            return Response.failure("stopped", String(localized: "Stopped in the Prompt Lab."))
        }
        if let failure = lab.failure { return Response.failure(lab.failureCode ?? "writer_failed", failure) }
        guard lab.changes != nil else {
            return Response.failure("writer_failed", String(localized: "The writer wrote nothing."))
        }
        var o = head.merging(["ok": true, "text": lab.text, "lab": labJSON()]) { a, _ in a }
        if let notice = lab.notice { o["notice"] = notice }
        return o
    }

    /// What the bar holds, as the answers say it.
    func labJSON() -> [String: Any] {
        let lab = LabModel.shared, rack = lab.rack
        let images = rack.inputs.enumerated().map { i, p -> [String: Any] in
            ["n": i + 1, "name": p.shown.name, "width": p.shown.width, "height": p.shown.height,
             "role": GestureOrder.LabAction.name(p.role), "used": rack.isEffective(p)]
        }
        // The bar's gesture under way, as its plate shows it: loading the writer, or writing.
        var writing: Any = NSNull()
        if let gesture = lab.running {
            var w: [String: Any] = ["gesture": gesture == .rewrite ? "rewrite" : "describe",
                                    "phase": lab.phase == .writing ? "writing" : "loading"]
            if let id = lab.describing, let n = rack.inputs.firstIndex(where: { $0.id == id }) { w["image"] = n + 1 }
            writing = w
        }
        return ["unfolded": lab.unfolded, "detached": lab.detached, "model": rack.identifier, "text": rack.prompt, "images": images,
                "unused": rack.unusedInputs, "writing": writing]
    }
}
