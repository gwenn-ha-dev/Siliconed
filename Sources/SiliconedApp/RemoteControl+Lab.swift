// **`lab`: the Prompt Lab's bar, from the terminal** — what it holds, folded or unfolded, an image in, a role, an image out.
// Each calls what the bar calls (`LabModel`, `AppState+Inputs.swift`), on the rack the window shows; the rules are `LabInputs`'.
//
// - **An image comes in as a drop brings it**: a file as the file, an image of the history as « Send to the Prompt Lab » does,
//   with the role a newcomer takes (`LabInputs.defaultRole`). All are read before any comes in: one unreadable file refuses the
//   command whole.
// - **A role the rack's model lacks is refused**, as the bar's menu does not offer it; the images keep theirs across a change of
//   model, without effect where the model lacks them (`"used": false`).

import AppKit
import Foundation
import Siliconed

extension RemoteControl {
    func lab(_ action: GestureOrder.LabAction, cwd: String) throws -> [String: Any] {
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
        }
        return ["ok": true, "lab": labJSON()]
    }

    /// What the bar holds, as the answers say it.
    func labJSON() -> [String: Any] {
        let lab = LabModel.shared, rack = lab.rack
        let images = rack.inputs.enumerated().map { i, p -> [String: Any] in
            ["n": i + 1, "name": p.shown.name, "width": p.shown.width, "height": p.shown.height,
             "role": GestureOrder.LabAction.name(p.role), "used": rack.isEffective(p)]
        }
        return ["unfolded": lab.unfolded, "model": rack.identifier, "text": rack.prompt, "images": images,
                "unused": rack.unusedInputs]
    }
}
