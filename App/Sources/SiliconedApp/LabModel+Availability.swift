// `LabModel`: what the Lab writes for — the rack shown, its model, its images — and what waits, gesture by gesture.

import AppKit
import Siliconed
import SwiftUI

extension LabModel {
    // MARK: Where it writes for

    /// The rack the lab fills: the one the window shows — this Mac's, or the other Mac's mirror.
    var rack: AppState { MainWindow.shared.app }
    /// Whether the window follows another Mac.
    var following: Bool { MainWindow.shared.carrier.mirror != nil }
    var subjects: SubjectLibrary { MainWindow.shared.local.subjects }

    /// The families the lab prepares for: those installed here, in the catalog's order — or, in a mirror, the model
    /// the other Mac's rack has chosen.
    var families: [Family] {
        if following { return rack.card.map { [$0.family] } ?? [] }
        let ready = Set(rack.cards.filter { rack.readySet.contains($0.id) }.map(\.family))
        return Family.allCases.filter(ready.contains)
    }

    /// The model the Lab writes for: the rack's — one setting, chosen in the rack.
    var target: Family? { WriterTarget.pick(chosen: nil, rack: rack.card?.family, installed: families) }

    /// The installed card the rack takes to render for `family`: its own when it is of that family.
    func card(for family: Family) -> ModelCard? {
        if let current = rack.card, current.family == family { return current }
        return rack.cards.first { $0.family == family && rack.readySet.contains($0.id) }
    }

    /// **How many images the rack's model reads** (0: none) — from its denoiser here, from the traits the other Mac
    /// published in a mirror.
    var capacity: Int { rack.imageCapacity }

    /// The same, for the model a gesture writes for (`silicontrol rewrite --for`).
    func capacity(for family: Family?) -> Int {
        guard let family else { return 0 }
        if following { return rack.promptNamesImages ? rack.maxReferences : 0 }
        guard let card = card(for: family), let model = rack.model(card.id) else { return 0 }
        return model.chain.text.output.readsImages ? model.chain.denoising.maxReferences : 0
    }

    /// **Whether the rack's model starts from an image** (img2img). Never in a mirror — the other Mac's traits do not say it,
    /// and its jobs carry no starting image.
    var startsFromImage: Bool { rack.startsFromImage }

    /// The model's start is an img2img's.
    var startIsImg2Img: Bool { capacity == 0 && startsFromImage }

    var offeredRoles: [LabInputs.Role] { LabInputs.roles(capacity: capacity, startsFromImage: startsFromImage) }

    // MARK: What waits, gesture by gesture

    /// **Why Rewrite and Describe wait**, in one sentence; `nil` when they can run. They are the only gestures that need the
    /// language model: everything else in the lab works without it, during a render too.
    var writerPause: (symbol: String, sentence: String)? { pause(rendering: rendering).map(Self.presentation) }

    /// The decision (`WriterPause.decide`), for the window and for `silicontrol rewrite` and `describe`: the render is passed in
    /// because the command reads the queue itself, not the copy this model keeps up to date for the window.
    func pause(rendering: Bool) -> WriterPause? {
        return WriterPause.decide(following: following, rendering: rendering, installed: installed, hasTarget: target != nil)
    }

    static func presentation(_ pause: WriterPause) -> (symbol: String, sentence: String) {
        switch pause {
        case .following: ("macbook", String(localized: "Rewrite and Describe are off while the window follows another Mac."))
        case .rendering: ("pause.circle", String(localized: "Rewrite and Describe wait while an image renders."))
        case .notInstalled: ("shippingbox", String(localized: "No writer is installed yet. Install a model to have one."))
        }
    }

    var canRewrite: Bool { rewriteRefusal == nil }

    var canDescribe: Bool { describeRefusal == nil }

    /// **Why Rewrite is greyed**, its hover; `nil` when it can write.
    var rewriteRefusal: String? {
        if let p = writerPause { return p.sentence }
        if isWriting { return String(localized: "The writer is writing: wait for it, or stop it") }
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return String(localized: "Write a prompt first: Rewrite starts from it") }
        return nil
    }

    /// **Why Describe is greyed**, its hover; `nil` when it can write.
    var describeRefusal: String? {
        if let p = writerPause { return p.sentence }
        return isWriting ? String(localized: "The writer is writing: wait for it, or stop it") : nil
    }
}
