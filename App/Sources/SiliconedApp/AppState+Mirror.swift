// **The other Mac's state, shown here** — the borrowing side of the remote Mac (`OtherMacs.swift`). A
// second `AppState`, `init(mirroring:)`, beside this Mac's own, which it never touches: back in local,
// everything is where it was.
//
//     the lender publishes (`Mirror.swift`)      the mirror keeps
//       catalog        ready models, traits, B     readySet, compatibleLoras, missing, memoryCeiling
//       queue          job running, waiting        currentJob, file, fraction, stage, steps, problem
//       entry + PNG    a history image             history (the thumbnail made here, from the PNG)
//       preview        a small PNG per step        preview
//
// - **It computes nothing**: its derived values (`busy`, `canRender`…) are the same
//   code reading what was published — the lender checks every job again before
//   queueing it.
// - **Its actions are sent** (`enqueue`, `cancel`, `remove`, `stopAll`, `clearQueue`, `delete`: each
//   starts with `if let mirror`). A job keeps the form's every choice, its seeds included: the lender
//   renders the same bits as if it had been asked on its own screen.
// - **Only this Mac's own** comes back: its jobs, its images, its series; what the other Mac renders
//   for itself or for other Macs is a count and a duration (`othersQueued`), said in the footer.
// - **Ordinals are the identities** (`t3`, `i7`), as for `silicontrol`: a job or an image keeps its
//   `UUID` here from one snapshot to the next, so that the selection holds.
// - **An edit's images go first**: a file's own bytes, an image in memory as a PNG, an image of the
//   lender's history by its ordinal (`RemoteReference`); the lender holds the render until they are there.
//   What the form cannot compute without the models — an edit's format — it asks
//   (`quote`), and waits for the answer before « Generate ».
// - **« Variations »** of an image: its own variations travel with the job, and an edit's images are
//   named by the image (`ofImage`) — they never came here.
// - **Enlarging and cropping** happen there, on its own pixels (`enlarge`, `crop` by ordinal); the
//   consent to download Apple's model is asked here. An enlargement's curtain lays under it its source
//   when that is in the history (`RemoteEntry.enlargedFrom`); else there is no curtain.
// - **Named subjects stay here** (incognito): a render that cites them sends the model's prompt and the
//   subjects' framed pictures after the edit's images (`RemoteJob.modelPrompt`, `subjectImages`), and
//   nothing else of them.
// - **A series** is asked whole (`grid`) and made there; the canvas shows the one it publishes, its cells'
//   images arriving as any history image, laid in their cells by their job.
// - **The statistics and the license sheet** are the other Mac's, for this Mac's renders (`MirrorStats.swift`).
// - **⌘Z after a removal**: the other Mac keeps the images removed in a trash of its session, per Mac;
//   « Undo » asks them back by ordinal (`restoreImages`) and « Redo » removes them again.
// - **After a cut** the other Mac says at each join what it still holds of ours (`synced`): what is not in
//   it is removed here, and a job it no longer knows ends as failed — nothing waits for ever.
// - **Its refusals are codes** (`RemoteRefusal`), said here in this Mac's language.

import AppKit
import Foundation
import ImageIO
import Siliconed

extension AppState {
    /// The chosen model's traits, as the other Mac published them.
    var traits: RemoteTraits? { mirrorCatalog?.traits[identifier] }

    /// The other Mac's answer, or `nil` while it is asked.
    func quoted(_ q: RemoteQuote) -> RemoteQuoteAnswer? {
        if let a = mirrorQuotes[q] { return a }
        mirror?.ask(q)
        return nil
    }

    /// The edit's quote: the format image 1 gives — only with an image to edit (`framingImage`).
    var editQuote: RemoteQuote? {
        guard mirror != nil, let first = framingImage else { return nil }
        return RemoteQuote(model: identifier, width: format.width, height: format.height,
                           first: [first.width, first.height], images: references.count)
    }

    /// The rack's three module names: from the chain here, from the traits in a mirror.
    var moduleNames: (text: String?, encoding: String?, decoding: String?) {
        if mirror != nil { return (traits?.text, traits?.encoding, traits?.decoding) }
        return (chain?.text.name, chain?.encoding?.name, chain?.decoding.name)
    }

    /// How the weights of the model on the rack are stored: read from its installed maps here, published by the lender in a mirror.
    var weightFormats: WeightFormats? {
        mirror != nil ? traits?.weights : model(identifier)?.formats
    }

    func readMirroredCatalog() {
        guard let c = mirrorCatalog else { compatibleLoras = []; missing = nil; return }
        // The lender names its LoRAs; their ranks stay there.
        compatibleLoras = (c.loras[identifier] ?? []).map { l in
            LoRACard(path: l.path, rawName: c.loraNames[l.path] ?? l.path, target: identifier, rank: 0, smallestRank: 0)
        }
        missing = readySet.contains(identifier) ? nil : String(localized: "Not installed on \(mirror?.macName ?? "")")
    }
}

extension AppState {
    /// The session with the other Mac changed: said where the problems are said; joined again, the
    /// problem goes.
    func linkChanged(_ s: RemoteLink.State, name: String) {
        switch s {
        case .connected(let hello):
            linkBlocker = nil; mirror?.down = nil
            if problem?.code == "remote" { problem = nil }
            // Asked before a cut: asked again.
            mirror?.asked = []
            if let link = mirror, let launch = hello.launch {
                if let known = link.launch, known != launch { link.restarted = true }
                link.launch = launch
            }
        case .connecting, .off: break
        case .waiting:
            let lost = String(localized: "Lost “\(name)”: trying again…")
            linkBlocker = lost; mirror?.down = lost
            remoteConfirming = false; uploadLinkChanged(s); noteCut()
            mirror?.failEnlargements(lost)
            problem = { var p = Problem(lost); p.code = "remote"; return p }()
        case .refused:
            let why = OtherMacsSection.state(s, found: true)
            linkBlocker = why; mirror?.down = why
            remoteConfirming = false; uploadLinkChanged(s)
            mirror?.failEnlargements(why)
            problem = { var p = Problem(why); p.code = "remote"; return p }()
        }
    }
}

// MARK: - Removal undone, and the join's sync

extension AppState {
    /// **The form that crosses a switch of Mac** (`MainWindow.compute(on:)`): what the user has typed and chosen,
    /// carried from the state left to the state shown. The one place that lists it — a setting added to the
    /// form is added here. The LoRAs, the edit's images and the starting image are not: they name files of one Mac's library
    /// or live in one Mac's memory — nor the brackets that name a LoRA.
    func copyForm(from other: AppState) {
        // The model first: changing it resets the steps and a start strength chosen for another family.
        prompt = other.prompt; identifier = other.identifier; steps = other.steps
        orientation = other.orientation; format = other.format; batch = other.batch
        // The seed before the variations: setting it drops them (`seed.didSet`).
        fixedSeed = other.fixedSeed; seed = other.seed; seedShown = other.seedShown; variations = other.variations
        detail = other.detail; solver = other.solver; chosenStartStrength = other.chosenStartStrength
        previews = other.previews
        brackets = other.brackets.filter { b in
            switch b.key {
            case .loraStrength, .comparedLoRAs: false
            default: true
            }
        }
        seriesPlaced = other.seriesPlaced
    }
}
