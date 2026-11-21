// **A refusal that crossed, said here** (`RemoteRefusal`): the engine's error through `Problem(EngineError)`,
// the app's own sentences by code, in this Mac's language; the sender's words only when they had no code.

import Foundation
import Siliconed

extension AppRefusal {
    /// The sentence, in the language of the Mac that says it.
    var sentence: String {
        switch self {
        case .imageGone(let o):
            String(localized: "Image i\(String(o)) is no longer in the history.")
        case .tooManyReferences(let count, let model, let max):
            String(localized: "\(count) references: \(ModelCard.withID(model)?.name ?? model) reads \(max) at most")
        case .editSurface(let w, let h):
            AppState.editSurfaceRefusal(width: w, height: h)
        case .loraNotInstalled(let name):
            String(localized: "This LoRA isn’t installed on this Mac: \(name)")
        case .seedCount:
            String(localized: "A render takes 1 to 8 images.")
        case .queueFull:
            String(localized: "The queue is full.")
        case .enlargeModelDownload:
            String(localized: "The enlargement model could not be downloaded")
        case .enlargeFailed:
            String(localized: "This image could not be enlarged")
        case .cropNeeds(let side):
            String(localized: "Cropping needs \(String(side)) px or more on each side")
        case .cropSmallest(let side):
            String(localized: "This image is already at the smallest size, \(String(side)) px")
        case .cropOutside(let w, let h):
            String(localized: "the frame leaves the image (\(Figures.size(w, h)))")
        case .cropWhole:
            String(localized: "the whole image is not a crop")
        case .imagesNotRestorable:
            String(localized: "These images can’t be restored.")
        case .subjectsIncomplete:
            String(localized: "A render that cites named subjects needs their prompt and images.")
        case .libraryBusy:
            String(localized: "That Mac is busy: its library can change once its render, installation or measure is done.")
        case .libraryUnreachable:
            String(localized: "The disk of that Mac’s library is not connected.")
        case .familyUnknown(let id):
            String(localized: "That Mac doesn’t offer \(id).")
        case .solverNotOffered(let model, let solver):
            String(localized: "\(ModelCard.withID(model)?.name ?? model) doesn’t offer the \(solver) solver")
        case .detailNotOffered(let model):
            String(localized: "\(ModelCard.withID(model)?.name ?? model) doesn’t offer Detail")
        case .removalDeclined:
            String(localized: "That Mac declined the removal: nothing was removed.")
        case .removalUnanswered:
            String(localized: "Nobody answered on that Mac: nothing was removed.")
        case .removalPending:
            String(localized: "That Mac is already asking about another removal: try again once it is answered.")
        case .licenseMismatch(let model):
            String(localized: "The license shown is not this Mac’s text for \(ModelCard.withID(model)?.name ?? model): nothing was accepted.")
        }
    }
}

extension AppState {
    /// An edit beyond `editingSurface`, refused by the side of that square.
    nonisolated static func editSurfaceRefusal(width: Int, height: Int) -> String {
        let side = String(Int(Double(editingSurface).squareRoot()))
        return String(localized: "the app edits up to \(side)² of area, not \(Figures.size(width, height))")
    }
}

extension Problem {
    init(_ r: RemoteRefusal) {
        switch r {
        case .engine(let e): self.init(e)
        case .app(let a): self.init(a.sentence)
        case .text(let t): self.init(t)
        }
    }
}

extension RemoteRefusal {
    /// The failure of a job: its error when it had one.
    init(_ p: Problem) { self = p.engine.map(RemoteRefusal.engine) ?? .text(p.text) }
}
