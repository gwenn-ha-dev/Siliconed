// `AppState` as a mirror: a removal undone, and what the other Mac still holds after a cut.

import AppKit
import Foundation
import ImageIO
import Siliconed

extension AppState {
    /// A removal in a mirror, asked of the other Mac, which keeps the images in its session trash; « Undo »
    /// asks them back and « Redo » removes them again. After the switch back to this Mac the undo is vain.
    func removeRemotely(_ ordinals: [Int]) {
        guard let mirror, !ordinals.isEmpty else { return }
        mirror.send(.deleteImages(ordinals))
        undoManager?.registerUndo(withTarget: self) { $0.restoreRemotely(ordinals) }
        undoManager?.setActionName(ordinals.count == 1 ? String(localized: "Remove Image")
                                                       : String(localized: "Remove \(ordinals.count) Images"))
    }

    private func restoreRemotely(_ ordinals: [Int]) {
        guard let mirror else { return }
        mirror.send(.restoreImages(ordinals))
        undoManager?.registerUndo(withTarget: self) { $0.removeRemotely(ordinals) }
        undoManager?.setActionName(ordinals.count == 1 ? String(localized: "Remove Image")
                                                       : String(localized: "Remove \(ordinals.count) Images"))
    }

    /// An image the other Mac made from a crop asked here: undone by removing it, like any removal.
    func registerRemoteUndo(of ordinal: Int, name: String) {
        undoManager?.registerUndo(withTarget: self) { $0.removeRemotely([ordinal]) }
        undoManager?.setActionName(name)
    }

    /// The link was cut: what is in flight is noted, for `settle`.
    func noteCut() {
        guard let link = mirror else { return }
        let held = Set(link.endings.map(\.ordinal))
        let jobs = Set(link.jobs.keys.filter { completed[$0] == nil && !held.contains($0) })
        let requests = Set(link.waiters.keys).union(link.requested.keys)
        // Lost again before the join's `synced` came: what the first cut noted is still in flight, and what
        // was asked in between too.
        if let before = link.cut { link.cut = (before.jobs.union(jobs), before.requests.union(requests)) }
        else { link.cut = (jobs, requests) }
    }

    /// **The other Mac's word on what it still holds of ours**, at each join: the history and the jobs are
    /// exactly those. An image or an end it does not list is dropped; a render or a job from before the
    /// cut that it does not know is over — said, so that no `wait` hangs.
    func settle(_ s: RemoteSync) {
        guard let link = mirror else { return }
        let keep = Set(s.entries)
        let stale = Set(history.map(\.ordinal)).union(link.entries.keys).subtracting(keep).sorted()
        if !stale.isEmpty {
            receive(.removed(stale))
            for o in stale { link.entries[o] = nil }
        }
        // Ends waiting for an image that is gone would wait for ever.
        link.endings = link.endings.map { var e = $0; e.images = e.images.filter(keep.contains); return e }
        let known = Set(s.jobs)
        if let cut = link.cut {
            let lost = Problem(String(localized: "“\(link.macName)” no longer knows this render."))
            let held = Set(link.endings.map(\.ordinal))
            for o in cut.jobs.sorted() where !known.contains(o) && completed[o] == nil && !held.contains(o) {
                guard let job = link.jobs[o] else { continue }
                problem = lost
                complete(job, .failed(lost))
            }
            for r in cut.requests.sorted() {
                link.requested[r] = nil
                link.waiters.removeValue(forKey: r)?(.refused(.text(lost.text)))
            }
        }
        link.cut = nil
        // The lender restarted: it numbers from 1 again, so the jobs and ends kept under the old numbers
        // would be taken for the new ones (a job shown with another's prompt, an end swallowed).
        if link.restarted {
            link.restarted = false
            link.jobs = [:]; link.grids = [:]; link.endings = []
            series = [:]; gridCells = [:]; shownSeries = nil
            forgetCompletedJobs()
        }
        endHeld()
    }
}
