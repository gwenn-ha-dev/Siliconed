// `MirrorServer`: the app's state observed, and a change published to every client.

import AppKit
import Foundation
import ImageIO
import Siliconed

extension MirrorServer {
    func watchQueue() {
        withObservationTracking {
            // Everything a client's view reads, whoever it belongs to.
            _ = (app.currentJob?.ordinal, app.file.map(\.ordinal), app.awaitingLicense.map(\.ordinal), app.enlarging)
            _ = (app.fraction, app.stage, app.currentStep, app.totalSteps, app.batchImage, app.batchSize)
            _ = (app.stageStates, app.stageSeconds, app.estimatedRemaining)
            _ = app.history.map(\.ordinal)
            _ = app.preview
            _ = app.series.mapValues(\.issues.count)
            _ = (app.measurements, app.speeds, app.sessionSeconds)
        } onChange: { [weak self] in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.scheduleQueue() } }
        }
    }

    func scheduleQueue() {
        guard !queueScheduled else { return }
        queueScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) { [weak self] in
            MainActor.assumeIsolated { self?.fireQueue() }
        }
    }

    private func fireQueue() {
        queueScheduled = false
        publishQueue()
        watchQueue()
    }

    private func publishQueue() {
        if !seen.isEmpty {
            let tick = Tick(app)
            for identity in seen.keys { publish(to: identity, tick) }
        }
        guard app.preview !== lastPreview else { return }
        lastPreview = app.preview
        // The preview is the running render's: its owner's alone.
        guard let owner = app.currentJob?.owner, seen[owner] != nil else { return }
        sendPreview(to: owner)
    }

    /// The preview of the render under way, if it is `identity`'s; else none — said, so that a preview
    /// the client kept across a cut does not stay on its canvas.
    func sendPreview(to identity: Data) {
        if app.currentJob?.owner == identity, let p = app.preview, let png = Self.png(p) {
            let id = blob()
            host.send(.preview(id), to: identity)
            host.send(blob: png, id: id, to: identity)
        } else {
            host.send(.preview(nil), to: identity)
        }
    }

    /// What `identity` was not sent yet: its queue, its images (oldest first: the mirror inserts each at
    /// its place), its series before them.
    func publish(to identity: Data, _ tick: Tick) {
        guard var s = seen[identity] else { return }
        let q = queueSnapshot(for: identity, tick)
        if q != s.queue { s.queue = q; host.send(.queue(q), to: identity) }
        // Its series before their images: the cells land under their tile.
        for run in app.series.values.sorted(by: { $0.ordinal < $1.ordinal }) where run.owner == identity {
            let x = remote(run)
            if s.series[run.ordinal] != x { s.series[run.ordinal] = x; host.send(.series(x), to: identity) }
        }
        let mine = tick.history[identity] ?? []
        let ordinals = Set(mine.map(\.ordinal))
        let gone = s.entries.subtracting(ordinals)
        if !gone.isEmpty { host.send(.removed(gone.sorted()), to: identity) }
        for e in mine.reversed() where !s.entries.contains(e.ordinal) { send(e, to: identity) }
        s.entries = ordinals
        for e in s.ended { host.send(.ended(e), to: identity) }
        s.ended = []
        publishStatistics(to: identity, mine, tick, in: &s)
        publishLicense(to: identity, in: &s)
        seen[identity] = s
    }

    func watchCatalog() {
        withObservationTracking {
            _ = app.readySet; _ = app.cards; _ = app.libraryRead
        } onChange: { [weak self] in
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.fireCatalog() } }
        }
    }

    private func fireCatalog() {
        let c = catalogSnapshot()
        if c != lastCatalog {
            lastCatalog = c
            if !seen.isEmpty { host.send(.catalog(c)) }
        }
        watchCatalog()
    }
}
