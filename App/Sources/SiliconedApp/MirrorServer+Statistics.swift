// `MirrorServer`: the Statistics window's figures and the licenses, per client.

import AppKit
import Foundation
import ImageIO
import Siliconed

extension MirrorServer {
    /// **The Statistics window's figures for `identity`**: its last measured render (this Mac's window
    /// shows the latest of anyone's, so each client's is kept when another's replaces it), the speeds of
    /// the models and formats it used, the estimates of its waiting jobs, its session.
    private func statisticsSnapshot(for identity: Data, _ images: [Entry], _ tick: Tick) -> RemoteStatistics {
        var st = RemoteStatistics()
        if app.measurements.job?.owner == identity { measured[identity] = app.measurements }
        if let m = measured[identity], let job = m.job {
            st.measures = RemoteMeasures(
                job: MirrorProjection.remote(job), elapsed: m.begin.map { Date().timeIntervalSince($0) } ?? 0,
                evaluations: m.plan?.denoising.evaluations,
                tokens: m.tokens, tokenizer: m.tokenizer, encoder: m.encoder, encoding: m.encoding, textCached: m.textCached,
                stepSeconds: m.stepSeconds, decodings: m.decodings, timings: m.timings, footprints: m.footprints,
                memory: m.memory.isEmpty ? nil : m.memory.thinned(to: RemoteMeasures.memoryBuckets))
        }
        let mine = (app.currentJob.map { [$0] } ?? []) + app.file + app.awaitingLicense
        var keys = Set(mine.filter { $0.owner == identity }.map(\.speedKey))
        for e in images {
            keys.insert(Speed.key(e.settings.identifier, e.settings.format, references: e.references.count,
                                  lora: !e.settings.loras.isEmpty))
        }
        for (k, v) in app.speeds where keys.contains(k) {
            st.speeds[k] = RemoteSpeed(full: v.full, text: v.text, image: v.image, decoding: v.decoding)
        }
        for j in app.file + app.awaitingLicense where j.owner == identity {
            if let e = tick.estimates[j.ordinal] { st.estimates[j.ordinal] = e }
        }
        let session = sessions[identity] ?? (0, 0)
        st.images = session.images; st.seconds = session.seconds
        return st
    }

    func publishStatistics(to identity: Data, _ images: [Entry], _ tick: Tick, in s: inout Seen) {
        let st = statisticsSnapshot(for: identity, images, tick)
        // The running render's age moves on its own: it alone is not a change.
        guard st.withoutClock != s.statistics?.withoutClock else { return }
        s.statistics = st
        host.send(.statistics(st), to: identity)
    }

    /// The license the client's first waiting job needs, asked on its screen; `nil` once none waits —
    /// the answer of either Mac closes the other's sheet.
    func publishLicense(to identity: Data, in s: inout Seen) {
        let want = app.awaitingLicense.first { $0.owner == identity }
            .flatMap { j in app.cards.first { $0.id == j.settings.identifier } }
            .map { card -> RemoteLicense in
                let v = app.library.variant(of: card.family)
                return RemoteLicense(model: card.id, variant: v.rawValue, digest: LicenseDigest.of(card, variant: v))
            }
        guard want != s.license || !s.licenseSent else { return }
        s.license = want
        s.licenseSent = true
        host.send(.licensing(want), to: identity)
    }

    /// **A client's click on its license sheet**, judged here as a local click would be: only while one of
    /// its jobs still waits for that license (the first answer, from either Mac, won otherwise), and an
    /// acceptance only for the digest of this Mac's own text. Declining takes its waiting jobs off.
    func answerLicense(_ a: RemoteLicenseAnswer, from identity: Data) {
        guard app.awaitingLicense.contains(where: { $0.owner == identity && $0.settings.identifier == a.model }),
              let card = app.cards.first(where: { $0.id == a.model }) else { return }
        guard a.accepted else {
            for j in app.awaitingLicense where j.owner == identity && j.settings.identifier == a.model { app.remove(j) }
            return
        }
        guard a.digest == LicenseDigest.of(card, variant: app.library.variant(of: card.family)) else {
            host.send(.refused(.app(.licenseMismatch(model: card.id))), to: identity)
            // The sheet is asked again.
            seen[identity]?.license = nil
            scheduleQueue()
            return
        }
        app.acceptLicense(card)
    }
}
