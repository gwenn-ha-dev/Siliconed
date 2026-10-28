import Foundation

/// **What the Statistics window reads, for one client** — published by the lending Mac beside the queue
/// (`SessionMessage.statistics`), so that the borrowing Mac's window shows its own renders, measured there.
///
/// **Per client, like the rest**: the figures of a render are those of the last one this client asked for
/// (kept by the lender when another Mac's render replaces them in its own window); the speeds are those of
/// the models and formats this client has used; the session is its images and its compute seconds. What
/// the lender renders for itself or for others never appears here.
///
/// **No `Date`**: a running render's age is `elapsed`, a duration; the borrower dates it on arrival.
package struct RemoteMeasures: Codable, Equatable, Sendable {
    package var job: RemoteJob
    /// Seconds since the render began, when this was sent (ignored when comparing two snapshots: it runs).
    package var elapsed: Double
    /// The plan the engine announced: its evaluations per image, and the half-size steps among them.
    package var evaluations: Int?
    package var reduced: Int?
    package var tokens: Int?
    package var tokenizer: Double?, encoder: Double?
    package var encoding: Double?
    package var textCached: Bool
    /// Per image of the batch, each DiT evaluation's duration.
    package var stepSeconds: [[Double]]
    package var decodings: [Double]
    package var timings: Engine.Timings?
    package var footprints: Engine.Footprints?

    package init(job: RemoteJob, elapsed: Double, evaluations: Int?, reduced: Int?, tokens: Int?, tokenizer: Double?,
                 encoder: Double?, encoding: Double?, textCached: Bool, stepSeconds: [[Double]], decodings: [Double],
                 timings: Engine.Timings?, footprints: Engine.Footprints?) {
        self.job = job; self.elapsed = elapsed; self.evaluations = evaluations; self.reduced = reduced
        self.tokens = tokens; self.tokenizer = tokenizer; self.encoder = encoder; self.encoding = encoding
        self.textCached = textCached; self.stepSeconds = stepSeconds; self.decodings = decodings
        self.timings = timings; self.footprints = footprints
    }
}

/// A learned pace (`Speed`), per model and format.
package struct RemoteSpeed: Codable, Equatable, Sendable {
    package var full: Double
    package var reducedAverage: Double?
    package var text: Double?
    package var image: Double
    package var decoding: Double
    package init(full: Double, reducedAverage: Double?, text: Double?, image: Double, decoding: Double) {
        self.full = full; self.reducedAverage = reducedAverage; self.text = text; self.image = image; self.decoding = decoding
    }
}

package struct RemoteStatistics: Codable, Equatable, Sendable {
    /// The last render of this client's that was measured; `nil`: none yet.
    package var measures: RemoteMeasures?
    /// `Speed.key` → pace, for the models and formats this client used.
    package var speeds: [String: RemoteSpeed] = [:]
    /// What the lender estimates each waiting job of this client takes, by its ordinal (absent: never measured).
    package var estimates: [Int: Double] = [:]
    /// This client's session: its images and the seconds they took to compute.
    package var images = 0
    package var seconds = 0.0
    package init() {}

    /// The same, but for the running render's age: two snapshots differing only by it are one.
    package var withoutClock: RemoteStatistics {
        var s = self
        s.measures?.elapsed = 0
        return s
    }
}
