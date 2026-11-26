import CoreAI
import Foundation

/// **A piece of what a preparation does ahead of a run**, one at a time on its client's queue (a run comes before the
/// pieces left, after the one in hand): a function specialized (`Functions.specialize`), or a stage of bands' reading
/// taken (`BandStage.read`). With nothing of its kin read on a Mac, a stage of bands opens at its soberest candidate
/// (`Stage.choose`) — on the mini, a stage of four candidates at 1024² 2.6 to 6.0 s a run against 0.5 to 0.9 at its
/// tallest (22 and 24/11/2026); on the M1 Pro, +3.7 to +4.8 s (25/11) —; once read, its next opening bounds the others
/// from that reading (`Stage.estimates`) and takes the first that holds.
public enum Preparation: Sendable {
    case specialization(Functions.Specialization)
    case reading(BandStage.Candidates)

    /// The unit a specialization compiles for; a reading's, nil.
    public var unit: ComputeUnitKind? { if case .specialization(let s) = self { s.unit } else { nil } }

    /// What it prepares: a function's asset, a stage's name.
    public var name: String {
        switch self {
        case .specialization(let s): s.asset.lastPathComponent
        case .reading(let c): c.name
        }
    }

    /// **Done**: the function specialized, or the stage read.
    public func run() async throws {
        switch self {
        case .specialization(let s): try await Functions.specialize(s)
        case .reading(let c): try await BandStage.read(c)
        }
    }
}
