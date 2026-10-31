import Foundation
import Synchronization

/// **The weight stream beside the computation: the disk's seconds of a render, planned with the blocks, not added to them.**
/// Pure. The bytes come from what the engine reads (`Streaming`), the rate from the map (`CapacityModel.diskBandwidth`), the computation
/// from the solver's blocks; `Combine` puts them together.
///
///     per pass (the half grid's evaluations, the full grid's)       C = its blocks' seconds (preparation included: the engine copies the
///                                                                    staged weight before the product, with the GPU idle — not overlapped)
///                                                                    R = evaluations · bytes / rate
///     the read of the visits to come runs under the product being computed, so the pass takes
///         max(C, R) + evaluations · exposed                          (`CapacityDisk.pass`)
///     overlapped = Σ blocks' seconds + Σ passes (max(0, R − C) + evaluations · exposed)      serial = Σ blocks' seconds + Σ R
///
/// What sets it apart — what a stream model written by analogy with "add the disk to the compute" would get wrong:
///
///   - **`serial` is the model the engine does not follow**: the stream reads ahead (`TailStream`, three buffers), so on a Mac whose
///     disk reads several times faster than the computation asks, the render runs at the computation's time.
///   - **the passes are timed apart**: the half grid's evaluations compute a quarter of the full ones and read the same bytes; they are
///     the ones the disk can bind, on a Mac that computes fast.
///   - **a resident weight reads nothing**: `bytesPerEvaluation` is the tail the plan streams, 0 where the map fits B.
extension CapacitySolver {

    /// **What the engine reads from the disk at each evaluation of the denoiser.**
    package struct Streaming: Sendable, Equatable {
        /// The bytes of the visits the stream reads, per evaluation (`MemoryPlan.streamedBytes`).
        package var bytesPerEvaluation: Double
        /// The bytes of the first visit(s): read at the start of an evaluation, with nothing to compute beneath them.
        package var firstBytes: Double
        package init(bytesPerEvaluation: Double, firstBytes: Double = 0) {
            self.bytesPerEvaluation = bytesPerEvaluation; self.firstBytes = firstBytes
        }
    }

    /// **The overlapped total**: the blocks' seconds, plus for each stream block the reads that the computation beneath it does not hide, plus
    /// what is exposed.
    package static let overlapped: Combine = { blocks in
        var total = 0.0
        for block in blocks {
            total += block.seconds
            if block.reading > 0 { total += max(0, block.reading - block.beneath) + block.exposed }
        }
        return total
    }

    /// **The total if nothing were read ahead**: the reads add to the computation. The refutation's witness, not a plan.
    package static let serial: Combine = { blocks in
        var total = 0.0
        for block in blocks { total += block.seconds + block.reading }
        return total
    }

    /// **The render's plan with its weight stream**: `solve`'s blocks, and for each pass of the denoising a block that stands for the
    /// disk, its computation beneath it the pass's own. `nil` when `solve` has no plan, or the map has no disk rate (`diskBandwidth`):
    /// then the stream is not timed, and says so by returning the plan without it.
    package static func solve(model: CapacityModel, request: CapacityClasses.Request, constraints: Constraints, streaming: Streaming,
                              combine: Combine = overlapped) -> Plan? {
        guard let base = solve(model: model, request: request, constraints: constraints, combine: sequential) else { return nil }
        guard let rate = model.diskBandwidth, rate > 0, streaming.bytesPerEvaluation > 0 else { return base }
        var blocks = base.blocks
        for (index, pass) in CapacityClasses.passes(request).enumerated() {
            guard let classes = CapacityClasses.classes(for: request, pass: index),
                  let part = solve(model: model, classes: classes, family: request.family, constraints: constraints, combine: sequential)
            else { continue }
            let n = Double(pass.evaluations)
            blocks.append(.init(classID: "stream:pass\(index)", phase: .denoising, executions: pass.evaluations, route: "disk",
                                rows: [:], product: nil, attention: nil, preparation: 0, compute: 0, busy: [:],
                                reading: n * streaming.bytesPerEvaluation / rate, beneath: part.blocks.reduce(0) { $0 + $1.seconds },
                                exposed: n * streaming.firstBytes / rate))
        }
        return Plan(blocks: blocks, seconds: combine(blocks), unpredicted: base.unpredicted)
    }
}

extension CapacitySolver.Streaming {
    /// **The stream of a card's render under its memory plan**, from the installed maps' headers — pure on them: the bytes the engine reads
    /// at each evaluation (`MemoryPlan.streamedBytes`), the same decision `ditTail` takes at the first block. `nil` for a family whose maps
    /// do not give their layers' bytes yet (Z-Image, Anima, MOSS-Image) or whose maps cannot be read. `firstBytes` is 0: the stream starts
    /// reading the next evaluation's first visits when the last one of this evaluation is complete, beneath the final layer and the
    /// evaluation's preamble (measured on Krea 2: the first ask of an evaluation waits 0 to 6 ms).
    package init?(card: ModelCard, plan: MemoryPlan.Render, library: Library) {
        switch card.family {
        case .krea2:
            guard let maps = MemoryNeed.Krea2Maps(card: card, library: library) else { return nil }
            self.init(bytesPerEvaluation: Double(maps.ditStreamedBytes(plan: plan, fraction: Krea2DiT.generationTail)))
        case .qwenImage21:
            guard let maps = QwenImage21Maps(card: card, library: library) else { return nil }
            self.init(bytesPerEvaluation: Double(maps.ditStreamedBytes(plan: plan, fraction: QwenImage21DiT.generationTail)))
        default: return nil
        }
    }
}

// MARK: - The priority of the readers

/// **The class of service the weight stream reads at, phase by phase — decided by the map, not by a default fixed on one Mac.**
/// A phase (the half grid's evaluations, the full grid's) reads the same bytes beside a computation of its own length:
///
///     need(phase) = streamed bytes / C(phase)            C = the solver's seconds of one evaluation at that phase's rows
///     class       = the lowest StreamClass whose measured mean ≥ need · (1 + ε)         (`CapacityDisk.choose`), else the default
///
/// The lower classes leave the performance cores to the computation (the full grid's evaluations ran ~4 % faster at `background` on the
/// Compact Krea 2) but read slower; a half-size evaluation computes a quarter of the time and reads as much, so it is the first to need the
/// default. The answer is memoised by what it depends on; a plan without a model, a map without readings or a family without classes
/// answers the default — the stream as it was.
extension CapacityDisk {
    private struct Phase: Hashable { var identity: String; var family: Family; var format: CapacityClasses.WeightFormat; var tokens: Int; var text: Int; var bytes: Int }
    private static let memo = Mutex<[Phase: StreamClass]>([:])

    /// **The seconds of one evaluation** at `tokens` image rows under `plan`, `nil` when the plan has no model or the model cannot time it.
    package static func evaluationSeconds(plan: CapacityPlan, family: Family, format: CapacityClasses.WeightFormat, tokens: Int,
                                          textRows: Int) -> Double? {
        guard let model = plan.model,
              let classes = CapacityClasses.evaluationClasses(family: family, format: format, tokens: tokens,
                                                              promptTokens: CapacityClasses.promptTokens(family, textRows: textRows)),
              let solved = CapacitySolver.solve(model: model, classes: classes, family: family, constraints: plan.constraints)
        else { return nil }
        return solved.seconds
    }

    /// **The class a phase's reads run at**: `bytes` read per evaluation, beside `tokens` image rows of computation.
    package static func streamClass(plan: CapacityPlan, family: Family, format: CapacityClasses.WeightFormat, tokens: Int, textRows: Int,
                                    bytes: Int) -> StreamClass {
        guard let model = plan.model, !model.diskClasses.isEmpty, bytes > 0 else { return .standard }
        let key = Phase(identity: plan.identity, family: family, format: format, tokens: tokens, text: textRows, bytes: bytes)
        if let known = memo.withLock({ $0[key] }) { return known }
        let chosen = evaluationSeconds(plan: plan, family: family, format: format, tokens: tokens, textRows: textRows).map {
            choose(bytes: Double(bytes), compute: $0, readings: model.diskReadings)
        } ?? .standard
        memo.withLock { $0[key] = chosen }
        return chosen
    }
}

extension Artifact {
    /// **The readers' priority for the phase a DiT evaluation is in**, once its stream exists: `tokens` image rows beside `textRows` text rows
    /// (the rows the DiT really computes). The one call every family that streams makes after `streamUnits`; no stream, no effect.
    package func setStreamPhase(family: Family, tokens: Int, textRows: Int) {
        guard let stream else { return }
        stream.setPriority(CapacityDisk.streamClass(plan: EngineSettings.effective.plan, family: family,
                                                    format: Artifact.weightFormat(header: header), tokens: tokens, textRows: textRows,
                                                    bytes: stream.sweepBytes))
    }
}
