import Foundation

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
