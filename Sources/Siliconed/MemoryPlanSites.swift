import Foundation
import Synchronization

// **The render's plan, as the sites of the engine ask it**: the render in progress (`MemoryPlan.during`, installed by `Engine`
// around a render) and what the DiT, the text encoder and the decoders read from it. The choice itself — `MemoryPlan.choose` — is pure and
// lives in `MemoryPlan.swift`; B, in `MemoryBudget.swift`.

extension MemoryPlan {
    // ── the render in progress ─────────────────────────────────────────────────────────────

    private static let state = Mutex<Render?>(nil)
    private static let last = Mutex<Render?>(nil)

    /// The render in progress, or `nil` outside one.
    package static var current: Render? { state.withLock { $0 } }
    /// The last render's plan, once it is over (or while it runs).
    package static var lastRender: Render? { last.withLock { $0 } ?? current }

    /// Wires a render's plan in for the duration of `body` — the engine, around a render (renders are
    /// serialized: the engine queue and `RenderLock.machine`).
    package static func during<T>(_ render: Render, _ body: () throws -> T) rethrows -> T {
        state.withLock { $0 = render }
        defer {
            let ended = state.withLock { s -> Render? in let r = s; s = nil; return r }
            last.withLock { $0 = ended }
        }
        return try body()
    }

    package static func note(_ decision: String) {
        state.withLock { $0?.decisions.append(decision) }
    }

    /// `note`, unless it is already the last decision — a site asked at every evaluation.
    package static func noteOnce(_ decision: String) {
        state.withLock { s in if s?.decisions.last != decision { s?.decisions.append(decision) } }
    }

    // ── what the sites ask ─────────────────────────────────────────────────────────────────

    /// **The DiT's tail**, where it opens its map (`DiT.forward`, `QwenImage21DiT.pass`): the first
    /// layer streamed and the staging buffers (`released`). The same at every evaluation of a render:
    /// nothing it depends on moves. Outside a render, the model's split, streamed; under
    /// `SILICONED_MAP_TAIL` (a forcing of measurement), that fraction, streamed, journaled.
    ///
    /// `fraction`: the model's own split, when it has measured one — the setting's (0.5) otherwise. A
    /// `map_tail` set in the profile or the environment still wins.
    package static func ditTail(artifact: Artifact, layerBlocks: [[String]],
                                fraction preferred: Double? = nil) -> (released: Int, streamed: Bool) {
        let layers = layerBlocks.count
        let spans = layerBlocks.map { artifact.span(of: $0) }
        let block = spans.max() ?? 0
        func decide(_ plan: (released: Int, streamed: Bool), _ why: String) -> (released: Int, streamed: Bool) {
            let buffers = plan.released < layers && plan.streamed ? TailStream.unitSlots : 0
            noteOnce(String(format: "map tail %d of %d layers, %d buffer(s) of %.0f MB%@", layers - plan.released,
                            layers, buffers, Double(block) / 1e6, why as NSString))
            return plan
        }
        let settings = EngineSettings.effective
        if settings.provenance[.mapTail] == .environment {
            return decide((Tail(fraction: settings.mapTail, streamed: true).released(layers: layers), true),
                          " (forced: SILICONED_MAP_TAIL)")
        }
        let fraction = settings.provenance[.mapTail] == nil ? preferred ?? settings.mapTail : settings.mapTail
        guard let render = current else {
            return decide(fraction > 0 ? (Tail(fraction: fraction, streamed: true).released(layers: layers), true) : (layers, false), "")
        }
        // Read in place, the map is charged whole (`MemoryPlan.choose`): never streamed.
        if render.request.mapInPlace { return decide((layers, false), " (the whole map read in place, charged in B)") }
        let mapBytes = artifact.tensors.values.reduce(0) { $0 + $1.bytes }
        let plan = released(spans: spans, mapBytes: mapBytes, ceiling: render.ceiling,
                            streamed: render.denoisingStreamed, cached: render.denoisingCached, fraction: fraction)
        let split = Prefetcher.firstReleasedLayer(of: layers, tail: fraction)
        let why = !plan.streamed && fraction > 0 ? " (the whole map in B)"
            : plan.released < split ? " (head: what B leaves beside the denoising)" : ""
        return decide(plan, why)
    }

    /// **The text encoder's buffers**, as it starts: the plan's. Outside a render, the reference (4). None for an encoder kept
    /// (`KeptEncoder`): its layers are in the process's pages, nothing is streamed.
    package static func encoderSlots(artifact: Artifact, layerBlocks: [[String]]) -> Int {
        guard let render = current else { return encoderSlots }
        if render.request.encoderKept { return 0 }
        let block = layerBlocks.map { artifact.span(of: $0) }.max() ?? 0
        note(String(format: "text encoder %d buffer(s) of %.0f MB", render.request.encoderSlots, Double(block) / 1e6))
        return render.request.encoderSlots
    }

    /// **A decoder's band**, as it starts decoding: the plan's — the measured size (FLUX's 2²³ elements,
    /// Qwen's 2¹⁸ pixels), the smallest proved one on a plan that could not hold it, or `wide` where the plan
    /// holds that (`Request.wideBands`: Flux's 2²⁵, Z-Image's decoder — the same bits). Outside a render, `standard` — the static a
    /// check may have set.
    package static func decoderBand(standard: Int, minimum: Int, wide: Int? = nil, unit: String) -> Int {
        guard let render = current else { return standard }
        let band = render.request.minimumBands ? min(standard, minimum)
            : render.request.wideBands ? max(standard, wide ?? standard) : standard
        note("decoder bands of 2^\(band.trailingZeroBitCount) \(unit)")
        return band
    }
}

extension Artifact {
    /// The bytes a run of tensors spans in the file — a streamed block's staging size.
    package func span(of names: [String]) -> Int {
        let ranges = names.compactMap { tensors[$0] }.map { ($0.offset, $0.offset + $0.bytes) }
        guard let first = ranges.map(\.0).min(), let end = ranges.map(\.1).max() else { return 0 }
        return end - first
    }
}
