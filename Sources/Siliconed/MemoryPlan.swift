import Foundation

// ── the plan, chosen in B ────────────────────────────────────────────────────────────────────

/// **The execution settings of one render, chosen in B before its first computation** —
/// beside `EngineSettings.effective`, which is read once per process. Nothing is read from the machine,
/// and nothing changes once the render runs: the same request on the same Mac takes the same plan.
///
///     request ── the lean plan: whole map streamed (three buffers), ── need ≤ B, or refused (Preflight)
///                no encoder buffer, the decoders' smallest bands
///                  + the text encoder's four buffers                ── kept if the need stays ≤ B
///                  + the decoders' standard bands                   ── kept if the need stays ≤ B
///                  + Flux's wide band                               ── kept if the need stays ≤ B
///                  + the DiT's map read in place, charged whole     ── kept if the need stays ≤ B (no stream then)
///                  + the text encoder kept, charged to every phase  ── kept if the need stays ≤ B and the map loses nothing
///                  └─ the map's head left to the page cache: what B leaves beside the denoising (`released`)
///
/// What only buys time is chosen here. What changes a bit — `mlpBudget`, the prefill's groups,
/// `maxMatrix`, `vaeRequestBlock`, the decoder's band alignment — is not, and stays fixed.
///
/// What a reading by analogy would get wrong:
///
///   - **Each knob lives in its own phase** — the encoder's buffers in `text`, the bands in `decoding`,
///     the map's buffers in `denoising` — and the need is the largest phase: each is kept or dropped
///     alone, and adding one never raises another phase. **Except the encoder kept** (`KeptEncoder`), which lives in all of them
///     and between renders: chosen last, beside everything else, and only where the DiT's map keeps all it had — read in place,
///     or whole in the room beside the denoising (with the layers in it). Where the map is streamed, that room is its head,
///     read at every evaluation: the encoder, read once per new prompt, does not take it.
///   - **No map read through the page cache is in the need** — B bounds the process's footprint
///     (`phys_footprint`), and a mapped map's pages are clean and file-backed: the kernel drops them
///     without writing, never swaps them, and they are nobody's footprint. So the text encoder with no
///     buffer (the lean plan) reads its whole map through the cache, a Qwen-Image-2.1 edit its whole DiT
///     map, uncharged — and the DiT's head is not charged either: it is **sized from B as a speed
///     choice**, not counted in it. It is the soft part of B — what B leaves beside the denoising's
///     anonymous need, so that the head has a chance to stay cached — and **capped at the model's own
///     split** (0.5; Qwen-Image-2.1's generation 1, no head): on a 16 GB Mac a head larger than half was read back more, not less
///     (measured: heads of 0.43–0.73 read 6.6–8.6 GB an evaluation, 5.4 at 0.5) — the page cache
///     holds less than any figure says. Only a map that fits whole in that room leaves the stream (no
///     buffer at all: a 32 GB Mac, an 8-bit map).
///   - **A map read in place is charged, whole** (`Request.mapInPlace`): wrapped in a GPU buffer (`Artifact.wrappedMap`), the map's
///     pages the CPU maps count in the footprint once the GPU has read it — the prefetcher's `WILLNEED` maps them all —, so the
///     reading in place is a choice of the plan with its price, the whole file beside the denoising's anonymous need, taken only
///     where that holds in B; otherwise the weights are copied into the reserve, the same bits. Taken, the map is never streamed:
///     its need is the cached one plus the file, which is the condition `released` keeps the map whole on, and more.
///   - **Never one buffer**: measured, one serializes the reads behind the computation (Z-Image 512²
///     14.4 s a step against 4.1 with two; Qwen 512 20 s against 6), the same bits. The lean plan streams
///     the whole map with the stream's buffers (`TailStream.unitSlots`): a budget too short for a head is a cache too short for it, and the
///     map through the cache is what swapped (234.7 GB through the compressor in a swapping render,
///     13.1 with the whole map streamed).
///   - **The check is against the plan's own need** (`Render.need`): a render on a smaller plan is
///     judged on that plan, not on the fastest one it did not take. A map left wholly to the cache
///     peaks below it (its buffers are charged, not allocated).
///
/// Outside a render (a check, a bench) there is no plan: each site keeps the reference values it had
/// before — the measurement tools stay what they were. `SILICONED_MAP_TAIL` set in the environment is a
/// forcing of measurement: the DiT then streams that fraction with its buffers, whatever B.
package enum MemoryPlan {

    /// The DiT's plan for its map, as the need charges it: the share of its layers streamed around the
    /// page cache, and whether it is streamed at all (`TailStream.unitSlots` buffers of its largest tensor).
    package struct Tail: Sendable, Equatable {
        package var fraction: Double
        package var streamed: Bool
        package static let none = Tail(fraction: 0, streamed: false)
        /// **The lean plan's map: all of it beside the cache, with the stream's buffers**.
        package static let lean = Tail(fraction: 1, streamed: true)
        /// The first of `layers` streamed (`layers`: none).
        package func released(layers: Int) -> Int {
            streamed ? Prefetcher.firstReleasedLayer(of: layers, tail: fraction) : layers
        }
    }

    /// **Whether this process's denoiser of `family` reads its map in place** (`WeightProducts.prepare`): a family whose blocks run on
    /// the GPU stream (`SILICONED_STREAM`, read once per process; `SILICONED_STREAM_WAIT` turns the stream off). A setting, not the
    /// machine's state.
    package static let streamedFamilies: Set<Family> = [.zImage, .qwenImage21, .mossImage, .krea2, .anima]
    package static func readsMapInPlace(_ family: Family) -> Bool {
        streamedFamilies.contains(family) && GPUStream.policy != .off && ProcessInfo.processInfo.environment["SILICONED_STREAM_WAIT"] == nil
    }

    /// **Whether a product may read its weight in place now** (`WeightProducts.prepare`): the render's plan says so — or, outside a
    /// render (a check, a bench: no memory contract), wherever the map is open.
    package static var mapInPlace: Bool { current?.request.mapInPlace ?? true }

    /// The text encoder's reference: four buffers (three layers ahead). Nothing between four and none:
    /// one buffer was measured worse than none on the DiT, two and three never.
    package static let encoderSlots = 4

    /// Flux's decoder (Z-Image): the smallest band proved and timed (7.3 s at 1024² against 4.4).
    package static let fluxMinimumBand = 1 << 20
    /// Flux's decoder, where B holds it: four times the standard band, the same bits — at 1024² 1.09 s against 1.21 in bf16
    /// (+0.2 GB), 2.34 against 2.51 in fp32 (+0.5 GB); wider bought nothing more.
    package static let fluxWideBand = 1 << 25
    /// Qwen-Image-2.1's decoder: a quarter of the measured band (bits proved down to 2 rows; time not measured).
    package static let qwenMinimumBand = 1 << 16

    /// **One render's plan**, chosen by `choose` before its first computation.
    package struct Render: Sendable {
        /// B, the plan's ceiling.
        package var ceiling: Int
        /// The request at the plan chosen: its encoder buffers, its bands (the map's tail: `.lean`'s
        /// buffers, the head decided by `released` where the map is open).
        package var request: MemoryNeed.Request
        /// The chosen plan's need, phase by phase — what the render's peak is checked against.
        package var need: MemoryNeed?
        /// The lean plan's peak: the floor the preflight judged.
        package var floor: Int
        /// The denoising's need with the map streamed (its buffers), and with it wholly cached.
        package var denoisingStreamed: Int
        package var denoisingCached: Int
        /// **The map made resident beside the text** (`MapWarmup`): read in place, and the wired map holds in B beside every phase
        /// before the denoising — its wired pages are not in the footprint, they are in the RAM.
        package var warmsMap = false
        /// What each site decided, in order, for the report (the report prints it).
        package var decisions: [String] = []
        /// The lean plan, all of it: no encoder buffer and the smallest bands.
        package var lean: Bool { request.encoderSlots == 0 && request.minimumBands }
    }

    /// **The plan, from the request and B**: the lean plan, then each buffer that buys time while the
    /// need stays ≤ B. Pure: the tests drive it with simulated ceilings. **A need that cannot be computed
    /// buys nothing**: no buffer taken, the whole map streamed (no head), `need` `nil` — the preflight
    /// refuses such a request before (`ModelCard.checkMemory`), and the engine refuses a plan without one.
    package static func choose(_ request: MemoryNeed.Request, ceiling: Int) -> Render {
        var plan = request.lean
        let floor = MemoryNeed.of(plan)?.peak ?? .max
        for faster in [{ (r: inout MemoryNeed.Request) in r.encoderSlots = encoderSlots },
                       { (r: inout MemoryNeed.Request) in r.minimumBands = false },
                       { (r: inout MemoryNeed.Request) in if !r.minimumBands { r.wideBands = true } }] {
            var candidate = plan
            faster(&candidate)
            if let peak = MemoryNeed.of(candidate)?.peak, peak <= ceiling { plan = candidate }
        }
        // The head and the stream are sized on the denoising's need without the map: what `released` compares the map with.
        let streamedNeed = MemoryNeed.of(plan)
        var cached = plan
        cached.tail = .none
        let cachedNeed = MemoryNeed.of(cached)
        // The map read in place, last: the whole file charged beside the cached denoising, kept where it holds in B and where the
        // family's need charges it (a map its products would not read in place costs nothing and is not taken).
        if request.mapInPlaceAvailable {
            var inPlace = cached
            inPlace.mapInPlace = true
            if let candidate = MemoryNeed.of(inPlace), candidate.inPlaceBytes > 0, candidate.peak <= ceiling { plan = inPlace }
        }
        var need = plan.mapInPlace ? MemoryNeed.of(plan) : streamedNeed
        var (denoisingStreamed, denoisingCached) = (streamedNeed?.need(of: .denoising), cachedNeed?.need(of: .denoising))
        // The text encoder kept, after everything else: its layers charged to every phase, kept where the need with them holds in B
        // **and the DiT's map loses nothing to them** — read in place, or still whole in the room the cache has beside the
        // denoising (`released`). Where the map is streamed, that room is its head, read at every evaluation: taking it for an
        // encoder read once per prompt would cost more than it saves.
        // The denoising's needs the head is sized on (`released`) carry the layers too: they stay beside it.
        if need != nil, let mapBytes = MemoryNeed.ditMapBytes(request), let cachedDenoising = denoisingCached {
            var kept = plan
            kept.encoderKept = true
            kept.encoderSlots = 0
            if let candidate = MemoryNeed.of(kept), candidate.keptEncoderBytes > 0, candidate.peak <= ceiling,
               plan.mapInPlace || mapBytes <= ceiling - (cachedDenoising + candidate.keptEncoderBytes) {
                plan = kept
                need = candidate
                denoisingStreamed = denoisingStreamed.map { $0 + candidate.keptEncoderBytes }
                denoisingCached = cachedDenoising + candidate.keptEncoderBytes
            }
        }
        // Unknown: the denoising takes all of B — no room left for a head (`released`).
        var render = Render(ceiling: ceiling, request: plan, need: need, floor: floor,
                            denoisingStreamed: denoisingStreamed ?? ceiling, denoisingCached: denoisingCached ?? ceiling)
        guard let need else {
            render.decisions.append(String(format: "plan: need unknown in B %.2f — refused", Double(ceiling) / 1e9))
            return render
        }
        // Read in place, the map is wired for the GPU before the denoising where the phases before it hold in B beside it.
        let before = [MemoryNeed.Phase.Kind.text, .vision, .references].compactMap { need.need(of: $0) }.max() ?? 0
        render.warmsMap = plan.mapInPlace && before + need.inPlaceBytes <= ceiling
        render.decisions.append(String(format: "plan: need %.2f GB in B %.2f (floor %.2f, process base %.2f)%@%@%@%@",
                                       Double(need.peak) / 1e9, Double(ceiling) / 1e9, Double(floor) / 1e9,
                                       Double(need.base) / 1e9, (render.lean ? " — lean plan" : "") as NSString,
                                       (plan.mapInPlace ? String(format: " — map read in place (%.2f GB charged)",
                                                                 Double(need.inPlaceBytes) / 1e9) : "") as NSString,
                                       (render.warmsMap ? " and made resident beside the text" : "") as NSString,
                                       (plan.encoderKept ? String(format: " — text encoder kept (%.2f GB charged)",
                                                                  Double(need.keptEncoderBytes) / 1e9) : "") as NSString))
        return render
    }

    /// **The map's head, from B** — pure: the first layer streamed and the staging buffers.
    /// `spans`: each layer's bytes in the map; `mapBytes`: the whole map; `streamed` / `cached`: the
    /// denoising's need with the buffers and without; `fraction`: the model's own split (its cap).
    ///
    /// The map fits whole beside the denoising's need: nothing streamed, no buffer. Otherwise the
    /// layers from the front whose bytes fit in what B leaves (the map's non-layer tensors, cached
    /// too, deducted first), never more than the model's split keeps.
    package static func released(spans: [Int], mapBytes: Int, ceiling: Int, streamed: Int, cached: Int,
                                 fraction: Double) -> (released: Int, streamed: Bool) {
        let layers = spans.count
        guard fraction > 0, layers > 0, mapBytes > ceiling - cached else { return (layers, false) }
        let room = ceiling - streamed - (mapBytes - spans.reduce(0, +))
        var head = 0, held = 0
        while head < layers, held + spans[head] <= room { held += spans[head]; head += 1 }
        let released = min(head, Prefetcher.firstReleasedLayer(of: layers, tail: fraction))
        return released < layers ? (released, true) : (layers, false)
    }

    /// **What the DiT's stream reads from the disk at each evaluation**, computed as the engine decides it (`ditTail`): the layers `released`
    /// from the front of the sequence are read again at every evaluation — the tail, each layer's `spans` — and the head stays in the cache.
    /// 0 where the map fits B whole (`!streamed`: read once, cold, not per evaluation). Pure; `mapBytes` is the whole map.
    package static func streamedBytes(spans: [Int], mapBytes: Int, plan render: Render, fraction: Double) -> Int {
        if render.request.mapInPlace { return 0 }
        let (released, streamed) = Self.released(spans: spans, mapBytes: mapBytes, ceiling: render.ceiling,
                                                 streamed: render.denoisingStreamed, cached: render.denoisingCached, fraction: fraction)
        return streamed ? spans[released...].reduce(0, +) : 0
    }

    // ── the verdict, without a render ──────────────────────────────────────────────────────

    /// **What a request gets in a given B, decided as the preflight and the plan decide it** — no
    /// computation, no weights beyond the maps' headers: the table the tests print for 16, 24, 32, 64 and
    /// 128 GB Macs, and what the plan report prints.
    package enum Verdict: Sendable, Equatable {
        /// The render goes: the lean plan's `floor` is within B, and `need` is the chosen plan's peak.
        case plan(floor: Int, need: Int, encoderBuffers: Bool, standardBands: Bool, mapInPlace: Bool = false, encoderKept: Bool = false)
        /// B is too small: `needed` is the floor, `mac` the smallest Mac sold whose B holds it.
        case refused(needed: Int, mac: Int)
        /// No Mac's RAM holds it (`MemoryNeed.limit`); `area` is the largest area the family addresses.
        case unaddressable(area: Int)
        /// The maps' headers cannot be read: refused (`EngineError.memoryNeedUnknown`).
        case unknown

        /// One word for a table: `ok` plus what the plan gives up, `needs 64 GB`, `unaddressable`.
        package var label: String {
            switch self {
            case let .plan(_, _, encoderBuffers, standardBands, mapInPlace, encoderKept):
                return "ok" + (encoderBuffers && standardBands ? "" : " (lean" + (encoderBuffers ? "" : ", no encoder buffers")
                               + (standardBands ? "" : ", small bands") + ")")
                    + (mapInPlace ? " +map in place" : "") + (encoderKept ? " +encoder kept" : "")
            case let .refused(_, mac): return "needs \(mac) GB"
            case .unaddressable: return "unaddressable"
            case .unknown: return "unknown"
            }
        }
    }

    /// **The verdict of `request` in `ceiling`**: the engine's limit, then the lean plan's floor against B,
    /// then the plan `choose` takes. Pure on the request's maps.
    package static func verdict(_ request: MemoryNeed.Request, ceiling: Int) -> Verdict {
        guard let floor = MemoryNeed.of(request.lean) else { return .unknown }
        if let area = floor.addressableArea(width: request.width, height: request.height) { return .unaddressable(area: area) }
        guard floor.peak <= ceiling else { return .refused(needed: floor.peak, mac: MemoryBudget.macNeeded(for: floor.peak)) }
        let plan = choose(request, ceiling: ceiling)
        return .plan(floor: floor.peak, need: plan.need?.peak ?? floor.peak,
                     encoderBuffers: plan.request.encoderSlots > 0 || plan.request.encoderKept, standardBands: !plan.request.minimumBands,
                     mapInPlace: plan.request.mapInPlace, encoderKept: plan.request.encoderKept)
    }
}
