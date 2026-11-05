import Foundation
import XCTest
@testable import Siliconed

/// **B and the plan chosen in it** (`MemoryBudget`, `MemoryPlan`) — without a byte of weights: B is
/// simulated, the needs are Qwen-Image-2.1's on the maps of `MemoryFixtures` and 6 P cores, a map's
/// layers are of Z-Image's order.
final class MemoryBudgetTests: XCTestCase {

    private let gib = 1 << 30

    /// A Qwen-Image-2.1 request as the card builds it (a typical prompt, each reference at the output's
    /// size), on the fixture's maps, no process base.
    private func request(_ w: Int, _ h: Int, references: Int = 0) -> MemoryNeed.Request {
        var r = ModelCard.qwenImage21.memoryRequest(width: w, height: h, references: Array(repeating: (w, h), count: references),
                                                    base: 0)
        r.threads = 6
        r.qwenMaps = MemoryFixtures.qwenMaps()
        return r
    }

    private func floor(_ w: Int, _ h: Int, references: Int = 0) -> Int {
        ModelCard.qwenImage21.memoryNeed(request(w, h, references: references))!
    }

    /// **A generation is not charged its edits' peak**; the need never falls as references are added,
    /// and a count with no tier for the format takes the next one up.
    func testTheNeedFollowsTheReferences() {
        // Each reference adds its tokens to the text and the prefix: the edits grow with them. A
        // generation is not an edit (its denoising streams the map's tail, an edit's K/V go to the file).
        let needs = (1...3).map { floor(1024, 1024, references: $0) }
        XCTAssertEqual(needs, needs.sorted()); XCTAssertEqual(Set(needs).count, 3)
        // The need is the computed floor: the lean plan's peak, below the default plan's.
        XCTAssertEqual(floor(1024, 1024), MemoryNeed.of(request(1024, 1024).lean)!.peak)
        XCTAssertLessThanOrEqual(floor(1024, 1024), MemoryNeed.of(request(1024, 1024))!.peak)
        XCTAssertLessThan(floor(512, 512), floor(1024, 1024))
        XCTAssertLessThan(floor(1024, 1024), floor(1024, 1536))
    }

    /// **The process's base is in the need**: B covers the whole process — what the app or the
    /// command line holds when the render starts —, so a larger base is a larger need, byte for byte,
    /// in every phase; the card reads it from the process itself (`OperatorFootprint.processBase`).
    func testTheBaseIsInTheNeed() {
        var r = request(1024, 1024)
        let bare = MemoryNeed.of(r)!
        r.base = 300_000_000   // an app with its history
        let app = MemoryNeed.of(r)!
        XCTAssertEqual(app.peak - bare.peak, 300_000_000)
        for kind in bare.phases.map(\.kind) { XCTAssertEqual(app.need(of: kind)! - bare.need(of: kind)!, 300_000_000) }
        // Refused one byte short with it, accepted without it.
        XCTAssertThrowsError(try ModelCard.qwenImage21.checkMemory(r, ceiling: MemoryNeed.of(r.lean)!.peak - 1))
        XCTAssertNoThrow(try ModelCard.qwenImage21.checkMemory(r, ceiling: MemoryNeed.of(r.lean)!.peak))
        // The card's own request reads this process.
        let read = ModelCard.qwenImage21.memoryRequest(width: 1024, height: 1024).base
        XCTAssertGreaterThan(read, 1_000_000)
        // The check compares the process's peaks with needs that hold the same base.
        let check = MemoryCheck(need: app, marks: [("text", app.need(of: .text)! - 1)])
        XCTAssertEqual(check.baseBytes, 300_000_000)
        XCTAssertTrue(check.exceeded.isEmpty)
        XCTAssertEqual(check.slack.count, 1)
    }

    /// **A need that cannot be computed is a refusal**, with its own code — never a need of 0 that
    /// every B would hold, nor a plan with buffers bought on it.
    func testAnUnknownNeedIsRefused() {
        var r = request(1024, 1024)
        r.qwenMaps = nil
        let library = Library(root: FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-empty-\(UUID())"))
        XCTAssertNil(QwenImage21Maps(card: .qwenImage21, library: library))
        // A card whose maps are read from a library that has none: its need is unknown.
        let q = ModelCard.qwenImage21
        let imported = ModelCard(id: "qwen-image-2.1/absent", name: "Absent", family: .qwenImage21,
                                 map: library.map("absent.silicon"), license: q.license, defaultSteps: q.defaultSteps,
                                 space: q.space, formats: q.formats)
        var unknown = r
        unknown.card = imported
        XCTAssertNil(MemoryNeed.of(unknown))
        XCTAssertThrowsError(try imported.checkMemory(unknown, ceiling: 1 << 40)) {
            XCTAssertEqual($0 as? EngineError, .memoryNeedUnknown(model: imported.id))
            XCTAssertEqual(($0 as? EngineError)?.code, "memory_need_unknown")
        }
        let plan = MemoryPlan.choose(unknown, ceiling: 1 << 40)
        XCTAssertNil(plan.need)
        XCTAssertTrue(plan.lean)
        XCTAssertEqual(plan.denoisingStreamed, 1 << 40)   // no room left for a head
        XCTAssertThrowsError(try Preflight.check(card: imported, width: 1024, height: 1024, installed: true,
                                                 licenseAccepted: true, ceiling: 1 << 40, memory: unknown))
    }

    /// `SILICONED_MEMORY_BUDGET_GB` replaces B (GiB, comma or point); without it, B is this Mac's RAM's.
    func testTheForcingReplacesB() {
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [MemoryBudget.forcingVariable: "4"]), 4 << 30)
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [MemoryBudget.forcingVariable: "4,5"]), 9 << 29)
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [MemoryBudget.forcingVariable: "22"]), 22 << 30)
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [MemoryBudget.forcingVariable: "0"]), 0)
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [:]), MemoryBudget.ceiling(physical: physical))
        // What `Int(Double)` cannot hold, or is no budget, is refused — B stays this Mac's.
        for text in ["inf", "-inf", "nan", "-1", "1e30", "1025", "four", ""] {
            XCTAssertNil(MemoryBudget.forcedCeiling([MemoryBudget.forcingVariable: text]), text)
            XCTAssertEqual(MemoryBudget.machineCeiling(environment: [MemoryBudget.forcingVariable: text]),
                           MemoryBudget.ceiling(physical: physical), text)
        }
        XCTAssertEqual(MemoryBudget.forcedCeiling([MemoryBudget.forcingVariable: "1024"]), 1 << 40)
    }

    // ── B ──────────────────────────────────────────────────────────────────────────────────

    /// **B = ¾ · RAM − 2 GiB**: 8 → 4, 16 → 10, 24 → 16, 32 → 22, 64 → 46; never negative below.
    func testBIsThreeQuartersOfTheRamLessTwo() {
        XCTAssertEqual([8, 16, 24, 32, 64].map { MemoryBudget.ceiling(physical: $0 * gib) / gib }, [4, 10, 16, 22, 46])
        XCTAssertEqual(MemoryBudget.ceiling(physical: 6 * gib), 5 * gib / 2)
        XCTAssertEqual(MemoryBudget.ceiling(physical: 2 * gib), 0)
    }

    /// **The inverse**: the smallest RAM, in whole GiB, whose B covers a need — the app's « needs a
    /// Mac with N GB ».
    func testTheSmallestMacInvertsB() {
        XCTAssertEqual(MemoryBudget.physicalNeeded(for: 4 * gib), 8)
        XCTAssertEqual(MemoryBudget.physicalNeeded(for: 4 * gib + 1), 9)
        XCTAssertEqual(MemoryBudget.physicalNeeded(for: 10 * gib), 16)
        XCTAssertEqual(MemoryBudget.physicalNeeded(for: 22 * gib), 32)
        for need in stride(from: 0, to: 40 * gib, by: 97_654_321) {
            let n = MemoryBudget.physicalNeeded(for: need)
            XCTAssertGreaterThanOrEqual(MemoryBudget.ceiling(physical: n * gib), need)
            XCTAssertTrue(n == 1 || MemoryBudget.ceiling(physical: (n - 1) * gib) < need, "\(need)")
        }
    }

    /// **The Mac to buy**: the inverse rounded up to a size sold — 9 GiB raw is a 16 GB Mac, 17 an 18.
    func testTheMacNeededIsASizeSold() {
        XCTAssertEqual(MemoryBudget.macNeeded(for: 4 * gib), 8)
        XCTAssertEqual(MemoryBudget.macNeeded(for: 4 * gib + 1), 16)
        XCTAssertEqual(MemoryBudget.macNeeded(for: MemoryBudget.ceiling(physical: 17 * gib)), 18)
        XCTAssertEqual(MemoryBudget.macNeeded(for: 1000 * gib), MemoryBudget.physicalNeeded(for: 1000 * gib))
        for need in stride(from: 0, to: 300 * gib, by: 977_654_321) {
            let n = MemoryBudget.macNeeded(for: need)
            XCTAssertGreaterThanOrEqual(MemoryBudget.ceiling(physical: n * gib), need)
            if n <= 512 { XCTAssertTrue(MemoryBudget.soldSizes.contains(n), "\(n)") }
        }
    }

    // ── the preflight ───────────────────────────────────────────────────────────────────────

    /// **The floor is judged against B, and against B alone** — the one check the app also asks.
    func testTheFloorIsRefusedAboveB() {
        // Refused one byte short, with both figures.
        let q = ModelCard.qwenImage21
        let floor = self.floor(1024, 1024)
        XCTAssertThrowsError(try q.checkMemory(request(1024, 1024), ceiling: floor - 1)) {
            XCTAssertEqual($0 as? EngineError, .insufficientMemory(needed: floor, budget: floor - 1))
        }
        XCTAssertNoThrow(try q.checkMemory(request(1024, 1024), ceiling: floor))
        XCTAssertNoThrow(try q.checkMemory(request(512, 512), ceiling: floor - 1))
        // A 16 GB Mac takes Qwen's worst case, 1024×1536 with 3 references.
        XCTAssertNoThrow(try q.checkMemory(request(1024, 1536, references: 3), ceiling: MemoryBudget.ceiling(physical: 16 * gib)))
        // The engine's request: a LoRA and a long prompt raise the need it is judged on.
        var plain = request(1024, 1024)
        plain.promptTokens = 50
        var loaded = plain; loaded.loraRanks = [128]
        XCTAssertGreaterThan(q.memoryNeed(loaded)!, q.memoryNeed(plain)!)
        var long = plain; long.promptTokens = 526
        XCTAssertGreaterThan(q.memoryNeed(long)!, q.memoryNeed(plain)!)
    }

    // ── the plan, chosen in B ───────────────────────────────────────────────────────────────

    /// **The fastest plan B holds, and nothing above it.** A roomy B takes every buffer; a B between
    /// the floor and the fastest plan keeps what fits; at the floor, the lean plan. The plan's need is
    /// never above B once the floor is.
    func testThePlanIsTheFastestBHolds() {
        let request = self.request(1024, 1024, references: 3)
        let floor = MemoryNeed.of(request.lean)!.peak
        let fastest = MemoryNeed.of(request)!.peak
        // The encoder's buffers are the largest tensor's now: they no longer raise an edit's peak.
        XCTAssertLessThanOrEqual(floor, fastest)
        let roomy = MemoryPlan.choose(request, ceiling: 16 << 30)
        XCTAssertEqual(roomy.request.encoderSlots, MemoryPlan.encoderSlots)
        XCTAssertFalse(roomy.request.minimumBands)
        XCTAssertEqual(roomy.need?.peak, fastest)
        XCTAssertEqual(roomy.floor, floor)
        XCTAssertFalse(roomy.lean)
        // At the floor, what raises the peak goes; what does not raise it stays — a band in a phase below
        // the peak is free, and so are the encoder's three small buffers.
        let tight = MemoryPlan.choose(request, ceiling: floor)
        XCTAssertEqual(tight.request.encoderSlots, MemoryPlan.encoderSlots)
        XCTAssertEqual(tight.need?.peak, floor)
        for ceiling in stride(from: floor, through: fastest, by: max(1, (fastest - floor) / 17)) {
            let plan = MemoryPlan.choose(request, ceiling: ceiling)
            XCTAssertLessThanOrEqual(plan.need!.peak, ceiling)
            XCTAssertGreaterThanOrEqual(plan.need!.peak, floor)
        }
        // The map's tail is the lean one in every plan: its buffers are charged, its head is not.
        XCTAssertEqual(roomy.request.tail, .lean)
        XCTAssertLessThanOrEqual(roomy.denoisingCached, roomy.denoisingStreamed)
    }

    /// **The map's head: what B leaves beside the denoising, capped at the model's split.** Thirty
    /// layers of 400 MB and 500 MB of other tensors (a 12.5 GB map, Z-Image's order).
    func testTheHeadIsWhatBLeavesCappedAtTheSplit() {
        let spans = [Int](repeating: 400_000_000, count: 30), map = 12_500_000_000
        func plan(_ ceiling: Int, _ fraction: Double = 0.5) -> (released: Int, streamed: Bool) {
            MemoryPlan.released(spans: spans, mapBytes: map, ceiling: ceiling, streamed: 3_000_000_000,
                                cached: 2_300_000_000, fraction: fraction)
        }
        // A 16 GB Mac's B: room for more than half, capped at half.
        XCTAssertEqual(plan(MemoryBudget.ceiling(physical: 16 << 30)).released, 15)
        // Less room, fewer layers: 1.5 GB left beside the denoising and the other tensors hold 3 layers.
        XCTAssertEqual(plan(3_000_000_000 + 500_000_000 + 1_500_000_000).released, 3)
        // No room: the whole map streamed.
        XCTAssertEqual(plan(3_000_000_000).released, 0)
        XCTAssertTrue(plan(3_000_000_000).streamed)
        // A model whose split is the whole map (Qwen-Image-2.1's generation): no head.
        XCTAssertEqual(plan(MemoryBudget.ceiling(physical: 16 << 30), 1).released, 0)
        // **The whole map fits beside the denoising's need without buffers: nothing streamed**.
        XCTAssertTrue(plan(map + 2_300_000_000) == (30, false))
        XCTAssertTrue(plan(map + 2_300_000_000, 1) == (30, false))
        XCTAssertTrue(plan(map + 2_300_000_000 - 1).streamed)
        // A `map_tail` of 0: everything through the cache, as before the tail.
        XCTAssertTrue(plan(3_000_000_000, 0) == (30, false))
        XCTAssertEqual(MemoryPlan.Tail.lean, MemoryPlan.Tail(fraction: 1, streamed: true))
        XCTAssertEqual(Prefetcher.firstReleasedLayer(of: 30, tail: MemoryPlan.Tail.lean.fraction), 0)
    }

    /// **The map read in place is a choice of the plan, with its price**: the whole DiT file charged to the denoising, taken only where
    /// the need with it holds in B — a 64 GB Mac's, not a 16 or an 8 GB one's, whose plans stay those of a process that cannot read in
    /// place (the same request, the same need, the same tail). Never where the process does not read in place.
    func testTheMapReadInPlaceIsChargedAndTakenOnlyWhereBHoldsIt() {
        var request = MemoryFixtures.zImageRequest(1024, 1024)
        XCTAssertFalse(request.mapInPlaceAvailable)   // no `SILICONED_STREAM` in the tests' process
        let without = request
        request.mapInPlaceAvailable = true
        let page = Int(getpagesize())
        let file = Artifact.mapBytes(header: MemoryFixtures.zImageMaps().dit)
        XCTAssertEqual(MemoryFixtures.zImageMaps().ditInPlaceBytes, (file + page - 1) / page * page)

        let roomy = MemoryPlan.choose(request, ceiling: MemoryBudget.ceiling(physical: 64 << 30))
        XCTAssertTrue(roomy.request.mapInPlace)
        XCTAssertEqual(roomy.request.tail, .none)
        let need = roomy.need!
        XCTAssertEqual(need.inPlaceBytes, MemoryFixtures.zImageMaps().ditInPlaceBytes)
        XCTAssertLessThanOrEqual(need.peak, roomy.ceiling)
        XCTAssertEqual(need.need(of: .denoising)!, roomy.denoisingCached + need.inPlaceBytes)
        // The head and the stream are sized as before, on the denoising's need without the map.
        let before = MemoryPlan.choose(without, ceiling: roomy.ceiling)
        XCTAssertFalse(before.request.mapInPlace)
        XCTAssertEqual(before.denoisingCached, roomy.denoisingCached)
        XCTAssertEqual(before.denoisingStreamed, roomy.denoisingStreamed)

        for ram in [8, 16] {
            let ceiling = MemoryBudget.ceiling(physical: ram << 30)
            let plan = MemoryPlan.choose(request, ceiling: ceiling), old = MemoryPlan.choose(without, ceiling: ceiling)
            XCTAssertFalse(plan.request.mapInPlace, "\(ram) GB")
            XCTAssertEqual(plan.need, old.need, "\(ram) GB")
            XCTAssertEqual(plan.request.tail, old.request.tail)
            XCTAssertEqual(plan.request.encoderSlots, old.request.encoderSlots)
            XCTAssertEqual(plan.request.minimumBands, old.request.minimumBands)
        }
        // One byte short of the need with the map: the copy into the reserve, nothing charged.
        let edge = MemoryPlan.choose(request, ceiling: need.peak - 1)
        XCTAssertFalse(edge.request.mapInPlace)
        XCTAssertEqual(edge.need?.inPlaceBytes, 0)
        XCTAssertTrue(MemoryPlan.choose(request, ceiling: need.peak).request.mapInPlace)

        // The products ask the render's plan; outside a render (a check), wherever the map is open.
        XCTAssertTrue(MemoryPlan.mapInPlace)
        MemoryPlan.during(edge) { XCTAssertFalse(MemoryPlan.mapInPlace) }
        MemoryPlan.during(roomy) { XCTAssertTrue(MemoryPlan.mapInPlace) }
    }

    /// **Outside a render, the reference values**; inside, the plan's — the same from the first site
    /// to the last — and the render's state is gone once it ends.
    func testTheRenderStateIsScopedToTheRender() {
        let request = self.request(1024, 1024)
        XCTAssertNil(MemoryPlan.current)
        XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 18, minimum: MemoryPlan.qwenMinimumBand, unit: "pixels"), 1 << 18)
        var tight = MemoryPlan.choose(request, ceiling: 16 << 30)
        tight.request.minimumBands = true
        MemoryPlan.during(tight) {
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 18, minimum: MemoryPlan.qwenMinimumBand, unit: "pixels"),
                           MemoryPlan.qwenMinimumBand)
            // A check that forced a smaller band keeps it.
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 12, minimum: MemoryPlan.qwenMinimumBand, unit: "pixels"), 1 << 12)
        }
        XCTAssertNil(MemoryPlan.current)
        XCTAssertEqual(MemoryPlan.lastRender?.decisions.count, 3)
        MemoryPlan.during(MemoryPlan.choose(request, ceiling: 16 << 30)) {
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 18, minimum: MemoryPlan.qwenMinimumBand, unit: "pixels"), 1 << 18)
        }
    }

    /// **The need is the economical plan's floor, computed**, below the fastest plan's. A rotated format needs what its transpose needs (the same shapes).
    func testTheNeedIsTheEconomicalFloor() {
        func comfort(_ w: Int, _ h: Int, references: Int = 0) -> Int { MemoryNeed.of(request(w, h, references: references))!.peak }
        for (w, h) in [(1024, 1024), (832, 1216), (1024, 1536)] {
            XCTAssertLessThanOrEqual(floor(w, h), comfort(w, h), "\(w)×\(h)")
            XCTAssertLessThanOrEqual(floor(w, h, references: 3), comfort(w, h, references: 3))
        }
        XCTAssertEqual(floor(832, 1216), floor(1216, 832))
    }
}
