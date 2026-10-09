import Foundation
import XCTest
import Synchronization
@testable import Siliconed

/// **The budget and the plan it derives** (`MemoryBudget`, `MemoryPlan`) — without a byte of weights:
/// the machine's state is injected (page counts, the forcing), the maps' sizes are Z-Image's and
/// Qwen-Image-2.1's, from the journal.
final class MemoryBudgetTests: XCTestCase {

    private let gib = 1 << 30
    private func budget(_ reclaimableGiB: Double) -> MemoryBudget {
        MemoryBudget(reclaimable: Int(reclaimableGiB * Double(gib)))
    }

    // The maps (measured: Z-Image bf16 12.33 GB, Qwen-Image-2.1 14.2 GB swept per evaluation; two
    // staging buffers ~0.7 GB for Z, ~0.9 for Qwen) and the floors of `ModelCard.memoryNeed`.
    private let zBlock = 360_000_000, qwenBlock = 450_000_000
    private var z512: Int { ModelCard.zImage.memoryNeed(width: 512, height: 512) }
    private var z1024: Int { ModelCard.zImage.memoryNeed(width: 1024, height: 1024) }
    private var z1024Comfort: Int { ModelCard.zImage.memoryComfort(width: 1024, height: 1024) }

    // ── the reading ─────────────────────────────────────────────────────────────────────────

    /// The formula: speculative pages are in `free_count` AND in `external_page_count` — counted once;
    /// executable pages are taken out of the file-backed ones; the process's own footprint is added.
    func testTheReadingCountsSpeculativePagesOnce() {
        let pages = MemoryBudget.Pages(free: 4_880, speculative: 797, purgeable: 4_994, external: 453_304, executable: 30_000)
        let page = 16_384
        XCTAssertEqual(MemoryBudget.reclaimable(pages, pageSize: page, ownFootprint: 0),
                       page * (4_880 - 797 + 4_994 + 453_304 - 30_000))
        XCTAssertEqual(MemoryBudget.reclaimable(pages, pageSize: page, ownFootprint: 500_000_000)
                       - MemoryBudget.reclaimable(pages, pageSize: page, ownFootprint: 0), 500_000_000)
        // Never negative, whatever the kernel says.
        XCTAssertEqual(MemoryBudget.reclaimable(.init(free: 1, speculative: 5, external: 2, executable: 9),
                                                pageSize: page, ownFootprint: 0), 0)
    }

    /// `vm_statistics64` read at its offsets: a rev5 kernel's `executable_count` is taken as written;
    /// an older kernel's shorter answer gets the assumed 2 GiB, never more than the file-backed pages.
    func testTheRawReadingFallsBackWhenTheKernelHasNoExecutableCount() {
        var raw = [integer_t](repeating: 0, count: 256)
        raw[0] = 4_880; raw[92 / 4] = 797; raw[88 / 4] = 4_994; raw[136 / 4] = 453_304
        raw[280 / 4] = 30_000; raw[280 / 4 + 1] = 1   // 2³² + 30 000: the high word is read too
        let page = 16_384
        let rev5 = MemoryBudget.pages(raw: raw, count: 104, pageSize: page)
        XCTAssertEqual(rev5, .init(free: 4_880, speculative: 797, purgeable: 4_994, external: 453_304,
                                   executable: (1 << 32) + 30_000))
        let rev4 = MemoryBudget.pages(raw: raw, count: 70, pageSize: page)
        XCTAssertEqual(rev4.executable, MemoryBudget.assumedExecutableBytes / page)
        XCTAssertEqual(rev4.external, 453_304)
        raw[136 / 4] = 1_000
        XCTAssertEqual(MemoryBudget.pages(raw: raw, count: 70, pageSize: page).executable, 1_000)
    }

    /// The reserve is subtracted, and the budget never goes below zero.
    func testTheReserveIsSubtracted() {
        XCTAssertEqual(MemoryBudget.reserve, 1_600_000_000)
        XCTAssertEqual(budget(8).available, (8 << 30) - 1_600_000_000)
        XCTAssertEqual(budget(1).available, 0)
    }

    /// `SILICONED_MEMORY_AVAILABLE_GB` replaces the reading (GiB, before the reserve, comma or point);
    /// without it, the machine's state, which holds at least this process.
    func testTheForcingReplacesTheReading() {
        let forced = MemoryBudget.current(environment: [MemoryBudget.forcingVariable: "8"])
        XCTAssertEqual(forced, MemoryBudget(reclaimable: 8 << 30, forced: true))
        XCTAssertEqual(forced.available, (8 << 30) - MemoryBudget.reserve)
        XCTAssertEqual(MemoryBudget.current(environment: [MemoryBudget.forcingVariable: "4,5"]).reclaimable, 9 << 29)
        let read = MemoryBudget.current(environment: [:])
        XCTAssertFalse(read.forced)
        XCTAssertGreaterThanOrEqual(read.reclaimable, Arena.processFootprint() / 2)
        XCTAssertLessThan(read.reclaimable, Int(ProcessInfo.processInfo.physicalMemory) + Arena.processFootprint())
    }

    // ── the preflight ───────────────────────────────────────────────────────────────────────

    /// **The floor is judged against the budget, reserve included** — the one check the app also asks.
    func testTheFloorIsRefusedBelowTheBudget() {
        // 3.5 GiB reclaimable → 2.16 GB available < Z-Image's 1024² floor (2.214 GB): refused, with both figures.
        XCTAssertThrowsError(try ModelCard.zImage.checkMemory(width: 1024, height: 1024, budget: budget(3.5))) {
            XCTAssertEqual($0 as? EngineError, .insufficientMemory(needed: 2_214_000_000, available: Int(3.5 * Double(1 << 30)) - 1_600_000_000))
        }
        XCTAssertNoThrow(try ModelCard.zImage.checkMemory(width: 512, height: 512, budget: budget(3.5)))
        XCTAssertNoThrow(try ModelCard.zImage.checkMemory(width: 1024, height: 1024, budget: budget(4)))
        // A measured ballast (5.05 GB reclaimable, 3 GiB incompressible elsewhere) swapped at the default plan;
        // the economical floor (3.2 GB) accepts it now. Whether it swaps there is a measurement's question:
        // if it does, the reserve rises, not this line.
        XCTAssertNoThrow(try ModelCard.zImage.checkMemory(width: 1024, height: 1536,
                                                         budget: MemoryBudget(reclaimable: 5_050_000_000)))
        // Every render observed without swap passes: the smallest reading (6.12 GB) holds the largest need
        // (Qwen's 1024² edit with 3 references, 4.51 GB).
        XCTAssertNoThrow(try ModelCard.qwenImage21.checkMemory(width: 1024, height: 1024,
                                                              budget: MemoryBudget(reclaimable: 6_120_000_000)))
        // Qwen at 1024×1536 (4.74 GB) needs 6.34 GB reclaimable.
        XCTAssertThrowsError(try ModelCard.qwenImage21.checkMemory(width: 1024, height: 1536, budget: budget(5.85)))
        XCTAssertNoThrow(try ModelCard.qwenImage21.checkMemory(width: 1024, height: 1536, budget: budget(6)))
    }

    // ── #1–#4 · the map's tail ──────────────────────────────────────────────────────────────

    /// **Simulated budgets 8, 12, 16, 32 GiB**: the fraction stays the measured 0.5 (its derivation waits for
    /// a timed comparison); only the two buffers depend on the budget.
    func testTheTailKeepsItsFractionAndDerivesItsBuffers() {
        let tails = [8.0, 12, 16, 32].map { MemoryPlan.tail(budget: budget($0), floor: z512, blockBytes: zBlock, fraction: 0.5) }
        XCTAssertEqual(tails, [MemoryPlan.Tail(fraction: 0.5, slots: 2)] * 4)
        XCTAssertEqual(MemoryPlan.tail(budget: budget(32), floor: z512, blockBytes: zBlock, fraction: 0), .none)
    }

    /// The default plan's half when the room above the floor holds its two buffers; otherwise the lean
    /// plan, **the whole map beside the cache with the same two buffers** — never the whole map
    /// through the cache, which is what swapped, and never one buffer.
    func testTheBuffersComeFromTheRoomAboveTheFloor() {
        XCTAssertEqual(MemoryPlan.tail(budget: budget(3.8), floor: z512, blockBytes: zBlock, fraction: 0.5),
                       MemoryPlan.Tail(fraction: 0.5, slots: 2))
        XCTAssertEqual(MemoryPlan.tail(budget: budget(3.6), floor: z512, blockBytes: zBlock, fraction: 0.5), .lean)
        XCTAssertEqual(MemoryPlan.Tail.lean, MemoryPlan.Tail(fraction: 1, slots: 2))
        let worst = ModelCard.qwenImage21.memoryNeed(width: 1024, height: 1536)
        XCTAssertEqual(MemoryPlan.tail(budget: budget(5.9), floor: worst, blockBytes: qwenBlock, fraction: 0.5), .lean)
        XCTAssertEqual(MemoryPlan.tail(budget: budget(7.0), floor: worst, blockBytes: qwenBlock, fraction: 0.5).slots, 2)
        XCTAssertEqual(Prefetcher.firstReleasedLayer(of: 30, tail: MemoryPlan.Tail.lean.fraction), 0)
    }

    // ── #5 · the text encoder ───────────────────────────────────────────────────────────────

    func testTheEncoderBuffersFollowTheRoomLeft() {
        let block = 385_000_000   // a Qwen3-VL-8B layer (measured: 282 → 1 820 MB with four)
        XCTAssertEqual(MemoryPlan.encoderSlots(budget: budget(8), footprint: 300_000_000, blockBytes: block), 4)
        XCTAssertEqual(MemoryPlan.encoderSlots(budget: MemoryBudget(reclaimable: 1_900_000_000, reserve: 0),
                                               footprint: 300_000_000, blockBytes: block), 4)
        XCTAssertEqual(MemoryPlan.encoderSlots(budget: MemoryBudget(reclaimable: 1_800_000_000, reserve: 0),
                                               footprint: 300_000_000, blockBytes: block), 0)
    }

    // ── #10, #11 · the decoders ─────────────────────────────────────────────────────────────

    /// The measured band when the budget read after the denoising covers the floor, the smallest
    /// proved one otherwise — never larger.
    func testTheDecoderBandFollowsTheDecodingBudget() {
        XCTAssertEqual(MemoryPlan.band(budget: budget(8), floor: z1024Comfort, standard: 1 << 23,
                                       minimum: MemoryPlan.fluxMinimumBand), 1 << 23)
        XCTAssertEqual(MemoryPlan.band(budget: budget(4), floor: z1024Comfort, standard: 1 << 23,
                                       minimum: MemoryPlan.fluxMinimumBand), 1 << 20)
        XCTAssertEqual(MemoryPlan.band(budget: budget(4), floor: z1024Comfort, standard: 1 << 18,
                                       minimum: MemoryPlan.qwenMinimumBand), 1 << 16)
        // A check that forced a smaller band keeps it.
        XCTAssertEqual(MemoryPlan.band(budget: budget(4), floor: z1024Comfort, standard: 1 << 12, minimum: 1 << 20), 1 << 12)
    }

    /// **Outside a render, the reference values**; inside, the decoders wait for the budget read after
    /// the denoising, and the render's state is gone once it ends.
    func testTheRenderStateIsScopedToTheRender() {
        XCTAssertNil(MemoryPlan.current)
        XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 23, minimum: 1 << 20, unit: "elements"), 1 << 23)
        let comfort = ModelCard.zImage.memoryComfort(width: 1024, height: 1024)
        MemoryPlan.during(budget: budget(5), floor: z1024, comfortable: comfort, pressure: { nil }) {
            XCTAssertEqual(MemoryPlan.current?.floor, z1024)
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 23, minimum: 1 << 20, unit: "elements"), 1 << 23)
            MemoryPlan.readForDecoding(environment: [MemoryBudget.forcingVariable: "4"])
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 23, minimum: 1 << 20, unit: "elements"), 1 << 20)
            MemoryPlan.readForDecoding(environment: [MemoryBudget.forcingVariable: "8"])
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 23, minimum: 1 << 20, unit: "elements"), 1 << 23)
        }
        XCTAssertNil(MemoryPlan.current)
        XCTAssertEqual(MemoryPlan.lastRender?.decisions.count, 2)
    }

    /// **The need is the economical plan's floor where it was measured**, the default plan's
    /// peak stays the decoders' threshold. Z-Image 1024²: 2.214 GB to render, 2.936 to take every
    /// buffer; 832×1216 now has its own tier (it borrowed the 1024²'s); 512² has no floor measured.
    func testTheNeedIsTheEconomicalFloor() {
        let z = ModelCard.zImage
        XCTAssertEqual(z.memoryNeed(width: 1024, height: 1024), 2_214_000_000)
        XCTAssertEqual(z.memoryComfort(width: 1024, height: 1024), 2_936_000_000)
        XCTAssertEqual(z.memoryNeed(width: 832, height: 1216), 2_077_000_000)
        XCTAssertEqual(z.memoryNeed(width: 1216, height: 832), 2_077_000_000)
        XCTAssertEqual(z.memoryNeed(width: 1024, height: 1536), 3_200_000_000)
        XCTAssertEqual(z.memoryComfort(width: 1024, height: 1536), 3_774_000_000)
        XCTAssertEqual(z.memoryNeed(width: 512, height: 512), 1_695_000_000)
        // The desktop of 07/10: 4.23 GB reclaimable, 2.63 available — refused at 1216×832
        // before (2.73 GB asked), accepted now; its decoding takes the smallest band.
        let desk = MemoryBudget(reclaimable: 4_230_000_000)
        XCTAssertNoThrow(try z.checkMemory(width: 1216, height: 832, budget: desk))
        XCTAssertNoThrow(try z.checkMemory(width: 1024, height: 1024, budget: desk))
        XCTAssertThrowsError(try z.checkMemory(width: 1024, height: 1536, budget: desk))
        XCTAssertEqual(MemoryPlan.tail(budget: desk, floor: z.memoryNeed(width: 1216, height: 832), blockBytes: zBlock), .lean)
        XCTAssertEqual(MemoryPlan.band(budget: desk, floor: z.memoryComfort(width: 1216, height: 832),
                                       standard: 1 << 23, minimum: MemoryPlan.fluxMinimumBand), 1 << 20)
    }

    // ── the pressure, read during the render ─────────────────────────────────────────────────

    /// The measured limit, 12/22 × (physical − wired): on this machine (17.18 GB, ~3 GB wired at rest)
    /// ≈ 7.7 GB — where the swap was seen to start (7.2–7.7 GB occupied).
    func testTheLimitIsM131s() {
        let p = MemoryPressure(compressor: 6_000_000_000, wired: 3_000_000_000, physical: 17_179_869_184)
        XCTAssertEqual(Double(p.limit) / 1e9, 7.734, accuracy: 0.01)
        XCTAssertEqual(Double(p.headroom) / 1e9, 1.734, accuracy: 0.01)
    }

    /// **Short of room, the render goes lean and stays lean**: the DiT's whole map streamed, the
    /// smallest decoder band, even when the launch budget would have given them; the room coming back
    /// does not bring them back. With room, nothing changes.
    func testTheRenderGoesLeanWhenTheCompressorRunsShort() {
        let short = MemoryPressure(compressor: 7_000_000_000, wired: 3_000_000_000, physical: 17_179_869_184)
        let roomy = MemoryPressure(compressor: 2_000_000_000, wired: 3_000_000_000, physical: 17_179_869_184)
        let reading = Mutex(roomy)
        MemoryPlan.during(budget: budget(12), floor: z1024, comfortable: z1024, pressure: { reading.withLock { $0 } }) {
            MemoryPlan.readForDecoding(environment: [MemoryBudget.forcingVariable: "12"])
            XCTAssertFalse(MemoryPlan.relieved("denoising"))
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 23, minimum: 1 << 20, unit: "elements"), 1 << 23)
            reading.withLock { $0 = short }
            XCTAssertEqual(MemoryPlan.decoderBand(standard: 1 << 23, minimum: 1 << 20, unit: "elements"), 1 << 20)
            reading.withLock { $0 = roomy }
            XCTAssertTrue(MemoryPlan.relieved("decoding"))
        }
        XCTAssertFalse(MemoryPlan.relieved("decoding"))   // outside a render
        // The forcing: a room of 1 GiB is short, 4 GiB is not; a forced budget alone reads no pressure.
        XCTAssertLessThan(MemoryPressure.current(environment: [MemoryPressure.forcingVariable: "1"])!.headroom, MemoryPressure.margin)
        XCTAssertGreaterThan(MemoryPressure.current(environment: [MemoryPressure.forcingVariable: "4"])!.headroom, MemoryPressure.margin)
        XCTAssertNil(MemoryPressure.current(environment: [MemoryBudget.forcingVariable: "8"]))
        // `GIB@SECONDS`: due only from that time into the render — never outside one.
        XCTAssertEqual(MemoryPressure.forcedRoom("1", elapsed: nil), 1 << 30)
        XCTAssertEqual(MemoryPressure.forcedRoom("0,5@20", elapsed: 25), 1 << 29)
        XCTAssertNil(MemoryPressure.forcedRoom("0,5@20", elapsed: 5))
        XCTAssertNil(MemoryPressure.forcedRoom("0,5@20", elapsed: nil))
        XCTAssertNil(MemoryPressure.forcedRoom("x", elapsed: 25))
    }

    /// **The watcher latches a short reading between two sites** (measured: read only at the
    /// evaluations' starts, the room swung back up before the next one). No site asks here; the next
    /// one finds the render lean although the room came back. The render that ends takes its watcher.
    func testTheWatcherLatchesBetweenTwoSites() {
        let short = MemoryPressure(compressor: 7_000_000_000, wired: 3_000_000_000, physical: 17_179_869_184)
        let roomy = MemoryPressure(compressor: 2_000_000_000, wired: 3_000_000_000, physical: 17_179_869_184)
        let reading = Mutex(roomy)
        MemoryPlan.during(budget: budget(12), floor: z1024, comfortable: z1024,
                          pressure: { reading.withLock { $0 } }, watch: 0.005) {
            reading.withLock { $0 = short }
            let deadline = Date().addingTimeInterval(2)
            while MemoryPlan.current?.relieved != true, Date() < deadline { usleep(2_000) }
            reading.withLock { $0 = roomy }
            XCTAssertEqual(MemoryPlan.current?.relieved, true)
            XCTAssertTrue(MemoryPlan.relieved("denoising"))
            XCTAssertTrue(MemoryPlan.current?.decisions.first?.contains("s into the render") ?? false)
        }
        MemoryPlan.during(budget: budget(12), floor: z1024, comfortable: z1024, pressure: { roomy }, watch: 0.005) {
            usleep(20_000)
            XCTAssertEqual(MemoryPlan.current?.relieved, false)
        }
    }
}

private func * <T>(array: [T], count: Int) -> [T] { Array((0..<count).map { _ in array }.joined()) }
