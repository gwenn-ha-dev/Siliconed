import XCTest
@testable import Siliconed

/// **Family × format × RAM → plan or refusal, computed** — no ceiling on the area, B decides: the
/// table a Mac of 16, 24, 32, 64 or 128 GB gets, from the formulas on the published shapes
/// (`MemoryFixtures`: the Standard maps, 6 P cores, a typical prompt, no process base — the same
/// figures on every Mac, the CI's included). Nothing renders and nothing is read from the machine.
///
/// `PRINT_FORMAT_TABLE=1 swift test --filter FormatTableTests` prints it.
final class FormatTableTests: XCTestCase {

    static let rams = [16, 24, 32, 64, 128]
    static let formats = [(1024, 1024), (1024, 1536), (1536, 1536), (2048, 2048), (2048, 3072)]
    static let cards: [ModelCard] = [.zImage, .qwenImage21, .anima, .krea2]

    private func request(_ card: ModelCard, _ w: Int, _ h: Int) -> MemoryNeed.Request {
        var r = card.memoryRequest(width: w, height: h, base: 0)
        r.threads = 6
        r.zImageMaps = MemoryFixtures.zImageMaps()
        r.qwenMaps = MemoryFixtures.qwenMaps()
        r.animaMaps = MemoryFixtures.animaMaps()
        r.krea2Maps = MemoryFixtures.krea2Maps()
        return r
    }

    /// B of a Mac of `gib` GiB of RAM, through the developer's simulation (`SILICONED_SIMULATED_RAM`).
    private func ceiling(_ gib: Int) -> Int {
        MemoryBudget.machineCeiling(environment: [MemoryBudget.simulatedRAMVariable: "\(gib)G"])
    }

    /// The verdict of one cell: the family's own format rules first (Anima's 2048 px), then B.
    private func verdict(_ card: ModelCard, _ w: Int, _ h: Int, ram: Int) -> String {
        if (try? Format.check(width: w, height: h, family: card.family)) == nil { return "side" }
        return MemoryPlan.verdict(request(card, w, h), ceiling: ceiling(ram)).label
    }

    private func floor(_ card: ModelCard, _ w: Int, _ h: Int) -> Int {
        MemoryNeed.of(request(card, w, h).lean)!.peak
    }

    /// The table, one line per family and format: the floor, the phase that sets it, and each RAM's verdict.
    private func table() -> [String] {
        var lines: [String] = []
        for card in Self.cards {
            for (w, h) in Self.formats {
                guard (try? Format.check(width: w, height: h, family: card.family)) != nil else {
                    lines.append("\(card.id) \(Format.label(width: w, height: h)): side > \(Format.maximumSide(for: card.family)) px")
                    continue
                }
                let need = MemoryNeed.of(request(card, w, h).lean)!
                let phase = need.phases.max { $0.bytes < $1.bytes }!.kind.rawValue
                let cells = Self.rams.map { "\($0)G \(verdict(card, w, h, ram: $0))" }.joined(separator: " | ")
                lines.append(String(format: "%@ %@: floor %.2f GB (%@) → %@ · %@", card.id as NSString,
                                    Format.label(width: w, height: h) as NSString, Double(need.peak) / 1e9, phase as NSString,
                                    MemoryBudget.macNeeded(for: need.peak) as NSNumber, cells as NSString))
            }
        }
        return lines
    }

    func testThePrintedTable() {
        let lines = table()
        if ProcessInfo.processInfo.environment["PRINT_FORMAT_TABLE"] != nil { lines.forEach { print($0) } }
        XCTAssertEqual(lines.count, Self.cards.count * Self.formats.count)
        // `PRINT_FORMAT_TERMS=1`: every phase of the largest formats, term by term (the pencil figures).
        if ProcessInfo.processInfo.environment["PRINT_FORMAT_TERMS"] != nil {
            for card in Self.cards {
                for (w, h) in [(2048, 2048), (2048, 3072)] where (try? Format.check(width: w, height: h, family: card.family)) != nil {
                    let need = MemoryNeed.of(request(card, w, h).lean)!
                    print("== \(card.id) \(Format.label(width: w, height: h)): peak \(need.peak / 1_000_000) MB, crossing \(need.crossing / 1_000_000) MB")
                    for phase in need.phases {
                        print("  \(phase.kind.rawValue) \(phase.bytes / 1_000_000) MB (own \(phase.own / 1_000_000), outside \(phase.outside / 1_000_000))")
                        for t in phase.terms where t.bytes >= 20_000_000 {
                            print("      \(t.name): \(t.bytes / 1_000_000) MB\(t.outside ? " outside" : "")")
                        }
                    }
                }
            }
        }
    }

    /// **The table, pinned**: the smallest Mac sold (GB) whose B holds each family's format, process base
    /// aside (the app's own footprint, a few hundred MB, comes on top). A change of a formula moves a cell
    /// here, and says so. `nil`: the family's own side bound (Anima, 2048 px).
    func testTheMinimumMacOfEachCell() {
        let expected: [String: [Int?]] = [
            // 1024², 1024×1536, 1536², 2048², 2048×3072
            "z-image": [8, 8, 8, 16, 18],
            "qwen-image-2.1": [8, 8, 8, 8, 16],
            "anima": [8, 8, 8, 16, nil],
            "krea2": [8, 8, 8, 16, 16],
        ]
        for card in Self.cards {
            let got = Self.formats.map { (w, h) -> Int? in
                (try? Format.check(width: w, height: h, family: card.family)) == nil ? nil
                    : MemoryBudget.macNeeded(for: floor(card, w, h))
            }
            XCTAssertEqual(got, expected[card.id]!, card.id)
        }
    }

    /// **B decides, and nothing else**: a cell is a plan exactly when the lean plan's floor is within the
    /// RAM's B, a refusal names the smallest Mac sold whose B holds it, and the area has no ceiling.
    func testBDecidesEveryCell() {
        for card in Self.cards {
            for (w, h) in Self.formats where (try? Format.check(width: w, height: h, family: card.family)) != nil {
                let floor = self.floor(card, w, h)
                for ram in Self.rams {
                    switch MemoryPlan.verdict(request(card, w, h), ceiling: ceiling(ram)) {
                    case let .plan(planFloor, need, _, _, _, _):
                        XCTAssertEqual(planFloor, floor)
                        XCTAssertLessThanOrEqual(floor, ceiling(ram), "\(card.id) \(w)×\(h) on \(ram) GB")
                        XCTAssertLessThanOrEqual(need, ceiling(ram), "\(card.id) \(w)×\(h) on \(ram) GB: the plan stays in B")
                        XCTAssertGreaterThanOrEqual(need, floor)
                    case let .refused(needed, mac):
                        XCTAssertEqual(needed, floor)
                        XCTAssertGreaterThan(floor, ceiling(ram), "\(card.id) \(w)×\(h) on \(ram) GB")
                        XCTAssertGreaterThan(mac, ram, "\(card.id) \(w)×\(h): the Mac named is bigger than this one")
                        XCTAssertGreaterThanOrEqual(MemoryBudget.ceiling(physical: mac << 30), floor)
                    case .unaddressable, .unknown:
                        XCTFail("\(card.id) \(w)×\(h) on \(ram) GB: \(MemoryPlan.verdict(request(card, w, h), ceiling: ceiling(ram)))")
                    }
                }
            }
        }
    }

    /// The floor never falls as the area grows (same family, formats in order of area), and a larger RAM
    /// never refuses what a smaller one takes.
    func testTheFloorGrowsWithTheAreaAndTheRAMOnlyHelps() {
        for card in Self.cards {
            let sizes = Self.formats.filter { (try? Format.check(width: $0.0, height: $0.1, family: card.family)) != nil }
            let floors = sizes.map { floor(card, $0.0, $0.1) }
            XCTAssertEqual(floors, floors.sorted(), card.id)
            for (w, h) in sizes {
                let cells = Self.rams.map { MemoryPlan.verdict(request(card, w, h), ceiling: ceiling($0)) }
                var seenPlan = false
                for cell in cells {
                    if case .plan = cell { seenPlan = true } else { XCTAssertFalse(seenPlan, "\(card.id) \(w)×\(h): a refusal after a plan") }
                }
            }
        }
    }

    /// **The widest tensor the engine can index** (`MemoryNeed.limit`): Z-Image's decoder holds 256
    /// channels at the image's resolution, so 8 388 607 pixels at most whatever the RAM — 2048×3072 passes,
    /// 2048×4096 and 3072² are refused, and a 1 TiB Mac does not change it.
    func testTheEngineLimitIsNoMacsMemory() {
        XCTAssertNil(MemoryNeed.of(request(.zImage, 2048, 3072).lean)!.addressableArea(width: 2048, height: 3072))
        for (w, h) in [(2048, 4096), (3072, 3072)] {
            let verdict = MemoryPlan.verdict(request(.zImage, w, h), ceiling: ceiling(1024))
            XCTAssertEqual(verdict, .unaddressable(area: 8_388_607), "\(w)×\(h)")
            XCTAssertThrowsError(try ModelCard.zImage.checkMemory(request(.zImage, w, h), ceiling: ceiling(1024))) {
                XCTAssertEqual($0 as? EngineError, .formatRefused(width: w, height: h, reason: .tooLarge, bound: 8_388_607))
            }
        }
        // Below it, B is the only judge: 2048×3072 on a 16 GB Mac says which Mac it needs.
        XCTAssertThrowsError(try ModelCard.zImage.checkMemory(request(.zImage, 2048, 3072), ceiling: ceiling(16))) {
            guard case let .insufficientMemory(needed, budget)? = $0 as? EngineError else { return XCTFail("\($0)") }
            XCTAssertEqual(budget, ceiling(16))
            XCTAssertGreaterThan(needed, budget)
        }
    }

    // ── the simulated RAM ───────────────────────────────────────────────────────────────────

    /// `SILICONED_SIMULATED_RAM`: a number of GiB, with or without its unit; B is the formula's for that RAM.
    func testTheSimulatedRAMGivesB() {
        for (text, gib) in [("32G", 32), ("32GB", 32), ("32GiB", 32), ("32", 32), (" 64 g ", 64), ("128G", 128), ("0,5", 0)] {
            if gib == 0 { XCTAssertNil(MemoryBudget.parseRAM(text), text); continue }
            XCTAssertEqual(MemoryBudget.parseRAM(text), gib << 30, text)
            XCTAssertEqual(ceiling(gib), MemoryBudget.ceiling(physical: gib << 30))
        }
        XCTAssertEqual(ceiling(32), 22 << 30)    // ¾ · 32 − 2 GiB
        XCTAssertEqual(ceiling(128), 94 << 30)
        for bad in ["", "G", "abc", "-4G", "0G", "inf", "nan", "2048G", "1e30"] { XCTAssertNil(MemoryBudget.parseRAM(bad), bad) }
    }

    /// The forcing of B itself wins over the simulated RAM; a simulation that does not read changes nothing.
    func testTheForcingWinsAndABadSimulationIsIgnored() {
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [MemoryBudget.forcingVariable: "4",
                                                                MemoryBudget.simulatedRAMVariable: "128G"]), 4 << 30)
        let real = MemoryBudget.ceiling(physical: Int(ProcessInfo.processInfo.physicalMemory))
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [MemoryBudget.simulatedRAMVariable: "lots"]), real)
        XCTAssertEqual(MemoryBudget.machineCeiling(environment: [:]), real)
    }

    /// **The live environment's simulation is the developer command line's alone**: until it asks
    /// (`allowSimulation`), the library — the app's — reads B from the real RAM whatever the environment says.
    func testTheAppDoesNotReadTheSimulation() {
        let before = MemoryBudget.machineCeiling()
        setenv(MemoryBudget.simulatedRAMVariable, "512G", 1)
        defer { unsetenv(MemoryBudget.simulatedRAMVariable) }
        XCTAssertFalse(MemoryBudget.isSimulated)
        XCTAssertEqual(MemoryBudget.machineCeiling(), before)
        XCTAssertNotEqual(before, ceiling(512))
    }

    // ── the formulas at the new sizes ───────────────────────────────────────────────────────

    /// **Past `Attention.maxMatrix` the SDPA runs in query slices, and the formula charges their
    /// concatenation** — for the head split's GPU heads (Z-Image, Anima) as for Krea 2's GQA: nothing at
    /// 1024², the GPU heads' output once more at 2048×3072.
    func testTheSlicesConcatenationIsCharged() {
        func terms(_ card: ModelCard, _ w: Int, _ h: Int) -> [MemoryNeed.Term] {
            MemoryNeed.of(request(card, w, h).lean)!.phases.filter { $0.kind == .denoising }.max { $0.bytes < $1.bytes }!.terms   // Krea 2 has two
        }
        for card in [ModelCard.zImage, .anima, .krea2] {
            XCTAssertTrue(terms(card, 1024, 1024).allSatisfy { !$0.name.contains("concatenation") }, card.id)
            XCTAssertTrue(terms(card, 2048, 2048).contains { $0.name.contains("concatenation") && $0.bytes > 0 }, card.id)
        }
        // The size is the GPU heads' output: rows · heads · headDim · 4 (Z-Image: 18 of 30 heads, 128 wide).
        let z = terms(.zImage, 2048, 2048).first { $0.name == "layers SDPA concatenation" }!
        let rows = 128 * 128 + DiT.capPadded(request(.zImage, 2048, 2048).promptTokens, config: DiT.Config(header: MemoryFixtures.zImageMaps().dit))
        XCTAssertEqual(z.bytes, rows * 18 * 128 * 4)
        // Qwen-Image-2.1 slices its own SDPAs (`attentionSlices`) below the bound: nothing to concatenate.
        XCTAssertTrue(terms(.qwenImage21, 2048, 3072).allSatisfy { !$0.name.contains("concatenation") })
    }

    /// The decoders' widest register, and their bytes, by pixel: linear in the area, as the formula walks them.
    func testTheDecodersScaleWithThePixels() {
        for (flux, perPixel) in [(true, 256), (false, 192)] {
            let small = flux ? VAE.plan(latentHeight: 128, latentWidth: 128) : AnimaVAE.plan(latentHeight: 128, latentWidth: 128)
            let large = flux ? VAE.plan(latentHeight: 256, latentWidth: 384) : AnimaVAE.plan(latentHeight: 256, latentWidth: 384)
            XCTAssertEqual(small.largestSlot / 4, perPixel * 1024 * 1024 + 0, "\(flux)")
            XCTAssertEqual(large.largestSlot / 4, perPixel * 2048 * 3072, "\(flux)")
        }
    }
}
