import Foundation
import Metal
@testable import SiliCore
import Testing

/// **A stage's plan, on declarations written by hand**: a file of five groups that lay out alike (one used once, four in
/// the cycle), a resident, tensors in the heap, apart and on a surface, two functions (P counts no share of theirs).
/// Every number below is counted by hand from the pages each tensor takes (16 KiB on Apple silicon).
@Suite struct StagePlans {
    static let page = 16384
    static let floor = 100 << 20
    /// A group: one f32 `[1024, 16]` read as f16 (32 KiB in the arena, 64 KiB of staging); the resident f32 `[4096]`.
    static let file: SafeTensors = {
        let group = [UInt8](repeating: 0, count: 1024 * 16 * 4)
        var tensors: [(String, String, [Int], [UInt8])] = ["first", "g0", "g1", "g2", "g3"].map { ("\($0).w", "F32", [1024, 16], group) }
        tensors.append(("r", "F32", [4096], [UInt8](repeating: 0, count: 4096 * 4)))
        tensors.append(("odd.w", "F32", [512, 16], [UInt8](repeating: 0, count: 512 * 16 * 4)))
        return try! SafeTensors(try! safetensors("stage.safetensors", tensors))
    }()
    static func declaration(_ name: String = "test", tensor: Int = 100_000) -> Stage.Declaration {
        var d = Stage.Declaration(name)
        d.tensors = [Stage.Tensor("x", bytes: tensor), Stage.Tensor("out", bytes: 1000, .apart), Stage.Tensor("s", surface: [8, 64], element: 2)]
        d.residents = [("r", "r", .float)]
        for g in ["first", "g0", "g1", "g2", "g3"] { d.groups[g] = [("w", "\(g).w", .half)] }
        d.slot = "g0"
        d.once = ["first"]
        d.cycle = ["g0", "g1", "g2", "g3"]
        d.functions = ["f": Stage.Function("test-f") { throw Failure("not built") },
                       "g": Stage.Function("test-g", unit: .neuralEngine) { throw Failure("not built") }]
        return d
    }

    /// The heap and apart, counted by hand: x 100 000 B in 7 pages, the resident 1, each group 2, the staging 4; apart,
    /// the exit's page and the surface's.
    static let tensors = 7 * page, resident = page, group = 2 * page, staging = 4 * page, apart = 2 * page

    static func plan(_ budget: Int, _ d: Stage.Declaration = declaration()) throws -> Stage.Plan {
        try Stage.plan(d, files: [file], budget: budget, floor: floor, depth: 2)
    }

    /// P with `kept` groups and a ring of `depth`, by hand.
    static func p(kept: Int, depth: Int) -> Int {
        floor + Arena.allocation(bytes: tensors + resident + (kept + depth) * group + staging) + apart
    }

    /// **The socle alone, whatever B**: the groups that ride ride the ring from the start (what of the cycle is kept is
    /// decided in flight, `Stage.signal`).
    @Test func theRingIsThereWhateverB() throws {
        for budget in [1 << 40, Self.p(kept: 0, depth: 2) + Stage.margin] {
            let plan = try Self.plan(budget)
            #expect(plan.kept.isEmpty && plan.depth == 2)
        }
    }

    /// P is the exact sum of its terms: the floor, the heap as Metal allocates it, what is apart.
    @Test func pIsTheSumOfItsTerms() throws {
        let plan = try Self.plan(1 << 30)
        #expect(plan.heap == Self.tensors + Self.resident + 2 * Self.group + Self.staging)
        #expect(plan.apart == Self.apart)
        #expect(plan.p == Self.floor + Arena.allocation(bytes: plan.heap) + plan.apart)
        #expect(plan.p == Self.p(kept: 0, depth: 2))
    }

    /// No group rides (no slots' group): every group kept, no ring.
    @Test func whereNoGroupRidesEveryGroupIsKept() throws {
        var d = Self.declaration()
        d.slot = nil
        let plan = try Self.plan(1 << 30, d)
        #expect(plan.kept == ["first", "g0", "g1", "g2", "g3"] && plan.depth == 0)
        #expect(plan.p == Self.p(kept: 5, depth: 0))
    }

    /// B under the finest cut: refused, with the number.
    @Test func aBudgetUnderTheFinestCutIsRefusedWithTheNumber() throws {
        let finest = Self.p(kept: 0, depth: 2)
        #expect(try Self.plan(finest + Stage.margin).kept.isEmpty)
        do {
            _ = try Self.plan(finest + Stage.margin - (1 << 20))
            Issue.record("planned under the finest cut")
        } catch let f as OverBudget {
            #expect(f.description.contains("by 1.0 MiB"), Comment(rawValue: f.description))
            #expect(f.description.contains(Memory.mib(finest)), Comment(rawValue: f.description))
            #expect(f.needed == finest + Stage.margin && f.budget == finest + Stage.margin - (1 << 20))
        }
    }

    /// A group that does not lay out as the slots' is kept whatever B: the ring cannot hold it.
    @Test func aGroupTheRingCannotHoldIsKept() throws {
        var d = Self.declaration()
        d.groups["odd"] = [("w", "odd.w", .half)]
        d.cycle.append("odd")
        let odd = Self.page  // [512, 16] in f16: 16 KiB
        let finest = Self.floor + Arena.allocation(bytes: Self.tensors + Self.resident + odd + 2 * Self.group + Self.staging) + Self.apart
        let plan = try Self.plan(finest + Stage.margin, d)
        #expect(plan.kept == ["odd"])
        #expect(plan.p == finest)
    }

    /// The weights as surfaces: apart from the heap.
    @Test func weightsOnSurfacesLieApart() throws {
        var d = Self.declaration()
        d.surfaces = true
        let plan = try Self.plan(1 << 30, d)
        #expect(plan.heap == Self.tensors + Self.staging)
        #expect(plan.apart == Self.apart + Self.resident + 2 * Self.group)
    }
}

/// **The signal on a live stage** (the arena on this Mac's GPU, nothing called): the socle's plan, a ring of 2; a footprint
/// simulated — a base the test sets, and the arena's heaps — that leaves room under B for two groups beside what kept groups
/// leave clear (`Stage.keepClear`).
/// The bars, counted by hand: past the signal, two groups kept, the others the ring's; a reserve of a group, one; a
/// reserve never read, nothing kept before the cycle's third turn, then two; past a climb that expects a group's growth,
/// nothing kept before two turns past it, then one.
@Suite struct StageSignals {
    /// The simulated footprint: `base` + what the arena allocated past what it had at the start.
    final class Footprint: @unchecked Sendable {
        let lock = NSLock()
        var base = 0, arena: Arena?, start = 0
        var bytes: Int { lock.withLock { base + (arena.map { $0.allocated - start } ?? 0) } }
    }

    /// Room for `room` groups beside what kept groups leave clear, the cycle run four turns, the signal after the uses in `signals`, a climb
    /// after use `climb.use` expecting `climb.expects`: the groups kept, and the first use that was no slot's.
    static func run(reserve: Int?, room: Int, signals: Set<Int>, climb: (use: Int, expects: Int)? = nil) async throws -> (kept: [String], firstKept: Int?)? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        var d = StagePlans.declaration()
        d.functions = [:]
        let socle = Arena.allocation(bytes: StagePlans.tensors + StagePlans.resident + 2 * StagePlans.group + StagePlans.staging) + StagePlans.apart
        let budget = StagePlans.floor + socle + Stage.keepClear
        let plan = try Stage.plan(d, files: [StagePlans.file], budget: budget, floor: StagePlans.floor, depth: 2)
        #expect(plan.kept.isEmpty && plan.depth == 2)
        var measured = ThisMac()
        measured.reserves = reserve.map { [Stage.key(d): $0] }
        let footprint = Footprint()
        let stage = try await Stage(device, plan: plan, files: [StagePlans.file], measured: measured) { footprint.bytes }
        defer { stage.close() }
        footprint.lock.withLock { (footprint.arena, footprint.start) = (stage.arena, stage.arena.allocated) }
        footprint.lock.withLock { footprint.base = budget - Stage.keepClear - room * Arena.allocation(bytes: StagePlans.group) }
        var firstKept: Int?
        for (m, g) in (["first"] + Array(repeating: d.cycle, count: 4).flatMap { $0 }).enumerated() {
            let held = try stage.acquire(g)
            if held.use == nil && firstKept == nil { firstKept = m }
            stage.release(held)
            if let climb, climb.use == m { #expect(stage.rekey("stage climbed", expects: climb.expects)) }
            if signals.contains(m) { stage.signal() }
        }
        return (stage.kept, firstKept)
    }

    @Test func aSignalKeepsWhatHoldsUnderB() async throws {
        guard let (kept, first) = try await Self.run(reserve: 0, room: 2, signals: [2]) else { return }
        #expect(kept.count == 2, "\(kept)")
        // The reader two uses ahead of the loop at the signal: the first it may keep is the fifth's.
        #expect(first.map { $0 >= 3 } ?? false, "\(String(describing: first))")
    }

    @Test func aReserveOfAGroupKeepsOneLess() async throws {
        guard let (kept, _) = try await Self.run(reserve: Arena.allocation(bytes: StagePlans.group), room: 2, signals: [2]) else { return }
        #expect(kept.count == 1, "\(kept)")
    }

    /// The reserve never read: the signals of the first two turns keep nothing; the third turn's, as the reserve 0.
    @Test func aReserveNeverReadWaitsForTheThirdTurn() async throws {
        guard let (none, _) = try await Self.run(reserve: nil, room: 2, signals: [2, 6]) else { return }
        #expect(none.isEmpty, "\(none)")
        guard let (kept, first) = try await Self.run(reserve: nil, room: 2, signals: [2, 6, 10]) else { return }
        #expect(kept.count == 2, "\(kept)")
        #expect(first.map { $0 >= 11 } ?? false, "\(String(describing: first))")
    }

    /// A climb in the first turn, its new functions never read, expecting a group's growth: the next turn's signal keeps
    /// nothing; the one after keeps one group, the growth counted as still to come.
    @Test func aClimbWaitsTwoTurnsAndCountsWhatItExpects() async throws {
        let group = Arena.allocation(bytes: StagePlans.group)
        guard let (none, _) = try await Self.run(reserve: 0, room: 2, signals: [6], climb: (1, group)) else { return }
        #expect(none.isEmpty, "\(none)")
        guard let (kept, _) = try await Self.run(reserve: 0, room: 2, signals: [6, 10], climb: (1, group)) else { return }
        #expect(kept.count == 1, "\(kept)")
    }
}

/// **What this Mac measured, in its file** (`ThisMac`): two writers at once, the system a measure is kept under.
@Suite(.serialized) struct ThisMacFile {
    init() {
        ThisMac.file = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../.build/tests/this-mac.json").standardizedFileURL
        try? FileManager.default.removeItem(at: ThisMac.file!)
    }

    /// **A stage of bands never read on this Mac is read ahead** (`BandStage.prepare`, `read`): every candidate unknown,
    /// what its opening takes the soberest for; once the soberest was read, none is.
    @Test func aStageNeverReadIsReadAhead() throws {
        let c = try BandStage.candidates("t", prefix: "bands-read", candidates: [1 << 12, 1 << 11], files: [], residents: [],
                                         budget: 1 << 40) { BandPlans.plan(rows: 100, positions: $0).0 }
        #expect(BandStage.unread(c))
        let soberest = c.socles.last!.declaration
        try ThisMac.update {
            $0.readings = [Stage.key(soberest): 1 << 20]
            $0.extents = [Stage.key(soberest): soberest.extent!]
        }
        #expect(!BandStage.unread(c))
    }

    /// Two writers at once, each its own opening of the file as two processes have: every measure of both is kept
    /// (each reads, changes and writes under the file's lock).
    @Test func twoWritersAtOnceLoseNothing() throws {
        DispatchQueue.concurrentPerform(iterations: 2) { w in
            for i in 0..<50 { try! ThisMac.update { $0.readings = ($0.readings ?? [:]).merging(["writer \(w) \(i)": i]) { $1 } } }
        }
        let shares = ThisMac.current.readings ?? [:]
        let lost = (0..<100).filter { shares["writer \($0 / 50) \($0 % 50)"] != $0 % 50 }
        #expect(lost.isEmpty, "\(lost.count) of 100 lost")
    }

    /// **A stage closed with no call keeps no reading** (a render cancelled while its functions load, a refusal past
    /// the opening): its key never read stays so, rekeyed or closed — what nothing called holds is no reading.
    @Test func aStageNeverCalledKeepsNoReading() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        var d = StagePlans.declaration("never called")
        d.functions = [:]
        d.inFlight = 3
        let plan = try Stage.plan(d, files: [StagePlans.file], budget: 1 << 40, floor: StagePlans.floor, depth: 2)
        let footprint = StageSignals.Footprint()
        let stage = try await Stage(device, plan: plan, files: [StagePlans.file], measured: ThisMac()) { footprint.bytes }
        // Core AI's share read 10 MiB: the footprint past the floor and the arena.
        let allocated = stage.arena.allocated
        footprint.lock.withLock { (footprint.arena, footprint.start, footprint.base) = (stage.arena, allocated, StagePlans.floor + allocated + (10 << 20)) }
        #expect(stage.rekey("never called, climbed", expects: 0))
        stage.close()
        #expect(ThisMac.current.reading(Stage.key(d)) == nil && ThisMac.current.reading("never called, climbed") == nil)
        #expect(ThisMac.current.reserves?[Stage.key(d)] == nil)
    }

    /// **A stage called keeps its reading**, its key's from its opening to a climb; the climb's key, its functions never
    /// called, none.
    @Test func aStageCalledKeepsItsReading() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        var d = StagePlans.declaration("called")
        d.functions = [:]
        d.inFlight = 4
        let plan = try Stage.plan(d, files: [StagePlans.file], budget: 1 << 40, floor: StagePlans.floor, depth: 2)
        let footprint = StageSignals.Footprint()
        let stage = try await Stage(device, plan: plan, files: [StagePlans.file], measured: ThisMac()) { footprint.bytes }
        // Core AI's share read 10 MiB: the footprint past the floor and the arena.
        let allocated = stage.arena.allocated
        footprint.lock.withLock { (footprint.arena, footprint.start, footprint.base) = (stage.arena, allocated, StagePlans.floor + allocated + (10 << 20)) }
        Stage.call(on: stage.stream)
        #expect(stage.rekey("called, climbed", expects: 0))
        stage.close()
        #expect(ThisMac.current.reading(Stage.key(d)) == 10 << 20)
        #expect(ThisMac.current.reading("called, climbed") == nil)
    }

    /// **A stage planned on what the stages before it let go keeps no first reading** (`Stage.Opening.trailing`): read
    /// while what trails leaves, it would come out short by as much; over a reading kept, at its largest.
    @Test func aStageOnWhatTrailsKeepsNoFirstReading() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        var d = StagePlans.declaration("on what trails")
        d.functions = [:]
        d.inFlight = 5
        let plan = try Stage.plan(d, files: [StagePlans.file], budget: 1 << 40, floor: StagePlans.floor, depth: 2)
        var opening = Stage.Opening()
        opening.trailing = true
        func run(share: Int) async throws {
            let footprint = StageSignals.Footprint()
            let stage = try await Stage(device, plan: plan, files: [StagePlans.file], measured: ThisMac(), opening: opening) { footprint.bytes }
            let allocated = stage.arena.allocated
            footprint.lock.withLock { (footprint.arena, footprint.start, footprint.base) = (stage.arena, allocated, StagePlans.floor + allocated + share) }
            Stage.call(on: stage.stream)
            stage.close()
        }
        try await run(share: 10 << 20)
        #expect(ThisMac.current.reading(Stage.key(d)) == nil)
        try ThisMac.update { $0.readings = ($0.readings ?? [:]).merging([Stage.key(d): 8 << 20]) { $1 } }
        try await run(share: 10 << 20)
        #expect(ThisMac.current.reading(Stage.key(d)) == 10 << 20)
        try await run(share: 6 << 20)
        #expect(ThisMac.current.reading(Stage.key(d)) == 10 << 20)
    }

    /// **What a stage planned on what trails waits for** (`Stage.waits`): nothing where it takes the same functions as
    /// on the rest and these were read on this Mac; else the wait — another form on the rest, never read, fused.
    @Test func aStageOnWhatTrailsWaitsOnlyWhereItChangesSomething() throws {
        var d = StagePlans.declaration("waits")
        d.functions = [:]
        let socle = try Stage.socle(d, files: [StagePlans.file], budget: 1 << 40)
        let (now, rested) = (try socle.on(floor: StagePlans.floor + (300 << 20)), try socle.on(floor: StagePlans.floor))
        let other = try Stage.plan(StagePlans.declaration("taller"), files: [StagePlans.file], budget: 1 << 40, floor: StagePlans.floor)
        var read = ThisMac(), fused = ThisMac()
        read.readings = [Stage.key(d): 10 << 20]
        (fused.readings, fused.fusedBy) = ([Stage.key(d): -1], [Stage.key(d): ThisMac.engine])
        #expect(!Stage.waits(now: now, rested: rested, measured: read))
        #expect(Stage.waits(now: now, rested: other, measured: read))
        #expect(Stage.waits(now: now, rested: rested, measured: ThisMac()))
        #expect(Stage.waits(now: now, rested: rested, measured: fused))
    }

    @Test func whatThisMacMeasuredIsKeptBySystem() throws {
        try ThisMac.update { $0.capacities = ["codes": .init(ratio: 0.37)]; $0.readings = ["stage f": 42] }
        #expect(ThisMac.current.capacities?["codes"] == ThisMac.Capacity(ratio: 0.37))
        #expect(ThisMac.current.readings?["stage f"] == 42)
        let all = try JSONSerialization.jsonObject(with: Data(contentsOf: ThisMac.url)) as? [String: Any]
        #expect(all?.keys.sorted() == [ThisMac.system])
    }
}

/// **A height chosen before its first call** (`Stage.estimates`, `Stage.choose`), on readings written by hand: three
/// forms, the tallest first, whose calls hold 4096, 2048 and 1024 rows' elements, of one kin, the same socle.
@Suite struct StageHeights {
    static func extent(_ rows: Int) -> Stage.Extent { Stage.Extent(kin: "kin f", elements: ["f results": rows * 64, "f largest": rows * 8]) }
    static let forms = [("tall", 4096), ("mid", 2048), ("low", 1024)].map { (key: $0.0, extent: Optional(extent($0.1))) }
    /// The readings, each −1 this engine's mark, each read beside its extent.
    static func measured(_ readings: [String: Int]) -> ThisMac {
        var m = ThisMac()
        m.readings = readings
        m.extents = Dictionary(uniqueKeysWithValues: forms.filter { readings[$0.key] != nil }.map { ($0.key, $0.extent!) })
        m.fusedBy = readings.filter { $0.value == -1 }.mapValues { _ in ThisMac.engine }
        return m
    }
    /// A reading times its ratio, by the swing, the margin beside.
    static func bound(_ bytes: Int) -> Int { StageExtents.bound(bytes) }
    static func plans(_ budget: Int) throws -> [Stage.Plan] {
        var d = StagePlans.declaration()
        d.functions = [:]
        let plan = try Stage.plan(d, files: [StagePlans.file], budget: budget, floor: StagePlans.floor, depth: 2)
        return [plan, plan, plan]
    }

    /// Never read: unknown, and the soberest taken.
    @Test func nothingReadTakesTheSoberest() throws {
        let e = Stage.estimates(extents: Self.forms, measured: ThisMac())
        #expect(e == [.unknown, .unknown, .unknown])
        #expect(try Stage.choose(try Self.plans(1 << 40), e) == 2)
    }

    /// One read: the taller bounded from it by their ratio, the swing and the margin; the tallest that holds taken.
    @Test func aReadingBoundsTheTaller() throws {
        let e = Stage.estimates(extents: Self.forms, measured: Self.measured(["low": 100 << 20]))
        #expect(e == [.scaled(Self.bound(400 << 20)), .scaled(Self.bound(200 << 20)), .read(100 << 20)])
        let socle = try Self.plans(1 << 40)[0].p
        #expect(try Stage.choose(try Self.plans(1 << 40), e) == 0)
        #expect(try Stage.choose(try Self.plans(socle + Stage.margin + Self.bound(300 << 20)), e) == 1)
    }

    /// **A shorter form is never estimated below a taller one's reading** (22/11/2026: the rows' scaling gave a DiT's
    /// calls of 2112 rows 987 MiB from 1943 read over a call a pass, and they held 1589): bounded by the reading itself.
    @Test func aShorterFormIsNeverBelowATallerReading() throws {
        let e = Stage.estimates(extents: Self.forms, measured: Self.measured(["tall": 400 << 20]))
        #expect(e == [.read(400 << 20), .capped(Self.bound(400 << 20)), .capped(Self.bound(400 << 20))])
        // The tallest read does not hold, nor the bound of the others: B lowered under every form read, the soberest
        // never read goes as on a Mac with no reading, not a refusal (decided 22/11/2026).
        let socle = try Self.plans(1 << 40)[0].p, budget = socle + Stage.margin + (300 << 20)
        #expect(try Stage.choose(try Self.plans(budget), e) == 2)
    }

    /// **The soberest read does not hold**: a taller form never read holds more — refused, the soberest read its number.
    @Test func noFormPastTheSoberestReadIsRefused() throws {
        let e = Stage.estimates(extents: Self.forms, measured: Self.measured(["mid": 200 << 20, "low": 100 << 20]))
        #expect(e == [.scaled(Self.bound(400 << 20)), .read(200 << 20), .read(100 << 20)])
        let socle = try Self.plans(1 << 40)[0].p, budget = socle + Stage.margin + (50 << 20)
        let refusal = #expect(throws: OverBudget.self) { try Stage.choose(try Self.plans(budget), e) }
        #expect(refusal?.needed == socle + (100 << 20) + Stage.margin)
    }

    /// The readings of `forms` beside one of their kin at another size (`other`, calls of `rows` rows' elements).
    static func measured(_ readings: [String: Int], other: (bytes: Int, rows: Int)) -> ThisMac {
        var m = measured(readings)
        m.readings = (m.readings ?? [:]).merging(["other size": other.bytes]) { $1 }
        m.extents = (m.extents ?? [:]).merging(["other size": extent(other.rows)]) { $1 }
        return m
    }

    /// **A bound from a form that holds no more** (`scaled`, a reading at another size times the ratio): a prediction —
    /// past B, refused before the call, never taken on paper (the soberest's at a size larger than any read).
    @Test func anExtrapolatedBoundPastBIsRefused() throws {
        let e = Stage.estimates(extents: Self.forms, measured: Self.measured([:], other: (50 << 20, 512)))
        #expect(e == [.scaled(Self.bound(400 << 20)), .scaled(Self.bound(200 << 20)), .scaled(Self.bound(100 << 20))])
        let socle = try Self.plans(1 << 40)[0].p, budget = socle + Stage.margin + Self.bound(100 << 20) - (1 << 20)
        let refusal = #expect(throws: OverBudget.self) { try Stage.choose(try Self.plans(budget), e) }
        #expect(refusal?.needed == socle + Self.bound(100 << 20) + Stage.margin && refusal?.budget == budget)
    }

    /// **A sober form fused, the taller ones bounded from another size**: a form taller than one fused holds more than
    /// it, and is never taken on paper — refused, the soberest's P its number.
    @Test func aSoberFusedFormNeverTakesTheOneAbove() throws {
        let e = Stage.estimates(extents: Self.forms, measured: Self.measured(["low": -1], other: (50 << 20, 512)))
        #expect(e == [.scaled(Self.bound(400 << 20)), .scaled(Self.bound(200 << 20)), .fused])
        let socle = try Self.plans(1 << 40)[0].p, budget = socle + Stage.margin + (50 << 20)
        let refusal = #expect(throws: OverBudget.self) { try Stage.choose(try Self.plans(budget), e) }
        #expect(refusal?.needed == socle + Stage.margin && refusal?.budget == budget)
    }

    /// Several read: a form's own reading; the others from the least of the bounds their kin's readings give.
    @Test func itsOwnReadingFirstThenTheLeastBound() throws {
        let e = Stage.estimates(extents: Self.forms, measured: Self.measured(["mid": 300 << 20, "low": 100 << 20]))
        #expect(e == [.scaled(Self.bound(400 << 20)), .read(300 << 20), .read(100 << 20)])
    }

    /// A form with no extent (a stage of one form): its own reading, or unknown — no other reading bounds it.
    @Test func aFormWithNoExtentIsItsReadingOrUnknown() {
        let m = Self.measured(["low": 100 << 20])
        #expect(Stage.estimates(extents: [("low", nil), ("other", nil)], measured: m) == [.read(100 << 20), .unknown])
    }

    /// A form the fuse stopped is never taken, and no reading of its own bounds another.
    @Test func aFusedFormIsNeverTaken() throws {
        let e = Stage.estimates(extents: Self.forms, measured: Self.measured(["tall": -1, "low": 100 << 20]))
        #expect(e == [.fused, .scaled(Self.bound(200 << 20)), .read(100 << 20)])
        #expect(try Stage.choose(try Self.plans(1 << 40), e) == 1)
    }

    /// **A mark another engine took is no reading** (decided 21/11/2026): the form tried once more, as never read — the
    /// mark of a file written before any engine was named alike. Fused again, the mark is this engine's, and holds;
    /// read, the reading replaces it.
    @Test func anotherEnginesMarkIsTriedOnceMore() throws {
        var m = Self.measured(["tall": -1, "mid": -1, "low": 100 << 20])
        m.fusedBy = ["tall": "another engine"]
        let e = Stage.estimates(extents: Self.forms, measured: m)
        #expect(e == [.scaled(Self.bound(400 << 20)), .scaled(Self.bound(200 << 20)), .read(100 << 20)])
        #expect(try Stage.choose(try Self.plans(1 << 40), e) == 0)

        m.fuse("tall")
        #expect(Stage.estimates(extents: Self.forms, measured: m)[0] == .fused)
        m.keep(reading: 50 << 20, under: "tall")
        #expect(m.reading("tall") == -1, "this engine's mark outlives a reading")

        m.keep(reading: 300 << 20, under: "mid")
        #expect(m.reading("mid") == 300 << 20 && m.fusedBy?["mid"] == nil)
        m.fuse("tall", engine: "an older engine")
        m.keep(reading: 500 << 20, under: "tall")
        #expect(m.reading("tall") == 500 << 20 && m.fusedBy?["tall"] == nil)
    }

    /// **A mark holds under the B it was taken under, not a larger one** (decided 22/11/2026): the one thing kept that
    /// depends on B. Under a larger B the form is tried once more, as never read; read there, the reading replaces the
    /// mark; under the same or a smaller B, the mark holds and a reading leaves it. A mark with no B (a file written
    /// before) holds under any.
    @Test func aMarkHoldsUnderTheBItWasTakenUnder() throws {
        var m = Self.measured(["low": 100 << 20])
        m.fuse("tall", budget: 8 << 30)
        #expect(m.fusedUnder?["tall"] == 8 << 30)
        #expect(Stage.estimates(extents: Self.forms, measured: m, budget: 8 << 30)[0] == .fused)
        #expect(Stage.estimates(extents: Self.forms, measured: m, budget: 6 << 30)[0] == .fused)
        #expect(Stage.estimates(extents: Self.forms, measured: m)[0] == .fused, "no B: the mark holds")
        #expect(Stage.estimates(extents: Self.forms, measured: m, budget: 12 << 30)[0] != .fused)

        m.keep(reading: 50 << 20, under: "tall", budget: 8 << 30)
        #expect(m.reading("tall") == -1, "under the same B, the mark outlives a reading")
        m.keep(reading: 500 << 20, under: "tall", budget: 12 << 30)
        #expect(m.reading("tall") == 500 << 20 && m.fusedBy?["tall"] == nil && m.fusedUnder?["tall"] == nil)

        m.fuse("mid")
        #expect(m.reading("mid", budget: 1 << 40) == -1, "a mark with no B holds under any")
    }

    /// The engine is named by its binary: a UUID of 32 hexadecimal digits, read once.
    @Test func theEngineIsNamedByItsBinary() {
        #expect(ThisMac.engine.count == 32 && ThisMac.engine.allSatisfy(\.isHexDigit))
    }

    /// **A reading holds for its functions launched as they were**: two calls in flight are another key (each holds
    /// its intermediates), one call the key it always was — a DiT's readings kept across the change.
    @Test func callsInFlightAreAnotherKey() {
        var d = Stage.Declaration("bands")
        d.functions["f-1"] = Stage.Function("f-1", unit: .gpu) { Aicode.Module(arguments: []) }
        let one = Stage.key(d)
        #expect(one == Stage.key([(asset: "f-1", unit: .gpu)]))
        #expect(one == "stage 788dd46466a4957e")  // sha256("gpu f-1"), the key of before
        d.inFlight = 2
        #expect(Stage.key(d) != one && Stage.key(d) == Stage.key([(asset: "f-1", unit: .gpu)], inFlight: 2))
    }

    /// None read and the soberest fused: the forms above it never tried hold more than it — refused, the soberest's P
    /// its number.
    @Test func noneReadAndTheSoberestFusedIsRefused() throws {
        let plans = try Self.plans(1 << 40)
        let e = Stage.estimates(extents: Self.forms, measured: Self.measured(["low": -1]))
        #expect(e == [.unknown, .unknown, .fused])
        let refusal = #expect(throws: OverBudget.self) { try Stage.choose(plans, e) }
        #expect(refusal?.needed == plans[2].p + Stage.margin && refusal?.budget == 1 << 40)
    }

    /// Every form fused: refused as a lack of memory, the soberest's P its number (the one a client says).
    @Test func everyFormFusedIsRefusedWithTheSoberestsP() throws {
        let plans = try Self.plans(1 << 40)
        let e = Stage.estimates(extents: Self.forms, measured: Self.measured(["tall": -1, "mid": -1, "low": -1]))
        let refusal = #expect(throws: OverBudget.self) { try Stage.choose(plans, e) }
        #expect(refusal?.needed == plans[2].p + Stage.margin && refusal?.budget == 1 << 40)
    }

    /// Known and none holding: refused, with the number.
    @Test func noFormThatHoldsIsRefused() throws {
        let socle = try Self.plans(1 << 40)[0].p
        let e = Stage.estimates(extents: Self.forms, measured: Self.measured(["low": 100 << 20]))
        let budget = socle + Stage.margin + (50 << 20)
        let refusal = #expect(throws: OverBudget.self) { try Stage.choose(try Self.plans(budget), e) }
        // The soberest form, its estimate and the margin counted: 100 MiB read, 50 short.
        #expect(refusal?.needed == socle + (100 << 20) + Stage.margin && refusal?.budget == budget)
    }
}

/// **A form never read, bounded from its kin read at other shapes** (`Stage.estimates(extents:)`), on readings and
/// extents written by hand: a function `f` reading `x` and writing `y`.
@Suite struct StageExtents {
    static func extent(_ x: Int, _ y: Int, kin: String = "kin f") -> Stage.Extent { Stage.Extent(kin: kin, elements: ["f#0 x": x, "f#0 y y out": y]) }
    /// A reading times its ratio, by the swing, the margin beside.
    static func bound(_ bytes: Int) -> Int { Int((Double(bytes) * Stage.swing).rounded(.up)) + Stage.margin }
    static func measured(_ read: [String: (reading: Int, extent: Stage.Extent)]) -> ThisMac {
        var m = ThisMac()
        m.readings = read.mapValues(\.reading)
        m.extents = read.mapValues(\.extent)
        m.fusedBy = read.filter { $0.value.reading == -1 }.mapValues { _ in ThisMac.engine }
        return m
    }

    /// Its own reading first; never read, the reading of its kin times its largest ratio of elements, by the swing, the margin beside.
    @Test func aLargerFormIsTheReadingTimesItsLargestRatio() {
        let m = Self.measured(["1024": (100 << 20, Self.extent(1000, 2000))])
        #expect(Stage.estimates(extents: [("1024", Self.extent(1000, 2000))], measured: m) == [.read(100 << 20)])
        #expect(Stage.estimates(extents: [("2048", Self.extent(1500, 2200))], measured: m) == [.scaled(Self.bound(150 << 20))])
    }

    /// A form whose calls hold less is bounded by the reading itself: Core AI's share beside the calls' (their functions'
    /// code) does not shrink with them.
    @Test func aSmallerFormIsNeverBelowTheReading() {
        let m = Self.measured(["1024": (100 << 20, Self.extent(1000, 2000))])
        #expect(Stage.estimates(extents: [("512", Self.extent(500, 1000))], measured: m) == [.capped(Self.bound(100 << 20))])
    }

    /// **The least bound says its nature**: from a form that holds no more, a prediction (`scaled`); from one that holds
    /// more, at most (`capped`).
    @Test func theLeastBoundSaysItsNature() {
        let m = Self.measured(["1024": (100 << 20, Self.extent(1000, 2000)), "1536": (110 << 20, Self.extent(1500, 2400))])
        #expect(Stage.estimates(extents: [("1280", Self.extent(1250, 2000))], measured: m) == [.capped(Self.bound(110 << 20))])
        #expect(Stage.estimates(extents: [("1024 again", Self.extent(1000, 2000))], measured: m) == [.scaled(Self.bound(100 << 20))])
    }

    /// Several readings of its kin: the least of their bounds.
    @Test func theLeastOfTheBounds() {
        let m = Self.measured(["1024": (100 << 20, Self.extent(1000, 2000)), "1536": (110 << 20, Self.extent(1500, 2400))])
        #expect(Stage.estimates(extents: [("2048", Self.extent(1500, 3000))], measured: m) == [.scaled(Self.bound(110 << 20 + (110 << 20) / 4))])
        #expect(Stage.estimates(extents: [("1280", Self.extent(1200, 2400))], measured: m) == [.scaled(Self.bound(110 << 20))])
    }

    /// Another kin, other arguments, a reading the fuse marked: no bound; its own mark, never taken.
    @Test func whatIsNotItsKinBoundsNothing() {
        let m = Self.measured(["other": (100 << 20, Self.extent(1000, 2000, kin: "kin g")),
                               "fewer": (100 << 20, Stage.Extent(kin: "kin f", elements: ["f#0 x": 1000])),
                               "fused": (-1, Self.extent(1000, 2000))])
        #expect(Stage.estimates(extents: [("new", Self.extent(1000, 2000))], measured: m) == [.unknown])
        #expect(Stage.estimates(extents: [("fused", Self.extent(1000, 2000))], measured: m) == [.fused])
    }

    /// A kin names its functions apart from their order, with the calls in flight and the engine that wrote them.
    @Test func theKinNamesTheFunctionsTheirFlightAndTheEngine() {
        let a = Stage.Extent(["f#0 x f16", "f#0 y out f16"], inFlight: 2, elements: [:])
        #expect(a.kin == Stage.Extent(["f#0 y out f16", "f#0 x f16"], inFlight: 2, elements: [:]).kin)
        #expect(a.kin != Stage.Extent(["f#0 x f16", "f#0 y out f16"], inFlight: 1, elements: [:]).kin)
        #expect(a.kin != Stage.Extent(["f#0 x f32", "f#0 y out f16"], inFlight: 2, elements: [:]).kin)
    }
}

/// **A stage opened beside the one that runs** (`Stage.beside`, `Ahead`, `planned`), on a simulated footprint: the one
/// that runs, its share read at 10 MiB on this Mac and 2 MiB now; beside it, a stage of the same socle.
@Suite struct StageBesides {
    static let share = 2 << 20, reading = 10 << 20

    /// The one that runs (`inFlight` naming its key apart from every other test's), its footprint at its floor, then past
    /// its arena by `share`.
    static func running(_ device: MTLDevice, inFlight: Int, read: Bool = true) async throws -> (Stage, StageSignals.Footprint, ThisMac) {
        var d = StagePlans.declaration("running")
        d.functions = [:]
        d.inFlight = inFlight
        d.cycle = []
        d.once = ["first", "g0", "g1", "g2", "g3"]
        let plan = try Stage.plan(d, files: [StagePlans.file], budget: 1 << 40, floor: StagePlans.floor)
        var measured = ThisMac()
        if read { measured.readings = [Stage.key(d): reading] }
        let footprint = StageSignals.Footprint()
        footprint.base = StagePlans.floor
        let stage = try await Stage(device, plan: plan, files: [StagePlans.file], measured: measured) { footprint.bytes }
        let allocated = stage.arena.allocated
        footprint.lock.withLock { (footprint.arena, footprint.start, footprint.base) = (stage.arena, allocated, StagePlans.floor + allocated + share) }
        return (stage, footprint, measured)
    }

    /// The one opened beside: its socle, its key read on this Mac or not.
    static func opened(inFlight: Int, read: Bool = true) throws -> (socle: Stage.Plan, measured: ThisMac) {
        var d = StagePlans.declaration("beside")
        d.functions = [:]
        d.inFlight = inFlight
        var measured = ThisMac()
        if read { measured.readings = [Stage.key(d): 4 << 20] }
        return (try Stage.socle(d, files: [StagePlans.file], budget: 1 << 40), measured)
    }

    /// Beside it where B holds both: planned on its peak (its footprint now, its share grown to its reading), both
    /// frozen until it closes, the second thawed then.
    @Test func aStageOpensBesideTheOneThatRunsWhereBHoldsBoth() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let (running, footprint, _) = try await Self.running(device, inFlight: 71)
        let (socle, measured) = try Self.opened(inFlight: 73)
        let ahead = Stage.Ahead()
        ahead.resolve(.stage(Stage.WeakStage(running)))
        let p = try Stage.$ahead.withValue(ahead) { try Stage.planned(measured: measured) { floor in ((), try socle.on(floor: floor)) } }
        #expect(p.opening.beside != nil && p.opening.trailing)
        #expect(p.plan.floor == footprint.bytes + Self.reading - Self.share)
        #expect(running.lock.withLock { running.frozen && running.frozenShare == Self.share })
        // Frozen, it keeps nothing in flight, and reads nothing of what the other takes.
        #expect(!running.holds(expects: 0))
        footprint.lock.withLock { footprint.base += 500 << 20 }
        running.observe()
        #expect(running.lock.withLock { running.largest } == Int.min)
        let beside = try await Stage(device, plan: p.plan, files: [StagePlans.file], measured: measured, opening: p.opening) { footprint.bytes }
        #expect(beside.lock.withLock { beside.frozen })
        running.close()
        #expect(!beside.lock.withLock { beside.frozen })
        beside.close()
    }

    /// After it, as alone, where a set of functions was never read here, or where it may still keep a group in flight:
    /// the one that runs thawed, the opening waits for its close.
    @Test func aStageOpensAfterTheOneThatRunsWhereNothingWasRead() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        for (runningRead, besideRead) in [(false, true), (true, false)] {
            let (running, _, _) = try await Self.running(device, inFlight: 75, read: runningRead)
            let (socle, measured) = try Self.opened(inFlight: 77, read: besideRead)
            #expect(running.beside(measured: measured) { floor in ((), try socle.on(floor: floor)) } == nil)
            #expect(!running.lock.withLock { running.frozen })
            let ahead = Stage.Ahead()
            ahead.resolve(.stage(Stage.WeakStage(running)))
            let closing = Task { try await Task.sleep(nanoseconds: 50_000_000); running.close() }
            let p = try Stage.$ahead.withValue(ahead) { try Stage.planned(measured: measured) { floor in ((), try socle.on(floor: floor)) } }
            try await closing.value
            #expect(p.opening.beside == nil && p.opening.after >= 0.04)
        }
    }

    /// No stage comes: alone; the render fails meanwhile: the opening throws.
    @Test func anOpeningAheadGoesAloneOrStops() throws {
        let alone = Stage.Ahead()
        alone.end()
        guard case .alone = try alone.wait() else { Issue.record("not alone"); return }
        let stopped = Stage.Ahead()
        stopped.end(cancelled: true)
        #expect(throws: CancellationError.self) { try stopped.wait() }
    }
}
