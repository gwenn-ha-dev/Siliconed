import Foundation
import Metal
import SiliCore
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

    /// The weights as surfaces: apart from the heap, their codes laid as the declaration says (the ANE reads `[out, in]`
    /// transposed in its function: `SharedWeights.integer`).
    @Test func weightsOnSurfacesLieApart() throws {
        var d = Self.declaration()
        d.surfaces = true
        d.codes = .outIn
        let plan = try Self.plan(1 << 30, d)
        #expect(plan.heap == Self.tensors + Self.staging)
        #expect(plan.apart == Self.apart + Self.resident + 2 * Self.group)
        #expect(plan.codes == .outIn)
        var e = Self.declaration()
        e.codes = .outIn
        #expect(try Self.plan(1 << 30, e).codes == .outIn)
    }

    /// The layouts are the declaration's (the plan reads no measure of this Mac): the functions are written for them
    /// (19/11: a measure kept between two reads handed the functions weights laid otherwise, and the worker stopped).
    @Test func theLayoutsAreTheDeclarations() throws {
        #expect(try Self.plan(1 << 30).codes == .inOut && Self.plan(1 << 30).bfloat16 == .outIn)
        var d = Self.declaration()
        (d.codes, d.bfloat16) = (.outIn, .inOut)
        let declared = try Self.plan(1 << 30, d)
        #expect(declared.codes == .outIn && declared.bfloat16 == .inOut)
    }
}

/// **The signal on a live stage** (the arena on this Mac's GPU, nothing called): the socle's plan, a ring of 2; a footprint
/// simulated — a base the test sets, and the arena's heaps — that leaves room under B for two groups beside the margin.
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

    /// Room for `room` groups beside the margin, the cycle run four turns, the signal after the uses in `signals`, a climb
    /// after use `climb.use` expecting `climb.expects`: the groups kept, and the first use that was no slot's.
    static func run(reserve: Int?, room: Int, signals: Set<Int>, climb: (use: Int, expects: Int)? = nil) async throws -> (kept: [String], firstKept: Int?)? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        var d = StagePlans.declaration()
        d.functions = [:]
        let socle = Arena.allocation(bytes: StagePlans.tensors + StagePlans.resident + 2 * StagePlans.group + StagePlans.staging) + StagePlans.apart
        let budget = StagePlans.floor + socle + Stage.margin
        let plan = try Stage.plan(d, files: [StagePlans.file], budget: budget, floor: StagePlans.floor, depth: 2)
        #expect(plan.kept.isEmpty && plan.depth == 2)
        var measured = ThisMac()
        measured.reserves = reserve.map { [Stage.key(d): $0] }
        let footprint = Footprint()
        let stage = try await Stage(device, plan: plan, files: [StagePlans.file], measured: measured) { footprint.bytes }
        defer { stage.close() }
        footprint.lock.withLock { (footprint.arena, footprint.start) = (stage.arena, stage.arena.allocated) }
        footprint.lock.withLock { footprint.base = budget - Stage.margin - room * Arena.allocation(bytes: StagePlans.group) }
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

    @Test func whatThisMacMeasuredIsKeptBySystem() throws {
        try ThisMac.update { $0.neuralUnits = ["block k": 1024]; $0.readings = ["stage f": 42] }
        #expect(ThisMac.current.neuralUnits?["block k"] == 1024)
        #expect(ThisMac.current.readings?["stage f"] == 42)
        let all = try JSONSerialization.jsonObject(with: Data(contentsOf: ThisMac.url)) as? [String: Any]
        #expect(all?.keys.sorted() == [ThisMac.system])
    }
}

/// **A height chosen before its first call** (`Stage.estimates`, `Stage.choose`), on readings written by hand: three
/// forms, the tallest first, of 4096, 2048 and 1024 rows, the same socle.
@Suite struct StageHeights {
    static let forms = [("tall", 4096), ("mid", 2048), ("low", 1024)].map { (key: $0.0, rows: $0.1) }
    static func measured(_ readings: [String: Int]) -> ThisMac { var m = ThisMac(); m.readings = readings; return m }
    static func plans(_ budget: Int) throws -> [Stage.Plan] {
        var d = StagePlans.declaration()
        d.functions = [:]
        let plan = try Stage.plan(d, files: [StagePlans.file], budget: budget, floor: StagePlans.floor, depth: 2)
        return [plan, plan, plan]
    }

    /// Never read: unknown, and the soberest taken.
    @Test func nothingReadTakesTheSoberest() throws {
        let e = Stage.estimates(Self.forms, measured: ThisMac())
        #expect(e == [.unknown, .unknown, .unknown])
        #expect(try Stage.choose(try Self.plans(1 << 40), e) == 2)
    }

    /// One read: the others scaled by their rows from it; the tallest that holds taken.
    @Test func aReadingScalesToTheOthers() throws {
        let e = Stage.estimates(Self.forms, measured: Self.measured(["low": 100 << 20]))
        #expect(e == [.scaled(400 << 20), .scaled(200 << 20), .read(100 << 20)])
        let socle = try Self.plans(1 << 40)[0].p
        #expect(try Stage.choose(try Self.plans(1 << 40), e) == 0)
        #expect(try Stage.choose(try Self.plans(socle + Stage.margin + (300 << 20)), e) == 1)
    }

    /// Several read: a form's own reading; the others from the tallest read.
    @Test func itsOwnReadingFirstThenTheTallestRead() throws {
        let e = Stage.estimates(Self.forms, measured: Self.measured(["mid": 300 << 20, "low": 100 << 20]))
        #expect(e == [.scaled(600 << 20), .read(300 << 20), .read(100 << 20)])
    }

    /// A form the fuse stopped is never taken, and no reading of its own is scaled from.
    @Test func aFusedFormIsNeverTaken() throws {
        let e = Stage.estimates(Self.forms, measured: Self.measured(["tall": -1, "low": 100 << 20]))
        #expect(e == [.fused, .scaled(200 << 20), .read(100 << 20)])
        #expect(try Stage.choose(try Self.plans(1 << 40), e) == 1)
    }

    /// None read and the soberest fused: the soberest form never tried, not a refusal.
    @Test func noneReadAndTheSoberestFusedTakesTheOneNeverTried() throws {
        let e = Stage.estimates(Self.forms, measured: Self.measured(["low": -1]))
        #expect(e == [.unknown, .unknown, .fused])
        #expect(try Stage.choose(try Self.plans(1 << 40), e) == 1)
    }

    /// Every form fused: refused as a lack of memory, the soberest's P its number (the one a client says).
    @Test func everyFormFusedIsRefusedWithTheSoberestsP() throws {
        let plans = try Self.plans(1 << 40)
        let e = Stage.estimates(Self.forms, measured: Self.measured(["tall": -1, "mid": -1, "low": -1]))
        let refusal = #expect(throws: OverBudget.self) { try Stage.choose(plans, e) }
        #expect(refusal?.needed == plans[2].p + Stage.margin && refusal?.budget == 1 << 40)
    }

    /// Known and none holding: refused, with the number.
    @Test func noFormThatHoldsIsRefused() throws {
        let socle = try Self.plans(1 << 40)[0].p
        let e = Stage.estimates(Self.forms, measured: Self.measured(["low": 100 << 20]))
        let budget = socle + Stage.margin + (50 << 20)
        let refusal = #expect(throws: OverBudget.self) { try Stage.choose(try Self.plans(budget), e) }
        // The soberest form, its estimate and the margin counted: 100 MiB read, 50 short.
        #expect(refusal?.needed == socle + (100 << 20) + Stage.margin && refusal?.budget == budget)
    }
}
