import Foundation
import Testing
@testable import Siliconed

/// **The two tiled attention kernels as candidates of the map** (`tiles`, `tiles16`): the solver takes them where the machine's own lines
/// make them faster, the fp16 one only where fp16 is allowed (`Constraints.allowsHalf`: the ANE allowed and the accelerators on, at every
/// version, the Standard's included — decided 3/11/2026), the memory need charges one term
/// per buffer, and a developer can force them. No GPU, no weights, no chip name: the lines below are synthetic.
@Suite struct TiledAttentionTests {
    typealias M = CapacityModel
    typealias CS = CapacitySolver
    let key = AttentionKey(sequence: 4128, heads: 30, headDim: 128)

    func model(graph: Double = 1.5e12, tiles: Double? = nil, tiles16: Double? = nil, flash: Double? = nil) -> M {
        var m = M()
        m.lines[M.Kernel(.gpu, .fp32)] = .init(t0: 0, rate: 3e12)
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: graph)
        if let tiles { m.lines[M.Kernel(.gpu, .tiles)] = .init(t0: 0, rate: tiles) }
        if let tiles16 { m.lines[M.Kernel(.gpu, .tiles16)] = .init(t0: 0, rate: tiles16) }
        if let flash { m.lines[M.Kernel(.gpu, .flash)] = .init(t0: 0, rate: flash) }
        return m
    }

    func assignment(_ m: M, half: Bool = true, excluded: Set<M.Kernel> = [], flash: Bool = false, key: AttentionKey? = nil) throws -> CS.AttentionSolution {
        let c = CS.Constraints(sustained: false, excluded: excluded, allowsANE: half)
        return try #require(CS.attention(key ?? self.key, flash: flash, model: m, constraints: c))
    }

    // MARK: The solver

    @Test func theTilesAreTakenWhereTheMapMeasuredThemFaster() throws {
        let s = try assignment(model(tiles: 2e12))
        #expect(s.assignment == .tiles && s.route == "tiles")
        #expect(abs(s.compute - 4.0 * 4128 * 4128 * 128 * 30 / 2e12) < 1e-9)
        #expect(try assignment(model(tiles: 1.2e12)).assignment == .graph, "slower than the graph: the graph")
        #expect(try assignment(model()).assignment == .graph, "no line, no candidate")
    }

    @Test func theHalfPrecisionKernelIsOfferedOnlyWhereFP16IsAllowed() throws {
        let m = model(tiles: 2e12, tiles16: 4e12)
        #expect(try assignment(m, half: true).assignment == .tiles16)
        #expect(try assignment(m, half: false).assignment == .tiles, "no ANE (a map that gives it nothing): the fp32 tiles, never the fp16")
        // the same through a request, as the planner builds the constraints: every version, the Standard's included (3/11/2026)
        for variant in Variant.allCases {
            let c = CS.Constraints(request: .init(family: .zImage, variant: variant, includesEncoder: false), sustained: false)
            #expect(c.allowsHalf, "\(variant)")
            #expect(CS.attention(key, flash: false, model: m, constraints: c)?.assignment == .tiles16, "\(variant)")
        }
        // the one rule: the ANE allowed and operands rounded at all
        let withoutANE = CS.Constraints(sustained: false, allowsANE: false), fp32 = CS.Constraints(sustained: false, allowsRounding: false)
        #expect(!withoutANE.allowsHalf && !fp32.allowsHalf)
        #expect(CS.attention(key, flash: false, model: m, constraints: fp32)?.assignment == .tiles)
        // an assignment someone else chose is not timed when it is not allowed
        #expect(CS.attention(assigned: .tiles16, key: key, model: m, constraints: withoutANE) == nil)
        #expect(CS.attention(assigned: .tiles16, key: key, model: m, constraints: .init(sustained: false))?.route == "tiles16")
    }

    @Test func aSlowerHalfKernelDoesNotWinJustBecauseItIsAllowed() throws {
        #expect(try assignment(model(tiles: 2e12, tiles16: 1.9e12)).assignment == .tiles)
        #expect(try assignment(model(tiles: 2e12, tiles16: 2e12)).assignment == .tiles, "neutral on a tie: the simpler")
    }

    @Test func aKernelTheJudgeRefusedIsNeverChosen() throws {
        let m = model(tiles: 2e12, tiles16: 4e12)
        #expect(try assignment(m, excluded: [M.Kernel(.gpu, .tiles16)]).assignment == .tiles)
        #expect(try assignment(m, excluded: [M.Kernel(.gpu, .tiles16), M.Kernel(.gpu, .tiles)]).assignment == .graph)
    }

    @Test func theFlashKernelStillCompetesWhereTheCallerTriesIt() throws {
        let m = model(tiles: 2e12, tiles16: 2.5e12, flash: 3e12)
        #expect(try assignment(m, flash: true).assignment == .flash)
        #expect(try assignment(m, flash: false).assignment == .tiles16)
    }

    @Test func aShapeTheKernelCannotTileIsLeftToTheGraph() throws {
        let odd = AttentionKey(sequence: 1024, heads: 16, headDim: 72)
        #expect(!AttentionAssignment.tiles.isRunnable(for: odd))
        #expect(try assignment(model(tiles: 9e12, tiles16: 9e12), key: odd).assignment == .graph)
        let grouped = AttentionKey(sequence: 4096, heads: 48, kvHeads: 12, headDim: 128)
        #expect(AttentionAssignment.tiles16.isRunnable(for: grouped))
        #expect(!AttentionAssignment(kernel: .tiles, cpuHeads: 4).isRunnable(for: grouped), "tiles run whole")
        #expect(try assignment(model(tiles: 9e12), key: grouped).assignment == .tiles)
    }

    // MARK: The plan

    @Test func aPlanFromTheMapAssignsTheHalfKernelOnlyToAProcessThatMayRunIt() {
        let m = model(tiles: 2e12, tiles16: 4e12)
        let compact = CapacityPlan(model: m, sustained: false, allowsANE: true)
        #expect(compact.assign(key) == .tiles16)
        #expect(compact.resources.halfAttention && compact.resources.halfPrecision, "its checks judge against the half-precision reference")
        let withoutANE = compact.restricted(allowingANE: false)
        #expect(withoutANE.assign(key) == .tiles)
        #expect(!withoutANE.resources.halfAttention)
        #expect(!compact.restricted(accelerators: false).resources.halfAttention, "the accelerators off: fp32")
        #expect(CapacityPlan(model: m, allowsANE: false).identity.hasSuffix("no-ane"))
        // the memo is by exact key: two lengths, two answers from the same lines
        #expect(compact.assign(AttentionKey(sequence: 1024, heads: 30, headDim: 128)) == .tiles16)
    }

    @Test func theLinesOfTheTilesEnterThePlansIdentity() {
        #expect(CapacityPlan(model: model(tiles: 2e12)).identity != CapacityPlan(model: model(tiles: 2.1e12)).identity)
        #expect(CapacityPlan(model: model()).identity != CapacityPlan(model: model(tiles: 2e12)).identity)
    }

    @Test func aDeveloperForcesTheTilesByTheirWords() {
        let tiles = CapacityPlan.forcing(named: "tiles"), half = CapacityPlan.forcing(named: "tiles16")
        #expect(tiles?.assign(key) == .tiles && tiles?.assign(key, flash: true) == .tiles)
        #expect(half?.assign(key) == .tiles16)
        #expect(tiles?.resources == .init() && half?.resources == .init(halfAttention: true))
        #expect(half?.resources.halfPrecision == true && tiles?.resources.halfPrecision == false)
        #expect(half?.identity.contains("tiles16") == true)
        #expect(CapacityPlan.forcing(named: "tiles,turbo") == nil)
        // a shape the kernel cannot tile keeps the engine's defaults
        #expect(tiles?.assign(AttentionKey(sequence: 1024, heads: 16, headDim: 72)).usesTiles == false)
        #expect(CapacityPlan.forcing(named: "amx,flash")?.assign(key, flash: true) == .flash, "the other words are untouched")
    }

    // MARK: The kernel's own sizes

    @Test func aPassKeepsItsScoresUnderTheBudget() {
        for (heads, s, keys) in [(30, 4128, 4128), (30, 1024, 1024), (24, 4096, 77), (48, 16384, 16384), (12, 777, 777)] {
            let t = TiledAttention.tiling(heads: heads, sequence: s, keys: keys)
            #expect(t.rows >= 1 && t.rows <= s && t.heads >= 1 && t.heads <= heads)
            // 64 rows is the least a long pass keeps: past the budget only when one block of 64 rows of keys is itself over it
            if t.rows > 64 { #expect(t.heads * t.rows * keys * 4 <= TiledAttention.scoreBudget, "\(heads) \(s) \(keys)") }
            if t.rows == s { #expect(t.heads * s * keys * 4 <= TiledAttention.scoreBudget || t.heads == 1) }
        }
        let long = TiledAttention.tiling(heads: 30, sequence: 4128, keys: 4128)
        #expect(long.rows % 64 == 0 && long.heads == 1)
        #expect(TiledAttention.tiling(heads: 24, sequence: 4096, keys: 77).rows == 4096, "few keys: the whole sequence in a pass")
    }

    @Test func theBuffersAreTheFormulasAndTheHalfOnesAddToThem() {
        let f32 = TiledAttention.bytes(heads: 30, kvHeads: 30, sequence: 4128, keys: 4128, headDim: 128, precision: .fp32)
        let f16 = TiledAttention.bytes(heads: 30, kvHeads: 30, sequence: 4128, keys: 4128, headDim: 128, precision: .fp16)
        let t = TiledAttention.tiling(heads: 30, sequence: 4128, keys: 4128)
        #expect(f32 == .init(scores: Arena.capacity(for: t.heads * t.rows * 4128 * 4), probabilities: 0, operands: 0))
        #expect(f16.scores == f32.scores)
        #expect(f16.probabilities == Arena.capacity(for: t.heads * t.rows * 4128 * 2))
        #expect(f16.operands == 3 * Arena.capacity(for: 4128 * 30 * 128 * 2))
        // grouped heads: k and v are the narrower
        let g = TiledAttention.bytes(heads: 48, kvHeads: 12, sequence: 4096, keys: 4096, headDim: 128, precision: .fp16)
        #expect(g.operands == Arena.capacity(for: 4096 * 48 * 128 * 2) + 2 * Arena.capacity(for: 4096 * 12 * 128 * 2))
    }

    // MARK: The memory need

    @Test func theMemoryNeedChargesOneTermPerBufferOfTheTiles() {
        for (assignment, precision) in [(AttentionAssignment.tiles, TiledAttention.Precision.fp32), (.tiles16, .fp16)] {
            let terms = MemoryNeed.attention("a", heads: 30, rows: 4128, keys: 4128, headDim: 128, causal: false, assignment: assignment, threads: 6)
            let b = TiledAttention.bytes(heads: 30, kvHeads: 30, sequence: 4128, keys: 4128, headDim: 128, precision: precision)
            #expect(terms == [.init("a tiles scores", b.scores), .init("a tiles probabilities", b.probabilities),
                              .init("a tiles half operands", b.operands)])
            #expect(terms.allSatisfy { !$0.outside }, "buffers of ours, sized by our function")
            #expect(!terms.contains { $0.name.contains("SDPA") }, "no graph is built")
        }
        let grouped = MemoryNeed.attention("k", heads: 48, rows: 4096, keys: 4096, headDim: 128, causal: false, assignment: .tiles16, threads: 6, group: 4)
        #expect(grouped.first { $0.name == "k tiles half operands" }?.bytes
                == TiledAttention.bytes(heads: 48, kvHeads: 12, sequence: 4096, keys: 4096, headDim: 128, precision: .fp16).operands)
        let graph = MemoryNeed.attention("a", heads: 30, rows: 4128, keys: 4128, headDim: 128, causal: false, threads: 6)
        #expect(graph.contains { $0.name == "a SDPA queries" }, "the graph's terms are unchanged")
    }

    @Test func aRequestOnTheTilesChargesThemInItsDenoisingPhase() throws {
        for (words, expected) in [("tiles", false), ("tiles16", true)] {
            var request = MemoryNeedTests.zRequest(width: 1024, height: 1024, promptTokens: 50)
            request.plan = try #require(CapacityPlan.forcing(named: words))
            let need = try #require(MemoryNeed.zImage(request, maps: MemoryFixtures.zImageMaps(bytesPerValue: 2)))
            let phase = try #require(need.phases.first { $0.kind == .denoising })
            #expect(phase.bytes("self-attention tiles scores") > 0 || phase.terms.contains { $0.name.contains("tiles scores") }, "\(words)")
            #expect((phase.terms.first { $0.name.contains("tiles probabilities") }?.bytes ?? 0 > 0) == expected, "\(words)")
        }
    }

    // MARK: The cut composed with the best GPU kernel

    func cut(graph: Double = 1.5e12, tiles: Double? = nil, tiles16: Double? = nil, flash: Double? = nil, cpu: Double = 0.5e12) -> M {
        var m = model(graph: graph, tiles: tiles, tiles16: tiles16, flash: flash)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: cpu)
        return m.withFreeSharing()
    }

    @Test func theCutTakesTheBestKernelForItsGPUHeads() throws {
        let m = cut(tiles: 2e12, tiles16: 4e12)
        let compact = try assignment(m, half: true)
        #expect(compact.assignment.splits && compact.assignment.gpuKernel == .tiles16, "\(compact.assignment)")
        #expect(compact.assignment.description.hasSuffix("+ tiles16"))
        let withoutANE = try assignment(m, half: false)
        #expect(withoutANE.assignment.splits && withoutANE.assignment.gpuKernel == .tiles, "no fp16 without its permission: \(withoutANE.assignment)")
        // a graph that is the best keeps the split as it was
        let plain = try assignment(cut(graph: 3e12, tiles: 2e12, tiles16: 2.5e12))
        #expect(plain.assignment.splits && plain.assignment.gpuKernel == .graph)
        // the cut gives the CPU fewer heads when the GPU's are faster
        #expect(compact.assignment.cpuHeads < plain.assignment.cpuHeads)
        // the time is the GPU's heads on their kernel and the CPU's, to the model
        #expect(compact.rows[M.Kernel(.gpu, .tiles16)] == key.heads - compact.assignment.cpuHeads)
    }

    @Test func flashIsAGPUKernelOfTheCutOnlyWhereTheCallerTriesIt() throws {
        let m = cut(graph: 1.5e12, flash: 3e12)
        #expect(try assignment(m, flash: true).assignment.gpuKernel == .flash)
        #expect(try assignment(m, flash: false).assignment.gpuKernel == .graph)
    }

    @Test func aCutOnAKernelTheJudgeRefusedIsNotOffered() throws {
        let m = cut(tiles: 2e12, tiles16: 4e12)
        let s = try assignment(m, excluded: [M.Kernel(.gpu, .tiles16)])
        #expect(s.assignment.gpuKernel == .tiles)
        #expect(CS.attention(assigned: AttentionAssignment(kernel: .split, cpuHeads: 10, gpuKernel: .tiles16), key: key, model: m,
                             constraints: .init(sustained: false, excluded: [M.Kernel(.gpu, .tiles16)])) == nil)
    }

    @Test func aCutRunsOnlyOnAKernelThatTakesTheShape() {
        let odd = AttentionKey(sequence: 4096, heads: 16, headDim: 72)
        #expect(AttentionAssignment(kernel: .split, cpuHeads: 4, gpuKernel: .graph).isRunnable(for: odd))
        #expect(!AttentionAssignment(kernel: .split, cpuHeads: 4, gpuKernel: .tiles16).isRunnable(for: odd))
        #expect(!AttentionAssignment(kernel: .split, cpuHeads: 4, gpuKernel: .flash).isRunnable(for: odd))
        let grouped = AttentionKey(sequence: 4096, heads: 48, kvHeads: 12, headDim: 128)
        #expect(AttentionAssignment(kernel: .split, cpuHeads: 20, gpuKernel: .tiles).isRunnable(for: grouped))
        #expect(!AttentionAssignment(kernel: .split, cpuHeads: 20, gpuKernel: .flash).isRunnable(for: grouped), "flash wants as many k/v heads as q heads")
        #expect(!AttentionAssignment(kernel: .graph, cpuHeads: 0, gpuKernel: .tiles).isRunnable(for: grouped), "a GPU kernel is the cut's")
    }

    @Test func theMemoryNeedChargesTheTilesOnTheGPUsHeadsOnly() {
        let a = AttentionAssignment(kernel: .split, cpuHeads: 13, gpuKernel: .tiles16)
        let terms = MemoryNeed.attention("a", heads: 30, rows: 4128, keys: 4128, headDim: 128, causal: false, assignment: a, threads: 6)
        let b = TiledAttention.bytes(heads: 30, kvHeads: 30, sequence: 4128, keys: 4128, headDim: 128, precision: .fp16, computed: 17)
        #expect(terms == [.init("a head split", HeadSplitAttention.hostBytes(heads: 30, sequence: 4128, keys: 4128, headDim: 128, cpuHeads: 13, threads: 6)),
                          .init("a tiles scores", b.scores), .init("a tiles probabilities", b.probabilities), .init("a tiles half operands", b.operands)])
        #expect(!terms.contains { $0.name.contains("SDPA") }, "no graph on the GPU's heads, no copy of their queries")
        let all = TiledAttention.bytes(heads: 30, kvHeads: 30, sequence: 4128, keys: 4128, headDim: 128, precision: .fp16)
        #expect(b.operands < all.operands, "17 heads of 30, not all")
        let fp32 = MemoryNeed.attention("a", heads: 30, rows: 4128, keys: 4128, headDim: 128, causal: false,
                                        assignment: .init(kernel: .split, cpuHeads: 13, gpuKernel: .tiles), threads: 6)
        #expect(fp32.suffix(2).allSatisfy { $0.bytes == 0 }, "no probabilities, no half copies in fp32")
        let flash = MemoryNeed.attention("a", heads: 30, rows: 4128, keys: 4128, headDim: 128, causal: false,
                                         assignment: .init(kernel: .split, cpuHeads: 13, gpuKernel: .flash), threads: 6)
        #expect(flash.count == 1)
    }
}
