import Foundation
import Testing
@testable import Siliconed

/// **The attention on bf16 operands and the fused form** (`tilesbf16`, `AttentionTile`): the bf16 kernel is the rounding a request
/// allows (the Standard's too), the fp16 one needs the ANE allowed as well (`Constraints.allowsHalf`, every version) and wins a tie; the form the map chose for a half-precision
/// kernel travels from its measurements into the model, its JSON, the plan's identity and the assignment the solver hands the callers;
/// the fused form holds its operands' copies alone. No GPU, no weights: synthetic lines.
@Suite struct AttentionTileTests {
    typealias M = CapacityModel
    typealias CS = CapacitySolver
    let key = AttentionKey(sequence: 4128, heads: 30, headDim: 128)

    func model(graph: Double = 1.5e12, tiles16: Double? = nil, tilesbf16: Double? = nil) -> M {
        var m = M()
        m.lines[M.Kernel(.gpu, .fp32)] = .init(t0: 0, rate: 3e12)
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: graph)
        if let tiles16 { m.lines[M.Kernel(.gpu, .tiles16)] = .init(t0: 0, rate: tiles16) }
        if let tilesbf16 { m.lines[M.Kernel(.gpu, .tilesbf16)] = .init(t0: 0, rate: tilesbf16) }
        return m
    }

    @Test func aTileReadsAsItIsWritten() throws {
        let t = AttentionTile(16, 32, simdgroups: 4)
        #expect(t.description == "16x32/4" && AttentionTile("16x32/4") == t)
        #expect(AttentionTile("16x32") == nil && AttentionTile("12x32/4") == nil && AttentionTile("16x40/4") == nil)
        let data = try JSONEncoder().encode([t])
        #expect(try JSONDecoder().decode([AttentionTile].self, from: data) == [t])
        #expect(Set(AttentionTile.candidates).count == AttentionTile.candidates.count)
    }

    @Test func theBF16KernelIsTheRoundingTheStandardMayTake() throws {
        let m = model(tilesbf16: 6e12)
        let standard = CS.Constraints(request: .init(family: .zImage, variant: .standard, includesEncoder: false), sustained: false)
        #expect(standard.allowsHalf && standard.allowsRounding, "the Standard: bf16 and fp16 both (2/11 and 3/11/2026)")
        let s = try #require(CS.attention(key, flash: false, model: m, constraints: standard))
        #expect(s.assignment.kernel == .tilesbf16 && s.route == "tilesbf16")
        let withoutANE = CS.Constraints(sustained: false, allowsANE: false)
        #expect(!withoutANE.allowsHalf && withoutANE.allowsRounding, "no ANE: the rounding stays, fp16 goes")
        #expect(CS.attention(key, flash: false, model: model(tiles16: 9e12, tilesbf16: 6e12), constraints: withoutANE)?.assignment.kernel == .tilesbf16)
        let refused = CS.Constraints(sustained: false, allowsANE: false, allowsRounding: false)
        #expect(CS.attention(key, flash: false, model: m, constraints: refused)?.assignment == .graph, "no rounding allowed: no bf16")
        #expect(CS.attention(key, flash: false, model: model(tilesbf16: 1.2e12), constraints: standard)?.assignment == .graph,
                "slower than the graph: the graph")
    }

    @Test func fp16WinsWithinTheNoiseWhereItIsAllowed() throws {
        let compact = CS.Constraints(request: .init(family: .zImage, variant: .compact, includesEncoder: false), sustained: false)
        #expect(CS.attention(key, flash: false, model: model(tiles16: 6e12, tilesbf16: 6e12), constraints: compact)?.assignment.kernel == .tiles16)
        #expect(CS.attention(key, flash: false, model: model(tiles16: 6e12, tilesbf16: 6.2e12), constraints: compact)?.assignment.kernel == .tiles16,
                "within the noise: the finer mantissa")
        #expect(CS.attention(key, flash: false, model: model(tiles16: 6e12, tilesbf16: 7e12), constraints: compact)?.assignment.kernel == .tilesbf16)
    }

    @Test func theMapsFormTravelsIntoTheAssignment() throws {
        var m = model(tiles16: 4e12, tilesbf16: 6e12)
        let fused = AttentionTile(16, 32, simdgroups: 4)
        m.attentionTiles[M.Kernel(.gpu, .tilesbf16)] = fused
        let s = try #require(CS.attention(key, flash: false, model: m, constraints: .init(sustained: false)))
        #expect(s.assignment == AttentionAssignment(kernel: .tilesbf16, tile: fused))
        #expect(s.assignment.description == "tilesbf16 16x32/4")
        // A kernel whose map kept the three stages runs them.
        m.attentionTiles = [:]
        #expect(try #require(CS.attention(key, flash: false, model: m, constraints: .init(sustained: false))).assignment.tile == nil)
    }

    @Test func theFormGoesFromTheMeasurementsIntoTheModelItsJSONAndThePlan() throws {
        var m = CapacityFit.Measurements()
        m.attentionTiles = ["gpu/tilesbf16": "16x32/4", "gpu/tiles16": "16x64/4", "gpu/bf16act": "16x32/4", "gpu/tiles": "16x32/4", "gpu/tiles16x": "1x1/1"]
        let model = CapacityFit.model(from: m)
        #expect(model.attentionTiles == [M.Kernel(.gpu, .tilesbf16): AttentionTile(16, 32, simdgroups: 4),
                                         M.Kernel(.gpu, .tiles16): AttentionTile(16, 64, simdgroups: 4)], "half-precision kernels only")
        let back = try JSONDecoder().decode(M.self, from: JSONEncoder().encode(model))
        #expect(back.attentionTiles == model.attentionTiles && back.rounded().attentionTiles == model.attentionTiles)
        var other = model
        other.attentionTiles[M.Kernel(.gpu, .tiles16)] = nil
        #expect(CapacityPlan(model: model).identity != CapacityPlan(model: other).identity)
    }

    @Test func theFusedFormHoldsItsOperandsCopiesAlone() {
        let staged = TiledAttention.bytes(heads: 30, kvHeads: 30, sequence: 4128, keys: 4128, headDim: 128, precision: .bf16)
        let fused = TiledAttention.bytes(heads: 30, kvHeads: 30, sequence: 4128, keys: 4128, headDim: 128, precision: .bf16,
                                         tile: AttentionTile(16, 32, simdgroups: 4))
        #expect(fused.scores == 0 && fused.probabilities == 0 && fused.operands == staged.operands && staged.scores > 0)
        let a = AttentionAssignment(kernel: .tilesbf16, tile: AttentionTile(16, 32, simdgroups: 4))
        let terms = MemoryNeed.attention("a", heads: 30, rows: 4128, keys: 4128, headDim: 128, causal: false, assignment: a, threads: 6)
        #expect(terms.reduce(0) { $0 + $1.bytes } == fused.asTotal)
    }

    @Test func aDeveloperCanForceTheBF16KernelAndTheFusedForm() throws {
        let plan = try #require(CapacityPlan.forcing(named: "tilesbf16,fused"))
        let a = plan.assign(key)
        #expect(a.kernel == .tilesbf16 && a.tile == AttentionTile.candidates[0])
        #expect(plan.identity.contains("tilesbf16") && plan.identity.contains("fused"))
        #expect(try #require(CapacityPlan.forcing(named: "tilesbf16")).assign(key).tile == nil)
        #expect(plan.resources.halfPrecision, "a check under it judges against the half-precision reference")
    }

    @Test func theCutComposesWithTheBF16Kernel() throws {
        var m = model(graph: 1.5e12, tilesbf16: 6e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        m = m.withFreeSharing()
        m.attentionTiles[M.Kernel(.gpu, .tilesbf16)] = AttentionTile(16, 64, simdgroups: 4)
        let s = try #require(CS.attention(key, flash: false, model: m, constraints: .init(sustained: false)))
        #expect(s.assignment.splits && s.assignment.gpuKernel == .tilesbf16, "\(s.assignment)")
        #expect(s.assignment.tile == AttentionTile(16, 64, simdgroups: 4))
    }
}
