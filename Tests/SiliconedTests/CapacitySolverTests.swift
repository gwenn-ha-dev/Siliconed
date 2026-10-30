import Foundation
import Testing
@testable import Siliconed

/// **The solver** (`CapacitySolver.swift`) on synthetic machines whose right answer is computed by hand, and its table read
/// back as the engine reads it. No GPU, no weights, no chip name.
@Suite struct CapacitySolverTests {
    typealias M = CapacityModel
    typealias CC = CapacityClasses
    typealias CS = CapacitySolver
    let gpu = M.Kernel(.gpu, .fp32), amx = M.Kernel(.amx, .fp32), ane = M.Kernel(.ane, .fp16)

    func machine(gpuRate: Double = 3e12, amxRate: Double? = nil, aneRate: Double? = nil) -> M {
        var m = M()
        m.lines[gpu] = .init(t0: 0, rate: gpuRate)
        if let amxRate { m.lines[amx] = .init(t0: 0, rate: amxRate) }
        if let aneRate { m.lines[ane] = .init(t0: 0, rate: aneRate) }
        return m
    }

    func product(_ k: Int, _ n: Int, _ rows: Int, _ format: CC.WeightFormat = .bf16, phase: CC.Phase = .denoising,
                 family: Family = .zImage, times: Int = 1) -> CC.Class {
        .init(phase: phase, shape: .product(k: k, n: n, rows: rows, format: format), executions: [family: times])
    }

    func attention(_ s: Int, heads: Int = 32, kv: Int? = nil, dim: Int = 128, cross: Bool = false,
                   family: Family = .qwenImage21, keys: Int? = nil) -> CC.Class {
        .init(phase: .denoising, shape: .attention(sequence: s, keys: keys ?? s, heads: heads, kvHeads: kv ?? heads, headDim: dim, cross: cross),
              executions: [family: 1])
    }

    func caps(amx: Bool = false, ane: Bool = false, flash: Bool = false, split: Bool = false, bf16: Bool = false,
              rounding: Bool = false, int8: Bool = false, q8: Bool = false) -> CapacityMap.Capabilities {
        .init(amx: amx, ane: ane, bf16Weights: bf16, bf16Rounding: rounding, int8Weights: int8, q8Weights: q8, flash: flash, headSplit: split)
    }

    func solve(_ model: M, _ classes: [CC.Class], family: Family = .zImage, variant: Variant = .compact,
               caps: CapacityMap.Capabilities) throws -> CS.Plan {
        try #require(CS.solve(model: model, classes: classes, constraints: .init(family: family, variant: variant, capabilities: caps, sustained: false)))
    }

    /// The route the engine names for a class under the plan's table.
    func engineRoute(_ plan: CS.Plan, _ c: CC.Class, family: Family = .zImage, variant: Variant = .compact) -> String {
        CC.route(of: c, family: family, table: plan.table, aneBroughtUp: variant != .standard, weightForms: true)
    }

    func seconds(_ k: Int, _ n: Int, rows: Int, rate: Double) -> Double { 2 * Double(k) * Double(n) * Double(rows) / rate }

    // MARK: engines

    @Test func aMacWithoutTheANEBalancesTheAMXByHand() throws {
        // GPU 3, AMX 1 TFLOP/s, nothing lost in sharing: the GPU keeps 3/4 of 4096 rows.
        let c = product(4096, 4096, 4096)
        let plan = try solve(machine(amxRate: 1e12), [c], caps: caps(amx: true))
        let block = try #require(plan.blocks.first)
        #expect(block.route == "gpu+amx")
        #expect(block.rows == [gpu: 3072, amx: 1024])
        #expect(abs(block.compute - seconds(4096, 4096, rows: 3072, rate: 3e12)) < 1e-9)
        #expect(plan.table.products.amx.enabled && !plan.table.products.ane.enabled)
        #expect(engineRoute(plan, c) == "gpu+amx")
    }

    @Test func anANEWorthFourGPUsIsServedWithoutChangingAnything() throws {
        // The same code, a machine whose ANE is 4× the GPU: it takes 4/5 of 5120 rows = 16 units of 256.
        let c = product(4096, 4096, 5120)
        let plan = try solve(machine(aneRate: 12e12), [c], caps: caps(ane: true))
        let block = try #require(plan.blocks.first)
        #expect(block.route == "gpu+ane")
        #expect(block.rows == [gpu: 1024, ane: 4096])
        #expect(abs(block.compute - seconds(4096, 4096, rows: 1024, rate: 3e12)) < 1e-9)
        #expect(engineRoute(plan, c) == "gpu+ane")
        // and the engine's own functions give the same rows from the table
        let r = plan.table.products.route(k: 4096, n: 4096, m: 5120)
        #expect(r.aneRows == 4096)
    }

    @Test func theStandardNeverHasTheANEWhateverItIsWorth() throws {
        let c = product(4096, 4096, 5120)
        let plan = try solve(machine(aneRate: 12e12), [c], variant: .standard, caps: caps(ane: true))
        #expect(plan.blocks[0].route == "gpu")
        #expect(!plan.table.products.ane.enabled)
        #expect(engineRoute(plan, c, variant: .standard) == "gpu")
    }

    @Test func anEncoderNeverHasTheANE() throws {
        let c = product(4096, 4096, 5120, phase: .encoding)
        let plan = try solve(machine(aneRate: 12e12), [c], caps: caps(ane: true))
        #expect(plan.blocks[0].route == "gpu")
        #expect(!plan.table.products.ane.enabled)
    }

    @Test func aTerribleSharingStaysOnOneEngine() throws {
        var m = machine(aneRate: 3e12)
        m.sharing["gpu+ane"] = .init(overhead: 0.5, keep: [.gpu: 0.2, .ane: 0.2])
        let plan = try solve(m, [product(4096, 4096, 1024)], caps: caps(ane: true))
        #expect(plan.blocks[0].route == "gpu")
        #expect(plan.blocks[0].rows == [gpu: 1024])
    }

    @Test func theANEOnlyTakesWholeUnitsAndPaysTheStaircase() throws {
        var m = machine(aneRate: 3e12)
        m.sharing["gpu+ane"] = .init(overhead: 0, keep: [.gpu: 0.9, .ane: 0.9])
        let plan = try solve(m, [product(3840, 10240, 4096 + 128)], caps: caps(ane: true))
        let rows = try #require(plan.blocks[0].rows[ane])
        #expect(rows % 256 == 0 && rows > 0)
        // no neighbouring unit does better than the one it took
        let model = m
        func t(_ r: Int) -> Double { model.seconds(m: 4224, k: 3840, n: 10240, rows: [gpu: 4224 - r, ane: r])! }
        #expect(t(rows) <= t(rows + 256) + 1e-12 && t(rows) <= t(rows - 256) + 1e-12)
    }

    @Test func threeEnginesTakeTheAMXPointAtTheGPUsSide() throws {
        var m = machine(amxRate: 1e12, aneRate: 3e12)
        m.sharing["gpu+amx+ane"] = .init(overhead: 0, keep: [.gpu: 1, .amx: 1, .ane: 1])
        let c = product(4096, 4096, 7168)
        let plan = try solve(m, [c], caps: caps(amx: true, ane: true))
        let block = plan.blocks[0]
        #expect(block.route == "gpu+ane+amx")
        #expect(block.rows[ane] == 3072 && block.rows[gpu] == 3072 && block.rows[amx] == 1024)
        // the AMX's point is the GPU's side (7168 − 3072 = 4096 rows), not the whole product
        #expect(plan.table.products.amx.grid.shares.points.contains { $0.m == 4096 })
        #expect(engineRoute(plan, c) == "gpu+ane+amx")
    }

    // MARK: weights

    @Test func aWeightIsPreparedAtEveryCallAndPaysForItsPath() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .int8)] = .init(t0: 0, rate: 3e12)
        let c = product(4096, 4096, 4096, .int8, times: 10)
        m.preparation[.int8] = [.widened: 1e-9, .native: 1e-11]
        var plan = try solve(m, [c], caps: caps(int8: true))
        #expect(plan.blocks[0].route == "gpu/int8")
        #expect(abs(plan.blocks[0].preparation - 10 * 4096 * 4096 * 1e-11) < 1e-9)
        m.preparation[.int8] = [.widened: 1e-12, .native: 1e-8]
        plan = try solve(m, [c], caps: caps(int8: true))
        #expect(plan.blocks[0].route == "gpu")
        #expect(plan.table.products.weight(.int8, k: 4096, n: 4096, m: 4096).path == .widened)
    }

    @Test func aKQuantHasItsOwnPreparation() throws {
        var m = machine()
        m.preparation[.bf16] = [.widened: 1e-11]
        m.kquantPreparation = 5e-10
        let plan = try solve(m, [product(2048, 2048, 1024, .q4k, times: 3)], variant: .light, caps: caps())
        #expect(abs(plan.blocks[0].preparation - 3 * 2048 * 2048 * 5e-10) < 1e-9)
        // a K-quant has no weight rule: the tensor is none of the kinds
        #expect(plan.table.products.weights.isEmpty)
    }

    @Test func theNativeInt8WeightGoesBesideTheAMXOnColumns() throws {
        var m = machine(amxRate: 1e12)
        m.lines[M.Kernel(.gpu, .int8)] = .init(t0: 0, rate: 3e12)
        m.lines[M.Kernel(.amx, .int8)] = .init(t0: 0, rate: 1e12)
        m.preparation[.int8] = [.widened: 1e-9, .native: 1e-11]
        let c = product(4096, 4096, 4096, .int8)
        let plan = try solve(m, [c], caps: caps(amx: true, int8: true))
        #expect(plan.blocks[0].route == "gpu+amx/int8")
        let columns = plan.blocks[0].rows[M.Kernel(.amx, .int8)] ?? 0
        #expect(columns == 1024)
        #expect(engineRoute(plan, c) == "gpu+amx/int8")
    }

    @Test func theBF16WeightAsPublishedWinsWhenItsPreparationIsCheaper() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .bf16)] = .init(t0: 0, rate: 3e12)
        m.preparation[.bf16] = [.widened: 1e-9, .bf16: 1e-11]
        let c = product(4096, 4096, 4096, .bf16, times: 4)
        let plan = try solve(m, [c], variant: .standard, caps: caps(bf16: true))
        #expect(plan.blocks[0].route == "gpu/bf16")
        #expect(engineRoute(plan, c, variant: .standard) == "gpu/bf16")
    }

    @Test func aRoundedActivationIsNotTheEncodersAndItsMemberRunsWidened() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .bf16act)] = .init(t0: 0, rate: 6e12)
        m.preparation[.bf16] = [.widened: 1e-10, .bf16: 0]
        let dit = product(4096, 4096, 4096, .bf16), enc = product(4096, 4096, 4096, .bf16, phase: .encoding)
        let plan = try solve(m, [dit, enc], variant: .standard, caps: caps(bf16: true, rounding: true))
        #expect(plan.blocks[0].route == "gpu/bf16act")
        #expect(plan.blocks[1].route == "gpu")
        #expect(plan.blocks[1].preparation > 0)
    }

    // MARK: attention

    @Test func theHeadsOfAnAttentionAreBalancedLikeAProduct() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        let a = attention(4096)
        let plan = try solve(m, [a], family: .qwenImage21, caps: caps(split: true))
        let rule = try #require(plan.table.attention.rules.first)
        #expect(rule.splitFraction == 0.25 && !rule.flash)
        #expect(plan.blocks[0].rows[M.Kernel(.cpu, .heads)] == 8)
        #expect(CC.route(of: a, family: .qwenImage21, table: plan.table) == "split")
        #expect(abs(plan.blocks[0].compute - 4.0 * 4096 * 4096 * 128 * 24 / 1.5e12) < 1e-9)
    }

    @Test func aGroupedAttentionIsCutByWholeGroups() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        let plan = try solve(m, [attention(4096, heads: 48, kv: 12, family: .krea2)], family: .krea2, caps: caps(split: true))
        let cpu = plan.blocks[0].rows[M.Kernel(.cpu, .heads)] ?? 0
        #expect(cpu > 0 && cpu % 4 == 0 && cpu < 48)
    }

    @Test func flashIsTakenOnlyWhereTheFamilyTriesIt() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.gpu, .flash)] = .init(t0: 0, rate: 3e12)
        let a = attention(4096, family: .zImage)
        let z = try solve(m, [a], family: .zImage, caps: caps(flash: true))
        #expect(z.blocks[0].route == "flash")
        let q = try solve(m, [attention(4096)], family: .qwenImage21, caps: caps(flash: true))
        #expect(q.blocks[0].route == "graph")
        #expect(!q.table.attention.rules.contains { $0.flash })
    }

    @Test func anAnimaCrossAttentionSharesTheRuleOfItsSelfAttention() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.gpu, .flash)] = .init(t0: 0, rate: 3e12)
        let own = attention(4096, family: .anima), cross = attention(4096, cross: true, family: .anima, keys: 512)
        let plan = try solve(m, [own, cross], family: .anima, caps: caps(flash: true))
        #expect(plan.table.attention.rules.count == 1)
        #expect(plan.blocks.map(\.route) == ["flash", "graph"])
    }

    // MARK: the table

    @Test func aMixedClassServesTheEncoderThroughWithoutANE() throws {
        let dit = product(4096, 4096, 5120), enc = product(4096, 4096, 5120, phase: .encoding)
        let plan = try solve(machine(aneRate: 12e12), [dit, enc], caps: caps(ane: true))
        #expect(plan.blocks[0].route == "gpu+ane" && plan.blocks[1].route == "gpu")
        let rule = try #require(plan.table.products.classes.first { $0.k == 4096 })
        #expect(rule.engines == .ane && rule.withoutANE == .gpu)
    }

    @Test func theANEGridSaysTheGPUsWholeForTheProductsItDoesNotTake() throws {
        // memory counts the ANE's rows off this grid: a product it leaves must not read the default half
        let on = product(4096, 4096, 5120), off = product(512, 512, 2048)
        var m = machine(aneRate: 12e12)
        m.lines[ane] = .init(t0: 0.05, rate: 12e12)        // too dear for a small product
        let plan = try solve(m, [on, off], caps: caps(ane: true))
        #expect(plan.blocks[0].route == "gpu+ane")
        #expect(plan.blocks[1].route == "gpu")
        #expect(plan.table.products.ane.grid.shares.ratio(k: 512, n: 512, m: 2048) == 0)
        #expect(plan.table.products.ane.gpuShare == 1)
    }

    @Test func aRealRequestIsReadBackByTheEngineAndTheTableSurvivesJSON() throws {
        var m = machine(amxRate: 1e12, aneRate: 6e12)
        m.sharing["gpu+ane"] = .init(overhead: 0.0002, keep: [.gpu: 0.8, .ane: 0.7])
        m.sharing["gpu+amx"] = .init(overhead: 0.0001, keep: [.gpu: 0.9, .amx: 0.8])
        m.sharing["gpu+amx+ane"] = .init(overhead: 0.0003, keep: [.gpu: 0.7, .amx: 0.6, .ane: 0.6])
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.gpu, .flash)] = .init(t0: 0, rate: 2.5e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.3e12)
        m.sharing["gpu+cpu"] = .init(overhead: 0.0001, keep: [.gpu: 0.95, .cpu: 0.9])
        m.preparation[.bf16] = [.widened: 1e-10]
        let all = caps(amx: true, ane: true, flash: true, split: true)
        for (family, variant) in [(Family.zImage, Variant.compact), (.qwenImage21, .light), (.anima, .standard)] {
            let request = CC.Request(family: family, variant: variant, includesEncoder: true)
            let classes = try #require(CC.classes(for: request))
            let plan = try #require(CS.solve(model: m, request: request, constraints: .init(request: request, capabilities: all, sustained: false)))
            #expect(plan.unpredicted.isEmpty)
            let run = classes.filter { ($0.executions[family] ?? 0) > 0 }
            #expect(run.count == plan.blocks.count)
            for (c, b) in zip(run, plan.blocks) {
                // the engine's own reading of the table: the encoders and the Standard take no ANE
                let engine = CC.route(of: c, family: family, table: plan.table, aneBroughtUp: variant != .standard, weightForms: true)
                #expect(b.route == engine, "\(family) \(variant) \(c.routeKey.id) \(c.phase)")
            }
            let data = try JSONEncoder().encode(plan.table)
            #expect(try JSONDecoder().decode(RouteTable.self, from: data) == plan.table)
            #expect(plan.seconds == plan.blocks.reduce(0) { $0 + $1.seconds })
            if variant == .standard { #expect(!plan.table.products.ane.enabled) }
        }
    }

    @Test func theFullModelIsNeverSlowerThanTheGPUAlone() throws {
        var full = machine(amxRate: 1e12, aneRate: 6e12)
        full.sharing["gpu+ane"] = .init(overhead: 0.0002, keep: [.gpu: 0.8, .ane: 0.7])
        full.sharing["gpu+amx"] = .init(overhead: 0.0001, keep: [.gpu: 0.9, .amx: 0.8])
        full.sharing["gpu+amx+ane"] = .init(overhead: 0.0003, keep: [.gpu: 0.7, .amx: 0.6, .ane: 0.6])
        let request = CC.Request(family: .krea2, variant: .compact)
        let withAll = try #require(CS.solve(model: full, request: request, constraints: .init(request: request, capabilities: caps(amx: true, ane: true), sustained: false)))
        let alone = try #require(CS.solve(model: machine(), request: request, constraints: .init(request: request, capabilities: caps(), sustained: false)))
        #expect(withAll.seconds <= alone.seconds)
        #expect(withAll.seconds < 0.8 * alone.seconds)
        // what the memory need reads agrees with the table: the ANE's programs exist exactly where a block takes ANE rows
        let aneProducts = withAll.blocks.contains { $0.rows[ane] != nil }
        #expect(withAll.table.products.usesANE == aneProducts)
        #expect(alone.table.products.usesANE == false && alone.table.products.usesAMX == false)
    }

    @Test func theOverlapHookReplacesTheSum() throws {
        let longest: CS.Combine = { blocks in blocks.map(\.seconds).max() ?? 0 }
        let solved = CS.solve(model: machine(), classes: [product(4096, 4096, 4096), product(2048, 2048, 4096)],
                              constraints: .init(family: .zImage, variant: .standard, capabilities: caps(), sustained: false),
                              combine: longest)
        let plan = try #require(solved)
        #expect(plan.seconds == plan.blocks.map(\.seconds).max())
    }

    @Test func aModelWithoutAGPULineCannotBePlanned() {
        #expect(CS.solve(model: M(), classes: [product(4096, 4096, 4096)],
                         constraints: .init(family: .zImage, variant: .standard, capabilities: caps())) == nil)
    }
}
