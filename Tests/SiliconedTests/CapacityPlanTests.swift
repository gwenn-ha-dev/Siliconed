import Foundation
import Testing
@testable import Siliconed

/// **The plan** (`CapacityPlan.swift`): the questions the engine puts, the answers it gets — exact rows, the engine's defaults
/// without a model, a process without the ANE, a probe's pin, a check's forcing. Pure: no GPU, no weights.
@Suite struct CapacityPlanTests {
    typealias M = CapacityModel
    let gpu = M.Kernel(.gpu, .fp32), amx = M.Kernel(.amx, .fp32), ane = M.Kernel(.ane, .fp16)

    func key(_ k: Int, _ n: Int, _ m: Int, _ weight: CapacityClasses.WeightFormat = .bf16,
             phase: CapacityClasses.Phase = .denoising) -> ProductKey {
        .init(k: k, n: n, m: m, weight: weight, phase: phase)
    }

    func machine() -> M {
        var m = M()
        m.lines[gpu] = .init(t0: 0, rate: 3e12)
        m.lines[amx] = .init(t0: 0, rate: 1e12)
        m.lines[ane] = .init(t0: 0, rate: 6e12)
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        m.sharing["gpu+ane"] = .init(overhead: 0.0002, keep: [.gpu: 0.8, .ane: 0.7])
        m.sharing["gpu+amx"] = .init(overhead: 0.0001, keep: [.gpu: 0.9, .amx: 0.8])
        m.sharing["gpu+amx+ane"] = .init(overhead: 0.0003, keep: [.gpu: 0.7, .amx: 0.6, .ane: 0.6])
        return m
    }

    // MARK: the defaults

    @Test func withoutAModelTheGPUAloneAndTheHeadSplitFromTheMinimumSequence() {
        let plan = CapacityPlan.defaults
        #expect(plan.resources == .init())
        for m in [1, 32, 511, 1056, 4128, 6176] {
            for weight in CapacityClasses.WeightFormat.allCases {
                for phase in CapacityClasses.Phase.allCases {
                    #expect(plan.assign(key(3840, 3840, m, weight, phase: phase)) == .gpuAlone(m))
                }
            }
        }
        // The cut is the reference share, whole kv groups, from 1 024 rows: what every process ran before a plan existed.
        for (heads, kv) in [(30, 30), (32, 32), (16, 16), (48, 12), (24, 24)] {
            let group = heads / kv
            for s in [64, 1023, 1024, 1056, 4128, 6176] {
                let a = plan.assign(AttentionKey(sequence: s, heads: heads, kvHeads: kv, headDim: 128), flash: true)
                #expect(!a.usesFlash)
                if s < HeadSplitAttention.minimumSequence {
                    #expect(a == .graph)
                } else {
                    #expect(a.kernel == .split)
                    #expect(a.cpuHeads == HeadSplitAttention.partition(heads: heads, fraction: 0.4, group: group).cpu.count)
                    #expect(a.cpuHeads % group == 0 && a.cpuHeads < heads)
                }
            }
        }
        #expect(plan.assign(AttentionKey(sequence: 4128, heads: 30, headDim: 128)).cpuHeads == 12, "Z-Image: 12 of 30 heads")
        #expect(plan.assign(AttentionKey(sequence: 4128, heads: 48, kvHeads: 12, headDim: 128)).cpuHeads == 20, "Krea 2: 5 of 12 groups")
    }

    /// The engine's defaults on the real classes of the five families: every product the GPU's whole, every attention the cut
    /// the engines ran before a plan existed (`partition` at the reference share, from 1 024 rows).
    @Test func theDefaultsAreTodaysRoutesOnEveryClassOfEveryFamily() throws {
        var seen = 0
        for family in Family.allCases {
            for variant in Variant.allCases {
                let request = CapacityClasses.Request(family: family, variant: variant, width: 1024, height: 1536, includesEncoder: true)
                guard let classes = CapacityClasses.classes(for: request) else { continue }
                for c in classes {
                    switch c.shape {
                    case let .product(k, n, rows, format):
                        #expect(CapacityPlan.defaults.assign(ProductKey(k: k, n: n, m: rows, weight: format, phase: c.phase)) == .gpuAlone(rows))
                    case let .attention(s, keys, heads, kv, dim, cross):
                        let a = CapacityPlan.defaults.assign(AttentionKey(sequence: s, keys: keys, heads: heads, kvHeads: kv, headDim: dim, cross: cross),
                                                             flash: CapacitySolver.triesFlash(family: family, cross: cross))
                        let group = heads / kv
                        let cpu = HeadSplitAttention.partition(heads: heads, fraction: HeadSplitAttention.referenceFraction, group: group).cpu.count
                        #expect(a == (s >= HeadSplitAttention.minimumSequence ? AttentionAssignment(kernel: .split, cpuHeads: cpu) : .graph),
                                "\(family) \(variant) S=\(s)")
                    }
                    seen += 1
                }
            }
        }
        #expect(seen > 100)
    }

    // MARK: the model's plan

    @Test func aModelPlanBringsUpTheEnginesItHasLinesFor() {
        #expect(CapacityPlan(model: machine()).resources == .init(amx: true, ane: true))
        #expect(CapacityPlan(model: machine(), allowsANE: false).resources == .init(amx: true))
        var m = machine()
        m.lines[M.Kernel(.gpu, .bf16)] = .init(t0: 0, rate: 1e12)
        m.lines[M.Kernel(.gpu, .q8)] = .init(t0: 0, rate: 1e12)
        let plan = CapacityPlan(model: m, excluded: [M.Kernel(.amx, .fp32)])
        #expect(plan.resources == .init(amx: false, ane: true, bf16Weights: true, q8Weights: true))
        #expect(plan.resources.usesTensor && plan.resources.halfPrecision)
    }

    @Test func aPlanAnswersTheSolversAnswerAndKeepsIt() throws {
        let m = machine()
        let plan = CapacityPlan(model: m)
        let k = key(4096, 4096, 5120)
        let a = plan.assign(k)
        let solved = try #require(CapacitySolver.product(k, model: m, constraints: .init(allowsANE: true)))
        #expect(a == solved.assignment)
        #expect(a.isRunnable(for: k) && a.ane > 0)
        #expect(plan.assign(k) == a, "the same key, the same answer")
        #expect(plan.answered.map(\.0) == [k])
        // a shape never seen is solved when it is asked, by the same solver, and the nearest point is nobody's
        let other = key(5000, 3000, 3333)
        let b = plan.assign(other)
        #expect(b == (try #require(CapacitySolver.product(other, model: m, constraints: .init(allowsANE: true)))).assignment)
        #expect(b.rows == 3333 && b.isRunnable(for: other))
    }

    @Test func theSameShapeInTheEncoderAndTheDenoiserAreTwoClasses() {
        let plan = CapacityPlan(model: machine())
        let dit = plan.assign(key(4096, 4096, 5120)), enc = plan.assign(key(4096, 4096, 5120, phase: .encoding))
        #expect(dit.ane > 0 && enc.ane == 0)
    }

    @Test func aProcessWithoutTheANEHasItsOwnPlan() {
        let plan = CapacityPlan(model: machine())
        let without = plan.restricted(allowingANE: false)
        #expect(plan.restricted(allowingANE: true) === plan)
        #expect(without !== plan && !without.usesANE && without.usesAMX)
        let k = key(4096, 4096, 5120)
        #expect(plan.assign(k).ane > 0)
        let a = without.assign(k)
        #expect(a.ane == 0, "computed without it: \(a)")
        // not the ANE's rows given to the GPU: a better split of the engines it has
        #expect(a.amx > 0 || a == .gpuAlone(5120))
        #expect(without.restricted(allowingANE: false) === without)
        #expect(plan.restricted(allowingANE: false) === without, "made once: its memo is as valuable as the plan's own")
    }

    @Test func theIdentityNamesTheModel() {
        let a = CapacityPlan(model: machine()), b = CapacityPlan(model: machine())
        #expect(a.identity == b.identity)
        var other = machine(); other.lines[gpu] = .init(t0: 0, rate: 4e12)
        #expect(CapacityPlan(model: other).identity != a.identity)
        #expect(a.restricted(allowingANE: false).identity != a.identity)
        #expect(CapacityPlan.defaults.identity == "default")
    }

    // MARK: a probe pins, a check forces

    @Test func aProbePinsTheRowsItMeasures() {
        let plan = CapacityPlan.probing(amx: true, ane: true)
        #expect(plan.resources == .init(amx: true, ane: true))
        let k = key(3840, 3840, 4096)
        #expect(plan.assign(k) == .gpuAlone(4096))
        plan.pin(k, ProductAssignment(gpu: 2048, amx: 512, ane: 1536))
        #expect(plan.assign(k) == ProductAssignment(gpu: 2048, amx: 512, ane: 1536))
        plan.pin(k, ProductAssignment(gpu: 640, amx: 3456))
        #expect(plan.assign(k).amx == 3456)
        plan.unpin(k)
        #expect(plan.assign(k) == .gpuAlone(4096))
    }

    @Test func aCheckForcesHalfTheRowsOnTheANEAndTheRoundingOnTheBF16Weights() {
        let ane = CapacityPlan.forcing(neuralEngine: true, bf16Activations: false)
        #expect(ane.resources == .init(ane: true))
        #expect(ane.assign(key(3840, 3840, 4096)) == ProductAssignment(gpu: 2048, ane: 2048))
        #expect(ane.assign(key(3840, 3840, 1056)) == ProductAssignment(gpu: 1056 - 512, ane: 512), "half of 1056 on the unit: 528 → 512")
        #expect(ane.assign(key(3840, 3840, 500)) == .gpuAlone(500), "under 512 rows: the GPU alone")
        #expect(ane.assign(key(3840, 3840, 4096, phase: .encoding)) == .gpuAlone(4096), "an encoder never")
        #expect(ane.restricted(allowingANE: false).assign(key(3840, 3840, 4096)) == .gpuAlone(4096), "a process without it folds the rows")
        let rounded = CapacityPlan.forcing(neuralEngine: false, bf16Activations: true)
        #expect(rounded.resources.roundedActivations && rounded.resources.bf16Weights && !rounded.usesANE)
        #expect(rounded.assign(key(3840, 3840, 4096)) == ProductAssignment(path: .bf16, roundsActivation: true, gpu: 4096))
        #expect(rounded.assign(key(3840, 3840, 255)) == .gpuAlone(255))
        #expect(rounded.assign(key(3840, 3840, 4096, .int8)) == .gpuAlone(4096), "the 8-bit weights keep the widened path")
        #expect(rounded.assign(key(3840, 3840, 4096, phase: .encoding)) == .gpuAlone(4096))
    }

    @Test func theJudgesAMXSplitIsTheCutTheJudgesWereTakenWith() {
        // `SILICONED_AMX=1` once meant: from 512 rows, the GPU keeps 0.631 of the rows on 256-row tiles, the AMX the rest.
        func cut(_ m: Int) -> Int { max(256, min(m - 256, Int((Double(m) * 0.631 / 256).rounded()) * 256)) }
        let plan = CapacityPlan.forcing(amx: true)
        #expect(plan.resources == .init(amx: true))
        for m in [512, 1024, 1040, 1056, 1536, 4096, 4117, 4128, 6176] {
            let a = plan.assign(key(3840, 3840, m))
            #expect(a == ProductAssignment(gpu: cut(m), amx: m - cut(m)), "m = \(m)")
            #expect(plan.assign(key(3840, 3840, m, .int8, phase: .encoding)) == a, "the encoders and every weight form too")
        }
        for m in [1, 32, 64, 511] { #expect(plan.assign(key(3840, 3840, m)) == .gpuAlone(m), "under 512 rows: the GPU alone") }
        #expect(plan.assign(AttentionKey(sequence: 4128, heads: 30, headDim: 128), flash: true).kernel == .split, "the attention keeps its default")
    }

    @Test func aDeveloperNamesAForcingByItsWords() {
        #expect(CapacityPlan.forcing(named: "") == nil)
        #expect(CapacityPlan.forcing(named: "turbo") == nil, "a word it does not know: the whole list unread")
        #expect(CapacityPlan.forcing(named: "amx,turbo") == nil)
        let both = CapacityPlan.forcing(named: "amx, flash")
        #expect(both?.resources == .init(amx: true))
        #expect(both?.assign(AttentionKey(sequence: 4128, heads: 30, headDim: 128), flash: true) == .flash)
        #expect(both?.assign(AttentionKey(sequence: 4128, heads: 30, headDim: 128), flash: false).kernel == .split,
                "a caller that does not try the flash kernel never gets it")
        let all = CapacityPlan.forcing(named: "amx,ane,bf16act")
        #expect(all?.resources == .init(amx: true, ane: true, bf16Weights: true, roundedActivations: true))
        // the ANE's rule is ahead of the AMX's on the products it takes; the AMX serves the rest
        #expect(all?.assign(key(3840, 3840, 4096)) == ProductAssignment(path: .bf16, roundsActivation: true, gpu: 4096))
        #expect(all?.assign(key(3840, 3840, 4096, .int8)) == ProductAssignment(gpu: 2048, ane: 2048))
        #expect(all?.assign(key(3840, 3840, 300, .int8)) == .gpuAlone(300))
        #expect(all?.assign(key(3840, 3840, 1024, .int8, phase: .encoding)).amx ?? 0 > 0)
    }

    /// **An unknown height is charged at the worst of the plan's answers over the heights it may take**: the solver's integers
    /// are not monotone in the height (the ANE changes unit, an engine leaves the set), so the bound's own answer can be under what
    /// a shorter prompt's makes the engine hold.
    @Test func aBoundedHeightIsChargedAtItsWorstAnswer() {
        var m = M()
        m.lines[gpu] = .init(t0: 0.0002, rate: 3.3e12); m.lines[amx] = .init(t0: 0.0005, rate: 1.6e12); m.lines[ane] = .init(t0: 0.003, rate: 3.0e12)
        m.sharing["gpu+ane"] = .init(overhead: 0.0004, keep: [.gpu: 0.85, .ane: 0.6])
        m.sharing["gpu+amx"] = .init(overhead: 0.0002, keep: [.gpu: 0.9, .amx: 0.8])
        m.sharing["gpu+amx+ane"] = .init(overhead: 0.0006, keep: [.gpu: 0.8, .amx: 0.6, .ane: 0.55])
        let plan = CapacityPlan(model: m)
        var underTheBound = 0
        for (k, n, bound, spread) in [(3840, 3840, 4128, 160), (3840, 10240, 4256, 300), (10240, 3840, 1568, 520), (4096, 4096, 6176, 128)] {
            let charged = MemoryNeed.gemm("t", bounded: [(bound, k, n)], spread: spread, plan: plan)
            let (workspace, surfaces) = (charged.first { $0.name == "t conductor workspace" }?.bytes ?? 0,
                                         charged.first { $0.name == "t Neural Engine" }?.bytes ?? 0)
            let atTheBound = MemoryNeed.gemm("t", products: [(bound, k, n)], plan: plan)
            for h in (bound - spread)...bound {
                let one = MemoryNeed.gemm("t", products: [(h, k, n)], plan: plan)
                let (w, e) = (one.first { $0.name == "t conductor workspace" }?.bytes ?? 0, one.first { $0.name == "t Neural Engine" }?.bytes ?? 0)
                #expect(w <= workspace && e <= surfaces, "\(h) of \(bound): \(w) > \(workspace) or \(e) > \(surfaces)")
                if w > (atTheBound.first { $0.name == "t conductor workspace" }?.bytes ?? 0)
                    || e > (atTheBound.first { $0.name == "t Neural Engine" }?.bytes ?? 0) { underTheBound += 1 }
            }
        }
        #expect(underTheBound > 0, "the bound's own answer is not an upper bound: the test must show it")
    }

    // MARK: what the engines can run

    @Test func theHardwareAndPrecisionLimitsAreChecked() {
        let k = key(3840, 3840, 4096)
        #expect(ProductAssignment(gpu: 4096).isRunnable(for: k))
        #expect(ProductAssignment(gpu: 640, amx: 3456).isRunnable(for: k), "any AMX rows: BNNS takes the ragged")
        #expect(ProductAssignment(gpu: 0, amx: 4096).isRunnable(for: k), "the AMX alone")
        #expect(ProductAssignment(gpu: 2048, ane: 2048).isRunnable(for: k))
        #expect(ProductAssignment(gpu: 0, ane: 4096).isRunnable(for: k), "the ANE alone")
        #expect(!ProductAssignment(gpu: 2000, ane: 2096).isRunnable(for: k), "the ANE's rows come in units of 256")
        #expect(!ProductAssignment(gpu: 2048, ane: 2048).isRunnable(for: key(3840, 3840, 4096, phase: .encoding)), "no ANE on an encoder")
        #expect(!ProductAssignment(gpu: 4000).isRunnable(for: k), "the rows add up to the height")
        #expect(!ProductAssignment(gpu: -1, amx: 4097).isRunnable(for: k))
        // a weight read as published
        #expect(ProductAssignment(path: .bf16, gpu: 4096).isRunnable(for: k))
        #expect(ProductAssignment(path: .bf16, roundsActivation: true, gpu: 4096).isRunnable(for: k))
        #expect(!ProductAssignment(path: .bf16, roundsActivation: true, gpu: 4096).isRunnable(for: key(3840, 3840, 4096, phase: .encoding)))
        #expect(!ProductAssignment(path: .bf16, gpu: 2048, amx: 2048).isRunnable(for: k), "TensorGEMM takes the product whole")
        #expect(!ProductAssignment(path: .bf16, gpu: 4096).isRunnable(for: key(3840, 3840, 4096, .int8)))
        #expect(!ProductAssignment(roundsActivation: true, gpu: 4096).isRunnable(for: k), "a widened weight is not rounded")
        let int8 = key(3840, 3840, 4096, .int8)
        #expect(ProductAssignment(path: .native, gpu: 4096).isRunnable(for: int8))
        #expect(ProductAssignment(path: .native, gpu: 4096, amxColumns: 1280).isRunnable(for: int8), "2 560 GPU columns: whole tiles")
        #expect(!ProductAssignment(path: .native, gpu: 4096, amxColumns: 1000).isRunnable(for: int8), "a GPU edge inside a tile")
        #expect(!ProductAssignment(path: .native, gpu: 4096, amxColumns: 3840).isRunnable(for: int8), "the GPU keeps a tile")
        #expect(!ProductAssignment(path: .native, gpu: 2048, ane: 2048).isRunnable(for: int8), "no ANE beside a native weight")
        #expect(!ProductAssignment(path: .native, gpu: 4096, amxColumns: 64).isRunnable(for: key(3840, 100, 4096, .int8)))
        #expect(!ProductAssignment(path: .native, gpu: 4096).isRunnable(for: k), "a native path needs an 8-bit weight")
    }

    @Test func anAttentionRunsWholeGroupsAndTheGPUKeepsOne() {
        let key = AttentionKey(sequence: 4096, heads: 48, kvHeads: 12, headDim: 128)
        #expect(key.group == 4)
        #expect(AttentionAssignment(kernel: .split, cpuHeads: 20).isRunnable(for: key))
        #expect(!AttentionAssignment(kernel: .split, cpuHeads: 18).isRunnable(for: key), "inside a group")
        #expect(!AttentionAssignment(kernel: .split, cpuHeads: 48).isRunnable(for: key), "the GPU keeps a group")
        #expect(AttentionAssignment.graph.isRunnable(for: key) && AttentionAssignment.flash.isRunnable(for: key))
        #expect(!AttentionAssignment(kernel: .graph, cpuHeads: 4).isRunnable(for: key))
    }

    @Test func routesAreNamedAsTheLedgerAndTheMapNameThem() {
        #expect(ProductAssignment.gpuAlone(10).engines == "gpu")
        #expect(ProductAssignment(gpu: 5, amx: 5).engines == "gpu+amx")
        #expect(ProductAssignment(gpu: 5, ane: 5).engines == "gpu+ane")
        #expect(ProductAssignment(gpu: 4, amx: 3, ane: 3).engines == "gpu+ane+amx")
        #expect(ProductAssignment(gpu: 0, amx: 10).engines == "amx")
        #expect(ProductAssignment(gpu: 0, ane: 10).engines == "ane")
        #expect(ProductAssignment(path: .bf16, gpu: 10).route(weight: .bf16) == "gpu/bf16")
        #expect(ProductAssignment(path: .bf16, roundsActivation: true, gpu: 10).route(weight: .bf16) == "gpu/bf16act")
        #expect(ProductAssignment(path: .native, gpu: 10).route(weight: .int8) == "gpu/int8")
        #expect(ProductAssignment(path: .native, gpu: 10, amxColumns: 64).route(weight: .q8) == "gpu+amx/q8")
    }

    @Test func aKeyAndAnAssignmentSurviveTheirText() throws {
        let k = key(3840, 10240, 4128, .q8)
        let a = ProductAssignment(path: .native, gpu: 4128, amxColumns: 2560)
        #expect(try JSONDecoder().decode(ProductKey.self, from: JSONEncoder().encode(k)) == k)
        #expect(try JSONDecoder().decode(ProductAssignment.self, from: JSONEncoder().encode(a)) == a)
        let t = AttentionKey(sequence: 4128, keys: 512, heads: 16, headDim: 128, cross: true)
        #expect(try JSONDecoder().decode(AttentionKey.self, from: JSONEncoder().encode(t)) == t)
        let s = AttentionAssignment(kernel: .split, cpuHeads: 6)
        #expect(try JSONDecoder().decode(AttentionAssignment.self, from: JSONEncoder().encode(s)) == s)
    }

    // MARK: the memory need asks the same questions

    @Test func theNeedAsksThePlanWhatTheEngineAsksIt() {
        let plan = CapacityPlan(model: machine())
        let products = [(m: 4128, k: 3840, n: 3840), (m: 4128, k: 3840, n: 10240), (m: 1056, k: 3840, n: 3840)]
        let terms = MemoryNeed.gemm("t", products: products, plan: plan)
        // what the engine will hold, by the same answers
        var workspace = 0
        var ane: [(rows: Int, k: Int, n: Int)] = []
        for p in products {
            let a = plan.assign(key(p.k, p.n, p.m))
            if a.amx > 0 {
                workspace = max(workspace, [false, true].map {
                    Conductor.workspaceBytes(rows: a.amx, k: p.k, n: p.n, weightIsTransposed: $0)
                }.max()!)
            }
            if a.ane > 0 { ane.append((a.ane, p.k, p.n)) }
        }
        #expect(ane.count > 0 && workspace > 0, "the test must put rows on both engines")
        #expect(terms.first { $0.name == "t conductor workspace" }?.bytes == Arena.capacity(for: workspace))
        #expect(terms.first { $0.name == "t Neural Engine" }?.bytes == NeuralEngine.bytes(products: ane))
        // an encoder's, or a process's without the ANE: none of the ANE's surfaces
        let without = MemoryNeed.gemm("t", products: products, plan: plan.restricted(allowingANE: false))
        #expect(!without.contains { $0.name == "t Neural Engine" })
        let encoder = MemoryNeed.gemm("t", products: products, plan: plan.restricted(allowingANE: false), phase: .encoding)
        #expect(!encoder.contains { $0.name == "t Neural Engine" })
        // the same inputs, the same bytes
        #expect(MemoryNeed.gemm("t", products: products, plan: plan) == terms)
    }
}
