import Foundation
import Testing
@testable import Siliconed

/// **The solver** (`CapacitySolver.swift`) on synthetic machines whose right answer is computed by hand: one class in, the exact
/// assignment out — whole rows, never a ratio. No GPU, no weights, no chip name.
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
        // The sets share nothing here: measured, and free (a set never measured is no candidate, `anUnmeasuredSetIsNeverACandidate`).
        return m.withFreeSharing()
    }

    func key(_ k: Int, _ n: Int, _ m: Int, _ weight: CC.WeightFormat = .bf16, phase: CC.Phase = .denoising) -> ProductKey {
        .init(k: k, n: n, m: m, weight: weight, phase: phase)
    }

    func attentionKey(_ s: Int, heads: Int = 32, kv: Int? = nil, dim: Int = 128, cross: Bool = false, keys: Int? = nil) -> AttentionKey {
        .init(sequence: s, keys: keys, heads: heads, kvHeads: kv, headDim: dim, cross: cross)
    }

    func constraints(ane: Bool = true) -> CS.Constraints { .init(sustained: false, allowsANE: ane) }

    func solve(_ model: M, _ key: ProductKey, ane: Bool = true) throws -> CS.ProductSolution {
        let s = try #require(CS.product(key, model: model, constraints: constraints(ane: ane)))
        #expect(s.assignment.isRunnable(for: key), "\(key): \(s.assignment)")
        return s
    }

    func solve(_ model: M, _ key: AttentionKey, flash: Bool = false) throws -> CS.AttentionSolution {
        let s = try #require(CS.attention(key, flash: flash, model: model, constraints: constraints()))
        #expect(s.assignment.isRunnable(for: key), "\(key): \(s.assignment)")
        return s
    }

    func seconds(_ k: Int, _ n: Int, rows: Int, rate: Double) -> Double { 2 * Double(k) * Double(n) * Double(rows) / rate }

    // MARK: engines

    @Test func aMacWithoutTheANEBalancesTheAMXByHand() throws {
        // GPU 3, AMX 1 TFLOP/s, nothing lost in sharing: the GPU keeps 3/4 of 4096 rows.
        let s = try solve(machine(amxRate: 1e12), key(4096, 4096, 4096))
        #expect(s.assignment == ProductAssignment(gpu: 3072, amx: 1024))
        #expect(s.route == "gpu+amx")
        #expect(s.rows == [gpu: 3072, amx: 1024])
        #expect(abs(s.compute - seconds(4096, 4096, rows: 3072, rate: 3e12)) < 1e-9)
    }

    @Test func theRowsAreOnTheGPUsTileAndTheAMXKeepsTheRemainder() throws {
        // 4117 rows: no multiple of 64 is the balance; the GPU's rows are one, the AMX's the ragged rest.
        let s = try solve(machine(amxRate: 1e12), key(4096, 4096, 4117))
        #expect(s.assignment.gpu % GEMM.rowQuantum == 0 && s.assignment.amx > 0)
        #expect(s.assignment.gpu + s.assignment.amx == 4117)
        // within one tile of the balance: no neighbouring cut does better
        let model = machine(amxRate: 1e12)
        func t(_ g: Int) -> Double { model.seconds(m: 4117, k: 4096, n: 4096, rows: [gpu: g, amx: 4117 - g])! }
        #expect(t(s.assignment.gpu) <= t(s.assignment.gpu + 64) + 1e-12 && t(s.assignment.gpu) <= t(s.assignment.gpu - 64) + 1e-12)
    }

    @Test func anANEWorthFourGPUsIsServedWithoutChangingAnything() throws {
        // The same code, a machine whose ANE is 4× the GPU: it takes 4/5 of 5120 rows = 16 units of 256.
        let s = try solve(machine(aneRate: 12e12), key(4096, 4096, 5120))
        #expect(s.assignment == ProductAssignment(gpu: 1024, ane: 4096))
        #expect(s.route == "gpu+ane")
        #expect(abs(s.compute - seconds(4096, 4096, rows: 1024, rate: 3e12)) < 1e-9)
    }

    @Test func aProcessWithoutTheANEIsSolvedWithoutIt() throws {
        let m = machine(aneRate: 12e12)
        let s = try solve(m, key(4096, 4096, 5120), ane: false)
        #expect(s.assignment == .gpuAlone(5120) && s.route == "gpu")
        // and the plan of such a process says the same, with no ANE to fall back from
        let plan = CapacityPlan(model: m, allowsANE: false)
        #expect(plan.assign(key(4096, 4096, 5120)) == .gpuAlone(5120))
        #expect(!plan.usesANE)
    }

    @Test func anEncoderNeverHasTheANE() throws {
        let s = try solve(machine(aneRate: 12e12), key(4096, 4096, 5120, phase: .encoding))
        #expect(s.assignment == .gpuAlone(5120))
    }

    @Test func aTerribleSharingStaysOnOneEngine() throws {
        var m = machine(aneRate: 3e12)
        m.sharing["gpu+ane"] = .init(overhead: 0.5, keep: [.gpu: 0.2, .ane: 0.2])
        let s = try solve(m, key(4096, 4096, 1024))
        // One engine: the GPU alone or the ANE alone, as fast as each other here — a tie the written order gives the ANE (`canonical`).
        #expect(s.assignment == .gpuAlone(1024) || s.assignment == ProductAssignment(path: .widened, gpu: 0, ane: 1024), "\(s.assignment)")
    }

    @Test func theANEOnlyTakesWholeUnitsAndPaysTheStaircase() throws {
        var m = machine(aneRate: 3e12)
        m.sharing["gpu+ane"] = .init(overhead: 0, keep: [.gpu: 0.9, .ane: 0.9])
        let s = try solve(m, key(3840, 10240, 4096 + 128))
        let rows = s.assignment.ane
        #expect(rows % 256 == 0 && rows > 0)
        // no neighbouring unit does better than the one it took
        let model = m
        func t(_ r: Int) -> Double { model.seconds(m: 4224, k: 3840, n: 10240, rows: [gpu: 4224 - r, ane: r])! }
        #expect(t(rows) <= t(rows + 256) + 1e-12 && t(rows) <= t(rows - 256) + 1e-12)
    }

    @Test func threeEnginesShareTheRowsAsIntegers() throws {
        var m = machine(amxRate: 1e12, aneRate: 3e12)
        m.sharing["gpu+amx+ane"] = .init(overhead: 0, keep: [.gpu: 1, .amx: 1, .ane: 1])
        let s = try solve(m, key(4096, 4096, 7168))
        #expect(s.route == "gpu+ane+amx")
        // GPU 3, ANE 3, AMX 1: 3072 + 1024 + 3072 rows
        #expect(s.assignment == ProductAssignment(gpu: 3072, amx: 1024, ane: 3072))
    }

    // MARK: the ANE against the AMX (`CapacitySolver.aneOverAMX`)

    /// GPU 3, AMX 1 TFLOP/s, the ANE as given; the three engines together made useless, so the duel is gpu+ane against gpu+amx.
    func duel(aneRate: Double) -> M {
        var m = machine(amxRate: 1e12, aneRate: aneRate)
        m.sharing["gpu+amx+ane"] = .init(overhead: 0, keep: [.gpu: 0.1, .amx: 0.1, .ane: 0.1])
        return m
    }

    func best(_ model: M, _ key: ProductKey, route: String) throws -> Double {
        try #require(CS.candidates(key, model: model, constraints: constraints()).filter { $0.route == route }.map(\.seconds).min())
    }

    @Test func theANEUpTo15PercentSlowerThanTheAMXIsTaken() throws {
        // 7168 rows: gpu+amx 5376 + 1792 (t ∝ 1792), gpu+ane 6144 + 1024 (t ∝ 2048): the ANE 14.3 % slower.
        let m = duel(aneRate: 0.5e12), k = key(4096, 4096, 7168)
        let (ane, amx) = (try best(m, k, route: "gpu+ane"), try best(m, k, route: "gpu+amx"))
        #expect(ane > amx && ane <= amx * (1 + CS.aneOverAMX))
        let s = try solve(m, k)
        #expect(s.route == "gpu+ane" && s.assignment == ProductAssignment(gpu: 6144, ane: 1024), "\(s.assignment)")
    }

    @Test func theANEMoreThan15PercentSlowerLeavesTheAMX() throws {
        let m = duel(aneRate: 0.25e12), k = key(4096, 4096, 7168)
        #expect(try best(m, k, route: "gpu+ane") > (try best(m, k, route: "gpu+amx")) * (1 + CS.aneOverAMX))
        #expect(try solve(m, k).route == "gpu+amx")
    }

    @Test func thePreferenceIsAgainstTheAMXOnlyNeverAgainstTheGPUAlone() throws {
        // The GPU alone on a rounded activation (3.8 TFLOP/s, t ∝ 1886) is slower than gpu+amx (1792) and faster than gpu+ane (2048,
        // within 15 % of gpu+amx). The AMX is dropped, and between what is left time decides: the GPU alone, not the ANE.
        var m = duel(aneRate: 0.5e12)
        m.lines[M.Kernel(.gpu, .bf16act)] = .init(t0: 0, rate: 3.8e12)
        let k = key(4096, 4096, 7168)
        let (ane, amx, gpu) = (try best(m, k, route: "gpu+ane"), try best(m, k, route: "gpu+amx"), try best(m, k, route: "gpu/bf16act"))
        #expect(amx < gpu && gpu < ane && ane <= amx * (1 + CS.aneOverAMX), "gpu+amx \(amx), gpu/bf16act \(gpu), gpu+ane \(ane)")
        #expect(try solve(m, k).route == "gpu/bf16act")
    }

    @Test func anEngineMayBeAlone() throws {
        // A Mac whose GPU is the slow engine: the AMX alone, then the ANE alone, with no GPU row.
        var slow = machine(gpuRate: 0.2e12, amxRate: 4e12)
        slow.sharing["gpu+amx"] = .init(overhead: 0.01, keep: [.gpu: 0.5, .amx: 0.5])
        var s = try solve(slow, key(4096, 4096, 4096))
        #expect(s.assignment == ProductAssignment(gpu: 0, amx: 4096) && s.route == "amx")
        var fast = machine(gpuRate: 0.2e12, aneRate: 12e12)
        fast.sharing["gpu+ane"] = .init(overhead: 0.01, keep: [.gpu: 0.5, .ane: 0.5])
        s = try solve(fast, key(4096, 4096, 4096))
        #expect(s.assignment == ProductAssignment(gpu: 0, ane: 4096) && s.route == "ane")
        // an ANE alone on a height that is not a unit still runs the whole product (the model pays the rounded rows)
        s = try solve(fast, key(4096, 4096, 4224))
        #expect(s.assignment.ane == 4224 && s.assignment.gpu == 0)
    }

    @Test func theFloorsOfTheModelsDomainKeepSmallProductsOnTheGPU() throws {
        let m = machine(amxRate: 3e12, aneRate: 12e12)
        let s = try solve(m, key(4096, 4096, 256))
        #expect(s.assignment == .gpuAlone(256), "under the AMX's and the ANE's floors: the GPU takes it")
    }

    // MARK: weights

    @Test func aWeightIsPreparedAtEveryCallAndPaysForItsPath() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .int8)] = .init(t0: 0, rate: 3e12)
        m.preparation[.int8] = [.widened: 1e-9, .native: 1e-11]
        var s = try solve(m, key(4096, 4096, 4096, .int8))
        #expect(s.route == "gpu/int8" && s.assignment.path == .native)
        #expect(abs(s.preparation - 4096 * 4096 * 1e-11) < 1e-9)
        m.preparation[.int8] = [.widened: 1e-12, .native: 1e-8]
        s = try solve(m, key(4096, 4096, 4096, .int8))
        #expect(s.route == "gpu" && s.assignment.path == .widened)
    }

    @Test func aLoRAsProductsHaveNothingToPrepareAndAreTimedByTheThinLines() throws {
        var m = machine(amxRate: 1e12)
        m.preparation[.bf16] = [.widened: 1e-9]
        m.kquantPreparation = 5e-9
        // The stack's operands are expanded beforehand (`CacheLoRA`): no preparation, whatever the model prices for a weight.
        let down = try solve(m, key(3072, 128, 4117, .loraDown))
        #expect(down.preparation == 0)
        // A product with a short side is timed by the thin lines when the probes measured them: t₀ + w·k·n + h·M·(k + n) + FLOP / R, M the rows the GPU computes (tiles of 64).
        m.thin[gpu] = .init(t0: 1e-4, rate: 2e12, perWeight: 1e-11, perRow: 1e-10)
        let line = try #require(m.thin[gpu])
        let expected = line.seconds(rows: CapacityModel.computedRows(gpu, rows: 4117), k: 3072, n: 128)
        #expect(abs(try #require(m.alone(gpu, rows: 4117, k: 3072, n: 128)) - expected) < 1e-12)
        #expect(m.alone(gpu, rows: 4117, k: 3072, n: 3072) != expected, "a product with no short side keeps the big products' line")
        // No set is offered on such a product — the loss of an engine beside another was measured on big ones —: the GPU alone, though
        // the AMX would be balanced by hand on a big product of the same machine.
        #expect(try solve(m, key(3072, 128, 4117, .loraDown)).assignment == .gpuAlone(4117))
        #expect(try solve(m, key(4096, 4096, 4117)).assignment.amx > 0)
        // An engine alone, when its thin line beats the GPU's: the AMX takes the whole product, and a `loraUp` pays the pass that adds
        // its rows to the output (three passes of m·n floats at the bandwidth).
        m.thin[amx] = .init(t0: 1e-5, rate: 8e12)
        m.memoryBandwidth = 100e9
        let up = try solve(m, key(128, 16384, 4117, .loraUp))
        #expect(up.assignment == ProductAssignment(gpu: 0, amx: 4117))
        let product = try #require(m.alone(amx, rows: 4117, k: 128, n: 16384))
        #expect(abs(up.compute - (product + 3 * 4 * 4117 * 16384 / 100e9)) < 1e-12)
        #expect(try solve(m, key(128, 16384, 4117, .bf16)).compute < up.compute, "the pass is a LoRA's alone")
        // The GPU's rows read the output once more, in their own kernel.
        let gpuUp = try solve(machine(), key(128, 16384, 4117, .loraUp))
        m.memoryBandwidth = 100e9
        var g = machine(); g.memoryBandwidth = 100e9
        let withRead = try solve(g, key(128, 16384, 4117, .loraUp))
        #expect(abs(withRead.compute - (gpuUp.compute + 4 * 4117 * 16384 / 100e9)) < 1e-12)
        // A slow memory makes the pass dearer than the AMX's help: the GPU keeps the product.
        m.memoryBandwidth = 1e9
        #expect(try solve(m, key(128, 16384, 4117, .loraUp)).assignment == .gpuAlone(4117))
    }

    @Test func aKQuantHasItsOwnPreparation() throws {
        var m = machine()
        m.preparation[.bf16] = [.widened: 1e-11]
        m.kquantPreparation = 5e-10
        let s = try solve(m, key(2048, 2048, 1024, .q4k))
        #expect(abs(s.preparation - 2048 * 2048 * 5e-10) < 1e-9)
        #expect(s.assignment == .gpuAlone(1024), "a K-quant widens: no native form")
    }

    /// **A K-quant's blocks read as published** (`gpu/q4kact`): on the denoiser where the map has the line and it beats the widened
    /// route with its preparation — the GPU alone, the activation rounded —; the encoder, a slower line, a refused rounding keep the
    /// widened weight. A tensor of the class the kernels cannot read (a Q6_K) asks the plan its answer without the route.
    @Test func aKQuantIsReadAsPublishedOnARoundedActivationWhereItIsFaster() throws {
        var m = machine(aneRate: 1e12)
        m.lines[M.Kernel(.gpu, .q4kact)] = .init(t0: 0, rate: 6e12)
        m.kquantPreparation = 5e-10
        m.kquantStaging = 1e-11
        let q4k = key(4096, 4096, 4096, .q4k)
        let s = try solve(m, q4k)
        #expect(s.route == "gpu/q4kact" && s.assignment == .gpuAlone(4096, path: .native, rounds: true))
        #expect(abs(s.preparation - 4096 * 4096 * 1e-11) < 1e-9, "the blocks copied, not widened")
        #expect(try solve(m, key(4096, 4096, 4096, .q4k, phase: .encoding)).assignment.path == .widened)
        let refused = try #require(CS.product(q4k, model: m, constraints: CS.Constraints(allowsANE: true, allowsRounding: false)))
        #expect(refused.assignment.path == .widened)
        var slow = m
        slow.lines[M.Kernel(.gpu, .q4kact)] = .init(t0: 0, rate: 1e11)
        #expect(try solve(slow, q4k).assignment.path == .widened)
        // The kernels alone: no ANE, no AMX, no fp32 activation on the blocks.
        #expect(!ProductAssignment(path: .native, roundsActivation: true, gpu: 2048, ane: 2048).isRunnable(for: q4k))
        #expect(!ProductAssignment(path: .native, roundsActivation: false, gpu: 4096).isRunnable(for: q4k))
        // The fallback of an unread tensor is the plan without the route, the widened one, and the plan's own answer elsewhere.
        let plan = CapacityPlan(model: m)
        #expect(plan.assign(q4k).path == .native)
        let widened = plan.assignWidened(q4k)
        #expect(widened.path == .widened && widened.isRunnable(for: q4k))
        #expect(plan.assignWidened(key(4096, 4096, 4096)) == plan.assign(key(4096, 4096, 4096)))
    }

    /// **No AMX beside a native weight**: the AMX's columns widened their share of the weight on every call, a line no probe measures
    /// as the route runs, and the widened `gpu+amx` covers the case with lines measured; the solver never offers them.
    @Test func theNativeInt8WeightNeverGoesBesideTheAMXOnColumns() throws {
        var m = machine(amxRate: 1e12)
        m.lines[M.Kernel(.gpu, .int8)] = .init(t0: 0, rate: 3e12)
        m.lines[M.Kernel(.amx, .int8)] = .init(t0: 0, rate: 1e12)
        m.preparation[.int8] = [.widened: 1e-9, .native: 1e-11]
        let s = try solve(m, key(4096, 4096, 4096, .int8))
        #expect(s.route != "gpu+amx/int8")
        #expect(s.assignment.amxColumns == 0)
        #expect(s.rows[M.Kernel(.amx, .int8)] == nil)
    }

    /// **The ANE beside the GPU's int8 kernel, both reading the integers**: offered where the model has the line `ane/int8`, chosen
    /// when it is faster, with its own sharing (`gpu/int8+ane/int8`) and the integers the engines run.
    @Test func theANEReadsTheIntegersWhenTheModelSaysItIsFaster() throws {
        let g8 = M.Kernel(.gpu, .int8), a8 = M.Kernel(.ane, .int8)
        var m = machine(aneRate: 1e9)   // the ANE is up (its fp16 line, too slow to take anything here)
        m.lines[g8] = .init(t0: 0, rate: 3e12)
        m.preparation[.int8] = [.widened: 1e-9, .native: 1e-11]
        let int8 = key(4096, 4096, 4096, .int8)
        #expect(try solve(m, int8).route == "gpu/int8", "no line for the ANE's integers: the GPU's kernel alone")
        m.lines[a8] = .init(t0: 0, rate: 9e12)
        // 9 / (3 + 9) of 4 096 rows = 3 072 = 12 units
        var s = try solve(m, int8)
        #expect(s.assignment == ProductAssignment(path: .native, gpu: 1024, ane: 3072))
        #expect(s.route == "gpu+ane/int8" && s.rows == [g8: 1024, a8: 3072])
        #expect(abs(s.preparation - 4096 * 4096 * 1e-11) < 1e-9, "the integers are copied, not widened")
        #expect(abs(s.compute - seconds(4096, 4096, rows: 1024, rate: 3e12)) < 1e-9)
        // the pair's own sharing: the GPU keeps 0.6 of its rate beside the ANE, 1.8 against 9 → 0.833 of the rows = 3 413; the unit
        // below it leaves the GPU 768 rows (14.3 ms), the unit above 512 (13.4 ms), the ANE alone 15.3 ms
        m.sharing["gpu/int8+ane/int8"] = .init(overhead: 0, keep: [.gpu: 0.6, .ane: 1])
        s = try solve(m, int8)
        #expect(s.assignment == ProductAssignment(path: .native, gpu: 512, ane: 3584), "\(s.assignment)")
        // the engines' key is not this pair's: an fp16 pair with a terrible keep changes nothing
        m.sharing["gpu+ane"] = .init(overhead: 0, keep: [.gpu: 1, .ane: 0.01])
        #expect(try solve(m, int8).assignment == s.assignment)
        // slower than the GPU alone: it stays out
        m.sharing = [:]
        m.lines[a8] = .init(t0: 0, rate: 0.1e12)
        #expect(try solve(m, int8).assignment.ane == 0)
        // an engine the judge refused is no candidate whatever its line says
        m.lines[a8] = .init(t0: 0, rate: 9e12)
        let refused = try #require(CS.product(int8, model: m, constraints: .init(sustained: false, excluded: [a8], allowsANE: true)))
        #expect(refused.route == "gpu/int8")
        // …nor without the ANE itself: a map that left its fp16 out brings no engine up for the integers
        let down = try #require(CS.product(int8, model: m, constraints: .init(sustained: false, excluded: [ane], allowsANE: true)))
        #expect(down.route == "gpu/int8")
        #expect(!CapacityPlan(model: m, excluded: [ane]).resources.aneInt8)
        // never an encoder, never a process without the ANE, never under the model's floor
        #expect(try solve(m, key(4096, 4096, 4096, .int8, phase: .encoding)).route == "gpu/int8")
        #expect(try solve(m, int8, ane: false).route == "gpu/int8")
        #expect(try solve(m, key(4096, 4096, 512, .int8)).route == "gpu/int8")
        // never a K the program cannot cut, never Q8_0's blocks
        #expect(try solve(m, key(3904, 4096, 4096, .int8)).route == "gpu/int8")
        m.lines[M.Kernel(.gpu, .q8)] = .init(t0: 0, rate: 3e12)
        m.preparation[.q8] = [.widened: 1e-9, .native: 1e-11]
        #expect(try solve(m, key(4096, 4096, 4096, .q8)).route == "gpu/q8")
        // the integers' answer is the one the plan gives
        #expect(CapacityPlan(model: m).assign(int8) == (try solve(m, int8)).assignment)
    }

    /// **An assignment decided elsewhere is timed with the rows the ANE takes**: the render's prediction asks `product(assigned:)` for the plan's
    /// answer, and the GPU's rows alone (what it read before) are a third of the seconds of a product the ANE shares.
    @Test func anAssignedIntegersProductIsTimedWithTheANEsRowsToo() throws {
        let g8 = M.Kernel(.gpu, .int8), a8 = M.Kernel(.ane, .int8)
        var m = machine(aneRate: 1e9)
        m.lines[g8] = .init(t0: 0, rate: 3e12)
        m.lines[a8] = .init(t0: 0, rate: 9e12)
        m.sharing["gpu/int8+ane/int8"] = .init(overhead: 0, keep: [.gpu: 0.9, .ane: 0.8])
        m.preparation[.int8] = [.widened: 1e-9, .native: 1e-11]
        let int8 = key(4096, 4096, 4096, .int8)
        let c = CS.Constraints(sustained: false)
        let chosen = try solve(m, int8)
        let timed = try #require(CS.product(assigned: chosen.assignment, key: int8, model: m, constraints: c))
        #expect(timed.route == chosen.route && timed.rows == chosen.rows)
        #expect(abs(timed.compute - chosen.compute) < 1e-12, "the same seconds as the solver's own candidate: \(timed.compute) against \(chosen.compute)")
        // The ANE alone is the whole product; its rows are no longer left out.
        let whole = try #require(CS.product(assigned: ProductAssignment(path: .native, gpu: 0, ane: 4096), key: int8, model: m, constraints: c))
        #expect(whole.route == "ane/int8" && whole.rows == [a8: 4096])
        #expect(abs(whole.compute - 2 * 4096.0 * 4096 * 4096 / 9e12) < 1e-9)
        // A plan that names the ANE where the model has no line for its integers is not timed.
        m.lines[a8] = nil
        #expect(CS.product(assigned: chosen.assignment, key: int8, model: m, constraints: c) == nil)
    }

    @Test func theANEMayTakeTheWholeIntegersProductWhenTheGPUIsTheSlowEngine() throws {
        let g8 = M.Kernel(.gpu, .int8), a8 = M.Kernel(.ane, .int8)
        var m = machine(gpuRate: 0.5e12, aneRate: 1e9)
        m.lines[g8] = .init(t0: 0, rate: 0.5e12)
        m.lines[a8] = .init(t0: 0, rate: 12e12)
        m.sharing["gpu/int8+ane/int8"] = .init(overhead: 0.01, keep: [.gpu: 0.5, .ane: 0.9])
        m.preparation[.int8] = [.widened: 1e-9, .native: 1e-11]
        let s = try solve(m, key(4096, 4096, 2048, .int8))
        #expect(s.assignment == ProductAssignment(path: .native, gpu: 0, ane: 2048), "\(s.assignment)")
        #expect(s.route == "ane/int8")
    }

    @Test func theBF16WeightAsPublishedWinsWhenItsPreparationIsCheaper() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .bf16)] = .init(t0: 0, rate: 3e12)
        m.preparation[.bf16] = [.widened: 1e-9, .bf16: 1e-11]
        let s = try solve(m, key(4096, 4096, 4096, .bf16))
        #expect(s.route == "gpu/bf16" && s.assignment.path == .bf16)
    }

    @Test func aRoundedActivationIsTheDenoisersAndTheEncoderRunsWidened() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .bf16act)] = .init(t0: 0, rate: 6e12)
        m.preparation[.bf16] = [.widened: 1e-10, .bf16: 0]
        let dit = try solve(m, key(4096, 4096, 4096, .bf16)), enc = try solve(m, key(4096, 4096, 4096, .bf16, phase: .encoding))
        #expect(dit.route == "gpu/bf16act" && dit.assignment.roundsActivation)
        #expect(enc.route == "gpu" && enc.preparation > 0)
        // below the rounding's floor the activation keeps its fp32
        #expect(try solve(m, key(64, 64, 128, .bf16)).assignment.roundsActivation == false)
        // **the Standard's too, where the map says it is faster** (P20, decided 2/11/2026): the ANE's absence is not a precision; its
        // plan brings the rounding up when the map has its line
        let standard = try solve(m, key(4096, 4096, 4096, .bf16), ane: false)
        #expect(standard.assignment.roundsActivation && standard.route == "gpu/bf16act")
        #expect(CapacityPlan(model: m, allowsANE: false).resources.roundedActivations)
        #expect(CapacityPlan(model: m).resources.roundedActivations)
        // ... and only there: on a GPU whose rounded kernel is slower than its fp32 product (an M1 Pro's), the operand stays fp32
        var slow = m
        slow.lines[M.Kernel(.gpu, .bf16act)] = .init(t0: 0, rate: 1e12)
        #expect(try solve(slow, key(4096, 4096, 4096, .bf16), ane: false).assignment.roundsActivation == false)
        // a constraint that refuses the rounding is obeyed
        let refused = try #require(CS.product(key(4096, 4096, 4096, .bf16), model: m,
                                              constraints: CS.Constraints(allowsANE: false, allowsRounding: false)))
        #expect(!refused.assignment.roundsActivation)
    }

    /// **The ANE beside the GPU's rounded rows on a bf16 weight read as published** (`gpu+ane/bf16act`): offered only where the model
    /// has that pair's own sharing — the engines' `gpu+ane` was measured on `gpu/fp32` and is not its κ —, chosen when faster, with the
    /// integers the engines run; never on an encoder, a process without the ANE, under the ANE's floor, or with the rounding refused.
    @Test func theANESharesTheRoundedBF16WeightOnlyByItsOwnSharing() throws {
        let rounded = M.Kernel(.gpu, .bf16act)
        var m = machine(aneRate: 6e12)
        m.lines[rounded] = .init(t0: 0, rate: 12e12)
        m.preparation[.bf16] = [.widened: 1e-10, .bf16: 1e-11]
        let bf16 = key(4096, 4096, 4096, .bf16)
        #expect(try solve(m, bf16).route == "gpu/bf16act", "no sharing of its own: the GPU's rounded kernel alone")
        // 6 / (12 + 6) of 4 096 rows = 1 365 → the units around it, 1 280 (5 units) and 1 536 (6): 1 280 ends both sides nearest
        m.sharing["gpu/bf16act+ane/fp16"] = .init(overhead: 0, keep: [.gpu: 1, .ane: 1])
        var s = try solve(m, bf16)
        #expect(s.route == "gpu+ane/bf16act" && s.assignment.path == .bf16 && s.assignment.roundsActivation, "\(s.route)")
        #expect(s.assignment.ane % NeuralEngine.rowUnit == 0 && s.assignment.ane > 0 && s.assignment.gpu + s.assignment.ane == 4096)
        #expect(s.rows == [rounded: s.assignment.gpu, ane: s.assignment.ane])
        #expect(abs(s.preparation - 4096 * 4096 * 1e-11) < 1e-9, "the bf16 copy, not a widening")
        // the pair's answer is the plan's, and the assigned product is timed the same
        #expect(CapacityPlan(model: m).assign(bf16) == s.assignment)
        let timed = try #require(CS.product(assigned: s.assignment, key: bf16, model: m, constraints: constraints()))
        #expect(timed.route == s.route && abs(timed.compute - s.compute) < 1e-12)
        // a keep that ruins it: the GPU's rounded kernel alone again
        m.sharing["gpu/bf16act+ane/fp16"] = .init(overhead: 0, keep: [.gpu: 0.3, .ane: 0.3])
        #expect(try solve(m, bf16).route == "gpu/bf16act")
        m.sharing["gpu/bf16act+ane/fp16"] = .init(overhead: 0, keep: [.gpu: 1, .ane: 1])
        // never an encoder, never without the ANE, never under its floor, never with the rounding refused
        #expect(try solve(m, key(4096, 4096, 4096, .bf16, phase: .encoding)).route == "gpu")
        #expect(try solve(m, bf16, ane: false).route == "gpu/bf16act")
        #expect(try solve(m, key(4096, 4096, 512, .bf16)).route == "gpu/bf16act")
        s = try #require(CS.product(bf16, model: m, constraints: CS.Constraints(sustained: false, allowsRounding: false)))
        #expect(!s.assignment.roundsActivation && s.route != "gpu+ane/bf16act")
        // an ANE the judge refused is no candidate
        let refused = try #require(CS.product(bf16, model: m, constraints: .init(sustained: false, excluded: [ane], allowsANE: true)))
        #expect(refused.route == "gpu/bf16act")
    }

    // MARK: attention

    @Test func theHeadsOfAnAttentionAreBalancedLikeAProduct() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        let s = try solve(m, attentionKey(4096))
        #expect(s.assignment == AttentionAssignment(kernel: .split, cpuHeads: 8))
        #expect(s.route == "split" && s.rows[M.Kernel(.cpu, .heads)] == 8)
        #expect(abs(s.compute - 4.0 * 4096 * 4096 * 128 * 24 / 1.5e12) < 1e-9)
    }

    @Test func aGroupedAttentionIsCutByWholeGroups() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        let s = try solve(m, attentionKey(4096, heads: 48, kv: 12))
        #expect(s.assignment.cpuHeads > 0 && s.assignment.cpuHeads % 4 == 0 && s.assignment.cpuHeads < 48)
    }

    @Test func flashIsTakenOnlyWhereTheCallerTriesIt() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.gpu, .flash)] = .init(t0: 0, rate: 3e12)
        let tried = try solve(m, attentionKey(4096, heads: 30), flash: true)
        #expect(tried.assignment == .flash && tried.route == "flash")
        let untried = try solve(m, attentionKey(4096, heads: 30), flash: false)
        #expect(untried.assignment == .graph)
    }

    @Test func aCrossAttentionIsItsOwnKey() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        let own = attentionKey(4096, heads: 16, dim: 128), cross = attentionKey(4096, heads: 16, dim: 128, cross: true, keys: 512)
        #expect(own != cross)
        let a = try solve(m, own), b = try solve(m, cross)
        // eight times fewer keys: the same cut of heads, a tenth of the time
        #expect(a.assignment.cpuHeads == b.assignment.cpuHeads)
        #expect(b.compute < a.compute / 7)
    }

    @Test func theSplitStartsAtTheMinimumSequence() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        #expect(try solve(m, attentionKey(HeadSplitAttention.minimumSequence - 1)).assignment == .graph)
        #expect(try solve(m, attentionKey(HeadSplitAttention.minimumSequence)).assignment.splits)
    }

    /// **The attention's pair has no ceiling** (`CapacityFit.overlap`): fitted on the two sequences every tier measures, the cut by heads
    /// stays a candidate on the longer sequences of the large formats — 1536 × 1024 (S = 6 144), 2048² (16 384) —, at the longest one's κ.
    @Test func theAttentionPairIsTimedBeyondTheLongestSequenceMeasured() throws {
        typealias Sizes = Diagnostic.CapacityProbes.Sizes
        var m = machine()
        let graph = M.Kernel(.gpu, .graph), heads = M.Kernel(.cpu, .heads)
        m.lines[graph] = .init(t0: 0, rate: 1.5e12)
        m.lines[heads] = .init(t0: 0, rate: 0.5e12)
        var truth = m
        truth.sharing["gpu+cpu"] = .init(keep: [.gpu: 0.97, .cpu: 0.95])
        let (h, d) = (Sizes.attentionHeads, Sizes.attentionHeadDim)
        let trials = Sizes.pairSequences.map { s -> CapacityFit.Measurements.Trial in
            let (k, rows) = (2 * s * s * d, [graph: h - 10, heads: 10])
            let ends = truth.busy(m: h, k: k, n: 1, rows: rows)
            let sides = rows.sorted { $0.key.engine < $1.key.engine }.map {
                CapacityFit.Measurements.Side(kernel: $0.key.description, rows: $0.value, solo: m.alone($0.key, rows: $0.value, k: k, n: 1)!,
                                              within: ends[$0.key]!)
            }
            return .init(set: "gpu+cpu", shape: [h, k, 1], pass: 2, sides: sides, wall: ends.values.max()!)
        }
        let fit = try #require(CapacityFit.overlap(set: "gpu+cpu", trials: trials, model: m)?.sharing)
        #expect(fit.kFrom == 1024 * 1024 * d && fit.kTo == nil)
        m.sharing["gpu+cpu"] = fit
        for s in [4096, 6144, 16384] { #expect(try solve(m, attentionKey(s, heads: h)).assignment.splits, "S = \(s)") }
    }

    // MARK: a render

    @Test func aRealRequestIsSolvedClassByClassAndEveryAnswerRuns() throws {
        var m = machine(amxRate: 1e12, aneRate: 6e12)
        m.sharing["gpu+ane"] = .init(overhead: 0.0002, keep: [.gpu: 0.8, .ane: 0.7])
        m.sharing["gpu+amx"] = .init(overhead: 0.0001, keep: [.gpu: 0.9, .amx: 0.8])
        m.sharing["gpu+amx+ane"] = .init(overhead: 0.0003, keep: [.gpu: 0.7, .amx: 0.6, .ane: 0.6])
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.gpu, .flash)] = .init(t0: 0, rate: 2.5e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.3e12)
        m.sharing["gpu+cpu"] = .init(overhead: 0.0001, keep: [.gpu: 0.95, .cpu: 0.9])
        m.preparation[.bf16] = [.widened: 1e-10]
        for (family, variant) in [(Family.zImage, Variant.compact), (.qwenImage21, .light), (.anima, .standard), (.krea2, .compact)] {
            let request = CC.Request(family: family, variant: variant, includesEncoder: true)
            let classes = try #require(CC.classes(for: request))
            let plan = try #require(CS.solve(model: m, request: request, constraints: .init(request: request, sustained: false)))
            #expect(plan.unpredicted.isEmpty)
            let run = classes.filter { ($0.executions[family] ?? 0) > 0 }
            #expect(run.count == plan.blocks.count)
            for (c, b) in zip(run, plan.blocks) {
                switch c.shape {
                case let .product(k, n, rows, format):
                    let key = ProductKey(k: k, n: n, m: rows, weight: format, phase: c.phase)
                    let a = try #require(b.product, "\(key)")
                    #expect(a.isRunnable(for: key), "\(family) \(variant) \(key): \(a)")
                    #expect(b.route == a.route(weight: format))
                    // the encoders take no ANE (the Standard does, as a render: 2/11/2026)
                    if c.phase == .encoding { #expect(!a.usesANE, "\(family) \(variant) \(key)") }
                    // the same answer alone, as the plan of a process gives it
                    #expect(CS.product(key, model: m, constraints: .init(request: request, sustained: false))?.assignment == a)
                case let .attention(s, keys, heads, kv, dim, cross):
                    let key = AttentionKey(sequence: s, keys: keys, heads: heads, kvHeads: kv, headDim: dim, cross: cross)
                    let a = try #require(b.attention, "\(key)")
                    #expect(a.isRunnable(for: key))
                    if !CS.triesFlash(family: family, cross: cross) { #expect(!a.usesFlash, "\(family) \(key)") }
                }
            }
            if variant == .standard { #expect(plan.blocks.contains { $0.product?.usesANE == true }, "the Standard's request allows the ANE") }
            #expect(plan.seconds == plan.blocks.reduce(0) { $0 + $1.seconds })
            let data = try JSONEncoder().encode(plan.blocks.compactMap(\.product))
            #expect(try JSONDecoder().decode([ProductAssignment].self, from: data) == plan.blocks.compactMap(\.product))
        }
    }

    @Test func theFullModelIsNeverSlowerThanTheGPUAlone() throws {
        var full = machine(amxRate: 1e12, aneRate: 6e12)
        full.sharing["gpu+ane"] = .init(overhead: 0.0002, keep: [.gpu: 0.8, .ane: 0.7])
        full.sharing["gpu+amx"] = .init(overhead: 0.0001, keep: [.gpu: 0.9, .amx: 0.8])
        full.sharing["gpu+amx+ane"] = .init(overhead: 0.0003, keep: [.gpu: 0.7, .amx: 0.6, .ane: 0.6])
        let request = CC.Request(family: .krea2, variant: .compact)
        let withAll = try #require(CS.solve(model: full, request: request, constraints: .init(request: request, sustained: false)))
        let alone = try #require(CS.solve(model: machine(), request: request, constraints: .init(request: request, sustained: false)))
        #expect(withAll.seconds <= alone.seconds)
        #expect(withAll.seconds < 0.8 * alone.seconds)
        // what the memory need reads agrees with the plan: the ANE's programs exist exactly where a block takes ANE rows
        let aneProducts = withAll.blocks.contains { ($0.product?.ane ?? 0) > 0 }
        #expect(CapacityPlan(model: full).usesANE == true && aneProducts)
        #expect(alone.blocks.allSatisfy { ($0.product?.ane ?? 0) == 0 && ($0.product?.amx ?? 0) == 0 })
    }

    @Test func theOverlapHookReplacesTheSum() throws {
        let longest: CS.Combine = { blocks in blocks.map(\.seconds).max() ?? 0 }
        let classes: [CC.Class] = [
            .init(phase: .denoising, shape: .product(k: 4096, n: 4096, rows: 4096, format: .bf16), executions: [.zImage: 1]),
            .init(phase: .denoising, shape: .product(k: 2048, n: 2048, rows: 4096, format: .bf16), executions: [.zImage: 1])]
        let solved = CS.solve(model: machine(), classes: classes, family: .zImage, constraints: .init(allowsANE: false), combine: longest)
        let plan = try #require(solved)
        #expect(plan.seconds == plan.blocks.map(\.seconds).max())
    }

    // MARK: a set nobody measured

    /// **A set of two engines or more with no sharing fitted is never a candidate** — not taken at keep 1, the best plan on paper: a bench saw the
    /// solver pick the AMX beside the ANE (no GPU) for a Qwen-Image-2.1 step, predicted 8.0 s, measured 10.3 (the pair keeps 0.77–0.88 and 0.91–0.95).
    @Test func anUnmeasuredSetIsNeverACandidate() throws {
        // A slow GPU, a fast AMX and a fast ANE: the AMX beside the ANE is the best set on paper.
        var m = M()
        m.lines[gpu] = .init(t0: 0, rate: 0.5e12)
        m.lines[amx] = .init(t0: 0, rate: 3e12)
        m.lines[ane] = .init(t0: 0, rate: 4e12)
        let k = key(4096, 4096, 4096)
        // Nothing measured but the lines: an engine alone (always measured — here the ANE, the fastest), never two for free.
        #expect(!m.measures([.gpu, .amx]) && !m.measures([.amx, .ane]) && m.measures([.gpu]) && m.measures([]))
        let alone = try solve(m, k)
        #expect(alone.assignment == ProductAssignment(gpu: 0, ane: 4096), "\(alone.assignment)")
        // The pairs with the GPU measured: the AMX beside the ANE is still not offered, nor is the triple.
        m.sharing["gpu+amx"] = .init(overhead: 0, keep: [.gpu: 1, .amx: 1])
        m.sharing["gpu+ane"] = .init(overhead: 0, keep: [.gpu: 1, .ane: 1])
        let pairs = try solve(m, k)
        #expect(pairs.assignment.gpu > 0, "the GPU is in every set that was measured: \(pairs.assignment)")
        #expect(!(pairs.assignment.amx > 0 && pairs.assignment.ane > 0), "\(pairs.assignment)")
        // Measured, the set is a candidate like another — and wins here, with the GPU out of it (the triple is still not measured).
        m.sharing["amx+ane"] = .init(overhead: 0, keep: [.amx: 0.8, .ane: 0.93])
        let measured = try solve(m, k)
        #expect(measured.assignment.gpu == 0 && measured.assignment.amx > 0 && measured.assignment.ane > 0, "\(measured.assignment)")
        // Whatever the machine, no answer of the solver names a set the model did not measure.
        var partial = machine(amxRate: 3e12, aneRate: 4e12)
        partial.sharing["gpu+amx"] = nil; partial.sharing["gpu+amx+ane"] = nil; partial.sharing["amx+ane"] = nil
        for rows in [1024, 1056, 2048, 4096, 4128, 6176] {
            for (kk, nn) in [(3840, 3840), (3840, 10240), (4096, 12288)] {
                let s = try solve(partial, key(kk, nn, rows))
                let engines = s.rows.keys.map(\.engine)
                #expect(partial.measures(engines), "\(rows): \(s.route)")
            }
        }
    }

    /// An assignment someone else chose is timed only on sets that were measured: the render's check says « not predicted », not a guess at keep 1.
    @Test func anAssignmentOnAnUnmeasuredSetIsNotPredicted() {
        var m = machine(amxRate: 3e12, aneRate: 4e12)
        m.sharing["amx+ane"] = nil
        let k = key(4096, 4096, 4096)
        #expect(CS.product(assigned: ProductAssignment(gpu: 0, amx: 2048, ane: 2048), key: k, model: m, constraints: constraints()) == nil)
        #expect(CS.product(assigned: ProductAssignment(gpu: 0, amx: 4096), key: k, model: m, constraints: constraints()) != nil, "an engine alone is always measured")
        m.sharing["amx+ane"] = .init(overhead: 0, keep: [.amx: 0.8, .ane: 0.93])
        #expect(CS.product(assigned: ProductAssignment(gpu: 0, amx: 2048, ane: 2048), key: k, model: m, constraints: constraints()) != nil)
    }

    /// The head split of an attention is the set of the graph and the CPU's heads: offered only once measured (`attentionSet`).
    @Test func theHeadSplitNeedsItsSharing() throws {
        var m = machine()
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        m.sharing["gpu+cpu"] = nil
        #expect(try solve(m, attentionKey(4096)).assignment == .graph)
        #expect(CS.attention(assigned: AttentionAssignment(kernel: .split, cpuHeads: 8), key: attentionKey(4096), model: m, constraints: constraints()) == nil)
        m.sharing["gpu+cpu"] = .init(overhead: 0, keep: [.gpu: 0.95, .cpu: 0.9])
        #expect(try solve(m, attentionKey(4096)).assignment.splits)
    }

    @Test func aModelWithoutAGPULineCannotBePlanned() {
        #expect(CS.product(key(4096, 4096, 4096), model: M(), constraints: .init()) == nil)
        #expect(CS.solve(model: M(), classes: [], family: .zImage, constraints: .init()) == nil)
    }
}
