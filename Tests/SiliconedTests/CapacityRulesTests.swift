import Foundation
import XCTest
@testable import Siliconed

/// **The rules that turn a capacity map into the engine's branches** (`CapacityRules.swift`), on synthetic
/// maps: what they choose, against which margin, and when a map is stale. No GPU, no weights.
final class CapacityRulesTests: XCTestCase {
    /// The rates of one map: per size (1024, 4096) and per model (Z-Image, Qwen-Image-2.1), up then down.
    private struct Rates {
        var gpu: [[Double]]                  // [size][product]: z up, z down, qwen up, qwen down
        var amx: [[Double]]
        var both: [[Double]]
        var graph: [[Double]]                // [size][model]
        var flash: [[Double]]
        var split: [[Double]]
        var drift = (gpu: 1.0, amx: 1.0)
        var amxError = 8e-7
        /// `TensorGEMM` on the bf16 weights, [size][product]; `nil`: no Metal 4.
        var tensor: [[Double]]? = nil
        var tensorError = 6e-7
        /// `TensorGEMM` on an activation rounded to bf16, [size][product], the rounding's pass in its time.
        var bf16Act: [[Double]]? = nil
        /// The CPU's bf16 widening and bf16 copy, G values/s (the M1 Pro's: 21.3 and ~40).
        var widen = 21.3, copy = 40.0
        /// The int8 products, [size][product]: alone, beside the AMX by columns; `nil`: no Metal 4.
        var int8: [[Double]]? = nil, int8Both: [[Double]]? = nil
        /// The CPU's convrot widening and int8 copy, G values/s (the M1 Pro's: 11.3 and 77).
        var rotated = 11.3, copy8 = 77.0
        /// The Q8_0 products, likewise; and the CPU's Q8_0 widening and copy.
        var q8: [[Double]]? = nil, q8Both: [[Double]]? = nil
        var q8Widen = 23.0, q8Copy = 70.0
        /// GPU + ANE, [size][product]; `nil`: a map before schema 10. Its error, threshold (bf16's + 25 %) and share.
        var ane: [[Double]]? = nil
        var aneError = 4e-4, aneThreshold = 4e-3, aneShare = 0.55
    }

    private static let machine = Diagnostic.Machine(chip: "Apple M1 Pro", hardwareModel: "MacBookPro18,3", performanceCores: 6,
                                                    efficiencyCores: 2, gpuCores: 14, memoryBytes: 17_179_869_184,
                                                    macOS: "27.0.1 (26A434)", version: "0.9 (90)", build: "0123456789ab")

    private func report(_ r: Rates, machine: Diagnostic.Machine = CapacityRulesTests.machine) -> Diagnostic.Report {
        func ok(_ rate: Double, _ error: Double = 1e-6) -> Diagnostic.Measure { .init(rate: rate, error: error, threshold: 1e-5) }
        var products: [Diagnostic.Product] = []
        for (b, block) in Diagnostic.blocks.enumerated() {
            for (h, m) in Diagnostic.heights.enumerated() {
                for (p, name) in ["up", "down"].enumerated() {
                    let i = 2 * b + p
                    let (k, n) = name == "up" ? (block.dim, block.hidden) : (block.hidden, block.dim)
                    let (g, a) = (r.gpu[h][i], r.amx[h][i])
                    products.append(.init(model: block.model, product: name, m: m, k: k, n: n, gpu: ok(g),
                                          amx: ok(a, r.amxError), gpuShare: g / (g + a), coexecution: ok(r.both[h][i]),
                                          coexecutionGPURows: nil, tensor: r.tensor.map { ok($0[h][i], r.tensorError) },
                                          bf16Act: r.bf16Act.map { ok($0[h][i], r.tensorError) },
                                          int8: r.int8.map { ok($0[h][i]) }, int8Coexecution: r.int8Both.map { ok($0[h][i]) },
                                          q8: r.q8.map { ok($0[h][i]) }, q8Coexecution: r.q8Both.map { ok($0[h][i]) },
                                          ane: r.ane.map { .init(rate: $0[h][i], error: r.aneError, threshold: r.aneThreshold) },
                                          aneGPUShare: r.ane.map { _ in r.aneShare }))
                }
            }
        }
        var attention: [Diagnostic.AttentionCapacity] = []
        for (b, block) in Diagnostic.blocks.enumerated() {
            for (h, s) in Diagnostic.heights.enumerated() {
                attention.append(.init(model: block.model, heads: block.heads, headDim: block.headDim, sequence: s,
                                       graph: ok(r.graph[h][b]), flash: ok(r.flash[h][b]), headSplit: ok(r.split[h][b])))
            }
        }
        let conditions = Diagnostic.Conditions(thermal: "nominal", lowPowerMode: false, onBattery: false, swapUsedBytes: 0)
        let first = products[0]
        return Diagnostic.Report(date: "2026-10-14T20:11:08Z", seconds: 16.1, machine: machine, cpuFeatures: [:],
                                 before: conditions, after: conditions, products: products, lowPrecision: [],
                                 attention: attention,
                                 widening: [.init(kind: "bf16", values: 39_321_600, cpu: .init(rate: r.widen, error: 0, threshold: 0)),
                                            .init(kind: "bf16 copy", values: 39_321_600, cpu: .init(rate: r.copy, error: 0, threshold: 0)),
                                            .init(kind: "int8 convrot", values: 39_321_600, cpu: .init(rate: r.rotated, error: 0, threshold: 0)),
                                            .init(kind: "int8 copy", values: 39_321_600, cpu: .init(rate: r.copy8, error: 0, threshold: 0)),
                                            .init(kind: "q8_0", values: 39_321_600, cpu: .init(rate: r.q8Widen, error: 0, threshold: 0)),
                                            .init(kind: "q8_0 copy", values: 39_321_600, cpu: .init(rate: r.q8Copy, error: 0, threshold: 0))],
                                 witness: .init(gpuBefore: first.gpu.rate, gpuAfter: first.gpu.rate * r.drift.gpu,
                                                amxBefore: first.amx?.rate, amxAfter: first.amx.map { $0.rate * r.drift.amx }))
    }

    /// **This M1 Pro's map of 14/10/2026** (on battery, the GPU ×0.84 over the run): co-execution and the
    /// head split on, flash off — today's defaults, the GPU's share 0.614 where the frozen one is 0.631.
    private var m1Pro: Rates {
        Rates(gpu: [[3.471, 2.84, 2.863, 3.518], [2.838, 2.887, 2.898, 2.937]],
              amx: [[2.222, 1.625, 2.192, 1.42], [2.139, 1.723, 2.2, 1.572]],
              both: [[3.516, 3.751, 3.883, 3.652], [4.542, 3.793, 4.581, 3.593]],
              graph: [[1.217, 1.569], [1.234, 1.333]], flash: [[1.274, 1.594], [1.3, 1.335]],
              split: [[1.71, 1.862], [2.014, 2.138]], drift: (2.923 / 3.471, 2.02 / 2.222))
    }

    func testM1ProMapGivesTodaysBranches() {
        let d = Diagnostic.decide(report(m1Pro))
        XCTAssertEqual(d.margin, 0.158, accuracy: 0.001)   // the witness's drift, above the 5 % floor
        XCTAssertTrue(d.amx.on)
        XCTAssertTrue(d.amx.passed)
        XCTAssertEqual(d.amx.gains.count, 2)
        XCTAssertEqual(d.amx.gains[0], 1.18, accuracy: 0.01)
        XCTAssertEqual(d.amx.gains[1], 1.41, accuracy: 0.01)
        XCTAssertEqual(d.amxFraction!, 0.614, accuracy: 0.001)
        XCTAssertFalse(d.flash.on, "a tie with MPSGraph stays on MPSGraph")
        XCTAssertTrue(d.headSplit.on)
        XCTAssertEqual(d.headSplit.gain, 1.45, accuracy: 0.01)
        let settings = d.settings
        XCTAssertEqual(settings["amx"] as? Bool, true)
        XCTAssertEqual(settings["flash"] as? Bool, false)
        XCTAssertEqual(settings["head_split"] as? Bool, true)
        XCTAssertEqual(settings["amx_fraction"] as? Double, d.amxFraction)
        XCTAssertEqual(settings["gemm_tensor"] as? Bool, false)
        XCTAssertNotNil(d.tensor)
        XCTAssertFalse(d.tensor!.passed, "no Metal 4 measure: the bf16 path is not a candidate")
    }

    /// **The M1 Pro with its `matmul2d`** (2.36 TFLOP/s on the bf16 weights, measured 17/10/2026): slower
    /// than MPS + AMX even without the widening — the default path stays.
    func testM1ProKeepsTheWidenedPath() {
        var r = m1Pro
        r.tensor = [[2.36, 2.3, 2.36, 2.3], [2.4, 2.35, 2.4, 2.35]]
        let d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.tensor!.passed)
        XCTAssertFalse(d.tensor!.on)
        XCTAssertLessThan(d.tensor!.gain, 1)
        XCTAssertEqual(d.settings["amx"] as? Bool, true)
        XCTAssertEqual(d.settings["gemm_tensor"] as? Bool, false)
    }

    /// **The M5's map of 17/10/2026**, its `matmul2d` at the bf16 × bf16 rate it measured (6.0):
    /// the bf16 path beats MPS + AMX, takes the products, and turns the AMX off in the settings.
    func testM5TakesTheBf16Path() {
        var r = m1Pro
        r.gpu = [[2.919, 2.831, 2.881, 2.847], [2.951, 2.872, 2.949, 2.835]]
        r.amx = [[1.805, 1.75, 1.812, 1.768], [1.838, 1.829, 1.828, 1.842]]
        r.both = [[3.35, 3.071, 3.353, 3.16], [4.569, 4.259, 4.579, 4.313]]
        r.drift = (2.924 / 2.919, 1.814 / 1.805)
        r.tensor = [[6.0, 5.9, 6.0, 5.9], [6.0, 5.9, 6.0, 5.9]]
        r.widen = 20.07
        r.copy = 35
        let d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.amx.on, "co-execution still measured a gain")
        XCTAssertTrue(d.tensor!.on)
        XCTAssertGreaterThan(d.tensor!.gains.min()!, 1.3)
        let settings = d.settings
        XCTAssertEqual(settings["gemm_tensor"] as? Bool, true)
        XCTAssertEqual(settings["amx"] as? Bool, false, "the bf16 path has no AMX")
        XCTAssertNil(settings["amx_fraction"])
    }

    /// **The M5 Pro's map of 20/10/2026** (GPU 16 cores): MPS fp32 5.8–6.0, `matmul2d` on the bf16 weights with the
    /// fp32 activation 6.0–6.3 (no gain), `bfloat × bfloat → float` 15.28 TFLOP/s, ×0.97 for the rounding's pass.
    private var m5Pro: Rates {
        var r = m1Pro
        r.gpu = [[5.983, 5.802, 5.9, 5.8], [5.9, 5.8, 5.9, 5.76]]
        r.amx = [[1.817, 1.63, 1.8, 1.7], [1.8, 1.7, 1.8, 1.19]]
        r.both = [[5.947, 5.28, 5.9, 5.3], [6.5, 6.0, 6.4, 5.5]]
        r.drift = (1.0, 1.0)
        r.tensor = [[6.322, 6.002, 6.3, 6.0], [6.3, 6.0, 6.3, 6.0]]
        r.widen = 39.8; r.copy = 59.93
        return r
    }

    /// **The activation rounded to bf16, decided by the map and not by the chip's name**: the M5 Pro's
    /// bf16 × bf16 rate turns it on, at more than twice the fp32 path, where `gemm_tensor` (the same weight, an
    /// fp32 activation: 6.0–6.3) is within a few per cent of it.
    func testM5ProTakesTheRoundedActivation() {
        var r = m5Pro
        r.bf16Act = r.gpu.map { $0.map { _ in 15.28 * 0.97 } }
        let d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.bf16Act!.passed)
        XCTAssertTrue(d.bf16Act!.on)
        XCTAssertGreaterThan(d.bf16Act!.gains.min()!, 2.0)
        XCTAssertEqual(d.settings["gemm_bf16_act"] as? Bool, true)
    }

    /// **The M1 Pro**: `matmul2d` bf16 × bf16 at 1.83 TFLOP/s before the rounding's pass, against
    /// MPS + AMX at 3.5–4.5 — off, whatever the margin.
    func testM1ProKeepsTheFp32Activation() {
        var r = m1Pro
        r.bf16Act = [[1.7, 1.7, 1.7, 1.7], [1.8, 1.8, 1.8, 1.8]]
        let d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.bf16Act!.passed)
        XCTAssertFalse(d.bf16Act!.on)
        XCTAssertLessThan(d.bf16Act!.gain, 1)
        XCTAssertEqual(d.settings["gemm_bf16_act"] as? Bool, false)
    }

    /// A map without the measure (before schema 11, or no Metal 4): no decision, the branch off; and a measure
    /// that fails its check is no candidate however fast.
    func testTheRoundedActivationNeedsItsMeasureAndItsCheck() {
        var r = m5Pro
        var d = Diagnostic.decide(report(r))
        XCTAssertNil(d.bf16Act)
        XCTAssertEqual(d.settings["gemm_bf16_act"] as? Bool, false)
        r.bf16Act = r.gpu.map { $0.map { _ in 14.8 } }
        r.tensorError = 3e-5                                    // over the fp32 threshold: a fault, not a rounding
        d = Diagnostic.decide(report(r))
        XCTAssertFalse(d.bf16Act!.passed)
        XCTAssertFalse(d.bf16Act!.on)
    }

    /// **Where the rounded rows lie**: after the weight's pages, in the reserve's tail, a multiple of 64 rows.
    func testTheRoundingPlanFitsTheReserve() throws {
        // Z-Image's w1 in a reserve of dim · hidden floats: the bf16 weight takes the first half.
        let (k, n) = (3840, 10240)
        let plan = try XCTUnwrap(WeightProducts.roundingPlan(k: k, n: n, capacity: k * n))
        XCTAssertEqual(plan.offset % 16384, 0)
        XCTAssertGreaterThanOrEqual(plan.offset, 2 * k * n)
        XCTAssertEqual(plan.rows % 64, 0)
        XCTAssertLessThanOrEqual(plan.offset + plan.rows * k * 2, k * n * 4)
        XCTAssertGreaterThanOrEqual(plan.rows, 64)
        // A reserve with no room beyond the weight, or an odd k: the fp32 operand.
        XCTAssertNil(WeightProducts.roundingPlan(k: k, n: n, capacity: k * n / 2 + 1000))
        XCTAssertNil(WeightProducts.roundingPlan(k: 3841, n: n, capacity: 3841 * n))
        // The 8-bit plan: rotated rows (fp32) then rounded rows (bf16), both inside the reserve.
        let eight = try XCTUnwrap(WeightProducts.int8Plan(k: 4096, n: 4096, capacity: 4096 * 4096, perColumn: true, group: 256,
                                                          amxColumns: 0, rounded: true))
        XCTAssertGreaterThan(eight.roundedOffset, eight.rotatedOffset)
        XCTAssertLessThanOrEqual(eight.roundedOffset + eight.chunk * 4096 * 2, 4096 * 4096 * 4)
        XCTAssertLessThanOrEqual(eight.rotatedOffset + eight.chunk * 4096 * 4, eight.roundedOffset)
    }

    /// **The preparation counts**: at the GPU's own rate, the bf16 path wins by the widening it skips —
    /// and loses it again when the copy is no faster than the widening.
    func testThePreparationIsInTheComparison() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.amx = r.gpu.map { $0.map { $0 * 0.1 } }              // an AMX that does not pay: `amx` off
        r.both = r.gpu
        r.tensor = r.gpu
        r.widen = 2; r.copy = 20                               // a slow widening: ×1.1–1.4 on the preparations alone
        var d = Diagnostic.decide(report(r))
        XCTAssertFalse(d.amx.on)
        XCTAssertTrue(d.tensor!.on)
        r.copy = r.widen
        d = Diagnostic.decide(report(r))
        XCTAssertEqual(d.tensor!.gain, 1, accuracy: 1e-9)
        XCTAssertFalse(d.tensor!.on)
    }

    /// **This M1 Pro's map of 18/10/2026**: the int8 GPU alone loses to MPS + AMX (2.4 against
    /// ~4.5 TFLOP/s), beside the AMX by columns it wins at 1024 rows (×1.19) and ties at 4096 — on, the
    /// AMX kept; and the same int8 product alone would stay off.
    func testM1ProTakesTheInt8PathBesideTheAMX() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.gpu = [[3.464, 3.509, 3.457, 3.507], [3.507, 3.603, 3.490, 3.599]]
        r.amx = [[2.232, 1.932, 2.242, 1.794], [2.393, 2.009, 2.404, 1.853]]
        r.both = [[3.985, 4.482, 3.990, 4.518], [5.481, 4.926, 5.080, 5.056]]
        r.tensor = [[2.309, 2.316, 2.319, 2.313], [2.334, 2.338, 2.344, 2.330]]
        r.int8 = [[2.4, 2.4, 2.4, 2.4], [2.45, 2.45, 2.45, 2.45]]
        r.int8Both = [[4.734, 4.122, 4.700, 4.018], [5.276, 4.547, 5.378, 4.609]]
        var d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.amx.on)
        XCTAssertFalse(d.tensor!.on)
        XCTAssertTrue(d.int8!.on)
        XCTAssertGreaterThan(d.int8!.gains[0], 1.1)
        XCTAssertEqual(d.settings["gemm_int8"] as? Bool, true)
        XCTAssertEqual(d.settings["amx"] as? Bool, true, "the int8 path keeps the AMX on its columns")
        r.int8Both = nil
        d = Diagnostic.decide(report(r))
        XCTAssertFalse(d.int8!.passed, "the AMX stays on: the int8 path is judged beside it, unmeasured here")
        XCTAssertEqual(d.settings["gemm_int8"] as? Bool, false)
    }

    /// **The Q8_0 product is decided apart from the int8 one** (this M1 Pro, 18/10/2026): beside the AMX it runs
    /// ×0.45–0.67 of the widened path where the int8 one wins — `gemm_q8` off, `gemm_int8` on, from one map.
    func testM1ProKeepsTheQ8PathOffWhereTheInt8OneWins() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.gpu = [[3.464, 3.509, 3.457, 3.507], [3.507, 3.603, 3.490, 3.599]]
        r.amx = [[2.232, 1.932, 2.242, 1.794], [2.393, 2.009, 2.404, 1.853]]
        r.both = [[3.985, 4.482, 3.990, 4.518], [5.481, 4.926, 5.080, 5.056]]
        r.tensor = [[2.309, 2.316, 2.319, 2.313], [2.334, 2.338, 2.344, 2.330]]
        r.int8 = [[2.4, 2.4, 2.4, 2.4], [2.45, 2.45, 2.45, 2.45]]
        r.int8Both = [[4.734, 4.122, 4.700, 4.018], [5.276, 4.547, 5.378, 4.609]]
        r.q8 = [[1.2, 1.2, 1.2, 1.2], [1.25, 1.25, 1.25, 1.25]]
        r.q8Both = [[2.1, 2.0, 2.1, 2.0], [2.4, 2.3, 2.4, 2.3]]
        var d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.int8!.on)
        XCTAssertTrue(d.q8!.passed)
        XCTAssertFalse(d.q8!.on)
        XCTAssertLessThan(d.q8!.gains.max()!, 1)
        XCTAssertEqual(d.settings["gemm_q8"] as? Bool, false)
        XCTAssertEqual(d.settings["gemm_int8"] as? Bool, true)
        // A chip where it wins beside the AMX takes it.
        r.q8Both = r.both.map { $0.map { $0 * 1.4 } }
        d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.q8!.on)
        XCTAssertEqual(d.settings["gemm_q8"] as? Bool, true)
        // Without its widening's rate, never.
        r.q8Widen = 0
        XCTAssertFalse(Diagnostic.decide(report(r)).q8!.on)
    }

    /// **The Neural Engine is a branch the map decides, exclusive with the AMX**: when both clear the margin,
    /// `gemm_ane` on with its share and `amx` off down to 90 % of the AMX's gain (thermal efficiency); below
    /// it the AMX; a product over its threshold, a gain within the margin, or a map without it, keeps it off.
    func testTheNeuralEngineIsDecidedAgainstTheAMX() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.ane = r.gpu.map { $0.map { $0 * 1.7 } }
        var d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.ane!.on)
        XCTAssertTrue(d.amx.passed)
        XCTAssertFalse(d.amx.on, "the larger gain keeps its branch")
        XCTAssertEqual(d.settings["gemm_ane"] as? Bool, true)
        XCTAssertEqual(d.settings["amx"] as? Bool, false)
        XCTAssertEqual(d.settings["ane_fraction"] as? Double, 0.55)
        XCTAssertNil(d.settings["amx_fraction"])
        // The AMX faster, the ANE at 93 % of its rates: the ANE all the same.
        r.ane = r.both.map { $0.map { $0 * 0.93 } }
        d = Diagnostic.decide(report(r))
        XCTAssertLessThan(d.ane!.gain, d.amx.gain)
        XCTAssertTrue(d.ane!.on)
        XCTAssertFalse(d.amx.on)
        XCTAssertEqual(d.settings["amx"] as? Bool, false)
        // At 85 %: the AMX, clearly faster.
        r.ane = r.both.map { $0.map { $0 * 0.85 } }
        d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.ane!.passed)
        XCTAssertFalse(d.ane!.on)
        XCTAssertTrue(d.amx.on)
        // Within the margin of the GPU alone: off, and the AMX keeps the rows.
        r.ane = r.gpu.map { $0.map { $0 * 1.02 } }
        d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.ane!.passed)
        XCTAssertFalse(d.ane!.on)
        XCTAssertTrue(d.amx.on)
        XCTAssertEqual(d.settings["gemm_ane"] as? Bool, false)
        XCTAssertNil(d.settings["ane_fraction"])
        // Faster, but one product over bf16's error + 25 %: off.
        r.ane = r.gpu.map { $0.map { $0 * 1.7 } }
        r.aneError = 5e-3
        d = Diagnostic.decide(report(r))
        XCTAssertFalse(d.ane!.passed)
        XCTAssertFalse(d.ane!.on)
        XCTAssertTrue(d.amx.on)
        // A map before schema 10: no decision, the branch off.
        r.ane = nil
        d = Diagnostic.decide(report(r))
        XCTAssertNil(d.ane)
        XCTAssertEqual(d.settings["gemm_ane"] as? Bool, false)
    }

    /// **Under `gemm_tensor` the AMX is off**: the int8 product is judged alone, against the GPU alone.
    func testInt8IsJudgedAloneWhenTheBf16PathTurnsTheAMXOff() {
        var r = m1Pro
        r.tensor = r.gpu.map { $0.map { $0 * 3 } }
        r.int8 = r.gpu.map { $0.map { $0 * 3 } }
        r.int8Both = r.gpu.map { $0.map { $0 * 0.1 } }        // ignored: no AMX beside it
        let d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.tensor!.on)
        XCTAssertTrue(d.int8!.on)
        XCTAssertEqual(d.settings["amx"] as? Bool, false)
    }

    /// A bf16 path that computed wrong, or a map without the copy's rate, is never taken.
    func testAWrongOrUnmeasuredBf16PathStaysOff() {
        var r = m1Pro
        r.tensor = r.gpu.map { $0.map { $0 * 3 } }
        XCTAssertTrue(Diagnostic.decide(report(r)).tensor!.on)
        r.tensorError = 2e-3
        XCTAssertFalse(Diagnostic.decide(report(r)).tensor!.on)
        r.tensorError = 6e-7
        var noCopy = report(r)
        noCopy.widening.removeAll { $0.kind == "bf16 copy" }
        XCTAssertFalse(Diagnostic.decide(noCopy).tensor!.passed)
        XCTAssertEqual(Diagnostic.decide(noCopy).settings["gemm_tensor"] as? Bool, false)
    }

    /// **An M5-like chip**: the AMX nearly as fast as the GPU (an M5, an M4). The co-execution
    /// nearly doubles the products; the GPU's share falls near one half.
    func testAMXCloseToTheGPUTurnsCoexecutionOn() {
        var r = m1Pro
        r.gpu = [[2.0, 2.0, 2.0, 2.0], [2.1, 2.1, 2.1, 2.1]]
        r.amx = [[1.8, 1.8, 1.8, 1.8], [1.85, 1.85, 1.85, 1.85]]
        r.both = [[3.5, 3.5, 3.5, 3.5], [3.8, 3.8, 3.8, 3.8]]
        r.drift = (0.99, 1.0)
        let d = Diagnostic.decide(report(r))
        XCTAssertEqual(d.margin, Diagnostic.minimumMargin)
        XCTAssertTrue(d.amx.on)
        XCTAssertEqual(d.amxFraction!, 2.05 / (2.05 + 1.825), accuracy: 0.01)
        XCTAssertLessThan(d.amxFraction!, 0.6)
    }

    /// **An engine that computed wrong is never chosen**, however fast.
    func testAFailedAMXCheckKeepsCoexecutionOff() {
        var r = m1Pro
        r.amxError = 3e-3
        let d = Diagnostic.decide(report(r))
        XCTAssertFalse(d.amx.passed)
        XCTAssertFalse(d.amx.on)
        XCTAssertNil(d.amxFraction)
        XCTAssertNil(d.settings["amx_fraction"])
        XCTAssertEqual(d.settings["amx"] as? Bool, false)
        // The same for a wrong co-execution, the AMX alone right.
        var wrong = report(m1Pro)
        wrong.products[3].coexecution = .init(rate: 9, error: 1e-2, threshold: 1e-5)
        XCTAssertFalse(Diagnostic.decide(wrong).amx.on)
        // And a missing engine: BNNS refused one shape.
        var missing = report(m1Pro)
        missing.products[5].amx = nil
        XCTAssertFalse(Diagnostic.decide(missing).amx.on)
        var noSplit = report(m1Pro)
        noSplit.attention[2].headSplit = nil
        XCTAssertFalse(Diagnostic.decide(noSplit).headSplit.on)
    }

    /// **A gain within the margin does not count**, and the margin is the map's own drift when larger.
    func testATieWithinTheMarginStaysOnTheGPUAlone() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.both = r.gpu.map { $0.map { $0 * 1.04 } }        // +4 %: under the 5 % floor
        r.flash = r.graph.map { $0.map { $0 * 1.06 } }     // +6 %: over it
        var d = Diagnostic.decide(report(r))
        XCTAssertEqual(d.margin, 0.05)
        XCTAssertFalse(d.amx.on)
        XCTAssertNil(d.amxFraction)
        XCTAssertTrue(d.amx.passed)
        XCTAssertTrue(d.flash.on == false, "the split (×1.45) is faster: Z-Image keeps it")
        r.split = r.graph                                    // no split gain: flash's +6 % now decides
        d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.flash.on)
        XCTAssertFalse(d.headSplit.on)
        r.drift = (0.92, 1.0)                                // a noisy map: 8 % of drift swallows the 6 %
        d = Diagnostic.decide(report(r))
        XCTAssertEqual(d.margin, 0.08, accuracy: 0.001)
        XCTAssertFalse(d.flash.on)
    }

    /// **A gain at 1024² does not buy a loss at 512²**: one size losing beyond the noise keeps it off.
    func testOneSizeLosingKeepsTheBranchOff() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.both = [r.gpu[0].map { $0 * 0.9 }, r.gpu[1].map { $0 * 1.8 }]   // geometric mean ×1.27, but 512² −10 %
        let d = Diagnostic.decide(report(r))
        XCTAssertGreaterThan(d.amx.gain, 1.2)
        XCTAssertFalse(d.amx.on)
    }

    /// Both attention paths winning: the split stays for Qwen-Image-2.1, flash only where it is the faster.
    func testFlashAndTheSplitBothWinning() {
        var r = m1Pro
        r.drift = (1.0, 1.0)
        r.flash = r.graph.map { $0.map { $0 * 2 } }
        let d = Diagnostic.decide(report(r))
        XCTAssertTrue(d.flash.on)
        XCTAssertTrue(d.headSplit.on)
    }

    /// The report carries its decisions — what a shared JSON says the Mac chose — and reads back.
    func testTheReportCarriesItsDecisions() throws {
        let original = report(m1Pro)
        XCTAssertEqual(original.decisions, Diagnostic.decide(original))
        let text = String(decoding: try original.json(pretty: false), as: UTF8.self)
        XCTAssertTrue(text.contains("\"decisions\":{\"amx\":{"), text)
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Report.self, from: try original.json()), original)
        // A report written before the decisions still reads, and the rules still decide on it.
        var older = try JSONSerialization.jsonObject(with: try original.json()) as! [String: Any]
        older.removeValue(forKey: "decisions")
        let read = try JSONDecoder().decode(Diagnostic.Report.self, from: try JSONSerialization.data(withJSONObject: older))
        XCTAssertNil(read.decisions)
        XCTAssertEqual(Diagnostic.decide(read), original.decisions)
    }

    // ── the fingerprint ─────────────────────────────────────────────────────────────────────

    /// **A map is this process's only if nothing of the fingerprint moved**: hardware, chip, macOS and its
    /// build, the binary's version and its stamp.
    func testAStaleFingerprintIsNotCurrent() {
        let now = Diagnostic.Fingerprint.current
        var machine = Self.machine
        (machine.hardwareModel, machine.chip, machine.macOS, machine.version, machine.build)
            = (now.hardwareModel, now.chip, now.macOS, now.version, now.build)
        let fresh = report(m1Pro, machine: machine)
        XCTAssertEqual(fresh.fingerprint, now)
        XCTAssertTrue(fresh.isCurrent)
        let changes: [(String, (inout Diagnostic.Machine) -> Void)] = [
            ("hardware", { $0.hardwareModel += "x" }), ("chip", { $0.chip = "Apple M5" }),
            ("macOS build", { $0.macOS += "a" }), ("version", { $0.version += ".1" }), ("build", { $0.build = "000000000000" }),
        ]
        for (what, change) in changes {
            var other = machine
            change(&other)
            XCTAssertFalse(report(m1Pro, machine: other).isCurrent, what)
        }
        // Not the memory, nor the GPU's figures: they do not change what a branch costs on one chip.
        var sameChip = machine
        sameChip.memoryBytes *= 2
        XCTAssertTrue(report(m1Pro, machine: sameChip).isCurrent)
        var otherSchema = fresh
        otherSchema.schema = Diagnostic.schema + 1
        XCTAssertFalse(otherSchema.isCurrent)
    }

    // ── the GPU's share per shape (`cuts` → `amx_shares` → `Conductor.Shares`) ───────────────────────

    /// **A map with its grid gives each product its own share**: `ρ = (1 − T*) / T*` at every point,
    /// handed to the engine as text with MPS's row tile as quantum; the share read back at a measured
    /// point is the point's.
    func testCutsGiveEachShapeItsShare() throws {
        var r = report(m1Pro)
        r.cuts = Diagnostic.cutDepths.flatMap { k in Diagnostic.heights.map { m in
            Diagnostic.Cut(k: k, n: Diagnostic.cutWidth, m: m, share: k < 8192 ? 0.62 : 0.69) } }
        let d = Diagnostic.decide(r)
        let text = try XCTUnwrap(d.amxShares)
        XCTAssertEqual(d.settings["amx_shares"] as? String, text)
        XCTAssertEqual(d.settings["amx_quantum"] as? Int, 64)
        let shares = try XCTUnwrap(Conductor.Shares(text: text))
        XCTAssertEqual(shares.points.count, Diagnostic.cutDepths.count * Diagnostic.heights.count)
        XCTAssertEqual(shares.share(k: 4096, n: 6144, m: 1024)!, 0.62, accuracy: 0.001)
        XCTAssertEqual(shares.share(k: 16384, n: 2048, m: 4096)!, 0.69, accuracy: 0.001)
        XCTAssertNil(Diagnostic.decide(report(m1Pro)).amxShares, "no grid: the scalar share serves")
    }

    /// **The ANE's grid gives each product its own share beside the ANE** (`aneCuts` → `ane_shares`), read back
    /// through `NeuralEngine.shareRange` — and only when `gemm_ane` is on.
    func testANECutsGiveEachShapeItsShare() throws {
        var rates = m1Pro
        rates.drift = (1.0, 1.0)
        rates.ane = rates.both.map { $0.map { $0 * 1.05 } }
        var r = report(rates)
        r.aneCuts = Diagnostic.aneShapes.flatMap { s in Diagnostic.heights.map { m in
            Diagnostic.Cut(k: s.k, n: s.n, m: m, share: m == 1024 ? 0.9 : 0.6) } }
        let d = Diagnostic.decide(r)
        XCTAssertTrue(d.ane!.on)
        let text = try XCTUnwrap(d.aneShares)
        XCTAssertEqual(d.settings["ane_shares"] as? String, text)
        let shares = try XCTUnwrap(Conductor.Shares(text: text))
        XCTAssertEqual(shares.share(k: 3840, n: 10240, m: 1024, range: NeuralEngine.shareRange)!, 0.9, accuracy: 0.001)
        XCTAssertEqual(shares.share(k: 3840, n: 10240, m: 4096, range: NeuralEngine.shareRange)!, 0.6, accuracy: 0.001)
        // The ANE off: no grid in the settings.
        rates.aneError = 1
        r = report(rates)
        r.aneCuts = [Diagnostic.Cut(k: 2048, n: 6144, m: 1024, share: 0.6)]
        XCTAssertNil(Diagnostic.decide(r).aneShares)
        XCTAssertNil(Diagnostic.decide(r).settings["ane_shares"])
    }

    /// **The co-execution off, no grid in the settings**: the shares are the conductor's, and there is none.
    func testCutsWithoutCoexecutionGiveNoShares() {
        var rates = m1Pro
        rates.amxError = 1e-3
        var off = report(rates)
        off.cuts = [Diagnostic.Cut(k: 2048, n: 6144, m: 1024, share: 0.6)]
        let d = Diagnostic.decide(off)
        XCTAssertFalse(d.amx.on)
        XCTAssertNil(d.amxShares)
        XCTAssertNil(d.settings["amx_shares"])
    }

    /// **The formula, pure**: two axes through the square point, `ρ(k, n) = ρ(k, w) · ρ(w, n) / ρ(w, w)`, each
    /// read linearly in `log k` or `log n` (the geometric middle gives the mean ratio), then in `log m`; the
    /// nearest point outside the grid; clamped to `Conductor.fractionRange`; a grid without its square point
    /// reads `k` only; the text reads back; an empty grid gives no share.
    /// **With walls, the ANE and the AMX share the shapes**: a shape goes to the ANE where its pair beats the
    /// GPU alone and keeps 90 % of the AMX's speed, to the AMX elsewhere (ρ = 0: the GPU's whole share); both
    /// stay on. No shape the ANE's: the ANE off.
    func testEachShapeGoesToItsEngine() throws {
        var rates = m1Pro
        rates.drift = (1.0, 1.0)
        rates.ane = rates.both.map { $0.map { $0 * 1.05 } }
        var r = report(rates)
        // 3840 → 10240: the ANE at 95 % of the AMX (kept); 16384 → 6144: the ANE beaten (the AMX's).
        r.aneCuts = [Diagnostic.Cut(k: 3840, n: 10240, m: 4096, share: 0.56, ane: 63, gpu: 94, amx: 60),
                     Diagnostic.Cut(k: 16384, n: 6144, m: 4096, share: 0.83, ane: 317, gpu: 273, amx: 183)]
        let d = Diagnostic.decide(r)
        XCTAssertTrue(d.ane!.on); XCTAssertTrue(d.amx.on)
        XCTAssertEqual(d.settings["amx"] as? Bool, true); XCTAssertEqual(d.settings["gemm_ane"] as? Bool, true)
        let shares = try XCTUnwrap(Conductor.Shares(text: try XCTUnwrap(d.aneShares)))
        XCTAssertEqual(shares.share(k: 3840, n: 10240, m: 4096, range: NeuralEngine.shareRange)!, 0.56, accuracy: 0.001)
        XCTAssertEqual(shares.share(k: 16384, n: 6144, m: 4096, range: NeuralEngine.shareRange)!, 1)
        XCTAssertEqual(NeuralEngine(gpuShare: 0.5, shares: shares).rows(4096, k: 16384, n: 6144), 0, "the AMX's shape")
        // At 85 % of the AMX: the AMX's too — no shape left, the ANE off.
        r.aneCuts![0].ane = 71
        XCTAssertFalse(Diagnostic.decide(r).ane!.on)
        // The AMX off (its measure failed): the ANE takes every shape its pair wins against the GPU alone.
        rates.amxError = 1e-3
        r = report(rates)
        r.aneCuts = [Diagnostic.Cut(k: 3840, n: 10240, m: 4096, share: 0.56, ane: 71, gpu: 94, amx: 60)]
        XCTAssertTrue(Diagnostic.decide(r).ane!.on)
        XCTAssertFalse(Diagnostic.decide(r).amx.on)
    }

    /// **The three engines' split from their pairs**: rows in proportion to each engine's rate — the M1 Pro's
    /// 3072 → 16384 at 4 096 rows (pairs 0.45 and 0.65) leaves the GPU's side 55.7 %; an AMX that takes nothing
    /// gives back the ANE's pair.
    func testThreeWayShareFollowsBothPairs() {
        XCTAssertEqual(Diagnostic.threeWayShare(ane: 0.45, amx: 0.65), 0.5573, accuracy: 1e-4)
        XCTAssertEqual(Diagnostic.threeWayShare(ane: 0.6, amx: 1), 0.6, accuracy: 1e-12)
    }

    /// **All three where they beat both pairs beyond the noise** (`ane_amx`): the shape's ANE share becomes the
    /// three's, the setting marks it (ρ 1), and the other shapes keep their pair; three that do not clear the margin
    /// leave no setting. Whichever pair the shape would have gone to.
    func testThreeEnginesTakeTheShapesWhereTheyAddUp() throws {
        var rates = m1Pro
        rates.drift = (1.0, 1.0)
        rates.ane = rates.both.map { $0.map { $0 * 1.05 } }
        var r = report(rates)
        r.cuts = Diagnostic.cutDepths.flatMap { k in Diagnostic.heights.map { m in
            Diagnostic.Cut(k: k, n: Diagnostic.cutWidth, m: m, share: 0.65) } }
        // 3072 → 16384: the three 8 % under the ANE's pair (M1 Pro, 26/10/2026); 8192 → 3072: the three slower (the ANE's pair).
        r.aneCuts = [Diagnostic.Cut(k: 3072, n: 16384, m: 4096, share: 0.45, ane: 65.0, gpu: 120, amx: 76.1, three: 59.6),
                     Diagnostic.Cut(k: 8192, n: 3072, m: 4096, share: 0.45, ane: 31.4, gpu: 58.8, amx: 43.7, three: 35.7)]
        let d = Diagnostic.decide(r)
        XCTAssertTrue(d.ane!.on); XCTAssertTrue(d.amx.on)
        let text = try XCTUnwrap(d.aneAMX)
        XCTAssertEqual(d.settings["ane_amx"] as? String, text)
        let engine = NeuralEngine(gpuShare: 0.5, shares: try XCTUnwrap(Conductor.Shares(text: try XCTUnwrap(d.aneShares))))
        engine.besideAMX = try XCTUnwrap(Conductor.Shares(text: text))
        XCTAssertTrue(engine.takesAMX(4096, k: 3072, n: 16384))
        XCTAssertFalse(engine.takesAMX(4096, k: 8192, n: 3072))
        let shares = engine.shares
        XCTAssertEqual(shares.share(k: 3072, n: 16384, m: 4096, range: NeuralEngine.shareRange)!,
                       Diagnostic.threeWayShare(ane: 0.45, amx: 0.65), accuracy: 0.001)
        XCTAssertEqual(shares.share(k: 8192, n: 3072, m: 4096, range: NeuralEngine.shareRange)!, 0.45, accuracy: 0.001)
        // Three within the margin of the best pair: no shape for them, no setting.
        r.aneCuts![0].three = 63
        XCTAssertNil(Diagnostic.decide(r).aneAMX)
        XCTAssertNil(Diagnostic.decide(r).settings["ane_amx"])
        // The ANE's pair out of its tolerance against the AMX's, the three still faster than both: the three.
        r.aneCuts![0] = Diagnostic.Cut(k: 3072, n: 16384, m: 4096, share: 0.45, ane: 90, gpu: 120, amx: 76.1, three: 65)
        XCTAssertNotNil(Diagnostic.decide(r).aneAMX)
    }

    /// The walls survive the report's JSON, after the share; a cut without them reads as before.
    func testCutWallsRoundTrip() throws {
        let cut = Diagnostic.Cut(k: 6144, n: 1536, m: 4096, share: 0.612, ane: 15.5, gpu: 24.4, amx: 17.4)
        let three = Diagnostic.Cut(k: 3072, n: 16384, m: 4096, share: 0.45, ane: 65, gpu: 120, amx: 76.1, three: 59.6)
        let data = try JSONEncoder().encode([cut, Diagnostic.Cut(k: 2048, n: 6144, m: 1024, share: 0.5), three])
        XCTAssertEqual(String(decoding: data, as: UTF8.self),
                       "[[6144,1536,4096,0.612,15.5,24.4,17.4],[2048,6144,1024,0.5],[3072,16384,4096,0.45,65,120,76.1,59.6]]")
        let read = try JSONDecoder().decode([Diagnostic.Cut].self, from: data)
        XCTAssertEqual(read.first, cut); XCTAssertEqual(read.last, three)
    }

    /// **A grid of a family's shapes** (the ANE's): a shape measured reads its own share, at its height
    /// interpolated in log m; another shape reads its nearest in log k + log n — never two axes' product,
    /// which an ANE's jumps from one shape to the next would make up.
    func testSharesReadTheShapeItselfOrItsNearest() {
        let t = { (share: Double) in (1 - share) / share }
        let shares = Conductor.Shares([.init(k: 3840, n: 3840, m: 4096, ratio: t(0.5)), .init(k: 3840, n: 10240, m: 4096, ratio: t(0.56)),
                                       .init(k: 6144, n: 6144, m: 4096, ratio: t(0.7)), .init(k: 3840, n: 10240, m: 1024, ratio: t(0.68)),
                                       .init(k: 3840, n: 3840, m: 1024, ratio: t(0.6)), .init(k: 6144, n: 6144, m: 1024, ratio: t(0.8))])
        let range = NeuralEngine.shareRange
        XCTAssertEqual(shares.share(k: 3840, n: 10240, m: 4096, range: range)!, 0.56, accuracy: 1e-9)
        XCTAssertEqual(shares.share(k: 3840, n: 10240, m: 1024, range: range)!, 0.68, accuracy: 1e-9)
        XCTAssertEqual(shares.share(k: 3840, n: 10240, m: 4128, range: range)!, 0.56, accuracy: 1e-9, "beyond the tallest: its share")
        XCTAssertEqual(shares.share(k: 4096, n: 4096, m: 4096, range: range)!, 0.5, accuracy: 1e-9, "nearest: 3 840²")
        XCTAssertEqual(shares.share(k: 6144, n: 5120, m: 4096, range: range)!, 0.7, accuracy: 1e-9, "nearest: 6 144²")
    }

    func testSharesComposeTheTwoAxes() throws {
        func point(_ k: Int, _ n: Int, _ m: Int, _ ratio: Double) -> Conductor.Shares.Point { .init(k: k, n: n, m: m, ratio: ratio) }
        let shares = Conductor.Shares([point(2048, 4096, 1024, 0.6), point(4096, 4096, 1024, 0.5), point(8192, 4096, 1024, 0.4),
                                       point(4096, 1024, 1024, 0.25), point(4096, 16384, 1024, 1.0),
                                       point(2048, 4096, 4096, 0.8), point(4096, 4096, 4096, 0.7), point(8192, 4096, 4096, 0.6),
                                       point(4096, 1024, 4096, 0.35), point(4096, 16384, 4096, 1.4)])
        XCTAssertEqual(shares.share(k: 4096, n: 4096, m: 1024)!, 1 / 1.5, accuracy: 1e-12)
        XCTAssertEqual(shares.share(k: 8192, n: 16384, m: 1024)!, 1 / 1.8, accuracy: 1e-12)
        XCTAssertEqual(shares.share(k: 2048, n: 2048, m: 1024)!, 1 / 1.45, accuracy: 1e-12)
        XCTAssertEqual(shares.share(k: 4096, n: 4096, m: 2048)!, 1 / 1.6, accuracy: 1e-12)
        XCTAssertEqual(shares.share(k: 1024, n: 4096, m: 512)!, 1 / 1.6, accuracy: 1e-12, "the nearest point outside")
        XCTAssertEqual(shares.share(k: 4096, n: 65536, m: 8192)!, 1 / 2.4, accuracy: 1e-12)
        let depthOnly = Conductor.Shares([point(2048, 6144, 1024, 0.6), point(8192, 6144, 1024, 0.4)])
        XCTAssertEqual(depthOnly.share(k: 4096, n: 1, m: 1024)!, 1 / 1.5, accuracy: 1e-12)
        XCTAssertEqual(depthOnly.share(k: 4096, n: 1, m: 1024), depthOnly.share(k: 4096, n: 65536, m: 1024))
        let skewed = Conductor.Shares([point(2048, 6144, 1024, 10), point(8192, 6144, 1024, 0)])
        XCTAssertEqual(skewed.share(k: 2048, n: 6144, m: 1024), Conductor.fractionRange.lowerBound)
        XCTAssertEqual(skewed.share(k: 8192, n: 6144, m: 1024), Conductor.fractionRange.upperBound)
        XCTAssertEqual(Conductor.Shares(text: shares.description), shares)
        XCTAssertTrue(shares.description.hasPrefix("2048×4096@1024=0.600,"))
        XCTAssertEqual(Conductor.Shares(text: ""), Conductor.Shares.none)
        XCTAssertNil(Conductor.Shares(text: "2048@1024=0.5"), "a grid without its n does not read")
        XCTAssertNil(Conductor.Shares.none.share(k: 4096, n: 6144, m: 1024))
    }
}
