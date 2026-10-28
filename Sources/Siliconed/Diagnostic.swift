import CryptoKit
import Foundation
import IOKit
import IOKit.ps
import Metal

/// **The diagnostic — a capacity map of the machine**: ~39 s on the M1 Pro, **a minute at most** on
/// the slowest Mac it aims at (each stage's seconds are in the report, `stages`), voluntary, not one
/// byte of weights, a JSON a user can share. It answers two things: what the fleet has (one report
/// per user; `tools/aggregate_diagnostics.py` makes the table by chip), and, per machine, the figures
/// that decide the engine's compute branches. **A figure is in it only if it decides a branch of the
/// engine or answers a fleet question.**
///
///     operands: uniform noise, the same on every machine (xorshift by blocks of 2¹⁶)
///        │                                                    the exact: fp64 on the CPU, sampled rows
///        │                                                    (`cblas_dgemm`, an attention in Double)
///        ├─ fp32 products, Z-Image's and Qwen-Image-2.1's blocks, [m, d]·[d, ffn] and [m, ffn]·[ffn, d],
///        │  m = 1024, 4096
///        │     GPU alone   GEMM.linear (MPS)                          ─┐
///        │     AMX alone   Conductor.multiply (BNNSMatMul)              ├─ share = gpu / (gpu + amx)  ─▶ amx, amxFraction
///        │     both        GEMM.linear through Conductor at that share ─┘  does the cut pay here?
///        │     bf16 weight GEMM.linear(bf16:) (TensorGEMM, matmul2d)  + the copy's time, against the
///        │                 better of the two above + the widening's time ─────────────────────────▶ gemm_tensor
///        │     bf16 weight and activation, the activation rounded just before the product
///        │                 (TensorGEMM, matmul2d bfloat × bfloat → float), the rounding's pass in the time,
///        │                 against the same reference + the widening ──────────────────────────────▶ gemm_bf16_act
///        │     int8 weight WeightProducts.int8Product (matmul2d float × int8, convrot on the activation),
///        │                 alone and beside the AMX on its columns, + the copy's time ─────────────▶ gemm_int8
///        │     Q8_0 weight the same call on scales per 32 rows (q · s into threadgroup memory),
///        │                 alone and beside the AMX on its columns, + the copy's time ─────────────▶ gemm_q8
///        ├─ fp16 and bf16 products, Z-Image's [1024, 3840]·[3840, 10240]                       ─▶ fleet
///        │     GPU MPSGraph (out fp16 / bf16) · BNNS (out fp32) · Metal 4 matmul2d (out fp32)
///        ├─ fp32 attention, S = 1024 and 4096, the models' heads × 128
///        │     MPSGraph (Attention) · FlashMatrix · HeadSplitAttention (cpuFraction)           ─▶ flash, the head split
///        ├─ widening of a Z-Image w1, 39.3 M values: bf16 → fp32, int8 × bf16 scale per column
///        │     CPU (Widen)                                                                    ─▶ the reference
///        │     and its bf16 copy, CPU (Widen.copyBfloat16)                                    ─▶ gemm_tensor
///        │     int8 convrot (Widen.dequantizeRotated) and the int8 copy, CPU                   ─▶ gemm_int8
///        │     Q8_0 (Widen.dequantize, fp16 scales per 32 rows) and its copy, CPU              ─▶ gemm_q8
///        └─ the first product again, GPU and AMX: the drift witness (invariant 6)
///     read, not measured: the machine, the CPU's FEAT_* flags, thermal / power / swap before and after
///
/// **Each figure** is a median of three wall-clock runs after a warm-up (one run when the warm-up
/// alone passed `longWarmUp`), and **each output** is judged against the exact: the worst row's
/// relative error, each row normalized by its own norm (`channelError`, the rows laid out as its
/// channels), `pass` at or under the item's threshold.
///
/// What sets it apart — what a diagnostic written by analogy with the checks would get wrong:
///
///   - **it judges the chip, never the weights.** Schema 4 rendered two DiT steps of each installed
///     model against an fp32 golden: an 8-bit install could only fail it (6.6·10⁻² against 5·10⁻⁴ on
///     an M4), and nothing in the code read what it measured. The operands here are synthetic, and
///     every product is a block's shape, read from the models' `Config` defaults;
///   - **the heights are multiples of 64.** Schema 4's witness ran at 511 rows, which `GEMM` submits
///     as 448 + 63 through the padding tier (`rowSplit`): two submissions, one of them at 64 rows —
///     its TFLOP/s was not a product's;
///   - **the AMX is the engine's BNNS, not `cblas`**: `Conductor.multiply`, the call the cut makes on
///     its worker thread. `cblas_sgemm` is another code (1.99 against 2.26 TFLOP/s, `AMXLegacy.h`);
///   - **the cut runs at the share just measured, frozen** (`initialFraction`, `driven: false`): the
///     point a servo converges to, without its first oscillations inside a three-run median;
///   - **against fp64, never against another fp32**: `cblas` fp32 has its own error, of the size
///     of the GPU's;
///   - **a half-precision output is judged against its own rounding**: the exact rounded to fp16 or
///     bf16 is the floor no implementation can pass under; `pass` within 1.25 × that floor
///     (invariant 3: no further than the oracle + 25 %), never under the fp32 threshold;
///   - **every output is poisoned before each engine** (NaN on the sampled rows): a submission that
///     fails silently leaves the previous engine's right answer, which would pass;
///   - **it holds the machine's render lock** (`RenderLock.machine`) and runs on the engine's queue:
///     never beside a render, of this process or another.
///
/// **Deliberately not measured**: the disk (no decision reads its speed); a model's step (its
/// weights, its plan and the page cache decide it, and a render's own figures already say it); the
/// memory (B is a function of the RAM alone, ¾ · RAM − 2 GiB); an SDPA in fp16 (P20: never under
/// bf16, sums in fp32 — a precision the engine refuses is no capacity). The map is held under B like a render (`MemoryNeedDiagnostic.swift`): its need is
/// computed from its shapes, its plan chosen before the first product, its peak `phys_footprint` read by step
/// (`Report.memory`). Schemas 10–12 had put the Neural Engine's grid on one engine: 4.9 GB at the peak on the M1 Pro
/// (28/10/2026; schema 9's 1.55 GB was never measured again), 1.56 GB of swap on a MacBook Neo. Cut in batches of
/// shapes, an engine each: 3.5 GB under B = 4 GiB, the same decisions.
///
/// **The engine reads it in the app, and only there**: the app measures before its first render when it
/// holds no current map — none yet, or one of another `Fingerprint` (hardware, chip, macOS build, the
/// app's build: an update measures again) — and whenever the user asks (« Measure This Mac »).
/// It keeps the report in its preferences and hands it to `EngineSettings.load(from:map:)`; the rules
/// (`CapacityRules.swift`, `decide`) turn it into `amx`, `amx_fraction`, `flash`, `head_split`,
/// `gemm_tensor`, `gemm_bf16_act`, `gemm_int8`, `gemm_q8`. The command line, the checks and the close-out never designate a map: today's defaults.
package enum Diagnostic {
    /// The JSON's version: the script that aggregates the reports reads it. **5** (14/10/2026): the
    /// capacity map; 4 and before rendered the installed models. `decisions` came into 5 later the same
    /// day, optional: a report without it is the same measures. **6** (17/10/2026): the bf16-weight
    /// products (`products[].tensor`), the `bf16 copy` widening and `stages`; the decisions add `tensor`.
    /// **7** (18/10/2026): `cuts`, the GPU's balanced share on a grid of shapes (`k` and `n` axes, both
    /// heights); the decisions add `amxShares`. **8** (18/10/2026): the int8-weight products
    /// (`products[].int8`, `int8Coexecution`, `int8GPUColumns`), the `int8 convrot` and `int8 copy`
    /// widenings; the decisions add `int8`. **9** (18/10/2026): the Q8_0-weight products (`products[].q8`,
    /// `q8Coexecution`, `q8GPUColumns`), the `q8_0` and `q8_0 copy` widenings; the decisions add `q8`.
    /// **10** (22/10/2026): the Neural Engine beside the GPU (`products[].ane`, `aneGPUShare`, and `aneCuts` on
    /// the cuts' grid); the decisions add `ane`, `aneFraction` and `aneShares`.
    /// **11** (25/10/2026): the products on an activation rounded to bf16 (`products[].bf16Act`); the decisions
    /// add `bf16Act`. **12** (26/10/2026): the three engines at once on the ANE's grid (`aneCuts`' eighth figure,
    /// `three`), and MOSS-Image's shapes in it; the decisions add `aneAMX`.
    package static let schema = 12
    package static let repository = "https://github.com/gwenn-ha-dev/Siliconed"

    /// **The fp32 threshold**: the AMX check's (`siliconed-dev amx`), a row of an fp32 product or
    /// attention no further than 10⁻⁵ from the exact. Measured on the M1 Pro (14/10/2026): the
    /// products 0.80–2.1·10⁻⁶ (the worst at k = 12 288, GPU and the cut alike), the attention
    /// 2.0–4.4·10⁻⁶ (flash the closest) — 2.3× under it at the worst, and a fault is orders above.
    package static let fp32Threshold = 1e-5
    /// A half-precision output passes within this factor of its rounding floor (invariant 3).
    package static let floorMargin = 1.25

    // ── the run ─────────────────────────────────────────────────────────────────────────────

    /// **The stage the diagnostic is at**, as `run` reports it to `onProgress` — an app says it in its
    /// own language; `description` is the English the command line prints.
    package enum Stage: String, Sendable, CaseIterable, CustomStringConvertible {
        case products = "fp32 matrix products"
        case lowPrecision = "fp16 and bf16 products"
        case attention = "attention"
        case widening = "weight widening"
        case witness = "drift witness"
        package var description: String { rawValue }
        /// Its key in the report's `stages`: short, the issue link's bytes are counted.
        package var key: String {
            switch self {
            case .products: return "products"
            case .lowPrecision: return "lowPrecision"
            case .attention: return "attention"
            case .widening: return "widening"
            case .witness: return "witness"
            }
        }
    }

    /// **Runs the diagnostic**: ~1 min on the M1 Pro, no weights, its peak under B (`Plan`). Synchronous and
    /// heavy: call it off the main thread, and never from the engine's queue.
    ///
    /// **It passes the doors a render passes**: it waits its turn on the engine's queue (`Engine.file`:
    /// never beside a render of this process) and holds the machine's render lock (a render of
    /// another process refuses it: `renderAlreadyRunning`). `cancellation` is checked between two
    /// measurements (`cancelled`).
    package static func run(cancellation: Cancellation? = nil,
                           onProgress: (@Sendable (_ stage: Stage) -> Void)? = nil) throws(EngineError) -> Report {
        dispatchPrecondition(condition: .notOnQueue(Engine.file))
        let result = Engine.file.sync {
            Result { () throws(EngineError) -> Report in try runQueued(cancellation: cancellation, onProgress: onProgress) }
        }
        return try result.get()
    }

    private static func runQueued(cancellation: Cancellation?,
                                  onProgress: (@Sendable (_ stage: Stage) -> Void)?) throws(EngineError) -> Report {
        let started = Date()
        let lock = try RenderLock.acquire(RenderLock.machine)
        defer { withExtendedLifetime(lock) {} }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw .internalFailure(component: "diagnostic", detail: "no Metal device")
        }
        // The frameworks' first use (Metal, MPSGraph, Core ML) is the process's base, as a render's (`OperatorFootprint.bringUp`).
        OperatorFootprint.bringUp()
        _ = NeuralEngine.bringUp
        let gauge = Gauge()
        // The plan, before the first computation: the map is held under B like a render (`MemoryNeedDiagnostic.swift`).
        let plan = try Plan.choose(budget: MemoryBudget.machineCeiling(), base: gauge.base)
        let bench = Bench(device: device, queue: queue, cancellation: cancellation, gauge: gauge)
        let before = conditions()
        var stages: [String: Double] = [:]
        func stage<T>(_ s: Stage, _ body: () throws -> T) rethrows -> T {
            onProgress?(s)
            let begin = Date()
            defer { stages[s.key] = Date().timeIntervalSince(begin) }
            return try body()
        }
        do {
            // The products' operands (~1.1 GB) are released before the attention's peak.
            gauge.mark("products", "operands")
            var operands: Operands? = Operands()
            let (products, cuts, aneCuts) = try stage(.products) { () throws -> ([Product], [Cut], [Cut]) in
                // Its parts timed too (`products.…`): the minute's budget is read, not guessed.
                var clock = Date()
                func lap(_ key: String) { stages["products." + key] = Date().timeIntervalSince(clock); clock = Date() }
                let products = try measureProducts(operands!, bench)
                lap("products")
                gauge.mark("cuts")
                let shares = products.compactMap(\.gpuShare).sorted()
                let cuts = try measureCuts(operands!, bench, start: shares.isEmpty ? 0.631 : shares[shares.count / 2])
                lap("cuts")
                gauge.mark("aneCuts")
                var three = 0.0
                let aneCuts = try measureANECuts(operands!, bench, cuts: cuts, products: products, batches: aneBatches(allotment: plan.aneAllotment),
                                              threeSeconds: &three)
                lap("aneCuts")
                stages["products.aneCuts.three"] = three
                return (products, cuts, aneCuts)
            }
            gauge.mark("lowPrecision")
            let lowPrecision = try stage(.lowPrecision) { try measureLowPrecision(operands!, bench) }
            operands = nil
            gauge.mark("attention")
            let attention = try stage(.attention) { try measureAttention(bench) }
            gauge.mark("widening")
            let widening = try stage(.widening) { try measureWidening(bench) }
            gauge.mark("witness")
            let witness = try stage(.witness) { try measureWitness(first: products[0], bench) }
            gauge.finish()
            let settings = EngineSettings.effective
            var forced: [String: String] = [:]
            for row in settings.rows where row.source != .byDefault { forced[row.key] = "\(row.value) (\(row.source.rawValue))" }
            return Report(date: ISO8601DateFormatter().string(from: started), seconds: Date().timeIntervalSince(started),
                          machine: machine(), cpuFeatures: cpuFeatures(), settings: forced, before: before,
                          after: conditions(), products: products, lowPrecision: lowPrecision, attention: attention,
                          widening: widening, witness: witness, stages: stages, cuts: cuts,
                          aneCuts: aneCuts.isEmpty ? nil : aneCuts, memory: Memory(plan: plan, peaks: gauge.peaks))
        } catch {
            throw EngineError(error)
        }
    }
}
