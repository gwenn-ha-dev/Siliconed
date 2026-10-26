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
/// bf16, sums in fp32 — a precision the engine refuses is no capacity). The peak `phys_footprint` is 1.55 GB on the
/// M1 Pro (18/10/2026, schema 9), the products' operands (~1.1 GB with the cuts' grid and the 8-bit weight,
/// released before the attention) its larger part — under B on the smallest Mac (8 GB, B = 4 GiB).
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

    // ── the report ──────────────────────────────────────────────────────────────────────────

    package struct Machine: Codable, Sendable, Equatable {
        /// `machdep.cpu.brand_string`: "Apple M1 Pro".
        package var chip: String
        /// `hw.model`: "MacBookPro18,3".
        package var hardwareModel: String
        package var performanceCores: Int
        package var efficiencyCores: Int
        /// IORegistry's `gpu-core-count` (`AGXAccelerator`); 0 if unreadable.
        package var gpuCores: Int
        package var memoryBytes: Int
        /// "26.0.1 (25A…)".
        package var macOS: String
        /// `CFBundleShortVersionString (CFBundleVersion)` of the app, else the git commit of the
        /// repository the binary was built in (`-dirty` if modified), else "unknown".
        package var version: String
        /// The executable's fingerprint: the first 12 hex digits of the SHA-256 of `RenderCache.build`
        /// (path, size, date) — two builds of one commit differ here, and no local path goes public.
        package var build: String
        /// `MTLDevice.name`: "Apple M1 Pro".
        package var gpu: String
        /// The highest `MTLGPUFamily.appleN` the device supports, "apple7" on the M1 family; "" if none.
        package var gpuFamily: String
        /// `MTLDevice.recommendedMaxWorkingSetSize`: what Metal lets the process keep resident.
        package var gpuWorkingSetBytes: Int

        package init(chip: String, hardwareModel: String, performanceCores: Int, efficiencyCores: Int,
                    gpuCores: Int, memoryBytes: Int, macOS: String, version: String, build: String,
                    gpu: String = "", gpuFamily: String = "", gpuWorkingSetBytes: Int = 0) {
            self.chip = chip; self.hardwareModel = hardwareModel; self.performanceCores = performanceCores
            self.efficiencyCores = efficiencyCores; self.gpuCores = gpuCores; self.memoryBytes = memoryBytes
            self.macOS = macOS; self.version = version; self.build = build
            self.gpu = gpu; self.gpuFamily = gpuFamily; self.gpuWorkingSetBytes = gpuWorkingSetBytes
        }

        /// "16 GB": the spec sheet's figure (2³⁰), what a reader of the table compares.
        package var memoryLabel: String { "\(Int((Double(memoryBytes) / 1_073_741_824).rounded())) GB" }
    }

    /// **The state of the machine**, read before the first measurement and after the last.
    package struct Conditions: Codable, Sendable, Equatable {
        /// `ProcessInfo.thermalState`: nominal, fair, serious, critical. A fanless Mac throttles from fair.
        package var thermal: String
        package var lowPowerMode: Bool
        /// The machine runs on its battery; `nil` when the power source does not say.
        package var onBattery: Bool?
        /// `vm.swapusage`'s used bytes: another process's swap, and memory this one may lack.
        package var swapUsedBytes: Int
        package init(thermal: String, lowPowerMode: Bool, onBattery: Bool?, swapUsedBytes: Int) {
            self.thermal = thermal; self.lowPowerMode = lowPowerMode; self.onBattery = onBattery
            self.swapUsedBytes = swapUsedBytes
        }

        /// **A machine at rest, as far as one reading says**: thermal state nominal, Low Power Mode off. The
        /// battery is not held against it: it did not throttle the maps taken on it (M5, M5 Pro: witness
        /// ×1.00). Swap is judged between two readings (`Report.isProvisional`), never from one.
        package var isNominal: Bool { thermal == "nominal" && !lowPowerMode }

        private enum CodingKeys: String, CodingKey { case thermal, lowPowerMode, onBattery, swapUsedBytes }
        /// Explicit `null`s, as everywhere in the report: "unknown" is said, not dropped.
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(thermal, forKey: .thermal); try c.encode(lowPowerMode, forKey: .lowPowerMode)
            try c.encode(onBattery, forKey: .onBattery); try c.encode(swapUsedBytes, forKey: .swapUsedBytes)
        }
    }

    /// **One engine on one item**: how fast, and how far from the exact. **Written `[rate, error,
    /// pass]`** — forty of them as objects made the report too long for an issue link (9 690 bytes
    /// against ~8 000); the thresholds they were judged by are the report's `thresholds`, and a
    /// half-precision product's own `threshold`.
    package struct Measure: Codable, Sendable, Equatable {
        /// TFLOP/s for a product or an attention (`2·m·k·n`, `4·S²·H·Dh`), G values/s for a widening;
        /// four significant digits.
        package var rate: Double
        /// The worst row's relative error against the exact (fp64), three significant digits. A
        /// non-finite error — a NaN in the output — is written as the largest finite double.
        package var error: Double
        /// At or under the item's threshold, judged before the rounding.
        package var pass: Bool

        package init(rate: Double, error: Double, threshold: Double) {
            self.rate = Diagnostic.significant(rate, 4)
            self.error = Diagnostic.significant(error, 3)
            self.pass = error.isFinite && error <= threshold
        }

        package init(from decoder: any Decoder) throws {
            var c = try decoder.unkeyedContainer()
            rate = try c.decode(Double.self); error = try c.decode(Double.self); pass = try c.decode(Bool.self)
        }

        package func encode(to encoder: any Encoder) throws {
            var c = encoder.unkeyedContainer()
            try c.encode(rate); try c.encode(error); try c.encode(pass)
        }
    }

    /// **What the measures were judged by**: one threshold for every fp32 output, 0 for a widening
    /// (exact by construction), and the margin over its floor a half-precision output gets.
    package struct Thresholds: Codable, Sendable, Equatable {
        package var fp32: Double
        package var widening: Double
        package var floorMargin: Double
        package init(fp32: Double = Diagnostic.fp32Threshold, widening: Double = 0, floorMargin: Double = Diagnostic.floorMargin) {
            self.fp32 = fp32; self.widening = widening; self.floorMargin = floorMargin
        }
    }

    /// **An fp32 product of a DiT block** — `[m, k] · [k, n]`, the weight stored `[input, output]` as
    /// the forge lays it out.
    package struct Product: Codable, Sendable, Equatable {
        /// The model whose block gives the shape: "z-image", "qwen-image-2.1".
        package var model: String
        /// "up" `[m, d]·[d, ffn]`, "down" `[m, ffn]·[ffn, d]`.
        package var product: String
        package var m, k, n: Int
        /// MPS through `GEMM.linear`, the conductor off.
        package var gpu: Measure
        /// `BNNSMatMul` through `Conductor.multiply`, on the calling thread; `nil` when BNNS refuses.
        package var amx: Measure?
        /// `gpu.rate / (gpu.rate + amx.rate)`: the share of rows the GPU would take — `amxFraction`.
        package var gpuShare: Double?
        /// `GEMM.linear` through the conductor, its cut frozen at `gpuShare`.
        package var coexecution: Measure?
        /// The rows the cut gave the GPU (`Conductor.cut`, a multiple of `amx_quantum`); `m` when the
        /// cut does not apply (`m < amx_min`).
        package var coexecutionGPURows: Int?
        /// `TensorGEMM` (`gemm_tensor`): the same product on the weight in bf16, the activations fp32;
        /// `nil` without Metal 4, or in a report of schema 5.
        package var tensor: Measure?
        /// `gemm_bf16_act`: the same product on the weight in bf16 **and the activation rounded to bf16** just
        /// before it (`TensorGEMM.linear(a:rounding:…)`, `matmul2d` `bfloat × bfloat → float`), the rounding's
        /// pass included in the time; judged against the exact product of the *rounded* activation (the
        /// check of the accumulation and of the kernel, as the half-precision figures of the fleet table —
        /// what the rounding costs the image is the check's, against diffusers in bf16). `nil` without Metal 4,
        /// or before schema 11.
        package var bf16Act: Measure?
        /// `gemm_int8`: the product on a published int8 weight (convrot of 256, one scale), read as it is
        /// (`WeightProducts.int8Product`, the GPU alone); `nil` without Metal 4, or before schema 8.
        package var int8: Measure?
        /// The same with the AMX on the last columns at `gpuShare` (`int8GPUColumns` to the GPU), the AMX's
        /// slice widened while the GPU computes; `nil` when the cut does not apply.
        package var int8Coexecution: Measure?
        package var int8GPUColumns: Int?
        /// `gemm_q8`: the product on a GGUF Q8_0 weight (int8, a scale per 32 rows and column), read as it
        /// is (`WeightProducts.int8Product` on a block plan, the GPU alone); `nil` without Metal 4, when this
        /// macOS's compiler refuses the kernel (`TensorGEMM.readsBlocks`), or before schema 9.
        package var q8: Measure?
        /// The same beside the AMX on the last columns at `gpuShare` (`q8GPUColumns` to the GPU).
        package var q8Coexecution: Measure?
        package var q8GPUColumns: Int?
        /// `gemm_ane`: `GEMM.linear` with the Neural Engine on the last rows (`NeuralEngine`, fp16, the weight's
        /// columns lifted), the GPU alone on the first, at `aneGPUShare`; judged against **1.25 × bf16's error**
        /// — the product of the input rounded to bf16 (the weight is), rounded to bf16 — (invariant 3: a
        /// half-precision path no further than bf16 + 25 %), not the fp32 threshold. `nil` under `NeuralEngine.minimumRows`, or before schema 10.
        package var ane: Measure?
        /// The GPU's share the pair was measured at: the one that balanced both sides at a first share of ½.
        package var aneGPUShare: Double?

        package init(model: String, product: String, m: Int, k: Int, n: Int, gpu: Measure, amx: Measure?,
                    gpuShare: Double?, coexecution: Measure?, coexecutionGPURows: Int?, tensor: Measure? = nil,
                    bf16Act: Measure? = nil, int8: Measure? = nil, int8Coexecution: Measure? = nil, int8GPUColumns: Int? = nil,
                    q8: Measure? = nil, q8Coexecution: Measure? = nil, q8GPUColumns: Int? = nil,
                    ane: Measure? = nil, aneGPUShare: Double? = nil) {
            self.model = model; self.product = product; self.m = m; self.k = k; self.n = n; self.gpu = gpu
            self.amx = amx; self.gpuShare = gpuShare.map { Diagnostic.significant($0, 3) }
            self.coexecution = coexecution; self.coexecutionGPURows = coexecutionGPURows; self.tensor = tensor
            self.bf16Act = bf16Act
            self.int8 = int8; self.int8Coexecution = int8Coexecution; self.int8GPUColumns = int8GPUColumns
            self.q8 = q8; self.q8Coexecution = q8Coexecution; self.q8GPUColumns = q8GPUColumns
            self.ane = ane; self.aneGPUShare = aneGPUShare.map { Diagnostic.significant($0, 3) }
        }

        private enum CodingKeys: String, CodingKey {
            case model, product, m, k, n, gpu, amx, gpuShare, coexecution, coexecutionGPURows, tensor, bf16Act, int8, int8Coexecution, int8GPUColumns,
                 q8, q8Coexecution, q8GPUColumns, ane, aneGPUShare
        }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(model, forKey: .model); try c.encode(product, forKey: .product)
            try c.encode(m, forKey: .m); try c.encode(k, forKey: .k); try c.encode(n, forKey: .n)
            try c.encode(gpu, forKey: .gpu); try c.encode(amx, forKey: .amx); try c.encode(gpuShare, forKey: .gpuShare)
            try c.encode(coexecution, forKey: .coexecution); try c.encode(coexecutionGPURows, forKey: .coexecutionGPURows)
            try c.encode(tensor, forKey: .tensor)
            // Absent rather than `null`: the link of an issue carries the whole report (`issueURL`).
            try c.encodeIfPresent(bf16Act, forKey: .bf16Act)
            try c.encodeIfPresent(int8, forKey: .int8)
            try c.encodeIfPresent(int8Coexecution, forKey: .int8Coexecution); try c.encodeIfPresent(int8GPUColumns, forKey: .int8GPUColumns)
            try c.encodeIfPresent(q8, forKey: .q8)
            try c.encodeIfPresent(q8Coexecution, forKey: .q8Coexecution); try c.encodeIfPresent(q8GPUColumns, forKey: .q8GPUColumns)
            try c.encodeIfPresent(ane, forKey: .ane); try c.encodeIfPresent(aneGPUShare, forKey: .aneGPUShare)
        }
    }

    /// **An fp32 attention** at a model's heads — `q`, `k`, `v` `[S, H·Dh]` as the GEMMs write them.
    package struct AttentionCapacity: Codable, Sendable, Equatable {
        package var model: String
        package var heads, headDim, sequence: Int
        /// `MPSGraph.scaledDotProductAttention` (`Attention`), the engine's default.
        package var graph: Measure?
        /// The in-house matrix flash kernel (`FlashMatrix`, the `flash` setting); `nil` when it refuses the device.
        package var flash: Measure?
        /// The heads cut between the GPU and the CPU (`HeadSplitAttention`, its `cpuFraction`).
        package var headSplit: Measure?
        /// What the head split's own run took, each part: the figures `HeadSplitAttention.balance` derives the
        /// share of heads from; `nil` before schema 11 or without a split.
        package var splitParts: SplitParts?

        /// **The two halves of one cut call**: `cpuHeads` heads on the CPU for `cpuSeconds`, the rest on the
        /// GPU for `gpuSeconds` (each part's wall time, as `HeadSplitAttention.run` returns them).
        package struct SplitParts: Codable, Sendable, Equatable {
            package var cpuHeads: Int
            package var gpuSeconds, cpuSeconds: Double
            package init(cpuHeads: Int, gpuSeconds: Double, cpuSeconds: Double) {
                self.cpuHeads = cpuHeads
                self.gpuSeconds = Diagnostic.significant(gpuSeconds, 4); self.cpuSeconds = Diagnostic.significant(cpuSeconds, 4)
            }
        }

        package init(model: String, heads: Int, headDim: Int, sequence: Int, graph: Measure?, flash: Measure?,
                    headSplit: Measure?, splitParts: SplitParts? = nil) {
            self.model = model; self.heads = heads; self.headDim = headDim; self.sequence = sequence
            self.graph = graph; self.flash = flash; self.headSplit = headSplit; self.splitParts = splitParts
        }

        private enum CodingKeys: String, CodingKey { case model, heads, headDim, sequence, graph, flash, headSplit, splitParts }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(model, forKey: .model); try c.encode(heads, forKey: .heads)
            try c.encode(headDim, forKey: .headDim); try c.encode(sequence, forKey: .sequence)
            try c.encode(graph, forKey: .graph); try c.encode(flash, forKey: .flash); try c.encode(headSplit, forKey: .headSplit)
            try c.encodeIfPresent(splitParts, forKey: .splitParts)
        }
    }

    /// **A weight widened to fp32** on the CPU, or copied as published — what a product pays before it
    /// computes. A report from an earlier build may also hold a `gpu` measure: decoding ignores it.
    package struct Widening: Codable, Sendable, Equatable {
        /// "bf16" (a shift), "int8" (`Float(q) · s`, one bf16 scale per column), "bf16 copy" (what
        /// `gemm_tensor` does instead of the widening: `Widen.copyBfloat16`, CPU only).
        package var kind: String
        package var values: Int
        package var cpu: Measure

        package init(kind: String, values: Int, cpu: Measure) {
            self.kind = kind; self.values = values; self.cpu = cpu
        }
    }

    /// **A product in fp16 or bf16** — the fleet question: does this machine compute half precision
    /// fast, and where.
    package struct LowPrecision: Codable, Sendable, Equatable {
        /// "mpsgraph" (the GPU through `MPSGraph`), "bnns" (the CPU's matrix unit), "matmul2d" (Metal 4's
        /// tensor operation — the M5's neural accelerators where there are some).
        package var engine: String
        /// The operands' type: "fp16", "bf16".
        package var type: String
        /// The result's type: the operands' for `MPSGraph`, "fp32" for the two others.
        package var output: String
        package var m, k, n: Int
        /// `nil` when the engine refuses the type or the device has no such engine.
        package var measure: Measure?
        /// The exact rounded to `output`: the error no implementation passes under.
        package var floor: Double
        /// `max(thresholds.fp32, floorMargin · floor)`.
        package var threshold: Double

        package init(engine: String, type: String, output: String, m: Int, k: Int, n: Int, measure: Measure?, floor: Double,
                    threshold: Double) {
            self.engine = engine; self.type = type; self.output = output; self.m = m; self.k = k; self.n = n
            self.measure = measure; self.floor = Diagnostic.significant(floor, 3)
            self.threshold = Diagnostic.significant(threshold, 3)
        }

        private enum CodingKeys: String, CodingKey { case engine, type, output, m, k, n, measure, floor, threshold }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(engine, forKey: .engine); try c.encode(type, forKey: .type); try c.encode(output, forKey: .output)
            try c.encode(m, forKey: .m); try c.encode(k, forKey: .k); try c.encode(n, forKey: .n)
            try c.encode(measure, forKey: .measure); try c.encode(floor, forKey: .floor)
            try c.encode(threshold, forKey: .threshold)
        }
    }

    /// **The GPU's share of one weight shape at one height** — the share that ends the GPU and the AMX
    /// together, measured in co-execution (`measureCuts`); the grid the frozen cut's formula reads
    /// (`amx_shares`, `Conductor.Shares`). **Written `[k, n, m, share]`**: twenty-two of them, the issue
    /// link's bytes are counted.
    package struct Cut: Codable, Sendable, Equatable {
        package var k, n, m: Int
        package var share: Double
        /// The ANE's grid only (`aneCuts`), milliseconds of one product: the GPU beside the ANE at `share`,
        /// the GPU alone, the GPU beside the AMX at its own grid's share (`nil` without co-execution) — what
        /// `CapacityRules` routes each shape by. Written after the share: `[k, n, m, share, ane, gpu, amx, three]`.
        /// `three` (schema 12): the three engines at once — the ANE on its rows at the share both pairs give
        /// (`Diagnostic.threeWayShare`), the GPU beside the AMX on the rest at the AMX's grid's share.
        package var ane, gpu, amx, three: Double?
        package init(k: Int, n: Int, m: Int, share: Double, ane: Double? = nil, gpu: Double? = nil, amx: Double? = nil,
                     three: Double? = nil) {
            self.k = k; self.n = n; self.m = m; self.share = Diagnostic.significant(share, 3)
            (self.ane, self.gpu, self.amx) = (ane.map { Diagnostic.significant($0, 3) }, gpu.map { Diagnostic.significant($0, 3) },
                                              amx.map { Diagnostic.significant($0, 3) })
            self.three = three.map { Diagnostic.significant($0, 3) }
        }
        package init(from decoder: any Decoder) throws {
            var c = try decoder.unkeyedContainer()
            k = try c.decode(Int.self); n = try c.decode(Int.self); m = try c.decode(Int.self); share = try c.decode(Double.self)
            if !c.isAtEnd { ane = try c.decodeNil() ? nil : try c.decode(Double.self) }
            if !c.isAtEnd { gpu = try c.decodeNil() ? nil : try c.decode(Double.self) }
            if !c.isAtEnd { amx = try c.decodeNil() ? nil : try c.decode(Double.self) }
            if !c.isAtEnd { three = try c.decodeNil() ? nil : try c.decode(Double.self) }
        }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.unkeyedContainer()
            try c.encode(k); try c.encode(n); try c.encode(m); try c.encode(share)
            if let ane, let gpu {
                try c.encode(ane); try c.encode(gpu)
                if let amx { try c.encode(amx); if let three { try c.encode(three) } }
            }
        }
    }

    /// **The first product, timed again at the end** (TFLOP/s): the drift over the diagnostic, which
    /// frames every figure of it (invariant 6). `before` is `products[0]`'s.
    package struct Witness: Codable, Sendable, Equatable {
        package var gpuBefore, gpuAfter: Double
        package var amxBefore, amxAfter: Double?

        package init(gpuBefore: Double, gpuAfter: Double, amxBefore: Double?, amxAfter: Double?) {
            self.gpuBefore = gpuBefore; self.gpuAfter = Diagnostic.significant(gpuAfter, 4)
            self.amxBefore = amxBefore; self.amxAfter = amxAfter.map { Diagnostic.significant($0, 4) }
        }

        private enum CodingKeys: String, CodingKey { case gpuBefore, gpuAfter, amxBefore, amxAfter }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(gpuBefore, forKey: .gpuBefore); try c.encode(gpuAfter, forKey: .gpuAfter)
            try c.encode(amxBefore, forKey: .amxBefore); try c.encode(amxAfter, forKey: .amxAfter)
        }
    }

    package struct Report: Codable, Sendable, Equatable {
        package var schema: Int
        /// ISO 8601, UTC.
        package var date: String
        /// The whole diagnostic, wall clock.
        package var seconds: Double
        package var machine: Machine
        /// The CPU's `hw.optional.arm.*` flags that bear on the products (`FEAT_BF16`, `FEAT_I8MM`,
        /// `FEAT_SME`, `FEAT_SME2`…) and `sme_max_svl_b`, as `sysctl` returns them; a flag this macOS does
        /// not know is absent.
        package var cpuFeatures: [String: Int]
        /// Every engine setting that is not its default, `"value (provenance)"`: a profile or an
        /// environment variable must not pass for the chip.
        package var settings: [String: String]
        package var before: Conditions
        package var after: Conditions
        package var products: [Product]
        /// The GPU's share per weight shape and height (`Cut`); `nil` in a report before schema 7.
        package var cuts: [Cut]?
        /// The GPU's share beside the Neural Engine on the same grid (`aneCuts` → `ane_shares`); `nil` before
        /// schema 10, or where no ANE program ran.
        package var aneCuts: [Cut]?
        package var lowPrecision: [LowPrecision]
        package var attention: [AttentionCapacity]
        package var widening: [Widening]
        package var witness: Witness
        /// **The seconds of each stage** (`Stage.key`), so the map's budget is read, not guessed;
        /// `nil` in a report of schema 5.
        package var stages: [String: Double]?
        package var thresholds: Thresholds
        /// **What these measures make this Mac use** (`Diagnostic.decide`), written so a shared report says
        /// it; `nil` in a report that predates it. The engine never reads this field: it applies the
        /// rules to the measures again (`EngineSettings.load(from:map:)`).
        package var decisions: Decisions?

        package init(schema: Int = Diagnostic.schema, date: String, seconds: Double, machine: Machine,
                    cpuFeatures: [String: Int], settings: [String: String] = [:], before: Conditions, after: Conditions,
                    products: [Product], lowPrecision: [LowPrecision], attention: [AttentionCapacity],
                    widening: [Widening], witness: Witness, thresholds: Thresholds = Thresholds(),
                    stages: [String: Double]? = nil, cuts: [Cut]? = nil, aneCuts: [Cut]? = nil) {
            self.schema = schema; self.date = date; self.seconds = Diagnostic.significant(seconds, 3)
            self.thresholds = thresholds
            self.machine = machine; self.cpuFeatures = cpuFeatures; self.settings = settings
            self.before = before; self.after = after; self.products = products; self.lowPrecision = lowPrecision
            self.attention = attention; self.widening = widening; self.witness = witness; self.cuts = cuts
            self.aneCuts = aneCuts
            self.stages = stages?.mapValues { Diagnostic.significant($0, 3) }
            self.decisions = Diagnostic.decide(products: products, attention: attention, widening: widening, witness: witness,
                                               cuts: cuts ?? [], aneCuts: aneCuts ?? [])
        }

        /// What the report is valid for: the machine and the binary that measured it.
        package var fingerprint: Fingerprint { Fingerprint(machine) }
        /// **The report still describes this process's machine and build** — what an app checks before
        /// handing it to `EngineSettings.load(from:map:)`; when it does not, the app offers to measure again.
        package var isCurrent: Bool { schema == Diagnostic.schema && fingerprint == .current }

        /// **Taken on a machine that was not at rest**: warm or in Low Power Mode before or after, or swap
        /// growing over the measure — another process pressing on memory (a Mac mini M4, 20/10/2026: swap
        /// 6.74 → 6.83 GB, MPSGraph's attention at ~1 TFLOP/s, half its sibling's). Such a map still
        /// serves — refusing a render on it would be worse —, and the app measures again at its next
        /// launch at rest.
        package var isProvisional: Bool {
            !before.isNominal || !after.isNominal || after.swapUsedBytes > before.swapUsedBytes
        }

        /// Every measure of the report, in its order.
        package var measures: [Measure] {
            products.flatMap { [$0.gpu] + [$0.amx, $0.coexecution, $0.tensor, $0.bf16Act, $0.int8, $0.int8Coexecution, $0.q8, $0.q8Coexecution].compactMap { $0 } }
                + lowPrecision.compactMap(\.measure)
                + attention.flatMap { [$0.graph, $0.flash, $0.headSplit].compactMap { $0 } }
                + widening.map(\.cpu)
        }
        /// Every engine that ran computed right.
        package var passes: Bool { measures.allSatisfy(\.pass) }

        /// **Stable JSON**: sorted keys, so two reports diff line by line. `pretty: false` for the
        /// issue URL, where every byte counts.
        package func json(pretty: Bool = true) throws -> Data {
            let encoder = JSONEncoder()
            encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
                                              : [.sortedKeys, .withoutEscapingSlashes]
            return try encoder.encode(self)
        }

        /// **"Report my configuration"**: a new GitHub issue from the `diagnostic.yml` form, its fields
        /// prefilled by their ids — `chip`, `memory`, `macos`, `version` and `report`, the whole JSON,
        /// which the issue's body then carries (the script that aggregates the reports reads it back from there).
        /// `includingReport: false` leaves `report` empty, the other fields filled: for a JSON too long
        /// for a link (browsers and GitHub cut beyond ~8 000 characters), pasted by the user instead.
        package func issueURL(includingReport: Bool = true) throws -> URL {
            var fields = [
                ("template", "diagnostic.yml"),
                ("labels", "diagnostic"),
                ("title", "Diagnostic: \(machine.chip) · \(machine.memoryLabel) · macOS \(machine.macOS)"),
                ("chip", machine.chip + (machine.gpuCores > 0 ? " (\(machine.gpuCores)-core GPU)" : "")),
                ("memory", machine.memoryLabel),
                ("macos", machine.macOS),
                ("version", machine.version),
            ]
            // Without `decisions`: the rules give them back from the measures (`diagnose --decide`), and
            // the link's bytes are counted.
            var measures = self
            measures.decisions = nil
            if includingReport { fields.append(("report", String(decoding: try measures.json(pretty: false), as: UTF8.self))) }
            return Diagnostic.issueURL(fields: fields)
        }
    }

    /// The URL of a new issue with `fields` as its query, **every byte that delimits a field escaped**:
    /// `URLComponents` leaves `+` alone, which GitHub reads back as a space — in a JSON exponent (`1e+20`)
    /// or a chip name. `,` and `:` stay as they are: a query may hold them (RFC 3986, `pchar`), they
    /// delimit nothing in a form's value, and escaped they were a quarter of a report's link (`%2C`, `%3A`:
    /// ~1 200 of 7 700 bytes for a schema 6 map). `;` stays escaped: some servers split a query on it.
    package static func issueURL(fields: [(String, String)]) -> URL {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~,:")
        let query = fields.map { key, value in
            "\(key)=\(value.addingPercentEncoding(withAllowedCharacters: allowed) ?? "")"
        }.joined(separator: "&")
        return URL(string: "\(repository)/issues/new?\(query)")!
    }

    /// `x` to `digits` significant digits, so the JSON says `3.312`, not `3.3123456789012`; a
    /// non-finite value becomes the largest finite double (JSON has no infinity).
    package static func significant(_ x: Double, _ digits: Int) -> Double {
        guard x.isFinite else { return x.isNaN || x > 0 ? .greatestFiniteMagnitude : -.greatestFiniteMagnitude }
        guard x != 0 else { return 0 }
        return Double(String(format: "%.\(digits - 1)e", x)) ?? x
    }

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

    /// **Runs the diagnostic**: ~1 min and 1.4 GB at the peak on the M1 Pro, no weights. Synchronous and
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
        let bench = Bench(device: device, queue: queue, cancellation: cancellation)
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
            var operands: Operands? = Operands()
            let (products, cuts, aneCuts) = try stage(.products) { () throws -> ([Product], [Cut], [Cut]) in
                // Its parts timed too (`products.…`): the minute's budget is read, not guessed.
                var clock = Date()
                func lap(_ key: String) { stages["products." + key] = Date().timeIntervalSince(clock); clock = Date() }
                let products = try measureProducts(operands!, bench)
                lap("products")
                let shares = products.compactMap(\.gpuShare).sorted()
                let cuts = try measureCuts(operands!, bench, start: shares.isEmpty ? 0.631 : shares[shares.count / 2])
                lap("cuts")
                var three = 0.0
                let aneCuts = try measureANECuts(operands!, bench, cuts: cuts, products: products, threeSeconds: &three)
                lap("aneCuts")
                stages["products.aneCuts.three"] = three
                return (products, cuts, aneCuts)
            }
            let lowPrecision = try stage(.lowPrecision) { try measureLowPrecision(operands!, bench) }
            operands = nil
            let attention = try stage(.attention) { try measureAttention(bench) }
            let widening = try stage(.widening) { try measureWidening(bench) }
            let witness = try stage(.witness) { try measureWitness(first: products[0], bench) }
            let settings = EngineSettings.effective
            var forced: [String: String] = [:]
            for row in settings.rows where row.source != .byDefault { forced[row.key] = "\(row.value) (\(row.source.rawValue))" }
            return Report(date: ISO8601DateFormatter().string(from: started), seconds: Date().timeIntervalSince(started),
                          machine: machine(), cpuFeatures: cpuFeatures(), settings: forced, before: before,
                          after: conditions(), products: products, lowPrecision: lowPrecision, attention: attention,
                          widening: widening, witness: witness, stages: stages, cuts: cuts,
                          aneCuts: aneCuts.isEmpty ? nil : aneCuts)
        } catch {
            throw EngineError(error)
        }
    }

    // ── the machine ─────────────────────────────────────────────────────────────────────────

    package static func machine() -> Machine {
        let device = MTLCreateSystemDefaultDevice()
        return Machine(chip: sysctlString("machdep.cpu.brand_string"), hardwareModel: sysctlString("hw.model"),
                       performanceCores: sysctlInteger("hw.perflevel0.physicalcpu") ?? 0,
                       efficiencyCores: sysctlInteger("hw.perflevel1.physicalcpu") ?? 0,
                       gpuCores: gpuCoreCount(), memoryBytes: sysctlInteger("hw.memsize") ?? 0,
                       macOS: macOSLabel(), version: binaryVersion(), build: buildStamp(),
                       gpu: device?.name ?? "none", gpuFamily: device.map(appleFamily) ?? "",
                       gpuWorkingSetBytes: Int(device?.recommendedMaxWorkingSetSize ?? 0))
    }

    /// "26.0.1 (25A…)": the version and, when the kernel says it, the build.
    private static func macOSLabel() -> String {
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let osBuild = sysctlString("kern.osversion")
        return "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)" + (osBuild == "unknown" ? "" : " (\(osBuild))")
    }

    /// The first 12 hex digits of the SHA-256 of `RenderCache.build` (the executable's path, size, date).
    /// **The engine that measured**: in an app, its Info.plist's `SiliconedEngine` — the digest `tools/app.sh`
    /// takes of the engine's sources (`Sources/Siliconed`, `Sources/AMXLegacy`, `Package.swift`) and of the
    /// compiler, so that an update moving only the app, its phrases or its docs keeps the map; elsewhere
    /// (the command line, a test), the executable's own stamp: every build measures again.
    private static func buildStamp() -> String {
        for bundle in enclosingBundles() {
            if let engine = bundle.infoDictionary?["SiliconedEngine"] as? String { return "engine " + engine }
        }
        return SHA256.hash(data: Data(RenderCache.build.utf8)).prefix(6).map { String(format: "%02x", $0) }.joined()
    }

    /// This executable's bundle, then the `.app` it sits in (a helper's `Contents/Helpers`).
    private static func enclosingBundles() -> [Bundle] {
        var bundles = [Bundle.main]
        if let path = Bundle.main.executableURL?.resolvingSymlinksInPath().path, let range = path.range(of: ".app/Contents/") {
            if let app = Bundle(path: String(path[..<range.lowerBound]) + ".app") { bundles.append(app) }
        }
        return bundles
    }

    /// **What a map is valid for**: the hardware model, the chip, macOS with its build, and the engine
    /// that measured it (`build`). Any of them changing — a new Mac restored from a backup, a macOS
    /// update that moves MPS, an update that moves a kernel — makes the map stale: the stale one is not
    /// applied, and the app measures again (~1 min). **The app's version is said, not compared**: the
    /// map measures the engine, and an update of the app alone does not move it.
    package struct Fingerprint: Codable, Sendable, Equatable, CustomStringConvertible {
        package var hardwareModel, chip, macOS, version, build: String

        package init(hardwareModel: String, chip: String, macOS: String, version: String, build: String) {
            self.hardwareModel = hardwareModel; self.chip = chip; self.macOS = macOS; self.version = version
            self.build = build
        }

        package init(_ machine: Machine) {
            self.init(hardwareModel: machine.hardwareModel, chip: machine.chip, macOS: machine.macOS,
                      version: machine.version, build: machine.build)
        }

        package static func == (a: Fingerprint, b: Fingerprint) -> Bool {
            (a.hardwareModel, a.chip, a.macOS, a.build) == (b.hardwareModel, b.chip, b.macOS, b.build)
        }

        /// This process's, read without Metal (`sysctl`, the bundle, the executable's stamp).
        package static var current: Fingerprint {
            Fingerprint(hardwareModel: sysctlString("hw.model"), chip: sysctlString("machdep.cpu.brand_string"),
                        macOS: macOSLabel(), version: binaryVersion(), build: buildStamp())
        }

        package var description: String { "\(chip) (\(hardwareModel)) · macOS \(macOS) · \(version) · \(build)" }
    }

    /// **The CPU flags a product can use**, read: half-precision arithmetic (`FEAT_FP16`, `FEAT_FHM`),
    /// bf16 (`FEAT_BF16`, `FEAT_EBF16`), the int8 matrix and dot products (`FEAT_I8MM`, `FEAT_DotProd`),
    /// and SME — the M4's matrix unit made architectural — with its outer products by type and its
    /// vector length. Not measured: BNNS decides what it runs on, and its products are (`LowPrecision`).
    package static let featureNames = [
        "FEAT_FP16", "FEAT_FHM", "FEAT_BF16", "FEAT_EBF16", "FEAT_I8MM", "FEAT_DotProd",
        "FEAT_SME", "FEAT_SME2", "SME_F32F32", "SME_F16F32", "SME_B16F32", "SME_I8I32",
        "FEAT_SME_F16F16", "FEAT_SME_B16B16",
    ]

    package static func cpuFeatures() -> [String: Int] {
        var flags: [String: Int] = [:]
        for name in featureNames { flags[name] = sysctlInteger("hw.optional.arm.\(name)") }
        flags["sme_max_svl_b"] = sysctlInteger("hw.optional.arm.sme_max_svl_b")
        return flags
    }

    /// The highest `appleN` family: `MTLGPUFamily.apple1` is 1001, and the next ones follow.
    private static func appleFamily(_ device: MTLDevice) -> String {
        for n in stride(from: 16, through: 1, by: -1) {
            if let family = MTLGPUFamily(rawValue: 1000 + n), device.supportsFamily(family) { return "apple\(n)" }
        }
        return ""
    }

    // ── the conditions ──────────────────────────────────────────────────────────────────────

    package static func conditions() -> Conditions {
        var usage = xsw_usage(), size = MemoryLayout<xsw_usage>.size
        let swap = sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 ? Int(usage.xsu_used) : 0
        return Conditions(thermal: thermalLabel(ProcessInfo.processInfo.thermalState),
                          lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, onBattery: onBattery(),
                          swapUsedBytes: swap)
    }

    package static func thermalLabel(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// The source the machine draws from: `nil` when IOKit does not say (a desktop says "AC Power").
    private static func onBattery() -> Bool? {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() else { return nil }
        return (type as String) == kIOPSBatteryPowerValue
    }

    private static func sysctlString(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return "unknown" }
        return String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// `nil` when the kernel does not know the name.
    private static func sysctlInteger(_ name: String) -> Int? {
        var value: Int64 = 0
        var size = MemoryLayout<Int64>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
        return size == 4 ? Int(Int32(truncatingIfNeeded: value)) : Int(value)
    }

    /// The `AGXAccelerator`'s `gpu-core-count` in the IORegistry — Metal does not say it.
    private static func gpuCoreCount() -> Int {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iterator) == KERN_SUCCESS else {
            return 0
        }
        defer { IOObjectRelease(iterator) }
        var service = IOIteratorNext(iterator)
        while service != 0 {
            defer { IOObjectRelease(service); service = IOIteratorNext(iterator) }
            if let value = IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? Int, value > 0 {
                return value
            }
        }
        return 0
    }

    /// The app's version; else, for a binary built from a clone, its commit — read with `git` only when a
    /// `.git` lies above the executable (a Mac without the developer tools would otherwise be asked to
    /// install them by a dialog).
    private static func binaryVersion() -> String {
        for bundle in enclosingBundles() {
            if let short = bundle.infoDictionary?["CFBundleShortVersionString"] as? String {
                let build = bundle.infoDictionary?["CFBundleVersion"] as? String
                return short + (build.map { " (\($0))" } ?? "")
            }
        }
        guard var folder = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent() else { return "unknown" }
        while folder.path != "/" {
            if FileManager.default.fileExists(atPath: folder.appendingPathComponent(".git").path) {
                guard let commit = git(["rev-parse", "--short=10", "HEAD"], in: folder), !commit.isEmpty else { return "unknown" }
                let dirty = !(git(["status", "--porcelain", "--untracked-files=no"], in: folder) ?? "").isEmpty
                return "git " + commit + (dirty ? "-dirty" : "")
            }
            folder = folder.deletingLastPathComponent()
        }
        return "unknown"
    }

    private static func git(_ arguments: [String], in folder: URL) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", folder.path] + arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// ── the common measures — shared by the diagnostic and every check of the CLI ──────────────────

/// Relative error channel by channel, each normalized by its own norm — **bounded from below**.
///
/// The denominator went wrong twice, in both directions. Too large at first: a global
/// `‖a−b‖/‖b‖` is a test on channel 85, which carries 94 to 99.7 % of the energy, and it
/// underestimates a fault elsewhere by 373×. Too small afterwards: a channel worth 3.5·10⁻⁵ when the
/// median is 9·10⁻² turns 7.45·10⁻⁹ of absolute error — the noise of a cancellation over 256
/// terms — into 2.1·10⁻⁴ relative, and cries foul.
///
/// Hence the floor at 1 % of the median norm: a channel weaker than that carries less than 10⁻⁴ of
/// the energy, and its relative precision decides nothing. Sensitivity remains whole over four
/// orders of magnitude, ample to cover a massive activation, which is **above** the
/// median and not below it.
///
/// A library function (`package`), shared by the diagnostic and every check, so a user's figure
/// reads exactly like a check's.
package func channelError(_ got: [Float], _ expected: UnsafeBufferPointer<Float>, columns: Int)
    -> (worst: Double, channel: Int, median: Double) {
    relativeErrorByColumn(got, expected, columns: columns)
}

/// The same against an **fp64** truth — the diagnostic's exact, which an fp32 copy would round by
/// 6·10⁻⁸ per value, the size of what it judges.
package func channelError(_ got: [Float], _ expected: UnsafeBufferPointer<Double>, columns: Int)
    -> (worst: Double, channel: Int, median: Double) {
    relativeErrorByColumn(got, expected, columns: columns)
}

private func relativeErrorByColumn<E: BinaryFloatingPoint>(_ got: [Float], _ expected: UnsafeBufferPointer<E>, columns: Int)
    -> (worst: Double, channel: Int, median: Double) {
    let rows = expected.count / columns
    var numerator = [Double](repeating: 0, count: columns)
    var denominator = [Double](repeating: 0, count: columns)
    for row in 0..<rows {
        for column in 0..<columns {
            let i = row * columns + column
            let e = Double(expected[i])
            let d = Double(got[i]) - e
            numerator[column] += d * d
            denominator[column] += e * e
        }
    }
    var norms = denominator.filter { $0 > 0 }.map { $0.squareRoot() }
    guard !norms.isEmpty else { return (0, 0, 0) }
    norms.sort()
    let floor = 0.01 * norms[norms.count / 2]

    // **`r > worst` is FALSE for a `NaN`.** A boundary that was entirely non-finite therefore came
    // out as "0.000e+00", median `nan`, verdict ✓ — in the project's most used comparison
    // function. A non-finite is not a small deviation: it is worth infinity, and it must make any
    // threshold fail.
    var worst = 0.0, channel = 0, live: [Double] = []
    for column in 0..<columns where denominator[column] > 0 {
        var r = numerator[column].squareRoot() / max(denominator[column].squareRoot(), floor)
        if r.isNaN { r = .infinity }
        live.append(r)
        if r > worst { worst = r; channel = column }
    }
    live.sort()
    return (worst, channel, live[live.count / 2])
}

/// **Which error judges a stage: the cumulative one, or its own.**
///
/// A check compares each boundary to the oracle's: the error it reads is **cumulative**, the
/// stage's own plus what it does to the error it receives. On a well-conditioned stage the second
/// term stays the size of the first, and the cumulative error is the right judge. On a badly
/// conditioned one it does not: the Flux encoder's one-head `mid_block` attention (logits up to 435)
/// takes 1.2·10⁻⁵ of fp32 rounding at its input to 6·10⁻⁴ at its output, where torch fp32 on MPS
/// lands at 8.6·10⁻⁴ and torch on CPU at 1.8·10⁻⁴ — the oracle is one draw among the fp32
/// implementations, and a ✗ there says nothing about the engine.
///
/// Hence the rule: the **amplification** is measured — the stage's output error over its input
/// error, both against the oracle — and above `amplificationLimit` the stage is judged on its
/// **local** error: fed with the oracle's input, its output against the oracle's, at the same
/// threshold. A fault inside the stage shows there whole; only the inherited error is set aside, and
/// the stages after it stay judged cumulatively. Never excused: an input identical to the oracle's
/// (the amplification is undefined, and the cumulative error already is the local one), and a
/// non-finite output.
package enum StageJudgement {
    /// Above this measured amplification, a stage is judged on its local error.
    package static let amplificationLimit = 10.0

    /// `output / input`, or `nil` when the input carries no error (or the figures are not finite).
    package static func amplification(inputError: Double, outputError: Double) -> Double? {
        guard inputError > 0, inputError.isFinite, outputError.isFinite else { return nil }
        return outputError / inputError
    }

    /// Whether the stage is judged on its local error rather than the cumulative one.
    package static func judgesLocally(inputError: Double, outputError: Double) -> Bool {
        guard let a = amplification(inputError: inputError, outputError: outputError) else { return false }
        return a > amplificationLimit
    }
}

/// **The peak `phys_footprint`, sampled** every 10 ms on a thread of its own — shared by
/// the checks that report a peak.
package final class FootprintPeak: @unchecked Sendable {
    private let lock = NSLock()
    private var peak = 0, running = true
    package init() {
        Thread.detachNewThread { [self] in
            while true {
                let now = Arena.processFootprint()
                lock.lock()
                peak = max(peak, now)
                let go = running
                lock.unlock()
                if !go { return }
                usleep(10_000)
            }
        }
    }
    package var value: Int { lock.lock(); defer { lock.unlock() }; return max(peak, Arena.processFootprint()) }
    /// Stops the sampling and returns the peak.
    @discardableResult package func stop() -> Int {
        lock.lock(); running = false; lock.unlock()
        return value
    }
}
