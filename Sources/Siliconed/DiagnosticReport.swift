// The diagnostic's report: the JSON types (machine, conditions, measures, products, cuts,
// witness), their encoding and the issue link. Split from `Diagnostic.swift`, which says what each figure decides.

import CryptoKit
import Foundation
import IOKit
import IOKit.ps
import Metal

extension Diagnostic {
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

    /// **What the map was allowed, computed, and took** (`MemoryNeedDiagnostic.swift`): B, the process's base when
    /// it started, the ANE engines it cut the grid into, each step's need (bytes, without the base) and its peak
    /// `phys_footprint` (with it, the kernel's interval maximum). A peak above `base + need` is a defect of the code.
    package struct Memory: Codable, Sendable, Equatable {
        package var budget, base: Int
        /// ANE engines the grid went through, one after the other.
        package var aneBatches: Int
        package var need: [String: Int]
        package var peak: [String: Int]

        package init(plan: Plan, peaks: [String: Int], aneBatches: Int? = nil) {
            budget = plan.budget; base = plan.base; need = plan.need.steps; peak = peaks
            self.aneBatches = aneBatches ?? Diagnostic.aneBatches(allotment: plan.aneAllotment).count
        }

        /// **The memory held**: every step's peak under `base + need`, and the largest of them under B.
        package var held: Bool {
            peak.allSatisfy { step, bytes in bytes <= base + (need[step] ?? 0) } && base + (need.values.max() ?? 0) <= budget
        }
    }


    // ── the map of schema 13: classes, regime, verdict ───────────────────────────────────────

    /// **One class of the map, as the report keeps it**: what it computes, its weight, the route chosen, the
    /// loss that stays on it (a fraction of the render), the pairs spent on it and, per route that was a candidate, its
    /// hot-time ratio to the reference and the log standard error of that ratio. One line of a JSON array, the ids
    /// being the bytes of the issue link: `[id, weight, chosen, loss, runs, status, [[route, ratio, error], …]]`.
    package struct MapClass: Codable, Sendable, Equatable {
        package var id: String
        package var weight: Double
        package var chosen: String?
        package var loss: Double
        package var runs: Int
        package var status: CapacityMeasure.Status
        package var timings: [Timing]

        package struct Timing: Codable, Sendable, Equatable {
            package var route: String
            package var ratio: Double
            package var error: Double
            package init(route: String, ratio: Double, error: Double) {
                self.route = route; self.ratio = Diagnostic.significant(ratio, 3); self.error = Diagnostic.significant(error, 2)
            }
            package init(from decoder: any Decoder) throws {
                var c = try decoder.unkeyedContainer()
                route = try c.decode(String.self); ratio = try c.decode(Double.self); error = try c.decode(Double.self)
            }
            package func encode(to encoder: any Encoder) throws {
                var c = encoder.unkeyedContainer()
                try c.encode(route); try c.encode(Diagnostic.significant(ratio, 3)); try c.encode(Diagnostic.significant(error, 2))
            }
        }

        package init(id: String, weight: Double, chosen: String?, loss: Double, runs: Int, status: CapacityMeasure.Status, timings: [Timing]) {
            self.id = id; self.weight = Diagnostic.significant(weight, 3); self.chosen = chosen
            self.loss = Diagnostic.significant(loss, 2); self.runs = runs; self.status = status
            self.timings = timings
        }

        private static let letters: [CapacityMeasure.Status: String] = [.closed: "c", .open: "o", .unmeasured: "u", .none: "n"]
        package init(from decoder: any Decoder) throws {
            var c = try decoder.unkeyedContainer()
            id = try c.decode(String.self); weight = try c.decode(Double.self); chosen = try c.decode(String?.self)
            loss = try c.decode(Double.self); runs = try c.decode(Int.self)
            let letter = try c.decode(String.self)
            guard let s = Self.letters.first(where: { $0.value == letter })?.key else {
                throw DecodingError.dataCorruptedError(in: c, debugDescription: "unknown status « \(letter) »")
            }
            status = s
            timings = try c.decode([Timing].self)
        }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.unkeyedContainer()
            try c.encode(id); try c.encode(Diagnostic.significant(weight, 3)); try c.encode(chosen)
            try c.encode(Diagnostic.significant(loss, 2)); try c.encode(runs); try c.encode(Self.letters[status]!)
            try c.encode(timings)
        }
    }

    /// **What the plan spent and what it leaves**: the classes, the residual loss against ε (fractions of the render), the
    /// seconds of the plan against its budget, the share of the catalogue under the cut, the seconds of exactness checks,
    /// and the routes judged wrong (`class route`) — excluded from their class only.
    package struct MapResult: Codable, Sendable, Equatable {
        package var classes: [MapClass]
        package var residualLoss: Double
        package var epsilon: Double
        package var spent: Double
        package var budget: Double
        package var exhausted: Bool
        package var tail: Double
        package var judged: Double
        package var wrong: [String]
        /// The families the map decided for (`Family.rawValue`): a map made for the models installed does not speak of the others.
        package var families: [String]

        package init(classes: [MapClass], residualLoss: Double, epsilon: Double, spent: Double, budget: Double, exhausted: Bool,
                     tail: Double, judged: Double, wrong: [String], families: [String] = []) {
            self.families = families
            self.classes = classes
            self.residualLoss = Diagnostic.significant(residualLoss, 3); self.epsilon = epsilon
            self.spent = Diagnostic.significant(spent, 3); self.budget = Diagnostic.significant(budget, 3)
            self.exhausted = exhausted; self.tail = Diagnostic.significant(tail, 2)
            self.judged = Diagnostic.significant(judged, 2); self.wrong = wrong
        }
    }

    /// **The regime the table was decided at**: each engine's witness at the start and at the end of the map, and the
    /// heat factor that came out of it (1 when inside the witness's own noise). Facts, not verdicts.
    package struct Regime: Codable, Sendable, Equatable {
        package var witnesses: [CapacityMap.EngineWitness]
        package var heat: [String: Double]
        /// **What the renders taught** (`RenderLoop`): a factor per engine, multiplying `heat` to predict a render. The routes were
        /// decided at `heat`; a regime learned does not move them, it moves the prediction — and a correction that would move a
        /// route is a map to measure again. `nil`: nothing learned.
        package var learned: [String: Double]?
        package init(witnesses: [CapacityMap.EngineWitness]) {
            self.witnesses = witnesses.map {
                CapacityMap.EngineWitness(engine: $0.engine, before: Diagnostic.significant($0.before, 4), after: Diagnostic.significant($0.after, 4),
                                          errorBefore: Diagnostic.significant($0.errorBefore, 2), errorAfter: Diagnostic.significant($0.errorAfter, 2))
            }
            heat = CapacityMap.heat(witnesses).factors.mapValues { Diagnostic.significant($0, 4) }
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
        /// The map's own memory (`Memory`); `nil` in a report that predates it. No decision reads it.
        package var memory: Memory?
        package var thresholds: Thresholds
        /// **What these measures make this Mac use** (`Diagnostic.decide`), written so a shared report says
        /// it; `nil` in a report that predates it. The engine never reads this field: it applies the
        /// rules to the measures again (`EngineSettings.load(from:map:)`).
        package var decisions: Decisions?
        /// **The route table this map made** (schema 13): what the engine reads (`EngineSettings.load(from:map:)`).
        package var routes: RouteTable?
        /// The classes it decided, the loss that stays, the seconds it spent (schema 13).
        package var map: MapResult?
        /// The regime it was decided at (schema 13).
        package var regime: Regime?
        /// Green or orange, and why (schema 13, `CapacityMap.Verdict`).
        package var verdict: CapacityMap.Verdict?
        /// **The last render against this map's prediction** (`RenderLoop.Note`): the report an app sends carries it; the map itself
        /// is kept without it.
        package var render: RenderLoop.Note?

        package init(schema: Int = Diagnostic.schema, date: String, seconds: Double, machine: Machine,
                    cpuFeatures: [String: Int], settings: [String: String] = [:], before: Conditions, after: Conditions,
                    products: [Product], lowPrecision: [LowPrecision], attention: [AttentionCapacity],
                    widening: [Widening], witness: Witness, thresholds: Thresholds = Thresholds(),
                    stages: [String: Double]? = nil, cuts: [Cut]? = nil, aneCuts: [Cut]? = nil,
                    memory: Memory? = nil, routes: RouteTable? = nil, map: MapResult? = nil, regime: Regime? = nil,
                    verdict: CapacityMap.Verdict? = nil) {
            self.schema = schema; self.date = date; self.seconds = Diagnostic.significant(seconds, 3)
            self.thresholds = thresholds
            self.machine = machine; self.cpuFeatures = cpuFeatures; self.settings = settings
            self.before = before; self.after = after; self.products = products; self.lowPrecision = lowPrecision
            self.attention = attention; self.widening = widening; self.witness = witness; self.cuts = cuts
            self.aneCuts = aneCuts; self.memory = memory
            self.stages = stages?.mapValues { Diagnostic.significant($0, 3) }
            self.routes = routes; self.map = map; self.regime = regime; self.verdict = verdict
            // A map of schema 13 decides by its classes; the rules of `CapacityRules.swift` are for the older reports.
            self.decisions = map != nil ? nil : Diagnostic.decide(products: products, attention: attention, widening: widening,
                                                                  witness: witness, cuts: cuts ?? [], aneCuts: aneCuts ?? [])
        }

        /// What the report is valid for: the machine and the binary that measured it.
        package var fingerprint: Fingerprint { Fingerprint(machine) }
        /// **The report still describes this process's machine and build** — what an app checks before
        /// handing it to `EngineSettings.load(from:map:)`; when it does not, the app offers to measure again.
        package var isCurrent: Bool { schema == Diagnostic.schema && fingerprint == .current }

        /// **The map is green (M6)**: complete, every route it uses judged right, its peak under its need under B, its seconds
        /// in the budget. Not a question of the hour — heat, the battery, Low Power Mode and another process's swap are
        /// facts of the regime (`before`, `after`, `regime`), and the pairs measure through them. A report of an
        /// older schema is never green: it is not read (`isCurrent`).
        package var isGreen: Bool { verdict?.green == true }

        /// **The map speaks of these families** (it was made for them, or for more). An app that installs a model of a family the
        /// map did not decide for measures again.
        package func covers(_ families: Set<String>) -> Bool {
            map.map { Set($0.families).isSuperset(of: families) } ?? false
        }

        /// Every measure of the report, in its order.
        package var measures: [Measure] {
            products.flatMap { [$0.gpu] + [$0.amx, $0.coexecution, $0.tensor, $0.bf16Act, $0.int8, $0.int8Coexecution, $0.q8, $0.q8Coexecution].compactMap { $0 } }
                + lowPrecision.compactMap(\.measure)
                + attention.flatMap { [$0.graph, $0.flash, $0.headSplit].compactMap { $0 } }
                + widening.map(\.cpu)
        }
        /// Every engine that ran computed right.
        package var passes: Bool { measures.allSatisfy(\.pass) && map?.wrong.isEmpty != false }

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
            // The table is what the classes' timings and the grids' shares make: the link keeps the timings.
            measures.routes = nil
            // …and the heaviest classes, each with the chosen route's ratio and dispersion: the rest is in the JSON the app keeps and
            // the clipboard (`includingReport: false`), the link's bytes are counted.
            if var map = measures.map {
                map.classes = map.classes.prefix(Diagnostic.linkClasses).map { c in
                    var c = c; c.timings = c.timings.filter { $0.route == c.chosen }; return c
                }
                measures.map = map
            }
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

    /// The classes of a map the link to an issue keeps (the heaviest).
    package static let linkClasses = 24

    /// `x` to `digits` significant digits, so the JSON says `3.312`, not `3.3123456789012`; a
    /// non-finite value becomes the largest finite double (JSON has no infinity).
    package static func significant(_ x: Double, _ digits: Int) -> Double {
        guard x.isFinite else { return x.isNaN || x > 0 ? .greatestFiniteMagnitude : -.greatestFiniteMagnitude }
        guard x != 0 else { return 0 }
        return Double(String(format: "%.\(digits - 1)e", x)) ?? x
    }
}
