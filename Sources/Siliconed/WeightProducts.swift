import Metal

/// **A product by a published weight, whatever the model**: `C[m, n] = A[m, k] · W`, plus the weight's
/// LoRA stack — the one place where the engine chooses how a weight is read and multiplied. A model
/// says what it computes (the weight's name, the shapes, its buffers); the settings, which the capacity
/// map decides (`CapacityRules`), say how.
///
///     name ─▶ the weight into the reserve, by the shortest path the settings allow
///              ├─ gemm_tensor, a bf16 weight   copied as it is (`Artifact.stage`)       ─▶ TensorGEMM
///              │     gemm_bf16_act             the same, the activation rounded to bf16 first ─▶ TensorGEMM
///              │                               (`roundsActivations`, `BFloatKernels`)
///              ├─ gemm_int8, a published int8  copied as it is, scale fp32 after it     ─▶ TensorGEMM, × s after the
///              │                                (convrot: x · R by chunks of rows in the reserve's tail)  sums
///              │     gemm_q8, a GGUF Q8_0       (scales per 32 rows: q · s in threadgroup memory)
///              │     + the AMX                  its columns widened alone, while the GPU   ─▶ BNNS on them
///              │                                computes the others
///              └─ otherwise                    widened on the CPU (`Artifact.materialize`) ─▶ MPS (+ AMX)
///          ─▶ C = A · W
///          ─▶ + the LoRA stack:  C += (A · down) · up          after the product, or
///                                W += down · up, then C = A · W  merged, where it costs fewer FLOPs
///
/// Every path computes in fp32 from the published values (a bf16 widening is a shift, an 8-bit one a
/// single multiplication): they differ in the order of the sums, hence the last bits, never in what is
/// computed. The default path is the one every model ran before this module, bit for bit.
///
/// What sets it apart — what a model writing its own product would get wrong:
///
///   - **the LoRA stack is added after the product, whichever path made it**: so a bf16-weight product
///     takes a LoRA like any other (the guard that kept Krea 2 under a LoRA widened was such a
///     local exception). The merge (`mergesLoRA`, Qwen-Image-2.1's: a rank-256 stack after the GEMM cost
///     2.0 s per 1024² evaluation, ∝ rows) needs an fp32 weight to add into: under `gemm_tensor` it falls
///     back to after the product;
///   - **the reserve is the model's, sized by its memory need** (`MemoryNeed`): this module wraps it, it
///     never allocates — a bf16 weight takes its first half, an int8 one its first quarter, then its
///     scales, then the rotated activation by as many rows as the rest holds (`int8Plan`);
///   - **an int8 weight's rotation goes to the activation**: `W = R · q · s` (`Widen.dequantizeRotated`)
///     gives `x · W = (x · R) · q · s` — `R` is applied to `x` in fp32, the integers are never touched;
///   - **`gemm_bf16_act` rounds the operand of a product and nothing else** (`roundsActivations`, which a
///     DiT sets and a text encoder does not): the caller's activation stays fp32 — the LoRA stack reads it —,
///     the rounded copy lives in the reserve's tail beside the weight (`roundingPlan`), the sums and the
///     output are fp32. Under `bf16ActivationMinimumRows` rows (the modulation, the embedders) the product
///     keeps its fp32 operand: nothing to win there, and the most amplified error;
///   - **the timings keep their names** (`staging`, `widening`, `gemm wall`, `gemm GPU`, `lora weights`,
///     `lora wall`, `lora GPU`): the models add them to theirs, so a budget reads the same as before.
package final class WeightProducts {
    private let gemm: GEMM
    private let artifact: Artifact
    private let reserve: UnsafeMutablePointer<Float>
    /// The reserve's floats: a bf16 weight may take twice as many values.
    private let capacity: Int
    private let transposed: Bool
    private let lora: (any LoRAStacks)?
    private let mid: UnsafeMutablePointer<Float>?
    private let midFloats: Int
    private let mergesLoRA: Bool
    /// The operand of a product is rounded to bf16 where the settings and the machine allow it (`gemm_bf16_act`).
    private let roundsActivations: Bool

    /// Per phase, accumulated since the last `resetTimings`.
    package private(set) var timings: [String: Double] = [:]
    package func resetTimings() { timings.removeAll() }

    /// - Parameters:
    ///   - reserve: the model's weight reserve, `capacity` floats (the largest weight it multiplies by).
    ///   - transposed: the weights are laid out `[k, n]` by the forge (`Artifact.linearWeightsTransposed`).
    ///   - lora: the stack per module, `nil` without a LoRA; `mid`, its `[rows, rank]` scratch of `midFloats`.
    ///   - mergesLoRA: the stack may be merged into the widened weight where it costs fewer FLOPs.
    package init(gemm: GEMM, artifact: Artifact, reserve: UnsafeMutablePointer<Float>, capacity: Int,
                 transposed: Bool, lora: (any LoRAStacks)? = nil, mid: UnsafeMutablePointer<Float>? = nil,
                 midFloats: Int = 0, mergesLoRA: Bool = false, roundsActivations: Bool = false) {
        self.gemm = gemm; self.artifact = artifact; self.reserve = reserve; self.capacity = capacity
        self.transposed = transposed; self.lora = lora; self.mid = mid; self.midFloats = midFloats
        self.mergesLoRA = mergesLoRA; self.roundsActivations = roundsActivations
    }

    private func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        let result = try body()
        timings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        return result
    }

    /// **`C = A · W` (+ the stack of `target`)**, `W` the map's `name`. `a` and `c` are the model's
    /// buffers, wrapped at their reserved size (`GEMM.wrap` memoizes by address). `target`: the module
    /// the LoRA stack is keyed by — `name` without `.weight` unless said.
    package func linear(_ name: String, a: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                        target: String? = nil) throws {
        let w = try gemm.wrap(UnsafeMutableRawPointer(reserve), bytes: capacity * 4, name: "w")
        let path = try prepare(name, count: k * n, m: m, n: n)
        let staged = path != .widened
        let target = target ?? String(name.dropLast(".weight".count))
        var stack: (down: MTLBuffer, up: MTLBuffer, rank: Int)?
        if let lora, let module = try timed("lora weights", { try lora.module(target, k: k, n: n) }) {
            stack = (try gemm.wrap(UnsafeMutableRawPointer(module.down), bytes: k * module.rank * 4, name: "loraDown"),
                     try gemm.wrap(UnsafeMutableRawPointer(module.up), bytes: module.rank * n * 4, name: "loraUp"),
                     module.rank)
        }
        let merged = stack != nil && mergesLoRA && !staged && LoRA.merges(rows: m, k: k, n: n)
        if merged, let stack {
            timed("lora wall") {
                timings["lora GPU", default: 0] += gemm.accumulatedLinear(a: stack.down, b: stack.up, c: w,
                                                                         m: k, k: stack.rank, n: n)
            }
        }
        var failure: String?
        timed("gemm wall") {
            switch path {
            case .bf16(let rounding?):
                timings["gemm GPU", default: 0] += gemm.linear(a: a, rounding: (w, rounding.offset, rounding.rows), bf16: w,
                                                              c: c, m: m, k: k, n: n, weightIsTransposed: transposed)
            case .bf16(nil):
                timings["gemm GPU", default: 0] += gemm.linear(a: a, bf16: w, c: c, m: m, k: k, n: n,
                                                              weightIsTransposed: transposed)
            case .int8(let plan):
                let r = Self.int8Product(gemm: gemm, plan: plan, reserve: w, a: a, c: c, m: m, k: k, n: n)
                timings["gemm GPU", default: 0] += r.gpu
                timings["widening", default: 0] += r.widening
                if r.status != 0 { failure = "\(name): the AMX refused its columns (BNNS \(r.status))" }
            case .widened:
                timings["gemm GPU", default: 0] += gemm.linear(a: a, b: w, c: c, m: m, k: k, n: n,
                                                              weightIsTransposed: transposed)
            }
        }
        // A refused slice leaves its columns as they were: a plausible image, wrong exactly there.
        if let failure { throw Artifact.Failure.badHeader(failure) }
        if !merged, let stack, let mid {
            let mb = try gemm.wrap(UnsafeMutableRawPointer(mid), bytes: midFloats * 4, name: "loraMid")
            timed("lora wall") {
                timings["lora GPU", default: 0] += gemm.lora(x: a, down: stack.down, mid: mb, up: stack.up, c: c,
                                                             m: m, k: k, r: stack.rank, n: n)
            }
        }
    }

    /// How the weight lies in the reserve, hence which product reads it.
    private enum Path: Equatable {
        case widened
        /// A bf16 weight staged as published; its activation rounded into the reserve's tail, or fp32.
        case bf16(Rounding?)
        case int8(Int8Plan)
    }

    /// Where the rounded activation lies in the reserve and how many rows at a time (a multiple of 64).
    package struct Rounding: Equatable {
        package let offset, rows: Int
    }

    /// **Where a bf16 activation fits beside a bf16 weight of `k · n` values** in a reserve of `capacity`
    /// floats: after the weight's pages, `rows` rows of `k` bf16 values, or `nil` when fewer than 64 rows
    /// would (the reserve is the model's, never grown for this path) or `k` is odd.
    package static func roundingPlan(k: Int, n: Int, capacity: Int) -> Rounding? {
        guard k % 2 == 0 else { return nil }
        let offset = (2 * k * n + 16383) / 16384 * 16384
        let rows = (capacity * 4 - offset) / (2 * k) / 64 * 64
        return rows >= 64 ? Rounding(offset: offset, rows: rows) : nil
    }

    /// An int8 weight in the reserve: `q` at 0, its fp32 scales at `scalesOffset`, the AMX's last
    /// `amxColumns` widened at `sliceOffset`, the rotated activation (convrot of `group`) at
    /// `rotatedOffset`, `chunk` rows at a time. Offsets in bytes, page-aligned.
    package struct Int8Plan: Equatable {
        package let scalesOffset, sliceOffset, rotatedOffset, chunk, amxColumns: Int
        package let perColumn: Bool
        package let group: Int?
        /// Scales per block of this many rows and column (GGUF Q8_0); `nil`: per column or tensor.
        package var block: Int? = nil
        /// The activation rounded to bf16 (`gemm_bf16_act`) at this offset, `chunk` rows at a time; 0: fp32.
        package var roundedOffset = 0
        package var layout: Widen.ScaleLayout {
            if let block { return .blocks(columns: 0, rows: block) }
            return perColumn ? .columns(0) : .tensor
        }
    }

    /// Where an int8 `[k, n]` weight, the AMX's widened columns and the rotated activation fit in
    /// `capacity` floats, or `nil` when fewer than 64 rows would: the reserve is the model's, never
    /// grown for this path.
    package static func int8Plan(k: Int, n: Int, capacity: Int, perColumn: Bool, group: Int?, amxColumns: Int,
                                 block: Int? = nil, rounded: Bool = false) -> Int8Plan? {
        func page(_ bytes: Int) -> Int { (bytes + 16383) / 16384 * 16384 }
        let scales = block.map { (k + $0 - 1) / $0 * n } ?? (perColumn ? n : 1)
        let scalesOffset = page(k * n), sliceOffset = page(scalesOffset + scales * 4)
        let rotatedOffset = page(sliceOffset + k * amxColumns * 4)
        guard rotatedOffset <= capacity * 4 else { return nil }
        if rounded {
            // The rotated rows (fp32, only under a rotation) then the same rows rounded (bf16): 6 k bytes a row.
            guard k % 2 == 0 else { return nil }
            let rows = (capacity * 4 - rotatedOffset) / (k * (group == nil ? 2 : 6)) / 64 * 64
            guard rows >= 64 else { return nil }
            let roundedOffset = group == nil ? rotatedOffset : (rotatedOffset + rows * k * 4 + 16383) / 16384 * 16384
            guard roundedOffset + rows * k * 2 <= capacity * 4 else { return nil }
            return Int8Plan(scalesOffset: scalesOffset, sliceOffset: sliceOffset, rotatedOffset: rotatedOffset,
                            chunk: rows, amxColumns: amxColumns, perColumn: perColumn, group: group, block: block,
                            roundedOffset: roundedOffset)
        }
        let rows = (capacity * 4 - rotatedOffset) / (k * 4) / 64 * 64
        guard group == nil || rows >= 64 else { return nil }
        return Int8Plan(scalesOffset: scalesOffset, sliceOffset: sliceOffset, rotatedOffset: rotatedOffset,
                        chunk: rows, amxColumns: amxColumns, perColumn: perColumn, group: group, block: block)
    }

    /// **The GPU's share of a convrot int8 product's columns** beside the AMX, `nil` for the GPU alone.
    /// The check sweeps it; otherwise the conductor's share of rows for the shape.
    package var int8Share: ((_ k: Int, _ n: Int, _ m: Int) -> Double)?

    /// **The int8 product of a plan, the weight already in `reserve`** (`q`, its fp32 scales): the GPU on
    /// the first columns — all of them without a conductor or AMX columns —, the CPU widening the last ones
    /// meanwhile, the AMX on them. What the map measures (`Diagnostic`) is this function. Returns the GPU
    /// seconds, the widening's, BNNS's status (0, or its refusal: those columns are then unwritten).
    package static func int8Product(gemm: GEMM, plan: Int8Plan, reserve w: MTLBuffer, a: MTLBuffer, c: MTLBuffer,
                                    m: Int, k: Int, n: Int) -> (gpu: Double, widening: Double, status: Int32) {
        let rotation = plan.group.map { (group: $0, scratch: w, offset: plan.rotatedOffset, chunk: plan.chunk) }
        let rounding = plan.roundedOffset > 0 ? (buffer: w, offset: plan.roundedOffset, rows: plan.chunk) : nil
        if rounding != nil { GEMM.roundedProducts.withLock { $0 += 1 } }
        guard plan.amxColumns > 0, let conductor = gemm.conductor else {
            return (gemm.tensor!.linear(a: a, int8: w, scales: w, scalesOffset: plan.scalesOffset, perColumn: plan.perColumn,
                                        block: plan.block, rotation: rotation, rounding: rounding,
                                        c: c, m: m, k: k, n: n), 0, 0)
        }
        let g = n - plan.amxColumns
        let commands = gemm.tensor!.submit(a: a, int8: w, scales: w, scalesOffset: plan.scalesOffset, perColumn: plan.perColumn,
                                           block: plan.block, rotation: rotation, c: c, m: m, k: k, n: n, columns: g)
        let raw = w.contents(), slice = raw + plan.sliceOffset, scales = raw + plan.scalesOffset
        let started = DispatchTime.now().uptimeNanoseconds
        if let group = plan.group {
            Widen.dequantizeRotated(raw, rows: k, columns: n, group: group, scale: scales,
                                    scaleType: .float32, layout: plan.perColumn ? .columns(n) : .tensor,
                                    columns: g..<n, into: slice.assumingMemoryBound(to: Float.self))
        } else {
            Widen.dequantizeColumns(raw, rows: k, columns: n, scales: scales.assumingMemoryBound(to: Float.self),
                                    layout: plan.layout, columns: g..<n, into: slice.assumingMemoryBound(to: Float.self))
        }
        let widening = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        let status = conductor.columns(a: a.contents(), slice: slice, c: c.contents(), m: m, k: k, n: n,
                                       first: g, width: plan.amxColumns)
        commands.waitUntilCompleted()
        return (Attention.verdict(commands, what: "matmul2d float × int8 \(m)×\(k)×\(n)"), widening, status)
    }

    /// The AMX's columns: a multiple of 64 (the GPU's tile), 0 without a conductor or under its
    /// `minimumRows`.
    private func amxColumns(k: Int, n: Int, m: Int) -> Int {
        guard n >= 128, let conductor = gemm.conductor, m >= Conductor.minimumRows else { return 0 }
        // A class the table keeps off the AMX (`gpu/int8`, or any engines but the AMX's) is the GPU's whole.
        if let forced = gemm.routes.products.forced(k: k, n: n, m: m, aneBroughtUp: gemm.neuralEngine != nil),
           forced != .amx, forced != .aneAMX { return 0 }
        let T = int8Share?(k, n, m) ?? conductor.shares.share(k: k, n: n, m: m) ?? conductor.initialFraction
        guard T < 1 else { return 0 }
        return Self.amxColumns(n: n, share: T)
    }

    /// The AMX's columns of `n` at the GPU's share `T`, held in `Conductor.fractionRange` as the rows' —
    /// what `MemoryNeed.gemm` bounds the BNNS workspace by. **The GPU's columns `n − amx` are a
    /// multiple of 64**, not the AMX's: `linear_int8` / `linear_q8` write whole 64 × 64 tiles clipped
    /// only at `n`, so a GPU edge inside a tile would overwrite the AMX's first columns. When
    /// `n % 64 ≠ 0` the AMX takes the ragged remainder; under 128 columns, nothing.
    package static func amxColumns(n: Int, share T: Double) -> Int {
        let ceiling = (n - 64) / 64 * 64                 // the GPU's most, leaving the AMX ≥ 64
        guard ceiling >= 64 else { return 0 }
        let T = min(Conductor.fractionRange.upperBound, max(Conductor.fractionRange.lowerBound, T))
        return n - max(64, min(ceiling, Int((T * Double(n) / 64).rounded()) * 64))
    }

    /// The weight into the reserve, by the path the settings allow.
    private func prepare(_ name: String, count: Int, m: Int, n: Int) throws -> Path {
        func expect(_ got: Int) throws {
            guard got == count else { throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected") }
        }
        // The route table says how this product reads its weight (`GEMM.weightRoute`): the bf16 weight as
        // published, its operand rounded to bf16 where the product is tall enough to gain (`gemm_bf16_act`),
        // the 8-bit weight as published, or widened.
        if let tensor = artifact.tensors[name], tensor.dtype == .bfloat16 {
            var route = gemm.weightRoute(.bf16, k: count / n, n: n, m: m, rounding: roundsActivations)
            var rounding: Rounding?
            if route.path == .bf16 && route.roundsActivation {
                rounding = Self.roundingPlan(k: count / n, n: n, capacity: capacity)
                // The rounding does not fit the reserve: the rule below it (the plain bf16 weight, if any).
                if rounding == nil { route = gemm.weightRoute(.bf16, k: count / n, n: n, m: m, rounding: false) }
            }
            if route.path == .bf16, let got = try timed("staging", {
                try artifact.stage(name, into: UnsafeMutableRawPointer(reserve), capacity: 2 * capacity)
            }) {
                try expect(got)
                return .bf16(rounding)
            }
        }
        if transposed, let tensor = artifact.tensors[name], tensor.dtype == .int8,
           let scale = tensor.scale, tensor.shape == [count / n, n],
           scale.block == nil ? scale.shape.count <= 1 : scale.shape.count == 2,
           case let route = gemm.weightRoute(scale.block == nil ? .int8 : .q8, k: count / n, n: n, m: m, rounding: roundsActivations),
           route.path == .native {
            let (k, perColumn, group) = (count / n, scale.shape.count == 1, tensor.rotation?.group)
            let roundsHere = route.roundsActivation
            // Rounded, the GPU alone: the AMX reads the fp32 activation, one rounding for some columns only.
            guard let plan = (roundsHere ? Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: perColumn, group: group,
                                                         amxColumns: 0, block: scale.block, rounded: true) : nil)
                    ?? Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: perColumn, group: group,
                                     amxColumns: amxColumns(k: k, n: n, m: m), block: scale.block)
                    ?? Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: perColumn, group: group, amxColumns: 0,
                                     block: scale.block)
            else { return try widen(name, count: count) }
            try expect(tensor.count)
            try timed("staging") { try artifact.stageInt8(name, into: UnsafeMutableRawPointer(reserve),
                                                          scalesAt: plan.scalesOffset) }
            return .int8(plan)
        }
        return try widen(name, count: count)
    }

    /// The weight widened to fp32 into the reserve, on the CPU.
    private func widen(_ name: String, count: Int) throws -> Path {
        let got = try timed("widening") { try artifact.materialize(name, into: reserve, capacity: capacity) }
        guard got == count else { throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected") }
        return .widened
    }
}

/// **The LoRA stack of one module, expanded**: `down [k, r]` and `up [r, n]`, or `nil` when no LoRA of
/// the stack touches it. `CacheLoRA` keeps them from one evaluation to the next, `StreamedLoRA` streams
/// Qwen-Image-2.1's by layer, `ExpandedLoRA` expands them at each call into two fixed slices.
package protocol LoRAStacks: AnyObject {
    func module(_ target: String, k: Int, n: Int) throws
        -> (down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>, rank: Int)?
}

extension CacheLoRA: LoRAStacks {}
extension StreamedLoRA: LoRAStacks {}

/// The stack expanded afresh at every call into `down` and `up` (sized for the largest module), as
/// Anima and Krea 2's text fusion do: no cache to charge, the expansion paid at each product.
package final class ExpandedLoRA: LoRAStacks {
    private let lora: LoRA
    private let down, up: UnsafeMutablePointer<Float>
    package init(lora: LoRA, down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>) {
        self.lora = lora; self.down = down; self.up = up
    }
    package func module(_ target: String, k: Int, n: Int) throws
        -> (down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>, rank: Int)? {
        guard lora.rank(target) > 0 else { return nil }
        return (down, up, try lora.materialize(target, k: k, n: n, down: down, up: up))
    }
}
