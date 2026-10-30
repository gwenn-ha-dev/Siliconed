import Metal

/// **A product by a published weight, whatever the model**: `C[m, n] = A[m, k] · W`, plus the weight's
/// LoRA stack — the one place where the engine chooses how a weight is read and multiplied. A model
/// says what it computes (the weight's name, the shapes, its buffers); the plan (`CapacityPlan`, asked at every call with the
/// product's exact class) says how: the form the weight is read in, the rows each engine takes.
///
///     name ─▶ the weight into the reserve, by the shortest path the settings allow
///              ├─ path bf16, a bf16 weight     copied as it is (`Artifact.stage`)       ─▶ TensorGEMM
///              │     rounding                  the same, the activation rounded to bf16 first ─▶ TensorGEMM
///              │                               (`roundsActivations`, `BFloatKernels`)
///              ├─ path native, a published int8  copied as it is, scale fp32 after it   ─▶ TensorGEMM, × s after the
///              │                                (convrot: x · R by chunks of rows in the reserve's tail)  sums
///              │     or a GGUF Q8_0             (scales per 32 rows: q · s in threadgroup memory)
///              │     + the AMX's columns        widened alone, while the GPU               ─▶ BNNS on them
///              │                                computes the others
///              │     + the ANE's last rows      the same integers and scales, read in place ─▶ the ANE's int8 program,
///              │                                (`ProductAssignment.ane` on a native weight)   beside the GPU's rows
///              └─ path widened                 widened on the CPU (`Artifact.materialize`) ─▶ MPS (+ AMX, + ANE rows)
///          ─▶ C = A · W
///          ─▶ + the LoRA stack:  C += (A · down) · up          after the product, or
///                                W += down · up, then C = A · W  merged, where it costs fewer FLOPs
///
/// Every path computes in fp32 from the published values (a bf16 widening is a shift, an 8-bit one a
/// single multiplication): they differ in the order of the sums, hence the last bits, never in what is
/// computed. The default plan's path is the one every model ran before this module, bit for bit.
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
///   - **the ANE reads the integers where the plan gives it rows of a native weight** (`aneInt8Rows`): nothing is widened, nothing
///     copied for it but its surface; the GPU's rows run the int8 kernel on the same reserve, each row one engine's. A program the
///     process cannot run for the weight (`NeuralEngine.accepts`: its rotation's groups, a refusal) hands its rows to the GPU;
///   - **a rounded activation rounds the operand of a product and nothing else** (`roundsActivations`, which a
///     DiT sets and a text encoder does not): the caller's activation stays fp32 — the LoRA stack reads it —,
///     the rounded copy lives in the reserve's tail beside the weight (`roundingPlan`), the sums and the
///     output are fp32. Under the plan's rows (the modulation, the embedders) the product
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
    /// The last int8 product had rows on the ANE, and it computed them (`classification`).
    private var aneTook = false
    /// ... and it took every row of the product: the ledger names the route `ane/int8`, as the plan does (`ProductAssignment.route`).
    private var aneAlone = false

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
        let (path, planned) = try prepare(name, count: k * n, m: m, n: n)
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
        // The render's check (`ProductLedger`): this product's wall seconds, as the class and the route it is.
        let ledger = ProductLedger.current
        let began = ledger == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        timed("gemm wall") {
            switch path {
            case .bf16(let rounding?):
                timings["gemm GPU", default: 0] += gemm.linear(a: a, rounding: (w, rounding.offset, rounding.rows), bf16: w,
                                                              c: c, m: m, k: k, n: n, weightIsTransposed: transposed)
            case .bf16(nil):
                timings["gemm GPU", default: 0] += gemm.linear(a: a, bf16: w, c: c, m: m, k: k, n: n,
                                                              weightIsTransposed: transposed)
            case .int8(let plan):
                // The ANE beside the GPU on the integers: the plan's last rows (`ProductAssignment.ane` on a native weight), read from
                // the reserve beside the GPU's int8 kernel. The engine takes them only if it can run them (`aneInt8Rows`).
                let aneRows = aneInt8Rows(planned, plan, m: m, k: k, n: n)
                if aneRows > 0, let ane = gemm.neuralEngine {
                    let r = Self.int8Product(gemm: gemm, plan: plan, reserve: w, a: a, c: c, m: m, k: k, n: n, ane: ane, aneRows: aneRows)
                    timings["gemm GPU", default: 0] += r.gpu
                    aneTook = r.computed
                    aneAlone = r.computed && aneRows == m
                } else {
                    aneTook = false; aneAlone = false
                    let r = Self.int8Product(gemm: gemm, plan: plan, reserve: w, a: a, c: c, m: m, k: k, n: n)
                    timings["gemm GPU", default: 0] += r.gpu
                    timings["widening", default: 0] += r.widening
                    if r.status != 0 { failure = "\(name): the AMX refused its columns (BNNS \(r.status))" }
                }
            case .widened:
                timings["gemm GPU", default: 0] += gemm.linear(a: a, b: w, c: c, m: m, k: k, n: n,
                                                              weightIsTransposed: transposed, weight: format(of: name),
                                                              assignment: planned)
            }
        }
        if let ledger {
            let (weight, route) = classification(of: name, path)
            ledger.add(.product(k: k, n: n, m: m, weight: weight), route: route,
                       seconds: Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9)
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

    /// **The class of the product that just ran, as the plan names it**: the weight's format (as published) and the route taken: the
    /// engines the widened product ran on (`GEMM.lastEngines`), or the weight's own form (`gpu/bf16`, `gpu/bf16act`, `gpu/int8`,
    /// `gpu+amx/int8`, `gpu/q8`, `gpu+amx/q8`).
    private func classification(of name: String, _ path: Path) -> (CapacityClasses.WeightFormat, String) {
        switch path {
        case .bf16(let rounding): return (.bf16, rounding == nil ? "gpu/bf16" : "gpu/bf16act")
        case .int8(let plan):
            let weight: CapacityClasses.WeightFormat = plan.block == nil ? .int8 : .q8
            if aneTook { return (weight, (aneAlone ? "ane/" : "gpu+ane/") + weight.rawValue) }
            return (weight, (plan.amxColumns > 0 && gemm.conductor != nil ? "gpu+amx/" : "gpu/") + weight.rawValue)
        case .widened:
            return (format(of: name), gemm.lastEngines)
        }
    }

    /// The published weight's format: an 8-bit one's (`int8`, or `q8` for blocks of scales), `q4k` for a GGUF-packed one (they
    /// widen), `bf16` for the rest.
    private func format(of name: String) -> CapacityClasses.WeightFormat {
        guard let tensor = artifact.tensors[name] else { return .bf16 }
        if tensor.dtype == .int8 { return tensor.scale?.block == nil ? .int8 : .q8 }
        return tensor.dtype.isPacked ? .q4k : .bf16
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

    /// **The ANE's rows of an int8 product read as published**: those the plan gave (`planned.ane`, the last ones), if this process
    /// can run them — the ANE built, the weight the program's kind (scales per column or for the tensor, never Q8_0's blocks; no
    /// activation rounded; no AMX columns), and a program that accepts its rotation's `group` — else 0, and the GPU takes the whole
    /// product. The plan's own runnability (`ProductAssignment.isRunnable`) is the hardware's; this is the process's.
    private func aneInt8Rows(_ planned: ProductAssignment, _ plan: Int8Plan, m: Int, k: Int, n: Int) -> Int {
        guard planned.path == .native, planned.ane > 0, plan.block == nil, plan.amxColumns == 0, plan.roundedOffset == 0,
              let ane = gemm.neuralEngine, planned.ane <= m,
              ane.accepts(rows: planned.ane, k: k, n: n, int8Group: plan.group) else { return 0 }
        return planned.ane
    }

    /// **The int8 product cut between the GPU and the ANE**, at the same time: the first `m − aneRows` rows on the GPU's int8
    /// kernel (`int8Product`, the plan's rotation of its own rows in the reserve's tail), the last `aneRows` on the ANE's int8
    /// program, which reads the integers and their scales from the same reserve — nothing widened, nothing copied for it but
    /// the surface. Each row is one engine's. The ANE's rows refused or non-finite: the whole product again on the GPU alone.
    /// `gpu`: the GPU's seconds (its kernel's, and the whole product's if it had to be redone); `gpuWall`, `aneWall`: each side's
    /// wall from the start, as `GEMM.lastSides`; `computed`: the ANE's rows were its own.
    package static func int8Product(gemm: GEMM, plan: Int8Plan, reserve w: MTLBuffer, a: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                                    ane: NeuralEngine, aneRows: Int) -> (gpu: Double, computed: Bool, gpuWall: Double, aneWall: Double) {
        let first = m - aneRows
        nonisolated(unsafe) let (pa, pc) = (a.contents().assumingMemoryBound(to: Float.self), c.contents().assumingMemoryBound(to: Float.self))
        let raw = w.contents()
        let weight = NeuralEngine.Int8Weight(values: raw.assumingMemoryBound(to: Int8.self),
                                             scales: (raw + plan.scalesOffset).assumingMemoryBound(to: Float.self),
                                             perColumn: plan.perColumn, group: plan.group)
        nonisolated(unsafe) var computed = false
        nonisolated(unsafe) var aneWall = 0.0
        let began = DispatchTime.now().uptimeNanoseconds
        let group = DispatchGroup()
        DispatchQueue.global(qos: .userInitiated).async(group: group) {
            computed = ane.product(a: pa, int8: weight, c: pc, first: first, rows: aneRows, k: k, n: n)
            aneWall = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
        }
        var gpu = first > 0 ? int8Product(gemm: gemm, plan: plan, reserve: w, a: a, c: c, m: first, k: k, n: n).gpu : 0
        let gpuWall = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
        group.wait()
        GEMM.neuralEngineTally.withLock { $0.handed += 1; if !computed { $0.givenBack += 1 } }
        if !computed { gpu += int8Product(gemm: gemm, plan: plan, reserve: w, a: a, c: c, m: m, k: k, n: n).gpu }
        return (gpu, computed, gpuWall, aneWall)
    }

    /// The weight into the reserve, by the path the plan assigned, and the assignment the widened product then runs under. A path
    /// this process cannot run (no `TensorGEMM`, no Q8_0 reading, no rounding kernels, a reserve too small for the rounding) is
    /// skipped, not refused: the plain form of the same weight, down to the widened one on the GPU.
    private func prepare(_ name: String, count: Int, m: Int, n: Int) throws -> (Path, ProductAssignment) {
        func expect(_ got: Int) throws {
            guard got == count else { throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected") }
        }
        let k = count / n
        let planned = gemm.assignment(k: k, n: n, m: m, weight: format(of: name))
        // How this product reads its weight (the plan's): the bf16 weight as published, its operand rounded to bf16 where the
        // plan says it gains, the 8-bit weight as published, or widened.
        if planned.path == .bf16, let tensor = artifact.tensors[name], tensor.dtype == .bfloat16, gemm.tensorBF16 || (planned.roundsActivation && gemm.bf16Activations) {
            var rounding: Rounding?
            if planned.roundsActivation && roundsActivations && gemm.bf16Activations {
                rounding = Self.roundingPlan(k: k, n: n, capacity: capacity)
            }
            // The rounding does not fit the reserve, or this caller does not round: the plain bf16 weight, if the process reads it.
            if (rounding != nil || gemm.tensorBF16),
               let got = try timed("staging", { try artifact.stage(name, into: UnsafeMutableRawPointer(reserve), capacity: 2 * capacity) }) {
                try expect(got)
                return (.bf16(rounding), planned)
            }
        }
        if planned.path == .native, transposed, let tensor = artifact.tensors[name], tensor.dtype == .int8,
           let scale = tensor.scale, tensor.shape == [k, n],
           scale.block == nil ? scale.shape.count <= 1 : scale.shape.count == 2,
           gemm.tensor != nil, scale.block == nil ? gemm.tensorInt8 : gemm.tensorQ8 {
            let (perColumn, group) = (scale.shape.count == 1, tensor.rotation?.group)
            let roundsHere = planned.roundsActivation && roundsActivations && gemm.bf16Activations
                && (scale.block == nil || gemm.tensor?.roundsBlocks == true)
            // The AMX's columns the plan gave: widened alone beside the GPU's. Rounded, the GPU alone: the AMX reads the fp32
            // activation, one rounding for some columns only.
            let amxColumns = gemm.conductor == nil ? 0 : planned.amxColumns
            guard let plan = (roundsHere ? Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: perColumn, group: group,
                                                         amxColumns: 0, block: scale.block, rounded: true) : nil)
                    ?? Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: perColumn, group: group,
                                     amxColumns: roundsHere ? 0 : amxColumns, block: scale.block)
                    ?? Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: perColumn, group: group, amxColumns: 0,
                                     block: scale.block)
            else { return (try widen(name, count: count), planned.path == .widened ? planned : .gpuAlone(m)) }
            try expect(tensor.count)
            try timed("staging") { try artifact.stageInt8(name, into: UnsafeMutableRawPointer(reserve),
                                                          scalesAt: plan.scalesOffset) }
            return (.int8(plan), planned)
        }
        return (try widen(name, count: count), planned.path == .widened ? planned : .gpuAlone(m))
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
