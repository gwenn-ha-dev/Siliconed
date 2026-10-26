import Accelerate
import Metal
import MetalPerformanceShaders
import Synchronization

/// The engine's GEMM: `MPSMatrixMultiplication` in fp32.
///
/// fp32 and not fp16, and the decision rests on three independent measurements:
/// the AMX has no other path, MPS is both faster *and* exact there, and the DiT
/// activations overflow fp16 by a factor of thirteen. This is not a compromise.
///
/// The buffers are wrapped with `bytesNoCopy` around the arena: MPS reads where the engine
/// writes, without a copy. That is what unified memory offers and what most engines pay for
/// anyway.
package final class GEMM {
    package let device: MTLDevice
    package let queue: MTLCommandQueue

    /// The arena's slices never move: neither do their `MTLBuffer`s.
    ///
    /// `makeBuffer(bytesNoCopy:)` is not a free wrapper — the kernel has to wire the pages into the
    /// GPU's address space. Doing it at every GEMM, i.e. 224 times per evaluation on regions up to
    /// 157 MB, costs more than the computation. Memoized by address.
    private var buffers: [UInt: MTLBuffer] = [:]

    /// And the operators do not move either: there are only five GEMM shapes in a block.
    ///
    /// Building one `MPSMatrixMultiplication` per call — 224 times per evaluation — dropped the
    /// GEMMs from 3.35 to 1.46 TFLOP/s. The object carries the kernel selection; redoing it every
    /// time means redoing the selection every time. Same fault as for the buffers, and it only shows
    /// up with instrumentation: the arithmetic count gave 14 s, the measurement 32.
    private struct Shape: Hashable { let m, k, n: Int; let transposed: Bool; var accumulates = false }
    private var operators: [Shape: MPSMatrixMultiplication] = [:]

    /// Count and GPU time per shape. The gap between 3.34 TFLOP/s on the bench and 1.49 in the
    /// engine was chased by hypotheses — alignment of M, `bytesNoCopy`, interleaved CPU writes,
    /// rebuilding the operator — and none held. Counting is less clever and safer.
    ///
    /// **`best` is there for a hypothesis this list did not contain: the asymmetry of the
    /// measurement itself.** The bench (`m4_gemm.swift`, since archived) timed the same way — `gpuEndTime −
    /// gpuStartTime`, one GEMM per buffer — but it kept the **best of five**, on the same
    /// buffers left warm. The engine sums *all* its executions, cold operands and clock dropped
    /// between two bursts: it is an average. Comparing a best to an average manufactures a gap
    /// that is not a defect. The engine's best against the bench's best settles it; and the gap
    /// between the engine's best and its average quantifies what cold starts and variance cost.
    package private(set) var census: [String: (count: Int, seconds: Double, best: Double)] = [:]
    private func record(_ key: String, _ seconds: Double) {
        let previous = census[key] ?? (0, 0, Double.infinity)
        census[key] = (previous.count + 1, previous.seconds + seconds, min(previous.best, seconds))
    }
    /// Traces the first `n` calls: a gap that is constant from the first one does not have the same
    /// cause as a gap that sets in gradually.
    package var trace = 0
    private var traced = 0
    private var descriptors: [Shape: (MPSMatrixDescriptor, MPSMatrixDescriptor, MPSMatrixDescriptor)] = [:]

    /// **The `m` values MPS must never see: 8 to 32 rows.** A single MPS GEMM of that height in the
    /// process — any one, on any queue — slows down **all** the MPS GEMMs that follow it, until the
    /// end of the process: 3.56 → 2.87 TFLOP/s on Anima's `ff` shape (measured
    /// one process per `m`). Below 8, MPS picks another kernel, which poisons nothing but does not
    /// round like the one for large heights (it returns other bits); above 32, nothing. It was
    /// first found in Krea 2's text fusion; it was also in the text encoder (m = prompt tokens) and
    /// in Z-Image's context refiner (m = 32).
    package static let poisonedRows = 8...32
    /// What gets submitted instead. An output row depends only on its input row, and heights 8 to 64
    /// take the same kernel: **same bits** (row for row).
    package static let submittedRows = 64

    /// **The padding tiers: where a GEMM of 8 to 32 rows is done at 64.** The caller does not have
    /// to size its slices for 64 rows: `A` is copied there (at most 32 × `k`, a few hundred
    /// KB), the product is written there, and its first `m` rows come back into `C`. Three buffers
    /// per `GEMM`, grown as needed — ~2.6 MB each at the widest (`k` = 10,240).
    private final class Tiers {
        let device: MTLDevice
        private var buffers: [String: MTLBuffer] = [:]
        init(device: MTLDevice) { self.device = device }
        private func buffer(_ name: String, _ floats: Int) -> MTLBuffer {
            if let t = buffers[name], t.length >= floats * 4 { return t }
            let t = device.makeBuffer(length: floats * 4, options: .storageModeShared)!
            t.label = "padding tier \(name)"
            buffers[name] = t
            return t
        }
        func a(_ floats: Int) -> MTLBuffer { buffer("a", floats) }
        func c(_ floats: Int) -> MTLBuffer { buffer("c", floats) }
        func mid(_ floats: Int) -> MTLBuffer { buffer("mid", floats) }
    }
    private lazy var tiers = Tiers(device: device)

    /// Whether `m` rows go through the padding tiers: 8 to 32 (`poisonedRows`), or a tail under 64 rows
    /// past a body (`rowSplit`) — what `linear` and `lora` do.
    package static func goesThroughTiers(_ m: Int) -> Bool { poisonedRows.contains(m) || rowSplit(m).tail > 0 }

    /// **The tiers' bytes over these products** — what `Tiers` grows to: `a` at `submittedRows · k`, `c`
    /// at `submittedRows · n` for the widest `k` and `n` among the products that go through them, and,
    /// under a LoRA of cumulative rank `rank`, `mid` at `submittedRows · rank`; each an `MTLBuffer`, a
    /// whole number of pages. 0 when no product goes through. An upper bound: the conductor's GPU slices
    /// are multiples of 256 rows and never go through, but whether it takes a product is decided at run.
    ///
    /// `bounded`: products whose height is only an upper bound (it follows the prompt's tokens, which
    /// the need bounds by its bytes) — any height from 8 to it may be the real one, and one of them goes
    /// through: charged whenever the bound reaches 8. A height that becomes a multiple of 64 would
    /// otherwise make the need *fall* as the prompt grows (by 6 MB at Qwen-Image-2.1's MLP width).
    package static func tierBytes(products: [(m: Int, k: Int, n: Int)], bounded: [(m: Int, k: Int, n: Int)] = [],
                                  rank: Int = 0) -> Int {
        let tiered = products.filter { goesThroughTiers($0.m) } + bounded.filter { $0.m >= poisonedRows.lowerBound }
        guard let k = tiered.map(\.k).max(), let n = tiered.map(\.n).max() else { return 0 }
        let rows = submittedRows
        return Arena.capacity(for: rows * k * 4) + Arena.capacity(for: rows * n * 4)
            + (rank > 0 ? Arena.capacity(for: rows * rank * 4) : 0)
    }

    /// **A GPU submission's height is a multiple of 64, or below 64 — never in between.**
    ///
    /// MPS's fast fp32 kernel (the one the padding tiers leave in place) writes **canonical NaN**
    /// into the even columns of every row from some row on, when the height is not a multiple of
    /// 64 — on operands that are finite, and with no error reported. Seen in Z-Image's DiT without
    /// the AMX (`w1`, 4,128 rows; `w3` of the first unified block at the 512²
    /// spectral evaluation, 1,056 rows, rows 832 to 1,055). On the faulty output buffer, every
    /// multiple of 64 from 64 to 1,024 is right, and 833, 900, 1,000, 1,025, 1,055, 1,056 are
    /// wrong from row 832; the same memory wrapped in a fresh `MTLBuffer`, or a fresh buffer, is
    /// right at 1,056. It is **not** a bound on the output size (3,277 × 10,240 = 2²⁵ was the
    /// hypothesis: 0.32 · 2²⁵ already fails), and it does not reproduce outside the engine —
    /// not with `bytesNoCopy`, untouched pages, the block's arena layout, nor after the slow
    /// kernel. What triggers it is not isolated; what avoids it is measured. The AMX's cut never
    /// saw it because it hands the GPU multiples of `amx_quantum` (256).
    ///
    /// So the rows split into a body, a multiple of 64 submitted as it is, and a tail of fewer
    /// than 64 rows that goes through the padding tier at 64. **Same bits**: an output row depends
    /// only on its input row, and the kernel rounds the same way at every height from 8 up
    /// (measured from 8 to 64, and for 1,056 to 4,128, halves and body + tier, five shapes).
    /// Below 64 rows, nothing changes (8 to 32 already go through the tier, 1 to 7 and 33 to 63
    /// keep their submission).
    package static let rowQuantum = 64
    /// `(body, tail)`: `body` rows submitted as they are, a multiple of `rowQuantum`, then `tail`
    /// rows through the tier, `0 ≤ tail < rowQuantum`. A height below the quantum, or a multiple of
    /// it, is one submission: `(m, 0)`.
    package static func rowSplit(_ m: Int) -> (body: Int, tail: Int) {
        let body = m / rowQuantum * rowQuantum
        return body == 0 ? (m, 0) : (body, m - body)
    }

    /// The conductor of GPU + AMX co-execution, if requested.
    package private(set) var conductor: Conductor?

    /// **The product on bf16 weights** (`gemm_tensor`, `TensorGEMM`), if requested and if this macOS
    /// compiles it; `nil` otherwise, and every product widens as before. A caller asks
    /// `Artifact.stage` for the bf16 weight and, given one, calls `linear(a:bf16:…)`.
    package private(set) var tensor: TensorGEMM?
    /// The Neural Engine on the last rows of a widened product (`gemm_ane`); `nil` when off.
    package private(set) var neuralEngine: NeuralEngine?
    /// The last product cut with the ANE: each side's wall seconds and rows, from which the map reads the
    /// share that would have ended both together.
    package private(set) var lastSides: (gpu: Double, gpuRows: Int, ane: Double, aneRows: Int)?
    /// Every product of this process the ANE was handed, and how many of them it gave back (refused or
    /// non-finite, recomputed on the GPU alone): a check run on the ANE says what the ANE actually computed.
    package static let neuralEngineTally = Mutex<(handed: Int, givenBack: Int)>((0, 0))
    /// What `tensor` takes: bf16 weights (`gemm_tensor`), published int8 ones (`gemm_int8`), GGUF Q8_0
    /// ones (`gemm_q8`).
    package let tensorBF16, tensorInt8, tensorQ8: Bool
    /// The products may take their activation rounded to bf16 (`gemm_bf16_act`, `BFloatKernels`): the
    /// library compiled. Which products do is `WeightProducts`'s to say.
    package let bf16Activations: Bool
    /// Products that went through the bf16 activation path in this process: a check says what ran.
    package static let roundedProducts = Mutex<Int>(0)

    /// - Parameter freezeCut: the GPU/AMX cut is not servo-controlled. This is the reproducible mode:
    ///   `T` stays the profile's, so `cut` is a pure function and two renders of the same
    ///   seed return the same bits. No effect when co-execution is off.
    /// - Parameter coexecution: the GPU/AMX conductor, `Conductor.enabled` by default. The diagnostic
    ///   asks for both on any chip: the GPU alone and the cut, measured on the same product.
    /// - Parameter tensorWeights: the bf16-weight product (`gemm_tensor`); the diagnostic asks for it on
    ///   any chip that has it, to measure it. `int8Weights`: the published-int8 one (`gemm_int8`);
    ///   `q8Weights`: the Q8_0 one (`gemm_q8`).
    package init(freezeCut: Bool = EngineSettings.effective.frozenCut, coexecution: Bool = Conductor.enabled,
                 tensorWeights: Bool = EngineSettings.effective.gemmTensor,
                 int8Weights: Bool = EngineSettings.effective.gemmInt8,
                 q8Weights: Bool = EngineSettings.effective.gemmQ8,
                 bf16Activations: Bool = EngineSettings.effective.gemmBF16Activations,
                 neuralEngine: Bool = EngineSettings.effective.gemmANE,
                 neuralEngineShare: Double = EngineSettings.effective.aneFraction) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw Failure.noDevice
        }
        self.device = device
        self.queue = queue
        if coexecution {
            conductor = Conductor(queue: queue, driven: !freezeCut)
        }
        if tensorWeights || int8Weights || q8Weights || bf16Activations {
            tensor = TensorGEMM(device: device, queue: queue, roundingActivations: bf16Activations)
        }
        self.bf16Activations = tensor?.roundsActivations == true
        tensorBF16 = tensor != nil && tensorWeights
        tensorInt8 = tensor != nil && int8Weights
        tensorQ8 = tensor?.readsBlocks == true && q8Weights
        if neuralEngine {
            self.neuralEngine = NeuralEngine(gpuShare: neuralEngineShare, shares: EngineSettings.effective.aneShares)
            self.neuralEngine?.besideAMX = EngineSettings.effective.aneAMX
        }
    }

    package enum Failure: Error, CustomStringConvertible {
        case noDevice
        case unalignedBuffer(String)
        package var description: String {
            switch self {
            case .noDevice: return "no Metal device"
            case .unalignedBuffer(let what):
                return "\(what): `bytesNoCopy` requires a page-aligned pointer and length"
            }
        }
    }

    /// Wraps an arena region without copying it. The page is the arena's, so
    /// alignment is a given — we check it anyway, because the silent failure of
    /// `bytesNoCopy` is a `nil` that would be mistaken for an allocation failure.
    package func wrap(_ pointer: UnsafeMutableRawPointer, bytes: Int, name: String) throws -> MTLBuffer {
        let page = Arena.alignment
        let rounded = (bytes + page - 1) / page * page
        let key = UInt(bitPattern: pointer)
        if let existing = buffers[key], existing.length >= rounded { return existing }
        guard Int(bitPattern: pointer) % page == 0 else { throw Failure.unalignedBuffer(name) }
        guard let buffer = device.makeBuffer(bytesNoCopy: pointer, length: rounded,
                                             options: .storageModeShared, deallocator: nil) else {
            throw Failure.unalignedBuffer(name)
        }
        buffer.label = name
        buffers[key] = buffer
        return buffer
    }

    /// `C = A × Bᵀ`, the shape an `nn.Linear` really has: the weights are stored `[output,
    /// input]`, so it is the right side that gets transposed, and MPS does it without materializing
    /// the transpose.
    ///
    /// - Parameters:
    ///   - a: `[m, k]` fp32, row-major
    ///   - b: `[n, k]` fp32, row-major — the weights as the model publishes them
    ///   - c: `[m, n]` fp32, written
    /// - Parameter weightIsTransposed: the weight is stored `[input, output]` by the forge, so there
    ///   is nothing to transpose. `transposeRight` costs **2.26×** on `w2`'s `K=10240` shape
    ///   (1.578 against 3.564 TFLOP/s) and −23.6 % on a whole block.
    package func linear(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                       weightIsTransposed: Bool = false) -> Double {
        // The Neural Engine on the last rows, the GPU on the first — alone, or beside the AMX on the shapes the
        // map gave all three (`ane_amx`); the AMX alone beside the GPU on the shapes the ANE's grid does not take
        // (`ane_shares`): at the same time, each row one engine's. Its rows refused or non-finite: the whole product again on the GPU — the bits of the GPU
        // alone, never a half-computed output.
        if let ane = neuralEngine, weightIsTransposed, case let rows = ane.rows(m, k: k, n: n), rows > 0, ane.accepts(rows: rows, k: k, n: n) {
            let first = m - rows
            nonisolated(unsafe) let (pa, pb, pc) = (a.contents().assumingMemoryBound(to: Float.self),
                                                    b.contents().assumingMemoryBound(to: Float.self),
                                                    c.contents().assumingMemoryBound(to: Float.self))
            nonisolated(unsafe) var computed = false
            let began = DispatchTime.now().uptimeNanoseconds
            let group = DispatchGroup()
            nonisolated(unsafe) var aneWall = 0.0
            DispatchQueue.global(qos: .userInitiated).async(group: group) {
                computed = ane.product(a: pa, w: pb, c: pc, first: first, rows: rows, k: k, n: n)
                aneWall = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
            }
            // The GPU's side alone, or beside the AMX where the map gave the shape all three (`ane_amx`).
            _ = gpuLinear(a: a, b: b, c: c, m: first, k: k, n: n, weightIsTransposed: weightIsTransposed,
                          conductor: ane.takesAMX(m, k: k, n: n))
            let gpuWall = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
            group.wait()
            lastSides = (gpuWall, first, aneWall, rows)
            Self.neuralEngineTally.withLock { $0.handed += 1; if !computed { $0.givenBack += 1 } }
            let key = "\(m)×\(k)×\(n)ᵀ ane \(rows)"
            record(key + " GPU side", gpuWall)
            if !computed { _ = gpuLinear(a: a, b: b, c: c, m: m, k: k, n: n, weightIsTransposed: weightIsTransposed, conductor: false) }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
            record(key, wall)
            return wall
        }
        return gpuLinear(a: a, b: b, c: c, m: m, k: k, n: n, weightIsTransposed: weightIsTransposed)
    }

    /// `linear` on the GPU (and the AMX's cut, unless `conductor` is false): what every product ran before
    /// the Neural Engine.
    private func gpuLinear(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                           weightIsTransposed: Bool, conductor allowed: Bool = true) -> Double {
        // From 8 to 32 rows, the GEMM goes through the padding tiers (see `poisonedRows`). The
        // GPU/AMX cut never applies there (`amx_min` = 512).
        if GEMM.poisonedRows.contains(m) {
            return padded(a: a, b: b, c: c, first: 0, rows: m, k: k, n: n,
                          weightIsTransposed: weightIsTransposed)
        }
        // The per-token cut, if the conductor takes it. It returns `nil` when it does not
        // apply — too few rows — and the original path resumes, unchanged.
        if allowed, let conductor, let wall = conductor.linear(a: a, b: b, c: c, m: m, k: k, n: n,
                                                      weightIsTransposed: weightIsTransposed,
                                                      multiply: { rows in
            submitRows(a: a, b: b, c: c, m: rows, k: k, n: n,
                       weightIsTransposed: weightIsTransposed)
        }) {
            // Under a separate key: **wall time, not GPU time**. Mixing them would make a
            // census that compares neither with itself nor with previous sessions.
            let key = "\(m)×\(k)×\(n)\(weightIsTransposed ? "ᵀ" : "")‖"
            record(key, wall)
            return wall
        }
        return submitRows(a: a, b: b, c: c, m: m, k: k, n: n,
                          weightIsTransposed: weightIsTransposed)
    }

    /// **`C = A · W` with `W` in bf16** (`tensor`, which the caller checked): `wb` holds the weight as
    /// `Artifact.stage` copied it, `[k, n]` when `weightIsTransposed`, `[n, k]` otherwise. Whole, any
    /// `m`: no padding tier, no AMX cut (`TensorGEMM`).
    package func linear(a: MTLBuffer, bf16 wb: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                        weightIsTransposed: Bool = false) -> Double {
        let elapsed = tensor!.linear(a: a, w: wb, c: c, m: m, k: k, n: n, weightIsTransposed: weightIsTransposed)
        let key = "\(m)×\(k)×\(n)\(weightIsTransposed ? "ᵀ" : "")⊗"
        record(key, elapsed)
        return elapsed
    }

    /// **The same on an activation rounded to bf16** (`bf16Activations`, which the caller checked): `A`
    /// rounded by chunks of `rounding.rows` rows into the reserve's tail, `wb` the staged weight.
    package func linear(a: MTLBuffer, rounding: TensorGEMM.Rounding, bf16 wb: MTLBuffer, c: MTLBuffer, m: Int, k: Int,
                        n: Int, weightIsTransposed: Bool = false) -> Double {
        let elapsed = tensor!.linear(a: a, rounding: rounding, w: wb, c: c, m: m, k: k, n: n,
                                     weightIsTransposed: weightIsTransposed)
        let key = "\(m)×\(k)×\(n)\(weightIsTransposed ? "ᵀ" : "")◐"
        record(key, elapsed)
        Self.roundedProducts.withLock { $0 += 1 }
        return elapsed
    }

    /// `m` rows on the GPU at heights MPS computes right (`rowQuantum`): the body as it is, the
    /// tail through the tier. Two submissions only when the height requires it.
    private func submitRows(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                            weightIsTransposed: Bool) -> Double {
        let (body, tail) = GEMM.rowSplit(m)
        let time = submit(a: a, b: b, c: c, m: body, k: k, n: n,
                          weightIsTransposed: weightIsTransposed)
        guard tail > 0 else { return time }
        return time + padded(a: a, b: b, c: c, first: body, rows: tail, k: k, n: n,
                             weightIsTransposed: weightIsTransposed)
    }

    /// Rows `first ..< first + rows` (fewer than 64) computed at 64 in the padding tier: `A`'s rows
    /// copied there, the product written there, its first `rows` rows copied back into `C`.
    private func padded(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, first: Int, rows: Int, k: Int, n: Int,
                        weightIsTransposed: Bool) -> Double {
        let l = GEMM.submittedRows
        let (pa, pc) = (tiers.a(l * k), tiers.c(l * n))
        pa.contents().copyMemory(from: a.contents() + first * k * 4, byteCount: rows * k * 4)
        let time = submit(a: pa, b: b, c: pc, m: l, k: k, n: n,
                          weightIsTransposed: weightIsTransposed)
        (c.contents() + first * n * 4).copyMemory(from: pc.contents(), byteCount: rows * n * 4)
        return time
    }

    /// **A weight's LoRA stack: `C += (x·down)·up`, in ONE submission.**
    ///
    /// The two skinny GEMMs are encoded in the **same** command buffer, and Metal guarantees
    /// ordering between two encoders of the same buffer. A measurement showed that a LoRA's overhead has a **floor per
    /// submission**, constant with the rank; making two of them instead of one would double that floor
    /// for nothing.
    ///
    /// **Outside the conductor, and that is deliberate.** The GPU/AMX cut hands rows to
    /// `BNNSMatMul` while the GPU does the others; it has no way to accumulate onto an
    /// existing output — it **writes** its slice. Routing this through it would leave `C` partly overwritten
    /// instead of accumulated, hence a plausible and wrong image. And there would be nothing to gain: at
    /// `r = 32`, `x·down` is a skinny GEMM limited by latency, not by compute.
    ///
    /// - Parameters:
    ///   - mid: `[m, r]`, the work slice — written then read back in the same buffer.
    ///   - c: `[m, n]`, the main GEMM's output, **accumulated** and not replaced.
    @discardableResult
    package func lora(x: MTLBuffer, down: MTLBuffer, mid: MTLBuffer, up: MTLBuffer, c: MTLBuffer,
                     m: Int, k: Int, r: Int, n: Int) -> Double {
        func padded(first: Int, rows: Int) -> Double {
            let l = GEMM.submittedRows
            let (px, pc, pm) = (tiers.a(l * k), tiers.c(l * n), tiers.mid(l * r))
            px.contents().copyMemory(from: x.contents() + first * k * 4, byteCount: rows * k * 4)
            // Accumulated, so the output goes into the tier before and comes back out after.
            pc.contents().copyMemory(from: c.contents() + first * n * 4, byteCount: rows * n * 4)
            let time = lora(x: px, down: down, mid: pm, up: up, c: pc, m: l, k: k, r: r, n: n)
            (c.contents() + first * n * 4).copyMemory(from: pc.contents(), byteCount: rows * n * 4)
            return time
        }
        if GEMM.poisonedRows.contains(m) { return padded(first: 0, rows: m) }
        // Heights that are not a multiple of 64: the body here, the tail through the tier
        // (`rowQuantum`) — the body's prefixes of `x`, `mid` and `c` are its own rows.
        let (body, tail) = GEMM.rowSplit(m)
        if tail > 0 {
            return lora(x: x, down: down, mid: mid, up: up, c: c, m: body, k: k, r: r, n: n)
                + padded(first: body, rows: tail)
        }
        let commands = queue.makeCommandBuffer()!
        encoder(Shape(m: m, k: k, n: r, transposed: true), into: commands, x, down, mid)
        encoder(Shape(m: m, k: r, n: n, transposed: true, accumulates: true), into: commands, mid, up, c)
        commands.commit()
        commands.waitUntilCompleted()
        let elapsed = Attention.verdict(commands, what: "LoRA \(m)×\(k)×\(r)→\(n)")
        let key = "lora \(m)×\(k)×\(r)→\(n)"
        record(key, elapsed)
        return elapsed
    }

    /// **`C += A · W`** — a GEMM that accumulates instead of writing, weight stored `[input, output]`.
    ///
    /// Qwen-Image-2.1's merged LoRA, `W += down · up` (`m` = the weight's `k`). Outside the conductor, like `lora` and for the
    /// same reason: the GPU/AMX cut writes its slice, it does not accumulate.
    ///
    /// Its rows split like `linear`'s and `lora`'s (`rowSplit`): a body, a multiple of 64, then a tail
    /// through the padding tier, accumulated there — MPS's fast kernel returns NaN at the heights in
    /// between. Qwen-Image-2.1's weights have `k` = 4 096 or 12 288, multiples of 64: one submission,
    /// the tier never grown.
    @discardableResult
    package func accumulatedLinear(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int) -> Double {
        func padded(first: Int, rows: Int) -> Double {
            let l = GEMM.submittedRows
            let (pa, pc) = (tiers.a(l * k), tiers.c(l * n))
            pa.contents().copyMemory(from: a.contents() + first * k * 4, byteCount: rows * k * 4)
            // Accumulated, so the output goes into the tier before and comes back out after.
            pc.contents().copyMemory(from: c.contents() + first * n * 4, byteCount: rows * n * 4)
            let time = accumulatedLinear(a: pa, b: b, c: pc, m: l, k: k, n: n)
            (c.contents() + first * n * 4).copyMemory(from: pc.contents(), byteCount: rows * n * 4)
            return time
        }
        if GEMM.poisonedRows.contains(m) { return padded(first: 0, rows: m) }
        let (body, tail) = GEMM.rowSplit(m)
        if tail > 0 { return accumulatedLinear(a: a, b: b, c: c, m: body, k: k, n: n) + padded(first: body, rows: tail) }
        let commands = queue.makeCommandBuffer()!
        encoder(Shape(m: m, k: k, n: n, transposed: true, accumulates: true), into: commands, a, b, c)
        commands.commit()
        commands.waitUntilCompleted()
        let elapsed = Attention.verdict(commands, what: "accumulated GEMM \(m)×\(k)×\(n)")
        let key = "\(m)×\(k)×\(n)ᵀ+"
        record(key, elapsed)
        return elapsed
    }

    /// A memoized encoding, without submission — the common body of `submit` and `lora`.
    ///
    /// `transposed: true` means "the weight is already stored `[input, output]`", so nothing to
    /// transpose: it is the forge's layout, and the one `LoRA` produces.
    private func encoder(_ shape: Shape, into commands: MTLCommandBuffer,
                         _ a: MTLBuffer, _ b: MTLBuffer, _ c: MTLBuffer) {
        let multiply: MPSMatrixMultiplication
        if let existing = operators[shape] {
            multiply = existing
        } else {
            let fresh = MPSMatrixMultiplication(device: device, transposeLeft: false,
                                               transposeRight: !shape.transposed,
                                               resultRows: shape.m, resultColumns: shape.n,
                                               interiorColumns: shape.k,
                                               alpha: 1.0, beta: shape.accumulates ? 1.0 : 0.0)
            operators[shape] = fresh
            multiply = fresh
        }
        let size = MemoryLayout<Float>.size
        let da = MPSMatrixDescriptor(rows: shape.m, columns: shape.k, rowBytes: shape.k * size, dataType: .float32)
        let db = shape.transposed
            ? MPSMatrixDescriptor(rows: shape.k, columns: shape.n, rowBytes: shape.n * size, dataType: .float32)
            : MPSMatrixDescriptor(rows: shape.n, columns: shape.k, rowBytes: shape.k * size, dataType: .float32)
        let dc = MPSMatrixDescriptor(rows: shape.m, columns: shape.n, rowBytes: shape.n * size, dataType: .float32)
        multiply.encode(commandBuffer: commands,
                        leftMatrix: MPSMatrix(buffer: a, descriptor: da),
                        rightMatrix: MPSMatrix(buffer: b, descriptor: db),
                        resultMatrix: MPSMatrix(buffer: c, descriptor: dc))
    }

    /// An MPS submission and its wait. It is `linear`'s original body, extracted so that the
    /// conductor can call it again on a **slice** of rows.
    private func submit(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                        weightIsTransposed: Bool) -> Double {
        var output = c
        var elapsed = submitOnce(a: a, b: b, c: output, m: m, k: k, n: n, weightIsTransposed: weightIsTransposed)
        // The first product of each (output buffer, shape) is checked; one that MPS got wrong gets a
        // fresh buffer over the same memory, at most twice, the others are trusted from then on.
        for _ in 0..<2 where !verified(output, m: m, n: n, a: a, k: k) {
            Warnings.emit("⚠ GEMM \(m)×\(k)×\(n): MPS wrote non-finite values from finite operands; recomputed in a fresh buffer")
            output = rewrapped(output)
            elapsed += submitOnce(a: a, b: b, c: output, m: m, k: k, n: n, weightIsTransposed: weightIsTransposed)
        }
        return elapsed
    }

    /// **Pairs (output buffer, shape) already checked, the buffer kept alive** so that its identifier
    /// is never another's. MPS's fast fp32 kernel writes canonical NaN into the even columns of a band
    /// of rows **for some output `MTLBuffer`s and not others**: the same memory wrapped in a fresh
    /// buffer is right, the same buffer is wrong again on every retry, layer after layer, step after
    /// step. It was first seen at 1 056 rows and the height taken for the cause (`rowSplit`); it came back at
    /// 4 160, a multiple of 64 — Z-Image 1024² under a LoRA, a 64-token caption, rows 384 to 511 of
    /// `w3`. What triggers it was not isolated (not the height alone, not a 2 GiB boundary of the
    /// GPU address, not untouched pages); what it depends on was measured, and a pair is therefore
    /// judged once: one sum over the output, a few ms, about twenty times a render.
    private var checked: [ObjectIdentifier: (buffer: MTLBuffer, shapes: Set<Int>)] = [:]

    /// Whether `c`'s first `m × n` floats are finite, or `a`'s are not (a non-finite input is the
    /// model's, not the kernel's) — `true` without looking for a pair already checked.
    private func verified(_ c: MTLBuffer, m: Int, n: Int, a: MTLBuffer, k: Int) -> Bool {
        let id = ObjectIdentifier(c), shape = m &* 1_000_003 &+ n
        if checked[id]?.shapes.contains(shape) == true { return true }
        func finite(_ buffer: MTLBuffer, _ count: Int) -> Bool {
            var sum: Float = 0
            vDSP_sve(buffer.contents().assumingMemoryBound(to: Float.self), 1, &sum, vDSP_Length(count))
            return sum.isFinite
        }
        let right = finite(c, m * n) || !finite(a, m * k)
        if right { checked[id, default: (c, [])].shapes.insert(shape) }
        return right
    }

    /// A fresh `MTLBuffer` over `c`'s memory, which `wrap` hands out from then on in its place.
    private func rewrapped(_ c: MTLBuffer) -> MTLBuffer {
        guard let fresh = device.makeBuffer(bytesNoCopy: c.contents(), length: c.length,
                                            options: .storageModeShared, deallocator: nil) else { return c }
        fresh.label = c.label
        let key = UInt(bitPattern: c.contents())
        if buffers[key] === c { buffers[key] = fresh }
        return fresh
    }

    /// One MPS submission and its wait.
    private func submitOnce(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                            weightIsTransposed: Bool) -> Double {
        let shape = Shape(m: m, k: k, n: n, transposed: weightIsTransposed)
        let multiply: MPSMatrixMultiplication
        if let existing = operators[shape] {
            multiply = existing
        } else {
            multiply = MPSMatrixMultiplication(device: device, transposeLeft: false,
                                               transposeRight: !weightIsTransposed,
                                               resultRows: m, resultColumns: n, interiorColumns: k,
                                               alpha: 1.0, beta: 0.0)
            operators[shape] = multiply
        }
        let shapes: (MPSMatrixDescriptor, MPSMatrixDescriptor, MPSMatrixDescriptor)
        if let existing = descriptors[shape] {
            shapes = existing
        } else {
            let size = MemoryLayout<Float>.size
            shapes = (MPSMatrixDescriptor(rows: m, columns: k, rowBytes: k * size, dataType: .float32),
                      weightIsTransposed
                        ? MPSMatrixDescriptor(rows: k, columns: n, rowBytes: n * size, dataType: .float32)
                        : MPSMatrixDescriptor(rows: n, columns: k, rowBytes: k * size, dataType: .float32),
                      MPSMatrixDescriptor(rows: m, columns: n, rowBytes: n * size, dataType: .float32))
            descriptors[shape] = shapes
        }
        let commands = queue.makeCommandBuffer()!
        multiply.encode(commandBuffer: commands,
                        leftMatrix: MPSMatrix(buffer: a, descriptor: shapes.0),
                        rightMatrix: MPSMatrix(buffer: b, descriptor: shapes.1),
                        resultMatrix: MPSMatrix(buffer: c, descriptor: shapes.2))
        commands.commit()
        commands.waitUntilCompleted()
        // Lesson 12, applied here as in `Attention`: a submission that fails leaves `C`
        // as it was and returns a zero time. Without this check, the engine multiplies what was lying around.
        let elapsed = Attention.verdict(commands, what: "GEMM \(m)×\(k)×\(n)")
        if trace > 0 { traced += 1
            if traced <= trace {
                Warnings.emit(String(format: "    [%3d] %d×%d×%d  %6.1f ms  %5.3f TFLOP/s",
                    traced, m, k, n, elapsed * 1e3,
                    2.0 * Double(m) * Double(k) * Double(n) / elapsed / 1e12))
            }
        }
        let key = "\(m)×\(k)×\(n)\(weightIsTransposed ? "ᵀ" : "")"
        record(key, elapsed)
        return elapsed
    }
}
