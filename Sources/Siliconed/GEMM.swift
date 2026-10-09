import Metal
import MetalPerformanceShaders

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
    /// measurement itself.** The bench (`m4_gemm.swift`) times the same way — `gpuEndTime −
    /// gpuStartTime`, one GEMM per buffer — but it keeps the **best of five**, on the same
    /// buffers left warm. The engine sums *all* its executions, cold operands and clock dropped
    /// between two bursts: it is an average. Comparing a best to an average manufactures a gap
    /// that is not a defect. The engine's best against the bench's best settles it; and the gap
    /// between the engine's best and its average quantifies what cold starts and variance cost.
    package private(set) var census: [String: (count: Int, seconds: Double, best: Double)] = [:]
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

    /// The GPU widener, **shared by the three sets of blocks**: it carries the memoization of the
    /// map's wrappers, and three instances would triple it for nothing — the three blocks
    /// read disjoint tensors (`noise_refiner.*`, `context_refiner.*`, `layers.*`).
    ///
    /// Optional, and silently so: if the kernel does not compile on a machine, the engine
    /// widens on the CPU as before. It is a speed path, not a correctness path — both
    /// return the same bits.
    package private(set) var widener: WidenGPU?

    /// The conductor of GPU + AMX co-execution, if requested **and** compatible.
    ///
    /// An incompatibility is reported and not endured: co-execution refused silently
    /// would give correct numbers but twice too slow, and the cause would be sought in the
    /// GEMM.
    package private(set) var conductor: Conductor?

    /// - Parameter freezeCut: the GPU/AMX cut is not servo-controlled. This is the reproducible mode:
    ///   `T` stays the profile's, so `cut` is a pure function and two renders of the same
    ///   seed return the same bits. No effect when co-execution is off.
    package init(freezeCut: Bool = EngineSettings.effective.frozenCut) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw Failure.noDevice
        }
        self.device = device
        self.queue = queue
        // **The default is the CPU, and a measurement says so.** Widening on the GPU costs
        // **+20 %** in the engine (24.98 → 30.07 s per evaluation) even though the kernel returns
        // 183 GB/s against 15. The CPU widened while the GPU computed; on the GPU,
        // the operation queues behind the GEMM, which is the bottleneck. `SILICONED_WIDEN_GPU=1`
        // turns the path back on — it is bit-exact, and it will become interesting again the day the
        // CPU is busy.
        if EngineSettings.effective.widenGPU {
            widener = try? WidenGPU(device: device, queue: queue)
        }
        if Conductor.enabled {
            do { conductor = try Conductor(queue: queue, fusedWiden: Block.fusedWiden,
                                           driven: !freezeCut) }
            catch { Warnings.emit("⚠ \(error)") }
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
    /// - Parameter before: something to encode a job **in the same command buffer**, right
    ///   before the GEMM. This is how the fused widening goes through: the weight is widened and
    ///   consumed without a second submission, and Metal guarantees ordering and coherence between two
    ///   encoders of the same buffer. The price is that the returned time then covers **both** — the
    ///   GEMM census stops being comparable with previous sessions, which is why this
    ///   is not the default.
    package func linear(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                       weightIsTransposed: Bool = false,
                       before: ((MTLCommandBuffer) -> Void)? = nil) -> Double {
        // From 8 to 32 rows, the GEMM goes through the padding tiers (see `poisonedRows`). The
        // GPU/AMX cut never applies there (`amx_min` = 512).
        if GEMM.poisonedRows.contains(m) {
            return padded(a: a, b: b, c: c, first: 0, rows: m, k: k, n: n,
                          weightIsTransposed: weightIsTransposed, before: before)
        }
        // The per-token cut, if the conductor takes it. It returns `nil` when it does not
        // apply — too few rows — and the original path resumes, unchanged.
        if let conductor, let wall = conductor.linear(a: a, b: b, c: c, m: m, k: k, n: n,
                                                      weightIsTransposed: weightIsTransposed,
                                                      multiply: { rows in
            submitRows(a: a, b: b, c: c, m: rows, k: k, n: n,
                       weightIsTransposed: weightIsTransposed, before: before)
        }) {
            // Under a separate key: **wall time, not GPU time**. Mixing them would make a
            // census that compares neither with itself nor with previous sessions.
            let key = "\(m)×\(k)×\(n)\(weightIsTransposed ? "ᵀ" : "")‖"
            let previous = census[key] ?? (0, 0, Double.infinity)
            census[key] = (previous.count + 1, previous.seconds + wall, min(previous.best, wall))
            return wall
        }
        return submitRows(a: a, b: b, c: c, m: m, k: k, n: n,
                          weightIsTransposed: weightIsTransposed, before: before)
    }

    /// `m` rows on the GPU at heights MPS computes right (`rowQuantum`): the body as it is, the
    /// tail through the tier. Two submissions only when the height requires it.
    private func submitRows(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                            weightIsTransposed: Bool,
                            before: ((MTLCommandBuffer) -> Void)?) -> Double {
        let (body, tail) = GEMM.rowSplit(m)
        let time = submit(a: a, b: b, c: c, m: body, k: k, n: n,
                          weightIsTransposed: weightIsTransposed, before: before)
        guard tail > 0 else { return time }
        return time + padded(a: a, b: b, c: c, first: body, rows: tail, k: k, n: n,
                             weightIsTransposed: weightIsTransposed, before: nil)
    }

    /// Rows `first ..< first + rows` (fewer than 64) computed at 64 in the padding tier: `A`'s rows
    /// copied there, the product written there, its first `rows` rows copied back into `C`.
    private func padded(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, first: Int, rows: Int, k: Int, n: Int,
                        weightIsTransposed: Bool,
                        before: ((MTLCommandBuffer) -> Void)?) -> Double {
        let l = GEMM.submittedRows
        let (pa, pc) = (tiers.a(l * k), tiers.c(l * n))
        pa.contents().copyMemory(from: a.contents() + first * k * 4, byteCount: rows * k * 4)
        let time = submit(a: pa, b: b, c: pc, m: l, k: k, n: n,
                          weightIsTransposed: weightIsTransposed, before: before)
        (c.contents() + first * n * 4).copyMemory(from: pc.contents(), byteCount: rows * n * 4)
        return time
    }

    /// **A weight's LoRA stack: `C += (x·bas)·haut`, in ONE submission.**
    ///
    /// The two skinny GEMMs are encoded in the **same** command buffer, and Metal guarantees
    /// ordering between two encoders of the same buffer — that is already what the fused widening
    /// relies on. A measurement showed that a LoRA's overhead has a **floor per
    /// submission**, constant with the rank; making two of them instead of one would double that floor
    /// for nothing.
    ///
    /// **Outside the conductor, and that is deliberate.** The GPU/AMX cut hands rows to
    /// `BNNSMatMul` while the GPU does the others; it has no way to accumulate onto an
    /// existing output — it **writes** its slice. Routing this through it would leave `C` partly overwritten
    /// instead of accumulated, hence a plausible and wrong image. And there would be nothing to gain: at
    /// `r = 32`, `x·bas` is a skinny GEMM limited by latency, not by compute.
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
        let previous = census[key] ?? (0, 0, Double.infinity)
        census[key] = (previous.count + 1, previous.seconds + elapsed, min(previous.best, elapsed))
        return elapsed
    }

    /// **`C += A · W`** — a GEMM that accumulates instead of writing, weight stored `[input, output]`.
    ///
    /// Two callers: the `to_out` of a FLUX.2 [klein] single block, which reads `[attention | MLP]`
    /// side by side (two products on the two halves of `k`, the second accumulated, make the same
    /// product without copying the two inputs into a third); and Qwen-Image-2.1's merged LoRA,
    /// `W += down · up` (`m` = the weight's `k`). Outside the conductor, like `lora` and for the
    /// same reason: the GPU/AMX cut writes its slice, it does not accumulate.
    ///
    /// ⚠️ **It does not split the rows into a multiple of 64 plus a padded tail** as `linear` and
    /// `lora` do (`rowSplit`): a height that is not a multiple of 64 can reach MPS's fast
    /// kernel that was caught returning NaN. Only the 8–32-row tiers are refused (precondition).
    @discardableResult
    package func accumulatedLinear(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int) -> Double {
        precondition(!GEMM.poisonedRows.contains(m), "accumulated GEMM at \(m) rows: outside the padding tiers")
        let commands = queue.makeCommandBuffer()!
        encoder(Shape(m: m, k: k, n: n, transposed: true, accumulates: true), into: commands, a, b, c)
        commands.commit()
        commands.waitUntilCompleted()
        let elapsed = Attention.verdict(commands, what: "accumulated GEMM \(m)×\(k)×\(n)")
        let key = "\(m)×\(k)×\(n)ᵀ+"
        let previous = census[key] ?? (0, 0, Double.infinity)
        census[key] = (previous.count + 1, previous.seconds + elapsed, min(previous.best, elapsed))
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
                        weightIsTransposed: Bool,
                        before: ((MTLCommandBuffer) -> Void)?) -> Double {
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
        before?(commands)
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
        let previous = census[key] ?? (0, 0, Double.infinity)
        census[key] = (previous.count + 1, previous.seconds + elapsed, min(previous.best, elapsed))
        return elapsed
    }
}
