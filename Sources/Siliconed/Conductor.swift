@preconcurrency import Accelerate
internal import AMXLegacy
import Darwin
import Foundation
import Metal
import MetalPerformanceShaders

/// **Stage 3: a GEMM cut by tokens, the GPU and the AMX on it at the same time.**
///
/// This is the biggest lever of the project after the spectral schedule.
/// A first measurement on two independent GEMMs: **+1.86 TFLOP/s net**, sum 5.333 against 3.478 for
/// the GPU alone, the AMX keeping 91 % of its solo throughput and costing the GPU only 3.2 %. A second
/// measured the cut itself — a single GEMM, shared reserve, spinning flag — at **5.360**, hence
/// *above* the first: reading a single weight instead of two brings in more than the rendezvous costs.
///
/// The thermal control framed the first (3.478 before, 3.468 after, −0.3 %): the GPU's drop is indeed
/// co-execution and nothing else.
///
/// ## What this file adds to the second measurement
///
/// The second measurement **froze** `T` at the ratio of the measured throughputs (0.631). That was enough to judge the cut,
/// not to absorb the thermal drift of a two-minute render — a series recorded **13.2 %** between
/// the first render and the fourth. A frozen `T` on a machine that slows down makes the fast
/// engine wait, and the wait is wall-clock time that neither of them counts as its own.
/// The conductor therefore servo-controls it, shape by shape, on the times actually observed.
///
/// ## Three things that cannot be guessed
///
/// 1. **The GPU always takes the first rows.** Its `MPSMatrix` therefore starts at offset zero, and
///    the question of a Metal offset's alignment never arises; it is the AMX, which works
///    on bare pointers, that takes the tail.
/// 2. **The reserve is shared** — both engines read the *same* fp32 weight and write into the
///    *same* output, each on its slice of rows. The first measurement read two distinct buffers, hence
///    314 MB of traffic where there is only 157: its measurement was **pessimistic** about
///    bandwidth, not optimistic.
/// 3. **Fused widening is incompatible with co-execution**, and silently so.
///    Fused, the weight is widened *in the GEMM's command buffer*: Metal orders its two
///    encoders, but the AMX is not in that buffer — it would read the reserve while the GPU
///    is still writing it. The conductor therefore refuses to start if both flags are set, rather
///    than return plausible and wrong numbers.
///
/// **`@unchecked Sendable`**: the AMX thread reads `self` while the calling thread works
/// on the GPU, and that is intended. Their baton pass is made of two semaphores (`ready`, `done`)
/// : the calling thread only writes the `Job` before `ready.signal()` and only reads the result after
/// `done.wait()`. Each writes its own slice of the output's rows (point 2 above).
package final class Conductor: @unchecked Sendable {
    /// A task entrusted to the AMX engine. Written by the main thread, read by the worker thread,
    /// the two separated by the rendezvous — hence no race, and no lock.
    private struct Job {
        var rows = 0, k = 0, n = 0, offset = 0
        var transposeWeight = false
        var stop = false
    }

    /// The rendezvous: a **semaphore** by default, a spinning flag under `SILICONED_AMX_SPIN`.
    ///
    /// A measurement had found the flag faster (5.360 TFLOP/s against 5.214, +2.8 %), but the flag makes
    /// `BNNSMatMul` return wrong products (see `spinning`): it survives only as the way to
    /// make the check fail. In flag mode, `OSMemoryBarrier` orders the write of the **result**
    /// before that of the flag (an aligned `Int32` is read and written in one instruction on arm64).
    private final class Rendezvous {
        private let flag = UnsafeMutablePointer<Int32>.allocate(capacity: 1)
        private let semaphore = DispatchSemaphore(value: 0)
        private let spinning: Bool
        init(spinning: Bool) { self.spinning = spinning; flag.pointee = 0 }
        deinit { flag.deallocate() }
        func signal() {
            if spinning { OSMemoryBarrier(); flag.pointee = 1 } else { semaphore.signal() }
        }
        func wait() {
            if spinning {
                while flag.pointee == 0 { }
                OSMemoryBarrier()
                flag.pointee = 0
            } else { semaphore.wait() }
        }
    }
    /// Read once from `EngineSettings.amx`: constant for the duration of the process.
    ///
    /// **On by default on the reference chip only** (`EngineSettings.referenceChip`), where it was
    /// measured in the engine; off elsewhere until the bench measures it. It changes the
    /// last bits, so the developer checks force it off: a verdict — a PSNR, a GEMM
    /// census, a footprint — is never skewed by a lever one forgot was on.
    package static let enabled = EngineSettings.effective.amx

    /// **`SILICONED_AMX_SPIN=1`** — the spinning flag, **which returns wrong results**.
    ///
    /// A measurement had preferred the flag to the semaphore on a throughput gap: 5.360 against 5.214 TFLOP/s,
    /// i.e. +2.8 %, and a wait recorded at 0 µs against 505. What it did not measure was the
    /// **result** — and the AMX check (against `cblas`) measures it: with the flag, `BNNSMatMul` returns a
    /// wrong product, **different at every run** (relative error 0.57 then 0.73 then 0.95 on the
    /// same inputs); without it, 1.146·10⁻⁶, which is its summation order and nothing else.
    ///
    /// The mechanism was not pursued further than the fact: a thread spinning idle
    /// (`while flag.pointee == 0 { }`) occupies a core from the start to the end of the render and never gives
    /// control back, and Accelerate comes out of it damaged. **It is the fact that decides**, and it is reproducible
    /// both ways.
    ///
    /// The flag stays, behind this name, because it is the **way to make** the check fail
    /// — a check that cannot fail checks nothing. But it is no longer the
    /// default, and the flag's +2.8 % was the price of a correct computation.
    package static let spinning = EngineSettings.effective.amxSpin

    /// The number of threads entrusted to BNNS. **Two**, because it was measured that there is one AMX block per
    /// P cluster — two on this machine — and that two threads return exactly the throughput of "all
    /// the threads" (1.969 TFLOP/s in both cases). Letting BNNS decide makes it partition
    /// according to the cores it believes are free, yet the conductor occupies one of them.
    package static let threads = EngineSettings.effective.amxThreads

    /// The cut's step. Big enough for the MPS operator cache to stay small, fine enough
    /// for the servo-control to keep something to correct with: at 4,128 rows, 256 leaves sixteen points.
    package static let quantum = EngineSettings.effective.amxQuantum

    /// Where the cut falls, **defined only once**. The AMX check must know the
    /// same thing as the conductor: recomputing it on its side is arranging a meeting at two
    /// places that will drift — and it happened at the very first change of quantum.
    package static func cut(_ m: Int, _ T: Double) -> Int {
        let q = Conductor.quantum
        return max(q, min(m - q, Int((Double(m) * T / Double(q)).rounded()) * q))
    }

    /// Below this, the cut brings nothing: the text blocks run at `m = 32`, where the rendezvous
    /// alone would cost more than the entrusted computation. The threshold is a setting, not a law.
    package static let minimumRows = EngineSettings.effective.amxMinimumRows

    private let queue: MTLCommandQueue
    private let job = UnsafeMutablePointer<Job>.allocate(capacity: 1)
    private let ready = Rendezvous(spinning: Conductor.spinning)
    private let done = Rendezvous(spinning: Conductor.spinning)
    private var worker: Thread?
    /// The time the AMX actually spent on its slice, written by the worker thread.
    private let amxSeconds = UnsafeMutablePointer<Double>.allocate(capacity: 1)
    private var workspace: UnsafeMutableRawPointer?
    private var workspaceBytes = 0

    /// The fraction of rows entrusted to the GPU, **per shape** — `k×n`. Two shapes have neither the same
    /// throughput ratio nor the same sensitivity to `m`: a single `T` would serve one and
    /// penalize the other.
    package private(set) var fraction: [String: Double] = [:]
    /// What each engine spent waiting for the other. This is the imbalance, and it is invisible
    /// if it is not recorded: neither the GPU nor the AMX counts it as its own.
    package private(set) var idleGPU = 0.0
    package private(set) var idleAMX = 0.0
    package private(set) var rendezvous = 0
    /// The FLOPs passed through the cut, and the wall-clock time they cost.
    package private(set) var coFLOP = 0.0
    package private(set) var coSeconds = 0.0

    package enum Failure: Error, CustomStringConvertible {
        case incompatible(String)
        package var description: String {
            switch self { case .incompatible(let why): return "co-execution: \(why)" }
        }
    }

    /// **Is the cut servo-controlled?** True by default, and it is the speed path: `T` follows the
    /// thermal drift, which was recorded at 13.2 % over four renders.
    ///
    /// False in reproducible mode. `T` then stays at `EngineSettings.effective.amxFraction` from the first GEMM
    /// to the last, so `cut(m, T)` is a pure function of the shape — and two renders of the same
    /// seed return **the same bits**. It is not an approximation: the cut does not reassociate
    /// any sum, each row of `C` being computed entirely by just one of the two engines.
    /// What is lost is the drift correction, not the accuracy.
    package let driven: Bool

    package init(queue: MTLCommandQueue, fusedWiden: Bool, driven: Bool = true) throws {
        self.driven = driven
        guard !fusedWiden else {
            throw Failure.incompatible(
                "SILICONED_WIDEN_FUSED widens the weight in the GEMM's command buffer; "
                + "the AMX is not in that buffer and would read the reserve while the GPU is writing it")
        }
        self.queue = queue
        job.pointee = Job()
        amxSeconds.pointee = 0
        let thread = Thread { [self] in
            while true {
                ready.wait()
                if job.pointee.stop { done.signal(); return }
                let j = job.pointee
                let started = DispatchTime.now().uptimeNanoseconds
                amx(j)
                amxSeconds.pointee = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
                done.signal()
            }
        }
        thread.qualityOfService = .userInitiated
        thread.name = "siliconed.amx"
        worker = thread
        thread.start()
    }

    deinit {
        if worker != nil {
            job.pointee = Job(stop: true)
            ready.signal()
            done.wait()
        }
        job.deallocate()
        amxSeconds.deallocate()
        workspace?.deallocate()
    }

    // ── the AMX engine ───────────────────────────────────────────────────────────────────────

    private var aPointer: UnsafeMutableRawPointer!
    private var bPointer: UnsafeMutableRawPointer!
    private var cPointer: UnsafeMutableRawPointer!

    /// **What BNNS answers, counted.** A `BNNSMatMul` that refuses throws nothing: it returns a status
    /// and writes not a single row — the AMX's slice stays as it was, and the result is wrong
    /// *exactly there*. Same family as the failing `MTLCommandBuffer` that throws nothing: an operation that
    /// fails silently reads as a computation, not as a breakdown.
    package private(set) var refusals = 0
    package private(set) var lastStatus: Int32 = 0
    package private(set) var lastWorkspace = 0

    private func amx(_ j: Job) {
        guard j.rows > 0 else { return }
        // The weight is stored `[input, output]` by forge v3, so nothing to transpose; if it
        // were not, it would be `[output, input]` and it is `B` that we would transpose — the same
        // decision as `transposeRight` on the MPS side, made at the same place.
        func nd(_ p: UnsafeMutableRawPointer, _ r: Int, _ columns: Int) -> BNNSNDArrayDescriptor {
            var d = BNNSNDArrayDescriptor()
            d.layout = BNNSDataLayoutRowMajorMatrix
            d.size = (columns, r, 0, 0, 0, 0, 0, 0)
            d.data = p
            d.data_type = .float
            d.data_scale = 1
            d.data_bias = 0
            return d
        }
        var A = nd(aPointer.advanced(by: j.offset * j.k * 4), j.rows, j.k)
        var B = j.transposeWeight ? nd(bPointer, j.n, j.k) : nd(bPointer, j.k, j.n)
        var C = nd(cPointer.advanced(by: j.offset * j.n * 4), j.rows, j.n)
        var params = BNNSFilterParameters()
        let size = amx_matmul_workspace(false, j.transposeWeight, 1.0, &A, &B, &C, &params)
        lastWorkspace = size
        guard size >= 0 else { refusals += 1; return }
        // The workspace is kept and grown, never reallocated: 13 MB per GEMM, 238 GEMMs per
        // evaluation, would be 3 GB of back-and-forth in the allocator for nothing.
        if size > workspaceBytes {
            workspace?.deallocate()
            workspace = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: 128)
            workspaceBytes = size
        }
        lastStatus = amx_matmul(false, j.transposeWeight, 1.0, &A, &B, &C,
                                size > 0 ? workspace : nil, &params)
        if lastStatus != 0 { refusals += 1 }
    }
    // ── the cut ──────────────────────────────────────────────────────────────────────────────

    /// `C = A × B`, the `m` rows shared between the GPU and the AMX. Returns the **wall-clock time**, which is
    /// the only one that makes sense here: adding up two engines that run together says nothing.
    ///
    /// Returns `nil` when the cut does not apply — too few rows — and the caller resumes its ordinary
    /// path. *A path that silently decides not to apply is a path of which we will not know,
    /// later, whether it was used.* Hence the `rendezvous` counter.
    package func linear(a: MTLBuffer, b: MTLBuffer, c: MTLBuffer,
                       m: Int, k: Int, n: Int, weightIsTransposed: Bool,
                       multiply: (Int) -> Double) -> Double? {
        guard m >= Conductor.minimumRows else { return nil }
        let key = "\(k)×\(n)"
        // The starting point is the measured throughput ratio (3.364 / 5.333). It does not stay
        // long: the first shapes correct it within a few blocks — unless the cut is
        // frozen, in which case it is the end-to-end value, and `fraction` stays empty.
        let T = fraction[key] ?? EngineSettings.effective.amxFraction
        // A multiple of `quantum` (256 by default), so a multiple of any tile: a slice that falls
        // in the middle of a tile makes the GPU pay for a partial row, and that is the kind of loss
        // that does not show in a total. **The cut is quantized, and not merely aligned.**
        //
        // `submit` is called with `m = gpuRows`, and `GEMM` memoizes its `MPSMatrixMultiplication`s
        // **per shape**. A continuous servo-control produces a different `gpuRows` at almost every
        // call — hence a new operator at every call, which was measured at **3.35 → 1.46
        // TFLOP/s**, and a cache that grows without bound. Rounding to `quantum` brings the number of
        // possible shapes down to a few dozen, which the first blocks create once and for all.
        let gpuRows = Conductor.cut(m, T)
        let amxRows = m - gpuRows

        aPointer = a.contents()
        bPointer = b.contents()
        cPointer = c.contents()
        job.pointee = Job(rows: amxRows, k: k, n: n, offset: gpuRows,
                          transposeWeight: !weightIsTransposed, stop: false)

        let began = DispatchTime.now().uptimeNanoseconds
        ready.signal()
        _ = multiply(gpuRows)                       // the GPU, rows 0..<gpuRows, offset zero
        let gpuDone = DispatchTime.now().uptimeNanoseconds
        done.wait()
        let ended = DispatchTime.now().uptimeNanoseconds

        let wall = Double(ended - began) * 1e-9
        let gpuWall = Double(gpuDone - began) * 1e-9
        let amxWall = amxSeconds.pointee
        idleGPU += max(0, wall - gpuWall)
        idleAMX += max(0, wall - amxWall)
        rendezvous += 2
        coFLOP += 2.0 * Double(m) * Double(k) * Double(n)
        coSeconds += wall

        // ── the servo-control ───────────────────────────────────────────────────────────────
        //
        // We equalize the **times**, not the rows: `T* = gpu_rate / (gpu_rate + amx_rate)`.
        // The correction is damped (a quarter of the way) because a whole step on a noisy
        // measurement oscillates, and an oscillation of `T` makes the two engines wait
        // alternately — exactly what we are trying to eliminate.
        if driven, gpuWall > 0, amxWall > 0 {
            let rateGPU = Double(gpuRows) / gpuWall, rateAMX = Double(amxRows) / amxWall
            let target = rateGPU / (rateGPU + rateAMX)
            fraction[key] = min(0.95, max(0.30, T + 0.25 * (target - T)))
        }
        return wall
    }

    /// What the cut returned, in a word — to display next to the GEMM census.
    package var summary: String {
        guard coSeconds > 0 else { return "co-execution: never applied" }
        // Frozen cut: `fraction` is empty by construction, so displaying it would say "T =" and
        // nothing. We write the value that actually served, and the fact that it did not move.
        let cut = driven
            ? fraction.sorted { $0.key < $1.key }
                  .map { "\($0.key) → \(String(format: "%.3f", $0.value))" }
                  .joined(separator: ", ")
            : String(format: "%.3f, frozen (reproducible)", EngineSettings.effective.amxFraction)
        return String(format: "co-execution: %.3f TFLOP/s over %.1f s wall, %d rendezvous   ·   "
                            + "GPU idle %.2f s, AMX %.2f s   ·   BNNS: %d refusals, status %d, "
                            + "workspace %d B   ·   T = %@",
                      coFLOP / coSeconds / 1e12, coSeconds, rendezvous, idleGPU, idleAMX,
                      refusals, Int(lastStatus), lastWorkspace, cut)
    }
}
