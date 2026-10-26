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
/// The second measurement **froze** `T` at the ratio of the measured throughputs (0.631). That was
/// enough to judge the cut, not to absorb the thermal drift of a two-minute render — a series recorded **13.2 %** between
/// the first render and the fourth. A frozen `T` on a machine that slows down makes the fast
/// engine wait, and the wait is wall-clock time that neither of them counts as its own.
/// The conductor therefore servo-controls it, shape by shape, on the times actually observed.
///
/// ## Two things that cannot be guessed
///
/// 1. **The GPU always takes the first rows.** Its `MPSMatrix` therefore starts at offset zero, and
///    the question of a Metal offset's alignment never arises; it is the AMX, which works
///    on bare pointers, that takes the tail.
/// 2. **The reserve is shared** — both engines read the *same* fp32 weight and write into the
///    *same* output, each on its slice of rows. The first measurement read two distinct buffers, hence
///    314 MB of traffic where there is only 157: its measurement was **pessimistic** about
///    bandwidth, not optimistic.
///
/// **`@unchecked Sendable`**: the AMX thread reads `self` while the calling thread works
/// on the GPU, and that is intended. Their baton pass is made of two semaphores (`ready`, `done`):
/// the calling thread only writes the `Job` before `ready.signal()` and only reads the result after
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
    /// **Off by default on every chip**: the app's capacity map turns it on where it measures it
    /// faster (`CapacityRules.swift`), a developer with `SILICONED_AMX=1`. It changes the
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

    /// **Where the servo keeps `T`**: the GPU's fraction never below 0.30 nor above 0.95 — so the AMX
    /// slice, and the BNNS workspace sized on it, is at most `m − cut(m, 0.30)` rows (`MemoryNeed.gemm`).
    package static let fractionRange = 0.30...0.95

    private let queue: MTLCommandQueue
    private let job = UnsafeMutablePointer<Job>.allocate(capacity: 1)
    private let ready = Rendezvous(spinning: Conductor.spinning)
    private let done = Rendezvous(spinning: Conductor.spinning)
    private var worker: Thread?
    /// The time the AMX actually spent on its slice, written by the worker thread.
    private let amxSeconds = UnsafeMutablePointer<Double>.allocate(capacity: 1)
    /// The BNNS workspace, outside `malloc`: `Pages`, unmapped with the conductor.
    private var workspace: Pages?

    /// The fraction of rows entrusted to the GPU, **per shape** — `k×n`. Two shapes have neither the same
    /// throughput ratio nor the same sensitivity to `m`: a single `T` would serve one and
    /// penalize the other.
    package private(set) var fraction: [String: Double] = [:]
    /// **The frozen cut's `T` computed per product** (`Shares`, from the map's `cuts` through
    /// `EngineSettings.amxShares`): with a grid, it replaces `initialFraction`. The diagnostic empties it to
    /// measure at its own share.
    package var shares = EngineSettings.effective.amxShares
    /// **The GPU's share before the servo moves it** — the whole of a frozen cut's `T`:
    /// `amxFraction` by default; the diagnostic sets the share it has just measured on the product.
    package var initialFraction = EngineSettings.effective.amxFraction
    /// What each engine spent waiting for the other. This is the imbalance, and it is invisible
    /// if it is not recorded: neither the GPU nor the AMX counts it as its own.
    package private(set) var idleGPU = 0.0
    package private(set) var idleAMX = 0.0
    package private(set) var rendezvous = 0
    /// The FLOPs passed through the cut, and the wall-clock time they cost.
    package private(set) var coFLOP = 0.0
    package private(set) var coSeconds = 0.0
    /// The same, **per shape** (`k×n`): a total hides which shape makes which engine wait — a frozen
    /// `T` fits the shapes it was measured on, not every model's.
    package private(set) var perShape: [String: ShapeRecord] = [:]
    package struct ShapeRecord {
        package var calls = 0
        package var wall = 0.0, idleGPU = 0.0, idleAMX = 0.0, flop = 0.0
        package init(calls: Int = 0, wall: Double = 0, idleGPU: Double = 0, idleAMX: Double = 0, flop: Double = 0) {
            (self.calls, self.wall, self.idleGPU, self.idleAMX, self.flop) = (calls, wall, idleGPU, idleAMX, flop)
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

    package init(queue: MTLCommandQueue, driven: Bool = true) {
        self.driven = driven
        self.queue = queue
        job.pointee = Job()
        amxSeconds.pointee = 0
        // **The worker must not own the conductor**: it captured `self` and never returned, so
        // no conductor ever reached `deinit` — every model's `GEMM` left its thread and its BNNS
        // workspace behind (65 MB after Z-Image's DiT, 149 MB after Qwen's three models, for the
        // rest of the process). It holds the rendezvous and the job, and reaches the conductor only
        // to run a job, which a caller holding the conductor is waiting on; `deinit` posts the stop.
        // What the worker holds crosses threads by design. **Under the default rendezvous — two
        // semaphores — every access to the job and to its time is ordered** (the caller writes, posts
        // `ready`, waits on `done`, then reads; a semaphore's signal happens-before its wait's return).
        // **Not under `SILICONED_AMX_SPIN`**: a plain `Int32` flag with `OSMemoryBarrier` is no
        // synchronization in Swift's memory model, and that mode is known to return wrong products
        // (`spinning`) — a fault injector kept so the AMX check can fail, never a mode of the product.
        struct Shared: @unchecked Sendable {
            let job: UnsafeMutablePointer<Job>, amxSeconds: UnsafeMutablePointer<Double>
            let ready: Rendezvous, done: Rendezvous
        }
        let shared = Shared(job: job, amxSeconds: amxSeconds, ready: ready, done: done)
        unowned(unsafe) let owner = self
        let thread = Thread {
            while true {
                shared.ready.wait()
                if shared.job.pointee.stop { shared.done.signal(); return }
                let j = shared.job.pointee
                let started = DispatchTime.now().uptimeNanoseconds
                owner.amx(j)
                shared.amxSeconds.pointee = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
                shared.done.signal()
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
        let (status, size) = Conductor.multiply(a: aPointer.advanced(by: j.offset * j.k * 4), b: bPointer,
                                                c: cPointer.advanced(by: j.offset * j.n * 4), rows: j.rows,
                                                k: j.k, n: j.n, transposeWeight: j.transposeWeight,
                                                workspace: &workspace)
        lastWorkspace = size
        if size >= 0 { lastStatus = status }
        if status != 0 { refusals += 1 }
    }

    /// **The AMX's side of the cut, on the calling thread**: `C [rows, n] = A [rows, k] · B`, `B` stored
    /// `[k, n]` (or `[n, k]` under `transposeWeight`), one `BNNSMatMul` — what `amx` runs on the worker
    /// thread, and what the diagnostic times as "the AMX alone". Returns BNNS's status (−1 when it
    /// refuses the shapes) and the workspace it asked for.
    ///
    /// The weight is stored `[input, output]` by forge v3, so nothing to transpose; if it were not, it
    /// would be `[output, input]` and it is `B` that we would transpose — the same decision as
    /// `transposeRight` on the MPS side, made at the same place.
    package static func multiply(a: UnsafeMutableRawPointer, b: UnsafeMutableRawPointer, c: UnsafeMutableRawPointer,
                                 rows: Int, k: Int, n: Int, transposeWeight: Bool,
                                 workspace: inout Pages?) -> (status: Int32, workspaceBytes: Int) {
        var (A, B, C) = descriptors(a: a, b: b, c: c, rows: rows, k: k, n: n, transposeWeight: transposeWeight)
        var params = BNNSFilterParameters()
        let size = amx_matmul_workspace(false, transposeWeight, 1.0, &A, &B, &C, &params)
        guard size >= 0 else { return (-1, size) }
        // The workspace is kept and grown, never reallocated per call: 13 to 67 MB per GEMM, 238 GEMMs
        // per evaluation, would be gigabytes of mapping and unmapping for nothing. It ends at the
        // largest `workspaceBytes` of the AMX slices this conductor ran.
        if size > (workspace?.capacity ?? 0) { workspace = Pages(bytes: size) }
        return (amx_matmul(false, transposeWeight, 1.0, &A, &B, &C, size > 0 ? workspace?.base : nil, &params), size)
    }

    /// **The AMX on columns `first ..< first + width` of `C [m, n]`, on the calling thread**: `C[:, first…]
    /// = A · slice`, the slice `[k, width]` contiguous (widened for it alone), `C` strided by `n` — the AMX's
    /// side of the cut by columns (`WeightProducts`, a published int8 weight the GPU reads as it is). The
    /// conductor's workspace, the worker idle: the products run one at a time. Returns BNNS's status.
    package func columns(a: UnsafeMutableRawPointer, slice: UnsafeMutableRawPointer, c: UnsafeMutableRawPointer,
                         m: Int, k: Int, n: Int, first: Int, width: Int) -> Int32 {
        var (A, B, C) = Conductor.columnDescriptors(a: a, slice: slice, c: c + first * 4, m: m, k: k, n: n, width: width)
        var params = BNNSFilterParameters()
        let size = amx_matmul_workspace(false, false, 1.0, &A, &B, &C, &params)
        guard size >= 0 else { refusals += 1; return -1 }
        if size > (workspace?.capacity ?? 0) { workspace = Pages(bytes: size) }
        lastWorkspace = size
        let status = Int32(amx_matmul(false, false, 1.0, &A, &B, &C, size > 0 ? workspace?.base : nil, &params))
        lastStatus = status
        if status != 0 { refusals += 1 }
        return status
    }

    /// **The BNNS workspace of the AMX's `width` columns** of `[m, k] × [k, n]` (`columns`): BNNS answers
    /// from the shapes and strides alone.
    package static func columnWorkspaceBytes(m: Int, k: Int, n: Int, width: Int) -> Int {
        guard width > 0 else { return 0 }
        var (A, B, C) = columnDescriptors(a: nil, slice: nil, c: nil, m: m, k: k, n: n, width: width)
        var params = BNNSFilterParameters()
        return max(0, amx_matmul_workspace(false, false, 1.0, &A, &B, &C, &params))
    }

    private static func columnDescriptors(a: UnsafeMutableRawPointer?, slice: UnsafeMutableRawPointer?,
                                          c: UnsafeMutableRawPointer?, m: Int, k: Int, n: Int, width: Int)
        -> (a: BNNSNDArrayDescriptor, b: BNNSNDArrayDescriptor, c: BNNSNDArrayDescriptor) {
        func nd(_ p: UnsafeMutableRawPointer?, rows: Int, columns: Int, stride: Int) -> BNNSNDArrayDescriptor {
            var d = BNNSNDArrayDescriptor()
            d.layout = BNNSDataLayoutRowMajorMatrix
            d.size = (columns, rows, 0, 0, 0, 0, 0, 0)
            d.stride = (1, stride, 0, 0, 0, 0, 0, 0)
            d.data = p; d.data_type = .float; d.data_scale = 1; d.data_bias = 0
            return d
        }
        return (nd(a, rows: m, columns: k, stride: k), nd(slice, rows: k, columns: width, stride: width),
                nd(c, rows: m, columns: width, stride: n))
    }

    /// **The BNNS workspace of an AMX slice of `rows` rows** of `[rows, k] × [k, n]`: what `amx` asks of
    /// BNNS for that job, with the same descriptors — BNNS answers from the shapes alone. A conductor
    /// holds the largest over the slices it ran (`Pages`, rounded to the page); the slice is
    /// `m − cut(m, T)` rows, at most `m − quantum`. 31 MB at Z-Image's `1536 × 3840 × 3840`, 67 MB at
    /// `1536 × 10240 × 3840`.
    package static func workspaceBytes(rows: Int, k: Int, n: Int, weightIsTransposed: Bool) -> Int {
        guard rows > 0 else { return 0 }
        var (A, B, C) = descriptors(a: nil, b: nil, c: nil, rows: rows, k: k, n: n, transposeWeight: !weightIsTransposed)
        var params = BNNSFilterParameters()
        return max(0, amx_matmul_workspace(false, !weightIsTransposed, 1.0, &A, &B, &C, &params))
    }

    private static func descriptors(a: UnsafeMutableRawPointer?, b: UnsafeMutableRawPointer?, c: UnsafeMutableRawPointer?,
                                    rows: Int, k: Int, n: Int, transposeWeight: Bool)
        -> (a: BNNSNDArrayDescriptor, b: BNNSNDArrayDescriptor, c: BNNSNDArrayDescriptor) {
        func nd(_ p: UnsafeMutableRawPointer?, _ r: Int, _ columns: Int) -> BNNSNDArrayDescriptor {
            var d = BNNSNDArrayDescriptor()
            d.layout = BNNSDataLayoutRowMajorMatrix
            d.size = (columns, r, 0, 0, 0, 0, 0, 0)
            d.data = p
            d.data_type = .float
            d.data_scale = 1
            d.data_bias = 0
            return d
        }
        return (nd(a, rows, k), transposeWeight ? nd(b, n, k) : nd(b, k, n), nd(c, rows, n))
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
        let T = fraction[key] ?? shares.share(k: k, n: n, m: m) ?? initialFraction
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
        perShape[key, default: ShapeRecord()].calls += 1
        perShape[key]!.wall += wall
        perShape[key]!.idleGPU += max(0, wall - gpuWall)
        perShape[key]!.idleAMX += max(0, wall - amxWall)
        perShape[key]!.flop += 2.0 * Double(m) * Double(k) * Double(n)

        // ── the servo-control ───────────────────────────────────────────────────────────────
        //
        // We equalize the **times**, not the rows: `T* = gpu_rate / (gpu_rate + amx_rate)`.
        // The correction is damped (a quarter of the way) because a whole step on a noisy
        // measurement oscillates, and an oscillation of `T` makes the two engines wait
        // alternately — exactly what we are trying to eliminate.
        if driven, gpuWall > 0, amxWall > 0 {
            let rateGPU = Double(gpuRows) / gpuWall, rateAMX = Double(amxRows) / amxWall
            let target = rateGPU / (rateGPU + rateAMX)
            fraction[key] = min(Conductor.fractionRange.upperBound, max(Conductor.fractionRange.lowerBound, T + 0.25 * (target - T)))
        }
        return wall
    }

    /// **The GPU's share of a product, computed from its shape and the machine's map** — never a constant of
    /// a machine or of a model:
    ///
    ///     T(k, n, m) = 1 / (1 + ρ(k, n, m))
    ///     ρ(k, n, m) = ρ(k, w, m) · ρ(w, n, m) / ρ(w, w, m)      ρ = the AMX's rate over the GPU's, side by side
    ///
    /// The map (`Diagnostic.Cut`) measures `ρ` along two axes through one square shape `w × w`: `k` at
    /// `n = w`, `n` at `k = w`, at both heights. Each axis is read linearly in `log k` or `log n` (the
    /// nearest point outside), the two composed, then read between the heights in `log m`. Any model's
    /// product enters by its own `k`, `n` and `m`: the grid is the machine's, the shapes the model's.
    ///
    /// Why two axes (measured on an M1 Pro): the AMX's rate falls with the length of its dot products `k` — the
    /// map's ρ goes from 0.52 at `k` = 2 048 to 0.36 at 16 384 (1 024 rows), and the engine's servo balances
    /// Anima's 2 048 × 8 192 at 0.63 and its 8 192 × 2 048 at 0.76, the same weight —; the width `n` may move
    /// it too (Krea 2's servo: 6 144 × 16 384 at 0.75, 6 144 × 6 144 at 0.69), and a measured axis lets it
    /// enter. This Mac's `n` axis came out flat (0.68–0.70 at `k` = 6 144), the gap with the engine not
    /// explained. Against one share for every shape, an evaluation: Anima −2 % at 512², −4 % at 1024²,
    /// Z-Image −2.7 %, Krea 2 −1.5 % at 1024², +2.5 % at 512² (its 6 144 × 16 384 at 0.677).
    ///
    /// **Pure**: the same map, shape and height give the same share — the frozen cut stays reproducible bit
    /// for bit on one machine. An empty grid gives `nil`: the scalar `amxFraction` then serves. A grid
    /// without its square point reads its `k` axis only.
    package struct Shares: Equatable, Sendable, CustomStringConvertible {
        /// `ρ` measured on the product `k × n` at height `m`.
        package struct Point: Equatable, Sendable {
            package let k, n, m: Int
            package let ratio: Double
            package init(k: Int, n: Int, m: Int, ratio: Double) { self.k = k; self.n = n; self.m = m; self.ratio = ratio }
        }
        package let points: [Point]
        package init(_ points: [Point]) {
            self.points = points.sorted { ($0.m, $0.k, $0.n) < ($1.m, $1.k, $1.n) }
        }
        package static let none = Shares([])
        package var isEmpty: Bool { points.isEmpty }

        /// `T(k, n, m)`, clamped to `range` (`Conductor.fractionRange` for the AMX's cut; the ANE's is wider).
        package func share(k: Int, n: Int, m: Int, range: ClosedRange<Double> = Conductor.fractionRange) -> Double? {
            ratio(k: k, n: n, m: m).map { min(range.upperBound, max(range.lowerBound, 1 / (1 + $0))) }
        }

        /// `ρ(k, n, m)` itself, read as `share` reads it, unclamped: what a grid of 0 and 1 (`ane_amx`) says of a product.
        package func ratio(k: Int, n: Int, m: Int) -> Double? {
            let heights = Array(Set(points.map(\.m))).sorted()
            guard !heights.isEmpty else { return nil }
            /// Linear in `log x` between the two points around `x`, the nearest outside.
            func between<T>(_ items: [T], _ x: Int, at: (T) -> Int, value: (T) -> Double) -> Double {
                let sorted = items.sorted { at($0) < at($1) }
                if x <= at(sorted[0]) { return value(sorted[0]) }
                if x >= at(sorted[sorted.count - 1]) { return value(sorted[sorted.count - 1]) }
                let i = sorted.firstIndex { at($0) >= x }!
                let (a, b) = (sorted[i - 1], sorted[i])
                let t = log(Double(x) / Double(at(a))) / log(Double(at(b)) / Double(at(a)))
                return value(a) + t * (value(b) - value(a))
            }
            func ratio(at h: Int) -> Double {
                let level = points.filter { $0.m == h }
                // The shape measured itself, when it was; then one axis, or two through one pivot (the AMX's
                // grid); otherwise (the ANE's grid, a family's shapes) the nearest shape in log k + log n.
                if let exact = level.first(where: { $0.k == k && $0.n == n }) { return exact.ratio }
                if level.allSatisfy({ $0.n == level[0].n }) { return between(level, k, at: \.k, value: \.ratio) }
                let pivot = level.first(where: { $0.k == $0.n })
                guard let pivot, level.allSatisfy({ $0.k == pivot.k || $0.n == pivot.n }) else {
                    func distance(_ p: Point) -> Double { abs(log(Double(p.k) / Double(k))) + abs(log(Double(p.n) / Double(n))) }
                    return level.min { distance($0) < distance($1) }!.ratio
                }
                let depth = between(level.filter { $0.n == pivot.n }, k, at: \.k, value: \.ratio)
                let width = between(level.filter { $0.k == pivot.k }, n, at: \.n, value: \.ratio)
                return depth * width / pivot.ratio
            }
            return between(heights.map { ($0, ratio(at: $0)) }, m, at: \.0, value: \.1)
        }

        /// `2048×6144@1024=0.601,…` — `k×n@m=ρ`, the setting's text (`amx_shares`), the profile's and the
        /// environment's.
        package var description: String {
            points.map { "\($0.k)×\($0.n)@\($0.m)=\(String(format: "%.3f", $0.ratio))" }.joined(separator: ",")
        }
        /// The text `description` writes; `nil` when a piece does not read. Empty: no grid.
        package init?(text: String) {
            var points: [Point] = []
            for piece in text.split(separator: ",") {
                let parts = piece.split(whereSeparator: { "×@=".contains($0) }).map(String.init)
                guard parts.count == 4, let k = Int(parts[0]), let n = Int(parts[1]), let m = Int(parts[2]),
                      let ratio = Double(parts[3]) else { return nil }
                points.append(Point(k: k, n: n, m: m, ratio: ratio))
            }
            self.init(points)
        }
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
            : String(format: "%.3f, frozen (reproducible)", initialFraction)
        return String(format: "co-execution: %.3f TFLOP/s over %.1f s wall, %d rendezvous   ·   "
                            + "GPU idle %.2f s, AMX %.2f s   ·   BNNS: %d refusals, status %d, "
                            + "workspace %d B   ·   T = %@",
                      coFLOP / coSeconds / 1e12, coSeconds, rendezvous, idleGPU, idleAMX,
                      refusals, Int(lastStatus), lastWorkspace, cut)
    }
}
