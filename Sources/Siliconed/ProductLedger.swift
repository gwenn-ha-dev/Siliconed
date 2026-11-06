import Foundation
import Synchronization

/// **What a render's denoising computed, by class, with the route each product and attention took and the wall
/// seconds it cost** — the render's side of the check against the plan's prediction (`RenderLoop`). The engine adds one line per
/// execution where it already knows what it runs (`WeightProducts.linear` for every product by a weight of the five
/// DiTs, the three attention kernels' `run`), so no model keeps a copy of it and no census is parsed back.
///
///     Engine.execute ── recording(into:) ──▶ chain.denoising.denoise ──▶ DiT ──▶ WeightProducts.linear ─┐
///                                                                              └▶ Attention / Flash / Split run ─┤
///                                                                                                              ▼
///     Render.classes = ledger.tallies          [(work, route) → executions, seconds]      ◀─────────── ProductLedger.add
///
/// What sets it apart — what a census written by analogy with `GEMM.census` would get wrong:
///
///   - **The class is the caller's, not the kernel's.** `GEMM.census` keys MPS submissions: a 4 117-row product is
///     there as a body of 4 096 rows and a tail of 64, and an int8 or Q8_0 product (`TensorGEMM`, straight from
///     `WeightProducts`) is not in it at all. The ledger is told the product's own `k × n @ m` and the weight's
///     format, at the call that chose how the weight is read — the one place that knows both and the route.
///   - **The route is the one taken, not the one planned**: `gpu`, `gpu+amx`, `gpu+ane`, `gpu+ane+amx` for a widened
///     weight (the ANE's refusal, the conductor's declining a short tail: what ran), `gpu/bf16`, `gpu/bf16act`,
///     `gpu/int8`, `gpu+amx/int8`, `gpu/q8`, `gpu+amx/q8` for a weight read as published; `graph`, `flash`, `split`
///     for an attention. The route names of the plan's assignments, so a line compares with its prediction as it stands.
///   - **Heights stay the render's** (4 117, not 4 096): merging within the comparison's gap is its own, where the
///     catalogue's heights meet them (`RenderLoop`).
///   - **Armed for the denoising only**, around `DenoisingModule.denoise` and for one render at a time (the engine's
///     queue): the text encoder's and the decoder's products are not the check's, and a check, the diagnostic or a
///     probe that builds a `GEMM` outside a render records nothing and pays a lock to learn it.
///   - **Free in bits and nearly in time**: the ledger reads the clock twice around a product (a few tens of
///     nanoseconds for a product of milliseconds) and changes neither the order of the work nor a route.
///   - **On the stream, the seconds are the engines', not the wall** (`Charge`): a call that only encodes has a wall of ~0.1 ms, so a
///     product or an attention queued on a `GPUStream` is charged its buffers' GPU seconds at the drain and its ANE rows' seconds on
///     the ANE's thread. Off the stream, the wall, as before: same lines, same seconds.
///   - **Attentions under 64 rows are no class** (`CapacityClasses`), so they are not recorded.
///   - **A route not taken is named, never only absent** (`Fallback`): the plan without an answer (`?? .gpuAlone`), a path the process
///     cannot run (`WeightProducts.prepare` skips down to the plain or widened weight), the ANE's rows not accepted or given back. The
///     tallies say what ran; the fallbacks say what was planned instead and why — a render on the GPU alone where the plan said
///     `gpu+ane/bf16act` is otherwise a slow class with nothing to say why.
package final class ProductLedger: Sendable {
    /// **Why a product did not run as planned.** Each is a place the engine used to fall back without a word.
    package enum Reason: String, Sendable, CaseIterable, Comparable {
        /// The plan has no model: every product on the GPU alone, widened (`CapacityPlan.solved`).
        case noModel = "no model"
        /// The model gave no answer for this product (no line for the GPU in fp32): the GPU alone, widened.
        case unsolved
        /// The bf16 weight as published, planned, cannot be read here (no bf16 tensor or kernel): widened.
        case bf16Unavailable = "bf16 unavailable"
        /// The rounded activation, planned, does not fit the reserve or this caller does not round: the bf16 weight unrounded.
        case roundingUnavailable = "rounding unavailable"
        /// The 8-bit weight as published, planned, cannot be read here (no kernel, a layout or a reserve that does not fit): widened.
        case int8Unavailable = "int8 unavailable"
        /// A K-quant's blocks, planned as published (`gpu/q4kact`), are of a type the kernels do not read (Q6_K, a legacy one), or the
        /// route cannot run here (no kernel, a reserve too small, a caller that does not round): the plan's answer without that route.
        case kquantUnread = "k-quant unread"
        /// The activation quantized to int8 (W8A8), planned, cannot be quantized here (no kernel, a rotation group or a row it does not
        /// take, a reserve too small, a caller that does not quantize): the integers read on the fp32 activation.
        case quantizingUnavailable = "quantizing unavailable"
        /// The ANE's rows, planned, not handed to it (no ANE in this process, a shape it does not accept): the GPU took them.
        case aneNotAccepted = "ane not accepted"
        /// The ANE's rows refused or not finite after the call: the whole product again on the GPU.
        case aneGivenBack = "ane given back"
        package static func < (a: Reason, b: Reason) -> Bool { allCases.firstIndex(of: a)! < allCases.firstIndex(of: b)! }
    }

    /// One line of fallbacks: `executions` of `work` planned on `planned` (the plan's route name) that ran otherwise, for `reason`.
    package struct Fallback: Sendable, Equatable {
        package var work: Work
        package var planned: String
        package var reason: Reason
        package var executions: Int
        package init(work: Work, planned: String, reason: Reason, executions: Int) {
            self.work = work; self.planned = planned; self.reason = reason; self.executions = executions
        }
    }

    /// What was computed: a product by a weight of a format, or an attention over `sequence` query rows.
    package enum Work: Hashable, Sendable {
        case product(k: Int, n: Int, m: Int, weight: CapacityClasses.WeightFormat)
        case attention(sequence: Int, heads: Int, headDim: Int)
    }

    /// One line: `executions` runs of `work` on `route`, `seconds` of wall clock in all.
    package struct Tally: Sendable, Equatable {
        package var work: Work
        package var route: String
        package var executions: Int
        package var seconds: Double

        package init(work: Work, route: String, executions: Int, seconds: Double) {
            self.work = work; self.route = route; self.executions = executions; self.seconds = seconds
        }
    }

    private struct Key: Hashable { var work: Work; var route: String }
    private let lines = Mutex<[Key: (executions: Int, seconds: Double)]>([:])
    private struct FallbackKey: Hashable { var work: Work; var planned: String; var reason: Reason }
    private let fallen = Mutex<[FallbackKey: Int]>([:])

    package init() {}

    /// `executions` 0: seconds added to the line of an execution already counted (`Charge`).
    package func add(_ work: Work, route: String, seconds: Double, executions: Int = 1) {
        lines.withLock {
            let kept = $0[Key(work: work, route: route)] ?? (0, 0)
            $0[Key(work: work, route: route)] = (kept.executions + executions, kept.seconds + seconds)
        }
    }

    // MARK: - On the stream

    /// **One execution queued on a stream (`GPUStream`), timed by the engines' own seconds instead of the caller's wall** — which, the
    /// call only encoding, is the encoder's ~0.1 ms. The stream charges it the GPU seconds of the buffers settled while it is open
    /// (`GPUStream.charge`), added at the drain — each buffer's own span, not `gpuEndTime − gpuStartTime`, which also holds its wait
    /// behind the buffer ahead and behind the ANE's rows (another class's seconds, counted there) —, and the ANE's seconds of the rows handed to it (`GPUStream.neuralEngine`), timed on
    /// the ANE's thread; the line gets `max(GPU, ANE, wall)` — the pair's sides run at once and the GPU waits for the ANE's rows before
    /// the next buffer, so the longer side is the class's span, as the map's pair is measured (`bf16ProductStreamed`, its wall). The
    /// execution is counted when the call returns (`finish`), its seconds as they arrive: at that return and at each drain after it, by
    /// difference, so a class split by a drain is still one execution of `max` seconds. Off the stream nothing makes one: the wall as before.
    package final class Charge: Sendable {
        private struct State { var gpu = 0.0, ane = 0.0, wall = 0.0, credited = 0.0, line: (work: Work, route: String)? }
        private let ledger: ProductLedger
        private let state = Mutex<State>(State())

        package init(ledger: ProductLedger) { self.ledger = ledger }

        /// A settled buffer's GPU seconds (the drain).
        package func addGPU(_ seconds: Double) { state.withLock { $0.gpu += seconds } }
        /// The ANE's side of a product, on its thread: the program and, on a refusal, the CPU's rows that stand for it.
        package func addANE(_ seconds: Double) { state.withLock { $0.ane += seconds } }

        /// The call returned: its class and route (known only now: the ANE taken or not), its wall, one execution.
        package func finish(_ work: Work, route: String, wall: Double) {
            state.withLock { $0.line = (work, route); $0.wall = wall }
            ledger.add(work, route: route, seconds: 0)
            credit()
        }

        /// What the execution's span has grown by since the last credit, into its line. Nothing before `finish`.
        package func credit() {
            let owed = state.withLock { s -> (work: Work, route: String, seconds: Double)? in
                guard let line = s.line else { return nil }
                let span = max(s.gpu, s.ane, s.wall)
                defer { s.credited = span }
                return (line.work, line.route, span - s.credited)
            }
            if let owed, owed.seconds > 0 { ledger.add(owed.work, route: owed.route, seconds: owed.seconds, executions: 0) }
        }
    }

    /// One execution of `work`, planned on `planned`, that ran otherwise for `reason`.
    package func fellBack(_ work: Work, planned: String, reason: Reason) {
        fallen.withLock { $0[FallbackKey(work: work, planned: planned, reason: reason), default: 0] += 1 }
    }

    /// The fallbacks, in a fixed order (reason, planned route, then the work's numbers); empty when every product ran as planned.
    package var fallbacks: [Fallback] {
        fallen.withLock { $0.map { Fallback(work: $0.key.work, planned: $0.key.planned, reason: $0.key.reason, executions: $0.value) } }
            .sorted { a, b in
                if a.reason != b.reason { return a.reason < b.reason }
                if a.planned != b.planned { return a.planned < b.planned }
                return String(describing: a.work) < String(describing: b.work)
            }
    }

    /// The lines, in a fixed order (route, then the work's numbers): two renders list alike.
    package var tallies: [Tally] {
        lines.withLock { $0.map { Tally(work: $0.key.work, route: $0.key.route, executions: $0.value.executions, seconds: $0.value.seconds) } }
            .sorted { a, b in
                if a.route != b.route { return a.route < b.route }
                return String(describing: a.work) < String(describing: b.work)
            }
    }

    // MARK: - Arming

    private static let armed = Mutex<ProductLedger?>(nil)

    /// The ledger the render in progress adds to, `nil` outside a denoising.
    package static var current: ProductLedger? { armed.withLock { $0 } }

    /// Runs `body` with `ledger` armed. One render at a time (`Engine.file`), so one ledger at a time.
    package static func recording<T>(into ledger: ProductLedger, _ body: () throws -> T) rethrows -> T {
        armed.withLock { $0 = ledger }
        defer { armed.withLock { $0 = nil } }
        return try body()
    }

    // MARK: - Timing a call

    /// A started clock on a ledger; `finish` adds the line. `nil` when nothing is recording. On a stream that queues (`stream` on), the
    /// execution is charged the stream's seconds (`Charge`), not the wall.
    package struct Stopwatch {
        let ledger: ProductLedger
        let work: Work
        let route: String
        let started: UInt64
        var stream: GPUStream? = nil
        var charge: Charge? = nil
        var previous: Charge? = nil

        package func finish() {
            let wall = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
            guard let charge, let stream else { ledger.add(work, route: route, seconds: wall); return }
            stream.charge = previous
            charge.finish(work, route: route, wall: wall)
        }
    }

    /// The route name of the QK-norm and RoPE before a self-attention (`pass`): never a route the attention took.
    package static let passRoute = "qk-norm+rope"

    /// Starts timing the QK-norm and RoPE before the self-attention of `sequence` query rows, on the CPU or the GPU: its seconds join the
    /// attention's class and no execution (`RenderLoop.compare`) — the plan prices the pass there (`CapacityModel.normRope`), and the
    /// attention's own clock starts after a CPU pass or a waited GPU pass (a GPU pass left in the queue runs inside it).
    package static func pass(sequence: Int, heads: Int, headDim: Int, stream: GPUStream? = nil) -> Stopwatch? {
        attention(sequence: sequence, heads: heads, headDim: headDim, route: passRoute, stream: stream)
    }

    /// Starts timing an attention of `sequence` query rows (none under 64: no class decides it) on `route`. `stream`: the queue's
    /// stream the work is settled on — when it queues, the stream's seconds are charged (`Charge`), not the wall.
    package static func attention(sequence: Int, heads: Int, headDim: Int, route: String, stream: GPUStream? = nil) -> Stopwatch? {
        guard sequence >= 64, let ledger = current else { return nil }
        var watch = Stopwatch(ledger: ledger, work: .attention(sequence: sequence, heads: heads, headDim: headDim),
                              route: route, started: DispatchTime.now().uptimeNanoseconds)
        if let stream, stream.isOn {
            let charge = Charge(ledger: ledger)
            watch.stream = stream; watch.charge = charge; watch.previous = stream.charge
            stream.charge = charge
        }
        return watch
    }
}
