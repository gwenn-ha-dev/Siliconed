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

    package func add(_ work: Work, route: String, seconds: Double) {
        lines.withLock {
            let kept = $0[Key(work: work, route: route)] ?? (0, 0)
            $0[Key(work: work, route: route)] = (kept.executions + 1, kept.seconds + seconds)
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

    /// A started clock on a ledger; `finish` adds the line. `nil` when nothing is recording.
    package struct Stopwatch {
        let ledger: ProductLedger
        let work: Work
        let route: String
        let started: UInt64

        package func finish() {
            ledger.add(work, route: route, seconds: Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9)
        }
    }

    /// Starts timing an attention of `sequence` query rows (none under 64: no class decides it) on `route`.
    package static func attention(sequence: Int, heads: Int, headDim: Int, route: String) -> Stopwatch? {
        guard sequence >= 64, let ledger = current else { return nil }
        return Stopwatch(ledger: ledger, work: .attention(sequence: sequence, heads: heads, headDim: headDim),
                         route: route, started: DispatchTime.now().uptimeNanoseconds)
    }
}
