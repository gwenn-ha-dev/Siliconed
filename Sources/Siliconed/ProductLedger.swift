import Foundation
import Synchronization

/// **What a render's denoising computed, by class, with the route each product and attention took and the wall
/// seconds it cost** — the render's side of the capacity map's check (`RenderLoop`). The engine adds one line per
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
///     kind, at the call that chose how the weight is read — the one place that knows both and the route.
///   - **The route is the one taken, not the one planned**: `gpu`, `gpu+amx`, `gpu+ane`, `gpu+ane+amx` for a widened
///     weight (the ANE's refusal, the conductor's declining a short tail: what ran), `gpu/bf16`, `gpu/bf16act`,
///     `gpu/int8`, `gpu+amx/int8`, `gpu/q8`, `gpu+amx/q8` for a weight read as published; `graph`, `flash`, `split`
///     for an attention. The route names of the map's plan, so a line compares with its prediction as it stands.
///   - **Heights stay the render's** (4 117, not 4 096): merging within the map's gap is the comparison's, where the
///     catalogue's heights meet them (`RenderLoop`).
///   - **Armed for the denoising only**, around `DenoisingModule.denoise` and for one render at a time (the engine's
///     queue): the text encoder's and the decoder's products are not the check's, and a check, the diagnostic or a
///     probe that builds a `GEMM` outside a render records nothing and pays a lock to learn it.
///   - **Free in bits and nearly in time**: the ledger reads the clock twice around a product (a few tens of
///     nanoseconds for a product of milliseconds) and changes neither the order of the work nor a route.
///   - **Attentions under 64 rows are no class** (`CapacityClasses`), so they are not recorded.
package final class ProductLedger: Sendable {
    /// What was computed: a product by a weight of a kind, or an attention over `sequence` query rows.
    package enum Work: Hashable, Sendable {
        case product(k: Int, n: Int, m: Int, kind: RouteTable.WeightKind)
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

    package init() {}

    package func add(_ work: Work, route: String, seconds: Double) {
        lines.withLock {
            let kept = $0[Key(work: work, route: route)] ?? (0, 0)
            $0[Key(work: work, route: route)] = (kept.executions + 1, kept.seconds + seconds)
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
