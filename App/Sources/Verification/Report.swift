import Foundation
import Siliconed

// **What a check prints, and how it ends** — the same for every stage, so that a figure reads alike
// from one model to the next and a script can rely on the exit code alone:
//
//     boundary                 vs oracle    median  channel    vs fp64   threshold
//     block 0                  3.214e-06  1.02e-06     1771          —    5.00e-04  ✓
//     model_out                8.002e-04  2.31e-04       12   1.10e-05    1.38e-05  ✓  (witness)
//     ✓ z-image dit 1024 · 11 measures · nearest its threshold: model_out 1.10e-05 of 1.38e-05
//
// Exit codes: 0 passed, 1 failed, 2 skipped (an input is missing: said, never counted as a pass),
// 64 usage. The last line is the verdict and starts with ✓, ✗ or —.

/// How a check ended.
package enum Outcome: Equatable, Sendable {
    case passed
    case failed
    case skipped(String)

    package var exitCode: Int32 {
        switch self {
        case .passed: return 0
        case .failed: return 1
        case .skipped: return 2
        }
    }
}

/// A check cannot run: a golden, a map, a fixture is missing. Thrown, it ends the check with exit 2.
package struct Skip: Error, CustomStringConvertible {
    package let reason: String
    package init(_ reason: String) { self.reason = reason }
    package var description: String { reason }
}

/// A usage error: the message is the whole answer. Exit 64.
package struct UsageError: Error, CustomStringConvertible {
    package let message: String
    package init(_ message: String) { self.message = message }
    package var description: String { message }
}

/// The measures of one run of a stage, printed as they come, and the verdict at the end.
package final class Report {
    package let title: String
    package private(set) var measures: [Measure] = []
    private var tensorHeaderPrinted = false

    /// What the half-precision reference of this run is, as its lines say it: `bf16`, or `W8A8` for a Compact.
    package let half: String

    /// `title`: what the verdict line names — `z-image dit 1024`.
    package init(title: String, half: String = "bf16") { (self.title, self.half) = (title, half) }

    /// Prints the measure's line and keeps it for the verdict. A tensor reads in columns under a header;
    /// a bound reads `value ≤ bound unit`; an identity reads its note.
    package func add(_ m: Measure) {
        measures.append(m)
        let name = m.boundary.count >= 26 ? m.boundary : m.boundary.padding(toLength: 26, withPad: " ", startingAt: 0)
        let mark = m.passed ? "✓" : "✗"
        let note = m.note.isEmpty ? "" : "  " + m.note
        switch m.kind {
        case .tensor:
            if !tensorHeaderPrinted {
                print("boundary".padding(toLength: 26, withPad: " ", startingAt: 0)
                      + "  vs oracle     median  channel     vs fp64   threshold")
                tensorHeaderPrinted = true
            }
            let witness = m.atWitness.map { String(format: "%10.3e", $0.engine) } ?? "         —"
            print(name + String(format: "  %9.3e  %9.2e  %7d  ", m.worst, m.median, m.channel) + witness
                  + String(format: "  %10.2e  ", m.threshold) + mark + note)
        case .bound(let unit, let above):
            let u = unit.isEmpty ? "" : " " + unit
            print(name + "  " + Self.number(m.worst) + (above ? " > " : " ≤ ") + Self.number(m.threshold) + u
                  + "  " + mark + note)
        case .identity:
            print(name + "  " + mark + note)
        }
    }

    /// `3.21e-06`, or `134.70` for a figure that reads better plain (dB, ulp counts).
    static func number(_ v: Double) -> String {
        guard v.isFinite else { return v.isNaN ? "nan" : "∞" }
        return v != 0 && (abs(v) < 1e-2 || abs(v) >= 1e5) ? String(format: "%.2e", v) : String(format: "%.2f", v)
    }

    package func add(_ ms: [Measure]) { ms.forEach(add) }

    /// A boundary against `goldens[name ?? label]` by `Judge.boundary`: absent from the golden, it is not a
    /// boundary the oracle records (nothing added); recorded there but not produced, it fails.
    package func add(_ goldens: Safetensors, _ label: String, _ engine: [Float]?, name: String? = nil,
                     witness: UnsafeBufferPointer<Float>? = nil, half: [Float]? = nil, columns: Int) {
        guard let expected = goldens.float32(name ?? label) else { return }
        guard let engine else { add(Judge.missing(label)); return }
        add(Judge.boundary(label, engine: engine, oracle: expected, witness: witness, half: half, halfName: self.half, columns: columns))
    }

    /// The verdict line, printed last; the outcome it states.
    @discardableResult
    package func verdict() -> Outcome {
        guard !measures.isEmpty else {
            print("✗ \(title) · nothing was measured")
            return .failed
        }
        let failing = measures.filter { !$0.passed }
        if failing.isEmpty {
            // The tensor nearest its threshold, by the figure that judged it (the witness's when there is one).
            let nearest = measures.filter { $0.kind == .tensor && $0.threshold > 0 && $0.judged.isFinite }
                .max { $0.judged / $0.threshold < $1.judged / $1.threshold }
            print("✓ \(title) · \(measures.count) \(measures.count == 1 ? "measure" : "measures")"
                  + (nearest.map { String(format: " · nearest its threshold: %@ %.2e of %.2e", $0.boundary, $0.judged, $0.threshold) } ?? ""))
            return .passed
        }
        print("✗ \(title) · \(failing.count) of \(measures.count) failing: "
              + failing.prefix(4).map(\.boundary).joined(separator: ", ") + (failing.count > 4 ? ", …" : ""))
        return .failed
    }
}
