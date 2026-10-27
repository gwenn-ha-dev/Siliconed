// The measures shared by the diagnostic and every check of the CLI: `channelError`,
// `StageJudgement` and the sampled footprint peak. Split from `Diagnostic.swift`.

import CryptoKit
import Foundation
import IOKit
import IOKit.ps
import Metal

// ── the common measures — shared by the diagnostic and every check of the CLI ──────────────────

/// Relative error channel by channel, each normalized by its own norm — **bounded from below**.
///
/// The denominator went wrong twice, in both directions. Too large at first: a global
/// `‖a−b‖/‖b‖` is a test on channel 85, which carries 94 to 99.7 % of the energy, and it
/// underestimates a fault elsewhere by 373×. Too small afterwards: a channel worth 3.5·10⁻⁵ when the
/// median is 9·10⁻² turns 7.45·10⁻⁹ of absolute error — the noise of a cancellation over 256
/// terms — into 2.1·10⁻⁴ relative, and cries foul.
///
/// Hence the floor at 1 % of the median norm: a channel weaker than that carries less than 10⁻⁴ of
/// the energy, and its relative precision decides nothing. Sensitivity remains whole over four
/// orders of magnitude, ample to cover a massive activation, which is **above** the
/// median and not below it.
///
/// A library function (`package`), shared by the diagnostic and every check, so a user's figure
/// reads exactly like a check's.
package func channelError(_ got: [Float], _ expected: UnsafeBufferPointer<Float>, columns: Int)
    -> (worst: Double, channel: Int, median: Double) {
    relativeErrorByColumn(got, expected, columns: columns)
}

/// The same against an **fp64** truth — the diagnostic's exact, which an fp32 copy would round by
/// 6·10⁻⁸ per value, the size of what it judges.
package func channelError(_ got: [Float], _ expected: UnsafeBufferPointer<Double>, columns: Int)
    -> (worst: Double, channel: Int, median: Double) {
    relativeErrorByColumn(got, expected, columns: columns)
}

private func relativeErrorByColumn<E: BinaryFloatingPoint>(_ got: [Float], _ expected: UnsafeBufferPointer<E>, columns: Int)
    -> (worst: Double, channel: Int, median: Double) {
    let rows = expected.count / columns
    var numerator = [Double](repeating: 0, count: columns)
    var denominator = [Double](repeating: 0, count: columns)
    for row in 0..<rows {
        for column in 0..<columns {
            let i = row * columns + column
            let e = Double(expected[i])
            let d = Double(got[i]) - e
            numerator[column] += d * d
            denominator[column] += e * e
        }
    }
    var norms = denominator.filter { $0 > 0 }.map { $0.squareRoot() }
    guard !norms.isEmpty else { return (0, 0, 0) }
    norms.sort()
    let floor = 0.01 * norms[norms.count / 2]

    // **`r > worst` is FALSE for a `NaN`.** A boundary that was entirely non-finite therefore came
    // out as "0.000e+00", median `nan`, verdict ✓ — in the project's most used comparison
    // function. A non-finite is not a small deviation: it is worth infinity, and it must make any
    // threshold fail.
    var worst = 0.0, channel = 0, live: [Double] = []
    for column in 0..<columns where denominator[column] > 0 {
        var r = numerator[column].squareRoot() / max(denominator[column].squareRoot(), floor)
        if r.isNaN { r = .infinity }
        live.append(r)
        if r > worst { worst = r; channel = column }
    }
    live.sort()
    return (worst, channel, live[live.count / 2])
}

/// **Which error judges a stage: the cumulative one, or its own.**
///
/// A check compares each boundary to the oracle's: the error it reads is **cumulative**, the
/// stage's own plus what it does to the error it receives. On a well-conditioned stage the second
/// term stays the size of the first, and the cumulative error is the right judge. On a badly
/// conditioned one it does not: the Flux encoder's one-head `mid_block` attention (logits up to 435)
/// takes 1.2·10⁻⁵ of fp32 rounding at its input to 6·10⁻⁴ at its output, where torch fp32 on MPS
/// lands at 8.6·10⁻⁴ and torch on CPU at 1.8·10⁻⁴ — the oracle is one draw among the fp32
/// implementations, and a ✗ there says nothing about the engine.
///
/// Hence the rule: the **amplification** is measured — the stage's output error over its input
/// error, both against the oracle — and above `amplificationLimit` the stage is judged on its
/// **local** error: fed with the oracle's input, its output against the oracle's, at the same
/// threshold. A fault inside the stage shows there whole; only the inherited error is set aside, and
/// the stages after it stay judged cumulatively. Never excused: an input identical to the oracle's
/// (the amplification is undefined, and the cumulative error already is the local one), and a
/// non-finite output.
package enum StageJudgement {
    /// Above this measured amplification, a stage is judged on its local error.
    package static let amplificationLimit = 10.0

    /// `output / input`, or `nil` when the input carries no error (or the figures are not finite).
    package static func amplification(inputError: Double, outputError: Double) -> Double? {
        guard inputError > 0, inputError.isFinite, outputError.isFinite else { return nil }
        return outputError / inputError
    }

    /// Whether the stage is judged on its local error rather than the cumulative one.
    package static func judgesLocally(inputError: Double, outputError: Double) -> Bool {
        guard let a = amplification(inputError: inputError, outputError: outputError) else { return false }
        return a > amplificationLimit
    }
}

/// **The peak `phys_footprint`, sampled** every 10 ms on a thread of its own — shared by
/// the checks that report a peak.
package final class FootprintPeak: @unchecked Sendable {
    private let lock = NSLock()
    private var peak = 0, running = true
    package init() {
        Thread.detachNewThread { [self] in
            while true {
                let now = Arena.processFootprint()
                lock.lock()
                peak = max(peak, now)
                let go = running
                lock.unlock()
                if !go { return }
                usleep(10_000)
            }
        }
    }
    package var value: Int { lock.lock(); defer { lock.unlock() }; return max(peak, Arena.processFootprint()) }
    /// Stops the sampling and returns the peak.
    @discardableResult package func stop() -> Int {
        lock.lock(); running = false; lock.unlock()
        return value
    }
}
