import Foundation

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
/// The judge's one figure: every stage reads it the same way.
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
