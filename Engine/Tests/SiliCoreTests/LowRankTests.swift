import CoreAI
import Foundation
import Testing
@testable import SiliCore

// MARK: - The low-rank path's factors, written into their places

/// `down [in, R] · up [R, out]` in f64 from the places' f32 values.
private func change(_ down: [Float], _ up: [Float], inn: Int, out: Int, rank: Int) -> [Double] {
    var d = [Double](repeating: 0, count: inn * out)
    for k in 0..<inn { for o in 0..<out { for j in 0..<rank { d[k * out + o] += Double(down[k * rank + j]) * Double(up[j * out + o]) } } }
    return d
}

/// The factors as a file stores them, `[r, in]` and `[out, r]`, transposed into `[in, R]` and `[R, out]`, the columns past
/// the rank zeros; in f32 the change is the file's to the last bit of an f32 product.
@Test func theFactorsAreTransposedAndPaddedWithZeros() throws {
    let (r, inn, out, R) = (3, 5, 4, 32)
    let down = (0..<r * inn).map { Float($0 % 7) - 3 }, up = (0..<out * r).map { Float($0 % 5) * 0.25 - 0.5 }
    var dp = [Float](repeating: .nan, count: inn * R), upp = [Float](repeating: .nan, count: R * out)
    try dp.withUnsafeMutableBytes { d in
        try upp.withUnsafeMutableBytes { u in
            try LowRank.write(down: down, up: up, rank: r, inn: inn, out: out, into: d.baseAddress!, u.baseAddress!, rank: R, element: .float32)
        }
    }
    for k in 0..<inn { for j in 0..<R { #expect(dp[k * R + j] == (j < r ? down[j * inn + k] : 0)) } }
    for j in 0..<R { for o in 0..<out { #expect(upp[j * out + o] == (j < r ? up[o * r + j] : 0)) } }
}

/// In f16 each column is balanced by a power of two: down's column at most 2 in L1, the change the same up to f16's
/// rounding of the factors — a column whose L1 is 600 (a factor of 2⁻⁹) as well as one of 10⁻³ (2¹¹).
@Test func inF16EachColumnIsBalancedByAPowerOfTwo() throws {
    let (r, inn, out, R) = (2, 300, 6, 32)
    let down = (0..<r * inn).map { i in i < inn ? Float(i % 2 == 0 ? 2 : -2) : Float(i % 3) * 1e-3 / Float(inn) }
    let up = (0..<out * r).map { Float($0 + 1) * 1e-2 }
    var dp = [Float16](repeating: 0, count: inn * R), upp = [Float16](repeating: 0, count: R * out)
    try dp.withUnsafeMutableBytes { d in
        try upp.withUnsafeMutableBytes { u in
            try LowRank.write(down: down, up: up, rank: r, inn: inn, out: out, into: d.baseAddress!, u.baseAddress!, rank: R, element: .float16)
        }
    }
    for j in 0..<r {
        let l1 = (0..<inn).reduce(0.0) { $0 + abs(Double(dp[$1 * R + j])) }
        #expect(l1 <= 2 && l1 > 1, "column \(j): L1 \(l1)")
    }
    let got = change(dp.map(Float.init), upp.map(Float.init), inn: inn, out: out, rank: R)
    let want = change((0..<inn * R).map { i in let (k, j) = i.quotientAndRemainder(dividingBy: R); return j < r ? down[j * inn + k] : 0 },
                      (0..<R * out).map { i in let (j, o) = i.quotientAndRemainder(dividingBy: out); return j < r ? up[o * r + j] : 0 },
                      inn: inn, out: out, rank: R)
    let e = zip(got, want).reduce(0.0) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }.squareRoot() / want.reduce(0.0) { $0 + $1 * $1 }.squareRoot()
    #expect(e < 2e-3, "relative error \(e)")
}

/// A factor past f16's range once balanced is refused, not rounded to infinity.
@Test func aFactorPastF16IsRefused() {
    var dp = [Float16](repeating: 0, count: 32), upp = [Float16](repeating: 0, count: 32)
    #expect(throws: Failure.self) {
        try dp.withUnsafeMutableBytes { d in
            try upp.withUnsafeMutableBytes { u in
                try LowRank.write(down: [2], up: [1e5], rank: 1, inn: 1, out: 1, into: d.baseAddress!, u.baseAddress!, rank: 32, element: .float16)
            }
        }
    }
}

/// The rank a function is written at: a multiple of 32, none for none.
@Test func aRankIsPaddedTo32() {
    #expect(LowRank.padded(0) == 0 && LowRank.padded(1) == 32 && LowRank.padded(32) == 32 && LowRank.padded(96) == 96 && LowRank.padded(97) == 128)
}
