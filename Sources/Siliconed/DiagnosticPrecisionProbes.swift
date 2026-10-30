// The half-precision types the judge rounds by (an ANE output is judged against bf16's own rounding, invariant 3) and the exact
// attention the attention probes are judged against. Split from `DiagnosticProbes.swift`.

import Foundation

extension Diagnostic {
    // ── fp16 and bf16 ───────────────────────────────────────────────────────────────────────

    /// A half-precision type, as the judge rounds it.
    enum Half: String, CaseIterable {
        case fp16, bf16
        /// `x` rounded to the nearest of the type, to nearest even, as an fp32.
        func round(_ x: Float) -> Float { self == .fp16 ? Float(Float16(x)) : widen(narrow(x)) }
        func narrow(_ x: Float) -> UInt16 {
            if self == .fp16 { return Float16(x).bitPattern }
            let bits = x.bitPattern
            return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
        }
        func widen(_ h: UInt16) -> Float { self == .fp16 ? Float(Float16(bitPattern: h)) : Float(bitPattern: UInt32(h) << 16) }
    }

    // ── the attention ───────────────────────────────────────────────────────────────────────

    /// **The exact attention of `rows` queries of every head**, fp64 on the CPU — `softmax(q·kᵀ/√Dh)·v`
    /// as the reference writes it, over all `S` keys. Laid out `[Dh, rows·H]`, the vector of query `i`
    /// at head `h` in column `i·H + h`.
    package static func exactAttention(q: UnsafePointer<Float>, k: UnsafePointer<Float>, v: UnsafePointer<Float>,
                                       sequence: Int, heads: Int, headDim: Int, rows: [Int]) -> [Double] {
        let vectors = rows.count * heads, width = heads * headDim
        let result = Pages(bytes: vectors * headDim * 8)
        nonisolated(unsafe) let (q, k, v, o) = (q, k, v, result.base.assumingMemoryBound(to: Double.self))
        let scale = 1 / Double(headDim).squareRoot()
        DispatchQueue.concurrentPerform(iterations: vectors) { c in
            let row = rows[c / heads], head = c % heads
            withUnsafeTemporaryAllocation(of: Double.self, capacity: sequence + headDim) { buffer in
                let scores = buffer.baseAddress!, sum = scores + sequence
                let query = q + row * width + head * headDim
                var maximum = -Double.infinity
                for j in 0..<sequence {
                    let key = k + j * width + head * headDim
                    var s = 0.0
                    for x in 0..<headDim { s += Double(query[x]) * Double(key[x]) }
                    scores[j] = s * scale
                    maximum = max(maximum, scores[j])
                }
                var total = 0.0
                for x in 0..<headDim { sum[x] = 0 }
                for j in 0..<sequence {
                    let p = Foundation.exp(scores[j] - maximum)
                    total += p
                    let value = v + j * width + head * headDim
                    for x in 0..<headDim { sum[x] += p * Double(value[x]) }
                }
                for x in 0..<headDim { o[x * vectors + c] = sum[x] / total }
            }
        }
        return Array(UnsafeBufferPointer(start: o, count: vectors * headDim))
    }
}
