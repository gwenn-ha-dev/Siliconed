import Foundation

/// **The bf16 path's weights**: a real tensor of a file as bf16, what a function of the bf16 path reads — a BF16 tensor
/// as it lies, straight from the file into its place, an F32 one rounded to nearest even once (as torch's
/// `from_pretrained(torch_dtype=bfloat16)` converts a parameter the file keeps in f32).
extension SafeTensors {
    /// The tensor `key` into `into` as bf16; its element count, checked against `count` when given.
    /// An F32 tensor passes through `staging` (its bytes at least), or pages of its own when none is given.
    @discardableResult
    public func readBFloat16(_ key: String, count: Int? = nil, into: UnsafeMutableRawPointer, staging: UnsafeMutableRawPointer? = nil) throws -> Int {
        let t = try real(key), n = t.shape.reduce(1, *)
        if let count, count != n { throw Failure("\(key): \(t.shape), \(count) values expected") }
        guard t.dtype == "F32" else { try read(t, into: into); return n }
        let pages = staging == nil ? Pages(bytes: n * 4) : nil
        try withExtendedLifetime(pages) {
            let through = staging ?? pages!.base
            try read(t, into: through)
            let from = through.assumingMemoryBound(to: Float.self), to = into.bindMemory(to: UInt16.self, capacity: n)
            for i in 0..<n { to[i] = BFloat16Bits.rounded(from[i]) }
        }
        return n
    }
}

/// bf16 as the bits of an f32's upper half.
public enum BFloat16Bits {
    /// `x` rounded to nearest even (a NaN stays a NaN).
    public static func rounded(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        guard x.isFinite else { return UInt16(bits >> 16) | (x.isNaN ? 0x40 : 0) }
        return UInt16((bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
    }

    /// The f32 a bf16 holds, exactly.
    public static func widened(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }
}
