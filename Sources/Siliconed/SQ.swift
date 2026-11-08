import Foundation
import Metal
import Synchronization

/// **SQ, Siliconed's own 4 and 5 bits** (the format: the card of each `gwenn-ha-dev/<model>-Siliconed-SQ4` repository): read as it is stored, decoded to the
/// int8 it is, multiplied by the Compact's int8 route.
///
///     one linear W [N, K], K a multiple of 256
///       row n:   s[n] fp32                                   (the map's `scale`, one per column of `[K, N]`)
///       group g of 16 consecutive k:  one byte  (M << bb) | B     m = M + 1,  b = unit · (B − zero)
///                                     16 codes  q ∈ [0, 2^bits)    v = m · (q − 2^(bits−1)) + b  ∈ [−128, 127]
///       SQ4  bb = 4: m ∈ [1, 16], b = 4 · (B − 8)       SQ5  bb = 5: m ∈ [1, 8], b = 2 · (B − 16)
///       W = (s ⊙ v) · R_K      R_K the regular Hadamard by blocks of 256 (Comfy's convrot, `W8A8.quantize`'s butterflies)
///
///     stored, by row:  [codes: K · bits / 8 bytes][group bytes: K / 16]  (`rowBytes`), the N rows end to end, then the scales
///     codes of a group: 2 · bits bytes, a little-endian bit string, code j at bits j · bits … (4 bits: k even in the low nibble)
///
///     map bytes ─▶ decode  v int8 [K, N], s fp32 [N]   (`decodeTransposed` on the CPU, `SQKernels` on the GPU, into the reserve)
///               ─▶ the int8 convrot product of the Compact: `float × int8` or W8A8, `x · W = s · ((x · R) · v)`
///
/// What sets it apart — what a port by analogy with GGUF's K-quants would get wrong:
///
///   - **the decoded group is an exact int8 under the row's one scale**: there is no scale per group to carry into the sum (a Q4_K
///     scales every 32 inputs), so `(x · R) · v` is one integer (W8A8) or one `float × int8` sum over the whole K, and `s` multiplies
///     after — the Compact's route bit for bit on the same integers. The decoding is lossless: nothing is rounded, nothing widened;
///   - **the offset `b` moves the window by whole steps of `unit`**, so a corner code of a skewed group could leave int8: the
///     quantizer clamps such codes, and the forge checks every group before it writes the map (`firstInvalidRow`) — the decoders
///     then trust the map;
///   - **the rotation is the int8's**: the map says `convrot` of 256, the product moves it to the activation as for a Compact;
///   - **decoded per product, not per map**: the map keeps 4.5 to 5.5 bits a weight; the integers live in the reserve the time of
///     one product (`WeightProducts`), where the Compact's copy would lie.
package enum SQ {
    /// The values of a group, and the rotation every SQ weight carries.
    package static let group = 16
    package static let rotationGroup = 256

    /// One codec: code bits, the group byte's split, the offset's unit and zero.
    package struct Codec: Sendable {
        package let bits, offsetBits, unit, zero: Int
        package var half: Int { 1 << (bits - 1) }
    }

    package static func codec(bits: Int) -> Codec? {
        switch bits {
        case 4: Codec(bits: 4, offsetBits: 4, unit: 4, zero: 8)
        case 5: Codec(bits: 5, offsetBits: 5, unit: 2, zero: 16)
        default: nil
        }
    }

    /// The bytes of one stored row of `k` inputs: its codes, then its group bytes.
    package static func rowBytes(k: Int, bits: Int) -> Int { k * bits / 8 + k / group }

    /// **One group decoded**: its `2 · bits` code bytes at `codes`, its group byte `byte`, the 16 values into `out` (`stride` apart).
    /// Returns `false` when a value leaves int8 (the map is then not one the quantizer wrote).
    @inline(__always)
    package static func decodeGroup(_ codes: UnsafePointer<UInt8>, byte: UInt8, codec c: Codec,
                                    into out: UnsafeMutablePointer<Int8>, stride: Int = 1) -> Bool {
        let m = Int(byte >> UInt8(c.offsetBits)) + 1
        let b = (Int(byte & UInt8((1 << c.offsetBits) - 1)) - c.zero) * c.unit
        var lo: UInt64 = 0, hi: UInt64 = 0
        for i in 0..<8 { lo |= UInt64(codes[i]) << UInt64(8 * i) }
        for i in 8..<(2 * c.bits) { hi |= UInt64(codes[i]) << UInt64(8 * (i - 8)) }
        let mask = UInt64((1 << c.bits) - 1)
        var valid = true
        for j in 0..<group {
            let p = j * c.bits
            let code: UInt64 = p + c.bits <= 64 ? lo >> UInt64(p) : p >= 64 ? hi >> UInt64(p - 64) : (lo >> UInt64(p)) | (hi << UInt64(64 - p))
            let v = m * (Int(code & mask) - c.half) + b
            valid = valid && v >= -128 && v <= 127
            out[j * stride] = Int8(truncatingIfNeeded: v)
        }
        return valid
    }

    /// **A stored row decoded**: `k` values into `out` (`stride` apart). `false`: a value left int8.
    package static func decodeRow(_ row: UnsafeRawPointer, k: Int, bits: Int, into out: UnsafeMutablePointer<Int8>,
                                  stride: Int = 1) -> Bool {
        let c = codec(bits: bits)!
        let codes = row.assumingMemoryBound(to: UInt8.self), bytes = codes + k * bits / 8
        var valid = true
        for g in 0..<(k / group) {
            valid = decodeGroup(codes + g * 2 * bits, byte: bytes[g], codec: c, into: out + g * group * stride, stride: stride) && valid
        }
        return valid
    }

    /// **The stored rows decoded into the map's int8 layout**: `rows` = N stored rows of `columns` = K inputs → `out` `[K, N]`, over
    /// the cores by bands of 64 outputs (each band decoded into its own scratch, then written by input rows).
    package static func decodeTransposed(_ values: UnsafeRawPointer, rows n: Int, columns k: Int, bits: Int,
                                         into out: UnsafeMutablePointer<Int8>) {
        let band = 64, bands = (n + band - 1) / band, stride = rowBytes(k: k, bits: bits)
        nonisolated(unsafe) let (source, destination) = (values, out)
        DispatchQueue.concurrentPerform(iterations: bands) { t in
            let first = t * band, count = min(band, n - first)
            let scratch = UnsafeMutablePointer<Int8>.allocate(capacity: count * k)
            defer { scratch.deallocate() }
            for r in 0..<count { _ = decodeRow(source + (first + r) * stride, k: k, bits: bits, into: scratch + r * k) }
            for i in 0..<k {
                let line = destination + i * n + first
                for r in 0..<count { line[r] = scratch[r * k + i] }
            }
        }
    }

    /// **The first stored row with a value outside int8**, `nil` when every group decodes: what the forge checks before it writes.
    package static func firstInvalidRow(_ values: UnsafeRawPointer, rows n: Int, columns k: Int, bits: Int) -> Int? {
        let stride = rowBytes(k: k, bits: bits)
        let scratch = UnsafeMutablePointer<Int8>.allocate(capacity: k)
        defer { scratch.deallocate() }
        return (0..<n).first { !decodeRow(values + $0 * stride, k: k, bits: bits, into: scratch) }
    }

    /// The logical weight `W [N, K] = (s ⊙ v) · R_K` in fp32, published layout — the tests' and the forge's reference
    /// (`QuantizedTensor.dequantized`): decoded, transposed to `[K, N]`, widened as a convrot (`Widen.dequantizeRotated`), back.
    package static func dequantized(_ values: UnsafeRawPointer, scales: UnsafePointer<Float>, rows n: Int, columns k: Int,
                                    bits: Int) -> [Float] {
        let integers = UnsafeMutablePointer<Int8>.allocate(capacity: k * n)
        defer { integers.deallocate() }
        decodeTransposed(values, rows: n, columns: k, bits: bits, into: integers)
        var kn = [Float](repeating: 0, count: k * n)
        kn.withUnsafeMutableBufferPointer { d in
            Widen.dequantizeRotated(UnsafeRawPointer(integers), rows: k, columns: n, group: rotationGroup, scale: UnsafeRawPointer(scales),
                                    scaleType: .float32, layout: .columns(n), into: d.baseAddress!)
        }
        var nk = [Float](repeating: 0, count: k * n)
        kn.withUnsafeBufferPointer { s in
            nk.withUnsafeMutableBufferPointer { d in Numerics.transpose(s.baseAddress!, rows: k, columns: n, to: d.baseAddress!) }
        }
        return nk
    }
}

/// **The GPU's decoding of an SQ weight into the reserve** (`sq_decode`): one thread per output column and group of 16, the same
/// arithmetic as `SQ.decodeGroup`, the integers written `[K, N]` (consecutive threads, consecutive bytes) and the scales copied —
/// what `Artifact.stageSQ` writes on the CPU, byte for byte. Compiled once per device, apart from the product libraries.
package final class SQKernels {
    let decode: MTLComputePipelineState
    /// The weights decoded on the GPU since the process started (`WeightProducts.prepareSQ`): what a check counts.
    package static let decodes = Mutex<Int>(0)

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // p: K, N, bits. src: the N stored rows (codes, then group bytes); q: int8 [K, N]; scales copied from si to so.
    kernel void sq_decode(device const uchar *src [[buffer(0)]], device char *q [[buffer(1)]], device const float *si [[buffer(2)]],
                          device float *so [[buffer(3)]], constant uint3 &p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
        const uint K = p.x, N = p.y, bits = p.z, n = gid.x, g = gid.y;
        if (n >= N || g >= K / 16) return;
        device const uchar *row = src + size_t(n) * (K * bits / 8 + K / 16);
        const uint byte = row[K * bits / 8 + g];
        const uint ob = bits == 4 ? 4 : 5, middle = 1u << (bits - 1);
        const int m = int(byte >> ob) + 1;
        const int b = (int(byte & ((1u << ob) - 1)) - int(middle)) * (bits == 4 ? 4 : 2);
        device const uchar *c = row + g * 2 * bits;
        ulong lo = 0; uint hi = 0;
        for (uint i = 0; i < 8; ++i) lo |= ulong(c[i]) << (8 * i);
        if (bits == 5) hi = uint(c[8]) | (uint(c[9]) << 8);
        const uint mask = (1u << bits) - 1;
        device char *out = q + size_t(g * 16) * N + n;
        for (uint j = 0; j < 16; ++j) {
            const uint pos = j * bits;
            const uint code = pos + bits <= 64 ? uint(lo >> pos) : pos >= 64 ? (hi >> (pos - 64)) : (uint(lo >> pos) | (hi << (64 - pos)));
            out[size_t(j) * N] = char(m * (int(code & mask) - int(middle)) + b);
        }
        if (g == 0) so[n] = si[n];
    }
    """

    init?(device: MTLDevice) {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        guard let library = MetalCompile.library(device, Self.source, options, what: "SQ decoding"),
              let function = library.makeFunction(name: "sq_decode"),
              let decode = MetalCompile.pipeline(device, function, what: "SQ sq_decode") else { return nil }
        self.decode = decode
    }

    /// The kernels of `device`, compiled on first use (`nil` when its compiler refuses them: the CPU decodes).
    package static func of(_ device: MTLDevice) -> SQKernels? {
        lock.lock(); defer { lock.unlock() }
        if let (d, k) = compiled, d === device { return k }
        let k = SQKernels(device: device)
        compiled = (device, k)
        return k
    }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var compiled: (MTLDevice, SQKernels?)?

    /// **Encodes the decoding** of the `[k, n]` SQ weight whose stored rows start at `source`'s `offset` (its scales at `scalesOffset`)
    /// into `w`: the integers at 0, the scales at `scalesAt`. Committed on `commands`' queue, not awaited.
    package func encode(_ commands: MTLCommandBuffer, source: MTLBuffer, offset: Int, scalesOffset: Int, into w: MTLBuffer,
                        scalesAt: Int, k: Int, n: Int, bits: Int) {
        let encoder = commands.checkedEncoder()
        encoder.setComputePipelineState(decode)
        encoder.setBuffer(source, offset: offset, index: 0)
        encoder.setBuffer(w, offset: 0, index: 1)
        encoder.setBuffer(source, offset: scalesOffset, index: 2)
        encoder.setBuffer(w, offset: scalesAt, index: 3)
        var p = SIMD3<UInt32>(UInt32(k), UInt32(n), UInt32(bits))
        encoder.setBytes(&p, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 4)
        encoder.dispatchThreads(MTLSize(width: n, height: k / SQ.group, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
        encoder.endEncoding()
    }
}
