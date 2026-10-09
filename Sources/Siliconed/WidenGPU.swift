import Foundation
import Metal

/// bf16 → fp32 widening **on the GPU**, read directly from the memory-mapped map — a setting
/// (`widen_gpu`, `widen_fused`, `widen_keep`), **off by default**.
///
/// **The engine refuted it on this machine: +20 % at 1024²**, because the widening is queued
/// behind the GEMM, which is the bottleneck. It stays as a branch for a chip where the CPU is the
/// bottleneck instead (fewer P cores, a busier AMX). The kernel itself was measured before being
/// written: it returns 183 GB/s against 15 on a CPU thread — ×12.1 —
/// and the result is **bit-for-bit identical** over 39.3 M values (an earlier probe: 153–172 GB/s on the GPU
/// against 46–62 on the CPU).
///
/// **What was at stake was not the small gain, it was the traffic.** The 5.66 G parameters of an
/// evaluation at 1024² amount to **34 GB read and written by a single CPU thread**, i.e. ~0.94 s — and the
/// measurements precisely accuse this traffic of starving the GPU during its GEMMs. Moving it does not *overlap* anything
/// (overlapping traffic does not create capacity): it gives it to the consumer that has
/// the bandwidth. On paper, ~0.20 s on the GPU instead of 0.94 on the CPU, i.e. ~5.2 s on a
/// 1024² render — and perhaps more if the GEMM recovers a share of the 2.710 → 3.46 TFLOP/s it
/// is missing. Only the engine could say — and it said no, on this machine.
///
/// **The zero-swap contract does not move** (every large allocation is a clean, file-backed page).
/// The kernel's measurement asked the only question that could bring it down:
/// does `makeBuffer(bytesNoCopy:)` accept a `PROT_READ` mapping? Yes. The map's pages
/// stay read-only, hence clean by construction, hence the kernel can only *drop* them —
/// never write them to swap. The "zero bytes of swap" guarantee remains a property of the
/// kernel and not a discipline.
package final class WidenGPU {
    /// `ushort4` read, `float4` written. A bf16 *is* the sixteen high-order bits of an fp32: there is
    /// no rounding, no table, no special case — infinities, NaNs and subnormals pass through
    /// intact. That is what makes it possible to demand **binary** equality with the CPU path instead of a
    /// tolerance.
    ///
    /// **`dequantize4` — an 8-bit map**: `w = float(q) · s`, the same single fp32
    /// multiplication as `Widen.dequantize` on the CPU. int8 → float is exact; fp8 E4M3 goes through
    /// a 256-entry table of fp32 *bit patterns* in `constant` memory (bit patterns, so that the NaN
    /// codes and the subnormals are the CPU table's exactly, whatever the compiler does to float
    /// literals). The scale sits in the same region of the map, `scaleOffset` bytes after the values
    /// (`Artifact.Scale`): one wrapper, never two reads that could come from different places. Its
    /// index is `((j / columns) / rows) · columns + j mod columns` — one formula for a tensor-wide
    /// scale (columns 1, rows = count), per column (rows = K) and per block of rows (GGUF).
    /// **Compiled with fast math off**: a product must be rounded once, exactly like the CPU's. The
    /// one place the GPU still differs: it flushes a subnormal fp32 product to zero. No map holds a
    /// scale that can produce one — the forge refuses a nonzero scale under 2⁻¹²⁶ (int8) or 2⁻¹¹⁷
    /// (fp8), `QuantizedLayouts.checkSubnormal`.
    ///
    /// **`dequantizeQ4_0`…`dequantizeQ6K` — a packed GGUF map**: ggml's blocks read in
    /// place, one thread per output and run of 32 inputs, written transposed into `[K, N]`. Every
    /// product in ggml's formulas is exact in fp32 and every result a multiple of 2⁻²⁴, so neither
    /// an FMA contraction, nor a reassociation, nor the flush of subnormals can move a bit from the
    /// CPU's `Widen.dequantizePacked` (reasoned there, tested at the bit in `KQuantTests`).
    private static var kernels: String {
        let table = Numerics.e4m3.map { String(format: "0x%08xu", $0.bitPattern) }.joined(separator: ",")
        return """
    #include <metal_stdlib>
    using namespace metal;
    kernel void widen4(device const ushort4 *src [[buffer(0)]], device float4 *dst [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
        ushort4 v = src[i];
        dst[i] = float4(as_type<float>(uint(v.x) << 16), as_type<float>(uint(v.y) << 16),
                        as_type<float>(uint(v.z) << 16), as_type<float>(uint(v.w) << 16));
    }

    constant uint E4M3[256] = {\(table)};

    struct Dequantization { uint kind; uint scaleType; uint scaleOffset; uint columns; uint rows; };

    kernel void dequantize4(device const uchar *map [[buffer(0)]], device float4 *dst [[buffer(1)]],
                            constant Dequantization &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        float4 v;
        if (p.kind == 0) {
            char4 q = ((device const char4 *)map)[i];
            v = float4(q);
        } else {
            uchar4 q = ((device const uchar4 *)map)[i];
            v = float4(as_type<float>(E4M3[q.x]), as_type<float>(E4M3[q.y]),
                       as_type<float>(E4M3[q.z]), as_type<float>(E4M3[q.w]));
        }
        if (p.scaleType != 0) {
            device const uchar *scales = map + p.scaleOffset;
            for (uint lane = 0; lane < 4; lane++) {
                uint j = 4 * i + lane;
                uint k = ((j / p.columns) / p.rows) * p.columns + j % p.columns;
                float s;
                if (p.scaleType == 1) s = as_type<float>(uint(((device const ushort *)scales)[k]) << 16);
                else if (p.scaleType == 2) s = float(((device const half *)scales)[k]);
                else s = ((device const float *)scales)[k];
                v[lane] = v[lane] * s;
            }
        }
        dst[i] = v;
    }

    // GGUF packed blocks: one thread per output row r and run of 32 inputs (a legacy
    // block, or a K-quant's sub-block); the map holds `[N][K/b]` blocks, the reserve is `[K, N]`, so
    // neighbouring threads (r, r+1) write neighbouring floats. ggml's arithmetic, every product exact
    // (`Widen.dequantizePacked`): a contraction into an FMA, or any reassociation of the products,
    // cannot change a bit, and the library is built with `MTLMathMode.safe` anyway.
    struct Packed { uint rows; uint perRow; };

    inline float half_at(device const uchar *p) { return float(as_type<half>(ushort(uint(p[0]) | uint(p[1]) << 8))); }

    // Q4_0, Q4_1, Q5_0, Q5_1: block j of row r; low nibbles = values 0–15, high = 16–31, Q5's fifth
    // bit of value i = bit i of the little-endian uint32 qh. Symmetric (no min): q − 8 or q − 16;
    // `hasMin`: d·q + m (Q4_1, Q5_1).
    inline void legacy(device const uchar *map, device float *dst, constant Packed &p, uint2 g,
                       uint size, bool five, bool hasMin) {
        uint r = g.x, j = g.y;
        device const uchar *b = map + (ulong(r) * p.perRow + j) * size;
        float d = half_at(b), m = hasMin ? half_at(b + 2) : 0.0f;
        uint at = hasMin ? 4 : 2;
        uint qh = 0;
        if (five) { qh = uint(b[at]) | uint(b[at + 1]) << 8 | uint(b[at + 2]) << 16 | uint(b[at + 3]) << 24; at += 4; }
        device const uchar *qs = b + at;
        int offset = hasMin ? 0 : (five ? 16 : 8);
        uint c0 = 32 * j;
        for (uint i = 0; i < 16; i++) {
            uint h0 = five ? ((qh >> i) & 1) << 4 : 0, h1 = five ? ((qh >> (i + 16)) & 1) << 4 : 0;
            int q0 = int((qs[i] & 0xF) | h0) - offset, q1 = int((qs[i] >> 4) | h1) - offset;
            if (hasMin) {
                dst[(c0 + i) * p.rows + r] = float(q0) * d + m;
                dst[(c0 + i + 16) * p.rows + r] = float(q1) * d + m;
            } else {
                dst[(c0 + i) * p.rows + r] = float(q0) * d;
                dst[(c0 + i + 16) * p.rows + r] = float(q1) * d;
            }
        }
    }

    kernel void dequantizeQ4_0(device const uchar *map [[buffer(0)]], device float *dst [[buffer(1)]],
                               constant Packed &p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
        legacy(map, dst, p, g, 18, false, false);
    }
    kernel void dequantizeQ4_1(device const uchar *map [[buffer(0)]], device float *dst [[buffer(1)]],
                               constant Packed &p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
        legacy(map, dst, p, g, 20, false, true);
    }
    kernel void dequantizeQ5_0(device const uchar *map [[buffer(0)]], device float *dst [[buffer(1)]],
                               constant Packed &p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
        legacy(map, dst, p, g, 22, true, false);
    }
    kernel void dequantizeQ5_1(device const uchar *map [[buffer(0)]], device float *dst [[buffer(1)]],
                               constant Packed &p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
        legacy(map, dst, p, g, 24, true, true);
    }

    // ggml's `get_scale_min_k4`.
    inline uchar2 scale_min_k4(uint j, device const uchar *s) {
        if (j < 4) return uchar2(s[j] & 63, s[j + 4] & 63);
        return uchar2((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }

    // Q4_K and Q5_K: sub-block t = 8j + s, values 32s…32s+31 of super-block j, nibble s mod 2 of
    // qs[32·(s/2) + l], Q5_K's fifth bit = bit s of qh[l].
    inline void k4k5(device const uchar *map, device float *dst, constant Packed &p, uint2 g, uint size, bool five) {
        uint r = g.x, j = g.y / 8, s = g.y % 8;
        device const uchar *b = map + (ulong(r) * p.perRow + j) * size;
        float d = half_at(b), dmin = half_at(b + 2);
        uchar2 sm = scale_min_k4(s, b + 4);
        float d1 = d * float(sm.x), m1 = dmin * float(sm.y);
        device const uchar *q = b + (five ? 48 : 16) + 32 * (s / 2);
        device const uchar *qh = b + 16;
        uint shift = 4 * (s % 2), c0 = 256 * j + 32 * s;
        for (uint l = 0; l < 32; l++) {
            uint v = (q[l] >> shift) & 0xF;
            if (five) v |= ((qh[l] >> s) & 1) << 4;
            dst[(c0 + l) * p.rows + r] = d1 * float(v) - m1;
        }
    }

    kernel void dequantizeQ4K(device const uchar *map [[buffer(0)]], device float *dst [[buffer(1)]],
                              constant Packed &p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
        k4k5(map, dst, p, g, 144, false);
    }

    kernel void dequantizeQ5K(device const uchar *map [[buffer(0)]], device float *dst [[buffer(1)]],
                              constant Packed &p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
        k4k5(map, dst, p, g, 176, true);
    }

    // Q6_K: sub-block t = 8j + 4h + s, values 128h + 32s + l: nibble s/2 of ql[64h + 32·(s mod 2) + l],
    // bits 2s, 2s+1 of qh[32h + l], scale scales[8h + 2s + l/16] (signed), d at byte 208.
    kernel void dequantizeQ6K(device const uchar *map [[buffer(0)]], device float *dst [[buffer(1)]],
                              constant Packed &p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
        uint r = g.x, j = g.y / 8, h = (g.y % 8) / 4, s = g.y % 4;
        device const uchar *b = map + (ulong(r) * p.perRow + j) * 210;
        float d = half_at(b + 208);
        device const uchar *ql = b + 64 * h + 32 * (s % 2);
        device const uchar *qh = b + 128 + 32 * h;
        device const char *sc = (device const char *)(b + 192 + 8 * h + 2 * s);
        uint nibble = 4 * (s / 2), bits = 2 * s, c0 = 256 * j + 128 * h + 32 * s;
        float d1 = d * float(sc[0]), d2 = d * float(sc[1]);
        for (uint l = 0; l < 32; l++) {
            int q = int(((ql[l] >> nibble) & 0xF) | (((qh[l] >> bits) & 3) << 4)) - 32;
            dst[(c0 + l) * p.rows + r] = (l < 16 ? d1 : d2) * float(q);
        }
    }
    """
    }

    /// A tensor's region in the map, wrapped for the GPU, and what the kernel must do with it.
    package struct Source {
        package let buffer: MTLBuffer
        /// `nil`: a bf16 tensor (`widen4`); otherwise its 8-bit layout (`dequantize4`).
        let dequantization: [UInt32]?
        /// A packed GGUF type (`dequantizeQ4_0`…`dequantizeQ6K`): its type, its `N` rows and its
        /// blocks per row (`K/32` or `K/256`).
        var packed: (kind: Artifact.DType, rows: Int, perRow: Int)? = nil
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let dequantizer: MTLComputePipelineState
    /// One per packed GGUF type.
    private let packedDequantizers: [Artifact.DType: MTLComputePipelineState]

    /// **`SILICONED_WIDEN_KEEP=1` — keep the map's wrappers, and pay 12 GB.**
    ///
    /// This is what the first version did, and it was a fault against criterion no. 4.
    /// Wrapping a region of the map in an `MTLBuffer` makes it enter the GPU's address
    /// space; the pages then stop being **clean** file pages that the kernel can simply drop,
    /// and they count in `phys_footprint`. Memoized, the DiT's ~240 wrappers end up
    /// retaining the entire map.
    ///
    /// **The kernel's measurement asked the wrong question.** It asked whether `makeBuffer(bytesNoCopy:)`
    /// *accepts* a `PROT_READ` mapping — yes — and concluded that "the zero-swap contract does not
    /// move". It did not ask what the acceptance **costs**. A measurement that only looks at a throughput says nothing about what is paid elsewhere.
    ///
    /// The default is therefore to **release the wrapper after use**: 0.067 ms of wiring per weight,
    /// 238 weights per evaluation, seven evaluations — ~112 ms on a 1024² render, against twelve
    /// gigabytes of footprint. The flag keeps the old behaviour so that both points can be
    /// measured, never so that it gets used.
    package static let keepBuffers = EngineSettings.effective.widenKeep

    /// One wrapper per tensor, memoized by name — **only under `SILICONED_WIDEN_KEEP`**.
    ///
    /// **The cost of wiring is counted, not assumed.** `makeBuffer(bytesNoCopy:)` makes the
    /// pages enter the GPU's address space, and it was already measured that doing one per call
    /// cost more than the computation. Here there are ~240 for the whole DiT, created once and
    /// reused by the seven evaluations: the first pays for them, the other six do not. The counter
    /// and the clock are displayed (by the widening check) because if the wiring weighed, the countermeasure was
    /// known — wrap the map in two or three big chunks and play on the offset,
    /// the forge's 16 KiB alignment allows it.
    ///
    /// **Measured, it is useless: 0.20 ms for three wrappers, i.e. 0.067 ms each.** The ~240
    /// of the DiT therefore cost ~16 ms, once, against the 6.6 s that CPU widening takes on a
    /// render. The avenue is closed before having been written — which is the point of a counter.
    private var sources: [String: MTLBuffer] = [:]
    package private(set) var wrapped = 0
    package private(set) var wrapSeconds = 0.0
    package private(set) var widened = 0            // values widened on the GPU
    package private(set) var fellBack = 0           // and those the CPU had to take back

    package enum Failure: Error, CustomStringConvertible {
        case noLibrary(String)
        case commandFailed(String)
        package var description: String {
            switch self {
            case .noLibrary(let why): return "widening kernel: \(why)"
            case .commandFailed(let why): return "GPU widening: \(why)"
            }
        }
    }

    package init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.device = device
        self.queue = queue
        let library: MTLLibrary
        let options = MTLCompileOptions()
        options.mathMode = .safe
        do { library = try device.makeLibrary(source: Self.kernels, options: options) }
        catch { throw Failure.noLibrary("\(error)") }
        guard let function = library.makeFunction(name: "widen4"),
              let dequantize = library.makeFunction(name: "dequantize4") else {
            throw Failure.noLibrary("widen4 or dequantize4 not found")
        }
        pipeline = try device.makeComputePipelineState(function: function)
        dequantizer = try device.makeComputePipelineState(function: dequantize)
        var packed: [Artifact.DType: MTLComputePipelineState] = [:]
        for (kind, name) in [(Artifact.DType.q4_0, "dequantizeQ4_0"), (.q4_1, "dequantizeQ4_1"), (.q5_0, "dequantizeQ5_0"),
                             (.q5_1, "dequantizeQ5_1"), (.q4_k, "dequantizeQ4K"), (.q5_k, "dequantizeQ5K"), (.q6_k, "dequantizeQ6K")] {
            guard let f = library.makeFunction(name: name) else { throw Failure.noLibrary("\(name) not found") }
            packed[kind] = try device.makeComputePipelineState(function: f)
        }
        packedDequantizers = packed
    }

    /// The wrapper of a tensor's region in the map, or `nil` if the GPU cannot
    /// take it — in which case the caller widens on the CPU, which is the previous path.
    ///
    /// The conditions are checked and not assumed: the tensor must be **bf16** or **8-bit** (the map
    /// is not homogeneous — `t_embedder` and `cap_embedder` are fp32 there, and widening them as
    /// bf16 would produce finite numbers, of the right order of magnitude, and wrong), and its count must
    /// be a **multiple of four**, since the kernels work by four.
    package func source(_ name: String, in artifact: Artifact) -> Source? {
        // A streamed tensor lives in a staging buffer that is reused: never wrap it.
        if artifact.stream?.contains(name) == true { return nil }
        // A rotated weight (convrot) has no kernel here: the CPU's `dequantizeRotated`
        // is the path, and the reference.
        guard let tensor = artifact.tensors[name],
              tensor.dtype == .bfloat16 || tensor.isQuantized || tensor.dtype.isPacked,
              tensor.rotation == nil, tensor.count % 4 == 0 else { return nil }
        if let block = tensor.dtype.packedBlock {
            // `[K, N]`, K whole blocks (`Artifact` checked it); indices in 32 bits.
            guard tensor.shape.count == 2, tensor.count < Int(UInt32.max),
                  let buffer = wrap(name, tensor, in: artifact) else { return nil }
            return Source(buffer: buffer, dequantization: nil,
                          packed: (tensor.dtype, tensor.shape[1], tensor.shape[0] / block.values))
        }
        var dequantization: [UInt32]?
        if tensor.isQuantized {
            let scaleType: UInt32, columns: Int, rows: Int
            switch tensor.scale?.dtype {
            case nil: scaleType = 0
            case .bfloat16?: scaleType = 1
            case .float16?: scaleType = 2
            default: scaleType = 3
            }
            switch tensor.scale?.layout ?? .tensor {
            case .tensor: (columns, rows) = (1, tensor.count)
            case .columns(let n): (columns, rows) = (n, tensor.count / n)
            case .blocks(let n, let b): (columns, rows) = (n, b)
            case .rows: return nil              // never in a map
            }
            guard tensor.count < Int(UInt32.max) else { return nil }
            dequantization = [tensor.dtype == .int8 ? 0 : 1, scaleType, UInt32(tensor.scale?.offset ?? 0),
                              UInt32(columns), UInt32(rows)]
        }
        guard let buffer = wrap(name, tensor, in: artifact) else { return nil }
        return Source(buffer: buffer, dequantization: dequantization)
    }

    private func wrap(_ name: String, _ tensor: Artifact.Tensor, in artifact: Artifact) -> MTLBuffer? {
        // `readThrough`'s single buffer is overwritten by the next tensor: never kept.
        let keep = WidenGPU.keepBuffers && !artifact.readsThrough
        if keep, let existing = sources[name] { return existing }
        guard let pointer = artifact.pointer(name) else { return nil }
        // The forge aligns each tensor on a page: the wrapper therefore starts at a page
        // boundary, and its length rounds up to the next page without biting into the next tensor —
        // the padding is already there, it is what `inspect` displays under "alignment".
        let rounded = (tensor.bytes + artifact.page - 1) / artifact.page * artifact.page
        guard tensor.offset + rounded <= artifact.size else { return nil }
        let started = DispatchTime.now().uptimeNanoseconds
        guard let buffer = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: pointer),
                                             length: rounded, options: .storageModeShared,
                                             deallocator: nil) else { return nil }
        wrapSeconds += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        wrapped += 1
        buffer.label = name
        // Without the flag, the wrapper is not retained: the caller uses it, then it dies
        // with its last reference and the kernel recovers clean file pages.
        if keep { sources[name] = buffer }
        return buffer
    }

    /// Encodes the widening into a command buffer that the caller owns — it is the form
    /// that allows chaining it to MPS **without a second submission** (see `GEMM.linear`).
    package func encode(into commands: MTLCommandBuffer, source: Source,
                       destination: MTLBuffer, count: Int) {
        guard let encoder = commands.makeComputeCommandEncoder() else { return }
        if let blocks = source.packed, let pipeline = packedDequantizers[blocks.kind], let block = blocks.kind.packedBlock {
            encoder.label = "dequantize \(blocks.kind.rawValue)"
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(source.buffer, offset: 0, index: 0)
            encoder.setBuffer(destination, offset: 0, index: 1)
            var p = [UInt32(blocks.rows), UInt32(blocks.perRow)]
            encoder.setBytes(&p, length: 8, index: 2)
            // x: the output row (neighbouring threads write neighbouring floats), y: the run of 32 inputs.
            let width = min(32, blocks.rows), height = max(1, min(pipeline.maxTotalThreadsPerThreadgroup, 256) / width)
            encoder.dispatchThreads(MTLSize(width: blocks.rows, height: blocks.perRow * block.values / 32, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1))
            encoder.endEncoding()
            widened += count
            return
        }
        let pipeline = source.dequantization == nil ? self.pipeline : dequantizer
        encoder.label = source.dequantization == nil ? "widen" : "dequantize"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(source.buffer, offset: 0, index: 0)
        encoder.setBuffer(destination, offset: 0, index: 1)
        if var p = source.dequantization {
            encoder.setBytes(&p, length: p.count * 4, index: 2)
        }
        let lanes = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: count / 4, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1))
        encoder.endEncoding()
        widened += count
    }

    /// A submission on its own, for the **separate** path — the one that keeps the instruments.
    ///
    /// Returns the GPU time. **A failing `MTLCommandBuffer` throws nothing**: it returns
    /// `gpuEndTime == gpuStartTime`, hence an infinite throughput, and the device stays upset — all
    /// the following measurements are worth zero. So we check the error *and* a non-zero time.
    @discardableResult
    package func run(source: Source, destination: MTLBuffer, count: Int) throws -> Double {
        guard let commands = queue.makeCommandBuffer() else {
            throw Failure.commandFailed("no command buffer")
        }
        encode(into: commands, source: source, destination: destination, count: count)
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw Failure.commandFailed(error.localizedDescription) }
        let elapsed = commands.gpuEndTime - commands.gpuStartTime
        guard elapsed > 0 else {
            throw Failure.commandFailed("zero GPU time over \(count) values — the device refused")
        }
        return elapsed
    }

    package func countFallback(_ count: Int) { fellBack += count }

}
