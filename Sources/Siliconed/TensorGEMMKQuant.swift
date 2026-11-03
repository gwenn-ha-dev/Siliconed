import Foundation
import Metal

/// **A K-quant weight read as published, beside an activation rounded to bf16** (`gpu/q4kact`): GGUF Q4_K and Q5_K, the Light's
/// blocks, multiplied on the GPU's tensor units without being widened — exactly, at the bit of the published values.
///
///     a Q4_K/Q5_K super-block of 256 inputs of one output j:  w = d·sc_s·q − dmin·m_s     (s: its eight sub-blocks of 32)
///                                                             d, dmin fp16; sc, m 6 bits; q 4 bits (Q5_K: 5)
///
///     blocks [N][K/256], as stored ─▶ unpack    Iq = sc·q  half [K, N]     ≤ 63·31 = 1953 < 2048: an integer fp16 holds exactly
///                                               d          fp32 [K/256, N]
///                                               Mn = dmin·m fp32 [K/32, N] 11 bits × 6: exact
///     A fp32 [rows, K]              ─▶ roundsums a_r = bf16(A) (nearest-even, as torch), S[i, s] = Σ_{k∈s} a_r[i, k] in fp32
///                                     mins      C = −S · Mn                        matmul2d float × float, fp32 sums
///                                     main      C += Σ_B d[B, j] · (a_r · Iq)_B    matmul2d bfloat × half per super-block of 256,
///                                                                                  its fp32 partial sums scaled per column in fp32
///
/// so that `C = a_r · W` with `W` the published values: `Σ_k a_r·w = Σ_B d_B·Σ_{k∈B} a_r·(sc·q) − Σ_s (dmin·m)_s·S_s`, every product
/// of two operands exact in fp32 (bf16 × an 11-bit integer: 8 + 11 bits), every sum fp32 (P20). The widened route computes the same
/// sum from `d·sc·q − dmin·m` rounded once to fp32 (`Widen.superBlockQ4K`); this one never rounds the weight at all.
///
/// What sets it apart — what a port by analogy with the 8-bit kernels would get wrong:
///
///   - **neither the bf16 nor the int8 kernel can read it exactly**: a Q4_K value has up to 21 significant bits (`d·sc·q`, then the
///     `dmin·m` subtracted), bf16 holds 8 — on the four Lights' tensors 0.3 to 18 % of the values are bf16 values, the rest lose
///     ~1.3·10⁻³ of themselves (median): a widening to bf16 would compute below the published weight, which P20 forbids. The integer
///     `q` is exact in int8, but its scale changes every 32 rows of K and column: one `matmul2d` per sub-block, scaled in between,
///     ran at a sixth of the int8 kernel's rate on an M5 Pro (4.1 against 24 TFLOP/s). Folding the sub-block's 6-bit scale into the
///     integer (`sc·q`, 10 bits — an fp16 holds it, an int8 does not) leaves one scale per 256 rows, a sixth as many cuts;
///   - **the mins are their own small product** (`dmin·m` against the sub-blocks' sums of the rounded activation, K/32 deep): inside
///     the main loop they cost a third of its time (reads per element), apart a tenth;
///   - **Q6_K is not read here**: `sc·(q − 32)` reaches 4096 with odd values past 2048 (2.6 % of a real tensor's), which no fp16
///     holds; nor the legacy Q4_0…Q5_1 (the Lights carry none). Those tensors take the widened route the plan gives without this one
///     (`CapacityPlan.assignWidened`);
///   - **the cooperative tensor's index** (`get_multidimensional_index`) gives each element's column for its scale: a compiler that
///     refuses it leaves the library `nil`, and the K-quants widened as before.
package final class KQuantKernels {
    let unpack, roundSums, mins, product: MTLComputePipelineState

    static let source = """
    #include <metal_stdlib>
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp::tensor_ops;

    // fp32 → bf16, nearest-even (`BFloatKernels`' rounding, torch's).
    inline uint rounded(float f) {
        uint u = as_type<uint>(f);
        if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
        return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    }

    // ggml's get_scale_min_k4: sub-block j's 6-bit scale and min.
    inline uchar2 scale_min(uint j, device const uchar *s) {
        if (j < 4) return uchar2(s[j] & 63, s[j + 4] & 63);
        return uchar2((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }

    // One thread per output column j (fastest: the writes coalesce) and sub-block s of 32 inputs.
    kernel void unpack_kquant(device const uchar *blocks [[buffer(0)]], device half *iq [[buffer(1)]], device float *d [[buffer(2)]],
                              device float *mins [[buffer(3)]], constant uint3 &knf [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
        const uint K = knf.x, N = knf.y, five = knf.z, j = gid.x, s = gid.y;
        if (j >= N || s >= K / 32) return;
        const uint B = s / 8, sub = s % 8;
        device const uchar *b = blocks + (j * (K / 256) + B) * (five ? 176 : 144);
        const float dd = float(as_type<half>(ushort(b[0] | (b[1] << 8))));
        const float dm = float(as_type<half>(ushort(b[2] | (b[3] << 8))));
        const uchar2 sm = scale_min(sub, b + 4);
        if (sub == 0) d[B * N + j] = dd;
        mins[s * N + j] = dm * float(sm.y);
        // Four groups of 64 values read from the same 32 bytes: the low nibbles sub-block 2g, the high ones 2g + 1; Q5_K's fifth
        // bit of sub-block s is bit s of qh[l].
        device const uchar *qs = b + (five ? 48 : 16) + 32 * (sub / 2);
        device const uchar *qh = b + 16;
        const float sc = float(sm.x);
        for (uint l = 0; l < 32; ++l) {
            uint q = (sub & 1) ? (qs[l] >> 4) : (qs[l] & 0xF);
            if (five) q |= ((qh[l] >> sub) & 1) << 4;
            iq[(B * 256 + 32 * sub + l) * N + j] = half(sc * float(q));
        }
    }

    // The rounded activation and each sub-block's sum of it, one thread per row and sub-block.
    kernel void round_sums(device const float *x [[buffer(0)]], device ushort *y [[buffer(1)]], device float *S [[buffer(2)]],
                           constant uint2 &rk [[buffer(3)]], uint i [[thread_position_in_grid]]) {
        const uint rows = rk.x, K = rk.y, subs = K / 32;
        if (i >= rows * subs) return;
        device const float *p = x + (i / subs) * K + (i % subs) * 32;
        device ushort *q = y + (i / subs) * K + (i % subs) * 32;
        float t = 0;
        for (uint l = 0; l < 32; ++l) { const uint r = rounded(p[l]); q[l] = ushort(r); t += as_type<float>(r << 16); }
        S[i] = t;
    }

    // C = −S · Mn.
    kernel void minus_mins(device float *S [[buffer(0)]], device float *Mn [[buffer(1)]], device float *c [[buffer(2)]],
                           constant uint3 &mkn [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]]) {
        const int M = mkn.x, K = mkn.y / 32, N = mkn.z;
        tensor<device float, dextents<int32_t, 2>, tensor_inline> A(S, dextents<int32_t, 2>(K, M));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> B(Mn, dextents<int32_t, 2>(N, K));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        constexpr auto d = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 64, y = g.y * 64;
        auto As = A.slice(0, y); auto Bs = B.slice(x, 0);
        auto t = op.template get_destination_cooperative_tensor<decltype(As), decltype(Bs), float>();
        for (uint16_t i = 0; i < t.get_capacity(); ++i) { if (t.is_valid_element(i)) t[i] = 0; }
        op.run(As, Bs, t);
        for (uint16_t i = 0; i < t.get_capacity(); ++i) { if (t.is_valid_element(i)) t[i] = -t[i]; }
        t.store(C.slice(x, y));
    }

    // C += Σ_B d[B, j] · (a_r · Iq)_B, a 64 × 64 tile of C, one matmul2d per super-block of 256 rows of K.
    kernel void linear_kquant(device bfloat *a [[buffer(0)]], device half *w [[buffer(1)]], device float *c [[buffer(2)]],
                              constant uint3 &mkn [[buffer(3)]], device const float *d [[buffer(4)]],
                              uint2 g [[threadgroup_position_in_grid]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z;
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device half, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(N, K));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        constexpr auto desc = matmul2d_descriptor(64, 64, 256, false, false, false);
        matmul2d<desc, execution_simdgroups<4>> op;
        const int x = g.x * 64, y = g.y * 64;
        auto A0 = A.slice(0, y); auto B0 = B.slice(x, 0); auto Cs = C.slice(x, y);
        auto acc = op.template get_destination_cooperative_tensor<decltype(A0), decltype(B0), float>();
        auto part = op.template get_destination_cooperative_tensor<decltype(A0), decltype(B0), float>();
        acc.load(Cs);
        for (int b = 0; b < K / 256; ++b) {
            auto As = A.slice(b * 256, y); auto Bs = B.slice(x, b * 256);
            op.run(As, Bs, part);
            #pragma unroll
            for (uint16_t i = 0; i < part.get_capacity(); ++i) {
                if (part.is_valid_element(i)) {
                    const int col = x + part.get_multidimensional_index(i)[0];
                    if (col < N) acc[i] = fma(part[i], d[b * N + col], acc[i]);
                }
            }
        }
        acc.store(Cs);
    }
    """

    /// `nil` when this macOS's compiler refuses the library or a kernel (said once, `MetalCompile`): the K-quants are then widened.
    init?(device: MTLDevice, options: MTLCompileOptions) {
        guard let library = MetalCompile.library(device, Self.source, options, what: "K-quant kernels") else { return nil }
        func pipeline(_ name: String) -> MTLComputePipelineState? {
            library.makeFunction(name: name).flatMap { MetalCompile.pipeline(device, $0, what: "K-quant \(name)") }
        }
        guard let unpack = pipeline("unpack_kquant"), let roundSums = pipeline("round_sums"), let mins = pipeline("minus_mins"),
              let product = pipeline("linear_kquant") else { return nil }
        self.unpack = unpack; self.roundSums = roundSums; self.mins = mins; self.product = product
    }

    /// The K-quant types the kernels read, and their block's bytes: Q4_K and Q5_K.
    package static func reads(_ dtype: Artifact.DType) -> Bool { dtype == .q4_k || dtype == .q5_k }
}

extension TensorGEMM {
    /// The K-quant kernels compiled (with the rounding's library: the route rounds its activation).
    package var readsKQuants: Bool { rounded?.kquant != nil }

    /// **Where a K-quant weight and its operands lie in the reserve** (bytes, page-aligned), for `linear(a:kquant:…)`: the blocks as
    /// stored at 0, `Iq` at `integers`, `d` at `scales`, `Mn` at `mins`, then by chunks of `rows` rows the sub-blocks' sums at `sums`
    /// and the rounded activation at `rounded`. `five`: Q5_K's blocks.
    package struct KQuantLayout: Equatable {
        package let integers, scales, mins, sums, rounded, rows: Int
        package let five: Bool
    }

    /// **The layout of a `[k, n]` K-quant weight in a reserve of `capacity` bytes**, or `nil` when `k` is not a whole number of
    /// super-blocks or fewer than 64 rows would fit (the reserve is the model's, sized for the widened fp32 weight, never grown here).
    package static func kquantLayout(k: Int, n: Int, capacity: Int, five: Bool) -> KQuantLayout? {
        guard k % 256 == 0, k > 0, n > 0 else { return nil }
        func page(_ bytes: Int) -> Int { (bytes + 16383) / 16384 * 16384 }
        let blocks = n * (k / 256) * (five ? 176 : 144)
        let integers = page(blocks), scales = page(integers + 2 * k * n)
        let mins = page(scales + 4 * (k / 256) * n), sums = page(mins + 4 * (k / 32) * n)
        // Per row: its K/32 sums (fp32) and its rounded values (bf16); the second region starts on its own page.
        var rows = max(0, capacity - sums - 16384) / (4 * (k / 32) + 2 * k) / 64 * 64
        while rows >= 64 && page(sums + 4 * (k / 32) * rows) + 2 * k * rows > capacity { rows -= 64 }
        guard rows >= 64 else { return nil }
        return KQuantLayout(integers: integers, scales: scales, mins: mins, sums: sums, rounded: page(sums + 4 * (k / 32) * rows),
                            rows: rows, five: five)
    }

    /// **`C [m, n] = round_bf16(A [m, k]) · W`, `W` a K-quant weight's blocks at `w`'s offset 0** (`[n][k/256]`, as the map stores
    /// them), laid out by `layout`: the unpacking once, then per chunk of rows the rounding and sums, the mins, the product. One
    /// submission; returns the GPU seconds.
    package func linear(a: MTLBuffer, kquant w: MTLBuffer, layout: KQuantLayout, c: MTLBuffer, m: Int, k: Int, n: Int) -> Double {
        let kernels = rounded!.kquant!
        let commands = makeCommands()
        let encoder = commands.checkedEncoder()
        encoder.setComputePipelineState(kernels.unpack)
        encoder.setBuffer(w, offset: 0, index: 0)
        encoder.setBuffer(w, offset: layout.integers, index: 1)
        encoder.setBuffer(w, offset: layout.scales, index: 2)
        encoder.setBuffer(w, offset: layout.mins, index: 3)
        var knf = SIMD3<UInt32>(UInt32(k), UInt32(n), layout.five ? 1 : 0)
        encoder.setBytes(&knf, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 4)
        encoder.dispatchThreads(MTLSize(width: n, height: k / 32, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 4, depth: 1))
        for first in stride(from: 0, to: m, by: layout.rows) {
            let rows = min(layout.rows, m - first)
            encoder.setComputePipelineState(kernels.roundSums)
            encoder.setBuffer(a, offset: first * k * 4, index: 0)
            encoder.setBuffer(w, offset: layout.rounded, index: 1)
            encoder.setBuffer(w, offset: layout.sums, index: 2)
            var rk = SIMD2<UInt32>(UInt32(rows), UInt32(k))
            encoder.setBytes(&rk, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
            encoder.dispatchThreads(MTLSize(width: rows * k / 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            var mkn = SIMD3<UInt32>(UInt32(rows), UInt32(k), UInt32(n))
            let grid = MTLSize(width: (n + 63) / 64, height: (rows + 63) / 64, depth: 1)
            encoder.setComputePipelineState(kernels.mins)
            encoder.setBuffer(w, offset: layout.sums, index: 0)
            encoder.setBuffer(w, offset: layout.mins, index: 1)
            encoder.setBuffer(c, offset: first * n * 4, index: 2)
            encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: kernels.mins.threadExecutionWidth * 4, height: 1, depth: 1))
            encoder.setComputePipelineState(kernels.product)
            encoder.setBuffer(w, offset: layout.rounded, index: 0)
            encoder.setBuffer(w, offset: layout.integers, index: 1)
            encoder.setBuffer(c, offset: first * n * 4, index: 2)
            encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
            encoder.setBuffer(w, offset: layout.scales, index: 4)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: kernels.product.threadExecutionWidth * 4, height: 1, depth: 1))
        }
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "matmul2d bfloat × half (K-quant) \(m)×\(k)×\(n)")
    }
}
