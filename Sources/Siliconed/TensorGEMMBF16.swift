import Foundation
import Metal

/// **The products on an activation rounded to bf16** (`gemm_bf16_act`): the kernels of `TensorGEMM` with
/// a `bfloat` first operand, and the pass that rounds it. Compiled only when asked, apart from the
/// default library: a Mac that does not take the path pays nothing for it.
///
///     A fp32 [rows, k] ─▶ round_bf16 (nearest-even, as torch's `.to(bfloat16)`) ─▶ scratch bf16
///     scratch × W       matmul2d  bfloat × bfloat → float     (a bf16 weight, as published)
///                                 bfloat × int8_t → float     (a published int8 weight, its scale after)
///                                 bfloat × float → float      (Q8_0: q · s into threadgroup memory)
///                                 bfloat × half → float       (Q4_K, Q5_K: sc · q per super-block, `KQuantKernels`)
///
/// What sets it apart:
///
///   - **only the operand is rounded**: the buffer the caller holds is not touched — the same activation
///     feeds the LoRA stack, and the queries, keys and values products each read it;
///   - **the scratch is the reserve's tail**, beside the weight (`WeightProducts.roundingPlan`), a chunk of
///     rows at a time, one submission: no memory the models' `MemoryNeed` does not already hold;
///   - **rounded by the engine's own code, not `bfloat(x)`**: the conversion's rounding on a GPU without
///     bf16 in hardware is the compiler's to choose; the integer sequence (add 0x7FFF and the lowest kept
///     bit, shift) is torch's rounding on every Mac, and what the check compares to diffusers' bf16;
///   - **on the stream, the producer rounds instead** (`WeightProducts.RoundedOperand`): where the weight is read in place and the
///     whole operand fits the reserve, the norm, the SwiGLU, the gate, the GELU or the attention (`TiledAttention.run(…rounded:)`, for
///     `to_out`) that makes it also writes it rounded there, with this same integer sequence (`roundingSource`), and the products that
///     read it encode no pass — one rounding for Q, K and V instead of three, none of its own for the others; the same bits (Z-Image
///     1 024²: 0.79 s of GPU per image in 1 808 passes, 1 792 of them gone);
///   - **the tile is the map's** (`MatmulTile`, `gpu/bf16act` and `gpu/int8act`), compiled apart; the kernels here are the
///     standard ones (64 × 32 for a bf16 weight, 64 × 64 for an int8 one), and the Q8_0 one keeps its 64 × 64 (its weight is staged in
///     threadgroup memory as `q · s` in fp32 — exact, but an fp32 operand, which the tensor units do not take at their half-precision
///     rate: the map says what it is worth).
package final class BFloatKernels {
    let round, byColumns, byRows, int8: MTLComputePipelineState
    let blocks: MTLComputePipelineState?
    /// The K-quants' kernels (`KQuantKernels`, a library of their own: a refusal there widens the K-quants and nothing else).
    let kquant: KQuantKernels?

    /// **The rounding itself**, one text for every kernel that writes a bf16 operand (`round_bf16` here, the norms and the SwiGLU
    /// that write the product's operand directly, `ElementwiseGPU`): the same integer sequence wherever it runs, so the operand a
    /// producer writes is the one this pass would have written, bit for bit.
    static let roundingSource = """
    // fp32 → bf16, nearest-even; a NaN stays a (quiet) NaN, an overflow rounds to infinity as torch's.
    inline uint rounded(float f) {
        uint u = as_type<uint>(f);
        if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
        return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    }
    """

    static let source = """
    #include <metal_stdlib>
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp::tensor_ops;

    template <typename Op, typename SA, typename SB, typename SC>
    inline void tile(Op op, SA a, SB b, SC c) {
        auto t = op.template get_destination_cooperative_tensor<SA, SB, float>();
        for (uint16_t i = 0; i < t.get_capacity(); ++i) { if (t.is_valid_element(i)) t[i] = 0; }
        op.run(a, b, t);
        t.store(c);
    }

    \(roundingSource)

    // Two values per thread, one 32-bit store: `pairs` of them (an even count of values).
    kernel void round_bf16(device const float2 *x [[buffer(0)]], device uint *y [[buffer(1)]],
                           constant uint &pairs [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= pairs) return;
        const float2 v = x[i];
        y[i] = rounded(v.x) | (rounded(v.y) << 16);
    }

    kernel void linear_kn(device bfloat *a [[buffer(0)]], device bfloat *w [[buffer(1)]], device float *c [[buffer(2)]],
                          constant uint3 &mkn [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z;
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(N, K));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 32, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(x, 0), C.slice(x, y));
    }

    kernel void linear_nk(device bfloat *a [[buffer(0)]], device bfloat *w [[buffer(1)]], device float *c [[buffer(2)]],
                          constant uint3 &mkn [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z;
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(K, N));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 32, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(0, x), C.slice(x, y));
    }

    kernel void linear_int8(device bfloat *a [[buffer(0)]], device int8_t *w [[buffer(1)]], device float *c [[buffer(2)]],
                            constant uint3 &mkn [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z;
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(N, K));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        constexpr auto d = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 64, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(x, 0), C.slice(x, y));
    }

    // `TensorGEMM.linear_q8` on a bfloat first operand: the same chunks of 64 rows of K.
    kernel void linear_q8(device bfloat *a [[buffer(0)]], device const int8_t *q [[buffer(1)]], device float *c [[buffer(2)]],
                          constant uint3 &mkn [[buffer(3)]], device const float *s [[buffer(4)]],
                          constant uint &B [[buffer(5)]],
                          uint2 g [[threadgroup_position_in_grid]], uint2 tp [[thread_position_in_threadgroup]],
                          uint2 tn [[threads_per_threadgroup]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z, x = g.x * 64, y = g.y * 64;
        threadgroup float wt[64 * 64];
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        tensor<threadgroup float, extents<int32_t, 64, 64>, tensor_inline> W(wt, extents<int32_t, 64, 64>());
        constexpr auto d = matmul2d_descriptor(64, 64, 64, false, false, false,
                                               matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<d, execution_simdgroups<4>> op;
        auto Ay = A.slice(0, y);
        auto acc = op.template get_destination_cooperative_tensor<decltype(Ay), decltype(W), float>();
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) { if (acc.is_valid_element(i)) acc[i] = 0; }
        for (int k0 = 0; k0 < K; k0 += 64) {
            for (uint i = tp.x; i < 64 * 64; i += tn.x) {
                const int row = k0 + int(i / 64), col = x + int(i % 64);
                wt[i] = (row < K && col < N) ? float(q[row * N + col]) * s[(row / int(B)) * N + col] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            auto As = A.slice(k0, y);
            op.run(As, W, acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        acc.store(C.slice(x, y));
    }
    """

    /// `nil` when this macOS's compiler refuses the library or a kernel but the Q8_0 one (kept apart, as
    /// `TensorGEMM`'s: those weights are then widened).
    init?(device: MTLDevice, options: MTLCompileOptions) {
        guard let library = MetalCompile.library(device, Self.source, options, what: "bf16 activation kernels") else { return nil }
        func pipeline(_ name: String) -> MTLComputePipelineState? {
            library.makeFunction(name: name).flatMap { MetalCompile.pipeline(device, $0, what: "bf16 \(name)") }
        }
        guard let round = pipeline("round_bf16"), let byColumns = pipeline("linear_kn"),
              let byRows = pipeline("linear_nk"), let int8 = pipeline("linear_int8") else { return nil }
        self.round = round; self.byColumns = byColumns; self.byRows = byRows; self.int8 = int8
        blocks = pipeline("linear_q8")
        kquant = KQuantKernels(device: device, options: options)
    }
}

extension TensorGEMM {
    /// Where an activation is rounded to: `rows` rows at a time into `buffer` at `offset` (bf16, 16-byte aligned).
    package typealias Rounding = (buffer: MTLBuffer, offset: Int, rows: Int)

    /// Rounds `count` fp32 values (even) at `x` to bf16 at `y`.
    func encodeRound(_ encoder: MTLComputeCommandEncoder, _ kernels: BFloatKernels,
                     from x: MTLBuffer, _ xOffset: Int, to y: MTLBuffer, _ yOffset: Int, count: Int) {
        encoder.setComputePipelineState(kernels.round)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(y, offset: yOffset, index: 1)
        var pairs = UInt32(count / 2)
        encoder.setBytes(&pairs, length: 4, index: 2)
        encoder.dispatchThreads(MTLSize(width: Int(pairs), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// **`C [m, n] = round_bf16(A [m, k]) · W`**, `W` bf16 as `linear(a:w:…)` takes it. The rounded rows go
    /// to `rounding` by chunks, each chunk's product written where its rows go; one submission. Returns
    /// the GPU seconds, the rounding's included. `k` is even. `operandRounded`: the producer of `A` already wrote its rounded rows at
    /// `rounding` (`WeightProducts.roundedOperand`), all `m` of them in one chunk — the pass is not encoded again.
    package func linear(a: MTLBuffer, rounding: Rounding, w: MTLBuffer, wOffset: Int = 0, c: MTLBuffer, m: Int, k: Int, n: Int,
                        weightIsTransposed: Bool, operandRounded: Bool = false) -> Double {
        precondition(!operandRounded || rounding.rows >= m, "an operand rounded by its producer is one chunk")
        let kernels = rounded!
        let tiled = tiles.products(CapacityModel.Kernel(.gpu, .bf16act), rows: m, k: k, n: n)
        let pipeline = (weightIsTransposed ? tiled?.byColumns : tiled?.byRows) ?? (weightIsTransposed ? kernels.byColumns : kernels.byRows)
        let tile = tiled?.tile ?? MatmulTile.standard(for: CapacityModel.Kernel(.gpu, .bf16act))
        let commands = makeCommands()
        let encoder = commands.checkedEncoder()
        for first in stride(from: 0, to: m, by: rounding.rows) {
            let rows = min(rounding.rows, m - first)
            if !operandRounded { encodeRound(encoder, kernels, from: a, first * k * 4, to: rounding.buffer, rounding.offset, count: rows * k) }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(rounding.buffer, offset: rounding.offset, index: 0)
            encoder.setBuffer(w, offset: wOffset, index: 1)
            encoder.setBuffer(c, offset: first * n * 4, index: 2)
            var mkn = SIMD3<UInt32>(UInt32(rows), UInt32(k), UInt32(n))
            encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
            tile.dispatch(encoder, pipeline, rows: rows, columns: n)
        }
        encoder.endEncoding()
        return commands.commitSettled(label: "gemm GPU", what: "matmul2d bfloat × bfloat \(m)×\(k)×\(n)")
    }
}

extension TensorGEMM {
    /// **`submit` on an activation rounded to bf16**: the same product on the first `columns` of `C`,
    /// committed and not awaited. Per chunk of `rounding.rows` rows: the convrot rotation if any (fp32, into
    /// `rotation.scratch`), the rounding of that operand into `rounding`, the product; then the scale's pass
    /// over `C`. Falls back to `nil` when the bf16 library lacks the Q8_0 kernel and `block` asks for it. `weightOffset`: where the
    /// integers start in `w` (`submit`'s).
    func submitRounded(a: MTLBuffer, int8 w: MTLBuffer, weightOffset: Int = 0, scales: MTLBuffer, scalesOffset: Int, perColumn: Bool,
                       block: Int?, rotation: (group: Int, scratch: MTLBuffer, offset: Int, chunk: Int)?,
                       rounding: Rounding, c: MTLBuffer, m: Int, k: Int, n: Int, columns: Int) -> MTLCommandBuffer? {
        let kernels = rounded!
        if block != nil && kernels.blocks == nil { return nil }
        // The int8 kernel at the tile the map chose for `gpu/int8act` (never beside the AMX: its columns may be wider than 64); the
        // Q8_0 one keeps its 64 × 64 tile (its weight is staged in threadgroup memory by that tile).
        let tiled = block == nil ? tiles.products(CapacityModel.Kernel(.gpu, .int8act), rows: m, k: k, n: n) : nil
        let tile = block == nil ? tiled?.tile ?? MatmulTile.standard(for: CapacityModel.Kernel(.gpu, .int8act)) : MatmulTile(64, 64, simdgroups: 4)
        let commands = makeCommands()
        let encoder = commands.checkedEncoder()
        let rows = rotation.map { min($0.chunk, rounding.rows) } ?? rounding.rows
        for first in stride(from: 0, to: m, by: rows) {
            let count = min(rows, m - first)
            var x = a, xOffset = first * k * 4
            if let rotation {
                var groupNorm = SIMD2<UInt32>(UInt32(rotation.group), (1 / Float(rotation.group).squareRoot()).bitPattern)
                encoder.setComputePipelineState(rotateGroups)
                encoder.setBuffer(a, offset: xOffset, index: 0)
                encoder.setBuffer(rotation.scratch, offset: rotation.offset, index: 1)
                encoder.setBytes(&groupNorm, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                encoder.dispatchThreadgroups(MTLSize(width: count * k / rotation.group, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                (x, xOffset) = (rotation.scratch, rotation.offset)
            }
            encodeRound(encoder, kernels, from: x, xOffset, to: rounding.buffer, rounding.offset, count: count * k)
            var mkn = SIMD3<UInt32>(UInt32(count), UInt32(k), UInt32(n))
            let pipeline: MTLComputePipelineState
            if let block, let blocks = kernels.blocks {
                pipeline = blocks
                encoder.setComputePipelineState(blocks)
                encoder.setBuffer(scales, offset: scalesOffset, index: 4)
                var rowsPerScale = UInt32(block)
                encoder.setBytes(&rowsPerScale, length: 4, index: 5)
            } else {
                pipeline = tiled?.int8 ?? kernels.int8
                encoder.setComputePipelineState(pipeline)
            }
            encoder.setBuffer(rounding.buffer, offset: rounding.offset, index: 0)
            encoder.setBuffer(w, offset: weightOffset, index: 1)
            encoder.setBuffer(c, offset: first * n * 4, index: 2)
            encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
            tile.dispatch(encoder, pipeline, rows: count, columns: columns)
        }
        if block == nil {
            encoder.setComputePipelineState(scaleColumns)
            encoder.setBuffer(c, offset: 0, index: 0)
            encoder.setBuffer(scales, offset: scalesOffset, index: 1)
            var nStrideWidth = SIMD3<UInt32>(UInt32(n), perColumn ? 1 : 0, UInt32(columns))
            encoder.setBytes(&nStrideWidth, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
            encoder.dispatchThreads(MTLSize(width: m * columns, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        encoder.endEncoding()
        commands.commit()
        return commands
    }
}
