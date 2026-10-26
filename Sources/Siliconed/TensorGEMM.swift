import Foundation
import Metal

/// **The product that reads the published weights as they are**: Metal 4's `matmul2d`, fp32
/// activations × bf16 weights → fp32 — a setting (`gemm_tensor`), **off by default**, taken where the
/// capacity map measures it faster than the path it replaces (`CapacityRules`).
///
///     path            the weight before the product           the product
///     default         bf16 → fp32 into the reserve (Widen)    MPS fp32, the AMX on its share of rows
///     gemm_tensor     bf16 copied into the reserve's head     matmul2d  float × bfloat → float
///     gemm_int8       int8 copied, its scales fp32 after it   matmul2d  float × int8_t → float, × s after;
///                                                             convrot: x · R first, on the GPU
///     gemm_q8         Q8_0 copied, its block scales fp32      q · s per 64 rows of K into threadgroup memory,
///                                                             matmul2d  float × float → float, accumulated
///
/// **Nothing is rounded that the default path does not round**: the weights *are* bf16 values (P19,
/// the widening is a shift), the activations stay fp32 — the header's `float × bfloat → float`
/// combination, `relaxed_precision` false — and the sums are fp32. What changes is the order of the
/// sums, hence the last bits, as the AMX's share does. Measured on the M1 Pro before the code: the
/// same worst error as `float × float` (2.3·10⁻⁷ at 1024×3840×10240), against the fp64 of the
/// unrounded activations.
///
/// What sets it apart — what a port of the MPS path by analogy would get wrong:
///
///   - **the speed is the machine's, never assumed**: on an M5 `matmul2d` runs on the GPU's neural
///     accelerators (6.0 TFLOP/s bf16 × bf16 against 2.9 for MPS fp32); on the M1 Pro, on the
///     shader cores, slower than MPS (2.36 against 3.46). Hence a setting the map decides, not a path
///     chosen by chip;
///   - **the weight is copied, not wrapped**: the reserve keeps its fp32 size, the bf16 weight takes
///     its first half (`Artifact.stage`). `MemoryNeed` does not move, a streamed tensor goes the same
///     way, and the map's pages are wired by no one. The copy reads what the widening read and writes
///     half — the map measures its rate beside the widening's;
///   - **only the `slice` API**: the kernel is compiled at run time by the user's macOS, not ours.
///     `static_slice`, `get_mask` and the header's other examples do not compile on every Metal 4
///     compiler; `slice` checks the edges, so any `m` and `n` are taken whole — no padding tier
///     (MPS's 8-to-32-row fault, `GEMM.poisonedRows`, is MPS's);
///   - **bf16: no co-execution**: the AMX reads fp32 weights, which this path no longer makes. The rules
///     compare this path to the better of MPS alone and MPS + AMX, and turn the AMX off when it wins;
///   - **int8: the AMX by columns**: beside the GPU's columns the AMX takes the last ones, widened for
///     it alone while the GPU computes (`submit`, `WeightProducts`) — by rows it would need the whole
///     weight widened, the reserve's size again. The scale is a pass of its own on the GPU's columns
///     (the cooperative tensor's index API is not on every compiler): one more rounding, as the widened
///     weight's own;
///   - **`gemm_bf16_act`: the activation rounded to bf16 just before the product** (`BFloatKernels`, a
///     second library compiled only when asked): `matmul2d` `bfloat × bfloat → float` (and `bfloat × int8_t`,
///     `bfloat × float` for Q8_0), sums in fp32, output fp32. The operand is rounded to nearest-even into a
///     scratch in the reserve's tail, a chunk of rows at a time — one rounding, of the operand and nothing
///     else: the activation the caller holds stays fp32 (the LoRA stack, the residual and the next norm read
///     it). P20 allows it (bf16 between two operations, fp32 accumulations); the map decides it by the
///     measured throughput, and the check judges it against diffusers in bf16;
///   - **the rotation moves to the activation**: `W = R · q · s` along K (`Widen.dequantizeRotated`),
///     `x · W = (x · R) · q · s`, `R = H/√G` — fp32 butterflies on `x`, the integers never touched.
package final class TensorGEMM {
    private let queue: MTLCommandQueue
    func makeCommands() -> MTLCommandBuffer { queue.makeCommandBuffer()! }
    /// `[k, n]` weights (the forge's layout), then `[n, k]` (a published `nn.Linear`).
    private let byColumns, byRows: MTLComputePipelineState
    /// The int8 product, the scale's pass, the rotation of the activation (`linear(a:int8:…)`).
    let int8, scaleColumns, rotateGroups: MTLComputePipelineState
    /// The product on scales per block of rows (GGUF Q8_0), `nil` when this macOS's compiler refuses
    /// a threadgroup operand: those weights are then widened, as before.
    private let blocks: MTLComputePipelineState?
    package var readsBlocks: Bool { blocks != nil }
    /// The products on an activation rounded to bf16 (`gemm_bf16_act`), `nil` unless asked at `init` and
    /// compiled.
    let rounded: BFloatKernels?
    package var roundsActivations: Bool { rounded != nil }
    /// ... and the Q8_0 kernel on it compiled.
    package var roundsBlocks: Bool { rounded?.blocks != nil }

    static let source = """
    #include <metal_stdlib>
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp::tensor_ops;

    // One 64 × 32 tile of C, written once from a cooperative tensor (`run` into a device tensor
    // would accumulate into what C held).
    template <typename Op, typename SA, typename SB, typename SC>
    inline void tile(Op op, SA a, SB b, SC c) {
        auto t = op.template get_destination_cooperative_tensor<SA, SB, float>();
        for (uint16_t i = 0; i < t.get_capacity(); ++i) { if (t.is_valid_element(i)) t[i] = 0; }
        op.run(a, b, t);
        t.store(c);
    }

    // Extents innermost first: A [M, K] row-major is (K, M).
    kernel void linear_kn(device float *a [[buffer(0)]], device bfloat *w [[buffer(1)]], device float *c [[buffer(2)]],
                          constant uint3 &mkn [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z;
        tensor<device float, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(N, K));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 32, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(x, 0), C.slice(x, y));
    }

    kernel void linear_nk(device float *a [[buffer(0)]], device bfloat *w [[buffer(1)]], device float *c [[buffer(2)]],
                          constant uint3 &mkn [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z;
        tensor<device float, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(K, N));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 32, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(0, x), C.slice(x, y));
    }

    // The published int8 `[K, N]` (the forge's layout), a 64 × 64 tile: C = A · Float(q), unscaled.
    kernel void linear_int8(device float *a [[buffer(0)]], device int8_t *w [[buffer(1)]], device float *c [[buffer(2)]],
                            constant uint3 &mkn [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z;
        tensor<device float, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(N, K));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        constexpr auto d = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 64, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(x, 0), C.slice(x, y));
    }

    // GGUF Q8_0 as published: q int8 [K, N], s fp32 [K/B, N] (one per block of B rows and column). Per
    // chunk of 64 rows of K, the threadgroup writes its 64 columns' q · s (exact in fp32: 8 bits × an
    // fp16) into threadgroup memory, and matmul2d accumulates A[64, 64] · W[64, 64] into the
    // cooperative destination. One matmul2d per 32-block was ×5 slower: the scale must not cut the sum.
    // Exact, but on the M1 Pro ×0.29 the widened path at 4096 rows: the threadgroup fp32 operand runs at
    // half MPS's rate, and the fill (each weight 32 times, once per 64-row tile) costs as much again.
    kernel void linear_q8(device float *a [[buffer(0)]], device const int8_t *q [[buffer(1)]], device float *c [[buffer(2)]],
                          constant uint3 &mkn [[buffer(3)]], device const float *s [[buffer(4)]],
                          constant uint &B [[buffer(5)]],
                          uint2 g [[threadgroup_position_in_grid]], uint2 tp [[thread_position_in_threadgroup]],
                          uint2 tn [[threads_per_threadgroup]]) {
        const int M = mkn.x, K = mkn.y, N = mkn.z, x = g.x * 64, y = g.y * 64;
        threadgroup float wt[64 * 64];
        tensor<device float, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
        tensor<threadgroup float, extents<int32_t, 64, 64>, tensor_inline> W(wt, extents<int32_t, 64, 64>());
        // multiply_accumulate: the default `multiply` overwrites the destination at each chunk of K.
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

    // C[r, j] *= s[j · stride] for the first `width` columns of C [M, N]: the scale, per column
    // (stride 1) or for the tensor (0); the columns past `width` are the AMX's, already scaled.
    kernel void scale_columns(device float *c [[buffer(0)]], device const float *s [[buffer(1)]],
                              constant uint3 &nStrideWidth [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        const uint width = nStrideWidth.z, j = i % width;
        c[(i / width) * nStrideWidth.x + j] *= s[j * nStrideWidth.y];
    }

    // y = x · R along each row, R = H/√G block-diagonal by groups of G (a power of 4 up to 256):
    // ComfyUI's convrot, moved from the weight to the activation. One group per threadgroup.
    kernel void rotate_groups(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                              constant uint2 &groupNorm [[buffer(2)]],
                              uint gid [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
        threadgroup float v[256];
        const uint G = groupNorm.x, base = gid * G;
        for (uint i = t; i < G; i += 64) v[i] = x[base + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint span = 1; span < G; span *= 4) {
            if (t < G / 4) {
                const uint j = (t / span) * 4 * span + t % span;
                const float a = v[j], b = v[j + span], c = v[j + 2 * span], d = v[j + 3 * span];
                const float p = a + b, m = a - b, u = c + d, w = c - d;
                v[j] = p + w; v[j + span] = p - w; v[j + 2 * span] = m + u; v[j + 3 * span] = u - m;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float norm = as_type<float>(groupNorm.y);
        for (uint i = t; i < G; i += 64) y[base + i] = v[i] * norm;
    }
    """

    /// `nil` on a device without Metal 4, or when this macOS's compiler refuses the
    /// kernel: the caller keeps the default path.
    package init?(device: MTLDevice, queue: MTLCommandQueue, roundingActivations: Bool = false) {
        guard device.supportsFamily(.metal4) else { return nil }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        options.mathMode = .safe
        guard let library = try? device.makeLibrary(source: TensorGEMM.source, options: options),
              let kn = library.makeFunction(name: "linear_kn"), let nk = library.makeFunction(name: "linear_nk"),
              let byColumns = try? device.makeComputePipelineState(function: kn),
              let byRows = try? device.makeComputePipelineState(function: nk) else { return nil }
        func pipeline(_ name: String) -> MTLComputePipelineState? {
            library.makeFunction(name: name).flatMap { try? device.makeComputePipelineState(function: $0) }
        }
        guard let int8 = pipeline("linear_int8"), let scaleColumns = pipeline("scale_columns"),
              let rotateGroups = pipeline("rotate_groups") else { return nil }
        self.queue = queue; self.byColumns = byColumns; self.byRows = byRows
        self.int8 = int8; self.scaleColumns = scaleColumns; self.rotateGroups = rotateGroups
        blocks = pipeline("linear_q8")
        if roundingActivations {
            guard let kernels = BFloatKernels(device: device, options: options) else { return nil }
            rounded = kernels
        } else {
            rounded = nil
        }
    }

    /// `C [m, n] = A [m, k] · W`, `A` and `C` fp32, `W` bf16 — `[k, n]` when `weightIsTransposed`
    /// (the forge's layout), `[n, k]` otherwise; all row-major from offset 0. Returns the GPU seconds.
    package func linear(a: MTLBuffer, w: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                        weightIsTransposed: Bool) -> Double {
        let pipeline = weightIsTransposed ? byColumns : byRows
        let commands = queue.makeCommandBuffer()!
        let encoder = commands.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(w, offset: 0, index: 1)
        encoder.setBuffer(c, offset: 0, index: 2)
        var mkn = SIMD3<UInt32>(UInt32(m), UInt32(k), UInt32(n))
        encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: (n + 31) / 32, height: (m + 63) / 64, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                                                    height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "matmul2d float × bfloat \(m)×\(k)×\(n)")
    }

    /// **`C [m, n] = A [m, k] · W` with `W = R · (q · s)` read as published** (`gemm_int8`): `q` int8
    /// `[k, n]` at `w`'s offset 0, its scales fp32 at `scales` (`n` of them, or one for the tensor:
    /// `perColumn` false), `R` the convrot rotation of group `rotation` (nil: none). The rotation goes
    /// to the activation — `x · R · q · s` — by chunks of `chunk` rows into `rotated` (fp32, the
    /// reserve's tail), each chunk's product written where its rows go. One submission; returns the
    /// GPU seconds.
    package func linear(a: MTLBuffer, int8 w: MTLBuffer, scales: MTLBuffer, scalesOffset: Int, perColumn: Bool,
                        block: Int? = nil, rotation: (group: Int, scratch: MTLBuffer, offset: Int, chunk: Int)?,
                        rounding: Rounding? = nil, c: MTLBuffer, m: Int, k: Int, n: Int) -> Double {
        let commands = submit(a: a, int8: w, scales: scales, scalesOffset: scalesOffset, perColumn: perColumn,
                              block: block, rotation: rotation, rounding: rounding, c: c, m: m, k: k, n: n, columns: n)
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "matmul2d float × int8 \(m)×\(k)×\(n)")
    }

    /// The same on the first `columns` of `C` only, **committed and not awaited**: the caller computes
    /// the other columns meanwhile (the AMX, `WeightProducts`), then waits on the buffer. `block`: the
    /// scales are per block of that many rows and column (GGUF Q8_0, `[k / block, n]`), applied inside
    /// the product — no rotation then (`Artifact` refuses one there).
    package func submit(a: MTLBuffer, int8 w: MTLBuffer, scales: MTLBuffer, scalesOffset: Int, perColumn: Bool,
                        block: Int? = nil, rotation: (group: Int, scratch: MTLBuffer, offset: Int, chunk: Int)?,
                        rounding: Rounding? = nil, c: MTLBuffer, m: Int, k: Int, n: Int, columns: Int) -> MTLCommandBuffer {
        if let rounding, let rounded = submitRounded(a: a, int8: w, scales: scales, scalesOffset: scalesOffset,
                                                     perColumn: perColumn, block: block, rotation: rotation,
                                                     rounding: rounding, c: c, m: m, k: k, n: n, columns: columns) {
            return rounded
        }
        let commands = queue.makeCommandBuffer()!
        let encoder = commands.makeComputeCommandEncoder()!
        var mkn = SIMD3<UInt32>(UInt32(m), UInt32(k), UInt32(n))
        if let block, let blocks {
            encoder.setComputePipelineState(blocks)
            encoder.setBuffer(a, offset: 0, index: 0)
            encoder.setBuffer(w, offset: 0, index: 1)
            encoder.setBuffer(c, offset: 0, index: 2)
            encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
            encoder.setBuffer(scales, offset: scalesOffset, index: 4)
            var rows = UInt32(block)
            encoder.setBytes(&rows, length: 4, index: 5)
            encoder.dispatchThreadgroups(MTLSize(width: (columns + 63) / 64, height: (m + 63) / 64, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: blocks.threadExecutionWidth * 4,
                                                                        height: 1, depth: 1))
            encoder.endEncoding()
            commands.commit()
            return commands
        }
        func product(_ x: MTLBuffer, _ xOffset: Int, first: Int, rows: Int) {
            encoder.setComputePipelineState(int8)
            encoder.setBuffer(x, offset: xOffset, index: 0)
            encoder.setBuffer(w, offset: 0, index: 1)
            encoder.setBuffer(c, offset: first * n * 4, index: 2)
            mkn.x = UInt32(rows)
            encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: (columns + 63) / 64, height: (rows + 63) / 64, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: int8.threadExecutionWidth * 4,
                                                                        height: 1, depth: 1))
        }
        if let rotation {
            var groupNorm = SIMD2<UInt32>(UInt32(rotation.group), (1 / Float(rotation.group).squareRoot()).bitPattern)
            for first in stride(from: 0, to: m, by: rotation.chunk) {
                let rows = min(rotation.chunk, m - first)
                encoder.setComputePipelineState(rotateGroups)
                encoder.setBuffer(a, offset: first * k * 4, index: 0)
                encoder.setBuffer(rotation.scratch, offset: rotation.offset, index: 1)
                encoder.setBytes(&groupNorm, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                encoder.dispatchThreadgroups(MTLSize(width: rows * k / rotation.group, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                product(rotation.scratch, rotation.offset, first: first, rows: rows)
            }
        } else {
            product(a, 0, first: 0, rows: m)
        }
        encoder.setComputePipelineState(scaleColumns)
        encoder.setBuffer(c, offset: 0, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        var nStrideWidth = SIMD3<UInt32>(UInt32(n), perColumn ? 1 : 0, UInt32(columns))
        encoder.setBytes(&nStrideWidth, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
        encoder.dispatchThreads(MTLSize(width: m * columns, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        return commands
    }
}
