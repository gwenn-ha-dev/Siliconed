import Foundation
import Metal
import Synchronization

/// **The attention in three stages, by tiles of query rows** — two candidate kernels of the capacity map (`AttentionAssignment.Kernel`
/// `tiles`, `tilesHalf`), neither ever the default: the map times them on each Mac and the solver picks.
///
///     per pass of `headsPerPass` heads and block of `blockRows` query rows (scores [heads, rows, keys] kept under `scoreBytes`):
///       scores  = Q_blk · K_hᵀ                 matmul2d, strided views of the callers' [S, H·Dh] arrays — no copy, no transposition
///       P       = softmax(scale · scores)      one simdgroup a row, sums in fp32, in place (fp32) or written as half (fp16)
///       O_blk   = P · V_h                      matmul2d, written at the head's column offset, row stride H·Dh
///
/// What sets it apart — what an attention written by analogy with the flash kernel or the graph would get wrong:
///
///   - **The products are Metal 4's `matmul2d`** (as `TensorGEMM`), so that where the GPU has matrix accelerators the two products
///     go through them, which `MPSMatrixMultiplication` does not promise; on a GPU without, they run on the shader cores — slower or
///     not, the map says. Only the `slice` API and `tensor_inline` views with strides, the forms every compiler takes.
///   - **The scores are materialized, but one block at a time**: a block of `blockRows × keys` per head (16 MiB by default) stays in the
///     GPU's last-level cache between the three stages; the graph's whole `S² · H` matrix (2 GB at 1 024²) does not. The scale is
///     applied in the softmax, where the products are plain `Q·Kᵀ`.
///   - **fp32 changes nothing but the order of the sums** (`tiles`): the same operands, the same fp32 sums, the softmax in fp32 with
///     `precise::exp`. The fp16 variant (`tilesHalf`) rounds Q, K, V and P to half — once each, nothing else — and sums, scales and
///     normalizes in fp32: operands as the ANE's fp16, sums as P20 requires. Whether a Mac may run it is the request's (Compact and
///     Light, never the Standard), whether it should is the map's.
///   - **Any length**: the tensors carry the real extents, so a block of 100 rows or a key length that is no multiple of 32 needs no
///     padding rows. Cross-attention (`keys ≠ sequence`) and grouped k/v heads are the same code; a causal mask is not (the callers
///     that need one never ask).
package final class TiledAttention {
    package enum Precision: Sendable { case fp32, fp16 }
    package enum Failure: Error, CustomStringConvertible {
        case unsupported(String)
        package var description: String { switch self { case .unsupported(let why): return "tiled attention: \(why)" } }
    }

    /// The scores one pass keeps (fp32, per head and block): sized for the last-level cache of the smallest Mac we map.
    package static let scoreBudget = 16 << 20

    package let heads: Int, kvHeads: Int, sequence: Int, keys: Int, headDim: Int
    /// The heads this instance computes: the first `computed` of `heads` (all of them alone; the GPU's share when the head split composes
    /// with it — the CPU's heads are the last, and their columns are not written here).
    package let computed: Int
    package let precision: Precision
    package let blockRows: Int
    package let headsPerPass: Int

    private let queue: MTLCommandQueue
    private let scoresPipeline, softmaxPipeline, productPipeline: MTLComputePipelineState
    private let halfPipeline: MTLComputePipelineState
    private let scores: MTLBuffer
    private let probabilities: MTLBuffer?
    private let half: (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer)?

    /// Whether the kernel takes this shape: the product writes the head's columns by tiles of 32, and the grouped k/v heads divide the
    /// q heads. The plan offers nothing else (`AttentionAssignment.isRunnable`).
    package static func runs(headDim: Int, heads: Int, kvHeads: Int) -> Bool {
        headDim >= 32 && headDim % 32 == 0 && kvHeads > 0 && heads % kvHeads == 0
    }

    /// The block and the heads of a pass for a length: whole sequence and several heads when it is short, else a block of the budget.
    package static func tiling(heads: Int, sequence: Int, keys: Int, budget: Int = scoreBudget) -> (rows: Int, heads: Int) {
        let rows = max(1, budget / (4 * max(1, keys)))
        if rows >= sequence { return (sequence, max(1, min(heads, rows / max(1, sequence)))) }
        return (max(64, rows / 64 * 64), 1)
    }

    /// The bytes the instance holds (private buffers), by the same function the memory need reads.
    /// The bytes for the `computed` first heads of `heads` (`nil`: all), grouped k/v heads following (`kvHeads` of `heads`).
    package struct Bytes: Equatable {
        /// The scores of a pass (fp32), the probabilities (fp16 only), the half copies of q, k and v (fp16 only).
        package var scores: Int, probabilities: Int, operands: Int
        package var asTotal: Int { scores + probabilities + operands }
    }
    package static func bytes(heads: Int, kvHeads: Int, sequence: Int, keys: Int, headDim: Int, precision: Precision,
                              computed: Int? = nil) -> Bytes {
        let computed = computed ?? heads
        let t = tiling(heads: computed, sequence: sequence, keys: keys)
        let group = max(1, heads / max(1, kvHeads))
        let s = Arena.capacity(for: t.heads * t.rows * keys * 4)
        switch precision {
        case .fp32: return Bytes(scores: s, probabilities: 0, operands: 0)
        case .fp16:
            return Bytes(scores: s, probabilities: Arena.capacity(for: t.heads * t.rows * keys * 2),
                         operands: Arena.capacity(for: sequence * computed * headDim * 2)
                             + 2 * Arena.capacity(for: keys * max(1, computed / group) * headDim * 2))
        }
    }

    package init(device: MTLDevice, queue: MTLCommandQueue, heads: Int, sequence: Int, headDim: Int, keys: Int? = nil, kvHeads: Int? = nil,
                 precision: Precision, computed: Int? = nil, blockRows: Int? = nil, headsPerPass: Int? = nil) throws {
        let keys = keys ?? sequence, kvHeads = kvHeads ?? heads
        let computed = computed ?? heads
        guard computed >= 1, computed <= heads, computed % max(1, heads / max(1, kvHeads)) == 0 else {
            throw Failure.unsupported("\(computed) of \(heads) heads")
        }
        guard device.supportsFamily(.metal4) else { throw Failure.unsupported("no Metal 4") }
        guard TiledAttention.runs(headDim: headDim, heads: heads, kvHeads: kvHeads), sequence > 0, keys > 0 else {
            throw Failure.unsupported("head dimension \(headDim), \(heads)/\(kvHeads) heads")
        }
        let library = try TiledAttention.library(device)
        let suffix = precision == .fp32 ? "f32" : "f16"
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw Failure.unsupported("no function \(name)") }
            return try device.makeComputePipelineState(function: f)
        }
        scoresPipeline = try pipeline("tiles_scores_" + suffix)
        softmaxPipeline = try pipeline("tiles_softmax_" + suffix)
        productPipeline = try pipeline("tiles_product_" + suffix)
        halfPipeline = try pipeline("tiles_to_half")
        let t = TiledAttention.tiling(heads: computed, sequence: sequence, keys: keys)
        self.computed = computed
        self.blockRows = blockRows ?? t.rows
        self.headsPerPass = headsPerPass ?? t.heads
        self.heads = heads; self.kvHeads = kvHeads; self.sequence = sequence; self.keys = keys; self.headDim = headDim
        self.precision = precision; self.queue = queue
        func buffer(_ n: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: n, options: .storageModePrivate) else { throw Failure.unsupported("no \(n) bytes") }
            return b
        }
        scores = try buffer(self.headsPerPass * self.blockRows * keys * 4)
        if precision == .fp16 {
            probabilities = try buffer(self.headsPerPass * self.blockRows * keys * 2)
            let kvComputed = computed / (heads / kvHeads)
            half = (try buffer(sequence * computed * headDim * 2), try buffer(keys * kvComputed * headDim * 2),
                    try buffer(keys * kvComputed * headDim * 2))
        } else { probabilities = nil; half = nil }
    }

    /// **The instance a caller runs an assignment with**, `nil` (said once) when this machine's compiler or the shape refuses it: the
    /// caller then keeps the graph — the plan asked for a kernel the map measured, so a refusal here is a defect worth a line.
    package static func make(_ a: AttentionAssignment, device: MTLDevice, queue: MTLCommandQueue, heads: Int, sequence: Int, headDim: Int,
                             keys: Int? = nil, kvHeads: Int? = nil) -> TiledAttention? {
        do {
            return try TiledAttention(device: device, queue: queue, heads: heads, sequence: sequence, headDim: headDim, keys: keys,
                                      kvHeads: kvHeads, precision: a.tilePrecision)
        } catch {
            Warnings.emit("⚠ tiled attention: falling back to MPSGraph at S=\(sequence) — \(error)")
            return nil
        }
    }

    private struct Params {
        var rows: UInt32, keys: UInt32, qWidth: UInt32, kvWidth: UInt32, headDim: UInt32, group: UInt32
        var headBase: UInt32, firstRow: UInt32, count: UInt32, scale: Float, outWidth: UInt32
    }

    /// **Runs the attention** over `q [S, H·Dh]`, `k` and `v [keys, KVH·Dh]` (fp32, offset 0), writing `out [S, H·Dh]`: one command
    /// buffer. Returns the GPU seconds.
    package func run(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, into out: MTLBuffer) -> Double {
        let clock = ProductLedger.attention(sequence: sequence, heads: heads, headDim: headDim,
                                            route: precision == .fp32 ? "tiles" : "tiles16")
        defer { clock?.finish() }
        let commands = submit(q: q, k: k, v: v, into: out)
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "tiled attention S=\(sequence) H=\(computed)/\(heads) \(precision)")
    }

    /// The same, **committed and not awaited**: the head split computes the CPU's heads meanwhile, then waits on the buffer.
    package func submit(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, into out: MTLBuffer) -> MTLCommandBuffer {
        let commands = queue.checkedCommandBuffer()
        let e = commands.checkedEncoder()
        e.label = precision == .fp32 ? "tiles_sdpa" : "tiles_sdpa_half"
        let outWidth = heads * headDim
        let group = heads / kvHeads
        // The widths the kernels read: the callers' (all heads), or — in half — the compact copy of the computed heads' columns.
        var qWidth = heads * headDim, kvWidth = kvHeads * headDim
        var qs = q, ks = k, vs = v
        if let half {
            let kvComputed = computed / group
            e.setComputePipelineState(halfPipeline)
            // (source, copy, rows, columns converted, source row stride): the first columns of each row, the computed heads'.
            for (from, to, rows, columns, stride) in [(q, half.q, sequence, computed * headDim, qWidth),
                                                      (k, half.k, keys, kvComputed * headDim, kvWidth),
                                                      (v, half.v, keys, kvComputed * headDim, kvWidth)] {
                var shape = SIMD3<UInt32>(UInt32(columns), UInt32(stride), UInt32(columns))
                e.setBuffer(from, offset: 0, index: 0); e.setBuffer(to, offset: 0, index: 1)
                e.setBytes(&shape, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
                e.dispatchThreads(MTLSize(width: columns, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            (qs, ks, vs) = (half.q, half.k, half.v)
            (qWidth, kvWidth) = (computed * headDim, kvComputed * headDim)
        }
        var first = 0
        while first < sequence {
            let rows = min(blockRows, sequence - first)
            var head = 0
            while head < computed {
                let count = min(headsPerPass, computed - head)
                var p = Params(rows: UInt32(rows), keys: UInt32(keys), qWidth: UInt32(qWidth), kvWidth: UInt32(kvWidth),
                               headDim: UInt32(headDim), group: UInt32(heads / kvHeads), headBase: UInt32(head),
                               firstRow: UInt32(first), count: UInt32(count), scale: 1 / Float(headDim).squareRoot(),
                               outWidth: UInt32(outWidth))
                e.setComputePipelineState(scoresPipeline)
                e.setBuffer(qs, offset: 0, index: 0); e.setBuffer(ks, offset: 0, index: 1); e.setBuffer(scores, offset: 0, index: 2)
                e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
                e.dispatchThreadgroups(MTLSize(width: (keys + 31) / 32, height: (rows + 63) / 64, depth: count),
                                       threadsPerThreadgroup: MTLSize(width: scoresPipeline.threadExecutionWidth * 4, height: 1, depth: 1))
                e.setComputePipelineState(softmaxPipeline)
                e.setBuffer(scores, offset: 0, index: 0)
                e.setBuffer(probabilities ?? scores, offset: 0, index: 1)
                e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
                e.dispatchThreadgroups(MTLSize(width: (rows * count + 7) / 8, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                e.setComputePipelineState(productPipeline)
                e.setBuffer(probabilities ?? scores, offset: 0, index: 0); e.setBuffer(vs, offset: 0, index: 1)
                e.setBuffer(out, offset: 0, index: 2)
                e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
                e.dispatchThreadgroups(MTLSize(width: headDim / 32, height: (rows + 63) / 64, depth: count),
                                       threadsPerThreadgroup: MTLSize(width: productPipeline.threadExecutionWidth * 4, height: 1, depth: 1))
                head += count
            }
            first += rows
        }
        e.endEncoding()
        commands.commit()
        return commands
    }

    // MARK: The kernels

    private static let libraries = Mutex<[UInt64: MTLLibrary]>([:])
    private static func library(_ device: MTLDevice) throws -> MTLLibrary {
        if let known = libraries.withLock({ $0[device.registryID] }) { return known }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        options.mathMode = .safe
        let built: MTLLibrary
        do { built = try device.makeLibrary(source: source, options: options) } catch {
            throw Failure.unsupported("the compiler refused the source — \(error.localizedDescription)")
        }
        libraries.withLock { $0[device.registryID] = built }
        return built
    }

    static let source = """
    #include <metal_stdlib>
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp::tensor_ops;

    struct Params {
        uint rows, keys, qWidth, kvWidth, headDim, group, headBase, firstRow, count;
        float scale;
        uint outWidth;
    };

    template <typename Op, typename SA, typename SB, typename SC>
    inline void tile(Op op, SA a, SB b, SC c) {
        auto t = op.template get_destination_cooperative_tensor<SA, SB, float>();
        for (uint16_t i = 0; i < t.get_capacity(); ++i) { if (t.is_valid_element(i)) t[i] = 0; }
        op.run(a, b, t);
        t.store(c);
    }

    // scores[z, r, j] = Σ_d Q[first + r, head·Dh + d] · K[j, kvhead·Dh + d]. Extents innermost first.
    template <typename T>
    inline void scores_body(device T *q, device T *k, device float *s, constant Params &p, uint3 g) {
        const int head = int(p.headBase + g.z), kvHead = head / int(p.group);
        const int rows = int(p.rows), keys = int(p.keys), dim = int(p.headDim);
        tensor<device T, dextents<int32_t, 2>, tensor_inline> A(q + ulong(p.firstRow) * p.qWidth + ulong(head) * dim,
                                                               dextents<int32_t, 2>(dim, rows), array<int32_t, 2>({1, int(p.qWidth)}));
        tensor<device T, dextents<int32_t, 2>, tensor_inline> B(k + ulong(kvHead) * dim,
                                                               dextents<int32_t, 2>(dim, keys), array<int32_t, 2>({1, int(p.kvWidth)}));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(s + ulong(g.z) * rows * keys, dextents<int32_t, 2>(keys, rows));
        constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 32, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(0, x), C.slice(x, y));
    }
    kernel void tiles_scores_f32(device float *q [[buffer(0)]], device float *k [[buffer(1)]], device float *s [[buffer(2)]],
                                 constant Params &p [[buffer(3)]], uint3 g [[threadgroup_position_in_grid]]) { scores_body<float>(q, k, s, p, g); }
    kernel void tiles_scores_f16(device half *q [[buffer(0)]], device half *k [[buffer(1)]], device float *s [[buffer(2)]],
                                 constant Params &p [[buffer(3)]], uint3 g [[threadgroup_position_in_grid]]) { scores_body<half>(q, k, s, p, g); }

    // One simdgroup a row of the pass's [count · rows, keys] scores: softmax(scale · s), sums in fp32.
    kernel void tiles_softmax_f32(device float *s [[buffer(0)]], device float *unused [[buffer(1)]], constant Params &p [[buffer(3)]],
                                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
        const uint row = tg * 8 + sg;
        if (row >= p.rows * p.count) return;
        device float *x = s + ulong(row) * p.keys;
        float m = -INFINITY;
        for (uint c = lane; c < p.keys; c += 32) m = max(m, x[c] * p.scale);
        m = simd_max(m);
        float sum = 0.0f;
        for (uint c = lane; c < p.keys; c += 32) { const float e = precise::exp(x[c] * p.scale - m); x[c] = e; sum += e; }
        const float inverse = 1.0f / simd_sum(sum);
        for (uint c = lane; c < p.keys; c += 32) x[c] *= inverse;
    }
    kernel void tiles_softmax_f16(device const float *s [[buffer(0)]], device half *out [[buffer(1)]], constant Params &p [[buffer(3)]],
                                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
        const uint row = tg * 8 + sg;
        if (row >= p.rows * p.count) return;
        const ulong base = ulong(row) * p.keys;
        float m = -INFINITY;
        for (uint c = lane; c < p.keys; c += 32) m = max(m, s[base + c] * p.scale);
        m = simd_max(m);
        float sum = 0.0f;
        for (uint c = lane; c < p.keys; c += 32) sum += precise::exp(s[base + c] * p.scale - m);
        const float inverse = 1.0f / simd_sum(sum);
        for (uint c = lane; c < p.keys; c += 32) out[base + c] = half(precise::exp(s[base + c] * p.scale - m) * inverse);
    }

    // O[first + r, head·Dh + n] = Σ_j P[z, r, j] · V[j, kvhead·Dh + n], written where the head's columns are.
    template <typename T>
    inline void product_body(device T *pr, device T *v, device float *o, constant Params &p, uint3 g) {
        const int head = int(p.headBase + g.z), kvHead = head / int(p.group);
        const int rows = int(p.rows), keys = int(p.keys), dim = int(p.headDim);
        tensor<device T, dextents<int32_t, 2>, tensor_inline> A(pr + ulong(g.z) * rows * keys, dextents<int32_t, 2>(keys, rows));
        tensor<device T, dextents<int32_t, 2>, tensor_inline> B(v + ulong(kvHead) * dim,
                                                               dextents<int32_t, 2>(dim, keys), array<int32_t, 2>({1, int(p.kvWidth)}));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(o + ulong(p.firstRow) * p.outWidth + ulong(head) * dim,
                                                                   dextents<int32_t, 2>(dim, rows), array<int32_t, 2>({1, int(p.outWidth)}));
        constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 32, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(x, 0), C.slice(x, y));
    }
    kernel void tiles_product_f32(device float *pr [[buffer(0)]], device float *v [[buffer(1)]], device float *o [[buffer(2)]],
                                  constant Params &p [[buffer(3)]], uint3 g [[threadgroup_position_in_grid]]) { product_body<float>(pr, v, o, p, g); }
    kernel void tiles_product_f16(device half *pr [[buffer(0)]], device half *v [[buffer(1)]], device float *o [[buffer(2)]],
                                  constant Params &p [[buffer(3)]], uint3 g [[threadgroup_position_in_grid]]) { product_body<half>(pr, v, o, p, g); }

    // The first `columns` of each row of a source with row stride `stride`, into a compact copy (x: column, y: row).
    kernel void tiles_to_half(device const float *x [[buffer(0)]], device half *y [[buffer(1)]], constant uint3 &shape [[buffer(2)]],
                              uint2 i [[thread_position_in_grid]]) {
        y[ulong(i.y) * shape.z + i.x] = half(x[ulong(i.y) * shape.y + i.x]);
    }
    """
}
