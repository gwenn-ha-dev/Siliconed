import Foundation
import Metal
import Synchronization

/// **The QK-norm and the RoPE before a self-attention, on the GPU** — a route beside the CPU's (`Ops.rmsNormRope`), which the map
/// departs: the CPU's pass leaves the GPU idle before every self-attention of the five denoisers, and the attention's call then pays the
/// GPU's wake (`CapacityModel.wake`); here the pass is a kernel queued right after the products that wrote Q and K, the GPU never idle. On
/// one Mac it is a candidate, never a verdict: the map times both routes (`Context.normRope`) and the solver takes the one a render pays
/// least for, wake included (`CapacitySolver`).
///
///     per vector (s, h), one simdgroup — Dh/2 pairs over 32 lanes:
///       ss    = Σ x²                                       fp32, each lane its pairs, then `simd_sum`
///       x     = x · (1 / √(ss/Dh + ε))  then  · w[d]       two roundings, as `vDSP_vsmul` then `vDSP_vmul`
///       pair  = (x[2j], x[2j+1]) | (x[j], x[j+Dh/2])         interleaved | half-split, the table's own layout
///       x'    = (a·cos − b·sin,  a·sin + b·cos)            each product rounded, then the sum: no contraction (`fp contract(off)`)
///
/// **Not the CPU's bits, and why**: the RoPE is (the same products and sums, the same roundings), the norm's sum of squares is not —
/// `vDSP_svesq`'s order is its own (four accumulators of four lanes up to 32 values, then neither 4 nor 8 accumulators by any tree at
/// 64 and 128: searched, not found), and a scale one ulp apart moves every value of the vector by as much. The route is therefore
/// judged against the oracle (the families' checks under `SILICONED_PLAN=normrope`), never by the md5 of the CPU's route.
///
/// What sets the families apart — what a port by analogy would get wrong (read in each family's self-attention):
///
///     family    norm (ε)                 heads q / k          RoPE table                  layout        the CPU's pass, before
///     Z-Image   RMS 1e-5, not the block's 30 / 30               3D, `[S, Dh/2, (cos,sin)]`   interleaved   norm q, norm k, rope: two passes
///     Qwen 2.1  RMS config.eps (1e-6)    32 / 32, K in the KV  3D, rows from `first`       interleaved   one pass (`rmsNormRope`)
///     Anima     RMS 1e-5                 16 / 16               3D Cosmos, scaled           interleaved   one pass; the cross's q normed alone
///     Krea 2    RMS config.eps, 1+w      48 / 12 (GQA)         3D                           interleaved   norm q, norm k, rope: two passes
///     MOSS      RMS 1e-6                 24 / 24 (text too)    4 axes, `[S, sin | cos]`     half-split    norm q, norm k, rope: two passes
///
/// Dh = 128 everywhere. The CPU's pass is one pass per row in all five now (`Ops.rmsNormRope`, `halfSplit` for MOSS): the same bits as the
/// two passes (`QKNormRopeTests`), and the pass the map's CPU line times; the two remain where a boundary is recorded between the norm and
/// the rotation (Z-Image's `layer0_q_normed`, MOSS's `block0_q_norm_out`), and the GPU's route then runs two kernels, waited for.
///
/// Krea 2's norm weight carries its `1 +` from the forge; Anima's and Krea 2's tables were converted to interleaved pairs by the forge
/// (`rope_layout`); MOSS's are not, and its table is `[sin(Dh/2) | cos(Dh/2)]`, not pairs. Qwen's K is written into the joint K/V at
/// row `first`, the prefix rows cached already rotated. A refutation stands beside this route: Krea 2's norms on the GPU with a
/// compensated sum moved its guided trajectory from fp64 (the QK-norm alone 1.35·10⁻³ against 6.0·10⁻⁴; `Krea2DiT.modulatedNorm`) — this
/// sum is a plain fp32 one, as the CPU's, and the checks decide again.
package final class QKNormRope: @unchecked Sendable {   // immutable after init; a queue is thread-safe
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    struct Params { uint vectors; uint heads; uint headDim; float eps; uint norm; uint rope; uint halfSplit; uint freqStride; };

    kernel void qk_norm_rope(device float *x [[buffer(0)]], constant float *weight [[buffer(1)]],
                             device const float *freqs [[buffer(2)]], constant Params &p [[buffer(3)]],
                             uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint groups [[simdgroups_per_threadgroup]]) {
        // The products and the sums as written, each rounded: no FMA (`.safe` alone does not forbid the contraction).
        #pragma METAL fp contract(off)
        uint vector = group * groups + sg;
        if (vector >= p.vectors) return;                 // a whole simdgroup leaves together
        device float *row = x + ulong(vector) * p.headDim;
        uint pairs = p.headDim / 2;
        float scale = 1.0f;
        if (p.norm != 0) {
            float ss = 0.0f;
            for (uint j = lane; j < pairs; j += 32) {
                float a = row[p.halfSplit != 0 ? j : 2 * j], b = row[p.halfSplit != 0 ? j + pairs : 2 * j + 1];
                float sa = a * a, sb = b * b;
                ss = ss + sa;
                ss = ss + sb;
            }
            float total = simd_sum(ss);
            float mean = precise::divide(total, float(p.headDim));
            scale = precise::divide(1.0f, precise::sqrt(mean + p.eps));
        }
        device const float *f = freqs + ulong(vector / p.heads) * p.freqStride;
        for (uint j = lane; j < pairs; j += 32) {
            uint i0 = p.halfSplit != 0 ? j : 2 * j, i1 = p.halfSplit != 0 ? j + pairs : 2 * j + 1;
            float a = row[i0], b = row[i1];
            if (p.norm != 0) {
                a = a * scale; a = a * weight[i0];
                b = b * scale; b = b * weight[i1];
            }
            if (p.rope != 0) {
                if (p.halfSplit != 0) {
                    // MOSS: x·cos + rotate_half(x)·sin, the table [sin | cos].
                    float s = f[j], c = f[pairs + j];
                    float ac = a * c, bs = (-b) * s, bc = b * c, asn = a * s;
                    a = ac + bs; b = bc + asn;
                } else {
                    // (re + i·im)·(cos + i·sin), the table's pairs (cos, sin).
                    float fr = f[2 * j], fi = f[2 * j + 1];
                    float rr = a * fr, ii = b * fi, ri = a * fi, ir = b * fr;
                    a = rr - ii; b = ri + ir;
                }
            }
            row[i0] = a; row[i1] = b;
        }
    }
    """

    /// The pairs the RoPE turns: adjacent (`Ops.rope`, the forge's interleaved tables) or `j` with `j + Dh/2` (MOSS-Image's table).
    package enum Layout: UInt32, Sendable { case interleaved = 0, halfSplit = 1 }

    /// One tensor `[rows, heads, Dh]` at `offset` bytes into `buffer`, normed by `weight` (`[Dh]`; `nil`: no norm) then rotated.
    package struct Operand {
        package var buffer: MTLBuffer
        package var offset: Int
        package var heads: Int
        package var weight: [Float]?
        package init(_ buffer: MTLBuffer, offset: Int = 0, heads: Int, weight: [Float]?) {
            self.buffer = buffer; self.offset = offset; self.heads = heads; self.weight = weight
        }
    }

    /// The head dimensions the kernel takes: even, its weight within `setBytes`' 4 KB.
    package static func runs(headDim: Int) -> Bool { headDim > 0 && headDim % 2 == 0 && headDim <= 1024 }

    private struct Params {
        var vectors: UInt32, heads: UInt32, headDim: UInt32, eps: Float, norm: UInt32, rope: UInt32, halfSplit: UInt32, freqStride: UInt32
    }

    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    package init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let f = library.makeFunction(name: "qk_norm_rope") else { throw GEMM.Failure.noDevice }
        pipeline = try device.makeComputePipelineState(function: f)
    }

    /// **One per command queue**, compiled once: every block of a family shares it.
    package static func shared(device: MTLDevice, queue: MTLCommandQueue) -> QKNormRope? {
        let key = ObjectIdentifier(queue)
        return built.withLock { cache in
            if let k = cache[key] { return k }
            guard let made = try? QKNormRope(device: device, queue: queue) else { return nil }
            cache[key] = made
            return made
        }
    }
    private static let built = Mutex<[ObjectIdentifier: QKNormRope]>([:])

    /// **Every operand in one command buffer**: each normed (where it has a weight) then rotated (where `freqs` is given, `[rows, Dh]`
    /// floats from its offset in bytes), in place. Returns the GPU's seconds when `wait`.
    ///
    /// `wait: false` is the route chained to the attention: committed and left on the queue, the attention's command buffer committed
    /// next runs after it (one queue, tracked buffers), and the GPU goes from the products to this kernel to the attention without an
    /// idle and without a completion paid in between. Whoever reads Q or K on the CPU next (a cut by heads, a boundary recorded) waits.
    @discardableResult
    package func run(_ operands: [Operand], rows: Int, headDim: Int, eps: Float, freqs: (buffer: MTLBuffer, offset: Int)?,
                     layout: Layout, wait: Bool = true) -> Double {
        guard rows > 0, !operands.isEmpty else { return 0 }
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            Attention.report("QK-norm + RoPE on the GPU: no command buffer, so nothing ran")
            return 0
        }
        encoder.setComputePipelineState(pipeline)
        let simdgroups = 8
        for o in operands {
            var p = Params(vectors: UInt32(rows * o.heads), heads: UInt32(o.heads), headDim: UInt32(headDim), eps: eps,
                           norm: o.weight == nil ? 0 : 1, rope: freqs == nil ? 0 : 1, halfSplit: layout.rawValue, freqStride: UInt32(headDim))
            encoder.setBuffer(o.buffer, offset: o.offset, index: 0)
            var weight = o.weight ?? [0]
            encoder.setBytes(&weight, length: weight.count * 4, index: 1)
            encoder.setBuffer(freqs?.buffer ?? o.buffer, offset: freqs?.offset ?? o.offset, index: 2)
            encoder.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
            let groups = (rows * o.heads + simdgroups - 1) / simdgroups
            encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32 * simdgroups, height: 1, depth: 1))
        }
        encoder.endEncoding()
        guard wait else {
            commands.addCompletedHandler { done in _ = Attention.verdict(done, what: "QK-norm + RoPE on the GPU") }
            commands.commit()
            return 0
        }
        commands.commit()
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "QK-norm + RoPE on the GPU")
    }
}
