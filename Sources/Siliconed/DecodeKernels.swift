import Foundation
import Metal

/// **The kernels of one decoded token**, for a language model whose weights sit resident in the process's
/// own pages and are read there, in place (`Qwen3VLDecoder`). One new row meets every weight once, so a token
/// is bandwidth-bound: the matrix engines of `TensorGEMM` (64-row tiles, a row of one) have nothing to do here,
/// and the weights must not be copied or widened first (7 GB of int8 and 1.2 GB of bf16 a token).
///
///     x ─ norm_rotate ─▶ r ─┬ gemv q ─┐
///                           ├ gemv k ─┼▶ qkv_post ─▶ q, K/V cache row ─ attn_decode ─ rotate_row ─ gemv o ─┐
///                           └ gemv v ─┘   (sum of the partials, QK-norm, RoPE, append)                        │
///     ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┘
///     └▶ norm_rotate (+ residual) ─▶ r ─┬ gemv gate ─┐
///                                       └ gemv up   ─┴▶ swiglu_rotate ─▶ r ─ gemv down ─▶ (residual, in the next norm)
///     ... 36 layers ..., then norm_rotate (final norm) ─▶ head_bf16 ─▶ logits ─ argmax_top2 ─▶ (id, top 1, top 2)
///
/// What sets these kernels apart, and what a port by analogy with the prefill's products would miss:
///
///   - **The rotation of a convrot weight goes to the activation, once per input, not once per product**:
///     `x · W = (x · R) · q · s` (`W = (q · s) · R`, `R = H/√G` symmetric), so q, k and v share one rotated row, gate
///     and up another, and the rotation is fused into the kernel that already holds the row (the norm, SwiGLU).
///     The integers are never touched; they are widened in registers (`float(char)`), the products
///     accumulate in fp32, the column's scale multiplies the sum after it — as `TensorGEMM`'s `scale_columns` does.
///   - **A product is cut along K into `splits` ranges whose partial sums are written apart** (`parts`, `[splits, n]`),
///     and the kernel that consumes the product adds them (in a fixed order: the same bits at each run): a `[4096, 1024]`
///     weight has two 512-column tiles, which fill two cores of the sixteen; cut in 64 along K it fills the GPU. No atomics,
///     no kernel of its own for the sum, the residual add and the norms fuse it.
///   - **The layout is the forge's, `[k, n]`**: the outputs are contiguous, so a simdgroup reads 512 consecutive bytes of one
///     input row (32 lanes × 16 columns) — the weight streams at the memory's width. The head is a published `nn.Linear`,
///     `[vocabulary, k]`: one simdgroup per row, the lanes along K, `simd_sum`.
///   - **`lm_head` stays bf16 and is read as it is**: a shift widens it, the products accumulate in fp32.
///   - **The argmax is the CPU's, to the tie**: the lowest identifier of the maximum (`torch.argmax`), the runner-up counted
///     with multiplicity (a tie has a gap of 0) — `Qwen3VLGenerator.Top2`'s rule, merged by pairs; only 12 bytes go back.
///   - **The norms are not the DiT's refuted ones**: the sum of squares is fp32, in a fixed order (per-thread, `simd_sum`, 32
///     partials), and the two roundings of `x · scale · w` are kept (`fp contract(off)`) as `Ops.rmsNorm` rounds them.
///   - **One query row needs no score matrix**: `attn_decode` is an online softmax per simdgroup (running maximum, sum and
///     accumulator over its keys, merged at the end), whatever the length of the cache.
package final class DecodeKernels {
    package let device: MTLDevice
    let gemvInt8, gemvBF16, head, argmax, normRotate, qkvPost, attention, rotateRow, swigluRotate: MTLComputePipelineState

    /// The columns of an int8 product one threadgroup covers (32 lanes × 16), and a bf16 one (32 × 8).
    package static let int8Tile = 512, bf16Tile = 256
    /// Simdgroups of a product's threadgroup: they split its K range between them.
    package static let simdgroups = 8
    /// Rows of the head one threadgroup computes (8 simdgroups × 2).
    package static let headRows = 16

    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // ── products ──────────────────────────────────────────────────────────────────────────

    struct GemvArgs { uint k, n, splits, scaleKind; };

    // The scale of column `col`, in the type it was published in (0: none — a bf16 weight).
    inline float column_scale(device const uchar *scales, uint kind, uint col) {
        switch (kind) {
        case 1: return ((device const float *)scales)[col];
        case 2: return as_type<float>(uint(((device const ushort *)scales)[col]) << 16);
        case 3: return float(((device const half *)scales)[col]);
        default: return 1.0f;
        }
    }

    // parts[split, col] = scale[col] · Σ_{k in the split's range} x[k] · float(q[k, col]); q int8 [k, n], n % 16 == 0.
    // A threadgroup is 8 simdgroups on the same 512 columns: simdgroup s takes rows k0 + s, k0 + s + 8, ... of its split
    // (the split's bounds are first·k/splits), four rows in flight; the eight partial sums are added in a fixed order.
    kernel void gemv_int8(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],
                          device const uchar *scales [[buffer(2)]], device float *parts [[buffer(3)]],
                          constant GemvArgs &a [[buffer(4)]],
                          uint2 g [[threadgroup_position_in_grid]], uint2 tp [[thread_position_in_threadgroup]]) {
        const uint t = tp.x;
        threadgroup float red[8 * 512];
        const uint sg = t / 32, lane = t % 32, col0 = g.x * 512 + lane * 16;
        const uint k0 = uint((ulong(g.y) * a.k) / a.splits), k1 = uint((ulong(g.y + 1) * a.k) / a.splits);
        float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
        if (col0 < a.n) {
            device const uchar *p = w + col0;
            uint r = k0 + sg;
            for (; r + 24 < k1; r += 32) {
                const uint4 u0 = *(device const uint4 *)(p + ulong(r) * a.n);
                const uint4 u1 = *(device const uint4 *)(p + ulong(r + 8) * a.n);
                const uint4 u2 = *(device const uint4 *)(p + ulong(r + 16) * a.n);
                const uint4 u3 = *(device const uint4 *)(p + ulong(r + 24) * a.n);
                const float4 x0 = float4(x[r]), x1 = float4(x[r + 8]), x2 = float4(x[r + 16]), x3 = float4(x[r + 24]);
                for (uint c = 0; c < 4; ++c) {
                    acc[c] = fma(x0, float4(as_type<char4>(u0[c])), acc[c]);
                    acc[c] = fma(x1, float4(as_type<char4>(u1[c])), acc[c]);
                    acc[c] = fma(x2, float4(as_type<char4>(u2[c])), acc[c]);
                    acc[c] = fma(x3, float4(as_type<char4>(u3[c])), acc[c]);
                }
            }
            for (; r < k1; r += 8) {
                const uint4 u = *(device const uint4 *)(p + ulong(r) * a.n);
                const float4 xr = float4(x[r]);
                for (uint c = 0; c < 4; ++c) acc[c] = fma(xr, float4(as_type<char4>(u[c])), acc[c]);
            }
        }
        for (uint c = 0; c < 4; ++c)
            for (uint j = 0; j < 4; ++j) red[sg * 512 + lane * 16 + c * 4 + j] = acc[c][j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = t; c < 512; c += 256) {
            const uint col = g.x * 512 + c;
            if (col >= a.n) continue;
            float s = 0.0f;
            for (uint i = 0; i < 8; ++i) s += red[i * 512 + c];
            parts[ulong(g.y) * a.n + col] = s * column_scale(scales, a.scaleKind, col);
        }
    }

    inline float4 bf16x4(uint2 u) {
        return float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xFFFF0000u),
                      as_type<float>(u.y << 16), as_type<float>(u.y & 0xFFFF0000u));
    }

    // The same on a bf16 weight [k, n] (a Standard map's layers), n % 8 == 0: 256 columns a threadgroup, no scale.
    kernel void gemv_bf16(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],
                          device const uchar *scales [[buffer(2)]], device float *parts [[buffer(3)]],
                          constant GemvArgs &a [[buffer(4)]],
                          uint2 g [[threadgroup_position_in_grid]], uint2 tp [[thread_position_in_threadgroup]]) {
        const uint t = tp.x;
        threadgroup float red[8 * 256];
        const uint sg = t / 32, lane = t % 32, col0 = g.x * 256 + lane * 8;
        const uint k0 = uint((ulong(g.y) * a.k) / a.splits), k1 = uint((ulong(g.y + 1) * a.k) / a.splits);
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f);
        if (col0 < a.n) {
            device const uchar *p = w + 2 * ulong(col0);
            uint r = k0 + sg;
            for (; r + 8 < k1; r += 16) {
                const uint4 u0 = *(device const uint4 *)(p + 2 * ulong(r) * a.n);
                const uint4 u1 = *(device const uint4 *)(p + 2 * ulong(r + 8) * a.n);
                const float4 x0 = float4(x[r]), x1 = float4(x[r + 8]);
                acc0 = fma(x0, bf16x4(u0.xy), acc0); acc1 = fma(x0, bf16x4(u0.zw), acc1);
                acc0 = fma(x1, bf16x4(u1.xy), acc0); acc1 = fma(x1, bf16x4(u1.zw), acc1);
            }
            for (; r < k1; r += 8) {
                const uint4 u = *(device const uint4 *)(p + 2 * ulong(r) * a.n);
                const float4 xr = float4(x[r]);
                acc0 = fma(xr, bf16x4(u.xy), acc0); acc1 = fma(xr, bf16x4(u.zw), acc1);
            }
        }
        for (uint j = 0; j < 4; ++j) { red[sg * 256 + lane * 8 + j] = acc0[j]; red[sg * 256 + lane * 8 + 4 + j] = acc1[j]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint col = g.x * 256 + t;
            if (col < a.n) {
                float s = 0.0f;
                for (uint i = 0; i < 8; ++i) s += red[i * 256 + t];
                parts[ulong(g.y) * a.n + col] = s;
            }
        }
    }

    // ── the head ──────────────────────────────────────────────────────────────────────────

    struct HeadArgs { uint k, n; };

    // logits[row] = Σ_k x[k] · bf16(w[row, k]), w [n, k] as published, k % 512 == 0. A simdgroup takes two rows (four 16-byte
    // loads in flight per lane), a threadgroup sixteen; fp32 accumulators, lanes added by simd_sum.
    kernel void head_bf16(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],
                          device float *logits [[buffer(2)]], constant HeadArgs &a [[buffer(3)]],
                          uint g [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
        const uint sg = t / 32, lane = t % 32, r0 = g * 16 + sg * 2, r1 = r0 + 1;
        device const uint4 *w0 = (device const uint4 *)(w + 2 * ulong(r0 < a.n ? r0 : 0) * a.k);
        device const uint4 *w1 = (device const uint4 *)(w + 2 * ulong(r1 < a.n ? r1 : 0) * a.k);
        device const float4 *xv = (device const float4 *)x;
        float4 s0 = float4(0.0f), s1 = float4(0.0f);
        for (uint q = lane; q < a.k / 8; q += 64) {
            const uint4 a0 = w0[q], a1 = w1[q], b0 = w0[q + 32], b1 = w1[q + 32];
            const float4 xa = xv[2 * q], xb = xv[2 * q + 1], xc = xv[2 * q + 64], xd = xv[2 * q + 65];
            s0 = fma(xa, bf16x4(a0.xy), s0); s0 = fma(xb, bf16x4(a0.zw), s0);
            s1 = fma(xa, bf16x4(a1.xy), s1); s1 = fma(xb, bf16x4(a1.zw), s1);
            s0 = fma(xc, bf16x4(b0.xy), s0); s0 = fma(xd, bf16x4(b0.zw), s0);
            s1 = fma(xc, bf16x4(b1.xy), s1); s1 = fma(xd, bf16x4(b1.zw), s1);
        }
        const float d0 = simd_sum(s0.x + s0.y + s0.z + s0.w), d1 = simd_sum(s1.x + s1.y + s1.z + s1.w);
        if (lane == 0) {
            if (r0 < a.n) logits[r0] = d0;
            if (r1 < a.n) logits[r1] = d1;
        }
    }

    // The two largest logits and the first index of the largest — `Top2`'s rule: the runner-up counts a tie.
    struct Top { float v1; uint i1; float v2; };

    inline Top merge_top(Top a, Top b) {
        const bool second = b.v1 > a.v1 || (b.v1 == a.v1 && b.i1 < a.i1);
        Top w = second ? b : a;
        w.v2 = max(w.v2, second ? a.v1 : b.v1);
        return w;
    }

    // One threadgroup of 1024: out = (id, bits of the largest, bits of the runner-up).
    kernel void argmax_top2(device const float *logits [[buffer(0)]], device uint *out [[buffer(1)]],
                            constant uint &n [[buffer(2)]],
                            uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
        threadgroup Top shared[32];
        Top top = {-INFINITY, 0xFFFFFFFFu, -INFINITY};
        for (uint i = t; i < n; i += 1024) {
            const float v = logits[i];
            if (v > top.v1) { top.v2 = top.v1; top.v1 = v; top.i1 = i; } else if (v > top.v2) { top.v2 = v; }
        }
        for (uint offset = 16; offset > 0; offset >>= 1) {
            Top other = {simd_shuffle_down(top.v1, offset), simd_shuffle_down(top.i1, offset), simd_shuffle_down(top.v2, offset)};
            top = merge_top(top, other);
        }
        if (lane == 0) shared[sg] = top;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            Top all = shared[0];
            for (uint i = 1; i < 32; ++i) all = merge_top(all, shared[i]);
            out[0] = all.i1; out[1] = as_type<uint>(all.v1); out[2] = as_type<uint>(all.v2);
        }
    }

    // ── the rest of a layer ───────────────────────────────────────────────────────────────

    // y = x · R in place in threadgroup memory, R = H/√G block-diagonal by groups of G (a power of 4): the butterflies
    // of `TensorGEMM.rotate_groups`, the same rounding. `v` holds `count` values (groups of G), `threads` threads.
    inline void rotate_in_place(threadgroup float *v, uint count, uint G, uint t, uint threads) {
        for (uint span = 1; span < G; span *= 4) {
            for (uint idx = t; idx < count / 4; idx += threads) {
                const uint j = (idx / (G / 4)) * G + ((idx % (G / 4)) / span) * 4 * span + (idx % (G / 4)) % span;
                const float a = v[j], b = v[j + span], c = v[j + 2 * span], d = v[j + 3 * span];
                const float p = a + b, m = a - b, u = c + d, w = c - d;
                v[j] = p + w; v[j + span] = p - w; v[j + 2 * span] = m + u; v[j + 3 * span] = u - m;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    struct NormArgs { uint n, parts, partStride, group; float eps, groupNorm; };

    // x += Σ parts (the residual, if any), then out = rotate(RMSNorm(x) · w) — one threadgroup of 1024 for the row. The
    // order of Ops.rmsNorm: scale = 1/√(Σx²/n + eps) (fp32 sums, fixed order), (x · scale) · w in two roundings.
    kernel void norm_rotate(device float *x [[buffer(0)]], device const float *parts [[buffer(1)]],
                            device const float *w [[buffer(2)]], device float *out [[buffer(3)]],
                            constant NormArgs &a [[buffer(4)]], threadgroup float *v [[threadgroup(0)]],
                            uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float partial[32];
        float ss = 0.0f;
        for (uint i = t; i < a.n; i += 1024) {
            float s = x[i];
            if (a.parts > 0) {
                float p = parts[i];
                for (uint q = 1; q < a.parts; ++q) p += parts[ulong(q) * a.partStride + i];
                s = s + p;
                x[i] = s;
            }
            v[i] = s;
            ss += s * s;
        }
        ss = simd_sum(ss);
        if (lane == 0) partial[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0; i < 32; ++i) total += partial[i];
        const float scale = precise::divide(1.0f, precise::sqrt(precise::divide(total, float(a.n)) + a.eps));
        for (uint i = t; i < a.n; i += 1024) v[i] = (v[i] * scale) * w[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (a.group > 0) rotate_in_place(v, a.n, a.group, t, 1024);
        const float norm = a.group > 0 ? a.groupNorm : 1.0f;
        for (uint i = t; i < a.n; i += 1024) out[i] = v[i] * norm;
    }

    struct PostArgs { uint heads, kvHeads, headDim, qParts, kParts, vParts; float eps; };

    // One threadgroup (headDim threads) per head of q, k and v: add the product's partials, then q and k are
    // RMSNormed by head (QK-norm) and rotated by RoPE on adjacent pairs, and k and v are written at the cache row
    // (the buffers arrive offset to it). `freqs`: (cos, sin) per pair, the row's.
    kernel void qkv_post(device const float *qp [[buffer(0)]], device const float *kp [[buffer(1)]],
                         device const float *vp [[buffer(2)]], device const float *qw [[buffer(3)]],
                         device const float *kw [[buffer(4)]], device const float *freqs [[buffer(5)]],
                         device float *q [[buffer(6)]], device float *keys [[buffer(7)]], device float *values [[buffer(8)]],
                         constant PostArgs &a [[buffer(9)]],
                         uint g [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float row[256];
        threadgroup float partial[8];
        const uint D = a.headDim, kvWidth = a.kvHeads * D, qWidth = a.heads * D;
        if (g >= a.heads + a.kvHeads) {
            const uint h = g - a.heads - a.kvHeads, i = h * D + t;
            float s = vp[i];
            for (uint p = 1; p < a.vParts; ++p) s += vp[ulong(p) * kvWidth + i];
            values[i] = s;
            return;
        }
        const bool isQ = g < a.heads;
        const uint h = isQ ? g : g - a.heads, i = h * D + t, parts = isQ ? a.qParts : a.kParts, width = isQ ? qWidth : kvWidth;
        device const float *src = isQ ? qp : kp;
        float s = src[i];
        for (uint p = 1; p < parts; ++p) s += src[ulong(p) * width + i];
        float ss = simd_sum(s * s);
        if (lane == 0) partial[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint i = 0; i < D / 32; ++i) total += partial[i];
        const float scale = precise::divide(1.0f, precise::sqrt(precise::divide(total, float(D)) + a.eps));
        row[t] = (s * scale) * (isQ ? qw : kw)[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint pair = t / 2;
        const float re = row[2 * pair], im = row[2 * pair + 1], fr = freqs[2 * pair], fi = freqs[2 * pair + 1];
        const float rotated = (t % 2 == 0) ? (re * fr - im * fi) : (re * fi + im * fr);
        (isQ ? q : keys)[i] = rotated;
    }

    struct AttentionArgs { uint heads, kvHeads, length; float scale; };

    // Attention of one query row over `length` cached rows (head dimension 128): a threadgroup per query head, eight
    // simdgroups each walking every eighth key with an online softmax (a lane holds four of the 128 dimensions),
    // merged through threadgroup memory. Scores scaled after the sum, as the CPU's sgemm alpha.
    kernel void attn_decode(device const float *q [[buffer(0)]], device const float *keys [[buffer(1)]],
                            device const float *values [[buffer(2)]], device float *out [[buffer(3)]],
                            constant AttentionArgs &a [[buffer(4)]],
                            uint h [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
        threadgroup float maxima[8], sums[8];
        threadgroup float4 accumulators[8 * 32];
        const uint kvWidth = a.kvHeads * 128, kvHead = h / (a.heads / a.kvHeads);
        const float4 query = ((device const float4 *)(q + h * 128))[lane];
        device const float4 *k = (device const float4 *)(keys + kvHead * 128) + lane;
        device const float4 *v = (device const float4 *)(values + kvHead * 128) + lane;
        const uint stride = kvWidth / 4;
        float m = -INFINITY, l = 0.0f;
        float4 acc = float4(0.0f);
        for (uint i = sg; i < a.length; i += 8) {
            const float s = simd_sum(dot(query, k[ulong(i) * stride])) * a.scale;
            const float top = max(m, s), alpha = precise::exp(m - top), p = precise::exp(s - top);
            l = l * alpha + p;
            acc = acc * alpha + p * v[ulong(i) * stride];
            m = top;
        }
        if (lane == 0) { maxima[sg] = m; sums[sg] = l; }
        accumulators[sg * 32 + lane] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < 32) {
            float top = maxima[0];
            for (uint i = 1; i < 8; ++i) top = max(top, maxima[i]);
            float den = 0.0f;
            float4 num = float4(0.0f);
            for (uint i = 0; i < 8; ++i) {
                const float w = precise::exp(maxima[i] - top);
                den += sums[i] * w;
                num += accumulators[i * 32 + t] * w;
            }
            ((device float4 *)(out + h * 128))[t] = num / den;
        }
    }

    // out = rotate(in) by groups of G: one threadgroup (64 threads) a group of 256 values at most.
    struct RotateArgs { uint n, group; float groupNorm; };

    kernel void rotate_row(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],
                           constant RotateArgs &a [[buffer(2)]],
                           uint g [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
        threadgroup float v[256];
        const uint G = a.group, base = g * G;
        for (uint i = t; i < G; i += 64) v[i] = in[base + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        rotate_in_place(v, G, G, t, 64);
        for (uint i = t; i < G; i += 64) out[base + i] = v[i] * a.groupNorm;
    }

    struct SwigluArgs { uint n, gateParts, upParts, stride, group; float groupNorm; };

    // out = rotate(silu(gate) · up) from the two products' partials: silu(g) = g / (1 + e^{-g}), the CPU's steps
    // (negate, exp, add, divide, multiply). A threadgroup (64 threads) a group; `group` 0: no rotation, blocks of 256.
    kernel void swiglu_rotate(device const float *gp [[buffer(0)]], device const float *up [[buffer(1)]],
                              device float *out [[buffer(2)]], constant SwigluArgs &a [[buffer(3)]],
                              uint g [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
        #pragma METAL fp contract(off)
        threadgroup float v[256];
        const uint G = a.group > 0 ? a.group : 256, base = g * G;
        for (uint i = t; i < G; i += 64) {
            const uint j = base + i;
            if (j >= a.n) { v[i] = 0.0f; continue; }
            float gate = gp[j], up_ = up[j];
            for (uint p = 1; p < a.gateParts; ++p) gate += gp[ulong(p) * a.stride + j];
            for (uint p = 1; p < a.upParts; ++p) up_ += up[ulong(p) * a.stride + j];
            const float silu = precise::divide(gate, 1.0f + precise::exp(-gate));
            v[i] = silu * up_;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (a.group > 0) rotate_in_place(v, G, G, t, 64);
        const float norm = a.group > 0 ? a.groupNorm : 1.0f;
        for (uint i = t; i < G; i += 64) { if (base + i < a.n) out[base + i] = v[i] * norm; }
    }
    """

    package init(device: MTLDevice) throws {
        self.device = device
        let options = MTLCompileOptions()
        options.mathMode = .safe          // `exp`, `sqrt` and the divisions as written, no fast-math
        let library = try device.makeLibrary(source: Self.source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw GEMM.Failure.noDevice }
            return try device.makeComputePipelineState(function: function)
        }
        gemvInt8 = try pipeline("gemv_int8"); gemvBF16 = try pipeline("gemv_bf16"); head = try pipeline("head_bf16")
        argmax = try pipeline("argmax_top2"); normRotate = try pipeline("norm_rotate"); qkvPost = try pipeline("qkv_post")
        attention = try pipeline("attn_decode"); rotateRow = try pipeline("rotate_row")
        swigluRotate = try pipeline("swiglu_rotate")
    }

    // MARK: - Weights

    /// **A resident `[k, n]` weight as the kernels read it**: int8 with its scales (per column, in the type they were
    /// published in) and an optional convrot of `group`, or bf16. `buffer` holds the values from its offset 0 (the page-aligned
    /// tensor); the scales lie in the same tensor (`scalesOffset`) or in a buffer of their own.
    package struct Matrix {
        package enum Format: Equatable { case int8, bfloat16 }
        package enum ScaleType: UInt32 { case none = 0, float32 = 1, bfloat16 = 2, float16 = 3 }
        package let buffer: MTLBuffer
        package let scales: MTLBuffer?
        package let scalesOffset: Int
        package let scaleType: ScaleType
        package let format: Format
        package let k: Int, n: Int
        /// The convrot group along K, 0 for none.
        package let group: Int

        package init(buffer: MTLBuffer, scales: MTLBuffer? = nil, scalesOffset: Int = 0, scaleType: ScaleType = .none,
                     format: Format, k: Int, n: Int, group: Int = 0) {
            self.buffer = buffer; self.scales = scales; self.scalesOffset = scalesOffset; self.scaleType = scaleType
            self.format = format; self.k = k; self.n = n; self.group = group
        }

        /// Whether the kernels can read it: columns a multiple of the lane's vector, K cut by the rotation's group.
        package var isReadable: Bool {
            switch format {
            case .int8: return n % 16 == 0 && scaleType != .none && (group == 0 || (Widen.isRotationGroup(group) && k % group == 0))
            case .bfloat16: return n % 8 == 0 && group == 0
            }
        }
        /// The columns of a threadgroup.
        package var tile: Int { format == .int8 ? DecodeKernels.int8Tile : DecodeKernels.bf16Tile }
    }

    /// **The K ranges a product is cut in**: enough threadgroups to fill the GPU (`target`), at most one range per 64 rows
    /// (eight a simdgroup). Computed with the int8 tile whatever the format, so the partials' size is a function of the shape.
    package static func splits(k: Int, n: Int, target: Int) -> Int {
        let tiles = (n + int8Tile - 1) / int8Tile
        return max(1, min((target + tiles - 1) / tiles, k / 64))
    }

    // MARK: - Encoding

    /// `parts[splits, n] = x · W` (the scale applied, not yet summed) for a row `x` of `k` floats.
    package func gemv(_ e: MTLComputeCommandEncoder, _ m: Matrix, x: MTLBuffer, xOffset: Int = 0,
                      parts: MTLBuffer, partsOffset: Int = 0, splits: Int) {
        e.setComputePipelineState(m.format == .int8 ? gemvInt8 : gemvBF16)
        e.setBuffer(x, offset: xOffset, index: 0)
        e.setBuffer(m.buffer, offset: 0, index: 1)
        e.setBuffer(m.scales ?? m.buffer, offset: m.scalesOffset, index: 2)
        e.setBuffer(parts, offset: partsOffset, index: 3)
        var args = SIMD4<UInt32>(UInt32(m.k), UInt32(m.n), UInt32(splits), m.scaleType.rawValue)
        e.setBytes(&args, length: 16, index: 4)
        e.dispatchThreadgroups(MTLSize(width: (m.n + m.tile - 1) / m.tile, height: splits, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32 * Self.simdgroups, height: 1, depth: 1))
    }

    /// `logits[n] = x · headᵀ` for `head` `[n, k]` bf16.
    package func headProduct(_ e: MTLComputeCommandEncoder, head w: MTLBuffer, rows n: Int, k: Int, x: MTLBuffer,
                             logits: MTLBuffer) {
        e.setComputePipelineState(head)
        e.setBuffer(x, offset: 0, index: 0)
        e.setBuffer(w, offset: 0, index: 1)
        e.setBuffer(logits, offset: 0, index: 2)
        var args = SIMD2<UInt32>(UInt32(k), UInt32(n))
        e.setBytes(&args, length: 8, index: 3)
        e.dispatchThreadgroups(MTLSize(width: (n + Self.headRows - 1) / Self.headRows, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// `result = (identifier, top 1, top 2)` of `logits[0..<n]`.
    package func argmaxTop2(_ e: MTLComputeCommandEncoder, logits: MTLBuffer, rows n: Int, result: MTLBuffer) {
        e.setComputePipelineState(argmax)
        e.setBuffer(logits, offset: 0, index: 0)
        e.setBuffer(result, offset: 0, index: 1)
        var count = UInt32(n)
        e.setBytes(&count, length: 4, index: 2)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// `x += Σ parts` (when `parts > 0`), `out = rotate(RMSNorm(x) · weight)` (no rotation when `group` is 0).
    /// `n` up to 8 192 (32 KB of threadgroup memory).
    package func normRotate(_ e: MTLComputeCommandEncoder, x: MTLBuffer, parts: MTLBuffer?, count: Int, stride: Int,
                            weight: MTLBuffer, weightOffset: Int, out: MTLBuffer, n: Int, group: Int, eps: Float) {
        precondition(n <= 8192 && n % 4 == 0 && (group == 0 || n % group == 0), "norm_rotate: \(n) values, group \(group)")
        e.setComputePipelineState(normRotate)
        e.setBuffer(x, offset: 0, index: 0)
        e.setBuffer(parts ?? x, offset: 0, index: 1)
        e.setBuffer(weight, offset: weightOffset, index: 2)
        e.setBuffer(out, offset: 0, index: 3)
        var args = NormArguments(n: UInt32(n), parts: UInt32(parts == nil ? 0 : count), partStride: UInt32(stride),
                                 group: UInt32(group), eps: eps, groupNorm: group > 0 ? 1 / Float(group).squareRoot() : 1)
        e.setBytes(&args, length: MemoryLayout<NormArguments>.size, index: 4)
        e.setThreadgroupMemoryLength(n * 4, index: 0)
        e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    struct NormArguments { var n, parts, partStride, group: UInt32; var eps, groupNorm: Float }

    /// `out = rotate(in)` by groups of `group`, `n` values.
    package func rotate(_ e: MTLComputeCommandEncoder, _ input: MTLBuffer, out: MTLBuffer, n: Int, group: Int) {
        e.setComputePipelineState(rotateRow)
        e.setBuffer(input, offset: 0, index: 0)
        e.setBuffer(out, offset: 0, index: 1)
        var args = RotateArguments(n: UInt32(n), group: UInt32(group), groupNorm: 1 / Float(group).squareRoot())
        e.setBytes(&args, length: MemoryLayout<RotateArguments>.size, index: 2)
        e.dispatchThreadgroups(MTLSize(width: n / group, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    struct RotateArguments { var n, group: UInt32; var groupNorm: Float }

    /// `out = rotate(silu(gate) · up)` from the partials of the two products (`stride` = their `n`).
    package func swiglu(_ e: MTLComputeCommandEncoder, gate: MTLBuffer, gateParts: Int, up: MTLBuffer, upParts: Int,
                        out: MTLBuffer, n: Int, group: Int) {
        e.setComputePipelineState(swigluRotate)
        e.setBuffer(gate, offset: 0, index: 0)
        e.setBuffer(up, offset: 0, index: 1)
        e.setBuffer(out, offset: 0, index: 2)
        var args = SwigluArguments(n: UInt32(n), gateParts: UInt32(gateParts), upParts: UInt32(upParts), stride: UInt32(n),
                                   group: UInt32(group), groupNorm: group > 0 ? 1 / Float(group).squareRoot() : 1)
        e.setBytes(&args, length: MemoryLayout<SwigluArguments>.size, index: 3)
        let g = group > 0 ? group : 256
        e.dispatchThreadgroups(MTLSize(width: (n + g - 1) / g, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    struct SwigluArguments { var n, gateParts, upParts, stride, group: UInt32; var groupNorm: Float }

    /// The sum of the three projections' partials, QK-norm and RoPE of q and k, k and v appended at the cache row
    /// (`keys`, `values` already offset to it).
    package func post(_ e: MTLComputeCommandEncoder, q: (parts: MTLBuffer, count: Int), k: (parts: MTLBuffer, count: Int),
                      v: (parts: MTLBuffer, count: Int), qNorm: MTLBuffer, kNorm: MTLBuffer, normOffsets: (q: Int, k: Int),
                      freqs: MTLBuffer, out: MTLBuffer, keys: (MTLBuffer, Int), values: (MTLBuffer, Int),
                      heads: Int, kvHeads: Int, headDim: Int, eps: Float) {
        e.setComputePipelineState(qkvPost)
        e.setBuffer(q.parts, offset: 0, index: 0)
        e.setBuffer(k.parts, offset: 0, index: 1)
        e.setBuffer(v.parts, offset: 0, index: 2)
        e.setBuffer(qNorm, offset: normOffsets.q, index: 3)
        e.setBuffer(kNorm, offset: normOffsets.k, index: 4)
        e.setBuffer(freqs, offset: 0, index: 5)
        e.setBuffer(out, offset: 0, index: 6)
        e.setBuffer(keys.0, offset: keys.1, index: 7)
        e.setBuffer(values.0, offset: values.1, index: 8)
        var args = PostArguments(heads: UInt32(heads), kvHeads: UInt32(kvHeads), headDim: UInt32(headDim), qParts: UInt32(q.count),
                                 kParts: UInt32(k.count), vParts: UInt32(v.count), eps: eps)
        e.setBytes(&args, length: MemoryLayout<PostArguments>.size, index: 9)
        e.dispatchThreadgroups(MTLSize(width: heads + 2 * kvHeads, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1))
    }

    struct PostArguments { var heads, kvHeads, headDim, qParts, kParts, vParts: UInt32; var eps: Float }

    /// Attention of the row `q` (`heads × 128`) over `length` cached rows.
    package func attend(_ e: MTLComputeCommandEncoder, q: MTLBuffer, keys: (MTLBuffer, Int), values: (MTLBuffer, Int),
                        out: MTLBuffer, heads: Int, kvHeads: Int, length: Int) {
        e.setComputePipelineState(attention)
        e.setBuffer(q, offset: 0, index: 0)
        e.setBuffer(keys.0, offset: keys.1, index: 1)
        e.setBuffer(values.0, offset: values.1, index: 2)
        e.setBuffer(out, offset: 0, index: 3)
        var args = AttentionArguments(heads: UInt32(heads), kvHeads: UInt32(kvHeads), length: UInt32(length), scale: 1 / Float(128).squareRoot())
        e.setBytes(&args, length: MemoryLayout<AttentionArguments>.size, index: 4)
        e.dispatchThreadgroups(MTLSize(width: heads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    struct AttentionArguments { var heads, kvHeads, length: UInt32; var scale: Float }
}
