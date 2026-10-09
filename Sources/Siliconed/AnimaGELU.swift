import Foundation
import Metal

/// **Anima's exact GELU, on the GPU**.
///
/// On the CPU, element-by-element `erff` cost **0.72 s per evaluation at 1024²** — 10% of the
/// time, and the GPU waited during that time, since the next GEMM depends on it.
///
/// Metal has no `erf`: it is written here, in two regimes, and **judged against double** by
/// a developer check before being wired in (the rule: never less precise than the path
/// it replaces).
///
///   - `|x| < √2`: the Taylor series of `erf` up to `x²⁵` (remainder < 10⁻¹⁰), written as
///     `½x + φ₀·x²·(1 − w·R(w))` so that the large terms round only once
///     (`fma`, compensated sum). **The judge demanded it twice**: the boundary at `|z| = 0.5`
///     gave 2.0 ulp just after it, the naive series `½x(1 + erf)` 1.79 ulp — against 1.37
///     for the CPU's `erff`;
///   - beyond: `erfc(z) = t · exp(−z² + P(t))`, `t = 1/(1 + z/2)` (Numerical Recipes, `erfcc`,
///     **relative** error < 1.2·10⁻⁷), with `z²` split by `fma` so the exponent loses
///     nothing, and `x − ½x·erfc` in a single `fma`. For `x < 0`, the GELU is `½ x · erfc(|z|)` **without subtraction**: the
///     negative tail is more accurate than the CPU's `½ x (1 + erff(z))`, which cancels almost everything there.
package final class AnimaGELU {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // |x| < √2: GELU(x) = ½x + φ₀·x²·(1 − w·R(w)), w = x²/2, φ₀ = 1/√(2π) — same series as
    // erf, but written so that the large terms round only once: x² and φ₀·x² as
    // two floats (fma), the final sum compensated; only the small term w·R(w) is in fp32.
    static inline float gelu_core(float v) {
        const float c_hi = 0.3989422917366028f, c_lo = -1.133517008e-08f;
        float u = v * v, u_lo = fma(v, v, -u);
        float w = 0.5f * u;
        float r = 1.0f/11975040000.0f;
        r = 1.0f/918086400.0f - w * r;
        r = 1.0f/76204800.0f  - w * r;
        r = 1.0f/6894720.0f   - w * r;
        r = 1.0f/685440.0f    - w * r;
        r = 1.0f/75600.0f     - w * r;
        r = 1.0f/9360.0f      - w * r;
        r = 1.0f/1320.0f      - w * r;
        r = 1.0f/216.0f       - w * r;
        r = 1.0f/42.0f        - w * r;
        r = 1.0f/10.0f        - w * r;
        r = 1.0f/3.0f         - w * r;             // R(w)
        float p = c_hi * u;
        float p_lo = fma(c_hi, u, -p) + (c_lo * u + c_hi * u_lo);
        float q = p * w * r;                       // φ₀·x²·w·R, small
        float a = 0.5f * v;                        // exact
        float s = a + p, bp = s - a;
        float e = (a - (s - bp)) + (p - bp);       // a + p = s + e, exactly
        return s + (e + (p_lo - q));
    }

    static inline float erfc_tail(float z) {           // z ≥ 1, relative error < 1.2e-7
        float t = 1.0f / (1.0f + 0.5f * z);
        float p = -1.26551223f + t*(1.00002368f + t*(0.37409196f + t*(0.09678418f
                + t*(-0.18628806f + t*(0.27886807f + t*(-1.13520398f + t*(1.48851587f
                + t*(-0.82215223f + t*0.17087277f))))))));
        float z2 = z * z;
        float low = fma(z, z, -z2);                    // z² = z2 + low, exactly
        return t * precise::exp(p - z2) * (1.0f - low);
    }

    static inline float gelu_exacte(float v) {
        float z = fabs(v) * 0.70710678118654752f;
        if (z < 1.0f) return gelu_core(v);
        float c = erfc_tail(z);
        return v >= 0 ? fma(-0.5f * v, c, v) : 0.5f * v * c;
    }

    kernel void gelu(device float *x [[buffer(0)]], constant uint &n [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        x[i] = gelu_exacte(x[i]);
    }

    // ERNIE-Image's MLP: `up · GELU(gate)`, the GELU rounded then the product — the two
    // roundings of the reference.
    kernel void geglu(device float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                      constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i >= n) return;
        a[i] = gelu_exacte(a[i]) * b[i];
    }
    """

    private let queue: MTLCommandQueue
    private let pipeline, pipelineGeglu: MTLComputePipelineState

    package init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.queue = queue
        let options = MTLCompileOptions()
        options.mathMode = .safe          // no fast-math: `exp` and `fma` as written
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let function = library.makeFunction(name: "gelu"), let geglu = library.makeFunction(name: "geglu") else {
            throw GEMM.Failure.noDevice
        }
        pipeline = try device.makeComputePipelineState(function: function)
        pipelineGeglu = try device.makeComputePipelineState(function: geglu)
    }

    /// `a ← GELU(a) · b` over `count` elements, in place in `a`. Returns the GPU time.
    @discardableResult
    package func geglu(_ a: MTLBuffer, _ b: MTLBuffer, count: Int) -> Double {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            return 0
        }
        var n = UInt32(count)
        encoder.setComputePipelineState(pipelineGeglu)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBytes(&n, length: 4, index: 2)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, pipelineGeglu.maxTotalThreadsPerThreadgroup),
                                                               height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "geglu")
    }

    /// In place, on a shared buffer. Returns the GPU time.
    @discardableResult
    package func run(_ buffer: MTLBuffer, count: Int) -> Double {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            return 0
        }
        var n = UInt32(count)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBytes(&n, length: 4, index: 1)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup),
                                                               height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        return commands.gpuEndTime - commands.gpuStartTime
    }
}
