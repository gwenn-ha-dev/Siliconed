import Accelerate
import Foundation

/// The operations that are not GEMMs, on the CPU: they weigh little in FLOPs, they carry all
/// the porting risk, and on the CPU they can be debugged. Those that measured faster on the GPU
/// moved there one by one (`ElementwiseGPU`: SwiGLU, gated residuals, GELU); the rest stays here.
///
/// Layout convention, everywhere: row-major, batch of one, hence `[S, …]`.
package enum Ops {

    /// `x · rsqrt(mean(x², last axis) + eps) · weight`
    ///
    /// The mean and not the sum, the eps **under** the root, and the weight applied afterwards — this is
    /// the definition of `diffusers.models.normalization.RMSNorm`, read and not assumed.
    package static func rmsNorm(_ x: UnsafePointer<Float>, weight: UnsafePointer<Float>,
                               into out: UnsafeMutablePointer<Float>,
                               rows: Int, columns: Int, eps: Float) {
        let n = vDSP_Length(columns)
        Parallel.rows(rows, width: columns) { first, howMany in
            for row in first..<(first + howMany) {
                let source = x + row * columns
                let destination = out + row * columns
                var sumOfSquares: Float = 0
                vDSP_svesq(source, 1, &sumOfSquares, n)
                var scale = 1 / (sumOfSquares / Float(columns) + eps).squareRoot()
                vDSP_vsmul(source, 1, &scale, destination, 1, n)
                vDSP_vmul(destination, 1, weight, 1, destination, 1, n)
            }
        }
    }

    /// `x[s, d] *= v[d]` — the adaLN modulation, which is a per-channel vector.
    package static func scaleRows(_ x: UnsafeMutablePointer<Float>, by v: UnsafePointer<Float>,
                                 rows: Int, columns: Int) {
        let n = vDSP_Length(columns)
        Parallel.rows(rows, width: columns) { first, howMany in
            for row in first..<(first + howMany) {
                vDSP_vmul(x + row * columns, 1, v, 1, x + row * columns, 1, n)
            }
        }
    }

    /// `x[s, d] += gate[d] · y[s, d]` — the residual carried by its gate.
    package static func gatedResidual(_ x: UnsafeMutablePointer<Float>, plus y: UnsafePointer<Float>,
                                     gate: UnsafePointer<Float>, rows: Int, columns: Int) {
        let n = vDSP_Length(columns)
        Parallel.rows(rows, width: columns) { first, howMany in
            for row in first..<(first + howMany) {
                vDSP_vma(y + row * columns, 1, gate, 1, x + row * columns, 1, x + row * columns, 1, n)
            }
        }
    }

    /// `silu(a) · b`, with `silu(x) = x · σ(x) = x / (1 + e^{−x})`.
    package static func siluGate(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>,
                                into out: UnsafeMutablePointer<Float>, count: Int,
                                scratch: UnsafeMutablePointer<Float>) {
        // The bounds are computed by cross product, never by `slice_size × index`:
        // an integer division truncates, and the remainder is then handled by no one. This defect
        // left eight elements out of 327,680 with their value from BEFORE the SwiGLU — one row in 32,
        // then propagated by the attention. It survived the switch to serial, because it was
        // in this splitting and not in `Parallel`.
        let slices = Parallel.threads * 2
        Parallel.rows(slices, width: count / slices) { first, howMany in
            let start = first * count / slices
            let end = (first + howMany) * count / slices
            siluGateSlice(a + start, b + start, into: out + start,
                          count: end - start, scratch: scratch + start)
        }
    }

    @inline(__always)
    private static func siluGateSlice(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>,
                                      into out: UnsafeMutablePointer<Float>, count: Int,
                                      scratch: UnsafeMutablePointer<Float>) {
        var n = Int32(count)
        var negativeOne: Float = -1
        vDSP_vsmul(a, 1, &negativeOne, scratch, 1, vDSP_Length(count))   // −a
        vvexpf(scratch, scratch, &n)                                    // e^{−a}
        var one: Float = 1
        vDSP_vsadd(scratch, 1, &one, scratch, 1, vDSP_Length(count))     // 1 + e^{−a}
        vDSP_vdiv(scratch, 1, a, 1, out, 1, vDSP_Length(count))          // a / (1 + e^{−a})
        vDSP_vmul(out, 1, b, 1, out, 1, vDSP_Length(count))              // · b
    }

    package static func tanhInPlace(_ x: UnsafeMutablePointer<Float>, count: Int) {
        var n = Int32(count)
        vvtanhf(x, x, &n)
    }

    package static func addScalar(_ x: UnsafeMutablePointer<Float>, _ value: Float, count: Int) {
        var v = value
        vDSP_vsadd(x, 1, &v, x, 1, vDSP_Length(count))
    }

    /// RoPE, on **adjacent pairs** `(2i, 2i+1)` and not by halves.
    ///
    /// This is pitfall 3.1, and it does not show: a LLaMA-style port, which pairs `i` and `i+D/2`,
    /// produces a result of the right shape, the right norm, and wrong. The reference does
    /// `view_as_complex(x.reshape(…, -1, 2))`, hence adjacent pairs.
    ///
    /// - Parameters:
    ///   - x: `[S, H, Dh]`, modified in place
    ///   - freqs: `[S, Dh/2, 2]` — (real, imaginary), shared by all the heads
    package static func rope(_ x: UnsafeMutablePointer<Float>, freqs: UnsafePointer<Float>,
                            sequence: Int, heads: Int, headDim: Int) {
        let pairs = headDim / 2
        Parallel.rows(sequence, width: heads * headDim) { first, howMany in
            for s in first..<(first + howMany) {
                let f = freqs + s * pairs * 2
                for h in 0..<heads {
                    let row = x + (s * heads + h) * headDim
                    for p in 0..<pairs {
                        let re = row[2 * p], im = row[2 * p + 1]
                        let fr = f[2 * p], fi = f[2 * p + 1]
                        row[2 * p]     = re * fr - im * fi
                        row[2 * p + 1] = re * fi + im * fr
                    }
                }
            }
        }
    }



    /// `y = W·x + b` with `W` as `[output, input]` and `x` a single vector — the adaLN, whose
    /// `M = 1` does not deserve the GPU.
    package static func gemv(weight: UnsafePointer<Float>, bias: UnsafePointer<Float>?,
                            x: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                            outputs: Int, inputs: Int, transposed: Bool = false) {
        // Transposed, the weight is stored `[input, output]`: it is `CblasTrans` and another `lda`.
        if transposed {
            cblas_sgemv(CblasRowMajor, CblasTrans, Int32(inputs), Int32(outputs),
                        1.0, weight, Int32(outputs), x, 1, 0.0, out, 1)
        } else {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(outputs), Int32(inputs),
                        1.0, weight, Int32(inputs), x, 1, 0.0, out, 1)
        }
        if let bias {
            vDSP_vadd(out, 1, bias, 1, out, 1, vDSP_Length(outputs))
        }
    }
}

extension Ops {
    /// `LayerNorm` **without affine**: `(x − mean) / √(variance + eps)`, biased variance.
    /// Z-Image's `FinalLayer` uses it (the only LayerNorm of that model, everything else being
    /// RMSNorm); so do other families (Qwen-Image-2.1's DiT and vision tower, Anima, Klein, ERNIE).
    package static func layerNorm(_ x: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                                 rows: Int, columns: Int, eps: Float) {
        let n = vDSP_Length(columns)
        Parallel.rows(rows, width: columns) { first, howMany in
            for row in first..<(first + howMany) {
                let source = x + row * columns, destination = out + row * columns
                var mean: Float = 0
                vDSP_meanv(source, 1, &mean, n)
                var negativeMean = -mean
                vDSP_vsadd(source, 1, &negativeMean, destination, 1, n)
                var sumOfSquares: Float = 0
                vDSP_svesq(destination, 1, &sumOfSquares, n)
                var scale = 1 / (sumOfSquares / Float(columns) + eps).squareRoot()
                vDSP_vsmul(destination, 1, &scale, destination, 1, n)
            }
        }
    }

    package static func siluInPlace(_ x: UnsafeMutablePointer<Float>, count: Int,
                                   scratch: UnsafeMutablePointer<Float>) {
        var n = Int32(count)
        var negativeOne: Float = -1
        vDSP_vsmul(x, 1, &negativeOne, scratch, 1, vDSP_Length(count))
        vvexpf(scratch, scratch, &n)
        var one: Float = 1
        vDSP_vsadd(scratch, 1, &one, scratch, 1, vDSP_Length(count))
        vDSP_vdiv(scratch, 1, x, 1, x, 1, vDSP_Length(count))
    }

    /// `[S, pH·pW·C]` → `[C, H, W]` — the exact mirror of the patchify:
    /// `view(H/pH, W/pW, pH, pW, C).permute(4, 0, 2, 1, 3)`, so the channel is the fastest
    /// within the token and the slowest within the image.
    package static func unpatchify(_ tokens: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                                  latentHeight: Int, latentWidth: Int, patch: Int, channels: Int) {
        let tilesHigh = latentHeight / patch, tilesWide = latentWidth / patch
        let perToken = patch * patch * channels
        for th in 0..<tilesHigh {
            for tw in 0..<tilesWide {
                let token = tokens + (th * tilesWide + tw) * perToken
                for ph in 0..<patch {
                    for pw in 0..<patch {
                        for c in 0..<channels {
                            let y = th * patch + ph, x = tw * patch + pw
                            out[(c * latentHeight + y) * latentWidth + x] = token[(ph * patch + pw) * channels + c]
                        }
                    }
                }
            }
        }
    }
}

extension Ops {
    /// The timestep embedding: `cat([cos(t·f), sin(t·f)])`, **cos before sin**.
    /// `t` arrives already multiplied by `t_scale` (1000) by the caller, as in the pipeline.
    package static func timestepEmbedding(_ t: Float, into out: UnsafeMutablePointer<Float>,
                                         dim: Int, maxPeriod: Float = 10000) {
        let half = dim / 2
        for j in 0..<half {
            let frequency = expf(-logf(maxPeriod) * Float(j) / Float(half))
            let angle = t * frequency
            out[j] = cosf(angle)
            out[half + j] = sinf(angle)
        }
    }

    /// `[C, H, W]` → `[(H/p)·(W/p), p·p·C]`, the channel fastest within the token.
    /// It is the reference's `view(C, H/p, p, W/p, p).permute(1, 3, 2, 4, 0)`, with `F = pF = 1`.
    package static func patchify(_ latent: UnsafePointer<Float>, into out: UnsafeMutablePointer<Float>,
                                channels: Int, height: Int, width: Int, patch: Int) {
        let tilesHigh = height / patch, tilesWide = width / patch
        let perToken = patch * patch * channels
        for th in 0..<tilesHigh {
            for tw in 0..<tilesWide {
                let token = out + (th * tilesWide + tw) * perToken
                for ph in 0..<patch {
                    for pw in 0..<patch {
                        for c in 0..<channels {
                            token[(ph * patch + pw) * channels + c] =
                                latent[(c * height + th * patch + ph) * width + tw * patch + pw]
                        }
                    }
                }
            }
        }
    }
}

/// The 3D RoPE tables. Axis 0 is a **segment index** and not a time (pitfall 3.2): the text
/// occupies `1…L`, the image occupies the constant value `L+1`, and axes 1 and 2 carry the
/// spatial coordinates. The image padding tokens receive `(0, 0, 0)`.
package struct RopeTables {
    package let axesDims: [Int]
    package let axesLens: [Int]
    package let theta: Float
    private var tables: [[Float]] = []      // per axis: [length][dims/2][2]

    package init(axesDims: [Int], axesLens: [Int], theta: Float) {
        self.axesDims = axesDims
        self.axesLens = axesLens
        self.theta = theta
        for (dims, length) in zip(axesDims, axesLens) {
            let half = dims / 2
            var table = [Float](repeating: 0, count: length * half * 2)
            for position in 0..<length {
                for j in 0..<half {
                    // `1 / theta^(2j/dims)`, angle = position · frequency — then `polar(1, angle)`.
                    let frequency = powf(theta, -Float(2 * j) / Float(dims))
                    let angle = Float(position) * frequency
                    table[(position * half + j) * 2] = cosf(angle)
                    table[(position * half + j) * 2 + 1] = sinf(angle)
                }
            }
            tables.append(table)
        }
    }

    package var complexPerToken: Int { axesDims.reduce(0) { $0 + $1 / 2 } }

    /// Writes `[complexPerToken, 2]` for a token whose positions are `ids`.
    package func write(ids: [Int], into out: UnsafeMutablePointer<Float>) {
        var offset = 0
        for axis in 0..<axesDims.count {
            let half = axesDims[axis] / 2
            let position = min(max(ids[axis], 0), axesLens[axis] - 1)
            tables[axis].withUnsafeBufferPointer { table in
                (out + offset * 2).update(from: table.baseAddress! + position * half * 2, count: half * 2)
            }
            offset += half
        }
    }
}
