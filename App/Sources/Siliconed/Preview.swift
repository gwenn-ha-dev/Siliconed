import CoreGraphics
import Foundation

extension Engine {
    /// **A step's preview: a tiny image of the latent, without a decoder.**
    ///
    /// The principle of ComfyUI and Draw Things: a linear projection of the latent's channels to
    /// RGB, one pixel per latent cell — 64×64 at 512², 128×128 at 1024² (half that in a space
    /// whose cell is 16×16 pixels) — enlarged on display. Two
    /// choices make it useful from the first steps:
    ///
    /// 1. **We show x̂₀, not x_t** — the prediction of the final image, not the noised state. In
    ///    flow matching (`x = (1 − σ)·x₀ + σ·ε`), the derivative `v = dx/dσ = ε − x₀` gives
    ///    `x̂₀ = x − σ·v`. Computed **after** the step's update, it is `x̂₀ = x_{i+1} − σ_{i+1}·v`
    ///    (same value: `x_{i+1} = x_i + (σ_{i+1} − σ_i)·v`) — at the last step, σ = 0 and the
    ///    preview is exactly the final latent. The projection is handed `dx/dσ`, the derivative the
    ///    render's `SolverWalk` steps with: a DiT that returns `−v` has it negated first. Under a
    ///    rule that keeps its own estimate (`SolverWalk.estimate`), that estimate is projected as is.
    ///    Check: the last step's preview against the decoded image (a developer check).
    /// 2. **The coefficients are fitted on our golden tensors**, by least squares, and not copied
    ///    (see `PreviewProjection`).
    ///
    /// **Cost: C × H × W multiply-adds** (262 k at 1024², 16 or 64 channels alike) and a byte array,
    /// against 10 to 38 s of evaluation. **Off by default** (`Request.previews`): without it, nothing is computed or
    /// copied. It works on a read of the latent, never on the latent itself: the denoising's fp32
    /// computation is untouched.
    package struct Preview: Sendable {
        /// The batch image (0 outside a batch) and the evaluation, from 1 to `total`.
        package let image: Int
        package let index: Int
        package let total: Int
        /// The latent grid: one pixel per cell.
        package let height: Int
        package let width: Int
        /// 8-bit RGBA, row by row, opaque alpha, **sRGB** — like `ImageRGB.rgba8()`.
        package let rgba: [UInt8]

        /// The preview as an sRGB `CGImage`, to be enlarged on display (nearest neighbor, or smoothed).
        package func cgImage() -> CGImage { PNG.cgImage(rgba: rgba, height: height, width: width) }
    }
}

/// **The C → RGB projection of a latent space** (16 channels here; a family's own space carries its own, fitted and
/// said in its folder):
/// `rgb = Wᵀ·z + b`, in `[-1, 1]`, then the post-processing of `ImageRGB.rgba8` (`x/2 + 0.5`,
/// clamped, ×255, rounded).
///
/// ## Where the numbers come from — the fitting script
///
/// Least squares C → 3 plus bias, on the (latent, decoded image) pairs of our golden tensors —
/// the image reduced by averaging blocks of the space's factor (8×8, 16×16), one latent cell
/// against one pixel. The latents
/// are in the **model's space** (what the DiT sees): for Flux, *before* `/0.3611 + 0.1159`; for
/// Qwen-Image, *before* `×std + mean`. The script reports the fit's error and that of ComfyUI's
/// coefficients (`latent_formats.py`, `Flux` and `Wan21`) on the same pairs, without copying them.
/// In cross-validation (one pair held out, judged on it), ours reach 26.1 to 30.8 dB on
/// Flux against 25.3 to 28.7 for ComfyUI, and 24.5 to 27.0 dB on Qwen-Image against 23.8 to 26.5
/// — better on nine pairs out of ten, worse by a third of a dB on `krea2-vae-512`. A modest but
/// measured gain; note that the 1024² and 768×512 pairs of a same model are crops of the same
/// image, hence not independent.
package struct PreviewProjection: Sendable, Hashable {
    /// `[channel][r, g, b]`, C × 3, one row per channel.
    package let weights: [Float]
    package let bias: [Float]

    package init(weights: [Float], bias: [Float]) {
        precondition(weights.count % 3 == 0 && bias.count == 3, "projection: 3 columns and 3 biases")
        self.weights = weights; self.bias = bias
    }

    package var channels: Int { weights.count / 3 }

    /// **x̂₀ = x − σ·v, projected**, cell by cell. `x` and `v` are `[channels, h, w]`, planar.
    /// `sigma = 0` projects `x` as is.
    package func apply(x: UnsafePointer<Float>, v: UnsafePointer<Float>?, sigma: Float,
                          height: Int, width: Int) -> [UInt8] {
        let plane = height * width
        var rgba = [UInt8](repeating: 255, count: plane * 4)
        for p in 0..<plane {
            var r = bias[0], g = bias[1], b = bias[2]
            for c in 0..<channels {
                var z = x[c * plane + p]
                if let v, sigma != 0 { z -= sigma * v[c * plane + p] }
                r += weights[c * 3] * z; g += weights[c * 3 + 1] * z; b += weights[c * 3 + 2] * z
            }
            for (k, value) in [r, g, b].enumerated() {
                rgba[p * 4 + k] = UInt8(max(0, min(1, value / 2 + 0.5)) * 255 + 0.5)
            }
        }
        return rgba
    }
}

extension PreviewProjection {
    // ⚙️ Written by the fitting script (development tooling) — do not edit by hand: rerun the script.
    // BEGIN COEFFICIENTS
    static let flux = PreviewProjection(weights: [
        -0.021722,  0.030442,  0.068579,
         0.030569,  0.058863,  0.091297,
         0.041753, -0.022963, -0.033362,
        -0.030951,  0.006380,  0.043529,
         0.065386,  0.048983,  0.024495,
        -0.026616,  0.007646,  0.006444,
         0.074529,  0.090695,  0.088466,
        -0.036164, -0.041829, -0.052455,
        -0.027163,  0.023047,  0.092066,
         0.095936,  0.062281, -0.015840,
        -0.000569,  0.033053,  0.028758,
         0.059258,  0.033929,  0.028479,
         0.051728,  0.045209,  0.047121,
        -0.105407, -0.058613, -0.082295,
        -0.017352, -0.054230, -0.040466,
        -0.089239, -0.060557, -0.039597
    ], bias: [0.040078, -0.005639, -0.042384])
    static let qwenImage = PreviewProjection(weights: [
        -0.039004,  0.023708,  0.328656,
        -0.023992, -0.030591, -0.025290,
         0.216827,  0.160770,  0.109125,
         0.131991,  0.213204,  0.060481,
        -0.040015, -0.059801, -0.087307,
         0.079022, -0.046904, -0.050131,
        -0.190713, -0.311770, -0.270786,
        -0.089385,  0.015674,  0.044169,
        -0.280212, -0.244080, -0.283342,
        -0.130107,  0.065055,  0.139009,
        -0.076211,  0.092954,  0.053454,
         0.038746,  0.077790,  0.087452,
        -0.083879,  0.081037,  0.157425,
        -0.009589, -0.044433, -0.026813,
         0.326472,  0.193572,  0.291136,
         0.106329,  0.063365,  0.123700
    ], bias: [-0.260957, -0.276968, -0.454771])
    // END COEFFICIENTS
}
