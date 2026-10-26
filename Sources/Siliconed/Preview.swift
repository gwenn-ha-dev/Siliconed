import CoreGraphics
import Foundation

extension Engine {
    /// **A step's preview: a tiny image of the latent, without a decoder.**
    ///
    /// The principle of ComfyUI and Draw Things: a linear projection of the latent's channels to
    /// RGB, one pixel per latent cell — 64×64 at 512², 128×128 at 1024² (half that for
    /// Qwen-Image-2.1 and MOSS-Image, whose cell is 16×16 pixels) — enlarged on display. Two
    /// choices make it useful from the first steps:
    ///
    /// 1. **We show x̂₀, not x_t** — the prediction of the final image, not the noised state. In
    ///    flow matching (`x = (1 − σ)·x₀ + σ·ε`), the derivative `v = dx/dσ = ε − x₀` gives
    ///    `x̂₀ = x − σ·v`. Computed **after** the step's update, it is `x̂₀ = x_{i+1} − σ_{i+1}·v`
    ///    (same value: `x_{i+1} = x_i + (σ_{i+1} − σ_i)·v`) — at the last step, σ = 0 and the
    ///    preview is exactly the final latent. The sign convention is verified for each model:
    ///    Anima and Krea 2 return `v` as is (`euler`: `x += Δσ·v`), Z-Image returns `−v` and the
    ///    sampler negates it (`Sampler`: `x += Δσ·negated`), so `negated` IS `dx/dσ`;
    ///    Qwen-Image-2.1 returns `v` as is (`QwenImage21Sampler`: `x += dt·v`), MOSS-Image too (`euler`).
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

/// **The C → RGB projection of a latent space** (16 channels, 64 for Qwen-Image-2.1, 128 for FLUX.2):
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
///
/// **Qwen-Image-2.1**: 64 channels, seven pairs at ~512² (the generation in 6 and 9 steps,
/// two edits, the encoder's two references — fitted on goldens of the base model under the
/// third-party turbo LoRA the family carried before the official Turbo; the latent space and the VAE
/// are the same, the Turbo's VAE being the base's rounded to bf16, so the fit was not redone), the
/// latent *before* its per-channel denormalization, RGB of the RGBA decode. One pair held out: 26.2
/// to 29.1 dB — optimistic, the pairs go by twos; both generations held out together, judged on them: **23.9 dB**. ComfyUI publishes nothing for
/// this space.
///
/// **FLUX.2** (MOSS-Image): 128 channels — 32 VAE channels × the 2×2 patch —, the latent the DiT sees (before the
/// batch-norm denormalization), one pixel per 16-px cell against the image's 16×16 block mean. No golden covers enough
/// images: twelve 512² renders of the installed chain, twelve prompts (portrait, animal, food, night street, sea, anime,
/// black and white, jungle, desert, interior, watercolour, pastel), seed 100 + i, final latent and decoded image (a
/// private maintainer command). One image held out, judged on it: **16.7 to 20.4 dB, 18.7 on average** — below the other
/// spaces: the cell's sub-pixel detail leaks into it as coloured speckle, which no ridge reduces (18.7 at every λ up to
/// 0.03) and a projection shared by the four sub-positions (32 → 3, a twice finer grid) does worse, 17.2 dB brought to
/// the cell. ComfyUI's factors were not compared.
package struct PreviewProjection: Sendable, Equatable {
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

extension LatentSpace {
    /// This space's preview projection (`PreviewProjection`), fitted by
    /// the fitting script. `nil` for a space no fit covers.
    package var preview: PreviewProjection? {
        switch self {
        case .flux: return .flux
        case .qwenImage: return .qwenImage
        case .qwenImage21: return .qwenImage21
        case .flux2: return .flux2
        default: return nil
        }
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
    static let qwenImage21 = PreviewProjection(weights: [
         0.004842, -0.005538, -0.009152,
         0.003870,  0.010174,  0.017347,
         0.066124,  0.040209, -0.005329,
         0.035138,  0.046121, -0.005377,
        -0.005690,  0.000120, -0.010971,
         0.010421,  0.020627,  0.016463,
         0.011977,  0.016070,  0.026829,
        -0.049198, -0.061993, -0.035031,
         0.020464,  0.015447,  0.022097,
        -0.096282, -0.035471, -0.006469,
        -0.025989, -0.043067, -0.057512,
        -0.016315, -0.008524, -0.021850,
         0.013721, -0.029879, -0.015928,
        -0.031230, -0.013414, -0.012528,
         0.042512,  0.030647,  0.008605,
        -0.011444, -0.018858, -0.022785,
         0.032248,  0.032188,  0.019050,
        -0.017852, -0.024517, -0.034113,
        -0.003410, -0.003745, -0.014594,
        -0.031835, -0.029210, -0.012230,
         0.005769,  0.005132, -0.005526,
         0.010815,  0.009622, -0.007613,
         0.026643,  0.016095,  0.031696,
         0.007323, -0.003253, -0.025652,
        -0.046462,  0.009825,  0.027096,
         0.007369,  0.036621,  0.079080,
        -0.014368, -0.001649,  0.007696,
        -0.003393,  0.007498, -0.013362,
        -0.006312,  0.011122,  0.017691,
        -0.012770, -0.018235, -0.025416,
         0.002446,  0.006986,  0.005707,
        -0.030742, -0.027469, -0.011763,
         0.003682,  0.008116,  0.004298,
        -0.032977, -0.006787, -0.030046,
         0.085405,  0.102019,  0.117396,
         0.007967,  0.013688,  0.014754,
        -0.007916,  0.014210,  0.020987,
         0.020394,  0.011857,  0.007694,
         0.000956,  0.004952,  0.020765,
         0.009696,  0.002855,  0.002099,
         0.002041,  0.000370,  0.000239,
         0.134764,  0.071739,  0.060966,
        -0.015617, -0.022160, -0.017979,
         0.045418,  0.019079,  0.002842,
         0.015152,  0.024388,  0.029838,
         0.025830,  0.023645,  0.019045,
         0.011731, -0.058613, -0.126899,
        -0.099453, -0.034334, -0.035551,
         0.018037,  0.007495,  0.007787,
         0.029511,  0.089145,  0.053372,
        -0.018908, -0.014312, -0.020246,
         0.030131, -0.000707,  0.000467,
         0.042585,  0.024022,  0.012312,
         0.050507, -0.036655, -0.087090,
         0.031318,  0.029243,  0.024195,
        -0.031438,  0.016232, -0.025503,
         0.027991,  0.016405,  0.001808,
        -0.025911, -0.028287, -0.039083,
        -0.020797, -0.002922, -0.010825,
        -0.014713, -0.004621,  0.004488,
         0.019820,  0.008262,  0.006191,
         0.002690,  0.007281,  0.007487,
         0.009711,  0.005954, -0.001892,
         0.021673,  0.019519,  0.025297
    ], bias: [-0.102026, -0.131619, -0.214295])
    static let flux2 = PreviewProjection(weights: [
         0.012039,  0.013185,  0.014823,
         0.017394,  0.014037,  0.016079,
         0.008738,  0.000431,  0.008769,
         0.007372,  0.004446,  0.003329,
        -0.003358, -0.007712,  0.008917,
         0.002088,  0.004858,  0.029830,
         0.000982, -0.009559,  0.012879,
        -0.004556,  0.001899,  0.027746,
        -0.009673, -0.012460, -0.001937,
        -0.003314, -0.006019,  0.005192,
         0.005934,  0.013373,  0.022729,
        -0.005121, -0.003057,  0.004082,
         0.013265,  0.074445,  0.147122,
         0.061592,  0.098769,  0.151296,
         0.053368,  0.101887,  0.147520,
         0.083995,  0.115801,  0.155697,
        -0.010617, -0.019319, -0.002424,
        -0.008136, -0.021525, -0.008612,
         0.012295,  0.008160,  0.010783,
         0.002107, -0.007256, -0.009787,
        -0.001099, -0.011038, -0.015787,
         0.003807,  0.000489,  0.005380,
         0.002687,  0.000291, -0.009021,
        -0.010447, -0.010141, -0.008765,
        -0.003017,  0.027537, -0.013812,
        -0.013704,  0.021581, -0.006332,
        -0.001161,  0.023492, -0.009308,
        -0.007006,  0.022712, -0.006493,
         0.003288,  0.009078,  0.015733,
         0.015861,  0.020151,  0.027680,
        -0.005094,  0.006583,  0.009811,
         0.008753,  0.014363,  0.013580,
        -0.118662, -0.091427, -0.056186,
        -0.109705, -0.075971, -0.021419,
        -0.105436, -0.074151, -0.019008,
        -0.073283, -0.020160,  0.046945,
         0.011946,  0.008353,  0.023407,
         0.002681, -0.006674,  0.002445,
        -0.002718, -0.010409, -0.002675,
         0.009655,  0.005607,  0.007477,
        -0.003563,  0.006630,  0.010357,
         0.000715,  0.003442,  0.024233,
        -0.003479,  0.006899,  0.012019,
        -0.001652,  0.002501,  0.006398,
        -0.003802, -0.009745, -0.027312,
         0.000522, -0.005941, -0.022291,
        -0.000761,  0.003582, -0.014108,
         0.004263,  0.002201, -0.018266,
         0.006126,  0.016088,  0.003986,
        -0.002876,  0.007290,  0.014442,
         0.006014,  0.006183,  0.000606,
         0.000223,  0.006611,  0.013410,
        -0.000177, -0.008404, -0.016749,
        -0.010178, -0.004104, -0.003785,
        -0.010873, -0.009270, -0.002606,
         0.001511, -0.001036, -0.003428,
        -0.020672, -0.017721, -0.025952,
        -0.008500, -0.014159, -0.002957,
         0.005189, -0.003396, -0.007665,
        -0.001559, -0.008642, -0.014890,
        -0.040115,  0.002848,  0.010488,
        -0.026464,  0.015025,  0.019404,
        -0.035352, -0.000215,  0.009546,
        -0.037458,  0.005339,  0.014309,
        -0.007867, -0.009234, -0.010695,
         0.007777,  0.000666, -0.007164,
         0.007884, -0.001978, -0.001229,
         0.018298,  0.030277,  0.036353,
        -0.006289, -0.014828, -0.003231,
         0.005493, -0.001595, -0.004100,
         0.000796,  0.002294,  0.000727,
         0.003865,  0.003418,  0.006972,
         0.002037,  0.004675,  0.018571,
        -0.000297, -0.002888, -0.002523,
         0.006159,  0.007264, -0.003530,
         0.000568,  0.016492,  0.006758,
         0.038578, -0.020335, -0.010923,
         0.028503, -0.025675, -0.004049,
         0.047943, -0.011020,  0.012667,
         0.040783, -0.013333,  0.008124,
         0.006236,  0.006929,  0.001601,
        -0.006720, -0.002693, -0.005612,
        -0.008758, -0.006993,  0.001386,
        -0.001700,  0.003717,  0.001454,
         0.005342,  0.004350,  0.011380,
        -0.001932, -0.004241,  0.004630,
        -0.000228, -0.002034, -0.005768,
        -0.005293, -0.007131, -0.007435,
        -0.003095,  0.004966, -0.010250,
        -0.003951,  0.003458, -0.005340,
        -0.002791,  0.004469,  0.000329,
        -0.011602,  0.000453, -0.004343,
         0.007230,  0.005053, -0.002592,
         0.016581,  0.007996,  0.019318,
        -0.003924, -0.009849, -0.004786,
        -0.008543, -0.012639, -0.002837,
        -0.015494, -0.000199, -0.002232,
        -0.002286, -0.004745, -0.016252,
        -0.021645, -0.000860, -0.004665,
         0.009269, -0.004007, -0.020298,
        -0.003549,  0.009504,  0.010507,
        -0.005931,  0.006491,  0.009099,
         0.011633,  0.029467,  0.031771,
         0.002734,  0.013202,  0.018982,
         0.003741, -0.004809,  0.001664,
         0.004810, -0.012003, -0.005377,
         0.006731, -0.005821,  0.009706,
         0.007721, -0.008420, -0.004096,
         0.009793,  0.011186,  0.018376,
         0.013965,  0.016478,  0.035518,
         0.018572,  0.010814,  0.022926,
         0.010717,  0.023656,  0.031846,
        -0.007693, -0.002568,  0.000978,
        -0.021172, -0.007422, -0.006695,
        -0.010291, -0.007543, -0.013182,
        -0.010662, -0.009879, -0.016291,
        -0.006437, -0.001836,  0.001977,
        -0.016990, -0.004370, -0.000899,
        -0.024690, -0.004641,  0.002021,
        -0.019858, -0.014403, -0.006667,
        -0.001123,  0.001152, -0.005337,
         0.002064,  0.006639, -0.007181,
         0.008010,  0.009261, -0.000765,
         0.007769,  0.000299, -0.000651,
         0.013287,  0.004057, -0.001464,
        -0.005048, -0.011306, -0.014555,
         0.006663,  0.005667, -0.007448,
         0.010945,  0.005011, -0.004665
    ], bias: [-0.066671, -0.087091, -0.170948])
    // END COEFFICIENTS
}
