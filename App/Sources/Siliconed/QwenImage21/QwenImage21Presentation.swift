// What the interface says of Qwen-Image-2.1: English sentences, the keys of the app's catalog (`tools/translations.sh` reads them here).

import Foundation

extension FamilyPresentation {
    package static let qwenImage21 = FamilyPresentation(
        symbol: "wand.and.stars",
        blurb: "Generates, and edits a photo by instruction: “change her jacket to red”.",
        details: [.standard: "The image model and the text encoder in the publisher’s precision (bf16), downloaded from Comfy-Org’s single files; the VAE and the tokenizer from Qwen’s repository.",
                  .compact: "8-bit weights published by Comfy-Org: the image model and the text encoder (int8 convrot). VAE and tokenizer as in the Standard version."],
        licenseName: "Qwen Research (non-commercial)")
}

extension PreviewProjection {
    /// **The 64 channels to RGB** (`PreviewProjection`), fitted by the fitting script: seven pairs at ~512² (the
    /// generation in 6 and 9 steps, two edits, the encoder's two references — fitted on goldens of the base model under
    /// the third-party turbo LoRA the family carried before the official Turbo; the latent space and the VAE are the
    /// same, the Turbo's VAE being the base's rounded to bf16, so the fit was not redone), the latent *before* its
    /// per-channel denormalization, RGB of the RGBA decode. One pair held out: 26.2 to 29.1 dB — optimistic, the pairs
    /// go by twos; both generations held out together, judged on them: **23.9 dB**. ComfyUI publishes nothing for this
    /// space.
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
}
