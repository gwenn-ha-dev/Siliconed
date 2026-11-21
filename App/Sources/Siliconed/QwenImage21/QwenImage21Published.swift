import Foundation

/// **Two files of `Qwen/Qwen-Image-2.1-Turbo`, as published**, written here **verbatim**, byte for byte, at revision
/// `d65dbc9a`: what the engine reads of them is a few numbers, and a file downloaded for eight numbers would be one
/// more thing to install. A test hashes each literal as git does (`blob <n>\0` + bytes, SHA-1) and finds the blob id the
/// Hub lists for the file at that revision: the text here *is* the file.
///
///     model_index.json    → `sample_sigmas`, the Turbo's eight σ (`QwenImage21DiT.sigmas`)
///     vae/config.json     → `latents_mean`, `latents_std`, the 64 channels' normalization (`QwenImage21VAEDecoder`)
package enum QwenImage21Published {
    /// The revision these bytes are from, and each file's git blob id there (`/api/models/…/tree`).
    package static let revision = "d65dbc9a7e8f6b5479e33dee6030eaab2a906509"
    package static let blobs: [(path: String, blob: String, text: String)] = [
        ("model_index.json", "cdc1dae2997ff637225c42c388196b38a9bbaae8", modelIndexText),
        ("vae/config.json", "9b780c2dd9601cb0b93ae8f462b58dd3a7468347", vaeConfigText),
    ]

    /// `sample_sigmas`: the schedule's nodes, as written (Python doubles), the terminal σ left out.
    package static let sampleSigmas: [Double] = doubles(modelIndexText, "sample_sigmas")
    /// The VAE's per-channel mean and standard deviation of the latents.
    package static let latentsMean: [Double] = doubles(vaeConfigText, "latents_mean")
    package static let latentsStd: [Double] = doubles(vaeConfigText, "latents_std")

    private static func doubles(_ text: String, _ key: String) -> [Double] {
        // The literals are fixed and tested: a parse failure is a programming error.
        (try! OrderedJSON.parse(Data(text.utf8)))[key]?.elements?.compactMap(\.double) ?? []
    }

    static let modelIndexText = #"""
        {
          "_class_name": "QwenImage21Pipeline",
          "_diffusers_version": "0.41.0.dev0",
          "processor": [
            "transformers",
            "Qwen3VLProcessor"
          ],
          "scheduler": [
            "diffusers",
            "FlowMatchEulerDiscreteScheduler"
          ],
          "text_encoder": [
            "transformers",
            "Qwen3VLForConditionalGeneration"
          ],
          "transformer": [
            "diffusers",
            "QwenImage21Transformer2DModel"
          ],
          "vae": [
            "diffusers",
            "AutoencoderKLQwenImage21"
          ],
          "sample_sigmas": [
            1.0,
            0.978453,
            0.95418,
            0.926626,
            0.89508,
            0.845148,
            0.704534,
            0.414568
          ]
        }

        """#

    static let vaeConfigText = #"""
        {
          "_class_name": "AutoencoderKLQwenImage21",
          "_diffusers_version": "0.41.0.dev0",
          "_name_or_path": "/cpfs02/user/yk475785/opensource/qwen-image2.1-opd0914-diffusers21/vae",
          "attn_scales": [],
          "base_dim": 96,
          "decoder_base_dim": 144,
          "dim_mult": [
            1,
            2,
            4,
            8,
            8
          ],
          "dropout": 0.0,
          "in_channels": 4,
          "is_residual": true,
          "latents_mean": [
            0.5126,
            0.7721,
            -0.0631,
            1.3506,
            -0.7855,
            -2.1025,
            -0.3458,
            1.3722,
            1.8873,
            -1.7177,
            -0.651,
            0.2732,
            0.7562,
            -0.6163,
            -1.0277,
            3.8363,
            2.021,
            0.0472,
            0.932,
            2.0087,
            2.4954,
            -0.1391,
            -1.4249,
            1.8464,
            -0.5236,
            1.2826,
            3.7046,
            -1.3035,
            2.7286,
            -1.4518,
            -1.9036,
            -1.9955,
            -0.0342,
            -1.0265,
            -0.7636,
            3.0555,
            0.0746,
            -3.0751,
            -0.1076,
            1.7376,
            -1.0914,
            -1.9435,
            -0.2784,
            -1.368,
            0.4809,
            -0.4433,
            0.3764,
            0.5729,
            -2.0595,
            1.096,
            -1.326,
            -2.0211,
            -5.0179,
            0.5275,
            4.0162,
            1.8505,
            0.3026,
            1.9373,
            1.4937,
            0.2632,
            0.5547,
            -1.7121,
            -0.1562,
            0.0304
          ],
          "latents_std": [
            3.2001,
            3.2936,
            3.4321,
            3.0091,
            3.1061,
            4.0379,
            4.0705,
            3.791,
            3.0785,
            3.65,
            3.9308,
            3.0904,
            2.8778,
            3.7675,
            3.732,
            5.0756,
            3.2864,
            4.0397,
            3.1317,
            4.0443,
            2.9249,
            3.9454,
            3.0988,
            4.2489,
            3.4896,
            3.8513,
            3.9323,
            3.4719,
            3.7498,
            4.283,
            3.5694,
            4.2467,
            3.9037,
            3.2947,
            5.077,
            3.5075,
            3.27,
            3.4767,
            2.8063,
            5.1125,
            3.5327,
            4.7833,
            3.1286,
            4.1819,
            3.8527,
            3.8312,
            3.5605,
            4.3875,
            3.9624,
            4.0168,
            3.5643,
            4.055,
            5.5614,
            4.2963,
            4.408,
            3.4959,
            3.8747,
            3.7608,
            3.5735,
            3.149,
            3.7662,
            3.6746,
            3.4563,
            3.8161
          ],
          "num_res_blocks": 2,
          "out_channels": 4,
          "patch_size": null,
          "scale_factor_spatial": 16,
          "scale_factor_temporal": 8,
          "temperal_downsample": [
            false,
            true,
            true,
            true
          ],
          "z_dim": 64
        }

        """#
}
