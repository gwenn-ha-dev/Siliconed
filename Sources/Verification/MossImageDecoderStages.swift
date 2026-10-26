import Foundation
import Siliconed

// **MOSS-Image's decoder, stage by stage** — `siliconed check moss-image vae`.
//
// What sets the decoder apart: FLUX.2's autoencoder (32 channels, patch 2×2), whose latent the PIPELINE
// de-normalizes with a BatchNorm's running statistics per patched channel and de-patchifies before the
// decoder. Two goldens at 512²: the encoder-side one (an image of the oracle's inputs, encoded by the
// reference, so the decoder is judged on a latent the DiT never made) and the final one (the reference
// pipeline's own last latent and image — the decoder's real input). The VAE is the installed family's
// (`store/composants/moss-image/vae.safetensors`).

enum MossImageDecoderStages {
    static let all: [Stage] = [vae]

    static let model = MossImageStages.model

    static let vae = Stage(
        model: model, name: "vae",
        proves: "FLUX.2's decoder: the pipeline's de-normalization and 2×2 de-patchification, then the image, plane by plane — on an encoded test image's latent and on the final latent of the reference pipeline",
        sizes: ["512"], oracle: "tools/oracle/moss-image.py vae --size {size}; vae-witness --size {size}; vae-final --size {size}", weights: ["VAE"],
        rule: "the de-patchified latent per channel, floor 5e-4; the image per plane, floor 5e-4, at the fp64 witness (no floor there)"
    ) { invocation in
        let report = Report(title: "\(model) vae \(invocation.size)")
        let path: String
        do { path = try Library.processWide.component(.mossImage, "vae.safetensors") } catch {
            throw Skip("VAE missing: siliconed install moss-image")
        }
        let witness = invocation.goldens.optional("moss/witness-moss-vae-f64-\(invocation.size).safetensors")
        var judged = 0
        for (stem, label) in [("moss/goldens-moss-vae-\(invocation.size)", "encoded test image"),
                              ("moss/goldens-moss-vae-final-\(invocation.size)", "final latent")] {
            guard let goldens = invocation.goldens.optional(stem + ".safetensors") else { continue }
            // `latent`: the DiT's, `[128, h/2, w/2]` (a leading 1 allowed); `latent_z`: the decoder's input.
            guard let entry = goldens.entries["latent"], entry.shape.count >= 3, let latent = goldens.float32("latent"),
                  let expected = goldens.float32("image") else { continue }
            let (h, w) = (2 * entry.shape[entry.shape.count - 2], 2 * entry.shape[entry.shape.count - 1])
            let decoder = try Flux2VAE(path: path, latentHeight: h, latentWidth: w)
            if let z = goldens.float32("latent_z") {
                let stage = try decoder.pipelinePart(latent: Array(latent))
                report.add(Judge.boundary("latent_z (\(label))", engine: stage, oracle: z, columns: h * w))
            }
            let (image, _) = try decoder.decode(latent: latent.baseAddress!)
            // The witness is the decoder in double on the encoded test image's latent alone.
            report.add(Judge.planes("image RGB (\(label))", engine: image, oracle: expected,
                                    witness: stem.contains("final") ? nil : witness?.float32("image"), planes: 3))
            judged += 1
        }
        guard judged > 0 else { throw Skip("golden missing: \(invocation.goldens.path("moss/goldens-moss-vae-\(invocation.size).safetensors"))") }
        return report
    }
}
