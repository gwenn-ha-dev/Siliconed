import Foundation
import Siliconed

// **Qwen-Image-2.1's VAE and sampler, stage by stage**: `vae-encode`, `vae`, `trajectory`.
extension QwenImage21Stages {
    static let encoding = Stage(
        model: model, name: "vae-encode",
        proves: "the VAE's encoder in bf16 on the oracle's two reference pictures, RGBA: the latent's mean, then the latent normalized in bf16 as the pipeline hands it to the DiT",
        oracle: "tools/oracle/qwen-image-2.1.py vae-encode; vae-encode-bf16", weights: ["VAE"],
        rule: "per latent plane: the rule against diffusers encoding in bf16"
    ) { invocation in
        let report = Report(title: "\(model) vae-encode \(resolution)")
        let goldens = try invocation.goldens.open("qwen21/goldens-qwen21-vae-encode-\(resolution).safetensors")
        let half = try invocation.goldens.half("qwen21/witness-bf16-qwen21-vae-encode-\(resolution).safetensors",
                                               script: "tools/oracle/qwen-image-2.1.py vae-encode-bf16")
        let path = Library.processWide.map(vaeMap)
        let channels = QwenImage21VAEDecoder.latentChannels
        for i in 0..<2 {
            guard let entry = goldens.entries["image_\(i)"], let image = goldens.float32("image_\(i)") else {
                throw Skip("\(goldens.path): image_\(i) missing")
            }
            let mean = try QwenImage21VAEEncoder.mean(of: Array(image), height: entry.shape[1], width: entry.shape[2], path: path)
            for (name, values) in [("latent_mean_raw_\(i)", mean), ("latent_\(i)", QwenImage21VAEEncoder.normalized(mean))] {
                guard let oracle = goldens.float32(name) else { report.add(Judge.missing(name)); continue }
                // Planes to tokens: a plane is then a column, which `Judge.boundary` judges one by one.
                func columns(_ planes: [Float]) -> [Float] { QwenImage21Render.tokens(planes: planes, channels: channels) }
                let expected = columns(Array(oracle))
                report.add(expected.withUnsafeBufferPointer {
                    Judge.boundary(name, engine: columns(values), oracle: $0, half: half.float32(name).map { columns(Array($0)) }, columns: channels)
                })
            }
        }
        return report
    }

    // MARK: - The decoder

    static let vae = Stage(
        model: model, name: "vae",
        proves: "the decoder in bf16 on the final latent of the oracle's trajectory, de-normalized in bf16 as the pipeline does: the RGBA image plane by plane",
        oracle: "tools/oracle/qwen-image-2.1.py vae; vae-witness; vae-bf16", weights: ["VAE"],
        rule: "per plane, at the fp64 witness: no further from it than diffusers decoding in bf16, + 25 %"
    ) { invocation in
        let report = Report(title: "\(model) vae \(invocation.size)")
        let goldens = try invocation.goldens.open("qwen21/goldens-qwen21-vae-\(invocation.size).safetensors")
        let witness = invocation.goldens.optional("qwen21/witness-qwen21-vae-f64-\(invocation.size).safetensors")
        let half = try invocation.goldens.half("qwen21/witness-bf16-qwen21-vae-\(invocation.size).safetensors",
                                               script: "tools/oracle/qwen-image-2.1.py vae-bf16")
        guard let entry = goldens.entries["vae_in"], let tokens = goldens.float32("latent_final"),
              let expected = goldens.float32("image") else {
            throw Skip("\(goldens.path): latent_final or image missing")
        }
        let (channels, h, w) = (entry.shape[0], entry.shape[1], entry.shape[2])
        // `_unpack_latents`: `[h·w, 64]` to planes, a plain transposition.
        let given = QwenImage21VAEDecoder.denormalized(QwenImage21Render.planes(tokens: Array(tokens), channels: channels))
        let pixels = try DecoderSession<QwenImage21VAEDecoder>(path: Library.processWide.map(vaeMap), latentHeight: h, latentWidth: w).pixels(given)
        report.add(Judge.planes("image RGBA", engine: pixels, oracle: expected, witness: witness?.float32("image"),
                                reference: half.float32("image").map(Array.init), planes: 4))
        return report
    }

    // MARK: - The sampler

    static let trajectory = Stage(
        model: model, name: "trajectory",
        proves: "the seed's noise bit for bit, the published 8-step σ schedule, then each step's velocity from the reference's own latent at that step, by the render's DiT session: the first step the whole sequence, the others the target over the kept keys and values (with --edit N, N reference pictures in the sequence)",
        options: ["--edit", "--map"], variants: edits,
        oracle: "tools/oracle/qwen-image-2.1.py trajectory [--edit N]; trajectory-bf16; with --edit N, trajectory-steps-bf16 --edit N",
        weights: ["DiT"],
        rule: "the noise and σ exact; each step by the rule against diffusers' same step in bf16"
    ) { invocation in
        let n = invocation.edit ?? 0
        // An edit's size is the oracle's (its last picture's ratio), not `--size`.
        let tag = n > 0 ? "-edit\(n)-\(resolution)" : "-\(invocation.size)", edit = n > 0 ? " --edit \(n)" : ""
        let report = Report(title: "\(model) trajectory\(n > 0 ? " edit \(n) \(resolution)" : " \(invocation.size)")")
        let path = try invocation.ditMap(ditMap)
        let stem = "qwen21/goldens-qwen21-trajectory\(tag)"
        let reference = try invocation.goldens.open(stem + ".safetensors")
        let info = try invocation.goldens.manifest(stem)
        let layout = try invocation.goldens.open("qwen21/goldens-qwen21-dit\(tag).safetensors")
        let shapes = (try invocation.goldens.manifest("qwen21/goldens-qwen21-dit\(tag)"))["img_shapes"] as? [[Int]]
        let text = try invocation.goldens.open((n > 0 ? "qwen21/goldens-qwen21-text-image-\(n)" : "qwen21/goldens-qwen21-text") + ".safetensors")
        // A generation's bf16 steps are written beside its free run; an edit's by a stage of their own.
        let half = n > 0
            ? try invocation.goldens.half("qwen21/witness-bf16-qwen21-trajectory-steps\(tag).safetensors",
                                          script: "tools/oracle/qwen-image-2.1.py trajectory-steps-bf16\(edit)")
            : try invocation.goldens.half("qwen21/witness-bf16-qwen21-trajectory\(tag).safetensors", script: "tools/oracle/qwen-image-2.1.py trajectory-bf16")
        guard let tokens = reference.float32("latent_init"), let sigmas = reference.float32("sigmas"),
              let seed = info["seed"] as? NSNumber, let height = info["height"] as? Int, let width = info["width"] as? Int,
              let shapes, let mask = layout.int32("img_mask"),
              let embeds = text.float32("prompt_embeds"), let embedsEntry = text.entries["prompt_embeds"] else {
            throw Skip("\(reference.path): the trajectory's inputs are missing")
        }
        let space = LatentSpace.qwenImage21, (h, w) = (height / space.factor, width / space.factor), plane = h * w
        // Drawn `(1, 1, 64, h, w)`, recorded packed `[h·w, 64]`.
        report.add(QwenImage21Render.planes(tokens: Array(tokens)).withUnsafeBufferPointer {
            Judge.identical("noise (seed \(seed))", Latent.noise(space, height: h, width: w, seed: seed.uint64Value).values, $0)
        })
        report.add(Judge.schedule(QwenImage21DiT.sigmas, sigmas))
        guard tokens.count == plane * space.channels else { throw Skip("\(reference.path): latent_init is not the target's \(h)×\(w)") }

        // Then each step from the reference's own latent, as the render runs them: one session, the first step filling
        // the conditions' keys and values, the others reading them.
        let slots = mask.prefix(embedsEntry.shape[0]).map { $0 != 0 }
        let sequence = try QwenImage21DiT.Sequence(slots: slots, references: shapes.dropLast().map { (height: $0[1], width: $0[2]) }, height: h, width: w)
        let context = embedsEntry.shape[1]
        let spoken = QwenImage21Render.spoken(HostTensor([slots.count, context], floats: Array(embeds.prefix(slots.count * context))), slots: slots).floats
        var references: HostTensor?
        if n > 0 {
            guard let latents = reference.float32("reference_latents"), latents.count == sequence.referenceRows * space.channels else {
                throw Skip("\(reference.path): reference_latents is not the sequence's references")
            }
            references = HostTensor([sequence.referenceRows, space.channels], floats: Array(latents))
        }
        let denoiser = try QwenImage21Render.Denoiser(map: path, sequence: sequence)
        try denoiser.prompt(text: HostTensor([sequence.textRows, context], floats: spoken), references: references)
        // Token-major, as the oracle writes it: its 64 channels are the latent's planes.
        // The render's own call (`RenderLoop`), under the reference's Euler (the render's default): Qwen-Image-2.1's
        // derivative is its velocity.
        try report.trajectory(reference, steps: sigmas.count - 1, half: half, columns: space.channels, name: { "velocity_\($0)" }) { step, latent in
            QwenImage21Render.tokens(planes: try denoiser.derivative(QwenImage21Render.planes(tokens: latent), sigma: QwenImage21DiT.sigmas[step],
                                                                     solver: .euler, first: step == 0))
        }
        return report
    }
}
