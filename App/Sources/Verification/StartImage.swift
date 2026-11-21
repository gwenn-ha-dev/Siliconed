import Foundation
import Siliconed

// What every family that starts from an image is judged on, the same way (`<family> img2img`):
//
//     picture ─ the VAE's encoder ─ z₀          by the rule against diffusers encoding in bf16; an f32 encoder per plane, at the floor
//     strength ─ the first step run, σ          exact
//     σ_s · ε + (1 − σ_s) · z₀                  bit for bit, from the reference's own z₀ and noise
//     the app's render from the picture         finite, the count of evaluations; its PSNR said, not judged

/// The reference's img2img, read, and its exact pieces judged.
struct StartImageReference {
    let picture: ImageRGB
    let strength: Double
    let steps: Int
    /// The first step run.
    let step: Int
    let seed: UInt64
    let latentHeight, latentWidth: Int
    /// The reference's final latent, in the family's latent planes.
    let final: [Float]

    /// Reads `stem`'s golden, encodes its picture with `encode` and judges z₀ (against `witness`, diffusers in bf16; an
    /// encoder in f32 has none and is judged per plane against the reference itself), the schedule, the first step and
    /// the starting latent.
    init(_ report: Report, _ invocation: Invocation, stem: String, witness: (file: String, script: String)?, space: LatentSpace,
         schedule: (Int) -> [Float], encode: (ImageRGB) throws -> [Float]) throws {
        let reference = try invocation.goldens.open("\(stem).safetensors")
        let half = try witness.map { try invocation.goldens.half("\($0.file).safetensors", script: $0.script) }
        let manifest = try invocation.goldens.manifest(stem)
        guard let entry = reference.entries["image"], let planes = reference.entries["z0"]?.shape[0], let image = reference.float32("image"), let z0 = reference.float32("z0"),
              let eps = reference.float32("eps"), let start = reference.float32("x_s"), let sigmas = reference.float32("sigmas"),
              let first = reference.float32("t_start")?.first, let final = reference.float32("latent_final"),
              let strength = manifest["strength"] as? NSNumber, let seed = manifest["seed"] as? NSNumber else {
            throw Skip("\(reference.path): img2img's inputs are missing")
        }
        let (height, width) = (entry.shape[1], entry.shape[2])
        picture = ImageRGB(pixels: Array(image), height: height, width: width)
        (latentHeight, latentWidth) = (height / space.factor, width / space.factor)
        steps = sigmas.count - 1
        let own = schedule(steps)
        report.add(Judge.schedule(own, sigmas))
        self.strength = strength.doubleValue
        step = Strength.startStep(steps: steps, strength: strength.doubleValue)
        report.add(Judge.holds("first step at strength \(strength)", step == Int(first), note: "\(step), the reference's \(Int(first))"))
        let mixed = Latent.start(image: Latent(space: space, height: latentHeight, width: latentWidth, values: Array(z0)),
                                 noise: Latent(space: space, height: latentHeight, width: latentWidth, values: Array(eps)), sigma: own[step])
        report.add(Judge.identical("starting latent", mixed.values, start))
        // The encoder last: the exact pieces above are judged whatever it computes on.
        if let half {
            report.add(reference, "z0", try encode(picture), half: half.float32("z0").map(Array.init), columns: latentHeight * latentWidth)
        } else {
            report.add(Judge.planes("z0", engine: try encode(picture), oracle: z0, planes: planes))
        }
        self.seed = seed.uint64Value
        self.final = Array(final)
    }

    /// The request the app would render: the reference's picture, strength, seed and steps.
    func request(_ prompt: String) -> Request {
        Request(prompt, width: picture.width, height: picture.height, seed: seed, steps: steps, image: picture, strength: strength)
    }

    /// The app's render from the picture against `decoded`, the reference's final latent decoded by the engine.
    func judge(_ report: Report, _ render: Engine.Render, decoded: [Float]) {
        let (psnr, _, _, notFinite) = decoded.withUnsafeBufferPointer { imageError(render.image.pixels, $0) }
        report.add(Judge.holds("render from the picture", notFinite == 0 && render.evaluations == steps - step,
                               note: String(format: "\(render.evaluations) evaluations, finite; %.2f dB from the reference's final latent decoded here, said and not judged", psnr)))
    }
}

/// The reference's final latent through the Qwen-Image VAE's decoder (Anima's and Krea 2's), as the render decodes.
func qwenImageDecoded(_ final: [Float], height h: Int, width w: Int, vae path: String) throws -> [Float] {
    try DecoderSession<QwenImageVAEDecoder>(path: path, latentHeight: h, latentWidth: w).pixels(QwenImageVAEDecoder.normalized(final))
}
