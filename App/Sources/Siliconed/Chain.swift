import Foundation
import os

// **The API: closed modules, typed wires, a chain that refuses to be wired wrongly.**
//
// A render is always the same sequence of stages — text, [image encoding,] denoising, decoding —
// and what distinguishes one model from another is *which* module occupies each stage. This
// file names the stages (four protocols), what flows between them (`TextFormat`, `Latent`,
// `ImageRGB`) and the wiring rule: **two modules connect only if the wire is of the same type**.
// Text encoded for Anima does not condition Krea 2's DiT, a Flux VAE latent is not decoded with
// Qwen-Image's — and `Chain` refuses it at construction, not after a minute of encoder.
//
// The system is **closed** (the Reason vision, not ComfyUI): the modules are the repository's,
// ported and verified here. The protocols are open to the package so a caller can compose its chains, not so
// it can write new links.
//
// What does not change, and must be read as a constraint: **each stage lives in its own scope**.
// A module builds its model, computes, and returns its memory on exit; the next stage begins only
// afterwards. That is what keeps a render under the machine's budget.

// ── the wires ──────────────────────────────────────────────────────────────────────────────

/// **What a text encoder produces, and therefore what a denoiser accepts.** Two formats are equal
/// only if they have the same name: the name describes the encoder, the layer taken and the shape.
package struct TextFormat: Hashable, Sendable, CustomStringConvertible {
    package let name: String
    /// **The text stage reads the reference images too** (`ImageTextModule`): the encoder sees them
    /// with the prompt, and the denoiser expects the slots they left in the sequence.
    package let readsImages: Bool
    package init(_ name: String, readsImages: Bool = false) { self.name = name; self.readsImages = readsImages }
    package var description: String { name }
}

/// **A latent's space**: the VAE that defined it. The same number of channels does not mean the
/// same space — Flux and Qwen-Image both have sixteen channels and do not decode each other.
package struct LatentSpace: Hashable, Sendable, CustomStringConvertible {
    package let name: String
    package let channels: Int
    /// Pixels per latent cell, along one side.
    package let factor: Int
    /// The projection a step's preview draws the latent with (`PreviewProjection`), fitted on this space.
    package let preview: PreviewProjection
    package var description: String { name }
}

/// A planar latent `[channels, height, width]`, and the space that gives it meaning.
package struct Latent: Sendable {
    package let space: LatentSpace
    package let height: Int
    package let width: Int
    package var values: [Float]

    package init(space: LatentSpace, height: Int, width: Int, values: [Float]) {
        self.space = space; self.height = height; self.width = width; self.values = values
    }

    /// The starting noise: **`torch.randn` on the CPU, to the bit** (`TorchNoise`), for the five families, so a seed
    /// gives diffusers' image. It is drawn in planar order `[channels, height, width]`, which is the order each
    /// pipeline draws its tensor in (`(1, 1, 64, h, w)` Qwen-Image-2.1, `(1, 16, h, w)` Z-Image and Krea 2 — Krea 2
    /// packs it afterwards —, `(1, 16, 1, h, w)` Anima, `(1, 128, h, w)` MOSS-Image): a square yields the same bits as
    /// before rectangles arrived. Every space has at least 16 channels, so a latent always takes the block path.
    package static func noise(_ space: LatentSpace, height: Int, width: Int, seed: UInt64) -> Latent {
        var values = [Float](repeating: 0, count: space.channels * height * width)
        var noise = TorchNoise(seed: seed)
        values.withUnsafeMutableBufferPointer { noise.fill($0.baseAddress!, count: $0.count) }
        return Latent(space: space, height: height, width: width, values: values)
    }
}

/// `[3, height, width]`, planar, in `[-1, 1]` — what `png()` and `cgImage()` read, and what an
/// image encoder receives. Reading a file or a `CGImage` and fitting to the format:
/// `ImageToImage.swift`.
///
/// **An image with transparency carries it twice**: `pixels` is always the image laid over white
/// (what every model but one reads), and `rgba` the source's own bytes, unpremultiplied, alpha
/// included — what Qwen-Image-2.1's edit reads, as diffusers hands its references in RGBA
/// (`QwenImage21DenoisingModule.preparedReferences`). An opaque image has no `rgba`: nothing changes for it.
package struct ImageRGB: Sendable, Equatable {
    package let pixels: [Float]
    package let height: Int
    package let width: Int
    /// `[height, width, 4]` interleaved 8-bit RGBA, **straight** (not premultiplied: a transparent
    /// pixel keeps its colour), when some pixel's alpha is under 255; `nil` for an opaque image.
    package let rgba: [UInt8]?

    package init(pixels: [Float], height: Int, width: Int) {
        self.init(pixels: pixels, height: height, width: width, rgba: nil)
    }

    package init(pixels: [Float], height: Int, width: Int, rgba: [UInt8]?) {
        precondition(height > 0 && width > 0, "ImageRGB: \(width)×\(height), an empty image")
        precondition(pixels.count == 3 * height * width,
                     "ImageRGB: \(pixels.count) values for 3 × \(height) × \(width)")
        precondition(rgba.map { $0.count == 4 * height * width } ?? true, "ImageRGB: RGBA of another size")
        self.pixels = pixels; self.height = height; self.width = width; self.rgba = rgba
    }
}

/// What a denoiser announces before starting: where it starts, how many evaluations.
package struct DenoisingPlan: Sendable {
    /// The first step executed of the full schedule: 0 in txt2img, `t_start` in img2img.
    package let startStep: Int
    /// The starting σ, `σ[startStep]`: 1 in txt2img.
    package let startSigma: Float
    /// The evaluations actually paid for, from the starting step to the end.
    package let evaluations: Int
}

// ── the modules ─────────────────────────────────────────────────────────────────────────────

/// The text stage: a prompt goes in, a text of its `output` format comes out. It returns its memory on exit.
package protocol TextModule: Sendable {
    /// What an app displays on the module's box ("Qwen3-VL-4B").
    var name: String { get }
    var output: TextFormat { get }
}

/// **A text stage that sees the reference images** (Qwen-Image-2.1's Qwen3-VL): the prompt and the
/// references go in together — already at the denoiser's size (`DenoisingModule.preparedReferences`),
/// the very copy the image encoder then reads. Without references, it is an ordinary text stage.
package protocol ImageTextModule: TextModule {
}

/// The denoising stage: it turns noise — or a noised image — into a latent, under a text of its
/// format.
///
/// **The starting step** (`start`) is the whole of img2img on the denoiser's side: it unrolls its
/// full schedule, computed for `step` (shift and μ included), but executes only steps
/// `start … step − 1`. The latent it receives is already noised to `σ[start]` (`Latent.start`).
package protocol DenoisingModule: Sendable {
    /// The identifier of the model whose DiT it carries (`z-image`, `anima`, `krea2`), as a render and a refusal name it.
    var model: String { get }
    var name: String { get }
    var entry: TextFormat { get }
    var space: LatentSpace { get }
    var defaultSteps: Int { get }
    /// The full schedule of `step` steps: `step + 1` σ values, ending in 0. It does not depend on
    /// the strength — img2img slices it, it does not recompute it.
    func sigmas(steps: Int) -> [Float]
    /// `height`, `width`: the latent grid, not the pixels.
    func plan(height: Int, width: Int, steps: Int, start: Int) -> DenoisingPlan
    /// **How many reference images it can read** (editing): 0 for a denoiser
    /// without. A reference is not a starting point (img2img): it enters the sequence, next to the
    /// generated image, and the denoiser re-reads it at each layer.
    var maxReferences: Int { get }
    /// **The reference images as this denoiser reads them**, prepared once for the whole render: the
    /// text stage (if it reads images) and the image encoder receive this same copy. By default,
    /// as given (a denoiser without references never receives any).
    func preparedReferences(_ images: [ImageRGB]) throws -> [ImageRGB]
    /// **The format of an edit whose request names none**, from its first reference's size. By
    /// default the reference's own size under 1024² (its ratio kept), raised to the floor and kept
    /// under the area cap.
    func editFormat(referenceWidth: Int, referenceHeight: Int) -> (width: Int, height: Int)
    /// **What the denoiser refuses before any computation**: a step count it has no schedule for, a
    /// starting image (img2img) when its pipeline has none. By default, nothing.
    func check(steps: Int, startImage: Bool) throws
    /// Whether the denoiser starts from an encoded image (img2img). By default yes; a pipeline
    /// without img2img (Qwen-Image-2.1) says no, and the chain then offers no image entry.
    var acceptsStartImage: Bool { get }
    /// The update rules its sampler walks (`Solver`). By default Euler alone: the chain refuses
    /// another before any computation rather than render Euler under another name.
    var solvers: [Solver] { get }
    /// Where its Detail's bell sits and how much it lies (`Detail.Bell`), calibrated on its own schedule. `nil`: it
    /// offers no Detail, and the chain refuses More and Most.
    var detailBell: Detail.Bell? { get }
}

/// The image encoding stage (img2img): an image in the render's format goes in, a latent of its
/// space comes out — normalized as the denoiser expects. It returns its memory on exit.
package protocol ImageEncodingModule: Sendable {
    var name: String { get }
    var space: LatentSpace { get }
}

/// The decoding stage: a latent of its space goes in, an image comes out.
package protocol DecodingModule: Sendable {
    var name: String { get }
    var space: LatentSpace { get }
}

package extension DenoisingModule {
    var maxReferences: Int { 0 }
    func preparedReferences(_ images: [ImageRGB]) throws -> [ImageRGB] { images }
    func editFormat(referenceWidth: Int, referenceHeight: Int) -> (width: Int, height: Int) {
        // The area brought under 1024² at the reference's ratio, each side floored to the multiple.
        let m = Format.multiple
        let scale = min(1, (Double(1024 * 1024) / Double(referenceWidth * referenceHeight)).squareRoot())
        var (w, h) = (max(Format.minimumSide, Int(Double(referenceWidth) * scale) / m * m),
                      max(Format.minimumSide, Int(Double(referenceHeight) * scale) / m * m))
        // Raising the short side to the floor can push a long strip past the edit's area (10:1 gave
        // 3232×512): the long side gives way, as in Qwen-Image-2.1's rule.
        if w * h > Format.editSurface {
            if w >= h { w = Format.editSurface / h / Format.multiple * Format.multiple }
            else { h = Format.editSurface / w / Format.multiple * Format.multiple }
        }
        return (w, h)
    }
    func check(steps: Int, startImage: Bool) throws {}
    var acceptsStartImage: Bool { true }
    var solvers: [Solver] { [.euler] }
    var detailBell: Detail.Bell? { nil }
    /// Whether it honours `Request.detail`.
    var acceptsDetail: Bool { detailBell != nil }
    /// The step counts there is a schedule for (`check(steps:)`): what `--steps` takes, and what a lender publishes.
    var scheduledSteps: [Int] {
        (1...Request.maximumSteps).filter { (try? check(steps: $0, startImage: false)) != nil }
    }
}

/// A module without a name of its own carries its type's: a hand-composed chain can still be
/// drawn.
package extension TextModule { var name: String { String(describing: Self.self) } }
package extension DenoisingModule { var name: String { String(describing: Self.self) } }
package extension ImageEncodingModule { var name: String { String(describing: Self.self) } }
package extension DecodingModule { var name: String { String(describing: Self.self) } }

// ── the chain ───────────────────────────────────────────────────────────────────────────────

/// **Three modules wired, plus the image encoder if there is one — and verified wirable.**
///
/// The encoder is optional **in the type** (a chain without it only does txt2img) and present in
/// every `Model` of the catalog. Its wire is the same as the decoder's: it must speak the
/// denoiser's space.
package struct Chain: Sendable {
    package let text: any TextModule
    package let denoising: any DenoisingModule
    package let decoding: any DecodingModule
    package let encoding: (any ImageEncodingModule)?

    package enum Failure: Error, CustomStringConvertible {
        case incompatibleText(output: TextFormat, entry: TextFormat)
        case latentIncompatible(output: LatentSpace, entry: LatentSpace)
        case imageIncompatible(output: LatentSpace, entry: LatentSpace)
        case textWithoutImages(TextFormat)
        package var description: String {
            switch self {
            case let .incompatibleText(output, entry):
                return "wire refused: the text comes out as « \(output) », the denoiser expects « \(entry) »"
            case let .textWithoutImages(format):
                return "wire refused: « \(format) » reads the reference images, and the text module does not take them"
            case let .latentIncompatible(output, entry):
                return "wire refused: the denoiser returns a latent of \(output), the decoder expects \(entry)"
            case let .imageIncompatible(output, entry):
                return "wire refused: the image encoder returns a latent of \(output), the denoiser expects \(entry)"
            }
        }
    }

    package init(text: any TextModule, denoising: any DenoisingModule,
                decoding: any DecodingModule, encoding: (any ImageEncodingModule)? = nil) throws(EngineError) {
        guard text.output == denoising.entry else {
            throw EngineError(Failure.incompatibleText(output: text.output, entry: denoising.entry))
        }
        // A format that reads images must come from a module that takes them: the denoiser would
        // otherwise find no slot for its references.
        guard !text.output.readsImages || text is any ImageTextModule else {
            throw EngineError(Failure.textWithoutImages(text.output))
        }
        guard denoising.space == decoding.space else {
            throw EngineError(Failure.latentIncompatible(output: denoising.space, entry: decoding.space))
        }
        if let encoding, encoding.space != denoising.space {
            throw EngineError(Failure.imageIncompatible(output: encoding.space, entry: denoising.space))
        }
        self.text = text; self.denoising = denoising; self.decoding = decoding
        self.encoding = encoding
    }

    /// **Whether the chain starts from an image** (img2img): an image encoder, and a denoiser that starts from one. What an
    /// app offers a starting image for; an editing reference is the denoiser's (`DenoisingModule.maxReferences`).
    package var acceptsStartImage: Bool { encoding != nil && denoising.acceptsStartImage }
}
