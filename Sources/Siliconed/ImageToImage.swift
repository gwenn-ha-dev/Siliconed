import Accelerate
import CoreGraphics
import Foundation
import ImageIO

// **img2img: an image comes in with the prompt, and denoising starts from it instead of pure noise.**
//
//     user image ─ ImageRGB(contentsOf:) ─ fitted(width:height:) ─ ImageEncodingModule ─ z₀
//     x = σ_s · ε + (1 − σ_s) · z₀          ε: the seed's noise, σ_s = σ[t_start]
//     denoising of steps t_start … N − 1 of the txt2img schedule, UNCHANGED
//
// The semantics are diffusers' exactly (`strength`, see `Strength`), so that each model stays
// verifiable against an oracle. Two deviations, deliberate and documented where they live: the
// encoder's **mean** rather than a sample (`VAEEncoder`), and the **centered crop** rather than
// stretching (`ImageRGB.fitted`).

// ── the input image ────────────────────────────────────────────────────────────────────────

extension ImageRGB {
    /// What fails when reading an image or writing its PNG.
    package enum Failure: Error, CustomStringConvertible {
        case unreadable(String)
        case encoding
        case writing(String)
        package var description: String {
            switch self {
            case .unreadable(let path): return "unreadable image: \(path) (PNG, JPEG, HEIC… anything ImageIO can read)"
            case .encoding: return "PNG: ImageIO refused to encode the image"
            case .writing(let path): return "PNG: cannot write \(path)"
            }
        }
    }

    /// **The largest side a decode keeps by default: 3072 px**, the longest side an accepted
    /// format can have (`Format.maxSurface / Format.minimumSide`, 1536·1024/512). Beyond
    /// that, the pixels would only be reduced by `fitted` — a 48 Mpx photo decoded at full size
    /// would weigh 576 MB of floats.
    public static let maximumDecodedSide = Format.maxSurface / Format.minimumSide

    /// **An image of any size, read by ImageIO** — PNG, JPEG, HEIC, TIFF… — returned as
    /// `[3, h, w]` in `[-1, 1]` (`2x − 1`, like `VaeImageProcessor.normalize`).
    ///
    /// - it is **downsampled at decode time** by ImageIO if its longest side exceeds `maxSide`
    ///   (proportions kept): the full size never exists in memory;
    /// - the **EXIF orientation is applied** (a portrait phone photo arrives upright): it is
    ///   ImageIO's thumbnail with `CreateThumbnailWithTransform`, forced from the image
    ///   (`FromImageAlways`) and not from the embedded thumbnail;
    /// - the rest is `init(cgImage:)`'s: 8-bit sRGB, transparency over white.
    public init(contentsOf url: URL, maxSide: Int = ImageRGB.maximumDecodedSide) throws(EngineError) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let l = properties[kCGImagePropertyPixelWidth] as? Int,
              let h = properties[kCGImagePropertyPixelHeight] as? Int, l > 0, h > 0 else {
            throw .imageUnreadable(file: url.path)
        }
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                        kCGImageSourceCreateThumbnailWithTransform: true,
                                        kCGImageSourceThumbnailMaxPixelSize: min(max(l, h), max(1, maxSide)),
                                        kCGImageSourceShouldCacheImmediately: true]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw .imageUnreadable(file: url.path)
        }
        // Already within `maxSide` (ImageIO reduced it): the bound is the thumbnail's, not a second one.
        try self.init(cgImage: image, name: url.path, maxSide: max(image.width, image.height, maxSide))
    }

    /// **An image already in memory** — drag-and-drop, pasteboard, `NSImage.cgImage(…)`:
    /// converted to 8-bit **sRGB** (the training images' space), transparency laid **over white**,
    /// then to `[3, h, w]` in `[-1, 1]`. No orientation is applied (a `CGImage` has none).
    ///
    /// **Bounded like a file** (`maxSide`, `maximumDecodedSide` by default): an image whose longest
    /// side exceeds it is drawn **directly** into a smaller context, proportions kept — a 20k × 20k
    /// pasted image would otherwise take 1.6 GB of bytes, then 4.8 GB of floats, before `Request`
    /// fits it to the format. An image within the bound comes out bit for bit as before.
    public init(cgImage image: CGImage, maxSide: Int = ImageRGB.maximumDecodedSide) throws(EngineError) {
        try self.init(cgImage: image, name: "CGImage \(image.width)×\(image.height)", maxSide: maxSide)
    }

    /// **The size an image is drawn at**: as it is when its longest side is within `maxSide`,
    /// otherwise scaled so that it is `maxSide`, each side rounded and at least 1.
    public static func decodedSize(width: Int, height: Int, maxSide: Int = ImageRGB.maximumDecodedSide)
            -> (width: Int, height: Int) {
        let longest = max(width, height), bound = max(1, maxSide)
        guard longest > bound else { return (width, height) }
        let scale = Double(bound) / Double(longest)
        return (max(1, min(bound, Int((Double(width) * scale).rounded()))),
                max(1, min(bound, Int((Double(height) * scale).rounded()))))
    }

    private init(cgImage image: CGImage, name: String, maxSide: Int) throws(EngineError) {
        guard let srgb = CGColorSpace(name: CGColorSpace.sRGB), image.width > 0, image.height > 0 else {
            throw .imageUnreadable(file: name)
        }
        let (width, height) = Self.decodedSize(width: image.width, height: image.height, maxSide: maxSide)
        let reduced = width != image.width || height != image.height
        var bytes = [UInt8](repeating: 255, count: height * width * 4)
        let drawn: Bool = bytes.withUnsafeMutableBytes { raw in
            guard let context = CGContext(data: raw.baseAddress, width: width, height: height,
                                           bitsPerComponent: 8, bytesPerRow: width * 4, space: srgb,
                                           bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            context.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            // At its own size, a copy (`.none`, as always); reduced, a filtered downsampling, as
            // ImageIO's thumbnail does for a file.
            context.interpolationQuality = reduced ? .high : .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw .imageUnreadable(file: name) }
        var planes = [Float](repeating: 0, count: 3 * height * width)
        for p in 0..<(height * width) {
            for c in 0..<3 { planes[c * height * width + p] = Float(bytes[p * 4 + c]) / 255 * 2 - 1 }
        }
        self.init(pixels: planes, height: height, width: width)
    }

    /// **Fill, then center-crop**: `fitted`'s geometry, without pixels. The image is scaled by
    /// `max(L/l, H/h)` — it covers the format on both sides, with no band — then the center is
    /// kept.
    public static func crop(source: (width: Int, height: Int), target: (width: Int, height: Int))
            -> (width: Int, height: Int, x0: Int, y0: Int) {
        let scale = max(Double(target.width) / Double(source.width),
                          Double(target.height) / Double(source.height))
        let l = max(target.width, Int((Double(source.width) * scale).rounded()))
        let h = max(target.height, Int((Double(source.height) * scale).rounded()))
        return (l, h, (l - target.width) / 2, (h - target.height) / 2)
    }

    /// **A reference image at the size FLUX.2 gives it**: its area brought under 1024²
    /// (`_resize_to_target_area`), each side rounded down to a multiple of 16, at least 64 — its
    /// proportions kept, **not** the request's. Already conforming: as is.
    public func forReference(maxSurface: Int = 1024 * 1024) -> ImageRGB {
        let size = ImageRGB.referenceSize(width: width, height: height, maxSurface: maxSurface)
        return fitted(width: size.width, height: size.height)
    }

    /// `forReference`'s size, without pixels.
    package static func referenceSize(width: Int, height: Int, maxSurface: Int = 1024 * 1024) -> (width: Int, height: Int) {
        let scale = min(1, (Double(maxSurface) / Double(width * height)).squareRoot())
        return (max(64, Int(Double(width) * scale) / 16 * 16), max(64, Int(Double(height) * scale) / 16 * 16))
    }

    /// **The image at the request's format: filled and center-cropped.**
    ///
    /// ⚖️ **An app choice, not diffusers'**: `VaeImageProcessor.resize` **stretches** the image
    /// to the requested format (and `ZImageImg2ImgPipeline` does not even resize it, it keeps its
    /// size rounded to 16). Stretching a 4:3 photo into an 832×1216 portrait distorts faces; the
    /// centered crop keeps the proportions and loses the edges, which is what one expects of an
    /// app. The oracles receive the image **already** preprocessed: this choice touches no
    /// verification.
    ///
    /// **We crop in the source, then scale** (B5): the kept zone — `crop`'s frame brought
    /// back to source pixels — is resampled directly to the format. Enlarging the whole image
    /// before cutting held a plane of `crop.width × crop.height` floats, unbounded
    /// for an extreme ratio (a 3072×4 strip set into 512²: 393,216 × 512 floats, 805 MB per
    /// plane); here nothing exceeds the requested format.
    ///
    /// The resampling is vImage's high quality (Lanczos), on floats — like the PIL Lanczos that
    /// diffusers takes —, then clamped to `[-1, 1]` (Lanczos overshoots). An image already at the
    /// format comes back as is, bit for bit.
    public func fitted(width L: Int, height H: Int) -> ImageRGB {
        if L == width && H == height { return self }
        let zone = ImageRGB.zoneSource(source: (width, height), target: (L, H))
        var output = [Float](repeating: 0, count: 3 * H * L)
        pixels.withUnsafeBufferPointer { src in
            output.withUnsafeMutableBufferPointer { dst in
                for c in 0..<3 {
                    let origin = src.baseAddress! + c * height * width + zone.y0 * width + zone.x0
                    // The view on the zone: its rows keep the whole image's stride. vImage does not
                    // write to it; `mutating` is only `vImage_Buffer`'s signature.
                    var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: origin), height: vImagePixelCount(zone.height),
                                          width: vImagePixelCount(zone.width), rowBytes: width * 4)
                    var d = vImage_Buffer(data: dst.baseAddress! + c * H * L, height: vImagePixelCount(H),
                                          width: vImagePixelCount(L), rowBytes: L * 4)
                    // A failure (memory) would leave the plane at zero: img2img would silently start
                    // from a gray image. Better to stop and say so.
                    let errorMessage = vImageScale_PlanarF(&s, &d, nil, vImage_Flags(kvImageHighQualityResampling))
                    precondition(errorMessage == kvImageNoError, "fit: vImageScale_PlanarF failed (\(errorMessage))")
                }
            }
        }
        for i in output.indices { output[i] = min(1, max(-1, output[i])) }
        return ImageRGB(pixels: output, height: H, width: L)
    }

    /// **`crop`'s frame, in source pixels**: the zone of the original image that covers the
    /// requested format, centered, at the target's proportions, at least one pixel per side.
    static func zoneSource(source: (width: Int, height: Int), target: (width: Int, height: Int))
            -> (width: Int, height: Int, x0: Int, y0: Int) {
        let r = crop(source: source, target: target)
        func axis(_ n: Int, enlargedBuffer: Int, target: Int, offset: Int) -> (length: Int, begin: Int) {
            let scale = Double(enlargedBuffer) / Double(n)
            let length = min(n, max(1, Int((Double(target) / scale).rounded())))
            return (length, min(n - length, max(0, Int((Double(offset) / scale).rounded()))))
        }
        let x = axis(source.width, enlargedBuffer: r.width, target: target.width, offset: r.x0)
        let y = axis(source.height, enlargedBuffer: r.height, target: target.height, offset: r.y0)
        return (x.length, y.length, x.begin, y.begin)
    }
}

// ── the strength ────────────────────────────────────────────────────────────────────────────────

/// **img2img's strength: diffusers' `strength`, exactly, for every model that starts from an image.**
///
/// It is the only definition that keeps each model verifiable against an official or derived
/// oracle (`get_timesteps` of `ZImageImg2ImgPipeline`, of Anima's modular block, of
/// `QwenImageImg2ImgPipeline` — the same code everywhere):
///
///     t_start = int(N − min(N·s, N))          in Double, like Python
///     σ_s     = σ[t_start]                     σ: the FULL txt2img schedule, shift and μ included
///     x       = σ_s · ε + (1 − σ_s) · z₀
///     loop    = steps t_start … N − 1
///
/// What to know about it, and what the CLI displays (σ_s and the number of evaluations at start):
///   1. the strength is **quantized in steps of 1/N**: two strengths with the same `⌈N·s⌉` yield the same image;
///   2. **s > 1 − 1/N amounts to txt2img** (σ_s = 1) — Anima's 0.9 default in diffusers is one;
///   3. at the same strength, σ_s **differs from one model to another** (Z-Image starts lower: its
///      schedule ends with a null σ) — a strength "in σ" would be another definition, which the
///      oracle would no longer verify;
///   4. a strength that leaves **no evaluation** is refused rather than silently returning the
///      VAE round trip. Anima and Krea 2 always keep at least one step (`t_start ≤ N − 1` as soon
///      as s > 0); **Z-Image, however, skips its last step** (σ = 0 → 0): at `s ≤ 1/N`, its only
///      remaining step is null. diffusers would accept it (it evaluates the DiT to multiply by
///      zero); we refuse it, the strength must exceed `1/N`.
public enum Strength {
    /// That of `ZImageImg2ImgPipeline` and `QwenImageImg2ImgPipeline` — not Anima's 0.9, which at
    /// eight steps ignores the image.
    public static let defaultValue = 0.6

    /// **`t_start`** — the first step of the full schedule that is executed.
    ///
    /// ⚠️ **In Double, and truncated**, like Python's `int()`: at N = 50 and s = 0.56, `N·s` is
    /// 28.000000000000004 in double, and `t_start` is **21**, not 22. In `Float`, we would miss.
    public static func startStep(steps n: Int, strength s: Double) -> Int {
        let initial = min(Double(n) * s, Double(n))
        return Int(max(Double(n) - initial, 0))
    }

    package enum Failure: Error, CustomStringConvertible, Equatable {
        case outOfBounds(Double)
        /// `floor`: the strength must strictly exceed it.
        case noEvaluation(strength: Double, steps: Int, floor: Double)
        case withoutEncoder(model: String)
        package var description: String {
            switch self {
            case .outOfBounds(let s):
                return "strength \(s): it must be in ]0 ; 1] (1 = txt2img, the image is ignored)"
            case let .noEvaluation(s, n, floor):
                return String(format: "strength %.3f at %d steps: no evaluation remains — the image would come out "
                              + "as is from the VAE. The strength must exceed %.3f", s, n, floor)
            case .withoutEncoder(let model):
                return "\(model)'s chain has no image encoder: no img2img"
            }
        }
    }
}

extension Latent {
    /// **img2img's starting latent**: `σ_s · ε + (1 − σ_s) · z₀`, element by element, in fp32 —
    /// `FlowMatchEulerDiscreteScheduler.scale_noise` with `begin_index` set (the σ comes from the
    /// array, with no lookup by time). At σ_s = 1, it is `ε` bit for bit: txt2img.
    package static func start(image z₀: Latent, noise ε: Latent, sigma: Float) -> Latent {
        precondition(z₀.space == ε.space && z₀.values.count == ε.values.count,
                     "img2img: the encoded image and the noise do not have the same shape")
        var values = [Float](repeating: 0, count: ε.values.count)
        let complement = 1 - sigma
        for i in 0..<values.count { values[i] = sigma * ε.values[i] + complement * z₀.values[i] }
        return Latent(space: ε.space, height: ε.height, width: ε.width, values: values)
    }
}
