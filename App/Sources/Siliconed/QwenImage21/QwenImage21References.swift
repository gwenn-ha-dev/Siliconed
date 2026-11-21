import Foundation

/// **An edit's reference pictures, between the request and the two encoders** — what `QwenImage21Pipeline.__call__`
/// does to `image=[…]` before any model runs.
///
///     reference ─ RGB8 (RGBA when transparent) ─ Pillow Lanczos to ~R² (sides ×32) ─┬─ Qwen3-VL's vision tower (over white)
///                  (one copy, `prepared`)                                           └─ the VAE's encoder (RGBA)
///
/// What a port by analogy would miss:
///
///   - **The references are sized by `calculate_dimensions(R², w/h)`** — round half to EVEN at 32 — with
///     `R = output_resolution`, whatever the output's size; one Lanczos (Pillow's, to the bit, `PillowResample`)
///     feeds both the encoder and the VAE.
///   - **Every reference is made RGBA** (`convert("RGBA")`): a transparent one keeps its alpha — Pillow premultiplies
///     around the Lanczos, the VAE reads the four channels as they are, and only the vision copy is laid over white
///     (`PillowAlpha`).
///   - **A latent cell is a vision patch** (16 px both): a reference's `h · w` latent tokens fill its `h · w / 4`
///     slots of the encoder's sequence, four a slot.
///   - The output follows the **last** reference without `height` / `width`. **The product's rule is the first**
///     (`outputSize`: the picture one edits), at ~R², each side ≥ 512, under `Format`'s ceiling; an explicit size wins.
package enum QwenImage21References {
    /// `output_resolution`: the pipeline's default, and the product's — references at ~1 MP.
    package static let outputResolution = 1024
    /// The pipeline's edits: 1 to 3 references.
    package static let maximum = 3

    /// **The product's output size** for an edit without an explicit size: the first reference's aspect ratio at ~R²
    /// (`calculate_dimensions`, sides at 32), then the product's bounds — each side ≥ `Format.minimumSide` (raised
    /// keeping the ratio), the area ≤ `Format.editSurface` (the long side cut). Always a multiple of 32.
    package static func outputSize(referenceWidth: Int, referenceHeight: Int,
                                   resolution: Int = outputResolution) -> (width: Int, height: Int) {
        let ratio = Double(referenceWidth) / Double(referenceHeight)
        var (w, h) = Qwen3VLImages.conditionSize(width: referenceWidth, height: referenceHeight, resolution: resolution)
        let floor = (Format.minimumSide + 31) / 32 * 32
        if w < floor { w = floor; h = max(floor, Int((Double(w) / ratio / 32).rounded()) * 32) }
        if h < floor { h = floor; w = max(floor, Int((Double(h) * ratio / 32).rounded()) * 32) }
        if w * h > Format.editSurface {
            if w >= h { w = Format.editSurface / h / 32 * 32 } else { h = Format.editSurface / w / 32 * 32 }
        }
        return (w, h)
    }

    /// **One reference as the VAE and the vision tower read it.** Opaque: RGB8 again (the `[-1, 1]` floats of an 8-bit
    /// picture round back to their bytes), Pillow's Lanczos. With transparency (`ImageRGB.rgba`): Pillow's Lanczos
    /// premultiplied (a copy at the same size: a transparent pixel keeps its RGB), `rgba` kept for the VAE, `pixels`
    /// laid over white for the vision tower.
    package static func prepared(_ image: ImageRGB, resolution: Int = outputResolution) -> ImageRGB {
        let size = Qwen3VLImages.conditionSize(width: image.width, height: image.height, resolution: resolution)
        guard let rgba = image.rgba else {
            return self.image(PillowResample.lanczos(rgb8(image), width: size.width, height: size.height))
        }
        let resized = PillowAlpha.lanczos(rgba, width: image.width, height: image.height, toWidth: size.width, toHeight: size.height)
        let white = PillowAlpha.overWhite(resized, width: size.width, height: size.height)
        return ImageRGB(pixels: self.image(white).pixels, height: size.height, width: size.width, rgba: resized)
    }

    /// `[3, h, w]` in `[-1, 1]` → interleaved 8-bit RGB: `round((x + 1)/2 · 255)`, the exact inverse of `image` on
    /// 8-bit sources.
    package static func rgb8(_ image: ImageRGB) -> Qwen3VLImages.RGB8 {
        let plane = image.width * image.height
        var bytes = [UInt8](repeating: 0, count: 3 * plane)
        for c in 0..<3 {
            for p in 0..<plane {
                let v = (image.pixels[c * plane + p] + 1) / 2 * 255
                bytes[p * 3 + c] = UInt8(max(0, min(255, v.rounded())))
            }
        }
        return Qwen3VLImages.RGB8(bytes: bytes, width: image.width, height: image.height)
    }

    /// 8-bit RGB → `[3, h, w]` in `[-1, 1]`, as `VaeImageProcessor` makes it (`x/255`, then `2x − 1`).
    package static func image(_ rgb: Qwen3VLImages.RGB8) -> ImageRGB {
        let plane = rgb.width * rgb.height
        var pixels = [Float](repeating: 0, count: 3 * plane)
        for p in 0..<plane { for c in 0..<3 { pixels[c * plane + p] = Float(rgb.bytes[p * 3 + c]) / 255 * 2 - 1 } }
        return ImageRGB(pixels: pixels, height: rgb.height, width: rgb.width)
    }

    /// **What the VAE's encoder reads**, `[4, H, W]`: an opaque picture with an alpha of +1 (255 → `255/255 · 2 − 1`);
    /// with transparency, the straight `rgba` planes, each byte `b/255 · 2 − 1`.
    package static func rgba(_ image: ImageRGB) -> [Float] {
        guard let bytes = image.rgba else { return image.pixels + [Float](repeating: 1, count: image.height * image.width) }
        let plane = image.height * image.width
        var planes = [Float](repeating: 0, count: 4 * plane)
        for p in 0..<plane { for c in 0..<4 { planes[c * plane + p] = Float(bytes[p * 4 + c]) / 255 * 2 - 1 } }
        return planes
    }
}
