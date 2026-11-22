import CoreGraphics
import Foundation
import ImageIO
import SiliCore

/// **What happens to a condition image before the vision tower** — `QwenImage21Pipeline.__call__`
/// then `Qwen3VLProcessor` (`Qwen2VLImageProcessorFast`), on 8-bit RGB:
///
///     image ─ calculate_dimensions(R², w/h): each side to the nearest multiple of 32
///           ─ PIL `resize(LANCZOS)` (the same image then goes to the VAE)        `lanczos`
///           ─ alpha over white, Pillow's paste (an opaque image is unchanged)   `PillowAlpha`
///           ─ smart_resize(factor 32): the identity on sizes already multiples of 32
///           ─ (x − 127.5) / 127.5                    rescale 1/255 and normalize 0.5/0.5, FUSED
///           ─ patches 16×16, in 2×2 merge-block order, each DUPLICATED in time  → [N, 1536]
///
/// Pitfalls: the processor's own resize (bicubic, torchvision) never runs on the pipeline's
/// images — the oracle measured `do_resize=False` identical to the bit — so it is not ported:
/// `pixels` refuses a size `smart_resize` would change. The normalization is the FUSED one
/// (`_fuse_mean_std_and_rescale_factor`: mean and std times 255), not `x/255` then `(· − 0.5)/0.5`
/// — one rounding apart. **Transparency is ported** (`PillowAlpha`): the pipeline makes every
/// reference RGBA, Pillow premultiplies around its Lanczos, and only this copy is laid over white
/// (`white.paste(img, mask=A)`) — the VAE reads the RGBA itself (`QwenImage21DenoisingModule.preparedReferences`).
/// `RGB8(contentsOf:)`, the checks' reader, still refuses transparency.
package enum Qwen3VLImages {
    /// One image, ready for the vision tower: `pixel_values` and `image_grid_thw[1:]`. The values
    /// live in `Pages`, `valueBytes` of them: 24 MB for a 1024² condition, held from the
    /// processor to the tower — `malloc`ed, they stayed in the footprint after it.
    package struct Pixels {
        private let storage: Pages
        package let gridHeight: Int, gridWidth: Int
        package var patches: Int { gridHeight * gridWidth }
        /// `[patches, 1536]`, valid while this value lives.
        package var values: UnsafeMutableBufferPointer<Float> {
            UnsafeMutableBufferPointer(start: storage.floats(), count: gridHeight * gridWidth * Pixels.patchValues)
        }
        /// `in_channels · temporal · patch²` of the published tower (`Qwen3VLVision.Config.patchValues`).
        package static let patchValues = 3 * 2 * 16 * 16

        package init(gridHeight: Int, gridWidth: Int) {
            self.gridHeight = gridHeight; self.gridWidth = gridWidth
            storage = Pages(bytes: Pixels.valueBytes(gridHeight: gridHeight, gridWidth: gridWidth))
        }

        package init(values: [Float], gridHeight: Int, gridWidth: Int) {
            self.init(gridHeight: gridHeight, gridWidth: gridWidth)
            precondition(values.count == gridHeight * gridWidth * Pixels.patchValues, "pixel_values: \(values.count) values")
            self.values.baseAddress!.update(from: values, count: values.count)
        }

        /// **What one condition image's `pixel_values` hold**: `[gh·gw, 1536]` fp32, rounded to the page.
        package static func valueBytes(gridHeight: Int, gridWidth: Int) -> Int {
            Pages.capacity(for: gridHeight * gridWidth * patchValues * 4)
        }
    }

    /// 8-bit RGB, row-major `[height, width, 3]`.
    package struct RGB8: Equatable {
        package let bytes: [UInt8]
        package let width: Int, height: Int
        package init(bytes: [UInt8], width: Int, height: Int) {
            precondition(bytes.count == 3 * width * height, "RGB8: \(bytes.count) bytes for \(width)×\(height)")
            self.bytes = bytes; self.width = width; self.height = height
        }

        /// A PNG/JPEG read by ImageIO, **without colour management**: the stored bytes, as PIL
        /// reads them (`Image.open(…).convert("RGB")`). Opaque 8-bit RGB or RGBA only.
        package init(contentsOf url: URL) throws {
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
                  image.bitsPerComponent == 8, let provider = image.dataProvider, let data = provider.data,
                  let base = CFDataGetBytePtr(data) else {
                throw ImageRGB.Failure.unreadable(url.path)
            }
            let (w, h) = (image.width, image.height), stride = image.bytesPerRow
            let channels = image.bitsPerPixel / 8
            guard channels == 3 || channels == 4 else { throw ImageRGB.Failure.unreadable(url.path) }
            let alpha = image.alphaInfo
            let first = alpha == .first || alpha == .premultipliedFirst || alpha == .noneSkipFirst ? 1 : 0
            var bytes = [UInt8](repeating: 0, count: 3 * w * h)
            for y in 0..<h {
                for x in 0..<w {
                    let pixel = base + y * stride + x * channels
                    if channels == 4, alpha != .noneSkipLast, alpha != .noneSkipFirst, alpha != .none {
                        guard pixel[first == 1 ? 0 : 3] == 255 else {
                            throw ImageRGB.Failure.unreadable("\(url.path): transparency is not supported here")
                        }
                    }
                    for c in 0..<3 { bytes[(y * w + x) * 3 + c] = pixel[first + c] }
                }
            }
            self.init(bytes: bytes, width: w, height: h)
        }
    }

    /// **`calculate_dimensions(R², w/h)`** of the pipeline: the area `R²` at the image's
    /// aspect ratio, each side rounded to the NEAREST multiple of 32 (Python's `round`: half to even).
    package static func conditionSize(width: Int, height: Int, resolution: Int) -> (width: Int, height: Int) {
        let ratio = Double(width) / Double(height)
        let w = (Double(resolution * resolution) * ratio).squareRoot()
        let h = w / ratio
        return (Int((w / 32).rounded(.toNearestOrEven)) * 32, Int((h / 32).rounded(.toNearestOrEven)) * 32)
    }

    /// **`smart_resize`** of `Qwen2VLImageProcessor` — `(height, width)`.
    package static func smartResize(height: Int, width: Int, factor: Int = 32,
                                    minPixels: Int = 65_536, maxPixels: Int = 16_777_216) -> (height: Int, width: Int) {
        func round(_ v: Double) -> Int { Int(v.rounded(.toNearestOrEven)) }
        var h = round(Double(height) / Double(factor)) * factor
        var w = round(Double(width) / Double(factor)) * factor
        if h * w > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            h = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if h * w < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            w = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (h, w)
    }

    /// The pipeline's preparation: Lanczos to `conditionSize`, then `pixels`.
    package static func prepare(_ image: RGB8, resolution: Int, patch: Int = 16, merge: Int = 2,
                                temporal: Int = 2) throws -> (resized: RGB8, pixels: Pixels) {
        let size = conditionSize(width: image.width, height: image.height, resolution: resolution)
        let resized = PillowResample.lanczos(image, width: size.width, height: size.height)
        return (resized, try pixels(resized, patch: patch, merge: merge, temporal: temporal))
    }

    package struct Unported: Error, CustomStringConvertible {
        package let description: String
    }

    /// The processor proper, on an image already at its size: normalization and patches.
    package static func pixels(_ image: RGB8, patch: Int = 16, merge: Int = 2, temporal: Int = 2) throws -> Pixels {
        let factor = patch * merge
        let target = smartResize(height: image.height, width: image.width, factor: factor)
        guard target == (image.height, image.width) else {
            throw Unported(description: "\(image.width)×\(image.height): smart_resize would resample to "
                           + "\(target.width)×\(target.height) (bicubic), which is not ported — the pipeline never asks it")
        }
        let (gh, gw) = (image.height / patch, image.width / patch)
        let perPatch = 3 * temporal * patch * patch
        guard perPatch == Pixels.patchValues else {
            throw Unported(description: "patches of \(perPatch) values: the published tower's are \(Pixels.patchValues)")
        }
        let result = Pixels(gridHeight: gh, gridWidth: gw)
        let values = result.values
        // `(x − 127.5) / 127.5`, both in fp32 like `tvF.normalize` (`sub_` then `div_`).
        let mean: Float = 127.5, std: Float = 127.5
        var index = 0
        for blockRow in 0..<(gh / merge) {
            for blockColumn in 0..<(gw / merge) {
                for innerRow in 0..<merge {
                    for innerColumn in 0..<merge {
                        let (py, px) = (blockRow * merge + innerRow, blockColumn * merge + innerColumn)
                        // `(c, t, y, x)`: the temporal copy is the whole 16×16 plane again.
                        for c in 0..<3 {
                            for _ in 0..<temporal {
                                for y in 0..<patch {
                                    let source = ((py * patch + y) * image.width + px * patch) * 3 + c
                                    for x in 0..<patch {
                                        values[index] = (Float(image.bytes[source + 3 * x]) - mean) / std
                                        index += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return result
    }

    /// The `(row, column)` of each patch, in the 2×2 merge-block order of `pixel_values`
    /// (`get_vision_position_ids`).
    package static func patchPositions(gridHeight gh: Int, gridWidth gw: Int, merge: Int = 2) -> [(row: Int, column: Int)] {
        var out: [(Int, Int)] = []
        out.reserveCapacity(gh * gw)
        for br in 0..<(gh / merge) {
            for bc in 0..<(gw / merge) {
                for ir in 0..<merge { for ic in 0..<merge { out.append((br * merge + ir, bc * merge + ic)) } }
            }
        }
        return out
    }

    /// **`get_vision_interpolation_indices_and_weights`** (bilinear, `align_corners=True`, border):
    /// four taps into the `side × side` table per patch, in merge-block order. fp32 like the
    /// reference: `src = i · (side − 1) / max(g − 1, 1)`, weights `1 − |src − tap|`, their product.
    package static func positionTaps(gridHeight gh: Int, gridWidth gw: Int, side: Int, merge: Int = 2)
            -> (indices: [Int], weights: [Float]) {
        func axis(_ i: Int, _ size: Int) -> [(Int, Float)] {
            let source = Float(i) * Float(side - 1) / Float(max(size - 1, 1))
            let floor = source.rounded(.down)
            return (0..<2).map { offset in
                let tap = min(max(Int(floor) + offset, 0), side - 1)
                return (tap, max(1 - abs(source - floor - Float(offset)), 0))
            }
        }
        var indices: [Int] = [], weights: [Float] = []
        for (row, column) in patchPositions(gridHeight: gh, gridWidth: gw, merge: merge) {
            let (hs, ws) = (axis(row, gh), axis(column, gw))
            for (ht, hw) in hs {
                for (wt, ww) in ws { indices.append(ht * side + wt); weights.append(hw * ww) }
            }
        }
        return (indices, weights)
    }

    /// **`get_rope_index`**: the three positions `(t, h, w)` of each row of the language model's sequence. Text counts
    /// up on the three axes; a picture's slots (a run of `slots`, `gridHeight/merge × gridWidth/merge` of them, row by
    /// row) freeze `t` at the position reached and count `h` and `w` from it; the position then advances by the
    /// larger side of the merged grid, not by the slot count.
    package static func rotaryPositions(slots: [Bool], grids: [(height: Int, width: Int)], merge: Int = 2) throws -> [(Int, Int, Int)] {
        var out: [(Int, Int, Int)] = [], position = 0, index = 0, next = 0
        out.reserveCapacity(slots.count)
        while index < slots.count {
            guard slots[index] else { out.append((position, position, position)); position += 1; index += 1; continue }
            guard next < grids.count else { throw Unported(description: "a picture slot without its picture") }
            let (h, w) = (grids[next].height / merge, grids[next].width / merge)
            next += 1
            guard index + h * w <= slots.count, slots[index..<(index + h * w)].allSatisfy({ $0 }) else {
                throw Unported(description: "picture \(next): \(h * w) slots expected at row \(index)")
            }
            for y in 0..<h { for x in 0..<w { out.append((position, position + y, position + x)) } }
            position += max(h, w)
            index += h * w
        }
        guard next == grids.count else { throw Unported(description: "\(grids.count) pictures for \(next) runs of slots") }
        return out
    }

    /// **`Qwen3VLTextRotaryEmbedding`'s table**, `[rows, headDim]` in f32 as torch computes it: `1 / θ^(2i/d)` by an
    /// f32 `pow`, the angle an f32 product. Frequency `i` reads the height when `i % 3 == 1`, the width when
    /// `i % 3 == 2`, both under `3 · section`, the time otherwise (`recomposition_frequencies`); the half is then
    /// repeated (`cat(f, f)`).
    package static func rotaryTable(_ positions: [(Int, Int, Int)], headDim: Int, theta: Double,
                                    sections: [Int] = [24, 20, 20]) -> (cos: [Float], sin: [Float]) {
        let half = headDim / 2
        let frequencies = (0..<half).map { 1 / powf(Float(theta), Float(2 * $0) / Float(headDim)) }
        var cos: [Float] = [], sin: [Float] = []
        cos.reserveCapacity(positions.count * headDim); sin.reserveCapacity(positions.count * headDim)
        for (t, h, w) in positions {
            let angles = (0..<half).map { i -> Float in
                let p = i % 3 == 1 && i < 3 * sections[1] ? h : i % 3 == 2 && i < 3 * sections[2] ? w : t
                return Float(p) * frequencies[i]
            }
            for _ in 0..<2 { for a in angles { let (c, s) = cosSin(Double(a)); cos.append(c); sin.append(s) } }
        }
        return (cos, sin)
    }
}

/// **Pillow's `Image.resize`**, 8 bits per channel, transcribed from `libImaging/Resample.c` so
/// that the pipeline's Lanczos is reproduced to the bit: separable, horizontal pass then vertical
/// pass, each through 8-bit; coefficients normalized in double then rounded to fixed point
/// (22 fractional bits); accumulation in integers starting at ½; clipped to [0, 255].
package enum PillowResample {
    static let precisionBits = 32 - 8 - 2

    /// The `LANCZOS` filter (support 3): `sinc(x) · sinc(x/3)`.
    static func lanczos(_ x: Double) -> Double {
        func sinc(_ x: Double) -> Double { x == 0 ? 1 : sin(x * .pi) / (x * .pi) }
        return -3 <= x && x < 3 ? sinc(x) * sinc(x / 3) : 0
    }

    /// `precompute_coeffs` + `normalize_coeffs_8bpc`: per output pixel, its first input pixel,
    /// its tap count, and `ksize` fixed-point weights.
    static func coefficients(input: Int, output: Int, support filterSupport: Double = 3,
                             filter: (Double) -> Double = lanczos) -> (bounds: [(Int, Int)], ksize: Int, k: [Int32]) {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let support = filterSupport * filterScale
        let ksize = Int(support.rounded(.up)) * 2 + 1
        var bounds: [(Int, Int)] = []
        var k = [Int32](repeating: 0, count: output * ksize)
        for xx in 0..<output {
            let center = (Double(xx) + 0.5) * scale
            let ss = 1 / filterScale
            // `(int)(center - support + 0.5)`: C truncation toward zero.
            var xmin = Int(center - support + 0.5); if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5); if xmax > input { xmax = input }
            xmax -= xmin
            var weights = [Double](repeating: 0, count: xmax), total = 0.0
            for x in 0..<xmax {
                let w = filter((Double(x + xmin) - center + 0.5) * ss)
                weights[x] = w; total += w
            }
            for x in 0..<xmax {
                let w = total != 0 ? weights[x] / total : weights[x]
                let fixed = w * Double(1 << precisionBits)
                k[xx * ksize + x] = Int32(w < 0 ? (-0.5 + fixed).rounded(.towardZero) : (0.5 + fixed).rounded(.towardZero))
            }
            bounds.append((xmin, xmax))
        }
        return (bounds, ksize, k)
    }

    static func clip8(_ v: Int) -> UInt8 {
        let shifted = v >> precisionBits
        return UInt8(min(max(shifted, 0), 255))
    }

    /// `Image.resize((width, height), Image.LANCZOS)` on 8-bit RGB.
    package static func lanczos(_ image: Qwen3VLImages.RGB8, width: Int, height: Int) -> Qwen3VLImages.RGB8 {
        if image.width == width && image.height == height { return image }
        return Qwen3VLImages.RGB8(bytes: lanczos(image.bytes, channels: 3, width: image.width, height: image.height,
                                                 toWidth: width, toHeight: height),
                                  width: width, height: height)
    }

    /// The same on `channels` interleaved 8-bit channels, each resampled alone — Pillow's 3- and
    /// 4-band paths do exactly that (`RGBa`: `PillowAlpha` premultiplies around it).
    package static func lanczos(_ bytes: [UInt8], channels: Int, width w0: Int, height h0: Int,
                                toWidth width: Int, toHeight height: Int) -> [UInt8] {
        var current = bytes, currentWidth = w0
        // `ImagingResampleInner`: horizontal first, only if the width changes; then vertical.
        if width != currentWidth {
            let c = coefficients(input: currentWidth, output: width)
            var out = [UInt8](repeating: 0, count: channels * width * h0)
            current.withUnsafeBufferPointer { src in
                for y in 0..<h0 {
                    for xx in 0..<width {
                        let (xmin, count) = c.bounds[xx]
                        for ch in 0..<channels {
                            var sum = 1 << (precisionBits - 1)
                            for x in 0..<count {
                                sum += Int(src[(y * currentWidth + xmin + x) * channels + ch]) * Int(c.k[xx * c.ksize + x])
                            }
                            out[(y * width + xx) * channels + ch] = clip8(sum)
                        }
                    }
                }
            }
            current = out; currentWidth = width
        }
        if height != h0 {
            let c = coefficients(input: h0, output: height)
            var out = [UInt8](repeating: 0, count: channels * width * height)
            current.withUnsafeBufferPointer { src in
                for yy in 0..<height {
                    let (ymin, count) = c.bounds[yy]
                    for x in 0..<width {
                        for ch in 0..<channels {
                            var sum = 1 << (precisionBits - 1)
                            for y in 0..<count {
                                sum += Int(src[((ymin + y) * width + x) * channels + ch]) * Int(c.k[yy * c.ksize + y])
                            }
                            out[(yy * width + x) * channels + ch] = clip8(sum)
                        }
                    }
                }
            }
            current = out
        }
        return current
    }
}

/// **Pillow's arithmetic on alpha**, 8 bits, from `libImaging` (`Convert.c`, `Paste.c`) — what the
/// Qwen-Image-2.1 pipeline does to a reference with transparency, checked to the bit against Pillow 12.3
/// on random bytes and by `qwen21-preprocess-alpha` against `__call__`:
///
///     RGBA → RGBa     c · α / 255 rounded: `MULDIV255` = (t + (t >> 8)) >> 8, t = c·α + 128
///     RGBa → RGBA     255 · c / α TRUNCATED, clipped to 255; α = 0 or 255 left as is
///     paste on white  `BLEND`: DIV255(255 · (255 − α) + c · α), the same rounding as MULDIV255
///     resize RGBA     RGBA → RGBa → Lanczos on the 4 bands → RGBA; a copy at the same size
package enum PillowAlpha {
    @inline(__always) static func div255(_ v: Int) -> UInt8 {
        let t = v + 128
        return UInt8((t + (t >> 8)) >> 8)
    }

    package static func premultiply(_ rgba: [UInt8]) -> [UInt8] {
        var out = rgba
        for p in stride(from: 0, to: rgba.count, by: 4) {
            let a = Int(rgba[p + 3])
            for c in 0..<3 { out[p + c] = div255(Int(rgba[p + c]) * a) }
        }
        return out
    }

    package static func unpremultiply(_ rgba: [UInt8]) -> [UInt8] {
        var out = rgba
        for p in stride(from: 0, to: rgba.count, by: 4) {
            let a = Int(rgba[p + 3])
            guard a != 0, a != 255 else { continue }
            for c in 0..<3 { out[p + c] = UInt8(min(255, 255 * Int(rgba[p + c]) / a)) }
        }
        return out
    }

    /// `Image.resize((width, height), LANCZOS)` on an RGBA image.
    package static func lanczos(_ rgba: [UInt8], width w0: Int, height h0: Int, toWidth width: Int, toHeight height: Int) -> [UInt8] {
        if w0 == width && h0 == height { return rgba }
        return unpremultiply(PillowResample.lanczos(premultiply(rgba), channels: 4, width: w0, height: h0,
                                                    toWidth: width, toHeight: height))
    }

    /// `white = Image.new("RGB", size, (255,)*3); white.paste(img, mask=img.getchannel("A"))`.
    package static func overWhite(_ rgba: [UInt8], width: Int, height: Int) -> Qwen3VLImages.RGB8 {
        var out = [UInt8](repeating: 0, count: 3 * width * height)
        for p in 0..<(width * height) {
            let a = Int(rgba[4 * p + 3])
            for c in 0..<3 { out[3 * p + c] = div255(255 * (255 - a) + Int(rgba[4 * p + c]) * a) }
        }
        return Qwen3VLImages.RGB8(bytes: out, width: width, height: height)
    }
}
