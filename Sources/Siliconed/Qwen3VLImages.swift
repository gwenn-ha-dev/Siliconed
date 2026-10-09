import CoreGraphics
import Foundation
import ImageIO

/// **What happens to a condition image before the vision tower** — `QwenImage21Pipeline.__call__`
/// then `Qwen3VLProcessor` (`Qwen2VLImageProcessorFast`), on 8-bit RGB:
///
///     image ─ calculate_dimensions(R², w/h): each side to the nearest multiple of 32
///           ─ PIL `resize(LANCZOS)` (the same image then goes to the VAE)        `lanczos`
///           ─ alpha over white (an opaque image is unchanged)
///           ─ smart_resize(factor 32): the identity on sizes already multiples of 32
///           ─ (x − 127.5) / 127.5                    rescale 1/255 and normalize 0.5/0.5, FUSED
///           ─ patches 16×16, in 2×2 merge-block order, each DUPLICATED in time  → [N, 1536]
///
/// Pitfalls: the processor's own resize (bicubic, torchvision) never runs on the pipeline's
/// images — the oracle measured `do_resize=False` identical to the bit — so it is not ported:
/// `pixels` refuses a size `smart_resize` would change. The normalization is the FUSED one
/// (`_fuse_mean_std_and_rescale_factor`: mean and std times 255), not `x/255` then `(· − 0.5)/0.5`
/// — one rounding apart. Images with transparency are not ported (Pillow premultiplies RGBA
/// around its resize); the app's images are RGB.
package enum Qwen3VLImages {
    /// One image, ready for the vision tower: `pixel_values` and `image_grid_thw[1:]`.
    package struct Pixels {
        package let values: [Float]
        package let gridHeight: Int, gridWidth: Int
        package var patches: Int { gridHeight * gridWidth }
        package init(values: [Float], gridHeight: Int, gridWidth: Int) {
            self.values = values; self.gridHeight = gridHeight; self.gridWidth = gridWidth
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
        var values = [Float](repeating: 0, count: gh * gw * perPatch)
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
        return Pixels(values: values, gridHeight: gh, gridWidth: gw)
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
        var current = image
        // `ImagingResampleInner`: horizontal first, only if the width changes; then vertical.
        if width != current.width {
            let c = coefficients(input: current.width, output: width)
            var out = [UInt8](repeating: 0, count: 3 * width * current.height)
            current.bytes.withUnsafeBufferPointer { src in
                for y in 0..<current.height {
                    for xx in 0..<width {
                        let (xmin, count) = c.bounds[xx]
                        for ch in 0..<3 {
                            var sum = 1 << (precisionBits - 1)
                            for x in 0..<count {
                                sum += Int(src[(y * current.width + xmin + x) * 3 + ch]) * Int(c.k[xx * c.ksize + x])
                            }
                            out[(y * width + xx) * 3 + ch] = clip8(sum)
                        }
                    }
                }
            }
            current = Qwen3VLImages.RGB8(bytes: out, width: width, height: current.height)
        }
        if height != current.height {
            let c = coefficients(input: current.height, output: height)
            var out = [UInt8](repeating: 0, count: 3 * width * height)
            current.bytes.withUnsafeBufferPointer { src in
                for yy in 0..<height {
                    let (ymin, count) = c.bounds[yy]
                    for x in 0..<width {
                        for ch in 0..<3 {
                            var sum = 1 << (precisionBits - 1)
                            for y in 0..<count {
                                sum += Int(src[((ymin + y) * width + x) * 3 + ch]) * Int(c.k[yy * c.ksize + y])
                            }
                            out[(yy * width + x) * 3 + ch] = clip8(sum)
                        }
                    }
                }
            }
            current = Qwen3VLImages.RGB8(bytes: out, width: width, height: height)
        }
        return current
    }
}
