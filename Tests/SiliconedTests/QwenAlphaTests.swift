import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import Siliconed

/// **A Qwen-Image-2.1 reference with transparency, as diffusers prepares it** — Pillow's alpha
/// arithmetic (`PillowAlpha`, values written by Pillow 12.3), the straight RGBA an `ImageRGB` keeps, and
/// the two copies `preparedReference` makes of it. The whole chain is checked against `__call__` by
/// `qwen21-preprocess-alpha`; here, what holds without a byte of weights or of the venv.
final class QwenAlphaTests: XCTestCase {

    private let pixels: [UInt8] = [200, 10, 255, 100, 37, 250, 1, 7, 90, 90, 90, 0, 255, 128, 3, 255]

    func testPremultiplyIsPillows() {
        XCTAssertEqual(PillowAlpha.premultiply(pixels), [78, 4, 100, 100, 1, 7, 0, 7, 0, 0, 0, 0, 255, 128, 3, 255])
    }

    /// Truncated, clipped; α 0 and 255 left as they are (a transparent pixel keeps its bytes).
    func testUnpremultiplyIsPillows() {
        XCTAssertEqual(PillowAlpha.unpremultiply([78, 4, 100, 100, 3, 7, 0, 7, 5, 5, 5, 0, 255, 128, 3, 255]),
                       [198, 10, 255, 100, 109, 255, 0, 7, 5, 5, 5, 0, 255, 128, 3, 255])
    }

    func testPasteOverWhiteIsPillows() {
        XCTAssertEqual(PillowAlpha.overWhite(pixels, width: 4, height: 1).bytes,
                       [233, 159, 255, 249, 255, 248, 255, 255, 255, 255, 128, 3])
    }

    /// `Image.fromarray(a, "RGBA").resize((5, 4), LANCZOS)` — premultiplied around the Lanczos.
    func testRGBAResizeIsPillows() {
        let source: [UInt8] = [255, 228, 34, 121, 243, 189, 6, 131, 102, 168, 82, 193, 187, 150, 81, 243, 205, 24, 236, 8, 246, 164, 231, 36, 210, 111, 172, 210, 174, 176, 218, 242, 169, 114, 205, 63, 160, 47, 212, 79, 72, 112, 120, 222, 69, 31, 95, 108, 36, 109, 238, 69, 51, 81, 228, 211, 202, 58, 202, 65, 76, 70, 193, 104, 42, 250, 209, 164, 2, 44, 178, 140, 98, 2, 243, 21, 50, 29, 14, 7, 53, 50, 151, 221, 46, 60, 230, 192, 164, 156, 127, 214, 154, 194, 195, 137, 72, 153, 74, 209, 58, 76, 105, 84, 10, 213, 226, 115, 168, 118, 214, 201, 69, 25, 185, 31, 36, 45, 158, 77, 46, 12, 218, 31, 12, 112, 24, 116, 11, 130, 22, 250, 129, 142, 80, 34, 93, 181, 32, 98, 165, 105, 50, 103, 85, 37, 100, 231, 132, 164, 21, 52, 236, 253, 147, 128, 146, 247, 38, 67, 243, 22, 19, 5, 49, 230, 23, 192, 69, 151, 224, 15, 77, 222, 200, 71, 118, 252, 140, 127, 199, 121, 53, 124, 77, 166, 239, 29, 215, 151, 17, 251]
        let expected: [UInt8] = [255, 188, 49, 88, 127, 141, 84, 206, 135, 126, 99, 113, 146, 90, 239, 82, 178, 132, 200, 187, 50, 148, 214, 127, 48, 63, 170, 84, 39, 54, 176, 98, 68, 77, 201, 195, 143, 143, 152, 124, 38, 134, 62, 171, 90, 178, 184, 87, 125, 76, 146, 146, 89, 82, 134, 74, 102, 182, 94, 92, 33, 173, 13, 113, 85, 224, 34, 66, 109, 141, 113, 126, 163, 155, 94, 100, 219, 186, 46, 131]
        XCTAssertEqual(PillowAlpha.lanczos(source, width: 8, height: 6, toWidth: 5, toHeight: 4), expected)
        XCTAssertEqual(PillowAlpha.lanczos(source, width: 8, height: 6, toWidth: 8, toHeight: 6), source, "a copy at the same size")
    }

    /// The generalized Lanczos gives the RGB path's bytes, unchanged.
    func testThreeBandLanczosUnchanged() {
        var bytes = [UInt8](repeating: 0, count: 3 * 37 * 23)
        for i in bytes.indices { bytes[i] = UInt8((i * 97 + i / 7) % 256) }
        let rgb = Qwen3VLImages.RGB8(bytes: bytes, width: 37, height: 23)
        XCTAssertEqual(PillowResample.lanczos(rgb, width: 64, height: 32).bytes,
                       PillowResample.lanczos(bytes, channels: 3, width: 37, height: 23, toWidth: 64, toHeight: 32))
    }

    // ── the straight RGBA an image keeps ────────────────────────────────────────────────────

    private func image(_ rgba: [UInt8], width: Int, height: Int) -> CGImage {
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 4 * width,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    /// A straight sRGB image is read as its bytes — the colour under α = 0 included, as Pillow reads it;
    /// `pixels` stays the image over white.
    func testAStraightImageKeepsItsBytes() throws {
        let decoded = try ImageRGB(cgImage: image(pixels, width: 4, height: 1))
        XCTAssertEqual(decoded.rgba, pixels)
        XCTAssertEqual(decoded.pixels[2], 1, "the transparent pixel is white in `pixels`")
    }

    /// The file path reads the full image (ImageIO's thumbnail is premultiplied).
    func testAPNGKeepsItsBytes() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-alpha-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image(pixels, width: 4, height: 1), nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        XCTAssertEqual(try ImageRGB(contentsOf: url).rgba, pixels)
    }

    /// A file's bytes sent by another Mac decode as the file does: transparency, and a JPEG turned by EXIF.
    func testBytesDecodeAsTheirFile() throws {
        var opaque = [UInt8](repeating: 0, count: 4 * 40 * 30)
        for p in 0..<(40 * 30) { opaque[4 * p] = UInt8(p % 256); opaque[4 * p + 1] = UInt8((p / 3) % 256); opaque[4 * p + 3] = 255 }
        for (bytes, w, h, type, orientation) in [(pixels, 4, 1, UTType.png, 1), (opaque, 40, 30, UTType.jpeg, 6)] {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-bytes-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: url) }
            let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(destination, image(bytes, width: w, height: h),
                                       [kCGImagePropertyOrientation: orientation] as CFDictionary)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
            let file = try ImageRGB(contentsOf: url)
            XCTAssertEqual(try ImageRGB(data: Data(contentsOf: url), name: "sent"), file)
            XCTAssertEqual(file.width, orientation == 6 ? h : w, "upright")
        }
    }

    /// An opaque image — even stored with an alpha channel — has no `rgba`, and is prepared as before.
    func testAnOpaqueImageIsUnchanged() throws {
        var opaque = [UInt8](repeating: 0, count: 4 * 40 * 30)
        for p in 0..<(40 * 30) { opaque[4 * p] = UInt8(p % 256); opaque[4 * p + 1] = UInt8((p / 3) % 256); opaque[4 * p + 3] = 255 }
        let decoded = try ImageRGB(cgImage: image(opaque, width: 40, height: 30))
        XCTAssertNil(decoded.rgba)
        let prepared = QwenImage21Pipeline.preparedReference(decoded, resolution: 128)
        XCTAssertEqual(prepared, QwenImage21Pipeline.image(QwenImage21Pipeline.reference(decoded, resolution: 128)))
        XCTAssertEqual(QwenImage21EncodingModule.rgba(prepared).suffix(prepared.width * prepared.height),
                       ArraySlice([Float](repeating: 1, count: prepared.width * prepared.height)))
    }

    /// A transparent reference: the VAE reads the straight RGBA with its alpha, the tower the copy over white.
    func testATransparentReferenceGivesTwoCopies() {
        let rgba = pixels + pixels + pixels + pixels   // 4×4, resized to 64² (R = 64)
        let source = ImageRGB(pixels: QwenImage21Pipeline.image(PillowAlpha.overWhite(rgba, width: 4, height: 4)).pixels,
                              height: 4, width: 4, rgba: rgba)
        let size = Qwen3VLImages.conditionSize(width: 4, height: 4, resolution: 64)
        let prepared = QwenImage21Pipeline.preparedReference(source, resolution: 64)
        let resized = PillowAlpha.lanczos(rgba, width: 4, height: 4, toWidth: size.width, toHeight: size.height)
        XCTAssertEqual(prepared.rgba, resized)
        XCTAssertEqual(QwenImage21Pipeline.rgb8(prepared), PillowAlpha.overWhite(resized, width: size.width, height: size.height))
        let planes = QwenImage21EncodingModule.rgba(prepared)
        let plane = size.width * size.height
        XCTAssertEqual(planes.count, 4 * plane)
        XCTAssertEqual(planes[3 * plane], Float(resized[3]) / 255 * 2 - 1, "the real alpha, not +1")
    }

    /// The cache key sees the RGBA: the same pixels over white with another alpha are another image.
    func testCacheKeySeesTheAlpha() {
        let white = QwenImage21Pipeline.image(PillowAlpha.overWhite(pixels, width: 4, height: 1)).pixels
        func key(_ image: ImageRGB) -> RenderCache.Key { var k = RenderCache.KeyBuilder("i"); k.add(image); return k.finish() }
        let opaque = ImageRGB(pixels: white, height: 1, width: 4)
        let transparent = ImageRGB(pixels: white, height: 1, width: 4, rgba: pixels)
        XCTAssertNotEqual(key(opaque), key(transparent))
        var reference = RenderCache.KeyBuilder("i"); reference.add(4); reference.add(1); reference.add(floats: white)
        XCTAssertEqual(key(opaque), reference.finish(), "an opaque image's key is unchanged")
    }
}
