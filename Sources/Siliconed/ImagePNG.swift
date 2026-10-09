import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

// **An image comes out one way only: `ImageRGB` → sRGB bytes → `CGImage` → PNG.**
//
// The models output sRGB: the image is **tagged sRGB** (a PNG without a profile would be displayed
// according to the screen). Non-regression is judged on the decoded pixels, and the close-out (two
// renders → same md5) holds because nothing in the PNG depends on the time. No AppKit: CoreGraphics
// and ImageIO only.

extension ImageRGB {
    /// **The pixels as 8-bit RGBA, row by row, opaque alpha** — what a UI keeps in a history
    /// rather than 12 MB of floats per image. The reference's post-processing: `(x/2 + 0.5)`,
    /// clamped to `[0, 1]`, ×255, rounded. The clamping is not cosmetic — the decoder outputs up
    /// to ±1.14, so without it the values overflow.
    public func rgba8() -> [UInt8] {
        var bytes = [UInt8](repeating: 255, count: height * width * 4)
        let plan = height * width
        for p in 0..<plan {
            for c in 0..<3 {
                let value = pixels[c * plan + p] / 2 + 0.5
                bytes[p * 4 + c] = UInt8(max(0, min(1, value)) * 255 + 0.5)
            }
        }
        return bytes
    }

    /// The image as an **sRGB** `CGImage`, 8 bits, RGBX.
    public func cgImage() -> CGImage {
        PNG.cgImage(rgba: rgba8(), height: height, width: width)
    }

    /// **The PNG, in memory**: sRGB, no timestamp, and the generation metadata in `iTXt` text
    /// chunks (UTF-8: a prompt is not always Latin-1), sorted by key — two identical renders
    /// yield the same bytes.
    public func png(metadata: [String: String] = [:]) throws(EngineError) -> Data {
        try PNG.data(cgImage(), metadata: metadata)
    }
}

/// PNG encoding and its text chunks. Reading (`text`) serves an app that re-reads an image's
/// parameters, like Draw Things.
public enum PNG {
    package static let srgb = CGColorSpace(name: CGColorSpace.sRGB)!

    /// 8-bit RGBX bytes → sRGB `CGImage`.
    static func cgImage(rgba: [UInt8], height: Int, width: Int) -> CGImage {
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: srgb,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false,
                       intent: .defaultIntent)!
    }

    /// **The only way to write a PNG in the repository** — renders, checks and contact sheets.
    public static func data(_ image: CGImage, metadata: [String: String] = [:]) throws(EngineError) -> Data {
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output as CFMutableData,
                                                                 UTType.png.identifier as CFString, 1, nil) else {
            throw .imageEncodingFailed
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw .imageEncodingFailed }
        return withText(output as Data, metadata)
    }

    /// Writes `data` to `path`, atomically.
    public static func write(_ data: Data, to path: String) throws(EngineError) {
        do { try data.write(to: URL(fileURLWithPath: path), options: .atomic) }
        catch { throw .imageWriteFailed(file: path) }
    }

    // ── the text chunks ───────────────────────────────────────────────────────────────────

    /// **Inserts one `iTXt` chunk per metadata item, right after `IHDR`.**
    ///
    /// ImageIO writes only its own keys (`Title`, `Comment`…): for `prompt`, `seed` or
    /// `model`, we insert the chunks ourselves. The format is simple and stable (PNG §11.3.4):
    /// length, type, data, CRC-32 of the type and data. Uncompressed `iTXt`:
    /// `key \0 0 0 \0 \0 UTF-8 text`. A key must fit in 1 to 79 Latin-1 bytes: we keep the keys
    /// ASCII, and a key outside the rule is ignored rather than yielding an invalid PNG.
    static func withText(_ png: Data, _ metadata: [String: String]) -> Data {
        let keys = metadata.keys.filter { key in
            !key.isEmpty && key.utf8.count <= 79 && key.unicodeScalars.allSatisfy { $0.value >= 32 && $0.value < 127 }
        }.sorted()
        guard !keys.isEmpty, png.count > 33 else { return png }
        var chunks = Data()
        for key in keys {
            var body = Data(key.utf8)
            body.append(contentsOf: [0, 0, 0, 0, 0])   // end of key, uncompressed, method, empty language, empty translation
            body.append(Data(metadata[key]!.utf8))
            chunks.append(chunk("iTXt", body))
        }
        // Signature (8) + IHDR (4 + 4 + 13 + 4): the first chunk is always IHDR.
        var output = png.prefix(33)
        output.append(chunks)
        output.append(png.dropFirst(33))
        return Data(output)
    }

    static func chunk(_ type: String, _ body: Data) -> Data {
        var d = Data()
        var length = UInt32(body.count).bigEndian
        withUnsafeBytes(of: &length) { d.append(contentsOf: $0) }
        let typeAndBody = Data(type.utf8) + body
        d.append(typeAndBody)
        var crc = crc32(typeAndBody).bigEndian
        withUnsafeBytes(of: &crc) { d.append(contentsOf: $0) }
        return d
    }

    /// **Re-reads a PNG's text chunks** (uncompressed `tEXt` and `iTXt`) — so an app can recover
    /// an image's prompt and seed, and so the test target can verify it. Returns an empty
    /// dictionary if it is not a PNG.
    public static func text(_ png: Data) -> [String: String] {
        let bytes = [UInt8](png)
        guard bytes.count > 8, bytes[0..<8] == [137, 80, 78, 71, 13, 10, 26, 10] else { return [:] }
        var output: [String: String] = [:]
        var i = 8
        while i + 12 <= bytes.count {
            let length = Int(bytes[i]) << 24 | Int(bytes[i + 1]) << 16 | Int(bytes[i + 2]) << 8 | Int(bytes[i + 3])
            let type = String(decoding: bytes[(i + 4)..<(i + 8)], as: UTF8.self)
            guard i + 12 + length <= bytes.count else { break }
            let body = Array(bytes[(i + 8)..<(i + 8 + length)])
            if type == "tEXt" || type == "iTXt", let end = body.firstIndex(of: 0) {
                let key = String(decoding: body[..<end], as: UTF8.self)
                if type == "tEXt" {
                    output[key] = String(bytes: body[(end + 1)...], encoding: .isoLatin1)
                } else if end + 2 < body.count, body[end + 1] == 0 {
                    // compression 0, method, then language \0 and translation \0.
                    var j = end + 3
                    for _ in 0..<2 { while j < body.count, body[j] != 0 { j += 1 }; j += 1 }
                    if j <= body.count { output[key] = String(decoding: body[min(j, body.count)...], as: UTF8.self) }
                }
            }
            if type == "IEND" { break }
            i += 12 + length
        }
        return output
    }

    /// CRC-32 (ISO 3309, reflected polynomial 0xEDB88320), PNG's.
    static func crc32(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in data {
            crc ^= UInt32(byte)
            for _ in 0..<8 { crc = (crc >> 1) ^ (crc & 1 == 1 ? 0xEDB8_8320 : 0) }
        }
        return crc ^ 0xFFFF_FFFF
    }
}
