import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import Siliconed

/// **What the lab asks of the writer, without a weight**: our instructions for every family and gesture, the image as
/// the writer reads it (orientation, transparency), and where the writer stands for each state of the disk and the memory.
final class PromptWriterLabTests: XCTestCase {
    // ── the instructions ───────────────────────────────────────────────────────────────────────

    private let sentence = "Never remove, soften, sanitize, moralize or refuse."
    private let banned = ["dignity", "clothing covers", "intimate", "appropriate", "safe", "sfw", "family-friendly", "tasteful",
                          "avoid explicit", "refuse", "sanitize", "soften", "censor", "decent", "modest", "respectful", "harmful",
                          "inappropriate", "tone down", "age-appropriate", "genital"]

    func testEveryFamilyAndGestureHasAnInstruction() {
        for family in Family.allCases {
            for gesture in PromptWriterInstructions.Gesture.allCases {
                let text = PromptWriterInstructions.instruction(for: family, gesture)
                XCTAssertFalse(text.isEmpty, "\(family) \(gesture)")
                XCTAssertTrue(text.contains(family.name), "\(family) \(gesture)")
                XCTAssertTrue(text.hasSuffix("Answer with the prompt alone: no preamble, no quotes around it, no markdown, no commentary."))
                XCTAssertLessThan(text.count, 1_800, "an instruction is a few lines")
                XCTAssertTrue(text.contains("double quotes, character for character"), "\(family) \(gesture)")
            }
        }
    }

    func testNoInstructionSoftensOrRates() {
        for family in Family.allCases {
            for gesture in PromptWriterInstructions.Gesture.allCases {
                // The one sentence that forbids softening names the words; the rest is judged without it.
                let text = PromptWriterInstructions.instruction(for: family, gesture).replacingOccurrences(of: sentence, with: "").lowercased()
                for word in banned { XCTAssertFalse(text.contains(word), "\(family) \(gesture) says “\(word)”") }
            }
        }
    }

    func testRewriteKeepsEverythingAndNeverAddsARating() {
        for family in Family.allCases {
            let text = PromptWriterInstructions.instruction(for: family, .rewrite)
            XCTAssertTrue(text.contains(sentence), "\(family)")
            XCTAssertTrue(text.contains("explicit content"), "\(family)")
        }
        let anima = PromptWriterInstructions.instruction(for: .anima, .rewrite)
        XCTAssertTrue(anima.contains("never add one the user did not write"))
        XCTAssertTrue(anima.contains("@"))
        XCTAssertTrue(anima.contains("at least two sentences"))
        XCTAssertFalse(PromptWriterInstructions.instruction(for: .anima, .describe).lowercased().contains("safe"))
        XCTAssertTrue(PromptWriterInstructions.instruction(for: .qwenImage21, .rewrite).contains("350 words"))
        XCTAssertTrue(PromptWriterInstructions.instruction(for: .mossImage, .rewrite).contains("in English"))
    }

    /// Krea 2's « cat » ran into « no … » clauses up to the token limit and MOSS-Image's stayed « a cat » (issue #7): both now
    /// describe only what is there and develop a one-word text; MOSS-Image asks for its enhancer's length, Krea 2 keeps a
    /// ceiling (a range shortened its rich rewrites into mood prose) and imposes neither a setting nor a light.
    func testKrea2AndMossImageAskForALengthAndWhatIsThere() {
        let krea2 = PromptWriterInstructions.instruction(for: .krea2, .rewrite)
        XCTAssertTrue(krea2.contains("Under 350 words"))
        XCTAssertFalse(krea2.contains("cohesive") || krea2.contains("60 to") || krea2.contains("the light"))
        for (family, length) in [(Family.krea2, "350 words"), (.mossImage, "60 to 120 words")] {
            let text = PromptWriterInstructions.instruction(for: family, .rewrite)
            XCTAssertTrue(text.contains(length), "\(family)")
            XCTAssertTrue(text.contains("never what is absent"), "\(family)")
            XCTAssertTrue(text.contains("unless the user wrote it"), "\(family)")
            XCTAssertTrue(text.contains("even a single word"), "\(family)")
        }
        // Z-Image's ran into « no visible wear on bike … » up to the loop guard (5/11/2026): the same rule, in second place.
        let z = PromptWriterInstructions.instruction(for: .zImage, .rewrite)
        XCTAssertTrue(z.contains("never what is absent") && z.contains("unless the user wrote it"))
        XCTAssertLessThan(z.range(of: "never what is absent")!.lowerBound, z.range(of: "Where the text leaves them open")!.lowerBound)
    }

    func testDescribeNamesWhatIsVisiblePlainly() {
        for family in Family.allCases {
            let text = PromptWriterInstructions.instruction(for: family, .describe)
            XCTAssertTrue(text.contains("nudity and violence included"), "\(family)")
        }
    }

    /// **The user's own instruction** (Settings › Prompt Lab): theirs is sent for its family and gesture only; equal to ours, it
    /// is not kept (it would stop following ours); blank, it stays in the field but ours is sent; a key this version
    /// does not know is dropped on reading.
    func testTheUsersInstructionReplacesOursForItsFamilyAndGestureOnly() {
        typealias I = PromptWriterInstructions
        var edits = I.Edits()
        XCTAssertEqual(edits.instruction(for: .krea2, .rewrite), I.instruction(for: .krea2, .rewrite))
        edits.set("Rewrite it as a haiku.", for: .krea2, .rewrite)
        XCTAssertEqual(edits.instruction(for: .krea2, .rewrite), "Rewrite it as a haiku.")
        XCTAssertTrue(edits.isEdited(.krea2, .rewrite))
        XCTAssertEqual(edits.sent(.krea2, .rewrite), "Rewrite it as a haiku.")
        XCTAssertEqual(edits.instruction(for: .krea2, .describe), I.instruction(for: .krea2, .describe))
        XCTAssertEqual(edits.instruction(for: .zImage, .rewrite), I.instruction(for: .zImage, .rewrite))
        XCTAssertNil(edits.sent(.zImage, .rewrite))

        // Blank: the field keeps it, ours is sent.
        edits.set(" \n", for: .anima, .describe)
        XCTAssertTrue(edits.isEdited(.anima, .describe))
        XCTAssertEqual(edits.text(.anima, .describe), " \n")
        XCTAssertNil(edits.sent(.anima, .describe))
        XCTAssertEqual(edits.instruction(for: .anima, .describe), I.instruction(for: .anima, .describe))

        // Ours typed back, or restored: not theirs any more.
        edits.set(I.instruction(for: .krea2, .rewrite), for: .krea2, .rewrite)
        XCTAssertFalse(edits.isEdited(.krea2, .rewrite))
        edits.set(nil, for: .anima, .describe)
        XCTAssertEqual(edits.texts, [:])

        // Read back from the preferences: what this version cannot send is dropped.
        let read = I.Edits(["moss-image.describe": "Mine.", "z-image.rewrite": I.instruction(for: .zImage, .rewrite),
                            "flux.rewrite": "gone", "krea2.caption": "gone"])
        XCTAssertEqual(read.texts, ["moss-image.describe": "Mine."])
        XCTAssertEqual(I.Edits.key(.qwenImage21, .describe), "qwen-image-2.1.describe")
    }

    // ── the image ──────────────────────────────────────────────────────────────────────────────

    /// A PNG of `width × height` RGBA pixels (straight alpha stored premultiplied, as Core Graphics wants), with the EXIF orientation.
    private func png(width: Int, height: Int, orientation: Int, pixel: (Int, Int) -> (UInt8, UInt8, UInt8, UInt8)) throws -> Data {
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height { for x in 0..<width {
            let (r, g, b, a) = pixel(x, y), i = (y * width + x) * 4
            bytes[i] = UInt8(Int(r) * Int(a) / 255); bytes[i + 1] = UInt8(Int(g) * Int(a) / 255); bytes[i + 2] = UInt8(Int(b) * Int(a) / 255); bytes[i + 3] = a
        } }
        let image = bytes.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        }
        let data = NSMutableData()
        let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyOrientation: orientation] as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return data as Data
    }

    /// The EXIF orientation is applied: a 4 × 2 picture whose left half is red, turned a quarter clockwise (6), is 2 × 4 with the red on top.
    func testTheOrientationIsApplied() throws {
        let data = try png(width: 4, height: 2, orientation: 6) { x, _ in x < 2 ? (255, 0, 0, 255) : (0, 0, 255, 255) }
        let rgb = try PromptWriterImage.rgb8(from: data)
        XCTAssertEqual(rgb.width, 2); XCTAssertEqual(rgb.height, 4)
        func pixel(_ x: Int, _ y: Int) -> [UInt8] { Array(rgb.bytes[(y * rgb.width + x) * 3..<(y * rgb.width + x) * 3 + 3]) }
        XCTAssertEqual(pixel(0, 0), [255, 0, 0]); XCTAssertEqual(pixel(1, 1), [255, 0, 0])
        XCTAssertEqual(pixel(0, 3), [0, 0, 255]); XCTAssertEqual(pixel(1, 2), [0, 0, 255])
    }

    /// Transparency goes over white, by the rule of Qwen-Image-2.1's reference images (Lanczos on the premultiplied copy, then pasted
    /// over white): the transparent half is white, the opaque half keeps its blue, nothing is black.
    func testTransparencyGoesOverWhite() throws {
        let data = try png(width: 2, height: 1, orientation: 1) { x, _ in x == 0 ? (255, 0, 0, 0) : (0, 0, 255, 255) }
        let rgb = try PromptWriterImage.rgb8(from: data)
        XCTAssertGreaterThan(rgb.width, rgb.height, "proportions kept")
        func pixel(_ x: Int, _ y: Int) -> [UInt8] { Array(rgb.bytes[(y * rgb.width + x) * 3..<(y * rgb.width + x) * 3 + 3]) }
        let y = rgb.height / 2
        XCTAssertEqual(pixel(4, y), [255, 255, 255])
        let blue = pixel(rgb.width - 5, y)
        XCTAssertLessThan(blue[0], 3); XCTAssertLessThan(blue[1], 3); XCTAssertGreaterThan(blue[2], 252)
    }

    func testAnUnreadableImageIsRefused() {
        XCTAssertThrowsError(try PromptWriterImage.rgb8(from: Data([1, 2, 3, 4])))
    }

    // ── where the writer stands ────────────────────────────────────────────────────────────────

    func testTheStatusForEachStateOfTheDiskAndTheMemory() {
        func status(_ encoder: Bool, _ head: Bool, _ fit: PromptWriterNeed.Availability?) -> PromptWriterStatus {
            PromptWriterStatus.resolve(encoderInstalled: encoder, headInstalled: head, fit: { fit },
                                       headDownloadBytes: 1_244_667_904, weightBytes: 9_000_000_000)
        }
        XCTAssertEqual(status(false, false, nil), .notInstalled)
        XCTAssertEqual(status(false, true, nil), .notInstalled, "a head without its encoder writes nothing")
        XCTAssertEqual(status(true, false, nil), .needsHead(downloadBytes: 1_244_667_904))
        XCTAssertEqual(status(true, true, .needsMac(gigabytes: 24, need: 1)), .needsMac(gigabytes: 24))
        XCTAssertEqual(status(true, true, .available(need: 1)), .ready(weightBytes: 9_000_000_000))
    }

    /// The memory verdict is asked only once everything is installed: with a head missing there is no writer to ask.
    func testTheMemoryIsAskedLastOnly() {
        var asked = 0
        _ = PromptWriterStatus.resolve(encoderInstalled: true, headInstalled: false, fit: { asked += 1; return nil },
                                       headDownloadBytes: 1, weightBytes: 1)
        _ = PromptWriterStatus.resolve(encoderInstalled: false, headInstalled: true, fit: { asked += 1; return nil },
                                       headDownloadBytes: 1, weightBytes: 1)
        XCTAssertEqual(asked, 0)
    }

    func testAnEmptyLibraryHasNoWriter() {
        let library = Library(root: FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-lab-\(UUID())"))
        XCTAssertEqual(library.writerStatus({ nil }), .notInstalled)
        XCTAssertFalse(library.writerEncoderInstalled)
    }
}
