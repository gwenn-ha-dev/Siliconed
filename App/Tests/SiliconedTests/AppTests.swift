import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import Siliconed

/// **What an app reads from the library** — catalog, errors, cancellation token, preview
/// projection, PNG chunks, progress. No map, no GPU.
final class AppTests: XCTestCase {

    // ── the catalog ────────────────────────────────────────────────────────────────────────

    /// Every suggested format is renderable, and every orientation is represented.
    func testRecommendedFormatsPassFormat() throws {
        for card in ModelCard.allCards {
            for f in card.formats { XCTAssertNoThrow(try Format.check(width: f.width, height: f.height), "\(f)") }
            XCTAssertEqual(Set(card.formats.map(\.orientation)), [.square, .portrait, .landscape])
        }
        XCTAssertEqual(RecommendedFormat(832, 1216).orientation, .portrait)
        XCTAssertEqual(RecommendedFormat(1216, 832).description, "1216×832 (landscape)")
    }

    /// The sheet says what the module does: default steps, space — a single truth.
    func testTheCardMatchesTheDenoiser() {
        let denoisers: [String: any DenoisingModule] = [
            "z-image": ZImageDenoisingModule(map: ""), "anima": AnimaDenoisingModule(map: ""), "krea2": Krea2DenoisingModule(map: ""),
            "moss-image": MossImageDenoisingModule(map: "", turbo: ""), "qwen-image-2.1": QwenImage21DenoisingModule(map: "")]
        XCTAssertEqual(Model.identifiers, ["z-image", "anima", "krea2", "moss-image", "qwen-image-2.1"])
        for card in ModelCard.allCards {
            let d = denoisers[card.id]!
            XCTAssertEqual(card.defaultSteps, d.defaultSteps)
            XCTAssertEqual(card.space, d.space)
            XCTAssertEqual(d.model, card.id)
            XCTAssertEqual(ModelCard.of(card.family), card)
        }
        XCTAssertFalse(ModelCard.anima.license.commercial)
        XCTAssertFalse(ModelCard.qwenImage21.license.commercial)
        XCTAssertTrue(ModelCard.krea2.license.requiredFilter)
    }

    /// An empty library does not throw: it has nothing, and says what is missing.
    func testAnEmptyLibraryDoesNotThrow() {
        let empty = Library(root: URL(fileURLWithPath: "/nulle-part-\(UUID())"))
        XCTAssertEqual(empty.models().count, 0)
        XCTAssertNotNil(ModelCard.krea2.missing(in: empty))
    }

    // ── the errors ─────────────────────────────────────────────────────────────────────────

    /// **Every internal failure folds into its named case** at the door — and none falls back to
    /// "The operation couldn't be completed".
    func testInternalFailuresFoldIntoTheirCase() {
        let folds: [(any Error, EngineError)] = [
            (Strength.Failure.outOfBounds(2), .strengthOutOfRange(strength: 2)),
            (Model.Failure.unknown("sd15"), .unknownModel(model: "sd15")),
            (MissingFile("/x", .published), .fileMissing(path: "/x")),
            (MissingFile("/x.safetensors", .map), .fileMissing(path: "/x.safetensors")),
            (Chain.Failure.latentIncompatible(output: .flux, entry: .qwenImage),
             .incompatibleChain(output: "\(LatentSpace.flux)", input: "\(LatentSpace.qwenImage)")),
            (Safetensors.Failure.cannotOpen("/vae", EACCES), .fileUnreadable(path: "/vae")),
            (Safetensors.Failure.cannotStat("/vae", EIO), .fileUnreadable(path: "/vae")),
            (Safetensors.Failure.cannotMap("/vae", ENOMEM), .fileUnreadable(path: "/vae")),
            (Safetensors.Failure.missingTensor(file: "/vae", name: "w"), .corruptMap(file: "/vae")),
            (ImageRGB.Failure.unreadable("/a.png"), .imageUnreadable(file: "/a.png")),
            (Request.Failure.emptyPrompt, .emptyPrompt),
            (Request.Failure.steps(0), .invalidSteps(steps: 0, allowed: [])),
            (CancellationError(), .cancelled),
            (NSError(domain: NSPOSIXErrorDomain, code: Int(ENOSPC)), .diskFull(needed: nil, available: nil)),
            (EngineError.renderAlreadyRunning, .renderAlreadyRunning),
        ]
        for (failure, expected) in folds {
            XCTAssertEqual(EngineError(failure), expected, "\(failure)")
        }
        for e in folds.map(\.1) {
            let text = e.localizedDescription
            XCTAssertFalse(text.contains("couldn’t be completed") || text.contains("couldn't be completed"), text)
            XCTAssertEqual(text, e.errorDescription)
        }
    }

    /// A write without room is the disk's, said as such — an image's (`PNG.write`) as a download's; a write refused is not.
    func testAWriteWithoutRoomIsTheDisks() throws {
        XCTAssertTrue(EngineError.ranOutOfRoom(NSError(domain: NSPOSIXErrorDomain, code: Int(ENOSPC))))
        XCTAssertTrue(EngineError.ranOutOfRoom(NSError(domain: NSCocoaErrorDomain, code: NSFileWriteOutOfSpaceError)))
        XCTAssertFalse(EngineError.ranOutOfRoom(NSError(domain: NSPOSIXErrorDomain, code: Int(EACCES))))
        XCTAssertFalse(EngineError.ranOutOfRoom(NSError(domain: NSCocoaErrorDomain, code: NSFileWriteNoPermissionError)))
        let nowhere = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-\(UUID())/none/a.png").path
        XCTAssertThrowsError(try PNG.write(Data([1]), to: nowhere)) { XCTAssertEqual($0 as? EngineError, .imageWriteFailed(file: nowhere)) }
    }

    /// "File absent" is an error of its own — not a bad header.
    func testAMissingFileIsNotABadHeader() {
        let path = "/nulle-part-\(UUID()).safetensors"
        XCTAssertThrowsError(try Safetensors(path: path)) {
            XCTAssertEqual($0 as? MissingFile, MissingFile(path, .published))
        }
    }

    // ── cancellation ────────────────────────────────────────────────────────────────────────

    func testTheTokenRaisesOnceRaised() throws {
        let token = Cancellation()
        XCTAssertFalse(token.isCancelled)
        XCTAssertNoThrow(try token.check())
        token.cancel(); token.cancel()
        XCTAssertTrue(token.isCancelled)
        XCTAssertThrowsError(try token.check()) { XCTAssertEqual($0 as? EngineError, .cancelled) }
        let noCancellation: Cancellation? = nil
        XCTAssertNoThrow(try noCancellation.check())
    }

    /// Raised from another thread, seen by the one that computes.
    func testTheTokenCrossesThreads() {
        let token = Cancellation()
        let finished = expectation(description: "raised")
        DispatchQueue.global().async { token.cancel(); finished.fulfill() }
        wait(for: [finished], timeout: 1)
        XCTAssertTrue(token.isCancelled)
    }

    // ── the preview ────────────────────────────────────────────────────────────────────────────

    /// x̂₀ = x − σ·v, projected; σ = 0 projects x as is; the bias alone gives the expected gray.
    func testTheProjectionTakesXHatZero() {
        // Two channels, 1 × 2 cells: channel 0 → red, channel 1 → blue.
        let p = PreviewProjection(weights: [1, 0, 0, 0, 0, 1], bias: [0, 0, 0])
        let x: [Float] = [0.5, -1, 0, 1], v: [Float] = [1, 0, 2, 0]
        // σ = 0.5: x̂₀ = [0, -1 | -1, 1] → red (0 → 128, -1 → 0), blue (-1 → 0, 1 → 255); green,
        // with no weight or bias, stays in the middle (0 → 128).
        XCTAssertEqual(p.apply(x: x, v: v, sigma: 0.5, height: 1, width: 2), [128, 128, 0, 255, 0, 128, 255, 255])
        XCTAssertEqual(p.apply(x: x, v: v, sigma: 0, height: 1, width: 2),
                       p.apply(x: x, v: nil, sigma: 0.5, height: 1, width: 2))
        // Clipped, never overflowing.
        let strong = PreviewProjection(weights: [10, -10, 0, 0, 0, 0], bias: [0, 0, 0.5])
        XCTAssertEqual(strong.apply(x: [1, 0], v: nil, sigma: 0, height: 1, width: 1), [255, 0, 191, 255])
    }

    /// Every space has its fitted coefficients, channels × 3, finite and non-zero.
    func testEverySpaceHasAProjection() throws {
        for space in [LatentSpace.flux, .qwenImage, .flux2] {
            let p = try XCTUnwrap(space.preview)
            XCTAssertEqual(p.channels, space.channels)
            XCTAssertTrue(p.weights.allSatisfy(\.isFinite) && p.weights.contains { $0 != 0 })
        }
        XCTAssertNotEqual(LatentSpace.flux.preview, LatentSpace.qwenImage.preview)
    }

    // ── images and PNG ────────────────────────────────────────────────────────────────

    /// The PNG carries its metadata (UTF-8 included), reads back, and two encodings are identical.
    func testThePNGCarriesItsMetadata() throws {
        let image = ImageRGB(pixels: (0..<(3 * 4 * 6)).map { (i: Int) -> Float in Float(i % 7) / 3 - 1 }, height: 4, width: 6)
        let meta = ["prompt": "une bibliothèque, « livres » — 書", "seed": "42", "model": "krea2"]
        let a = try image.png(metadata: meta), b = try image.png(metadata: meta)
        XCTAssertEqual(a, b, "no date: two identical encodings")
        XCTAssertEqual(PNG.text(a), meta)
        XCTAssertEqual(PNG.text(try image.png()), [:])
        // Read back by ImageIO: a valid PNG, with the right pixels (the CRC is correct, otherwise it refuses).
        let source = try XCTUnwrap(CGImageSourceCreateWithData(a as CFData, nil))
        let reread = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual(reread.width, 6); XCTAssertEqual(reread.height, 4)
        XCTAssertEqual(reread.colorSpace?.name, CGColorSpace.sRGB as CFString)
        // Redrawn in an sRGB RGBX context: same space, hence no conversion — the bytes must be
        // those of `rgba8`.
        var bytes = [UInt8](repeating: 0, count: 6 * 4 * 4)
        bytes.withUnsafeMutableBytes { raw in
            let c = CGContext(data: raw.baseAddress, width: 6, height: 4, bitsPerComponent: 8, bytesPerRow: 24,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            c.draw(reread, in: CGRect(x: 0, y: 0, width: 6, height: 4))
        }
        let withoutAlpha = { (o: [UInt8]) in o.enumerated().filter { $0.offset % 4 != 3 }.map(\.element) }
        XCTAssertEqual(withoutAlpha(bytes), withoutAlpha(image.rgba8()))
    }

    /// PNG's CRC-32: the reference value of an empty "IEND".
    func testTheCRCIsPNGs() {
        XCTAssertEqual(PNG.crc32(Data("IEND".utf8)), 0xAE42_6082)
    }

    /// The rounding of `rgba8` is the reference's: (x/2 + 0.5), clipped, ×255, +0.5.
    func testRoundingIsTheReferences() {
        let image = ImageRGB(pixels: [-1.2, -1, 0, 1, 1.14, 0.5, 0, 0, 0, 0, 0, 0], height: 1, width: 4)
        XCTAssertEqual(image.rgba8().enumerated().filter { $0.offset % 4 == 0 }.map(\.element), [0, 0, 128, 255])
    }

    // ── progress ────────────────────────────────────────────────────────────────────────

    func testProgressGoesFromZeroToOneWithoutGoingBack() {
        let plan = Engine.Plan(denoising: DenoisingPlan(startStep: 0, startSigma: 1, evaluations: 2),
                               seeds: [1, 2], stages: [.text, .denoising, .decoding])
        var a = Engine.Progress()
        var events: [Engine.Event] = [.start(plan), .stage(.text, image: 0), .text(tokens: 3, tokenizer: 0, encoder: 1)]
        for n in 0..<2 {
            events += [.stage(.denoising, image: n),
                           .step(image: n, index: 1, total: 2, sigma: 1, seconds: 10, latentGridHeight: 64, latentGridWidth: 64),
                           .step(image: n, index: 2, total: 2, sigma: 0.5, seconds: 10, latentGridHeight: 64, latentGridWidth: 64),
                           .stage(.decoding, image: n), .decoding(seconds: 1)]
        }
        var previousValue = -1.0
        for e in events {
            a.receive(e)
            XCTAssertGreaterThanOrEqual(a.fraction, previousValue)
            previousValue = a.fraction
        }
        XCTAssertEqual(a.fraction, 1, accuracy: 1e-12)
        XCTAssertEqual(a.estimatedRemaining ?? -1, 0, accuracy: 1e-9)
        var b = Engine.Progress()
        b.receive(.start(plan))
        XCTAssertNil(b.estimatedRemaining, "no estimate before the first step")
        b.receive(.text(tokens: 1, tokenizer: 0, encoder: 0))
        b.receive(.step(image: 0, index: 1, total: 2, sigma: 1, seconds: 10, latentGridHeight: 64, latentGridWidth: 64))
        // Remaining: 1 eval + 0.2 (image 1) + 2.2 (image 2) = 3.4 units at 10 s.
        XCTAssertEqual(b.estimatedRemaining ?? 0, 34, accuracy: 1e-9)
        // B3: a second `start` starts again from zero — the remaining time is not drawn from the
        // previous render's steps (here 10 s), but from its own (1 s).
        b.receive(.start(plan))
        XCTAssertEqual(b.fraction, 0)
        XCTAssertNil(b.estimatedRemaining, "the previous render no longer counts")
        b.receive(.text(tokens: 1, tokenizer: 0, encoder: 0))
        b.receive(.step(image: 0, index: 1, total: 2, sigma: 1, seconds: 1, latentGridHeight: 64, latentGridWidth: 64))
        XCTAssertEqual(b.estimatedRemaining ?? 0, 3.4, accuracy: 1e-9)
    }

    /// A render's metadata: enough to redo it, and no date — and, in img2img, the fact that the
    /// source image is not in it.
    func testTheMetadataOfARender() {
        let r = Engine.Render(model: "anima", prompt: "p", seed: 7, steps: 8,
                             loras: [LoRAEntry("store/niji.safetensors", strength: 0.8)],
                             image: ImageRGB(pixels: [0, 0, 0], height: 1, width: 1),
                             evaluations: 8, strength: 0.6,
                             startStep: 3, startSigma: 0.8, timings: .init(), footprints: .init(), tokens: 1)
        XCTAssertEqual(r.metadata, ["prompt": "p", "seed": "7", "model": "anima", "steps": "8", "format": "1x1",
                                       "Software": "Siliconed", "strength": "0.60",
                                       "image": "source not included", "lora": "niji.safetensors:0.80",
                                       "parameters": r.parameters()])
    }

    /// What a dropped PNG gives back (`Engine.Render.Recipe`): the round trip through the bytes,
    /// several LoRAs in their order, missing keys, unreadable values, another program's PNG.
    func testARecipeReadsBackFromTheMetadata() throws {
        let r = Engine.Render(model: "qwen-image-2.1", prompt: "woman posing in a library, « livres »", seed: 4_000_000_000_123,
                             steps: 6, loras: [LoRAEntry("store/flat.safetensors", strength: 0.8),
                                               LoRAEntry("/x/a:b.safetensors", strength: -0.25)],
                             image: ImageRGB(pixels: [Float](repeating: 0, count: 3 * 16 * 32), height: 16, width: 32), evaluations: 6, strength: nil,
                             startStep: 0, startSigma: 1, timings: .init(), footprints: .init(), tokens: 1)
        let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: PNG.text(try r.png())))
        XCTAssertEqual(recipe.model, "qwen-image-2.1")
        XCTAssertEqual(recipe.prompt, r.prompt)
        XCTAssertEqual(recipe.seed, 4_000_000_000_123)
        XCTAssertEqual(recipe.steps, 6)
        XCTAssertEqual(recipe.width, 32); XCTAssertEqual(recipe.height, 16)
        // File names only, in order; a colon in the name survives (split at the last one).
        XCTAssertEqual(recipe.loras, [LoRAEntry("flat.safetensors", strength: 0.8),
                                      LoRAEntry("a:b.safetensors", strength: -0.25)])
        XCTAssertNil(recipe.strength); XCTAssertFalse(recipe.sourceImageMissing)
        XCTAssertEqual(recipe.unreadable, [])

        // img2img: the strength comes back, and the fact that the source image does not.
        let i2i = try XCTUnwrap(Engine.Render.Recipe(metadata: ["Software": "Siliconed", "strength": "0.60",
                                                                "image": "source not included"]))
        XCTAssertEqual(i2i.strength, 0.6); XCTAssertTrue(i2i.sourceImageMissing)
        // Missing keys are nil, not invented.
        XCTAssertNil(i2i.model); XCTAssertNil(i2i.prompt); XCTAssertNil(i2i.seed); XCTAssertNil(i2i.steps)
        XCTAssertNil(i2i.width); XCTAssertEqual(i2i.loras, [])

        // Unreadable values: nil, and named; a bad LoRA entry is skipped, the good ones kept.
        let bad = try XCTUnwrap(Engine.Render.Recipe(metadata: [
            "Software": "Siliconed", "model": "z-image", "seed": "-3", "steps": "0", "format": "wide",
            "strength": "2", "lora": "flat.safetensors:0.50,broken,:0.3,x.safetensors:nan"]))
        XCTAssertEqual(bad.model, "z-image")
        XCTAssertNil(bad.seed); XCTAssertNil(bad.steps); XCTAssertNil(bad.width)
        XCTAssertNil(bad.strength)
        XCTAssertEqual(bad.loras, [LoRAEntry("flat.safetensors", strength: 0.5)])
        XCTAssertEqual(bad.unreadable, ["format", "lora", "seed", "steps", "strength"])
        // Above the cap, the steps are unreadable too: 2·10⁹ would allocate the schedule (8 GB) first.
        let many = try XCTUnwrap(Engine.Render.Recipe(metadata: ["Software": "Siliconed", "steps": "2000000000"]))
        XCTAssertNil(many.steps); XCTAssertEqual(many.unreadable, ["steps"])
        XCTAssertEqual(Engine.Render.Recipe(metadata: ["Software": "Siliconed", "steps": "\(Request.maximumSteps)"])?.steps,
                       Request.maximumSteps)
        // A key this version does not read (an image of another version carries some) is left alone, not unreadable.
        let other = try XCTUnwrap(Engine.Render.Recipe(metadata: ["Software": "Siliconed", "seed": "7", "faster-rendering": "true"]))
        XCTAssertEqual(other.seed, 7); XCTAssertEqual(other.unreadable, [])

        // Another program's PNG, or none: nothing is interpreted.
        XCTAssertNil(Engine.Render.Recipe(metadata: ["prompt": "{\"3\": {\"class_type\": \"KSampler\"}}"]))
        XCTAssertNil(Engine.Render.Recipe(metadata: ["Software": "ComfyUI", "prompt": "p", "seed": "1"]))
        XCTAssertNil(Engine.Render.Recipe(metadata: PNG.text(Data("not a png".utf8))))
    }

    /// The text always, and mandatory; the image only with an encoder, optional. A module without
    /// a name of its own bears its type's.
    func testTheChainDeclaresItsEntries() throws {
        let without = try Chain(text: FakeTextModule(), denoising: FakeDenoisingModule(), decoding: FakeDecodingModule())
        XCTAssertEqual(without.entries, [Chain.Entry(kind: .text, isRequired: true, module: "FakeTextModule")])
        let with = try Chain(text: FakeTextModule(), denoising: FakeDenoisingModule(), decoding: FakeDecodingModule(),
                              encoding: FakeEncodingModule())
        XCTAssertEqual(with.entries.map(\.kind), [.text, .image])
        XCTAssertEqual(with.entries[1], Chain.Entry(kind: .image, isRequired: false, module: "VAE factice"))
    }
}

/// A counter that a `@Sendable` closure increments.

// ── the `async` facade, on a dummy chain ────────────────────────────────────────────────────
//
// Modules with no map: what is judged here is the plumbing — the engine's queue, the order of the refusals, the stream.

private struct FakeTextModule: TextModule {
    var output: TextFormat { TextFormat("factice") }
}

/// Two evaluations.
private struct FakeDenoisingModule: DenoisingModule {
    var model: String { "factice" }
    var entry: TextFormat { TextFormat("factice") }
    var space: LatentSpace { .flux }
    var defaultSteps: Int { 2 }
    func sigmas(steps: Int) -> [Float] { [1, 0.5, 0] }
    func plan(height: Int, width: Int, steps: Int, start: Int) -> DenoisingPlan {
        DenoisingPlan(startStep: 0, startSigma: 1, evaluations: 2)
    }
}

private struct FakeDecodingModule: DecodingModule {
    var space: LatentSpace { .flux }
}

private struct FakeEncodingModule: ImageEncodingModule {
    var name: String { "VAE factice" }
    var space: LatentSpace { .flux }
}

final class EntriesTests: XCTestCase {
}

final class FacadeTests: XCTestCase {
    private func model() throws -> Model {
        Model(card: .zImage, chain: try Chain(text: FakeTextModule(), denoising: FakeDenoisingModule(), decoding: FakeDecodingModule()))
    }

    /// The request's own refusals come first, then the model's: no family is translated for the engine yet.
    func testTheFacadeRefusesWhatTheEngineDoesNotRunYet() throws {
        do { _ = try Engine().render(Request("p", resolution: 512, seed: 7), model: try model()); XCTFail("rendered") }
        catch { XCTAssertEqual(error as? EngineError, .notOnEngine(model: "z-image"), "\(error)") }
        do { _ = try Engine().render(Request(" ", resolution: 512), model: try model()); XCTFail("rendered") }
        catch { XCTAssertEqual(error as? EngineError, .emptyPrompt, "\(error)") }
    }
}
