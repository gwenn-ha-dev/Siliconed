import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import Siliconed

/// **What an app reads from the library** — catalog, errors, cancellation token, preview
/// projection, PNG chunks, progress. No map, no GPU.
final class AppTests: XCTestCase {

    /// These renders stop before any weight: the preflight's memory is not what they test, and the
    /// machine's state would make them depend on what else runs (`MemoryBudget`). A budget that fits all.
    override class func setUp() {
        super.setUp()
        setenv(MemoryBudget.forcingVariable, "64", 1)
    }

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

    /// A memory refusal offers the largest format that fits: the chosen orientation first, then the
    /// area; never under 512 on a side, never the current one, nothing when none fits.
    func testTheLargestFittingFormat() {
        let need = { (f: RecommendedFormat) in f.width * f.height }   // a need proportional to the area
        let formats = ModelCard.recommendedFormats + [RecommendedFormat(512, 768), RecommendedFormat(384, 512)]
        let portrait = RecommendedFormat(832, 1216)
        // Everything fits but the current one: a portrait stays a portrait, the first listed of equals.
        XCTAssertEqual(RecommendedFormat.largestFitting(formats, available: .max, current: portrait, need: need),
                       RecommendedFormat(896, 1152))
        // Only up to 512×768 fits: the portrait one, not 512².
        XCTAssertEqual(RecommendedFormat.largestFitting(formats, available: 512 * 768, current: portrait, need: need),
                       RecommendedFormat(512, 768))
        // No landscape fits: the largest of any orientation.
        XCTAssertEqual(RecommendedFormat.largestFitting(formats, available: 512 * 768, current: RecommendedFormat(1216, 832),
                                                        need: need),
                       RecommendedFormat(512, 768))
        // 384×512 fits by its need, but is under the 512 floor: nothing to offer.
        XCTAssertNil(RecommendedFormat.largestFitting(formats, available: 512 * 512 - 1, current: portrait, need: need))
    }

    /// The sheet says what the module does: default steps, space — a single truth.
    func testTheCardMatchesTheDenoiser() {
        let denoisers: [String: any DenoisingModule] = [
            "z-image": ZImageDenoisingModule(map: ""), "anima": AnimaDenoisingModule(map: ""), "krea2": Krea2DenoisingModule(map: ""),
            "klein-4b": KleinDenoisingModule(map: ""), "ernie-image": ErnieDenoisingModule(map: ""),
            "qwen-image-2.1": QwenImage21DenoisingModule(map: "", turbo: "", scheduler: .init())]
        // Qwen-Image-2.1 joins last when the zero swap holds at its worst case (`ModelCard.qwenImage21`):
        // both states pass, nothing else does.
        let published = ["z-image", "anima", "krea2", "klein-4b", "ernie-image"]
        XCTAssertTrue(Model.identifiers == published || Model.identifiers == published + ["qwen-image-2.1"],
                      "\(Model.identifiers)")
        // Hidden card, hidden family; visible card, offered family — never one without the other.
        XCTAssertEqual(Family.offered.contains(.qwenImage21), ModelCard.allCards.contains(.qwenImage21))
        // The hidden card is judged too: it is already the one `Model.named` and `of(_:)` hand out.
        for card in ModelCard.allCards + (ModelCard.allCards.contains(.qwenImage21) ? [] : [.qwenImage21]) {
            let d = denoisers[card.id]!
            XCTAssertEqual(card.defaultSteps, d.defaultSteps)
            XCTAssertEqual(card.space, d.space)
            XCTAssertEqual(d.model, card.id)
            XCTAssertEqual(ModelCard.of(card.family), card)
        }
        XCTAssertFalse(ModelCard.anima.license.commercial)
        XCTAssertTrue(ModelCard.krea2.license.requiredFilter)
        XCTAssertFalse(ModelCard.qwenImage21.license.commercial)
        XCTAssertFalse(ModelCard.qwenImage21.license.requiredFilter)
    }

    /// An empty library does not throw: it has nothing, and says what is missing.
    func testAnEmptyLibraryDoesNotThrow() {
        let empty = Library(root: URL(fileURLWithPath: "/nulle-part-\(UUID())"))
        XCTAssertEqual(empty.models().count, 0)
        XCTAssertEqual(empty.loras().count, 0)
        XCTAssertNotNil(ModelCard.krea2.missing(in: empty))
    }

    private func header(_ target: String?, name: String = "Niji_semi_realism_v5", kind: String = "lora") -> [String: Any] {
        var e: [String: Any] = ["kind": kind, "nom": name, "rang": 64,
                                "source": ["metadata": ["modelspec.resolution": "1024x1024", "ss_sd_model_name": ""]]]
        if let target { e["cible"] = ["modele": target, "kind": "\(target)-dit"] }
        return e
    }

    func testALoRACardIsReadFromTheHeader() throws {
        let f = try XCTUnwrap(LoRACard(header: header("anima"), path: "/s/niji.lora.silicon"))
        XCTAssertEqual(f.name, "Niji semi realism v5")
        XCTAssertEqual(f.target, "anima")
        XCTAssertEqual(f.rank, 64)
        XCTAssertEqual(f.resolution, "1024x1024")
        XCTAssertNil(f.trainedOn, "an empty metadata field is not a value")
        XCTAssertEqual(f.entry(strength: 0.5), LoRAEntry("/s/niji.lora.silicon", strength: 0.5))
        XCTAssertNil(LoRACard(header: header(nil), path: "x"), "without a target: refused at render time, absent from the catalog")
        XCTAssertNil(LoRACard(header: header("anima", kind: "z-image-turbo-dit"), path: "x"))
    }

    func testTheFilterKeepsOnlyCompatibleLoRAs() {
        let cards = [("krea2", "Incase_Krea_v2"), ("anima", "Niji"), ("z-image", "PopArt"), ("anima", "Autre")]
            .compactMap { LoRACard(header: header($0.0, name: $0.1), path: "/s/\($0.1)") }
        XCTAssertEqual(LoRACard.filter(cards, for: "anima").map(\.name), ["Autre", "Niji"])
        XCTAssertEqual(LoRACard.filter(cards, for: "krea2").map(\.name), ["Incase Krea v2"])
        XCTAssertEqual(LoRACard.filter(cards, for: nil).count, 4)
        XCTAssertEqual(LoRACard.filter(cards, for: "flux").count, 0)
    }

    // ── the errors ─────────────────────────────────────────────────────────────────────────

    /// **Every internal failure folds into its named case** at the door — and none falls back to
    /// "The operation couldn't be completed".
    func testInternalFailuresFoldIntoTheirCase() {
        let folds: [(any Error, EngineError)] = [
            (Strength.Failure.outOfBounds(2), .strengthOutOfRange(strength: 2)),
            (LoRA.Failure.wrongTarget(path: "p", name: "PopArt", forgedFor: "z-image", model: "krea2"),
             .loraForOtherModel(lora: "PopArt", target: "z-image", model: "krea2")),
            (LoRA.Failure.missingTarget("/l.lora.silicon"), .unsupportedLoRA(file: "/l.lora.silicon", reason: .noTarget)),
            (Model.Failure.unknown("sd15"), .unknownModel(model: "sd15")),
            (MissingFile("/x", .published), .fileMissing(path: "/x")),
            (MissingFile("/x.silicon", .map), .fileMissing(path: "/x.silicon")),
            (Chain.Failure.latentIncompatible(output: .flux, entry: .qwenImage),
             .incompatibleChain(output: "\(LatentSpace.flux)", input: "\(LatentSpace.qwenImage)")),
            (Artifact.Failure.badHeader("x"), .corruptMap(file: nil)),
            (Artifact.Failure.inFile("/m", .truncated("x")), .corruptMap(file: "/m")),
            (Artifact.Failure.cannotOpen("/m", ENOENT), .fileMissing(path: "/m")),
            (Artifact.Failure.cannotOpen("/m", EACCES), .fileUnreadable(path: "/m")),
            (Safetensors.Failure.cannotOpen("/vae", EACCES), .fileUnreadable(path: "/vae")),
            (Safetensors.Failure.cannotStat("/vae", EIO), .fileUnreadable(path: "/vae")),
            (Safetensors.Failure.cannotMap("/vae", ENOMEM), .fileUnreadable(path: "/vae")),
            (Safetensors.Failure.missingTensor(file: "/vae", name: "w"), .corruptMap(file: "/vae")),
            (ImageRGB.Failure.unreadable("/a.png"), .imageUnreadable(file: "/a.png")),
            (ImageRGB.Failure.encoding, .imageEncodingFailed),
            (Sampler.Failure.belowFloor(side: 32), .imageTooSmall(side: 256, minimum: 512)),
            (Request.Failure.emptyPrompt, .emptyPrompt),
            (Request.Failure.steps(0), .invalidSteps(steps: 0, allowed: [])),
            (QwenImage21Pipeline.Failure.steps(8), .invalidSteps(steps: 8, allowed: [5, 6, 7, 9])),
            (EngineSettings.Failure.alreadyRead(profileRead: "/a", requested: "/b"),
             .settingsAlreadyLoaded(loaded: "/a", requested: "/b")),
            (CancellationError(), .cancelled),
            (NSError(domain: NSPOSIXErrorDomain, code: Int(ENOSPC)), .diskFull(needed: nil, available: nil)),
            (EngineError.renderAlreadyRunning, .renderAlreadyRunning),
        ]
        for (failure, expected) in folds {
            XCTAssertEqual(EngineError(failure), expected, "\(failure)")
        }
        for e in [Arena.Failure.allocationFailed(1) as any Error, Sampler.Failure.missingSide, GEMM.Failure.noDevice] {
            guard case .internalFailure = EngineError(e) else { return XCTFail("\(e)") }
        }
        for e in folds.map(\.1) + [EngineError(Arena.Failure.allocationFailed(1))] {
            let text = e.localizedDescription
            XCTAssertFalse(text.contains("couldn’t be completed") || text.contains("couldn't be completed"), text)
            XCTAssertEqual(text, e.errorDescription)
        }
    }

    /// "File absent" is an error of its own — not a bad header.
    func testAMissingFileIsNotABadHeader() {
        let path = "/nulle-part-\(UUID()).silicon"
        XCTAssertThrowsError(try Artifact(path: path)) {
            XCTAssertEqual($0 as? MissingFile, MissingFile(path, .map))
        }
        XCTAssertThrowsError(try Artifact.header(path)) {
            XCTAssertEqual($0 as? MissingFile, MissingFile(path, .map))
        }
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

    /// The context relays the token, and a step done on a cancelled render throws after emitting.
    func testAStepDoneOnACancelledRenderThrows() {
        let token = Cancellation()
        let receivedEvents = Counter()
        let context = Context(reproducible: true, cancellation: token) { if case .step = $0 { receivedEvents.add() } }
        context.beginDenoising(evaluations: 3)
        let x = [Float](repeating: 0, count: 16), v = x
        XCTAssertNoThrow(try context.stepDone(sigma: 1, seconds: 0, evaluatedHeight: 1, evaluatedWidth: 1, space: .flux, x: x, v: v,
                                              nextSigma: 0.5, fullHeight: 1, fullWidth: 1))
        token.cancel()
        XCTAssertThrowsError(try context.stepDone(sigma: 0.5, seconds: 0, evaluatedHeight: 1, evaluatedWidth: 1, space: .flux, x: x, v: v,
                                                  nextSigma: 0, fullHeight: 1, fullWidth: 1))
        XCTAssertEqual(receivedEvents.value, 2)
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

    /// Both spaces have their fitted coefficients, 16 × 3, finite and non-zero.
    func testTheTwoSpacesHaveAProjection() throws {
        for space in [LatentSpace.flux, .qwenImage] {
            let p = try XCTUnwrap(space.preview)
            XCTAssertEqual(p.channels, space.channels)
            XCTAssertTrue(p.weights.allSatisfy(\.isFinite) && p.weights.contains { $0 != 0 })
        }
        XCTAssertNotEqual(LatentSpace.flux.preview, LatentSpace.qwenImage.preview)
    }

    // ── images and PNG ────────────────────────────────────────────────────────────────

    /// The PNG carries its metadata (UTF-8 included), reads back, and two encodings are identical.
    func testThePNGCarriesItsMetadata() throws {
        let image = ImageRGB(pixels: (0..<(3 * 4 * 6)).map { Float($0 % 7) / 3 - 1 }, height: 4, width: 6)
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
        let plan = Engine.Plan(denoising: DenoisingPlan(startStep: 0, startSigma: 1, evaluations: 2, reduced: 0,
                                                            reducedHeight: 64, reducedWidth: 64),
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
        // The spectral: two half-size steps (3 s) say nothing about the five full ones that follow.
        let spectral = Engine.Plan(denoising: DenoisingPlan(startStep: 0, startSigma: 1, evaluations: 7, reduced: 2,
                                                                reducedHeight: 64, reducedWidth: 64),
                                   seeds: [1], stages: [.text, .denoising, .decoding])
        var c = Engine.Progress()
        c.receive(.start(spectral))
        c.receive(.text(tokens: 1, tokenizer: 0, encoder: 0))
        for i in 1...2 {
            c.receive(.step(image: 0, index: i, total: 7, sigma: 1, seconds: 3, latentGridHeight: 64, latentGridWidth: 64))
            XCTAssertNil(c.estimatedRemaining, "no estimate drawn from the reduced steps alone")
        }
        c.receive(.step(image: 0, index: 3, total: 7, sigma: 1, seconds: 10, latentGridHeight: 128, latentGridWidth: 128))
        // Remaining: 4 full ones at 10 s + a decoding at 0.2 of a full step.
        XCTAssertEqual(c.estimatedRemaining ?? 0, 42, accuracy: 1e-9)
    }

    /// A render's metadata: enough to redo it, and no date — and, in img2img, the fact that the
    /// source image is not in it.
    func testTheMetadataOfARender() {
        let r = Engine.Render(model: "anima", prompt: "p", seed: 7, steps: 8,
                             loras: [LoRAEntry("store/niji.lora.silicon", strength: 0.8)], loraSummary: nil,
                             image: ImageRGB(pixels: [0, 0, 0], height: 1, width: 1), reproducible: true,
                             evaluations: 8, sketch: nil, spectral: 0, strength: 0.6,
                             startStep: 3, startSigma: 0.8, timings: .init(), footprints: .init(), tokens: 1)
        XCTAssertEqual(r.metadata, ["prompt": "p", "seed": "7", "model": "anima", "steps": "8", "format": "1x1",
                                       "reproducible": "true", "Software": "Siliconed", "strength": "0.60",
                                       "image": "source not included", "lora": "niji.lora.silicon:0.80"])
    }

    /// What a dropped PNG gives back (`Engine.Render.Recipe`): the round trip through the bytes,
    /// several LoRAs in their order, missing keys, unreadable values, another program's PNG.
    func testARecipeReadsBackFromTheMetadata() throws {
        let r = Engine.Render(model: "qwen-image-2.1", prompt: "woman posing in a library, « livres »", seed: 4_000_000_000_123,
                             steps: 6, loras: [LoRAEntry("store/flat.lora.silicon", strength: 0.8),
                                               LoRAEntry("/x/a:b.lora.silicon", strength: -0.25)], loraSummary: nil,
                             image: ImageRGB(pixels: [Float](repeating: 0, count: 3 * 16 * 32), height: 16, width: 32),
                             reproducible: false, evaluations: 6, sketch: nil, spectral: 0, strength: nil,
                             startStep: 0, startSigma: 1, timings: .init(), footprints: .init(), tokens: 1)
        let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: PNG.text(try r.png())))
        XCTAssertEqual(recipe.model, "qwen-image-2.1")
        XCTAssertEqual(recipe.prompt, r.prompt)
        XCTAssertEqual(recipe.seed, 4_000_000_000_123)
        XCTAssertEqual(recipe.steps, 6)
        XCTAssertEqual(recipe.width, 32); XCTAssertEqual(recipe.height, 16)
        // File names only, in order; a colon in the name survives (split at the last one).
        XCTAssertEqual(recipe.loras, [LoRAEntry("flat.lora.silicon", strength: 0.8),
                                      LoRAEntry("a:b.lora.silicon", strength: -0.25)])
        XCTAssertNil(recipe.strength); XCTAssertFalse(recipe.sourceImageMissing)
        XCTAssertEqual(recipe.reproducible, false)
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
            "reproducible": "maybe", "strength": "2", "lora": "flat.lora.silicon:0.50,broken,:0.3,x.lora.silicon:nan"]))
        XCTAssertEqual(bad.model, "z-image")
        XCTAssertNil(bad.seed); XCTAssertNil(bad.steps); XCTAssertNil(bad.width); XCTAssertNil(bad.reproducible)
        XCTAssertNil(bad.strength)
        XCTAssertEqual(bad.loras, [LoRAEntry("flat.lora.silicon", strength: 0.5)])
        XCTAssertEqual(bad.unreadable, ["format", "lora", "reproducible", "seed", "steps", "strength"])
        // Above the cap, the steps are unreadable too: 2·10⁹ would allocate the schedule (8 GB) first.
        let many = try XCTUnwrap(Engine.Render.Recipe(metadata: ["Software": "Siliconed", "steps": "2000000000"]))
        XCTAssertNil(many.steps); XCTAssertEqual(many.unreadable, ["steps"])
        XCTAssertEqual(Engine.Render.Recipe(metadata: ["Software": "Siliconed", "steps": "\(Request.maximumSteps)"])?.steps,
                       Request.maximumSteps)

        // Another program's PNG, or none: nothing is interpreted.
        XCTAssertNil(Engine.Render.Recipe(metadata: ["prompt": "{\"3\": {\"class_type\": \"KSampler\"}}"]))
        XCTAssertNil(Engine.Render.Recipe(metadata: ["Software": "ComfyUI", "prompt": "p", "seed": "1"]))
        XCTAssertNil(Engine.Render.Recipe(metadata: PNG.text(Data("not a png".utf8))))
    }
}

/// A counter that a `@Sendable` closure increments.
private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func add() { lock.lock(); n += 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return n }
}

// ── the `async` facade, on a dummy chain ────────────────────────────────────────────────────
//
// Modules with no map or GPU: what is judged here is the plumbing — the engine's queue, the
// token tied to the `Task`, the event stream, the batch that encodes the text only once.

private struct FakeTextModule: TextModule {
    var output: TextFormat { TextFormat("factice") }
    func encoder(_ prompt: String, context: Context) throws -> Conditioning {
        Conditioning(format: output, rows: 1, width: 1, values: [0], tokens: 1)
    }
}

/// Two evaluations; `lent`: each waits 2 s, checking the token every millisecond.
private struct FakeDenoisingModule: DenoisingModule {
    var lent = false
    var model: String { "factice" }
    var entry: TextFormat { TextFormat("factice") }
    var space: LatentSpace { .flux }
    var defaultSteps: Int { 2 }
    func sigmas(steps: Int) -> [Float] { [1, 0.5, 0] }
    func plan(height: Int, width: Int, steps: Int, start: Int, withLoRA: Bool) -> DenoisingPlan {
        DenoisingPlan(startStep: 0, startSigma: 1, evaluations: 2, reduced: 0,
                         reducedHeight: height, reducedWidth: width)
    }
    func denoise(_ latent: inout Latent, text: Conditioning, steps: Int, start: Int, lora: LoRA?,
                   context: Context) throws {
        let n = latent.values.count
        try euler(&latent, sigmas: sigmas(steps: steps), context: context) { x, _ in
            if lent { for _ in 0..<2000 { try context.check(); usleep(1000) } }
            return [Float](repeating: 0.1, count: n)
        }
    }
}

private struct FakeDecodingModule: DecodingModule {
    var space: LatentSpace { .flux }
    func decode(_ latent: Latent, context: Context) throws -> ImageRGB {
        ImageRGB(pixels: [Float](repeating: latent.values[0], count: 3 * latent.height * 8 * latent.width * 8),
                 height: latent.height * 8, width: latent.width * 8)
    }
}

private struct FakeEncodingModule: ImageEncodingModule {
    var name: String { "VAE factice" }
    var space: LatentSpace { .flux }
    func encoder(_ image: ImageRGB, context: Context) throws -> Latent {
        Latent(space: space, height: image.height / 8, width: image.width / 8,
               values: [Float](repeating: 0, count: 16 * image.height / 8 * image.width / 8))
    }
}

final class EntriesTests: XCTestCase {
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

final class FacadeTests: XCTestCase {
    private func model(lent: Bool = false) throws -> Model {
        Model(card: .zImage, chain: try Chain(text: FakeTextModule(), denoising: FakeDenoisingModule(lent: lent),
                                                  decoding: FakeDecodingModule()))
    }

    func testTheFacadeRendersWithoutBlocking() async throws {
        var request = Request("p", resolution: 512, seed: 7)
        request.previews = true
        let renderResult = try await Engine().render(request, model: try model())
        XCTAssertEqual(renderResult.seed, 7)
        XCTAssertEqual(renderResult.image.height, 512)
    }

    /// Cancelling the `Task` raises the token: `EngineError.cancelled`, well before the slow render's 4 s.
    func testCancellingTheTaskCancelsTheRender() async throws {
        let m = try model(lent: true)
        let begin = Date()
        let task = Task { try await Engine().render(Request("p", resolution: 512), model: m) }
        try await Task.sleep(nanoseconds: 50_000_000)
        task.cancel()
        do { _ = try await task.value; XCTFail("the render should have been cancelled") }
        catch { XCTAssertEqual(error as? EngineError, .cancelled, "\(error)") }
        XCTAssertLessThan(Date().timeIntervalSince(begin), 1.5)
    }

    /// A batch's stream: one text, two images, steps numbered 1…2, one preview per step, and each
    /// image as `.image` — the result is in the stream.
    func testABatchsStreamCarriesItsImages() async throws {
        var request = Request("p", resolution: 512)
        request.previews = true
        var texts = 0, step: [Int] = [], previews = 0, images: [UInt64] = []
        for try await event in Engine().events(request, seeds: [3, 4], model: try model()) {
            switch event {
            case .stage(.text, _): texts += 1
            case let .step(_, index, total, _, _, _, _): step.append(index); XCTAssertEqual(total, 2)
            case .preview(let a): previews += 1; XCTAssertEqual(a.rgba.count, 64 * 64 * 4)
            case let .image(_, renderResult): images.append(renderResult.seed)
            default: break
            }
        }
        XCTAssertEqual(texts, 1)
        XCTAssertEqual(step, [1, 2, 1, 2])
        XCTAssertEqual(previews, 4)
        XCTAssertEqual(images, [3, 4])
    }
}
