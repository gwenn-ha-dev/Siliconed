import Foundation
import os
import XCTest
@testable import Siliconed

/// **The closed enumeration and the preflights** — without a byte of weights: the machine's memory,
/// the catalog's verdict and the license registry are injected, the registry and the lock live in a
/// temporary folder.
final class EngineErrorTests: XCTestCase {

    // ── the enumeration ─────────────────────────────────────────────────────────────────────

    /// Each case has its own stable code — the key an app translates by — and an English sentence.
    func testEveryCaseHasAUniqueCodeAndASentence() {
        let codes = EngineError.samples.map(\.code)
        XCTAssertEqual(Set(codes).count, codes.count, "two cases share a code: \(codes)")
        XCTAssertEqual(codes.count, 45, "a case was added: give it a sample, a code, a sentence")
        for e in EngineError.samples {
            XCTAssertTrue(e.code.allSatisfy { $0.isLowercase || $0 == "_" }, e.code)
            let sentence = e.errorDescription ?? ""
            XCTAssertFalse(sentence.isEmpty, e.code)
            XCTAssertEqual(e.localizedDescription, sentence, e.code)
            XCTAssertTrue("\(e)".hasPrefix(sentence), e.code)
        }
        // The refinements of a reason are stable keys too.
        XCTAssertEqual(EngineError.FormatRefusal.allCases.map(\.rawValue), ["not_multiple", "side_too_long", "too_large"])
        XCTAssertEqual(EngineError.LoRARefusal.allCases.map(\.rawValue), ["not_a_lora", "no_target", "inconsistent"])
        XCTAssertEqual(EngineError.LibraryRefusal.allCases.map(\.rawValue),
                       ["file_size_limit", "read_only", "not_a_library", "already_a_library", "same_library", "copy_damaged"])
        XCTAssertEqual(EngineError.GridRefusal.allCases.map(\.rawValue),
                       ["empty_axis", "same_axis_twice", "no_such_group", "no_such_lora", "group_not_on_axis",
                        "invalid_value", "unreadable_axis", "lora_across_models"])
    }

    /// A refusal of the prompt's own text is an input error with its own case — never an
    /// `internalFailure` ("open an issue"), which would blame the engine for what the user typed.
    func testThePromptsOwnRefusalsHaveTheirCase() {
        XCTAssertEqual(EngineError(Qwen3VLPrompt.Failure.reservedText("<|image_pad|>")),
                       .promptReservedText(text: "<|image_pad|>"))
        XCTAssertEqual(EngineError(Qwen3VLPrompt.Failure.tooLong(tokens: 600, max: 512)),
                       .promptTooLong(tokens: 600, max: 512))
        XCTAssertEqual(EngineError.promptTooLong(tokens: 600, max: 512).code, "prompt_too_long")
        XCTAssertEqual(EngineError.promptReservedText(text: "x").code, "prompt_reserved_text")
        let long = EngineError.promptTooLong(tokens: 600, max: 512).errorDescription ?? ""
        XCTAssertTrue(long.contains("600") && long.contains("512"), long)
        // The template's own count stays a breakdown: the user cannot cause it.
        guard case .internalFailure(component: "prompt", _) = EngineError(Qwen3VLPrompt.Failure.slots(images: 1, slots: 2))
        else { return XCTFail() }
    }

    /// **An image that is not finite is refused, never delivered** (`FiniteImage`, the render's last
    /// guard): one NaN or ±∞ among the latent's values refuses the image, its count goes to the
    /// warnings; large, tiny, subnormal and signed-zero values are finite and pass.
    func testANonFiniteLatentIsRefusedWithAWarning() throws {
        var latent: [Float] = (0..<16 * 64 * 64).map { (i: Int) -> Float in Float(i % 97) / 48 - 1 }
        latent[3] = .greatestFiniteMagnitude; latent[4] = -.leastNonzeroMagnitude; latent[5] = -0
        XCTAssertEqual(FiniteImage.nonFiniteCount(latent), 0)
        let said = OSAllocatedUnfairLock<[String]>(initialState: [])
        try Warnings.during({ line in said.withLock { $0.append(line) } }) {
            XCTAssertTrue(FiniteImage.isFinite(latent, "the final latent", image: 0))
        }
        XCTAssertTrue(said.withLock { $0 }.isEmpty)

        for bad: Float in [.nan, .signalingNaN, .infinity, -.infinity] {
            var broken = latent
            broken[broken.count - 1] = bad; broken[1000] = bad
            XCTAssertEqual(FiniteImage.nonFiniteCount(broken), 2, "\(bad)")
            said.withLock { $0 = [] }
            try Warnings.during({ line in said.withLock { $0.append(line) } }) {
                XCTAssertFalse(FiniteImage.isFinite(broken, "the decoded image", image: 1))
            }
            let lines = said.withLock { $0 }
            XCTAssertEqual(lines.count, 1)
            XCTAssertTrue(lines.first?.contains("image 2: the decoded image carries 2 non-finite values out of 65536") ?? false,
                          "\(lines)")
        }
        // Said to a person without a word of numerics; the code is the stable key the app translates.
        let e = EngineError.nonFiniteImage
        XCTAssertEqual(e.code, "non_finite_image")
        XCTAssertEqual(EngineError(e), e)
        for word in ["NaN", "finite", "latent", "∞"] {
            XCTAssertFalse("\(e)".contains(word), word)
        }
    }

    /// The door: whatever a body throws comes out as an `EngineError`.
    func testTheBoundaryLetsOnlyEngineErrorsOut() {
        XCTAssertThrowsError(try EngineError.boundary { throw Request.Failure.emptyPrompt }) {
            XCTAssertEqual($0 as? EngineError, .emptyPrompt)
        }
        XCTAssertThrowsError(try EngineError.boundary { throw URLError(.networkConnectionLost) }) {
            guard case EngineError.downloadInterrupted = $0 else { return XCTFail("\($0)") }
        }
        XCTAssertEqual(try EngineError.boundary { 7 }, 7)
    }

    /// A model missing from the library is `modelNotInstalled`, under its identifier.
    func testAModelThatIsNotInstalledSaysSo() {
        let empty = Library(root: temporary())
        XCTAssertThrowsError(try Model.named("z-image", in: empty)) {
            XCTAssertEqual($0 as? EngineError, .modelNotInstalled(model: "z-image"))
        }
        XCTAssertThrowsError(try Model.qwenImage21(in: empty)) {
            XCTAssertEqual($0 as? EngineError, .modelNotInstalled(model: "qwen-image-2.1"))
        }
        XCTAssertThrowsError(try Model.named("sd15", in: empty)) {
            XCTAssertEqual($0 as? EngineError, .unknownModel(model: "sd15"))
        }
        XCTAssertNotNil(ModelCard.zImage.missing(in: empty), "the catalog still says which file")
    }

    // ── the preflights ──────────────────────────────────────────────────────────────────────

    /// B of a 16 GB Mac: 10 GiB.
    private let sixteen = MemoryBudget.ceiling(physical: 16 << 30)

    private func preflight(_ card: ModelCard = .zImage, _ w: Int = 1024, _ h: Int = 1024, installed: Bool = true,
                           license: Bool = true, memory: Int? = nil) throws {
        // `memory` is B, the one figure the refusal reads (a 16 GB Mac's by default).
        try Preflight.check(card: card, width: w, height: h, installed: installed, licenseAccepted: license,
                            ceiling: memory ?? sixteen, memory: request(card, w, h))
    }

    /// The card's request on the fixtures' maps (`MemoryFixtures`), 6 P cores, no process base: the
    /// same need on every Mac, the CI's included, maps installed or not.
    private func request(_ card: ModelCard, _ w: Int, _ h: Int, references: Int = 0) -> MemoryNeed.Request {
        var r = card.memoryRequest(width: w, height: h, references: Array(repeating: (w, h), count: references), base: 0)
        r.threads = 6
        r.zImageMaps = MemoryFixtures.zImageMaps()
        r.qwenMaps = MemoryFixtures.qwenMaps()
        r.animaMaps = MemoryFixtures.animaMaps()
        r.krea2Maps = MemoryFixtures.krea2Maps()
        r.mossMaps = MossImageFamilyTests.maps()
        return r
    }

    private func need(_ card: ModelCard, _ w: Int, _ h: Int, references: Int = 0) -> Int {
        card.memoryNeed(request(card, w, h, references: references))!
    }

    /// The order is the contract: installed, license, format, memory — the first that fails throws.
    func testThePreflightsAreCheckedInOrder() {
        XCTAssertNoThrow(try preflight())
        XCTAssertThrowsError(try preflight(.zImage, 256, 256, installed: false, license: false, memory: 1)) {
            XCTAssertEqual($0 as? EngineError, .modelNotInstalled(model: "z-image"))
        }
        XCTAssertThrowsError(try preflight(.qwenImage21, 256, 256, license: false, memory: 1)) {
            XCTAssertEqual($0 as? EngineError, .licenseNotAccepted(model: "qwen-image-2.1"))
        }
        XCTAssertThrowsError(try preflight(.zImage, 256, 512, memory: 1)) {
            XCTAssertEqual($0 as? EngineError, .imageTooSmall(side: 256, minimum: Format.minimumSide))
        }
        XCTAssertThrowsError(try preflight(.zImage, 1000, 1000)) {
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 1000, height: 1000, reason: .notMultiple, bound: 16))
        }
        // No area ceiling: 2048² passes the format rules (B judges it, below); a side past the table does not.
        XCTAssertThrowsError(try preflight(.zImage, 8224, 512)) {
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 8224, height: 512, reason: .sideTooLong, bound: Format.maximumSide))
        }
        // The family's own rules, at the preflight: Qwen-Image-2.1's 32 (848×1216 used to die after the
        // encoder), Anima's 2048 px a side (its reference's RoPE table) — whatever B.
        XCTAssertThrowsError(try preflight(.qwenImage21, 848, 1216, memory: .max)) {
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 848, height: 1216, reason: .notMultiple, bound: 32))
        }
        XCTAssertNoThrow(try preflight(.zImage, 848, 1216, memory: .max))
        XCTAssertThrowsError(try preflight(.anima, 2560, 512, memory: .max)) {
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 2560, height: 512, reason: .sideTooLong, bound: 2048))
        }
        XCTAssertNoThrow(try preflight(.anima, 2048, 768, memory: .max))
        XCTAssertNoThrow(try preflight(.krea2, 2560, 512, memory: .max))
        let qwen = need(.qwenImage21, 1024, 1024)
        XCTAssertThrowsError(try preflight(.qwenImage21, 1024, 1024, memory: qwen - 1)) {
            XCTAssertEqual($0 as? EngineError, .insufficientMemory(needed: qwen, budget: qwen - 1))
        }
    }

    /// **The memory refused is the floor, computed from the request**: on a 16 GB Mac (B = 10 GiB)
    /// every model fits at 1024×1536, every one computed (Krea 2 too: `MemoryNeed.krea2`); with
    /// B = 1 GiB none does at 1024² (Anima needs 1.35 GB there). On the fixtures' maps and 6 P cores: the figures do
    /// not depend on the Mac (the CI's runner, with fewer P cores, once found a generation above an
    /// edit — true there, and no property: no test compares them any more).
    func testTheMemoryNeedIsComputed() {
        for card in ModelCard.allCards {
            let needs = [(512, 512), (832, 1216), (1024, 1024), (1024, 1536)].map { need(card, $0, $1) }
            XCTAssertNoThrow(try preflight(card, 1024, 1536, memory: sixteen), card.id)
            XCTAssertThrowsError(try preflight(card, 1024, 1024, memory: 1 << 30), card.id)
            // Never decreases with the area.
            XCTAssertEqual(needs, needs.sorted(), card.id)
        }
        // An edit is charged its references: more references, never less.
        XCTAssertLessThan(need(.qwenImage21, 1024, 1024, references: 1), need(.qwenImage21, 1024, 1024, references: 3))
    }

    // ── the license registry ────────────────────────────────────────────────────────────────

    func testTheLicenseRegistryIsReadAndWritten() throws {
        let library = Library(root: temporary())
        XCTAssertTrue(library.acceptedLicenses().isEmpty, "nothing is accepted by default")
        XCTAssertFalse(library.isLicenseAccepted(.qwenImage21))
        try library.acceptLicense(.qwenImage21)
        try library.acceptLicense(.zImage)
        XCTAssertTrue(library.isLicenseAccepted(.qwenImage21))
        XCTAssertEqual(Library(root: library.root).acceptedLicenses(),
                       ["qwen-image-2.1": "Qwen Research (non-commercial)", "z-image": "Apache 2.0"])
        // A license whose text changed asks again.
        let file = URL(fileURLWithPath: library.acceptedLicensesFile)
        try Data(#"{"z-image": "MIT"}"#.utf8).write(to: file)
        XCTAssertFalse(library.isLicenseAccepted(.zImage))
        try Data("not json".utf8).write(to: file)
        XCTAssertTrue(library.acceptedLicenses().isEmpty, "an unreadable registry accepts nothing")
    }

    /// Every family says where to read its license: Hub pages at a pinned 40-hex revision — the very
    /// revision the installation downloads from, the DiT's repository first.
    func testEveryLicenseLinksToThePinnedRevision() throws {
        let page = try NSRegularExpression(pattern: "^https://huggingface\\.co/([^/]+/[^/]+)/blob/([0-9a-f]{40})/[^/]+$")
        for f in Family.allCases {
            let card = ModelCard.of(f), urls = card.license.urls
            XCTAssertFalse(urls.isEmpty, "\(f): no license URL")
            let downloaded = Set((Installer.components(f) + Installer.encoder(f) + Installer.dit(f) + Installer.companion(f))
                .map { "\($0.repository.name)@\($0.repository.revision)" })
            for (i, url) in urls.enumerated() {
                let s = url.absoluteString
                let m = try XCTUnwrap(page.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)), s)
                let repository = String(s[Range(m.range(at: 1), in: s)!]), sha = String(s[Range(m.range(at: 2), in: s)!])
                XCTAssertTrue(downloaded.contains("\(repository)@\(sha)"), "\(s): not a repository@revision the installation takes")
                if i == 0 {
                    let dit = try XCTUnwrap(Installer.dit(f).first).repository
                    XCTAssertEqual("\(repository)@\(sha)", "\(dit.name)@\(dit.revision)", "\(f): the model's license comes first")
                }
            }
        }
        XCTAssertEqual(ModelCard.qwenImage21.license.urls.map(\.lastPathComponent), ["LICENSE"])
        let imported = ModelCard(isImported: "x", slug: "x", family: .krea2, map: "/tmp/x.silicon")
        XCTAssertEqual(imported.license.urls, ModelCard.krea2.license.urls, "an imported model reads its family's")
    }

    // ── one render on the machine ───────────────────────────────────────────────────────────

    /// A second holder of the render lock — another process, or here another descriptor — is refused
    /// without waiting; the lock returns when the holder drops it. (The machine's path itself:
    /// `RobustnessTests.testTheRenderLockIsTheMachines`.)
    func testASecondRenderIsRefusedWhileTheLockIsHeld() throws {
        let folder = temporary()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let path = folder.appendingPathComponent("render.lock").path
        var first = try RenderLock.acquire(path)
        XCTAssertNotNil(first)
        XCTAssertThrowsError(try RenderLock.acquire(path)) {
            XCTAssertEqual($0 as? EngineError, .renderAlreadyRunning)
        }
        first = nil
        XCTAssertNotNil(try RenderLock.acquire(path))
        XCTAssertNil(try RenderLock.acquire("/nowhere-\(UUID())/render.lock"), "no lock file, no lock")
    }

    private func temporary() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-errors-\(UUID())")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}
