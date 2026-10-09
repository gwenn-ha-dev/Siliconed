import Foundation
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
        XCTAssertEqual(codes.count, 38, "a case was added: give it a sample, a code, a sentence")
        for e in EngineError.samples {
            XCTAssertTrue(e.code.allSatisfy { $0.isLowercase || $0 == "_" }, e.code)
            let sentence = e.errorDescription ?? ""
            XCTAssertFalse(sentence.isEmpty, e.code)
            XCTAssertEqual(e.localizedDescription, sentence, e.code)
            XCTAssertTrue("\(e)".hasPrefix(sentence), e.code)
        }
        // The refinements of a reason are stable keys too.
        XCTAssertEqual(EngineError.FormatRefusal.allCases.map(\.rawValue), ["not_multiple", "too_large"])
        XCTAssertEqual(EngineError.LoRARefusal.allCases.map(\.rawValue), ["not_a_lora", "no_target", "inconsistent"])
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

    private let sixteen = 16 << 30, eight = 8 << 30

    private func preflight(_ card: ModelCard = .zImage, _ w: Int = 1024, _ h: Int = 1024, installed: Bool = true,
                           license: Bool = true, memory: Int? = nil) throws {
        // `memory` is what the budget makes available (no reserve here: the reserve is `MemoryBudgetTests`').
        try Preflight.check(card: card, width: w, height: h, installed: installed, licenseAccepted: license,
                            budget: MemoryBudget(reclaimable: memory ?? sixteen, reserve: 0))
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
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 1000, height: 1000, reason: .notMultiple))
        }
        XCTAssertThrowsError(try preflight(.zImage, 2048, 2048)) {
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 2048, height: 2048, reason: .tooLarge))
        }
        XCTAssertThrowsError(try preflight(.zImage, 1024, 1024, memory: 2 << 30)) {
            XCTAssertEqual($0 as? EngineError, .insufficientMemory(needed: 2_214_000_000, available: 2 << 30))
        }
    }

    /// **The memory refused is the floor, a measurement, not a guess**: with 8 GiB available every
    /// model fits at 1024² and up to 1024×1536 (Krea 2's 7.8 GB is the largest peak measured); with
    /// 2 GiB none does at 1024² (Z-Image's banded decoder, brought its 1024² under 4 GB).
    func testTheMemoryNeedIsTheWorstMeasuredPeak() {
        for card in ModelCard.allCards {
            XCTAssertNoThrow(try preflight(card, 1024, 1536, memory: eight), card.id)
            XCTAssertThrowsError(try preflight(card, 1024, 1024, memory: 2 << 30), card.id)
            // Never decreases with the area, and never zero.
            let needs = [(512, 512), (832, 1216), (1024, 1024), (1024, 1536)].map { card.memoryNeed(width: $0, height: $1) }
            XCTAssertEqual(needs, needs.sorted(), card.id)
            XCTAssertGreaterThan(needs[0], 0, card.id)
        }
        XCTAssertEqual(ModelCard.zImage.memoryNeed(width: 512, height: 512), 1_695_000_000)
        // The economical plan's floor (the default plan's 3.774 GB buys time above it).
        XCTAssertEqual(ModelCard.zImage.memoryNeed(width: 1024, height: 1536), 3_200_000_000)
        // Anima's 832×1216 peaks above its 1024²: the 1024² takes the larger.
        XCTAssertEqual(ModelCard.anima.memoryNeed(width: 1024, height: 1024), 5_300_000_000)
        // Never measured at 1024×1536: the largest peak any family measured there.
        XCTAssertEqual(ModelCard.klein4b.memoryNeed(width: 1024, height: 1536), 7_800_000_000)
        XCTAssertEqual(ModelCard.qwenImage21.memoryNeed(width: 1024, height: 1536), 4_740_000_000)
        // A later measurement replaced the earlier 4.68 GB (a decoder since banded): an edit below 1024² holds the 1024² tier.
        XCTAssertEqual(ModelCard.qwenImage21.memoryNeed(width: 1024, height: 1024), 4_510_000_000)
        // Z-Image's 1024² floor is the economical plan's denoising, its decoding a few MB below.
        XCTAssertEqual(ModelCard.zImage.memoryNeed(width: 1024, height: 1024), 2_214_000_000)
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
        try library.revokeLicense(.qwenImage21)
        XCTAssertFalse(library.isLicenseAccepted(.qwenImage21))
        XCTAssertTrue(library.isLicenseAccepted(.zImage))
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
            let downloaded = Set((Installer.components(f) + Installer.encoder(f) + Installer.dit(f)
                                  + [Installer.turbo(f)].compactMap { $0 })
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
        XCTAssertEqual(ModelCard.qwenImage21.license.urls.map(\.lastPathComponent), ["LICENSE", "LICENSE"])
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
