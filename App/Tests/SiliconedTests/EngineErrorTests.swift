import Foundation
import os
import XCTest
@testable import SiliCore
@testable import Siliconed

/// **The closed enumeration and the preflights** — without a byte of weights: the catalog's verdict and
/// the license registry are injected, the registry and the lock live in a temporary folder.
final class EngineErrorTests: XCTestCase {

    // ── the enumeration ─────────────────────────────────────────────────────────────────────

    /// Each case has its own stable code — the key an app translates by — and an English sentence.
    func testEveryCaseHasAUniqueCodeAndASentence() {
        let codes = EngineError.samples.map(\.code)
        XCTAssertEqual(Set(codes).count, codes.count, "two cases share a code: \(codes)")
        XCTAssertEqual(codes.count, 49, "a case was added: give it a sample, a code, a sentence")
        for e in EngineError.samples {
            XCTAssertTrue(e.code.allSatisfy { $0.isLowercase || $0 == "_" }, e.code)
            let sentence = e.errorDescription ?? ""
            XCTAssertFalse(sentence.isEmpty, e.code)
            XCTAssertEqual(e.localizedDescription, sentence, e.code)
            XCTAssertTrue("\(e)".hasPrefix(sentence), e.code)
        }
        // The refinements of a reason are stable keys too.
        XCTAssertEqual(EngineError.FormatRefusal.allCases.map(\.rawValue), ["not_multiple", "side_too_long"])
        XCTAssertEqual(EngineError.LoRARefusal.allCases.map(\.rawValue), ["not_a_lora", "no_target", "inconsistent", "unsupported_kind"])
        XCTAssertEqual(EngineError.LibraryRefusal.allCases.map(\.rawValue),
                       ["file_size_limit", "read_only", "not_a_library", "already_a_library", "same_library", "copy_damaged"])
        XCTAssertEqual(EngineError.GridRefusal.allCases.map(\.rawValue),
                       ["empty_axis", "same_axis_twice", "no_such_group", "no_such_lora", "group_not_on_axis",
                        "invalid_value", "unreadable_axis", "lora_across_models", "no_start_image"])
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
        // The engine's refusal under B is a lack of memory, with its numbers — not an internal failure.
        XCTAssertThrowsError(try EngineError.boundary { throw OverBudget(needed: 5 << 30, budget: 4 << 30, description: "text") }) {
            XCTAssertEqual($0 as? EngineError, .insufficientMemory(needed: 5 << 30, budget: 4 << 30))
        }
    }

    /// A model missing from the library is `modelNotInstalled`, under its identifier.
    func testAModelThatIsNotInstalledSaysSo() {
        let empty = Library(root: temporary())
        XCTAssertThrowsError(try Model.named("z-image", in: empty)) {
            XCTAssertEqual($0 as? EngineError, .modelNotInstalled(model: "z-image"))
        }
        XCTAssertThrowsError(try Model.named("sd15", in: empty)) {
            XCTAssertEqual($0 as? EngineError, .unknownModel(model: "sd15"))
        }
        XCTAssertNotNil(ModelCard.zImage.missing(in: empty), "the catalog still says which file")
    }

    // ── the preflights ──────────────────────────────────────────────────────────────────────

    private func preflight(_ card: ModelCard = .zImage, _ w: Int = 1024, _ h: Int = 1024, license: Bool = true) throws {
        try Preflight.check(card: card, width: w, height: h, licenseAccepted: license)
    }

    /// The order is the contract: installed (the model's construction, `Model.named`), license, format — the first
    /// that fails throws. Memory is the engine's: it plans the render under B or refuses.
    func testThePreflightsAreCheckedInOrder() {
        XCTAssertNoThrow(try preflight())
        XCTAssertThrowsError(try preflight(.krea2, 256, 256, license: false)) {
            XCTAssertEqual($0 as? EngineError, .licenseNotAccepted(model: "krea2"))
        }
        XCTAssertThrowsError(try preflight(.zImage, 256, 512)) {
            XCTAssertEqual($0 as? EngineError, .imageTooSmall(side: 256, minimum: Format.minimumSide))
        }
        XCTAssertThrowsError(try preflight(.zImage, 1000, 1000)) {
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 1000, height: 1000, reason: .notMultiple, bound: 16))
        }
        // No area ceiling: 2048² passes the format rules (B is the engine's); a side past the table does not.
        XCTAssertThrowsError(try preflight(.zImage, 8224, 512)) {
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 8224, height: 512, reason: .sideTooLong, bound: Format.maximumSide))
        }
        // The family's own rules, at the preflight: Anima's 2048 px a side (its reference's RoPE table) — whatever B.
        XCTAssertNoThrow(try preflight(.zImage, 848, 1216))
        XCTAssertThrowsError(try preflight(.anima, 2560, 512)) {
            XCTAssertEqual($0 as? EngineError, .formatRefused(width: 2560, height: 512, reason: .sideTooLong, bound: 2048))
        }
        XCTAssertNoThrow(try preflight(.anima, 2048, 768))
        XCTAssertNoThrow(try preflight(.krea2, 2560, 512))
    }

    // ── the license registry ────────────────────────────────────────────────────────────────

    func testTheLicenseRegistryIsReadAndWritten() throws {
        let library = Library(root: temporary())
        XCTAssertTrue(library.acceptedLicenses().isEmpty, "nothing is accepted by default")
        XCTAssertFalse(library.isLicenseAccepted(.krea2))
        try library.acceptLicense(.krea2)
        try library.acceptLicense(.zImage)
        XCTAssertTrue(library.isLicenseAccepted(.krea2))
        XCTAssertEqual(Library(root: library.root).acceptedLicenses(),
                       ["krea2": ModelCard.krea2.license.text, "z-image": "Apache 2.0"])
        // A license whose text changed asks again.
        let file = URL(fileURLWithPath: library.acceptedLicensesFile)
        try Data(#"{"z-image": "MIT"}"#.utf8).write(to: file)
        XCTAssertFalse(library.isLicenseAccepted(.zImage))
        try Data("not json".utf8).write(to: file)
        XCTAssertTrue(library.acceptedLicenses().isEmpty, "an unreadable registry accepts nothing")
    }

    // ── one render on the machine ───────────────────────────────────────────────────────────

    /// A second holder of the render lock — another process, or here another descriptor — is refused
    /// without waiting; the lock returns when the holder drops it. (The machine's path itself:
    /// `RobustnessTests.testTheRenderLockIsTheMachines`.)
    func testASecondRenderIsRefusedWhileTheLockIsHeld() throws {
        let folder = temporary()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let path = folder.appendingPathComponent("render.lock").path
        var first = try RenderLock.acquire(path) { _ in }
        XCTAssertNotNil(first)
        XCTAssertThrowsError(try RenderLock.acquire(path) { _ in }) {
            XCTAssertEqual($0 as? EngineError, .renderAlreadyRunning)
        }
        first = nil
        XCTAssertNotNil(try RenderLock.acquire(path) { _ in })
        XCTAssertNil(try RenderLock.acquire("/nowhere-\(UUID())/render.lock") { _ in }, "no lock file, no lock")
    }

    private func temporary() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-errors-\(UUID())")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}
