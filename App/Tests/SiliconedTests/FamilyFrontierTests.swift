import Foundation
import XCTest
@testable import Siliconed

/// **A family lives in its folder**: the interface and the rest of the library read its traits and never
/// name it. A new family is its folder, its case in `Family` and its stages in `Verification` — nothing else.
final class FamilyFrontierTests: XCTestCase {
    private static let sources = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../../Sources").standardizedFileURL

    private func swiftFiles(under folder: String) -> [URL] {
        let root = Self.sources.appendingPathComponent(folder)
        let all = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)?.compactMap { $0 as? URL } ?? []
        return all.filter { $0.pathExtension == "swift" }
    }

    /// The code of the interface and of the library outside the families' folders names no family: neither a case of
    /// `Family`, nor a type of a family's folder, nor its identifier in a string. Comments may: they tell what was
    /// measured, on which model.
    func testNothingOutsideAFamilysFolderNamesIt() throws {
        let cases = Family.allCases.map { "\($0)" }                       // zImage, anima, …
        let ids = Family.allCases.map { NSRegularExpression.escapedPattern(for: $0.rawValue) }
        let types = cases.map { $0.prefix(1).uppercased() + $0.dropFirst() }   // ZImage…, Anima…
        let pattern = #"\.("# + cases.joined(separator: "|") + #")\b|\b("# + types.joined(separator: "|") + #")[A-Z]\w*|"("#
            + ids.joined(separator: "|") + #")""#
        let regex = try NSRegularExpression(pattern: pattern)
        let folders = Set(types)
        // `Family.swift` is where the cases are declared; `Errors.swift` holds one sample of every error, with
        // plausible model names in them.
        let exempt: Set<String> = ["Family.swift", "Errors.swift"]
        let files = swiftFiles(under: "SiliconedApp") + swiftFiles(under: "Siliconed").filter { file in
            !exempt.contains(file.lastPathComponent) && folders.isDisjoint(with: file.pathComponents)
        }
        XCTAssertGreaterThan(files.count, 100)
        var hits: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for (i, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let code = String(line.range(of: "//").map { line[..<$0.lowerBound] } ?? line[...])
                if let m = regex.firstMatch(in: code, range: NSRange(code.startIndex..., in: code)) {
                    hits.append("\(file.lastPathComponent):\(i + 1): \((code as NSString).substring(with: m.range))")
                }
            }
        }
        XCTAssertTrue(hits.isEmpty, hits.joined(separator: "\n"))
    }

    /// **Nor its words**: no string of the interface writes a family's name, its model's or its license's, and none of the
    /// library outside the families' folders writes its license's — what is said of a family, it declares
    /// (`FamilyPresentation`, `FamilyTraits`), and the interface reads it there. Compared without case: « Qwen-Image-2.1 » in a
    /// sentence is the identifier as much as a name.
    func testNoStringOutsideAFamilysFolderSaysItsNameOrLicense() throws {
        let licenses = Set(Family.allCases.map { $0.traits.card.license.text.lowercased() })
        let names = Set(Family.allCases.flatMap { f in [f.rawValue, f.traits.name, f.traits.card.name].map { $0.lowercased() } })
        let literal = try NSRegularExpression(pattern: #""(?:[^"\\]|\\.)*""#)
        let folders = Set(Family.allCases.map { "\($0)".prefix(1).uppercased() + "\($0)".dropFirst() })
        let exempt: Set<String> = ["Family.swift", "Errors.swift"]
        let library = swiftFiles(under: "Siliconed").filter { file in
            !exempt.contains(file.lastPathComponent) && folders.isDisjoint(with: file.pathComponents)
        }
        var hits: [String] = []
        for (files, words) in [(swiftFiles(under: "SiliconedApp"), licenses.union(names)), (library, licenses)] {
            for file in files {
                let text = try String(contentsOf: file, encoding: .utf8)
                for (i, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                    let code = String(line.range(of: "//").map { line[..<$0.lowerBound] } ?? line[...])
                    for m in literal.matches(in: code, range: NSRange(code.startIndex..., in: code)) {
                        let string = (code as NSString).substring(with: m.range).lowercased()
                        if let word = words.first(where: { string.contains($0) }) {
                            hits.append("\(file.lastPathComponent):\(i + 1): \(word) in \(string)")
                        }
                    }
                }
            }
        }
        XCTAssertTrue(hits.isEmpty, hits.joined(separator: "\n"))
    }

    /// Each family has its folder, and says of itself what the interface shows: a sentence, a symbol, and nothing
    /// about a version it does not publish.
    func testEachFamilyDeclaresWhatTheInterfaceSays() {
        for family in Family.allCases {
            let folder = "\(family)".prefix(1).uppercased() + "\(family)".dropFirst()
            XCTAssertFalse(swiftFiles(under: "Siliconed/\(folder)").isEmpty, folder)
            let said = family.traits.presentation
            XCTAssertFalse(String(localized: said.blurb).isEmpty, "\(family)")
            XCTAssertFalse(said.symbol.isEmpty, "\(family)")
            for variant in Set(said.summaries.keys).union(said.details.keys).union(said.sources.keys) {
                XCTAssertTrue(family.variants.contains(variant), "\(family) says something of \(variant), which it does not publish")
            }
        }
        XCTAssertEqual(Family.default, Family.allCases.first)
        XCTAssertEqual(Infotext.civitaiModelVersion(model: Family.default.rawValue), Family.default.traits.civitaiModelVersion)
        XCTAssertNil(Infotext.civitaiModelVersion(model: "no-such-model"))
    }
}
