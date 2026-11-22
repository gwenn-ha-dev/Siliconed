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

    /// **What lies outside the families' folders**: the interface, the library but the families' folders and the file
    /// that declares their cases, and the command lines' modules (`silicontrol`, its help, `measure`) — the help's pages
    /// alone exempt, a manual that names the models (what it says of them is held against their traits,
    /// `HelpTraitsTests`).
    private func outside() -> [URL] {
        let folders = Set(Family.allCases.map { "\($0)".prefix(1).uppercased() + "\($0)".dropFirst() })
        let library = swiftFiles(under: "Siliconed").filter { file in
            file.lastPathComponent != "Family.swift" && folders.isDisjoint(with: file.pathComponents)
        }
        let pages: Set = ["HelpContract.swift", "HelpRender.swift", "HelpLibrary.swift", "HelpWriter.swift", "HelpSettings.swift"]
        let tools = ["Silicontrol", "SilicontrolHelp", "Measurement"].flatMap(swiftFiles)
            .filter { !pages.contains($0.lastPathComponent) }
        return library + tools
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
        let files = swiftFiles(under: "SiliconedApp") + outside()
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

    /// **Nor its words**: no string of the interface or of the command lines' modules writes a family's name, its model's
    /// or its license's, and none of the library outside the families' folders writes its license's — what is said of a family, it declares
    /// (`FamilyPresentation`, `FamilyTraits`), and the interface reads it there. Compared without case: « Qwen-Image-2.1 » in a
    /// sentence is the identifier as much as a name.
    func testNoStringOutsideAFamilysFolderSaysItsNameOrLicense() throws {
        let licenses = Set(Family.allCases.map { $0.traits.card.license.text.lowercased() })
        let names = Set(Family.allCases.flatMap { f in [f.rawValue, f.traits.name, f.traits.card.name].map { $0.lowercased() } })
        let literal = try NSRegularExpression(pattern: #""(?:[^"\\]|\\.)*""#)
        let all = outside()
        let inLibrary = { (file: URL) in file.path.hasPrefix(Self.sources.appendingPathComponent("Siliconed").path + "/") }
        let tools = all.filter { !inLibrary($0) }, library = all.filter(inLibrary)
        var hits: [String] = []
        for (files, words) in [(swiftFiles(under: "SiliconedApp") + tools, licenses.union(names)), (library, licenses)] {
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
