import Foundation
import XCTest
@testable import Siliconed
@testable import SilicontrolHelp

/// **What `silicontrol`'s help says of a family is what the family declares.** The help is Foundation only and names the
/// models in its prose (the one exemption of `FamilyFrontierTests`): each sentence that states a trait — which models start
/// from an image, which edits and how many references it reads, who writes prompts, a side's multiple, a longest side, a
/// starting image's strength — is held here against the family's traits and its denoiser, in both languages. A family
/// whose traits change, or a new one, fails here until the help says it.
final class HelpTraitsTests: XCTestCase {
    /// Each family's denoiser, for what its chain reads (a starting image, references).
    private let denoisers: [Family: any DenoisingModule] = [
        .zImage: ZImageDenoisingModule(map: ""), .anima: AnimaDenoisingModule(map: ""), .krea2: Krea2DenoisingModule(map: ""),
        .mossImage: MossImageDenoisingModule(map: "", turbo: ""), .qwenImage21: QwenImage21DenoisingModule(map: ""),
    ]

    /// The ways the help names a family: its identifier, its name, its model's name with and without « Turbo ».
    private func names(_ f: Family) -> [String] {
        let card = f.traits.card.name
        return [f.rawValue, f.traits.name, card, card.replacingOccurrences(of: " Turbo", with: "")]
    }

    private func family(named word: String) -> Family? { Family.allCases.first { names($0).contains(word) } }

    /// The help's text, each run of spaces and line breaks one space: a sentence reads across the page's wrapping.
    private func text(_ pages: RemoteControlHelp.Pages) -> String {
        let all = ([pages.overview] + RemoteControlHelp.order.compactMap { pages.pages[$0] }).joined(separator: "\n\n")
        return all.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
    }

    private func matches(_ pattern: String, in text: String) throws -> [[String]] {
        let regex = try NSRegularExpression(pattern: pattern)
        let ns = text as NSString
        return regex.matches(in: text, range: NSRange(location: 0, length: ns.length)).map { m in
            (0..<m.numberOfRanges).map { m.range(at: $0).location == NSNotFound ? "" : ns.substring(with: m.range(at: $0)) }
        }
    }

    /// One name, the longest first so that « Anima Turbo » is not read « Anima ».
    private var oneName: String {
        let all = Family.allCases.flatMap(names).sorted { $0.count > $1.count }.map(NSRegularExpression.escapedPattern)
        return #"(?<![\w-])(?:"# + all.joined(separator: "|") + #")(?![\w-])"#
    }

    private var nameList: String { "\(oneName)(?:(?:, | and | et )\(oneName))*" }

    private func families(in list: String) throws -> [Family] {
        try matches(oneName, in: list).compactMap { family(named: $0[0]) }
    }

    func testTheHelpSaysWhatTheFamiliesDeclare() throws {
        XCTAssertEqual(Set(denoisers.keys), Set(Family.allCases), "a family's denoiser is missing here")
        let starting = Set(Family.allCases.filter { denoisers[$0]!.acceptsStartImage })
        let editing = Set(Family.allCases.filter { denoisers[$0]!.maxReferences > 0 })
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            var text = text(pages)
            let language = pages.overview.contains("THE COMMANDS") ? "en" : "fr"

            // A starting image's strength: « Krea 2 0.65, Z-Image and Anima 0.75, MOSS-Image 0.80 ».
            let strengths = try matches("(\(nameList)) (0\\.\\d+)", in: text)
            XCTAssertFalse(strengths.isEmpty, language)
            for m in strengths {
                for f in try families(in: m[1]) {
                    XCTAssertEqual(f.traits.startStrength, Double(m[2]), "\(language): \(m[0])")
                }
                text = text.replacingOccurrences(of: m[0], with: "")
            }

            // Every other list of models is all of them, or those that start from an image.
            var startLists = 0
            for m in try matches("\(oneName)(?:(?:, | and | et )\(oneName))+", in: text) {
                let named = try families(in: m[0])
                XCTAssertEqual(named.count, Set(named).count, "\(language): \(m[0])")
                if Set(named) == starting { startLists += 1; continue }
                XCTAssertEqual(Set(named), Set(Family.allCases), "\(language): « \(m[0]) » is neither every model nor those that start from an image")
            }
            XCTAssertGreaterThanOrEqual(startLists, 4, language)

            // Editing: the references a model reads, the examples of --ref, the lab's reference role.
            let maxima = try matches(#""max_references" \((\#(oneName)) ?: (\d+)\)"#, in: text)
            XCTAssertFalse(maxima.isEmpty, language)
            for m in maxima {
                let f = try XCTUnwrap(family(named: m[1]))
                XCTAssertEqual(denoisers[f]!.maxReferences, Int(m[2]), "\(language): \(m[0])")
            }
            let edits = try matches(#"--model (\#(oneName))(?: [^ ]+){0,6}? --ref "#, in: text) + (try matches(#"reference \((\#(oneName))\)"#, in: text))
            XCTAssertFalse(edits.isEmpty, language)
            for m in edits { XCTAssertTrue(editing.contains(try XCTUnwrap(family(named: m[1]))), "\(language): \(m[0])") }

            // The writer: the model to install for it, and whose encoder it is.
            let writers = try matches(#"(?:carries a writer|n'a de rédacteur) ?(?::|→) install (\#(oneName))"#, in: text)
                + (try matches(#"(?:That writer is |Ce rédacteur est l'encodeur de texte de )(\#(oneName))"#, in: text))
            XCTAssertGreaterThanOrEqual(writers.count, 3, language)
            for m in writers { XCTAssertNotNil(try XCTUnwrap(family(named: m[1])).traits.writer, "\(language): \(m[0])") }

            // A side's multiple and a longest side: « (32 for qwen-image-2.1) », « (≤ 2048 pour anima) », and both ways —
            // every family with a rule of its own has its sentence.
            var multiples: [Family: Int] = [:], longest: [Family: Int] = [:]
            for m in try matches(#"\((≤ )?(\d+)(?: for | pour | ?: )(\#(oneName))\)"#, in: text) {
                let f = try XCTUnwrap(family(named: m[3])), n = Int(m[2])
                if m[1].isEmpty { multiples[f] = n; XCTAssertEqual(f.traits.sideMultiple, n, "\(language): \(m[0])") }
                else { longest[f] = n; XCTAssertEqual(f.traits.maximumSide, n, "\(language): \(m[0])") }
            }
            for m in try matches(#"(?:on|sur) (\#(oneName))\W.{0,40}?\((\d+) px"#, in: text) {
                XCTAssertEqual(try XCTUnwrap(family(named: m[1])).traits.maximumSide, Int(m[2]), "\(language): \(m[0])")
            }
            for f in Family.allCases {
                XCTAssertEqual(multiples[f], f.traits.sideMultiple, "\(language): \(f)'s multiple")
                let shorter = f.traits.maximumSide.flatMap { $0 < Format.maximumSide ? $0 : nil }
                XCTAssertEqual(longest[f], shorter, "\(language): \(f)'s longest side")
            }
        }
    }

    /// `set`'s usages are `SetOrder`'s: each in the overview and on its page.
    func testSetsUsagesAreTheOrders() {
        let pages = RemoteControlHelp.english
        for (_, usage) in SetOrder.usages {
            XCTAssertTrue(pages.overview.contains(usage), usage)
            XCTAssertTrue(pages.pages["set"]?.contains("silicontrol " + usage) == true, usage)
        }
    }
}
