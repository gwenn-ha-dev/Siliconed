import XCTest
@testable import Siliconed

/// **Named subjects** (`NamedSubjects`): `@alice` in a prompt → the model's prompt and its references.
/// Pure: the subjects live in the app's memory, nothing on disk.
final class NamedSubjectsTests: XCTestCase {
    private typealias N = NamedSubjects

    private func subject(_ name: String, _ kind: N.Kind, pictures: Int = 1) -> N.Subject {
        N.Subject(name: name, kind: kind, pictures: (1...max(pictures, 1)).prefix(pictures).map { N.Picture(file: "\($0).jpg") })
    }

    private lazy var library = [subject("alice", .person), subject("red-coat", .garment),
                                subject("the-beach", .place), subject("Bob", .person, pictures: 2),
                                subject("lamp", .object), subject("ink", .style)]

    private func names(_ p: String) -> [String] { N.mentions(in: p).map(\.name) }

    // ── mentions ──

    func testMentionsAreReadInOrderWithTheirRanges() {
        let p = "@alice in @red-coat on @the-beach."
        let m = N.mentions(in: p)
        XCTAssertEqual(m.map(\.name), ["alice", "red-coat", "the-beach"])
        XCTAssertEqual(m.map { String(p[$0.range]) }, ["@alice", "@red-coat", "@the-beach"])
    }

    func testWhatIsNotAMention() {
        XCTAssertEqual(names("mail me@home.org"), [])            // after a letter
        XCTAssertEqual(names("a lone @ sign, @, @-"), [])         // nothing a name has
        XCTAssertEqual(names("@@alice"), [])                       // an @ after an @
        XCTAssertEqual(names("(@alice)"), ["alice"])
        XCTAssertEqual(names("@alice_ and @bob-, @élodie"), ["alice", "bob", "élodie"])  // trailing joiners left out
        XCTAssertEqual(names("@a-b_c2"), ["a-b_c2"])
        XCTAssertEqual(names("line\n@alice"), ["alice"])
    }

    // ── names ──

    func testNamesAreCheckedForMentions() {
        XCTAssertNil(N.check(name: "red-coat"))
        XCTAssertNil(N.check(name: "élodie_2"))
        XCTAssertEqual(N.check(name: ""), .empty)
        XCTAssertEqual(N.check(name: String(repeating: "a", count: 33)), .tooLong)
        XCTAssertEqual(N.check(name: "red coat"), .characters)
        XCTAssertEqual(N.check(name: "-alice"), .characters)
        XCTAssertEqual(N.check(name: "alice-"), .characters)
        XCTAssertEqual(N.check(name: "Alice", taken: ["alice"]), .taken)
        // A valid name is cited whole.
        for n in ["red-coat", "a_b-c", "x"] { XCTAssertEqual(names("@\(n)!"), [n]) }
    }

    func testSuggestedNamesComeFromAFileName() {
        XCTAssertEqual(N.suggestedName("IMG 2041 (copy).jpeg"), "IMG-2041-copy-jpeg")
        XCTAssertEqual(N.suggestedName("  red coat  "), "red-coat")
        XCTAssertEqual(N.suggestedName("..."), "")
        XCTAssertNil(N.check(name: N.suggestedName("Portrait de Marie, été 2026.heic")))
    }

    // ── the rewrite ──

    func testAPromptWithoutMentionIsUnchanged() throws {
        for p in ["woman posing in a library", "", "me@home", "  spaces  "] {
            let r = try N.rewrite(p, subjects: library, capacity: 3)
            XCTAssertEqual(r.prompt, p)
            XCTAssertEqual(r.placed, [])
        }
        // Even for a model that reads no image.
        XCTAssertEqual(try N.rewrite("woman posing in a library", subjects: [], capacity: 0).prompt, "woman posing in a library")
    }

    func testTheNameNeverReachesTheModelAndTheLegendNumbersByFirstAppearance() throws {
        let r = try N.rewrite("@alice in @red-coat on @the-beach", subjects: library, capacity: 3)
        XCTAssertEqual(r.placed.map(\.number), [1, 2, 3])
        XCTAssertEqual(r.placed.map(\.subject.name), ["alice", "red-coat", "the-beach"])
        XCTAssertEqual(r.cited.map(\.name), ["alice", "red-coat", "the-beach"])
        let (i1, i2, i3) = (N.Phrasing.reference(1), N.Phrasing.reference(2), N.Phrasing.reference(3))
        let lines = r.prompt.components(separatedBy: "\n")
        XCTAssertEqual(lines[0], "the person in \(i1) in the garment in \(i2) on the place in \(i3)")
        XCTAssertEqual(lines[1], "")
        XCTAssertTrue(lines[2].hasPrefix("Take only their identity from \(i1)"), lines[2])
        XCTAssertTrue(lines[3].hasPrefix("Take only the garment from \(i2)"), lines[3])
        XCTAssertTrue(lines[4].hasPrefix("Use \(i3) as the setting"), lines[4])
        XCTAssertEqual(lines.count, 5)
        XCTAssertFalse(r.prompt.contains("@"))
        for name in ["alice", "red coat", "red-coat", "beach"] { XCTAssertFalse(r.prompt.lowercased().contains(name), name) }
    }

    /// The user's own case: an article before a mention is not doubled.
    func testAnArticleTheUserWroteIsNotDoubled() throws {
        let lib = [subject("alice", .person), subject("chambre", .place)]
        let r = try N.rewrite("redraw @alice in the @chambre, she is alone", subjects: lib, capacity: 3)
        let (i1, i2) = (N.Phrasing.reference(1), N.Phrasing.reference(2))
        XCTAssertTrue(r.prompt.hasPrefix("redraw the person in \(i1) in the place in \(i2), she is alone\n\n"), r.prompt)
        XCTAssertTrue(try N.rewrite("a photo of a @lamp", subjects: library, capacity: 3).prompt
            .hasPrefix("a photo of a object in \(i1)"))
    }

    func testASubjectCitedTwiceIsPlacedOnce() throws {
        let r = try N.rewrite("@lamp on a desk, light from @LAMP", subjects: library, capacity: 3)
        XCTAssertEqual(r.placed.count, 1)
        let i1 = N.Phrasing.reference(1)
        XCTAssertTrue(r.prompt.hasPrefix("the object in \(i1) on a desk, light from the object in \(i1)\n\n"), r.prompt)
    }

    func testAccentsCaseAndAddressesInARewrite() throws {
        let lib = [subject("Élodie", .person), subject("café-noir", .place)]
        // Any case, precomposed or not: the mention is found; an address is not a mention.
        let r = try N.rewrite("@élodie at @CAFÉ-NOIR, mail elodie@cafe.fr", subjects: lib, capacity: 3)
        XCTAssertEqual(r.placed.map(\.subject.name), ["Élodie", "café-noir"])
        let (i1, i2) = (N.Phrasing.reference(1), N.Phrasing.reference(2))
        XCTAssertTrue(r.prompt.hasPrefix("the person in \(i1) at the place in \(i2), mail elodie@cafe.fr\n\n"), r.prompt)
        let decomposed = "@E\u{301}lodie"
        XCTAssertEqual(try N.rewrite(decomposed, subjects: lib, capacity: 3).placed.map(\.subject.name), ["Élodie"])
    }

    func testTwoPicturesOfOneSubjectTakeTwoPlaces() throws {
        let r = try N.rewrite("@Bob with @lamp", subjects: library, capacity: 3)
        XCTAssertEqual(r.placed.map(\.number), [1, 2, 3])
        XCTAssertEqual(r.placed.map(\.picture.file), ["1.jpg", "2.jpg", "1.jpg"])
        let both = N.Phrasing.references([1, 2])
        XCTAssertTrue(r.prompt.hasPrefix("the same person in \(both) with the object in \(N.Phrasing.reference(3))"), r.prompt)
        XCTAssertTrue(r.prompt.contains("Take only their identity from \(both)"), r.prompt)
    }

    func testAnEditsImagesComeFirst() throws {
        let r = try N.rewrite("put @alice in image 1", subjects: library, capacity: 3, first: 2)
        XCTAssertEqual(r.placed.map(\.number), [2])
        XCTAssertTrue(r.prompt.hasPrefix("put the person in \(N.Phrasing.reference(2)) in image 1"), r.prompt)
        XCTAssertThrowsError(try N.rewrite("@alice @lamp", subjects: library, capacity: 3, first: 3)) {
            XCTAssertEqual($0 as? N.Failure, .tooManyImages(count: 4, max: 3))
        }
    }

    func testTheRefusals() {
        func refusal(_ p: String, capacity: Int = 3) -> N.Failure? {
            do { _ = try N.rewrite(p, subjects: library, capacity: capacity); return nil } catch { return error }
        }
        XCTAssertEqual(refusal("@alice and @carol"), .unknownSubject(name: "carol"))
        // Three subjects, four pictures; the cap is ours even when the model reads four.
        XCTAssertEqual(refusal("@Bob @alice @lamp", capacity: 4), .tooManyImages(count: 4, max: 3))
        XCTAssertEqual(refusal("@alice @lamp", capacity: 1), .tooManyImages(count: 2, max: 1))
        XCTAssertEqual(refusal("@alice", capacity: 0), .modelReadsNoImages)
        // An unknown name is said before the model's limits: it is the user's typo to fix first.
        XCTAssertEqual(refusal("@nobody", capacity: 0), .unknownSubject(name: "nobody"))
        let bare = N.Subject(name: "ghost", kind: .object, pictures: [])
        XCTAssertThrowsError(try N.rewrite("@ghost", subjects: [bare], capacity: 3)) {
            XCTAssertEqual($0 as? N.Failure, .noPicture(name: "ghost"))
        }
    }

    func testEveryKindHasItsMentionAndItsInstruction() throws {
        let i1 = N.Phrasing.reference(1)
        for kind in N.Kind.allCases {
            let m = N.Phrasing.mention(kind, [1], after: "")
            XCTAssertTrue(m.hasSuffix(i1), m)
            let t = N.Phrasing.retention(kind, from: i1)
            XCTAssertTrue(t.contains(i1) && t.hasSuffix("."), t)
        }
        XCTAssertTrue(try N.rewrite("in the manner of @ink", subjects: library, capacity: 3).prompt
            .hasPrefix("in the manner of the style of \(i1)\n\n"))
        XCTAssertTrue(try N.rewrite("in the style of @ink", subjects: library, capacity: 3).prompt
            .hasPrefix("in the style of \(i1)\n\n"))
    }

    // ── the image's metadata ──

    /// The PNG keeps the prompt typed — what a drop or « Reuse » restores —, the model's beside it;
    /// nobody cited, the engine's metadata untouched.
    func testTheMetadataKeepsThePromptTyped() throws {
        let engine = ["prompt": "alice in a library\n\nalice is the person in <image1>.", "seed": "42",
                      "model": "qwen-image-2.1", "steps": "8", "format": "1024x1024",
                      "Software": "Siliconed"]
        XCTAssertEqual(N.metadata(engine, typed: "@alice in a library", modelPrompt: nil), engine)
        let m = N.metadata(engine, typed: "@alice in a library", modelPrompt: engine["prompt"])
        XCTAssertEqual(m["prompt"], "@alice in a library")
        XCTAssertEqual(m[N.modelPromptKey], engine["prompt"])
        XCTAssertEqual(Set(m.keys), Set(engine.keys).union([N.modelPromptKey]))
        let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: m))
        XCTAssertEqual(recipe.prompt, "@alice in a library")
        XCTAssertEqual(recipe.unreadable, [])
        // Through the PNG's text chunks and back.
        let image = ImageRGB(pixels: [Float](repeating: 0, count: 3 * 2 * 2), height: 2, width: 2)
        let png = try image.png(metadata: m)
        XCTAssertEqual(PNG.text(png)["prompt"], "@alice in a library")
        XCTAssertEqual(PNG.text(png)[N.modelPromptKey], engine["prompt"])
    }

    // ── frames ──

    func testFramesClampAndRoundByTheirEnds() {
        let c = N.Frame(x: -0.2, y: 0.5, width: 2, height: 0.8).clamped()
        XCTAssertEqual([c.x, c.width, c.height], [0, 1, 0.8])
        XCTAssertEqual(c.y, 0.2, accuracy: 1e-12)
        XCTAssertEqual(N.Frame(x: .nan, y: 0, width: 0, height: 1).clamped().width, 0.02)
        let p = N.Frame(x: 0.25, y: 0.1, width: 0.5, height: 0.3).pixels(width: 1001, height: 333)
        XCTAssertEqual(p.x, 250); XCTAssertEqual(p.width, 751 - 250)   // ends 250.25 → 250, 750.75 → 751
        XCTAssertEqual(p.y, 33); XCTAssertEqual(p.height, 133 - 33)
        let whole = N.Frame.whole.pixels(width: 640, height: 480)
        XCTAssertEqual([whole.x, whole.y, whole.width, whole.height], [0, 0, 640, 480])
        let tiny = N.Frame(x: 0.999, y: 0.999, width: 0.001, height: 0.001).pixels(width: 10, height: 10)
        XCTAssertGreaterThanOrEqual(tiny.width, 1)
        XCTAssertLessThanOrEqual(tiny.x + tiny.width, 10)
    }
}
