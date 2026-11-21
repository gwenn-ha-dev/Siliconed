import XCTest
@testable import Siliconed
@testable import SilicontrolHelp

/// **The starting image of an img2img, as the app holds it** (`StartImage`, `LabInputs` with `startsFromImage`, `start-image`): the
/// role chosen by a gesture and never by default, the start alone carried with the text as typed, the strength in hundredths, the
/// recommended format nearest the image's proportions, and the remote control's arguments and pages.
final class StartImageTests: XCTestCase {
    private typealias L = LabInputs

    private func slots(_ roles: L.Role...) -> [L.Slot<Int>] { roles.enumerated().map { L.Slot(id: $0.offset, role: $0.element) } }

    // ── the role, in the Prompt Lab ──

    /// The defaults settled on the sheets, and what they redraw at eight steps: Krea 2 six steps of eight (as 75 % does),
    /// Z-Image and Anima six, MOSS-Image seven — none so high that the image is ignored (the full schedule).
    func testEachFamilyStartsFromItsOwnStrength() {
        XCTAssertEqual(StartImage.defaultStrength(for: .krea2), 0.65)
        XCTAssertEqual(StartImage.defaultStrength(for: .zImage), 0.75)
        XCTAssertEqual(StartImage.defaultStrength(for: .anima), 0.75)
        XCTAssertEqual(StartImage.defaultStrength(for: .mossImage), 0.80)
        XCTAssertEqual(StartImage.defaultStrength(for: nil), 0.65)
        XCTAssertEqual(Strength.startStep(steps: 8, strength: StartImage.defaultStrength(for: .krea2)), 2)
        XCTAssertEqual(Strength.startStep(steps: 8, strength: StartImage.defaultStrength(for: .zImage)), 2)
        XCTAssertEqual(Strength.startStep(steps: 8, strength: StartImage.defaultStrength(for: .mossImage)), 1)
        for family in Family.allCases { XCTAssertEqual(StartImage.snapped(StartImage.defaultStrength(for: family)), StartImage.defaultStrength(for: family)) }
    }

    func testAnImg2ImgModelOffersTheStartAndNoReference() {
        XCTAssertEqual(L.roles(capacity: 0, startsFromImage: true), [.start, .neutral])
        XCTAssertEqual(L.roles(capacity: 0, startsFromImage: false), [.neutral])
        // Qwen-Image-2.1 reads its images: the edited image and the references, never a start.
        XCTAssertEqual(L.roles(capacity: 3, startsFromImage: false), [.edited, .reference, .neutral])
    }

    /// A picture dropped to be described never makes the next render an img2img: its start is a gesture.
    func testANewcomerIsNeverAnImg2ImgStart() {
        XCTAssertEqual(L.defaultRole(adding: [] as [L.Slot<Int>], capacity: 0), .neutral)
        XCTAssertEqual(L.defaultRole(adding: slots(.neutral, .neutral), capacity: 0), .neutral)
    }

    /// One start: choosing another sends the previous one back to neutral — never to a reference.
    func testChoosingAnotherStartSendsTheFirstBackToNeutral() {
        XCTAssertEqual(L.assigning(.start, to: 1, in: slots(.start, .neutral)).map(\.role), [.neutral, .start])
    }

    func testChangingTheModelKeepsTheStartWhereItIsOffered() {
        // Z-Image → Anima: the start stays.
        XCTAssertEqual(L.conforming(slots(.start, .neutral), capacity: 0, startsFromImage: true).map(\.role), [.start, .neutral])
        // A second start (never made by the window) has nowhere else to go.
        XCTAssertEqual(L.conforming(slots(.start, .start), capacity: 0, startsFromImage: true).map(\.role), [.start, .neutral])
        // A model that neither reads nor starts from an image: the start keeps its role, without effect.
        XCTAssertEqual(L.conforming(slots(.start, .neutral), capacity: 0, startsFromImage: false).map(\.role), [.start, .neutral])
        XCTAssertFalse(L.isEffective(.start, capacity: 0, startsFromImage: false))
    }

    /// The start alone goes, the text as typed — `@` is the model's own there (Anima's artists).
    func testAnImg2ImgCarriesItsStartAndTheTextAsTyped() throws {
        let text = "@greg_rutkowski, a lighthouse at dusk"
        let c = try L.carry(text: text, slots: slots(.neutral, .start), subjects: [], capacity: 0, startsFromImage: true).get()
        XCTAssertEqual(c.prompt, text)
        XCTAssertEqual(c.images, [.plain(1, .start)])
        XCTAssertTrue(c.cited.isEmpty)
        // Without a start, the text alone; without img2img, no image at all.
        XCTAssertEqual(try L.carry(text: text, slots: slots(.neutral), subjects: [], capacity: 0, startsFromImage: true).get().images, [])
        XCTAssertEqual(try L.carry(text: text, slots: slots(.start), subjects: [], capacity: 0).get().images, [])
    }

    // ── the strength ──

    func testTheStrengthIsKeptInHundredths() {
        XCTAssertEqual(StartImage.snapped(0.654), 0.65)
        XCTAssertEqual(StartImage.snapped(0.656), 0.66)
        XCTAssertEqual(StartImage.snapped(-0.2), 0)
        XCTAssertEqual(StartImage.snapped(1.7), 1)
        XCTAssertEqual(StartImage.snapped(.nan), 0)
        // What the PNG writes (`%.2f`) reads back to the same number.
        let s = StartImage.snapped(0.65)
        XCTAssertEqual(Double(String(format: "%.2f", s)), s)
    }

    func testAStrengthReadsAsTypedOrNot() {
        XCTAssertEqual(StartImage.strength("0.65"), 0.65)
        XCTAssertEqual(StartImage.strength("0,4"), 0.4)
        XCTAssertEqual(StartImage.strength("65%"), 0.65)
        XCTAssertEqual(StartImage.strength(" 100 % "), 1)
        XCTAssertEqual(StartImage.strength("1"), 1)
        // 0 leaves no evaluation, beyond 1 is not a strength, and words are not numbers.
        for text in ["0", "0%", "1.2", "120%", "-0.5", "nan", "inf", "strong", ""] { XCTAssertNil(StartImage.strength(text), text) }
    }

    /// The engine's quantization, which the rack's tooltip counts in steps: at 8 steps, 65 % starts at step 2 (6 redrawn).
    func testTheStrengthLeavesStepsAsDiffusersCountsThem() {
        XCTAssertEqual(Strength.startStep(steps: 8, strength: 0.65), 2)
        XCTAssertEqual(Strength.startStep(steps: 8, strength: 1), 0)
        XCTAssertEqual(Strength.startStep(steps: 8, strength: 0.01), 7)
        XCTAssertEqual(Strength.startStep(steps: 8, strength: 0), 8)
    }

    // ── the format ──

    func testTheFormatFollowsTheImageAtTheRacksScale() {
        let formats = ModelCard.recommendedFormats
        let square = RecommendedFormat(1024, 1024)
        // A square photo stays at the rack's scale: 1024² from 1024², 512² from 512².
        XCTAssertEqual(StartImage.recommendedFormat(width: 3000, height: 3000, among: formats, current: square), square)
        XCTAssertEqual(StartImage.recommendedFormat(width: 3000, height: 3000, among: formats, current: RecommendedFormat(512, 512)),
                       RecommendedFormat(512, 512))
        // A 2:3 portrait takes the nearest proportions: 832×1216 (0.684), not 896×1152 (0.778).
        XCTAssertEqual(StartImage.recommendedFormat(width: 2000, height: 3000, among: formats, current: square), RecommendedFormat(832, 1216))
        // A 4:3 landscape: 1152×896 (1.286) is nearer than 1216×832 (1.462).
        XCTAssertEqual(StartImage.recommendedFormat(width: 4032, height: 3024, among: formats, current: square), RecommendedFormat(1152, 896))
        // A 16:9 one: 1344×768.
        XCTAssertEqual(StartImage.recommendedFormat(width: 1920, height: 1080, among: formats, current: square), RecommendedFormat(1344, 768))
        XCTAssertNil(StartImage.recommendedFormat(width: 0, height: 10, among: formats, current: square))
        XCTAssertNil(StartImage.recommendedFormat(width: 10, height: 10, among: [], current: square))
    }

    // ── the remote control ──

    private func parse(_ arguments: [String]) -> GestureOrder? {
        if case .success(let order) = GestureOrder.parse("start-image", arguments) { return order }
        return nil
    }

    func testStartImageArguments() {
        XCTAssertTrue(GestureOrder.verbs.contains("start-image"))
        XCTAssertEqual(parse([]), .startImage(.status))
        XCTAssertEqual(parse(["photo.png"]), .startImage(.carry("photo.png", strength: nil)))
        XCTAssertEqual(parse(["i7", "--strength", "0.4"]), .startImage(.carry("i7", strength: 0.4)))
        XCTAssertEqual(parse(["--strength", "80%"]), .startImage(.strength(0.8)))
        XCTAssertEqual(parse(["--remove"]), .startImage(.remove))
        XCTAssertEqual(GestureOrder.parse("start-image", ["--strength", "0"]), .failure(.badValue("--strength", "0")))
        XCTAssertEqual(GestureOrder.parse("start-image", ["--strength"]), .failure(.flagWithoutValue("--strength")))
        for wrong in [["a.png", "b.png"], ["a.png", "--remove"], ["--remove", "--strength", "0.5"]] {
            if case .failure(.usage) = GestureOrder.parse("start-image", wrong) {} else { XCTFail("\(wrong)") }
        }
        XCTAssertEqual(GestureOrder.parse("start-image", ["--yes"]), .failure(.unknownFlag("--yes")))
    }

    /// Documented in both languages: the command in the overview, its page, `add --image`, and the codes it answers.
    func testTheStartImageIsDocumented() {
        for pages in [RemoteControlHelp.english, RemoteControlHelp.french] {
            XCTAssertTrue(pages.overview.contains("start-image [<image|i7>] [--strength S]"))
            XCTAssertNotNil(RemoteControlHelp.answer(["start-image", "--help"], pages)?["help"] as? String)
            let page = pages.pages["start-image"] ?? ""
            for word in ["--strength", "--remove", "\"start_image\"", "no_start_image", "img2img", "0.65"] {
                XCTAssertTrue(page.contains(word), word)
            }
            let add = pages.pages["add"] ?? ""
            XCTAssertTrue(add.contains("--image IMAGE|i7"))
            XCTAssertTrue(add.contains("--strength"))
            XCTAssertTrue(pages.pages["models"]?.contains("\"start_image\"") == true)
            XCTAssertTrue(pages.pages["errors"]?.contains("no_start_image") == true)
        }
    }
}
