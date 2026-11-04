import XCTest
@testable import Siliconed

/// **The render's inputs** (`LabInputs`): roles, numbering, refusals, the mentions kept through a rewrite — and the
/// proof that what the render receives is what the rack's former Subjects mode made at the click.
final class LabInputsTests: XCTestCase {
    private typealias L = LabInputs
    private typealias N = NamedSubjects

    private func subject(_ name: String, _ kind: N.Kind, pictures: Int = 1) -> N.Subject {
        N.Subject(name: name, kind: kind, pictures: (1...pictures).map { N.Picture(file: "\($0).jpg") })
    }

    private lazy var library = [subject("alice", .person), subject("red-coat", .garment), subject("the-beach", .place),
                                subject("Bob", .person, pictures: 2), subject("lamp", .object)]

    private func slots(_ roles: L.Role...) -> [L.Slot<Int>] { roles.enumerated().map { L.Slot(id: $0.offset, role: $0.element) } }

    private func carried(_ text: String, _ slots: [L.Slot<Int>] = [], capacity: Int = 3) throws -> L.Carry<Int> {
        try L.carry(text: text, slots: slots, subjects: library, capacity: capacity).get()
    }

    private func refusal(_ text: String, _ slots: [L.Slot<Int>] = [], capacity: Int = 3) -> L.Refusal? {
        if case .failure(let r) = L.carry(text: text, slots: slots, subjects: library, capacity: capacity) { return r }
        return nil
    }

    // ── the roles ──

    func testADroppedImageTakesTheNextPlaceUnderAModelThatReadsImages() {
        var zone: [L.Slot<Int>] = []
        var roles: [L.Role] = []
        for id in 0..<5 {
            let role = L.defaultRole(adding: zone, capacity: 3)
            zone.append(L.Slot(id: id, role: role))
            roles.append(role)
        }
        XCTAssertEqual(roles, [.edited, .reference, .reference, .neutral, .neutral])
        // The edited place is filled first when it is free.
        XCTAssertEqual(L.defaultRole(adding: slots(.reference, .neutral), capacity: 3), .edited)
        // A model that reads no image, img2img included: a newcomer is neutral, its start is a gesture.
        XCTAssertEqual(L.defaultRole(adding: [] as [L.Slot<Int>], capacity: 0), .neutral)
        XCTAssertEqual(L.roles(capacity: 0), [.neutral])
        XCTAssertEqual(L.roles(capacity: 0, startsFromImage: true), [.start, .neutral])
        XCTAssertEqual(L.roles(capacity: 3), [.edited, .reference, .neutral])
        XCTAssertEqual(L.roles(capacity: 3, startsFromImage: true), [.edited, .reference, .neutral], "an edit model offers no start")
    }

    func testTheEditedImageAndTheStartAreUnique() {
        let zone = L.assigning(.edited, to: 2, in: slots(.edited, .reference, .reference))
        XCTAssertEqual(zone.map(\.role), [.reference, .reference, .edited])
        // A new start sends the previous one back to neutral, never to another role.
        XCTAssertEqual(L.assigning(.start, to: 1, in: slots(.start, .neutral)).map(\.role), [.neutral, .start])
        // Another role leaves the others alone.
        XCTAssertEqual(L.assigning(.neutral, to: 1, in: slots(.edited, .reference)).map(\.role), [.edited, .neutral])
        // An unknown id changes nothing.
        XCTAssertEqual(L.assigning(.edited, to: 9, in: slots(.edited)).map(\.role), [.edited])
    }

    func testAChangeOfModelNeverGivesNorConvertsARole() {
        // Qwen-Image-2.1 → Z-Image: the edited image does not become the start of an img2img nobody asked for — it keeps
        // its role, without effect, and nothing goes.
        let z = L.conforming(slots(.edited, .reference, .neutral), capacity: 0, startsFromImage: true)
        XCTAssertEqual(z.map(\.role), [.edited, .reference, .neutral])
        XCTAssertEqual(try L.carry(text: "a quay", slots: z, subjects: [], capacity: 0, startsFromImage: true).get().images, [])
        XCTAssertEqual(L.unused(z, capacity: 0, startsFromImage: true), 3)
        // Back to Qwen-Image-2.1: they work again.
        XCTAssertEqual(L.conforming(z, capacity: 3).map(\.role), [.edited, .reference, .neutral])
        XCTAssertEqual(L.unused(z, capacity: 3), 1)
        // Z-Image → Qwen-Image-2.1: the start does not become the edited image, and a neutral image stays neutral.
        let qwen = L.conforming(slots(.start, .neutral), capacity: 3)
        XCTAssertEqual(qwen.map(\.role), [.start, .neutral])
        XCTAssertEqual(try L.carry(text: "a quay", slots: qwen, subjects: [], capacity: 3).get().images, [])
        XCTAssertFalse(L.isEffective(.start, capacity: 3))
        // A newcomer under Qwen takes image 1: the start without effect holds no place.
        XCTAssertEqual(L.defaultRole(adding: qwen, capacity: 3), .edited)
        // Between two models that start from an image, the start stays.
        XCTAssertEqual(L.conforming(slots(.neutral, .start), capacity: 0, startsFromImage: true).map(\.role), [.neutral, .start])
        // A model that reads nothing and starts from nothing: nothing goes.
        XCTAssertEqual(L.unused(slots(.start, .edited), capacity: 0), 2)
        // Between two edit models, the roles stay; a second edited image becomes a reference.
        XCTAssertEqual(L.conforming(slots(.edited, .reference), capacity: 3).map(\.role), [.edited, .reference])
        XCTAssertEqual(L.conforming(slots(.edited, .edited), capacity: 3).map(\.role), [.edited, .reference])
    }

    func testAModelThatReadsFewerImagesLeavesTheLastOnesNeutralTheEditedOneLast() {
        XCTAssertEqual(L.conforming(slots(.edited, .reference, .reference), capacity: 2).map(\.role), [.edited, .reference, .neutral])
        XCTAssertEqual(L.conforming(slots(.reference, .reference, .edited), capacity: 1).map(\.role), [.neutral, .neutral, .edited])
    }

    // ── the numbering ──

    func testTheEditedImageIsImageOneWhateverTheOrderOfTheImagesAndOfTheWords() throws {
        // The reference sits first, the edited image after it: the edited image still takes place 1.
        let r = try carried("@alice beside the dog", slots(.reference, .edited))
        XCTAssertEqual(r.images, [.plain(1, .edited), .plain(0, .reference), .subject(N.Placed(subject: library[0], picture: library[0].pictures[0], number: 3))])
        XCTAssertTrue(r.prompt.hasPrefix("the person in \(N.Phrasing.reference(3)) beside the dog"), r.prompt)
    }

    func testTheCitedSubjectsTakeThePlacesThatFollowInTheOrderOfTheText() throws {
        let r = try carried("@red-coat on @alice", slots(.edited))
        let placed = r.images.compactMap { image -> N.Placed? in if case .subject(let p) = image { p } else { nil } }
        XCTAssertEqual(placed.map(\.number), [2, 3])
        XCTAssertEqual(placed.map(\.subject.name), ["red-coat", "alice"])
        XCTAssertEqual(r.images.first, .plain(0, .edited))
        XCTAssertEqual(r.cited.map(\.name), ["red-coat", "alice"])
        // The same rule, called as the queue calls it (`AppState.namedJob`).
        let direct = try N.rewrite("@red-coat on @alice", subjects: library, capacity: 3, first: 2)
        XCTAssertEqual(r.prompt, direct.prompt)
    }

    func testWithoutAnEditedImageTheSubjectsTakeOneTwoThree() throws {
        let r = try carried("@alice in @red-coat on @the-beach")
        XCTAssertEqual(r.images.count, 3)
        XCTAssertEqual(r.images.compactMap { if case .subject(let p) = $0 { p.number } else { nil } }, [1, 2, 3])
        // Neutral images are not sent.
        let described = try carried("@alice", slots(.neutral, .neutral))
        XCTAssertEqual(described.images.count, 1)
    }

    func testASubjectWithTwoPicturesTakesTwoPlaces() throws {
        let r = try carried("@Bob with a dog", slots(.edited))
        XCTAssertEqual(r.images.count, 3)
        XCTAssertEqual(r.images.compactMap { if case .subject(let p) = $0 { p.number } else { nil } }, [2, 3])
    }

    func testAPromptThatCitesNobodySendsTheImagesAlone() throws {
        let r = try carried("a dog in a library", slots(.edited, .reference, .neutral))
        XCTAssertEqual(r.prompt, "a dog in a library")
        XCTAssertEqual(r.images, [.plain(0, .edited), .plain(1, .reference)])
        XCTAssertTrue(try carried("a dog", slots(.neutral)).images.isEmpty)
    }

    // ── the refusals ──

    func testThreeImagesAtMost() {
        XCTAssertEqual(refusal("a dog", slots(.edited, .reference, .reference, .reference)), .tooManyImages(count: 4, max: 3))
        XCTAssertEqual(refusal("@alice", slots(.edited, .reference, .reference)), .subject(.tooManyImages(count: 4, max: 3)))
        // A subject with two pictures counts twice.
        XCTAssertEqual(refusal("@Bob @lamp", slots(.edited)), .subject(.tooManyImages(count: 4, max: 3)))
        XCTAssertNil(refusal("@alice @lamp", slots(.edited)))
        // A model that reads two images only.
        XCTAssertEqual(refusal("a dog", slots(.edited, .reference, .reference), capacity: 2), .tooManyImages(count: 3, max: 2))
    }

    func testAnUnknownSubjectIsRefusedNeverPassedAsText() {
        XCTAssertEqual(refusal("@nobody on a quay"), .subject(.unknownSubject(name: "nobody")))
    }

    func testAMentionInsideAGroupIsRefused() throws {
        XCTAssertEqual(refusal("a {@alice|dog} on a quay"), .mentionInGroup(name: "alice"))
        XCTAssertEqual(refusal("@lamp and a {red|@alice}"), .mentionInGroup(name: "alice"))
        // A group without `@` goes through as it is; the mention outside it is translated.
        let r = try carried("@alice in a {library|greenhouse}")
        XCTAssertTrue(r.prompt.contains("{library|greenhouse}"), r.prompt)
        XCTAssertEqual(try PromptAlternatives(r.prompt).count, 2, "the legend adds no group")
        // An escaped brace opens nothing; a mention after a closed group is outside.
        XCTAssertNil(refusal("a \\{ and @alice"))
        XCTAssertNil(refusal("a {red|blue} @alice"))
        // Unbalanced: the rack's to refuse, with its own sentence.
        XCTAssertNil(refusal("a {red @alice"))
    }

    // ── what is greyed does not go ──

    private func sent(_ text: String, capacity: Int, carried: Int) -> String {
        L.sent(text, leaving: L.inert(in: text, known: Set(library.map(\.id)), capacity: capacity, carried: carried))
    }

    func testUnderAModelThatReadsNoImageASubjectIsGreyedAndLeftOutAnotherAtStays() {
        let text = "@alice posing in a library, @incase style"
        let inert = L.inert(in: text, known: Set(library.map(\.id)), capacity: 0, carried: 0)
        XCTAssertEqual(inert.map(\.why), [.subject])
        XCTAssertEqual(String(text[inert[0].range]), "@alice")
        // Anima's artist is the model's own word; the subject takes the space after it, at the start.
        XCTAssertEqual(sent(text, capacity: 0, carried: 0), "posing in a library, @incase style")
        XCTAssertEqual(sent("a girl with @Alice", capacity: 0, carried: 0), "a girl with")
        // Under a model that reads images, a subject goes (translated by the rewrite).
        XCTAssertEqual(sent(text, capacity: 3, carried: 1), text)
    }

    func testAReferenceToAnImageTheRenderDoesNotCarryIsLeftOut() {
        XCTAssertEqual(sent("put the dog of <image3> next to the woman in <image1>", capacity: 3, carried: 2),
                       "put the dog of next to the woman in <image1>")
        XCTAssertEqual(sent("Picture 2 and image 1, then IMAGE 4", capacity: 3, carried: 1), "and image 1, then")
        XCTAssertEqual(L.inert(in: "<image 3>", known: [], capacity: 3, carried: 2).map(\.why), [.image(3)])
        // Under a model that reads none, no image is carried: every reference is left out.
        XCTAssertEqual(sent("image 1 of a cat", capacity: 0, carried: 0), "of a cat")
        // Not a reference: a number glued to the word, or zero.
        XCTAssertEqual(sent("image2 and image 0", capacity: 3, carried: 0), "image2 and image 0")
    }

    func testNothingGreyedLeavesTheTextToTheByte() {
        let text = "a  quay,   at night\n\n<image1> stays"
        XCTAssertTrue(L.inert(in: text, known: Set(library.map(\.id)), capacity: 3, carried: 1).isEmpty)
        XCTAssertEqual(sent(text, capacity: 3, carried: 1), text)
    }

    func testAModelThatReadsNoImageKeepsTheTextAndTheAtSign() throws {
        // Anima's artists: `@incase` is the model's own word, whatever the library holds.
        let r = try carried("a girl, @incase, @alice", slots(.neutral), capacity: 0)
        XCTAssertEqual(r.prompt, "a girl, @incase, @alice")
        XCTAssertTrue(r.images.isEmpty)
        XCTAssertNil(refusal("a {@incase|b}", capacity: 0))
        // An img2img receives its start alone, wherever it sits.
        let start = try L.carry(text: "a dog", slots: slots(.neutral, .start), subjects: library, capacity: 0, startsFromImage: true).get()
        XCTAssertEqual(start.images, [.plain(1, .start)])
        XCTAssertEqual(start.prompt, "a dog")
    }

    // ── what the rack received before, and receives now ──

    /// **The former Subjects mode, as the rack made it at the click** (`AppState.rewrite`, then `namedJob`): the edit's
    /// images were put aside in that mode (`first` = 1), the capacity was the model's, the prompt that of the form's
    /// expansion, and the render read the model's prompt with the subjects' pictures in the order of the rewrite.
    private func formerRack(_ typed: String, capacity: Int) throws -> Request {
        let expanded = try PromptAlternatives(typed).prompts[0]
        let r = try N.rewrite(expanded, subjects: library, capacity: capacity, first: 1)
        var request = Request(r.prompt, width: 512, height: 512, seed: 42)
        request.references = r.placed.map(picture)
        return request
    }

    /// Now: carry, then what the render reads.
    private func lab(_ typed: String, capacity: Int) throws -> Request {
        let carry = try L.carry(text: typed, slots: [L.Slot<Int>](), subjects: library, capacity: capacity).get()
        var request = Request(carry.prompt, width: 512, height: 512, seed: 42)
        request.references = carry.images.map { image in
            guard case .subject(let p) = image else { fatalError("a lab without plain image") }
            return picture(p)
        }
        return request
    }

    /// A picture stands for itself by its colour: the subject, the picture and the number are all in it.
    private func picture(_ p: N.Placed) -> ImageRGB {
        let letters = (p.subject.name + p.picture.file).unicodeScalars.reduce(0) { $0 + Int($1.value) }
        let value = Float(letters % 251) / 255 + Float(p.number) / 10
        return ImageRGB(pixels: [Float](repeating: value, count: 12), height: 2, width: 2)
    }

    func testTheLabCarriesExactlyWhatTheFormerSubjectsModeSent() throws {
        for scene in ["@alice in @red-coat on @the-beach",     // three subjects, one picture each
                      "@Bob with a dog at the door",           // one subject with two pictures
                      "a portrait of @alice, then @alice again"] {
            let former = try formerRack(scene, capacity: 3)
            let carried = try lab(scene, capacity: 3)
            XCTAssertEqual(carried, former, scene)
            XCTAssertEqual(carried.prompt, former.prompt)
            XCTAssertEqual(carried.references.count, former.references.count)
        }
        XCTAssertEqual(try lab("@Bob with a dog at the door", capacity: 3).references.count, 2, "one subject, two images")
        XCTAssertEqual(try lab("@alice in @red-coat on @the-beach", capacity: 3).references.count, 3)
    }

    func testTheSceneWithAGroupIsTheFormerRacksOnceExpanded() throws {
        // The rack expands the group of the prompt the lab carried: each expansion is what the former mode rewrote.
        let typed = "@alice in a {library|greenhouse}"
        let carry = try L.carry(text: typed, slots: [L.Slot<Int>](), subjects: library, capacity: 3).get()
        let expansions = try PromptAlternatives(carry.prompt).prompts
        XCTAssertEqual(expansions.count, 2)
        for (carried, former) in zip(expansions, try PromptAlternatives(typed).prompts) {
            XCTAssertEqual(carried, try N.rewrite(former, subjects: library, capacity: 3).prompt)
        }
    }

    // ── the mentions, kept through a rewrite ──

    func testTheMentionsAreMaskedByANeutralTokenAndPutBack() {
        let text = "@Alice on a quay, then @alice at the door, with @lamp"
        let guarded = L.MentionGuard(text)
        XCTAssertEqual(guarded.masked, "[S1] on a quay, then [S1] at the door, with [S2]", "one token per subject, case apart")
        XCTAssertFalse(guarded.masked.contains("@"))
        XCTAssertEqual(guarded.tokens.map(\.mention), ["@Alice", "@lamp"])
        let back = guarded.restore(guarded.masked)
        XCTAssertEqual(back.text, "@Alice on a quay, then @Alice at the door, with @lamp")
        XCTAssertTrue(back.intact)
    }

    func testARewriteThatKeepsTheTokensMovesTheMentionsWithThem() {
        let guarded = L.MentionGuard("@alice on a quay")
        let written = "A rain-soaked railway quay at night, neon reflections; [S1] stands under a lamp, looking up."
        let back = guarded.restore(written)
        XCTAssertEqual(back.text, "A rain-soaked railway quay at night, neon reflections; @alice stands under a lamp, looking up.")
        XCTAssertTrue(back.intact)
        // Case and spaces inside the brackets are forgiven.
        XCTAssertEqual(guarded.restore("[s 1] waits").text, "@alice waits")
        XCTAssertTrue(guarded.restore("[s 1] waits").intact)
    }

    func testALostMentionIsReportedNeverDroppedInSilence() {
        let guarded = L.MentionGuard("@alice and @lamp on a quay")
        let back = guarded.restore("A woman and a lamp on a quay, [S2] glowing.")
        XCTAssertEqual(back.lost, ["@alice"])
        XCTAssertFalse(back.intact)
        XCTAssertEqual(back.text, "A woman and a lamp on a quay, @lamp glowing.")
        // Nothing of the mention left: said, whatever the rest.
        XCTAssertEqual(guarded.restore("").lost, ["@alice", "@lamp"])
    }

    func testADeformedTokenIsReported() {
        let guarded = L.MentionGuard("@alice and @lamp")
        for written in ["[S1 and [S2]", "[S1] and [S7]", "[S1] and [S 2"] {
            let back = guarded.restore(written)
            XCTAssertTrue(back.deformed || !back.lost.isEmpty, written)
            XCTAssertFalse(back.intact, written)
        }
    }

    func testATextWithoutMentionIsLeftAlone() {
        let guarded = L.MentionGuard("woman posing in a library")
        XCTAssertTrue(guarded.isEmpty)
        XCTAssertEqual(guarded.masked, "woman posing in a library")
        XCTAssertEqual(guarded.restore("A candid photograph of a woman").text, "A candid photograph of a woman")
        XCTAssertTrue(guarded.restore("anything").intact)
    }

    func testATokenTheTextAlreadyUsesIsNotTaken() {
        let guarded = L.MentionGuard("the card [S1] and @alice")
        XCTAssertFalse(guarded.masked.contains("[S2]"))
        XCTAssertTrue(guarded.masked.contains("the card [S1]"), "the user's own bracket is untouched")
        let back = guarded.restore(guarded.masked)
        XCTAssertEqual(back.text, "the card [S1] and @alice")
        XCTAssertTrue(back.intact)
    }
}
