import XCTest
@testable import Siliconed

/// **The Prompt Lab's inputs** (`LabInputs`): roles, numbering, refusals, the mentions kept through a rewrite — and the
/// proof that what the lab carries to the rack is what the rack's former Subjects mode made at the click.
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

    func testADroppedImageIsTheStartThenReferencesThenDescribeOnly() {
        var zone: [L.Slot<Int>] = []
        var roles: [L.Role] = []
        for id in 0..<5 {
            let role = L.defaultRole(adding: zone, capacity: 3)
            zone.append(L.Slot(id: id, role: role))
            roles.append(role)
        }
        XCTAssertEqual(roles, [.start, .reference, .reference, .describeOnly, .describeOnly])
        // A model that reads no image: every image stays in the lab.
        XCTAssertEqual(L.defaultRole(adding: [] as [L.Slot<Int>], capacity: 0), .describeOnly)
        XCTAssertEqual(L.roles(capacity: 0), [.describeOnly])
        XCTAssertEqual(L.roles(capacity: 3), [.start, .reference, .describeOnly])
    }

    func testChoosingTheStartMakesThePreviousOneAReference() {
        let zone = L.assigning(.start, to: 2, in: slots(.start, .reference, .reference))
        XCTAssertEqual(zone.map(\.role), [.reference, .reference, .start])
        XCTAssertEqual(zone.filter { $0.role == .start }.count, 1)
        // Another role leaves the others alone.
        XCTAssertEqual(L.assigning(.describeOnly, to: 1, in: slots(.start, .reference)).map(\.role), [.start, .describeOnly])
        // An unknown id changes nothing.
        XCTAssertEqual(L.assigning(.start, to: 9, in: slots(.start)).map(\.role), [.start])
    }

    func testChangingTheModelAimedAtDropsTheRolesItDoesNotOffer() {
        XCTAssertEqual(L.conforming(slots(.start, .reference, .describeOnly), capacity: 0).map(\.role),
                       [.describeOnly, .describeOnly, .describeOnly])
        XCTAssertEqual(L.conforming(slots(.start, .reference), capacity: 3).map(\.role), [.start, .reference])
        XCTAssertEqual(L.conforming(slots(.start, .start), capacity: 3).map(\.role), [.start, .reference])
    }

    // ── the numbering ──

    func testTheStartIsImageOneWhateverTheOrderOfTheZoneAndOfTheWords() throws {
        // The reference sits first in the zone, the start after it: the start still takes place 1.
        let r = try carried("@alice beside the dog", slots(.reference, .start))
        XCTAssertEqual(r.images, [.plain(1, .start), .plain(0, .reference), .subject(N.Placed(subject: library[0], picture: library[0].pictures[0], number: 3))])
        XCTAssertTrue(r.prompt.hasPrefix("the person in \(N.Phrasing.reference(3)) beside the dog"), r.prompt)
    }

    func testTheCitedSubjectsTakeThePlacesThatFollowInTheOrderOfTheText() throws {
        let r = try carried("@red-coat on @alice", slots(.start))
        let placed = r.images.compactMap { image -> N.Placed? in if case .subject(let p) = image { p } else { nil } }
        XCTAssertEqual(placed.map(\.number), [2, 3])
        XCTAssertEqual(placed.map(\.subject.name), ["red-coat", "alice"])
        XCTAssertEqual(r.images.first, .plain(0, .start))
        XCTAssertEqual(r.cited.map(\.name), ["red-coat", "alice"])
        // The same rule, called as the lab calls it.
        let direct = try N.rewrite("@red-coat on @alice", subjects: library, capacity: 3, first: 2)
        XCTAssertEqual(r.prompt, direct.prompt)
    }

    func testWithoutAStartTheSubjectsTakeOneTwoThree() throws {
        let r = try carried("@alice in @red-coat on @the-beach")
        XCTAssertEqual(r.images.count, 3)
        XCTAssertEqual(r.images.compactMap { if case .subject(let p) = $0 { p.number } else { nil } }, [1, 2, 3])
        // Images that are only described are not sent.
        let described = try carried("@alice", slots(.describeOnly, .describeOnly))
        XCTAssertEqual(described.images.count, 1)
    }

    func testASubjectWithTwoPicturesTakesTwoPlaces() throws {
        let r = try carried("@Bob with a dog", slots(.start))
        XCTAssertEqual(r.images.count, 3)
        XCTAssertEqual(r.images.compactMap { if case .subject(let p) = $0 { p.number } else { nil } }, [2, 3])
    }

    func testAPromptThatCitesNobodySendsTheZoneAlone() throws {
        let r = try carried("a dog in a library", slots(.start, .reference, .describeOnly))
        XCTAssertEqual(r.prompt, "a dog in a library")
        XCTAssertEqual(r.images, [.plain(0, .start), .plain(1, .reference)])
        XCTAssertTrue(try carried("a dog", slots(.describeOnly)).images.isEmpty)
    }

    // ── the refusals ──

    func testThreeImagesAtMost() {
        XCTAssertEqual(refusal("a dog", slots(.start, .reference, .reference, .reference)), .tooManyImages(count: 4, max: 3))
        XCTAssertEqual(refusal("@alice", slots(.start, .reference, .reference)), .subject(.tooManyImages(count: 4, max: 3)))
        // A subject with two pictures counts twice.
        XCTAssertEqual(refusal("@Bob @lamp", slots(.start)), .subject(.tooManyImages(count: 4, max: 3)))
        XCTAssertNil(refusal("@alice @lamp", slots(.start)))
        // A model that reads two images only.
        XCTAssertEqual(refusal("a dog", slots(.start, .reference, .reference), capacity: 2), .tooManyImages(count: 3, max: 2))
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

    func testAModelThatReadsNoImageKeepsTheTextAndTheAtSign() throws {
        // Anima's artists: `@incase` is the model's own word, whatever the library holds.
        let r = try carried("a girl, @incase, @alice", slots(.describeOnly), capacity: 0)
        XCTAssertEqual(r.prompt, "a girl, @incase, @alice")
        XCTAssertTrue(r.images.isEmpty)
        XCTAssertNil(refusal("a {@incase|b}", capacity: 0))
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

    /// The lab: carry, then what the rack's Prompt and Images boxes hold and the render reads.
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

    // ── « The rack will receive » ──

    private func receipt(_ text: String, _ zone: [L.Slot<Int>] = [], capacity: Int = 3, rackPrompt: String = "",
                         rackImages: Int = 0, changesModel: Bool = false) -> L.Receipt<Int> {
        L.receipt(text: text, slots: zone, subjects: library, capacity: capacity, rackPrompt: rackPrompt,
                  rackImages: rackImages, changesModel: changesModel)
    }

    func testTheLineSaysWhatTheRackWillReceiveAndWhatItReplaces() throws {
        let r = receipt("@alice on a quay", slots(.start), rackPrompt: "old", rackImages: 2)
        guard case .carries(let c) = r.outcome else { return XCTFail("\(r.outcome)") }
        XCTAssertEqual(c.images.count, 2)
        XCTAssertEqual(r.first, .plain(0, .start), "the format is image 1's")
        XCTAssertTrue(r.replacesPrompt)
        XCTAssertEqual(r.replacedImages, 2)
        XCTAssertTrue(r.canCarry)
        // No image to send: the rack's own stay.
        let text = receipt("a dog in a library", rackPrompt: "old", rackImages: 2)
        XCTAssertEqual(text.replacedImages, 0)
        XCTAssertNil(text.first)
        // The same prompt as the rack's replaces nothing.
        XCTAssertFalse(receipt("a dog", rackPrompt: "a dog").replacesPrompt)
        // Images alone leave the rack's Prompt in place.
        let images = receipt("  ", slots(.start), rackPrompt: "old", rackImages: 1)
        XCTAssertFalse(images.replacesPrompt)
        XCTAssertEqual(images.replacedImages, 1)
        XCTAssertTrue(images.canCarry)
        XCTAssertTrue(receipt("x", changesModel: true).changesModel)
    }

    func testTheLineSaysWhyItCannotInOneSentenceOfData() {
        XCTAssertEqual(receipt("   ").outcome, .nothing)
        XCTAssertEqual(receipt("   ", slots(.describeOnly)).outcome, .nothing)
        XCTAssertFalse(receipt("").canCarry)
        XCTAssertEqual(receipt("@nobody").outcome, .refused(.subject(.unknownSubject(name: "nobody"))))
        XCTAssertEqual(receipt("a {@alice|b}").outcome, .refused(.mentionInGroup(name: "alice")))
        XCTAssertEqual(receipt("x", slots(.start, .reference, .reference, .reference)).outcome,
                       .refused(.tooManyImages(count: 4, max: 3)))
        XCTAssertFalse(receipt("@nobody", rackPrompt: "old", rackImages: 3).replacesPrompt, "a refused carry replaces nothing")
        XCTAssertEqual(receipt("@nobody", rackImages: 3).replacedImages, 0)
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
