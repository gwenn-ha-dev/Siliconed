import Foundation

// **The Prompt Lab's inputs, and what it carries to the rack** — the rules, without a window.
//
//     Images zone                        Text                          the model aimed at
//     ┌ 1 start ┐ ┌ 2 reference ┐ ┌ @alice ┐ ┌ describe only ┐    "@alice on a quay, at night"     capacity 3 (Qwen-Image-2.1)
//     └─────────┘ └─────────────┘ └────────┘ └───────────────┘           │                           capacity 0 (the others)
//          │ the gesture decides, never the words      │ cited, in the order of the text
//          ▼                                           ▼
//     LabInputs.carry ─▶ prompt  "the person in <image3> on a quay, at night\n\nTake only their identity from <image3>: …"
//                        images  1 start · 2 reference · 3 @alice          ◀ what the rack's Prompt and Images boxes receive
//
// What the lab carries is **exactly what the rack's former Subjects mode made at the click**
// (`NamedSubjects.rewrite`, then the subjects' pictures after the edit's own): the rack no longer knows the
// subjects, it receives their translation. Only the rules are here — the window (`Lab.swift`) holds the pictures.
//
// - **The starting image is chosen by a gesture, never by the order of the words.** It is image 1 — the one the
//   model edits, whose proportions give the format —, the references come next, and the subjects the text cites
//   take the following places in the order they are first named (`first:` of `NamedSubjects.rewrite`).
// - **Three images at most** (`NamedSubjects.maxImages`), the starting image and the references counted with the
//   subjects: the worst case measured is 1024×1536 with three references.
// - **A model that reads no images** (capacity 0) keeps the text as typed — `@` is then its own character (Anima's
//   artists) — and every image is « describe only »: it stays in the lab and goes nowhere — **unless the model starts from
//   an image** (img2img: Z-Image, Anima, Krea 2; `startsFromImage`): then one image may be the start, the image the render
//   re-noises and denoises, chosen by a gesture as for Qwen-Image-2.1. A newcomer is never a start there: dropping a picture
//   to describe it must not turn the next render into an img2img nobody asked for.
// - **A `@` inside a `{a|b}` group is refused**: the rack expands the group, and each expansion would need its own
//   translation. A group without `@` goes through as it is.
package enum LabInputs {
    // MARK: Roles

    /// What the render makes of an image of the lab.
    package enum Role: String, CaseIterable, Sendable {
        /// The image the model edits (Qwen-Image-2.1), or the one it starts from (img2img): image 1, one only, and it gives
        /// the format.
        case start
        /// Context the model sees: images 2 and 3.
        case reference
        /// Nothing: the image stays in the lab, to be described.
        case describeOnly

        /// Whether the render receives it.
        package var isSent: Bool { self != .describeOnly }
    }

    /// An image of the zone, by whatever names it in the window.
    package struct Slot<ID: Hashable & Sendable>: Sendable {
        package var id: ID
        package var role: Role
        package init(id: ID, role: Role) { self.id = id; self.role = role }
    }

    /// **The roles a model offers**: all three when it reads images (`capacity` > 0); the start and « describe only » when it
    /// reads none but starts from an image (`startsFromImage`, img2img); « describe only » otherwise.
    package static func roles(capacity: Int, startsFromImage: Bool = false) -> [Role] {
        if capacity > 0 { return [.start, .reference, .describeOnly] }
        return startsFromImage ? [.start, .describeOnly] : [.describeOnly]
    }

    /// **The role of an image just dropped**: the first one sent is the start, the next ones references — as long as
    /// the render has room (`capacity`, three at most); beyond, or with a model that reads no image, « describe only » —
    /// an img2img model's included: its start is a gesture, never a default.
    package static func defaultRole<ID>(adding slots: [Slot<ID>], capacity: Int) -> Role {
        let limit = min(capacity, NamedSubjects.maxImages)
        let sent = slots.filter { $0.role.isSent }
        guard limit > 0, sent.count < limit else { return .describeOnly }
        return sent.isEmpty ? .start : .reference
    }

    /// **Gives `id` a role.** The start is unique: choosing it for one image makes the previous start a reference — or
    /// « describe only » for a model that reads no references (img2img: `startsFromImage`, `capacity` 0).
    package static func assigning<ID>(_ role: Role, to id: ID, in slots: [Slot<ID>], capacity: Int = 3) -> [Slot<ID>] {
        var out = slots
        guard let i = out.firstIndex(where: { $0.id == id }) else { return out }
        if role == .start {
            for j in out.indices where out[j].role == .start { out[j].role = capacity > 0 ? .reference : .describeOnly }
        }
        out[i].role = role
        return out
    }

    /// **After a change of the model aimed at**: an image whose role it no longer offers becomes « describe only »;
    /// a second start (never made by the window) becomes a reference — or « describe only » where there are no references.
    package static func conforming<ID>(_ slots: [Slot<ID>], capacity: Int, startsFromImage: Bool = false) -> [Slot<ID>] {
        let offered = roles(capacity: capacity, startsFromImage: startsFromImage)
        var out = slots
        var started = false
        for i in out.indices {
            if !offered.contains(out[i].role) { out[i].role = .describeOnly }
            if out[i].role == .start {
                if started { out[i].role = offered.contains(.reference) ? .reference : .describeOnly }
                started = true
            }
        }
        return out
    }

    // MARK: What the lab carries

    /// What stops the lab from carrying its inputs, each said in one sentence by the window.
    package enum Refusal: Error, Equatable, Sendable {
        /// A `@name` the library does not know, no picture, or too many images with the subjects.
        case subject(NamedSubjects.Failure)
        /// The starting image and the references alone exceed what a render takes.
        case tooManyImages(count: Int, max: Int)
        /// `@name` inside a `{a|b}` group.
        case mentionInGroup(name: String)
    }

    /// One image of the render, numbered by its place in `Carry.images` (from 1).
    package enum Image<ID: Hashable & Sendable>: Equatable, Sendable {
        case plain(ID, Role)
        case subject(NamedSubjects.Placed)
    }

    /// **What the rack's Prompt and Images boxes receive.**
    package struct Carry<ID: Hashable & Sendable>: Equatable, Sendable {
        /// The prompt the model reads: the text, its `@names` translated (`image 2`, a legend after).
        package let prompt: String
        /// The images in the order of their numbers, the first one giving the format.
        package let images: [Image<ID>]
        /// The subjects cited, in order of first appearance.
        package let cited: [NamedSubjects.Subject]
    }

    /// **Translates the text and numbers the images.** `capacity` is how many images the model aimed at reads
    /// (0: none), `subjects` the library. The starting image goes first, then the references in the zone's order,
    /// then the subjects the text cites. A model that reads none but starts from an image (`startsFromImage`) receives the
    /// text as typed and its start alone.
    package static func carry<ID>(text: String, slots: [Slot<ID>], subjects: [NamedSubjects.Subject],
                                  capacity: Int, startsFromImage: Bool = false) -> Result<Carry<ID>, Refusal> {
        guard capacity > 0 else {
            let start = startsFromImage ? slots.first { $0.role == .start }.map { [Image<ID>.plain($0.id, .start)] } ?? [] : []
            return .success(Carry(prompt: text, images: start, cited: []))
        }
        let start = slots.filter { $0.role == .start }
        let references = slots.filter { $0.role == .reference }
        let plain = (start + references).map { Image<ID>.plain($0.id, $0.role) }
        let limit = min(capacity, NamedSubjects.maxImages)
        if plain.count > limit { return .failure(.tooManyImages(count: plain.count, max: limit)) }

        let found = NamedSubjects.mentions(in: text)
        if let inGroup = found.first(where: { isInsideGroup($0, in: text) }) {
            return .failure(.mentionInGroup(name: inGroup.name))
        }
        let rewritten: NamedSubjects.Rewritten
        do {
            rewritten = try NamedSubjects.rewrite(text, subjects: subjects, capacity: capacity, first: plain.count + 1)
        } catch {
            return .failure(.subject(error))
        }
        return .success(Carry(prompt: rewritten.prompt, images: plain + rewritten.placed.map { .subject($0) },
                              cited: rewritten.cited))
    }

    // MARK: « The rack will receive »

    /// **The line under the lab**: what « Carry to the Rack » will hand over and replace — or why it can not. The window
    /// says it in words; the facts are here.
    package struct Receipt<ID: Hashable & Sendable>: Equatable, Sendable {
        package enum Outcome: Equatable, Sendable {
            /// Nothing to carry: no text, no image sent.
            case nothing
            case refused(Refusal)
            case carries(Carry<ID>)
        }
        package let outcome: Outcome
        /// The rack's Prompt is replaced (⌘Z brings it back): the lab has a text, and it differs from the rack's.
        package let replacesPrompt: Bool
        /// The rack's images that are replaced, none of which can be brought back: 0 when the lab sends no image.
        package let replacedImages: Int
        /// The rack takes the lab's model: it was on another family.
        package let changesModel: Bool

        /// Image 1, which gives the format, when the lab sends images.
        package var first: Image<ID>? {
            if case .carries(let c) = outcome { c.images.first } else { nil }
        }
        /// Whether the button is on.
        package var canCarry: Bool { if case .carries = outcome { true } else { false } }
    }

    /// **What carrying would do now.** `rackPrompt` and `rackImages` are the rack's own (for an img2img model, its starting
    /// image: 0 or 1), `changesModel` whether the rack must change family to receive it.
    package static func receipt<ID>(text: String, slots: [Slot<ID>], subjects: [NamedSubjects.Subject], capacity: Int,
                                    startsFromImage: Bool = false,
                                    rackPrompt: String, rackImages: Int, changesModel: Bool) -> Receipt<ID> {
        let blank = text.allSatisfy(\.isWhitespace)
        switch carry(text: text, slots: slots, subjects: subjects, capacity: capacity, startsFromImage: startsFromImage) {
        case .failure(let refusal):
            return Receipt(outcome: .refused(refusal), replacesPrompt: false, replacedImages: 0, changesModel: changesModel)
        case .success(let c):
            guard !blank || !c.images.isEmpty else {
                return Receipt(outcome: .nothing, replacesPrompt: false, replacedImages: 0, changesModel: changesModel)
            }
            return Receipt(outcome: .carries(c), replacesPrompt: !blank && c.prompt != rackPrompt,
                           replacedImages: c.images.isEmpty ? 0 : rackImages, changesModel: changesModel)
        }
    }

    /// Whether the mention sits between an unescaped `{` and its `}` (`PromptAlternatives`' syntax: `\{`, `\}`
    /// are literal characters). An unbalanced prompt is the rack's to refuse: nothing is inside a group then.
    private static func isInsideGroup(_ mention: NamedSubjects.Mention, in text: String) -> Bool {
        guard (try? PromptAlternatives(text, limit: .max)) != nil else { return false }
        var open = false
        var i = text.startIndex
        while i < mention.range.lowerBound {
            let c = text[i]
            if c == "\\" {
                let next = text.index(after: i)
                if next < text.endIndex, "{}|".contains(text[next]) {
                    i = text.index(after: next)
                    continue
                }
            } else if c == "{" {
                open = true
            } else if c == "}" {
                open = false
            }
            i = text.index(after: i)
        }
        return open
    }

    // MARK: The mentions, kept through a rewrite

    /// **The `@names` of a text, put aside while the writer rewrites it.** The writer's instructions know nothing of
    /// them (the only `@` they speak of are Anima's and MOSS's artists), and a language model reads `@alice` as
    /// content to embroider. Each distinct mention becomes a neutral token, `[S1]`, which the writer has nothing to
    /// say about; the token is put back afterwards. **A mention the writer lost or deformed is reported, never
    /// dropped in silence**: the text then holds fewer mentions than the one rewritten, and `restore` says which.
    package struct MentionGuard: Sendable, Equatable {
        /// The text the writer reads.
        package let masked: String
        /// Each token and the mention it stands for, as typed.
        package let tokens: [(token: String, mention: String)]

        package static func == (a: Self, b: Self) -> Bool {
            a.masked == b.masked && a.tokens.map(\.token) == b.tokens.map(\.token) && a.tokens.map(\.mention) == b.tokens.map(\.mention)
        }

        /// What came back from the writer.
        package struct Restored: Equatable, Sendable {
            package let text: String
            /// The mentions whose token is not in the writer's text.
            package let lost: [String]
            /// The text holds something that looks like a token and is none (`[S 9]`, `[S1`): a mention may be deformed.
            package let deformed: Bool
            /// Nothing was lost or deformed.
            package var intact: Bool { lost.isEmpty && !deformed }
        }

        /// Whether the text cites anybody: when not, the writer reads it as it is.
        package var isEmpty: Bool { tokens.isEmpty }

        private let letter: Character

        package init(_ text: String) {
            // A letter no `[X1]` of the text itself uses.
            let used = Set(Self.residue(in: text).map { $0.letter })
            let letter = "SQZXWV".first { !used.contains($0) } ?? "S"
            self.letter = letter
            var table: [(token: String, mention: String)] = []
            var keys: [String: String] = [:]
            var masked = ""
            var i = text.startIndex
            for m in NamedSubjects.mentions(in: text) {
                masked += text[i..<m.range.lowerBound]
                let key = NamedSubjects.key(m.name)
                let token: String
                if let known = keys[key] {
                    token = known
                } else {
                    token = "[\(letter)\(table.count + 1)]"
                    keys[key] = token
                    table.append((token: token, mention: String(text[m.range])))
                }
                masked += token
                i = m.range.upperBound
            }
            masked += text[i...]
            self.masked = masked
            self.tokens = table
        }

        /// **Puts the mentions back** in what the writer wrote. A token is found whatever its case and the spaces
        /// inside it (`[s 1]`).
        package func restore(_ written: String) -> Restored {
            var text = written
            var lost: [String] = []
            for (n, entry) in tokens.enumerated() {
                let pattern = "\\[\\s*\(letter)\\s*\(n + 1)\\s*\\]"
                guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { continue }
                let range = NSRange(text.startIndex..., in: text)
                if regex.firstMatch(in: text, range: range) == nil {
                    lost.append(entry.mention)
                    continue
                }
                text = regex.stringByReplacingMatches(in: text, range: range,
                                                      withTemplate: NSRegularExpression.escapedTemplate(for: entry.mention))
            }
            let rest = Self.residue(in: text).contains { $0.letter == letter }
            return Restored(text: text, lost: lost, deformed: rest || Self.opening(in: text, letter: letter))
        }

        /// `[X12]`-looking pieces of a text, tolerant of case and spaces.
        private static func residue(in text: String) -> [(letter: Character, number: Int)] {
            guard let regex = try? NSRegularExpression(pattern: "\\[\\s*([A-Za-z])\\s*(\\d+)\\s*\\]") else { return [] }
            return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap { m in
                guard let l = Range(m.range(at: 1), in: text), let n = Range(m.range(at: 2), in: text),
                      let c = text[l].uppercased().first, let k = Int(text[n]) else { return nil }
                return (c, k)
            }
        }

        /// A bracket opened and never closed around the token's letter and a digit: `[S1`, `[S2 `.
        private static func opening(in text: String, letter: Character) -> Bool {
            guard let regex = try? NSRegularExpression(pattern: "\\[\\s*\(letter)\\s*\\d+(?!\\s*\\d*\\s*\\])", options: [.caseInsensitive])
            else { return false }
            return regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
        }
    }
}
