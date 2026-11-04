import Foundation

// **The render's inputs: one text, one set of images with their roles, and what the model receives** — the rules,
// without a window.
//
//     the images (AppState.inputs)        the text (AppState.prompt)        the model chosen
//     ┌ 1 edited ┐ ┌ 2 reference ┐ ┌ neutral ┐   "@alice on a quay, at night"     capacity 3 (Qwen-Image-2.1)
//     └──────────┘ └─────────────┘ └─────────┘          │                         capacity 0 (the others)
//          │ the gesture decides, never the words        │ cited, in the order of the text
//          ▼                                             ▼
//     LabInputs.carry ─▶ prompt  "the person in <image3> on a quay, at night\n\nTake only their identity from <image3>: …"
//                        images  1 edited · 2 reference · 3 @alice        ◀ what the render reads, made when it is queued
//
// What the render receives is **exactly what the rack's former Subjects mode made at the click**
// (`NamedSubjects.rewrite`, then the subjects' pictures after the edit's own). The translation is made when the render is
// queued (`AppState.namedJob`), never written back into the text: the text keeps its `@names`. Only the rules are here —
// the rack (`AppState+Inputs.swift`) holds the images.
//
// - **A role is chosen by a gesture, never by the order of the words.** The edited image is image 1 — the one the
//   model edits, whose proportions give the format —, the references come next, and the subjects the text cites
//   take the following places in the order they are first named (`first:` of `NamedSubjects.rewrite`).
// - **Two roles that look alike and are not** (4/11): the image Qwen-Image-2.1 *edits* and the image an img2img
//   (Z-Image, Anima, Krea 2, MOSS) *starts from* are two intentions. A change of model never turns one into the other:
//   it would start an img2img nobody asked for. **A change of model never gives a role either**: an image keeps its
//   role, which has no effect where the new model lacks it (`isEffective`) — the window says so, and counts it with the
//   images that do not go (`unused`); back on a model that has it, it works again (`conforming`).
// - **Where an image lands**: under a model that reads images, a newcomer takes the next place (edited, then references)
//   while there is room; elsewhere it is neutral — dropping a picture to describe it must not turn the next render into
//   an img2img (`defaultRole`).
// - **Three images at most** (`NamedSubjects.maxImages`), the edited image and the references counted with the
//   subjects: the worst case measured is 1024×1536 with three references.
// - **A model that reads no images** (capacity 0) keeps the text as typed — `@` is then its own character (Anima's
//   artists) — and receives, at most, its starting image; only the `@names` of the library's subjects are left out.
// - **What is greyed does not go** (`inert`, `sent`): a subject's `@name` under a model that reads no image, a reference to
//   an image the render does not carry (`<image3>`, `image 3`, `Picture 3` with two images). The field greys them, the
//   model's prompt leaves them out with the space before them; the typed text keeps them, for when they are read again.
// - **A `@` inside a `{a|b}` group is refused**: the rack expands the group, and each expansion would need its own
//   translation. A group without `@` goes through as it is.
package enum LabInputs {
    // MARK: Roles

    /// What the render makes of an image.
    package enum Role: String, CaseIterable, Sendable {
        /// The image the model edits (Qwen-Image-2.1): image 1, one only, and it gives the format.
        case edited
        /// Context the model sees: the images after the edited one.
        case reference
        /// The image an img2img re-noises and denoises (Z-Image, Anima, Krea 2, MOSS): one only, its proportions choose the
        /// format among the model's.
        case start
        /// Nothing: the image waits — to be described, or given a role.
        case neutral

        /// Whether the render receives it.
        package var isSent: Bool { self != .neutral }
    }

    /// **Whether a role does something under a model**: offered by it, and not neutral.
    package static func isEffective(_ role: Role, capacity: Int, startsFromImage: Bool = false) -> Bool {
        role.isSent && roles(capacity: capacity, startsFromImage: startsFromImage).contains(role)
    }

    /// **The images that do not go to the render**: neutral, or with a role the model lacks.
    package static func unused<ID>(_ slots: [Slot<ID>], capacity: Int, startsFromImage: Bool = false) -> Int {
        slots.filter { !isEffective($0.role, capacity: capacity, startsFromImage: startsFromImage) }.count
    }

    /// An image, by whatever names it in the window.
    package struct Slot<ID: Hashable & Sendable>: Sendable {
        package var id: ID
        package var role: Role
        package init(id: ID, role: Role) { self.id = id; self.role = role }
    }

    /// **The roles a model offers**: edited, reference, neutral when it reads images (`capacity` > 0); the start and
    /// neutral when it reads none but starts from an image (`startsFromImage`, img2img); neutral otherwise.
    package static func roles(capacity: Int, startsFromImage: Bool = false) -> [Role] {
        if capacity > 0 { return [.edited, .reference, .neutral] }
        return startsFromImage ? [.start, .neutral] : [.neutral]
    }

    /// **The role of an image just dropped**: under a model that reads images, the edited place if it is free, a
    /// reference otherwise — as long as the render has room (`capacity`, three at most); beyond, or with a model that
    /// reads no image, neutral — an img2img model's included: its start is a gesture, never a default.
    package static func defaultRole<ID>(adding slots: [Slot<ID>], capacity: Int) -> Role {
        let limit = min(capacity, NamedSubjects.maxImages)
        guard limit > 0, slots.filter({ $0.role == .edited || $0.role == .reference }).count < limit else { return .neutral }
        return slots.contains { $0.role == .edited } ? .reference : .edited
    }

    /// **Gives `id` a role.** The edited image and the start are unique: choosing one for an image makes the previous
    /// edited image a reference, the previous start neutral.
    package static func assigning<ID>(_ role: Role, to id: ID, in slots: [Slot<ID>]) -> [Slot<ID>] {
        var out = slots
        guard let i = out.firstIndex(where: { $0.id == id }) else { return out }
        if role == .edited { for j in out.indices where out[j].role == .edited { out[j].role = .reference } }
        if role == .start { for j in out.indices where out[j].role == .start { out[j].role = .neutral } }
        out[i].role = role
        return out
    }

    /// **After a change of model**: each image keeps its role — never another one, even where the new model lacks it
    /// (it then has no effect: `isEffective`); a second edited image (never made by the window) becomes a reference, a
    /// second start neutral; and the effective images beyond what the model reads become neutral, from the last.
    package static func conforming<ID>(_ slots: [Slot<ID>], capacity: Int, startsFromImage: Bool = false) -> [Slot<ID>] {
        var out = slots
        var edited = false, started = false
        for i in out.indices {
            if out[i].role == .edited {
                if edited { out[i].role = .reference }
                edited = true
            }
            if out[i].role == .start {
                if started { out[i].role = .neutral }
                started = true
            }
        }
        // The edited image is the last to go: it gives the format, and the limit is never 0 where it is offered.
        let limit = capacity > 0 ? min(capacity, NamedSubjects.maxImages) : (startsFromImage ? 1 : 0)
        let effective = { (role: Role) in isEffective(role, capacity: capacity, startsFromImage: startsFromImage) }
        var excess = out.filter { effective($0.role) }.count - limit
        for i in out.indices.reversed() where excess > 0 && effective(out[i].role) && out[i].role != .edited {
            out[i].role = .neutral
            excess -= 1
        }
        return out
    }

    // MARK: What the render receives

    /// What stops a render from being queued, each said in one sentence by the rack.
    package enum Refusal: Error, Equatable, Sendable {
        /// A `@name` the library does not know, no picture, or too many images with the subjects.
        case subject(NamedSubjects.Failure)
        /// The edited image and the references alone exceed what a render takes.
        case tooManyImages(count: Int, max: Int)
        /// `@name` inside a `{a|b}` group.
        case mentionInGroup(name: String)
    }

    /// One image of the render, numbered by its place in `Carry.images` (from 1).
    package enum Image<ID: Hashable & Sendable>: Equatable, Sendable {
        case plain(ID, Role)
        case subject(NamedSubjects.Placed)
    }

    /// **What the render receives**: made when it is queued, never shown as text in the rack.
    package struct Carry<ID: Hashable & Sendable>: Equatable, Sendable {
        /// The prompt the model reads: the text, its `@names` translated (`image 2`, a legend after) — before the greyed
        /// words leave it, when the render is queued (`sent`).
        package let prompt: String
        /// The images in the order of their numbers, the first one giving the format.
        package let images: [Image<ID>]
        /// The subjects cited, in order of first appearance.
        package let cited: [NamedSubjects.Subject]
    }

    /// **Translates the text and numbers the images.** `capacity` is how many images the model aimed at reads
    /// (0: none), `subjects` the library. The edited image goes first, then the references in their order,
    /// then the subjects the text cites. A model that reads none but starts from an image (`startsFromImage`) receives the
    /// text as typed and its start alone.
    package static func carry<ID>(text: String, slots: [Slot<ID>], subjects: [NamedSubjects.Subject],
                                  capacity: Int, startsFromImage: Bool = false) -> Result<Carry<ID>, Refusal> {
        guard capacity > 0 else {
            let start = startsFromImage ? slots.first { $0.role == .start }.map { [Image<ID>.plain($0.id, .start)] } ?? [] : []
            return .success(Carry(prompt: text, images: start, cited: []))
        }
        let edited = slots.filter { $0.role == .edited }
        let references = slots.filter { $0.role == .reference }
        let plain = (edited + references).map { Image<ID>.plain($0.id, $0.role) }
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

    // MARK: What is greyed does not go

    /// **A word of the text the render does not send**, and why.
    package struct Inert: Equatable, Sendable {
        package enum Why: Equatable, Sendable {
            /// A subject of the library, under a model that reads no image.
            case subject
            /// A reference to image N, past those the render carries.
            case image(Int)
        }
        package let range: Range<String.Index>
        package let why: Why
    }

    /// `<image3>`, `image 3`, `Picture 3` — the three forms `NamedSubjects.Phrasing.Reference` knows, in any case.
    nonisolated(unsafe) private static let imageReference = try! NSRegularExpression(
        pattern: "<image\\s*([0-9]+)>|\\b(?:image|picture)\\s+([0-9]+)\\b", options: [.caseInsensitive])

    /// **The words the model does not read**, in the order of the text: under a model that reads no image (`capacity` 0),
    /// every `@name` the library knows (`known`: the subjects' keys, `NamedSubjects.key`); under any model, a reference to an
    /// image past the `carried` ones (the edit's images and the subjects' pictures). Another `@` is the model's own text.
    package static func inert(in text: String, known: Set<String>, capacity: Int, carried: Int) -> [Inert] {
        var found: [Inert] = []
        if capacity == 0 {
            for m in NamedSubjects.mentions(in: text) where known.contains(NamedSubjects.key(m.name)) {
                found.append(Inert(range: m.range, why: .subject))
            }
        }
        let all = NSRange(text.startIndex..., in: text)
        for match in imageReference.matches(in: text, range: all) {
            let group = match.range(at: 1).location != NSNotFound ? match.range(at: 1) : match.range(at: 2)
            guard let digits = Range(group, in: text), let n = Int(text[digits]), n >= 1, n > carried,
                  let range = Range(match.range, in: text), !found.contains(where: { $0.range.overlaps(range) }) else { continue }
            found.append(Inert(range: range, why: .image(n)))
        }
        return found.sorted { $0.range.lowerBound < $1.range.lowerBound }
    }

    /// **The text the model reads**: `text` without its inert words, each taking the space before it (or after it, at the
    /// start of a line). Nothing inert: `text` itself, to the byte.
    package static func sent(_ text: String, leaving inert: [Inert]) -> String {
        guard !inert.isEmpty else { return text }
        // In UTF-16 offsets, from the end: the indices of `text` are not read in a string being cut.
        let out = NSMutableString(string: text)
        for word in inert.sorted(by: { $0.range.lowerBound > $1.range.lowerBound }) {
            var r = NSRange(word.range, in: text)
            if r.location > 0, out.character(at: r.location - 1) == 0x20 {
                r.location -= 1; r.length += 1
            } else if r.location + r.length < out.length, out.character(at: r.location + r.length) == 0x20 {
                r.length += 1
            }
            out.deleteCharacters(in: r)
        }
        return out as String
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
