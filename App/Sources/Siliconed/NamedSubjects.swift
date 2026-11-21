import Foundation

/// **Named subjects: `@alice in @red-coat on @the-beach` is a prompt and three reference images.**
///
///     the user's prompt  "@alice in @red-coat on @the-beach"           the library: name · kind · picture(s) + frame
///        │ mentions: `@` + letters, digits, - _ (case-insensitive)       alice    person   alice.jpg [0.2 0 0.5 0.9]
///        ▼                                                               red-coat garment coat.png  [whole]
///     subjects in order of first appearance, each its pictures in order  the-beach place   beach.heic, beach2.jpg
///        │ numbered from `first` (after an edit's own images)
///        ▼
///     model prompt   "the person in <image1> in the garment in <image2> on the same place in <image3> and <image4>
///
///                     Take only their identity from <image1>: face, hair and build.
///                     Take only the garment from <image2>: …
///                     Use <image3> and <image4> as the setting: …"   ← refused: 4 > 3 images
///     images         [alice.jpg ▭, coat.png ▭, beach.heic ▭, …]   each cropped to its frame, in that order
///
/// The engine does not change: the images are `Request.references`, which
/// Qwen-Image-2.1's encoder reads as `<image1>`, `<image2>`… in front of the prompt
/// (`QwenImage21Text.template`), and the render starts from fresh noise at the chosen format. What
/// this file does is only words: **the text designates, the image describes**. The kind's noun and its
/// retention instruction come from the kind, the number from the order; the name is the user's alone.
///
/// What sets it apart, and what a port of ComfyUI's or MiniMax's "named references" by analogy would miss:
///
///   - **The name never reaches the model.** `@alice` becomes "the person in <image1>": the kind and the
///     `<imageN>` label the template already puts in front of each image (Qwen's official edit-prompt
///     rewriter writes it so). A name is a word, and the model reads words as content: "redraw Alice"
///     on a drawing of a woman gave Alice in Wonderland; `@neo` or `@paris` would leak the same
///     way. A legend after the prompt says what to take from each image. An article the user wrote
///     before the mention is not doubled ("in the @room" → "in the place in <image2>").
///   - **One subject, one or two places.** A subject with two pictures takes two consecutive numbers
///     ("the same person in <image1> and <image2>"); citing it twice does not place it twice.
///   - **Numbered by first appearance**, after the edit's own images when there are some
///     (`first`): an edit's image 1 stays image 1.
///   - **Three images at most per render**, whatever the model reads (it reads four in its card): the
///     product's measured worst case is 1024×1536 with three references (measured on the M1 Pro: 4.76 GB, 0 swap).
///   - **The kind is a closed list**: each instruction is written, and judged by eye, once.
///   - **An unknown `@name` is refused**, never passed through as text: a typo would otherwise render
///     without its picture and nobody would see why.
///
/// Every word the model reads is in `Phrasing`: the reference form (`<image1>`, Qwen's own convention), the instructions, where the legend goes.
package enum NamedSubjects {

    /// **The most images one render takes**, the edit's own included.
    package static let maxImages = 3
    /// A subject's pictures: one, or two of the same subject.
    package static let maxPictures = 2
    /// A name's length, in characters.
    package static let maxNameLength = 32

    // ── the kinds ──

    /// **What a subject is, hence what the model keeps of its picture** — a closed list.
    package enum Kind: String, CaseIterable, Codable, Sendable {
        case person, garment, object, place, style
    }

    // ── a subject ──

    /// **A frame on a picture**, in fractions of the upright image (EXIF orientation applied): a
    /// rectangle, never a mask (by design: no masks in the app). The frame is kept, not a cropped copy: the
    /// picture can be reframed later.
    package struct Frame: Codable, Equatable, Hashable, Sendable {
        package var x, y, width, height: Double

        package init(x: Double, y: Double, width: Double, height: Double) {
            self.x = x; self.y = y; self.width = width; self.height = height
        }

        package static let whole = Frame(x: 0, y: 0, width: 1, height: 1)

        /// Inside the image, at least `minimum` of each side — what a hand-edited JSON or a drag past
        /// the edge leaves usable.
        package func clamped(minimum: Double = 0.02) -> Frame {
            let w = min(max(width.isFinite ? width : 1, minimum), 1)
            let h = min(max(height.isFinite ? height : 1, minimum), 1)
            return Frame(x: min(max(x.isFinite ? x : 0, 0), 1 - w), y: min(max(y.isFinite ? y : 0, 0), 1 - h),
                         width: w, height: h)
        }

        /// **The pixels it covers** on an upright `width × height` image: its ends rounded
        /// (invariant 4, `first · n / slices` — never `size × index`), at least one pixel each way.
        package func pixels(width W: Int, height H: Int) -> (x: Int, y: Int, width: Int, height: Int) {
            let f = clamped()
            func span(_ start: Double, _ length: Double, _ n: Int) -> (Int, Int) {
                let a = min(max(Int((start * Double(n)).rounded()), 0), max(n - 1, 0))
                let b = min(max(Int(((start + length) * Double(n)).rounded()), a + 1), n)
                return (a, b - a)
            }
            let (px, pw) = span(f.x, f.width, W), (py, ph) = span(f.y, f.height, H)
            return (px, py, pw, ph)
        }
    }

    /// One picture of a subject: its file (inside the subject's folder) and its frame.
    package struct Picture: Codable, Equatable, Sendable {
        package var file: String
        package var frame: Frame
        package init(file: String, frame: Frame = .whole) { self.file = file; self.frame = frame }
    }

    /// **A subject of the library**: its name (without `@`), its kind, one or two pictures.
    package struct Subject: Codable, Equatable, Sendable, Identifiable {
        package var name: String
        package var kind: Kind
        package var pictures: [Picture]
        package var id: String { NamedSubjects.key(name) }

        package init(name: String, kind: Kind, pictures: [Picture]) {
            self.name = name; self.kind = kind; self.pictures = pictures
        }
    }

    // ── names and mentions ──

    /// Two names are the same subject when they are equal ignoring case: `@Alice` cites `alice`.
    package static func key(_ name: String) -> String { name.lowercased() }

    private static func isNameCharacter(_ c: Character) -> Bool { c.isLetter || c.isNumber || c == "-" || c == "_" }
    private static func isJoiner(_ c: Character) -> Bool { c == "-" || c == "_" }

    /// What a name typed at creation can be refused for.
    package enum NameProblem: Equatable, Sendable {
        case empty, tooLong, characters, taken
    }

    /// **A name a mention can cite whole**: 1 to `maxNameLength` letters, digits, `-` or `_`, not
    /// starting or ending with `-` `_` (a mention would stop before them), and not one of `taken`
    /// (ignoring case).
    package static func check(name: String, taken: [String] = []) -> NameProblem? {
        if name.isEmpty { return .empty }
        if name.count > maxNameLength { return .tooLong }
        guard name.allSatisfy(isNameCharacter), let a = name.first, let z = name.last,
              !isJoiner(a), !isJoiner(z) else { return .characters }
        if taken.contains(where: { key($0) == key(name) }) { return .taken }
        return nil
    }

    /// A name made of what was typed or of a file's name: spaces and other characters become `-`,
    /// cut at `maxNameLength` — what the creation sheet proposes, never imposed.
    package static func suggestedName(_ text: String) -> String {
        var s = ""
        for c in text {
            if isNameCharacter(c) { s.append(c) } else if !(s.last.map(isJoiner) ?? true) { s.append("-") }
        }
        while let z = s.last, isJoiner(z) { s.removeLast() }
        while let a = s.first, isJoiner(a) { s.removeFirst() }
        return String(s.prefix(maxNameLength)).trimmingCharacters(in: CharacterSet(charactersIn: "-_"))
    }

    /// **A `@name` in a prompt**: the name, and the range of `@name` in the prompt.
    package struct Mention: Equatable, Sendable {
        package let name: String
        package let range: Range<String.Index>
    }

    /// **The mentions of a prompt, in reading order.** A mention is `@` followed by name characters
    /// (letters, digits, `-`, `_`), its trailing `-` `_` left out (`@alice_` is `alice` then `_`).
    /// An `@` after a letter or a digit is not one (`me@home.org`), nor an `@` followed by nothing
    /// a name has.
    package static func mentions(in prompt: String) -> [Mention] {
        var found: [Mention] = []
        var i = prompt.startIndex
        var previous: Character?
        while i < prompt.endIndex {
            let c = prompt[i]
            if c == "@", !(previous.map { $0.isLetter || $0.isNumber || $0 == "@" } ?? false) {
                var end = prompt.index(after: i)
                var lastSolid: String.Index?
                while end < prompt.endIndex, isNameCharacter(prompt[end]) {
                    if !isJoiner(prompt[end]) { lastSolid = end }
                    end = prompt.index(after: end)
                }
                if let lastSolid {
                    let stop = prompt.index(after: lastSolid)
                    let start = prompt.index(after: i)
                    found.append(Mention(name: String(prompt[start..<stop]), range: i..<stop))
                    previous = prompt[lastSolid]
                    i = stop
                    continue
                }
            }
            previous = c
            i = prompt.index(after: i)
        }
        return found
    }

    // ── the rewrite ──

    /// **What refuses a prompt before any render** — each said in the user's language by the app.
    package enum Failure: Error, Equatable, Sendable, CustomStringConvertible {
        /// `@name` names no subject of the library.
        case unknownSubject(name: String)
        /// The images the render would take — the edit's and the subjects' — exceed the maximum.
        case tooManyImages(count: Int, max: Int)
        /// The model's encoder does not read images: its prompt cannot name them.
        case modelReadsNoImages
        /// A subject without a picture (a hand-edited folder).
        case noPicture(name: String)

        package var description: String {
            switch self {
            case .unknownSubject(let n): "no subject is called @\(n)"
            case let .tooManyImages(n, max): "\(n) images: a render takes \(max) at most"
            case .modelReadsNoImages: "this model does not read images: its prompt cannot cite subjects"
            case .noPicture(let n): "the subject @\(n) has no picture"
            }
        }
    }

    /// One image of the render: whose, which picture, and the number the prompt gives it.
    package struct Placed: Equatable, Sendable {
        package let subject: Subject
        package let picture: Picture
        package let number: Int
    }

    /// **The prompt the model reads, and its images in order.** `placed` is empty when the prompt
    /// cites nobody — then `prompt` is the user's, character for character.
    package struct Rewritten: Equatable, Sendable {
        package let prompt: String
        package let placed: [Placed]
        /// The subjects cited, in order of first appearance.
        package let cited: [Subject]
    }

    /// **Rewrites `prompt`** for a model whose encoder reads `capacity` images (0: none), after
    /// `first − 1` images already placed (an edit's). `capacity` is capped at `maxImages`.
    package static func rewrite(_ prompt: String, subjects: [Subject], capacity: Int,
                                first: Int = 1) throws(Failure) -> Rewritten {
        let found = mentions(in: prompt)
        guard !found.isEmpty else { return Rewritten(prompt: prompt, placed: [], cited: []) }
        let library = Dictionary(subjects.map { (key($0.name), $0) }, uniquingKeysWith: { a, _ in a })
        var cited: [Subject] = []
        for m in found {
            guard let s = library[key(m.name)] else { throw .unknownSubject(name: m.name) }
            if !cited.contains(where: { key($0.name) == key(s.name) }) { cited.append(s) }
        }
        guard capacity > 0 else { throw .modelReadsNoImages }
        if let bare = cited.first(where: { $0.pictures.isEmpty }) { throw .noPicture(name: bare.name) }
        let limit = min(capacity, maxImages)
        let count = (first - 1) + cited.reduce(0) { $0 + min($1.pictures.count, maxPictures) }
        guard count <= limit else { throw .tooManyImages(count: count, max: limit) }

        var placed: [Placed] = []
        var number = first
        var sentences: [String] = []
        var images: [String: [Int]] = [:]
        for s in cited {
            let numbers = s.pictures.prefix(maxPictures).map { picture -> Int in
                placed.append(Placed(subject: s, picture: picture, number: number))
                number += 1
                return number - 1
            }
            images[key(s.name)] = numbers
            sentences.append(Phrasing.retention(s.kind, from: Phrasing.references(numbers)))
        }
        // The body: each `@name` becomes what it is and where — "the person in <image1>". **The name
        // never reaches the model**: it is the user's label, and the model reads a word as content —
        // "redraw Alice" on a drawing of a woman gave Alice in Wonderland.
        var body = ""
        var i = prompt.startIndex
        for m in found {
            body += prompt[i..<m.range.lowerBound]
            let s = library[key(m.name)]!
            body += Phrasing.mention(s.kind, images[key(s.name)]!, after: body)
            i = m.range.upperBound
        }
        body += prompt[i...]
        return Rewritten(prompt: Phrasing.compose(body: body, legend: sentences), placed: placed, cited: cited)
    }

    // ── the image's metadata ──

    /// The PNG key that keeps the model's prompt, beside `prompt`.
    package static let modelPromptKey = "model_prompt"

    /// **A render's metadata, the prompt the user typed in front** (`Engine.Render.metadata`): the
    /// engine read the rewrite, but what « Reuse » and a PNG dropped on the canvas give back
    /// (`Engine.Render.Recipe.prompt`) is the prompt with its `@names`; the model's prompt is kept in
    /// addition, under its own key. `modelPrompt` `nil` (nobody cited): `metadata` unchanged.
    /// Civitai's `parameters` (`Infotext`) shows the typed prompt too: it opens with the engine's.
    package static func metadata(_ metadata: [String: String], typed: String, modelPrompt: String?) -> [String: String] {
        guard let modelPrompt else { return metadata }
        var m = metadata
        if let parameters = m["parameters"], let engine = m["prompt"], parameters.hasPrefix(engine) {
            m["parameters"] = typed + parameters.dropFirst(engine.count)
        }
        m["prompt"] = typed
        m[modelPromptKey] = modelPrompt
        return m
    }

    // ── the words ──

    /// **Every word the model reads.** Qwen's own convention: its
    /// official edit-prompt rewriter (`QwenLM/Qwen-Image-2.1@1993c3a4`,
    /// `prompt_rewrite/prompts/system_prompt_edit.txt:59`) MUST write `<image1>`, `<image2>`… literally — the very label the template puts in front of each
    /// image, plain text and not a special token — and bans "the first image", "图1", "image A". Its
    /// instructions are affirmative (`:190`) and say what is taken from each image (`:61`). Each decision
    /// is one line here.
    enum Phrasing {
        /// The candidate forms: `<image1>` (Qwen's rule), `image 1` (plain prose), `Picture 1` (Qwen-Image-Edit-2509's template).
        enum Reference { case tag, natural, picture }
        /// **The form in use.**
        static let form = Reference.tag

        /// How the prompt refers to one image.
        static func reference(_ n: Int) -> String {
            switch form {
            case .tag: "<image\(n)>"
            case .natural: "image \(n)"
            case .picture: "Picture \(n)"
            }
        }

        /// The images of one subject, each named (as Qwen's rewriter does: never compressed into a range).
        static func references(_ numbers: [Int]) -> String {
            let each = numbers.map(reference)
            guard let last = each.last, each.count > 1 else { return each.first ?? "" }
            return each.dropLast().joined(separator: ", ") + " and " + last
        }

        /// The kind's noun: "the **person** in <image1>".
        static func noun(_ kind: Kind) -> String {
            switch kind {
            case .person: "person"
            case .garment: "garment"
            case .object: "object"
            case .place: "place"
            case .style: "style"
            }
        }

        /// What the model takes from the picture, said affirmatively (as Qwen's rewriter does; the « not … »
        /// variants are there if the affirmative leaks). Against the model's habit, a person's
        /// identity in Qwen's examples includes the clothes (`edit_example.jsonl:4`): "face, hair and
        /// build" alone is the first thing to judge by eye.
        static func retention(_ kind: Kind, from: String) -> String {
            switch kind {
            case .person: "Take only their identity from \(from): face, hair and build."
            case .garment: "Take only the garment from \(from): its cut, color, material and details, worn by the subject."
            case .object: "Take only the object from \(from): its exact design, markings and count."
            case .place: "Use \(from) as the setting: its scenery, architecture and light."
            case .style: "Render the whole image in the medium, palette, lighting and texture of \(from)."
            }
        }

        /// **What a mention becomes in the prompt**: "the person in <image1>", "the same person in <image1>
        /// and <image2>", "the style of <image3>". `after`, the text before it: an article the user wrote
        /// ("in the @room") is not doubled, nor "style of" ("in the style of @ink").
        static func mention(_ kind: Kind, _ numbers: [Int], after: String) -> String {
            let from = references(numbers)
            let tail = after.lowercased()
            if kind == .style {
                return tail.hasSuffix("style of ") ? from : "the style of \(from)"
            }
            let article = [" the ", " a ", " an "].contains { (" " + tail).hasSuffix($0) } ? "" : "the "
            return "\(article)\(numbers.count > 1 ? "same " : "")\(noun(kind)) in \(from)"
        }

        /// Where the legend goes: after the prompt, one sentence per subject, a blank line between.
        static func compose(body: String, legend: [String]) -> String {
            let text = body.trimmingCharacters(in: .whitespacesAndNewlines)
            return ([text] + [legend.joined(separator: "\n")]).filter { !$0.isEmpty }.joined(separator: "\n\n")
        }
    }
}
