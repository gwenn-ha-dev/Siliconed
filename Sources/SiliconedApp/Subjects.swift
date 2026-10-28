// **Named subjects in the app — experimental, off by default**.
//
//     the Images box        a switch: Images (1 to 3 numbered images, an edit) or Subjects (a library
//        │                  of named pictures, as many as one wants) — one or the other, never both;
//        │                  a click on a subject cites it, a drop or « + » names a new one (`SubjectSheet.swift`)
//        ▼
//     the prompt            `@alice in @red-coat` — each cited subject drawn as a chip in the text
//        │                  (`PromptField.swift`), completed at `@`, an unknown one underlined; 3 at most
//        ▼
//     the click             each expanded prompt rewritten (`NamedSubjects.rewrite`): the model's prompt,
//                           hidden but shown on demand, and the cited subjects' framed pictures sent as
//                           `Request.references` — the engine is unchanged
//
// **One switch turns it on** (`SubjectLibrary.enabledKey`, « Experimental » in the settings window); off, the
// app is exactly the app without it: the prompt is a `TextEditor`, nothing is rewritten, no switch.
// Only Qwen-Image-2.1 shows it — the one visible family whose encoder reads images.
//
// **Incognito, like the history**: the subjects live in this process and die with it. A picture is
// read whole into memory when it is named (the file can go afterwards); nothing is written to disk —
// and the `subjects/` folder an earlier version kept in the library is deleted at launch.

import AppKit
import ImageIO
import Siliconed
import SwiftUI

/// What a subject's picture is made from: a file (read whole when the subject is named), or PNG bytes.
enum SubjectSource {
    case file(URL)
    case png(Data)

    /// The bytes, as given — never re-encoded.
    var data: Data? {
        switch self {
        case .file(let url): try? Data(contentsOf: url)
        case .png(let data): data
        }
    }
}

/// **The subjects, as the app holds them — in memory only**: the list, their pictures' bytes, their
/// thumbnails, the switch.
@MainActor @Observable
final class SubjectLibrary {
    /// **The switch** — a flag in the product, against the rule « one version lives, no flag in the product »:
    /// an exception wanted for an experimental feature, so that it ships hidden until its words have
    /// been judged by eye. One key; off by default. A preference, not the user's content.
    static let enabledKey = "namedSubjects"
    var enabled = UserDefaults.standard.bool(forKey: SubjectLibrary.enabledKey) {
        didSet {
            guard enabled != oldValue else { return }
            UserDefaults.standard.set(enabled, forKey: Self.enabledKey)
        }
    }

    private(set) var all: [NamedSubjects.Subject] = []
    /// Bumped at each change: the prompt field redraws its chips (a thumbnail, a name changed).
    private(set) var version = 0
    /// The pictures' bytes, by subject key and picture file (`Picture.file`: a name in memory, `1.jpg`).
    @ObservationIgnored private var bytes: [String: Data] = [:]
    /// The thumbnails, by subject, picture and frame — decoded once, from a 512-px copy.
    @ObservationIgnored private var thumbnails: [String: NSImage] = [:]

    init(library: Library) {
        // An earlier version kept the subjects on disk: incognito now, the folder goes.
        try? FileManager.default.removeItem(at: library.root.appendingPathComponent("subjects", isDirectory: true))
    }

    func subject(named name: String) -> NamedSubjects.Subject? {
        all.first { $0.id == NamedSubjects.key(name) }
    }

    private static func key(_ s: NamedSubjects.Subject, _ p: NamedSubjects.Picture) -> String { "\(s.id)/\(p.file)" }

    /// A picture's bytes, as given.
    func data(_ s: NamedSubjects.Subject, _ p: NamedSubjects.Picture) -> Data? { bytes[Self.key(s, p)] }

    /// A picture decoded upright, its long side at most `maxSide` (`nil`: full size).
    func upright(_ s: NamedSubjects.Subject, _ p: NamedSubjects.Picture, maxSide: Int? = nil) -> CGImage? {
        data(s, p).flatMap { SubjectPictures.upright($0, maxSide: maxSide) }
    }

    /// **A picture's thumbnail, framed** — what the chips, the prompt and the sheet show.
    func thumbnail(_ s: NamedSubjects.Subject, picture: Int = 0) -> NSImage? {
        guard s.pictures.indices.contains(picture) else { return nil }
        let p = s.pictures[picture]
        let key = "\(Self.key(s, p))/\(p.frame.x),\(p.frame.y),\(p.frame.width),\(p.frame.height)"
        if let known = thumbnails[key] { return known }
        guard let upright = upright(s, p, maxSide: 512), let cropped = SubjectPictures.crop(upright, p.frame) else { return nil }
        let image = NSImage(cgImage: FinishedImage.reduce(cropped, side: 160) ?? cropped, size: .zero)
        thumbnails[key] = image
        return image
    }

    /// **The references a rewrite places**, in its order: each picture decoded from its bytes, cropped
    /// to its frame. Never dropped, even undecodable: the prompt's numbers count every picture — the
    /// render then fails naming it.
    func references(_ r: NamedSubjects.Rewritten) -> [ReferenceImage] {
        r.placed.map { placed in
            let picture = placed.subject.pictures.firstIndex(of: placed.picture) ?? 0
            let image = upright(placed.subject, placed.picture).flatMap { SubjectPictures.crop($0, placed.picture.frame) }
            return ReferenceImage(source: .subject(image, name: "@" + placed.subject.name),
                                  vignette: thumbnail(placed.subject, picture: picture) ?? NSImage(),
                                  name: "@" + placed.subject.name, width: image?.width ?? 1024, height: image?.height ?? 1024)
        }
    }

    // ── the user's gestures ──

    func create(name: String, kind: NamedSubjects.Kind,
                pictures: [(source: SubjectSource, frame: NamedSubjects.Frame)]) -> Problem? {
        if let p = NamedSubjects.check(name: name, taken: all.map(\.name)) { return Problem(p.sentence) }
        guard (1...NamedSubjects.maxPictures).contains(pictures.count) else {
            return Problem(String(localized: "A subject has one or two pictures."))
        }
        var s = NamedSubjects.Subject(name: name, kind: kind, pictures: [])
        var read: [String: Data] = [:]
        for (i, p) in pictures.enumerated() {
            guard let data = p.source.data else { return Problem(SubjectSource.unreadable(p.source)) }
            let picture = NamedSubjects.Picture(file: "\(i + 1)", frame: p.frame.clamped())
            s.pictures.append(picture)
            read[Self.key(s, picture)] = data
        }
        bytes.merge(read) { $1 }
        all.append(s)
        changed()
        return nil
    }

    /// **Changes a subject**: its name, its kind, its frames, its pictures — `added` appended (two at
    /// most), the pictures `subject` no longer lists forgotten.
    func update(_ old: String, to subject: NamedSubjects.Subject,
                adding added: [(source: SubjectSource, frame: NamedSubjects.Frame)]) -> Problem? {
        guard let i = all.firstIndex(where: { $0.id == NamedSubjects.key(old) }) else {
            return Problem(String(localized: "No subject is called @\(old)."))
        }
        let others = all.map(\.name).filter { NamedSubjects.key($0) != NamedSubjects.key(old) }
        if let p = NamedSubjects.check(name: subject.name, taken: others) { return Problem(p.sentence) }
        guard (1...NamedSubjects.maxPictures).contains(subject.pictures.count + added.count) else {
            return Problem(String(localized: "A subject has one or two pictures."))
        }
        let before = all[i]
        var s = subject
        var kept: [String: Data] = [:]
        s.pictures = s.pictures.map { p in
            kept[Self.key(s, NamedSubjects.Picture(file: p.file, frame: p.frame))] = data(before, p)
            return NamedSubjects.Picture(file: p.file, frame: p.frame.clamped())
        }
        for p in added {
            guard let data = p.source.data else { return Problem(SubjectSource.unreadable(p.source)) }
            let used = Set(s.pictures.map(\.file))
            let file = (1...9).map(String.init).first { !used.contains($0) } ?? UUID().uuidString
            let picture = NamedSubjects.Picture(file: file, frame: p.frame.clamped())
            s.pictures.append(picture)
            kept[Self.key(s, picture)] = data
        }
        forget(before)
        bytes.merge(kept) { $1 }
        all[i] = s
        changed()
        return nil
    }

    func delete(_ s: NamedSubjects.Subject) {
        forget(s)
        all.removeAll { $0.id == s.id }
        changed()
    }

    private func forget(_ s: NamedSubjects.Subject) {
        for p in s.pictures { bytes[Self.key(s, p)] = nil }
    }

    private func changed() {
        all.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        thumbnails = [:]
        version += 1
    }
}

extension SubjectSource {
    /// Why a picture could not be read, naming its file without the home folder.
    static func unreadable(_ source: SubjectSource) -> String {
        switch source {
        case .file(let url): String(localized: "This image can't be read: \(Library.withoutHome(url.path))")
        case .png: String(localized: "This image can't be read")
        }
    }
}

/// **Reading a picture upright**, whatever its EXIF orientation — the frame is in that space.
enum SubjectPictures {
    /// Decoded upright, its long side at most `maxSide` (`nil`: full size).
    static func upright(_ url: URL, maxSide: Int? = nil) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return upright(source, maxSide: maxSide, url: url)
    }

    static func upright(_ data: Data, maxSide: Int? = nil) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return upright(source, maxSide: maxSide, url: nil)
    }

    private static func upright(_ source: CGImageSource, maxSide: Int?, url: URL?) -> CGImage? {
        var side = maxSide
        if side == nil, let url, let s = ReferenceImage.size(of: url) { side = max(s.width, s.height) }
        if side == nil, let p = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let w = p[kCGImagePropertyPixelWidth] as? Int, let h = p[kCGImagePropertyPixelHeight] as? Int {
            side = max(w, h)
        }
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                        kCGImageSourceThumbnailMaxPixelSize: side ?? 4096,
                                        kCGImageSourceCreateThumbnailWithTransform: true]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    /// The frame's pixels of an upright image (`Frame.pixels`: its ends rounded).
    static func crop(_ image: CGImage, _ frame: NamedSubjects.Frame) -> CGImage? {
        let p = frame.pixels(width: image.width, height: image.height)
        return image.cropping(to: CGRect(x: p.x, y: p.y, width: p.width, height: p.height))
    }
}

// MARK: - The kinds: their words and their colors

extension NamedSubjects.Kind {
    /// The kind's name, in the user's language.
    var label: String {
        switch self {
        case .person: String(localized: "Person", comment: "Kind of named subject")
        case .garment: String(localized: "Garment", comment: "Kind of named subject")
        case .object: String(localized: "Object", comment: "Kind of named subject")
        case .place: String(localized: "Place", comment: "Kind of named subject")
        case .style: String(localized: "Style", comment: "Kind of named subject")
        }
    }

    /// What the model takes from the picture — the instruction `NamedSubjects.Phrasing` writes, said
    /// to the user.
    var taken: String {
        switch self {
        case .person: String(localized: "Their face, hair and build — not the clothes or the background.")
        case .garment: String(localized: "The garment alone — not who wears it.")
        case .object: String(localized: "The object alone, its exact design.")
        case .place: String(localized: "The setting and its light — not its people.")
        case .style: String(localized: "The look: medium, palette, lighting — not the content.")
        }
    }

    var symbol: String {
        switch self {
        case .person: "person.fill"
        case .garment: "tshirt.fill"
        case .object: "cube.fill"
        case .place: "mountain.2.fill"
        case .style: "paintpalette.fill"
        }
    }

    /// **One color for every kind: the image wire's lavender** (`Palette.lavender`) — a subject is an
    /// image to the model. A hue per kind put rose and olive in the prompt, which read as an error and
    /// a warning; the kind is said by its words (`label`, `taken`) and its symbol.
    var nsColor: NSColor { Self.dynamic(dark: 0xC8AEF8, light: 0x8668B6) }
    var color: Color { Color(nsColor: nsColor) }

    private static func dynamic(dark: Int, light: Int) -> NSColor {
        func color(_ v: Int) -> NSColor {
            NSColor(srgbRed: CGFloat(v >> 16 & 0xFF) / 255, green: CGFloat(v >> 8 & 0xFF) / 255, blue: CGFloat(v & 0xFF) / 255, alpha: 1)
        }
        return NSColor(name: nil) { a in a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? color(dark) : color(light) }
    }
}

// MARK: - The refusals, in the user's language

extension Problem {
    init(_ f: NamedSubjects.Failure) {
        switch f {
        case .unknownSubject(let name):
            self.init(String(localized: "No subject is called @\(name)."),
                      suggestion: String(localized: "Name it first: drop an image under Subjects, in the Images box."))
        case let .tooManyImages(count, max):
            self.init(String(localized: "This prompt takes \(String(count)) images: a render takes \(String(max)) at most."),
                      suggestion: String(localized: "A subject with two pictures counts twice."))
        case .modelReadsNoImages:
            self.init(String(localized: "This model doesn't read images: its prompt can't cite subjects."))
        case .noPicture(let name):
            self.init(String(localized: "The subject @\(name) has no picture."))
        }
    }
}

extension NamedSubjects.NameProblem {
    var sentence: String {
        switch self {
        case .empty: String(localized: "Give it a name.")
        case .tooLong: String(localized: "A name has \(NamedSubjects.maxNameLength) characters at most.")
        case .characters: String(localized: "Letters, digits, - and _ only, without spaces.")
        case .taken: String(localized: "A subject already has this name.")
        }
    }
}

// MARK: - The rack's and the render's side

extension AppState {
    /// **The Images box offers its Subjects mode**: the switch is on and the chosen model's encoder
    /// reads images (Qwen-Image-2.1).
    var subjectsAvailable: Bool { subjects.enabled && promptNamesImages && maxReferences > 0 }
    /// **The subjects are shown, and `@names` are read**: the box is in its Subjects mode.
    var subjectsActive: Bool { subjectsAvailable && referenceMode == .subjects }

    /// **One expanded prompt, rewritten** — `nil` when the feature is off or the prompt cites nobody.
    /// For the rack's model and edit images by default; the remote control passes its request's — a
    /// `model` it holds, or in a mirror (no model chain here) the `capacity` the other Mac published.
    func rewrite(_ expanded: String, model: Model? = nil, capacity given: Int? = nil,
                 editImages: Int? = nil) -> Result<NamedSubjects.Rewritten, NamedSubjects.Failure>? {
        // The rack reads `@names` in its Subjects mode only; the remote control, which brings its own
        // images, whenever the switch is on — and, like the rack, only for a model that reads images:
        // elsewhere `@` is the model's own (Anima's card writes an artist `@incase`).
        guard model == nil && given == nil ? subjectsActive : subjects.enabled, !NamedSubjects.mentions(in: expanded).isEmpty else { return nil }
        let capacity = given ?? model.map { $0.chain.text.output.readsImages ? $0.chain.denoising.maxReferences : 0 }
            ?? (promptNamesImages ? maxReferences : 0)
        guard capacity > 0 else { return nil }
        let first = (editImages ?? references.count) + 1
        return Result { () throws(NamedSubjects.Failure) in
            try NamedSubjects.rewrite(expanded, subjects: subjects.all, capacity: capacity, first: first)
        }
    }

    /// **The rewrite the rack shows**: the first expanded prompt's (a series cites the same subjects
    /// unless a group holds one — each job is rewritten on its own at the click).
    var shownRewrite: Result<NamedSubjects.Rewritten, NamedSubjects.Failure>? {
        rewrite((try? alternatives.get().prompts.first) ?? prompt)
    }

    /// What the subjects refuse, said like the rest of the form — the first refusal of any expansion.
    var subjectsBlocker: String? {
        guard subjectsActive, let prompts = try? alternatives.get().prompts else { return nil }
        for p in prompts {
            if case .failure(let f)? = rewrite(p) { return Problem(f).title }
        }
        return nil
    }

    /// The images the subjects add to the next render (the first expansion's): what the cost, the
    /// plan and the learned speed count with the edit's own.
    var subjectImageCount: Int {
        if case .success(let r)? = shownRewrite { return r.placed.count }
        return 0
    }

    /// **The job's prompt and references for one expanded prompt**: rewritten when it cites subjects,
    /// the subjects' framed pictures after the edit's images.
    /// In a mirror the job goes with the rewrite and the pictures (`RemoteJob.modelPrompt`): the subjects stay here.
    func namedJob(_ expanded: String, settings: inout RenderSettings, references edit: [ReferenceImage],
                  model: Model? = nil, capacity: Int? = nil) -> [ReferenceImage] {
        guard case .success(let r)? = rewrite(expanded, model: model, capacity: capacity, editImages: edit.count),
              !r.placed.isEmpty else { return edit }
        let placed = subjects.references(r)
        settings.modelPrompt = r.prompt
        settings.subjectImages = placed.count
        settings.references = edit.count + placed.count
        return edit + placed
    }

    /// **`@name` inserted where the prompt's caret is** (a click on a chip) — appended when the field
    /// is not on screen. A space before it when it would glue to a word.
    func cite(_ s: NamedSubjects.Subject) {
        if let field = PromptTextView.current, field.window != nil {
            field.insertMention(s.name)
            return
        }
        let glue = prompt.last.map { !$0.isWhitespace } ?? false
        prompt += (glue ? " " : "") + "@\(s.name) "
    }

    // ── the sheet ──

    /// The open panel, then the sheet: an image to name.
    func chooseSubjectImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.prompt = String(localized: "Choose")
        panel.message = String(localized: "An image to name: a person, a garment, an object, a place or a style.")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        nameSubject(.file(url))
    }

    /// A file or PNG bytes dropped on the subjects shelf: the sheet opens on it.
    func nameSubject(_ source: SubjectSource) {
        guard let draft = SubjectDraft(new: source, taken: subjects.all.map(\.name)) else {
            flash(String(localized: "This image can't be read"))
            return
        }
        subjectDraft = draft
    }

    func editSubject(_ s: NamedSubjects.Subject) {
        subjectDraft = SubjectDraft(editing: s, library: subjects)
    }

    func deleteSubject(_ s: NamedSubjects.Subject) {
        subjects.delete(s)
        flash(String(localized: "@\(s.name) deleted"))
    }

    /// Asked first: prompts that cite it are refused afterwards.
    func confirmDeletion(_ s: NamedSubjects.Subject) {
        let alert = NSAlert()
        alert.messageText = String(localized: "Delete @\(s.name)?")
        alert.informativeText = String(localized: "Prompts that cite it will be refused.")
        alert.addButton(withTitle: String(localized: "Delete"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.buttons[0].hasDestructiveAction = true
        if alert.runModal() == .alertFirstButtonReturn { deleteSubject(s) }
    }

    /// A subject renamed: the prompt's mentions of it follow.
    func renameMentions(_ old: String, to new: String) {
        var text = prompt
        for m in NamedSubjects.mentions(in: prompt).reversed() where NamedSubjects.key(m.name) == NamedSubjects.key(old) {
            if let r = Range(NSRange(m.range, in: prompt), in: text) { text.replaceSubrange(r, with: "@" + new) }
        }
        prompt = text
    }
}
