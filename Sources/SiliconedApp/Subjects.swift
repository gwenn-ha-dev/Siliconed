// **Named subjects in the Prompt Lab** (`Lab.swift`): the library of named pictures.
//
//     the lab's Images zone   a library of named pictures, as many as one wants: a drop, « + » or « Name… » on an
//        │                    image names a new one (`SubjectSheet.swift`); each is a tile (`LabImages.swift`)
//        ▼
//     the lab's text          `@alice in @red-coat` — each cited subject drawn as a chip in the text
//        │                    (`PromptField.swift`), completed at `@`, an unknown one underlined; 3 images at most
//        ▼
//     « Carry to the Rack »   the text translated (`NamedSubjects.rewrite`, through `LabInputs.carry`): the model's
//                             prompt goes to the rack's Prompt, the cited subjects' framed pictures to its Images
//                             box, numbered — the rack knows nothing of subjects, and the engine is unchanged
//
// Only Qwen-Image-2.1 reads them — the one visible family whose encoder reads images; for the others `@` stays the
// model's own character (Anima's artists). `silicontrol subjects` and `silicontrol add "@name"` act on the same
// library and translate at the `add` (`namedJob`).
//
// **Incognito, like the history**: the subjects live in this process and die with it. A picture is
// read whole into memory when it is named (the file can go afterwards); nothing is written to disk —
// and the `subjects/` folder an earlier version kept in the library is deleted at launch, as is the switch
// that turned the subjects on (they are no longer a beta setting).

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
/// thumbnails.
@MainActor @Observable
final class SubjectLibrary {
    /// The retired beta switch (« Named Subjects », 0.17): deleted from the preferences at launch.
    private static let retiredSwitch = "namedSubjects"

    private(set) var all: [NamedSubjects.Subject] = []
    /// Bumped at each change: the lab's text redraws its chips (a thumbnail, a name changed).
    private(set) var version = 0
    /// The pictures' bytes, by subject key and picture file (`Picture.file`: a name in memory, `1.jpg`).
    @ObservationIgnored private var bytes: [String: Data] = [:]
    /// The thumbnails, by subject, picture and frame — decoded once, from a 512-px copy.
    @ObservationIgnored private var thumbnails: [String: NSImage] = [:]
    @ObservationIgnored private var sizes: [String: (width: Int, height: Int)] = [:]

    init(library: Library) {
        // An earlier version kept the subjects on disk: incognito now, the folder goes — and the switch, no more a setting.
        UserDefaults.standard.removeObject(forKey: Self.retiredSwitch)
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

    /// **A picture's size inside its frame**, upright, in pixels — read from the header, without decoding the picture:
    /// what the Prompt Lab shows of the format an image 1 gives.
    func framedSize(_ s: NamedSubjects.Subject, _ p: NamedSubjects.Picture) -> (width: Int, height: Int)? {
        let key = "\(Self.key(s, p))/\(p.frame.x),\(p.frame.y),\(p.frame.width),\(p.frame.height)"
        if let known = sizes[key] { return known }
        guard let data = data(s, p), let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = properties[kCGImagePropertyPixelWidth] as? Int, let h = properties[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        // Orientations 5 to 8 turn the picture on its side: the frame is in the upright space.
        let turned = (properties[kCGImagePropertyOrientation] as? Int).map { (5...8).contains($0) } ?? false
        let px = p.frame.pixels(width: turned ? h : w, height: turned ? w : h)
        sizes[key] = (px.width, px.height)
        return sizes[key]
    }

    /// **The references a rewrite places**, in its order: each picture decoded from its bytes, cropped
    /// to its frame. Never dropped, even undecodable: the prompt's numbers count every picture — the
    /// render then fails naming it.
    func references(_ placed: [NamedSubjects.Placed]) -> [ReferenceImage] {
        placed.map { placed in
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
        sizes = [:]
        version += 1
    }
}

extension SubjectSource {
    /// Why a picture could not be read, naming its file without the home folder.
    static func unreadable(_ source: SubjectSource) -> String {
        switch source {
        case .file(let url): String(localized: "This image can’t be read: \(Library.withoutHome(url.path))")
        case .png: String(localized: "This image can’t be read")
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
                      suggestion: String(localized: "Name it first: in the Prompt Lab, drop an image and choose Name…"))
        case let .tooManyImages(count, max):
            self.init(String(localized: "This prompt takes \(String(count)) images: a render takes \(String(max)) at most."),
                      suggestion: String(localized: "A subject with two pictures counts twice."))
        case .modelReadsNoImages:
            self.init(String(localized: "This model doesn’t read images: its prompt can’t cite subjects."))
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

// MARK: - The remote control's side

extension AppState {
    /// **One expanded prompt, rewritten** for `silicontrol add` — `nil` when the prompt cites nobody. For a `model`
    /// it holds, or in a mirror (no model chain here) the `capacity` the other Mac published; `editImages` are the
    /// images the request brings itself, the subjects taking the places after them. The rack itself rewrites nothing:
    /// the lab translates, and the rack receives the translation.
    func rewrite(_ expanded: String, model: Model? = nil, capacity given: Int? = nil,
                 editImages: Int? = nil) -> Result<NamedSubjects.Rewritten, NamedSubjects.Failure>? {
        guard !NamedSubjects.mentions(in: expanded).isEmpty else { return nil }
        // Only for a model that reads images: elsewhere `@` is the model's own (Anima's card writes an artist `@incase`).
        guard let capacity = given ?? model.map({ $0.chain.text.output.readsImages ? $0.chain.denoising.maxReferences : 0 }),
              capacity > 0 else { return nil }
        let first = (editImages ?? references.count) + 1
        return Result { () throws(NamedSubjects.Failure) in
            try NamedSubjects.rewrite(expanded, subjects: subjects.all, capacity: capacity, first: first)
        }
    }

    /// **The job's prompt and references for one expanded prompt**: rewritten when it cites subjects,
    /// the subjects' framed pictures after the edit's images — and without what the field greys (`LabInputs.inert`): a
    /// subject under a model that reads no image, a reference to an image the render does not carry. The text kept as
    /// typed, the model's prompt goes beside it (`modelPrompt`) whenever it differs.
    /// In a mirror the job goes with the rewrite and the pictures (`RemoteJob.modelPrompt`): the subjects stay here.
    func namedJob(_ expanded: String, settings: inout RenderSettings, references edit: [ReferenceImage],
                  model: Model? = nil, capacity: Int? = nil) -> [ReferenceImage] {
        var text = expanded
        if let reads = capacity ?? model.map({ $0.chain.text.output.readsImages ? $0.chain.denoising.maxReferences : 0 }) {
            // The subjects' pictures counted first: a reference past them does not go.
            var pictures = 0
            if case .success(let r)? = rewrite(expanded, model: model, capacity: capacity, editImages: edit.count) {
                pictures = subjects.references(r.placed).count
            }
            let inert = LabInputs.inert(in: expanded, known: Set(subjects.all.map(\.id)), capacity: reads,
                                        carried: edit.count + pictures)
            text = LabInputs.sent(expanded, leaving: inert)
        }
        guard case .success(let r)? = rewrite(text, model: model, capacity: capacity, editImages: edit.count),
              !r.placed.isEmpty else {
            if text != expanded { settings.modelPrompt = text }
            return edit
        }
        let placed = subjects.references(r.placed)
        settings.modelPrompt = r.prompt
        settings.subjectImages = placed.count
        settings.references = edit.count + placed.count
        return edit + placed
    }
}
