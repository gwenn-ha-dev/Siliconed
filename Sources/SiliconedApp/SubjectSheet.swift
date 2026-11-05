// **Naming a subject**: an image dropped or opened, its frame, a name, a kind — the sheet.
//
//     ┌ Name a Subject ─────────────────────────────────────────────┐
//     │ ┌──────────────────────┐   Name  @ alice                     │
//     │ │░░░░┌──────────┐░░░░░│   Kind  ● Person   face, hair…      │
//     │ │░░░░│  framed  │░░░░░│         ○ Garment  …                 │
//     │ │░░░░└──────────┘░░░░░│         …                            │
//     │ └──────────────────────┘                                     │
//     │  Whole Image · Add a Second Picture…        Cancel  Create  │
//     └─────────────────────────────────────────────────────────────┘
//
// **The whole image starts framed**: most pictures need nothing. The frame is a rectangle — drag inside
// to move it, an edge or a corner to resize it, outside to draw another —, never a mask
// (by design: no masks in the app). It serves the picture with two people, the coat worn by someone else. The
// frame is kept with the picture, not a cropped copy: reopening the sheet reframes it.

import AppKit
import Siliconed
import SwiftUI

/// **The sheet's state**: a new subject from one picture, or an existing one being edited.
struct SubjectDraft: Identifiable {
    struct Picture: Identifiable {
        let id = UUID()
        /// The picture the subject already holds, or `nil` for one being added.
        var existing: NamedSubjects.Picture?
        var source: SubjectSource?
        /// Upright, for display only (its long side ≤ 1600): the frame is in fractions.
        let image: CGImage
        var frame: NamedSubjects.Frame
    }

    let id = UUID()
    /// `nil`: a new subject.
    let original: NamedSubjects.Subject?
    var name: String
    var kind: NamedSubjects.Kind
    var pictures: [Picture]
    /// The other subjects' names: a name must differ from them (ignoring case).
    let taken: [String]

    static let displaySide = 1600

    init?(new source: SubjectSource, taken: [String]) {
        guard let picture = Self.picture(source) else { return nil }
        original = nil
        self.taken = taken
        pictures = [picture]
        kind = .person
        // A name proposed from the file's, never imposed — and only when it is free.
        var suggested = ""
        if case .file(let url) = source {
            suggested = NamedSubjects.suggestedName(url.deletingPathExtension().lastPathComponent).lowercased()
        }
        name = NamedSubjects.check(name: suggested, taken: taken) == nil ? suggested : ""
    }

    @MainActor init(editing s: NamedSubjects.Subject, library: SubjectLibrary) {
        original = s
        name = s.name
        kind = s.kind
        taken = library.all.map(\.name).filter { NamedSubjects.key($0) != s.id }
        // Its bytes decoded once already, when it was named.
        pictures = s.pictures.compactMap { p in
            library.upright(s, p, maxSide: Self.displaySide).map { Picture(existing: p, source: nil, image: $0, frame: p.frame) }
        }
    }

    var count: Int { pictures.count }

    static func picture(_ source: SubjectSource) -> Picture? {
        let image: CGImage?
        switch source {
        case .file(let url): image = SubjectPictures.upright(url, maxSide: displaySide)
        case .png(let data): image = SubjectPictures.upright(data, maxSide: displaySide)
        }
        return image.map { Picture(existing: nil, source: source, image: $0, frame: .whole) }
    }

    var nameProblem: NamedSubjects.NameProblem? { NamedSubjects.check(name: name, taken: taken) }
}

struct SubjectSheet: View {
    let lab: LabModel
    @State var draft: SubjectDraft
    /// Why the last gesture failed, in one sentence.
    @State private var failure: String?
    @State private var shown = 0
    @State private var confirmsDeletion = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(draft.original == nil ? String(localized: "Name a Subject") : String(localized: "Edit @\(draft.original!.name)"))
                .font(.headline)
            HStack(alignment: .top, spacing: 20) {
                pictureColumn
                VStack(alignment: .leading, spacing: 14) {
                    nameField
                    kindChooser
                    Spacer(minLength: 0)
                    preview
                }
                .frame(width: 270)
            }
            if let failure {
                Text(failure).font(.caption).foregroundStyle(Palette.failure)
            }
            HStack {
                if draft.original != nil {
                    Button("Delete Subject…", role: .destructive) { confirmsDeletion = true }
                }
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(draft.original == nil ? String(localized: "Create") : String(localized: "Save")) { save() }
                    .keyboardShortcut(.defaultAction).tint(Palette.control)
                    .disabled(draft.nameProblem != nil || draft.count == 0)
            }
        }
        .padding(20)
        .frame(width: 760)
        .confirmationDialog(String(localized: "Delete @\(draft.original?.name ?? "")?"), isPresented: $confirmsDeletion) {
            Button("Delete", role: .destructive) {
                if let s = draft.original { lab.deleteSubject(s) }
                dismiss()
            }
        } message: {
            Text("Prompts that cite it will be refused.")
        }
    }

    // ── the picture and its frame ──

    private var pictureColumn: some View {
        VStack(alignment: .leading, spacing: 8) {
            if draft.pictures.indices.contains(shown) {
                FrameEditor(image: draft.pictures[shown].image, frame: $draft.pictures[shown].frame, tint: draft.kind.color)
                    .frame(width: 430, height: 380)
                    .id(draft.pictures[shown].id)
            }
            HStack(spacing: 8) {
                if draft.pictures.count > 1 {
                    Picker("Picture", selection: $shown) {
                        ForEach(draft.pictures.indices, id: \.self) { i in Text(verbatim: "\(i + 1)").tag(i) }
                    }
                    .pickerStyle(.segmented).labelsHidden().fixedSize().tint(Palette.control)
                    .help("The subject’s two pictures: each has its own frame")
                }
                Button("Whole Image") { draft.pictures[shown].frame = .whole }
                    .disabled(draft.pictures.indices.contains(shown) ? draft.pictures[shown].frame == .whole : true)
                Spacer()
                if draft.count < NamedSubjects.maxPictures {
                    Button("Add a Second Picture…") { addPicture() }
                        .help("Two pictures of the same subject: the model sees one subject, from two views")
                } else if draft.pictures.count > 1 {
                    Button("Remove This Picture") {
                        draft.pictures.remove(at: shown)
                        shown = 0
                    }
                }
            }
            .controlSize(.small)
            Text("Drag to frame what the model should look at: inside to move, an edge to resize, outside to draw another frame.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: 430, alignment: .leading)
        }
    }

    private func addPicture() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.prompt = String(localized: "Choose")
        panel.message = String(localized: "Another picture of the same subject.")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard let p = SubjectDraft.picture(.file(url)) else {
            failure = String(localized: "This image can’t be read")
            return
        }
        draft.pictures.append(p)
        shown = draft.pictures.count - 1
    }

    // ── name and kind ──

    private var nameField: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Name").font(.callout).foregroundStyle(.secondary)
            HStack(spacing: 2) {
                Text(verbatim: "@").foregroundStyle(.secondary)
                TextField("Name", text: $draft.name, prompt: Text(verbatim: "alice"))
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
            }
            Group {
                if let p = draft.nameProblem, p != .empty {
                    Text(p.sentence).foregroundStyle(Palette.failure)
                } else {
                    Text("Write @\(draft.name.isEmpty ? "name" : draft.name) in the prompt to cite it.")
                        .foregroundStyle(.secondary)
                }
            }
            .font(.caption)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var kindChooser: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Kind").font(.callout).foregroundStyle(.secondary)
            ForEach(NamedSubjects.Kind.allCases, id: \.self) { kind in
                let chosen = draft.kind == kind
                Button { draft.kind = kind } label: {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: kind.symbol)
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(kind.color)
                            .frame(width: 20, height: 20)
                            .background(RoundedRectangle(cornerRadius: 5, style: .continuous).fill(kind.color.opacity(0.16)))
                        VStack(alignment: .leading, spacing: 1) {
                            Text(kind.label).font(.callout.weight(chosen ? .semibold : .regular))
                            Text(kind.taken).font(.caption).foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 8).padding(.vertical, 5)
                    .background(RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(chosen ? kind.color.opacity(0.10) : Color.clear))
                    .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .strokeBorder(chosen ? kind.color : Color.clear, lineWidth: 1.25))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(chosen ? .isSelected : [])
            }
        }
    }

    /// The chip as the prompt will draw it.
    private var preview: some View {
        HStack(spacing: 6) {
            Text("In the prompt:").font(.caption).foregroundStyle(.secondary)
            SubjectChip(name: draft.name.isEmpty ? "name" : draft.name, kind: draft.kind,
                        thumbnail: draft.pictures.first.flatMap { p in
                            SubjectPictures.crop(p.image, p.frame).map { NSImage(cgImage: $0, size: .zero) }
                        })
        }
    }

    // ── saving ──

    private func save() {
        let problem: Problem?
        if let original = draft.original {
            var s = original
            s.name = draft.name
            s.kind = draft.kind
            s.pictures = draft.pictures.compactMap { p in p.existing.map { NamedSubjects.Picture(file: $0.file, frame: p.frame) } }
            let added = draft.pictures.compactMap { p in p.existing == nil ? p.source.map { ($0, p.frame) } : nil }
            problem = lab.subjects.update(original.name, to: s, adding: added)
        } else {
            problem = lab.subjects.create(name: draft.name, kind: draft.kind,
                                          pictures: draft.pictures.compactMap { p in p.source.map { ($0, p.frame) } })
        }
        if let problem { failure = problem.title } else {
            if let old = draft.original?.name, old != draft.name { lab.renameMentions(old, to: draft.name) }
            dismiss()
        }
    }
}

// MARK: - The frame on the picture

/// **A rectangle on the image**: the outside dimmed, the frame's edge in the kind's color, its corners
/// marked. One drag gesture: started inside, it moves the frame; on an edge or a corner, it resizes
/// it; outside, it draws a new one.
private struct FrameEditor: View {
    let image: CGImage
    @Binding var frame: NamedSubjects.Frame
    let tint: Color

    /// Which edges the drag moves: none is a move; `draw` is a new frame from the drag's start.
    private struct Grip: Equatable { var left = false, right = false, top = false, bottom = false, draw = false }
    @State private var start: NamedSubjects.Frame?
    @State private var grip = Grip()
    private static let reach: CGFloat = 9
    private static let smallest = 0.04

    var body: some View {
        GeometryReader { g in
            let r = fitted(in: g.size)
            ZStack(alignment: .topLeading) {
                Image(decorative: image, scale: 1)
                    .resizable().interpolation(.high)
                    .frame(width: r.width, height: r.height)
                    .offset(x: r.minX, y: r.minY)
                Canvas { c, _ in
                    let f = rect(frame, in: r)
                    var outside = Path(r)
                    outside.addPath(Path(f))
                    c.fill(outside, with: .color(.black.opacity(0.5)), style: FillStyle(eoFill: true))
                    c.stroke(Path(f), with: .color(.white.opacity(0.9)), lineWidth: 3)
                    c.stroke(Path(f), with: .color(tint), lineWidth: 1.5)
                    for p in [CGPoint(x: f.minX, y: f.minY), CGPoint(x: f.maxX, y: f.minY),
                              CGPoint(x: f.minX, y: f.maxY), CGPoint(x: f.maxX, y: f.maxY)] {
                        let k = CGRect(x: p.x - 4, y: p.y - 4, width: 8, height: 8)
                        c.fill(Path(roundedRect: k, cornerRadius: 2), with: .color(.white))
                        c.stroke(Path(roundedRect: k, cornerRadius: 2), with: .color(tint), lineWidth: 1.25)
                    }
                }
                .allowsHitTesting(false)
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0).onChanged { v in drag(v, in: r) }.onEnded { _ in start = nil })
            .accessibilityElement()
            .accessibilityLabel(Text("Frame"))
            .accessibilityValue(Text(verbatim: "\(Int(frame.width * 100)) % × \(Int(frame.height * 100)) %"))
            .accessibilityAction(named: Text("Whole Image")) { frame = .whole }
        }
        .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Palette.well))
    }

    private func fitted(in size: CGSize) -> CGRect {
        let s = min(size.width / CGFloat(image.width), size.height / CGFloat(image.height))
        let w = CGFloat(image.width) * s, h = CGFloat(image.height) * s
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }

    private func rect(_ f: NamedSubjects.Frame, in r: CGRect) -> CGRect {
        CGRect(x: r.minX + f.x * r.width, y: r.minY + f.y * r.height, width: f.width * r.width, height: f.height * r.height)
    }

    private func drag(_ v: DragGesture.Value, in r: CGRect) {
        guard r.width > 0, r.height > 0 else { return }
        if start == nil {
            start = frame
            let f = rect(frame, in: r), p = v.startLocation, k = Self.reach
            let inY = p.y > f.minY - k && p.y < f.maxY + k, inX = p.x > f.minX - k && p.x < f.maxX + k
            grip = Grip(left: inY && abs(p.x - f.minX) < k, right: inY && abs(p.x - f.maxX) < k,
                        top: inX && abs(p.y - f.minY) < k, bottom: inX && abs(p.y - f.maxY) < k)
            if grip == Grip() && !f.contains(p) { grip.draw = true }
        }
        guard let s = start else { return }
        let dx = v.translation.width / r.width, dy = v.translation.height / r.height
        let m = Self.smallest
        if grip.draw {
            let a = CGPoint(x: (v.startLocation.x - r.minX) / r.width, y: (v.startLocation.y - r.minY) / r.height)
            let b = CGPoint(x: (v.location.x - r.minX) / r.width, y: (v.location.y - r.minY) / r.height)
            let x0 = min(max(min(a.x, b.x), 0), 1), x1 = min(max(max(a.x, b.x), 0), 1)
            let y0 = min(max(min(a.y, b.y), 0), 1), y1 = min(max(max(a.y, b.y), 0), 1)
            guard x1 - x0 >= m, y1 - y0 >= m else { return }
            frame = NamedSubjects.Frame(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
            return
        }
        if grip == Grip() {
            frame = NamedSubjects.Frame(x: min(max(s.x + dx, 0), 1 - s.width), y: min(max(s.y + dy, 0), 1 - s.height),
                                        width: s.width, height: s.height)
            return
        }
        var left = s.x, right = s.x + s.width, top = s.y, bottom = s.y + s.height
        if grip.left { left = min(max(s.x + dx, 0), right - m) }
        if grip.right { right = max(min(s.x + s.width + dx, 1), left + m) }
        if grip.top { top = min(max(s.y + dy, 0), bottom - m) }
        if grip.bottom { bottom = max(min(s.y + s.height + dy, 1), top + m) }
        frame = NamedSubjects.Frame(x: left, y: top, width: right - left, height: bottom - top)
    }
}

// MARK: - A subject's chip

/// **A subject as a chip**: its framed thumbnail, `@name`, an edge in its kind's color — the same
/// drawing as in the prompt (`PromptField.swift`), so that what is cited is recognized at once.
struct SubjectChip: View {
    let name: String
    let kind: NamedSubjects.Kind
    let thumbnail: NSImage?
    var pictures = 1

    var body: some View {
        HStack(spacing: 5) {
            ZStack {
                if let thumbnail {
                    Image(nsImage: thumbnail).resizable().aspectRatio(contentMode: .fill)
                } else {
                    Image(systemName: kind.symbol).font(.system(size: 9)).foregroundStyle(kind.color)
                }
            }
            .frame(width: 18, height: 18)
            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
            Text(verbatim: "@" + name).font(.callout).lineLimit(1)
            if pictures > 1 {
                Text(verbatim: "×\(pictures)").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            }
        }
        .padding(.leading, 3).padding(.trailing, 8).padding(.vertical, 3)
        .background(Capsule().fill(kind.color.opacity(0.13)))
        .overlay(Capsule().strokeBorder(kind.color, lineWidth: 1))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: "@" + name))
        .accessibilityValue(Text(kind.label))
    }
}
