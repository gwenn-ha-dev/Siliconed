// **The Prompt Lab's images**, wrapped under the text: those the render reads first, in their numbers, then the rest.
//
//     ┌──┐ 1 · Image to Edit ▾   ┌──┐ 2 · @alice          the render's, by number: the edited image, the references,
//     │▣ │ Describe              │▣ │ Person               the subjects the text cites
//     └──┘                       └──┘
//     ┌┄┄┐ No Role ▾            ┌┄┄┐ @bob               then what does not go: the images without a role (or one the
//     ┊▣ ┊ 3 / 3: full          ┊▣ ┊ Object              model lacks), the subjects the text does not cite; last, ＋
//     └┄┄┘ Describe             └┄┄┘ Describe
//
// - **The edge says the role**: the image wire's rose, full for the image the model edits or starts from, lighter for
//   a reference or a cited subject; dashed for an image that does not go — neutral, or with a role the model lacks, which
//   the tile says (« “Reference”: no effect with <model> »). A change of model never gives nor converts a role (`LabInputs`).
// - **Under a model whose encoder reads images** (`capacity` > 0) an image takes the next place as it comes in; its menu
//   changes it (Image to Edit, Reference, No Role) or names it. Full (three images with the cited subjects), a newcomer
//   comes in without a role and says so.
// - **Under a model that starts from an image** (`startIsImg2Img`) an image comes in without a role: « Start From This
//   Image » makes it the start (its tile then carries the rack's strength), Describe writes what it shows into the text.
//   **No subject** under a model that reads no image: the field greys their `@names`.
// - **Right-click**: Crop…, Replace…, Name…, Reuse These Settings (a PNG Siliconed wrote), Remove.
// - A **subject** (`SubjectSheet.swift`) goes only when the text cites it by `@name`, after the images the roles send; the
//   others stay at the end, to be cited by a click or a drag into the text.
// - **A tile is as wide as its picture and its words**, these at most `LabTile.meta`: tiles flow side by side while the
//   box's width holds them, two portraits and their buttons in the column's width.
// - **The grip under the tiles** gives the box more rows, or fewer: the tiles keep their size, more of them show; past
//   the rows given, the box scrolls. It stops on whole rows only, the bottoms the tiles are laid at (`TileRows`): what is
//   kept is a number of rows, not a height that would cut a row once the column's width changes.

import AppKit
import Siliconed
import SwiftUI

struct LabImages: View {
    @Bindable var model: LabModel
    /// How many rows of tiles the box shows, dragged by the grip under them and kept.
    @AppStorage(LabTile.rowsKey) private var rows = LabTile.defaultRows
    /// Each tile's frame in the tiles' own space, keyed by its entry: the rows' bottoms are read from them.
    @State private var frames: [String: CGRect] = [:]
    @State private var dragStart: CGFloat?
    private nonisolated static let space = "lab images"
    private static let addID = "add"
    /// The box's height until its tiles are laid out: one tile's.
    private static let unlaid = LabTile.height + 4

    /// A tile of the box: an image of the rack, or a subject of the library.
    private enum Entry: Identifiable {
        case picture(InputImage, number: Int?)
        case subject(NamedSubjects.Subject, numbers: [Int])
        var id: String {
            switch self {
            case .picture(let p, _): "picture \(p.id)"
            case .subject(let s, _): "subject \(s.id)"
            }
        }
        /// The first place the render gives it, `nil` when it does not go.
        var place: Int? {
            switch self {
            case .picture(_, let n): n
            case .subject(_, let ns): ns.min()
            }
        }
    }

    /// **The tiles in their order**: what the render reads, by its number; then the images that do not go, as they came;
    /// then the subjects the text does not cite, as the library keeps them. Under a model that reads no image, no subject.
    private var entries: [Entry] {
        var plain: [InputImage.ID: Int] = [:], cited: [String: [Int]] = [:]
        if case .success(let carry) = model.rack.carried {
            for (i, image) in carry.images.enumerated() {
                switch image {
                case .plain(let id, _): plain[id] = i + 1
                case .subject(let p): cited[p.subject.id, default: []].append(i + 1)
                }
            }
        }
        let subjects = model.capacity > 0 ? model.subjects.all : []
        let all = model.pictures.map { Entry.picture($0, number: plain[$0.id]) }
            + subjects.map { Entry.subject($0, numbers: cited[$0.id] ?? []) }
        let going = all.filter { $0.place != nil }.sorted { $0.place! < $1.place! }
        return going + all.filter { $0.place == nil }
    }

    var body: some View {
        // The tiles and the rows they make, once per drawing.
        let entries = self.entries
        let stops = rowStops(entries)
        VStack(alignment: .leading, spacing: 4) {
            ScrollView(.vertical) {
                FlowRow(spacing: 14) {
                    ForEach(entries) { entry in
                        Group {
                            switch entry {
                            case .picture(let picture, let number): PictureTile(model: model, picture: picture, number: number)
                            case .subject(let subject, let numbers): SubjectTile(model: model, subject: subject, numbers: numbers)
                            }
                        }
                        .onGeometryChange(for: CGRect.self) { $0.frame(in: .named(Self.space)) } action: { frames[entry.id] = $0 }
                    }
                    addTile
                        .onGeometryChange(for: CGRect.self) { $0.frame(in: .named(Self.space)) } action: { frames[Self.addID] = $0 }
                }
                .padding(.vertical, 2)
                .coordinateSpace(.named(Self.space))
            }
            .frame(height: TileRows.height(stops, rows: rows, empty: Self.unlaid))
            .scrollIndicators(.automatic)
            if stops.count > 1 { grip(stops) }
            Text(hint).font(.caption).foregroundStyle(Palette.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var hint: LocalizedStringKey {
        if model.startIsImg2Img {
            switch model.pictures.count {
            case 0: return "Drop an image here to describe it, or to start the render from it."
            case 1: return "The render can start from one image: it keeps its composition, the strength says how far it moves away."
            default: return "The render can start from one image: it keeps its composition, the strength says how far it moves away. The others stay here, to be described."
            }
        }
        if model.capacity == 0 {
            return "This model does not read images: they stay here, to be described."
        }
        if model.pictures.isEmpty && model.subjects.all.isEmpty {
            return "Drop images here. The first one is the image to edit; a name makes a picture citable as @name in the text."
        }
        // The count is the model's (`capacity`), never written here: a model that reads more says more.
        if !model.rack.edits {
            return "No image to edit: the render makes a new image from these, at the size chosen. This model reads \(model.capacity) at most."
        }
        return "Image 1 is the one edited and gives the format. This model reads \(model.capacity) at most."
    }

    /// **The heights at which the box shows whole rows** (`TileRows.stops`), from the frames of the tiles shown — the
    /// padding under the last row included. The rows kept, or all there are when fewer, make the box's height.
    private func rowStops(_ entries: [Entry]) -> [CGFloat] {
        let ids = Set(entries.map(\.id) + [Self.addID])
        return TileRows.stops(Array(frames.filter { ids.contains($0.key) }.values), padding: 2)
    }

    /// **The box's grip**: dragged down, more rows of tiles show; up, fewer — a row at a time, the nearest to the hand;
    /// never fewer than one, never more than there are.
    private func grip(_ stops: [CGFloat]) -> some View {
        Capsule().fill(Palette.secondary.opacity(0.5)).frame(width: 36, height: 4)
            .frame(maxWidth: .infinity).frame(height: 10)
            .contentShape(Rectangle())
            .pointerStyle(.rowResize)
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { drag in
                    let start = dragStart ?? TileRows.height(stops, rows: rows, empty: Self.unlaid)
                    dragStart = start
                    if let n = TileRows.rows(nearest: start + drag.translation.height, in: stops, limit: LabTile.rows.upperBound) {
                        rows = n
                    }
                }
                .onEnded { _ in dragStart = nil })
            .help("Drag to show more images, or fewer")
            .accessibilityHidden(true)
    }

    private var addTile: some View {
        Menu {
            Button("Open Images…") { model.chooseImages() }
            Button("Name an Image…") { model.chooseImageToName() }
        } label: {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .strokeBorder(style: StrokeStyle(lineWidth: 1.25, dash: [4, 3]))
                .foregroundStyle(WireKind.image.color.opacity(0.7))
                .overlay(Image(systemName: "plus").foregroundStyle(WireKind.image.color))
                .frame(width: LabTile.width(aspect: 0.75), height: LabTile.height)
                .contentShape(Rectangle())
        }
        .menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden)
        .help("Open images — or drop them here, or send one from the history (Send to the Prompt Lab)")
        .accessibilityLabel(Text("Add an Image"))
    }
}

enum LabTile {
    /// The pictures' height: a face, a composition read at a glance; the box shows more of them, never larger.
    static let height: Double = 112
    /// The rows the box shows: one at least, ten at most; two by default — the three images a render reads at most
    /// (`NamedSubjects.maxImages`) and one more.
    static let rows: ClosedRange<Int> = 1...10
    static let defaultRows = 2
    static let rowsKey = "labImagesRows"
    /// The widest a tile's words go, beside its picture: a longer caption wraps under it. A tile takes only what its words
    /// need (`CappedWidth`), so that two portraits and their buttons hold side by side in the column.
    static let meta: CGFloat = 150

    /// A picture's width: its own proportions, between a tall portrait and a wide landscape.
    static func width(aspect: Double) -> Double { height * min(max(aspect, 0.5), 1.8) }

    /// The edge of a tile's picture: the image wire's colour, full for the image the render edits or starts from, lighter
    /// for a reference or a subject; dashed for what does not go.
    @ViewBuilder static func edge(goes: Bool, main: Bool) -> some View {
        let shape = RoundedRectangle(cornerRadius: 6, style: .continuous)
        if goes {
            shape.strokeBorder(WireKind.image.color.opacity(main ? 1 : 0.6), lineWidth: 2)
        } else {
            shape.strokeBorder(Color.secondary.opacity(0.6), style: StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
        }
    }
}

/// **A width as the content asks, at most `width`**: the content laid out at the lesser of the two, a longer text wrapping
/// under it. SwiftUI's `frame(maxWidth:)` clamps the frame but lays an unproposed text on one line, past the frame.
private struct CappedWidth: Layout {
    let width: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let w = min(width, proposal.width ?? width, subviews.map { $0.sizeThatFits(.unspecified).width }.max() ?? 0)
        let h = subviews.map { $0.sizeThatFits(ProposedViewSize(width: w, height: nil)).height }.max() ?? 0
        return CGSize(width: w, height: h)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for s in subviews { s.place(at: bounds.origin, proposal: ProposedViewSize(width: bounds.width, height: nil)) }
    }
}

private extension View {
    func cappedWidth(_ width: CGFloat) -> some View { CappedWidth(width: width) { self } }
}

/// « Describe »: what the image shows, written into the text.
private struct DescribeButton: View {
    let model: LabModel
    let action: () -> Void

    var body: some View {
        Button("Describe", action: action)
            .help("Writes what the image shows into the text, at the caret.")
            .unavailable(model.describeRefusal)
            .controlSize(.small)
    }
}

// MARK: - A plain image

private struct PictureTile: View {
    let model: LabModel
    let picture: InputImage
    let number: Int?
    @State private var hover = false

    private var rack: AppState { model.rack }
    private var goes: Bool { rack.isEffective(picture) }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(nsImage: picture.shown.vignette).resizable().scaledToFill()
                .frame(width: LabTile.width(aspect: Double(picture.shown.width) / Double(max(1, picture.shown.height))),
                       height: LabTile.height).clipped()
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .overlay { LabTile.edge(goes: goes, main: picture.role == .edited || picture.role == .start) }
                .removalOnHover(hover, padding: 2, help: "Remove", label: "Remove") { model.remove(picture) }
                .overlay { if model.describing == picture.id { ProgressView().controlSize(.small) } }
                .accessibilityLabel(Text("An image of the lab"))
            VStack(alignment: .leading, spacing: 3) {
                if model.capacity > 0 { roleMenu } else if goes { roleLabel }
                if let caption { Text(caption).font(.caption2).foregroundStyle(Palette.secondary).lineLimit(2) }
                if model.startIsImg2Img {
                    if picture.role == .start { StartStrength(model: model) }
                    else {
                        Button("Start From This Image") { model.startFrom(picture) }
                            .controlSize(.small)
                            .help("The render starts from this image and keeps its composition")
                    }
                }
                DescribeButton(model: model) { model.describe(picture) }
            }
            .cappedWidth(LabTile.meta)
        }
        .onHover { hover = $0 }
        .contextMenu { menu }
    }

    /// Why it does not go, when it has a role: the model lacks it. Or, without a role under a model that reads images, that
    /// the render is full.
    private var caption: String? {
        if picture.role != .neutral, !goes, let name = rack.card?.name {
            return String(localized: "“\(picture.role.label)”: no effect with \(name)")
        }
        let limit = min(model.capacity, NamedSubjects.maxImages)
        if picture.role == .neutral, limit > 0, rack.references.count >= limit {
            return String(localized: "\(String(rack.references.count)) / \(String(limit)): full")
        }
        return nil
    }

    private var roleLabel: some View {
        Text(verbatim: (number.map { "\($0) · " } ?? "") + picture.role.label)
            .font(.caption.weight(.semibold)).foregroundStyle(goes ? WireKind.image.color : Palette.secondary).lineLimit(1)
            .help(picture.role.explanation(img2img: model.startIsImg2Img))
    }

    private var roleMenu: some View {
        Menu {
            ForEach(model.offeredRoles, id: \.self) { role in
                Button { model.setRole(role, of: picture) } label: {
                    if role == picture.role { Label(role.label, systemImage: "checkmark") } else { Text(role.label) }
                }
            }
            Divider()
            Button("Name…") { model.name(picture) }
        } label: {
            roleLabel
        }
        .menuStyle(.button).buttonStyle(.plain).fixedSize()
        .help(picture.role.explanation(img2img: model.startIsImg2Img))
    }

    @ViewBuilder private var menu: some View {
        Button("Crop…") { rack.beginCrop(picture.shown) }
            .disabled(rack.cropRefusal(width: picture.shown.width, height: picture.shown.height) != nil)
        Button("Replace…") { rack.replaceReference(picture.shown.id) }
        Button("Name…") { model.name(picture) }
        if picture.role == .start, model.startIsImg2Img {
            Button("Reset to \(StartWords.percent(rack.defaultStartStrength))") { rack.setStartStrength(rack.defaultStartStrength) }
        }
        if let bytes = picture.bytes, AppState.isRecipe(bytes) {
            Button("Reuse These Settings") { _ = rack.restoreSettings(fromPNG: bytes, file: picture.shown.name) }
        }
        Divider()
        Button("Remove", role: .destructive) { model.remove(picture) }
    }
}

/// **The starting image's strength, on its tile** — its one place: the rack reads it in its port, it does not set it.
/// Orange, and said, when it is so high that the image is ignored.
private struct StartStrength: View {
    let model: LabModel

    var body: some View {
        let rack = model.rack
        VStack(alignment: .leading, spacing: 1) {
            HStack {
                Text("Strength").foregroundStyle(Palette.secondary)
                Spacer(minLength: 2)
                Text(verbatim: StartWords.percent(rack.startStrength)).monospacedDigit()
                    .onTapGesture(count: 2) { rack.setStartStrength(rack.defaultStartStrength) }
            }
            .font(.caption)
            Slider(value: Binding(get: { rack.startStrength }, set: { rack.setStartStrength($0) }), in: 0...1)
                .controlSize(.mini).tint(Palette.control)
                .accessibilityLabel(Text("Strength"))
                .accessibilityValue(Text(verbatim: StartWords.percent(rack.startStrength)))
            if rack.startIgnored {
                Text("Image ignored: lower the strength").font(.caption2).foregroundStyle(Palette.caution)
                    .lineLimit(2).fixedSize(horizontal: false, vertical: true)
                    .help(rack.startCaption)
            }
        }
        // The slider at the words' full width: alone it would take the little it asks.
        .frame(width: LabTile.meta)
        .help(StartWords.explanation(rack.startRedraws))
        .bracketable(rack, .startStrength, gutter: 18)
    }
}

extension LabInputs.Role {
    var label: String {
        switch self {
        case .edited: String(localized: "Image to Edit", comment: "The role of the image the model edits")
        case .reference: String(localized: "Reference", comment: "The role of an image the model sees as context")
        case .start: String(localized: "Starting image", comment: "The role of the image an img2img starts from")
        case .neutral: String(localized: "No Role", comment: "An image of the lab that does not go to the render")
        }
    }

    /// What the role does, for a model that starts from an image (img2img) or not.
    func explanation(img2img: Bool) -> String {
        if img2img, self == .start { return String(localized: "The image the render starts from: it keeps its composition; the size follows it.") }
        return explanation
    }

    var explanation: String {
        switch self {
        case .edited: String(localized: "The image the model edits: image 1, which gives the format.")
        case .reference: String(localized: "The model sees it as context, after the image to edit.")
        case .start: String(localized: "The image the render starts from: it keeps its composition; the size follows it.")
        case .neutral: String(localized: "Stays in the lab: it is not sent to the render.")
        }
    }
}

// MARK: - A subject

private struct SubjectTile: View {
    let model: LabModel
    let subject: NamedSubjects.Subject
    let numbers: [Int]

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            ZStack {
                if let thumbnail = model.subjects.thumbnail(subject) {
                    Image(nsImage: thumbnail).resizable().scaledToFill()
                } else {
                    subject.kind.color.opacity(0.15).overlay(Image(systemName: subject.kind.symbol).foregroundStyle(subject.kind.color))
                }
            }
            .frame(width: LabTile.width(aspect: 0.75), height: LabTile.height).clipped()
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay { LabTile.edge(goes: !numbers.isEmpty, main: false) }
            VStack(alignment: .leading, spacing: 3) {
                Button { model.cite(subject) } label: {
                    Text(verbatim: (numbers.isEmpty ? "" : numbers.map(String.init).joined(separator: "+") + " · ") + "@" + subject.name)
                        .font(.caption.weight(.semibold)).lineLimit(1)
                        .foregroundStyle(numbers.isEmpty ? Palette.secondary : WireKind.image.color)
                }
                .buttonStyle(.plain)
                .help("Cite in the Text")
                .draggable("@\(subject.name) ")
                Text(subject.kind.label)
                    .font(.caption2).foregroundStyle(Palette.secondary).lineLimit(2)
                DescribeButton(model: model) { model.describe(subject) }
            }
            .cappedWidth(LabTile.meta)
        }
        .contextMenu {
            Button("Cite in the Text") { model.cite(subject) }
            Button("Edit…") { model.edit(subject) }
            Divider()
            Button("Delete…", role: .destructive) { model.confirmDeletion(subject) }
        }
    }
}
