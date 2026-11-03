// **The Prompt Lab's Images zone**: a library of images, as long as one wants, each with its role.
//
//     Images
//     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐
//     │ ▣ 1        × │ │ ▣ 2        × │ │ ▣ 3  ×2    × │ │        │   a number: the place the rack will give it
//     │ 1 · Start  ▾ │ │ 2 · Reference▾│ │ (▣ @alice)   │ │   +    │   a role menu — the model aimed at offers only
//     │ Describe Beta│ │ Describe Beta│ │ Describe Beta│ │        │   the roles it can use
//     │ Name…        │ │ Name…        │ │ Edit…        │ └────────┘
//     └──────────────┘ └──────────────┘ └──────────────┘
//
// A **plain image** is dropped, opened, or sent from the history; its menu says what the render makes of it
// (`LabInputs.Role`): the starting image (image 1, which gives the format), a reference, or nothing (« describe only »).
// For a model that starts from an image (img2img), the tile of an image to describe offers « Start From This Image », and the
// starting image's tile carries its strength, 0 to 100 % — the rack's Images box receives both.
// A **subject** is an image that has a name, a kind and a frame (`SubjectSheet.swift`): it goes to the render only when the
// text cites it by `@name`, and takes the places after the images the roles send. Describe is the language model's, Name… opens
// the sheet, × removes. Nothing is written to disk.

import AppKit
import Siliconed
import SwiftUI

struct LabImages: View {
    @Bindable var model: LabModel

    /// Where the rack will place each image, by the carried list's order (from 1).
    private var numbers: (plain: [LabModel.Picture.ID: Int], subjects: [String: [Int]]) {
        guard case .carries(let carry) = model.receipt.outcome else { return ([:], [:]) }
        var plain: [LabModel.Picture.ID: Int] = [:]
        var subjects: [String: [Int]] = [:]
        for (i, image) in carry.images.enumerated() {
            switch image {
            case .plain(let id, _): plain[id] = i + 1
            case .subject(let p): subjects[p.subject.id, default: []].append(i + 1)
            }
        }
        return (plain, subjects)
    }

    var body: some View {
        let numbers = numbers
        VStack(alignment: .leading, spacing: 6) {
            Text("Images").font(.callout.weight(.semibold))
            ScrollView(.horizontal) {
                HStack(alignment: .top, spacing: 8) {
                    ForEach(model.pictures) { picture in
                        PictureTile(model: model, picture: picture, number: numbers.plain[picture.id])
                    }
                    ForEach(model.subjects.all) { subject in
                        SubjectTile(model: model, subject: subject, numbers: numbers.subjects[subject.id] ?? [])
                    }
                    addTile
                }
                .padding(.vertical, 2)
            }
            .scrollIndicators(.automatic)
            Text(hint).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var hint: LocalizedStringKey {
        if model.startIsImg2Img {
            return model.pictures.isEmpty
                ? "Drop an image here to describe it, or to start the render from it."
                : "The render can start from one image: it keeps its composition, the strength says how far it moves away. The others stay here, to be described."
        }
        if model.capacity == 0 {
            return "This model does not read images: they stay here, to be described."
        }
        if model.pictures.isEmpty && model.subjects.all.isEmpty {
            return "Drop images here. The first one is the image to edit; a name makes a picture citable as @name in the text."
        }
        return "Image 1 is the one edited and gives the format. The rack takes 3 images at most."
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
                .frame(width: LabTile.width, height: LabTile.picture)
                .contentShape(Rectangle())
        }
        .menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden)
        .help("Open images — or drop them here, or send one from the history (Open in the Prompt Lab)")
        .accessibilityLabel(Text("Add an Image"))
    }
}

enum LabTile {
    static let width: CGFloat = 132
    static let picture: CGFloat = 92
}

/// The place the rack gives an image, drawn on its corner.
private struct NumberBadge: View {
    let text: String
    var body: some View {
        Text(verbatim: text).font(.caption.weight(.semibold)).monospacedDigit()
            .padding(.horizontal, 6).padding(.vertical, 1)
            .background(Capsule().fill(.regularMaterial))
            .overlay(Capsule().strokeBorder(Palette.lavender, lineWidth: 1))
            .padding(4)
    }
}

/// « Describe », with the beta label the language model's gestures carry.
private struct DescribeButton: View {
    let model: LabModel
    let action: () -> Void

    var body: some View {
        HStack(spacing: 4) {
            Button("Describe", action: action)
                .disabled(!model.canDescribe)
                .help(model.writerPause.map { Text(verbatim: $0.sentence) } ?? Text("Writes what the image shows into the text, at the caret."))
            BetaBadge(help: "Beta: Rewrite and Describe use a language model still being settled, and may change.")
        }
        .controlSize(.small)
    }
}

// MARK: - A plain image

private struct PictureTile: View {
    let model: LabModel
    let picture: LabModel.Picture
    let number: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Image(nsImage: picture.thumbnail).resizable().scaledToFill()
                .frame(width: LabTile.width, height: LabTile.picture).clipped()
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .overlay {
                    // The starting image of an img2img, framed in the image wire's colour: the one the render starts from.
                    if model.startIsImg2Img, picture.role == .start {
                        RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(WireKind.image.color, lineWidth: 2)
                    }
                }
                .overlay(alignment: .topLeading) { if let number { NumberBadge(text: String(number)) } }
                .overlay(alignment: .topTrailing) {
                    Button { model.remove(picture) } label: {
                        Image(systemName: "xmark.circle.fill").symbolRenderingMode(.palette)
                            .foregroundStyle(.white, .black.opacity(0.55))
                    }
                    .buttonStyle(.plain).padding(4)
                    .help("Remove")
                    .accessibilityLabel(Text("Remove"))
                }
                .overlay { if model.describing == picture.id { ProgressView().controlSize(.small) } }
                .accessibilityLabel(Text("An image of the lab"))
            roleMenu
            if model.startIsImg2Img {
                if picture.role == .start { StartStrength(model: model) }
                else {
                    Button("Start From This Image") { model.startFrom(picture) }
                        .controlSize(.small)
                        .help("The render starts from this image and keeps its composition")
                }
            }
            DescribeButton(model: model) { model.describe(picture) }
            Button("Name…") { model.name(picture) }
                .buttonStyle(.link).controlSize(.small)
                .help("Gives it a name, a kind and a frame, to cite it as @name in the text")
        }
        .frame(width: LabTile.width)
    }

    private var roleMenu: some View {
        Menu {
            ForEach(model.offeredRoles, id: \.self) { role in
                Button { model.setRole(role, of: picture) } label: {
                    if role == picture.role { Label(role.label, systemImage: "checkmark") } else { Text(role.label) }
                }
            }
        } label: {
            Text(verbatim: (number.map { "\($0) · " } ?? "") + picture.role.label).lineLimit(1)
        }
        .menuStyle(.button).controlSize(.small)
        .help(picture.role.explanation(img2img: model.startIsImg2Img))
    }
}

/// **The starting image's strength, on its tile**: what it will be carried with — the rack's slider retouches it afterwards.
private struct StartStrength: View {
    @Bindable var model: LabModel

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack {
                Text("Strength").foregroundStyle(.secondary)
                Spacer(minLength: 2)
                Text(verbatim: StartWords.percent(model.startStrength)).monospacedDigit()
                    .onTapGesture(count: 2) { model.setStartStrength(model.defaultStartStrength) }
            }
            .font(.caption)
            Slider(value: Binding(get: { model.startStrength }, set: { model.setStartStrength($0) }), in: 0...1)
                .controlSize(.mini)
                .accessibilityLabel(Text("Strength"))
                .accessibilityValue(Text(verbatim: StartWords.percent(model.startStrength)))
        }
        .help(StartWords.explanation(nil))
    }
}

extension LabInputs.Role {
    var label: String {
        switch self {
        case .start: String(localized: "Starting image", comment: "The role of the image the model edits")
        case .reference: String(localized: "Reference", comment: "The role of an image the model sees as context")
        case .describeOnly: String(localized: "Describe only", comment: "The role of an image that goes nowhere")
        }
    }

    /// What the role does, for a model that starts from an image (img2img) or not.
    func explanation(img2img: Bool) -> String {
        if img2img, self == .start { return String(localized: "The image the render starts from: it keeps its composition; the size follows it.") }
        return explanation
    }

    var explanation: String {
        switch self {
        case .start: String(localized: "The image the model edits: image 1, which gives the format.")
        case .reference: String(localized: "The model sees it as context, after the starting image.")
        case .describeOnly: String(localized: "Stays in the lab: it is not sent to the render.")
        }
    }
}

// MARK: - A subject

private struct SubjectTile: View {
    let model: LabModel
    let subject: NamedSubjects.Subject
    let numbers: [Int]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ZStack {
                if let thumbnail = model.subjects.thumbnail(subject) {
                    Image(nsImage: thumbnail).resizable().scaledToFill()
                } else {
                    subject.kind.color.opacity(0.15).overlay(Image(systemName: subject.kind.symbol).foregroundStyle(subject.kind.color))
                }
            }
            .frame(width: LabTile.width, height: LabTile.picture).clipped()
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(subject.kind.color, lineWidth: 1.5))
            .overlay(alignment: .topLeading) {
                if !numbers.isEmpty { NumberBadge(text: numbers.map(String.init).joined(separator: "+")) }
            }
            .overlay(alignment: .topTrailing) {
                Button { model.confirmDeletion(subject) } label: {
                    Image(systemName: "xmark.circle.fill").symbolRenderingMode(.palette)
                        .foregroundStyle(.white, .black.opacity(0.55))
                }
                .buttonStyle(.plain).padding(4)
                .help("Delete…")
                .accessibilityLabel(Text("Delete…"))
            }
            Button { model.cite(subject) } label: {
                SubjectChip(name: subject.name, kind: subject.kind, thumbnail: nil, pictures: subject.pictures.count)
            }
            .buttonStyle(.plain)
            .help(Text(verbatim: "\(subject.kind.label) — \(subject.kind.taken)"))
            .draggable("@\(subject.name) ")
            DescribeButton(model: model) { model.describe(subject) }
            Button("Edit…") { model.edit(subject) }
                .buttonStyle(.link).controlSize(.small)
        }
        .frame(width: LabTile.width)
        .contextMenu {
            Button("Cite in the Text") { model.cite(subject) }
            Button("Edit…") { model.edit(subject) }
            Divider()
            Button("Delete…", role: .destructive) { model.confirmDeletion(subject) }
        }
    }
}
