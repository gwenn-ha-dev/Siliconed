// **The Prompt Lab's window** (`Lab.swift` holds the state): the Images zone (`LabImages.swift`), the Text, and the line that
// says what the rack will receive.

import AppKit
import Siliconed
import SwiftUI

struct LabView: View {
    @Bindable var model: LabModel
    @State private var typing = false
    @State private var targeted = false

    private var families: [Family] { model.families }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            LabImages(model: model)
            textZone
            rackLine
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .dropDestination(for: DroppedImage.self) { items, _ in
            guard !items.isEmpty else { return false }
            for item in items { model.load(item) }
            return true
        } isTargeted: { targeted = $0 }
        .overlay {
            if targeted {
                RoundedRectangle(cornerRadius: 12).strokeBorder(Color.primary.opacity(0.35), style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                    .padding(6).allowsHitTesting(false)
            }
        }
        .sheet(item: $model.draft, onDismiss: { model.sheetClosed() }) { draft in SubjectSheet(lab: model, draft: draft) }
        .onChange(of: model.rolesKey) { model.conformRoles() }
    }

    // MARK: Pieces

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            if families.count > 1, let target = model.target {
                Picker("Write for", selection: Binding(get: { target }, set: { model.chosenTarget = $0 })) {
                    ForEach(families, id: \.self) { Text(verbatim: $0.name).tag($0) }
                }
                .fixedSize()
                .disabled(model.isWriting)
            } else if let target = model.target {
                Text("Write for \(target.name)").foregroundStyle(.secondary)
            }
            Spacer()
            if !model.following, case .ready(let name, _) = model.availability {
                Text("Writer: \(name)").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var card: some View {
        RoundedRectangle(cornerRadius: 10).fill(Palette.surface)
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(typing ? Palette.amber : Palette.hairline, lineWidth: typing ? 1 : 0.5))
    }

    // MARK: The text

    private var textZone: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Text").font(.callout.weight(.semibold))
                Spacer()
                Button("Clear") { model.text = "" }
                    .buttonStyle(.link).font(.callout)
                    .disabled(model.text.isEmpty || model.isWriting)
                    .help("Empties the text. ⌘Z in the text brings it back.")
                Button("Use the Rack’s Prompt") { model.useRackPrompt() }
                    .buttonStyle(.link).font(.callout)
                    .disabled(model.rack.prompt.isEmpty || model.isWriting)
            }
            ZStack(alignment: .topLeading) {
                PromptField(text: $model.text, subjects: model.subjects, focused: $typing, editable: !model.isWriting)
                    .opacity(model.changes == nil ? 1 : 0)
                if model.text.isEmpty {
                    Text("Type or paste a prompt. Write @ to cite a subject.")
                        .foregroundStyle(.tertiary).padding(.leading, 5).padding(.top, 3)
                        .allowsHitTesting(false)
                }
                if let changes = model.changes {
                    // The marks lie over the text, which stays under them: ⌘Z still undoes the gesture. While the writer writes,
                    // they are the text it has written so far, against the one it started from.
                    ScrollView {
                        Text(Self.shown(PromptDiff.pieces(from: changes.before, to: changes.after)))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    // The text starts at the top and grows down, as typed text does; once it fills the field, the last words
                    // stay in sight while it comes. A single anchor at the bottom drew a short text at the foot of the field,
                    // and threw it back to the top when the writer ended.
                    .defaultScrollAnchor(.top, for: .alignment)
                    .defaultScrollAnchor(.top, for: .initialOffset)
                    .defaultScrollAnchor(model.isWriting ? .bottom : nil, for: .sizeChanges)
                    .background(Palette.surface)
                    .allowsHitTesting(false)
                }
            }
            .padding(6)
            .frame(minHeight: 110, maxHeight: .infinity)
            .background(card)
            actions
            status
        }
        .frame(maxHeight: .infinity)
    }

    private var actions: some View {
        HStack(spacing: 8) {
            if model.isWriting {
                Button { model.stop() } label: { Text("Stop").frame(minWidth: 70) }
                    .controlSize(.large)
                    .keyboardShortcut(.cancelAction)
            } else {
                Button { model.rewrite() } label: { Text("Rewrite").fontWeight(.semibold).frame(minWidth: 70) }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!model.canRewrite)
                BetaBadge(help: "Beta: Rewrite and Describe use a language model still being settled, and may change.")
            }
            Spacer()
            if model.changes != nil, !model.isWriting {
                Button("Edit") { model.dismissChanges() }
                    .buttonStyle(.link).font(.callout)
                    .help("Puts the marks of the rewrite away. ⌘Z in the text brings the previous text back.")
            }
        }
    }

    /// What the writer is doing, why it waits, what went wrong — in as many lines as it has things to say.
    @ViewBuilder private var status: some View {
        switch model.phase {
        case .loading(let fraction):
            HStack(spacing: 8) {
                if let fraction { ProgressView(value: fraction).frame(maxWidth: 160) } else { ProgressView().controlSize(.small) }
                Text("Preparing the writer…").font(.caption).foregroundStyle(.secondary)
            }
        case .writing:
            // The text itself comes in the field, marked.
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Writing…").font(.caption).foregroundStyle(.secondary)
            }
        case .idle:
            EmptyView()
        }
        if let pause = model.writerPause {
            Label(pause.sentence, systemImage: pause.symbol).font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        headInstall
        if let failure = model.failure {
            Text(failure).font(.caption).foregroundStyle(Palette.failure)
        } else if model.stopped {
            Text("Stopped: an image started rendering.").font(.caption).foregroundStyle(.secondary)
        }
        if let notice = model.notice {
            Label(notice, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The writer's head: one button when the encoder is there and the head is not; the progress while it comes; the failure in red.
    @ViewBuilder private var headInstall: some View {
        if model.installingHead {
            HStack(spacing: 8) {
                if let fraction = Self.percent(MainWindow.shared.local.forgeJournal.last) { ProgressView(value: fraction) } else { ProgressView().controlSize(.small) }
                Text("Installing the writer…").font(.caption).foregroundStyle(.secondary)
                Button("Stop") { MainWindow.shared.local.cancelForge() }.buttonStyle(.link).font(.caption)
            }
        } else if case .unavailable(.needsHead) = model.availability, !model.following, !model.rendering {
            Button("Install the Writer") { model.installHead(MainWindow.shared.local) }
                .disabled(!MainWindow.shared.local.managementPossible)
        }
        if let why = model.headFailure {
            Text(why).font(.caption).foregroundStyle(Palette.failure)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// `  … 40 %`, the line the installer's journal writes while a file comes.
    static func percent(_ line: String?) -> Double? {
        guard let line, line.hasSuffix(" %"), let n = line.split(separator: " ").dropLast().last.flatMap({ Int($0) }) else { return nil }
        return Double(n) / 100
    }

    // MARK: The rack will receive

    private var rackLine: some View {
        let receipt = model.receipt
        return VStack(alignment: .leading, spacing: 6) {
            Divider()
            Text("The rack will receive").font(.callout.weight(.semibold))
            switch receipt.outcome {
            case .nothing:
                Text("Nothing to carry yet: write a text, or choose an image to send.").font(.caption).foregroundStyle(.secondary)
            case .refused(let refusal):
                Label(refusal.sentence, systemImage: "exclamationmark.circle").font(.caption).foregroundStyle(Palette.failure)
                    .fixedSize(horizontal: false, vertical: true)
            case .carries(let carry):
                if !carry.prompt.allSatisfy(\.isWhitespace) {
                    Text(verbatim: carry.prompt).font(.callout).lineLimit(3).textSelection(.enabled)
                        .help(carry.prompt).frame(maxWidth: .infinity, alignment: .leading)
                }
                if !carry.images.isEmpty { imagesLine(carry) }
                if let replaces = Self.replaces(receipt, start: model.startIsImg2Img) {
                    Text(replaces).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                if receipt.changesModel, let target = model.target {
                    Text("The rack will switch to \(target.name).").font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack {
                Spacer()
                if model.carried { Label("Carried", systemImage: "checkmark").foregroundStyle(.secondary) }
                Button("Carry to the Rack") { model.carry() }
                    .controlSize(.large)
                    .disabled(!receipt.canCarry || model.isWriting)
            }
        }
    }

    /// The images in their order, then the format image 1 gives.
    private func imagesLine(_ carry: LabInputs.Carry<LabModel.Picture.ID>) -> some View {
        HStack(spacing: 10) {
            ForEach(Array(carry.images.enumerated()), id: \.offset) { i, image in
                HStack(spacing: 4) {
                    Text(verbatim: "\(i + 1)").monospacedDigit().foregroundStyle(.secondary)
                    thumbnail(of: image)
                        .frame(width: 22, height: 22).clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                }
                .help(Text(verbatim: description(of: image, number: i + 1)))
            }
            if model.startIsImg2Img {
                Text("Strength \(StartWords.percent(model.startStrength))").monospacedDigit().foregroundStyle(.secondary)
            }
            Spacer(minLength: 4)
            if let f = model.outputFormat {
                Text(verbatim: "\(f.width) × \(f.height)").monospacedDigit().foregroundStyle(.secondary)
                    .help(model.startIsImg2Img ? "The size follows the starting image’s proportions; it stays free in the rack, which keeps the centre."
                                               : "Image 1 gives the format.")
            } else {
                Text("Image 1 gives the format.").foregroundStyle(.secondary)
            }
        }
        .font(.caption)
    }

    @ViewBuilder private func thumbnail(of image: LabInputs.Image<LabModel.Picture.ID>) -> some View {
        switch image {
        case .plain(let id, _):
            if let p = model.picture(id) { Image(nsImage: p.thumbnail).resizable().scaledToFill() } else { Color.secondary.opacity(0.2) }
        case .subject(let p):
            let index = p.subject.pictures.firstIndex(of: p.picture) ?? 0
            if let t = model.subjects.thumbnail(p.subject, picture: index) {
                Image(nsImage: t).resizable().scaledToFill()
            } else {
                p.subject.kind.color.opacity(0.2)
            }
        }
    }

    private func description(of image: LabInputs.Image<LabModel.Picture.ID>, number: Int) -> String {
        switch image {
        case .plain(_, let role): "\(number) · \(role.label)"
        case .subject(let p): "\(number) · @\(p.subject.name) — \(p.subject.kind.label)"
        }
    }

    /// What « Carry to the Rack » replaces, in one sentence; the images of the rack have no undo, and the sentence says so. `start`:
    /// the image sent is an img2img's starting image.
    static func replaces(_ r: LabInputs.Receipt<LabModel.Picture.ID>, start: Bool = false) -> String? {
        guard case .carries(let carry) = r.outcome else { return nil }
        let sendsImages = !carry.images.isEmpty
        if start, sendsImages {
            switch (r.replacesPrompt, r.replacedImages > 0) {
            case (true, true): return String(localized: "Replaces the rack’s prompt (⌘Z brings it back) and its starting image (it can’t be brought back).")
            case (true, false): return String(localized: "Replaces the rack’s prompt (⌘Z brings it back) and gives it a starting image.")
            case (false, true): return String(localized: "Replaces the rack’s starting image. It can’t be brought back.")
            case (false, false): return String(localized: "Gives the rack a starting image.")
            }
        }
        switch (r.replacesPrompt, sendsImages, r.replacedImages > 0) {
        case (true, false, _): return String(localized: "Replaces the rack’s prompt. ⌘Z in its field brings it back.")
        case (false, true, true): return String(localized: "Replaces the rack’s images. They can’t be brought back.")
        case (true, true, true): return String(localized: "Replaces the rack’s prompt (⌘Z brings it back) and its images (they can’t be brought back).")
        case (true, true, false): return String(localized: "Replaces the rack’s prompt (⌘Z brings it back) and fills its images.")
        case (false, true, false): return String(localized: "Fills the rack’s images.")
        case (false, false, _): return String(localized: "The rack already holds this.")
        }
    }

    // MARK: The diff, drawn

    /// Additions in the accent on a faint wash (a colour alone would not do for everyone), removals
    /// struck through and quiet. A removed piece's leading space stays outside the strike.
    static func shown(_ pieces: [PromptDiff.Piece]) -> AttributedString {
        var out = AttributedString()
        for piece in pieces {
            switch piece.kind {
            case .same:
                out += AttributedString(piece.text)
            case .added:
                var a = AttributedString(piece.text)
                a.foregroundColor = Palette.live
                a.backgroundColor = Palette.live.opacity(0.12)
                out += a
            case .removed:
                let lead = piece.text.prefix { $0.isWhitespace }
                out += AttributedString(String(lead))
                var a = AttributedString(String(piece.text.dropFirst(lead.count)))
                a.strikethroughStyle = .single
                a.foregroundColor = Color.secondary
                out += a
            }
        }
        return out
    }
}

// MARK: - The refusals, in the user's language

extension LabInputs.Refusal {
    var sentence: String {
        switch self {
        case .subject(let failure):
            Problem(failure).title
        case let .tooManyImages(count, max):
            String(localized: "The lab sends \(String(count)) images: a render takes \(String(max)) at most. Set some to Describe only.")
        case .mentionInGroup(let name):
            String(localized: "@\(name) can’t be cited inside a {…|…} group: move it out of the braces.")
        }
    }
}
