// **The Prompt Lab's bar, under the image** (`Lab.swift` holds the state): the rack's text and images (`LabImages.swift`) —
// one text, one set of images, nothing to carry.
//
// - **Folded**: one line under the image, in a room of its own (`foldedRoom`) — the images that go, as small thumbnails, the
//   text cut to the line, Rewrite (✦) and the chevron. A click on the text unfolds it.
// - **Unfolded**: a glass plate that rises over the image (`MainView`), never pushing it: pushed, the window took the image's
//   height plus the bar's and grew past the screen. Its title names the model it writes for — the menu of the model is the
//   rack's, one place. A click on the image folds it.
// - **Rewrite and Describe wait for a render** (the render first), and the folded line says so.

import AppKit
import Siliconed
import SwiftUI

struct LabView: View {
    @Bindable var model: LabModel
    @State private var typing = false
    @State private var targeted = false

    /// The folded line's height, and the room the canvas keeps for it under the image (with the margins `MainView` gives it).
    static let foldedHeight: CGFloat = 40
    static let foldedRoom = foldedHeight + 8 + 12

    var body: some View {
        Group {
            if model.unfolded {
                // Its own height, whatever the canvas offers: it lies over the image, which keeps its size.
                unfoldedBar.padding(.horizontal, 12).padding(.vertical, 10).fixedSize(horizontal: false, vertical: true)
            } else {
                foldedBar.padding(.horizontal, 12).frame(height: Self.foldedHeight)
            }
        }
        .glass(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay {
            if targeted { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Color.accentColor, lineWidth: 2) }
        }
        .dropDestination(for: DroppedImage.self) { items, _ in
            guard !items.isEmpty else { return false }
            for item in items { model.load(item) }
            return true
        } isTargeted: { targeted = $0 }
        .sheet(item: $model.draft, onDismiss: { model.sheetClosed() }) { draft in SubjectSheet(lab: model, draft: draft) }
        .onAppear { model.watchRender(); model.refresh() }
    }

    /// The model the text is written for: the rack's card, else its family.
    private var modelName: String? { model.rack.card?.name ?? model.target?.name }

    // MARK: Folded

    private var foldedBar: some View {
        HStack(spacing: 10) {
            sentThumbnails
            Button { model.unfold() } label: {
                Group {
                    if model.text.isEmpty {
                        Text("Type the idea… (⌥⌘L)").foregroundStyle(.tertiary)
                    } else {
                        Text(verbatim: model.text.replacingOccurrences(of: "\n", with: " "))
                    }
                }
                .lineLimit(1).truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Unfold the Prompt Lab (⌥⌘L)")
            if let pause = model.writerPause, model.rendering {
                Text(pause.sentence).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Button { model.unfold(); model.rewrite() } label: {
                Image(systemName: "sparkles").foregroundStyle(WireKind.text.color)
            }
            .buttonStyle(.bordered).controlSize(.small)
            .disabled(!model.canRewrite)
            .help("Rewrite")
            .accessibilityLabel(Text("Rewrite"))
            chevron
        }
    }

    /// The images the next render reads, as small thumbnails — what the rack shows as its inputs.
    @ViewBuilder private var sentThumbnails: some View {
        let rack = model.rack
        let shown = rack.references.prefix(3) + (rack.usedStart.map { [$0] } ?? [])
        if !shown.isEmpty {
            HStack(spacing: 4) {
                ForEach(Array(shown)) { image in
                    Image(nsImage: image.vignette).resizable().scaledToFill()
                        .frame(width: 18, height: 24).clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 3, style: .continuous).strokeBorder(WireKind.image.color, lineWidth: 1))
                }
                if rack.usedStart != nil {
                    Text(verbatim: StartWords.percent(rack.startStrength)).font(.caption).monospacedDigit().foregroundStyle(.secondary)
                }
            }
        }
    }

    private var chevron: some View {
        Button { model.toggle() } label: {
            Image(systemName: model.unfolded ? "chevron.down" : "chevron.up").font(.system(size: 11, weight: .bold))
                .frame(width: 18, height: 18)
        }
        .buttonStyle(.bordered).controlSize(.small)
        .help(model.unfolded ? "Fold the Prompt Lab (⌥⌘L)" : "Unfold the Prompt Lab (⌥⌘L)")
        .accessibilityLabel(Text(model.unfolded ? "Fold the Prompt Lab" : "Unfold the Prompt Lab"))
    }

    // MARK: Unfolded

    private var unfoldedBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            textZone
            LabImages(model: model)
            status
            if case .failure(let refusal) = model.rack.carried {
                Label(refusal.sentence, systemImage: "exclamationmark.circle").font(.caption).foregroundStyle(Palette.failure)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text("Prompt Lab").font(.callout.weight(.semibold))
            if let modelName { Text("Write for \(modelName)").font(.caption).foregroundStyle(.secondary) }
            Spacer(minLength: 8)
            if model.isWriting {
                Button("Stop") { model.stop() }.controlSize(.small)
            } else {
                Button { model.rewrite() } label: {
                    Label("Rewrite", systemImage: "sparkles")
                }
                .controlSize(.small)
                .tint(WireKind.text.color)
                .disabled(!model.canRewrite)
                .help(model.writerPause.map { Text(verbatim: $0.sentence) } ?? Text("Rewrite"))
            }
            chevron
        }
    }

    private var card: some View {
        RoundedRectangle(cornerRadius: 8).fill(Palette.well)
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(typing ? Palette.amber : .clear, lineWidth: 1))
    }

    // MARK: The text

    private var textZone: some View {
        ZStack(alignment: .topLeading) {
            PromptField(text: $model.text, subjects: model.subjects, focused: $typing, editable: !model.isWriting,
                        inert: model.rack.imageCapacity == 0
                            ? String(localized: "\(model.rack.card?.name ?? "") does not read subjects") : nil)
                .opacity(model.changes == nil ? 1 : 0)
            if model.text.isEmpty {
                Text("Type or paste a prompt. Write @ to cite a subject.")
                    .foregroundStyle(.tertiary).padding(.leading, 5).padding(.top, 3)
                    .allowsHitTesting(false)
            }
            if let changes = model.changes {
                // The marks lie over the text, which stays under them: ⌘Z still undoes the gesture. While the writer writes,
                // they are the text it has written so far, against the one it started from. Written, a click puts them away
                // and the caret goes into the text: it is always yours to edit.
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
                .background(Palette.well)
                .contentShape(Rectangle())
                .onTapGesture {
                    guard !model.isWriting else { return }
                    model.dismissChanges()
                    if let field = PromptTextView.current { field.window?.makeFirstResponder(field) }
                }
                .allowsHitTesting(!model.isWriting)
            }
        }
        .padding(6)
        .frame(height: 96)
        .background(card)
        .overlay(alignment: .topTrailing) {
            // A search field's clear button, where the eye looks for it; ⌘Z in the text brings the text back.
            if !model.text.isEmpty, !model.isWriting, model.changes == nil {
                Button { model.text = "" } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary) }
                    .buttonStyle(.plain).padding(6)
                    .help("Clear the prompt (⌘Z brings it back)")
                    .accessibilityLabel(Text("Clear Prompt"))
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
            String(localized: "The lab sends \(String(count)) images: a render takes \(String(max)) at most. Set some to No Role.")
        case .mentionInGroup(let name):
            String(localized: "@\(name) can’t be cited inside a {…|…} group: move it out of the braces.")
        }
    }
}
