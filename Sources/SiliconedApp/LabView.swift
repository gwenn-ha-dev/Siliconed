// **The Prompt Lab's bar, under the image** (`Lab.swift` holds the state): the rack's text and images (`LabImages.swift`) —
// one text, one set of images, nothing to carry.
//
// - **Folded**: one line under the image, in a room of its own (`foldedRoom`) — the images that go, as small thumbnails, the
//   text cut to the line, Rewrite (✦), the time and « Generate » (`LabGenerate.swift`), the chevron. A click on the text
//   unfolds it.
// - **Unfolded**: a plate of glass, its edges fading (`vapour`), that rises over the image (`MainView`), never pushing it: pushed, the window took
//   the image's height plus the bar's and grew past the screen. Its title names the model it writes for — the menu of the
//   model is the rack's, one place —, its foot holds the cost, « Explore » and « Generate ». A click on the image folds it.
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
        .vapour(cornerRadius: 16)
        .overlay {
            if targeted { RoundedRectangle(cornerRadius: 16, style: .continuous).strokeBorder(Color.accentColor, lineWidth: 2) }
        }
        .dropDestination(for: DroppedImage.self) { items, _ in
            guard !items.isEmpty else { return false }
            for item in items { model.load(item) }
            return true
        } isTargeted: { targeted = $0 }
        .sheet(item: $model.draft, onDismiss: { model.sheetClosed() }) { draft in SubjectSheet(lab: model, draft: draft) }
        .onAppear { model.watchRender(); model.refresh() }
    }

    /// What the model reads of the text — the field greys the rest. Refused (too many images): no reference greyed.
    private var reading: PromptReading {
        let rack = model.rack
        let carried = (try? rack.carried.get()).map { carry in
            carry.images.filter { if case .plain(_, .start) = $0 { false } else { true } }.count
        } ?? Int.max
        return PromptReading(capacity: rack.imageCapacity, carried: carried, model: rack.card?.name ?? rack.identifier)
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
            GenerateControls(app: model.rack, compact: true)
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
            Divider().opacity(0.6)
            GenerateControls(app: model.rack)
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
                        reading: reading)
                .opacity(model.changes == nil ? 1 : 0)
            if model.text.isEmpty {
                Text("Type or paste a prompt. Write @ to cite a subject.")
                    .foregroundStyle(.tertiary).padding(.leading, 5).padding(.top, 3)
                    .allowsHitTesting(false)
            }
            if let changes = model.changes {
                // The marks lie over the text, which stays under them: ⌘Z still undoes the gesture. While the writer writes,
                // they are the text it has written so far, against the one it started from, and the last words stay in sight;
                // they scroll by hand all along. Written, a click puts them away and the caret goes into the text: it is
                // always yours to edit.
                MarkedText(pieces: PromptDiff.pieces(from: changes.before, to: changes.after), writing: model.isWriting) {
                    guard !model.isWriting else { return }
                    model.dismissChanges()
                    if let field = PromptTextView.current { field.window?.makeFirstResponder(field) }
                }
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

}

// MARK: - The writer's marks

/// **The writer's marks, over the text**: the text it is writing (or wrote), additions in the accent on a faint wash (a colour
/// alone would not do for everyone), removals struck through and quiet — in a text view of its own, read only, with the field's
/// font and line spacing, so that it scrolls as any text does, by the wheel or two fingers, while the writer writes too. A
/// SwiftUI `ScrollView` took no scroll while it wrote (it took no hit, so that a click could not cut the gesture short) and
/// did not follow the words: a long description went past the field's foot, out of sight until the text was edited.
///
/// **While it writes, the end of what it has added stays in sight** — not the text's end: Describe writes at the caret,
/// which may be in the middle. A hand that scrolls away holds the view where it put it, until it comes back to that end.
private struct MarkedText: NSViewRepresentable {
    let pieces: [PromptDiff.Piece]
    let writing: Bool
    let click: () -> Void

    @MainActor final class Coordinator {
        var shown: [PromptDiff.Piece] = []
        /// The hand scrolled away from the words coming: the view stays where it put it.
        var held = false
        /// The view scrolls itself: not the hand.
        var following = false
        var observer: NSObjectProtocol?
    }

    /// A click goes to the gesture, not to a selection: the view is the field's picture, not a field.
    final class View: NSTextView {
        var clicked: (() -> Void)?
        override func mouseDown(with event: NSEvent) { clicked?() }
        override var acceptsFirstResponder: Bool { false }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: NSViewRepresentableContext<Self>) -> NSScrollView {
        let storage = NSTextStorage()
        let layout = NSLayoutManager()
        storage.addLayoutManager(layout)
        let container = NSTextContainer(size: NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude))
        container.widthTracksTextView = true
        layout.addTextContainer(container)
        let view = View(frame: .zero, textContainer: container)
        view.isEditable = false
        view.isSelectable = false
        view.drawsBackground = false
        view.isVerticallyResizable = true
        view.isHorizontallyResizable = false
        view.autoresizingMask = [.width]
        view.minSize = .zero
        view.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        // The field's own inset (`PromptField`): the words do not move when the marks give way to the text.
        view.textContainerInset = NSSize(width: 0, height: 3)
        let scroll = NSScrollView()
        scroll.documentView = view
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        // Any scroll the view did not do itself is the hand's — the wheel's too, which sends no live-scroll notification.
        let clip = scroll.contentView
        clip.postsBoundsChangedNotifications = true
        let c = context.coordinator
        c.observer = NotificationCenter.default.addObserver(forName: NSView.boundsDidChangeNotification, object: clip,
                                                            queue: .main) { [weak clip, weak view] _ in
            MainActor.assumeIsolated {
                guard !c.following, let clip, let view else { return }
                c.held = clip.bounds.maxY < view.frame.height - 4
            }
        }
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: NSViewRepresentableContext<Self>) {
        guard let view = scroll.documentView as? View else { return }
        view.clicked = click
        let c = context.coordinator
        guard c.shown != pieces else { return }
        c.shown = pieces
        view.textStorage?.setAttributedString(Self.drawn(pieces))
        guard writing, !c.held else { return }
        c.following = true
        view.layoutManager?.ensureLayout(for: view.textContainer!)
        view.scrollRangeToVisible(NSRange(location: Self.endOfAdded(pieces), length: 0))
        c.following = false
    }

    static func dismantleNSView(_ scroll: NSScrollView, coordinator: Coordinator) {
        coordinator.observer.map(NotificationCenter.default.removeObserver)
        coordinator.observer = nil
    }

    /// Where the last addition ends, in the drawn text (struck removals included): the place the writer is writing.
    static func endOfAdded(_ pieces: [PromptDiff.Piece]) -> Int {
        var at = 0, end = 0
        for piece in pieces {
            at += (piece.text as NSString).length
            if piece.kind == .added { end = at }
        }
        return end == 0 ? at : end
    }

    /// A removed piece's leading space stays outside the strike.
    static func drawn(_ pieces: [PromptDiff.Piece]) -> NSAttributedString {
        let out = NSMutableAttributedString()
        let base = PromptStyle.base
        for piece in pieces {
            switch piece.kind {
            case .same:
                out.append(NSAttributedString(string: piece.text, attributes: base))
            case .added:
                var a = base
                a[.foregroundColor] = NSColor.controlAccentColor
                a[.backgroundColor] = NSColor.controlAccentColor.withAlphaComponent(0.12)
                out.append(NSAttributedString(string: piece.text, attributes: a))
            case .removed:
                let lead = String(piece.text.prefix { $0.isWhitespace })
                out.append(NSAttributedString(string: lead, attributes: base))
                var a = base
                a[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
                a[.foregroundColor] = NSColor.secondaryLabelColor
                out.append(NSAttributedString(string: String(piece.text.dropFirst(lead.count)), attributes: a))
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
