// **The Prompt Lab, at the head of the column** (`Lab.swift` holds the state): the Text box and the Images box, the first two
// of the chain, each the column's width — one text, one set of images, nothing to carry. The text's wire passes behind the
// Images box to the text encoder (`Rack.swift`).
//
// - **The field grows with its text**, and its bottom edge is dragged: the height the hand gives is kept, the text never
//   hidden under it (issue 16 asked for height).
// - **Rewrite and Describe wait for a render** (the render first), and the box says so.

import AppKit
import Siliconed
import SwiftUI

/// **The Text box**: the model it writes for, the writer's gesture, the field, and what the writer is doing.
struct TextBox: View {
    @Bindable var model: LabModel
    @State private var typing = false
    /// The field's height as the hand left it: the least it takes, the text growing it beyond.
    @State private var kept: CGFloat = {
        let h = UserDefaults.standard.double(forKey: TextBox.heightKey)
        return h > 0 ? min(max(h, TextBox.heights.lowerBound), TextBox.heights.upperBound) : 96
    }()
    @State private var dragStart: CGFloat?

    static let heightKey = "promptHeight"
    static let heights: ClosedRange<CGFloat> = 64...640

    var body: some View {
        // The writer's gesture on the box's title line: the field starts right under it.
        InputBox(title: "Text", symbol: "text.quote", kind: .text, fed: !model.text.isEmpty,
                 explanation: Text("What the text encoder under it reads")) {
            rewriteButton
        } content: {
            VStack(alignment: .leading, spacing: 8) {
                yourInstructions
                textZone
                status
            }
        }
        .onAppear { model.watchRender(); model.refresh() }
        // A model installed or removed: the writer may come or go — the box said « No writer is installed yet » until a
        // render ended.
        .onChange(of: model.rack.readySet) { model.refresh() }
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

    @ViewBuilder private var rewriteButton: some View {
        if model.isWriting {
            Button("Stop") { model.stop() }.controlSize(.small)
        } else {
            Button { model.rewrite() } label: {
                Label("Rewrite", systemImage: "sparkles")
            }
            .controlSize(.small)
            .tint(WireKind.text.color)
            .help(modelName.map { Text("Write for \($0)") } ?? Text("Rewrite"))
            .unavailable(model.rewriteRefusal)
            .fixedSize()
        }
    }

    /// **The writer follows the user's instruction** for this model (Settings › Prompt Lab): said where it acts, a click
    /// opens it.
    @ViewBuilder private var yourInstructions: some View {
        let edited = model.editedInstructions
        if let target = model.target, let first = edited.first {
            Button(edited.count > 1 ? "Your Instructions" : "Your Instruction") {
                SettingsWindow.shared.showInstruction(target, first)
            }
            .buttonStyle(.chartLink).font(.caption)
            .help(edited.count > 1 ? String(localized: "Rewrite and Describe follow your instructions for \(target.name), not Siliconed’s.")
                  : first == .rewrite ? String(localized: "Rewrite follows your instruction for \(target.name), not Siliconed’s.")
                  : String(localized: "Describe follows your instruction for \(target.name), not Siliconed’s."))
        }
    }

    private var card: some View {
        RoundedRectangle(cornerRadius: 8).fill(Palette.well)
            .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(typing ? Palette.peach : Palette.hairline, lineWidth: 1))
    }

    // MARK: The text

    private var textZone: some View {
        ZStack(alignment: .topLeading) {
            PromptField(text: $model.text, subjects: model.subjects, focused: $typing, editable: !model.isWriting,
                        reading: reading, series: model.rack, minimumHeight: kept)
                .opacity(model.changes == nil ? 1 : 0)
            // Not under the marks: Describe writes into an empty text, which stays empty until its last word.
            if model.text.isEmpty, model.changes == nil {
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
        // The series' column, as in the rack: at the text's right, the ✕ that clears it and, under it, its ± (`PromptBracketMark`).
        .padding(.trailing, 18)
        .padding(6)
        // As tall as the text, never less than the hand left it (`PromptField.sizeThatFits`).
        .background(card)
        .overlay(alignment: .topTrailing) {
            VStack(spacing: 6) {
                // A search field's clear button, where the eye looks for it; ⌘Z in the text brings the text back.
                if !model.text.isEmpty, !model.isWriting, model.changes == nil {
                    Button { model.text = "" } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary) }
                        .buttonStyle(.plain)
                        .help("Clear the prompt (⌘Z brings it back)")
                        .accessibilityLabel(Text("Clear Prompt"))
                } else {
                    Color.clear.frame(width: 14, height: 14)
                }
                PromptBracketMark(app: model.rack, refusal: model.isWriting ? String(localized: "The writer is writing: wait for it, or stop it")
                                  : model.changes != nil ? String(localized: "Click the text first, to put the rewrite’s marks away") : nil)
            }
            .padding(.top, 7).padding(.trailing, 7)
        }
        .overlay(alignment: .bottom) { grip }
    }

    /// **The field's bottom edge, dragged**: the height it keeps from now on, as a window's edge.
    private var grip: some View {
        Capsule().fill(Palette.hairline).frame(width: 32, height: 4)
            .frame(maxWidth: .infinity).frame(height: 10)
            .contentShape(Rectangle())
            .offset(y: 5)
            .pointerStyle(.rowResize)
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { drag in
                    let start = dragStart ?? kept
                    dragStart = start
                    kept = min(max(start + drag.translation.height, Self.heights.lowerBound), Self.heights.upperBound)
                }
                .onEnded { _ in
                    dragStart = nil
                    UserDefaults.standard.set(Double(kept), forKey: Self.heightKey)
                })
            .help("Drag to make the field taller or shorter")
            .accessibilityHidden(true)
    }

    /// What the writer is doing, why it waits, what went wrong — in as many lines as it has things to say.
    @ViewBuilder private var status: some View {
        switch model.phase {
        case .loading(let fraction):
            HStack(spacing: 8) {
                if let fraction { ProgressView(value: fraction).tint(Palette.live).frame(maxWidth: 160) } else { ProgressView().controlSize(.small) }
                Text("Preparing the writer…").font(.caption).foregroundStyle(Palette.secondary)
            }
        case .writing:
            // The text itself comes in the field, marked.
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Writing…").font(.caption).foregroundStyle(Palette.secondary)
            }
        case .idle:
            EmptyView()
        }
        if let pause = model.writerPause {
            Label(pause.sentence, systemImage: pause.symbol).font(.caption).foregroundStyle(Palette.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        if let failure = model.failure {
            Text(failure).font(.caption).foregroundStyle(Palette.failure)
        } else if model.stopped {
            Text("Stopped: an image started rendering.").font(.caption).foregroundStyle(Palette.secondary)
        }
        if let notice = model.notice {
            Label(notice, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(Palette.caution)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// **The text on one line** — the head of the column, pinned while it scrolls (`Sidebar`): a `{…}` group said as the field
    /// draws its chip, its words between dots in the series' yellow, rather than its braces. The groups are the parser's
    /// (`PromptAlternatives.spans`, its words unescaped): an escaped brace stays a character, and a text that does not parse
    /// draws no group, as the field draws no chip.
    static func line(_ text: String) -> AttributedString {
        let flat = text.replacingOccurrences(of: "\n", with: " ")
        let groups = (try? PromptAlternatives(flat, limit: .max).groups) ?? []
        var out = AttributedString()
        var rest = flat.startIndex
        for (span, words) in zip(PromptAlternatives.spans(flat), groups) {
            out += AttributedString(String(flat[rest..<span.range.lowerBound]))
            var group = AttributedString(words.map { $0.trimmingCharacters(in: .whitespaces) }.joined(separator: " · "))
            group.foregroundColor = Palette.bracketText
            group.backgroundColor = Palette.bracketWash
            out += group
            rest = span.range.upperBound
        }
        out += AttributedString(String(flat[rest...]))
        return out
    }
}

/// **The Images box**: the images the next render reads, each with its role — or one to describe, or to start from. On a
/// model that edits, how many go of how many it takes.
struct ImagesBox: View {
    @Bindable var model: LabModel
    @State private var targeted = false

    var body: some View {
        let app = model.rack
        let used = app.references.count
        InputBox(title: "Images", symbol: "photo.on.rectangle", kind: .image, fed: used > 0 || app.usedStart != nil,
                 explanation: hint(app)) {
            if app.editingPossible {
                Text(verbatim: "\(used)/\(app.maxReferences)")
                    .foregroundStyle(used > app.maxReferences ? Palette.failure : Palette.secondary)
            }
        } content: {
            LabImages(model: model)
        }
        .overlay { if targeted { boxShape.strokeBorder(Palette.live, lineWidth: 2) } }
        .dropDestination(for: DroppedImage.self) { items, _ in
            guard !items.isEmpty else { return false }
            for item in items { model.load(item) }
            return true
        } isTargeted: { targeted = $0 }
        .sheet(item: $model.draft, onDismiss: { model.sheetClosed() }) { draft in SubjectSheet(lab: model, draft: draft) }
    }

    private func hint(_ app: AppState) -> Text {
        if app.usedStart != nil, !app.editingPossible {
            return Text(verbatim: app.startCaption + "\n" + StartWords.explanation(app.startRedraws))
        }
        guard app.editingPossible else { return Text("Drop an image here to describe it, or for the render to start from it.") }
        if app.references.isEmpty {
            return Text("Optional. Drop an image here and the prompt becomes an instruction: “change her jacket to red”.")
        }
        if !app.edits { return Text("No image to edit: the render makes a new image from these, at the size chosen.") }
        if app.promptNamesImages && app.maxReferences > 1 {
            return Text("Image 1 is the one edited. The prompt can name them: “put the dog of image 2 next to the woman in image 1”.")
        }
        return Text("Image 1 is the one edited: the prompt says what changes.")
    }
}

/// **A width that moves frame by frame**: animatable, the layout is made again at every frame — the canvas beside the column
/// follows the fold instead of jumping to its last place at the first frame. The rack's column folds by it (`RackColumn`).
struct SlidingWidth: ViewModifier, @preconcurrency Animatable {
    var width: CGFloat
    var animatableData: CGFloat {
        get { width }
        set { width = newValue }
    }

    func body(content: Content) -> some View { content.frame(width: width, alignment: .trailing) }
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
                a[.foregroundColor] = Palette.addedInk
                a[.backgroundColor] = Palette.addedWash
                out.append(NSAttributedString(string: piece.text, attributes: a))
            case .removed:
                let lead = String(piece.text.prefix { $0.isWhitespace })
                out.append(NSAttributedString(string: lead, attributes: base))
                var a = base
                a[.strikethroughStyle] = NSUnderlineStyle.single.rawValue
                a[.foregroundColor] = Palette.secondaryInk
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
