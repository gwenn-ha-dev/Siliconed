// **The prompt field when named subjects are on: `@alice` drawn as a chip, in the text itself.**
//
//     typed      woman posing with @alice in @red-coat and @bob
//     drawn      woman posing with (▣ alice) in (▣ red-coat) and @bob
//                                  └ lavender   └ lavender       └ unknown: dotted underline
//
// **The text stays text.** A chip is the characters `@alice`, styled — not an attachment: copying gives
// `@alice`, pasting `@alice` makes a chip, the caret and the selection go through it as through a word,
// undo is the text's, and VoiceOver reads what is written. The `@` is not drawn: its advance is widened
// (`.kern`) to make room for the thumbnail, drawn there by the layout manager with the chip's fill and
// edge (`SubjectLayoutManager`). The view is plain text (`isRichText = false`): the pasteboard never
// carries the styling.
//
// **Completion at `@`**: typing `@` and letters opens a list of the subjects whose name starts so,
// thumbnails first, under the caret (`SubjectCompletion`); ↑ ↓ choose, Return or Tab take, Esc closes.
// The window's key monitor leaves Esc to it while it is open (`SubjectCompletion.isOpen`).
//
// TextKit 1 on purpose: `drawBackground` / `drawGlyphs` of an `NSLayoutManager` subclass are the one
// stable place to draw behind and above given glyphs, and a prompt is a few hundred characters.

import AppKit
import Siliconed
import SwiftUI

extension NSAttributedString.Key {
    /// A known subject's chip: `<kind>:<key>` (`NamedSubjects.key`), read by the layout manager.
    static let subjectChip = NSAttributedString.Key("SiliconedSubjectChip")
}

/// **The field, for SwiftUI**: the same place, size and font as the `TextEditor` it replaces.
struct PromptField: NSViewRepresentable {
    let app: AppState
    @Binding var focused: Bool

    func makeCoordinator() -> Coordinator { Coordinator(app: app, focused: $focused) }

    func makeNSView(context: NSViewRepresentableContext<Self>) -> NSScrollView {
        let storage = NSTextStorage()
        let layout = SubjectLayoutManager()
        storage.addLayoutManager(layout)
        let container = NSTextContainer(size: NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude))
        container.widthTracksTextView = true
        layout.addTextContainer(container)
        let view = PromptTextView(frame: .zero, textContainer: container)
        view.isRichText = false
        view.importsGraphics = false
        view.allowsUndo = true
        view.drawsBackground = false
        view.isVerticallyResizable = true
        view.isHorizontallyResizable = false
        view.autoresizingMask = [.width]
        view.minSize = .zero
        view.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        // A chip reaches 2 pt above and below its line (`SubjectLayoutManager.chips`): room for the first.
        view.textContainerInset = NSSize(width: 0, height: 3)
        view.font = PromptStyle.font
        view.typingAttributes = PromptStyle.base
        view.delegate = context.coordinator
        view.focusChanged = { [weak coordinator = context.coordinator] typing in
            coordinator?.focused.wrappedValue = typing
            // The list leaves with the keyboard: left open, it would keep Esc from every other field.
            if !typing { coordinator?.completion.close() }
        }
        view.setAccessibilityLabel(String(localized: "Prompt"))
        view.setAccessibilityHelp(String(localized: "Write @ and a subject’s name to cite it."))
        layout.thumbnail = { [weak coordinator = context.coordinator] key in coordinator?.thumbnail(key) }
        context.coordinator.view = view
        PromptTextView.current = view

        let scroll = NSScrollView()
        scroll.documentView = view
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        view.string = app.prompt
        context.coordinator.restyle()
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: NSViewRepresentableContext<Self>) {
        let c = context.coordinator
        guard let view = c.view else { return }
        // Read here, so that a change of either redraws: the text set elsewhere (« Reuse These
        // Settings », the remote control), a subject created, renamed or deleted.
        let text = app.prompt, version = app.subjects.version
        if view.string != text, !view.hasMarkedText() {
            // Through the text system, as a paste would be: one undo step. `string =` would leave the
            // undo stack holding ranges of the old text — ⌘Z would then undo into the wrong place.
            let selection = view.selectedRange()
            let all = NSRange(location: 0, length: (view.string as NSString).length)
            if view.shouldChangeText(in: all, replacementString: text) {
                view.textStorage?.replaceCharacters(in: all, with: text)
                view.didChangeText()
            } else {
                view.string = text
                view.undoManager?.removeAllActions(withTarget: view)
            }
            view.setSelectedRange(NSRange(location: min(selection.location, (text as NSString).length), length: 0))
            // A text set from outside is not being typed: no completion list for it.
            c.completion.close()
            c.restyle()
        } else if c.styledVersion != version, !view.hasMarkedText() {
            c.restyle()
        }
        c.styledVersion = version
    }

    static func dismantleNSView(_ scroll: NSScrollView, coordinator: Coordinator) {
        coordinator.completion.close()
        if PromptTextView.current === coordinator.view { PromptTextView.current = nil }
    }

    @MainActor
    final class Coordinator: NSObject, NSTextViewDelegate {
        let app: AppState
        var focused: Binding<Bool>
        weak var view: PromptTextView?
        var styledVersion = -1
        let completion = SubjectCompletion()

        init(app: AppState, focused: Binding<Bool>) {
            self.app = app
            self.focused = focused
        }

        func thumbnail(_ key: String) -> NSImage? {
            app.subjects.all.first { $0.id == key }.flatMap { app.subjects.thumbnail($0) }
        }

        // ── the text changes ──

        func textDidChange(_ notification: Notification) {
            guard let view else { return }
            if app.prompt != view.string { app.prompt = view.string }
            // A character being composed (a dead key: ^ then e on an AZERTY keyboard) keeps its
            // marked-text styling until it is committed.
            guard !view.hasMarkedText() else { return }
            restyle()
            complete()
        }

        func textViewDidChangeSelection(_ notification: Notification) {
            // The style of the character before the caret must not leak into what is typed next.
            view?.typingAttributes = PromptStyle.base
            if completion.isShown { complete() }
        }

        func textView(_ textView: NSTextView, shouldChangeTextIn range: NSRange, replacementString: String?) -> Bool {
            textView.typingAttributes = PromptStyle.base
            return true
        }

        /// **The chips**: every mention restyled — known ones as chips, unknown ones underlined.
        /// Attributes only: no undo step, no text change.
        func restyle() {
            guard let view, let storage = view.textStorage else { return }
            let text = storage.string
            let all = NSRange(location: 0, length: storage.length)
            storage.beginEditing()
            storage.setAttributes(PromptStyle.base, range: all)
            var names: [String] = []
            for m in NamedSubjects.mentions(in: text) {
                let r = NSRange(m.range, in: text)
                if let s = app.subjects.subject(named: m.name) {
                    names.append(contentsOf: s.name.split(whereSeparator: { $0 == "-" || $0 == "_" }).map(String.init))
                    storage.addAttributes([.subjectChip: "\(s.kind.rawValue):\(s.id)",
                                           .toolTip: "@\(s.name) — \(s.kind.label)"], range: r)
                    // The `@`, not drawn: its advance makes the thumbnail's room.
                    storage.addAttributes([.foregroundColor: NSColor.clear, .kern: PromptStyle.atKern],
                                          range: NSRange(location: r.location, length: 1))
                    storage.addAttribute(.kern, value: PromptStyle.trailing, range: NSRange(location: r.upperBound - 1, length: 1))
                } else {
                    storage.addAttributes([.underlineStyle: NSUnderlineStyle.single.rawValue | NSUnderlineStyle.patternDot.rawValue,
                                           .underlineColor: NSColor.systemRed.withAlphaComponent(0.75),
                                           .foregroundColor: NSColor.secondaryLabelColor,
                                           .toolTip: String(localized: "No subject is called @\(m.name).")], range: r)
                }
            }
            storage.endEditing()
            view.typingAttributes = PromptStyle.base
            // A subject's name is no spelling mistake.
            NSSpellChecker.shared.setIgnoredWords(names, inSpellDocumentWithTag: view.spellCheckerDocumentTag)
        }

        // ── completion ──

        /// The mention being typed: `@` and name characters up to the caret, `@` after a space or at
        /// the start — its range and what follows the `@`.
        private func typedMention() -> (range: NSRange, prefix: String)? {
            guard let view, view.selectedRange().length == 0 else { return nil }
            let text = view.string as NSString
            let caret = view.selectedRange().location
            var i = caret
            while i > 0 {
                let c = text.substring(with: NSRange(location: i - 1, length: 1))
                if c == "@" {
                    let before = i - 1 > 0 ? text.substring(with: NSRange(location: i - 2, length: 1)) : " "
                    guard !(before.first.map { $0.isLetter || $0.isNumber || $0 == "@" } ?? false) else { return nil }
                    return (NSRange(location: i - 1, length: caret - i + 1), text.substring(with: NSRange(location: i, length: caret - i)))
                }
                guard let ch = c.first, ch.isLetter || ch.isNumber || ch == "-" || ch == "_" else { return nil }
                i -= 1
            }
            return nil
        }

        func complete() {
            guard let view, view.window != nil, let typed = typedMention() else { completion.close(); return }
            let key = NamedSubjects.key(typed.prefix)
            let matches = app.subjects.all.filter { $0.id.hasPrefix(key) && $0.id != key }
                + app.subjects.all.filter { $0.id == key }
            guard !matches.isEmpty, !(matches.count == 1 && matches[0].id == key) else { completion.close(); return }
            let caret = view.firstRect(forCharacterRange: NSRange(location: typed.range.location, length: 0), actualRange: nil)
            completion.show(matches.prefix(8).map { s in (s, app.subjects.thumbnail(s)) }, under: caret, parent: view.window!) { [weak self] s in
                self?.accept(s, replacing: typed.range)
            }
        }

        private func accept(_ s: NamedSubjects.Subject, replacing range: NSRange) {
            guard let view else { return }
            completion.close()
            // The range is re-read: the click came later than the keystroke that opened the list.
            let r = typedMention()?.range ?? range
            view.insertText("@\(s.name) ", replacementRange: r)
        }

        func textView(_ textView: NSTextView, doCommandBy selector: Selector) -> Bool {
            guard completion.isShown else { return false }
            switch selector {
            case #selector(NSResponder.moveDown(_:)): completion.move(1)
            case #selector(NSResponder.moveUp(_:)): completion.move(-1)
            case #selector(NSResponder.insertNewline(_:)), #selector(NSResponder.insertTab(_:)): completion.accept()
            // Esc reaches a text view as `cancelOperation:` or, from its own binding, `complete:`.
            case #selector(NSResponder.cancelOperation(_:)), #selector(NSTextView.complete(_:)): completion.close()
            default: return false
            }
            return true
        }
    }
}

/// The field's type: its metrics, and how much room the `@` makes for the thumbnail.
enum PromptStyle {
    nonisolated(unsafe) static let font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
    /// A little air between lines: two chips on consecutive lines do not touch.
    nonisolated(unsafe) static let paragraph: NSParagraphStyle = {
        let p = NSMutableParagraphStyle()
        p.lineSpacing = 4
        return p
    }()
    nonisolated(unsafe) static let base: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.textColor, .paragraphStyle: paragraph]
    /// The thumbnail's side, and the chip's height around the text.
    static let thumbnail: CGFloat = 14
    static let leading: CGFloat = 2, gap: CGFloat = 4, trailing: CGFloat = 5
    /// The `@`'s added advance: its own glyph is invisible, the thumbnail and its margins take its place.
    static let atKern: CGFloat = {
        let at = ("@" as NSString).size(withAttributes: [.font: font]).width
        return leading + thumbnail + gap - at
    }()
}

/// **The text view**: it says when it takes or loses the keyboard (the box's edge follows), and
/// inserts a mention where the caret is (a click on a chip in the subjects shelf).
final class PromptTextView: NSTextView {
    /// The field on screen — at most one: the rack has one prompt.
    @MainActor static weak var current: PromptTextView?
    var focusChanged: ((Bool) -> Void)?

    override func becomeFirstResponder() -> Bool {
        let ok = super.becomeFirstResponder()
        if ok { focusChanged?(true) }
        return ok
    }

    override func resignFirstResponder() -> Bool {
        let ok = super.resignFirstResponder()
        if ok { focusChanged?(false) }
        return ok
    }

    func insertMention(_ name: String) {
        window?.makeFirstResponder(self)
        let r = selectedRange()
        let text = string as NSString
        let glue = r.location > 0 && !(text.substring(with: NSRange(location: r.location - 1, length: 1)).first?.isWhitespace ?? true)
        insertText((glue ? " " : "") + "@\(name) ", replacementRange: r)
    }
}

/// **Draws the chips**: the fill behind the text, the thumbnail and the edge above it — the edge after
/// the selection, so a selected chip stays recognizable.
final class SubjectLayoutManager: NSLayoutManager {
    var thumbnail: ((String) -> NSImage?)?

    override func drawBackground(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
        chips(in: glyphsToShow, at: origin) { rect, _, color, _ in
            color.withAlphaComponent(0.14).setFill()
            NSBezierPath(roundedRect: rect, xRadius: rect.height / 2, yRadius: rect.height / 2).fill()
        }
        super.drawBackground(forGlyphRange: glyphsToShow, at: origin)
    }

    override func drawGlyphs(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
        super.drawGlyphs(forGlyphRange: glyphsToShow, at: origin)
        chips(in: glyphsToShow, at: origin) { rect, key, color, first in
            let edge = NSBezierPath(roundedRect: rect.insetBy(dx: 0.5, dy: 0.5), xRadius: rect.height / 2 - 0.5, yRadius: rect.height / 2 - 0.5)
            edge.lineWidth = 1
            color.setStroke()
            edge.stroke()
            guard first else { return }
            let side = PromptStyle.thumbnail
            let box = NSRect(x: rect.minX + PromptStyle.leading, y: rect.midY - side / 2, width: side, height: side)
            NSGraphicsContext.saveGraphicsState()
            NSBezierPath(roundedRect: box, xRadius: 3.5, yRadius: 3.5).addClip()
            if let image = self.thumbnail?(key) {
                // Aspect fill: the frame's middle, the thumbnail's square.
                let s = max(side / image.size.width, side / image.size.height)
                let w = image.size.width * s, h = image.size.height * s
                image.draw(in: NSRect(x: box.midX - w / 2, y: box.midY - h / 2, width: w, height: h),
                           from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
            } else {
                color.withAlphaComponent(0.5).setFill()
                box.fill()
            }
            NSGraphicsContext.restoreGraphicsState()
        }
    }

    /// Each chip's rectangles in the glyphs shown (a chip that wraps has one per line), centered on
    /// the text's line whatever the line spacing.
    private func chips(in glyphs: NSRange, at origin: NSPoint, _ draw: (NSRect, String, NSColor, Bool) -> Void) {
        guard let storage = textStorage, let container = textContainers.first else { return }
        let characters = characterRange(forGlyphRange: glyphs, actualGlyphRange: nil)
        let font = PromptStyle.font
        var found: [(NSRect, String, NSColor, Bool)] = []
        storage.enumerateAttribute(.subjectChip, in: characters) { value, range, _ in
            guard let tag = value as? String, let colon = tag.firstIndex(of: ":") else { return }
            let key = String(tag[tag.index(after: colon)...])
            let color = (NamedSubjects.Kind(rawValue: String(tag[..<colon])) ?? .person).nsColor
            let glyphRange = glyphRange(forCharacterRange: range, actualCharacterRange: nil)
            var first = true
            enumerateLineFragments(forGlyphRange: glyphRange) { lineRect, _, _, lineGlyphs, _ in
                let part = NSIntersectionRange(lineGlyphs, glyphRange)
                guard part.length > 0 else { return }
                let bounds = self.boundingRect(forGlyphRange: part, in: container)
                let baseline = lineRect.minY + self.location(forGlyphAt: part.location).y
                let top = baseline - font.ascender - 2, bottom = baseline - font.descender + 2
                let rect = NSRect(x: bounds.minX + origin.x, y: top + origin.y,
                                  width: bounds.width, height: bottom - top)
                found.append((rect, key, color, first))
                first = false
            }
        }
        for (rect, key, color, first) in found { draw(rect, key, color, first) }
    }
}

// MARK: - Completion

/// **The list under the caret**: the subjects whose name starts with what follows the `@`, each with
/// its thumbnail and kind. A borderless child window that never takes the keyboard: the text view
/// keeps it, and forwards ↑ ↓ Return Tab Esc.
@MainActor
final class SubjectCompletion {
    static private(set) var isOpen = false
    private var panel: NSPanel?
    private let model = Model()

    @MainActor @Observable
    final class Model {
        var items: [(subject: NamedSubjects.Subject, thumbnail: NSImage?)] = []
        var selected = 0
        var take: ((NamedSubjects.Subject) -> Void)?
    }

    var isShown: Bool { panel?.isVisible ?? false }

    func show(_ items: [(NamedSubjects.Subject, NSImage?)], under caret: NSRect, parent: NSWindow,
              take: @escaping (NamedSubjects.Subject) -> Void) {
        let previous = model.items.map(\.subject.id)
        model.items = items.map { (subject: $0.0, thumbnail: $0.1) }
        if previous != model.items.map(\.subject.id) { model.selected = 0 }
        model.take = take
        let p = panel ?? makePanel()
        let height = CGFloat(items.count) * 30 + 10
        p.setContentSize(NSSize(width: 240, height: height))
        p.setFrameTopLeftPoint(NSPoint(x: caret.minX - 6, y: caret.minY - 4))
        if p.parent == nil { parent.addChildWindow(p, ordered: .above) }
        p.orderFront(nil)
        Self.isOpen = true
    }

    func move(_ by: Int) {
        guard !model.items.isEmpty else { return }
        model.selected = (model.selected + by + model.items.count) % model.items.count
    }

    func accept() {
        guard model.items.indices.contains(model.selected) else { close(); return }
        model.take?(model.items[model.selected].subject)
    }

    func close() {
        if let p = panel {
            p.parent?.removeChildWindow(p)
            p.orderOut(nil)
        }
        Self.isOpen = false
    }

    private func makePanel() -> NSPanel {
        let p = NSPanel(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        p.isFloatingPanel = true
        p.hasShadow = true
        p.backgroundColor = .clear
        p.isOpaque = false
        p.isReleasedWhenClosed = false
        p.contentView = NSHostingView(rootView: CompletionList(model: model))
        panel = p
        return p
    }
}

private struct CompletionList: View {
    let model: SubjectCompletion.Model

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(Array(model.items.enumerated()), id: \.element.subject.id) { i, item in
                HStack(spacing: 8) {
                    SubjectChip(name: item.subject.name, kind: item.subject.kind, thumbnail: item.thumbnail,
                                pictures: item.subject.pictures.count)
                    Spacer(minLength: 4)
                    Text(item.subject.kind.label).font(.caption).foregroundStyle(.secondary)
                }
                .padding(.horizontal, 6).frame(height: 28)
                .background(RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(i == model.selected ? Color.accentColor.opacity(0.22) : .clear))
                .contentShape(Rectangle())
                .onTapGesture { model.take?(item.subject) }
                .onHover { if $0 { model.selected = i } }
            }
        }
        .padding(5)
        .frame(width: 240, alignment: .topLeading)
        .background(VisualEffect().clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous)))
        .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(Palette.hairline))
    }
}

/// The menu's material behind the list.
private struct VisualEffect: NSViewRepresentable {
    func makeNSView(context: NSViewRepresentableContext<Self>) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = .menu
        v.blendingMode = .behindWindow
        v.state = .active
        return v
    }
    func updateNSView(_ nsView: NSVisualEffectView, context: NSViewRepresentableContext<Self>) {}
}
