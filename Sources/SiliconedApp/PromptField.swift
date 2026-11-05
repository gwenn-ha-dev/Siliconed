// **The Prompt Lab's text: `@alice` drawn as a chip, in the text itself.**
//
//     typed      woman posing with @alice in @red-coat and @bob
//     drawn      woman posing with (▣ alice) in (▣ red-coat) and @bob
//                                  └ rose       └ rose           └ unknown: dotted underline
//
// **Greyed where the model does not read it** (`reading`, the rule is `LabInputs.inert`): under a model that takes no image,
// a subject's chip is drawn grey, its thumbnail faded; a reference to an image the render does not carry (`<image3>` with two)
// is grey too. The hover says why. The render leaves them out (`AppState.namedJob`); the text keeps them, for when they are
// read again.
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
// **A series' group is a chip too** (`{red|blue}`, `PromptAlternatives`): drawn in the series' yellow (`Palette.bracket`), its
// braces and bars not drawn — a dot between its words. A click on it opens its bubble (`BracketBubble`), whose words rewrite the
// group where it is typed; braces typed by hand make one as well. Words become a group by the ± at the card's right (`PromptBracketMark`: the
// selection, or the word at the caret) or by « Vary This Word… » at the head of the right-click menu — added by the delegate
// (`textView(_:menu:for:at:)`): with an override of `menu(for:)`, the user saw only the system's menu (4/11/2026). The
// text stays the characters typed, as for a subject.
//
// TextKit 1 on purpose: `drawBackground` / `drawGlyphs` of an `NSLayoutManager` subclass are the one
// stable place to draw behind and above given glyphs, and a prompt is a few hundred characters.

import AppKit
import Siliconed
import SwiftUI

extension NSAttributedString.Key {
    /// A known subject's chip: `<kind>:<key>` (`NamedSubjects.key`), read by the layout manager.
    static let subjectChip = NSAttributedString.Key("SiliconedSubjectChip")
    /// A chip the model does not read: drawn grey (`PromptField.inert`).
    static let subjectChipInert = NSAttributedString.Key("SiliconedSubjectChipInert")
    /// A series' group, from `{` to `}`: its rank in the text (`SeriesKey.group`).
    static let seriesGroup = NSAttributedString.Key("SiliconedSeriesGroup")
    /// A group's `|`: not drawn, a dot in its place.
    static let seriesSeparator = NSAttributedString.Key("SiliconedSeriesSeparator")
}

/// **The field, for SwiftUI**: the same font as a `TextEditor`.
struct PromptField: NSViewRepresentable {
    @Binding var text: String
    let subjects: SubjectLibrary
    @Binding var focused: Bool
    /// Off while the writer writes: the text it will replace does not move under it.
    var editable = true
    /// What the model aimed at reads of the text, or `nil`: nothing greyed.
    var reading: PromptReading? = nil
    /// The rack whose series the text's groups are: a group's bubble, « Vary This Word… ». `nil`: groups drawn, no bubble.
    var series: AppState? = nil

    func makeCoordinator() -> Coordinator { Coordinator(text: $text, subjects: subjects, focused: $focused) }

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
        Palette.ink(view)
        view.delegate = context.coordinator
        view.focusChanged = { [weak coordinator = context.coordinator] typing in
            coordinator?.focused.wrappedValue = typing
            // The list leaves with the keyboard: left open, it would keep Esc from every other field.
            if !typing { coordinator?.completion.close() }
        }
        view.listShown = { [weak coordinator = context.coordinator] in coordinator?.completion.isShown ?? false }
        view.setAccessibilityLabel(String(localized: "Prompt"))
        view.setAccessibilityHelp(String(localized: "Write @ and a subject’s name to cite it."))
        layout.thumbnail = { [weak coordinator = context.coordinator] key in coordinator?.thumbnail(key) }
        context.coordinator.series = series
        view.openBracket = { [weak coordinator = context.coordinator] target, rect in coordinator?.openBracket(target, at: rect) }
        context.coordinator.view = view
        PromptTextView.current = view

        let scroll = NSScrollView()
        scroll.documentView = view
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        view.string = text
        context.coordinator.restyle()
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: NSViewRepresentableContext<Self>) {
        let c = context.coordinator
        guard let view = c.view else { return }
        c.series = series
        // Read here, so that a change of either redraws: the text set elsewhere (the rack's prompt taken, a
        // rewrite), a subject created, renamed or deleted.
        let text = self.text, version = subjects.version
        if view.isEditable != editable { view.isEditable = editable }
        if c.reading != reading, !view.hasMarkedText() {
            c.reading = reading
            c.restyle()
        }
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
        var text: Binding<String>
        let subjects: SubjectLibrary
        var focused: Binding<Bool>
        weak var view: PromptTextView?
        weak var series: AppState?
        var styledVersion = -1
        var reading: PromptReading?
        let completion = SubjectCompletion()

        init(text: Binding<String>, subjects: SubjectLibrary, focused: Binding<Bool>) {
            self.text = text
            self.subjects = subjects
            self.focused = focused
        }

        func thumbnail(_ key: String) -> NSImage? {
            subjects.all.first { $0.id == key }.flatMap { subjects.thumbnail($0) }
        }

        /// A group's bubble, or words to vary: over the text, where they are.
        func openBracket(_ target: BracketTarget, at rect: NSRect) {
            guard let series, let view else { return }
            BracketPopover.show(series, target: target, relativeTo: rect, of: view)
        }

        /// **« Vary This Word… » at the head of the right-click menu**: the selection, or the word under the click.
        func textView(_ textView: NSTextView, menu: NSMenu, for event: NSEvent, at charIndex: Int) -> NSMenu? {
            (textView as? PromptTextView)?.addVary(to: menu, at: charIndex)
            return menu
        }

        // ── the text changes ──

        func textDidChange(_ notification: Notification) {
            guard let view else { return }
            if text.wrappedValue != view.string { text.wrappedValue = view.string }
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
            let inert = reading.map {
                LabInputs.inert(in: text, known: Set(subjects.all.map(\.id)), capacity: $0.capacity, carried: $0.carried)
            } ?? []
            let greyedSubjects = Set(inert.filter { $0.why == .subject }.map(\.range.lowerBound))
            for m in NamedSubjects.mentions(in: text) {
                let greyed = greyedSubjects.contains(m.range.lowerBound)
                let r = NSRange(m.range, in: text)
                if let s = subjects.subject(named: m.name) {
                    names.append(contentsOf: s.name.split(whereSeparator: { $0 == "-" || $0 == "_" }).map(String.init))
                    let why = greyed ? String(localized: "\(reading?.model ?? "") does not read subjects") : s.kind.label
                    storage.addAttributes([.subjectChip: "\(s.kind.rawValue):\(s.id)", .toolTip: "@\(s.name) — \(why)"], range: r)
                    if greyed {
                        // Secondary, not tertiary as the greyed words: on the chip's fill, tertiary left the name unreadable
                        // (seen on capture); the faded thumbnail and the grey edge already say it does not go.
                        storage.addAttributes([.subjectChipInert: true, .foregroundColor: NSColor.secondaryLabelColor], range: r)
                    }
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
            // The series' groups: the whole group a chip, its braces and bars not drawn.
            let groupHelp = String(localized: "A series: Generate renders each of these words. Click to change them.")
            for (g, span) in PromptAlternatives.spans(text).enumerated() {
                let r = NSRange(span.range, in: text)
                storage.addAttributes([.seriesGroup: g, .toolTip: groupHelp], range: r)
                storage.addAttributes([.foregroundColor: NSColor.clear, .kern: PromptStyle.groupPadding],
                                      range: NSRange(location: r.location, length: 1))
                storage.addAttribute(.foregroundColor, value: NSColor.clear, range: NSRange(location: r.upperBound - 1, length: 1))
                for a in span.alternatives.dropLast() {
                    let bar = NSRange(location: NSRange(a, in: text).upperBound, length: 1)
                    storage.addAttributes([.foregroundColor: NSColor.clear, .seriesSeparator: true, .kern: PromptStyle.groupPadding],
                                          range: bar)
                }
            }
            for word in inert {
                guard case .image(let n) = word.why else { continue }
                storage.addAttributes([.foregroundColor: NSColor.tertiaryLabelColor,
                                       .toolTip: String(localized: "No image \(String(n)) goes with this render: these words are left out.")],
                                      range: NSRange(word.range, in: text))
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
            let matches = subjects.all.filter { $0.id.hasPrefix(key) && $0.id != key }
                + subjects.all.filter { $0.id == key }
            guard !matches.isEmpty, !(matches.count == 1 && matches[0].id == key) else { completion.close(); return }
            let caret = view.firstRect(forCharacterRange: NSRange(location: typed.range.location, length: 0), actualRange: nil)
            completion.show(matches.prefix(8).map { s in (s, subjects.thumbnail(s)) }, under: caret, parent: view.window!) { [weak self] s in
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

/// **What the model aimed at reads of the text**: how many images it takes (0: none), how many the render carries (the
/// edit's images and the subjects' pictures, not an img2img's start), its name for the hover.
struct PromptReading: Equatable {
    var capacity: Int
    var carried: Int
    var model: String
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
    /// The room after a group's `{` and around its dots.
    static let groupPadding: CGFloat = 3
    /// The `@`'s added advance: its own glyph is invisible, the thumbnail and its margins take its place.
    static let atKern: CGFloat = {
        let at = ("@" as NSString).size(withAttributes: [.font: font]).width
        return leading + thumbnail + gap - at
    }()
}

/// **The text view**: it says when it takes or loses the keyboard, and inserts a mention where the caret is
/// (a click on a subject of the lab's Images zone).
final class PromptTextView: NSTextView {
    /// The field on screen — at most one: the lab has one text.
    @MainActor static weak var current: PromptTextView?
    var focusChanged: ((Bool) -> Void)?
    /// Whether the list of subjects is shown under the caret: Esc closes it before it leaves the field.
    var listShown: (() -> Bool)?
    /// A group clicked, or words to vary: the bubble, at this rectangle of the view.
    var openBracket: ((BracketTarget, NSRect) -> Void)?

    /// **A click on a group** places the caret, as in any word, and opens the group's bubble.
    override func mouseDown(with event: NSEvent) {
        super.mouseDown(with: event)
        guard event.clickCount == 1, let openBracket, let storage = textStorage, selectedRange().length == 0 else { return }
        let caret = selectedRange().location
        guard caret < storage.length else { return }
        var r = NSRange()
        guard let g = storage.attribute(.seriesGroup, at: caret, effectiveRange: &r) as? Int, caret > r.location else { return }
        openBracket(.key(.group(g)), rect(of: r))
    }

    /// **The words a series would vary**: the selection when `index` falls in it (or no index is given: the card's ±), else the
    /// word at `index` or just before the caret — never blank. Inside a group: that group, to change its words.
    func varyTarget(at index: Int?) -> BracketTarget? {
        guard openBracket != nil, isEditable, let storage = textStorage, storage.length > 0 else { return nil }
        let length = storage.length
        let chosen = selectedRange()
        let at = min(max(index ?? chosen.location, 0), length)
        var r = NSRange()
        for probe in [min(at, length - 1), max(at - 1, 0)] {
            if let g = storage.attribute(.seriesGroup, at: probe, effectiveRange: &r) as? Int, chosen.length == 0 || index != nil {
                return .key(.group(g))
            }
        }
        let text = string as NSString
        func blank(_ r: NSRange) -> Bool { text.substring(with: r).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        var words: NSRange
        if chosen.length > 0, index == nil || NSLocationInRange(at, chosen) {
            words = chosen
        } else {
            words = selectionRange(forProposedRange: NSRange(location: min(at, length - 1), length: 0), granularity: .selectByWord)
            if blank(words), at > 0 {
                words = selectionRange(forProposedRange: NSRange(location: at - 1, length: 0), granularity: .selectByWord)
            }
        }
        guard !blank(words) else { return nil }
        var inGroup = false
        storage.enumerateAttribute(.seriesGroup, in: words) { v, _, stop in if v != nil { inGroup = true; stop.pointee = true } }
        return inGroup ? nil : .words(words, prompt: string)
    }

    /// Adds « Vary This Word… » at the head of `menu` when there are words to vary at `index`.
    func addVary(to menu: NSMenu, at index: Int) {
        guard let target = varyTarget(at: index) else { return }
        var title = String(localized: "Vary This Word…")
        if case .key = target { title = String(localized: "Change These Words…") }
        let item = NSMenuItem(title: title, action: #selector(varyFromMenu(_:)), keyEquivalent: "")
        item.target = self
        item.representedObject = index
        menu.insertItem(item, at: 0)
        menu.insertItem(.separator(), at: 1)
    }

    @objc private func varyFromMenu(_ sender: NSMenuItem) { vary(at: sender.representedObject as? Int) }

    /// **The bubble for the words at `index`** (nil: the selection or the caret): `false` when there is nothing to vary there.
    @discardableResult
    func vary(at index: Int? = nil) -> Bool {
        guard let target = varyTarget(at: index) else { return false }
        switch target {
        case let .words(words, _):
            setSelectedRange(words)
            openBracket?(target, rect(of: words))
        case .key(.group(let g)):
            var r = NSRange(location: 0, length: 0)
            textStorage?.enumerateAttribute(.seriesGroup, in: NSRange(location: 0, length: textStorage?.length ?? 0)) { v, range, stop in
                if v as? Int == g { r = range; stop.pointee = true }
            }
            openBracket?(target, rect(of: r))
        default:
            return false
        }
        return true
    }

    /// The first line's rectangle of these characters, in the view.
    private func rect(of range: NSRange) -> NSRect {
        guard let layout = layoutManager, let container = textContainer else { return bounds }
        let glyphs = layout.glyphRange(forCharacterRange: range, actualCharacterRange: nil)
        var rect = NSRect.zero
        layout.enumerateEnclosingRects(forGlyphRange: glyphs, withinSelectedGlyphRange: NSRange(location: NSNotFound, length: 0),
                                       in: container) { r, stop in rect = r; stop.pointee = true }
        return rect.offsetBy(dx: textContainerOrigin.x, dy: textContainerOrigin.y)
    }

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

    /// **Replaces `range` through the text system**, as a paste would: one undo step, which is how ⌘Z undoes a gesture of
    /// the writer. `string =` would leave the undo stack holding ranges of the old text.
    func replace(_ range: NSRange, with text: String) {
        if shouldChangeText(in: range, replacementString: text) {
            textStorage?.replaceCharacters(in: range, with: text)
            didChangeText()
        } else {
            string = (string as NSString).replacingCharacters(in: range, with: text)
            undoManager?.removeAllActions(withTarget: self)
        }
        setSelectedRange(NSRange(location: range.location + (text as NSString).length, length: 0))
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
        chips(in: glyphsToShow, at: origin) { rect, _, color, _, _ in
            color.withAlphaComponent(0.14).setFill()
            NSBezierPath(roundedRect: rect, xRadius: rect.height / 2, yRadius: rect.height / 2).fill()
        }
        let accent = Palette.bracketInk
        groups(in: glyphsToShow, at: origin) { rect in
            accent.withAlphaComponent(0.14).setFill()
            NSBezierPath(roundedRect: rect, xRadius: 5, yRadius: 5).fill()
        }
        super.drawBackground(forGlyphRange: glyphsToShow, at: origin)
    }

    override func drawGlyphs(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
        super.drawGlyphs(forGlyphRange: glyphsToShow, at: origin)
        let accent = Palette.bracketInk
        groups(in: glyphsToShow, at: origin) { rect in
            let edge = NSBezierPath(roundedRect: rect.insetBy(dx: 0.5, dy: 0.5), xRadius: 4.5, yRadius: 4.5)
            edge.lineWidth = 1
            accent.withAlphaComponent(0.7).setStroke()
            edge.stroke()
        }
        dots(in: glyphsToShow, at: origin) { center in
            accent.setFill()
            NSBezierPath(ovalIn: NSRect(x: center.x - 1.75, y: center.y - 1.75, width: 3.5, height: 3.5)).fill()
        }
        chips(in: glyphsToShow, at: origin) { rect, key, color, first, inert in
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
                           from: .zero, operation: .sourceOver, fraction: inert ? 0.35 : 1, respectFlipped: true, hints: nil)
            } else {
                color.withAlphaComponent(0.5).setFill()
                box.fill()
            }
            NSGraphicsContext.restoreGraphicsState()
        }
    }

    /// Each chip's rectangles in the glyphs shown (a chip that wraps has one per line), centered on
    /// the text's line whatever the line spacing.
    private func chips(in glyphs: NSRange, at origin: NSPoint, _ draw: (NSRect, String, NSColor, Bool, Bool) -> Void) {
        guard let storage = textStorage, let container = textContainers.first else { return }
        let characters = characterRange(forGlyphRange: glyphs, actualGlyphRange: nil)
        let font = PromptStyle.font
        var found: [(NSRect, String, NSColor, Bool, Bool)] = []
        storage.enumerateAttribute(.subjectChip, in: characters) { value, range, _ in
            guard let tag = value as? String, let colon = tag.firstIndex(of: ":") else { return }
            let key = String(tag[tag.index(after: colon)...])
            let inert = storage.attribute(.subjectChipInert, at: range.location, effectiveRange: nil) != nil
            let color = inert ? NSColor.secondaryLabelColor : (NamedSubjects.Kind(rawValue: String(tag[..<colon])) ?? .person).nsColor
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
                found.append((rect, key, color, first, inert))
                first = false
            }
        }
        for (rect, key, color, first, inert) in found { draw(rect, key, color, first, inert) }
    }

    /// Each group's rectangles in the glyphs shown, one per line it spans — the line's text height and a point around.
    private func groups(in glyphs: NSRange, at origin: NSPoint, _ draw: (NSRect) -> Void) {
        guard let storage = textStorage, let container = textContainers.first else { return }
        let characters = characterRange(forGlyphRange: glyphs, actualGlyphRange: nil)
        let font = PromptStyle.font
        var found: [NSRect] = []
        storage.enumerateAttribute(.seriesGroup, in: characters) { value, range, _ in
            guard value != nil else { return }
            let glyphRange = glyphRange(forCharacterRange: range, actualCharacterRange: nil)
            enumerateLineFragments(forGlyphRange: glyphRange) { lineRect, _, _, lineGlyphs, _ in
                let part = NSIntersectionRange(lineGlyphs, glyphRange)
                guard part.length > 0 else { return }
                let bounds = self.boundingRect(forGlyphRange: part, in: container)
                let baseline = lineRect.minY + self.location(forGlyphAt: part.location).y
                let top = baseline - font.ascender - 1, bottom = baseline - font.descender + 1
                found.append(NSRect(x: bounds.minX + origin.x, y: top + origin.y, width: bounds.width, height: bottom - top))
            }
        }
        for rect in found { draw(rect) }
    }

    /// The centre of each group's `|` shown: where its dot goes, at the words' middle height.
    private func dots(in glyphs: NSRange, at origin: NSPoint, _ draw: (NSPoint) -> Void) {
        guard let storage = textStorage, let container = textContainers.first else { return }
        let characters = characterRange(forGlyphRange: glyphs, actualGlyphRange: nil)
        let font = PromptStyle.font
        var found: [NSPoint] = []
        storage.enumerateAttribute(.seriesSeparator, in: characters) { value, range, _ in
            guard value != nil else { return }
            for i in range.location..<NSMaxRange(range) {
                let glyph = glyphIndexForCharacter(at: i)
                let bounds = boundingRect(forGlyphRange: NSRange(location: glyph, length: 1), in: container)
                let line = lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil)
                let baseline = line.minY + location(forGlyphAt: glyph).y
                found.append(NSPoint(x: bounds.midX + origin.x, y: baseline - font.xHeight / 2 + origin.y))
            }
        }
        for p in found { draw(p) }
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
                    .fill(i == model.selected ? Palette.live.opacity(0.22) : .clear))
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
