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
// The window's key monitor leaves Esc to it while it is open (`PromptTextView.listShown`).
//
// **A series' group is a chip too** (`{red|blue}`, `PromptAlternatives`): drawn in the series' yellow (`Palette.bracket`), its
// braces and bars not drawn — a dot between its words. A click on it opens its bubble (`BracketBubble`), whose words rewrite the
// group where it is typed; braces typed by hand make one as well. Words become a group by the ± at the card's right (`PromptBracketMark`: the
// selection, or the word at the caret) or by « Vary This Word… » at the head of the right-click menu — added by the delegate
// (`textView(_:menu:for:at:)`): with an override of `menu(for:)`, only the system's menu shows. The
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
    /// The least height it takes, its text growing it beyond (`sizeThatFits`): the whole field is a place to click.
    var minimumHeight: CGFloat = 22

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
        // No scroller: the field is as tall as its text (`sizeThatFits`), the column scrolls. With legacy scrollers (a mouse,
        // or « Show scroll bars: Always ») one appeared on a paste longer than the field, took its width from the text,
        // which wrapped to more lines than measured and spilled below the box (issue 18).
        scroll.hasVerticalScroller = false
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
            // The whole field drawn again, at the layer: the text view invalidates only what is visible of it, and a text
            // set while the column has scrolled the field out of sight (`reuse` with the rack on the DiT) kept the old
            // lines past the new text's end in the layer, drawn when the field came back.
            view.needsDisplay = true
            view.layer?.setNeedsDisplay()
        } else if c.styledVersion != version, !view.hasMarkedText() {
            c.restyle()
        }
        c.styledVersion = version
    }

    /// **As tall as its text**: the Text box gives it at least the height the hand left (`TextBox`), and the text grows it
    /// beyond — the column scrolls, the field does not hide its words.
    func sizeThatFits(_ proposal: ProposedViewSize, nsView scroll: NSScrollView, context: NSViewRepresentableContext<Self>) -> CGSize? {
        guard let view = scroll.documentView as? NSTextView, let layout = view.layoutManager, let container = view.textContainer,
              let width = proposal.width, width.isFinite, width > 0 else { return nil }
        if abs(container.size.width - width) > 0.5 { container.size = NSSize(width: width, height: .greatestFiniteMagnitude) }
        layout.ensureLayout(for: container)
        let height = layout.usedRect(for: container).height + 2 * view.textContainerInset.height
        return CGSize(width: width, height: max(height.rounded(.up), minimumHeight))
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
                        storage.addAttributes([.subjectChipInert: true, .foregroundColor: Palette.secondaryInk], range: r)
                    }
                    // The `@`, not drawn: its advance makes the thumbnail's room.
                    storage.addAttributes([.foregroundColor: NSColor.clear, .kern: PromptStyle.atKern],
                                          range: NSRange(location: r.location, length: 1))
                    storage.addAttribute(.kern, value: PromptStyle.trailing, range: NSRange(location: r.upperBound - 1, length: 1))
                } else {
                    storage.addAttributes([.underlineStyle: NSUnderlineStyle.single.rawValue | NSUnderlineStyle.patternDot.rawValue,
                                           .underlineColor: NSColor.systemRed.withAlphaComponent(0.75),
                                           .foregroundColor: Palette.secondaryInk,
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
