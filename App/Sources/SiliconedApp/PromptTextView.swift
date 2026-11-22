// The prompt's text view: its focus, a mention inserted, a group's bubble opened from the text.

import AppKit
import Siliconed
import SwiftUI

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
