// The ± marks: a knob's (`BracketMark`, `.bracketable`), the text's (`PromptBracketMark`), and the popover that opens the bubble over the text.

import AppKit
import Siliconed
import SwiftUI

// MARK: - A knob bracketable

/// **An axis, as an arrow**: ↔ the columns, ↕ the rows, in the series' yellow; beyond two axes, the orange warning.
struct AxisMark: View {
    let slot: Int?
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        // By day the yellow is the arrow's ground (`Palette`).
        let day = scheme != .dark && slot != nil
        Image(systemName: slot == 0 ? "arrow.left.and.right" : slot == 1 ? "arrow.up.and.down" : "exclamationmark.triangle.fill")
            .foregroundStyle(slot == nil ? Palette.caution : day ? Palette.onBracket : Palette.bracket)
            .padding(.horizontal, day ? 3 : 0).padding(.vertical, day ? 1 : 0)
            .background(RoundedRectangle(cornerRadius: 3, style: .continuous).fill(day ? Palette.bracketFill : .clear))
            .help(slot == 0 ? String(localized: "Columns") : slot == 1 ? String(localized: "Rows")
                  : String(localized: "Two axes at most: this one does not render"))
    }
}

/// **The series' mark, at the right of what can vary**: always shown, in the series' yellow — outlined it offers, filled the knob
/// varies. A click opens the bubble. One that could do nothing is not drawn, its place kept.
struct BracketMark: View {
    @Bindable var app: AppState
    let key: SeriesKey
    /// The column the marks stand in, at the right of a box's settings: every control before it ends on one line, marked or not.
    static let gutter: CGFloat = 20

    var body: some View {
        let varies = app.varies(key)
        if varies || app.canBracket(key) {
            Button { app.bracketing = key } label: {
                BracketGlyph(varies: varies)
            }
            .buttonStyle(.borderless)
            .help(varies ? String(localized: "Bracketed: Generate renders each value. Click to change them.")
                         : String(localized: "Bracket: Generate renders this setting at several values"))
            .accessibilityLabel(Text("Bracket…"))
        } else {
            // Its place kept by the glyph itself, unseen: a frame around nothing takes no room, and the line's controls then
            // ran to the box's edge (« Add » past every other control).
            Image(systemName: "plusminus.circle").hidden().accessibilityHidden(true)
        }
    }
}

extension View {
    /// **A knob that can be bracketed**: its ± at its right, in `gutter` (and right-click › « Bracket… » with `menu`); bracketed,
    /// its values under it with the arrow of their axis, a click on them reopens the bubble.
    func bracketable(_ app: AppState, _ key: SeriesKey, menu: Bool = true, gutter: CGFloat = BracketMark.gutter) -> some View {
        modifier(Bracketable(app: app, key: key, menu: menu, gutter: gutter))
    }
}

private struct Bracketable: ViewModifier {
    @Bindable var app: AppState
    let key: SeriesKey
    let menu: Bool
    let gutter: CGFloat

    func body(content: Content) -> some View {
        let slot = app.seriesLayout.slot(of: key)
        withMenu(VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 0) {
                content
                BracketMark(app: app, key: key).frame(width: gutter, alignment: .trailing)
            }
            if app.varies(key) {
                Button { app.bracketing = key } label: {
                    HStack(spacing: 4) {
                        AxisMark(slot: slot)
                        Text(verbatim: app.axisWords(key)).lineLimit(1).truncationMode(.middle)
                            .foregroundStyle(slot == nil ? Palette.caution : Palette.bracketText)
                    }
                    .font(.caption)
                }
                .buttonStyle(.plain)
                .help("Bracketed: Generate renders each value")
            }
        })
        .popover(isPresented: Binding(get: { app.bracketing == key },
                                      set: { if !$0, app.bracketing == key { app.bracketing = nil } }),
                 arrowEdge: .trailing) {
            BracketBubble(app: app, target: .key(key)) { app.bracketing = nil }
                // No value selected on opening: the popover gave its first field the focus, its text selected in the system's blue.
                .onAppear { DispatchQueue.main.async { NSApp.keyWindow?.makeFirstResponder(nil) } }
        }
    }

    /// Right-click › « Bracket… » where the knob has no menu of its own (a LoRA's card has one, which says it).
    @ViewBuilder private func withMenu<V: View>(_ v: V) -> some View {
        if menu { v.contextMenu { Button("Bracket…") { app.bracketing = key } } } else { v }
    }
}

// MARK: - From the text

/// **The text's ±, at the card's right**, under its ✕: the selection, or the word at the caret, becomes a group — or the group the caret is
/// in opens; filled once the text has a group. The same mark as the rack's, so that the text reads as something that varies too.
struct PromptBracketMark: View {
    @Bindable var app: AppState
    /// Why the ± cannot act now (the writer busy with the text), its hover.
    var refusal: String? = nil

    var body: some View {
        let varies = !app.promptGroups.isEmpty
        Button {
            if PromptTextView.current?.vary() != true { NSSound.beep() }
        } label: {
            BracketGlyph(varies: varies)
        }
        .buttonStyle(.borderless)
        .help("Vary a word: select it, or put the caret in it, then click — Generate renders each word you give")
        .accessibilityLabel(Text("Vary This Word…"))
        .unavailable(refusal ?? (app.prompt.isEmpty ? String(localized: "Write a prompt first: its words are what varies") : nil))
    }
}

/// **The bubble over the text**: for words selected (« Vary This Word… ») or a group's chip clicked — an AppKit popover,
/// the field being AppKit's.
@MainActor
enum BracketPopover {
    private static var shown: NSPopover?

    static func show(_ app: AppState, target: BracketTarget, relativeTo rect: NSRect, of view: NSView) {
        shown?.close()
        let popover = NSPopover()
        popover.behavior = .transient
        let host = NSHostingController(rootView: BracketBubble(app: app, target: target) { [weak popover] in popover?.close() })
        host.sizingOptions = [.preferredContentSize]
        popover.contentViewController = host
        popover.show(relativeTo: rect, of: view, preferredEdge: .maxY)
        shown = popover
    }
}
