// **The app's colors: the « Icon » chart** — drawn from the app's icon,
// its indigo night, its mint glow, the peach and the rose of its gradient, and **not the same by day and by night**:
//
//                     dark: the icon's night             light: pearl paper, indigo ink
//     canvas          #131424                            #F9F6F2
//     column          #1B1C2F                            #EAE5DE
//     card            #2A283D                            #FEFDFB
//     what runs       mint #56C0B0, dark label           indigo #3F3A8A, white label
//     text wire       peach #F2B279                      #995505
//     image wire      rose #EFA6D6                       #964A7C
//     latent wire     sky #7FCAFE                        #196EA9
//     a series        yellow #FFD60A, as an ink          yellow #FFD60A as a ground, dark ink on it; its rim #B89400
//     refused         red #FC5852                        #D02C2A
//     secondary text  the system's                       black at 60 %
//
// - **By day, the bars are set first**: 4.5:1 for a text, 3:1 for a line or a glyph, on the card, the column and the canvas.
//   The first day chart missed them where the night one held — a wire at rest 2.8:1 on the column (6.7 at night), the system's
//   secondary text 4.0:1 on a card (5.4), the column 1.16 from its cards. So: the wires at OKLCH 0.52 instead of 0.60 (as a
//   text 5.4:1 or more on a card, at rest 3.4:1 or more on the column), the secondary text the chart's own (`secondary`,
//   5.3:1 or more on every ground), the column a notch darker (1.23 from its cards). **The series' yellow is a ground by
//   day, not an ink**: as an ink it read at 2.8:1 on a card, and darkened to pass it
//   landed on the text wire (OKLab ΔE 0.06). So a highlighter: the mark a yellow disc with a dark ±, a group of words
//   washed in yellow, their ink the text's (`BracketGlyph`, `bracketWash`, `onBracket` at 11:1 on the yellow).
// - **The three wires share lightness and chroma** (OKLCH 0.81 / 0.105 at night, 0.52 / 0.12 by day): only the hue tells
//   them apart, so that none shouts over the others. The two closest of the six roles (three wires, what runs, a series, a
//   refusal) stay apart (OKLab ΔE ≥ 0.104), and so do the wires for a
//   deuteranope (0.069 at night).
// - **What runs is the icon's mint, dimmed** (at full strength it is superb in a screenshot and too much
//   during a render): the icon's #6EF1DD (OKLCH 0.88 / 0.12 / 182°) brought to 0.74 at the same hue, its chroma in
//   proportion (0.101) — 60 % of the light, the same color; its dark label at 8.6:1, 0.121 from the sky wire, its nearest.
// - **What runs is not the system's accent**: mint at night, indigo by day, whatever accent System Settings holds.
//   Mint cannot take macOS's white label (1.3:1): the buttons that start something are drawn here (`ProminentButtonStyle`),
//   mint with a dark label and a halo at night.
// - **The system's own controls** take the icon's indigo, never the system's blue, which was the one color on screen from
//   outside the chart. A segment or a checkbox, by `.tint(Palette.control)` on the control itself: the
//   chart's indigo by day, a lighter one at night (#6660CB, OKLCH 0.55 / 0.16) so that their white label holds (5.1:1) —
//   not mint, on which a white label reads at 1.3:1. **Not a tint over the window**: macOS 26 paints the label of every
//   menu and bordered button in it (« Euler », « Add » in violet, 3:1 at night). A link, by `.chartLink`: the system's
//   `.link` keeps its blue whatever the tint; at night a lighter indigo (#999BEE, 6.6:1 on the column). In the prompt
//   and the writer's instructions (`Palette.ink`), the caret in the text wire's peach and a selection in a faint indigo,
//   not the system's blue: both are AppKit text views, since SwiftUI's `TextEditor` has no setting for a selection's color.
// - **An icon wears its wire** (`IconTile`): a full tile of the color of the wire its stage gives, the glyph hollowed in the
//   card's color; the stage at work wears what runs, a refused one red.
// - Edges and recesses stay translucent (`hairline`, `well`): over the indigo, they take its tint by themselves.

import AppKit
import SwiftUI

enum Palette {
    /// **The edition's name** (`EditionMark`): an Atlantic evening, the sea's teal into its dusk — by day deeper, to hold
    /// on the pearl paper.
    static let editionSea = dynamic(dark: 0x5FD3C6, light: 0x157A78)
    static let editionDusk = dynamic(dark: 0xB79CF5, light: 0x5B3FA8)
    static let peach = Color(nsColor: peachInk)
    /// The same, for AppKit: the prompt's caret.
    static let peachInk = dynamicInk(dark: 0xF2B279, light: 0x995505)
    static let rose = dynamic(dark: 0xEFA6D6, light: 0x964A7C)
    static let sky = dynamic(dark: 0x7FCAFE, light: 0x196EA9)
    /// **A series** (bracketing): the mark at the left of what can vary, the text's groups, the axes beside « Generate » — one
    /// yellow everywhere, a hue of its own: what runs is mint or indigo, peach the text's wire.
    static let bracketInk = dynamicInk(dark: 0xFFD60A, light: 0xB89400)
    static let bracket = Color(nsColor: bracketInk)
    /// The series' yellow as a ground: by day under a mark's glyph, full.
    static let bracketFill = Color(nsColor: NSColor(srgbRed: 1, green: 0xD6 / 255, blue: 0x0A / 255, alpha: 1))
    /// A glyph on `bracketFill`.
    static let onBracket = Color(nsColor: NSColor(srgbRed: 0x2B / 255, green: 0x24 / 255, blue: 0, alpha: 1))
    /// Behind the words of a series: a faint wash at night, where the yellow is their edge; by day the highlighter.
    static let bracketWashInk = NSColor(name: nil) { a in
        NSColor(srgbRed: 1, green: 0xD6 / 255, blue: 0x0A / 255, alpha: a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? 0.14 : 0.45)
    }
    static let bracketWash = Color(nsColor: bracketWashInk)
    /// The words of a series, outside the field: yellow at night; by day the text's ink, on `bracketWash`.
    static let bracketText = Color(nsColor: NSColor(name: nil) { a in
        a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? NSColor(srgbRed: 1, green: 0xD6 / 255, blue: 0x0A / 255, alpha: 1) : .labelColor
    })
    /// What runs: the stage at work, the steps done, the image followed, the button that starts something.
    static let live = Color(nsColor: liveInk)
    /// The same, for AppKit: the Dock tile's progress.
    static let liveInk = dynamicInk(dark: 0x56C0B0, light: 0x3F3A8A)
    /// A selected segment, a checked box: `.tint(Palette.control)` on the control.
    static let control = dynamic(dark: 0x6660CB, light: 0x3F3A8A)
    /// A selection in the prompt (AppKit): `control`, faint, under the words.
    static let selectionInk = dynamicInk(dark: 0x6660CB, light: 0x3F3A8A, alpha: 0.38)
    /// **The words the writer brings** (Rewrite, Describe), over the text: in the text wire's peach, on a faint wash of it.
    static let addedInk = peachInk
    static let addedWash = dynamicInk(dark: 0xF2B279, light: 0x995505, alpha: 0.12)
    /// **A text view in the chart's ink**: the caret in the text wire's peach, a selection in a faint indigo — never the
    /// system's blue. Every editable text of the app goes through it (`PromptField`, the writer's instructions).
    @MainActor static func ink(_ view: NSTextView) {
        view.insertionPointColor = peachInk
        view.selectedTextAttributes = [.backgroundColor: selectionInk]
    }
    /// A link's words (`ChartLinkStyle`).
    static let link = dynamic(dark: 0x999BEE, light: 0x3F3A8A)
    /// A label or a glyph on `live`.
    static let onLive = dynamic(dark: 0x0F1021, light: 0xFCFCFC)
    static let failure = dynamic(dark: 0xFC5852, light: 0xD02C2A)
    /// Behind the image: the color is the image's, the ground stays dark at night, pale by day.
    static let canvas = dynamic(dark: 0x131424, light: 0xF9F6F2)
    /// The rack's column; the ground of the Settings.
    static let column = Color(nsColor: columnInk)
    static let columnInk = dynamicInk(dark: 0x1B1C2F, light: 0xEAE5DE)
    /// **A secondary text** — a setting's name, a tag, a cost: the system's at night; by day black at 60 %, where the system's
    /// 50 % reads at 4.0:1 on a card, under the bar, in the small sizes the rack writes most of its words in.
    static let secondary = Color(nsColor: secondaryInk)
    /// The same, for AppKit: the prompt's greyed words.
    static let secondaryInk = NSColor(name: nil) { a in
        a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? .secondaryLabelColor : NSColor(white: 0, alpha: 0.6)
    }
    /// A box: a card a step above the column.
    static let surface = dynamic(dark: 0x2A283D, light: 0xFEFDFB)
    /// A field inside a box: a recess.
    static let well = Color.primary.opacity(0.08)
    /// A card's edge, a separator: a notch firmer than the system's `separatorColor` (10 %), which
    /// let a box melt into the column on both grounds.
    static let hairline = Color(nsColor: NSColor(name: nil) { a in
        a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? NSColor(white: 1, alpha: 0.16) : NSColor(white: 0, alpha: 0.14)
    })

    private static func dynamic(dark: Int, light: Int) -> Color { Color(nsColor: dynamicInk(dark: dark, light: light)) }

    /// The same, for AppKit's drawing (the text's groups).
    private static func dynamicInk(dark: Int, light: Int, alpha: CGFloat = 1) -> NSColor {
        func color(_ v: Int) -> NSColor {
            NSColor(srgbRed: CGFloat(v >> 16 & 0xFF) / 255, green: CGFloat(v >> 8 & 0xFF) / 255, blue: CGFloat(v & 0xFF) / 255, alpha: alpha)
        }
        return NSColor(name: nil) { a in
            a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? color(dark) : color(light)
        }
    }
}

/// **The series' ±**: at night the glyph in yellow, outlined (it offers) or filled (it varies); by day a yellow disc, pale
/// or full, under a dark ± — the glyph's own size either way, so that the marks' column does not move.
struct BracketGlyph: View {
    let varies: Bool
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let glyph = Image(systemName: varies ? "plusminus.circle.fill" : "plusminus.circle")
        if scheme == .dark {
            glyph.foregroundStyle(Palette.bracket)
        } else {
            glyph.hidden().overlay {
                Circle().fill(Palette.bracketFill.opacity(varies ? 1 : 0.3))
                    .overlay(Circle().strokeBorder(Palette.bracket, lineWidth: 1))
                    .overlay(Image(systemName: "plusminus").font(.system(size: 8, weight: .heavy)).foregroundStyle(Palette.onBracket))
                    .padding(0.5)
            }
        }
    }
}

/// **A stage's or a port's icon**: a full tile of `color`, the glyph hollowed in the card's color.
struct IconTile: View {
    let symbol: String
    let color: Color
    /// 20 in a box's title; 26 before a setting.
    var side: CGFloat = 20

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: side * 0.55, weight: .semibold))
            .foregroundStyle(Palette.surface)
            .frame(width: side, height: side)
            .background(RoundedRectangle(cornerRadius: side / 4, style: .continuous).fill(color))
            // Always beside its title: VoiceOver read the glyph's name (« memory chip », « cpu ») before every setting.
            .accessibilityHidden(true)
    }
}

/// **The button that starts something** (« Generate », « Install… », « Crop »): filled with what runs, its label in
/// `onLive`; at night, a halo of the same mint. Its size follows the control size it is given, as the system's does.
struct ProminentButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View { Face(configuration: configuration) }

    private struct Face: View {
        let configuration: Configuration
        @Environment(\.isEnabled) private var enabled
        @Environment(\.controlSize) private var size
        @Environment(\.colorScheme) private var scheme

        var body: some View {
            let large = size == .large || size == .extraLarge
            let shape = RoundedRectangle(cornerRadius: large ? 8 : 6, style: .continuous)
            let night = enabled && scheme == .dark
            configuration.label
                .font(.body)
                .foregroundStyle(enabled ? AnyShapeStyle(Palette.onLive) : AnyShapeStyle(Palette.secondary))
                .padding(.horizontal, large ? 14 : 10)
                .frame(minHeight: large ? 30 : 24)
                .background(shape.fill(enabled ? AnyShapeStyle(Palette.live) : AnyShapeStyle(Palette.well)))
                .overlay(shape.strokeBorder(.white.opacity(night ? 0.35 : 0), lineWidth: 0.5))
                .shadow(color: night ? Palette.live.opacity(0.45) : .clear, radius: 7)
                .opacity(configuration.isPressed ? 0.78 : 1)
                .contentShape(shape)
        }
    }
}

extension ButtonStyle where Self == ProminentButtonStyle {
    static var prominent: ProminentButtonStyle { ProminentButtonStyle() }
}

/// **A link in the chart's indigo** (`.chartLink`): the system's `.link`, its words in `Palette.link`.
struct ChartLinkStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View { Face(configuration: configuration) }

    private struct Face: View {
        let configuration: Configuration
        @Environment(\.isEnabled) private var enabled

        var body: some View {
            configuration.label
                .foregroundStyle(enabled ? AnyShapeStyle(Palette.link) : AnyShapeStyle(.tertiary))
                .opacity(configuration.isPressed ? 0.7 : 1)
                .contentShape(Rectangle())
        }
    }
}

extension ButtonStyle where Self == ChartLinkStyle {
    static var chartLink: ChartLinkStyle { ChartLinkStyle() }
}
