// **The app's colors: the « Icon » chart** (4/11/2026, chosen by the user on a board of five) — drawn from the app's icon,
// its indigo night, its mint glow, the peach and the rose of its gradient, and **not the same by day and by night**:
//
//                     dark: the icon's night             light: pearl paper, indigo ink
//     canvas          #131424                            #F9F6F2
//     column          #1B1C2F                            #F0ECE7
//     card            #2A283D                            #FEFDFB
//     what runs       mint #56C0B0, dark label           indigo #3F3A8A, white label
//     text wire       peach #F2B279                      #B36D2A
//     image wire      rose #EFA6D6                       #B06194
//     latent wire     sky #7FCAFE                        #3786C3
//     a series        yellow #FFD60A                     #B89400      (kept: the user's own, 4/11)
//     refused         red #FC5852                        #D02C2A
//
// - **The three wires share lightness and chroma** (OKLCH 0.81 / 0.105 at night, 0.60 / 0.12 by day): only the hue tells
//   them apart, so that none shouts over the others. The two closest of the six roles (three wires, what runs, a series, a
//   refusal) stay as far apart as in the former chart (OKLab ΔE ≥ 0.104 against 0.109), and so do the wires for a
//   deuteranope (0.069 against 0.082 at night).
// - **What runs is the icon's mint, dimmed** (5/11, the user: « superb in a screenshot, too much in use during a render »,
//   then « softer still »): the icon's #6EF1DD (OKLCH 0.88 / 0.12 / 182°) brought to 0.74 at the same hue, its chroma in
//   proportion (0.101) — 60 % of the light, the same color; its dark label at 8.6:1, 0.121 from the sky wire, its nearest.
// - **What runs is no longer the system's accent**: mint at night, indigo by day, whatever accent System Settings holds.
//   Mint cannot take macOS's white label (1.3:1): the buttons that start something are drawn here (`ProminentButtonStyle`),
//   mint with a dark label and a halo at night.
// - **The system's own controls** take the icon's indigo, never the system's blue, which was the one color on screen from
//   outside the chart (the user, 4/11). A segment or a checkbox, by `.tint(Palette.control)` on the control itself: the
//   chart's indigo by day, a lighter one at night (#6660CB, OKLCH 0.55 / 0.16) so that their white label holds (5.1:1) —
//   not mint, on which a white label reads at 1.3:1. **Not a tint over the window**: macOS 26 paints the label of every
//   menu and bordered button in it (« Euler », « Add » in violet, 3:1 at night). A link, by `.chartLink`: the system's
//   `.link` keeps its blue whatever the tint; at night a lighter indigo (#999BEE, 6.6:1 on the column). In the prompt
//   (`PromptField`), the caret in the text wire's peach and a selection in a faint indigo, not the system's blue.
// - **An icon wears its wire** (`IconTile`): a full tile of the color of the wire its stage gives, the glyph hollowed in the
//   card's color; the stage at work wears what runs, a refused one red. It replaces the former rule — a bare glyph, no chip,
//   so that nothing in a box's title would read as a button —: the chart's choice, to be judged on screen.
// - Edges and recesses stay translucent (`hairline`, `well`): over the indigo, they take its tint by themselves.

import AppKit
import SwiftUI

enum Palette {
    static let peach = Color(nsColor: peachInk)
    /// The same, for AppKit: the prompt's caret.
    static let peachInk = dynamicInk(dark: 0xF2B279, light: 0xB36D2A)
    static let rose = dynamic(dark: 0xEFA6D6, light: 0xB06194)
    static let sky = dynamic(dark: 0x7FCAFE, light: 0x3786C3)
    /// **A series** (bracketing): the mark at the left of what can vary, the text's groups, the axes beside « Generate » — one
    /// yellow everywhere, a hue of its own: what runs is mint or indigo, peach the text's wire (the user, 4/11/2026).
    static let bracketInk = dynamicInk(dark: 0xFFD60A, light: 0xB89400)
    static let bracket = Color(nsColor: bracketInk)
    /// What runs: the stage at work, the steps done, the image followed, the button that starts something.
    static let live = dynamic(dark: 0x56C0B0, light: 0x3F3A8A)
    /// A selected segment, a checked box: `.tint(Palette.control)` on the control.
    static let control = dynamic(dark: 0x6660CB, light: 0x3F3A8A)
    /// A selection in the prompt (AppKit): `control`, faint, under the words.
    static let selectionInk = dynamicInk(dark: 0x6660CB, light: 0x3F3A8A, alpha: 0.38)
    /// A link's words (`ChartLinkStyle`).
    static let link = dynamic(dark: 0x999BEE, light: 0x3F3A8A)
    /// A label or a glyph on `live`.
    static let onLive = dynamic(dark: 0x0F1021, light: 0xFCFCFC)
    static let failure = dynamic(dark: 0xFC5852, light: 0xD02C2A)
    /// Behind the image: the color is the image's, the ground stays dark at night, pale by day.
    static let canvas = dynamic(dark: 0x131424, light: 0xF9F6F2)
    /// The rack's column; the ground of the Settings.
    static let column = Color(nsColor: columnInk)
    static let columnInk = dynamicInk(dark: 0x1B1C2F, light: 0xF0ECE7)
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
                .foregroundStyle(enabled ? AnyShapeStyle(Palette.onLive) : AnyShapeStyle(.secondary))
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
