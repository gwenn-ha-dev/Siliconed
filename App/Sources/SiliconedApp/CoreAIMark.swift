// **Core AI, at the foot of the column, facing « Generate »** — the one place the window says it compiles:
//
//     ● Core AI compiling                    [ Generate ]
//               ▔▔▔▔▔▔                         the word lit as far as the preparation is
//
// - **A light**: dim at rest, the live ink while Core AI compiles (teal at night, indigo by day), its glow a still
//   shadow. No spinner, no pulse: an animation that never stops redraws at the display's rate, beside a render.
// - **The word « compiling » is the progress bar**: lit from its left as far as a preparation is, over every key of its
//   burst (`AppState.preparationFraction`, told at most twice a second); whole for a render's own compilation, whose
//   count is not known. Nothing else here: what compiles, and the rest, are the Machine window's.
// - **A click opens the Machine window.**

import Siliconed
import SwiftUI

struct CoreAIMark: View {
    let app: AppState

    var body: some View {
        let on = app.compilingCoreAI, fraction = app.preparationFraction
        Button { MainWindow.shared.showMachine() } label: {
            HStack(alignment: .firstTextBaseline, spacing: 7) {
                Circle()
                    .fill(on ? Palette.live : Palette.secondary.opacity(0.35))
                    .frame(width: 8, height: 8)
                    .shadow(color: on ? Palette.live.opacity(0.85) : .clear, radius: 3)
                    .alignmentGuide(.firstTextBaseline) { $0[.bottom] - 1 }
                Text(verbatim: "Core AI").foregroundStyle(on ? Color.primary : Palette.secondary)
                if on { LitWord(fraction: fraction ?? 1) }
            }
            .font(.callout)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Core AI compiles for this Mac what the renders need, once. The Machine window says what.")
        .accessibilityLabel(Text(verbatim: on ? (["Core AI", String(localized: "compiling")] + [fraction.map { Figures.percent(Int(($0 * 100).rounded())) }].compactMap { $0 })
                                                    .joined(separator: ", ") : "Core AI"))
    }
}

/// « compiling », lit in the live ink from its left up to `fraction`, the rest faint.
private struct LitWord: View {
    let fraction: Double

    var body: some View {
        let word = Text("compiling")
        word.foregroundStyle(Palette.secondary.opacity(0.55))
            .overlay(alignment: .leading) {
                word.foregroundStyle(Palette.live)
                    .mask(alignment: .leading) {
                        GeometryReader { g in Rectangle().frame(width: g.size.width * CGFloat(min(max(fraction, 0), 1))) }
                    }
            }
    }
}
