// **« Generate », pinned at the foot of the window's column** (`Sidebar.swift`), under the chain it runs:
//
//     …the text, the images, the stages…
//     ─────────────────────────────────────────────────
//     ↔ red, blue, green  ✕
//     ⓘ 1 image of the Prompt Lab not used
//     what stops it
//     ● Core AI compiling                    [ Generate 3 Images ]
//
// - **No « Explore » beside it**: a series is the knobs' and the words' bracketing, which « Generate » renders
//   whole, in finished images; what varies is said above it, one line per axis (`SeriesSummary`).
// - **No cost here**: the size, the memory, the time and how many renders are the chain's, said under it (`RenderEstimate`);
//   here, what varies, and the button that says how many images it makes.
// - **What stops it is said beside it**: the reason, whole, and for a size this Mac cannot hold, the largest it can —
//   except an empty text, which its field's placeholder says already (the button's help still does).
// - **What the Images box holds and the render will not read** is counted here too, never in silence.
// - **Core AI's light, last, facing the button** (`CoreAIMark`): whether it compiles, and how far.
// - ⌘↩ from anywhere in the window (the Render menu); when something stops the render, ⌘↩ unfolds the column, where the
//   reason is said, instead of doing nothing.

import AppKit
import Siliconed
import SwiftUI

struct GenerateControls: View {
    @Bindable var app: AppState

    var body: some View {
        // The series read once, for the reason and for the count: each reading builds its grid again (the carry, the prompt's
        // parse), and the two read it twice at every key typed.
        let series = app.seriesReading
        let blocker = app.blocker(reading: series)
        let renders = series.renders
        // What the click will vary, what it leaves unused and what stops it, stacked at the left of the button, centred on
        // it: one band, as low as the button when nothing more is to be said.
        HStack(alignment: .center, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                SeriesSummary(app: app)
                unusedLine
                if let blocker, blocker != AppState.writeAPrompt {
                    Text(blocker).font(.callout).fixedSize(horizontal: false, vertical: true)
                }
                CoreAIMark(app: app)
            }
            Spacer(minLength: 8)
            generate(blocker: blocker, renders: renders)
        }
    }

    private func generate(blocker: String?, renders: Int) -> some View {
        Button { app.render() } label: {
            Text(Self.title(app, renders: renders)).fontWeight(.semibold).lineLimit(1)
        }
        .buttonStyle(.prominent)
        .controlSize(.large)
        .fixedSize()
        .help(Text(verbatim: "⌘↩"))
        .unavailable(blocker)
    }

    /// The button's words: what the click makes — several images (`renders`: the series', one image each), an edit, a place
    /// in the queue.
    static func title(_ app: AppState, renders n: Int) -> String {
        if app.busy { return n > 1 ? String(localized: "Add \(n) Renders to Queue") : String(localized: "Add to Queue") }
        if app.edits { return n > 1 ? String(localized: "Edit Image \(n) Ways") : String(localized: "Edit Image") }
        return n > 1 ? String(localized: "Generate \(n) Images") : String(localized: "Generate")
    }

    @ViewBuilder private var unusedLine: some View {
        let unused = app.unusedInputs
        if unused > 0 {
            Label(Self.unusedSentence(unused), systemImage: "info.circle")
                .font(.caption).foregroundStyle(Palette.secondary)
                .help("Without a role, or with one this model lacks: give it a role in the Images box")
        }
    }

    static func unusedSentence(_ unused: Int) -> String {
        unused == 1 ? String(localized: "1 image of the Prompt Lab not used")
                    : String(localized: "\(unused) images of the Prompt Lab not used")
    }
}
