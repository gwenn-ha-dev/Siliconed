// **« Generate », at the end of the text** — in the Prompt Lab's bar under the image, where the hand is when the idea is
// written (`LabView.swift`), not at the foot of the rack:
//
//     folded     ▣ ▣  woman posing in a library…          ✦   ≈ 31 s  [ Generate ]  ˄
//     unfolded   …the text, the images…
//                1024 × 1024 · 9.2 GB · ≈ 31 s                               [ Generate ]
//
// - **No « Explore » beside it** (4/11): a series is the knobs' and the words' bracketing, which « Generate » renders
//   whole, in finished images; what varies is said above it, one line per axis (`SeriesSummary`).
// - **The cost before the click** (rule 7): folded, the time alone (and how many renders a series makes); unfolded, the
//   size, the memory the engine will compare to this Mac's, how many renders, the time — learned on this Mac.
// - **What stops it is said beside it**, in place of the cost: the reason (the whole sentence unfolded, cut to the line
//   folded, whole on hover), and for a size this Mac cannot hold, the largest it can. An empty text says nothing folded: the
//   field's own « Type the idea… » says it.
// - **What the lab holds and the render will not read** is counted here too, never in silence.
// - ⌘↩ from anywhere in the window (the Render menu). The queue, the measure of this Mac and what went wrong stay under the
//   rack (`Sidebar`): they are the chain's, not the idea's.

import AppKit
import Siliconed
import SwiftUI

struct GenerateControls: View {
    @Bindable var app: AppState
    /// The folded line: one line, the time alone, the reason cut to it.
    var compact = false

    var body: some View {
        // Read once per drawing: the memory half computes the need against B (`MemoryBudget`).
        let form = app.formBlocker
        let refusal = form == nil ? app.memoryRefusal : nil
        let blocker = form ?? refusal?.text
        if compact {
            HStack(spacing: 8) {
                unusedMark
                if let blocker {
                    if !app.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { folded(blocker) }
                } else {
                    time
                }
                generate(disabled: blocker != nil)
            }
        } else {
            VStack(alignment: .leading, spacing: 6) {
                SeriesSummary(app: app)
                unusedLine
                HStack(alignment: .center, spacing: 10) {
                    VStack(alignment: .leading, spacing: 3) {
                        if let blocker {
                            Text(blocker).font(.callout).fixedSize(horizontal: false, vertical: true)
                            // A memory refusal is never a dead end: a size this Mac renders.
                            if let f = refusal?.smaller {
                                Button(String(localized: "Use \(String(f.width)) × \(String(f.height))")) { app.useFormat(f) }
                                    .buttonStyle(.link).font(.caption)
                                    .help("The largest of this model’s sizes this Mac renders")
                            }
                        } else {
                            cost
                        }
                    }
                    Spacer(minLength: 8)
                    generate(disabled: blocker != nil)
                }
            }
        }
    }

    private func generate(disabled: Bool) -> some View {
        Button { app.render() } label: {
            Text(Self.title(app)).fontWeight(.semibold).lineLimit(1)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(compact ? .regular : .large)
        .fixedSize()
        .disabled(disabled)
        .help(Text(verbatim: "⌘↩"))
    }

    /// The button's words: what the click makes — several images, an edit, a place in the queue.
    static func title(_ app: AppState) -> String {
        let n = app.seriesRenders
        if app.busy { return n > 1 ? String(localized: "Add \(n) Renders to Queue") : String(localized: "Add to Queue") }
        if !app.references.isEmpty { return n > 1 ? String(localized: "Edit Image \(n) Ways") : String(localized: "Edit Image") }
        let images = app.seriesImages
        return images > 1 ? String(localized: "Generate \(images) Images") : String(localized: "Generate")
    }

    // MARK: Folded

    /// The reason, cut to the line; whole on hover, and a click unfolds the bar, where it is whole.
    private func folded(_ reason: String) -> some View {
        Button { LabModel.shared.unfold() } label: {
            Label(reason, systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundStyle(.secondary)
                .lineLimit(1).truncationMode(.tail)
        }
        .buttonStyle(.plain)
        .frame(maxWidth: 260, alignment: .trailing)
        .help(reason)
    }

    @ViewBuilder private var time: some View {
        if app.missing == nil, app.card != nil {
            HStack(spacing: 6) {
                if app.seriesRenders > 1 {
                    Text("\(app.seriesRenders) renders").help(app.seriesHelp)
                }
                if let t = app.plannedCost {
                    Text(verbatim: "≈ " + Figures.duration(t))
                        .help("Learned on this Mac: the first render at this size, with this model, sets the estimate")
                }
            }
            .font(.caption).monospacedDigit().foregroundStyle(.secondary)
        }
    }

    /// Folded, the images the render will not read are a mark: its sentence on hover, a click unfolds the bar where they are.
    @ViewBuilder private var unusedMark: some View {
        let unused = app.unusedInputs
        if unused > 0 {
            Button { LabModel.shared.unfold() } label: {
                Label { Text(verbatim: "\(unused)") } icon: { Image(systemName: "info.circle") }
                    .font(.caption).monospacedDigit()
            }
            .buttonStyle(.plain).foregroundStyle(.secondary)
            .help(Self.unusedSentence(unused))
            .accessibilityLabel(Text(verbatim: Self.unusedSentence(unused)))
        }
    }

    // MARK: Unfolded

    @ViewBuilder private var unusedLine: some View {
        let unused = app.unusedInputs
        if unused > 0 {
            Label(Self.unusedSentence(unused), systemImage: "info.circle")
                .font(.caption).foregroundStyle(.secondary)
                .help("Without a role, or with one this model lacks: give it a role in the Prompt Lab")
        }
    }

    static func unusedSentence(_ unused: Int) -> String {
        unused == 1 ? String(localized: "1 image of the Prompt Lab not used")
                    : String(localized: "\(unused) images of the Prompt Lab not used")
    }

    /// **Rule 7: the cost before the click** — the size, the memory the engine will compare to the machine's, how many renders,
    /// and the time, learned on this Mac in this session (never a constant from another machine).
    @ViewBuilder private var cost: some View {
        if app.missing == nil, let card = app.card {
            let f = app.outputFormat
            HStack(spacing: 14) {
                Label {
                    Text(verbatim: "\(f.width) × \(f.height)").monospacedDigit()
                } icon: { Image(systemName: "aspectratio") }
                if let need = app.memoryNeed {
                    Label {
                        Text(Figures.memory(need)).monospacedDigit()
                    } icon: { Image(systemName: "memorychip") }
                    .help(String(localized: "The memory \(card.name) needs at this size (computed)"))
                }
                if app.seriesRenders > 1 {
                    // A series: how many renders, before the click.
                    Label {
                        Text("\(app.seriesRenders) renders").monospacedDigit()
                    } icon: { Image(systemName: "square.grid.3x3") }
                    .help(app.seriesHelp)
                }
                Label {
                    if let t = app.plannedCost {
                        Text(verbatim: "≈ " + Figures.duration(t)).monospacedDigit()
                    } else {
                        Text("after one render")
                    }
                } icon: { Image(systemName: "clock") }
                .help("Learned on this Mac: the first render at this size, with this model, sets the estimate")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .labelStyle(CompactLabel())
        }
    }
}

/// An icon and its text, close together.
private struct CompactLabel: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 4) { configuration.icon; configuration.title }
    }
}
