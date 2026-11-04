// **« Generate », at the end of the text** — in the Prompt Lab's bar under the image, where the hand is when the idea is
// written (`LabView.swift`), not at the foot of the rack:
//
//     folded     ▣ ▣  woman posing in a library…          ✦   [ Generate ]  ˄
//     unfolded   …the text, the images…
//                ↔ red, blue, green  ✕                                   [ Generate 3 Images ]
//
// - **No « Explore » beside it** (4/11): a series is the knobs' and the words' bracketing, which « Generate » renders
//   whole, in finished images; what varies is said above it, one line per axis (`SeriesSummary`).
// - **No cost here** (the user, 4/11/2026: « c'est de la technique pure »): the size, the memory, the time and how many renders
//   are the chain's, said under it in the rack (`RenderEstimate`); here, the idea, what varies, and the button that says
//   how many images it makes.
// - **What stops it is said beside it**: the reason (the whole sentence unfolded, cut to the line
//   folded, whole on hover), and for a size this Mac cannot hold, the largest it can. An empty text says nothing folded: the
//   field's own « Type the idea… » says it. A Mac not measured yet: « Measure This Mac… », folded or not — the render never
//   starts the measure, its sheet asks for a Mac at rest first (4/11).
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
                    if app.mapBlocker != nil { measure }
                    else if !app.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { folded(blocker) }
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
                            if app.mapBlocker != nil, !app.diagnosticRunning { measure }
                            // A memory refusal is never a dead end: a size this Mac renders.
                            if let f = refusal?.smaller {
                                Button(String(localized: "Use \(String(f.width)) × \(String(f.height))")) { app.useFormat(f) }
                                    .buttonStyle(.link).font(.caption)
                                    .help("The largest of this model’s sizes this Mac renders")
                            }
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

    /// **The way to the measure** (`mapBlocker`): its sheet, which says what it asks — a Mac at rest — before its « Start ».
    @ViewBuilder private var measure: some View {
        if app.diagnosticRunning {
            Text("Siliconed is measuring this Mac.").font(.caption).foregroundStyle(.secondary)
        } else {
            Button("Measure This Mac…") { app.openDiagnostic() }
                .buttonStyle(.link).font(.caption)
                .help("About a minute, the Mac at rest: Siliconed plans its renders from this measure")
        }
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
}
