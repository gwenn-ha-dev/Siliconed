// **The window's left column: the rack, and under it what the click costs**.
//
//     the rack     the chain top to bottom, each module with its settings (`Rack.swift`)
//     ─────────
//     the cost before the click (rule 7: memory and time), « Generate », the queue, what went wrong
//
// No form beside the rack: a setting lives in the box of the stage that reads it. No mask, no setting
// between two steps: what is set is taken whole at the click (`Job`), and changing it afterwards
// prepares the next render.
//
// **The XY grid sits here, above « Generate »** (`Grid.swift`), not in a box of the rack: it is no
// stage of the chain and sets nothing a stage reads — it says how many renders the click makes and
// what differs between them, which is what this footer counts (memory, renders, time).

import AppKit
import Siliconed
import SwiftUI

struct Sidebar: View {
    @Bindable var app: AppState

    var body: some View {
        ScrollView(.vertical) {
            RackView(app: app)
        }
        .scrollIndicators(.automatic)
        .safeAreaInset(edge: .bottom, spacing: 0) { Footer(app: app) }
        // A `.safetensors` or a `.gguf` dropped on the column is imported (model or LoRA).
        .dropDestination(for: URL.self) { urls, _ in
            let s = urls.filter(AppState.importable)
            guard !s.isEmpty, app.managementPossible else { return false }
            app.perform(s)
            return true
        }
    }
}

/// **A size typed by hand**, judged by the engine's own rules before it is taken.
struct CustomSize: View {
    let app: AppState
    @State private var text = ""
    @State private var refusal: Problem?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Size").font(.headline)
            HStack {
                TextField("Size", text: $text, prompt: Text(verbatim: "832x1216"))
                    .labelsHidden()
                    .frame(width: 140)
                    .onSubmit(apply)
                Button("Use", action: apply).keyboardShortcut(.defaultAction)
            }
            if let refusal {
                Text(refusal.title).font(.caption).foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("Width × height: each side 512 at least and a multiple of 16, 1024 × 1536 at most.")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(width: 280)
        .onAppear { text = "\(app.format.width)x\(app.format.height)" }
    }

    private func apply() {
        refusal = app.setCustomFormat(text)
        if refusal == nil { dismiss() }
    }
}

/// **A reference's thumbnail**: its number (the one the prompt says), the image, and on hover a
/// button to remove it. Dragged onto another thumbnail, it takes that one's place; its menu moves,
/// replaces or removes it.
struct ReferenceThumbnail: View {
    static let side: CGFloat = 60
    let app: AppState
    let reference: ReferenceImage
    let number: Int
    let last: Bool
    @State private var hover = false
    @State private var target = false

    /// What a thumbnail carries when dragged: its id, tagged so that no other text matches.
    private static let prefix = "siliconed-reference:"

    var body: some View {
        VStack(spacing: 3) {
            Image(nsImage: reference.vignette).resizable().scaledToFill()
                .frame(width: Self.side, height: Self.side)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(target ? Color.accentColor : Color(nsColor: .separatorColor),
                                  lineWidth: target ? 2 : 1))
                .overlay(alignment: .topLeading) {
                    Text(verbatim: "\(number)")
                        .font(.caption2.weight(.bold)).monospacedDigit()
                        .foregroundStyle(.white)
                        .frame(width: 18, height: 18)
                        .background(Circle().fill(number == 1 ? Color.accentColor : Color.black.opacity(0.6)))
                        .padding(4)
                }
                .overlay(alignment: .topTrailing) {
                    if hover {
                        Button { app.removeReference(reference.id) } label: {
                            Image(systemName: "xmark.circle.fill").symbolRenderingMode(.palette)
                                .foregroundStyle(.white, .black.opacity(0.6))
                        }
                        .buttonStyle(.plain)
                        .padding(4)
                        .help("Remove")
                    }
                }
            Text(number == 1 ? String(localized: "edited") : String(localized: "image \(number)"))
                .font(.caption2).foregroundStyle(number == 1 ? .primary : .secondary)
        }
        .onHover { hover = $0 }
        .help(Text(verbatim: "\(reference.name) · \(reference.width) × \(reference.height)"))
        .draggable(Self.prefix + reference.id.uuidString) {
            Image(nsImage: reference.vignette).resizable().scaledToFit().frame(width: 48, height: 48)
        }
        .dropDestination(for: String.self) { items, _ in
            guard let item = items.first, item.hasPrefix(Self.prefix),
                  let id = UUID(uuidString: String(item.dropFirst(Self.prefix.count))),
                  app.references.contains(where: { $0.id == id }) else { return false }
            app.moveReference(id, to: number - 1)
            return true
        } isTargeted: { target = $0 }
        .contextMenu {
            if number > 1 {
                Button("Make It Image 1") { app.moveReference(reference.id, to: 0) }
                Button("Move Left") { app.moveReference(reference.id, by: -1) }
            }
            if !last { Button("Move Right") { app.moveReference(reference.id, by: 1) } }
            Divider()
            Button("Replace…") { app.replaceReference(reference.id) }
            Button("Remove", role: .destructive) { app.removeReference(reference.id) }
        }
    }
}

// MARK: - Under the form: the cost, generate, the queue, what went wrong

private struct Footer: View {
    @Bindable var app: AppState

    var body: some View {
        // Read once per drawing: the memory half asks the machine (`MemoryBudget`).
        let form = app.formBlocker
        let refusal = form == nil ? app.memoryRefusal : nil
        let memory = refusal?.text
        let blocker = form ?? memory
        VStack(alignment: .leading, spacing: 10) {
            Button { MainWindow.shared.showExploration() } label: {
                Label("Explore…", systemImage: "square.grid.3x3")
            }
            .buttonStyle(.borderless)
            .help("Sweep seeds, a LoRA's strength, words of the prompt, models or the images to edit — small sketches, fast (⌥⌘G)")
            Divider()
            cost
            Button { app.render() } label: {
                Text(label).fontWeight(.semibold).frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.extraLarge)
            .disabled(blocker != nil)
            .help(Text(verbatim: "⌘↩"))
            // Only while the memory is what greys it out: read it again every 2 s, so that freeing
            // memory elsewhere gives « Generate » back without touching the form.
            .task(id: memory != nil) {
                guard memory != nil else { return }
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(2))
                    app.memoryTick &+= 1
                }
            }

            // **No `fixedSize(vertical:)` on a text of this footer** (nor in `ProblemView`): a refusal of
            // two lines held at its ideal height made the footer — a `safeAreaInset` of the rack —
            // taller than the column: it left the window and the rack scrolled to its end, nothing left
            // to click (a `{` left open, a grid's LoRA axis without strengths). Measured: the column
            // folds the same way with the footer in a `VStack`. Wrapping on its own, the text fits.
            if let reason = blocker {
                Text(reason).font(.caption).foregroundStyle(.secondary)
                // A memory refusal is never a dead end: who holds the memory, and a size that fits.
                if let refusal {
                    if let holders = refusal.holders {
                        Text(holders).font(.caption).foregroundStyle(.tertiary)
                    }
                    if let f = refusal.smaller {
                        Button(String(localized: "Use \(String(f.width)) × \(String(f.height))")) { app.useFormat(f) }
                            .buttonStyle(.link).font(.caption)
                            .help("The largest of this model's sizes that fits in the memory free now")
                    }
                }
            } else if app.diagnosticRunning {
                Text("The diagnostic is running: renders wait for it.").font(.caption).foregroundStyle(.secondary)
            }
            if !app.file.isEmpty { queue }
            if let title = app.forgeInProgress {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text(title).font(.callout).lineLimit(2)
                        Spacer()
                        Button("Stop") { app.cancelForge() }
                            .buttonStyle(.link).font(.caption)
                            .help("Stops the installation: the files already downloaded are kept, the one in progress will start over")
                    }
                    if let line = ForgeProgress.line(app.forgeJournal) {
                        Text(line).font(.caption).monospacedDigit().foregroundStyle(.secondary).lineLimit(2)
                    }
                }
                .help(app.forgeJournal.joined(separator: "\n"))
            }
            if let problem = app.problem {
                ProblemView(problem: problem) { app.problem = nil }
            }
            if let line = app.journalLine {
                Label(line, systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    .help(app.journal.joined(separator: "\n"))
            }
            ForEach(Array(zip(app.profileLines, app.profileWarnings)), id: \.1) { line, raw in
                Label(line, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
                    .help(raw)
            }
        }
        .padding(14)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }

    /// **Rule 7: the cost before the click** — the memory the engine will compare to the machine's, and
    /// the time, learned on this Mac in this session (never a constant from another machine).
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
                    .help(String(localized: "The memory \(card.name) needs at this size (measured peak)"))
                }
                if app.variantCount > 1 {
                    // A prompt with alternatives is a series: how many renders, before the click.
                    Label {
                        Text("\(app.variantCount) renders").monospacedDigit()
                    } icon: { Image(systemName: "square.stack") }
                    .help(app.alternativesHelp)
                }
                Label {
                    if let t = app.plannedCost {
                        Text(verbatim: "≈ " + Figures.duration(t)).monospacedDigit()
                    } else {
                        Text("after one render")
                    }
                } icon: { Image(systemName: "clock") }
                .help("Learned on this Mac: the first render at this size, with this model, sets the estimate")
                Spacer(minLength: 0)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .labelStyle(CompactLabel())
        }
    }

    private var label: String {
        let n = app.variantCount
        if app.busy { return n > 1 ? String(localized: "Add \(n) Renders to Queue") : String(localized: "Add to Queue") }
        if !app.references.isEmpty { return n > 1 ? String(localized: "Edit Image \(n) Ways") : String(localized: "Edit Image") }
        let images = n * max(1, app.batch)
        return images > 1 ? String(localized: "Generate \(images) Images") : String(localized: "Generate")
    }

    /// The renders that wait, in order; each can be removed, or everything stops.
    private var queue: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Waiting · \(app.file.count)")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Button("Stop All", role: .destructive) { app.stopAll() }
                    .buttonStyle(.link).font(.caption)
                    .help("Clear the queue and stop the current render")
            }
            ForEach(Array(app.file.enumerated()), id: \.element.id) { n, job in
                HStack(spacing: 8) {
                    Text(verbatim: "\(n + 1)").font(.caption2).monospacedDigit().foregroundStyle(.tertiary).frame(width: 14)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(job.settings.prompt).font(.caption).lineLimit(1).truncationMode(.tail)
                        Text(job.legend).font(.caption2).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if let e = app.estimation(job) {
                        Text(verbatim: "≈ " + Figures.duration(e)).font(.caption2).monospacedDigit().foregroundStyle(.secondary)
                    }
                    Button { app.remove(job) } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary) }
                        .buttonStyle(.borderless)
                        .help("Remove from the queue")
                }
            }
        }
        .padding(9)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
    }
}

/// An icon and its text, close together.
private struct CompactLabel: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 4) { configuration.icon; configuration.title }
    }
}

/// **A problem, as the user reads it**: the sentence, what to do, and — smaller, selectable — the
/// engine's technical detail when there is one (an issue quotes it).
struct ProblemView: View {
    let problem: Problem
    var close: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
            VStack(alignment: .leading, spacing: 3) {
                Text(problem.title).fontWeight(.medium)
                if let s = problem.suggestion { Text(s).foregroundStyle(.secondary) }
                if let d = problem.detail {
                    Text(d).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(4)
                }
            }
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
            if let close {
                Button(action: close) { Image(systemName: "xmark") }
                    .buttonStyle(.borderless).help("Close")
            }
        }
        .font(.callout)
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 7).fill(Color.red.opacity(0.1)))
    }
}
