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
// **« Explore » stands beside « Generate »** (`Exploration.swift`): the same intention — make images
// from this chain —, many small ones instead of one; not a setting, so not in the rack's header with
// « Models… ». The cost line above counts only what « Generate » makes.
// The queue scrolls past a few rows, so that the footer never pushes the rack out of the column.

import AppKit
import Siliconed
import SwiftUI

struct Sidebar: View {
    @Bindable var app: AppState

    var body: some View {
        // The footer under the scroll view, not a `safeAreaInset` of it: as an inset, the clip view
        // shrank by the footer's height AND the scroll view kept it as a bottom content inset, so the
        // band of rack just above the footer was drawn but took no click (the LoRA row, « Import… »)
        // until scrolled up — hit testing measured on macOS 27, 13/10/2026.
        VStack(spacing: 0) {
            ScrollView(.vertical) {
                RackView(app: app)
            }
            .scrollIndicators(.automatic)
            Footer(app: app)
        }
        // A `.safetensors` or a `.gguf` dropped on the column is imported (model or LoRA).
        .dropDestination(for: URL.self) { urls, _ in
            let s = urls.filter(AppState.importable)
            guard !s.isEmpty, app.importPossible else { return false }
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
                // The chosen model's rules: never a step it would refuse (Qwen-Image-2.1 32, Anima 2048 px).
                Text("Width × height: each side from 512 to \(String(app.formatMaximumSide)) and a multiple of \(String(app.formatMultiple)).")
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
                    .strokeBorder(target ? Color.accentColor : Palette.hairline,
                                  lineWidth: target ? 2 : 1))
                .overlay(alignment: .topLeading) {
                    Text(verbatim: "\(number)")
                        .font(.caption2.weight(.bold)).monospacedDigit()
                        .foregroundStyle(.white)
                        .frame(width: 18, height: 18)
                        .background(Circle().fill(number == 1 ? Color.accentColor : Color.black.opacity(0.6)))
                        .padding(4)
                }
                .overlay(alignment: .bottomTrailing) {
                    if hover, app.cropRefusal(width: reference.width, height: reference.height) == nil {
                        Button { app.beginCrop(reference) } label: {
                            Image(systemName: "crop").font(.caption2.weight(.semibold))
                                .foregroundStyle(.white)
                                .frame(width: 18, height: 18)
                                .background(Circle().fill(Color.black.opacity(0.6)))
                        }
                        .buttonStyle(.plain)
                        .padding(4)
                        .help("Crop this image (the frame keeps at least \(String(Format.minimumSide)) px a side)")
                    }
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
            Button("Crop…") { app.beginCrop(reference) }
                .disabled(app.cropRefusal(width: reference.width, height: reference.height) != nil)
            Button("Replace…") { app.replaceReference(reference.id) }
            Button("Remove", role: .destructive) { app.removeReference(reference.id) }
        }
    }
}

// MARK: - Under the form: the cost, generate, the queue, what went wrong

private struct Footer: View {
    @Bindable var app: AppState

    var body: some View {
        // Read once per drawing: the memory half computes the need against B (`MemoryBudget`).
        let form = app.formBlocker
        let refusal = form == nil ? app.memoryRefusal : nil
        let memory = refusal?.text
        let blocker = form ?? memory
        VStack(alignment: .leading, spacing: 10) {
            cost
            // **No `fixedSize(vertical:)` on a text of this footer** (nor in `ProblemView`): a refusal of
            // two lines held at its ideal height made the footer taller than the column: it left the
            // window and the rack scrolled to its end, nothing left to click (a `{` left open, a grid's
            // LoRA axis without strengths) — measured with the footer as a `safeAreaInset`, the same
            // in a `VStack`. Wrapping on its own, the text fits.
            // What stops « Generate » is said above it, in the body's size: it is what to do next, read
            // before the figures — the cost line above is caption-sized.
            if let reason = blocker {
                Text(reason).font(.callout)
                // A memory refusal is never a dead end: a size this Mac renders.
                if let f = refusal?.smaller {
                    Button(String(localized: "Use \(String(f.width)) × \(String(f.height))")) { app.useFormat(f) }
                        .buttonStyle(.link).font(.caption)
                        .help("The largest of this model's sizes this Mac renders")
                }
            } else if case let .running(done, total, line) = app.diagnostic {
                // Before the first render, the measure it waits for: why nothing starts yet, and how far.
                Text(app.settingsRead ? "The diagnostic is running: renders wait for it."
                                      : "Siliconed measures this Mac before its first render — about a minute.")
                    .font(.callout)
                Text(verbatim: "\(line) · \(done + 1)/\(total)").font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                Button { app.render() } label: {
                    Text(label).fontWeight(.semibold).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(blocker != nil)
                .help(Text(verbatim: "⌘↩"))
                Button { MainWindow.shared.showExploration() } label: {
                    Label("Explore", systemImage: "square.grid.3x3")
                }
                .buttonStyle(.bordered)
                .fixedSize()
                .help("Sweep seeds, a LoRA's strength, words of the prompt, models or the images to edit — small sketches, fast (⌥⌘G)")
            }
            .controlSize(.extraLarge)
            if app.busy { queue }
            // In a mirror: the other Mac busy with renders that are not this Mac's — why this one waits.
            if app.othersQueued > 0, let mirror = app.mirror {
                Label(app.othersSeconds.map { s in
                    String(localized: "Renders for others on “\(mirror.macName)”: \(app.othersQueued), about \(Duration.seconds(s).formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated)))")
                } ?? String(localized: "Renders for others on “\(mirror.macName)”: \(app.othersQueued)"),
                      systemImage: "person.2")
                    .font(.caption).foregroundStyle(.secondary)
            }
            // The models being copied from the library's disk to this Mac (`AppState+Place.swift`).
            if let c = app.copying {
                VStack(alignment: .leading, spacing: 4) {
                    Text(c.title).font(.callout).lineLimit(2)
                    ProgressView(value: c.fraction).controlSize(.small)
                }
                .help("The library is on another disk: what a render reads is copied to this Mac first, once, and removed when Siliconed quits")
            }
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
                    if let line = app.forgeLine {
                        Text(line).font(.caption).monospacedDigit().foregroundStyle(.secondary).lineLimit(2)
                    }
                }
                .help(app.forgeHelp)
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
                    .help(String(localized: "The memory \(card.name) needs at this size (computed)"))
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

    /// **What remains to be done**: the running render first — a batch is one job, its images rendered
    /// one after the other, so an empty wait list said nothing of the images still to come —,
    /// then the renders that wait, in order; each can be removed, or everything stops.
    private var queue: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let job = app.currentJob { running(job) }
            if !app.file.isEmpty { waiting }
        }
        .padding(9)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
    }

    /// The running render: where its batch is, the time left, and a stop that leaves the queue going.
    private func running(_ job: Job) -> some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.mini).frame(width: 14)
            VStack(alignment: .leading, spacing: 1) {
                Text(job.settings.prompt).font(.caption).lineLimit(1).truncationMode(.tail)
                Text(app.batchSize > 1
                     ? job.legend + " · " + String(localized: "image \(app.batchImage + 1)/\(app.batchSize)")
                     : job.legend)
                    .font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            if app.remaining(at: .now) != nil {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    Text(verbatim: "≈ " + Figures.duration(app.remaining(at: context.date) ?? 0))
                        .font(.caption2).monospacedDigit().foregroundStyle(.secondary)
                }
            }
            Button { app.cancel() } label: { Image(systemName: "stop.circle.fill").foregroundStyle(.secondary) }
                .buttonStyle(.borderless)
                .help("Stop (⌘.)")
        }
        .frame(height: Self.rowHeight)
    }

    private var waiting: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Waiting · \(app.file.count)")
                    .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                Spacer()
                Button("Stop All", role: .destructive) { app.stopAll() }
                    .buttonStyle(.link).font(.caption)
                    .help("Clear the queue and stop the current render")
            }
            // **The queue scrolls past a few rows**: grown with each job, it pushed the footer above the
            // window's height and the rack's scroll view, squeezed to nothing, could not be scrolled back.
            // A fixed height, not one measured: a scroll view in this footer takes what it is offered.
            if app.file.count > Self.visibleRows {
                ScrollView(.vertical) { rows }
                    .frame(height: (CGFloat(Self.visibleRows) + 0.5) * (Self.rowHeight + Self.rowSpacing))
            } else {
                rows
            }
        }
    }

    private static let visibleRows = 4
    private static let rowHeight: CGFloat = 30
    private static let rowSpacing: CGFloat = 6

    private var rows: some View {
        VStack(alignment: .leading, spacing: Self.rowSpacing) {
            ForEach(Array(app.file.enumerated()), id: \.element.id) { n, job in
                HStack(spacing: 8) {
                    Text(verbatim: "\(n + 1)").font(.caption2).monospacedDigit().foregroundStyle(.secondary).frame(width: 14)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(job.settings.prompt).font(.caption).lineLimit(1).truncationMode(.tail)
                        Text(job.legend).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    }
                    Spacer()
                    if let e = app.estimation(job) {
                        Text(verbatim: "≈ " + Figures.duration(e)).font(.caption2).monospacedDigit().foregroundStyle(.secondary)
                    }
                    Button { app.remove(job) } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
                        .buttonStyle(.borderless)
                        .help("Remove from the queue")
                }
                .frame(height: Self.rowHeight)
            }
        }
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
        .background(RoundedRectangle(cornerRadius: 7).fill(Palette.failure.opacity(0.1)))
    }
}
