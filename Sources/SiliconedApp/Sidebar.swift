// **The window's left column: the rack, and under it what the chain is doing**.
//
//     the rack     the chain top to bottom, each module with its settings (`Rack.swift`)
//     ─────────
//     the measure of this Mac under way, the queue, a copy or an installation, what went wrong
//
// No form beside the rack: a setting lives in the box of the stage that reads it. No mask, no setting
// between two steps: what is set is taken whole at the click (`Job`), and changing it afterwards
// prepares the next render. **« Generate » is not here**: it is at the end of the text, in the Prompt
// Lab under the image (`LabGenerate.swift`), with its cost and what stops it — the hand is there when
// the idea is written. The footer is drawn only when it has something to say.
// The queue scrolls past a few rows, so that the footer never pushes the rack out of the column.

import AppKit
import Siliconed
import SwiftUI

struct Sidebar: View {
    @Bindable var app: AppState
    @State private var chainAway = false

    var body: some View {
        // The footer under the scroll view, not a `safeAreaInset` of it: as an inset, the clip view
        // shrank by the footer's height AND the scroll view kept it as a bottom content inset, so the
        // band of rack just above the footer was drawn but took no click (the LoRA row, « Import… »)
        // until scrolled up — hit testing measured on macOS 27, 13/10/2026.
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    RackView(app: app, chainAway: $chainAway)
                }
                .scrollIndicators(.automatic)
                // Once the chain's boxes have scrolled away (setting the DiT, its LoRAs), the chain stays in sight in small,
                // lit as they are; a click scrolls back to them (`RackMiniChain.swift`).
                .overlay(alignment: .top) {
                    if chainAway {
                        Button { withAnimation(Motion.pick(.easeOut(duration: 0.3))) { proxy.scrollTo(RackView.top, anchor: .top) } } label: {
                            // A band the column's width, flush with its top, as a section header: a floating plate let
                            // the rack show above it.
                            MiniChain(app: app)
                                .padding(.vertical, 8)
                                .frame(maxWidth: .infinity)
                                .background(.bar)
                                .overlay(alignment: .bottom) { Rectangle().fill(Palette.hairline).frame(height: 1) }
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .help("The chain, while the rack scrolls: a click brings its boxes back")
                        .transition(.opacity)
                    }
                }
                .animation(Motion.pick(.easeOut(duration: 0.2)), value: chainAway)
            }
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

// MARK: - Under the rack: the queue, what went wrong

private struct Footer: View {
    @Bindable var app: AppState

    /// Something to say: an empty bar under the rack would be a band of nothing.
    private var speaks: Bool {
        if case .running = app.diagnostic { return true }
        return app.busy || app.othersQueued > 0 || app.copying != nil || app.forgeInProgress != nil || app.problem != nil
            || app.journalLine != nil || !app.settingsLines.isEmpty
    }

    var body: some View {
        if speaks {
            VStack(alignment: .leading, spacing: 10) {
                // **No `fixedSize(vertical:)` on a text of this footer** (nor in `ProblemView`): a text of two lines held at its
                // ideal height made the footer taller than the column: it left the window and the rack scrolled to its end,
                // nothing left to click — measured with the footer as a `safeAreaInset`, the same in a `VStack`. Wrapping on its
                // own, the text fits.
                if case let .running(done, total, line) = app.diagnostic {
                    // Before the first render, the measure it waits for: why nothing starts yet, and how far.
                    Text(app.settingsRead ? "The diagnostic is running: renders wait for it."
                                          : "Siliconed measures this Mac before its first render — about a minute.")
                        .font(.callout)
                    Text(verbatim: "\(line) · \(done + 1)/\(total)").font(.caption).foregroundStyle(.secondary)
                }
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
                ForEach(Array(zip(app.settingsLines, app.settingsWarnings)), id: \.1) { line, raw in
                    Label(line, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
                        .help(raw)
                }
            }
            .padding(14)
            .background(.bar)
            .overlay(alignment: .top) { Divider() }
        }
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
            // Still, not a spinner: the one under the image is the render's only one.
            Image(systemName: "play.fill").font(.system(size: 8, weight: .bold))
                .foregroundStyle(Palette.live).frame(width: 14)
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
            Button { app.cancel() } label: { Image(systemName: "stop.circle.fill").foregroundStyle(.secondary).frame(width: 24, height: 24).contentShape(Rectangle()) }
                .buttonStyle(.borderless)
                .help("Stop (⌘.)")
                .accessibilityLabel(Text("Stop"))
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
                    Button { app.remove(job) } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary).frame(width: 24, height: 24).contentShape(Rectangle()) }
                        .buttonStyle(.borderless)
                        .help("Remove from the queue")
                        .accessibilityLabel(Text("Remove from the queue"))
                }
                .frame(height: Self.rowHeight)
            }
        }
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
                Button(action: close) { Image(systemName: "xmark").frame(width: 24, height: 24).contentShape(Rectangle()) }
                    .buttonStyle(.borderless).help("Close")
                    .accessibilityLabel(Text("Close"))
            }
        }
        .font(.callout)
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 7).fill(Palette.failure.opacity(0.1)))
    }
}
