// **The window's column: the chain from the idea to the button, and pinned at its foot « Generate »**.
//
//     the chain    the text, the images, the stages top to bottom, each with its settings (`Rack.swift`) — it scrolls
//     ─────────
//     an installation, the renders of others, what went wrong, the journal; what stops the render, the axes of a series,
//     « Generate » (`LabGenerate.swift`) — the queue is time's, on the right (`HistoryBanner.swift`)
//
// No form beside the chain: a setting lives in the box of the stage that reads it. No mask, no setting between two steps:
// what is set is taken whole at the click (`Job`), and changing it afterwards prepares the next render. **The column
// scrolls, and it is meant to**: on a 13-inch screen it holds at rest, a LoRA added makes it scroll; once the text has
// left its top, the chain in small is pinned there, with the text's first words (`RackMiniChain.swift`).

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
        // until scrolled up — hit testing measured on macOS 27.
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
                            // the rack show above it. The text's head above the chain: what is being made, then how.
                            VStack(alignment: .leading, spacing: 6) {
                                if !app.prompt.isEmpty {
                                    Text(TextBox.line(app.prompt)).font(.callout).lineLimit(1).truncationMode(.tail)
                                        .padding(.horizontal, 14)
                                }
                                MiniChain(app: app).frame(maxWidth: .infinity)
                            }
                                .padding(.vertical, 8)
                                .frame(maxWidth: .infinity)
                                .background(.bar)
                                .overlay(alignment: .bottom) { Rectangle().fill(Palette.hairline).frame(height: 1) }
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                                .help("The text and the chain, while the column scrolls: a click brings them back")
                        .transition(.opacity)
                    }
                }
                .animation(Motion.pick(.easeOut(duration: 0.2)), value: chainAway)
                // `silicontrol rack top|dit`: the scroll the hand would make — asked while the column was folded, it is
                // made once the column is drawn (`RackColumn` unfolds it).
                .onChange(of: app.rackScroll?.count) { scroll(proxy) }
                .onAppear { scroll(proxy) }
            }
            Footer(app: app)
        }
    }

    /// The scroll `silicontrol rack` asked, made once.
    private func scroll(_ proxy: ScrollViewProxy) {
        guard let place = app.rackScroll?.place, place == .top || place == .dit else { return }
        app.rackScroll = nil
        withAnimation(Motion.pick(.easeOut(duration: 0.3))) {
            proxy.scrollTo(place == .top ? RackView.top : RackView.dit, anchor: .top)
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
                TextField("Size", text: $text, prompt: Text(verbatim: Figures.size(832, 1216)))
                    .labelsHidden()
                    .frame(width: 140)
                    .onSubmit(apply)
                Button("Use", action: apply).keyboardShortcut(.defaultAction).tint(Palette.control)
            }
            if let refusal {
                Text(refusal.title).font(.caption).foregroundStyle(Palette.failure)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                // The chosen model's rules (`Format.check`): never a multiple nor a side it would refuse.
                Text("Width × height: each side from 512 to \(String(app.formatMaximumSide)) and a multiple of \(String(app.formatMultiple)).")
                    .font(.caption).foregroundStyle(Palette.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(width: 280)
        .onAppear { text = Figures.size(app.format.width, app.format.height) }
    }

    private func apply() {
        refusal = app.setCustomFormat(text)
        if refusal == nil { dismiss() }
    }
}

// MARK: - The column’s foot: what went wrong, « Generate »

private struct Footer: View {
    @Bindable var app: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // **No `fixedSize(vertical:)` on a text of this footer** (nor in `ProblemView`): a text of two lines held at its
            // ideal height made the footer taller than the column: it left the window and the rack scrolled to its end,
            // nothing left to click — measured with the footer as a `safeAreaInset`, the same in a `VStack`. Wrapping on its
            // own, the text fits.
            // In a mirror: the other Mac busy with renders that are not this Mac's — why this one waits.
            if app.othersQueued > 0, let mirror = app.mirror {
                Label(app.othersSeconds.map { s in
                    String(localized: "Renders for others on “\(mirror.macName)”: \(app.othersQueued), about \(Figures.duration(s))")
                } ?? String(localized: "Renders for others on “\(mirror.macName)”: \(app.othersQueued)"),
                      systemImage: "person.2")
                    .font(.caption).foregroundStyle(Palette.secondary)
            }
            if let title = app.libraryJob {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text(title).font(.callout).lineLimit(2)
                        Spacer()
                        Button("Stop") { app.cancelLibraryJob() }
                            .buttonStyle(.chartLink).font(.caption)
                            .help(app.libraryStopHelp)
                    }
                    if let line = app.libraryJobLine {
                        Text(line).font(.caption).monospacedDigit().foregroundStyle(Palette.secondary).lineLimit(2)
                    }
                }
                .help(app.libraryJobHelp)
            }
            if let problem = app.problem {
                ProblemView(problem: problem) { app.problem = nil }
            }
            if let line = app.journalLine {
                Label(line, systemImage: "info.circle")
                    .font(.caption).foregroundStyle(Palette.secondary).lineLimit(2)
                    .help(app.journal.joined(separator: "\n"))
            }
            GenerateControls(app: app)
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .overlay(alignment: .top) { Divider() }
    }
}

/// **A problem, as the user reads it**: the sentence, what to do, and — smaller, selectable — the
/// engine's technical detail when there is one (an issue quotes it).
struct ProblemView: View {
    let problem: Problem
    var close: (() -> Void)? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Palette.failure)
            VStack(alignment: .leading, spacing: 3) {
                Text(problem.title).fontWeight(.medium)
                if let s = problem.suggestion { Text(s).foregroundStyle(Palette.secondary) }
                if let d = problem.detail {
                    Text(d).font(.caption.monospaced()).foregroundStyle(Palette.secondary).lineLimit(4)
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
