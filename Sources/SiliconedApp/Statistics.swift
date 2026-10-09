// **The statistics window**: a floating palette that says where the render stands, what each
// stage cost, what remains — for it and for the queue —, and what the last render weighed.
//
// Everything comes from the engine's events (`Measures`) and from the speeds learned in the session: nothing
// is estimated from a constant, and nothing is written. A wait for a model or a format
// never rendered has no estimate, and the window says so rather than inventing one.

import AppKit
import Siliconed
import SwiftUI

struct Statistics: View {
    let app: AppState

    var body: some View {
        // The timer advances every second, even between two events (a Krea 2 step lasts 35 s).
        TimelineView(.periodic(from: .now, by: 1)) { context in
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    inProgress(now: context.date)
                    stages
                    waitingQueue(now: context.date)
                    last
                    session
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(minWidth: 320, minHeight: 360)
    }

    private var m: Measures { app.measurements }

    // ── The render in progress ──

    @ViewBuilder private func inProgress(now: Date) -> some View {
        Section(title: "Current Render", right: m.job.map(\.legend)) {
            if app.inProgress, let begin = m.begin {
                ProgressView(value: app.fraction)
                StatLine("elapsed", Figures.duration(now.timeIntervalSince(begin)))
                if let remaining = app.remaining(at: now) {
                    StatLine("remaining", "≈ " + Figures.duration(remaining))
                    StatLine("expected end", Figures.hour(now.addingTimeInterval(remaining)))
                } else {
                    StatLine("remaining", String(localized: "after the first full step"))
                }
                if app.batchSize > 1 {
                    StatLine("image", String(localized: "\(app.batchImage + 1) of \(app.batchSize)"))
                }
            } else {
                Text(app.history.isEmpty ? "No renders yet." : "Nothing is running.")
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
    }

    // ── The stages: what each one cost (render in progress, or the last finished) ──

    private var inProgressOrNone: String { String(localized: "running…") }

    @ViewBuilder private var stages: some View {
        if m.job != nil {
            Section(title: "Stages") {
                if let tokens = m.tokens, let t = m.tokenizer, let e = m.encoder {
                    StatLine("text", String(localized: "\(tokens) tokens · \(Figures.seconds(t + e))"),
                          detail: String(localized: "tokenizer \(Figures.seconds(t)) · encoder \(Figures.seconds(e))"))
                } else {
                    StatLine("text", app.stage == .text ? inProgressOrNone : "—")
                }
                if m.job?.references.isEmpty == false {
                    StatLine("image", m.encoding.map(Figures.seconds) ?? (app.stage == .image ? inProgressOrNone : "—"))
                }
                dit
                StatLine("VAE", m.decodings.isEmpty
                      ? (app.stage == .decoding ? inProgressOrNone : "—")
                      : m.decodings.map(Figures.seconds).joined(separator: " · "))
            }
        }
    }

    /// The DiT: the steps done out of those planned, the average, and each step's duration as bars —
    /// the first re-reads the map, a slow step shows.
    @ViewBuilder private var dit: some View {
        let image = min(app.batchImage, max(0, m.stepSeconds.count - 1))
        let durations = m.stepSeconds.indices.contains(image) ? m.stepSeconds[image] : []
        let plannedCount = m.plan?.denoising.evaluations ?? 0
        StatLine("DiT", plannedCount > 0 ? String(localized: "\(durations.count)/\(plannedCount) steps") : "—",
              detail: m.speed.map { String(localized: "average \(Figures.seconds($0.full)) per full step") })
        if plannedCount > 0 {
            Bars(durations: durations, plannedCount: plannedCount, reduced: m.reduced, mean: m.speed?.full)
                .frame(height: 38)
                .padding(.top, 2)
        }
    }

    // ── The queue ──

    @ViewBuilder private func waitingQueue(now: Date) -> some View {
        let (remaining, isComplete) = app.remainingInQueue(at: now)
        Section(title: "Queue", right: app.file.isEmpty ? String(localized: "empty")
                                                             : String(localized: "\(app.file.count) waiting")) {
            if !app.file.isEmpty || app.inProgress {
                StatLine("all done in", isComplete ? "≈ " + Figures.duration(remaining)
                                                : String(localized: "at least \(Figures.duration(remaining))"))
                StatLine("around", isComplete ? Figures.hour(now.addingTimeInterval(remaining))
                                      : String(localized: "\(Figures.hour(now.addingTimeInterval(remaining))) at the earliest"))
            }
            ForEach(Array(app.file.enumerated()), id: \.element.id) { n, t in
                StatLine(verbatim: "\(n + 1)", t.legend,
                      detail: app.estimation(t).map { "≈ " + Figures.duration($0) }
                        ?? String(localized: "never rendered here: no estimate"))
            }
        }
    }

    // ── The last finished render ──

    @ViewBuilder private var last: some View {
        if let c = m.timings, let e = m.footprints {
            Section(title: "Last Image") {
                StatLine("total", Figures.seconds(c.total))
                StatLine("time", [(String(localized: "text"), c.text), (String(localized: "image"), c.encoding),
                                ("DiT", c.denoising), ("VAE", c.decoding)]
                    .filter { $0.1 > 0 }.map { "\($0.0) \(Figures.seconds($0.1))" }.joined(separator: " · "))
                // The peaks, read before each release — those the CLI prints.
                StatLine("memory (peak)", [(String(localized: "text"), e.encoder), (String(localized: "image"), e.encoding),
                                        ("DiT", e.denoising), ("VAE", e.end)]
                    .filter { $0.1 > 0 }.map { "\($0.0) \(Figures.memory($0.1))" }.joined(separator: " · "))
            }
        }
    }

    // ── The session ──

    private var session: some View {
        Section(title: "Session") {
            StatLine("images", "\(app.sessionImages)")
            StatLine("compute", Figures.duration(app.sessionSeconds))
            ForEach(app.speeds.sorted { $0.key < $1.key }, id: \.key) { key, v in
                StatLine(verbatim: label(key), String(localized: "\(Figures.seconds(v.full)) per step"))
            }
        }
    }

    /// A learned speed's key (`Speed.key`: `z-image|1024×1024|edit1|LoRA`) as the user reads it:
    /// `Z-Image Turbo · 1024 × 1024 · edit, 1 image · LoRA`.
    private func label(_ key: String) -> String {
        key.split(separator: "|").enumerated().map { i, part -> String in
            if i == 0 { return app.modelName(String(part)) }
            if part.hasPrefix("edit"), let n = Int(part.dropFirst(4)) {
                return n == 1 ? String(localized: "edit, 1 image") : String(localized: "edit, \(n) images")
            }
            return part.replacingOccurrences(of: "×", with: " × ")
        }.joined(separator: " · ")
    }
}

/// A section title, a complement on the right, then its rows — on a card.
private struct Section<Content: View>: View {
    let title: LocalizedStringKey
    var right: String? = nil
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(title).textCase(.uppercase).font(.caption.weight(.semibold)).tracking(0.7).foregroundStyle(.secondary)
                Spacer()
                if let right { Text(right).font(.caption).foregroundStyle(.secondary).lineLimit(1) }
            }
            content
        }
        .padding(11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor).opacity(0.8)))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color(nsColor: .separatorColor)))
    }
}

/// `label ……… value`, and a discreet detail below.
private struct StatLine: View {
    let label: Text, value: String
    var detail: String?
    init(_ label: LocalizedStringKey, _ value: String, detail: String? = nil) {
        self.label = Text(label); self.value = value; self.detail = detail
    }
    init(verbatim label: String, _ value: String, detail: String? = nil) {
        self.label = Text(verbatim: label); self.value = value; self.detail = detail
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(alignment: .firstTextBaseline) {
                label.foregroundStyle(.secondary)
                Spacer(minLength: 12)
                Text(value).monospacedDigit().multilineTextAlignment(.trailing)
            }
            if let detail {
                Text(detail).font(.caption2).foregroundStyle(.tertiary).frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .font(.callout)
    }
}

/// One bar per evaluation: full for those done (height = duration), hollow for those
/// that remain (at the average of the full steps). The first ones, at half-size (the spectral), are
/// shorter by nature: they are drawn paler, so as not to pass for an anomaly.
private struct Bars: View {
    let durations: [Double]
    let plannedCount: Int
    let reduced: Int
    let mean: Double?

    var body: some View {
        GeometryReader { g in
            let n = max(plannedCount, durations.count)
            let ceiling = max(durations.max() ?? 0, mean ?? 0, 0.001)
            let width = (g.size.width - CGFloat(n - 1) * 3) / CGFloat(max(n, 1))
            HStack(alignment: .bottom, spacing: 3) {
                ForEach(0..<n, id: \.self) { i in
                    let mergeDone = i < durations.count
                    let reducedAverage = i < reduced
                    let value = mergeDone ? durations[i] : (reducedAverage ? ceiling * 0.35 : (mean ?? ceiling * 0.5))
                    RoundedRectangle(cornerRadius: 2)
                        .fill(mergeDone ? Color.accentColor.opacity(reducedAverage ? 0.5 : 1) : Color.secondary.opacity(0.18))
                        .frame(width: width, height: max(2, g.size.height * CGFloat(value / ceiling)))
                        .help(mergeDone ? (reducedAverage ? String(localized: "step \(i + 1), half-size: \(Figures.seconds(value))")
                                               : String(localized: "step \(i + 1): \(Figures.seconds(value))"))
                                    : String(localized: "step \(i + 1): pending"))
                }
            }
            .frame(maxHeight: .infinity, alignment: .bottom)
        }
    }
}
