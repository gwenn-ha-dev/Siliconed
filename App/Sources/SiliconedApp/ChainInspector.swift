// **The « Chain » inspector**: a floating panel (`Inspectors.swift`) that says where the render stands in the chain's
// words — each stage, its module, its weights, its time (what Core AI compiled for it included), the unit that computed
// it —, then what remains for the queue, the last image and the session. What the Mac does in time — the memory, the
// TFLOP/s, Core AI's work — is the Machine window's (`MachineInspector.swift`).
//
// Its lines keep their number while a render runs: a model's first render compiles in bursts up to its last band of
// decoding, and a line that came and went with each compilation moved everything under it, several times a second.
//
// Everything comes from the engine's events (`Measures`) and from the speeds learned in the session: nothing
// is estimated from a constant, and nothing is written. A wait for a model or a format never rendered has no estimate,
// and the panel says so rather than inventing one.

import AppKit
import Siliconed
import SwiftUI

struct ChainInspector: View {
    let app: AppState
    let focus: InspectorFocus

    var body: some View {
        ScrollViewReader { reader in
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    // The clock advances every second, even between two events (a step can last 35 s): only the
                    // sections that read it are drawn again with it, and only while it says something — at rest, with
                    // nothing waiting, it stops (a panel left open redrew both sections every second for nothing). The
                    // chain reads it for a compilation's burst (`Measures.burst`), said for 1.5 s past its last event.
                    TimelineView(.animation(minimumInterval: 1, paused: !app.inProgress)) { context in inProgress(now: context.date) }
                    TimelineView(.animation(minimumInterval: 1, paused: !app.inProgress)) { context in stages(now: context.date) }
                    TimelineView(.animation(minimumInterval: 1, paused: !app.inProgress && app.file.isEmpty)) { context in
                        waitingQueue(now: context.date)
                    }
                    last
                    session
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            // A stage's line in the rack opens the panel at its own line.
            .onChange(of: focus.asked) {
                if let stage = focus.stage { withAnimation(Motion.pick(.default)) { reader.scrollTo(stage, anchor: .top) } }
            }
        }
        .frame(minWidth: 340, minHeight: 360)
    }

    private var m: Measures { app.measurements }

    // ── The render in progress ──

    @ViewBuilder private func inProgress(now: Date) -> some View {
        PanelSection(title: "Current Render", right: m.job.map(\.legend)) {
            if app.inProgress, let begin = m.begin {
                ProgressView(value: app.fraction).tint(Palette.live)
                StatLine("elapsed", Figures.duration(now.timeIntervalSince(begin)))
                let remaining = app.compilingCoreAI ? nil : app.remaining(at: now)
                StatLine("remaining", app.compilingCoreAI ? String(localized: "after the compilation")
                                      : remaining.map { "≈ " + Figures.duration($0) } ?? String(localized: "after the first step"))
                StatLine("expected end", remaining.map { Figures.hour(now.addingTimeInterval($0)) } ?? "—")
                if app.batchSize > 1 {
                    StatLine("image", String(localized: "\(app.batchImage + 1) of \(app.batchSize)"))
                }
            } else {
                Text(app.history.isEmpty ? "No renders yet." : "Nothing is running.")
                    .font(.callout).foregroundStyle(Palette.secondary)
            }
        }
    }

    // ── The chain, stage by stage: what each is, and what it cost (render in progress, or the last finished) ──

    private var inProgressOrNone: String { String(localized: "running…") }

    /// The stages the render runs, or those of the chain on the rack before any render.
    private var shownStages: [Engine.Stage] {
        if let stages = m.plan?.stages { return stages }
        var s: [Engine.Stage] = [.text]
        // The rack's own condition (`drawsImageEncoder`): an img2img start draws the encoder too, and a click on its line in the
        // rack came here to a line that was not drawn.
        if m.job?.references.isEmpty == false || app.moduleNames.encoding != nil && app.drawsImageEncoder { s.append(.image) }
        return s + [.denoising, .decoding]
    }

    private func stages(now: Date) -> some View {
        PanelSection(title: "Chain") {
            ForEach(Array(shownStages.enumerated()), id: \.element) { n, stage in
                if n > 0 { Divider() }
                let held = m.held.first { $0.stage == stage }
                StageLine(title: title(stage), what: what(stage), time: time(stage), note: note(stage, held, now: now),
                          units: held?.neuralEngine.flatMap { $0 > 0 ? $0 : nil }, highlighted: focus.stage == stage)
                .id(stage)
            }
        }
    }

    /// What the stage's time holds besides its computing: the session's cache, what Core AI compiled for it.
    private func note(_ stage: Engine.Stage, _ held: Engine.Held?, now: Date) -> String? {
        var notes: [String] = []
        if stage == .text && m.textCached { notes.append(String(localized: "read from the session’s cache")) }
        if app.inProgress && m.compiling(stage, at: now) {
            notes.append(String(localized: "Core AI compiling…"))
        } else if let c = held?.compilation, c > 0 {
            notes.append(String(localized: "\(Figures.seconds(c)) compiled by Core AI"))
        }
        return notes.isEmpty ? nil : notes.joined(separator: " · ")
    }

    private func title(_ stage: Engine.Stage) -> LocalizedStringKey {
        switch stage {
        case .text: "Text encoder"
        case .image: "Image encoder"
        case .denoising: "DiT"
        case .decoding: "VAE decoder"
        }
    }

    /// The module and how its weights are stored; the DiT's model — the one the times below are of, the last render's — and
    /// its latent.
    private func what(_ stage: Engine.Stage) -> String {
        let names = app.moduleNames, formats = app.weightFormats
        let parts: [String?] = switch stage {
        case .text: [names.text, formats?.text.map { String(localized: "weights \($0.label)") }]
        case .image: [names.encoding, formats?.vae.map { String(localized: "weights \($0.label)") }]
        case .denoising: [m.job?.settings.modelName ?? app.card?.name, formats?.dit.map { String(localized: "weights \($0.label)") },
                          app.chain.map { String(localized: "latent · \($0.denoising.space.channels) ch") }]
        case .decoding: [names.decoding, formats?.vae.map { String(localized: "weights \($0.label)") }]
        }
        return parts.compactMap { $0 }.joined(separator: " · ")
    }

    private func time(_ stage: Engine.Stage) -> String {
        guard m.job != nil else { return "—" }
        switch stage {
        case .text:
            if let tokens = m.tokens, let t = m.tokenizer, let e = m.encoder {
                return String(localized: "\(tokens) tokens · \(Figures.seconds(t + e))")
            }
        case .image:
            if let e = m.encoding { return Figures.seconds(e) }
        case .denoising:
            let image = min(app.batchImage, max(0, m.stepSeconds.count - 1))
            let done = m.stepSeconds.indices.contains(image) ? m.stepSeconds[image].count : 0
            if m.planned > 0 { return String(localized: "\(done)/\(m.planned) steps") }
        case .decoding:
            if !m.decodings.isEmpty { return m.decodings.map(Figures.seconds).joined(separator: " · ") }
        }
        return app.stage == stage && app.inProgress ? inProgressOrNone : "—"
    }

    // ── The queue ──

    @ViewBuilder private func waitingQueue(now: Date) -> some View {
        let (remaining, isComplete) = app.remainingInQueue(at: now)
        PanelSection(title: "Queue", right: app.file.isEmpty ? String(localized: "empty")
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
        // Its Neural Engine's share is the chain's, above; its memory, the Machine window's.
        if let c = m.timings {
            PanelSection(title: "Last Image") {
                StatLine("total", Figures.seconds(c.total))
            }
        }
    }

    // ── The session ──

    private var session: some View {
        PanelSection(title: "Session") {
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
            return Format.parse(String(part)).map { Figures.size($0.width, $0.height) } ?? String(part)
        }.joined(separator: " · ")
    }
}

/// **A stage of the chain**: its name and its time, then what it is, what its time holds, who computed it —
/// highlighted when the rack asked for it.
private struct StageLine: View {
    let title: LocalizedStringKey
    let what: String
    let time: String
    let note: String?
    /// The Neural Engine's share of the stage's arithmetic; `nil` when it took none (the text encoder and the VAE run on
    /// the GPU alone, by the engine's construction: `Layers`, `BandStage`).
    let units: Double?
    let highlighted: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(alignment: .firstTextBaseline) {
                Text(title).fontWeight(.medium)
                Spacer(minLength: 12)
                Text(time).monospacedDigit().foregroundStyle(Palette.secondary)
            }
            .font(.callout)
            if !what.isEmpty { Text(what).font(.caption).foregroundStyle(Palette.secondary) }
            if let note { Text(note).font(.caption).monospacedDigit() }
            if let units { UnitShare(neuralEngine: units) }
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 6).fill(highlighted ? Palette.live.opacity(0.12) : .clear))
    }
}

/// **Who computed, GPU and Neural Engine**: the share of the arithmetic of the products by weights each unit ran — the
/// engine's own count (`Ledger`), product by product, not an estimate. A bar split at the share, both figures under it;
/// the Neural Engine's part in what runs, the GPU's neutral.
struct UnitShare: View {
    let neuralEngine: Double

    static let help = String(localized: "Of the arithmetic of the products by weights, the share each unit computed — counted by the engine, product by product")

    /// Compact, for a table: « GPU 72% · ANE 28% » (`72 %` in French: `Figures.percent`), or the one unit that did it all.
    static func words(_ rate: Double) -> String {
        let ane = Int((rate * 100).rounded())
        switch ane {
        case 0: return String(localized: "GPU \(Figures.percent(100))")
        case 100: return String(localized: "ANE \(Figures.percent(100))")
        default: return String(localized: "GPU \(Figures.percent(100 - ane)) · ANE \(Figures.percent(ane))")
        }
    }

    var body: some View {
        let ane = Int((neuralEngine * 100).rounded())
        VStack(alignment: .leading, spacing: 2) {
            GeometryReader { g in
                HStack(spacing: ane == 0 || ane == 100 ? 0 : 1.5) {
                    Rectangle().fill(Color.secondary.opacity(0.45)).frame(width: g.size.width * (1 - neuralEngine))
                    Rectangle().fill(Palette.live)
                }
            }
            .frame(height: 5)
            .clipShape(Capsule())
            HStack {
                Text("GPU \(Figures.percent(100 - ane))").foregroundStyle(Palette.secondary)
                Spacer()
                Text("Neural Engine \(Figures.percent(ane))").foregroundStyle(ane > 0 ? Palette.live : Palette.secondary)
            }
            .font(.caption2.weight(.medium)).monospacedDigit()
        }
        .help(Self.help)
        .accessibilityElement(children: .ignore)
        // The caption's words, not the table's « ANE »: VoiceOver spelled it out.
        .accessibilityLabel(String(localized: "GPU \(Figures.percent(100 - ane))") + ", " + String(localized: "Neural Engine \(Figures.percent(ane))"))
    }
}
