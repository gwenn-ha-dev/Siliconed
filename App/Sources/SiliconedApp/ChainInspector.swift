// **The « Chain » inspector**: a floating panel (`Inspectors.swift`) that says where the render stands, what Core AI
// compiles — ahead of a render (a preparation: its model, its size, how far) or in it, stage by stage —, then the chain
// stage by stage — its module, its weights, its time, the Neural Engine's share where it computed; the DiT step by step,
// each step's TFLOP/s, the block's and each unit's, and where the share moved — and the memory of the render in one
// small graph, in time (the footprint's min and max, the limit), what remains for the queue, and the session.
//
// Its lines keep their number while a render runs: a model's first render compiles in bursts up to its last band of
// decoding, and a line that came and went with each compilation moved everything under it, several times a second.
//
// Everything comes from the engine's events (`Measures`) and from the speeds learned in the session: nothing
// is estimated from a constant, and nothing is written. The memory is the engine's ledger, carried by the events
// (`Engine.Event.held`): never a sample taken here. A wait for a model or a format never rendered has no estimate,
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
                    // nothing waiting, it stops (a panel left open redrew both sections every second for nothing).
                    TimelineView(.animation(minimumInterval: 1, paused: !app.inProgress)) { context in inProgress(now: context.date) }
                    TimelineView(.animation(minimumInterval: 1, paused: !app.inProgress)) { context in coreAI(now: context.date) }
                    stages
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
        Section(title: "Current Render", right: m.job.map(\.legend)) {
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

    // ── Core AI: what it compiles, ahead of a render or in it ──

    /// **What Core AI compiles**: a preparation under way (`AppState.compilingAhead`) — its model, its size, the
    /// specialization it is at of its key's, the models after it —; the render's, under way or the last, stage by stage
    /// (`Engine.Held.compilation`).
    @ViewBuilder private func coreAI(now: Date) -> some View {
        Section(title: "Core AI", right: app.compilingCoreAI ? String(localized: "compiling") : nil) {
            if let p = app.compilingAhead {
                StatLine("preparing", "\(app.modelName(p.model)) · \(Figures.size(p.width, p.height)) · \(p.at)/\(p.of)",
                         detail: p.next.isEmpty ? nil : String(localized: "then \(p.next.map(app.modelName).formatted(.list(type: .and)))"))
            }
            if m.job != nil {
                let compiled = m.held.filter { $0.compilation > 0 }, total = compiled.map(\.compilation).reduce(0, +)
                StatLine(app.inProgress ? "this render" : "last render",
                         app.inProgress && m.compiling(at: now) ? String(localized: "compiling…")
                             : total > 0 ? String(localized: "\(Figures.seconds(total)) compiled") : String(localized: "nothing compiled"),
                         detail: compiled.isEmpty ? nil : compiled.map { "\(name($0.stage)) \(Figures.seconds($0.compilation))" }.joined(separator: " · "))
            } else if app.compilingAhead == nil {
                Text("Nothing compiling.").font(.callout).foregroundStyle(Palette.secondary)
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

    private var stages: some View {
        Section(title: "Chain") {
            ForEach(Array(shownStages.enumerated()), id: \.element) { n, stage in
                if n > 0 { Divider() }
                StageLine(title: title(stage), what: what(stage), time: time(stage),
                          note: stage == .text && m.textCached ? String(localized: "read from the session’s cache") : nil,
                          units: m.held.first { $0.stage == stage }?.neuralEngine.flatMap { $0 > 0 ? $0 : nil },
                          highlighted: focus.stage == stage) {
                    if stage == .denoising { dit }
                }
                .id(stage)
            }
            if !m.memory.buckets.isEmpty {
                Divider()
                MemoryChart(series: m.memory, held: m.held, label: short).padding(.horizontal, 6)
            }
        }
    }

    /// The stage's short name, in a sentence.
    private func name(_ stage: Engine.Stage) -> String {
        switch stage {
        case .text: String(localized: "Text")
        case .image: String(localized: "Image")
        case .denoising: String(localized: "DiT")
        case .decoding: String(localized: "VAE")
        }
    }

    private func short(_ stage: Engine.Stage) -> LocalizedStringKey {
        switch stage {
        case .text: "Text"
        case .image: "Image"
        case .denoising: "DiT"
        case .decoding: "VAE"
        }
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

    /// **The DiT step by step** (`Engine.Cycle`, the image under way's in a batch): each step's TFLOP/s as a bar — those to
    /// come hollow —, the most reached, then a line a step: where the Neural Engine's share stood, the TFLOP/s of the
    /// block and of each unit, the seconds. Every render, not only a model's first: the share is followed at every step.
    @ViewBuilder private var dit: some View {
        let image = min(app.batchImage, max(0, m.stepSeconds.count - 1))
        let cycles = Array(m.cycles(image: image))
        if m.planned > 0 {
            HStack {
                Text(verbatim: "TFLOP/s")
                Spacer()
                if let top = cycles.map(\.tflops).max(), top > 0 { Text("max \(Figures.tflops(top))") }
            }
            .font(.caption2).monospacedDigit().foregroundStyle(Palette.secondary)
            .padding(.top, 2)
            RateBars(rates: cycles.map(\.tflops), planned: m.planned)
                .frame(height: 38)
            if !cycles.isEmpty { CycleTable(cycles: cycles).padding(.top, 4) }
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
        // Its memory and its Neural Engine's share are the chain's, above.
        if let c = m.timings {
            Section(title: "Last Image") {
                StatLine("total", Figures.seconds(c.total))
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
            return Format.parse(String(part)).map { Figures.size($0.width, $0.height) } ?? String(part)
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
                Text(title).textCase(.uppercase).font(.caption.weight(.semibold)).tracking(0.7).foregroundStyle(Palette.secondary)
                Spacer()
                if let right { Text(right).font(.caption).foregroundStyle(Palette.secondary).lineLimit(1) }
            }
            content
        }
        .padding(11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor).opacity(0.8)))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Palette.hairline))
    }
}

/// **A stage of the chain**: its name and its time, then what it is, then what it held — highlighted when the rack asked
/// for it.
private struct StageLine<Extra: View>: View {
    let title: LocalizedStringKey
    let what: String
    let time: String
    let note: String?
    /// The Neural Engine's share of the stage's arithmetic; `nil` when the stage ran no session.
    let units: Double?
    let highlighted: Bool
    @ViewBuilder var extra: Extra

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
            extra
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 6).fill(highlighted ? Palette.live.opacity(0.12) : .clear))
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
                label.foregroundStyle(Palette.secondary)
                Spacer(minLength: 12)
                Text(value).monospacedDigit().multilineTextAlignment(.trailing)
            }
            if let detail {
                Text(detail).font(.caption2).foregroundStyle(Palette.secondary).frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .font(.callout)
    }
}

/// **The DiT's steps, a line each** (`Engine.Cycle`): its number; the Neural Engine's rows, of those it could take
/// (« GPU only » at none), marked where the share moved; the block's TFLOP/s — what its calls executed over its own
/// seconds — and each unit's over its calls (— where they were not timed: the GPU alone, a block's last step); its
/// seconds.
private struct CycleTable: View {
    let cycles: [Engine.Cycle]

    var body: some View {
        Grid(alignment: .trailing, horizontalSpacing: 10, verticalSpacing: 2) {
            GridRow {
                Text("step"); Text("ANE rows"); Text("all"); Text(verbatim: "ANE"); Text(verbatim: "GPU"); Text("time")
            }
            .foregroundStyle(Palette.secondary)
            ForEach(Array(cycles.enumerated()), id: \.offset) { n, c in
                GridRow {
                    Text(verbatim: "\(n + 1)").foregroundStyle(Palette.secondary)
                    HStack(spacing: 3) {
                        if n > 0, cycles[n - 1].rows != c.rows {
                            Image(systemName: "arrowtriangle.right.fill").font(.system(size: 6)).foregroundStyle(Palette.live)
                        }
                        Text(c.rows > 0 ? String(c.rows) : String(localized: "GPU only"))
                    }
                    .help(String(localized: "\(c.rows) of \(c.shareable) rows on the Neural Engine, by \(c.unit)"))
                    Text(Figures.tflops(c.tflops))
                    Text(c.neuralEngine.map(Figures.tflops) ?? "—")
                    Text(c.gpu.map(Figures.tflops) ?? "—")
                    Text(Figures.seconds(c.seconds))
                }
            }
        }
        .font(.caption2).monospacedDigit()
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

/// **One bar per DiT step**: its TFLOP/s for those done (the block's: what its calls executed over its own seconds),
/// hollow for those to come.
private struct RateBars: View {
    let rates: [Double]
    let planned: Int

    var body: some View {
        GeometryReader { g in
            let n = max(planned, rates.count)
            let ceiling = max(rates.max() ?? 0, 0.001)
            let width = (g.size.width - CGFloat(n - 1) * 3) / CGFloat(max(n, 1))
            HStack(alignment: .bottom, spacing: 3) {
                ForEach(0..<n, id: \.self) { i in
                    let done = i < rates.count
                    RoundedRectangle(cornerRadius: 2)
                        .fill(done ? Palette.live : Color.secondary.opacity(0.25))
                        .frame(width: max(1, width), height: done ? max(2, g.size.height * CGFloat(rates[i] / ceiling)) : 2)
                        .help(done ? String(localized: "step \(i + 1): \(Figures.tflops(rates[i])) TFLOP/s")
                                   : String(localized: "step \(i + 1): pending"))
                }
            }
            .frame(maxHeight: .infinity, alignment: .bottom)
        }
    }
}

/// **The memory of the chain in time**: the process's footprint — what the limit bounds, what Activity Monitor shows —
/// over the render, the seconds from its start on the x axis. Each pixel column draws the lowest and highest footprint
/// of the 20 ms buckets it covers (`MemorySeries.envelope`), so a ring slot filled then let go shows as a tooth at any
/// width, and a flat stretch as a line; the limit is a dashed line across, the stages' starts thin separators. A stage's
/// span says, hovered, its base, its peak under the limit and the groups of weights it kept.
struct MemoryChart: View {
    let series: Engine.MemorySeries
    let held: [Engine.Held]
    let label: (Engine.Stage) -> LocalizedStringKey

    /// The stages' spans, each from its start to the next's (the last to the series' end).
    private struct Span: Identifiable {
        let id: Int
        let stage: Engine.Stage
        let from, to: Double
    }

    private var duration: Double { max(series.duration, 0.001) }
    private var spans: [Span] {
        let marks = series.marks
        return marks.indices.map { i in Span(id: i, stage: marks[i].stage, from: marks[i].at, to: i + 1 < marks.count ? marks[i + 1].at : duration) }
    }
    private var limit: Int { held.map(\.budget).max() ?? 0 }
    private var ceiling: Double { Double(max(limit, series.buckets.map(\.high).max() ?? 0, 1)) * 1.08 }

    /// What the column used to say, for the stage's span.
    static func help(_ h: Engine.Held?) -> String {
        guard let h, h.process > 0 else { return "" }
        let base = max(0, h.process - h.held) + h.counted
        var lines = [String(localized: "base \(Figures.memory(base)) · peak \(Figures.memory(h.process)) under the limit of \(Figures.memory(h.budget))")]
        if h.groups > 0 {
            lines.append(h.kept >= h.groups ? String(localized: "all \(h.groups) weight groups kept in memory")
                                            : String(localized: "\(h.kept) of \(h.groups) weight groups kept in memory, the others read from the disk"))
        }
        return lines.joined(separator: "\n")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                Text(Figures.seconds(duration)).monospacedDigit()
                Spacer(minLength: 6)
                if limit > 0 { Text("limit \(Figures.memory(limit))").monospacedDigit() }
            }
            .font(.caption2.weight(.medium)).foregroundStyle(Palette.secondary)
            GeometryReader { g in
                let w = g.size.width, h = g.size.height
                ZStack(alignment: .topLeading) {
                    Canvas { context, size in draw(context, size) }
                    ForEach(spans) { s in
                        let x = w * CGFloat(s.from / duration), width = max(1, w * CGFloat((s.to - s.from) / duration))
                        Color.clear.contentShape(Rectangle()).frame(width: width, height: h).offset(x: x)
                            .help(Self.help(held.first { $0.stage == s.stage }))
                    }
                }
            }
            .frame(height: 72)
            GeometryReader { g in
                ForEach(spans) { s in
                    let x = g.size.width * CGFloat(s.from / duration), width = g.size.width * CGFloat((s.to - s.from) / duration)
                    // A span too narrow for its word keeps its separator and its help only.
                    if width >= 26 { Text(label(s.stage)).frame(width: width).offset(x: x) }
                }
            }
            .frame(height: 12)
            .font(.caption2).foregroundStyle(Palette.secondary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: spans.map { span in Self.help(held.first { $0.stage == span.stage }) }.filter { !$0.isEmpty }.joined(separator: "\n")))
    }

    private func draw(_ context: GraphicsContext, _ size: CGSize) {
        let top = ceiling, h = size.height
        let y: (Int) -> CGFloat = { h - h * CGFloat(Double($0) / top) }
        let envelope = series.envelope(columns: max(1, Int(size.width)), duration: duration)
        let wide = size.width / CGFloat(max(envelope.count, 1))
        // Under the lowest reading, the base; between the lowest and the highest, the teeth.
        var base = Path(), teeth = Path()
        for (c, e) in envelope.enumerated() {
            guard let e else { continue }
            let x = CGFloat(c) * wide
            base.addRect(CGRect(x: x, y: y(e.low), width: wide, height: h - y(e.low)))
            teeth.addRect(CGRect(x: x, y: y(e.high), width: wide, height: max(1.5, y(e.low) - y(e.high))))
        }
        context.fill(base, with: .color(Palette.secondary.opacity(0.38)))
        context.fill(teeth, with: .color(Palette.live))
        for s in spans.dropFirst() {
            let x = size.width * CGFloat(s.from / duration)
            context.stroke(Path { p in p.move(to: CGPoint(x: x, y: 0)); p.addLine(to: CGPoint(x: x, y: h)) },
                           with: .color(Palette.secondary.opacity(0.5)), lineWidth: 0.5)
        }
        if limit > 0 {
            context.stroke(Path { p in p.move(to: CGPoint(x: 0, y: y(limit))); p.addLine(to: CGPoint(x: size.width, y: y(limit))) },
                           with: .color(Palette.secondary), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
        }
    }
}
