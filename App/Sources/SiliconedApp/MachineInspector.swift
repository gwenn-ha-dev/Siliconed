// **The « Machine » window**: what the Mac does, in time — a floating panel (`Inspectors.swift`), opened by its button
// in the window's band, beside the Chain inspector's, by Window › Machine (⌥⌘U) or by Core AI's light at the foot of
// the column (`CoreAIMark`):
//
//     [▣] MacBook Pro
//         Apple M1 Pro · 16 GB · macOS 27.2
//         Siliconed holds at most 10 GB of this Mac’s 16 GB.
//     CORE AI                                              compiling
//         compiling                   Anima Turbo · 1024 × 1024 · 12/45
//                                                then Z-Image Turbo
//         Compiled for This Mac                       1.2 GB  [Erase…]
//     THE RENDER IN TIME                    Anima Turbo · 1024 × 1024
//         12.3 s · Denoising · 4.5 GB, weights 3.9 GB · ANE 1.5 + GPU 2.1 TFLOP/s
//         MEMORY  ▁▁▂▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▅▃       ┄┄ Siliconed’s limit
//         COMPUTE, IN TFLOP/S ░░▅▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇░░
//         0       10 s      20 s      30 s
//         Text |          DiT              | VAE
//         [ ] Debug Live
//
// - **The Mac that computes**: this one, or the lender in a mirror — named by its model, not by the name its owner gave
//   it —; the memory Siliconed may hold of it, said as Siliconed's.
// - **Core AI**: what it compiles now, the models after it; what it compiled for this Mac, its weight on the disk, and
//   « Erase… » (`EraseCompiledButton`, as in Settings › Macs). Nothing said when nothing compiles: no news, no line.
// - **The render in time, one axis for both graphs.** The memory: under each instant the base — the process's footprint
//   before the engine's stages opened —, the model's weights they hold (`Engine.MemoryLevel`: the residents, the groups
//   kept, the ring's slots), and above them the intermediates (the stages' tensors and Core AI's working memory, what
//   comes and goes within the calls), under Siliconed's limit. The compute, in TFLOP/s: each DiT cycle at its rate, the
//   Neural Engine's part over the GPU's (`Engine.Cycle.aneFraction`), a mark where the share moved; the text encoder's
//   and the VAE's at their rate over their calls' span (`Engine.Held.work`, the VAE's an image each). A stage that told
//   no rate (a text read from the cache, a Mac lent that does not say it) is shaded.
// - **Debug Live** (`AppState.debugLive`, this Mac's, off at every launch): the next renders record their trace, which
//   `silicontrol trace` reads; under the pointer, a DiT step's units busy over its seconds.
// - **Its refresh is the data's**: the memory comes four times a second while a stage is open, a cycle at its end; the
//   pointer redraws the readout and the line under it, not the graphs (`Equatable` lanes).

import AppKit
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

struct MachineInspector: View {
    let app: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                mac
                // Read every second for a render's compilation burst (`Measures.burst`); paused at rest.
                TimelineView(.animation(minimumInterval: 1, paused: !app.inProgress)) { context in coreAI(now: context.date) }
                PanelSection(title: "The Render in Time", right: m.job?.legend) {
                    if m.memory.buckets.isEmpty {
                        Text(app.inProgress ? "The render begins." : "No renders yet.")
                            .font(.callout).foregroundStyle(Palette.secondary)
                    } else {
                        RenderTimeline(series: m.memory, held: m.held,
                                       cycles: m.held.first { $0.stage == .denoising }?.cycles ?? [])
                    }
                    if app.mirror == nil { debugLive }
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minWidth: 440, minHeight: 420)
    }

    private var m: Measures { app.measurements }

    /// The switch, and what it does said under it.
    private var debugLive: some View {
        VStack(alignment: .leading, spacing: 2) {
            Toggle("Debug Live", isOn: Binding(get: { app.debugLive }, set: { app.debugLive = $0 }))
                .toggleStyle(.checkbox)
            Text("The next renders record where each unit’s time goes, read by silicontrol trace. Off at every launch.")
                .font(.caption).foregroundStyle(Palette.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .padding(.top, 6)
    }

    // ── The Mac ──

    private static let this = Mac.current()

    /// The Mac that computes: this one, or the lender in a mirror (`nil` fields until it said its machine; `lender`, its
    /// name, only in a mirror).
    private var machine: (hardwareModel: String?, chip: String?, memory: Int?, macOS: String?, lender: String?) {
        if let mirror = app.mirror {
            let r = app.mirrorMachine
            return (r?.hardwareModel, r?.chip, r?.memoryBytes, nil, mirror.macName)
        }
        return (Self.this.hardwareModel, Self.this.chip, Self.this.memoryBytes, Self.this.macOS, nil)
    }

    /// « MacBook Pro », as the system names `hw.model`; `nil` for one it does not know.
    private static func modelName(_ hardwareModel: String?) -> String? {
        guard let h = hardwareModel,
              let t = UTType(tag: h, tagClass: UTTagClass(rawValue: "com.apple.device-model-code"), conformingTo: nil),
              !t.isDynamic else { return nil }
        return t.localizedDescription
    }

    private var mac: some View {
        let mac = machine
        return HStack(alignment: .center, spacing: 12) {
            MacIcon(hardwareModel: mac.hardwareModel).frame(width: 44, height: 44)
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: Self.modelName(mac.hardwareModel) ?? mac.chip ?? "Mac").font(.headline)
                Text(verbatim: [mac.chip, mac.memory.map(Figures.memory), mac.macOS.map { "macOS \($0)" }].compactMap { $0 }
                    .joined(separator: " · "))
                    .font(.callout).foregroundStyle(Palette.secondary)
                if let lender = mac.lender {
                    Text("Renders on “\(lender)”").font(.caption).foregroundStyle(Palette.secondary)
                }
                if let memory = mac.memory {
                    Text(mac.lender == nil ? "Siliconed holds at most \(Figures.memory(app.memoryCeiling)) of this Mac’s \(Figures.memory(memory))."
                                           : "Siliconed holds at most \(Figures.memory(app.memoryCeiling)) of that Mac’s \(Figures.memory(memory)).")
                        .font(.caption).foregroundStyle(Palette.secondary)
                        .help("Siliconed’s memory limit, set in Settings › Macs: the most the engine holds while it renders")
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 4)
    }

    // ── Core AI ──

    /// **What Core AI compiles, and what it compiled**: a preparation under way (`AppState.compilingAhead`) — its model,
    /// its size, the specialization it is at of its key's, the models after it —, or the render's own; then what is
    /// compiled for this Mac (not a mirror's: its disk is the other Mac's).
    @ViewBuilder private func coreAI(now: Date) -> some View {
        let rendering = app.inProgress && m.compiling(at: now)
        if app.mirror == nil || app.compilingAhead != nil || rendering {
            PanelSection(title: "Core AI", right: app.compilingCoreAI ? String(localized: "compiling") : nil) {
                if let p = app.compilingAhead {
                    StatLine("compiling", "\(app.modelName(p.model)) · \(Figures.size(p.width, p.height)) · \(p.at)/\(p.of)",
                             detail: p.next.isEmpty ? nil : String(localized: "then \(p.next.map(app.modelName).formatted(.list(type: .and)))"))
                } else if rendering, let job = m.job {
                    StatLine("compiling", job.legend)
                }
                if app.mirror == nil { CompiledLine(app: app) }
            }
        }
    }
}

/// **What Core AI compiled for this Mac**: its weight on the disk, weighed when the window opens and when a compilation
/// or an erasure ends, and « Erase… ».
private struct CompiledLine: View {
    let app: AppState

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text("Compiled for This Mac").foregroundStyle(Palette.secondary)
            Spacer(minLength: 12)
            Text(verbatim: words).monospacedDigit()
            EraseCompiledButton(app: app)
        }
        .font(.callout)
        .task(id: app.compilingCoreAI || app.erasingCompiled) {
            if !app.compilingCoreAI && !app.erasingCompiled { await app.weighCompiled() }
        }
    }

    private var words: String {
        if app.erasingCompiled { return String(localized: "Erasing…") }
        guard let bytes = app.compiledBytes else { return "—" }
        return bytes == 0 ? String(localized: "Nothing compiled yet.") : Figures.disk(bytes)
    }
}

// MARK: - The render in time

/// **The memory and the compute of one render, on one time axis**, the readout of the instant under the pointer above
/// them.
private struct RenderTimeline: View {
    let series: Engine.MemorySeries
    let held: [Engine.Held]
    let cycles: [Engine.Cycle]
    /// The instant under the pointer, in seconds on the render's clock.
    @State private var hover: Double?

    var body: some View {
        let t = Timeline(series: series, held: held, cycles: cycles)
        VStack(alignment: .leading, spacing: 5) {
            Text(verbatim: readout(t, at: hover)).font(.caption).monospacedDigit().lineLimit(2)
                .frame(maxWidth: .infinity, minHeight: 30, alignment: .topLeading)
            title("Memory", right: nil)
            lane(height: 130, t) { MemoryLane(timeline: t).equatable() }
            Legend(items: [(MemoryLane.base, String(localized: "base")),
                           (MemoryLane.weights, String(localized: "model weights")),
                           (MemoryLane.intermediates, String(localized: "intermediates"))],
                   dashed: t.limit > 0 ? String(localized: "Siliconed’s limit, \(Figures.memory(t.limit))") : nil)
            title("Compute, in TFLOP/s", right: t.laid.map(\.total).max().map { String(localized: "max \(Figures.tflops($0)) TFLOP/s") })
                .padding(.top, 6)
            lane(height: 110, t) { ComputeLane(timeline: t).equatable() }
            Legend(items: [(Palette.live, String(localized: "Neural Engine")), (ComputeLane.gpu, "GPU")],
                   mark: t.laid.contains { $0.moved } ? String(localized: "share moved") : nil)
            TimeAxis(timeline: t).equatable().frame(height: 28)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: readout(t, at: nil)))
    }

    private func title(_ title: LocalizedStringKey, right: String?) -> some View {
        HStack {
            Text(title).font(.caption.weight(.semibold)).foregroundStyle(Palette.secondary)
            Spacer()
            if let right { Text(verbatim: right).font(.caption2).monospacedDigit().foregroundStyle(Palette.secondary) }
        }
    }

    /// A graph, the pointer's line over it: hovered, the instant is the readout's.
    private func lane(height: CGFloat, _ t: Timeline, @ViewBuilder _ content: () -> some View) -> some View {
        let graph = content()
        return GeometryReader { g in
            ZStack(alignment: .topLeading) {
                graph
                if let hover {
                    Rectangle().fill(Color.primary.opacity(0.55)).frame(width: 1, height: g.size.height)
                        .offset(x: g.size.width * CGFloat(hover / t.duration))
                }
            }
            .contentShape(Rectangle())
            .onContinuousHover { phase in
                switch phase {
                case .active(let p): hover = min(max(0, Double(p.x / max(g.size.width, 1)) * t.duration), t.duration)
                case .ended: hover = nil
                }
            }
        }
        .frame(height: height)
    }

    /// At `at`: its time, its stage, the footprint and the weights in it, the weight groups kept, the compute; without
    /// it, the whole render.
    private func readout(_ t: Timeline, at: Double?) -> String {
        guard let at else {
            var parts = [Figures.seconds(t.duration)]
            if let peak = t.peak { parts.append(String(localized: "peak \(Figures.memory(peak))")) }
            if let top = t.laid.map(\.total).max(), top > 0 { parts.append(String(localized: "max \(Figures.tflops(top)) TFLOP/s")) }
            return parts.joined(separator: " · ")
        }
        var parts = [Figures.seconds(at)]
        let span = t.span(at: at), stage = span?.stage
        if let stage { parts.append(stage.label) }
        if let f = t.footprint(at: at) {
            let weights = t.level(at: at).flatMap { $0.floor == nil || $0.weights == 0 ? nil : $0.weights }
            parts.append(weights.map { String(localized: "\(Figures.memory(f)), weights \(Figures.memory($0))") } ?? Figures.memory(f))
        }
        if let span, span.groups > 0 {
            parts.append(span.kept >= span.groups ? String(localized: "all \(span.groups) weight groups kept")
                                                   : String(localized: "\(span.kept) of \(span.groups) weight groups kept"))
        }
        if let c = t.cycle(at: at) {
            parts.append(c.ane > 0 || c.cycle ? String(localized: "ANE \(Figures.tflops(c.ane)) + GPU \(Figures.tflops(c.gpu)) TFLOP/s")
                                              : String(localized: "GPU \(Figures.tflops(c.gpu)) TFLOP/s"))
            if c.cycle { parts.append(c.rows > 0 ? String(localized: "\(c.rows) rows on the ANE") : String(localized: "GPU only")) }
            if let busy = c.busy {
                let ane = Figures.percent(Int((busy.ane * 100).rounded())), gpu = Figures.percent(Int((busy.gpu * 100).rounded()))
                parts.append(String(localized: "busy: ANE \(ane), GPU \(gpu)"))
            }
        } else if let stage, stage != .denoising, !t.counted.contains(stage) {
            parts.append(String(localized: "rates not counted"))
        }
        return parts.joined(separator: " · ")
    }
}

/// **What both graphs read**, drawn from the render's events once per change: the stages' spans, what they held in time,
/// the DiT's cycles laid in time, the scales.
private struct Timeline: Equatable {
    struct Span: Equatable {
        let stage: Engine.Stage
        let from, to: Double
        /// The groups of weights of its largest engine stage, and those it kept in memory.
        let groups, kept: Int
        /// The process's footprint before it opened, `nil` where it told none: for a series without levels.
        let floor: Int?
    }

    /// A DiT cycle on the render's clock, its block's rate split between the units; or another stage's calls over
    /// their span (`Engine.Work`: not a cycle, no rows).
    struct Laid: Equatable {
        let from, to: Double
        let ane, gpu: Double
        let rows: Int
        /// Its rows on the Neural Engine are not the cycle before's.
        let moved: Bool
        var cycle = true
        /// Under Debug Live: each unit's busy seconds over the cycle's (`Engine.Occupancy`).
        var busy: Busy? = nil
        var total: Double { ane + gpu }
        struct Busy: Equatable { let ane, gpu: Double }
    }

    let series: Engine.MemorySeries
    let spans: [Span]
    let laid: [Laid]
    /// The stages other than the DiT that told their rates.
    let counted: Set<Engine.Stage>
    let duration: Double
    let limit: Int

    init(series: Engine.MemorySeries, held: [Engine.Held], cycles: [Engine.Cycle]) {
        self.series = series
        // A cycle has its end on the series' clock; one from a Mac that does not say it is laid after the one before,
        // from the DiT's start.
        var laid: [Laid] = [], cursor = series.marks.first { $0.stage == .denoising }?.at ?? 0
        for (n, c) in cycles.enumerated() {
            let to = c.end ?? cursor + c.seconds, from = max(0, to - c.seconds)
            let ane = c.tflops * (c.aneFraction ?? 0)
            var l = Laid(from: from, to: to, ane: ane, gpu: c.tflops - ane, rows: c.rows, moved: n > 0 && cycles[n - 1].rows != c.rows)
            l.busy = c.occupancy.flatMap { o in o.seconds > 0 ? Laid.Busy(ane: o.ane / o.seconds, gpu: o.gpu / o.seconds) : nil }
            laid.append(l)
            cursor = to
        }
        // The other stages' calls, each run at its rate over its span: one that does not say its end is not laid.
        var counted = Set<Engine.Stage>()
        for h in held where h.stage != .denoising {
            for w in h.work {
                guard let to = w.end, w.seconds > 0 else { continue }
                let ane = w.tflops * w.aneFraction
                laid.append(Laid(from: max(0, to - w.seconds), to: to, ane: ane, gpu: w.tflops - ane, rows: 0, moved: false, cycle: false))
                counted.insert(h.stage)
            }
        }
        self.counted = counted
        self.laid = laid.sorted { $0.from < $1.from }
        let duration = max(series.duration, laid.last?.to ?? 0, 0.001)
        self.duration = duration
        let marks = series.marks
        spans = marks.indices.map { i in
            let h = held.first { $0.stage == marks[i].stage }
            return Span(stage: marks[i].stage, from: marks[i].at, to: i + 1 < marks.count ? marks[i + 1].at : duration,
                        groups: h?.groups ?? 0, kept: h?.kept ?? 0, floor: h.flatMap { $0.process > 0 ? $0.process - $0.held : nil })
        }
        limit = held.map(\.budget).max() ?? 0
    }

    var peak: Int? { series.buckets.map(\.high).max() }

    func span(at t: Double) -> Span? { spans.last { $0.from <= t } }

    func cycle(at t: Double) -> Laid? { laid.first { $0.from <= t && t < $0.to } }

    /// What the engine's stages held at `t` (`Engine.MemoryLevel`); from a Mac that does not say it, the stage's floor
    /// and no weights.
    func level(at t: Double) -> Engine.MemoryLevel? {
        if series.levels.isEmpty { return span(at: t).map { Engine.MemoryLevel(at: $0.from, floor: $0.floor, weights: 0) } }
        return series.levels.last { $0.at <= t }
    }

    /// The highest footprint of the bucket at `t`.
    func footprint(at t: Double) -> Int? {
        let i = Int(t / series.step)
        return series.buckets.last { $0.index <= i }?.high
    }

    /// Grid lines a step apart, the step the smallest of `steps` that leaves at most `count` of them under `top`.
    static func grid(_ top: Double, steps: [Double], count: Int) -> [Double] {
        let step = steps.first { top / $0 <= Double(count) } ?? steps.last!
        return stride(from: step, through: top, by: step).map { $0 }
    }
}

/// **The memory lane**: under each pixel column, up to the highest footprint of the 20 ms buckets it covers
/// (`MemorySeries.envelope`), the base grey up to the floor the stages opened on, the model's weights they hold over it,
/// the intermediates above; no stage open, the whole column is the process's. Siliconed's limit dashed across, the
/// stages' starts thin separators, a grid in round gigabytes.
private struct MemoryLane: View, Equatable {
    let timeline: Timeline

    static let base = Palette.secondary.opacity(0.38)
    static let weights = Palette.live
    static let intermediates = Palette.live.opacity(0.35)

    var body: some View {
        Canvas { context, size in draw(context, size) }
    }

    private func draw(_ context: GraphicsContext, _ size: CGSize) {
        let t = timeline, h = size.height
        let top = Double(max(t.limit, t.peak ?? 0, 1)) * 1.08
        let y: (Int) -> CGFloat = { h - h * CGFloat(Double($0) / top) }
        let gib = Double(1 << 30)
        for line in Timeline.grid(top / gib, steps: [0.5, 1, 2, 4, 8, 16, 32, 64], count: 4) {
            let at = y(Int(line * gib))
            context.stroke(Path { $0.move(to: CGPoint(x: 0, y: at)); $0.addLine(to: CGPoint(x: size.width, y: at)) },
                           with: .color(Palette.hairline), lineWidth: 0.5)
            context.draw(Text(verbatim: Figures.memory(Int(line * gib))).font(.system(size: 9)).foregroundStyle(Palette.secondary),
                         at: CGPoint(x: 2, y: at - 1), anchor: .bottomLeading)
        }
        let columns = max(1, Int(size.width))
        let envelope = t.series.envelope(columns: columns, duration: t.duration)
        let wide = size.width / CGFloat(max(envelope.count, 1))
        var base = Path(), weights = Path(), intermediates = Path()
        func band(_ path: inout Path, _ x: CGFloat, _ from: Int, _ to: Int) {
            if to > from { path.addRect(CGRect(x: x, y: y(to), width: wide, height: y(from) - y(to))) }
        }
        for (c, e) in envelope.enumerated() {
            guard let e else { continue }
            let x = CGFloat(c) * wide, at = (Double(c) + 0.5) / Double(columns) * t.duration
            guard let level = t.level(at: at), let floor = level.floor else { band(&base, x, 0, e.high); continue }
            let f = min(floor, e.high), w = min(floor + level.weights, e.high)
            band(&base, x, 0, f)
            band(&weights, x, f, w)
            band(&intermediates, x, w, e.high)
        }
        context.fill(base, with: .color(Self.base))
        context.fill(weights, with: .color(Self.weights))
        context.fill(intermediates, with: .color(Self.intermediates))
        for s in t.spans.dropFirst() {
            let x = size.width * CGFloat(s.from / t.duration)
            context.stroke(Path { $0.move(to: CGPoint(x: x, y: 0)); $0.addLine(to: CGPoint(x: x, y: h)) },
                           with: .color(Palette.secondary.opacity(0.5)), lineWidth: 0.5)
        }
        if t.limit > 0 {
            let at = y(t.limit)
            context.stroke(Path { $0.move(to: CGPoint(x: 0, y: at)); $0.addLine(to: CGPoint(x: size.width, y: at)) },
                           with: .color(Palette.secondary), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
        }
    }
}

/// **The compute lane, in TFLOP/s**: each DiT cycle a column as wide as it lasted, at its block's rate — the GPU's part
/// under, the Neural Engine's over it —, a mark at the top where the share moved; the text encoder's and the VAE's calls
/// each a column over their span; the stages that told no rate shaded.
private struct ComputeLane: View, Equatable {
    let timeline: Timeline

    static let gpu = Palette.secondary.opacity(0.5)

    var body: some View {
        Canvas { context, size in draw(context, size) }
    }

    private func draw(_ context: GraphicsContext, _ size: CGSize) {
        let t = timeline, h = size.height
        let top = max(t.laid.map(\.total).max() ?? 0, 0.5) * 1.15
        let y: (Double) -> CGFloat = { h - h * CGFloat($0 / top) }
        let x: (Double) -> CGFloat = { size.width * CGFloat($0 / t.duration) }
        for s in t.spans where s.stage != .denoising && !t.counted.contains(s.stage) {
            context.fill(Path(CGRect(x: x(s.from), y: 0, width: max(1, x(s.to) - x(s.from)), height: h)), with: .color(Palette.well))
        }
        for line in Timeline.grid(top, steps: [0.5, 1, 2, 5, 10, 20, 50], count: 4) {
            let at = y(line)
            context.stroke(Path { $0.move(to: CGPoint(x: 0, y: at)); $0.addLine(to: CGPoint(x: size.width, y: at)) },
                           with: .color(Palette.hairline), lineWidth: 0.5)
            context.draw(Text(verbatim: "\(Figures.tflops(line)) TFLOP/s").font(.system(size: 9)).foregroundStyle(Palette.secondary),
                         at: CGPoint(x: 2, y: at - 1), anchor: .bottomLeading)
        }
        var gpu = Path(), ane = Path(), marks = Path()
        for c in t.laid {
            // A hair between two cycles, so that each reads as one.
            let left = x(c.from), width = max(1, x(c.to) - left - (x(c.to) - left > 3 ? 1 : 0))
            gpu.addRect(CGRect(x: left, y: y(c.gpu), width: width, height: h - y(c.gpu)))
            if c.ane > 0 { ane.addRect(CGRect(x: left, y: y(c.total), width: width, height: y(c.gpu) - y(c.total))) }
            if c.moved {
                marks.move(to: CGPoint(x: left - 3.5, y: 0)); marks.addLine(to: CGPoint(x: left + 3.5, y: 0))
                marks.addLine(to: CGPoint(x: left, y: 6)); marks.closeSubpath()
            }
        }
        context.fill(gpu, with: .color(Self.gpu))
        context.fill(ane, with: .color(Palette.live))
        context.fill(marks, with: .color(Palette.live))
    }
}

/// **The time under both lanes**: round seconds from the render's start, then each stage's name under its span, centred,
/// past its edges if it must (a VAE of 2 s in a render of 40 s is 24 points wide); a span too narrow even for that keeps
/// its separators only.
private struct TimeAxis: View, Equatable {
    let timeline: Timeline

    var body: some View {
        GeometryReader { g in
            let t = timeline, w = g.size.width
            ZStack(alignment: .topLeading) {
                ForEach(Timeline.grid(t.duration, steps: [1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600], count: 6), id: \.self) { s in
                    Text(verbatim: Figures.duration(s)).font(.system(size: 9)).monospacedDigit().foregroundStyle(Palette.secondary)
                        .fixedSize()
                        .offset(x: w * CGFloat(s / t.duration) - 8)
                }
                ForEach(Array(t.spans.enumerated()), id: \.offset) { _, s in
                    let x = w * CGFloat(s.from / t.duration), width = w * CGFloat((s.to - s.from) / t.duration)
                    if width >= 14 {
                        Text(Self.short(s.stage)).font(.caption2).foregroundStyle(Palette.secondary)
                            .fixedSize().frame(width: width).offset(x: x, y: 14)
                    }
                }
            }
        }
    }

    private static func short(_ stage: Engine.Stage) -> LocalizedStringKey {
        switch stage {
        case .text: "Text"
        case .image: "Image"
        case .denoising: "DiT"
        case .decoding: "VAE"
        }
    }
}

/// A graph's key: a swatch and its word each, a dashed line's word, the share's mark.
private struct Legend: View {
    let items: [(Color, String)]
    var dashed: String? = nil
    var mark: String? = nil

    var body: some View {
        HStack(spacing: 12) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                HStack(spacing: 4) {
                    RoundedRectangle(cornerRadius: 2).fill(item.0).frame(width: 9, height: 9)
                    Text(verbatim: item.1)
                }
            }
            if let dashed {
                HStack(spacing: 4) {
                    Path { $0.move(to: CGPoint(x: 0, y: 4.5)); $0.addLine(to: CGPoint(x: 14, y: 4.5)) }
                        .stroke(Palette.secondary, style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                        .frame(width: 14, height: 9)
                    Text(verbatim: dashed)
                }
            }
            if let mark {
                HStack(spacing: 4) {
                    Image(systemName: "arrowtriangle.down.fill").font(.system(size: 7)).foregroundStyle(Palette.live)
                    Text(verbatim: mark)
                }
            }
            Spacer(minLength: 0)
        }
        .font(.caption2).foregroundStyle(Palette.secondary)
    }
}
