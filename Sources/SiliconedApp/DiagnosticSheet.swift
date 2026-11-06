// **« Measure This Mac »: what each engine of this Mac computes, alone and together — the model the engine plans a render
// from** (`Diagnostic.swift`). Sharing the report is an option behind « Share… », never advised nor the default button: the
// sheet is for the tuning, not for the issue (the user, 19/10 and 31/10/2026).
//
// Three states in one sheet: what the diagnostic does and how long (about a minute, five at most, no model needed; it occupies
// the GPU and the CPU, so it waits for an empty queue — and the queue waits for it); its progress; then
// the report as a person reads it — the map first (always: a finished measure is a map, applied as it stands), then how sure
// each engine's figure is and what the measure found, said plainly, with no grade; then per engine what it reaches against the
// peak the silicon announces (a bar where the inventory read one: the GPU's, and only for its matrix products), the column
// « Exact » (✓/✗: computed right against the exact result or not), what the engines keep of their speed when they work together, the
// memory and the duration, and the last render's gap to the model's prediction, which goes with the report. Every column is titled:
// the sheet carries four signs (the card, the bars, the ✓, the ±) and none is left to a tooltip. « Share… ▸ Open a GitHub
// Issue » copies the report and opens GitHub's form
// (`report.issueURL()`), where the user pastes it; « Copy JSON » alone serves whoever has no GitHub account. Nothing is
// sent without a click, and the issue is submitted in the browser, by the user.
//
// **The colour is where it happens**: the map's card is green, the bars are the accent, the rest is the
// system's. Nothing here is orange: an uncertainty is a number beside a figure, a finding a sentence. No chip is named in a decision here — the chip's name is a label, read from the report.

import AppKit
import Siliconed
import SwiftUI

struct DiagnosticSheet: View {
    @Bindable var app: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var notice: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Measure This Mac").font(.title2.weight(.semibold))
                Text("Siliconed measures what each engine of this Mac computes, alone and together, then plans every calculation of a render from it.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let name = app.mirror?.macName {
                    Text("This measures “\(name)”, the Mac that computes for you.")
                        .font(.callout.weight(.medium))
                }
            }
            .padding(20)
            Divider()
            Group {
                switch app.diagnostic {
                case .ready: ready
                case let .running(done, total, line): running(done: done, total: total, line: line)
                case .finished(let report): ReportView(report: report, status: Self.mapStatus(app), render: app.reportNote(for: report))
                case .failed(let problem): ProblemView(problem: problem).padding(20)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            Divider()
            footer.padding(16).background(.bar)
        }
        .frame(width: 620, height: 680)
    }

    // ── before ──

    private var ready: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Matrix products and attention on the GPU, the CPU’s matrix unit and the Neural Engine, alone and together, each result checked against the exact one: what Siliconed plans a render with. No model is needed, nothing is downloaded. About a minute, five at most; the queue waits meanwhile.")
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
            mapStatus
            if let reason = app.diagnosticBlocker {
                Label(reason, systemImage: "hourglass").foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(20)
    }

    /// **Where this session stands against the model**: one in use; a newer one waiting for the end of the render under way;
    /// none yet, or a stale one — the first render measures.
    @ViewBuilder private var mapStatus: some View {
        Text(Self.mapStatus(app)).font(.callout).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// The sentence of `mapStatus` — the settings window says it too.
    static func mapStatus(_ app: AppState) -> LocalizedStringKey {
        let map = app.mapFacts
        if let used = map.inUse {
            return used == map.stored
                ? "This session plans with this Mac’s measure of \(day(used.date))."
                : "This session plans with the measure of \(day(used.date)); the newer one applies to the next render."
        }
        if let stored = map.stored, !stored.current {
            return "The measure of \(day(stored.date)) was taken with another macOS or another version of Siliconed: this Mac is measured again before the next render."
        }
        return "This Mac is measured before its first render: Siliconed plans its calculations from it."
    }

    /// A report's ISO date as a day the user reads.
    static func day(_ iso: String) -> String {
        guard let date = ISO8601DateFormatter().date(from: iso) else { return iso }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    // ── during ──

    private func running(done: Int, total: Int, line: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            // The bar moves when a stage ends, and nothing else moves: no wheel turning beside the timings.
            ProgressView(value: Double(done), total: Double(max(total, 1)))
            HStack(spacing: 8) {
                Text(line).font(.callout)
                Spacer()
                Text(verbatim: "\(done + 1)/\(total)").font(.callout).monospacedDigit().foregroundStyle(.secondary)
            }
        }
        .padding(20)
    }

    // ── the buttons ──

    @ViewBuilder private var footer: some View {
        HStack(spacing: 10) {
            if let notice {
                Text(notice).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if !app.diagnosticRunning {
                // Said before the click that starts it (the user, 4/11/2026: a map measured on a busy Mac, with no word of it before).
                Label("Leave the Mac at rest during the measure: no render, export, compilation or game.", systemImage: "moon.zzz")
                    .font(.callout.weight(.medium))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            switch app.diagnostic {
            case .ready:
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Start") { app.startDiagnostic() }
                    .keyboardShortcut(.defaultAction).tint(Palette.control)
                    .disabled(app.diagnosticBlocker != nil)
            case .running:
                Button("Stop") { app.cancelDiagnostic() }.keyboardShortcut(.cancelAction)
            case .failed:
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Measure Again") { notice = nil; app.startDiagnostic() }
                    .keyboardShortcut(.defaultAction).tint(Palette.control)
                    .disabled(app.diagnosticBlocker != nil)
            case .finished(let report):
                // Sharing the report is an option the user goes to, never the default: « Close » is.
                Menu("Share…") {
                    Button("Open a GitHub Issue") { notice = app.openIssue(report) }
                    Button("Copy JSON") { app.copyJSON(report) }
                }
                .fixedSize()
                .help("Copies the report and opens GitHub’s form; you paste it and submit it there")
                Button("Measure Again") { notice = nil; app.startDiagnostic() }
                    .disabled(app.diagnosticBlocker != nil)
                Button("Close") { dismiss() }.keyboardShortcut(.defaultAction).tint(Palette.control)
            }
        }
    }
}

/// **The report, as a person reads it** — the map, how sure each engine's figure is, what the measure found, then one line per
/// kernel: what it reaches (TFLOP/s, and a bar against the peak the inventory read where there is one) and ✓ when it computed right,
/// ✗ when the judge left it out; the memory the measure held, its duration, the last render's gap; the machine. The JSON says the rest.
private struct ReportView: View {
    let report: Diagnostic.Report
    /// Where this session stands against the model (`DiagnosticSheet.mapStatus`).
    let status: LocalizedStringKey
    /// The last render under this very report, if there was one (`AppState.reportNote(for:)`): it goes with the report.
    var render: RenderLoop.Note?

    /// **The routes in the order the sheet reads them**: the products on the weights widened to fp32 or fp16, then on the weights as
    /// published, then the attentions. A route the model measured and this list does not know follows, by its engine and form.
    private static let order: [CapacityModel.Kernel] = [
        .init(.gpu, .fp32), .init(.amx, .fp32), .init(.ane, .fp16),
        .init(.gpu, .bf16), .init(.gpu, .bf16act), .init(.gpu, .int8), .init(.gpu, .int8act), .init(.gpu, .int8x8), .init(.ane, .int8), .init(.gpu, .q8), .init(.gpu, .q8act),
        .init(.gpu, .graph), .init(.gpu, .flash), .init(.gpu, .tiles), .init(.gpu, .tiles16), .init(.gpu, .tilesbf16), .init(.cpu, .heads)]

    /// The label of a route, as a person reads it.
    static func label(for kernel: CapacityModel.Kernel) -> String {
        switch (kernel.engine, kernel.form) {
        case (.gpu, .fp32): String(localized: "Matrix products, GPU")
        case (.amx, .fp32): String(localized: "Matrix products, CPU matrix unit")
        case (.ane, .fp16): String(localized: "Matrix products, Neural Engine")
        case (.gpu, .bf16): String(localized: "Matrix products, GPU tensor units on the bf16 weights")
        case (.gpu, .bf16act): String(localized: "Matrix products, GPU tensor units, bf16 weights and activations")
        case (.gpu, .int8): String(localized: "Matrix products on the 8-bit weights, read as published")
        case (.ane, .int8): String(localized: "Matrix products on the 8-bit weights, Neural Engine")
        case (.gpu, .int8act): String(localized: "Matrix products on the 8-bit weights, GPU tensor units, bf16 activations")
        case (.gpu, .int8x8): String(localized: "Matrix products on the 8-bit weights, GPU tensor units, 8-bit activations")
        case (.gpu, .q8): String(localized: "Matrix products on the GGUF 8-bit weights, read as published")
        case (.gpu, .q8act): String(localized: "Matrix products on the GGUF 8-bit weights, GPU tensor units, bf16 activations")
        case (.gpu, .graph): String(localized: "Attention, system kernel")
        case (.gpu, .flash): String(localized: "Attention, Siliconed kernel")
        case (.gpu, .tiles): String(localized: "Attention, three stages by tiles")
        case (.gpu, .tiles16): String(localized: "Attention, three stages by tiles, half-precision operands")
        case (.gpu, .tilesbf16): String(localized: "Attention, GPU tensor units, bf16 operands")
        case (.cpu, .heads): String(localized: "Attention, split between GPU and CPU")
        default: engineName(kernel.engine) + " · " + kernel.form.rawValue
        }
    }

    /// An attention kernel: the peak of the matrix products says nothing of it.
    private static func isAttention(_ kernel: CapacityModel.Kernel) -> Bool { [.graph, .flash, .heads, .tiles, .tiles16, .tilesbf16].contains(kernel.form) }

    /// The kernels to list: every route the model has a figure for, then those the judge left out without one.
    static func kernels(of report: Diagnostic.Report) -> [CapacityModel.Kernel] {
        var all = Set(report.reached.map(\.kernel))
        for key in report.judge.keys { if let kernel = CapacityModel.Kernel(key) { all.insert(kernel) } }
        let known = order.filter(all.contains)
        let rest = all.subtracting(known).sorted { ($0.engine, $0.form.rawValue) < ($1.engine, $1.form.rawValue) }
        return known + rest
    }

    var body: some View {
        // The map first: the sheet opens on it, whatever the list below it holds.
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    mapCard.id("map")
                    assurance
                    found
                    reached
                    together
                    facts
                    machine
                }
                .padding(20)
            }
            .onAppear { proxy.scrollTo("map", anchor: .top) }
        }
    }

    // ── the map ──

    /// **The map, always**: the card carries the colour — it is where the answer is. A finished measure is applied as it stands; what it
    /// found, and how sure each figure is, come below it as plain facts.
    private var mapCard: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "checkmark.circle.fill").font(.title2).foregroundStyle(.green)
            VStack(alignment: .leading, spacing: 8) {
                Text("This Mac is measured.").font(.headline)
                Text(status).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Text("A measure applies to the next render, never to one under way. From then on, a seed can give an image a little different from before it.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .background(Color.green.opacity(0.10), in: boxShape)
        .overlay(boxShape.strokeBorder(Color.green.opacity(0.35), lineWidth: 1))
    }

    // ── how sure it is ──

    /// **How sure each figure is**, as the measure saw it: the plus-or-minus of the engines and the sets (half the spread of the identical
    /// calls of their chains, the worst of them), and the cores the other programs used. A free Mac reads ±1–3 %; where the calls differed
    /// more than that the figure is read on its quick side — a disturbance only adds time —, and the line says so.
    @ViewBuilder private var assurance: some View {
        let rows = Self.margins(report)
        VStack(alignment: .leading, spacing: 6) {
            Text("How sure the measure is").font(.headline)
            Text("± is how far apart identical calls were during the measure, for each engine and each set: the margin of its figures.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                ForEach(rows, id: \.name) { row in
                    GridRow {
                        Text(verbatim: row.name).foregroundStyle(.secondary)
                        Text(verbatim: "± " + Figures.percent(Int((100 * row.plusMinus).rounded()))).monospacedDigit()
                    }
                }
            }
            .font(.callout)
            if rows.contains(where: { $0.plusMinus > Diagnostic.Regime.Rules.spread / 2 }) {
                Text("Identical calls differed by more than a free Mac’s: those figures are read on their quick side.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            } else {
                Text("Identical calls agreed to within these margins.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if let cores = report.foreignCores {
                Text("Other programs used \(cores.formatted(.number.precision(.fractionLength(1)))) cores while it measured.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .textSelection(.enabled)
    }

    /// The margin of each engine and each set, by the words the sheet uses: the worst plus-or-minus of the chains that name them (the
    /// integers' pair with the fp32 one). Sorted as the engines are, alone first.
    static func margins(_ report: Diagnostic.Report) -> [(name: String, plusMinus: Double)] {
        var worst: [String: (order: Int, plusMinus: Double)] = [:]
        for u in report.uncertainty {
            let engines = u.subject.split(separator: "+").compactMap { part -> CapacityModel.Engine? in
                CapacityModel.Kernel(String(part))?.engine ?? CapacityModel.Engine(rawValue: String(part))
            }
            guard !engines.isEmpty else { continue }
            let name = engines.contains(.cpu) ? String(localized: "Attention, split between GPU and CPU")
                                              : engines.map(engineName).joined(separator: " + ")
            let order = engines.count * 10 + (engines.map { [CapacityModel.Engine.gpu, .amx, .ane, .cpu].firstIndex(of: $0) ?? 0 }.min() ?? 0)
            let current = worst[name]
            worst[name] = (order, max(current?.plusMinus ?? 0, u.plusMinus))
        }
        return worst.sorted { ($0.value.order, $0.key) < ($1.value.order, $1.key) }.map { (name: $0.key, plusMinus: $0.value.plusMinus) }
    }

    // ── what it found ──

    /// **What the measure found**, one sentence each and what the plan does with it: not a grade, not an alarm. Nothing when it found nothing.
    @ViewBuilder private var found: some View {
        if !report.findings.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text("What the measure found").font(.headline)
                Grid(alignment: .topLeading, horizontalSpacing: 12, verticalSpacing: 4) {
                    ForEach(Array(report.findings.enumerated()), id: \.offset) { _, finding in
                        let words = Self.words(finding, in: report)
                        GridRow {
                            Text(words.subject).fontWeight(.medium).frame(maxWidth: 230, alignment: .leading)
                            Text(words.sentence).fixedSize(horizontal: false, vertical: true)
                                .help(Text(verbatim: finding.text))
                        }
                    }
                }
                .font(.callout)
            }
            .textSelection(.enabled)
        }
    }

    /// **A finding in words**: what it concerns (an engine, a kernel, two or three engines together, the memory, the minute) and what the
    /// plan does with it, one sentence that stands alone — no engine's name inside it, so that no language has to agree with it.
    static func words(_ finding: Diagnostic.Finding, in report: Diagnostic.Report) -> (subject: String, sentence: String) {
        let cause = finding.cause(excluding: report.excluded)
        let subject: String
        switch cause {
        case .noFigure, .computedWrong, .notShared, .implausible: subject = name(of: finding)
        case .disturbed: subject = String(localized: "Other programs")
        case .memoryOverNeed, .memoryOverBudget, .swapGrew: subject = String(localized: "Memory")
        case .tookTooLong: subject = String(localized: "Duration")
        }
        let sentence = switch cause {
        case .noFigure: String(localized: "No figure was measured: the plan does not offer it.")
        case .computedWrong: String(localized: "It computed wrong: the plan has no model of it, and the engine’s defaults answer.")
        case .notShared: String(localized: "They were not measured together: the plan does not offer them as a set.")
        case .implausible: String(localized: "A figure that looks wrong, probably because the Mac was busy: the plan bounds it or leaves it out.")
        case .memoryOverNeed: String(localized: "The measure used more memory than had been counted for it.")
        case .memoryOverBudget: String(localized: "The measure needs more memory than this Mac allows.")
        case .swapGrew: String(localized: "The Mac swapped during the measure: other apps held its memory.")
        case .tookTooLong: String(localized: "The measure took longer than five minutes.")
        case .disturbed: String(localized: "The Mac was busy while it was measured: the figures that differed are read on their quick side.")
        }
        return (subject, sentence)
    }

    /// The engine's name for a product kernel (the form it reads is in the tooltip), the row's label for an attention kernel — an
    /// attention and a product of one engine are two things to the reader —; two or three engines together, joined by « + ».
    private static func name(of finding: Diagnostic.Finding) -> String {
        if let kernel = finding.kernel {
            if isAttention(kernel) { return label(for: kernel) }
            return engineName(kernel.engine)
        }
        if finding.engines.contains(.cpu) { return String(localized: "Attention, split between GPU and CPU") }
        return finding.engines.map(engineName).joined(separator: " + ")
    }

    static func engineName(_ engine: CapacityModel.Engine) -> String {
        switch engine {
        case .gpu: String(localized: "GPU")
        case .amx: String(localized: "CPU matrix unit")
        case .ane: String(localized: "Neural Engine")
        case .cpu: String(localized: "CPU")
        }
    }

    // ── what each engine reaches ──

    /// **What each engine reaches, alone**, one line per route the measure has a figure for. Four columns, each titled: the speed
    /// (TFLOP/s: millions of millions of operations a second); the share of the GPU's peak, the only peak the Mac announces —
    /// its cores and its clock — and only for its matrix products, because a percentage of a product's peak misleads on an attention;
    /// and whether the route computed right (✓), within its threshold of the exact result, or not (✗).
    @ViewBuilder private var reached: some View {
        let peak = report.model.crest[.gpu]
        VStack(alignment: .leading, spacing: 6) {
            Text("What each engine reaches").font(.headline)
            Text("Each route alone, in trillions of operations a second (TFLOP/s). The bar is the share of the GPU’s peak, which its cores and clock announce; only the GPU has one, and only its matrix products are compared to it. ✓: the result is within its threshold of the exact one.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                GridRow {
                    Color.clear.frame(height: 1)
                    Text("Speed").gridColumnAlignment(.trailing)
                    Group {
                        if let peak { Text("Share of the GPU’s peak (\(Self.rate(peak)))") } else { Text(verbatim: "") }
                    }
                    Text("Exact").gridColumnAlignment(.center)
                }
                .font(.caption.weight(.medium)).foregroundStyle(.secondary)
                ForEach(Self.kernels(of: report), id: \.description) { kernel in
                    let line = report.reached.first { $0.kernel == kernel }
                    let excluded = report.judge[kernel.description]?.excluded == true
                    GridRow {
                        Text(verbatim: Self.label(for: kernel)).foregroundStyle(.secondary)
                            .frame(maxWidth: 250, alignment: .leading).fixedSize(horizontal: false, vertical: true)
                        if let line, !excluded {
                            Text(verbatim: Self.rate(line.rate)).monospacedDigit()
                            share(line, attention: Self.isAttention(kernel))
                            verdict(true)
                        } else {
                            Text(verbatim: "—").foregroundStyle(.secondary)
                            Color.clear.frame(width: Self.barWidth, height: 1)
                            verdict(false, died: report.judge[kernel.description]?.reason == .died)
                        }
                    }
                }
            }
            .font(.callout)
            .textSelection(.enabled)
        }
    }

    // ── the engines together ──

    /// What a set of engines keeps: per engine, its share of its own speed alone while the set works on one product, and what the set
    /// reaches against the sum of what its engines reach alone. Nothing where no set was measured.
    struct Together { var name: String; var keeps: [(engine: CapacityModel.Engine, keep: Double)]; var ofSum: Double? }

    /// The route an engine runs alone, for the sum: the kernel a set's key names, else the engine's product on the widened weight.
    private static func alone(_ engine: CapacityModel.Engine, named kernel: CapacityModel.Kernel?) -> CapacityModel.Kernel {
        if let kernel { return kernel }
        switch engine {
        case .gpu: return .init(.gpu, .fp32)
        case .amx: return .init(.amx, .fp32)
        case .ane: return .init(.ane, .fp16)
        case .cpu: return .init(.cpu, .heads)
        }
    }

    static func together(_ report: Diagnostic.Report) -> [Together] {
        let engines: [CapacityModel.Engine] = [.gpu, .amx, .ane, .cpu]
        return report.model.sharing.compactMap { key, sharing in
            let parts = key.split(separator: "+").map(String.init)
            let kernels = parts.map { CapacityModel.Kernel($0) }
            let members = parts.compactMap { part in CapacityModel.Kernel(part)?.engine ?? CapacityModel.Engine(rawValue: part) }
            guard members.count >= 2 else { return nil }
            // The CPU shares only the attention (whole kv groups beside the GPU's graph): its set is that attention, and the GPU's
            // speed alone in it is the graph's, not its matrix products'.
            let attention = members.contains(.cpu)
            var keeps: [(engine: CapacityModel.Engine, keep: Double)] = []
            var sum = 0.0, joint = 0.0
            for (member, kernel) in zip(members, kernels) {
                guard let keep = sharing.keep[member] else { continue }
                keeps.append((member, keep))
                let solo = attention && member == .gpu ? CapacityModel.Kernel(.gpu, .graph) : alone(member, named: kernel)
                if let rate = report.model.lines[solo]?.rate {
                    sum += rate; joint += rate * keep
                }
            }
            guard !keeps.isEmpty else { return nil }
            var name = attention ? String(localized: "Attention, split between GPU and CPU") : members.map(engineName).joined(separator: " + ")
            if kernels.contains(where: { $0?.form == .int8 }) { name += ", " + String(localized: "8-bit weights") }
            // The GPU's rows on the activation rounded to bf16 (the Standard's pair, the integers' beside the rounded kernel): not the fp32 pair.
            if kernels.contains(where: { [.bf16act, .int8act, .q8act].contains($0?.form) }) { name += ", " + String(localized: "bf16 activations") }
            return Together(name: name, keeps: keeps.sorted { engines.firstIndex(of: $0.engine)! < engines.firstIndex(of: $1.engine)! },
                            ofSum: sum > 0 ? joint / sum : nil)
        }
        .sorted { ($0.keeps.count, $0.name) < ($1.keeps.count, $1.name) }
    }

    /// **The engines together**: what each keeps of its speed when they work on the same product, then what the set reaches against
    /// the sum of its engines alone. The plan splits every product by these figures; none is a verdict.
    @ViewBuilder private var together: some View {
        let sets = Self.together(report)
        if !sets.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text("The engines together").font(.headline)
                Text("When engines share a product, each keeps only part of the speed it has alone: they compete for memory and for power. Then the set reaches less than the sum of its engines alone, and more than its best one.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                Grid(alignment: .topLeading, horizontalSpacing: 14, verticalSpacing: 6) {
                    GridRow {
                        Text("Engines")
                        Text("Each keeps of its speed alone")
                        Text("The set reaches")
                    }
                    .font(.caption.weight(.medium)).foregroundStyle(.secondary)
                    ForEach(sets, id: \.name) { set in
                        GridRow {
                            Text(verbatim: set.name).foregroundStyle(.secondary)
                                .frame(maxWidth: 190, alignment: .leading).fixedSize(horizontal: false, vertical: true)
                            Text(verbatim: set.keeps.map { Self.engineName($0.engine) + " " + Figures.percent(Int((100 * $0.keep).rounded())) }
                                .joined(separator: " · "))
                                .fixedSize(horizontal: false, vertical: true)
                            if let share = set.ofSum {
                                Text("\(Figures.percent(Int((100 * share).rounded()))) of the sum of its engines alone").monospacedDigit()
                            } else {
                                Text(verbatim: "—").foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .font(.callout)
                .textSelection(.enabled)
            }
        }
    }

    private static let barWidth: CGFloat = 80

    /// The share of the peak, as a bar and a figure — the GPU's matrix products only (capped at the bar's end: a tensor form may run above the
    /// fp32 peak); nothing for an attention or an engine without a peak, the column's room stays so that the figures line up.
    @ViewBuilder private func share(_ line: Diagnostic.Report.Reach, attention: Bool) -> some View {
        if !attention, let fraction = line.fraction, let crest = line.crest {
            HStack(spacing: 8) {
                Capsule().fill(Palette.well)
                    .overlay(alignment: .leading) {
                        Capsule().fill(Palette.live).frame(width: Self.barWidth * min(max(fraction, 0.02), 1))
                    }
                    .frame(width: Self.barWidth, height: 6)
                Text(verbatim: Figures.percent(Int((100 * fraction).rounded()))).monospacedDigit()
            }
            .help("Peak read from the silicon: \(Self.rate(crest))")
        } else {
            Color.clear.frame(width: Self.barWidth, height: 1)
        }
    }

    static func rate(_ flops: Double) -> String {
        (flops * 1e-12).formatted(.number.precision(.fractionLength(2))) + " TFLOP/s"
    }

    /// ✓ when the route computed right; ✗ when it computed wrong, or — `died` — when the process that loaded it was gone (it is left out
    /// of the plan all the same, and the measure was taken again without it).
    @ViewBuilder private func verdict(_ pass: Bool, died: Bool = false) -> some View {
        if pass {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).help("Computed right: within its threshold of the exact result")
        } else if died {
            Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
                .help("The process that loaded this route died: it is left out of the plan, and no render loads it.")
        } else {
            Image(systemName: "xmark.circle.fill").foregroundStyle(.red).help("Computed wrong: beyond its threshold of the exact result")
        }
    }

    // ── what the measure cost, and what the last render says of it ──

    private var facts: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("The measure used up to \(Figures.memory(report.memory.peak)) of the \(Figures.memory(report.memory.budget)) of memory this Mac allows.")
            Text("\(Figures.duration(report.seconds)) in all")
            if let render, let predicted = render.predicted {
                Text("The last render’s matrix products and attention took \(Figures.seconds(render.measured)) where \(Figures.seconds(predicted)) was predicted; this goes with the report.")
            }
        }
        .font(.caption).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }

    private var machine: some View {
        let m = report.machine
        return Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
            GridRow {
                Text("Chip").foregroundStyle(.secondary)
                Text(verbatim: "\(m.chip) · \(m.hardwareModel)")
            }
            GridRow {
                Text("Cores").foregroundStyle(.secondary)
                Text("\(m.performanceCores) performance, \(m.efficiencyCores) efficiency, \(m.gpuCores) GPU")
            }
            GridRow {
                Text("Memory").foregroundStyle(.secondary)
                Text(verbatim: m.memoryLabel)
            }
            GridRow {
                Text("System").foregroundStyle(.secondary)
                Text(verbatim: "macOS \(m.macOS) · Siliconed \(m.version)")
            }
        }
        .font(.callout)
        .textSelection(.enabled)
    }
}
