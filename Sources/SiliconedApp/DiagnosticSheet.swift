// **« Measure This Mac »: what each engine of this Mac computes, alone and together — the model the engine plans a render
// from —, then sent as a GitHub issue only if the user wants** (`Diagnostic.swift`). The sheet says the tuning first:
// sharing is an option, not its purpose (the user, 19/10/2026).
//
// Three states in one sheet: what the diagnostic does and how long (under a minute, no model needed; it occupies
// the GPU and the CPU, so it waits for an empty queue — and the queue waits for it); its progress; then
// the report as a person reads it — the map first (always: a finished measure is a map, applied as it stands), then how sure
// each engine's figure is and what the measure found, said plainly, with no grade; then per engine what it reaches against the
// peak the silicon announces (a bar where the inventory read one) and ✓/✗ (computed right against the exact or not), the memory
// and the duration, and the last render's gap to the model's prediction, which goes with the report. « Open the issue » copies the report and opens GitHub's form
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
                Text("Siliconed measures what each engine of this Mac computes, alone and together, then plans every calculation of a render from it. Sharing the result is optional: a prefilled GitHub issue, opened only by your click.")
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
            Text("Matrix products and attention on the GPU, the CPU's matrix unit and the Neural Engine, alone and together, each result checked against the exact one: what Siliconed plans a render with. No model is needed, nothing is downloaded. Under a minute; the queue waits meanwhile.")
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

    /// **Where this session stands against the model**: one in use; a newer one waiting for the next
    /// launch; none yet, or a stale one — the first render measures.
    @ViewBuilder private var mapStatus: some View {
        Text(Self.mapStatus(app)).font(.callout).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// The sentence of `mapStatus` — the settings window says it too.
    static func mapStatus(_ app: AppState) -> LocalizedStringKey {
        let map = app.mapFacts
        if let used = map.inUse {
            if let drift = map.drift, let predicted = drift.predicted {
                return driftSentence(measured: drift.measured, predicted: predicted)
            }
            return used == map.stored
                ? "This session plans with this Mac's measure of \(day(used.date))."
                : "This session plans with the measure of \(day(used.date)); the newer one applies from the next launch."
        }
        if let stored = map.stored, !stored.current {
            return "The measure of \(day(stored.date)) was taken with another macOS or another version of Siliconed: this Mac is measured again before the next render."
        }
        return "This Mac is measured before its first render: Siliconed plans its calculations from it."
    }

    /// **A render ran off this Mac's measure** (the footer says it, the chip's tooltip and the sheet too): what it took, what the measure
    /// predicted, and what to do — measure again, which applies from the next launch: the plan never changes between two renders.
    static func driftSentence(measured: Double, predicted: Double) -> LocalizedStringKey {
        "The last render took \(Figures.seconds(measured)) where this Mac's measure predicted \(Figures.seconds(predicted)). If the Mac has changed (heat, other programs), measure it again: the new measure applies from the next launch."
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
            }
            Spacer()
            switch app.diagnostic {
            case .ready:
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Start") { app.startDiagnostic() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(app.diagnosticBlocker != nil)
            case .running:
                Button("Stop") { app.cancelDiagnostic() }.keyboardShortcut(.cancelAction)
            case .failed:
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Measure Again") { notice = nil; app.startDiagnostic() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(app.diagnosticBlocker != nil)
            case .finished(let report):
                Button("Measure Again") { notice = nil; app.startDiagnostic() }
                    .disabled(app.diagnosticBlocker != nil)
                Button("Copy JSON") { app.copyJSON(report) }
                    .help("To send the report another way, without a GitHub account")
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Open the Issue") { notice = app.openIssue(report) }
                    .keyboardShortcut(.defaultAction)
                    .help("Copies the report and opens GitHub's form; you paste it and submit it there")
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

    /// The label of a kernel's row, in the order the sheet shows them.
    private static var rows: [(kernel: CapacityModel.Kernel, label: String)] {
        [(.init(.gpu, .fp32), String(localized: "Matrix products, GPU")),
         (.init(.amx, .fp32), String(localized: "Matrix products, CPU matrix unit")),
         (.init(.ane, .fp16), String(localized: "Matrix products, Neural Engine")),
         (.init(.gpu, .bf16), String(localized: "Matrix products, GPU tensor units on the bf16 weights")),
         (.init(.gpu, .int8), String(localized: "Matrix products on the 8-bit weights, read as published")),
         (.init(.gpu, .q8), String(localized: "Matrix products on the GGUF 8-bit weights, read as published")),
         (.init(.gpu, .graph), String(localized: "Attention, system kernel")),
         (.init(.gpu, .flash), String(localized: "Attention, Siliconed kernel")),
         (.init(.cpu, .heads), String(localized: "Attention, split between GPU and CPU"))]
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
                Text("Identical calls differed by more than a free Mac's: those figures are read on their quick side.")
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
        case .computedWrong: String(localized: "It computed wrong: the plan has no model of it, and the engine's defaults answer.")
        case .notShared: String(localized: "They were not measured together: the plan does not offer them as a set.")
        case .implausible: String(localized: "A figure that looks wrong, probably because the Mac was busy: the plan bounds it or leaves it out.")
        case .memoryOverNeed: String(localized: "The measure used more memory than had been counted for it.")
        case .memoryOverBudget: String(localized: "The measure needs more memory than this Mac allows.")
        case .swapGrew: String(localized: "The Mac swapped during the measure: other apps held its memory.")
        case .tookTooLong: String(localized: "The measure took longer than a minute.")
        case .disturbed: String(localized: "The Mac was busy while it was measured: the figures that differed are read on their quick side.")
        }
        return (subject, sentence)
    }

    /// The engine's name for a product kernel (the form it reads is in the tooltip), the row's label for an attention kernel — an
    /// attention and a product of one engine are two things to the reader —; two or three engines together, joined by « + ».
    private static func name(of finding: Diagnostic.Finding) -> String {
        if let kernel = finding.kernel {
            if [.graph, .flash, .heads].contains(kernel.form), let row = rows.first(where: { $0.kernel == kernel }) { return row.label }
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

    /// **What each engine reaches**, against the peak read: a bar and `3.53 TFLOP/s · 76 %` where the inventory gave a peak, the rate
    /// alone where it did not; a kernel the judge left out, ✗.
    @ViewBuilder private var reached: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("What each engine reaches").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                ForEach(Self.rows, id: \.kernel.description) { row in
                    let line = report.reached.first { $0.kernel == row.kernel }
                    let excluded = report.judge[row.kernel.description]?.excluded == true
                    if line != nil || excluded {
                        GridRow {
                            Text(verbatim: row.label).foregroundStyle(.secondary)
                            if let line, !excluded {
                                bar(line)
                                figure(line).monospacedDigit()
                                verdict(true)
                            } else {
                                Color.clear.frame(width: Self.barWidth, height: 1)
                                Text(verbatim: "—").foregroundStyle(.secondary)
                                verdict(false, died: report.judge[row.kernel.description]?.reason == .died)
                            }
                        }
                    }
                }
            }
            .font(.callout)
            .textSelection(.enabled)
        }
    }

    private static let barWidth: CGFloat = 80

    /// The share of the peak read, as a level (capped at the bar's end: a tensor form may run above the fp32 peak); nothing where no
    /// peak was read — the bar's room stays, so that the figures line up.
    @ViewBuilder private func bar(_ line: Diagnostic.Report.Reach) -> some View {
        if let fraction = line.fraction, let crest = line.crest {
            Capsule().fill(Palette.well)
                .overlay(alignment: .leading) {
                    Capsule().fill(Palette.live).frame(width: Self.barWidth * min(max(fraction, 0.02), 1))
                }
                .frame(width: Self.barWidth, height: 6)
                .help("Peak read from the silicon: \(Self.rate(crest))")
        } else {
            Color.clear.frame(width: Self.barWidth, height: 1)
        }
    }

    /// `3.53 TFLOP/s`, then ` · 76 %` of the peak read where there is one.
    private func figure(_ line: Diagnostic.Report.Reach) -> some View {
        HStack(spacing: 0) {
            Text(verbatim: Self.rate(line.rate))
            if let fraction = line.fraction {
                Text(verbatim: " · " + Figures.percent(Int((100 * fraction).rounded()))).foregroundStyle(.secondary)
            }
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
                Text("The last render's matrix products and attention took \(Figures.seconds(render.measured)) where \(Figures.seconds(predicted)) was predicted; this goes with the report.")
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
