// **« Measure This Mac »: what this Mac computes, measured — the map the engine tunes itself to —, then
// sent as a GitHub issue only if the user wants** (`Diagnostic.swift`). The sheet says the tuning first:
// sharing is an option, not its purpose (the user, 19/10/2026).
//
// Three states in one sheet: what the diagnostic does and how long (~1 min, no model needed; it occupies
// the GPU and the CPU, so it waits for an empty queue — and the queue waits for it); its progress; then
// the report as a person reads it — the chip, the memory, and per capacity its throughput and ✓/✗, the
// output computed right against the exact or not. « Open the issue » copies the report and opens GitHub's
// form (`report.issueURL(includingReport: false)`), where the user pastes it; « Copy JSON » alone serves
// whoever has no GitHub account. Nothing is sent without a click, and the issue is submitted in the
// browser, by the user.

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
                Text("Siliconed measures what this Mac's chip computes, and how fast, then turns on only the compute paths that are right and faster here. Sharing the result is optional: a prefilled GitHub issue, opened only by your click.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(20)
            Divider()
            Group {
                switch app.diagnostic {
                case .ready: ready
                case let .running(done, total, line): running(done: done, total: total, line: line)
                case .finished(let report): ReportView(report: report, inUse: report == app.mapInUse)
                case .failed(let problem): ProblemView(problem: problem).padding(20)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            Divider()
            footer.padding(16).background(.bar)
        }
        .frame(width: 620, height: 560)
    }

    // ── before ──

    private var ready: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Matrix products, attention and weight widening, on the GPU and on the CPU, each checked against the exact result. No model is needed, nothing is downloaded. About a minute; the queue waits meanwhile.")
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

    /// **Where this session stands against the map**: a map in use; a newer one waiting for the next
    /// launch; none yet, or a stale one — the first render measures.
    @ViewBuilder private var mapStatus: some View {
        Text(Self.mapStatus(app)).font(.callout).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// The sentence of `mapStatus` — the settings window says it too.
    static func mapStatus(_ app: AppState) -> LocalizedStringKey {
        if let used = app.mapInUse {
            if used == app.storedMap, used.isProvisional {
                return "This session uses the measure of \(day(used.date)), taken while the Mac was warm, in Low Power Mode or short of memory: it is measured again at its next launch at rest."
            }
            return used == app.storedMap
                ? "This session uses the choices of this Mac's measure of \(day(used.date))."
                : "This session uses the measure of \(day(used.date)); the newer one applies from the next launch."
        }
        if app.remeasuringProvisional, let stored = app.storedMap {
            return "The measure of \(day(stored.date)) was taken while the Mac was not at rest: it is measured again before the next render."
        }
        if let stored = app.storedMap, !stored.isCurrent {
            return "The measure of \(day(stored.date)) was taken with another macOS or another version of Siliconed: this Mac is measured again before the next render."
        }
        return "This Mac is measured before its first render: Siliconed chooses its compute paths from it."
    }

    /// A report's ISO date as a day the user reads.
    static func day(_ iso: String) -> String {
        guard let date = ISO8601DateFormatter().date(from: iso) else { return iso }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    // ── during ──

    private func running(done: Int, total: Int, line: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            ProgressView(value: Double(done), total: Double(max(total, 1)))
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(line).font(.callout)
                Spacer()
                Text(verbatim: "\(done + 1)/\(total)").font(.callout).monospacedDigit().foregroundStyle(.secondary)
            }
            Text("The machine is busy: the timings are better left undisturbed.")
                .font(.caption).foregroundStyle(.secondary)
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

/// **The report, as a person reads it** — one line per capacity: the median throughput of its
/// measures, and ✓ when every one of them computed right. The JSON says each measure.
private struct ReportView: View {
    let report: Diagnostic.Report
    /// This report is the map this session's settings were read with.
    var inUse = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                machine
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 5) {
                    let products = report.products
                    row("Matrix products, GPU", products.map(\.gpu), unit: "TFLOP/s")
                    row("Matrix products, CPU matrix unit", products.compactMap(\.amx), unit: "TFLOP/s")
                    row("Matrix products, GPU and CPU together", products.compactMap(\.coexecution), unit: "TFLOP/s")
                    row("Matrix products, GPU tensor units on the bf16 weights", products.compactMap(\.tensor), unit: "TFLOP/s")
                    row("Matrix products on the 8-bit weights, GPU and CPU together", products.compactMap(\.int8Coexecution), unit: "TFLOP/s")
                    row("Matrix products on the GGUF 8-bit weights, GPU and CPU together", products.compactMap(\.q8Coexecution), unit: "TFLOP/s")
                    Divider().gridCellUnsizedAxes(.horizontal)
                    row("Attention, system kernel", report.attention.compactMap(\.graph), unit: "TFLOP/s")
                    row("Attention, Siliconed kernel", report.attention.compactMap(\.flash), unit: "TFLOP/s")
                    row("Attention, split between GPU and CPU", report.attention.compactMap(\.headSplit), unit: "TFLOP/s")
                    Divider().gridCellUnsizedAxes(.horizontal)
                    row("Weight widening, CPU", report.widening.map(\.cpu), unit: "G/s")
                    Divider().gridCellUnsizedAxes(.horizontal)
                    row("Half precision, GPU", engine: "mpsgraph")
                    row("Half precision, CPU matrix unit", engine: "bnns")
                    row("Half precision, GPU tensor units (Metal 4)", engine: "matmul2d")
                }
                .font(.callout)
                .textSelection(.enabled)
                if let d = report.decisions { decisions(d) }
                Text("\(Figures.duration(report.seconds)) in all")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(20)
        }
    }

    /// One capacity: the median rate of its measures, ✓ when all pass; "not available" without any.
    @ViewBuilder private func row(_ label: LocalizedStringKey, _ measures: [Diagnostic.Measure], unit: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            if measures.isEmpty {
                Text("not available").foregroundStyle(.secondary)
                Text(verbatim: "")
            } else {
                let rates = measures.map(\.rate).sorted()
                Text(verbatim: String(format: "%.2f ", rates[rates.count / 2]) + unit).monospacedDigit()
                verdict(measures.allSatisfy(\.pass))
            }
        }
    }

    /// Half precision: fp16 and bf16 side by side, one engine.
    @ViewBuilder private func row(_ label: LocalizedStringKey, engine: String) -> some View {
        let items = report.lowPrecision.filter { $0.engine == engine }
        let measured = items.compactMap { item in item.measure.map { (item.type, $0) } }
        GridRow {
            Text(label).foregroundStyle(.secondary)
            if measured.isEmpty {
                Text("not available").foregroundStyle(.secondary)
                Text(verbatim: "")
            } else {
                Text(verbatim: measured.map { String(format: "%@ %.2f", $0.0 as NSString, $0.1.rate) }.joined(separator: " · ")
                     + " TFLOP/s").monospacedDigit()
                verdict(measured.allSatisfy { $0.1.pass })
            }
        }
    }

    /// **What these measures make this Mac use** (`Diagnostic.decide`): each branch, used or not, and the
    /// gain behind it — what a shared report says the machine chose.
    @ViewBuilder private func decisions(_ d: Diagnostic.Decisions) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("What this Mac uses").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 5) {
                // The bf16 path has no AMX: when it is taken, co-execution is not, whatever it measured.
                let tensorOn = d.tensor?.on == true
                decision("Matrix products, GPU and CPU together", d.amx, on: d.amx.on && !tensorOn,
                         detail: d.amx.on && !tensorOn ? d.amxFraction.map { String(format: " · GPU %.0f %%", $0 * 100) } ?? "" : "")
                if let tensor = d.tensor { decision("Matrix products, GPU tensor units on the bf16 weights", tensor) }
                if let int8 = d.int8 { decision("Matrix products on the 8-bit weights, read as published", int8) }
                if let q8 = d.q8 { decision("Matrix products on the GGUF 8-bit weights, read as published", q8) }
                decision("Attention, Siliconed kernel", d.flash)
                decision("Attention, split between GPU and CPU", d.headSplit)
            }
            .font(.callout)
            .textSelection(.enabled)
            let noise = String(format: "%.0f %%", d.margin * 100)
            Text(inUse ? "In use in this session; a gain within this run's noise (±\(noise)) does not count."
                       : "In use from the next launch; a gain within this run's noise (±\(noise)) does not count.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private func decision(_ label: LocalizedStringKey, _ x: Diagnostic.Decision, on: Bool? = nil,
                                       detail: String = "") -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(on ?? x.on ? "Used" : "Not used")
            if x.passed {
                Text(verbatim: String(format: "×%.2f", x.gain) + detail).monospacedDigit()
            } else {
                verdict(false)
            }
        }
    }

    @ViewBuilder private func verdict(_ pass: Bool) -> some View {
        if pass {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).help("Computed right: within its threshold of the exact result")
        } else {
            Image(systemName: "xmark.circle.fill").foregroundStyle(.red).help("Computed wrong: beyond its threshold of the exact result")
        }
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
