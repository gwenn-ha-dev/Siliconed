// **« Measure This Mac »: what this Mac computes, measured — the model the engine plans with —, then
// sent as a GitHub issue only if the user wants** (`Diagnostic.swift`). The sheet says the tuning first:
// sharing is an option, not its purpose (the user, 19/10/2026).
//
// Three states in one sheet: what the diagnostic does and how long (under a minute, no model needed; it occupies
// the GPU and the CPU, so it waits for an empty queue — and the queue waits for it); its progress; then
// the report as a person reads it — the chip, the memory, and per engine what it reaches against the peak
// the silicon announces and ✓/✗ (computed right against the exact or not), the verdict and the duration.
// « Open the issue » copies the report and opens GitHub's form (`report.issueURL()`), where the user pastes
// it; « Copy JSON » alone serves whoever has no GitHub account. Nothing is sent without a click, and the issue
// is submitted in the browser, by the user.

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
                case .finished(let report): ReportView(report: report, inUse: app.inUse(report))
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
            Text("Matrix products and attention on the GPU, the AMX and the Neural Engine, alone and together, each checked against the exact result: the figures this Mac renders with. No model is needed, nothing is downloaded. Under a minute; the queue waits meanwhile.")
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
            if used == map.stored, !used.green {
                return "This session uses the measure of \(day(used.date)), which was not complete (a missing or aberrant figure, its memory, its swap or its minute): it is measured again at its next launch."
            }
            return used == map.stored
                ? "This session uses the choices of this Mac's measure of \(day(used.date))."
                : "This session uses the measure of \(day(used.date)); the newer one applies from the next launch."
        }
        if map.remeasuring, let stored = map.stored {
            return "The measure of \(day(stored.date)) was not complete (a missing or aberrant figure, its memory, its swap or its minute): it is measured again before the next render."
        }
        if let stored = map.stored, !stored.current {
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

/// **The report, as a person reads it** — the machine, then one line per kernel: what it reaches (TFLOP/s, and against the peak the
/// inventory read where there is one) and ✓ when it computed right, ✗ when the judge left it out; the verdict; the duration. The JSON
/// says the rest.
private struct ReportView: View {
    let report: Diagnostic.Report
    /// This report is the map this session's settings were read with.
    var inUse = false

    /// The label of a kernel's row, in the order the sheet shows them.
    private static let rows: [(kernel: CapacityModel.Kernel, label: LocalizedStringKey)] = [
        (.init(.gpu, .fp32), "Matrix products, GPU"),
        (.init(.amx, .fp32), "Matrix products, CPU matrix unit"),
        (.init(.ane, .fp16), "Matrix products, Neural Engine"),
        (.init(.gpu, .bf16), "Matrix products, GPU tensor units on the bf16 weights"),
        (.init(.gpu, .int8), "Matrix products on the 8-bit weights, read as published"),
        (.init(.gpu, .q8), "Matrix products on the GGUF 8-bit weights, read as published"),
        (.init(.gpu, .graph), "Attention, system kernel"),
        (.init(.gpu, .flash), "Attention, Siliconed kernel"),
        (.init(.cpu, .heads), "Attention, split between GPU and CPU"),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                machine
                reached
                if report.verdict.green {
                    Text("The measure is complete.").font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("The measure is not complete: \(report.verdict.reasons.joined(separator: ", ")).")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                Text("\(Figures.duration(report.seconds)) in all")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(20)
        }
    }

    /// **What each engine reaches**, against the peak read: `3.53 / 4.65 TFLOP/s · 76 %`; a kernel the judge left out, ✗.
    @ViewBuilder private var reached: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("What each engine reaches").font(.headline)
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 5) {
                ForEach(Self.rows, id: \.kernel.description) { row in
                    let line = report.reached.first { $0.kernel == row.kernel }
                    let excluded = report.judge[row.kernel.description]?.excluded == true
                    if line != nil || excluded {
                        GridRow {
                            Text(row.label).foregroundStyle(.secondary)
                            if let line, !excluded {
                                Text(verbatim: Self.figure(line)).monospacedDigit()
                                verdict(true)
                            } else {
                                Text(verbatim: "—").foregroundStyle(.secondary)
                                verdict(false)
                            }
                        }
                    }
                }
            }
            .font(.callout)
            .textSelection(.enabled)
        }
    }

    /// `3.53 TFLOP/s`, then ` · 76 % of 4.65` — the share of the peak the inventory read for that engine, if it read one.
    static func figure(_ line: Diagnostic.Report.Reach) -> String {
        let rate = line.rate * 1e-12
        guard let crest = line.crest, let fraction = line.fraction else { return String(format: "%.2f TFLOP/s", rate) }
        return String(format: "%.2f / %.2f TFLOP/s · %.0f %%", rate, crest * 1e-12, 100 * fraction)
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
