// **« Report My Configuration »: what this Mac does, measured, then sent as a GitHub issue if the user
// wants** (`docs/API.md` §3.13).
//
// Three states in one sheet: what the diagnostic does and how long (it loads each model's weights,
// ~15 s per model, so it waits for an empty queue — and the queue waits for it); its progress; then the
// report as a person reads it — the chip, the memory, and per model the step in steady state, the
// render it implies (« estimated », never « measured »), the memory peak, the swap, the deviation from
// the reference ✓/✗. « Open the issue » opens GitHub's form prefilled by `report.issueURL()`; nothing
// is sent without that click, and the issue is submitted in the browser, by the user.

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
                Text("Report My Configuration").font(.title2.weight(.semibold))
                Text("Measures the installed models on this Mac, at 512² — then, if you want, opens a prefilled GitHub issue. Nothing is sent without your click.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(20)
            Divider()
            Group {
                switch app.diagnostic {
                case .ready: ready
                case let .running(done, total, line): running(done: done, total: total, line: line)
                case .finished(let report): ReportView(report: report)
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
            Text("For each model below: the text encoder, two steps of the denoiser — the second, judged against the reference computed in fp32 —, and the decoder. About 15 seconds per model; the queue waits meanwhile.")
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
            ForEach(app.diagnosticCards) { card in
                Label(card.name, systemImage: "cpu")
            }
            if let reason = app.diagnosticBlocker {
                Label(reason, systemImage: "hourglass").foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(20)
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
            case .finished(let report):
                Button("Measure Again") { notice = nil; app.startDiagnostic() }
                    .disabled(app.diagnosticBlocker != nil)
                Button("Copy JSON") { app.copyJSON(report) }
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Open the Issue") { notice = app.openIssue(report) }
                    .keyboardShortcut(.defaultAction)
                    .help("Opens GitHub's form, prefilled with this report; you submit it there")
            }
        }
    }
}

/// **The report, as a person reads it** — the JSON says the same, and more.
private struct ReportView: View {
    let report: Diagnostic.Report

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                machine
                ForEach(report.models, id: \.model) { ModelLine(model: $0) }
                Text("\(Figures.duration(report.totalSeconds)) in all · peak \(Figures.memory(report.peakFootprintBytes)) · \(report.swapouts) pages swapped out")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(20)
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

/// One model: its timings, its memory, and the verdict against the reference.
private struct ModelLine: View {
    let model: Diagnostic.ModelReport

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(model.name).font(.headline)
                Spacer()
                verdict
            }
            if let error = model.error {
                // The app's sentence for the engine's case; the engine's text under it, as the issue quotes it.
                if let failure = model.failure {
                    let p = Problem(failure)
                    Text(p.title).font(.callout).foregroundStyle(.red)
                    if let s = p.suggestion { Text(s).font(.caption).foregroundStyle(.secondary) }
                }
                Text(verbatim: error).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
            } else if let s = model.seconds {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 3) {
                    GridRow {
                        Text("Step in steady state").foregroundStyle(.secondary)
                        Text(Figures.seconds(s.steadyEvaluation)).monospacedDigit()
                    }
                    if let e = model.estimatedRenderSeconds {
                        GridRow {
                            Text("Render at 512², estimated").foregroundStyle(.secondary)
                            Text(verbatim: "≈ " + Figures.duration(e)).monospacedDigit()
                        }
                    }
                    GridRow {
                        Text("Memory peak").foregroundStyle(.secondary)
                        Text(Figures.memory(model.peakFootprintBytes)).monospacedDigit()
                    }
                    GridRow {
                        Text("Swap").foregroundStyle(.secondary)
                        Text(model.swapouts == 0 ? String(localized: "none") : String(localized: "\(model.swapouts) pages"))
                            .foregroundStyle(model.swapouts == 0 ? Color.primary : Color.orange)
                    }
                    if let d = model.deviation {
                        GridRow {
                            Text("Deviation from the reference").foregroundStyle(.secondary)
                            Text(verbatim: String(format: "%.2e", d.worst) + (d.pass ? " ≤ " : " > ") + String(format: "%.0e", d.threshold))
                                .monospacedDigit()
                        }
                    }
                }
                .font(.callout)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
    }

    @ViewBuilder private var verdict: some View {
        if model.error != nil {
            Label("failed", systemImage: "xmark.octagon.fill").foregroundStyle(.red)
        } else if let d = model.deviation {
            if d.pass {
                Label("matches the reference", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            } else {
                Label("differs from the reference", systemImage: "xmark.circle.fill").foregroundStyle(.red)
            }
        } else {
            Label("timed (no reference yet)", systemImage: "clock").foregroundStyle(.secondary)
        }
    }
}
