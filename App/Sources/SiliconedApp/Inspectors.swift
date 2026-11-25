// **Two floating panels over the app**, each answering its own question:
//
// - **« Chain »**: where the render stands, in the chain's words — each stage, its module, its time, the unit that
//   computed it; the queue, the last image, the session (`ChainInspector.swift`). A stage's line in the rack opens it at
//   that stage.
// - **« Machine »**: what the Mac does, in time — the Mac, what Core AI compiles, and the render's memory and compute on
//   one time axis (`MachineInspector.swift`). Core AI's light at the foot of the column opens it.
//
// Panels and not panes of the window: a pane would cover the image or resize it at each opening. They keep nothing —
// neither position nor restored state (incognito) — and follow the switch to another Mac's mirror by themselves.
//
// **What the engine did for one image** is not a window: it is the popover of the ⓘ under the image (`ImageInspector`),
// beside what it describes, gone with a click elsewhere.

import AppKit
import Siliconed
import SwiftUI

/// Which stage the « Chain » inspector was asked to show, and how many times — a second click on the same line scrolls
/// again.
@MainActor @Observable
final class InspectorFocus {
    var stage: Engine.Stage?
    var asked = 0
}

@MainActor
final class Inspectors {
    enum Kind { case chain, machine }

    private let carrier: Carrier
    private let local: AppState
    private var panels: [Kind: NSPanel] = [:]
    let focus = InspectorFocus()

    init(carrier: Carrier, local: AppState) { (self.carrier, self.local) = (carrier, local) }

    func hide() { panels.values.forEach { $0.orderOut(nil) } }

    func isShown(_ kind: Kind) -> Bool { panels[kind]?.isVisible == true }

    func toggle(_ kind: Kind, beside window: NSWindow?) {
        if isShown(kind) { panels[kind]?.orderOut(nil); return }
        show(kind, beside: window)
    }

    /// « Chain », scrolled to `stage`'s line and that line marked.
    func show(stage: Engine.Stage, beside window: NSWindow?) {
        focus.stage = stage
        focus.asked += 1
        show(.chain, beside: window)
    }

    func show(_ kind: Kind, beside window: NSWindow?) {
        let p = panels[kind] ?? make(kind, beside: window)
        panels[kind] = p
        p.orderFront(nil)
    }

    private func make(_ kind: Kind, beside window: NSWindow?) -> NSPanel {
        let size = kind == .chain ? NSSize(width: 380, height: 640) : NSSize(width: 520, height: 600)
        let p = NSPanel(contentRect: NSRect(origin: .zero, size: size),
                        styleMask: [.titled, .closable, .resizable, .utilityWindow, .nonactivatingPanel],
                        backing: .buffered, defer: false)
        // Not « Chain »: the detached column's window bears that name, and the Window menu listed two « Chain ».
        p.title = kind == .chain ? String(localized: "Chain Inspector") : String(localized: "Machine")
        p.isFloatingPanel = true
        p.hidesOnDeactivate = false
        p.isRestorable = false
        p.isReleasedWhenClosed = false
        let focus = focus
        p.contentViewController = kind == .chain
            ? Self.host(InspectorRoot(carrier: carrier, local: local) { ChainInspector(app: $0, focus: focus) })
            : Self.host(InspectorRoot(carrier: carrier, local: local) { MachineInspector(app: $0) })
        p.setContentSize(size)
        if let f = window?.frame {
            // The chain along the window's right edge, over the history; the machine beside it, further in.
            let x = kind == .chain ? f.maxX - 400 : f.maxX - 940
            p.setFrameTopLeftPoint(NSPoint(x: x, y: f.maxY - 60))
        } else {
            p.center()
        }
        return p
    }

    /// The panel's own size kept: the content does not size it.
    private static func host(_ view: some View) -> NSViewController {
        let h = NSHostingController(rootView: view)
        h.sizingOptions = []
        return h
    }
}

/// A panel's content: the state the window shows now, this Mac's or a mirror's.
private struct InspectorRoot<Content: View>: View {
    let carrier: Carrier
    let local: AppState
    @ViewBuilder let content: (AppState) -> Content

    var body: some View {
        let app = carrier.mirror ?? local
        content(app).id(ObjectIdentifier(app))
    }
}

// MARK: - « Image »

/// **What the engine did for the image shown**: its settings, and the time, memory and units of each stage — the figures
/// the CLI prints, for each history image. The popover of the ⓘ under the image.
struct ImageInspector: View {
    let app: AppState

    /// As tall as what it says — a popover takes its content's height —, scrolling only past 640 points (a long model
    /// prompt opened, many LoRAs).
    var body: some View {
        ViewThatFits(in: .vertical) {
            content
            ScrollView { content }
        }
        .frame(width: 420)
        .frame(maxHeight: 640)
    }

    @ViewBuilder private var content: some View {
        if let entry = app.selectedEntry {
            ImageDetails(app: app, entry: entry)
        } else {
            Text("No image is shown.")
                .font(.callout).foregroundStyle(Palette.secondary)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(40)
        }
    }
}

private struct ImageDetails: View {
    let app: AppState
    let entry: Entry

    var body: some View {
        let r = entry.settings, c = entry.timings, m = entry.footprints
        // An enlargement or a crop was not rendered: no evaluations, no stage; its zeros said « 0 of 8 steps », « 0.00 s ».
        let rendered = entry.enlarged == nil && entry.cropped == nil
        VStack(alignment: .leading, spacing: 12) {
            Text(r.prompt).font(.callout).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            // Named subjects: what the model read, under what was typed — to understand a miss.
            if let model = r.modelPrompt {
                DisclosureGroup("Model Prompt") {
                    Text(model).font(.caption).foregroundStyle(Palette.secondary).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .font(.caption)
            }
            Text(entry.legend).font(.caption).monospacedDigit().foregroundStyle(Palette.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 5) {
                rowLine("Model", r.modelName)
                rowLine("Size", Figures.size(entry.width, entry.height))
                rowLine("Seed", String(entry.seed))
                if rendered { rowLine("Evaluations", String(localized: "\(entry.evaluations) of \(r.steps) steps")) }
                // A cell: its series, and its place in the grid.
                if let at = app.cell(of: entry), let run = app.series[at.grid] {
                    rowLine("Series", String(localized: "\(run.name), column \(at.cell % run.grid.columns + 1), row \(at.cell / run.grid.columns + 1)"))
                }
                if r.editing { rowLine("References", String(r.references - r.subjectImages)) }
                if r.subjectImages > 0 { rowLine("Subject images", String(r.subjectImages)) }
                // The variation seeds, in the order they turned the noise, each with its strength.
                if r.isVariation {
                    rowLine("Variation", r.variations.map { "\($0.seed) · \(VariationWords.label($0.strength))" }
                        .joined(separator: " → "))
                }
                ForEach(r.loras) { l in
                    rowLine("LoRA", "\(app.loraName(l.path)) · \(Figures.strength(l.strength))")
                }
                if rendered {
                    Divider().gridCellColumns(4)
                    GridRow {
                        Text("Stage").foregroundStyle(Palette.secondary)
                        Text("Time").foregroundStyle(Palette.secondary).gridColumnAlignment(.trailing)
                        Text("Memory held").foregroundStyle(Palette.secondary).gridColumnAlignment(.trailing)
                        Text("Computed on").foregroundStyle(Palette.secondary).gridColumnAlignment(.trailing)
                    }
                    .font(.caption)
                    stage("Text", c.text, m, .text)
                    if c.encoding > 0 { stage("Image", c.encoding, m, .image) }
                    stage("DiT", c.denoising, m, .denoising)
                    stage("VAE", c.decoding, m, .decoding)
                    GridRow {
                        Text("Total").fontWeight(.semibold)
                        Text(Figures.seconds(c.total)).fontWeight(.semibold).monospacedDigit()
                        Text(verbatim: "")
                        Text(m.neuralEngine.map(UnitShare.words) ?? "—").fontWeight(.semibold).monospacedDigit()
                    }
                } else if c.total > 0 {
                    // macOS's super-resolution: its time, the one figure it has.
                    Divider().gridCellColumns(4)
                    GridRow {
                        Text("Total").fontWeight(.semibold)
                        Text(Figures.seconds(c.total)).fontWeight(.semibold).monospacedDigit()
                    }
                }
            }
            .font(.callout)
            HStack {
                Button("Copy Prompt") { copy(r.prompt) }
                Button("Copy Seed") { copy(String(entry.seed)) }
                Spacer()
                Button("Reuse These Settings") { app.restoreSettings(entry) }
            }
            .controlSize(.small)
        }
        .padding(16)
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        app.flash(String(localized: "Copied"))
    }

    private func rowLine(_ title: LocalizedStringKey, _ value: String) -> some View {
        GridRow {
            Text(title).foregroundStyle(Palette.secondary)
            Text(value).textSelection(.enabled).gridCellColumns(3).lineLimit(2).truncationMode(.middle)
        }
    }

    private func stage(_ title: LocalizedStringKey, _ seconds: Double, _ m: Engine.Footprints, _ stage: Engine.Stage) -> some View {
        let held = m.stages.first { $0.stage == stage }
        return GridRow {
            Text(title)
            Text(Figures.seconds(seconds)).monospacedDigit()
            Text(held.map { Figures.memory($0.held) } ?? "—").monospacedDigit().foregroundStyle(Palette.secondary)
                .help(held.map { String(localized: "base \(Figures.memory($0.counted)) · peak \(Figures.memory($0.held)) under the limit of \(Figures.memory($0.budget))") } ?? "")
            Text(held.map { UnitShare.words($0.neuralEngine ?? 0) } ?? "—").monospacedDigit().foregroundStyle(Palette.secondary)
                .help(UnitShare.help)
        }
    }
}
