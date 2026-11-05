// **A series on the canvas: its grid, filling in as its cells come** (`Series.swift` says what varies).
//
//     « Generate » with something bracketed
//        │ one job per cell (`AppState.enqueueGrid`), each a finished image into the history
//        ▼
//     the canvas   the grid (`SeriesCanvas`): cells fill in as they come, the running one shows its preview and, under
//                  the grid, its settings live; it scrolls, and pinch, ⌘+ ⌘− or the slider size the cells. The grid waits
//                  at the head of the history strip once an image is chosen.
//        │ a cell
//        ▼
//     a click chooses it and says its settings; a double-click shows its image, which is in the history like any other.
//     « Add Sheet to History »: the cells side by side, labelled — an image of the history.
//
// What an analogy with a contact sheet would miss:
//
// - **A cell is a finished image** (4/11): the render « Generate » would make alone, at the rack's size and steps —
//   nothing to finish. The grid only lays the history's images out.
// - **A new series does not replace the last one's cells**: « Generate » queues, as it always does; the canvas shows the
//   newest grid, and « Stop » takes the shown one's waiting cells out of the queue.
// - **The sheet only when asked**: the canvas is the series' sheet; `silicontrol grid` draws its own when its last
//   cell ends (`GridRun.sheet`).

import AppKit
import Siliconed
import SwiftUI

// MARK: - The state

extension AppState {
    /// Whether the series shown still has cells on their way.
    var exploring: Bool {
        guard let r = explored else { return false }
        return r.jobs.contains { j in currentJob?.ordinal == j || file.contains { $0.ordinal == j } }
    }

    /// **« Stop »**: the series' cells still waiting leave the queue; the running one stops.
    func stopExploring() {
        guard let r = explored else { return }
        for job in file where r.jobs.contains(job.ordinal) { remove(job) }
        if let c = currentJob, r.jobs.contains(c.ordinal) { cancel() }
    }

    /// What a column or row says, for the series shown.
    func exploreLabels(_ axis: RenderGrid.Axis?, of run: GridRun) -> [String] {
        guard let axis else { return [] }
        let stack = run.settings.loras
        var name = "LoRA"
        if case .loraStrength(let slot, _) = axis, stack.indices.contains(slot) { name = loraName(stack[slot].path) }
        return run.grid.values(of: axis).map { GridWords.label($0, loraName: name, modelName: modelName, addedName: loraName) }
    }
}

// MARK: - The grid, in the main window's canvas

/// **The series' cells in the canvas**: they fill in as they come, the running one shows its preview; pinch, ⌘+ ⌘− or
/// the slider size them, the canvas scrolls. A click chooses a cell, a double-click shows its image.
struct SeriesCanvas: View {
    @Bindable var app: AppState
    let run: GridRun
    @State private var chosen: Int?
    @GestureState private var pinch: CGFloat = 1

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView([.vertical, .horizontal]) {
                cells.padding(24)
            }
            .gesture(MagnifyGesture()
                .updating($pinch) { value, state, _ in state = value.magnification }
                .onEnded { value in app.exploreCellSide = AppState.clampedCellSide(app.exploreCellSide * value.magnification) })
            // The chosen cell — or, while the series runs, the one rendering, live.
            if let i = chosen, run.cellJobs.indices.contains(i), run.entries[i] != nil {
                Divider()
                detail(i)
            } else if let i = running {
                Divider()
                detail(i, live: true)
            }
        }
    }

    /// The cell rendering now, if it is one of this grid's.
    private var running: Int? { app.currentJob.flatMap { run.jobs.firstIndex(of: $0.ordinal) } }

    private var side: CGFloat { AppState.clampedCellSide(app.exploreCellSide * pinch) }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "square.grid.3x3").foregroundStyle(.secondary)
            Text(run.grid.title).font(.headline).lineLimit(1).truncationMode(.middle)
            Spacer()
            Text("\(run.entries.count) of \(run.jobs.count)").monospacedDigit().foregroundStyle(.secondary)
            HStack(spacing: 4) {
                Image(systemName: "photo").imageScale(.small)
                Slider(value: $app.exploreCellSide, in: AppState.cellSides).frame(width: 110).controlSize(.small).tint(Palette.control)
                Image(systemName: "photo").imageScale(.large)
            }
            .foregroundStyle(.secondary)
            .help("Cell size — or pinch, ⌘+ and ⌘−")
            if app.exploring {
                Button("Stop") { app.stopExploring() }
                    .help("The cells still waiting leave the queue; the running one stops")
            }
            Button("Add Sheet to History") { app.addSheet(run) }
                .disabled(run.entries.isEmpty || app.exploring)
                .help("The images side by side, labelled — an image of the history, in memory like the others")
            Button { app.showsExploration = false } label: { Image(systemName: "xmark") }
                .buttonStyle(.borderless)
                .help("Back to the images — the grid stays at the head of the strip")
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
    }

    private var cells: some View {
        let g = run.grid
        let columns = app.exploreLabels(g.x, of: run), rows = app.exploreLabels(g.y, of: run)
        let f = run.settings.format
        let (w, h) = f.width >= f.height ? (side, side * CGFloat(f.height) / CGFloat(f.width))
                                         : (side * CGFloat(f.width) / CGFloat(f.height), side)
        return Grid(horizontalSpacing: 8, verticalSpacing: 8) {
            GridRow {
                if !rows.isEmpty { Color.clear.frame(width: 1, height: 1) }
                ForEach(columns.indices, id: \.self) { c in
                    Text(columns[c]).font(.callout.weight(.semibold)).lineLimit(2).multilineTextAlignment(.center)
                        .frame(width: w)
                }
            }
            ForEach(0..<g.rows, id: \.self) { r in
                GridRow {
                    if !rows.isEmpty {
                        Text(rows[r]).font(.callout.weight(.semibold)).lineLimit(3).multilineTextAlignment(.trailing)
                            .frame(width: 120, alignment: .trailing)
                    }
                    ForEach(0..<g.columns, id: \.self) { c in
                        cell(r * g.columns + c).frame(width: w, height: h)
                    }
                }
            }
        }
    }

    @ViewBuilder private func cell(_ i: Int) -> some View {
        let job = run.jobs[i]
        ZStack {
            if let e = run.entries[i] {
                CellImage(entry: e, side: side)
            } else if app.currentJob?.ordinal == job {
                if let p = app.preview { Image(decorative: p, scale: 1).resizable().interpolation(.medium).opacity(0.85) }
                else { Rectangle().fill(.quaternary) }
                ProgressView().controlSize(.small)
            } else {
                Rectangle().fill(Palette.well)
                Text(run.issues[i] == nil ? String(localized: "Waiting") : GridWords.note(run.issues[i]))
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .overlay(RoundedRectangle(cornerRadius: 4)
            .strokeBorder(chosen == i ? Palette.live : running == i ? Palette.live : .clear, lineWidth: 3))
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { showImage(i) }
        .onTapGesture { chosen = run.entries[i] == nil ? nil : i }
        .contextMenu {
            if let e = run.entries[i] {
                Button("Show This Image") { showImage(i) }
                Button("Reuse These Settings") { app.restoreSettings(e) }
            }
        }
        .help(run.cellJobs.indices.contains(i)
              ? run.cellJobs[i].settings.prompt + " · " + String(localized: "seed \(String(run.cellJobs[i].seeds[0]))") : "")
    }

    /// The cell's image, shown as the history shows it — if it is still there.
    private func showImage(_ i: Int) {
        guard let e = run.entries[i], app.history.contains(where: { $0.id == e.id }) else { return }
        app.show(e.id)
    }

    private func detail(_ i: Int, live: Bool = false) -> some View {
        let job = run.cellJobs[i]
        let loras = job.settings.loras.map { "\(app.loraName($0.path)) \(Figures.strength($0.strength))" }
        return HStack(alignment: .center, spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                if live {
                    Text("Rendering cell \(i + 1) of \(run.jobs.count) — step \(app.currentStep)/\(app.totalSteps)")
                        .font(.caption.weight(.semibold)).foregroundStyle(Palette.live).monospacedDigit()
                }
                Text(job.settings.prompt).lineLimit(2)
                Text(verbatim: ([job.settings.modelName, String(localized: "\(job.settings.steps) steps"),
                                 String(localized: "seed \(String(job.seeds[0]))")] + loras).joined(separator: " · "))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if !live, let e = run.entries[i] {
                Button("Reuse These Settings") { app.restoreSettings(e) }
                    .help("The rack takes this image’s settings — model, prompt, size, LoRAs, seed — without rendering")
                Button("Show This Image") { showImage(i) }
                    .buttonStyle(.prominent)
                    .help("The image, as the history shows it (double-click a cell)")
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
    }
}

/// A cell's image: its thumbnail while it is small, its PNG decoded once it is shown larger.
private struct CellImage: View {
    let entry: Entry
    let side: CGFloat
    @State private var full: CGImage?

    var body: some View {
        Image(side > 240 ? (full ?? entry.vignette) : entry.vignette, scale: 1,
              label: Text(verbatim: "\(entry.settings.prompt) · \(entry.legend)"))
            .resizable().interpolation(.high)
            .task(id: side > 240) { if side > 240, full == nil { full = await Entry.decoded(entry.png).image } }
    }
}
