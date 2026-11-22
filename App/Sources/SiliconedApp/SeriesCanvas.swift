// **A series on the canvas and in the history** (`Series.swift` says what varies, `Grid.swift` how it renders).
//
//     « Generate » with something bracketed, `silicontrol grid`
//        │ one job per cell (`AppState.enqueueGrid`), each a finished image into the history
//        ▼
//     the history  the series' tile, at its place in time, its cells under it — folded or not (`strip`)
//     the canvas   the grid (`SeriesCanvas`): cells fill in as they come, the running one shows its preview and, under
//                  the grid, its settings live; it fills the room and follows the window, larger or smaller, until
//                  pinch, ⌘+ ⌘− or the slider of the top band (`SeriesBar`) size the cells; past the room, it scrolls
//        │ a cell
//        ▼
//     a click chooses it and says its settings; a double-click shows its image, which is in the history like any other.
//
// What an analogy with a contact sheet would miss:
//
// - **A cell is a finished image**: the render « Generate » would make alone, at the rack's size and steps —
//   nothing to finish. The grid only lays the history's images out, found there by their job (`cell(of:)`): a cell
//   taken out of the history is empty here too, until ⌘Z.
// - **A new series does not replace the last one**: « Generate » queues, as it always does; every series of the session
//   keeps its tile, and its click opens its grid again. « Stop » takes the shown one's waiting cells out of the queue.
// - **The sheet is an export of the tile** (« Export Sheet… », « Copy Sheet »), never an image of the history.

import AppKit
import Siliconed
import SwiftUI

// MARK: - The state

/// **A row of the history column**: an image — a cell under its series' tile or not —, or a series' tile.
enum StripRow: Identifiable {
    case image(Entry, inSeries: Bool)
    case series(GridRun)

    var id: String {
        switch self {
        case .image(let e, _): e.id.uuidString
        case .series(let r): r.name
        }
    }
}

extension AppState {
    /// **The series and the cell an image of the history is** — `nil` for any other, and for what was made from a cell (a
    /// crop, an enlargement: they keep its job).
    func cell(of e: Entry) -> (grid: Int, cell: Int)? {
        e.enlarged == nil && e.cropped == nil ? gridCells[e.job] : nil
    }

    /// The images of a series the history holds now, by cell.
    func cells(of run: GridRun) -> [Int: Entry] {
        var found: [Int: Entry] = [:]
        for e in history {
            if let at = cell(of: e), at.grid == run.ordinal, found[at.cell] == nil { found[at.cell] = e }
        }
        return found
    }

    var shownRun: GridRun? { shownSeries.flatMap { series[$0] } }

    /// **The history column, row by row**: newest first; a series' tile where its newest cell is — or, with none yet,
    /// where its first job falls among the images' —, its cells under it unless folded. A series with no cell left and
    /// none on the way has no tile.
    var strip: [StripRow] {
        guard !series.isEmpty else { return history.map { .image($0, inSeries: false) } }
        var byGrid: [Int: [Entry]] = [:]
        for e in history { if let at = cell(of: e) { byGrid[at.grid, default: []].append(e) } }
        var waiting = series.values.filter { byGrid[$0.ordinal] == nil && !$0.isComplete }.sorted { $0.ordinal > $1.ordinal }
        var rows: [StripRow] = []
        var placed = Set<Int>()
        func flush(above job: Int?) {
            while let w = waiting.first, job.map({ $0 < (w.jobs.first ?? 0) }) ?? true {
                rows.append(.series(w))
                waiting.removeFirst()
            }
        }
        for e in history {
            flush(above: e.job)
            if let at = cell(of: e), let run = series[at.grid] {
                guard placed.insert(at.grid).inserted else { continue }
                rows.append(.series(run))
                if !foldedSeries.contains(at.grid) { rows += (byGrid[at.grid] ?? []).map { .image($0, inSeries: true) } }
            } else {
                rows.append(.image(e, inSeries: false))
            }
        }
        flush(above: nil)
        return rows
    }

    /// **The images the arrows walk**, in the column's order: a folded series' cells are skipped.
    var stripImages: [Entry] {
        strip.compactMap { if case .image(let e, _) = $0 { e } else { nil } }
    }

    /// Whether a series still has cells on their way.
    func isRunning(_ run: GridRun) -> Bool { !run.isComplete }

    /// **« Stop »**: the series' cells still waiting leave the queue; the running one stops.
    func stop(_ run: GridRun) {
        for job in file where run.jobs.contains(job.ordinal) { remove(job) }
        if let c = currentJob, run.jobs.contains(c.ordinal) { cancel() }
    }

    func toggleFold(_ run: GridRun) {
        if foldedSeries.remove(run.ordinal) == nil { foldedSeries.insert(run.ordinal) }
    }

    /// **The tile's ✕**: the series leaves the history — its cells, which ⌘Z brings back, and those still waiting, which
    /// leave the queue.
    func removeSeries(_ run: GridRun) {
        stop(run)
        if shownSeries == run.ordinal { shownSeries = nil }
        let ids = Set(cells(of: run).values.map(\.id))
        delete(history.filter { ids.contains($0.id) })
    }

    /// Whether the shown series' cells follow the window: ⌘0 has nothing to give back (nor before its grid is laid out).
    var seriesFollowsWindow: Bool {
        guard let f = seriesFit, f.run == shownSeries else { return true }
        return f.follows(exploreCellSide)
    }

    /// **⌘0 on a series' grid**: its cells take the size that fills the canvas, and follow the window again.
    func fitSeries() {
        guard var f = seriesFit, f.run == shownSeries else { return }
        exploreCellSide = f.restore()
        seriesFit = f
    }

    /// What a column or row says, for a series.
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
    /// The room the cells have (how they were last sized to fill it: `AppState.seriesFit`).
    @State private var area: CGSize = .zero

    var body: some View {
        let images = app.cells(of: run)
        VStack(spacing: 0) {
            ScrollView([.vertical, .horizontal]) {
                cells(images).padding(SeriesFit.margin)
            }
            // **A series fills the room**: every cell in sight, as large as they can be, the window larger or smaller.
            // Once the hand sizes them, they keep its size; another series fits again, and ⌘0 (`AppState.fitSeries`).
            .onGeometryChange(for: CGSize.self) { $0.size } action: { size in
                area = size
                fit()
            }
            .onChange(of: run.ordinal) { chosen = nil; fit() }  // another series: no cell of it was chosen
            .gesture(MagnifyGesture()
                .updating($pinch) { value, state, _ in state = value.magnification }
                .onEnded { value in app.exploreCellSide = AppState.clampedCellSide(app.exploreCellSide * value.magnification) })
            // The chosen cell — or, while the series runs, the one rendering, live (as `showsBand` reads it).
            if let i = chosen, run.cellJobs.indices.contains(i), let e = images[i] {
                Divider()
                detail(i, e)
            } else if let i = running {
                Divider()
                detail(i, nil)
            }
        }
    }

    /// The cells sized to the room — unless the hand sized them, or only the band under the grid came or went
    /// (`SeriesFit.refit`).
    private func fit() {
        guard area.width > 0, area.height > 0 else { return }
        let g = run.grid, f = run.settings.format
        let fitting = SeriesFit.side(in: area, columns: g.columns, rows: g.rows, aspect: CGFloat(f.width) / CGFloat(f.height),
                                     rowLabels: !app.exploreLabels(g.y, of: run).isEmpty,
                                     columnLabels: !app.exploreLabels(g.x, of: run).isEmpty)
        let (fit, side) = SeriesFit.refit(app.seriesFit, run: run.ordinal, band: showsBand, current: app.exploreCellSide,
                                          fitting: fitting)
        if let side { app.exploreCellSide = side }
        if app.seriesFit != fit { app.seriesFit = fit }
    }

    /// Whether the band under the grid shows: a chosen cell's settings, or the running one's.
    private var showsBand: Bool {
        if let i = chosen, run.cellJobs.indices.contains(i), app.cells(of: run)[i] != nil { return true }
        return running != nil
    }

    /// The cell rendering now, if it is one of this grid's.
    private var running: Int? { app.currentJob.flatMap { run.jobs.firstIndex(of: $0.ordinal) } }

    private var side: CGFloat { AppState.clampedCellSide(app.exploreCellSide * pinch) }

    private func cells(_ images: [Int: Entry]) -> some View {
        let g = run.grid
        let columns = app.exploreLabels(g.x, of: run), rows = app.exploreLabels(g.y, of: run)
        let f = run.settings.format
        let (w, h) = f.width >= f.height ? (side, side * CGFloat(f.height) / CGFloat(f.width))
                                         : (side * CGFloat(f.width) / CGFloat(f.height), side)
        return Grid(horizontalSpacing: SeriesFit.gap, verticalSpacing: SeriesFit.gap) {
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
                            .frame(width: SeriesFit.rowLabelWidth, alignment: .trailing)
                    }
                    ForEach(0..<g.columns, id: \.self) { c in
                        cell(r * g.columns + c, images[r * g.columns + c]).frame(width: w, height: h)
                    }
                }
            }
        }
    }

    @ViewBuilder private func cell(_ i: Int, _ entry: Entry?) -> some View {
        let job = run.jobs[i]
        ZStack {
            if let e = entry {
                CellImage(entry: e, side: side)
            } else if let p = app.prerenders[job] {
                // Its own last pre-render, kept in its place until its image comes — even once the next cell runs.
                Image(decorative: p, scale: 1).resizable().interpolation(.medium).opacity(0.85)
                if app.currentJob?.ordinal == job { ProgressView().controlSize(.small) }
            } else if app.currentJob?.ordinal == job {
                Rectangle().fill(.quaternary)
                ProgressView().controlSize(.small)
            } else {
                Rectangle().fill(Palette.well)
                Text(note(i))
                    .font(.caption).foregroundStyle(Palette.secondary)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .overlay(RoundedRectangle(cornerRadius: 4)
            .strokeBorder(chosen == i ? Palette.live : running == i ? Palette.live : .clear, lineWidth: 3))
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { if let entry { app.show(entry.id) } }
        .onTapGesture { chosen = entry == nil ? nil : i }
        .contextMenu {
            if let e = entry {
                Button("Show This Image") { app.show(e.id) }
                Button("Reuse These Settings") { app.restoreSettings(e) }
            }
        }
        .help(run.cellJobs.indices.contains(i)
              ? run.cellJobs[i].settings.prompt + " · " + String(localized: "seed \(String(run.cellJobs[i].seeds[0]))") : "")
    }

    /// Why a cell has no image: on its way, or how it ended — a cell rendered then taken out of the history is removed.
    private func note(_ i: Int) -> String {
        switch run.issues[i] {
        case nil: String(localized: "Waiting")
        case .finished?: GridWords.note(.removed)
        case let issue: GridWords.note(issue)
        }
    }

    /// The chosen cell's settings, with its image (`entry`) — or, without, the cell rendering now.
    private func detail(_ i: Int, _ entry: Entry?) -> some View {
        let live = entry == nil
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
                    .font(.caption).foregroundStyle(Palette.secondary)
            }
            Spacer()
            if let e = entry {
                Button("Reuse These Settings") { app.restoreSettings(e) }
                    .help("The rack takes this image’s settings — model, prompt, size, LoRAs, seed — without rendering")
                Button("Show This Image") { app.show(e.id) }
                    .buttonStyle(.prominent)
                    .help("The image, as the history shows it (double-click a cell)")
            }
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
    }
}

/// **The grid's controls, in the window's top band**, at the top of the window as the grid is the canvas: what the series is,
/// how many cells came, their size, « Stop » while it runs, the sheet's export, and the way back to the image.
struct SeriesBar: View {
    @Bindable var app: AppState
    let run: GridRun

    var body: some View {
        let count = app.cells(of: run).count
        HStack(spacing: 12) {
            Image(systemName: "square.grid.3x3").foregroundStyle(Palette.bracket)
            Text(run.grid.title).font(.callout.weight(.semibold)).lineLimit(1).truncationMode(.middle)
                .frame(maxWidth: 280, alignment: .leading)
            Text("\(count) of \(run.jobs.count)").font(.callout).monospacedDigit().foregroundStyle(Palette.secondary)
            HStack(spacing: 4) {
                Image(systemName: "photo").imageScale(.small)
                Slider(value: $app.exploreCellSide, in: AppState.cellSides).frame(width: 110).controlSize(.small).tint(Palette.control)
                Image(systemName: "photo").imageScale(.large)
            }
            .foregroundStyle(Palette.secondary)
            .help("Cell size — or pinch, ⌘+ and ⌘−")
            if app.isRunning(run) {
                Button("Stop") { app.stop(run) }
                    .help("The cells still waiting leave the queue; the running one stops")
            }
            Button("Export Sheet…") { app.exportSheet(run) }
                .help("The images side by side, labelled, as a PNG file")
                .unavailable(count == 0 ? String(localized: "No cell has its image yet.") : nil)
            Button { app.shownSeries = nil } label: { Image(systemName: "xmark") }
                .buttonStyle(.borderless)
                .help("Back to the image — the series stays in the history, under its tile")
                .accessibilityLabel(Text("Back to the Image"))
        }
        .controlSize(.small)
        .padding(.horizontal, 10)
        .fixedSize()
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
