// **The exploration: a quick sweep of settings — driven from a floating panel, shown in the canvas.**
//
//     the panel    a prompt of its own (`{a|b}` groups make axes), the rack's LoRA stack, a small
//                  format, X and Y — seeds, a LoRA's strength, LoRAs to compare, a group of the prompt,
//                  steps, models, the edit's images (`ExplorationPanel`, ⌥⌘G)
//        │ « Explore » (⌘↩ in the panel)
//        ▼
//     the queue    one job per cell (`AppState.enqueueGrid(sketch: true)`), each stopped as a sketch
//                  where its image is already readable (`Sketch`: σ ≤ 0.8 — 3 evaluations of 7 for
//                  Z-Image; Qwen-Image-2.1 does not sketch, its cells are finished images), its picture
//                  the model's estimate x̂₀ — or
//                  later, the panel says (`exploreStop`): under a LoRA the style can settle late,
//                  and at the plan's count the cells are finished images
//        │
//        ▼
//     the canvas   the main window shows the grid (`ExplorationCanvas`): cells fill in as they come,
//                  the running one shows its preview and, under the grid, its settings live; it
//                  scrolls, and pinch, ⌘+ ⌘− or the slider size the cells. The grid waits at the head
//                  of the history strip when an image is shown.
//        │ a cell
//        ▼
//     « Finish This Image » (double-click): the rack takes the cell whole — model, prompt, size,
//                  steps, LoRAs, edit images, seed — then renders it to its end: the sketch's first
//                  evaluations are the finished image's, bit for bit, so it is the image announced.
//     « Use These Settings »: the same, without rendering.
//
// What an analogy with a contact sheet would miss:
//
// - **Small means 512, never less** (`Format.minimumSide`: the models are out of their domain under
//   it). The speed comes from stopping early, not from the size: a Z-Image cell costs ~17 s instead of
//   ~33 s at 512², text included (measured; 6 cells in 98 s through the app). Portrait and landscape are
//   512 on their short side.
// - **A sketch is not an image of the history**: the grid keeps them until the next exploration;
//   « Add Sheet to History » makes the sheet an image like the others (incognito: in memory).
// - **A new exploration replaces the last**: its cells still waiting leave the queue, the running one
//   stops — exploring is trying, not stacking up.
// - **The rack's model, LoRAs, seed and edit images are the base**: the panel varies them, it does not
//   duplicate them. A model axis drops the LoRAs (each is made for one model).

import AppKit
import Siliconed
import SwiftUI

/// The exploration's format: 512 on the short side.
enum ExploreShape: String, CaseIterable, Identifiable {
    case square, portrait, landscape
    var id: String { rawValue }
    var format: RecommendedFormat {
        switch self {
        case .square: RecommendedFormat(512, 512)
        case .portrait: RecommendedFormat(512, 768)
        case .landscape: RecommendedFormat(768, 512)
        }
    }
}

// MARK: - The state

extension AppState {
    var exploreFormat: RecommendedFormat { exploreShape.format }

    /// Whether the exploration varies the model — then the rack's LoRAs are left out.
    var exploresModels: Bool { [gridX, gridY].contains { $0.kind == .model } }

    /// **What the cells' renders would be**: each model read once per catalog (`AppState.model`).
    private func cellPlans(_ g: RenderGrid) -> [(model: String, steps: Int, plan: DenoisingPlan?)] {
        let format = exploreFormat, stack = !exploresModels && !activeLoRAs.isEmpty
        return g.cells.map { cell in
            let withLoRA = stack || cell.addedLoRA != nil
            let id = cell.model ?? identifier
            let steps = cell.steps ?? cards.first { $0.id == id }?.defaultSteps ?? self.steps
            guard let m = model(id) else { return (id, steps, nil) }
            let d = m.chain.denoising, f = d.space.factor
            let plan = d.plan(height: format.height / f, width: format.width / f, steps: steps, start: 0,
                              withLoRA: withLoRA || detail != .normal)
            let stop = exploreStop > 0 ? exploreStop : m.sketchEvaluations(steps: steps, width: format.width, height: format.height)
            return (id, steps, plan.sketched(stop))
        }
    }

    /// Why « Explore » is greyed out, or `nil`.
    var exploreBlocker: String? {
        if let reason = installBlocker { return reason }
        if !exploresModels, missing != nil { return String(localized: "This model isn't installed.") }
        if explorePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return String(localized: "Write a prompt.") }
        let g: RenderGrid
        switch grid {
        case .failure(let e): return Problem(e).text
        case .success(let ok): g = ok
        }
        let waiting = file.filter { j in explored?.jobs.contains(j.ordinal) != true }.count
        if waiting + g.count > Self.maxQueueLength {
            let room = Self.maxQueueLength - waiting
            return String(localized: "This grid makes \(String(g.count)) renders: the queue has room for \(String(room)).")
        }
        for (id, steps, _) in cellPlans(g) {
            guard let m = model(id) else {
                return String(localized: "\(modelName(id)) isn't installed.")
            }
            if (try? m.chain.denoising.check(steps: steps, startImage: false)) == nil {
                return String(localized: "\(modelName(id)) has no schedule for \(String(steps)) steps.")
            }
        }
        return nil
    }

    /// **The exploration's cost**, learned on this Mac — `nil` until each model has rendered at this size.
    var exploreCost: Double? {
        guard let g = try? grid.get() else { return nil }
        let stack = !exploresModels && !activeLoRAs.isEmpty
        var total = 0.0
        for (cell, c) in zip(g.cells, cellPlans(g)) {
            let withLoRA = stack || cell.addedLoRA != nil
            let refs = cell.image != nil ? 1 : references.count
            guard let plan = c.plan, let v = speeds[Speed.key(c.model, exploreFormat, references: refs, lora: withLoRA)] else { return nil }
            total += v.duration(plan, images: 1)
        }
        return total
    }

    /// Whether the exploration shown still has cells on their way.
    var exploring: Bool {
        guard let r = explored else { return false }
        return r.jobs.contains { j in currentJob?.ordinal == j || file.contains { $0.ordinal == j } }
    }

    /// **Explores**: the last exploration's cells still waiting leave the queue, the running one stops,
    /// then one sketch per cell. `seed`: a sheet's, when it is redone; otherwise the rack's rule.
    func explore(seed imposed: UInt64? = nil) {
        guard exploreBlocker == nil else { return }
        stopExploring()
        let base = imposed ?? (fixedSeed ? seed : UInt64.random(in: 0...UInt64(UInt32.max)))
        guard let g = try? grid(seed: base).get() else { return }
        let settings = RenderSettings(identifier: identifier, modelName: card?.name ?? identifier, prompt: explorePrompt,
                                      format: exploreFormat, steps: steps, loras: exploresModels ? [] : activeLoRAs,
                                      references: references.count, detail: detail)
        followsRender = false
        enqueueGrid(g, settings: settings, references: references, previews: true, sketch: true, stopAfter: exploreStop)
        showsExploration = true
    }

    /// The exploration's cells leave the queue; the running one stops.
    func stopExploring() {
        guard let r = explored else { return }
        for job in file where r.jobs.contains(job.ordinal) { remove(job) }
        if let c = currentJob, r.jobs.contains(c.ordinal) { cancel() }
    }

    /// **The rack takes the cell, whole**: model, prompt, size, steps, LoRAs (an added one included),
    /// edit images and seed — the form then says exactly what « Generate » will render.
    func applyCell(_ i: Int) {
        guard let r = explored, r.cellJobs.indices.contains(i) else { return }
        let job = r.cellJobs[i]
        restoreSettings(job.settings)
        references = job.references
        seed = job.seeds[0]
        fixedSeed = true
        // A cell is never a variation (`explore` passes none): a « Reuse » made before must not turn
        // the finished image — `seed`'s didSet does not fire when the seed is the same.
        variations = []
    }

    /// **« Use These Settings »**: the rack takes the cell, nothing renders.
    func useCell(_ i: Int) {
        applyCell(i)
        flash(String(localized: "Settings of the cell taken by the rack"))
    }

    /// **« Finish This Image »**: the rack takes the cell, then renders it to its end — same seed, same
    /// size: the image its sketch announced, in the history, the canvas following it.
    func finishCell(_ i: Int) {
        guard let r = explored, r.cellJobs.indices.contains(i) else { return }
        applyCell(i)
        showsExploration = false
        render(imposedFormat: r.cellJobs[i].settings.format)
    }

    /// **The rack's model's plan at the exploration's size**: its evaluations, and where a sketch stops
    /// by itself — what the panel's « Sketch » line counts in.
    var exploreEvaluations: (all: Int, automatic: Int)? {
        guard let d = chain?.denoising else { return nil }
        let f = d.space.factor, format = exploreFormat
        let all = d.plan(height: format.height / f, width: format.width / f, steps: steps, start: 0).evaluations
        return (all, d.sketchEvaluations(steps: steps, height: format.height / f, width: format.width / f))
    }

    /// What a column or row says, for the exploration shown.
    func exploreLabels(_ axis: RenderGrid.Axis?, of run: GridRun) -> [String] {
        guard let axis else { return [] }
        let stack = run.settings.loras
        var name = "LoRA"
        if case .loraStrength(let slot, _) = axis, stack.indices.contains(slot) { name = loraName(stack[slot].path) }
        return run.grid.values(of: axis).map { GridWords.label($0, loraName: name, modelName: modelName, addedName: loraName) }
    }
}

// MARK: - The panel

extension MainWindow {
    /// **The exploration's panel**: floating over the main window, it only drives — the cells show in
    /// the main window's canvas. Kept once opened; it retains nothing (incognito).
    func showExploration() {
        if explorationPanel == nil {
            if app.explorePrompt.isEmpty { app.explorePrompt = app.prompt }
            app.gridDefaults()
            // Not `.nonactivatingPanel`: its prompt is typed in.
            let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 360, height: 720),
                            styleMask: [.titled, .closable, .resizable, .utilityWindow],
                            backing: .buffered, defer: false)
            p.title = String(localized: "Exploration")
            p.isFloatingPanel = true
            p.hidesOnDeactivate = false
            p.isRestorable = false
            p.isReleasedWhenClosed = false
            p.becomesKeyOnlyIfNeeded = false
            let host = NSHostingController(rootView: ExplorationPanel(app: app))
            host.sizingOptions = []
            p.contentViewController = host
            p.setContentSize(NSSize(width: 360, height: 720))
            p.contentMinSize = NSSize(width: 320, height: 520)
            p.center()
            explorationPanel = p
        }
        NSApp.activate()
        explorationPanel?.makeKeyAndOrderFront(nil)
    }
}

/// **What the panel holds**: the form, the cost, « Explore ». The cells are the canvas's.
struct ExplorationPanel: View {
    @Bindable var app: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Prompt").font(.callout.weight(.semibold))
                    TextEditor(text: $app.explorePrompt)
                        .font(.body)
                        .frame(minHeight: 70, maxHeight: 140)
                        .overlay(RoundedRectangle(cornerRadius: 5).stroke(.quaternary))
                    Text("Write {brunette|blonde|red-haired} to make a group an axis.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if !app.exploresModels {
                    // The rack's model, chosen here too: exploring is where one tries another model.
                    Picker("Model", selection: $app.identifier) {
                        ForEach(app.visibleReady) { card in Text(card.name).tag(card.id) }
                    }
                    .font(.callout)
                    .help("The rack's model: changing it here changes it there")
                    loras
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Base").font(.callout.weight(.semibold))
                    Text(base).font(.caption).foregroundStyle(.secondary).lineLimit(3)
                        .help("The rack's LoRAs, seed and images to edit: the exploration varies them")
                }
                depth
                Picker("Format", selection: $app.exploreShape) {
                    Image(systemName: "square").help("512 × 512").tag(ExploreShape.square)
                    Image(systemName: "rectangle.portrait").help("512 × 768").tag(ExploreShape.portrait)
                    Image(systemName: "rectangle").help("768 × 512").tag(ExploreShape.landscape)
                }
                .pickerStyle(.segmented)
                .help("Small and fast: 512 on the short side — the models are out of their domain below")
                VStack(alignment: .leading, spacing: 10) {
                    AxisRow(app: app, title: "X", form: $app.gridX, allowsNone: false)
                    AxisRow(app: app, title: "Y", form: $app.gridY, allowsNone: true)
                }
                .font(.callout)
                footer
            }
            .padding(16)
        }
    }

    /// **Where a cell stops, in one line**: « Sketch at n steps of N » — the rack's N, and the n
    /// steps a cell runs before its picture is taken — or « Finished images, N steps » when n reaches
    /// the end. n counts steps as well as evaluations: the only null step (Z-Image's, σ = 0 → 0) is
    /// the schedule's last, never among a sketch's.
    ///
    /// The line's stepper moves n when the rack's model sketches; when it does not (Qwen-Image-2.1,
    /// `DenoisingModule.sketches`), its cells are finished images and the stepper moves N — a
    /// shallower sketch of it is an image no one can judge, so it is not offered.
    private var depth: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let e = app.exploreEvaluations {
                if app.chain?.denoising.sketches ?? true {
                    Stepper(value: Binding(get: { app.exploreStop == 0 ? e.automatic : min(app.exploreStop, e.all) },
                                           set: { app.exploreStop = $0 >= e.all ? e.all : max(1, $0) }),
                            in: 1...e.all) {
                        Text(sketchLine(e)).monospacedDigit()
                    }
                    .help("A sketch stops early: the later it stops, the closer to the finished image (a LoRA's style can settle late), the longer it takes. The step count is the rack's")
                } else {
                    Stepper(value: $app.steps, in: 1...50) {
                        Text(sketchLine(e)).monospacedDigit()
                    }
                    .help("This model does not sketch: its cells are finished images. The step count is the rack's (a steps axis overrides it)")
                }
                if app.exploreStop != 0 {
                    Button("Automatic") { app.exploreStop = 0 }.buttonStyle(.link).font(.caption)
                }
            }
        }
        .font(.callout)
    }

    private func sketchLine(_ e: (all: Int, automatic: Int)) -> String {
        let n = app.exploreStop == 0 ? e.automatic : min(app.exploreStop, e.all)
        return n >= e.all ? String(localized: "Finished images, \(app.steps) steps")
                          : String(localized: "Sketch at \(n) steps of \(app.steps)")
    }

    /// **The LoRA stack — the rack's own**: one chosen here is the rack's too.
    private var loras: some View {
        VStack(alignment: .leading, spacing: 3) {
            // A trap measured by its first user: filled, this list STACKS its LoRAs on every cell —
            // comparing them is an axis's job (« LoRA to Compare »).
            Text("LoRAs on Every Cell").font(.callout.weight(.semibold))
            Text("Stacked together, the rack's own. To compare LoRAs, choose « LoRA to Compare » on an axis.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ForEach($app.loras) { $slot in LoRARow(app: app, slot: $slot) }
            if app.activeLoRAs.count > 1 {
                Label("\(app.activeLoRAs.count) LoRAs stacked on every cell", systemImage: "square.stack.3d.up")
                    .font(.caption).foregroundStyle(.orange)
            }
            if app.canAddLoRA {
                Button { app.loras.append(LoRASlot()) } label: { Label("LoRA", systemImage: "plus") }
                    .buttonStyle(.borderless).font(.caption)
                    .help("Add a LoRA to the stack (3 at most) — the rack's stack: a LoRA axis then sweeps its strength")
            } else if app.compatibleLoras.isEmpty {
                Text("No LoRA for this model").font(.caption).foregroundStyle(.tertiary)
            }
        }
    }

    private var base: String {
        var parts = app.exploresModels ? [String(localized: "model on an axis")] : []
        parts.append(app.fixedSeed ? String(localized: "seed \(String(app.seed))") : String(localized: "random seed"))
        if !app.references.isEmpty { parts.append(String(localized: "edit, \(app.references.count) images")) }
        return parts.joined(separator: " · ")
    }

    private var footer: some View {
        let blocker = app.exploreBlocker   // once per drawing
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                if let g = try? app.grid.get() {
                    Label("\(g.count) sketches", systemImage: "square.grid.3x3").monospacedDigit()
                }
                Label {
                    if let t = app.exploreCost { Text(verbatim: "≈ " + Figures.duration(t)).monospacedDigit() }
                    else { Text("after one exploration") }
                } icon: { Image(systemName: "clock") }
                .help("Learned on this Mac, model by model at this size")
            }
            .font(.caption).foregroundStyle(.secondary)
            if let why = blocker {
                Text(why).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Button { app.explore() } label: {
                    Text("Explore").fontWeight(.semibold).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(blocker != nil)
                if app.exploring {
                    Button("Stop") { app.stopExploring() }
                        .controlSize(.large)
                        .help("The cells still waiting leave the queue; the running one stops")
                }
            }
            if app.explored != nil, !app.showsExploration {
                Button("Show the Grid") { app.showsExploration = true; MainWindow.shared.show() }
                    .buttonStyle(.link)
            }
        }
    }
}

// MARK: - The grid, in the main window's canvas

/// **The exploration's cells in the canvas**: they fill in as they come, the running one shows its
/// preview; pinch, ⌘+ ⌘− or the slider size them, the canvas scrolls. A click chooses a cell, a
/// double-click finishes it.
struct ExplorationCanvas: View {
    @Bindable var app: AppState
    let run: GridRun
    @State private var chosen: Int?
    @GestureState private var pinch: CGFloat = 1

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().opacity(0.5)
            ScrollView([.vertical, .horizontal]) {
                cells.padding(24)
            }
            .gesture(MagnifyGesture()
                .updating($pinch) { value, state, _ in state = value.magnification }
                .onEnded { value in app.exploreCellSide = AppState.clampedCellSide(app.exploreCellSide * value.magnification) })
            // The chosen cell — or, while the exploration runs, the one rendering, live.
            if let i = chosen, run.cellJobs.indices.contains(i), run.entries[i] != nil {
                Divider().opacity(0.5)
                detail(i)
            } else if let i = running {
                Divider().opacity(0.5)
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
                Slider(value: $app.exploreCellSide, in: AppState.cellSides).frame(width: 110).controlSize(.small)
                Image(systemName: "photo").imageScale(.large)
            }
            .foregroundStyle(.secondary)
            .help("Cell size — or pinch, ⌘+ and ⌘−")
            Button("Add Sheet to History") { app.addSheet(run) }
                .disabled(run.entries.isEmpty || app.exploring)
                .help("The sketches side by side, labelled — an image of the history, in memory like the others")
            Button { MainWindow.shared.showExploration() } label: { Image(systemName: "slider.horizontal.3") }
                .help("The exploration's panel (⌥⌘G)")
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
                Rectangle().fill(.quinary)
                Text(run.issues[i] == nil ? String(localized: "Waiting") : GridWords.note(run.issues[i]))
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 4))
        .overlay(RoundedRectangle(cornerRadius: 4)
            .strokeBorder(chosen == i ? Color.accentColor : running == i ? Palette.live : .clear, lineWidth: 3))
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { if run.entries[i] != nil { app.finishCell(i) } }
        .onTapGesture { chosen = run.entries[i] == nil ? nil : i }
        .contextMenu {
            if run.entries[i] != nil {
                Button("Finish This Image") { app.finishCell(i) }
                Button("Use These Settings") { app.useCell(i) }
            }
        }
        .help(run.cellJobs.indices.contains(i)
              ? run.cellJobs[i].settings.prompt + " · " + String(localized: "seed \(String(run.cellJobs[i].seeds[0]))") : "")
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
                                 String(localized: "seed \(String(job.seeds[0]))")] + loras
                                + (run.entries[i]?.sketch.map { [String(localized: "sketch after \($0) evaluations")] } ?? []))
                        .joined(separator: " · "))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if !live { actions(i) }
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
    }

    @ViewBuilder private func actions(_ i: Int) -> some View {
        HStack(spacing: 10) {
            Button("Use These Settings") { app.useCell(i) }
                .help("The rack takes this cell whole — model, prompt, size, steps, LoRAs, seed — without rendering")
            Button("Finish This Image") { app.finishCell(i) }
                .buttonStyle(.borderedProminent)
                .help("The rack takes this cell, then renders it to its end: the image this sketch announced (double-click a cell)")
        }
    }
}

/// A cell's image: its thumbnail while it is small, its PNG decoded once it is shown larger.
private struct CellImage: View {
    let entry: Entry
    let side: CGFloat
    @State private var full: CGImage?

    var body: some View {
        Image(decorative: side > 240 ? (full ?? entry.vignette) : entry.vignette, scale: 1)
            .resizable().interpolation(.high)
            .task(id: side > 240) { if side > 240, full == nil { full = await Entry.decoded(entry.png).image } }
    }
}
