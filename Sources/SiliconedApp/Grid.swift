// **The XY grid in the app: the form's two axes, the cells in the queue, the sheet.**
//
//     the exploration  X and Y: a group of the prompt, the seed, the steps, a LoRA's strength, the
//     window           model, the edit's images — and their values (`GridAxisForm`, `Exploration.swift`)
//        │ « Explore » (or `silicontrol grid`)
//        ▼
//     RenderGrid       the library checks it and lists its cells, row by row (`RenderGrid.swift`)
//        │ `AppState.enqueueGrid`: one ordinary job per cell, the run remembered (`GridRun`)
//        ▼
//     the queue        each cell renders like any other job — a sketch from the window (it stops where
//                      the image is already readable, `Sketch`, and stays in the window), a finished
//                      image from `silicontrol grid` (it joins the history)
//        │ the last cell ends (done, failed, stopped or removed)
//        ▼
//     the sheet        drawn off the main thread (`SheetDrawing`): title, labels, each cell reduced to
//                      512 px at most — an entry of the history, saved, copied, dragged like the others;
//                      for an exploration, only when asked (« Add Sheet to History »)
//
// Choices an analogy with ComfyUI's « XY plot » would not make:
//
// - **No second queue, no engine path**: a cell is a `Job`, its image an `Entry`. The grid only
//   remembers which job is which cell, to lay their images out when the last one ends.
// - **A cell that did not render is drawn empty, with why** (failed, stopped, removed): the sheet still
//   compares what did render. A grid of which no cell rendered makes no sheet.
// - **A group of the prompt that is not an axis is refused** (`RenderGrid`): one image per cell.
// - **The batch does not apply**: a cell is one image; the seed axis is how a grid varies seeds.
// - **Incognito**: the sheet is an image of the history like the others — in memory, written only by
//   « Save », « Copy », a drag or `silicontrol save`. Its cells' PNGs are kept by the run only until
//   the sheet is drawn.

import AppKit
import CoreText
import ImageIO
import Siliconed
import SwiftUI

// MARK: - The form

/// **One axis as the form sets it**: what it varies, and the values typed.
struct GridAxisForm: Equatable {
    enum Kind: Hashable {
        /// No Y axis: one row.
        case none
        /// The `n`-th `{…}` of the prompt (from 0).
        case prompt(Int)
        case seed
        case steps
        /// The strength of this LoRA of the stack.
        case lora(LoRASlot.ID)
        /// The models ticked in `models`.
        case model
        /// The LoRAs ticked in `added` (and none, if `addedNone`), one per cell, at `addedStrength`.
        case addedLoRA
        /// Each image of the edit, alone.
        case image
    }
    var kind: Kind
    /// The seed axis: `seedCount` seeds from the form's seed, or the list typed.
    var listSeeds = false
    var seedCount = 4
    var seeds = ""
    var steps = "4, 8, 12"
    var strengths = ""
    var models: [String] = []
    var added: [String] = []
    var addedNone = true
    var addedStrength = 0.8

    /// **Integers separated by commas, semicolons or spaces** — `nil` when one does not read.
    static func integers<T: FixedWidthInteger>(_ text: String) -> [T]? {
        let tokens = text.components(separatedBy: CharacterSet(charactersIn: ",; \t\n")).filter { !$0.isEmpty }
        let values = tokens.compactMap { T($0) }
        return values.count == tokens.count ? values : nil
    }

    /// **Decimals in the user's language**: `0.4, 0.8` in English, `0,4 ; 0,8` (or `0,4 0,8`) in French —
    /// where the comma is the decimal separator, it separates nothing.
    static func decimals(_ text: String) -> [Double]? {
        let comma = Locale.current.decimalSeparator == ","
        let tokens = text.components(separatedBy: CharacterSet(charactersIn: comma ? "; \t\n" : ",; \t\n")).filter { !$0.isEmpty }
        let values = tokens.compactMap { Double(comma ? $0.replacingOccurrences(of: ",", with: ".") : $0) }
        return values.count == tokens.count ? values : nil
    }

    /// Strengths written back as the user types them.
    static func written(_ values: [Double]) -> String {
        let comma = Locale.current.decimalSeparator == ","
        return values.map(Figures.strength).joined(separator: comma ? " ; " : ", ")
    }

    /// The axis the library checks, or why not: `stack` is the LoRA stack the render will take.
    func axis(stack: [LoRASlot]) throws(EngineError) -> RenderGrid.Axis? {
        switch kind {
        case .none: return nil
        case .prompt(let g): return .prompt(group: g)
        case .seed:
            if !listSeeds { return .consecutiveSeeds(count: seedCount) }
            guard let s: [UInt64] = Self.integers(seeds) else { throw .gridRefused(reason: .unreadableAxis) }
            return .seeds(s)
        case .steps:
            guard let s: [Int] = Self.integers(steps) else { throw .gridRefused(reason: .unreadableAxis) }
            return .steps(s)
        case .lora(let id):
            guard let slot = stack.firstIndex(where: { $0.id == id }) else { throw .gridRefused(reason: .noSuchLoRA) }
            // Left empty, the strengths the field shows as its placeholder: what is seen is what renders.
            let typed = strengths.trimmingCharacters(in: .whitespaces)
            guard let v = typed.isEmpty ? [0.4, 0.6, 0.8, 1] : Self.decimals(typed) else { throw .gridRefused(reason: .unreadableAxis) }
            return .loraStrength(slot: slot, values: v)
        case .model: return .models(models)
        case .addedLoRA: return .addedLoRAs((addedNone ? [""] : []) + added, strength: addedStrength)
        case .image: return .images
        }
    }

    /// The form of an axis the library describes — a sheet's « Reuse These Settings », a remote grid.
    init(_ axis: RenderGrid.Axis?, stack: [LoRASlot]) {
        self.init(kind: .none)
        switch axis {
        case nil: break
        case .prompt(let g)?: kind = .prompt(g)
        case .seeds(let s)?: kind = .seed; listSeeds = true; seeds = s.map(String.init).joined(separator: ", ")
        case .consecutiveSeeds(let n)?: kind = .seed; seedCount = n
        case .steps(let s)?: kind = .steps; steps = s.map(String.init).joined(separator: ", ")
        case let .loraStrength(slot, values)?:
            kind = stack.indices.contains(slot) ? .lora(stack[slot].id) : .none
            strengths = Self.written(values)
        case .models(let m)?: kind = .model; models = m
        case let .addedLoRAs(l, f)?: kind = .addedLoRA; addedNone = l.contains(""); added = l.filter { !$0.isEmpty }; addedStrength = f
        case .images?: kind = .image
        }
    }

    init(kind: Kind) { self.kind = kind }
}

// MARK: - A grid on its way

/// **A grid in the queue**: its cells' jobs, what each became, and what its sheet will say. In memory,
/// until its sheet is drawn.
struct GridRun {
    let ordinal: Int
    let grid: RenderGrid
    /// What the cells share: the prompt as typed (its groups included), the format, the stack.
    let settings: RenderSettings
    /// The job of each cell, in the grid's order (`RenderGrid.cells`).
    let jobs: [Int]
    /// The same jobs whole: what « Finish This Image » and « Use These Settings » start from.
    var cellJobs: [Job] = []
    let references: Int
    /// An exploration: its cells are sketches (`Job.sketch`), kept here and not in the history.
    var sketch = false
    var entries: [Int: Entry] = [:]
    var issues: [Int: Issue] = [:]
    var name: String { "g\(ordinal)" }
    var isComplete: Bool { issues.count == jobs.count }
}

/// What a sheet keeps of its grid: enough to label it, export it, and set the form again.
struct GridInfo {
    let ordinal: Int
    let grid: RenderGrid
    /// How many cells did not render.
    let empty: Int
    var name: String { "g\(ordinal)" }
}

// MARK: - The words of a sheet

/// **The labels, in the user's language** — computed on the main thread, drawn off it.
enum GridWords {
    /// What an axis varies, as the form's menu and the sheet say it.
    static func name(_ axis: RenderGrid.Axis, grid: RenderGrid, stack: [LoRASlot], loraName: (String) -> String) -> String {
        switch axis {
        case .prompt(let g):
            return grid.alternatives.groups.count > 1 ? String(localized: "prompt, group \(g + 1)") : String(localized: "prompt")
        case .seeds, .consecutiveSeeds: return String(localized: "seed")
        case .steps: return String(localized: "steps")
        case .loraStrength(let slot, _):
            return stack.indices.contains(slot) ? loraName(stack[slot].path) : "LoRA"
        case .models: return String(localized: "model")
        case .addedLoRAs: return "LoRA"
        case .images: return String(localized: "image to edit")
        }
    }

    /// One value, as a column or row says it. `modelName`: a model's name from its identifier.
    /// `addedName`: an added LoRA's name from its path.
    static func label(_ v: RenderGrid.Value, loraName: String, modelName: (String) -> String = { $0 },
                      addedName: (String) -> String = { $0 }) -> String {
        switch v {
        case .alternative(let a):
            let t = a.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? String(localized: "(nothing)") : t
        case .seed(let s): return String(localized: "seed \(String(s))")
        case .steps(let n): return String(localized: "\(n) steps")
        case .strength(let f): return "\(loraName) \(Figures.strength(f))"
        case .model(let m): return modelName(m)
        case .addedLoRA(let l): return l.isEmpty ? String(localized: "no LoRA") : addedName(l)
        case .image(let i): return String(localized: "image \(i + 1)")
        }
    }

    /// An axis as `silicontrol grid` reads it (`RenderGrid.axis`): decimals with a point, whatever
    /// the language — it is a syntax. `loras`: the stack's short names, as `--lora` takes them.
    static func command(_ axis: RenderGrid.Axis, loras: [String]) -> String {
        switch axis {
        case .prompt(let g): return g == 0 ? "prompt" : "prompt:\(g + 1)"
        case .seeds(let s): return "seed=" + s.map(String.init).joined(separator: ",")
        case .consecutiveSeeds(let n): return "seeds=\(n)"
        case .steps(let s): return "steps=" + s.map(String.init).joined(separator: ",")
        case let .loraStrength(slot, values):
            let v = values.map { String(format: "%.2f", $0) }.joined(separator: ",")
            return (slot == 0 || !loras.indices.contains(slot) ? "lora" : "lora:\(loras[slot])") + "=" + v
        case .models(let m): return "model=" + m.joined(separator: ",")
        case let .addedLoRAs(l, f):
            return "loras:" + String(format: "%.2f", f) + "=" + l.map { $0.isEmpty ? "none" : RemoteControl.shortName($0) }.joined(separator: ",")
        case .images: return "image"
        }
    }

    /// Why a cell is empty.
    static func note(_ issue: Issue?) -> String {
        switch issue {
        case .failed?: String(localized: "Failed")
        case .stopped?: String(localized: "Stopped")
        case .removed?: String(localized: "Removed")
        case .finished?, nil: String(localized: "No image")
        }
    }
}

// MARK: - The sheet, drawn

/// **What the sheet is drawn from** — values only, so the drawing runs off the main thread.
struct SheetDrawing: Sendable {
    struct Cell: Sendable {
        let column: Int, row: Int
        /// The cell's image, or `nil` and why not.
        let png: Data?
        let note: String
    }
    let layout: RenderGrid.Sheet
    let cells: [Cell]
    let title: String
    let subtitle: String
    let columnLabels: [String]
    let rowLabels: [String]
    let metadata: [String: String]

    /// The sheet's PNG and its thumbnail. One cell decoded at a time, straight to its size on the sheet
    /// (ImageIO's thumbnail): a 1024² cell never lives at full size here.
    func draw() throws -> (png: Data, vignette: CGImage) {
        let w = layout.width, h = layout.height
        guard let c = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw EngineError.imageEncodingFailed
        }
        // A contact sheet: white paper, dark ink — it reads the same in any app it lands in.
        c.setFillColor(CGColor(gray: 1, alpha: 1))
        c.fill(CGRect(x: 0, y: 0, width: w, height: h))
        c.interpolationQuality = .high
        let ink = CGColor(gray: 0.1, alpha: 1), quiet = CGColor(gray: 0.45, alpha: 1)

        let side = max(layout.cellWidth, layout.cellHeight)
        for cell in cells {
            let r = flipped(layout.cell(column: cell.column, row: cell.row))
            if let png = cell.png, let image = Self.reduced(png, side: side) {
                c.draw(image, in: r)
            } else {
                c.setFillColor(CGColor(gray: 0.93, alpha: 1))
                c.fill(r)
                text(cell.note, in: layout.cell(column: cell.column, row: cell.row), size: layout.labelSize,
                     color: quiet, center: true, wrap: true, c)
            }
        }
        for (i, label) in columnLabels.enumerated() where i < layout.columns {
            text(label, in: layout.columnLabel(i), size: layout.labelSize, weight: true, color: ink, center: true, wrap: false, c)
        }
        for (i, label) in rowLabels.enumerated() where i < layout.rows && layout.rowLabelWidth > 0 {
            text(label, in: layout.rowLabel(i), size: layout.labelSize, weight: true, color: ink, center: false, wrap: true, c)
        }
        // The title on top (two lines at most), what the cells share right under it.
        let band = layout.title
        let foot = CGFloat(layout.labelSize) * 1.6
        let used = text(title, in: CGRect(x: band.minX, y: band.minY, width: band.width, height: band.height - foot),
                        size: layout.titleSize, weight: true, color: ink, center: false, wrap: true, top: true, c)
        text(subtitle, in: CGRect(x: band.minX, y: band.minY + used + CGFloat(layout.gap) / 2, width: band.width, height: foot),
             size: layout.labelSize, color: quiet, center: false, wrap: false, c)

        guard let image = c.makeImage() else { throw EngineError.imageEncodingFailed }
        let png = try PNG.data(image, metadata: metadata)
        return (png, FinishedImage.reduce(image, side: 240) ?? image)
    }

    /// The top-left rectangles of `RenderGrid.Sheet`, in Core Graphics' bottom-left space.
    private func flipped(_ r: CGRect) -> CGRect {
        CGRect(x: r.minX, y: CGFloat(layout.height) - r.maxY, width: r.width, height: r.height)
    }

    /// A PNG decoded straight to `side` pixels on its long side.
    private static func reduced(_ png: Data, side: Int) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(png as CFData, nil) else { return nil }
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                        kCGImageSourceThumbnailMaxPixelSize: side,
                                        kCGImageSourceCreateThumbnailWithTransform: true]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    /// **Text in a rectangle** (top-left coordinates), with Core Text — safe off the main thread.
    /// One line truncated with an ellipsis, or wrapped lines (those that do not fit are left out);
    /// centered vertically, or from the top. Returns the height the text took.
    @discardableResult
    private func text(_ s: String, in rect: CGRect, size: Int, weight: Bool = false, color: CGColor,
                      center: Bool, wrap: Bool, top: Bool = false, _ c: CGContext) -> CGFloat {
        guard !s.isEmpty, rect.width > 0, rect.height > 0,
              let font = CTFontCreateUIFontForLanguage(weight ? .emphasizedSystem : .system, CGFloat(size), nil) else { return 0 }
        var alignment: CTTextAlignment = center ? .center : .left
        let paragraph = withUnsafeBytes(of: &alignment) { raw in
            var setting = CTParagraphStyleSetting(spec: .alignment, valueSize: MemoryLayout<CTTextAlignment>.size,
                                                  value: raw.baseAddress!)
            return CTParagraphStyleCreate(&setting, 1)
        }
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
            NSAttributedString.Key(kCTParagraphStyleAttributeName as String): paragraph,
        ]
        let string = NSAttributedString(string: s, attributes: attributes)
        let r = flipped(rect)
        c.saveGState()
        defer { c.restoreGState() }
        c.textMatrix = .identity
        if wrap {
            let setter = CTFramesetterCreateWithAttributedString(string)
            let fitted = CTFramesetterSuggestFrameSizeWithConstraints(setter, CFRange(), nil, r.size, nil)
            let height = min(r.height, ceil(fitted.height))
            let y = top ? r.maxY - height : r.minY + (r.height - height) / 2
            let frame = CTFramesetterCreateFrame(setter, CFRange(), CGPath(rect: CGRect(x: r.minX, y: y, width: r.width, height: height), transform: nil), nil)
            CTFrameDraw(frame, c)
            return height
        } else {
            let full = CTLineCreateWithAttributedString(string)
            let ellipsis = CTLineCreateWithAttributedString(NSAttributedString(string: "…", attributes: attributes))
            let line = CTLineCreateTruncatedLine(full, Double(r.width), .end, ellipsis) ?? full
            var ascent: CGFloat = 0, descent: CGFloat = 0, leading: CGFloat = 0
            let width = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, &leading))
            let x = center ? r.minX + (r.width - width) / 2 : r.minX
            c.textPosition = CGPoint(x: x, y: r.minY + (r.height - ascent - descent) / 2 + descent)
            CTLineDraw(line, c)
            return ascent + descent
        }
    }
}

// MARK: - An axis in the form

/// One axis: what it varies, then its values.
struct AxisRow: View {
    @Bindable var app: AppState
    let title: String
    @Binding var form: GridAxisForm
    let allowsNone: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(verbatim: title).font(.callout.weight(.semibold)).foregroundStyle(.secondary).frame(width: 14)
                Picker("Axis \(title)", selection: $form.kind) {
                    if allowsNone { Text("None").tag(GridAxisForm.Kind.none) }
                    ForEach(Array(app.promptGroups.enumerated()), id: \.offset) { g, options in
                        Text(verbatim: "{" + options.joined(separator: "|") + "}").tag(GridAxisForm.Kind.prompt(g))
                    }
                    Text("Seed").tag(GridAxisForm.Kind.seed)
                    Text("Steps").tag(GridAxisForm.Kind.steps)
                    ForEach(app.loras.filter { !$0.path.isEmpty }) { slot in
                        Text("LoRA \(app.loraName(slot.path))").tag(GridAxisForm.Kind.lora(slot.id))
                    }
                    if !app.compatibleLoras.isEmpty { Text("LoRA to Compare").tag(GridAxisForm.Kind.addedLoRA) }
                    Text("Model").tag(GridAxisForm.Kind.model)
                    if !app.references.isEmpty { Text("Image to Edit").tag(GridAxisForm.Kind.image) }
                }
                .labelsHidden()
                .help("What varies from one column (X) or row (Y) to the next")
            }
            values.padding(.leading, 20)
        }
    }

    @ViewBuilder private var values: some View {
        switch form.kind {
        case .none:
            EmptyView()
        case .prompt(let g):
            Text(app.promptGroups.indices.contains(g) ? app.promptGroups[g].map { $0.isEmpty ? "∅" : $0 }.joined(separator: " · ") : "—")
                .font(.caption).foregroundStyle(.secondary).lineLimit(2)
        case .seed:
            HStack(spacing: 6) {
                Picker("Seeds", selection: $form.listSeeds) {
                    Text("Consecutive").tag(false)
                    Text("List").tag(true)
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
                .help("The seeds that follow the rack's seed, or the seeds you list")
                if form.listSeeds {
                    TextField("Seeds", text: $form.seeds, prompt: Text(verbatim: "42, 7, 1000"))
                        .labelsHidden().textFieldStyle(.roundedBorder)
                } else {
                    Stepper(value: $form.seedCount, in: 1...RenderGrid.maxCells) {
                        Text("\(form.seedCount) from the seed").monospacedDigit()
                    }
                    .fixedSize()
                }
            }
            .font(.caption)
        case .steps:
            TextField("Steps", text: $form.steps, prompt: Text(verbatim: "4, 8, 12"))
                .labelsHidden().textFieldStyle(.roundedBorder).font(.caption)
                .help("Step counts, separated by commas")
        case .lora:
            TextField("Strengths", text: $form.strengths, prompt: Text(verbatim: GridAxisForm.written([0.4, 0.6, 0.8, 1])))
                .labelsHidden().textFieldStyle(.roundedBorder).font(.caption)
                .help("Strengths for this LoRA; the other LoRAs keep theirs")
        case .model:
            // The models ready and shown, ticked in the order they are listed; each at its own steps.
            VStack(alignment: .leading, spacing: 2) {
                ForEach(app.visibleReady, id: \.id) { card in
                    Toggle(card.name, isOn: Binding(
                        get: { form.models.contains(card.id) },
                        set: { on in
                            if on { form.models = app.visibleReady.map(\.id).filter { form.models.contains($0) || $0 == card.id } }
                            else { form.models.removeAll { $0 == card.id } }
                        }))
                }
            }
            .font(.caption)
            .help("Each model at its own number of steps; the LoRAs are left out, each is made for one model")
        case .addedLoRA:
            // One LoRA per cell, added to the stack — and the cell without, to see what it changes.
            VStack(alignment: .leading, spacing: 2) {
                Toggle("No LoRA", isOn: $form.addedNone)
                ForEach(app.compatibleLoras) { card in
                    Toggle(card.name, isOn: Binding(
                        get: { form.added.contains(card.path) },
                        set: { on in
                            if on { form.added = app.compatibleLoras.map(\.path).filter { form.added.contains($0) || $0 == card.path } }
                            else { form.added.removeAll { $0 == card.path } }
                        }))
                }
                HStack {
                    Slider(value: $form.addedStrength, in: 0...1.5)
                    Text(Figures.strength(form.addedStrength)).monospacedDigit().frame(width: 38, alignment: .trailing)
                }
                .help("The strength of each LoRA compared")
            }
            .font(.caption)
            .help("Each cell adds one of these LoRAs to the stack, at one strength")
        case .image:
            Text("Each of the \(app.references.count) images alone, with the same instruction")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
