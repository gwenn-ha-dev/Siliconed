// **The XY grid in the app: the form's two axes, the cells in the queue, the series' tile, its sheet.**
//
//     a series         what « Generate » varies — the text's groups, the lot, the knobs bracketed
//                      (`Series.swift`) — or the axes of `silicontrol grid`
//        │
//        ▼
//     RenderGrid       the library checks it and lists its cells, row by row (`RenderGrid.swift`)
//        │ `AppState.enqueueGrid`: one ordinary job per cell, the run kept (`GridRun`) for the session
//        ▼
//     the queue        each cell renders like any other job, a finished image into the history
//        │
//        ▼
//     the history      the series' tile heads its cells, at its place in time (`SeriesCanvas.swift`); a click
//                      opens its grid on the canvas
//        │ « Export Sheet… », « Copy Sheet », `silicontrol sheet g2 FILE`, `grid --wait --out D`
//        ▼
//     the sheet        drawn off the main thread (`SheetDrawing`): title, labels, each cell reduced to 512 px
//                      at most — a file or the pasteboard, never an image of the history
//
// Choices an analogy with ComfyUI's « XY plot » would not make:
//
// - **No second queue, no engine path**: a cell is a `Job`, its image an `Entry`. The run only
//   remembers which job is which cell; the images are the history's, found there by their job.
// - **A cell that did not render is drawn empty, with why** (failed, stopped, removed — from the history
//   too): the sheet still compares what did render. A series of which no cell is left draws no sheet.
// - **A group of the prompt that is not an axis is refused** (`RenderGrid`): one image per cell.
// - **The batch does not apply**: a cell is one image; the seed axis is how a grid varies seeds.
// - **Incognito**: the sheet is written only when asked, where asked; the run holds no pixel.

import AppKit
import CoreText
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

// MARK: - A series

/// **A series of the session**: its cells' jobs and how each ended. Kept as long as the app runs — a few
/// settings per cell —, its images being the history's (`AppState.cells(of:)`).
struct GridRun {
    let ordinal: Int
    let grid: RenderGrid
    /// What the cells share: the prompt as typed (its groups included), the format, the stack.
    let settings: RenderSettings
    /// The job of each cell, in the grid's order (`RenderGrid.cells`).
    let jobs: [Int]
    /// The same jobs whole: each cell's settings, as the canvas says them.
    var cellJobs: [Job] = []
    let references: Int
    var issues: [Int: Issue] = [:]
    /// Who asked for it (`Entry.owner`).
    var owner: Data? = nil
    var name: String { "g\(ordinal)" }
    var isComplete: Bool { issues.count == jobs.count }
}

// MARK: - The sheet, asked

extension AppState {
    /// **What the sheet says**, in the user's language: the prompt as typed, what the cells share,
    /// the axes' values.
    func sheetDrawing(_ run: GridRun) -> SheetDrawing? {
        let images = cells(of: run)
        guard !images.isEmpty else { return nil }
        let g = run.grid, r = run.settings
        let stack = r.loras
        func lora(_ axis: RenderGrid.Axis?) -> Int? { if case .loraStrength(let slot, _)? = axis { slot } else { nil } }
        func labels(_ axis: RenderGrid.Axis) -> [String] {
            let name = lora(axis).flatMap { stack.indices.contains($0) ? loraName(stack[$0].path) : nil } ?? "LoRA"
            return g.values(of: axis).map { GridWords.label($0, loraName: name, modelName: modelName, addedName: loraName) }
        }
        func varies(_ test: (RenderGrid.Axis) -> Bool) -> Bool { test(g.x) || g.y.map(test) == true }
        // What every cell shares: what no axis varies. A model axis gives each its own steps.
        let models = varies { if case .models = $0 { true } else { false } }
        var shared = (models ? [] : [r.modelName]) + [Figures.size(r.format.width, r.format.height)]
        if !models, !varies({ if case .steps = $0 { true } else { false } }) { shared.append(String(localized: "\(r.steps) steps")) }
        if !varies({ switch $0 { case .seeds, .consecutiveSeeds: true; default: false } }) {
            shared.append(String(localized: "seed \(String(g.base.seed))"))
        }
        for (i, l) in stack.enumerated() where lora(g.x) != i && lora(g.y) != i {
            shared.append("\(loraName(l.path)) \(Figures.strength(l.strength))")
        }
        if r.solver != .euler, !varies({ if case .solvers = $0 { true } else { false } }) { shared.append(r.solver.choice) }
        if run.references > 0, !varies({ if case .images = $0 { true } else { false } }) {
            shared.append(String(localized: "edit, \(run.references) images"))
        }
        let axes = [g.x] + (g.y.map { [$0] } ?? [])
        let names = axes.map { GridWords.name($0, grid: g, stack: stack, loraName: loraName) }
        let layout = g.sheet(imageWidth: r.format.width, imageHeight: r.format.height)
        let cells = g.cells.enumerated().map { i, cell in
            // A cell rendered, then taken out of the history: removed, as the canvas says it.
            var issue = run.issues[i]
            if images[i] == nil, case .finished? = issue { issue = .removed }
            return SheetDrawing.Cell(column: cell.column, row: cell.row, png: images[i]?.png, note: GridWords.note(issue))
        }
        return SheetDrawing(layout: layout, cells: cells, title: g.title, subtitle: shared.joined(separator: " · "),
                            columnLabels: labels(g.x), rowLabels: g.y.map(labels) ?? [],
                            metadata: ["Software": "Siliconed grid", "Title": g.title,
                                       "Description": "x: \(names[0])" + (names.count > 1 ? " · y: \(names[1])" : "")
                                           + " · " + shared.joined(separator: " · "),
                                       "grid": "\(g.columns)x\(g.rows)"])
    }


    /// **The sheet's file name**: `grid-z-image-2x2-512x512-42` — the grid, the cells' format, the base seed (the first
    /// of `seeds=N`), none when the seeds are listed: no cell need have it.
    func sheetName(_ run: GridRun) -> String {
        let g = run.grid, r = run.settings
        let listed = [g.x, g.y].contains { if case .seeds? = $0 { true } else { false } }
        return (["grid", r.identifier.replacingOccurrences(of: "/", with: "-"), "\(g.columns)x\(g.rows)",
                 "\(r.format.width)x\(r.format.height)"] + (listed ? [] : ["\(g.base.seed)"])).joined(separator: "-")
    }

    /// The sheet's PNG, drawn off the main thread from the cells the history holds now — `nil` when it holds none.
    func drawSheet(_ run: GridRun) async throws -> Data? {
        guard let drawing = sheetDrawing(run) else { return nil }
        return try await Self.draw(drawing)
    }

    private static func draw(_ drawing: SheetDrawing) async throws -> Data {
        try await Task.detached(priority: .userInitiated) { try drawing.draw() }.value
    }

    /// **« Export Sheet… »**: the series' sheet as a PNG, where the panel says — the sheet as the click found it.
    func exportSheet(_ run: GridRun) {
        guard let drawing = sheetDrawing(run) else { flash(String(localized: "No cell has an image")); return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = sheetName(run) + ".png"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do {
                try PNG.write(try await Self.draw(drawing), to: url.path)
                flash(String(localized: "Saved: \(url.lastPathComponent)"))
            } catch {
                problem = Problem((error as? EngineError) ?? .imageWriteFailed(file: url.path))
            }
        }
    }

    /// **« Copy Sheet »**: the series' sheet on the pasteboard, as a PNG.
    func copySheet(_ run: GridRun) {
        Task {
            do {
                guard let png = try await drawSheet(run) else { flash(String(localized: "No cell has an image")); return }
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setData(png, forType: .png)
                flash(String(localized: "Sheet copied"))
            } catch {
                problem = Problem(EngineError(error))
            }
        }
    }

    /// **The sheet written for `silicontrol`**: into `path` — a `.png` file, or a folder where it takes its name
    /// (`sheetName`), never over a file. Returns the path written, `nil` when no cell has an image.
    func writeSheet(_ run: GridRun, to path: String) async throws -> String? {
        guard let png = try await drawSheet(run) else { return nil }
        let url = path.lowercased().hasSuffix(".png") ? URL(fileURLWithPath: path)
            : Self.freeURL(in: URL(fileURLWithPath: path, isDirectory: true), base: sheetName(run), ext: "png")
        try PNG.write(png, to: url.path)
        return url.path
    }
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
        case .solvers: return String(localized: "solver")
        case .startStrength: return String(localized: "starting image")
        case .detail: return String(localized: "Detail")
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
        case .solver(let s): return s.choice
        case .startStrength(let f): return String(localized: "start \(StartWords.percent(f))")
        case .detail(let d): return d.choice
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
        case .solvers(let s): return "solver=" + s.map(\.rawValue).joined(separator: ",")
        case .startStrength(let v): return "start=" + v.map { String(format: "%.2f", $0) }.joined(separator: ",")
        case .detail(let d): return "detail=" + d.map(\.rawValue).joined(separator: ",")
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

    /// The sheet's PNG. One cell decoded at a time, straight to its size on the sheet (ImageIO's thumbnail): a 1024²
    /// cell never lives at full size here.
    func draw() throws -> Data {
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
        return try PNG.data(image, metadata: metadata)
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
