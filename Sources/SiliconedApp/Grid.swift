// **The XY grid in the app: the form's two axes, the cells in the queue, the sheet.**
//
//     a series         what « Generate » varies — the text's groups, the lot, the knobs bracketed
//                      (`Series.swift`) — or the axes of `silicontrol grid`
//        │
//        ▼
//     RenderGrid       the library checks it and lists its cells, row by row (`RenderGrid.swift`)
//        │ `AppState.enqueueGrid`: one ordinary job per cell, the run remembered (`GridRun`)
//        ▼
//     the queue        each cell renders like any other job, a finished image into the history
//        │ the last cell ends (done, failed, stopped or removed)
//        ▼
//     the sheet        drawn off the main thread (`SheetDrawing`): title, labels, each cell reduced to
//                      512 px at most — an entry of the history, saved, copied, dragged like the others;
//                      for a series, only when asked (« Add Sheet to History »)
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
    /// The same jobs whole: each cell's settings, as the canvas says them.
    var cellJobs: [Job] = []
    let references: Int
    /// The sheet is drawn when the last cell ends (`silicontrol grid`); a series' only when asked.
    var sheet = true
    var entries: [Int: Entry] = [:]
    var issues: [Int: Issue] = [:]
    /// Who asked for it (`Entry.owner`).
    var owner: Data? = nil
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
