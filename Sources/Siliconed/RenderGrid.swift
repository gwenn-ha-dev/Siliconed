import CoreGraphics
import Foundation

// **An XY grid: two settings swept, one render per cell, and the sheet that lays them side by side.**
//
//     prompt "woman posing in a {library|greenhouse}"   seed 42   steps 8   LoRA strengths [0.8]
//     x: .prompt(group: 0)        y: .seeds([42, 7])
//        │ check: every group is an axis, the axes differ, each has a value, ≤ maxCells cells
//        ▼
//                     library                         greenhouse
//       42   "…in a library"  seed 42      "…in a greenhouse"  seed 42        cells, row by row:
//        7   "…in a library"  seed 7       "…in a greenhouse"  seed 7         (0,0) (1,0) (0,1) (1,1)
//        │
//        ▼  one render per cell, then the sheet (`Sheet`): a title band, the column labels, the row
//           labels, the cells reduced to at most `Sheet.maxCellSide`
//
// The idea is ComfyUI's « XY plot ». What an analogy would miss:
//
// - **A cell is one ordinary render**: one prompt, one seed, one step count, one LoRA stack. The grid
//   builds nothing the engine does not already take (a `Request`), and an app queues its cells as
//   any other render — the engine never hears of a grid.
// - **The order is the sheet's reading order**: row by row, X varying the fastest. The queue renders
//   in that order, so the images arrive as the sheet is read, and the n-th cell of a grid is the same
//   settings from one run to the next.
// - **A group of alternatives that is not an axis is refused** (`GridRefusal.groupNotOnAxis`), not
//   multiplied as `PromptAlternatives` alone would: a cell holds one image, and a forgotten group
//   would silently make each cell a stack — and double the bill. Put it on an axis, or escape it.
// - **Only what varies is an axis.** The prompt's groups, the seed, the step count, one LoRA's
//   strength, the model, the image an edit starts from. Not a batch: a cell is one image (the seed
//   axis is the batch of a grid).
// - **A model axis gives each cell its model's own step count** (`Cell.steps` nil) unless steps are
//   an axis too: Z-Image takes 8, Qwen-Image-2.1 6, klein 4 — the base's count is one model's. And it
//   refuses a LoRA stack (`loraAcrossModels`): a LoRA is forged for one model, the others would refuse
//   it cell by cell.
// - **A LoRA axis compares LoRAs** (`.addedLoRAs`): each cell adds one of them to the base's stack,
//   all at one strength — or none (`""`), the cell that says what the LoRA changes. It is not the
//   strength axis, which sweeps one LoRA of the stack; the two can be X and Y. Refused across models,
//   for the same reason as the stack.
// - **An image axis splits an edit's images**: the base names how many images the edit has, and each
//   cell edits one of them alone (`Cell.image`) with the same instruction — not the images read
//   together, which is what an edit without the axis does.
// - **`seeds(count:)` counts from the base seed** (`s, s+1, …`), like a batch, and is resolved only
//   once the grid is checked: `seeds=1000000` is a refusal, not a million-element array.
// - **The sheet stays small**: each cell is reduced to `Sheet.maxCellSide` (512 px) on its long side,
//   and all the cells together to `Sheet.maxCellsArea` (12 Mpx) — 64 cells of 512² come out at 432 px,
//   a sheet of ~14 Mpx, ~55 MB as RGBA while it is drawn. The images keep their full size in the
//   history; the sheet is the comparison.

/// **An XY grid**: the base settings, one or two axes, and the cells they make.
public struct RenderGrid: Equatable, Sendable {

    /// **What an axis varies.** Indices count from 0 (`group` in `PromptAlternatives.groups`, `slot`
    /// in the LoRA stack).
    public enum Axis: Equatable, Sendable {
        /// The alternatives of the prompt's `group`-th `{…}`, in the order typed.
        case prompt(group: Int)
        /// These seeds, in this order.
        case seeds([UInt64])
        /// `count` consecutive seeds from the base seed: `s, s+1, …` (wrapping past `UInt64.max`).
        case consecutiveSeeds(count: Int)
        /// These step counts.
        case steps([Int])
        /// These strengths for the `slot`-th LoRA of the stack; the others keep theirs.
        case loraStrength(slot: Int, values: [Double])
        /// These models, by identifier (`z-image`, `qwen-image-2.1`…), in this order.
        case models([String])
        /// One LoRA added to the stack per cell, from `loras` (a path or a name, as the caller keeps
        /// them; `""`: none), all at `strength`.
        case addedLoRAs([String], strength: Double)
        /// Each of the base's `images` edit images, alone: index 0 is the first.
        case images
    }

    /// One value of an axis, as an app labels it (and formats it in its language).
    public enum Value: Equatable, Sendable {
        /// An alternative of a group, as typed (`""` for an empty one).
        case alternative(String)
        case seed(UInt64)
        case steps(Int)
        case strength(Double)
        case model(String)
        /// A LoRA an axis adds (`""`: none).
        case addedLoRA(String)
        /// An edit image, by its index in the base's images (from 0).
        case image(Int)
    }

    /// **What every cell shares**, unless an axis says otherwise. `prompt` is the prompt as typed,
    /// its `{…}` groups included; `loraStrengths` the stack's strengths, in its order.
    public struct Base: Equatable, Sendable {
        public var prompt: String
        public var seed: UInt64
        public var steps: Int
        public var loraStrengths: [Double]
        /// How many images the edit has — what an image axis splits (0: no edit).
        public var images: Int

        public init(prompt: String, seed: UInt64, steps: Int, loraStrengths: [Double] = [], images: Int = 0) {
            self.prompt = prompt
            self.seed = seed
            self.steps = steps
            self.loraStrengths = loraStrengths
            self.images = images
        }
    }

    /// **One render of the grid**: where it sits on the sheet, and its settings.
    public struct Cell: Equatable, Sendable {
        public let column: Int, row: Int
        /// The expanded prompt — no brace left but the escaped ones, resolved.
        public let prompt: String
        public let seed: UInt64
        /// `nil`: the cell's model's own count — a model axis without a steps axis.
        public let steps: Int?
        public let loraStrengths: [Double]
        /// The cell's model; `nil`: the base's (no model axis).
        public let model: String?
        /// The LoRA this cell adds to the stack, and its strength; `nil`: none.
        public let addedLoRA: String?
        public let addedStrength: Double
        /// The one edit image this cell starts from (index in the base's); `nil`: all of them, read
        /// together (no image axis).
        public let image: Int?
    }

    /// **The most cells a grid may have** — the app's queue holds that many renders.
    public static let maxCells = PromptAlternatives.maxCombinations

    public let base: Base
    /// The columns.
    public let x: Axis
    /// The rows; `nil`: one row.
    public let y: Axis?
    /// The prompt, parsed: its groups are the prompt axes' values.
    public let alternatives: PromptAlternatives

    /// Checks the grid, and refuses — before any render — what would not make one: a prompt that
    /// does not parse (`promptSyntax`), an axis without a value or naming what is not there, X and Y
    /// varying the same setting, a group that is no axis, a value no render takes
    /// (`gridRefused`), more than `limit` cells (`tooManyGridCells`).
    public init(_ base: Base, x: Axis, y: Axis? = nil, limit: Int = maxCells) throws(EngineError) {
        // The groups' product is no limit here: the axes are, and every group must be one.
        let alternatives = try PromptAlternatives(base.prompt, limit: .max)
        let groups = alternatives.groups
        let axes = [x] + (y.map { [$0] } ?? [])
        for a in axes {
            switch a {
            case .prompt(let g):
                guard groups.indices.contains(g) else { throw .gridRefused(reason: .noSuchGroup) }
            case .seeds(let s):
                guard !s.isEmpty else { throw .gridRefused(reason: .emptyAxis) }
            case .consecutiveSeeds(let n):
                guard n >= 1 else { throw .gridRefused(reason: n == 0 ? .emptyAxis : .invalidValue) }
            case .steps(let s):
                guard !s.isEmpty else { throw .gridRefused(reason: .emptyAxis) }
                guard s.allSatisfy({ $0 >= 1 }) else { throw .gridRefused(reason: .invalidValue) }
            case let .loraStrength(slot, values):
                guard base.loraStrengths.indices.contains(slot) else { throw .gridRefused(reason: .noSuchLoRA) }
                guard !values.isEmpty else { throw .gridRefused(reason: .emptyAxis) }
                guard values.allSatisfy(\.isFinite) else { throw .gridRefused(reason: .invalidValue) }
            case .models(let m):
                guard !m.isEmpty else { throw .gridRefused(reason: .emptyAxis) }
                guard Set(m).count == m.count, !m.contains(where: \.isEmpty) else { throw .gridRefused(reason: .invalidValue) }
                guard base.loraStrengths.isEmpty else { throw .gridRefused(reason: .loraAcrossModels) }
            case .images:
                guard base.images >= 1 else { throw .gridRefused(reason: .emptyAxis) }
            case let .addedLoRAs(l, strength):
                guard !l.isEmpty else { throw .gridRefused(reason: .emptyAxis) }
                guard Set(l).count == l.count, strength.isFinite else { throw .gridRefused(reason: .invalidValue) }
            }
        }
        // A LoRA is forged for one model: none across models, neither the stack nor an added one.
        let varies = axes.map(Self.setting)
        if varies.contains("model"), varies.contains("added lora") { throw .gridRefused(reason: .loraAcrossModels) }
        if let y, Self.setting(x) == Self.setting(y) { throw .gridRefused(reason: .sameAxisTwice) }
        let onAxes = Set(axes.compactMap { if case .prompt(let g) = $0 { g } else { nil } })
        guard onAxes.count == groups.count else { throw .gridRefused(reason: .groupNotOnAxis) }
        // An empty group (`{}`) has one alternative: `""`. It is a 1-wide axis, never an empty one.
        let (count, overflow) = Self.size(x, groups, images: base.images)
            .multipliedReportingOverflow(by: y.map { Self.size($0, groups, images: base.images) } ?? 1)
        guard !overflow, count <= limit else { throw .tooManyGridCells(count: overflow ? .max : count, max: limit) }
        self.base = base
        self.x = x
        self.y = y
        self.alternatives = alternatives
    }

    /// What an axis varies, to tell two axes apart: the seed is one setting however it is listed.
    private static func setting(_ a: Axis) -> String {
        switch a {
        case .prompt(let g): "prompt \(g)"
        case .seeds, .consecutiveSeeds: "seed"
        case .steps: "steps"
        case .loraStrength(let slot, _): "lora \(slot)"
        case .models: "model"
        case .images: "image"
        case .addedLoRAs: "added lora"
        }
    }

    private static func size(_ a: Axis, _ groups: [[String]], images: Int) -> Int {
        switch a {
        case .prompt(let g): groups[g].count
        case .seeds(let s): s.count
        case .consecutiveSeeds(let n): n
        case .steps(let s): s.count
        case .loraStrength(_, let v): v.count
        case .models(let m): m.count
        case .images: images
        case .addedLoRAs(let l, _): l.count
        }
    }

    public var columns: Int { Self.size(x, alternatives.groups, images: base.images) }
    public var rows: Int { y.map { Self.size($0, alternatives.groups, images: base.images) } ?? 1 }
    /// How many renders the grid queues: one per cell.
    public var count: Int { columns * rows }

    /// **The values of an axis**, in its order — a column's or a row's label.
    public func values(of axis: Axis) -> [Value] {
        switch axis {
        case .prompt(let g): alternatives.groups[g].map(Value.alternative)
        case .seeds(let s): s.map(Value.seed)
        case .consecutiveSeeds(let n): (0..<n).map { .seed(base.seed &+ UInt64($0)) }
        case .steps(let s): s.map(Value.steps)
        case .loraStrength(_, let v): v.map(Value.strength)
        case .models(let m): m.map(Value.model)
        case .images: (0..<base.images).map(Value.image)
        case .addedLoRAs(let l, _): l.map(Value.addedLoRA)
        }
    }

    /// **The cells, row by row** — X varies the fastest: the sheet's reading order, and the queue's.
    public var cells: [Cell] {
        let xs = values(of: x)
        let ys: [Value?] = y.map { values(of: $0).map { Optional($0) } } ?? [nil]
        var result: [Cell] = []
        result.reserveCapacity(xs.count * ys.count)
        for (row, vy) in ys.enumerated() {
            for (column, vx) in xs.enumerated() {
                var choice: [Int: String] = [:]
                var seed = base.seed, strengths = base.loraStrengths
                var steps: Int? = base.steps, model: String? = nil, image: Int? = nil
                var added: String? = nil, addedStrength = 0.0
                var stepsSet = false
                for (axis, v) in [(Optional(x), Optional(vx)), (y, vy)] {
                    guard let axis, let v else { continue }
                    switch (axis, v) {
                    case (.prompt(let g), .alternative(let a)): choice[g] = a
                    case (_, .seed(let s)): seed = s
                    case (_, .steps(let n)): steps = n; stepsSet = true
                    case (.loraStrength(let slot, _), .strength(let f)): strengths[slot] = f
                    case (_, .model(let m)): model = m
                    case (_, .image(let i)): image = i
                    case (.addedLoRAs(_, let f), .addedLoRA(let l)): added = l.isEmpty ? nil : l; addedStrength = f
                    default: break
                    }
                }
                if model != nil, !stepsSet { steps = nil }
                result.append(Cell(column: column, row: row, prompt: prompt(choice), seed: seed, steps: steps,
                                   loraStrengths: strengths, model: model,
                                   addedLoRA: added, addedStrength: added == nil ? 0 : addedStrength, image: image))
            }
        }
        return result
    }

    /// The prompt with the `g`-th group replaced by `choice[g]`.
    private func prompt(_ choice: [Int: String]) -> String {
        var s = "", g = 0
        for piece in alternatives.pieces {
            switch piece {
            case .text(let t): s += t
            case .alternatives(let a): s += choice[g] ?? a.first ?? ""; g += 1
            }
        }
        return s
    }

    /// **The sheet's title**: the prompt as typed, its groups written back as `{a|b}` (escapes
    /// resolved — a title is read, not parsed).
    public var title: String {
        alternatives.pieces.map { piece in
            switch piece {
            case .text(let t): t
            case .alternatives(let a): "{" + a.joined(separator: "|") + "}"
            }
        }.joined()
    }

    // ── an axis, written ──

    /// **An axis as `silicontrol grid` writes it**, `kind[:which][=values]`:
    ///
    ///     prompt        the prompt's first group          prompt:2      its second
    ///     seed=42,7     these seeds                       seeds=4       4 from the base seed
    ///     steps=6,8,10  these step counts
    ///     lora=0.4,0.8  the stack's first LoRA            lora:NAME=…   the LoRA named NAME in
    ///                                                     lora:2=…      `loras` (or the 2nd)
    ///     model=z-image,qwen-image-2.1                    image         each edit image alone
    ///     loras=none,flat,ink   one LoRA added per cell   loras:0.6=…   at that strength (1 otherwise)
    ///
    /// Values are separated by commas, decimals written with a point: the syntax of a command line,
    /// not of a language. `loras`: the stack's names, in its order. What does not read is
    /// `gridRefused(.unreadableAxis)`; a LoRA not in `loras`, `gridRefused(.noSuchLoRA)`. The
    /// grid's own checks (`init`) come after.
    public static func axis(_ text: String, loras: [String] = []) throws(EngineError) -> Axis {
        let t = text.trimmingCharacters(in: .whitespaces)
        let (head, list): (Substring, Substring?) = {
            guard let eq = t.firstIndex(of: "=") else { return (t[...], nil) }
            return (t[..<eq], t[t.index(after: eq)...])
        }()
        let kind: Substring, which: Substring?
        if let colon = head.firstIndex(of: ":") {
            kind = head[..<colon]; which = head[head.index(after: colon)...]
        } else {
            kind = head; which = nil
        }
        func values<T>(_ parse: (String) -> T?) throws(EngineError) -> [T] {
            guard let list else { throw .gridRefused(reason: .unreadableAxis) }
            var out: [T] = []
            for token in list.split(separator: ",", omittingEmptySubsequences: false) {
                guard let v = parse(token.trimmingCharacters(in: .whitespaces)) else {
                    throw .gridRefused(reason: .unreadableAxis)
                }
                out.append(v)
            }
            return out
        }
        switch kind.lowercased() {
        case "prompt":
            guard list == nil else { throw .gridRefused(reason: .unreadableAxis) }
            guard let which else { return .prompt(group: 0) }
            guard let n = Int(which), n >= 1 else { throw .gridRefused(reason: .unreadableAxis) }
            return .prompt(group: n - 1)
        case "seed" where which == nil:
            return .seeds(try values { UInt64($0) })
        case "seeds" where which == nil:
            let n = try values { Int($0) }
            guard n.count == 1 else { throw .gridRefused(reason: .unreadableAxis) }
            return .consecutiveSeeds(count: n[0])
        case "steps" where which == nil:
            return .steps(try values { Int($0) })
        case "model" where which == nil, "models" where which == nil:
            return .models(try values { $0.isEmpty ? nil : $0 })
        case "image" where which == nil && list == nil, "images" where which == nil && list == nil:
            return .images
        case "loras":
            let strength: Double
            if let which {
                guard let f = Double(which), f.isFinite else { throw .gridRefused(reason: .unreadableAxis) }
                strength = f
            } else {
                strength = 1
            }
            return .addedLoRAs(try values { $0.isEmpty ? nil : ($0.lowercased() == "none" ? "" : $0) }, strength: strength)
        case "lora":
            let strengths = try values { Double($0).flatMap { $0.isFinite ? $0 : nil } }
            guard let which else {
                guard !loras.isEmpty else { throw .gridRefused(reason: .noSuchLoRA) }
                return .loraStrength(slot: 0, values: strengths)
            }
            if let slot = loras.firstIndex(of: String(which)) { return .loraStrength(slot: slot, values: strengths) }
            if let n = Int(which), loras.indices.contains(n - 1) { return .loraStrength(slot: n - 1, values: strengths) }
            throw .gridRefused(reason: .noSuchLoRA)
        default:
            throw .gridRefused(reason: .unreadableAxis)
        }
    }

    // ── the sheet ──

    /// The sheet of this grid, for renders of `width × height`.
    public func sheet(imageWidth: Int, imageHeight: Int) -> Sheet {
        Sheet(columns: columns, rows: rows, imageWidth: imageWidth, imageHeight: imageHeight, rowLabels: y != nil)
    }

    /// **Where everything goes on the sheet**, in pixels, the origin at the top left:
    ///
    ///     ┌──────────────────────────────────────────────┐
    ///     │ title (2 lines) · what the cells share       │  titleHeight
    ///     │            │ label x0 │ label x1 │ …         │  headerHeight
    ///     │ label y0   │  cell    │  cell    │           │
    ///     │ label y1   │  cell    │  cell    │           │
    ///     └──────────────────────────────────────────────┘
    ///       margin, rowLabelWidth (0 without Y), gap between cells
    ///
    /// Pure arithmetic: an app draws the images and the text in these rectangles.
    public struct Sheet: Equatable, Sendable {
        /// A cell's long side at most: a 1024² render is reduced to 512 on the sheet.
        public static let maxCellSide = 512
        /// All the cells together at most (pixels): the sheet of 64 cells stays near 14 Mpx.
        public static let maxCellsArea = 12_000_000
        /// A cell's long side at least, however many cells.
        public static let minCellSide = 128

        public let columns: Int, rows: Int
        public let cellWidth: Int, cellHeight: Int
        /// The font sizes, in pixels: they follow the cells, so that a label reads at the cells' scale.
        public let labelSize: Int, titleSize: Int
        public let margin: Int, gap: Int
        public let titleHeight: Int, headerHeight: Int, rowLabelWidth: Int
        public let width: Int, height: Int

        public init(columns: Int, rows: Int, imageWidth: Int, imageHeight: Int, rowLabels: Bool) {
            let columns = max(1, columns), rows = max(1, rows)
            let long = max(1, imageWidth, imageHeight), short = max(1, min(imageWidth, imageHeight))
            // The long side such that all the cells together stay within `maxCellsArea`.
            let byArea = Int((Double(Self.maxCellsArea) / Double(columns * rows) * Double(long) / Double(short)).squareRoot())
            let side = max(Self.minCellSide, min(Self.maxCellSide, long, byArea) / 8 * 8)
            cellWidth = max(1, Int((Double(imageWidth) * Double(side) / Double(long)).rounded()))
            cellHeight = max(1, Int((Double(imageHeight) * Double(side) / Double(long)).rounded()))
            self.columns = columns
            self.rows = rows
            labelSize = max(13, Int((Double(side) / 24).rounded()))
            titleSize = Int((Double(labelSize) * 1.3).rounded())
            margin = max(16, labelSize)
            gap = max(6, labelSize / 3)
            // Two lines of title, one line of what the cells share, and air under them.
            titleHeight = Int((Double(titleSize) * 2.7 + Double(labelSize) * 1.6).rounded()) + gap
            headerHeight = labelSize * 2
            rowLabelWidth = rowLabels ? max(140, labelSize * 9) : 0
            width = 2 * margin + (rowLabels ? rowLabelWidth + gap : 0) + columns * cellWidth + (columns - 1) * gap
            height = 2 * margin + titleHeight + headerHeight + rows * cellHeight + (rows - 1) * gap
        }

        private var left: Int { margin + (rowLabelWidth > 0 ? rowLabelWidth + gap : 0) }
        private var top: Int { margin + titleHeight + headerHeight }

        public func cell(column: Int, row: Int) -> CGRect {
            CGRect(x: left + column * (cellWidth + gap), y: top + row * (cellHeight + gap),
                   width: cellWidth, height: cellHeight)
        }

        /// Above its column, the width of a cell.
        public func columnLabel(_ column: Int) -> CGRect {
            CGRect(x: left + column * (cellWidth + gap), y: margin + titleHeight, width: cellWidth, height: headerHeight)
        }

        /// Left of its row, the height of a cell; empty without a Y axis.
        public func rowLabel(_ row: Int) -> CGRect {
            CGRect(x: margin, y: top + row * (cellHeight + gap), width: rowLabelWidth, height: cellHeight)
        }

        /// The title band, the sheet's width inside the margins.
        public var title: CGRect {
            CGRect(x: margin, y: margin, width: width - 2 * margin, height: titleHeight - gap)
        }
    }
}
