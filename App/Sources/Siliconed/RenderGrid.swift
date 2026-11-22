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
//   strength, the model, the image an edit starts from, the solver, the starting image's strength
//   (img2img), the Detail. Not a batch: a cell is one image
//   (the seed axis is the batch of a grid).
// - **A solver axis names solvers, not what a model takes**: a cell keeps its solver (`Cell.solver`),
//   and the app that queues it decides what a model walking Euler alone does with it — as it does
//   for the rack's choice. Same seed, same σ, same evaluations: only the rule between two steps moves.
// - **A model axis gives each cell its model's own step count** (`Cell.steps` nil) unless steps are
//   an axis too: the base's count is one model's, and the families need not agree. And it
//   refuses a LoRA stack (`loraAcrossModels`): a LoRA is made for one model, the others would refuse
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
package struct RenderGrid: Equatable, Sendable {

    /// **What an axis varies.** Indices count from 0 (`group` in `PromptAlternatives.groups`, `slot`
    /// in the LoRA stack).
    package enum Axis: Equatable, Sendable, Codable {
        /// The alternatives of the prompt's `group`-th `{…}`, in the order typed.
        case prompt(group: Int)
        /// These seeds, in this order.
        case seeds([UInt64])
        /// `count` consecutive seeds from the base seed: `s, s+1, …` (wrapping past `UInt64.max`).
        case consecutiveSeeds(count: Int)
        /// These step counts, each in `1...Request.maximumSteps`.
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
        /// These solvers, in this order.
        case solvers([Solver])
        /// These strengths of the starting image (img2img), in `(0, 1]`: 1 ignores the image; 0, which would keep it, is refused.
        case startStrength([Double])
        /// These levels of Detail, in this order.
        case detail([Detail])
    }

    /// One value of an axis, as an app labels it (and formats it in its language).
    package enum Value: Equatable, Sendable {
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
        case solver(Solver)
        case startStrength(Double)
        case detail(Detail)
    }

    /// **What every cell shares**, unless an axis says otherwise. `prompt` is the prompt as typed,
    /// its `{…}` groups included; `loraStrengths` the stack's strengths, in its order.
    package struct Base: Equatable, Sendable, Codable {
        package var prompt: String
        package var seed: UInt64
        package var steps: Int
        package var loraStrengths: [Double]
        /// How many images the edit has — what an image axis splits (0: no edit).
        package var images: Int
        /// Whether the cells start from an image (img2img) — what a start strength axis needs.
        package var start: Bool

        package init(prompt: String, seed: UInt64, steps: Int, loraStrengths: [Double] = [], images: Int = 0,
                     start: Bool = false) {
            self.prompt = prompt
            self.seed = seed
            self.steps = steps
            self.loraStrengths = loraStrengths
            self.images = images
            self.start = start
        }
    }

    /// **One render of the grid**: where it sits on the sheet, and its settings.
    package struct Cell: Equatable, Sendable {
        package let column: Int, row: Int
        /// The expanded prompt — no brace left but the escaped ones, resolved.
        package let prompt: String
        package let seed: UInt64
        /// `nil`: the cell's model's own count — a model axis without a steps axis.
        package let steps: Int?
        package let loraStrengths: [Double]
        /// The cell's model; `nil`: the base's (no model axis).
        package let model: String?
        /// The LoRA this cell adds to the stack, and its strength; `nil`: none.
        package let addedLoRA: String?
        package let addedStrength: Double
        /// The one edit image this cell starts from (index in the base's); `nil`: all of them, read
        /// together (no image axis).
        package let image: Int?
        /// The cell's solver; `nil`: the base's (no solver axis).
        package let solver: Solver?
        /// The starting image's strength; `nil`: the base's (no start strength axis).
        package let startStrength: Double?
        /// The cell's Detail; `nil`: the base's (no Detail axis).
        package let detail: Detail?
    }

    /// **The most cells a grid may have** — the app's queue holds that many renders.
    package static let maxCells = PromptAlternatives.maxCombinations

    package let base: Base
    /// The columns.
    package let x: Axis
    /// The rows; `nil`: one row.
    package let y: Axis?
    /// The prompt, parsed: its groups are the prompt axes' values.
    package let alternatives: PromptAlternatives

    /// Checks the grid, and refuses — before any render — what would not make one: a prompt that
    /// does not parse (`promptSyntax`), an axis without a value or naming what is not there, X and Y
    /// varying the same setting, a group that is no axis, a value no render takes
    /// (`gridRefused`), more than `limit` cells (`tooManyGridCells`).
    package init(_ base: Base, x: Axis, y: Axis? = nil, limit: Int = maxCells) throws(EngineError) {
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
                guard s.allSatisfy({ (1...Request.maximumSteps).contains($0) }) else { throw .gridRefused(reason: .invalidValue) }
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
            case .solvers(let s):
                guard !s.isEmpty else { throw .gridRefused(reason: .emptyAxis) }
                guard Set(s).count == s.count else { throw .gridRefused(reason: .invalidValue) }
            case .startStrength(let v):
                guard base.start else { throw .gridRefused(reason: .noStartImage) }
                guard !v.isEmpty else { throw .gridRefused(reason: .emptyAxis) }
                // The bounds a render admits (`Strength`): above 0, at most 1.
                guard v.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 1 }) else { throw .gridRefused(reason: .invalidValue) }
            case .detail(let d):
                guard !d.isEmpty else { throw .gridRefused(reason: .emptyAxis) }
                guard Set(d).count == d.count else { throw .gridRefused(reason: .invalidValue) }
            }
        }
        // A LoRA is made for one model: none across models, neither the stack nor an added one.
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
        case .solvers: "solver"
        case .startStrength: "start"
        case .detail: "detail"
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
        case .solvers(let s): s.count
        case .startStrength(let v): v.count
        case .detail(let d): d.count
        }
    }

    package var columns: Int { Self.size(x, alternatives.groups, images: base.images) }
    package var rows: Int { y.map { Self.size($0, alternatives.groups, images: base.images) } ?? 1 }
    /// How many renders the grid queues: one per cell.
    package var count: Int { columns * rows }

    /// **The values of an axis**, in its order — a column's or a row's label.
    package func values(of axis: Axis) -> [Value] {
        switch axis {
        case .prompt(let g): alternatives.groups[g].map(Value.alternative)
        case .seeds(let s): s.map(Value.seed)
        case .consecutiveSeeds(let n): (0..<n).map { .seed(base.seed &+ UInt64($0)) }
        case .steps(let s): s.map(Value.steps)
        case .loraStrength(_, let v): v.map(Value.strength)
        case .models(let m): m.map(Value.model)
        case .images: (0..<base.images).map(Value.image)
        case .addedLoRAs(let l, _): l.map(Value.addedLoRA)
        case .solvers(let s): s.map(Value.solver)
        case .startStrength(let v): v.map(Value.startStrength)
        case .detail(let d): d.map(Value.detail)
        }
    }

    /// **The cells, row by row** — X varies the fastest: the sheet's reading order, and the queue's.
    package var cells: [Cell] {
        let xs = values(of: x)
        let ys: [Value?] = y.map { values(of: $0).map { Optional($0) } } ?? [nil]
        var result: [Cell] = []
        result.reserveCapacity(xs.count * ys.count)
        for (row, vy) in ys.enumerated() {
            for (column, vx) in xs.enumerated() {
                var choice: [Int: String] = [:]
                var seed = base.seed, strengths = base.loraStrengths
                var steps: Int? = base.steps, model: String? = nil, image: Int? = nil
                var added: String? = nil, addedStrength = 0.0, solver: Solver? = nil
                var start: Double? = nil, detail: Detail? = nil
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
                    case (_, .solver(let s)): solver = s
                    case (_, .startStrength(let f)): start = f
                    case (_, .detail(let d)): detail = d
                    default: break
                    }
                }
                if model != nil, !stepsSet { steps = nil }
                result.append(Cell(column: column, row: row, prompt: prompt(choice), seed: seed, steps: steps,
                                   loraStrengths: strengths, model: model,
                                   addedLoRA: added, addedStrength: added == nil ? 0 : addedStrength, image: image,
                                   solver: solver, startStrength: start, detail: detail))
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
    package var title: String {
        alternatives.pieces.map { piece in
            switch piece {
            case .text(let t): t
            case .alternatives(let a): "{" + a.joined(separator: "|") + "}"
            }
        }.joined()
    }
}
