import Foundation

// **A series: what « Generate » renders when something varies** — the prompt's `{…}` groups, the lot's seeds, the knobs
// bracketed — laid out on two axes at most, columns then rows, then rendered as a grid (`RenderGrid`): one finished image
// per cell, each into the history.
//
//     present   group 0 · seeds · LoRA « flat »          what varies, in this order: the text's groups, the lot, the knobs
//     placed    [seeds, —]                                where the hand put some (« Bracket… », its Columns | Rows)
//        │ the placed keep their place, the others fill the free one in the order present
//        ▼
//     columns seeds · rows group 0 · excess LoRA « flat »   beyond two: the excess, which the app refuses to render
//
// What an analogy with a camera's bracketing would miss:
//
// - **Two axes at most, the lot one of them.** A third thing that varies is never multiplied in silence: « Generate »
//   says it, and the user takes one away (its ✕ beside the button).
// - **Placing an axis where another one is replaces it** (`place`): the one pushed out is named, so that the app says it
//   before the click and brings that setting back to one value (the lot to 1, a group to its first word, a knob
//   unbracketed).
// - **One axis alone is the columns**: a series of one axis reads as a row, whatever place it was given.

/// **What a series varies** — one axis of it.
package enum SeriesKey: Hashable, Sendable, Codable {
    /// The prompt's `n`-th `{…}` group (from 0).
    case group(Int)
    /// The lot: consecutive seeds from the rack's.
    case seeds
    /// The strength of one LoRA of the stack, named by its path.
    case loraStrength(String)
    /// LoRAs compared: one added per cell, and the cell without.
    case comparedLoRAs
    /// The starting image's strength (img2img).
    case startStrength
    case solver
    case detail
}

/// **Where the axes of a series go**, columns then rows, and what is beyond two.
package struct SeriesLayout: Equatable, Sendable {
    package let columns: SeriesKey?
    package let rows: SeriesKey?
    /// What varies beyond the two axes: the series does not render while it is not empty.
    package let excess: [SeriesKey]

    /// The axes, columns first.
    package var axes: [SeriesKey] { [columns, rows].compactMap { $0 } }

    /// `present`: what varies, in its order (the text's groups, the lot, the knobs in the order bracketed). `placed`:
    /// `[columns, rows]` as the hand set them — a key no longer present is ignored there.
    package init(present: [SeriesKey], placed: [SeriesKey?]) {
        var slots: [SeriesKey?] = (0..<2).map { i in
            guard placed.indices.contains(i), let k = placed[i], present.contains(k) else { return nil }
            return k
        }
        if slots[0] != nil, slots[0] == slots[1] { slots[1] = nil }
        var excess: [SeriesKey] = []
        for k in present where !slots.contains(k) {
            if let free = slots.firstIndex(where: { $0 == nil }) { slots[free] = k } else { excess.append(k) }
        }
        if slots[0] == nil { slots = [slots[1], nil] }
        columns = slots[0]
        rows = slots[1]
        self.excess = excess
    }

    /// The place a key holds: 0 the columns, 1 the rows, `nil` none.
    package func slot(of key: SeriesKey) -> Int? { columns == key ? 0 : rows == key ? 1 : nil }

    /// **The free place for a key not yet placed**: the rows when the columns are taken, else the columns — what the
    /// bubble offers first, so that a new axis replaces nothing unless the hand says so.
    package var freeSlot: Int { columns == nil ? 0 : 1 }

    /// **`key` placed at `slot`** (0 columns, 1 rows), with what else varies: the places the hand now sets, and the key
    /// pushed out of that place, if another one held it. A key moved from the other place frees it.
    package static func place(_ key: SeriesKey, at slot: Int, present: [SeriesKey],
                              placed: [SeriesKey?]) -> (placed: [SeriesKey?], replaced: SeriesKey?) {
        let now = SeriesLayout(present: present.filter { $0 != key }, placed: placed)
        var slots: [SeriesKey?] = [now.columns, now.rows]
        let slot = slot == 0 ? 0 : 1
        let replaced = slots[slot]
        slots[slot] = key
        return (slots, replaced)
    }

    /// **Columns and rows exchanged** — the places the hand now sets.
    package var swapped: [SeriesKey?] { [rows, columns] }
}

/// **The values a bracket offers before the hand changes them**: `count` values `step` apart, centered on `center` (the
/// knob's value), rounded to the hundredth, held within `range`, each once — three around the value by default.
package enum Bracketing {
    package static func around(_ center: Double, step: Double, count: Int, in range: ClosedRange<Double>) -> [Double] {
        let n = max(1, count)
        var out: [Double] = []
        for i in 0..<n {
            let v = ((center + (Double(i) - Double(n - 1) / 2) * step) * 100).rounded() / 100
            let held = min(max(v, range.lowerBound), range.upperBound)
            if !out.contains(held) { out.append(held) }
        }
        return out
    }
}

/// **A series' grid sized to the canvas** (`SeriesCanvas`): every cell in sight, as large as the room lets them, the window
/// larger or smaller — until the hand sizes them (pinch, ⌘+ ⌘−, the slider): they keep its size then, until another series,
/// or ⌘0 that gives the size back to the room.
package struct SeriesFit: Equatable, Sendable {
    /// The canvas around the cells, in points: its margin on each side, the gap between two cells (and between the labels
    /// and the cells), a row label's width, a column label's height (two lines of a callout).
    package static let margin: CGFloat = 24, gap: CGFloat = 8, rowLabelWidth: CGFloat = 120, columnLabelHeight: CGFloat = 44
    /// A cell's long side, in points: the slider's range, the pinch's and ⌘+ ⌘−'s bounds.
    package static let sides: ClosedRange<CGFloat> = 96...1024
    package static func clamped(_ side: CGFloat) -> CGFloat { min(max(side, sides.lowerBound), sides.upperBound) }

    /// The series fitted.
    package let run: Int
    /// Whether the band under the grid (a cell's settings) was there.
    package var band: Bool
    /// The side the cells got: while they keep it, they follow the room.
    package var side: CGFloat
    /// The side that fills the room now — what ⌘0 gives back.
    package var fitting: CGFloat

    /// Whether cells of side `current` follow the room — ⌘0 has nothing to give back.
    package func follows(_ current: CGFloat) -> Bool { current == side }

    /// **The side at which the whole grid holds in `room`**, its labels and gaps included: `aspect` the cells' width over
    /// their height, `rowLabels` and `columnLabels` whether the axes have words beside and above the cells.
    package static func side(in room: CGSize, columns: Int, rows: Int, aspect: CGFloat, rowLabels: Bool,
                             columnLabels: Bool) -> CGFloat {
        let beside = rowLabels ? rowLabelWidth + gap : 0
        let above = (columnLabels ? columnLabelHeight : 0) + gap
        let w = (room.width - 2 * margin - beside - CGFloat(columns - 1) * gap) / CGFloat(columns)
        let h = (room.height - 2 * margin - above - CGFloat(rows - 1) * gap) / CGFloat(rows)
        return clamped(min(w / min(1, aspect), h / min(1, 1 / aspect)))
    }

    /// **The room or the series changed**: the fit kept, and the side the cells take — `nil`, they keep theirs. They keep
    /// it when the hand sized them (their side is no longer the fitted one), or when only the band came or went: a click on
    /// a cell shows its band, and the cells must not move under the pointer, nor at each cell of a running series.
    package static func refit(_ last: SeriesFit?, run: Int, band: Bool, current: CGFloat,
                              fitting: CGFloat) -> (fit: SeriesFit, side: CGFloat?) {
        if var f = last, f.run == run {
            f.fitting = fitting
            guard f.follows(current) else { return (f, nil) }
            guard f.band == band else { f.band = band; return (f, nil) }
        }
        return (SeriesFit(run: run, band: band, side: fitting, fitting: fitting), fitting)
    }

    /// **⌘0**: the cells take the side that fills the room, and follow it again.
    package mutating func restore() -> CGFloat {
        side = fitting
        return side
    }
}
