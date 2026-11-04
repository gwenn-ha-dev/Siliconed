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
/// knob's value), rounded to the hundredth, held within `range`, each once — three around the value by default (4/11).
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
