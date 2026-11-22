// **The series: what « Generate » renders when something varies — and how the hand says it varies.**
//
//     the text     a woman in a (red · blue · green) dress  ±  a group of words: the card's ± on the selection or the word at
//                                                              the caret, right-click › « Vary This Word… », or braces typed
//     the rack     Seed ± · Detail ± · Solver ± · LoRA ±       a knob bracketed: its ± at its right, right-click › « Bracket… »;
//     the lab      the starting image's strength ±             LoRA ± (the stack's title): « Compare LoRAs… »
//        │ what varies, placed on two axes at most (`SeriesLayout`)
//        ▼
//     beside Generate   ↔  red, blue, green   ✕               one line per axis; its ✕ brings that setting back to one
//                       ↕  3 seeds            ✕   ⇄            value; ⇄ exchanges columns and rows
//                       [ Generate 9 Images ]
//        │ a grid (`RenderGrid`): one job per cell (`AppState.enqueueGrid`), each a finished image into the history
//        ▼
//     the canvas   the grid fills in as the cells come (`SeriesCanvas`), until an image is chosen;
//                  « Add Sheet to History » makes the sheet an image
//
// **Seen, never guessed**: the same ±, always
// shown, in one yellow (`Palette.bracket`), at the right of everything that can vary — outlined it offers, filled it varies —;
// what varies says its values under it, with the arrow of its axis (↔ the columns, ↕ the rows), in the same yellow. A ± that
// could do nothing is not drawn (« Compare LoRAs » with no other LoRA to compare), its place kept.
//
// What an analogy with a camera's bracketing would miss:
//
// - **Finished images only**: no small format — each cell is the render « Generate » would make alone,
//   at the rack's size and steps, at its seed, and it goes to the history like any other.
// - **The lot is the seed axis**: « Seeds 3 » is three seeds from the rack's (s, s+1, s+2) — drawn, or the fixed
//   one's. Its stepper and the ± on the Seed's line set the same count (`batch`), under one name, Seeds; the ± also chooses
//   the axis. Alone or beside another axis it is a series, a row or a column of the grid, its tile
//   in the history like any other; its cells still render in one pass, as a lot (`enqueueGrid` trains cells that differ
//   only by their seed). No seed is typed in the app: listed seeds are `silicontrol grid`'s.
// - **Neither the model nor the steps**: the five models are distilled for their step count, and a sweep across
//   models is `silicontrol grid`'s; so is each image of an edit alone.
// - **A bracket that no longer applies does not vary** (its LoRA switched off or removed, Detail on a model without it, no starting
//   image): its values wait, unseen, and come back with their knob.
// - **A series is two values or more**: one is the knob itself. A bracket left with fewer does not vary (`Bracket.values`), and
//   the bubble does not offer to make one — « Compare LoRAs » needs a LoRA beside the image without.
// - **Three things varying never render**: « Generate » says it, and a ✕ takes one away. Placing an axis where another
//   is replaces that one, and the bubble says which before the click (`SeriesLayout.place`).

import AppKit
import Siliconed
import SwiftUI

/// **A knob bracketed**: its values, kept while the knob applies.
struct Bracket: Equatable {
    let key: SeriesKey
    /// A LoRA's strengths, or the starting image's (0…1).
    var numbers: [Double] = []
    /// LoRAs compared, by path — `""` the cell without — each added at `strength`.
    var loras: [String] = []
    var strength = 0.8
    var solvers: [Solver] = []
    var details: [Detail] = []

    /// **How many images it makes along its axis** — a series is two or more: one value is the knob itself.
    var values: Int {
        switch key {
        case .loraStrength, .startStrength: Set(numbers).count
        case .comparedLoRAs: loras.contains { !$0.isEmpty } ? loras.count : 0
        case .solver: solvers.count
        case .detail: details.count
        case .group, .seeds: 2
        }
    }
}

/// **What a bubble brackets**: an axis that is (or is about to be) — or words of the text, about to become a group.
enum BracketTarget: Equatable {
    case key(SeriesKey)
    /// The characters `range` (UTF-16) of the prompt, as it was when the bubble opened.
    case words(NSRange, prompt: String)
}

// MARK: - The state

extension AppState {
    /// Whether a bracket's knob applies to the render as set: a LoRA on in the stack, a starting image, Detail offered…
    func applies(_ key: SeriesKey) -> Bool {
        switch key {
        case .group, .seeds: true
        case .loraStrength(let path): activeLoRAs.contains { $0.path == path }
        case .comparedLoRAs: !compatibleLoras.isEmpty
        case .startStrength: usedStart != nil
        case .solver: offeredSolvers.count > 1
        case .detail: offersDetail
        }
    }

    /// The brackets whose knob applies and that make a series (two values or more), in the order bracketed.
    var activeBrackets: [Bracket] { brackets.filter { applies($0.key) && $0.values > 1 } }

    /// Whether the ± may open: « Compare LoRAs » needs two LoRAs for this model, one of them not already in the stack, and room
    /// for it — a model with one compares it with the image without by its strength's ± (0 beside its value).
    func canBracket(_ key: SeriesKey) -> Bool {
        guard case .comparedLoRAs = key, !varies(key) else { return true }
        return compatibleLoras.count > 1 && canAddLoRA && compatibleLoras.contains { l in !loras.contains { $0.path == l.path } }
    }

    func bracket(_ key: SeriesKey) -> Bracket? { activeBrackets.first { $0.key == key } }

    /// **What varies**, in its order: the text's groups, the lot, the knobs bracketed.
    var seriesPresent: [SeriesKey] {
        promptGroups.indices.map(SeriesKey.group) + (batch > 1 ? [.seeds] : []) + activeBrackets.map(\.key)
    }

    var seriesLayout: SeriesLayout { SeriesLayout(present: seriesPresent, placed: seriesPlaced) }

    /// Whether a key is one of the series' axes, or beyond them.
    func varies(_ key: SeriesKey) -> Bool { seriesPresent.contains(key) }

    /// **The grid axis a key stands for**, its values as bracketed.
    func gridAxis(_ key: SeriesKey) -> RenderGrid.Axis? {
        switch key {
        case .group(let g): return .prompt(group: g)
        case .seeds: return .consecutiveSeeds(count: max(1, batch))
        case .loraStrength(let path):
            guard let slot = activeLoRAs.firstIndex(where: { $0.path == path }), let b = bracket(key) else { return nil }
            return .loraStrength(slot: slot, values: b.numbers)
        case .comparedLoRAs: return bracket(key).map { .addedLoRAs($0.loras, strength: $0.strength) }
        case .startStrength: return bracket(key).map { .startStrength($0.numbers) }
        case .solver: return bracket(key).map { .solvers($0.solvers) }
        case .detail: return bracket(key).map { .detail($0.details) }
        }
    }

    /// **The grid « Generate » renders**, checked by the library — or `nil` when nothing varies. `seed`: the base seed (the
    /// one drawn at the click for a random seed).
    func seriesGrid(seed: UInt64) -> Result<RenderGrid?, EngineError> {
        let layout = seriesLayout
        guard layout.excess.isEmpty, let columns = layout.columns else { return .success(nil) }
        return Result { () throws(EngineError) in
            guard let x = gridAxis(columns) else { throw .gridRefused(reason: .emptyAxis) }
            var y: RenderGrid.Axis?
            if let rows = layout.rows {
                guard let a = gridAxis(rows) else { throw .gridRefused(reason: .emptyAxis) }
                y = a
            }
            let g = try RenderGrid(.init(prompt: prompt, seed: seed, steps: steps, loraStrengths: activeLoRAs.map(\.strength),
                                         images: references.count, start: usedStart != nil), x: x, y: y)
            guard !refusesNegativeStrength(g) else { throw .gridRefused(reason: .invalidValue) }
            return g
        }
    }

    /// **What « Generate » makes of the series**, in one reading: why it does not render (three things varying, or a grid the
    /// library refuses), and how many jobs it queues — one per cell, one for a render; one image each. Each reading builds
    /// the grid again (the carry, the prompt's parse): a body that needs both reads it once.
    var seriesReading: (blocker: String?, renders: Int) {
        let layout = seriesLayout
        if !layout.excess.isEmpty {
            return (String(localized: "\(String(layout.axes.count + layout.excess.count)) things vary: two at most. Take one away with its ✕."), 1)
        }
        switch seriesGrid(seed: seed) {
        case .failure(let e): return (Problem(e).text, 1)
        case .success(let g): return (nil, g?.count ?? 1)
        }
    }

    /// **How many jobs « Generate » queues**, and images it makes: one per cell of the series, one for a render.
    var seriesRenders: Int { seriesReading.renders }

    // MARK: Gestures

    /// **A knob bracketed** (or its values changed), at `slot`: 0 the columns, 1 the rows.
    func setBracket(_ b: Bracket, at slot: Int) {
        if let i = brackets.firstIndex(where: { $0.key == b.key }) { brackets[i] = b } else { brackets.append(b) }
        place(b.key, at: slot)
    }

    /// **The lot as the seed axis**: `count` seeds from the rack's, at `slot` — the same count as the Seeds stepper's.
    func setSeeds(_ count: Int, at slot: Int) {
        batch = max(1, count)
        if batch > 1 { place(.seeds, at: slot) }
    }

    /// **A group of words written in the text** — a word turned into a group (« Vary This Word… »), or a group's words
    /// changed — at `slot`. Fewer than two words: the group goes back to its word.
    func setGroup(_ words: [String], target: BracketTarget, at slot: Int) {
        let written = PromptAlternatives.group(words)
        switch target {
        case .key(.group(let g)):
            guard words.count > 1 else { removeAxis(.group(g)); return }
            prompt = PromptAlternatives.replacing(group: g, in: prompt, with: written)
            place(.group(g), at: slot)
        case let .words(range, before):
            guard words.count > 1, prompt == before, let r = Range(range, in: prompt) else { return }
            let g = PromptAlternatives.spans(prompt).filter { $0.range.lowerBound < r.lowerBound }.count
            // The groups after it move up a rank, and where the hand placed them follows.
            seriesPlaced = seriesPlaced.map { k in
                if case .group(let h)? = k, h >= g { return .group(h + 1) }
                return k
            }
            prompt.replaceSubrange(r, with: written)
            place(.group(g), at: slot)
        default:
            return
        }
    }

    /// **`key` placed at `slot`**: what held that place is brought back to one value (`removeAxis`).
    private func place(_ key: SeriesKey, at slot: Int) {
        let (placed, replaced) = SeriesLayout.place(key, at: slot, present: seriesPresent, placed: seriesPlaced)
        seriesPlaced = placed
        if let r = replaced, r != key { removeAxis(r) }
    }

    /// **An axis taken away** — its ✕: the lot back to 1, a group back to its first word, a knob unbracketed.
    func removeAxis(_ key: SeriesKey) {
        seriesPlaced = seriesPlaced.map { $0 == key ? nil : $0 }
        switch key {
        case .group(let g):
            prompt = PromptAlternatives.collapsing(group: g, in: prompt)
            seriesPlaced = seriesPlaced.map { k in
                if case .group(let h)? = k, h > g { return .group(h - 1) }
                return k
            }
        case .seeds:
            batch = 1
        default:
            brackets.removeAll { $0.key == key }
        }
    }

    /// **⇄**: the columns become the rows.
    func swapAxes() { seriesPlaced = seriesLayout.swapped }

    /// **The series takes a grid** — a sheet's « Reuse This Grid », a grid `silicontrol grid` queued: its groups are the
    /// text's (the caller sets the prompt as typed), its consecutive seeds the lot, its knobs bracketed, each at its place.
    /// What the app does not vary (models, steps, listed seeds, each image of an edit) is left out: returned, to be said.
    @discardableResult
    func setSeries(from g: RenderGrid) -> [RenderGrid.Axis] {
        brackets = []
        batch = 1
        var placed: [SeriesKey?] = [nil, nil]
        var left: [RenderGrid.Axis] = []
        let stack = activeLoRAs
        for (i, axis) in ([g.x] + (g.y.map { [$0] } ?? [])).enumerated() {
            var key: SeriesKey?
            switch axis {
            case .prompt(let n): key = .group(n)
            case .consecutiveSeeds(let n): batch = n; key = .seeds
            case let .loraStrength(slot, values):
                guard stack.indices.contains(slot) else { left.append(axis); continue }
                key = .loraStrength(stack[slot].path)
                brackets.append(Bracket(key: key!, numbers: values))
            case let .addedLoRAs(l, f): key = .comparedLoRAs; brackets.append(Bracket(key: .comparedLoRAs, loras: l, strength: f))
            case .startStrength(let v): key = .startStrength; brackets.append(Bracket(key: .startStrength, numbers: v))
            case .solvers(let s): key = .solver; brackets.append(Bracket(key: .solver, solvers: s))
            case .detail(let d): key = .detail; brackets.append(Bracket(key: .detail, details: d))
            case .seeds, .steps, .models, .images: left.append(axis)
            }
            placed[i] = key
        }
        seriesPlaced = placed
        return left
    }

    /// **What a bubble offers before the hand changes it**: the values bracketed, or three around the knob.
    func draftBracket(_ key: SeriesKey) -> Bracket {
        if let b = brackets.first(where: { $0.key == key }) { return b }
        switch key {
        case .loraStrength(let path):
            let v = loras.first { $0.path == path }?.strength ?? LoRASlot().strength
            return Bracket(key: key, numbers: Bracketing.around(v, step: 0.2, count: 3, in: loraStrengthRange))
        case .startStrength:
            return Bracket(key: key, numbers: Bracketing.around(startStrength, step: 0.15, count: 3, in: 0...1))
        case .solver: return Bracket(key: key, solvers: offeredSolvers)
        case .detail: return Bracket(key: key, details: Detail.allCases)
        case .comparedLoRAs:
            let others = compatibleLoras.map(\.path).filter { p in !loras.contains { $0.path == p } }
            return Bracket(key: key, loras: [""] + others.prefix(3), strength: LoRASlot().strength)
        case .group, .seeds: return Bracket(key: key)
        }
    }

    // MARK: Words

    /// **An axis in a few words** — its line beside « Generate », what the bubble says it replaces.
    func axisWords(_ key: SeriesKey) -> String {
        switch key {
        case .group(let g):
            let groups = promptGroups
            guard groups.indices.contains(g) else { return "" }
            return groups[g].map { a in
                let t = a.trimmingCharacters(in: .whitespacesAndNewlines)
                return t.isEmpty ? String(localized: "(nothing)") : t
            }.joined(separator: ", ")
        case .seeds:
            return String(localized: "\(max(1, batch)) seeds")
        case .loraStrength(let path):
            return loraName(path) + " " + (bracket(key)?.numbers ?? []).map(Figures.strength).joined(separator: " · ")
        case .comparedLoRAs:
            return (bracket(key)?.loras ?? []).map { $0.isEmpty ? String(localized: "no LoRA") : loraName($0) }.joined(separator: ", ")
        case .startStrength:
            let values = (bracket(key)?.numbers ?? []).map(StartWords.percent).joined(separator: " · ")
            return String(localized: "start \(values)")
        case .solver:
            return (bracket(key)?.solvers ?? []).map(\.choice).joined(separator: ", ")
        case .detail:
            let values = (bracket(key)?.details ?? []).map(\.choice).joined(separator: ", ")
            return String(localized: "Detail \(values)")
        }
    }

    /// The knob a key brackets, as the bubble's title names it.
    func knobName(_ key: SeriesKey) -> String {
        switch key {
        case .group: String(localized: "Words")
        case .seeds: String(localized: "Seeds")
        case .loraStrength(let path): String(localized: "Strength of \(loraName(path))")
        case .comparedLoRAs: String(localized: "LoRAs Compared")
        case .startStrength: String(localized: "Starting Image Strength")
        case .solver: String(localized: "Solver")
        case .detail: String(localized: "Detail")
        }
    }
}

// MARK: - Beside « Generate »

/// **What the series varies, in one place**: a line per axis, ↔ the columns and ↕ the rows, each with its ✕; ⇄ when there
/// are two; beyond two, the excess in orange, which keeps « Generate » off.
struct SeriesSummary: View {
    @Bindable var app: AppState

    var body: some View {
        let layout = app.seriesLayout
        if !layout.axes.isEmpty {
            VStack(alignment: .leading, spacing: 3) {
                ForEach(Array(layout.axes.enumerated()), id: \.offset) { i, key in
                    line(i, key, swap: i == 0 && layout.rows != nil)
                }
                ForEach(layout.excess, id: \.self) { key in
                    line(nil, key)
                }
            }
            .font(.caption)
        }
    }

    private func line(_ slot: Int?, _ key: SeriesKey, swap: Bool = false) -> some View {
        HStack(spacing: 6) {
            AxisMark(slot: slot).frame(width: 14)
            Text(verbatim: app.axisWords(key)).lineLimit(1).truncationMode(.middle)
                .foregroundStyle(slot == nil ? Palette.caution : .primary)
            Button { app.removeAxis(key) } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.borderless).foregroundStyle(Palette.secondary)
                .help(removeHelp(key))
                .accessibilityLabel(Text(verbatim: removeHelp(key)))
            if swap {
                Button { app.swapAxes() } label: { Image(systemName: "arrow.left.arrow.right") }
                    .buttonStyle(.borderless).foregroundStyle(Palette.secondary)
                    .help("Exchange the columns and the rows")
                    .accessibilityLabel(Text("Exchange the columns and the rows"))
            }
        }
    }

    private func removeHelp(_ key: SeriesKey) -> String {
        switch key {
        case .group: String(localized: "Back to the first word")
        case .seeds: String(localized: "Back to one image")
        default: String(localized: "No longer bracketed")
        }
    }
}
