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
// **Seen, never guessed** (the user, 4/11/2026: « le bracketing ne se voit pas, on ne peut pas le deviner »): the same ±, always
// shown, in one yellow (`Palette.bracket`), at the right of everything that can vary — outlined it offers, filled it varies —;
// what varies says its values under it, with the arrow of its axis (↔ the columns, ↕ the rows), in the same yellow. A ± that
// could do nothing is not drawn (« Compare LoRAs » with no other LoRA to compare), its place kept.
//
// What an analogy with the former exploration, or with a camera's bracketing, would miss:
//
// - **Finished images only** (4/11): no small format — each cell is the render « Generate » would make alone,
//   at the rack's size and steps, at its seed, and it goes to the history like any other.
// - **The lot is the seed axis** (4/11): « Batch 3 » is three seeds from the rack's (s, s+1, s+2) — drawn, or the fixed
//   one's. Its ± is on the Seed's line (the user, 4/11: « celle des graines sur la ligne des graines »). Alone, it renders as it always did, one job of three images (same pixels); beside another axis, it is a row or
//   a column of the grid. No seed is typed in the app: listed seeds are `silicontrol grid`'s.
// - **Neither the model nor the steps** (4/11): the five models are distilled for their step count, and a sweep across
//   models is `silicontrol grid`'s; so is each image of an edit alone.
// - **A bracket that no longer applies does not vary** (its LoRA switched off or removed, Detail on Anima, no starting
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

    /// **The grid « Generate » renders**, checked by the library — or `nil` when nothing varies but the lot, which renders
    /// as one job of its images, as it always did. `seed`: the base seed (the one drawn at the click for a random seed).
    func seriesGrid(seed: UInt64) -> Result<RenderGrid?, EngineError> {
        let layout = seriesLayout
        guard layout.excess.isEmpty, let columns = layout.columns, layout.axes != [.seeds] else { return .success(nil) }
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

    /// **Why the series does not render**, or `nil`: three things varying, or a grid the library refuses.
    var seriesBlocker: String? {
        let layout = seriesLayout
        if !layout.excess.isEmpty {
            return String(localized: "\(String(layout.axes.count + layout.excess.count)) things vary: two at most. Take one away with its ✕.")
        }
        if case .failure(let e) = seriesGrid(seed: seed) { return Problem(e).text }
        return nil
    }

    /// **How many jobs « Generate » queues**: one per cell of the series, one for a render and its lot.
    var seriesRenders: Int {
        if case .success(let g?) = seriesGrid(seed: seed) { return g.count }
        return 1
    }

    /// **How many images « Generate » makes**: the cells, or the lot.
    var seriesImages: Int {
        if case .success(let g?) = seriesGrid(seed: seed) { return g.count }
        return max(1, batch)
    }

    // MARK: Gestures

    /// **A knob bracketed** (or its values changed), at `slot`: 0 the columns, 1 the rows.
    func setBracket(_ b: Bracket, at slot: Int) {
        if let i = brackets.firstIndex(where: { $0.key == b.key }) { brackets[i] = b } else { brackets.append(b) }
        place(b.key, at: slot)
    }

    /// **The lot as the seed axis**: `count` seeds from the rack's, at `slot`.
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

    /// **What a bubble offers before the hand changes it**: the values bracketed, or three around the knob (4/11).
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
            return String(localized: "\(String(max(1, batch))) seeds")
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
        case .seeds: String(localized: "Seed")
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
                .foregroundStyle(slot == nil ? Color(nsColor: .systemOrange) : .primary)
            Button { app.removeAxis(key) } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help(removeHelp(key))
                .accessibilityLabel(Text(verbatim: removeHelp(key)))
            if swap {
                Button { app.swapAxes() } label: { Image(systemName: "arrow.left.arrow.right") }
                    .buttonStyle(.borderless).foregroundStyle(.secondary)
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

// MARK: - A knob bracketable

/// **An axis, as an arrow**: ↔ the columns, ↕ the rows, in the series' yellow; beyond two axes, the orange warning.
struct AxisMark: View {
    let slot: Int?

    var body: some View {
        Image(systemName: slot == 0 ? "arrow.left.and.right" : slot == 1 ? "arrow.up.and.down" : "exclamationmark.triangle.fill")
            .foregroundStyle(slot == nil ? Color(nsColor: .systemOrange) : Palette.bracket)
            .help(slot == 0 ? String(localized: "Columns") : slot == 1 ? String(localized: "Rows")
                  : String(localized: "Two axes at most: this one does not render"))
    }
}

/// **The series' mark, at the right of what can vary**: always shown, in the series' yellow — outlined it offers, filled the knob
/// varies. A click opens the bubble. One that could do nothing is not drawn, its place kept (the user, 4/11/2026: « autant la
/// cacher »).
struct BracketMark: View {
    @Bindable var app: AppState
    let key: SeriesKey
    /// The column the marks stand in, at the right of a box's settings: every control before it ends on one line, marked or not.
    static let gutter: CGFloat = 20

    var body: some View {
        let varies = app.varies(key)
        if varies || app.canBracket(key) {
            Button { app.bracketing = key } label: {
                Image(systemName: varies ? "plusminus.circle.fill" : "plusminus.circle")
                    .foregroundStyle(Palette.bracket)
            }
            .buttonStyle(.borderless)
            .help(varies ? String(localized: "Bracketed: Generate renders each value. Click to change them.")
                         : String(localized: "Bracket: Generate renders this setting at several values"))
            .accessibilityLabel(Text("Bracket…"))
        }
    }
}

extension View {
    /// **A knob that can be bracketed**: its ± at its right, in `gutter` (and right-click › « Bracket… » with `menu`); bracketed,
    /// its values under it with the arrow of their axis, a click on them reopens the bubble.
    func bracketable(_ app: AppState, _ key: SeriesKey, menu: Bool = true, gutter: CGFloat = BracketMark.gutter) -> some View {
        modifier(Bracketable(app: app, key: key, menu: menu, gutter: gutter))
    }
}

private struct Bracketable: ViewModifier {
    @Bindable var app: AppState
    let key: SeriesKey
    let menu: Bool
    let gutter: CGFloat

    func body(content: Content) -> some View {
        let slot = app.seriesLayout.slot(of: key)
        withMenu(VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 0) {
                content
                BracketMark(app: app, key: key).frame(width: gutter, alignment: .trailing)
            }
            if app.varies(key) {
                Button { app.bracketing = key } label: {
                    HStack(spacing: 4) {
                        AxisMark(slot: slot)
                        Text(verbatim: app.axisWords(key)).lineLimit(1).truncationMode(.middle)
                            .foregroundStyle(slot == nil ? Color(nsColor: .systemOrange) : Palette.bracket)
                    }
                    .font(.caption)
                }
                .buttonStyle(.plain)
                .help("Bracketed: Generate renders each value")
            }
        })
        .popover(isPresented: Binding(get: { app.bracketing == key },
                                      set: { if !$0, app.bracketing == key { app.bracketing = nil } }),
                 arrowEdge: .trailing) {
            BracketBubble(app: app, target: .key(key)) { app.bracketing = nil }
                // No value selected on opening: the popover gave its first field the focus, its text selected in the system's blue.
                .onAppear { DispatchQueue.main.async { NSApp.keyWindow?.makeFirstResponder(nil) } }
        }
    }

    /// Right-click › « Bracket… » where the knob has no menu of its own (a LoRA's card has one, which says it).
    @ViewBuilder private func withMenu<V: View>(_ v: V) -> some View {
        if menu { v.contextMenu { Button("Bracket…") { app.bracketing = key } } } else { v }
    }
}

// MARK: - The bubble

/// **« Bracket… »**: the values as pills, each with its ✕, a + — for strengths a step and a count that make them again
/// around the knob's value —, the axis they make (and what it replaces), Remove, Bracket.
struct BracketBubble: View {
    @Bindable var app: AppState
    let target: BracketTarget
    let close: () -> Void

    @State private var draft: Bracket
    @State private var words: [String]
    @State private var seeds: Int
    @State private var step: Double
    @State private var slot: Int

    init(app: AppState, target: BracketTarget, close: @escaping () -> Void) {
        self.app = app
        self.target = target
        self.close = close
        let key: SeriesKey
        var words: [String] = []
        switch target {
        case .key(let k):
            key = k
            if case .group(let g) = k, app.promptGroups.indices.contains(g) { words = app.promptGroups[g] }
        case let .words(range, prompt):
            key = .group(-1)
            words = [(prompt as NSString).substring(with: range), "", ""]
        }
        _draft = State(initialValue: app.draftBracket(key))
        _words = State(initialValue: words)
        _seeds = State(initialValue: app.batch > 1 ? app.batch : 3)
        if case .startStrength = key { _step = State(initialValue: 0.15) } else { _step = State(initialValue: 0.2) }
        let layout = app.seriesLayout
        _slot = State(initialValue: layout.slot(of: key) ?? layout.freeSlot)
    }

    private var key: SeriesKey { draft.key }
    private var isWords: Bool { if case .group = key { true } else { false } }
    private var exists: Bool { if case .key(let k) = target { app.varies(k) } else { false } }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(verbatim: String(localized: "Bracket · \(app.knobName(key))")).font(.headline)
            values
            if !valid {
                Text(hint).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            axis
            HStack {
                if exists {
                    Button("Remove") {
                        if case .key(let k) = target { app.removeAxis(k) }
                        close()
                    }
                }
                Spacer()
                Button("Cancel", action: close).keyboardShortcut(.cancelAction)
                Button("Bracket", action: commit).keyboardShortcut(.defaultAction).tint(Palette.control).disabled(!valid)
            }
        }
        .padding(14)
        .frame(width: 340)
    }

    // ── the values ──

    @ViewBuilder private var values: some View {
        switch key {
        case .group: wordPills
        case .seeds:
            Stepper(value: $seeds, in: 2...8) {
                Text("\(String(seeds)) seeds from the rack’s").monospacedDigit()
            }
            .help("The seeds that follow the rack’s seed: drawn at the click, or the fixed one")
        case .loraStrength, .startStrength: numberPills
        case .solver:
            choicePills(app.offeredSolvers, label: \.choice, selected: Binding(get: { draft.solvers }, set: { draft.solvers = $0 }))
        case .detail:
            choicePills(Detail.allCases, label: \.choice, selected: Binding(get: { draft.details }, set: { draft.details = $0 }))
        case .comparedLoRAs:
            VStack(alignment: .leading, spacing: 8) {
                choicePills([""] + app.compatibleLoras.map(\.path), label: { $0.isEmpty ? String(localized: "No LoRA") : app.loraName($0) },
                            selected: Binding(get: { draft.loras }, set: { draft.loras = $0 }))
                HStack {
                    Text("Strength").foregroundStyle(.secondary)
                    Slider(value: $draft.strength, in: app.loraStrengthRange).controlSize(.small).tint(Palette.control)
                    Text(verbatim: Figures.strength(draft.strength)).monospacedDigit().frame(width: 36, alignment: .trailing)
                }
                .font(.callout)
            }
        }
    }

    private var isPercent: Bool { if case .startStrength = key { true } else { false } }
    private var range: ClosedRange<Double> { isPercent ? 0...1 : app.loraStrengthRange }
    private var center: Double {
        switch key {
        case .startStrength: app.startStrength
        case .loraStrength(let path): app.loras.first { $0.path == path }?.strength ?? LoRASlot().strength
        default: 0
        }
    }

    private var numberPills: some View {
        VStack(alignment: .leading, spacing: 8) {
            FlowRow {
                ForEach(Array(draft.numbers.enumerated()), id: \.offset) { i, _ in
                    pill {
                        numberField(i)
                    } remove: { draft.numbers.remove(at: i) }
                }
                Button { addNumber() } label: { Image(systemName: "plus") }
                    .buttonStyle(.bordered).controlSize(.small)
                    .help("One more value, a step after the last")
            }
            HStack(spacing: 14) {
                Stepper(value: Binding(get: { step }, set: { step = max(0.05, min(0.5, ($0 * 100).rounded() / 100)); regenerate() }),
                        in: 0.05...0.5, step: 0.05) {
                    Text(verbatim: String(localized: "Step \(isPercent ? StartWords.percent(step) : Figures.strength(step))")).monospacedDigit()
                }
                Stepper(value: Binding(get: { draft.numbers.count }, set: { regenerate(count: $0) }), in: 2...9) {
                    Text(verbatim: String(localized: "Count \(String(draft.numbers.count))")).monospacedDigit()
                }
            }
            .font(.callout)
            .help("The values again, around the knob’s")
        }
    }

    /// A value as the knob says it: a percentage for the start's strength, two decimals for a LoRA's.
    @ViewBuilder private func numberField(_ i: Int) -> some View {
        if isPercent {
            TextField(value: number(i), format: FloatingPointFormatStyle<Double>.Percent().precision(.fractionLength(0))) { EmptyView() }
                .labelsHidden().textFieldStyle(.plain).monospacedDigit().frame(width: 40)
        } else {
            TextField(value: number(i), format: FloatingPointFormatStyle<Double>().precision(.fractionLength(2))) { EmptyView() }
                .labelsHidden().textFieldStyle(.plain).monospacedDigit().frame(width: 36)
        }
    }

    private func number(_ i: Int) -> Binding<Double> {
        Binding(get: { draft.numbers.indices.contains(i) ? draft.numbers[i] : 0 },
                set: { v in
                    guard draft.numbers.indices.contains(i), v.isFinite else { return }
                    draft.numbers[i] = min(max((v * 100).rounded() / 100, range.lowerBound), range.upperBound)
                })
    }

    private func addNumber() {
        let next = min(max(((draft.numbers.last ?? center) + step) * 100, range.lowerBound * 100), range.upperBound * 100).rounded() / 100
        if !draft.numbers.contains(next) { draft.numbers.append(next) }
    }

    private func regenerate(count: Int? = nil) {
        draft.numbers = Bracketing.around(center, step: step, count: count ?? max(1, draft.numbers.count), in: range)
    }

    private var wordPills: some View {
        FlowRow {
            ForEach(Array(words.enumerated()), id: \.offset) { i, _ in
                pill {
                    TextField("word", text: Binding(get: { words.indices.contains(i) ? words[i] : "" },
                                                    set: { if words.indices.contains(i) { words[i] = $0 } }))
                        .textFieldStyle(.plain).frame(minWidth: 50, maxWidth: 140).fixedSize()
                } remove: { words.remove(at: i) }
            }
            Button { words.append("") } label: { Image(systemName: "plus") }
                .buttonStyle(.bordered).controlSize(.small)
                .help("One more word")
        }
    }

    private func choicePills<T: Hashable>(_ all: [T], label: @escaping (T) -> String, selected: Binding<[T]>) -> some View {
        FlowRow {
            ForEach(all, id: \.self) { value in
                let on = selected.wrappedValue.contains(value)
                Button {
                    // In the order offered: the cells' order is the list's.
                    if on { selected.wrappedValue.removeAll { $0 == value } }
                    else { selected.wrappedValue = all.filter { selected.wrappedValue.contains($0) || $0 == value } }
                } label: {
                    Text(verbatim: label(value)).lineLimit(1)
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(Capsule().fill(on ? Palette.bracket.opacity(0.25) : Palette.well))
                        .overlay(Capsule().strokeBorder(on ? Palette.bracket : Palette.hairline))
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func pill<Content: View>(@ViewBuilder _ content: () -> Content, remove: @escaping () -> Void) -> some View {
        HStack(spacing: 3) {
            content()
            Button(action: remove) { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .accessibilityLabel(Text("Remove"))
        }
        .padding(.leading, 8).padding(.trailing, 4).padding(.vertical, 3)
        .background(Capsule().fill(Palette.well))
    }

    // ── the axis ──

    private var axis: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("Axis", selection: $slot) {
                Label("Columns", systemImage: "arrow.left.and.right").tag(0)
                Label("Rows", systemImage: "arrow.up.and.down").tag(1)
            }
            .pickerStyle(.segmented).tint(Palette.control)
            if let r = replaced {
                Text(verbatim: String(localized: "Replaces: \(app.axisWords(r))"))
                    .font(.caption).foregroundStyle(Color(nsColor: .systemOrange))
            }
        }
    }

    /// What this place holds now, which the bracket would push out.
    private var replaced: SeriesKey? {
        let placing: SeriesKey = { if case .key(let k) = target { k } else { .group(-1) } }()
        let r = SeriesLayout.place(placing, at: slot, present: app.seriesPresent, placed: app.seriesPlaced).replaced
        return r == placing ? nil : r
    }

    // ── done ──

    /// Why « Bracket » waits.
    private var hint: String {
        if case .comparedLoRAs = key { return String(localized: "Choose a LoRA to compare with the image without.") }
        return String(localized: "Two values at least: one alone is the setting itself.")
    }

    private var cleanWords: [String] { words.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty } }

    /// Two values or more: one is the knob itself (`Bracket.values`).
    private var valid: Bool {
        switch key {
        case .group: cleanWords.count > 1
        case .seeds: seeds > 1
        default: draft.values > 1
        }
    }

    private func commit() {
        guard valid else { return }
        switch key {
        case .group: app.setGroup(cleanWords, target: target, at: slot)
        case .seeds: app.setSeeds(seeds, at: slot)
        default: app.setBracket(draft, at: slot)
        }
        close()
    }
}

/// Pills laid out in rows, wrapping at the bubble's width.
private struct FlowRow: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(proposal.width ?? 320, subviews)
        let height = rows.map(\.height).reduce(0, +) + spacing * CGFloat(max(0, rows.count - 1))
        return CGSize(width: proposal.width ?? rows.map(\.width).max() ?? 0, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in arrange(bounds.width, subviews) {
            var x = bounds.minX
            for i in row.items {
                let size = subviews[i].sizeThatFits(.unspecified)
                subviews[i].place(at: CGPoint(x: x, y: y + (row.height - size.height) / 2), proposal: .unspecified)
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private func arrange(_ width: CGFloat, _ subviews: Subviews) -> [(items: [Int], width: CGFloat, height: CGFloat)] {
        var rows: [(items: [Int], width: CGFloat, height: CGFloat)] = []
        var current: [Int] = [], x: CGFloat = 0, height: CGFloat = 0
        for i in subviews.indices {
            let size = subviews[i].sizeThatFits(.unspecified)
            if !current.isEmpty, x + size.width > width {
                rows.append((current, x - spacing, height))
                current = []; x = 0; height = 0
            }
            current.append(i); x += size.width + spacing; height = max(height, size.height)
        }
        if !current.isEmpty { rows.append((current, x - spacing, height)) }
        return rows
    }
}

// MARK: - From the text

/// **The text's ±, at the card's right**, under its ✕: the selection, or the word at the caret, becomes a group — or the group the caret is
/// in opens; filled once the text has a group. The same mark as the rack's, so that the text reads as something that varies too.
struct PromptBracketMark: View {
    @Bindable var app: AppState
    var disabled = false

    var body: some View {
        let varies = !app.promptGroups.isEmpty
        Button {
            if PromptTextView.current?.vary() != true { NSSound.beep() }
        } label: {
            Image(systemName: varies ? "plusminus.circle.fill" : "plusminus.circle")
                .foregroundStyle(Palette.bracket)
        }
        .buttonStyle(.borderless)
        .disabled(disabled || app.prompt.isEmpty)
        .help("Vary a word: select it, or put the caret in it, then click — Generate renders each word you give")
        .accessibilityLabel(Text("Vary This Word…"))
    }
}

/// **The bubble over the text**: for words selected (« Vary This Word… ») or a group's chip clicked — an AppKit popover,
/// the field being AppKit's.
@MainActor
enum BracketPopover {
    private static var shown: NSPopover?

    static func show(_ app: AppState, target: BracketTarget, relativeTo rect: NSRect, of view: NSView) {
        shown?.close()
        let popover = NSPopover()
        popover.behavior = .transient
        let host = NSHostingController(rootView: BracketBubble(app: app, target: target) { [weak popover] in popover?.close() })
        host.sizingOptions = [.preferredContentSize]
        popover.contentViewController = host
        popover.show(relativeTo: rect, of: view, preferredEdge: .maxY)
        shown = popover
    }
}
