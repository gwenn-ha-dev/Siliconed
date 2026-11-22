// « Bracket… »: the bubble that sets a series' values and its axis.

import AppKit
import Siliconed
import SwiftUI

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
    private var exists: Bool { if case .key(let k) = target { app.varies(k) } else { false } }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(verbatim: String(localized: "Bracket · \(app.knobName(key))")).font(.headline)
            values
            if !valid {
                Text(hint).font(.caption).foregroundStyle(Palette.secondary).fixedSize(horizontal: false, vertical: true)
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
                Button("Bracket", action: commit).keyboardShortcut(.defaultAction).tint(Palette.control).unavailable(valid ? nil : hint)
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
            // Which seeds, said: the fixed seed and the ones after it, or a first one drawn at the click.
            Stepper(value: $seeds, in: 2...8) {
                if app.fixedSeed {
                    Text("\(seeds) seeds: \(String(app.seed)) to \(String(app.seed &+ UInt64(seeds - 1)))").monospacedDigit()
                } else {
                    Text("\(seeds) seeds, the first drawn at random").monospacedDigit()
                }
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
                    Text("Strength").foregroundStyle(Palette.secondary)
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
                    .accessibilityLabel(Text("One more value, a step after the last"))
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
                .accessibilityLabel(Text("One more word"))
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
                        .background(Capsule().fill(on ? Palette.bracketFill.opacity(0.3) : Palette.well))
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
                .buttonStyle(.borderless).foregroundStyle(Palette.secondary)
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
                    .font(.caption).foregroundStyle(Palette.caution)
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
