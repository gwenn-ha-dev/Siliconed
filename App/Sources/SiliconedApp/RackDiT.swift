// The DiT's box, the stage with choices, and the LoRA stack it holds. Split from
// `Rack.swift`, whose header says how the boxes behave.

import AppKit
import Siliconed
import SwiftUI

// MARK: - The DiT: the stage with choices

/// **The DiT and everything it is told**: the model, its license, the size, the seed, how many
/// images (the steps only off the model's recipe), the LoRA stack — then what it costs, or its steps while it runs.
struct DiTBox: View {
    @Bindable var app: AppState
    @State private var customSize = false
    /// The seed's field, typed or greyed: ten digits, the most a seed drawn has (`Request.drawnSeed`).
    private static let seedWidth: CGFloat = 100

    var body: some View {
        StageBox(title: "DiT", symbol: "engine.combustion", kind: .latent, state: app.state(of: .denoising),
                 explanation: "The model that draws (a diffusion transformer): from noise, step by step, it makes the image the prompt describes") {
            // The model, and on the same line the library's door, at the right where every line of the box ends — before the
            // column of the ± marks (`BracketMark.gutter`), as « Add » under it: the rack's header that held it is gone.
            HStack(spacing: 8) {
                Picker("Model", selection: $app.identifier) {
                    ForEach(app.visibleCards) { card in
                        if app.readySet.contains(card.id) {
                            Text(card.name).tag(card.id)
                        } else {
                            Text("\(card.name) (not installed)").tag(card.id)
                        }
                    }
                }
                .labelsHidden()
                .help("The denoiser: the publisher’s model, or a fine-tune you imported")
                Spacer(minLength: 0)
                Button("Models…") { app.managementOpen = true }
                    .buttonStyle(.chartLink)
                    .font(.callout)
                    .help("Install, import, remove — and what each takes on disk (⇧⌘M)")
                    .fixedSize()
            }
            .padding(.trailing, BracketMark.gutter)
            if let card = app.card {
                if app.missing != nil {
                    HStack {
                        Text("Not installed").font(.callout).foregroundStyle(Palette.secondary)
                        Spacer()
                        Button("Install…") { app.installSelectedModel() }
                            .controlSize(.small)
                            .unavailable(app.managementRefusal)
                    }
                } else {
                    // The labels on one line when they fit; else what is stored and what computes stay together on a line of
                    // their own, the license and the latent on another, then each pair alone — a label is never truncated.
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 5) { license(card); weightsFormat; latent }
                        VStack(alignment: .leading, spacing: 5) {
                            HStack(spacing: 5) { weightsFormat }
                            HStack(spacing: 5) { license(card); latent }
                        }
                        VStack(alignment: .leading, spacing: 5) {
                            HStack(spacing: 5) { weightsFormat }
                            license(card)
                            latent
                        }
                    }
                    Divider().padding(.vertical, 2)
                    // The LoRA first: what changes the image most, under the model it modifies.
                    loras
                    Divider().padding(.vertical, 2)
                    settings
                }
            }
            if app.state(of: .denoising) == .active, app.totalSteps > 0, app.currentJob?.settings.identifier == app.identifier {
                progress
            } else {
                Cost(app.stageCost(.denoising))
            }
        }
    }

    private func license(_ card: ModelCard) -> some View {
        Tag(text: Text(verbatim: card.licenseName),
            symbol: card.license.commercial ? "checkmark.seal" : "info.circle")
            .help(card.license.commercial ? String(localized: "Commercial use allowed")
                  : card.imagesCommercial ? String(localized: "Non-commercial model: personal and research use only. Its images may be used commercially.")
                  : String(localized: "Non-commercial: personal and research use only"))
    }

    @ViewBuilder private var weightsFormat: some View {
        WeightsTag(format: app.weightFormats?.dit)
    }

    @ViewBuilder private var latent: some View {
        if let space = app.chain?.denoising.space {
            Tag(text: Text("latent · \(space.channels) ch"), dot: WireKind.latent.color)
                .help(Text(verbatim: "\(space.name) · ÷\(space.factor)"))
        }
    }

    // ── Size in two scales (Beta): a proportion, then a resolution; the table is `SizeScales` ──

    /// The two menus and the pixels they give. The shown choice is read from the form's size, so a size typed by hand, or
    /// restored with an image, shows « Custom » instead of a proportion it is not. A greyed entry says why (the model's
    /// limit), as the default menu does.
    @ViewBuilder private var scales: some View {
        let current = app.sizeCell
        let resolution = current?.resolution ?? .oneK
        let ratio = current?.ratio ?? SizeRatio(1, 1)
        Field(title: "Size") {
            Picker("Proportion", selection: Binding<SizeRatio?>(
                get: { current?.ratio },
                set: { if let r = $0 { app.selectScale(r, resolution) } })) {
                if current == nil { Text("Custom").tag(SizeRatio?.none) }
                ForEach(Array([[SizeRatio(1, 1)], SizeScales.portraits, SizeScales.landscapes].enumerated()), id: \.offset) { group in
                    if group.offset > 0 { Divider() }
                    ForEach(group.element, id: \.self) { r in
                        let cell = SizeScales.cell(r, resolution)!
                        scaleItem(cell, title: ratioTitle(cell)).tag(Optional(r))
                    }
                }
            }
            .labelsHidden().fixedSize()
            .help(approximateHelp(current))
            Picker("Resolution", selection: Binding<SizeResolution?>(
                get: { current?.resolution },
                set: { if let q = $0 { app.selectScale(ratio, q) } })) {
                if current == nil { Text("Custom").tag(SizeResolution?.none) }
                ForEach(SizeResolution.allCases, id: \.self) { q in
                    scaleItem(SizeScales.cell(ratio, q)!, title: q.rawValue).tag(Optional(q))
                }
            }
            .labelsHidden().fixedSize()
            .help("1K is about 1 megapixel, 2K about 4")
            Text(verbatim: Figures.size(app.format.width, app.format.height)).monospacedDigit().foregroundStyle(Palette.secondary)
        } accessory: {
            sizePencil
        }
    }

    /// The pencil of a size typed by hand, in the marks' column.
    private var sizePencil: some View {
        Button { customSize = true } label: { Image(systemName: "pencil") }
            .buttonStyle(.borderless)
            .help("Type a size (width × height)")
            .accessibilityLabel(Text("Type a size (width × height)"))
            .popover(isPresented: $customSize, arrowEdge: .trailing) { CustomSize(app: app) }
    }

    /// An entry of either menu: its title, or why it is greyed.
    @ViewBuilder private func scaleItem(_ cell: SizeCell, title: String) -> some View {
        if let why = app.scaleUnavailable(cell) {
            Text(verbatim: "\(title) · \(why)").disabled(true)
        } else {
            Text(verbatim: title)
        }
    }

    /// « 3:4 », or « ≈ 9:16 » where the pixels at this resolution are not quite that proportion.
    private func ratioTitle(_ cell: SizeCell) -> String {
        cell.isApproximate ? "≈ \(cell.ratio)" : cell.ratio.description
    }

    /// The footnote of an approximate proportion, as a tooltip.
    private func approximateHelp(_ cell: SizeCell?) -> Text {
        guard let cell, cell.isApproximate else { return Text("The proportion of the size; the resolution is next to it") }
        let pixels = Figures.size(cell.width, cell.height)
        let really = cell.actualRatio.description, asked = cell.ratio.description
        let off = cell.deviation.formatted(.percent.precision(.fractionLength(1)))
        return Text("\(pixels) is really \(really), \(off) off \(asked): an exact \(asked) in multiples of 32 has no size near this resolution.")
    }

    // ── what the DiT is told: size, seed, how many, steps ──

    @ViewBuilder private var settings: some View {
        // The size is chosen unless an image is edited: references or cited subjects alone take the menu's (`framingImage`).
        if !app.edits, app.sizeScalesBeta {
            scales
        } else if !app.edits {
            Field(title: "Size") {
                Picker("Orientation", selection: $app.orientation) {
                    ForEach([RecommendedFormat.Orientation.square, .portrait, .landscape], id: \.self) { o in
                        Image(systemName: o.symbol).help(o.label).accessibilityLabel(Text(verbatim: o.label)).tag(o)
                    }
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize().tint(Palette.control)
                Picker("Size", selection: $app.format) {
                    ForEach(app.pickerFormats, id: \.self) { f in
                        Text(verbatim: Figures.size(f.width, f.height)).tag(f)
                    }
                }
                .labelsHidden()
            } accessory: {
                sizePencil
            }
        } else {
            let f = app.outputFormat
            Field(title: "Size") {
                Text(verbatim: Figures.size(f.width, f.height)).monospacedDigit()
            } accessory: {
                Image(systemName: "lock.fill").imageScale(.small).foregroundStyle(Palette.secondary)
            }
            .help("An edit keeps the framing of image 1: about 1 megapixel at its proportions, each side at least 512, 1024² at most")
        }
        // The lot is the series' seed axis: its ± on the Seed's line, choosing how many seeds and where they go — the same count
        // as the Seeds stepper under.
        // **Fixed, a box** (issue 17): two states, one click. Off, the field shows the seed the last render drew (`render`),
        // « — » before the first one, greyed with the dice, a text that selects and copies (a disabled field does neither);
        // on, that seed stays — the same image again.
        Field(title: "Seed", app: app, bracket: .seeds) {
            if app.fixedSeed {
                TextField("Seed", value: $app.seed, format: .number.grouping(.never))
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .frame(width: Self.seedWidth)
            } else {
                Text(verbatim: app.seedShown ? String(app.seed) : "—")
                    .monospacedDigit()
                    .foregroundStyle(Palette.secondary)
                    .textSelection(.enabled)
                    .lineLimit(1)
                    .padding(.horizontal, 4)
                    .frame(width: Self.seedWidth, height: 22, alignment: .trailing)
                    .background(RoundedRectangle(cornerRadius: 5).fill(Color.primary.opacity(0.05)))
                    .help(app.seedShown ? String(localized: "The seed the last render drew: select it to copy it, check Fixed to keep it")
                                        : String(localized: "No seed drawn yet: the next render draws one, check Fixed to choose it"))
            }
            Button { app.seed = Request.drawnSeed() } label: { Image(systemName: "dice") }
                .buttonStyle(.borderless)
                .help("Draw a seed")
                .accessibilityLabel(Text("Draw a seed"))
                .unavailable(app.fixedSeed ? nil : String(localized: "Check Fixed to choose the seed: unchecked, each render draws its own"))
            Toggle("Fixed", isOn: $app.fixedSeed)
                .toggleStyle(.checkbox).fixedSize()
                .help("Unchecked: a new seed for every render. Checked: the same seed, the same image again")
        }
        if !app.variations.isEmpty {
            Field(title: "Variation") {
                Text(verbatim: app.variations.map { VariationWords.label($0.strength) }.joined(separator: " + "))
            } accessory: {
                Button { app.variations = [] } label: { Image(systemName: "xmark.circle.fill") }
                    .buttonStyle(.borderless).foregroundStyle(Palette.secondary)
                    .help("Remove the variation: the seed’s own image")
                    .accessibilityLabel(Text("Remove the variation: the seed’s own image"))
            }
            .help("The next render is a variation of the seed’s image, like the image it came from")
        }
        // **Seeds**: how many, the one name the window gives the count, here and in the Seed's ± (`setSeeds`). Above 1 it is the
        // series' seed axis (consecutive seeds from the Seed's), its line beside Generate with its ✕ and ⇄; from the terminal,
        // `silicontrol bracket seed` opens the Seed's bubble.
        Field(title: "Seeds") {
            Stepper(value: $app.batch, in: 1...8) { Text(verbatim: "\(app.batch)").monospacedDigit() }
                .fixedSize()
        }
        .help("How many seeds, one image each: consecutive seeds from the Seed’s, the prompt encoded only once")
        // **The steps, only off the recipe**: the five models are distilled for 8, and fewer is softer, more is longer and no
        // better — a choice that only costs. The line comes when something else set them (a PNG's settings reused,
        // `silicontrol`), as the variation's does, with its way back.
        // With Choose Steps on (Beta), the line is always there and the count is the hand's; a count the model has no schedule
        // for greys Generate with the counts it has (`blocker`), as for any other source.
        let recipe = app.card?.defaultSteps ?? 8
        if app.stepsBeta {
            Field(title: "Steps") {
                Stepper(value: $app.steps, in: 1...Request.maximumSteps) { Text(verbatim: "\(app.steps)").monospacedDigit() }
                    .fixedSize()
            } accessory: {
                if app.steps != recipe { backToRecipe(recipe) }
            }
            .help(String(localized: "\(recipe) steps: the count this model is distilled for. Fewer for a LoRA distilled to fewer steps"))
        } else if app.steps != recipe {
            Field(title: "Steps") {
                Text(verbatim: "\(app.steps)").monospacedDigit()
            } accessory: {
                backToRecipe(recipe)
            }
            .help(String(localized: "\(recipe) steps: the count this model is distilled for"))
        }
        // Not on a model that offers none (`DenoisingModule.acceptsDetail`): a choice that changes nothing is not shown.
        if app.offersDetail {
            Field(title: "Detail", app: app, bracket: .detail) {
                Picker("Detail", selection: $app.detail) {
                    ForEach(Detail.allCases, id: \.self) { Text(verbatim: $0.choice).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize().tint(Palette.control)
            }
            .help("Adds fine texture and small details; Most can over-sharpen")
        }
        // More Solvers, the models in front: how a step moves to the next. Euler is the distillation's own.
        if app.offeredSolvers.count > 1 {
            Field(title: "Solver", app: app, bracket: .solver) {
                Picker("Solver", selection: $app.solver) {
                    ForEach(app.offeredSolvers, id: \.self) { Text(verbatim: $0.choice).tag($0) }
                }
                .labelsHidden().fixedSize()
            }
            .help("How each step moves to the next, with the same steps: Euler is the one this model was distilled for")
        }
        Toggle("Preview every step", isOn: $app.previews)
            .font(.callout).tint(Palette.control)
            .help("Shows the image taking shape on the canvas after each step. It does not change the image.")
    }

    /// The ✕ that brings the steps back to the model's own count, in the marks' column: there or not, the count stays put.
    private func backToRecipe(_ recipe: Int) -> some View {
        Button { app.steps = recipe } label: { Image(systemName: "xmark.circle.fill") }
            .buttonStyle(.borderless).foregroundStyle(Palette.secondary)
            .help(String(localized: "Back to \(recipe) steps, the count this model is distilled for"))
            .accessibilityLabel(Text("Back to \(recipe) steps, the count this model is distilled for"))
    }

    /// **The LoRA stack, in the open** — as reachable as the steps or the seed: a title with one « Add » (this model's LoRAs,
    /// then Import…), then one card per LoRA: on or off, its strength under the hand, its trigger words a click away from
    /// the prompt. Nothing hides behind a popover or a hover.
    @ViewBuilder private var loras: some View {
        VStack(alignment: .leading, spacing: 6) {
            let header = HStack(spacing: 8) {
                Text("LoRA").foregroundStyle(Palette.secondary)
                if !app.loras.isEmpty {
                    Text(verbatim: "\(app.loras.count)/\(AppState.maxLoRA)").monospacedDigit().foregroundStyle(Palette.secondary)
                }
                Spacer(minLength: 4)
                LoRAAddMenu(app: app)
            }
            .font(.callout)
            // The title's ±: « Compare LoRAs », one of them per image beside the image without — where this model has some.
            if app.applies(.comparedLoRAs) {
                header.bracketable(app, .comparedLoRAs, menu: false)
            } else {
                header.padding(.trailing, BracketMark.gutter)
            }
            if app.compatibleLoras.isEmpty, app.loras.isEmpty {
                Text("No LoRA for this model").font(.caption).foregroundStyle(Palette.secondary)
            }
            ForEach($app.loras) { $slot in LoRARow(app: app, slot: $slot) }
        }
    }

    /// **The steps, on the box itself**: one segment per DiT evaluation, and the time left. The
    /// latent's preview is the canvas's, not repeated here.
    private var progress: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text("step \(app.currentStep)/\(app.totalSteps)").monospacedDigit()
                Spacer()
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    if app.compilingCoreAI {
                        Text("Preparing for this Mac, once…").foregroundStyle(Palette.secondary)
                    } else if let r = app.remaining(at: context.date) {
                        Text("\(Figures.duration(r)) left").monospacedDigit().foregroundStyle(Palette.secondary)
                    }
                }
            }
            .font(.caption)
            if app.totalSteps <= 60 {
                HStack(spacing: 2) {
                    ForEach(1...app.totalSteps, id: \.self) { i in
                        Capsule().fill(i <= app.currentStep ? Palette.live : Palette.well).frame(height: 4)
                    }
                }
                .animation(Motion.pick(.easeOut(duration: 0.3)), value: app.currentStep)
            }
        }
    }
}

/// **« Add »**, the stack's one way in: the model's LoRAs (those in the stack greyed; all greyed at `AppState.maxLoRA`, the
/// limit), then Import… for a file from Civitai — the same import as File › Import a Model or LoRA… (`chooseToImport`).
/// The pick goes in at 0.8, on. Greyed only when it would open on nothing: no LoRA for this model and no import possible.
struct LoRAAddMenu: View {
    let app: AppState

    var body: some View {
        let used = Set(app.loras.map(\.path))
        Menu {
            ForEach(app.compatibleLoras) { l in
                Button(l.name) { app.loras.append(LoRASlot(path: l.path)) }
                    .disabled(used.contains(l.path) || !app.canAddLoRA)
            }
            if !app.compatibleLoras.isEmpty { Divider() }
            Button("Import…") { app.chooseToImport() }
                .disabled(!app.importPossible)
            // A series: one of them added per image, and the image without (`SeriesKey.comparedLoRAs`).
            if app.compatibleLoras.count > 1 {
                Divider()
                Button("Compare LoRAs…") { app.bracketing = .comparedLoRAs }
                    .disabled(!app.canBracket(.comparedLoRAs))
            }
        } label: {
            Label("Add", systemImage: "plus")
        }
        .menuStyle(.button).buttonStyle(.bordered).controlSize(.small).fixedSize()
        .help(String(localized: "A LoRA of this model (\(AppState.maxLoRA) at most), or one to import: a .safetensors file, as published on Civitai"))
        .unavailable(app.compatibleLoras.isEmpty ? app.importRefusal : nil)
    }
}

/// **A LoRA of the stack, a card**: on top, its switch, its name and the button that takes it out; under them its strength,
/// a slider and the figure, which can be typed; then its trigger words when its file gives them — without them many LoRAs
/// do nothing, and a hover was the only place they showed: a click puts them at the head of the prompt. Off, the card stays,
/// greyed, its strength kept. Right-click: Replace With, Reset to 0.8, Remove.
struct LoRARow: View {
    let app: AppState
    @Binding var slot: LoRASlot

    /// The card's right column: its ✕, the strength's ±.
    private static let gutter: CGFloat = 22

    var body: some View {
        let card = app.loraCard(slot.path)
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Toggle(isOn: $slot.on) { EmptyView() }
                    .toggleStyle(.checkbox).labelsHidden().tint(Palette.control)
                    .help("On, it goes into the render. Off, it stays here with its strength, out of the render.")
                    .accessibilityLabel(Text("Use this LoRA"))
                Text(card?.name ?? String(localized: "Choose…"))
                    .fontWeight(.medium).lineLimit(1).truncationMode(.middle)
                    .help(details)
                Spacer(minLength: 4)
                // In the column of the strength's ± under it: the same width, against the same edge.
                Button(action: remove) {
                    Image(systemName: "xmark.circle.fill").frame(width: Self.gutter, height: 20, alignment: .trailing).contentShape(Rectangle())
                }
                .buttonStyle(.borderless).foregroundStyle(Palette.secondary)
                .help("Remove this LoRA from the stack")
                .accessibilityLabel(Text("Remove this LoRA from the stack"))
            }
            HStack(spacing: 8) {
                Slider(value: $slot.strength, in: app.loraStrengthRange)
                    .controlSize(.small).tint(Palette.control)
                    .accessibilityLabel(Text("Strength"))
                TextField(value: strength, format: .number.precision(.fractionLength(2))) { Text("Strength") }
                    .labelsHidden().textFieldStyle(.roundedBorder).controlSize(.small).monospacedDigit()
                    .multilineTextAlignment(.trailing).frame(width: 48)
                    .help("Strength: 0.8 attenuates, 1 as trained, more emphasizes")
            }
            // Its ± at the right of the strength it varies, under the card's ✕.
            .bracketable(app, .loraStrength(slot.path), menu: false, gutter: Self.gutter)
            .padding(.leading, 22)
            .opacity(slot.on ? 1 : 0.45)
            if let card, !card.triggers.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(card.triggers, id: \.self) { trigger in TriggerButton(app: app, trigger: trigger) }
                }
                .padding(.leading, 22)
                .opacity(slot.on ? 1 : 0.45)
            }
        }
        .contextMenu {
            Menu("Replace With") {
                ForEach(app.compatibleLoras) { l in
                    Button(l.name) { slot.path = l.path }
                        .disabled(l.path == slot.path || app.loras.contains { $0.path == l.path })
                }
            }
            Button("Reset to 0.8") { slot.strength = LoRASlot().strength }
            Button("Bracket…") { app.bracketing = .loraStrength(slot.path) }
            Divider()
            Button("Remove", action: remove)
        }
        .font(.callout)
        .padding(.horizontal, 8).padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Palette.well))
    }

    /// The figure, typed: held within the slider's range.
    private var strength: Binding<Double> {
        let range = app.loraStrengthRange
        return Binding(get: { slot.strength }, set: { slot.strength = min(max($0, range.lowerBound), range.upperBound) })
    }

    private func remove() {
        // The id is read BEFORE: `slot` is a binding on `app.loras`, and re-reading it
        // in `removeAll`'s predicate accesses the array during its mutation.
        let id = slot.id
        app.loras.removeAll { $0.id == id }
    }

    /// The name's tooltip: what the file says of itself.
    private var details: String {
        guard let f = app.loraCard(slot.path) else { return String(localized: "Choose a LoRA") }
        let line = ([f.name, (f.smallestRank == f.rank ? String(localized: "rank \(f.rank)") : String(localized: "ranks \(f.smallestRank)–\(f.rank)")), f.resolution.map { String(localized: "trained at \($0)") }, f.trainedOn]
            .compactMap { $0 }).joined(separator: " · ")
        return ([line] + f.fromItsMetadata).joined(separator: "\n")
    }
}

/// **A trigger word, a click from the prompt**: it goes at the head of the text, where LoRAs are trained to find it; once
/// there, the button says so and does nothing more.
private struct TriggerButton: View {
    let app: AppState
    let trigger: String

    var body: some View {
        let there = app.prompt.localizedCaseInsensitiveContains(trigger)
        Button {
            let text = app.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
            app.prompt = text.isEmpty ? trigger : trigger + ", " + text
        } label: {
            Label { Text(verbatim: trigger).lineLimit(2).multilineTextAlignment(.leading) }
                icon: { Image(systemName: there ? "checkmark" : "plus") }
                .font(.caption)
        }
        .buttonStyle(.bordered).controlSize(.small)
        .help("Its trigger words, as its file gives them: a click puts them at the head of the prompt.")
        .unavailable(there ? String(localized: "In the prompt") : nil)
    }
}

extension LoRACard {
    /// What the file says of itself beyond its name, one line each: the trainer's name when it is
    /// not the one shown, the trigger words when the file states them. Empty for a file that says neither — then nothing is shown.
    var fromItsMetadata: [String] {
        var lines: [String] = []
        if let trainerName { lines.append(String(localized: "Named “\(trainerName)” in its metadata")) }
        if !triggers.isEmpty {
            lines.append(String(localized: "Trigger words: \(triggers.joined(separator: ", "))"))
        }
        return lines
    }
}
