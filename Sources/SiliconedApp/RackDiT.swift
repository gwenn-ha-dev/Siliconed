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

    var body: some View {
        StageBox(title: "DiT", symbol: "engine.combustion", state: app.state(of: .denoising),
                 explanation: "The model that draws (a diffusion transformer): from noise, step by step, it makes the image the prompt describes") {
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
            if let card = app.card {
                if app.missing != nil {
                    HStack {
                        Text("Not installed").font(.callout).foregroundStyle(.secondary)
                        Spacer()
                        Button("Install…") { app.installSelectedModel() }
                            .controlSize(.small)
                            .disabled(!app.managementPossible)
                    }
                } else {
                    // The labels on one line when they fit; the license alone above the others when not.
                    // The labels on one line when they fit; else what is stored and what computes stay together on a line of
                    // their own, the license and the latent on another, then each pair alone — a label is never truncated.
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 5) { license(card); precision; latent }
                        VStack(alignment: .leading, spacing: 5) {
                            HStack(spacing: 5) { precision }
                            HStack(spacing: 5) { license(card); latent }
                        }
                        VStack(alignment: .leading, spacing: 5) {
                            HStack(spacing: 5) { precision }
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

    @ViewBuilder private var precision: some View {
        // What is stored, then what computes: two labels, two words, so that « int8 » and « fp32 » are never one thing.
        WeightsTag(format: app.weightFormats?.dit, accelerators: app.acceleratorsInEffect)
        // A label only: the choice is the switch below, in the settings of the box (`switches`). What it says is what this Mac's map
        // gives the accelerators (`AcceleratorWords`), the Neural Engine first.
        let offer = app.acceleratorsInEffect
        Tag(text: Text("computes \(AcceleratorWords.tag(offer))"), symbol: offer.neuralEngine ? "leaf" : offer.isNone ? nil : "bolt")
            .help(AcceleratorWords.place(offer).map { String(localized: "fp32, except part of the DiT \($0)") }
                  ?? String(localized: "The products of the DiT are computed in fp32, the image verified against the reference."))
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
    /// limit, or the Mac it needs), as the default menu does.
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
            Text(verbatim: "\(app.format.width) × \(app.format.height)").monospacedDigit().foregroundStyle(.secondary)
            Button { customSize = true } label: { Image(systemName: "pencil") }
                .buttonStyle(.borderless)
                .help("Type a size (width × height)")
                .accessibilityLabel(Text("Type a size (width × height)"))
                .popover(isPresented: $customSize, arrowEdge: .trailing) { CustomSize(app: app) }
        }
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
        let pixels = "\(cell.width) × \(cell.height)"
        let really = cell.actualRatio.description, asked = cell.ratio.description
        let off = cell.deviation.formatted(.percent.precision(.fractionLength(1)))
        return Text("\(pixels) is really \(really), \(off) off \(asked): an exact \(asked) in multiples of 32 has no size near this resolution.")
    }

    // ── what the DiT is told: size, seed, how many, steps ──

    @ViewBuilder private var settings: some View {
        if app.references.isEmpty, app.sizeScalesBeta {
            scales
        } else if app.references.isEmpty {
            Field(title: "Size") {
                Picker("Orientation", selection: $app.orientation) {
                    ForEach([RecommendedFormat.Orientation.square, .portrait, .landscape], id: \.self) { o in
                        Image(systemName: o.symbol).help(o.label).accessibilityLabel(Text(verbatim: o.label)).tag(o)
                    }
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
                // What this Mac cannot render is greyed out ahead, with the Mac it needs (B, `macNeeded`).
                // **Its width is a plain size's, never its longest entry's**: a menu picker sizes to its
                // widest item, and the squares list « 2048 × 2048 — needs a Mac with 32 GB » — each switch
                // to square widened the rack's column. A hidden one-item picker sets the width; the real
                // one is laid over it, so its list never reaches the layout.
                Picker("Size", selection: .constant(0)) { Text(verbatim: "0000 × 0000").tag(0) }
                    .labelsHidden().fixedSize().hidden().accessibilityHidden(true)
                    .overlay(alignment: .trailing) {
                        Picker("Size", selection: $app.format) {
                            ForEach(app.pickerFormats, id: \.self) { f in
                                if let n = app.macNeeded(f, images: 0) {
                                    Text("\(String(f.width)) × \(String(f.height)) — needs a Mac with \(String(n)) GB")
                                        .tag(f).disabled(true)
                                } else {
                                    Text(verbatim: "\(f.width) × \(f.height)").tag(f)
                                }
                            }
                        }
                        .labelsHidden()
                    }
                Button { customSize = true } label: { Image(systemName: "pencil") }
                    .buttonStyle(.borderless)
                    .help("Type a size (width × height)")
                    .accessibilityLabel(Text("Type a size (width × height)"))
                    .popover(isPresented: $customSize, arrowEdge: .trailing) { CustomSize(app: app) }
            }
        } else {
            let f = app.outputFormat
            Field(title: "Size") {
                Text(verbatim: "\(f.width) × \(f.height)").monospacedDigit()
                Image(systemName: "lock.fill").imageScale(.small).foregroundStyle(.secondary)
            }
            .help("An edit keeps the framing of image 1: about 1 megapixel at its proportions, each side at least 512, 1024² at most")
        }
        Field(title: "Seed") {
            if app.fixedSeed {
                TextField("Seed", value: $app.seed, format: .number.grouping(.never))
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .frame(width: 100)
                Button { app.seed = UInt64.random(in: 0...UInt64(UInt32.max)) } label: { Image(systemName: "dice") }
                    .buttonStyle(.borderless)
                    .help("Draw a seed")
                    .accessibilityLabel(Text("Draw a seed"))
            }
            Picker("Seed", selection: $app.fixedSeed) {
                Text("Random").tag(false)
                Text("Fixed").tag(true)
            }
            .labelsHidden().fixedSize()
            .help("Random: a new seed for every render. Fixed: the same image again")
        }
        if !app.variations.isEmpty {
            Field(title: "Variation") {
                Text(verbatim: app.variations.map { VariationWords.label($0.strength) }.joined(separator: " + "))
                Button { app.variations = [] } label: { Image(systemName: "xmark.circle.fill").frame(width: 24, height: 24).contentShape(Rectangle()) }
                    .buttonStyle(.borderless).foregroundStyle(.secondary)
                    .help("Remove the variation: the seed’s own image")
                    .accessibilityLabel(Text("Remove the variation: the seed’s own image"))
            }
            .help("The next render is a variation of the seed’s image, like the image it came from")
        }
        // The lot is the series' seed axis (4/11): bracketing the seed is choosing how many, and where they go.
        Field(title: "Batch", app: app, bracket: .seeds) {
            Stepper(value: $app.batch, in: 1...8) { Text(verbatim: "\(app.batch)").monospacedDigit() }
                .fixedSize()
        }
        .help("Consecutive seeds; the prompt is encoded only once")
        // **The steps, only off the recipe**: the five models are distilled for 8, and fewer is softer, more is longer and no
        // better — a choice that only costs. The line comes when something else set them (a PNG's settings reused,
        // `silicontrol`), as the variation's does, with its way back.
        let recipe = app.card?.defaultSteps ?? 8
        if app.steps != recipe {
            Field(title: "Steps") {
                Text(verbatim: "\(app.steps)").monospacedDigit()
                Button { app.steps = recipe } label: { Image(systemName: "xmark.circle.fill").frame(width: 24, height: 24).contentShape(Rectangle()) }
                    .buttonStyle(.borderless).foregroundStyle(.secondary)
                    .help(String(localized: "Back to \(recipe) steps, the count this model is distilled for"))
                    .accessibilityLabel(Text("Back to \(recipe) steps, the count this model is distilled for"))
            }
            .help(String(localized: "\(recipe) steps: the count this Turbo model is distilled for"))
        }
        // Not on a model that offers none (Anima): a choice that changes nothing is not shown.
        if app.offersDetail {
            Field(title: "Detail", app: app, bracket: .detail) {
                Picker("Detail", selection: $app.detail) {
                    ForEach(Detail.allCases, id: \.self) { Text(verbatim: $0.choice).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
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
            .font(.callout)
            .help("Shows the image taking shape on the canvas after each step. It does not change the image.")
            .padding(.leading, BracketMark.gutter)
        switches.padding(.leading, BracketMark.gutter)
    }

    /// **The two switches that change how the DiT computes**, each in the box of what it changes. A switch that cannot apply is
    /// greyed and says why, in a sentence, instead of disappearing or turning into a label. The decisions are `DiTSwitches`'s.
    private var switches: some View {
        VStack(alignment: .leading, spacing: 9) { switchRows }
    }

    @ViewBuilder private var switchRows: some View {
        let id = app.identifier
        let reason = DiTSwitches.acceleratorsUnavailable(measured: app.settingsRead, offered: app.acceleratorsOffered)
        VStack(alignment: .leading, spacing: 2) {
            Toggle("Accelerators", isOn: Binding(get: { reason == nil && app.accelerators(for: id) },
                                                 set: { app.setAccelerators($0, for: id) }))
                .font(.callout)
                .disabled(reason != nil)
                .help(AcceleratorWords.switchHelp(app.acceleratorOffer ?? .none))
            if let reason {
                Text(Self.sentence(reason))
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        // Faster Rendering is a Beta option: no switch without it.
        if app.fasterRenderingBeta {
            let reason = app.fasterRenderingUnavailable
            VStack(alignment: .leading, spacing: 2) {
                Toggle("Faster Rendering", isOn: Binding(get: { app.fasterRenderingShown(for: id) },
                                                         set: { app.setFasterRendering($0, for: id) }))
                    .font(.callout)
                    .disabled(reason != nil)
                    .help("From 1024² up, the first steps at half size: faster, a slightly different image.")
                if let reason {
                    Text(Self.sentence(reason)).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    /// Why Faster Rendering would do nothing for this render, in a sentence.
    private static func sentence(_ reason: DiTSwitchUnavailable) -> String {
        switch reason {
        case .sizeUnder1024: String(localized: "Needs 1024 px or more on each side.")
        case .withLoRADetailOrEdit: String(localized: "Not with a LoRA, Detail or an edit.")
        case .stepsOffRecipe: String(localized: "Only at the model’s own step count.")
        case .notMeasured, .noAccelerator: AcceleratorWords.unavailable(reason)
        }
    }

    /// **The LoRA stack, in the open** — as reachable as the steps or the seed: a title with one « Add » (this model's LoRAs,
    /// then Import…), then one card per LoRA: on or off, its strength under the hand, its trigger words a click away from
    /// the prompt. Nothing hides behind a popover or a hover.
    @ViewBuilder private var loras: some View {
        VStack(alignment: .leading, spacing: 6) {
            let header = HStack(spacing: 8) {
                Text("LoRA").foregroundStyle(.secondary)
                if !app.loras.isEmpty {
                    Text(verbatim: "\(app.loras.count)/\(AppState.maxLoRA)").monospacedDigit().foregroundStyle(.secondary)
                }
                Spacer(minLength: 4)
                LoRAAddMenu(app: app)
            }
            .font(.callout)
            // The title's ±: « Compare LoRAs », one of them per image beside the image without — where this model has some.
            if app.applies(.comparedLoRAs) {
                header.bracketable(app, .comparedLoRAs, menu: false)
            } else {
                header.padding(.leading, BracketMark.gutter)
            }
            Group {
                if app.compatibleLoras.isEmpty, app.loras.isEmpty {
                    Text("No LoRA for this model").font(.caption).foregroundStyle(.secondary)
                }
                ForEach($app.loras) { $slot in LoRARow(app: app, slot: $slot) }
            }
            .padding(.leading, BracketMark.gutter)
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
                    if let r = app.remaining(at: context.date) {
                        Text("\(Figures.duration(r)) left").monospacedDigit().foregroundStyle(.secondary)
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

/// **What the accelerators do, in words** — one place for the DiT box, the rack's legend, the image's panel and `silicontrol`, read
/// from what this Mac's map gives them (`AcceleratorOffer`), never from the chip's name: the Neural Engine first, the GPU's tensor
/// units in half precision said in clear, « fp32 » where neither works (a Mac whose map chooses none, or the switch off).
enum AcceleratorWords {
    /// **Where part of the DiT goes**, as the end of a sentence: « on the Neural Engine (fp16) and the GPU’s tensor units (bf16) ».
    /// `nil`: nowhere, fp32 throughout.
    static func place(_ o: AcceleratorOffer) -> String? {
        let units = o.tensorUnitsText
        switch (o.neuralEngine, o.tensorUnits.isEmpty) {
        case (true, true): return String(localized: "on the Neural Engine (fp16)")
        case (true, false): return String(localized: "on the Neural Engine (fp16) and the GPU’s tensor units (\(units))")
        case (false, false): return String(localized: "on the GPU’s tensor units (\(units))")
        case (false, true): return nil
        }
    }

    /// The DiT's « computes » tag: `fp32`, `fp32 · ANE`, `fp32 · GPU bf16`, `fp32 · ANE · GPU bf16, fp16` — names, not words.
    static func tag(_ o: AcceleratorOffer) -> String {
        (["fp32"] + (o.neuralEngine ? ["ANE"] : []) + (o.tensorUnits.isEmpty ? [] : ["GPU " + o.tensorUnitsText]))
            .joined(separator: " · ")
    }

    /// The switch's hover: what it turns on, on this Mac, and what off gives.
    static func switchHelp(_ o: AcceleratorOffer) -> String {
        guard let place = place(o) else {
            return String(localized: "The Neural Engine and the GPU’s tensor units (bf16), where this Mac’s measure chooses them; fp32 throughout when you turn them off.")
        }
        return String(localized: "Part of the DiT \(place), where this Mac’s measure finds it faster or more frugal: an image a little different from fp32’s, checked against the reference in bf16. Off, the DiT is in fp32 throughout, the image verified against the reference.")
    }

    /// Why the switch has nothing to choose.
    static func unavailable(_ reason: DiTSwitchUnavailable) -> String {
        switch reason {
        case .notMeasured: String(localized: "This Mac is not measured yet: once it is (Measure This Mac…), the accelerators can be chosen.")
        default: String(localized: "This Mac’s measure gives the accelerators nothing to do: the DiT is computed in fp32.")
        }
    }
}

/// **« Add »**, the stack's one way in: the model's LoRAs (those in the stack greyed; all greyed at three, the limit), then
/// Import… for a file from Civitai. The pick goes in at 0.8, on.
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
            Divider()
            // A series: one of them added per image, and the image without (`SeriesKey.comparedLoRAs`).
            Button("Compare LoRAs…") { app.bracketing = .comparedLoRAs }
                .disabled(!app.canAddLoRA || app.compatibleLoras.allSatisfy { used.contains($0.path) })
        } label: {
            Label("Add", systemImage: "plus")
        }
        .menuStyle(.button).buttonStyle(.bordered).controlSize(.small).fixedSize()
        .help("A LoRA of this model (3 at most), or one to import: a .safetensors as published on Civitai")
    }
}

/// **A LoRA of the stack, a card**: on top, its switch, its name and the button that takes it out; under them its strength,
/// a slider and the figure, which can be typed; then its trigger words when its file gives them — without them many LoRAs
/// do nothing, and a hover was the only place they showed: a click puts them at the head of the prompt. Off, the card stays,
/// greyed, its strength kept. Right-click: Replace With, Reset to 0.8, Remove.
struct LoRARow: View {
    let app: AppState
    @Binding var slot: LoRASlot

    var body: some View {
        let card = app.loraCard(slot.path)
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Toggle(isOn: $slot.on) { EmptyView() }
                    .toggleStyle(.checkbox).labelsHidden()
                    .help("On, it goes into the render. Off, it stays here with its strength, out of the render.")
                    .accessibilityLabel(Text("Use this LoRA"))
                Text(card?.name ?? String(localized: "Choose…"))
                    .fontWeight(.medium).lineLimit(1).truncationMode(.middle)
                    .help(details)
                Spacer(minLength: 4)
                Button(action: remove) {
                    Image(systemName: "xmark.circle.fill").frame(width: 20, height: 20).contentShape(Rectangle())
                }
                .buttonStyle(.borderless).foregroundStyle(.secondary)
                .help("Remove this LoRA from the stack")
                .accessibilityLabel(Text("Remove this LoRA from the stack"))
            }
            HStack(spacing: 8) {
                Slider(value: $slot.strength, in: app.loraStrengthRange)
                    .controlSize(.small)
                    .accessibilityLabel(Text("Strength"))
                TextField(value: strength, format: .number.precision(.fractionLength(2))) { Text("Strength") }
                    .labelsHidden().textFieldStyle(.roundedBorder).controlSize(.small).monospacedDigit()
                    .multilineTextAlignment(.trailing).frame(width: 48)
                    .help("Strength: 0.8 attenuates, 1 as trained, more emphasizes")
            }
            // Its ± under the switch, at the left of the strength it varies.
            .bracketable(app, .loraStrength(slot.path), menu: false, gutter: 22)
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
        .disabled(there)
        .help(there ? Text("In the prompt") : Text("Its trigger words, as its file gives them: a click puts them at the head of the prompt."))
    }
}

extension LoRACard {
    /// What the file says of itself beyond its name, one line each: the trainer's name when it is
    /// not the one shown, the trigger words when the file states them. Empty for a map forged
    /// before these keys, or a file that says neither — then nothing is shown.
    var fromItsMetadata: [String] {
        var lines: [String] = []
        if let trainerName { lines.append(String(localized: "Named “\(trainerName)” in its metadata")) }
        if !triggers.isEmpty {
            lines.append(String(localized: "Trigger words: \(triggers.joined(separator: ", "))"))
        }
        return lines
    }
}
