// The DiT's box, the stage with choices, and the LoRA stack it holds. Split from
// `Rack.swift`, whose header says how the boxes behave.

import AppKit
import Siliconed
import SwiftUI

// MARK: - The DiT: the stage with choices

/// **The DiT and everything it is told**: the model, its license, the size, the seed, how many
/// images, the steps, the LoRA stack — then what it costs, or its steps while it runs.
struct DiTBox: View {
    @Bindable var app: AppState
    @State private var customSize = false

    var body: some View {
        StageBox(title: "DiT", symbol: "sparkles", state: app.state(of: .denoising),
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
        Field(title: "Batch") {
            Stepper(value: $app.batch, in: 1...8) { Text(verbatim: "\(app.batch)").monospacedDigit() }
                .fixedSize()
        }
        .help("Consecutive seeds; the prompt is encoded only once")
        Field(title: "Steps") {
            Stepper(value: $app.steps, in: 1...50) { Text(verbatim: "\(app.steps)").monospacedDigit() }
                .fixedSize()
        }
        .help(String(localized: "\(app.card?.defaultSteps ?? 8) steps: the count this Turbo model is distilled for"))
        // Not on a model that offers none (Anima): a choice that changes nothing is not shown.
        if app.offersDetail {
            Field(title: "Detail") {
                Picker("Detail", selection: $app.detail) {
                    ForEach(Detail.allCases, id: \.self) { Text(verbatim: $0.choice).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
            }
            .help("Adds fine texture and small details; Most can over-sharpen")
        }
        // More Solvers, the models in front: how a step moves to the next. Euler is the distillation's own.
        if app.offeredSolvers.count > 1 {
            Field(title: "Solver") {
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
        switches
    }

    /// **The two switches that change how the DiT computes**, each in the box of what it changes. A switch that cannot apply is
    /// greyed and says why, in a sentence, instead of disappearing or turning into a label. The decisions are `DiTSwitches`'s.
    @ViewBuilder private var switches: some View {
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

    /// **The LoRA stack, in the open** — as reachable as the steps or the seed: a title with « Add »
    /// (a menu of this model's LoRAs: one click opens it, one picks), then one row per LoRA with its
    /// strength's slider always under the hand. Nothing hides behind a popover.
    @ViewBuilder private var loras: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("LoRA").foregroundStyle(.secondary)
                if !app.loras.isEmpty {
                    Text(verbatim: "\(app.loras.count)/\(AppState.maxLoRA)").monospacedDigit().foregroundStyle(.secondary)
                }
                Spacer(minLength: 4)
                if app.compatibleLoras.isEmpty {
                    Text("No LoRA for this model").foregroundStyle(.secondary)
                } else {
                    LoRAAddMenu(app: app)
                }
                Button("Import…") { app.chooseToImport() }
                    .buttonStyle(.link)
                    .disabled(!app.importPossible)
                    .help("A model (.safetensors or .gguf) or a LoRA (.safetensors), as published on Civitai")
            }
            .font(.callout)
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

/// **« Add »**: the model's LoRAs not yet in the stack, in a menu — the pick goes in at 0.8, its
/// slider ready. Greyed at three, the stack's limit.
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
        case .notMeasured: String(localized: "This Mac is not measured yet: its first render measures it, then the accelerators can be chosen.")
        default: String(localized: "This Mac’s measure gives the accelerators nothing to do: the DiT is computed in fp32.")
        }
    }
}

struct LoRAAddMenu: View {
    let app: AppState

    var body: some View {
        let used = Set(app.loras.map(\.path))
        Menu {
            ForEach(app.compatibleLoras.filter { !used.contains($0.path) }) { l in
                Button(l.name) { app.loras.append(LoRASlot(path: l.path)) }
            }
        } label: {
            Label("Add", systemImage: "plus")
        }
        .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
        .disabled(!app.canAddLoRA || app.compatibleLoras.allSatisfy { used.contains($0.path) })
        .help("Add a LoRA to the stack (3 at most)")
    }
}

/// **A LoRA of the stack, in the open**: its name (a menu to swap it for another), its strength and a
/// remove button on one line; the slider under them, full width. A double click on the figure puts the
/// strength back to 0.8.
struct LoRARow: View {
    let app: AppState
    @Binding var slot: LoRASlot

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Menu {
                    ForEach(app.compatibleLoras) { l in
                        Button(l.name) { slot.path = l.path }.disabled(l.path == slot.path)
                    }
                } label: {
                    Text(app.loraCard(slot.path)?.name ?? String(localized: "Choose…"))
                        .lineLimit(1).truncationMode(.middle)
                }
                .menuStyle(.borderlessButton).menuIndicator(.hidden)
                .help(details)
                Spacer(minLength: 4)
                Text(Figures.strength(slot.strength)).monospacedDigit().foregroundStyle(.secondary)
                    .onTapGesture(count: 2) { slot.strength = LoRASlot().strength }
                    .help("Strength — double-click for 0.8")
                    .accessibilityLabel(Text("Strength"))
                    .accessibilityAddTraits(.isStaticText)
                Button {
                    // The id is read BEFORE: `slot` is a binding on `app.loras`, and re-reading it
                    // in `removeAll`'s predicate accesses the array during its mutation.
                    let id = slot.id
                    app.loras.removeAll { $0.id == id }
                } label: { Image(systemName: "xmark.circle.fill").frame(width: 24, height: 24).contentShape(Rectangle()) }
                    .buttonStyle(.borderless).foregroundStyle(.secondary)
                    .help("Remove this LoRA from the stack")
                    .accessibilityLabel(Text("Remove this LoRA from the stack"))
            }
            Slider(value: $slot.strength, in: app.loraStrengthRange)
                .controlSize(.small)
                .accessibilityLabel(Text("Strength"))
        }
        .contextMenu {
            Button("Reset to 0.8") { slot.strength = LoRASlot().strength }
        }
        .font(.callout)
        .padding(.horizontal, 8).padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Palette.well))
    }

    /// The name's tooltip: what the file says of itself.
    private var details: String {
        guard let f = app.loraCard(slot.path) else { return String(localized: "Choose a LoRA") }
        let line = ([f.name, (f.smallestRank == f.rank ? String(localized: "rank \(f.rank)") : String(localized: "ranks \(f.smallestRank)–\(f.rank)")), f.resolution.map { String(localized: "trained at \($0)") }, f.trainedOn]
            .compactMap { $0 }).joined(separator: " · ")
        return ([line] + f.fromItsMetadata).joined(separator: "\n")
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
