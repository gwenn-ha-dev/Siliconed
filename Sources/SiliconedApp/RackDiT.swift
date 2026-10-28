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
            .help("The denoiser: the publisher's model, or a fine-tune you imported")
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
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 5) { license(card); technical }
                        VStack(alignment: .leading, spacing: 5) { license(card); HStack(spacing: 5) { technical } }
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

    @ViewBuilder private var technical: some View {
        if app.neuralEngineOffered {
            // The model's choice, per installed version: a click switches it for the next renders.
            let on = app.neuralEngine(for: app.identifier)
            Button { app.setNeuralEngine(!on, for: app.identifier) } label: {
                Tag(text: Text(verbatim: on ? "fp32 · ANE" : "fp32"), symbol: on ? "leaf" : nil)
            }
            .buttonStyle(.plain)
            .help(on ? String(localized: "Part of the DiT runs in fp16 on the Neural Engine: less energy (about a fifth on an M1 Pro), an image a little different from fp32's. Click for fp32 throughout.")
                     : String(localized: "The DiT in fp32 throughout, the image verified against the reference. Click to run part of it in fp16 on the Neural Engine: less energy (about a fifth on an M1 Pro), an image a little different."))
        } else {
            Tag(text: Text(verbatim: "fp32"))
        }
        if let space = app.chain?.denoising.space {
            Tag(text: Text("latent · \(space.channels) ch"), dot: WireKind.latent.color)
                .help(Text(verbatim: "\(space.name) · ÷\(space.factor)"))
        }
    }

    // ── what the DiT is told: size, seed, how many, steps ──

    @ViewBuilder private var settings: some View {
        if app.references.isEmpty {
            Field(title: "Size") {
                Picker("Orientation", selection: $app.orientation) {
                    ForEach([RecommendedFormat.Orientation.square, .portrait, .landscape], id: \.self) { o in
                        Image(systemName: o.symbol).help(o.label).tag(o)
                    }
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
                // What this Mac cannot render is greyed out ahead, with the Mac it needs (B, `macNeeded`).
                Picker("Size", selection: $app.format) {
                    ForEach(app.pickerFormats, id: \.self) { f in
                        if let n = app.macNeeded(f, images: app.subjectImageCount) {
                            Text("\(String(f.width)) × \(String(f.height)) — needs a Mac with \(String(n)) GB")
                                .tag(f).disabled(true)
                        } else {
                            Text(verbatim: "\(f.width) × \(f.height)").tag(f)
                        }
                    }
                }
                .labelsHidden().fixedSize()
                Button { customSize = true } label: { Image(systemName: "pencil") }
                    .buttonStyle(.borderless)
                    .help("Type a size (width × height)")
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
                Button { app.variations = [] } label: { Image(systemName: "xmark.circle.fill") }
                    .buttonStyle(.borderless).foregroundStyle(.secondary)
                    .help("Remove the variation: the seed's own image")
            }
            .help("The next render is a variation of the seed's image, like the image it came from")
        }
        Field(title: "Images") {
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
        // Developer Mode, the models in front: how a step moves to the next. Euler is the distillation's own.
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
                .animation(.easeOut(duration: 0.3), value: app.currentStep)
            }
        }
    }
}

/// **« Add »**: the model's LoRAs not yet in the stack, in a menu — the pick goes in at 0.8, its
/// slider ready. Greyed at three, the stack's limit.
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
                Button {
                    // The id is read BEFORE: `slot` is a binding on `app.loras`, and re-reading it
                    // in `removeAll`'s predicate accesses the array during its mutation.
                    let id = slot.id
                    app.loras.removeAll { $0.id == id }
                } label: { Image(systemName: "xmark.circle.fill") }
                    .buttonStyle(.borderless).foregroundStyle(.secondary)
                    .help("Remove this LoRA from the stack")
            }
            Slider(value: $slot.strength, in: app.loraStrengthRange)
                .controlSize(.small)
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
