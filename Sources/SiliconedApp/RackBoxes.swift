// The rack's box furniture (title, ports, fields, tags, cost) and the two input nodes,
// prompt and images. Split from `Rack.swift`, whose header says how the boxes behave.

import AppKit
import Siliconed
import SwiftUI

// MARK: - A box

/// **A stage's box**: its symbol and name, its state on the right, then what fills it. Idle: a quiet
/// card. Working: the accent's edge and a faint wash of it. Done: a discreet check. Refused: red. It
/// takes the width it is given.
struct StageBox<Content: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let state: StageState
    /// What the stage does, in plain words: shown when the pointer rests on its name.
    let explanation: LocalizedStringKey
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            BoxTitle(title: title, symbol: symbol, tint: state == .active ? Palette.live : .secondary,
                     explanation: explanation) {
                // No spinner while working: the edge and the wash say it, and the one under the image
                // says the render runs — three turning at once was noise.
                switch state {
                case .idle, .active: EmptyView()
                case .done: Image(systemName: "checkmark").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                case .failed: Image(systemName: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(Palette.failure)
                }
            }
            content
        }
        .boxed(edge: edge, width: state == .idle || state == .done ? 1 : 1.5,
               wash: state == .active ? Palette.live : state == .failed ? Palette.failure : nil)
        .animation(Motion.pick(.easeInOut(duration: 0.25)), value: state)
    }

    private var edge: Color {
        switch state {
        case .idle, .done: Palette.hairline
        case .active: Palette.live
        case .failed: Palette.failure
        }
    }
}

/// A box's first line: its symbol in the wire's color, its name, and what it says on the right.
private struct BoxTitle<Trailing: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let tint: Color
    var explanation: LocalizedStringKey? = nil
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 7) {
            HStack(spacing: 7) {
                // The bare glyph in the wire's color: no chip around it, so that what is framed in a box's title is a button.
                Image(systemName: symbol)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(tint)
                    .frame(width: 20, height: 20)
                Text(title).font(.callout.weight(.semibold))
                    .lineLimit(2).fixedSize(horizontal: false, vertical: true)
            }
            .contentShape(Rectangle())
            .help(explanation.map { Text($0) } ?? Text(verbatim: ""))
            Spacer(minLength: 4)
            trailing
        }
    }
}

extension View {
    /// The box: padding, the card's fill, its edge — and, while it runs or failed, a faint wash of
    /// that color.
    fileprivate func boxed(edge: Color, width: CGFloat = 1, wash: Color? = nil) -> some View {
        padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(boxShape.fill(Palette.surface))
            .background(boxShape.fill(Color(nsColor: .windowBackgroundColor)))
            .overlay { if let wash { boxShape.fill(wash.opacity(0.06)).allowsHitTesting(false) } }
            .overlay(boxShape.strokeBorder(edge, lineWidth: width))
            .shadow(color: .black.opacity(0.08), radius: 2, y: 1)
    }
}

/// **A box the signal starts in** — the prompt, an edit's images: its chip in the color of the wire
/// it sends; its edge takes the accent while it is being typed in (`focused`).
struct PortBox<Trailing: View, Content: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let kind: WireKind
    var focused = false
    @ViewBuilder var trailing: Trailing
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            BoxTitle(title: title, symbol: symbol, tint: kind.color) {
                trailing.font(.caption).monospacedDigit().foregroundStyle(.secondary)
            }
            content
        }
        .boxed(edge: focused ? Palette.live : Palette.hairline, width: focused ? 1.5 : 1)
        .animation(Motion.pick(.easeInOut(duration: 0.15)), value: focused)
    }
}

extension PortBox where Trailing == EmptyView {
    init(title: LocalizedStringKey, symbol: String, kind: WireKind, focused: Bool = false, @ViewBuilder content: () -> Content) {
        self.init(title: title, symbol: symbol, kind: kind, focused: focused, trailing: { EmptyView() }, content: content)
    }
}

/// **A setting in a box**: its name on the left, its controls on the right — or under it, when the
/// box is too narrow for both on one line.
struct Field<Controls: View>: View {
    let title: LocalizedStringKey
    @ViewBuilder var controls: Controls

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
                Text(title).foregroundStyle(.secondary)
                Spacer(minLength: 8)
                controls
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(title).foregroundStyle(.secondary)
                HStack(spacing: 8) { controls }
            }
        }
        .font(.callout)
    }
}

/// A variation's strength, in words: the app's two, or a percentage for one set elsewhere.
enum VariationWords {
    static func label(_ strength: Double) -> String {
        switch Variation.Amount.named(strength) {
        case .subtle: return String(localized: "Subtle")
        case .strong: return String(localized: "Strong")
        case nil: break
        }
        return Figures.percent(Int((strength * 100).rounded()))
    }
}

/// **A module the family sets**: its name in a slot, locked. The slot is where a choice will be
/// offered once there is one to make — not an empty menu now.
struct ModuleSlot: View {
    let name: String?
    var body: some View {
        HStack(spacing: 5) {
            Text(name ?? "—").lineLimit(1).truncationMode(.middle)
            Spacer(minLength: 2)
            Image(systemName: "lock.fill").imageScale(.small).foregroundStyle(.secondary)
        }
        .font(.callout)
        .padding(.horizontal, 8).padding(.vertical, 5)
        .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Palette.well))
        .help("Set by the model’s family")
    }
}

/// **How a module's weights are stored** (`WeightFormat`): « weights int8 », the name verbatim and the word said in the user's
/// language, so that it is never taken for the label of the computation (« computes fp32 »). The hover says what it is in plain words.
struct WeightsTag: View {
    let format: WeightFormat?
    /// What the accelerators do for this module on this Mac (the DiT's, `AppState.acceleratorsInEffect`); `none` for an encoder or
    /// the VAE, which never take them.
    var accelerators = AcceleratorOffer.none
    /// What the module is, in the sentence of the hover: its stage.
    var body: some View {
        if let format {
            Tag(text: Text("weights \(format.label)"), symbol: "shippingbox")
                .help(Self.explanation(format) + " " + Self.computation(accelerators))
        }
    }

    static func explanation(_ f: WeightFormat) -> String {
        switch f {
        case .int8: String(localized: "The weights are stored in 8 bits (int8), as published: a quarter of fp32’s size.")
        case .q8_0: String(localized: "The weights are stored in 8 bits in blocks of 32 (GGUF Q8_0), as published: a quarter of fp32’s size.")
        case .fp8: String(localized: "The weights are stored in 8 bits (fp8), as published.")
        case .q4_k_m, .q4_k, .q4_0, .q4_1: String(localized: "The weights are stored in about 4 bits (GGUF \(f.label)), as published: an eighth of fp32’s size.")
        case .q5_k_m, .q5_k, .q5_0, .q5_1, .q6_k: String(localized: "The weights are stored in 5 to 6 bits (GGUF \(f.label)), as published.")
        case .bf16: String(localized: "The weights are stored in bf16, as the publisher released them: half of fp32’s size.")
        case .fp16: String(localized: "The weights are stored in fp16: half of fp32’s size.")
        case .fp32: String(localized: "The weights are stored in fp32.")
        }
    }

    /// What computes the products: fp32, or fp32 but for the part this Mac's map gives the accelerators.
    static func computation(_ o: AcceleratorOffer) -> String {
        if let place = AcceleratorWords.place(o) { return String(localized: "The products are computed in fp32, except part of them \(place).") }
        return String(localized: "The products are computed in fp32.")
    }
}

/// A label: a small capsule, neutral — or with a dot of color (the latent's wire).
struct Tag: View {
    let text: Text
    var symbol: String? = nil
    var dot: Color? = nil
    var body: some View {
        HStack(spacing: 4) {
            if let dot { Circle().fill(dot).frame(width: 6, height: 6) }
            if let symbol { Image(systemName: symbol).imageScale(.small) }
            text
        }
        .font(.caption2.weight(.medium))
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .padding(.horizontal, 7).padding(.vertical, 2.5)
        .background(Capsule().fill(Palette.well))
    }
}

/// The time a stage costs: what it just took in the render on screen (no « ≈ »), or the estimate
/// learned on this Mac — or nothing yet.
struct Cost: View {
    let seconds: Double?
    let measured: Bool
    init(_ cost: (seconds: Double?, measured: Bool)) { seconds = cost.seconds; measured = cost.measured }
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "clock").imageScale(.small)
            Text(seconds.map { (measured ? "" : "≈ ") + Figures.seconds($0) } ?? "—").monospacedDigit()
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .help(measured ? "What this stage just took"
                       : "Learned on this Mac: the first render at this size, with this model, sets the estimate")
    }
}

// MARK: - The inputs

/// **The prompt is typed in its box**: what to make — or, when editing, what changes. A real text view
/// (`TextEditor`, an `NSTextView`): ⌘A, ⌘C, ⌘Z, the spelling, a long prompt that scrolls. The grip
/// under it gives it the height one wants; Esc leaves it (the arrows then walk the strip). The wand facing
/// its title opens the Prompt Lab; the cross in its corner empties it, and ⌘Z brings it back.
struct PromptNode: View {
    @Bindable var app: AppState
    @FocusState private var typing: Bool
    @State private var height: CGFloat = 72
    @GestureState private var stretch: CGFloat = 0
    private static let heights: ClosedRange<CGFloat> = 44...480

    var body: some View {
        PortBox(title: "Prompt", symbol: "text.quote", kind: .text, focused: typing) {
            // The Prompt Lab, one click from the prompt it prepares (Window ▸ Prompt Lab, ⌥⌘L).
            // A bezelled button in the prompt's color: the title row's only frame, so it reads as something to click.
            Button { LabWindow.shared.show() } label: {
                Image(systemName: "wand.and.sparkles")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(WireKind.text.color)
                    .padding(.horizontal, 2)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Open the Prompt Lab (⌥⌘L)")
            .accessibilityLabel(Text("Prompt Lab"))
        } content: {
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $app.prompt)
                        .accessibilityLabel(Text("Prompt"))
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .scrollIndicators(.automatic)
                        // Room for the clear button: a line never runs under it.
                        .contentMargins(.trailing, app.prompt.isEmpty ? 0 : 18, for: .scrollContent)
                        .focused($typing)
                    if app.prompt.isEmpty {
                        Text(app.references.isEmpty ? "Describe the image…" : "Say what changes…")
                            .font(.body).foregroundStyle(.tertiary)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
                .overlay(alignment: .topTrailing) {
                    // A search field's clear button, where the eye looks for it; ⌘Z (Edit ▸ Undo) brings the prompt back.
                    if !app.prompt.isEmpty {
                        Button { MainWindow.shared.setPrompt("", of: app, action: String(localized: "Clear Prompt")) } label: {
                            Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
                        }
                        .buttonStyle(.plain)
                        .padding(.trailing, 2)
                        .help("Clear the prompt (⌘Z brings it back)")
                        .accessibilityLabel(Text("Clear Prompt"))
                    }
                }
                .frame(height: min(max(height + stretch, Self.heights.lowerBound), Self.heights.upperBound))
                grip
            }
        }
    }

    /// The handle: dragged, the field grows or shrinks; its cursor says so.
    private var grip: some View {
        Capsule().fill(Color.secondary.opacity(0.5)).frame(width: 34, height: 4)
            .frame(maxWidth: .infinity).frame(height: 12)
            .contentShape(Rectangle())
            .onHover { inside in if inside { NSCursor.resizeUpDown.push() } else { NSCursor.pop() } }
            .gesture(DragGesture(minimumDistance: 1)
                .updating($stretch) { v, s, _ in s = v.translation.height }
                .onEnded { v in
                    height = min(max(height + v.translation.height, Self.heights.lowerBound), Self.heights.upperBound)
                })
            .help("Drag to resize the prompt")
            .padding(.bottom, -6)
    }
}

/// **The images the prompt works on, numbered as it names them**: image 1 is the one edited — the
/// output takes its format —, the others are what the prompt borrows from. Added by the button, by a
/// drop here or on the canvas (there, a PNG Siliconed wrote restores its settings instead), or by
/// « Edit This Image » in the history; reordered by dragging a
/// thumbnail onto another, or from its menu. Only for a model that edits (rule 2: the inputs follow
/// the model). The Prompt Lab fills it too, with the starting image, the references and the subjects its text
/// cites (« Carry to the Rack »). The title counts what the next render sends.
struct ImagesNode: View {
    @Bindable var app: AppState
    private var used: Int { app.references.count }
    var body: some View {
        PortBox(title: "Images", symbol: "photo.on.rectangle", kind: .image) {
            Text(used == 0 ? String(localized: "up to \(app.maxReferences) images") : "\(used)/\(app.maxReferences)")
                .foregroundStyle(used > app.maxReferences ? Palette.failure : .secondary)
        } content: {
            images
        }
        .dropDestination(for: DroppedImage.self) { items, _ in
            var files: [URL] = []
            for item in items {
                switch item {
                case .file(let url) where !AppState.importable(url): files.append(url)
                case .file: break
                case .png(let data): app.addReference(png: data)
                }
            }
            app.addReferences(files)
            return items.contains { if case .file(let u) = $0 { !AppState.importable(u) } else { true } }
        }
    }

    @ViewBuilder private var images: some View {
        // One more image beyond this Mac's memory (B): the « + » greyed out, the sentence below.
        let refusal = app.canAddReference ? app.nextReferenceRefusal() : nil
        HStack(alignment: .top, spacing: 6) {
            ForEach(Array(app.references.enumerated()), id: \.element.id) { n, reference in
                ReferenceThumbnail(app: app, reference: reference, number: n + 1,
                                   last: n == app.references.count - 1)
            }
            if app.canAddReference {
                Button { app.chooseReferences() } label: {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .strokeBorder(style: StrokeStyle(lineWidth: 1.25, dash: [4, 3]))
                        .foregroundStyle(WireKind.image.color.opacity(0.7))
                        .overlay(Image(systemName: "plus").foregroundStyle(WireKind.image.color))
                        .frame(width: ReferenceThumbnail.side, height: ReferenceThumbnail.side)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(refusal != nil).opacity(refusal == nil ? 1 : 0.35)
                .help(app.references.isEmpty ? "The image to edit — or drop or paste it here (⌘V)"
                                             : "Add an image the prompt can draw from — or drop or paste it here (⌘V)")
                .accessibilityLabel(Text("Add an Image"))
            }
            Spacer(minLength: 0)
        }
        Group {
            if let refusal {
                Text(refusal)
            } else if app.references.isEmpty {
                Text("Optional. Add an image and the prompt becomes an instruction: “change her jacket to red”.")
            } else if app.promptNamesImages && app.maxReferences > 1 {
                Text("Image 1 is the one edited. The prompt can name them: “put the dog of image 2 next to the woman in image 1”.")
                    .help("The model reads them as <image1>, <image2>… in front of the prompt")
            } else {
                Text("Image 1 is the one edited: the prompt says what changes.")
            }
        }
        .font(.caption).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
}

/// **The image the render starts from** (img2img), on a model that starts from one: shown only once the Prompt Lab (or
/// `silicontrol start-image`) has carried one — the rack has no empty slot for it, the Lab is where it is chosen. Its thumbnail,
/// and its strength on a slider from 0 to 100 %, like a LoRA's: retouched between two renders. Low keeps the image's composition
/// and colours, 100 % ignores it; the tooltip says how many of the steps the strength leaves. A file dropped here replaces it.
struct StartImageNode: View {
    @Bindable var app: AppState
    @State private var hover = false

    var body: some View {
        PortBox(title: "Starting Image", symbol: "photo", kind: .image) {
            Text(verbatim: StartWords.percent(app.startStrength))
                .onTapGesture(count: 2) { app.setStartStrength(app.defaultStartStrength) }
                .help("Strength — double-click for \(StartWords.percent(app.defaultStartStrength))")
        } content: {
            if let start = app.usedStart {
                HStack(alignment: .center, spacing: 10) {
                    thumbnail(start)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Strength").font(.callout).foregroundStyle(.secondary)
                        Slider(value: Binding(get: { app.startStrength }, set: { app.setStartStrength($0) }), in: 0...1)
                            .controlSize(.small)
                            .accessibilityLabel(Text("Strength"))
                            .accessibilityValue(Text(verbatim: StartWords.percent(app.startStrength)))
                        HStack {
                            Text("Faithful")
                            Spacer(minLength: 4)
                            Text("Free")
                        }
                        .font(.caption2).foregroundStyle(.tertiary)
                    }
                    .help(StartWords.explanation(app.startRedraws))
                }
                let caption = app.startCaption
                Text(caption.text)
                    .font(.caption)
                    .foregroundStyle(caption.tone == .failure ? Palette.failure
                                     : caption.tone == .warning ? Color(nsColor: .systemOrange) : Color.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .contextMenu {
            Button("Reset to \(StartWords.percent(app.defaultStartStrength))") { app.setStartStrength(app.defaultStartStrength) }
            Button("Replace…") { app.chooseStartReplacement() }
            Divider()
            Button("Remove", role: .destructive) { app.removeStart() }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first(where: { $0.isFileURL && !AppState.importable($0) }) else { return false }
            app.replaceStart(with: url)
            return true
        }
    }

    private func thumbnail(_ start: ReferenceImage) -> some View {
        Image(nsImage: start.vignette).resizable().scaledToFill()
            .frame(width: ReferenceThumbnail.side, height: ReferenceThumbnail.side)
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(WireKind.image.color, lineWidth: 1.5))
            .overlay(alignment: .topTrailing) {
                if hover {
                    Button { app.removeStart() } label: {
                        Image(systemName: "xmark.circle.fill").symbolRenderingMode(.palette)
                            .foregroundStyle(.white, .black.opacity(0.6))
                            .frame(width: 24, height: 24).contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .help("Remove: the next render starts from noise")
                    .accessibilityLabel(Text("Remove"))
                }
            }
            .onHover { hover = $0 }
            .help(Text(verbatim: "\(start.name) · \(start.width) × \(start.height)"))
            .accessibilityLabel(Text(verbatim: start.name))
    }
}
