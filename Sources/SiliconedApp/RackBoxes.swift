// The rack's box furniture (title, ports, fields, tags, cost) and its input ports, read only: the prompt, the images or
// the starting image. Split from `Rack.swift`, whose header says how the boxes behave.

import AppKit
import Siliconed
import SwiftUI

// MARK: - A box

/// **A stage's box**: its icon and name, its state on the right, then what fills it. Idle: a quiet
/// card, its icon in the color of the wire it gives. Working: the edge, the icon and a faint wash of what runs.
/// Done: a discreet check. Refused: red. It takes the width it is given.
struct StageBox<Content: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    /// The wire the stage gives: its icon's color at rest.
    let kind: WireKind
    let state: StageState
    /// What the stage does, in plain words: shown when the pointer rests on its name.
    let explanation: LocalizedStringKey
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            BoxTitle(title: title, symbol: symbol,
                     tint: state == .active ? Palette.live : state == .failed ? Palette.failure : kind.color,
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

/// A box's first line: its icon (`IconTile`), its name, and what it says on the right.
private struct BoxTitle<Trailing: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let tint: Color
    var explanation: LocalizedStringKey? = nil
    /// A port's name: read, not a stage's.
    var quiet = false
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 7) {
            HStack(spacing: 7) {
                IconTile(symbol: symbol, color: tint)
                Text(title).font(.callout.weight(quiet ? .medium : .semibold))
                    .foregroundStyle(quiet ? .secondary : .primary)
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
            .overlay { if let wash { boxShape.fill(wash.opacity(0.06)).allowsHitTesting(false) } }
            .overlay(boxShape.strokeBorder(edge, lineWidth: width))
            .shadow(color: .black.opacity(0.08), radius: 2, y: 1)
    }
}

/// **A box the signal starts in** — the text, an edit's images: its icon in the color of the wire it sends. **An outline,
/// not a card**: nothing is typed in it, and a white card at the rack's top left, the first thing the eye meets, was taken
/// for the place to write (4/11) — the stages below stay cards, the ports are only drawn around.
struct PortBox<Trailing: View, Content: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let kind: WireKind
    @ViewBuilder var trailing: Trailing
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            BoxTitle(title: title, symbol: symbol, tint: kind.color, quiet: true) {
                trailing.font(.caption).monospacedDigit().foregroundStyle(.secondary)
            }
            content
        }
        // No fill, its name in the secondary color: a recess filled with `well` was the cards' 7 % in dark, the same white.
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(boxShape.strokeBorder(Palette.hairline, lineWidth: 1))
    }
}

extension PortBox where Trailing == EmptyView {
    init(title: LocalizedStringKey, symbol: String, kind: WireKind, @ViewBuilder content: () -> Content) {
        self.init(title: title, symbol: symbol, kind: kind, trailing: { EmptyView() }, content: content)
    }
}

/// **A setting in a box**: its name on the left, its controls on the right — or under it, when the
/// box is too narrow for both on one line. **At its right, the series' column** (`BracketMark.gutter`): the ± of a knob that can
/// vary (`bracket`), empty for the others, so that every control ends on one line — a ± shown on hover after some controls only
/// pushed those in (the user, 4/11/2026: « Random » out of line with what is above it).
struct Field<Controls: View>: View {
    let title: LocalizedStringKey
    var app: AppState? = nil
    var bracket: SeriesKey? = nil
    @ViewBuilder var controls: Controls

    var body: some View {
        if let app, let bracket {
            line.bracketable(app, bracket)
        } else {
            line.padding(.trailing, BracketMark.gutter)
        }
    }

    private var line: some View {
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
        case .sq: String(localized: "The weights are stored in 4 to 5 bits (SQ, Siliconed’s own format) and decoded to 8 bits at each product.")
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

// **The rack's inputs, read only**: what the next render sends, written and chosen in the Prompt Lab under the image
// (`LabView.swift`) — the rack says what goes, it never takes it twice, the strength of a starting image included. Two small
// ports side by side, each above the encoder it feeds, so that each wire drops straight down: the text over the text
// encoder, the images over the image encoder. In outline, not cards (`PortBox`). A click on either unfolds the lab (⌥⌘L).

extension View {
    /// A port: read here, set in the Prompt Lab — a click unfolds it.
    fileprivate func opensLab(help: Text) -> some View {
        contentShape(Rectangle())
            .onTapGesture { LabModel.shared.unfold() }
            .help(help)
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isButton)
    }
}

/// **The text, as a port**: where it comes from, not the text — it is read in the bar under the image. Named « Text », what
/// the text encoder under it reads: « Prompt » sent the eye looking for a field here.
struct PromptPort: View {
    @Bindable var app: AppState

    var body: some View {
        PortBox(title: "Text", symbol: "text.quote", kind: .text) {
            Text(app.prompt.isEmpty ? "to write in the Prompt Lab" : "from the Prompt Lab")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .opensLab(help: Text("Written in the Prompt Lab, under the image (⌥⌘L)"))
    }
}

/// **The images the prompt works on, numbered as it names them**, on a model that edits: image 1 is the one edited — the
/// output takes its format —, the others are what the prompt borrows from, then the subjects the text cites. Their roles are
/// chosen in the Prompt Lab (a drop on the canvas, « Send to the Prompt Lab » or « Edit with … » in the history, ⌘V).
struct ImagesPort: View {
    @Bindable var app: AppState
    private var used: Int { app.references.count }

    var body: some View {
        PortBox(title: "Images", symbol: "photo.on.rectangle", kind: .image) {
            Text(verbatim: "\(used)/\(app.maxReferences)")
                .foregroundStyle(used > app.maxReferences ? Palette.failure : .secondary)
        } content: {
            if used == 0 {
                Text("Optional").font(.caption).foregroundStyle(.secondary)
            } else {
                HStack(spacing: 4) {
                    ForEach(Array(app.references.enumerated()), id: \.element.id) { n, reference in
                        PortThumbnail(image: reference, number: n + 1)
                    }
                }
            }
            // One more image beyond this Mac's memory (B): said here, read before adding one in the lab.
            if app.canAddReference, let refusal = app.nextReferenceRefusal() {
                Text(refusal).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        .opensLab(help: hint)
    }

    private var hint: Text {
        if app.references.isEmpty {
            return Text("Optional. Drop an image in the Prompt Lab and the prompt becomes an instruction: “change her jacket to red”.")
        }
        if app.promptNamesImages && app.maxReferences > 1 {
            return Text("Image 1 is the one edited. The prompt can name them: “put the dog of image 2 next to the woman in image 1”.")
        }
        return Text("Image 1 is the one edited: the prompt says what changes.")
    }
}

/// **The image the render starts from** (img2img), on a model that starts from one: its thumbnail and its strength, read
/// here — the strength is set on its tile in the Prompt Lab, one place. Orange when it is so high that the image is ignored.
struct StartPort: View {
    @Bindable var app: AppState

    var body: some View {
        PortBox(title: "Starting Image", symbol: "photo", kind: .image) {
            if let start = app.usedStart {
                HStack(alignment: .center, spacing: 8) {
                    PortThumbnail(image: start, number: nil)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Strength").foregroundStyle(.secondary)
                        Text(verbatim: StartWords.percent(app.startStrength)).monospacedDigit()
                            .foregroundStyle(app.startIgnored ? Color(nsColor: .systemOrange) : .primary)
                    }
                    .font(.caption)
                }
            }
        }
        .opensLab(help: Text(verbatim: app.startCaption.text + "\n" + StartWords.explanation(app.startRedraws)))
    }
}

/// A port's image, small: its number when the prompt can name it (1 in the accent: the one edited), the image wire's edge.
private struct PortThumbnail: View {
    static let side: CGFloat = 34
    let image: ReferenceImage
    let number: Int?

    /// A subject the prompt cites: it comes and goes with its `@name`.
    private var cited: Bool { if case .subject = image.source { true } else { false } }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: 5, style: .continuous)
        Image(nsImage: image.vignette).resizable().scaledToFill()
            .frame(width: Self.side, height: Self.side)
            .clipShape(shape)
            .overlay(shape.strokeBorder(WireKind.image.color.opacity(number ?? 1 == 1 ? 1 : 0.6), lineWidth: 1.5))
            .overlay(alignment: .topLeading) {
                if let number {
                    Text(verbatim: "\(number)")
                        .font(.system(size: 9, weight: .bold)).monospacedDigit()
                        .foregroundStyle(number == 1 ? AnyShapeStyle(Palette.onLive) : AnyShapeStyle(.white))
                        .frame(width: 13, height: 13)
                        .background(Circle().fill(number == 1 ? AnyShapeStyle(Palette.live) : AnyShapeStyle(.black.opacity(0.6))))
                        .padding(2)
                }
            }
            .help(cited ? Text("\(image.name), cited by the prompt: delete the name there to take it out")
                        : Text(verbatim: "\(image.name) · \(image.width) × \(image.height)"))
            .accessibilityLabel(Text(verbatim: number.map { "\($0). \(image.name)" } ?? image.name))
    }
}
