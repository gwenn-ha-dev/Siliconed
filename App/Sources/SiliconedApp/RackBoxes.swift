// The rack's box furniture: the stages' boxes and lines, the inputs' boxes, fields, tags, cost. Split from `Rack.swift`,
// whose header says how the boxes behave.

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
                     explanation: Text(explanation)) {
                // No spinner while working: the edge and the wash say it, and the one under the image
                // says the render runs — three turning at once was noise.
                switch state {
                case .idle, .active: EmptyView()
                case .done: Image(systemName: "checkmark").font(.caption2.weight(.bold)).foregroundStyle(Palette.secondary)
                case .failed: Image(systemName: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(Palette.failure)
                }
            }
            content
        }
        .boxed(state)
        .animation(Motion.pick(.easeInOut(duration: 0.25)), value: state)
    }
}

/// A box's first line: its icon (`IconTile`), its name, and what it says on the right.
private struct BoxTitle<Trailing: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let tint: Color
    /// Shown when the pointer rests on the icon or the name. Only one `.help` there: an inner empty one hid the box's own.
    var explanation: Text? = nil
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 7) {
            HStack(spacing: 7) {
                IconTile(symbol: symbol, color: tint)
                Text(title).font(.callout.weight(.semibold))
                    .lineLimit(2).fixedSize(horizontal: false, vertical: true)
            }
            .contentShape(Rectangle())
            .help(explanation ?? Text(verbatim: ""))
            Spacer(minLength: 4)
            trailing
        }
    }
}

extension View {
    /// The box: padding, the card's fill, its edge — and, while it runs or failed, a faint wash of
    /// that color.
    fileprivate func boxed(edge: Color, width: CGFloat = 1, wash: Color? = nil, vertical: CGFloat = 12) -> some View {
        padding(.horizontal, 12).padding(.vertical, vertical)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(boxShape.fill(Palette.surface))
            .overlay { if let wash { boxShape.fill(wash.opacity(0.06)).allowsHitTesting(false) } }
            .overlay(boxShape.strokeBorder(edge, lineWidth: width))
            .shadow(color: .black.opacity(0.08), radius: 2, y: 1)
    }

    /// A stage's box as its state says: the hairline at rest or done, the accent's edge and wash while it works, red where
    /// the engine refused.
    fileprivate func boxed(_ state: StageState, vertical: CGFloat = 12) -> some View {
        boxed(edge: state == .active ? Palette.live : state == .failed ? Palette.failure : Palette.hairline,
              width: state == .idle || state == .done ? 1 : 1.5,
              wash: state == .active ? Palette.live : state == .failed ? Palette.failure : nil, vertical: vertical)
    }
}

/// **A box the signal starts in** — the text, the images: written and chosen here, the only place they come from. A card as
/// the stages are, its icon in the color of the wire it sends, and on its right whether something goes.
struct InputBox<Trailing: View, Content: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let kind: WireKind
    /// Something goes by it: its icon full and a dot of its wire's color; nothing, the icon faint and a ring.
    let fed: Bool
    var explanation: Text? = nil
    @ViewBuilder var trailing: Trailing
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            BoxTitle(title: title, symbol: symbol, tint: kind.color.opacity(fed ? 1 : 0.4), explanation: explanation) {
                HStack(spacing: 6) {
                    trailing.font(.caption).monospacedDigit().foregroundStyle(Palette.secondary)
                    Circle().fill(fed ? kind.color : .clear)
                        .overlay(Circle().strokeBorder(fed ? kind.color : Palette.secondary, lineWidth: 1))
                        .frame(width: 8, height: 8)
                        .accessibilityHidden(true)
                }
            }
            content
        }
        .boxed(edge: Palette.hairline)
    }
}

/// **A stage the family fixes, on one line**: its icon, its name, its module, its time — the text encoders, the VAE. Its
/// weights, what it held and who computed it are the « Chain » inspector's: a click opens it at this stage's line. Lit as a
/// box is while it works, red where the engine refused.
struct StageRow: View {
    let title: LocalizedStringKey
    let symbol: String
    let kind: WireKind
    let state: StageState
    let explanation: LocalizedStringKey
    let module: String?
    let cost: (seconds: Double?, measured: Bool)
    let stage: Engine.Stage

    var body: some View {
        Button { MainWindow.shared.inspect(stage) } label: {
            HStack(spacing: 7) {
                IconTile(symbol: symbol, color: state == .active ? Palette.live : state == .failed ? Palette.failure : kind.color)
                Text(title).font(.callout.weight(.semibold)).lineLimit(1).fixedSize()
                Text(verbatim: module ?? "—").font(.callout).foregroundStyle(Palette.secondary)
                    .lineLimit(1).truncationMode(.middle)
                Spacer(minLength: 4)
                if state == .failed {
                    Image(systemName: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(Palette.failure)
                }
                Cost(cost).fixedSize()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(Text("\(Text(explanation)) — \(Text("The Chain inspector shows its weights, its time and its memory"))"))
        .boxed(state, vertical: 9)
        .animation(Motion.pick(.easeInOut(duration: 0.25)), value: state)
    }
}

/// **A setting in a box**: its name on the left, its controls on the right — or under it, when the
/// box is too narrow for both on one line. **At its right, one column every line keeps, filled or not** (`BracketMark.gutter`):
/// the line's one mark — the ± of a knob that can vary (`bracket`), or else its accessory (`accessory`: the pencil of a size,
/// the ✕ that brings a value back, a lock). So every control ends on one line and every mark stands in one column: an
/// accessory among the controls pushed them in by its width, and a ✕ that comes and goes moved its stepper (issue 16: five
/// right edges in the DiT's box). A line with a ± keeps what else it has among its controls (the seed's die).
struct Field<Controls: View, Accessory: View>: View {
    let title: LocalizedStringKey
    var app: AppState? = nil
    var bracket: SeriesKey? = nil
    @ViewBuilder var controls: Controls
    @ViewBuilder var accessory: Accessory

    var body: some View {
        if let app, let bracket {
            line.bracketable(app, bracket)
        } else {
            HStack(spacing: 0) {
                line
                // The column is kept whether the accessory is there or not (a ✕ that comes and goes), and its glyph is centered
                // where a ± is: as wide as it, against the column's right edge.
                Color.clear.frame(width: BracketMark.gutter, height: 16)
                    .overlay(alignment: .trailing) { accessory.frame(width: 16) }
            }
        }
    }

    private var line: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
                Text(title).foregroundStyle(Palette.secondary)
                Spacer(minLength: 8)
                controls
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(title).foregroundStyle(Palette.secondary)
                HStack(spacing: 8) { controls }
            }
        }
        .font(.callout)
    }
}

extension Field where Accessory == EmptyView {
    init(title: LocalizedStringKey, app: AppState? = nil, bracket: SeriesKey? = nil, @ViewBuilder controls: () -> Controls) {
        self.init(title: title, app: app, bracket: bracket, controls: controls, accessory: { EmptyView() })
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

/// **How a module's weights are stored** (`WeightFormat`): « weights int8 », the name verbatim and the word said in the user's
/// language. The hover says what it is in plain words.
struct WeightsTag: View {
    let format: WeightFormat?
    var body: some View {
        if let format {
            Tag(text: Text("weights \(format.label)"), symbol: "shippingbox")
                .help(Self.explanation(format))
        }
    }

    static func explanation(_ f: WeightFormat) -> String {
        switch f {
        case .int8: String(localized: "The weights are stored in 8 bits (int8), as published: a quarter of fp32’s size.")
        case .fp8: String(localized: "The weights are stored in 8 bits (fp8), as published.")
        case .bf16: String(localized: "The weights are stored in bf16, as the publisher released them: half of fp32’s size.")
        case .fp16: String(localized: "The weights are stored in fp16: half of fp32’s size.")
        case .fp32: String(localized: "The weights are stored in fp32.")
        }
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
        .foregroundStyle(Palette.secondary)
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
        .foregroundStyle(Palette.secondary)
        .help(measured ? "What this stage just took"
                       : "Learned on this Mac: the first render at this size, with this model, sets the estimate")
    }
}

// MARK: - A control greyed, and why

extension View {
    /// **A control that cannot act now, saying why on hover** — the app's one rule for a greyed button: `reason` `nil`, the
    /// control as it is, its own help; a reason, it is greyed and the reason is its help. A disabled control shows no help
    /// (AppKit gives a disabled control no tooltip), so the reason is carried by a clear layer over it, outside the
    /// disabled subtree — which also keeps the click from reaching it. VoiceOver reads the reason as the control's hint.
    func unavailable(_ reason: String?) -> some View { modifier(Unavailable(reason: reason)) }
}

private struct Unavailable: ViewModifier {
    let reason: String?

    func body(content: Content) -> some View {
        content
            .disabled(reason != nil)
            .accessibilityHint(reason.map { Text(verbatim: $0) } ?? Text(verbatim: ""))
            .overlay {
                if let reason {
                    Rectangle().fill(.clear).contentShape(Rectangle()).help(reason).accessibilityHidden(true)
                }
            }
    }
}
