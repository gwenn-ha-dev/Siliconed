// **The rack: the chain of the render, drawn from `model.chain`, as the window's left column**.
//
//     ┌ Prompt ─────────────────┐   the text field itself
//   ┌─┤                         │
//   │ └─────────────────────────┘
//   │ ┌ Images ─────────────────┐   only if the model edits: 1 to 3 numbered thumbnails
//   │ └────┬──────────────┬─────┘
//   │      │image         │image
//   │ ┌Text encoder┐ ┌Image encoder┐
//   └▶│ (sees them)│ │             │
//     └─────┬──────┘ └──────┬──────┘
//           │text           │latent
//     ┌──────── DiT ────────────┐   model ▾, license · fp32 · latent, size, seed, images, steps, detail,
//     └────────────┬────────────┘   step previews, LoRA stack — and, running, its steps and preview
//                  │latent
//     ┌ VAE decoder ────────────┐
//     └─────────────────────────┘   → the image, on the canvas: a result, not a stage
//
// **A module and its settings are one box**: what the next render reads sits in the box of the stage
// that reads it — the prompt in the prompt, an edit's images in the images, the size, the seed and the
// steps in the DiT. There is no other form. The latent is not a box: it is what the DiT makes, its
// previews are drawn in the DiT. Nor is the image: it is the result, shown on the canvas. The wires
// between the boxes are of their true type (text amber, image lavender, latent sea green — `Palette`),
// and run top to bottom, in the direction of the signal; the one wire that skips a box (the prompt to
// the text encoder, past an edit's images) runs down the left rail. **Color is spent only where
// something happens**: at rest the boxes are quiet cards and the wires muted; during a render the box
// of the stage at work wears the accent and the wire it reads brightens and widens; what is done keeps
// a discreet check; where the engine refused, the box turns red. Under each box, what it costs: the
// time learned on this Mac, the DiT's memory peak (rule 7).
//
// **Only the DiT has choices of module today** — the model (the publisher's, or an imported
// fine-tune) and its LoRA stack. The text encoder and the VAE are fixed by the family: their module
// sits in a *slot* that says so (a lock, no empty menu). The slot is where an alternative encoder or
// VAE will plug in when the library can import one — the drawing will not change, the lock will
// become a menu.
//
// The boxes take the column's width: the column is narrow or wide (`MainView`), the boxes follow,
// and a row of controls that no longer fits on one line goes on two (`Field`).

import AppKit
import Siliconed
import SwiftUI

// MARK: - The nodes and the wires

/// The places of the rack: each reports its frame, and the wires are drawn between them.
private enum Node: Hashable { case prompt, images, textEncoder, imageEncoder, dit, vae }

private struct NodeFrames: PreferenceKey {
    static let defaultValue: [Node: Anchor<CGRect>] = [:]
    static func reduce(value: inout [Node: Anchor<CGRect>], nextValue: () -> [Node: Anchor<CGRect>]) {
        value.merge(nextValue()) { $1 }
    }
}

extension View {
    fileprivate func node(_ n: Node) -> some View {
        anchorPreference(key: NodeFrames.self, value: .bounds) { [n: $0] }
    }
}

/// **The rack's colors.** Three hues for the three wires — amber, lavender, sea green — at the same
/// lightness and the same chroma (OKLCH 0.80 / 0.105 on a dark ground, 0.58 / 0.12 on a light one), so
/// that none shouts over the others and each reads on both grounds. Everything else is the system's:
/// the accent for what runs, red for what failed, the separator and the control colors for the rest.
enum Palette {
    static let amber = dynamic(dark: 0xE5B46E, light: 0xA36E09)
    static let lavender = dynamic(dark: 0xC8AEF8, light: 0x8668B6)
    static let seaGreen = dynamic(dark: 0x67D4C0, light: 0x00917D)
    /// What runs: the system's accent, as a selection or a progress bar elsewhere in macOS.
    static let live = Color.accentColor
    static let failure = Color(nsColor: .systemRed)
    /// A box: a card a step above the column — lifted in dark, white in light, as System Settings does.
    static let surface = Color(nsColor: NSColor(name: nil) { a in
        a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? NSColor(white: 1, alpha: 0.055) : NSColor(white: 1, alpha: 0.85)
    })
    /// A field inside a box: a recess.
    static let well = Color.primary.opacity(0.06)
    static let hairline = Color(nsColor: .separatorColor)

    private static func dynamic(dark: Int, light: Int) -> Color {
        func color(_ v: Int) -> NSColor {
            NSColor(srgbRed: CGFloat(v >> 16 & 0xFF) / 255, green: CGFloat(v >> 8 & 0xFF) / 255, blue: CGFloat(v & 0xFF) / 255, alpha: 1)
        }
        return Color(nsColor: NSColor(name: nil) { a in
            a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? color(dark) : color(light)
        })
    }
}

/// The boxes' shape: a continuous rounded rectangle, the system's own.
let boxShape = RoundedRectangle(cornerRadius: 10, style: .continuous)

/// What a wire carries: three types, three hues, the same everywhere in the app.
enum WireKind {
    case text, image, latent
    var color: Color {
        switch self {
        case .text: Palette.amber
        case .image: Palette.lavender
        case .latent: Palette.seaGreen
        }
    }
}

private struct Edge: Identifiable {
    let from: Node, to: Node
    let kind: WireKind
    /// The stage whose work moves it: it runs while that stage is active, stays lit once done.
    let stage: Engine.Stage
    /// False: an optional input left empty (no edit image) — drawn thin and faint.
    var live = true
    /// True: it skips the box between its ends, and runs down the left rail instead of through it.
    var rail = false
    var id: String { "\(from)-\(to)" }
}

/// The left rail's distance to the boxes: the column's leading padding leaves it room.
private let railInset: CGFloat = 9

/// **The wires**: from the bottom of a box to the top of the next, straight down where the two
/// overlap — a box under another reads where it sits. In its color, muted at rest; the wire the
/// working stage reads, full and wider; an empty optional input, dashed gray. A small round plug where
/// it enters. Nothing moves — the width and the brightness say it.
private struct Wires: View {
    let edges: [Edge]
    let frames: [Node: Anchor<CGRect>]
    let state: (Engine.Stage) -> StageState

    var body: some View {
        GeometryReader { g in
            Canvas { c, _ in
                for e in edges {
                    guard let a = frames[e.from].map({ g[$0] }), let b = frames[e.to].map({ g[$0] }) else { continue }
                    let (p, end) = e.rail ? Self.rail(a, b) : Self.drop(a, b)
                    let s = e.live ? state(e.stage) : .idle
                    let color: Color = !e.live ? Palette.hairline
                        : s == .idle ? e.kind.color.opacity(0.85) : e.kind.color
                    let width: CGFloat = s == .active ? 3 : 1.75
                    c.stroke(p, with: .color(color),
                             style: StrokeStyle(lineWidth: width, lineCap: .round, dash: e.live ? [] : [3, 4]))
                    let r: CGFloat = s == .active ? 4 : 3
                    c.fill(Path(ellipseIn: CGRect(x: end.x - r, y: end.y - r, width: 2 * r, height: 2 * r)), with: .color(color))
                }
            }
        }
        .allowsHitTesting(false)
    }

    /// Down from `a` into `b`: at the middle of what they share horizontally, so that two boxes
    /// side by side each send their wire straight down into the box below.
    private static func drop(_ a: CGRect, _ b: CGRect) -> (Path, CGPoint) {
        let lo = max(a.minX, b.minX), hi = min(a.maxX, b.maxX)
        let x0 = lo < hi ? (lo + hi) / 2 : a.midX
        let x1 = lo < hi ? x0 : b.midX
        let start = CGPoint(x: x0, y: a.maxY), end = CGPoint(x: x1, y: b.minY)
        let dy = max(8, (end.y - start.y) * 0.6)
        var p = Path()
        p.move(to: start)
        p.addCurve(to: end, control1: CGPoint(x: start.x, y: start.y + dy), control2: CGPoint(x: end.x, y: end.y - dy))
        return (p, end)
    }

    /// Out of `a`'s left side, down the rail, into `b`'s left side: past the boxes in between.
    private static func rail(_ a: CGRect, _ b: CGRect) -> (Path, CGPoint) {
        let x = min(a.minX, b.minX) - railInset, r: CGFloat = 6
        let start = CGPoint(x: a.minX, y: a.maxY - 16), end = CGPoint(x: b.minX, y: b.minY + 18)
        var p = Path()
        p.move(to: start)
        p.addLine(to: CGPoint(x: x + r, y: start.y))
        p.addQuadCurve(to: CGPoint(x: x, y: start.y + r), control: CGPoint(x: x, y: start.y))
        p.addLine(to: CGPoint(x: x, y: end.y - r))
        p.addQuadCurve(to: CGPoint(x: x + r, y: end.y), control: CGPoint(x: x, y: end.y))
        p.addLine(to: end)
        return (p, end)
    }
}

// MARK: - The rack

struct RackView: View {
    @Bindable var app: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            header
            PromptNode(app: app).node(.prompt)
            if app.editingPossible { ImagesNode(app: app).node(.images) }
            HStack(alignment: .top, spacing: 10) {
                textEncoder.node(.textEncoder)
                if encodesImages { imageEncoder.node(.imageEncoder) }
            }
            DiTBox(app: app).node(.dit)
            vae.node(.vae)
            legend
        }
        .padding(.leading, 12 + railInset)
        .padding(.trailing, 14)
        .padding(.vertical, 14)
        .backgroundPreferenceValue(NodeFrames.self) { frames in
            Wires(edges: edges, frames: frames, state: app.state(of:))
        }
    }

    /// The chosen model edits by reference and has an image encoder: its box is drawn.
    private var encodesImages: Bool { app.editingPossible && app.chain?.encoding != nil }

    private var edges: [Edge] {
        let live = !app.references.isEmpty
        var e = [Edge(from: .prompt, to: .textEncoder, kind: .text, stage: .text, rail: app.editingPossible)]
        if app.editingPossible {
            // Qwen-Image-2.1's encoder sees the images (`TextFormat.readsImages`): one more wire.
            if app.promptNamesImages { e.append(Edge(from: .images, to: .textEncoder, kind: .image, stage: .text, live: live)) }
            if encodesImages {
                e.append(Edge(from: .images, to: .imageEncoder, kind: .image, stage: .image, live: live))
                e.append(Edge(from: .imageEncoder, to: .dit, kind: .latent, stage: .denoising, live: live))
            }
        }
        e += [Edge(from: .textEncoder, to: .dit, kind: .text, stage: .denoising),
              Edge(from: .dit, to: .vae, kind: .latent, stage: .decoding)]
        return e
    }

    // ── what the rack says above and below ──

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text("Chain").font(.headline)
            Spacer()
            Button("Models…") { app.managementOpen = true }
                .buttonStyle(.link)
                .font(.callout)
                .help("Install, import, remove — and what each takes on disk (⇧⌘M)")
        }
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 14) {
                ForEach([(WireKind.text, String(localized: "text")), (.image, String(localized: "image")),
                         (.latent, String(localized: "latent"))], id: \.1) { kind, name in
                    HStack(spacing: 5) {
                        Capsule().fill(kind.color).frame(width: 14, height: 3)
                        Text(name)
                    }
                }
            }
            Text("fp32 end to end, every stage verified against the reference")
                .fixedSize(horizontal: false, vertical: true)
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
    }

    // ── the fixed stages ──

    private var textEncoder: some View {
        StageBox(title: "Text encoder", symbol: "character.textbox", state: app.state(of: .text)) {
            ModuleSlot(name: app.chain?.text.name)
            if app.promptNamesImages {
                Tag(text: Text("sees the images"), symbol: "eye")
                    .help("The encoder reads the edit's images with the prompt: the prompt can name them (image 1, image 2)")
            }
            Cost(seconds: app.stageEstimate(.text))
        }
    }

    private var imageEncoder: some View {
        StageBox(title: "Image encoder", symbol: "photo", state: app.state(of: .image)) {
            ModuleSlot(name: app.chain?.encoding?.name)
            Cost(seconds: app.references.isEmpty ? nil : app.stageEstimate(.image))
        }
    }

    private var vae: some View {
        StageBox(title: "VAE decoder", symbol: "square.grid.3x3.square", state: app.state(of: .decoding)) {
            ModuleSlot(name: app.chain?.decoding.name)
            Cost(seconds: app.stageEstimate(.decoding))
        }
    }
}

// MARK: - A box

/// **A stage's box**: its symbol and name, its state on the right, then what fills it. Idle: a quiet
/// card. Working: the accent's edge and a faint wash of it. Done: a discreet check. Refused: red. It
/// takes the width it is given.
private struct StageBox<Content: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let state: StageState
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            BoxTitle(title: title, symbol: symbol, tint: state == .active ? Palette.live : .secondary) {
                switch state {
                case .idle: EmptyView()
                case .active: ProgressView().controlSize(.mini)
                case .done: Image(systemName: "checkmark").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                case .failed: Image(systemName: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(Palette.failure)
                }
            }
            content
        }
        .boxed(edge: edge, width: state == .idle || state == .done ? 1 : 1.5,
               wash: state == .active ? Palette.live : state == .failed ? Palette.failure : nil)
        .animation(.easeInOut(duration: 0.25), value: state)
    }

    private var edge: Color {
        switch state {
        case .idle, .done: Palette.hairline
        case .active: Palette.live
        case .failed: Palette.failure
        }
    }
}

/// A box's first line: a small tinted chip with its symbol, its name, and what it says on the right.
private struct BoxTitle<Trailing: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let tint: Color
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: symbol)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 20, height: 20)
                .background(RoundedRectangle(cornerRadius: 5, style: .continuous).fill(tint.opacity(0.16)))
            Text(title).font(.callout.weight(.semibold))
                .lineLimit(2).fixedSize(horizontal: false, vertical: true)
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
private struct PortBox<Trailing: View, Content: View>: View {
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
        .animation(.easeInOut(duration: 0.15), value: focused)
    }
}

extension PortBox where Trailing == EmptyView {
    init(title: LocalizedStringKey, symbol: String, kind: WireKind, focused: Bool = false, @ViewBuilder content: () -> Content) {
        self.init(title: title, symbol: symbol, kind: kind, focused: focused, trailing: { EmptyView() }, content: content)
    }
}

/// **A setting in a box**: its name on the left, its controls on the right — or under it, when the
/// box is too narrow for both on one line.
private struct Field<Controls: View>: View {
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
private struct ModuleSlot: View {
    let name: String?
    var body: some View {
        HStack(spacing: 5) {
            Text(name ?? "—").lineLimit(1).truncationMode(.middle)
            Spacer(minLength: 2)
            Image(systemName: "lock.fill").imageScale(.small).foregroundStyle(.tertiary)
        }
        .font(.callout)
        .padding(.horizontal, 8).padding(.vertical, 5)
        .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Palette.well))
        .help("Set by the model's family")
    }
}

/// A label: a small capsule, neutral — or with a dot of color (the latent's wire).
private struct Tag: View {
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

/// The time a stage costs, learned on this Mac — or nothing yet.
private struct Cost: View {
    let seconds: Double?
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "clock").imageScale(.small)
            Text(seconds.map { "≈ " + Figures.seconds($0) } ?? "—").monospacedDigit()
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .help("Learned on this Mac: the first render at this size, with this model, sets the estimate")
    }
}

// MARK: - The inputs

/// **The prompt is typed in its box**: what to make — or, when editing, what changes. A real text view
/// (`TextEditor`, an `NSTextView`): ⌘A, ⌘C, ⌘Z, the spelling, a long prompt that scrolls. The grip
/// under it gives it the height one wants; Esc leaves it (the arrows then walk the strip).
private struct PromptNode: View {
    @Bindable var app: AppState
    @FocusState private var typing: Bool
    @State private var height: CGFloat = 72
    @GestureState private var stretch: CGFloat = 0
    private static let heights: ClosedRange<CGFloat> = 44...480

    var body: some View {
        PortBox(title: "Prompt", symbol: "text.quote", kind: .text, focused: typing) {
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $app.prompt)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .scrollIndicators(.automatic)
                        .focused($typing)
                    if app.prompt.isEmpty {
                        Text(app.references.isEmpty ? "Describe the image…" : "Say what changes…")
                            .font(.body).foregroundStyle(.tertiary)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
                .frame(height: min(max(height + stretch, Self.heights.lowerBound), Self.heights.upperBound))
                grip
            }
        }
    }

    /// The handle: dragged, the field grows or shrinks; its cursor says so.
    private var grip: some View {
        Capsule().fill(Color.secondary.opacity(0.35)).frame(width: 34, height: 4)
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

/// **The images of an edit, numbered as the prompt names them**: image 1 is the one edited — the
/// output takes its format —, the others are what the prompt borrows from. Added by the button, by a
/// drop here or on the canvas (there, a PNG Siliconed wrote restores its settings instead), or by
/// « Edit This Image » in the history; reordered by dragging a
/// thumbnail onto another, or from its menu. Only for a model that edits (rule 2: the inputs follow
/// the model).
private struct ImagesNode: View {
    let app: AppState
    var body: some View {
        PortBox(title: "Edit", symbol: "photo.on.rectangle", kind: .image) {
            Text(app.references.isEmpty ? String(localized: "up to \(app.maxReferences) images")
                                        : "\(app.references.count)/\(app.maxReferences)")
        } content: {
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
                    .help(app.references.isEmpty ? "The image to edit — or drop it here"
                                                 : "Add an image the prompt can draw from — or drop it here")
                }
                Spacer(minLength: 0)
            }
            Group {
                if app.references.isEmpty {
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
}

// MARK: - The DiT: the stage with choices

/// **The DiT and everything it is told**: the model, its license, the size, the seed, how many
/// images, the steps, the LoRA stack — then what it costs, or its steps while it runs.
private struct DiTBox: View {
    @Bindable var app: AppState
    @State private var customSize = false

    var body: some View {
        StageBox(title: "DiT", symbol: "sparkles", state: app.state(of: .denoising)) {
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
                    settings
                    Divider().padding(.vertical, 2)
                    loras
                }
            }
            if app.state(of: .denoising) == .active, app.totalSteps > 0, app.currentJob?.settings.identifier == app.identifier {
                progress
            } else {
                HStack(spacing: 10) {
                    Cost(seconds: app.stageEstimate(.denoising))
                    if let need = app.memoryNeed {
                        HStack(spacing: 4) {
                            Image(systemName: "memorychip").imageScale(.small)
                            // Said in words: under "Install…", a bare figure reads as the download.
                            Text("\(Figures.memory(need)) of memory").monospacedDigit()
                        }
                        .font(.caption).foregroundStyle(.secondary)
                        .help("The memory this render needs at this size (measured peak)")
                    }
                }
            }
        }
    }

    private func license(_ card: ModelCard) -> some View {
        Tag(text: Text(verbatim: card.licenseName),
            symbol: card.license.commercial ? "checkmark.seal" : "info.circle")
            .help(card.license.commercial ? String(localized: "Commercial use allowed")
                                          : String(localized: "Non-commercial: personal and research use only"))
    }

    @ViewBuilder private var technical: some View {
        Tag(text: Text(verbatim: "fp32"))
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
                Picker("Size", selection: $app.format) {
                    ForEach(app.pickerFormats, id: \.self) { f in
                        Text(verbatim: "\(f.width) × \(f.height)").tag(f)
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
                Image(systemName: "lock.fill").imageScale(.small).foregroundStyle(.tertiary)
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
        Field(title: "Detail") {
            Picker("Detail", selection: $app.detail) {
                ForEach(Detail.allCases, id: \.self) { Text(verbatim: $0.choice).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden().fixedSize()
        }
        .help("Adds fine texture and small details; Most can over-sharpen")
        Toggle("Preview every step", isOn: $app.previews)
            .font(.callout)
    }

    /// The LoRA stack: one row each, its strength; a row opens its settings.
    @ViewBuilder private var loras: some View {
        VStack(alignment: .leading, spacing: 3) {
            ForEach($app.loras) { $slot in LoRARow(app: app, slot: $slot) }
            HStack {
                if app.canAddLoRA {
                    Button { app.loras.append(LoRASlot()) } label: { Label("LoRA", systemImage: "plus") }
                        .buttonStyle(.borderless)
                        .help("Add a LoRA to the stack (3 at most)")
                } else if app.compatibleLoras.isEmpty {
                    Text("No LoRA for this model").foregroundStyle(.tertiary)
                }
                Spacer()
                Button("Import…") { app.chooseToImport() }
                    .buttonStyle(.link)
                    .disabled(!app.managementPossible)
                    .help("A model (.safetensors or .gguf) or a LoRA (.safetensors), as published on Civitai")
            }
            .font(.caption)
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

/// A LoRA of the stack, in one line; a click opens its choice and strength.
struct LoRARow: View {
    let app: AppState
    @Binding var slot: LoRASlot
    @State private var editing = false

    var body: some View {
        Button { editing = true } label: {
            HStack(spacing: 5) {
                Image(systemName: "slider.horizontal.3").imageScale(.small).foregroundStyle(.secondary)
                Text(app.loraCard(slot.path)?.name ?? String(localized: "Choose…")).lineLimit(1).truncationMode(.middle)
                Spacer(minLength: 4)
                if !slot.path.isEmpty { Text(Figures.strength(slot.strength)).monospacedDigit().foregroundStyle(.secondary) }
            }
            .font(.caption)
            .padding(.horizontal, 8).padding(.vertical, 5)
            .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Palette.well))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onAppear { if slot.path.isEmpty { editing = true } }
        .popover(isPresented: $editing, arrowEdge: .trailing) {
            VStack(alignment: .leading, spacing: 10) {
                Picker("LoRA", selection: $slot.path) {
                    Text("Choose…").tag("")
                    ForEach(app.compatibleLoras) { l in Text(l.name).tag(l.path) }
                }
                HStack {
                    Slider(value: $slot.strength, in: 0...1.5)
                    Text(Figures.strength(slot.strength)).monospacedDigit().frame(width: 38, alignment: .trailing)
                }
                if let f = app.loraCard(slot.path) {
                    Text([String(localized: "rank \(f.rank)"),
                          f.resolution.map { String(localized: "trained at \($0)") }, f.trainedOn]
                            .compactMap { $0 }.joined(separator: " · "))
                        .font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Spacer()
                    Button("Remove", role: .destructive) {
                        // The id is read BEFORE: `slot` is a binding on `app.loras`, and re-reading it
                        // in `removeAll`'s predicate accesses the array during its mutation.
                        let id = slot.id
                        editing = false
                        app.loras.removeAll { $0.id == id }
                    }
                }
            }
            .padding(14)
            .frame(width: 300)
        }
    }
}
