// **The rack: the chain of the render, drawn from `model.chain`, as the window's left column**.
//
//     ┌ Images ─────────────────┐   only if the model edits: the images the prompt works on, 1 to 3,
//   ┌─┤ ▣ ▣ +                   ├─┐ numbered — or, with « Named Subjects » on (experimental,
//   │ │                         │ │ `Subjects.swift`) and the box switched to Subjects, the named
//   │ └─────────────────────────┘ │ subjects the prompt cites by @name: one or the other
//   │ ┌ Prompt ─────────────────┐ │ the text field itself, under what it names
//   │ └──────────┬──────────────┘ │
//   │ ┌Text encoder┐ ┌Image encoder┐
//   └▶│ (sees them)│ │             │◀┘
//     └─────┬──────┘ └──────┬──────┘
//           │text           │latent
//     ┌──────── DiT ────────────┐   model ▾, license · fp32 · latent, size, seed, images, steps, detail,
//     └────────────┬────────────┘   step previews, LoRA stack — and, running, its steps and preview
//                  │latent
//     ┌ VAE decoder ────────────┐
//     └─────────────────────────┘   → the image, on the canvas: a result, not a stage
//
// **A module and its settings are one box**: what the next render reads sits in the box of the stage
// that reads it — the images in the images, the prompt in the prompt, the size, the seed and the
// steps in the DiT. There is no other form. The latent is not a box: it is what the DiT makes, its
// previews are drawn in the DiT. Nor is the image: it is the result, shown on the canvas. The wires
// between the boxes are of their true type (text amber, image lavender, latent sea green — `Palette`),
// and run top to bottom, in the direction of the signal; the images come first, as the gesture does —
// lay the picture down, then say what to make of it —, so their wires skip the prompt: down the left
// rail into the text encoder, down the right one into the image encoder. **Color is spent only where
// something happens**: at rest the boxes are quiet cards and the wires muted; during a render the box
// of the stage at work wears the accent and the wire it reads brightens and widens; what is done keeps
// a discreet check; where the engine refused, the box turns red. Under each box, what it costs: the
// time it just took, set as soon as the stage ends — before, the one learned on this Mac. The memory a render needs is the whole chain's: it is said once, in the
// summary under the rack (`Sidebar`), not under the DiT. Each stage's name stays the technical one —
// it is what the illustrator will meet elsewhere —; hovering it says, in plain words, what it does.
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
        a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? NSColor(white: 1, alpha: 0.07) : NSColor(white: 1, alpha: 0.85)
    })
    /// A field inside a box: a recess.
    static let well = Color.primary.opacity(0.08)
    /// A card's edge, a separator: a notch firmer than the system's `separatorColor` (10 %), which
    /// let a box melt into the column on both grounds.
    static let hairline = Color(nsColor: NSColor(name: nil) { a in
        a.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? NSColor(white: 1, alpha: 0.16) : NSColor(white: 0, alpha: 0.14)
    })

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
    /// Set: it skips the box between its ends, and runs down that side's rail instead of through it.
    var rail: Rail? = nil
    var id: String { "\(from)-\(to)" }
}

private enum Rail { case left, right }

/// The rails' distance to the boxes: the column's paddings leave them room.
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
                    let (p, end) = e.rail.map { Self.rail(a, b, $0) } ?? Self.drop(a, b)
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

    /// Out of `a`'s side, down the rail, into `b`'s same side: past the boxes in between.
    private static func rail(_ a: CGRect, _ b: CGRect, _ side: Rail) -> (Path, CGPoint) {
        let left = side == .left, r: CGFloat = left ? 6 : -6
        let x = left ? min(a.minX, b.minX) - railInset : max(a.maxX, b.maxX) + railInset
        let start = CGPoint(x: left ? a.minX : a.maxX, y: a.maxY - 16)
        let end = CGPoint(x: left ? b.minX : b.maxX, y: b.minY + 18)
        let turn = abs(r)
        var p = Path()
        p.move(to: start)
        p.addLine(to: CGPoint(x: x + r, y: start.y))
        p.addQuadCurve(to: CGPoint(x: x, y: start.y + turn), control: CGPoint(x: x, y: start.y))
        p.addLine(to: CGPoint(x: x, y: end.y - turn))
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
            if app.editingPossible { ImagesNode(app: app).node(.images) }
            PromptNode(app: app).node(.prompt)
            HStack(alignment: .top, spacing: 10) {
                textEncoder.node(.textEncoder)
                if encodesImages { imageEncoder.node(.imageEncoder) }
            }
            DiTBox(app: app).node(.dit)
            vae.node(.vae)
            legend
        }
        .padding(.leading, 12 + railInset)
        .padding(.trailing, 12 + railInset)
        .padding(.vertical, 14)
        .backgroundPreferenceValue(NodeFrames.self) { frames in
            Wires(edges: edges, frames: frames, state: app.state(of:))
        }
    }

    /// The chosen model edits by reference and has an image encoder: its box is drawn.
    private var encodesImages: Bool { app.editingPossible && app.chain?.encoding != nil }

    private var edges: [Edge] {
        // An edit's images and the subjects a prompt cites are the same images to the model: one wire each way.
        let live = !app.references.isEmpty || app.subjectImageCount > 0
        var e = [Edge(from: .prompt, to: .textEncoder, kind: .text, stage: .text)]
        if app.editingPossible {
            // Qwen-Image-2.1's encoder sees the images (`TextFormat.readsImages`): one more wire.
            if app.promptNamesImages { e.append(Edge(from: .images, to: .textEncoder, kind: .image, stage: .text, live: live, rail: .left)) }
            if encodesImages {
                e.append(Edge(from: .images, to: .imageEncoder, kind: .image, stage: .image, live: live, rail: .right))
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
            // The engine's settings come from this Mac's measure (`CapacityRules`): the title and its
            // chip open it, orange until it is taken — the first render waits for it — or while the one
            // kept is provisional (taken off rest), a green chip once a current one is kept.
            let measured = app.storedMap.map { $0.isCurrent && !$0.isProvisional } == true
            Button { app.openDiagnostic() } label: {
                HStack(spacing: 6) {
                    Text("Chain").font(.headline)
                    Image(systemName: "cpu").font(.body).foregroundStyle(measured ? .green : .orange)
                }
                .foregroundStyle(measured ? .primary : Color.orange)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(DiagnosticSheet.mapStatus(app))
            Spacer()
            // « Explore » is beside « Generate » (`Sidebar`): a gesture that makes images, not a setting.
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
            Text("fp32 end to end, every stage verified against the reference; on request, part of the DiT on the Neural Engine")
                .fixedSize(horizontal: false, vertical: true)
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
    }

    // ── the fixed stages ──

    private var textEncoder: some View {
        StageBox(title: "Text encoder", symbol: "character.textbox", state: app.state(of: .text),
                 explanation: "A language model reads the prompt and turns it into what the DiT understands") {
            ModuleSlot(name: app.chain?.text.name)
            if app.promptNamesImages {
                Tag(text: Text("sees the images"), symbol: "eye")
                    .help("The encoder reads the edit's images with the prompt: the prompt can name them (image 1, image 2)")
            }
            Cost(app.stageCost(.text))
        }
    }

    private var imageEncoder: some View {
        StageBox(title: "Image encoder", symbol: "photo", state: app.state(of: .image),
                 explanation: "Turns the images to edit into the compact form the DiT works in, the latent") {
            ModuleSlot(name: app.chain?.encoding?.name)
            Cost(app.references.isEmpty && app.subjectImageCount == 0 ? (nil, false) : app.stageCost(.image))
        }
    }

    private var vae: some View {
        StageBox(title: "VAE decoder", symbol: "square.grid.3x3.square", state: app.state(of: .decoding),
                 explanation: "Turns the DiT's result, the latent, into pixels: the final image") {
            ModuleSlot(name: app.chain?.decoding.name)
            Cost(app.stageCost(.decoding))
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
    /// What the stage does, in plain words: shown when the pointer rests on its name.
    let explanation: LocalizedStringKey
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            BoxTitle(title: title, symbol: symbol, tint: state == .active ? Palette.live : .secondary,
                     explanation: explanation) {
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
    var explanation: LocalizedStringKey? = nil
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 7) {
            HStack(spacing: 7) {
                Image(systemName: symbol)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(tint)
                    .frame(width: 20, height: 20)
                    .background(RoundedRectangle(cornerRadius: 5, style: .continuous).fill(tint.opacity(0.16)))
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
            Image(systemName: "lock.fill").imageScale(.small).foregroundStyle(.secondary)
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

/// The time a stage costs: what it just took in the render on screen (no « ≈ »), or the estimate
/// learned on this Mac — or nothing yet.
private struct Cost: View {
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
/// under it gives it the height one wants; Esc leaves it (the arrows then walk the strip).
private struct PromptNode: View {
    @Bindable var app: AppState
    @FocusState private var typing: Bool
    /// The named subjects' field says itself when it has the keyboard (`PromptField`).
    @State private var fieldTyping = false
    @State private var height: CGFloat = 72
    @GestureState private var stretch: CGFloat = 0
    private static let heights: ClosedRange<CGFloat> = 44...480

    var body: some View {
        PortBox(title: "Prompt", symbol: "text.quote", kind: .text, focused: typing || fieldTyping) {
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    // Named subjects on: the field that draws `@name` as chips. Off: the app as it was.
                    if app.subjectsActive {
                        PromptField(app: app, focused: $fieldTyping)
                    } else {
                        TextEditor(text: $app.prompt)
                            .font(.body)
                            .scrollContentBackground(.hidden)
                            .scrollIndicators(.automatic)
                            .focused($typing)
                    }
                    if app.prompt.isEmpty {
                        Text(app.references.isEmpty ? "Describe the image…" : "Say what changes…")
                            .font(.body).foregroundStyle(.tertiary)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
                .frame(height: min(max(height + stretch, Self.heights.lowerBound), Self.heights.upperBound))
                grip
                if app.subjectsActive, let rewrite = app.shownRewrite { ModelPromptRow(app: app, rewrite: rewrite) }
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
/// the model). **With Named Subjects on, the box is images OR subjects** (`AppState.referenceMode`):
/// numbered images, or a library of named subjects — as many as one wants — that the prompt cites by
/// `@name`, 3 at most. The title counts what the next render sends.
private struct ImagesNode: View {
    @Bindable var app: AppState
    private var subjects: Bool { app.subjectsActive }
    private var used: Int { subjects ? app.subjectImageCount : app.references.count }
    var body: some View {
        PortBox(title: "Images", symbol: "photo.on.rectangle", kind: .image) {
            Text(used == 0 ? String(localized: "up to \(app.maxReferences) images") : "\(used)/\(app.maxReferences)")
                .foregroundStyle(used > app.maxReferences ? Palette.failure : .secondary)
        } content: {
            if app.subjectsAvailable {
                Picker("Images", selection: $app.referenceMode) {
                    Text("Images").tag(AppState.ReferenceMode.images)
                    Text("Subjects").tag(AppState.ReferenceMode.subjects)
                }
                .pickerStyle(.segmented).labelsHidden()
                .padding(.bottom, 2)
            }
            if subjects { SubjectShelf(app: app) } else { images }
        }
        .dropDestination(for: DroppedImage.self) { items, _ in
            if subjects {
                // Here an image is named: the sheet opens on the first one.
                switch items.first {
                case .file(let url)? where !AppState.importable(url): app.nameSubject(.file(url))
                case .png(let data)?: app.nameSubject(.png(data))
                default: return false
                }
                return true
            }
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

// MARK: - The DiT: the stage with choices

/// **The DiT and everything it is told**: the model, its license, the size, the seed, how many
/// images, the steps, the LoRA stack — then what it costs, or its steps while it runs.
private struct DiTBox: View {
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
        // Developer Mode, Qwen-Image-2.1: the σ ComfyUI walks for this model, to compare with the official ones.
        if app.offersComfySchedule {
            Toggle("ComfyUI schedule", isOn: $app.comfySchedule)
                .font(.callout)
                .help("Walks the noise levels ComfyUI uses for this model (the base model's), instead of the Turbo's own: to compare, same seed, same steps")
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
                    .disabled(!app.managementPossible)
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
