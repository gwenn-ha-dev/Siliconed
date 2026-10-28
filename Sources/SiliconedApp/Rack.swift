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
        .overlay { LinkEdge(app: app) }
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
            let measured = app.mapMeasured
            Button { app.openDiagnostic() } label: {
                HStack(spacing: 6) {
                    Text("Chain").font(.headline).fixedSize()
                    Image(systemName: "cpu").font(.body).foregroundStyle(measured ? .green : .orange)
                }
                .foregroundStyle(measured ? .primary : Color.orange)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(DiagnosticSheet.mapStatus(app))
            ComputeOnMenu(app: app)
            Spacer()
            // « Explore » is beside « Generate » (`Sidebar`): a gesture that makes images, not a setting.
            Button("Models…") { app.managementOpen = true }
                .buttonStyle(.link)
                .font(.callout)
                .help("Install, import, remove — and what each takes on disk (⇧⌘M)")
                .fixedSize()
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
            ModuleSlot(name: app.moduleNames.text)
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
            ModuleSlot(name: app.moduleNames.encoding)
            Cost(app.references.isEmpty && app.subjectImageCount == 0 ? (nil, false) : app.stageCost(.image))
        }
    }

    private var vae: some View {
        StageBox(title: "VAE decoder", symbol: "square.grid.3x3.square", state: app.state(of: .decoding),
                 explanation: "Turns the DiT's result, the latent, into pixels: the final image") {
            ModuleSlot(name: app.moduleNames.decoding)
            Cost(app.stageCost(.decoding))
        }
    }
}

// MARK: - Where the chain computes

/// **« On: This Mac ▾ »**, beside « Chain »: the whole chain moves to a paired Mac, or back
/// (`MainWindow.compute(on:)`) — the Render menu's « Compute On », where the hand already is. Shown
/// only once a Mac is paired to compute on; one Bonjour does not see is grayed, and says so.
private struct ComputeOnMenu: View {
    let app: AppState

    var body: some View {
        let w = MainWindow.shared
        if let borrowed = w.otherMacs.pairings?.borrowed, !borrowed.isEmpty {
            let shown = borrowed.first { $0.id == w.carrier.showing }
            Menu {
                Toggle("This Mac", isOn: Binding(get: { shown == nil }, set: { if $0 { w.compute(on: nil) } }))
                Divider()
                ForEach(borrowed, id: \.id) { b in
                    let found = w.otherMacs.found[b.id] != nil
                    Toggle(isOn: Binding(get: { shown?.id == b.id }, set: { if $0 { w.compute(on: b) } })) {
                        if found || shown?.id == b.id { Text(verbatim: b.name) }
                        else { Text("\(b.name) — not on the network") }
                    }
                    .disabled(!found && shown?.id != b.id)
                    // What is known of it: chip, memory, B, map, models.
                    if let machine = w.otherMacs.machines[b.id] { Text(verbatim: machine.summary) }
                }
                Divider()
                Button("Other Macs…") { SettingsWindow.shared.show(w.local) }
            } label: {
                // A long name gives way, never « Chain » or « Models… » beside it.
                Text("On: \(shown?.name ?? String(localized: "This Mac"))").font(.callout)
                    .lineLimit(1).truncationMode(.middle)
            }
            .menuStyle(.borderlessButton)
            .fixedSize(horizontal: false, vertical: true)
            .layoutPriority(-1)
            .help(shown.map { OtherMacsSection.state(w.otherMacs.links[$0.id], found: w.otherMacs.found[$0.id] != nil) }
                  ?? String(localized: "The chain computes on this Mac"))
        }
    }
}

/// **The rack's edge while another Mac computes**: blue while connected, orange and dashed while the
/// link is being made or retried, red once refused — a line where the chain is, not a tint over it.
private struct LinkEdge: View {
    let app: AppState

    var body: some View {
        let w = MainWindow.shared
        if app.mirror != nil, let id = w.carrier.showing {
            let (color, dashed): (Color, Bool) = switch w.otherMacs.links[id] {
            case .connected?: (Color(nsColor: .systemBlue), false)
            case .refused?: (Palette.failure, false)
            default: (Color(nsColor: .systemOrange), true)
            }
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(color, style: StrokeStyle(lineWidth: 2, dash: dashed ? [6, 4] : []))
                .padding(4)
                .allowsHitTesting(false)
        }
    }
}
