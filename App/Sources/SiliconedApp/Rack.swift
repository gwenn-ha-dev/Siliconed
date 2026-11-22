// **The window's column: the chain of the render, from the idea to the button, top to bottom** — one column.
//
//   ┌ Text ───────────── for <model> ● ┐   written here (`LabBoxes.swift`): the field grows with its text, its edge drags
//   │ a lighthouse on a cliff…         │
//   └──┬───────────────────────────────┘
//   ┌ Images ──────────────────── 1/3 ● ┐   the images the render reads, each with its role; one to describe, or to start
//   └──┼──────────────────┬─────────────┘   from. The text's wire passes behind it
//   ▣ Text encoder  Qwen3-4B      1.4 s     one line each: icon, name, module, time — a click opens the « Chain »
//   ▣ Image encoder VAE Flux      0.3 s     inspector at that stage, where its weights, memory and unit are said
//   ┌──┴──────────── DiT ──────────┴───┐   model ▾ Models…, license · weights · latent, size, seed, images, detail,
//   └────────────────┬─────────────────┘   LoRA stack — and, running, its steps
//   ▣ VAE decoder   VAE Flux      0.7 s
//   ≈ 1024 × 1024 · ≈ 24 s
//   ─────────────────────────────────── pinned at the column's foot: what stops the render, the axes of a series, Generate
//
// **A module and its settings are one box**: what the next render reads sits in the box of the stage that reads it — the size
// and the seed in the DiT. The latent is not a box: it is what the DiT makes. Nor is the image: it is the result, shown on the
// canvas. The wires are of their true type (text peach, image rose, latent sky — `Palette.swift`), and run top to bottom in
// the direction of the signal, each in its lane — the text's on the left, the images' on the right, the latent down the
// middle —, under the boxes they pass. **Color is spent only where something happens**: at rest the boxes are quiet cards,
// only their icons in their wire's color, and the wires muted; during a render the stage at work wears what runs (mint at
// night, indigo by day) and the wire it reads brightens and widens; what is done keeps a discreet check; where the engine
// refused, it turns red. Each stage says its time: what it just took, set as soon as the stage ends — before, the one learned
// on this Mac. Each stage's name stays the technical one — it is what the illustrator will meet elsewhere —; hovering it
// says, in plain words, what it does.
//
// **Only the DiT has choices of module today** — the model (the publisher's, or an imported fine-tune) and its LoRA stack.
// The text encoder and the VAE are fixed by the family: one line each. **The license stays in the DiT's box**: it is the one
// that says what may be done with the image.
//
// The boxes take the column's width: the column is narrow or wide (`RackColumn`), the boxes follow, and a row of controls
// that no longer fits on one line goes on two (`Field`).

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

/// The boxes' shape: a continuous rounded rectangle, the system's own.
let boxShape = RoundedRectangle(cornerRadius: 12, style: .continuous)

/// What a wire carries: three types, three hues, the same everywhere in the app.
enum WireKind {
    case text, image, latent
    var color: Color {
        switch self {
        case .text: Palette.peach
        case .image: Palette.rose
        case .latent: Palette.sky
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
    /// Where it runs across the column's width, 0 at the left edge: the text's lane, the images', the latent's.
    var lane: CGFloat
    var id: String { "\(from)-\(to)" }

    static let textLane: CGFloat = 0.22, imageLane: CGFloat = 0.78, crossLane: CGFloat = 0.6, latentLane: CGFloat = 0.5
}

/// **The wires**: from the bottom of a box to the top of another, straight down in their lane, under the boxes they pass. In
/// its color, muted at rest; the wire the working stage reads, full and wider; an empty optional input, dashed gray. A small
/// round plug where it enters. Nothing moves — the width and the brightness say it.
private struct Wires: View {
    let edges: [Edge]
    let frames: [Node: Anchor<CGRect>]
    let state: (Engine.Stage) -> StageState

    var body: some View {
        GeometryReader { g in
            Canvas { c, _ in
                for e in edges {
                    guard let a = frames[e.from].map({ g[$0] }), let b = frames[e.to].map({ g[$0] }) else { continue }
                    let (p, end) = Self.drop(a, b, lane: e.lane)
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

    /// Down from `a` into `b`, in its lane: the boxes all take the column's width, so the lane is the same x in both.
    private static func drop(_ a: CGRect, _ b: CGRect, lane: CGFloat) -> (Path, CGPoint) {
        let x = a.minX + a.width * lane
        let start = CGPoint(x: x, y: a.maxY), end = CGPoint(x: b.minX + b.width * lane, y: b.minY)
        var p = Path()
        p.move(to: start)
        p.addLine(to: end)
        return (p, end)
    }
}

// MARK: - The rack

struct RackView: View {
    @Bindable var app: AppState
    /// Set while the encoders' boxes have their top above the column's: the sidebar then pins the chain in small (`MiniChain`).
    var chainAway: Binding<Bool>? = nil

    /// The head of the rack, where the mini chain's click scrolls back.
    static let top = "rack.top"
    /// The DiT's box, where `silicontrol rack dit` scrolls.
    static let dit = "rack.dit"

    var body: some View {
        let lab = LabModel.shared
        VStack(alignment: .leading, spacing: 14) {
            TextBox(model: lab).node(.prompt).id(Self.top)
                // Away once the text has left the column's top: the head pins the chain in small, and the text's first words.
                .onGeometryChange(for: Bool.self) { $0.frame(in: .scrollView).maxY < 24 } action: { chainAway?.wrappedValue = $0 }
            ImagesBox(model: lab).node(.images)
            textEncoder.node(.textEncoder)
            if encodesImages { imageEncoder.node(.imageEncoder) }
            DiTBox(app: app).node(.dit).id(Self.dit)
            vae.node(.vae)
            RenderEstimate(app: app)
            legend
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 14)
        .backgroundPreferenceValue(NodeFrames.self) { frames in
            Wires(edges: edges, frames: frames, state: app.state(of:))
        }
        .overlay { LinkEdge(app: app) }
    }

    /// The chosen model edits by reference and has an image encoder — or starts from the image the rack holds: its box is drawn.
    private var encodesImages: Bool { app.drawsImageEncoder }

    /// A starting image is carried, and the model starts from it (img2img): its port shows it, wired to the image encoder alone.
    private var showsStart: Bool { !app.editingPossible && app.usedStart != nil }

    private var edges: [Edge] {
        let live = !app.references.isEmpty
        var e = [Edge(from: .prompt, to: .textEncoder, kind: .text, stage: .text, lane: Edge.textLane)]
        if showsStart {
            // The text encoder does not see a starting image: only the image encoder reads it, and the DiT starts from its latent.
            e.append(Edge(from: .images, to: .imageEncoder, kind: .image, stage: .image, lane: Edge.imageLane))
            e.append(Edge(from: .imageEncoder, to: .dit, kind: .latent, stage: .denoising, lane: Edge.imageLane))
        }
        if app.editingPossible {
            // An encoder that sees the images (`TextFormat.readsImages`): one more wire, into the encoder just under.
            if app.promptNamesImages {
                e.append(Edge(from: .images, to: .textEncoder, kind: .image, stage: .text, live: live, lane: Edge.crossLane))
            }
            if encodesImages {
                e.append(Edge(from: .images, to: .imageEncoder, kind: .image, stage: .image, live: live, lane: Edge.imageLane))
                e.append(Edge(from: .imageEncoder, to: .dit, kind: .latent, stage: .denoising, live: live, lane: Edge.imageLane))
            }
        }
        e += [Edge(from: .textEncoder, to: .dit, kind: .text, stage: .denoising, lane: Edge.textLane),
              Edge(from: .dit, to: .vae, kind: .latent, stage: .decoding, lane: Edge.latentLane)]
        return e
    }

    // ── what the rack says above and below ──

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
        }
        .font(.caption2)
        .foregroundStyle(Palette.secondary)
    }

    // ── the fixed stages ──

    private var textEncoder: some View {
        StageRow(title: "Text encoder", symbol: "character.textbox", kind: .text, state: app.state(of: .text),
                 explanation: "A language model reads the prompt and turns it into what the DiT understands",
                 module: app.moduleNames.text, cost: app.stageCost(.text), stage: .text)
    }

    private var imageEncoder: some View {
        StageRow(title: "Image encoder", symbol: "photo", kind: .latent, state: app.state(of: .image),
                 explanation: showsStart ? "Turns the starting image into the latent the DiT starts from, noised to the strength"
                                         : "Turns the images to edit into the compact form the DiT works in, the latent",
                 module: app.moduleNames.encoding, cost: app.stageCost(.image), stage: .image)
    }

    private var vae: some View {
        StageRow(title: "VAE decoder", symbol: "square.grid.3x3.square", kind: .image, state: app.state(of: .decoding),
                 explanation: "Turns the DiT’s result, the latent, into pixels: the final image",
                 module: app.moduleNames.decoding, cost: app.stageCost(.decoding), stage: .decoding)
    }
}

// MARK: - What the next render costs

/// **The next render, before the click**, under the chain that makes it: the size, how many renders a series makes, and the
/// time — learned on this Mac in this session, never a constant from another machine. Here, not under the text: these are
/// the chain's, not the idea's.
struct RenderEstimate: View {
    @Bindable var app: AppState

    var body: some View {
        if app.missing == nil, let card = app.card {
            let f = app.outputFormat
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 12) { figures(card, f) }
                VStack(alignment: .leading, spacing: 4) { figures(card, f) }
            }
            .font(.caption)
            .foregroundStyle(Palette.secondary)
            .labelStyle(CompactLabel())
        }
    }

    @ViewBuilder private func figures(_ card: ModelCard, _ f: RecommendedFormat) -> some View {
        // Read once: each reading builds the series' grid again.
        let renders = app.seriesRenders
        Label {
            Text(verbatim: Figures.size(f.width, f.height)).monospacedDigit()
        } icon: { Image(systemName: "aspectratio") }
        if renders > 1 {
            Label {
                Text("\(renders) renders").monospacedDigit()
            } icon: { Image(systemName: "square.grid.3x3") }
            .help(app.seriesHelp)
        }
        Label {
            if let t = app.plannedCost(renders: renders) {
                Text(verbatim: "≈ " + Figures.duration(t)).monospacedDigit()
            } else {
                Text("after one render")
            }
        } icon: { Image(systemName: "clock") }
        .help("Learned on this Mac: the first render at this size, with this model, sets the estimate")
    }
}

/// An icon and its text, close together.
private struct CompactLabel: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 4) { configuration.icon; configuration.title }
    }
}

// MARK: - Where the chain computes

/// **« On: This Mac ▾ »**, in the window's top band beside the rack's fold: the whole chain moves to a paired Mac, or
/// back (`MainWindow.compute(on:)`) — the Render menu's « Compute On », where the hand already is. Shown only once a
/// Mac is paired to compute on, and seen with the column folded; one Bonjour does not see is grayed, and says so.
struct ComputeOnMenu: View {
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
                Button("Other Macs…") { SettingsWindow.shared.show(.macs) }
            } label: {
                // By their middles: on a baseline, the Mac's icon — an image, its baseline its foot — stood above the words.
                HStack(alignment: .center, spacing: 6) {
                    MacIcon(hardwareModel: app.computingHardwareModel).frame(width: 20, height: 20)
                    Text("On: \(shown?.name ?? String(localized: "This Mac"))").font(.callout)
                        .lineLimit(1).truncationMode(.middle)
                }
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
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
            case .connected?: (Palette.link, false)
            case .refused?: (Palette.failure, false)
            default: (Palette.caution, true)
            }
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(color, style: StrokeStyle(lineWidth: 2, dash: dashed ? [6, 4] : []))
                .padding(4)
                .allowsHitTesting(false)
        }
    }
}
