// **The rack: the chain of the render, drawn from `model.chain`, as the window's left column**.
//
//   ┌ Text ──────────┐ ┌ Images ──── 1/3 ┐   the inputs, read only, in outline: written and chosen in the Prompt Lab
//   │ from the Lab   │ │ ▣ ▣             │   under the image (`LabView.swift`). The images port only on a model that
//   └───────┬────────┘ └──┬──────┬───────┘   edits (its images, numbered) or starts from an image (img2img: the
//           │text  ╭──────╯image │image      starting image and its strength, set on its tile in the lab)
//   ┌Text encoder────┐ ┌Image encoder────┐   Qwen-Image-2.1's text encoder sees the images too: one short wire
//   └───────┬────────┘ └───────┬─────────┘   across, from the images into the text encoder's right shoulder
//           │text              │latent
//     ┌──────── DiT ────────────┐   model ▾, license · fp32 · latent, size, seed, images, detail, step previews,
//     └────────────┬────────────┘   LoRA stack — and, running, its steps and preview
//                  │latent
//     ┌ VAE decoder ────────────┐
//     └─────────────────────────┘   → the image, on the canvas: a result, not a stage
//
// **A module and its settings are one box**: what the next render reads sits in the box of the stage
// that reads it — the size and the seed in the DiT. The inputs are the exception, read only: the text and
// the images are the Prompt Lab's, and the rack shows what goes without taking it twice. The latent is
// not a box: it is what the DiT makes, its previews are drawn in the DiT. Nor is the image: it is the
// result, shown on the canvas. « Generate » is not in the rack either: it is at the end of the text, in the
// lab, where the hand is when the idea is written. The wires between the boxes are of their true type (text
// peach, image rose, latent sky — `Palette.swift`), and run top to bottom, in the direction of the
// signal, each port above the encoder it feeds. **Color is spent only where
// something happens**: at rest the boxes are quiet cards, only their icons in their wire's color, and the wires
// muted; during a render the box of the stage at work wears what runs (mint at night, indigo by day) and the
// wire it reads brightens and widens; what is done keeps
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
    /// Set: it crosses from the images port to the text encoder beside it, into the encoder's right shoulder.
    var across = false
    var id: String { "\(from)-\(to)" }
}

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
                    let (p, end) = e.across ? Self.across(a, b) : Self.drop(a, b)
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

    /// From the images port's left part into the text encoder's right shoulder: a short curve across, clear of the prompt's
    /// wire, which drops into the encoder's middle.
    private static func across(_ a: CGRect, _ b: CGRect) -> (Path, CGPoint) {
        let start = CGPoint(x: a.minX + a.width * 0.22, y: a.maxY)
        let end = CGPoint(x: b.minX + b.width * 0.8, y: b.minY)
        let dy = max(8, (end.y - start.y) * 0.7)
        var p = Path()
        p.move(to: start)
        p.addCurve(to: end, control1: CGPoint(x: start.x, y: start.y + dy), control2: CGPoint(x: end.x, y: end.y - dy))
        return (p, end)
    }
}

// MARK: - The rack

struct RackView: View {
    @Bindable var app: AppState
    /// Set while the encoders' boxes have their top above the column's: the sidebar then pins the chain in small (`MiniChain`).
    var chainAway: Binding<Bool>? = nil
    @State private var hoversTitle = false

    /// The head of the rack, where the mini chain's click scrolls back.
    static let top = "rack.top"
    /// The DiT's box, where `silicontrol rack dit` scrolls.
    static let dit = "rack.dit"

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            header.id(Self.top)
            // The inputs, each above the encoder it feeds: the columns of the two rows are the same halves.
            HStack(alignment: .top, spacing: 10) {
                PromptPort(app: app).node(.prompt)
                if app.editingPossible { ImagesPort(app: app).node(.images) }
                else if showsStart { StartPort(app: app).node(.images) }
            }
            HStack(alignment: .top, spacing: 10) {
                textEncoder.node(.textEncoder)
                if encodesImages { imageEncoder.node(.imageEncoder) }
            }
            // Away once their titles and lights are: on a tall window the rack's bottom still shows their feet.
            .onGeometryChange(for: Bool.self) { $0.frame(in: .scrollView).minY < 0 } action: { chainAway?.wrappedValue = $0 }
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
    private var encodesImages: Bool { (app.editingPossible && app.chain?.encoding != nil) || showsStart }

    /// A starting image is carried, and the model starts from it (img2img): its port shows it, wired to the image encoder alone.
    private var showsStart: Bool { !app.editingPossible && app.usedStart != nil }

    private var edges: [Edge] {
        let live = !app.references.isEmpty
        var e = [Edge(from: .prompt, to: .textEncoder, kind: .text, stage: .text)]
        if showsStart {
            // The text encoder does not see a starting image: only the image encoder reads it, and the DiT starts from its latent.
            e.append(Edge(from: .images, to: .imageEncoder, kind: .image, stage: .image))
            e.append(Edge(from: .imageEncoder, to: .dit, kind: .latent, stage: .denoising))
        }
        if app.editingPossible {
            // Qwen-Image-2.1's encoder sees the images (`TextFormat.readsImages`): one more wire.
            if app.promptNamesImages { e.append(Edge(from: .images, to: .textEncoder, kind: .image, stage: .text, live: live, across: true)) }
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
            // The engine's plan comes from this Mac's measure (`Diagnostic.Report`): the title and the Mac's own
            // icon open it — greyed until a measure is kept (the first render waits for it), in its colours after.
            let measured = app.mapMeasured
            Button { app.openDiagnostic() } label: {
                HStack(spacing: 6) {
                    Text("Chain").font(.headline).fixedSize()
                    MacIcon(hardwareModel: app.computingHardwareModel)
                        .frame(width: 26, height: 26)
                        .saturation(measured ? 1 : 0).opacity(measured ? 1 : 0.5)
                }
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(hoversTitle ? Palette.well : .clear))
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .onHover { hoversTitle = $0 }
            .padding(.leading, -6)
            .accessibilityLabel(Text("Measure This Mac"))
            .accessibilityHint(Text(DiagnosticSheet.mapStatus(app)))
            .help(DiagnosticSheet.mapStatus(app))
            ComputeOnMenu(app: app)
            Spacer()
            Button("Models…") { app.managementOpen = true }
                .buttonStyle(.chartLink)
                .font(.callout)
                .help("Install, import, remove — and what each takes on disk (⇧⌘M)")
                .fixedSize()
        }
    }

    /// **What the chain computes in, for the version installed and the « Accelerators » switch, on this Mac**: the Standard is the
    /// publisher's weights in fp32, each stage verified against the reference; a Compact or a Light computes in fp32 too, on 8- or 4-bit
    /// weights as a third party published them — « verified against the reference » is not said of them. Part of the DiT goes where this
    /// Mac's map puts the accelerators (`AcceleratorWords.place`): the Neural Engine, the GPU's tensor units, both, or neither.
    private var precision: String {
        let quantized = app.card.flatMap { app.installedVariant(of: $0.family) }.map { $0 != .standard } ?? false
        if let place = AcceleratorWords.place(app.acceleratorsInEffect) {
            return quantized ? String(localized: "The weights as published, in fp32 except part of the DiT \(place)")
                             : String(localized: "fp32, except part of the DiT \(place)")
        }
        return quantized ? String(localized: "The weights as published, computed in fp32")
                         : String(localized: "fp32 end to end, every stage verified against the reference")
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
            Text(precision)
                .fixedSize(horizontal: false, vertical: true)
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
    }

    // ── the fixed stages ──

    private var textEncoder: some View {
        StageBox(title: "Text encoder", symbol: "character.textbox", kind: .text, state: app.state(of: .text),
                 explanation: "A language model reads the prompt and turns it into what the DiT understands") {
            ModuleSlot(name: app.moduleNames.text)
            WeightsTag(format: app.weightFormats?.text)
            if app.promptNamesImages {
                Tag(text: Text("sees the images"), symbol: "eye")
                    .help("The encoder reads the edit’s images with the prompt: the prompt can name them (image 1, image 2)")
            }
            Cost(app.stageCost(.text))
        }
    }

    private var imageEncoder: some View {
        StageBox(title: "Image encoder", symbol: "photo", kind: .latent, state: app.state(of: .image),
                 explanation: showsStart ? "Turns the starting image into the latent the DiT starts from, noised to the strength"
                                         : "Turns the images to edit into the compact form the DiT works in, the latent") {
            ModuleSlot(name: app.moduleNames.encoding)
            WeightsTag(format: app.weightFormats?.vae)
            Cost(app.sentImages == 0 ? (nil, false) : app.stageCost(.image))
        }
    }

    private var vae: some View {
        StageBox(title: "VAE decoder", symbol: "square.grid.3x3.square", kind: .image, state: app.state(of: .decoding),
                 explanation: "Turns the DiT’s result, the latent, into pixels: the final image") {
            ModuleSlot(name: app.moduleNames.decoding)
            WeightsTag(format: app.weightFormats?.vae)
            Cost(app.stageCost(.decoding))
        }
    }
}

// MARK: - What the next render costs

/// **The next render, before the click** (rule 7), under the chain that makes it: the size, the memory the engine will compare
/// to this Mac's (computed), how many renders a series makes, and the time — learned on this Mac in this session, never a
/// constant from another machine. Here, not under the text (the user, 4/11/2026: « c'est de la technique pure »).
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
            .foregroundStyle(.secondary)
            .labelStyle(CompactLabel())
        }
    }

    @ViewBuilder private func figures(_ card: ModelCard, _ f: RecommendedFormat) -> some View {
        Label {
            Text(verbatim: "\(f.width) × \(f.height)").monospacedDigit()
        } icon: { Image(systemName: "aspectratio") }
        if let need = app.memoryNeed {
            Label {
                Text(Figures.memory(need)).monospacedDigit()
            } icon: { Image(systemName: "memorychip") }
            .help(String(localized: "The memory \(card.name) needs at this size (computed)"))
        }
        if app.seriesRenders > 1 {
            Label {
                Text("\(app.seriesRenders) renders").monospacedDigit()
            } icon: { Image(systemName: "square.grid.3x3") }
            .help(app.seriesHelp)
        }
        Label {
            if let t = app.plannedCost {
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
                Button("Other Macs…") { SettingsWindow.shared.show(w.local, pane: .macs) }
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
            case .connected?: (Palette.link, false)
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
