// **The curtain: two images in one frame, a line drawn across them** — what a LoRA, a setting or a
// variation changed, seen at the same place.
//
//     ┌──────────────┬──────────────┐
//     │ 42           ┃           43 │   A (left of the line) over B (right of it): the same pixels
//     │      A       ◀▶      B      │   of the same frame — only the line decides which one shows.
//     │              ┃              │
//     └──────────────┴──────────────┘
//
// Exactly two images marked in the strip, of the same size: the curtain is the default; the switch in
// the bar under the image (`CurtainSwitch`) goes back to side by side, and is remembered. Two sizes:
// side by side only, the switch says why.
//
// - The line is dragged; a click anywhere brings it there. ← → move it by a twentieth of the frame
//   (`MainWindow.key`, only while the curtain is shown — Esc leaves the comparison as before).
// - Zoom (pinch, ⌘+ ⌘− ⌘0 — the app's `zoom`) and the two-finger scroll apply to the two images
//   together, at the same place: one transform, two layers. The line stays where it is on screen.
// - The frame is the viewport: zoomed in, the images are cut at its edges, so the line always runs
//   from top to bottom of what is seen.
// - A is the left one in the strip, B the right one: the order the strip shows them in.
// - Each side's tag says what differs between the two (`Curtain.tags`): only the seed → « Seed 42 » /
//   « Seed 43 »; nothing → their short names, as the remote control gives them.

import AppKit
import SwiftUI

/// What the curtain keeps between two comparisons: its mode (remembered across launches) and where the
/// line is (for the session — the next pair starts where the last one was left).
@MainActor @Observable
final class Curtain {
    static let shared = Curtain()
    private static let key = "comparisonCurtain"

    /// The user's choice with two images; a pair of different sizes is shown side by side whatever it is.
    var preferred: Bool = UserDefaults.standard.object(forKey: Curtain.key) as? Bool ?? true {
        didSet { UserDefaults.standard.set(preferred, forKey: Self.key) }
    }
    /// The line, from 0 (all B) to 1 (all A).
    var position: CGFloat = 0.5

    /// ← → : a twentieth of the frame per press.
    func nudge(_ direction: CGFloat) {
        withAnimation(.easeOut(duration: 0.12)) { position = min(max(position + direction / 20, 0), 1) }
    }

    /// The two images compared, when the curtain can show them: exactly two, of the same size.
    static func pair(_ app: AppState) -> (a: (entry: Entry, image: CGImage), b: (entry: Entry, image: CGImage))? {
        let c = app.compared
        guard c.count == 2, c[0].entry.width == c[1].entry.width, c[0].entry.height == c[1].entry.height else { return nil }
        return (c[0], c[1])
    }

    /// Whether the canvas shows the curtain now: two images marked, or an edit against its original.
    static func shown(_ app: AppState) -> Bool { shared.preferred && pair(app) != nil || app.showsOriginal }

    /// **Each side's tag: what differs between the two**, in the fewest words — the seed, the model,
    /// the steps, the LoRAs, an edit's references, the prompt. Nothing differs (a redo): their names.
    static func tags(_ a: Entry, _ b: Entry) -> (String, String) {
        var parts: [(String, String)] = []
        if a.seed != b.seed {
            parts.append((String(localized: "Seed \(String(a.seed))"), String(localized: "Seed \(String(b.seed))")))
        }
        if a.settings.modelName != b.settings.modelName { parts.append((a.settings.modelName, b.settings.modelName)) }
        if a.settings.steps != b.settings.steps {
            parts.append((String(localized: "\(String(a.settings.steps)) steps"), String(localized: "\(String(b.settings.steps)) steps")))
        }
        if loras(a) != loras(b) { parts.append((loras(a), loras(b))) }
        if a.settings.references != b.settings.references {
            parts.append((references(a.settings.references), references(b.settings.references)))
        }
        if a.settings.prompt != b.settings.prompt { parts.append((short(a.settings.prompt), short(b.settings.prompt))) }
        if parts.isEmpty { return (a.name, b.name) }
        return (parts.map(\.0).joined(separator: " · "), parts.map(\.1).joined(separator: " · "))
    }

    private static func loras(_ e: Entry) -> String {
        let used = e.settings.loras.filter { !$0.path.isEmpty }
        guard !used.isEmpty else { return String(localized: "No LoRA") }
        return used.map { slot in
            var name = URL(fileURLWithPath: slot.path).deletingPathExtension().lastPathComponent
            if name.hasSuffix(".lora") { name = String(name.dropLast(5)) }
            return name + " " + Figures.strength(slot.strength)
        }.joined(separator: ", ")
    }

    private static func references(_ n: Int) -> String {
        switch n {
        case 0: String(localized: "No reference")
        case 1: String(localized: "1 reference")
        default: String(localized: "\(String(n)) references")
        }
    }

    private static func short(_ prompt: String) -> String {
        prompt.count <= 36 ? prompt : String(prompt.prefix(35)) + "…"
    }
}

/// **The curtain itself**, in the place of the side-by-side grid — or of an edit, its original
/// under it. A and B are drawn in a `width × height` frame, the size of the image they compare.
struct CurtainView: View {
    @Bindable var app: AppState
    let a: CGImage
    let b: CGImage
    let width: Int, height: Int
    let tags: (String, String)

    private var curtain: Curtain { .shared }
    /// The two images' common offset when zoomed in, in the frame's points.
    @State private var offset: CGSize = .zero
    @GestureState private var pinch: CGFloat = 1
    @State private var dragging = false
    @State private var frame: CGSize = .zero
    @State private var hovering = false
    @State private var scrollMonitor: Any?

    private var zoom: CGFloat { app.zoom * pinch }

    var body: some View {
        let w = CGFloat(width), h = CGFloat(height)
        let (tagA, tagB) = tags
        ZStack {
            layer(b)
            layer(a)
                .mask(alignment: .leading) {
                    Rectangle().frame(width: frame.width * curtain.position)
                }
        }
        .aspectRatio(w / h, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(alignment: .topLeading) { tag(tagA, .leading).opacity(curtain.position > 0.12 ? 1 : 0) }
        .overlay(alignment: .topTrailing) { tag(tagB, .trailing).opacity(curtain.position < 0.88 ? 1 : 0) }
        .overlay(alignment: .leading) { handle }
        .animation(.easeOut(duration: 0.15), value: curtain.position > 0.12)
        .animation(.easeOut(duration: 0.15), value: curtain.position < 0.88)
        .onGeometryChange(for: CGSize.self) { $0.size } action: { frame = $0 }
        .contentShape(Rectangle())
        .pointerStyle(.columnResize)
        .gesture(lineGesture)
        .simultaneousGesture(zoomGesture)
        .simultaneousGesture(TapGesture(count: 2).onEnded {
            withAnimation(.spring(duration: 0.3)) { app.zoom = 1; offset = .zero }
        })
        .onHover { hovering = $0 }
        .shadow(color: .black.opacity(0.35), radius: 18, y: 6)
        .frame(maxWidth: w, maxHeight: h)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onChange(of: app.zoom) { _, z in withAnimation(.spring(duration: 0.3)) { offset = z == 1 ? .zero : clamped(offset) } }
        .onAppear(perform: watchScroll)
        .onDisappear { scrollMonitor.map(NSEvent.removeMonitor); scrollMonitor = nil }
        .help("Drag the line, or press ← →, to reveal one image or the other")
    }

    /// One image, under the common zoom and offset.
    private func layer(_ image: CGImage) -> some View {
        Image(decorative: image, scale: 1)
            .resizable()
            .interpolation(.high)
            .scaleEffect(zoom)
            .offset(offset)
    }

    /// The line and its knob: white with a shadow, legible on any image; the knob takes the accent
    /// while it is held.
    private var handle: some View {
        let x = frame.width * curtain.position
        return ZStack {
            Rectangle()
                .fill(.white)
                .frame(width: 2)
                .shadow(color: .black.opacity(0.5), radius: 2)
            Image(systemName: "arrow.left.and.right")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(dragging ? Color.white : Color.primary)
                .frame(width: 30, height: 30)
                .background(Circle().fill(dragging ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.regularMaterial)))
                .overlay(Circle().strokeBorder(.white.opacity(0.9), lineWidth: 1.5))
                .shadow(color: .black.opacity(0.35), radius: 4, y: 1)
        }
        .frame(width: 30)
        .offset(x: x - 15)
        .allowsHitTesting(false)
    }

    private func tag(_ text: String, _ side: Alignment) -> some View {
        Text(verbatim: text)
            .font(.caption).monospacedDigit()
            .lineLimit(1).truncationMode(.middle)
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(.regularMaterial, in: Capsule())
            .frame(maxWidth: max(frame.width * 0.45, 60), alignment: side)
            .fixedSize(horizontal: false, vertical: true)
            .padding(10)
            .allowsHitTesting(false)
    }

    /// A press anywhere brings the line there; dragging carries it.
    private var lineGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                dragging = true
                MainWindow.shared.releaseTextFocus()
                guard frame.width > 0 else { return }
                curtain.position = min(max(value.location.x / frame.width, 0), 1)
            }
            .onEnded { _ in dragging = false }
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .updating($pinch) { value, state, _ in state = value.magnification }
            .onEnded { value in
                app.zoom = min(max(app.zoom * value.magnification, 1), 8)
                offset = app.zoom == 1 ? .zero : clamped(offset)
            }
    }

    /// Two fingers (or a wheel) move both images, zoomed in; fitted, the scroll goes on as usual.
    private func watchScroll() {
        guard scrollMonitor == nil else { return }
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            guard hovering, app.zoom > 1 else { return event }
            let k: CGFloat = event.hasPreciseScrollingDeltas ? 1 : 10
            offset = clamped(CGSize(width: offset.width + event.scrollingDeltaX * k,
                                    height: offset.height + event.scrollingDeltaY * k))
            return nil
        }
    }

    /// Zoomed in, the images always cover the frame: no empty band at an edge.
    private func clamped(_ o: CGSize) -> CGSize {
        let x = frame.width * (app.zoom - 1) / 2, y = frame.height * (app.zoom - 1) / 2
        return CGSize(width: min(max(o.width, -x), x), height: min(max(o.height, -y), y))
    }
}

/// **Side by side, or the curtain**: two icons in the bar under the images, the active one in the
/// accent. Two images of different sizes: grayed, and its help says why.
struct CurtainSwitch: View {
    let app: AppState
    private var curtain: Curtain { .shared }

    var body: some View {
        let possible = Curtain.pair(app) != nil
        HStack(spacing: 2) {
            segment("square.split.2x1", active: !curtain.preferred, possible: possible) {
                curtain.preferred = false
            }
            .help(possible ? "Side by side" : "The two images are not the same size: they can only be shown side by side")
            segment("square.lefthalf.filled", active: curtain.preferred, possible: possible) {
                curtain.preferred = true
            }
            .help(possible ? "Curtain: drag the line across the two images (← →)" : "The two images are not the same size: they can only be shown side by side")
        }
        .padding(2)
        .background(Palette.well, in: RoundedRectangle(cornerRadius: 7))
    }

    /// Not `.disabled`: a disabled button shows no help, and the help is the reason.
    private func segment(_ symbol: String, active: Bool, possible: Bool, action: @escaping () -> Void) -> some View {
        Button {
            guard possible else { return }
            withAnimation(.easeInOut(duration: 0.2)) { action() }
        } label: {
            Image(systemName: symbol)
                .imageScale(.medium)
                .frame(width: 30, height: 22)
                .foregroundStyle(!possible ? AnyShapeStyle(.tertiary) : active ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.secondary))
                .background(RoundedRectangle(cornerRadius: 5).fill(active && possible ? Palette.surface : .clear))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
