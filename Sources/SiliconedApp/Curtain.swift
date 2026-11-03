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
// the bar under the image (`CurtainSwitch`) goes back to side by side, or to the switch — and is remembered.
// Two sizes: side by side only, the switch says why.
//
// - The line is dragged; a click anywhere brings it there. ← → move it by a twentieth of the frame
//   (`MainWindow.key`, only while the curtain is shown — Esc leaves the comparison as before).
// - **The switch** (`Mode.alternate`, issue #10): the same two layers, the line hidden and parked at an end — one image
//   at a time, at the same place and zoom, which is what the eye compares best. A click on the image shows the other;
//   ← shows A and → shows B (A is the left one in the strip). The swap is instant: a fade would blend them.
// - **Zoomed in, a drag moves the images** (issue #10, `CurtainGesture`): anywhere in the switch — a click still
//   switches —, anywhere but on the line in the curtain, whose line then moves only by its handle and ← →.
// - Zoom (pinch, a mouse's wheel, ⌘+ ⌘− ⌘0 — the app's `zoom`) and the two-finger scroll apply to the two images
//   together, at the same place: one transform, two layers. The line stays where it is on screen.
//   A zoom keeps the point under the pointer still (`ZoomAnchor`), as on the canvas.
// - The frame is the viewport: zoomed in, the images are cut at its edges, so the line always runs
//   from top to bottom of what is seen.
// - A is the left one in the strip, B the right one: the order the strip shows them in.
// - Each side's tag says what differs between the two (`Curtain.tags`): only the seed → « Seed 42 » /
//   « Seed 43 »; nothing → their short names, as the remote control gives them.

import AppKit
import Siliconed
import SwiftUI

/// What the curtain keeps between two comparisons: its mode (remembered across launches) and where the
/// line is (for the session — the next pair starts where the last one was left).
@MainActor @Observable
final class Curtain {
    static let shared = Curtain()
    /// The mode's key.
    private static let key = "comparisonMode"

    /// How two images of one size are shown.
    enum Mode: String { case sideBySide, curtain, alternate }

    /// The user's choice with two images; a pair of different sizes is shown side by side whatever it is.
    var mode: Mode = Curtain.storedMode() {
        didSet {
            UserDefaults.standard.set(mode.rawValue, forKey: Self.key)
            // Entering the switch shows A; leaving it, the line comes back to the middle.
            if mode == .alternate { position = 1 } else if oldValue == .alternate { position = 0.5 }
        }
    }
    /// The two images in one frame — the curtain or the switch, not side by side.
    var preferred: Bool { mode != .sideBySide }
    var alternates: Bool { mode == .alternate }

    private static func storedMode() -> Mode {
        UserDefaults.standard.string(forKey: key).flatMap(Mode.init(rawValue:)) ?? .curtain
    }

    /// The line, from 0 (all B) to 1 (all A).
    var position: CGFloat = Curtain.storedMode() == .alternate ? 1 : 0.5

    /// ← → : the curtain moves its line a twentieth of the frame per press; the switch shows A (←) or B (→).
    func nudge(_ direction: CGFloat) {
        if alternates { position = direction < 0 ? 1 : 0; return }
        withAnimation(Motion.pick(.easeOut(duration: 0.12))) { position = min(max(position + direction / 20, 0), 1) }
    }

    /// The switch: the other image.
    func swap() { position = position > 0.5 ? 0 : 1 }

    /// The two images compared, when the curtain can show them: exactly two, of the same proportions —
    /// an enlargement and its original are one picture at two sizes, each layer is fitted to the frame.
    static func pair(_ app: AppState) -> (a: (entry: Entry, image: CGImage), b: (entry: Entry, image: CGImage))? {
        let c = app.compared
        guard c.count == 2, c[0].entry.width * c[1].entry.height == c[1].entry.width * c[0].entry.height else { return nil }
        return (c[0], c[1])
    }

    /// Whether the canvas shows the curtain now: two images marked, or an edit against its original.
    static func shown(_ app: AppState) -> Bool { shared.preferred && pair(app) != nil || app.showsOriginal }

    /// **Each side's tag: what differs between the two**, in the fewest words — the seed, the model,
    /// the steps, the LoRAs, an edit's references, the prompt. Nothing differs (a redo): their names.
    static func tags(_ a: Entry, _ b: Entry) -> (String, String) {
        var parts: [(String, String)] = []
        if a.width != b.width {
            parts.append(("\(a.width)×\(a.height)", "\(b.width)×\(b.height)"))
        }
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
    /// What the press under way carries (`CurtainGesture.role`), decided where it began; `nil` between presses.
    @State private var pressRole: CurtainGesture.Role?
    /// How far the press under way has moved the images (zoomed in), not yet added to `offset`.
    @State private var panning: CGSize = .zero
    @State private var frame: CGSize = .zero
    @State private var hovering = false
    /// Where the pointer is over the frame — what a zoom keeps still (`ZoomAnchor`).
    @State private var pointer: CGPoint?
    /// The offset a pinch just settled, applied once its zoom lands (`onChange(of: app.zoom)`).
    @State private var settled: CGSize?
    @State private var scrollMonitor: Any?

    private var zoom: CGFloat { app.zoom * pinch }

    var body: some View {
        let w = CGFloat(width), h = CGFloat(height)
        let (tagA, tagB) = tags
        ZStack {
            layer(b, liveOffset)
            layer(a, liveOffset)
                .mask(alignment: .leading) {
                    Rectangle().frame(width: frame.width * curtain.position)
                }
        }
        .aspectRatio(w / h, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(alignment: .topLeading) { tag(tagA, .leading).opacity(curtain.position > 0.12 ? 1 : 0) }
        .overlay(alignment: .topTrailing) { tag(tagB, .trailing).opacity(curtain.position < 0.88 ? 1 : 0) }
        .overlay(alignment: .leading) { if !curtain.alternates { handle } }
        .animation(Motion.pick(.easeOut(duration: 0.15)), value: curtain.position > 0.12)
        .animation(Motion.pick(.easeOut(duration: 0.15)), value: curtain.position < 0.88)
        .onGeometryChange(for: CGSize.self) { $0.size } action: { frame = $0 }
        .contentShape(Rectangle())
        .pointerStyle(cursor)
        .gesture(pressGesture)
        .simultaneousGesture(zoomGesture)
        .simultaneousGesture(TapGesture(count: 2).onEnded {
            withAnimation(Motion.pick(.spring(duration: 0.3))) { app.zoom = 1; offset = .zero }
        })
        .onContinuousHover { phase in
            switch phase {
            case .active(let p): hovering = true; pointer = p
            case .ended: hovering = false; pointer = nil
            }
        }
        .shadow(color: .black.opacity(0.35), radius: 18, y: 6)
        .frame(maxWidth: w, maxHeight: h)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onChange(of: app.zoom) { z1, z2 in
            let target = settled ?? ZoomAnchor.offset(offset, from: z1, to: z2, at: pointer, in: frame)
            settled = nil
            withAnimation(Motion.pick(.spring(duration: 0.3))) { offset = z2 == 1 ? .zero : clamped(target) }
        }
        .onAppear(perform: watchScroll)
        .onDisappear { scrollMonitor.map(NSEvent.removeMonitor); scrollMonitor = nil; hovering = false }
        .help(help)
        // The line is dragged: VoiceOver moves it with the adjustable actions instead (up and down).
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: tagA + " / " + tagB))
        .accessibilityValue(Text(verbatim: curtain.alternates ? (curtain.position > 0.5 ? tagA : tagB)
                                                               : Figures.percent(Int((curtain.position * 100).rounded()))))
        .accessibilityAdjustableAction { direction in
            curtain.nudge(direction == .increment ? 1 : -1)
        }
    }

    /// One image, under the common zoom and offset.
    private func layer(_ image: CGImage, _ o: CGSize) -> some View {
        Image(decorative: image, scale: 1)
            .resizable()
            .interpolation(.high)
            .scaleEffect(zoom)
            .offset(o)
    }

    /// The offset as drawn: during a drag, where the hand has carried the images; during a pinch, moved so that the
    /// point under the fingers stays there.
    private var liveOffset: CGSize {
        let o = panning == .zero ? offset : clamped(CGSize(width: offset.width + panning.width, height: offset.height + panning.height))
        return pinch == 1 ? o : ZoomAnchor.offset(o, from: app.zoom, to: zoom, at: pointer, in: frame)
    }

    private var zoomed: Bool { app.zoom > 1 }
    private var lineX: CGFloat { frame.width * curtain.position }

    /// The pointer says what a press would carry: the line (on it, or anywhere when fitted), the images (zoomed in),
    /// nothing to drag in the fitted switch.
    private var cursor: PointerStyle {
        let role = pressRole ?? CurtainGesture.role(alternates: curtain.alternates, zoomed: zoomed,
                                                    startX: pointer?.x ?? -.infinity, lineX: lineX)
        switch role {
        case .line: return .columnResize
        case .pan: return pressRole == .pan ? .grabActive : .grabIdle
        case .none: return .default
        }
    }

    private var help: LocalizedStringKey {
        switch (curtain.alternates, zoomed) {
        case (true, false): "Click the image, or press ← →, to show one image or the other"
        case (false, false): "Drag the line, or press ← →, to reveal one image or the other"
        case (true, true): "Click the image, or press ← →, to show one image or the other; drag it to move it"
        case (false, true): "Drag the line, or press ← →, to reveal one image or the other; drag the image to move it"
        }
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

    /// **One gesture for every press** (`CurtainGesture`): where it begins decides whether it carries the line or the
    /// images; its release, whether it was the switch's click.
    private var pressGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let role: CurtainGesture.Role
                if let r = pressRole { role = r } else {
                    MainWindow.shared.releaseTextFocus()
                    role = CurtainGesture.role(alternates: curtain.alternates, zoomed: zoomed, startX: value.startLocation.x, lineX: lineX)
                    pressRole = role
                }
                switch role {
                case .line:
                    dragging = true
                    guard frame.width > 0 else { return }
                    curtain.position = min(max(value.location.x / frame.width, 0), 1)
                case .pan:
                    panning = CurtainGesture.panned(value.translation, role: role)
                case .none:
                    break
                }
            }
            .onEnded { value in
                switch CurtainGesture.release(alternates: curtain.alternates, role: pressRole ?? .none, translation: value.translation) {
                case .swap: curtain.swap()
                case .keepPan: offset = clamped(CGSize(width: offset.width + value.translation.width, height: offset.height + value.translation.height))
                case .nothing: break
                }
                pressRole = nil
                panning = .zero
                dragging = false
            }
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .updating($pinch) { value, state, _ in state = value.magnification }
            .onEnded { value in
                let z = min(max(app.zoom * value.magnification, 1), 8)
                let target = ZoomAnchor.offset(offset, from: app.zoom, to: z, at: pointer, in: frame)
                if z == app.zoom { offset = z == 1 ? .zero : clamped(target) } else { settled = target; app.zoom = z }
            }
    }

    /// A mouse's wheel zooms toward the pointer (`Wheel`); two fingers (or a horizontal wheel) move both images, zoomed in;
    /// fitted, the scroll goes on as usual.
    private func watchScroll() {
        guard scrollMonitor == nil else { return }
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            guard hovering else { return event }
            guard let move = Wheel.zoomedOrMoved(event, app: app) else { return nil }
            guard app.zoom > 1 else { return event }
            offset = clamped(CGSize(width: offset.width + move.width, height: offset.height + move.height))
            return nil
        }
    }

    /// Zoomed in, the images always cover the frame: no empty band at an edge.
    private func clamped(_ o: CGSize) -> CGSize {
        let x = frame.width * (app.zoom - 1) / 2, y = frame.height * (app.zoom - 1) / 2
        return CGSize(width: min(max(o.width, -x), x), height: min(max(o.height, -y), y))
    }
}

/// **Side by side, the curtain, or the switch**: three icons in the bar under the images, the active one in the
/// accent. Two images of different sizes: grayed, and its help says why.
struct CurtainSwitch: View {
    let app: AppState
    private var curtain: Curtain { .shared }

    var body: some View {
        let possible = Curtain.pair(app) != nil
        HStack(spacing: 2) {
            segment("square.split.2x1", name: "Side by Side", active: curtain.mode == .sideBySide, possible: possible) {
                curtain.mode = .sideBySide
            }
            .help(possible ? "Side by side" : "The two images are not the same size: they can only be shown side by side")
            segment("square.lefthalf.filled", name: "Curtain", active: curtain.mode == .curtain, possible: possible) {
                curtain.mode = .curtain
            }
            .help(possible ? "Curtain: drag the line across the two images (← →)" : "The two images are not the same size: they can only be shown side by side")
            segment("rectangle.2.swap", name: "Switch", active: curtain.mode == .alternate, possible: possible) {
                curtain.mode = .alternate
            }
            .help(possible ? "Switch: one image at a time, at the same place and zoom — click it, or ← for A and → for B"
                           : "The two images are not the same size: they can only be shown side by side")
        }
        .padding(2)
        .background(Palette.well, in: RoundedRectangle(cornerRadius: 7))
    }

    /// Not `.disabled`: a disabled button shows no help, and the help is the reason.
    private func segment(_ symbol: String, name: LocalizedStringKey, active: Bool, possible: Bool, action: @escaping () -> Void) -> some View {
        Button {
            guard possible else { return }
            withAnimation(Motion.pick(.easeInOut(duration: 0.2))) { action() }
        } label: {
            Image(systemName: symbol)
                .imageScale(.medium)
                .frame(width: 30, height: 22)
                .foregroundStyle(!possible ? AnyShapeStyle(.tertiary) : active ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.secondary))
                .background(RoundedRectangle(cornerRadius: 5).fill(active && possible ? Palette.surface : .clear))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text(name))
        .accessibilityAddTraits(active && possible ? .isSelected : [])
        .accessibilityHint(possible ? Text(verbatim: "") : Text("The two images are not the same size: they can only be shown side by side"))
    }
}

/// **A mouse's wheel zooms; a trackpad's two fingers move** (issue #10). A trackpad and a Magic Mouse send their scrolls in
/// phases (began, changed, ended, then the momentum's); a wheel never does — the phases tell the devices apart where the deltas'
/// precision would not (a mouse driver's smooth scrolling sends precise deltas too). A mouse cannot pinch: its vertical wheel is
/// its only zoom. A horizontal wheel — a thumb wheel, or ⇧ with the wheel — still moves the image.
enum Wheel {
    static func isWheel(_ event: NSEvent) -> Bool { event.phase.isEmpty && event.momentumPhase.isEmpty }

    /// The zoom after a wheel's vertical event, `nil` for anything else: turned away from the hand, in; toward it, out — whatever
    /// the system's natural scrolling (`isDirectionInvertedFromDevice` undoes it). A notch is 15 %; a smooth-scrolling driver's
    /// precise deltas count about ten points to a notch.
    static func zoom(_ z: CGFloat, by event: NSEvent) -> CGFloat? {
        guard isWheel(event), abs(event.scrollingDeltaY) > abs(event.scrollingDeltaX) else { return nil }
        let away = event.scrollingDeltaY * (event.isDirectionInvertedFromDevice ? -1 : 1)
        let notches = event.hasPreciseScrollingDeltas ? away / 10 : away
        return min(max(z * pow(1.15, notches), 1), 8)
    }

    /// **A wheel event as the canvas and the curtain read it, the same way**: a wheel zooms (`app.zoom`), anything else
    /// is a move, handed back as the offset it adds (two fingers; a horizontal wheel). `nil`: the wheel zoomed, the event is taken.
    @MainActor static func zoomedOrMoved(_ event: NSEvent, app: AppState) -> CGSize? {
        if let z = zoom(app.zoom, by: event) { if z != app.zoom { app.zoom = z }; return nil }
        let k: CGFloat = event.hasPreciseScrollingDeltas ? 1 : 10
        return CGSize(width: event.scrollingDeltaX * k, height: event.scrollingDeltaY * k)
    }
}

/// **Zooming toward the pointer**, as Preview does: the point under it stays under it. An image drawn
/// centered in a frame of `size`, scaled by `z` about that center then moved by `o`, shows its point
/// `u` (from its center) at `center + u·z + o`; keeping that fixed from `z1` to `z2` gives the new
/// offset below. Without a pointer (⌘+ from the menu, the pointer elsewhere), the frame's center holds.
enum ZoomAnchor {
    static func offset(_ o: CGSize, from z1: CGFloat, to z2: CGFloat, at pointer: CGPoint?, in size: CGSize) -> CGSize {
        guard z1 > 0 else { return o }
        let k = z2 / z1
        let q = pointer.map { CGSize(width: $0.x - size.width / 2, height: $0.y - size.height / 2) } ?? .zero
        return CGSize(width: q.width - (q.width - o.width) * k, height: q.height - (q.height - o.height) * k)
    }
}
