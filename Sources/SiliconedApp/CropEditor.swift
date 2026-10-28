import AppKit
import Siliconed
import SwiftUI

// **« Crop »: one tool, the same gesture on a history image and on an edit's reference.** C opens it
// on the image shown (a reference's menu opens it on that reference), the way Photos and Pixelmator
// do: handles on the corners and the sides, the rule of thirds while a side moves, the proportions
// the rack's size picker offers (plus the image's own, plus free), Return to accept, Esc to leave.
//
//     CropSession ── the frame (`Crop.Frame`: the arithmetic, in `Siliconed/Crop.swift`) ── CropEditor: the gesture
//       │
//       ├─ a history image → a NEW image at the head of the history; the original stays (⌘Z removes the crop)
//       └─ a reference     → replaces that reference, at its number (⌘Z brings the previous one back)
//
// What the frame decides — never under `Format.minimumSide` on a side, the proportions kept to the
// pixel — is in `Crop` and tested there; this file only turns a pointer into a request for it.

/// The crop in progress: what is cut, from which image, and where the frame stands.
struct CropSession {
    enum Target { case entry(Entry.ID), reference(ReferenceImage.ID) }
    let target: Target
    let image: CGImage
    let frame: Crop.Frame
    var box: Crop.Box
    var shape = Crop.Shape.free

    init(target: Target, image: CGImage) {
        self.target = target
        self.image = image
        frame = Crop.Frame(width: image.width, height: image.height)
        box = frame.whole
    }

    var ratio: Crop.Ratio? { frame.ratio(of: shape) }
    /// Whole image: nothing to cut.
    var isWhole: Bool { box == frame.whole }
}

extension AppState {
    // MARK: Asking

    /// Why « Crop » cannot take this image, in a sentence — `nil` when it can.
    func cropRefusal(_ entry: Entry) -> String? {
        if entry.grid != nil { return String(localized: "A grid is not cropped") }
        if entry.sketch != nil { return String(localized: "A sketch is not cropped") }
        return cropRefusal(width: entry.width, height: entry.height)
    }

    func cropRefusal(width: Int, height: Int) -> String? {
        let frame = Crop.Frame(width: width, height: height)
        guard !frame.canCrop else { return nil }
        let side = String(Format.minimumSide)
        return min(width, height) < Format.minimumSide
            ? String(localized: "Cropping needs \(side) px or more on each side")
            : String(localized: "This image is already at the smallest size, \(side) px")
    }

    /// **C**: the image shown, or the toast that says why not.
    func beginCropOfShownImage() {
        guard cropping == nil, let entry = selectedEntry, !(inProgress && followsRender) else { return }
        if let refusal = cropRefusal(entry) { flash(refusal); return }
        beginCrop(entry)
    }

    func beginCrop(_ entry: Entry) {
        guard cropping == nil, cropRefusal(entry) == nil else { return }
        show(entry.id)
        Task {
            let image: CGImage?
            if let shown = selectedImage, shown.id == entry.id { image = shown.image } else { image = await Entry.decoded(entry.png).image }
            guard let image, selection == entry.id, cropping == nil else { return }
            startCrop(CropSession(target: .entry(entry.id), image: image))
        }
    }

    func beginCrop(_ reference: ReferenceImage) {
        guard cropping == nil else { return }
        if let refusal = cropRefusal(width: reference.width, height: reference.height) { flash(refusal); return }
        let id = reference.id, w = reference.width, h = reference.height
        Task {
            let image = await Task.detached(priority: .userInitiated) { Decoded(reference.displayed(width: w, height: h)) }.value.image
            guard let image, cropping == nil, references.contains(where: { $0.id == id }) else { return }
            startCrop(CropSession(target: .reference(id), image: image))
        }
    }

    private func startCrop(_ session: CropSession) {
        MainWindow.shared.releaseTextFocus()
        showsExploration = false
        followsRender = false
        comparesOriginal = false
        zoom = 1
        cropping = session
    }

    // MARK: Shaping

    func setCropShape(_ shape: Crop.Shape) {
        guard var s = cropping, s.frame.supports(shape) else { return }
        s.shape = shape
        if let r = s.frame.ratio(of: shape), let fitted = s.frame.fit(s.box, to: r) { s.box = fitted }
        cropping = s
    }

    /// Portrait ↔ landscape: the proportion turned a quarter, the frame refitted around its center.
    func swapCropOrientation() {
        guard var s = cropping, let r = s.ratio, r.num != r.den else { return }
        let turned = r.swapped
        guard let fitted = s.frame.fit(s.box, to: turned) else { return }
        s.shape = .fixed(turned)
        s.box = fitted
        cropping = s
    }

    func cancelCrop() { cropping = nil }

    // MARK: Accepting

    /// **Return**: the frame's pixels become an image — a new one at the head of the history, or the
    /// reference itself, cropped. The whole image is not a crop: nothing happens.
    func commitCrop() {
        guard let s = cropping else { return }
        cropping = nil
        guard !s.isWhole else { return }
        switch s.target {
        case .entry(let id):
            guard let entry = history.first(where: { $0.id == id }) else { return }
            // In a mirror, the other Mac cuts its own pixels: the same box, the same image, in its history.
            if let mirror { mirror.send(.crop(entry.ordinal, s.box)); return }
            Task { await cropped(entry, image: s.image, box: s.box) }
        case .reference(let id):
            cropReference(id, image: s.image, box: s.box)
        }
    }

    /// The frame's pixels as an image of their own: copied out, so the parent's memory is not kept alive.
    private nonisolated static func cut(_ image: CGImage, _ box: Crop.Box) -> CGImage? {
        guard let part = image.cropping(to: CGRect(x: box.x, y: box.y, width: box.width, height: box.height)) else { return nil }
        let alpha: CGImageAlphaInfo = switch image.alphaInfo {
        case .none, .noneSkipLast, .noneSkipFirst: .noneSkipLast
        default: .premultipliedLast
        }
        guard let c = CGContext(data: nil, width: box.width, height: box.height, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: alpha.rawValue) else { return nil }
        c.draw(part, in: CGRect(x: 0, y: 0, width: box.width, height: box.height))
        return c.makeImage()
    }

    /// A crop asked by another Mac (`MirrorServer`): the box in this image's own pixels.
    func crop(_ entry: Entry, box: Crop.Box) async {
        guard let image = await Entry.decoded(entry.png).image,
              box.x >= 0, box.y >= 0, box.maxX <= image.width, box.maxY <= image.height, box.width > 0, box.height > 0 else { return }
        await cropped(entry, image: image, box: box)
    }

    private func cropped(_ entry: Entry, image: CGImage, box: Crop.Box) async {
        var metadata = entry.metadata
        metadata["cropped"] = "\(box.width)x\(box.height) at \(box.x),\(box.y), from \(image.width)x\(image.height)"
        let made = await Task.detached(priority: .userInitiated) { [metadata] () -> (Data, Decoded, Decoded)? in
            guard let part = Self.cut(image, box), let png = try? PNG.data(part, metadata: metadata) else { return nil }
            return (png, Decoded(part), Decoded(FinishedImage.reduce(part, side: 240) ?? part))
        }.value
        guard let (png, full, small) = made, let image = full.image, let vignette = small.image else {
            flash(String(localized: "This image could not be cropped"))
            return
        }
        lastImage += 1
        let result = Entry(ordinal: lastImage, job: entry.job, png: png, vignette: vignette,
                           width: image.width, height: image.height, seed: entry.seed, timings: Engine.Timings(),
                           footprints: Engine.Footprints(), evaluations: 0, settings: entry.settings,
                           metadata: metadata, enlarged: entry.enlarged, cropped: box, owner: entry.owner)
        // Kept: the crop is still an enlargement's pixels, and `enlargeRefusal` refuses a second pass.
        history.insert(result, at: 0)
        selectedImage = (result.id, image)
        selection = result.id
        marked = []; anchor = result.id
        undoManager?.registerUndo(withTarget: self) { $0.delete(result) }
        undoManager?.setActionName(String(localized: "Crop"))
    }

    private func cropReference(_ id: ReferenceImage.ID, image: CGImage, box: Crop.Box) {
        guard let i = references.firstIndex(where: { $0.id == id }), let part = Self.cut(image, box) else { return }
        let old = references[i]
        let vignette = FinishedImage.reduce(part, side: 240) ?? part
        let new = ReferenceImage(source: .image(part), vignette: NSImage(cgImage: vignette, size: .zero), name: old.name,
                                 width: part.width, height: part.height)
        references[i] = new
        undoManager?.registerUndo(withTarget: self) { app in
            guard let j = app.references.firstIndex(where: { $0.id == new.id }) else { return }
            app.references[j] = old
        }
        undoManager?.setActionName(String(localized: "Crop"))
        flash(String(localized: "Image \(i + 1) cropped to \(part.width) × \(part.height)"))
    }
}

// MARK: - The view

/// **The frame over the image**, in the canvas's place: the outside dimmed, the thirds while a side
/// moves, handles on the corners and the middle of the sides, and under it the proportions, the size,
/// « Cancel » and « Crop ».
struct CropEditor: View {
    @Bindable var app: AppState
    let session: CropSession

    /// Where a drag began: the frame then, and the sides it moves (none: the whole frame).
    private struct Grab { var box: Crop.Box; var edges: Crop.Edges }
    @State private var grab: Grab?
    @State private var hover: Crop.Edges?
    private static let reach: CGFloat = 12

    var body: some View {
        VStack(spacing: 14) {
            GeometryReader { g in
                let r = fitted(in: g.size)
                ZStack(alignment: .topLeading) {
                    Image(decorative: session.image, scale: 1)
                        .resizable().interpolation(.high)
                        .frame(width: r.width, height: r.height)
                        .offset(x: r.minX, y: r.minY)
                    Canvas { c, _ in draw(c, in: r) }
                        .allowsHitTesting(false)
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0)
                    .onChanged { drag($0, in: r) }
                    .onEnded { _ in grab = nil })
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let p): hover = edges(at: p, in: r)
                    case .ended: hover = nil
                    }
                }
                .pointerStyle(pointer)
                .accessibilityElement()
                .accessibilityLabel(Text("Crop frame"))
                .accessibilityValue(Text(verbatim: "\(session.box.width) × \(session.box.height)"))
            }
            bar
        }
        .padding(28)
        .transition(.opacity)
    }

    // ── geometry ──

    private func fitted(in size: CGSize) -> CGRect {
        let s = min(size.width / CGFloat(session.image.width), size.height / CGFloat(session.image.height), 1)
        let w = CGFloat(session.image.width) * s, h = CGFloat(session.image.height) * s
        return CGRect(x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h)
    }

    /// Image pixels per point in `r`.
    private func scale(_ r: CGRect) -> CGFloat { CGFloat(session.image.width) / max(r.width, 1) }

    private func rect(_ b: Crop.Box, in r: CGRect) -> CGRect {
        let k = 1 / scale(r)
        return CGRect(x: r.minX + CGFloat(b.x) * k, y: r.minY + CGFloat(b.y) * k,
                      width: CGFloat(b.width) * k, height: CGFloat(b.height) * k)
    }

    /// The sides under the pointer: a corner, a side, all of them (inside: the frame moves) or `nil`.
    private func edges(at p: CGPoint, in r: CGRect) -> Crop.Edges? {
        let f = rect(session.box, in: r), k = Self.reach
        let inY = p.y > f.minY - k && p.y < f.maxY + k, inX = p.x > f.minX - k && p.x < f.maxX + k
        var e = Crop.Edges()
        if inY && abs(p.x - f.minX) < k { e.insert(.left) }
        if inY && abs(p.x - f.maxX) < k { e.insert(.right) }
        if inX && abs(p.y - f.minY) < k { e.insert(.top) }
        if inX && abs(p.y - f.maxY) < k { e.insert(.bottom) }
        if e.isEmpty { return f.contains(p) ? [] : nil }
        return e
    }

    private var pointer: PointerStyle? {
        if grab != nil, grab?.edges.isEmpty == true { return .grabActive }
        guard let e = grab?.edges ?? hover else { return nil }
        if e.isEmpty { return .grabIdle }
        let h: FrameResizePosition? = e.contains(.left) ? (e.contains(.top) ? .topLeading : e.contains(.bottom) ? .bottomLeading : .leading)
            : e.contains(.right) ? (e.contains(.top) ? .topTrailing : e.contains(.bottom) ? .bottomTrailing : .trailing)
            : e.contains(.top) ? .top : e.contains(.bottom) ? .bottom : nil
        return h.map { .frameResize(position: $0) }
    }

    // ── the gesture ──

    private func drag(_ v: DragGesture.Value, in r: CGRect) {
        guard r.width > 0, var s = app.cropping else { return }
        if grab == nil {
            MainWindow.shared.releaseTextFocus()
            guard let e = edges(at: v.startLocation, in: r) else { return }
            grab = Grab(box: s.box, edges: e)
        }
        guard let g = grab else { return }
        let k = scale(r)
        let dx = v.translation.width * k, dy = v.translation.height * k
        if g.edges.isEmpty {
            s.box = s.frame.move(g.box, dx: Int(dx.rounded()), dy: Int(dy.rounded()))
        } else {
            // Each dragged side goes where it was, plus the pointer's travel.
            let x = g.edges.contains(.left) ? CGFloat(g.box.x) : CGFloat(g.box.maxX)
            let y = g.edges.contains(.top) ? CGFloat(g.box.y) : CGFloat(g.box.maxY)
            s.box = s.frame.resize(g.box, edges: g.edges, to: CGPoint(x: x + dx, y: y + dy), ratio: s.ratio)
        }
        app.cropping = s
    }

    // ── the drawing ──

    private func draw(_ c: GraphicsContext, in r: CGRect) {
        let f = rect(session.box, in: r)
        var outside = Path(r)
        outside.addPath(Path(f))
        c.fill(outside, with: .color(.black.opacity(0.55)), style: FillStyle(eoFill: true))
        if grab != nil {
            var thirds = Path()
            for i in 1...2 {
                let t = CGFloat(i) / 3
                thirds.move(to: CGPoint(x: f.minX + f.width * t, y: f.minY)); thirds.addLine(to: CGPoint(x: f.minX + f.width * t, y: f.maxY))
                thirds.move(to: CGPoint(x: f.minX, y: f.minY + f.height * t)); thirds.addLine(to: CGPoint(x: f.maxX, y: f.minY + f.height * t))
            }
            c.stroke(thirds, with: .color(.white.opacity(0.55)), lineWidth: 0.75)
        }
        c.stroke(Path(f), with: .color(.white.opacity(0.9)), lineWidth: 1)
        // The grips: an L at each corner, a short bar in the middle of each side.
        let arm = min(18, f.width / 3, f.height / 3), lit = Color.accentColor
        func grip(_ p: Path, _ on: Bool) {
            c.stroke(p, with: .color(.black.opacity(0.35)), style: StrokeStyle(lineWidth: 5, lineCap: .round, lineJoin: .round))
            c.stroke(p, with: .color(on ? lit : .white), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
        }
        let active = grab?.edges ?? hover ?? Crop.Edges()
        func corner(_ x: CGFloat, _ y: CGFloat, _ sx: CGFloat, _ sy: CGFloat, _ edges: Crop.Edges) {
            var p = Path()
            p.move(to: CGPoint(x: x + sx * arm, y: y)); p.addLine(to: CGPoint(x: x, y: y)); p.addLine(to: CGPoint(x: x, y: y + sy * arm))
            grip(p, active == edges)
        }
        corner(f.minX, f.minY, 1, 1, [.left, .top]); corner(f.maxX, f.minY, -1, 1, [.right, .top])
        corner(f.minX, f.maxY, 1, -1, [.left, .bottom]); corner(f.maxX, f.maxY, -1, -1, [.right, .bottom])
        func side(_ a: CGPoint, _ b: CGPoint, _ edge: Crop.Edges) {
            var p = Path(); p.move(to: a); p.addLine(to: b)
            grip(p, active == edge)
        }
        let half = min(10, f.width / 6), halfH = min(10, f.height / 6)
        side(CGPoint(x: f.midX - half, y: f.minY), CGPoint(x: f.midX + half, y: f.minY), .top)
        side(CGPoint(x: f.midX - half, y: f.maxY), CGPoint(x: f.midX + half, y: f.maxY), .bottom)
        side(CGPoint(x: f.minX, y: f.midY - halfH), CGPoint(x: f.minX, y: f.midY + halfH), .left)
        side(CGPoint(x: f.maxX, y: f.midY - halfH), CGPoint(x: f.maxX, y: f.midY + halfH), .right)
    }

    // ── the bar ──

    private var bar: some View {
        HStack(spacing: 12) {
            Menu {
                shapeItem(.free, "Free")
                shapeItem(.original, "Original")
                Divider()
                ForEach(Crop.Ratio.offered, id: \.self) { shapeItem(.fixed($0), Text(verbatim: $0.label)) }
            } label: {
                Label(shapeName, systemImage: "aspectratio")
            }
            .menuStyle(.borderlessButton).fixedSize()
            .help("The frame's proportions — the ones the size picker offers, the image's own, or free")
            Button { app.swapCropOrientation() } label: { Image(systemName: "rectangle.portrait.rotate") }
                .disabled(!canTurn)
                .help("Turn the frame: portrait or landscape")
            Divider().frame(height: 22)
            Text(verbatim: "\(session.box.width) × \(session.box.height)")
                .font(.callout).monospacedDigit()
                .frame(minWidth: 96)
                .help(Text("Never under \(String(Format.minimumSide)) px on a side"))
            Divider().frame(height: 22)
            Button("Cancel") { app.cancelCrop() }
                .help("Leave without cropping (Esc)")
            Button("Crop") { app.commitCrop() }
                .buttonStyle(.borderedProminent)
                .disabled(session.isWhole)
                .help(session.isWhole ? Text("Move a side of the frame first") : Text("Crop the image (↩)"))
        }
        .imageScale(.large)
        .padding(.horizontal, 16).padding(.vertical, 8)
        .glass(Capsule())
        .zIndex(2)
    }

    private var shapeName: String {
        switch session.shape {
        case .free: String(localized: "Free")
        case .original: String(localized: "Original")
        case .fixed(let r): r.label
        }
    }

    private var canTurn: Bool {
        guard let r = session.ratio, r.num != r.den else { return false }
        return session.frame.supports(.fixed(r.swapped))
    }

    private func shapeItem(_ shape: Crop.Shape, _ title: Text) -> some View {
        Toggle(isOn: Binding(get: { session.shape == shape }, set: { _ in app.setCropShape(shape) })) { title }
            .disabled(!session.frame.supports(shape))
    }

    private func shapeItem(_ shape: Crop.Shape, _ title: LocalizedStringKey) -> some View {
        shapeItem(shape, Text(title))
    }
}
