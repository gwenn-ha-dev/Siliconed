// **The window's column, folded or unfolded**: one column from the idea to the button (`Sidebar`), and one fold for it —
// the toolbar's button beside the lights, ⌥⌘R, `silicontrol rack fold|unfold`. Folded, it is gone, as a sidebar: the image
// takes the window.
//
// - **Unfolded**: as wide as the hand left it (its right edge drags, `widths`).
// - **What only the column says is never said to a folded one**: a problem unfolds it (its sentence is at the column's
//   foot), so does ⌘↩ when something stops the render, and so does a scroll asked of it (`silicontrol rack`).
// - **Folding stops a gesture of the writer under way** in the Text box: its marks would be written where nobody sees them.
// - **Detached, the column is a floating window of its own** (`RackPanel`): the text and the chain beside the image — on
//   another screen, or over it —, the image taking the whole window. Closing that window docks the column back. Kept
//   across launches, as the fold is; while it is detached, « unfold » brings its window forward.

import AppKit
import Siliconed
import SwiftUI

@MainActor @Observable
final class RackColumnModel {
    static let shared = RackColumnModel()
    static let foldedKey = "rackFolded", widthKey = "rackWidth", detachedKey = "rackDetached"
    /// The widths the hand may give the column: the Text box's field and the DiT's rows on the narrowest, a 13-inch
    /// screen's canvas kept on the widest.
    static let widths: ClosedRange<CGFloat> = 340...560
    static let ideal: CGFloat = 400

    private(set) var folded = UserDefaults.standard.bool(forKey: RackColumnModel.foldedKey)
    /// In its own window (`RackPanel`), not in the main one.
    private(set) var detached = UserDefaults.standard.bool(forKey: RackColumnModel.detachedKey)
    var width: CGFloat = {
        let kept = UserDefaults.standard.double(forKey: RackColumnModel.widthKey)
        return kept > 0 ? min(max(kept, RackColumnModel.widths.lowerBound), RackColumnModel.widths.upperBound) : RackColumnModel.ideal
    }() {
        didSet { UserDefaults.standard.set(Double(width), forKey: Self.widthKey) }
    }

    /// Gone from the main window: folded, or in its own.
    var outOfWindow: Bool { folded || detached }

    func fold() { set(true) }
    /// What must be seen in the column — a problem, a scroll asked — is: unfolded, or its window brought forward.
    func unfold() { if detached { MainWindow.shared.rackPanel.show() } else { set(false) } }
    func toggle() { if detached { MainWindow.shared.rackPanel.show() } else { set(!folded) } }

    /// The column into its own window, or back into the main one.
    func setDetached(_ value: Bool) {
        guard detached != value else { return }
        detached = value
        UserDefaults.standard.set(value, forKey: Self.detachedKey)
        if value {
            set(false)
            MainWindow.shared.rackPanel.show()
        } else {
            MainWindow.shared.rackPanel.close()
        }
    }

    private func set(_ value: Bool) {
        guard folded != value else { return }
        folded = value
        UserDefaults.standard.set(value, forKey: Self.foldedKey)
        if value, LabModel.shared.phase != .idle { LabModel.shared.stop() }
    }
}

struct RackColumn: View {
    @Bindable var app: AppState
    let model: RackColumnModel
    /// The components moved aside (Space): the column goes as folded, without being folded.
    let away: Bool
    @State private var dragStart: CGFloat?

    var body: some View {
        let gone = model.outOfWindow || away
        ZStack(alignment: .trailing) {
            if !gone {
                // The column keeps its width and slides out under the window's left edge, clipped, as a drawer.
                Sidebar(app: app).frame(width: model.width)
                    .transition(.offset(x: -model.width).combined(with: .asymmetric(
                        insertion: .opacity.animation(.easeIn(duration: 0.12).delay(0.04)),
                        removal: .opacity.animation(.easeOut(duration: 0.1)))))
            }
        }
        .modifier(SlidingWidth(width: gone ? 0 : model.width))
        .frame(maxHeight: .infinity)
        .clipped()
        .columnGlass(app.glass)
        .overlay(alignment: .trailing) { if !gone { resizer } }
        .onChange(of: app.problem != nil) { _, said in if said { model.unfold() } }
        // A column unfolded for a scroll was not there to hear it: the scroll is said again once the rack is drawn.
        .onChange(of: app.rackScroll?.count) {
            guard model.folded, let asked = app.rackScroll else { return }
            model.unfold()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { app.rackScroll = (asked.place, asked.count + 1) }
        }
    }

    /// The column's right edge, dragged. Its cursor by `pointerStyle`: a push on hover lost its pop when the column folded
    /// or detached under the pointer (⌥⌘R, Space), and the resize cursor stayed.
    private var resizer: some View {
        Color.clear.frame(width: 7).contentShape(Rectangle())
            .pointerStyle(.columnResize)
            .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
                .onChanged { drag in
                    let start = dragStart ?? model.width
                    dragStart = start
                    model.width = min(max(start + drag.translation.width, RackColumnModel.widths.lowerBound), RackColumnModel.widths.upperBound)
                }
                .onEnded { _ in dragStart = nil })
            .accessibilityHidden(true)
    }
}

/// **The column in a floating window of its own**: the same `Sidebar`, the state the main window shows (this Mac's, or a
/// mirror's). Closing it docks the column back (`RackColumnModel.setDetached`); the main window closing hides it.
@MainActor
final class RackPanel: NSObject, NSWindowDelegate {
    private let carrier: Carrier
    private let local: AppState
    private(set) var window: NSPanel?

    init(carrier: Carrier, local: AppState) { (self.carrier, self.local) = (carrier, local) }

    func show() {
        let p = window ?? make()
        window = p
        p.makeKeyAndOrderFront(nil)
    }

    /// Docked back: the window goes, and is made again at the next detach.
    func close() {
        window?.delegate = nil
        window?.close()
        window = nil
    }

    /// The main window closes: the column's window goes with it, still detached for the next one.
    func hide() { window?.orderOut(nil) }

    func windowWillClose(_ notification: Notification) {
        window = nil
        RackColumnModel.shared.setDetached(false)
    }

    /// **The smallest the panel gets**: the column's narrowest, and the rack's head with a few of its rows. Held here
    /// (`MinimumContent`): the hosting controller sets `contentMinSize` back to zero, and the panel could be dragged up to
    /// its buttons (issue 17; the main window's issue #7).
    static let minimumContent = NSSize(width: RackColumnModel.widths.lowerBound, height: 480)

    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
        MinimumContent.clamp(frameSize, of: sender, to: Self.minimumContent)
    }

    private func make() -> NSPanel {
        let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: RackColumnModel.ideal, height: 760),
                        styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
                        backing: .buffered, defer: false)
        p.title = String(localized: "Chain")
        p.titlebarAppearsTransparent = true
        p.isFloatingPanel = true
        p.becomesKeyOnlyIfNeeded = false
        p.hidesOnDeactivate = false
        p.isRestorable = false
        p.isReleasedWhenClosed = false
        p.delegate = self
        let host = NSHostingController(rootView: RackPanelRoot(carrier: carrier, local: local))
        host.sizingOptions = []
        p.contentViewController = host
        p.setContentSize(NSSize(width: RackColumnModel.shared.width, height: 760))
        // Where the column was: along the main window's left edge, outside it when the screen has the room.
        p.setFrameAutosaveName("RackPanel")
        if !p.setFrameUsingName("RackPanel"), let f = MainWindow.shared.window?.frame {
            p.setFrameTopLeftPoint(NSPoint(x: max(0, f.minX - p.frame.width - 8), y: f.maxY))
        }
        return p
    }
}

private struct RackPanelRoot: View {
    let carrier: Carrier
    let local: AppState

    var body: some View {
        let app = carrier.mirror ?? local
        Sidebar(app: app)
            .columnGlass(app.glass)
            .id(ObjectIdentifier(app))
    }
}
