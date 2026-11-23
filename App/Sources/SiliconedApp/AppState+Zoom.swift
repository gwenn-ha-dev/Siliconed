// **`AppState`: View › Zoom In, Zoom Out, Zoom to Fit** (⌘+ ⌘− ⌘0) — the menu's items and `silicontrol zoom in|out|fit`,
// one gesture each. They act on what the canvas shows: a series' grid, its cells' size; otherwise the chosen image's zoom.

import CoreGraphics
import Siliconed

extension AppState {
    var canZoomIn: Bool { shownSeries != nil || selectedImage != nil }
    var canZoomOut: Bool { shownSeries != nil || zoom != 1 }
    /// A series whose cells already fill the canvas, or an image already fitted: ⌘0 has nothing to give back.
    var canZoomToFit: Bool { shownSeries != nil ? !seriesFillsWindow : zoom != 1 }

    /// One step of the gesture, where the menu item would be enabled; otherwise nothing, as the greyed item does nothing.
    func zoom(_ step: GestureOrder.ZoomStep) {
        switch step {
        case .zoomIn where canZoomIn:
            if shownSeries != nil { exploreCellSide = Self.clampedCellSide(exploreCellSide * 1.25) }
            else { zoom = min(zoom * 1.5, zoomLimit) }
        case .zoomOut where canZoomOut:
            if shownSeries != nil { exploreCellSide = Self.clampedCellSide(exploreCellSide / 1.25) }
            else { zoom = max(zoom / 1.5, 1) }
        case .fit where canZoomToFit:
            if shownSeries != nil { fitSeries() } else { zoom = 1 }
        default: break
        }
    }
}
