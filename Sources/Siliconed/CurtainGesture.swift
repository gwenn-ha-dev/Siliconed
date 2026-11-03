import CoreGraphics

// **What a press on the curtain does** (issue #10) — the rule, without a window (`CurtainView` in the app).
//
//                     fitted (zoom 1)                      zoomed in
//     curtain    press anywhere: the line comes      press on the line (its knob's width): the line follows;
//                there and follows                   anywhere else: the images follow the pointer
//     switch     click: the other image              click: the other image; drag: the images follow
//
// - **Zoomed in, the image is what one wants to move**: the line only moves by its handle (and ← →), so that
//   looking around a detail never drags the line away from it. Fitted there is nothing to move: a press brings
//   the line, as it always did.
// - **A click is a press that hardly moved** (`clickSlop`): in the switch, a hand that trembles by a point
//   still switches, and a drag that has gone past the slop never switches at its end. The images start
//   following only past the slop too — a click does not nudge them.
// - The handle is hit within half the knob's width of the line, over the whole height: the line runs from top
//   to bottom of what is seen, and the knob is only where the eye finds it.
package enum CurtainGesture {
    /// What the pointer carries for the rest of the press.
    package enum Role: Equatable, Sendable {
        /// The curtain's line follows the pointer.
        case line
        /// The two images follow the pointer (zoomed in).
        case pan
        /// Nothing follows; only the release counts (the switch, fitted).
        case none
    }

    /// What the release does, after the press was given its `Role`.
    package enum Release: Equatable, Sendable {
        /// The switch shows the other image.
        case swap
        /// The images keep the offset they were dragged to.
        case keepPan
        case nothing
    }

    /// How far a press may move and still be a click, in points.
    package static let clickSlop: CGFloat = 4
    /// Half the knob's width: how near the line a press takes the line rather than the images.
    package static let handleReach: CGFloat = 15

    /// The role of a press starting at `x` (in the frame's points), the line drawn at `lineX`.
    package static func role(alternates: Bool, zoomed: Bool, startX x: CGFloat, lineX: CGFloat) -> Role {
        if alternates { return zoomed ? .pan : .none }
        guard zoomed else { return .line }
        return abs(x - lineX) <= handleReach ? .line : .pan
    }

    /// Whether a press moved by `translation` is still a click.
    package static func isClick(_ translation: CGSize) -> Bool {
        (translation.width * translation.width + translation.height * translation.height).squareRoot() <= clickSlop
    }

    /// The offset the images are drawn at while they follow: none until the press has left the slop, so that a
    /// click leaves them where they are.
    package static func panned(_ translation: CGSize, role: Role) -> CGSize {
        role == .pan && !isClick(translation) ? translation : .zero
    }

    /// What the release of a press of `role`, moved by `translation`, does.
    package static func release(alternates: Bool, role: Role, translation: CGSize) -> Release {
        if isClick(translation) { return alternates ? .swap : .nothing }
        return role == .pan ? .keepPan : .nothing
    }
}
