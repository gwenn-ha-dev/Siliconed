import Foundation

// **Cropping: the arithmetic of the frame, apart from the gesture.** The app draws a rectangle on an
// image (a render, or an edit's reference) and the user moves its sides; everything that decides
// *which* rectangles exist is here, in pixels of the image, so that a test can drag it a thousand
// ways without a window.
//
// What a port by analogy would miss:
//
// - **Nothing under `Format.minimumSide` on a side** — the same floor as a render's: the frame
//   refuses to shrink past it instead of letting it happen and complaining. A cropped image goes back
//   into the models (edit it, enlarge it), and 512 is where they stop being in their domain. An image
//   already under the floor on a side cannot be cropped at all (`Frame.canCrop`).
// - **No multiple of 16 or 32 is asked of the frame**: an edit's reference is rescaled by the
//   pipeline (`QwenImage21Pipeline`'s format rule floors the sides itself), and a cropped history
//   image is a picture, not a render's size.
// - **A fixed proportion is exact in integers**: 7:9 is a frame of 7k × 9k pixels, so the ratio holds
//   to the pixel however the user drags — except for the image's own proportion, when it does not
//   reduce to small numbers (1023×769 is coprime): there the height is the width's rounded product.
// - **A proportion the image cannot hold at the floor is not offered**: 4:7 on a 1500×600 image
//   would need 896 px of height.
// - **Resizing by a side keeps the opposite side and the middle of the other axis**; by a corner, the
//   opposite corner. A proportion that no longer fits where it is moves its frame (`fit`).

package enum Crop {
    /// A rectangle in pixels of the image, origin at the top left.
    package struct Box: Equatable, Hashable, Sendable, Codable {
        package var x: Int, y: Int, width: Int, height: Int
        package init(x: Int, y: Int, width: Int, height: Int) {
            self.x = x; self.y = y; self.width = width; self.height = height
        }
        package var maxX: Int { x + width }
        package var maxY: Int { y + height }
    }

    /// **A proportion**, width : height. `step` is the grain of the widths that hold it exactly.
    package struct Ratio: Equatable, Hashable, Sendable {
        package let num: Int, den: Int
        package let step: Int

        /// The proportions with a numerator or a denominator beyond this are not reduced: their
        /// exact multiples would be too coarse to drag.
        private static let reducedLimit = 64

        /// `w : h` reduced when the reduction is small enough to be a grain, taken as it is otherwise.
        package init(_ w: Int, _ h: Int) {
            let w = max(w, 1), h = max(h, 1)
            let g = Self.gcd(w, h)
            if w / g <= Self.reducedLimit, h / g <= Self.reducedLimit {
                num = w / g; den = h / g; step = w / g
            } else {
                num = w; den = h; step = 1
            }
        }

        private static func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }

        package var swapped: Ratio { Ratio(den, num) }
        package var label: String { "\(num):\(den)" }

        /// The height of a frame `w` wide: exact on the grain, rounded between.
        package func height(forWidth w: Int) -> Int { (w * den + num / 2) / num }

        /// **The proportions the app already offers for a render's format** (`ModelCard.recommendedFormats`),
        /// reduced, in the picker's order: 1:1, then the portraits and the landscapes.
        package static let offered: [Ratio] = {
            var seen: [Ratio] = []
            for f in ModelCard.recommendedFormats {
                let r = Ratio(f.width, f.height)
                if !seen.contains(r) { seen.append(r) }
            }
            return seen
        }()
    }

    /// What the user holds: free, the image's own proportion, or one of the offered.
    package enum Shape: Equatable, Hashable, Sendable {
        case free, original, fixed(Ratio)
    }

    /// The sides a drag moves; two of them (a horizontal and a vertical) make a corner.
    package struct Edges: OptionSet, Sendable {
        package let rawValue: Int
        package init(rawValue: Int) { self.rawValue = rawValue }
        package static let left = Edges(rawValue: 1), right = Edges(rawValue: 2)
        package static let top = Edges(rawValue: 4), bottom = Edges(rawValue: 8)
        package var horizontal: Bool { !isDisjoint(with: [.left, .right]) }
        package var vertical: Bool { !isDisjoint(with: [.top, .bottom]) }
    }

    /// **An image and its floor: the frames it can take.**
    package struct Frame: Equatable, Sendable {
        package let width: Int, height: Int
        package let minimum: Int

        package init(width: Int, height: Int, minimum: Int = Format.minimumSide) {
            self.width = width; self.height = height; self.minimum = minimum
        }

        /// The whole image — where a frame starts.
        package var whole: Box { Box(x: 0, y: 0, width: width, height: height) }

        /// Whether any frame smaller than the image exists: both sides at the floor or more, one of
        /// them above it.
        package var canCrop: Bool {
            width >= minimum && height >= minimum && (width > minimum || height > minimum)
        }

        /// The image's own proportion.
        package var originalRatio: Ratio { Ratio(width, height) }

        /// The ratio a shape holds (`nil`: free).
        package func ratio(of shape: Shape) -> Ratio? {
            switch shape {
            case .free: nil
            case .original: originalRatio
            case .fixed(let r): r
            }
        }

        /// Whether a frame of that proportion fits in the image at the floor.
        package func supports(_ shape: Shape) -> Bool {
            guard let r = ratio(of: shape) else { return true }
            return widths(r, roomWidth: width, roomHeight: height) != nil
        }

        // ── the widths a proportion allows ──

        /// The smallest and largest widths of a frame of that proportion in a room — `nil` if none.
        private func widths(_ r: Ratio, roomWidth: Int, roomHeight: Int) -> ClosedRange<Int>? {
            var low = max(minimum, (minimum * r.num + r.den - 1) / r.den)
            low = (low + r.step - 1) / r.step * r.step
            while r.height(forWidth: low) < minimum { low += r.step }
            var high = min(roomWidth, roomHeight * r.num / r.den + r.step) / r.step * r.step
            while high > 0, high > roomWidth || r.height(forWidth: high) > roomHeight { high -= r.step }
            return low <= high ? low...high : nil
        }

        /// A frame of that proportion near `wanted` pixels wide, clamped to what the room and the floor allow.
        private func size(_ r: Ratio, wanted: Double, roomWidth: Int, roomHeight: Int) -> (width: Int, height: Int)? {
            guard let range = widths(r, roomWidth: roomWidth, roomHeight: roomHeight) else { return nil }
            let snapped = Int((wanted / Double(r.step)).rounded()) * r.step
            let w = min(max(snapped, range.lowerBound), range.upperBound)
            return (w, r.height(forWidth: w))
        }

        // ── the gestures ──

        /// **Moves the frame**, kept inside the image.
        package func move(_ box: Box, dx: Int, dy: Int) -> Box {
            Box(x: min(max(box.x + dx, 0), width - box.width), y: min(max(box.y + dy, 0), height - box.height),
                width: box.width, height: box.height)
        }

        /// **Drags `edges` of the frame to the point** (pixels of the image, fractions allowed).
        /// Free, the sides go where the pointer is, kept apart by the floor and inside the image;
        /// with a proportion, the opposite side or corner stays and the other axis stays centered.
        package func resize(_ box: Box, edges: Edges, to p: CGPoint, ratio: Ratio?) -> Box {
            guard let r = ratio else { return resizeFree(box, edges: edges, to: p) }
            let growsLeft = edges.contains(.left), growsTop = edges.contains(.top)
            // The fixed side of each axis; a side not dragged is centered.
            let anchorX = growsLeft ? box.maxX : box.x, anchorY = growsTop ? box.maxY : box.y
            let centerX = Double(box.x) + Double(box.width) / 2, centerY = Double(box.y) + Double(box.height) / 2

            let roomW = edges.horizontal ? (growsLeft ? anchorX : width - anchorX)
                                         : 2 * Int(min(centerX, Double(width) - centerX))
            let roomH = edges.vertical ? (growsTop ? anchorY : height - anchorY)
                                       : 2 * Int(min(centerY, Double(height) - centerY))
            var wanted = 0.0
            if edges.horizontal { wanted = abs(Double(p.x) - Double(anchorX)) }
            if edges.vertical { wanted = max(wanted, abs(Double(p.y) - Double(anchorY)) * Double(r.num) / Double(r.den)) }
            guard let s = size(r, wanted: wanted, roomWidth: roomW, roomHeight: roomH) else { return box }

            let x = edges.horizontal ? (growsLeft ? anchorX - s.width : anchorX)
                                     : min(max(Int((centerX - Double(s.width) / 2).rounded()), 0), width - s.width)
            let y = edges.vertical ? (growsTop ? anchorY - s.height : anchorY)
                                   : min(max(Int((centerY - Double(s.height) / 2).rounded()), 0), height - s.height)
            return Box(x: x, y: y, width: s.width, height: s.height)
        }

        private func resizeFree(_ box: Box, edges: Edges, to p: CGPoint) -> Box {
            var left = box.x, right = box.maxX, top = box.y, bottom = box.maxY
            let px = Int(p.x.rounded()), py = Int(p.y.rounded())
            if edges.contains(.left) { left = min(max(px, 0), right - minimum) }
            if edges.contains(.right) { right = max(min(px, width), left + minimum) }
            if edges.contains(.top) { top = min(max(py, 0), bottom - minimum) }
            if edges.contains(.bottom) { bottom = max(min(py, height), top + minimum) }
            return Box(x: left, y: top, width: right - left, height: bottom - top)
        }

        /// **The frame once the proportion changes**: the largest of that proportion inside the
        /// current one, centered on it; if the floor leaves none there, the one of about the same area
        /// around the same center, brought inside the image. `nil` for a proportion the image cannot hold.
        package func fit(_ box: Box, to r: Ratio) -> Box? {
            let cx = Double(box.x) + Double(box.width) / 2, cy = Double(box.y) + Double(box.height) / 2
            if let s = size(r, wanted: Double(width), roomWidth: box.width, roomHeight: box.height) {
                return centered(s, on: (cx, cy))
            }
            let area = Double(box.width * box.height)
            guard let s = size(r, wanted: (area * Double(r.num) / Double(r.den)).squareRoot(),
                               roomWidth: width, roomHeight: height) else { return nil }
            return centered(s, on: (cx, cy))
        }

        private func centered(_ s: (width: Int, height: Int), on c: (Double, Double)) -> Box {
            Box(x: min(max(Int((c.0 - Double(s.width) / 2).rounded()), 0), width - s.width),
                y: min(max(Int((c.1 - Double(s.height) / 2).rounded()), 0), height - s.height),
                width: s.width, height: s.height)
        }
    }
}
