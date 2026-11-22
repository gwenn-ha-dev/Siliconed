import Foundation

/// **A box of tiles shown by whole rows** — the Prompt Lab's images (`LabImages` in the app): the tiles flow side by side
/// in rows, and the box's grip stops on whole rows only, so that what is kept is a number of rows, not a height that would
/// cut a row once the box's width changes.
package enum TileRows {
    /// **The heights at which the box shows whole rows**, the first row's then each next one's, from each tile's frame in the
    /// box: tiles whose middles lie within a point of each other share a row (a row centers its tiles), a row's bottom is
    /// its tallest tile's, and `padding` goes under it.
    package static func stops(_ frames: [CGRect], padding: CGFloat) -> [CGFloat] {
        var lines: [(middle: CGFloat, bottom: CGFloat)] = []
        for f in frames.sorted(by: { $0.midY < $1.midY }) {
            if let last = lines.last, f.midY - last.middle < 1 { lines[lines.count - 1].bottom = max(last.bottom, f.maxY) }
            else { lines.append((f.midY, f.maxY)) }
        }
        return lines.map { $0.bottom + padding }
    }

    /// The box's height: `rows` rows (one at least), or all there are when fewer; `empty` until the tiles are laid out.
    package static func height(_ stops: [CGFloat], rows: Int, empty: CGFloat) -> CGFloat {
        guard !stops.isEmpty else { return empty }
        return stops[min(max(rows, 1), stops.count) - 1]
    }

    /// **The rows a drag of the grip asks for**: the count whose height is the nearest to `wanted`, among the first `limit`
    /// rows; `nil` before the tiles are laid out.
    package static func rows(nearest wanted: CGFloat, in stops: [CGFloat], limit: Int) -> Int? {
        let stops = stops.prefix(limit)
        return stops.indices.min(by: { abs(stops[$0] - wanted) < abs(stops[$1] - wanted) }).map { $0 + 1 }
    }
}
