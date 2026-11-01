import Foundation

// **The two-scale Size choice** (Settings › Advanced › Beta): a proportion, then a resolution, and the app works out the
// pixels. One table for every family, a pure function: the rack's menus and `silicontrol add`/`grid` (`3:4@2K`) read
// the same cells, so that a size means the same pixels wherever it is asked. The default Size menu (`ModelCard.recommendedFormats`)
// is untouched; this is a second way to reach a format, and `Format.check` still has the last word.

/// A proportion offered by the two-scale Size choice: width : height, as the user reads it (`3:4` is a portrait).
package struct SizeRatio: Sendable, Hashable, CustomStringConvertible {
    package let width: Int
    package let height: Int
    package init(_ width: Int, _ height: Int) { self.width = width; self.height = height }
    package var description: String { "\(width):\(height)" }
    package var orientation: RecommendedFormat.Orientation {
        width == height ? .square : (height > width ? .portrait : .landscape)
    }
}

/// 1K is about one megapixel (like 1024²), 2K about four (like 2048²): the names image tools use.
package enum SizeResolution: String, Sendable, Hashable, CaseIterable {
    case oneK = "1K", twoK = "2K"
}

/// What one cell of the table is: the pixels, and how far their proportion is from the one asked.
package struct SizeCell: Sendable, Hashable {
    package let ratio: SizeRatio
    package let resolution: SizeResolution
    package let width: Int
    package let height: Int
    package var format: RecommendedFormat { RecommendedFormat(width, height) }
    package var area: Int { width * height }
    /// `|actual / asked − 1|` of the proportions: 0 where the proportion is exact.
    package var deviation: Double {
        abs(Double(width) * Double(ratio.height) / (Double(height) * Double(ratio.width)) - 1)
    }
    /// The proportion the pixels really are, reduced: 768 × 1344 is 4:7.
    package var actualRatio: SizeRatio {
        func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
        let g = gcd(width, height)
        return SizeRatio(width / g, height / g)
    }
    /// Shown as « ≈ 9:16 » when the pixels are not that proportion to the tenth of a percent.
    package var isApproximate: Bool { deviation >= 0.001 }
}

package enum SizeScaleError: Error, Equatable, Sendable {
    /// Not `W:H@1K|2K`.
    case unreadable(String)
    /// A well-formed proportion that the table does not have.
    case unknownRatio(String)
    case unknownResolution(String)
}

package enum SizeScales {
    /// **Every side a multiple of 32** — Qwen-Image-2.1's rule (VAE ×16, patch ×2); the other families take 16, so 32 serves all.
    package static let multiple = 32

    /// The proportions, square first, then portraits from stockiest to most elongated, then their transposes. The issue that
    /// asked for the choice listed 1:1, 4:5, 3:4, 2:3, 9:16, 3:2 and 16:9; 5:4 and 4:3 complete the landscapes
    /// so that the orientation switch offers as many landscapes as portraits.
    package static let portraits = [SizeRatio(4, 5), SizeRatio(3, 4), SizeRatio(2, 3), SizeRatio(9, 16)]
    package static let landscapes = [SizeRatio(5, 4), SizeRatio(4, 3), SizeRatio(3, 2), SizeRatio(16, 9)]
    package static let ratios = [SizeRatio(1, 1)] + portraits + landscapes

    package static func ratios(for orientation: RecommendedFormat.Orientation) -> [SizeRatio] {
        switch orientation {
        case .square: return [SizeRatio(1, 1)]
        case .portrait: return portraits
        case .landscape: return landscapes
        }
    }

    /// **The pixels, portrait cells (width, height) for 1K then 2K.** Each is a multiple of 32 on both sides, picked
    /// for the closest proportion within about one megapixel (1K) or four (2K):
    /// - 4:5, 3:4, 2:3 are exact at both scales (the 2K cells double the 1K ones).
    /// - **9:16 is not reachable at 1K**: an exact 9:16 in multiples of 32 comes only in steps of 288 × 512 (576 × 1024 is
    ///   0.59 Mpx, 864 × 1536 is 1.33 Mpx), so 1K keeps the classic 768 × 1344, which is a 4:7 (1.6 % off) and is shown
    ///   as « ≈ 9:16 ». At 2K, 1440 × 2560 (3.69 Mpx) is exactly 9:16.
    private static let portraitPixels: [SizeRatio: [SizeResolution: (Int, Int)]] = [
        SizeRatio(4, 5): [.oneK: (896, 1120), .twoK: (1792, 2240)],
        SizeRatio(3, 4): [.oneK: (864, 1152), .twoK: (1728, 2304)],
        SizeRatio(2, 3): [.oneK: (832, 1248), .twoK: (1664, 2496)],
        SizeRatio(9, 16): [.oneK: (768, 1344), .twoK: (1440, 2560)],
    ]

    /// The cell of `ratio` at `resolution`; `nil` for a proportion the table does not have. A landscape is its portrait turned.
    package static func cell(_ ratio: SizeRatio, _ resolution: SizeResolution) -> SizeCell? {
        if ratio == SizeRatio(1, 1) {
            let side = resolution == .oneK ? 1024 : 2048
            return SizeCell(ratio: ratio, resolution: resolution, width: side, height: side)
        }
        if let (w, h) = portraitPixels[ratio]?[resolution] {
            return SizeCell(ratio: ratio, resolution: resolution, width: w, height: h)
        }
        if let (w, h) = portraitPixels[SizeRatio(ratio.height, ratio.width)]?[resolution] {
            return SizeCell(ratio: ratio, resolution: resolution, width: h, height: w)
        }
        return nil
    }

    /// The table cell whose pixels are `format`, if the format is one — what the rack shows selected.
    package static func cell(matching format: RecommendedFormat) -> SizeCell? {
        for ratio in ratios {
            for resolution in SizeResolution.allCases {
                if let c = cell(ratio, resolution), c.width == format.width, c.height == format.height { return c }
            }
        }
        return nil
    }

    /// **Why `family` does not render the cell, or `nil`**: the engine's own rules (`Format.check`), so the menu greys
    /// exactly what the preflight would refuse — Anima's 2048 px a side leaves its 2K to the square. The memory is the
    /// caller's (`macNeeded`), as for the default menu.
    package static func refusal(_ cell: SizeCell, family: Family) -> EngineError? {
        do { try Format.check(width: cell.width, height: cell.height, family: family); return nil }
        catch { return error }
    }

    /// **`3:4@2K`** → the cell. `@` and `:` mean nothing to a shell, so it is typed without quotes; `x` is already
    /// `WxH`'s. Case does not matter (`3:4@2k`).
    package static func parse(_ text: String) throws(SizeScaleError) -> SizeCell {
        let parts = text.split(separator: "@", omittingEmptySubsequences: false)
        guard parts.count == 2 else { throw .unreadable(text) }
        let sides = parts[0].split(separator: ":", omittingEmptySubsequences: false)
        guard sides.count == 2, let w = Int(sides[0]), let h = Int(sides[1]), w > 0, h > 0 else { throw .unreadable(text) }
        guard let resolution = SizeResolution(rawValue: parts[1].uppercased()) else { throw .unknownResolution(String(parts[1])) }
        guard let c = cell(SizeRatio(w, h), resolution) else { throw .unknownRatio("\(w):\(h)") }
        return c
    }

    /// Whether `text` is written as a scale (a colon and an `@`), so a caller can tell it from `WxH`.
    package static func looksLikeScale(_ text: String) -> Bool { text.contains("@") && text.contains(":") }

    /// The proportions the table has, as written, for a refusal's help.
    package static var ratioList: String { ratios.map(\.description).joined(separator: ", ") }
}
