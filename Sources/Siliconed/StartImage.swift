import Foundation

// **The starting image, as an app holds it** — img2img for the families whose chain starts from an image (Z-Image, Anima,
// Krea 2: `DenoisingModule.acceptsStartImage` with an image encoder). The render itself is `ImageToImage.swift`; here are
// the rules the window and the remote control share, pure:
//
//     the Prompt Lab ── a picture made the start ─▶ « Carry to the Rack » ─▶ the rack's Images box: its thumbnail, its strength
//                                                                     └─▶ Size: the recommended format nearest its proportions
//
// - **The strength is diffusers' `strength`, untouched** (`Strength`): the percentage the slider shows is that number, the one the
//   PNG writes (« Denoising strength ») and the developer's `--strength` takes. 0 % keeps the image (no evaluation: refused), 100 %
//   ignores it. Kept in hundredths: the slider, the PNG's two decimals and the command line give back the same request.
// - **The format follows the image, the size stays free**: carrying a start sets the rack's size to the model's recommended format
//   nearest the image's proportions, at the scale the rack had; any other format is still the user's to choose, the engine
//   filling it and cropping the centre (`ImageRGB.fitted`). Nothing is stretched.
package enum StartImage {
    /// **The strength a starting image comes with, by family** — settled on fixed-seed sheets (a surfer and her two friends
    /// turned into fishermen, then teenagers of the eighties; seed 7, strengths 0.30 to 0.90): where the people change and the
    /// pose stays. Krea 2 keeps the pose best and changes them from 65 %; Z-Image and Anima at 75 %; MOSS-Image holds to the
    /// image and needs 80 %. At eight steps the strength has eight real notches (`Strength.startStep`): 65 % redraws six of
    /// them as 75 % does, 80 % seven. Another family (none starts from an image yet) takes 65 %.
    package static func defaultStrength(for family: Family?) -> Double {
        switch family {
        case .zImage, .anima: 0.75
        case .mossImage: 0.80
        default: 0.65
        }
    }

    /// **How a strength is kept**: in hundredths, within `0…1` — what a slider dragged anywhere, a PNG's `%.2f` and a typed
    /// `0.655` all come back to.
    package static func snapped(_ strength: Double) -> Double {
        guard strength.isFinite else { return 0 }
        return min(1, max(0, (strength * 100).rounded() / 100))
    }

    /// **A strength as typed** — `0.65`, `0,65` or `65%` — or `nil` when it does not read or lies outside `]0 ; 1]`.
    package static func strength(_ text: String) -> Double? {
        let t = text.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")
        let value: Double?
        if t.hasSuffix("%") { value = Double(t.dropLast().trimmingCharacters(in: .whitespaces)).map { $0 / 100 } }
        else { value = Double(t) }
        guard let v = value, v.isFinite, v > 0, v <= 1 else { return nil }
        return v
    }

    /// **The recommended format for a starting image of `width × height`**: among `formats` (the model's), the one whose
    /// proportions are nearest the image's (in log ratio: 2:3 and 3:2 are as far from 1:1); among those, the one whose area is
    /// nearest `current`'s — the scale the rack had. `nil` when `formats` is empty or the image has no size.
    package static func recommendedFormat(width: Int, height: Int, among formats: [RecommendedFormat],
                                          current: RecommendedFormat) -> RecommendedFormat? {
        guard width > 0, height > 0 else { return nil }
        let ratio = log(Double(width) / Double(height))
        func distance(_ f: RecommendedFormat) -> Double { abs(log(Double(f.width) / Double(f.height)) - ratio) }
        let area = Double(current.width * current.height)
        func scale(_ f: RecommendedFormat) -> Double { abs(log(Double(f.width * f.height) / area)) }
        // Proportions first, to a tolerance that keeps 896×1152 and 832×1216 apart; then the scale; then the catalog's order.
        return formats.enumerated().min { a, b in
            let da = distance(a.element), db = distance(b.element)
            if abs(da - db) > 1e-9 { return da < db }
            let sa = scale(a.element), sb = scale(b.element)
            if abs(sa - sb) > 1e-9 { return sa < sb }
            return a.offset < b.offset
        }?.element
    }
}
