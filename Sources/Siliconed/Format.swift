import Foundation

// **The formats a render accepts** (`Format`): the per-side floor, the multiple of 16, the longest
// side, and what a family adds — a coarser multiple, a shorter longest side. There is no area ceiling:
// what a format costs is computed (`MemoryNeed`) and judged against B (`MemoryBudget`). The request that
// carries them is in `Request.swift`.

/// **The formats a render accepts — three rules, each with its reason, and none of them a memory
/// ceiling.**
///
/// 1. **Each side ≥ 512 px** (latent ≥ 64). This is the repository's floor
///    (`Spectral.minimumSide`), read **per side** and not per area: 832×1216 has the area of
///    1024², but a 416×608 would have that of 512² with one side outside the models' domain. The
///    floor applies to *every* evaluation — hence the spectral schedule's rule, which goes down
///    only if both half-sizes stay at the floor (`Spectral.divisible`).
/// 2. **Each side a multiple of 16**: the VAE divides by 8, the DiTs' patch by 2.
/// 3. **Each side at most `maximumSide(for:)`**: what a position table or a 32-bit index can address,
///    not what memory can hold. It is the guard against an absurd request (`99 999 999 984²` would
///    overflow the product), and nothing more.
///
/// **The area is not bounded here.** It used to be (1024×1536, "the last format measured"); a Mac
/// with 32, 64 or 128 GB renders what its B holds (`MemoryBudget.ceiling`), and the preflight
/// decides it from the formula (`ModelCard.checkMemory`) before the first computation: the
/// request's lean plan against B, or `insufficientMemory`, which says what the render needs. A format
/// the formula cannot price on this engine's own limits (`MemoryNeed.limit`) is refused there too.
///
/// **A family may ask more** (`check(width:height:family:)`, what the preflight applies), each read
/// in its reference:
///
/// - **Qwen-Image-2.1: each side a multiple of 32** — its VAE divides by 16, the patch by 2.
///   `pipeline_qwenimage21.py` floors a side to 32 without a word; the engine refuses it instead,
///   before any computation: 848×1216 would otherwise die after the encoder.
/// - **Anima: each side at most 2048 px** — the reference's RoPE table holds 128 positions per side
///   (`AnimaDiT.Config.ropePositions`); past it the reference fails, and the engine would extrapolate.
package enum Format {
    /// The smallest side, in pixels: `Spectral.minimumSide` latent cells of 8 px.
    package static let minimumSide = Spectral.minimumSide * 8
    package static let multiple = 16

    /// **The longest side any model draws, in pixels: 8 192** — Z-Image's RoPE table holds 512 positions
    /// on each spatial axis (`DiT.Config.publishedAxesLens`), one per 16 px, and clamps past them without
    /// a word. Qwen-Image-2.1 and Krea 2 compute their positions by formula and have no table of their
    /// own; they take the same guard, beyond which the VAEs' 32-bit element indexes (`MemoryNeed.limit`)
    /// would refuse the format in any case.
    package static let maximumSide = (DiT.Config.publishedAxesLens[1] * 2) * LatentSpace.flux.factor

    /// **The area an edit's output is kept under, in pixels (1024×1536)** — not a memory bound. An edit
    /// takes the first reference's proportions at about 1 megapixel (the reference pipeline's own
    /// `calculate_dimensions`); raising a long strip's short side to the floor can push its area past
    /// that, and its long side then gives way down to this area. The figure is the one the edit
    /// formats were measured and judged at; B still decides whether the Mac renders them.
    package static let editSurface = 1024 * 1536

    /// **The multiple each side of `family` must be**: 32 for Qwen-Image-2.1 (VAE ×16, patch ×2), 16
    /// otherwise.
    package static func multiple(for family: Family) -> Int {
        family == .qwenImage21 ? 32 : multiple
    }

    /// **The longest side `family` renders**, in pixels: Anima's RoPE table (128 positions × patch 2 ×
    /// VAE 8 = 2048), `maximumSide` otherwise (Z-Image's table gives it).
    package static func maximumSide(for family: Family) -> Int {
        switch family {
        case .anima:
            return min(maximumSide, AnimaDiT.Config.publishedRopePositions * 2 * LatentSpace.qwenImage.factor)
        case .zImage, .krea2, .qwenImage21, .mossImage: return maximumSide
        }
    }

    /// Throws if the format is not renderable by any model. The order of the rules is that of the
    /// most useful error.
    package static func check(width: Int, height: Int) throws(EngineError) {
        try check(width: width, height: height, multiple: multiple, maximumSide: maximumSide)
    }

    /// **Throws if `family` does not render the format** — the general rules, then the family's own:
    /// what the preflight checks, what the app's size field judges.
    package static func check(width: Int, height: Int, family: Family) throws(EngineError) {
        try check(width: width, height: height, multiple: multiple(for: family), maximumSide: maximumSide(for: family))
    }

    private static func check(width: Int, height: Int, multiple: Int, maximumSide side: Int) throws(EngineError) {
        guard width >= minimumSide, height >= minimumSide else {
            throw .imageTooSmall(side: min(width, height), minimum: minimumSide)
        }
        guard width % multiple == 0, height % multiple == 0 else {
            throw .formatRefused(width: width, height: height, reason: .notMultiple, bound: multiple)
        }
        // Each side bounded BEFORE anything is multiplied.
        guard width <= side, height <= side else {
            throw .formatRefused(width: width, height: height, reason: .sideTooLong, bound: side)
        }
    }

    /// **`LxH`: width × height**, the order of Draw Things and of screens — `832x1216` is a
    /// portrait. `×` counts as `x`; a lone integer is a square. Checks nothing but the syntax.
    package static func parse(_ text: String) -> (width: Int, height: Int)? {
        let pieces = text.lowercased().replacingOccurrences(of: "×", with: "x")
            .split(separator: "x", omittingEmptySubsequences: false)
        switch pieces.count {
        case 1: return Int(pieces[0]).map { ($0, $0) }
        case 2:
            guard let l = Int(pieces[0]), let h = Int(pieces[1]) else { return nil }
            return (l, h)
        default: return nil
        }
    }

    /// `512` for a square, `832x1216` otherwise — what a file name carries. Square names do not
    /// change: the close-out and the journal find them again.
    package static func fileLabel(width: Int, height: Int) -> String {
        width == height ? "\(width)" : "\(width)x\(height)"
    }

    /// `512²` or `832×1216` — what is displayed.
    package static func label(width: Int, height: Int) -> String {
        width == height ? "\(width)²" : "\(width)×\(height)"
    }
}
