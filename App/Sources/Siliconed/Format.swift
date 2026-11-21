import Foundation

// **The formats a render accepts** (`Format`): the per-side floor, the multiple of 16, the longest
// side, and what a family adds — a coarser multiple, a shorter longest side. There is no area ceiling:
// the engine plans the render under B or refuses it. The request that carries them is in `Request.swift`.

/// **The formats a render accepts — three rules, each with its reason, and none of them a memory
/// ceiling.**
///
/// 1. **Each side ≥ 512 px** (`minimumSide`), read **per side** and not per area: 832×1216 has the
///    area of 1024², but a 416×608 would have that of 512² with one side outside the models' domain.
/// 2. **Each side a multiple of 16**: the VAE divides by 8, the DiTs' patch by 2.
/// 3. **Each side at most `maximumSide(for:)`**: what a position table or a 32-bit index can address,
///    not what memory can hold. It is the guard against an absurd request (`99 999 999 984²` would
///    overflow the product), and nothing more.
///
/// **The area is not bounded here**: a Mac with 32, 64 or 128 GB renders what its B holds. Memory is the
/// engine's: it plans each session under B (`MemoryBudget.machineCeiling`) or refuses it, and the render
/// fails with `insufficientMemory`, which says what the plan needed.
///
/// **A family may ask more** (`check(width:height:family:)`, what the preflight applies), each read
/// in its reference and declared in its traits: a coarser multiple (`FamilyTraits.sideMultiple`: a VAE
/// that divides by 16 under a patch of 2), refused before any computation rather than floored without
/// a word as a pipeline may; a shorter longest side (`FamilyTraits.maximumSide`: a RoPE table's
/// positions, past which the reference fails and the engine would extrapolate).
package enum Format {
    /// **The smallest side, in pixels: 512.** Below 512² the
    /// models are out of their domain, so nothing rendered there — neither image nor timing — means
    /// anything. **Per side**, rectangles included.
    package static let minimumSide = 512
    package static let multiple = 16

    /// **The longest side any render takes, in pixels: 8 192** — the guard against an absurd request, past which the VAEs'
    /// 32-bit element indexes would not address the picture. A family whose model draws less says so in its traits
    /// (`FamilyTraits.maximumSide`: a RoPE table's positions).
    package static let maximumSide = 8192

    /// **The area an edit's output is kept under, in pixels (1024×1536)** — not a memory bound. An edit
    /// takes the first reference's proportions at about 1 megapixel (the reference pipeline's own
    /// `calculate_dimensions`); raising a long strip's short side to the floor can push its area past
    /// that, and its long side then gives way down to this area. The figure is the one the edit
    /// formats were measured and judged at; B still decides whether the Mac renders them.
    package static let editSurface = 1024 * 1536

    /// **The multiple each side of `family` must be.**
    package static func multiple(for family: Family) -> Int { family.traits.sideMultiple ?? multiple }

    /// **The longest side `family` renders**, in pixels.
    package static func maximumSide(for family: Family) -> Int { min(maximumSide, family.traits.maximumSide ?? maximumSide) }

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
    /// portrait. `×` counts as `x`, spaces around it nothing (`832 × 1216`, as `label` writes it);
    /// a lone integer is a square. Checks nothing but the syntax.
    package static func parse(_ text: String) -> (width: Int, height: Int)? {
        let pieces = text.lowercased().replacingOccurrences(of: "×", with: "x")
            .split(separator: "x", omittingEmptySubsequences: false).map { $0.trimmingCharacters(in: .whitespaces) }
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
