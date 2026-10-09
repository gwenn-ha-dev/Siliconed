import Foundation

/// **A variation of an image: its starting noise turned a little towards another seed's** — Draw
/// Things' and AUTOMATIC1111's "variation seed", without the two fields: an app offers `subtle` and
/// `strong` (`Amount`), and the composition of the image it starts from survives in proportion.
///
///     origin seed ──► noise a ─┐
///                              ├─► cos θ · a + sin θ · b,   θ = strength · π/2   ──► the render, unchanged
///     variation seed ─► noise b┘
///
/// What distinguishes it from the reference implementations, and a port by analogy would miss:
///
///   - **the mix is a rotation, not a slerp.** For two independent standard normal draws,
///     `cos θ · a + sin θ · b` is *exactly* a standard normal again, whatever θ (cos² + sin² = 1):
///     the denoiser always receives the noise it was trained on. AUTOMATIC1111's `slerp`
///     (`modules/rng.py`) interpolates along the arc between the two draws using their **measured**
///     angle ω — `acos` of the normalized dot product along `dim=1` of a `[C, H, W]` noise, i.e. per
///     channel and column, over the rows — so its variance is 1 only where ω happens to be π/2; ours
///     takes ω = π/2, its expectation, and is exact in distribution. At ω = π/2 the two coincide.
///   - **the second noise comes from the same generator, at the same shape**, as the first
///     (`Latent.noise`): `Noise` (splitmix64 + Box-Muller) for five families, `TorchNoise` (PyTorch's
///     CPU `randn`) for Qwen-Image-2.1. A variation is therefore defined on the latent, not on pixels:
///     the same pair of seeds at another format is another image.
///   - **strength 0 is the original noise to the bit** — a special case, not `cos 0 · a + sin 0 · b`
///     (which would also be `a`, but only by the grace of `sin 0 = 0` and `−0` arithmetic; the
///     contract is stated, not derived). Strength 1 is the variation seed's noise alone: a re-roll.
///   - **variations chain.** `Request.variations` is applied in order, each to the result of the
///     previous one: a variation of a variation turns *that* image's noise, so its cousins stay
///     close to it, not to its parent. Since each step keeps a standard normal, so does the chain.
///   - computed in fp64 element by element, rounded once to fp32 — the order of operations of one
///     element is fixed, so the same pair of seeds gives the same bits on any machine.
public struct Variation: Sendable, Hashable {
    public var seed: UInt64
    /// In `[0, 1]`: 0 the original noise, 1 the variation seed's. Outside, clamped; not finite, 0.
    public var strength: Double

    public init(seed: UInt64, strength: Double) { self.seed = seed; self.strength = strength }

    /// **The two strengths an app offers**, named and never shown as numbers — calibrated at 512²,
    /// seed 42:
    ///
    ///   - `subtle`: same framing, light, clothes, pose and person; a hand, a face's expression, the
    ///     shelves change. **Per family**, because Qwen's turbo schedule decides who is in the image in
    ///     its first steps (σ 1 · 0.96 · 0.92): at 0.1 one variation seed in two gave another woman,
    ///     at 0.2 another outfit, while 0.03 · 0.05 · 0.07 kept the same person on eight seeds out of
    ///     eight (PSNR to the original 23–25 · 21–23.5 · 20–22 dB). Qwen takes 0.05, with that margin
    ///     on both sides; Z-Image keeps 0.1 (at 0.05 it already moves a hand: no strength keeps every
    ///     detail). The other families, never calibrated, take Z-Image's.
    ///   - `strong` 0.5: the same idea (a woman in a library, the same kind of shot) with the pose,
    ///     the clothes and the framing that move. Z-Image keeps its framing longer (its seeds vary
    ///     little), at 0.7 it starts to change shot; Qwen at 0.7 is nearly another image.
    public enum Amount: String, CaseIterable, Sendable {
        case subtle, strong

        public func strength(for family: Family) -> Double {
            switch (self, family) {
            case (.subtle, .qwenImage21): 0.05
            case (.subtle, _): 0.1
            case (.strong, _): 0.5
            }
        }

        /// The amount whose strength this is, for some family — to name a variation read back.
        public static func named(_ strength: Double) -> Amount? {
            allCases.first { a in Family.allCases.contains { a.strength(for: $0) == strength } }
        }
    }

    /// The strength actually applied.
    package var effectiveStrength: Double { strength.isFinite ? min(1, max(0, strength)) : 0 }

    /// `noise ← cos θ · noise + sin θ · other`, θ = strength · π/2, in fp64 rounded to fp32.
    /// Strength 0: `noise` untouched.
    package static func mix(_ noise: inout [Float], _ other: [Float], strength: Double) {
        precondition(noise.count == other.count, "Variation: \(noise.count) values against \(other.count)")
        let s = strength.isFinite ? min(1, max(0, strength)) : 0
        guard s > 0 else { return }
        let theta = s * Double.pi / 2
        let c = Foundation.cos(theta), sn = Foundation.sin(theta)
        noise.withUnsafeMutableBufferPointer { a in
            other.withUnsafeBufferPointer { b in
                for i in 0..<a.count { a[i] = Float(c * Double(a[i]) + sn * Double(b[i])) }
            }
        }
    }

    /// `seed:strength,seed:strength` — what the PNG metadata carries (`Engine.Render.metadata`). The
    /// strength is written in its shortest exact form, so that it reads back to the same `Double`.
    package static func text(_ variations: [Variation]) -> String {
        variations.map { "\($0.seed):\($0.strength)" }.joined(separator: ",")
    }

    /// The inverse of `text`; `nil` if any entry does not parse.
    package static func parse(_ text: String) -> [Variation]? {
        var out: [Variation] = []
        for entry in text.split(separator: ",") {
            let pieces = entry.trimmingCharacters(in: .whitespaces).split(separator: ":")
            guard pieces.count == 2, let seed = UInt64(pieces[0]), let s = Double(pieces[1]),
                  s.isFinite, s >= 0, s <= 1 else { return nil }
            out.append(Variation(seed: seed, strength: s))
        }
        return out.isEmpty ? nil : out
    }
}

extension Latent {
    /// **The starting noise of a seed, turned by its variations** (`Variation`), in order. No
    /// variation, or only null strengths: exactly `noise(_:height:width:seed:)`, to the bit.
    package static func noise(_ space: LatentSpace, height: Int, width: Int, seed: UInt64,
                              variations: [Variation]) -> Latent {
        var latent = noise(space, height: height, width: width, seed: seed)
        for v in variations where v.effectiveStrength > 0 {
            let other = noise(space, height: height, width: width, seed: v.seed)
            Variation.mix(&latent.values, other.values, strength: v.strength)
        }
        return latent
    }
}
