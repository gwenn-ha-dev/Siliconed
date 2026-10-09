import Accelerate
import Foundation
import XCTest
@testable import Siliconed

/// **Variations** (`Variation`): strength 0 is the original noise to the bit, the mix keeps a
/// standard normal, a pair of seeds always gives the same bits, and the PNG carries it back.
final class VariationTests: XCTestCase {

    private func bits(_ l: Latent) -> [UInt32] { l.values.map(\.bitPattern) }

    /// Strength 0 — alone, or among others — and no variation at all: the seed's noise, to the bit,
    /// for both generators (`Noise` and Qwen's `TorchNoise`).
    func testStrengthZeroIsTheOriginalNoiseToTheBit() {
        for space in [LatentSpace.flux, .qwenImage21] {
            let plain = Latent.noise(space, height: 16, width: 24, seed: 42)
            XCTAssertEqual(bits(Latent.noise(space, height: 16, width: 24, seed: 42, variations: [])), bits(plain))
            XCTAssertEqual(bits(Latent.noise(space, height: 16, width: 24, seed: 42,
                                             variations: [Variation(seed: 7, strength: 0)])), bits(plain))
            // A non-finite or negative strength counts as 0; above 1, as 1.
            XCTAssertEqual(bits(Latent.noise(space, height: 16, width: 24, seed: 42,
                                             variations: [Variation(seed: 7, strength: .nan),
                                                          Variation(seed: 8, strength: -1)])), bits(plain))
            XCTAssertEqual(bits(Latent.noise(space, height: 16, width: 24, seed: 42,
                                             variations: [Variation(seed: 7, strength: 3)])),
                           bits(Latent.noise(space, height: 16, width: 24, seed: 42,
                                             variations: [Variation(seed: 7, strength: 1)])))
            // Strength 1: the variation seed's noise (cos π/2 is 6·10⁻¹⁷, under half an ulp of any
            // value that is not tiny: compared to within one ulp).
            let full = Latent.noise(space, height: 16, width: 24, seed: 42, variations: [Variation(seed: 7, strength: 1)])
            let other = Latent.noise(space, height: 16, width: 24, seed: 7)
            for (a, b) in zip(full.values, other.values) { XCTAssertEqual(a, b, accuracy: max(1e-15, b.ulp)) }
        }
    }

    /// The same pair of seeds gives the same bits; another variation seed, another noise; the order
    /// of a chain matters.
    func testAPairOfSeedsIsDeterministic() {
        let v = [Variation(seed: 1001, strength: Variation.Amount.subtle.strength(for: .zImage))]
        let a = Latent.noise(.qwenImage21, height: 16, width: 16, seed: 42, variations: v)
        let b = Latent.noise(.qwenImage21, height: 16, width: 16, seed: 42, variations: v)
        XCTAssertEqual(bits(a), bits(b))
        let c = Latent.noise(.qwenImage21, height: 16, width: 16, seed: 42,
                             variations: [Variation(seed: 1002, strength: Variation.Amount.subtle.strength(for: .zImage))])
        XCTAssertNotEqual(bits(a), bits(c))
        let chain1 = Latent.noise(.flux, height: 16, width: 16, seed: 42,
                                  variations: [Variation(seed: 1, strength: 0.3), Variation(seed: 2, strength: 0.6)])
        let chain2 = Latent.noise(.flux, height: 16, width: 16, seed: 42,
                                  variations: [Variation(seed: 2, strength: 0.6), Variation(seed: 1, strength: 0.3)])
        XCTAssertNotEqual(bits(chain1), bits(chain2))
        // A chain is the first step's result turned again.
        var step = Latent.noise(.flux, height: 16, width: 16, seed: 42, variations: [Variation(seed: 1, strength: 0.3)]).values
        Variation.mix(&step, Latent.noise(.flux, height: 16, width: 16, seed: 2).values, strength: 0.6)
        XCTAssertEqual(step.map(\.bitPattern), bits(chain1))
    }

    /// **The mix keeps a standard normal**: on 2¹⁸ values, mean ≈ 0 and variance ≈ 1 at every strength
    /// (a slerp with the measured angle would not), and the correlation with the origin is cos θ.
    func testTheMixKeepsAStandardNormal() {
        let n = 1 << 18
        // Widened and summed in fp64 by vDSP: the element loops cost ~0.15 s in a debug build.
        func wide(_ x: [Float]) -> [Double] { vDSP.floatToDouble(x) }
        let a = wide(Latent.noise(.flux, height: 128, width: 128, seed: 42).values)
        XCTAssertEqual(a.count, n)
        for s in [0.1, Variation.Amount.subtle.strength(for: .zImage), Variation.Amount.strong.strength(for: .zImage), 0.9] {
            let m = wide(Latent.noise(.flux, height: 128, width: 128, seed: 42, variations: [Variation(seed: 99, strength: s)]).values)
            var sum = 0.0, squares = 0.0, cross = 0.0
            vDSP_sveD(m, 1, &sum, vDSP_Length(n))
            vDSP_svesqD(m, 1, &squares, vDSP_Length(n))
            vDSP_dotprD(m, 1, a, 1, &cross, vDSP_Length(n))
            let mean = sum / Double(n), variance = squares / Double(n) - mean * mean
            // σ of the mean estimate: 1/√n ≈ 0.002; of the variance: √(2/n) ≈ 0.0028. 5σ bounds.
            XCTAssertEqual(mean, 0, accuracy: 0.01, "strength \(s)")
            XCTAssertEqual(variance, 1, accuracy: 0.014, "strength \(s)")
            XCTAssertEqual(cross / Double(n), cos(s * .pi / 2), accuracy: 0.014, "strength \(s)")
        }
    }

    /// The PNG carries the variations, in order, and reads them back to the same `Double`s; the
    /// origin seed stays in `seed`. Unreadable: said, not guessed.
    func testTheVariationsReadBackFromThePNG() throws {
        let vs = [Variation(seed: 4_000_000_000_123, strength: Variation.Amount.subtle.strength(for: .zImage)), Variation(seed: 9, strength: 0.1)]
        var r = Engine.Render(model: "z-image", prompt: "a 30 year old woman posing in a library", seed: 42, steps: 8,
                              loras: [], loraSummary: nil,
                              image: ImageRGB(pixels: [Float](repeating: 0, count: 3 * 16 * 16), height: 16, width: 16),
                              reproducible: true, evaluations: 7, sketch: nil, spectral: 0, strength: nil,
                              startStep: 0, startSigma: 1, timings: .init(), footprints: .init(), tokens: 1)
        XCTAssertNil(r.metadata["variation"], "an ordinary render writes no variation")
        r.variations = vs
        let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: PNG.text(try r.png())))
        XCTAssertEqual(recipe.seed, 42)
        XCTAssertEqual(recipe.variations, vs)
        XCTAssertEqual(recipe.unreadable, [])
        let bad = try XCTUnwrap(Engine.Render.Recipe(metadata: ["Software": "Siliconed", "variation": "12:1.5"]))
        XCTAssertEqual(bad.variations, []); XCTAssertEqual(bad.unreadable, ["variation"])
    }

    /// A request with variations differs from one without (an app relaunches on `onChange`), and the
    /// variation is not part of what the render cache keys: only the noise sees it.
    func testARequestCarriesItsVariations() {
        var a = Request("p", resolution: 512, seed: 42)
        let b = a
        a.variations = [Variation(seed: 1, strength: Variation.Amount.strong.strength(for: .zImage))]
        XCTAssertNotEqual(a, b)
    }

    /// Each family's two amounts: subtle below strong, and a strength read back from a PNG named again
    /// whatever the family (Qwen's subtle is its own).
    func testTheAmountsArePerFamilyAndNamedBack() {
        for f in Family.allCases {
            XCTAssertLessThan(Variation.Amount.subtle.strength(for: f), Variation.Amount.strong.strength(for: f))
            for a in Variation.Amount.allCases { XCTAssertEqual(Variation.Amount.named(a.strength(for: f)), a) }
        }
        XCTAssertEqual(Variation.Amount.subtle.strength(for: .qwenImage21), 0.05)
        XCTAssertNil(Variation.Amount.named(0.3))
    }
}
