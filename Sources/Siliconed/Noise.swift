import Foundation

/// The initial noise of a trajectory, reproducible from a seed.
///
/// **This is not PyTorch's generator, and that is deliberate**:
/// reproducing its `randn_tensor` is not the point — reproducing the *trajectory* is. The
/// checks therefore start from the reference's `latent_init`, re-read and not removed; this generator
/// is only for free prompts, where there is no reference to compare against.
///
/// **A consequence to know**: the same seed gives the same image with us and **another**
/// image with `diffusers`. This is a property of the engine, not a defect to fix — fixing it
/// would require porting PyTorch's Mersenne Twister and its rejection-based normal, for zero
/// value.
package struct Noise {
    /// `splitmix64` — a single-state generator, with no table, no warm-up, and where each
    /// output bit depends on all the state bits. It is more than enough: what we ask of
    /// this noise is to be *reproducible* and *structureless*, not to be cryptographic.
    private var state: UInt64

    package init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }

    private mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    /// A uniform float in `(0, 1]` — never zero, because `log(0)` is the only way to
    /// make Box-Muller fail and it happens once in 2⁵³.
    private mutating func uniform() -> Double {
        Double((next() >> 11) &+ 1) * (1.0 / 9007199254740992.0)
    }

    /// `count` standard normal values, by Box-Muller.
    ///
    /// The polar form would be faster but it **rejects**, so its cost depends on the draw and
    /// it does not parallelize by deterministic slices. Here the number of draws is fixed, which
    /// makes the output a function of the seed alone — what one wants from a reproducible noise.
    package mutating func fill(_ destination: UnsafeMutablePointer<Float>, count: Int) {
        var i = 0
        while i < count {
            let u1 = uniform(), u2 = uniform()
            let radius = (-2 * Foundation.log(u1)).squareRoot()
            let angle = 2 * Double.pi * u2
            destination[i] = Float(radius * Foundation.cos(angle))
            if i + 1 < count { destination[i + 1] = Float(radius * Foundation.sin(angle)) }
            i += 2
        }
    }
}

/// **PyTorch's CPU `randn`, to the bit** — for Qwen-Image-2.1 only, whose pipeline (`prepare_latents`)
/// draws `randn((1, 1, 64, h, w), generator=torch.Generator("cpu").manual_seed(seed))`: the same seed
/// gives the reference's starting noise, so a free render can be put next to diffusers' and the
/// trajectory check starts from the engine's own noise instead of a golden one.
///
/// What `at::normal_fill` (`ATen/native/cpu/DistributionTemplates.h`, the path of a contiguous fp32
/// tensor of at least 16 values) does, and a port by analogy would miss:
///
///   - the generator is the 32-bit **MT19937**, seeded with the low 32 bits of the seed
///     (`at::mt19937(seed)`, `init_with_uint32`);
///   - **all the uniforms are drawn first**, one 32-bit word each, `(w & (2²⁴ − 1)) · 2⁻²⁴`;
///   - Box-Muller then runs **in blocks of 16**, pairing value `j` with `j + 8` (not `2j`, `2j + 1`):
///     `u₁ = 1 − data[j]`, `u₂ = data[j + 8]`, `r = √(−2 ln u₁)` in float, `θ = 2π·u₂` **in double**
///     (`2.0f * c10::pi<double> * u2`) rounded to float, `data[j] = r·cos θ`, `data[j + 8] = r·sin θ`;
///   - a size that is not a multiple of 16 **redraws** 16 fresh uniforms over the last 16 values
///     and transforms them again — the tail overwrites values already transformed. That path is
///     ported but NOT exact: `randn(20)` differs by one ulp in one value. A latent never takes it
///     (64 channels); the multiple-of-16 path is checked to the bit (tests, `qwen21-trajectory`).
///
/// The float `log`, `cos`, `sin`, `sqrt` are the platform's libm, the one PyTorch calls on this Mac.
package struct TorchNoise {
    private var state = [UInt32](repeating: 0, count: 624)
    private var index = 624

    package init(seed: UInt64) {
        state[0] = UInt32(truncatingIfNeeded: seed)
        for j in 1..<624 {
            let previous = state[j - 1]
            state[j] = 1_812_433_253 &* (previous ^ (previous >> 30)) &+ UInt32(j)
        }
    }

    /// One tempered MT19937 word.
    private mutating func next() -> UInt32 {
        if index >= 624 {
            for k in 0..<624 {
                let y = (state[k] & 0x8000_0000) | (state[(k + 1) % 624] & 0x7FFF_FFFF)
                state[k] = state[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1) != 0 ? 0x9908_B0DF : 0)
            }
            index = 0
        }
        var y = state[index]
        index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C_5680
        y ^= (y << 15) & 0xEFC6_0000
        y ^= y >> 18
        return y
    }

    /// `uniform_real_distribution<float>(0, 1)`: 24 bits, times 2⁻²⁴.
    private mutating func uniform() -> Float { Float(next() & 0xFF_FFFF) * Float(1.0 / 16_777_216.0) }

    private static func boxMuller16(_ data: UnsafeMutablePointer<Float>) {
        for j in 0..<8 {
            let u1 = 1 - data[j], u2 = data[j + 8]
            let radius = (-2 * Foundation.log(u1)).squareRoot()
            let theta = Float(2.0 * Double.pi * Double(u2))
            data[j] = radius * Foundation.cos(theta)
            data[j + 8] = radius * Foundation.sin(theta)
        }
    }

    /// `count` values of `torch.randn` (count ≥ 16, the only path a latent takes).
    package mutating func fill(_ data: UnsafeMutablePointer<Float>, count: Int) {
        precondition(count >= 16, "TorchNoise: fewer than 16 values take another path in PyTorch")
        for i in 0..<count { data[i] = uniform() }
        var i = 0
        while i + 16 <= count { TorchNoise.boxMuller16(data + i); i += 16 }
        if count % 16 != 0 {
            let tail = data + count - 16
            for k in 0..<16 { tail[k] = uniform() }
            TorchNoise.boxMuller16(tail)
        }
    }
}
