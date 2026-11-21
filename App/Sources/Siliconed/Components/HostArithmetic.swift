import Foundation

// **What the families compute alike on the CPU**, as their references compute it there: a table's cosine and sine, a
// timestep's sinusoid, a value rounded to bfloat16.

/// **A cosine and a sine of a table the CPU hands a function** (a RoPE table, a timestep's sinusoid): of `angle` as the
/// reference forms it — an f32 product given as its Double, an f64 one as it is —, computed in f64 and rounded once to
/// f32, the f32 nearest the true value. Every table passes here. `cosf` and `sinf` were a step off on 48 of the 8192
/// values of Qwen3's table at 64 rows, each farther from f64 (19/11): Krea 2's tap 32 past its threshold, Z's encoder
/// off the proto's bits.
func cosSin(_ angle: Double) -> (cos: Float, sin: Float) { (Float(Foundation.cos(angle)), Float(Foundation.sin(angle))) }

/// **A timestep's sinusoid in float32 as torch computes it**, the time already scaled (`t · 1000` where the model
/// multiplies it): `[cos | sin]` of `scaled · exp(−ln(10⁴) · i / half)`, each operation one float rounding. Computed on
/// the CPU, as diffusers' `get_timestep_embedding` runs under disabled autocast.
func timestepSinusoid(_ scaled: Float, half: Int) -> [Float] {
    let args = (0..<half).map { i -> Float in scaled * expf(Float(-log(10_000.0)) * Float(i) / Float(half)) }
    return args.map { cosSin(Double($0)).cos } + args.map { cosSin(Double($0)).sin }
}

/// **bfloat16 as torch rounds it**: round to nearest, ties to even, kept as a Float.
package enum BFloat {
    package static func round(_ x: Float) -> Float {
        guard x.isFinite else { return x }
        var bits = x.bitPattern
        bits &+= 0x7FFF &+ ((bits >> 16) & 1)
        return Float(bitPattern: bits & 0xFFFF_0000)
    }
}
