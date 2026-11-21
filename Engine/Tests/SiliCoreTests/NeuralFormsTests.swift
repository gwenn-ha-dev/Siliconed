import SiliCore
import Testing

/// A gate's factor: the least power of two at or past its largest magnitude, within f16's normal exponents.
@Test func aGatesFactorIsItsPowerOfTwo() {
    #expect(Gate.factor(largest: 0) == 1)
    #expect(Gate.factor(largest: 0.5) == 0.5)
    #expect(Gate.factor(largest: 8.75) == 16)
    #expect(Gate.factor(largest: 30.5) == 32)
    #expect(Gate.factor(largest: 1e-9) == 0x1p-14)
    #expect(Gate.factor(largest: 1e9) == 0x1p15)
}

/// A row's exponent on the CPU, by each of the graph's two rules: they part only at an exact power of two past 1
/// (½ rounded away from zero), where `ranged` puts the largest at 2^(e₀−1) and `ranged16` at 2^e₀; each within its
/// factor's exponents.
@Test func aRowsExponentFollowsTheGraphsTwoRules() {
    #expect(RowExponent.ranged(largest: 0x1p14, f32Factor: false) == 1)
    #expect(RowExponent.ranged16(largest: 0x1p14) == 0)
    #expect(RowExponent.ranged(largest: 3000, f32Factor: false) == RowExponent.ranged16(largest: 3000))
    #expect(RowExponent.ranged(largest: 0x1p-14, f32Factor: false) == -13)
    #expect(RowExponent.ranged(largest: 0x1p-14, f32Factor: true) == -28)
    #expect(RowExponent.ranged(largest: 0x1p-3, f32Factor: true) == -17)
    #expect(RowExponent.ranged(largest: 1e30, f32Factor: false) == 15)
    #expect(RowExponent.ranged16(largest: 1e30) == 15)
}
