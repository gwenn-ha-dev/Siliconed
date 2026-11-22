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

/// A row of zeros has a factor, the graph's floor of 2⁻¹⁴ (log₂ 0 is −∞, which no `Int` holds); a row is split into its
/// factor and its values over it, the largest under 2^e₀.
@Test func aRowOfZerosHasAFactorAndARowSplits() {
    #expect(RowExponent.ranged(largest: 0, f32Factor: false) == RowExponent.ranged(largest: 0x1p-14, f32Factor: false))
    #expect(RowExponent.ranged(largest: 0, f32Factor: true) == -28)
    #expect(RowExponent.ranged16(largest: 0) == -13)
    let zeros = RowExponent.split([Float](repeating: 0, count: 4), .ranged16)
    #expect(zeros.factor == 0x1p-13 && zeros.unit == [0, 0, 0, 0])
    let row = RowExponent.split([3000, -1, 0.5], .ranged(f32Factor: true))
    #expect(row.factor == Float(sign: .plus, exponent: RowExponent.ranged(largest: 3000, f32Factor: true), significand: 1))
    #expect(row.unit == [3000 / row.factor, -1 / row.factor, 0.5 / row.factor])
    #expect(row.unit.map(abs).max()! <= 0x1p14 && row.unit.map(abs).max()! > 0x1p13)
}
