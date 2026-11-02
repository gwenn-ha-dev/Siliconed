import Testing
@testable import Verification

/// The engine's summation-order draws take the oracle's permutation to the bit (`krea2.py`'s `residual_permutation`):
/// the values below are the Python definition's own output.
struct ResidualPermutationTests {
    @Test func splitMixMatchesTheReference() {
        var state: UInt64 = 0
        #expect(ResidualPermutation.splitMix64(&state) == 0xE220_A839_7B1D_CDAF)
    }

    @Test func fisherYatesMatchesTheOracle() {
        #expect(ResidualPermutation.fisherYates(24, seed: 1)
                == [7, 20, 21, 12, 23, 22, 11, 3, 5, 16, 18, 15, 4, 19, 10, 8, 6, 9, 13, 1, 14, 0, 2, 17])
        #expect(ResidualPermutation.fisherYates(24, seed: 8)
                == [8, 19, 20, 0, 7, 14, 6, 5, 15, 4, 17, 12, 9, 23, 18, 3, 21, 11, 10, 2, 16, 1, 13, 22])
        #expect(Array(ResidualPermutation.fisherYates(6144, seed: 1).prefix(8)) == [3648, 3207, 521, 4425, 2727, 3134, 921, 2089])
    }

    @Test func groupsMoveWholeAndSeedZeroIsTheIdentity() {
        let perm = ResidualPermutation.residual(dim: 6144, seed: 3)
        #expect(ResidualPermutation.movesWholeGroups(perm, group: 256))
        #expect(Array(perm.prefix(2)) == [16 * 256, 16 * 256 + 1])
        #expect(!ResidualPermutation.movesWholeGroups(ResidualPermutation.residual(dim: 6144, seed: 3, group: nil), group: 256))
        #expect(ResidualPermutation.residual(dim: 512, seed: 0) == Array(0..<512))
    }

    @Test func restoredUndoesPermuted() {
        let perm = ResidualPermutation.residual(dim: 8, seed: 5, group: nil)
        let x: [Float] = (0..<24).map(Float.init)
        let moved = x.withUnsafeBufferPointer { ResidualPermutation.permuted($0, perm) }
        #expect(moved[0] == Float(perm[0]) && moved[8 + 3] == Float(8 + perm[3]))
        #expect(ResidualPermutation.restored(moved, perm) == x)
    }
}
