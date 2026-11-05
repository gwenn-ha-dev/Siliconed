import Foundation
import Metal
import Testing
@testable import Siliconed

/// **Krea 2's modulated norm on the GPU** (`ModulatedNormGPU`) against the CPU's (`Ops.rmsNorm` then `Ops.modulate`): within fp32
/// channel by channel on drawn rows, and **to the bit** where the sum of squares is exact in any order (values on a coarse grid), which
/// proves that everything after the sum is the CPU's operations in its order. Random tensors, no weights.
@Suite struct ModulatedNormGPUTests {
    func random(_ count: Int, seed: UInt64, scale: Float) -> [Float] {
        var state = seed &* 0x9E3779B97F4A7C15 | 1
        return (0..<count).map { _ in
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return (Float(state >> 40) / Float(1 << 24) * 2 - 1) * scale
        }
    }

    /// The CPU's route, as `Krea2DiT.modulatedNorm` calls it.
    func cpu(_ x: [Float], w: [Float], scale: [Float], shift: [Float], rows: Int, columns: Int, eps: Float) -> [Float] {
        var out = [Float](repeating: 0, count: rows * columns)
        out.withUnsafeMutableBufferPointer { o in
            Ops.rmsNorm(x, weight: w, into: o.baseAddress!, rows: rows, columns: columns, eps: eps)
            Ops.modulate(o.baseAddress!, scale: scale, shift: shift, rows: rows, columns: columns)
        }
        return out
    }

    /// The GPU's: `x`, the bank `[w | 1 + scale]` and the shift in page-aligned pages, as the DiT lays them out.
    func gpu(_ x: [Float], w: [Float], scale: [Float], shift: [Float], rows: Int, columns: Int, eps: Float) throws -> [Float]? {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return nil }
        let kernel = try ModulatedNormGPU(device: device, queue: queue)
        func pages(_ values: [Float]) throws -> (Pages, MTLBuffer) {
            let p = Pages(bytes: values.count * 4)
            p.floats().update(from: values, count: values.count)
            let b = try #require(device.makeBuffer(bytesNoCopy: p.base, length: p.capacity, options: .storageModeShared, deallocator: nil))
            return (p, b)
        }
        let factor = scale.map { $0 + 1 }                                         // one rounding, as `Ops.modulate`'s `vDSP_vadd`
        let (xp, xb) = try pages(x)
        let (bp, bb) = try pages([Float](repeating: 0, count: 3) + w + factor)   // offsets that are not a page's start
        let (sp, sb) = try pages(shift)
        let (op, ob) = try pages([Float](repeating: .nan, count: rows * columns))
        kernel.run((xb, 0), weight: (bb, 3), factor: (bb, 3 + columns), shift: (sb, 0), into: (ob, 0), rows: rows, columns: columns, eps: eps)
        return withExtendedLifetime((xp, bp, sp)) { Array(UnsafeBufferPointer(start: op.floats(), count: rows * columns)) }
    }

    @Test(arguments: [(6144, 37), (2560, 12), (1000, 5)])
    func theGPUsNormIsTheCPUsWithinFP32(_ shape: (columns: Int, rows: Int)) throws {
        let (columns, rows) = shape
        let x = random(rows * columns, seed: 1, scale: 30)
        let (w, scale, shift) = (random(columns, seed: 2, scale: 2), random(columns, seed: 3, scale: 1.5), random(columns, seed: 4, scale: 3))
        guard let got = try gpu(x, w: w, scale: scale, shift: shift, rows: rows, columns: columns, eps: 1e-6) else { return }
        let expected = cpu(x, w: w, scale: scale, shift: shift, rows: rows, columns: columns, eps: 1e-6)
        let e = expected.withUnsafeBufferPointer { channelError(got, $0, columns: columns) }
        #expect(e.worst < 1e-6, "\(columns)×\(rows): \(e.worst) at channel \(e.channel)")
    }

    /// Values on a grid of quarters up to 4: every square and every partial sum is exact, so the two sums agree to the bit whatever their
    /// order, and so must everything after them.
    @Test func whereTheSumIsExactTheGPUsNormIsTheCPUsToTheBit() throws {
        let (columns, rows) = (6144, 9)
        let x = random(rows * columns, seed: 5, scale: 4).map { ($0 * 4).rounded() / 4 }
        let (w, scale, shift) = (random(columns, seed: 6, scale: 2), random(columns, seed: 7, scale: 1.5), random(columns, seed: 8, scale: 3))
        guard let got = try gpu(x, w: w, scale: scale, shift: shift, rows: rows, columns: columns, eps: 1e-5) else { return }
        let expected = cpu(x, w: w, scale: scale, shift: shift, rows: rows, columns: columns, eps: 1e-5)
        #expect(got.map(\.bitPattern) == expected.map(\.bitPattern))
    }
}
