import Foundation
import Metal
import Testing
@testable import Siliconed

/// **Anima's residual and modulated LayerNorm on the GPU** (`AnimaElementwiseGPU`, the stream's) against the CPU's (`Ops.gatedResidual`,
/// `AnimaOps.modulate`): the residual to the bit (the same product and sum, no contraction), the modulation within fp32 channel by channel
/// and no farther from the fp64 LayerNorm than the CPU's; either half alone or both fused. Random rows, no weights.
@Suite struct AnimaElementwiseGPUTests {
    func random(_ count: Int, seed: UInt64, scale: Float, offset: Float = 0) -> [Float] {
        var state = seed &* 0x9E3779B97F4A7C15 | 1
        return (0..<count).map { _ in
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return (Float(state >> 40) / Float(1 << 24) * 2 - 1) * scale + offset
        }
    }

    /// The modulation in double from fp32 inputs: the witness both sides are measured against.
    func witness(_ x: [Float], shift: [Float], scale: [Float], rows: Int, columns: Int) -> [Double] {
        var out = [Double](repeating: 0, count: x.count)
        for r in 0..<rows {
            let row = (0..<columns).map { Double(x[r * columns + $0]) }
            let mean = row.reduce(0, +) / Double(columns)
            let variance = row.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(columns)
            let inverse = 1 / (variance + 1e-6).squareRoot()
            for c in 0..<columns {
                out[r * columns + c] = (row[c] - mean) * inverse * (1 + Double(scale[c])) + Double(shift[c])
            }
        }
        return out
    }

    /// `rows` rows of `columns`: Anima's width and one that is not a multiple of the threadgroup; residuals near 0, and climbing to 10⁵
    /// off a large mean (block 13's are 2.5·10⁵).
    @Test(arguments: [(37, 2048, Float(4), Float(0)), (19, 1000, 3, 0.5), (11, 2048, 1e5, 2e5)])
    func theResidualIsTheCPUsToTheBitAndTheModulationWithinFP32(_ shape: (rows: Int, columns: Int, scale: Float, offset: Float)) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return }
        let gpu = try AnimaElementwiseGPU(device: device, queue: queue)
        let (rows, columns) = (shape.rows, shape.columns)
        let n = rows * columns
        let x = random(n, seed: 1, scale: shape.scale, offset: shape.offset), y = random(n, seed: 2, scale: shape.scale)
        let gate = random(columns, seed: 3, scale: 1.5), shift = random(columns, seed: 4, scale: 0.8)
        let scale = random(columns, seed: 5, scale: 0.9)

        // x, y, out, and the bank [gate | shift | scale], each on its pages
        let pages = (0..<3).map { _ in Pages(bytes: n * 4) }, bank = Pages(bytes: 3 * columns * 4)
        func wrap(_ p: Pages) throws -> MTLBuffer {
            try #require(device.makeBuffer(bytesNoCopy: p.base, length: p.capacity, options: .storageModeShared, deallocator: nil))
        }
        let xb = try wrap(pages[0]), yb = try wrap(pages[1]), ob = try wrap(pages[2]), bb = try wrap(bank)
        typealias S = AnimaElementwiseGPU.Slice
        let x0: S = (xb, 0), y0: S = (yb, 0), out0: S = (ob, 0), gate0: S = (bb, 0), shift0: S = (bb, columns), scale0: S = (bb, 2 * columns)
        pages[1].floats().update(from: y, count: n)
        bank.floats().update(from: gate, count: columns)
        bank.floats(at: columns * 4).update(from: shift, count: columns)
        bank.floats(at: 2 * columns * 4).update(from: scale, count: columns)
        func read(_ p: Pages) -> [Float] { Array(UnsafeBufferPointer(start: p.floats(), count: n)) }

        // the CPU's residual, then its modulation of the sum
        var added = x
        added.withUnsafeMutableBufferPointer { Ops.gatedResidual($0.baseAddress!, plus: y, gate: gate, rows: rows, columns: columns) }
        var cpu = [Float](repeating: 0, count: n), one = [Float](repeating: 0, count: columns)
        cpu.withUnsafeMutableBufferPointer {
            AnimaOps.modulate(added, into: $0.baseAddress!, shift: shift, scale: scale, scaleOne: &one, rows: rows, columns: columns)
        }
        let exact = witness(added, shift: shift, scale: scale, rows: rows, columns: columns)
        let cpuError = exact.withUnsafeBufferPointer { channelError(cpu, $0, columns: columns) }

        // 1. the residual alone, to the bit
        pages[0].floats().update(from: x, count: n)
        gpu.run(x0, residual: (y: y0, gate: gate0), modulate: nil, rows: rows, columns: columns, eps: 1e-6)
        #expect(read(pages[0]).map(\.bitPattern) == added.map(\.bitPattern), "residual")

        // 2. the modulation alone, of the sum, and 3. both fused from `x`: the same values
        for fused in [false, true] {
            pages[0].floats().update(from: fused ? x : added, count: n)
            pages[2].floats().update(repeating: .nan, count: n)
            var residual: (y: S, gate: S)?
            if fused { residual = (y: y0, gate: gate0) }
            gpu.run(x0, residual: residual, modulate: (shift: shift0, scale: scale0, out: out0), rows: rows, columns: columns, eps: 1e-6)
            #expect(read(pages[0]).map(\.bitPattern) == added.map(\.bitPattern), "x, fused \(fused)")
            let got = read(pages[2])
            let againstCPU = cpu.withUnsafeBufferPointer { channelError(got, $0, columns: columns) }
            let againstExact = exact.withUnsafeBufferPointer { channelError(got, $0, columns: columns) }
            #expect(againstCPU.worst < 1e-6, "fused \(fused): \(againstCPU.worst) from the CPU at channel \(againstCPU.channel)")
            // No farther from the fp64 LayerNorm than the CPU, within the rule's 25 % (an ulp's noise on a near-exact path).
            #expect(againstExact.worst <= max(1.25 * cpuError.worst, 1e-7),
                    "fused \(fused): \(againstExact.worst) from fp64 against the CPU's \(cpuError.worst)")
        }
    }
}
