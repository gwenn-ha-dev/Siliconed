import Foundation
import Metal
import Testing
@testable import Siliconed

/// **Qwen-Image-2.1's modulated LayerNorm on the GPU** (`ModulatedLayerNormGPU`) against the CPU's (`Ops.layerNorm` with `scale`): within
/// fp32 against each row's scale, no further from the exact (in double) than the CPU's pass, on rows far from zero (where a sum of squares
/// not centred would lose its digits), and reading its scale at a band's offset. Random tensors, no weights.
@Suite struct ModulatedLayerNormGPUTests {
    func random(_ count: Int, seed: UInt64, scale: Float, offset: Float = 0) -> [Float] {
        var state = seed &* 0x9E3779B97F4A7C15 | 1
        return (0..<count).map { _ in
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return (Float(state >> 40) / Float(1 << 24) * 2 - 1) * scale + offset
        }
    }

    /// The LayerNorm in double, times the scale: the exact the two passes are judged against.
    func exact(_ x: [Float], _ scale: [Float], rows: Int, columns: Int, eps: Float) -> [Double] {
        var out = [Double](repeating: 0, count: x.count)
        for r in 0..<rows {
            let row = (0..<columns).map { Double(x[r * columns + $0]) }
            let mean = row.reduce(0, +) / Double(columns)
            let variance = row.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(columns)
            let inverse = 1 / (variance + Double(eps)).squareRoot()
            for c in 0..<columns { out[r * columns + c] = (row[c] - mean) * inverse * Double(scale[c]) }
        }
        return out
    }

    @Test(arguments: [(rows: 37, columns: 3072, offset: Float(0)), (rows: 19, columns: 3072, offset: 40), (rows: 5, columns: 768, offset: -3)])
    func theGPUsNormIsTheCPUsWithinFP32AndNoFurtherFromTheExact(_ shape: (rows: Int, columns: Int, offset: Float)) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return }
        let norm = try ModulatedLayerNormGPU(device: device, queue: queue)
        let (rows, columns, eps): (Int, Int, Float) = (shape.rows, shape.columns, 1e-6)
        let x = random(rows * columns, seed: 7, scale: 12, offset: shape.offset)
        // The modulation slot of two bands, `[1 + s | …]` each: the kernel reads the second at its offset.
        let band = 3 * columns
        let modulation = random(2 * band, seed: 11, scale: 0.8, offset: 1)
        let scale = Array(modulation[band..<(band + columns)])

        var cpu = [Float](repeating: 0, count: x.count)
        x.withUnsafeBufferPointer { source in
            cpu.withUnsafeMutableBufferPointer {
                Ops.layerNorm(source.baseAddress!, into: $0.baseAddress!, rows: rows, columns: columns, eps: eps, scale: scale)
            }
        }
        let xb = try #require(device.makeBuffer(bytes: x, length: x.count * 4, options: .storageModeShared))
        let mb = try #require(device.makeBuffer(bytes: modulation, length: modulation.count * 4, options: .storageModeShared))
        let ob = try #require(device.makeBuffer(length: x.count * 4, options: .storageModeShared))
        norm.run((xb, 0), scale: (mb, band), into: (ob, 0), rows: rows, columns: columns, eps: eps)
        let gpu = Array(UnsafeBufferPointer(start: ob.contents().assumingMemoryBound(to: Float.self), count: x.count))

        let truth = exact(x, scale, rows: rows, columns: columns, eps: eps)
        var (worstApart, worstGPU, worstCPU) = (0.0, 0.0, 0.0)
        // Each error against its row's root mean square: an element near the mean is a cancellation in either pass (`x − mean`,
        // the mean's own rounding), not a fault of the kernel.
        let rms = (0..<rows).map { r in
            ((r * columns)..<((r + 1) * columns)).reduce(0) { $0 + truth[$1] * truth[$1] } / Double(columns) }.map { $0.squareRoot() }
        for i in x.indices {
            let magnitude = rms[i / columns]
            worstApart = max(worstApart, abs(Double(gpu[i]) - Double(cpu[i])) / magnitude)
            worstGPU = max(worstGPU, abs(Double(gpu[i]) - truth[i]) / magnitude)
            worstCPU = max(worstCPU, abs(Double(cpu[i]) - truth[i]) / magnitude)
        }
        // A row far from zero (mean 40, spread 7) moves by its mean's last bits, a few ulps of 40 against the row's scale.
        #expect(worstApart < 1e-5, "GPU against CPU \(worstApart)")
        // Never further from the exact than the pass it replaces (a margin of half an ulp's worth for the ties of the last rounding).
        #expect(worstGPU <= worstCPU * 1.5 + 1e-7, "GPU \(worstGPU) against CPU \(worstCPU) from the exact")
    }
}
