import Foundation
import Metal
import XCTest
@testable import Siliconed

/// **MOSS-Image's modulated norm on the GPU** (`MossImageNormGPU`), against the CPU's route it replaces where the denoiser streams
/// (`Ops.rmsNorm` then `Ops.modulate`) and against double: within fp32 channel by channel (the sum of squares in another order), never
/// further from the exact than the CPU. The modulation's slices read at an offset in one buffer, as the block reads them. Random
/// tensors, no weights.
final class MossImageNormGPUTests: XCTestCase {
    private var keep: [Pages] = []

    func random(_ count: Int, seed: UInt64, scale: Float) -> [Float] {
        var state = seed &* 0x9E3779B97F4A7C15 | 1
        return (0..<count).map { _ in
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return (Float(state >> 40) / Float(1 << 24) * 2 - 1) * scale
        }
    }

    func testTheGPUsModulatedNormIsTheCPUsWithinFP32() throws {
        let gemm = try GEMM(neuralEngine: false)
        let norm = try MossImageNormGPU(device: gemm.device, queue: gemm.queue)
        let page = Arena.alignment / 4
        func buffer(_ floats: [Float]) throws -> MTLBuffer {
            let count = max((floats.count + page - 1) / page * page, page)
            let pages = Pages(bytes: count * 4)
            let pointer = pages.floats()
            for i in 0..<count { pointer[i] = i < floats.count ? floats[i] : 0 }
            keep.append(pages)
            return try gemm.wrap(UnsafeMutableRawPointer(pointer), bytes: count * 4, name: "t")
        }
        // MOSS's width and a width that is not a multiple of the threadgroup; rows that are not a multiple of anything.
        for (rows, d) in [(37, 3072), (5, 300)] {
            let x = random(rows * d, seed: 1, scale: 20)
            let w = random(d, seed: 2, scale: 2)
            // The modulation as the block holds it: `[shift_a | scale_a | gate_a | shift_f | scale_f | gate_f]`, the feed-forward's read.
            let modulation = random(6 * d, seed: 3, scale: 1.5)
            let (xb, wb, mb, ob) = (try buffer(x), try buffer(w), try buffer(modulation), try buffer([Float](repeating: 0, count: rows * d)))
            norm.normModulate((xb, 0), weight: (wb, 0), scale: (mb, 4 * d), shift: (mb, 3 * d), into: (ob, 0), rows: rows, columns: d,
                              eps: MossImageDiT.eps)
            let got = Array(UnsafeBufferPointer(start: ob.contents().assumingMemoryBound(to: Float.self), count: rows * d))

            var cpu = [Float](repeating: 0, count: rows * d)
            let scale = Array(modulation[(4 * d)..<(5 * d)]), shift = Array(modulation[(3 * d)..<(4 * d)])
            Ops.rmsNorm(x, weight: w, into: &cpu, rows: rows, columns: d, eps: MossImageDiT.eps)
            Ops.modulate(&cpu, scale: scale, shift: shift, rows: rows, columns: d)
            let e = cpu.withUnsafeBufferPointer { channelError(got, $0, columns: d) }
            XCTAssertLessThan(e.worst, 1e-6, "\(rows)×\(d): \(e.worst) at channel \(e.channel)")

            // Against double, channel by channel (`channelError`): the GPU's route no further from the exact than the CPU's + 25 %.
            var exact = [Double](repeating: 0, count: rows * d)
            for r in 0..<rows {
                var ss = 0.0
                for j in 0..<d { ss += Double(x[r * d + j]) * Double(x[r * d + j]) }
                let rs = 1 / (ss / Double(d) + Double(MossImageDiT.eps)).squareRoot()
                for j in 0..<d { exact[r * d + j] = Double(x[r * d + j]) * rs * Double(w[j]) * (1 + Double(scale[j])) + Double(shift[j]) }
            }
            let (gpuExact, cpuExact) = exact.withUnsafeBufferPointer { (channelError(got, $0, columns: d), channelError(cpu, $0, columns: d)) }
            XCTAssertLessThan(gpuExact.worst, max(1.25 * cpuExact.worst, 1e-7), "\(rows)×\(d): GPU \(gpuExact.worst), CPU \(cpuExact.worst)")
        }
    }
}
