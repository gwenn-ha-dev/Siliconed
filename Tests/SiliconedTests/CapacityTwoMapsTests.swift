import Foundation
import Testing
@testable import Siliconed

/// **Two maps of one Mac, one plan: no discrete switch under the error.** A fitted term that crosses zero keeps the line it belongs to (the
/// thin line's share of the bandwidth, the AMX's fixed cost), the GPU's thin lines read one fixed cost, two GPU forms that pay it are told
/// apart to the wide margin, and the GPU's wake under a cut by heads lates the GPU's side alone. Synthetic numbers: no GPU, no weights.
@Suite struct CapacityTwoMapsTests {
    typealias M = CapacityModel
    typealias F = CapacityFit
    typealias CS = CapacitySolver

    func thinSolos(_ kernel: M.Kernel, t0: Double, share: Double, arithmetic: M.Line, bandwidth: Double,
                   bias: (Int) -> Double = { _ in 0 }) -> [F.Measurements.Solo] {
        let (wb, ab) = M.streamedBytes(kernel.form)
        return Diagnostic.CapacityProbes.Sizes.full.thin.enumerated().map { i, s in
            let rows = Double(M.computedRows(kernel, rows: s.m))
            let stream = (wb * Double(s.k * s.n) + ab * rows * Double(s.k + s.n)) / bandwidth
            let t = t0 + share * stream + 2 * rows * Double(s.k * s.n) / arithmetic.rate + bias(i) * stream
            return .init(kernel: kernel.description, shape: [s.m, s.k, s.n], flop: 2 * Double(s.m * s.k * s.n), seconds: t)
        }
    }

    // MARK: A term at zero keeps its line

    @Test func aThinShareThatFitsUnderZeroIsAShareOfZeroNotAMissingLine() throws {
        // Every thin solo the fixed cost and the arithmetic, a hair under on the shapes that stream the most: the free share is negative.
        // The line was dropped there (the kernel then timed by its wide line, t₀ 7× smaller); it is kept, its share zero.
        let kernel = M.Kernel(.gpu, .bf16act), arithmetic = M.Line(t0: 1e-4, rate: 2e13), bandwidth = 2.2e11
        let solos = thinSolos(kernel, t0: 1.6e-4, share: 0, arithmetic: arithmetic, bandwidth: bandwidth, bias: { $0 % 2 == 0 ? -0.03 : 0 })
        let line = try #require(F.thinLine(kernel: kernel, solos: solos, arithmetic: arithmetic, bandwidth: bandwidth))
        #expect(line.perWeight == 0 && line.perRow == 0)
        #expect(abs(line.t0 / 1.6e-4 - 1) < 0.05, "\(line.t0)")
        // the same solos a hair over: the share small and positive, the line close to the one at zero — continuous
        let over = try #require(F.thinLine(kernel: kernel, solos: thinSolos(kernel, t0: 1.6e-4, share: 0.01, arithmetic: arithmetic, bandwidth: bandwidth),
                                           arithmetic: arithmetic, bandwidth: bandwidth))
        #expect(abs(over.seconds(rows: 64, k: 2560, n: 1024) / line.seconds(rows: 64, k: 2560, n: 1024) - 1) < 0.05)
    }

    @Test func anAMXFixedCostUnderZeroIsZeroAndItsKnotsStay() throws {
        // The AMX's rate falling with k, its solos a hair faster than no fixed cost would allow: the free fit's t₀ is negative. The knots
        // were dropped for the plain line there (t₀ 0.20 ms in one map, 0.68 in the next); they stay, t₀ zero.
        let kernel = M.Kernel(.amx, .fp32)
        let rate: [Int: Double] = [2048: 1.80e12, 3840: 1.83e12, 8192: 1.89e12, 12288: 1.85e12]
        let solos = [(1024, 2048, 2048), (2048, 3840, 3840), (1024, 8192, 1536), (1024, 12288, 4096), (512, 3840, 3840)].map { (m, k, n) -> F.Measurements.Solo in
            let flop = 2 * Double(m * k * n)
            return .init(kernel: kernel.description, shape: [m, k, n], flop: flop, seconds: flop / rate[k]! - 2e-5)
        }
        let line = try #require(F.productLine(kernel: kernel, solos: solos))
        #expect(line.t0 == 0)
        #expect(line.rateByK?.count == 4)
    }

    // MARK: The GPU's thin lines on one fixed cost

    @Test func theGPUsThinLinesReadOneFixedCost() throws {
        let bandwidth = 2.2e11
        let fp32 = M.Kernel(.gpu, .fp32), bf16 = M.Kernel(.gpu, .bf16), amx = M.Kernel(.amx, .fp32)
        let lines: [M.Kernel: M.Line] = [fp32: .init(t0: 6e-5, rate: 5.9e12), bf16: .init(t0: 1.8e-4, rate: 6.4e12), amx: .init(t0: 2e-4, rate: 1.8e12)]
        // one fixed cost, 0.15 ms, each form its own share; the AMX its own fixed cost
        let groups: [M.Kernel: [F.Measurements.Solo]] = [
            fp32: thinSolos(fp32, t0: 1.5e-4, share: 0.2, arithmetic: lines[fp32]!, bandwidth: bandwidth),
            bf16: thinSolos(bf16, t0: 1.5e-4, share: 0.1, arithmetic: lines[bf16]!, bandwidth: bandwidth),
            amx: thinSolos(amx, t0: 3e-5, share: 1.0, arithmetic: lines[amx]!, bandwidth: bandwidth),
        ]
        let thin = F.thinLines(groups, lines: lines, bandwidth: bandwidth)
        let (a, b, x) = (try #require(thin[fp32]), try #require(thin[bf16]), try #require(thin[amx]))
        #expect(a.t0 == b.t0 && abs(a.t0 / 1.5e-4 - 1) < 1e-6)
        #expect(abs(a.perWeight! / (0.2 * 4 / bandwidth) - 1) < 1e-6 && abs(b.perWeight! / (0.1 * 2 / bandwidth) - 1) < 1e-6)
        #expect(abs(x.t0 / 3e-5 - 1) < 1e-6, "an engine other than the GPU keeps its own")
        // two forms whose own fits disagree on the fixed cost (0.17 and 0.13 ms): one, between them
        let apart = F.thinLines([fp32: thinSolos(fp32, t0: 1.7e-4, share: 0.2, arithmetic: lines[fp32]!, bandwidth: bandwidth),
                                 bf16: thinSolos(bf16, t0: 1.3e-4, share: 0.1, arithmetic: lines[bf16]!, bandwidth: bandwidth)],
                                lines: lines, bandwidth: bandwidth)
        let (c, d) = (try #require(apart[fp32]), try #require(apart[bf16]))
        #expect(c.t0 == d.t0 && c.t0 > 1.3e-4 && c.t0 < 1.7e-4, "\(c.t0)")
        // and the map's model fits them so
        var m = F.Measurements()
        m.memoryBandwidth = bandwidth
        m.thin = groups[fp32]! + groups[bf16]!
        m.solos = [fp32, bf16].flatMap { k in [(2048, 3840, 3840), (1024, 3840, 3840), (4096, 6144, 6144)].map { (r, kk, n) -> F.Measurements.Solo in
            .init(kernel: k.description, shape: [r, kk, n], flop: 2 * Double(r * kk * n), seconds: lines[k]!.seconds(rows: r, k: kk, n: n))
        } }
        let model = F.model(from: m)
        #expect(model.thin[fp32]?.t0 == model.thin[bf16]?.t0)
    }

    /// **Two GPU forms of a thin product, one fixed cost: their difference to the wide margin.** A text encoder's `2560x1024` at 19 rows, the
    /// weight as published 5.4 % faster than the widened one (no widening paid, half the bytes): under the thin margin (15 %, the fixed costs
    /// two maps read apart) the written order gave it to the widened weight, and a map that read the gap 15.5 % to the bf16 one. With one
    /// fixed cost, it is no part of the difference: past the wide margin (5 %), the faster form; under it, the written order still.
    @Test func twoGPUFormsOfAThinProductAreToldApartToTheWideMargin() throws {
        func machine(bf16Weight: Double, bf16FixedCost: Double = 1e-4) -> M {
            var m = M()
            let fp32 = M.Kernel(.gpu, .fp32), bf16 = M.Kernel(.gpu, .bf16)
            m.lines[fp32] = .init(t0: 6e-5, rate: 6e12)
            m.lines[bf16] = .init(t0: 1.8e-4, rate: 6e12)
            m.thin[fp32] = .init(t0: 1e-4, rate: 6e12, perWeight: 4e-12, perRow: 4e-12)
            m.thin[bf16] = .init(t0: bf16FixedCost, rate: 6e12, perWeight: bf16Weight, perRow: 2e-12)
            m.preparation[.bf16] = [.widened: 0, .bf16: 0]
            m.noise = 0.003
            return m
        }
        let key = ProductKey(k: 2560, n: 1024, m: 19, weight: .bf16, phase: .encoding)
        let c = CS.Constraints(sustained: false, allowsANE: false, allowsRounding: false)
        func route(_ m: M) throws -> String { try #require(CS.product(key, model: m, constraints: c)).route }
        func gap(_ m: M) throws -> Double {
            let all = CS.candidates(key, model: m, constraints: c)
            let widened = try #require(all.first { $0.route == "gpu" }).seconds, bf16 = try #require(all.first { $0.route == "gpu/bf16" }).seconds
            return widened / bf16 - 1
        }
        let far = machine(bf16Weight: 1e-12), near = machine(bf16Weight: 3.6e-12)
        let (farGap, nearGap) = (try gap(far), try gap(near))
        #expect(farGap > 0.05 && farGap < 0.15, "\(farGap)")
        #expect(try route(far) == "gpu/bf16")
        #expect(nearGap > 0 && nearGap < 0.05, "\(nearGap)")
        #expect(try route(near) == "gpu", "under the wide margin, the written order")
        // two fixed costs (a map whose forms were fitted apart): the thin margin, the written order
        #expect(try route(machine(bf16Weight: 1e-12, bf16FixedCost: 0.99e-4)) == "gpu")
    }

    // MARK: The wake under a cut by heads

    /// **The wake lates the GPU's side, not the call** (a cut by heads: the CPU's heads start as the call does). Where the CPU's side ends
    /// past the GPU's end and its wake, the call pays nothing; where the GPU's side is the last, the whole wake; alone on the GPU, the wake.
    @Test func underACutByHeadsTheWakeLatesTheGPUsSideAlone() throws {
        var m = M()
        m.lines[M.Kernel(.gpu, .fp32)] = .init(t0: 0, rate: 3e12)
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 2e12)
        m.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        m.sharing[M.key([M.Engine.gpu, .cpu])] = .init(overhead: 0, keep: [.gpu: 1, .cpu: 1])
        let c = CS.Constraints(sustained: false, allowsANE: false)
        let key = AttentionKey(sequence: 4096, heads: 16, headDim: 128)
        func compute(_ cpuHeads: Int, wake: Double?) throws -> Double {
            var w = m; w.wake = wake
            let a = cpuHeads == 0 ? AttentionAssignment(kernel: .graph) : AttentionAssignment(kernel: .split, cpuHeads: cpuHeads, gpuKernel: .graph)
            return try #require(CS.attention(assigned: a, key: key, model: w, constraints: c)).compute
        }
        let wake = 0.00085
        // the CPU's heads at a quarter of the GPU's speed: 8 heads each side, the CPU's side four times as long — the wake hidden under it
        #expect(abs(try compute(8, wake: wake) - compute(8, wake: nil)) < 1e-12)
        // one head on the CPU: the GPU's side is the last, the whole wake
        #expect(abs(try compute(1, wake: wake) - compute(1, wake: nil) - wake) < 1e-12)
        #expect(abs(try compute(0, wake: wake) - compute(0, wake: nil) - wake) < 1e-12)
    }
}
