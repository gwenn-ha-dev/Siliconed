import Foundation
import Testing
@testable import Siliconed

/// **The GPU's wake** (`CapacityProbesWake.swift`): read from its probe's medians, kept by the model, paid by a self-attention and never by a
/// cross-attention, the same for every candidate. Synthetic numbers, no GPU.
@Suite struct CapacityWakeTests {
    typealias M = CapacityModel
    typealias S = CapacityFit.Measurements.Solo

    func probe(_ medians: [(idleMicroseconds: Int, ms: Double)]) -> [S] {
        medians.map { S(kernel: "gpu/tiles16", shape: [1024, 24, 128, $0.idleMicroseconds], flop: 0, seconds: $0.ms * 1e-3) }
    }

    @Test func theWakeIsTheStepsPlateauOverTheCallBackToBack() throws {
        // One M5 Pro's shape: nothing at 0.5 and 1 ms of idle, 0.85 ms from 2 ms on.
        let w = try #require(CapacityFit.wake(probe([(0, 1.08), (500, 1.11), (1000, 1.09), (2000, 2.00), (4000, 1.97), (8000, 1.91)])))
        #expect(abs(w.seconds - 0.86e-3) < 1e-6)
        #expect(w.idle == 0.002)
    }

    @Test func noStepNoWakeAndNoProbeNoAnswer() throws {
        let flat = try #require(CapacityFit.wake(probe([(0, 1.08), (500, 1.07), (1000, 1.08), (2000, 1.06), (4000, 1.07), (8000, 1.06)])))
        #expect(flat.seconds == 0 && flat.idle == 0)
        #expect(CapacityFit.wake(nil) == nil)
        #expect(CapacityFit.wake(probe([(0, 1.08), (8000, 1.9)])) == nil)   // one idle: no plateau to read
    }

    @Test func theModelKeepsItsWakeThroughItsText() throws {
        var m = M()
        m.lines[M.Kernel(.gpu, .fp32)] = .init(t0: 0, rate: 3e12)
        m.wake = 0.00085; m.wakeIdle = 0.002
        let back = try JSONDecoder().decode(M.self, from: JSONEncoder().encode(m))
        #expect(back.wake == 0.00085 && back.wakeIdle == 0.002)
        // a map without it reads as before
        var none = m; none.wake = nil; none.wakeIdle = nil
        #expect(try JSONDecoder().decode(M.self, from: JSONEncoder().encode(none)).wake == nil)
    }

    @Test func aSelfAttentionPaysTheWakeACrossAttentionDoesNot() throws {
        var m = M()
        m.lines[M.Kernel(.gpu, .fp32)] = .init(t0: 0, rate: 3e12)
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        let c = CapacitySolver.Constraints(sustained: false, allowsANE: false)
        let own = AttentionKey(sequence: 4096, heads: 16, headDim: 128)
        let cross = AttentionKey(sequence: 4096, keys: 512, heads: 16, headDim: 128, cross: true)
        let before = (try #require(CapacitySolver.attention(own, flash: false, model: m, constraints: c)),
                      try #require(CapacitySolver.attention(cross, flash: false, model: m, constraints: c)))
        m.wake = 0.00085
        let after = (try #require(CapacitySolver.attention(own, flash: false, model: m, constraints: c)),
                     try #require(CapacitySolver.attention(cross, flash: false, model: m, constraints: c)))
        #expect(abs(after.0.compute - before.0.compute - 0.00085) < 1e-12)
        #expect(after.0.assignment == before.0.assignment)
        #expect(after.0.busy[.gpu]! - before.0.busy[.gpu]! > 0.00084)
        #expect(after.1.compute == before.1.compute && after.1.assignment == before.1.assignment)
    }
}
