import Foundation
import Testing
@testable import Siliconed

@Suite struct CapacityModelTests {
    typealias M = CapacityModel
    let gpu = M.Kernel(.gpu, .fp32), amx = M.Kernel(.amx, .fp32), ane = M.Kernel(.ane, .fp16)

    func model(overhead: Double = 0, keep: [M.Engine: Double] = [:]) -> M {
        var m = M()
        m.lines[gpu] = .init(t0: 0.001, rate: 3e12)
        m.lines[amx] = .init(t0: 0.002, rate: 1e12)
        m.lines[ane] = .init(t0: 0.003, rate: 4e12)
        m.sharing["gpu+ane"] = .init(overhead: overhead, keep: keep)
        return m
    }

    @Test func aloneIsTheLine() throws {
        let m = model()
        let s = try #require(m.split(m: 1024, k: 3840, n: 3840, kernels: [gpu]))
        #expect(s.rows == [gpu: 1024])
        #expect(abs(s.seconds - m.lines[gpu]!.seconds(2 * 1024 * 3840 * 3840)) < 1e-12)
    }

    @Test func balancedSidesEndTogetherAndBeatEachAlone() throws {
        let m = model()
        let s = try #require(m.split(m: 4096, k: 3840, n: 10240, kernels: [gpu, amx]))
        #expect(s.rows.values.reduce(0, +) == 4096)
        let w = 2.0 * 4096 * 3840 * 10240
        #expect(s.seconds < m.lines[gpu]!.seconds(w))
        // Closed form T* = (W + Σ t₀R) / ΣR, within a row's worth of time.
        let tStar = (w + 0.001 * 3e12 + 0.002 * 1e12) / 4e12
        #expect(abs(s.seconds - tStar) < 2 * 3840 * 10240 / 1e12)
    }

    @Test func sharingSlowsTheSet() throws {
        let free = try #require(model().split(m: 4096, k: 3840, n: 10240, kernels: [gpu, ane]))
        let shared = try #require(model(overhead: 0.002, keep: [.gpu: 0.7, .ane: 0.8]).split(m: 4096, k: 3840, n: 10240, kernels: [gpu, ane]))
        #expect(shared.seconds > free.seconds)
    }

    @Test func aneRowsComeInUnits() throws {
        let s = try #require(model().split(m: 4096, k: 3840, n: 10240, kernels: [gpu, ane]))
        let rows = s.rows[ane] ?? 0
        #expect(rows % 256 == 0)
        #expect(s.rows.values.reduce(0, +) == 4096)
    }

    @Test func anEngineThatCannotPayItsOverheadLeaves() throws {
        // A tiny product: the ANE's fixed cost alone exceeds the GPU's whole time.
        var m = model()
        m.lines[ane] = .init(t0: 1, rate: 4e12)
        let s = try #require(m.split(m: 1024, k: 512, n: 512, kernels: [gpu, ane]))
        #expect(s.rows == [gpu: 1024])
    }

    @Test func roundTripsThroughJSON() throws {
        var m = model(overhead: 0.001, keep: [.gpu: 0.6])
        m.crest[.gpu] = 4.6e12; m.heat[.amx] = 1.9; m.memoryBandwidth = 2e11
        m.preparation[.int8] = [.widened: 1e-10]
        m.kquantPreparation = 3e-10
        let data = try JSONEncoder().encode(m)
        #expect(try JSONDecoder().decode(M.self, from: data) == m)
    }

    /// Every dictionary keyed by what it holds: a flat array of alternating keys and values is not a report a person reads.
    @Test func isWrittenWithReadableKeys() throws {
        var m = model(overhead: 0.001, keep: [.gpu: 0.6, .ane: 0.7])
        m.crest[.gpu] = 4.6e12; m.heat[.amx] = 1.9
        m.preparation[.bf16] = [.widened: 4e-11, .bf16: 2e-11]
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = try #require(JSONSerialization.jsonObject(with: encoder.encode(m)) as? [String: Any])
        #expect((json["lines"] as? [String: Any])?.keys.sorted() == ["amx/fp32", "ane/fp16", "gpu/fp32"])
        #expect(((json["sharing"] as? [String: Any])?["gpu+ane"] as? [String: Any])?["keep"] as? [String: Double] == ["gpu": 0.6, "ane": 0.7])
        #expect(json["crest"] as? [String: Double] == ["gpu": 4.6e12])
        #expect(json["heat"] as? [String: Double] == ["amx": 1.9])
        #expect(json["preparation"] as? [String: [String: Double]] == ["bf16": ["widened": 4e-11, "bf16": 2e-11]])
        #expect(json["memoryBandwidth"] == nil, "an unread figure is absent")
    }

    @Test func roundedKeepsSixDigitsAndIsIdempotent() {
        var m = model()
        m.lines[gpu] = .init(t0: 0.00029956094692327773, rate: 3527704232234.0015)
        m.sharing["gpu+ane"] = .init(overhead: 1.329099878648668e-05, keep: [.gpu: 0.872517312362546])
        let r = m.rounded()
        #expect(r.lines[gpu] == .init(t0: 0.000299561, rate: 3.5277e12))
        #expect(r.sharing["gpu+ane"]?.keep[.gpu] == 0.872517)
        #expect(r.rounded() == r)
        #expect(r.lines[amx] == m.lines[amx], "figures already short are untouched")
    }
}
