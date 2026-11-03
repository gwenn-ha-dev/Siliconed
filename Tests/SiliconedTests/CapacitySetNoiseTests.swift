import Foundation
import Testing
@testable import Siliconed

/// **Two maps, one plan**: a short call read on its fast mode (`CapacityProbes.steady`), a set's figure from the median of three chains
/// where two disagreed and its error from their MAD (`CapacityFit.chains`, `needsAgain`), and a set taken only where its gain passes
/// twice the error of the difference (`CapacitySolver`'s `passingTheirError`). Pure: no GPU, no weights.
@Suite struct CapacitySetNoiseTests {
    typealias M = CapacityModel
    typealias CS = CapacitySolver
    typealias P = Diagnostic.CapacityProbes
    let gpu = M.Kernel(.gpu, .fp32), ane = M.Kernel(.ane, .fp16)

    // MARK: A short call, its fast mode

    @Test func aShortFigureReadsFifteenCallsAndALongOneWhatItDid() {
        #expect(P.steadyCalls(0.0003) == 15 && P.steadyCalls(0.001) == 15)
        #expect(P.steadyCalls(0.0015) == 13)
        // from 5 ms up, four calls and their median, as before
        #expect(P.steadyCalls(0.005) == 4 && P.steadyCalls(0.02) == 4)
        #expect(P.steadyCalls(0.05) == 2 && P.steadyCalls(0.2) == 1)
        let four = [0.0101, 0.0100, 0.0103, 0.0102]
        #expect(P.figure(four) == CapacityFit.median(four))
    }

    @Test func halfTheCallsLateReadsTheFastMode() {
        // An M5 Pro's thin product: 0.26 ms of work, and one call in two whose completion reaches the CPU 0.7 ms late.
        let calls: [Double] = (0..<15).map { (i: Int) -> Double in
            let base: Double = i % 2 == 0 ? 0.00026 : 0.00096
            return base + 1e-6 * Double(i)
        }
        #expect(P.figure(calls) < 0.00028, "\(P.figure(calls))")
        // the median of the first four, the old figure, fell between the two modes
        #expect(CapacityFit.median(Array(calls.prefix(4))) > 0.0006)
    }

    // MARK: A set's chains: the median of three, its error from the MAD

    func pairModel(mapNoise: Double? = nil) -> M {
        var m = M()
        m.lines[gpu] = .init(t0: 0, rate: 3e12)
        m.lines[ane] = .init(t0: 0, rate: 1e12)
        m.sharing["gpu+ane"] = .init(overhead: 0, keep: [.gpu: 1, .ane: 1])
        m.noise = mapNoise
        return m
    }

    func trial(_ pass: Int, wall: Double, shape: [Int] = [4096, 3840, 3840]) -> CapacityFit.Measurements.Trial {
        .init(set: "gpu+ane", shape: shape, pass: pass,
              sides: [.init(kernel: "gpu/fp32", rows: 3072, solo: wall, within: wall), .init(kernel: "ane/fp16", rows: 1024, solo: wall, within: wall)],
              wall: wall)
    }

    let long = [4096, 12288, 4096]
    var again: Int { CapacityFit.Measurements.Trial.again }

    @Test func aSingleDisturbedChainMovesNeitherTheFigureNorItsError() throws {
        // The integers' rounded pair under one map of an M5 Pro: 7.01 then 5.32 ms at the same rows; a third chain, 5.30.
        let chosen = CapacityFit.chains(set: "gpu+ane", trials: [trial(1, wall: 7.01e-3), trial(2, wall: 5.32e-3), trial(again, wall: 5.30e-3),
                                                                 trial(3, wall: 21.5e-3, shape: long)], model: pairModel())
        let short = try #require(chosen.used.first { $0.shape != long })
        #expect(short.wall == 5.32e-3, "the median chain, not the disturbed one")
        #expect(chosen.used.count == 2)
        let noise = try #require(chosen.noise)
        // MAD 0.02 ms over 5.32: σ 0.56 %, the median of three's error 0.40 % — the 7.01 ms chain weighs nothing
        #expect(abs(noise - 1.2533 * 1.4826 * (0.02 / 5.32) / 3.0.squareRoot()) < 1e-6, "\(noise)")
    }

    @Test func threeChainsApartGiveTheirSpread() throws {
        // The rounded bf16 pair of one map: 6.19 and 7.21 ms, a third between them.
        let noise = try #require(CapacityFit.chains(set: "gpu+ane", trials: [trial(1, wall: 6.19e-3), trial(2, wall: 7.21e-3), trial(again, wall: 6.70e-3)],
                                                    model: pairModel()).noise)
        #expect(abs(noise - 1.2533 * 1.4826 * (0.51 / 6.70) / 3.0.squareRoot()) < 1e-6, "\(noise)")
    }

    @Test func twoChainsCarryTheMapsError() {
        // two chains are one degree of freedom: no error of the set's own, the second pass read as before
        let chosen = CapacityFit.chains(set: "gpu+ane", trials: [trial(1, wall: 6.54e-3), trial(2, wall: 5.79e-3), trial(3, wall: 21.5e-3, shape: long)],
                                        model: pairModel())
        #expect(chosen.noise == nil)
        #expect(Set(chosen.used.map(\.pass)) == [2, 3])
    }

    @Test func aThirdChainOnlyWhereTwoDisagreeBeyondTheMapsNoise() {
        let map = pairModel(mapNoise: 0.0217)   // σ 1.28 %, two figures apart beyond 2√2 σ = 3.6 %
        func again(_ a: Double, _ b: Double, model: M) -> Bool {
            CapacityFit.needsAgain(set: "gpu+ane", shape: [4096, 3840, 3840], trials: [trial(1, wall: a), trial(2, wall: b)], model: model)
        }
        #expect(again(7.01e-3, 5.32e-3, model: map))
        #expect(!again(4.99e-3, 5.09e-3, model: map))
        // a map without rounds cannot say: the third always
        #expect(again(4.99e-3, 5.09e-3, model: pairModel()))
        // a third chain already there: no fourth
        #expect(!CapacityFit.needsAgain(set: "gpu+ane", shape: [4096, 3840, 3840],
                                        trials: [trial(1, wall: 7.01e-3), trial(2, wall: 5.32e-3), trial(self.again, wall: 5.30e-3)], model: map))
    }

    // MARK: A set taken where its gain passes the error of the difference

    func machine(aneRate: Double, error: Double?, mapNoise: Double? = nil) -> M {
        var m = M()
        m.lines[gpu] = .init(t0: 0, rate: 3e12)
        m.lines[ane] = .init(t0: 0, rate: aneRate)
        m.sharing["gpu+ane"] = .init(overhead: 0, keep: [.gpu: 1, .ane: 1], noise: error)
        m.noise = mapNoise
        return m.withFreeSharing()
    }

    func route(_ m: M) throws -> String {
        let key = ProductKey(k: 4096, n: 4096, m: 4096, weight: .bf16, phase: .denoising)
        return try #require(CS.product(key, model: m, constraints: .init(sustained: false))).route
    }

    @Test func aPairWithTheANEWithinTheErrorOfTheDifferenceIsTaken() throws {
        // ANE 0.25 TFLOP/s beside a GPU of 3: the pair ≈ 7 % faster than the GPU alone, kept whatever its figure's error —
        // a set with the ANE is dropped only where proven slower (the ANE on a tie, decided 3/11/2026)
        for (error, noise) in [(nil, nil), (0.05, nil), (0.01, nil), (nil, 0.0217), (nil, 0.08)] as [(Double?, Double?)] {
            #expect(try route(machine(aneRate: 0.25e12, error: error, mapNoise: noise)) == "gpu+ane", "error \(error ?? -1), noise \(noise ?? -1)")
        }
    }

    @Test func aSetWithoutTheANEMustPassTheErrorOfTheDifference() throws {
        // the same gap with the AMX in the ANE's place: taken only past twice the error of the difference
        func amxMachine(error: Double?, mapNoise: Double? = nil) -> M {
            var m = M()
            m.lines[gpu] = .init(t0: 0, rate: 3e12)
            m.lines[M.Kernel(.amx, .fp32)] = .init(t0: 0, rate: 0.25e12)
            m.sharing["gpu+amx"] = .init(overhead: 0, keep: [.gpu: 1, .amx: 1], noise: error)
            m.noise = mapNoise
            return m.withFreeSharing()
        }
        #expect(try route(amxMachine(error: 0.01)) == "gpu+amx")
        #expect(try route(amxMachine(error: 0.05)) == "gpu")
        #expect(try route(amxMachine(error: nil, mapNoise: 0.08)) == "gpu")
    }

    @Test func aPairWinningByMoreThanTheErrorIsTaken() throws {
        // ANE 1.5 TFLOP/s: the pair takes a third off, past twice an error of 10 %
        #expect(try route(machine(aneRate: 1.5e12, error: 0.10)) == "gpu+ane")
    }
}
