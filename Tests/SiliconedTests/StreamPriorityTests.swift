import Foundation
import Testing
@testable import Siliconed

/// **The class of service of the weight stream's readers is the map's equation** (`CapacityDisk.choose`, `streamClass`): the lowest class whose
/// measured mean covers what a phase must read under its computation, else the default. Synthetic machine, no weights, no chip name.
@Suite struct StreamPriorityTests {
    typealias R = CapacityDisk.Reading
    let gpu = CapacityModel.Kernel(.gpu, .fp32)

    func model(background: R?, utility: R? = nil) -> CapacityModel {
        var m = CapacityModel()
        m.lines[gpu] = .init(t0: 0, rate: 3e12)
        m.diskBandwidth = 5e9
        m.diskClasses["default"] = R(rate: 5e9, mean: 4.8e9)
        if let background { m.diskClasses["background"] = background }
        if let utility { m.diskClasses["utility"] = utility }
        return m.withFreeSharing()
    }

    func plan(_ m: CapacityModel) -> CapacityPlan { CapacityPlan(model: m, allowsANE: false) }

    @Test func theClassesAreOrderedLowestFirst() {
        #expect(StreamClass.allCases == [.background, .utility, .standard])
        #expect(StreamClass.background < .utility && .utility < .standard)
    }

    @Test func theLowestClassThatCoversThePhaseIsChosen() {
        let readings: [StreamClass: R] = [.background: R(rate: 3e9, mean: 2.5e9), .utility: R(rate: 4e9, mean: 3.5e9), .standard: R(rate: 5e9, mean: 4.8e9)]
        // 12 GB under 18 s: 0.67 GB/s, covered by the background class.
        #expect(CapacityDisk.choose(bytes: 12e9, compute: 18, readings: readings) == .background)
        // 12 GB under 5 s: 2.4 GB/s plus the margin (2.58): the background mean 2.5 does not cover it, the utility's does.
        #expect(CapacityDisk.choose(bytes: 12e9, compute: 5, readings: readings) == .utility)
        // 12 GB under 3 s: 4 GB/s: only the default reads that fast.
        #expect(CapacityDisk.choose(bytes: 12e9, compute: 3, readings: readings) == .standard)
    }

    @Test func theMeanDecidesNotTheBestPass() {
        // The best pass of the background class covers the need, its mean does not: the stream would be starved.
        let readings: [StreamClass: R] = [.background: R(rate: 3e9, mean: 1e9)]
        #expect(CapacityDisk.choose(bytes: 12e9, compute: 6, readings: readings) == .standard)
    }

    @Test func theMarginIsTheUncertaintyOfThePredictedComputation() {
        let readings: [StreamClass: R] = [.background: R(rate: 2e9, mean: 2e9)]
        // Need exactly 2 GB/s: covered with no margin, not with the diagnostic's (half its spread rule).
        #expect(CapacityDisk.choose(bytes: 12e9, compute: 6, readings: readings, uncertainty: 0) == .background)
        #expect(CapacityDisk.choose(bytes: 12e9, compute: 6, readings: readings) == .standard)
        #expect(CapacityDisk.computeUncertainty == Diagnostic.Regime.Rules.spread / 2)
    }

    @Test func nothingMeasuredOrNothingToReadIsTheDefault() {
        #expect(CapacityDisk.choose(bytes: 12e9, compute: 18, readings: [:]) == .standard)
        #expect(CapacityDisk.choose(bytes: 0, compute: 18, readings: [.background: R(rate: 1e12, mean: 1e12)]) == .standard)
        #expect(CapacityDisk.choose(bytes: 1e9, compute: 0, readings: [.background: R(rate: 1e12, mean: 1e12)]) == .standard)
    }

    @Test func aBackgroundThatCoversTheFullGridButNotTheHalfGridChoosesBackgroundThenDefault() throws {
        let probe = plan(model(background: nil))
        let (full, half) = (1024 * 1024 / 256, 512 * 512 / 256)       // image tokens of the full and the half grid
        let cFull = try #require(CapacityDisk.evaluationSeconds(plan: probe, family: .krea2, format: .int8, tokens: full, textRows: 16))
        let cHalf = try #require(CapacityDisk.evaluationSeconds(plan: probe, family: .krea2, format: .int8, tokens: half, textRows: 16))
        #expect(cHalf < cFull / 2)
        let bytes = 12_000_000_000
        // A background mean 20 % above the full grid's need (margin included) and well under the half grid's.
        let mean = Double(bytes) / cFull * (1 + CapacityDisk.computeUncertainty) * 1.2
        #expect(mean < Double(bytes) / cHalf)
        let p = plan(model(background: R(rate: mean * 1.1, mean: mean)))
        #expect(CapacityDisk.streamClass(plan: p, family: .krea2, format: .int8, tokens: full, textRows: 16, bytes: bytes) == .background)
        #expect(CapacityDisk.streamClass(plan: p, family: .krea2, format: .int8, tokens: half, textRows: 16, bytes: bytes) == .standard)
    }

    @Test func theTextRowsGiveBackThePromptThatMakesThem() {
        for family in [Family.zImage, .qwenImage21, .krea2, .mossImage] {
            let rows = CapacityClasses.textRows(family, promptTokens: 11)
            #expect(CapacityClasses.textRows(family, promptTokens: CapacityClasses.promptTokens(family, textRows: rows)) == rows, "\(family)")
        }
    }

    @Test func aPlanWithoutAModelOrReadingsKeepsTheDefault() {
        #expect(CapacityDisk.streamClass(plan: .defaults, family: .krea2, format: .int8, tokens: 4096, textRows: 16, bytes: 12_000_000_000) == .standard)
        var bare = model(background: nil); bare.diskClasses = [:]
        #expect(CapacityDisk.streamClass(plan: plan(bare), family: .krea2, format: .int8, tokens: 4096, textRows: 16, bytes: 12_000_000_000) == .standard)
    }

    @Test func theReportKeepsTheReadingsByClass() throws {
        let m = model(background: R(rate: 3e9, mean: 2.5e9))
        let decoded = try JSONDecoder().decode(CapacityModel.self, from: JSONEncoder().encode(m))
        #expect(decoded.diskClasses == m.diskClasses)
        #expect(decoded.diskReadings[.background] == R(rate: 3e9, mean: 2.5e9))
        // A report written before the classes were measured reads as none.
        var old = CapacityModel(); old.diskBandwidth = 4e9
        #expect(try JSONDecoder().decode(CapacityModel.self, from: JSONEncoder().encode(old)).diskClasses.isEmpty)
        var measured = CapacityFit.Measurements()
        measured.diskClasses["background"] = R(rate: 3.123456789e9, mean: 2.5e9)
        let again = try JSONDecoder().decode(CapacityFit.Measurements.self, from: JSONEncoder().encode(measured.rounded()))
        #expect(again.diskClasses["background"]?.rate == 3.1235e9)
    }

    @Test func theProbeReadsEveryClassOfTheScratchFile() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("disk-classes-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let readings = try #require(CapacityDisk.measureClasses(in: directory, bytes: 64 << 20, passes: 2, beside: 0))
        #expect(Set(readings.keys) == Set(StreamClass.allCases))
        for r in readings.values { #expect(r.rate >= r.mean && r.mean > 1e6) }
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }
}
