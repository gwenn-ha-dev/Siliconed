import Foundation
import Testing
@testable import Siliconed

/// **The disk beside the computation** (`CapacityDisk.swift`, `CapacityStream.swift`): the rate a probe reads, the overlap `Combine` makes,
/// on a synthetic machine whose right answer is computed by hand. No weights, no chip name.
@Suite struct CapacityStreamTests {
    typealias M = CapacityModel
    typealias CS = CapacitySolver
    let gpu = M.Kernel(.gpu, .fp32)

    func machine(disk: Double?) -> M {
        var m = M()
        m.lines[gpu] = .init(t0: 0, rate: 3e12)
        m.diskBandwidth = disk
        return m.withFreeSharing()
    }

    let request = CapacityClasses.Request(family: .krea2, variant: .standard, width: 1024, height: 1024, fasterRendering: true,
                                          promptTokens: 11, steps: 8, includesEncoder: false)

    // MARK: the probe's arithmetic

    @Test func theRateIsTheFastestPass() {
        // A disturbance only adds time: the 2 s pass is a busy one, the rate is the 1 s pass's.
        #expect(CapacityDisk.rate(bytes: 4_000_000_000, passes: [1.0, 2.0, 1.4]) == 4e9)
        #expect(CapacityDisk.rate(bytes: 4_000_000_000, passes: [0, -1]) == nil)
        #expect(CapacityDisk.rate(bytes: 0, passes: [1]) == nil)
    }

    @Test func aPassHidesTheReadBeneathTheComputationAndPaysTheExcess() {
        // C = 18 s, R = 3 s: the disk is hidden; five evaluations, 0.1 s exposed at the start of each.
        #expect(abs(CapacityDisk.pass(compute: 18, reading: 3, exposed: 0.1, evaluations: 5) - 5 * 18.1) < 1e-9)
        // C = 2, R = 3: the disk binds, for its excess only.
        #expect(abs(CapacityDisk.pass(compute: 2, reading: 3, exposed: 0, evaluations: 3) - 9) < 1e-9)
    }

    @Test func theProbeReadsAScratchFileAndLeavesNothing() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("disk-probe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let rate = try #require(CapacityDisk.measure(in: directory, bytes: 64 << 20, passes: 2))
        #expect(rate > 1e7 && rate < 1e12)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
        #expect(CapacityDisk.measure(in: directory.appendingPathComponent("absent"), bytes: 1 << 20, passes: 1) == nil)
    }

    // MARK: Combine

    func streamBlock(reading: Double, beneath: Double, exposed: Double = 0) -> CS.Block {
        .init(classID: "stream", phase: .denoising, executions: 1, route: "disk", rows: [:], product: nil, attention: nil,
              preparation: 0, compute: 0, busy: [:], reading: reading, beneath: beneath, exposed: exposed)
    }
    func compute(_ seconds: Double) -> CS.Block {
        .init(classID: "c\(seconds)", phase: .denoising, executions: 1, route: "gpu", rows: [:], product: nil, attention: nil,
              preparation: 0.25 * seconds, compute: 0.75 * seconds, busy: [:])
    }

    @Test func overlappedIsTheComputationWhenTheDiskIsFasterAndTheDiskForItsExcess() {
        let hidden = [compute(10), streamBlock(reading: 4, beneath: 10, exposed: 0.5)]
        #expect(abs(CS.overlapped(hidden) - 10.5) < 1e-12)
        #expect(abs(CS.serial(hidden) - 14) < 1e-12)
        let binding = [compute(10), streamBlock(reading: 13, beneath: 10)]
        #expect(abs(CS.overlapped(binding) - 13) < 1e-12)
        #expect(CS.sequential(binding) == 10)
        // Without a stream block the three are the sum.
        #expect(CS.overlapped([compute(3), compute(4)]) == 7 && CS.serial([compute(3), compute(4)]) == 7)
    }

    // MARK: the plan with its stream

    @Test func aStreamedPlanReadsEachPassBesideItsOwnComputation() throws {
        let model = machine(disk: 4e9)
        let constraints = CS.Constraints(sustained: false, allowsANE: false)
        let bytes = 24e9
        let plain = try #require(CS.solve(model: model, request: request, constraints: constraints))
        let streamed = try #require(CS.solve(model: model, request: request, constraints: constraints,
                                             streaming: .init(bytesPerEvaluation: bytes, firstBytes: 0.4e9)))
        let passes = CapacityClasses.passes(request)
        let stream = streamed.blocks.filter { $0.route == "disk" }
        #expect(stream.count == passes.count && passes.count == 2)
        for (block, pass) in zip(stream, passes) {
            #expect(abs(block.reading - Double(pass.evaluations) * bytes / 4e9) < 1e-9)
            #expect(abs(block.exposed - Double(pass.evaluations) * 0.4e9 / 4e9) < 1e-9)
            #expect(block.beneath > 0 && block.seconds == 0)
        }
        // The blocks of the plan are the plain plan's, untouched, and the stream only adds what it cannot hide.
        #expect(streamed.blocks.filter { $0.route != "disk" } == plain.blocks)
        let extra = stream.reduce(0) { $0 + max(0, $1.reading - $1.beneath) + $1.exposed }
        #expect(abs(streamed.seconds - (plain.seconds + extra)) < 1e-9)
        #expect(streamed.seconds <= CS.serial(streamed.blocks))
        // A disk twenty times faster costs no more than its first visits.
        var fast = model; fast.diskBandwidth = 80e9
        let quick = try #require(CS.solve(model: fast, request: request, constraints: constraints,
                                          streaming: .init(bytesPerEvaluation: bytes, firstBytes: 0.4e9)))
        #expect(quick.seconds < plain.seconds + 0.2)
    }

    @Test func aSlowDiskBindsTheHalfGridBeforeTheFullGrid() throws {
        let constraints = CS.Constraints(sustained: false, allowsANE: false)
        let slow = try #require(CS.solve(model: machine(disk: 1e9), request: request, constraints: constraints,
                                         streaming: .init(bytesPerEvaluation: 24e9)))
        let stream = slow.blocks.filter { $0.route == "disk" }
        // Per evaluation the half grid computes less than the full one but reads the same bytes.
        let (half, full) = (stream[0], stream[1])
        #expect(half.beneath / Double(half.executions) < full.beneath / Double(full.executions))
        #expect(half.reading / Double(half.executions) == full.reading / Double(full.executions))
    }

    @Test func noDiskRateOrNothingStreamedLeavesThePlanAsItWas() throws {
        let constraints = CS.Constraints(sustained: false, allowsANE: false)
        let plain = try #require(CS.solve(model: machine(disk: nil), request: request, constraints: constraints))
        let unmeasured = try #require(CS.solve(model: machine(disk: nil), request: request, constraints: constraints,
                                               streaming: .init(bytesPerEvaluation: 24e9)))
        #expect(unmeasured.blocks == plain.blocks && unmeasured.seconds == plain.seconds)
        let resident = try #require(CS.solve(model: machine(disk: 4e9), request: request, constraints: constraints,
                                             streaming: .init(bytesPerEvaluation: 0)))
        #expect(resident.blocks == plain.blocks)
    }

    @Test func theModelKeepsItsDiskRateThroughJSON() throws {
        let model = machine(disk: 5.2e9)
        let decoded = try JSONDecoder().decode(M.self, from: JSONEncoder().encode(model))
        #expect(decoded.diskBandwidth == 5.2e9)
    }
}
