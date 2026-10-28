import Foundation
import XCTest
@testable import Siliconed

/// **What a lending Mac says of itself** (`Remote/Machine.swift`): the messages cross as JSON and come back equal.
final class RemoteMachineTests: XCTestCase {
    private func report() -> Diagnostic.Report {
        let machine = Diagnostic.Machine(chip: "Apple M4", hardwareModel: "Mac16,10", performanceCores: 4, efficiencyCores: 6,
                                         gpuCores: 10, memoryBytes: 17_179_869_184, macOS: "27.0.1 (26A434)",
                                         version: "0.17 (170)", build: "x+y")
        let conditions = Diagnostic.Conditions(thermal: "nominal", lowPowerMode: false, onBattery: false, swapUsedBytes: 0)
        return Diagnostic.Report(date: "2026-10-28T10:00:00Z", seconds: 61.5, machine: machine, cpuFeatures: [:],
                                 before: conditions, after: conditions, products: [], lowPrecision: [], attention: [],
                                 widening: [], witness: .init(gpuBefore: 3.4, gpuAfter: 3.4, amxBefore: nil, amxAfter: nil))
    }

    func testMachineAndItsAsksTravelAsJSON() throws {
        let map = RemoteMap(date: "2026-10-28T10:00:00Z", current: true, green: true)
        var m = RemoteMachine(chip: "Apple M4", memoryBytes: 17_179_869_184, ceiling: 10_737_418_240, ready: ["z-image", "qwen-image-2.1"],
                              stored: map, inUse: map, warnings: ["« x » ignored in the profile"])
        XCTAssertTrue(m.measured)
        var messages: [SessionMessage] = [.machine(m)]
        for d: RemoteDiagnostic in [.running(done: 2, total: 5), .finished(report()), .failed(.app(.machineBusy)),
                                    .failed(.engine(.renderAlreadyRunning))] {
            m.diagnostic = d
            messages.append(.machine(m))
        }
        for b in [RemoteBlocker.measuring, .rendering, .installing] {
            m.blocker = b
            messages.append(.machine(m))
        }
        messages += [.measure, .stopMeasure, .setFasterRendering(true), .setFasterRendering(false),
                     .refused(.app(.machineBusy))]
        for message in messages {
            XCTAssertEqual(try JSONDecoder().decode(SessionMessage.self, from: JSONEncoder().encode(message)), message)
        }
    }

    func testMapIsMeasuredOnlyWhenCurrentAndGreen() {
        var m = RemoteMachine(chip: "c", memoryBytes: 1, ceiling: 1, ready: [])
        XCTAssertFalse(m.measured, "no map")
        m.stored = RemoteMap(date: "d", current: true, green: false)
        XCTAssertFalse(m.measured, "orange")
        m.stored = RemoteMap(date: "d", current: false, green: true)
        XCTAssertFalse(m.measured, "another macOS or build")
        m.stored = RemoteMap(date: "d", current: true, green: true)
        XCTAssertTrue(m.measured)
    }

    func testTheMenuKeepsNeitherReportNorPhase() throws {
        var m = RemoteMachine(chip: "c", memoryBytes: 1, ceiling: 1, ready: ["z-image"], warnings: ["w"],
                              diagnostic: .finished(report()), blocker: .rendering)
        m.stored = RemoteMap(date: "d", current: true, green: true)
        let kept = m.forMenu
        XCTAssertEqual(kept.diagnostic, .idle)
        XCTAssertNil(kept.blocker)
        XCTAssertTrue(kept.warnings.isEmpty)
        XCTAssertEqual(kept.stored, m.stored)
        XCTAssertEqual(kept.ready, ["z-image"])
        XCTAssertLessThan(try JSONEncoder().encode(kept).count, 400)
    }

    func testMapFromAReport() {
        let r = report()
        let m = RemoteMap(r)
        XCTAssertEqual(m.date, r.date)
        XCTAssertEqual(m.green, r.isGreen)
        XCTAssertEqual(m.current, r.isCurrent)
    }
}
