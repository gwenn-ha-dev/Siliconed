import Foundation
import XCTest
@testable import Siliconed

/// **The silicon inventory without a GPU**: the analysis on values captured from an M1 Pro
/// (`ioreg -rc AGXAccelerator`, `ioreg -l` on the `gpu` node and `ane0`), the refusals when a layout
/// disagrees with the cut the node declares, and the report that carries it (compact, and an older one
/// without it still decoding).
final class InventoryTests: XCTestCase {
    private typealias Inventory = Diagnostic.Inventory

    private func data(hex: String) -> Data {
        Data(stride(from: 0, to: hex.count, by: 2).map {
            UInt8(hex[hex.index(hex.startIndex, offsetBy: $0)..<hex.index(hex.startIndex, offsetBy: $0 + 2)], radix: 16)!
        })
    }

    /// `"perf-states" = <…>` of the M1 Pro's gpu node: 2 tables of 7 entries `(Hz, mV)`, 0 Hz first.
    private let perfStates = "0000000090010000009e2c176702000080c5f71c7702000000b29f2690020000003c592ec2020000008bef390303000000643f4d7d03000000000000"
        + "90010000009e2c176702000080c5f71c7702000000b29f268d020000003c592ebc020000008bef39fd02000000643f4d71030000"

    func testGPUClockFromTheNodesOwnTable() {
        XCTAssertEqual(Inventory.gpuMaxMHz(perfStates: data(hex: perfStates), stateCount: 7, tableCount: 2, activeStates: 6), 1296)
    }

    func testGPUClockRefusedWhenTheLayoutDisagreesWithTheDeclaredCut() {
        let table = data(hex: perfStates)
        XCTAssertNil(Inventory.gpuMaxMHz(perfStates: table, stateCount: 7, tableCount: 1, activeStates: 6), "length is not the cut")
        XCTAssertNil(Inventory.gpuMaxMHz(perfStates: table, stateCount: 7, tableCount: 2, activeStates: 5), "active states differ")
        XCTAssertNil(Inventory.gpuMaxMHz(perfStates: table.dropLast(8), stateCount: 7, tableCount: 2, activeStates: 6))
        XCTAssertNil(Inventory.gpuMaxMHz(perfStates: nil, stateCount: 7, tableCount: 2, activeStates: 6))
        XCTAssertNil(Inventory.gpuMaxMHz(perfStates: table, stateCount: nil, tableCount: 2, activeStates: 6))
        // Two bins whose tops differ: no single maximum.
        var other = table
        other[other.count - 8] = 0x00; other[other.count - 7] = 0x5a; other[other.count - 6] = 0x62; other[other.count - 5] = 0x4d
        XCTAssertNil(Inventory.gpuMaxMHz(perfStates: other, stateCount: 7, tableCount: 2, activeStates: 6))
    }

    func testGPUConfigurationOfTheM1Pro() {
        // "GPUConfigurationVariable" = {"num_gps"=8,"gpu_gen"=13,"is_sksm"=0,"usc_gen"=2,"num_cores"=16,"num_mgpus"=2,
        //  "gpu_var"="S","core_mask_list"=(253,253),"num_frags"=16}
        let c = Inventory.gpuConfiguration(["num_gps": 8, "gpu_gen": 13, "is_sksm": 0, "usc_gen": 2, "num_cores": 16, "num_mgpus": 2,
                                            "gpu_var": "S", "core_mask_list": [253, 253], "num_frags": 16])
        XCTAssertEqual(c.cores, 16); XCTAssertEqual(c.gen, 13); XCTAssertEqual(c.mask, [253, 253])
        let none = Inventory.gpuConfiguration(nil)
        XCTAssertNil(none.cores); XCTAssertNil(none.gen); XCTAssertNil(none.mask)
        XCTAssertNil(Inventory.gpuConfiguration(["num_cores": 0, "core_mask_list": []]).cores)
        XCTAssertNil(Inventory.gpuConfiguration(["core_mask_list": []]).mask)
    }

    func testRegistryScalars() {
        XCTAssertEqual(Inventory.integer32(data(hex: "06000000")), 6)
        XCTAssertEqual(Inventory.integer32(data(hex: "60000000")), 0x60)
        XCTAssertNil(Inventory.integer32(data(hex: "0600")))
        XCTAssertNil(Inventory.integer32(nil))
        // "compatible" = <"ane,t8020"> (the registry returns the NUL-terminated string)
        XCTAssertEqual(Inventory.firstString(Data("ane,t8020".utf8) + Data([0])), "ane,t8020")
        XCTAssertEqual(Inventory.firstString(Data("a,b".utf8) + Data([0]) + Data("c".utf8)), "a,b")
        XCTAssertNil(Inventory.firstString(Data([0])))
    }

    func testClustersRoundUp() {
        // hw.perflevel0: 6 cores, cpusperl2 3 → 2 clusters; perflevel1: 2 cores, 2 per L2 → 1.
        XCTAssertEqual(Inventory.CPULevel(name: "Performance", cores: 6, coresPerL2: 3).clusters, 2)
        XCTAssertEqual(Inventory.CPULevel(name: "Efficiency", cores: 2, coresPerL2: 2).clusters, 1)
        XCTAssertEqual(Inventory.CPULevel(name: "Performance", cores: 5, coresPerL2: 4).clusters, 2)
        XCTAssertNil(Inventory.CPULevel(name: "Performance", cores: 5, coresPerL2: nil).clusters)
        XCTAssertNil(Inventory.CPULevel(name: "Performance", cores: 5, coresPerL2: 0).clusters)
    }

    func testNominalGPUPeak() throws {
        // 14 × 128 × 2 × 1.296 GHz
        let peak = try XCTUnwrap(Inventory(gpuMaxMHz: 1296).gpuFP32Peak(enabledCores: 14))
        XCTAssertEqual(peak, 4.644864e12, accuracy: 1e6)
        XCTAssertNil(Inventory(gpuMaxMHz: 1296).gpuFP32Peak(enabledCores: 0))
        XCTAssertNil(Inventory().gpuFP32Peak(enabledCores: 14))
    }

    func testEncodingIsCompactAndRoundTrips() throws {
        let inventory = Inventory(gpuSiliconCores: 16, gpuGen: 13, gpuCoreMask: [253, 253], gpuMaxMHz: 1296,
                                  cpuLevels: [.init(name: "Performance", cores: 6, coresPerL2: 3),
                                              .init(name: "Efficiency", cores: 2, coresPerL2: 2)],
                                  ane: .init(compatible: "ane,t8020", type: 96, subtype: 0))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(inventory)
        XCTAssertLessThan(bytes.count, 330, String(decoding: bytes, as: UTF8.self))
        XCTAssertEqual(try JSONDecoder().decode(Inventory.self, from: bytes), inventory)
        // Unread fields are absent, not null.
        XCTAssertEqual(String(decoding: try encoder.encode(Inventory()), as: UTF8.self), "{}")
    }

    func testMachineWithoutInventoryDecodesAndStaysAbsent() throws {
        let machine = Diagnostic.Machine(chip: "Apple M1 Pro", hardwareModel: "MacBookPro18,3", performanceCores: 6, efficiencyCores: 2,
                                         gpuCores: 14, memoryBytes: 17_179_869_184, macOS: "27.0", version: "x", build: "y")
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let old = try encoder.encode(machine)
        XCTAssertFalse(String(decoding: old, as: UTF8.self).contains("inventory"))
        let decoded = try JSONDecoder().decode(Diagnostic.Machine.self, from: old)
        XCTAssertNil(decoded.inventory); XCTAssertNil(decoded.gpuFP32Peak)
        var withInventory = machine
        withInventory.inventory = Inventory(gpuMaxMHz: 1296)
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Machine.self, from: try encoder.encode(withInventory)), withInventory)
        XCTAssertNotNil(withInventory.gpuFP32Peak)
    }

    /// On a runner with no `AGXAccelerator` the reading must not crash; what it returns there is `nil`, here whatever the Mac says.
    func testLiveReadingDoesNotCrash() {
        let inventory = Inventory.read()
        if let levels = inventory.cpuLevels { XCTAssertTrue(levels.allSatisfy { $0.cores > 0 }) }
        if let mhz = inventory.gpuMaxMHz { XCTAssertTrue((100...6000).contains(mhz)) }
    }
}
