import CoreAI
import Foundation
import SiliCore
import Synchronization
import Testing

/// **A block prepared opens without compiling** (`Block.prepare`, `Functions.specialize`): this process's
/// specializations erased, the shared block of `aSharedBlockIsTheGPUAlones` at 1024 rows — two units of 512 the ANE may
/// take, in flight — prepared, the ANE's first, each specialization told as a compilation; then opened in flight at its
/// start, and on the GPU alone (the share of none a move takes first): neither opening compiles.
extension Runs {
    @Test(.enabled(if: metalEnabled))
    func aPreparedBlockOpensWithoutCompiling() async throws {
        let bench = try await Bench()
        let measured = FileManager.default.temporaryDirectory.appendingPathComponent("this-mac-prepare-\(UUID().uuidString).json")
        let before = ThisMac.file
        ThisMac.file = measured
        defer { ThisMac.file = before; try? FileManager.default.removeItem(at: measured) }
        let file = try SafeTensors(try weightsFile("prepare.safetensors", [("L.w", "BF16", [64, 64], [UInt8](repeating: 0, count: 64 * 64 * 2))]))
        let d = Self.sharedDeclaration(rows: 1024, heads: 4, hd: 64)

        Functions.eraseSpecializations(of: [ProcessInfo.processInfo.processName])
        let specializations = try Block.prepare(d, files: [file], budget: 1 << 40, share: .inFlight)
        let units = specializations.map { $0.unit == .neuralEngine ? "ane" : "gpu" }
        #expect(units.first == "ane" && units == units.sorted(), "the ANE's first: \(units)")
        let compilations = Mutex(0)
        let observation = Stages.observe { if case .compiling = $0 { compilations.withLock { $0 += 1 } } }
        defer { observation.cancel() }
        for s in specializations { try await s.run() }
        print("prepared: \(units.filter { $0 == "ane" }.count) of the ANE's, \(units.filter { $0 == "gpu" }.count) of the GPU's")
        #expect(compilations.withLock { $0 } == specializations.count, "\(compilations.withLock { $0 }) compilations told of \(specializations.count)")

        for share in [Block.Share.inFlight, .gpu] {
            let block = try await Block.open(bench.device, d, files: [file], budget: 1 << 40, share: share)
            defer { block.close() }
            #expect(!block.stage.compiled, "\(share): its opening compiled")
        }
        #expect(compilations.withLock { $0 } == specializations.count, "the openings compiled")
    }
}
