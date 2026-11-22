import Foundation
import XCTest
@testable import SiliCore
@testable import Siliconed

/// **The VAEs' plans, without a byte of weights**: one function a layer's shape whatever the picture's height, what
/// each pass hands back, the boundaries it reads, every weight a resident (`Bands`).
final class VAEBandsTests: XCTestCase {
    private let flux = FluxVAEDecoder.vae, flux2 = Flux2VAEDecoder.vae

    /// The functions a plan builds, by their keys and shapes.
    private func shapes(_ b: Bands) throws -> Set<Bands.Shaped> { Set(try b.modules().keys) }

    /// **A picture as tall as it likes, the same functions**: at one width, every height decodes and encodes with the
    /// functions of the others (the head and the tail aside, whole: the attention reads every position), a taller one
    /// with more calls.
    func testTheFunctionsDoNotMoveWithTheHeight() throws {
        func banded(_ b: Bands) throws -> Set<Bands.Shaped> { try shapes(b).filter { !$0.key.contains("-head-") && !$0.key.contains("-tail-") && !$0.key.contains("head(") && !$0.key.contains("tail(") } }
        for vae in [flux, flux2] {
            let plans = [512, 832, 1024, 1248].map { FluxVAEPlan(vae, .decode, height: $0 / vae.factor, width: 832 / vae.factor, positions: FluxVAE.candidates[0]) }
            XCTAssert(try plans.allSatisfy { try banded($0.bands) == banded(plans[0].bands) }, vae.name)
            XCTAssertLessThan(plans[0].bands.callCount, plans[3].bands.callCount)
            let encoders = [512, 1024, 1248].map { FluxVAEPlan(vae, .encode, height: $0, width: 832, positions: FluxVAE.candidates[0]) }
            XCTAssert(try encoders.allSatisfy { try banded($0.bands) == banded(encoders[0].bands) }, vae.name)
        }
        let qwen = [512, 832, 1024, 1248].map { QwenImageVAEDecoder.plan(h: $0 / 8, w: 104, positions: QwenImageVAE.positions[0]) }
        XCTAssert(try qwen.allSatisfy { try banded($0.bands) == banded(qwen[0].bands) })
        let qwen21 = [512, 832, 1024].map { QwenImage21VAEDecoder.plan(h: $0 / 16, w: 52, positions: QwenImageVAE.positions[0]) }
        XCTAssert(try qwen21.allSatisfy { try banded($0.bands) == banded(qwen21[0].bands) })
    }

    /// **Every function builds**, the decoders' and the encoders', at a size and a short band: what a pass loads.
    func testEveryFunctionBuilds() throws {
        for vae in [flux, flux2] {
            for p in [FluxVAEPlan(vae, .decode, height: 512 / vae.factor, width: 768 / vae.factor, positions: FluxVAE.candidates[3]),
                      FluxVAEPlan(vae, .encode, height: 512, width: 768, positions: FluxVAE.candidates[3])] {
                XCTAssertEqual(try p.bands.modules().count, p.bands.functionCount, "\(vae.name) \(p.direction)")
            }
        }
        for p in [QwenImageVAEDecoder.plan(h: 64, w: 96, positions: QwenImageVAE.positions[3]), QwenImageVAEEncoder.plan(height: 512, width: 768, positions: QwenImageVAE.positions[3]),
                  QwenImage21VAEDecoder.plan(h: 32, w: 48, positions: QwenImageVAE.positions[3]), QwenImage21VAEEncoder.plan(height: 512, width: 768, positions: QwenImageVAE.positions[3])] {
            XCTAssertEqual(try p.bands.modules().count, p.bands.functionCount, p.network)
        }
    }

    /// One function a layer's shape: Z's decoder, its 35 layers, in 20 functions (the head, 19 by bands).
    func testLayersOfOneShapeShareAFunction() {
        let plan = FluxVAEPlan(flux, .decode, height: 128, width: 128, positions: FluxVAE.candidates[0])
        XCTAssertEqual(plan.bands.functionCount, 20)
    }

    /// The decoder hands back the picture, the encoder the latent its decoder reads: Flux's 16 channels at an eighth,
    /// Flux 2's 128 patched at a sixteenth.
    func testThePassesHandBackTheirPlanes() {
        for vae in [flux, flux2] {
            let d = FluxVAEPlan(vae, .decode, height: 1024 / vae.factor, width: 768 / vae.factor, positions: FluxVAE.candidates[0])
            XCTAssertEqual([d.output.channels, d.output.rows, d.output.width], [3, 1024, 768])
            XCTAssertEqual([d.input.channels, d.input.rows, d.input.width], [vae.latentChannels, 1024 / vae.factor, 768 / vae.factor])
        }
        let a = FluxVAEPlan(flux, .encode, height: 1024, width: 512, positions: FluxVAE.candidates[0])
        XCTAssertEqual([a.output.channels, a.output.rows, a.output.width], [16, 128, 64])
        let b = FluxVAEPlan(flux2, .encode, height: 1024, width: 512, positions: FluxVAE.candidates[0])
        XCTAssertEqual([b.output.channels, b.output.rows, b.output.width], [128, 64, 32])
        let q = QwenImage21VAEDecoder.plan(h: 32, w: 48, positions: QwenImageVAE.positions[0])
        XCTAssertEqual([q.result!.channels, q.result!.rows, q.result!.width], [4, 512, 768])
        let e = QwenImageVAEEncoder.plan(height: 512, width: 768, positions: QwenImageVAE.positions[0])
        XCTAssertEqual([e.result!.channels, e.result!.rows, e.result!.width], [16, 64, 96])
    }

    /// The boundaries the oracle writes are marked, in the decoder's order.
    func testTheBoundariesAreMarkedInOrder() {
        let plan = FluxVAEPlan(flux2, .decode, height: 32, width: 32, positions: FluxVAE.candidates[0])
        let marks = plan.bands.marks
        XCTAssertEqual(marks.map(\.name), ["decoder_mid", "decoder_up0", "decoder_up1", "decoder_up2"])
        XCTAssertEqual(marks.map(\.call), marks.map(\.call).sorted())
        XCTAssert(Set(Flux2VAEDecoder.tapNames).isSubset(of: marks.map(\.name)))
    }

    /// Every weight a resident read by its recipe, no kernel in an asset: the kernels of Wan's causal VAE at their last
    /// time slice, the attention's 1×1s as the product's matrix.
    func testEveryWeightIsAResident() {
        let plan = QwenImageVAEDecoder.plan(h: 64, w: 64, positions: QwenImageVAE.positions[0])
        XCTAssert(plan.bands.weights.contains("decoder.up_blocks.0.resnets.0.conv1.weight"))
        XCTAssert(plan.bands.weights.contains("decoder.mid_block.attentions.0.to_qkv.weight"))
        let z = FluxVAEPlan(flux, .decode, height: 64, width: 64, positions: FluxVAE.candidates[0])
        XCTAssert(z.bands.weights.contains("decoder.conv_out.weight"))
        XCTAssertFalse(z.bands.weights.contains { $0.hasPrefix("encoder.") })
    }

    /// Wan's convolution to the pixels lies in the tallest candidate's bands in every plan: its bits do not move with B.
    func testThePixelsConvolutionKeepsItsBands() throws {
        for (tall, short) in [(QwenImageVAEDecoder.plan(h: 128, w: 128, positions: 1 << 14), QwenImageVAEDecoder.plan(h: 128, w: 128, positions: 1 << 11)),
                              (QwenImage21VAEDecoder.plan(h: 64, w: 64, positions: 1 << 14), QwenImage21VAEDecoder.plan(h: 64, w: 64, positions: 1 << 11))] {
            func outs(_ p: QwenImageVAE.Plan) throws -> Set<Int> { Set(try shapes(p.bands).filter { $0.key.contains("out(") }.map(\.geometry.rows)) }
            XCTAssertEqual(try outs(tall), try outs(short))
            XCTAssertFalse(try outs(tall).isEmpty)
            XCTAssertNotEqual(try shapes(tall.bands), try shapes(short.bands))
        }
    }
}

/// **The compilations' mark**, beside their folder, says whose they are; what this Mac measured is read in the app space.
final class CompilationsTests: XCTestCase {
    /// What this Mac measured is read in the engine folder of the library's app space.
    func testTheMeasuresAreTheAppSpaces() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("measures-\(UUID().uuidString)")
        let (folder, file) = (Functions.folder, ThisMac.file)
        defer { try? fm.removeItem(at: root); Functions.folder = folder; ThisMac.file = file }
        let library = Library(root: root.appendingPathComponent("library"), local: root.appendingPathComponent("local"))
        CoreAI.place(library, functions: root.appendingPathComponent("caches/functions"))
        XCTAssertEqual(ThisMac.url, library.local.appendingPathComponent("engine/this-mac.json"))
        XCTAssertFalse(ThisMac.url.path.hasPrefix(library.root.path + "/"), "the measures stay on this Mac when the library moves")
    }

    func testTheMarkSaysWhoseTheCompilationsAre() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("mark-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        let functions = root.appendingPathComponent("Siliconed/functions")
        XCTAssertTrue(CoreAI.due(installed: "1.0 (1)", folder: functions))
        CoreAI.markCompiled(for: "1.0 (1)", folder: functions)
        XCTAssertFalse(CoreAI.due(installed: "1.0 (1)", folder: functions))
        XCTAssertTrue(CoreAI.due(installed: "1.1 (2)", folder: functions))
        // Erasing the assets leaves the mark: it is beside them.
        try fm.createDirectory(at: functions.appendingPathComponent("a-00.aimodel"), withIntermediateDirectories: true)
        try Data(count: 5000).write(to: functions.appendingPathComponent("a-00.aimodel/main.hash"))
        XCTAssertGreaterThanOrEqual(CoreAI.compiled(functions, under: root), 5000)
        Functions.erase(functions)
        XCTAssertEqual(CoreAI.compiled(functions, under: root), 0)
        XCTAssertFalse(CoreAI.due(installed: "1.0 (1)", folder: functions))
    }

    /// The mark names the engine's binary too: its UUID read from the Mach-O, this test's own process standing for it
    /// (universal or not); a file that is no binary has none.
    func testTheMarkReadsABinarysUUID() throws {
        let me = URL(fileURLWithPath: ProcessInfo.processInfo.arguments[0])
        let uuid = try XCTUnwrap(CoreAI.binaryUUID(me))
        XCTAssertEqual(uuid.count, 32)
        XCTAssertTrue(uuid.allSatisfy(\.isHexDigit))
        XCTAssertEqual(CoreAI.binaryUUID(me), uuid)
        let text = FileManager.default.temporaryDirectory.appendingPathComponent("not-a-binary-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: text) }
        try Data("not a binary at all, long enough to be read as a header".utf8).write(to: text)
        XCTAssertNil(CoreAI.binaryUUID(text))
    }
}
