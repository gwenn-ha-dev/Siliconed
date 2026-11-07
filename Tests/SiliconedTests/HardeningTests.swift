import Foundation
import XCTest
@testable import Siliconed

/// **What a user's file, a client or an import must never do**: download behind the user's back, freeze
/// the app on a regular expression, carry a NaN into a weight, leave a map behind a refusal. Files
/// written in memory or in a temporary folder: no byte of weights, no network.
final class HardeningTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-hardening-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// A safetensors of F32 tensors, with `metadata`.
    private func safetensors(_ tensors: [(String, [Int], [Float])], metadata: [String: String] = [:],
                             name: String = "lora.safetensors") throws -> String {
        var table: [OrderedJSON.Pair] = []
        if !metadata.isEmpty {
            table.append(.init("__metadata__", .object(metadata.sorted { $0.key < $1.key }.map { .init($0.key, .string($0.value)) })))
        }
        var data = Data(), at = 0
        for (name, shape, values) in tensors {
            let bytes = values.withUnsafeBufferPointer { Data(buffer: $0) }
            table.append(.init(name, .object([.init("dtype", .string("F32")), .init("shape", .list(shape.map { .integer($0) })),
                                              .init("data_offsets", .list([.integer(at), .integer(at + bytes.count)]))])))
            data.append(bytes); at += bytes.count
        }
        var json = Array(OrderedJSON.object(table).jsonText.utf8)
        while json.count % 8 != 0 { json.append(0x20) }
        var n = UInt64(json.count).littleEndian
        var file = withUnsafeBytes(of: &n) { Data($0) }
        file.append(contentsOf: json); file.append(data)
        let path = root.appendingPathComponent(name).path
        try file.write(to: URL(fileURLWithPath: path))
        return path
    }

    /// A Qwen-Image-2.1 LoRA by its names (`img_mlp.gate_up`), rank 2.
    private func qwenLoRA(alpha: Float? = nil, metadata: [String: String] = [:]) throws -> String {
        var tensors: [(String, [Int], [Float])] = [
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight", [2, 2], [1, 2, 3, 4]),
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight", [6, 2], (0..<12).map(Float.init)),
        ]
        if let alpha { tensors.append(("diffusion_model.transformer_blocks.0.img_mlp.gate_up.alpha", [], [alpha])) }
        return try safetensors(tensors, metadata: metadata)
    }

    private let qwenReference: [String: [Int]] = ["transformer_blocks.0.img_mlp.gate_layer.weight": [3, 2],
                                                  "transformer_blocks.0.img_mlp.proj.weight": [3, 2],
                                                  "transformer_blocks.0.img_mlp.out.weight": [2, 3]]

    // ── an import downloads nothing ────────────────────────────────────────────────────────────

    /// A LoRA for a family not installed is refused by name — the family to install first —, and
    /// nothing is downloaded: no `telechargements/`, no component, no map.
    func testAnImportForAFamilyNotInstalledDownloadsNothing() throws {
        let library = Library(root: root.appendingPathComponent("library"))
        let lora = try qwenLoRA()
        XCTAssertThrowsError(try ModelImport.importFile(lora, in: library)) {
            XCTAssertEqual($0 as? EngineError, .importNeedsFamily(file: "lora.safetensors", model: ModelCard.of(.qwenImage21).id))
        }
        let fm = FileManager.default
        XCTAssertFalse(fm.fileExists(atPath: library.root.appendingPathComponent("telechargements").path))
        XCTAssertFalse(fm.fileExists(atPath: library.root.appendingPathComponent("store/composants").path))
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: library.map("")).filter { $0.hasSuffix(".silicon") }, [])
    }

    /// Flattened kohya names (`lora_unet_…`) are compared with the families installed only; none
    /// installed is a refusal of the file, and a real failure reading a family's reference (a disk
    /// full) is that failure — never swallowed as « not this family ».
    func testKohyaNamesAreComparedWithTheInstalledFamiliesOnly() throws {
        let lora = try safetensors([("lora_unet_layers_0_attention_to_q.lora_down.weight", [2, 2], [1, 2, 3, 4]),
                                    ("lora_unet_layers_0_attention_to_q.lora_up.weight", [2, 2], [1, 2, 3, 4])])
        let map = root.appendingPathComponent("out.lora.silicon").path
        XCTAssertThrowsError(try ForgeLoRA.forge(file: lora, to: map, reference: { _ in nil })) {
            XCTAssertTrue($0 is Numerics.Failure, "\($0)")
        }
        XCTAssertThrowsError(try ForgeLoRA.forge(file: lora, to: map, reference: { _ in
            throw EngineError.diskFull(needed: nil, available: nil)
        })) { XCTAssertEqual($0 as? EngineError, .diskFull(needed: nil, available: nil)) }
        // Imposed on a family not installed: that family, by name.
        XCTAssertThrowsError(try ForgeLoRA.forge(file: lora, to: map, family: .zImage, reference: { _ in nil })) {
            XCTAssertEqual($0 as? EngineError, .importNeedsFamily(file: "lora.safetensors", model: ModelCard.of(.zImage).id))
        }
    }

    // ── alpha_pattern: a regular expression of the file ───────────────────────────────────────

    /// `.*.*.*…z` would backtrack exponentially in `^(.*\.)?(KEY)$` against a module that does not
    /// end in `z`: refused before any match, at once.
    func testAPathologicalAlphaPatternIsRefusedAtOnce() throws {
        let key = String(repeating: ".*", count: 40) + "z"
        let metadata = ["lora_adapter_metadata": #"{"lora_alpha": 4, "alpha_pattern": {""# + key + #"": 1}}"#]
        let begin = Date()
        XCTAssertThrowsError(try ForgeLoRA.PEFTScale.read(metadata)) { XCTAssertTrue("\($0)".contains("unbounded"), "\($0)") }
        let lora = try qwenLoRA(metadata: metadata)
        XCTAssertThrowsError(try ForgeLoRA.forge(file: lora, to: root.appendingPathComponent("p.lora.silicon").path,
                                                 reference: { _ in self.qwenReference }))
        XCTAssertLessThan(Date().timeIntervalSince(begin), 0.1)
        // What PEFT keys really are pass: a module's name, a suffix with one wildcard, a bounded repeat.
        for good in ["transformer_blocks.0.attn.to_q", ".*to_k", "transformer_blocks\\.[0-9]+\\.attn\\.to_v", "a{2,3}b"] {
            XCTAssertNoThrow(try ForgeLoRA.PEFTScale.checkPattern(good), good)
        }
        for bad in ["a+b+", ".*x.*", "a{2,}b*", "(a+)+"] {
            XCTAssertThrowsError(try ForgeLoRA.PEFTScale.checkPattern(bad), bad)
        }
        // Escaped and in a class, `*` and `+` are characters.
        XCTAssertEqual(ForgeLoRA.PEFTScale.unboundedQuantifiers(#"a\*b[+*].*"#), 1)
    }

    // ── values that are not numbers ───────────────────────────────────────────────────────────

    /// `NaN`, `Infinity` and `1e999` parse as JSON numbers (Python writes the first two): a
    /// `lora_alpha` or an `alpha_pattern` value of them is refused, as a `.alpha` tensor of them is.
    func testANonFiniteAlphaIsRefused() throws {
        for alpha in ["NaN", "Infinity", "-Infinity", "1e999"] {
            XCTAssertThrowsError(try ForgeLoRA.PEFTScale.read(["lora_adapter_metadata": #"{"lora_alpha": "# + alpha + "}"]), alpha)
            XCTAssertThrowsError(try ForgeLoRA.PEFTScale.read(
                ["lora_adapter_metadata": #"{"lora_alpha": 4, "alpha_pattern": {"to_q": "# + alpha + "}}"]), alpha)
        }
        XCTAssertEqual(try ForgeLoRA.PEFTScale.read(["lora_adapter_metadata": #"{"lora_alpha": 4}"#])?.alpha, 4)
        for alpha in [Float.nan, .infinity] {
            let lora = try qwenLoRA(alpha: alpha)
            XCTAssertThrowsError(try ForgeLoRA.forge(file: lora, to: root.appendingPathComponent("a.lora.silicon").path,
                                                     reference: { _ in self.qwenReference })) {
                XCTAssertTrue("\($0)".contains("not finite"), "\($0)")
            }
        }
        XCTAssertNoThrow(try ForgeLoRA.forge(file: try qwenLoRA(alpha: 2), to: root.appendingPathComponent("b.lora.silicon").path,
                                             reference: { _ in self.qwenReference }))
    }

    /// sd-scripts writes `.alpha` as an `I64` scalar when the network alpha was an `int` (the
    /// Anima LoRAs of HF): read as diffusers' `.item()`, α/r folded into up.
    func testAnIntegerAlphaIsRead() throws {
        let lora = root.appendingPathComponent("int-alpha.safetensors").path
        let f32 = { (v: [Float]) in v.withUnsafeBufferPointer { Data(buffer: $0) } }
        var four = Int64(4).littleEndian
        try ForgeDiT.writeSafetensors([
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.alpha", "I64", [], withUnsafeBytes(of: &four) { Data($0) }),
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight", "F32", [2, 2], f32([1, 2, 3, 4])),
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight", "F32", [6, 2], f32((0..<12).map(Float.init))),
        ], to: lora)
        let report = try ForgeLoRA.forge(file: lora, to: root.appendingPathComponent("i.lora.silicon").path,
                                         reference: { _ in self.qwenReference })
        XCTAssertTrue(report.notes.contains { $0.contains("α/r = 2") }, "\(report.notes)")
    }

    /// `transformer_blocks.N.attn.to_q` alone is Qwen-Image-2.1's and Krea 2's diffusers name: the
    /// installed shapes decide, and two families that both fit are no answer.
    func testNamesSharedByTwoFamiliesAreDecidedByTheShapes() throws {
        let lora = try safetensors([("transformer.transformer_blocks.0.attn.to_q.lora.down.weight", [2, 4], Array(repeating: 1, count: 8)),
                                    ("transformer.transformer_blocks.0.attn.to_q.lora.up.weight", [4, 2], Array(repeating: 1, count: 8))])
        let map = root.appendingPathComponent("s.lora.silicon").path
        let qwen = ["transformer_blocks.0.attn.to_q.weight": [4, 4]], other = ["transformer_blocks.0.attn.to_q.weight": [6, 6]]
        let report = try ForgeLoRA.forge(file: lora, to: map, reference: { f in
            f == .qwenImage21 ? qwen : f == .krea2 ? other : nil })
        XCTAssertEqual(report.family, .qwenImage21)
        XCTAssertThrowsError(try ForgeLoRA.forge(file: lora, to: map, reference: { f in
            f == .qwenImage21 || f == .krea2 ? qwen : nil })) { XCTAssertTrue("\($0)".contains("not recognized"), "\($0)") }
    }

    /// A LoRA that also trains the text encoder is refused by what it is, not as unknown kohya names.
    func testATextEncoderLoRAIsRefusedAsSuch() throws {
        let lora = try safetensors([("lora_te_layers_0_self_attn_q_proj.lora_down.weight", [2, 2], [1, 2, 3, 4]),
                                    ("lora_te_layers_0_self_attn_q_proj.lora_up.weight", [2, 2], [1, 2, 3, 4]),
                                    ("lora_unet_blocks_0_self_attn_q_proj.lora_down.weight", [2, 2], [1, 2, 3, 4]),
                                    ("lora_unet_blocks_0_self_attn_q_proj.lora_up.weight", [2, 2], [1, 2, 3, 4])])
        XCTAssertThrowsError(try ForgeLoRA.forge(file: lora, to: root.appendingPathComponent("t.lora.silicon").path,
                                                 reference: { _ in [:] })) { XCTAssertTrue("\($0)".contains("text encoder"), "\($0)") }
    }

    // ── memory: a convrot DiT imported ────────────────────────────────────────────────────────

    /// **An imported int8 convrot DiT widens through a scratch outside the arenas** — Z-Image's and
    /// Anima's as Qwen-Image-2.1's and Krea 2's: the denoising charges its largest weight's
    /// (`Widen.rotationScratchBytes`), an unrotated map nothing.
    func testAConvrotDiTChargesItsScratchForZImageAndAnima() throws {
        let workers = Widen.widenWorkers
        func largest(_ shapes: [[Int]]) -> Int {
            shapes.map { Widen.rotationScratchBytes(rows: $0[0], columns: $0[1], group: 256, workers: workers) }.max()!
        }
        let z = MemoryFixtures.zImageMaps(bytesPerValue: 1, rotated: true), zPlain = MemoryFixtures.zImageMaps(bytesPerValue: 1)
        XCTAssertEqual(z.ditRotationScratch, largest([[3840, 3840], [3840, 10240], [10240, 3840], [256, 15360]]))
        XCTAssertGreaterThan(z.ditRotationScratch, 0)
        XCTAssertEqual(zPlain.ditRotationScratch, 0)
        let request = MemoryFixtures.zImageRequest(1024, 1024)
        let a = try XCTUnwrap(MemoryNeed.zImage(request, maps: z)), b = try XCTUnwrap(MemoryNeed.zImage(request, maps: zPlain))
        XCTAssertEqual(a.phases[1].bytes("DiT rotation scratch"), z.ditRotationScratch)
        XCTAssertEqual(a.phases[1].bytes - b.phases[1].bytes, z.ditRotationScratch)

        let anima = MemoryFixtures.animaMaps(ditBytesPerValue: 1, rotated: true), animaPlain = MemoryFixtures.animaMaps(ditBytesPerValue: 1)
        XCTAssertEqual(anima.ditRotationScratch,
                       largest([[2048, 2048], [1024, 2048], [2048, 8192], [8192, 2048], [2048, 256], [256, 6144]]))
        XCTAssertEqual(animaPlain.ditRotationScratch, 0)
        let ar = MemoryFixtures.animaRequest(1024, 1024, maps: anima), ap = MemoryFixtures.animaRequest(1024, 1024, maps: animaPlain)
        let c = MemoryNeed.anima(ar, maps: anima), d = MemoryNeed.anima(ap, maps: animaPlain)
        XCTAssertEqual(c.phases[1].bytes("DiT rotation scratch"), anima.ditRotationScratch)
        XCTAssertEqual(c.phases[1].bytes - d.phases[1].bytes, anima.ditRotationScratch)
    }

    /// **An SQ weight's widened route also holds the integers it decodes to** (`K · N` bytes, outside every arena): the same
    /// header with `sq5` charges that on top of the int8 convrot's scratch.
    func testAnSQWeightChargesItsIntegersBesideTheRotationScratch() {
        func header(_ dtype: String) -> [String: Any] {
            ["tensors": ["w": ["dtype": dtype, "shape": [3840, 10240], "rotation": ["group": 256]]]]
        }
        let int8 = Widen.rotationScratchBytes(header: header("int8"))
        XCTAssertGreaterThan(int8, 0)
        XCTAssertEqual(Widen.rotationScratchBytes(header: header("sq5")), int8 + Arena.capacity(for: 3840 * 10240))
        XCTAssertEqual(Widen.rotationScratchBytes(header: header("sq4")), int8 + Arena.capacity(for: 3840 * 10240))
    }
}
