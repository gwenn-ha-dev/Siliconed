import CryptoKit
import XCTest
@testable import Siliconed

/// **The forge, without a byte of weights**: the pure rules — names, roundings, headers — that the
/// map comparison of the developer checks then judges on the real files.
final class ForgeTests: XCTestCase {
    /// A map's header re-reads the published `config.json`: key order and numbers preserved.
    func testOrderedJSONKeepsOrderAndNumbers() throws {
        let text = #"{"b": 1, "a": [1.0, 1e-05, null, true], "\u00e9": "x\"y", "c": {"z": 256.0}}"#
        let v = try OrderedJSON.parse(Data(text.utf8))
        XCTAssertEqual(v.pairs?.map(\.key), ["b", "a", "é", "c"])
        XCTAssertEqual(v.jsonText, #"{"b": 1, "a": [1.0, 1e-05, null, true], "\u00e9": "x\"y", "c": {"z": 256.0}}"#)
        XCTAssertEqual(OrderedJSON.reprPython(256), "256.0")
        XCTAssertEqual(OrderedJSON.reprPython(1e-5), "1e-05")
        XCTAssertEqual(OrderedJSON.reprPython(52.84136962890625), "52.84136962890625")
    }

    /// fp32 → bf16 rounded to nearest even, like `torch.Tensor.to(torch.bfloat16)`.
    func testBF16RoundingIsTorchs() {
        let x: [Float] = [1, 1 + 1.0 / 256, 1 + 3.0 / 512, -2.5, .nan]
        var h = [UInt16](repeating: 0, count: x.count)
        let inexact = x.withUnsafeBufferPointer { s in h.withUnsafeMutableBufferPointer {
            Numerics.toBFloat16(s.baseAddress!, count: s.count, to: $0.baseAddress!) } }
        XCTAssertEqual(h, [0x3F80, 0x3F80, 0x3F81, 0xC020, 0x7FC0])   // 1+2⁻⁸: tie → even
        XCTAssertEqual(inexact, 2)
    }

    func testFP8E4M3ReadsExactly() {
        XCTAssertEqual(Numerics.e4m3[0x38], 1)          // 0 0111 000
        XCTAssertEqual(Numerics.e4m3[0x7E], 448)        // the largest finite
        XCTAssertEqual(Numerics.e4m3[0x01], pow(2, -9)) // the smallest subnormal
        XCTAssertTrue(Numerics.e4m3[0x7F].isNaN)
    }

    /// Krea 2's original naming (ComfyUI, `krea-ai/krea-2`) → diffusers, checked on real tensors
    /// at import; here, the rules.
    func testTheNamesOfKrea2() {
        XCTAssertEqual(Recipes.krea2("blocks.3.attn.wq.weight", shape: [6144, 6144])?.0, "transformer_blocks.3.attn.to_q.weight")
        XCTAssertEqual(Recipes.krea2("blocks.3.attn.qknorm.knorm.scale", shape: [128])?.0, "transformer_blocks.3.attn.norm_k.weight")
        XCTAssertEqual(Recipes.krea2("blocks.3.mod.lin", shape: [36864])?.1, [6, 6144])
        XCTAssertEqual(Recipes.krea2("txtfusion.refiner_blocks.1.mlp.up.weight", shape: [6912, 2560])?.0,
                       "text_fusion.refiner_blocks.1.ff.up.weight")
        XCTAssertEqual(Recipes.krea2("txtmlp.0.scale", shape: [2560])?.0, "txt_in.norm.weight")
        XCTAssertNil(Recipes.krea2("blocks.3.inconnu.weight", shape: [1]))
    }

    func testTheFamilyIsRecognizedByItsNames() {
        XCTAssertEqual(Recipes.family(fromNames: ["model.diffusion_model.layers.0.attention.qkv.weight",
                                                 "model.diffusion_model.cap_embedder.1.weight"]), .zImage)
        XCTAssertEqual(Recipes.family(fromNames: ["blocks.0.attn.wq.weight"]), .krea2)
        XCTAssertEqual(Recipes.family(fromNames: ["net.blocks.0.self_attn.q_proj.weight"]), .anima)
        XCTAssertNil(Recipes.family(fromNames: ["encoder.down.0.weight"]))
    }

    /// The trainer's name and the trigger words, from the metadata of three published LoRAs
    /// (their headers, copied): ai-toolkit with a trigger, Civitai's trainer without one, kohya
    /// with a captioned dataset.
    func testWhatALoRASaysOfItself() {
        let oiled = ["name": "my_first_lora_v1_copy_copy_copy_copy", "ss_output_name": "my_first_lora_v1_copy_copy_copy_copy",
                     "ss_tag_frequency": #"{"1_OiledSkin": {"OiledSkin": 1}}"#]
        XCTAssertEqual(ForgeLoRA.trainerName(oiled), "my_first_lora_v1_copy_copy_copy_copy")
        XCTAssertEqual(ForgeLoRA.triggers(oiled), ["OiledSkin"])

        let incaseKrea = ["name": "epoch_1", "modelspec.title": "", "ss_output_name": "training_1356263-20260704185321882",
                          "ss_tag_frequency": #"{"1_": {"": 1}}"#]
        XCTAssertNil(ForgeLoRA.trainerName(incaseKrea), "an empty title and an epoch counter are no names")
        XCTAssertEqual(ForgeLoRA.triggers(incaseKrea), [], "ai-toolkit without a trigger")

        let incaseAnima = ["modelspec.title": "InCase", "ss_output_name": "InCase",
                           "ss_tag_frequency": #"{"Cleaned": {"interracial": 5, "1girl": 51, "hetero": 39, "dark_skin": 41}}"#]
        XCTAssertEqual(ForgeLoRA.trainerName(incaseAnima), "InCase")
        XCTAssertEqual(ForgeLoRA.triggers(incaseAnima), [], "a captioned folder: no tag is singled out")

        XCTAssertEqual(ForgeLoRA.triggers(["ss_tag_frequency": #"{"10_sks woman": {"sks woman": 20}, "1_x": {"b": 1, "a": 1, "c": 3}}"#]),
                       ["sks woman", "c", "a", "b"], "caption-less folders: folder order, then falling count")
        XCTAssertEqual(ForgeLoRA.triggers(["modelspec.trigger_phrase": " kelvoran style ", "ss_tag_frequency": #"{"1_x": {"x": 1}}"#]),
                       ["kelvoran style"], "an explicit field wins")
        XCTAssertEqual(ForgeLoRA.triggers(["trigger_words": #"["a", "b", "a", ""]"#]), ["a", "b"])
        XCTAssertEqual(ForgeLoRA.triggers(["ss_tag_frequency": "not json"]), [])
    }

    func testTheLoRAKeys() throws {
        XCTAssertEqual(try ForgeLoRA.parseKey("diffusion_model.layers.0.attention.to_q.lora_A.default.weight")?.module,
                       "layers.0.attention.to_q")
        XCTAssertEqual(try ForgeLoRA.parseKey("transformer.img_in.lora_B.weight")?.role, "up")
        XCTAssertEqual(try ForgeLoRA.parseKey("lora_unet_layers_0_attention_qkv.alpha")?.role, "alpha")
        XCTAssertThrowsError(try ForgeLoRA.parseKey("diffusion_model.blocks.0.attn.wq.dora_scale"))
        XCTAssertEqual(try ForgeLoRA.targets("layers.2.attention.qkv", family: .zImage).map(\.module),
                       ["layers.2.attention.to_q", "layers.2.attention.to_k", "layers.2.attention.to_v"])
        XCTAssertEqual(try ForgeLoRA.targets("blocks.1.attn.wo", family: .krea2).map(\.module), ["transformer_blocks.1.attn.to_out.0"])
        XCTAssertEqual(try ForgeLoRA.targets("blocks.1.self_attn.q_proj", family: .anima).map(\.module), ["transformer_blocks.1.attn1.to_q"])
    }

    // ── Qwen-Image-2.1 ─────────────────────────────────────────────────────────────────────

    /// The DiT is recognized by its names; a LoRA in PEFT's naming (`transformer.` prefix) by
    /// its modules — the modulation and the σ embedding included.
    func testQwenImage21NamesAndTurboKeys() throws {
        XCTAssertEqual(Recipes.family(fromNames: ["transformer_blocks.0.img_mlp.gate_layer.weight",
                                                 "txt_in.text_norm.weight", "modulation.1.weight"]), .qwenImage21)
        let k = try XCTUnwrap(try ForgeLoRA.parseKey("transformer.modulation.1.lora_A.weight"))
        XCTAssertEqual(k.module, "modulation.1")
        XCTAssertEqual(k.role, "down")
        XCTAssertEqual(try ForgeLoRA.parseKey("transformer.transformer_blocks.31.attn.to_out.0.lora_B.weight")?.role, "up")
        for m in ["modulation.1", "time_text_embed.timestep_embedder.linear_1", "transformer_blocks.3.img_mlp.out"] {
            XCTAssertEqual(try ForgeLoRA.targets(m, family: .qwenImage21).map(\.module), [m])
        }
    }

    /// **ai-toolkit's fused `img_mlp.gate_up`**, on a miniature Qwen (d = 2, MLP 3, r = 2): `up`
    /// is cut by rows, `[0, 3)` → `gate_layer` (the SiLU side), `[3, 6)` → `proj`, `down` shared —
    /// and the PEFT scale (`lora_adapter_metadata`, prefixed `transformer.`, an `alpha_pattern`)
    /// folded into `up`. Values are small integers: bf16 keeps them exactly.
    func testQwenImage21FusedGateUpAndPEFTScale() throws {
        XCTAssertEqual(try ForgeLoRA.targets("transformer_blocks.0.img_mlp.gate_up", family: .qwenImage21,
                                             ref: ["transformer_blocks.0.img_mlp.gate_layer.weight": [3, 2],
                                                   "transformer_blocks.0.img_mlp.proj.weight": [3, 2]]).map(\.module),
                       ["transformer_blocks.0.img_mlp.gate_layer", "transformer_blocks.0.img_mlp.proj"])

        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("forge-gate-up-\(UUID().uuidString)").path
        try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: folder) }
        let a: [Float] = [1, 2, 3, 4]                                  // A [r = 2, k = 2]
        let b: [Float] = (0..<12).map { (i: Int) -> Float in Float(i + 1) }                  // B [2h = 6, r = 2]
        let aq: [Float] = [1, -1, 2, 0], bq: [Float] = [2, 4, 6, 8]      // to_q: A [2, 2], B [2, 2]
        let tensors: [(String, [Int], [Float])] = [
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight", [2, 2], a),
            ("diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight", [6, 2], b),
            ("diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight", [2, 2], aq),
            ("diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight", [2, 2], bq),
        ]
        let peft = #"{"transformer.lora_alpha": 4, "transformer.r": 2, "transformer.alpha_pattern": {"attn.to_q": 1}, "#
            + #""text_encoder.lora_alpha": 64, "transformer.use_rslora": false}"#
        var table: [OrderedJSON.Pair] = [.init("__metadata__", .object([.init("lora_adapter_metadata", .string(peft))]))]
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
        let source = folder + "/mini.safetensors", map = folder + "/mini.lora.silicon"
        try file.write(to: URL(fileURLWithPath: source))

        let ref: [String: [Int]] = ["transformer_blocks.0.attn.to_q.weight": [2, 2],
                                    "transformer_blocks.0.img_mlp.gate_layer.weight": [3, 2],
                                    "transformer_blocks.0.img_mlp.proj.weight": [3, 2],
                                    "transformer_blocks.0.img_mlp.out.weight": [2, 3]]
        let report = try ForgeLoRA.forge(file: source, to: map, reference: { _ in ref })
        XCTAssertEqual(report.family, .qwenImage21, "recognized by `gate_up` alone")
        XCTAssertEqual(report.modules, 3)
        XCTAssertTrue(report.notes.contains { $0.contains("gate_up") }, "\(report.notes)")

        let m = try Artifact(path: map)
        // The published file's sha256, what Civitai credits the LoRA by (`Infotext`).
        XCTAssertEqual((m.header["source"] as? [String: Any])?["sha256"] as? String,
                       SHA256.hash(data: file).map { String(format: "%02x", $0) }.joined())
        func values(_ name: String) throws -> [Float] {
            let t = try XCTUnwrap(m.tensors[name], name)
            XCTAssertEqual(t.dtype, .bfloat16)
            var out = [Float](repeating: 0, count: t.count)
            try out.withUnsafeMutableBufferPointer { try Numerics.toFloat32(m.pointer(name)!, dtype: "BF16", count: t.count, to: $0.baseAddress!) }
            return out
        }
        // `.down` is Aᵀ [k, r]; `.up` is (scale·B)ᵀ [r, n].
        let aT: [Float] = [1, 3, 2, 4]
        XCTAssertEqual(try values("transformer_blocks.0.img_mlp.gate_layer.down"), aT)
        XCTAssertEqual(try values("transformer_blocks.0.img_mlp.proj.down"), aT)
        // α/r = 4/2 = 2. Gate rows 0–2 of B: (1,2) (3,4) (5,6); up rows 3–5: (7,8) (9,10) (11,12).
        XCTAssertEqual(try values("transformer_blocks.0.img_mlp.gate_layer.up"), [2, 6, 10, 4, 8, 12])
        XCTAssertEqual(try values("transformer_blocks.0.img_mlp.proj.up"), [14, 18, 22, 16, 20, 24])
        // `alpha_pattern` "attn.to_q": α = 1, ×0.5.
        XCTAssertEqual(try values("transformer_blocks.0.attn.to_q.up"), [1, 3, 2, 4])
    }

    /// `lora_adapter_metadata` as diffusers writes it: bare or `transformer.` keys, a text encoder's
    /// ignored, PEFT's pattern match (`(.*\.)?(key)$`, first key in the JSON's order), rsLoRA.
    func testPEFTAdapterMetadata() throws {
        XCTAssertNil(try ForgeLoRA.PEFTScale.read(["format": "pt"]))
        XCTAssertNil(try ForgeLoRA.PEFTScale.read(["lora_adapter_metadata": #"{"text_encoder.lora_alpha": 8}"#]))
        let bare = try XCTUnwrap(try ForgeLoRA.PEFTScale.read(["lora_adapter_metadata": #"{"lora_alpha": 32, "r": 16}"#]))
        XCTAssertEqual(bare.scale("transformer_blocks.0.attn.to_q", r: 16), 2)
        let turbo = try XCTUnwrap(try ForgeLoRA.PEFTScale.read(["lora_adapter_metadata":
            #"{"transformer.alpha_pattern": {}, "transformer.lora_alpha": 256, "transformer.r": 256, "transformer.use_rslora": false}"#]))
        XCTAssertEqual(turbo.scale("modulation.1", r: 256), 1, "α = r: exactly 1, the map unchanged")
        let p = try XCTUnwrap(try ForgeLoRA.PEFTScale.read(["lora_adapter_metadata":
            #"{"lora_alpha": 8, "alpha_pattern": {"transformer_blocks.1.attn.to_q": 2, "to_q": 4}, "use_rslora": true}"#]))
        XCTAssertEqual(p.scale("transformer_blocks.1.attn.to_q", r: 4), 1)       // 2/√4, first key wins
        XCTAssertEqual(p.scale("transformer_blocks.2.attn.to_q", r: 4), 2)       // 4/√4
        XCTAssertEqual(p.scale("transformer_blocks.2.attn.add_q_proj", r: 4), 4) // 8/√4: `to_q` must be whole
        XCTAssertEqual(p.scale("transformer_blocks.2.attn.xto_q", r: 4), 4)
        XCTAssertThrowsError(try ForgeLoRA.PEFTScale.read(["lora_adapter_metadata": #"{"lora_alpha": 8, "use_dora": true}"#]))
    }

    /// The DiT's file order is its execution: inputs and the shared modulation, the 32 blocks,
    /// the output. The zero-centered `text_norm` alone goes to fp32 (`+ 1` folded).
    func testQwenImage21DiTOrderAndDTypes() {
        let names = ["proj_out.weight", "transformer_blocks.10.attn.to_q.weight", "modulation.1.weight",
                     "transformer_blocks.2.img_mlp.out.weight", "norm_out.linear.weight", "img_in.weight"]
        XCTAssertEqual(ForgeDiT.order(names, family: .qwenImage21),
                       ["img_in.weight", "modulation.1.weight", "transformer_blocks.2.img_mlp.out.weight",
                        "transformer_blocks.10.attn.to_q.weight", "norm_out.linear.weight", "proj_out.weight"])
        XCTAssertEqual(ForgeDiT.qwenZeroCentered, ["txt_in.text_norm.weight"])
        XCTAssertEqual(ForgeDiT.dtype("transformer_blocks.0.attn.norm_q.weight", family: .qwenImage21), .bfloat16)
    }

    /// The vision tower's fused `qkv`: q and k permuted per head, v untouched — on a miniature
    /// (2 heads of 4): the source row of each forged row.
    func testQwen3VLVisionQKVPermutation() {
        XCTAssertEqual(ForgeText.qkOrder(rows: 24, head: 4),
                       [0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15] + Array(16..<24))
    }

    /// The encoder's order: the vision tower first, each deepstack merger right after the block
    /// it taps, the final merger; then `embed_tokens` and the layers.
    func testQwen3VLEncoderOrder() throws {
        let vision = try OrderedJSON.parse(Data(#"{"deepstack_visual_indexes": [8, 16, 24]}"#.utf8))
        let names = ["layers.0.mlp.up_proj.weight", "embed_tokens.weight", "visual.merger.linear_fc1.weight",
                     "visual.deepstack_merger_list.1.norm.weight", "visual.blocks.17.norm1.weight",
                     "visual.blocks.16.norm1.weight", "visual.pos_embed.weight", "visual.patch_embed.proj.weight"]
        let order = names.sorted { a, b in
            let x = ForgeText.rank(a, vision: vision), y = ForgeText.rank(b, vision: vision)
            return (x.0, x.1, x.2, a) < (y.0, y.1, y.2, b)
        }
        XCTAssertEqual(order, ["visual.patch_embed.proj.weight", "visual.pos_embed.weight", "visual.blocks.16.norm1.weight",
                               "visual.deepstack_merger_list.1.norm.weight", "visual.blocks.17.norm1.weight",
                               "visual.merger.linear_fc1.weight", "embed_tokens.weight", "layers.0.mlp.up_proj.weight"])
    }

    /// Every file of the Standard from the official Turbo at its pinned revision — DiT shards pinned by
    /// sha256, the encoder in one file, the schedule in `model_index.json`; no turbo LoRA, no file of
    /// the base model.
    func testQwenImage21Installation() throws {
        let turbo = "Qwen/Qwen-Image-2.1-Turbo@d65dbc9a7e8f6b5479e33dee6030eaab2a906509"
        let standard = Installer.components(.qwenImage21) + Installer.encoder(.qwenImage21) + Installer.dit(.qwenImage21)
            + Installer.licenseFiles(.qwenImage21)
        XCTAssertEqual(Set(standard.map { "\($0.repository.name)@\($0.repository.revision)" }), [turbo])
        XCTAssertEqual(Installer.encoder(.qwenImage21).map(\.path), ["text_encoder/config.json", "text_encoder/model.safetensors"])
        XCTAssertEqual(Installer.dit(.qwenImage21).compactMap(\.sha256),
                       ["6cccd922767f01694461bdcf1f34ea7b771f3442e4d5e338ea9aa55277431cc0",
                        "69f53ebb063d2f606bdaff7e22e0e2144f28d978d73dc75e6e4293fc5982eab5"])
        XCTAssertEqual(Set(Installer.components(.qwenImage21).compactMap(\.destination)),
                       ["processor/tokenizer.json", "processor/tokenizer_config.json", "processor/chat_template.jinja",
                        "processor/processor_config.json", "vae.safetensors", "vae.json", "transformer.json",
                        "scheduler.json", "model_index.json"])
        XCTAssertEqual(Installer.licenseURLs(.qwenImage21).count, 1, "one license file: the Turbo's")
        XCTAssertEqual(Family.qwenImage21.ditMap, "qwen-image-2.1-turbo-dit.v0.silicon")
        XCTAssertEqual(Family(ditKind: "qwen-image-2.1-turbo-dit"), .qwenImage21)
        XCTAssertNil(Family(ditKind: Family.supersededQwenImage21Kind), "a base-model map is not the Turbo")

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-qwen-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let b = Library(root: root)
        try FileManager.default.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
        try place(.qwenImage21, in: b)
        let i = Installer(library: b)
        XCTAssertTrue(i.isReady(.qwenImage21))
        XCTAssertTrue(i.supersededFiles(.qwenImage21).isEmpty)
        try b.uninstall(.qwenImage21)
        XCTAssertEqual(b.occupancy().total, 0)
    }

    /// **A superseded installation** (base model + third-party turbo LoRA) is recognized and not ready:
    /// its DiT map and its whole components folder are what the next installation removes first; the
    /// encoder map, the same tensors, stays. Uninstalling removes them too.
    func testASupersededQwenInstallationIsRecognized() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-qwen-old-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let b = Library(root: root)
        try FileManager.default.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
        try place(.qwenImage21, in: b, dit: false)
        let i = Installer(library: b), folder = i.componentsFolder(.qwenImage21)
        try Data(repeating: 4, count: 10).write(to: URL(fileURLWithPath: folder + "/viggle-turbo-v0.3.lora.silicon"))
        try Data(repeating: 3, count: 1000).write(to: URL(fileURLWithPath: b.map(Family.supersededQwenImage21Maps[0])))
        XCTAssertFalse(i.isReady(.qwenImage21))
        XCTAssertEqual(Set(i.supersededFiles(.qwenImage21)), [b.map(Family.supersededQwenImage21Maps[0]), folder])
        XCTAssertFalse(i.supersededFiles(.qwenImage21).contains(b.map(Family.qwenImage21.encoderMap)))
        XCTAssertNotNil(ModelCard.qwenImage21.missing(in: b))
        XCTAssertEqual(try XCTUnwrap(b.occupancy().families.first { $0.family == .qwenImage21 }).dit, 1000,
                       "the superseded DiT map is counted until it goes")
        try b.uninstall(.qwenImage21)
        XCTAssertEqual(b.occupancy().total, 0, "nothing of it is kept")
        // The base's `preprocessor_config.json` alone marks it as well.
        try place(.qwenImage21, in: b)
        XCTAssertTrue(i.supersededFiles(.qwenImage21).isEmpty)
        try Data().write(to: URL(fileURLWithPath: folder + "/processor/preprocessor_config.json"))
        XCTAssertFalse(i.presentComponents(.qwenImage21))
    }

    /// **A superseded Compact or Light** (a third party's 8-bit or GGUF DiT of the base model, under
    /// the same turbo LoRA) is recognized the same way: its DiT map and the components folder go, the
    /// Compact's 8-bit encoder stays (the Turbo's Compact and Light read it), and the version is read
    /// from the old DiT map, for the reinstallation to keep it — a Light stays a Light.
    func testASupersededQwenCompactOrLightIsRecognized() throws {
        for (v, old) in [(Variant.compact, "qwen-image-2.1-dit.compact.v0.silicon"), (.light, "qwen-image-2.1-dit.light.v0.silicon")] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-qwen-old-\(v)-\(UUID())")
            defer { try? FileManager.default.removeItem(at: root) }
            let b = Library(root: root)
            try FileManager.default.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
            try place(.qwenImage21, in: b, dit: false, variant: v)
            let i = Installer(library: b), folder = i.componentsFolder(.qwenImage21)
            try Data(repeating: 4, count: 10).write(to: URL(fileURLWithPath: folder + "/viggle-turbo-v0.3.lora.silicon"))
            try Data(repeating: 3, count: 1000).write(to: URL(fileURLWithPath: b.map(old)))
            XCTAssertTrue(Family.supersededQwenImage21Maps.contains(old))
            XCTAssertFalse(i.isReady(.qwenImage21))
            XCTAssertEqual(Set(i.supersededFiles(.qwenImage21)), [b.map(old), folder], v.rawValue)
            XCTAssertFalse(i.supersededFiles(.qwenImage21).contains(b.map(Family.qwenImage21.encoderMap(v))))
            XCTAssertEqual(i.supersededVariant(.qwenImage21), v)
            XCTAssertEqual(try XCTUnwrap(b.occupancy().families.first { $0.family == .qwenImage21 }).dit, 1000,
                           "the superseded DiT map is counted until it goes")
            // The old DiT map alone marks it too (a components folder already redone).
            try FileManager.default.removeItem(atPath: folder + "/viggle-turbo-v0.3.lora.silicon")
            XCTAssertEqual(Set(i.supersededFiles(.qwenImage21)), [b.map(old), folder])
            try b.uninstall(.qwenImage21)
            XCTAssertEqual(b.occupancy().total, 0, "nothing of it is kept")
        }
    }

    /// **Replaced weights** (`Family.supersededVersionMaps`): Z-Image's and Anima's Compact until 0.20 (a GGUF Q8_0 DiT map,
    /// `.compact.v0`, replaced by the int8 convrot `.compact.v1`) and every Light until 0.21 (a GGUF Q4_K_M, `.light.v0`, replaced
    /// by our SQ 4-bit `.light.v1`); Anima's with its adapter. The old map is recognized, the family is not ready, the next
    /// installation keeps the version it was; its encoder and its components stay (the new version's own); uninstalling removes
    /// it too. One version lives.
    func testASupersededVersionIsRecognized() throws {
        let cases: [(Family, Variant)] = [(.zImage, .compact), (.anima, .compact)] + [Family.zImage, .anima, .qwenImage21, .krea2].map { ($0, .light) }
        for (f, v) in cases {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-old-\(v)-\(f.rawValue)-\(UUID())")
            defer { try? FileManager.default.removeItem(at: root) }
            let b = Library(root: root)
            try FileManager.default.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
            try place(f, in: b, dit: false, variant: v)
            let i = Installer(library: b), folder = i.componentsFolder(f)
            let old = try XCTUnwrap(f.supersededVersionMaps.first { $0.variant == v }).map
            let suffix = v == .compact ? ".compact" : ".light"
            XCTAssertNotEqual(old, f.ditMap(v))
            XCTAssertTrue(old.hasSuffix(suffix + ".v0.silicon") && f.ditMap(v).hasSuffix(suffix + ".v1.silicon"), "\(f.rawValue) \(v)")
            XCTAssertTrue(i.supersededFiles(f).isEmpty, "nothing superseded before the old map is there")
            try Data(repeating: 3, count: 1000).write(to: URL(fileURLWithPath: b.map(old)))
            var expected = [b.map(old)]
            if f == .anima {
                let adapter = adapterPath(fromMap: b.map(old))
                try Data(repeating: 5, count: 200).write(to: URL(fileURLWithPath: adapter))
                expected.append(adapter)
            }
            XCTAssertEqual(Set(i.supersededFiles(f)), Set(expected), f.rawValue)
            XCTAssertFalse(i.supersededFiles(f).contains(b.map(f.encoderMap(v))), "the encoder is the new version's too")
            XCTAssertFalse(i.supersededFiles(f).contains(folder), "the components are the new version's too")
            XCTAssertFalse(i.isReady(f), "the old map is never rendered")
            XCTAssertEqual(i.supersededVariant(f), v, "the reinstallation keeps the version")
            XCTAssertEqual(try XCTUnwrap(b.occupancy().families.first { $0.family == f }).dit, f == .anima ? 1200 : 1000,
                           "the superseded map is counted until it goes")
            // Once the new version is forged, nothing is superseded any more — the old map was removed first.
            for path in expected { try FileManager.default.removeItem(atPath: path) }
            try place(f, in: b, variant: v)
            XCTAssertTrue(i.supersededFiles(f).isEmpty)
            XCTAssertTrue(i.isReady(f))
            XCTAssertEqual(b.ditMap(f), f.ditMap(v))
            // An uninstallation removes an old map left beside it.
            try Data(repeating: 3, count: 1000).write(to: URL(fileURLWithPath: b.map(old)))
            try b.uninstall(f)
            XCTAssertEqual(b.occupancy().total, 0, "nothing of it is kept")
        }
        XCTAssertTrue(Family.mossImage.supersededVersionMaps.isEmpty)
        XCTAssertEqual(Family.krea2.supersededVersionMaps.map(\.variant), [.light], "Krea's Compact never changed")
        XCTAssertEqual(Family.qwenImage21.supersededVersionMaps.map(\.variant), [.light])
        XCTAssertEqual(Set(Family.qwenImage21.supersededMaps), Set(Family.supersededQwenImage21Maps + ["qwen-image-2.1-turbo-dit.light.v0.silicon"]))
    }

    /// **An imported Qwen map of the superseded kind is removed**, not left out of sight: the catalog
    /// would not list it (no family has that kind) and the engine refuses it. Its map and its
    /// adapter go, with a journal line; an import of the Turbo's kind, and one of another family, stay.
    func testASupersededImportIsRemoved() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-old-import-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let b = Library(root: root)
        try FileManager.default.createDirectory(at: b.imported, withIntermediateDirectories: true)
        func map(_ file: String, kind: String, name: String) throws -> String {
            let path = b.imported.appendingPathComponent(file).path
            let t = MapWriter.Tensor(name: "w", shape: [4], dtype: .bfloat16, transposed: nil) { [0, 0, 0, 0] }
            _ = try MapWriter.write(to: path, tensors: [t], header: { _ in
                [.init("format", 1), .init("page", .integer(MapWriter.page)), .init("kind", .string(kind)), .init("nom", .string(name))]
            })
            return path
        }
        let old = try map("old-qwen.silicon", kind: Family.supersededQwenImage21Kind, name: "Old Qwen")
        try Data(repeating: 1, count: 10).write(to: URL(fileURLWithPath: adapterPath(fromMap: old)))
        let turbo = try map("new-qwen.silicon", kind: Family.qwenImage21.ditKind, name: "New Qwen")
        let z = try map("mine.silicon", kind: Family.zImage.ditKind, name: "Mine")

        var lines: [String] = []
        XCTAssertEqual(b.removeSupersededImports { lines.append($0) }, [old])
        XCTAssertEqual(lines.count, 1)
        XCTAssertTrue(lines[0].contains("Old Qwen"), lines[0])
        for gone in [old, adapterPath(fromMap: old)] { XCTAssertFalse(FileManager.default.fileExists(atPath: gone), gone) }
        for kept in [turbo, z] { XCTAssertTrue(FileManager.default.fileExists(atPath: kept), kept) }
        XCTAssertEqual(Set(b.importedCards().map(\.name)), ["New Qwen", "Mine"])

        // The listing does it by itself.
        let again = try map("old-again.silicon", kind: Family.supersededQwenImage21Kind, name: "Again")
        XCTAssertEqual(b.importedCards().count, 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: again))
        XCTAssertTrue(b.removeSupersededImports { _ in }.isEmpty)
    }

    func testTheNameOfAnImport() {
        XCTAssertEqual(ModelImport.slug("Mon Modèle v2.1 (fp8)"), "mon-modele-v2-1-fp8")
    }

    // The H0 gate (an fp8 import refused) is lifted by H1a: `QuantizedMapTests` checks
    // that an fp8 checkpoint is read 8-bit (`testAnFP8CheckpointIsAccepted`).

    // ── managing the library ───────────────────────────────────────────────────────────────

    /// A family installed for show: files of a few bytes at the right paths.
    func place(_ f: Family, in b: Library, dit: Bool = true, variant: Variant = .standard) throws {
        let i = Installer(library: b)
        let fm = FileManager.default
        for c in Installer.components(f) + [Installer.File(repository: .init(name: "", revision: ""), path: "", destination: "dit.json")] {
            let path = i.componentsFolder(f) + "/" + c.destination!
            try fm.createDirectory(atPath: (path as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
            try Data(repeating: 1, count: 10).write(to: URL(fileURLWithPath: path))
        }
        try Data(repeating: 2, count: 100).write(to: URL(fileURLWithPath: b.map(f.encoderMap(variant))))
        if dit { try Data(repeating: 3, count: 1000).write(to: URL(fileURLWithPath: b.map(f.ditMap(variant)))) }
    }

    /// **Uninstalling does not break the neighbor**: Z-Image's DiT, encoder and components go,
    /// Krea 2's stay. The inventory counts each family's maps once.
    func testUninstallSparesTheNeighbor() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-gestion-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let b = Library(root: root)
        try FileManager.default.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
        try place(.zImage, in: b)
        try place(.krea2, in: b)

        let o = b.occupancy()
        let z = try XCTUnwrap(o.families.first { $0.family == .zImage })
        XCTAssertTrue(z.isReady)
        XCTAssertEqual(z.dit, 1000)
        XCTAssertEqual(z.encoder, 100)
        XCTAssertFalse(try XCTUnwrap(o.families.first { $0.family == .anima }).present)
        XCTAssertEqual(o.total, 2 * 1000 + 2 * 100 + o.families.reduce(0) { $0 + $1.components })
        XCTAssertTrue(try XCTUnwrap(o.families.first { $0.family == .krea2 }).present)

        try b.uninstall(.zImage)
        for gone in [Family.zImage.ditMap, Family.zImage.encoderMap] {
            XCTAssertFalse(FileManager.default.fileExists(atPath: b.map(gone)), gone)
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: b.map(Family.krea2.encoderMap)))
        XCTAssertFalse(b.occupancy().families.first { $0.family == .zImage }!.present)
        XCTAssertTrue(b.occupancy().families.first { $0.family == .krea2 }!.present)

        try b.uninstall(.krea2)
        XCTAssertEqual(b.occupancy().total, 0, "nothing left")
    }

    /// **One version at a time**: switching Z-Image to Compact removes its Standard DiT and
    /// encoder; the Compact encoder goes with the Compact. The render reads the maps of the version
    /// on disk, and a neighbor family is never touched.
    func testSwitchingVersionReplacesTheOtherAndSparesTheNeighbor() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-versions-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let b = Library(root: root)
        let fm = FileManager.default
        try fm.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
        try place(.zImage, in: b)
        try place(.krea2, in: b)
        XCTAssertEqual(b.variant(of: .zImage), .standard)
        XCTAssertEqual(b.ditMap(.zImage), Family.zImage.ditMap)

        // What `Installer.install(.zImage, variant: .compact)` does before its first byte, then its maps.
        XCTAssertEqual(try b.release(.zImage, keeping: .compact), 1100, "the Standard DiT and its encoder")
        try place(.zImage, in: b, variant: .compact)
        XCTAssertFalse(fm.fileExists(atPath: b.map(Family.zImage.ditMap)))
        XCTAssertFalse(fm.fileExists(atPath: b.map(Family.zImage.encoderMap)))
        XCTAssertEqual(b.variant(of: .zImage), .compact)
        XCTAssertEqual(b.ditMap(.zImage), "z-image-turbo-dit.compact.v1.silicon")
        XCTAssertEqual(b.encoderMap(.zImage), "qwen3-4b-encoder.compact.v0.silicon")
        let o = b.occupancy()
        let z = try XCTUnwrap(o.families.first { $0.family == .zImage })
        XCTAssertEqual(z.variant, .compact)
        XCTAssertTrue(z.isReady)
        XCTAssertNil(try XCTUnwrap(o.families.first { $0.family == .anima }).variant, "nothing of Anima is there")
        XCTAssertEqual(o.total, 2 * 1000 + 2 * 100 + o.families.reduce(0) { $0 + $1.components })
        XCTAssertTrue(try XCTUnwrap(o.families.first { $0.family == .krea2 }).present)

        // Back to Standard: the Compact maps go, all of them.
        XCTAssertEqual(try b.release(.zImage, keeping: .standard), 1100)
        XCTAssertEqual(b.variant(of: .zImage), .standard)

        // Uninstalling a Compact removes its encoder, never the neighbor's.
        try place(.zImage, in: b, variant: .compact)
        try b.uninstall(.zImage)
        for v in Variant.allCases {
            XCTAssertFalse(fm.fileExists(atPath: b.map(Family.zImage.ditMap(v))))
        }
        XCTAssertFalse(fm.fileExists(atPath: b.map(Family.zImage.encoderMap(.compact))))
        XCTAssertTrue(fm.fileExists(atPath: b.map(Family.krea2.encoderMap)))
        XCTAssertTrue(b.occupancy().families.first { $0.family == .krea2 }!.present)
    }

    /// **The Compact and the Light share their encoder**: switching between them
    /// replaces the DiT alone, the 8-bit encoder stays and is counted once; leaving both for the
    /// Standard removes it, and so does uninstalling.
    func testCompactAndLightShareTheirEncoder() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-light-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let b = Library(root: root)
        let fm = FileManager.default
        try fm.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
        XCTAssertEqual(Family.zImage.encoderMap(.light), Family.zImage.encoderMap(.compact))
        XCTAssertEqual(Family.qwenImage21.encoderMap(.light), Family.qwenImage21.encoderMap(.compact))
        XCTAssertNotEqual(Family.zImage.ditMap(.light), Family.zImage.ditMap(.compact))
        XCTAssertNotEqual(Family.qwenImage21.ditMap(.light), Family.qwenImage21.ditMap(.compact))

        try place(.zImage, in: b, variant: .compact)
        XCTAssertEqual(b.variant(of: .zImage), .compact)
        // Compact → Light: only the DiT is added, and only the DiT goes.
        try Data("light".utf8).write(to: URL(fileURLWithPath: b.intendedVariantPath(.zImage)))
        let plan = try b.installPlan(.zImage, variant: .light, baseDiT: true, free: nil)
        XCTAssertEqual(plan.replacing, .compact)
        XCTAssertEqual(plan.added, Family.zImage.installedSize(.light).dit, "the encoder is there")
        XCTAssertEqual(plan.freed, 1000, "the Compact's DiT; its encoder is the Light's")
        XCTAssertEqual(b.variant(of: .zImage), .compact, "the Light has no DiT yet: not complete")
        try Data(repeating: 4, count: 700).write(to: URL(fileURLWithPath: b.map(Family.zImage.ditMap(.light))))
        XCTAssertEqual(b.variant(of: .zImage), .light, "both complete for an instant: the version aimed at")
        XCTAssertEqual(try b.release(.zImage, keeping: .light), 1000)
        XCTAssertTrue(fm.fileExists(atPath: b.map(Family.zImage.encoderMap(.light))))
        XCTAssertEqual(b.variant(of: .zImage), .light)
        XCTAssertEqual(b.ditMap(.zImage), "z-image-turbo-dit.light.v1.silicon")
        XCTAssertEqual(b.encoderMap(.zImage), "qwen3-4b-encoder.compact.v0.silicon")
        let z = try XCTUnwrap(b.occupancy().families.first { $0.family == .zImage })
        XCTAssertEqual(z.variant, .light)
        XCTAssertEqual(z.encoder, 100, "one encoder, counted once")
        XCTAssertEqual(z.dit, 700)
        // Without the version aimed at, a Light DiT on disk says Light.
        try fm.removeItem(atPath: b.intendedVariantPath(.zImage))
        XCTAssertEqual(b.variant(of: .zImage), .light)

        // Light → Standard: the Light's DiT and the shared encoder go.
        try Data(repeating: 2, count: 100).write(to: URL(fileURLWithPath: b.map(Family.zImage.encoderMap)))
        try Data(repeating: 3, count: 1000).write(to: URL(fileURLWithPath: b.map(Family.zImage.ditMap)))
        try Data("standard".utf8).write(to: URL(fileURLWithPath: b.intendedVariantPath(.zImage)))
        XCTAssertEqual(try b.release(.zImage, keeping: .standard), 800)
        XCTAssertEqual(b.variant(of: .zImage), .standard)

        // Uninstalling a Light removes its DiT and its encoder.
        try b.release(.zImage, keeping: .light)
        try place(.zImage, in: b, variant: .light)
        try b.uninstall(.zImage)
        XCTAssertEqual(b.occupancy().total, 0)
    }

    /// **The Light is preselected on a Mac of 8 GB, the Compact elsewhere** — preselected, never
    /// imposed: a family with one version is always Standard.
    func testTheLightIsPreselectedOnEightGigabytes() {
        let gib: UInt64 = 1 << 30
        for f in [Family.zImage, .qwenImage21, .anima, .krea2] {
            XCTAssertEqual(f.preselectedVariant(physicalMemory: 8 * gib), .light, f.name)
            XCTAssertEqual(f.preselectedVariant(physicalMemory: 16 * gib), .compact, f.name)
            XCTAssertEqual(f.preselectedVariant(physicalMemory: 12 * gib), .compact, f.name)
        }
        for f in Family.allCases where f.variants == [.standard] {
            XCTAssertEqual(f.preselectedVariant(physicalMemory: 8 * gib), .standard, f.name)
        }
    }

    /// **A switch never leaves a family without a version**: the new version is forged beside
    /// the old one, which the render keeps reading until the new one is complete; only then does the
    /// old one go (`Installer.install` ends with `release`). Here the installation's steps, map by map.
    func testTheOldVersionRendersUntilTheNewOneIsComplete() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-switch-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let b = Library(root: root)
        let fm = FileManager.default
        try fm.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
        try place(.zImage, in: b)
        let standard = (dit: Family.zImage.ditMap, encoder: Family.zImage.encoderMap)

        // Standard → Compact. Before the first byte: the version aimed at, nothing removed.
        try Data("compact".utf8).write(to: URL(fileURLWithPath: b.intendedVariantPath(.zImage)))
        XCTAssertEqual(b.variant(of: .zImage), .standard)
        // The Compact encoder forged, its DiT not yet (a stop, a cut, a full disk here): Standard still.
        try Data(repeating: 2, count: 100).write(to: URL(fileURLWithPath: b.map(Family.zImage.encoderMap(.compact))))
        XCTAssertEqual(b.variant(of: .zImage), .standard, "the Compact is not complete")
        XCTAssertEqual(b.ditMap(.zImage), standard.dit)
        XCTAssertEqual(b.encoderMap(.zImage), standard.encoder)
        XCTAssertTrue(Installer(library: b).isReady(.zImage))
        XCTAssertNil(try XCTUnwrap(b.cards().first { $0.id == "z-image" }).missing(in: b), "Z-Image still renders")
        XCTAssertEqual(try b.resolvedVariant(.zImage, requested: nil, baseDiT: true), .standard, "a command without a flag keeps it")
        let z = try XCTUnwrap(b.occupancy().families.first { $0.family == .zImage })
        XCTAssertEqual(z.variant, .standard)
        XCTAssertEqual(z.encoder, 200, "the half-forged Compact's encoder occupies the disk too")
        // The Compact DiT forged: both complete for an instant — the version aimed at wins.
        try Data(repeating: 3, count: 1000).write(to: URL(fileURLWithPath: b.map(Family.zImage.ditMap(.compact))))
        XCTAssertEqual(b.variant(of: .zImage), .compact)
        // Then the old one goes: its DiT and its encoder.
        XCTAssertEqual(try b.release(.zImage, keeping: .compact), 1100)
        XCTAssertEqual(b.variant(of: .zImage), .compact)
        XCTAssertFalse(fm.fileExists(atPath: b.map(standard.encoder)))

        // Compact → Standard: the Standard encoder forged, its DiT not yet — Compact still.
        try Data("standard".utf8).write(to: URL(fileURLWithPath: b.intendedVariantPath(.zImage)))
        try Data(repeating: 2, count: 100).write(to: URL(fileURLWithPath: b.map(standard.encoder)))
        XCTAssertEqual(b.variant(of: .zImage), .compact, "the Standard has no DiT yet: it is not complete")
        XCTAssertEqual(b.ditMap(.zImage), Family.zImage.ditMap(.compact))
        try Data(repeating: 3, count: 1000).write(to: URL(fileURLWithPath: b.map(standard.dit)))
        XCTAssertEqual(b.variant(of: .zImage), .standard)
        XCTAssertEqual(try b.release(.zImage, keeping: .standard), 1100, "the Compact's DiT and encoder")

        // A switch that had to remove the old version first, stopped before the new one's first map:
        // nothing is complete, the version aimed at is what the panel and a resumed install take.
        try Data("compact".utf8).write(to: URL(fileURLWithPath: b.intendedVariantPath(.zImage)))
        try b.release(.zImage, keeping: .compact)
        XCTAssertEqual(b.variant(of: .zImage), .compact)
        XCTAssertEqual(b.installedVariant(of: .zImage), .compact)
        XCTAssertEqual(try b.resolvedVariant(.zImage, requested: nil, baseDiT: true), .compact, "resumed in the version aimed at")
        XCTAssertNotNil(try XCTUnwrap(b.cards().first { $0.id == "z-image" }).missing(in: b))
    }

    /// **Whether both versions fit at once, said before the first byte** (`Library.installPlan`): the
    /// old version goes first only when the disk cannot hold both, and only if its space then makes
    /// the installation fit. Sparse files of the maps' real sizes: no byte written.
    func testThePlanSaysWhenTheOldVersionMustGoFirst() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-plan-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let b = Library(root: root)
        let fm = FileManager.default
        try fm.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
        try place(.zImage, in: b)
        func sparse(_ name: String, _ bytes: Int) throws {
            let h = try XCTUnwrap(FileHandle(forWritingAtPath: b.map(name)))
            try h.truncate(atOffset: UInt64(bytes)); try h.close()
        }
        try sparse(Family.zImage.ditMap, 1_000_000_000)
        try sparse(Family.zImage.encoderMap, 500_000_000)
        let c = Family.zImage.installedSize(.compact)
        let peak = c.dit + c.encoder + c.dit
        XCTAssertEqual(Installer(library: b).spaceNeeded(.zImage, variant: .compact, baseDiT: true), peak)

        // Room for both: the Standard renders until the Compact is complete.
        let roomy = try b.installPlan(.zImage, variant: .compact, baseDiT: true, free: 2 * peak)
        XCTAssertEqual(roomy.variant, .compact)
        XCTAssertEqual(roomy.replacing, .standard)
        XCTAssertEqual(roomy.freed, 1_500_000_000, "its DiT and its encoder (nobody else reads it)")
        XCTAssertEqual(roomy.added, c.dit + c.encoder, "the components are there")
        XCTAssertFalse(roomy.removesFirst)
        XCTAssertTrue(roomy.fits)
        // Not for both, but with the Standard's space: it goes first, and the plan says so.
        let tight = try b.installPlan(.zImage, variant: .compact, baseDiT: true, free: peak - 500_000_000)
        XCTAssertTrue(tight.removesFirst)
        XCTAssertTrue(tight.fits)
        // Not even then: nothing is removed, the installation is refused before its first byte.
        let short = try b.installPlan(.zImage, variant: .compact, baseDiT: true, free: peak - 2_000_000_000)
        XCTAssertFalse(short.removesFirst)
        XCTAssertFalse(short.fits)
        // The same version: nothing replaced, nothing to add.
        let same = try b.installPlan(.zImage, variant: nil, baseDiT: true, free: 0)
        XCTAssertEqual(same.variant, .standard)
        XCTAssertNil(same.replacing)
        XCTAssertEqual(same.added, 0)

        // An encoder already there counts for nothing: Z-Image back to Standard with the Standard
        // encoder left on disk — only the DiT is added.
        try b.release(.zImage, keeping: .compact)
        try place(.zImage, in: b, variant: .compact)
        try Data(repeating: 2, count: 100).write(to: URL(fileURLWithPath: b.map(Family.zImage.encoderMap)))
        let back = try b.installPlan(.zImage, variant: .standard, baseDiT: true, free: nil)
        XCTAssertEqual(back.replacing, .compact)
        XCTAssertEqual(back.added, Family.zImage.installedSize(.standard).dit)
    }

    /// **A command that names no version never changes it** — and what an imported model lacks
    /// (`baseDiT` false) never changes it either: the version installed is imposed, Standard if none,
    /// and a version that contradicts it is refused before anything.
    func testNoFlagAndNoDiTKeepTheVersionInstalled() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-keep-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let b = Library(root: root)
        try FileManager.default.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)

        // Nothing installed — an encoder map alone does not make Z-Image installed: Standard.
        try Data(repeating: 2, count: 100).write(to: URL(fileURLWithPath: b.map(Family.zImage.encoderMap)))
        XCTAssertNil(b.installedVariant(of: .zImage))
        XCTAssertEqual(try b.resolvedVariant(.zImage, requested: nil, baseDiT: true), .standard)
        XCTAssertEqual(try b.resolvedVariant(.zImage, requested: nil, baseDiT: false), .standard)
        XCTAssertEqual(try b.resolvedVariant(.zImage, requested: .compact, baseDiT: true), .compact)
        XCTAssertThrowsError(try b.resolvedVariant(.zImage, requested: .compact, baseDiT: false)) {
            XCTAssertTrue("\($0)".contains("not compact"), "\($0)")
        }

        try FileManager.default.removeItem(atPath: b.map(Family.zImage.encoderMap))
        // Z-Image in Compact for imported models only (its encoder and components, no DiT).
        try place(.zImage, in: b, dit: false, variant: .compact)
        XCTAssertEqual(b.installedVariant(of: .zImage), .compact)
        XCTAssertEqual(b.variant(of: .zImage), .compact)
        XCTAssertEqual(try b.resolvedVariant(.zImage, requested: nil, baseDiT: true), .compact, "no flag: kept")
        XCTAssertEqual(try b.resolvedVariant(.zImage, requested: nil, baseDiT: false), .compact)
        XCTAssertThrowsError(try b.resolvedVariant(.zImage, requested: .standard, baseDiT: false)) {
            XCTAssertTrue("\($0)".contains("version installed (compact)"), "\($0)")
        }
        XCTAssertEqual(try b.resolvedVariant(.zImage, requested: .standard, baseDiT: true), .standard, "asked by name: switches")
        let plan = try b.installPlan(.zImage, variant: nil, baseDiT: false, free: nil)
        XCTAssertEqual(plan.variant, .compact)
        XCTAssertNil(plan.replacing, "the prerequisites of an import replace nothing")

    }

    /// **Anima's three versions read one encoder, the Standard's**: switching between them removes
    /// the other DiT and its text adapter, never the encoder; each version is complete with it.
    func testAnimaVersionsShareTheStandardEncoder() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-anima-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let b = Library(root: root)
        try FileManager.default.createDirectory(atPath: b.map(""), withIntermediateDirectories: true)
        XCTAssertEqual(Set(Variant.allCases.map(Family.anima.encoderMap)), [Family.anima.encoderMap])
        XCTAssertEqual(Set(Variant.allCases.map(Family.anima.ditMap)).count, 3)
        try place(.anima, in: b)
        let adapter = adapterPath(fromMap: b.map(Family.anima.ditMap))
        try Data(repeating: 4, count: 10).write(to: URL(fileURLWithPath: adapter))
        XCTAssertEqual(b.variant(of: .anima), .standard)
        // The Light forged beside the Standard: the Standard's DiT and adapter go, the encoder stays.
        try Data(repeating: 3, count: 500).write(to: URL(fileURLWithPath: b.map(Family.anima.ditMap(.light))))
        try Data("light".utf8).write(to: URL(fileURLWithPath: b.intendedVariantPath(.anima)))
        XCTAssertEqual(b.variant(of: .anima), .light)
        XCTAssertEqual(Set(b.releasable(.anima, keeping: .light)), [b.map(Family.anima.ditMap), adapter])
        try b.release(.anima, keeping: .light)
        XCTAssertTrue(FileManager.default.fileExists(atPath: b.map(Family.anima.encoderMap)))
        XCTAssertEqual(b.variant(of: .anima), .light)
        XCTAssertTrue(Installer(library: b).isReady(.anima))
        // Nothing of the encoder is counted as the other versions' (it is the current one's).
        let plan = try b.installPlan(.anima, variant: .compact, baseDiT: true, free: nil)
        XCTAssertEqual(plan.added, Family.anima.installedSize(.compact).dit, "the encoder is there: only the DiT is added")
        XCTAssertEqual(plan.replacing, .light)
    }

    /// **Krea's easter egg is set aside, never placed**: the two tensors the GGUFs of Krea 2 Turbo carry
    /// beside the model (`last.down`, `last.up`) are ignored by the recipe, which still refuses any
    /// other unknown name.
    func testKreaEasterEggIsSetAside() throws {
        final class Names: TensorCatalog {
            let names: [String]
            init(_ n: [String]) { names = n }
            func shape(_ name: String) -> [Int]? { [6144, 6144] }
            func dtype(_ name: String) -> String? { "BF16" }
            func read(_ name: String) throws -> [Float] { [] }
        }
        let n = try Recipes.normalize(Names(["blocks.0.attn.wq.weight", "last.down.weight", "last.up.weight"]), family: .krea2)
        XCTAssertEqual(Array(n.dit.keys), ["transformer_blocks.0.attn.to_q.weight"])
        XCTAssertEqual(Set(n.ignored), Recipes.kreaEasterEgg)
        XCTAssertThrowsError(try Recipes.normalize(Names(["blocks.0.attn.wq.weight", "last.sideways.weight"]), family: .krea2))
    }

    /// A job that would not fit is refused before writing a byte.
    func testSpaceIsCheckedBeforehand() throws {
        let b = Library(root: FileManager.default.temporaryDirectory.appendingPathComponent("n-existe-pas-\(UUID())"))
        let free = try XCTUnwrap(b.freeSpace(), "the temporary folder's volume, through its first existing parent")
        XCTAssertNoThrow(try b.checkSpace(1))
        XCTAssertThrowsError(try b.checkSpace(free + 1 << 40)) {
            guard case let EngineError.diskFull(needed, available) = $0 else { return XCTFail("\($0)") }
            XCTAssertEqual(needed, free + 1 << 40)
            XCTAssertNotNil(available)
        }
    }

    /// **An installation is judged against the disk before its first byte**, from the measured sizes:
    /// everything missing, plus the sources of the largest part while it is forged — as large as its
    /// map, since a map is never wider nor narrower than its sources.
    func testAnInstallationKnowsItsSizeBeforeDownloading() {
        let b = Library(root: FileManager.default.temporaryDirectory.appendingPathComponent("n-existe-pas-\(UUID())"))
        let i = Installer(library: b)
        let q = Family.qwenImage21.installedSize(.standard)
        XCTAssertEqual(i.spaceNeeded(.qwenImage21, baseDiT: true), q.dit + q.encoder + q.components + q.encoder)
        let z = Family.zImage.installedSize(.standard)
        XCTAssertEqual(i.spaceNeeded(.zImage, baseDiT: false), z.encoder + z.components + z.encoder)
        let c = Family.zImage.installedSize(.compact)
        XCTAssertEqual(i.spaceNeeded(.zImage, variant: .compact, baseDiT: true), c.dit + c.encoder + c.components + c.dit)
    }
}
