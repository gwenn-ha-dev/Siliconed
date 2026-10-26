import XCTest
@testable import Siliconed
@testable import Verification

/// **MOSS-Image as a family, without a byte of weights**: what the installation fetches (pinned revisions, sha256 of
/// every weight file), the names its maps have, the chain's wires (a wrong module is refused before any computation)
/// and its computed need — the encoder streamed, the DMD LoRA always charged, the decoder's heaps.
final class MossImageFamilyTests: XCTestCase {

    // MARK: - what the installation fetches

    func testTheRevisionsAreThePinnedOnes() {
        XCTAssertEqual(Installer.mossImage.name, "OpenMOSS-Team/MOSS-Image-SFT")
        XCTAssertEqual(Installer.mossImage.revision, "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9")
        XCTAssertEqual(Installer.mossTurbo.name, "lodestones/MOSS-Image-SFT-8-steps-dmd")
        XCTAssertEqual(Installer.mossTurbo.revision, "f95df83b5e90ae91ea412434b3f9b7f1c8c6314f")
        XCTAssertEqual(Installer.companion(.mossImage).map(\.path), ["lora.safetensors"])
        XCTAssertEqual(Installer.companion(.mossImage).first?.repository.name, Installer.mossTurbo.name)
    }

    /// Every weight file is pinned by its sha256 and its size; the small text files carry none.
    func testEveryWeightFileIsPinned() {
        let weights = Installer.encoder(.mossImage).filter { $0.path.hasSuffix(".safetensors") }
            + Installer.dit(.mossImage).filter { $0.path.hasSuffix(".safetensors") }
            + Installer.companion(.mossImage)
            + Installer.components(.mossImage).filter { $0.path.hasSuffix(".safetensors") || $0.path.hasSuffix("tokenizer.json") }
        XCTAssertEqual(weights.count, 4 + 3 + 1 + 2)
        for file in weights {
            XCTAssertEqual(file.sha256?.count, 64, file.path)
            XCTAssertGreaterThan(file.size ?? 0, 0, file.path)
        }
        // The shards' sizes are the published ones: 16.79 GB of text model, 10.16 GB of DiT, 1.19 GB of LoRA.
        XCTAssertEqual(Installer.encoder(.mossImage).compactMap(\.size).reduce(0, +), 4_973_910_744 + 4_988_364_288 + 4_954_202_112 + 1_869_001_848)
        XCTAssertEqual(Installer.dit(.mossImage).compactMap(\.size).reduce(0, +), 4_991_978_016 + 4_983_137_016 + 189_549_184)
        XCTAssertEqual(Installer.companion(.mossImage).first?.size, 1_189_176_800)
        // Everything comes from the pinned repositories.
        for file in Installer.encoder(.mossImage) + Installer.dit(.mossImage) + Installer.components(.mossImage) {
            XCTAssertEqual(file.repository.name, Installer.mossImage.name, file.path)
        }
        // Every component lands where the engine reads it.
        XCTAssertEqual(Installer.components(.mossImage).compactMap(\.destination).sorted(),
                       ["tokenizer/tokenizer.json", "transformer.json", "vae.json", "vae.safetensors"])
    }

    func testTheStandardIsTheOnlyVersion() {
        XCTAssertEqual(Family.mossImage.variants, [.standard])
        XCTAssertEqual(Family.mossImage.preselectedVariant(), .standard)
        XCTAssertEqual(Family.mossImage.preselectedVariant(physicalMemory: 8 << 30), .standard)
        XCTAssertTrue(Family.offered.contains(.mossImage))
        XCTAssertEqual(Family(ditKind: Family.mossImage.ditKind), .mossImage)
    }

    /// The names the stages, the engine and the installation agree on.
    func testTheMapsHaveOneName() {
        XCTAssertEqual(Family.mossImage.ditMap, MossImageStages.ditMap)
        XCTAssertEqual(Family.mossImage.ditKind, MossImageDiT.kind)
        XCTAssertEqual(Family.mossImage.encoderMap, "qwen3.5-9b-encoder.v0.silicon")
        XCTAssertEqual(Family.mossImage.ditMap(.standard), Family.mossImage.ditMap)
        XCTAssertEqual(MossImageStages.model, ModelCard.mossImage.id)
        XCTAssertEqual(MossImageDenoisingModule(map: "", turbo: "").model, MossImageForge.model)
        let library = Library(root: URL(fileURLWithPath: "/nowhere"))
        XCTAssertEqual(Installer(library: library).companionMap(.mossImage), "/nowhere/store/composants/moss-image/turbo.lora.silicon")
        XCTAssertNil(Installer(library: library).companionMap(.zImage))
    }

    func testTheLicensesAreTheModelsAndTheLoRAs() {
        let urls = ModelCard.mossImage.license.urls.map(\.absoluteString)
        XCTAssertEqual(urls.count, 2)
        XCTAssertTrue(urls[0].contains("OpenMOSS-Team/MOSS-Image-SFT/blob/f972bd4b4b9e3852133cc5da954d87bc32cd8fb9/LICENSE"), urls[0])
        XCTAssertTrue(urls[1].contains("lodestones/MOSS-Image-SFT-8-steps-dmd/blob/f95df83b5e90ae91ea412434b3f9b7f1c8c6314f/README.md"), urls[1])
        XCTAssertTrue(ModelCard.mossImage.license.commercial)
        XCTAssertFalse(ModelCard.mossImage.license.requiredFilter)
        XCTAssertEqual(ModelCard.of(.mossImage), ModelCard.mossImage)
        XCTAssertEqual(ModelCard.withID("moss-image"), ModelCard.mossImage)
    }

    /// A checkpoint of this family is not recognized (no import in this version), and its LoRAs are refused by name.
    func testNothingIsImportedIntoIt() {
        let names = ["layers.0.proj.q.weight", "layers.0.proj.ffn_up_gate.weight", "shared_block_modulation.weight"]
        XCTAssertNil(Recipes.family(fromNames: names))
    }

    // MARK: - the chain

    func testTheChainWiresAndRefusesTheWrongOnes() throws {
        let text = MossImageTextModule(map: "", tokenizer: ""), dit = MossImageDenoisingModule(map: "", turbo: "")
        let decoder = Flux2DecodingModule(path: "")
        XCTAssertNoThrow(try Chain(text: text, denoising: dit, decoding: decoder))
        XCTAssertEqual(text.output, dit.entry)
        XCTAssertEqual(dit.space, decoder.space)
        XCTAssertEqual(dit.space.channels, 128)
        XCTAssertEqual(dit.space.factor, 16)
        // A text for another DiT, a decoder for another latent: refused at the wiring, not after a minute of encoder.
        XCTAssertThrowsError(try Chain(text: ZImageTextModule(map: "", tokenizer: ""), denoising: dit, decoding: decoder))
        XCTAssertThrowsError(try Chain(text: text, denoising: Krea2DenoisingModule(map: ""), decoding: decoder))
        XCTAssertThrowsError(try Chain(text: text, denoising: dit, decoding: QwenImageDecodingModule(path: "")))
        XCTAssertThrowsError(try Chain(text: text, denoising: dit, decoding: FluxDecodingModule(path: "")))
        // No image encoder, no reference: the entries are the prompt's.
        XCTAssertEqual(try Chain(text: text, denoising: dit, decoding: decoder).entries.map(\.kind), [.text])
    }

    func testTheRecipeIsEightStepsOfEulerWithoutCFG() {
        let dit = MossImageDenoisingModule(map: "", turbo: "")
        XCTAssertEqual(dit.defaultSteps, 8)
        XCTAssertEqual(ModelCard.mossImage.defaultSteps, 8)
        XCTAssertEqual(dit.sigmas(steps: 8), [1, 0.968, 0.926, 0.871, 0.791, 0.671, 0.46300002932548523, 0.024, 0])
        XCTAssertEqual(dit.solvers, [.euler])
        XCTAssertFalse(dit.acceptsStartImage)
        XCTAssertEqual(dit.maxReferences, 0)
        let plan = dit.plan(height: 64, width: 64, steps: 8, start: 0, withLoRA: false)
        XCTAssertEqual(plan.evaluations, 8)
        XCTAssertEqual(plan.reduced, 0)
        // A dense schedule has fewer evaluations than steps (the σ are snapped to the training grid, then deduplicated).
        XCTAssertEqual(dit.plan(height: 64, width: 64, steps: 250, start: 0, withLoRA: false).evaluations, 246)
    }

    /// 512² and up, on a multiple of 16 — the DiT's cell.
    func testTheFormatRules() throws {
        XCTAssertNoThrow(try Format.check(width: 1024, height: 1024, family: .mossImage))
        XCTAssertNoThrow(try Format.check(width: 832, height: 1216, family: .mossImage))
        XCTAssertThrowsError(try Format.check(width: 1000, height: 1024, family: .mossImage))
        XCTAssertThrowsError(try Format.check(width: 256, height: 256, family: .mossImage))
    }

    // MARK: - the text, pure

    func testTheTemplateIsTheChatTemplateWithoutASystemTurn() {
        XCTAssertEqual(Qwen35Prompt.template("a cat"), "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n")
        let bound = MemoryNeed.Request.promptTokens("a 30 year old woman posing in a library", family: .mossImage)
        XCTAssertGreaterThanOrEqual(bound, 21)   // the bytes bound the tokens (21 for this prompt)
        XCTAssertLessThanOrEqual(MemoryNeed.Request.promptTokens(text: 100_000, family: .mossImage), Qwen35Prompt.maxTokens)
    }

    // MARK: - the need

    /// MOSS-Image's maps as the formula reads them: the DiT's 42 layers (published shapes, bf16) and the language
    /// model's 32 (24 Gated DeltaNet, 8 full attention), on 16 KB pages.
    static func maps(rank: Int = 128) -> MemoryNeed.MossMaps {
        let (d, f, page) = (3072, 8192, 16384)
        let ditLayer: [(String, [Int])] = [
            ("k_norm.weight", [128]), ("norm1.weight", [d]), ("norm2.weight", [d]), ("proj.ffn_down.weight", [f, d]),
            ("proj.ffn_up_gate.weight", [d, 2 * f]), ("proj.k.weight", [d, d]), ("proj.o.weight", [d, d]),
            ("proj.q.weight", [d, d]), ("proj.v.weight", [d, d]), ("q_norm.weight", [128]),
        ]
        var dit = MemoryFixtures.table(layers: 42, prefix: MossImageDiT.layerPrefix, tensors: ditLayer, bytesPerValue: 2)
        dit["config"] = [String: Any]()
        let (h, i) = (4096, 12288)
        let types = (0..<32).map { $0 % 4 == 3 ? "full_attention" : "linear_attention" }
        var table: [String: Any] = [:], order: [String] = [], offset = page
        func add(_ name: String, _ shape: [Int], bytesPerValue: Int = 2) {
            let bytes = shape.reduce(1, *) * bytesPerValue
            table[name] = ["offset": offset, "bytes": bytes, "shape": shape]
            order.append(name)
            offset += (bytes + page - 1) / page * page
        }
        for layer in 0..<32 {
            let p = "layers.\(layer)."
            add(p + "input_layernorm.weight", [h], bytesPerValue: 4)
            if types[layer] == "full_attention" {
                add(p + "self_attn.q_proj.weight", [h, 8192]); add(p + "self_attn.k_proj.weight", [h, 1024])
                add(p + "self_attn.v_proj.weight", [h, 1024]); add(p + "self_attn.q_norm.weight", [256], bytesPerValue: 4)
                add(p + "self_attn.k_norm.weight", [256], bytesPerValue: 4); add(p + "self_attn.o_proj.weight", [4096, h])
            } else {
                add(p + "linear_attn.in_proj_qkv.weight", [h, 8192]); add(p + "linear_attn.in_proj_z.weight", [h, 4096])
                add(p + "linear_attn.in_proj_b.weight", [h, 32]); add(p + "linear_attn.in_proj_a.weight", [h, 32])
                add(p + "linear_attn.conv1d.weight", [8192, 4], bytesPerValue: 4); add(p + "linear_attn.A_log", [32], bytesPerValue: 4)
                add(p + "linear_attn.dt_bias", [32], bytesPerValue: 4); add(p + "linear_attn.norm.weight", [128], bytesPerValue: 4)
                add(p + "linear_attn.out_proj.weight", [4096, h])
            }
            add(p + "post_attention_layernorm.weight", [h], bytesPerValue: 4)
            add(p + "mlp.gate_proj.weight", [h, i]); add(p + "mlp.up_proj.weight", [h, i]); add(p + "mlp.down_proj.weight", [i, h])
        }
        let encoder: [String: Any] = [
            "page": page, "tensors": table, "order": order,
            "config": ["hidden_size": h, "num_hidden_layers": 32, "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
                       "intermediate_size": i, "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
                       "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "layer_types": types, "rms_norm_eps": 1e-6,
                       "rope_parameters": ["rope_type": "default", "partial_rotary_factor": 0.25, "rope_theta": 10_000_000.0]]
                as [String: Any]]
        return MemoryNeed.MossMaps(dit: dit, encoder: encoder, turboRank: rank)!
    }

    private func request(_ w: Int, _ h: Int, tokens: Int = 21, ranks: [Int] = [], maps: MemoryNeed.MossMaps = maps()) -> MemoryNeed.Request {
        var request = MemoryNeed.Request(card: .mossImage, width: w, height: h, promptTokens: tokens, loraRanks: ranks)
        request.threads = 6
        request.mossMaps = maps
        return request
    }

    private func need(_ w: Int, _ h: Int, tokens: Int = 21, ranks: [Int] = [], maps: MemoryNeed.MossMaps = maps()) throws -> MemoryNeed {
        try XCTUnwrap(MemoryNeed.mossImage(request(w, h, tokens: tokens, ranks: ranks, maps: maps), maps: maps))
    }

    func testTheMapsAreReadFromTheirHeaders() {
        let m = Self.maps()
        XCTAssertEqual(m.encoderUnit, 4096 * 12288 * 2)   // an MLP matrix in bf16: the stream's buffer
        XCTAssertEqual(m.encoderConfig.fullLayer.filter { $0 }.count, 8)
        XCTAssertEqual(m.turboRank, 128)
        XCTAssertEqual(m.dit.layers, 42)
        // Not a MOSS-Image encoder, a LoRA without a rank: no maps, so no need.
        XCTAssertNil(MemoryNeed.MossMaps(dit: m.ditHeader, encoder: [:], turboRank: 128))
        XCTAssertNil(MemoryNeed.MossMaps(dit: m.ditHeader, encoder: ["config": [String: Any]()], turboRank: 128))
    }

    /// Three phases, each the constructors' own function; the encoder streams, the LoRA is always charged.
    func testThePhasesAreTheConstructorsFunctions() throws {
        let maps = Self.maps()
        let n = try need(1024, 1024)
        XCTAssertEqual(n.phases.map(\.kind), [.text, .denoising, .decoding])
        let text = n.phases[0]
        XCTAssertEqual(text.bytes("encoder arena"), Qwen35Encoder.arenaBytes(config: maps.encoderConfig, sequence: 21))
        XCTAssertEqual(text.bytes("encoder staging"), TailStream.unitSlots * maps.encoderUnit)
        let lean = try XCTUnwrap(MemoryNeed.mossImage(request(1024, 1024, maps: maps).lean, maps: maps))
        XCTAssertEqual(lean.phases[0].bytes("encoder staging"), 0)   // the lean plan: no encoder buffer
        // The DMD LoRA is in every denoising phase, whatever the request carries: its cache is 1.19 GB at rank 128.
        let denoising = n.phases[1]
        XCTAssertGreaterThan(denoising.bytes("LoRA cache"), 1_180_000_000)
        XCTAssertEqual(denoising.bytes("LoRA down staging"), 0)
        XCTAssertEqual(denoising.bytes("DiT arena"), MossImageDiT.arenaBytes(config: maps.dit, gridHeight: 64, gridWidth: 64, maxText: 21, rank: 128))
        // A second LoRA of the request joins the stack: its rank, a staged `down`.
        let two = try need(1024, 1024, ranks: [32])
        XCTAssertGreaterThan(two.phases[1].bytes("DiT arena"), denoising.bytes("DiT arena"))
        XCTAssertEqual(two.phases[1].bytes("LoRA down staging"), LoRA.downBytes(k: 8192, rank: 128))
        // Crossing: the encoded text [21, 4096], the latent [128, 64, 64] and the modulation table.
        XCTAssertEqual(n.crossing, 21 * 4096 * 4 + 128 * 64 * 64 * 4 + MossImageDiT.tableBytes(config: maps.dit, count: Request.maximumSteps))
        // The decoding is the VAE's plan at the image's /8 grid, and the image it returns.
        let plan = Flux2VAE.plan(latentHeight: 128, latentWidth: 128)
        XCTAssertEqual(n.phases[2].bytes("FLUX.2 decoder registers"), plan.registers)
        XCTAssertEqual(n.phases[2].bytes("FLUX.2 decoder host"), plan.host)
    }

    func testTheNeedGrowsWithTheFormatAndTheFloorFitsB() throws {
        let sizes: [(Int, Int)] = [(512, 512), (1024, 1024), (1536, 1536), (2048, 2048)]
        let peaks = try sizes.map { try need($0.0, $0.1).peak }
        XCTAssertEqual(peaks, peaks.sorted())
        XCTAssertEqual(Set(peaks).count, peaks.count)
        // The floor (the lean plan) of 1024² fits B on a 16 GB Mac, ¾·RAM − 2 GiB, with the process's base left for the runtime.
        let b = 12 * (1 << 30) - 2 * (1 << 30)
        let floor = try XCTUnwrap(MemoryNeed.mossImage(request(1024, 1024).lean, maps: Self.maps()))
        XCTAssertLessThan(floor.peak, b - 600_000_000, "\(Double(floor.peak) / 1e9) GB")
        // A longer prompt costs more rows in the encoder and in the DiT.
        XCTAssertGreaterThan(try need(1024, 1024, tokens: 1000).peak, try need(1024, 1024, tokens: 21).peak)
        // The batch's other latents wait for the decodings.
        var batch = request(1024, 1024)
        batch.batch = 3
        let three = try XCTUnwrap(MemoryNeed.mossImage(batch, maps: Self.maps()))
        XCTAssertEqual(three.crossing - (try need(1024, 1024)).crossing, 2 * 128 * 64 * 64 * 4)
    }

    /// Unknown maps, unknown need: the render is refused rather than priced at zero.
    func testWithoutMapsThereIsNoNeed() {
        let empty = Library(root: URL(fileURLWithPath: "/nulle-part-\(UUID())"))
        XCTAssertNil(ModelCard.mossImage.memoryNeed(request: empty))
    }
}

private extension ModelCard {
    func memoryNeed(request library: Library) -> Int? {
        memoryNeed(memoryRequest(width: 1024, height: 1024, library: library))
    }
}
