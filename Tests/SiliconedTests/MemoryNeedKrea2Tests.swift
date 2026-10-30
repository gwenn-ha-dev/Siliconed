import XCTest
@testable import Siliconed

/// **Krea 2's computed need, without a byte of weights.** The DiT's and the fusion's arenas are built on
/// maps written by hand and compared with their shape functions; the formula runs on headers written
/// in memory with Krea 2's published shapes (`MemoryFixtures.krea2Maps`). No figure here depends on
/// the Mac: the terms pinned are the arenas, the staging blocks and the VAE's plans, never the AMX's
/// workspace (`EngineSettings`).
final class MemoryNeedKrea2Tests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-krea2-need-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// A `SILICON` v3 map: `tensors` laid out from `dataStart`, each on a 16-byte page, zeros.
    private func map(_ header: [String: Any], tensors: [(name: String, shape: [Int])] = []) throws -> String {
        let page = 16, dataStart = 1 << 16
        var table: [String: Any] = [:], offset = dataStart
        for t in tensors {
            let bytes = t.shape.reduce(1, *) * 2
            table[t.name] = ["offset": offset, "bytes": bytes, "shape": t.shape, "dtype": "bfloat16"]
            offset += (bytes + page - 1) / page * page
        }
        var full = header
        full["page"] = page; full["tensors"] = table; full["order"] = tensors.map(\.name)
        let json = try JSONSerialization.data(withJSONObject: full)
        precondition(16 + json.count <= dataStart, "header longer than the test's layout")
        var bytes = Array("SILICON".utf8) + [3]
        for i in 0..<8 { bytes.append(UInt8((UInt64(json.count) >> (8 * UInt64(i))) & 0xff)) }
        bytes += json
        bytes += [UInt8](repeating: 0, count: max(offset, dataStart + 16) - bytes.count)
        let path = root.appendingPathComponent("m-\(UUID()).silicon").path
        try Data(bytes).write(to: URL(fileURLWithPath: path))
        return path
    }

    private func krea2Header(_ config: [String: Any]) -> [String: Any] {
        ["kind": "krea2-turbo-dit", "rope_layout": "interleaved", "rmsnorm_plus_one_folded": true,
         "linear_weights_transposed": true, "config": config]
    }

    private var tinyDiT: [String: Any] {
        krea2Header(["num_attention_heads": 2, "num_key_value_heads": 1, "attention_head_dim": 8, "intermediate_size": 32,
                     "num_layers": 1, "in_channels": 16])
    }

    private func need(_ w: Int, _ h: Int, tokens: Int = 59, ranks: [Int] = [],
                      references: [(width: Int, height: Int)] = [],
                      maps: MemoryNeed.Krea2Maps = MemoryFixtures.krea2Maps()) -> MemoryNeed {
        MemoryNeed.krea2(MemoryFixtures.krea2Request(w, h, tokens: tokens, ranks: ranks, references: references, maps: maps),
                         maps: maps)
    }

    private func pages(_ floats: [Int]) -> Int { floats.reduce(0) { $0 + Arena.capacity(for: $1 * 4) } }

    // ── the formula is the allocation ─────────────────────────────────────────────────────────

    /// The DiT's and the fusion's arenas, built, are their shape functions — with and without a LoRA
    /// stack (two LoRAs on one module: cumulative rank 7).
    func testTheArenasAreTheFormula() throws {
        let artifact = try Artifact(path: try map(tinyDiT))
        let config = Krea2DiT.Config(header: artifact.header)
        for (h, w, text) in [(8, 8, 6), (16, 8, 10)] {
            let dit = try Krea2DiT(artifact: artifact, latentHeight: h, latentWidth: w, maxText: text)
            XCTAssertEqual(dit.arenaCapacity, Krea2DiT.arenaBytes(config: config, latentHeight: h, latentWidth: w,
                                                                  maxText: text, rank: 0))
            let fusion = try Krea2TextFusion(artifact: artifact, maxTokens: text)
            XCTAssertEqual(fusion.arenaCapacity, Krea2TextFusion.arenaBytes(outDim: config.dim, maxTokens: text, rank: 0))
        }
        func lora(_ rank: Int, _ targets: [(String, Int, Int)]) throws -> LoRA.Layer {
            let tensors = targets.flatMap { [("\($0.0).down", [$0.1, rank]), ("\($0.0).up", [rank, $0.2])] }
            let path = try map(["kind": "lora"], tensors: tensors.map { (name: $0.0, shape: $0.1) })
            return LoRA.Layer(artifact: try Artifact(path: path), strength: 1, name: path)
        }
        let (d, hidden) = (config.dim, config.hidden)
        let stack = try LoRA(layers: [
            lora(4, [("transformer_blocks.0.attn.to_q", d, d), ("transformer_blocks.0.ff.down", hidden, d)]),
            lora(3, [("transformer_blocks.0.attn.to_q", d, d)]),
        ])
        XCTAssertEqual(stack.maxRank, 7)
        let dit = try Krea2DiT(artifact: artifact, latentHeight: 8, latentWidth: 8, maxText: 6, lora: stack)
        XCTAssertEqual(dit.arenaCapacity, Krea2DiT.arenaBytes(config: config, latentHeight: 8, latentWidth: 8, maxText: 6, rank: 7))
        let fusion = try Krea2TextFusion(artifact: artifact, maxTokens: 6, lora: stack)
        XCTAssertEqual(fusion.arenaCapacity, Krea2TextFusion.arenaBytes(outDim: d, maxTokens: 6, rank: 7))
        // The need charges every module a stack may touch: the two above among them.
        let modules = Krea2DiT.loraModules(config: config)
        XCTAssertEqual(modules.count, 8 * config.layers + 2)   // the blocks, img_in, the final projection
        XCTAssertTrue(modules.contains { $0.k == d && $0.n == d } && modules.contains { $0.k == hidden && $0.n == d })
    }

    /// **A long prompt is held by the fusion's arena**: at 200 tokens the former sum (`normed`, `merged`,
    /// `o` counted at 2560 + 2 × 6144 for three 6144-wide slices, 8 MB of margin) was 1.2 MB short of
    /// its own slices — the fusion threw past ~170 tokens. The published widths, no tensor read.
    func testTheFusionHoldsALongPrompt() throws {
        let artifact = try Artifact(path: try map(krea2Header([:])))
        XCTAssertEqual(Krea2DiT.Config(header: artifact.header).dim, 6144)
        let fusion = try Krea2TextFusion(artifact: artifact, maxTokens: 200)
        XCTAssertEqual(fusion.arenaCapacity, 649_887_744)
        let former = (7 * 2400 * 2560 + 3 * 2400 * 6912 + 6144 * 6144 + 2400 * 6144 * 2) * 4 + 16 * Arena.alignment + (8 << 20)
        XCTAssertLessThan(former, fusion.arenaCapacity)
    }

    /// **The arenas hold what is written, not a byte of margin**: at 1024² and 25 text rows, the
    /// published shapes, slice by slice, each on its page.
    func testTheArenasAtTenTwentyFourAreTheirSlices() {
        let config = MemoryFixtures.krea2Maps().dit
        let (s, t, d, h, kv) = (4096, 25, 6144, 16_384, 1536)
        let n = s + t
        let dit = pages(Array(repeating: n * d, count: 6) + [n * kv, n * kv, n * h, 6 * d, d * h, s * 64, s * 64, n * 128])
        XCTAssertEqual(Krea2DiT.arenaBytes(config: config, latentHeight: 128, latentWidth: 128, maxText: t, rank: 0), dit)
        XCTAssertEqual(dit, 1_335_459_840)   // h1 lies over the attention's scratch: 270 MB less
        let r = 12 * t
        let fusion = pages([r * 2560, r * 6144, r * 2560, r * 2560, r * 2560, r * 2560, r * 6144, r * 6144,
                            r * 6912, r * 6912, r * 6912, 6144 * 6144])
        XCTAssertEqual(Krea2TextFusion.arenaBytes(outDim: d, maxTokens: t, rank: 0), fusion)
        XCTAssertEqual(fusion, 213_434_368)
    }

    // ── the formula on Krea 2's shapes ────────────────────────────────────────────────────────

    /// **Computed phase by phase**: text, the fusion, the DiT, decoding, each from its constructors'
    /// functions; the peak grows with the format (512² < 1024² < 1024×1536).
    func testTheNeedIsComputedPhaseByPhase() {
        let maps = MemoryFixtures.krea2Maps()
        var peaks: [Int] = []
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            let n = need(w, h)
            XCTAssertEqual(n.phases.map(\.kind), [.text, .denoising, .denoising, .decoding])
            XCTAssertEqual(n.phases[0].bytes("encoder arena"), TextEncoder.arenaBytes(config: maps.encoder, sequence: 59))
            XCTAssertEqual(n.phases[0].bytes("taps stack"), Arena.capacity(for: 59 * 12 * 2560 * 4))
            XCTAssertEqual(n.phases[1].bytes("fusion arena"), Krea2TextFusion.arenaBytes(outDim: 6144, maxTokens: 25, rank: 0))
            XCTAssertEqual(n.phases[2].bytes("DiT arena"),
                           Krea2DiT.arenaBytes(config: maps.dit, latentHeight: h / 8, latentWidth: w / 8, maxText: 25, rank: 0))
            XCTAssertEqual(n.phases[2].bytes("map staging"), TailStream.unitSlots * maps.ditUnitStaging)
            let plan = AnimaVAE.plan(latentHeight: h / 8, latentWidth: w / 8)
            XCTAssertEqual(n.phases[3].bytes("Qwen-Image decoder"), plan.registers + plan.workspace + plan.host + plan.outside)
            // The taps, the fused text, the modulation table (50 σ at most) and the latent cross every phase.
            XCTAssertEqual(n.crossing, 25 * 12 * 2560 * 4 + 25 * 6144 * 4 + 50 * 7 * 6144 * 4 + 16 * (h / 8) * (w / 8) * 4)
            // The DiT is the peak.
            XCTAssertEqual(n.phases.map(\.bytes).max(), n.phases[2].bytes)
            peaks.append(n.peak)
        }
        XCTAssertEqual(peaks, peaks.sorted()); XCTAssertEqual(Set(peaks).count, 3)
    }

    /// **A staging block is the map's largest layer**: 868 MB in bf16 (the 8 Linears of a block and its
    /// fp32 norms and table, each on its pages), about half in 8 bits, a quarter in 4. The three
    /// versions differ there, and in the encoder's staging — nowhere else.
    func testTheVersionsDifferInTheirStagingAlone() {
        XCTAssertEqual(MemoryFixtures.krea2Maps().ditLayerStaging, 868_466_688)
        XCTAssertEqual(MemoryFixtures.krea2Maps(ditBytesPerValue: 1).ditLayerStaging, 434_651_136)
        XCTAssertEqual(MemoryFixtures.krea2Maps(ditBytesPerValue: 0).ditLayerStaging, 217_300_992)
        let standard = need(1024, 1024)
        for bytes in [1, 0] {
            let other = need(1024, 1024, maps: MemoryFixtures.krea2Maps(ditBytesPerValue: bytes, encoderBytesPerValue: 1))
            let maps = MemoryFixtures.krea2Maps(ditBytesPerValue: bytes, encoderBytesPerValue: 1)
            XCTAssertEqual(other.phases[2].bytes("map staging"), TailStream.unitSlots * maps.ditUnitStaging)
            XCTAssertEqual(other.phases[2].bytes - other.phases[2].bytes("map staging") - other.phases[2].bytes("modulation"),
                           standard.phases[2].bytes - standard.phases[2].bytes("map staging") - standard.phases[2].bytes("modulation"))
            XCTAssertLessThan(other.phases[0].bytes, standard.phases[0].bytes)
            XCTAssertEqual(other.phases[1], standard.phases[1]); XCTAssertEqual(other.phases[3], standard.phases[3])
        }
        // A map left whole to the cache (`MemoryPlan.Tail.none`): no staging block.
        var cached = MemoryFixtures.krea2Request(1024, 1024)
        cached.tail = .none
        XCTAssertEqual(MemoryNeed.krea2(cached, maps: MemoryFixtures.krea2Maps()).phases[2].bytes("map staging"), 0)
        // The lean plan drops the encoder's buffers: the text phase only.
        let lean = MemoryNeed.krea2(MemoryFixtures.krea2Request(1024, 1024).lean, maps: MemoryFixtures.krea2Maps())
        XCTAssertEqual(lean.phases[0].bytes("encoder staging"), 0)
        XCTAssertEqual(lean.phases[1...], standard.phases[1...])
    }

    /// **A rotated weight widens through a scratch outside the arenas** (`Widen.dequantizeRotated`):
    /// each job alive at once holds its tile's integers, `G × w` `Int16` (2 MB at G = 256, w = 4096),
    /// and the call its `n` factors. The Compact's encoder and DiT are Comfy-Org's int8 convrot: the
    /// text phase charges the encoder's largest (its `gate_proj`, `[2560, 9728]`: 10 groups × 3 tiles),
    /// the DiT's phase the DiT's; an unrotated map charges nothing. Found by the Compact's measured text
    /// stage, 4 to 7 MB above a formula that did not hold it.
    func testARotatedWeightChargesItsScratch() {
        XCTAssertEqual(Widen.rotationScratchBytes(rows: 2560, columns: 9728, group: 256, workers: 10),
                       Arena.capacity(for: 9728 * 4) + 10 * 2_097_152)
        XCTAssertEqual(Widen.rotationScratchBytes(rows: 256, columns: 4096, group: 256, workers: 10),
                       16384 + 2_097_152, "one job: never more workers than jobs")
        XCTAssertEqual(Widen.rotationScratchBytes(rows: 512, columns: 1024, group: 256, workers: 10),
                       16384 + 2 * 524_288)
        XCTAssertEqual(Widen.rotationScratchBytes(header: ["tensors": ["w": ["shape": [512, 1024]]]]), 0, "not rotated")
        let workers = Widen.widenWorkers
        let rotated = MemoryFixtures.krea2Maps(ditBytesPerValue: 1, encoderBytesPerValue: 1, rotated: true)
        XCTAssertEqual(rotated.encoderRotationScratch,
                       Widen.rotationScratchBytes(rows: 2560, columns: 9728, group: 256, workers: workers))
        XCTAssertEqual(rotated.ditRotationScratch,
                       Widen.rotationScratchBytes(rows: 6144, columns: 16_384, group: 256, workers: workers))
        let plain = MemoryFixtures.krea2Maps(ditBytesPerValue: 1, encoderBytesPerValue: 1)
        XCTAssertEqual(plain.encoderRotationScratch, 0); XCTAssertEqual(plain.ditRotationScratch, 0)
        let a = need(512, 512, maps: rotated), b = need(512, 512, maps: plain)
        XCTAssertEqual(a.phases[0].bytes - b.phases[0].bytes, rotated.encoderRotationScratch)
        XCTAssertEqual(a.phases[0].bytes("encoder rotation scratch"), rotated.encoderRotationScratch)
        XCTAssertEqual(a.phases[2].bytes("DiT rotation scratch"), rotated.ditRotationScratch)
    }

    /// **The timestep weights the tabulation widens**: the three published fp32 and kept so (the
    /// installed Standard's header, 17/10), read in place — and a map that would narrow one widens it
    /// for its call: `time_mod_proj`, 906 MB.
    func testTheTabulationWidensWhatIsNotFloat() {
        let maps = MemoryFixtures.krea2Maps()
        XCTAssertEqual(maps.timestepWidening, 0)
        var header: [String: Any] = ["tensors": ["time_mod_proj.weight": ["dtype": "bfloat16", "shape": [6144, 36_864]]]]
        XCTAssertEqual(Krea2DiT.timestepWidening(header: header, config: maps.dit), 905_969_664)
        header = ["tensors": ["time_embed.linear_1.weight": ["dtype": "float32"], "time_embed.linear_2.weight": ["dtype": "float32"],
                              "time_mod_proj.weight": ["dtype": "float32"]]]
        XCTAssertEqual(Krea2DiT.timestepWidening(header: header, config: maps.dit), 0)
        // Alive before any evaluation, the arena not yet written: the phase is the larger of the two moments.
        let n = need(512, 512)
        let tabulation = Krea2DiT.tabulationBytes(config: maps.dit, count: Request.maximumSteps) + maps.timestepWidening
        let others = n.phases[2].terms.filter { !$0.name.hasPrefix("modulation tabulation") }.reduce(0) { $0 + $1.bytes }
        XCTAssertEqual(n.phases[2].bytes("modulation tabulation"), max(0, tabulation - others))
    }

    /// **The text follows the prompt, not 512**: the encoder's rows are the template's 34, the prompt
    /// capped where `Krea2Text.admitted` refuses (1 024), and the suffix's 5; the DiT's are the prompt's alone.
    func testTheTextFollowsThePrompt() {
        XCTAssertEqual(MemoryNeed.Request.promptTokens(text: 20, family: .krea2), 59)
        XCTAssertEqual(MemoryNeed.Request.promptTokens(text: 9000, family: .krea2), 1063)
        XCTAssertEqual(MemoryNeed.Request.promptTokens("woman posing in a library", family: .krea2), 64)
        let short = need(1024, 1024, tokens: 59), long = need(1024, 1024, tokens: 1063)
        for phase in 0..<3 { XCTAssertLessThan(short.phases[phase].bytes, long.phases[phase].bytes) }
        XCTAssertEqual(long.phases[2].bytes("DiT arena"),
                       Krea2DiT.arenaBytes(config: MemoryFixtures.krea2Maps().dit, latentHeight: 128, latentWidth: 128,
                                           maxText: Request.maximumPromptTokens + Krea2Text.suffixTokens, rank: 0))
        XCTAssertEqual(need(1024, 1024, tokens: 1063), need(1024, 1024, tokens: 4000))
    }

    /// **The GQA SDPA past `maxMatrix` runs by query slices** — 1024×1536 — and MPSGraph concatenates them:
    /// one more copy of the queries' size. Cut by groups, the GPU holds 7 of the 12 groups, 28 of the 48 heads:
    /// its slices and its copies are theirs, and the CPU's five groups are the head split's arena.
    func testTheSlicedAttentionIsCharged() {
        XCTAssertEqual(Attention.groupedSlices(heads: 48, rows: 4121, keys: 4121), 1)
        XCTAssertEqual(Attention.groupedSlices(heads: 48, rows: 4608, keys: 4608), 2)
        XCTAssertEqual(Attention.groupedSlices(heads: 48, rows: 6169, keys: 6169), 3)
        XCTAssertEqual(HeadSplitAttention.partition(heads: 48, fraction: HeadSplitAttention.referenceFraction, group: 4).cpu, 28..<48)
        XCTAssertEqual(Attention.groupedSlices(heads: 28, rows: 4121, keys: 4121), 1)
        XCTAssertEqual(Attention.groupedSlices(heads: 28, rows: 6169, keys: 6169), 2)
        XCTAssertEqual(need(1024, 1024).phases[2].bytes("DiT SDPA concatenation"), 0)
        let tall = need(1024, 1536)
        XCTAssertEqual(tall.phases[2].bytes("DiT SDPA concatenation"), 6169 * 28 * 128 * 4)
        XCTAssertEqual(tall.phases[2].bytes("DiT SDPA queries"), 6169 * 28 * 128 * 4)
        XCTAssertEqual(tall.phases[2].bytes("DiT head split"),
                       HeadSplitAttention.hostBytes(heads: 48, sequence: 6169, headDim: 128, cpuHeads: 20,
                                                    threads: MemoryFixtures.krea2Request(1024, 1536, tokens: 59, ranks: [], references: [], maps: MemoryFixtures.krea2Maps()).threads))
        // At 512² (1,024 image rows) the attention is not cut (`Krea2DiT.headSplitRows`): one SDPA, all 48 heads.
        let small = need(512, 512)
        XCTAssertEqual(small.phases[2].bytes("DiT head split"), 0)
        XCTAssertGreaterThan(need(1024, 1024).phases[2].bytes("DiT head split"), 0)
    }

    /// **A LoRA costs three things in the DiT** — its `mid` slice, the cache of every module it may touch
    /// (~431 MB at rank 32), a `down` staged with two LoRAs or more — and its slices in the fusion.
    func testTheLoRAIsCharged() {
        let config = MemoryFixtures.krea2Maps().dit
        let bare = need(1024, 1024), one = need(1024, 1024, ranks: [32]), two = need(1024, 1024, ranks: [32, 16])
        XCTAssertEqual(one.phases[2].bytes("DiT arena"),
                       Krea2DiT.arenaBytes(config: config, latentHeight: 128, latentWidth: 128, maxText: 25, rank: 32))
        XCTAssertEqual(one.phases[2].bytes("LoRA cache"), 430_997_504)   // img_in's 802 816 among them
        XCTAssertEqual(one.phases[2].bytes("LoRA down staging"), 0)
        XCTAssertEqual(two.phases[2].bytes("LoRA down staging"), LoRA.downBytes(k: 16_384, rank: 32))
        XCTAssertEqual(two.phases[2].bytes("DiT arena"),
                       Krea2DiT.arenaBytes(config: config, latentHeight: 128, latentWidth: 128, maxText: 25, rank: 48))
        XCTAssertEqual(one.phases[1].bytes("fusion arena"), Krea2TextFusion.arenaBytes(outDim: 6144, maxTokens: 25, rank: 32))
        XCTAssertEqual(two.phases[1].bytes("LoRA down staging"), LoRA.downBytes(k: 6912, rank: 32))
        XCTAssertEqual(one.phases[0], bare.phases[0]); XCTAssertEqual(one.phases[3], bare.phases[3])
    }

    /// **img2img**: the starting image is encoded at the render's format by the Qwen-Image encoder; the
    /// image as handed and its z₀ cross the render.
    func testTheStartImageIsEncodedAtTheFormat() {
        let bare = need(1024, 1024), other = need(1024, 1024, references: [(800, 600)])
        XCTAssertEqual(other.phases.map(\.kind), [.text, .references, .denoising, .denoising, .decoding])
        XCTAssertEqual(other.phases[1].bytes("fitted copy"), 3 * 1024 * 1024 * 4)
        XCTAssertEqual(other.crossing - bare.crossing, 3 * 800 * 600 * 4 + 16 * 128 * 128 * 4)
    }

    /// **The plan in B**: on a 16 GB Mac (B = 10 GiB) every format of every version passes the
    /// preflight with a rank-32 LoRA, and the map streams (26 GB, more than B): the plan's need ≤ B, the
    /// two staging blocks inside it. A library without Krea 2's maps: unknown, and refused.
    func testThePlanIsChosenInB() throws {
        let ceiling = MemoryBudget.ceiling(physical: 16 << 30)
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            for dit in [2, 1, 0] {
                let maps = MemoryFixtures.krea2Maps(ditBytesPerValue: dit, encoderBytesPerValue: dit == 2 ? 2 : 1)
                let request = MemoryFixtures.krea2Request(w, h, tokens: 546, ranks: [32], maps: maps)
                XCTAssertNoThrow(try ModelCard.krea2.checkMemory(request, ceiling: ceiling), "\(w)×\(h) \(dit)")
                let plan = MemoryPlan.choose(request, ceiling: ceiling)
                let need = try XCTUnwrap(plan.need)
                XCTAssertLessThanOrEqual(need.peak, ceiling)
                XCTAssertEqual(plan.request.encoderSlots, MemoryPlan.encoderSlots)
            }
        }
        // The bf16 map's 28 blocks: none held beside the denoising, the whole map streamed.
        let spans = Array(repeating: MemoryFixtures.krea2Maps().ditLayerStaging, count: 28)
        let plan = MemoryPlan.choose(MemoryFixtures.krea2Request(1024, 1024), ceiling: ceiling)
        XCTAssertEqual(MemoryPlan.released(spans: spans, mapBytes: 26_300_000_000, ceiling: ceiling,
                                           streamed: plan.denoisingStreamed, cached: plan.denoisingCached,
                                           fraction: Krea2DiT.generationTail).released, 0)
        var bare = MemoryFixtures.krea2Request(1024, 1024)
        bare.krea2Maps = nil
        bare.library = Library(root: root)
        XCTAssertNil(MemoryNeed.of(bare))
        XCTAssertThrowsError(try ModelCard.krea2.checkMemory(bare, ceiling: 1 << 40)) {
            XCTAssertEqual($0 as? EngineError, .memoryNeedUnknown(model: "krea2"))
        }
    }

    /// **Each phase is checked against its own need** (`MemoryCheck`): the fusion and the DiT both under
    /// the `denoising` mark, judged against the larger of the two.
    func testThePhasesAreChecked() {
        let n = need(1024, 1024)
        XCTAssertEqual(n.need(of: .denoising), max(n.phases[1].bytes, n.phases[2].bytes) + n.crossing)
        let marks = [("start", 0), ("text", n.need(of: .text)!), ("denoising", n.need(of: .denoising)! + 1),
                     ("denoiser released", 0), ("decoding", n.need(of: .decoding)!)]
        let check = MemoryCheck(need: n, marks: marks, peakMethod: .kernelInterval)
        XCTAssertEqual(check.exceeded.map(\.name), ["denoising"])
    }
}
