import XCTest
@testable import Siliconed

/// **Anima's computed need, without a byte of weights.** The DiT's arena is built on a map written by
/// hand and compared with its shape function; the formula runs on headers written in memory with
/// Anima's published shapes (`MemoryFixtures.animaMaps`). No figure here depends on the Mac: the terms
/// pinned are the arenas and the VAE's plans, never the AMX's workspace (`EngineSettings`).
final class MemoryNeedAnimaTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-anima-need-\(UUID())")
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

    private let tinyDiT: [String: Any] = [
        "kind": "anima-turbo-dit", "rope_layout": "interleaved",
        "config": ["num_attention_heads": 2, "attention_head_dim": 8, "mlp_ratio": 4.0, "num_layers": 1,
                   "in_channels": 4, "patch_size": [1, 2, 2], "adaln_lora_dim": 4, "text_embed_dim": 8] as [String: Any],
    ]

    private func need(_ w: Int, _ h: Int, tokens: Int = 50, ranks: [Int] = [],
                      references: [(width: Int, height: Int)] = [],
                      maps: MemoryNeed.AnimaMaps = MemoryFixtures.animaMaps()) -> MemoryNeed {
        MemoryNeed.anima(MemoryFixtures.animaRequest(w, h, tokens: tokens, ranks: ranks, references: references, maps: maps),
                         maps: maps)
    }

    // ── the formula is the allocation ─────────────────────────────────────────────────────────

    /// The DiT's arena, built, is its shape function — with and without a LoRA stack (two LoRAs on
    /// one module: cumulative rank 7; the widest module they touch sizes `down` and `up`).
    func testTheArenaIsTheFormula() throws {
        let artifact = try Artifact(path: try map(tinyDiT))
        let config = AnimaDiT.Config(header: artifact.header)
        for (h, w, text) in [(8, 8, 6), (16, 8, 10)] {
            let dit = try AnimaDiT(artifact: artifact, latentHeight: h, latentWidth: w, textLength: text)
            XCTAssertEqual(dit.arenaCapacity, AnimaDiT.arenaBytes(config: config, latentHeight: h, latentWidth: w,
                                                                  textLength: text, lora: nil))
        }
        let (d, hidden) = (config.dim, config.hidden)
        func lora(_ rank: Int, _ targets: [(String, Int, Int)]) throws -> LoRA.Layer {
            let tensors = targets.flatMap { [("\($0.0).down", [$0.1, rank]), ("\($0.0).up", [rank, $0.2])] }
            let path = try map(["kind": "lora"], tensors: tensors.map { (name: $0.0, shape: $0.1) })
            return LoRA.Layer(artifact: try Artifact(path: path), strength: 1, name: path)
        }
        let stack = try LoRA(layers: [
            lora(4, [("transformer_blocks.0.attn1.to_q", d, d), ("transformer_blocks.0.ff.net.2", hidden, d)]),
            lora(3, [("transformer_blocks.0.attn1.to_q", d, d)]),
        ])
        XCTAssertEqual(stack.maxRank, 7)
        let dit = try AnimaDiT(artifact: artifact, latentHeight: 8, latentWidth: 8, textLength: 6, lora: stack)
        XCTAssertEqual(dit.arenaCapacity, AnimaDiT.arenaBytes(config: config, latentHeight: 8, latentWidth: 8, textLength: 6,
                                                              lora: (7, stack.maxEntry, stack.maxOutput)))
        // The need charges the widest module a stack may touch: never below the stack's own.
        let bounds = AnimaDiT.loraBounds(config: config)
        XCTAssertGreaterThanOrEqual(bounds.entry, stack.maxEntry)
        XCTAssertGreaterThanOrEqual(bounds.output, stack.maxOutput)
    }

    /// **The arena holds what is written, not a byte of margin**: at 1024², the published shapes,
    /// slice by slice, each on its page — the double scratch of the modulation (`d × 3d`) and the text
    /// k/v of the 28 blocks included.
    func testTheArenaAtTenTwentyFourIsItsSlices() {
        let config = MemoryFixtures.animaMaps().dit
        let (s, d, h, t) = (4096, 2048, 8192, 512)
        func pages(_ bytes: [Int]) -> Int { bytes.reduce(0) { $0 + Arena.capacity(for: $1) } }
        let floats = [Array(repeating: s * d, count: 7), [d * h, s * 68, s * 64, s * 128, t * 1024, 28 * t * d, 28 * t * d,
                                     2 * d]].flatMap { $0 }
        let expected = pages(floats.map { $0 * 4 } + [3 * d * d * 8])
        XCTAssertEqual(AnimaDiT.arenaBytes(config: config, latentHeight: 128, latentWidth: 128, textLength: 512, lora: nil),
                       expected)
        XCTAssertEqual(expected, 643_907_584)   // h1 lies over q…merged: 134 MB less
    }

    /// **The modulation table** (`AnimaDiT.tabulate`): per σ, `temb` 3d, `embedded` d, the 84 sublayers'
    /// `3d` and `norm_out`'s `2d`; then the computation's scratch at that many rows. The need charges it
    /// at `tableRows` (8) whatever the steps — a longer render tabulates 8 σ at a time.
    func testTheModulationTableIsItsSlices() {
        let config = MemoryFixtures.animaMaps().dit
        let (m, d) = (8, 2048)
        func pages(_ bytes: [Int]) -> Int { bytes.reduce(0) { $0 + Arena.capacity(for: $1) } }
        XCTAssertEqual(AnimaDiT.ModulationRow.floats(config), 4 * d + 28 * 9 * d + 2 * d)
        let expected = pages([m * (4 * d + 28 * 9 * d + 2 * d) * 4, m * d * 4, m * d * 4, m * 3 * d * 4, m * 256 * 4,
                              m * d * 8, m * 3 * d * 8])
        XCTAssertEqual(AnimaDiT.tableBytes(config: config, count: m), expected)
        XCTAssertEqual(expected, 17_776_640)
        XCTAssertEqual(AnimaDiT.tableRows, 8)
        XCTAssertEqual(need(1024, 1024).phases[1].bytes("modulation table"), expected)
        XCTAssertEqual(need(512, 512).phases[1].bytes("modulation table"), expected)
    }

    /// The conditioner's arena: 512 T5 rows reading 50 Qwen rows, in double, one `mlp.0` weight and
    /// its fp32 widening — and it grows with the Qwen rows (`source`, then `k`, `v` past 512).
    func testTheConditionerArenaIsItsSlices() {
        let (t, q, d, h) = (512, 50, 1024, 4096)
        func pages(_ bytes: [Int]) -> Int { bytes.reduce(0) { $0 + Arena.capacity(for: $1) } }
        let expected = pages([q * d, t * d, t * d, t * d, t * d, t * d, t * d, t * d, t * h, h * d, h, d, t].map { $0 * 8 }
                             + [h * d * 4, d * 4])
        XCTAssertEqual(AnimaConditioner.arenaBytes(t5: t, qwen: q), expected)
        XCTAssertEqual(expected, 96_960_512)
        XCTAssertLessThan(AnimaConditioner.arenaBytes(t5: t, qwen: 50), AnimaConditioner.arenaBytes(t5: t, qwen: 512))
    }

    // ── the formula on Anima's shapes ─────────────────────────────────────────────────────────

    /// **Computed, never the fallback**: text, denoising, decoding, each from its constructors'
    /// functions; the peak grows with the format (512² < 1024² < 1024×1536).
    func testTheNeedIsComputedPhaseByPhase() {
        let maps = MemoryFixtures.animaMaps()
        var peaks: [Int] = []
        let rows = AnimaConditioner.Shape.maximumTokens
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            let n = need(w, h)
            XCTAssertEqual(n.phases.map(\.kind), [.text, .denoising, .decoding])
            XCTAssertEqual(n.phases[1].bytes("DiT arena"),
                           AnimaDiT.arenaBytes(config: maps.dit, latentHeight: h / 8, latentWidth: w / 8, textLength: rows, lora: nil))
            XCTAssertEqual(n.phases[0].bytes("conditioner arena"), AnimaConditioner.arenaBytes(t5: rows, qwen: 50))
            let plan = AnimaVAE.plan(latentHeight: h / 8, latentWidth: w / 8)
            XCTAssertEqual(n.phases[2].bytes("Qwen-Image decoder"),
                           plan.registers + plan.workspace + plan.host + plan.outside)
            // The conditioning (its bound × 1024) and the latent cross every phase.
            XCTAssertEqual(n.crossing, rows * 1024 * 4 + 16 * (h / 8) * (w / 8) * 4)
            peaks.append(n.peak)
        }
        XCTAssertEqual(peaks, peaks.sorted()); XCTAssertEqual(Set(peaks).count, 3)
    }

    /// **Standard, Compact, Light: one denoising need.** The DiT's map is read through the page cache
    /// and widened into the arena's fp32 `reserve` whatever its type: the formula does not read it.
    /// The encoder's type does show — its staging buffers, read from its header —, in the text phase alone.
    func testTheThreeVersionsShareTheDenoising() {
        let standard = need(1024, 1024)
        for dit in [1, 0] {
            XCTAssertEqual(need(1024, 1024, maps: MemoryFixtures.animaMaps(ditBytesPerValue: dit)), standard)
        }
        let eight = need(1024, 1024, maps: MemoryFixtures.animaMaps(ditBytesPerValue: 1, encoderBytesPerValue: 1))
        XCTAssertLessThan(eight.phases[0].bytes, standard.phases[0].bytes)
        XCTAssertEqual(eight.phases[1...], standard.phases[1...])
        // The lean plan drops the encoder's buffers: the text phase only.
        let lean = MemoryNeed.anima(MemoryFixtures.animaRequest(1024, 1024).lean, maps: MemoryFixtures.animaMaps())
        XCTAssertEqual(lean.phases[0].bytes("encoder staging"), 0)
        XCTAssertEqual(lean.phases[1...], standard.phases[1...])
    }

    /// **The DiT's text is charged at its bound whatever the prompt** (T5's rows are not bounded by
    /// the bytes): a long prompt grows the encoder and the conditioner's `source`, never the denoising
    /// (its text k/v cache is `[28, 1 025, d]`).
    func testTheDiTsTextIsChargedAtItsBound() {
        let short = need(1024, 1024, tokens: 5), long = need(1024, 1024, tokens: 500)
        XCTAssertLessThan(short.phases[0].bytes, long.phases[0].bytes)
        XCTAssertEqual(short.phases[1], long.phases[1])
        // The Qwen ids are refused past 1 024: beyond, nothing grows.
        XCTAssertEqual(need(1024, 1024, tokens: 1024), need(1024, 1024, tokens: 4000))
        XCTAssertEqual(MemoryNeed.Request.promptTokens("woman posing in a library", family: .anima), 25)
        XCTAssertEqual(MemoryNeed.Request.promptTokens(text: 9000, family: .anima), 1024)
    }

    /// **A LoRA lives in the DiT's arena**, at the stack's cumulative rank and the widest module it may
    /// touch; two LoRAs stage a `down` (`LoRA.downBytes`) beside it.
    func testTheLoRAIsInTheArena() {
        let config = MemoryFixtures.animaMaps().dit
        let bounds = AnimaDiT.loraBounds(config: config)
        XCTAssertEqual(bounds.entry, 8192); XCTAssertEqual(bounds.output, 8192)
        let bare = need(1024, 1024), one = need(1024, 1024, ranks: [32]), two = need(1024, 1024, ranks: [32, 16])
        func arena(_ rank: Int) -> Int {
            AnimaDiT.arenaBytes(config: config, latentHeight: 128, latentWidth: 128,
                                textLength: AnimaConditioner.Shape.maximumTokens,
                                lora: rank > 0 ? (rank, bounds.entry, bounds.output) : nil)
        }
        XCTAssertEqual(one.phases[1].bytes("DiT arena") - bare.phases[1].bytes("DiT arena"), arena(32) - arena(0))
        XCTAssertEqual(two.phases[1].bytes("DiT arena"), arena(48))
        XCTAssertEqual(two.phases[1].bytes("LoRA down staging"), LoRA.downBytes(k: 8192, rank: 32))
        XCTAssertEqual(one.phases[1].bytes("LoRA down staging"), 0)
        XCTAssertEqual(one.phases[0], bare.phases[0]); XCTAssertEqual(one.phases[2], bare.phases[2])
    }

    /// **img2img**: the starting image is encoded at the render's format — a `fitted` copy when it was
    /// handed at another size —, by the Qwen-Image encoder's two heaps; the image as handed and its z₀
    /// cross the render.
    func testTheStartImageIsEncodedAtTheFormat() {
        let bare = need(1024, 1024)
        let same = need(1024, 1024, references: [(1024, 1024)]), other = need(1024, 1024, references: [(800, 600)])
        XCTAssertEqual(same.phases.map(\.kind), [.text, .references, .denoising, .decoding])
        let plan = VAEEncoder.plan(family: .qwenImage, height: 1024, width: 1024)
        XCTAssertEqual(same.phases[1].bytes("Qwen-Image encoder"), plan.registers + plan.workspace + plan.host + plan.outside)
        XCTAssertEqual(same.phases[1].bytes("fitted copy"), 0)
        XCTAssertEqual(other.phases[1].bytes("fitted copy"), 3 * 1024 * 1024 * 4)
        let latent = 16 * 128 * 128 * 4
        XCTAssertEqual(same.crossing - bare.crossing, 3 * 1024 * 1024 * 4 + latent)
        XCTAssertEqual(other.crossing - bare.crossing, 3 * 800 * 600 * 4 + latent)
    }

    /// **Each phase is checked against its own need** (`MemoryCheck`): the engine's boundaries map to
    /// the formula's phases, and a phase above its need is a defect.
    func testThePhasesAreChecked() {
        let n = need(1024, 1024, references: [(1024, 1024)])
        let marks = [("start", 0), ("text", n.need(of: .text)!), ("image", n.need(of: .references)!),
                     ("denoising", n.need(of: .denoising)! + 1), ("denoiser released", 0), ("decoding", n.need(of: .decoding)!)]
        let check = MemoryCheck(need: n, marks: marks, peakMethod: .kernelInterval)
        XCTAssertEqual(check.phases.filter { $0.needBytes != nil }.map(\.name), ["text", "image", "denoising", "denoiser released", "decoding"])
        XCTAssertEqual(check.exceeded.map(\.name), ["denoising"])
    }

    /// **The plan in B**: on a 16 GB Mac (B = 10 GiB) and an 8 GB one (B = 4 GiB), every format of every
    /// version passes the preflight and takes the fastest plan (the encoder's four buffers, or its 0.9 GB kept where the map stays
    /// whole beside it), its need ≤ B; a B below the floor refuses before any computation.
    func testThePlanIsChosenInB() throws {
        for physical in [16, 8] {
            let ceiling = MemoryBudget.ceiling(physical: physical << 30)
            for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
                for dit in [2, 1, 0] {
                    let request = MemoryFixtures.animaRequest(w, h, maps: MemoryFixtures.animaMaps(ditBytesPerValue: dit))
                    XCTAssertNoThrow(try ModelCard.anima.checkMemory(request, ceiling: ceiling), "\(physical) GB \(w)×\(h)")
                    let plan = MemoryPlan.choose(request, ceiling: ceiling)
                    let need = try XCTUnwrap(plan.need)
                    XCTAssertLessThanOrEqual(need.peak, ceiling)
                    XCTAssertTrue(plan.request.encoderKept || plan.request.encoderSlots == MemoryPlan.encoderSlots)
                    XCTAssertEqual(plan.request.encoderKept, need.keptEncoderBytes > 0)
                    XCTAssertLessThanOrEqual(plan.floor, need.peak)
                }
            }
        }
        let request = MemoryFixtures.animaRequest(1024, 1024)
        let floor = try XCTUnwrap(ModelCard.anima.memoryNeed(request))
        XCTAssertThrowsError(try ModelCard.anima.checkMemory(request, ceiling: floor - 1)) {
            XCTAssertEqual($0 as? EngineError, .insufficientMemory(needed: floor, budget: floor - 1))
        }
        // A library without Anima's maps: the need is unknown, and the render refused.
        var bare = request
        bare.animaMaps = nil
        bare.library = Library(root: root)
        XCTAssertNil(MemoryNeed.of(bare))
        XCTAssertThrowsError(try ModelCard.anima.checkMemory(bare, ceiling: 1 << 40)) {
            XCTAssertEqual($0 as? EngineError, .memoryNeedUnknown(model: "anima"))
        }
    }
}
