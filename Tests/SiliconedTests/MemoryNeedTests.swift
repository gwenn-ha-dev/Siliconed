import XCTest
@testable import Siliconed

/// **The computed memory need, without a byte of weights.** The arenas are built on maps
/// written by hand (a header, a few bytes of data) and compared with the formula's functions; the
/// formula itself runs on headers written in memory with Z-Image's real shapes.
final class MemoryNeedTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-need-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    // ── maps written by hand ──────────────────────────────────────────────────────────────────

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

    private let tinyEncoder: [String: Any] = [
        "last_layer": 1,
        "config": ["hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                   "intermediate_size": 16],
    ]
    private let tinyDiT: [String: Any] = [
        "config": ["dim": 64, "n_heads": 2, "n_layers": 1, "n_refiner_layers": 1, "in_channels": 16,
                   "cap_feat_dim": 16, "axes_dims": [8, 12, 12], "axes_lens": [64, 32, 32]],
    ]

    // ── the formula is the allocation ─────────────────────────────────────────────────────────

    /// A map linked into the store weighs what it points to, alone or in a folder: read on the link,
    /// a few dozen bytes made the need wrong and a render was refused.
    func testAFileReachedByALinkWeighsItsFile() throws {
        let file = root.appendingPathComponent("map.bin").path
        try Data(count: 4_096).write(to: URL(fileURLWithPath: file))
        let folder = root.appendingPathComponent("store")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let link = folder.appendingPathComponent("linked.bin").path
        try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: file)
        XCTAssertEqual(Library.fileSize(link), 4_096)
        XCTAssertEqual(MemoryNeed.InstalledFiles.size(link), 4_096)
        XCTAssertEqual(Library.size(link), 4_096)
        XCTAssertEqual(Library.size(folder.path), 4_096)
        XCTAssertNil(Library.fileSize(root.appendingPathComponent("absent").path))
    }

    /// **A broken link is a file absent** — no size, the same stamp, the need's « unknown » —, and a
    /// folder weighs each file it reaches once: a linked folder what it holds, a link beside its
    /// target nothing more.
    func testABrokenLinkIsAnAbsentFileAndAFolderCountsEachFileOnce() throws {
        let fm = FileManager.default
        let store = root.appendingPathComponent("store")
        try fm.createDirectory(at: store.appendingPathComponent("composants"), withIntermediateDirectories: true)
        let file = store.appendingPathComponent("map.silicon").path
        try Data(count: 4_096).write(to: URL(fileURLWithPath: file))
        try Data(count: 100).write(to: store.appendingPathComponent("composants/vae.safetensors"))
        let absent = root.appendingPathComponent("absent").path
        let broken = store.appendingPathComponent("broken.silicon").path
        try fm.createSymbolicLink(atPath: broken, withDestinationPath: absent)
        XCTAssertNil(Library.fileSize(broken))
        XCTAssertEqual(MemoryNeed.InstalledFiles.size(broken), MemoryNeed.InstalledFiles.size(absent))
        XCTAssertEqual(MemoryNeed.InstalledFiles(paths: [broken]).stamps, MemoryNeed.InstalledFiles(paths: [absent]).stamps)
        XCTAssertEqual(Library.size(broken), 0)
        XCTAssertEqual(Library.regularFiles(under: store.path).map(\.path).filter { $0.hasSuffix("broken.silicon") }, [])

        let linked = store.appendingPathComponent("again.silicon").path
        try fm.createSymbolicLink(atPath: linked, withDestinationPath: file)
        XCTAssertEqual(Library.fileSize(linked), Library.fileSize(file))
        XCTAssertEqual(Library.size(store.path), 4_196, "the map once, though reached twice")

        // A store that is a link, and a folder linked inside a store: walked as folders.
        let linkedStore = root.appendingPathComponent("linked-store").path
        try fm.createSymbolicLink(atPath: linkedStore, withDestinationPath: store.path)
        XCTAssertEqual(Library.size(linkedStore), 4_196)
        let other = root.appendingPathComponent("other/store")
        try fm.createDirectory(at: other, withIntermediateDirectories: true)
        try fm.createSymbolicLink(atPath: other.appendingPathComponent("composants").path,
                                  withDestinationPath: store.appendingPathComponent("composants").path)
        XCTAssertEqual(Library.size(other.path), 100)
        // A link back to a parent is walked once.
        try fm.createSymbolicLink(atPath: store.appendingPathComponent("composants/up").path, withDestinationPath: store.path)
        XCTAssertEqual(Library.size(store.path), 4_196)
    }

    func testTheEncoderArenaIsTheFormula() throws {
        let artifact = try Artifact(path: try map(tinyEncoder))
        let config = try TextEncoder.Config(header: artifact.header)
        for sequence in [1, 5, 77] {
            let encoder = try TextEncoder(artifact: artifact, sequence: sequence)
            XCTAssertEqual(encoder.arenaCapacity, TextEncoder.arenaBytes(config: config, sequence: sequence))
        }
    }

    func testTheDiTArenasAndItsLoRACacheAreTheFormula() throws {
        let artifact = try Artifact(path: try map(tinyDiT))
        let config = DiT.Config(header: artifact.header)
        for (h, w, caption) in [(8, 8, 5), (16, 8, 40)] {
            let dit = try DiT(artifact: artifact, latentHeight: h, latentWidth: w, capLength: caption)
            XCTAssertEqual(dit.arenaCapacity, DiT.arenaBytes(config: config, latentHeight: h, latentWidth: w,
                                                             capLength: caption, loraRank: 0))
            XCTAssertEqual(dit.cacheLoRACapacity, 0)
        }
        // A LoRA of rank 4 on two modules, one of them also at rank 3 in a second LoRA (cumulative 7).
        let (d, hidden) = (config.dim, config.hidden)
        func lora(_ rank: Int, _ targets: [(String, Int, Int)]) throws -> LoRA.Layer {
            let tensors = targets.flatMap { [("\($0.0).down", [$0.1, rank]), ("\($0.0).up", [rank, $0.2])] }
            let path = try map(["kind": "lora"], tensors: tensors.map { (name: $0.0, shape: $0.1) })
            return LoRA.Layer(artifact: try Artifact(path: path), strength: 1, name: path)
        }
        let stack = try LoRA(layers: [
            lora(4, [("layers.0.attention.to_q", d, d), ("layers.0.feed_forward.w2", hidden, d)]),
            lora(3, [("layers.0.attention.to_q", d, d)]),
        ])
        XCTAssertEqual(stack.maxRank, 7)
        let dit = try DiT(artifact: artifact, latentHeight: 8, latentWidth: 8, capLength: 5, lora: stack)
        XCTAssertEqual(dit.arenaCapacity, DiT.arenaBytes(config: config, latentHeight: 8, latentWidth: 8,
                                                         capLength: 5, loraRank: 7))
        XCTAssertEqual(dit.cacheLoRACapacity,
                       CacheLoRA.arenaBytes(CacheLoRA.bytes(modules: [(d, d, 7), (hidden, d, 4)])))
    }

    /// **The arenas hold what is written, not a byte of margin**: at 1024² with a 50-token
    /// prompt, the DiT's four arenas written out slice by slice, each on its page.
    func testTheDiTArenasAtTenTwentyFourAreTheirSlices() {
        let config = DiT.Config(header: [:])
        let (d, h, image, cap) = (3840, 10240, 4096, 64), s = image + cap
        func pages(_ floats: [Int]) -> Int { floats.reduce(0) { $0 + Arena.capacity(for: $1 * 4) } }
        func block(_ s: Int) -> Int { pages(Array(repeating: s * d, count: 6) + [s * h, 4 * d, d * h, 4 * d]) }
        let own = pages([2560 * d, 256, 1024, image * 64, image * d, cap * 2560, cap * d, s * 128, s * d, d, s * d, s * 64])
        XCTAssertEqual(DiT.arenaBytes(config: config, latentHeight: 128, latentWidth: 128, capLength: 50, loraRank: 0),
                       own + max(block(image), block(cap), block(s)))   // the blocks take turns in one arena
    }

    func testTheStagingBufferIsTheFormula() throws {
        let tensors: [(name: String, shape: [Int])] = (0..<3).flatMap { (l: Int) -> [(name: String, shape: [Int])] in
            [(name: "layers.\(l).a", shape: [8, 3 + l]), (name: "layers.\(l).b", shape: [5 * (l + 1)])]
        }
        let path = try map(["kind": "test"], tensors: tensors)
        let artifact = try Artifact(path: path)
        let prefixes = (1..<3).map(DiT.layerPrefix)
        let blocks = prefixes.map { p in artifact.order.filter { $0.hasPrefix(p) } }
        try artifact.streamTail(blocks: blocks, slots: 1)
        XCTAssertEqual(artifact.stream?.slotBytes,
                       Artifact.stagingBytes(header: artifact.header, prefixes: prefixes, fileSize: artifact.size))
    }

    // ── the formula on Z-Image's shapes ───────────────────────────────────────────────────────

    /// Z-Image's maps with the published shapes (`MemoryFixtures.zImageMaps`).
    private static func zImageMaps(bytesPerValue: Int = 2) -> MemoryNeed.ZImageMaps {
        MemoryFixtures.zImageMaps(bytesPerValue: bytesPerValue)
    }

    /// A Z-Image request on 6 P cores, no process base: the same figures on every Mac.
    static func zRequest(width: Int, height: Int, references: [(width: Int, height: Int)] = [], promptTokens: Int,
                         loraRanks: [Int] = []) -> MemoryNeed.Request {
        var request = MemoryNeed.Request(card: .zImage, width: width, height: height, references: references,
                                         promptTokens: promptTokens, loraRanks: loraRanks)
        request.threads = 6
        return request
    }

    private func need(_ w: Int, _ h: Int, tokens: Int = 50, ranks: [Int] = [], references: [(width: Int, height: Int)] = [],
                      maps: MemoryNeed.ZImageMaps = MemoryNeedTests.zImageMaps()) throws -> MemoryNeed {
        let request = Self.zRequest(width: w, height: h, references: references,
                                         promptTokens: tokens, loraRanks: ranks)
        return try XCTUnwrap(MemoryNeed.zImage(request, maps: maps))
    }

    private func own(_ n: MemoryNeed, _ kind: MemoryNeed.Phase.Kind) -> Int {
        n.phases.filter { $0.kind == kind }.map(\.own).max() ?? 0
    }


    private func phase(_ n: MemoryNeed, _ kind: MemoryNeed.Phase.Kind) -> MemoryNeed.Phase? {
        n.phases.first { $0.kind == kind }
    }

    private func bytes(_ n: MemoryNeed, _ kind: MemoryNeed.Phase.Kind) -> Int {
        n.phases.filter { $0.kind == kind }.map(\.bytes).max() ?? 0
    }

    private func sum(_ terms: [MemoryNeed.Term]) -> Int { terms.reduce(0) { $0 + $1.bytes } }

    // ── what makes a render larger makes its need larger ─────────────────────────────────────

    func testTheNeedGrowsWithTheAreaThePromptAndTheRank() throws {
        let formats = [(512, 512), (832, 1216), (1024, 1024), (1024, 1536)]
        for tokens in [50, 400] {
            for ranks in [[], [64]] as [[Int]] {
                let peaks = try formats.map { try need($0.0, $0.1, tokens: tokens, ranks: ranks).peak }
                XCTAssertEqual(peaks, peaks.sorted(), "area, \(tokens) tokens, ranks \(ranks)")
            }
        }
        for (w, h) in formats {
            let bound = MemoryNeed.Request.promptTokens(text: Request.maximumPromptTokens, family: .zImage)
            let byTokens = try [1, 50, 100, 400, bound].map { try need(w, h, tokens: $0).peak }
            XCTAssertEqual(byTokens, byTokens.sorted())
            XCTAssertLessThan(byTokens[0], byTokens[4])
            // A longer prompt is refused (`Tokenizer.admitChat`): past the bound, it costs no more.
            XCTAssertEqual(try need(w, h, tokens: bound + 500).peak, byTokens[4])
            let byRank = try [[], [16], [64], [64, 32]].map { try need(w, h, ranks: $0).peak }
            XCTAssertEqual(byRank, byRank.sorted())
            // The decoding can be the peak now that the blocks share one arena: the denoising itself grows.
            XCTAssertLessThanOrEqual(byRank[0], byRank[1])
            let denoising = try [[], [16]].map { phase(try need(w, h, ranks: $0), .denoising)!.bytes }
            XCTAssertLessThan(denoising[0], denoising[1])
        }
        // An img2img start: the larger the image handed, the larger the need.
        let starts = try [(512, 512), (1024, 1024), (2048, 2048)].map { try need(1024, 1024, references: [$0]).peak }
        XCTAssertEqual(starts, starts.sorted()); XCTAssertEqual(Set(starts).count, 3)
    }

    /// 512², 1024² and 1024×1536 take the same path: the same phases, the tail's two buffers at all three.
    func testOnePathForEveryFormat() throws {
        let kinds = try [(512, 512), (1024, 1024), (1024, 1536)].map { try need($0.0, $0.1).phases.map(\.kind) }
        XCTAssertEqual(Set(kinds.map { $0.map(\.rawValue).joined(separator: ",") }).count, 1)
        XCTAssertEqual(kinds[0], [.text, .denoising, .decoding])
        // The tail (half of 30 layers, two buffers of a bf16 layer) is in the denoising at every format.
        let bf16Layer = Artifact.stagingBytes(header: Self.zImageMaps().dit, prefixes: [DiT.layerPrefix(29)], fileSize: .max)
        XCTAssertGreaterThan(bf16Layer, 360_000_000)
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            let all = try need(w, h)
            var plan = Self.zRequest(width: w, height: h, promptTokens: 50)
            plan.tail = .none
            let noTail = try XCTUnwrap(MemoryNeed.zImage(plan, maps: Self.zImageMaps()))
            // Three buffers of the layer's largest tensor (w1: 3 840 × 10 240 in bf16), not two of a layer.
            XCTAssertEqual(own(all, .denoising) - own(noTail, .denoising), TailStream.unitSlots * 78_643_200)
            // The lean plan: the whole map through the same two buffers, no encoder buffer, small bands.
            let lean = try XCTUnwrap(MemoryNeed.zImage(plan.lean, maps: Self.zImageMaps()))
            XCTAssertLessThanOrEqual(lean.peak, all.peak)
            XCTAssertEqual(phase(lean, .text)!.bytes("encoder staging"), 0)
        }
    }

    // ── the formula is the sum of the shape functions ────────────────────────────────────────

    /// **The text phase, term by term**: the encoder's arena and its staging buffers, the causal
    /// mask in our pages, MPSGraph's copy of the queries, the GEMM's state and workspace.
    func testTheTextIsItsShapeFunctions() throws {
        let maps = Self.zImageMaps()
        let encoder = try TextEncoder.Config(header: maps.encoder)
        for tokens in [1, 50, 400] {
            let text = try XCTUnwrap(phase(try need(1024, 1024, tokens: tokens), .text))
            let layer = Artifact.largestRun(header: maps.encoder, prefixes: TextEncoder.layerPrefixes(encoder), fileSize: .max)
            let (h, a, kv, i) = (encoder.hidden, encoder.attentionWidth, encoder.kvWidth, encoder.intermediate)
            XCTAssertEqual(text.bytes, TextEncoder.arenaBytes(config: encoder, sequence: tokens)
                            + TextEncoder.stagingSlots(granted: MemoryPlan.encoderSlots) * layer
                            + Attention.maskBytes(sequence: tokens, keys: tokens)
                            + tokens * 32 * 128 * 4
                            + sum(MemoryNeed.gemm("", bounded: [(tokens, h, a), (tokens, h, kv), (tokens, a, h),
                                                                (tokens, h, i), (tokens, i, h)], plan: .defaults,
                                                  phase: .encoding)), "\(tokens) tokens")
            XCTAssertEqual(text.bytes("encoder causal mask"), Arena.capacity(for: tokens * tokens * 4))
        }
    }

    /// **The denoising, term by term**: the DiT's four arenas, the map's buffers, the sampler's latents,
    /// the full-size attentions (head splits' host arenas and the GPU heads' queries, the context
    /// refiners' MPSGraph SDPA), the GEMM — nothing else.
    func testTheDenoisingIsItsShapeFunctions() throws {
        let maps = Self.zImageMaps()
        let dit = DiT.Config(header: maps.dit)
        let released = MemoryPlan.Tail(fraction: 0.5, streamed: true).released(layers: dit.layers)
        let staging = TailStream.unitSlots * Artifact.largestRun(header: maps.dit, prefixes: (released..<dit.layers).map(DiT.layerPrefix),
                                                                 fileSize: maps.ditSize)
        let cpuHeads = HeadSplitAttention.partition(heads: dit.heads, fraction: HeadSplitAttention.referenceFraction).cpu.count
        let gpuHeads = dit.heads - cpuHeads
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            let n = try need(w, h)
            let latent = 16 * (h / 8) * (w / 8) * 4
            let image = (h / 16) * (w / 16), cap = 64, s = image + cap
            func split(_ rows: Int) -> Int {
                HeadSplitAttention.hostBytes(heads: 30, sequence: rows, headDim: 128, cpuHeads: cpuHeads, threads: 6) + rows * gpuHeads * 128 * 4
            }
            let attentions = [image, s].map(split).reduce(0, +) + cap * 30 * 128 * 4
            // The caption's rows and the sequence's follow the tokens' bound (the tiers charged for them).
            let gemm = sum(MemoryNeed.gemm("", products: [(image, 64, 3840)],
                                           bounded: [(s, 3840, 3840), (s, 3840, 10240), (s, 10240, 3840),
                                                     (cap, 2560, 3840), (s, 3840, 64)], plan: .defaults))
            XCTAssertEqual(GEMM.tierBytes(products: [], bounded: [(s, 3840, 10240), (s, 10240, 3840)]),
                           2 * Arena.capacity(for: 64 * 10240 * 4))
            XCTAssertEqual(phase(n, .denoising)!.bytes,
                           DiT.arenaBytes(config: dit, latentHeight: h / 8, latentWidth: w / 8, capLength: 50, loraRank: 0)
                            + staging + latent + attentions + gemm, "\(w)×\(h)")
        }
        // A LoRA: its cache over every module a block can apply it to, the `mid` slices in the arenas;
        // two LoRAs, one module's `down` staging besides.
        let one = try need(1024, 1024, ranks: [64]), two = try need(1024, 1024, ranks: [64, 32])
        XCTAssertEqual(phase(one, .denoising)!.bytes("LoRA cache"),
                       CacheLoRA.arenaBytes(CacheLoRA.bytes(modules: DiT.loraModules(config: dit).map { ($0.k, $0.n, 64) })))
        XCTAssertEqual(phase(one, .denoising)!.bytes("LoRA down"), 0)
        XCTAssertEqual(phase(two, .denoising)!.bytes("LoRA down"), LoRA.downBytes(k: 10240, rank: 64))
    }

    /// **One attention per block, at the full grid**: two head splits (image refiners, layers) and the
    /// context refiners' MPSGraph SDPA, with or without a LoRA; the DiT's attention is not causal.
    func testOnlyTheFullSizeAttentionsAreAlive() throws {
        for ranks in [[], [16]] as [[Int]] {
            let terms = phase(try need(1024, 1024, ranks: ranks), .denoising)!.terms
            XCTAssertEqual(terms.filter { $0.name.hasSuffix("head split") }.map(\.name), ["image refiners head split", "layers head split"])
            XCTAssertEqual(terms.filter { $0.name.hasSuffix("SDPA queries") }.count, 3)
            XCTAssertFalse(terms.contains { $0.name.contains("causal mask") })
        }
    }

    /// **The VAEs are their plans**: the decoder's two heaps and the image it returns, the
    /// encoder's at the render's format and the `fitted` copy when the image handed differs.
    func testTheVAEsAreTheirPlans() throws {
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            let n = try need(w, h)
            let plan = VAE.plan(latentHeight: h / 8, latentWidth: w / 8)
            XCTAssertEqual(phase(n, .decoding)!.bytes, plan.bytes + plan.outside)
            let start = try need(w, h, references: [(w, h)])
            let encoder = VAEEncoder.plan(family: .flux, height: h, width: w)
            XCTAssertEqual(phase(start, .references)!.bytes, encoder.bytes + encoder.outside)
            // Nothing of the encoder is carried into the denoising.
            XCTAssertEqual(phase(start, .denoising)!.bytes, phase(n, .denoising)!.bytes)
        }
        let other = try need(1024, 1024, references: [(2048, 1024)])
        XCTAssertEqual(phase(other, .references)!.bytes("fitted copy"), 3 * 1024 * 1024 * 4)
        // The lean plan's decoder takes the smallest band.
        var lean = Self.zRequest(width: 1024, height: 1024, promptTokens: 50).lean
        lean.tail = .lean
        let leanNeed = try XCTUnwrap(MemoryNeed.zImage(lean, maps: Self.zImageMaps()))
        let leanPlan = VAE.plan(latentHeight: 128, latentWidth: 128, band: min(VAE.bandElements, MemoryPlan.fluxMinimumBand))
        XCTAssertEqual(phase(leanNeed, .decoding)!.bytes, leanPlan.bytes + leanPlan.outside)
    }

    /// **A batch**: the other images' latents cross every phase, the images decoded before the last
    /// wait in the decoding; the DiT, kept from one image to the next (`Context.keptDenoiser`), is one.
    func testABatchHoldsItsOtherImages() throws {
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            var request = Self.zRequest(width: w, height: h, promptTokens: 50)
            let one = try XCTUnwrap(MemoryNeed.zImage(request, maps: Self.zImageMaps()))
            request.batch = 4
            let four = try XCTUnwrap(MemoryNeed.zImage(request, maps: Self.zImageMaps()))
            XCTAssertEqual(four.crossing - one.crossing, 3 * 16 * (h / 8) * (w / 8) * 4)
            XCTAssertEqual(bytes(four, .decoding) - bytes(one, .decoding), 3 * 3 * w * h * 4)
            XCTAssertEqual(bytes(four, .denoising), bytes(one, .denoising))
        }
    }

    /// **The engine and the formula agree on a LoRA stack**: the request takes each LoRA's own rank
    /// (`LoRA.layerRanks`, what `Chain.execute` passes), so two LoRAs stage a `down` (`LoRA.downBytes`)
    /// in the formula as they do in `LoRA.materialize` — the engine used to pass `[maxRank]` alone, and
    /// the staging was never charged.
    func testTheLoRAStackIsTheEngines() throws {
        let artifact = try Artifact(path: try map(tinyDiT))
        let config = DiT.Config(header: artifact.header)
        let (d, hidden) = (config.dim, config.hidden)
        func lora(_ rank: Int, _ targets: [(String, Int, Int)]) throws -> LoRA.Layer {
            let tensors = targets.flatMap { [("\($0.0).down", [$0.1, rank]), ("\($0.0).up", [rank, $0.2])] }
            let path = try map(["kind": "lora"], tensors: tensors.map { (name: $0.0, shape: $0.1) })
            return LoRA.Layer(artifact: try Artifact(path: path), strength: 1, name: path)
        }
        let one = try LoRA(layers: [lora(4, [("layers.0.attention.to_q.weight", d, d)])])
        let two = try LoRA(layers: [lora(4, [("layers.0.attention.to_q.weight", d, d), ("layers.0.feed_forward.w2.weight", hidden, d)]),
                                    lora(3, [("layers.0.attention.to_q.weight", d, d)])])
        XCTAssertEqual(one.layerRanks, [4])
        XCTAssertEqual(two.layerRanks, [4, 3])
        XCTAssertLessThanOrEqual(two.maxRank, two.layerRanks.reduce(0, +))   // the arenas' rank, bounded
        let single = try need(1024, 1024, ranks: one.layerRanks), stacked = try need(1024, 1024, ranks: two.layerRanks)
        XCTAssertEqual(phase(single, .denoising)!.bytes("LoRA down"), 0)
        XCTAssertEqual(phase(stacked, .denoising)!.bytes("LoRA down"), LoRA.downBytes(k: 10240, rank: 4))
    }

    /// img2img: an encoding phase at the render's format, and the image handed plus its z₀ crossing.
    func testImageToImageAddsItsPhaseAndWhatCrosses() throws {
        let plain = try need(1024, 1024), start = try need(1024, 1024, references: [(1024, 1024)])
        XCTAssertEqual(start.phases.filter { $0.kind == .references }.count, 1)
        XCTAssertEqual(start.crossing - plain.crossing, (3 * 1024 * 1024 + 16 * 128 * 128) * 4)
        let other = try need(1024, 1024, references: [(2048, 1024)])
        XCTAssertGreaterThan(own(other, .references), own(start, .references))   // the `fitted` copy
    }

    /// **No fitted term remains**: what is outside our hand is only the shapes' copies — the queries'
    /// at exactly `rows·heads·headDim·4`, the VAEs' `outside` —, no constant of the frameworks, and
    /// every other byte is ours. The base is the request's (0 for a need of shapes alone).
    func testNoFittedTermRemains() throws {
        for request in [(512, 512, [Int]()), (1024, 1536, [64])] {
            let n = try need(request.0, request.1, ranks: request.2, references: [(request.0, request.1)])
            for term in n.phases.flatMap(\.terms) where term.outside {
                XCTAssertTrue(term.name.hasSuffix("SDPA queries") || term.name.hasSuffix(" outside"), term.name)
                XCTAssertEqual(term.bytes % 4, 0)
            }
            XCTAssertFalse(n.phases.flatMap(\.terms).contains { $0.name.contains("executable") || $0.name.contains("MPS GEMM") })
            XCTAssertEqual(n.base, 0)
        }
        XCTAssertEqual(OperatorFootprint.sdpaQueries(rows: 4160, heads: 30, headDim: 128), 4160 * 30 * 128 * 4)
    }

    /// **The Neural Engine's surfaces go to the phases whose products it takes**: the DiT's, when the render
    /// lets it (`Request.neuralEngine`) — never the encoder's, whose `GEMM` never has it, however long the prompt.
    func testNeuralEngineIsChargedToTheDenoiserItServes() throws {
        for on in [false, true] {
            var request = Self.zRequest(width: 1024, height: 1024, promptTokens: 1100)
            // A plan that gives the ANE rows of the DiT's products (a check's strictest case, on any Mac).
            request.plan = .forcing(neuralEngine: true, bf16Activations: false)
            request.neuralEngine = on
            let n = try XCTUnwrap(MemoryNeed.zImage(request, maps: Self.zImageMaps()))
            func charged(_ kind: MemoryNeed.Phase.Kind) -> Bool {
                n.phases.filter { $0.kind == kind }.flatMap(\.terms).contains { $0.name.hasSuffix("Neural Engine") }
            }
            XCTAssertFalse(charged(.text))
            XCTAssertEqual(charged(.denoising), on)
        }
    }

    /// The map's type is in its header: an 8-bit map's tail buffers are about half a bf16 one's.
    func testAnEightBitMapNeedsSmallerBuffers() throws {
        let bf16 = try need(1024, 1024), int8 = try need(1024, 1024, maps: Self.zImageMaps(bytesPerValue: 1))
        let saved = own(bf16, .denoising) - own(int8, .denoising)
        XCTAssertGreaterThan(saved, TailStream.unitSlots * 38_000_000)   // 3 buffers of the largest tensor (w1): half the bytes
        XCTAssertLessThan(saved, TailStream.unitSlots * 40_000_000)
        XCTAssertLessThan(own(int8, .text), own(bf16, .text))
    }

    // ── the need costs milliseconds ──────────────────────────────────────────────────────────

    /// **What the memory keeps is what the walk returns**: every VAE plan the need reads, at two formats
    /// (square, portrait) and both bands, equals its walk done again; Z-Image's stream buffers, read once per map,
    /// equal `Artifact.largestRun` over every tail. Then a need, once its shapes are known, costs
    /// under 5 ms — the preflight's and the app's (it greys out each format it refuses).
    func testTheShapeFunctionsAreKeptAndTheNeedIsCheap() throws {
        let qwen = QwenImage21VAE.Config()
        for (w, h) in [(512, 512), (832, 1216)] {
            for band in [VAE.bandElements, MemoryPlan.fluxMinimumBand] {
                XCTAssertEqual(VAE.plan(latentHeight: h / 8, latentWidth: w / 8, band: band),
                               VAE.walk(latentHeight: h / 8, latentWidth: w / 8, band: band), "Flux decoder \(w)×\(h)")
            }
            for pixels in [QwenImage21VAEDecoder.bandPixels, MemoryPlan.qwenMinimumBand] {
                XCTAssertEqual(QwenImage21VAEDecoder.plan(config: qwen, latentHeight: h / 16, latentWidth: w / 16, bandPixels: pixels),
                               QwenImage21VAEDecoder.walk(config: qwen, latentHeight: h / 16, latentWidth: w / 16, bandPixels: pixels),
                               "Qwen decoder \(w)×\(h)")
            }
            XCTAssertEqual(VAEEncoder.plan(family: .flux, height: h, width: w), VAEEncoder.walk(family: .flux, height: h, width: w))
            XCTAssertEqual(QwenImage21VAEEncoder.plan(config: qwen, height: h, width: w),
                           QwenImage21VAEEncoder.walk(config: qwen, height: h, width: w))
        }
        let maps = Self.zImageMaps(), dit = DiT.Config(header: maps.dit)
        for released in 0...dit.layers {
            XCTAssertEqual(maps.ditUnitStaging(released..<dit.layers),
                           Artifact.largestRun(header: maps.dit, prefixes: (released..<dit.layers).map(DiT.layerPrefix),
                                               fileSize: maps.ditSize), "tail from \(released)")
        }
        let encoder = try TextEncoder.Config(header: maps.encoder)
        XCTAssertEqual(maps.encoderLayerStaging,
                       Artifact.largestRun(header: maps.encoder, prefixes: TextEncoder.layerPrefixes(encoder),
                                             fileSize: maps.encoderSize))

        // Hot: the product's worst case (1024×1536, three references), both families, both plans.
        let references = [(width: 1024, height: 1024), (width: 1216, height: 832), (width: 1024, height: 1536)]
        let qwenRequest = MemoryFixtures.qwenRequest(1024, 1536, references: references, tokens: 200, ranks: [32])
        let zRequest = Self.zRequest(width: 1024, height: 1536, references: [(1024, 1536)],
                                          promptTokens: 200, loraRanks: [32])
        func needs() -> [MemoryNeed] {
            [MemoryNeed.of(qwenRequest)!, MemoryNeed.of(qwenRequest.lean)!,
             MemoryNeed.zImage(zRequest, maps: maps)!, MemoryNeed.zImage(zRequest.lean, maps: maps)!]
        }
        let cold = needs()
        let started = Date()
        let hot = needs()
        let each = Date().timeIntervalSince(started) / Double(hot.count)
        XCTAssertEqual(cold, hot)
        XCTAssertLessThan(each, 0.005, String(format: "a need: %.2f ms", each * 1000))
    }

    /// The decoding carries the denoiser's Neural Engine surfaces (released late by Core ML), exactly the denoising's term; with no
    /// Neural Engine term in the denoising, the need is unchanged.
    func testDecodingCarriesTheDenoisersSurfaces() {
        let surfaces = NeuralEngine.bytes(products: [(512, 3840, 3840)])
        XCTAssertGreaterThan(surfaces, 0)
        let decoder = MemoryNeed.Term("decoder", 1000)
        func need(_ terms: [MemoryNeed.Term]) -> MemoryNeed {
            MemoryNeed(phases: [.init(kind: .denoising, terms: terms), .init(kind: .decoding, terms: [decoder])], crossing: 0, base: 0)
        }
        let with = need([.init(MemoryNeed.ditSurfacesTerm, surfaces), .init("other", 5)]).carryingSurfaces()
        XCTAssertEqual(with.need(of: .decoding), 1000 + surfaces)
        XCTAssertEqual(with.need(of: .denoising), surfaces + 5)
        let without = need([.init("other", 5)])
        XCTAssertEqual(without.carryingSurfaces(), without)
    }
}
