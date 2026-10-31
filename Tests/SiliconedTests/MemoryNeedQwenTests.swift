import XCTest
@testable import Siliconed

/// **Qwen-Image-2.1's memory need, without a byte of weights**: the formula
/// charges what the constructors allocate — the same static functions —, and it grows with every
/// input that makes a render larger. On the maps of `MemoryFixtures` and 6 P cores, unless a test says
/// otherwise: no figure depends on the Mac that runs them.
final class MemoryNeedQwenTests: XCTestCase {
    typealias Grid = QwenImage21Sequence.Grid

    /// The library prompt's template: 27 tokens, 6 more per reference (its tags).
    static func request(_ width: Int, _ height: Int, references: [(width: Int, height: Int)] = [],
                        tokens: Int? = nil, ranks: [Int] = [], threads: Int = 6) -> MemoryNeed.Request {
        MemoryFixtures.qwenRequest(width, height, references: references, tokens: tokens ?? 27 + 6 * references.count,
                                   ranks: ranks, threads: threads)
    }

    static func need(_ request: MemoryNeed.Request) -> MemoryNeed { MemoryNeed.of(request)! }

    static func shapes(_ request: MemoryNeed.Request) -> QwenImage21Need { QwenImage21Need(request, maps: request.qwenMaps!) }

    static func phase(_ need: MemoryNeed, _ kind: MemoryNeed.Phase.Kind) -> MemoryNeed.Phase? {
        need.phases.first { $0.kind == kind }
    }

    // ── the formula is the constructors' ─────────────────────────────────────────────────────

    /// The prefill's groups on the oracle's edit-2 layout: one group for the whole prefix (2 072 rows
    /// under the budget of max(1 024, 1 024) + 60 text rows? no — 2 072 > 1 084: two groups), and the
    /// scratch is the largest group.
    func testThePrefillGroupsAreTheDiTs() throws {
        let slots = [Bool](repeating: false, count: 8) + [Bool](repeating: true, count: 247)
            + [Bool](repeating: false, count: 6) + [Bool](repeating: true, count: 256)
            + [Bool](repeating: false, count: 46) + [Bool](repeating: true, count: 256)
        let s = try QwenImage21Sequence(slots: slots, images: [Grid(height: 26, width: 38), Grid(height: 32, width: 32),
                                                               Grid(height: 32, width: 32)])
        let layout = QwenImage21DiT.layout(s, fullPass: false)
        // Budget: max(S 1 024, largest image 1 024) + 60 text rows = 1 084.
        XCTAssertEqual(layout.groups.map { [$0.start, $0.end] }, [[0, 1002], [1002, 2072]])
        XCTAssertEqual(layout.scratch, 1070)
        XCTAssertEqual(QwenImage21DiT.layout(s, fullPass: true).scratch, 3096)
    }

    /// The arena of a 1024² generation, written out float by float — what `QwenImage21DiT.init` asks.
    func testTheDiTArenaIsItsSlots() {
        let need = Self.shapes(Self.request(1024, 1024))
        let seq = need.sequence
        XCTAssertEqual(seq.target, 4096); XCTAssertEqual(seq.prefix, 13)   // 27 − 14 text rows
        let (d, h, all, n) = (4096, 12_288, seq.count, 4096)
        let mlpRows = QwenImage21DiT.mlpRows(scratch: n, hidden: h)
        XCTAssertEqual(mlpRows, 4096)   // 403 MB: one chunk
        // Every slice a whole number of pages (a row of `d` is one page) but the RoPE table's: no margin.
        let floats = all * d + 2 * n * d + 2 * all * d + mlpRows * h + mlpRows * h + d * h
            + 4096 * 64 + 2 * 5 * d
        XCTAssertEqual(QwenImage21DiT.arenaBytes(config: need.dit, rows: all, prefix: seq.prefix, scratch: n,
                                                 largestImage: 4096, rank: 0, fullPass: false),
                       floats * 4 + Arena.capacity(for: all * 128 * 4))
        XCTAssertEqual(QwenImage21DiT.cacheArenaBytes(config: need.dit, prefix: 13),
                       QwenImage21CachePolicy.bytes(prefix: 13))
    }

    /// The denoising phase is the sum of the statics the DiT, its cache and its LoRA call, plus the
    /// staging blocks of a generation, plus the sampler's copies.
    func testTheDenoisingIsTheConstructorsSum() {
        let request = Self.request(1024, 1024, ranks: [32])
        let need = Self.shapes(request)
        let seq = need.sequence
        XCTAssertEqual(need.storage, .memory)
        let scratch = QwenImage21DiT.layout(seq, fullPass: false).scratch
        let arena = QwenImage21DiT.arenaBytes(config: need.dit, rows: seq.count, prefix: seq.prefix, scratch: scratch,
                                              largestImage: seq.target, rank: 32, fullPass: false)
        let cache = QwenImage21DiT.cacheArenaBytes(config: need.dit, prefix: seq.prefix)
        // A block is a layer's run in the map, read from its header: the 7 weights and the two QK-norms,
        // each on its own 16 KB pages — 436 240 384 bytes in bf16, as the installed map lays it.
        let block = 4 * 4096 * 4096 * 2 + 3 * 4096 * 12_288 * 2 + 2 * 16384
        XCTAssertEqual(block, 436_240_384)
        XCTAssertEqual(need.maps.ditLayerStaging, block)
        // The stream's buffers hold the largest tensor (a 4 096 × 12 288 weight, 100 663 296 bytes in bf16), three of them.
        XCTAssertEqual(need.maps.ditUnitStaging, 4096 * 12_288 * 2)
        let staging = TailStream.unitSlots * need.maps.ditUnitStaging
        let module = { (k: Int, n: Int) in (k, n, 32) }
        let lora = 2 * StreamedLoRA.layout([
            "attn.to_q": module(4096, 4096), "attn.to_k": module(4096, 4096), "attn.to_v": module(4096, 4096),
            "attn.to_out.0": module(4096, 4096), "img_mlp.gate_layer": module(4096, 12_288),
            "img_mlp.proj": module(4096, 12_288), "img_mlp.out": module(12_288, 4096)]).bytes
        let sampler = 64 * 4096 * 4 + 8 * 5 * 4096 * 4
        let phase = need.denoisingPhase
        XCTAssertEqual(phase.bytes("DiT arena") + phase.bytes("K/V arena") + phase.bytes("map staging")
                        + phase.bytes("LoRA slots") + phase.bytes("sampler"), arena + cache + staging + lora + sampler)
        // The rest: the attentions the DiT keeps, its GEMM, and the modulation's widening when it is
        // the larger moment — nothing unnamed.
        let rest = phase.terms.filter { !["DiT arena", "K/V arena", "map staging", "LoRA slots", "sampler"].contains($0.name) }
        for term in rest {
            XCTAssertTrue(term.name.hasPrefix("DiT ") || term.name.hasPrefix("modulation widening"), term.name)
        }
        // The widened `modulation.1` (`d × 4d`) is alive before the attentions, never beside them.
        let running = rest.filter { !$0.name.hasPrefix("modulation") }.reduce(0) { $0 + $1.bytes }
        XCTAssertEqual(running + phase.bytes("modulation widening"),
                       max(running, Krea2Exact.widenedBytes(inputs: 4096, outputs: 4 * 4096)
                            + Arena.capacity(for: 4096 * 32 * 4) + Arena.capacity(for: 32 * 4 * 4096 * 4)))
    }

    /// `StreamedLoRA`'s slot at rank 256 on the 7 Linears of a block: the "2 × 84 MB" of its header.
    func testTheLoRASlotIsItsHeaders() {
        let r = 256
        let slot = StreamedLoRA.layout([
            "attn.to_q": (4096, 4096, r), "attn.to_k": (4096, 4096, r), "attn.to_v": (4096, 4096, r),
            "attn.to_out.0": (4096, 4096, r), "img_mlp.gate_layer": (4096, 12_288, r),
            "img_mlp.proj": (4096, 12_288, r), "img_mlp.out": (12_288, 4096, r)])
        XCTAssertEqual(slot.bytes, 81_920 * r * 4)   // every piece already a whole page
        XCTAssertEqual(Double(slot.bytes) / 1e6, 84, accuracy: 0.5)
        XCTAssertEqual(slot.layout["attn.to_k"]!.down, 0)   // sorted by suffix
    }

    /// The decoder's tail at 1024²: a 256-wide input, bands of 64 kept rows + 2 × 6 of margin; its
    /// input is `x` 576 and `skip` 1152 channels at ×4 — what `decode` checks it meets.
    func testTheDecoderBandIsTheTails() {
        let config = QwenImage21VAE.Config()
        let input = QwenImage21VAEDecoder.tailInput(config: config)
        XCTAssertEqual([input.x, input.skip, input.scale], [576, 1152, 4])
        let band = QwenImage21VAEDecoder.band(config: config, tailHeight: 256, tailWidth: 256, bandPixels: 1 << 18)
        XCTAssertEqual([band.margin, band.core, band.window], [6, 64, 76])
        // A tail shorter than a band reads it whole.
        XCTAssertEqual(QwenImage21VAEDecoder.band(config: config, tailHeight: 40, tailWidth: 40, bandPixels: 1 << 18).window, 40)
    }

    /// The vision tower's arena is its constructor's formula; one reference of 1024² is 4 096 patches.
    func testTheVisionArenaIsTheTowers() {
        let config = Qwen3VLVision.Config.published
        XCTAssertEqual([config.hidden, config.intermediate, config.patchValues, config.merged], [1152, 4304, 1536, 4608])
        let n = 4096, d = 1152
        let largest = 4608 * 4608, widest = 4608
        XCTAssertEqual(Qwen3VLVision.arenaBytes(config: config, patches: n),
                       (n * 1536 + 10 * n * d + n * 4304 + n / 4 * 4096 + n * 72 + largest) * 4
                        + 3 * Arena.capacity(for: widest * 4))   // whole pages but the bias rows: no margin
        let need = Self.shapes(Self.request(1024, 1024, references: [(1024, 1024)]))
        XCTAssertEqual(need.references[0].patches, 4096)
        XCTAssertEqual(need.encoderTokens, 33 + 1024)
        XCTAssertEqual(need.sequence.prefix, 19 + 4096)
    }

    /// **The need never falls as the prompt grows, across the K/V's move to the file.** The tokens are a
    /// bound (the prompt's bytes): past 256 text rows the bound says « file », the real prefix may still
    /// be in memory — where the K/V arena and the staging blocks cost more than the file's longer arena.
    /// Both placements are charged while either is possible: the need at the bound is never below the
    /// need at a shorter one (it dropped by ~1 GB here before, 12/10/2026).
    func testTheNeedNeverFallsAcrossTheFileThreshold() {
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            for lean in [false, true] {
                // Every 23 tokens, every token around the threshold (P = 256 at 270), and around each
                // height that becomes a multiple of 64 (a GEMM's tiers, `GEMM.tierBytes`).
                let edges = [64, 128, 192, 256].flatMap { rows in (rows + 13)...(rows + 15) } + [31, 32, 33, 34]
                let tokens = Set(Array(stride(from: 27, through: 300, by: 23)) + Array(264...278) + edges + [400, 526]).sorted()
                var previous = (tokens: 0, denoising: 0, peak: 0)
                for t in tokens {
                    var request = Self.request(w, h, tokens: t)
                    if lean { request = request.lean }
                    let need = Self.need(request)
                    let denoising = Self.phase(need, .denoising)!.bytes
                    XCTAssertGreaterThanOrEqual(denoising, previous.denoising, "\(w)×\(h) \(previous.tokens) → \(t) tokens")
                    XCTAssertGreaterThanOrEqual(need.peak, previous.peak, "\(w)×\(h) \(previous.tokens) → \(t) tokens")
                    previous = (t, denoising, need.peak)
                }
            }
        }
        // Past the threshold, the bound says « file » and the memory case at 256 rows is charged too.
        let long = Self.shapes(Self.request(1024, 1024, tokens: 271))
        XCTAssertEqual(long.storage, .file)
        XCTAssertEqual(long.placements.map(\.storage), [.file, .memory])
        XCTAssertEqual(long.placements.last?.textRows, QwenImage21CachePolicy.largestMemoryPrefix)
        let atThreshold = Self.shapes(Self.request(1024, 1024, tokens: 270))
        XCTAssertEqual(atThreshold.storage, .memory)   // P = 256
        XCTAssertEqual(atThreshold.placements.count, 1)
        XCTAssertGreaterThanOrEqual(long.denoisingPhase.bytes, atThreshold.denoisingPhase.bytes)
        // An edit's prefix is always past it: one placement, the file.
        XCTAssertEqual(Self.shapes(Self.request(1024, 1024, references: [(1024, 1024)])).placements.map(\.storage), [.file])
    }

    /// **The formula places the K/V by the engine's own rule, on the engine's own prefix.** For the
    /// exact tokens of a prompt (what the encoder returns), the formula's prefix is the one the module
    /// and the sampler compute from the conditioning — its rows, its image slots — and the placement
    /// is `QwenImage21CachePolicy.storage` on it; the formula charges that placement among its own.
    func testThePlacementIsTheEnginesForTheSamePrompt() throws {
        for references in [[], [(width: 1024, height: 1024)], [(width: 1216, height: 832), (width: 640, height: 640)]]
            as [[(width: Int, height: Int)]] {
            for tokens in [27, 200, 269, 270, 271, 300, 526] {
                let need = Self.shapes(Self.request(1024, 1024, references: references,
                                                    tokens: tokens + 6 * references.count))
                // The conditioning the encoder returns: its rows past the system turn, image slots included.
                let slots = need.references.reduce(0) { $0 + $1.slots }
                let conditioningRows = need.textRows + slots
                let enginePrefix = QwenImage21CachePolicy.prefix(conditioningRows: conditioningRows, imageSlots: slots)
                XCTAssertEqual(need.sequence.prefix, enginePrefix, "\(references.count) refs, \(tokens) tokens")
                // The sampler's own sequence, from the same slots (`QwenImage21Sampler.run`).
                var textSlots = [Bool](repeating: false, count: 3)
                for (i, r) in need.references.enumerated() {
                    textSlots += [Bool](repeating: false, count: QwenImage21Need.tokensBefore)
                    textSlots += [Bool](repeating: true, count: r.slots)
                    if i < need.references.count - 1 { textSlots += [Bool](repeating: false, count: QwenImage21Need.tokensAfter) }
                }
                textSlots += [Bool](repeating: false, count: conditioningRows - textSlots.count)
                let sampler = try QwenImage21Sequence(
                    slots: textSlots + [Bool](repeating: true, count: 64 * 64 / 4),
                    images: need.references.map { .init(height: $0.latent.height, width: $0.latent.width) }
                        + [.init(height: 64, width: 64)])
                XCTAssertEqual(sampler.prefix, enginePrefix)
                let engine = QwenImage21CachePolicy.storage(prefix: sampler.prefix)
                XCTAssertEqual(need.storage, engine, "\(references.count) refs, \(tokens) tokens")
                XCTAssertTrue(need.placements.contains { $0.storage == engine })
            }
        }
    }

    // ── what makes a render larger makes its need larger ─────────────────────────────────────

    func testTheNeedGrowsWithTheArea() {
        let peaks = [(512, 512), (832, 1216), (1024, 1024), (1024, 1536)].map { Self.need(Self.request($0.0, $0.1)).peak }
        XCTAssertEqual(peaks, peaks.sorted()); XCTAssertEqual(Set(peaks).count, peaks.count)
        let edits = [(512, 512), (1024, 1024), (1024, 1536)].map {
            Self.need(Self.request($0.0, $0.1, references: [(1024, 1024)])).peak
        }
        XCTAssertEqual(edits, edits.sorted()); XCTAssertEqual(Set(edits).count, edits.count)
    }

    func testTheNeedGrowsWithTheReferences() {
        let refs: [(width: Int, height: Int)] = [(1024, 1024), (1024, 1024), (1024, 1024)]
        let peaks = (0...3).map { Self.need(Self.request(1024, 1024, references: Array(refs.prefix($0)))).peak }
        // From one reference on: an edit streams nothing, and its encoder's buffers are the largest tensor's now,
        // not a layer's — a generation (no reference) holds the map's staging units and can lie above a first edit.
        XCTAssertEqual(Array(peaks.dropFirst()), peaks.dropFirst().sorted()); XCTAssertEqual(Set(peaks).count, 4)
        // Each phase too, the decoding aside (it never sees the references).
        for kind in [MemoryNeed.Phase.Kind.text, .denoising] {
            let own = (0...3).map { Self.phase(Self.need(Self.request(1024, 1024, references: Array(refs.prefix($0)))), kind)!.own }
            // From one reference on: a generation's denoising holds staging blocks an edit does not.
            XCTAssertEqual(Array(own.dropFirst()), own.dropFirst().sorted(), "\(kind)")
        }
    }

    /// A reference's size: every original is prepared at ~R² (`conditionSize`), so what grows with it
    /// is the original held through the render — never less for a larger one.
    func testTheNeedGrowsWithAReferencesSize() {
        let sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
        let peaks = sizes.map { Self.need(Self.request(1024, 1024, references: [$0])).peak }
        XCTAssertEqual(peaks, peaks.sorted())
        XCTAssertLessThan(peaks.first!, peaks.last!)
    }

    func testTheNeedGrowsWithThePrompt() {
        for references in [[], [(width: 1024, height: 1024)]] as [[(width: Int, height: Int)]] {
            // Across the generation's K/V move to the file past 256 text rows too (above).
            let tokens = references.isEmpty ? [27, 100, 200, 270, 300, 526] : [33, 100, 400, 540]
            let peaks = tokens.map { Self.need(Self.request(1024, 1024, references: references, tokens: $0)).peak }
            XCTAssertEqual(peaks, peaks.sorted())
        }
    }

    func testTheNeedGrowsWithTheRank() {
        // The peak never falls with the rank; it may not rise either: the decoder (3.3 GB at 1024²,
        // measured) holds a generation's peak, and a LoRA lives in the denoising.
        let ranks: [[Int]] = [[], [16], [64], [128, 128]]
        let peaks = ranks.map { Self.need(Self.request(1024, 1024, ranks: $0)).peak }
        XCTAssertEqual(peaks, peaks.sorted())
        let generation = ranks.map { Self.phase(Self.need(Self.request(1024, 1024, ranks: $0)), .denoising)!.bytes }
        XCTAssertEqual(generation, generation.sorted()); XCTAssertEqual(Set(generation).count, generation.count)
        let denoising = [[], [64], [256]].map {
            Self.phase(Self.need(Self.request(1024, 1024, references: [(1024, 1024)], ranks: $0)), .denoising)!.own
        }
        XCTAssertEqual(denoising, denoising.sorted()); XCTAssertEqual(Set(denoising).count, 3)
    }

    /// Three references of mixed sizes: each prepared at its own ratio, the tower sized for the
    /// largest, the prefill in groups, every phase present.
    func testThreeReferencesOfMixedSizes() {
        let request = Self.request(1024, 1536, references: [(3000, 2000), (800, 1200), (640, 640)])
        let need = Self.shapes(request)
        XCTAssertEqual(need.references.map { [$0.prepared.width, $0.prepared.height] }, [[1248, 832], [832, 1248], [1024, 1024]])
        XCTAssertEqual(need.sequence.images.count, 4)
        XCTAssertEqual(need.sequence.prefix, need.textRows + 78 * 52 * 2 + 64 * 64)
        XCTAssertEqual(need.storage, .file)
        let full = Self.need(request)
        XCTAssertEqual(Set(full.phases.map(\.kind)), Set(MemoryNeed.Phase.Kind.allCases))
        XCTAssertEqual(full.peak, full.phases.map(\.bytes).max()! + full.crossing + full.base)
        // The vision tower is sized for the largest image (4 096 patches each here: 78·52 = 4 056).
        XCTAssertEqual(Self.phase(full, .vision)!.bytes("vision arena"),
                       Qwen3VLVision.arenaBytes(config: need.vision, patches: 4096))
        // Its 72-wide heads run padded to 128 (`Attention.workingHeadDim`, issue #11): the queries' copy and the
        // padded q, k, v and output at that width.
        XCTAssertEqual(Self.phase(full, .vision)!.bytes("vision SDPA queries"), 4096 * 16 * 128 * 4)
        XCTAssertEqual(Self.phase(full, .vision)!.bytes("vision head padding"), 4 * 4096 * 16 * 128 * 4)
        // More than the same with three 1024² references? The prefix is shorter (4 056 + 4 056 + 4 096).
        let squares = Self.need(Self.request(1024, 1536, references: [(1024, 1024), (1024, 1024), (1024, 1024)]))
        XCTAssertLessThan(Self.phase(full, .denoising)!.own, Self.phase(squares, .denoising)!.own)
        // The denoising: the target's SDPA cut by heads, one GEMM.
        let terms = Self.phase(full, .denoising)!.terms
        XCTAssertTrue(terms.contains { $0.name.hasSuffix("head split") })
        XCTAssertLessThanOrEqual(terms.filter { $0.name.hasSuffix("GEMM tiers") }.count, 1)
    }

    // ── the VAEs, the attentions: their shape functions ──────────────────────────────────────

    /// **The VAEs are their plans**: the encoder's two heaps and host copies at the largest
    /// reference with the RGBA the module hands it; the decoder's, then the RGB copy beyond its heaps.
    func testTheVAEsAreTheirPlans() {
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            let need = Self.shapes(Self.request(w, h, references: [(w, h), (640, 640)]))
            let decoder = QwenImage21VAEDecoder.plan(config: need.vae, latentHeight: h / 16, latentWidth: w / 16)
            XCTAssertEqual(need.decodingPhase.bytes,
                           max(decoder.bytes, decoder.host + 3 * w * h * 4) + decoder.outside, "\(w)×\(h)")
            let largest = need.references.map(\.prepared).max { $0.width * $0.height < $1.width * $1.height }!
            XCTAssertEqual(need.referencesPhase!.bytes,
                           4 * largest.width * largest.height * 4
                            + { $0.bytes + $0.outside }(QwenImage21VAEEncoder.plan(config: need.vae, height: largest.height, width: largest.width)))
        }
        // The lean plan's decoder takes the smallest band.
        let lean = Self.shapes(Self.request(1024, 1024).lean)
        XCTAssertEqual(lean.decodingPhase.bytes("Qwen decoder registers"),
                       QwenImage21VAEDecoder.plan(config: lean.vae, latentHeight: 64, latentWidth: 64,
                                                  bandPixels: min(QwenImage21VAEDecoder.bandPixels, MemoryPlan.qwenMinimumBand)).registers)
    }

    /// **Nothing of the encoder reaches the later phases** (`finish` hands its heaps back before it
    /// returns): an edit's denoising and decoding carry no encoder term.
    func testTheEncoderDoesNotOutliveItself() {
        let edit = Self.shapes(Self.request(1024, 1024, references: [(1024, 1024)]))
        for phase in [edit.denoisingPhase, edit.decodingPhase] {
            XCTAssertFalse(phase.terms.contains { $0.name.hasPrefix("Qwen encoder") })
        }
    }

    /// Every attention the DiT keeps, by its cache key: the causal text segments with their masks, the
    /// reference segments plain, the target cut by heads.
    func testEveryCachedAttentionIsCharged() {
        let need = Self.shapes(Self.request(1024, 1024, references: [(1024, 1024)]))
        let terms = need.denoisingPhase.terms
        let texts = need.sequence.segments.filter(\.isText).count
        let masks = terms.filter { $0.name.hasSuffix("causal mask") }
        XCTAssertGreaterThanOrEqual(masks.count, 1)
        XCTAssertLessThanOrEqual(masks.count, texts)
        XCTAssertGreaterThanOrEqual(terms.filter { $0.name.hasSuffix("head split") }.count, 1)
        let attention = terms.filter { $0.name.hasSuffix("causal mask") || $0.name.hasSuffix("head split")
            || $0.name.hasSuffix("SDPA queries") }.reduce(0) { $0 + $1.bytes }
        XCTAssertLessThan(attention, 600_000_000)
        // The encoder's causal SDPA: its mask is the square of its sequence, in our pages.
        XCTAssertEqual(need.textPhase.bytes("encoder causal mask"),
                       Attention.maskBytes(sequence: need.encoderTokens, keys: need.encoderTokens))
    }

    /// **No fitted term remains**: the outside terms are the shapes' copies only — MPSGraph's queries, the
    /// vision tower's padded heads and the VAEs' SDPA — and no constant of the frameworks (`OperatorFootprint`) is charged.
    func testNoFittedTermRemains() {
        for request in [Self.request(1024, 1024), Self.request(1024, 1536, references: [(1024, 1024), (640, 640)], ranks: [64])] {
            let need = Self.need(request)
            for term in need.phases.flatMap(\.terms) where term.outside {
                XCTAssertTrue(term.name.hasSuffix("SDPA queries") || term.name.hasSuffix("head padding")
                              || term.name.hasSuffix(" outside"), term.name)
            }
            XCTAssertFalse(need.phases.flatMap(\.terms).contains { $0.name.contains("executable") || $0.name.contains("MPS GEMM") })
            XCTAssertEqual(need.base, 0)
        }
    }

    // ── the inputs the need was blind to (12/10/2026) ───────────────────────────────────────

    /// **The head split's threads, injected**: on 2 to 12 P cores, an edit with more references never
    /// needs less than with fewer, and each head split charged is exactly `HeadSplitAttention.hostBytes`
    /// at that thread count — the arena `init` maps. (The CI's runner, with fewer P cores, had a
    /// generation above a three-reference edit: true there — a generation's staging blocks against
    /// thinner per-thread scratches —, so no test compares a generation with an edit any more.)
    func testThreadsAreAnInputNotTheMachine() {
        let refs: [(width: Int, height: Int)] = [(1024, 1024), (1216, 832), (1024, 1536)]
        for threads in 2...12 {
            for (w, h) in [(1024, 1024), (1024, 1536)] {
                for lean in [false, true] {
                    let peaks = (1...3).map { n -> Int in
                        var r = Self.request(w, h, references: Array(refs.prefix(n)), threads: threads)
                        if lean { r = r.lean }
                        return Self.need(r).peak
                    }
                    XCTAssertEqual(peaks, peaks.sorted(), "\(threads) threads, \(w)×\(h)\(lean ? " lean" : "")")
                }
            }
            let need = Self.shapes(Self.request(1024, 1024, references: [(1024, 1024)], threads: threads))
            let target = need.sequence.target
            let cpu = HeadSplitAttention.partition(heads: 32, fraction: HeadSplitAttention.referenceFraction).cpu.count
            let gpu = 32 - cpu
            let split = MemoryNeed.attention("t", heads: 32, rows: target, keys: need.sequence.count, headDim: 128,
                                             causal: false, assignment: .init(kernel: .split, cpuHeads: cpu), threads: threads)
            XCTAssertEqual(split, [.init("t head split", HeadSplitAttention.hostBytes(heads: 32, sequence: target,
                                                                                    keys: need.sequence.count, headDim: 128,
                                                                                    cpuHeads: cpu, threads: threads)),
                                   .init("t SDPA queries", target * gpu * 128 * 4, outside: true)])
            // The need's own head splits are those, at this thread count — never the machine's.
            let charged = need.denoisingPhase.terms.filter { $0.name.hasSuffix("head split") }
            XCTAssertFalse(charged.isEmpty)
            for term in charged {
                let other = Self.shapes(Self.request(1024, 1024, references: [(1024, 1024)], threads: threads + 1))
                    .denoisingPhase.terms.first { $0.name == term.name }!
                XCTAssertLessThan(term.bytes, other.bytes, "a thread more, a scratch more")
            }
        }
    }

    /// **The tokenizer is charged while it lives** — its tables, from the file's counts, in the vision
    /// and text phases (`Qwen3VLEncoder` holds it), never after.
    func testTheTokenizerIsCharged() {
        let need = Self.need(Self.request(1024, 1024, references: [(1024, 1024)]))
        XCTAssertEqual(MemoryFixtures.qwenTokenizerTables, 28_213_248)
        for kind in [MemoryNeed.Phase.Kind.vision, .text] {
            XCTAssertEqual(Self.phase(need, kind)!.bytes("tokenizer tables"), MemoryFixtures.qwenTokenizerTables, "\(kind)")
        }
        for kind in [MemoryNeed.Phase.Kind.references, .denoising, .decoding] {
            XCTAssertEqual(Self.phase(need, kind)!.bytes("tokenizer"), 0, "\(kind)")
        }
    }

    /// **A batch of 2 and more**: the other images' latents cross every phase, the images decoded
    /// before the last wait in the decoding; one DiT at a time — the next image keeps the previous
    /// full DiT only when it builds no half-size one first (`keepsDenoiser`), so the denoising is
    /// the same as one image's.
    func testABatchHoldsItsOtherImages() {
        for (w, h) in [(1024, 1024), (1024, 1536)] {
            var one = Self.request(w, h), three = one
            three.batch = 3
            let a = Self.need(one), b = Self.need(three)
            let latent = 64 * (h / 16) * (w / 16) * 4
            XCTAssertEqual(b.crossing - a.crossing, 2 * latent)
            XCTAssertEqual(Self.phase(b, .decoding)!.bytes - Self.phase(a, .decoding)!.bytes, 2 * 3 * w * h * 4)
            XCTAssertEqual(Self.phase(b, .denoising)!.bytes, Self.phase(a, .denoising)!.bytes)
            XCTAssertGreaterThan(b.peak, a.peak)
            one.batch = 2
            XCTAssertLessThan(Self.need(one).peak, b.peak)
        }
        // A generation that halves first (only when `SILICONED_SPECTRAL` forces it, since 0.10) does not
        // keep the full DiT; one at full size — the product's, at every format — does.
        let module = QwenImage21DenoisingModule(map: "", scheduler: .init())
        if Spectral.forced == nil {
            XCTAssertEqual(module.plan(height: 64, width: 64, steps: 8, start: 0, withLoRA: false).reduced, 0)
        }
        XCTAssertFalse(QwenImage21DenoisingModule.keepsDenoiser(reduced: 2))
        XCTAssertTrue(QwenImage21DenoisingModule.keepsDenoiser(reduced: 0))
        XCTAssertEqual(module.plan(height: 32, width: 32, steps: 8, start: 0, withLoRA: false).reduced, 0)
    }

    /// **The map's type is read from its header**: an 8-bit map's staging blocks are about half a
    /// bf16 one's, and so are the encoder's buffers.
    func testAnEightBitMapNeedsSmallerBuffers() {
        let bf16 = MemoryFixtures.qwenMaps(), int8 = MemoryFixtures.qwenMaps(bytesPerValue: 1)
        XCTAssertEqual(bf16.ditLayerStaging, 436_240_384)
        XCTAssertLessThan(int8.ditLayerStaging, bf16.ditLayerStaging * 6 / 10)
        XCTAssertLessThan(int8.encoderLayerStaging, bf16.encoderLayerStaging * 6 / 10)
        let generation = Self.request(1024, 1024)
        let a = Self.need(generation)
        let b = Self.need(MemoryFixtures.qwenRequest(1024, 1024, tokens: 27, maps: int8))
        XCTAssertEqual(Self.phase(a, .denoising)!.bytes("map staging") - Self.phase(b, .denoising)!.bytes("map staging"),
                       TailStream.unitSlots * (bf16.ditUnitStaging - int8.ditUnitStaging))
    }

    /// **Maps that do not parse make no maps** — and no need: the render is refused, never planned on 0.
    func testUnreadableMapsMakeNoNeed() {
        XCTAssertNil(QwenImage21Maps(dit: [:], encoder: [:], tokenizerTables: 1))
        // A DiT header without a tensor table: no staging block to size.
        let encoder: [String: Any] = ["last_layer": 1, "config": ["hidden_size": 8, "num_attention_heads": 2,
                                                                  "num_key_value_heads": 1, "head_dim": 4,
                                                                  "intermediate_size": 16]]
        XCTAssertNil(QwenImage21Maps(dit: ["config": [String: Any]()], encoder: encoder, tokenizerTables: 1))
    }
}
