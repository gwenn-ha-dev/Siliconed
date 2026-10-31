import XCTest
@testable import Siliconed

/// **Z-Image's DiT outside its blocks, without a byte of weights**: the image's padding to 32 and the
/// LoRA on the modules around the blocks. A one-layer DiT (`d = 384`) on a map written by hand, its blocks
/// zero — each is then the identity (gates `tanh(0)`, norms of weight 0) —, the outer modules carrying
/// values exact in bf16.
final class ZImageDiTTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-zimage-dit-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    /// A `SILICON` v3 map, each tensor bf16 on a 16-byte page: its `values` (exact in bf16), zeros without.
    private func map(_ header: [String: Any], tensors: [(name: String, shape: [Int], values: [Float]?)]) throws -> String {
        let page = 16, dataStart = 1 << 16
        var table: [String: Any] = [:], offset = dataStart, data: [(Int, [Float])] = []
        for t in tensors {
            let count = t.shape.reduce(1, *)
            table[t.name] = ["offset": offset, "bytes": count * 2, "shape": t.shape, "dtype": "bfloat16"]
            if let v = t.values { precondition(v.count == count); data.append((offset, v)) }
            offset += (count * 2 + page - 1) / page * page
        }
        var full = header
        full["page"] = page; full["tensors"] = table; full["order"] = tensors.map(\.name)
        let json = try JSONSerialization.data(withJSONObject: full)
        precondition(16 + json.count <= dataStart, "header longer than the test's layout")
        var bytes = Array("SILICON".utf8) + [3]
        for i in 0..<8 { bytes.append(UInt8((UInt64(json.count) >> (8 * UInt64(i))) & 0xff)) }
        bytes += json
        bytes += [UInt8](repeating: 0, count: max(offset, dataStart + 16) - bytes.count)
        for (start, values) in data {
            for (i, v) in values.enumerated() {
                let bits = v.bitPattern >> 16
                precondition(Float(bitPattern: bits << 16) == v, "\(v) is not exact in bf16")
                bytes[start + 2 * i] = UInt8(bits & 0xff); bytes[start + 2 * i + 1] = UInt8(bits >> 8)
            }
        }
        let path = root.appendingPathComponent("m-\(UUID()).silicon").path
        try Data(bytes).write(to: URL(fileURLWithPath: path))
        return path
    }

    /// Values exact in bf16: `((i·a + b) mod 13 − 6) / 32`.
    private func values(_ count: Int, _ a: Int, _ b: Int) -> [Float] {
        (0..<count).map { (i: Int) -> Float in Float((i * a + b) % 13 - 6) / 32 }
    }

    /// `d = 384`, not smaller: a block's weight reserve is `d · hidden` floats, sized for the widest of its
    /// weights — on Z-Image's shapes, never its modulation `256 × 4d`, which a `d` under 384 would overflow.
    private let (d, capDim, inDim) = (384, 16, 64)
    private let config: [String: Any] = ["dim": 384, "n_heads": 3, "n_layers": 1, "n_refiner_layers": 1, "in_channels": 16,
                                         "cap_feat_dim": 16, "axes_dims": [32, 48, 48], "axes_lens": [64, 32, 32]]

    /// The DiT's map: the outer modules with values, `[k, n]` as the forge lays them out; the blocks zero.
    private func dit() throws -> Artifact {
        let (h, a) = (1024, 256)   // `hidden = ⌊d / 3 · 8⌋`, the step embedding's width
        var tensors: [(name: String, shape: [Int], values: [Float]?)] = [
            ("t_embedder.mlp.0.weight", [a, 1024], values(a * 1024, 5, 1)), ("t_embedder.mlp.0.bias", [1024], values(1024, 3, 2)),
            ("t_embedder.mlp.2.weight", [1024, a], values(1024 * a, 7, 3)), ("t_embedder.mlp.2.bias", [a], values(a, 2, 5)),
            ("all_x_embedder.2-1.weight", [inDim, d], values(inDim * d, 4, 1)), ("all_x_embedder.2-1.bias", [d], values(d, 3, 4)),
            ("cap_embedder.0.weight", [capDim], values(capDim, 1, 9)),
            ("cap_embedder.1.weight", [capDim, d], values(capDim * d, 6, 2)), ("cap_embedder.1.bias", [d], values(d, 5, 6)),
            ("cap_pad_token", [1, d], values(d, 2, 7)), ("x_pad_token", [1, d], values(d, 9, 8)),
        ]
        for (prefix, modulated) in [("noise_refiner.0.", true), ("context_refiner.0.", false), ("layers.0.", true)] {
            if modulated {
                tensors += [(prefix + "adaLN_modulation.0.weight", [a, 4 * d], nil), (prefix + "adaLN_modulation.0.bias", [4 * d], nil)]
            }
            tensors += [(prefix + "attention_norm1.weight", [d], nil), (prefix + "attention.to_q.weight", [d, d], nil),
                        (prefix + "attention.to_k.weight", [d, d], nil), (prefix + "attention.to_v.weight", [d, d], nil),
                        (prefix + "attention.norm_q.weight", [128], nil), (prefix + "attention.norm_k.weight", [128], nil),
                        (prefix + "attention.to_out.0.weight", [d, d], nil), (prefix + "attention_norm2.weight", [d], nil),
                        (prefix + "ffn_norm1.weight", [d], nil), (prefix + "feed_forward.w1.weight", [d, h], nil),
                        (prefix + "feed_forward.w3.weight", [d, h], nil), (prefix + "feed_forward.w2.weight", [h, d], nil),
                        (prefix + "ffn_norm2.weight", [d], nil)]
        }
        tensors += [("all_final_layer.2-1.adaLN_modulation.1.weight", [a, d], values(a * d, 3, 7)),
                    ("all_final_layer.2-1.adaLN_modulation.1.bias", [d], values(d, 4, 3)),
                    ("all_final_layer.2-1.linear.weight", [d, inDim], values(d * inDim, 8, 5)),
                    ("all_final_layer.2-1.linear.bias", [inDim], values(inDim, 6, 1))]
        return try Artifact(path: try map(["kind": "z-image-turbo-dit", "linear_weights_transposed": true, "config": config],
                                          tensors: tensors))
    }

    private func lora(_ targets: [(String, Int, Int)], rank: Int = 2, strength: Float = 0.5) throws -> LoRA {
        let tensors = targets.flatMap { t -> [(name: String, shape: [Int], values: [Float]?)] in
            [("\(t.0).down", [t.1, rank], values(t.1 * rank, 4, 3)), ("\(t.0).up", [rank, t.2], values(rank * t.2, 6, 5))]
        }
        let path = try map(["kind": "lora"], tensors: tensors)
        return try LoRA(layers: [LoRA.Layer(artifact: try Artifact(path: path), strength: strength, name: path)])
    }

    /// A latent of 8 × 10 cells: 4 × 5 = 20 patches, padded to 32; a caption of 5 rows, padded to 32.
    private let (latentHeight, latentWidth, capLength) = (8, 10, 5)
    private lazy var latent = (0..<(16 * latentHeight * latentWidth)).map { (i: Int) -> Float in Float((i * 7) % 11 - 5) / 8 }
    private lazy var caps = (0..<(capLength * capDim)).map { (i: Int) -> Float in Float((i * 5) % 9 - 4) / 4 }

    private func evaluate(_ artifact: Artifact, lora: LoRA? = nil) throws -> (out: [Float], boundaries: [String: [Float]]) {
        let dit = try DiT(artifact: artifact, latentHeight: latentHeight, latentWidth: latentWidth,
                          capLength: capLength, lora: lora)
        dit.recordBoundaries = true
        let out = try latent.withUnsafeBufferPointer { l in
            try caps.withUnsafeBufferPointer { c in try dit.forward(latent: l.baseAddress!, caps: c.baseAddress!, sigma: 0.8) }
        }
        return (out, dit.boundaries)
    }

    // ── the image's padding (`_pad_with_ids`) ─────────────────────────────────────────────────

    /// 20 patches → 32 rows: the copies of the last patch through `x_embedder`, then `x_pad_token` in the
    /// stream, at RoPE position (0, 0, 0); the caption after the padded image; the velocity at `C·h·w`.
    func testTheImageIsPaddedToThirtyTwoWithItsLearnedToken() throws {
        let artifact = try dit()
        let config = DiT.Config(header: artifact.header)
        XCTAssertEqual(DiT.imagePadded(latentHeight: latentHeight, latentWidth: latentWidth, config: config), 32)
        XCTAssertEqual(DiT.imagePadded(latentHeight: 152, latentWidth: 104, config: config), 3968, "832×1216")
        XCTAssertEqual(DiT.imagePadded(latentHeight: 128, latentWidth: 128, config: config), 4096, "1024²: unchanged")
        let (out, b) = try evaluate(artifact)
        XCTAssertEqual(out.count, 16 * latentHeight * latentWidth)
        let (patches, rows) = (20, 32)
        let embedded = try XCTUnwrap(b["x_embedder_out"]), refined = try XCTUnwrap(b["noise_refiner_out"])
        XCTAssertEqual(embedded.count, rows * d)
        XCTAssertEqual(refined.count, rows * d)
        for row in patches..<rows {
            XCTAssertEqual(Array(embedded[row * d..<(row + 1) * d]), Array(embedded[(patches - 1) * d..<patches * d]),
                           "row \(row): the last patch's embedding")
            XCTAssertEqual(Array(refined[row * d..<(row + 1) * d]), values(d, 9, 8), "row \(row): x_pad_token")
        }
        XCTAssertEqual(Array(refined[0..<patches * d]), Array(embedded[0..<patches * d]), "identity blocks")
        let unified = try XCTUnwrap(b["unified_in"])
        XCTAssertEqual(unified.count, (rows + 32) * d)
        XCTAssertEqual(Array(unified[(rows + capLength) * d..<(rows + capLength + 1) * d]), values(d, 2, 7),
                       "the caption's padding after the image's")
        let freqs = try XCTUnwrap(b["block_freqs_cis"]), perToken = (32 + 48 + 48) / 2 * 2
        XCTAssertEqual(freqs.count, (rows + 32) * perToken)
        for row in patches..<rows {
            let pairs = freqs[row * perToken..<(row + 1) * perToken]
            XCTAssertEqual(pairs.enumerated().filter { $0.offset % 2 == 0 }.map(\.element), Array(repeating: 1, count: perToken / 2))
            XCTAssertEqual(pairs.enumerated().filter { $0.offset % 2 == 1 }.map(\.element), Array(repeating: 0, count: perToken / 2))
        }
    }

    // ── the LoRA on the modules outside the blocks ────────────────────────────────────────────

    /// Each of the six outer modules, adapted alone, moves its own boundary — they used to be computed
    /// without the stack the forge kept for them. `x_embedder`'s move is the stack's product, exactly.
    func testALoRAOnTheOuterModulesIsApplied() throws {
        let artifact = try dit()
        let base = try evaluate(artifact).boundaries
        for (module, k, n, boundary) in [("t_embedder.mlp.0", 256, 1024, "t_embedder_out"),
                                         ("t_embedder.mlp.2", 1024, 256, "t_embedder_out"),
                                         ("all_x_embedder.2-1", inDim, d, "x_embedder_out"),
                                         ("cap_embedder.1", capDim, d, "cap_embedder_out"),
                                         ("all_final_layer.2-1.adaLN_modulation.1", 256, d, "final_layer_out"),
                                         ("all_final_layer.2-1.linear", d, inDim, "final_layer_out")] {
            XCTAssertTrue(DiT.appliesLoRA(module), module)
            let adapted = try evaluate(artifact, lora: try lora([(module, k, n)])).boundaries
            let (before, after) = (try XCTUnwrap(base[boundary]), try XCTUnwrap(adapted[boundary]))
            XCTAssertEqual(before.count, after.count)
            let moved = zip(before, after).map { abs($0 - $1) }.max() ?? 0
            XCTAssertGreaterThan(moved, 1e-3, "\(module) → \(boundary): left out")
        }
        // `x_embedder`: x · W + b + s · (x · down) · up, the patches as `Ops.patchify` lays them out.
        let adapted = try evaluate(artifact, lora: try lora([("all_x_embedder.2-1", inDim, d)])).boundaries
        var tokens = [Float](repeating: 0, count: 32 * inDim)
        latent.withUnsafeBufferPointer { l in
            tokens.withUnsafeMutableBufferPointer {
                Ops.patchify(l.baseAddress!, into: $0.baseAddress!, channels: 16, height: latentHeight, width: latentWidth, patch: 2)
            }
        }
        for row in 20..<32 { for c in 0..<inDim { tokens[row * inDim + c] = tokens[19 * inDim + c] } }
        let (down, up) = (values(inDim * 2, 4, 3), values(2 * d, 6, 5))
        let before = try XCTUnwrap(base["x_embedder_out"]), after = try XCTUnwrap(adapted["x_embedder_out"])
        var worst: Float = 0
        for row in 0..<32 {
            let mid = (0..<2).map { q in (0..<inDim).reduce(Float(0)) { $0 + tokens[row * inDim + $1] * down[$1 * 2 + q] } }
            for c in 0..<d {
                let expected = 0.5 * (0..<2).reduce(Float(0)) { $0 + mid[$1] * up[$1 * d + c] }
                worst = max(worst, abs(after[row * d + c] - before[row * d + c] - expected))
            }
        }
        XCTAssertLessThan(worst, 1e-5)
    }

    /// What the engine does not adapt is refused before any computation, never left out; what the forge
    /// renames from the original names lands on a module the engine adapts.
    func testWhatIsNotAppliedIsRefusedAndTheForgeNamesLandOnAppliedModules() throws {
        let artifact = try dit()
        for module in ["cap_embedder.0", "context_refiner.0.adaLN_modulation.0", "layers.0.attention_norm1"] {
            XCTAssertFalse(DiT.appliesLoRA(module), module)
            XCTAssertThrowsError(try DiT(artifact: artifact, latentHeight: latentHeight, latentWidth: latentWidth,
                                         capLength: capLength, lora: try lora([(module, 16, 16)]))) {
                XCTAssertTrue("\($0)".contains(module), "\($0)")
            }
        }
        for (published, target) in [("x_embedder", "all_x_embedder.2-1"), ("final_layer.linear", "all_final_layer.2-1.linear"),
                                    ("final_layer.adaLN_modulation.1", "all_final_layer.2-1.adaLN_modulation.1"),
                                    ("cap_embedder.1", "cap_embedder.1"), ("t_embedder.mlp.0", "t_embedder.mlp.0"),
                                    ("t_embedder.mlp.2", "t_embedder.mlp.2"), ("layers.3.attention.out", "layers.3.attention.to_out.0")] {
            let targets = try ForgeLoRA.targets(published, family: .zImage).map(\.module)
            XCTAssertEqual(targets, [target], published)
            XCTAssertTrue(DiT.appliesLoRA(target), target)
        }
        let config = DiT.Config(header: artifact.header)
        let targets = DiT.loraTargets(config: config)
        XCTAssertTrue(targets.allSatisfy { DiT.appliesLoRA($0.name) })
        XCTAssertEqual(targets.count, 8 * (config.refiners + config.layers) + 7 * config.refiners + 6)
        XCTAssertEqual(DiT.loraModules(config: config).map(\.k), targets.map(\.k))
    }

    /// Under a stack, the DiT's own arena carries `loraMid [S, r]`: the allocation is the need's formula.
    func testTheOwnArenaWithAStackIsTheFormula() throws {
        let artifact = try dit()
        let config = DiT.Config(header: artifact.header)
        let stack = try lora([("all_final_layer.2-1.linear", d, inDim)], rank: 3)
        let dit = try DiT(artifact: artifact, latentHeight: latentHeight, latentWidth: latentWidth, capLength: capLength,
                          lora: stack)
        XCTAssertEqual(dit.arenaCapacity, DiT.arenaBytes(config: config, latentHeight: latentHeight, latentWidth: latentWidth,
                                                         capLength: capLength, loraRank: 3))
        XCTAssertEqual(dit.cacheLoRACapacity, CacheLoRA.arenaBytes(CacheLoRA.bytes(modules: [(d, inDim, 3)])))
    }

    /// **Without the head split, Z-Image's SDPA is sliced past `maxMatrix`** (18/10): at 1024×1536 the
    /// sequence is 6 144 image rows and a padded caption — 30 × 6 176² × 4 = 4.6 GB in one SDPA, the
    /// regime where Krea 2's died; two query slices keep each under the bound. At 1024² one SDPA, as before.
    func testZImageAttentionIsSlicedPastTheBound() {
        XCTAssertEqual(Attention.groupedSlices(heads: 30, rows: 6144 + 32, keys: 6144 + 32), 2)
        XCTAssertEqual(Attention.groupedSlices(heads: 30, rows: 4096 + 32, keys: 4096 + 32), 1)
        XCTAssertEqual(Attention.groupedSlices(heads: 30, rows: 4096 + 32, keys: 4096 + 32, limit: 30 * 4128 * 4128 * 4 / 3), 3)
    }
}
