import Foundation

/// **The forge of a DiT**: a normalized source → the map that its family's engine reads.
///
/// It replaces the earlier Python forges, rule for rule — and the same
/// recipe serves the DiT published by the vendor as well as the checkpoint a user brings: only the
/// name normalization differs (`Recipes.normalize`). What each family imposes:
///
///   · **Z-Image** — bf16, except `t_embedder` and `cap_embedder` kept in fp32
///     (`transformer_z_image.py:312`, "precision sensitive layers").
///   · **Krea 2** — the norms in fp32 with `Krea2RMSNorm`'s `1 +` folded offline; the blocks'
///     `Linear` in bf16, everything else (time, modulation, inputs, output, tables) in fp32: what
///     the vendor publishes, and what the engine reads.
///   · **Anima** — bf16, and the RoPE moved from Cosmos's split convention to `Ops.rope`'s
///     interleaved one by permuting the outputs of `attn1.to_q/to_k` and `norm_q/norm_k` per head.
///   · **Qwen-Image-2.1** — bf16 as published, except `txt_in.text_norm`, a "zero-centered" RMSNorm
///     (the checkpoint stores `scale − 1`): written in fp32 with the `+ 1` folded, which is exactly
///     what the reference computes (`weight.float() + 1`, `QwenImage21ZeroCenterRMSNorm`). The
///     other norms (`norm_q`, `norm_k`) are plain RMSNorms. Its RoPE multiplies complex pairs
///     `(2p, 2p+1)` (`apply_rotary_emb_qwen`, `use_real=False`): interleaved, nothing permuted.
///
/// All: `Linear` weights transposed `[input, output]`, file order = execution order.
package enum ForgeDiT {
    package struct Report {
        package let tally: MapWriter.Tally
        package let naming: String
        package let ignored: Int
        package let dtypesSource: [String: Int]
        /// 8-bit weights kept as published, by publisher's format.
        package let quantizedFormats: [String: Int]
        package let kept8bit: Int
        /// Packed GGUF weights (Q4_0…Q6_K) kept as published (blocks copied whole, rows moved only).
        package let keptPacked: Int
        /// 8-bit weights a transformation forced into fp32, with the reason.
        package let dequantized: [String]
        /// Non-8-bit tensors kept in fp32 because bf16 (or fp16) would have rounded them.
        package let widened: [String]
        /// Tensors published in fp16 and kept in fp16.
        package let kept16bit: Int
        /// Anima adapter tensors that came 8-bit and were written dequantized in fp32.
        package let adapterDequantized: Int
        package let notes: [String]
        package let adapter: String?
        package var rows: [String] {
            var l = ["naming recognized: \(naming)",
                     "dtypes read: " + dtypesSource.sorted { $0.key < $1.key }.map { "\($0.key) ×\($0.value)" }.joined(separator: ", ")]
            if !quantizedFormats.isEmpty {
                l.append((quantizedFormats.keys.allSatisfy { $0.hasSuffix("Q8_0") || !$0.hasPrefix("GGUF") } ? "8-bit weights read: " : "quantized weights read: ") + quantizedFormats.sorted { $0.key < $1.key }.map { "\($0.key) ×\($0.value)" }
                    .joined(separator: ", "))
                if kept8bit > 0 || keptPacked == 0 {
                    l.append("8-bit kept as published (bytes and scales copied, rows moved or transposed only): \(kept8bit) tensors")
                }
                if keptPacked > 0 {
                    l.append("GGUF 4–6-bit kept as published (blocks copied whole, rows moved only): \(keptPacked) tensors")
                }
            }
            if !dequantized.isEmpty {
                l.append("stored dequantized in fp32 (a transformation cannot stay \(keptPacked > 0 ? "quantized" : "8-bit")): "
                         + "\(dequantized.count) — "
                         + dequantized.prefix(4).joined(separator: "; ") + (dequantized.count > 4 ? "; …" : ""))
            }
            if kept16bit > 0 { l.append("fp16 kept as published: \(kept16bit) tensors") }
            if adapterDequantized > 0 {
                l.append("text adapter: \(adapterDequantized) 8-bit tensors written dequantized in fp32 (the conditioner reads no 8-bit)")
            }
            if !widened.isEmpty {
                l.append("kept in fp32 (a narrower type would round them): \(widened.count) — "
                         + widened.prefix(4).joined(separator: ", ") + (widened.count > 4 ? ", …" : ""))
            }
            l += notes
            if ignored > 0 { l.append("\(ignored) embedded tensors ignored (encoder, VAE: the family's are used)") }
            l.append("values rounded: \(tally.inexact) out of \(tally.parameters)"
                     + (tally.inexact == 0 ? " — bit-exact" : ""))
            l.append(String(format: "%@ : %.2f GB, %d tensors, max |weight| %.4f",
                            (tally.path as NSString).lastPathComponent, Double(tally.bytes) / 1e9,
                            tally.tensors, tally.absMax))
            if let a = adapter { l.append("text adapter: \((a as NSString).lastPathComponent)") }
            return l
        }
    }

    /// After normalization — for FLUX.2 [klein] 4B, the 169 published tensors and their 110 split
    /// parts (`Recipes.flux2Split`): 279.
    static let expectedCounts: [Family: Int] = [.zImage: 521, .krea2: 430, .anima: 567, .klein4b: 279, .ernie: 409,
                                                  .qwenImage21: 297]
    static let animaHeadSize = 128

    /// The order the engine reads in: a sequential read of the file is an execution.
    package static func order(_ names: [String], family: Family) -> [String] {
        func index(_ n: String, _ k: Int) -> Int { Int(n.split(separator: ".")[k]) ?? 0 }
        let rank: (String) -> (Int, Int) = { n in
            switch family {
            case .zImage:
                if n.hasPrefix("noise_refiner.") { return (1, index(n, 1)) }
                if n.hasPrefix("context_refiner.") { return (2, index(n, 1)) }
                if n.hasPrefix("layers.") { return (3, index(n, 1)) }
                if n.contains("final_layer") { return (4, 0) }
                return (0, 0)
            case .krea2:
                if n.hasPrefix("transformer_blocks.") { return (2, index(n, 1)) }
                if n.hasPrefix("final_layer.") { return (3, 0) }
                if n.hasPrefix("text_fusion.layerwise_blocks.") { return (1, index(n, 2)) }
                if n.hasPrefix("text_fusion.projector") { return (1, 50) }
                if n.hasPrefix("text_fusion.refiner_blocks.") { return (1, 100 + index(n, 2)) }
                if n.hasPrefix("txt_in.") { return (1, 200) }
                return (0, 0)
            case .anima:
                if n.hasPrefix("transformer_blocks.") { return (1, index(n, 1)) }
                if n.hasPrefix("norm_out.") || n.hasPrefix("proj_out.") { return (2, 0) }
                return (0, 0)
            case .qwenImage21:
                if n.hasPrefix("transformer_blocks.") { return (1, index(n, 1)) }
                if n.hasPrefix("norm_out.") || n.hasPrefix("proj_out.") { return (2, 0) }
                return (0, 0)
            case .ernie:
                if n.hasPrefix("layers.") { return (1, index(n, 1)) }
                if n.hasPrefix("final_norm.") || n.hasPrefix("final_linear.") { return (2, 0) }
                return (0, 0)
            case .klein4b:
                if n.hasPrefix("transformer_blocks.") { return (1, index(n, 1)) }
                if n.hasPrefix("single_transformer_blocks.") { return (2, index(n, 1)) }
                if n.hasPrefix("norm_out.") || n.hasPrefix("proj_out.") { return (3, 0) }
                return (0, 0)
            }
        }
        let keys: [(Int, Int, String)] = names.map { n in let r = rank(n); return (r.0, r.1, n) }
        return keys.sorted { $0 < $1 }.map { $0.2 }
    }

    /// Krea 2: which kind of tensor (the earlier Python forge's `kind`).
    enum KreaKind { case norm, table, linear, simple }
    static func kreaKind(_ name: String, shape: [Int]) -> KreaKind {
        if name.firstMatch(#"(^|\.)(norm|norm1|norm2|norm_q|norm_k)\.weight$"#) != nil { return .norm }
        if name.hasSuffix("scale_shift_table") { return .table }
        if shape.count == 2 && name.hasSuffix(".weight") { return .linear }
        return .simple
    }

    /// The map's dtype. It depends only on the name: an fp8 or fp16 checkpoint produces a map with
    /// the same dtypes as the vendor's — the engine has only one layout to read.
    static func dtype(_ name: String, family: Family) -> MapWriter.DType {
        switch family {
        case .zImage:
            return name.contains("t_embedder") || name.contains("cap_embedder") ? .float32 : .bfloat16
        case .anima, .klein4b, .ernie, .qwenImage21:
            return .bfloat16
        case .krea2:
            if name.firstMatch(#"\.attn\.to_(q|k|v|gate|out\.0)\.weight$"#) != nil
                || name.firstMatch(#"\.ff\.(gate|up|down)\.weight$"#) != nil { return .bfloat16 }
            return .float32
        }
    }

    static func transposed(_ name: String, shape: [Int], family: Family) -> Bool {
        if family == .krea2 { return kreaKind(name, shape: shape) == .linear }
        return shape.count == 2 && name.hasSuffix(".weight")
    }

    /// The source row of each row of `interleave` — what moves an 8-bit weight's rows (and their
    /// scales) the same way.
    static func interleaveOrder(rows: Int, head: Int) -> [Int] {
        let half = head / 2
        return (0..<rows).map { l in
            let h = l / head, j = l % head
            return h * head + (j % 2 == 0 ? j / 2 : half + j / 2)
        }
    }

    /// `0, Dh/2, 1, Dh/2+1, …` per head, on axis 0 — the split RoPE becomes interleaved.
    static func interleave(_ v: [Float], rows: Int, columns: Int, head: Int) -> [Float] {
        permutingRows(v, interleaveOrder(rows: rows, head: head), columns: columns)
    }

    /// Row `l` of the result = row `order[l]` of `v` (`columns` values per row) — what
    /// `QuantizedTensor.selectingRows` does to an 8-bit weight, on fp32 values.
    static func permutingRows(_ v: [Float], _ order: [Int], columns: Int) -> [Float] {
        var output = [Float](repeating: 0, count: v.count)
        v.withUnsafeBufferPointer { s in
            output.withUnsafeMutableBufferPointer { d in
                for (l, origin) in order.enumerated() {
                    (d.baseAddress! + l * columns).update(from: s.baseAddress! + origin * columns, count: columns)
                }
            }
        }
        return output
    }

    /// Qwen-Image-2.1's zero-centered RMSNorm (`QwenImage21ZeroCenterRMSNorm`): the only one.
    static let qwenZeroCentered = ["txt_in.text_norm.weight"]

    static func permutedRoPE(_ name: String) -> Bool {
        name.firstMatch(#"\.attn1\.(to_[qk]|norm_[qk])\.weight$"#) != nil
    }

    /// **Forges a DiT's map.**
    ///
    /// - Parameters:
    ///   - config: the family's published `transformer/config.json` (copied into the header, which
    ///     the engine reads).
    ///   - reference: the published names and shapes of the vendor's DiT — an imported checkpoint
    ///     must have exactly the same ones. `nil`: only the count is checked.
    ///   - name: for an imported model, the displayed name; it goes into the header.
    package static func forge(source: TensorSource, family: Family, config: OrderedJSON,
                               to path: String, reference: [String: [Int]]? = nil,
                               name: String? = nil, descriptionSource: OrderedJSON,
                               reserve: ((Int) throws -> Void)? = nil,
                               checkingCount: Bool = true,
                               progressHandler: ((Int, Int, String) -> Void)? = nil) throws -> Report {
        let n = try Recipes.normalize(source, family: family)
        let names = Array(n.dit.keys)
        if checkingCount, let expected = expectedCounts[family], names.count != expected {
            let example = reference.map { r in Set(r.keys).symmetricDifference(names).sorted().prefix(4) } ?? []
            throw Numerics.Failure(description: "\(names.count) DiT tensors, \(expected) expected for \(family.name)"
                                  + (example.isEmpty ? "" : " — deviations: \(Array(example))"))
        }
        if let r = reference {
            let missingNames = Set(r.keys).subtracting(names).sorted(), excess = Set(names).subtracting(r.keys).sorted()
            guard missingNames.isEmpty, excess.isEmpty else {
                throw Numerics.Failure(description: "names disagree with \(family.name) : \(missingNames.count) missing "
                                      + "\(missingNames.prefix(3)), \(excess.count) unknown \(excess.prefix(3))")
            }
            for (k, p) in n.dit where r[k] != p.shape {
                throw Numerics.Failure(description: "\(k) : shape \(p.shape), \(r[k]!) expected — this is not a \(family.name)")
            }
        }
        let order = order(names, family: family)
        var permuted = 0
        var dtypes: [String: Int] = [:]
        var tensors: [MapWriter.Tensor] = []
        var kept8bit = 0, kept16bit = 0, keptPacked = 0
        var dequantized: [String] = [], widened: [String] = []
        for mapName in order {
            let p = n.dit[mapName]!
            dtypes[p.dtypeSource, default: 0] += 1
            if family == .anima && permutedRoPE(mapName) { permuted += 1 }
            let (t, outcome) = try tensor(mapName, p, family: family)
            switch outcome {
            case .kept8bit: kept8bit += 1
            case .keptPacked: keptPacked += 1
            case .kept16bit: kept16bit += 1
            case .dequantized(let why): dequantized.append(mapName + why)
            case .widened: widened.append(mapName)
            case .asNamed: break
            }
            tensors.append(t)
        }
        if family == .anima && permuted != 28 * 4 {
            throw Numerics.Failure(description: "RoPE permutation: \(permuted) tensors touched, 112 expected")
        }
        let parameters = tensors.reduce(0) { $0 + $1.shape.reduce(1, *) }
        let config = config.withoutPrivateKeys

        let tally = try MapWriter.write(to: path, tensors: tensors, header: { m in
            var source = descriptionSource.pairs ?? []
            source += [.init("tensor_count", .integer(order.count)), .init("parameters", .integer(parameters))]
            if family != .krea2 { source.append(.init("content_sha256", .string(m?.sha256 ?? ""))) }
            var h: [OrderedJSON.Pair] = [
                .init("format", 1), .init("kind", .string(family.ditKind)), .init("page", .integer(MapWriter.page)),
                // A bf16 map keeps the value it always had (its header is compared byte for byte);
                // one that holds 8-bit, fp16 or a tensor fp32 by necessity holds several: "mixed".
                .init("map_dtype", .string(kept8bit + keptPacked + kept16bit + widened.count + dequantized.count > 0
                                           || family == .krea2 || family == .qwenImage21 ? "mixed" : "bfloat16")),
                .init("linear_weights_transposed", true),
            ]
            switch family {
            case .zImage:
                h += [.init("config", config), .init("keep_fp32", .list(["t_embedder", "cap_embedder"])),
                      .init("source", .object(source)), .init("weight_absmax", .real(Double(m?.absMax ?? 0)))]
            case .krea2:
                h += [.init("rope_layout", "interleaved"), .init("rmsnorm_plus_one_folded", true),
                      .init("config", config), .init("source", .object(source))]
            case .anima:
                h += [.init("rope_layout", "interleaved"), .init("config", config), .init("source", .object(source)),
                      .init("weight_absmax", .real(Double(m?.absMax ?? 0)))]
            case .ernie:
                // ERNIE's RoPE (repeated angles, `rotate_half`) is neither interleaved nor split:
                // nothing is permuted, `ErnieRope` applies it as is.
                h += [.init("rope_layout", "ernie"), .init("config", config),
                      .init("source", .object(source)), .init("weight_absmax", .real(Double(m?.absMax ?? 0)))]
            case .qwenImage21:
                h += [.init("rope_layout", "interleaved"),
                      .init("zero_centered_norms_folded", .list(qwenZeroCentered.map { .string($0) })),
                      .init("config", config), .init("source", .object(source)),
                      .init("weight_absmax", .real(Double(m?.absMax ?? 0)))]
            case .klein4b:
                // FLUX.2's RoPE is interleaved to begin with: nothing is permuted.
                h += [.init("rope_layout", "interleaved"), .init("fused_split", true), .init("config", config),
                      .init("source", .object(source)), .init("weight_absmax", .real(Double(m?.absMax ?? 0)))]
            }
            if let name { h.append(.init("nom", .string(name))) }
            h.append(.init("order", .list(order.map { .string($0) })))
            return h
        }, reserve: reserve, progressHandler: progressHandler)

        var adapter: String?, adapterDequantized = 0
        if family == .anima {
            guard !n.adapter.isEmpty else {
                throw Numerics.Failure(description: "no text adapter (`llm_adapter.*`) in this file: "
                                      + "an Anima is published with it (ComfyUI format)")
            }
            let a = adapterPath(fromMap: path)
            adapterDequantized = try writeAdapter(source: source, keys: n.adapter, to: a)
            adapter = a
        }
        return Report(tally: tally, naming: n.naming, ignored: n.ignored.count, dtypesSource: dtypes,
                      quantizedFormats: source.quantizedFormats, kept8bit: kept8bit, keptPacked: keptPacked,
                      dequantized: dequantized,
                      widened: widened, kept16bit: kept16bit, adapterDequantized: adapterDequantized,
                      notes: source.notes, adapter: adapter)
    }

    /// What became of a tensor in the map.
    package enum Outcome: Equatable {
        /// The dtype its name gives (bf16 or fp32), exactly.
        case asNamed
        /// 8-bit, bytes and scales as published.
        case kept8bit
        /// A packed GGUF type (Q4_0…Q6_K), blocks as published.
        case keptPacked
        /// 8-bit in the source, fp32 in the map — and why.
        case dequantized(String)
        /// Published in fp16, kept in fp16.
        case kept16bit
        /// fp32 because the narrower type its name or its source gives would have rounded it.
        case widened
    }

    /// What a forge decides for one tensor from its name and shape alone — the rest is the same for
    /// every map, DiT or text encoder (`tensor(_:_:_:)`).
    package struct Rule {
        /// A `Linear` written `[input, output]`.
        package var transposed: Bool
        /// The dtype its name gives, written so when that is exact (`.bfloat16` or `.float32`).
        package var named: MapWriter.DType
        /// Rows moved: row `l` of the map = published row `rowOrder[l]` (axis 0, whatever the shape —
        /// a 1-D norm has one value per row). A per-head RoPE permutation; an 8-bit weight's rows
        /// carry their scales (`QuantizedTensor.selectingRows`). `nil`: none.
        package var rowOrder: [Int]? = nil
        /// A zero-centered norm's `+ 1` folded: arithmetic, so written in fp32, never 8-bit.
        package var foldsOne = false
    }

    /// A DiT's rule: its family's transposition, dtypes, folded norms and RoPE permutation.
    package static func rule(_ mapName: String, shape: [Int], family: Family) -> Rule {
        Rule(transposed: transposed(mapName, shape: shape, family: family), named: dtype(mapName, family: family),
             rowOrder: family == .anima && permutedRoPE(mapName) ? interleaveOrder(rows: shape[0], head: animaHeadSize) : nil,
             foldsOne: family == .krea2 && kreaKind(mapName, shape: shape) == .norm
                || family == .qwenImage21 && qwenZeroCentered.contains(mapName))
    }

    /// One tensor of a DiT's map: `tensor(_:_:_:)` under its family's rule.
    package static func tensor(_ mapName: String, _ p: Provenance, family: Family) throws -> (MapWriter.Tensor, Outcome) {
        try tensor(mapName, p, rule(mapName, shape: p.shape, family: family))
    }

    /// **One tensor of the map — never wider, never narrower than given.** An 8-bit weight stays
    /// 8-bit (bytes and scales copied, rows moved, transposed); an fp16 one stays fp16; anything else
    /// goes through fp32 and is written in the dtype its name gives — unless that would round it.
    /// Shared by the DiT's forge and the text encoder's (`ForgeText`).
    package static func tensor(_ mapName: String, _ p: Provenance, _ rule: Rule) throws -> (MapWriter.Tensor, Outcome) {
        let tr = rule.transposed
        let finalShape = tr ? [p.shape[1], p.shape[0]] : p.shape
        let norm = rule.foldsOne
        let count = p.shape.reduce(1, *)

        // ── An 8-bit weight stays 8-bit: bytes and scales copied, rows moved, transposed ──
        // A transposed Linear keeps any scale (per row → per column of `[K, N]`; by block, GGUF's
        // `[N, K/b]` → `[K/b, N]`). Any other tensor (a cast bias, Krea's `mod.lin` reshaped) is read
        // in its own layout, which the engine dequantizes with a tensor-wide scale only
        // (`Artifact.Scale`: `[]`). A packed GGUF type stays packed only as a transposed `Linear`:
        // its blocks keep their published rows, the engine transposes (`Artifact.DType.q4_k`).
        let packed = p.quantization?.kind.packedBlock
        if let q = p.quantization, let readQuantized = p.readQuantized, !norm, p.dequantizedBecause == nil,
           tr ? p.shape.count == 2 && (q.block.map { p.shape[1] % $0 == 0 } ?? true) && (q.rotation.map { p.shape[1] % $0 == 0 } ?? true)
                && (packed.map { p.shape[1] % $0.values == 0 } ?? true)
              : packed == nil && !q.perRow && q.block == nil && q.rotation == nil {
            let dtype = MapWriter.DType(rawValue: q.kind.rawValue)!
            let scale = q.scaleType.map { type in
                q.block.map { MapWriter.Scale(dtype: type, shape: [p.shape[1] / $0, p.shape[0]], block: $0) }
                    ?? MapWriter.Scale(dtype: type, shape: q.perRow ? [p.shape[0]] : [])
            }
            let rows = p.shape[0]
            var t = MapWriter.Tensor(name: mapName, shape: finalShape, dtype: dtype, scale: scale, transposed: tr) {
                var t = try readQuantized()
                guard t.rows * t.columns == count, !tr || (t.rows == p.shape[0] && t.columns == p.shape[1]) else {
                    throw Numerics.Failure(description: "\(mapName) : 8-bit \([t.rows, t.columns]) for \(p.shape)")
                }
                if let order = rule.rowOrder {
                    // Rows of the map's own layout (a 1-D norm: one value per row).
                    if !tr { (t.rows, t.columns) = (rows, count / rows) }
                    t = t.selectingRows(order)
                }
                let w = t.dequantized()
                let absMax = w.withUnsafeBufferPointer { Numerics.absMax($0.baseAddress!, count: $0.count) }
                guard t.block == q.block, t.rotation == q.rotation, t.kind == q.kind else {
                    throw Numerics.Failure(description: "\(mapName) : scale blocks, rotation or type changed")
                }
                // A packed type's blocks keep their published rows: the map says `[K, N]`, the bytes
                // stay `[N][K/b]` (`Artifact.DType.q4_k`).
                if packed != nil { return .init(values: t.values, scale: [], absMax: absMax) }
                return .init(values: tr ? t.transposedValues() : t.values,
                             scale: t.block != nil ? t.transposedBlockScales() : t.scale, absMax: absMax)
            }
            // ComfyUI's convrot: `q` and its scale as published, the rotation said in the entry
            // (`Artifact.Rotation`); the engine undoes it when it widens (`Widen.dequantizeRotated`).
            t.rotation = q.rotation
            return (t, packed != nil ? .keptPacked : .kept8bit)
        }

        // ── Everything else goes through fp32, where every published value fits exactly ──
        let produce: () throws -> [Float] = {
            var v = try p.read()
            if let order = rule.rowOrder { v = permutingRows(v, order, columns: count / p.shape[0]) }
            if norm { v.withUnsafeMutableBufferPointer { b in for i in b.indices { b[i] += 1 } } }
            guard tr else { return v }
            var t = [Float](unsafeUninitializedCapacity: v.count) { _, c in c = v.count }
            v.withUnsafeBufferPointer { s in
                t.withUnsafeMutableBufferPointer { d in
                    Numerics.transpose(s.baseAddress!, rows: p.shape[0], columns: p.shape[1], to: d.baseAddress!)
                }
            }
            return t
        }
        let named: MapWriter.DType = norm ? .float32 : rule.named
        if p.quantization != nil || p.dequantizedBecause != nil {
            // An arithmetic transformation (a folded `1 +`), a per-row or per-block scale on a tensor
            // that is not a transposed Linear, a column share that cuts a block: its exact values are
            // not 8-bit values times a scale the map can hold.
            let why = p.dequantizedBecause ?? (norm ? " (folded `1 +`)"
                : packed != nil ? " (GGUF \(p.dtypeSource) blocks on a tensor that is not a Linear)"
                : p.quantization?.rotation != nil ? " (convrot on a tensor that is not a Linear)"
                : p.quantization?.block != nil ? " (block scales on a tensor that is not a Linear)"
                : " (per-row scale on a tensor that is not a Linear)")
            return (.init(name: mapName, shape: finalShape, dtype: .float32, transposed: tr, produce: produce),
                    .dequantized(why))
        }
        // **The forge never rounds**: a value the narrower type cannot carry keeps fp32.
        // Costs one extra read of such tensors.
        func exact(_ check: (UnsafePointer<Float>, Int) -> Bool) throws -> Bool {
            try produce().withUnsafeBufferPointer { check($0.baseAddress!, $0.count) }
        }
        if p.dtypeSource == "F16" && (named == .bfloat16 || norm) {
            // Published fp16 stays fp16 — a folded norm too, if `w + 1` still is an fp16.
            if try exact(Numerics.exactInFloat16) {
                return (.init(name: mapName, shape: finalShape, dtype: .float16, transposed: tr, produce: produce), .kept16bit)
            }
            return (.init(name: mapName, shape: finalShape, dtype: .float32, transposed: tr, produce: produce), .widened)
        }
        if named == .bfloat16 && p.dtypeSource != "BF16" {
            // A checkpoint published in fp32 whose values are bf16's (Z-Image's) still gives the
            // bf16 map, byte for byte.
            if !(try exact(Numerics.exactInBFloat16)) {
                return (.init(name: mapName, shape: finalShape, dtype: .float32, transposed: tr, produce: produce), .widened)
            }
        }
        return (.init(name: mapName, shape: finalShape, dtype: named, transposed: tr, produce: produce), .asNamed)
    }

    /// Anima's adapter, copied as is into a safetensors of its own, under the names that
    /// `AnimaConditioner` reads (`model.diffusion_model.llm_adapter.*`). A dtype the conditioner
    /// does not read (fp8, int8) goes to fp32, dequantized — never rounded to bf16.
    /// Returns the number of tensors written dequantized.
    @discardableResult
    static func writeAdapter(source: TensorSource, keys: [String], to path: String) throws -> Int {
        var pieces: [(name: String, dtype: String, shape: [Int], bytes: Data)] = []
        var dequantized = 0
        for key in keys {
            let name = "model.diffusion_model." + Recipes.withoutPrefix(key)
            let dtype = source.dtype(key)!, shape = source.shape(key)!
            if source.quantization(key) == nil, ["BF16", "F32", "F16"].contains(dtype), let (f, e) = source.raw(key) {
                pieces.append((name, dtype, shape, Data(bytes: f.pointer(key)!, count: e.bytes)))
            } else {
                let v = try source.read(key)
                pieces.append((name, "F32", shape, v.withUnsafeBytes { Data($0) }))
                dequantized += 1
            }
        }
        try writeSafetensors(pieces, to: path)
        return dequantized
    }

    /// A minimal safetensors: JSON header aligned to 8 bytes, then the data in order.
    package static func writeSafetensors(_ pieces: [(name: String, dtype: String, shape: [Int], bytes: Data)],
                                          to path: String) throws {
        var pairs: [OrderedJSON.Pair] = [], position = 0
        for m in pieces {
            pairs.append(.init(m.name, .object([.init("dtype", .string(m.dtype)),
                                                .init("shape", .list(m.shape.map { .integer($0) })),
                                                .init("data_offsets", .list([.integer(position), .integer(position + m.bytes.count)]))])))
            position += m.bytes.count
        }
        var json = Array(OrderedJSON.object(pairs).jsonText.utf8)
        while json.count % 8 != 0 { json.append(0x20) }
        var n = UInt64(json.count).littleEndian
        var data = withUnsafeBytes(of: &n) { Data($0) }
        data.append(contentsOf: json)
        for m in pieces { data.append(m.bytes) }
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }
}

extension TensorSource {
    /// The file and entry of a tensor, for a copy without conversion.
    package func raw(_ name: String) -> (Safetensors, Safetensors.Entry)? {
        for f in files { if let e = f.entries[name] { return (f, e) } }
        return nil
    }
}
