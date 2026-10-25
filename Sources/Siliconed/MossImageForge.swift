import Foundation

/// **MOSS-Image's maps, forged from the published files** — what the family's forge will be: the DiT, and
/// the DMD LoRA (`lodestones/MOSS-Image-SFT-8-steps-dmd`). The family's wiring (`Families`, `Installation`,
/// `Recipes`) is not here; this is the part that knows the names and the layouts, and it is what the
/// DiT's checks forge their maps with.
///
///   - **the DiT**: every tensor as published (bf16), the `Linear` weights written `[input, output]`
///     (`transposed`), the file in execution order. `rope.bands` is not a weight (the code derives it in fp32,
///     and the reference re-derives it on load: a rounded copy would be a defect), so it is left out. 462
///     tensors. Nothing is permuted, nothing folded: the RMSNorms are plain and the RoPE is half-split
///     (`MossImageRope`);
///   - **the LoRA**: kohya keys `diffusion_model.<module>.lora_down.weight` `[r, k]`, `.lora_up.weight` `[n, r]`,
///     `.alpha` (one per module, a scalar), written `<module>.down` `[k, r]` and `<module>.up` `[r, n]` with
///     `alpha / r` folded into `up` — **in fp32 as published** (the file is fp32 and the delta is smaller than
///     bf16's step: a bf16 map would discard most of the distillation). The strength stays the caller's.
package enum MossImageForge {
    /// The tensors of a published DiT that are not weights.
    static let notWeights: Set<String> = ["rope.bands"]

    /// The published DiT, in execution order: the timestep path, the embedders, the text blocks, the
    /// layers, the head.
    package static func order(_ names: [String]) -> [String] {
        func rank(_ name: String) -> (Int, Int) {
            func index(_ k: Int) -> Int { Int(name.split(separator: ".")[k]) ?? 0 }
            if name.hasPrefix("time_embedding.") { return (0, index(1)) }
            if name.hasPrefix("shared_block_modulation") { return (1, 0) }
            if name.hasPrefix("video_head_modulation") { return (2, 0) }
            if name.hasPrefix("video_embedder") { return (3, 0) }
            if name.hasPrefix("text_embedder.proj") { return (4, 0) }
            if name.hasPrefix("text_embedder.blocks.") { return (5, index(2)) }
            if name.hasPrefix("layers.") { return (6, index(1)) }
            return (7, 0)   // final_norm_video, final_linear_video
        }
        let keys = names.map { (rank($0), $0) }
        return keys.sorted { $0.0 != $1.0 ? $0.0 < $1.0 : $0.1 < $1.1 }.map(\.1)
    }

    /// The published tensors the map needs: `layers.N.*` for N < `layers`, the text blocks, the rest.
    package static func expectedCount(layers: Int) -> Int { 15 - notWeights.count + 10 * layers + 28 }

    /// **Forges the DiT's map.** `config`: the published `transformer/config.json`. `dtype`: bf16, as published,
    /// unless a check wants the source's own (fp32).
    @discardableResult
    package static func forgeDiT(source: TensorSource, config: OrderedJSON, to path: String,
                                 dtype: MapWriter.DType = .bfloat16, name: String? = nil) throws -> MapWriter.Tally {
        let names = order(source.names.filter { !notWeights.contains($0) })
        let layers = config["num_layers"]?.integer ?? 42
        guard names.count == expectedCount(layers: layers) else {
            throw Numerics.Failure(description: "\(names.count) DiT tensors, \(expectedCount(layers: layers)) expected for MOSS-Image")
        }
        var tensors: [MapWriter.Tensor] = []
        for tensorName in names {
            guard let shape = source.shape(tensorName) else {
                throw Numerics.Failure(description: "\(tensorName) absent from the source")
            }
            let linear = shape.count == 2 && tensorName.hasSuffix(".weight")
            tensors.append(.init(name: tensorName, shape: linear ? [shape[1], shape[0]] : shape, dtype: dtype,
                                 transposed: linear) {
                let values = try source.read(tensorName)
                guard linear else { return values }
                var moved = [Float](repeating: 0, count: values.count)
                values.withUnsafeBufferPointer { s in moved.withUnsafeMutableBufferPointer { d in
                    Numerics.transpose(s.baseAddress!, rows: shape[0], columns: shape[1], to: d.baseAddress!) } }
                return moved
            })
        }
        let parameters = tensors.reduce(0) { $0 + $1.shape.reduce(1, *) }
        return try MapWriter.write(to: path, tensors: tensors, header: { m in
            var h: [OrderedJSON.Pair] = [
                .init("format", 1), .init("kind", .string(MossImageDiT.kind)), .init("page", .integer(MapWriter.page)),
                .init("map_dtype", .string(dtype.rawValue)), .init("linear_weights_transposed", true),
                .init("rope_layout", "half-split"),
                .init("config", config.withoutPrivateKeys),
                .init("source", .object([.init("tensor_count", .integer(names.count)), .init("parameters", .integer(parameters))])),
                .init("weight_absmax", .real(Double(m?.absMax ?? 0))),
            ]
            if let name { h.append(.init("nom", .string(name))) }
            h.append(.init("order", .list(names.map { .string($0) })))
            return h
        })
    }

    /// The model identifier a LoRA map is checked against (`LoRA.checkTarget`).
    package static let model = "moss-image"

    /// **Forges a kohya LoRA's map** (`diffusion_model.` prefix, `lora_down`/`lora_up`/`alpha`), refusing
    /// anything else it finds: a module outside the DiT's `Linear`s that the engine adapts, a full delta, a
    /// shape that does not compose. `strict` modules are those `MossImageDiT.appliesLoRA` accepts.
    @discardableResult
    package static func forgeLoRA(source: TensorSource, config: MossImageDiT.Config, to path: String,
                                  name: String) throws -> MapWriter.Tally {
        let prefix = "diffusion_model."
        var pairs: [String: (down: String, up: String, alpha: String?)] = [:]
        for key in source.names {
            guard key.hasPrefix(prefix) else {
                throw Numerics.Failure(description: "\(key): not a kohya key (`diffusion_model.<module>.lora_down.weight`)")
            }
            let rest = String(key.dropFirst(prefix.count))
            if let module = rest.stripping(suffix: ".lora_down.weight") { pairs[module, default: ("", "", nil)].down = key }
            else if let module = rest.stripping(suffix: ".lora_up.weight") { pairs[module, default: ("", "", nil)].up = key }
            else if let module = rest.stripping(suffix: ".alpha") { pairs[module, default: ("", "", nil)].alpha = key }
            else { throw Numerics.Failure(description: "\(key): neither a down, an up nor an alpha — a LoRA of another kind") }
        }
        let shapes = Dictionary(uniqueKeysWithValues: MossImageDiT.loraTargets(config: config).map { ($0.name, ($0.k, $0.n)) })
        let modules = order(pairs.keys.map { $0 + ".weight" }).map { String($0.dropLast(".weight".count)) }
        var tensors: [MapWriter.Tensor] = []
        var ranks = Set<Int>()
        for module in modules {
            let pair = pairs[module]!
            guard !pair.down.isEmpty, !pair.up.isEmpty else {
                throw Numerics.Failure(description: "\(module): a down without its up, or the opposite")
            }
            guard MossImageDiT.appliesLoRA(module), let (k, n) = shapes[module],
                  let a = source.shape(pair.down), let b = source.shape(pair.up),
                  a.count == 2, b.count == 2, a[1] == k, b[0] == n, a[0] == b[1] else {
                throw Numerics.Failure(description: "\(module): not a module the engine adapts, or a shape that does not fit [k, n]")
            }
            let r = a[0]
            ranks.insert(r)
            // alpha / r, folded into `up`; no alpha: 1 (diffusers reads none, ComfyUI neither).
            let scale = try pair.alpha.flatMap { try source.scalar($0) }.map { $0 / Float(r) } ?? 1
            tensors.append(.init(name: module + ".down", shape: [k, r], dtype: .float32, transposed: nil) {
                let values = try source.read(pair.down)
                var moved = [Float](repeating: 0, count: values.count)
                values.withUnsafeBufferPointer { s in moved.withUnsafeMutableBufferPointer { d in
                    Numerics.transpose(s.baseAddress!, rows: r, columns: k, to: d.baseAddress!) } }
                return moved
            })
            tensors.append(.init(name: module + ".up", shape: [r, n], dtype: .float32, transposed: nil) {
                var values = try source.read(pair.up)
                if scale != 1 { for i in values.indices { values[i] *= scale } }
                var moved = [Float](repeating: 0, count: values.count)
                values.withUnsafeBufferPointer { s in moved.withUnsafeMutableBufferPointer { d in
                    Numerics.transpose(s.baseAddress!, rows: n, columns: r, to: d.baseAddress!) } }
                return moved
            })
        }
        let sorted = ranks.sorted()
        return try MapWriter.write(to: path, tensors: tensors, header: { _ in
            [.init("format", 1), .init("kind", "lora"), .init("page", .integer(MapWriter.page)),
             .init("map_dtype", "float32"), .init("linear_weights_transposed", true),
             .init("cible", .object([.init("modele", .string(model)), .init("kind", .string(MossImageDiT.kind))])),
             .init("nom", .string(name)),
             .init("rang", sorted.count == 1 ? .integer(sorted[0]) : .list(sorted.map { .integer($0) })),
             .init("echelle", .real(1)),
             .init("modules", .list(modules.map { .string($0) })),
             .init("order", .list(tensors.map { .string($0.name) }))]
        })
    }
}

private extension String {
    func stripping(suffix: String) -> String? { hasSuffix(suffix) ? String(dropLast(suffix.count)) : nil }
}
