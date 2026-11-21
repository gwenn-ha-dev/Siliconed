import Foundation

/// **How a module's weights are stored on this Mac**, read from what is installed — the dtypes of the file's matrices, weighted by
/// their values — and never deduced from the name of the version. It is the **storage**: what the products compute in is the
/// family's graph.
///
/// A file keeps a few tensors in another form (norms, embeddings, a modulation in bf16): the format is the one that holds most of the
/// values of the matrices, and the label is its exact name (`int8`, `bf16`…), the same in every language.
package enum WeightFormat: String, Codable, Sendable, CaseIterable {
    case int8, fp8, bf16, fp16, fp32

    /// The exact name, verbatim.
    package var label: String { rawValue }

    // ── reading ──

    /// **The format of safetensors headers** (`F32`, `F16`, `BF16`, `F8_E4M3`, `I8`), a file's or its shards' together. `nil` for
    /// headers with no matrix.
    package static func read(safetensorsHeaders headers: [[String: Any]]) -> WeightFormat? {
        var values: [WeightFormat: Int] = [:]
        let formats: [String: WeightFormat] = ["BF16": .bf16, "F16": .fp16, "F32": .fp32, "F8_E4M3": .fp8, "I8": .int8]
        for header in headers {
            for (name, raw) in header where name != "__metadata__" {
                guard let entry = raw as? [String: Any], let shape = entry["shape"] as? [Int], shape.count >= 2,
                      let format = (entry["dtype"] as? String).flatMap({ formats[$0] }), let count = Safetensors.byteCount(shape, width: 1) else { continue }
                values[format, default: 0] += count
            }
        }
        // The class holding most of the values wins; ties go to the smaller form, so that the result does not depend on the dictionary's order.
        return values.max { ($0.value, $1.key.rawValue) < ($1.value, $0.key.rawValue) }?.key
    }

    /// The format of a published file on disk — a safetensors, or the shards behind an index (`MapWeights.files`) — by its
    /// headers alone (8 bytes of length, then JSON); `nil` when unreadable.
    package static func read(at path: String) -> WeightFormat? {
        let headers = MapWeights.files(path).filter { !$0.hasSuffix(".index.json") }.compactMap(Safetensors.header)
        return headers.isEmpty ? nil : read(safetensorsHeaders: headers)
    }
}

/// **The storage of each module of a model**: the DiT, the text encoder, and the VAE — whose file also holds the image encoder.
package struct WeightFormats: Codable, Sendable, Equatable {
    package var dit: WeightFormat?
    package var text: WeightFormat?
    package var vae: WeightFormat?
    package init(dit: WeightFormat? = nil, text: WeightFormat? = nil, vae: WeightFormat? = nil) {
        self.dit = dit; self.text = text; self.vae = vae
    }

    /// Read from what `library` has installed for `card`: its DiT, its encoder, the family's VAE.
    package static func read(card: ModelCard, in library: Library) -> WeightFormats {
        func map(_ name: String) -> String { name.hasPrefix("/") ? name : library.map(name) }
        let dit = card.map ?? library.ditMap(card.family)
        return WeightFormats(
            dit: WeightFormat.read(at: map(dit)),
            text: WeightFormat.read(at: map(library.encoderMap(card.family))),
            vae: (try? library.component(card.family, "vae.safetensors")).flatMap { WeightFormat.read(at: $0) })
    }
}
