import Foundation

/// **How a module's weights are stored on this Mac**, read from what is installed — the dtypes of the map's matrices, weighted by
/// their values — and never deduced from the name of the version: the Compact of one family is Comfy-Org's int8 convrot, another's is
/// a GGUF Q8_0 whose encoder stays the Standard's. It is the **storage**; the products are computed in fp32 (or the Neural Engine's
/// fp16, where the plan gives it rows), which the DiT's other label says.
///
/// A map keeps a few tensors in another form (norms, embeddings, a modulation in bf16): the format is the one that holds most of the
/// values of the matrices, and the label is its exact name (`int8`, `Q8_0`, `Q4_K_M`, `bf16`…), the same in every language.
package enum WeightFormat: String, Codable, Sendable, CaseIterable {
    case int8, q8_0, fp8, q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q4_k_m, q5_k_m, q6_k, bf16, fp16, fp32
    /// Siliconed's own 4-bit weights (4, 5 and 8 bits by layer, `SQ`), decoded to int8 at each product.
    case sq

    /// The exact name, verbatim.
    package var label: String {
        switch self {
        case .int8: "int8"
        case .bf16: "bf16"
        case .fp16: "fp16"
        case .fp32: "fp32"
        case .fp8: "fp8"
        default: rawValue.uppercased()
        }
    }

    /// A packed or 8-bit form: the weights are read as published, smaller than the bf16 they would be.
    package var isQuantized: Bool { ![.bf16, .fp16, .fp32].contains(self) }

    // ── reading ──

    /// **The format of a Siliconed map's header** (`Artifact.header`): its `tensors` table. A scaled int8 with a scale per block is
    /// GGUF's Q8_0; one with a scale per column (and a rotation, for ComfyUI's convrot) is plain int8. Packed K-quants mixed with
    /// Q6_K are the `_M` mixtures. `nil` for a header with no matrix.
    package static func read(mapHeader header: [String: Any]) -> WeightFormat? {
        guard let table = header["tensors"] as? [String: [String: Any]] else { return nil }
        var values: [String: Int] = [:]
        for (_, entry) in table {
            guard let shape = entry["shape"] as? [Int], shape.count >= 2, let dtype = entry["dtype"] as? String,
                  let count = Artifact.elementCount(shape) else { continue }
            var key = dtype
            if dtype == "sq4" || dtype == "sq5" { key = "sq" }
            if dtype == "int8", let scale = entry["scale"] as? [String: Any], scale["block"] != nil { key = "q8_0" }
            values[key, default: 0] += count
        }
        return decide(values)
    }

    /// **The format of a safetensors file's header** (`F32`, `F16`, `BF16`, `F8_E4M3`, `I8`): the VAE's.
    package static func read(safetensorsHeader header: [String: Any]) -> WeightFormat? {
        var values: [String: Int] = [:]
        for (name, raw) in header where name != "__metadata__" {
            guard let entry = raw as? [String: Any], let shape = entry["shape"] as? [Int], shape.count >= 2,
                  let dtype = entry["dtype"] as? String, let count = Artifact.elementCount(shape) else { continue }
            let key = switch dtype {
            case "BF16": "bfloat16"
            case "F16": "float16"
            case "F32": "float32"
            case "F8_E4M3": "float8_e4m3"
            case "I8": "int8"
            default: dtype
            }
            values[key, default: 0] += count
        }
        return decide(values)
    }

    /// The class holding most of the values wins; ties go to the smaller form, so that the result does not depend on the dictionary's order.
    private static func decide(_ values: [String: Int]) -> WeightFormat? {
        guard let top = values.max(by: { ($0.value, $1.key) < ($1.value, $0.key) })?.key else { return nil }
        switch top {
        case "int8": return .int8
        case "q8_0": return .q8_0
        case "sq": return .sq
        case "float8_e4m3": return .fp8
        case "bfloat16": return .bf16
        case "float16": return .fp16
        case "float32": return .fp32
        case "q4_k": return values["q6_k"] != nil ? .q4_k_m : .q4_k
        case "q5_k": return values["q6_k"] != nil ? .q5_k_m : .q5_k
        default: return WeightFormat(rawValue: top)
        }
    }

    /// The format of a map on disk; `nil` when unreadable.
    package static func read(mapAt path: String) -> WeightFormat? {
        (try? Artifact.header(path)).flatMap { read(mapHeader: $0) }
    }

    /// The format of a safetensors file on disk (its header alone: 8 bytes of length, then JSON).
    package static func read(safetensorsAt path: String) -> WeightFormat? {
        guard let file = FileHandle(forReadingAtPath: path) else { return nil }
        defer { try? file.close() }
        guard let prologue = try? file.read(upToCount: 8), prologue.count == 8 else { return nil }
        let length = prologue.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * UInt64($1.offset)) }
        guard length > 0, length < 64 << 20, let json = try? file.read(upToCount: Int(length)), json.count == Int(length),
              let object = (try? JSONSerialization.jsonObject(with: json)) as? [String: Any] else { return nil }
        return read(safetensorsHeader: object)
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

    /// Read from what `library` has installed for `card`: its DiT map, its encoder's, the family's VAE.
    package static func read(card: ModelCard, in library: Library) -> WeightFormats {
        func map(_ name: String) -> String { name.hasPrefix("/") ? name : library.map(name) }
        let dit = card.map ?? library.ditMap(card.family)
        return WeightFormats(
            dit: WeightFormat.read(mapAt: map(dit)),
            text: WeightFormat.read(mapAt: map(library.encoderMap(card.family))),
            vae: (try? library.component(card.family, "vae.safetensors")).flatMap { WeightFormat.read(safetensorsAt: $0) })
    }
}
