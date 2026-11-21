import Foundation
import Siliconed

// **Which file a check judges, and whether it is the one the golden was made for.** The Standard's goldens come
// from the reference's bf16 weights; a file of other weights (a Compact's int8) rounds differently by construction, and
// judged against that golden it fails for what it is, not for a fault. So every stage names its file in its first line
// and a file of another variant than its golden's is skipped, with what to do.
//
//     map ~/Library/Application Support/Siliconed/store/anima-turbo-v1.1.safetensors · 3.9 GB · bfloat16 (standard weights)

/// The kind of weights a file holds, read from its tensor table alone.
package enum MapVariant: String, Sendable {
    /// bf16, fp16 or fp32 tensors only: the weights as published (the Standard).
    case standard
    /// At least one int8 tensor (the Compact).
    case eightBit = "8-bit"

    /// The variant of a file whose tensors have these dtypes.
    package static func of(dtypes: Set<String>) -> MapVariant { dtypes.contains("int8") ? .eightBit : .standard }
}

/// What the header says of a map.
package struct MapSummary: Sendable {
    package let path: String
    package let bytes: Int
    /// Tensor count by dtype.
    package let dtypes: [String: Int]
    package var variant: MapVariant { .of(dtypes: Set(dtypes.keys)) }

    package init(path: String, bytes: Int, dtypes: [String: Int]) {
        (self.path, self.bytes, self.dtypes) = (path, bytes, dtypes)
    }

    /// Reads the table alone: a publisher's file or shards — its weights counted by type, what is not a weight (`U8`: a
    /// quantization's note) left out.
    package init(path: String) throws {
        var counts: [String: Int] = [:]
        let names = ["BF16": "bfloat16", "F16": "float16", "F32": "float32", "I8": "int8"]
        let files = MapWeights.files(path).filter { !$0.hasSuffix(".index.json") }
        for file in files {
            guard FileManager.default.fileExists(atPath: file) else { throw MissingFile(file, .published) }
            guard let header = Safetensors.header(file) else { throw Safetensors.Failure.badHeader("\(file): no header the format allows") }
            for (key, value) in header where key != "__metadata__" {
                if let dtype = (value as? [String: Any])?["dtype"] as? String, let name = names[dtype] { counts[name, default: 0] += 1 }
            }
        }
        self.init(path: path, bytes: MapWeights.files(path).reduce(0) { $0 + (Library.fileSize($1) ?? 0) }, dtypes: counts)
    }

    /// The line a check prints before it judges: path, size, the weights' types.
    package var line: String {
        let types = dtypes.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
            .map { dtypes.count == 1 ? $0.key : "\($0.key) ×\($0.value)" }.joined(separator: ", ")
        return "map \(Library.withoutHome(path)) · " + String(format: "%.1f GB", Double(bytes) / 1e9)
            + " · \(types.isEmpty ? "no tensor table" : types) (\(variant.rawValue) weights)"
    }

    /// Why this map cannot be judged against a golden made for `golden`; `nil` when it can. The one
    /// place that decides it — a golden of another variant (a Compact's) passes its own here.
    package func mismatch(golden: MapVariant) -> String? {
        guard variant != golden else { return nil }
        return "this map holds \(variant.rawValue) weights, the golden was made for \(golden.rawValue) ones — "
            + "its error against the reference would be the quantization's, not a fault; "
            + (golden == .standard ? "judge the installed Standard map (drop --map)" : "use the golden of this variant")
    }
}

package enum MapCheck {
    /// Prints the map's line and refuses it (`Skip`) when it is not the golden's variant.
    @discardableResult
    package static func use(_ path: String, golden: MapVariant = .standard) throws -> String {
        let summary = try MapSummary(path: path)
        print(summary.line)
        if let why = summary.mismatch(golden: golden) { throw Skip(why) }
        return path
    }
}
