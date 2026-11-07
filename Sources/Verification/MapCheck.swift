import Foundation
import Siliconed

// **Which map a check judges, and whether it is the one the golden was made for.** The goldens come
// from the reference's published weights; a map forged from other weights (a Compact's int8, a
// Light's SQ 4- and 5-bit tensors) rounds differently by construction, and judged against that golden it fails
// for what it is, not for a fault. So every stage names its map in its first line and a map of
// another variant than its golden's is skipped, with what to do.
//
//     map ~/Library/Application Support/Siliconed/store/anima-turbo-dit.v0.silicon · 3.9 GB · bfloat16 (standard weights)

/// The kind of weights a map holds, read from its tensor table alone.
package enum MapVariant: String, Sendable {
    /// bf16, fp16 or fp32 tensors only: the weights as published (the Standard).
    case standard
    /// At least one 8-bit tensor (`int8`, `float8_e4m3`) and no GGUF block.
    case eightBit = "8-bit"
    /// At least one packed tensor: our SQ 4- or 5-bit (`sq4`, `sq5`: the Light), or a GGUF block (`q4_0` … `q6_k`: a Light until 0.21,
    /// still readable by the forge).
    case packed

    /// The variant of a map whose tensors have these dtypes (`Artifact.DType` raw values).
    package static func of(dtypes: Set<String>) -> MapVariant {
        if dtypes.contains(where: { Artifact.DType(rawValue: $0).map { $0.isPacked || $0.sqBits != nil } == true }) { return .packed }
        if dtypes.contains(where: { Artifact.DType(rawValue: $0)?.isQuantized == true }) { return .eightBit }
        return .standard
    }
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

    /// Reads the header alone (no mapping of the tensors).
    package init(path: String) throws {
        let header = try Artifact.header(path)
        let size = Library.fileSize(path) ?? 0      // through a link: a linked map weighs its file
        var counts: [String: Int] = [:]
        for entry in (header["tensors"] as? [String: [String: Any]] ?? [:]).values {
            if let dtype = entry["dtype"] as? String { counts[dtype, default: 0] += 1 }
        }
        self.init(path: path, bytes: size, dtypes: counts)
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
