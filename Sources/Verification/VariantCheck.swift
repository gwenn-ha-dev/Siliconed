import Foundation
import Siliconed

// **The Compact and the Light, judged from outside.** `siliconed check <model> dit --variant compact|light`
// runs the engine on the map of that version (`Family.ditMap(_:)`, or `--map`) against a golden the oracle
// wrote from the same inputs as the Standard's: the reference DiT in float32, with the weights the third
// party published (the Compact's 8-bit) dequantized by its own library, or our own SQ file (the Light's 4 and 5 bits)
// decoded by the oracle's reference decoder of the published format (`tools/oracle/quantized.py`). The
// goldens are `goldens-…-<variant>-<size>` beside the Standard's, the bf16 bar `witness-bf16-…-<variant>-<size>`.
//
// A map is of a version when its tensor table says so (`MapSummary.quantizedVersion`): the Compact holds
// 8-bit tensors (a GGUF Q8_0 is kept as int8 with its block scales) and no packed one, the Light at least
// one (`sq4`, `sq5`; a GGUF block before 0.21). `MapVariant` already separates them (`.eightBit`, `.packed`); the version is the name the
// golden carries, so that a Light is never judged against a Compact's golden, nor the reverse.

extension MapSummary {
    /// The version this map is: `nil` for the Standard's weights.
    package var quantizedVersion: Variant? {
        switch variant {
        case .packed: return .light
        case .eightBit: return .compact
        case .standard: return nil
        }
    }

    /// Why this map cannot be judged against the golden of `version`; `nil` when it can.
    package func mismatch(version: Variant) -> String? {
        guard quantizedVersion != version else { return nil }
        let held = quantizedVersion?.rawValue ?? "Standard"
        return "this map holds the \(held) version's weights, the golden was made for the \(version.rawValue)'s — "
            + "use the \(version.rawValue) map (drop --map) or the golden of the version you judge"
    }
}

/// The golden's file name for a version: `goldens-dit-512` → `goldens-dit-compact-512`, the variant before
/// the size, as `quantized.stem` writes it. The Standard (`nil`) keeps the name.
package func variantStem(_ stem: String, _ version: Variant?) -> String {
    guard let version, let cut = stem.lastIndex(of: "-") else { return stem }
    return String(stem[..<cut]) + "-\(version.rawValue)" + String(stem[cut...])
}

extension Invocation {
    /// `--variant compact|light`; `nil` for the Standard.
    package func quantizedVariant() throws -> Variant? {
        guard let text = extras["--variant"] else { return nil }
        guard let version = Variant(rawValue: text), version != .standard else {
            throw UsageError("--variant \(text): compact or light (the Standard is the default)")
        }
        return version
    }

    /// The DiT map of the version under test: `--map`, else the installed map of `family`'s `version`
    /// (named apart from the Standard's, so that it can sit beside it). It is printed, and refused when it
    /// is not that version's. Without a version, `ditMap(_:)` as it stands.
    package func ditMap(_ standard: String, family: Family, version: Variant?) throws -> String {
        guard let version else { return try ditMap(standard) }
        let path = map ?? Library.processWide.map(family.ditMap(version))
        guard FileManager.default.fileExists(atPath: path) else { throw Skip("map missing: \(path) — siliconed install") }
        let summary = try MapSummary(path: path)
        print(summary.line)
        if let why = summary.mismatch(version: version) { throw Skip(why) }
        return path
    }
}
