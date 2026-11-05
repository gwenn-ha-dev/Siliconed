import Foundation

/// **A render said in A1111's "infotext"** — the `parameters` text chunk, the one format Civitai
/// reads from a PNG made off its site, written beside the app's own keys (`Engine.Render.metadata`).
///
///     woman posing in a library <lora:popart:0.70>                                   ← prompt + tags
///     Steps: 8, Sampler: Euler, CFG scale: 1, Seed: 1303787694, Size: 1152x896, Model: Z-Image Turbo,
///       Lora hashes: "popart: 9f86d081…(64 hex)", Civitai resources: [{"type":"checkpoint",
///       "modelVersionId":2442439}], Version: Siliconed 0.8                           ← one line
///
/// The reader is `@civitai/generation-metadata` (github.com/civitai/media-metadata, 0.4.0, the
/// version civitai.com pins): `image/parsers/automatic1111.ts` claims a PNG whose `parameters`
/// contains `Steps: `, the prompt is every line but the details line, the details line is
/// `Key: value` pairs split at commas. The resources are then credited in the database
/// (`get_image_resources.sql` in civitai/civitai) by **joining each hash to every hash Civitai
/// stores for a file, of every type at once** (AutoV2 = sha256[0:10], AutoV3, the whole SHA256…).
///
/// What the reader does that a writer by analogy would miss:
///
///   - **a `prompt` text chunk is taken for ComfyUI's graph** (`detectComfy`: `exif.prompt` is
///     enough), `JSON.parse` fails on plain text, and the image comes out with no metadata at all —
///     the app's own `prompt` key alone would do exactly that. The parsers run in order, A1111 first: with a
///     `parameters` holding `Steps: `, ComfyUI's is never tried;
///   - **a LoRA's weight comes only from a `<lora:name:weight>` tag in the prompt**; `Lora hashes`
///     gives the hash and no weight (stored as 1). The tag's name must match `[A-Za-z0-9_.-]+` and
///     its weight `[0-9.]+` — no minus sign: a negative strength gets no tag. The hash joins the tag
///     by **equal name**, hence one sanitized name for both (`name(_:)`);
///   - **the whole 64-hex sha256 is written**, not A1111's 10 or 12: it matches the `SHA256` row
///     every scanned file has, where a 12-hex `sha256` prefix matches nothing (Civitai's
///     `docs/image-resource-hash-matching.md`);
///   - **`Civitai resources: [{…}]`** names a model version directly, with no hash: the only way
///     for the base model, installed from Hugging Face in several files that no Civitai file
///     equals — and for a family's own LoRA (`civitaiRecipeLoRA`), which the user never imported.
///     Its regex is non-greedy up to the first `}]`, and wants `, ` before it;
///   - a value holding `,` `"` or a newline is JSON-quoted (A1111's `quote()`), or the pair
///     scanner splits it — but a quoted value that looks like `word: …` is read back as a nested
///     block (`KEY_VALUE_SHAPE` in `a1111-text.ts`): `Model: "Real: v2"` becomes `{Real: v2}`. A
///     colon in a free value is written `-`;
///   - **`CFG scale: 1` is always written**: the three offered families are distilled, run without
///     guidance, and a reader who reuses the settings (Civitai's "Remix") otherwise starts from its
///     own default, 3.5 or 7 — a scale that burns the image.
///
/// No date anywhere: two identical renders of one version stay two identical files.
package enum Infotext {
    /// A LoRA as the text names it.
    package struct LoRA: Equatable {
        package let name: String
        package let strength: Double
        package let sha256: String?
        package init(name: String, strength: Double, sha256: String?) {
            self.name = name; self.strength = strength; self.sha256 = sha256
        }
    }

    /// **The publisher's model on Civitai**: the version Civitai's official account hosts, checked
    /// through the public API (`/api/v1/model-versions/<id>`, 13/10/2026) — its DiT file is byte for
    /// byte Comfy-Org's single-file repackaging of the publisher's weights (same sha256 as on the
    /// Hub): Z Image Turbo 2442439 (`2407613050b8…`, `z_image_turbo_bf16`), Qwen Image 2.1 "v2.1
    /// Turbo" 3400828 (`8b74237a7f50…`, `qwen_image_2.1_turbo_bf16`; its int8 file `8144a0182964…`
    /// is the Compact's DiT), Anima "turbo-v1.1" 3263843 on circlestone_labs' own page
    /// (`fba11953276b…`, `anima-turbo-v1.1` of `circlestone-labs/Anima`, the Standard's DiT; the
    /// Compact is Bedovyy's int8 convrot of it, the Light vanes430's GGUF — checked 15/10 and 5/11/2026). Krea 2 Turbo
    /// "krea2_turbo_bf16" 3064584 (`78bbf8f4165e…`, the Standard's DiT; its int8 file `8e4eeda70dd5…`,
    /// the Compact's, is version 3091481 of the same page — checked 19/10/2026): Krea has no Civitai
    /// page and CivitaiOfficial's « Krea 2 » is the hosted API, so the page Civitai joins these files
    /// to is a user's upload of Comfy-Org's repository — the one every Krea 2 render there points at.
    /// MOSS-Image "SFT" 3409867 (CivitaiOfficial, checked 29/10/2026): Comfy-Org's int8 convrot DiT
    /// (`b4d1475bc7ee…`) and encoder (`32e3787472f7…`), the Compact's two files, are among its files.
    package static func civitaiModelVersion(model: String) -> Int? {
        switch model {
        case "z-image": return 2_442_439
        case "qwen-image-2.1": return 3_400_828
        case "anima": return 3_263_843
        case "krea2": return 3_064_584
        case "moss-image": return 3_409_867
        default: return nil
        }
    }

    /// **The LoRA a page's recipe runs under**, credited beside it at strength 1: MOSS-Image's 8 steps are
    /// lodestones' DMD LoRA (`Installation.companion`), byte for byte tsolful's "Moss_SFT_8_steps_dmd_v0.1"
    /// 3411385 (`4abb14e39c1a…`, checked 29/10/2026). Without it, a Remix runs the SFT alone at 8 steps
    /// and CFG 1, and gets mush.
    package static func civitaiRecipeLoRA(checkpoint: Int) -> Int? { checkpoint == 3_409_867 ? 3_411_385 : nil }

    /// A1111's sampler names, as Civitai lists them.
    package static func sampler(_ solver: Solver) -> String {
        switch solver {
        case .euler: return "Euler"
        case .eulerAncestral: return "Euler a"
        case .dpmpp2m: return "DPM++ 2M"
        case .unipc: return "UniPC"
        }
    }

    /// **The app's version** (`CFBundleShortVersionString`), only inside the app's own bundle: the
    /// CLI and the test runner have none of theirs to give, and say `Siliconed` alone.
    package static let appVersion: String? = {
        guard Bundle.main.bundleIdentifier == "dev.gwenn-ha.siliconed" else { return nil }
        return Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
    }()

    /// **The text.** `fasterRendering`: the half-size start was in play. `strength`: the img2img's, `nil` in txt2img. `civitaiModelVersion`: the base
    /// model's page, `nil` for an imported fine-tune or a family Civitai is not checked for.
    package static func text(prompt: String, steps: Int, sampler solver: Solver, seed: UInt64,
                             width: Int, height: Int, model: String, strength: Double?, loras: [LoRA],
                             civitaiModelVersion: Int?, version: String?, fasterRendering: Bool = false) -> String {
        let named = loras.map { (name: name($0.name), lora: $0) }
        let tags = named.filter { $0.lora.strength >= 0 && $0.lora.strength.isFinite }
            .map { "<lora:\($0.name):\(String(format: "%.2f", $0.lora.strength))>" }
        var details = ["Steps: \(steps)", "Sampler: \(sampler(solver))", "CFG scale: 1", "Seed: \(seed)",
                       "Size: \(width)x\(height)", "Model: \(value(model))"]
        if let strength { details.append("Denoising strength: \(String(format: "%.2f", strength))") }
        let hashes = named.compactMap { n in n.lora.sha256.map { "\(n.name): \($0.lowercased())" } }
        if !hashes.isEmpty { details.append("Lora hashes: \"\(hashes.joined(separator: ", "))\"") }
        if let id = civitaiModelVersion {
            var resources = ["{\"type\":\"checkpoint\",\"modelVersionId\":\(id)}"]
            if let lora = civitaiRecipeLoRA(checkpoint: id) {
                resources.append("{\"type\":\"lora\",\"weight\":1,\"modelVersionId\":\(lora)}")
            }
            details.append("Civitai resources: [\(resources.joined(separator: ","))]")
        }
        // The half-size start moves the image: a reader who redoes the settings needs it. Civitai keeps unknown pairs.
        if fasterRendering { details.append("Faster rendering: true") }
        details.append("Version: " + value(version.map { "Siliconed \($0)" } ?? "Siliconed"))
        return ([prompt] + tags).joined(separator: " ") + "\n" + details.joined(separator: ", ")
    }

    /// A LoRA's name reduced to what the `<lora:…>` tag accepts: `[A-Za-z0-9_.-]`, anything else `_`.
    package static func name(_ raw: String) -> String {
        let kept = String(raw.unicodeScalars.map { s -> Character in
            (s.isASCII && (CharacterSet.alphanumerics.contains(s) || "_.-".unicodeScalars.contains(s))) ? Character(s) : "_"
        })
        return kept.isEmpty ? "lora" : kept
    }

    /// A free value (a model's name, the version): its colons `-`, then `quote`d.
    package static func value(_ raw: String) -> String { quote(raw.replacingOccurrences(of: ":", with: "-")) }

    /// A1111's `quote()`: JSON-quoted when the value would break the pair scanner.
    package static func quote(_ value: String) -> String {
        guard value.contains(where: { ",:\"\n".contains($0) }) else { return value }
        var out = "\""
        for s in value.unicodeScalars {
            switch s {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case _ where s.value < 0x20: out += String(format: "\\u%04x", s.value)
            default: out.unicodeScalars.append(s)
            }
        }
        return out + "\""
    }

    /// 64 hexadecimal digits — what a map's `source.sha256` must be to be written.
    package static func isSHA256(_ s: String) -> Bool {
        s.utf8.count == 64 && s.allSatisfy(\.isHexDigit)
    }
}
