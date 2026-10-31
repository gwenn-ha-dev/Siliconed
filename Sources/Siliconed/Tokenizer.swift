import Foundation

/// The byte-level BPE of the Qwen2/Qwen3 family, read from a published `tokenizer.json` — and the
/// few variants the other families publish in the same format.
///
/// The Z-Image Turbo encoder is a Qwen3-4B; Anima's is a Qwen3-0.6B, Krea 2's a Qwen3-VL-4B,
/// Qwen-Image-2.1's a Qwen3-VL-8B: the same tokenizer, nothing beyond the supported families.
///
/// This is **not** a general `tokenizer.json` interpreter: every field it does not reproduce
/// is checked and **refused**, not ignored — the same discipline the artifact applies to tensor
/// names. A tokenizer that silently falls back to a neighboring behavior produces
/// tokens, and the image that comes out is simply the other one.
///
/// **Ported from a donor project's Qwen tokenizer** (since retired), which had sixty-five tests
/// behind it, including a fuzz corpus. The logic was not touched; what was added is
/// what the Z-Image pipeline asks for and the donor's version did not do: **fixed-length
/// padding** and the **mask**.
///
/// > **A forge lead, noted and not taken.** Reading 7 MB of JSON and building two tables of
/// > 151 k entries at every startup is exactly what the forge exists to remove:
/// > the vocabulary is an immutable table that should live in the artifact, mapped. To do
/// > when the path is right, not before — "the dumb path" first. Meanwhile the two tables are
/// > read from the mapped file into `Pages` (`ByteTable`), not into `malloc`.
package final class Tokenizer {

    package enum Failure: Error, CustomStringConvertible {
        case fileNotFound(String)
        case malformed(String)
        /// A field of `tokenizer.json` that this port does not reproduce.
        case unsupported(String, found: String)

        package var description: String {
            switch self {
            case .fileNotFound(let p): return "tokenizer.json not found: \(p)"
            case .malformed(let m): return "tokenizer.json malformed: \(m)"
            case .unsupported(let field, let found):
                return "tokenizer.json carries \(field) = \(found), which this port does not reproduce"
            }
        }
    }

    /// The 256 byte symbols of the GPT-2 alphabet, as one-scalar strings.
    ///
    /// Printable ASCII and most of Latin-1 represent themselves; the remaining 68 bytes
    /// are lifted from U+0100 on, in byte order. This range contains
    /// only precomposed letters — **no combining marks** — so a symbol never merges
    /// with its neighbor into a single grapheme, and the BPE can treat the sequence as an array.
    static let byteSymbols: [String] = {
        var symbols = [String](repeating: "", count: 256)
        var direct = Set<Int>()
        for range in [33...126, 161...172, 174...255] {
            for b in range { symbols[b] = String(UnicodeScalar(UInt32(b))!); direct.insert(b) }
        }
        var lifted = 0
        for b in 0..<256 where !direct.contains(b) {
            symbols[b] = String(UnicodeScalar(UInt32(256 + lifted))!)
            lifted += 1
        }
        return symbols
    }()

    /// `symbol → id` and `left \0 right → rank`, in `Pages` (`ByteTable`).
    private let vocabulary: ByteTable
    private let ranks: ByteTable
    private let splitter: NSRegularExpression
    /// The added tokens, indexed by first **scalar**, longest first.
    ///
    /// Scalars and not `Character`, and the nuance is not cosmetic: a `Character` is a
    /// grapheme cluster, so `<think>` followed by a variation selector ends up in the cluster
    /// `>️` and `hasPrefix("<think>")` becomes false. The donor's fuzz caught exactly that, and
    /// no deliberate case would have.
    private let addedByFirst: [Unicode.Scalar: [(scalars: [Unicode.Scalar], id: Int)]]
    private let addedIdentifiers: [Int: String]
    /// The added tokens the file marks `special` (`<|im_end|>`, `<|endoftext|>`, `<|image_pad|>`…): what
    /// `skip_special_tokens` leaves out of a generated text. `<think>` and `<tool_call>` are added, not special.
    package let specialIdentifiers: Set<Int>

    private let normalizeNFC: Bool

    // MARK: - Reading

    package convenience init(directory: String) throws {
        try self.init(file: directory + "/tokenizer.json")
    }

    package init(file path: String) throws {
        guard FileManager.default.fileExists(atPath: path) else { throw Failure.fileNotFound(path) }
        // Mapped, never read into `malloc`; `model.vocab` and `model.merges` are read from these bytes
        // into `ByteTable`s, and `JSONSerialization` parses the rest with those two values emptied
        // (`{}`, `[]`): its object graph and two 151 k-entry dictionaries are what the text phase
        // left in the footprint.
        let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .alwaysMapped)
        let (rootValue, vocabulary, ranks) = try data.withUnsafeBytes { raw throws -> (Any, ByteTable, ByteTable) in
            let json = TokenizerJSON(bytes: raw)
            do {
                guard let model = try json.members(of: 0, ["model"])["model"],
                      raw[model.lowerBound] == 0x7B else { throw Failure.malformed("no model") }
                let tables = try json.members(of: model.lowerBound, ["vocab", "merges"])
                guard let vocab = tables["vocab"], raw[vocab.lowerBound] == 0x7B else { throw Failure.malformed("no model.vocab") }
                guard let merges = tables["merges"], raw[merges.lowerBound] == 0x5B else { throw Failure.malformed("no model.merges") }
                let (vocabulary, ranks) = try Self.tables(json, vocab: vocab, merges: merges)
                // The document without the two tables, in pages of its own for the parse.
                let cuts = [(vocab, Array("{}".utf8)), (merges, Array("[]".utf8))].sorted { $0.0.lowerBound < $1.0.lowerBound }
                let stripped = Pages(bytes: raw.count)
                var n = 0, from = 0
                for (range, empty) in cuts + [(raw.count..<raw.count, [])] {
                    (stripped.base + n).copyMemory(from: raw.baseAddress! + from, byteCount: range.lowerBound - from)
                    n += range.lowerBound - from
                    for b in empty { stripped.base.storeBytes(of: b, toByteOffset: n, as: UInt8.self); n += 1 }
                    from = range.upperBound
                }
                let root = try withExtendedLifetime(stripped) {
                    try JSONSerialization.jsonObject(with: Data(bytesNoCopy: stripped.base, count: n, deallocator: .none))
                }
                return (root, vocabulary, ranks)
            } catch let error as TokenizerJSON.Malformed {
                throw Failure.malformed("not JSON at byte \(error.at)")
            }
        }
        guard let root = rootValue as? [String: Any] else {
            throw Failure.malformed("the top level is not an object")
        }

        // What we reproduce, field by field. **The regular expression comes from the file**: a
        // hard-coded copy would drift the day a neighboring model published another, and without
        // saying anything.
        // NFC, or no normalization: `null`.
        let normalizer = root["normalizer"]
        if normalizer is NSNull || normalizer == nil {
            normalizeNFC = false
        } else {
            try Self.require((normalizer as? [String: Any])?["type"] as? String, equals: "NFC", field: "normalizer.type")
            normalizeNFC = true
        }
        guard let pre = root["pre_tokenizer"] as? [String: Any],
              pre["type"] as? String == "Sequence",
              let stages = pre["pretokenizers"] as? [[String: Any]], stages.count == 2 else {
            throw Failure.unsupported("pre_tokenizer",
                found: "\((root["pre_tokenizer"] as? [String: Any])?["type"] ?? "absent")")
        }
        try Self.require(stages[0]["type"] as? String, equals: "Split", field: "pre_tokenizer[0]")
        try Self.require(stages[0]["behavior"] as? String, equals: "Isolated",
                         field: "pre_tokenizer[0].behavior")
        try Self.require(String(describing: stages[0]["invert"] ?? "absent"), equals: "0",
                         field: "pre_tokenizer[0].invert")
        guard let pattern = (stages[0]["pattern"] as? [String: Any])?["Regex"] as? String else {
            throw Failure.unsupported("pre_tokenizer[0].pattern", found: "not a Regex")
        }
        try Self.requireByteLevel(stages[1], field: "pre_tokenizer[1]")
        // The decoder plays no part in encoding, and `tokenizers`' `ByteLevel` ignores
        // `add_prefix_space` when decoding: Anima's Qwen3-0.6B publishes it as `true`, and it is inert.
        try Self.require((root["decoder"] as? [String: Any])?["type"] as? String,
                         equals: "ByteLevel", field: "decoder.type")
        // Post-processing: `ByteLevel` (Z-Image), or a template that adds NOTHING (Anima,
        // `transformers` 5: `single = [A]`, no special token). Any other template would add
        // what this port does not place.
        let post = root["post_processor"] as? [String: Any]
        if post?["type"] as? String == "TemplateProcessing" {
            let single = post?["single"] as? [[String: Any]] ?? []
            guard single.count == 1, single[0]["Sequence"] != nil else {
                throw Failure.unsupported("post_processor", found: "a template that adds tokens")
            }
        } else {
            try Self.requireByteLevel(post, field: "post_processor")
        }

        guard let model = root["model"] as? [String: Any] else { throw Failure.malformed("no model") }
        try Self.require(model["type"] as? String, equals: "BPE", field: "model.type")
        // Each of these fields changes the meaning of the merge loop below: none is assumed.
        for (field, expected) in [("dropout", "absent"), ("unk_token", "absent"),
                                  ("continuing_subword_prefix", ""), ("end_of_word_suffix", "")] {
            // An empty suffix or prefix and a `null` field say the same thing.
            let value = model[field].flatMap { $0 is NSNull ? nil : String(describing: $0) }
            let loaded = value ?? (expected.isEmpty ? "" : "absent")
            try Self.require(loaded, equals: expected, field: "model.\(field)")
        }
        for field in ["fuse_unk", "byte_fallback"] {
            try Self.require(String(describing: model[field] ?? 0), equals: "0", field: "model.\(field)")
        }
        // `ignore_merges` (a piece already in the vocabulary taken without merging) is not ported.
        try Self.require(String(describing: model["ignore_merges"] ?? 0), equals: "0", field: "model.ignore_merges")


        var byFirst: [Unicode.Scalar: [(scalars: [Unicode.Scalar], id: Int)]] = [:]
        var identifiers: [Int: String] = [:]
        var specials = Set<Int>()
        for entry in (root["added_tokens"] as? [[String: Any]] ?? []) {
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int,
                  let first = content.unicodeScalars.first else {
                throw Failure.malformed("added_tokens without content or id")
            }
            // `lstrip`/`rstrip` would eat the surrounding space, `single_word` would refuse a
            // match inside a word, `normalized` would run the content through NFC
            // first. Qwen sets none; a file that did would require three more
            // behaviors here, so we refuse rather than ignore.
            for flag in ["lstrip", "rstrip", "single_word", "normalized"] {
                try Self.require(String(describing: entry[flag] ?? 0), equals: "0",
                                 field: "added_tokens[\(content)].\(flag)")
            }
            byFirst[first, default: []].append((Array(content.unicodeScalars), id))
            identifiers[id] = content
            if entry["special"] as? Bool == true { specials.insert(id) }
        }
        for key in byFirst.keys { byFirst[key]?.sort { $0.scalars.count > $1.scalars.count } }

        self.vocabulary = vocabulary
        self.ranks = ranks
        self.tableBytes = vocabulary.mapped + ranks.mapped
        self.splitter = try NSRegularExpression(pattern: pattern)
        self.addedByFirst = byFirst
        self.addedIdentifiers = identifiers
        self.specialIdentifiers = specials
    }

    /// What the two tables map, from the file's counts — released with the tokenizer.
    package let tableBytes: Int

    /// **`tableBytes` of the tokenizer `file` would build, without building it**: `model.vocab`'s and
    /// `model.merges`' entries counted and their spans measured in the mapped file, as `init` sizes its
    /// two `ByteTable`s (`ByteTable.mappedBytes`) — what the memory need charges while a tokenizer lives
    /// (`QwenImage21Maps`). A scan of the file, no table, no `JSONSerialization`.
    package static func tableBytes(file path: String) throws -> Int {
        guard FileManager.default.fileExists(atPath: path) else { throw Failure.fileNotFound(path) }
        let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .alwaysMapped)
        return try data.withUnsafeBytes { raw throws -> Int in
            let json = TokenizerJSON(bytes: raw)
            do {
                guard let model = try json.members(of: 0, ["model"])["model"],
                      raw[model.lowerBound] == 0x7B else { throw Failure.malformed("no model") }
                let tables = try json.members(of: model.lowerBound, ["vocab", "merges"])
                guard let vocab = tables["vocab"], let merges = tables["merges"] else { throw Failure.malformed("no model.vocab") }
                var vocabulary = 0, ranks = 0
                try json.elements(vocab) { _ in vocabulary += 1 }
                try json.elements(merges) { _ in ranks += 1 }
                return ByteTable.mappedBytes(entries: vocabulary, keyBytes: vocab.count)
                    + ByteTable.mappedBytes(entries: ranks, keyBytes: merges.count)
            } catch let error as TokenizerJSON.Malformed {
                throw Failure.malformed("not JSON at byte \(error.at)")
            }
        }
    }

    /// The vocabulary's id of `symbol`, and the rank of the merge `left right` — the tables as read.
    package func id(of symbol: String) -> Int? { vocabulary.value(symbol) }
    package func rank(of left: String, _ right: String) -> Int? { ranks.value(left + "\u{0}" + right) }
    package var vocabularyEntries: Int { vocabulary.count }
    package var mergeEntries: Int { ranks.count }

    /// `model.vocab` and `model.merges` from the raw bytes. Two published spellings of a merge: a pair
    /// of strings (recent `tokenizers`) and a single string joined by a space (older files). Both
    /// exist in the wild; neither is ambiguous, because a byte symbol is never a space.
    private static func tables(_ json: TokenizerJSON, vocab: Range<Int>, merges: Range<Int>) throws
        -> (vocabulary: ByteTable, ranks: ByteTable) {
        var entries = 0
        try json.elements(vocab) { _ in entries += 1 }
        var vocabulary = ByteTable(entries: entries, keyBytes: vocab.count)
        try json.elements(vocab) { i in
            let (n, next) = try json.string(i, into: vocabulary.pendingKey)
            let value = try json.integer(try json.expect(0x3A, next))
            vocabulary.commit(length: n, value: value)
        }
        var count = 0
        try json.elements(merges) { _ in count += 1 }
        var ranks = ByteTable(entries: count, keyBytes: merges.count)
        var rank = 0
        try json.elements(merges) { i in
            let key = ranks.pendingKey
            let start = json.space(i)
            if json.bytes[start] == 0x5B {
                let (left, next) = try json.string(start + 1, into: key)
                key[left] = 0
                let (right, after) = try json.string(try json.expect(0x2C, next), into: key + left + 1)
                guard json.bytes[json.space(after)] == 0x5D else {
                    throw Failure.malformed("model.merges[\(rank)] is not a pair")
                }
                ranks.commit(length: left + 1 + right, value: rank)
            } else {
                let (n, _) = try json.string(start, into: key)
                guard let space = (0..<n).first(where: { key[$0] == 0x20 }) else {
                    throw Failure.malformed("model.merges[\(rank)] is neither a pair nor a joined string")
                }
                key[space] = 0
                ranks.commit(length: n, value: rank)
            }
            rank += 1
        }
        return (vocabulary, ranks)
    }

    private static func require(_ value: String?, equals expected: String, field: String) throws {
        guard value == expected else {
            throw Failure.unsupported(field, found: value.map { "\"\($0)\"" } ?? "absent")
        }
    }

    private static func requireByteLevel(_ stage: [String: Any]?, field: String) throws {
        try require(stage?["type"] as? String, equals: "ByteLevel", field: "\(field).type")
        // `add_prefix_space` would insert a leading space that the goldens never see, and
        // `use_regex` would apply a **second** split after the one above.
        try require(String(describing: stage?["add_prefix_space"] ?? 0), equals: "0",
                    field: "\(field).add_prefix_space")
        if field.hasPrefix("pre_tokenizer") {
            try require(String(describing: stage?["use_regex"] ?? 0), equals: "0",
                        field: "\(field).use_regex")
        }
    }

    // MARK: - Encoding

    /// The text as identifiers, with no special token added.
    ///
    /// **The order is the reference's and it is not interchangeable**: added tokens
    /// are extracted from the **raw** text first, and only what remains between them is normalized,
    /// split and merged. Normalizing first would let NFC alter a token's spelling
    /// before it is recognized.
    package func encode(_ text: String) -> [Int] {
        var identifiers: [Int] = []
        for segment in split(text) {
            switch segment {
            case .added(let id): identifiers.append(id)
            case .text(let run): encodeRun(run, into: &identifiers)
            }
        }
        return identifiers
    }

    private enum Segment { case added(Int); case text(String) }

    /// Leftmost, longest sweep over the raw text, scalar by scalar.
    private func split(_ text: String) -> [Segment] {
        guard !addedByFirst.isEmpty else { return [.text(text)] }
        let scalars = Array(text.unicodeScalars)
        var segments: [Segment] = []
        var pending = String.UnicodeScalarView()
        var index = 0
        while index < scalars.count {
            var matched: (scalars: [Unicode.Scalar], id: Int)?
            for candidate in addedByFirst[scalars[index]] ?? [] {
                guard index + candidate.scalars.count <= scalars.count else { continue }
                var offset = 0
                while offset < candidate.scalars.count,
                      scalars[index + offset] == candidate.scalars[offset] { offset += 1 }
                if offset == candidate.scalars.count { matched = candidate; break }  // sorted, longest first
            }
            if let matched {
                if !pending.isEmpty { segments.append(.text(String(pending))); pending = String.UnicodeScalarView() }
                segments.append(.added(matched.id))
                index += matched.scalars.count
            } else {
                pending.append(scalars[index]); index += 1
            }
        }
        if !pending.isEmpty { segments.append(.text(String(pending))) }
        return segments
    }

    /// NFC, pre-split, byte alphabet, merges.
    ///
    /// **The regular expression runs on UTF-16 (ICU) and the byte alphabet on
    /// UTF-8**: this boundary is the only place where this can be wrong in a way nothing
    /// else notices. That is what the astral, ZWJ and combining-mark cases of the donor's
    /// corpus are for.
    private func encodeRun(_ run: String, into identifiers: inout [Int]) {
        let normalized = normalizeNFC ? run.precomposedStringWithCanonicalMapping : run
        let text = normalized as NSString
        var cursor = 0
        func merge(_ piece: String) {
            guard !piece.isEmpty else { return }
            identifiers.append(contentsOf: bpe(Array(piece.utf8).map { Self.byteSymbols[Int($0)] }))
        }
        splitter.enumerateMatches(in: normalized, range: NSRange(location: 0, length: text.length)) { match, _, _ in
            guard let match else { return }
            // `Isolated` keeps what the expression did not capture as pieces in their own right.
            // Qwen's pattern *seems* to cover everything — but "seems" is not a reason to
            // throw the gaps overboard.
            if match.range.location > cursor {
                merge(text.substring(with: NSRange(location: cursor, length: match.range.location - cursor)))
            }
            merge(text.substring(with: match.range))
            cursor = match.range.location + match.range.length
        }
        if cursor < text.length { merge(text.substring(from: cursor)) }
    }

    /// Repeatedly merges the adjacent pair of lowest rank.
    ///
    /// Merging the leftmost occurrence then rescanning is exactly the reference's "merge
    /// all occurrences in one left-to-right pass": the best-ranked pair
    /// stays so as long as its occurrences are not exhausted, and overlapping
    /// occurrences are consumed in the same order.
    private func bpe(_ input: [String]) -> [Int] {
        var parts = input
        if parts.count > 1 {
            while true {
                var best = Int.max, at = -1
                for i in 0..<(parts.count - 1) {
                    if let rank = ranks.value(parts[i] + "\u{0}" + parts[i + 1]), rank < best { best = rank; at = i }
                }
                if at < 0 { break }
                parts[at] += parts[at + 1]
                parts.remove(at: at + 1)
            }
        }
        // `byte_fallback` is disabled and the 256 byte symbols are in the vocabulary, so
        // a lookup cannot fail — unless the file is not the one we validated.
        return parts.map { vocabulary.value($0) ?? -1 }
    }

    // MARK: - The chat template, and what the pipeline does with it

    /// `apply_chat_template([{user, prompt}], add_generation_prompt=True, enable_thinking=True)`
    /// for the Qwen3 template, reduced to the only form the pipeline ever asks for.
    ///
    /// **No Jinja engine**: with a single user turn and thinking enabled, the template is
    /// this literal and nothing else. `goldens-text.json` records the string `transformers`
    /// produces — the shortcut is **pinned**, not assumed.
    package static let chatPrefix = "<|im_start|>user\n"
    package static let chatSuffix = "<|im_end|>\n<|im_start|>assistant\n"

    /// The padding token: `<|endoftext|>`. Read from `goldens-text.safetensors`, not deduced.
    package static let padToken = 151643

    /// What the engine gives the text encoder: `[maxLength]` identifiers and their mask.
    ///
    /// **The pipeline pads to fixed length** (`padding="max_length"`, 512) and does not cut
    /// short sequences — the mask carries the information, not the length. `cap_feats` is then
    /// the masked extraction: 19 rows out of 512 for the bench prompt (`pipeline_z_image.py`
    /// 217-245).
    ///
    /// > **Truncation is silent and it costs the assistant header**: at 512, a
    /// > long prompt is cut mid-sentence and loses `<|im_end|>\n<|im_start|>assistant\n`
    /// > entirely. That is the reference's behavior, so it is **reproduced and not corrected** —
    /// > the useful budget is 504 tokens, and saying so at admission is the caller's job.
    package func encodeChat(user prompt: String, maxLength: Int = 512) -> (ids: [Int], mask: [Int]) {
        Self.pad(encode(Self.chatPrefix + prompt + Self.chatSuffix), maxLength: maxLength)
    }

    /// `encodeChat` for a render: the whole prompt in its template, unpadded (the encoder reads the
    /// real tokens alone), **refused past `Request.maximumPromptTokens`**, never truncated.
    /// `encodeChat` keeps the reference's cut at 512, for the checks.
    package func admitChat(user prompt: String) throws -> [Int] {
        let ids = encode(Self.chatPrefix + prompt + Self.chatSuffix)
        let template = encode(Self.chatPrefix + Self.chatSuffix).count
        try Request.admitPrompt(tokens: ids.count - template, max: Request.maximumPromptTokens)
        return ids
    }

    /// **Truncation and padding, separated from encoding.** This is the half of `encodeChat` that
    /// does not depend on the vocabulary — hence the half that can be checked in a millisecond, without
    /// the 7 MB of JSON that the tokenizer parses at startup.
    ///
    /// This is not a test convenience: it is the rule that carries **the documented trap
    /// above**. At 512, a long prompt is cut mid-sentence and loses
    /// `<|im_end|>\n<|im_start|>assistant\n` entirely — the reference's behavior, reproduced
    /// and not corrected. A rule described in a comment and never executed is
    /// a rule one believes is upheld.
    static func pad(_ identifiers: [Int], maxLength: Int) -> (ids: [Int], mask: [Int]) {
        var ids = identifiers
        if ids.count > maxLength { ids = Array(ids.prefix(maxLength)) }
        let realCount = ids.count
        ids += Array(repeating: padToken, count: maxLength - realCount)
        return (ids, Array(repeating: 1, count: realCount)
                   + Array(repeating: 0, count: maxLength - realCount))
    }

    // MARK: - Decoding

    private lazy var spellings: [Int: String] = {
        var table = [Int: String](minimumCapacity: vocabulary.count + addedIdentifiers.count)
        vocabulary.forEach { symbol, id in table[id] = symbol }
        for (id, content) in addedIdentifiers { table[id] = content }
        return table
    }()

    private lazy var byteOfSymbol: [String: UInt8] = {
        var table = [String: UInt8](minimumCapacity: 256)
        for (byte, symbol) in Self.byteSymbols.enumerated() { table[symbol] = UInt8(byte) }
        return table
    }()

    /// The identifiers back to text — the inverse of `encode`, to **read a failure** rather than
    /// stare at two lists of integers. Added tokens come back spelled out, and invalid
    /// UTF-8 becomes U+FFFD, which is what the published `errors: "replace"` asks for.
    package func decode(_ identifiers: [Int]) -> String {
        var bytes: [UInt8] = [], out = ""
        func flush() {
            guard !bytes.isEmpty else { return }
            out += String(decoding: bytes, as: UTF8.self)
            bytes.removeAll(keepingCapacity: true)
        }
        for id in identifiers {
            if let content = addedIdentifiers[id] { flush(); out += content; continue }
            guard let spelling = spellings[id] else { continue }
            for symbol in spelling.unicodeScalars { bytes.append(byteOfSymbol[String(symbol)] ?? 0) }
        }
        flush()
        return out
    }

    // MARK: - Token bytes, for a stream

    /// **The bytes each identifier decodes to**, in two flat regions (`TokenBytes`): what an incremental
    /// detokenizer needs, since one byte-level token may hold half a UTF-8 character and `decode` of
    /// a single identifier would replace it. `rows`: the identifiers that exist (a model's logits);
    /// those the tokenizer does not know decode to nothing. Added tokens are their content's bytes.
    package func tokenBytes(rows: Int) -> TokenBytes {
        var scalarByte: [Unicode.Scalar: UInt8] = [:]
        for (byte, symbol) in Self.byteSymbols.enumerated() { scalarByte[symbol.unicodeScalars.first!] = UInt8(byte) }
        var lengths = [Int](repeating: 0, count: rows)
        vocabulary.forEach { symbol, id in if id >= 0, id < rows { lengths[id] = symbol.unicodeScalars.count } }
        for (id, content) in addedIdentifiers where id >= 0 && id < rows { lengths[id] = content.utf8.count }
        let result = TokenBytes(lengths: lengths)
        vocabulary.forEach { symbol, id in
            guard id >= 0, id < rows else { return }
            result.fill(id) { out in for (i, scalar) in symbol.unicodeScalars.enumerated() { out[i] = scalarByte[scalar] ?? 0 } }
        }
        for (id, content) in addedIdentifiers where id >= 0 && id < rows {
            result.fill(id) { out in for (i, byte) in content.utf8.enumerated() { out[i] = byte } }
        }
        return result
    }

    /// **What `tokenBytes(rows:)` maps, bounded from the file alone** (a scan, like `tableBytes(file:)`): the
    /// offsets exactly, the bytes by the text of `model.vocab` and of `added_tokens` — a token decodes to
    /// at most as many bytes as its spelling has UTF-8 bytes, quotes and ids included.
    package static func detokenizationBytes(file path: String, rows: Int) throws -> Int {
        guard FileManager.default.fileExists(atPath: path) else { throw Failure.fileNotFound(path) }
        let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .alwaysMapped)
        return try data.withUnsafeBytes { raw throws -> Int in
            let json = TokenizerJSON(bytes: raw)
            do {
                guard let model = try json.members(of: 0, ["model"])["model"], raw[model.lowerBound] == 0x7B,
                      let vocab = try json.members(of: model.lowerBound, ["vocab"])["vocab"] else { throw Failure.malformed("no model.vocab") }
                let added = try json.members(of: 0, ["added_tokens"])["added_tokens"]?.count ?? 0
                return TokenBytes.mappedBytes(rows: rows, spelledBytes: vocab.count + added)
            } catch let error as TokenizerJSON.Malformed {
                throw Failure.malformed("not JSON at byte \(error.at)")
            }
        }
    }
}

/// **Identifier → bytes**: `offsets[id] ..< offsets[id + 1]` into one blob, both in `Pages` (unmapped with the
/// value, never in `malloc`).
package struct TokenBytes {
    private let offsets: Pages
    private let blob: Pages
    package let rows: Int

    init(lengths: [Int]) {
        rows = lengths.count
        offsets = Pages(bytes: (lengths.count + 1) * MemoryLayout<UInt32>.size)
        blob = Pages(bytes: max(1, lengths.reduce(0, +)))
        let table = offsets.base.assumingMemoryBound(to: UInt32.self)
        var at = 0
        for (id, n) in lengths.enumerated() { table[id] = UInt32(at); at += n }
        table[lengths.count] = UInt32(at)
    }

    func fill(_ id: Int, _ write: (UnsafeMutablePointer<UInt8>) -> Void) {
        let table = offsets.base.assumingMemoryBound(to: UInt32.self)
        write(blob.base.assumingMemoryBound(to: UInt8.self) + Int(table[id]))
    }

    /// The bytes of `id`; empty outside the table.
    package func bytes(of id: Int) -> UnsafeBufferPointer<UInt8> {
        guard id >= 0, id < rows else { return UnsafeBufferPointer(start: nil, count: 0) }
        let table = offsets.base.assumingMemoryBound(to: UInt32.self)
        return UnsafeBufferPointer(start: blob.base.assumingMemoryBound(to: UInt8.self) + Int(table[id]),
                                   count: Int(table[id + 1] - table[id]))
    }

    /// What a table of `rows` identifiers whose spellings take `spelledBytes` maps at most.
    package static func mappedBytes(rows: Int, spelledBytes: Int) -> Int {
        Arena.capacity(for: (rows + 1) * MemoryLayout<UInt32>.size) + Arena.capacity(for: max(1, spelledBytes))
    }

    /// What this table maps.
    package var mapped: Int { offsets.capacity + blob.capacity }
}
