import Foundation

/// Anima's **T5** tokenizer — a SentencePiece unigram, read from the published `tokenizer.json`.
///
/// Anima does not use it for a T5 encoder: the conditioner reads its identifiers as **learned
/// queries** (an `embed` table of 32,128 rows) that go looking for meaning in the
/// Qwen states. A wrong identifier therefore shows up on no norm — it changes one query
/// row, and the image follows.
///
/// Same discipline as `Tokenizer`: this is **not** a general `tokenizer.json` interpreter.
/// Each stage is checked and **refused** if it is not the one we reproduce:
///
///     added tokens       extracted from the RAW text first (`normalized: false`)
///     normalizer         `Precompiled` — SentencePiece's `nmt_nfkc` table, a double-array trie
///     pre-split          `WhitespaceSplit` then `Metaspace("▁", always, split)`
///     model              `Unigram`, Viterbi over the scores, unknowns fused, no byte fallback
///     post-processing    `A </s>`
///
/// The reference is `tokenizers` (Rust), checked against its outputs by the developer checks
/// and by a corpus of edge cases.
package final class T5Tokenizer {
    package enum Failure: Error, CustomStringConvertible {
        case unreadable(String)
        case unsupported(String)
        package var description: String {
            switch self {
            case .unreadable(let p): return "T5 tokenizer unreadable: \(p)"
            case .unsupported(let f): return "T5 tokenizer: \(f) not reproduced — refused"
            }
        }
    }

    private let pieces: [String: (id: Int, score: Double)]
    private let longestPiece: Int
    private let unknownId: Int, unknownScore: Double
    private let endId: Int
    /// The added tokens, as sequences of scalars, longest first.
    private let added: [(scalars: [Unicode.Scalar], id: Int)]
    private let charsmap: Charsmap

    package convenience init(directory: String) throws {
        try self.init(file: directory + "/tokenizer.json")
    }

    package init(file path: String) throws {
        guard let data = FileManager.default.contents(atPath: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Failure.unreadable(path)
        }
        guard let model = json["model"] as? [String: Any], model["type"] as? String == "Unigram",
              let vocab = model["vocab"] as? [[Any]], let unk = model["unk_id"] as? Int else {
            throw Failure.unsupported("model other than `Unigram`")
        }
        if model["byte_fallback"] as? Bool == true { throw Failure.unsupported("byte_fallback") }
        if let fuse = model["fuse_unk"] as? Bool, !fuse { throw Failure.unsupported("fuse_unk = false") }

        var pieces: [String: (Int, Double)] = [:]
        var longest = 0, lowest = Double.infinity
        for (id, entry) in vocab.enumerated() {
            guard entry.count == 2, let piece = entry[0] as? String, let score = (entry[1] as? NSNumber)?.doubleValue else {
                throw Failure.unsupported("vocabulary entry \(id)")
            }
            pieces[piece] = (id, score)
            longest = max(longest, piece.unicodeScalars.count)
            lowest = min(lowest, score)
        }
        self.pieces = pieces
        self.longestPiece = longest
        self.unknownId = unk
        // `tokenizers`: `min_score − K_UNK_PENALTY`, the penalty being 10.
        self.unknownScore = lowest - 10

        guard let normalizer = json["normalizer"] as? [String: Any], normalizer["type"] as? String == "Precompiled",
              let encoded = normalizer["precompiled_charsmap"] as? String,
              let blob = Data(base64Encoded: encoded) else {
            throw Failure.unsupported("normalizer other than `Precompiled`")
        }
        self.charsmap = try Charsmap(blob)

        guard let pre = json["pre_tokenizer"] as? [String: Any], pre["type"] as? String == "Sequence",
              let stages = pre["pretokenizers"] as? [[String: Any]], stages.count == 2,
              stages[0]["type"] as? String == "WhitespaceSplit",
              stages[1]["type"] as? String == "Metaspace", stages[1]["replacement"] as? String == "▁",
              stages[1]["prepend_scheme"] as? String == "always", stages[1]["split"] as? Bool == true else {
            throw Failure.unsupported("pre-split other than WhitespaceSplit + Metaspace(always, split)")
        }

        guard let post = json["post_processor"] as? [String: Any], post["type"] as? String == "TemplateProcessing",
              let single = post["single"] as? [[String: Any]], single.count == 2,
              (single[1]["SpecialToken"] as? [String: Any])?["id"] as? String == "</s>",
              let end = pieces["</s>"]?.0 else {
            throw Failure.unsupported("post-processing other than `A </s>`")
        }
        self.endId = end

        var added: [([Unicode.Scalar], Int)] = []
        for token in json["added_tokens"] as? [[String: Any]] ?? [] {
            guard let content = token["content"] as? String, let id = token["id"] as? Int else { continue }
            if token["normalized"] as? Bool == true || token["lstrip"] as? Bool == true
                || token["rstrip"] as? Bool == true || token["single_word"] as? Bool == true {
                throw Failure.unsupported("added token \"\(content)\" normalized or trimmed")
            }
            added.append((Array(content.unicodeScalars), id))
        }
        self.added = added.sorted { $0.0.count > $1.0.count }
    }

    /// The text as identifiers, `</s>` included, truncated to `maxLength` **before** the `</s>` — which is
    /// what `tokenizers` does under `truncation=True`: the end token always survives.
    package func encode(_ text: String, maxLength: Int = 512) -> [Int] {
        var ids: [Int] = []
        var run = String.UnicodeScalarView()
        func flush() {
            if !run.isEmpty { encodeRun(String(run), into: &ids); run.removeAll() }
        }
        let scalars = Array(text.unicodeScalars)
        var i = 0
        scan: while i < scalars.count {
            for (token, id) in added where i + token.count <= scalars.count
                && Array(scalars[i..<(i + token.count)]) == token {
                flush(); ids.append(id); i += token.count
                continue scan
            }
            run.append(scalars[i]); i += 1
        }
        flush()
        return Array(ids.prefix(maxLength - 1)) + [endId]
    }

    private func encodeRun(_ run: String, into ids: inout [Int]) {
        let normalized = charsmap.normalize(run)
        var word = String.UnicodeScalarView()
        func emitWord() {
            guard !word.isEmpty else { return }
            // Metaspace: `▁` in front (unless already there), then a cut before each `▁`,
            // the delimiter staying glued to what follows it.
            var marked = Array(word)
            if marked.first != "▁" { marked.insert("▁", at: 0) }
            var start = 0
            for j in 1...marked.count where j == marked.count || marked[j] == "▁" {
                viterbi(Array(marked[start..<j]), into: &ids)
                start = j
            }
            word.removeAll()
        }
        for scalar in normalized.unicodeScalars {
            if scalar.properties.isWhitespace { emitWord() } else { word.append(scalar) }
        }
        emitWord()
    }

    /// The highest-scoring split. A position that no one-scalar piece covers
    /// receives the unknown, and consecutive unknowns make just one (`fuse_unk`).
    private func viterbi(_ scalars: [Unicode.Scalar], into ids: inout [Int]) {
        let n = scalars.count
        var best = [Double](repeating: -.infinity, count: n + 1)
        var back = [(start: Int, id: Int)](repeating: (0, 0), count: n + 1)
        best[0] = 0
        for start in 0..<n where best[start] > -.infinity {
            var single = false
            var piece = String.UnicodeScalarView()
            for length in 1...min(longestPiece, n - start) {
                piece.append(scalars[start + length - 1])
                guard let (id, score) = pieces[String(piece)] else { continue }
                if length == 1 { single = true }
                let candidate = best[start] + score
                if candidate > best[start + length] { best[start + length] = candidate; back[start + length] = (start, id) }
            }
            if !single {
                let candidate = best[start] + unknownScore
                if candidate > best[start + 1] { best[start + 1] = candidate; back[start + 1] = (start, unknownId) }
            }
        }
        var path: [Int] = []
        var end = n
        while end > 0 { path.append(back[end].id); end = back[end].start }
        // Merging does not cross words: `tokenizers` does it within each lattice.
        var previous: Int? = nil
        for id in path.reversed() {
            if id == unknownId, previous == unknownId { continue }
            ids.append(id); previous = id
        }
    }

    /// SentencePiece's `precompiled_charsmap` table: a **double-array** trie (darts-clone)
    /// over UTF-8 bytes, each leaf of which points into a block of normalized strings
    /// terminated by a zero. Transcribed from `spm_precompiled`, the crate `tokenizers` uses.
    struct Charsmap {
        private let units: [UInt32]
        private let normalized: [UInt8]

        init(_ blob: Data) throws {
            let bytes = [UInt8](blob)
            guard bytes.count >= 4 else { throw Failure.unsupported("empty charsmap") }
            let trieBytes = Int(UInt32(bytes[0]) | UInt32(bytes[1]) << 8 | UInt32(bytes[2]) << 16 | UInt32(bytes[3]) << 24)
            guard trieBytes % 4 == 0, 4 + trieBytes <= bytes.count else { throw Failure.unsupported("truncated charsmap") }
            var units = [UInt32](repeating: 0, count: trieBytes / 4)
            for k in 0..<units.count {
                let o = 4 + 4 * k
                units[k] = UInt32(bytes[o]) | UInt32(bytes[o + 1]) << 8 | UInt32(bytes[o + 2]) << 16 | UInt32(bytes[o + 3]) << 24
            }
            self.units = units
            self.normalized = Array(bytes[(4 + trieBytes)...])
        }

        private static func hasLeaf(_ u: UInt32) -> Bool { (u >> 8) & 1 == 1 }
        private static func value(_ u: UInt32) -> Int { Int(u & ((1 << 31) - 1)) }
        private static func label(_ u: UInt32) -> UInt32 { u & ((1 << 31) | 0xFF) }
        private static func offset(_ u: UInt32) -> Int { Int((u >> 10) << ((u & (1 << 9)) >> 6)) }

        /// `common_prefix_search`, of which only the **first** result is kept — the shortest
        /// prefix — like `spm_precompiled::transform`.
        private func transform(_ chunk: [UInt8]) -> [UInt8]? {
            var node = Charsmap.offset(units[0])
            for c in chunk {
                if c == 0 { break }
                node ^= Int(c)
                guard node < units.count else { return nil }
                let unit = units[node]
                guard Charsmap.label(unit) == UInt32(c) else { return nil }
                node ^= Charsmap.offset(unit)
                if Charsmap.hasLeaf(unit), node < units.count {
                    let start = Charsmap.value(units[node])
                    var end = start
                    while end < normalized.count, normalized[end] != 0 { end += 1 }
                    return Array(normalized[start..<end])
                }
            }
            return nil
        }

        /// Grapheme by grapheme: the whole grapheme if it is under six bytes and the
        /// table knows it, otherwise scalar by scalar.
        func normalize(_ text: String) -> String {
            var out: [UInt8] = []
            for grapheme in text {
                let whole = Array(String(grapheme).utf8)
                if whole.count < 6, let mapped = transform(whole) { out += mapped; continue }
                for scalar in String(grapheme).unicodeScalars {
                    let part = Array(String(scalar).utf8)
                    out += transform(part) ?? part
                }
            }
            return String(decoding: out, as: UTF8.self)
        }
    }
}
