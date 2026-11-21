import Foundation

/// **« Use this model » on Hugging Face**: a model page's button opens the site's relay page, which opens
/// `siliconed://hf?repo_id=<owner/name>[&file=<path>]` (`HubLink`). The app then asks the Hub what the repository
/// holds (`HubRepository`) and offers what an import could take:
///
///     link ──▶ HubRepository (api/models/<id>, revision pinned to its commit)
///                ├─ a family's own repository ──▶ that family's installation (its files pinned)
///                └─ .safetensors ──▶ examined on their header (`HubImport.examine`)
///                                   ──▶ chosen ──▶ downloaded, sha256 verified
///                                   ──▶ `ModelImport.importFile`, as a file dropped
///
/// **The link comes from the web**: nothing in it is trusted — a repository is `owner/name`, a file is a relative
/// path with neither `..` nor `\`, and nothing downloads before the user has seen the repository, the file, its size
/// and its license. An import still downloads nothing: the download is here, before it.
///
/// **A repository mixes models** (one puts an SDXL and an Anima LoRA side by side): each file is judged on its own
/// header, by the import's own rules, before it is offered.

// MARK: - The link

package struct HubLink: Equatable, Sendable {
    package let repository: String
    /// The file the Hub's dropdown chose, when it chose one. A path, not yet a file an import takes (`isFile`): a
    /// repository of another format still opens, and says that nothing in it imports.
    package let file: String?

    package static let scheme = "siliconed"

    /// `siliconed://hf?repo_id=…&file=…`; `nil` for anything else, or anything malformed.
    package init?(_ url: URL) {
        guard url.scheme?.lowercased() == Self.scheme, url.host?.lowercased() == "hf",
              let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems,
              let id = items.first(where: { $0.name == "repo_id" })?.value, Self.isRepository(id) else { return nil }
        let file = items.first(where: { $0.name == "file" })?.value.flatMap { $0.isEmpty ? nil : $0 }
        if let file, !Self.isPath(file) { return nil }
        self.init(repository: id, file: file)
    }

    package init(repository: String, file: String? = nil) {
        self.repository = repository
        self.file = file
    }

    package var url: URL {
        var c = URLComponents()
        c.scheme = Self.scheme
        c.host = "hf"
        c.queryItems = [URLQueryItem(name: "repo_id", value: repository)] + (file.map { [URLQueryItem(name: "file", value: $0)] } ?? [])
        return c.url!
    }

    /// The Hub's rule for a name (`huggingface_hub.utils.validate_repo_id`): letters, digits, `-`, `_`, `.`, at most
    /// 96 characters, neither `--` nor `..`, not starting or ending with `-` or `.`.
    package static func isRepository(_ s: String) -> Bool {
        let parts = s.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2 else { return false }
        return parts.allSatisfy { p in
            p.count >= 1 && p.count <= 96 && !p.contains("--") && !p.contains("..")
                && p.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || "-_.".contains($0)) }
                && !"-.".contains(p.first!) && !"-.".contains(p.last!)
        }
    }

    /// A path inside a repository: relative, no empty, `.` or `..` component, no `\`, no control character, a weights
    /// file by its extension (`.safetensors`, or the `.gguf` the Hub's dropdown also hands).
    package static func isPath(_ s: String) -> Bool {
        let lower = s.lowercased()
        guard s.count <= 1024, lower.hasSuffix(".safetensors") || lower.hasSuffix(".gguf"),
              !s.contains("\\"), !s.unicodeScalars.contains(where: { $0.properties.generalCategory == .control }) else { return false }
        return s.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }

    /// A path an import takes: a `.safetensors`.
    package static func isFile(_ s: String) -> Bool { isPath(s) && s.lowercased().hasSuffix(".safetensors") }
}

// MARK: - What the Hub says of a repository

package struct HubRepository: Sendable {
    package struct File: Equatable, Sendable {
        package let path: String
        package let size: Int
        package let sha256: String
        package init(path: String, size: Int, sha256: String) { (self.path, self.size, self.sha256) = (path, size, sha256) }
    }

    package let id: String
    /// The commit the Hub served: every byte downloaded is read at it, never at `main`.
    package let commit: String
    /// `false`, `"auto"` or `"manual"` on the Hub: a license to accept there, under an account the app does not have —
    /// its files are not offered (`AppState.open`).
    package let gated: Bool
    package let license: String?
    /// The `.safetensors` files stored in LFS (the only ones with a sha256 announced).
    package let files: [File]

    package init(json: Data) throws(EngineError) {
        guard let o = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let id = o["id"] as? String, HubLink.isRepository(id),
              let commit = o["sha"] as? String, commit.count == 40, commit.allSatisfy(\.isHexDigit) else {
            throw .internalFailure(component: "hub", detail: "the Hub's answer is not a model's description")
        }
        self.id = id
        self.commit = commit
        gated = (o["gated"] as? String).map { !$0.isEmpty } ?? (o["gated"] as? Bool ?? false)
        let tags = o["tags"] as? [String] ?? []
        let card = o["cardData"] as? [String: Any]
        license = (card?["license"] as? String) ?? tags.first { $0.hasPrefix("license:") }.map { String($0.dropFirst(8)) }
        files = (o["siblings"] as? [[String: Any]] ?? []).compactMap { s in
            guard let path = s["rfilename"] as? String, HubLink.isFile(path),
                  let lfs = s["lfs"] as? [String: Any], let sha = lfs["sha256"] as? String,
                  sha.count == 64, sha.allSatisfy(\.isHexDigit), let size = lfs["size"] as? Int, size > 8 else { return nil }
            return File(path: path, size: size, sha256: sha.lowercased())
        }
    }

    /// `api/models/<id>?blobs=true`, as anyone reads it.
    package static func fetch(_ id: String) throws(EngineError) -> HubRepository {
        guard HubLink.isRepository(id), let u = URL(string: "https://huggingface.co/api/models/\(id)?blobs=true") else {
            throw .downloadRefused(file: id, status: 404)
        }
        let request = URLRequest(url: u)
        let answer: (Data, URLResponse)
        do { answer = try blocking { try await URLSession.shared.data(for: request) } } catch {
            throw .downloadInterrupted(file: id)
        }
        let status = (answer.1 as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else { throw .downloadRefused(file: id, status: status) }
        return try HubRepository(json: answer.0)
    }

    /// **The family whose installation this repository's button opens**, read from the families' own declarations:
    /// the repository its license is read in (its publisher's), or one that holds, byte for byte, the DiT a version
    /// of it installs (importing that file would only copy what the installation lays down).
    package var installs: Family? {
        Family.allCases.first { f in
            f.traits.card.license.urls.first.map { $0.host == "huggingface.co" && $0.path.hasPrefix("/\(id)/") } ?? false
        } ?? Family.allCases.first { f in
            f.variants.contains { v in
                f.files(v).contains { p in
                    p.place == f.ditMap(v) && files.contains { $0.size == p.bytes && .sha256($0.sha256) == p.digest }
                }
            }
        }
    }

    /// **What an import could take, in the order offered**: the file the link chose first, then the others by path.
    /// Never a diffusers shard (`-00001-of-00003`: an import takes one file), a VAE or a text encoder.
    package func candidates(preferring chosen: String? = nil) -> [File] {
        let skipped = ["vae", "text_encoder", "text_encoder_2", "tokenizer", "image_encoder", "controlnet"]
        let kept = files.filter { f in
            let components = f.path.lowercased().split(separator: "/").map(String.init)
            return !components.dropLast().contains(where: skipped.contains)
                && f.path.range(of: #"-\d{5}-of-\d{5}\.safetensors$"#, options: .regularExpression) == nil
        }
        return kept.sorted { a, b in
            if (a.path == chosen) != (b.path == chosen) { return a.path == chosen }
            return a.path.localizedStandardCompare(b.path) == .orderedAscending
        }
    }

    /// The file as a download takes it: pinned at the commit read, judged by the size and the digest the Hub listed.
    package func published(_ f: File) -> PublishedFile {
        PublishedFile(Repository(id, commit), f.path, at: "", bytes: f.size, sha256: f.sha256)
    }
}

// MARK: - Examined, then downloaded

package enum HubImport {
    /// What a file would import as, judged on its header alone (a few hundred KB).
    package enum Verdict: Equatable, Sendable {
        case lora(Family)
        case model(Family)
        /// A model whose family has no version installed: the installation comes first (an import downloads nothing).
        case modelNeedsFamily(Family)
        /// What the header alone cannot judge — a header too long to read ahead, ComfyUI's quantization texts too
        /// scattered through the data: the import judges it once downloaded.
        case unexamined
        /// The import's refusal, as it would say it after the download.
        case refused(String)

        /// What « Import » may download.
        package var importable: Bool {
            switch self {
            case .lora, .model, .unexamined: true
            case .modelNeedsFamily, .refused: false
            }
        }
    }

    /// The longest header read ahead of a download. A LoRA's is a few hundred KB, a DiT's under 1 MB; a repository
    /// that declares more is judged once downloaded, not fetched 24 times over for a sheet.
    static let headerRead = 16 << 20

    /// **The import's verdict before a byte of weights**: the header by two `Range` requests, written to a file of
    /// the published size whose data stays sparse (APFS allocates nothing there), then `ModelImport.recognize` on
    /// it — the import's own rules. **Values aside but the ones the rules read**: ComfyUI's `comfy_quant` are JSON
    /// texts in the data (`MapWeights.init`); left as zeros they read as « not a quantization this app reads », and
    /// every int8 convrot would be refused. They are fetched by `Range` and written in place (`contentRanges`); too
    /// scattered to fetch (`contentRanges`), the file is `unexamined` rather than judged on zeros. A publisher's
    /// undistilled model is told by the digest the Hub lists, never by hashing a sparse file.
    /// `installed`: the families with a version whole in the library that would receive the file.
    package static func examine(_ r: HubRepository, _ f: HubRepository.File, installed: Set<Family>) throws(EngineError) -> Verdict {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("siliconed-hub-\(UUID().uuidString).safetensors").path
        defer { try? FileManager.default.removeItem(atPath: temporary) }
        let written: Bool = try EngineError.boundary {
            let u = r.published(f).repository.url(f.path)
            var head = try Downloader.range(u, 0, 7)
            let declared = head.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard let n = Safetensors.headerLength(declared, fileSize: f.size), n > 0 else {
                throw EngineError.importRefused(file: (f.path as NSString).lastPathComponent,
                                                detail: "header length \(UInt64(littleEndian: declared))")
            }
            guard n <= headerRead else { return false }
            head += try Downloader.range(u, 8, 8 + n - 1)
            guard let contents = contentRanges(header: head, size: f.size) else { return false }
            try sparse(head, size: f.size, contents: zip(contents, try Downloader.ranges(u, contents)).map { ($0, $1) }, at: temporary)
            return true
        }
        guard written else { return .unexamined }
        do {
            let (kind, family) = try ModelImport.recognize(temporary, sha256: f.sha256)
            if kind == .lora { return .lora(family) }
            return installed.contains(family) ? .model(family) : .modelNeedsFamily(family)
        } catch .importRefused(_, let detail) {
            return .refused(detail)
        }
    }

    /// **The byte ranges to fetch so that the examination reads what the import would**: every `comfy_quant` text,
    /// absolute in the file, those less than `gap` apart merged into one request. Empty: nothing to fetch. `nil`:
    /// more than `requests` ranges or more than `bytes` in all — the file is judged once downloaded. A header that
    /// does not parse, or an offset outside the file, gives no range: the reader refuses the header itself.
    /// **256 requests**: an int8 DiT keeps one text beside each of its layers, megabytes apart — 224 of 72 bytes in a
    /// Krea 2 fine-tune of 13.5 GB, which sixteen requests left unexamined, to be refused once downloaded.
    static func contentRanges(header head: Data, size: Int, gap: Int = 1 << 20, requests: Int = 256,
                              bytes: Int = 4 << 20) -> [Range<Int>]? {
        guard head.count > 8, let o = try? JSONSerialization.jsonObject(with: head.subdata(in: 8..<head.count)) as? [String: Any] else { return [] }
        var pieces: [Range<Int>] = []
        for (name, v) in o where name.hasSuffix(".comfy_quant") {
            guard let e = v as? [String: Any], let d = e["data_offsets"] as? [Int], d.count == 2, d[0] >= 0, d[0] < d[1],
                  head.count + d[1] <= size else { continue }
            pieces.append((head.count + d[0])..<(head.count + d[1]))
        }
        var merged: [Range<Int>] = []
        for p in pieces.sorted(by: { $0.lowerBound < $1.lowerBound }) {
            if let last = merged.last, p.lowerBound - last.upperBound <= gap {
                merged[merged.count - 1] = last.lowerBound..<max(last.upperBound, p.upperBound)
            } else {
                merged.append(p)
            }
        }
        guard merged.count <= requests, merged.reduce(0, { $0 + $1.count }) <= bytes else { return nil }
        return merged
    }

    /// The examined file: `head`, then `size` bytes in all, sparse but for `contents`, each at its place.
    static func sparse(_ head: Data, size: Int, contents: [(Range<Int>, Data)], at path: String) throws {
        FileManager.default.createFile(atPath: path, contents: head)
        let h = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
        defer { try? h.close() }
        try h.truncate(atOffset: UInt64(size))
        for (range, data) in contents {
            try h.seek(toOffset: UInt64(range.lowerBound))
            try h.write(contentsOf: data)
        }
    }

    /// **Where a file from the Hub waits for its import**: a folder of its own under the library's `televersements`,
    /// which goes with the file once the import has judged it (`AppState.perform`) and is emptied at launch
    /// otherwise. Named by the file's digest: a download stopped resumes when the link is opened again.
    package static func destination(_ r: HubRepository, _ f: HubRepository.File, in b: Library) -> String {
        let folder = "hub-" + r.id.replacingOccurrences(of: "/", with: "--") + "-" + f.sha256.prefix(12)
        return UploadStore.directory(in: b).appendingPathComponent(folder)
            .appendingPathComponent((f.path as NSString).lastPathComponent).path
    }

    /// The download, at the commit examined, verified against the sha256 the Hub listed. Refused before the first
    /// byte when the volume cannot hold the file and its copy in the library (`UploadStore.needed`).
    package static func download(_ r: HubRepository, _ f: HubRepository.File, in b: Library, cancellation: Cancellation? = nil,
                                 received: (@Sendable (Int) -> Void)? = nil,
                                 journal: @escaping @Sendable (String) -> Void = { _ in }) throws(EngineError) -> String {
        let destination = destination(r, f, in: b)
        let needed = UploadStore.needed(for: Int64(f.size))
        if let free = b.freeSpace(), Int64(free) < needed {
            throw .diskFull(needed: Int(clamping: needed), available: free)
        }
        try EngineError.boundary {
            try Downloader.fetch(r.published(f), to: destination, cancellation: cancellation, received: received, journal: journal)
        }
        return destination
    }
}
