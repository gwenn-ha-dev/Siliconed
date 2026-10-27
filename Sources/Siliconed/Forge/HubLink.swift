import Foundation

/// **« Use this model » on Hugging Face**: a model page's button opens the site's relay page, which
/// opens `siliconed://hf?repo_id=<owner/name>[&file=<path>]` (`HubLink`). The app then asks the Hub
/// what the repository holds (`HubRepository`) and offers what an import could take:
///
///     link ──▶ HubRepository (api/models/<id>, revision pinned to its commit)
///                ├─ a family's own repository ──▶ that family's installation (sha256 pinned)
///                └─ .safetensors / .gguf ──▶ examined on their header (`HubImport.examine`)
///                                            ──▶ chosen ──▶ downloaded, sha256 verified
///                                            ──▶ `ModelImport.importFile`, as a file dropped
///
/// **The link comes from the web**: nothing in it is trusted — a repository is `owner/name`, a
/// file is a relative path with neither `..` nor `\`, and nothing downloads before the user has
/// seen the repository, the file, its size and its license. An import still downloads nothing:
/// the download is here, before it.
///
/// **A repository mixes models** (one puts an SDXL and an Anima LoRA side by side): each
/// file is judged on its own header, by the forge's own rules, before it is offered.

// MARK: - The link

package struct HubLink: Equatable, Sendable {
    package let repository: String
    /// The file the Hub's dropdown chose (a GGUF repository's quant), when it chose one.
    package let file: String?

    package static let scheme = "siliconed"

    /// `siliconed://hf?repo_id=…&file=…`; `nil` for anything else, or anything malformed.
    package init?(_ url: URL) {
        guard url.scheme?.lowercased() == Self.scheme, url.host?.lowercased() == "hf",
              let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems,
              let id = items.first(where: { $0.name == "repo_id" })?.value, Self.isRepository(id) else { return nil }
        let file = items.first(where: { $0.name == "file" })?.value.flatMap { $0.isEmpty ? nil : $0 }
        if let file, !Self.isFile(file) { return nil }
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

    /// The Hub's rule for a name (`huggingface_hub.utils.validate_repo_id`): letters, digits, `-`,
    /// `_`, `.`, at most 96 characters, neither `--` nor `..`, not starting or ending with `-` or `.`.
    package static func isRepository(_ s: String) -> Bool {
        let parts = s.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2 else { return false }
        return parts.allSatisfy { p in
            p.count >= 1 && p.count <= 96 && !p.contains("--") && !p.contains("..")
                && p.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || "-_.".contains($0)) }
                && !"-.".contains(p.first!) && !"-.".contains(p.last!)
        }
    }

    /// A path inside a repository: relative, no empty, `.` or `..` component, no `\`, no control
    /// character, a `.safetensors` or a `.gguf`.
    package static func isFile(_ s: String) -> Bool {
        let lower = s.lowercased()
        guard s.count <= 1024, lower.hasSuffix(".safetensors") || lower.hasSuffix(".gguf"),
              !s.contains("\\"), !s.unicodeScalars.contains(where: { $0.properties.generalCategory == .control }) else { return false }
        return s.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
}

// MARK: - What the Hub says of a repository

package struct HubRepository: Sendable {
    package struct File: Equatable, Sendable {
        package let path: String
        package let size: Int
        package let sha256: String
    }

    package let id: String
    /// The commit the Hub served: every byte downloaded is read at it, never at `main`.
    package let commit: String
    /// `false`, `"auto"` or `"manual"` on the Hub: a license to accept there, and a token here.
    package let gated: Bool
    package let license: String?
    package let tags: [String]
    /// The `.safetensors` and `.gguf` files stored in LFS (the only ones with a sha256 announced).
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
        tags = o["tags"] as? [String] ?? []
        let card = o["cardData"] as? [String: Any]
        license = (card?["license"] as? String) ?? tags.first { $0.hasPrefix("license:") }.map { String($0.dropFirst(8)) }
        files = (o["siblings"] as? [[String: Any]] ?? []).compactMap { s in
            guard let path = s["rfilename"] as? String, HubLink.isFile(path),
                  let lfs = s["lfs"] as? [String: Any], let sha = lfs["sha256"] as? String,
                  sha.count == 64, sha.allSatisfy(\.isHexDigit), let size = lfs["size"] as? Int, size > 8 else { return nil }
            return File(path: path, size: size, sha256: sha.lowercased())
        }
    }

    /// `api/models/<id>?blobs=true` — with the token `hf auth login` left, for a gated repository.
    package static func fetch(_ id: String) throws(EngineError) -> HubRepository {
        guard HubLink.isRepository(id), let u = URL(string: "https://huggingface.co/api/models/\(id)?blobs=true") else {
            throw .downloadRefused(file: id, status: 404)
        }
        let request = Downloader.request(u)
        let answer: (Data, URLResponse)
        do { answer = try wait { try await URLSession.shared.data(for: request) } } catch {
            throw .downloadInterrupted(file: id)
        }
        let status = (answer.1 as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else { throw .downloadRefused(file: id, status: status) }
        return try HubRepository(json: answer.0)
    }

    /// **The repositories a family is installed from, or is the base of**: the button shows on these
    /// pages, on their adapters' and their quantizations'. Z-Image's and Qwen-Image-2.1's base models
    /// are trained on, and their LoRAs are applied to the Turbo; Krea 2's Turbo is gated, and the
    /// installation takes nothing from it (`Installer`), but it is what LoRAs name. MOSS-Image installs
    /// from its SFT; LoRAs trained on its Base apply to it too, as the community uses them.
    package static let bases: [(id: String, family: Family, installed: Bool)] = [
        (Installer.zImage.name, .zImage, true), ("Tongyi-MAI/Z-Image", .zImage, false),
        (Installer.anima.name, .anima, true),
        (Installer.qwenImage21Turbo.name, .qwenImage21, true), ("Qwen/Qwen-Image-2.1", .qwenImage21, false),
        ("krea/Krea-2-Turbo", .krea2, true), (Installer.krea2Comfy.name, .krea2, false),
        (Installer.mossImage.name, .mossImage, true), ("OpenMOSS-Team/MOSS-Image-Base", .mossImage, false),
        (Installer.mossComfy.name, .mossImage, false),
    ]

    /// The family whose installation this repository is: its page's button installs the family.
    package var installs: Family? { Self.bases.first { $0.installed && $0.id == id }?.family }

    /// The family the card declares (`base_model:adapter:…`, `…:quantized:…`, `…:finetune:…`,
    /// `…:merge:…`) — what the Hub decided to show the button by; the header has the last word.
    package var declaredFamily: Family? {
        for t in tags where t.hasPrefix("base_model:") {
            let parts = t.split(separator: ":", maxSplits: 2)
            if parts.count == 3, let b = Self.bases.first(where: { $0.id == parts[2] }) { return b.family }
        }
        return nil
    }

    /// **What an import could take, in the order offered**: the file the link chose first, then the
    /// others by path. Never a diffusers shard (`-00001-of-00003`: an import takes one file), a VAE
    /// or a text encoder.
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

    package func downloadURL(_ f: File) -> URL { Downloader.url(id, revision: commit, path: f.path) }
    package func page(_ f: File) -> URL { Downloader.page(id, revision: commit, path: f.path) }
}

// MARK: - Examined, then downloaded

package enum HubImport {
    /// What a GGUF's header takes: the metadata and one entry per tensor — unsloth's Z-Image and
    /// vanes430's Anima are read whole in it (453 tensors for Z-Image).
    static let ggufPrefix = 4 << 20

    /// What a file would import as, judged on its header alone (a few hundred KB).
    package enum Verdict: Equatable, Sendable {
        case lora(Family)
        case model(Family)
        /// A LoRA for a family not installed: its shapes are judged once the family is.
        case loraNeedsFamily(Family)
        /// A GGUF whose header is longer than what was read: the import judges it once downloaded.
        case unexamined
        /// The forge's refusal, as it would say it after the download.
        case refused(String)
    }

    /// **The forge's verdict before a byte of weights**: the safetensors header by two `Range`
    /// requests — a GGUF's first `ggufPrefix` bytes, its header having no length up front —, written
    /// to a file of the published size whose data stays sparse (APFS allocates nothing there), then
    /// `ModelImport.examine` on it: the import's own rules, values aside.
    package static func examine(_ r: HubRepository, _ f: HubRepository.File, in b: Library) throws(EngineError) -> Verdict {
        let gguf = f.path.lowercased().hasSuffix(".gguf")
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("siliconed-hub-\(UUID().uuidString)" + (gguf ? ".gguf" : ".safetensors")).path
        defer { try? FileManager.default.removeItem(atPath: temporary) }
        let read: Int = try EngineError.boundary {
            let u = r.downloadURL(f)
            var head: Data
            if gguf {
                head = try Downloader.range(u, 0, min(f.size, ggufPrefix) - 1)
            } else {
                head = try Downloader.range(u, 0, 7)
                let declared = UInt64(littleEndian: head.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
                guard let n = Int(exactly: declared), n > 0, n <= Safetensors.maximumHeader, 8 + n <= f.size else {
                    throw Numerics.Failure(description: "\(f.path) : header length \(declared)")
                }
                head += try Downloader.range(u, 8, 8 + n - 1)
            }
            FileManager.default.createFile(atPath: temporary, contents: head)
            let h = try FileHandle(forWritingTo: URL(fileURLWithPath: temporary))
            defer { try? h.close() }
            try h.truncate(atOffset: UInt64(f.size))
            return head.count
        }
        // A GGUF header that runs past what was read would be judged on zeros: no verdict from it.
        if gguf, let mapped = try? Data(contentsOf: URL(fileURLWithPath: temporary), options: .alwaysMapped) {
            do { _ = try mapped.withUnsafeBytes { try GGUF.parse($0.baseAddress!, size: mapped.count, readable: read) } }
            catch is GGUF.Unread { return .unexamined } catch {}
        }
        do {
            let (kind, family) = try ModelImport.examine(temporary, in: b)
            return kind == .lora ? .lora(family) : .model(family)
        } catch .importRefused(_, let detail) {
            return .refused(detail)
        } catch .importNeedsFamily(_, let model) {
            guard let f = Family.allCases.first(where: { ModelCard.of($0).id == model }) else { throw .importNeedsFamily(file: f.path, model: model) }
            return .loraNeedsFamily(f)
        }
    }

    /// Where a file from the Hub waits for its import: `telechargements/hub/`, which the Models
    /// sheet counts and « Discard » empties; the import throws it away (`AppState.perform`).
    package static func downloads(in b: Library) -> String { Installer(library: b).downloads + "/hub" }

    package static func destination(_ r: HubRepository, _ f: HubRepository.File, in b: Library) -> String {
        downloads(in: b) + "/" + r.id.replacingOccurrences(of: "/", with: "--") + "/" + f.path
    }

    /// The download, at the commit examined, verified against the sha256 the Hub listed.
    package static func download(_ r: HubRepository, _ f: HubRepository.File, to destination: String,
                                 cancellation: Cancellation? = nil,
                                 journal: @escaping @Sendable (String) -> Void = { _ in }) throws(EngineError) {
        try EngineError.boundary {
            try Downloader.download(r.id, revision: r.commit, path: f.path, to: destination, pinned: f.sha256,
                                    cancellation: cancellation, journal: journal)
        }
    }
}
