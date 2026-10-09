import CryptoKit
import Foundation

/// **What a render does not redo when only the seed changes** — kept on disk, under the library,
/// between two renders, two processes, the CLI and the app.
///
///     <racine>/cache/conditioning/<key>.bin   the text stage's output (every model) and the image
///                                             encoder's latents (img2img start, editing references):
///                                             a small header, then the floats — LRU by mtime under `budget`
///     <racine>/cache/kv/<key>.kv              Qwen-Image-2.1's conditions K/V of the LAST edit (one per
///                                             phase: two in the 9-step mode), the file the prefill wrote
///
/// **Bit for bit, by construction**: what comes back is the floats that were written, and a key
/// names everything that determined them — the module's type and parameters, each file it reads
/// (path, size, modification date: a reinstall changes the key), the prompt, the bytes of each image,
/// the settings that change a computation's bits, and **the binary that computed them** (a rebuilt
/// engine starts from an empty cache rather than trust floats of another version). A key that
/// misses only costs the computation it would have saved. The LoRAs are not in the conditioning
/// key: none reaches a text encoder or a VAE (they are applied by the DiTs).
///
/// What is NOT kept, and why: the noise and everything after it (they depend on the seed); the
/// DiT (rebuilding it costs milliseconds, its weights are re-read at every evaluation); a text-
/// only Qwen prefix's K/V (tens of rows computed inside the first step, nothing to gain).
///
/// **Off**: `SILICONED_CACHE=0` (or `"cache": false` in the profile, `EngineSettings.cache`); the
/// checks never see it — they build their `Context` without it, and the close-out script turns it off
/// for the product lines it runs (a second render read from the first's cache would judge nothing).
package struct RenderCache: Sendable {
    package let directory: URL
    /// The conditioning folder's ceiling, in bytes: beyond it, the least recently used entries go.
    /// A Qwen edit with three references keeps ~65 MB (`[T, 4096]` floats), a Z-Image prompt ~0.2 MB.
    package var budget: Int
    /// **The kept K/V must leave this much free on the volume**, after it is written: an edit's
    /// cache is up to 6.4 GB (1024×1536, three references). Below, the render falls back on the
    /// unlinked file of before, which disappears with it.
    package var keptReserve: Int64

    package static let defaultBudget = 512 << 20
    package static let defaultKeptReserve: Int64 = 8 << 30

    package init(directory: URL, budget: Int = RenderCache.defaultBudget,
                 keptReserve: Int64 = RenderCache.defaultKeptReserve) {
        self.directory = directory; self.budget = budget; self.keptReserve = keptReserve
    }

    package var conditioningFolder: URL { directory.appendingPathComponent("conditioning") }
    package var kvFolder: URL { directory.appendingPathComponent("kv") }

    // ── keys ─────────────────────────────────────────────────────────────────────────────

    /// A SHA-256 over **length-prefixed** fields: `("ab", "c")` and `("a", "bc")` are two keys.
    package struct Key: Hashable, Sendable, CustomStringConvertible {
        package let hex: String
        package var description: String { hex }
    }

    package struct KeyBuilder {
        private var hasher = SHA256()
        package init(_ kind: String) { add(kind) }

        package mutating func add(_ text: String) { add(bytes: Array(text.utf8), tag: 0x53) }
        package mutating func add(_ number: Int) { add(String(number)) }
        package mutating func add(floats: [Float]) {
            floats.withUnsafeBytes { add(raw: $0, tag: 0x46) }
        }
        package mutating func add(flags: [Bool]) { add(bytes: flags.map { $0 ? 1 : 0 }, tag: 0x42) }
        /// An image: its size, then its planar floats.
        package mutating func add(_ image: ImageRGB) {
            add(image.width); add(image.height); add(floats: image.pixels)
        }
        package mutating func add(_ latent: Latent) {
            add(latent.space.name); add(latent.height); add(latent.width); add(floats: latent.values)
        }

        private mutating func add(bytes: [UInt8], tag: UInt8) {
            bytes.withUnsafeBytes { add(raw: $0, tag: tag) }
        }
        private mutating func add(raw: UnsafeRawBufferPointer, tag: UInt8) {
            var header = [tag] + withUnsafeBytes(of: UInt64(raw.count).littleEndian, Array.init)
            header.withUnsafeMutableBytes { hasher.update(bufferPointer: UnsafeRawBufferPointer($0)) }
            hasher.update(bufferPointer: raw)
        }

        package func finish() -> Key {
            Key(hex: hasher.finalize().map { String(format: "%02x", $0) }.joined())
        }
    }

    /// **A file's revision**: its path, size and modification date (to the nanosecond) — a folder's
    /// is that of each regular file under it, in name order. A missing path is named as such.
    package static func stamp(_ path: String) -> String {
        let fm = FileManager.default
        var folder: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &folder) else { return path + " (missing)" }
        func one(_ p: String) -> String {
            var s = stat()
            guard stat(p, &s) == 0 else { return p + " (unreadable)" }
            return "\(p) \(s.st_size) \(s.st_mtimespec.tv_sec).\(s.st_mtimespec.tv_nsec)"
        }
        guard folder.boolValue else { return one(path) }
        let files = (fm.enumerator(atPath: path)?.allObjects as? [String] ?? []).sorted()
            .map { (path as NSString).appendingPathComponent($0) }
            .filter { var d: ObjCBool = false; return fm.fileExists(atPath: $0, isDirectory: &d) && !d.boolValue }
        return files.map(one).joined(separator: "\n")
    }

    /// **The binary that computes**: the executable's revision. A rebuilt engine (another version, the
    /// CLI or the app) does not read what another one wrote.
    package static let build: String = Bundle.main.executableURL.map { stamp($0.resolvingSymlinksInPath().path) } ?? "unknown"

    /// **The settings that change a computation's bits** (AMX co-execution and its split, the SDPA
    /// in fp16, the flash kernel, the widening path), plus the render's `reproducible`.
    package static func numerics(reproducible: Bool) -> String {
        let s = EngineSettings.effective
        return "amx \(s.amx) \(s.amxFraction) \(s.amxThreads) \(s.amxQuantum) \(s.amxMinimumRows) "
            + "frozen \(reproducible) flash \(s.flash) \(s.flashRows ?? 0) fp16 \(s.sdpaFp16) "
            + "widen \(s.widenGPU) \(s.widenFused) parallel \(s.parallel)"
    }

    /// The key of a text stage's output: the module, the prompt, the images it sees.
    package static func textKey(_ module: any CacheIdentified, prompt: String, images: [ImageRGB],
                                reproducible: Bool) -> Key {
        var k = KeyBuilder("conditioning v1")
        k.add(build); k.add(numerics(reproducible: reproducible))
        for part in module.cacheIdentity { k.add(part) }
        k.add(prompt)
        k.add(images.count)
        for image in images { k.add(image) }
        return k.finish()
    }

    /// The key of an image encoder's latent.
    package static func imageKey(_ module: any CacheIdentified, image: ImageRGB, reproducible: Bool) -> Key {
        var k = KeyBuilder("latent v1")
        k.add(build); k.add(numerics(reproducible: reproducible))
        for part in module.cacheIdentity { k.add(part) }
        k.add(image)
        return k.finish()
    }

    // ── the conditioning folder ──────────────────────────────────────────────────────────

    /// `SLCACHE1`, the header's length (UInt32 LE), the header (JSON), the floats (fp32 LE).
    private static let magic = Array("SLCACHE1".utf8)

    package static func encode(header: [String: Any], values: [Float]) throws -> Data {
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var data = Data(magic)
        withUnsafeBytes(of: UInt32(json.count).littleEndian) { data.append(contentsOf: $0) }
        data.append(json)
        values.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }

    package static func decode(_ data: Data) -> (header: [String: Any], values: [Float])? {
        let m = magic.count
        guard data.count >= m + 4, Array(data.prefix(m)) == magic else { return nil }
        let length = data.subdata(in: m..<(m + 4)).withUnsafeBytes { Int(UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))) }
        let start = m + 4 + length
        guard length > 0, data.count >= start, (data.count - start) % 4 == 0,
              let header = (try? JSONSerialization.jsonObject(with: data.subdata(in: (m + 4)..<start))) as? [String: Any]
        else { return nil }
        var values = [Float](repeating: 0, count: (data.count - start) / 4)
        values.withUnsafeMutableBytes { _ = data.copyBytes(to: $0, from: start..<data.count) }
        return (header, values)
    }

    package static func header(_ c: Conditioning, key: Key) -> [String: Any] {
        ["key": key.hex, "type": "conditioning", "format": c.format.name, "readsImages": c.format.readsImages,
         "rows": c.rows, "width": c.width, "tokens": c.tokens,
         "slots": String(c.imageSlots.map { $0 ? "1" : "0" })]
    }

    package static func conditioning(header h: [String: Any], values: [Float], key: Key) -> Conditioning? {
        guard h["key"] as? String == key.hex, h["type"] as? String == "conditioning",
              let format = h["format"] as? String, let readsImages = h["readsImages"] as? Bool,
              let rows = h["rows"] as? Int, let width = h["width"] as? Int, let tokens = h["tokens"] as? Int,
              let slots = h["slots"] as? String, values.count == rows * width,
              slots.isEmpty || slots.count == rows else { return nil }
        return Conditioning(format: TextFormat(format, readsImages: readsImages), rows: rows, width: width,
                            values: values, tokens: tokens, imageSlots: slots.map { $0 == "1" })
    }

    package static func header(_ l: Latent, key: Key) -> [String: Any] {
        ["key": key.hex, "type": "latent", "space": l.space.name, "channels": l.space.channels,
         "factor": l.space.factor, "height": l.height, "width": l.width]
    }

    package static func latent(header h: [String: Any], values: [Float], key: Key) -> Latent? {
        guard h["key"] as? String == key.hex, h["type"] as? String == "latent",
              let name = h["space"] as? String, let channels = h["channels"] as? Int, let factor = h["factor"] as? Int,
              let height = h["height"] as? Int, let width = h["width"] as? Int,
              values.count == channels * height * width else { return nil }
        return Latent(space: LatentSpace(name: name, channels: channels, factor: factor), height: height, width: width,
                      values: values)
    }

    private func entry(_ key: Key) -> URL { conditioningFolder.appendingPathComponent(key.hex + ".bin") }

    /// The entry, if it is there and whole — touched (its mtime is the LRU's clock). A damaged one is removed.
    private func read<T>(_ key: Key, _ make: ([String: Any], [Float]) -> T?) -> T? {
        let url = entry(key)
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard let (header, values) = RenderCache.decode(data), let value = make(header, values) else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: url.path)
        return value
    }

    /// Written under a temporary name, then renamed: a reader never sees half an entry. Best effort —
    /// a full disk or a read-only library only loses the saving.
    private func write(_ key: Key, header: [String: Any], values: [Float]) {
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: conditioningFolder, withIntermediateDirectories: true)
            let data = try RenderCache.encode(header: header, values: values)
            guard data.count <= budget else { return }
            let temporary = conditioningFolder.appendingPathComponent(".\(key.hex).\(getpid()).\(UUID().uuidString)")
            try data.write(to: temporary)
            if rename(temporary.path, entry(key).path) != 0 { try? fm.removeItem(at: temporary) }
            RenderCache.evict(folder: conditioningFolder, budget: budget, keeping: entry(key))
        } catch {}
    }

    package func conditioning(_ key: Key) -> Conditioning? {
        read(key) { RenderCache.conditioning(header: $0, values: $1, key: key) }
    }
    package func store(_ c: Conditioning, key: Key) { write(key, header: RenderCache.header(c, key: key), values: c.values) }
    package func latent(_ key: Key) -> Latent? {
        read(key) { RenderCache.latent(header: $0, values: $1, key: key) }
    }
    package func store(_ l: Latent, key: Key) { write(key, header: RenderCache.header(l, key: key), values: l.values) }

    /// **Least recently used first, until the folder fits `budget`** — `keeping` (the entry just
    /// written) never goes. Returns what was removed.
    @discardableResult
    package static func evict(folder: URL, budget: Int, keeping: URL? = nil) -> [URL] {
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey, .isRegularFileKey]
        guard let items = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: keys) else { return [] }
        var entries = items.compactMap { url -> (url: URL, size: Int, date: Date)? in
            guard let v = try? url.resourceValues(forKeys: Set(keys)), v.isRegularFile == true else { return nil }
            return (url, v.fileSize ?? 0, v.contentModificationDate ?? .distantPast)
        }
        var total = entries.reduce(0) { $0 + $1.size }
        entries.sort { ($0.date, $0.url.lastPathComponent) < ($1.date, $1.url.lastPathComponent) }
        var removed: [URL] = []
        for e in entries where total > budget && e.url.standardizedFileURL != keeping?.standardizedFileURL {
            if (try? fm.removeItem(at: e.url)) != nil { total -= e.size; removed.append(e.url) }
        }
        return removed
    }

    // ── the kept K/V ─────────────────────────────────────────────────────────────────────

    /// The kept K/V file of a key (`QwenImage21KVFile`).
    package func kvPath(_ key: Key) -> String { kvFolder.appendingPathComponent(key.hex + ".kv").path }

    /// **One edit kept, never two**: everything in `kv/` that is not one of `keys` goes — what the
    /// previous edit kept, and a file a killed render left half-written.
    package func keepOnlyKV(_ keys: [Key]) {
        let fm = FileManager.default
        try? fm.createDirectory(at: kvFolder, withIntermediateDirectories: true)
        let kept = Set(keys.map { $0.hex + ".kv" })
        for name in (try? fm.contentsOfDirectory(atPath: kvFolder.path)) ?? [] where !kept.contains(name) {
            try? fm.removeItem(atPath: kvFolder.appendingPathComponent(name).path)
        }
    }
}

/// **What determines a module's output, beyond its input**: its type, its parameters, and the
/// revision of each file it reads (`RenderCache.stamp`). A module without it is never cached.
package protocol CacheIdentified {
    var cacheIdentity: [String] { get }
}

extension Library {
    /// `<racine>/cache/` — what renders keep between them (`RenderCache`). Everything in it can be
    /// thrown away: it only saves recomputing.
    public var cacheFolder: URL { root.appendingPathComponent("cache") }

    /// **Empties the render cache** (`cacheFolder`). A render reading it meanwhile keeps its open file.
    public func emptyCache() throws(EngineError) {
        if Self.exists(cacheFolder.path) { try EngineError.boundary { try FileManager.default.removeItem(at: cacheFolder) } }
    }

    /// The library's render cache, unless the settings turn it off (`EngineSettings.cache`).
    package var renderCache: RenderCache? {
        EngineSettings.effective.cache ? RenderCache(directory: cacheFolder) : nil
    }
}

// ── what each module's output depends on ─────────────────────────────────────────────────

private func identity(_ module: Any, _ parameters: [String], files: [String]) -> [String] {
    [String(describing: type(of: module))] + parameters + files.map(RenderCache.stamp)
}

extension ZImageTextModule: CacheIdentified {
    package var cacheIdentity: [String] { identity(self, [output.name], files: [map, tokenizer]) }
}
extension AnimaTextModule: CacheIdentified {
    package var cacheIdentity: [String] { identity(self, [output.name], files: [map, tokenizerQwen, tokenizerT5, adapter]) }
}
extension Krea2TextModule: CacheIdentified {
    package var cacheIdentity: [String] {
        identity(self, [output.name, sockets.map(String.init).joined(separator: ",")], files: [map, tokenizer])
    }
}
extension KleinTextModule: CacheIdentified {
    package var cacheIdentity: [String] { identity(self, [output.name], files: [map, tokenizer]) }
}
extension ErnieTextModule: CacheIdentified {
    package var cacheIdentity: [String] { identity(self, [output.name], files: [map, tokenizer]) }
}
extension QwenImage21TextModule: CacheIdentified {
    package var cacheIdentity: [String] { identity(self, [output.name], files: [map, tokenizer]) }
}
extension FluxEncodingModule: CacheIdentified {
    package var cacheIdentity: [String] { identity(self, [space.name], files: [path]) }
}
extension QwenImageEncodingModule: CacheIdentified {
    package var cacheIdentity: [String] { identity(self, [space.name], files: [path]) }
}
extension Flux2EncodingModule: CacheIdentified {
    package var cacheIdentity: [String] { identity(self, [space.name, "eps \(eps.bitPattern)"], files: [path]) }
}
extension QwenImage21EncodingModule: CacheIdentified {
    package var cacheIdentity: [String] { identity(self, [space.name], files: [path]) }
}
