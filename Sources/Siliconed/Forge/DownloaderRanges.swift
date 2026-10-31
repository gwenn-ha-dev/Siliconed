import Foundation

// **A tensor fetched by byte range from a big safetensors file** — the writer's head is 1.25 GB of a 17.5 GB text
// encoder the family already installs. Streamed to a file (never held in memory), checked against a sha256 pinned
// for exactly those bytes before it is used, and laid in a small safetensors file the forge reads like any shard.

extension Downloader {
    /// A range pinned in the recipe: the tensor's name, its absolute start in the file, its length, the sha256 of those bytes.
    package struct PinnedRange: Equatable, Sendable {
        package let name: String, begin: Int, length: Int, sha256: String
        package init(name: String, begin: Int, length: Int, sha256: String) {
            self.name = name; self.begin = begin; self.length = length; self.sha256 = sha256
        }
    }

    /// A pinned range with what the file's own header says of the tensor.
    package struct TensorRange: Equatable, Sendable {
        package let name: String, dtype: String, shape: [Int], begin: Int, length: Int, sha256: String
        /// The last byte, inclusive (an HTTP `Range`).
        package var last: Int { begin + length - 1 }
    }

    /// **Where each tensor lives, from the safetensors header**: the data starts after `8 + headerLength` bytes and each
    /// tensor's `data_offsets` are relative to it. The plan is refused when a header says other than the pin — a range
    /// of the wrong bytes would fail its hash later; failing here costs nothing.
    package static func plan(_ pinned: [PinnedRange], headerLength: Int, header: [String: Any], file: String) throws -> [TensorRange] {
        try pinned.map { pin in
            guard let entry = header[pin.name] as? [String: Any], let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int], let offsets = entry["data_offsets"] as? [Int], offsets.count == 2 else {
                throw EngineError.downloadCorrupt(file: "\(file): \(pin.name)")
            }
            let begin = 8 + headerLength + offsets[0]
            guard begin == pin.begin, offsets[1] - offsets[0] == pin.length else {
                throw EngineError.downloadCorrupt(file: "\(file): \(pin.name)")
            }
            return TensorRange(name: pin.name, dtype: dtype, shape: shape, begin: begin, length: pin.length, sha256: pin.sha256)
        }
    }

    /// The plan from the remote file's header, read by two small `Range` requests.
    package static func plan(_ pinned: [PinnedRange], remoteHeaderOf url: URL) throws -> [TensorRange] {
        let prefix = try range(url, 0, 7)
        let n = prefix.withUnsafeBytes { Int(exactly: $0.loadUnaligned(as: UInt64.self)) } ?? 0
        guard n > 0, n <= Safetensors.maximumHeader else { throw EngineError.downloadCorrupt(file: url.lastPathComponent) }
        guard let header = try JSONSerialization.jsonObject(with: try range(url, 8, 8 + n - 1)) as? [String: Any] else {
            throw EngineError.downloadCorrupt(file: url.lastPathComponent)
        }
        return try plan(pinned, headerLength: n, header: header, file: url.lastPathComponent)
    }

    /// **Checks a range's file**: its size and its sha256. Anything else removes the file and is refused.
    package static func verify(_ path: String, _ r: TensorRange) throws {
        let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? -1
        guard size == r.length, try sha256(path) == r.sha256 else {
            try? FileManager.default.removeItem(atPath: path)
            throw EngineError.downloadCorrupt(file: r.name)
        }
    }

    /// **Downloads one range to `destination`**, streamed by URLSession to a file; a file already there and right is kept.
    /// Only a `206` is taken (a server that ignores `Range` answers `200` with the whole 17.5 GB).
    package static func downloadRange(_ url: URL, _ r: TensorRange, to destination: String, cancellation: Cancellation?,
                                      journal: @escaping @Sendable (String) -> Void) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination) {
            if (try? verify(destination, r)) != nil { return }
        }
        try fm.createDirectory(atPath: (destination as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        journal("↓ \(url.lastPathComponent) [\(r.name)] (\(String(format: "%.2f", Double(r.length) / 1e9)) GB)")
        var request = request(url)
        request.setValue("bytes=\(r.begin)-\(r.last)", forHTTPHeaderField: "Range")
        let frozen = request
        let tracking = Tracker(label: r.name, cancellation: cancellation, journal: journal)
        let session = URLSession(configuration: .default, delegate: tracking, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let temporary: URL, response: URLResponse
        do {
            (temporary, response) = try wait { try await session.download(for: frozen) }
        } catch let e as URLError where e.code == .cancelled && cancellation?.isCancelled == true {
            throw EngineError.cancelled
        } catch is URLError {
            throw EngineError.downloadInterrupted(file: r.name)
        }
        guard let h = response as? HTTPURLResponse, h.statusCode == 206 else {
            try? fm.removeItem(at: temporary)
            throw EngineError.downloadRefused(file: r.name, status: (response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        try? fm.removeItem(atPath: destination)
        try fm.moveItem(at: temporary, to: URL(fileURLWithPath: destination))
        try verify(destination, r)
    }

    /// **The tensors' files laid in one safetensors file**: a header with each tensor's dtype, shape and offsets, then
    /// the bytes in the given order, copied in blocks.
    package static func assembleSafetensors(_ tensors: [(TensorRange, String)], to path: String) throws {
        var entries: [(String, Any)] = [], offset = 0
        for (r, _) in tensors {
            entries.append((r.name, ["dtype": r.dtype, "shape": r.shape, "data_offsets": [offset, offset + r.length]] as [String: Any]))
            offset += r.length
        }
        var json = try JSONSerialization.data(withJSONObject: Dictionary(uniqueKeysWithValues: entries), options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        let fm = FileManager.default
        try? fm.removeItem(atPath: path)
        guard fm.createFile(atPath: path, contents: nil), let out = FileHandle(forWritingAtPath: path) else {
            throw EngineError.fileUnreadable(path: path)
        }
        defer { try? out.close() }
        var length = UInt64(json.count).littleEndian
        try out.write(contentsOf: Data(bytes: &length, count: 8))
        try out.write(contentsOf: json)
        for (_, file) in tensors {
            guard let input = FileHandle(forReadingAtPath: file) else { throw EngineError.fileUnreadable(path: file) }
            defer { try? input.close() }
            while let block = try input.read(upToCount: 8 << 20), !block.isEmpty { try autoreleasepool { try out.write(contentsOf: block) } }
        }
    }
}
