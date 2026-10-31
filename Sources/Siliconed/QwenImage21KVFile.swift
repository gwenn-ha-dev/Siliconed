import Foundation

/// **Where the conditions' K/V live**, by the size of the prefix. Measured on a 1-reference edit
/// (P = 4,092, 4.3 GB of cache, 768×512 then 1248×832): in memory, the render swapped (+167 k pages
/// out, 8.3 GB peak); on disk (`QwenImage21KVFile`), the same bits, no swap, and faster (10.5 against
/// 12.0 s per step: the read of layer `l + 1` hides behind layer `l`, where the memory copy did not);
/// recomputed (a full pass per step), 29.8 s per step and other bits. So: a text-only prefix (a
/// generation, tens of rows) in memory, anything with an image on disk.
package enum QwenImage21CachePolicy {
    /// `2 · layers · P · d · 4` bytes.
    package static func bytes(prefix: Int, layers: Int = 32, dim: Int = 4096) -> Int { 2 * layers * prefix * dim * 4 }
    package static let memoryBudget = 256 << 20

    /// **The one placement rule** — the sampler's, the module's and the memory need's (`QwenImage21Need`).
    package static func storage(prefix: Int) -> QwenImage21DiT.CacheStorage {
        bytes(prefix: prefix) <= memoryBudget ? .memory : .file
    }

    /// The longest prefix whose K/V stay in memory (256 rows).
    package static var largestMemoryPrefix: Int { memoryBudget / bytes(prefix: 1) }

    /// **The DiT's prefix of a conditioning**: its rows, each image slot being 4 rows of the DiT (one 2×2
    /// group of latent tokens) where the encoder has one — what `QwenImage21Sequence.prefix` counts.
    package static func prefix(conditioningRows rows: Int, imageSlots: Int) -> Int { rows + 3 * imageSlots }
}

/// **The conditions' K/V on disk** — `QwenImage21DiT.CacheStorage.file`: one unlinked temporary file
/// (it disappears with the process, even killed), written once by the prefill, read back layer by layer
/// at each step. `F_NOCACHE`: the reads and writes go straight between the SSD and the DiT's own
/// `keys`/`values` rows, never through the unified buffer cache — the file never occupies memory,
/// neither ours nor the system's. Every offset and length is a multiple of a 16 KB page (`d = 4096`
/// floats), as direct I/O requires.
///
/// The step reads layer `l + 1` while layer `l` finishes (`prefetch` after its attention, `wait` before
/// the next): the prefix rows `[0, P)` of `keys`/`values` are free from the attention on, the target
/// writes only rows `[P, P + S)`.
///
/// **Kept between renders** (`kept(at:…)`, the render cache's `kv/<key>.kv`): the same file, written
/// under a temporary name, then — once the prefill has written its last layer — synced, closed by a
/// trailer page (magic, key, shape) and renamed. The next render with the same key opens it
/// `complete`: no prefill, the steps read it as they read the file they would have written.
package final class QwenImage21KVFile {
    package private(set) var path: String
    package let bytesPerLayer: Int
    package let layers: Int
    private let fd: Int32
    private let lanes = 8
    private var pending: (layer: Int, group: DispatchGroup, failure: ErrorBox)?
    /// The conditions' K/V are all there (a kept file read back): the prefill has nothing to do.
    package private(set) var complete = false
    /// This render writes the file the next one may read back (`kept`, not complete yet).
    package let writesKept: Bool
    /// Where a file being kept goes once complete, and its trailer's key; `nil` for the unlinked one.
    private var destination: (path: String, key: String)?
    /// The trailer: one page after the layers, `SLKV1` + JSON (key, layers, bytes per layer).
    package static let trailerBytes = Arena.alignment

    private final class ErrorBox: @unchecked Sendable {
        private let lock = NSLock()
        private var error: Int32 = 0
        func fail(_ e: Int32) { lock.lock(); if error == 0 { error = e }; lock.unlock() }
        var value: Int32 { lock.lock(); defer { lock.unlock() }; return error }
    }

    package struct Failure: Error, CustomStringConvertible {
        package let description: String
    }

    /// - Parameter halfBytes: `P · d · 4`, one of K or V for one layer.
    package init(halfBytes: Int, layers: Int, directory: String = NSTemporaryDirectory()) throws {
        precondition(halfBytes % Arena.alignment == 0, "QwenImage21KVFile: a half that is not whole pages")
        bytesPerLayer = 2 * halfBytes; self.layers = layers; writesKept = false
        let total = Int64(bytesPerLayer) * Int64(layers)
        let free = (try? URL(fileURLWithPath: directory).resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? 0
        guard free > total + (2 << 30) else {
            // The K/V cache of an edit, with 2 GB to spare.
            throw EngineError.diskFull(needed: Int(total) + (2 << 30), available: Int(free))
        }
        let path = (directory as NSString).appendingPathComponent("siliconed-kv-\(getpid())-\(UUID().uuidString)")
        let fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0o600)
        guard fd >= 0 else { throw Failure(description: "K/V cache: cannot create \(path) (errno \(errno))") }
        unlink(path)
        _ = fcntl(fd, F_NOCACHE, 1)
        self.path = path; self.fd = fd
    }

    private init(fd: Int32, path: String, halfBytes: Int, layers: Int, complete: Bool,
                 destination: (path: String, key: String)?) {
        self.fd = fd; self.path = path; self.bytesPerLayer = 2 * halfBytes; self.layers = layers
        self.complete = complete; self.destination = destination; self.writesKept = destination != nil
        _ = fcntl(fd, F_NOCACHE, 1)
    }

    /// **The kept file of `key`**: read back if it is there and whole (`complete`), otherwise one to
    /// write next to it, renamed to `path` by `seal`. `nil` when keeping it would leave less than
    /// `reserve` bytes free (or the folder cannot be written): the caller then takes the unlinked file.
    package static func kept(at path: String, key: String, halfBytes: Int, layers: Int, reserve: Int64) -> QwenImage21KVFile? {
        precondition(halfBytes % Arena.alignment == 0, "QwenImage21KVFile: a half that is not whole pages")
        let total = Int64(2 * halfBytes) * Int64(layers)
        let fd = open(path, O_RDONLY)
        if fd >= 0 {
            var s = stat()
            if fstat(fd, &s) == 0, Int64(s.st_size) == total + Int64(trailerBytes),
               let trailer = readTrailer(fd, offset: off_t(total)),
               trailer.key == key, trailer.layers == layers, trailer.bytesPerLayer == 2 * halfBytes {
                utimes(path, nil)   // its date says when it last served
                return QwenImage21KVFile(fd: fd, path: path, halfBytes: halfBytes, layers: layers, complete: true,
                                         destination: nil)
            }
            close(fd)
            unlink(path)
        }
        let folder = (path as NSString).deletingLastPathComponent
        let free = (try? URL(fileURLWithPath: folder).resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage ?? 0
        guard free > total + Int64(trailerBytes) + reserve else { return nil }
        let partial = path + ".\(getpid())-\(UUID().uuidString).partial"
        let writing = open(partial, O_RDWR | O_CREAT | O_EXCL, 0o600)
        guard writing >= 0 else { return nil }
        return QwenImage21KVFile(fd: writing, path: partial, halfBytes: halfBytes, layers: layers, complete: false,
                                 destination: (path, key))
    }

    private static func readTrailer(_ fd: Int32, offset: off_t) -> (key: String, layers: Int, bytesPerLayer: Int)? {
        var page = [UInt8](repeating: 0, count: trailerBytes)
        guard page.withUnsafeMutableBytes({ pread(fd, $0.baseAddress!, trailerBytes, offset) }) == trailerBytes,
              Array(page.prefix(5)) == Array("SLKV1".utf8) else { return nil }
        let json = page.dropFirst(5).prefix { $0 != 0 }
        guard let h = (try? JSONSerialization.jsonObject(with: Data(json))) as? [String: Any],
              let key = h["key"] as? String, let layers = h["layers"] as? Int, let bytes = h["bytesPerLayer"] as? Int
        else { return nil }
        return (key, layers, bytes)
    }

    /// **The prefill is done**: the layers to the disk (`fsync`), the trailer after them, the rename to
    /// the kept name — the file is now what the next render with the same key reads. Best effort: a
    /// failure here (a full disk, the cache emptied meanwhile) only loses the keeping; the render
    /// goes on reading its open file.
    package func seal() {
        guard let (target, key) = destination else { return }
        destination = nil
        pending?.group.wait()
        let header = (try? JSONSerialization.data(withJSONObject: ["key": key, "layers": layers, "bytesPerLayer": bytesPerLayer] as [String: Any],
                                                  options: [.sortedKeys])) ?? Data()
        guard 5 + header.count < Self.trailerBytes else { unlink(path); return }
        let page = UnsafeMutableRawPointer.allocate(byteCount: Self.trailerBytes, alignment: Arena.alignment)
        defer { page.deallocate() }
        page.initializeMemory(as: UInt8.self, repeating: 0, count: Self.trailerBytes)
        Array("SLKV1".utf8).withUnsafeBytes { page.copyMemory(from: $0.baseAddress!, byteCount: 5) }
        header.withUnsafeBytes { (page + 5).copyMemory(from: $0.baseAddress!, byteCount: header.count) }
        guard fsync(fd) == 0,
              pwrite(fd, page, Self.trailerBytes, off_t(layers * bytesPerLayer)) == Self.trailerBytes,
              rename(path, target) == 0 else {
            unlink(path)
            return
        }
        path = target
    }

    /// **`seal`, only behind a finite output**: the first evaluation's velocity, which every condition's
    /// K/V reaches (the joint attention reads them all from every target row) — a single non-finite one
    /// makes it non-finite. Otherwise the file is not kept (`deinit` unlinks it) and this render goes on
    /// reading its open file: a kept file is read back by every later render of the same edit, without a
    /// prefill, and would hand them its NaN (`RenderCache.allFinite`). Returns `false` when a file
    /// being kept was given up for a non-finite output.
    @discardableResult
    package func seal(after output: [Float]) -> Bool {
        guard destination != nil else { return true }
        guard RenderCache.allFinite(output) else { return false }
        seal()
        return true
    }

    deinit {
        pending?.group.wait()
        close(fd)
        // A kept file left unsealed (a cancelled render, a failure in the first step) is not kept.
        if destination != nil { unlink(path) }
    }

    /// Both halves of one layer, split into `lanes` page-aligned pieces: `(offset in the file, pointer, bytes)`.
    private func pieces(layer: Int, keys: UnsafeMutablePointer<Float>, values: UnsafeMutablePointer<Float>)
            -> [(offset: off_t, pointer: UnsafeMutableRawPointer, bytes: Int)] {
        let half = bytesPerLayer / 2, pages = half / Arena.alignment, perHalf = lanes / 2
        var out: [(offset: off_t, pointer: UnsafeMutableRawPointer, bytes: Int)] = []
        for (h, base) in [UnsafeMutableRawPointer(keys), UnsafeMutableRawPointer(values)].enumerated() {
            for piece in 0..<perHalf {
                let first = piece * pages / perHalf * Arena.alignment, end = (piece + 1) * pages / perHalf * Arena.alignment
                guard end > first else { continue }
                out.append((off_t(layer * bytesPerLayer + h * half + first), base + first, end - first))
            }
        }
        return out
    }

    private func run(_ write: Bool, layer: Int, keys: UnsafeMutablePointer<Float>, values: UnsafeMutablePointer<Float>,
                     group: DispatchGroup, failure: ErrorBox) {
        let fd = self.fd
        for piece in pieces(layer: layer, keys: keys, values: values) {
            nonisolated(unsafe) let pointer = piece.pointer
            let (offset, bytes) = (piece.offset, piece.bytes)
            DispatchQueue.global(qos: .userInitiated).async(group: group) {
                var done = 0
                while done < bytes {
                    let n = write ? pwrite(fd, pointer + done, bytes - done, offset + off_t(done))
                                  : pread(fd, pointer + done, bytes - done, offset + off_t(done))
                    if n <= 0 { failure.fail(n == 0 ? EIO : errno); return }
                    done += n
                }
            }
        }
    }

    /// Writes layer `layer`'s K/V (`keys`, `values`: `[P, d]` each), and returns once they are on disk.
    package func write(layer: Int, keys: UnsafeMutablePointer<Float>, values: UnsafeMutablePointer<Float>) throws {
        let group = DispatchGroup(), failure = ErrorBox()
        run(true, layer: layer, keys: keys, values: values, group: group, failure: failure)
        group.wait()
        if failure.value != 0 { throw Failure(description: "K/V cache: write failed (errno \(failure.value))") }
    }

    /// Starts reading layer `layer` into `keys`/`values`; `wait` collects it.
    package func prefetch(layer: Int, keys: UnsafeMutablePointer<Float>, values: UnsafeMutablePointer<Float>) {
        pending?.group.wait()
        let group = DispatchGroup(), failure = ErrorBox()
        run(false, layer: layer, keys: keys, values: values, group: group, failure: failure)
        pending = (layer, group, failure)
    }

    /// Layer `layer` in `keys`/`values`: the prefetched one, or read now.
    package func wait(layer: Int, keys: UnsafeMutablePointer<Float>, values: UnsafeMutablePointer<Float>) throws {
        if pending?.layer != layer { prefetch(layer: layer, keys: keys, values: values) }
        guard let (_, group, failure) = pending else { return }
        group.wait()
        pending = nil
        if failure.value != 0 { throw Failure(description: "K/V cache: read failed (errno \(failure.value))") }
    }
}
