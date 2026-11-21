import Foundation

/// **A safetensors file, or the shards a model is published in**, read with `pread` straight into the arena: each file
/// `u64` header length · JSON header · raw tensors, the header's `data_offsets` relative to the end of the header. A
/// tensor knows its shard (`Tensor.file`): a group of weights may lie across two (a transformer's layers 15 and 35 cut
/// between two shards of its published files). Positional reads only: the ring's reader shares it.
public final class SafeTensors: @unchecked Sendable {
    public struct Tensor: Sendable {
        public let dtype: String
        public let shape: [Int]
        public let start: Int
        public let end: Int
        /// Its shard, by its rank in `paths`.
        public let file: Int
        public init(dtype: String, shape: [Int], start: Int, end: Int, file: Int = 0) {
            self.dtype = dtype; self.shape = shape; self.start = start; self.end = end; self.file = file
        }
    }
    /// The shards, in order; `path`, the first (what a message names).
    public let paths: [String]
    public var path: String { paths[0] }
    private let fds: [Int32]
    private let bases: [Int]
    public let tensors: [String: Tensor]
    private let uncached: Bool
    /// The prefix the keys were read under (`init(shards:uncached:under:)`), kept by a reopening.
    private let prefix: String

    /// `uncached`: read past the system's file cache (`F_NOCACHE`): weights read again at every pass, on a Mac whose
    /// memory cannot hold the file, would only push other pages out.
    public convenience init(_ path: String, uncached: Bool = false, under prefix: String = "") throws {
        try self.init(shards: [path], uncached: uncached, under: prefix)
    }

    /// The shards of one set of weights, every key in one of them (a key in two: the first's). `under`: the keys under
    /// this prefix alone, named without it (a checkpoint saved whole keeps one model's tensors under its own).
    public init(shards paths: [String], uncached: Bool = false, under prefix: String = "") throws {
        precondition(!paths.isEmpty, "no shard")
        self.paths = paths
        self.uncached = uncached
        self.prefix = prefix
        var fds: [Int32] = [], bases: [Int] = [], t: [String: Tensor] = [:]
        do {
            for (i, path) in paths.enumerated() {
                let fd = open(path, O_RDONLY)
                guard fd >= 0 else { throw Failure("\(path): cannot open") }
                fds.append(fd)
                if uncached { _ = fcntl(fd, F_NOCACHE, 1) }
                var length: UInt64 = 0
                guard pread(fd, &length, 8, 0) == 8 else { throw Failure("\(path): no header") }
                var header = [UInt8](repeating: 0, count: Int(length))
                guard pread(fd, &header, Int(length), 8) == Int(length),
                      let json = try? JSONSerialization.jsonObject(with: Data(header)) as? [String: Any] else { throw Failure("\(path): bad header") }
                bases.append(8 + Int(length))
                for (key, v) in json where key.hasPrefix(prefix) {
                    let k = String(key.dropFirst(prefix.count))
                    guard t[k] == nil, let d = v as? [String: Any], let dtype = d["dtype"] as? String, let shape = d["shape"] as? [Int],
                          let offsets = d["data_offsets"] as? [Int] else { continue }
                    t[k] = Tensor(dtype: dtype, shape: shape, start: offsets[0], end: offsets[1], file: i)
                }
            }
        } catch {
            for fd in fds { close(fd) }
            throw error
        }
        (self.fds, self.bases, tensors) = (fds, bases, t)
    }

    deinit { for fd in fds { close(fd) } }

    /// The same shards opened again, past the system's cache or not, under the same prefix: the ring's reader's own.
    public func reopened(uncached: Bool) throws -> SafeTensors { try SafeTensors(shards: paths, uncached: uncached, under: prefix) }

    /// **The file's fingerprint**, for an asset that carries its weights as constants (its name must change with them):
    /// FNV-1a 64 of its header — each tensor's key, type, shape and place — and of 16 samples of 64 bytes spread over
    /// each tensor, its first and last among them. A fine-tune changes every tensor it trains throughout: two files of
    /// the same shapes whose weights differ give two fingerprints, at the cost of a few thousand small reads.
    public lazy var fingerprint: String = {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        func mix(_ bytes: UnsafeRawBufferPointer) { for byte in bytes { h = (h ^ UInt64(byte)) &* 0x100_0000_01b3 } }
        var sample = [UInt8](repeating: 0, count: 64)
        for (key, t) in tensors.sorted(by: { $0.key < $1.key }) {
            Array("\(key):\(t.dtype):\(t.shape):\(t.start):\(t.end);".utf8).withUnsafeBytes(mix)
            let count = t.end - t.start, n = min(64, count)
            guard n > 0 else { continue }
            let last = count - n
            for i in 0..<16 {
                let offset = last * i / 15
                sample.withUnsafeMutableBytes { _ = pread(fds[t.file], $0.baseAddress!, n, off_t(bases[t.file] + t.start + offset)) }
                sample.withUnsafeBytes { mix(UnsafeRawBufferPointer(rebasing: $0[0..<n])) }
            }
        }
        return String(h, radix: 16)
    }()

    /// The tensor `key`, of `dtype` and, unless nil, `shape`.
    public func tensor(_ key: String, _ dtype: String, _ shape: [Int]?) throws -> Tensor {
        guard let t = tensors[key] else { throw Failure("\(path): no tensor \(key)") }
        guard t.dtype == dtype, shape == nil || t.shape == shape! else { throw Failure("\(key): \(t.dtype) \(t.shape), expected \(dtype) \(shape.map { "\($0)" } ?? "")") }
        return t
    }

    /// A real tensor `key`, f32 or bf16 (the recipes read both: a model may keep its floats in either).
    public func real(_ key: String) throws -> Tensor {
        guard let t = tensors[key] else { throw Failure("\(path): no tensor \(key)") }
        guard t.dtype == "F32" || t.dtype == "BF16" else { throw Failure("\(key): \(t.dtype), expected F32 or BF16") }
        return t
    }

    /// The tensor's bytes into `into`.
    public func read(_ t: Tensor, into: UnsafeMutableRawPointer) throws {
        var done = 0
        let count = t.end - t.start
        while done < count {
            let n = pread(fds[t.file], into + done, min(count - done, 1 << 30), off_t(bases[t.file] + t.start + done))
            guard n > 0 else { throw Failure("\(paths[t.file]): read failed at \(t.start + done)") }
            done += n
        }
    }

    /// `count` bytes of `t` from its byte `offset` (a row of a table) into `into`.
    public func read(_ t: Tensor, at offset: Int, count: Int, into: UnsafeMutableRawPointer) throws {
        guard offset + count <= t.end - t.start, pread(fds[t.file], into, count, off_t(bases[t.file] + t.start + offset)) == count else {
            throw Failure("\(paths[t.file]): read failed at \(t.start + offset)")
        }
    }

    /// A small real tensor (f32 or bf16) as f32, exact: a gain, a scale vector the CPU reads.
    public func floats(_ key: String) throws -> [Float] {
        let t = try real(key), n = t.shape.reduce(1, *)
        var raw = [UInt8](repeating: 0, count: t.end - t.start)
        try raw.withUnsafeMutableBytes { try read(t, into: $0.baseAddress!) }
        return raw.withUnsafeBytes { b in
            t.dtype == "F32" ? Array(b.bindMemory(to: Float.self).prefix(n)) : b.bindMemory(to: UInt16.self).prefix(n).map { Float(bitPattern: UInt32($0) << 16) }
        }
    }

    /// The bytes of a small tensor as they lie (a note a publisher writes beside a weight).
    public func bytes(_ key: String) throws -> [UInt8] {
        guard let t = tensors[key] else { throw Failure("\(path): no tensor \(key)") }
        var raw = [UInt8](repeating: 0, count: t.end - t.start)
        try raw.withUnsafeMutableBytes { try read(t, into: $0.baseAddress!) }
        return raw
    }

}
