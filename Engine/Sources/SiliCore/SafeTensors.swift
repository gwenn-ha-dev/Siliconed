import Foundation

/// **A safetensors file, or the shards a model is published in**, read with `pread` straight into the arena: each file
/// `u64` header length · JSON header · raw tensors, the header's `data_offsets` relative to the end of the header. A
/// tensor knows its shard (`Tensor.file`): a group of weights may lie across two (a transformer's layers 15 and 35 cut
/// between two shards of its published files). Positional reads only: the ring's reader shares it.
///
/// **An F16 or F8 tensor is read as the F32 it holds** (`Tensor.Stored`): listed as F32, its bytes `[start, end)`
/// those of the f32s, widened as they are read. Every recipe then takes it as it takes a file's f32s — f16 where a
/// function reads f16, bf16 where it reads bf16, as torch loads a fine-tune saved in fp16 into a bf16 model.
///   - F16: exact (back to the file's values, to the bit, where a function reads f16). A fine-tune may keep its floats
///     in fp16: its author's int8 file with the unquantized weights in F16 was refused until 22/11/2026.
///   - F8_E4M3 times its one scale (`<key>_scale`, ComfyUI's « fp8 scaled »: `.weight` F8_E4M3, `.weight_scale` F32
///     `[]`): the code exact in an f32, the product rounded once to f32 — then to bf16 by a bf16 recipe, where
///     ComfyUI rounds the scale and the product in bf16. A scale of more than one value (by rows) is not read: the
///     tensor stays F8_E4M3, refused by name (`real`). No third arithmetic: an fp8 file
///     is read as a file of f32s would be, by the bf16 or the f16 functions (decided 22/11/2026).
public final class SafeTensors: @unchecked Sendable {
    public struct Tensor: Sendable {
        public let dtype: String
        public let shape: [Int]
        public let start: Int
        public let end: Int
        /// Its shard, by its rank in `paths`.
        public let file: Int
        /// What the file stores for a tensor listed as F32: its f32s, or narrower values widened as they are read.
        public enum Stored: Sendable, Equatable {
            case asListed
            case f16
            case f8(scale: Float)
            /// Bytes the file holds per value listed (4 as an f32).
            var width: Int { switch self { case .asListed: 4; case .f16: 2; case .f8: 1 } }
        }
        /// `dtype` says F32, `end − start` its f32 bytes; the file holds `end − start` · `width` / 4 of them from `start`.
        public let stored: Stored
        public init(dtype: String, shape: [Int], start: Int, end: Int, file: Int = 0, stored: Stored = .asListed) {
            self.dtype = dtype; self.shape = shape; self.start = start; self.end = end; self.file = file; self.stored = stored
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
                var file: [String: Tensor] = [:]
                for (key, v) in json where key.hasPrefix(prefix) {
                    let k = String(key.dropFirst(prefix.count))
                    guard t[k] == nil, let d = v as? [String: Any], let dtype = d["dtype"] as? String, let shape = d["shape"] as? [Int],
                          let offsets = d["data_offsets"] as? [Int] else { continue }
                    file[k] = Tensor(dtype: dtype, shape: shape, start: offsets[0], end: offsets[1], file: i)
                }
                for (k, raw) in file {
                    let stored: Tensor.Stored
                    switch raw.dtype {
                    case "F16": stored = .f16
                    case "F8_E4M3":
                        guard let scale = try Self.scale(file[k + "_scale"], fd: fd, base: bases[i]) else { t[k] = raw; continue }
                        stored = .f8(scale: scale)
                    default: t[k] = raw; continue
                    }
                    let n = raw.end - raw.start
                    t[k] = Tensor(dtype: "F32", shape: raw.shape, start: raw.start, end: raw.start + n * 4 / stored.width, file: i, stored: stored)
                }
            }
        } catch {
            for fd in fds { close(fd) }
            throw error
        }
        (self.fds, self.bases, tensors) = (fds, bases, t)
    }

    deinit { for fd in fds { close(fd) } }

    /// An F8 tensor's one scale, its `<key>_scale`: 1 when there is none, nil when it holds more than one value.
    private static func scale(_ t: Tensor?, fd: Int32, base: Int) throws -> Float? {
        guard let t else { return 1 }
        guard t.shape.reduce(1, *) == 1 else { return nil }
        var raw: UInt32 = 0
        let bytes = t.end - t.start
        guard bytes <= 4, pread(fd, &raw, bytes, off_t(base + t.start)) == bytes else { throw Failure("\(t): unreadable scale") }
        switch t.dtype {
        case "F32": return Float(bitPattern: raw)
        case "BF16": return Float(bitPattern: raw << 16)
        case "F16": return Float(Float16(bitPattern: UInt16(truncatingIfNeeded: raw)))
        default: return nil
        }
    }

    /// The same shards opened again, past the system's cache or not, under the same prefix: the ring's reader's own.
    public func reopened(uncached: Bool) throws -> SafeTensors { try SafeTensors(shards: paths, uncached: uncached, under: prefix) }

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

    /// The tensor's bytes into `into` (an F16 or F8 tensor's as f32s, `Tensor.Stored`).
    public func read(_ t: Tensor, into: UnsafeMutableRawPointer) throws {
        try read(t, at: 0, count: t.end - t.start, into: into)
    }

    /// `count` bytes of `t` from its byte `offset` (a row of a table) into `into`.
    public func read(_ t: Tensor, at offset: Int, count: Int, into: UnsafeMutableRawPointer) throws {
        guard offset + count <= t.end - t.start else { throw Failure("\(paths[t.file]): read failed at \(t.start + offset)") }
        guard t.stored != .asListed else { return try readStored(t, from: offset, count: count, into: into) }
        guard offset % 4 == 0, count % 4 == 0 else { throw Failure("\(paths[t.file]): a narrow tensor read at f32 byte \(offset), \(count) bytes") }
        // The stored values into the top of the place, then widened from the front: of n values w bytes wide, the f32 i
        // overwrites stored values from (4i − (4 − w)·n) / w on, up to i at most (i < n): each already read.
        let w = t.stored.width, n = count / 4, top = into + (4 - w) * n
        try readStored(t, from: offset / 4 * w, count: n * w, into: top)
        let from = UnsafeRawPointer(top)
        switch t.stored {
        case .asListed: break
        case .f16:
            for i in 0..<n {
                into.storeBytes(of: Float(Float16(bitPattern: from.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self))), toByteOffset: 4 * i, as: Float.self)
            }
        case .f8(let scale):
            let codes = Self.e4m3
            for i in 0..<n { into.storeBytes(of: codes[Int(from.load(fromByteOffset: i, as: UInt8.self))] * scale, toByteOffset: 4 * i, as: Float.self) }
        }
    }

    /// The 256 values of float8 e4m3fn (1 sign, 4 exponent bits biased by 7, 3 mantissa bits; no infinity, S.1111.111 NaN),
    /// each exact in an f32.
    static let e4m3: [Float] = (0..<256).map { b in
        let sign: Float = b & 0x80 != 0 ? -1 : 1, e = (b >> 3) & 0xF, m = b & 7
        if e == 0xF && m == 7 { return .nan }
        return sign * (e == 0 ? Float(m) / 8 * 0x1p-6 : (1 + Float(m) / 8) * Float(sign: .plus, exponent: e - 7, significand: 1))
    }

    /// `count` bytes of the file from `t`'s byte `from`, as they lie.
    private func readStored(_ t: Tensor, from: Int, count: Int, into: UnsafeMutableRawPointer) throws {
        var done = 0
        while done < count {
            let n = pread(fds[t.file], into + done, min(count - done, 1 << 30), off_t(bases[t.file] + t.start + from + done))
            guard n > 0 else { throw Failure("\(paths[t.file]): read failed at \(t.start + from + done)") }
            done += n
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
