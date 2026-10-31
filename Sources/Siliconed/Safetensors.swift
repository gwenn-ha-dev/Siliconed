import Darwin
import Foundation

/// **A checkpoint, read-only**: the golden tensors, the published sources the forge reads, the files an import is given, the VAEs.
/// Eight bytes of length, a JSON header, then the bytes. We map it rather than read it — the golden tensors at 1024² weigh 764 MB, and nothing
/// requires copying them to compare them.
///
/// A **GGUF** file (`"GGUF"` magic) opens through the same door (`GGUF.parse`): its
/// tensors become entries under their published names, in torch shape (`ne` reversed), dtype
/// `F32`, `F16`, `BF16`, `Q8_0` (interleaved blocks, which `QuantizedLayouts` reads) or one of the
/// packed 4- to 6-bit types (`Q4_0`…`Q6_K`, kept whole); its key/values become `metadata`. The forge and an import see one more checkpoint format, not a
/// second reader.
package final class Safetensors {
    package struct Entry {
        package let name: String
        package let dtype: String
        package let shape: [Int]
        package let offset: Int          // absolute, within the file
        package let bytes: Int
        package var count: Int { shape.reduce(1, *) }
    }

    package enum Failure: Error, CustomStringConvertible {
        /// `open` failed (`ENOENT` is thrown as a `MissingFile` before this), with its `errno`.
        case cannotOpen(String, Int32)
        /// `fstat` failed on an open descriptor, with its `errno`.
        case cannotStat(String, Int32)
        /// `mmap` refused the file, with its `errno` (`ENOMEM`: no address space for it).
        case cannotMap(String, Int32)
        case badHeader(String)
        case unknownDType(String, String)
        /// A tensor that a graph asks for and the file does not have (or not in a readable dtype).
        case missingTensor(file: String, name: String)
        package var description: String {
            switch self {
            case let .cannotOpen(p, code): return "safetensors: cannot open \(p): \(String(cString: strerror(code)))"
            case let .cannotStat(p, code): return "safetensors: cannot stat \(p): \(String(cString: strerror(code)))"
            case let .cannotMap(p, code): return "safetensors: cannot map \(p): \(String(cString: strerror(code)))"
            case .badHeader(let why): return "safetensors: header — \(why)"
            case .unknownDType(let n, let d): return "safetensors: \(n) has an unknown dtype \(d)"
            case .missingTensor(let f, let n): return "safetensors: \(f) — \(n) missing or of unknown dtype"
            }
        }
    }

    package private(set) var entries: [String: Entry] = [:]
    /// `__metadata__`, if any: what the trainer declared (name, alpha, base model…).
    package private(set) var metadata: [String: String] = [:]
    /// The names in file order (that of the bytes) — a dictionary has none.
    package private(set) var order: [String] = []
    package let path: String
    /// A GGUF file: its format version and alignment. `nil`: a safetensors.
    package private(set) var gguf: (version: Int, alignment: Int)?
    private let base: UnsafeRawPointer
    private let size: Int
    private let descriptor: Int32

    package init(path: String) throws {
        self.path = path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            let code = errno
            if code == ENOENT { throw MissingFile(path, .published) }
            throw Failure.cannotOpen(path, code)
        }
        var info = stat()
        guard fstat(fd, &info) == 0 else { let code = errno; close(fd); throw Failure.cannotStat(path, code) }
        // Eight bytes of length and at least `{}`: anything shorter is not a checkpoint, whatever
        // its name — said as such, not as a file that cannot be opened.
        guard info.st_size > 8 else {
            close(fd)
            throw Failure.badHeader("\(path) is \(info.st_size) bytes long, too short to be a safetensors or GGUF file")
        }
        size = Int(info.st_size)
        guard let mapped = mmap(nil, size, PROT_READ, MAP_PRIVATE | MAP_FILE, fd, 0), mapped != MAP_FAILED
        else { let code = errno; close(fd); throw Failure.cannotMap(path, code) }
        descriptor = fd
        base = UnsafeRawPointer(mapped)

        if size >= 4, Array(UnsafeRawBufferPointer(start: base, count: 4)) == GGUF.magic {
            let h = try GGUF.parse(base, size: size)
            gguf = (h.version, h.alignment)
            metadata = h.metadata
            for e in h.entries { entries[e.name] = e }
            order = h.entries.sorted { $0.offset < $1.offset }.map(\.name)
            return
        }

        // Every number below comes from the file: each is bounded before it is used, and every
        // product is checked, so that a hostile header is refused by name instead of trapping or
        // pointing outside the mapping.
        let declared = base.load(as: UInt64.self)
        guard let headerLength = Int(exactly: declared), headerLength <= Safetensors.maximumHeader,
              headerLength <= size - 8 else {
            throw Failure.badHeader("length \(declared) (the file has \(size) bytes; the format allows "
                                    + "\(Safetensors.maximumHeader) bytes of header at most)")
        }
        let dataStart = 8 + headerLength, dataBytes = size - dataStart
        let json = Data(bytes: base.advanced(by: 8), count: headerLength)
        guard let parsed = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw Failure.badHeader("not a JSON object")
        }
        if let meta = parsed["__metadata__"] as? [String: Any] {
            for (k, v) in meta { metadata[k] = v as? String ?? "\(v)" }
        }
        for (name, value) in parsed where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int],
                  let bounds = entry["data_offsets"] as? [Int], bounds.count == 2 else {
                throw Failure.badHeader("\(name) is incomplete")
            }
            let width: Int
            switch dtype {
            case "F32": width = 4
            case "F16", "BF16", "I16", "U16": width = 2
            case "I32", "U32": width = 4
            case "I64", "U64", "F64": width = 8
            // What quantized checkpoints publish (ComfyUI "scaled" fp8…): the forge
            // knows how to read them; the engine only ever reads its own maps.
            // `F8_E8M0`: mxfp8's block exponents — read only to be refused by name.
            case "F8_E4M3", "F8_E5M2", "F8_E8M0", "I8", "U8", "BOOL": width = 1
            default: throw Failure.unknownDType(name, dtype)
            }
            guard 0 <= bounds[0], bounds[0] <= bounds[1], bounds[1] <= dataBytes else {
                throw Failure.badHeader("\(name): data_offsets \(bounds) outside the \(dataBytes) bytes of data")
            }
            guard shape.allSatisfy({ $0 >= 0 }), let expected = Safetensors.byteCount(shape, width: width) else {
                throw Failure.badHeader("\(name): shape \(shape) as \(dtype) is not a size")
            }
            let bytes = bounds[1] - bounds[0]
            guard bytes == expected else {
                throw Failure.badHeader("\(name): \(bytes) bytes for \(shape) as \(dtype)")
            }
            entries[name] = Entry(name: name, dtype: dtype, shape: shape,
                                  offset: dataStart + bounds[0], bytes: bytes)
        }
        order = entries.values.sorted { $0.offset < $1.offset }.map(\.name)
    }

    /// The safetensors specification's ceiling on the JSON header (100 MB): no real checkpoint
    /// comes near it, and it keeps a forged length from copying the whole file into a `Data`.
    package static let maximumHeader = 100_000_000

    /// `shape`'s element count times `width`, or `nil` if the product does not fit an `Int`.
    package static func byteCount(_ shape: [Int], width: Int) -> Int? {
        var n = width
        for d in shape {
            let (p, overflow) = n.multipliedReportingOverflow(by: d)
            guard !overflow else { return nil }
            n = p
        }
        return n
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), size)
        close(descriptor)
        if let fd = uncachedOpened, fd >= 0 { close(fd) }
    }

    package func pointer(_ name: String) -> UnsafeRawPointer? {
        guard let e = entries[name] else { return nil }
        return base.advanced(by: e.offset)
    }

    /// **A tensor's bytes read with `pread`, not through the mapping** — what the forge reads a
    /// source with, once. Measured on a cold file on the M1 Pro: a fault per 16 KB page of the mapping reads
    /// 0.56 GB/s on one thread, 0.95 on eight; `pread` by 8 MB, 4.5 GB/s on one. The descriptor is
    /// `F_NOCACHE`: a source read once (tens of GB at an installation) does not push the user's
    /// pages out of memory for nothing — a page already in the cache is still read from it.
    package func copy(_ e: Entry, into destination: UnsafeMutableRawPointer) throws {
        let fd = uncachedDescriptor
        guard fd >= 0 else { throw Failure.cannotOpen(path, errno) }
        var done = 0
        while done < e.bytes {
            let n = pread(fd, destination + done, min(e.bytes - done, Safetensors.readBlock), off_t(e.offset + done))
            guard n > 0 else { throw Failure.badHeader("\(path): \(e.name) cut short at byte \(done) of \(e.bytes)") }
            done += n
        }
    }

    /// The same into a new array.
    package func bytes(_ e: Entry) throws -> [UInt8] {
        try [UInt8](unsafeUninitializedCapacity: e.bytes) { b, n in
            try copy(e, into: b.baseAddress!)
            n = e.bytes
        }
    }

    static let readBlock = 8 << 20
    /// Opened at the first `copy`, from any thread (the forge produces several tensors at once).
    private var uncachedOpened: Int32?
    private let opening = NSLock()
    private var uncachedDescriptor: Int32 {
        opening.withLock {
            if let fd = uncachedOpened { return fd }
            let fd = open(path, O_RDONLY)
            if fd >= 0 { _ = fcntl(fd, F_NOCACHE, 1) }
            uncachedOpened = fd
            return fd
        }
    }

    /// Materializes a tensor as fp32, whatever its dtype on disk. The VAE is published in
    /// bf16: reading it as fp32 would give finite and wrong numbers (the step 2 mistake).
    package func materialize(_ name: String) -> [Float]? {
        guard let e = entries[name] else { return nil }
        var out = [Float](repeating: 0, count: e.count)
        guard e.count == 0 || out.withUnsafeMutableBufferPointer({
            materialize(name, into: $0.baseAddress!, capacity: e.count) }) != nil else { return nil }
        return out
    }

    /// **`materialize` into a caller's buffer** of `capacity` floats — the same widenings, written in
    /// place (an arena slice: `AnimaConditioner`). Returns the count written; `nil` for a tensor that
    /// is missing, of another dtype, or larger than `capacity` (refused before a value is written).
    package func materialize(_ name: String, into out: UnsafeMutablePointer<Float>, capacity: Int) -> Int? {
        guard let e = entries[name], let p = pointer(name), e.count <= capacity else { return nil }
        switch e.dtype {
        case "F32": UnsafeMutableRawPointer(out).copyMemory(from: p, byteCount: e.bytes)
        case "BF16": Widen.bfloat16ToFloat32(source: p, destination: UnsafeMutableRawPointer(out), count: e.count)
        case "F16":
            let half = p.assumingMemoryBound(to: Float16.self)
            for i in 0..<e.count { out[i] = Float(half[i]) }
        default: return nil
        }
        return e.count
    }

    /// The integers of a golden, without conversion. `input_ids` and `attention_mask` are published as
    /// `I32`: materializing them as fp32 would work — 151,643 fits exactly in a `Float` —
    /// but comparing tokens through floats is an invitation not to see a one-bit error
    /// on a vocabulary identifier.
    package func int32(_ name: String) -> UnsafeBufferPointer<Int32>? {
        guard let e = entries[name], e.dtype == "I32" || e.dtype == "U32",
              let p = pointer(name) else { return nil }
        return UnsafeBufferPointer(start: p.assumingMemoryBound(to: Int32.self), count: e.count)
    }

    /// An fp32 view, without a copy. Fails if the tensor is not F32 — we do not convert
    /// silently: an oracle that changes type along the way is no longer an oracle.
    package func float32(_ name: String) -> UnsafeBufferPointer<Float>? {
        guard let e = entries[name], e.dtype == "F32", let p = pointer(name) else { return nil }
        return UnsafeBufferPointer(start: p.assumingMemoryBound(to: Float.self), count: e.count)
    }
}
