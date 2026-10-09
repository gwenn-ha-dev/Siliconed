import Darwin
import Foundation

/// A forged artifact, mapped read-only.
///
/// The map is the engine's only large allocation that is not the arena, and it is deliberately
/// file-backed: the kernel can reclaim a clean file page only by *discarding* it, never by writing
/// it to swap. That is the whole of the no-swap guarantee — not a budget, a property of the pages.
///
/// Every tensor starts on a page boundary, so a mapped page is a piece of one tensor and never two
/// halves of different ones. The file is laid out in execution order, so reading it forward is
/// running the model, which is what makes prefetching a `madvise` and not a scheduler.
package final class Artifact {
    package struct Tensor {
        package let name: String
        package let offset: Int          // absolute, in the file
        package let bytes: Int
        package let shape: [Int]
        package let dtype: DType
        /// The scale of an 8-bit tensor (`int8`, `float8_e4m3`); `nil` for the others, and for an
        /// fp8 published without one (a plain cast: `w = f8`).
        package let scale: Scale?
        /// A rotation of the input to undo after the scale (ComfyUI's int8 `convrot`):
        /// `nil` for every other tensor.
        package var rotation: Rotation? = nil

        package var count: Int { shape.reduce(1, *) }
        package var isQuantized: Bool { dtype.isQuantized }
    }

    /// **The scale of an 8-bit tensor, stored inside the tensor's own bytes**: the values, then — on
    /// the next 16-byte boundary — the scales, in the type they were published in. One region, one
    /// page-aligned `offset`/`bytes`: the prefetcher, the streamed tail (`TailStream`), the cache
    /// hints and the GPU wrapper all see a tensor and its scale as one thing, so they can never be
    /// read apart. Shapes: `[]` (one for the tensor), `[N]` (one per column of `[K, N]`, i.e. per
    /// output), `[K/b, N]` with `block` = b (per column and per block of b rows — GGUF Q8_0).
    package struct Scale: Equatable {
        /// From the start of the tensor, a multiple of 16.
        package let offset: Int
        package let dtype: DType
        package let shape: [Int]
        package let block: Int?
        package var count: Int { shape.reduce(1, *) }

        /// Where `Widen.dequantize` finds the scale of value `(r, c)`.
        package var layout: Widen.ScaleLayout {
            switch shape.count {
            case 0: return .tensor
            case 1: return .columns(shape[0])
            default: return .blocks(columns: shape[1], rows: block ?? 1)
            }
        }
    }

    /// **A published rotation of the input** (`"rotation": {"kind": "convrot", "group": 256}` in
    /// the tensor's entry): the map keeps `q` `[K, N]` and its scale `[N]` as published, and the
    /// weight is `W = (q · s) · R` along K — `Widen.dequantizeRotated`. Only `convrot` exists; any
    /// other kind refuses the map: a rotation read as nothing would give a plausible, wrong weight.
    package struct Rotation: Equatable {
        package let kind: String
        package let group: Int
    }

    package enum DType: String {
        case bfloat16, float16, float32
        /// Published 8-bit weights, kept as they came: `w = Float(q) · s`.
        case int8, float8_e4m3
        /// **GGUF blocks, kept as published**: ggml's legacy blocks of 32 values
        /// (`q4_0`…`q5_1`) and K-quant super-blocks of 256 (`q4_k`…`q6_k`), copied whole, bytes
        /// unchanged — `Widen.dequantizePacked`. A tensor of these types has the map's shape `[K, N]`
        /// (what `materialize` writes, like every `Linear`), but its bytes keep the **published**
        /// layout: `N` rows (outputs), each `K / values` blocks in file order. The engine transposes
        /// while it dequantizes. No `scale` entry: the scales are inside the blocks.
        case q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k
        /// Bytes per value — **not for a packed type**, whose values have no size of their own
        /// (`packedBlock`); every caller branches on that first.
        package var size: Int {
            switch self {
            case .float32: return 4
            case .bfloat16, .float16: return 2
            case .int8, .float8_e4m3: return 1
            case .q4_0, .q4_1, .q5_0, .q5_1, .q4_k, .q5_k, .q6_k:
                preconditionFailure("\(self): packed blocks, no size per value")
            }
        }
        /// An 8-bit type, with a `scale` beside its values. Not the packed GGUF types.
        package var isQuantized: Bool { self == .int8 || self == .float8_e4m3 }
        /// One ggml block: its values along a row and its bytes (`ggml-common.h`: `block_q4_0`…
        /// `block_q5_1`, `QK4_0` = … = 32; `block_q4_K`…`block_q6_K`, `QK_K` = 256); `nil`: not packed.
        package var packedBlock: (values: Int, bytes: Int)? {
            switch self {
            case .q4_0: return (32, 18)
            case .q4_1: return (32, 20)
            case .q5_0: return (32, 22)
            case .q5_1: return (32, 24)
            case .q4_k: return (256, 144)
            case .q5_k: return (256, 176)
            case .q6_k: return (256, 210)
            default: return nil
            }
        }
        package var isPacked: Bool { packedBlock != nil }
    }

    package enum Failure: Error, CustomStringConvertible {
        case cannotOpen(String, Int32)
        case badMagic([UInt8])
        case truncated(String)
        case badHeader(String)
        case unaligned(name: String, offset: Int, page: Int)
        /// A tensor with more values than the buffer the model reads it into (`materialize`): the
        /// map is not the one this model expects. Refused before a single value is written.
        case doesNotFit(name: String, count: Int, capacity: Int)
        /// **Not a fault of the map**: the engine asked something of itself it cannot do (a sequence
        /// beyond what it reserved, a call out of order). Folded into `internalFailure`, not into a
        /// damaged file — said as "map: header", these sent the user to reinstall a model that was fine.
        case misuse(String)
        /// One of the above, raised while opening `path` — what lets the door name the damaged file.
        indirect case inFile(String, Failure)

        package var description: String {
            switch self {
            case let .inFile(path, failure): return "\(path): \(failure)"
            case .cannotOpen(let path, let code): return "cannot open \(path): \(String(cString: strerror(code)))"
            case .badMagic(let bytes): return "not a Siliconed map (signature \(bytes))"
            case .truncated(let what): return "truncated map: \(what)"
            case .badHeader(let why): return "map: header — \(why)"
            case .unaligned(let name, let offset, let page):
                return "\(name) starts at \(offset), not a multiple of \(page) — the mapping would straddle two pages"
            case let .doesNotFit(name, count, capacity):
                return "\(name): \(count) values for a buffer of \(capacity) — not the map this model reads"
            case .misuse(let why): return "engine: \(why)"
            }
        }
    }

    /// **The header alone** — 16 bytes then the JSON, without mapping the tensors: what the
    /// catalog reads to describe a LoRA without opening it.
    package static func header(_ path: String) throws -> [String: Any] {
        do { return try readHeader(path) } catch let e as Failure { throw naming(path, e) }
    }

    /// A reader's failure, with the file it was reading (`cannotOpen` already has it).
    private static func naming(_ path: String, _ failure: Failure) -> Failure {
        switch failure {
        case .cannotOpen, .inFile: return failure
        default: return .inFile(path, failure)
        }
    }

    private static func readHeader(_ path: String) throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: path) else { throw MissingFile(path, .map) }
        guard let file = FileHandle(forReadingAtPath: path) else { throw Failure.cannotOpen(path, errno) }
        defer { try? file.close() }
        let prologue = [UInt8](file.readData(ofLength: 16))
        guard prologue.count == 16 else { throw Failure.truncated("less than a prologue") }
        guard Array(prologue.prefix(7)) == signature, supportedVersions.contains(prologue[7]) else {
            throw Failure.badMagic(Array(prologue.prefix(8)))
        }
        let raw = prologue[8..<16].enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * UInt64($1.offset)) }
        // The length comes from the file, so the file bounds it before anything is allocated for
        // it: a damaged prologue must not ask `readData` for exabytes.
        var info = stat()
        guard fstat(file.fileDescriptor, &info) == 0 else { throw Failure.cannotOpen(path, errno) }
        guard let length = headerLength(raw, fileSize: Int(info.st_size)) else {
            throw Failure.truncated("header length \(raw)")
        }
        let json = file.readData(ofLength: length)
        guard json.count == length,
              let object = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw Failure.badHeader("not a JSON object")
        }
        return object
    }

    /// `SILICON` then a version byte. v2: bf16 map — the weights published as `float32` are
    /// bf16 values, so bf16 carries them bit for bit and widening is a shift.
    /// v3: transposed `Linear` weights, `[input, output]`.
    ///
    /// The reader accepts a **range** and exposes the version: what the map carries is written in
    /// the map, not in the code that reads it.
    private static let signature: [UInt8] = Array("SILICON".utf8)
    private static let supportedVersions: ClosedRange<UInt8> = 2...3
    package private(set) var version: UInt8 = 0

    package let path: String
    package let size: Int
    package let page: Int
    package let header: [String: Any]
    package private(set) var tensors: [String: Tensor] = [:]
    /// Tensor names in the order the file lays them out, which is the order the engine reads them.
    package private(set) var order: [String] = []

    private let base: UnsafeRawPointer
    private let descriptor: Int32

    package convenience init(path: String) throws {
        do { try self.init(opening: path) } catch let e as Failure { throw Self.naming(path, e) }
    }

    private init(opening path: String) throws {
        self.path = path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            if errno == ENOENT { throw MissingFile(path, .map) }
            throw Failure.cannotOpen(path, errno)
        }
        var info = stat()
        guard fstat(fd, &info) == 0 else { let code = errno; close(fd); throw Failure.cannotOpen(path, code) }
        let size = Int(info.st_size)
        guard size > 16 else { close(fd); throw Failure.truncated("shorter than a prologue") }
        guard let mapped = mmap(nil, size, PROT_READ, MAP_PRIVATE | MAP_FILE, fd, 0),
              mapped != MAP_FAILED else {
            let code = errno; close(fd); throw Failure.cannotOpen(path, code)
        }
        // Everything is read into locals, and the map is given back on every failure: `deinit`
        // only runs for an object whose properties were all set, so a throw from here on would
        // otherwise leak the mapping and the descriptor.
        let parsed: Parsed
        do {
            parsed = try Self.parse(UnsafeRawPointer(mapped), size: size)
        } catch {
            munmap(mapped, size)
            close(fd)
            throw error
        }
        self.size = size
        self.descriptor = fd
        self.base = UnsafeRawPointer(mapped)
        self.version = parsed.version
        self.header = parsed.header
        self.page = parsed.page
        self.order = parsed.order
        self.tensors = parsed.tensors
    }

    /// What `init(opening:)` reads from the mapped bytes, every number bounded before it is used.
    private struct Parsed {
        let version: UInt8
        let header: [String: Any]
        let page: Int
        let order: [String]
        let tensors: [String: Tensor]
    }

    /// **The header's length, if the file can hold it.** The prologue's 8 bytes are a `UInt64`: a
    /// damaged one must neither trap on the conversion to `Int` nor overflow `16 + length`.
    package static func headerLength(_ raw: UInt64, fileSize: Int) -> Int? {
        guard fileSize >= 16, raw <= UInt64(fileSize - 16) else { return nil }
        return Int(raw)
    }

    /// **The number of values of a shape**; `nil` for a negative dimension or a product that overflows.
    package static func elementCount(_ shape: [Int]) -> Int? {
        var count = 1
        for dimension in shape {
            guard dimension >= 0 else { return nil }
            let (product, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard !overflow else { return nil }
            count = product
        }
        return count
    }

    private static func parse(_ base: UnsafeRawPointer, size: Int) throws -> Parsed {
        let head = base.assumingMemoryBound(to: UInt8.self)
        let got = Array(UnsafeBufferPointer(start: head, count: 8))
        guard Array(got.prefix(7)) == Self.signature,
              Self.supportedVersions.contains(got[7]) else { throw Failure.badMagic(got) }

        let rawLength = base.loadUnaligned(fromByteOffset: 8, as: UInt64.self)
        guard let jsonLength = headerLength(rawLength, fileSize: size) else {
            throw Failure.truncated("header length \(rawLength)")
        }
        let json = Data(bytes: base.advanced(by: 16), count: jsonLength)
        guard let parsed = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw Failure.badHeader("not a JSON object")
        }

        guard let page = parsed["page"] as? Int else { throw Failure.badHeader("no page size") }
        // `offset % page` below: a page of 0 would trap, a negative one would accept anything.
        guard page > 0 else { throw Failure.badHeader("page size \(page)") }
        guard let table = parsed["tensors"] as? [String: [String: Any]] else {
            throw Failure.badHeader("no tensor table")
        }
        let order = (parsed["order"] as? [String]) ?? table.keys.sorted()

        var tensors: [String: Tensor] = [:]
        for (name, entry) in table {
            guard let offset = entry["offset"] as? Int,
                  let bytes = entry["bytes"] as? Int,
                  let shape = entry["shape"] as? [Int],
                  let raw = entry["dtype"] as? String, let dtype = DType(rawValue: raw) else {
                throw Failure.badHeader("tensor \(name) is incomplete")
            }
            // Signs first: a negative offset that is a multiple of the page passes the alignment
            // test and points before the map.
            guard offset >= 0, bytes >= 0 else {
                throw Failure.badHeader("\(name): offset \(offset), \(bytes) bytes")
            }
            // The alignment claim is checked, not trusted: it is what the no-swap and prefetch
            // stories both rest on.
            guard offset % page == 0 else { throw Failure.unaligned(name: name, offset: offset, page: page) }
            guard offset <= size, bytes <= size - offset else { throw Failure.truncated("\(name) runs past the end") }
            guard let count = elementCount(shape) else { throw Failure.badHeader("\(name): shape \(shape)") }
            var scale: Scale?
            if let s = entry["scale"] {
                scale = try Self.scale(s, of: name, shape: shape, dtype: dtype)
            }
            let expected: (Int, Bool)
            if let block = dtype.packedBlock {
                // `[K, N]`, stored as N rows of K/b blocks: K must be whole blocks.
                guard shape.count == 2, shape[0] % block.values == 0, entry["scale"] == nil,
                      entry["rotation"] == nil else {
                    throw Failure.badHeader("\(name): \(raw) on \(shape) — only a `[K, N]` whose K is a multiple of "
                                            + "\(block.values), without a scale or a rotation")
                }
                let total = (count / block.values).multipliedReportingOverflow(by: block.bytes)
                expected = (total.partialValue, total.overflow)
            } else if let scale {
                let (scaleBytes, overflow) = scale.count.multipliedReportingOverflow(by: scale.dtype.size)
                let total = scale.offset.addingReportingOverflow(scaleBytes)
                expected = (total.partialValue, overflow || total.overflow)
            } else {
                let total = count.multipliedReportingOverflow(by: dtype.size)
                expected = (total.partialValue, total.overflow)
            }
            guard !expected.1, bytes == expected.0 else {
                throw Failure.badHeader("\(name): \(bytes) bytes for \(shape) of \(raw)")
            }
            var tensor = Tensor(name: name, offset: offset, bytes: bytes, shape: shape, dtype: dtype, scale: scale)
            if let r = entry["rotation"] {
                tensor.rotation = try Self.rotation(r, of: name, shape: shape, dtype: dtype, scale: scale)
            }
            tensors[name] = tensor
        }
        guard Set(order) == Set(tensors.keys) else {
            throw Failure.badHeader("`order` and `tensors` disagree")
        }
        return Parsed(version: got[7], header: parsed, page: page, order: order, tensors: tensors)
    }

    /// An 8-bit tensor's `scale` entry, checked against its shape: a scale read with the wrong
    /// period would give every weight a plausible, wrong factor.
    private static func scale(_ raw: Any, of name: String, shape: [Int], dtype: DType) throws -> Scale {
        guard dtype.isQuantized, let s = raw as? [String: Any], let offset = s["offset"] as? Int,
              let t = s["dtype"] as? String, let scaleType = DType(rawValue: t),
              [.bfloat16, .float16, .float32].contains(scaleType),
              let scaleShape = s["shape"] as? [Int] else {
            throw Failure.badHeader("\(name): malformed scale")
        }
        let block = s["block"] as? Int
        guard let count = elementCount(shape), elementCount(scaleShape) != nil else {
            throw Failure.badHeader("\(name): a scale \(scaleShape) for \(shape)")
        }
        let fits: Bool
        switch scaleShape.count {
        case 0: fits = block == nil
        case 1: fits = shape.count == 2 && scaleShape[0] == shape[1] && block == nil
        case 2: fits = shape.count == 2 && scaleShape[1] == shape[1] && (block ?? 0) > 0
                    && elementCount([scaleShape[0], block!]) == shape[0]
        default: fits = false
        }
        guard fits, offset >= count, offset % 16 == 0 else {
            throw Failure.badHeader("\(name): a scale \(scaleShape) (block \(block.map(String.init) ?? "-")) "
                                    + "at \(offset) does not fit \(shape)")
        }
        return Scale(offset: offset, dtype: scaleType, shape: scaleShape, block: block)
    }

    /// A tensor's `rotation`, checked: only ComfyUI's `convrot`, on an int8 `[K, N]` with a scale
    /// per column or for the tensor, its group a power of 4 dividing K.
    private static func rotation(_ raw: Any, of name: String, shape: [Int], dtype: DType, scale: Scale?) throws -> Rotation {
        guard let r = raw as? [String: Any], let kind = r["kind"] as? String, let group = r["group"] as? Int else {
            throw Failure.badHeader("\(name): malformed rotation")
        }
        guard kind == "convrot" else {
            throw Failure.badHeader("\(name): rotation \"\(kind)\" unknown — refused rather than read without it")
        }
        guard dtype == .int8, shape.count == 2, let scale, scale.block == nil, scale.shape.count <= 1,
              Widen.isRotationGroup(group), shape[0] % group == 0 else {
            throw Failure.badHeader("\(name): rotation convrot of group \(group) on \(dtype.rawValue) \(shape) — not read")
        }
        return Rotation(kind: kind, group: group)
    }

    deinit {
        stream = nil   // its reads finish before the descriptor closes
        through = nil
        munmap(UnsafeMutableRawPointer(mutating: base), size)
        close(descriptor)
    }

    /// A pointer to the weight. Into the map, without a copy — except for a tensor of the streamed
    /// tail that the prefetcher has staged (`TailStream`): then its staging copy, once read; and, in
    /// `readThrough` mode, always a copy.
    package func pointer(_ name: String) -> UnsafeRawPointer? {
        guard let t = tensors[name] else { return nil }
        if let stream, let staged = stream.pointer(name, offset: t.offset) { return staged }
        if let through { return through.read(name, t, page: page) }
        return base.advanced(by: t.offset)
    }

    /// **Never through the page cache — for a tool that reads a whole map once** (a developer
    /// check, never the engine). Every `pointer` then returns a copy of the tensor read
    /// with `pread` on an `F_NOCACHE` descriptor into one reused anonymous buffer (the last tensor
    /// read; `materialize` and the GPU wrapper of the same tensor share it). Reading 6–7 GB of map
    /// through the mapping left every page in the unified buffer cache — `MADV_DONTNEED` on a file
    /// mapping does not take them out (`TailStream` measured it with `mincore`) — and the kernel
    /// compressed then swapped the other processes' memory to make room: +18 524 swapouts for a check
    /// with 840 MB of RSS, on a quiet machine. Not `streamTail`: `WidenGPU` declines a streamed
    /// tensor (its slot is reused under the engine's prefetcher), and the check reads on the GPU too.
    package func readThrough() throws {
        guard through == nil else { return }
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw Failure.cannotOpen(path, errno) }
        _ = fcntl(fd, F_NOCACHE, 1)
        through = ReadThrough(descriptor: fd)
    }
    private var through: ReadThrough?
    /// In `readThrough` mode, a pointer lives until the next tensor is read: nothing may keep it.
    package var readsThrough: Bool { through != nil }

    /// The pointer into the map itself, never a staging copy: for the prefetcher's `madvise`.
    package func mappedPointer(_ name: String) -> UnsafeRawPointer? {
        tensors[name].map { base.advanced(by: $0.offset) }
    }

    /// **The streamed tail**: these blocks are read with `pread` + `F_NOCACHE` into staging
    /// buffers rather than paged in through the map, so they never enter the page cache, and the
    /// head of the map — what the cache can hold — survives from one evaluation to the next. Under
    /// LRU, a map larger than the cache swept in file order otherwise misses every page, every time.
    /// `blocks` are the tail's blocks in file order, each a run of contiguous tensors; `slots`
    /// staging buffers of the largest block are allocated (anonymous memory: measured, not assumed).
    /// Idempotent for the same blocks and buffers.
    package func streamTail(blocks: [[String]], slots: Int) throws {
        guard !blocks.isEmpty, slots > 0 else { stream = nil; return }
        if let stream, stream.blocks == blocks, stream.slotCount == slots { return }
        stream = nil
        stream = try TailStream(artifact: self, blocks: blocks, slots: slots)
    }
    package private(set) var stream: TailStream?

    /// Asks the kernel to bring a range in before it is needed. The access order is known ahead
    /// (it is the file order), so prefetching needs no policy — only a hint, issued early enough.
    package func prefetch(_ name: String) {
        guard let t = tensors[name] else { return }
        let start = base.advanced(by: t.offset)
        _ = madvise(UnsafeMutableRawPointer(mutating: start), t.bytes, MADV_WILLNEED)
    }

    /// **The opposite hint: these pages will not be read again soon.** `MADV_DONTNEED` moves them to
    /// the front of the kernel's reclaim queue — they are clean and file-backed, dropped without a
    /// write — instead of letting them age behind the other processes' memory, which the kernel would
    /// otherwise compress, then swap, to make room for the next weights. For a map larger than what the
    /// machine can cache (Qwen-Image-2.1: 14.2 GB of DiT, 16.3 GB of encoder), keeping a page read
    /// once per evaluation saves nothing: it is evicted before it is read again.
    package func dropFromCache(_ names: [String]) {
        // A staged block gives its buffer back: its pages never were in the cache.
        if let stream, stream.release(names) { return }
        let page = Int(getpagesize())
        for name in names {
            guard let t = tensors[name] else { continue }
            // Whole pages inside the tensor only: its neighbours' pages may still be in use.
            let first = (t.offset + page - 1) / page * page, end = (t.offset + t.bytes) / page * page
            guard end > first else { continue }
            _ = madvise(UnsafeMutableRawPointer(mutating: base.advanced(by: first)), end - first, MADV_DONTNEED)
        }
    }

    /// The whole map, once a stage has finished with it.
    package func dropFromCache() {
        _ = madvise(UnsafeMutableRawPointer(mutating: base), size, MADV_DONTNEED)
    }

    package var parameters: Int { tensors.values.reduce(0) { $0 + $1.count } }

    /// Forge v3 lays out `Linear` weights as `[input, output]`. The engine must know it:
    /// the layout is a property of the artifact, not a constant of the code.
    package var linearWeightsTransposed: Bool { header["linear_weights_transposed"] as? Bool ?? false }

    /// Materializes a tensor as fp32 into a caller's buffer, **respecting its dtype**.
    ///
    /// The map is not homogeneous: `t_embedder` and `cap_embedder` are fp32 in it because the
    /// reference declares them `precision sensitive layers`. Widening their bytes as if they were
    /// bf16 does not crash and does not produce infinity — it produces finite numbers, of the right
    /// order of magnitude, and wrong. That is why the dtype is read here once and for all rather than
    /// on every call.
    ///
    /// **`capacity` is the number of floats `destination` holds**, and a tensor that has more is
    /// refused before a single value is written (`doesNotFit`): the count comes from the map, the
    /// buffer from the model, and a map that is not the one the model expects would otherwise write
    /// past the end of an arena slice — the callers' `got == k·n` checks came after the writing.
    @discardableResult
    package func materialize(_ name: String, into destination: UnsafeMutablePointer<Float>, capacity: Int) throws -> Int {
        guard let tensor = tensors[name], let source = pointer(name) else {
            throw Failure.badHeader("\(name) missing from the artifact")
        }
        guard tensor.count <= capacity else {
            throw Failure.inFile(path, .doesNotFit(name: name, count: tensor.count, capacity: capacity))
        }
        switch tensor.dtype {
        case .bfloat16:
            Widen.bfloat16ToFloat32(source: source, destination: UnsafeMutableRawPointer(destination),
                                    count: tensor.count)
        case .float32:
            destination.update(from: source.assumingMemoryBound(to: Float.self), count: tensor.count)
        case .float16:
            // Weights published in fp16 are stored in fp16 (never wider, never narrower than given);
            // widening them is exact. The activations never are: they stay fp32 (trap 3.11).
            Widen.float16ToFloat32(source, count: tensor.count, into: destination)
        case .q4_0, .q4_1, .q5_0, .q5_1, .q4_k, .q5_k, .q6_k:
            Widen.dequantizePacked(source, kind: tensor.dtype, rows: tensor.shape[1], columns: tensor.shape[0],
                                        transposing: true, into: destination)
        case .int8, .float8_e4m3:
            if let rotation = tensor.rotation, let scale = tensor.scale {
                Widen.dequantizeRotated(source, rows: tensor.shape[0], columns: tensor.shape[1], group: rotation.group,
                                        scale: source + scale.offset, scaleType: scale.dtype, layout: scale.layout,
                                        into: destination)
                return tensor.count
            }
            Widen.dequantize(source, kind: tensor.dtype, count: tensor.count,
                             scale: tensor.scale.map { source + $0.offset }, scaleType: tensor.scale?.dtype ?? .float32,
                             layout: tensor.scale?.layout ?? .tensor, into: destination)
        }
        return tensor.count
    }

    /// `materialize` into a whole buffer: its count is the capacity.
    @discardableResult
    package func materialize(_ name: String, into buffer: UnsafeMutableBufferPointer<Float>) throws -> Int {
        guard let base = buffer.baseAddress else {
            throw Failure.inFile(path, .doesNotFit(name: name, count: tensors[name]?.count ?? 0, capacity: 0))
        }
        return try materialize(name, into: base, capacity: buffer.count)
    }

    /// **Rows `[first, first + count)` of a 2-D tensor**, in fp32 — what a table of embeddings
    /// reads (one token, a few positions) without widening the whole table. Every dtype the map
    /// carries, 8-bit included (its scale is per column or per block of rows: the rows asked for
    /// take theirs).
    package func materializeRows(_ name: String, first: Int, count rows: Int,
                                 into destination: UnsafeMutablePointer<Float>) throws {
        guard let tensor = tensors[name], let source = pointer(name), tensor.shape.count == 2 else {
            throw Failure.badHeader("\(name) missing from the artifact, or not a table")
        }
        let width = tensor.shape[1]
        // A rotated weight mixes the rows of each group: a few rows are not a few rows of it.
        guard tensor.rotation == nil else {
            throw Failure.badHeader("\(name): rotated (convrot) — its rows cannot be read apart")
        }
        // A packed GGUF type is stored by outputs (`DType.q4_k`): a row of `[K, N]` is a column of
        // its blocks. The forge never writes a table that way (only a transposed `Linear` keeps them).
        guard !tensor.dtype.isPacked else {
            throw Failure.badHeader("\(name): \(tensor.dtype.rawValue) blocks — stored by output, its rows are not read apart")
        }
        guard first >= 0, rows >= 0, first + rows <= tensor.shape[0] else {
            throw Failure.badHeader("\(name): rows \(first)..<\(first + rows) outside \(tensor.shape)")
        }
        let start = first * width, n = rows * width
        switch tensor.dtype {
        case .bfloat16:
            Widen.bfloat16ToFloat32(source: source + 2 * start, destination: UnsafeMutableRawPointer(destination), count: n)
        case .float32:
            destination.update(from: (source + 4 * start).assumingMemoryBound(to: Float.self), count: n)
        case .float16:
            Widen.float16ToFloat32(source + 2 * start, count: n, into: destination)
        case .q4_0, .q4_1, .q5_0, .q5_1, .q4_k, .q5_k, .q6_k:
            preconditionFailure("refused above")
        case .int8, .float8_e4m3:
            Widen.dequantize(source + start, kind: tensor.dtype, count: n,
                             scale: tensor.scale.map { source + $0.offset }, scaleType: tensor.scale?.dtype ?? .float32,
                             layout: tensor.scale?.layout ?? .tensor, firstRow: first, into: destination)
        }
    }
}

/// `Artifact.readThrough`'s buffer: one tensor at a time, page-aligned anonymous memory rounded up to
/// whole pages (what `WidenGPU` wraps), grown when a larger tensor comes.
private final class ReadThrough {
    private let descriptor: Int32
    private var slot: UnsafeMutableRawPointer?
    private var slotBytes = 0
    private var current: String?

    init(descriptor: Int32) { self.descriptor = descriptor }

    deinit {
        if let slot { munmap(slot, slotBytes) }
        close(descriptor)
    }

    /// The tensor's bytes, read once per tensor; `nil` if the file cannot give them (never the map instead).
    func read(_ name: String, _ t: Artifact.Tensor, page: Int) -> UnsafeRawPointer? {
        if current == name, let slot { return UnsafeRawPointer(slot) }
        current = nil
        let rounded = max(page, (t.bytes + page - 1) / page * page)
        if rounded > slotBytes {
            if let slot { munmap(slot, slotBytes) }
            slot = nil; slotBytes = 0
            guard let p = mmap(nil, rounded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0), p != MAP_FAILED else { return nil }
            slot = p; slotBytes = rounded
        }
        guard let slot else { return nil }
        var done = 0
        while done < t.bytes {
            let n = pread(descriptor, slot + done, t.bytes - done, off_t(t.offset + done))
            guard n > 0 else { return nil }
            done += n
        }
        current = name
        return UnsafeRawPointer(slot)
    }
}
