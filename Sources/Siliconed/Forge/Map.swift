import CryptoKit
import Darwin
import Foundation

/// **Published safetensors (or a GGUF), read as a single set of tensors.**
///
/// A diffusers model comes in shards (`*.safetensors.index.json`), a Civitai checkpoint in one
/// file; the forge only sees names. An 8-bit weight (fp8, int8 — `QuantizedLayouts`) appears under
/// its model name (`<l>.weight`) with its published shape and dtype; its scale, its zero point,
/// `input_scale`, `comfy_quant` are not tensors of the model and disappear from the names.
/// `read` gives it dequantized in fp32; `readQuantized`, the bytes and scales as published — what an
/// 8-bit map keeps.
package final class TensorSource {
    package let files: [Safetensors]
    private var index: [String: (file: Safetensors, entry: Safetensors.Entry)] = [:]
    /// In file order, then byte order.
    package private(set) var names: [String] = []
    /// The 8-bit weights, by model name.
    private var groups: [String: QuantizedGroup] = [:]
    /// What the reading of the 8-bit layouts noticed (ignored `input_scale`, activation quantization…).
    package private(set) var notes: [String] = []
    /// `format → count` of the 8-bit weights, for the journal.
    package var quantizedFormats: [String: Int] {
        groups.values.reduce(into: [:]) { $0[$1.format, default: 0] += 1 }
    }

    package init(paths: [String]) throws {
        files = try paths.map { try Safetensors(path: $0) }
        for f in files { if let g = f.gguf { notes.append("GGUF v\(g.version), alignment \(g.alignment)"
            + (f.metadata["general.architecture"].map { ", architecture \($0)" } ?? "")) } }
        var raw: [String: (Safetensors, Safetensors.Entry)] = [:]
        var order: [String] = []
        for f in files {
            for name in f.order {
                guard raw[name] == nil else {
                    throw Numerics.Failure(description: "\(name) present in two files")
                }
                raw[name] = (f, f.entries[name]!)
                order.append(name)
            }
        }
        let metadata = files.reduce(into: [String: String]()) { $0.merge($1.metadata) { a, _ in a } }
        let recognized = try QuantizedLayouts.recognize(raw, order: order, metadata: metadata)
        groups = recognized.groups
        notes += recognized.notes
        // torchao's values live under `<l>._weight_qdata`: the model name takes their place.
        let logicalOf = Dictionary(uniqueKeysWithValues: groups.map { ($1.values.entry.name, $0) })
        for name in order {
            // A consumed name is never a model tensor — except torchao's values, whose name the
            // model name replaces (`_weight_qdata` is consumed *and* grouped).
            if let logical = logicalOf[name], groups[logical] != nil {
                index[logical] = groups[logical]!.values
                names.append(logical)
            } else if !recognized.consumed.contains(name) {
                index[name] = raw[name]
                names.append(name)
            }
        }
    }

    /// A diffusers folder (`transformer/`, `text_encoder/`): its shards, or its single file.
    package convenience init(folder: String) throws {
        try self.init(paths: try Self.shards(folder: folder))
    }

    /// The files of a diffusers folder: its index's shards, or its single files.
    package static func shards(folder: String) throws -> [String] {
        let fm = FileManager.default
        let content = try fm.contentsOfDirectory(atPath: folder)
        if let idx = content.first(where: { $0.hasSuffix(".safetensors.index.json") }) {
            let json = try OrderedJSON.read(folder + "/" + idx)
            var fragments: [String] = []
            for p in json["weight_map"]?.pairs ?? [] {
                if let f = p.value.text, !fragments.contains(f) { fragments.append(f) }
            }
            return fragments.sorted().map { folder + "/" + $0 }
        }
        let singleFiles = content.filter { $0.hasSuffix(".safetensors") }.sorted()
        guard !singleFiles.isEmpty else { throw MissingFile(folder + "/*.safetensors", .published) }
        return singleFiles.map { folder + "/" + $0 }
    }

    package var metadata: [String: String] {
        files.reduce(into: [:]) { $0.merge($1.metadata) { a, _ in a } }
    }

    /// An SQ weight's is the `[N, K]` its codes stand for (`QuantizedGroup.sqShape`).
    package func shape(_ name: String) -> [Int]? { groups[name]?.sqShape ?? index[name]?.entry.shape }

    package func dtype(_ name: String) -> String? { index[name]?.entry.dtype }

    /// The tensor in fp32 — an 8-bit one dequantized (`w = Float(q) · s`, exact or one rounding).
    package func read(_ name: String) throws -> [Float] {
        if let g = groups[name] { return try g.read().dequantized() }
        guard let (f, e) = index[name] else {
            throw Numerics.Failure(description: "\(name) absent from the source")
        }
        var v = [Float](unsafeUninitializedCapacity: e.count) { buffer, n in n = e.count }
        try v.withUnsafeMutableBytes { d in
            if e.dtype == "F32" { return try f.copy(e, into: d.baseAddress!) }
            // Read whole, then widened from memory: the file is read by `pread`, never by page faults.
            let raw = try f.bytes(e)
            try raw.withUnsafeBytes {
                try Numerics.toFloat32($0.baseAddress!, dtype: e.dtype, count: e.count,
                                       to: d.baseAddress!.assumingMemoryBound(to: Float.self))
            }
        }
        return v
    }

    /// **A scalar's value** — a float as `read` gives it, or an integer: sd-scripts saves a LoRA's
    /// `.alpha` as `torch.tensor(alpha)`, an `I64` when the network alpha was given as an `int`,
    /// and diffusers reads it back with `.item()`. Integers stop here: `toFloat32` would let an
    /// `I32` of packed 4-bit weights pass as numbers.
    package func scalar(_ name: String) throws -> Float? {
        guard groups[name] == nil, let (f, e) = index[name], e.count == 1 else { return try read(name).first }
        let size = ["I64": 8, "I32": 4, "I16": 2][e.dtype]
        guard let size else { return try read(name).first }
        let b = try f.bytes(e)
        guard b.count == size else { throw Numerics.Failure(description: "\(name) : \(b.count) bytes for one \(e.dtype)") }
        return b.withUnsafeBytes { p in
            switch size {
            case 8: Float(Int64(littleEndian: p.loadUnaligned(as: Int64.self)))
            case 4: Float(Int32(littleEndian: p.loadUnaligned(as: Int32.self)))
            default: Float(Int16(littleEndian: p.loadUnaligned(as: Int16.self)))
            }
        }
    }

    /// **A 16-bit tensor's bits as published** (`BF16`, `F16`), nothing widened: what the forge
    /// writes as is when its rule moves values without computing (`ForgeDiT.tensor`). `nil`: any
    /// other dtype, or an 8-bit weight.
    package func read16(_ name: String) throws -> [UInt16]? {
        guard groups[name] == nil, let (f, e) = index[name], e.dtype == "BF16" || e.dtype == "F16" else { return nil }
        return try [UInt16](unsafeUninitializedCapacity: e.count) { b, n in
            try f.copy(e, into: UnsafeMutableRawPointer(b.baseAddress!))
            n = e.count
        }
    }

    /// An 8-bit weight's description (`kind`, scale type, per row, by block), without reading it.
    package func quantization(_ name: String) -> QuantizedKind? {
        groups[name].map { g in (g.kind, g.scaleType, g.perRow, g.block, g.rotation) }
    }

    /// An 8-bit weight's bytes and scales, as published.
    package func readQuantized(_ name: String) throws -> QuantizedTensor {
        guard let g = groups[name] else { throw Numerics.Failure(description: "\(name) is not 8-bit") }
        return try g.read()
    }
}

/// **A `SILICON\x03` map to write**: the header, then the tensors, produced several at a time and
/// hashed in file order.
///
/// Same rules as the Python forges it replaces: each tensor starts on a 16 KiB page, the prologue
/// takes a whole number of pages, the table offsets are absolute, and peak memory is bounded — the
/// first pass only places shapes, the second holds at most `inFlight` bytes of tensors being
/// produced or waiting for the hash (one larger tensor alone).
package enum MapWriter {
    package static let signature: [UInt8] = Array("SILICON".utf8) + [3]
    package static let page = 16384

    /// `float16`: weights published in fp16, kept in fp16. `int8` and `float8_e4m3`: published
    /// 8-bit values, written as they came. `q4_0`…`q5_1`, `q4_k`…`q6_k`: GGUF's packed
    /// blocks, written as they came, by output rows (`Artifact.DType.q4_k`).
    package enum DType: String, Sendable {
        case bfloat16, float16, float32, int8, float8_e4m3, q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k, sq4, sq5
        var packedBlock: (values: Int, bytes: Int)? { Artifact.DType(rawValue: rawValue)?.packedBlock }
        var sqBits: Int? { Artifact.DType(rawValue: rawValue)?.sqBits }
    }

    /// **An 8-bit tensor's scale** (`Artifact.Scale`): stored after the values, on the next 16-byte
    /// boundary, inside the tensor's own region — one offset, one length, one page run.
    package struct Scale {
        /// `bfloat16`, `float16` or `float32`: the published type, never converted.
        package let dtype: Artifact.DType
        /// `[]`, `[N]`, or `[K/b, N]` with `block` = b.
        package let shape: [Int]
        package let block: Int?
        package init(dtype: Artifact.DType, shape: [Int], block: Int? = nil) {
            self.dtype = dtype; self.shape = shape; self.block = block
        }
        var bytes: Int { shape.reduce(1, *) * dtype.size }
    }

    /// What an 8-bit tensor produces: its values in the map's layout, its scales' bytes, and the
    /// largest |w| once dequantized (for `weight_absmax`).
    package struct QuantizedPayload {
        package let values: [UInt8]
        package let scale: [UInt8]
        package let absMax: Float
        package init(values: [UInt8], scale: [UInt8], absMax: Float) {
            self.values = values; self.scale = scale; self.absMax = absMax
        }
    }

    /// A 16-bit tensor's bits in the map's layout, and its largest |w| (`weight_absmax`).
    package struct Bits {
        package let values: [UInt16]
        package let absMax: Float
        package init(values: [UInt16], absMax: Float) { self.values = values; self.absMax = absMax }
    }

    package struct Tensor {
        package let name: String
        package let shape: [Int]
        package let dtype: DType
        /// `nil`: the `transposed` key is not written (LoRAs do not carry it).
        package let transposed: Bool?
        /// The values, final layout, in fp32 (`bfloat16`, `float16` and `float32` tensors).
        package let produce: () throws -> [Float]
        /// An 8-bit tensor's scale (`nil`: an fp8 cast without one).
        package let scale: Scale?
        /// An 8-bit tensor's bytes, final layout.
        package let produceQuantized: (() throws -> QuantizedPayload)?
        /// ComfyUI's convrot group (`Artifact.Rotation`): written as `"rotation": {"kind": "convrot",
        /// "group": G}`; `nil`: not written.
        package var rotation: Int? = nil

        /// A bf16 or fp16 tensor's final bits, moved from the published ones without going through
        /// fp32 (`ForgeDiT.tensor`): the same bytes as `produce` then rounded, which is exact on
        /// them. `nil` result: `produce` (a NaN, whose payload the fp32 round trip may change).
        package var produceBits: (() throws -> Bits?)? = nil

        package init(name: String, shape: [Int], dtype: DType, transposed: Bool?,
                     produce: @escaping () throws -> [Float]) {
            self.name = name; self.shape = shape; self.dtype = dtype; self.transposed = transposed
            self.produce = produce; self.scale = nil; self.produceQuantized = nil
        }

        package init(name: String, shape: [Int], dtype: DType, scale: Scale?, transposed: Bool?,
                     produce: @escaping () throws -> QuantizedPayload) {
            precondition(dtype == .int8 || dtype == .float8_e4m3 || dtype.packedBlock != nil && scale == nil
                         || dtype.sqBits != nil && scale != nil)
            self.name = name; self.shape = shape; self.dtype = dtype; self.transposed = transposed
            self.produce = { [] }; self.scale = scale; self.produceQuantized = produce
        }

        var count: Int { shape.reduce(1, *) }
        var quantized: Bool { produceQuantized != nil }
        /// What producing it holds at most, its output included — what the pipeline budgets
        /// (`write`): 16-bit bits read then moved (2 + 2 bytes a value); 8-bit bytes read then moved,
        /// and an fp32 dequantization for `weight_absmax` (convrot, GGUF 4–6 bits); otherwise read
        /// (≤ 4), widened (4), moved (4), written (≤ 4).
        var working: Int {
            if produceBits != nil { return count * 4 }
            if quantized { return 2 * bytes + 4 * count }
            return 16 * count
        }
        /// The bytes of the values alone: one per 8-bit value, or a packed type's blocks.
        var valueBytes: Int {
            if let bits = dtype.sqBits { return shape[1] * SQ.rowBytes(k: shape[0], bits: bits) }
            return dtype.packedBlock.map { count / $0.values * $0.bytes } ?? count
        }
        /// Where the scale starts, from the tensor's start.
        var scaleOffset: Int { (valueBytes + 15) / 16 * 16 }
        var bytes: Int {
            switch dtype {
            case .float32: return count * 4
            case .bfloat16, .float16: return count * 2
            case .int8, .float8_e4m3, .sq4, .sq5: return scale.map { scaleOffset + $0.bytes } ?? count
            case .q4_0, .q4_1, .q5_0, .q5_1, .q4_k, .q5_k, .q6_k: return valueBytes
            }
        }
    }

    package struct Tally {
        package let path: String
        package let bytes: Int
        package let parameters: Int
        package let tensors: Int
        /// Values rounded when converting to bf16 — zero when the source was already bf16.
        package let inexact: Int
        package let absMax: Float
        package let sha256: String
    }

    /// What the header can only state once the bytes have been read.
    package struct Measures {
        package let absMax: Float
        package let sha256: String
    }

    /// Writes the map. `header` receives the measurements (fingerprint, max |weight|) and returns the
    /// header pairs **before** `tensors` and `data_start`, in the desired order.
    ///
    /// `reserve` receives the map's exact size before anything is written — the disk check of an
    /// import is made on the real map, not on a guess from the source's size.
    package static func write(to path: String, tensors: [Tensor],
                               header: (Measures?) -> [OrderedJSON.Pair],
                               reserve: ((Int) throws -> Void)? = nil,
                               progressHandler: ((Int, Int, String) -> Void)? = nil) throws -> Tally {
        // ── Pass 1: place ────────────────────────────────────────────────────────────────
        var relativeOffsets: [Int] = [], end = 0
        for t in tensors {
            let p = end + (page - end % page) % page
            relativeOffsets.append(p)
            end = p + t.bytes
        }
        func prologue(_ begin: Int, _ measurements: Measures?) -> [UInt8] {
            var table: [OrderedJSON.Pair] = []
            for (k, t) in tensors.enumerated() {
                var e: [OrderedJSON.Pair] = [
                    .init("offset", .integer(relativeOffsets[k] + begin)), .init("bytes", .integer(t.bytes)),
                    .init("shape", .list(t.shape.map { .integer($0) })), .init("dtype", .string(t.dtype.rawValue)),
                ]
                if let tr = t.transposed { e.append(.init("transposed", .boolean(tr))) }
                if let sc = t.scale {
                    var d: [OrderedJSON.Pair] = [.init("offset", .integer(t.scaleOffset)), .init("dtype", .string(sc.dtype.rawValue)),
                                                 .init("shape", .list(sc.shape.map { .integer($0) }))]
                    if let b = sc.block { d.append(.init("block", .integer(b))) }
                    e.append(.init("scale", .object(d)))
                }
                if let g = t.rotation {
                    e.append(.init("rotation", .object([.init("kind", .string("convrot")), .init("group", .integer(g))])))
                }
                table.append(.init(t.name, .object(e)))
            }
            let h = OrderedJSON.object(header(measurements) + [.init("tensors", .object(table)),
                                                         .init("data_start", .integer(begin))])
            let json = Array(h.jsonText.utf8)
            var n = UInt64(json.count).littleEndian
            return signature + withUnsafeBytes(of: &n) { Array($0) } + json
        }
        // The measurements are only known at the end: we place with the longest possible spelling
        // (64 digits of fingerprint, a |max| with 17 significant digits), and the real prologue,
        // shorter or equal, fits in the reserved space.
        let worst = Measures(absMax: -Float.greatestFiniteMagnitude.nextDown, sha256: String(repeating: "f", count: 64))
        var begin = (prologue(0, worst).count + page - 1) / page * page
        while prologue(begin, worst).count > begin { begin += page }

        try reserve?(begin + end)

        // ── Pass 2: produce over the cores, hash in order ───────────────────────────────
        // **The fingerprint is one SHA-256 over the tensors in file order** (`content_sha256`):
        // it alone stays sequential. The tensors are produced several at a time — read, moved,
        // converted, written at their own offset — while this thread hashes them in order as they
        // come. Before, a forge was one tensor at a time, the source read by page faults (0.5–0.9 GB/s)
        // and every step on one core; the hash is now its floor (≈ 2.4 GB/s).
        let temporary = path + ".partiel"
        // Not truncated at `open`: another forge of the same map may be writing it. **The lock decides**
        // — taken, the file is ours and emptied; held, we refuse and leave it alone. `O_NOFOLLOW`: a
        // link planted under the name is not written through.
        let fd = open(temporary, O_CREAT | O_WRONLY | O_NOFOLLOW, 0o644)
        guard fd >= 0 else { throw Numerics.Failure(description: "cannot write \(temporary)") }
        // Locked while it is written: `Library.emptyDownloads` purges only the `.partiel` nobody holds.
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            close(fd)
            throw Numerics.Failure(description: "\(temporary) is being written by another forge")
        }
        // **Written past the page cache.** Produced over the cores, a 14 GB map was written
        // faster than the cache flushes it, and its dirty pages pushed other processes' memory to
        // swap (+1448 swapouts on a quiet machine, in 11 s). Read once, by the engine, later:
        // nothing is lost by not keeping it.
        _ = fcntl(fd, F_NOCACHE, 1)
        var closed = false
        defer { if !closed { close(fd); unlink(temporary) } }
        guard ftruncate(fd, 0) == 0, ftruncate(fd, off_t(begin + end)) == 0 else {
            throw Numerics.Failure(description: "\(temporary) : ftruncate — disk full?")
        }
        let pipeline = Pipeline(tensors: tensors)
        nonisolated(unsafe) let unsafeTensors = tensors
        let (start, offsets) = (begin, relativeOffsets)
        let workers = DispatchGroup()
        for _ in 0..<min(producers, tensors.count) {
            DispatchQueue.global(qos: .userInitiated).async(group: workers) {
                while let k = pipeline.take() {
                    do {
                        let made = try produce(unsafeTensors[k])
                        for (bytes, at) in made.parts {
                            try bytes.withBytes { try put(fd, $0, at: start + offsets[k] + at, temporary) }
                        }
                        pipeline.done(k, made)
                    } catch { pipeline.fail(error) }
                }
            }
        }
        var sha256 = SHA256()
        var inexact = 0, parameters = 0
        var absMax: Float = 0
        var failure: Error?
        for (k, t) in tensors.enumerated() {
            guard let made = pipeline.wait(k) else { failure = pipeline.failure; break }
            progressHandler?(k, tensors.count, t.name)
            parameters += t.count
            inexact += made.inexact
            absMax = max(absMax, made.absMax)
            sha256.update(data: Data(t.name.utf8))
            for (bytes, _) in made.parts { bytes.withBytes { if $0.count > 0 { sha256.update(bufferPointer: $0) } } }
            pipeline.release(k)
        }
        workers.wait()
        if let failure { throw failure }
        let measurements = Measures(absMax: absMax, sha256: sha256.finalize().map { String(format: "%02x", $0) }.joined())
        let actual = prologue(begin, measurements)
        guard actual.count <= begin else { throw Numerics.Failure(description: "prologue longer than its space") }
        let n = actual.withUnsafeBytes { pwrite(fd, $0.baseAddress!, $0.count, 0) }
        guard n == actual.count, fsync(fd) == 0 else { throw Numerics.Failure(description: "\(temporary) : prologue not written") }
        // `rename` alone replaces an existing map atomically: no instant without a map at `path`
        // (an `unlink` first would open one), and a reader that has the old one mapped keeps it.
        // Renamed before being closed, so that the `.partiel` is never unlocked under its name.
        guard rename(temporary, path) == 0 else { throw Numerics.Failure(description: "\(path) : cannot rename") }
        close(fd); closed = true
        return Tally(path: path, bytes: begin + end, parameters: parameters, tensors: tensors.count,
                     inexact: inexact, absMax: absMax, sha256: measurements.sha256)
    }
    /// Tensors produced at once; and the bytes they may hold together (`Tensor.working`) — one
    /// larger than that is produced alone. Measured on Qwen's DiT: 1 GB in flight peaked at 0.9–1.0 GB
    /// and 9–14 s, 512 MB at 0.7–0.8 GB and 9 s, 256 MB at 0.55 GB and 16 s; the encoder's peak is
    /// its embedding table (1.24 GB, alone) whatever the budget.
    static let producers = 4
    static let inFlight = 512 << 20

    /// What producing a tensor gives: its bytes, each at its offset from the tensor's start.
    struct Produced {
        var parts: [(Bytes, Int)]
        var absMax: Float
        var inexact = 0
    }

    enum Bytes {
        case u8([UInt8]), u16([UInt16]), f32([Float])
        func withBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
            switch self {
            case .u8(let a): return try a.withUnsafeBytes(body)
            case .u16(let a): return try a.withUnsafeBytes(body)
            case .f32(let a): return try a.withUnsafeBytes(body)
            }
        }
    }

    /// One tensor in the map's bytes — the work a producer does, on any thread.
    static func produce(_ t: Tensor) throws -> Produced {
        if let produceQuantized = t.produceQuantized {
            // The published bytes, as they came: nothing converted, nothing rounded.
            let q = try produceQuantized()
            guard q.values.count == t.valueBytes, q.scale.count == (t.scale?.bytes ?? 0) else {
                throw Numerics.Failure(description: "\(t.name) : \(q.values.count) values and \(q.scale.count) "
                                       + "scale bytes for \(t.shape)")
            }
            return Produced(parts: [(.u8(q.values), 0)] + (q.scale.isEmpty ? [] : [(.u8(q.scale), t.scaleOffset)]),
                            absMax: q.absMax)
        }
        if let produceBits = t.produceBits, let b = try produceBits() {
            guard b.values.count == t.count, t.dtype == .bfloat16 || t.dtype == .float16 else {
                throw Numerics.Failure(description: "\(t.name) : \(b.values.count) 16-bit values for \(t.shape) \(t.dtype)")
            }
            return Produced(parts: [(.u16(b.values), 0)], absMax: b.absMax)
        }
        let values = try t.produce()
        guard values.count == t.count else {
            throw Numerics.Failure(description: "\(t.name) : \(values.count) values for \(t.shape)")
        }
        let m = values.withUnsafeBufferPointer { Numerics.absMax($0.baseAddress!, count: $0.count) }
        switch t.dtype {
        case .int8, .float8_e4m3, .q4_0, .q4_1, .q5_0, .q5_1, .q4_k, .q5_k, .q6_k, .sq4, .sq5:
            throw Numerics.Failure(description: "\(t.name) : a quantized tensor without its bytes")
        case .float32:
            return Produced(parts: [(.f32(values), 0)], absMax: m)
        case .float16, .bfloat16:
            var inexact = 0
            let h = [UInt16](unsafeUninitializedCapacity: values.count) { hb, n in
                values.withUnsafeBufferPointer { v in
                    inexact = t.dtype == .float16
                        ? Numerics.toFloat16(v.baseAddress!, count: v.count, to: hb.baseAddress!)
                        : Numerics.toBFloat16(v.baseAddress!, count: v.count, to: hb.baseAddress!)
                }
                n = values.count
            }
            return Produced(parts: [(.u16(h), 0)], absMax: m, inexact: inexact)
        }
    }

    static func put(_ fd: Int32, _ raw: UnsafeRawBufferPointer, at offset: Int, _ temporary: String) throws {
        var remaining = raw.count, pos = 0
        while remaining > 0 {
            let n = pwrite(fd, raw.baseAddress! + pos, min(remaining, 1 << 30), off_t(offset + pos))
            guard n > 0 else { throw Numerics.Failure(description: "\(temporary) : write — disk full?") }
            remaining -= n; pos += n
        }
    }

    /// **The producers' bookkeeping**: a tensor is taken in file order, when the bytes in flight
    /// leave room for it (or nothing else is in flight); its result waits for the hash. Taking in
    /// order is what makes the budget deadlock-free: the tensor the hash waits for always got its
    /// room before any later one did.
    final class Pipeline: @unchecked Sendable {
        private let cost: [Int]
        private let condition = NSCondition()
        private var next = 0, inFlight = 0
        private var results: [Int: Produced] = [:]
        private(set) var failure: Error?

        init(tensors: [Tensor]) { cost = tensors.map(\.working) }

        func take() -> Int? {
            condition.lock(); defer { condition.unlock() }
            while failure == nil, next < cost.count, inFlight > 0, inFlight + cost[next] > MapWriter.inFlight { condition.wait() }
            guard failure == nil, next < cost.count else { return nil }
            defer { next += 1 }
            inFlight += cost[next]
            return next
        }

        func done(_ k: Int, _ p: Produced) {
            condition.lock(); results[k] = p; condition.broadcast(); condition.unlock()
        }

        func fail(_ e: Error) {
            condition.lock(); if failure == nil { failure = e }; condition.broadcast(); condition.unlock()
        }

        /// Tensor `k`'s result, once produced; `nil`: a producer failed.
        func wait(_ k: Int) -> Produced? {
            condition.lock(); defer { condition.unlock() }
            while results[k] == nil, failure == nil { condition.wait() }
            return failure == nil ? results[k] : nil
        }

        /// Hashed: its bytes go, its room goes back.
        func release(_ k: Int) {
            condition.lock(); results[k] = nil; inFlight -= cost[k]; condition.broadcast(); condition.unlock()
        }
    }
}

/// **Do two maps say the same thing?** Same tensors, same shapes, same dtypes, same bytes — and
/// the same header, apart from what describes where and when it was forged (`source`,
/// `data_start`, the offsets). This is the verdict that allows discarding the old one.
package enum MapComparison {
    package struct Verdict: CustomStringConvertible {
        package var equalTensors = 0
        package var differences: [String] = []
        package var remarks: [String] = []
        package var allIdentical: Bool { differences.isEmpty }
        package var description: String {
            (allIdentical ? "✓ \(equalTensors) tensors identical byte for byte"
                        : "✗ \(differences.count) difference(s)\n  " + differences.prefix(12).joined(separator: "\n  "))
                + (remarks.isEmpty ? "" : "\n  (header: " + remarks.joined(separator: " ; ") + ")")
        }
    }

    package static func compare(_ a: String, _ b: String) throws -> Verdict {
        let x = try Artifact(path: a), y = try Artifact(path: b)
        var v = Verdict()
        if x.order != y.order {
            if Set(x.order) != Set(y.order) {
                let missingList = Set(x.order).subtracting(y.order).sorted(), too = Set(y.order).subtracting(x.order).sorted()
                v.differences.append("names: \(missingList.count) missing from B \(missingList.prefix(3)), \(too.count) extra \(too.prefix(3))")
                return v
            }
            v.differences.append("tensor order differs")
        }
        for name in x.order {
            let s = x.tensors[name]!, t = y.tensors[name]!
            guard s.shape == t.shape, s.dtype == t.dtype else {
                v.differences.append("\(name) : \(s.shape) \(s.dtype) versus \(t.shape) \(t.dtype)"); continue
            }
            guard s.rotation == t.rotation else {
                v.differences.append("\(name) : rotation \(String(describing: s.rotation)) versus \(String(describing: t.rotation))"); continue
            }
            guard s.scale == t.scale, s.bytes == t.bytes else {
                v.differences.append("\(name) : scale \(String(describing: s.scale)) versus \(String(describing: t.scale))"); continue
            }
            // An 8-bit tensor's bytes cover its values *and* its scales: both are compared.
            if memcmp(x.pointer(name)!, y.pointer(name)!, s.bytes) == 0 { v.equalTensors += 1 }
            else { v.differences.append("\(name) : bytes differ") }
        }
        let ignored: Set<String> = ["source", "data_start", "tensors"]
        let keys = Set(x.header.keys).union(y.header.keys).subtracting(ignored)
        for key in keys.sorted() {
            let p = x.header[key], q = y.header[key]
            if key == "weight_absmax", let m = p as? Double, let n = q as? Double {
                if Float(m) != Float(n) { v.differences.append("weight_absmax \(m) versus \(n)") }
                continue
            }
            switch (p, q) {
            case (nil, _?): v.remarks.append("\"\(key)\" only in B")
            case (_?, nil): v.remarks.append("\"\(key)\" only in A")
            case let (p?, q?):
                if !(p as AnyObject).isEqual(q) { v.differences.append("header \"\(key)\" differs") }
            default: break
            }
        }
        for name in x.tensors.keys {
            let dx = (x.header["tensors"] as? [String: [String: Any]])?[name]?["transposed"] as? Bool
            let dy = (y.header["tensors"] as? [String: [String: Any]])?[name]?["transposed"] as? Bool
            if dx != dy { v.differences.append("\(name) : transposed \(String(describing: dx)) versus \(String(describing: dy))") }
        }
        return v
    }
}
