import CryptoKit
import Darwin
import Foundation

/// **Published safetensors, read as a single set of tensors.**
///
/// A diffusers model comes in shards (`*.safetensors.index.json`), a Civitai checkpoint in one
/// file; the forge only sees names. An 8-bit weight (fp8, int8 — `QuantizedLayouts`) appears under
/// its model name (`<l>.weight`) with its published shape and dtype; its scale, its zero point,
/// `input_scale`, `comfy_quant` are not tensors of the model and disappear from the names.
/// `read` gives it dequantized in fp32; `readQuantized`, the bytes and scales as published — what an
/// 8-bit map keeps.
package final class TensorSource {
    package struct Reading {
        package let shape: [Int]
        package let dtype: String
        package let read: () throws -> [Float]
        package var number: Int { shape.reduce(1, *) }
    }

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
        let fm = FileManager.default
        let content = try fm.contentsOfDirectory(atPath: folder)
        if let idx = content.first(where: { $0.hasSuffix(".safetensors.index.json") }) {
            let json = try OrderedJSON.read(folder + "/" + idx)
            var fragments: [String] = []
            for p in json["weight_map"]?.pairs ?? [] {
                if let f = p.value.text, !fragments.contains(f) { fragments.append(f) }
            }
            try self.init(paths: fragments.sorted().map { folder + "/" + $0 })
        } else {
            let singleFiles = content.filter { $0.hasSuffix(".safetensors") }.sorted()
            guard !singleFiles.isEmpty else { throw MissingFile(folder + "/*.safetensors", .published) }
            try self.init(paths: singleFiles.map { folder + "/" + $0 })
        }
    }

    package var metadata: [String: String] {
        files.reduce(into: [:]) { $0.merge($1.metadata) { a, _ in a } }
    }

    package func contains(_ name: String) -> Bool { index[name] != nil }

    package func shape(_ name: String) -> [Int]? { index[name]?.entry.shape }

    package func dtype(_ name: String) -> String? { index[name]?.entry.dtype }

    /// The tensor in fp32 — an 8-bit one dequantized (`w = Float(q) · s`, exact or one rounding).
    package func read(_ name: String) throws -> [Float] {
        if let g = groups[name] { return try g.read().dequantized() }
        guard let (f, e) = index[name], let p = f.pointer(e.name) else {
            throw Numerics.Failure(description: "\(name) absent from the source")
        }
        var v = [Float](unsafeUninitializedCapacity: e.count) { buffer, n in n = e.count }
        try v.withUnsafeMutableBufferPointer {
            try Numerics.toFloat32(p, dtype: e.dtype, count: e.count, to: $0.baseAddress!)
        }
        return v
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

    package func reading(_ name: String) -> Reading? {
        guard let (_, e) = index[name] else { return nil }
        return Reading(shape: e.shape, dtype: e.dtype) { [unowned self] in try self.read(name) }
    }
}

/// **A `SILICON\x03` map to write**: the header, then tensors produced one at a time.
///
/// Same rules as the Python forges it replaces: each tensor starts on a 16 KiB page, the prologue
/// takes a whole number of pages, the table offsets are absolute, and peak memory is **one**
/// tensor — the first pass only places shapes.
package enum MapWriter {
    package static let signature: [UInt8] = Array("SILICON".utf8) + [3]
    package static let page = 16384

    /// `float16`: weights published in fp16, kept in fp16. `int8` and `float8_e4m3`: published
    /// 8-bit values, written as they came. `q4_0`…`q5_1`, `q4_k`…`q6_k`: GGUF's packed
    /// blocks, written as they came, by output rows (`Artifact.DType.q4_k`).
    package enum DType: String, Sendable {
        case bfloat16, float16, float32, int8, float8_e4m3, q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k
        var packedBlock: (values: Int, bytes: Int)? { Artifact.DType(rawValue: rawValue)?.packedBlock }
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

        package init(name: String, shape: [Int], dtype: DType, transposed: Bool?,
                     produce: @escaping () throws -> [Float]) {
            self.name = name; self.shape = shape; self.dtype = dtype; self.transposed = transposed
            self.produce = produce; self.scale = nil; self.produceQuantized = nil
        }

        package init(name: String, shape: [Int], dtype: DType, scale: Scale?, transposed: Bool?,
                     produce: @escaping () throws -> QuantizedPayload) {
            precondition(dtype == .int8 || dtype == .float8_e4m3 || dtype.packedBlock != nil && scale == nil)
            self.name = name; self.shape = shape; self.dtype = dtype; self.transposed = transposed
            self.produce = { [] }; self.scale = scale; self.produceQuantized = produce
        }

        var count: Int { shape.reduce(1, *) }
        var quantized: Bool { produceQuantized != nil }
        /// The bytes of the values alone: one per 8-bit value, or a packed type's blocks.
        var valueBytes: Int {
            dtype.packedBlock.map { count / $0.values * $0.bytes } ?? count
        }
        /// Where the scale starts, from the tensor's start.
        var scaleOffset: Int { (count + 15) / 16 * 16 }
        var bytes: Int {
            switch dtype {
            case .float32: return count * 4
            case .bfloat16, .float16: return count * 2
            case .int8, .float8_e4m3: return scale.map { scaleOffset + $0.bytes } ?? count
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

        // ── Pass 2: produce, convert, write, one tensor at a time ─────────────────────
        let temporary = path + ".partiel"
        let fd = open(temporary, O_CREAT | O_TRUNC | O_WRONLY, 0o644)
        guard fd >= 0 else { throw Numerics.Failure(description: "cannot write \(temporary)") }
        // Locked while it is written: `Library.emptyDownloads` purges only the `.partiel` nobody holds.
        _ = flock(fd, LOCK_EX | LOCK_NB)
        var closed = false
        defer { if !closed { close(fd); unlink(temporary) } }
        guard ftruncate(fd, off_t(begin + end)) == 0 else {
            throw Numerics.Failure(description: "\(temporary) : ftruncate — disk full?")
        }
        var sha256 = SHA256()
        var inexact = 0, parameters = 0
        var absMax: Float = 0
        func put(_ raw: UnsafeRawBufferPointer, at offset: Int) throws {
            var remaining = raw.count, pos = 0
            while remaining > 0 {
                let n = pwrite(fd, raw.baseAddress! + pos, min(remaining, 1 << 30), off_t(offset + pos))
                guard n > 0 else { throw Numerics.Failure(description: "\(temporary) : write — disk full?") }
                remaining -= n; pos += n
            }
        }
        for (k, t) in tensors.enumerated() {
            progressHandler?(k, tensors.count, t.name)
            if let produceQuantized = t.produceQuantized {
                // The published bytes, as they came: nothing converted, nothing rounded.
                let q = try produceQuantized()
                guard q.values.count == t.valueBytes, q.scale.count == (t.scale?.bytes ?? 0) else {
                    throw Numerics.Failure(description: "\(t.name) : \(q.values.count) values and \(q.scale.count) "
                                           + "scale bytes for \(t.shape)")
                }
                parameters += t.count
                absMax = max(absMax, q.absMax)
                sha256.update(data: Data(t.name.utf8))
                try q.values.withUnsafeBytes { v in
                    sha256.update(bufferPointer: v)
                    try put(v, at: begin + relativeOffsets[k])
                }
                try q.scale.withUnsafeBytes { s in
                    guard s.count > 0 else { return }
                    sha256.update(bufferPointer: s)
                    try put(s, at: begin + relativeOffsets[k] + t.scaleOffset)
                }
                continue
            }
            let values = try t.produce()
            guard values.count == t.shape.reduce(1, *) else {
                throw Numerics.Failure(description: "\(t.name) : \(values.count) values for \(t.shape)")
            }
            parameters += values.count
            try values.withUnsafeBufferPointer { v in
                absMax = max(absMax, Numerics.absMax(v.baseAddress!, count: v.count))
                let written: (UnsafeRawBufferPointer) throws -> Void = { raw in
                    sha256.update(data: Data(t.name.utf8))
                    sha256.update(bufferPointer: raw)
                    var remaining = raw.count, pos = 0
                    while remaining > 0 {
                        let n = pwrite(fd, raw.baseAddress! + pos, min(remaining, 1 << 30),
                                       off_t(begin + relativeOffsets[k] + pos))
                        guard n > 0 else { throw Numerics.Failure(description: "\(temporary) : write — disk full?") }
                        remaining -= n; pos += n
                    }
                }
                switch t.dtype {
                case .int8, .float8_e4m3, .q4_0, .q4_1, .q5_0, .q5_1, .q4_k, .q5_k, .q6_k:
                    throw Numerics.Failure(description: "\(t.name) : a quantized tensor without its bytes")
                case .float32:
                    try written(UnsafeRawBufferPointer(v))
                case .float16:
                    var h = [UInt16](unsafeUninitializedCapacity: v.count) { _, n in n = v.count }
                    try h.withUnsafeMutableBufferPointer { hb in
                        inexact += Numerics.toFloat16(v.baseAddress!, count: v.count, to: hb.baseAddress!)
                        try written(UnsafeRawBufferPointer(hb))
                    }
                case .bfloat16:
                    var h = [UInt16](unsafeUninitializedCapacity: v.count) { _, n in n = v.count }
                    try h.withUnsafeMutableBufferPointer { hb in
                        inexact += Numerics.toBFloat16(v.baseAddress!, count: v.count, to: hb.baseAddress!)
                        try written(UnsafeRawBufferPointer(hb))
                    }
                }
            }
        }
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
        for (name, s) in x.tensors {
            let dx = (x.header["tensors"] as? [String: [String: Any]])?[name]?["transposed"] as? Bool
            let dy = (y.header["tensors"] as? [String: [String: Any]])?[name]?["transposed"] as? Bool
            if dx != dy { v.differences.append("\(name) : transposed \(String(describing: dx)) versus \(String(describing: dy))") }
            _ = s
        }
        return v
    }
}
