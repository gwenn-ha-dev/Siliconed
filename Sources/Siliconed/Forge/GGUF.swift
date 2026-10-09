import Foundation

/// **A GGUF checkpoint, read as a set of tensors**.
///
/// The layout, as llama.cpp's `gguf.c` writes it and as the published DiT files carry it (read on
/// the real headers of `unsloth/Z-Image-Turbo-GGUF` and `unsloth/Qwen-Image-2.1-GGUF`, versioned
/// as `tools/fixtures/quant/*-gguf/header.json`):
///
///     "GGUF" · u32 version (3) · u64 tensor count · u64 key/value count
///     key/value × n      key: u64 length + UTF-8 · u32 type · value (scalars, string, array)
///     tensor info × n    name · u32 n_dims · u64 ne[n_dims] · u32 ggml type · u64 offset
///     padding to `general.alignment` (32 when the key is absent)
///     data               each tensor at its offset from here, itself aligned
///
/// What a port by analogy with safetensors would miss:
///
///   · **The dimensions are stored in reverse.** `ne[0]` is the fastest-moving one — a `Linear`'s
///     *input* `K`. The torch shape is `ne` reversed: `[N, K]` for a weight, exactly what the same
///     tensor has in a safetensors. The entries below carry the torch shape.
///   · **Q8_0 interleaves its scales with its values**: each row of `K` is `K/32` blocks of 34
///     bytes — an fp16 `d`, then 32 int8 `q` — and `w = d · q` (ggml `dequantize_row_q8_0`,
///     diffusers `dequantize_blocks_Q8_0`). A block never crosses a row, so a row slice is
///     contiguous. `QuantizedGroup.read` takes the two apart: values `[N, K]` and scales
///     `[N, K/32]`, which the forge transposes into the map's `[K, N]` and `[K/32, N]`.
///   · **The names are the publisher's**, not llama.cpp's: ComfyUI's for Z-Image (`attention.qkv`,
///     no prefix), diffusers' behind `model.diffusion_model.` for Qwen-Image-2.1 — the family
///     recipes read them as they read a safetensors. The metadata says little (`general.architecture`
///     is `lumina2` for Z-Image; the Qwen file has no key at all) and decides nothing.
///   · Plain tensors are F32 (norms, biases — and sometimes a whole `Linear`: Qwen's
///     `norm_out.linear`), BF16 or F16, kept as published by the forge's rules.
///   · **The 4- to 6-bit types are packed blocks** along a row, their scales inside: the
///     legacy Q4_0, Q4_1, Q5_0, Q5_1 by 32 values (18, 20, 22, 24 bytes), the K-quants Q4_K, Q5_K,
///     Q6_K by super-blocks of 256 (144, 176, 210 bytes). They are never taken apart: the map keeps
///     each row's blocks as published (`Artifact.DType.q4_k`), and the engine dequantizes them with
///     ggml's arithmetic (`Widen.dequantizePacked`). A row must be whole blocks — ggml quantizes no
///     other row, and a file that says otherwise is refused. A published file mixes them tensor by
///     tensor (unsloth's Q4_0 carries Q4_1 on every `w2`; a Q4_K_M, Q5_K, Q6_K, Q8_0 and BF16).
///
/// **Read: F32, F16, BF16, Q8_0, Q6_K, Q5_K, Q4_K, Q5_1, Q5_0, Q4_1, Q4_0 — 4 bits is the floor**
/// Any other ggml type — Q3_K, Q2_K, the I-quants, the ternary TQ, MXFP4, Q8_1,
/// Q8_K — refuses the file in its entirety, by name: a model some of whose layers are below the
/// floor is not imported with those layers widened.
package enum GGUF {
    package static let magic: [UInt8] = Array("GGUF".utf8)

    /// The ggml types a GGUF file may carry (`ggml.h`, `enum ggml_type`), for the refusals' names.
    static let typeNames: [UInt32: String] = [
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
        16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
        22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
        29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
    ]

    /// Q8_0: 32 values per block, an fp16 scale in front of them.
    package static let q8Block = 32
    package static let q8BlockBytes = 34

    /// The packed types read, by ggml type: their name and the map's type (block size and bytes:
    /// `Artifact.DType.packedBlock`).
    package static let packedTypes: [UInt32: (name: String, dtype: Artifact.DType)] = [
        2: ("Q4_0", .q4_0), 3: ("Q4_1", .q4_1), 6: ("Q5_0", .q5_0), 7: ("Q5_1", .q5_1),
        12: ("Q4_K", .q4_k), 13: ("Q5_K", .q5_k), 14: ("Q6_K", .q6_k),
    ]

    package struct Header {
        package var version = 0
        package var alignment = 32
        /// Where the tensors' data starts (absolute).
        package var dataStart = 0
        /// Key → value, scalars and strings as text, arrays summarized (`[n × type]`).
        package var metadata: [String: String] = [:]
        package var entries: [Safetensors.Entry] = []
    }

    /// The longest metadata string read (a key, a value, an array item). A DiT's metadata says a few
    /// words; even an LLM's chat template stays under a few hundred kilobytes.
    package static let maximumString = 16 << 20
    /// The widest alignment accepted (llama.cpp writes 32): it only pads, and a forged one would
    /// overflow the rounding of the data's start.
    package static let maximumAlignment = 1 << 16

    package static func refuse(_ why: String) -> Numerics.Failure { Numerics.Failure(description: "GGUF: " + why) }

    /// Parses the header of a mapped GGUF file of `size` bytes and checks every tensor against it.
    package static func parse(_ base: UnsafeRawPointer, size: Int) throws -> Header {
        var p = 0
        func need(_ n: Int) throws {
            guard n >= 0, p + n <= size else { throw refuse("truncated header (at byte \(p))") }
        }
        func u32() throws -> UInt32 { try need(4); defer { p += 4 }; return base.loadUnaligned(fromByteOffset: p, as: UInt32.self) }
        func u64() throws -> UInt64 { try need(8); defer { p += 8 }; return base.loadUnaligned(fromByteOffset: p, as: UInt64.self) }
        func count(_ v: UInt64, _ what: String) throws -> Int {
            guard v <= UInt64(size) else { throw refuse("\(what) \(v) larger than the file") }
            return Int(v)
        }
        func string() throws -> String {
            let n = try count(try u64(), "string length")
            guard n <= maximumString else { throw refuse("a string of \(n) bytes (at byte \(p))") }
            try need(n)
            defer { p += n }
            guard let s = String(bytes: UnsafeRawBufferPointer(start: base + p, count: n), encoding: .utf8) else {
                throw refuse("a string that is not UTF-8 (at byte \(p))")
            }
            return s
        }
        let scalarSizes: [UInt32: Int] = [0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8]
        let valueTypes: [UInt32: String] = [0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
                                           7: "bool", 8: "string", 9: "array", 10: "u64", 11: "i64", 12: "f64"]
        func scalar(_ t: UInt32) throws -> String {
            guard let n = scalarSizes[t] else { throw refuse("metadata value type \(t)") }
            try need(n)
            defer { p += n }
            switch t {
            case 0: return "\(base.load(fromByteOffset: p, as: UInt8.self))"
            case 1: return "\(base.load(fromByteOffset: p, as: Int8.self))"
            case 2: return "\(base.loadUnaligned(fromByteOffset: p, as: UInt16.self))"
            case 3: return "\(base.loadUnaligned(fromByteOffset: p, as: Int16.self))"
            case 4: return "\(base.loadUnaligned(fromByteOffset: p, as: UInt32.self))"
            case 5: return "\(base.loadUnaligned(fromByteOffset: p, as: Int32.self))"
            case 6: return "\(base.loadUnaligned(fromByteOffset: p, as: Float.self))"
            case 7: return base.load(fromByteOffset: p, as: UInt8.self) != 0 ? "true" : "false"
            case 10: return "\(base.loadUnaligned(fromByteOffset: p, as: UInt64.self))"
            case 11: return "\(base.loadUnaligned(fromByteOffset: p, as: Int64.self))"
            default: return "\(base.loadUnaligned(fromByteOffset: p, as: Double.self))"
            }
        }
        /// One metadata value. An array's items are checked and skipped, never kept; an array of
        /// arrays is refused, as llama.cpp's `gguf.cpp` refuses it — which bounds the recursion.
        func value(_ t: UInt32, inArray: Bool = false) throws -> String {
            switch t {
            case 8: return try string()
            case 9:
                guard !inArray else { throw refuse("an array of arrays (at byte \(p))") }
                let itemType = try u32(), n = try count(try u64(), "array length")
                // Each item takes at least its own bytes (a string, its 8-byte length): an array that
                // cannot fit in what remains is refused before one item is read.
                let itemBytes = itemType == 8 ? 8 : (scalarSizes[itemType] ?? 1)
                guard n <= (size - p) / itemBytes else { throw refuse("an array of \(n) items larger than the file") }
                for _ in 0..<n { _ = try value(itemType, inArray: true) }
                return "[\(n) × \(valueTypes[itemType] ?? "\(itemType)")]"
            default: return try scalar(t)
            }
        }

        var h = Header()
        try need(4)
        guard Array(UnsafeRawBufferPointer(start: base, count: 4)) == magic else { throw refuse("not a GGUF file") }
        p = 4
        let version = try u32()
        // Version 1 counted in u32; a big-endian file reads as a huge version.
        guard version == 2 || version == 3 else {
            throw refuse("format version \(version) — only versions 2 and 3, little-endian, are read")
        }
        h.version = Int(version)
        let tensorCount = try count(try u64(), "tensor count"), keyCount = try count(try u64(), "key count")
        for _ in 0..<keyCount {
            let key = try string()
            let t = try u32()
            h.metadata[key] = try value(t)
        }
        if let a = h.metadata["general.alignment"] {
            guard let n = Int(a), n > 0, n <= maximumAlignment, n & (n - 1) == 0 else { throw refuse("general.alignment \(a)") }
            h.alignment = n
        }
        var infos: [(name: String, ne: [Int], type: UInt32, offset: Int)] = []
        for _ in 0..<tensorCount {
            let name = try string()
            let dims = Int(try u32())
            guard (1...4).contains(dims) else { throw refuse("\(name): \(dims) dimensions") }
            var ne: [Int] = []
            for _ in 0..<dims { ne.append(try count(try u64(), "\(name) dimension")) }
            let type = try u32()
            let offset = try count(try u64(), "\(name) offset")
            infos.append((name, ne, type, offset))
        }
        h.dataStart = (p + h.alignment - 1) / h.alignment * h.alignment

        // ── The types: one tensor that is not read refuses the whole file ──────────────────
        var refused: [String: [String]] = [:]
        for i in infos where ![0, 1, 8, 30].contains(i.type) && packedTypes[i.type] == nil {
            refused[typeNames[i.type] ?? "type \(i.type)", default: []].append(i.name)
        }
        if !refused.isEmpty {
            let list = refused.sorted { $0.key < $1.key }
                .map { "\($0.key) on \($0.value.count) tensor(s), e.g. \($0.value[0])" }.joined(separator: "; ")
            throw refuse(list + " — Siliconed reads F32, F16, BF16, Q8_0, Q6_K, Q5_K, Q4_K, Q5_1, Q5_0, Q4_1 and "
                         + "Q4_0, each kept as published; 4 bits is the floor (Q3_K, Q2_K, the I-quants, TQ, MXFP4 "
                         + "and Q8_1 are not read); a file that holds such a tensor is refused in its entirety")
        }

        var seen = Set<String>()
        for i in infos {
            guard seen.insert(i.name).inserted else { throw refuse("\(i.name) present twice") }
            // Each dimension is at most the file's size, but four of them multiply past an `Int`.
            guard let count = Safetensors.byteCount(i.ne, width: 1) else {
                throw refuse("\(i.name): dimensions \(i.ne) overflow")
            }
            let dtype: String, bytes: Int?
            switch i.type {
            case 0: (dtype, bytes) = ("F32", Safetensors.byteCount([count], width: 4))
            case 1: (dtype, bytes) = ("F16", Safetensors.byteCount([count], width: 2))
            case 30: (dtype, bytes) = ("BF16", Safetensors.byteCount([count], width: 2))
            case let t where packedTypes[t] != nil:
                let k = packedTypes[t]!, block = k.dtype.packedBlock!
                guard i.ne[0] % block.values == 0 else {
                    throw refuse("\(i.name): \(k.name) on rows of \(i.ne[0]) values, not a multiple of \(block.values)")
                }
                (dtype, bytes) = (k.name, Safetensors.byteCount([count / block.values], width: block.bytes))
            default:
                guard i.ne[0] % q8Block == 0 else {
                    throw refuse("\(i.name): Q8_0 on rows of \(i.ne[0]) values, not a multiple of \(q8Block)")
                }
                (dtype, bytes) = ("Q8_0", Safetensors.byteCount([count / q8Block], width: q8BlockBytes))
            }
            guard let bytes else { throw refuse("\(i.name): dimensions \(i.ne) overflow") }
            guard i.offset % h.alignment == 0 else { throw refuse("\(i.name): offset \(i.offset) not aligned on \(h.alignment)") }
            // `dataStart` and `offset` are each at most the file's size (plus an alignment), so
            // their sum fits; `bytes` is compared by subtraction.
            let absolute = h.dataStart + i.offset
            guard absolute <= size, bytes <= size - absolute else {
                throw refuse("\(i.name): \(bytes) bytes at \(absolute) run past the end of the file (\(size) bytes)")
            }
            h.entries.append(Safetensors.Entry(name: i.name, dtype: dtype, shape: Array(i.ne.reversed()),
                                               offset: absolute, bytes: bytes))
        }
        // No two tensors may share bytes.
        let sorted = h.entries.sorted { $0.offset < $1.offset }
        for (a, b) in zip(sorted, sorted.dropFirst()) where a.offset + a.bytes > b.offset {
            throw refuse("\(a.name) and \(b.name) overlap")
        }
        return h
    }

    /// One Q8_0 tensor `[rows, columns]`, its interleaved blocks taken apart: the int8 values
    /// `[rows, columns]` and the fp16 scales `[rows, columns/32]`, row-major, bytes unchanged.
    package static func deinterleaveQ8_0(_ source: UnsafeRawPointer, rows: Int, columns: Int) -> (values: [UInt8], scales: [UInt8]) {
        let blocks = rows * columns / q8Block
        let values = [UInt8](unsafeUninitializedCapacity: rows * columns) { v, c in
            c = rows * columns
            for b in 0..<blocks {
                (v.baseAddress! + b * q8Block).initialize(from: (source + b * q8BlockBytes + 2).assumingMemoryBound(to: UInt8.self),
                                                           count: q8Block)
            }
        }
        let scales = [UInt8](unsafeUninitializedCapacity: 2 * blocks) { s, c in
            c = 2 * blocks
            for b in 0..<blocks {
                let d = source + b * q8BlockBytes
                s[2 * b] = d.load(as: UInt8.self)
                s[2 * b + 1] = d.load(fromByteOffset: 1, as: UInt8.self)
            }
        }
        return (values, scales)
    }
}
