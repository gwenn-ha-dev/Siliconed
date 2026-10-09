import Foundation

/// What the forge needs to know of an 8-bit source tensor before reading it: its kind, its scale
/// type (`nil`: none), one scale per row or for the tensor, or one per `block` values along a row;
/// `rotation`: ComfyUI's convrot group along the input (`nil`: none).
package typealias QuantizedKind = (kind: Artifact.DType, scaleType: Artifact.DType?, perRow: Bool, block: Int?, rotation: Int?)

/// **An 8-bit tensor as published, and what the forge may do to it**.
///
/// The bytes and their scales are copied, never recomputed: the only transformations allowed are
/// the ones that *move* bytes without changing any — a transposition into `[input, output]`, a
/// selection or a permutation of rows (a fused qkv split, Anima's RoPE moved to interleaved), a
/// selection of columns (FLUX.2's single-block `to_out`). Each row carries its scale with it; a
/// tensor-wide scale goes with every part. Anything arithmetic (a folded `1 +`) cannot stay 8-bit
/// and is stored dequantized in fp32 by `ForgeDiT`, which says so.
///
/// Layout: `[rows, columns]` = the published `[N, K]` (`N` outputs), row-major. Scales by block
/// (GGUF Q8_0, `block` = 32): `[rows, columns / block]`, row-major — each row carries its own
/// blocks, so a selection of rows moves them with it; a selection of columns keeps them only on a
/// block boundary (`columnsKeepBlocks`).
///
/// **A packed GGUF type** (`.q4_0`…`.q5_1`, `.q4_k`…`.q6_k`): `values` are each row's
/// blocks, bytes as published (`rowBytes` per row, `columns / b` blocks, b = 32 or 256), no
/// separate scale. A selection of rows moves whole rows; a selection of columns moves whole
/// blocks, on a block boundary only (the recipe refuses any other: `Recipes.columns`). The map
/// keeps this layout as is — it is never transposed nor unpacked (`transposedValues` is not for it).
package struct QuantizedTensor {
    /// `.int8` or `.float8_e4m3`; or a packed GGUF type, `.q4_0`…`.q6_k`.
    package var kind: Artifact.DType
    package var rows: Int
    package var columns: Int
    package var values: [UInt8]
    /// The scales' bytes, in their published type; empty: no scale (an fp8 cast, `w = f8`).
    package var scale: [UInt8]
    package var scaleType: Artifact.DType
    /// One scale per row (per output); otherwise one for the tensor — or per block, below.
    package var perRow: Bool
    /// One scale per `block` values along each row (GGUF Q8_0: 32); `nil`: none.
    package var block: Int? = nil
    /// ComfyUI's convrot: the weight is `(q · s) · R`, `R` one block per `rotation`
    /// columns (inputs) — `Widen.dequantizeRotated`. A selection of rows keeps it; a selection of
    /// columns only on a group boundary (`columnsKeepBlocks`). `nil`: no rotation.
    package var rotation: Int? = nil

    var scaleSize: Int { scaleType.size }
    /// Scales per row: 1 (per row), `columns / block` (by block), 0 (one for the tensor or none, or packed).
    var scalesPerRow: Int { block.map { columns / $0 } ?? (perRow ? 1 : 0) }
    /// The bytes of one row of `values`: `columns`, or a packed type's blocks.
    package var rowBytes: Int {
        kind.packedBlock.map { columns / $0.values * $0.bytes } ?? columns
    }

    /// `w = Float(q) · s`, published layout — `Widen.dequantize`, the very routine the engine runs.
    /// By block, the engine reads the map's `[K/b, N]`: here each value is multiplied by its block's
    /// scale one by one — int8 × fp16 is exact in fp32, so there is no rounding to agree on.
    package func dequantized() -> [Float] {
        let n = rows * columns
        if kind.isPacked {
            // ggml's arithmetic, the engine's own routine, in the published layout.
            return [Float](unsafeUninitializedCapacity: n) { b, c in
                c = n
                values.withUnsafeBytes { v in
                    Widen.dequantizePacked(v.baseAddress!, kind: kind, rows: rows, columns: columns,
                                                transposing: false, into: b.baseAddress!)
                }
            }
        }
        if let rotation {
            // Published layout `[N, K]`: the engine's own routine on the transposed bytes (the same
            // exact-then-rounded-once values, over the cores: an import reads 6 G of them), then back.
            precondition(kind == .int8 && block == nil && columns % rotation == 0)
            let t = transposedValues()
            var map = [Float](unsafeUninitializedCapacity: n) { _, c in c = n }
            t.withUnsafeBytes { v in
                scale.withUnsafeBytes { s in
                    map.withUnsafeMutableBufferPointer { d in
                        Widen.dequantizeRotated(v.baseAddress!, rows: columns, columns: rows, group: rotation,
                                                scale: s.baseAddress!, scaleType: scaleType,
                                                layout: perRow ? .columns(rows) : .tensor, into: d.baseAddress!)
                    }
                }
            }
            return [Float](unsafeUninitializedCapacity: n) { b, c in
                c = n
                map.withUnsafeBufferPointer { Numerics.transpose($0.baseAddress!, rows: columns, columns: rows, to: b.baseAddress!) }
            }
        }
        if let block {
            precondition(kind == .int8 && scaleType == .float16)
            let per = columns / block
            return [Float](unsafeUninitializedCapacity: n) { b, c in
                c = n
                values.withUnsafeBytes { v in
                    scale.withUnsafeBytes { s in
                        let q = v.baseAddress!.assumingMemoryBound(to: Int8.self), d = b.baseAddress!
                        for r in 0..<rows {
                            for j in 0..<per {
                                let x = Float(Float16(bitPattern: s.loadUnaligned(fromByteOffset: 2 * (r * per + j), as: UInt16.self)))
                                let first = r * columns + j * block
                                for i in first..<(first + block) { d[i] = Float(q[i]) * x }
                            }
                        }
                    }
                }
            }
        }
        return [Float](unsafeUninitializedCapacity: n) { b, c in
            c = n
            values.withUnsafeBytes { v in
                scale.withUnsafeBytes { s in
                    Widen.dequantize(v.baseAddress!, kind: kind, count: n,
                                     scale: scale.isEmpty ? nil : s.baseAddress!, scaleType: scaleType,
                                     layout: perRow ? .rows(columns: columns) : .tensor, into: b.baseAddress!)
                }
            }
        }
    }

    /// Rows `order[0], order[1], …` — a split (consecutive) or a permutation; scales follow.
    package func selectingRows(_ order: [Int]) -> QuantizedTensor {
        var t = self
        t.rows = order.count
        let width = rowBytes
        t.values = [UInt8](unsafeUninitializedCapacity: order.count * width) { b, c in
            c = order.count * width
            values.withUnsafeBufferPointer { v in
                for (i, r) in order.enumerated() {
                    (b.baseAddress! + i * width).update(from: v.baseAddress! + r * width, count: width)
                }
            }
        }
        let w = scaleSize * scalesPerRow
        if w > 0 { t.scale = order.flatMap { r in scale[(r * w)..<((r + 1) * w)] } }
        return t
    }

    package func rows(begin: Int, number: Int) -> QuantizedTensor { selectingRows(Array(begin..<(begin + number))) }

    /// Can columns `[begin, begin + number)` keep their scales? Always, unless scales go by block
    /// and the share cuts one: a block's scale belongs to all its 32 inputs, half a block would need
    /// a scale of its own that nobody published.
    package static func columnsKeepBlocks(begin: Int, number: Int, block: Int?) -> Bool {
        block.map { begin % $0 == 0 && number % $0 == 0 } ?? true
    }

    /// Columns `[begin, begin + number)` — a share of the **input**: each row keeps its own scale;
    /// by block, the share takes its blocks' scales (on a block boundary only: `columnsKeepBlocks`).
    package func columns(begin: Int, number: Int) -> QuantizedTensor {
        if let packed = kind.packedBlock {
            // Whole blocks of each row, bytes unchanged.
            let (qk, bytes) = packed
            precondition(QuantizedTensor.columnsKeepBlocks(begin: begin, number: number, block: qk))
            var t = self
            t.columns = number
            let width = rowBytes, first = begin / qk * bytes, length = number / qk * bytes
            t.values = (0..<rows).flatMap { r in values[(r * width + first)..<(r * width + first + length)] }
            return t
        }
        precondition(QuantizedTensor.columnsKeepBlocks(begin: begin, number: number, block: block)
                     && QuantizedTensor.columnsKeepBlocks(begin: begin, number: number, block: rotation))
        var t = self
        t.columns = number
        t.values = (0..<rows).flatMap { r in values[(r * columns + begin)..<(r * columns + begin + number)] }
        if let block {
            let w = scaleSize, per = columns / block, first = begin / block, n = number / block
            t.scale = (0..<rows).flatMap { (r: Int) -> ArraySlice<UInt8> in
                let lower: Int = (r * per + first) * w, upper: Int = (r * per + first + n) * w
                return scale[lower..<upper]
            }
        }
        return t
    }

    /// The map's scales by block: `[columns / block, rows]` (`[K/b, N]`) from `[rows, columns / block]`.
    package func transposedBlockScales() -> [UInt8] {
        guard let block else { return scale }
        let per = columns / block, w = scaleSize
        var out = [UInt8](repeating: 0, count: scale.count)
        for r in 0..<rows {
            for j in 0..<per {
                for b in 0..<w { out[(j * rows + r) * w + b] = scale[(r * per + j) * w + b] }
            }
        }
        return out
    }

    /// The map's layout: values `[columns, rows]` (`[K, N]`), the scales untouched — a per-row
    /// scale of `[N, K]` is the per-column scale `[N]` of `[K, N]`.
    package func transposedValues() -> [UInt8] {
        precondition(!kind.isPacked, "a packed GGUF type's blocks keep their rows")
        let (n, k) = (rows, columns)
        return [UInt8](unsafeUninitializedCapacity: n * k) { b, c in
            c = n * k
            let d = b.baseAddress!
            values.withUnsafeBufferPointer { v in
                let s = v.baseAddress!
                // By tiles, so that both sides stay in cache.
                let tile = 64
                for r0 in stride(from: 0, to: n, by: tile) {
                    for c0 in stride(from: 0, to: k, by: tile) {
                        for r in r0..<min(r0 + tile, n) {
                            for col in c0..<min(c0 + tile, k) { d[col * n + r] = s[r * k + col] }
                        }
                    }
                }
            }
        }
    }
}

/// How one published 8-bit weight is laid out in its file: where its values and its scale are.
struct QuantizedGroup {
    let values: (file: Safetensors, entry: Safetensors.Entry)
    let kind: Artifact.DType
    let scale: (file: Safetensors, entry: Safetensors.Entry)?
    let perRow: Bool
    /// What published it, for the journal: `torchao Float8Tensor`, `SDNQ`, `ComfyUI float8_e4m3fn`…
    let format: String
    /// GGUF Q8_0: blocks of `block` values along each row, each behind its fp16 scale, interleaved
    /// in `values` (no separate `scale` entry). `nil`: any other layout.
    var block: Int? = nil
    /// ComfyUI's convrot group (`QuantizedTensor.rotation`); `nil`: none.
    var rotation: Int? = nil

    /// By block (Q8_0, a packed type), a row is the last dimension (`ne[0]`, the input): a 1-D
    /// GGUF tensor is one row.
    var rows: Int { byRow ? values.entry.count / max(columns, 1) : values.entry.shape.first ?? 1 }
    var columns: Int { byRow ? values.entry.shape.last ?? 1 : values.entry.count / max(rows, 1) }
    private var byRow: Bool { block != nil || kind.isPacked }
    var scaleType: Artifact.DType? {
        block != nil ? .float16 : scale.flatMap { QuantizedGroup.scaleType($0.entry.dtype) }
    }

    func read() throws -> QuantizedTensor {
        guard let p = values.file.pointer(values.entry.name) else {
            throw Numerics.Failure(description: "\(values.entry.name) unreadable")
        }
        if kind.isPacked {
            // A packed type: the blocks as published, nothing taken apart. Their scales are checked
            // here, where the bytes are read anyway, not at recognition: blocks of 18–210 bytes put a
            // scale on every page, and a scan there faulted in the whole 4–6 GB file at each open.
            try QuantizedLayouts.checkPacked(values, kind: kind)
            return QuantizedTensor(kind: kind, rows: rows, columns: columns,
                                   values: [UInt8](UnsafeRawBufferPointer(start: p, count: values.entry.bytes)),
                                   scale: [], scaleType: .float32, perRow: false)
        }
        if let block {
            let (v, s) = GGUF.deinterleaveQ8_0(p, rows: rows, columns: columns)
            return QuantizedTensor(kind: kind, rows: rows, columns: columns, values: v, scale: s,
                                   scaleType: .float16, perRow: false, block: block)
        }
        let v = [UInt8](UnsafeRawBufferPointer(start: p, count: values.entry.bytes))
        var s: [UInt8] = [], type: Artifact.DType = .float32
        if let scale {
            s = [UInt8](UnsafeRawBufferPointer(start: scale.file.pointer(scale.entry.name)!, count: scale.entry.bytes))
            type = QuantizedGroup.scaleType(scale.entry.dtype)!
        }
        return QuantizedTensor(kind: kind, rows: rows, columns: columns, values: v, scale: s, scaleType: type, perRow: perRow,
                               rotation: rotation)
    }

    static func scaleType(_ dtype: String) -> Artifact.DType? {
        ["F32": .float32, "BF16": .bfloat16, "F16": .float16][dtype]
    }
}

/// **Reading the published 8-bit layouts** — the names and descriptions each publisher writes,
/// taken from real headers (`tools/fixtures/quant/*/header.json`), never guessed:
///
///   · **ComfyUI** "scaled" / `comfy_quant`: `<l>.weight` F8_E4M3 or I8, `<l>.weight_scale` (older:
///     `<l>.scale_weight`) F32 `[]` or `[N, 1]`, maybe `<l>.input_scale`, maybe `<l>.comfy_quant`
///     (U8 bytes of a JSON `{"format": …}`) or `__metadata__._quantization_metadata`, and the
///     `scaled_fp8` marker. An fp8 without any scale is a plain cast: `w = f8`.
///   · **torchao** (unsloth): `<l>._weight_qdata` + `<l>._weight_scale` F32 `[N, 1]` (+
///     `<l>._weight_zero_point` I8 `[N, 1]` for an `Int8Tensor`), described by a JSON in
///     `__metadata__` under `<l>.weight`: `_type`, `block_size` (`[1, K]` = per row).
///   · **SDNQ**: `<l>.weight` I8 `[N, K]` + `<l>.scale` BF16 `[N, 1]`, symmetric
///     (`dequantize_symmetric`: `weight.to(scale.dtype) * scale`).
///
///   · **ComfyUI int8 convrot**: `comfy_quant` `{"format": "int8_tensorwise", "convrot":
///     true, "convrot_groupsize": 256}` (the flag and the size may also sit under `"params"`, as
///     ComfyUI's loader reads them; size 256 by default), `<l>.weight` I8 `[N, K]` = `W·R` quantized
///     per row, `<l>.weight_scale` F32 `[N, 1]`. Kept as published with its group
///     (`QuantizedTensor.rotation`); the map says `"rotation"`.
///
/// Refused, the whole file with it: anything 4-bit (packed `U8`, `weight_scale_2`, an nvfp4/mxfp4
/// format), mxfp8 (E8M0 exponents), fp8 by 128×128 blocks (`weight_scale_inv`), fp8 E5M2, a `convrot` on anything but a per-row int8 or with a
/// group that is not a power of 4 up to 256 dividing K, an int8 with a nonzero zero point, an SDNQ
/// with SVD factors, a scale of any other shape.
enum QuantizedLayouts {
    struct Recognized {
        var groups: [String: QuantizedGroup] = [:]
        /// Published names that are part of a group or a marker, not tensors of the model.
        var consumed: Set<String> = []
        var notes: [String] = []
    }

    static let fourBitFormats = ["nvfp4", "mxfp4", "int4", "uint4", "fp4", "nf4"]

    static func refuse(_ why: String) -> Numerics.Failure { Numerics.Failure(description: why) }

    /// A layer's convrot group, as ComfyUI's loader reads it (`comfy/ops.py`,
    /// `_load_quantized_module`): `convrot` at the top of the JSON or under `params`, the size beside
    /// it, 256 by default. `nil`: not rotated.
    static func convrotGroup(_ json: [String: Any], layer: String) throws -> Int? {
        let params = json["params"] as? [String: Any] ?? [:]
        func flag(_ v: Any?) -> Bool { (v as? Bool) ?? (v as? NSNumber)?.boolValue ?? false }
        guard flag(json["convrot"] ?? params["convrot"]) else { return nil }
        let size = json["convrot_groupsize"] ?? params["convrot_groupsize"]
        guard let g = size == nil ? 256 : (size as? Int) ?? (size as? NSNumber)?.intValue, Widen.isRotationGroup(g) else {
            throw refuse("\(layer): convrot group \(String(describing: size)) — only a power of 4 up to "
                         + "\(Widen.largestRotationGroup) is read")
        }
        return g
    }

    /// **No product may be subnormal in fp32.** The M1's GPU flushes a subnormal fp32 product to
    /// zero where the CPU keeps it, so `Float(q) · s` below 2⁻¹²⁶ would not be the same weight on
    /// both paths (`Widen.dequantize`). The smallest nonzero |q| is 1 for int8 and 2⁻⁹ for fp8
    /// E4M3 (its smallest subnormal): a nonzero scale under 2⁻¹²⁶ (int8) or 2⁻¹¹⁷ (fp8) could
    /// produce one, and is refused — no published checkpoint comes near (scales ~10⁻⁴).
    /// Rotated (convrot, group G): the smallest nonzero rotated value is `1 / √G`.
    static func checkSubnormal(_ scale: (Safetensors, Safetensors.Entry), kind: Artifact.DType, rotation: Int? = nil) throws {
        let (f, e) = scale
        guard let type = QuantizedGroup.scaleType(e.dtype), let p = f.pointer(e.name) else { return }
        let smallest: Float = (kind == .int8 ? 1 : 0x1p-9) / Float(rotation ?? 1).squareRoot()
        for i in 0..<e.count {
            let v: Float
            switch type {
            case .bfloat16: v = Float(bitPattern: UInt32(p.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self)) << 16)
            case .float16: v = Float(Float16(bitPattern: p.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self)))
            default: v = p.loadUnaligned(fromByteOffset: 4 * i, as: Float.self)
            }
            if v != 0, abs(v) * smallest < Float.leastNormalMagnitude {
                throw refuse("\(e.name): scale \(v) — its products can fall below 2⁻¹²⁶ (subnormal), where the "
                             + "GPU and the CPU would not give the same weight")
            }
        }
    }

    /// **A packed block's fp16 scales must be finite** (`d`, and `dmin` or `m`). No published file
    /// has another; a forged one would give NaN weights whose payload the CPU and the GPU need not
    /// agree on. (Subnormals cannot arise: every nonzero term is a multiple of 2⁻²⁴ — `Widen.dequantizePacked`.)
    static func checkPacked(_ tensor: (Safetensors, Safetensors.Entry), kind: Artifact.DType) throws {
        let (f, e) = tensor
        guard let bytes = kind.packedBlock?.bytes, let p = f.pointer(e.name) else { return }
        // Q6_K: `d` at the end; Q4_K, Q5_K, Q4_1, Q5_1: `d` then `dmin`/`m`; Q4_0, Q5_0: `d` alone.
        let offsets = kind == .q6_k ? [208] : [.q4_0, .q5_0].contains(kind) ? [0] : [0, 2]
        for b in 0..<(e.bytes / bytes) {
            for o in offsets where !Float16(bitPattern: p.loadUnaligned(fromByteOffset: b * bytes + o, as: UInt16.self)).isFinite {
                throw refuse("\(e.name): block \(b) has a non-finite \(o == 2 ? "min" : "d") — not a published "
                             + "\(e.dtype); a file that holds such a tensor is refused in its entirety")
            }
        }
    }

    static func recognize(_ raw: [String: (Safetensors, Safetensors.Entry)], order: [String],
                          metadata: [String: String]) throws -> Recognized {
        var r = Recognized()

        // ── GGUF Q8_0 (`GGUF.parse` has refused every quantized type it does not read, file and all) ─
        // `w = d · q`: int8 × fp16, exact in fp32 and never subnormal (|q| ≥ 1, d ≥ 2⁻²⁴).
        for name in order where raw[name]!.1.dtype == "Q8_0" {
            r.groups[name] = QuantizedGroup(values: raw[name]!, kind: .int8, scale: nil, perRow: false,
                                            format: "GGUF Q8_0", block: GGUF.q8Block)
        }
        // ── GGUF Q4_0…Q5_1 and K-quants: blocks kept whole ────────────────────────
        for name in order {
            guard let k = GGUF.packedTypes.values.first(where: { $0.name == raw[name]!.1.dtype }) else { continue }
            r.groups[name] = QuantizedGroup(values: raw[name]!, kind: k.dtype, scale: nil, perRow: false,
                                            format: "GGUF \(k.name)")
        }

        // ── ComfyUI's per-layer formats: `comfy_quant` tensors, or the metadata's table ─────
        var comfy: [String: [String: Any]] = [:]       // layer → its JSON
        var rotations: [String: Int] = [:]              // layer → its convrot group
        if let text = metadata["_quantization_metadata"], let data = text.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            for (layer, v) in json["layers"] as? [String: Any] ?? [:] {
                if let d = v as? [String: Any] { comfy[layer] = d }
            }
        }
        for name in order where name.hasSuffix(".comfy_quant") {
            let (f, e) = raw[name]!
            let bytes = Data(bytes: f.pointer(name)!, count: e.bytes)
            guard let json = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
                throw refuse("\(name): a `comfy_quant` that is not JSON")
            }
            comfy[String(name.dropLast(".comfy_quant".count))] = json
            r.consumed.insert(name)
        }

        // ── What is refused, before anything is accepted: one 4-bit layer refuses the file ──
        func fourBit(_ name: String, _ what: String) -> Numerics.Failure {
            refuse("4-bit layer (\(what) on \(name)): Siliconed keeps 8-bit weights as published and "
                   + "does not read 4-bit ones; a file that mixes them is refused in its entirety")
        }
        for name in order {
            let e = raw[name]!.1
            if name.hasSuffix(".weight_scale_2") { throw fourBit(name, "a second-level scale, nvfp4") }
            if e.dtype == "U8" && !name.hasSuffix(".comfy_quant") {
                throw refuse("\(name): U8 (4-bit packed, or asymmetric uint8) not supported; a file that holds "
                             + "such a layer is refused in its entirety")
            }
            if e.dtype == "F8_E8M0" {
                throw refuse("mxfp8 (E8M0 block exponents on \(name)): not supported")
            }
            if e.dtype == "F8_E5M2" { throw refuse("fp8 E5M2 (\(name)): not supported — only E4M3 is") }
            // The official fp8 of Qwen and DeepSeek: `w = f8 · s[⌊n/128⌋, ⌊k/128⌋]`, one scale per
            // 128×128 block. No scale name below matches it: the weight would be read as a plain
            // cast `w = f8` and the scale kept as a tensor of the model — wrong weights, silently.
            if name.hasSuffix(".weight_scale_inv") {
                throw refuse("fp8 by blocks of 128×128 (`weight_scale_inv` on \(name), the format of Qwen's and DeepSeek's "
                             + "official fp8): not supported — a scale per 2-D block is not one per tensor or per row, "
                             + "and reading the weight without it would give wrong values")
            }
        }
        for (layer, json) in comfy.sorted(by: { $0.key < $1.key }) {
            let format = (json["format"] as? String ?? "").lowercased()
            if fourBitFormats.contains(where: { format.contains($0) }) { throw fourBit(layer, format) }
            if format.contains("mxfp8") { throw refuse("mxfp8 (\(layer)): not supported") }
            guard ["float8_e4m3fn", "int8_tensorwise"].contains(format) else {
                throw refuse("\(layer): ComfyUI quantization format \"\(format)\" not supported")
            }
            if let g = try convrotGroup(json, layer: layer) {
                guard format == "int8_tensorwise" else {
                    throw refuse("\(layer): `convrot` on \"\(format)\" — ComfyUI rotates only int8_tensorwise")
                }
                rotations[layer] = g
            }
        }

        // ComfyUI's `scaled_fp8` marker is an *empty* F8_E4M3 tensor: a flag, not a weight.
        if raw["scaled_fp8"] != nil { r.consumed.insert("scaled_fp8") }

        // ── torchao (unsloth): `_weight_qdata`, described in the metadata ─────────────────
        var activationsNoted = false
        for name in order where name.hasSuffix("._weight_qdata") {
            let base = String(name.dropLast("._weight_qdata".count))
            let logical = base + ".weight"
            let values = raw[name]!
            guard let text = metadata[logical], let data = text.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let type = json["_type"] as? String, let d = json["_data"] as? [String: Any] else {
                throw refuse("\(name): a torchao tensor without its description in the metadata")
            }
            let parts = Set(json["_tensor_data_names"] as? [String] ?? [])
            let kind: Artifact.DType
            switch (type, values.1.dtype) {
            case ("Float8Tensor", "F8_E4M3"): kind = .float8_e4m3
            case ("Int8Tensor", "I8"): kind = .int8
            default: throw refuse("\(logical): torchao \(type) of \(values.1.dtype) — not supported")
            }
            guard parts.isSubset(of: ["qdata", "scale", "zero_point"]), parts.contains("scale") else {
                throw refuse("\(logical): torchao parts \(parts.sorted()) — only qdata, scale and a zero point are read")
            }
            if kind == .float8_e4m3, let dt = (d["float8_dtype"] as? [String: Any])?["_data"] as? String,
               dt != "float8_e4m3fn" {
                throw refuse("\(logical): torchao float8 \(dt) — only float8_e4m3fn is read")
            }
            guard let scale = raw[base + "._weight_scale"], QuantizedGroup.scaleType(scale.1.dtype) != nil else {
                throw refuse("\(logical): torchao scale missing or of an unreadable type")
            }
            let shape = values.1.shape, block = d["block_size"] as? [Int] ?? []
            let perRow: Bool
            if shape.count == 2, block == [1, shape[1]], scale.1.count == shape[0] { perRow = true }
            else if block == shape, scale.1.count == 1 { perRow = false }
            else { throw refuse("\(logical): torchao block \(block) on \(shape) — only per row or per tensor") }
            // An `Int8Tensor`'s zero point: torchao's symmetric weights carry one, all zeros
            // (`choose_qparams_affine`, SYMMETRIC). `(q − 0) · s` is `q · s`; anything else would need
            // nine bits, so it is checked, not assumed.
            if let zp = raw[base + "._weight_zero_point"] {
                guard zp.1.dtype == "I8", zp.1.count == scale.1.count else {
                    throw refuse("\(logical): zero point \(zp.1.dtype) \(zp.1.shape) — not read")
                }
                let p = zp.0.pointer(zp.1.name)!.assumingMemoryBound(to: UInt8.self)
                if let i = (0..<zp.1.count).first(where: { p[$0] != 0 }) {
                    throw refuse("\(logical): asymmetric int8 (zero point \(Int8(bitPattern: p[i])) on row \(i)) — "
                                 + "not supported: `q − z` does not fit in 8 bits")
                }
                r.consumed.insert(zp.1.name)
            } else if parts.contains("zero_point") {
                throw refuse("\(logical): its zero point is declared but absent")
            }
            if d["act_quant_kwargs"] is [String: Any], !activationsNoted {
                activationsNoted = true
                r.notes.append("torchao declares an 8-bit quantization of the activations (`act_quant_kwargs`): "
                               + "ignored — activations stay in fp32")
            }
            try checkSubnormal(scale, kind: kind)
            r.consumed.formUnion([name, scale.1.name])
            r.groups[logical] = QuantizedGroup(values: values, kind: kind, scale: scale, perRow: perRow,
                                               format: "torchao \(type)")
        }
        if let stray = order.first(where: { $0.contains("._weight_") && !r.consumed.contains($0) }) {
            throw refuse("\(stray): a torchao component that is not read")
        }

        // ── ComfyUI, SDNQ, plain fp8: the values under the weight's own name ───────────────
        var inputScales = 0
        for name in order {
            let (_, e) = raw[name]!
            // An empty 8-bit tensor carries no weight (a marker): left to the recipe, never grouped.
            guard e.dtype == "F8_E4M3" || e.dtype == "I8", e.count > 0, !r.consumed.contains(name) else { continue }
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            let kind: Artifact.DType = e.dtype == "I8" ? .int8 : .float8_e4m3
            for extra in [".zero_point", ".svd_up", ".svd_down"] where raw[base + extra] != nil {
                throw refuse("\(name): SDNQ with `\(extra.dropFirst())` — not supported (only symmetric int8)")
            }
            // SDNQ's `<l>.scale` (int8 or fp8: `w = q · s` either way). Only beside a `<l>.weight`:
            // `prenorm.scale`, `query_norm.scale` (Krea 2, FLUX.2) are norms of their own.
            let candidates = [".weight_scale", ".scale_weight"] + (name.hasSuffix(".weight") ? [".scale"] : [])
            let found = candidates.compactMap { raw[base + $0] }
            guard found.count <= 1 else { throw refuse("\(name): two scales") }
            let scale = found.first
            if let json = comfy[base] {
                let format = json["format"] as? String ?? ""
                guard (format == "float8_e4m3fn") == (kind == .float8_e4m3) else {
                    throw refuse("\(name): \(e.dtype) declared as \"\(format)\"")
                }
            }
            guard scale != nil || kind == .float8_e4m3 else {
                throw refuse("\(name): int8 without a scale")
            }
            if scale == nil, comfy[base] != nil {
                throw refuse("\(name): declared \"float8_e4m3fn\" by ComfyUI but published without its "
                             + "`weight_scale` — reading it as a plain cast would be a guess")
            }
            var perRow = false
            if let s = scale {
                guard QuantizedGroup.scaleType(s.1.dtype) != nil else {
                    throw refuse("\(s.1.name): a scale of type \(s.1.dtype)")
                }
                let n = e.shape.first ?? 1
                if s.1.count == 1 { perRow = false }
                else if s.1.count == n, s.1.shape == [n, 1] || s.1.shape == [n] { perRow = true }
                else {
                    throw refuse("\(s.1.name): a scale of shape \(s.1.shape) for \(e.shape) — only one per "
                                 + "tensor or one per row is read")
                }
                try checkSubnormal(s, kind: kind)
                r.consumed.insert(s.1.name)
            }
            let rotation = rotations[base]
            if let g = rotation {
                guard kind == .int8, perRow, e.shape.count == 2, e.shape[1] % g == 0 else {
                    throw refuse("\(name): convrot of group \(g) on \(e.dtype) \(e.shape) "
                                 + "(\(perRow ? "per row" : "one scale")) — only a per-row int8 whose input divides by the group")
                }
                if let s = scale { try checkSubnormal(s, kind: kind, rotation: g) }
            }
            let format = comfy[base].map { "ComfyUI \($0["format"] as? String ?? "")" + (rotation != nil ? " convrot" : "") }
                ?? (scale?.1.name.hasSuffix(".scale") == true ? "SDNQ" : scale == nil ? "fp8 cast" : "ComfyUI scaled")
            var group = QuantizedGroup(values: raw[name]!, kind: kind, scale: scale, perRow: perRow, format: format)
            group.rotation = rotation
            r.groups[name] = group
        }
        for name in order where !r.consumed.contains(name) {
            if name == "scaled_fp8" { r.consumed.insert(name) }
            else if name.hasSuffix(".input_scale") || name.hasSuffix(".scale_input") {
                inputScales += 1
                r.consumed.insert(name)
            } else if (name.hasSuffix(".weight_scale") || name.hasSuffix(".scale_weight")) {
                // A scale whose weight is not 8-bit: nothing to multiply (as before the 8-bit maps).
                r.consumed.insert(name)
                r.notes.append("\(name): a scale without an 8-bit weight — ignored")
            }
        }
        if inputScales > 0 {
            r.notes.append("\(inputScales) `input_scale` ignored: activations never go down to 8 bits")
        }
        return r
    }
}
