import CoreAI
import Foundation
import Metal
import SiliCore

/// **Qwen3's layers on the engine** (`SiliCore.Layers`), what the encoder declares: the stream (and its row factors on
/// f16) and the RoPE tables a row, a group of weights a layer, the palier's layer function; writing, each layer's keys
/// and values for `capacity` rows and the function over one row and the caches (the decode); with a head, its weights
/// by stretches and its function (`Qwen3Head`). The engine plans it under B before anything is allocated — every layer
/// kept where it holds, else a ring in the loop's order —, names, binds and runs it. The CPU's side: the embedding, the
/// tables, the rows written into the stream and read from it, the final norm.
///
/// The loop's order: every layer once (the encoder, the teacher-forced continuation), or every layer over and over (the
/// writer: the prefill, then each token reads every layer).
extension Qwen3Encoder {
    /// What the writer adds beside the layers: the caches' rows and the decode's function.
    struct Writing { let capacity: Int }

    /// The file behind `path`: itself, or the shards its index names (`MapWeights.files`).
    static func file(_ path: String) throws -> SafeTensors {
        let shards = MapWeights.files(path).filter { !$0.hasSuffix(".index.json") }
        guard !shards.isEmpty else { throw ProgramError("\(path): an index that names no shard") }
        return try SafeTensors(shards: shards)
    }

    // MARK: - The declaration

    /// **The layers' declaration**: the stream and its factors, apart — a function holds its last call's outputs past
    /// its own release, and one in the heap holds the heap (measured 19/11: the encoder's last y and y_factor, 320 KiB,
    /// held its 3.4 GB heap 2 s after the stage was let go) —; the tables; a group a layer (the first the slots' layout);
    /// writing, the caches (a layer's own rows, apart for the same reason) and the token's row, tables and mask, read
    /// over and over; the head's.
    func declaration(_ a: Qwen3Arithmetic, rows: Int, writing: Writing?, head: Qwen3Head.Shape?) -> Layers.Declaration {
        let c = config, e = a.element.size, model = self
        var d = Layers.Declaration("Qwen3 \(a.name) on \(rows) rows" + (writing.map { ", \($0.capacity) cached" } ?? ""), rows: rows)
        d.perRow = [.init("x", width: c.width, a.scalar, next: "y", factor: a.half ? "x_factor" : nil, apart: true),
                    .init("cos", width: c.headDim, .float32), .init("sin", width: c.headDim, .float32)]
        if a.half { d.perRow.append(.init("x_factor", width: 1, .float32, next: "y_factor", apart: true)) }
        d.layers = (0..<c.layers).map { Block.Group(Self.layer($0), form: "layer", weights: entries($0, a)) }
        let cached = writing != nil
        d.kinds["layer"] = { b in model.layerModule(a, b.codes, .causal(rows: b.rows), cache: cached) }
        if let writing {
            let n = writing.capacity
            for i in 0..<c.layers {
                d.perRow += ["k", "v"].map { .init("\(Self.layer(i)).\($0)", width: c.kvHeads * c.headDim, a.scalar, rows: n, apart: true) }
            }
            d.tensors = [.init("token", bytes: c.width * e), .init("token_cos", bytes: c.headDim * 4), .init("token_sin", bytes: c.headDim * 4),
                         .init("mask", bytes: (n + 1) * e)]
            if a.half { d.tensors.append(.init("token_factor", bytes: 4)) }
            d.edges["decode"] = { b in model.layerModule(a, b.codes, .cached(capacity: n), cache: true) }
            d.again = true
        }
        if let head { Qwen3Head.declare(head, self, a, in: &d) }
        return d
    }

    /// Layer `i`'s group.
    static func layer(_ i: Int) -> String { "layer\(i)" }

    /// **The layers opened** under B (`MemoryBudget.machineCeiling()`, the footprint now its floor).
    func open(_ file: SafeTensors, _ a: Qwen3Arithmetic, rows: Int, writing: Writing? = nil, head: Qwen3Head.Shape? = nil) async throws -> Layers {
        guard let device = MTLCreateSystemDefaultDevice() else { throw ProgramError("no Metal device") }
        return try await Layers.open(device, declaration(a, rows: rows, writing: writing, head: head), files: [file], budget: MemoryBudget.machineCeiling())
    }

    // MARK: - The token

    /// **The token at `row` through every layer over the caches** (`Layers.through`): the token's row, tables and mask
    /// read by every layer, its keys and values written straight into the cache's row `row` (the same call reads that row
    /// masked: −10⁴, whatever it holds weighs nothing; no other call of the token reads it). What the head reads, as Core
    /// AI's values.
    func decode(_ layers: Layers, _ a: Qwen3Arithmetic, row: Int) async throws -> [String: InferenceFunction.AsyncValue] {
        var entry: [String: Block.Bind] = ["x": .tensor("token")]
        if a.half { entry["x_factor"] = .tensor("token_factor") }
        return try await layers.through("decode", row: row, entry: entry) { layer in
            ["cos": .tensor("token_cos"), "sin": .tensor("token_sin"), "cache_k": .whole("\(layer).k"), "cache_v": .whole("\(layer).v")]
        }
    }

    // MARK: - The CPU's side of the stream

    /// **The sequence's first rows**: the table's rows for `ids`, a picture's rows at its slots; the tables at positions
    /// 0 … (or the pictures' three-axis one); the rows past `ids` zeros (finite: a padded row is never read by a real
    /// one, but 0 × NaN would be).
    func embed(_ layers: Layers, _ a: Qwen3Arithmetic, _ ids: [Int], pictures: Pictures?, table: Table, file: SafeTensors) async throws {
        let d = config.width
        var values = try embedding(ids, table: table, file: file)
        if let pictures {
            var from = 0
            for run in pictures.runs {
                for r in run { for k in 0..<d { values[r * d + k] = pictures.merged[from * d + k] } ; from += 1 }
            }
        }
        await write(layers, a, values, rows: 0..<ids.count)
        let table = rope(rows: layers.declaration.rows)
        var cosValues = table.cos, sinValues = table.sin
        if let pictures {
            cosValues.replaceSubrange(0..<pictures.cos.count, with: pictures.cos)
            sinValues.replaceSubrange(0..<pictures.sin.count, with: pictures.sin)
        }
        let (cos, sin) = (await layers.rows("cos"), await layers.rows("sin"))
        cosValues.withUnsafeBytes { cos.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        sinValues.withUnsafeBytes { sin.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    }

    /// The token `id` at position `position` into the decode's row and tables; `masked`, the mask over the cache.
    func token(_ layers: Layers, _ a: Qwen3Arithmetic, _ id: Int, position: Int, masked: [Float], table: Table, file: SafeTensors) async throws {
        let factors = a.half ? await layers.pointer("token_factor").assumingMemoryBound(to: Float.self) : nil
        Self.write(try embedding([id], table: table, file: file), rows: 0..<1, width: config.width, half: a.half, to: await layers.pointer("token"), factors: factors)
        let table = rope(rows: 1, start: position)
        let (cos, sin) = (await layers.pointer("token_cos"), await layers.pointer("token_sin"))
        table.cos.withUnsafeBytes { cos.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        table.sin.withUnsafeBytes { sin.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        Self.write(masked, as: a.element, to: await layers.pointer("mask"))
    }

    /// True values written into the stream at `range`'s rows (`write(_:rows:width:half:to:factors:)`).
    func write(_ layers: Layers, _ a: Qwen3Arithmetic, _ values: [Float], rows range: Range<Int>) async {
        let factors = a.half ? await layers.rows("x_factor").assumingMemoryBound(to: Float.self) : nil
        Self.write(values, rows: range, width: config.width, half: a.half, to: await layers.rows("x"), factors: factors)
    }

    /// True values written at `range`'s rows: bf16 rounded; f16 each row under F = 2^e, its largest in [2¹³, 2¹⁴]
    /// (`ranged`'s rule for the stream, on the CPU).
    static func write(_ values: [Float], rows range: Range<Int>, width d: Int, half: Bool, to: UnsafeMutableRawPointer,
                      factors: UnsafeMutablePointer<Float>?) {
        for (k, r) in range.enumerated() {
            let row = values[(k * d)..<((k + 1) * d)]
            if half {
                let (unit, f) = RowExponent.split(row, .ranged(f32Factor: true))
                factors![r] = f
                let h = to.assumingMemoryBound(to: Float16.self) + r * d
                for (j, v) in unit.enumerated() { h[j] = Float16(v) }
            } else {
                let b = to.assumingMemoryBound(to: UInt16.self) + r * d
                for (j, v) in row.enumerated() { b[j] = UInt16(truncatingIfNeeded: BFloat.round(v).bitPattern >> 16) }
            }
        }
    }

    /// `values` written in `element` (f16 or bf16, rounded to nearest; a bf16 value is exact) at `to`.
    static func write(_ values: [Float], as element: Aicode.Element, to: UnsafeMutableRawPointer) {
        switch element {
        case .f16: let h = to.assumingMemoryBound(to: Float16.self); for (i, v) in values.enumerated() { h[i] = Float16(v) }
        case .bf16: let b = to.assumingMemoryBound(to: UInt16.self); for (i, v) in values.enumerated() { b[i] = UInt16(truncatingIfNeeded: BFloat.round(v).bitPattern >> 16) }
        default: let f = to.assumingMemoryBound(to: Float.self); for (i, v) in values.enumerated() { f[i] = v }
        }
    }

    /// **A deepstack addition** (`_deepstack_process`): `rows` added at the pictures' slots, as torch adds in bf16 (one
    /// rounding); on f16, the true sum, each row ranged again.
    func add(_ layers: Layers, _ a: Qwen3Arithmetic, _ rows: [Float], at runs: [Range<Int>]) async {
        let d = config.width
        var from = 0
        for run in runs {
            var values = await layers.values("x", rows: run)
            for k in 0..<values.count { values[k] += rows[from * d + k] }
            await write(layers, a, values, rows: run)
            from += run.count
        }
    }

    /// **The model's final norm** (`Qwen3RMSNorm`) over `n` rows of true values: the statistics in f32; in bf16 the
    /// normalized value rounded, then × γ rounded again, as transformers runs it; on f16, in f32.
    func finalNorm(_ values: [Float], rows n: Int, _ a: Qwen3Arithmetic, file: SafeTensors) throws -> [Float] {
        try Self.normed(values, rows: n, gain: file.floats(prefix + "norm.weight"), eps: config.eps, bf16: !a.half)
    }

    static func normed(_ values: [Float], rows n: Int, gain: [Float], eps: Double, bf16: Bool) -> [Float] {
        let d = gain.count
        var out = values
        for r in 0..<n {
            var s: Float = 0
            for j in 0..<d { let v = values[r * d + j]; s += v * v }
            let k = 1 / (s / Float(d) + Float(eps)).squareRoot()
            for j in 0..<d {
                let v = values[r * d + j] * k
                out[r * d + j] = bf16 ? BFloat.round(BFloat.round(v) * gain[j]) : v * gain[j]
            }
        }
        return out
    }
}

extension Qwen3Encoder {
    // MARK: - The table

    /// **The embedding table, its header and scales read once** (made before the stage that reads it, so that its floor
    /// counts it): a bf16 table, or int8 codes with each row's scale and whether ComfyUI rotated them. A token a pass
    /// (`token`) then reads its row of codes alone; the rotation's matrix is the process's (`convrot`).
    struct Table {
        let codes: SafeTensors.Tensor
        /// Each row's scale, as f32, for int8 codes.
        let scales: [Float]?
        let rotated: Bool
    }

    /// R = H₂₅₆ by its entries ±1 (`Aicode.convrotMatrix`), built once: a token's row reads it at every pass.
    static let convrot: [Float] = Aicode.convrotMatrix().entries.map(Float.init)

    func table(_ file: SafeTensors) throws -> Table {
        let name = prefix + "embed_tokens", d = config.width
        guard let table = file.tensors[name + ".weight"] else { throw ProgramError("\(file.path): no tensor \(name).weight") }
        precondition(table.shape == [config.vocabulary, d], "\(name): \(table.shape)")
        switch table.dtype {
        case "BF16": return Table(codes: table, scales: nil, rotated: false)
        case "I8":
            let st = try Weights.scales(name, in: file, out: config.vocabulary), n = config.vocabulary
            var raw = [UInt8](repeating: 0, count: st.end - st.start)
            try raw.withUnsafeMutableBytes { try file.read(st, into: $0.baseAddress!) }
            let scales = raw.withUnsafeBytes { b in
                st.dtype == "F32" ? Array(b.bindMemory(to: Float.self).prefix(n)) : b.bindMemory(to: UInt16.self).prefix(n).map { Float(bitPattern: UInt32($0) << 16) }
            }
            let rotated = try Qwen3Arithmetic.rotated(file, name)
            if rotated { _ = Self.convrot.count }
            return Table(codes: table, scales: scales, rotated: rotated)
        default: throw ProgramError("\(name).weight: \(table.dtype)")
        }
    }

    /// **The table's rows for `ids`**, in f32: a bf16 table exact; an int8 one its codes times their row's scale, and with
    /// ComfyUI's rotation, `(q · s) · R` (R the normalized Hadamard by 256 its codes carry, `Aicode.convrotMatrix`).
    func embedding(_ ids: [Int], table t: Table, file: SafeTensors) throws -> [Float] {
        let d = config.width, table = t.codes
        var out = [Float](repeating: 0, count: ids.count * d)
        guard let scales = t.scales else {
            var raw = [UInt16](repeating: 0, count: d)
            for (r, id) in ids.enumerated() {
                try raw.withUnsafeMutableBytes { try file.read(table, at: id * d * 2, count: d * 2, into: $0.baseAddress!) }
                for c in 0..<d { out[r * d + c] = Float(bitPattern: UInt32(raw[c]) << 16) }
            }
            return out
        }
        let matrix = Self.convrot
        var raw = [Int8](repeating: 0, count: d), row = [Float](repeating: 0, count: d)
        for (r, id) in ids.enumerated() {
            try raw.withUnsafeMutableBytes { try file.read(table, at: id * d, count: d, into: $0.baseAddress!) }
            let s = scales[id]
            for c in 0..<d { row[c] = Float(raw[c]) * s }
            if t.rotated {
                // R = H / 16 by groups of 256, symmetric: the row's group g times R.
                for g in stride(from: 0, to: d, by: 256) {
                    for j in 0..<256 {
                        var acc: Float = 0
                        for i in 0..<256 { acc += row[g + i] * matrix[i * 256 + j] }
                        out[r * d + g + j] = acc / 16
                    }
                }
            } else {
                for c in 0..<d { out[r * d + c] = row[c] }
            }
        }
        return out
    }
}
