import Accelerate
import Metal
import XCTest
@testable import Siliconed

/// **An 8-bit checkpoint becomes an 8-bit map** — on synthetic
/// safetensors written here, one per published layout: the map keeps the published bytes (only
/// transposed) and their scales, and the engine's dequantization, CPU and GPU, is `Float(q) · s`
/// **bit for bit**.
final class QuantizedMapTests: XCTestCase {
    // ── writing synthetic files ───────────────────────────────────────────────────────────

    struct Piece { let name: String; let dtype: String; let shape: [Int]; let bytes: [UInt8] }

    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-8bit-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private func write(_ pieces: [Piece], metadata: [String: String] = [:], as file: String = "model.safetensors") throws -> String {
        var header: [String: Any] = [:]
        var position = 0
        for p in pieces {
            header[p.name] = ["dtype": p.dtype, "shape": p.shape, "data_offsets": [position, position + p.bytes.count]]
            position += p.bytes.count
        }
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        var data = withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) } + json
        for p in pieces { data.append(contentsOf: p.bytes) }
        let path = root.appendingPathComponent(file).path
        try data.write(to: URL(fileURLWithPath: path))
        return path
    }

    private var generator = SystemRandomNumberGenerator()

    /// Random 8-bit values; fp8 without its NaN codes (the conversion test covers them).
    private func values(_ count: Int, fp8: Bool) -> [UInt8] {
        (0..<count).map { _ in
            var b = UInt8.random(in: 0...255, using: &generator)
            if fp8 && b & 0x7F == 0x7F { b ^= 0x01 }
            return b
        }
    }

    /// Scales in their published type: fp32 ones with a full mantissa (the product rounds), bf16
    /// ones as bf16 (the product is exact).
    private func scales(_ count: Int, type: String) -> [UInt8] {
        (0..<count).flatMap { _ -> [UInt8] in
            let s = Float.random(in: 1e-4...3e-2, using: &generator)
            switch type {
            case "BF16": return withUnsafeBytes(of: UInt16(s.bitPattern >> 16).littleEndian) { Array($0) }
            case "F16": return withUnsafeBytes(of: Float16(s).bitPattern.littleEndian) { Array($0) }
            default: return withUnsafeBytes(of: s.bitPattern.littleEndian) { Array($0) }
            }
        }
    }

    private func scaleValue(_ bytes: [UInt8], _ i: Int, type: String) -> Float {
        switch type {
        case "BF16": return Float(bitPattern: UInt32(UInt16(bytes[2 * i]) | UInt16(bytes[2 * i + 1]) << 8) << 16)
        case "F16": return Float(Float16(bitPattern: UInt16(bytes[2 * i]) | UInt16(bytes[2 * i + 1]) << 8))
        default: return bytes[(4 * i)..<(4 * i + 4)].withUnsafeBytes { Float(bitPattern: $0.loadUnaligned(as: UInt32.self)) }
        }
    }

    /// The reference, written independently of the engine: `Float(q) · s`, map layout `[K, N]`.
    private func reference(_ q: [UInt8], rows n: Int, columns k: Int, fp8: Bool,
                           scale: [UInt8], type: String, perRow: Bool) -> [Float] {
        var out = [Float](repeating: 0, count: n * k)
        for r in 0..<n {
            let s: Float = scale.isEmpty ? 1 : scaleValue(scale, perRow ? r : 0, type: type)
            for c in 0..<k {
                let b = q[r * k + c]
                let v = fp8 ? Numerics.e4m3[Int(b)] : Float(Int8(bitPattern: b))
                out[c * n + r] = scale.isEmpty ? v : v * s
            }
        }
        return out
    }

    private func transposed(_ q: [UInt8], rows n: Int, columns k: Int) -> [UInt8] {
        var t = [UInt8](repeating: 0, count: n * k)
        for r in 0..<n { for c in 0..<k { t[c * n + r] = q[r * k + c] } }
        return t
    }

    /// Normalizes and forges a source the way `ForgeDiT.forge` does, tensor by tensor, into a map.
    private func forge(_ source: TensorSource, family: Family) throws -> (Artifact, [String: ForgeDiT.Outcome]) {
        let n = try Recipes.normalize(source, family: family)
        var tensors: [MapWriter.Tensor] = [], outcomes: [String: ForgeDiT.Outcome] = [:]
        for name in n.dit.keys.sorted() {
            let (t, o) = try ForgeDiT.tensor(name, n.dit[name]!, family: family)
            tensors.append(t)
            outcomes[name] = o
        }
        let path = root.appendingPathComponent("map-\(UUID()).silicon").path
        var planned = 0
        let tally = try MapWriter.write(to: path, tensors: tensors, header: { _ in
            [.init("format", 1), .init("page", .integer(MapWriter.page)), .init("linear_weights_transposed", true)]
        }, reserve: { planned = $0 })
        XCTAssertEqual(tally.inexact, 0, "the forge never rounds")
        XCTAssertEqual(planned, tally.bytes, "the disk is checked against the real map")
        return (try Artifact(path: path), outcomes)
    }

    private lazy var gpu: (WidenGPU, MTLDevice)? = {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let w = try? WidenGPU(device: device, queue: queue) else { return nil }
        return (w, device)
    }()

    /// The map's tensor dequantized on the CPU (`materialize`) and, when there is a GPU, by
    /// `WidenGPU` — both equal to `expected`, bit for bit.
    private func assertDequantizes(_ map: Artifact, _ name: String, to expected: [Float],
                                   file: StaticString = #filePath, line: UInt = #line) throws {
        let t = try XCTUnwrap(map.tensors[name], file: file, line: line)
        var cpu = [Float](repeating: -1, count: t.count)
        try cpu.withUnsafeMutableBufferPointer { _ = try map.materialize(name, into: $0) }
        XCTAssertEqual(cpu.map(\.bitPattern), expected.map(\.bitPattern), "CPU \(name)", file: file, line: line)
        guard let (widener, device) = gpu else { return }
        guard let source = widener.source(name, in: map) else {
            // The one case `WidenGPU` declines: a page-rounded wrapper that would run past the end
            // of the file (the map's last tensor) — the CPU path, as for bf16.
            let rounded = (t.bytes + map.page - 1) / map.page * map.page
            XCTAssertGreaterThan(t.offset + rounded, map.size, "GPU path for \(name)", file: file, line: line)
            return
        }
        let buffer = try XCTUnwrap(device.makeBuffer(length: t.count * 4, options: .storageModeShared))
        try widener.run(source: source, destination: buffer, count: t.count)
        let g = buffer.contents().assumingMemoryBound(to: Float.self)
        XCTAssertEqual((0..<t.count).map { g[$0].bitPattern }, expected.map(\.bitPattern), "GPU \(name)", file: file, line: line)
    }

    private func mapBytes(_ map: Artifact, _ name: String) -> (values: [UInt8], scale: [UInt8]) {
        let t = map.tensors[name]!, p = map.pointer(name)!
        let values = [UInt8](UnsafeRawBufferPointer(start: p, count: t.count))
        let scale = t.scale.map { [UInt8](UnsafeRawBufferPointer(start: p + $0.offset, count: $0.count * $0.dtype.size)) } ?? []
        return (values, scale)
    }

    // ── the conversions ────────────────────────────────────────────────────────────────────

    /// fp8 E4M3 → fp32 through fp16: equal to the table on all 256 codes, NaN and subnormals included.
    func testE4M3WideningIsTheTable() {
        let codes = (0..<256).map(UInt8.init)
        var out = [Float](repeating: 0, count: 256)
        codes.withUnsafeBufferPointer { c in out.withUnsafeMutableBufferPointer {
            Widen.float8E4M3ToFloat32(c.baseAddress!, count: 256, into: $0.baseAddress!) } }
        for b in 0..<256 {
            let t = Numerics.e4m3[b]
            if t.isNaN { XCTAssertTrue(out[b].isNaN, "code \(b)") } else { XCTAssertEqual(out[b].bitPattern, t.bitPattern, "code \(b)") }
        }
    }

    // ── each published layout ─────────────────────────────────────────────────────────────

    /// One layout: a fused qkv (split into three by the Z-Image recipe — rows and their scales
    /// move together), a plain linear, a bf16 norm (kept bf16, byte for byte) and an fp32 tensor
    /// that bf16 cannot carry (kept fp32: the forge never rounds).
    private func checkLayout(fp8: Bool, scaleType: String, perRow: Bool,
                             naming: (_ base: String) -> (values: String, scale: String),
                             extra: (_ base: String, _ n: Int) -> [Piece] = { _, _ in [] },
                             more: [Piece] = [],
                             metadata: (_ bases: [String: [Int]]) -> [String: String] = { _ in [:] }) throws {
        // Counts a multiple of 4 (the GPU path) but not of 16: the scale starts after a padding.
        let (n, k) = (6, 6)
        let qkv = values(3 * n * k, fp8: fp8), qkvScale = scales(perRow ? 3 * n : 1, type: scaleType)
        let w1 = values(2 * n * k, fp8: fp8), w1Scale = scales(perRow ? 2 * n : 1, type: scaleType)
        let dtype = fp8 ? "F8_E4M3" : "I8"
        func scaleShape(_ rows: Int) -> [Int] { perRow ? [rows, 1] : [] }
        let norm: [UInt8] = (0..<k).flatMap { i in withUnsafeBytes(of: UInt16(0x3F80 + i).littleEndian) { Array($0) } }
        let inexact: [UInt8] = (0..<k).flatMap { i in withUnsafeBytes(of: (Float(1) + Float(i) / 1000).bitPattern) { Array($0) } }
        let q = naming("layers.0.attention.qkv"), f = naming("layers.0.feed_forward.w1")
        var pieces = [
            Piece(name: q.values, dtype: dtype, shape: [3 * n, k], bytes: qkv),
            Piece(name: q.scale, dtype: scaleType, shape: scaleShape(3 * n), bytes: qkvScale),
            Piece(name: f.values, dtype: dtype, shape: [2 * n, k], bytes: w1),
            Piece(name: f.scale, dtype: scaleType, shape: scaleShape(2 * n), bytes: w1Scale),
            Piece(name: "layers.0.attention_norm1.weight", dtype: "BF16", shape: [k], bytes: norm),
            Piece(name: "layers.0.ffn_norm1.weight", dtype: "F32", shape: [k], bytes: inexact),
            Piece(name: "cap_embedder.0.weight", dtype: "BF16", shape: [k], bytes: norm),
        ]
        pieces += extra("layers.0.attention.qkv", 3 * n) + extra("layers.0.feed_forward.w1", 2 * n) + more
        let path = try write(pieces, metadata: metadata(["layers.0.attention.qkv": [3 * n, k], "layers.0.feed_forward.w1": [2 * n, k]]))
        let source = try TensorSource(paths: [path])
        XCTAssertEqual(Recipes.family(fromNames: source.names), .zImage)
        XCTAssertFalse(source.names.contains("scaled_fp8"), "a marker, not a tensor of the model")
        let (map, outcomes) = try forge(source, family: .zImage)

        let kind: Artifact.DType = fp8 ? .float8_e4m3 : .int8
        let scaleDType: Artifact.DType = ["F32": .float32, "BF16": .bfloat16, "F16": .float16][scaleType]!
        for (part, name) in ["to_q", "to_k", "to_v"].enumerated() {
            let mapName = "layers.0.attention.\(name).weight"
            XCTAssertEqual(outcomes[mapName], .kept8bit)
            let t = try XCTUnwrap(map.tensors[mapName])
            XCTAssertEqual(t.dtype, kind)
            XCTAssertEqual(t.shape, [k, n])
            XCTAssertEqual(t.scale?.dtype, scaleDType)
            XCTAssertEqual(t.scale?.shape, perRow ? [n] : [])
            XCTAssertEqual(t.scale?.offset, 48, "36 values, then 12 bytes of padding, then the scales")
            let rows = Array(qkv[(part * n * k)..<((part + 1) * n * k)])
            let s = perRow ? Array(qkvScale[(part * n * scaleDType.size)..<((part + 1) * n * scaleDType.size)]) : qkvScale
            let stored = mapBytes(map, mapName)
            XCTAssertEqual(stored.values, transposed(rows, rows: n, columns: k), "the published bytes, transposed")
            XCTAssertEqual(stored.scale, s, "the published scales, in their type")
            try assertDequantizes(map, mapName, to: reference(rows, rows: n, columns: k, fp8: fp8, scale: s, type: scaleType, perRow: perRow))
        }
        let w = "layers.0.feed_forward.w1.weight"
        XCTAssertEqual(mapBytes(map, w).values, transposed(w1, rows: 2 * n, columns: k))
        try assertDequantizes(map, w, to: reference(w1, rows: 2 * n, columns: k, fp8: fp8, scale: w1Scale, type: scaleType, perRow: perRow))
        // The source's own dequantization (what a transformation falling back to fp32 would store)
        // is the same routine: the same bits, in the published layout.
        let read = try source.read(f.values.hasSuffix("._weight_qdata") ? w : f.values)
        let ref = reference(w1, rows: 2 * n, columns: k, fp8: fp8, scale: w1Scale, type: scaleType, perRow: perRow)
        XCTAssertEqual(read.map(\.bitPattern), (0..<(2 * n * k)).map { i in ref[(i % k) * 2 * n + i / k].bitPattern })

        XCTAssertEqual(map.tensors["layers.0.attention_norm1.weight"]?.dtype, .bfloat16)
        XCTAssertEqual([UInt8](UnsafeRawBufferPointer(start: map.pointer("layers.0.attention_norm1.weight")!, count: 2 * k)), norm)
        XCTAssertEqual(outcomes["layers.0.ffn_norm1.weight"], .widened)
        XCTAssertEqual(map.tensors["layers.0.ffn_norm1.weight"]?.dtype, .float32, "not exact in bf16: kept fp32")
        XCTAssertEqual([UInt8](UnsafeRawBufferPointer(start: map.pointer("layers.0.ffn_norm1.weight")!, count: 4 * k)), inexact)

        // One row of a table read alone takes its scales (`materializeRows`).
        var two = [Float](repeating: 0, count: 2 * n)
        try two.withUnsafeMutableBufferPointer { try map.materializeRows(w, first: 3, count: 1, into: $0.baseAddress!) }
        let full = reference(w1, rows: 2 * n, columns: k, fp8: fp8, scale: w1Scale, type: scaleType, perRow: perRow)
        XCTAssertEqual(two.map(\.bitPattern), Array(full[(3 * 2 * n)..<(4 * 2 * n)]).map(\.bitPattern))
    }

    /// ComfyUI "scaled" fp8: `weight` F8_E4M3 + `weight_scale` F32 `[]`, an `input_scale`, a
    /// `comfy_quant` JSON, the `scaled_fp8` marker.
    func testFP8ScaledPerTensorComfyUI() throws {
        try checkLayout(fp8: true, scaleType: "F32", perRow: false,
                        naming: { ($0 + ".weight", $0 + ".weight_scale") },
                        extra: { base, _ in
                            [Piece(name: base + ".input_scale", dtype: "F32", shape: [], bytes: [0, 0, 0x80, 0x3F]),
                             Piece(name: base + ".comfy_quant", dtype: "U8", shape: [27],
                                   bytes: Array(#"{"format": "float8_e4m3fn"}"#.utf8))]
                        },
                        more: [Piece(name: "scaled_fp8", dtype: "F8_E4M3", shape: [0], bytes: [])])
    }

    /// torchao `Float8Tensor` (unsloth): `_weight_qdata` + `_weight_scale` F32 `[N, 1]`, described in
    /// the metadata.
    func testFP8PerRowTorchao() throws {
        try checkLayout(fp8: true, scaleType: "F32", perRow: true,
                        naming: { ($0 + "._weight_qdata", $0 + "._weight_scale") },
                        metadata: { bases in torchao(bases, type: "Float8Tensor", parts: ["qdata", "scale"]) })
    }

    /// SDNQ int8: `weight` I8 + `scale` BF16 `[N, 1]`.
    func testInt8PerRowBF16SDNQ() throws {
        try checkLayout(fp8: false, scaleType: "BF16", perRow: true, naming: { ($0 + ".weight", $0 + ".scale") })
    }

    /// torchao `Int8Tensor` (unsloth): F32 scale per row and a zero point that is all zeros.
    func testInt8PerRowF32Torchao() throws {
        try checkLayout(fp8: false, scaleType: "F32", perRow: true,
                        naming: { ($0 + "._weight_qdata", $0 + "._weight_scale") },
                        extra: { base, n in [Piece(name: base + "._weight_zero_point", dtype: "I8", shape: [n, 1],
                                                   bytes: [UInt8](repeating: 0, count: n))] },
                        metadata: { bases in torchao(bases, type: "Int8Tensor", parts: ["zero_point", "qdata", "scale"]) })
    }

    /// ComfyUI int8 per row: `weight` I8 + `weight_scale` F32 `[N, 1]`, `int8_tensorwise` without convrot.
    func testInt8PerRowComfyUI() throws {
        try checkLayout(fp8: false, scaleType: "F32", perRow: true,
                        naming: { ($0 + ".weight", $0 + ".weight_scale") },
                        extra: { base, _ in
                            let json = Array(#"{"format": "int8_tensorwise"}"#.utf8)
                            return [Piece(name: base + ".comfy_quant", dtype: "U8", shape: [json.count], bytes: json)]
                        })
    }

    /// An fp16 scale (GGUF's type) goes through the same path.
    func testInt8PerTensorF16Scale() throws {
        try checkLayout(fp8: false, scaleType: "F16", perRow: false, naming: { ($0 + ".weight", $0 + ".weight_scale") })
    }

    private func torchao(_ bases: [String: [Int]], type: String, parts: [String]) -> [String: String] {
        var m: [String: String] = [:]
        for (base, shape) in bases {
            m[base + ".weight"] = #"{"_type": "\#(type)", "_data": {"block_size": [1, \#(shape[1])], "act_quant_kwargs": {"_type": "x", "_data": {}}, "dtype": {"_type": "torch.dtype", "_data": "bfloat16"}}, "_tensor_data_names": [\#(parts.map { "\"\($0)\"" }.joined(separator: ", "))]}"#
        }
        return m
    }

    // ── the transformations that move rows or columns ─────────────────────────────────────

    /// Anima's RoPE permutation on an 8-bit `to_q` (rows **and** their scales move), and FLUX.2's
    /// single-block `to_out` split on its **input** (each row keeps its scale).
    func testPermutationAndColumnSplit() throws {
        let (n, k) = (128, 4)
        let q = values(n * k, fp8: false), s = scales(n, type: "F32")
        let path = try write([Piece(name: "transformer_blocks.0.attn1.to_q.weight", dtype: "I8", shape: [n, k], bytes: q),
                              Piece(name: "transformer_blocks.0.attn1.to_q.weight_scale", dtype: "F32", shape: [n, 1], bytes: s),
                              Piece(name: "transformer_blocks.0.norm1.weight", dtype: "BF16", shape: [2], bytes: [0x80, 0x3F, 0x80, 0x3F])],
                             as: "anima.safetensors")
        let (map, _) = try forge(try TensorSource(paths: [path]), family: .anima)
        let order = ForgeDiT.interleaveOrder(rows: n, head: ForgeDiT.animaHeadSize)
        let pq = order.flatMap { q[($0 * k)..<(($0 + 1) * k)] }, ps = order.flatMap { s[($0 * 4)..<(($0 + 1) * 4)] }
        XCTAssertEqual(mapBytes(map, "transformer_blocks.0.attn1.to_q.weight").scale, ps)
        try assertDequantizes(map, "transformer_blocks.0.attn1.to_q.weight",
                              to: reference(pq, rows: n, columns: k, fp8: false, scale: ps, type: "F32", perRow: true))

        let (d, m) = (4, 8)                                   // to_out: input = attention (d) + MLP (m − d)
        let o = values(d * m, fp8: true), os = scales(d, type: "BF16")
        let path2 = try write([Piece(name: "single_transformer_blocks.0.attn.to_out.weight", dtype: "F8_E4M3", shape: [d, m], bytes: o),
                               Piece(name: "single_transformer_blocks.0.attn.to_out.weight_scale", dtype: "BF16", shape: [d, 1], bytes: os)],
                              as: "klein.safetensors")
        let (map2, outcomes) = try forge(try TensorSource(paths: [path2]), family: .klein4b)
        XCTAssertEqual(outcomes["single_transformer_blocks.0.attn.to_out.mlp.weight"], .kept8bit)
        let mlp = (0..<d).flatMap { r in o[(r * m + d)..<((r + 1) * m)] }
        try assertDequantizes(map2, "single_transformer_blocks.0.attn.to_out.mlp.weight",
                              to: reference(mlp, rows: d, columns: m - d, fp8: true, scale: os, type: "BF16", perRow: true))
    }

    /// A folded `1 +` (Qwen-Image-2.1's zero-centered norm) on an 8-bit tensor cannot stay 8-bit:
    /// it is stored dequantized in **fp32**, `Float(q) · s + 1`, and the outcome says why.
    func testAnArithmeticTransformationFallsBackToFP32() throws {
        let q = values(8, fp8: true)
        let path = try write([Piece(name: "txt_in.text_norm.weight", dtype: "F8_E4M3", shape: [8], bytes: q)], as: "qwen.safetensors")
        let source = try TensorSource(paths: [path])
        let n = try Recipes.normalize(source, family: .qwenImage21)
        let (t, outcome) = try ForgeDiT.tensor("txt_in.text_norm.weight", n.dit["txt_in.text_norm.weight"]!, family: .qwenImage21)
        XCTAssertEqual(t.dtype, .float32)
        XCTAssertEqual(outcome, .dequantized(" (folded `1 +`)"))
        XCTAssertEqual(try t.produce().map(\.bitPattern), q.map { (Numerics.e4m3[Int($0)] + 1).bitPattern })
    }

    // ── what is refused ───────────────────────────────────────────────────────────────────

    private func refusal(_ pieces: [Piece], metadata: [String: String] = [:]) throws -> String {
        let path = try write(pieces, metadata: metadata, as: "refused-\(UUID()).safetensors")
        do { _ = try TensorSource(paths: [path]); XCTFail("accepted"); return "" }
        catch let e as Numerics.Failure { return e.description }
    }

    private let fp8Layer = [Piece(name: "layers.0.attention.to_q.weight", dtype: "F8_E4M3", shape: [2, 2], bytes: [0x38, 0x40, 0xB8, 0]),
                            Piece(name: "layers.0.attention.to_q.weight_scale", dtype: "F32", shape: [], bytes: [0, 0, 0x80, 0x3F])]

    func testFourBitIsRefusedAndAMixedFileWithIt() throws {
        // nvfp4 (ComfyUI / BFL): packed U8 values, an fp8 block scale, a second-level fp32 scale.
        let nvfp4 = [Piece(name: "layers.1.attention.to_q.weight", dtype: "U8", shape: [2, 1], bytes: [0x12, 0x34]),
                     Piece(name: "layers.1.attention.to_q.weight_scale", dtype: "F8_E4M3", shape: [2, 1], bytes: [0x38, 0x38]),
                     Piece(name: "layers.1.attention.to_q.weight_scale_2", dtype: "F32", shape: [], bytes: [0, 0, 0x80, 0x3F])]
        XCTAssertTrue(try refusal(nvfp4).contains("4-bit"))
        let mixed = try refusal(fp8Layer + nvfp4)
        XCTAssertTrue(mixed.contains("4-bit") && mixed.contains("in its entirety"), mixed)
        // Declared as nvfp4 by `comfy_quant`, even with nothing else showing it.
        let json = Array(#"{"format": "nvfp4"}"#.utf8)
        XCTAssertTrue(try refusal(fp8Layer + [Piece(name: "layers.0.attention.to_q.comfy_quant", dtype: "U8", shape: [json.count], bytes: json)])
            .contains("4-bit"))
    }

    // ── ComfyUI's int8 convrot ─────────────────────────────────────────────────

    /// The regular Hadamard of size `g`, normalized, **built by Kronecker products** as comfy_kitchen's
    /// `_build_hadamard` does — independently of the engine's butterflies.
    private func hadamard(_ g: Int) -> [Double] {
        let h4: [Double] = [1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1]
        var h = h4, size = 4
        while size < g {
            var k = [Double](repeating: 0, count: size * 4 * size * 4)
            for i in 0..<size { for j in 0..<size { for a in 0..<4 { for b in 0..<4 {
                k[(i * 4 + a) * size * 4 + j * 4 + b] = h[i * size + j] * h4[a * 4 + b]
            } } } }
            h = k; size *= 4
        }
        return h.map { $0 / Double(g).squareRoot() }
    }

    /// `W = (q · s) · R` in fp64, rounded once to fp32, map layout `[K, N]`. **Exact here**, so
    /// the order of the sums is free: `R`'s entries are ±1/√G with G a power of 4 (a power of two),
    /// `q · s` is an int8 times an fp32 (32 significant bits), and a sum of G ≤ 256 of them stays
    /// under 2⁵³ units of `s`'s last place. The product is thus one fp64 matrix product per group
    /// (`vDSP_mmulD`, nothing of the engine's): the same bits as the triple loop it replaces, which
    /// cost ~1 s of checked subscripts in a debug build.
    private func rotatedReference(_ q: [UInt8], rows n: Int, columns k: Int, scale: [UInt8], group g: Int) -> [Float] {
        let h = hadamard(g)
        let s = (0..<n).map { Double(scaleValue(scale, $0, type: "F32")) }
        var out = [Float](repeating: 0, count: n * k)
        var block = [Double](repeating: 0, count: g * n), product = [Double](repeating: 0, count: g * n)
        for first in stride(from: 0, to: k, by: g) {
            // block[j, r] = q[r, first + j] · s[r]; product[c, r] = Σⱼ R[c, j] · block[j, r].
            for j in 0..<g { for r in 0..<n { block[j * n + r] = Double(Int8(bitPattern: q[r * k + first + j])) * s[r] } }
            vDSP_mmulD(h, 1, block, 1, &product, 1, vDSP_Length(g), vDSP_Length(n), vDSP_Length(g))
            for i in 0..<(g * n) { out[first * n + i] = Float(product[i]) }
        }
        return out
    }

    private func comfyQuant(_ base: String, _ json: String) -> Piece {
        let b = Array(json.utf8)
        return Piece(name: base + ".comfy_quant", dtype: "U8", shape: [b.count], bytes: b)
    }

    /// A rotated layer as ComfyUI publishes it, with the extremes that bound `m` (rows whose int8
    /// values follow the signs of `H`'s first row, at 127 / −128).
    private func convrotLayer(_ base: String, n: Int, k: Int, group g: Int, json: String? = nil) -> (pieces: [Piece], q: [UInt8], s: [UInt8]) {
        var q = values(n * k, fp8: false)
        let h = hadamard(g)
        for c in 0..<k where n >= 2 {
            let plus = h[c % g] > 0
            q[c] = UInt8(bitPattern: plus ? 127 : -128)
            q[k + c] = UInt8(bitPattern: plus ? -128 : 127)
        }
        let s = scales(n, type: "F32")
        return ([Piece(name: base + ".weight", dtype: "I8", shape: [n, k], bytes: q),
                 Piece(name: base + ".weight_scale", dtype: "F32", shape: [n, 1], bytes: s),
                 comfyQuant(base, json ?? #"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": \#(g)}"#)], q, s)
    }

    /// The map keeps `q` and `s` as published and says `"rotation"`; the engine's `W` is the
    /// correctly rounded `(q · s) · R` — bit for bit, on the SIMD tiles (N = 72: 64 + 8) and the
    /// scalar ones (N = 12) —, and the forge's own dequantization (the fp32 path) agrees. The GPU
    /// declines a rotated weight; a table read by rows refuses it.
    func testConvrotIsExactlyRotated() throws {
        let a = convrotLayer("transformer_blocks.0.attn.to_q", n: 72, k: 512, group: 256)
        // The size under `params`, as ComfyUI's loader also reads it.
        let b = convrotLayer("transformer_blocks.0.attn.to_k", n: 12, k: 256, group: 256,
                             json: #"{"format": "int8_tensorwise", "params": {"convrot": true}}"#)
        let path = try write(a.pieces + b.pieces, as: "convrot.safetensors")
        let source = try TensorSource(paths: [path])
        XCTAssertEqual(source.quantizedFormats["ComfyUI int8_tensorwise convrot"], 2)
        let (map, outcomes) = try forge(source, family: .qwenImage21)
        for (name, layer, n, k) in [("transformer_blocks.0.attn.to_q.weight", a, 72, 512), ("transformer_blocks.0.attn.to_k.weight", b, 12, 256)] {
            XCTAssertEqual(outcomes[name], .kept8bit)
            let t = try XCTUnwrap(map.tensors[name])
            XCTAssertEqual(t.rotation, Artifact.Rotation(kind: "convrot", group: 256))
            XCTAssertEqual(mapBytes(map, name).values, transposed(layer.q, rows: n, columns: k))
            XCTAssertEqual(mapBytes(map, name).scale, layer.s)
            let expected = rotatedReference(layer.q, rows: n, columns: k, scale: layer.s, group: 256)
            var cpu = [Float](repeating: -1, count: t.count)
            try cpu.withUnsafeMutableBufferPointer { _ = try map.materialize(name, into: $0) }
            XCTAssertEqual(cpu.map(\.bitPattern), expected.map(\.bitPattern), name)
            // The forge's `dequantized()` (published layout): the same values.
            let forged = try source.readQuantized(name).dequantized()
            XCTAssertEqual((0..<(n * k)).map { forged[$0].bitPattern }, (0..<(n * k)).map { expected[($0 % k) * n + $0 / k].bitPattern })
            if let (widener, _) = gpu { XCTAssertNil(widener.source(name, in: map), "no GPU kernel for a rotated weight") }
            var row = [Float](repeating: 0, count: n)
            XCTAssertThrowsError(try row.withUnsafeMutableBufferPointer { try map.materializeRows(name, first: 0, count: 1, into: $0.baseAddress!) })
        }
    }

    /// A share of the **input** keeps the rotation on a group boundary only; off it, the weight is
    /// stored dequantized in fp32 (rotated), and the outcome says why. FLUX.2's single-block
    /// `to_out` (input = attention d, then MLP), group 16.
    func testConvrotColumnSplit() throws {
        for (d, m, kept) in [(16, 48, true), (8, 32, false)] {
            let layer = convrotLayer("single_transformer_blocks.0.attn.to_out", n: d, k: m, group: 16)
            let path = try write(layer.pieces, as: "klein-\(d).safetensors")
            let (map, outcomes) = try forge(try TensorSource(paths: [path]), family: .klein4b)
            let name = "single_transformer_blocks.0.attn.to_out.mlp.weight"
            let full = rotatedReference(layer.q, rows: d, columns: m, scale: layer.s, group: 16)    // [m, d]
            let mlp = Array(full[(d * d)...])
            var cpu = [Float](repeating: -1, count: mlp.count)
            try cpu.withUnsafeMutableBufferPointer { _ = try map.materialize(name, into: $0) }
            XCTAssertEqual(cpu.map(\.bitPattern), mlp.map(\.bitPattern), "d = \(d)")
            if kept {
                XCTAssertEqual(outcomes[name], .kept8bit)
                XCTAssertEqual(map.tensors[name]?.rotation?.group, 16)
            } else {
                XCTAssertEqual(outcomes[name], .dequantized(" (columns [8, 32) cut its convrot groups of 16)"))
                XCTAssertEqual(map.tensors[name]?.dtype, .float32)
            }
        }
    }

    func testConvrotRefusals() throws {
        let one = [UInt8](repeating: 1, count: 4 * 256)
        let scale = Piece(name: "layers.0.attention.to_q.weight_scale", dtype: "F32", shape: [4, 1], bytes: [UInt8](repeating: 0, count: 0)
                          + (0..<4).flatMap { _ in [0, 0, 0x80, 0x3F] as [UInt8] })
        func detail(_ json: String, k: Int = 256, dtype: String = "I8", perTensor: Bool = false) throws -> String {
            try refusal([Piece(name: "layers.0.attention.to_q.weight", dtype: dtype, shape: [4, k], bytes: Array(one.prefix(4 * k))),
                         perTensor ? Piece(name: "layers.0.attention.to_q.weight_scale", dtype: "F32", shape: [], bytes: [0, 0, 0x80, 0x3F]) : scale,
                         comfyQuant("layers.0.attention.to_q", json)])
        }
        XCTAssertTrue(try detail(#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 128}"#).contains("convrot group"))
        XCTAssertTrue(try detail(#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 1024}"#).contains("convrot group"))
        XCTAssertTrue(try detail(#"{"format": "int8_tensorwise", "convrot": true}"#, k: 64).contains("divides by the group"))
        XCTAssertTrue(try detail(#"{"format": "int8_tensorwise", "convrot": true}"#, perTensor: true).contains("per-row int8"))
        XCTAssertTrue(try detail(#"{"format": "float8_e4m3fn", "convrot": true}"#, dtype: "F8_E4M3").contains("convrot"))
        // A scale whose rotated products could be subnormal (smallest |m| / √G = 1/16).
        let tiny = (0..<4).flatMap { _ in withUnsafeBytes(of: Float(0x1p-123).bitPattern.littleEndian) { Array($0) } }
        let subnormal = try refusal([Piece(name: "layers.0.attention.to_q.weight", dtype: "I8", shape: [4, 256], bytes: one),
                                     Piece(name: "layers.0.attention.to_q.weight_scale", dtype: "F32", shape: [4, 1], bytes: tiny),
                                     comfyQuant("layers.0.attention.to_q", #"{"format": "int8_tensorwise", "convrot": true}"#)])
        XCTAssertTrue(subnormal.contains("subnormal"), subnormal)
    }

    /// A map whose rotation is of an unknown kind is refused by the reader.
    func testUnknownRotationRefusesTheMap() throws {
        let layer = convrotLayer("transformer_blocks.0.attn.to_q", n: 8, k: 256, group: 256)
        let (map, _) = try forge(try TensorSource(paths: [try write(layer.pieces, as: "r.safetensors")]), family: .qwenImage21)
        var bytes = try Data(contentsOf: URL(fileURLWithPath: map.path))
        let from = Data(#""convrot""#.utf8), to = Data(#""sylvest""#.utf8)
        let range = try XCTUnwrap(bytes.range(of: from))
        bytes.replaceSubrange(range, with: to)
        let other = root.appendingPathComponent("unknown.silicon").path
        try bytes.write(to: URL(fileURLWithPath: other))
        XCTAssertThrowsError(try Artifact(path: other)) { XCTAssertTrue("\($0)".contains("sylvest"), "\($0)") }
    }

    func testMXFP8AsymmetricInt8AndOddScalesAreRefused() throws {
        XCTAssertTrue(try refusal([Piece(name: "a.weight", dtype: "F8_E4M3", shape: [32], bytes: [UInt8](repeating: 0x38, count: 32)),
                                   Piece(name: "a.weight_scale", dtype: "F8_E8M0", shape: [1], bytes: [127])]).contains("mxfp8"))
        let meta = torchao(["a": [2, 2]], type: "Int8Tensor", parts: ["zero_point", "qdata", "scale"])
        XCTAssertTrue(try refusal([Piece(name: "a._weight_qdata", dtype: "I8", shape: [2, 2], bytes: [1, 2, 3, 4]),
                                   Piece(name: "a._weight_scale", dtype: "F32", shape: [2, 1], bytes: [0, 0, 0x80, 0x3F, 0, 0, 0x80, 0x3F]),
                                   Piece(name: "a._weight_zero_point", dtype: "I8", shape: [2, 1], bytes: [0, 3])], metadata: meta)
            .contains("asymmetric"))
        XCTAssertTrue(try refusal([Piece(name: "a.weight", dtype: "I8", shape: [2, 4], bytes: [UInt8](repeating: 1, count: 8)),
                                   Piece(name: "a.weight_scale", dtype: "F32", shape: [2, 2], bytes: [UInt8](repeating: 0, count: 16))])
            .contains("only one per tensor or one per row"))
        XCTAssertTrue(try refusal([Piece(name: "a.weight", dtype: "I8", shape: [2, 2], bytes: [1, 2, 3, 4])]).contains("without a scale"))
    }

    /// **fp8 by 128×128 blocks** (Qwen's and DeepSeek's official fp8): `weight` F8_E4M3 next
    /// to `weight_scale_inv` F32 `[⌈N/128⌉, ⌈K/128⌉]`. Without the refusal the weight was read as a
    /// plain cast and the scale kept as a model tensor; now the file is refused, saying why — and so is
    /// a file where only one layer is so.
    func testBlockwiseFP8IsRefused() throws {
        let block = [Piece(name: "model.layers.0.mlp.up_proj.weight", dtype: "F8_E4M3", shape: [256, 128],
                           bytes: values(256 * 128, fp8: true)),
                     Piece(name: "model.layers.0.mlp.up_proj.weight_scale_inv", dtype: "F32", shape: [2, 1],
                           bytes: scales(2, type: "F32"))]
        let why = try refusal(block)
        XCTAssertTrue(why.contains("weight_scale_inv") && why.contains("128×128") && why.contains("not supported"), why)
        XCTAssertTrue(try refusal(fp8Layer + block).contains("weight_scale_inv"))
        // The import says so too, before anything is forged.
        let path = try write(block, as: "blockwise-fp8.safetensors")
        XCTAssertThrowsError(try ModelImport.recognize(path)) { e in
            guard case EngineError.importRefused(_, let detail) = e else { return XCTFail("\(e)") }
            XCTAssertTrue(detail.contains("weight_scale_inv"), detail)
        }
    }

    /// **The H0 gate is lifted**: an fp8 Z-Image checkpoint is recognized and read 8-bit, not
    /// refused, and not widened.
    func testAnFP8CheckpointIsAccepted() throws {
        let path = try write(fp8Layer + [Piece(name: "cap_embedder.0.weight", dtype: "BF16", shape: [2], bytes: [0x80, 0x3F, 0x80, 0x3F])],
                             as: "model-fp8.safetensors")
        let (kind, family) = try ModelImport.recognize(path)
        XCTAssertEqual(kind, .model)
        XCTAssertEqual(family, .zImage)
        let s = try TensorSource(paths: [path])
        XCTAssertEqual(s.quantization("layers.0.attention.to_q.weight")?.kind, .float8_e4m3)
        XCTAssertEqual(s.quantizedFormats, ["ComfyUI scaled": 1])
        XCTAssertEqual(try s.read("layers.0.attention.to_q.weight"), [1, 2, -1, 0])
    }

    /// SDNQ's `.scale` beside an fp8 weight: `w = f8 · s`.
    func testFP8PerRowSDNQ() throws {
        try checkLayout(fp8: true, scaleType: "BF16", perRow: true, naming: { ($0 + ".weight", $0 + ".scale") })
    }

    // ── review additions ─────────────────────────────────────────────────────────────────

    /// `[K/b, N]` (GGUF Q8_0's layout, H3): rows read from the middle of a block take the right
    /// block's scales, and the GPU agrees.
    func testBlockScalesAndRowsFromMidBlock() throws {
        let (k, n, b) = (8, 4, 4)
        let q = values(k * n, fp8: false), s = scales((k / b) * n, type: "F16")
        let path = root.appendingPathComponent("blocks.silicon").path
        let t = MapWriter.Tensor(name: "w", shape: [k, n], dtype: .int8, scale: .init(dtype: .float16, shape: [k / b, n], block: b),
                                 transposed: true) { .init(values: q, scale: s, absMax: 0) }
        let pad = MapWriter.Tensor(name: "z", shape: [4], dtype: .bfloat16, transposed: nil) { [0, 0, 0, 0] }
        _ = try MapWriter.write(to: path, tensors: [t, pad], header: { _ in [.init("format", 1), .init("page", .integer(MapWriter.page))] })
        let map = try Artifact(path: path)
        XCTAssertEqual(map.tensors["w"]?.scale?.layout, .blocks(columns: n, rows: b))
        let expected = (0..<(k * n)).map { i in Float(Int8(bitPattern: q[i])) * scaleValue(s, (i / n / b) * n + i % n, type: "F16") }
        try assertDequantizes(map, "w", to: expected)
        var rows = [Float](repeating: 0, count: 3 * n)
        try rows.withUnsafeMutableBufferPointer { try map.materializeRows("w", first: 3, count: 3, into: $0.baseAddress!) }
        XCTAssertEqual(rows.map(\.bitPattern), expected[(3 * n)..<(6 * n)].map(\.bitPattern), "rows 3, 4, 5: blocks 0 and 1")
    }

    /// FLUX.2's `final_layer.adaLN_modulation.1` (halves swapped) and `to_qkv_mlp_proj` (unequal row
    /// shares) on 8-bit weights: rows and their scales move together.
    func testSwappedHalvesAndRowSharesStay8Bit() throws {
        let (d, h) = (4, 2), rows = 3 * d + 2 * h
        let fused = values(rows * d, fp8: true), fs = scales(rows, type: "F32")
        let last = values(2 * d * d, fp8: false), ls = scales(2 * d, type: "BF16")
        let path = try write([Piece(name: "single_blocks.0.linear1.weight", dtype: "F8_E4M3", shape: [rows, d], bytes: fused),
                              Piece(name: "single_blocks.0.linear1.weight_scale", dtype: "F32", shape: [rows, 1], bytes: fs),
                              Piece(name: "final_layer.adaLN_modulation.1.weight", dtype: "I8", shape: [2 * d, d], bytes: last),
                              Piece(name: "final_layer.adaLN_modulation.1.scale", dtype: "BF16", shape: [2 * d, 1], bytes: ls),
                              Piece(name: "single_blocks.0.norm.query_norm.scale", dtype: "BF16", shape: [2], bytes: [0x80, 0x3F, 0x80, 0x3F])],
                             as: "klein.safetensors")
        let (map, outcomes) = try forge(try TensorSource(paths: [path]), family: .klein4b)
        XCTAssertEqual(outcomes["norm_out.linear.weight"], .kept8bit)
        let swapped = Array(last[(d * d)...] + last[..<(d * d)]), sws = Array(ls[(2 * d)...] + ls[..<(2 * d)])
        XCTAssertEqual(mapBytes(map, "norm_out.linear.weight").scale, sws)
        try assertDequantizes(map, "norm_out.linear.weight",
                              to: reference(swapped, rows: 2 * d, columns: d, fp8: false, scale: sws, type: "BF16", perRow: true))
        let up = Array(fused[((3 * d + h) * d)...]), us = Array(fs[((3 * d + h) * 4)...])
        let name = "single_transformer_blocks.0.attn.to_qkv_mlp_proj.up.weight"
        XCTAssertEqual(outcomes[name], .kept8bit)
        try assertDequantizes(map, name, to: reference(up, rows: h, columns: d, fp8: true, scale: us, type: "F32", perRow: true))
    }

    /// 8-bit tensors that are not Linears (a cast norm, an int8 bias with one scale) stay 8-bit.
    func testNonLinear8BitStays8Bit() throws {
        let norm = values(8, fp8: true), bias = values(8, fp8: false)
        let path = try write([Piece(name: "layers.0.attention_norm1.weight", dtype: "F8_E4M3", shape: [8], bytes: norm),
                              Piece(name: "layers.0.adaLN_modulation.0.bias", dtype: "I8", shape: [8], bytes: bias),
                              Piece(name: "layers.0.adaLN_modulation.0.bias_scale_unused", dtype: "BF16", shape: [1], bytes: [0x80, 0x3F]),
                              Piece(name: "layers.0.adaLN_modulation.0.weight_scale", dtype: "F32", shape: [], bytes: [0, 0, 0, 0x3F]),
                              Piece(name: "cap_embedder.0.weight", dtype: "BF16", shape: [4], bytes: [UInt8](repeating: 0, count: 8))],
                             as: "nonlinear.safetensors")
        // The int8 bias has no scale of its own: refused (an int8 without a scale is not a weight).
        XCTAssertThrowsError(try TensorSource(paths: [path]))
        let path2 = try write([Piece(name: "layers.0.attention_norm1.weight", dtype: "F8_E4M3", shape: [8], bytes: norm),
                               Piece(name: "layers.0.adaLN_modulation.0.bias", dtype: "I8", shape: [8], bytes: bias),
                               Piece(name: "layers.0.adaLN_modulation.0.bias.weight_scale", dtype: "F32", shape: [], bytes: [0, 0, 0, 0x3F]),
                               Piece(name: "cap_embedder.0.weight", dtype: "BF16", shape: [4], bytes: [UInt8](repeating: 0, count: 8))],
                              as: "nonlinear2.safetensors")
        let (map, outcomes) = try forge(try TensorSource(paths: [path2]), family: .zImage)
        XCTAssertEqual(outcomes["layers.0.attention_norm1.weight"], .kept8bit)
        XCTAssertEqual(map.tensors["layers.0.attention_norm1.weight"]?.dtype, .float8_e4m3)
        XCTAssertNil(map.tensors["layers.0.attention_norm1.weight"]?.scale)
        try assertDequantizes(map, "layers.0.attention_norm1.weight", to: norm.map { Numerics.e4m3[Int($0)] })
        XCTAssertEqual(outcomes["layers.0.adaLN_modulation.0.bias"], .kept8bit)
        XCTAssertEqual(mapBytes(map, "layers.0.adaLN_modulation.0.bias").values, bias)
        try assertDequantizes(map, "layers.0.adaLN_modulation.0.bias", to: bias.map { Float(Int8(bitPattern: $0)) * 0.5 })
    }

    /// **fp16 is kept fp16** (never wider, never narrower): bytes copied; a folded `1 +` stays fp16
    /// when `w + 1` is an fp16, and goes fp32 when it is not.
    func testFP16IsKeptFP16() throws {
        let h: [Float16] = [0.5, -1.25, 3, 0.0009765625]
        let bytes = h.flatMap { withUnsafeBytes(of: $0.bitPattern.littleEndian) { Array($0) } }
        let path = try write([Piece(name: "layers.0.attention_norm1.weight", dtype: "F16", shape: [4], bytes: bytes),
                              Piece(name: "layers.0.feed_forward.w2.weight", dtype: "F16", shape: [2, 2], bytes: bytes),
                              Piece(name: "cap_embedder.0.weight", dtype: "F16", shape: [4], bytes: bytes)], as: "f16.safetensors")
        let (map, outcomes) = try forge(try TensorSource(paths: [path]), family: .zImage)
        XCTAssertEqual(outcomes["layers.0.attention_norm1.weight"], .kept16bit)
        XCTAssertEqual(map.tensors["layers.0.attention_norm1.weight"]?.dtype, .float16)
        XCTAssertEqual([UInt8](UnsafeRawBufferPointer(start: map.pointer("layers.0.attention_norm1.weight")!, count: 8)), bytes)
        var w = [Float](repeating: 0, count: 4)
        try w.withUnsafeMutableBufferPointer { _ = try map.materialize("layers.0.attention_norm1.weight", into: $0) }
        XCTAssertEqual(w, h.map(Float.init))
        XCTAssertEqual(map.tensors["cap_embedder.0.weight"]?.dtype, .float32, "Z-Image's precision-sensitive layers stay fp32")
        // A transposed fp16 Linear, one row read alone: `[K, N]` = `[[0.5, 3], [-1.25, 2⁻¹⁰]]`.
        XCTAssertEqual(map.tensors["layers.0.feed_forward.w2.weight"]?.dtype, .float16)
        var row = [Float](repeating: 0, count: 2)
        try row.withUnsafeMutableBufferPointer { try map.materializeRows("layers.0.feed_forward.w2.weight", first: 1, count: 1, into: $0.baseAddress!) }
        XCTAssertEqual(row, [-1.25, 0x1p-10])

        let exact: [Float16] = [0.5, 0x1p-10]                     // 1 + 2⁻¹⁰ is an fp16 (ulp of 1)
        let tiny: [Float16] = [0.5, Float16(bitPattern: 0x0400)]   // 1 + 2⁻¹⁴ is not
        for (values, expected) in [(exact, ForgeDiT.Outcome.kept16bit), (tiny, .widened)] {
            let b = values.flatMap { withUnsafeBytes(of: $0.bitPattern.littleEndian) { Array($0) } }
            let p = try write([Piece(name: "txt_in.text_norm.weight", dtype: "F16", shape: [2], bytes: b)], as: "qwen16-\(UUID()).safetensors")
            let n = try Recipes.normalize(try TensorSource(paths: [p]), family: .qwenImage21)
            let (t, o) = try ForgeDiT.tensor("txt_in.text_norm.weight", n.dit["txt_in.text_norm.weight"]!, family: .qwenImage21)
            XCTAssertEqual(o, expected)
            XCTAssertEqual(t.dtype, expected == .kept16bit ? .float16 : .float32)
            XCTAssertEqual(try t.produce(), values.map { Float($0) + 1 })
        }
    }

    /// A scale whose products can be subnormal is refused (the GPU would flush them to zero).
    func testSubnormalProductsAreRefused() throws {
        let tiny = withUnsafeBytes(of: Float(1e-39).bitPattern.littleEndian) { Array($0) }
        XCTAssertTrue(try refusal([Piece(name: "a.weight", dtype: "I8", shape: [2, 2], bytes: [1, 2, 3, 4]),
                                   Piece(name: "a.weight_scale", dtype: "F32", shape: [], bytes: tiny)]).contains("subnormal"))
        let small = withUnsafeBytes(of: Float(0x1p-120).bitPattern.littleEndian) { Array($0) }   // fine for int8, not for fp8
        XCTAssertTrue(try refusal([Piece(name: "a.weight", dtype: "F8_E4M3", shape: [2, 2], bytes: [1, 2, 3, 4]),
                                   Piece(name: "a.weight_scale", dtype: "F32", shape: [], bytes: small)]).contains("subnormal"))
        let path = try write([Piece(name: "a.weight", dtype: "I8", shape: [2, 2], bytes: [1, 2, 3, 4]),
                              Piece(name: "a.weight_scale", dtype: "F32", shape: [], bytes: small)], as: "int8-small.safetensors")
        XCTAssertNoThrow(try TensorSource(paths: [path]))
    }

    /// ComfyUI says float8_e4m3fn but no scale is published: refused, not read as a cast. A U8 is
    /// not called 4-bit when it may be an asymmetric uint8.
    func testDeclaredFP8WithoutScaleAndU8() throws {
        let json = Array(#"{"format": "float8_e4m3fn"}"#.utf8)
        XCTAssertTrue(try refusal([Piece(name: "a.weight", dtype: "F8_E4M3", shape: [2, 2], bytes: [1, 2, 3, 4]),
                                   Piece(name: "a.comfy_quant", dtype: "U8", shape: [json.count], bytes: json)])
            .contains("without its `weight_scale`"))
        XCTAssertTrue(try refusal([Piece(name: "a.weight", dtype: "U8", shape: [2, 2], bytes: [1, 2, 3, 4])])
            .contains("4-bit packed, or asymmetric uint8"))
    }

    /// An Anima adapter that came 8-bit is written dequantized in fp32, and counted for the journal.
    func testAnimaAdapterDequantizedIsCounted() throws {
        let q = values(8, fp8: false)
        let path = try write([Piece(name: "llm_adapter.embed.weight", dtype: "I8", shape: [2, 4], bytes: q),
                              Piece(name: "llm_adapter.embed.scale", dtype: "BF16", shape: [2, 1], bytes: [0x80, 0x3F, 0, 0x40]),
                              Piece(name: "llm_adapter.norm.weight", dtype: "BF16", shape: [2], bytes: [0x80, 0x3F, 0x80, 0x3F])],
                             as: "adapter-src.safetensors")
        let source = try TensorSource(paths: [path])
        let out = root.appendingPathComponent("adapter.safetensors").path
        XCTAssertEqual(try ForgeDiT.writeAdapter(source: source, keys: source.names, to: out), 1)
        let s = try Safetensors(path: out)
        XCTAssertEqual(s.entries["model.diffusion_model.llm_adapter.embed.weight"]?.dtype, "F32")
        XCTAssertEqual(s.materialize("model.diffusion_model.llm_adapter.embed.weight"),
                       (0..<8).map { Float(Int8(bitPattern: q[$0])) * ($0 < 4 ? 1 : 2) })
        XCTAssertEqual(s.entries["model.diffusion_model.llm_adapter.norm.weight"]?.dtype, "BF16")
    }

    /// **`ForgeDiT.forge` itself**, on a synthetic 8-bit Z-Image: the map, the disk reservation
    /// and the journal.
    func testForgeDiTOnASyntheticFile() throws {
        let (n, k) = (4, 8)
        let qkv = values(3 * n * k, fp8: false), s = scales(3 * n, type: "BF16")
        let path = try write([Piece(name: "layers.0.attention.qkv.weight", dtype: "I8", shape: [3 * n, k], bytes: qkv),
                              Piece(name: "layers.0.attention.qkv.scale", dtype: "BF16", shape: [3 * n, 1], bytes: s),
                              Piece(name: "cap_embedder.0.weight", dtype: "BF16", shape: [4], bytes: [UInt8](repeating: 0, count: 8)),
                              Piece(name: "scaled_fp8", dtype: "F8_E4M3", shape: [0], bytes: [])], as: "zimage-sdnq.safetensors")
        let out = root.appendingPathComponent("forged.silicon").path
        var reserved = 0
        let r = try ForgeDiT.forge(source: try TensorSource(paths: [path]), family: .zImage, config: .object([]), to: out,
                                   descriptionSource: .object([]), reserve: { reserved = $0 }, checkingCount: false)
        XCTAssertEqual(r.kept8bit, 3)
        XCTAssertEqual(r.tally.inexact, 0)
        XCTAssertEqual(reserved, r.tally.bytes)
        XCTAssertTrue(r.rows.contains { $0.contains("8-bit kept as published") && $0.contains("3 tensors") }, "\(r.rows)")
        let map = try Artifact(path: out)
        XCTAssertEqual(map.header["map_dtype"] as? String, "mixed")
        XCTAssertEqual(Set(map.order), ["layers.0.attention.to_q.weight", "layers.0.attention.to_k.weight",
                                        "layers.0.attention.to_v.weight", "cap_embedder.0.weight"])
        let v = Array(qkv[(2 * n * k)...]), vs = Array(s[(2 * n * 2)...])
        try assertDequantizes(map, "layers.0.attention.to_v.weight",
                              to: reference(v, rows: n, columns: k, fp8: false, scale: vs, type: "BF16", perRow: true))
    }

    // ── the text encoder's forge ───────────────────────────────────────────

    /// The per-head RoPE permutation, written here as a scatter (published row `h·D + i` → map row
    /// `h·D + 2i`, or `h·D + 2(i − D/2) + 1`): → the published row of each map row.
    private func ropeOrder(rows: Int, head: Int) -> [Int] {
        var order = [Int](repeating: -1, count: rows)
        for s in 0..<rows {
            let h = s / head, i = s % head
            order[h * head + (i < head / 2 ? 2 * i : 2 * (i - head / 2) + 1)] = s
        }
        return order
    }

    private func forgeEncoder(_ pieces: [Piece], config: String, family: Family) throws -> Artifact {
        let path = try write(pieces, as: "encoder-\(UUID()).safetensors")
        let out = root.appendingPathComponent("encoder-\(UUID()).silicon").path
        let tally = try ForgeText.forge(source: try TensorSource(paths: [path]), published: try OrderedJSON.parse(Data(config.utf8)),
                                         family: family, to: out, directory: "text_encoder")
        XCTAssertEqual(tally.inexact, 0, "the forge never rounds")
        return try Artifact(path: out)
    }

    /// **A bf16 encoder gives the bytes it always gave**: every tensor of a synthetic Qwen3-VL (the
    /// Qwen-Image-2.1 encoder: language model, vision tower, a `Conv3d` patch embedding) is its
    /// published bf16 bytes, rows permuted per head (q, k, their norms; the q and k thirds of the vision
    /// qkv), `Linear`s transposed — computed here from the published bytes, nothing widened; the final
    /// norm is dropped, `map_dtype` stays "bfloat16".
    func testAnEncoderForgeKeepsABF16SourceByteForByte() throws {
        let lm = "model.language_model.", v = "model.visual."
        let shapes: [(String, [Int])] = [
            (lm + "embed_tokens.weight", [6, 8]), (lm + "layers.0.self_attn.q_proj.weight", [8, 8]),
            (lm + "layers.0.self_attn.k_proj.weight", [4, 8]), (lm + "layers.0.self_attn.v_proj.weight", [4, 8]),
            (lm + "layers.0.self_attn.q_norm.weight", [4]), (lm + "layers.0.self_attn.k_norm.weight", [4]),
            (lm + "layers.0.input_layernorm.weight", [8]), (lm + "norm.weight", [8]),
            (v + "blocks.0.attn.qkv.weight", [24, 8]), (v + "blocks.0.attn.qkv.bias", [24]),
            (v + "patch_embed.proj.weight", [8, 3, 2, 2, 2]), (v + "pos_embed.weight", [4, 8])]
        let pieces = shapes.map { Piece(name: $0.0, dtype: "BF16", shape: $0.1, bytes: scales($0.1.reduce(1, *), type: "BF16")) }
        let config = #"{"text_config": {"head_dim": 4, "num_hidden_layers": 1}, "vision_config": {"hidden_size": 8, "num_heads": 2}}"#
        let map = try forgeEncoder(pieces, config: config, family: .qwenImage21)
        XCTAssertEqual(map.header["map_dtype"] as? String, "bfloat16")
        XCTAssertEqual(map.header["dropped"] as? [String], [lm + "norm.weight"])
        XCTAssertEqual(map.order.count, shapes.count - 1)
        for piece in pieces where piece.name != lm + "norm.weight" {
            let name = piece.name.hasPrefix(v) ? String(piece.name.dropFirst("model.".count)) : String(piece.name.dropFirst(lm.count))
            let t = try XCTUnwrap(map.tensors[name], name)
            XCTAssertEqual(t.dtype, .bfloat16, name)
            let rows = piece.shape[0], columns = piece.shape.reduce(1, *) / rows
            var order = Array(0..<rows)
            if name.hasSuffix("q_proj.weight") || name.hasSuffix("k_proj.weight") || name.hasSuffix("_norm.weight") {
                order = ropeOrder(rows: rows, head: 4)
            } else if name.contains("attn.qkv") {
                let q = ropeOrder(rows: 8, head: 4)
                order = q + q.map { $0 + 8 } + Array(16..<24)
            }
            let linear = piece.shape.count >= 2 && !name.hasSuffix("embed_tokens.weight") && !name.hasSuffix("pos_embed.weight")
            var expected = [UInt8](repeating: 0, count: piece.bytes.count)
            for l in 0..<rows {
                for c in 0..<columns {
                    let to = linear ? c * rows + l : l * columns + c, from = order[l] * columns + c
                    expected[2 * to] = piece.bytes[2 * from]
                    expected[2 * to + 1] = piece.bytes[2 * from + 1]
                }
            }
            XCTAssertEqual(t.shape, linear ? [columns, rows] : piece.shape, name)
            XCTAssertEqual([UInt8](UnsafeRawBufferPointer(start: map.pointer(name)!, count: t.bytes)), expected, name)
        }
    }

    /// **An 8-bit encoder stays 8-bit**: Disty0's SDNQ (int8, bf16 scale per row — Z-Image's
    /// Qwen3-4B) and unsloth's ComfyUI int8 convrot (Qwen-Image-2.1's Qwen3-VL). q and k keep their
    /// bytes and scales, rows moved per head — the scales with their rows, the rotation (along the
    /// input) untouched — and dequantize to the publisher's `Float(q)·s` (or the correctly rounded
    /// `(q·s)·R`) of the permuted rows, bit for bit, CPU and GPU; the bf16 tensors stay bf16, and the
    /// header says "mixed".
    func testAnEightBitEncoderStaysEightBit() throws {
        // SDNQ, Z-Image.
        let (n, k) = (8, 8)
        let q = values(n * k, fp8: false), s = scales(n, type: "BF16"), kq = values(4 * k, fp8: false), ks = scales(4, type: "BF16")
        let sdnq = try forgeEncoder([
            Piece(name: "model.embed_tokens.weight", dtype: "BF16", shape: [6, k], bytes: scales(6 * k, type: "BF16")),
            Piece(name: "model.layers.0.self_attn.q_proj.weight", dtype: "I8", shape: [n, k], bytes: q),
            Piece(name: "model.layers.0.self_attn.q_proj.scale", dtype: "BF16", shape: [n, 1], bytes: s),
            Piece(name: "model.layers.0.self_attn.k_proj.weight", dtype: "I8", shape: [4, k], bytes: kq),
            Piece(name: "model.layers.0.self_attn.k_proj.scale", dtype: "BF16", shape: [4, 1], bytes: ks),
            Piece(name: "model.layers.0.self_attn.q_norm.weight", dtype: "BF16", shape: [4], bytes: scales(4, type: "BF16"))],
            config: #"{"head_dim": 4, "num_hidden_layers": 36}"#, family: .zImage)
        XCTAssertEqual(sdnq.header["map_dtype"] as? String, "mixed")
        XCTAssertEqual(sdnq.tensors["embed_tokens.weight"]?.dtype, .bfloat16)
        for (name, values, scale, rows) in [("layers.0.self_attn.q_proj.weight", q, s, n), ("layers.0.self_attn.k_proj.weight", kq, ks, 4)] {
            let t = try XCTUnwrap(sdnq.tensors[name])
            XCTAssertEqual(t.dtype, .int8)
            XCTAssertEqual(t.scale?.dtype, .bfloat16)
            let order = ropeOrder(rows: rows, head: 4)
            let movedValues = order.flatMap { values[($0 * k)..<(($0 + 1) * k)] }, movedScale = order.flatMap { scale[(2 * $0)..<(2 * $0 + 2)] }
            XCTAssertEqual(mapBytes(sdnq, name).values, transposed(movedValues, rows: rows, columns: k))
            XCTAssertEqual(mapBytes(sdnq, name).scale, movedScale)
            try assertDequantizes(sdnq, name, to: reference(movedValues, rows: rows, columns: k, fp8: false,
                                                            scale: movedScale, type: "BF16", perRow: true))
        }

        // ComfyUI int8 convrot, Qwen-Image-2.1 (groups of 16 along the input).
        let layer = convrotLayer("model.language_model.layers.0.self_attn.q_proj", n: 8, k: 32, group: 16)
        let convrot = try forgeEncoder(layer.pieces + [
            Piece(name: "model.language_model.embed_tokens.weight", dtype: "BF16", shape: [6, 32], bytes: scales(6 * 32, type: "BF16"))],
            config: #"{"text_config": {"head_dim": 4, "num_hidden_layers": 1}, "vision_config": {"hidden_size": 8, "num_heads": 2}}"#,
            family: .qwenImage21)
        let name = "layers.0.self_attn.q_proj.weight"
        let t = try XCTUnwrap(convrot.tensors[name])
        XCTAssertEqual(t.rotation, Artifact.Rotation(kind: "convrot", group: 16))
        let order = ropeOrder(rows: 8, head: 4)
        let published = rotatedReference(layer.q, rows: 8, columns: 32, scale: layer.s, group: 16)
        let expected: [Float] = (0..<(8 * 32)).map { (i: Int) -> Float in published[(i / 8) * 8 + order[i % 8]] }   // [K, N]: column l ← published row order[l]
        var cpu = [Float](repeating: -1, count: t.count)
        try cpu.withUnsafeMutableBufferPointer { _ = try convrot.materialize(name, into: $0) }
        XCTAssertEqual(cpu.map(\.bitPattern), expected.map(\.bitPattern))
        XCTAssertEqual(mapBytes(convrot, name).scale, order.flatMap { layer.s[(4 * $0)..<(4 * $0 + 4)] })
    }
}
