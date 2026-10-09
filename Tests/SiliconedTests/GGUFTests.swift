import Metal
import XCTest
@testable import Siliconed

/// **A GGUF Q8_0 checkpoint becomes an 8-bit map** — on small GGUF
/// files written here byte by byte: dimensions stored in reverse, Q8_0's interleaved blocks taken
/// apart into values `[K, N]` and fp16 scales `[K/32, N]`, a split along the output that moves
/// rows with their blocks, a split along the input that keeps the blocks only on their boundary,
/// F32/F16/BF16 kept by the forge's rules, any other ggml type refused with the file. The engine's
/// reading, CPU and GPU, is `Float(q) · Float(d)` **bit for bit**.
final class GGUFTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-gguf-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    // ── writing a GGUF ──────────────────────────────────────────────────────────────────────

    /// A tensor in torch shape `[N, K]` (written as `ne = [K, N]`), its ggml type and its bytes.
    struct Tensor { let name: String; let shape: [Int]; let type: UInt32; let bytes: [UInt8] }

    private func string(_ s: String) -> [UInt8] {
        withUnsafeBytes(of: UInt64(s.utf8.count).littleEndian) { Array($0) } + Array(s.utf8)
    }
    private func le<T: FixedWidthInteger>(_ v: T) -> [UInt8] { withUnsafeBytes(of: v.littleEndian) { Array($0) } }

    private func write(_ tensors: [Tensor], alignment: Int? = nil, version: UInt32 = 3, name: String = "model.gguf") throws -> String {
        var kv: [UInt8] = [], keys: UInt64 = 1
        kv += string("general.architecture") + le(UInt32(8)) + string("test")
        if let alignment { kv += string("general.alignment") + le(UInt32(4)) + le(UInt32(alignment)); keys += 1 }
        // An array, to be skipped by its type.
        kv += string("test.array") + le(UInt32(9)) + le(UInt32(5)) + le(UInt64(3)) + le(Int32(1)) + le(Int32(2)) + le(Int32(3))
        keys += 1
        let a = alignment ?? 32
        var infos: [UInt8] = [], data: [UInt8] = []
        for t in tensors {
            infos += string(t.name) + le(UInt32(t.shape.count))
            for d in t.shape.reversed() { infos += le(UInt64(d)) }
            infos += le(t.type) + le(UInt64(data.count))
            data += t.bytes
            data += [UInt8](repeating: 0, count: (a - data.count % a) % a)
        }
        var file = Array("GGUF".utf8) + le(version) + le(UInt64(tensors.count)) + le(keys) + kv + infos
        file += [UInt8](repeating: 0, count: (a - file.count % a) % a)
        file += data
        let path = root.appendingPathComponent(name).path
        try Data(file).write(to: URL(fileURLWithPath: path))
        return path
    }

    private var generator = SystemRandomNumberGenerator()

    /// A Q8_0 tensor `[n, k]`: its interleaved bytes, and its values and fp16 scales apart. Some
    /// scales are special: zero, the smallest fp16 subnormal, a negative one.
    private func q8(_ n: Int, _ k: Int) -> (bytes: [UInt8], q: [Int8], d: [UInt16]) {
        let blocks = n * k / 32
        var bytes: [UInt8] = [], q: [Int8] = [], d: [UInt16] = []
        for b in 0..<blocks {
            let scale: UInt16
            switch b {
            case 1: scale = 0                                   // a zero block
            case 2: scale = 0x0001                              // fp16's smallest subnormal, 2⁻²⁴
            case 3: scale = Float16(-0.0123).bitPattern
            default: scale = Float16(Float.random(in: 1e-4...3e-2, using: &generator)).bitPattern
            }
            d.append(scale)
            bytes += le(scale)
            for _ in 0..<32 {
                let v = Int8.random(in: -128...127, using: &generator)
                q.append(v)
                bytes.append(UInt8(bitPattern: v))
            }
        }
        return (bytes, q, d)
    }

    /// `Float(q) · Float(d)`, written independently of the engine, published layout `[n, k]`.
    private func reference(_ t: (bytes: [UInt8], q: [Int8], d: [UInt16]), _ n: Int, _ k: Int) -> [Float] {
        (0..<(n * k)).map { i in Float(t.q[i]) * Float(Float16(bitPattern: t.d[i / 32])) }
    }

    private func transposed(_ v: [Float], rows n: Int, columns k: Int) -> [Float] {
        var t = [Float](repeating: 0, count: n * k)
        for r in 0..<n { for c in 0..<k { t[c * n + r] = v[r * k + c] } }
        return t
    }

    private func forge(_ source: TensorSource, family: Family) throws -> (Artifact, [String: ForgeDiT.Outcome]) {
        let n = try Recipes.normalize(source, family: family)
        var tensors: [MapWriter.Tensor] = [], outcomes: [String: ForgeDiT.Outcome] = [:]
        for name in n.dit.keys.sorted() {
            let (t, o) = try ForgeDiT.tensor(name, n.dit[name]!, family: family)
            tensors.append(t)
            outcomes[name] = o
        }
        tensors.append(MapWriter.Tensor(name: "~end", shape: [4], dtype: .bfloat16, transposed: nil) { [0, 0, 0, 0] })
        let path = root.appendingPathComponent("map-\(UUID()).silicon").path
        let tally = try MapWriter.write(to: path, tensors: tensors, header: { _ in
            [.init("format", 1), .init("page", .integer(MapWriter.page)), .init("linear_weights_transposed", true)]
        })
        XCTAssertEqual(tally.inexact, 0, "the forge never rounds")
        return (try Artifact(path: path), outcomes)
    }

    private lazy var gpu: (WidenGPU, MTLDevice)? = {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let w = try? WidenGPU(device: device, queue: queue) else { return nil }
        return (w, device)
    }()

    private func assertDequantizes(_ map: Artifact, _ name: String, to expected: [Float],
                                   file: StaticString = #filePath, line: UInt = #line) throws {
        let t = try XCTUnwrap(map.tensors[name], file: file, line: line)
        var cpu = [Float](repeating: -1, count: t.count)
        try cpu.withUnsafeMutableBufferPointer { _ = try map.materialize(name, into: $0) }
        XCTAssertEqual(cpu.map(\.bitPattern), expected.map(\.bitPattern), "CPU \(name)", file: file, line: line)
        guard let (widener, device) = gpu, let source = widener.source(name, in: map) else { return }
        let buffer = try XCTUnwrap(device.makeBuffer(length: t.count * 4, options: .storageModeShared))
        try widener.run(source: source, destination: buffer, count: t.count)
        let g = buffer.contents().assumingMemoryBound(to: Float.self)
        XCTAssertEqual((0..<t.count).map { g[$0].bitPattern }, expected.map(\.bitPattern), "GPU \(name)", file: file, line: line)
    }

    private func bytes(_ v: [Float]) -> [UInt8] { v.flatMap { le($0.bitPattern) } }
    private func bf16(_ v: [Float]) -> [UInt8] { v.flatMap { le(UInt16($0.bitPattern >> 16)) } }
    private func f16(_ v: [Float]) -> [UInt8] { v.flatMap { le(Float16($0).bitPattern) } }

    // ── the tests ───────────────────────────────────────────────────────────────────────────

    /// Z-Image's ComfyUI names in a GGUF: a fused qkv in Q8_0 (split in three, rows and their blocks
    /// together), a Q8_0 MLP, an F32 norm, a BF16 linear, an F16 bias — on a 64-byte alignment.
    func testZImageQ8_0BecomesAnEightBitMap() throws {
        let (n, k) = (8, 96)
        let qkv = q8(3 * n, k), w1 = q8(2 * n, k)
        let norm = (0..<k).map { _ in Float.random(in: 0.5...1.5, using: &generator) }   // not bf16-exact: fp32
        let lin = (0..<(n * k)).map { _ in Float(bitPattern: UInt32(UInt16.random(in: 0x3c00...0x3f00, using: &generator)) << 16) }
        let bias = (0..<n).map { _ in Float(Float16(Float.random(in: -1...1, using: &generator))) }
        let path = try write([
            Tensor(name: "layers.0.attention.qkv.weight", shape: [3 * n, k], type: 8, bytes: qkv.bytes),
            Tensor(name: "layers.0.feed_forward.w1.weight", shape: [2 * n, k], type: 8, bytes: w1.bytes),
            Tensor(name: "layers.0.attention.q_norm.weight", shape: [k], type: 0, bytes: bytes(norm)),
            Tensor(name: "cap_embedder.1.weight", shape: [n, k], type: 30, bytes: bf16(lin)),
            Tensor(name: "cap_embedder.1.bias", shape: [n], type: 1, bytes: f16(bias)),
        ], alignment: 64)
        let source = try TensorSource(paths: [path])
        XCTAssertEqual(source.shape("layers.0.attention.qkv.weight"), [3 * n, k], "ne reversed: torch's [N, K]")
        XCTAssertEqual(source.dtype("layers.0.attention.qkv.weight"), "Q8_0")
        XCTAssertEqual(source.quantization("layers.0.attention.qkv.weight")?.block, 32)
        XCTAssertEqual(source.quantizedFormats, ["GGUF Q8_0": 2])
        XCTAssertTrue(source.notes.contains { $0.contains("GGUF v3, alignment 64, architecture test") }, "\(source.notes)")
        XCTAssertEqual(try source.read("layers.0.attention.qkv.weight").map(\.bitPattern),
                       reference(qkv, 3 * n, k).map(\.bitPattern), "the fp32 reading of the source")

        let (map, outcomes) = try forge(source, family: .zImage)
        let full = reference(qkv, 3 * n, k)
        for (part, x) in ["to_q", "to_k", "to_v"].enumerated() {
            let name = "layers.0.attention.\(x).weight"
            XCTAssertEqual(outcomes[name], .kept8bit)
            let t = try XCTUnwrap(map.tensors[name])
            XCTAssertEqual(t.dtype, .int8)
            XCTAssertEqual(t.shape, [k, n])
            XCTAssertEqual(t.scale?.shape, [k / 32, n])
            XCTAssertEqual(t.scale?.block, 32)
            XCTAssertEqual(t.scale?.dtype, .float16)
            let rows = Array(full[(part * n * k)..<((part + 1) * n * k)])
            try assertDequantizes(map, name, to: transposed(rows, rows: n, columns: k))
        }
        try assertDequantizes(map, "layers.0.feed_forward.w1.weight", to: transposed(reference(w1, 2 * n, k), rows: 2 * n, columns: k))
        // The published bytes, only moved: the map's values are the published q, transposed.
        let t = map.tensors["layers.0.feed_forward.w1.weight"]!, p = map.pointer("layers.0.feed_forward.w1.weight")!
        let mapped = (0..<t.count).map { Int8(bitPattern: p.load(fromByteOffset: $0, as: UInt8.self)) }
        XCTAssertEqual(mapped, (0..<(2 * n * k)).map { i in w1.q[(i % (2 * n)) * k + i / (2 * n)] })

        XCTAssertEqual(map.tensors["layers.0.attention.norm_q.weight"]?.dtype, .float32, "fp32 that bf16 cannot carry")
        try assertDequantizes(map, "layers.0.attention.norm_q.weight", to: norm)
        XCTAssertEqual(map.tensors["cap_embedder.1.weight"]?.dtype, .float32, "Z-Image keeps cap_embedder in fp32")
        try assertDequantizes(map, "cap_embedder.1.weight", to: transposed(lin, rows: n, columns: k))
        XCTAssertEqual(outcomes["cap_embedder.1.bias"], .asNamed, "named fp32: the fp16 bias is fp32, exactly")
        try assertDequantizes(map, "cap_embedder.1.bias", to: bias)
    }

    /// The map's rows `[first, first + count)` of a GGUF weight — through its blocks of 32 inputs.
    func testRowsAcrossBlocks() throws {
        let (n, k) = (4, 128)
        let w = q8(n, k)
        let path = try write([Tensor(name: "transformer_blocks.0.attn.to_q.weight", shape: [n, k], type: 8, bytes: w.bytes),
                              Tensor(name: "txt_in.text_norm.weight", shape: [8], type: 0, bytes: bytes([Float](repeating: 0, count: 8)))])
        let (map, _) = try forge(try TensorSource(paths: [path]), family: .qwenImage21)
        let expected = transposed(reference(w, n, k), rows: n, columns: k)
        for (first, count) in [(20, 50), (31, 2), (0, 128), (96, 32)] {
            var rows = [Float](repeating: .nan, count: count * n)
            try rows.withUnsafeMutableBufferPointer {
                try map.materializeRows("transformer_blocks.0.attn.to_q.weight", first: first, count: count, into: $0.baseAddress!)
            }
            XCTAssertEqual(rows.map(\.bitPattern), expected[(first * n)..<((first + count) * n)].map(\.bitPattern),
                           "rows \(first)..<\(first + count)")
        }
    }

    /// Qwen-Image-2.1's `img_mlp.gate_up` (ComfyUI): `[gate_layer; proj]`, each half with its blocks.
    func testQwenGateUpSplitsByRows() throws {
        let (h, k) = (6, 64)
        let w = q8(2 * h, k)
        let path = try write([
            Tensor(name: "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight", shape: [2 * h, k], type: 8, bytes: w.bytes),
            Tensor(name: "model.diffusion_model.txt_in.text_norm.weight", shape: [8], type: 30, bytes: bf16([Float](repeating: 0.5, count: 8))),
        ])
        let source = try TensorSource(paths: [path])
        XCTAssertEqual(Recipes.family(fromNames: source.names), .qwenImage21)
        let normalized = try Recipes.normalize(source, family: .qwenImage21)
        XCTAssertEqual(normalized.naming, "diffusers, MLP gate_up fused (ComfyUI)")
        let (map, _) = try forge(source, family: .qwenImage21)
        let full = reference(w, 2 * h, k)
        try assertDequantizes(map, "transformer_blocks.0.img_mlp.gate_layer.weight",
                              to: transposed(Array(full[0..<(h * k)]), rows: h, columns: k))
        try assertDequantizes(map, "transformer_blocks.0.img_mlp.proj.weight",
                              to: transposed(Array(full[(h * k)...]), rows: h, columns: k))
        try assertDequantizes(map, "txt_in.text_norm.weight", to: [Float](repeating: 1.5, count: 8))
    }

    /// FLUX.2's single-block `to_out`, cut on its input: on a block boundary the two shares stay
    /// 8-bit with their blocks' scales; across a block, both go to fp32, and the journal says why.
    func testColumnSplitKeepsBlocksOnlyOnTheirBoundary() throws {
        for d in [64, 48] {
            let k = 128, m = k - d
            let w = q8(d, k)
            let path = try write([Tensor(name: "single_blocks.0.linear2.weight", shape: [d, k], type: 8, bytes: w.bytes)],
                                 name: "klein-\(d).gguf")
            let (map, outcomes) = try forge(try TensorSource(paths: [path]), family: .klein4b)
            let full = reference(w, d, k)
            for (part, begin, number) in [("attn", 0, d), ("mlp", d, m)] {
                let name = "single_transformer_blocks.0.attn.to_out.\(part).weight"
                let share = (0..<d).flatMap { r in full[(r * k + begin)..<(r * k + begin + number)] }
                try assertDequantizes(map, name, to: transposed(share, rows: d, columns: number))
                if d % 32 == 0 {
                    XCTAssertEqual(outcomes[name], .kept8bit)
                    XCTAssertEqual(map.tensors[name]?.scale?.shape, [number / 32, d])
                } else {
                    XCTAssertEqual(map.tensors[name]?.dtype, .float32)
                    guard case .dequantized(let why)? = outcomes[name] else { return XCTFail("\(name): \(String(describing: outcomes[name]))") }
                    XCTAssertTrue(why.contains("cut its blocks"), why)
                }
            }
        }
    }

    /// Any ggml type below the floor (Q3_K, Q2_K, the I-quants, Q8_1, MXFP4…) or unknown
    /// refuses the file — by name, at the import's door. (Q4_0…Q6_K are read: `KQuantTests`.)
    func testOtherTypesRefuseTheWholeFile() throws {
        let ok = q8(4, 32)
        for (type, label) in [(UInt32(11), "Q3_K"), (10, "Q2_K"), (9, "Q8_1"), (20, "IQ4_NL"), (39, "MXFP4"), (77, "type 77")] {
            let path = try write([Tensor(name: "layers.0.attention.qkv.weight", shape: [4, 32], type: 8, bytes: ok.bytes),
                                  Tensor(name: "layers.0.feed_forward.w1.weight", shape: [4, 32], type: type, bytes: [UInt8](repeating: 0, count: 256))],
                                 name: "mixed-\(label).gguf")
            XCTAssertThrowsError(try TensorSource(paths: [path])) { e in
                XCTAssertTrue("\(e)".contains(label) && "\(e)".contains("refused in its entirety"), "\(e)")
            }
            XCTAssertThrowsError(try ModelImport.recognize(path)) { e in
                guard case EngineError.importRefused(_, let detail) = e else { return XCTFail("\(e)") }
                XCTAssertTrue(detail.contains(label), detail)
            }
        }
        // A malformed file is refused too, never read past its end.
        let bad = try write([Tensor(name: "w", shape: [4, 33], type: 8, bytes: [UInt8](repeating: 0, count: 4 * 34))], name: "bad.gguf")
        XCTAssertThrowsError(try TensorSource(paths: [bad])) { XCTAssertTrue("\($0)".contains("multiple of 32"), "\($0)") }
        let old = try write([Tensor(name: "w", shape: [4, 32], type: 8, bytes: ok.bytes)], version: 1, name: "v1.gguf")
        XCTAssertThrowsError(try TensorSource(paths: [old])) { XCTAssertTrue("\($0)".contains("version 1"), "\($0)") }
        var truncated = try Data(contentsOf: URL(fileURLWithPath: old.replacingOccurrences(of: "v1", with: "bad")))
        truncated.removeLast(40)
        let cut = root.appendingPathComponent("cut.gguf").path
        try truncated.write(to: URL(fileURLWithPath: cut))
        XCTAssertThrowsError(try TensorSource(paths: [cut]))
    }

    /// **leejet's Z-Image GGUFs** (stable-diffusion.cpp) publish `cap_pad_token` and `x_pad_token` as
    /// `[3840]`: GGUF drops the trailing `ne` of 1 that unsloth keeps (`[1, 3840]`, the vendor's shape).
    /// The recipe adds the leading 1 back for these two names only, values unchanged; any other 1-D
    /// tensor, and a 2-D pad token, keeps the shape it was published with.
    func testZImagePadTokensWithoutTheirLeadingOne() throws {
        let cap: [Float] = (0..<8).map { Float($0) * 0.25 - 1 }, x: [Float] = (0..<8).map { Float($0) * -0.5 }
        let path = try write([Tensor(name: "cap_pad_token", shape: [8], type: 0, bytes: bytes(cap)),
                              Tensor(name: "x_pad_token", shape: [8], type: 0, bytes: bytes(x)),
                              Tensor(name: "layers.0.attention.q_norm.weight", shape: [8], type: 0, bytes: bytes(cap))])
        let n = try Recipes.normalize(try TensorSource(paths: [path]), family: .zImage)
        XCTAssertEqual(n.dit["cap_pad_token"]?.shape, [1, 8])
        XCTAssertEqual(n.dit["x_pad_token"]?.shape, [1, 8])
        XCTAssertEqual(try n.dit["cap_pad_token"]?.read().map(\.bitPattern), cap.map(\.bitPattern))
        XCTAssertEqual(try n.dit["x_pad_token"]?.read().map(\.bitPattern), x.map(\.bitPattern))
        XCTAssertEqual(n.dit["layers.0.attention.q_norm.weight"]?.shape, [8], "only the pad tokens are reshaped")
        // unsloth's `[1, 8]` unchanged; a pad token of another 2-D shape is not reshaped either.
        let kept = try write([Tensor(name: "cap_pad_token", shape: [1, 8], type: 0, bytes: bytes(cap)),
                              Tensor(name: "x_pad_token", shape: [2, 4], type: 0, bytes: bytes(x))], name: "kept.gguf")
        let m = try Recipes.normalize(try TensorSource(paths: [kept]), family: .zImage)
        XCTAssertEqual(m.dit["cap_pad_token"]?.shape, [1, 8])
        XCTAssertEqual(m.dit["x_pad_token"]?.shape, [2, 4])
    }
}
