import Metal
import XCTest
@testable import Siliconed

/// **A GGUF 4- to 6-bit checkpoint becomes a map of packed blocks** — on
/// small GGUF files written here byte by byte: Q4_0, Q4_1, Q5_0, Q5_1 blocks and Q4_K, Q5_K, Q6_K
/// super-blocks with extreme bits (all 0, all 1), zero, −0, negative, subnormal and largest fp16
/// scales, nonzero and negative mins, and seeded random content. The engine's reading — CPU and GPU —
/// is **ggml's formula at the bit**, against a scalar dequantizer written below straight from
/// `ggml-quants.c` (`dequantize_row_q4_0` … `_q6_K`), independently of the engine. Rows move whole
/// (a fused qkv), columns only by whole blocks (a cut inside one refuses the file), and anything
/// below 4 bits refuses the file.
final class KQuantTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-kquant-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    // ── ggml's reference, transcribed (scalar) ──────────────────────────────────────────────

    private static func fp16(_ x: [UInt8], _ i: Int) -> Float {
        Float(Float16(bitPattern: UInt16(x[i]) | UInt16(x[i + 1]) << 8))
    }

    /// `get_scale_min_k4(j, q, &d, &m)`.
    private static func scaleMin(_ j: Int, _ q: ArraySlice<UInt8>) -> (UInt8, UInt8) {
        let q = Array(q)
        if j < 4 { return (q[j] & 63, q[j + 4] & 63) }
        return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4))
    }

    /// `dequantize_row_q4_0` (`five`: `_q5_0`), `_q4_1` (`_q5_1`) with `hasMin`, one block of 32.
    static func referenceLegacy(_ x: [UInt8], five: Bool, hasMin: Bool) -> [Float] {
        let d = fp16(x, 0), m: Float = hasMin ? fp16(x, 2) : 0
        var at = hasMin ? 4 : 2, qh: UInt32 = 0
        if five { qh = UInt32(x[at]) | UInt32(x[at + 1]) << 8 | UInt32(x[at + 2]) << 16 | UInt32(x[at + 3]) << 24; at += 4 }
        var y = [Float](repeating: .nan, count: 32)
        for j in 0..<16 {
            let qs = x[at + j]
            if five {
                let xh0 = Int(((qh >> UInt32(j + 0)) << 4) & 0x10), xh1 = Int((qh >> UInt32(j + 12)) & 0x10)
                let x0 = (Int(qs & 0x0F) | xh0) - (hasMin ? 0 : 16), x1 = (Int(qs >> 4) | xh1) - (hasMin ? 0 : 16)
                y[j] = hasMin ? Float(x0) * d + m : Float(x0) * d
                y[j + 16] = hasMin ? Float(x1) * d + m : Float(x1) * d
            } else {
                let x0 = Int(qs & 0x0F) - (hasMin ? 0 : 8), x1 = Int(qs >> 4) - (hasMin ? 0 : 8)
                y[j] = hasMin ? Float(x0) * d + m : Float(x0) * d
                y[j + 16] = hasMin ? Float(x1) * d + m : Float(x1) * d
            }
        }
        return y
    }

    /// `dequantize_row_q4_K`, one block: `d, dmin, scales[12], qs[128]`.
    static func referenceQ4K(_ x: [UInt8]) -> [Float] {
        let d = fp16(x, 0), min = fp16(x, 2), scales = x[4..<16]
        var y: [Float] = [], q = 16, s = 0
        for _ in stride(from: 0, to: 256, by: 64) {
            var (sc, m) = scaleMin(s + 0, scales)
            let d1 = d * Float(sc), m1 = min * Float(m)
            (sc, m) = scaleMin(s + 1, scales)
            let d2 = d * Float(sc), m2 = min * Float(m)
            for l in 0..<32 { y.append(d1 * Float(x[q + l] & 0xF) - m1) }
            for l in 0..<32 { y.append(d2 * Float(x[q + l] >> 4) - m2) }
            q += 32; s += 2
        }
        return y
    }

    /// `dequantize_row_q5_K`, one block: `d, dmin, scales[12], qh[32], qs[128]`.
    static func referenceQ5K(_ x: [UInt8]) -> [Float] {
        let d = fp16(x, 0), min = fp16(x, 2), scales = x[4..<16]
        var y: [Float] = [], ql = 48, s = 0
        var u1: UInt8 = 1, u2: UInt8 = 2
        for _ in stride(from: 0, to: 256, by: 64) {
            var (sc, m) = scaleMin(s + 0, scales)
            let d1 = d * Float(sc), m1 = min * Float(m)
            (sc, m) = scaleMin(s + 1, scales)
            let d2 = d * Float(sc), m2 = min * Float(m)
            for l in 0..<32 { y.append(d1 * Float(Int(x[ql + l] & 0xF) + (x[16 + l] & u1 != 0 ? 16 : 0)) - m1) }
            for l in 0..<32 { y.append(d2 * Float(Int(x[ql + l] >> 4) + (x[16 + l] & u2 != 0 ? 16 : 0)) - m2) }
            ql += 32; s += 2
            u1 <<= 2; u2 <<= 2
        }
        return y
    }

    /// `dequantize_row_q6_K`, one block: `ql[128], qh[64], scales[16], d`.
    static func referenceQ6K(_ x: [UInt8]) -> [Float] {
        let d = fp16(x, 208)
        var y = [Float](repeating: .nan, count: 256)
        var ql = 0, qh = 128, sc = 192, out = 0
        for _ in stride(from: 0, to: 256, by: 128) {
            for l in 0..<32 {
                let s = l / 16
                let q1 = Int(Int8(bitPattern: (x[ql + l + 0] & 0xF) | (((x[qh + l] >> 0) & 3) << 4))) - 32
                let q2 = Int(Int8(bitPattern: (x[ql + l + 32] & 0xF) | (((x[qh + l] >> 2) & 3) << 4))) - 32
                let q3 = Int(Int8(bitPattern: (x[ql + l + 0] >> 4) | (((x[qh + l] >> 4) & 3) << 4))) - 32
                let q4 = Int(Int8(bitPattern: (x[ql + l + 32] >> 4) | (((x[qh + l] >> 6) & 3) << 4))) - 32
                func scale(_ i: Int) -> Float { Float(Int8(bitPattern: x[sc + i])) }
                y[out + l + 0] = d * scale(s + 0) * Float(q1)
                y[out + l + 32] = d * scale(s + 2) * Float(q2)
                y[out + l + 64] = d * scale(s + 4) * Float(q3)
                y[out + l + 96] = d * scale(s + 6) * Float(q4)
            }
            out += 128; ql += 64; qh += 32; sc += 8
        }
        return y
    }

    // ── synthetic super-blocks ──────────────────────────────────────────────────────────────

    /// SplitMix64: the same blocks at every run.
    private struct Seeded: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }
    private var random = Seeded(state: 0x5111_C0DE)

    private func bytes(_ n: Int, _ fill: UInt8? = nil) -> [UInt8] {
        (0..<n).map { _ in fill ?? UInt8.random(in: 0...255, using: &random) }
    }
    private func le16(_ v: UInt16) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8)] }
    private func randomHalf(_ range: ClosedRange<Float>) -> UInt16 {
        Float16(Float.random(in: range, using: &random)).bitPattern
    }

    /// Special fp16 values for `d`, `dmin`: zero, −0, 1, smallest subnormal, largest finite, negative.
    private static let specialHalves: [UInt16] = [0x0000, 0x8000, 0x3C00, 0x0001, 0x7BFF, Float16(-0.0123).bitPattern]

    /// Block `b` of `n` bytes that starts with fp16 `d` (and `dmin` or `m` when `halves` = 2) —
    /// Q4_0…Q5_1, Q4_K, Q5_K. The first eight of every ten are the extreme ones.
    private func blockWithHalves(_ b: Int, bytes n: Int, halves: Int) -> [UInt8] {
        func head(_ d: UInt16, _ m: UInt16) -> [UInt8] { le16(d) + (halves == 2 ? le16(m) : []) }
        let rest = n - 2 * halves
        switch b % 10 {
        case 0: return bytes(n, 0)                                                    // all zero
        case 1: return head(0x3C00, 0x3800) + bytes(rest, 0xFF)                       // every q, sc, m at its max
        case 2: return head(randomHalf(-0.03 ... -0.001), randomHalf(-0.01 ... -0.0001)) + bytes(rest)
        case 3: return head(0x0001, 0x0001) + bytes(rest)                             // subnormal fp16 scales
        case 4: return head(0x7BFF, 0x7BFF) + bytes(rest, 0xFF)                       // the largest products
        case 5: return head(randomHalf(0.001...0.03), randomHalf(0.001...0.01)) + bytes(rest, 0)   // q = 0: the min alone
        case 6: return head(0x8000, randomHalf(0.001...0.01)) + bytes(rest)           // d = −0
        case 7: return head(randomHalf(0.001...0.03), 0x8000) + bytes(rest)           // min = −0
        default:
            let d = Self.specialHalves[Int.random(in: 0..<Self.specialHalves.count, using: &random)]
            return head(d, randomHalf(-0.01...0.01)) + bytes(rest)
        }
    }

    /// Block `b` of a Q6_K tensor: `ql, qh, scales, d`.
    private func blockQ6(_ b: Int) -> [UInt8] {
        switch b % 10 {
        case 0: return bytes(210, 0)
        case 1: return bytes(192, 0xFF) + (0..<16).map { UInt8(bitPattern: $0 % 2 == 0 ? Int8.min : Int8.max) } + le16(0x3C00)
        case 2: return bytes(208) + le16(randomHalf(-0.03 ... -0.001))
        case 3: return bytes(208) + le16(0x0001)
        case 4: return bytes(192, 0xFF) + bytes(16, 0x80) + le16(0x7BFF)              // sc = −128, q = 31
        case 5: return bytes(192) + bytes(16, 0) + le16(0x3C00)                      // zero scales
        case 6: return bytes(192, 0) + bytes(16) + le16(0x2C00)                      // q = −32 everywhere
        case 7: return bytes(208) + le16(0x8000)
        default: return bytes(208) + le16(randomHalf(0.0005...0.05))
        }
    }

    /// A packed tensor `[n, k]` of ggml type `type`: its bytes, and its reference in published layout.
    private func tensor(_ type: UInt32, _ n: Int, _ k: Int) -> (bytes: [UInt8], reference: [Float]) {
        var data: [UInt8] = [], reference: [Float] = []
        let per = [12, 13, 14].contains(type) ? 256 : 32
        for b in 0..<(n * k / per) {
            let block: [UInt8], values: [Float]
            switch type {
            case 2: block = blockWithHalves(b, bytes: 18, halves: 1); values = Self.referenceLegacy(block, five: false, hasMin: false)
            case 3: block = blockWithHalves(b, bytes: 20, halves: 2); values = Self.referenceLegacy(block, five: false, hasMin: true)
            case 6: block = blockWithHalves(b, bytes: 22, halves: 1); values = Self.referenceLegacy(block, five: true, hasMin: false)
            case 7: block = blockWithHalves(b, bytes: 24, halves: 2); values = Self.referenceLegacy(block, five: true, hasMin: true)
            case 12: block = blockWithHalves(b, bytes: 144, halves: 2); values = Self.referenceQ4K(block)
            case 13: block = blockWithHalves(b, bytes: 176, halves: 2); values = Self.referenceQ5K(block)
            default: block = blockQ6(b); values = Self.referenceQ6K(block)
            }
            data += block
            reference += values
        }
        return (data, reference)
    }

    static let types: [(UInt32, String, Artifact.DType)] = [
        (2, "Q4_0", .q4_0), (3, "Q4_1", .q4_1), (6, "Q5_0", .q5_0), (7, "Q5_1", .q5_1),
        (12, "Q4_K", .q4_k), (13, "Q5_K", .q5_k), (14, "Q6_K", .q6_k),
    ]

    // ── writing a GGUF, forging a map ───────────────────────────────────────────────────────

    struct Tensor { let name: String; let shape: [Int]; let type: UInt32; let bytes: [UInt8] }

    private func string(_ s: String) -> [UInt8] {
        withUnsafeBytes(of: UInt64(s.utf8.count).littleEndian) { Array($0) } + Array(s.utf8)
    }
    private func le<T: FixedWidthInteger>(_ v: T) -> [UInt8] { withUnsafeBytes(of: v.littleEndian) { Array($0) } }

    private func write(_ tensors: [Tensor], name: String = "model.gguf") throws -> String {
        let a = 32
        var infos: [UInt8] = [], data: [UInt8] = []
        for t in tensors {
            infos += string(t.name) + le(UInt32(t.shape.count))
            for d in t.shape.reversed() { infos += le(UInt64(d)) }
            infos += le(t.type) + le(UInt64(data.count))
            data += t.bytes
            data += [UInt8](repeating: 0, count: (a - data.count % a) % a)
        }
        var file = Array("GGUF".utf8) + le(UInt32(3)) + le(UInt64(tensors.count)) + le(UInt64(1))
            + string("general.architecture") + le(UInt32(8)) + string("test") + infos
        file += [UInt8](repeating: 0, count: (a - file.count % a) % a)
        file += data
        let path = root.appendingPathComponent(name).path
        try Data(file).write(to: URL(fileURLWithPath: path))
        return path
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

    private func transposed(_ v: [Float], rows n: Int, columns k: Int) -> [Float] {
        var t = [Float](repeating: 0, count: n * k)
        for r in 0..<n { for c in 0..<k { t[c * n + r] = v[r * k + c] } }
        return t
    }

    /// CPU (`materialize`) and GPU (`WidenGPU`) both give `expected`, bit for bit — the GPU path is
    /// required for a K-quant whenever there is a GPU.
    private func assertDequantizes(_ map: Artifact, _ name: String, to expected: [Float],
                                   file: StaticString = #filePath, line: UInt = #line) throws {
        let t = try XCTUnwrap(map.tensors[name], file: file, line: line)
        var cpu = [Float](repeating: -1, count: t.count)
        try cpu.withUnsafeMutableBufferPointer { _ = try map.materialize(name, into: $0) }
        XCTAssertEqual(cpu.map(\.bitPattern), expected.map(\.bitPattern), "CPU \(name)", file: file, line: line)
        guard let (widener, device) = gpu else { return }
        let source = widener.source(name, in: map)
        if t.dtype.isPacked { XCTAssertNotNil(source, "a packed type has its GPU kernel", file: file, line: line) }
        guard let source else { return }
        let buffer = try XCTUnwrap(device.makeBuffer(length: t.count * 4, options: .storageModeShared))
        memset(buffer.contents(), 0xFF, t.count * 4)
        try widener.run(source: source, destination: buffer, count: t.count)
        let g = buffer.contents().assumingMemoryBound(to: Float.self)
        XCTAssertEqual((0..<t.count).map { g[$0].bitPattern }, expected.map(\.bitPattern), "GPU \(name)", file: file, line: line)
    }

    // ── the tests ───────────────────────────────────────────────────────────────────────────

    /// On a Mac with a GPU, the widening kernels compile — otherwise every GPU comparison here
    /// (and in `GGUFTests`, `QuantizedMapTests`) would be skipped in silence.
    func testTheKernelsCompile() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return }
        XCTAssertNoThrow(try WidenGPU(device: device, queue: queue))
    }

    /// The engine's routine against ggml's formula, both layouts, every kind of block — and row
    /// counts that are not multiples of the transposing tile (256 rows, 4-row transposes): one
    /// partial tile, then a whole one and a partial one.
    func testPackedBlocksDequantizeAtTheBit() {
        for (type, label, kind) in Self.types {
            for (n, k) in [(21, 768), (263, 512)] {
                let t = tensor(type, n, k)
                XCTAssertTrue(t.reference.contains { $0 != 0 } && t.reference.allSatisfy(\.isFinite), label)
                for transposing in [false, true] {
                    var out = [Float](repeating: .nan, count: n * k)
                    t.bytes.withUnsafeBytes { b in
                        out.withUnsafeMutableBufferPointer {
                            Widen.dequantizePacked(b.baseAddress!, kind: kind, rows: n, columns: k,
                                                        transposing: transposing, into: $0.baseAddress!)
                        }
                    }
                    let expected = transposing ? transposed(t.reference, rows: n, columns: k) : t.reference
                    XCTAssertEqual(out.map(\.bitPattern), expected.map(\.bitPattern), "\(label) \(n)×\(k) transposing \(transposing)")
                }
            }
        }
    }

    /// Z-Image's ComfyUI names in a packed GGUF: the fused qkv split in three (whole rows of
    /// blocks), an MLP weight; each kept as published — the map's bytes are the published
    /// rows — and read by the engine, CPU and GPU, at the bit. Two forges compare identical.
    func testZImageGGUFBecomesAPackedMap() throws {
        for (type, label, kind) in Self.types {
            let (n, k) = (20, 512)
            let qkv = tensor(type, 3 * n, k), w1 = tensor(type, 2 * n, k)
            let path = try write([
                Tensor(name: "layers.0.attention.qkv.weight", shape: [3 * n, k], type: type, bytes: qkv.bytes),
                Tensor(name: "layers.0.feed_forward.w1.weight", shape: [2 * n, k], type: type, bytes: w1.bytes),
            ], name: "z-\(label).gguf")
            let source = try TensorSource(paths: [path])
            XCTAssertEqual(source.dtype("layers.0.attention.qkv.weight"), label)
            XCTAssertEqual(source.quantization("layers.0.attention.qkv.weight")?.kind, kind)
            XCTAssertEqual(source.quantizedFormats, ["GGUF \(label)": 2])
            XCTAssertEqual(try source.read("layers.0.attention.qkv.weight").map(\.bitPattern), qkv.reference.map(\.bitPattern),
                           "\(label): the fp32 reading of the source")

            let (map, outcomes) = try forge(source, family: .zImage)
            let rowBytes = k / kind.packedBlock!.values * kind.packedBlock!.bytes
            for (part, x) in ["to_q", "to_k", "to_v"].enumerated() {
                let name = "layers.0.attention.\(x).weight"
                XCTAssertEqual(outcomes[name], .keptPacked, "\(label) \(name)")
                let t = try XCTUnwrap(map.tensors[name])
                XCTAssertEqual(t.dtype, kind)
                XCTAssertEqual(t.shape, [k, n], "the map's shape is [K, N], as for every Linear")
                XCTAssertNil(t.scale)
                XCTAssertEqual(t.bytes, n * rowBytes)
                // The published super-blocks, rows moved, bytes unchanged.
                let mapped = [UInt8](UnsafeRawBufferPointer(start: map.pointer(name)!, count: t.bytes))
                XCTAssertEqual(mapped, Array(qkv.bytes[(part * n * rowBytes)..<((part + 1) * n * rowBytes)]), "\(label) \(name) bytes")
                let rows = Array(qkv.reference[(part * n * k)..<((part + 1) * n * k)])
                try assertDequantizes(map, name, to: transposed(rows, rows: n, columns: k))
            }
            try assertDequantizes(map, "layers.0.feed_forward.w1.weight", to: transposed(w1.reference, rows: 2 * n, columns: k))

            // A row of `[K, N]` is a column of the super-blocks: refused, never read wrong.
            var row = [Float](repeating: 0, count: n)
            XCTAssertThrowsError(try row.withUnsafeMutableBufferPointer {
                try map.materializeRows("layers.0.attention.to_q.weight", first: 0, count: 1, into: $0.baseAddress!)
            }) { XCTAssertTrue("\($0)".contains("blocks — stored by output"), "\($0)") }

            let (again, _) = try forge(source, family: .zImage)
            let verdict = try MapComparison.compare(map.path, again.path)
            XCTAssertTrue(verdict.allIdentical, "\(label): \(verdict)")
            XCTAssertEqual(verdict.equalTensors, map.order.count)
        }
    }

    /// Qwen-Image-2.1's fused `img_mlp.gate_up` (ComfyUI) in Q6_K: two halves of rows; and a
    /// K-quant that is not a `Linear` goes to fp32, at the bit, and says why.
    func testRowSharesAndANonLinearKQuant() throws {
        let (h, k) = (6, 256)
        let w = tensor(14, 2 * h, k), pad = tensor(12, 1, 512)
        let path = try write([
            Tensor(name: "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight", shape: [2 * h, k], type: 14, bytes: w.bytes),
            Tensor(name: "model.diffusion_model.txt_in.text_norm.weight", shape: [8], type: 0,
                   bytes: [Float](repeating: 0.5, count: 8).flatMap { le($0.bitPattern) }),
        ], name: "qwen.gguf")
        let (map, outcomes) = try forge(try TensorSource(paths: [path]), family: .qwenImage21)
        try assertDequantizes(map, "transformer_blocks.0.img_mlp.gate_layer.weight",
                              to: transposed(Array(w.reference[0..<(h * k)]), rows: h, columns: k))
        try assertDequantizes(map, "transformer_blocks.0.img_mlp.proj.weight",
                              to: transposed(Array(w.reference[(h * k)...]), rows: h, columns: k))
        XCTAssertEqual(outcomes["transformer_blocks.0.img_mlp.proj.weight"], .keptPacked)

        // An F32 norm beside it: a GGUF dimension may not exceed the file's size (`GGUF.parse`).
        let zPath = try write([Tensor(name: "cap_pad_token", shape: [1, 512], type: 12, bytes: pad.bytes),
                               Tensor(name: "layers.0.attention.q_norm.weight", shape: [512], type: 0,
                                      bytes: [Float](repeating: 1, count: 512).flatMap { le($0.bitPattern) })], name: "pad.gguf")
        let (z, zOutcomes) = try forge(try TensorSource(paths: [zPath]), family: .zImage)
        XCTAssertEqual(z.tensors["cap_pad_token"]?.dtype, .float32)
        guard case .dequantized(let why)? = zOutcomes["cap_pad_token"] else {
            return XCTFail("\(String(describing: zOutcomes["cap_pad_token"]))")
        }
        XCTAssertTrue(why.contains("not a Linear"), why)
        try assertDequantizes(z, "cap_pad_token", to: pad.reference)
    }

    /// FLUX.2's single-block `to_out`, cut on its input: on a block boundary (256 for a K-quant, 32
    /// for Q4_0…Q5_1) both shares stay packed with their bytes; inside a block, the file is refused.
    func testColumnSplitOnlyOnABlockBoundary() throws {
        let k = 512
        let d = 256, w = tensor(13, d, k)
        let path = try write([Tensor(name: "single_blocks.0.linear2.weight", shape: [d, k], type: 13, bytes: w.bytes)], name: "klein.gguf")
        let (map, outcomes) = try forge(try TensorSource(paths: [path]), family: .klein4b)
        for (part, begin, number) in [("attn", 0, d), ("mlp", d, k - d)] {
            let name = "single_transformer_blocks.0.attn.to_out.\(part).weight"
            XCTAssertEqual(outcomes[name], .keptPacked, name)
            let share = (0..<d).flatMap { r in w.reference[(r * k + begin)..<(r * k + begin + number)] }
            try assertDequantizes(map, name, to: transposed(share, rows: d, columns: number))
        }

        let cut = 128, c = tensor(12, cut, k)
        let bad = try write([Tensor(name: "single_blocks.0.linear2.weight", shape: [cut, k], type: 12, bytes: c.bytes)], name: "klein-cut.gguf")
        XCTAssertThrowsError(try Recipes.normalize(try TensorSource(paths: [bad]), family: .klein4b)) { e in
            XCTAssertTrue("\(e)".contains("Q4_K") && "\(e)".contains("inside a block of 256") && "\(e)".contains("[0, 128)"), "\(e)")
        }

        // Q4_0's blocks are 32: a share of 64 inputs keeps them, one of 48 cuts one.
        let (d0, k0) = (64, 128), w0 = tensor(2, d0, k0)
        let legacy = try write([Tensor(name: "single_blocks.0.linear2.weight", shape: [d0, k0], type: 2, bytes: w0.bytes)], name: "klein-q4_0.gguf")
        let (m0, o0) = try forge(try TensorSource(paths: [legacy]), family: .klein4b)
        for (part, begin, number) in [("attn", 0, d0), ("mlp", d0, k0 - d0)] {
            let name = "single_transformer_blocks.0.attn.to_out.\(part).weight"
            XCTAssertEqual(o0[name], .keptPacked, name)
            let share = (0..<d0).flatMap { r in w0.reference[(r * k0 + begin)..<(r * k0 + begin + number)] }
            try assertDequantizes(m0, name, to: transposed(share, rows: d0, columns: number))
        }
        let c0 = tensor(7, 48, 128)
        let bad0 = try write([Tensor(name: "single_blocks.0.linear2.weight", shape: [48, 128], type: 7, bytes: c0.bytes)], name: "klein-q5_1-cut.gguf")
        XCTAssertThrowsError(try Recipes.normalize(try TensorSource(paths: [bad0]), family: .klein4b)) { e in
            XCTAssertTrue("\(e)".contains("Q5_1") && "\(e)".contains("inside a block of 32"), "\(e)")
        }
    }

    /// 4 bits is the floor: Q3_K, Q2_K, the I-quants, TQ, MXFP4 and Q8_1 refuse the whole file, by
    /// name; so does a row that is not whole blocks, or a non-finite fp16 scale.
    func testBelowTheFloorAndMalformedAreRefused() throws {
        let ok = tensor(12, 4, 256)
        for (type, label) in [(UInt32(11), "Q3_K"), (10, "Q2_K"), (23, "IQ4_XS"), (20, "IQ4_NL"), (34, "TQ1_0"), (39, "MXFP4"), (9, "Q8_1")] {
            let path = try write([Tensor(name: "layers.0.attention.qkv.weight", shape: [4, 256], type: 12, bytes: ok.bytes),
                                  Tensor(name: "layers.0.feed_forward.w1.weight", shape: [4, 256], type: type,
                                         bytes: [UInt8](repeating: 0, count: 4 * 210))],
                                 name: "below-\(label).gguf")
            XCTAssertThrowsError(try TensorSource(paths: [path])) { e in
                XCTAssertTrue("\(e)".contains(label) && "\(e)".contains("4 bits is the floor")
                              && "\(e)".contains("refused in its entirety"), "\(e)")
            }
            XCTAssertThrowsError(try ModelImport.recognize(path)) { e in
                guard case EngineError.importRefused(_, let detail) = e else { return XCTFail("\(e)") }
                XCTAssertTrue(detail.contains(label), detail)
            }
        }
        let short = try write([Tensor(name: "w", shape: [4, 128], type: 14, bytes: [UInt8](repeating: 0, count: 4 * 210))], name: "short.gguf")
        XCTAssertThrowsError(try TensorSource(paths: [short])) { XCTAssertTrue("\($0)".contains("multiple of 256"), "\($0)") }
        let odd = try write([Tensor(name: "w", shape: [4, 48], type: 2, bytes: [UInt8](repeating: 0, count: 4 * 27))], name: "odd.gguf")
        XCTAssertThrowsError(try TensorSource(paths: [odd])) { XCTAssertTrue("\($0)".contains("multiple of 32"), "\($0)") }

        for (offset, type) in [(0, UInt32(12)), (2, 13), (208, 14), (0, 2), (2, 3), (0, 6), (2, 7)] {
            var b = tensor(type, 2, 256).bytes
            b[offset] = 0x00; b[offset + 1] = 0x7C                                   // fp16 +∞
            let path = try write([Tensor(name: "layers.0.feed_forward.w1.weight", shape: [2, 256], type: type, bytes: b)],
                                 name: "inf-\(type).gguf")
            // Checked where the blocks are read (the forge), not at the file's opening: a scan there
            // would fault in the whole file.
            let source = try TensorSource(paths: [path])
            XCTAssertThrowsError(try forge(source, family: .zImage)) { XCTAssertTrue("\($0)".contains("non-finite"), "\($0)") }
        }
    }

    /// A map whose packed entry does not add up is refused at the door, never read past.
    func testAMalformedPackedEntryRefusesTheMap() throws {
        let t = tensor(12, 2, 256)
        let good = MapWriter.Tensor(name: "w", shape: [256, 2], dtype: .q4_k, scale: nil, transposed: true) {
            .init(values: t.bytes, scale: [], absMax: 0)
        }
        let path = root.appendingPathComponent("ok.silicon").path
        // A tensor after it: the GPU wraps whole pages, which the map's last tensor would lack.
        let end = MapWriter.Tensor(name: "~end", shape: [4], dtype: .bfloat16, transposed: nil) { [0, 0, 0, 0] }
        _ = try MapWriter.write(to: path, tensors: [good, end], header: { _ in [.init("format", 1), .init("page", .integer(MapWriter.page))] })
        let map = try Artifact(path: path)
        XCTAssertEqual(map.tensors["w"]?.bytes, 2 * 144)
        try assertDequantizes(map, "w", to: transposed(t.reference, rows: 2, columns: 256))

        // The same file, the entry's shape changed to a K that is not whole super-blocks.
        var data = try Data(contentsOf: URL(fileURLWithPath: path))
        let (from, to) = data.range(of: Data("[256,2]".utf8)) != nil ? ("[256,2]", "[128,4]") : ("[256, 2]", "[128, 4]")
        let range = try XCTUnwrap(data.range(of: Data(from.utf8)))
        data.replaceSubrange(range, with: Data(to.utf8))
        let bad = root.appendingPathComponent("bad.silicon").path
        try data.write(to: URL(fileURLWithPath: bad))
        XCTAssertThrowsError(try Artifact(path: bad)) { XCTAssertTrue("\($0)".contains("multiple of 256"), "\($0)") }
    }
}
