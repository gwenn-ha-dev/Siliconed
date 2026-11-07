import Metal
import XCTest
@testable import Siliconed

/// **SQ, read and decoded** (`SQ`): a group of 16 is `v = m · (q − 2^(bits−1)) + b` exactly, written here from the format's own
/// definition (independently of the decoder); a row is its groups; the map's int8 is `[K, N]`; the forge puts the SQ linears of a
/// file over the publisher's weights, and the map gives back `W = (s ⊙ v) · R` — the int8 convrot of a Compact; the GPU's decoding
/// writes the CPU's bytes.
final class SQTests: XCTestCase {
    private var generator = SystemRandomNumberGenerator()
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-sq-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    // ── the format, from its definition ────────────────────────────────────────────────────────────

    /// `(bits, offset bits, unit)`: SQ4 `(M << 4) | B`, b = 4 · (B − 8); SQ5 `(M << 5) | B`, b = 2 · (B − 16).
    private func layout(_ bits: Int) -> (offsetBits: Int, unit: Int) { bits == 4 ? (4, 4) : (5, 2) }

    /// One group's value j, by the format's formula.
    private func value(code q: Int, byte: UInt8, bits: Int) -> Int {
        let (ob, unit) = layout(bits)
        let m = Int(byte) >> ob + 1, b = (Int(byte) & ((1 << ob) - 1) - (1 << (bits - 1))) * unit
        return m * (q - (1 << (bits - 1))) + b
    }

    /// 16 codes as a little-endian bit string of `2 · bits` bytes (code j at bits j · bits …).
    private func pack(_ codes: [Int], bits: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: 2 * bits)
        for (j, q) in codes.enumerated() {
            for t in 0..<bits where (q >> t) & 1 == 1 { out[(j * bits + t) / 8] |= 1 << UInt8((j * bits + t) % 8) }
        }
        return out
    }

    /// A random row that decodes inside int8: per group a random (m, B), its codes drawn among those that stay in [−128, 127].
    private func row(k: Int, bits: Int) -> (stored: [UInt8], values: [Int8]) {
        var codes: [UInt8] = [], bytes: [UInt8] = [], values: [Int8] = []
        for _ in 0..<(k / SQ.group) {
            var byte: UInt8, allowed: [Int]
            repeat {
                byte = UInt8.random(in: 0...255, using: &generator)
                let b = byte
                allowed = (0..<(1 << bits)).filter { (-128...127).contains(value(code: $0, byte: b, bits: bits)) }
            } while allowed.isEmpty
            let q = (0..<SQ.group).map { _ in allowed.randomElement(using: &generator)! }
            codes += pack(q, bits: bits)
            bytes.append(byte)
            values += q.map { Int8(value(code: $0, byte: byte, bits: bits)) }
        }
        return (codes + bytes, values)
    }

    func testAGroupIsItsFormula() {
        for bits in [4, 5] {
            let k = 512
            let (stored, values) = row(k: k, bits: bits)
            XCTAssertEqual(stored.count, SQ.rowBytes(k: k, bits: bits), "SQ\(bits): K·bits/8 code bytes then K/16 group bytes")
            var out = [Int8](repeating: 0, count: k)
            let valid = stored.withUnsafeBytes { s in out.withUnsafeMutableBufferPointer { SQ.decodeRow(s.baseAddress!, k: k, bits: bits, into: $0.baseAddress!) } }
            XCTAssertTrue(valid)
            XCTAssertEqual(out, values, "SQ\(bits)")
        }
    }

    /// The extremes: m = 16, the top code and the largest offset leave int8 (16 · 7 + 28 = 140) — the decoder says so.
    func testAGroupOutsideInt8IsSaid() {
        let codes = pack([Int](repeating: 15, count: 16), bits: 4), byte: UInt8 = 0xFF
        XCTAssertEqual(value(code: 15, byte: byte, bits: 4), 140)
        var out = [Int8](repeating: 0, count: 16)
        let valid = (codes + [byte]).withUnsafeBytes { s in out.withUnsafeMutableBufferPointer { SQ.decodeRow(s.baseAddress!, k: 16, bits: 4, into: $0.baseAddress!) } }
        XCTAssertFalse(valid)
        let stored = codes + [byte]
        XCTAssertEqual(stored.withUnsafeBytes { SQ.firstInvalidRow($0.baseAddress!, rows: 1, columns: 16, bits: 4) }, 0)
    }

    /// N = 70 outputs: a band of 64 and one of 6, written `[K, N]`.
    func testRowsDecodeTransposed() {
        let (n, k) = (70, 256)
        for bits in [4, 5] {
            let rows = (0..<n).map { _ in row(k: k, bits: bits) }
            let stored = rows.flatMap(\.stored)
            var out = [Int8](repeating: 0, count: n * k)
            stored.withUnsafeBytes { s in out.withUnsafeMutableBufferPointer { SQ.decodeTransposed(s.baseAddress!, rows: n, columns: k, bits: bits, into: $0.baseAddress!) } }
            for r in 0..<n { for c in 0..<k where out[c * n + r] != rows[r].values[c] { return XCTFail("SQ\(bits) (\(r), \(c))") } }
        }
    }

    // ── the forge and the map ──────────────────────────────────────────────────────────────────────

    private struct Piece { let name: String; let dtype: String; let shape: [Int]; let bytes: [UInt8] }

    private func write(_ pieces: [Piece], metadata: [String: String] = [:], as file: String) throws -> String {
        var header: [String: Any] = [:], position = 0
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

    /// R256 = (H4 ⊗ H4 ⊗ H4 ⊗ H4) / 16, H4 = 1 except −1 on its anti-diagonal: entry (i, j) the product over base-4 digits.
    private func hadamard(_ i: Int, _ j: Int) -> Double {
        var p = 1.0, (a, b) = (i, j)
        for _ in 0..<4 { if a % 4 + b % 4 == 3 { p = -p }; a /= 4; b /= 4 }
        return p / 16
    }

    /// SQ4 for w1, SQ5 for w2 and SQ8 for to_q over a publisher's bf16 file that holds all three: the SQ ones replace theirs, the
    /// map keeps the stored rows, and widens to `W = (s ⊙ v) · R`; `stageSQ` writes the int8 `[K, N]` and the scales.
    func testTheForgePutsSQOverThePublishersWeights() throws {
        let (n, k) = (24, 512)
        let r256 = (0..<(256 * 256)).map { hadamard($0 / 256, $0 % 256) }
        let names = ["layers.0.feed_forward.w1", "layers.0.feed_forward.w2", "layers.0.attention.to_q"]
        let bf16 = [UInt8](repeating: 0x3F, count: 2 * n * k)
        let base = try write(names.map { Piece(name: $0 + ".weight", dtype: "BF16", shape: [n, k], bytes: bf16) }
                             + [Piece(name: "layers.0.attention_norm1.weight", dtype: "BF16", shape: [k], bytes: Array(bf16.prefix(2 * k)))],
                             as: "base.safetensors")
        var pieces: [Piece] = [], integers: [String: [Int8]] = [:], scales: [String: [Float]] = [:]
        for (name, bits) in zip(names, [4, 5, 8]) {
            let s = (0..<n).map { _ in Float.random(in: 1e-4...3e-3, using: &generator) }
            scales[name] = s
            pieces.append(Piece(name: name + ".sq.scale", dtype: "F32", shape: [n], bytes: s.withUnsafeBytes { Array($0) }))
            if bits == 8 {
                let q = (0..<(n * k)).map { _ in Int8.random(in: -128...127, using: &generator) }
                integers[name] = q
                pieces.append(Piece(name: name + ".sq.codes", dtype: "I8", shape: [n, k], bytes: q.map { UInt8(bitPattern: $0) }))
                continue
            }
            let rows = (0..<n).map { _ in row(k: k, bits: bits) }
            integers[name] = rows.flatMap(\.values)
            let codeBytes = k * bits / 8
            pieces.append(Piece(name: name + ".sq.codes", dtype: "U8", shape: [n, codeBytes], bytes: rows.flatMap { $0.stored.prefix(codeBytes) }))
            pieces.append(Piece(name: name + ".sq.groups", dtype: "U8", shape: [n, k / 16], bytes: rows.flatMap { $0.stored.suffix(k / 16) }))
        }
        let tiers = "{\"layers.0.feed_forward.w1\": \"SQ4\", \"layers.0.feed_forward.w2\": \"SQ5\", \"layers.0.attention.to_q\": \"SQ8\"}"
        let sq = try write(pieces, metadata: ["format": "SQ", "tiers": tiers], as: "sq.safetensors")
        let source = try TensorSource(paths: [base, sq])
        XCTAssertEqual(source.names.filter { $0.hasSuffix(".weight") }.count, 4, "each SQ linear once, under its model name")
        XCTAssertFalse(source.names.contains { $0.contains(".sq.") })
        XCTAssertEqual(source.quantizedFormats, ["Siliconed SQ4": 1, "Siliconed SQ5": 1, "Siliconed SQ8": 1])
        XCTAssertEqual(source.shape("layers.0.feed_forward.w1.weight"), [n, k], "the shape the codes stand for")

        let normalized = try Recipes.normalize(source, family: .zImage)
        var tensors: [MapWriter.Tensor] = []
        for name in normalized.dit.keys.sorted() { tensors.append(try ForgeDiT.tensor(name, normalized.dit[name]!, family: .zImage).0) }
        let path = root.appendingPathComponent("sq.silicon").path
        _ = try MapWriter.write(to: path, tensors: tensors, header: { _ in
            [.init("format", 1), .init("page", .integer(MapWriter.page)), .init("linear_weights_transposed", true)]
        })
        let map = try Artifact(path: path)
        for (name, kind) in zip(names, [Artifact.DType.sq4, .sq5, .int8]) {
            let t = try XCTUnwrap(map.tensors[name + ".weight"])
            XCTAssertEqual(t.dtype, kind)
            XCTAssertEqual(t.shape, [k, n])
            XCTAssertEqual(t.rotation, Artifact.Rotation(kind: "convrot", group: 256))
            XCTAssertEqual(t.scale?.shape, [n])
            // The widened weight against the definition, in fp64: W[c, r] = s[r] · Σ_j v[r, j] · R[j, c] (one 256 block).
            var w = [Float](repeating: .nan, count: k * n)
            try w.withUnsafeMutableBufferPointer { _ = try map.materialize(name + ".weight", into: $0) }
            let v = integers[name]!, s = scales[name]!
            var worst = 0.0
            for r in 0..<n {
                for c in 0..<k {
                    let block = c / 256 * 256
                    let dot = (0..<256).reduce(0.0) { (sum: Double, j: Int) -> Double in sum + Double(v[r * k + block + j]) * r256[j * 256 + c - block] }
                    let exact = Double(s[r]) * dot
                    worst = max(worst, abs(Double(w[c * n + r]) - exact) / max(abs(exact), 1e-30))
                }
            }
            XCTAssertLessThanOrEqual(worst, 1.2e-7, "\(name): (s ⊙ v) · R, within an ulp")
            guard kind != .int8 else { continue }
            // The CPU's staging: the integers `[K, N]`, then the scales.
            let scalesAt = (k * n + 15) / 16 * 16
            var staged = [UInt8](repeating: 0, count: scalesAt + 4 * n)
            try staged.withUnsafeMutableBytes { try map.stageSQ(name + ".weight", into: $0.baseAddress!, scalesAt: scalesAt) }
            for r in 0..<n { for c in 0..<k where Int8(bitPattern: staged[c * n + r]) != v[r * k + c] { return XCTFail("\(name) staged (\(r), \(c))") } }
            XCTAssertEqual(Array(staged[scalesAt...]), s.withUnsafeBytes { Array($0) })
            // The GPU's decoding: the same bytes.
            guard let device = MTLCreateSystemDefaultDevice(), let kernels = SQKernels.of(device), let queue = device.makeCommandQueue(),
                  let offset = map.mappedOffset(name + ".weight"), let wrapped = map.wrappedMap(on: device),
                  let reserve = device.makeBuffer(length: staged.count, options: .storageModeShared) else { continue }
            let commands = queue.makeCommandBuffer()!
            kernels.encode(commands, source: wrapped, offset: offset, scalesOffset: offset + t.scale!.offset, into: reserve,
                           scalesAt: scalesAt, k: k, n: n, bits: kind.sqBits!)
            commands.commit()
            commands.waitUntilCompleted()
            XCTAssertEqual(memcmp(reserve.contents(), staged, staged.count), 0, "\(name): the GPU writes the CPU's bytes")
        }
    }

    /// A stored group that decodes outside int8 refuses the file at the forge.
    func testTheForgeRefusesAGroupOutsideInt8() throws {
        let (n, k) = (2, 256)
        var stored = row(k: k, bits: 4).stored + row(k: k, bits: 4).stored
        let codeBytes = k / 2
        for i in 0..<8 { stored[SQ.rowBytes(k: k, bits: 4) + i] = 0xFF }         // row 1, group 0: every code 15
        stored[SQ.rowBytes(k: k, bits: 4) + codeBytes] = 0xFF                    // m = 16, b = 28: 140
        let rows = [Array(stored.prefix(SQ.rowBytes(k: k, bits: 4))), Array(stored.suffix(SQ.rowBytes(k: k, bits: 4)))]
        let s: [Float] = [1e-3, 1e-3]
        let path = try write([Piece(name: "layers.0.feed_forward.w1.sq.codes", dtype: "U8", shape: [n, codeBytes], bytes: rows.flatMap { $0.prefix(codeBytes) }),
                              Piece(name: "layers.0.feed_forward.w1.sq.groups", dtype: "U8", shape: [n, k / 16], bytes: rows.flatMap { $0.suffix(k / 16) }),
                              Piece(name: "layers.0.feed_forward.w1.sq.scale", dtype: "F32", shape: [n], bytes: s.withUnsafeBytes { Array($0) })],
                             metadata: ["format": "SQ", "tiers": "{\"layers.0.feed_forward.w1\": \"SQ4\"}"], as: "bad.safetensors")
        let source = try TensorSource(paths: [path])
        XCTAssertThrowsError(try source.readQuantized("layers.0.feed_forward.w1.weight")) { error in
            XCTAssertTrue("\(error)".contains("row 1"), "\(error)")
        }
    }

    /// Anima's publisher file is ComfyUI's (`blocks.N.self_attn.q_proj`), the SQ file names its linears the diffusers way: the SQ
    /// tensor replaces the one that maps to its name (once, not twice), and the RoPE permutation of `to_q` moves its stored rows.
    func testAnimaSQReplacesTheComfyTensorAndPermutesItsRows() throws {
        let (n, k) = (256, 256), head = ForgeDiT.animaHeadSize
        let bf16 = [UInt8](repeating: 0x3F, count: 2 * n * k)
        let base = try write([Piece(name: "blocks.0.self_attn.q_proj.weight", dtype: "BF16", shape: [n, k], bytes: bf16),
                              Piece(name: "blocks.0.self_attn.k_proj.weight", dtype: "BF16", shape: [n, k], bytes: bf16)],
                             as: "comfy.safetensors")
        let s = (0..<n).map { _ in Float.random(in: 1e-4...3e-3, using: &generator) }
        let rows = (0..<n).map { _ in row(k: k, bits: 4) }
        let codeBytes = k / 2
        let sq = try write([Piece(name: "transformer_blocks.0.attn1.to_q.sq.codes", dtype: "U8", shape: [n, codeBytes], bytes: rows.flatMap { $0.stored.prefix(codeBytes) }),
                            Piece(name: "transformer_blocks.0.attn1.to_q.sq.groups", dtype: "U8", shape: [n, k / 16], bytes: rows.flatMap { $0.stored.suffix(k / 16) }),
                            Piece(name: "transformer_blocks.0.attn1.to_q.sq.scale", dtype: "F32", shape: [n], bytes: s.withUnsafeBytes { Array($0) })],
                           metadata: ["format": "SQ", "tiers": "{\"transformer_blocks.0.attn1.to_q\": \"SQ4\"}"], as: "sq.safetensors")
        let source = try TensorSource(paths: [base, sq])
        let normalized = try Recipes.normalize(source, family: .anima)
        XCTAssertEqual(Set(normalized.dit.keys), ["transformer_blocks.0.attn1.to_q.weight", "transformer_blocks.0.attn1.to_k.weight"])
        let q = try XCTUnwrap(normalized.dit["transformer_blocks.0.attn1.to_q.weight"])
        XCTAssertEqual(q.quantization?.kind, .sq4, "the SQ tensor stands in for the ComfyUI one")
        XCTAssertNil(normalized.dit["transformer_blocks.0.attn1.to_k.weight"]?.quantization)
        // The forged tensor is the SQ rows moved by the interleave, and its widened weight the published one's permuted.
        let (tensor, outcome) = try ForgeDiT.tensor("transformer_blocks.0.attn1.to_q.weight", q, family: .anima)
        XCTAssertEqual(outcome, .keptPacked)
        XCTAssertEqual(tensor.dtype, .sq4)
        let path = root.appendingPathComponent("anima.silicon").path
        _ = try MapWriter.write(to: path, tensors: [tensor], header: { _ in
            [.init("format", 1), .init("page", .integer(MapWriter.page)), .init("linear_weights_transposed", true)]
        })
        let map = try Artifact(path: path)
        var widened = [Float](repeating: .nan, count: k * n)
        try widened.withUnsafeMutableBufferPointer { _ = try map.materialize("transformer_blocks.0.attn1.to_q.weight", into: $0) }
        let published = try XCTUnwrap(source.quantization("transformer_blocks.0.attn1.to_q.weight")).map { _ in try source.readQuantized("transformer_blocks.0.attn1.to_q.weight") }
        let expected = try XCTUnwrap(published).selectingRows(ForgeDiT.interleaveOrder(rows: n, head: head)).dequantized()   // [N, K]
        for r in 0..<n { for c in 0..<k where widened[c * n + r] != expected[r * k + c] { return XCTFail("(\(r), \(c))") } }
    }
}
