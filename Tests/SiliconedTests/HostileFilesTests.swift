import Foundation
import XCTest
@testable import Siliconed

/// **A file that lies is refused by name, never trusted**: a safetensors or GGUF header whose
/// numbers point outside the file or overflow, a JSON nested without end or with a broken
/// surrogate pair, a 1-D tensor under a matrix's name, a Hub response that ignores `Range` or
/// announces no sha256, an `alpha_pattern` shaped for catastrophic backtracking, a `.partiel` left
/// by a crash, an import over an existing name. All written here, in a temporary folder.
final class HostileFilesTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-hostile-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private func le<T: FixedWidthInteger>(_ v: T) -> [UInt8] { withUnsafeBytes(of: v.littleEndian) { Array($0) } }

    private func file(_ name: String, _ bytes: [UInt8]) throws -> String {
        let path = root.appendingPathComponent(name).path
        try Data(bytes).write(to: URL(fileURLWithPath: path))
        return path
    }

    /// A safetensors: the length, the JSON header as given, then `data` bytes.
    private func safetensors(_ name: String, header: String, data: Int, length: UInt64? = nil) throws -> String {
        let json = Array(header.utf8)
        return try file(name, le(length ?? UInt64(json.count)) + json + [UInt8](repeating: 0, count: data))
    }

    private func assertRefused(_ path: String, _ fragment: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try Safetensors(path: path), file: file, line: line) { e in
            guard case Safetensors.Failure.badHeader(let why) = e else { return XCTFail("\(e)", file: file, line: line) }
            XCTAssertTrue(why.contains(fragment), why, file: file, line: line)
        }
    }

    // ── safetensors ─────────────────────────────────────────────────────────────────────────

    func testSafetensorsValidFileReadsAsBefore() throws {
        let p = try safetensors("ok.safetensors", header: #"{"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}, "#
                                + #""__metadata__": {"format": "pt"}}"#, data: 24)
        let s = try Safetensors(path: p)
        XCTAssertEqual(s.entries["w"]?.shape, [2, 3])
        XCTAssertEqual(s.entries["w"]?.bytes, 24)
        XCTAssertEqual(s.metadata["format"], "pt")
        XCTAssertEqual(s.float32("w")?.count, 6)
    }

    func testSafetensorsTooShortIsNamedNotUnopenable() throws {
        for n in [0, 3, 8] {
            assertRefused(try file("short\(n).safetensors", [UInt8](repeating: 1, count: n)), "too short")
        }
    }

    func testSafetensorsHeaderLengthIsBounded() throws {
        // Beyond `Int`, beyond the file, beyond the specification's 100 MB.
        assertRefused(try safetensors("huge.safetensors", header: "{}", data: 0, length: .max), "length")
        assertRefused(try safetensors("past.safetensors", header: "{}", data: 0, length: 3), "length")
        assertRefused(try safetensors("spec.safetensors", header: "{}", data: 0, length: 100_000_001), "length")
    }

    func testSafetensorsOffsetsAndShapesAreBounded() throws {
        let cases: [(String, String)] = [
            (#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [-4, 0]}}"#, "data_offsets"),
            (#"{"w": {"dtype": "F32", "shape": [1], "data_offsets": [8, 4]}}"#, "data_offsets"),
            (#"{"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}"#, "data_offsets"),    // past the 8 bytes
            (#"{"w": {"dtype": "F32", "shape": [-1, -1], "data_offsets": [0, 4]}}"#, "not a size"),
            (#"{"w": {"dtype": "U8", "shape": [4294967296, 4294967296, 4], "data_offsets": [0, 0]}}"#, "not a size"),
            (#"{"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}"#, "8 bytes for [3]"),
        ]
        for (k, (header, fragment)) in cases.enumerated() {
            assertRefused(try safetensors("bad\(k).safetensors", header: header, data: 8), fragment)
        }
    }

    // ── GGUF ────────────────────────────────────────────────────────────────────────────────

    private func ggufString(_ s: String) -> [UInt8] { le(UInt64(s.utf8.count)) + Array(s.utf8) }

    /// A GGUF with the given key/values and tensor infos (as bytes), padded to `size`.
    private func gguf(_ name: String, keys: Int, kv: [UInt8], tensors: Int = 0, infos: [UInt8] = [], size: Int = 256) throws -> String {
        var b = Array("GGUF".utf8) + le(UInt32(3)) + le(UInt64(tensors)) + le(UInt64(keys)) + kv + infos
        if b.count < size { b += [UInt8](repeating: 0, count: size - b.count) }
        return try file(name, b)
    }

    private func assertGGUFRefused(_ path: String, _ fragment: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try Safetensors(path: path), file: file, line: line) { e in
            XCTAssertTrue("\(e)".contains(fragment), "\(e)", file: file, line: line)
        }
    }

    func testGGUFArrayOfArraysIsRefused() throws {
        let inner = le(UInt32(5)) + le(UInt64(1)) + le(Int32(7))
        let kv = ggufString("nested") + le(UInt32(9)) + le(UInt32(9)) + le(UInt64(1)) + inner
        assertGGUFRefused(try gguf("nested.gguf", keys: 1, kv: kv), "array of arrays")
        // An array of scalars still reads.
        let flat = ggufString("flat") + le(UInt32(9)) + inner
        XCTAssertEqual(try Safetensors(path: try gguf("flat.gguf", keys: 1, kv: flat)).metadata["flat"], "[1 × i32]")
    }

    func testGGUFHugeCountsAreRefusedBeforeReading() throws {
        // A string of 2^40 bytes, an array of thirty u64 with room for twenty-five, a huge alignment.
        assertGGUFRefused(try gguf("string.gguf", keys: 1, kv: le(UInt64(1) << 40)), "larger than the file")
        let array = ggufString("a") + le(UInt32(9)) + le(UInt32(10)) + le(UInt64(30))
        assertGGUFRefused(try gguf("array.gguf", keys: 1, kv: array), "larger than the file")
        let alignment = ggufString("general.alignment") + le(UInt32(10)) + le(UInt64(1) << 62)
        assertGGUFRefused(try gguf("align.gguf", keys: 1, kv: alignment), "general.alignment")
    }

    func testGGUFOverflowingDimensionsAreRefused() throws {
        // Four dimensions, each within the file's 70,000 bytes, whose product overflows an `Int`.
        var info = ggufString("w") + le(UInt32(4))
        for _ in 0..<4 { info += le(UInt64(69_000)) }
        info += le(UInt32(0)) + le(UInt64(0))
        assertGGUFRefused(try gguf("dims.gguf", keys: 0, kv: [], tensors: 1, infos: info, size: 70_000), "overflow")
    }

    // ── OrderedJSON ─────────────────────────────────────────────────────────────────────────

    func testOrderedJSONDepthIsBounded() throws {
        let deep = OrderedJSON.maximumDepth
        XCTAssertNoThrow(try OrderedJSON.parse(Data((String(repeating: "[", count: deep) + String(repeating: "]", count: deep)).utf8)))
        let n = 100_000
        XCTAssertThrowsError(try OrderedJSON.parse(Data((String(repeating: "[", count: n) + String(repeating: "]", count: n)).utf8))) {
            XCTAssertTrue("\($0)".contains("nested deeper"), "\($0)")
        }
        XCTAssertThrowsError(try OrderedJSON.parse(Data(String(repeating: #"{"a":"#, count: deep + 1).utf8)))
    }

    func testOrderedJSONBrokenSurrogatePairIsRefused() throws {
        // Built from pieces: an escape written whole in this file is one an editor may decode.
        func escaped(_ halves: String...) -> Data { Data(("\"" + halves.map { "\\" + "u" + $0 }.joined() + "\"").utf8) }
        XCTAssertEqual(try OrderedJSON.parse(escaped("d83d", "de00")).text, "\u{1F600}")
        XCTAssertThrowsError(try OrderedJSON.parse(escaped("D800", "0041"))) {
            XCTAssertTrue("\($0)".contains("low half"), "\($0)")
        }
        // A lone high half, not followed by an escape, stays what it was: the replacement character.
        var lone = escaped("d800"); lone.insert(UInt8(ascii: "x"), at: lone.count - 1)
        XCTAssertEqual(try OrderedJSON.parse(lone).text, "\u{FFFD}x")
    }

    // ── a recipe on a 1-D tensor ────────────────────────────────────────────────────────────

    final class Shapes: TensorCatalog {
        let shapes: [String: [Int]]
        init(_ s: [String: [Int]]) { shapes = s }
        var names: [String] { Array(shapes.keys) }
        func shape(_ name: String) -> [Int]? { shapes[name] }
        func dtype(_ name: String) -> String? { "BF16" }
        func read(_ name: String) throws -> [Float] { [] }
    }

    func testFlux2SplitRefusesAVectorUnderAMatrixName() throws {
        for name in ["single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight", "single_transformer_blocks.0.attn.to_out.weight"] {
            XCTAssertThrowsError(try Recipes.flux2Split(Shapes(["p": [12]]), "p", name)) {
                XCTAssertTrue("\($0)".contains("a matrix expected"), "\($0)")
            }
        }
        // The same names as matrices still split.
        XCTAssertEqual(try Recipes.flux2Split(Shapes(["p": [4 * 3 + 2 * 6, 4]]), "p",
                                              "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight").count, 5)
        XCTAssertEqual(try Recipes.flux2Split(Shapes(["p": [4, 10]]), "p", "single_transformer_blocks.0.attn.to_out.weight").count, 2)
    }

    func testFusedRowsRefuseAScalarOrAnUnevenSplit() throws {
        // Split while the names are normalized, before any shape is checked: a scalar under a qkv name
        // was indexed past its end (a crash, not a refusal).
        for shape in [[], [0, 4], [7, 4]] {
            XCTAssertThrowsError(try Recipes.rows(Shapes(["p": shape]), "p", part: 0, outOf: 3)) {
                XCTAssertTrue("\($0)".contains("row blocks"), "\($0)")
            }
        }
        XCTAssertEqual(try Recipes.rows(Shapes(["p": [6, 4]]), "p", part: 2, outOf: 3).shape, [2, 4])
        XCTAssertThrowsError(try Recipes.normalize(Shapes(["layers.0.attention.qkv.weight": []]), family: .zImage))
    }

    // ── the download ────────────────────────────────────────────────────────────────────────

    func testWeightFilesRequireASHA256() throws {
        let sha = String(repeating: "a", count: 64), other = String(repeating: "b", count: 64)
        // A pin the Hub does not announce: checked against the pin alone, not "mismatch".
        XCTAssertEqual(try Downloader.expectedSHA(path: "m.safetensors", pinned: sha, announced: nil, status: 302, file: "m"), sha)
        XCTAssertEqual(try Downloader.expectedSHA(path: "m.safetensors", pinned: sha, announced: sha, status: 302, file: "m"), sha)
        XCTAssertThrowsError(try Downloader.expectedSHA(path: "m.gguf", pinned: sha, announced: other, status: 302, file: "m")) {
            guard case EngineError.downloadCorrupt = $0 else { return XCTFail("\($0)") }
        }
        XCTAssertEqual(try Downloader.expectedSHA(path: "m.safetensors", pinned: nil, announced: other, status: 302, file: "m"), other)
        for path in ["vae.safetensors", "dit.gguf"] {
            XCTAssertThrowsError(try Downloader.expectedSHA(path: path, pinned: nil, announced: nil, status: 200, file: path)) {
                guard case EngineError.downloadRefused(path, 200) = $0 else { return XCTFail("\($0)") }
            }
        }
        // A config or a tokenizer is no LFS object: no sha256 to ask for.
        XCTAssertNil(try Downloader.expectedSHA(path: "config.json", pinned: nil, announced: nil, status: 200, file: "c"))
    }

    func testRangeWantsA206OfExactlyTheBytesAsked() throws {
        let url = URL(string: "https://huggingface.co/x/resolve/r/m.safetensors")!
        func response(_ status: Int) -> URLResponse { HTTPURLResponse(url: url, statusCode: status, httpVersion: nil, headerFields: nil)! }
        XCTAssertNoThrow(try Downloader.checkRange(response(206), Data(count: 8), 0, 7, file: "m"))
        XCTAssertThrowsError(try Downloader.checkRange(response(200), Data(count: 8), 0, 7, file: "m")) {
            guard case EngineError.downloadRefused(_, 200) = $0 else { return XCTFail("\($0)") }
        }
        XCTAssertThrowsError(try Downloader.checkRange(response(206), Data(count: 5), 0, 7, file: "m")) {
            guard case EngineError.downloadInterrupted = $0 else { return XCTFail("\($0)") }
        }
    }

    // ── alpha_pattern ───────────────────────────────────────────────────────────────────────

    private func peft(_ pattern: String) throws -> ForgeLoRA.PEFTScale? {
        let body = OrderedJSON.object([.init("lora_alpha", 4), .init("r", 2),
                                       .init("alpha_pattern", .object([.init(pattern, 1)]))]).jsonText
        return try ForgeLoRA.PEFTScale.read(["lora_adapter_metadata": body])
    }

    func testAlphaPatternIsBoundedAndLiteralFirst() throws {
        XCTAssertThrowsError(try peft(String(repeating: "a", count: ForgeLoRA.PEFTScale.maximumPattern + 1))) {
            XCTAssertTrue("\($0)".contains("characters"), "\($0)")
        }
        XCTAssertThrowsError(try peft("(a+)+b")) { XCTAssertTrue("\($0)".contains("repeats a group"), "\($0)") }
        // Literal, suffix, and a real regular expression still match as PEFT would.
        let literal = try XCTUnwrap(try peft("attn.to_q"))
        XCTAssertEqual(literal.scale("blocks.0.attn.to_q", r: 2), 0.5)
        XCTAssertEqual(literal.scale("attn.to_q", r: 2), 0.5)
        XCTAssertEqual(literal.scale("blocks.0.attn.to_qk", r: 2), 2)
        let regex = try XCTUnwrap(try peft(".*to_[qk]"))
        XCTAssertEqual(regex.scale("blocks.0.attn.to_k", r: 2), 0.5)
        XCTAssertEqual(regex.scale("blocks.0.attn.to_v", r: 2), 2)
    }

    // ── the library ─────────────────────────────────────────────────────────────────────────

    func testOrphanPartialsAreCountedAndPurgedButNotOneBeingWritten() throws {
        let library = Library(root: root)
        let fm = FileManager.default
        let store = root.appendingPathComponent("store").path
        for folder in [store, store + "/importes", store + "/composants/z-image"] {
            try fm.createDirectory(atPath: folder, withIntermediateDirectories: true)
        }
        let orphans = [store + "/z-image-dit.silicon.partiel", store + "/importes/mine.silicon.partiel",
                       store + "/composants/z-image/vae.safetensors.partiel"]
        let busy = store + "/qwen.lora.silicon.partiel"
        for p in orphans + [busy] { try Data(count: 1000).write(to: URL(fileURLWithPath: p)) }
        try Data(count: 10).write(to: URL(fileURLWithPath: store + "/kept.silicon"))

        XCTAssertEqual(Set(library.partials()), Set((orphans + [busy]).map { URL(fileURLWithPath: $0).standardizedFileURL.path }))
        let o = library.occupancy()
        XCTAssertEqual(o.downloads, 4000)
        // The component's `.partiel` is counted with the downloads, not twice.
        XCTAssertEqual(o.families.first { $0.family == .zImage }?.components, 0)

        let held = Library.holdPartial(busy)
        defer { close(held) }
        try library.emptyDownloads()
        for p in orphans { XCTAssertFalse(fm.fileExists(atPath: p), p) }
        XCTAssertTrue(fm.fileExists(atPath: busy), "a .partiel being written is left alone")
        XCTAssertTrue(fm.fileExists(atPath: store + "/kept.silicon"))
    }

    func testImportNeverOverwritesAnEarlierOne() throws {
        let folder = root.path
        XCTAssertEqual(ModelImport.freePath("mine", suffix: ".silicon", in: folder), folder + "/mine.silicon")
        try Data().write(to: URL(fileURLWithPath: folder + "/mine.silicon"))
        try Data().write(to: URL(fileURLWithPath: folder + "/mine-2.silicon"))
        XCTAssertEqual(ModelImport.freePath("mine", suffix: ".silicon", in: folder), folder + "/mine-3.silicon")
        XCTAssertEqual(ModelImport.slug("日本語"), "", "a name wholly outside ASCII: the LoRA falls back to \"lora\"")
    }

    /// The chunks of a large download: contiguous, covering, never empty, at most 64 MB, bounds by
    /// endpoints — whatever the size (a prime, one byte over a chunk, the 9.35 GB encoder).
    func testDownloadChunksCoverTheFile() {
        for size in [1, 64 << 20, (64 << 20) + 1, 1_000_000_007, 9_349_769_248] {
            let c = Downloader.chunks(size: size)
            XCTAssertEqual(c.first?.lowerBound, 0)
            XCTAssertEqual(c.last?.upperBound, size)
            for (a, b) in zip(c, c.dropFirst()) { XCTAssertEqual(a.upperBound, b.lowerBound) }
            XCTAssertTrue(c.allSatisfy { !$0.isEmpty && $0.count <= 64 << 20 }, "\(size)")
            XCTAssertEqual(c.count, (size + (64 << 20) - 1) / (64 << 20))
        }
    }

    /// A resumed download keeps only the chunks of the same file: another size or sha256, nothing.
    func testDownloadResumesOnlyTheSameFile() {
        let sha = String(repeating: "a", count: 64)
        let ledger = "100 \(sha)\n0\n3\n7\n"
        XCTAssertEqual(Downloader.resumedChunks(ledger: ledger, size: 100, sha: sha), [0, 3, 7])
        XCTAssertEqual(Downloader.resumedChunks(ledger: ledger, size: 101, sha: sha), [])
        XCTAssertEqual(Downloader.resumedChunks(ledger: ledger, size: 100, sha: String(repeating: "b", count: 64)), [])
        XCTAssertEqual(Downloader.resumedChunks(ledger: "100 -\n2\n", size: 100, sha: nil), [2])
        XCTAssertEqual(Downloader.resumedChunks(ledger: "", size: 100, sha: nil), [])
        // A line a crash cut ("1" of "12") is not a chunk; nor is a head without its newline.
        XCTAssertEqual(Downloader.resumedChunks(ledger: "100 -\n2\n1", size: 100, sha: nil), [2])
        XCTAssertEqual(Downloader.resumedChunks(ledger: "100 -", size: 100, sha: nil), [])
    }
}
