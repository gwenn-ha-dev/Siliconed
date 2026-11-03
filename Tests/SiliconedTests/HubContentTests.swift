import Foundation
import XCTest
@testable import Siliconed

/// **The Hub's examination reads the texts the import reads** (`HubImport.contentRanges`, `sparse`) — a
/// ComfyUI int8 file written here, examined whole and over a sparse copy: no weight, no network.
final class HubContentTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-hub-content-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    private struct Piece { let name: String; let dtype: String; let shape: [Int]; let bytes: [UInt8] }

    /// The header (8 bytes of length, the JSON padded to 8) and the whole file.
    private func safetensors(_ pieces: [Piece]) -> (head: Data, file: Data) {
        var table: [OrderedJSON.Pair] = [], data = Data(), at = 0
        for p in pieces {
            table.append(.init(p.name, .object([.init("dtype", .string(p.dtype)), .init("shape", .list(p.shape.map { .integer($0) })),
                                                .init("data_offsets", .list([.integer(at), .integer(at + p.bytes.count)]))])))
            data.append(contentsOf: p.bytes); at += p.bytes.count
        }
        var json = Array(OrderedJSON.object(table).jsonText.utf8)
        while json.count % 8 != 0 { json.append(0x20) }
        var head = withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) }
        head.append(contentsOf: json)
        return (head, head + data)
    }

    private func text(_ name: String, _ s: String) -> Piece { Piece(name: name, dtype: "U8", shape: [s.utf8.count], bytes: Array(s.utf8)) }

    /// A Qwen-Image-2.1 DiT by its names, one layer int8 per row as ComfyUI writes it, `quant` its text.
    private func comfyQwen(quant: String) -> [Piece] {
        let scale = withUnsafeBytes(of: Float(0.01)) { Array($0) }
        return [Piece(name: "transformer_blocks.0.img_mlp.gate_layer.weight", dtype: "I8", shape: [2, 4], bytes: [1, 2, 3, 4, 5, 6, 7, 8]),
                Piece(name: "transformer_blocks.0.img_mlp.gate_layer.weight_scale", dtype: "F32", shape: [2, 1], bytes: scale + scale),
                Piece(name: "txt_in.text_norm.weight", dtype: "BF16", shape: [2], bytes: [0x80, 0x3F, 0x80, 0x3F]),
                text("transformer_blocks.0.img_mlp.gate_layer.comfy_quant", quant),
                text("transformer_blocks.0.attn.comfy_attention.config", #"{"attention": "comfy_kitchen_int8"}"#)]
    }

    /// What the Hub's examination makes of `pieces`: over a sparse copy, the ranges fetched from the whole file.
    private func examined(_ pieces: [Piece], fetching: Bool = true) throws -> (whole: Result<Family, Error>, sparse: Result<Family, Error>) {
        let (head, file) = safetensors(pieces)
        let whole = root.appendingPathComponent("whole-\(UUID()).safetensors").path
        try file.write(to: URL(fileURLWithPath: whole))
        let sparse = root.appendingPathComponent("sparse-\(UUID()).safetensors").path
        let ranges = try XCTUnwrap(HubImport.contentRanges(header: head, size: file.count))
        try HubImport.sparse(head, size: file.count, contents: fetching ? ranges : [], at: sparse) { file.subdata(in: $0) }
        let library = Library(root: root.appendingPathComponent("lib"))
        func verdict(_ path: String) -> Result<Family, Error> { Result { try ModelImport.examine(path, in: library).1 } }
        return (verdict(whole), verdict(sparse))
    }

    private func detail(_ r: Result<Family, Error>) -> String? {
        if case .failure(EngineError.importRefused(_, let d)) = r { return d }
        return nil
    }

    func testComfyTextsAreFetchedAndTheVerdictIsTheWholeFiles() throws {
        let (whole, sparse) = try examined(comfyQwen(quant: #"{"format": "int8_tensorwise"}"#))
        XCTAssertEqual(try whole.get(), .qwenImage21)
        XCTAssertEqual(try sparse.get(), .qwenImage21)
    }

    /// The defect: the same file over zeros read its texts as « not JSON ».
    func testWithoutTheTextsTheFileWasRefused() throws {
        let (_, sparse) = try examined(comfyQwen(quant: #"{"format": "int8_tensorwise"}"#), fetching: false)
        XCTAssertTrue(detail(sparse)?.contains("not JSON") == true, "\(sparse)")
    }

    /// A wrong text stays refused, with the import's own words.
    func testAWrongTextIsRefusedAsTheImportWould() throws {
        for quant in [#"{"format": "nvfp4"}"#, "not a text"] {
            let (whole, sparse) = try examined(comfyQwen(quant: quant))
            XCTAssertNotNil(detail(whole), quant)
            XCTAssertEqual(detail(sparse), detail(whole), quant)
        }
    }

    func testRangesAreTheTextsMergedAndAbsolute() throws {
        let pieces = comfyQwen(quant: #"{"format": "int8_tensorwise"}"#)
        let (head, file) = safetensors(pieces)
        let ranges = try XCTUnwrap(HubImport.contentRanges(header: head, size: file.count))
        // The two texts are written last, one after the other: one request, from the first to the end.
        let first = head.count + pieces.dropLast(2).reduce(0) { $0 + $1.bytes.count }
        XCTAssertEqual(ranges, [first..<file.count])
        // Apart by more than `gap`: two requests; more than `requests`: none, judged once downloaded.
        XCTAssertEqual(HubImport.contentRanges(header: head, size: file.count, gap: 0)?.count, 1)
        XCTAssertNil(HubImport.contentRanges(header: head, size: file.count, gap: 0, requests: 0))
        XCTAssertNil(HubImport.contentRanges(header: head, size: file.count, bytes: 10))
    }

    func testScatteredTextsAreLeftToTheImport() throws {
        var pieces: [Piece] = []
        for i in 0..<20 {
            pieces.append(Piece(name: "layers.\(i).weight", dtype: "I8", shape: [64], bytes: Array(repeating: 1, count: 64)))
            pieces.append(text("layers.\(i).comfy_quant", #"{"format": "int8_tensorwise"}"#))
        }
        let (head, file) = safetensors(pieces)
        XCTAssertEqual(HubImport.contentRanges(header: head, size: file.count)?.count, 1, "close together: one request")
        XCTAssertNil(HubImport.contentRanges(header: head, size: file.count, gap: 16))
        // Nothing to read, nothing fetched; an offset past the file is left to the reader.
        XCTAssertEqual(HubImport.contentRanges(header: safetensors([pieces[0]]).head, size: 1 << 30), [])
        XCTAssertEqual(HubImport.contentRanges(header: head, size: head.count + 10), [])
    }
}
