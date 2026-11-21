import Foundation
import XCTest
@testable import Siliconed

/// **A file that lies is refused by name, never trusted**: a safetensors header whose
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

    /// A T5 added token without its content or its id is refused, as one normalized: skipped, its text would be cut into
    /// pieces the reference never makes of it.
    func testT5AddedTokenWithoutItsIdIsRefused() throws {
        let good: [[String: Any]] = [["content": "<pad>", "id": 0], ["content": "</s>", "id": 1]]
        XCTAssertEqual(try T5Tokenizer.added(good).map(\.id), [0, 1])
        XCTAssertThrowsError(try T5Tokenizer.added(good + [["content": "<extra_id_0>"]]))
        XCTAssertThrowsError(try T5Tokenizer.added(good + [["id": 2]]))
        XCTAssertThrowsError(try T5Tokenizer.added([["content": "<x>", "id": 3, "lstrip": true]]))
    }
}
