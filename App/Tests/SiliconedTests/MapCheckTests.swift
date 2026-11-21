import Foundation
import Testing
@testable import Siliconed
@testable import Verification

/// `check` names the map it judges and skips one that is not the golden's variant.
struct MapCheckTests {
    private func summary(_ dtypes: [String: Int]) -> MapSummary { MapSummary(path: "/tmp/x.safetensors", bytes: 3_900_000_000, dtypes: dtypes) }

    @Test func variantFollowsTheTensorTypes() {
        #expect(summary(["bfloat16": 300, "float32": 4]).variant == .standard)
        #expect(summary(["bfloat16": 10, "float16": 2]).variant == .standard)
        #expect(summary(["int8": 200, "bfloat16": 20]).variant == .eightBit)
    }

    @Test func quantizedMapIsRefusedAgainstTheStandardGolden() throws {
        let why = try #require(summary(["int8": 200, "bfloat16": 20]).mismatch(golden: .standard))
        #expect(why.contains("8-bit") && why.contains("drop --map"))
        #expect(summary(["bfloat16": 300]).mismatch(golden: .standard) == nil)
    }

    @Test func aGoldenOfAnotherVariantAcceptsItsOwn() {
        #expect(summary(["int8": 200]).mismatch(golden: .eightBit) == nil)
        #expect(summary(["bfloat16": 300]).mismatch(golden: .eightBit) != nil)
    }

    @Test func lineNamesPathSizeAndTypes() {
        let line = summary(["bfloat16": 300, "float32": 4]).line
        #expect(line == "map /tmp/x.safetensors · 3.9 GB · bfloat16 ×300, float32 ×4 (standard weights)")
        #expect(summary(["bfloat16": 1]).line.contains("· bfloat16 (standard"))
    }

    @Test func headerOfARealFileIsRead() throws {
        let path = NSTemporaryDirectory() + "mapcheck-\(UUID().uuidString).safetensors"
        defer { try? FileManager.default.removeItem(atPath: path) }
        let json = #"{"a":{"dtype":"I8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[2,6]},"a.comfy_quant":{"dtype":"U8","shape":[2],"data_offsets":[6,8]}}"#
        var data = Data()
        withUnsafeBytes(of: UInt64(json.utf8.count).littleEndian) { data.append(contentsOf: $0) }
        data.append(Data(json.utf8)); data.append(Data(count: 8))
        try data.write(to: URL(fileURLWithPath: path))
        let s = try MapSummary(path: path)
        #expect(s.dtypes == ["int8": 1, "bfloat16": 1] && s.variant == .eightBit)
        #expect(throws: Skip.self) { try MapCheck.use(path) }
    }
}
