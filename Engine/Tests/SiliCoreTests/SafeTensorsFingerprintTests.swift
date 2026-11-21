import XCTest
@testable import SiliCore

/// **A file's fingerprint names its weights**: the same shapes and other weights, another name; the same bytes, the same.
final class SafeTensorsFingerprintTests: XCTestCase {
    /// A safetensors file of two f32 tensors, `[4, 300]` and `[7]`, its values from `values`.
    private func file(_ name: String, _ values: (Int) -> Float) throws -> SafeTensors {
        let header = #"{"a":{"dtype":"F32","shape":[4,300],"data_offsets":[0,4800]},"b":{"dtype":"F32","shape":[7],"data_offsets":[4800,4828]}}"#
        var data = Data()
        withUnsafeBytes(of: UInt64(header.utf8.count).littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: header.utf8)
        for i in 0..<1207 { withUnsafeBytes(of: values(i)) { data.append(contentsOf: $0) } }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fingerprint-\(name)-\(ProcessInfo.processInfo.processIdentifier).safetensors")
        try data.write(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return try SafeTensors(url.path)
    }

    func testOtherWeightsOfTheSameShapesAnotherName() throws {
        let base = try file("base") { Float($0) * 0.5 }
        let same = try file("same") { Float($0) * 0.5 }
        let tuned = try file("tuned") { Float($0) * 0.5 + 0.01 }
        XCTAssertEqual(base.tensors.mapValues(\.shape), tuned.tensors.mapValues(\.shape))
        XCTAssertEqual(base.fingerprint, same.fingerprint)
        XCTAssertNotEqual(base.fingerprint, tuned.fingerprint)
    }

    /// A change in one tensor alone, a sample's worth deep inside it, is seen.
    func testOneTensorChangedIsSeen() throws {
        let base = try file("one-base") { Float($0) }
        let tuned = try file("one-tuned") { $0 >= 1200 ? -Float($0) : Float($0) }
        XCTAssertNotEqual(base.fingerprint, tuned.fingerprint)
    }
}
