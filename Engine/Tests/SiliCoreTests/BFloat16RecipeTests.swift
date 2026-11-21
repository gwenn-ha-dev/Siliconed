import Foundation
import Metal
import Testing
@testable import SiliCore

/// The bf16 path's recipe on a file written here: a BF16 tensor as it lies, an F32 one rounded, parts joined along
/// their first axis one after the other.
@Test func theBFloat16RecipeJoinsAndRounds() throws {
    let a: [UInt16] = [0x3F80, 0x4000, 0x4040, 0x4080]         // [2, 2] bf16: 1 2 3 4
    let b: [Float] = [5, 6, 1 + 0x1p-8, 1 + 3 * 0x1p-8, 7, 8]  // [3, 2] f32
    let header = #"{"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[3,2],"data_offsets":[8,32]}}"#
    var bytes = withUnsafeBytes(of: UInt64(header.utf8.count).littleEndian) { Array($0) } + Array(header.utf8)
    bytes += a.withUnsafeBytes { Array($0) } + b.withUnsafeBytes { Array($0) }
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("bf16-recipe-\(getpid()).safetensors")
    try Data(bytes).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    let file = try SafeTensors(url.path)
    let entries: [Weights.Entry] = [("ab", "a", .bfloat16(joined: ["b"])), ("b", "b", .bfloat16(joined: []))]
    let layout = try Weights.layout(entries, in: file)
    #expect(layout.map(\.shape) == [[5, 2], [3, 2]])
    #expect(layout.map(\.bytes) == [20, 12])
    #expect(try Weights.staging(entries, in: file) == 24)
    guard let device = MTLCreateSystemDefaultDevice() else { return }
    let arena = try Arena(device, bytes: try Weights.size(entries, in: file))
    let placed = try Weights.load(entries, from: file, into: arena)
    let ab = arena.pointer(placed["ab"]!.offset).assumingMemoryBound(to: UInt16.self)
    #expect((0..<10).map { BFloat16Bits.widened(ab[$0]) } == [1, 2, 3, 4, 5, 6, 1, 1 + 0x1p-6, 7, 8])
}
