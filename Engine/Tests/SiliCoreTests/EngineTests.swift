import Foundation
import Metal
import Testing
@testable import SiliCore

// MARK: - The frontier

/// The engine knows no word of a workload: a model writes its functions with the engine's tools, the engine holds the
/// memory and the units.
@Test func engineKnowsNoWorkload() throws {
    let forbidden = #"\b(dit|diffusion|sigma|σ|timesteps?|prompts?|latents?|denois\w*|vae|lora|adapters?|images?|pixels?|videos?|texts?|tokens?|llm|captions?|samplers?|schedulers?|cfg|guidance|qwen\w*|krea\w*|anima|moss|flux|z-image|siliconed|velocit\w*|decoders?|z's|z at|z compact\w*)\b"#
    let regex = try NSRegularExpression(pattern: forbidden, options: [.caseInsensitive])
    let sources = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../../Sources/SiliCore").standardizedFileURL
    let files = try FileManager.default.contentsOfDirectory(at: sources, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "swift" }
    #expect(!files.isEmpty)
    var hits: [String] = []
    for f in files {
        let text = try String(contentsOf: f, encoding: .utf8)
        for (i, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let s = String(line)
            if let m = regex.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) {
                hits.append("\(f.lastPathComponent):\(i + 1): \((s as NSString).substring(with: m.range))")
            }
        }
    }
    #expect(hits.isEmpty, "\(hits)")
}

// MARK: - The arena

/// A tensor takes its pages: a page of 16 KiB on Apple silicon, every tensor starting on one.
@Test func aTensorTakesItsPages() {
    #expect(Arena.page == 16384)
    #expect(Arena.aligned(0) == 0)
    #expect(Arena.aligned(1) == 16384)
    #expect(Arena.aligned(16384) == 16384)
    #expect(Arena.aligned(16385) == 32768)
    #expect(Arena.allocation(bytes: 0) == 16384)
    #expect(Arena.allocation(bytes: 3 * 16384 + 7) == 4 * 16384)
    #expect(Pages.capacity(for: 0) == 16384 && Pages.capacity(for: 16385) == 32768)
}

private let device = MTLCreateSystemDefaultDevice()

/// Tensors placed one after the other, each on a page, the heap the sum of what they take to the byte; what is taken
/// apart (a buffer of its own, a surface) is counted apart, past the heap's offsets.
@Test(.enabled(if: device != nil)) func theArenaPlacesEachTensorOnAPage() throws {
    let sizes = [1, 16384, 16385, 100, 64 * 1024]
    let bytes = sizes.reduce(0) { $0 + Arena.aligned($1) }
    let apart = Arena.aligned(1000) + Arena.aligned(3 * 64 * 2)
    let arena = try Arena(device!, bytes: bytes + apart, apart: apart)
    #expect(arena.size == Arena.allocation(bytes: bytes))
    var expected = 0
    for n in sizes {
        let offset = arena.take(n)
        #expect(offset == expected && offset % Arena.page == 0, "a tensor of \(n) B at \(offset), expected \(expected)")
        #expect(arena.buffer(offset).length == Arena.aligned(n))
        expected += Arena.aligned(n)
    }
    #expect(arena.used == expected - Arena.aligned(sizes.last!) + sizes.last!)

    let heaps = arena.allocated
    let exit = arena.take(apart: 1000)
    #expect(exit == arena.size && arena.allocated - heaps == Arena.aligned(1000) && arena.buffer(exit).heap == nil)
    let s = arena.take(surface: [3, 64], element: 2)
    let surface = try #require(arena.surface(s))
    #expect(s == arena.size + Arena.aligned(1000) && arena.allocated - heaps == apart)
    #expect(surface.width == 64 && surface.height == 3 && surface.bytesPerRow == 128)
    #expect(arena.surface(exit) == nil)
    // The GPU's buffer over the surface is its bytes.
    arena.pointer(s).storeBytes(of: 0x3c00, as: UInt16.self)
    surface.lock(options: .readOnly, seed: nil)
    #expect(surface.baseAddress.load(as: UInt16.self) == 0x3c00)
    surface.unlock(options: .readOnly, seed: nil)
}

// MARK: - Weights

/// A safetensors file of the given tensors, under `.build/tests/`.
func safetensors(_ name: String, _ tensors: [(String, String, [Int], [UInt8])]) throws -> String {
    let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("../../.build/tests").standardizedFileURL
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    var header: [String: Any] = [:], data: [UInt8] = [], at = 0
    for (key, dtype, shape, bytes) in tensors {
        header[key] = ["dtype": dtype, "shape": shape, "data_offsets": [at, at + bytes.count]]
        data += bytes; at += bytes.count
    }
    let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var file = withUnsafeBytes(of: UInt64(json.count).littleEndian, Array.init) + [UInt8](json)
    file += data
    let url = folder.appendingPathComponent(name)
    try Data(file).write(to: url)
    return url.path
}

private func bytes<T>(_ values: [T]) -> [UInt8] { values.withUnsafeBytes { Array($0) } }
private func bf16(_ x: Float) -> UInt16 { UInt16(x.bitPattern >> 16) }

extension Weights {
    /// `entries` read from `file` into `arena` as a stage reads them (`Weights.place`, then `Weights.read` through a
    /// staging), the staging in `Pages` of its own.
    static func load(_ entries: [Entry], from file: SafeTensors, into arena: Arena, codes: CodesLayout = .inOut) throws -> [String: Placed] {
        let placed = place(try layout(entries, in: file, codes: codes), in: arena)
        let staging = Pages(bytes: try Self.staging(entries, in: file, codes: codes))
        try withExtendedLifetime(staging) {
            try read(entries, from: file, into: placed, arena: arena, staging: staging.base, codes: codes)
        }
        return placed
    }
}

/// **A shared form's product, by its weights**: the arguments its function declares are the slots its entry lays in the
/// arena, a leading 1 before each — the codes `[in, out]` (`codes`) or as the file lays them `[out, in]` (`integer`, the
/// plan's layout `outIn`), their scales `[1, 1, out]`; the Standard's weight `<key>.weight` rounded to f16 `[in, out]`.
@Test func aSharedProductsArgumentsAreItsEntrysSlots() throws {
    let (out, inn) = (96, 64)
    let path = try safetensors("shared-weights.safetensors", [
        ("p.weight", "I8", [out, inn], [UInt8](repeating: 1, count: out * inn)),
        ("p.weight_scale", "F32", [out, 1], bytes([Float](repeating: 0.01, count: out))),
        ("q.weight", "BF16", [out, inn], bytes([UInt16](repeating: bf16(0.5), count: out * inn))),
    ])
    let file = try SafeTensors(path)
    #expect(SharedWeights.codes.layout == .inOut && SharedWeights.integer.layout == .outIn && SharedWeights.half.layout == .inOut)
    #expect(SharedWeights.codes.isCodes && SharedWeights.integer.isCodes && !SharedWeights.half.isCodes)
    for (weights, key, shape) in [(SharedWeights.codes, "p", [1, inn, out]), (.integer, "p", [1, out, inn]), (.half, "q", [1, inn, out])] {
        let entry = weights.entry("w", key: key)
        #expect(entry.name == "w" && entry.key == (weights.isCodes ? key : key + ".weight") && entry.recipe == (weights.isCodes ? .codes : .halfTransposed))
        let arguments = weights.arguments("w", inn: inn, out: out)
        let slots = try Weights.layout([entry], in: file, codes: weights.layout)
        #expect(arguments.map(\.name) == slots.map(\.name), "\(weights)")
        #expect(arguments.map(\.shape) == slots.map { $0.shape.count == 1 ? [1, 1] + $0.shape : [1] + $0.shape }, "\(weights)")
        #expect(arguments[0].shape == shape, "\(weights)")
        #expect(arguments.map(\.element) == (weights.isCodes ? [.si8, .f16] : [.f16]), "\(weights)")
        #expect(slots.map(\.type) == (weights.isCodes ? [.int8, .float16] : [.float16]), "\(weights)")
    }
}

/// What a file's weights become in the arena by their recipes, from its header alone: codes transposed with their
/// scales in f16, reals rounded to f16 or kept in f32, transposed or not, a target of one f16.
@Test func recipesSayWhatTheWeightsTakeInTheArena() throws {
    let (out, inn) = (48, 80)
    let path = try safetensors("recipes.safetensors", [
        ("a.weight", "I8", [out, inn], [UInt8](repeating: 1, count: out * inn)),
        ("a.weight_scale", "F32", [out, 1], bytes([Float](repeating: 0.01, count: out))),
        ("gain", "BF16", [inn], bytes([UInt16](repeating: bf16(1.5), count: inn))),
        ("bias", "F32", [out], bytes([Float](repeating: 0.25, count: out))),
        ("w", "F32", [out, inn], bytes((0..<out * inn).map { Float($0) })),
    ])
    let file = try SafeTensors(path)
    let entries: [Weights.Entry] = [("a", "a", .codes), ("gain", "gain", .half), ("gain32", "gain", .float), ("bias", "bias", .float),
                                    ("w", "w", .halfTransposed), ("w32", "w", .floatTransposed), ("e0", "", .target(["a"]))]
    let layout = try Weights.layout(entries, in: file)
    let slots = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    #expect(layout.map(\.name) == ["a", "a_scale", "gain", "gain32", "bias", "w", "w32", "e0"])
    #expect(slots["a"]!.shape == [inn, out] && slots["a"]!.type == .int8 && slots["a"]!.bytes == out * inn)
    #expect(slots["a_scale"]!.shape == [out] && slots["a_scale"]!.type == .float16 && slots["a_scale"]!.bytes == 2 * out)
    #expect(slots["gain"]!.shape == [inn] && slots["gain"]!.type == .float16 && slots["gain"]!.bytes == 2 * inn)
    #expect(slots["gain32"]!.type == .float32 && slots["gain32"]!.bytes == 4 * inn)
    #expect(slots["w"]!.shape == [inn, out] && slots["w"]!.bytes == 2 * out * inn)
    #expect(slots["w32"]!.shape == [inn, out] && slots["w32"]!.type == .float32 && slots["w32"]!.bytes == 4 * out * inn)
    #expect(slots["e0"]!.shape == [1, 1, 1] && slots["e0"]!.bytes == 2)
    #expect(try Weights.size(entries, in: file) == layout.reduce(0) { $0 + Arena.aligned($1.bytes) })
    // The staging: the largest tensor converted; an f32 read as it is needs none.
    #expect(try Weights.staging(entries, in: file) == 4 * out * inn)
    #expect(try Weights.staging([("bias", "bias", .float)], in: file) == 0)
}

/// A file read under a prefix: its keys under it alone, named without it, their bytes where they were.
@Test func aFileReadUnderAPrefixNamesItsKeysWithoutIt() throws {
    let path = try safetensors("under.safetensors", [
        ("model.diffusion_model.a.weight", "F32", [2], bytes([Float(1), 2])), ("first_stage_model.b", "F32", [1], bytes([Float(3)])),
    ])
    let whole = try SafeTensors(path), dit = try SafeTensors(path, under: "model.diffusion_model.")
    #expect(Set(dit.tensors.keys) == ["a.weight"])
    #expect(dit.tensors["a.weight"]!.start == whole.tensors["model.diffusion_model.a.weight"]!.start)
    #expect(Set(try SafeTensors(path, under: "").tensors.keys) == Set(whole.tensors.keys))
    // The ring's reader reopens the file (past the cache): under the same prefix, or a weight kept under it is not found.
    #expect(Set(try dit.reopened(uncached: true).tensors.keys) == ["a.weight"])
}

/// The weights read into the arena by their recipes: the codes transposed, the reals rounded once, the target
/// computed from the scales once read.
@Test(.enabled(if: device != nil)) func recipesReadTheWeightsIntoTheArena() throws {
    let (out, inn) = (48, 80)
    let codes = (0..<out * inn).map { Int8(truncatingIfNeeded: $0 * 7 % 255 - 127) }
    let scales = (0..<out).map { Float($0 + 1) * 0x1p-12 }
    let w = (0..<out * inn).map { Float($0) / 3 }
    let path = try safetensors("read.safetensors", [
        ("a.weight", "I8", [out, inn], bytes(codes)), ("a.weight_scale", "F32", [out, 1], bytes(scales)),
        ("gain", "BF16", [inn], bytes((0..<inn).map { bf16(Float($0) / 8) })), ("w", "F32", [out, inn], bytes(w)),
    ])
    let file = try SafeTensors(path)
    let entries: [Weights.Entry] = [("a", "a", .codes), ("gain", "gain", .half), ("w", "w", .halfTransposed), ("e0", "", .target(["a"]))]
    let arena = try Arena(device!, bytes: try Weights.size(entries, in: file))
    let placed = try Weights.load(entries, from: file, into: arena)
    let q = arena.pointer(placed["a"]!.offset).assumingMemoryBound(to: Int8.self)
    let s = arena.pointer(placed["a_scale"]!.offset).assumingMemoryBound(to: Float16.self)
    let g = arena.pointer(placed["gain"]!.offset).assumingMemoryBound(to: Float16.self)
    let wt = arena.pointer(placed["w"]!.offset).assumingMemoryBound(to: Float16.self)
    for o in 0..<out { for i in 0..<inn { #expect(q[i * out + o] == codes[o * inn + i]); #expect(wt[i * out + o] == Float16(w[o * inn + i])) } }
    for o in 0..<out { #expect(s[o] == Float16(scales[o])) }
    for i in 0..<inn { #expect(g[i] == Float16(Float(i) / 8)) }
    let e0 = arena.pointer(placed["e0"]!.offset).assumingMemoryBound(to: Float16.self).pointee
    #expect(Double(e0) == Weights.target16(scales: UnsafePointer(s), count: out, k: inn))
}

/// The transposed bf16 recipe: a bf16 file's weight `[out, in]` as `[in, out]` bit for bit (by tiles of 8 and by the
/// loop past them), an f32 one rounded once to bf16 (nearest, ties to even).
@Test(.enabled(if: device != nil)) func aBF16WeightIsTransposedBitForBit() throws {
    for (out, inn) in [(48, 80), (12, 20)] {
        let w = (0..<out * inn).map { bf16(Float($0) / 7 - 100) }
        let f = (0..<out * inn).map { Float($0) * 1.001 + 0x1p-9 }
        let path = try safetensors("bf16-\(out).safetensors", [("w", "BF16", [out, inn], bytes(w)), ("f", "F32", [out, inn], bytes(f))])
        let file = try SafeTensors(path)
        let entries: [Weights.Entry] = [("w", "w", .bfloatTransposed), ("f", "f", .bfloatTransposed)]
        let slots = Dictionary(uniqueKeysWithValues: try Weights.layout(entries, in: file).map { ($0.name, $0) })
        #expect(slots["w"]!.shape == [inn, out] && slots["w"]!.type == .bfloat16 && slots["w"]!.bytes == 2 * out * inn)
        let arena = try Arena(device!, bytes: try Weights.size(entries, in: file))
        let placed = try Weights.load(entries, from: file, into: arena)
        func at(_ name: String) -> UnsafeMutablePointer<UInt16> { arena.pointer(placed[name]!.offset).assumingMemoryBound(to: UInt16.self) }
        for o in 0..<out {
            for i in 0..<inn {
                #expect(at("w")[i * out + o] == w[o * inn + i])
                #expect(at("f")[i * out + o] == BFloat16Bits.rounded(f[o * inn + i]))
            }
        }
    }
}

/// `target16` on the CPU: ⌊log₂(65 504 / 128 k)⌋ − m − 1 for max |s| in (2^m, 2^(m+1)], capped at ⌊log₂(65 504 / 2 k)⌋.
@Test func target16IsTheRuleOfTheGuard() {
    func t(_ top: Float16, k: Int) -> Double { var s: [Float16] = [0.001, -top, 0.0005]; return s.withUnsafeMutableBufferPointer { Weights.target16(scales: UnsafePointer($0.baseAddress!), count: 3, k: k) } }
    // k = 3840: ⌊log₂(65504 / 491 520)⌋ = −3; the cap ⌊log₂(65504 / 7680)⌋ = 3.
    #expect(t(0x1p-4, k: 3840) == -3 - (-5) - 1)        // 2⁻⁴ in (2⁻⁵, 2⁻⁴]: m = −5
    #expect(t(0x1.01p-4, k: 3840) == -3 - (-4) - 1)     // just past 2⁻⁴: m = −4
    #expect(t(0x1p-7, k: 3840) == 3)                    // m = −8 would give 4: the cap
}

/// The codes transposed by NEON tiles are the byte loop's.
@Test func transposedTilesAreTheByteLoop() {
    let (out, inn) = (48, 80)
    let codes = (0..<out * inn).map { Int8(truncatingIfNeeded: $0 * 31 % 251) }
    var a = [Int8](repeating: 0, count: out * inn), b = a
    codes.withUnsafeBufferPointer { c in
        a.withUnsafeMutableBufferPointer { Weights.transposedTiles(c.baseAddress!, out, inn, into: $0.baseAddress!) }
        b.withUnsafeMutableBufferPointer { Weights.transposedRows(c.baseAddress!, out, inn, into: $0.baseAddress!) }
    }
    #expect(a == b)
    #expect((0..<out).allSatisfy { o in (0..<inn).allSatisfy { i in a[i * out + o] == codes[o * inn + i] } })
}

/// The 16-bit values transposed by NEON tiles are the plain loop's.
@Test func transposedHalvesAreTheLoop() {
    let (out, inn) = (40, 72)
    let values = (0..<out * inn).map { UInt16(truncatingIfNeeded: $0 &* 2654435761 >> 7) }
    var a = [UInt16](repeating: 0, count: out * inn)
    values.withUnsafeBufferPointer { v in a.withUnsafeMutableBufferPointer { Weights.transposedHalves(v.baseAddress!, out, inn, into: $0.baseAddress!, rounding: false) } }
    #expect((0..<out).allSatisfy { o in (0..<inn).allSatisfy { i in a[i * out + o] == values[o * inn + i] } })
}

/// bf16 transposed and rounded to f16 in one pass is the plain loop's `Float16(Float(bf16))`, to the bit: normals,
/// f16's subnormals, ±0, and the values past 65 504 (∞, which `largestMagnitude` reports).
@Test func halvesFromBFloat16AreTheLoop() {
    let (out, inn) = (48, 64)
    let values = (0..<out * inn).map { i -> UInt16 in
        let special: [UInt16] = [0x0000, 0x8000, 0x3f80, 0xbf80, 0x3880, 0x3780, 0x3380, 0x477f, 0x4780, 0x7f80]
        return i < special.count ? special[i] : UInt16(truncatingIfNeeded: (i &* 2654435761) >> 9) & 0xbfff | 0x3000
    }
    var a = [UInt16](repeating: 0, count: out * inn)
    values.withUnsafeBufferPointer { v in a.withUnsafeMutableBufferPointer { Weights.transposedHalves(v.baseAddress!, out, inn, into: $0.baseAddress!, rounding: true) } }
    #expect((0..<out).allSatisfy { o in (0..<inn).allSatisfy { i in
        a[i * out + o] == Float16(Float(bitPattern: UInt32(values[o * inn + i]) << 16)).bitPattern
    } })
    #expect(a.withUnsafeBufferPointer { Weights.largestMagnitude($0.baseAddress!, count: a.count) } >= 0x7c00)
    let finite = a.map { $0 & 0x7fff >= 0x7c00 ? 0x3c00 : $0 }
    #expect(finite.withUnsafeBufferPointer { Weights.largestMagnitude($0.baseAddress!, count: finite.count) } < 0x7c00)
}

/// The target of f16 weights: the largest e₀ with k · max|w| · 2^e₀ ≤ 65 504.
@Test func theTargetOfHalves() {
    let w: [Float16] = [0.25, -3, 1.5, 0]
    let e0 = w.withUnsafeBufferPointer { Weights.target16(weights: $0.baseAddress!, count: 4, k: 3840) }
    #expect(e0 == (log2(65504.0 / (3840 * 3))).rounded(.down))
    #expect(Double(3840 * 3) * pow(2, e0) <= 65504 && Double(3840 * 3) * pow(2, e0 + 1) > 65504)
}
