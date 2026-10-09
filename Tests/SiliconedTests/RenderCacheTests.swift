import Foundation
import XCTest
@testable import Siliconed

/// **The render cache**: its keys name everything that determines a value, what it reads back is
/// the bits it wrote, its folder stays under budget, and a kept K/V file is read back only whole and
/// under its own key. Temporary folders, no map, no GPU.
final class RenderCacheTests: XCTestCase {
    private var folder: URL!

    override func setUpWithError() throws {
        folder = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-cache-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: folder) }

    private func key(_ fields: [String]) -> RenderCache.Key {
        var k = RenderCache.KeyBuilder("test")
        for f in fields { k.add(f) }
        return k.finish()
    }

    // ── keys ────────────────────────────────────────────────────────────────────────────

    func testKeysAreStableAndLengthPrefixed() {
        XCTAssertEqual(key(["a", "b"]), key(["a", "b"]))
        XCTAssertEqual(key(["a"]).hex.count, 64)
        XCTAssertNotEqual(key(["ab", "c"]), key(["a", "bc"]))
        XCTAssertNotEqual(key(["a", ""]), key(["a"]))
        var a = RenderCache.KeyBuilder("conditioning v1"), b = RenderCache.KeyBuilder("latent v1")
        a.add("x"); b.add("x")
        XCTAssertNotEqual(a.finish(), b.finish(), "the kind is part of the key")
    }

    /// One bit of one pixel, the prompt, the image count: each moves the key.
    func testTextKeyNamesThePromptAndEveryImageBit() throws {
        let map = folder.appendingPathComponent("encoder.silicon"), tokenizer = folder.appendingPathComponent("tokenizer")
        try Data([1, 2, 3]).write(to: map)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))
        let module = QwenImage21TextModule(map: map.path, tokenizer: tokenizer.path)
        let image = ImageRGB(pixels: [Float](repeating: 0.25, count: 3 * 4 * 4), height: 4, width: 4)
        var flipped = image.pixels
        flipped[17] = Float(bitPattern: flipped[17].bitPattern ^ 1)
        let other = ImageRGB(pixels: flipped, height: 4, width: 4)
        let base = RenderCache.textKey(module, prompt: "p", images: [image], reproducible: true)
        XCTAssertEqual(base, RenderCache.textKey(module, prompt: "p", images: [image], reproducible: true))
        XCTAssertNotEqual(base, RenderCache.textKey(module, prompt: "p ", images: [image], reproducible: true))
        XCTAssertNotEqual(base, RenderCache.textKey(module, prompt: "p", images: [other], reproducible: true))
        XCTAssertNotEqual(base, RenderCache.textKey(module, prompt: "p", images: [], reproducible: true))
        XCTAssertNotEqual(base, RenderCache.textKey(module, prompt: "p", images: [image, image], reproducible: true))
        XCTAssertNotEqual(base, RenderCache.textKey(module, prompt: "p", images: [image], reproducible: false))
        // Another module type reading the same files is another key.
        let z = ZImageTextModule(map: map.path, tokenizer: tokenizer.path)
        XCTAssertNotEqual(RenderCache.textKey(z, prompt: "p", images: [], reproducible: true),
                          RenderCache.textKey(module, prompt: "p", images: [], reproducible: true))
    }

    /// A reinstalled file (other size, other date) — of the map or inside the tokenizer folder — is another key.
    func testAFileRevisionChangesTheKey() throws {
        let map = folder.appendingPathComponent("encoder.silicon"), tokenizer = folder.appendingPathComponent("tokenizer")
        try Data([1, 2, 3]).write(to: map)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        let json = tokenizer.appendingPathComponent("tokenizer.json")
        try Data("{}".utf8).write(to: json)
        let module = ZImageTextModule(map: map.path, tokenizer: tokenizer.path)
        let before = RenderCache.textKey(module, prompt: "p", images: [], reproducible: true)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 1_000_000)], ofItemAtPath: json.path)
        let touched = RenderCache.textKey(module, prompt: "p", images: [], reproducible: true)
        XCTAssertNotEqual(before, touched)
        try Data([1, 2, 3, 4]).write(to: map)
        XCTAssertNotEqual(touched, RenderCache.textKey(module, prompt: "p", images: [], reproducible: true))
        XCTAssertTrue(RenderCache.stamp(folder.appendingPathComponent("absent").path).hasSuffix("(missing)"))
    }

    // ── what comes back ─────────────────────────────────────────────────────────────────

    /// Every bit back — a negative zero, a NaN payload, a subnormal — with the slots and the format.
    func testConditioningAndLatentComeBackBitForBit() {
        let cache = RenderCache(directory: folder)
        let values: [Float] = [0, -0.0, 1.5, .leastNonzeroMagnitude, Float(bitPattern: 0x7fc0_1234), -3.25e-7]
        let c = Conditioning(format: .qwenImage21, rows: 3, width: 2, values: values, tokens: 3, imageSlots: [false, true, true])
        let k = key(["conditioning"])
        XCTAssertNil(cache.conditioning(k))
        cache.store(c, key: k)
        let back = try! XCTUnwrap(cache.conditioning(k))
        XCTAssertEqual(back.values.map(\.bitPattern), values.map(\.bitPattern))
        XCTAssertEqual(back.format, .qwenImage21)
        XCTAssertEqual(back.imageSlots, [false, true, true])
        XCTAssertEqual([back.rows, back.width, back.tokens], [3, 2, 3])

        let l = Latent(space: .flux2, height: 1, width: 1, values: (0..<128).map { Float($0) * -0.5 })
        let lk = key(["latent"])
        cache.store(l, key: lk)
        let latent = try! XCTUnwrap(cache.latent(lk))
        XCTAssertEqual(latent.space, .flux2)
        XCTAssertEqual(latent.values.map(\.bitPattern), l.values.map(\.bitPattern))
        // A latent is not a conditioning, even under its own key.
        XCTAssertNil(cache.conditioning(lk))
    }

    /// A damaged entry, or one renamed under another key, is a miss — and is removed.
    func testADamagedOrMisnamedEntryIsAMiss() throws {
        let cache = RenderCache(directory: folder)
        let c = Conditioning(format: .zImage, rows: 1, width: 4, values: [1, 2, 3, 4], tokens: 1)
        let k = key(["a"]), other = key(["b"])
        cache.store(c, key: k)
        let path = cache.conditioningFolder.appendingPathComponent(k.hex + ".bin")
        try FileManager.default.copyItem(at: path, to: cache.conditioningFolder.appendingPathComponent(other.hex + ".bin"))
        XCTAssertNil(cache.conditioning(other), "the header names its own key")
        var data = try Data(contentsOf: path)
        data.removeLast(2)
        try data.write(to: path)
        XCTAssertNil(cache.conditioning(k))
        XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))
    }

    // ── the budget ──────────────────────────────────────────────────────────────────────

    /// Least recently used first, down to the budget; the entry just written stays.
    func testEvictionGoesByModificationDate() throws {
        let names = ["old", "middle", "recent", "new"]
        for (i, name) in names.enumerated() {
            let url = folder.appendingPathComponent(name)
            try Data(count: 100).write(to: url)
            try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: Double(1_000 * (i + 1)))],
                                                  ofItemAtPath: url.path)
        }
        let removed = RenderCache.evict(folder: folder, budget: 250, keeping: folder.appendingPathComponent("old"))
        XCTAssertEqual(removed.map(\.lastPathComponent), ["middle", "recent"])
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: folder.path)), ["old", "new"])
        XCTAssertEqual(RenderCache.evict(folder: folder, budget: 1_000), [])
    }

    /// A read touches its entry: after it, the other one is the least recently used.
    func testAReadMakesAnEntryRecent() throws {
        let one = Conditioning(format: .zImage, rows: 1, width: 64, values: [Float](repeating: 1, count: 64), tokens: 1)
        let probe = try RenderCache.encode(header: RenderCache.header(one, key: key(["a"])), values: one.values).count
        let cache = RenderCache(directory: folder, budget: 2 * probe + probe / 2)
        let a = key(["a"]), b = key(["b"]), c = key(["c"])
        cache.store(one, key: a)
        cache.store(one, key: b)
        let past = Date(timeIntervalSince1970: 1_000)
        for k in [a, b] {
            try FileManager.default.setAttributes([.modificationDate: past],
                                                  ofItemAtPath: cache.conditioningFolder.appendingPathComponent(k.hex + ".bin").path)
        }
        XCTAssertNotNil(cache.conditioning(a))   // a is now the recent one
        cache.store(one, key: c)                  // three do not fit: b goes
        XCTAssertNotNil(cache.conditioning(a))
        XCTAssertNil(cache.conditioning(b))
        XCTAssertNotNil(cache.conditioning(c))
    }

    /// An entry larger than the whole budget is not written.
    func testAnEntryOverTheBudgetIsNotKept() {
        let cache = RenderCache(directory: folder, budget: 64)
        let c = Conditioning(format: .zImage, rows: 1, width: 64, values: [Float](repeating: 1, count: 64), tokens: 1)
        cache.store(c, key: key(["big"]))
        XCTAssertNil(cache.conditioning(key(["big"])))
    }

    // ── the kept K/V ────────────────────────────────────────────────────────────────────

    private func page(_ value: UInt8) -> UnsafeMutablePointer<Float> {
        let p = UnsafeMutableRawPointer.allocate(byteCount: Arena.alignment, alignment: Arena.alignment)
        p.initializeMemory(as: UInt8.self, repeating: value, count: Arena.alignment)
        return p.assumingMemoryBound(to: Float.self)
    }

    /// Written, sealed, read back whole under its key; another key does not read it; an unsealed one
    /// leaves nothing behind.
    func testAKeptKVFileIsReadBackOnlyWholeAndUnderItsKey() throws {
        let path = folder.appendingPathComponent("x.kv").path
        let half = Arena.alignment
        let (k0, v0, k1, v1, out0, out1) = (page(1), page(2), page(3), page(4), page(0), page(0))
        defer { for p in [k0, v0, k1, v1, out0, out1] { UnsafeMutableRawPointer(p).deallocate() } }

        do {   // a render cancelled before its prefill ends: nothing is kept
            let file = try XCTUnwrap(QwenImage21KVFile.kept(at: path, key: "k", halfBytes: half, layers: 2, reserve: 0))
            try file.write(layer: 0, keys: k0, values: v0)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path), [])

        do {
            let file = try XCTUnwrap(QwenImage21KVFile.kept(at: path, key: "k", halfBytes: half, layers: 2, reserve: 0))
            XCTAssertFalse(file.complete)
            XCTAssertTrue(file.writesKept)
            try file.write(layer: 0, keys: k0, values: v0)
            try file.write(layer: 1, keys: k1, values: v1)
            file.seal()
            XCTAssertEqual(file.path, path)
        }
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: path)[.size] as? Int,
                       2 * 2 * half + QwenImage21KVFile.trailerBytes)

        do {
            let file = try XCTUnwrap(QwenImage21KVFile.kept(at: path, key: "k", halfBytes: half, layers: 2, reserve: 0))
            XCTAssertTrue(file.complete)
            XCTAssertFalse(file.writesKept)
            try file.wait(layer: 1, keys: out0, values: out1)
            XCTAssertEqual(memcmp(out0, k1, half), 0)
            XCTAssertEqual(memcmp(out1, v1, half), 0)
            try file.wait(layer: 0, keys: out0, values: out1)
            XCTAssertEqual(memcmp(out0, k0, half), 0)
            XCTAssertEqual(memcmp(out1, v0, half), 0)
        }

        // Another key, another shape: not read back — and the stale file is gone.
        let other = try XCTUnwrap(QwenImage21KVFile.kept(at: path, key: "other", halfBytes: half, layers: 2, reserve: 0))
        XCTAssertFalse(other.complete)
        XCTAssertFalse(FileManager.default.fileExists(atPath: path))
    }

    /// A kept file that would leave less than the reserve free is not kept.
    func testAKeptKVFileRespectsTheReserve() {
        XCTAssertNil(QwenImage21KVFile.kept(at: folder.appendingPathComponent("x.kv").path, key: "k",
                                            halfBytes: Arena.alignment, layers: 2, reserve: Int64.max / 2))
    }

    /// One edit kept: the other keys' files, and the half-written ones, go.
    func testOnlyTheCurrentEditsKVIsKept() throws {
        let cache = RenderCache(directory: folder)
        let keep = key(["keep"]), old = key(["old"])
        try FileManager.default.createDirectory(at: cache.kvFolder, withIntermediateDirectories: true)
        for name in [keep.hex + ".kv", old.hex + ".kv", keep.hex + ".kv.123-abc.partial"] {
            try Data([0]).write(to: cache.kvFolder.appendingPathComponent(name))
        }
        cache.keepOnlyKV([keep])
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: cache.kvFolder.path), [keep.hex + ".kv"])
    }

    /// The library counts the cache in what it occupies, and gives it back.
    func testTheLibraryCountsAndEmptiesTheCache() throws {
        let library = Library(root: folder)
        let cache = RenderCache(directory: library.cacheFolder)
        cache.store(Conditioning(format: .zImage, rows: 1, width: 4, values: [1, 2, 3, 4], tokens: 1), key: key(["a"]))
        XCTAssertGreaterThan(library.occupancy().cache, 16)
        try library.emptyCache()
        XCTAssertEqual(library.occupancy().cache, 0)
        XCTAssertNoThrow(try library.emptyCache())
    }
}
