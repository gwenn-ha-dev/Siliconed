import XCTest
@testable import Siliconed

/// **A seed gives diffusers' noise on the five families**: the engine's starting latent equals, bit for bit,
/// `randn_tensor(shape, generator=torch.Generator("cpu").manual_seed(42), dtype=float32)` flattened — the shape
/// each pipeline's `prepare_latents` draws, whose row-major order *is* the engine's planar `[C, h, w]`.
/// Values read once with diffusers 0.41.0 / torch 2.14.1 (indices 0-3, 15, 16 — the first block boundary —, the
/// middle and the last three); a 512² and a rectangle per family.
final class StartingNoiseTests: XCTestCase {

    private func check(_ space: LatentSpace, h: Int, w: Int, at indices: [Int], _ expected: [UInt32],
                       file: StaticString = #filePath, line: UInt = #line) {
        let latent = Latent.noise(space, height: h, width: w, seed: 42)
        XCTAssertEqual(latent.values.count, space.channels * h * w, file: file, line: line)
        XCTAssertEqual(indices.map { latent.values[$0].bitPattern }, expected, file: file, line: line)
    }

    /// Z-Image: `(1, 16, h/8, w/8)`; 512² then 768×512.
    func testZImage() {
        check(.flux, h: 64, w: 64, at: [0, 1, 2, 3, 15, 16, 32768, 32769, 65533, 65534, 65535],
              [0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0x3f432f9f, 0x3fd23771, 0xbf48a5f1, 0xbeb6b9f3,
               0x3f385ee7, 0xbed8c0c1, 0x3fcf8d26])
        check(.flux, h: 96, w: 64, at: [0, 1, 2, 3, 15, 16, 49152, 49153, 98301, 98302, 98303],
              [0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0x3f432f9f, 0x3fd23771, 0x3fa1f5d2, 0x3e30ba0b,
               0x3e5dfb6f, 0x3f961b8f, 0x3f23d508])
    }

    /// Anima: `(1, 16, 1, h/8, w/8)` — the frame axis of length 1 adds no order. Same space as Krea 2.
    func testAnima() {
        check(.qwenImage, h: 64, w: 64, at: [0, 1, 2, 3, 15, 16, 32768, 32769, 65533, 65534, 65535],
              [0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0x3f432f9f, 0x3fd23771, 0xbf48a5f1, 0xbeb6b9f3,
               0x3f385ee7, 0xbed8c0c1, 0x3fcf8d26])
        check(.qwenImage, h: 96, w: 64, at: [0, 1, 2, 3, 15, 16, 49152, 49153, 98301, 98302, 98303],
              [0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0x3f432f9f, 0x3fd23771, 0x3fa1f5d2, 0x3e30ba0b,
               0x3e5dfb6f, 0x3f961b8f, 0x3f23d508])
    }

    /// Krea 2: `prepare_latents` draws `(1, 16, h/8, w/8)` planar *then* packs it 2×2 into tokens: the engine
    /// keeps the planar tensor and packs at the DiT, so the draw is the planar one. 512² then 608×416.
    func testKrea2() {
        check(.qwenImage, h: 64, w: 64, at: [0, 1, 2, 3, 15, 16, 32768, 32769, 65533, 65534, 65535],
              [0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0x3f432f9f, 0x3fd23771, 0xbf48a5f1, 0xbeb6b9f3,
               0x3f385ee7, 0xbed8c0c1, 0x3fcf8d26])
        check(.qwenImage, h: 52, w: 76, at: [0, 1, 2, 3, 15, 16, 31616, 31617, 63229, 63230, 63231],
              [0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0x3f432f9f, 0x3fd23771, 0xbe912ac3, 0x3fbae79e,
               0xbeb7cab8, 0x3f92f73d, 0x3dee7df8])
    }

    /// MOSS-Image: `(1, 128, h/16, w/16)`; 512² then 608×416.
    func testMossImage() {
        check(.flux2, h: 32, w: 32, at: [0, 1, 2, 3, 15, 16, 65536, 65537, 131069, 131070, 131071],
              [0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0x3f432f9f, 0x3fd23771, 0x3f636eb4, 0x40076cf6,
               0x3e5e3715, 0x3f1d4a81, 0xbf31a466])
        check(.flux2, h: 38, w: 26, at: [0, 1, 2, 3, 15, 16, 63232, 63233, 126461, 126462, 126463],
              [0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0x3f432f9f, 0x3fd23771, 0xbf96fae3, 0xbdb0fee5,
               0x3e50ec61, 0xbf5b3d0a, 0xbe4a7e20])
    }

    /// Every space draws the same stream: a seed's first values do not depend on the space.
    func testOneStreamForEverySpace() {
        let a = Latent.noise(.flux, height: 8, width: 8, seed: 3).values
        let b = Latent.noise(.qwenImage21, height: 4, width: 4, seed: 3).values
        XCTAssertEqual(Array(a.prefix(b.count)), b)
    }
}
