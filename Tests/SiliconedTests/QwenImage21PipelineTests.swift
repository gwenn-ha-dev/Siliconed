import XCTest
@testable import Siliconed

/// Qwen-Image-2.1's pipeline without a byte of weights: the schedule, the noise, the sizes, the wire.
/// The expected bits are diffusers' (`FlowMatchEulerDiscreteScheduler` with Viggle's config) and
/// PyTorch 2.14's (`torch.randn` on a CPU generator), printed by the venv.
final class QwenImage21PipelineTests: XCTestCase {
    private func bits(_ v: [Float]) -> [UInt32] { v.map(\.bitPattern) }

    func testTheTurboScheduleIsDiffusersToTheBit() throws {
        let s = QwenImage21Pipeline.Scheduler()
        XCTAssertEqual(s.mu(imageTokens: 1024), 0.5387096774193548, accuracy: 1e-15)
        XCTAssertEqual(bits(try QwenImage21Pipeline.sigmas(steps: 6, imageTokens: 1024, scheduler: s)),
                       [0x3f800000, 0x3f766a1a, 0x3f6c4d6f, 0x3f5650c9, 0x3f21aac8, 0x3eba25d8, 0])
        let σ4096 = try QwenImage21Pipeline.sigmas(steps: 6, imageTokens: 4096, scheduler: s)
        XCTAssertEqual(bits(σ4096), [0x3f800000, 0x3f77bec2, 0x3f6ef092, 0x3f5b70ee, 0x3f2ab082, 0x3eccd96c, 0])
        XCTAssertEqual(bits(QwenImage21Pipeline.timesteps(σ4096)),
                       [0x447a0000, 0x4471f049, 0x446956ef, 0x44564c48, 0x4426b05f, 0x43c80c53])
        // μ beyond `max_image_seq_len` is extrapolated (1024×1536: 6,144 tokens).
        XCTAssertEqual(bits(try QwenImage21Pipeline.sigmas(steps: 6, imageTokens: 6144, scheduler: s)),
                       [0x3f800000, 0x3f7887f9, 0x3f708300, 0x3f5e8eee, 0x3f3074eb, 0x3ed9a7c8, 0])
        // The 9-step mode's nodes, `1/6` and `1/12` as Python's doubles made float32.
        XCTAssertEqual(bits(try QwenImage21Pipeline.sigmas(steps: 9, imageTokens: 4056, scheduler: s)),
                       [0x3f800000, 0x3f7a8a44, 0x3f74db65, 0x3f6ee858, 0x3f5b60be, 0x3f2a9324, 0x3ecc9a02,
                        0x3e921df6, 0x3e1d52c4, 0])
        // At 512² (1,024 tokens), the oracle's `trajectory9 512` (Viggle's README recipe, run): σ and timesteps.
        let σ9 = try QwenImage21Pipeline.sigmas(steps: 9, imageTokens: 1024, scheduler: s)
        XCTAssertEqual(bits(σ9), [0x3f800000, 0x3f79a931, 0x3f731c18, 0x3f6c4d6f, 0x3f5650c9, 0x3f21aac8, 0x3eba25d8,
                                  0x3e82b20b, 0x3e0a087d, 0])
        XCTAssertEqual(bits(QwenImage21Pipeline.timesteps(σ9)),
                       [0x447a0000, 0x4473cf3a, 0x446d696f, 0x4466c39e, 0x44514ae4, 0x441de0c7, 0x43b5c8f5,
                        0x437f43bd, 0x4306cc4a])
        // The README switches the LoRA off after step index 6: 7 turbo steps, then 2 of the base model.
        XCTAssertEqual(QwenImage21Pipeline.turboSteps(9), 7)
        XCTAssertEqual(QwenImage21Pipeline.turboSteps(6), 6)
        XCTAssertThrowsError(try QwenImage21Pipeline.nodes(steps: 8))
    }

    func testTheSchedulerConfigRefusesTheBaseTerminalShift() {
        let base = #"{"use_dynamic_shifting": true, "time_shift_type": "exponential", "shift_terminal": 0.02}"#
        XCTAssertThrowsError(try QwenImage21Pipeline.Scheduler(json: Data(base.utf8)))
        let turbo = #"{"use_dynamic_shifting": true, "time_shift_type": "exponential", "shift_terminal": null, "max_image_seq_len": 8192, "max_shift": 0.9}"#
        XCTAssertEqual(try QwenImage21Pipeline.Scheduler(json: Data(turbo.utf8)), QwenImage21Pipeline.Scheduler())
    }

    func testTheNoiseIsTorchRandn() {
        // `torch.randn((1, 1, 64, 2, 3), generator=manual_seed(0))`: 384 values, a multiple of 16.
        var a = [Float](repeating: 0, count: 384)
        var g = TorchNoise(seed: 0)
        a.withUnsafeMutableBufferPointer { g.fill($0.baseAddress!, count: 384) }
        XCTAssertEqual(bits(Array(a.prefix(4))), [0xbf901b85, 0xbf93808a, 0xbe804bd6, 0xbede255e])
        XCTAssertEqual(bits(Array(a.suffix(3))), [0x3f0dda9a, 0xbe3a03d5, 0xbe7019cb])
        // A count that is not a multiple of 16 (never a latent: 64 channels) is NOT claimed to the bit:
        // `torch.randn(20, seed 42)` differs in one value by one ulp (index 14) — PyTorch's tail path
        // rounds differently somewhere. 19 of 20 agree.
        var b = [Float](repeating: 0, count: 20)
        var h = TorchNoise(seed: 42)
        b.withUnsafeMutableBufferPointer { h.fill($0.baseAddress!, count: 20) }
        let torch: [UInt32] = [0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0xbf4214e2, 0x3f8a0650, 0x3f4d0143,
                               0x3fd71e93, 0x3eb63341, 0xbf2fc686, 0xbefc9934, 0x3e774894, 0xbe6d2eed, 0x3d2b0c00,
                               0xbe80ce79, 0x3f5c1fb0, 0xbe9e9487, 0xbeca9a91, 0x3f4dac3c, 0xbf1f20e0]
        XCTAssertEqual(zip(bits(b), torch).filter { $0 != $1 }.count, 1)
        // The product's latent draws it, Qwen-Image-2.1 only.
        XCTAssertEqual(bits(Array(Latent.noise(.qwenImage21, height: 2, width: 3, seed: 0).values)), bits(a))
    }

    func testTheOutputSizeFollowsTheFirstReference() {
        // 1216×832 (the test photo) at ~1024²: 1248×832, the oracle's own condition size at 1024.
        XCTAssertTrue(QwenImage21Pipeline.outputSize(referenceWidth: 1216, referenceHeight: 832) == (1248, 832))
        XCTAssertTrue(QwenImage21Pipeline.outputSize(referenceWidth: 1024, referenceHeight: 1024) == (1024, 1024))
        // An extreme panorama: the short side raised to 512, the area kept under 1024×1536.
        let wide = QwenImage21Pipeline.outputSize(referenceWidth: 8000, referenceHeight: 1000)
        XCTAssertGreaterThanOrEqual(wide.height, Format.minimumSide)
        XCTAssertLessThanOrEqual(wide.width * wide.height, Format.maxSurface)
        XCTAssertEqual(wide.width % 32, 0); XCTAssertEqual(wide.height % 32, 0)
        for (w, h) in [(300, 4000), (4000, 3000), (640, 480), (1, 1000)] {
            let s = QwenImage21Pipeline.outputSize(referenceWidth: w, referenceHeight: h)
            XCTAssertNoThrow(try Format.check(width: s.width, height: s.height), "\(w)×\(h) → \(s)")
        }
    }

    func testRGB8RoundTripsThroughImageRGB() {
        let bytes = (0..<(3 * 7 * 5)).map { UInt8(($0 * 37) % 256) }
        let rgb = Qwen3VLImages.RGB8(bytes: bytes, width: 7, height: 5)
        XCTAssertEqual(QwenImage21Pipeline.rgb8(QwenImage21Pipeline.image(rgb)), rgb)
    }

    func testAFormatThatReadsImagesNeedsAnImageTextModule() {
        XCTAssertTrue(TextFormat.qwenImage21.readsImages)
        XCTAssertFalse(TextFormat.klein.readsImages)
    }
}
