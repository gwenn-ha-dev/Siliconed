import XCTest
@testable import Siliconed

/// Qwen-Image-2.1-Turbo's pipeline without a byte of weights: the schedule, the noise, the sizes, the wire.
/// The expected bits are diffusers 0.41.0's (`retrieve_timesteps(sigmas=sample_sigmas, mu=μ)` with the
/// Turbo's `scheduler_config.json`, run in the venv: `tools/turbo_survey.py schedule`) and PyTorch 2.14's
/// (`torch.randn` on a CPU generator), printed by the venv.
final class QwenImage21PipelineTests: XCTestCase {
    private func bits(_ v: [Float]) -> [UInt32] { v.map(\.bitPattern) }

    /// The Turbo's `d65dbc9a` files, as published.
    private let config = #"""
        {"_class_name": "FlowMatchEulerDiscreteScheduler", "_diffusers_version": "0.41.0.dev0", "base_image_seq_len": 256,
         "base_shift": 0.5, "invert_sigmas": false, "max_image_seq_len": 8192, "max_shift": 0.9, "num_train_timesteps": 1000,
         "shift": 1.0, "shift_terminal": null, "stochastic_sampling": false, "time_shift_type": "exponential",
         "use_beta_sigmas": false, "use_dynamic_shifting": false, "use_exponential_sigmas": false, "use_karras_sigmas": false}
        """#
    private let modelIndex = #"""
        {"_class_name": "QwenImage21Pipeline", "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
         "sample_sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]}
        """#

    /// **σ and timesteps to the bit, as diffusers gives them** — the same in generation and editing, at
    /// 512², 1024², 1024×1536 (μ is computed and ignored): `float32(sample_sigmas)`, then 0.
    func testTheTurboScheduleIsDiffusersToTheBit() throws {
        let s = try QwenImage21Pipeline.Scheduler(config: Data(config.utf8), modelIndex: Data(modelIndex.utf8))
        XCTAssertEqual(s, QwenImage21Pipeline.Scheduler(), "the published defaults")
        let σ = s.sigmas
        XCTAssertEqual(bits(σ), [0x3f800000, 0x3f7a7be5, 0x3f744524, 0x3f6d375d, 0x3f6523f6, 0x3f585b9f, 0x3f345c57,
                                 0x3ed44242, 0])
        XCTAssertEqual(bits(QwenImage21Pipeline.timesteps(σ)),
                       [0x447a0000, 0x44749cfe, 0x446e8b85, 0x4467a811, 0x445fc51e, 0x44534979, 0x4430222d, 0x43cf48b4])
        // `scheduler.step`'s `dt = σ_next − σ`, in float32.
        XCTAssertEqual(bits(zip(σ.dropFirst(), σ).map { $0 - $1 }),
                       [0xbcb08360, 0xbcc6d820, 0xbce1b8e0, 0xbd013670, 0xbd4c8570, 0xbe0ffd20, 0xbe94766c, 0xbed44242])
        // The round trip the pipeline makes, `(σ·1000)/1000`, gives σ back to the bit for these nodes.
        for x in σ.dropLast() { XCTAssertEqual((x * 1000 / 1000).bitPattern, x.bitPattern) }
        // The module: the published schedule, 8 steps only.
        let module = QwenImage21DenoisingModule(map: "", scheduler: s)
        XCTAssertEqual(module.defaultSteps, 8)
        XCTAssertEqual(bits(module.sigmas(steps: 8)), bits(σ))
        XCTAssertEqual(QwenImage21Pipeline.acceptedSteps, [8])
        XCTAssertNoThrow(try QwenImage21Pipeline.check(steps: 8))
        for n in [5, 6, 7, 9] { XCTAssertThrowsError(try QwenImage21Pipeline.check(steps: n), "\(n)") }
    }

    /// The configs that would move the published σ are refused: the base model's dynamic shift and
    /// terminal stretch, a non-unit shift, Karras σ; and a `model_index.json` without `sample_sigmas`.
    func testTheSchedulerRefusesWhatWouldMoveTheSchedule() {
        func scheduler(_ c: String, _ m: String? = nil) throws -> QwenImage21Pipeline.Scheduler {
            try QwenImage21Pipeline.Scheduler(config: Data(c.utf8), modelIndex: Data((m ?? modelIndex).utf8))
        }
        let base = #"{"use_dynamic_shifting": true, "time_shift_type": "exponential", "shift_terminal": 0.02}"#
        XCTAssertThrowsError(try scheduler(base))
        XCTAssertThrowsError(try scheduler(#"{"use_dynamic_shifting": false, "shift": 3.0}"#))
        XCTAssertThrowsError(try scheduler(#"{"use_dynamic_shifting": false, "shift_terminal": 0.02}"#))
        XCTAssertThrowsError(try scheduler(#"{"use_dynamic_shifting": false, "use_karras_sigmas": true}"#))
        XCTAssertThrowsError(try scheduler(#"{"use_dynamic_shifting": false}"#, #"{"_class_name": "QwenImage21Pipeline"}"#))
        XCTAssertThrowsError(try scheduler(#"{"use_dynamic_shifting": false}"#, #"{"sample_sigmas": [1.0, 0.5]}"#))
        XCTAssertNoThrow(try scheduler(#"{"use_dynamic_shifting": false, "shift": 1.0, "shift_terminal": null}"#))
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
        // An extreme panorama: the short side raised to 512, the area kept under the edit's 1024×1536.
        let wide = QwenImage21Pipeline.outputSize(referenceWidth: 8000, referenceHeight: 1000)
        XCTAssertGreaterThanOrEqual(wide.height, Format.minimumSide)
        XCTAssertLessThanOrEqual(wide.width * wide.height, Format.editSurface)
        XCTAssertEqual(wide.width % 32, 0); XCTAssertEqual(wide.height % 32, 0)
        for (w, h) in [(300, 4000), (4000, 3000), (640, 480), (1, 1000)] {
            let s = QwenImage21Pipeline.outputSize(referenceWidth: w, referenceHeight: h)
            XCTAssertNoThrow(try Format.check(width: s.width, height: s.height), "\(w)×\(h) → \(s)")
        }
    }

    func testRGB8RoundTripsThroughImageRGB() {
        let bytes = (0..<(3 * 7 * 5)).map { (i: Int) -> UInt8 in UInt8((i * 37) % 256) }
        let rgb = Qwen3VLImages.RGB8(bytes: bytes, width: 7, height: 5)
        XCTAssertEqual(QwenImage21Pipeline.rgb8(QwenImage21Pipeline.image(rgb)), rgb)
    }

    func testAFormatThatReadsImagesNeedsAnImageTextModule() {
        XCTAssertTrue(TextFormat.qwenImage21.readsImages)
        XCTAssertFalse(TextFormat.zImage.readsImages)
    }
}
