import XCTest
@testable import Siliconed

/// **What was missing BELOW the developer checks.**
///
/// The close-out is an end-to-end verification: it compares the engine to 2.6 GB of golden
/// tensors, it needs 20 GB of weights and a GPU, and it costs five minutes. It is the right way to
/// judge the **numerics** — a unit test would say nothing about a PSNR of 134.70 dB.
///
/// But nothing, until now, caught a regression in a **pure** function: a schedule, a divisor rule,
/// a tiling round trip, an arena alignment. These things are verified in milliseconds, without a
/// byte of map, and they are not covered by a check that requires the whole machine — that is, they
/// are **never** verified between two close-outs.
///
/// These tests found their first defect before having been run even once: a per-step divisor rule
/// contradicted the step count it was derived from.
///
/// **Rule of this target**: no test reads a file, opens the GPU or exceeds a second. What needs
/// the map belongs in the close-out, not here.
final class PureFunctionsTests: XCTestCase {

    // ── the schedule ───────────────────────────────────────────────────────────────────────

    /// **diffusers 0.41.0's Z-Image schedule, bit for bit**: `get_default_z_image_sigmas(N)` through
    /// the published scheduler's `set_timesteps` (shift 3, static), then the terminal 0 — printed by
    /// `retrieve_timesteps` in the venv, as float32 bit patterns. N = 8 is the official recipe's
    /// eight DiT forwards (since 0.8.2); the others exercise torch's two-sided `linspace` (its halves,
    /// odd and even N, a single step).
    private let referenceSchedules: [Int: [UInt32]] = [
        1: [0x3f800000, 0x00000000],
        2: [0x3f800000, 0x3f400000, 0x00000000],
        3: [0x3f800000, 0x3f5b6db6, 0x3f199999, 0x00000000],
        8: [0x3f800000, 0x3f745d17, 0x3f666666, 0x3f555555, 0x3f400000, 0x3f249249, 0x3f000000, 0x3e99999a, 0x00000000],
        9: [0x3f800000, 0x3f75c290, 0x3f69bd39, 0x3f5b6db6, 0x3f4a1af3, 0x3f34b4b5, 0x3f199999, 0x3eec4ec6, 0x3e8ba2e9, 0x00000000],
        13: [0x3f800000, 0x3f7914c3, 0x3f715f17, 0x3f68ba2e, 0x3f5ef7be, 0x3f53dcb0, 0x3f471c72, 0x3f3851eb, 0x3f26f4de, 0x3f124925, 0x3ef286bc, 0x3eb4b4b6, 0x3e4cccce, 0x00000000],
        20: [0x3f800000, 0x3f7b9611, 0x3f76db6d, 0x3f71c71d, 0x3f6c4ec6, 0x3f666666, 0x3f5fffff, 0x3f590b21, 0x3f51745d, 0x3f492494, 0x3f400000, 0x3f35e50d, 0x3f2aaaab, 0x3f1e1e1d, 0x3f100000, 0x3f000000, 0x3edb6db8, 0x3eb13b15, 0x3e800000, 0x3e0ba2e9, 0x00000000],
        28: [0x3f800000, 0x3f7ce0c7, 0x3f799998, 0x3f762763, 0x3f7286bc, 0x3f6eb3e4, 0x3f6aaaab, 0x3f666666, 0x3f61e1e1, 0x3f5d1744, 0x3f580001, 0x3f5294a4, 0x3f4ccccc, 0x3f469ee5, 0x3f400000, 0x3f38e38e, 0x3f313b14, 0x3f28f5c2, 0x3f200000, 0x3f1642c9, 0x3f0ba2ea, 0x3f000000, 0x3ee66667, 0x3eca1af1, 0x3eaaaaac, 0x3e878788, 0x3e400000, 0x3dccccce, 0x00000000],
        50: [0x3f800000, 0x3f7e4531, 0x3f7c7e3e, 0x3f7aaaaa, 0x3f78c9e9, 0x3f76db6d, 0x3f74de9b, 0x3f72d2d2, 0x3f70b767, 0x3f6e8ba4, 0x3f6c4ec6, 0x3f6a0001, 0x3f679e7a, 0x3f65294a, 0x3f629f7a, 0x3f5fffff, 0x3f5d49c2, 0x3f5a7b95, 0x3f579436, 0x3f549249, 0x3f51745d, 0x3f4e38e5, 0x3f4ade32, 0x3f476278, 0x3f43c3c4, 0x3f400000, 0x3f3c14e5, 0x3f37ffff, 0x3f33bea3, 0x3f2f4dea, 0x3f2aaaaa, 0x3f25d174, 0x3f20be82, 0x3f1b6db6, 0x3f15da89, 0x3f100000, 0x3f09d89e, 0x3f035e51, 0x3ef914c1, 0x3eeaaaa9, 0x3edb6db6, 0x3ecb4b4a, 0x3eba2e8c, 0x3ea80001, 0x3e94a529, 0x3e7ffffe, 0x3e53dcb1, 0x3e249249, 0x3de38e38, 0x3d6c4ec5, 0x00000000],
    ]

    func testScheduleIsDiffusersToTheBit() {
        for (steps, bits) in referenceSchedules {
            XCTAssertEqual(FlowMatchSchedule(steps: steps).sigmas.map(\.bitPattern), bits, "\(steps) steps")
        }
    }

    func testEightStepsAreEightEvaluations() {
        let s = FlowMatchSchedule(steps: 8)
        // `steps + 1` values, ending with zero, strictly decreasing: every step moves the latent.
        // The 0.36 schedule (`linspace(1, 0, N)` + 0) ended `0, 0` and gave N − 1 evaluations.
        XCTAssertEqual(s.sigmas.count, 9)
        XCTAssertEqual(s.sigmas.last, 0)
        XCTAssertEqual(s.sigmas[0], 1)
        for i in 0..<(s.sigmas.count - 1) {
            XCTAssertGreaterThan(s.sigmas[i], s.sigmas[i + 1], "σ must decrease strictly; at index \(i) it does not")
        }
        XCTAssertEqual(ZImageDenoisingModule(map: "").plan(height: 64, width: 64, steps: 8, start: 0).evaluations, 8)
    }

    func testScheduleGivesBackTheReferenceValues() {
        // The σ of the reference's schedule, as in `FlowMatchSchedule`'s documentation.
        let expected: [Float] = [1, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3, 0]
        let s = FlowMatchSchedule(steps: 8)
        for (i, value) in expected.enumerated() {
            XCTAssertEqual(s.sigmas[i], value, accuracy: 1e-3,
                           "σ[\(i)]: the schedule has changed, or the shift is no longer 3")
        }
    }

    /// **Krea 2's plan**: the recipe's eight evaluations, one per step from the start.
    func testKrea2Plan() {
        let module = Krea2DenoisingModule(map: "")
        let plan = module.plan(height: 128, width: 128, steps: 8, start: 0)
        XCTAssertEqual(plan.evaluations, 8)
        XCTAssertEqual(plan.startSigma, 1)
        XCTAssertEqual(module.plan(height: 128, width: 128, steps: 8, start: 3).evaluations, 5)
    }

    // ── the formats ───────────────────────────────────────────────────────────────────────

    func testFormatFloorPerSide() {
        XCTAssertNoThrow(try Format.check(width: 512, height: 512))
        XCTAssertNoThrow(try Format.check(width: 512, height: 768))
        for (l, h) in [(496, 1024), (1024, 496), (256, 256), (448, 2048)] {
            XCTAssertThrowsError(try Format.check(width: l, height: h)) {
                XCTAssertEqual($0 as? EngineError, .imageTooSmall(side: min(l, h), minimum: 512))
            }
        }
    }

    func testFormatIsMultipleOf16() {
        for (l, h) in [(520, 512), (512, 1000), (840, 1216)] {
            XCTAssertThrowsError(try Format.check(width: l, height: h)) {
                XCTAssertEqual($0 as? EngineError, .formatRefused(width: l, height: h, reason: .notMultiple, bound: 16))
            }
        }
        XCTAssertNoThrow(try Format.check(width: 528, height: 1008))
    }

    /// An app's usual formats pass, 1024² included — and so do the larger ones: the area has no ceiling
    /// (B decides, `FormatTableTests`), only the sides are bounded.
    func testFormatHasNoSurfaceCeiling() {
        for (l, h) in [(1024, 1024), (832, 1216), (1216, 832), (768, 1344), (1344, 768),
                       (896, 1152), (1152, 896), (1024, 1536), (1536, 1024)] {
            XCTAssertNoThrow(try Format.check(width: l, height: h), "\(l)×\(h)")
        }
        for (l, h) in [(2048, 2048), (1040, 1536), (1536, 1040), (1264, 1264), (1024, 1552), (2048, 3072)] {
            XCTAssertNoThrow(try Format.check(width: l, height: h), "\(l)×\(h)")
        }
    }

    /// `WxH`: width first, like Draw Things — `832x1216` is a portrait.
    func testFormatReadsWidthByHeight() {
        XCTAssertTrue(Format.parse("832x1216")! == (832, 1216))
        XCTAssertTrue(Format.parse("1216X832")! == (1216, 832))
        XCTAssertTrue(Format.parse("768×1344")! == (768, 1344))
        XCTAssertTrue(Format.parse("512")! == (512, 512))
        // Spaces around the sign, as a size is written for the eye.
        for spaced in ["832 × 1216", "832 x 1216", "832x 1216", " 832 X 1216 "] { XCTAssertTrue(Format.parse(spaced)! == (832, 1216), spaced) }
        for unreadable in ["", "x", "832x", "x1216", "832x1216x2", "a library", "83 2x1216", " x 1216"] {
            XCTAssertNil(Format.parse(unreadable), unreadable)
        }
        XCTAssertEqual(Format.fileLabel(width: 512, height: 512), "512",
                       "square names do not change — the close-out finds them again")
        XCTAssertEqual(Format.fileLabel(width: 832, height: 1216), "832x1216")
        XCTAssertEqual(Format.label(width: 1216, height: 832), "1216×832")
    }

    // ── the noise ──────────────────────────────────────────────────────────────────────────

    func testSameSeedSameNoise() {
        func sample(_ seed: UInt64) -> [Float] {
            var b = Noise(seed: seed)
            var v = [Float](repeating: 0, count: 256)
            v.withUnsafeMutableBufferPointer { b.fill($0.baseAddress!, count: 256) }
            return v
        }
        // The first half of the "same seed → same image" contract. The second — the frozen GPU/AMX
        // cut — is verified by a developer check, which needs the map.
        XCTAssertEqual(sample(42), sample(42))
        XCTAssertNotEqual(sample(42), sample(43))
        XCTAssertTrue(sample(42).allSatisfy { $0.isFinite }, "Box-Muller must never see log(0)")
    }

    func testNoiseIsRoughlyCenteredAndReduced() {
        var b = Noise(seed: 7)
        let sampleCount = 100_000
        var v = [Float](repeating: 0, count: sampleCount)
        v.withUnsafeMutableBufferPointer { b.fill($0.baseAddress!, count: sampleCount) }
        let mean = v.reduce(0, +) / Float(v.count)
        let variance = v.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(v.count)
        XCTAssertEqual(mean, 0, accuracy: 0.02)
        XCTAssertEqual(variance, 1, accuracy: 0.05)
    }

    // ── the settings ──────────────────────────────────────────────────────────────────────

    /// Anima Turbo's schedule, against the σ the oracle wrote (`goldens-anima-trajectory`,
    /// `FlowMatchEulerDiscreteScheduler`, shift 3) — copied here so the test opens no file.
    func testAnimaScheduleIsTheOraclesOne() {
        let expectedCounts: [Float] = [1, 0.954545, 0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3, 0]
        let schedule = AnimaDiT.sigmas(steps: 8)
        XCTAssertEqual(schedule.count, expectedCounts.count)
        for (a, b) in zip(schedule, expectedCounts) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    }

    // ── Krea 2 ───────────────────────────────────────────────────────────────────────────

    /// Krea 2 Turbo's schedule — EXPONENTIAL shift at μ = 1.15 —, against the σ that
    /// `FlowMatchEulerDiscreteScheduler` gives under `retrieve_timesteps(…, mu=1.15)`, copied here.
    func testKrea2ScheduleIsTheReferencesOne() {
        let expectedCounts: [Float] = [1, 0.956724, 0.904531, 0.840349, 0.759511, 0.654567, 0.512844, 0.310901, 0]
        let schedule = Krea2DiT.sigmas(steps: 8)
        XCTAssertEqual(schedule.count, expectedCounts.count)
        for (a, b) in zip(schedule, expectedCounts) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    }

    /// Krea 2's license as the app shows it: whole (its last section reached), and still carrying the
    /// § 3.1 notice the app quotes apart, word for word.
    func testKrea2LicenseCarriesItsNotice() {
        let t = Krea2License.text
        XCTAssertTrue(t.hasPrefix("KREA 2 COMMUNITY LICENSE AGREEMENT"))
        XCTAssertTrue(t.contains("\"" + Krea2License.attribution + "\""))
        XCTAssertTrue(t.contains("4.2 Content Filtering Requirement."))
        XCTAssertTrue(t.hasSuffix("will not operate as a waiver of such right or provision."))
        XCTAssertFalse(t.contains("\n_\n"), "a PDF extraction artefact left in")
    }
}
