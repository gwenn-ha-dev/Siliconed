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
/// These tests found their first defect before having been run even once: the rule of
/// `Spectral.divisors` contradicted `Spectral.steps` on the only step where the two pronounce
/// together. See the comment of `Spectral.divisors`.
///
/// **Rule of this target**: no test reads a file, opens the GPU or exceeds a second. What needs
/// the map belongs in the close-out, not here.
final class PureFunctionsTests: XCTestCase {

    // ── the schedule ───────────────────────────────────────────────────────────────────────

    func testScheduleEightStepsGivesSevenEvaluations() {
        let s = FlowMatchSchedule(steps: 8)
        // `steps + 1` values, ending with zero — and the last step moves nothing, hence
        // `N steps = N−1 evaluations` (measured).
        XCTAssertEqual(s.sigmas.count, 9)
        XCTAssertEqual(s.sigmas.last, 0)
        XCTAssertEqual(s.effectiveSteps, 7)
    }

    func testScheduleIsStrictlyDecreasingAndStartsAtOne() {
        let s = FlowMatchSchedule(steps: 8)
        XCTAssertEqual(s.sigmas[0], 1, accuracy: 1e-6,
                       "σ₀ = shift·1/(1+(shift−1)·1) = 1 whatever the shift")
        // **Strictly decreasing up to the second-to-last, and NOT up to the last.** The bare
        // schedule already ends at zero (`raw = 0` at the last point), and the pipeline appends one
        // — hence two zeros at the tail. It is not a wart: it is *the* mechanism of `N steps = N−1 evaluations`, a last step
        // that moves nothing, so `N steps = N−1 evaluations`.
        for i in 0..<(s.sigmas.count - 2) {
            XCTAssertGreaterThan(s.sigmas[i], s.sigmas[i + 1],
                                 "σ must decrease strictly; at index \(i) it does not")
        }
        XCTAssertEqual(s.sigmas[s.sigmas.count - 2], 0,
                       "the second-to-last σ is already zero — this is what makes the last step null")
    }

    func testScheduleGivesBackTheReferenceValues() {
        // The σ of the reference's schedule, as in the spectral schedule documentation.
        let expectedCounts: [Float] = [1, 0.947, 0.882, 0.8, 0.692, 0.545, 0.333, 0]
        let s = FlowMatchSchedule(steps: 8)
        for (i, expected) in expectedCounts.enumerated() {
            XCTAssertEqual(s.sigmas[i], expected, accuracy: 1e-3,
                           "σ[\(i)]: the schedule has changed, or the shift is no longer 3")
        }
    }

    func testTheModelTimeIsInverted() {
        // The DiT receives `(1000 − t)/1000 = 1 − σ`. Confusing it with σ gives a plausible and
        // wrong image — it is the first of the three traps of `Sampler`.
        let s = FlowMatchSchedule(steps: 8)
        XCTAssertEqual(s.modelTime(at: 0), 0, accuracy: 1e-6)
        XCTAssertEqual(s.modelTime(at: 8), 1, accuracy: 1e-6)
    }

    // ── the spectral schedule ──────────────────────────────────────────────────────────────

    func testSpectralDoesNotApplyBelow1024() {
        let s = FlowMatchSchedule(steps: 8)
        // 512² → latent 64: `k` half-size evaluations there would be 256², outside the model's
        // domain (256² was removed from the bench for this reason).
        XCTAssertEqual(Spectral.steps(height: 64, width: 64, schedule: s), 0)
        XCTAssertEqual(Spectral.steps(height: 128, width: 128, schedule: s), 2,
                       "the rule f* ≤ 0.10 gives k = 2 at 1024² — this is M35, not §7b″'s k=3")
    }

    func testDivisorsFollowTheSameRuleAsTheStepCount() {
        let s = FlowMatchSchedule(steps: 8)
        let d = Spectral.divisors(height: 128, width: 128, schedule: s)
        // One divisor per STEP, hence eight — `Sampler.divisor(_:)` indexes them over `0..<8`. The
        // eighth is that of the step that moves nothing (σ = 0), and it is 1 like the others.
        // The threshold alone would give [4, 2, 1, …], but a quarter of 1024² is 256², below the
        // 512² floor: the first step drops to 2.
        XCTAssertEqual(d, [2, 2, 1, 1, 1, 1, 1, 1])
        // **The property that found the defect**: both functions derive from the same threshold,
        // so they must agree. The number of reduced-size steps according to `divisors` is the `k`
        // that `steps` announces. Written `seuil / d`, the rule gave [4,1,1,…] and this assertion
        // failed.
        XCTAssertEqual(d.filter { $0 >= 2 }.count, Spectral.steps(height: 128, width: 128, schedule: s))
    }

    func testNoDivisorGoesBelow512() {
        let s = FlowMatchSchedule(steps: 8)
        for latent in [128, 256] {
            let d = Spectral.divisors(height: latent, width: latent, schedule: s)
            XCTAssertTrue(d.allSatisfy { latent / $0 >= Spectral.minimumSide }, "latent \(latent)")
        }
    }

    func testDivisorsAreAllOneBelow1024() {
        let s = FlowMatchSchedule(steps: 8)
        let d = Spectral.divisors(height: 64, width: 64, schedule: s)
        XCTAssertEqual(d.count, 8, "one divisor per step")
        XCTAssertTrue(d.allSatisfy { $0 == 1 })
    }

    /// **The floor is per side**: 832×1216 (latent 104 × 152) has the area of 1024², but its
    /// half-size would be 416 px tall — below the floor. Hence `k = 0`, on both axes.
    func testSpectralIsZeroIfAHalfSideFallsBelowTheFloor() {
        let s = FlowMatchSchedule(steps: 8)
        for (h, w) in [(152, 104), (104, 152), (168, 96), (96, 168), (128, 64), (64, 192)] {
            XCTAssertEqual(Spectral.steps(height: h, width: w, schedule: s), 0, "\(w * 8)×\(h * 8)")
            XCTAssertTrue(Spectral.divisors(height: h, width: w, schedule: s).allSatisfy { $0 == 1 })
        }
        // Both half-sizes at the floor: the spectral applies, as for the square.
        XCTAssertEqual(Spectral.steps(height: 128, width: 192, schedule: s), 2)
        XCTAssertEqual(Spectral.divisors(height: 192, width: 128, schedule: s), [2, 2, 1, 1, 1, 1, 1, 1])
    }

    /// An odd half-latent cannot be patchified (patch ×2): 1040 px → latent 130 → 65. The
    /// spectral declines there instead of throwing at the first step.
    func testSpectralIsZeroIfTheHalfLatentDoesNotPatchify() {
        let s = FlowMatchSchedule(steps: 8)
        XCTAssertEqual(Spectral.steps(height: 130, width: 130, schedule: s), 0)
        XCTAssertEqual(Spectral.steps(height: 132, width: 132, schedule: s), 2)
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
                XCTAssertEqual($0 as? EngineError, .formatRefused(width: l, height: h, reason: .notMultiple))
            }
        }
        XCTAssertNoThrow(try Format.check(width: 528, height: 1008))
    }

    /// An app's usual formats pass, 1024² included; beyond the measured ceiling, refusal.
    func testFormatSurfaceCeiling() {
        for (l, h) in [(1024, 1024), (832, 1216), (1216, 832), (768, 1344), (1344, 768),
                       (896, 1152), (1152, 896), (1024, 1536), (1536, 1024)] {
            XCTAssertNoThrow(try Format.check(width: l, height: h), "\(l)×\(h)")
        }
        // 1024×1536 is the measured ceiling; 16 px more are enough to exceed it.
        for (l, h) in [(2048, 2048), (1040, 1536), (1536, 1040), (1264, 1264), (1024, 1552)] {
            XCTAssertThrowsError(try Format.check(width: l, height: h)) {
                XCTAssertEqual($0 as? EngineError, .formatRefused(width: l, height: h, reason: .tooLarge))
            }
        }
    }

    /// `WxH`: width first, like Draw Things — `832x1216` is a portrait.
    func testFormatReadsWidthByHeight() {
        XCTAssertTrue(Format.parse("832x1216")! == (832, 1216))
        XCTAssertTrue(Format.parse("1216X832")! == (1216, 832))
        XCTAssertTrue(Format.parse("768×1344")! == (768, 1344))
        XCTAssertTrue(Format.parse("512")! == (512, 512))
        for unreadable in ["", "x", "832x", "x1216", "832x1216x2", "a library", "832 x 1216"] {
            XCTAssertNil(Format.parse(unreadable), unreadable)
        }
        XCTAssertEqual(Format.fileLabel(width: 512, height: 512), "512",
                       "square names do not change — the close-out finds them again")
        XCTAssertEqual(Format.fileLabel(width: 832, height: 1216), "832x1216")
        XCTAssertEqual(Format.label(width: 1216, height: 832), "1216×832")
    }

    /// 𝓟 then 𝓘 on a rectangular grid: the axes do not cross. A ramp in `y` alone must stay one —
    /// swapping the axes would turn it into a ramp in `x`.
    func testRectangleResamplingKeepsItsAxes() {
        let (h, w) = (4, 6)
        let full = (0..<(2 * h * 2 * w)).map { Float($0 / (2 * w)) }   // [1, 2h, 2w], rampe en y
        var low = [Float](repeating: .nan, count: h * w)
        Resample.decimate(full, into: &low, channels: 1, height: h, width: w)
        for y in 0..<h { for x in 0..<w { XCTAssertEqual(low[y * w + x], Float(2 * y + 1)) } }
        var high = [Float](repeating: .nan, count: 4 * h * w)
        Resample.bilinearDouble(low, into: &high, channels: 1, height: h, width: w)
        for y in 0..<(2 * h) {
            let rowLine = high[(y * 2 * w)..<((y + 1) * 2 * w)]
            XCTAssertTrue(rowLine.allSatisfy { $0 == rowLine.first! }, "row \(y): constant in x")
        }
    }

    // ── the GPU/AMX cut ──────────────────────────────────────────────────────────────────

    func testTheCutIsQuantizedAndBounded() {
        let q = Conductor.quantum
        // The quantum exists so that the MPS operator cache stays small: a different `gpuRows`
        // on each call made them drop from 3.35 to 1.46 TFLOP/s.
        for m in [1024, 2048, 4128, 8192] {
            for T in stride(from: 0.05, through: 0.95, by: 0.05) {
                let c = Conductor.cut(m, T)
                XCTAssertEqual(c % q, 0, "unquantized cut: m=\(m) T=\(T) → \(c)")
                XCTAssertGreaterThanOrEqual(c, q, "both engines must have work")
                XCTAssertLessThanOrEqual(c, m - q)
            }
        }
    }

    func testTheCutIsMonotonicInT() {
        let m = 4128
        var previous = 0
        for T in stride(from: 0.30, through: 0.95, by: 0.01) {
            let c = Conductor.cut(m, T)
            XCTAssertGreaterThanOrEqual(c, previous, "increasing T must give the GPU more rows")
            previous = c
        }
    }

    func testTheCutIsAPureFunction() {
        // This is **the whole reproducibility contract**: with `T` frozen, the cut depends only on
        // the shape, so two renders of the same seed yield the same bits.
        for _ in 0..<100 {
            XCTAssertEqual(Conductor.cut(4128, 0.631), Conductor.cut(4128, 0.631))
        }
    }

    // ── the GPU heights ───────────────────────────────────────────────────────────────

    func testRowSplitCoversEveryRowOnceAtSafeHeights() {
        let q = GEMM.rowQuantum
        for m in Array(1...300) + [833, 1055, 1056, 2048, 3277, 4096, 4115, 4128, 8191] {
            let (body, tail) = GEMM.rowSplit(m)
            XCTAssertEqual(body + tail, m, "rows lost or doubled: m=\(m)")
            XCTAssertTrue(tail >= 0 && tail < q, "the tail goes through a \(q)-row tier: m=\(m) → \(tail)")
            if m < q {
                XCTAssertEqual(body, m, "below the quantum nothing changes: m=\(m)")
            } else {
                XCTAssertEqual(body % q, 0, "a body that is not a multiple of \(q): m=\(m) → \(body)")
            }
        }
        // The two faulty heights seen in Z-Image's DiT without the AMX.
        XCTAssertTrue(GEMM.rowSplit(1056) == (1024, 32))
        XCTAssertTrue(GEMM.rowSplit(4128) == (4096, 32))
        // The tail tier must hold the tail.
        XCTAssertGreaterThanOrEqual(GEMM.submittedRows, q - 1)
    }

    func testCutHeightsNeedNoSplit() {
        // The GPU/AMX cut already hands the GPU multiples of its quantum: the split must not
        // add a submission there.
        XCTAssertEqual(Conductor.quantum % GEMM.rowQuantum, 0)
        for m in [1024, 2048, 4128] {
            XCTAssertEqual(GEMM.rowSplit(Conductor.cut(m, 0.631)).tail, 0)
        }
    }

    // ── the arena ───────────────────────────────────────────────────────────────────────────

    func testArenaAlignedAndRefusingToGrow() throws {
        let liveBefore = Arena.live
        do {
            let a = try Arena(capacity: 1 << 20)
            XCTAssertEqual(Arena.live, liveBefore + 1)
            let p = try a.reserve("x", bytes: 1000)
            XCTAssertEqual(Int(bitPattern: p) % Arena.alignment, 0,
                           "`bytesNoCopy` requires a page-aligned pointer")
            // Two slices of the same name would be two views of the same memory under two names.
            XCTAssertThrowsError(try a.reserve("x", bytes: 8))
            // A request that does not fit is a sizing error, not a `realloc`.
            XCTAssertThrowsError(try a.reserve("énorme", bytes: 1 << 30))
        }
        XCTAssertEqual(Arena.live, liveBefore, "a destroyed arena must give back its pages")
    }

    func testArenaSlicesDoNotOverlap() throws {
        let a = try Arena(capacity: 1 << 20)
        let x = try a.reserve("x", bytes: 4096)
        let y = try a.reserve("y", bytes: 4096)
        XCTAssertGreaterThanOrEqual(abs(Int(bitPattern: y) - Int(bitPattern: x)), 4096)
        XCTAssertEqual(a.pointer("x"), x)
        XCTAssertNil(a.pointer("jamais réservé"))
    }

    // ── the bf16 → fp32 widening ───────────────────────────────────────────────────────

    func testWideningIsAShiftNotAConversion() {
        // A bf16 *is* the sixteen high bits of an fp32: no rounding, no table.
        let patterns: [(UInt16, Float)] = [
            (0x3F80, 1.0), (0xBF80, -1.0), (0x4000, 2.0), (0xC000, -2.0),
            (0x0000, 0.0), (0x3F00, 0.5), (0x7F80, .infinity), (0xFF80, -.infinity),
        ]
        let source = patterns.map { $0.0 }
        var destination = [Float](repeating: .nan, count: source.count)
        source.withUnsafeBytes { s in
            destination.withUnsafeMutableBytes { d in
                Widen.bfloat16ToFloat32(source: s.baseAddress!, destination: d.baseAddress!,
                                        count: patterns.count)
            }
        }
        for (i, (_, expected)) in patterns.enumerated() {
            XCTAssertEqual(destination[i].bitPattern, expected.bitPattern,
                           "pattern \(i): \(destination[i]) instead of \(expected)")
        }
    }

    func testWideningHandlesTheNonVectorizedTail() {
        // The kernel advances by eight then finishes one by one. A count that is not a multiple of
        // eight exercises both paths, and that is exactly where a badly bounded loop hides.
        for count in [1, 7, 8, 9, 15, 17, 33] {
            let source = [UInt16](repeating: 0x3F80, count: count)   // 1.0 everywhere
            var destination = [Float](repeating: .nan, count: count)
            source.withUnsafeBytes { s in
                destination.withUnsafeMutableBytes { d in
                    Widen.bfloat16ToFloat32(source: s.baseAddress!, destination: d.baseAddress!,
                                            count: count)
                }
            }
            XCTAssertTrue(destination.allSatisfy { $0 == 1.0 }, "count = \(count)")
        }
    }

    func testWideningAndNarrowingCompose() {
        // The published weights are bf16 values: narrowing then widening them must give the
        // same bits, otherwise the map is not bit-exact as announced.
        let values: [Float] = [1, -1, 0.5, 2, 0, 1024, -0.001953125]
        var narrow = [UInt16](repeating: 0, count: values.count)
        var wideValues = [Float](repeating: 0, count: values.count)
        values.withUnsafeBytes { v in
            narrow.withUnsafeMutableBytes { e in
                Widen.float32ToBfloat16(source: v.baseAddress!, destination: e.baseAddress!,
                                        count: values.count)
            }
        }
        narrow.withUnsafeBytes { e in
            wideValues.withUnsafeMutableBytes { l in
                Widen.bfloat16ToFloat32(source: e.baseAddress!, destination: l.baseAddress!,
                                        count: values.count)
            }
        }
        XCTAssertEqual(wideValues, values)
    }

    // ── the tiling ─────────────────────────────────────────────────────────────────────────

    func testPatchifyAndUnpatchifyAreInverses() {
        let channels = 16, side = 8, patch = 2
        let count = channels * side * side
        var latent = [Float](repeating: 0, count: count)
        for i in 0..<count { latent[i] = Float(i) }
        var tokens = [Float](repeating: 0, count: count)
        var returned = [Float](repeating: -1, count: count)
        latent.withUnsafeBufferPointer { l in
            tokens.withUnsafeMutableBufferPointer { j in
                Ops.patchify(l.baseAddress!, into: j.baseAddress!, channels: channels,
                             height: side, width: side, patch: patch)
            }
        }
        tokens.withUnsafeBufferPointer { j in
            returned.withUnsafeMutableBufferPointer { r in
                Ops.unpatchify(j.baseAddress!, into: r.baseAddress!, latentHeight: side,
                               latentWidth: side, patch: patch, channels: channels)
            }
        }
        XCTAssertEqual(returned, latent, "the tiling must be a permutation, hence reversible")
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

    func testAProfileHasNoRightToWriteTheImage() {
        // A profile tunes speed and room, never the image. It is enforced by `EngineSettings.outsideProfile`; this test is here so
        // that adding an image setting does not forget to register there.
        for forbidden in ["spectral", "sdpa_fp16", "vae_tuile", "amx_spin"] {
            XCTAssertTrue(EngineSettings.outsideProfile.contains(forbidden),
                          "\"\(forbidden)\" changes the image: a profile must not be able to write it")
        }
        // And the counterpart: `amx_fraction` changes the speed, never the image — each row of the
        // output is computed by a single engine — so a profile MUST be able to write it.
        XCTAssertFalse(EngineSettings.outsideProfile.contains("amx_fraction"))
    }

    func testTheSDPAHeadSplitCoversEveryHeadAndEveryRowOnce() {
        // Heads [0, g) on the GPU, [g, H) on the CPU, by blocks of query rows. A head computed
        // twice, or a row forgotten, would read as a plausible attention — hence the census.
        for heads in [24, 30, 32] {
            for fraction in [0.0, 0.01, 0.2, 0.4, 0.433, 0.5, 0.99, 1.0, 2.0] {
                let (gpu, cpu) = HeadSplitAttention.partition(heads: heads, fraction: fraction)
                XCTAssertEqual(gpu.lowerBound, 0)
                XCTAssertEqual(gpu.upperBound, cpu.lowerBound, "the two ranges must touch")
                XCTAssertEqual(cpu.upperBound, heads)
                XCTAssertGreaterThanOrEqual(gpu.count, 1, "the GPU keeps at least one head")
                XCTAssertEqual(cpu.count, min(heads - 1, Int((Double(heads) * fraction).rounded())))
                for sequence in [1024, 1056, 4128, 4115] {
                    let items = HeadSplitAttention.items(cpu: cpu, sequence: sequence, rows: HeadSplitAttention.blockRows)
                    // One assertion per case, not per item: ~10⁵ XCTAssert calls made this test slow.
                    let misplaced = items.first {
                        !cpu.contains($0.head) || $0.row % HeadSplitAttention.blockRows != 0 || $0.rows <= 0
                    }
                    XCTAssertNil(misplaced, "a CPU head, a block boundary fixed by position, at least one row")
                    guard misplaced == nil else { continue }
                    // The census by intervals, not by rows (10⁷ checked subscripts in a debug build):
                    // a head's blocks, sorted, must tile [0, S) end to end — a gap leaves a row out, an
                    // overlap computes one twice — and a GPU head has none.
                    var blocks = [[(row: Int, rows: Int)]](repeating: [], count: heads)
                    for item in items { blocks[item.head].append((item.row, item.rows)) }
                    for h in 0..<heads {
                        var end = 0, tiled = true
                        for b in blocks[h].sorted(by: { $0.row < $1.row }) {
                            if b.row != end { tiled = false; break }
                            end += b.rows
                        }
                        let expected = cpu.contains(h) ? sequence : 0
                        XCTAssertTrue(tiled && end == expected,
                                      "head \(h), S = \(sequence): every row exactly \(expected == 0 ? 0 : 1) time(s)")
                    }
                }
            }
        }
        // The product's fraction is frozen: Z-Image's 30 heads, 18 on the GPU, 12 on the CPU.
        XCTAssertEqual(HeadSplitAttention.partition(heads: 30, fraction: HeadSplitAttention.cpuFraction).cpu, 18..<30)
    }

    func testTheMachineIsReadInsteadOfQuoted() {
        // The notes announced "10 CPU (8P+2E)" for a machine that has 8. The fix was to read
        // `sysctl`; this test is here so that the reading stays wired up.
        let m = EngineSettings.Machine.detected()
        XCTAssertFalse(m.hardwareModel.isEmpty)
        XCTAssertNotEqual(m.hardwareModel, "unknown")
        XCTAssertGreaterThan(m.cores, 0)
        XCTAssertGreaterThan(m.ramMiB, 0)
        XCTAssertLessThanOrEqual(m.performanceCores, m.cores)
    }

    /// Anima Turbo's schedule, against the σ the oracle wrote (`goldens-anima-trajectory`,
    /// `FlowMatchEulerDiscreteScheduler`, shift 3) — copied here so the test opens no file.
    func testAnimaScheduleIsTheOraclesOne() {
        let expectedCounts: [Float] = [1, 0.954545, 0.9, 0.833333, 0.75, 0.642857, 0.5, 0.3, 0]
        let schedule = AnimaDiT.sigmas(steps: 8)
        XCTAssertEqual(schedule.count, expectedCounts.count)
        for (a, b) in zip(schedule, expectedCounts) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    }

    /// The 3D VAE reduced to a 2D VAE: each `[O, I, 3, kh, kw]` kernel keeps its **`t = 2`**
    /// slice, the only one that the two zero frames of the causal padding let work. The central
    /// slice (the reflex of a centered convolution) would give a plausible and wrong image; this
    /// test refuses it, and lets a weight that is not 5D pass through as is.
    func testTheVAEKeepsTheLastTemporalSlice() {
        // O = 2, I = 1, T = 3, 1×2: the value encodes (o, t, j) to read where each output comes from.
        var w: [Float] = []
        for o in 0..<2 { for t in 0..<3 { for j in 0..<2 { w.append(Float(o * 100 + t * 10 + j)) } } }
        let (slice, shape) = AnimaVAE.lastSlice(w, shape: [2, 1, 3, 1, 2])
        XCTAssertEqual(shape, [2, 1, 1, 2])
        XCTAssertEqual(slice, [20, 21, 120, 121])
        let flat = AnimaVAE.lastSlice([1, 2, 3], shape: [3])
        XCTAssertEqual(flat.values, [1, 2, 3]); XCTAssertEqual(flat.shape, [3])
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

    /// `patchify` then `unpatchify` give back the latent, and the token stores the channel as the
    /// SLOWEST: the index `(c·2 + ph)·2 + pw` of `_pack_latents`. A "fast channel" tiling (Z-Image's)
    /// would pass the round trip — it is the second assertion that refuses it.
    func testKrea2TilingStoresTheSlowestChannel() {
        // A rectangle (4 tall, 6 wide): swapping the axes would break the round trip.
        let channels = 3, height = 4, width = 6
        let latent = (0..<(channels * height * width)).map(Float.init)
        var tokens = [Float](repeating: -1, count: latent.count)
        var back = [Float](repeating: -1, count: latent.count)
        latent.withUnsafeBufferPointer { l in tokens.withUnsafeMutableBufferPointer {
            Krea2Ops.patchify(l.baseAddress!, into: $0.baseAddress!, channels: channels,
                              height: height, width: width) } }
        tokens.withUnsafeBufferPointer { t in back.withUnsafeMutableBufferPointer {
            Krea2Ops.unpatchify(t.baseAddress!, into: $0.baseAddress!, channels: channels,
                                height: height, width: width) } }
        XCTAssertEqual(back, latent)
        // token (0, 0): channel 0 → (0,0) (0,1) (1,0) (1,1), then channel 1 → +24
        XCTAssertEqual(Array(tokens[0..<8]), [0, 1, 6, 7, 24, 25, 30, 31])
        // token (0, 1) — the next one on the ROW: columns 2 and 3.
        XCTAssertEqual(Array(tokens[12..<16]), [2, 3, 8, 9])
    }

    /// Krea 2's RoPE: the text at the origin does not rotate (cos 1, sin 0), and the image carries
    /// `(0, h, w)` — the first 16 pairs (axis t) never rotate for an image.
    func testKrea2RoPELeavesTheTextStill() {
        let pairs = 64, text = 2, tiles = 2
        var table = [Float](repeating: .nan, count: (text + tiles * tiles) * pairs * 2)
        table.withUnsafeMutableBufferPointer {
            Krea2Rope.write(into: $0.baseAddress!, textRows: text, tilesHigh: tiles, tilesWide: tiles,
                            axes: [32, 48, 48], theta: 1000)
        }
        for p in 0..<(text * pairs) { XCTAssertEqual(table[2 * p], 1); XCTAssertEqual(table[2 * p + 1], 0) }
        let last = (text + 3) * pairs * 2           // the image (1, 1)
        for p in 0..<16 { XCTAssertEqual(table[last + 2 * p], 1); XCTAssertEqual(table[last + 2 * p + 1], 0) }
        XCTAssertEqual(table[last + 2 * 16], Float(cos(1.0)), accuracy: 1e-7, "axis h, frequency 1, position 1")
        XCTAssertFalse(table.contains { $0.isNaN })
    }

    /// GQA: the `q` head of index `h` reads the `k`/`v` head of index `h / 4` (`repeat_interleave`,
    /// not `repeat` — which would make it read `h mod 12`).
    func testKrea2GQARepeatsConsecutively() {
        let kvHeads = 2, group = 3, headDim = 1
        let source: [Float] = [10, 20]                 // one row, two k/v heads
        var wide = [Float](repeating: 0, count: kvHeads * group)
        source.withUnsafeBufferPointer { s in wide.withUnsafeMutableBufferPointer {
            Krea2Ops.expandKV(s.baseAddress!, into: $0.baseAddress!, rows: 1, kvHeads: kvHeads,
                              group: group, headDim: headDim) } }
        XCTAssertEqual(wide, [10, 10, 10, 20, 20, 20])
    }
}
