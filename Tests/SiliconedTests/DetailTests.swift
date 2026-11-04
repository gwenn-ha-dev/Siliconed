import XCTest
@testable import Siliconed

/// **The Detail Daemon's schedule against the node**. The values below were printed by
/// an oracle script which runs the node's
/// `make_detail_daemon_schedule` and `get_dd_schedule` verbatim (ComfyUI-Detail-Daemon `3394e44`).
final class DetailTests: XCTestCase {

    private let sets: [Detail.Schedule] = [
        .init(amount: 0.5),
        .init(amount: 1.0),
        .init(amount: 0.3, start: 0.1, end: 0.9, bias: 0.3, exponent: 2, startOffset: 0.05,
              endOffset: -0.1, fade: 0.2, smooth: false),
        .init(amount: -0.4, start: 0.9, end: 0.4, bias: 0.7, exponent: 0.5, smooth: true),
    ]

    /// `[set][steps]` for steps 6, 8, 9.
    private let multipliers: [[[Double]]] = [
        [[0, 0, 0.5, 0.24999999999999997, 0, 0],
         [0, 0, 0.12499999999999997, 0.37499999999999994, 0.5, 0.24999999999999997, 0, 0],
         [0, 0, 0, 0.24999999999999997, 0.5, 0.24999999999999997, 0, 0, 0]],
        [[0, 0, 1, 0.49999999999999994, 0, 0],
         [0, 0, 0.24999999999999994, 0.74999999999999989, 1, 0.49999999999999994, 0, 0],
         [0, 0, 0, 0.49999999999999994, 1, 0.49999999999999994, 0, 0, 0]],
        [[0.040000000000000008, 0.090000000000000011, 0.24000000000000005, 0, -0.080000000000000016, -0.080000000000000016],
         [0.040000000000000008, 0.040000000000000008, 0.24000000000000005, 0.10000000000000001, 0,
          -0.060000000000000012, -0.080000000000000016, -0.080000000000000016],
         [0.040000000000000008, 0.040000000000000008, 0.090000000000000011, 0.24000000000000005,
          0.10000000000000001, 0, -0.060000000000000012, -0.080000000000000016, -0.080000000000000016]],
        [[0, 0, -0.40000000000000002, 0, 0, 0],
         [0, 0, 0, -0.40000000000000002, 0, 0, 0, 0],
         [0, 0, 0, -0.40000000000000002, 0, 0, 0, 0, 0]],
    ]

    func testTheScheduleIsTheNodes() {
        for (s, set) in sets.enumerated() {
            for (k, steps) in [6, 8, 9].enumerated() {
                let got = Detail.multipliers(steps: steps, set)
                let want = multipliers[s][k]
                XCTAssertEqual(got.count, want.count)
                for (g, w) in zip(got, want) { XCTAssertEqual(g, w, accuracy: 1e-15, "set \(s), \(steps) steps") }
            }
        }
    }

    /// The σ the model receives, on Z-Image's schedule (diffusers 0.41: `linspace(1, 1/N, N)`, shift 3,
    /// then 0 — since 0.8.2): float32 to the bit. Recomputed on that schedule by the node's
    /// `make_detail_daemon_schedule` and wrapper semantics (float32 schedule, factor `1 − 0.1·m`).
    func testTheModelSigmasAreTheWrappers() {
        let cases: [(steps: Int, amount: Double, want: [Float])] = [
            (8, 0.5, [1, 0.9545454382896423, 0.8887500166893005, 0.8020833134651184, 0.7124999761581421, 0.6267856955528259, 0.5, 0.30000001192092896, 0]),
            (8, 1.0, [1, 0.9545454382896423, 0.8774999976158142, 0.7708333134651184, 0.6749999523162842, 0.6107142567634583, 0.5, 0.30000001192092896, 0]),
            (9, 1.0, [1, 0.9600000381469727, 0.9130435585975647, 0.8142856359481812, 0.7105263471603394, 0.6705882549285889, 0.5999999642372131, 0.46153849363327026, 0.27272728085517883, 0]),
            (6, 1.0, [1, 0.9375, 0.7714285254478455, 0.7124999761581421, 0.5999999642372131, 0.375, 0]),
        ]
        for c in cases {
            let σ = FlowMatchSchedule(steps: c.steps).sigmas
            let got = Detail.adjusted(σ, start: 0, .init(amount: c.amount))
            XCTAssertEqual(got.count, σ.count)
            XCTAssertEqual(got.last, 0)
            for (g, w) in zip(got, c.want) { XCTAssertEqual(g, w, accuracy: 2e-7 * max(1, w), "\(c.steps) steps, \(c.amount)") }
        }
        // img2img: the bell over the tail that runs, steps before `start` untouched.
        let σ = FlowMatchSchedule(steps: 8).sigmas
        let tail = Detail.adjusted(σ, start: 3, .init(amount: 1))
        XCTAssertEqual(Array(tail[..<3]), Array(σ[..<3]))
        for (g, w) in zip(tail[3...], [0.8333333134651184, 0.75, 0.5785713791847229, 0.5, 0.30000001192092896, 0] as [Float]) {
            XCTAssertEqual(g, w, accuracy: 2e-7)
        }
    }

    /// `normal` hands back the schedule itself; the two others only lower σ, never raise it, and
    /// leave the first step alone from 4 steps on.
    func testNormalIsTheIdentity() {
        for steps in [1, 4, 6, 8, 9] {
            let σ = FlowMatchSchedule(steps: steps).sigmas
            for bell in [Detail.Bell.zImage, .late] {
                XCTAssertEqual(Detail.normal.modelSigmas(σ, bell: bell).map(\.bitPattern), σ.map(\.bitPattern))
                for level in [Detail.more, .most] {
                    let m = level.modelSigmas(σ, bell: bell)
                    XCTAssertEqual(m.count, σ.count)
                    // (At 1 step the node puts its whole bell on the only step: faithful, and harmless.)
                    if steps >= 4 { XCTAssertEqual(m.first, σ.first) }
                    for (a, b) in zip(m, σ) { XCTAssertLessThanOrEqual(a, b) }
                }
            }
            // No bell (Anima, Krea 2 — they refuse Detail upstream): the σ themselves, whatever the level.
            for level in Detail.allCases {
                XCTAssertEqual(level.modelSigmas(σ, bell: nil).map(\.bitPattern), σ.map(\.bitPattern))
            }
        }
        for bell in [Detail.Bell.zImage, .late] {
            XCTAssertNil(Detail.normal.schedule(bell))
            XCTAssertGreaterThan(Detail.most.schedule(bell)!.amount, Detail.more.schedule(bell)!.amount)
        }
        // The Turbo's 8 steps (its only count): the last alone, σ 0.415 — σ 0.705 still decides the face.
        for level in [Detail.more, .most] {
            let s = level.schedule(.late)!
            XCTAssertEqual(Detail.multipliers(steps: 8, s), [0, 0, 0, 0, 0, 0, 0, s.amount])
        }
        XCTAssertNotEqual(Detail.more.modelSigmas(FlowMatchSchedule(steps: 8).sigmas, bell: .zImage), FlowMatchSchedule(steps: 8).sigmas)
    }

    /// **Z-Image's bell**: on its 8 evaluations, σ 0.75 at half and σ 0.643 in full, nothing
    /// else — neither the σ ≥ 0.833 where the pose is decided nor σ 0.5, where it only sows specks.
    func testZImagesBellSitsOnTwoEvaluations() {
        for level in [Detail.more, .most] {
            let s = level.schedule(.zImage)!
            let m = Detail.multipliers(steps: 8, s)
            for (g, w) in zip(m, [0, 0, 0, 0, 0.5, 1, 0, 0].map { $0 * s.amount }) { XCTAssertEqual(g, w, accuracy: 1e-15) }
        }
        let σ = FlowMatchSchedule(steps: 8).sigmas
        for (level, want) in [(Detail.more, [0.75 * 0.9, 0.6428571428571429 * 0.8]),
                              (.most, [0.75 * 0.875, 0.6428571428571429 * 0.75])] {
            let told = level.modelSigmas(σ, bell: .zImage)
            XCTAssertEqual(Array(told[..<4]).map(\.bitPattern), Array(σ[..<4]).map(\.bitPattern))
            XCTAssertEqual(Array(told[6...]).map(\.bitPattern), Array(σ[6...]).map(\.bitPattern))
            for (g, w) in zip(told[4...5], want) { XCTAssertEqual(Double(g), w, accuracy: 1e-6) }
        }
        // 1024²: half the amount, never another bell.
        XCTAssertEqual(Detail.more.modelSigmas(σ, bell: .zImage, scale: 0.5)[5], σ[5] * 0.9, accuracy: 1e-6)
    }

    /// **The amount shrinks as the image grows**, whatever the bell: ×1 at 512², ×½ at 1024², ×0.41
    /// at 1024×1536 — the calibrated amount times the scale, to the bit, on both bells.
    func testTheAmountFollowsTheSide() {
        XCTAssertEqual(Detail.scale(width: 512, height: 512), 1)
        XCTAssertEqual(Detail.scale(width: 1024, height: 1024), 0.5)
        XCTAssertEqual(Detail.scale(width: 1024, height: 1536), 0.4082, accuracy: 1e-4)
        let σ = FlowMatchSchedule(steps: 8).sigmas
        for bell in [Detail.Bell.zImage, .late] {
            var half = Detail.most.schedule(bell)!
            half.amount *= 0.5
            XCTAssertEqual(Detail.most.modelSigmas(σ, bell: bell, scale: 0.5).map(\.bitPattern),
                           Detail.adjusted(σ, start: 0, half).map(\.bitPattern))
        }
        XCTAssertLessThan(Detail.most.modelSigmas(σ, bell: .late)[7], Detail.most.modelSigmas(σ, bell: .late, scale: 0.5)[7])
        XCTAssertEqual(Detail.normal.modelSigmas(σ, bell: .zImage, scale: 0.5).map(\.bitPattern), σ.map(\.bitPattern))
    }

    /// The PNG says it only when it is not `normal`, and gives it back; an unknown value is said.
    func testThePNGCarriesTheDetail() throws {
        func render(_ detail: Detail) -> Engine.Render {
            Engine.Render(model: "z-image", prompt: "woman posing in a library", seed: 42, steps: 8, loras: [],
                          loraSummary: nil, image: ImageRGB(pixels: [Float](repeating: 0, count: 3 * 4), height: 2, width: 2), evaluations: 8, spectral: 0, strength: nil,
                          startStep: 0, startSigma: 1, timings: .init(), footprints: .init(), tokens: 1, detail: detail)
        }
        XCTAssertNil(render(.normal).metadata["detail"])
        for level in Detail.allCases {
            let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: PNG.text(try render(level).png())))
            XCTAssertEqual(recipe.detail, level)
            XCTAssertEqual(recipe.unreadable, [])
        }
        var m = render(.most).metadata
        m["detail"] = "extreme"
        let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: m))
        XCTAssertEqual(recipe.detail, .normal)
        XCTAssertEqual(recipe.unreadable, ["detail"])
    }

    /// **Anima and Krea 2 offer no Detail** (Anima: mosaic blocks or nothing visible; Krea 2: another
    /// pose and face, no texture); the others do, and the chain refuses More and Most on them before
    /// any computation (`Request.Failure.detail`).
    func testAnimaAndKrea2RefuseDetail() {
        XCTAssertFalse(AnimaDenoisingModule(map: "").acceptsDetail)
        XCTAssertFalse(Krea2DenoisingModule(map: "").acceptsDetail)
        XCTAssertTrue(ZImageDenoisingModule(map: "").acceptsDetail)
        XCTAssertEqual(EngineError(Request.Failure.detail(model: "anima")), .detailUnsupported(model: "anima"))
    }
}
