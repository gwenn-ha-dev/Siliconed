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

    /// The σ the model receives, on Z-Image's schedule (diffusers' `…, 0, 0`): float32 to the bit.
    func testTheModelSigmasAreTheWrappers() {
        let cases: [(steps: Int, amount: Double, want: [Float])] = [
            (8, 0.5, [1, 0.947368383, 0.871323466, 0.769999921, 0.657692254, 0.531818151, 0.333333313, 0]),
            (8, 1.0, [1, 0.947368383, 0.860294104, 0.73999995, 0.623076856, 0.518181741, 0.333333313, 0]),
            (9, 1.0, [1, 0.954545438, 0.899999976, 0.791666627, 0.674999952, 0.610714257, 0.5, 0.300000012, 0]),
            (6, 1.0, [1, 0.923076987, 0.73636359, 0.633333266, 0.428571403, 0]),
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
        for (g, w) in zip(tail[3...], [0.799999952, 0.692307651, 0.49090904, 0.333333313, 0] as [Float]) {
            XCTAssertEqual(g, w, accuracy: 2e-7)
        }
    }

    /// `normal` hands back the schedule itself; the two others only lower σ, never raise it, and
    /// leave the first step alone from 4 steps on.
    func testNormalIsTheIdentity() {
        for steps in [1, 4, 6, 8, 9] {
            let σ = FlowMatchSchedule(steps: steps).sigmas
            XCTAssertEqual(Detail.normal.modelSigmas(σ).map(\.bitPattern), σ.map(\.bitPattern))
            for level in [Detail.more, .most] {
                let m = level.modelSigmas(σ)
                XCTAssertEqual(m.count, σ.count)
                // (At 1 step the node puts its whole bell on the only step: faithful, and harmless.)
                if steps >= 4 { XCTAssertEqual(m.first, σ.first) }
                for (a, b) in zip(m, σ) { XCTAssertLessThanOrEqual(a, b) }
            }
        }
        for bell in [Detail.Bell.standard, .late] {
            XCTAssertNil(Detail.normal.schedule(bell))
            XCTAssertGreaterThan(Detail.most.schedule(bell)!.amount, Detail.more.schedule(bell)!.amount)
        }
        // Qwen-Image-2.1's 6 steps: the late bell touches only the 5th.
        XCTAssertEqual(Detail.multipliers(steps: 6, Detail.more.schedule(.late)!), [0, 0, 0, 0, 1, 0])
        XCTAssertNotEqual(Detail.more.modelSigmas(FlowMatchSchedule(steps: 8).sigmas), FlowMatchSchedule(steps: 8).sigmas)
    }

    /// **The standard bell shrinks as the image grows**: ×1 at 512², ×½ at 1024², ×0.41
    /// at 1024×1536 (Most 0.82) — Most at 1024² is More at 512², to the bit; the late bell (Qwen) too.
    func testTheAmountFollowsTheSide() {
        XCTAssertEqual(Detail.scale(width: 512, height: 512), 1)
        XCTAssertEqual(Detail.scale(width: 1024, height: 1024), 0.5)
        XCTAssertEqual(Detail.scale(width: 1024, height: 1536), 0.4082, accuracy: 1e-4)
        let σ = FlowMatchSchedule(steps: 8).sigmas
        XCTAssertEqual(Detail.most.modelSigmas(σ, scale: 0.5).map(\.bitPattern), Detail.more.modelSigmas(σ).map(\.bitPattern))
        let q: [Float] = [1, 0.963, 0.923, 0.837, 0.632, 0.36, 0]
        XCTAssertLessThan(Detail.most.modelSigmas(q, bell: .late)[4], Detail.most.modelSigmas(q, bell: .late, scale: 0.5)[4])
        XCTAssertEqual(Detail.normal.modelSigmas(σ, scale: 0.5).map(\.bitPattern), σ.map(\.bitPattern))
    }

    /// The PNG says it only when it is not `normal`, and gives it back; an unknown value is said.
    func testThePNGCarriesTheDetail() throws {
        func render(_ detail: Detail) -> Engine.Render {
            Engine.Render(model: "z-image", prompt: "woman posing in a library", seed: 42, steps: 8, loras: [],
                          loraSummary: nil, image: ImageRGB(pixels: [Float](repeating: 0, count: 3 * 4), height: 2, width: 2),
                          reproducible: true, evaluations: 7, sketch: nil, spectral: 0, strength: nil,
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
}
