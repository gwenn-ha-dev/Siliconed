import Foundation
import XCTest
@testable import Siliconed

/// **The « Accelerators » switch, on a real map**: an M5 Pro's model (`Fixtures/capacity/m5-pro-model.json`, the model of a capacity
/// map measured on 2/11/2026), whose lines publish the ANE (fp16, int8), the activation rounded to bf16 (`bf16act`) and the attention on
/// fp16 and bf16 operands (`tiles16`, `tilesbf16`). On, the plan is the one the solver gave before the switch existed, class by class; off,
/// no class of any family takes the ANE, a rounded activation or a half-precision attention, and the process brings none of them up.
final class AcceleratorsTests: XCTestCase {
    static func model() throws -> CapacityModel {
        let file = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/capacity/m5-pro-model.json")
        return try JSONDecoder().decode(CapacityModel.self, from: Data(contentsOf: file))
    }

    /// Every denoising class of every family's every version at 1024² and 512²: the questions a render puts to the plan.
    static func classes() -> [(ProductKey?, AttentionKey?, Bool)] {
        var out: [(ProductKey?, AttentionKey?, Bool)] = []
        for family in Family.allCases {
            for variant in Variant.allCases {
                for side in [512, 1024] {
                    let request = CapacityClasses.Request(family: family, variant: variant, width: side, height: side)
                    for c in CapacityClasses.classes(for: request) ?? [] where c.phase == .denoising {
                        switch c.shape {
                        case let .product(k, n, rows, format):
                            out.append((ProductKey(k: k, n: n, m: rows, weight: format, phase: c.phase), nil, false))
                        case let .attention(s, keys, heads, kv, dim, cross):
                            out.append((nil, AttentionKey(sequence: s, keys: keys, heads: heads, kvHeads: kv, headDim: dim, cross: cross),
                                        CapacitySolver.triesFlash(family: family, cross: cross)))
                        }
                    }
                }
            }
        }
        return out
    }

    /// On: the same answers as the solver gives under the constraints a render had before the switch (the ANE where it was brought up,
    /// fp16 operands its permission, the rounding always allowed) — the same bits as before.
    func testOnIsThePlanAsBefore() throws {
        let model = try Self.model()
        let base = CapacityPlan(model: model)
        let classes = Self.classes()
        XCTAssertGreaterThan(classes.count, 50)
        for ane in [true, false] {
            let plan = base.forDenoiser(.on(neuralEngine: ane))
            XCTAssertTrue(plan === base.restricted(allowingANE: ane), "on is the plan the ANE's switch gave")
            let before = CapacitySolver.Constraints(allowsANE: ane)
            for (product, attention, flash) in classes {
                if let product {
                    XCTAssertEqual(plan.assign(product),
                                   CapacitySolver.product(product, model: model, constraints: before)?.assignment ?? .gpuAlone(product.m),
                                   "\(product)")
                }
                if let attention {
                    XCTAssertEqual(plan.assign(attention, flash: flash),
                                   CapacitySolver.attention(attention, flash: flash, model: model, constraints: before)?.assignment ?? .defaults(attention),
                                   "\(attention)")
                }
            }
        }
        XCTAssertEqual(base.forDenoiser(.on(neuralEngine: true)).identity, base.identity)
    }

    /// The map does use them, on: a render with the switch on is not fp32 on this Mac (else « off » would prove nothing).
    func testOnThisMapTheyAreUsed() throws {
        let plan = CapacityPlan(model: try Self.model()).forDenoiser(.on(neuralEngine: true))
        let classes = Self.classes()
        XCTAssertTrue(plan.resources.halfPrecision)
        XCTAssertTrue(classes.contains { $0.0.map { plan.assign($0).roundsActivation } ?? false }, "some product rounds its activation")
        XCTAssertTrue(classes.contains { c in c.1.map { a in
            let k = plan.assign(a, flash: c.2); return k.kernel == .tiles16 || k.kernel == .tilesbf16 || k.gpuKernel == .tiles16 || k.gpuKernel == .tilesbf16
        } ?? false }, "some attention runs on half-precision operands")
        XCTAssertEqual(AcceleratorOffer(plan: plan), AcceleratorOffer(neuralEngine: true, tensorUnits: ["bf16", "fp16"]))
        // Without the ANE (a map that gives it nothing): fp16 operands were its permission, bf16 stays.
        XCTAssertEqual(AcceleratorOffer(plan: plan.restricted(allowingANE: false)), AcceleratorOffer(tensorUnits: ["bf16"]))
    }

    /// **Off: fp32 throughout.** No ANE row, no rounded activation, no attention on half-precision operands, for any class of any family;
    /// the engines that would round are not brought up; what stays is exact against the fp32 product (a published weight read as it is,
    /// beside an fp32 operand).
    func testOffIsFP32Throughout() throws {
        let base = CapacityPlan(model: try Self.model())
        for start in [base, base.restricted(allowingANE: false)] {
            let plan = start.restricted(accelerators: false)
            XCTAssertEqual(plan.identity, base.forDenoiser(.off).identity, "one plan for the render cache")
            XCTAssertFalse(plan.allowsANE); XCTAssertFalse(plan.allowsHalfPrecision)
            XCTAssertFalse(plan.usesANE)
            XCTAssertFalse(plan.resources.halfPrecision)
            XCTAssertFalse(plan.resources.roundedActivations); XCTAssertFalse(plan.resources.halfAttention)
            XCTAssertFalse(plan.resources.aneInt8)
            XCTAssertEqual(plan.gpuHalfPrecisions, [])
            XCTAssertEqual(AcceleratorOffer(plan: plan), .none)
            XCTAssertTrue(plan.identity.hasSuffix(" no-ane fp32"))
            XCTAssertNotEqual(plan.identity, base.identity)
            XCTAssertNotEqual(plan.identity, base.restricted(allowingANE: false).identity)
            var native = false
            for (product, attention, flash) in Self.classes() {
                if let product {
                    let a = plan.assign(product)
                    XCTAssertEqual(a.ane, 0, "\(product)"); XCTAssertFalse(a.roundsActivation, "\(product)")
                    if a.path != .widened { native = true }
                }
                if let attention {
                    let a = plan.assign(attention, flash: flash)
                    for k in [a.kernel, a.gpuKernel] { XCTAssertTrue(k != .tiles16 && k != .tilesbf16, "\(attention): \(a)") }
                }
            }
            XCTAssertTrue(native, "the published weights read as they are stay: bf16, int8, Q8_0 beside an fp32 operand")
        }
    }

    /// A forcing that names them (a check's strictest case), off: the same weights read as published, the operand fp32, the tiles in fp32.
    func testAForcingIsFoldedToFP32() {
        let forced = CapacityPlan.forcing(neuralEngine: true, bf16Activations: true, tilesbf16: true)
        XCTAssertTrue(forced.resources.halfPrecision)
        let off = forced.restricted(accelerators: false)
        XCTAssertFalse(off.resources.halfPrecision)
        XCTAssertTrue(off.identity.hasSuffix(" fp32"))
        let key = ProductKey(k: 3840, n: 3840, m: 4096, weight: .bf16)
        XCTAssertTrue(forced.assign(key).roundsActivation)
        let a = off.assign(key)
        XCTAssertEqual(a.path, .bf16); XCTAssertFalse(a.roundsActivation); XCTAssertEqual(a.ane, 0)
        let attention = AttentionKey(sequence: 4096, heads: 30, headDim: 128)
        XCTAssertEqual(forced.assign(attention).kernel, .tilesbf16)
        XCTAssertEqual(off.assign(attention).kernel, .tiles)
    }

    /// The request's switch reaches the admission: off is fp32, on brings the ANE up where the plan uses it.
    func testTheSwitchIsResolvedAtAdmission() {
        XCTAssertEqual(Accelerators(requested: false, usesANE: true), .off)
        XCTAssertEqual(Accelerators(requested: true, usesANE: true), .on(neuralEngine: true))
        XCTAssertEqual(Accelerators(requested: true, usesANE: false), .on(neuralEngine: false))
        XCTAssertFalse(Accelerators.off.neuralEngine); XCTAssertFalse(Accelerators.off.isOn)
        var request = Request("x", resolution: 512)
        XCTAssertTrue(request.accelerators, "on by default")
        request.neuralEngine = false
        XCTAssertFalse(request.accelerators, "the former name sets the switch")
    }

    /// The render cache tells a K/V written in fp32 throughout from one written with the ANE off only.
    func testTheCacheKeyTellsThemApart() {
        XCTAssertNotEqual(RenderCache.numerics(.off), RenderCache.numerics(.on(neuralEngine: false)))
    }

    /// Off is written in the PNG and read back; on writes nothing new.
    func testThePNGSaysOff() throws {
        var render = Engine.Render(model: "krea2", prompt: "woman posing in a library", seed: 42, steps: 8, loras: [],
                                   loraSummary: nil, image: ImageRGB(pixels: [Float](repeating: 0, count: 3 * 4), height: 2, width: 2),
                                   evaluations: 8, sketch: nil, spectral: 0, strength: nil,
                                   startStep: 0, startSigma: 1, timings: .init(), footprints: .init(), tokens: 1)
        XCTAssertNil(render.metadata["accelerators"])
        XCTAssertNil(Engine.Render.Recipe(metadata: render.metadata)?.accelerators)
        render.accelerators = false
        let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: PNG.text(try render.png())))
        XCTAssertEqual(recipe.accelerators, false)
        XCTAssertEqual(recipe.unreadable, [])
    }
}
