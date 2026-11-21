import SiliCore
import XCTest
@testable import Siliconed

/// **Krea 2's functions, without a byte of weights**: every function the DiT and its text fusion load, written on each
/// path (bf16; the Compact's shared forms), with and without a LoRA's low rank, reads only arguments it declares and writes
/// outputs of the shapes its callers' views take; what a file is recognized against and what a LoRA adapts, from the config.
final class Krea2FamilyTests: XCTestCase {
    let d = Krea2Functions.d

    func outputs(_ m: Aicode.Module) -> [String: [Int]] {
        Dictionary(uniqueKeysWithValues: m.outputs.map { ($0.0, Aicode.shape(m.body.types[$0.1]).shape) })
    }

    /// The Standard's functions at 512² (1024 image rows, a palier of 512), at the loop's tile sizes, bare and under a LoRA.
    func testTheStandardsFunctionsAreWritten() {
        let (n, keys) = (1024, 1536)
        for ranks in [[:], ["wq": 32, "wv": 64, "gate": 32, "down": 32]] as [[String: Int]] {
            XCTAssertEqual(outputs(Krea2Functions.module("timestep", rows: n, keys: keys)), ["temb": [1, d], "temb_mod": [6, d]])
            XCTAssertEqual(outputs(Krea2Functions.module("embed", rows: n, keys: keys)), ["x": [n, d]])
            XCTAssertEqual(outputs(Krea2Functions.module("final", rows: n, keys: keys)), ["velocity": [n, 64]])
            for rows in [1024, 64] {
                XCTAssertEqual(outputs(Krea2Functions.module("pre", rows: rows, keys: keys, ranks: Krea2Session.ranks("pre", ranks))),
                               ["q": [1, 48, rows, 128], "k": [1, rows, 12, 128], "v": [1, rows, 12, 128]])
                XCTAssertEqual(outputs(Krea2Functions.module("post", rows: rows, keys: keys, ranks: Krea2Session.ranks("post", ranks))), ["y": [rows, d]])
                XCTAssertEqual(outputs(Krea2Functions.module("post-taps", rows: rows, keys: keys, ranks: Krea2Session.ranks("post", ranks))),
                               ["y": [rows, d], "attention": [rows, d], "feed": [rows, d]])
            }
        }
    }

    /// The shared forms at 512 rows, on codes and on halves, bare and under a LoRA: what the GPU and the ANE call.
    func testTheSharedFormsAreWritten() {
        let (r, keys, kv) = (512, 1536, 1536)
        for weights in [SharedWeights.codes, .half] {
            for ranks in [[:], ["wq": 32, "wk": 32, "wo": 64, "mgate": 32, "up": 32, "down": 32]] as [[String: Int]] {
                XCTAssertEqual(outputs(Krea2Shared.module("pre", rows: r, keys: keys, weights: weights, ranks: Krea2Session.ranks("pre", ranks))),
                               ["qh": [1, r, d], "kh": [1, r, kv], "vh": [1, r, kv], "gh": [1, r, d], "pre_factor": [1, r, 1]])
                XCTAssertEqual(outputs(Krea2Shared.module("out", rows: r, keys: keys, weights: weights, ranks: Krea2Session.ranks("out", ranks))),
                               ["y": [1, r, d], "y_factor": [1, r, 1]])
                XCTAssertEqual(outputs(Krea2Shared.module("ffn", rows: r, keys: keys, weights: weights, ranks: Krea2Session.ranks("ffn", ranks))),
                               ["y": [1, r, d], "y_factor": [1, r, 1]])
            }
        }
    }

    /// The text fusion's, bf16 on both files: 32 tokens' twelve taps a tile, the palier for the refiners and `txtmlp`.
    func testTheFusionsFunctionsAreWritten() {
        let (w, t, p) = (Krea2Functions.textWidth, Krea2Functions.tokens, 512)
        for ranks in [[:], ["wq": 32, "up": 64]] as [[String: Int]] {
            XCTAssertEqual(outputs(Krea2Functions.module("layerwise", rows: 12 * t, keys: 12 * t, ranks: ranks)), ["y": [12 * t, w]])
            XCTAssertEqual(outputs(Krea2Functions.module("refiner", rows: p, keys: p, ranks: ranks)), ["y": [p, w]])
        }
        XCTAssertEqual(outputs(Krea2Functions.module("projector", rows: t, keys: t)), ["y": [t, w]])
        XCTAssertEqual(outputs(Krea2Functions.module("txtmlp", rows: p, keys: p)), ["y": [p, d]])
    }

    /// An asset's name says its LoRA's ranks, in one order.
    /// **No layout this Mac keeps reaches Krea 2's functions** (`CodesLayout`, `ThisMac.forms`): the functions are written
    /// for one layout each, so a plan that took another (the kept `out-in` of a former measure) would hand them weights
    /// laid otherwise — it crashed the worker before 78a6f09, a `precondition` on the codes of the W8A16 path since gone.
    /// The Standard's and the fusion's products are `bfloat16`, read as the file lays them `[out, in]` whatever is kept
    /// (the function transposes); the shared forms' codes are surfaces, which the plan lays `[in, out]` whatever is kept.
    func testNoKeptLayoutReachesTheFunctions() {
        let g = Krea2Session.Geometry(latentHeight: 64, latentWidth: 64, text: 512)
        func free(_ r: Recipe) -> Bool {
            switch r {
            case .bfloat16, .float, .half, .halfLast, .target: true
            case .codes, .product, .halfTransposed, .floatTransposed, .bfloatTransposed: false
            }
        }
        for ranks in [[:], ["wq": 32, "down": 32]] as [[String: Int]] {
            let standard = Krea2Session.declaration(g, .standard, ranks: ranks, taps: true)
            XCTAssertFalse(standard.forms.contains { $0.steps.contains { $0.unit == .either } })
            for e in standard.residents + standard.groups.flatMap({ $0.weights }) { XCTAssertTrue(free(e.recipe), "the Standard's \(e.name): \(e.recipe)") }
            let fusion = Krea2Fusion.declaration(palier: 512, ranks: ["layerwise": ranks, "refiner": ranks])
            for e in fusion.residents + fusion.groups.values.flatMap({ $0 }) { XCTAssertTrue(free(e.recipe), "the fusion's \(e.name): \(e.recipe)") }
            // The shared forms' products: the codes laid [in, out] by the plan, or the Standard's rounded to f16 [in, out].
            for arithmetic in [Arithmetic.shared(.codes, .gpu), .shared(.codes, .told(512)), .shared(.codes, .inFlight), .shared(.half, .inFlight)] {
                let shared = Krea2Session.declaration(g, arithmetic, ranks: ranks, taps: false)
                XCTAssertTrue(shared.forms.contains { $0.steps.contains { $0.unit == .either } }, "\(arithmetic): the shared forms")
                for e in shared.residents + shared.groups.flatMap({ $0.weights }) where !free(e.recipe) {
                    XCTAssertEqual(e.recipe, arithmetic.weights == .codes ? .codes : .halfTransposed, "\(arithmetic): \(e.name): \(e.recipe)")
                }
            }
        }
    }

    /// **The declaration is whole**: every group a block of the form it names, its passes inside the keys, the stream and
    /// its factors, the tapped functions only where a step taps; each step's function is written at the rows a call takes.
    func testTheDeclarationIsWritten() throws {
        let g = Krea2Session.Geometry(latentHeight: 64, latentWidth: 64, text: 37)
        XCTAssertEqual(g.keys, 1024 + 512)
        for arithmetic in [Arithmetic.standard, .shared(.codes, .gpu), .shared(.half, .inFlight)] {
            let ranks = ["wq": 32, "down": 32]
            let d = Krea2Session.declaration(g, arithmetic, ranks: ranks, taps: true)
            XCTAssertTrue(d.tapped)
            XCTAssertEqual(d.groups.map(\.name), Krea2Session.blocks)
            XCTAssertEqual(d.cycle, Krea2Session.blocks)
            XCTAssertEqual(d.passes.map(\.rows), [0..<(1024 + 37)])
            XCTAssertEqual(d.perRow.first { $0.name == "x" }?.factor, arithmetic == .standard ? nil : "x_factor")
            let form = d.forms[0]
            XCTAssertEqual(form.steps.filter(\.taps).map(\.name), arithmetic == .standard ? ["post"] : ["out", "ffn"])
            for step in form.steps {
                for tapped in [false, true] {
                    let m = try form.module(step.name, Block.Build(rows: 1088, codes: .inOut, tapped: tapped && step.taps))
                    let names = Set(m.outputs.map(\.0))
                    XCTAssertEqual(names.contains("attention") || names.contains("feed"), tapped && step.taps, "\(step.name) tapped \(tapped)")
                }
            }
            XCTAssertNotNil(d.tensors.first { $0.name == "blocks.27.wq_down" })
            XCTAssertNil(d.tensors.first { $0.name == "blocks.27.wk_down" })
        }
    }

    func testAnAssetNamesItsRanks() {
        XCTAssertEqual(Krea2Functions.named([:]), "")
        XCTAssertEqual(Krea2Functions.named(["wq": 32, "down": 64]), "-lora-down64-wq32")
    }

    /// The published names: the blocks' and the fusion's blocks' products are adapted; a norm, a block past the 28th,
    /// the projector and the embedders, which no function adapts, are not.
    func testWhatALoRAAdapts() {
        let modules = Krea2Transformer.loraModules()
        for name in ["blocks.0.attn.wq", "blocks.27.attn.gate", "blocks.0.mlp.up", "txtfusion.refiner_blocks.0.attn.wo", "txtfusion.layerwise_blocks.1.mlp.down"] {
            XCTAssertNotNil(modules[name], name)
        }
        for name in ["blocks.28.attn.wq", "blocks.0.prenorm", "txtfusion.projector", "txtmlp.1", "first"] { XCTAssertNil(modules[name], name) }
        XCTAssertEqual(modules.count, 28 * 8 + 4 * 8)
        XCTAssertEqual(modules["blocks.0.attn.wk"].map { [$0.out, $0.inn] }, [1536, d])
        XCTAssertEqual(modules["blocks.0.mlp.down"].map { [$0.out, $0.inn] }, [d, 16384])
        XCTAssertEqual(Krea2Transformer.tensors()["blocks.0.attn.wq.weight"], [d, d])
        XCTAssertEqual(Krea2Transformer.tensors()["last.modulation.lin"], [2, d])
    }
}
