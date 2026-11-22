import Foundation
import SiliCore
import XCTest
@testable import Siliconed

/// Qwen-Image-2.1's DiT on the engine, without a map: what it reads, how its rows are laid out, what its stage declares.
final class QwenImage21LoopTests: XCTestCase {
    typealias F = QwenImage21Functions
    typealias Loop = QwenImage21Loop

    /// Every key a path reads is a tensor of the published DiT, at its shape; and every tensor is read by the bf16 path.
    func testEveryKeyReadIsAPublishedTensor() {
        let tensors = F.tensors()
        XCTAssertEqual(tensors.count, 9 + 32 * 8)
        XCTAssertEqual(tensors["transformer_blocks.31.img_mlp.gate_up.weight"], [24576, 4096])
        XCTAssertEqual(tensors["transformer_blocks.0.img_mlp.out.weight"], [4096, 12288])
        for path in [F.Path.bfloat, .shared] {
            let keys = (0..<32).flatMap { F.layer($0, path: path) } + F.others()
            for (name, key, recipe) in keys {
                if case .target = recipe { continue }
                XCTAssertNotNil(tensors[recipe == .codes ? key + ".weight" : key], "\(path) \(name): \(key)")
            }
        }
        let read = Set(((0..<32).flatMap { F.layer($0, path: .bfloat) } + F.others()).map(\.key))
        XCTAssertEqual(read, Set(tensors.keys))
    }

    /// A LoRA adapts every product, and the fused SwiGLU input by either half under diffusers' names.
    func testTheLoRAModulesCoverEveryProductAndTheSwiGLUHalves() {
        let modules = F.loraModules()
        for (_, shape, name) in F.layerModules() {
            XCTAssertEqual(modules[name].map { [$0.out, $0.inn] }, shape, name)
        }
        XCTAssertEqual(modules["transformer_blocks.7.img_mlp.gate_layer"].map { [$0.out, $0.inn] }, [12288, 4096])
        XCTAssertEqual(modules["transformer_blocks.7.img_mlp.proj"].map { [$0.out, $0.inn] }, [12288, 4096])
        XCTAssertEqual(modules["transformer_blocks.7.img_mlp.proj"]?.product, "transformer_blocks.7.img_mlp.gate_up")
        XCTAssertEqual(modules["transformer_blocks.7.img_mlp.proj"]?.row, 12288)
        XCTAssertEqual(modules["transformer_blocks.7.img_mlp.gate_layer"]?.row, 0)
        XCTAssertEqual(modules["img_in"].map { [$0.out, $0.inn] }, [4096, 64])
    }

    /// The prefix within a palier of 256 keys, the target within a multiple of 64.
    func testTheRowsOfEachPass() {
        let g = Loop.Geometry(QwenImage21DiT.Sequence(text: 19, height: 48, width: 40))
        XCTAssertEqual([g.prefix, g.prefixKeys, g.target, g.targetKeys, g.keys, g.rows], [19, 256, 1920, 1920, 2176, 1920])
    }

    /// What each arithmetic declares: its stream, the layer's keys a tensor each per group at the prefix's rows, the two
    /// passes bound to their own keys, tables and modulation, the ANE's gains tensors in the shared forms only.
    func testTheDeclarationOfEachArithmetic() {
        let g = Loop.Geometry(QwenImage21DiT.Sequence(text: 19, height: 32, width: 32))
        for (arithmetic, form) in [(Arithmetic.standard, "bfloat"), (.shared(.codes, .gpu), "shared"), (.shared(.codes, .inFlight), "shared"),
                                   (.shared(.half, .inFlight), "shared")] {
            let shared = form == "shared"
            let d = Loop.declaration(g, arithmetic)
            XCTAssertEqual(d.groups.count, 32)
            XCTAssertEqual(d.groups.map(\.form), Array(repeating: form, count: 32))
            XCTAssertEqual(d.once, d.cycle)
            XCTAssertEqual(d.rows, g.rows)
            XCTAssertEqual(d.passes.map(\.name), ["prefix", "target"])
            XCTAssertEqual(d.passes.map(\.limit), [g.prefixKeys, g.targetKeys])
            XCTAssertEqual(d.passes[0].bind["k"], .rows("kp"))
            XCTAssertEqual(d.passes[1].bind["k"], .rows("kt"))
            XCTAssertEqual(d.passes[0].bind["scale1"], .row("mod0", 0))
            XCTAssertEqual(d.passes[1].bind["scale1"], shared ? nil : .row("mod", 0))
            let keys = d.perRow.first { $0.name == "layer.31.kp" }
            XCTAssertNil(d.perRow.first { $0.name == "kp" })
            XCTAssertEqual(keys?.rows, g.prefixKeys)
            XCTAssertEqual(d.tensors.contains { $0.name == "scale1" }, shared)
            XCTAssertEqual(d.forms[0].steps.contains { $0.unit == .either }, shared)
            XCTAssertEqual(d.edges.keys.sorted(), ["embed-prefix", "embed-target", "final", "timestep"])
        }
    }

    /// Every function each form declares is written whole, its graph built and its arguments the names the declaration
    /// binds — without a LoRA and under one adapting every product: what would trap at a render's first load traps here.
    func testEveryFunctionIsWritten() throws {
        let g = Loop.Geometry(QwenImage21DiT.Sequence(text: 19, height: 32, width: 32))
        let every = Dictionary(uniqueKeysWithValues: (F.outerLoRA() + F.layerLoRA(0)).map { ($0.name, 32) })
        for ranks in [F.Ranks(), every] {
            for arithmetic in [Arithmetic.standard, .shared(.codes, .gpu), .shared(.codes, .inFlight), .shared(.half, .inFlight)] {
                let d = Loop.declaration(g, arithmetic, ranks: ranks)
                let path = Loop.path(arithmetic)
                for step in d.forms[0].steps {
                    for rows in [64, 1024] {
                        let m = try d.forms[0].module(step.name, Block.Build(rows: rows))
                        XCTAssertFalse(m.outputs.isEmpty, "\(path) \(step.name)")
                        XCTAssertEqual(F.arguments(step.name, path: path, weights: arithmetic.weights, rows: rows, prefix: g.prefixKeys, target: g.targetKeys, ranks: ranks).map(\.name),
                                       m.argumentNames, "\(path) \(step.name)")
                    }
                }
                for (name, build) in d.edges.sorted(by: { $0.key < $1.key }) {
                    XCTAssertFalse(try build(Block.Build(rows: 0)).outputs.isEmpty, "\(path) \(name)")
                }
                // The factors: outside the layers by their names, a layer's under its group.
                let factors = d.tensors.filter { $0.name.hasSuffix("_down") || $0.name.hasSuffix("_up") }
                XCTAssertEqual(factors.count, ranks.isEmpty ? 0 : 2 * (F.outerLoRA().count + 32 * F.layerLoRA(0).count), "\(path)")
                if !ranks.isEmpty, let down = factors.first(where: { $0.name == "layer.5.gate_up_down" }) {
                    XCTAssertEqual(down.bytes, 4096 * 32 * 2)
                }
            }
        }
    }
}
