import CoreAI
import Foundation
import Metal
import SiliCore
import Testing

/// **A row at a time through layers read again, their reserve never read on this Mac: the layers end up kept.** A
/// language model that writes runs its layers once over the prompt (`forward`), then one row a token through all of
/// them (`through`). The stage keeps from the first signal of the cycle's third turn when it has read no reserve for
/// these functions (`Stage.signal`); a row through the layers must give that signal as a pass does, at every turn,
/// until the stage keeps. Measured 20/11 on the mini before the fix: `through` gave it once, on its first row, so a
/// writing whose functions this Mac had not met kept nothing and read its 36 layers through the ring at every token
/// (Qwen3-VL-8B in bf16: 1.5 tokens/s against 10.6 once the reserve was read).
///
/// Four bf16 layers of a product, B far above them, `this-mac.json` a file of its own and empty. The bar was written
/// before the run: every layer kept after six rows.
extension Runs {
    @Test(.enabled(if: metalEnabled))
    func layersReadARowAtATimeAreKeptWithoutAReserve() async throws {
        let bench = try await Bench()
        let (rows, d) = (64, 2048)
        let names = (0..<4).map { "layer\($0)" }
        var rng = Seeded(state: 23)
        var tensors: [(String, String, [Int], [UInt8])] = []
        for name in names {
            let w = rng.uniform(d * d, -0.02...0.02).map { BFloat16Bits.rounded($0) }
            tensors.append(("\(name).w", "BF16", [d, d], w.withUnsafeBytes { Array($0) }))
        }
        let file = try SafeTensors(try weightsFile("layers-through.safetensors", tensors))

        // A file of its own, no reserve in it: the functions met for the first time, as a writing's are at each new form.
        let measured = FileManager.default.temporaryDirectory.appendingPathComponent("this-mac-layers-\(UUID().uuidString).json")
        let before = ThisMac.file
        ThisMac.file = measured
        defer { ThisMac.file = before; try? FileManager.default.removeItem(at: measured) }

        var l = Layers.Declaration("run-layers-through", rows: rows)
        l.perRow = [.init("x", width: d, .bfloat16, next: "y", apart: true)]
        l.tensors = [.init("token", bytes: d * 2)]
        l.layers = names.map { Block.Group($0, form: "layer", weights: [("w", "\($0).w", .bfloat16(joined: []))]) }
        l.kinds["layer"] = { b in
            var m = Aicode.Module(arguments: [("x", Aicode.tensor([b.rows, d], .bf16)), ("w", Aicode.tensor([d, d], .bf16))])
            m.outputs = [("y", m.body.linear(m.argument("x"), m.argument("w")))]
            return m
        }
        l.edges["token"] = { _ in
            var m = Aicode.Module(arguments: [("x", Aicode.tensor([1, d], .bf16)), ("w", Aicode.tensor([d, d], .bf16))])
            m.outputs = [("y", m.body.linear(m.argument("x"), m.argument("w")))]
            return m
        }
        l.again = true

        let layers = try await Layers.open(bench.device, l, files: [file], budget: 1 << 40)
        defer { layers.close() }
        #expect(layers.stage.reserve == nil, "a reserve read: the test needs functions never met")
        let x = await layers.rows("x").bindMemory(to: UInt16.self, capacity: rows * d)
        for i in 0..<rows * d { x[i] = BFloat16Bits.rounded(Float(rng.next() * 2 - 1)) }
        try await layers.forward()
        for _ in 0..<6 {
            let t = await layers.pointer("token").bindMemory(to: UInt16.self, capacity: d)
            for i in 0..<d { t[i] = BFloat16Bits.rounded(Float(rng.next() * 2 - 1)) }
            _ = try await layers.through("token", row: 0, entry: ["x": .tensor("token")])
            await layers.finish()
        }
        let kept = layers.stage.kept
        print("layers through: \(layers.stage.plan.line); kept \(kept.count) of \(names.count) \(kept)")
        #expect(Set(kept) == Set(names), "kept \(kept)")
    }
}
