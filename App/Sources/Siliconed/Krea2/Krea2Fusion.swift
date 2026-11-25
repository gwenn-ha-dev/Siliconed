import CoreAI
import Foundation
import Metal
import SiliCore

/// **Krea 2's text fusion, once a prompt** (`Krea2TextFusion` and `txt_in`): the twelve taps of Qwen3-VL to the text's
/// rows of the DiT's stream, in bf16 on the GPU, in an arena of its own let go before the DiT's.
///
///     taps [T, 12, 2560] ─ layerwise ×2 (by tiles of 32 tokens) ─ projector ─ refiner ×2 (the palier) ─ txtmlp ─ [T, 6144]
///
/// It reads no latent and no time: in the reference it runs at every step, here once (the same values).
/// The refiners and `txtmlp` read the text's palier (`Krea2Session.Geometry.palier`), the rows past it zeros, masked as
/// keys; the layerwise blocks attend within a token, so a tile of 32 tokens is 32 independent sequences of 12. A LoRA lands
/// on its four blocks' products as on the DiT's (`Krea2Functions`' `ranks`), each kind of block its own ranks.
enum Krea2Fusion {
    static let width = Krea2Functions.textWidth, taps = Krea2Functions.taps, tokens = Krea2Functions.tokens
    static let layerwise = ["txtfusion.layerwise_blocks.0.", "txtfusion.layerwise_blocks.1."]
    static let refiners = ["txtfusion.refiner_blocks.0.", "txtfusion.refiner_blocks.1."]

    struct Text {
        /// `text_fused` `[T, 2560]` (a check's, empty otherwise) and `txt_in_out` `[T, 6144]`, bf16 values.
        let fused, rows: [Float]
    }

    static func activations(palier p: Int) -> [(String, Int)] {
        [("a", p * taps * width * 2), ("b", p * taps * width * 2), ("c", p * width * 2), ("d", p * width * 2), ("mask", p * 2)]
    }

    static func asset(_ kind: String, rows: Int, ranks: [String: Int]) -> String { "krea2-\(kind)-\(rows)-bf16" + Krea2Functions.named(ranks) }

    /// The blocks of a kind of function: the layerwise's or the refiners'.
    static func blocks(_ kind: String) -> [String] { kind == "layerwise" ? layerwise : kind == "refiner" ? refiners : [] }

    /// A block's products as a LoRA adapts them (`LoRAProduct`), `b` its prefix.
    static func loraProducts(_ b: String) -> [LoRAProduct] {
        let (w, h) = (width, Krea2Functions.textHidden)
        return Krea2Functions.products.map { name, module in
            let (inn, out) = name == "down" ? (h, w) : name == "up" || name == "mgate" ? (w, h) : (w, w)
            return LoRAProduct(name: name, key: b + module, inn: inn, out: out)
        }
    }

    /// Each kind of block's ranks: the largest landing on one of its blocks' products, by argument (`LoRAStack.ranks`).
    static func ranks(_ loras: LoRAStack) -> [String: [String: Int]] {
        Dictionary(uniqueKeysWithValues: ["layerwise", "refiner"].map { kind in (kind, loras.ranks(blocks(kind).flatMap(loraProducts), on: .krea2)) })
    }

    /// The functions by kind, at their rows: the layerwise blocks' a tile of 32 tokens' taps, the projector's 32 tokens,
    /// the refiners' and `txtmlp`'s the palier.
    static func shapes(palier p: Int) -> [(kind: String, rows: Int)] {
        [("layerwise", taps * tokens), ("projector", tokens), ("refiner", p), ("txtmlp", p)]
    }

    /// **What the fusion needs** (`Stage.Declaration`): its activations, `txtmlp`'s output apart, every weight resident
    /// (four blocks of 2560 and `txtmlp`, read once a prompt: no ring to plan).
    static func declaration(palier p: Int, ranks: [String: [String: Int]]) -> SiliCore.Stage.Declaration {
        var d = SiliCore.Stage.Declaration("Krea 2's text fusion")
        d.tensors = activations(palier: p).map { SiliCore.Stage.Tensor($0.0, bytes: $0.1) }
            + [SiliCore.Stage.Tensor("out", bytes: p * Krea2Functions.d * 2, .apart)]
        for (kind, r) in ranks { for b in blocks(kind) { d.tensors += LoRAStack.tensors(loraProducts(b), ranks: r, element: .bfloat16, prefix: "lora." + b) } }
        d.residents = Krea2Functions.fusionOthers
            + (layerwise + refiners).flatMap { b in Krea2Functions.fusionBlock(b).map { ($0.name + "@" + b, $0.key, $0.recipe) } }
        for (kind, rows) in shapes(palier: p) {
            let r = ranks[kind] ?? [:]
            d.functions[kind] = SiliCore.Stage.Function(asset(kind, rows: rows, ranks: r)) { Krea2Functions.module(kind, rows: rows, keys: rows, ranks: r) }
        }
        return d
    }

    /// **What `text` would specialize** for `t` text rows, nothing allocated (`Stage.prepare`).
    static func prepare(rows t: Int, file: SafeTensors, loras: LoRAStack = .none, budget: Int) throws -> [Functions.Specialization] {
        try SiliCore.Stage.prepare(declaration(palier: Krea2Session.Geometry.palier(t), ranks: ranks(loras)), files: [file], budget: budget)
    }

    /// `hidden`, the taps `[T, 12, 2560]` (`Krea2Render.encode`), to the text's rows; `fused`: `text_fused` too (a
    /// check's). The stage planned under `budget` (`Stage.open`); let go before returning. What it hands back is made
    /// before its plan, so that its floor counts it (made while it lives, it is read as Core AI's share).
    static func text(_ hidden: HostTensor, file: SafeTensors, loras: LoRAStack = .none, budget: Int, fused keeping: Bool = false) async throws -> Text {
        let t = hidden.shape[0], p = Krea2Session.Geometry.palier(t)
        precondition(hidden.shape == [t, taps, width], "krea2: taps \(hidden.shape)")
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("no Metal device") }
        let ranks = ranks(loras)
        let declaration = declaration(palier: p, ranks: ranks)
        var textFused = [Float](repeating: 0, count: keeping ? t * width : 0), textRows = [Float](repeating: 0, count: t * Krea2Functions.d)
        let stage = try await SiliCore.Stage.open(device, declaration, files: [file], budget: budget)
        defer { stage.close() }
        for (kind, r) in ranks {
            for b in blocks(kind) { try loras.write(loraProducts(b), on: .krea2, ranks: r, element: .bfloat16, prefix: "lora." + b, into: stage.pointer) }
        }
        let a = stage.pointer("a").assumingMemoryBound(to: UInt16.self)
        for i in 0..<hidden.floats.count { a[i] = BFloat16Bits.rounded(hidden.floats[i]) }
        let mask = stage.pointer("mask").assumingMemoryBound(to: UInt16.self)
        for j in 0..<p { mask[j] = j < t ? 0 : BFloat16Bits.rounded(-1e4) }
        let run = Run(stage: stage, ranks: ranks)

        // The layerwise blocks, a tile of 32 tokens at a time, from one buffer to the other; the projector after them.
        let tiles = (t + tokens - 1) / tokens, rows = taps * tokens, tileBytes = rows * width * 2
        for (i, b) in layerwise.enumerated() {
            let (from, to) = i % 2 == 0 ? ("a", "b") : ("b", "a")
            for k in 0..<tiles { try await run.call("layerwise", block: b, rows: rows, from: from, to: to, bytes: k * tileBytes, out: k * tileBytes) }
        }
        let last = layerwise.count % 2 == 0 ? "a" : "b"
        for k in 0..<tiles {
            try await run.call("projector", rows: tokens, from: last, to: "c", bytes: k * tileBytes, out: k * tokens * width * 2)
        }
        for (i, b) in refiners.enumerated() {
            let (from, to) = i % 2 == 0 ? ("c", "d") : ("d", "c")
            try await run.call("refiner", block: b, rows: p, from: from, to: to)
        }
        let fused = refiners.count % 2 == 0 ? "c" : "d"
        try await run.call("txtmlp", rows: p, from: fused, to: "out")
        func widen(_ name: String, into out: inout [Float]) {
            let x = stage.pointer(name).assumingMemoryBound(to: UInt16.self)
            for i in out.indices { out[i] = Float(bitPattern: UInt32(x[i]) << 16) }
        }
        widen(fused, into: &textFused)
        widen("out", into: &textRows)
        return Text(fused: textFused, rows: textRows)
    }

    /// A call of the fusion's: `kind` at its rows, from `from` at byte `bytes` into `to` at byte `out`, the block's weights
    /// (`block`) or the others.
    struct Run {
        let stage: SiliCore.Stage
        let ranks: [String: [String: Int]]

        func call(_ kind: String, block: String = layerwise[0], rows: Int, from: String, to: String,
                  bytes: Int = 0, out: Int = 0) async throws {
            func view(_ name: String, _ offset: Int, _ shape: [Int]) -> View {
                stage.view(name, shape, .bfloat16, bytes: offset)
            }
            var inputs: [String: View] = [:]
            let suffix = kind == "layerwise" || kind == "refiner" ? "@" + block : ""
            for (name, shape, _) in Krea2Functions.arguments(kind, rows: rows, keys: rows, ranks: ranks[kind] ?? [:]) {
                switch name {
                case "x": inputs[name] = view(from, bytes, shape)
                case "mask": inputs[name] = view("mask", 0, shape)
                case _ where name.hasSuffix("_down") || name.hasSuffix("_up"): inputs[name] = view("lora." + block + name, 0, shape)
                default:
                    guard let placed = stage.residents[name + suffix] ?? stage.residents[name] else { preconditionFailure("\(kind): no \(name)") }
                    inputs[name] = View(buffer: stage.arena.buffer(placed.offset), type: placed.type, shape: shape)
                }
            }
            let width = kind == "txtmlp" ? Krea2Functions.d : Krea2Functions.textWidth
            try await SiliCore.Run.call(stage.functions[kind]!, stage.stream, inputs: inputs, outputs: [("y", view(to, out, [rows, width]))])
        }
    }
}
