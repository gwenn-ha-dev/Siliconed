import CoreAI
import Foundation
import SiliCore

/// **The language model's head on the layers' stage** (`lm_head(norm(x))`): the final norm's gain resident, `lm_head` in
/// tensors by stretches of rows as the file lays them (`[stretch, width]`, a tensor each), one function over `rows` rows
/// and every stretch, called once (`headModule`); the logits in f32, a tensor a stretch, apart (a function's last outputs:
/// `Qwen3Encoder.declaration`). Qwen3-VL-8B's head is its own (151 936 × 4096, untied): 1.16 GiB in bf16, read as 16
/// stretches of 9496 rows.
final class Qwen3Head {
    /// The head's rows, and whether it reads rows of its own (the teacher-forced continuation: `head_x`) or the
    /// decode's values.
    struct Shape { let rows: Int; let input: Bool }

    let shape: Shape
    let model: Qwen3Encoder
    let arithmetic: Qwen3Arithmetic
    let stretch: Int, count: Int

    /// The stretches a vocabulary is read by: the fewest of 16 or less that divide it.
    static func stretches(_ vocabulary: Int) -> Int { [16, 8, 4, 2, 1].first { vocabulary % $0 == 0 }! }

    /// What the head adds to the layers' declaration: the norm resident, the stretches' weights and logits, its own rows
    /// with `input`, its function.
    static func declare(_ shape: Shape, _ model: Qwen3Encoder, _ a: Qwen3Arithmetic, in d: inout Layers.Declaration) {
        let c = model.config, n = stretches(c.vocabulary), s = c.vocabulary / n, e = a.element.size, rows = shape.rows
        d.residents.append(("norm", model.prefix + "norm.weight", a.half ? .half : .bfloat16(joined: [])))
        for k in 0..<n {
            d.tensors += [.init("head_w\(k)", bytes: s * c.width * (a.half ? 1 : e)), .init("logits\(k)", bytes: rows * s * 4, .apart)]
            if a.half { d.tensors.append(.init("head_s\(k)", bytes: s * 2)) }
        }
        if shape.input {
            d.tensors.append(.init("head_x", bytes: rows * c.width * e))
            if a.half { d.tensors.append(.init("head_xf", bytes: rows * 4)) }
        }
        d.edges["head"] = { _ in model.headModule(a, rows: rows, stretch: s, count: n) }
    }

    /// The head of `layers`, its stretches read from `file` into their tensors.
    init(_ shape: Shape, _ model: Qwen3Encoder, _ a: Qwen3Arithmetic, layers: Layers, file: SafeTensors) async throws {
        let c = model.config, d = c.width
        (self.shape, self.model, self.arithmetic) = (shape, model, a)
        count = Self.stretches(c.vocabulary)
        stretch = c.vocabulary / count
        guard let table = file.tensors["lm_head.weight"], table.shape == [c.vocabulary, d] else {
            throw ProgramError("lm_head: \(file.tensors["lm_head.weight"]?.shape.description ?? "missing")")
        }
        guard a.half == (table.dtype == "I8") else { throw ProgramError("lm_head: \(table.dtype) beside the layers' \(a.name)") }
        let size = a.half ? 1 : a.element.size, s = stretch
        var scales: [Float] = []
        if a.half {
            guard case .w8a8(let suffix, _) = a else { preconditionFailure() }
            scales = try file.floats("lm_head" + suffix)
        }
        // The codes by pieces of k as the head's products read them (`Weights.codesShape`): one piece, the file's rows.
        let (pieces, piece) = a.half ? try Weights.pieces(d) : (1, d)
        var rows = [UInt8](repeating: 0, count: pieces > 1 ? s * d : 0)
        for k in 0..<count {
            let into = await layers.pointer("head_w\(k)")
            if pieces == 1 {
                try file.read(table, at: k * s * d * size, count: s * d * size, into: into)
            } else {
                try rows.withUnsafeMutableBytes { try file.read(table, at: k * s * d, count: s * d, into: $0.baseAddress!) }
                for p in 0..<pieces { for o in 0..<s { rows.withUnsafeBytes { (into + (p * s + o) * piece).copyMemory(from: $0.baseAddress! + o * d + p * piece, byteCount: piece) } } }
            }
            // Every stretch's scales, whatever its pieces: left unwritten, the head's products are 0 and every token the
            // first, « ! » (issue 19, 0.31).
            if a.half { Qwen3Encoder.write(Array(scales[(k * s)..<((k + 1) * s)]), as: .f16, to: await layers.pointer("head_s\(k)")) }
        }
    }

    /// The head's own rows (`x`, and `x_factor` on f16), as its arguments are bound.
    var input: [String: Block.Bind] { ["x": .tensor("head_x"), "x_factor": .tensor("head_xf")] }

    /// **The logits of the head's rows** — read from `values` (the decode's, as Core AI's values) or as `bind` says —
    /// into their tensors (`logits<k>`): one call over every stretch, one wait.
    func run(_ layers: Layers, bind: [String: Block.Bind] = [:], values: [String: InferenceFunction.AsyncValue] = [:]) async throws {
        var stretches: [String: Block.Bind] = [:]
        for k in 0..<count {
            stretches["w\(k)"] = .tensor("head_w\(k)")
            stretches["logits\(k)"] = .tensor("logits\(k)")
            if arithmetic.half { stretches["w_scale\(k)"] = .tensor("head_s\(k)") }
        }
        try layers.chain("head", bind: bind.merging(stretches) { a, _ in a }, values: values)
        await layers.finish()
    }

    /// The first `n` rows of the logits `run` wrote, `[n, vocabulary]` row by row, into `out`: an array its caller made
    /// before the stage was planned, so that the floor counts it (made while the stage was open, the teacher-forced
    /// logits passed P by 14 MiB, `check qwen-image-2.1 writer`, 19/11).
    func read(_ layers: Layers, rows n: Int, into out: inout [Float]) async {
        let c = model.config, s = stretch, rows = shape.rows
        precondition(n <= rows && out.count >= n * c.vocabulary, "the head's \(n) rows into \(out.count) values")
        for k in 0..<count {
            let l = await layers.pointer("logits\(k)").assumingMemoryBound(to: Float.self)
            for r in 0..<n { for j in 0..<s { out[r * c.vocabulary + k * s + j] = l[r * s + j] } }
        }
    }

    /// The best token of a row of logits: ties to the lowest identifier (`torch.argmax`).
    static func argmax(_ logits: ArraySlice<Float>) -> Int {
        var best = logits.startIndex
        for i in logits.indices where logits[i] > logits[best] { best = i }
        return best - logits.startIndex
    }
}
