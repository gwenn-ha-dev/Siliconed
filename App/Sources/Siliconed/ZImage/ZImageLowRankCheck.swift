import CoreAI
import Foundation
import Metal
import SiliCore

/// **The low-rank paths of Z-Image's block functions against the CPU** (`siliconed check z-image block-lora`): layer
/// 0's functions over the first 512 rows of the oracle's stream (keys the same rows, none masked), each product of each
/// function in turn adapted alone by seeded factors of rank 32 — so that its input is the function's without a LoRA —
/// and the change the function makes, its result with the factors less its result without, against the CPU's
/// `x · down · up` on the same input and the factors as the function read them. The factors scaled by a power of two
/// for a change as large as the product's result: the bound then reads the arithmetic of the path, not its size.
///
///     standard   the block's three functions (`ZImageBlock`): modulation, pre, post
///     shared     the shared forms (`ZImageShared`) on the GPU: modulation, pre, out, ffn; and pre, out, ffn on the ANE,
///                each with its products adapted together, against the GPU's — on the weights a render takes
///                (`Arithmetic.of`): the Compact's codes, the Standard's in f16 under `--neural-engine on`
///
/// Without a LoRA: each function with every factor zero gives the bits of the function of no factor.
package enum ZImageLowRankCheck {
    typealias Tap = ZImageDiT.BlockTap

    /// A figure of the check: the error of a change against the CPU's (`bar`, its bound), or bits that must be the same
    /// (`bar` nil: `value` the values that differ).
    package struct Result: Sendable {
        package let name: String
        package let value: Double
        package let bar: Double?
        package let note: String
    }

    static let rows = 512, rank = 32
    static var d: Int { ZImageBlock.d }
    static var hidden: Int { ZImageBlock.hidden }

    /// Each product's function, the tap of its input (nil: the function's own argument, `adaln` or `o`) and of its result.
    typealias Case = (function: String, product: String, input: Tap?, output: [Tap])
    static let blockCases: [Case] = [("modulation", "ada", nil, [.adaln]), ("pre", "qkv", .attentionIn, [.q, .k, .v]),
                                     ("post", "out", .sdpa, [.attention]), ("post", "w1", .ffnIn, [.w1]), ("post", "w3", .ffnIn, [.w3]),
                                     ("post", "w2", .swiglu, [.feedForward])]
    static let sharedCases: [Case] = [("modulation", "ada", nil, [.adaln]), ("pre", "qkv", .attentionIn, [.q, .k, .v]), ("out", "out", nil, [.attention]),
                                      ("ffn", "w1", .ffnIn, [.w1]), ("ffn", "w3", .ffnIn, [.w3]), ("ffn", "w2", .swiglu, [.feedForward])]

    static func width(_ t: Tap) -> Int {
        switch t {
        case .adaln: 4 * d
        case .w1, .w3, .swiglu: hidden
        default: d
        }
    }
    static func count(_ t: Tap) -> Int { (t == .adaln ? 1 : rows) * width(t) }
    static let shapes = Dictionary(uniqueKeysWithValues: ZImageDiT.products("layers.0").map { ($0.name, (inn: $0.inn, out: $0.out)) })

    /// **The check on the map at `path`**: its arithmetic's block functions (the Standard's, or the shared forms on the
    /// weights a render takes, `Arithmetic.of`). `x`: the oracle's stream `[rows ≥ 512, d]`, `adaln` its `[256]`.
    package static func run(map path: String, x: [Float], adaln: [Float]) throws -> (described: String, results: [Result]) {
        try blocking { try await check(path, x, adaln) }
    }

    private static func check(_ path: String, _ x: [Float], _ adaln: [Float]) async throws -> (described: String, results: [Result]) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("no Metal device") }
        let file = try ZImageDiT.file(path)
        let arithmetic = Arithmetic.of(codes: Arithmetic.codes(file, marker: ZImageDiT.codesMarker), neuralEngine: CoreAI.neuralEngineInForce)
        precondition(x.count >= rows * d && adaln.count == ZImageBlock.adalnDim, "x of \(x.count) values, adaln of \(adaln.count)")
        // The seeded factors, made before any stage: what the stages' floor counts.
        let largest = shapes.values.map { max($0.inn, $0.out) }.max()!
        let down = UnsafeMutableBufferPointer<Float>.allocate(capacity: rank * largest), up = UnsafeMutableBufferPointer<Float>.allocate(capacity: rank * largest)
        defer { down.deallocate(); up.deallocate() }
        var session = Session(device, file, arithmetic, x: x, adaln: adaln, down: down, up: up)
        let results = try await session.run()
        return (session.described, results)
    }

    /// One arithmetic's stage and its runs.
    struct Session {
        let device: MTLDevice, file: SafeTensors, form: ZImageFunctions.Form, weights: SharedWeights
        let x: [Float], adaln: [Float]
        let down: UnsafeMutableBufferPointer<Float>, up: UnsafeMutableBufferPointer<Float>
        var described = ""
        var cases: [Case] { form == .shared ? sharedCases : blockCases }
        var plainFunctions: [String] { form == .shared ? ["modulation", "pre", "heads", "attention", "out", "ffn"] : ZImageBlock.functions }
        var element: NDArray.ScalarType { form == .bf16 ? .bfloat16 : .float16 }
        var bar: Double { form == .bf16 ? 4 * 0x1p-8 : 4 * 0x1p-11 }

        init(_ device: MTLDevice, _ file: SafeTensors, _ arithmetic: Arithmetic, x: [Float], adaln: [Float],
             down: UnsafeMutableBufferPointer<Float>, up: UnsafeMutableBufferPointer<Float>) {
            (self.device, self.file, self.x, self.adaln, self.down, self.up) = (device, file, x, adaln, down, up)
            (form, weights) = (ZImageFunctions.Form(arithmetic), arithmetic.weights)
        }

        /// The products a LoRA adapts in `f`: the Compact's modulation is the Standard's function under the guard.
        static func adapted(_ f: String, _ form: ZImageFunctions.Form) -> [String] {
            (form == .shared && f != "modulation" ? ZImageShared.adapted[f] : ZImageBlock.adapted[f]) ?? []
        }

        /// What `f`'s outputs are compared on with every factor zero, against the function of no factor: (tensor, its
        /// variant's, values, bytes each).
        func outputs(_ f: String) -> [(String, String, Int, Int)] {
            guard form == .shared else {
                return (["modulation": ["scales", "gates"], "pre": ["q", "k", "v"], "post": ["y"]][f] ?? []).map { name in
                    (name, "zero." + name, ["scales", "gates"].contains(name) ? 4 * ZImageBlock.gainRow : rows * d, 2)
                }
            }
            switch f {
            case "modulation": return ZImageShared.gains.map { ($0, "zero." + $0, d, 2) }
            case "pre": return [("qkv", "zero.pre", rows * 3 * d, 2), ("qkv_factor", "zero.pre-f", rows, 2)]
            case "out": return [("mid", "zero.out", rows * d, 2), ("mid_factor", "zero.out-f", rows, 2)]
            default: return [("y", "zero.ffn", rows * d, 2), ("y_factor", "zero.ffn-f", rows, 2)]
            }
        }

        // MARK: - The functions

        /// A function's name, its edge's: its form, rows, ranks and taps; `ane`, the ANE's specialization.
        func name(_ f: String, ranks: [String: Int] = [:], taps: [Tap] = [], ane: Bool = false) -> String {
            let r = f == "modulation" ? 1 : rows
            let low = Self.adapted(f, form).compactMap { p in ranks[p].map { "\(p)\($0)" } }
            return (ane ? "ane-" : "") + "\(f)-\(r)" + (low.isEmpty ? "" : "-lora-" + low.joined(separator: "-"))
                + (taps.isEmpty ? "" : "-taps-" + taps.map(\.rawValue).joined(separator: "-"))
        }

        /// Every call the check makes: (function, ranks, taps, unit).
        var calls: [(f: String, ranks: [String: Int], taps: [Tap], ane: Bool)] {
            var all: [(String, [String: Int], [Tap], Bool)] = plainFunctions.map { ($0, [:], [], false) }
            for f in plainFunctions where !Self.adapted(f, form).isEmpty {
                all.append((f, Dictionary(uniqueKeysWithValues: Self.adapted(f, form).map { ($0, rank) }), [], false))
            }
            for c in cases {
                let taps = (c.input.map { [$0] } ?? []) + c.output
                all += [(c.function, [:], taps, false), (c.function, [c.product: rank], taps, false)]
            }
            if form == .shared {
                for f in ZImageShared.neural { all += [(f, [:], [], true), (f, Dictionary(uniqueKeysWithValues: Self.adapted(f, form).map { ($0, rank) }), [], true)] }
            }
            return all
        }

        /// **Layer 0's functions as edges of a block** (`Block`): every call's function, the ANE's among them; the
        /// tensors each reads and writes by their names, layer 0's weights resident.
        func declaration() -> Block.Declaration {
            var n = Block.Declaration("Z-Image's layer 0 with low-rank paths (\(form), \(rows) rows)", rows: rows)
            let e = 2
            var t: [SiliCore.Stage.Tensor] = []
            func heap(_ name: String, _ bytes: Int) { t.append(.init(name, bytes: bytes)) }
            func surface(_ name: String, _ shape: [Int]) { t.append(.init(name, surface: shape, element: e)) }
            heap("adaln", ZImageBlock.adalnDim * 4)
            for (name, shape) in [("cos", rows * 128 * 4), ("sin", rows * 128 * 4)] { heap(name, shape) }
            heap("mask", rows * e)
            for name in ["q", "k", "v"] { heap(name, rows * d * e) }
            let tapped = Set(cases.flatMap { ($0.input.map { [$0] } ?? []) + $0.output })
            for tap in tapped { heap("base.\(tap.rawValue)", count(tap) * 4); heap("lora.\(tap.rawValue)", count(tap) * 4) }
            if form == .shared {
                for (name, n) in [("x", d), ("x_factor", 1), ("qkv", 3 * d), ("qkv_factor", 1), ("o", d), ("mid", d), ("mid_factor", 1), ("y", d), ("y_factor", 1)] {
                    surface(name, [rows, n])
                }
                for gain in ZImageShared.gains { surface(gain, [1, d]); surface("zero." + gain, [1, d]) }
                for unit in ["gpu", "ane", "zero"] {
                    for variant in unit == "zero" ? [""] : ["base", "lora"] {
                        let at = variant.isEmpty ? "\(unit)." : "\(unit)-\(variant)."
                        for (f, n) in [("pre", 3 * d), ("out", d), ("ffn", d)] { surface(at + f, [rows, n]); surface(at + f + "-f", [rows, 1]) }
                    }
                }
                for (p, s) in ZImageLowRankCheck.shapes where p != "ada" { surface(p + "_down", [s.inn, rank]); surface(p + "_up", [rank, s.out]) }
                heap("ada_down", ZImageBlock.adalnDim * rank * e); heap("ada_up", rank * 4 * d * e)
            } else {
                for (name, bytes) in [("x", rows * d * e), ("x_factor", rows * 4), ("scales", 4 * ZImageTransformer.gainBytes), ("gates", 4 * ZImageTransformer.gainBytes), ("y", rows * d * e),
                                      ("y_factor", rows * 4)] {
                    heap(name, bytes); heap("zero." + name, bytes)
                }
                for name in ["q", "k", "v"] { heap("zero." + name, rows * d * e) }
                for (p, s) in ZImageLowRankCheck.shapes { heap(p + "_down", s.inn * rank * e); heap(p + "_up", rank * s.out * e) }
            }
            n.tensors = t
            n.residents = ZImageFunctions.entries("layers.0", form: form, weights: weights)
            for c in calls {
                let (f, ranks, taps, form, weights, rows, ane) = (c.f, c.ranks, c.taps, form, weights, ZImageLowRankCheck.rows, c.ane)
                n.edges[name(f, ranks: ranks, taps: taps, ane: c.ane)] = { b in
                    if form == .shared {
                        return f == "modulation" ? ZImageShared.modulation(weights: weights, ranks: ranks, taps: taps)
                            : ZImageShared.module(f, rows: rows, keys: rows, weights: weights, ranks: ranks, taps: taps, neural: ane)
                    }
                    let r = f == "modulation" ? 1 : rows
                    return ZImageBlock.module(f, rows: r, keys: r, form: form, ranks: ranks, taps: taps)
                }
                if c.ane { n.neuralEdges.insert(name(f, ranks: ranks, taps: taps, ane: true)) }
            }
            return n
        }

        // MARK: - The calls

        /// What `f` writes, where it is not the tensor of its output's name: into `out` (a variant's: `zero.` or
        /// `<unit>-<variant>.`); and what it reads under another name.
        func binding(_ f: String, out: String = "") -> [String: Block.Bind] {
            if form == .shared {
                func variant(_ f: String) -> [String: Block.Bind] {
                    [f == "pre" ? "qkv" : "y": .tensor(out + f), f == "pre" ? "qkv_factor" : "y_factor": .tensor(out + f + "-f")]
                }
                switch f {
                case "modulation": return Dictionary(uniqueKeysWithValues: ZImageShared.gains.map { ($0, Block.Bind.tensor(out + $0)) })
                case "pre": return out.isEmpty ? [:] : variant(f)
                // The activations, not the product's codes of the same name.
                case "heads": return ["qkv": .tensor("qkv")]
                case "out": return out.isEmpty ? ["y": .tensor("mid"), "y_factor": .tensor("mid_factor")] : variant(f)
                case "ffn": return ["x": .tensor("mid"), "x_factor": .tensor("mid_factor")].merging(out.isEmpty ? [:] : variant(f)) { $1 }
                default: return [:]
                }
            }
            let gains: [String: Block.Bind] = ["scale1": .row("scales", 0), "gate1": .row("gates", 1), "scale2": .row("scales", 2), "gate2": .row("gates", 3)]
            let outputs = ["modulation": ["scales", "gates"], "pre": ["q", "k", "v"], "post": ["y", "y_factor"]][f] ?? []
            return gains.merging(out.isEmpty ? [:] : Dictionary(uniqueKeysWithValues: outputs.map { ($0, Block.Bind.tensor(out + $0)) })) { $1 }
        }

        /// One call of `f`, done when this returns: its arguments by `binding`, the weights and the factors by their
        /// names; its outputs `binding`'s, or its taps into `<into>.<tap>`.
        func call(_ block: Block, _ f: String, ranks: [String: Int], taps: [Tap] = [], into: String = "", out: String = "", ane: Bool = false) async throws {
            var bind = binding(f, out: out)
            for t in taps { bind[t.rawValue] = .tensor("\(into).\(t.rawValue)") }
            try await block.call(name(f, ranks: ranks, taps: taps, ane: ane), bind)
            await block.settled()
        }

        // MARK: - The run

        mutating func run() async throws -> [Result] {
            let block = try await Block.open(device, declaration(), files: [file], budget: MemoryBudget.machineCeiling(), share: .gpu)
            defer { block.close() }
            let stage = block.stage
            described = "\(form)" + (form == .bf16 ? "" : ", \(weights)") + ", " + stage.plan.line
            try write(stage)
            func all(_ f: String) -> [String: Int] { Dictionary(uniqueKeysWithValues: Self.adapted(f, form).map { ($0, rank) }) }
            var results: [Result] = []
            // The plain chain, every factor zero.
            for f in plainFunctions { try await call(block, f, ranks: [:]) }
            // Without a LoRA: every factor zero, the same bits.
            for f in plainFunctions where !Self.adapted(f, form).isEmpty {
                try await call(block, f, ranks: all(f), out: "zero.")
                let differing = outputs(f).reduce(0) { $0 + differ(stage, $1.0, $1.1, count: $1.2, bytes: $1.3) }
                results.append(Result(name: "\(form) \(f), every factor zero", value: Double(differing), bar: nil, note: "against the function of no factor"))
            }
            for c in cases {
                let taps = (c.input.map { [$0] } ?? []) + c.output
                try await call(block, c.function, ranks: [:], taps: taps, into: "base")
                let s = ZImageLowRankCheck.shapes[c.product]!
                let x = input(stage, c, "base")
                seed(s.inn, s.out, Self.seedOf(c.product))
                // As large a change as the result: the factors scaled by a power of two.
                let result = norm(stage, c.output.map { "base.\($0.rawValue)" }), change = changeNorm(stage, c, x, s.inn, s.out)
                let scale = Float(sign: .plus, exponent: Int(log2(result / max(change, 1e-30)).rounded()), significand: 1)
                for i in 0..<(s.out * rank) { up[i] *= scale }
                try LowRank.write(down: down.baseAddress!, up: up.baseAddress!, rank: rank, inn: s.inn, out: s.out,
                                  into: stage.pointer(c.product + "_down"), stage.pointer(c.product + "_up"), rank: rank, element: element)
                try await call(block, c.function, ranks: [c.product: rank], taps: taps, into: "lora")
                if let t = c.input {
                    results.append(Result(name: "\(form) \(c.function) \(c.product): its input", value: Double(differ(stage, "base.\(t.rawValue)", "lora.\(t.rawValue)", count: count(t), bytes: 4)),
                                          bar: nil, note: "with its factors and without"))
                }
                let (error, size) = compare(stage, c, x, s.inn, s.out)
                results.append(Result(name: "\(form) \(c.function) \(c.product): the change", value: error, bar: bar,
                                      note: String(format: "‖Δ‖ / ‖y‖ %.2f against the CPU's x · down · up", size)))
            }
            if form == .shared {
                // The ANE: each function with its products adapted together, against the GPU's, beside the same without.
                for f in ZImageShared.neural {
                    var e: [String: Double] = [:]
                    for (variant, ranks) in [("base", [String: Int]()), ("lora", all(f))] {
                        try await call(block, f, ranks: ranks, out: "gpu-\(variant).")
                        try await call(block, f, ranks: ranks, out: "ane-\(variant).", ane: true)
                        e[variant] = trueDistance(stage, "ane-\(variant).\(f)", "gpu-\(variant).\(f)", n: f == "pre" ? 3 * d : d)
                    }
                    results.append(Result(name: "shared \(f) on the ANE, adapted", value: e["lora"]!, bar: 2 * e["base"]! + 0x1p-9,
                                          note: String(format: "against the GPU's; without a LoRA %.2e (the bound 2 × that + 2⁻⁹)", e["base"]!)))
                }
            }
            return results
        }

        // MARK: - The stage's inputs

        /// The stream (bf16 rounded; in the shared forms x̂ under 2¹⁴ beside its row factor, within f16's normals),
        /// adaln, the RoPE tables of the first rows, the mask all zeros, every factor zero.
        func write(_ stage: SiliCore.Stage) throws {
            switch form {
            case .bf16:
                let p = stage.pointer("x").bindMemory(to: UInt16.self, capacity: rows * d)
                for i in 0..<rows * d { p[i] = BFloat16Bits.rounded(x[i]) }
            case .shared:
                let p = stage.pointer("x").bindMemory(to: Float16.self, capacity: rows * d)
                for r in 0..<rows {
                    let (unit, factor) = RowExponent.split(x[(r * d)..<((r + 1) * d)], .ranged(f32Factor: false))
                    stage.pointer("x_factor").assumingMemoryBound(to: Float16.self)[r] = Float16(factor)
                    for i in 0..<d { p[r * d + i] = Float16(unit[i]) }
                }
            }
            _ = adaln.withUnsafeBytes { memcpy(stage.pointer("adaln"), $0.baseAddress!, $0.count) }
            // The first rows of a 512² image's tables: its image rows.
            let dit = ZImageDiT(config: .published, latentHeight: 64, latentWidth: 64, capLength: 1), rope = dit.rope(rows: dit.positions().count)
            _ = rope.cos.withUnsafeBytes { memcpy(stage.pointer("cos"), $0.baseAddress!, rows * 128 * 4) }
            _ = rope.sin.withUnsafeBytes { memcpy(stage.pointer("sin"), $0.baseAddress!, rows * 128 * 4) }
            memset(stage.pointer("mask"), 0, rows * 2)
            for (p, s) in ZImageLowRankCheck.shapes {
                memset(stage.pointer(p + "_down"), 0, s.inn * rank * 2)
                memset(stage.pointer(p + "_up"), 0, rank * s.out * 2)
            }
        }

        static func seedOf(_ product: String) -> UInt64 { UInt64(["ada", "qkv", "out", "w1", "w3", "w2"].firstIndex(of: product)! + 1) }

        /// Seeded factors in [−1, 1): `down` `[r, in]`, `up` `[out, r]`, as a file stores them.
        func seed(_ inn: Int, _ out: Int, _ seed: UInt64) {
            var state = seed
            func next() -> Float {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                return Float(Int64(bitPattern: state) >> 11) / Float(1 << 52)
            }
            for i in 0..<(rank * inn) { down[i] = next() }
            for i in 0..<(out * rank) { up[i] = next() }
        }

        // MARK: - What is compared

        /// The product's input, true values `[rows, in]`: its tap, or the function's own argument.
        func input(_ stage: SiliCore.Stage, _ c: Case, _ run: String) -> UnsafePointer<Float>? {
            c.input.map { UnsafePointer(stage.pointer("\(run).\($0.rawValue)").assumingMemoryBound(to: Float.self)) }
        }

        /// Row `r` of the input, column `k`: the tap's f32, or the argument's (adaln f32; o f16, a true value).
        func value(_ stage: SiliCore.Stage, _ c: Case, _ x: UnsafePointer<Float>?, _ r: Int, _ k: Int, _ inn: Int) -> Float {
            if let x { return x[r * inn + k] }
            if c.function == "modulation" { return stage.pointer("adaln").assumingMemoryBound(to: Float.self)[k] }
            return Float(stage.pointer("o").assumingMemoryBound(to: Float16.self)[r * inn + k])
        }

        func norm(_ stage: SiliCore.Stage, _ names: [String]) -> Float {
            var sum = 0.0
            for name in names {
                let tap = Tap(rawValue: String(name.split(separator: ".").last!))!, p = stage.pointer(name).assumingMemoryBound(to: Float.self)
                for i in 0..<count(tap) { sum += Double(p[i]) * Double(p[i]) }
            }
            return Float(sum.squareRoot())
        }

        /// The CPU's change `x · down · up` row by row, each value given to `each(row, column, Δ)`; `down` `[in, r]` and
        /// `up` `[r, out]` read through `d` and `u`.
        func changes(_ stage: SiliCore.Stage, _ c: Case, _ x: UnsafePointer<Float>?, _ inn: Int, _ out: Int,
                     down d: (Int, Int) -> Float, up u: (Int, Int) -> Float, _ each: (Int, Int, Double) -> Void) {
            let n = c.function == "modulation" ? 1 : rows
            var t = [Float](repeating: 0, count: rank)
            for r in 0..<n {
                for j in 0..<rank { t[j] = 0 }
                for k in 0..<inn {
                    let v = value(stage, c, x, r, k, inn)
                    for j in 0..<rank { t[j] += v * d(k, j) }
                }
                for o in 0..<out {
                    var sum: Float = 0
                    for j in 0..<rank { sum += t[j] * u(j, o) }
                    each(r, o, Double(sum))
                }
            }
        }

        /// ‖x · downᵀ · upᵀ‖ of the seeded factors (`[r, in]`, `[out, r]`, unscaled), x the base run's input.
        func changeNorm(_ stage: SiliCore.Stage, _ c: Case, _ x: UnsafePointer<Float>?, _ inn: Int, _ out: Int) -> Float {
            var sum = 0.0
            changes(stage, c, x, inn, out, down: { k, j in down[j * inn + k] }, up: { j, o in up[o * rank + j] }) { _, _, v in sum += v * v }
            return Float(sum.squareRoot())
        }

        /// A tensor's element `i` widened to f32.
        func element(_ stage: SiliCore.Stage, _ name: String, _ i: Int) -> Float {
            let bits = stage.pointer(name).assumingMemoryBound(to: UInt16.self)[i]
            return form == .bf16 ? Float(bitPattern: UInt32(bits) << 16) : Float(Float16(bitPattern: bits))
        }

        /// **The change the function made against the CPU's**: (lora − base) at the product's result, against
        /// x · down · up with the factors as the function read them; and ‖Δ‖ / ‖y‖.
        func compare(_ stage: SiliCore.Stage, _ c: Case, _ x: UnsafePointer<Float>?, _ inn: Int, _ out: Int) -> (error: Double, size: Double) {
            // The factors as written, widened: `[in, R]`, `[R, out]`.
            for i in 0..<(inn * rank) { down[i] = element(stage, c.product + "_down", i) }
            for i in 0..<(rank * out) { up[i] = element(stage, c.product + "_up", i) }
            let base = c.output.map { stage.pointer("base.\($0.rawValue)").assumingMemoryBound(to: Float.self) }
            let lora = c.output.map { stage.pointer("lora.\($0.rawValue)").assumingMemoryBound(to: Float.self) }
            let w = width(c.output[0])
            var (error, change, result) = (0.0, 0.0, 0.0)
            changes(stage, c, x, inn, out, down: { k, j in down[k * rank + j] }, up: { j, o in up[j * out + o] }) { r, o, v in
                let (t, col) = o.quotientAndRemainder(dividingBy: w), i = r * w + col
                let y = Double(base[t][i]), made = Double(lora[t][i]) - y
                error += (made - v) * (made - v); change += v * v; result += y * y
            }
            return ((error / change).squareRoot(), (change / result).squareRoot())
        }

        /// How many values of `count` elements of `bytes` each differ between tensors `a` and `b`.
        func differ(_ stage: SiliCore.Stage, _ a: String, _ b: String, count: Int, bytes: Int) -> Int {
            let (p, q) = (stage.pointer(a), stage.pointer(b))
            return (0..<count).reduce(0) { $0 + (memcmp(p + $1 * bytes, q + $1 * bytes, bytes) == 0 ? 0 : 1) }
        }

        /// ‖a − b‖ / ‖b‖ of two of the shared forms' outputs, true values: `[rows, n]` f16 times `[rows, 1]` f16.
        func trueDistance(_ stage: SiliCore.Stage, _ a: String, _ b: String, n: Int) -> Double {
            let (pa, pb) = (stage.pointer(a).assumingMemoryBound(to: Float16.self), stage.pointer(b).assumingMemoryBound(to: Float16.self))
            let (fa, fb) = (stage.pointer(a + "-f").assumingMemoryBound(to: Float16.self), stage.pointer(b + "-f").assumingMemoryBound(to: Float16.self))
            var (num, den) = (0.0, 0.0)
            for r in 0..<rows {
                for i in 0..<n {
                    let (u, v) = (Double(pa[r * n + i]) * Double(fa[r]), Double(pb[r * n + i]) * Double(fb[r]))
                    num += (u - v) * (u - v); den += v * v
                }
            }
            return (num / den).squareRoot()
        }
    }
}
