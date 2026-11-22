import CoreAI
import Foundation
import Metal
import SiliCore

/// **The low-rank paths of Z-Image's block functions against the CPU** (`siliconed check z-image block-lora`): the
/// render's own layer 0 (`ZImageTransformer.declaration`, its `Part`) over the oracle's stream, every product adapted at
/// rank 32 as a LoRA landing on all of them; each product's factors seeded in turn, the others zero — so that its input
/// is the function's without a LoRA —, and the change the function makes on the first 512 rows, its result with the
/// factors less its result without, against the CPU's `x · down · up` on the same input and the factors as the function
/// read them. The factors scaled by a power of two for a change as large as the product's result: the bound then reads
/// the arithmetic of the path, not its size.
///
///     standard   the block's three functions (`ZImageBlock`): modulation, pre, post
///     shared     the shared forms (`ZImageShared`) on the GPU: modulation, pre, out, ffn — on the weights a render
///                takes (`Arithmetic.of`): the Compact's codes, the Standard's in f16 under `--neural-engine on`
///
/// Every run is the GPU's alone, tapped (`Part.tapped`), but the ANE's: under `--neural-engine on` in the shared forms,
/// the render's layer (`Part.layer`) with the ANE's share told (its first 512 rows), as a render gives it its rows —
/// pre, out and ffn on the ANE's own products (q64 on its copy of the codes), each against the GPU's on the same rows,
/// with every product adapted and without. Their inputs are the layer's chain: pre's the stream, out's and ffn's what the
/// ANE's steps before them gave.
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
    /// Every product of layer 0 at the check's rank: the render's ranks for a LoRA that lands on all of them.
    static let ranks = shapes.mapValues { _ in rank }

    /// **The check on the map at `path`**: its arithmetic's block functions (the Standard's, or the shared forms on the
    /// weights a render takes, `Arithmetic.of`). `x`: the oracle's stream `[sequence, d]` for a latent `[16, h, w]` and a
    /// caption of `capLength` tokens, `adaln` its `[256]`.
    package static func run(map path: String, x: [Float], adaln: [Float], latentHeight: Int, latentWidth: Int,
                            capLength: Int) throws -> (described: String, results: [Result]) {
        try blocking { try await check(path, x, adaln, latentHeight, latentWidth, capLength) }
    }

    private static func check(_ path: String, _ x: [Float], _ adaln: [Float], _ h: Int, _ w: Int,
                              _ capLength: Int) async throws -> (described: String, results: [Result]) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("no Metal device") }
        let file = try ZImageDiT.file(path)
        let arithmetic = Arithmetic.of(codes: Arithmetic.codes(file, marker: ZImageDiT.codesMarker), neuralEngine: CoreAI.neuralEngineInForce)
        let g = try ZImageTransformer.Geometry(latentHeight: h, latentWidth: w, capLength: capLength)
        precondition(x.count == g.sequence * d && g.sequence >= rows && adaln.count == ZImageBlock.adalnDim,
                     "x of \(x.count) values for \(g.sequence) rows, adaln of \(adaln.count)")
        // Everything the check keeps, made before any stage: what the stages' floor counts.
        let kept = Kept()
        var session = Session(device, file, arithmetic, g, x: x, adaln: adaln, kept: kept)
        let results = try await session.run()
        return (session.described, results)
    }

    /// **Storage made before any stage**, so that the stages' floor counts it (made while a stage lives, it is read as
    /// Core AI's share); let go with the check.
    final class Kept {
        private var all: [UnsafeMutableRawBufferPointer] = []
        func floats(_ n: Int) -> UnsafeMutableBufferPointer<Float> {
            let b = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
            b.initialize(repeating: 0)
            all.append(UnsafeMutableRawBufferPointer(b))
            return b
        }
        func bytes(_ n: Int) -> UnsafeMutableRawBufferPointer {
            let b = UnsafeMutableRawBufferPointer.allocate(byteCount: n, alignment: 16)
            b.initializeMemory(as: UInt8.self, repeating: 0)
            all.append(b)
            return b
        }
        deinit { all.forEach { $0.deallocate() } }
    }

    /// One arithmetic's stages and their runs.
    struct Session {
        let device: MTLDevice, file: SafeTensors, arithmetic: Arithmetic, g: ZImageTransformer.Geometry
        let form: ZImageFunctions.Form, weights: SharedWeights
        let x: [Float], adaln: [Float]
        /// The seeded factors, `[r, in]` and `[out, r]`.
        let down: UnsafeMutableBufferPointer<Float>, up: UnsafeMutableBufferPointer<Float>
        /// The boundaries of the run with every factor zero (`base`) and of the run with one product's (`lora`); the
        /// shared forms' `o`, the base run's.
        let base: [Tap: UnsafeMutableBufferPointer<Float>], lora: [Tap: UnsafeMutableBufferPointer<Float>], o: UnsafeMutableBufferPointer<Float>
        /// Each function's outputs, the function of no factor's (`plain`) and with every factor zero (`zero`).
        let plain: [String: UnsafeMutableRawBufferPointer], zero: [String: UnsafeMutableRawBufferPointer]
        /// Each product's factors as the check wrote them, for the ANE's runs.
        let written: [String: (down: UnsafeMutableRawBufferPointer, up: UnsafeMutableRawBufferPointer)]
        /// pre's, out's and ffn's results on the ANE's rows, true values: the GPU's and the ANE's, without and with the factors.
        let units: [String: UnsafeMutableBufferPointer<Float>]
        var described = ""
        var cases: [Case] { form == .shared ? sharedCases : blockCases }
        /// The functions a LoRA adapts, the render's.
        var functions: [String] { form == .shared ? ["modulation", "pre", "out", "ffn"] : ZImageBlock.functions }
        var element: NDArray.ScalarType { form == .bf16 ? .bfloat16 : .float16 }
        var bar: Double { form == .bf16 ? 4 * 0x1p-8 : 4 * 0x1p-11 }
        /// The ANE's runs: the render's only under the switch, in the shared forms.
        var neural: Bool { form == .shared && CoreAI.neuralEngineInForce }

        init(_ device: MTLDevice, _ file: SafeTensors, _ arithmetic: Arithmetic, _ g: ZImageTransformer.Geometry, x: [Float], adaln: [Float], kept: Kept) {
            (self.device, self.file, self.arithmetic, self.g, self.x, self.adaln) = (device, file, arithmetic, g, x, adaln)
            (form, weights) = (ZImageFunctions.Form(arithmetic), arithmetic.weights)
            let largest = shapes.values.map { max($0.inn, $0.out) }.max()!
            (down, up) = (kept.floats(rank * largest), kept.floats(rank * largest))
            let form = form, cases = form == .shared ? sharedCases : blockCases
            let tapped = Set(cases.flatMap { ($0.input.map { [$0] } ?? []) + $0.output })
            base = Dictionary(uniqueKeysWithValues: tapped.map { ($0, kept.floats(count($0))) })
            lora = Dictionary(uniqueKeysWithValues: tapped.map { ($0, kept.floats(count($0))) })
            o = kept.floats(form == .shared ? rows * d : 0)
            let outputs = Self.outputs(form, rows: g.sequence)
            plain = outputs.mapValues { kept.bytes($0.bytes) }
            zero = outputs.mapValues { kept.bytes($0.bytes) }
            let element = form == .bf16 ? NDArray.ScalarType.bfloat16 : .float16
            written = shapes.mapValues { s in
                let b = LowRank.bytes(inn: s.inn, out: s.out, rank: rank, element: element)
                return (down: kept.bytes(b.down), up: kept.bytes(b.up))
            }
            let neural = form == .shared && CoreAI.neuralEngineInForce
            units = neural ? Dictionary(uniqueKeysWithValues: ["gpu-base", "gpu-lora", "ane-base", "ane-lora"].flatMap { run in
                ZImageShared.neural.map { f in ("\(run).\(f)", kept.floats(rows * Self.columns(f))) }
            }) : [:]
        }

        /// The products a LoRA adapts in `f`: the Compact's modulation is the Standard's function under the guard.
        static func adapted(_ f: String, _ form: ZImageFunctions.Form) -> [String] {
            (form == .shared && f != "modulation" ? ZImageShared.adapted[f] : ZImageBlock.adapted[f]) ?? []
        }

        /// **What each function writes**, compared bit for bit between the function of no factor and the same with every
        /// factor zero: by its tensor's name, the function, whether a per-row tensor (every row of the block), its bytes.
        /// The last function's is the stream (`x`, the pass having swapped it). Over the `k` rows the layer computes, the
        /// sequence: the keys past it are never written, and hold what each stage's arena left there.
        static func outputs(_ form: ZImageFunctions.Form, rows k: Int) -> [String: (function: String, perRow: Bool, bytes: Int)] {
            let e = 2
            var all: [(String, String, Bool, Int)]
            if form == .shared {
                all = ZImageShared.gains.map { ($0, "modulation", false, d * e) }
                    + [("qkv", "pre", true, k * 3 * d * e), ("qkv_factor", "pre", true, k * e), ("mid", "out", true, k * d * e),
                       ("mid_factor", "out", true, k * e), ("x", "ffn", true, k * d * e), ("x_factor", "ffn", true, k * e)]
            } else {
                all = ["scales", "gates"].map { ($0, "modulation", false, 4 * ZImageTransformer.gainBytes) }
                    + ["q", "k", "v"].map { ($0, "pre", true, k * d * e) } + [("x", "post", true, k * d * e)]
            }
            return Dictionary(uniqueKeysWithValues: all.map { ($0.0, ($0.1, $0.2, $0.3)) })
        }

        /// The columns of pre's, out's and ffn's result: `qkv`, `mid`, the stream.
        static func columns(_ f: String) -> Int { f == "pre" ? 3 * d : d }
        static func result(_ f: String) -> String { ["pre": "qkv", "out": "mid", "ffn": "x"][f]! }

        // MARK: - The run

        mutating func run() async throws -> [Result] {
            var results: [Result] = [], line = ""
            // The function of no factor: the render's layer without a LoRA.
            try await opened(ranks: [:], part: .tapped, share: .gpu) { block in
                try await layer(block, tapped: true)
                await keep(block, into: plain)
            }
            try await opened(ranks: ranks, part: .tapped, share: .gpu) { block in
                line = block.stage.plan.line
                // Every factor zero: the same bits, function by function.
                try await layer(block, tapped: true)
                await keep(block, into: zero)
                for f in functions where !Self.adapted(f, form).isEmpty {
                    let differing = Self.outputs(form, rows: g.sequence).filter { $0.value.function == f }.keys.reduce(0) { $0 + differ(plain[$1]!, zero[$1]!) }
                    results.append(Result(name: "\(form) \(f), every factor zero", value: Double(differing), bar: nil, note: "against the function of no factor"))
                }
                await taps(block, into: base)
                if form == .shared { await block.values("o", rows: 0..<rows, into: o) }
                if neural { await keep(block, "gpu-base") }
                for c in cases {
                    let s = shapes[c.product]!
                    seed(s.inn, s.out, Self.seedOf(c.product))
                    // As large a change as the result: the factors scaled by a power of two.
                    let result = norm(c.output.map { base[$0]! }), change = changeNorm(c, s.inn, s.out)
                    let scale = Float(sign: .plus, exponent: Int(log2(result / max(change, 1e-30)).rounded()), significand: 1)
                    for i in 0..<(s.out * rank) { up[i] *= scale }
                    let (downPlace, upPlace) = (await block.pointer(factor(c.product, "down")), await block.pointer(factor(c.product, "up")))
                    try LowRank.write(down: down.baseAddress!, up: up.baseAddress!, rank: rank, inn: s.inn, out: s.out,
                                      into: downPlace, upPlace, rank: rank, element: element)
                    try await layer(block, tapped: true)
                    await taps(block, into: lora)
                    if let t = c.input {
                        results.append(Result(name: "\(form) \(c.function) \(c.product): its input", value: Double(differ(base[t]!, lora[t]!)),
                                              bar: nil, note: "with its factors and without"))
                    }
                    let (error, size) = compare(c, downPlace, upPlace, s.inn, s.out)
                    results.append(Result(name: "\(form) \(c.function) \(c.product): the change", value: error, bar: bar,
                                          note: String(format: "‖Δ‖ / ‖y‖ %.2f against the CPU's x · down · up", size)))
                    // The product's factors kept for the ANE's runs, then zero again: the next product's input is the
                    // function's without a LoRA.
                    let w = written[c.product]!
                    memcpy(w.down.baseAddress!, downPlace, w.down.count); memcpy(w.up.baseAddress!, upPlace, w.up.count)
                    memset(downPlace, 0, w.down.count); memset(upPlace, 0, w.up.count)
                }
                if neural {
                    await write(block)
                    try await layer(block, tapped: true)
                    await keep(block, "gpu-lora")
                }
            }
            described = "\(form)" + (form == .bf16 ? "" : ", \(weights)") + ", " + line
            guard neural else { return results }
            // The ANE: the render's layer with the ANE's share told, its first rows; each of its steps against the GPU's.
            guard rows < g.image else { throw Failure("the ANE's \(rows) rows leave the GPU none of the image's \(g.image)") }
            try await opened(ranks: ranks, part: .layer, share: .told(rows)) { block in
                try await layer(block, tapped: false)
                await keep(block, "ane-base")
                await write(block)
                try await layer(block, tapped: false)
                await keep(block, "ane-lora")
            }
            for f in ZImageShared.neural {
                let e = ["base", "lora"].map { trueDistance(units["ane-\($0).\(f)"]!, units["gpu-\($0).\(f)"]!) }
                results.append(Result(name: "shared \(f) on the ANE, adapted", value: e[1], bar: 2 * e[0] + 0x1p-9,
                                      note: String(format: "against the GPU's, in the layer's chain; without a LoRA %.2e (the bound 2 × that + 2⁻⁹)", e[0])))
            }
            return results
        }

        /// **The render's layer 0 at `ranks`** (`ZImageTransformer.declaration`, `part`) opened on `share`, every factor
        /// zero, `body` run on it, then let go.
        func opened(ranks: [String: Int], part: ZImageTransformer.Part, share: Block.Share, _ body: (Block) async throws -> Void) async throws {
            let block = try await Block.open(device, ZImageTransformer.declaration(g, arithmetic, ranks: ranks, part: part), files: [file],
                                             budget: MemoryBudget.machineCeiling(), share: share)
            defer { block.close() }
            for (p, r) in ranks {
                let b = LowRank.bytes(inn: shapes[p]!.inn, out: shapes[p]!.out, rank: r, element: element)
                memset(await block.pointer(factor(p, "down")), 0, b.down)
                memset(await block.pointer(factor(p, "up")), 0, b.up)
            }
            try await body(block)
        }

        /// One run of the layer from the oracle's input: the stream given again (the pass before swapped it).
        func layer(_ block: Block, tapped: Bool) async throws {
            await ZImageLayerZero.give(block, g, form, x: x, adaln: adaln)
            try await ZImageLayerZero.run(block, form: form, tapped: tapped)
        }

        /// A product's factor in the stage, under layer 0's prefix (`ZImageTransformer.lowRankTensors`).
        func factor(_ product: String, _ side: String) -> String { "layers.0.\(product)_\(side)" }

        /// Every product's factors as written, into the stage.
        func write(_ block: Block) async {
            for (p, w) in written {
                memcpy(await block.pointer(factor(p, "down")), w.down.baseAddress!, w.down.count)
                memcpy(await block.pointer(factor(p, "up")), w.up.baseAddress!, w.up.count)
            }
        }

        // MARK: - What a run leaves

        /// Each function's outputs' bytes, into `into`.
        func keep(_ block: Block, into: [String: UnsafeMutableRawBufferPointer]) async {
            for (name, o) in Self.outputs(form, rows: g.sequence) {
                let p = o.perRow ? await block.rows(name) : await block.pointer(name)
                memcpy(into[name]!.baseAddress!, p, o.bytes)
            }
        }

        /// pre's, out's and ffn's results on the ANE's rows, true values, as run `run`'s.
        func keep(_ block: Block, _ run: String) async {
            for f in ZImageShared.neural { await block.values(Self.result(f), rows: 0..<rows, into: units["\(run).\(f)"]!) }
        }

        /// The boundaries the cases read, on the first rows, into `into`.
        func taps(_ block: Block, into: [Tap: UnsafeMutableBufferPointer<Float>]) async {
            for (t, b) in into {
                let p = t == .adaln ? await block.pointer(t.rawValue) : await block.rows(t.rawValue)
                memcpy(b.baseAddress!, p, b.count * 4)
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

        /// Row `r` of the product's input, column `k`: its tap in the base run, or the function's own argument (adaln f32;
        /// o f16, a true value).
        func value(_ c: Case, _ r: Int, _ k: Int, _ inn: Int) -> Float {
            if let t = c.input { return base[t]![r * inn + k] }
            if c.function == "modulation" { return adaln[k] }
            return o[r * inn + k]
        }

        func norm(_ values: [UnsafeMutableBufferPointer<Float>]) -> Float {
            var sum = 0.0
            for v in values { for x in v { sum += Double(x) * Double(x) } }
            return Float(sum.squareRoot())
        }

        /// The CPU's change `x · down · up` row by row, each value given to `each(row, column, Δ)`; `down` `[in, r]` and
        /// `up` `[r, out]` read through `d` and `u`.
        func changes(_ c: Case, _ inn: Int, _ out: Int, down d: (Int, Int) -> Float, up u: (Int, Int) -> Float, _ each: (Int, Int, Double) -> Void) {
            let n = c.function == "modulation" ? 1 : rows
            var t = [Float](repeating: 0, count: rank)
            for r in 0..<n {
                for j in 0..<rank { t[j] = 0 }
                for k in 0..<inn {
                    let v = value(c, r, k, inn)
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
        func changeNorm(_ c: Case, _ inn: Int, _ out: Int) -> Float {
            var sum = 0.0
            changes(c, inn, out, down: { k, j in down[j * inn + k] }, up: { j, o in up[o * rank + j] }) { _, _, v in sum += v * v }
            return Float(sum.squareRoot())
        }

        /// A factor's element `i` widened to f32.
        func element(_ p: UnsafeMutableRawPointer, _ i: Int) -> Float {
            let bits = p.assumingMemoryBound(to: UInt16.self)[i]
            return form == .bf16 ? Float(bitPattern: UInt32(bits) << 16) : Float(Float16(bitPattern: bits))
        }

        /// **The change the function made against the CPU's**: (lora − base) at the product's result, against
        /// x · down · up with the factors as the function read them (`downPlace`, `upPlace`); and ‖Δ‖ / ‖y‖.
        func compare(_ c: Case, _ downPlace: UnsafeMutableRawPointer, _ upPlace: UnsafeMutableRawPointer, _ inn: Int, _ out: Int) -> (error: Double, size: Double) {
            // The factors as written, widened: `[in, R]`, `[R, out]`.
            for i in 0..<(inn * rank) { down[i] = element(downPlace, i) }
            for i in 0..<(rank * out) { up[i] = element(upPlace, i) }
            let (b, l) = (c.output.map { base[$0]! }, c.output.map { lora[$0]! })
            let w = width(c.output[0])
            var (error, change, result) = (0.0, 0.0, 0.0)
            changes(c, inn, out, down: { k, j in down[k * rank + j] }, up: { j, o in up[j * out + o] }) { r, o, v in
                let (t, col) = o.quotientAndRemainder(dividingBy: w), i = r * w + col
                let y = Double(b[t][i]), made = Double(l[t][i]) - y
                error += (made - v) * (made - v); change += v * v; result += y * y
            }
            return ((error / change).squareRoot(), (change / result).squareRoot())
        }

        /// How many 2-byte elements of `a` and `b` differ.
        func differ(_ a: UnsafeMutableRawBufferPointer, _ b: UnsafeMutableRawBufferPointer) -> Int {
            let (p, q) = (a.bindMemory(to: UInt16.self), b.bindMemory(to: UInt16.self))
            return (0..<p.count).reduce(0) { $0 + (p[$1] == q[$1] ? 0 : 1) }
        }

        /// How many values of `a` and `b` differ, bit for bit.
        func differ(_ a: UnsafeMutableBufferPointer<Float>, _ b: UnsafeMutableBufferPointer<Float>) -> Int {
            (0..<a.count).reduce(0) { $0 + (a[$1].bitPattern == b[$1].bitPattern ? 0 : 1) }
        }

        /// ‖a − b‖ / ‖b‖ of two results, true values.
        func trueDistance(_ a: UnsafeMutableBufferPointer<Float>, _ b: UnsafeMutableBufferPointer<Float>) -> Double {
            var (num, den) = (0.0, 0.0)
            for i in 0..<a.count {
                let (u, v) = (Double(a[i]), Double(b[i]))
                num += (u - v) * (u - v); den += v * v
            }
            return (num / den).squareRoot()
        }
    }
}
