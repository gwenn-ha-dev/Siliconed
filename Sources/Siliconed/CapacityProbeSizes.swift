import Foundation

/// **How big the probes are, and what they are projected to cost — set by what the machine first measured, never by its name.** Pure: no Metal, no clock. The probes (`CapacityProbes`) take their sizes from here; the memory plan sizes the
/// operands from the same list; the report writes each stage's projection beside its seconds.
///
///     tier 0   solos 4 · 30 · 309 GFLOP     sharing 30 · 60 · 121 GFLOP     attention S = 1 024 · 4 096     (the sizes of every Mac so far)
///     tier 1   solos 4 · 30 · 69            sharing 30 · 60                  attention S = 1 024 · 2 048
///     tier 2   solos 4 · 30                 sharing 30                       attention S = 1 024
///     every    the attention's pair (the cut by heads, GPU + CPU) at S = 1 024 · 4 096
///
///     engine rates  ──▶  projection(stage) = Σ runs · FLOP / rate  +  the exact rows' fp64 on the CPU  (+ constants of the stage)
///     one pulse, timed                      +  the tiles' trials: candidates × rounds × shapes, each a steady figure at its kernel's rate
///                                           +  steps × the pulse: one closes every step through the guard (`steps`)
///        │
///        ├─ before the first product   the memory plan takes the largest tier whose need fits B (`MemoryNeedDiagnostic`)
///        └─ after the first witnesses  from what each engine did on the same product (`tier(for:)`): a tier steps down only while its
///                                       projection, with the margin the heat takes (`margin`), does not fit what is left of the ceiling
///                                       (`Diagnostic.ceiling`, five minutes: in practice the memory alone chooses)
///
/// What sets it apart — what a size chosen by analogy with "scale by the chip" would get wrong:
///
///   - **The first measurement sets the tier; the memory sets its ceiling.** The operands are allocated at the largest tier B holds (and the
///     memory plan counts them), then the witnesses can only step down: a lower tier's operands are inside the allocated ones
///     (`Sizes.operandsFit`, held by a test), so no byte is added after the plan. The inventory's peak is not a guess to size by: the AMX
///     and the Neural Engine have none, and half the GPU's nominal peak puts the Neo's AMX and Neural Engine at under a third and a sixth of
///     what they did.
///   - **No tier is a different measure**: each keeps three kinds of size at least for the lines that need them (a small, a middle, a
///     large one — a line is fitted on t₀ and a rate), and the verdict's aberrant-line rule judges the fit whatever the tier.
///   - **A tier never shrinks the attention's pair** (`pairSequences`): the plan offers the cut by heads only from the k's the pair was
///     measured at (`CapacityFit.overlap`), and its sequences are the models' — 1 024 the half grid of a 1024² render, 4 096 its full grid.
///     A tier 1 that measured it at 2 048 took the cut off every full-size attention of the five families: the M1 Pro, stepped down by
///     three stages added the same evening, rendered 8 to 25 % slower. The kernels alone keep the tier's sequences: theirs are lines,
///     extrapolated like the products'; at a sequence only the pair runs, its sides are the lines' (`CapacityFit.overlap` reads them so).
///     The point costs ~0.7 s on a hot Neo and keeps it at its tier.
///   - **Every tier measures the sets at a short and a long k** (`chains`): the plan offers a pair only within the k's it was measured at
///     (`CapacityFit.overlap`), so a tier with one k took every pair off the products past twice it — a DiT's down projection, its largest
///     product (k = 10 240 or 12 288) — and left them to the GPU alone, with nothing said.
///   - **A tier steps down on the projection of what remains**, not on a chip class: the Mac that is slow because it is hot, throttled
///     or low on power is served the same way as the Mac that is slow because it is small. A fanless Mac pays `margin` for the heat the
///     witnesses measured at the start of the run and the run then adds.
///   - **The projection is the one number to tune and to audit**: each stage's seconds are written with it (`Report.stages`), so a
///     fleet of reports says how far it is from the truth on each Mac.
///   - **What a stage does beside its figures is counted, not left to the margin.** The tiles' trials (`MatmulTile`, 330 to 384 a map) and
///     a pulse after every step (400 to 560, ~18 ms each) were outside the projection: five maps of three Macs (schemas 19 to 22) took
///     1.7 to 2.1 times what they projected, the heat's margin (`margin`) eaten by them alone, and the guard's budget (`retryBudget`) set
///     against half the run. Both have a count the sizes give (`steps`, `tileTrials`): the trials are priced like any figure, the pulse
///     at what one took on this Mac a moment before. What stays outside is what no count gives: a step measured again (the guard's own
///     budget), a figure measured once more (`again`), a candidate in doubt measured again, a compilation.
extension Diagnostic.CapacityProbes {
    /// FLOP/s each engine does alone on a product of the witness's size (measured: `witness`), the CPU's attention heads — and what the
    /// projection counts beside the figures: a pulse's seconds, whether the GPU runs `matmul2d` (the tiles' trials).
    package struct Rates: Sendable, Equatable {
        package var gpu: Double
        package var amx: Double?
        package var ane: Double?
        package var cpu: Double
        /// **Seconds one pulse takes on this Mac** (`Pulser`: fourteen empty submissions, two loops on one core, two on the crowd), timed
        /// before the projection on its fast side of two; 0 counts none.
        package var pulse: Double
        /// The GPU runs the `matmul2d` kernels whose tile the map chooses (`MatmulTile.kernels`): their candidates are timed.
        package var tiled: Bool
        package init(gpu: Double, amx: Double? = nil, ane: Double? = nil, cpu: Double = 1.0e12, pulse: Double = 0, tiled: Bool = true) {
            self.gpu = gpu; self.amx = amx; self.ane = ane; self.cpu = cpu; self.pulse = pulse; self.tiled = tiled
        }
    }

    /// One tier's lists of shapes.
    package struct Sizes: Sendable, Equatable {
        package typealias Shape = (m: Int, k: Int, n: Int)
        package var tier: Int
        /// The solos' products: ≈ 4, 30, 309 GFLOP at tier 0 — k and n of a DiT's order, three different products, because an
        /// engine's rate depends on k and n as much as on W: the line is the machine's, not a shape's.
        package var solo: [Shape]
        /// The fp32 solos (GPU, AMX, ANE), taken as chains: 34 to 121 GFLOP at tier 0, chosen to condition each term of a kernel's seconds
        /// (`CapacityFit.productLine`) — two k's at one weight size and three weight sizes (the ANE's lift of the weight against its rows), three
        /// k's for the AMX's rate by k (2 048, 3 840, 12 288) —, no point under 10 GFLOP (its fixed part is all of it).
        package var product: [Shape]
        /// The ANE's fewest rows, taken as chains (fp16 and integers): a call costs it a fixed part whatever its rows, which products of 1 024 rows and
        /// more cannot tell from the host's passes — and the half grid's classes give it 256 to 768 rows.
        package var small: [Shape]
        /// **The products with a short side** (`CapacityModel.thinSide`), each kernel alone on each (one weight for all: a probe re-reads the same operand, a render does not): a short `k` or a short `n` at 2 048 rows (a LoRA's
        /// `x · down` and `mid · up`, whatever the model), few rows through a wide weight (a short prompt, a half grid's last rows). Generic shapes: no model's, no LoRA's rank — 128 and 256 are the sides the line is asked about.
        /// The last one does almost no work (10 µs of streaming and arithmetic beside a 0.1 ms call): it holds the line's `t₀`, which the others,
        /// all of them streaming and computing for as long as they wait, traded against the bandwidth's share — two maps a minute apart put the
        /// rounded activation's cost per weight 2.5 times apart on the same points (within 10 % of each other), each with its own `t₀`.
        package var thin: [Shape]
        /// The ANE's integer route taken as chains (`ane/int8`): the products' shapes but one — the host's passes over the activation and the output are told
        /// from the weight's lift and the engine's rate by a thin product (n = 1 536) and a cut K (k = 8 192, 12 288).
        package var integers: [Shape]
        /// The sets' chains (`CapacityFit.overlap`): a short k and, where the tier allows, a long one — κ by k.
        package var chains: [Shape]
        /// The pair of 8-bit kernels' products: 30, 60 and 121 GFLOP at tier 0.
        package var sharing: [Shape]
        package var attention: [Int]
        package var stairRows: [Int]

        /// The ANE's staircase: one k × n, row counts that are odd and even multiples of 256, and two halves of a unit.
        package static let stair = (k: 3840, n: 3840)
        package static let attentionHeads = 24, attentionHeadDim = 128
        /// The attention's pair, the same in every tier: the half and the full grid of a 1024² render.
        package static let pairSequences = [1024, 4096]
        /// Every sequence the attention stage runs: the tier's kernels' and the pair's.
        package var attentionRuns: [Int] { Array(Set(attention + Self.pairSequences)).sorted() }
        /// The witness: Z-Image's `[3840, 3840]` square at 2 048 rows — big enough for each engine to be timed on a few
        /// milliseconds, small enough to be run fifteen times in a second.
        package static let witness: Shape = (2048, 3840, 3840)

        package static let tiers: [Sizes] = [
            Sizes(tier: 0, solo: [(512, 2048, 2048), (1024, 3840, 3840), (4096, 6144, 6144)],
                  product: [(2048, 2048, 4096), (2048, 3840, 3840), (4096, 3840, 3840), (1024, 12288, 4096), (2048, 8192, 1536)],
                  small: [(512, 3840, 3840)],
                  thin: [(256, 3072, 8192), (2048, 3072, 128), (2048, 128, 3072), (2048, 256, 8192), (32, 3072, 8192), (256, 1536, 3072), (256, 256, 512)],
                  integers: [(2048, 2048, 4096), (2048, 3840, 3840), (1024, 12288, 4096), (2048, 8192, 1536)],
                  chains: [(4096, 3840, 3840), (4096, 12288, 4096)],
                  sharing: [(1024, 3840, 3840), (2048, 3840, 3840), (4096, 3840, 3840)], attention: [1024, 4096],
                  stairRows: [768, 896, 1024, 1792, 1920, 2048, 2304, 4096]),
            Sizes(tier: 1, solo: [(512, 2048, 2048), (1024, 3840, 3840), (2048, 4096, 4096)],
                  product: [(1024, 2048, 4096), (2048, 3840, 3840), (1024, 12288, 4096), (1024, 8192, 1536)],
                  small: [(512, 3840, 3840)],
                  thin: [(256, 3072, 8192), (2048, 3072, 128), (2048, 128, 3072), (32, 3072, 8192), (256, 1536, 3072), (256, 256, 512)],
                  integers: [(1024, 2048, 4096), (2048, 3840, 3840), (1024, 12288, 4096), (1024, 8192, 1536)],
                  chains: [(2048, 3840, 3840), (2048, 12288, 4096)],
                  sharing: [(1024, 3840, 3840), (2048, 3840, 3840)], attention: [1024, 2048],
                  stairRows: [768, 1024, 1792, 2048, 2304]),
            Sizes(tier: 2, solo: [(512, 2048, 2048), (1024, 3840, 3840)],
                  product: [(1024, 2048, 4096), (1024, 3840, 3840), (2048, 3840, 3840), (1024, 8192, 1536)],
                  small: [(512, 3840, 3840)],
                  thin: [(256, 3072, 8192), (2048, 3072, 128), (2048, 128, 3072), (32, 3072, 8192), (256, 1536, 3072), (256, 256, 512)],
                  integers: [(1024, 2048, 4096), (1024, 3840, 3840), (2048, 3840, 3840), (1024, 8192, 1536)], chains: [(2048, 3840, 3840), (2048, 12288, 4096)], sharing: [(1024, 3840, 3840)], attention: [1024],
                  stairRows: [768, 1024, 1792, 2048]),
        ]
        package static let full = tiers[0]

        package static func == (a: Sizes, b: Sizes) -> Bool { a.tier == b.tier }

        /// Every product the tier runs through the engines (the witness's included): what the operands and the engines' surfaces are sized to.
        package var products: [Shape] {
            solo + product + integers + small + thin + chains + sharing + stairRows.map { (m: $0, k: Self.stair.k, n: Self.stair.n) } + [Self.witness]
        }

        /// **The wide product a tile is chosen on** (`chooseTiles`): the largest solo's work on a long `k` (twice its `k`, half its `n`), inside
        /// the surfaces the tier's products already size (`products` leaves it out: it runs on the GPU's tiles alone, and adds neither an operand
        /// nor an ANE program). A square product cannot see a tile that loses on a long `k`, and a long `k` sees the square's order: on an
        /// M5 Pro (5/11/2026), `bf16act` 128 × 64 and 128 × 128 within 2 % of each other on 4 128 × 3 840², 13 % apart on 4 128 × 10 240 × 3 840,
        /// the candidates in the same order on both and on a long `n`, the gaps widest on the long `k` (6/11/2026: a DiT's down projection ran at
        /// 17.8 TFLOP/s against 20.8 for its up projection, on the tile the square had chosen).
        package var wide: [Shape] {
            guard let s = solo.last else { return [] }
            return [(m: s.m, k: 2 * s.k, n: s.n / 2)]
        }

        /// A lower tier's operands are inside the allocated ones: every dimension of `a`, `b`, `c` is no larger.
        package func operandsFit(in other: Sizes) -> Bool {
            func biggest(_ s: Sizes) -> (a: Int, b: Int, c: Int) {
                let p = s.products
                return (p.map { $0.m * $0.k }.max()!, p.map { $0.k * $0.n }.max()!, p.map { $0.m * $0.n }.max()!)
            }
            let (x, y) = (biggest(self), biggest(other))
            return x.a <= y.a && x.b <= y.b && x.c <= y.c
        }

        // MARK: Projection

        /// Timed runs of a figure: the judged one, then the steady ones (`CapacityProbes.steady`: two to six).
        static let runs = 4.0
        /// Seconds of the CPU's fp64 truth per `k · n` values of a shape (64 rows: the widening of `B`, then `cblas_dgemm`).
        static let truthPerValue = 1.7e-9
        /// What a kernel's rate is to the GPU's fp32 product (M1 Pro, three runs: bf16 0.62, bf16 rounded 0.49, int8 0.97, Q8_0 0.31). The
        /// 8-bit ones on a rounded activation were never run on a GPU without bf16 in hardware: priced as the rounded bf16 kernel (the same
        /// bfloat operand) and as Q8_0's (the same fill of threadgroup memory); the K-quant's as the rounded bf16 kernel (its operands' rate).
        /// W8A8 (`int8x8`) by the same rule, as the rounded kernels: it runs their tensor operations, not the int8 kernel's simdgroups, and
        /// no GPU without int8 tensor units has timed it. Where both ran, it was never slower than the rounded int8 one (M5 Pro: 3.4 to 6.3
        /// times the fp32 product, against 2.7 to 3.4): an upper bound in seconds. Priced as the int8 kernel (0.97) it hid half its time.
        static let formRatio: [String: Double] = ["bf16": 0.62, "bf16act": 0.49, "int8": 0.97, "q8": 0.31, "int8act": 0.49, "q8act": 0.31,
                                                     "q4kact": 0.49, "int8x8": 0.49]
        /// Attention kernels' rates to the GPU's product (M1 Pro: the system graph 0.40, the flash kernel 0.46).
        static let graphRatio = 0.40, flashRatio = 0.46
        /// The three stages by tiles, fp32 and fp16 operands (M1 Pro: 0.43 and 0.49 of the GPU's product; the matrix units of a bigger GPU
        /// are another figure, and the projection only prices the stage).
        static let tilesRatio = 0.43, tiles16Ratio = 0.49
        /// The forms of the two half-precision kernels: the three stages and the fused tiles each, `AttentionTile.rounds` single runs,
        /// at the tier's longest sequence, priced at the fp16 kernel's rate (a fused form is faster: an upper bound).
        static let formRuns = 2.0 * Double(1 + AttentionTile.candidates.count) * Double(AttentionTile.rounds)
        /// The seconds of those forms' choice, inside the attention's stage.
        package func formSeconds(_ rates: Rates) -> Double {
            let l = Double(attention.max() ?? 0)
            return Self.formRuns * 4 * l * l * Double(Self.attentionHeads) * Double(Self.attentionHeadDim) / (rates.gpu * Self.tiles16Ratio)
        }

        /// **The seconds the 8-bit products on a rounded activation add to each stage** (`int8act`, `q8act`: their solos, their thin
        /// runs, their pair with the ANE's integers), inside `projection`. No M1 Pro has timed them yet: the check against its stages leaves
        /// them out until one does, as it does the attention's forms.
        package func roundedEightBitSeconds(_ rates: Rates) -> [String: Double] {
            let g = rates.gpu
            func flop(_ s: Shape) -> Double { 2 * Double(s.m) * Double(s.k) * Double(s.n) }
            func truth(_ shapes: [Shape], kinds: Double) -> Double {
                Set(shapes.map { [$0.k, $0.n] }).reduce(0.0) { (sum: Double, kn: [Int]) -> Double in sum + Double(kn[0]) * Double(kn[1]) } * Self.truthPerValue * kinds
            }
            let forms = ["int8act", "q8act"].reduce(0) { $0 + 1 / Self.formRatio[$1]! }
            var p: [String: Double] = [:]
            // The K-quant's solos (`q4kact`, its two truths) are on the big shapes only: no thin product takes it.
            p["solos 8-bit and rounded"] = Self.runs * solo.reduce(0) { $0 + flop($1) } * (forms + 1 / Self.formRatio["q4kact"]!) / g
                + truth(solo, kinds: 4)
            p["thin products"] = Self.runs * thin.reduce(0) { $0 + flop($1) } * forms / g + truth(thin, kinds: 2)
            if let a = rates.ane {
                p["sharing int8"] = pairPasses.reduce(0) { $0 + chain(flop($1), rate: 0.8 * (g * Self.formRatio["int8act"]! + a)) + 0.15 }
            }
            return p
        }

        /// **The seconds W8A8 adds** (`gpu/int8x8`: its solos on the eight shapes of the rounded kernels, its thin runs, one truth of its
        /// own codes per shape), inside `projection`. No M1 Pro has timed it: the check against its stages leaves it out.
        package func quantizedSeconds(_ rates: Rates) -> [String: Double] {
            func flop(_ s: Shape) -> Double { 2 * Double(s.m) * Double(s.k) * Double(s.n) }
            func truth(_ shapes: [Shape]) -> Double {
                Set(shapes.map { [$0.k, $0.n] }).reduce(0.0) { (sum: Double, kn: [Int]) -> Double in sum + Double(kn[0]) * Double(kn[1]) } * Self.truthPerValue
            }
            let rate = rates.gpu * Self.formRatio["int8x8"]!
            let big = (solo + product).filter { Self.takes(.int8x8, $0) }, small = thin.filter { Self.takes(.int8x8, $0) }
            return ["solos 8-bit and rounded": Self.runs * big.reduce(0) { $0 + flop($1) } / rate + truth(big),
                    "thin products": Self.runs * small.reduce(0) { $0 + flop($1) } / rate + truth(small)]
        }

        /// **The seconds the rounded bf16 pair adds** (`gpu/bf16act+ane/fp16`, its own stage `sharing rounded`), inside `projection`: the chain
        /// passes of the fp32 sets at the pair's rate; its truths and floors are the fp32 sets' (the same shapes, kept). No M1 Pro has timed it.
        package func roundedPairSeconds(_ rates: Rates) -> [String: Double] {
            guard let a = rates.ane else { return [:] }
            func flop(_ s: Shape) -> Double { 2 * Double(s.m) * Double(s.k) * Double(s.n) }
            return ["sharing rounded": pairPasses.reduce(0) { $0 + chain(flop($1), rate: 0.8 * (rates.gpu * Self.formRatio["bf16act"]! + a)) + 0.15 }]
        }

        /// **The seconds the rounded kernels' extra points add** (`gpu/bf16act`, `gpu/int8act` on the fp32 solos' shapes besides their own), inside
        /// `projection`'s « solos 8-bit and rounded »: eight points a line, so that one the Mac disturbed is told apart (`CapacityFit.kept`). No M1 Pro
        /// has timed them.
        package func roundedProductSeconds(_ rates: Rates) -> [String: Double] {
            func flop(_ s: Shape) -> Double { 2 * Double(s.m) * Double(s.k) * Double(s.n) }
            let values = Set(product.map { [$0.k, $0.n] }).reduce(0.0) { (sum: Double, kn: [Int]) -> Double in sum + Double(kn[0]) * Double(kn[1]) }
            let forms = ["bf16act", "int8act"].reduce(0) { $0 + 1 / (rates.gpu * Self.formRatio[$1]!) }
            return ["solos 8-bit and rounded": Self.runs * product.reduce(0) { $0 + flop($1) } * forms + values * Self.truthPerValue * 2]
        }

        // MARK: What the stages do beside their figures

        /// **A steady figure's runs, judged one included** (`solo`, `CapacityProbes.steady`): the judged run, two runs, then the calls
        /// `steadyCalls` reads after the first steady one — on a few-millisecond product, a fiftieth of a second of them.
        static func steadySeconds(_ t: Double) -> Double {
            t * Double(2 + Diagnostic.CapacityProbes.steadyCalls(t))
        }

        /// **Whether `solo` times a GPU or AMX kernel of this form on this shape** — the conditions it puts on the shape alone: an 8-bit
        /// weight's k in whole groups or blocks, its plan and a rounded activation's rows in the reserve (`WeightProducts`), the reserve
        /// being `k · n` floats.
        static func takes(_ form: CapacityModel.Form, _ s: Shape) -> Bool {
            switch form {
            case .bf16act: return WeightProducts.roundingPlan(k: s.k, n: s.n, capacity: s.k * s.n) != nil
            case .int8x8:
                return (s.k * s.n) % 2 == 0 && s.k % Diagnostic.int8Group == 0
                    && WeightProducts.int8Plan(k: s.k, n: s.n, capacity: s.k * s.n, perColumn: false, group: Diagnostic.int8Group,
                                               amxColumns: 0, quantized: true) != nil
            case .int8, .int8act, .q8, .q8act:
                let int8 = form == .int8 || form == .int8act
                return (s.k * s.n) % 2 == 0 && s.k % (int8 ? Diagnostic.int8Group : Diagnostic.q8Block) == 0
                    && WeightProducts.int8Plan(k: s.k, n: s.n, capacity: s.k * s.n, perColumn: false, group: int8 ? Diagnostic.int8Group : nil,
                                               amxColumns: 0, block: int8 ? nil : Diagnostic.q8Block, rounded: form == .int8act || form == .q8act) != nil
            default: return true
            }
        }

        /// **The shapes a tiled kernel's candidates are timed on** (`chooseTiles`): the wide products (`wide`) for the wide tile, the thin
        /// shapes for the thin one, each that the kernel takes (`takes`).
        func tileShapes(_ kernel: CapacityModel.Kernel) -> (wide: [Shape], thin: [Shape]) {
            let takes = { (s: Shape) in Self.takes(kernel.form, s) }
            return (wide.filter(takes), thin.filter { CapacityModel.isThin(rows: $0.m, k: $0.k, n: $0.n) && takes($0) })
        }

        /// **The trials of a tiled kernel** (`chooseTiles`): each candidate, `MatmulTile.rounds(thin:)` rounds over its class's shapes — the
        /// rounds a candidate in doubt is measured again are not counted (no count gives them).
        func tileTrials(_ kernel: CapacityModel.Kernel) -> [Shape] {
            let (wide, thin) = tileShapes(kernel)
            let wideRound = Array(repeating: wide, count: MatmulTile.rounds(thin: false)).flatMap { $0 }
            let thinRound = Array(repeating: thin, count: MatmulTile.rounds(thin: true)).flatMap { $0 }
            return Array(repeating: wideRound, count: MatmulTile.candidates(for: kernel).count).flatMap { $0 }
                + Array(repeating: thinRound, count: MatmulTile.candidates(for: kernel, thin: true).count).flatMap { $0 }
        }

        /// The stage a tiled kernel's tiles are chosen in (`probes`): bf16 before its solos, the others before theirs.
        static func tileStage(_ kernel: CapacityModel.Kernel) -> String { kernel.form == .bf16 ? "solos fp32" : "solos 8-bit and rounded" }

        /// **The seconds of the tiles' trials**, inside `projection`: each a steady figure at the kernel's rate (`formRatio`).
        package func tileTrialSeconds(_ rates: Rates) -> [String: Double] {
            guard rates.tiled else { return [:] }
            var p: [String: Double] = [:]
            for kernel in MatmulTile.kernels {
                let rate = rates.gpu * Self.formRatio[kernel.form.rawValue]!
                p[Self.tileStage(kernel), default: 0] += tileTrials(kernel).reduce(0) { $0 + Self.steadySeconds(2 * Double($1.m) * Double($1.k) * Double($1.n) / rate) }
            }
            return p
        }

        /// **The steps each stage takes through the guard** (`Context.guarded`, a pulse each), measured once: the figures `probes` times
        /// on these sizes and the engines `rates` has, the tiles' trials and the attention's forms included.
        package func steps(_ rates: Rates) -> [String: Int] {
            let amx = rates.amx != nil ? 1 : 0, ane = rates.ane != nil ? 1 : 0
            func trials(_ stage: String) -> Int {
                rates.tiled ? MatmulTile.kernels.filter { Self.tileStage($0) == stage }.reduce(0) { $0 + tileTrials($1).count } : 0
            }
            var s: [String: Int] = [:]
            // The GPU's fp32, the AMX's and the ANE's chains on the products, the ANE's fewest rows, the bf16 solos.
            s["solos fp32"] = product.count * (1 + amx + ane) + ane * small.count + solo.count + trials("solos fp32")
            s["stairs"] = ane * stairRows.count
            // Six kernels at each of the tier's sequences (graph, flash, three by tiles, the CPU's heads), the pair at each of its own,
            // the GPU's wake, the two half-precision kernels' forms.
            s["attention"] = 6 * attention.count + Self.pairSequences.count + (attention.count > 1 ? 1 : 0)
                + 2 * (1 + AttentionTile.candidates.count) * AttentionTile.rounds
            let sets = amx + ane + amx * ane * 2
            s["sharing"] = sets * pairPasses.count
            // The rounded kernels and W8A8 on eight shapes (W8A8 on those it takes), the other four 8-bit ones on the solos, the ANE's integers.
            s["solos 8-bit and rounded"] = 2 * (solo.count + product.count) + (solo + product).filter { Self.takes(.int8x8, $0) }.count
                + 4 * solo.count + ane * (integers.count + small.count)
                + trials("solos 8-bit and rounded")
            // Ten kernels on each thin shape they take (`thinSolos`), the ANE on those of a whole unit of rows.
            let forms: [CapacityModel.Form] = [.fp32, .bf16, .bf16act, .int8, .q8, .int8act, .int8x8, .q8act]
            s["thin products"] = forms.reduce(0) { sum, form in sum + thin.filter { Self.takes(form, $0) }.count } + amx * thin.count
                + ane * thin.filter { $0.m >= NeuralEngine.rowUnit }.count
            s["sharing int8"] = ane * Diagnostic.CapacityProbes.Context.integerPairs.count * pairPasses.count
            s["sharing rounded"] = ane * pairPasses.count
            return s
        }

        /// **The seconds of the pulses**, inside `projection`: one per step at what one took on this Mac.
        package func pulseSeconds(_ rates: Rates) -> [String: Double] {
            steps(rates).mapValues { Double($0) * rates.pulse }
        }

        /// The chain passes of the fp32 sets (two on the short shape, one on the long), which the integer pairs take too.
        var pairPasses: [Shape] { [chains.first].compactMap { $0 } + [chains.first].compactMap { $0 } + (chains.count > 1 ? [chains[1]] : []) }

        /// A chain's seconds: the ramp, the window, at most the cap — or the calls it needs where they are long (`chainCalls` after `chainWarmCalls`).
        func chain(_ flop: Double, rate: Double) -> Double {
            typealias P = Diagnostic.CapacityProbes
            return min(P.chainCap, max(P.chainWarm + P.chainWindow, Double(P.chainWarmCalls + P.chainCalls) * flop / rate))
        }

        /// **The seconds each stage is projected to take** (the names of `m.budget`), from the engines' rates: `FLOP / rate` of each
        /// timed run, the truths the judge computes on the CPU, and the stages' constants (a program's compile, a pipeline's build).
        package func projection(_ rates: Rates) -> [String: Double] {
            let g = rates.gpu, x = rates.amx, a = rates.ane
            func flop(_ s: Shape) -> Double { 2 * Double(s.m) * Double(s.k) * Double(s.n) }
            func truth(_ shapes: [Shape], kinds: Double = 1) -> Double {
                let weights: Set<[Int]> = Set(shapes.map { [$0.k, $0.n] })
                let values: Double = weights.reduce(0.0) { (sum: Double, w: [Int]) -> Double in sum + Double(w[0]) * Double(w[1]) }
                return values * Self.truthPerValue * kinds
            }
            let soloFlop = solo.reduce(0) { $0 + flop($1) }
            var p: [String: Double] = [:]
            p["preparation"] = 0.2
            let witnessFlop = flop(Self.witness)
            let witnessTime = 7 * witnessFlop * (1 / g + (x.map { 1 / $0 } ?? 0) + (a.map { 1 / $0 } ?? 0)) + 0.05
            p["witnesses start"] = witnessTime
            p["witnesses end"] = Double(Diagnostic.CapacityProbes.endRounds) * witnessTime + Double(Diagnostic.CapacityProbes.endRounds - 1) * Diagnostic.CapacityProbes.endPause
            p["bandwidth"] = 0.02
            p["disk"] = 2.0   // the default class's three passes, then the two lower ones beside a computation (`CapacityDisk.classBudget`)
            // The GPU's fp32 solos are steady figures (it has no ramp), the AMX's and the ANE's chains.
            let productTime = product.reduce(0) { sum, s in
                sum + Self.runs * flop(s) / g + (x.map { chain(flop(s), rate: $0) } ?? 0) + (a.map { chain(flop(s), rate: $0) } ?? 0) + 0.1
            }
            let smallTime = a.map { rate in small.reduce(0) { $0 + chain(flop($1), rate: rate) + 0.1 } + truth(small) } ?? 0
            p["solos fp32"] = productTime + smallTime + Self.runs * soloFlop / (g * Self.formRatio["bf16"]!) + truth(product + solo, kinds: a == nil ? 1 : 2)
            // The short-sided products: the GPU's forms steady, the AMX's and the ANE's, the truths.
            let thinFlop = thin.reduce(0) { $0 + flop($1) }
            let gpuForms = 1 + ["bf16", "bf16act", "int8", "q8", "int8act", "q8act"].reduce(0) { $0 + 1 / Self.formRatio[$1]! }
            // The ANE compiles a program per shape it takes (those of a whole unit of rows: ≈ 0.37 s each on the M1 Pro, the stage measured
            // 2.53 s); the GPU's and the AMX's runs are a few milliseconds.
            let aneShapes = thin.filter { $0.m >= NeuralEngine.rowUnit }.count
            p["thin products"] = Self.runs * thinFlop * gpuForms / g + truth(thin, kinds: 6)
                + (x.map { Self.runs * thinFlop / $0 + 0.1 } ?? 0)
                + (a.map { Self.runs * thinFlop / $0 + Double(aneShapes) * 0.37 } ?? 0)
            let stairFlop = stairRows.reduce(0) { $0 + 2 * Double($1) * Double(Self.stair.k) * Double(Self.stair.n) }
            p["stairs"] = a.map { Self.runs * stairFlop / $0 + Double(stairRows.count) * (0.03 + truth([(1, Self.stair.k, Self.stair.n)])) } ?? 0
            let (h, d) = (Double(Self.attentionHeads), Double(Self.attentionHeadDim))
            func attentionFlop(_ sequences: [Int]) -> Double { sequences.reduce(0) { $0 + 4 * Double($1) * Double($1) * h * d } }
            // The five kernels alone at the tier's sequences, then the graph beside the CPU's heads at their balance at the pair's.
            let kernels = 1 / (g * Self.graphRatio) + 1 / (g * Self.flashRatio) + 1 / (g * Self.tilesRatio) + 2 / (g * Self.tiles16Ratio) + 1 / rates.cpu
            p["attention"] = Self.runs * (attentionFlop(attention) * kernels + attentionFlop(Self.pairSequences) / (g * Self.graphRatio + rates.cpu))
                + formSeconds(rates)
                + Double(attentionRuns.count) * 0.08
            // A chain per set and pass (the pairs with the GPU, the three, the AMX beside the ANE; a pass 3 where a long k is run), at the sets' rate.
            let setRate = 0.75 * (g + (x ?? 0) + (a ?? 0))
            let sets = (x != nil ? 1 : 0) + (a != nil ? 1 : 0) + ((x != nil && a != nil) ? 2 : 0)
            let passes = pairPasses
            p["sharing"] = Double(sets) * passes.reduce(0) { $0 + chain(flop($1), rate: setRate) + 0.15 } + truth(chains, kinds: 2)
            // The ANE's integers run as chains on the products' shapes (the host's passes are told apart by the thin one), the GPU's 8-bit kernels as steady figures.
            let integersAlone = a.map { rate in (integers + small).reduce(0) { $0 + chain(flop($1), rate: rate) + 0.1 } + truth(integers + small) } ?? 0
            // The rounded truths of the 8-bit ones are the floors of their half-precision bar (two kinds more).
            p["solos 8-bit and rounded"] = Self.runs * soloFlop * ["bf16act", "int8", "q8", "int8act", "q8act", "q4kact"].reduce(0) { $0 + 1 / (g * Self.formRatio[$1]!) }
                + truth(solo, kinds: 7) + integersAlone
            // The pairs of integer kernels (the GPU's on the fp32 activation, then on the rounded one): the chain passes of the fp32 sets'
            // (two on the short shape, one on the long), at the pair's rate.
            if let a {
                p["sharing int8"] = ["int8", "int8act"].reduce(0) { sum, form in
                    sum + passes.reduce(0) { $0 + chain(flop($1), rate: 0.8 * (g * Self.formRatio[form]! + a)) + 0.15 }
                } + truth(chains)
            }
            p.merge(roundedPairSeconds(rates)) { $1 }
            p.merge(roundedProductSeconds(rates)) { $0 + $1 }
            p.merge(quantizedSeconds(rates)) { $0 + $1 }
            p.merge(tileTrialSeconds(rates)) { $0 + $1 }
            p.merge(pulseSeconds(rates)) { $0 + $1 }
            return p
        }

        package func projectedSeconds(_ rates: Rates) -> Double { projection(rates).values.reduce(0, +) }

        /// The heat a run adds beyond what its start shows, and the noise of the projection: a tier is kept only while this much
        /// of the ceiling is left (the fanless Neo lost 1.9 × of the AMX over a minute). It is a margin only while the projection counts
        /// what the stages do (`steps`, `tileTrialSeconds`): without the trials and the pulses, the runs took 1.7 to 2.1 times it.
        package static let margin = 1.6
        /// **What the guard may spend** measuring a step again (`CapacityProbes.Context.guarded`): the seconds the ceiling leaves beyond the
        /// run as projected with `expectation`, at most `retryCap`.
        package static let expectation = 1.3, retryCap = 15.0
        /// Seconds the inventory, the fit and the report take beyond the probes.
        package static let reserve = 2.0

        /// **The first tier whose projection fits `budget` seconds**, starting from `first`; the last tier when none does.
        package static func tier(for rates: Rates, budget: Double = Diagnostic.ceiling - reserve, startingAt first: Int = 0) -> Sizes {
            for t in tiers[first...] where t.projectedSeconds(rates) * margin <= budget { return t }
            return tiers.last!
        }
    }
}
