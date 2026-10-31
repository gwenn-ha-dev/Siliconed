import Foundation

/// **How big the probes are, and what they are projected to cost — set by what the machine first measured, never by its name.** Pure: no Metal, no clock. The probes (`CapacityProbes`) take their sizes from here; the memory plan sizes the
/// operands from the same list; the report writes each stage's projection beside its seconds.
///
///     tier 0   solos 4 · 30 · 309 GFLOP     sharing 30 · 60 · 121 GFLOP     attention S = 1 024 · 4 096     (the sizes of every Mac so far)
///     tier 1   solos 4 · 30 · 69            sharing 30 · 60                  attention S = 1 024 · 2 048
///     tier 2   solos 4 · 30                 sharing 30                       attention S = 1 024
///
///     engine rates  ──▶  projection(stage) = Σ runs · FLOP / rate  +  the exact rows' fp64 on the CPU  (+ constants of the stage)
///        │
///        ├─ before the first product   the memory plan takes the largest tier whose need fits B (`MemoryNeedDiagnostic`)
///        └─ after the first witnesses  from what each engine did on the same product (`tier(for:)`): a tier steps down while its
///                                       projection, with the margin the heat takes (`margin`), does not fit what is left of the minute
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
///   - **A tier steps down on the projection of what remains**, not on a chip class: the Mac that is slow because it is hot, throttled
///     or low on power is served the same way as the Mac that is slow because it is small. A fanless Mac pays `margin` for the heat the
///     witnesses measured at the start of the run and the run then adds.
///   - **The projection is the one number to tune and to audit**: each stage's seconds are written with it (`Report.stages`), so a
///     fleet of reports says how far it is from the truth on each Mac.
extension Diagnostic.CapacityProbes {
    /// FLOP/s each engine does alone on a product of the witness's size (measured: `witness`), the CPU's attention heads.
    package struct Rates: Sendable, Equatable {
        package var gpu: Double
        package var amx: Double?
        package var ane: Double?
        package var cpu: Double
        package init(gpu: Double, amx: Double? = nil, ane: Double? = nil, cpu: Double = 1.0e12) {
            self.gpu = gpu; self.amx = amx; self.ane = ane; self.cpu = cpu
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
        /// The witness: Z-Image's `[3840, 3840]` square at 2 048 rows — big enough for each engine to be timed on a few
        /// milliseconds, small enough to be run fifteen times in a second.
        package static let witness: Shape = (2048, 3840, 3840)

        package static let tiers: [Sizes] = [
            Sizes(tier: 0, solo: [(512, 2048, 2048), (1024, 3840, 3840), (4096, 6144, 6144)],
                  product: [(2048, 2048, 4096), (2048, 3840, 3840), (4096, 3840, 3840), (1024, 12288, 4096), (2048, 8192, 1536)],
                  small: [(512, 3840, 3840)],
                  integers: [(2048, 2048, 4096), (2048, 3840, 3840), (1024, 12288, 4096), (2048, 8192, 1536)],
                  chains: [(4096, 3840, 3840), (4096, 12288, 4096)],
                  sharing: [(1024, 3840, 3840), (2048, 3840, 3840), (4096, 3840, 3840)], attention: [1024, 4096],
                  stairRows: [768, 896, 1024, 1792, 1920, 2048, 2304, 4096]),
            Sizes(tier: 1, solo: [(512, 2048, 2048), (1024, 3840, 3840), (2048, 4096, 4096)],
                  product: [(1024, 2048, 4096), (2048, 3840, 3840), (1024, 12288, 4096), (1024, 8192, 1536)],
                  small: [(512, 3840, 3840)],
                  integers: [(1024, 2048, 4096), (2048, 3840, 3840), (1024, 12288, 4096), (1024, 8192, 1536)],
                  chains: [(2048, 3840, 3840), (2048, 12288, 4096)],
                  sharing: [(1024, 3840, 3840), (2048, 3840, 3840)], attention: [1024, 2048],
                  stairRows: [768, 1024, 1792, 2048, 2304]),
            Sizes(tier: 2, solo: [(512, 2048, 2048), (1024, 3840, 3840)],
                  product: [(1024, 2048, 4096), (1024, 3840, 3840), (2048, 3840, 3840), (1024, 8192, 1536)],
                  small: [(512, 3840, 3840)],
                  integers: [(1024, 2048, 4096), (1024, 3840, 3840), (2048, 3840, 3840), (1024, 8192, 1536)], chains: [(2048, 3840, 3840)], sharing: [(1024, 3840, 3840)], attention: [1024],
                  stairRows: [768, 1024, 1792, 2048]),
        ]
        package static let full = tiers[0]

        package static func == (a: Sizes, b: Sizes) -> Bool { a.tier == b.tier }

        /// Every product the tier runs through the engines (the witness's included): what the operands and the engines' surfaces are sized to.
        package var products: [Shape] {
            solo + product + integers + small + chains + sharing + stairRows.map { (m: $0, k: Self.stair.k, n: Self.stair.n) } + [Self.witness]
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
        /// What a kernel's rate is to the GPU's fp32 product (M1 Pro, three runs: bf16 0.62, bf16 rounded 0.49, int8 0.97, Q8_0 0.31).
        static let formRatio: [String: Double] = ["bf16": 0.62, "bf16act": 0.49, "int8": 0.97, "q8": 0.31]
        /// Attention kernels' rates to the GPU's product (M1 Pro: the system graph 0.40, the flash kernel 0.46).
        static let graphRatio = 0.40, flashRatio = 0.46

        /// **The seconds each stage is projected to take** (the names of `m.budget`), from the engines' rates: `FLOP / rate` of each
        /// timed run, the truths the judge computes on the CPU, and the stages' constants (a program's compile, a pipeline's build).
        package func projection(_ rates: Rates) -> [String: Double] {
            let g = rates.gpu, x = rates.amx, a = rates.ane
            func flop(_ s: Shape) -> Double { 2 * Double(s.m) * Double(s.k) * Double(s.n) }
            func truth(_ shapes: [Shape], kinds: Double = 1) -> Double {
                Set(shapes.map { [$0.k, $0.n] }).reduce(0) { $0 + Double($1[0]) * Double($1[1]) } * Self.truthPerValue * kinds
            }
            let soloFlop = solo.reduce(0) { $0 + flop($1) }
            var p: [String: Double] = [:]
            p["preparation"] = 0.2
            let witnessFlop = flop(Self.witness)
            let witnessTime = 7 * witnessFlop * (1 / g + (x.map { 1 / $0 } ?? 0) + (a.map { 1 / $0 } ?? 0)) + 0.05
            p["witnesses start"] = witnessTime
            p["witnesses end"] = witnessTime
            p["bandwidth"] = 0.02
            p["disk"] = 0.6
            // A chain's seconds: the ramp, the window, at most the cap — or the calls it needs where they are long (`chainCalls` after `chainWarmCalls`).
            func chain(_ flop: Double, rate: Double) -> Double {
                typealias P = Diagnostic.CapacityProbes
                return min(P.chainCap, max(P.chainWarm + P.chainWindow, Double(P.chainWarmCalls + P.chainCalls) * flop / rate))
            }
            // The GPU's fp32 solos are steady figures (it has no ramp), the AMX's and the ANE's chains.
            let productTime = product.reduce(0) { sum, s in
                sum + Self.runs * flop(s) / g + (x.map { chain(flop(s), rate: $0) } ?? 0) + (a.map { chain(flop(s), rate: $0) } ?? 0) + 0.1
            }
            let smallTime = a.map { rate in small.reduce(0) { $0 + chain(flop($1), rate: rate) + 0.1 } + truth(small) } ?? 0
            p["solos fp32"] = productTime + smallTime + Self.runs * soloFlop / (g * Self.formRatio["bf16"]!) + truth(product + solo, kinds: a == nil ? 1 : 2)
            let stairFlop = stairRows.reduce(0) { $0 + 2 * Double($1) * Double(Self.stair.k) * Double(Self.stair.n) }
            p["stairs"] = a.map { Self.runs * stairFlop / $0 + Double(stairRows.count) * (0.03 + truth([(1, Self.stair.k, Self.stair.n)])) } ?? 0
            let (h, d) = (Double(Self.attentionHeads), Double(Self.attentionHeadDim))
            let attentionFlop = attention.reduce(0) { $0 + 4 * Double($1) * Double($1) * h * d }
            // The three kernels alone, then the graph beside the CPU's heads at their balance.
            p["attention"] = Self.runs * attentionFlop * (1 / (g * Self.graphRatio) + 1 / (g * Self.flashRatio) + 1 / rates.cpu
                                                          + 1 / (g * Self.graphRatio + rates.cpu)) + Double(attention.count) * 0.08
            // A chain per set and pass (the pairs with the GPU, the three, the AMX beside the ANE; a pass 3 where a long k is run), at the sets' rate.
            let setRate = 0.75 * (g + (x ?? 0) + (a ?? 0))
            let sets = (x != nil ? 1 : 0) + (a != nil ? 1 : 0) + ((x != nil && a != nil) ? 2 : 0)
            let passes = [chains.first].compactMap { $0 } + [chains.first].compactMap { $0 } + (chains.count > 1 ? [chains[1]] : [])
            p["sharing"] = Double(sets) * passes.reduce(0) { $0 + chain(flop($1), rate: setRate) + 0.15 } + truth(chains, kinds: 2)
            // The ANE's integers run as chains on the products' shapes (the host's passes are told apart by the thin one), the GPU's 8-bit kernels as steady figures.
            let integersAlone = a.map { rate in (integers + small).reduce(0) { $0 + chain(flop($1), rate: rate) + 0.1 } + truth(integers + small) } ?? 0
            p["solos 8-bit and rounded"] = Self.runs * soloFlop * (1 / (g * Self.formRatio["bf16act"]!) + 1 / (g * Self.formRatio["int8"]!)
                                                                       + 1 / (g * Self.formRatio["q8"]!)) + truth(solo, kinds: 3) + integersAlone
            // The pair of integer kernels: the chain passes of the fp32 sets' (two on the short shape, one on the long), at the pair's rate.
            if let a {
                let pairRate = 0.8 * (g * Self.formRatio["int8"]! + a)
                p["sharing int8"] = passes.reduce(0) { $0 + chain(flop($1), rate: pairRate) + 0.15 } + truth(chains)
            }
            return p
        }

        package func projectedSeconds(_ rates: Rates) -> Double { projection(rates).values.reduce(0, +) }

        /// The heat a run adds beyond what its start shows, and the noise of the projection: a tier is kept only while this much
        /// of the minute is left (the fanless Neo lost 1.9 × of the AMX over a minute).
        package static let margin = 1.6
        /// **What the guard may spend** measuring a step again (`CapacityProbes.Context.guarded`): the seconds the minute leaves beyond the
        /// run as projected with `expectation` (the projection is within 5–10 % of the seconds the stages take here), at most `retryCap`.
        package static let expectation = 1.3, retryCap = 15.0
        /// Seconds the inventory, the fit and the report take beyond the probes.
        package static let reserve = 2.0

        /// **The first tier whose projection fits `budget` seconds**, starting from `first`; the last tier when none does.
        package static func tier(for rates: Rates, budget: Double = Diagnostic.minute - reserve, startingAt first: Int = 0) -> Sizes {
            for t in tiers[first...] where t.projectedSeconds(rates) * margin <= budget { return t }
            return tiers.last!
        }
    }
}
