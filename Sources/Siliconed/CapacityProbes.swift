import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

/// **The capacity probes: what a Mac's engines do alone, and what they lose together — a few dozen runs, no weights, the numbers
/// `CapacityModel` is made of.** The machine layer; the arithmetic that turns its measurements into a model is `CapacityFit` (pure,
/// tested), the sizes and their projection `CapacityProbeSizes`, what the measure found `DiagnosticFindings`. The routes are the
/// existing ones, forced one by one (`GEMM`, `Conductor`, `NeuralEngine`, `TensorGEMM`, `WeightProducts`, the three attentions); each
/// keeps the judge against the fp64 truth of the checks' rule (a wrong route fills nothing and is excluded from the plan).
///
///     run
///      ├─ plan the memory          operands, engines' surfaces, truths, attention, copy buffers — under B, before the first product,
///      │                           at the largest tier of sizes B holds
///      ├─ preparation              the weights' widening and copy rates                                              (measureWidening)
///      ├─ witnesses, start         one product on the GPU, the AMX, the ANE                                          → heat, the rates the tier is projected from
///      ├─ bandwidth                a copy kernel on the GPU                                                          → memoryBandwidth
///      ├─ disk                     the weight stream's read of a scratch file on the library's volume                → diskBandwidth
///      ├─ solos                    each kernel alone at ≈ 4, 30, 300 GFLOP                                           → lines
///      ├─ stairs                   the ANE alone at 8 row counts, one k × n                                          → aneRowUnit
///      ├─ attention                graph, flash, the CPU's heads, alone at two sequences                            → lines
///      ├─ sharing                  per set: each side alone at its rows, then the set (a balanced pass at three sizes) → keep, overhead
///      ├─ 8-bit and rounded kernels (they stage their weight in the reserve: after every fp32 route), `ane/int8` among them (as chains)
///      ├─ sharing int8             the GPU's int8 kernel beside the ANE's, both reading the integers staged in the reserve (chains, a long k),
///      │                           the GPU's on the fp32 activation, then on the activation rounded to bf16
///      ├─ sharing rounded          the GPU's rounded rows on a bf16 weight read as published beside the ANE's fp16 program on the same copy
///      │                           (chains, a long k: the Standard's own pair, `gpu/bf16act+ane/fp16`)
///      └─ witnesses, end
///
///     every timed step goes through `Context.guarded`: measured, then a pulse (`Regime`), and again after a wait while the pulse or the
///     chain's own calls say the Mac was busy and `retryBudget` has seconds
///
/// What sets it apart — what a probe written by analogy with timing routes against each other would get wrong:
///
///   - **a route is timed alone, not against another**: the map's ratios had no absolute second; a line needs seconds at
///     three sizes, and the sizes are three different products (512×2048², 1024×3840², 4096×6144²), because an engine's
///     rate depends on k and n as much as on W — the line is the machine's, not a shape's.
///   - **the sizes follow the machine** (`Sizes`): the memory plan takes the largest tier B holds, the witnesses' own rates can only step
///     it down — the ceiling is held on the slowest Mac without a line of the probes naming one.
///   - **the ANE alone is `NeuralEngine.product` over all the rows** (`first: 0, rows: m`), not a pair at the smallest share
///     (`shareRange` starts at 0.05, and the GPU would keep a twentieth of the product and its own launch). Its time is the
///     side's whole: the CPU's lift of the weight and conversion of the activation included, as in a render.
///   - **the AMX alone is `Conductor.multiply`**, BNNS on the calling thread — what the map's witness times.
///   - **each side of a set is timed twice**: alone at the exact rows the set gives it, and within the set by its own clock
///     (`GEMM.lastSides` for the ANE and the GPU side, the conductor's per-shape wall minus the idle of each side for the GPU
///     and the AMX). The difference is what the set costs that side (`keep`); what the wall adds beyond the slowest side is
///     the set's own (`overhead`). The first pass balances on the solo lines; the second, on the keep the first measured —
///     a set is fitted on balanced sides only, as it will run.
///   - **the ANE's engine holds its surfaces until it is released**: one engine for the solos, the stairs and the pair, one
///     for the three, both counted in the memory plan; a program is compiled at its first call (a new row count is a new
///     program), which a warm-up pays — the compile seconds are reported apart.
///   - **`ane/int8` is the whole route a render takes** (`WeightProducts.int8Product(…ane:)`): the integers copied into the
///     program's surface, their column sums, the rotation of the activation (convrot, group 256) on the CPU, the scale in the
///     host's fp32 pass — judged against the fp64 `(A·R)·q·s`, at the bar of the fp16 route (1.25 × bf16's own error on the
///     same product). Its partner in a set is `gpu/int8`, which reads the same reserve; the pair keeps its own sharing
///     (`gpu/int8+ane/int8`), not the fp32 pair's;
///   - **the CPU's heads cannot run without the GPU's**: the split keeps one head on the GPU (`partition`), so the CPU line is
///     measured with 23 heads of 24 on the CPU and one beside them.
///   - **the judge keeps what it saw**: for each kernel the worst channel error and the threshold of that product, and a set that
///     computed wrong as a whole (`Report.judge`).
extension Diagnostic {
    package enum CapacityProbes {
        package typealias Fit = CapacityFit
        package typealias Kernel = CapacityModel.Kernel
        package typealias Engine = CapacityModel.Engine

        /// The copy kernel's buffers, each.
        package static let bandwidthBytes = 128 << 20
        /// **A chain**: the call back to back, the first `chainWarm` seconds and `chainWarmCalls` calls thrown away (the cluster that drives
        /// the AMX and the ANE's host side ramps in a few hundred milliseconds: two similar calls after the first are the ramp, not the
        /// engine), then the median over at least `chainWindow` seconds and `chainCalls` calls, at most `chainCap` in all — the length
        /// is in calls and measured seconds, so a Mac whose calls are three times longer reads three times fewer of them, not three
        /// times longer.
        package static let chainWarm = 0.15, chainWarmCalls = 3, chainWindow = 0.4, chainCalls = 8, chainCap = 1.5
        /// **The end's witnesses, read in rounds** (the heat's numerator): three rounds a fifth of a second apart, the fast side kept.
        package static let endRounds = 3, endPause = 0.2

        /// **What each stage's seconds buy: the error the plan would make without them** (`Diagnostic.StageTime.removes`). A stage that
        /// removes no named bias has no place in the map; a stage added comes with its line here (a test holds every stage to one).
        package static let biases: [String: String] = [
            "setup": "a first call's compilation and buffers' first touch counted in a figure: paid before any figure is timed",
            "preparation": "the weight's widening or staging counted as compute: its CPU seconds per value, measured apart, priced per call — a K-quant's own (its super-blocks decoded and transposed, or copied as stored), not a bf16 shift's",
            "witnesses start": "a Mac's state read as its speed: each engine's rate at the start, against which the end tells the heat",
            "bandwidth": "a memory-bound product priced at the arithmetic rate: the memory's own rate",
            "disk": "every volume and quality of service read at one speed: the stream's own volume, by class",
            "solos fp32": "an engine's rate guessed from one shape, or bent by one disturbed figure: the lines (fixed cost, rate, per-weight cost) of the GPU, the AMX and the ANE, each figure far over the line of the others measured once more, and the GPU's bf16 kernel at the tile chosen among candidates",
            "stairs": "the GPU's time taken as smooth in rows: its steps at the tile's edges",
            "attention": "an attention priced as products: its kernels and fused forms timed as they run, the form chosen among candidates, and the GPU's wake a call pays after the CPU left it idle (the render's QK-norm and RoPE)",
            "sharing": "two engines' rates added as if each ran alone, or one disturbed chain read as the set: what each keeps beside the other (κ), by weight length, and a third chain where two at one shape disagree beyond the map's noise",
            "solos 8-bit and rounded": "a rounded, 8-bit or K-quant product priced on the fp32 line, or on a line one disturbed figure bends: their own lines, the rounded ones on eight shapes so a disturbed figure is told apart and measured once more, their tiles chosen on rounds paired with the fastest's, a candidate in doubt measured again beside it",
            "thin products": "a product of few rows or a short side priced on the wide line, or its fixed cost traded against its streaming, or a call's completion latency read as its cost: its own per-row and per-weight costs, a shape that does almost no work to hold the fixed cost, each figure on the fast mode of fifteen calls",
            "sharing int8": "the GPU and the ANE on the integers priced as two solos: what each keeps beside the other, at every weight length even when the model gives the ANE nothing (the last rows it had), and a third chain where two at one shape disagree beyond the map's noise",
            "sharing rounded": "the GPU's rounded rows and the ANE on one bf16 copy priced as two solos, or a gain within the error of its figure taken for one: what each keeps beside the other, and a third chain where two at one shape disagree beyond the map's noise, the median of three read and its MAD for the error; the ANE's per-call widening of the copy is the preparation's rate, not a second probe; every weight length measured, with the last rows the model gave when it gives the ANE nothing",
            "witnesses end": "the heat read as noise, or a burst read as heat: the end's witnesses in three rounds, their ratio applied to every rate",
            "fit": "none: the fit's own seconds, so that the map's total is accounted for",
        ]
        /// Products the fit never saw, for `validate`: the truth test of a model (a probe is judged against a long measure).
        package static let validationShapes: [(m: Int, k: Int, n: Int)] = [(2048, 3840, 10240), (1536, 4096, 4096), (3072, 3072, 3072), (1024, 2048, 8192)]

        /// A model's seconds on a product it was not fitted on, against a long measure (five warm-ups, the median of fifteen
        /// runs; a set's: the median of the steady runs at the rows the model chose).
        package struct Check: Codable, Sendable {
            package var what: String
            package var shape: [Int]
            package var rows: [String: Int]
            package var predicted: Double
            package var measured: Double
            package var error: Double { (predicted - measured) / measured }
        }

        package struct ScanPoint: Codable, Sendable {
            package var aneRows: Int
            package var wall: Double
            package var predicted: Double
            package var gpuSide: Double
            package var aneSide: Double
        }

        package struct Details: Codable, Sendable {
            /// Per kernel: the line's relative error on its own points (median, worst).
            package var lineError: [String: [Double]] = [:]
            package var stairs: Fit.Stairs?
            package var sharing: [String: Fit.SharingFit] = [:]
            /// Seconds of ANE programs' compile, summed over the run.
            package var aneCompile: Double = 0
            /// The pair around its balance (pass 0, never fitted): the model's seconds on each row count against the wall.
            package var scan: [ScanPoint] = []
            /// The ANE's rows at which the model's seconds are least, and the measured wall's.
            package var scanBest: [Int] = []
            /// The ANE's rows at which the two sides' own clocks cross (linear between the scan's points).
            package var scanCrossing: Double?
            /// `validate`, when asked.
            package var validation: [Check] = []
            /// The ANE's routes judged again on a massive activation (`Massive`).
            package var massive: [Massive] = []
        }

        /// **A half-precision route judged on a massive activation**: a few columns of the activation at 300 × the rest, the exact
        /// product in fp64, the error of the route and bf16's own on the same product, in both of the project's measures — the
        /// sampled row by its norm over the outputs (`row`, the map's) and each output column across the sampled rows (`column`, the
        /// channel of a model's boundary). The bar is `max(1e-5, 1.25 × bf16's own)` on each: a route over it on either is wrong.
        package struct Massive: Codable, Sendable {
            package var kernel: String
            package var shape: [Int]
            package var columns: [Int]
            package var rowError: Double, rowFloor: Double
            package var columnError: Double, columnFloor: Double
            package var pass: Bool
        }

        /// **What a run of the probes gives**: the raw measurements, the model they fit (before the report rounds it), what the judge said,
        /// the memory, each stage's seconds with its projection, and the fits' details (a developer's).
        package struct Result: Codable, Sendable {
            package var date: String
            package var seconds: Double
            /// The tier of sizes the probes ran at (`Sizes.tier`).
            package var tier: Int
            package var measurements: Fit.Measurements
            package var model: CapacityModel
            package var judge: [String: Diagnostic.Judge]
            package var memory: Diagnostic.Memory
            package var stages: [String: Diagnostic.StageTime]
            package var details: Details
            /// What the run says of the Mac it ran on (`Diagnostic.Regime`).
            package var regime: Diagnostic.Regime = Diagnostic.Regime()
        }

        // MARK: - The run

        /// **Runs the probes** on `machine` (read once by the caller: its inventory sizes them). Takes the machine's render lock and
        /// runs on whatever queue called: `Diagnostic.run` brings it the engine's.
        package static func run(machine: Diagnostic.Machine, cancellation: Cancellation? = nil, validate: Bool = false,
                                skipping requested: Set<Kernel> = [], log: ((String) -> Void)? = nil,
                                onStage: ((Diagnostic.Stage) -> Void)? = nil) throws(EngineError) -> Result {
            // The integers' route rides on the half-precision one (`CapacityPlan.Resources`): without the one, nothing of the other.
            var skipped = requested
            if skipped.contains(Kernel(.ane, .fp16)) { skipped.insert(Kernel(.ane, .int8)) }
            let started = Date()
            let lock = try RenderLock.acquire(RenderLock.machine)
            defer { withExtendedLifetime(lock) {} }
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw .internalFailure(component: "capacity probes", detail: "no Metal device")
            }
            OperatorFootprint.bringUp()
            if !skipped.contains(Kernel(.ane, .fp16)) { _ = NeuralEngine.bringUp }
            let gauge = Gauge()
            let clock = now()
            // The largest tier whose need fits B; the witnesses step it down only if the ceiling does not hold.
            let plan = try Plan.choose(budget: MemoryBudget.machineCeiling(), base: gauge.base) {
                need(sizes: Sizes.tiers[$0], validating: validate)
            }
            do {
                let context = try Context(device: device, queue: queue, cancellation: cancellation, gauge: gauge, validating: validate,
                                          sizes: Sizes.tiers[plan.tier], machine: machine, started: clock, skipped: skipped, log: log, onStage: onStage)
                context.setup = now() - clock
                var details = Details()
                var m = Fit.Measurements()
                try context.probes(&m, &details)
                let fitting = now()
                var model = Fit.model(from: m, allocationUnit: NeuralEngine.rowUnit)
                if let peak = machine.gpuFP32Peak { model.crest[.gpu] = peak }
                for (k, l) in model.lines {
                    let mine = m.solos.filter { $0.kernel == k.description && $0.seconds > 0 }
                    // A product's line has its shape's terms (the weight's lift, the host's passes, the rate by k): its error is the model's seconds at the shape.
                    let shaped = mine.compactMap { s -> Double? in
                        guard k.form.isProduct, s.shape.count == 3,
                              let t = model.alone(k, rows: s.shape[0], k: s.shape[1], n: s.shape[2]) else { return nil }
                        return abs(t / s.seconds - 1)
                    }
                    if !mine.isEmpty, shaped.count == mine.count {
                        details.lineError[k.description] = [Fit.median(shaped), shaped.max() ?? 0]
                    } else {
                        let e = Fit.relativeError(l, mine.map { Fit.Point(flop: $0.flop, seconds: $0.seconds) })
                        details.lineError[k.description] = [e.median, e.worst]
                    }
                }
                details.stairs = Fit.stairs(m.stairs.map { ($0.rows, $0.seconds) })
                for set in Set(m.trials.map(\.set)) {
                    details.sharing[set] = Fit.overlap(set: set, trials: Fit.chains(set: set, trials: m.trials, model: model).used, model: model)
                }
                details.aneCompile = context.aneCompile()
                let g = Kernel(.gpu, .fp32), a = Kernel(.ane, .fp16)
                for t in m.trials where t.pass == 0 && t.set == "gpu+ane" && t.shape.count == 3 {
                    guard let r = t.sides.first(where: { $0.kernel == a.description })?.rows, let gs = t.sides.first(where: { $0.kernel == g.description }),
                          let p = model.seconds(m: t.shape[0], k: t.shape[1], n: t.shape[2], rows: [a: r, g: t.shape[0] - r]) else { continue }
                    details.scan.append(ScanPoint(aneRows: r, wall: t.wall, predicted: p, gpuSide: gs.within ?? 0,
                                                  aneSide: t.sides.first { $0.kernel == a.description }?.within ?? 0))
                }
                if validate { details.validation = context.validate(model) }
                details.scan.sort { $0.aneRows < $1.aneRows }
                if let p = details.scan.min(by: { $0.predicted < $1.predicted }), let w = details.scan.min(by: { $0.wall < $1.wall }) {
                    details.scanBest = [p.aneRows, w.aneRows]
                }
                for (lo, hi) in zip(details.scan, details.scan.dropFirst()) where (lo.gpuSide - lo.aneSide) * (hi.gpuSide - hi.aneSide) <= 0 {
                    let d0 = lo.gpuSide - lo.aneSide, d1 = hi.gpuSide - hi.aneSide
                    details.scanCrossing = Double(lo.aneRows) + Double(hi.aneRows - lo.aneRows) * (d1 == d0 ? 0 : d0 / (d0 - d1))
                    break
                }
                let memory = Diagnostic.Memory(budget: plan.budget, base: plan.base, need: plan.need.peak, peak: FootprintInterval.peak())
                gauge.finish()
                let stages = context.stageTimes(fit: now() - fitting)
                context.watch.finish(into: &context.regime)
                return Result(date: ISO8601DateFormatter().string(from: started), seconds: Date().timeIntervalSince(started),
                              tier: context.sizes.tier, measurements: m, model: model, judge: context.judgements(m.wrong),
                              memory: memory, stages: stages, details: details, regime: context.regime.rounded())
            } catch {
                throw EngineError(error)
            }
        }

        // MARK: - The kernels at the shapes of a render (a bench's entry)

        /// **One kernel chained at the shapes asked**, judged and timed exactly as the probes do (`Context.solo`): what a bench compares the model's
        /// lines with, class by class. `kernels`: names as the model writes them (`gpu/fp32`, `ane/int8`…). The operands are sized for
        /// the largest shape; the memory is the caller's to hold under B (a bench of one family's classes does).
        package static func measureKernels(machine: Diagnostic.Machine, kernels: [String], shapes: [Sizes.Shape], chained: Bool = true,
                                           log: ((String) -> Void)? = nil) throws(EngineError) -> [Fit.Measurements.Solo] {
            let lock = try RenderLock.acquire(RenderLock.machine)
            defer { withExtendedLifetime(lock) {} }
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw .internalFailure(component: "capacity probes", detail: "no Metal device")
            }
            OperatorFootprint.bringUp()
            _ = NeuralEngine.bringUp
            do {
                var sizes = Sizes.full
                sizes.solo = shapes; sizes.product = []; sizes.chains = []; sizes.sharing = []; sizes.attention = []; sizes.stairRows = []
                let context = try Context(device: device, queue: queue, cancellation: nil, gauge: Gauge(), validating: false, sizes: sizes,
                                          machine: machine, started: now(), skipped: [], log: log, onStage: nil)
                var m = Fit.Measurements()
                let probes = kernels.compactMap { name in Context.Probe.allCases.first { $0.kernel.description == name } }
                for shape in shapes {
                    context.solos(probes, &m, shapes: [shape], chained: chained)
                }
                return m.solos
            } catch {
                throw EngineError(error)
            }
        }

        // MARK: - Memory

        /// The sum of everything that may be alive at once, from the shapes alone: `sizes`' tier (and the validation's products when asked).
        static func need(sizes: Sizes, validating: Bool = false) -> Need {
            let shapes = sizes.products + (validating ? validationShapes : [])
            let operands = Operands.sizes(products: shapes.map { ($0.m, $0.k, $0.n, WeightKind.q8) }).bytes
            let ps = shapes.map { (m: $0.m, k: $0.k, n: $0.n) }
            let gemm = MemoryNeed.gemmBound("diagnostic", products: ps, int8: true).reduce(0) { $0 + $1.bytes }
            let multiply = ps.map { Arena.capacity(for: Conductor.workspaceBytes(rows: $0.m, k: $0.k, n: $0.n, weightIsTransposed: true)) }.max() ?? 0
            let exact = ps.map { p in
                let r = Diagnostic.sampledRows
                return r * p.k * 12 + (1 << 25) + r * p.n * 8 * 5 + r * p.n * 4
            }.max() ?? 0
            let points = shapes.map { GridPoint(m: $0.m, k: $0.k, n: $0.n) }
            // Two engines hold their surfaces until the end: the one of the solos, the stairs and the pairs, and the three's. Only the
            // first runs the integers' route — alone on the solos, beside the GPU on the sets' products.
            let integers = (sizes.solo + sizes.integers + sizes.sharing + sizes.chains).map { GridPoint(m: $0.m, k: $0.k, n: $0.n) }
            let ane = engineBytes(points) + programCount(points) * programBytes
            let aneIntegers = engineBytes(points, integers: integers) + programCount(points, integers: integers) * programBytes
            let (h, d) = (Sizes.attentionHeads, Sizes.attentionHeadDim)
            let attention = sizes.attentionRuns.map { s -> Int in
                let split = HeadSplitAttention.hostBytes(heads: h, sequence: s, headDim: d, cpuHeads: h - 1)
                    + OperatorFootprint.sdpaQueries(rows: s, heads: 1, headDim: d)
                // The tiled kernels (one alive at a time, the fp16 variant the larger): the scores of a pass, the probabilities, the half operands.
                let tiled = TiledAttention.bytes(heads: h, kvHeads: h, sequence: s, keys: s, headDim: d, precision: .fp16)
                    .asTotal
                return 4 * Arena.capacity(for: s * h * d * 4) + max(split, OperatorFootprint.sdpaQueries(rows: s, heads: h, headDim: d), tiled)
                    + sampledQueries * h * d * 8 * 2
            }.max() ?? 0
            let truths = 160 << 20
            // The disk stage runs beside the operands: its staging slots and scratch piece are charged (`CapacityDisk.probeMemoryBytes`).
            let routes = operands + 3 * gemm + ane + aneIntegers + multiply + exact + truths + attention + 2 * bandwidthBytes
                + CapacityDisk.probeMemoryBytes()
            let w = Diagnostic.blocks[0], count = w.dim * w.hidden
            let widening = [count * 4, count * 2, count * 2, count + 2 * w.hidden, count, 1 << 20].map { Arena.capacity(for: $0) }.reduce(0, +)
            return Need(steps: ["widening": widening + frameworks, "routes": routes + frameworks])
        }

        // MARK: - The context

        enum Truth: Hashable { case fp32, rounded, int8, roundedInt8, q8, roundedQ8, q4k, roundedQ4k }
        struct TruthKey: Hashable { let truth: Truth; let m, k, n: Int }

        struct Buffers { let a, b, c: MTLBuffer }

        static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) * 1e-9 }

        /// Seconds of one execution of `body`; `nil` when it refuses.
        static func timed(_ body: () -> Bool) -> Double? {
            let t = now()
            guard body() else { return nil }
            return now() - t
        }

        /// **A steady figure**: runs until one is no longer 10 % faster than the one before (at most six), then more of them when the
        /// runs are short — `shortCalls` in all, or as many as `shortWindow` holds, four at least — read as a chain is (`figure`: the
        /// median, or the median of the quickest third when the calls disagree). Between 30 and 150 ms the median of two; a longer run
        /// is its own figure.
        ///
        /// **A short call's completion reaches the CPU in one of two modes.** An empty submission's latency (the pulse's `late`) is
        /// bimodal on an M5 Pro — 0.13–0.2 ms or 0.7–1.0 ms, about one call in two —, and a median of four sub-millisecond calls fell
        /// in either: two maps read the same thin product 62 to 238 % apart (`gpu/bf16act 256x1536x3072`: 0.58 ms against 0.26), and
        /// 40 classes of 342 changed route. A render pays that latency once per command buffer of many products, never per product: the
        /// fast mode is the product's cost. With half the calls in the slow mode, the quickest third of fifteen holds fast ones only
        /// with a probability of 0.94 (binomial); `shortWindow` bounds a figure at a fiftieth of a second, so that from 5 ms up the four
        /// calls and their median are what they were.
        static func steady(_ body: () -> Bool) -> Double? {
            var previous = Double.infinity, last = Double.infinity
            for i in 0..<6 {
                guard let t = timed(body) else { return nil }
                last = t
                if i >= 1, t >= previous * 0.9 {
                    var sample = [t]
                    for _ in 0..<(steadyCalls(t) - 1) {
                        guard let u = timed(body) else { return nil }
                        sample.append(u)
                    }
                    return figure(sample)
                }
                previous = t
            }
            return last
        }

        /// A short figure's calls (`steady`): fifteen at most, as many as a fiftieth of a second holds, four at least under 30 ms.
        package static let shortCalls = 15, shortWindow = 0.02
        /// The calls `steady` reads after the first steady one, that one included.
        package static func steadyCalls(_ t: Double) -> Int {
            if t >= 0.15 { return 1 }
            if t >= 0.03 { return 2 }
            return min(shortCalls, max(4, Int(shortWindow / max(t, 1e-9))))
        }

        /// **A chain's figure** (`chainWarm`…): the median call over the window after the ramp (its quickest third when the calls disagree,
        /// `figure`), and the calls it holds. `nil` when the body refuses.
        static func chain(_ body: () -> Bool) -> (seconds: Double, calls: Int, spread: Double)? {
            let begin = now()
            var times: [Double] = []
            var calls = 0
            while true {
                let start = now()
                guard body() else { return nil }
                let t = now() - start
                calls += 1
                if calls > chainWarmCalls, start - begin >= chainWarm { times.append(t) }
                let elapsed = now() - begin
                if elapsed >= chainWarm + chainWindow, times.count >= chainCalls { break }
                if elapsed >= chainCap, !times.isEmpty { break }
            }
            lastSpread = max(lastSpread, spread(times))
            return (figure(times), times.count, spread(times))
        }

        /// The dispersion of the figure the last `steady` or `chain` returned: what the guard reads beside the pulse. One thread measures.
        nonisolated(unsafe) static var lastSpread = 0.0

        final class Context {
            // The enclosing enum's names, for the extensions of this class in `CapacityProbes{Solos,Sharing,Validation}.swift`.
            typealias Fit = CapacityFit
            typealias Kernel = CapacityModel.Kernel
            typealias Engine = CapacityModel.Engine
            typealias CapacityProbes = Diagnostic.CapacityProbes
            typealias Sizes = CapacityProbes.Sizes
            typealias Truth = CapacityProbes.Truth
            typealias Check = CapacityProbes.Check
            let bench: Bench
            let log: ((String) -> Void)?
            let operands: Operands
            let alone: GEMM, both: GEMM, neural: GEMM, three: GEMM
            let aloneB: Buffers, bothB: Buffers, neuralB: Buffers, threeB: Buffers, bf16: MTLBuffer
            var workspace: Pages?
            var truths: [TruthKey: [Double]] = [:]
            var massives: [Massive] = []
            var floors: [[Int]: Double] = [:]
            /// The solos `again` has measured once more (kernel and shape): never a second time.
            var measuredAgain: Set<String> = []
            var dirty = 0
            var clock = CapacityProbes.now()
            /// The sizes in force: the tier the memory was planned at, stepped down once the witnesses have said what the engines do.
            var sizes: Sizes
            let machine: Diagnostic.Machine
            /// When the run began, the stage under way (for the progress), each stage's seconds and projection.
            let started: Double
            let validating: Bool
            /// The routes this measure leaves out: a process died on them (`Diagnostic.Child`). Never touched, never announced.
            let skipped: Set<Kernel>
            let onStage: ((Diagnostic.Stage) -> Void)?
            var announced: Diagnostic.Stage?
            var setup = 0.0
            let watch = LoadWatch()
            var regime = Diagnostic.Regime()
            var seconds: [String: Double] = [:]
            var projected: [String: Double] = [:]
            /// What the judge saw, per kernel (or set): the worst error against its threshold, and whether it computed wrong.
            var seen: [String: (error: Double, threshold: Double, wrong: Bool)] = [:]

            init(device: MTLDevice, queue: MTLCommandQueue, cancellation: Cancellation?, gauge: Gauge, validating: Bool, sizes: Sizes,
                 machine: Diagnostic.Machine, started: Double, skipped: Set<Kernel>, log: ((String) -> Void)?,
                 onStage: ((Diagnostic.Stage) -> Void)?) throws {
                bench = Bench(device: device, queue: queue, cancellation: cancellation, gauge: gauge)
                self.log = log; self.sizes = sizes; self.machine = machine; self.started = started; self.onStage = onStage
                self.skipped = skipped
                self.validating = validating
                let all = sizes.products + (validating ? CapacityProbes.validationShapes : [])
                operands = Operands(Operands.sizes(products: all.map { ($0.m, $0.k, $0.n, WeightKind.q8) }))
                alone = try GEMM(plan: .probing(bf16Weights: true, int8Weights: true, q8Weights: true, roundedActivations: true))
                both = try GEMM(plan: .probing(amx: true, int8Weights: true, q8Weights: true))
                neural = try GEMM(plan: .probing(ane: true))
                three = try GEMM(plan: .probing(amx: true, ane: true))
                func wrap(_ g: GEMM, _ o: Operands) throws -> Buffers {
                    Buffers(a: try g.wrap(o.a.base, bytes: o.a.capacity, name: "probe.a"), b: try g.wrap(o.b.base, bytes: o.b.capacity, name: "probe.b"),
                            c: try g.wrap(o.c.base, bytes: o.c.capacity, name: "probe.c"))
                }
                aloneB = try wrap(alone, operands); bothB = try wrap(both, operands)
                neuralB = try wrap(neural, operands); threeB = try wrap(three, operands)
                bf16 = try alone.wrap(operands.bf16.base, bytes: operands.bf16.capacity, name: "probe.bf16")
            }

            func aneCompile() -> Double {
                (neural.neuralEngine?.timings["compile"] ?? 0) + (three.neuralEngine?.timings["compile"] ?? 0)
            }

            /// Seconds since the last lap.
            func lap() -> Double {
                let t = CapacityProbes.now(), d = t - clock
                clock = t
                return d
            }

            // MARK: Truths and the judge

            func restoreB() {
                guard dirty > 0 else { return }
                Widen.bfloat16ToFloat32(source: operands.bf16.base, destination: operands.b.base, count: min(dirty, operands.bf16.capacity / 2))
                dirty = 0
            }

            func truth(_ kind: Truth, m: Int, k: Int, n: Int) -> [Double] {
                let key = TruthKey(truth: kind, m: m, k: k, n: n)
                if let t = truths[key] { return t }
                restoreB()
                let rows = Diagnostic.sampleRows(m, count: Diagnostic.sampledRows)
                let a = operands.a.floats()
                let result: [Double]
                switch kind {
                case .fp32:
                    result = Diagnostic.exactRows(rows, a: a, b: operands.b.floats(), k: k, n: n)
                case .rounded:
                    var rounded = [Float](repeating: 0, count: rows.count * k)
                    for (i, row) in rows.enumerated() { for j in 0..<k { rounded[i * k + j] = Half.bf16.round(a[row * k + j]) } }
                    result = rounded.withUnsafeBufferPointer {
                        Diagnostic.exactRows(Array(0..<rows.count), a: $0.baseAddress!, b: operands.b.floats(), k: k, n: n)
                    }
                case .int8:
                    result = Diagnostic.exactRows(rows, a: a, q: operands.q.base.assumingMemoryBound(to: Int8.self), scale: Double(Diagnostic.int8Scale),
                                                  group: Diagnostic.int8Group, k: k, n: n)
                case .roundedInt8:
                    var rounded = [Float](repeating: 0, count: rows.count * k)
                    for (i, row) in rows.enumerated() { for j in 0..<k { rounded[i * k + j] = Half.bf16.round(a[row * k + j]) } }
                    result = rounded.withUnsafeBufferPointer {
                        Diagnostic.exactRows(Array(0..<rows.count), a: $0.baseAddress!, q: operands.q.base.assumingMemoryBound(to: Int8.self),
                                             scale: Double(Diagnostic.int8Scale), group: Diagnostic.int8Group, k: k, n: n)
                    }
                case .q8:
                    result = Diagnostic.exactRows(rows, a: a, q: operands.q.base.assumingMemoryBound(to: Int8.self),
                                                  blockScales: operands.q8Scales.floats(), block: Diagnostic.q8Block, k: k, n: n)
                case .roundedQ8:
                    var rounded = [Float](repeating: 0, count: rows.count * k)
                    for (i, row) in rows.enumerated() { for j in 0..<k { rounded[i * k + j] = Half.bf16.round(a[row * k + j]) } }
                    result = rounded.withUnsafeBufferPointer {
                        Diagnostic.exactRows(Array(0..<rows.count), a: $0.baseAddress!, q: operands.q.base.assumingMemoryBound(to: Int8.self),
                                             blockScales: operands.q8Scales.floats(), block: Diagnostic.q8Block, k: k, n: n)
                    }
                case .q4k, .roundedQ4k:
                    // The published values of the Q4_K blocks, widened into `b` by the engine's own decoder (ggml's arithmetic at the
                    // bit), which `restoreB` puts back.
                    Widen.dequantizePacked(operands.kquant.base, kind: .q4_k, rows: n, columns: k, transposing: true, into: operands.b.floats())
                    dirty = max(dirty, k * n)
                    if kind == .q4k {
                        result = Diagnostic.exactRows(rows, a: a, b: operands.b.floats(), k: k, n: n)
                    } else {
                        var rounded = [Float](repeating: 0, count: rows.count * k)
                        for (i, row) in rows.enumerated() { for j in 0..<k { rounded[i * k + j] = Half.bf16.round(a[row * k + j]) } }
                        result = rounded.withUnsafeBufferPointer {
                            Diagnostic.exactRows(Array(0..<rows.count), a: $0.baseAddress!, b: operands.b.floats(), k: k, n: n)
                        }
                    }
                    restoreB()
                }
                truths[key] = result
                return result
            }

            /// bf16's own rounding of this product: the floor no half-precision path can pass under (invariant 3). `truthKind`: the exact
            /// product it is the floor of — the widened weight's (`fp32`), the integers' with their rotation (`int8`), a Q8_0's (`q8`).
            func aneThreshold(_ truthKind: Truth, m: Int, k: Int, n: Int) -> Double {
                let (exact, witnessKind, code): (Truth, Truth, Int) = switch truthKind {
                case .int8, .roundedInt8: (.int8, .roundedInt8, 1)
                case .q8, .roundedQ8: (.q8, .roundedQ8, 2)
                case .q4k, .roundedQ4k: (.q4k, .roundedQ4k, 3)
                case .fp32, .rounded: (.fp32, .rounded, 0)
                }
                if let f = floors[[m, k, n, code]] { return max(Diagnostic.fp32Threshold, Diagnostic.floorMargin * f) }
                let rows = Diagnostic.sampleRows(m, count: Diagnostic.sampledRows)
                let witness = truth(witnessKind, m: m, k: k, n: n).map { Double(Half.bf16.round(Float($0))) }
                let floor = Diagnostic.judge(witness.map(Float.init), truth(exact, m: m, k: k, n: n), columns: rows.count, threshold: 1).error
                floors[[m, k, n, code]] = floor
                return max(Diagnostic.fp32Threshold, Diagnostic.floorMargin * floor)
            }

            /// The truth of a product and, for a half-precision route (the ANE's, an 8-bit product on a rounded activation), bf16's floor
            /// under it — both computed before a weight is staged in `b`: a truth not yet kept restores `b` from its bf16 copy, and would
            /// take the staged integers away.
            func prime(_ kind: Truth, ane: Bool, m: Int, k: Int, n: Int) {
                _ = truth(kind, m: m, k: k, n: n)
                if ane { _ = aneThreshold(kind, m: m, k: k, n: n) }
            }

            /// Keeps the judge's word on `name` (a kernel, or a set of engines): the worst error in proportion to its threshold, and a verdict
            /// that stays wrong once it was.
            func keep(_ name: String, error: Double, threshold: Double, pass: Bool) {
                let ratio = error.isFinite ? error / threshold : .infinity
                if let old = seen[name] {
                    let before = old.error.isFinite ? old.error / old.threshold : .infinity
                    seen[name] = (ratio > before ? error : old.error, ratio > before ? threshold : old.threshold, old.wrong || !pass)
                } else {
                    seen[name] = (error, threshold, !pass)
                }
            }

            /// **The judge's words**: every kernel and set that was judged, and — for those in `wrong` — excluded.
            func judgements(_ wrong: [String]) -> [String: Diagnostic.Judge] {
                var out: [String: Diagnostic.Judge] = [:]
                for (name, v) in seen { out[name] = Diagnostic.Judge(error: v.error, threshold: v.threshold, excluded: v.wrong || wrong.contains(name)) }
                for kernel in skipped { out[kernel.description] = Diagnostic.Judge(error: 0, threshold: 0, excluded: true, reason: .died) }
                return out
            }

            /// Runs `body` once with the sampled rows poisoned, and judges them: the route's error, `nil` when it refused. `name` is the
            /// kernel (or the set) the judge's word is kept under.
            func judged(_ name: String, _ truthKind: Truth, ane: Bool, m: Int, k: Int, n: Int, _ body: () -> Bool) -> (pass: Bool, error: Double)? {
                let exact = truth(truthKind, m: m, k: k, n: n)
                let threshold = ane ? aneThreshold(truthKind, m: m, k: k, n: n) : Diagnostic.fp32Threshold
                let rows = Diagnostic.sampleRows(m, count: Diagnostic.sampledRows)
                let out = operands.c.floats()
                Diagnostic.poison(out, rows: rows, n: n)
                guard body() else { return nil }
                let measure = Diagnostic.judge(Diagnostic.gather(out, rows: rows, n: n), exact, columns: rows.count, threshold: threshold)
                keep(name, error: measure.error, threshold: threshold, pass: measure.pass)
                return (measure.pass, measure.error)
            }

            // MARK: The massive activation

            /// The activation's massive columns: four, in different rotation groups (K is a multiple of 256 where it matters).
            static func massiveColumns(_ k: Int) -> [Int] { [k / 7, k / 3, k / 2 + 5, k - 11] }
            static let massiveScale: Float = 300

            /// **The route judged on an activation with a few massive columns** (`Massive`): the middle size's `a` with its four columns
            /// ×300, `run` once, the sampled rows against the fp64 product of that activation (`truthKind`: the widened weight's or
            /// the integers'), and bf16's own error on the same product as the bar. The activation is put back after. A route that
            /// rounds the activation in fp16 spreads, or concentrates, the error of a massive channel differently — the rotation of
            /// a convrot weight (the int8 route) spreads a channel's rounding over the 256 outputs of its group; the fp16 route keeps
            /// it with the weight it multiplies — and the row measure alone does not see it: the column does.
            func massive(_ truthKind: Truth, kernel: String, m: Int, k: Int, n: Int, _ run: () -> Bool) -> Massive? {
                let ints = truthKind == .int8
                if !ints { restoreB() }                 // the widened weight, not the integers staged in it
                let rows = Diagnostic.sampleRows(m, count: Diagnostic.sampledRows)
                let a = operands.a.floats(), columns = Self.massiveColumns(k)
                var saved = [Float](repeating: 0, count: columns.count * m)
                for (i, c) in columns.enumerated() {
                    for r in 0..<m { saved[i * m + r] = a[r * k + c]; a[r * k + c] *= Self.massiveScale }
                }
                defer { for (i, c) in columns.enumerated() { for r in 0..<m { a[r * k + c] = saved[i * m + r] } } }
                func exact(_ activation: UnsafePointer<Float>, _ pick: [Int]) -> [Double] {
                    ints ? Diagnostic.exactRows(pick, a: activation, q: operands.q.base.assumingMemoryBound(to: Int8.self),
                                                scale: Double(Diagnostic.int8Scale), group: Diagnostic.int8Group, k: k, n: n)
                         : Diagnostic.exactRows(pick, a: activation, b: operands.b.floats(), k: k, n: n)
                }
                let truth = exact(UnsafePointer(a), rows)
                var rounded = [Float](repeating: 0, count: rows.count * k)
                for (i, row) in rows.enumerated() { for j in 0..<k { rounded[i * k + j] = Half.bf16.round(a[row * k + j]) } }
                let witness = rounded.withUnsafeBufferPointer { exact($0.baseAddress!, Array(0..<rows.count)) }.map { Half.bf16.round(Float($0)) }
                // The two measures of a result laid out `[n, rows]`: by sampled row, and by output column (laid out `[rows, n]`).
                func measures(_ got: [Float]) -> (row: Double, column: Double) {
                    let row = truth.withUnsafeBufferPointer { channelError(got, $0, columns: rows.count) }.worst
                    var gotT = [Float](repeating: 0, count: got.count), truthT = [Double](repeating: 0, count: got.count)
                    for j in 0..<n { for i in 0..<rows.count { gotT[i * n + j] = got[j * rows.count + i]; truthT[i * n + j] = truth[j * rows.count + i] } }
                    let column = truthT.withUnsafeBufferPointer { channelError(gotT, $0, columns: n) }.worst
                    return (row, column)
                }
                let floor = measures(witness)
                let out = operands.c.floats()
                Diagnostic.poison(out, rows: rows, n: n)
                guard run() else { return nil }
                let error = measures(Diagnostic.gather(out, rows: rows, n: n))
                let bar = { (floor: Double) in max(Diagnostic.fp32Threshold, Diagnostic.floorMargin * floor) }
                let check = Massive(kernel: kernel, shape: [m, k, n], columns: columns, rowError: error.row, rowFloor: floor.row,
                                    columnError: error.column, columnFloor: floor.column,
                                    pass: error.row <= bar(floor.row) && error.column <= bar(floor.column))
                massives.append(check)
                log?(String(format: "  massive %@ %dx%dx%d  row %.2e (bf16 %.2e)  column %.2e (bf16 %.2e)  %@", kernel, m, k, n, error.row, floor.row,
                            error.column, floor.column, check.pass ? "✓" : "WRONG"))
                return check
            }

            // MARK: Engines alone

            func gpu(_ m: Int, _ k: Int, _ n: Int) -> Bool {
                _ = alone.linear(a: aloneB.a, b: aloneB.b, c: aloneB.c, m: m, k: k, n: n, weightIsTransposed: true)
                return true
            }

            func amx(_ m: Int, _ k: Int, _ n: Int) -> Bool {
                Conductor.multiply(a: operands.a.base, b: operands.b.base, c: operands.c.base, rows: m, k: k, n: n,
                                   transposeWeight: false, workspace: &workspace).status == 0
            }

            /// The ANE over all `rows` of an `m`-row product: `first: 0`, the whole height, no GPU beside it.
            func ane(_ engine: NeuralEngine, rows: Int, _ k: Int, _ n: Int) -> Bool {
                let before = engine.fallbacks
                let ok = engine.product(a: operands.a.floats(), w: operands.b.floats(), c: operands.c.floats(), first: 0, rows: rows, k: k, n: n)
                return ok && engine.fallbacks == before
            }

            /// The int8 operands as a render stages them (`WeightProducts.int8Plan`: the integers at the reserve's head, the scale
            /// after, the rotation of group 256), no AMX columns; `nil` where the plan does not fit or K is not a whole number of groups.
            func int8Plan(_ k: Int, _ n: Int, rounded: Bool = false) -> WeightProducts.Int8Plan? {
                guard (k * n) % 2 == 0, k % Diagnostic.int8Group == 0 else { return nil }
                return WeightProducts.int8Plan(k: k, n: n, capacity: k * n, perColumn: false, group: Diagnostic.int8Group, amxColumns: 0,
                                               rounded: rounded)
            }

            /// The integers and their scale into the reserve (`b`), which the bf16 copy restores.
            func stageInt8(_ plan: WeightProducts.Int8Plan, k: Int, n: Int) {
                Widen.copyBfloat16(source: operands.q.base, destination: operands.b.base, count: k * n / 2)
                (operands.b.base + plan.scalesOffset).storeBytes(of: Diagnostic.int8Scale, as: Float.self)
                dirty = max(dirty, k * n)
            }

            /// The ANE over `rows` rows of an int8 product staged in `b`: `first: 0`, no GPU beside it, as `ane` for the fp16 program.
            func aneInt8(_ engine: NeuralEngine, _ plan: WeightProducts.Int8Plan, rows: Int, _ k: Int, _ n: Int) -> Bool {
                let raw = operands.b.base
                let weight = NeuralEngine.Int8Weight(values: raw.assumingMemoryBound(to: Int8.self),
                                                     scales: (raw + plan.scalesOffset).assumingMemoryBound(to: Float.self),
                                                     perColumn: plan.perColumn, group: plan.group)
                let before = engine.fallbacks
                let ok = engine.product(a: operands.a.floats(), int8: weight, c: operands.c.floats(), first: 0, rows: rows, k: k, n: n)
                return ok && engine.fallbacks == before
            }

            // MARK: Probes

            func probes(_ m: inout Fit.Measurements, _ d: inout Details) throws {
                /// The progress group each probe belongs to: the app names it in the user's language.
                let groups: [String: Diagnostic.Stage] = [
                    "preparation": .widening, "witnesses start": .witness, "bandwidth": .witness, "disk": .witness, "solos fp32": .engines, "stairs": .engines,
                    "attention": .engines, "sharing": .sharing, "solos 8-bit and rounded": .forms, "thin products": .forms, "witnesses end": .regime,
                ]
                func stage(_ name: String, _ body: () throws -> Void) throws {
                    try bench.check()
                    if let group = groups[name], group != announced { announced = group; onStage?(group) }
                    _ = lap()
                    watch.begin(name)
                    try body()
                    watch.end(name)
                    seconds[name] = lap()
                    log?(String(format: "%@ %.2f s", name, seconds[name]!) + (projected[name].map { String(format: " (projected %.2f)", $0) } ?? ""))
                }
                try stage("preparation") { m.preparation = try preparation() }
                try stage("witnesses start") { m.witnesses = witnesses().map { .init(engine: $0.key, before: $0.value, after: 0) } }
                project(m.witnesses)
                try stage("bandwidth") { m.memoryBandwidth = bandwidth() }
                try stage("disk") {
                    // The volume the render's stream reads: the library's, or this Mac's working copy of it (`CapacityDisk.streamFolder`).
                    let folder = CapacityDisk.streamFolder(for: Library.processWide)
                    let readings = CapacityDisk.measureClasses(in: FileManager.default.isWritableFile(atPath: folder.path)
                                                               ? folder : FileManager.default.temporaryDirectory)
                    m.diskBandwidth = readings?[.standard]?.rate
                    m.diskClasses = Dictionary(uniqueKeysWithValues: (readings ?? [:]).map { ($0.key.rawValue, $0.value) })
                }
                try stage("solos fp32") { solos([.gpu_fp32], &m, shapes: sizes.product); solos([.amx_fp32, .ane_fp16], &m, shapes: sizes.product, chained: true); solos([.ane_fp16], &m, shapes: sizes.small, chained: true); chooseTiles(.gpu_bf16, &m); solos([.gpu_bf16], &m) }
                try stage("stairs") { stairs(&m) }
                try stage("attention") { try attention(&m) }
                try stage("sharing") { sharing(&m) }
                try stage("solos 8-bit and rounded") {
                    chooseTiles(.gpu_bf16act, &m); chooseTiles(.gpu_int8, &m); chooseTiles(.gpu_int8act, &m)
                    // The rounded kernels the Standard and the Compact run on every large class take the fp32 solos' shapes too: eight points,
                    // so that a figure the Mac disturbed is told from the line and left out (`CapacityFit.kept`) — of three, none can be.
                    solos([.gpu_bf16act, .gpu_int8act], &m, shapes: sizes.solo + sizes.product)
                    solos([.gpu_int8, .gpu_q8, .gpu_q8act, .gpu_q4kact], &m)
                    solos([.ane_int8], &m, shapes: sizes.integers + sizes.small, chained: true)
                }
                try stage("thin products") { thinSolos(&m) }
                try stage("sharing int8") { sharingInt8(&m) }
                try stage("sharing rounded") { sharingRounded(&m) }
                d.massive = massives
                try stage("witnesses end") {
                    restoreB()
                    // The heat is the end's witness over the start's, applied to every rate of the plan (`CapacityModel.heat`): one
                    // round read through a burst of another program took a fifth off a fanned GPU that holds its rate for minutes.
                    // A disturbance only adds time and lasts less than a second; the heat lasts. So the end is read in three rounds
                    // a fifth of a second apart, each engine on its fast side.
                    var end = witnesses()
                    for _ in 1..<CapacityProbes.endRounds {
                        Thread.sleep(forTimeInterval: CapacityProbes.endPause)
                        for (engine, t) in witnesses() { end[engine] = min(end[engine] ?? t, t) }
                    }
                    m.witnesses = m.witnesses.map { w in .init(engine: w.engine, before: w.before, after: end[w.engine] ?? w.before) }
                }
            }

            // MARK: The Mac while it measures

            /// The pulse's instruments, made when the first step asks (a few tens of milliseconds, once).
            lazy var pulser = Pulser(threads: machine.performanceCores, device: bench.device, queue: bench.queue)
            /// The pulse that closed the step before.
            var lastPulse: Diagnostic.Regime.Pulse?
            /// Seconds the guard may still spend measuring again and waiting for the Mac to be quiet (`project`).
            var retryBudget = 0.0
            /// The most times a step is measured.
            static let attempts = 3

            /// **A pulse**: how the Mac is just now. Recorded in the report when it closes a step.
            func takePulse(step: String, attempt: Int, record: Bool) -> Diagnostic.Regime.Pulse {
                let w = pulser.wait()
                let c = pulser.crowdAndAlone()
                let pulse = Diagnostic.Regime.Pulse(at: CapacityProbes.now() - started, step: step, attempt: attempt, crowd: c.crowd, alone: c.alone,
                                                    wait: w.median, late: w.longest)
                if record { regime.pulses.append(pulse) }
                return pulse
            }

            /// **A step measured when the Mac is free, or as free as the ceiling allows.** `body` is the step's timing: the figure it
            /// returns is kept when the pulse that closes it says the threads still had their cores and the GPU its promptness, and
            /// the calls inside it agreed (`Regime.Rules`). Otherwise it waits — for a quiet pulse, at most two seconds — and measures
            /// it again, three times at most and while `retryBudget` has seconds. What it could not redo is kept — its chains read on the fast side
            /// (`figure`) — and said by the findings (`Regime.disturbances`) as an uncertainty: a figure taken on a busy Mac is never passed for
            /// the Mac's, and never refused either.
            func guarded<T>(_ step: String, _ body: () -> T?) -> T? {
                if lastPulse == nil { lastPulse = takePulse(step: "start", attempt: 0, record: true) }
                if let last = lastPulse, regime.busy(last), retryBudget > 0 { waitForQuiet(step) }
                var attempt = 0
                while true {
                    attempt += 1
                    let began = CapacityProbes.now()
                    CapacityProbes.lastSpread = 0
                    let result = body()
                    let spread = CapacityProbes.lastSpread
                    let end = takePulse(step: step, attempt: attempt, record: true)
                    lastPulse = end
                    regime.spreads[step] = spread
                    guard regime.busy(end) || spread > Diagnostic.Regime.Rules.spread else {
                        if attempt > 1 { regime.repeated += 1 }
                        return result
                    }
                    let cost = CapacityProbes.now() - began
                    guard attempt < Self.attempts, retryBudget > cost else { return result }
                    retryBudget -= cost
                    regime.repeatedSeconds += cost
                    log?(String(format: "  %@: the Mac was busy (crowd %.2f, wait %.2f ms, spread %.2f): measured again", step, end.crowd, end.wait * 1e3, spread))
                    waitForQuiet(step)
                }
            }

            /// Polls the pulse every quarter of a second until it is quiet, at most two seconds and what the budget has left.
            func waitForQuiet(_ step: String) {
                let began = CapacityProbes.now()
                let limit = min(2.0, retryBudget)
                while CapacityProbes.now() - began < limit {
                    Thread.sleep(forTimeInterval: 0.25)
                    let pulse = takePulse(step: step, attempt: 0, record: false)
                    lastPulse = pulse
                    if !regime.busy(pulse) { break }
                }
                let spent = CapacityProbes.now() - began
                retryBudget -= spent
                regime.repeatedSeconds += spent
            }

            /// **Projects the rest of the run from what the engines did on the witness**, and steps the tier down while the ceiling does not hold.
            /// The sizes only shrink here: the operands were planned for the tier the inventory allowed, and a lower tier's are inside them.
            func project(_ witnesses: [Fit.Measurements.Witness]) {
                let flop = 2.0 * Double(Sizes.witness.m) * Double(Sizes.witness.k) * Double(Sizes.witness.n)
                func rate(_ engine: String) -> Double? { witnesses.first { $0.engine == engine }.flatMap { $0.before > 0 ? flop / $0.before : nil } }
                guard let gpu = rate("gpu") else { return }
                let cores = Double(machine.performanceCores + machine.efficiencyCores)
                let rates = Rates(gpu: gpu, amx: rate("amx"), ane: rate("ane"), cpu: cores > 0 ? 0.125e12 * cores : 1.0e12)
                let left = Diagnostic.ceiling - Sizes.reserve - (CapacityProbes.now() - started)
                let chosen = Sizes.tier(for: rates, budget: left, startingAt: sizes.tier)
                if chosen.tier != sizes.tier {
                    log?("the projection \(Int(sizes.projectedSeconds(rates) * Sizes.margin)) s does not fit the \(Int(left)) s left: tier \(sizes.tier) → \(chosen.tier)")
                    sizes = chosen
                }
                // Preparation and the first witnesses have run: they were not projected.
                projected = sizes.projection(rates).filter { $0.key != "preparation" && $0.key != "witnesses start" }
                // What the ceiling leaves beyond the run as projected is what the guard may spend: none on a Mac that fills it.
                retryBudget = max(0, min(Sizes.retryCap, left - sizes.projectedSeconds(rates) * Sizes.expectation))
            }

            /// Each stage's seconds with its projection, and the fixed costs beside them.
            func stageTimes(fit: Double) -> [String: Diagnostic.StageTime] {
                var out: [String: Diagnostic.StageTime] = [:]
                for name in Set(seconds.keys).union(projected.keys) {
                    out[name] = Diagnostic.StageTime(seconds: seconds[name] ?? 0, projected: projected[name], removes: CapacityProbes.biases[name])
                }
                out["setup"] = Diagnostic.StageTime(seconds: setup, removes: CapacityProbes.biases["setup"])
                out["fit"] = Diagnostic.StageTime(seconds: fit, removes: CapacityProbes.biases["fit"])
                return out
            }

            func preparation() throws -> [String: [String: Double]] {
                let widening = try Diagnostic.measureWidening(bench)
                func per(_ kind: String) -> Double {
                    guard let w = widening.first(where: { $0.kind == kind }) else { return 0 }
                    return w.cpu.error == 0 && w.cpu.rate > 0 ? 1e-9 / w.cpu.rate : 0
                }
                // A widening that is not exact at the bit is a defect of the code, not of the Mac.
                if let bad = widening.first(where: { $0.cpu.error != 0 }) {
                    throw EngineError.internalFailure(component: "capacity probes", detail: "the \(bad.kind) widening is not exact")
                }
                return [WeightKind.bf16.rawValue: [WeightPath.widened.rawValue: per("bf16"), WeightPath.bf16.rawValue: per("bf16 copy")],
                        WeightKind.int8.rawValue: [WeightPath.widened.rawValue: per("int8 convrot"), WeightPath.native.rawValue: per("int8 copy")],
                        WeightKind.q8.rawValue: [WeightPath.widened.rawValue: per("q8_0"), WeightPath.native.rawValue: per("q8_0 copy")],
                        // A K-quant has no `WeightKind`: the fit reads these two into `kquantPreparation` and `kquantStaging`.
                        CapacityModel.kquantKind: [WeightPath.widened.rawValue: per("q4_k"), WeightPath.native.rawValue: per("q4_k copy")]]
            }

            /// The same product on each engine alone, median of five (after two warm-ups).
            func witnesses() -> [String: Double] {
                let (m, k, n) = Sizes.witness
                var result: [String: Double] = [:]
                func median(_ body: () -> Bool) -> Double? {
                    for _ in 0..<2 { guard body() else { return nil } }
                    var t: [Double] = []
                    for _ in 0..<5 { guard let s = CapacityProbes.timed(body) else { return nil }; t.append(s) }
                    return Fit.median(t)
                }
                restoreB()
                if let t = median({ self.gpu(m, k, n) }) { result["gpu"] = t }
                if let t = median({ self.amx(m, k, n) }) { result["amx"] = t }
                if !skipped.contains(Kernel(.ane, .fp16)), let e = neural.neuralEngine, let t = median({ self.ane(e, rows: m, k, n) }) { result["ane"] = t }
                return result
            }

            func bandwidth() -> Double? {
                let device = bench.device
                let source = """
                #include <metal_stdlib>
                using namespace metal;
                kernel void copy(device const float4* s [[buffer(0)]], device float4* d [[buffer(1)]], uint i [[thread_position_in_grid]]) { d[i] = s[i]; }
                """
                guard let library = try? device.makeLibrary(source: source, options: nil), let function = library.makeFunction(name: "copy"),
                      let pipeline = try? device.makeComputePipelineState(function: function),
                      let s = device.makeBuffer(length: bandwidthBytes, options: .storageModePrivate),
                      let d = device.makeBuffer(length: bandwidthBytes, options: .storageModePrivate) else { return nil }
                let count = bandwidthBytes / 16
                var times: [Double] = []
                for i in 0..<8 {
                    guard let cb = bench.queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { return nil }
                    e.setComputePipelineState(pipeline)
                    e.setBuffer(s, offset: 0, index: 0); e.setBuffer(d, offset: 0, index: 1)
                    e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    e.endEncoding()
                    cb.commit(); cb.waitUntilCompleted()
                    if i >= 2 { times.append(cb.gpuEndTime - cb.gpuStartTime) }
                }
                let t = Fit.median(times)
                return t > 0 ? 2 * Double(bandwidthBytes) / t : nil
            }

        }
    }
}
