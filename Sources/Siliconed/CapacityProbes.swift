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
///      ├─ solos                    each kernel alone at ≈ 4, 30, 300 GFLOP                                           → lines
///      ├─ stairs                   the ANE alone at 8 row counts, one k × n                                          → aneRowUnit
///      ├─ attention                graph, flash, the CPU's heads, alone at two sequences                            → lines
///      ├─ sharing                  per set: each side alone at its rows, then the set (a balanced pass at three sizes) → keep, overhead
///      ├─ 8-bit and rounded kernels (they stage their weight in the reserve: after every fp32 route), `ane/int8` among them (as chains)
///      ├─ sharing int8             the GPU's int8 kernel beside the ANE's, both reading the integers staged in the reserve (chains, a long k)
///      └─ witnesses, end
///
///     every timed step goes through `Context.guarded`: measured, then a pulse (`Regime`), and again after a wait while the pulse or the
///     chain's own calls say the Mac was busy and `retryBudget` has seconds
///
/// What sets it apart — what a probe written by analogy with the old map's route timings would get wrong:
///
///   - **a route is timed alone, not against another**: the map's ratios had no absolute second; a line needs seconds at
///     three sizes, and the sizes are three different products (512×2048², 1024×3840², 4096×6144²), because an engine's
///     rate depends on k and n as much as on W — the line is the machine's, not a shape's.
///   - **the sizes follow the machine** (`Sizes`): the memory plan takes the largest tier B holds, the witnesses' own rates can only step
///     it down — the minute is held on the slowest Mac without a line of the probes naming one.
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
            // The largest tier whose need fits B; the witnesses step it down if the minute does not hold.
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
                        guard [.fp32, .bf16, .bf16act, .int8, .q8, .fp16].contains(k.form), s.shape.count == 3,
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
                    let passes = Set(m.trials.filter { $0.set == set }.map(\.pass))
                    let from = passes.contains(3) ? 2 : (passes.max() ?? 0)
                    details.sharing[set] = Fit.overlap(set: set, trials: m.trials.filter { $0.pass >= from }, model: model)
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
            let attention = sizes.attention.map { s -> Int in
                let split = HeadSplitAttention.hostBytes(heads: h, sequence: s, headDim: d, cpuHeads: h - 1)
                    + OperatorFootprint.sdpaQueries(rows: s, heads: 1, headDim: d)
                return 4 * Arena.capacity(for: s * h * d * 4) + max(split, OperatorFootprint.sdpaQueries(rows: s, heads: h, headDim: d))
                    + sampledQueries * h * d * 8 * 2
            }.max() ?? 0
            let truths = 160 << 20
            let routes = operands + 3 * gemm + ane + aneIntegers + multiply + exact + truths + attention + 2 * bandwidthBytes
            let w = Diagnostic.blocks[0], count = w.dim * w.hidden
            let widening = [count * 4, count * 2, count * 2, count + 2 * w.hidden, count, 1 << 20].map { Arena.capacity(for: $0) }.reduce(0, +)
            return Need(steps: ["widening": widening + frameworks, "routes": routes + frameworks])
        }

        // MARK: - The context

        private enum Truth: Hashable { case fp32, rounded, int8, roundedInt8, q8 }
        private struct TruthKey: Hashable { let truth: Truth; let m, k, n: Int }

        private struct Buffers { let a, b, c: MTLBuffer }

        private static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) * 1e-9 }

        /// Seconds of one execution of `body`; `nil` when it refuses.
        private static func timed(_ body: () -> Bool) -> Double? {
            let t = now()
            guard body() else { return nil }
            return now() - t
        }

        /// **A steady figure**: runs until one is no longer 10 % faster than the one before (at most six), then a few more
        /// when the runs are short, and the median. A long run (> 150 ms) is its own figure.
        private static func steady(_ body: () -> Bool) -> Double? {
            var previous = Double.infinity, last = Double.infinity
            for i in 0..<6 {
                guard let t = timed(body) else { return nil }
                last = t
                if i >= 1, t >= previous * 0.9 {
                    var sample = [t]
                    for _ in 0..<(t < 0.03 ? 3 : t < 0.15 ? 1 : 0) {
                        guard let u = timed(body) else { return nil }
                        sample.append(u)
                    }
                    return Fit.median(sample)
                }
                previous = t
            }
            return last
        }

        /// **A chain's figure** (`chainWarm`…): the median call over the window after the ramp (its quickest third when the calls disagree,
        /// `figure`), and the calls it holds. `nil` when the body refuses.
        private static func chain(_ body: () -> Bool) -> (seconds: Double, calls: Int, spread: Double)? {
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

        private final class Context {
            let bench: Bench
            let log: ((String) -> Void)?
            let operands: Operands
            let alone: GEMM, both: GEMM, neural: GEMM, three: GEMM
            let aloneB: Buffers, bothB: Buffers, neuralB: Buffers, threeB: Buffers, bf16: MTLBuffer
            var workspace: Pages?
            var truths: [TruthKey: [Double]] = [:]
            var massives: [Massive] = []
            var floors: [[Int]: Double] = [:]
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
            private var seen: [String: (error: Double, threshold: Double, wrong: Bool)] = [:]

            init(device: MTLDevice, queue: MTLCommandQueue, cancellation: Cancellation?, gauge: Gauge, validating: Bool, sizes: Sizes,
                 machine: Diagnostic.Machine, started: Double, skipped: Set<Kernel>, log: ((String) -> Void)?,
                 onStage: ((Diagnostic.Stage) -> Void)?) throws {
                bench = Bench(device: device, queue: queue, cancellation: cancellation, gauge: gauge)
                self.log = log; self.sizes = sizes; self.machine = machine; self.started = started; self.onStage = onStage
                self.skipped = skipped
                self.validating = validating
                let all = sizes.products + (validating ? CapacityProbes.validationShapes : [])
                operands = Operands(Operands.sizes(products: all.map { ($0.m, $0.k, $0.n, WeightKind.q8) }))
                alone = try GEMM(freezeCut: true, plan: .probing(bf16Weights: true, roundedActivations: true))
                both = try GEMM(freezeCut: true, plan: .probing(amx: true, int8Weights: true, q8Weights: true))
                neural = try GEMM(freezeCut: true, plan: .probing(ane: true))
                three = try GEMM(freezeCut: true, plan: .probing(amx: true, ane: true))
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

            private func truth(_ kind: Truth, m: Int, k: Int, n: Int) -> [Double] {
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
                }
                truths[key] = result
                return result
            }

            /// bf16's own rounding of this product: the floor no fp16 path can pass under (invariant 3). `truthKind`: the exact
            /// product it is the floor of — the widened weight's (`fp32`) or the integers' with their rotation (`int8`).
            private func aneThreshold(_ truthKind: Truth, m: Int, k: Int, n: Int) -> Double {
                let ints = truthKind == .int8
                if let f = floors[[m, k, n, ints ? 1 : 0]] { return max(Diagnostic.fp32Threshold, Diagnostic.floorMargin * f) }
                let rows = Diagnostic.sampleRows(m, count: Diagnostic.sampledRows)
                let witness = truth(ints ? .roundedInt8 : .rounded, m: m, k: k, n: n).map { Double(Half.bf16.round(Float($0))) }
                let floor = Diagnostic.judge(witness.map(Float.init), truth(truthKind, m: m, k: k, n: n), columns: rows.count, threshold: 1).error
                floors[[m, k, n, ints ? 1 : 0]] = floor
                return max(Diagnostic.fp32Threshold, Diagnostic.floorMargin * floor)
            }

            /// The truth of a product and, for the ANE, bf16's floor under it — both computed before a weight is staged in `b`: a truth
            /// not yet kept restores `b` from its bf16 copy, and would take the staged integers away.
            func prime(_ kind: Truth, ane: Bool, m: Int, k: Int, n: Int) {
                _ = truth(kind, m: m, k: k, n: n)
                if ane { _ = aneThreshold(kind == .int8 ? .int8 : .fp32, m: m, k: k, n: n) }
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
                let threshold = ane ? aneThreshold(truthKind == .int8 ? .int8 : .fp32, m: m, k: k, n: n) : Diagnostic.fp32Threshold
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
            func int8Plan(_ k: Int, _ n: Int) -> WeightProducts.Int8Plan? {
                guard (k * n) % 2 == 0, k % Diagnostic.int8Group == 0 else { return nil }
                return WeightProducts.int8Plan(k: k, n: n, capacity: k * n, perColumn: false, group: Diagnostic.int8Group, amxColumns: 0)
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
                    "preparation": .widening, "witnesses start": .witness, "bandwidth": .witness, "solos fp32": .engines, "stairs": .engines,
                    "attention": .engines, "sharing": .sharing, "solos 8-bit and rounded": .forms, "witnesses end": .regime,
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
                try stage("solos fp32") { solos([.gpu_fp32], &m, shapes: sizes.product); solos([.amx_fp32, .ane_fp16], &m, shapes: sizes.product, chained: true); solos([.ane_fp16], &m, shapes: sizes.small, chained: true); solos([.gpu_bf16], &m) }
                try stage("stairs") { stairs(&m) }
                try stage("attention") { try attention(&m) }
                try stage("sharing") { sharing(&m) }
                try stage("solos 8-bit and rounded") { solos([.gpu_bf16act, .gpu_int8, .gpu_q8], &m); solos([.ane_int8], &m, shapes: sizes.integers + sizes.small, chained: true) }
                try stage("sharing int8") { sharingInt8(&m) }
                d.massive = massives
                try stage("witnesses end") {
                    restoreB()
                    let end = witnesses()
                    m.witnesses = m.witnesses.map { w in .init(engine: w.engine, before: w.before, after: end[w.engine] ?? w.before) }
                }
            }

            // MARK: The Mac while it measures

            /// The pulse's instruments, made when the first step asks (a few tens of milliseconds, once).
            private lazy var pulser = Pulser(threads: machine.performanceCores, device: bench.device, queue: bench.queue)
            /// The pulse that closed the step before.
            private var lastPulse: Diagnostic.Regime.Pulse?
            /// Seconds the guard may still spend measuring again and waiting for the Mac to be quiet (`project`).
            var retryBudget = 0.0
            /// The most times a step is measured.
            static let attempts = 3

            /// **A pulse**: how the Mac is just now. Recorded in the report when it closes a step.
            private func takePulse(step: String, attempt: Int, record: Bool) -> Diagnostic.Regime.Pulse {
                let w = pulser.wait()
                let c = pulser.crowdAndAlone()
                let pulse = Diagnostic.Regime.Pulse(at: CapacityProbes.now() - started, step: step, attempt: attempt, crowd: c.crowd, alone: c.alone,
                                                    wait: w.median, late: w.longest)
                if record { regime.pulses.append(pulse) }
                return pulse
            }

            /// **A step measured when the Mac is free, or as free as the minute allows.** `body` is the step's timing: the figure it
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
            private func waitForQuiet(_ step: String) {
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

            /// **Projects the rest of the run from what the engines did on the witness**, and steps the tier down while the minute does not hold.
            /// The sizes only shrink here: the operands were planned for the tier the inventory allowed, and a lower tier's are inside them.
            func project(_ witnesses: [Fit.Measurements.Witness]) {
                let flop = 2.0 * Double(Sizes.witness.m) * Double(Sizes.witness.k) * Double(Sizes.witness.n)
                func rate(_ engine: String) -> Double? { witnesses.first { $0.engine == engine }.flatMap { $0.before > 0 ? flop / $0.before : nil } }
                guard let gpu = rate("gpu") else { return }
                let cores = Double(machine.performanceCores + machine.efficiencyCores)
                let rates = Rates(gpu: gpu, amx: rate("amx"), ane: rate("ane"), cpu: cores > 0 ? 0.125e12 * cores : 1.0e12)
                let left = Diagnostic.minute - Sizes.reserve - (CapacityProbes.now() - started)
                let chosen = Sizes.tier(for: rates, budget: left, startingAt: sizes.tier)
                if chosen.tier != sizes.tier {
                    log?("the projection \(Int(sizes.projectedSeconds(rates) * Sizes.margin)) s does not fit the \(Int(left)) s left: tier \(sizes.tier) → \(chosen.tier)")
                    sizes = chosen
                }
                // Preparation and the first witnesses have run: they were not projected.
                projected = sizes.projection(rates).filter { $0.key != "preparation" && $0.key != "witnesses start" }
                // What the minute leaves beyond the run as projected is what the guard may spend: none on a Mac that fills it.
                retryBudget = max(0, min(Sizes.retryCap, left - sizes.projectedSeconds(rates) * Sizes.expectation))
            }

            /// Each stage's seconds with its projection, and the fixed costs beside them.
            func stageTimes(fit: Double) -> [String: Diagnostic.StageTime] {
                var out = seconds.mapValues { Diagnostic.StageTime(seconds: $0) }
                for (name, p) in projected { out[name] = Diagnostic.StageTime(seconds: seconds[name] ?? 0, projected: p) }
                out["setup"] = Diagnostic.StageTime(seconds: setup)
                out["fit"] = Diagnostic.StageTime(seconds: fit)
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
                        WeightKind.q8.rawValue: [WeightPath.widened.rawValue: per("q8_0"), WeightPath.native.rawValue: per("q8_0 copy")]]
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

            // MARK: Solos

            enum Probe: String, CaseIterable { case gpu_fp32, amx_fp32, ane_fp16, ane_int8, gpu_bf16, gpu_bf16act, gpu_int8, gpu_q8
                var kernel: Kernel {
                    switch self {
                    case .gpu_fp32: return Kernel(.gpu, .fp32)
                    case .amx_fp32: return Kernel(.amx, .fp32)
                    case .ane_fp16: return Kernel(.ane, .fp16)
                    case .ane_int8: return Kernel(.ane, .int8)
                    case .gpu_bf16: return Kernel(.gpu, .bf16)
                    case .gpu_bf16act: return Kernel(.gpu, .bf16act)
                    case .gpu_int8: return Kernel(.gpu, .int8)
                    case .gpu_q8: return Kernel(.gpu, .q8)
                    }
                }
            }

            func solos(_ probes: [Probe], _ m: inout Fit.Measurements, shapes: [Sizes.Shape]? = nil, chained: Bool = false) {
                for probe in probes {
                    for (rows, k, n) in (shapes ?? sizes.solo).map({ ($0.m, $0.k, $0.n) }) {
                        guard let s = solo(probe, m: rows, k: k, n: n, chained: chained) else { continue }
                        if s.error != nil, s.seconds == 0 {
                            m.wrong.append(probe.kernel.description)
                            log?(String(format: "  %@ %dx%dx%d  WRONG, error %.3e", probe.kernel.description, rows, k, n, s.error ?? 0))
                            continue
                        }
                        m.solos.append(s)
                        log?(String(format: "  %@ %dx%dx%d  %.2f ms  %.2f TFLOP/s  err %.2e", probe.kernel.description, rows, k, n,
                                    s.seconds * 1e3, s.flop / s.seconds * 1e-12, s.error ?? 0))
                    }
                }
            }

            /// One kernel on one product: judged on its first run, then timed. A kernel that judged wrong returns seconds 0.
            func solo(_ probe: Probe, m: Int, k: Int, n: Int, chained: Bool = false) -> Fit.Measurements.Solo? {
                guard !skipped.contains(probe.kernel) else { return nil }
                let flop = 2.0 * Double(m) * Double(k) * Double(n)
                let name = probe.kernel.description
                RouteBeacon.announce(name)
                let run: () -> Bool
                var truthKind = Truth.fp32
                var isANE = false
                var staged: (() -> Void)?
                switch probe {
                case .gpu_fp32:
                    restoreB(); run = { self.gpu(m, k, n) }
                case .amx_fp32:
                    restoreB(); run = { self.amx(m, k, n) }
                case .ane_fp16:
                    guard let e = neural.neuralEngine else { return nil }
                    restoreB(); isANE = true
                    run = { self.ane(e, rows: m, k, n) }
                case .ane_int8:
                    guard let e = neural.neuralEngine, let plan = int8Plan(k, n) else { return nil }
                    truthKind = .int8; isANE = true
                    staged = { [self] in stageInt8(plan, k: k, n: n) }
                    run = { self.aneInt8(e, plan, rows: m, k, n) }
                case .gpu_bf16:
                    guard alone.tensor != nil, alone.tensorBF16 else { return nil }
                    restoreB()
                    run = { _ = self.alone.linear(a: self.aloneB.a, bf16: self.bf16, c: self.aloneB.c, m: m, k: k, n: n, weightIsTransposed: true); return true }
                case .gpu_bf16act:
                    guard alone.bf16Activations, let plan = WeightProducts.roundingPlan(k: k, n: n, capacity: k * n) else { return nil }
                    truthKind = .rounded
                    staged = { [self] in
                        Widen.copyBfloat16(source: operands.bf16.base, destination: operands.b.base, count: k * n)
                        dirty = max(dirty, k * n)
                    }
                    run = {
                        _ = self.alone.linear(a: self.aloneB.a, rounding: (self.aloneB.b, plan.offset, plan.rows), bf16: self.aloneB.b,
                                              c: self.aloneB.c, m: m, k: k, n: n, weightIsTransposed: true)
                        return true
                    }
                case .gpu_int8, .gpu_q8:
                    let isInt8 = probe == .gpu_int8
                    guard isInt8 ? both.tensorInt8 : both.tensorQ8, (k * n) % 2 == 0, isInt8 ? k % Diagnostic.int8Group == 0 : k % Diagnostic.q8Block == 0,
                          let plan = WeightProducts.int8Plan(k: k, n: n, capacity: k * n, perColumn: false, group: isInt8 ? Diagnostic.int8Group : nil,
                                                             amxColumns: 0, block: isInt8 ? nil : Diagnostic.q8Block) else { return nil }
                    truthKind = isInt8 ? .int8 : .q8
                    staged = { [self] in
                        Widen.copyBfloat16(source: operands.q.base, destination: operands.b.base, count: k * n / 2)
                        if isInt8 {
                            (operands.b.base + plan.scalesOffset).storeBytes(of: Diagnostic.int8Scale, as: Float.self)
                        } else {
                            (operands.b.base + plan.scalesOffset).copyMemory(from: operands.q8Scales.base, byteCount: (k + Diagnostic.q8Block - 1) / Diagnostic.q8Block * n * 4)
                        }
                        dirty = max(dirty, k * n)
                    }
                    run = { WeightProducts.int8Product(gemm: self.both, plan: plan, reserve: self.bothB.b, a: self.bothB.a, c: self.bothB.c, m: m, k: k, n: n).status == 0 }
                }
                // The truth first (it restores B for the fp32 ones), then the staged weight.
                prime(truthKind, ane: isANE, m: m, k: k, n: n)
                staged?()
                guard let verdict = judged(name, truthKind, ane: isANE, m: m, k: k, n: n, run) else { return nil }
                guard verdict.pass else { return .init(kernel: name, shape: [m, k, n], flop: flop, seconds: 0, error: verdict.error) }
                // An ANE route is judged again on a massive activation, at the middle size (the exact product of a few more rows costs
                // what the solo does): over its bar there, the route is wrong on this Mac, as on any product.
                if isANE, sizes.solo.count > 1 || sizes.product.count > 1,
                   (sizes.product.count > 1 && (m, k, n) == (sizes.product[1].m, sizes.product[1].k, sizes.product[1].n))
                    || (sizes.solo.count > 1 && (m, k, n) == (sizes.solo[1].m, sizes.solo[1].k, sizes.solo[1].n)),
                   let massive = massive(truthKind, kernel: name, m: m, k: k, n: n, run), !massive.pass {
                    return .init(kernel: name, shape: [m, k, n], flop: flop, seconds: 0, error: max(massive.rowError, massive.columnError))
                }
                guard let t = guarded("\(name) \(m)x\(k)x\(n)", { () -> Double? in
                    chained ? CapacityProbes.chain(run)?.seconds : CapacityProbes.steady(run)
                }) else { return nil }
                return .init(kernel: name, shape: [m, k, n], flop: flop, seconds: t, error: verdict.error)
            }

            // MARK: The ANE's staircase

            func stairs(_ m: inout Fit.Measurements) {
                guard !skipped.contains(Kernel(.ane, .fp16)), let engine = neural.neuralEngine else { return }
                RouteBeacon.announce("ane/fp16")
                restoreB()
                m.stairShape = [Sizes.stair.k, Sizes.stair.n]
                for rows in sizes.stairRows {
                    let (k, n) = Sizes.stair
                    let run = { self.ane(engine, rows: rows, k, n) }
                    guard let v = judged("ane/fp16", .fp32, ane: true, m: rows, k: k, n: n, run) else { continue }
                    guard v.pass else { m.wrong.append("ane/fp16"); continue }
                    guard let t = guarded("stair \(rows)", { CapacityProbes.steady(run) }) else { continue }
                    m.stairs.append(.init(rows: rows, seconds: t))
                    log?(String(format: "  stair %d rows  %.2f ms", rows, t * 1e3))
                }
            }

            // MARK: Attention

            func attention(_ measurements: inout Fit.Measurements) throws {
                let (h, d) = (Sizes.attentionHeads, Sizes.attentionHeadDim)
                let floats = sizes.attention.max()! * h * d
                let pages = (q: Pages(bytes: floats * 4), k: Pages(bytes: floats * 4), v: Pages(bytes: floats * 4), out: Pages(bytes: floats * 4))
                Diagnostic.fill(pages.q.floats(), count: floats, scale: 3, seed: 11)
                Diagnostic.fill(pages.k.floats(), count: floats, scale: 3, seed: 12)
                Diagnostic.fill(pages.v.floats(), count: floats, scale: 1, seed: 13)
                let buffers = (q: try bench.buffer(pages.q), k: try bench.buffer(pages.k), v: try bench.buffer(pages.v), out: try bench.buffer(pages.out))
                let out = pages.out.floats()
                for s in sizes.attention {
                    autoreleasepool {
                        let rows = Diagnostic.sampleRows(s, count: Diagnostic.sampledQueries)
                        let exact = Diagnostic.exactAttention(q: pages.q.floats(), k: pages.k.floats(), v: pages.v.floats(), sequence: s, heads: h, headDim: d, rows: rows)
                        let flop = 4.0 * Double(s) * Double(s) * Double(h) * Double(d)
                        func verdict(_ kernel: String) -> (pass: Bool, error: Double) {
                            let width = h * d, vectors = rows.count * h
                            var got = [Float](repeating: 0, count: vectors * d)
                            for (i, row) in rows.enumerated() {
                                for head in 0..<h { for x in 0..<d { got[x * vectors + i * h + head] = out[row * width + head * d + x] } }
                            }
                            let judged = Diagnostic.judge(got, exact, columns: vectors, threshold: Diagnostic.fp32Threshold)
                            keep(kernel, error: judged.error, threshold: Diagnostic.fp32Threshold, pass: judged.pass)
                            return judged
                        }
                        func poison() { for row in rows { (out + row * h * d).update(repeating: .nan, count: h * d) } }
                        func record(_ kernel: Kernel, flop: Double, run: () -> Bool, shape: [Int]) {
                            poison()
                            guard run() else { return }
                            let v = verdict(kernel.description)
                            guard v.pass else { measurements.wrong.append(kernel.description); return }
                            guard let t = guarded("\(kernel.description) S=\(s)", { CapacityProbes.steady(run) }) else { return }
                            measurements.solos.append(.init(kernel: kernel.description, shape: shape, flop: flop, seconds: t, error: v.error))
                            log?(String(format: "  %@ S=%d  %.2f ms  %.2f TFLOP/s", kernel.description, s, t * 1e3, flop / t * 1e-12))
                        }
                        let graph = Attention(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d)
                        RouteBeacon.announce("gpu/graph")
                        record(Kernel(.gpu, .graph), flop: flop, run: { _ = graph.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out); return true }, shape: [s, h, d])
                        RouteBeacon.announce("gpu/flash")
                        if !skipped.contains(Kernel(.gpu, .flash)), let flash = try? FlashMatrix(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d) {
                            record(Kernel(.gpu, .flash), flop: flop, run: { _ = flash.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out); return true }, shape: [s, h, d])
                        }
                        // The CPU's heads: all but one of them, the figure being the CPU part's own wall.
                        RouteBeacon.announce("cpu/heads")
                        if !skipped.contains(Kernel(.cpu, .heads)), let split = try? HeadSplitAttention(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d,
                                                               cpuFraction: Double(h - 1) / Double(h)) {
                            nonisolated(unsafe) let (qp, kp, vp, op) = (pages.q.floats(), pages.k.floats(), pages.v.floats(), out)
                            var cpuSeconds = 0.0
                            let run = {
                                let r = split.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out, q: qp, k: kp, v: vp, out: op)
                                cpuSeconds = r.cpu
                                return true
                            }
                            poison()
                            if run() {
                                let v = verdict("cpu/heads")
                                if !v.pass { measurements.wrong.append("cpu/heads") } else {
                                    let heads = guarded("cpu/heads S=\(s)") { () -> Double? in
                                        var times: [Double] = []
                                        var previous = Double.infinity
                                        for i in 0..<5 {
                                            _ = run()
                                            if i >= 1 { times.append(cpuSeconds) }
                                            if i >= 1, cpuSeconds >= previous * 0.9, times.count >= 2 { break }
                                            previous = cpuSeconds
                                        }
                                        return Fit.median(Array(times.suffix(3)))
                                    }
                                    guard let t = heads else { return }
                                    let cpuFlop = 4.0 * Double(s) * Double(s) * Double(d) * Double(split.cpuHeads)
                                    measurements.solos.append(.init(kernel: Kernel(.cpu, .heads).description, shape: [s, h, d, split.cpuHeads],
                                                                    flop: cpuFlop, seconds: t, error: v.error))
                                    log?(String(format: "  cpu/heads S=%d (%d heads)  %.2f ms  %.2f TFLOP/s", s, split.cpuHeads, t * 1e3, cpuFlop / t * 1e-12))
                                    attentionSet(&measurements, sequence: s, cpuSolo: (split.cpuHeads, t), flop: flop, pages: pages, buffers: buffers,
                                                 verdict: verdict, poison: poison)
                                }
                            }
                        }
                    }
                }
            }

            /// **The attention as a set of two engines**: the system graph on the GPU's heads beside the CPU's, at the heads the two solo lines
            /// balance, each side by the clock the split gives it (`HeadSplitAttention.run`: the GPU part's, the CPU part's wall). A side alone is its
            /// solo line scaled to its heads (both are proportional to them: the graph's and the heads' t₀ are nil). The same fit as the products'
            /// sets (`Fit.sharing`): a trial of `heads` rows of `2·S²·Dh` each.
            func attentionSet(_ measurements: inout Fit.Measurements, sequence s: Int, cpuSolo: (heads: Int, seconds: Double), flop: Double,
                              pages: (q: Pages, k: Pages, v: Pages, out: Pages),
                              buffers: (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, out: MTLBuffer),
                              verdict: (String) -> (pass: Bool, error: Double), poison: () -> Void) {
                let (h, d) = (Sizes.attentionHeads, Sizes.attentionHeadDim)
                let key = CapacityModel.key([.gpu, .cpu])
                guard let graph = measurements.solos.last(where: { $0.kernel == Kernel(.gpu, .graph).description && $0.shape.first == s })?.seconds else { return }
                // Rates over all the heads: the GPU's (the graph alone), the CPU's (its heads over its wall).
                let cpuRate = Double(cpuSolo.heads) / cpuSolo.seconds, gpuRate = Double(h) / graph
                let fraction = cpuRate / (cpuRate + gpuRate)
                guard let split = try? HeadSplitAttention(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d,
                                                          cpuFraction: fraction), split.cpuHeads > 0, split.gpuHeads > 0 else { return }
                nonisolated(unsafe) let (qp, kp, vp, op) = (pages.q.floats(), pages.k.floats(), pages.v.floats(), pages.out.floats())
                var sides: (gpu: Double, cpu: Double) = (0, 0)
                let run = {
                    sides = split.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out, q: qp, k: kp, v: vp, out: op)
                    return true
                }
                poison()
                guard run() else { return }
                let judged = verdict(key)
                let shape = [h, 2 * s * s * d, 1]
                guard judged.pass else {
                    measurements.trials.append(.init(set: key, shape: shape, pass: 2, sides: [], wall: 0, error: judged.error))
                    log?("  \(key) S=\(s): WRONG, error \(judged.error)")
                    return
                }
                var walls: [Double] = [], gpu: [Double] = [], cpu: [Double] = []
                _ = guarded("\(key) S=\(s)") { () -> Bool? in
                    walls = []; gpu = []; cpu = []
                    var previous = Double.infinity
                    for i in 0..<6 {
                        guard let wall = CapacityProbes.timed(run) else { return nil }
                        if i >= 1 { walls.append(wall); gpu.append(sides.gpu); cpu.append(sides.cpu) }
                        if i >= 2, wall >= previous * 0.9 { break }
                        previous = wall
                    }
                    return true
                }
                guard !walls.isEmpty else { return }
                let gpuAlone = graph * Double(split.gpuHeads) / Double(h), cpuAlone = cpuSolo.seconds * Double(split.cpuHeads) / Double(cpuSolo.heads)
                measurements.trials.append(.init(
                    set: key, shape: shape, pass: 2,
                    sides: [.init(kernel: Kernel(.gpu, .graph).description, rows: split.gpuHeads, solo: gpuAlone, within: Fit.median(gpu)),
                            .init(kernel: Kernel(.cpu, .heads).description, rows: split.cpuHeads, solo: cpuAlone, within: Fit.median(cpu))],
                    wall: Fit.median(walls), error: judged.error))
                log?(String(format: "  %@ S=%d  %d + %d heads  wall %.2f ms  gpu %.2f/%.2f cpu %.2f/%.2f ms", key, s, split.gpuHeads, split.cpuHeads,
                            Fit.median(walls) * 1e3, gpuAlone * 1e3, Fit.median(gpu) * 1e3, cpuAlone * 1e3, Fit.median(cpu) * 1e3))
            }

            // MARK: Validation

            /// **The model on products it never saw**: each kernel alone, then each set at the rows the model's own split gives,
            /// against a long measure (fifteen runs after five warm-ups; a set's, the trial's steady median).
            func validate(_ model: CapacityModel) -> [Check] {
                restoreB()
                var checks: [Check] = []
                func long(_ body: () -> Bool) -> Double? {
                    for _ in 0..<5 { guard body() else { return nil } }
                    var t: [Double] = []
                    for _ in 0..<15 { guard let s = CapacityProbes.timed(body) else { return nil }; t.append(s) }
                    return Fit.median(t)
                }
                let g = Kernel(.gpu, .fp32), x = Kernel(.amx, .fp32), a = Kernel(.ane, .fp16), b = Kernel(.gpu, .bf16)
                for (m, k, n) in validationShapes.map({ ($0.m, $0.k, $0.n) }) {
                    let flop = 2.0 * Double(m) * Double(k) * Double(n)
                    var runs: [(Kernel, () -> Bool)] = [(g, { self.gpu(m, k, n) }), (x, { self.amx(m, k, n) })]
                    if let e = neural.neuralEngine { runs.append((a, { self.ane(e, rows: m, k, n) })) }
                    if alone.tensorBF16 {
                        runs.append((b, { _ = self.alone.linear(a: self.aloneB.a, bf16: self.bf16, c: self.aloneB.c, m: m, k: k, n: n, weightIsTransposed: true); return true }))
                    }
                    for (kernel, body) in runs {
                        guard let line = model.lines[kernel], let t = long(body) else { continue }
                        checks.append(Check(what: kernel.description, shape: [m, k, n], rows: [kernel.description: m], predicted: model.alone(kernel, rows: m, k: k, n: n) ?? line.seconds(flop), measured: t))
                        log?(String(format: "  validate %@ %dx%dx%d  model %.2f ms  long %.2f ms  %+.1f %%", kernel.description, m, k, n,
                                    checks.last!.predicted * 1e3, t * 1e3, checks.last!.error * 100))
                    }
                }
                for (m, k, n) in validationShapes.map({ ($0.m, $0.k, $0.n) }) {
                    for set in [[g, x], [g, a], [g, x, a]] {
                        let key = CapacityModel.key(set.map(\.engine))
                        guard model.sharing[key] != nil, let t = trial(set, key: key, shape: (m, k, n), model: model, pass: 3),
                              !t.sides.isEmpty else { continue }
                        var rows: [Kernel: Int] = [:]
                        for s in t.sides { if let kernel = Kernel(s.kernel) { rows[kernel] = s.rows } }
                        guard let p = model.seconds(m: m, k: k, n: n, rows: rows) else { continue }
                        checks.append(Check(what: key, shape: [m, k, n], rows: Dictionary(uniqueKeysWithValues: t.sides.map { ($0.kernel, $0.rows) }),
                                            predicted: p, measured: t.wall))
                        log?(String(format: "  validate %@ %dx%dx%d  model %.2f ms  measured %.2f ms  %+.1f %%", key, m, k, n, p * 1e3, t.wall * 1e3, checks.last!.error * 100))
                    }
                }
                // The integers' route, on the same products: the ANE alone, then the GPU's int8 kernel beside it.
                let g8 = Kernel(.gpu, .int8), a8 = Kernel(.ane, .int8)
                if let e = neural.neuralEngine, let line = model.lines[a8] {
                    for (m, k, n) in validationShapes.map({ ($0.m, $0.k, $0.n) }) {
                        guard let plan = int8Plan(k, n) else { continue }
                        restoreB(); stageInt8(plan, k: k, n: n)
                        guard let t = long({ self.aneInt8(e, plan, rows: m, k, n) }) else { continue }
                        let flop = 2.0 * Double(m) * Double(k) * Double(n)
                        checks.append(Check(what: a8.description, shape: [m, k, n], rows: [a8.description: m], predicted: model.alone(a8, rows: m, k: k, n: n) ?? line.seconds(flop), measured: t))
                        log?(String(format: "  validate %@ %dx%dx%d  model %.2f ms  long %.2f ms  %+.1f %%", a8.description, m, k, n,
                                    line.seconds(flop) * 1e3, t * 1e3, checks.last!.error * 100))
                    }
                    let key = CapacityModel.key([g8, a8])
                    for (m, k, n) in validationShapes.map({ ($0.m, $0.k, $0.n) }) where model.sharing[key] != nil {
                        guard let t = trialInt8(key: key, shape: (m, k, n), model: model, pass: 3), !t.sides.isEmpty else { continue }
                        var rows: [Kernel: Int] = [:]
                        for s in t.sides { if let kernel = Kernel(s.kernel) { rows[kernel] = s.rows } }
                        guard let p = model.seconds(m: m, k: k, n: n, rows: rows) else { continue }
                        checks.append(Check(what: key, shape: [m, k, n], rows: Dictionary(uniqueKeysWithValues: t.sides.map { ($0.kernel, $0.rows) }),
                                            predicted: p, measured: t.wall))
                        log?(String(format: "  validate %@ %dx%dx%d  model %.2f ms  measured %.2f ms  %+.1f %%", key, m, k, n, p * 1e3, t.wall * 1e3, checks.last!.error * 100))
                    }
                    restoreB()
                }
                return checks
            }

            // MARK: Sharing of the integers

            /// **The GPU's int8 kernel beside the ANE's**, both reading the integers staged in the reserve — the pair a render of a Compact makes
            /// under `ane_int8` — as the fp32 sets are measured: chains, the lines' balance (pass 1), the balance of the κ it gave (pass 2, the
            /// same shape) and a long k (pass 3, on pass 2's κ), the fit reading the passes from the second. The long k is the point: the
            /// « down » projections of the three int8 families (k = 8 192 to 16 384) were left to the ANE alone for want of a pair measured
            /// there, which cost them a third of their seconds.
            func sharingInt8(_ measurements: inout Fit.Measurements) {
                restoreB()
                var model = Fit.model(from: measurements, allocationUnit: NeuralEngine.rowUnit)
                let g = Kernel(.gpu, .int8), a = Kernel(.ane, .int8)
                guard model.lines[g] != nil, model.lines[a] != nil, both.tensorInt8, neural.neuralEngine != nil, let short = sizes.chains.first else { return }
                let key = CapacityModel.key([g, a])
                for (pass, shape) in [(1, short), (2, short)] + (sizes.chains.count > 1 ? [(3, sizes.chains[1])] : []) {
                    if let t = trialInt8(key: key, shape: shape, model: model, pass: pass) {
                        measurements.trials.append(t)
                        log?(String(format: "  %@ pass %d  %dx%dx%d  wall %.2f ms  sides %@", key, pass, shape.m, shape.k, shape.n, t.wall * 1e3,
                                    t.sides.map { String(format: "%@ %d rows line %.1f within %.1f", $0.kernel, $0.rows, $0.solo * 1e3, ($0.within ?? 0) * 1e3) }.joined(separator: " | ")))
                    }
                    let trials = pass == 1 ? measurements.trials : measurements.trials.filter { $0.pass >= 2 }
                    if let fit = Fit.overlap(set: key, trials: trials, model: model) { model.sharing[key] = fit.sharing }
                }
            }

            private func trialInt8(key: String, shape: (m: Int, k: Int, n: Int), model: CapacityModel, pass: Int) -> Fit.Measurements.Trial? {
                let (m, k, n) = shape
                let g = Kernel(.gpu, .int8), a = Kernel(.ane, .int8)
                // The balance of a chain is asked of the model without the sets' domains: a chain at a k the pair was not measured at yet is how it gets it.
                var open = model
                for (name, var sharing) in model.sharing { sharing.kFrom = nil; sharing.kTo = nil; open.sharing[name] = sharing }
                guard let plan = int8Plan(k, n), let engine = neural.neuralEngine,
                      let split = open.split(m: m, k: k, n: n, kernels: [g, a]), let aneRows = split.rows[a], let gpuRows = split.rows[g],
                      aneRows > 0, gpuRows > 0 else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the model gives a side no row"); return nil }
                // The rows the engines can run for this key: the plan's own rule (`isRunnable`) and the process's (the program loads).
                guard ProductAssignment(path: .native, gpu: gpuRows, ane: aneRows).isRunnable(for: ProductKey(k: k, n: n, m: m, weight: .int8)),
                      engine.accepts(rows: aneRows, k: k, n: n, int8Group: plan.group) else {
                    log?("  \(key) \(m)x\(k)x\(n) pass \(pass): rows \(aneRows) not accepted by the engines"); return nil
                }
                prime(.int8, ane: true, m: m, k: k, n: n)
                stageInt8(plan, k: k, n: n)
                RouteBeacon.announce(g.description); RouteBeacon.announce(a.description)
                let buffers = bothB
                func once() -> (wall: Double, within: [String: Double])? {
                    var result: (gpu: Double, computed: Bool, gpuWall: Double, aneWall: Double)?
                    guard let wall = CapacityProbes.timed({
                        result = WeightProducts.int8Product(gemm: both, plan: plan, reserve: buffers.b, a: buffers.a, c: buffers.c,
                                                            m: m, k: k, n: n, ane: engine, aneRows: aneRows)
                        return result?.computed == true
                    }), let r = result else { return nil }
                    return (wall, [a.description: r.aneWall, g.description: r.gpuWall])
                }
                guard let verdict = judged(key, .int8, ane: true, m: m, k: k, n: n, { once() != nil }) else {
                    log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the set refused (fallback)"); return nil
                }
                guard verdict.pass else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): WRONG, error \(verdict.error)"); return .init(set: key, shape: [m, k, n], pass: pass, sides: [], wall: 0, error: verdict.error) }
                // The chain, as the fp32 sets' (`trial`): back to back, the ramp thrown away, the median of the window.
                var samples: [(wall: Double, within: [String: Double])] = []
                guard guarded("\(key) \(m)x\(k)x\(n) pass \(pass)", { () -> Bool? in
                    samples = []
                    let begin = CapacityProbes.now()
                    var calls = 0
                    while true {
                        let start = CapacityProbes.now()
                        guard let s = once() else { return nil }
                        calls += 1
                        if calls > chainWarmCalls, start - begin >= chainWarm { samples.append(s) }
                        let elapsed = CapacityProbes.now() - begin
                        if elapsed >= chainWarm + chainWindow, samples.count >= chainCalls { break }
                        if elapsed >= chainCap, !samples.isEmpty { break }
                    }
                    CapacityProbes.lastSpread = max(CapacityProbes.lastSpread, CapacityProbes.spread(samples.map(\.wall)))
                    return true
                }) != nil else { return nil }
                let used = CapacityProbes.quick(samples)
                let wall = Fit.median(used.map(\.wall))
                // `solo` is the line's seconds at the rows the side took (the reference of κ, as the fp32 sets').
                var sides: [Fit.Measurements.Side] = []
                for (kernel, rows) in [(g, gpuRows), (a, aneRows)] {
                    guard let solo = model.alone(kernel, rows: rows, k: k, n: n) else { return nil }
                    let within = used.compactMap { $0.within[kernel.description] }
                    sides.append(.init(kernel: kernel.description, rows: rows, solo: solo, within: within.isEmpty ? nil : Fit.median(within)))
                }
                return .init(set: key, shape: [m, k, n], pass: pass, sides: sides, wall: wall, error: verdict.error)
            }

            // MARK: Sharing

            /// One engine's solo time at `rows` rows of `k × n`, judged nowhere (the set is judged whole).
            private func soloSide(_ kernel: Kernel, rows: Int, k: Int, n: Int, engine: NeuralEngine?) -> Double? {
                guard rows > 0 else { return nil }
                switch kernel.engine {
                case .gpu: return CapacityProbes.steady { self.gpu(rows, k, n) }
                case .amx: return CapacityProbes.steady { self.amx(rows, k, n) }
                case .ane: return engine.flatMap { e in CapacityProbes.steady { self.ane(e, rows: rows, k, n) } }
                case .cpu: return nil
                }
            }

            /// The rows each kernel takes of `m`, from the model's balanced split (the ANE's in units); nil when it gives no ANE row.
            private func rows(_ kernels: [Kernel], model: CapacityModel, m: Int, k: Int, n: Int) -> [Kernel: Int]? {
                guard let s = model.split(m: m, k: k, n: n, kernels: kernels), kernels.allSatisfy({ (s.rows[$0] ?? 0) > 0 }) else { return nil }
                return s.rows
            }

            /// **The sets, as chains.** Every set of two or more engines that have a line, the smallest first (a set's κ is read through the
            /// sets that remain when its first side ends; none is ever taken at keep 1): the lines' balance (pass 1, κ = 1), then the balance of
            /// the κ it gave (pass 2, the same short shape) and a long k (pass 3, on pass 2's κ). The fit reads the passes from the second.
            func sharing(_ measurements: inout Fit.Measurements) {
                restoreB()
                var model = Fit.model(from: measurements, allocationUnit: NeuralEngine.rowUnit)
                let g = Kernel(.gpu, .fp32), x = Kernel(.amx, .fp32), a = Kernel(.ane, .fp16)
                let have = [g, x, a].filter { model.lines[$0] != nil }
                var sets: [[Kernel]] = []
                for size in 2...max(2, have.count) where have.count >= 2 {
                    for mask in 1..<(1 << have.count) where mask.nonzeroBitCount == size {
                        sets.append(have.indices.filter { mask & (1 << $0) != 0 }.map { have[$0] })
                    }
                }
                let chains = sizes.chains
                for set in sets {
                    let key = CapacityModel.key(set.map(\.engine))
                    for (pass, shape) in [(1, chains[0]), (2, chains[0])] + (chains.count > 1 ? [(3, chains[1])] : []) {
                        if let trial = trial(set, key: key, shape: shape, model: model, pass: pass) {
                            measurements.trials.append(trial)
                            log?(String(format: "  %@ pass %d  %dx%dx%d  wall %.2f ms  sides %@", key, pass, shape.m, shape.k, shape.n,
                                        trial.wall * 1e3, trial.sides.map { String(format: "%@ %d rows line %.1f within %.1f", $0.kernel, $0.rows, $0.solo * 1e3, ($0.within ?? 0) * 1e3) }.joined(separator: " | ")))
                        }
                        let trials = pass == 1 ? measurements.trials : measurements.trials.filter { $0.pass >= 2 }
                        if let fit = Fit.overlap(set: key, trials: trials, model: model) { model.sharing[key] = fit.sharing }
                    }
                }
            }

            private func trial(_ set: [Kernel], key: String, shape: (m: Int, k: Int, n: Int), model: CapacityModel, pass: Int,
                               aneRows forced: Int? = nil) -> Fit.Measurements.Trial? {
                let (m, k, n) = shape
                // The balance of a chain is asked of the model without the sets' domains: a chain at a k the sets were not measured at yet is how they get it.
                var open = model
                for (name, var sharing) in model.sharing { sharing.kFrom = nil; sharing.kTo = nil; open.sharing[name] = sharing }
                var planned = rows(set, model: open, m: m, k: k, n: n)
                if let forced, let p = planned { planned = p.merging([Kernel(.ane, .fp16): forced, Kernel(.gpu, .fp32): m - forced]) { $1 } }
                guard let plan = planned else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the model gives a side no row"); return nil }
                let g = Kernel(.gpu, .fp32), x = Kernel(.amx, .fp32), a = Kernel(.ane, .fp16)
                let hasANE = set.contains(a), hasAMX = set.contains(x)
                let engine = (hasAMX ? three : neural).neuralEngine
                let gemm = hasANE ? (hasAMX ? three : neural) : both
                let buffers = hasANE ? (hasAMX ? threeB : neuralB) : bothB
                let aneRows = plan[a] ?? 0
                let first = m - aneRows
                // The GPU's rows by the conductor's quantum, the AMX the rest of the GPU's side.
                // Without the GPU (the AMX beside the ANE) the side is the AMX's whole.
                let hasGPU = set.contains(g)
                var gpuRows = hasGPU ? plan[g] ?? first : 0, amxRows = hasGPU ? 0 : first
                var fraction = 1.0
                if hasAMX && hasGPU {
                    fraction = Double(plan[g]!) / Double(plan[g]! + plan[x]!)
                    // The GPU's rows on MPS's tile, the AMX the rest of the side (it keeps one tile at least).
                    let tile = GEMM.rowQuantum
                    gpuRows = max(tile, min(first - tile, Int((Double(first) * fraction / Double(tile)).rounded()) * tile))
                    amxRows = first - gpuRows
                    fraction = Double(gpuRows) / Double(first)
                }
                let key2 = "\(k)×\(n)"
                // The set's one call, and the sides' own clocks it leaves.
                // The plan answers this key with these exact rows until the next set is pinned.
                func configure() -> Bool {
                    guard !hasANE || engine != nil else { return false }
                    let key = ProductKey(k: k, n: n, m: m)
                    let assignment = ProductAssignment(gpu: gpuRows, amx: amxRows, ane: aneRows)
                    guard assignment.rows == m, assignment.isRunnable(for: key) else { return false }
                    gemm.plan.unpinAll()
                    gemm.plan.pin(key, assignment)
                    return true
                }
                guard configure() else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): rows \(aneRows)/\(gpuRows)/\(amxRows) not accepted by the engines"); return nil }
                for kernel in set { RouteBeacon.announce(kernel.description) }
                func once() -> (wall: Double, within: [String: Double])? {
                    let before = gemm.conductor?.perShape[key2] ?? Conductor.ShapeRecord()
                    let fallbacks = engine?.fallbacks ?? 0
                    guard let wall = CapacityProbes.timed({ _ = gemm.linear(a: buffers.a, b: buffers.b, c: buffers.c, m: m, k: k, n: n, weightIsTransposed: true); return true }) else { return nil }
                    if let e = engine, e.fallbacks != fallbacks { return nil }
                    var within: [String: Double] = [:]
                    if hasANE, let sides = gemm.lastSides {
                        within[a.description] = sides.ane
                        if hasGPU { within[g.description] = sides.gpu }
                    }
                    if let c = gemm.conductor, let after = c.perShape[key2], !hasANE || hasAMX {
                        let w = after.wall - before.wall
                        if hasGPU { within[g.description] = w - (after.idleGPU - before.idleGPU) }
                        within[x.description] = w - (after.idleAMX - before.idleAMX)
                    }
                    return (wall, within)
                }
                // Judged once: the set's output on the sampled rows (every quantum of the cut is visited), then warmed.
                guard let verdict = judged(key, .fp32, ane: hasANE, m: m, k: k, n: n, { once() != nil }) else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the set refused (fallback)"); return nil }
                guard verdict.pass else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): WRONG, error \(verdict.error)"); return .init(set: key, shape: [m, k, n], pass: pass, sides: [], wall: 0, error: verdict.error) }
                // The chain: back to back, the ramp thrown away, the median of the window (`chain`); the set's clock per call.
                var samples: [(wall: Double, within: [String: Double])] = []
                guard guarded("\(key) \(m)x\(k)x\(n) pass \(pass)", { () -> Bool? in
                    samples = []
                    let begin = CapacityProbes.now()
                    var calls = 0
                    while true {
                        let start = CapacityProbes.now()
                        guard let s = once() else { return nil }
                        calls += 1
                        if calls > chainWarmCalls, start - begin >= chainWarm { samples.append(s) }
                        let elapsed = CapacityProbes.now() - begin
                        if elapsed >= chainWarm + chainWindow, samples.count >= chainCalls { break }
                        if elapsed >= chainCap, !samples.isEmpty { break }
                    }
                    CapacityProbes.lastSpread = max(CapacityProbes.lastSpread, CapacityProbes.spread(samples.map(\.wall)))
                    return true
                }) != nil else { return nil }
                let used = CapacityProbes.quick(samples)
                let wall = Fit.median(used.map(\.wall))
                // `solo` is the line's seconds at the rows the side took: the reference of κ (the line's own bias at these rows is in κ).
                var sides: [Fit.Measurements.Side] = []
                for kernel in set {
                    let r = kernel == a ? aneRows : kernel == x ? amxRows : gpuRows
                    guard let solo = model.alone(kernel, rows: r, k: k, n: n) else { return nil }
                    let within = used.compactMap { $0.within[kernel.description] }
                    sides.append(.init(kernel: kernel.description, rows: r, solo: solo, within: within.isEmpty ? nil : Fit.median(within)))
                }
                return .init(set: key, shape: [m, k, n], pass: pass, sides: sides, wall: wall, error: verdict.error)
            }
        }
    }
}
