import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

/// **The capacity map's probes: what a Mac's engines do alone, and what they lose together — a few dozen runs, no weights,
/// the numbers `CapacityModel` is made of.** The machine layer; the arithmetic that turns its measurements into a model is
/// `CapacityFit` (pure, tested). The routes are the existing ones, forced one by one (`GEMM`, `Conductor`, `NeuralEngine`,
/// `TensorGEMM`, `WeightProducts`, the three attentions); each keeps the judge against the fp64 truth of the checks' rule
/// (a wrong route fills nothing).
///
///     run
///      ├─ plan the memory          operands, engines' surfaces, truths, attention, copy buffers — under B, before the first product
///      ├─ preparation              the weights' widening and copy rates                                              (measureWidening)
///      ├─ witnesses, start         one product on the GPU, the AMX, the ANE                                          → heat
///      ├─ bandwidth                a copy kernel on the GPU                                                          → memoryBandwidth
///      ├─ solos                    each kernel alone at ≈ 4, 30, 300 GFLOP                                           → lines
///      ├─ stairs                   the ANE alone at 8 row counts, one k × n                                          → aneRowUnit
///      ├─ attention                graph, flash, the CPU's heads, alone at two sequences                            → lines
///      ├─ sharing                  per set: each side alone at its rows, then the set (a balanced pass at three sizes) → keep, overhead
///      ├─ 8-bit and rounded kernels (they stage their weight in the reserve: after every fp32 route)
///      └─ witnesses, end
///
/// What sets it apart — what a probe written by analogy with the old map's route timings would get wrong:
///
///   - **a route is timed alone, not against another**: the map's ratios had no absolute second; a line needs seconds at
///     three sizes, and the sizes are three different products (512×2048², 1024×3840², 4096×6144²), because an engine's
///     rate depends on k and n as much as on W — the line is the machine's, not a shape's.
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
///   - **the CPU's heads cannot run without the GPU's**: the split keeps one head on the GPU (`partition`), so the CPU line is
///     measured with 23 heads of 24 on the CPU and one beside them.
extension Diagnostic {
    package enum CapacityProbes {
        package typealias Fit = CapacityFit
        package typealias Kernel = CapacityModel.Kernel
        package typealias Engine = CapacityModel.Engine

        /// The three sizes of the solos: ≈ 4, 30 and 309 GFLOP, k and n of a DiT's order.
        package static let soloShapes: [(m: Int, k: Int, n: Int)] = [(512, 2048, 2048), (1024, 3840, 3840), (4096, 6144, 6144)]
        /// The ANE's staircase: one k × n, row counts that are odd and even multiples of 256, and two halves of a unit.
        package static let stairShape = (k: 3840, n: 3840)
        package static let stairRows = [768, 896, 1024, 1792, 1920, 2048, 2304, 4096]
        /// The sets' sizes: 30, 60 and 121 GFLOP.
        package static let sharingShapes: [(m: Int, k: Int, n: Int)] = [(1024, 3840, 3840), (2048, 3840, 3840), (4096, 3840, 3840)]
        package static let attentionSequences = [1024, 4096]
        package static let attentionHeads = 24, attentionHeadDim = 128
        /// The copy kernel's buffers, each.
        package static let bandwidthBytes = 128 << 20
        package static let witnessShape = (m: 2048, k: 3840, n: 3840)
        /// Products the fit never saw, for `validate`: the truth test of a model (a probe is judged against a long measure of the same thing).
        package static let validationShapes: [(m: Int, k: Int, n: Int)] = [(2048, 3840, 10240), (1536, 4096, 4096), (3072, 3072, 3072), (1024, 2048, 8192)]

        package struct Memory: Codable, Sendable {
            package var budget: Int, base: Int, need: Int, peak: Int
        }

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
            package var memory: Memory?
        }

        package struct Result: Codable, Sendable {
            package var date: String
            package var seconds: Double
            package var machine: Machine
            package var measurements: Fit.Measurements
            package var model: CapacityModel
            package var details: Details
        }

        // MARK: - The run

        package static func run(cancellation: Cancellation? = nil, validate: Bool = false, log: ((String) -> Void)? = nil) throws(EngineError) -> Result {
            let started = Date()
            let lock = try RenderLock.acquire(RenderLock.machine)
            defer { withExtendedLifetime(lock) {} }
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw .internalFailure(component: "capacity probes", detail: "no Metal device")
            }
            OperatorFootprint.bringUp()
            _ = NeuralEngine.bringUp
            let gauge = Gauge()
            let plan = try Plan.choose(budget: MemoryBudget.machineCeiling(), base: gauge.base) { _ in need(validating: validate) }
            do {
                let context = try Context(device: device, queue: queue, cancellation: cancellation, gauge: gauge, validating: validate, log: log)
                var details = Details()
                var m = Fit.Measurements()
                try context.probes(&m, &details)
                var model = Fit.model(from: m, allocationUnit: NeuralEngine.rowUnit)
                let machine = Diagnostic.machine()
                if let peak = machine.gpuFP32Peak { model.crest[.gpu] = peak }
                for (k, l) in model.lines {
                    let pts = m.solos.filter { $0.kernel == k.description }.map { Fit.Point(flop: $0.flop, seconds: $0.seconds) }
                    let e = Fit.relativeError(l, pts)
                    details.lineError[k.description] = [e.median, e.worst]
                }
                details.stairs = Fit.stairs(m.stairs.map { ($0.rows, $0.seconds) })
                for set in Set(m.trials.map(\.set)) {
                    let last = m.trials.filter { $0.set == set }.map(\.pass).max().map { Set([$0]) }
                    details.sharing[set] = Fit.sharing(set: set, trials: m.trials, lines: model.lines, using: last)
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
                details.memory = Memory(budget: plan.budget, base: plan.base, need: plan.need.peak, peak: FootprintInterval.peak())
                gauge.finish()
                return Result(date: ISO8601DateFormatter().string(from: started), seconds: Date().timeIntervalSince(started),
                              machine: machine, measurements: m, model: model, details: details)
            } catch {
                throw EngineError(error)
            }
        }

        // MARK: - Memory

        /// The sum of everything that may be alive at once, from the shapes alone.
        static func need(validating: Bool = false) -> Need {
            let shapes = soloShapes + sharingShapes + (validating ? validationShapes : []) + stairRows.map { (m: $0, k: stairShape.k, n: stairShape.n) }
                + [(m: witnessShape.m, k: witnessShape.k, n: witnessShape.n)]
            let operands = Operands.sizes(products: shapes.map { ($0.m, $0.k, $0.n, RouteTable.WeightKind.q8) }).bytes
            let ps = shapes.map { (m: $0.m, k: $0.k, n: $0.n) }
            let gemm = MemoryNeed.gemm("diagnostic", products: ps, amx: true, int8: true, ane: false).reduce(0) { $0 + $1.bytes }
            let multiply = ps.map { Arena.capacity(for: Conductor.workspaceBytes(rows: $0.m, k: $0.k, n: $0.n, weightIsTransposed: true)) }.max() ?? 0
            let exact = ps.map { p in
                let r = Diagnostic.sampledRows
                return r * p.k * 12 + (1 << 25) + r * p.n * 8 * 5 + r * p.n * 4
            }.max() ?? 0
            let points = shapes.map { GridPoint(m: $0.m, k: $0.k, n: $0.n) }
            let ane = engineBytes(points) + programCount(points) * programBytes
            let (h, d) = (attentionHeads, attentionHeadDim)
            let attention = attentionSequences.map { s -> Int in
                let split = HeadSplitAttention.hostBytes(heads: h, sequence: s, headDim: d, cpuFraction: Double(h - 1) / Double(h))
                    + OperatorFootprint.sdpaQueries(rows: s, heads: 1, headDim: d)
                return 4 * Arena.capacity(for: s * h * d * 4) + max(split, OperatorFootprint.sdpaQueries(rows: s, heads: h, headDim: d))
                    + sampledQueries * h * d * 8 * 2
            }.max() ?? 0
            let truths = 160 << 20
            let routes = operands + 3 * gemm + 2 * ane + multiply + exact + truths + attention + 2 * bandwidthBytes
            let w = Diagnostic.blocks[0], count = w.dim * w.hidden
            let widening = [count * 4, count * 2, count * 2, count + 2 * w.hidden, count, 1 << 20].map { Arena.capacity(for: $0) }.reduce(0, +)
            return Need(steps: ["widening": widening + frameworks, "routes": routes + frameworks])
        }

        // MARK: - The context

        private enum Truth: Hashable { case fp32, rounded, int8, q8 }
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

        private final class Context {
            let bench: Bench
            let log: ((String) -> Void)?
            let operands: Operands
            let alone: GEMM, both: GEMM, neural: GEMM, three: GEMM
            let aloneB: Buffers, bothB: Buffers, neuralB: Buffers, threeB: Buffers, bf16: MTLBuffer
            var workspace: Pages?
            var truths: [TruthKey: [Double]] = [:]
            var floors: [[Int]: Double] = [:]
            var dirty = 0
            var clock = CapacityProbes.now()

            init(device: MTLDevice, queue: MTLCommandQueue, cancellation: Cancellation?, gauge: Gauge, validating: Bool, log: ((String) -> Void)?) throws {
                bench = Bench(device: device, queue: queue, cancellation: cancellation, gauge: gauge)
                self.log = log
                let all = (CapacityProbes.soloShapes + CapacityProbes.sharingShapes + (validating ? CapacityProbes.validationShapes : [])
                    + CapacityProbes.stairRows.map { (m: $0, k: CapacityProbes.stairShape.k, n: CapacityProbes.stairShape.n) }
                    + [(m: CapacityProbes.witnessShape.m, k: CapacityProbes.witnessShape.k, n: CapacityProbes.witnessShape.n)])
                operands = Operands(Operands.sizes(products: all.map { ($0.m, $0.k, $0.n, RouteTable.WeightKind.q8) }))
                alone = try GEMM(freezeCut: true, coexecution: false, tensorWeights: true, bf16Activations: true)
                both = try GEMM(freezeCut: true, coexecution: true, tensorWeights: false, int8Weights: true, q8Weights: true)
                neural = try GEMM(freezeCut: true, coexecution: false, tensorWeights: false, neuralEngine: true, neuralEngineShare: 0.5)
                three = try GEMM(freezeCut: true, coexecution: true, tensorWeights: false, neuralEngine: true, neuralEngineShare: 0.5)
                func wrap(_ g: GEMM, _ o: Operands) throws -> Buffers {
                    Buffers(a: try g.wrap(o.a.base, bytes: o.a.capacity, name: "probe.a"), b: try g.wrap(o.b.base, bytes: o.b.capacity, name: "probe.b"),
                            c: try g.wrap(o.c.base, bytes: o.c.capacity, name: "probe.c"))
                }
                aloneB = try wrap(alone, operands); bothB = try wrap(both, operands)
                neuralB = try wrap(neural, operands); threeB = try wrap(three, operands)
                bf16 = try alone.wrap(operands.bf16.base, bytes: operands.bf16.capacity, name: "probe.bf16")
                neural.neuralEngine?.shares = .none
                three.neuralEngine?.shares = .none
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
                case .q8:
                    result = Diagnostic.exactRows(rows, a: a, q: operands.q.base.assumingMemoryBound(to: Int8.self),
                                                  blockScales: operands.q8Scales.floats(), block: Diagnostic.q8Block, k: k, n: n)
                }
                truths[key] = result
                return result
            }

            /// bf16's own rounding of this product: the floor no fp16 path can pass under (invariant 3).
            private func aneThreshold(m: Int, k: Int, n: Int) -> Double {
                if let f = floors[[m, k, n]] { return max(Diagnostic.fp32Threshold, Diagnostic.floorMargin * f) }
                let rows = Diagnostic.sampleRows(m, count: Diagnostic.sampledRows)
                let witness = truth(.rounded, m: m, k: k, n: n).map { Double(Half.bf16.round(Float($0))) }
                let floor = Diagnostic.judge(witness.map(Float.init), truth(.fp32, m: m, k: k, n: n), columns: rows.count, rate: 0, threshold: 1).error
                floors[[m, k, n]] = floor
                return max(Diagnostic.fp32Threshold, Diagnostic.floorMargin * floor)
            }

            /// Runs `body` once with the sampled rows poisoned, and judges them: the route's error, `nil` when it refused.
            func judged(_ truthKind: Truth, ane: Bool, m: Int, k: Int, n: Int, _ body: () -> Bool) -> (pass: Bool, error: Double)? {
                let exact = truth(truthKind, m: m, k: k, n: n)
                let threshold = ane ? aneThreshold(m: m, k: k, n: n) : Diagnostic.fp32Threshold
                let rows = Diagnostic.sampleRows(m, count: Diagnostic.sampledRows)
                let out = operands.c.floats()
                Diagnostic.poison(out, rows: rows, n: n)
                guard body() else { return nil }
                let measure = Diagnostic.judge(Diagnostic.gather(out, rows: rows, n: n), exact, columns: rows.count, rate: 0, threshold: threshold)
                return (measure.pass, measure.error)
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

            // MARK: Probes

            func probes(_ m: inout Fit.Measurements, _ d: inout Details) throws {
                func stage(_ name: String, _ body: () throws -> Void) rethrows {
                    _ = lap()
                    try body()
                    m.budget[name] = lap()
                    log?(String(format: "%@ %.2f s", name, m.budget[name]!))
                }
                try stage("preparation") { m.preparation = try preparation() }
                stage("witnesses start") { m.witnesses = witnesses().map { .init(engine: $0.key, before: $0.value, after: 0) } }
                stage("bandwidth") { m.memoryBandwidth = bandwidth() }
                stage("solos fp32") { solos([.gpu_fp32, .amx_fp32, .ane_fp16, .gpu_bf16], &m) }
                stage("stairs") { stairs(&m) }
                try stage("attention") { try attention(&m) }
                stage("sharing") { sharing(&m) }
                stage("solos 8-bit and rounded") { solos([.gpu_bf16act, .gpu_int8, .gpu_q8], &m) }
                stage("witnesses end") {
                    restoreB()
                    let end = witnesses()
                    m.witnesses = m.witnesses.map { w in .init(engine: w.engine, before: w.before, after: end[w.engine] ?? w.before) }
                }
            }

            func preparation() throws -> [RouteTable.WeightKind: [RouteTable.WeightPath: Double]] {
                let widening = try Diagnostic.measureWidening(bench)
                func per(_ kind: String) -> Double {
                    guard let r = widening.first(where: { $0.kind == kind })?.cpu.rate, r > 0 else { return 0 }
                    return 1e-9 / r
                }
                return [.bf16: [.widened: per("bf16"), .bf16: per("bf16 copy")],
                        .int8: [.widened: per("int8 convrot"), .native: per("int8 copy")],
                        .q8: [.widened: per("q8_0"), .native: per("q8_0 copy")]]
            }

            /// The same product on each engine alone, median of five (after two warm-ups).
            func witnesses() -> [String: Double] {
                let (m, k, n) = witnessShape
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
                if let e = neural.neuralEngine, let t = median({ self.ane(e, rows: m, k, n) }) { result["ane"] = t }
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

            enum Probe: String { case gpu_fp32, amx_fp32, ane_fp16, gpu_bf16, gpu_bf16act, gpu_int8, gpu_q8
                var kernel: Kernel {
                    switch self {
                    case .gpu_fp32: return Kernel(.gpu, .fp32)
                    case .amx_fp32: return Kernel(.amx, .fp32)
                    case .ane_fp16: return Kernel(.ane, .fp16)
                    case .gpu_bf16: return Kernel(.gpu, .bf16)
                    case .gpu_bf16act: return Kernel(.gpu, .bf16act)
                    case .gpu_int8: return Kernel(.gpu, .int8)
                    case .gpu_q8: return Kernel(.gpu, .q8)
                    }
                }
            }

            func solos(_ probes: [Probe], _ m: inout Fit.Measurements) {
                for probe in probes {
                    for (rows, k, n) in soloShapes.map({ ($0.m, $0.k, $0.n) }) {
                        guard let s = solo(probe, m: rows, k: k, n: n) else { continue }
                        if s.error != nil, s.seconds == 0 { m.wrong.append(probe.kernel.description); continue }
                        m.solos.append(s)
                        log?(String(format: "  %@ %dx%dx%d  %.2f ms  %.2f TFLOP/s  err %.2e", probe.kernel.description, rows, k, n,
                                    s.seconds * 1e3, s.flop / s.seconds * 1e-12, s.error ?? 0))
                    }
                }
            }

            /// One kernel on one product: judged on its first run, then timed. A kernel that judged wrong returns seconds 0.
            func solo(_ probe: Probe, m: Int, k: Int, n: Int) -> Fit.Measurements.Solo? {
                let flop = 2.0 * Double(m) * Double(k) * Double(n)
                let name = probe.kernel.description
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
                _ = truth(truthKind, m: m, k: k, n: n)
                staged?()
                guard let verdict = judged(truthKind, ane: isANE, m: m, k: k, n: n, run) else { return nil }
                guard verdict.pass else { return .init(kernel: name, shape: [m, k, n], flop: flop, seconds: 0, error: verdict.error) }
                guard let t = CapacityProbes.steady(run) else { return nil }
                return .init(kernel: name, shape: [m, k, n], flop: flop, seconds: t, error: verdict.error)
            }

            // MARK: The ANE's staircase

            func stairs(_ m: inout Fit.Measurements) {
                guard let engine = neural.neuralEngine else { return }
                restoreB()
                m.stairShape = [stairShape.k, stairShape.n]
                for rows in stairRows {
                    let (k, n) = stairShape
                    let run = { self.ane(engine, rows: rows, k, n) }
                    guard let v = judged(.fp32, ane: true, m: rows, k: k, n: n, run) else { continue }
                    guard v.pass else { m.wrong.append("ane/fp16@\(rows)"); continue }
                    guard let t = CapacityProbes.steady(run) else { continue }
                    m.stairs.append(.init(rows: rows, seconds: t))
                    log?(String(format: "  stair %d rows  %.2f ms", rows, t * 1e3))
                }
            }

            // MARK: Attention

            func attention(_ measurements: inout Fit.Measurements) throws {
                let (h, d) = (attentionHeads, attentionHeadDim)
                let floats = attentionSequences.max()! * h * d
                let pages = (q: Pages(bytes: floats * 4), k: Pages(bytes: floats * 4), v: Pages(bytes: floats * 4), out: Pages(bytes: floats * 4))
                Diagnostic.fill(pages.q.floats(), count: floats, scale: 3, seed: 11)
                Diagnostic.fill(pages.k.floats(), count: floats, scale: 3, seed: 12)
                Diagnostic.fill(pages.v.floats(), count: floats, scale: 1, seed: 13)
                let buffers = (q: try bench.buffer(pages.q), k: try bench.buffer(pages.k), v: try bench.buffer(pages.v), out: try bench.buffer(pages.out))
                let out = pages.out.floats()
                for s in attentionSequences {
                    autoreleasepool {
                        let rows = Diagnostic.sampleRows(s, count: Diagnostic.sampledQueries)
                        let exact = Diagnostic.exactAttention(q: pages.q.floats(), k: pages.k.floats(), v: pages.v.floats(), sequence: s, heads: h, headDim: d, rows: rows)
                        let flop = 4.0 * Double(s) * Double(s) * Double(h) * Double(d)
                        func verdict() -> Diagnostic.Measure {
                            let width = h * d, vectors = rows.count * h
                            var got = [Float](repeating: 0, count: vectors * d)
                            for (i, row) in rows.enumerated() {
                                for head in 0..<h { for x in 0..<d { got[x * vectors + i * h + head] = out[row * width + head * d + x] } }
                            }
                            return Diagnostic.judge(got, exact, columns: vectors, rate: 0, threshold: Diagnostic.fp32Threshold)
                        }
                        func poison() { for row in rows { (out + row * h * d).update(repeating: .nan, count: h * d) } }
                        func record(_ kernel: Kernel, flop: Double, run: () -> Bool, shape: [Int]) {
                            poison()
                            guard run() else { return }
                            let v = verdict()
                            guard v.pass else { measurements.wrong.append(kernel.description); return }
                            guard let t = CapacityProbes.steady(run) else { return }
                            measurements.solos.append(.init(kernel: kernel.description, shape: shape, flop: flop, seconds: t, error: v.error))
                            log?(String(format: "  %@ S=%d  %.2f ms  %.2f TFLOP/s", kernel.description, s, t * 1e3, flop / t * 1e-12))
                        }
                        let graph = Attention(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d)
                        record(Kernel(.gpu, .graph), flop: flop, run: { _ = graph.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out); return true }, shape: [s, h, d])
                        if let flash = try? FlashMatrix(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d) {
                            record(Kernel(.gpu, .flash), flop: flop, run: { _ = flash.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out); return true }, shape: [s, h, d])
                        }
                        // The CPU's heads: all but one of them, the figure being the CPU part's own wall.
                        if let split = try? HeadSplitAttention(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d,
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
                                let v = verdict()
                                if !v.pass { measurements.wrong.append("cpu/heads") } else {
                                    var times: [Double] = []
                                    var previous = Double.infinity
                                    for i in 0..<5 {
                                        _ = run()
                                        if i >= 1 { times.append(cpuSeconds) }
                                        if i >= 1, cpuSeconds >= previous * 0.9, times.count >= 2 { break }
                                        previous = cpuSeconds
                                    }
                                    let cpuFlop = 4.0 * Double(s) * Double(s) * Double(d) * Double(split.cpuHeads)
                                    let t = Fit.median(Array(times.suffix(3)))
                                    measurements.solos.append(.init(kernel: Kernel(.cpu, .heads).description, shape: [s, h, d, split.cpuHeads],
                                                                    flop: cpuFlop, seconds: t, error: v.error))
                                    log?(String(format: "  cpu/heads S=%d (%d heads)  %.2f ms  %.2f TFLOP/s", s, split.cpuHeads, t * 1e3, cpuFlop / t * 1e-12))
                                }
                            }
                        }
                    }
                }
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
                        checks.append(Check(what: kernel.description, shape: [m, k, n], rows: [kernel.description: m], predicted: line.seconds(flop), measured: t))
                        log?(String(format: "  validate %@ %dx%dx%d  model %.2f ms  long %.2f ms  %+.1f %%", kernel.description, m, k, n,
                                    line.seconds(flop) * 1e3, t * 1e3, checks.last!.error * 100))
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
                return checks
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

            func sharing(_ measurements: inout Fit.Measurements) {
                restoreB()
                var model = Fit.model(from: measurements, allocationUnit: NeuralEngine.rowUnit)
                let g = Kernel(.gpu, .fp32), x = Kernel(.amx, .fp32), a = Kernel(.ane, .fp16)
                var sets: [[Kernel]] = []
                if model.lines[g] != nil, model.lines[x] != nil { sets.append([g, x]) }
                if model.lines[g] != nil, model.lines[a] != nil { sets.append([g, a]) }
                if model.lines[g] != nil, model.lines[x] != nil, model.lines[a] != nil { sets.append([g, x, a]) }
                for set in sets {
                    let key = CapacityModel.key(set.map(\.engine))
                    // Pass 1 at the middle size on the solo lines; pass 2 at the three sizes on the keep it measured.
                    var first = true
                    for pass in [1, 2] {
                        let shapes = pass == 1 ? [sharingShapes[1]] : sharingShapes
                        for shape in shapes {
                            if let trial = trial(set, key: key, shape: shape, model: model, pass: pass) {
                                measurements.trials.append(trial)
                                log?(String(format: "  %@ pass %d  %dx%dx%d  wall %.2f ms  sides %@", key, pass, shape.m, shape.k, shape.n,
                                            trial.wall * 1e3, trial.sides.map { String(format: "%@ %d rows solo %.1f within %.1f", $0.kernel, $0.rows, $0.solo * 1e3, ($0.within ?? 0) * 1e3) }.joined(separator: " | ")))
                            }
                        }
                        if first, let fit = Fit.sharing(set: key, trials: measurements.trials, lines: model.lines, using: [1]) {
                            model.sharing[key] = fit.sharing
                            first = false
                        }
                    }
                    // The balance's neighbourhood: the pair at 4 096 rows, the ANE's rows around the balance (pass 0, never fitted).
                    if set == [g, a] {
                        let shape = sharingShapes[2]
                        let at = rows(set, model: model, m: shape.m, k: shape.k, n: shape.n)?[a] ?? shape.m / 2
                        let centre = at / 256 * 256
                        for r in stride(from: centre - 512, through: centre + 768, by: 256) where r >= 256 && r < shape.m {
                            if let t = trial(set, key: key, shape: shape, model: model, pass: 0, aneRows: r) {
                                measurements.trials.append(t)
                                log?(String(format: "  scan %@ %d ANE rows  wall %.2f ms  gpu %.2f ane %.2f", key, r, t.wall * 1e3,
                                            (t.sides.first { $0.kernel == g.description }?.within ?? 0) * 1e3,
                                            (t.sides.first { $0.kernel == a.description }?.within ?? 0) * 1e3))
                            }
                        }
                    }
                }
            }

            private func trial(_ set: [Kernel], key: String, shape: (m: Int, k: Int, n: Int), model: CapacityModel, pass: Int,
                               aneRows forced: Int? = nil) -> Fit.Measurements.Trial? {
                let (m, k, n) = shape
                var planned = rows(set, model: model, m: m, k: k, n: n)
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
                var gpuRows = plan[g] ?? first, amxRows = 0
                var fraction = 1.0
                if hasAMX {
                    fraction = Double(plan[g]!) / Double(plan[g]! + plan[x]!)
                    gpuRows = Conductor.cut(first, fraction)
                    amxRows = first - gpuRows
                    fraction = Double(gpuRows) / Double(first)
                }
                let key2 = "\(k)×\(n)"
                // The set's one call, and the sides' own clocks it leaves.
                func configure() -> Bool {
                    if hasANE {
                        guard let e = engine else { return false }
                        e.shares = .none
                        e.gpuShare = 1 - Double(aneRows) / Double(m)
                        guard e.rows(m, k: k, n: n) == aneRows else { return false }
                        if hasAMX {
                            e.besideAMX = Conductor.Shares([.init(k: k, n: n, m: m, ratio: 1)])
                            gemm.conductor?.shares = Conductor.Shares([.init(k: k, n: n, m: m, ratio: (1 - fraction) / fraction)])
                        } else {
                            e.besideAMX = .none
                        }
                    } else if let c = gemm.conductor {
                        c.shares = .none
                        c.initialFraction = Double(gpuRows) / Double(m)
                        guard Conductor.cut(m, c.initialFraction) == gpuRows else { return false }
                    }
                    return true
                }
                guard configure() else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): rows \(aneRows)/\(gpuRows)/\(amxRows) not accepted by the engines"); return nil }
                func once() -> (wall: Double, within: [String: Double])? {
                    let before = gemm.conductor?.perShape[key2] ?? Conductor.ShapeRecord()
                    let fallbacks = engine?.fallbacks ?? 0
                    guard let wall = CapacityProbes.timed({ _ = gemm.linear(a: buffers.a, b: buffers.b, c: buffers.c, m: m, k: k, n: n, weightIsTransposed: true); return true }) else { return nil }
                    if let e = engine, e.fallbacks != fallbacks { return nil }
                    var within: [String: Double] = [:]
                    if hasANE, let sides = gemm.lastSides {
                        within[a.description] = sides.ane
                        within[g.description] = sides.gpu
                    }
                    if let c = gemm.conductor, let after = c.perShape[key2], !hasANE || hasAMX {
                        let w = after.wall - before.wall
                        within[g.description] = w - (after.idleGPU - before.idleGPU)
                        within[x.description] = w - (after.idleAMX - before.idleAMX)
                    }
                    return (wall, within)
                }
                // Judged once: the set's output on the sampled rows (every quantum of the cut is visited), then warmed.
                guard let verdict = judged(.fp32, ane: hasANE, m: m, k: k, n: n, { once() != nil }) else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): the set refused (fallback)"); return nil }
                guard verdict.pass else { log?("  \(key) \(m)x\(k)x\(n) pass \(pass): WRONG, error \(verdict.error)"); return .init(set: key, shape: [m, k, n], pass: pass, sides: [], wall: 0, error: verdict.error) }
                var samples: [(wall: Double, within: [String: Double])] = []
                var previous = Double.infinity
                for i in 0..<6 {
                    guard let s = once() else { return nil }
                    if i >= 1, s.wall >= previous * 0.9 {
                        samples.append(s)
                        for _ in 0..<(s.wall < 0.15 ? 2 : 1) { if let t = once() { samples.append(t) } }
                        break
                    }
                    previous = s.wall
                }
                guard !samples.isEmpty else { return nil }
                let wall = Fit.median(samples.map(\.wall))
                var sides: [Fit.Measurements.Side] = []
                for kernel in set {
                    let r = kernel == a ? aneRows : kernel == x ? amxRows : gpuRows
                    guard let solo = soloSide(kernel, rows: r, k: k, n: n, engine: engine) else { return nil }
                    let within = samples.compactMap { $0.within[kernel.description] }
                    sides.append(.init(kernel: kernel.description, rows: r, solo: solo, within: within.isEmpty ? nil : Fit.median(within)))
                }
                return .init(set: key, shape: [m, k, n], pass: pass, sides: sides, wall: wall, error: verdict.error)
            }
        }
    }
}
