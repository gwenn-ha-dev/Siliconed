import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

// The probes' solos, the ANE's staircase and the attentions — each engine, each kernel alone (`CapacityProbes.run`).

extension Diagnostic.CapacityProbes.Context {
    // MARK: Solos

    enum Probe: String, CaseIterable { case gpu_fp32, amx_fp32, ane_fp16, ane_int8, gpu_bf16, gpu_bf16act, gpu_int8, gpu_q8, gpu_int8act, gpu_q8act, gpu_q4kact
        var kernel: Kernel {
            switch self {
            case .gpu_q4kact: return Kernel(.gpu, .q4kact)
            case .gpu_int8act: return Kernel(.gpu, .int8act)
            case .gpu_q8act: return Kernel(.gpu, .q8act)
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
            again(probe, &m, chained: chained)
        }
    }

    /// **The kernel's points far above the line of the others, measured once more** (`Fit.suspects`), the faster kept: a figure the Mac
    /// disturbed comes back to the line, a shape the kernel is slow at stays. One solo per suspect (a few tenths of a second at most), and
    /// once only: `solos` runs a kernel at more than one set of shapes, and a point measured again stays a suspect of the next pass when
    /// it is the kernel's own shape (`ane/fp16 2048x8192x1536`, 1 % apart, measured three times in two maps).
    func again(_ probe: Probe, _ m: inout Fit.Measurements, chained: Bool) {
        let name = probe.kernel.description
        let mine = m.solos.indices.filter { m.solos[$0].kernel == name && m.solos[$0].seconds > 0 }
        for i in Fit.suspects(mine.map { Fit.Point(flop: m.solos[$0].flop, seconds: m.solos[$0].seconds) }) {
            let j = mine[i], shape = m.solos[j].shape
            guard shape.count == 3, measuredAgain.insert("\(name) \(shape)").inserted,
                  let s = solo(probe, m: shape[0], k: shape[1], n: shape[2], chained: chained), s.seconds > 0 else { continue }
            log?(String(format: "  %@ %dx%dx%d  again: %.2f ms (was %.2f)", name, shape[0], shape[1], shape[2], s.seconds * 1e3, m.solos[j].seconds * 1e3))
            if s.seconds < m.solos[j].seconds { m.solos[j].seconds = s.seconds }
        }
    }

    /// **The form of a half-precision attention kernel** (`AttentionTile`): the three stages and every fused candidate, judged at the
    /// kernel's bar, then timed in alternated rounds of one run each (`AttentionTile.rounds`) at the tier's longest sequence, its fast side
    /// (the least of the rounds), and
    /// `MatmulTile.choose`'s rule, the three stages first — no gain within the margin, no change of bits. The chosen form runs the kernel's
    /// line; it goes in the report with every candidate's times (`tileTrials`, `"gpu/tiles16 16x32/4"`, `"… staged"`).
    func chooseForm(_ kernel: Kernel, _ m: inout Fit.Measurements, flop: Double, shape: [Int], threshold: Double,
                    build: (AttentionTile?) -> TiledAttention?, run: @escaping (TiledAttention) -> Void, poison: () -> Void,
                    verdict: (String, Double) -> (pass: Bool, error: Double)) -> AttentionTile? {
        let candidates: [AttentionTile?] = [nil] + AttentionTile.candidates.map { $0 }
        var built: [AttentionTile?: TiledAttention] = [:]
        var refused: Set<AttentionTile?> = []
        for c in candidates {
            guard let t = build(c) else { refused.insert(c); continue }
            poison()
            run(t)
            let name = "\(kernel) \(c?.description ?? "staged")"
            let v = verdict(name, threshold)
            guard v.pass else { refused.insert(c); m.tileTrials = (m.tileTrials ?? []) + [.init(kernel: name, shape: shape, flop: flop, seconds: 0, error: v.error)]; continue }
            built[c] = t
        }
        var fastest: [AttentionTile?: Double] = [:], every: [AttentionTile?: [Double]] = [:]
        for _ in 0..<AttentionTile.rounds {
            for c in candidates {
                guard let t = built[c], let seconds = guarded("\(kernel) form \(c?.description ?? "staged")", {
                    CapacityProbes.timed { run(t); return true }
                }) else { continue }
                m.tileTrials = (m.tileTrials ?? []) + [.init(kernel: "\(kernel) \(c?.description ?? "staged")", shape: shape, flop: flop, seconds: seconds, error: nil)]
                fastest[c] = min(fastest[c] ?? .infinity, seconds)
                every[c, default: []].append(seconds)
            }
        }
        let chosen = MatmulTile.choose(fastest, among: candidates, standard: nil, rounds: every)
        log?("  form \(kernel): " + candidates.compactMap { c in fastest[c].map { String(format: "%@ %.2f ms", c?.description ?? "staged", $0 * 1e3) } }
                .joined(separator: " · ") + " → \(chosen?.description ?? "staged")")
        if let chosen { m.attentionTiles = (m.attentionTiles ?? [:]).merging([kernel.description: chosen.description]) { $1 } }
        return chosen
    }

    /// **The tiles of a `matmul2d` kernel** (`MatmulTile`): every candidate, judged, then timed in three alternated rounds — on the largest solo
    /// product for the wide tile (a render's height; the middle one, 1.5 ms on an M5 Pro, ran 30 % apart between two maps), over the thin shapes
    /// (`Sizes.thin`) for the thin one —, each shape on its fast side (the least of the rounds, as the map reads every disturbed figure), the
    /// shapes summed, and `MatmulTile.choose`. The chosen tiles run the kernel's lines and its products, and go in the report with every
    /// candidate's solos.
    func chooseTiles(_ probe: Probe, _ m: inout Fit.Measurements) {
        let kernel = probe.kernel
        guard MatmulTile.kernels.contains(kernel), !skipped.contains(kernel), !sizes.solo.isEmpty,
              let tensor = (probe == .gpu_int8 ? both : alone).tensor else { return }
        let candidates = MatmulTile.candidates(for: kernel), standard = MatmulTile.standard(for: kernel)
        // Each candidate's seconds per shape, round by round (`every[tile][shape]`), the candidates alternated within a round.
        func run(_ tiles: [MatmulTile], _ shapes: [Sizes.Shape], thin: Bool, rounds: Int,
                 into every: inout [MatmulTile: [Int: [Double]]], refused: inout Set<MatmulTile>) {
            for _ in 0..<rounds {
                for tile in tiles where !refused.contains(tile) {
                    guard tensor.tiles.use(tile, for: kernel, thin: thin) else { refused.insert(tile); continue }
                    for (i, shape) in shapes.enumerated() {
                        guard let s = solo(probe, m: shape.m, k: shape.k, n: shape.n) else { continue }
                        m.tileTrials = (m.tileTrials ?? []) + [.init(kernel: "\(kernel) \(tile)", shape: s.shape, flop: s.flop, seconds: s.seconds, error: s.error)]
                        guard s.seconds > 0 else { refused.insert(tile); break }
                        every[tile, default: [:]][i, default: []].append(s.seconds)
                    }
                }
            }
        }
        // The fast sides summed over the shapes every candidate left kept, and each round's sum (a round counts where every shape has it).
        func summed(_ every: [MatmulTile: [Int: [Double]]], refused: Set<MatmulTile>, shapes: Int) -> (seconds: [MatmulTile: Double], rounds: [MatmulTile: [Double]]) {
            let kept = every.filter { !refused.contains($0.key) }
            let common = kept.values.map { Set($0.keys) }.reduce(Set(0..<shapes)) { $0.intersection($1) }
            var seconds: [MatmulTile: Double] = [:], rounds: [MatmulTile: [Double]] = [:]
            for (tile, times) in kept where !common.isEmpty {
                seconds[tile] = common.reduce(0) { $0 + times[$1]!.min()! }
                let n = common.map { times[$0]!.count }.min()!
                rounds[tile] = (0..<n).map { r in common.reduce(0) { $0 + times[$1]![r] } }
            }
            return (seconds.filter { $0.value > 0 }, rounds)
        }
        func decide(_ shapes: [Sizes.Shape], thin: Bool) -> MatmulTile {
            let margin = thin ? MatmulTile.thinMargin : MatmulTile.margin
            var every: [MatmulTile: [Int: [Double]]] = [:], refused: Set<MatmulTile> = []
            run(candidates, shapes, thin: thin, rounds: MatmulTile.rounds, into: &every, refused: &refused)
            var measured = summed(every, refused: refused, shapes: shapes.count)
            // A candidate whose rounds fall on both sides of the margin is measured again beside the fastest, alternated: the answer is
            // the rounds', not one round's. All the doubtful ones in the same rounds, so that each round of the fastest pairs with each.
            let doubt = MatmulTile.ambiguous(measured.seconds, among: candidates, standard: standard, margin: margin, rounds: measured.rounds)
            if let fastest = measured.seconds.min(by: { $0.value < $1.value })?.key, !doubt.isEmpty {
                run(doubt + [fastest], shapes, thin: thin, rounds: MatmulTile.extraRounds, into: &every, refused: &refused)
                measured = summed(every, refused: refused, shapes: shapes.count)
                log?("  tiles \(kernel)\(thin ? " thin" : ""): measured again " + doubt.map(\.description).joined(separator: ", ") + " beside \(fastest)")
            }
            let chosen = MatmulTile.choose(measured.seconds, among: candidates, standard: standard, margin: margin, rounds: measured.rounds)
            let gaps = MatmulTile.gaps(measured.seconds, rounds: measured.rounds) ?? [:]
            log?("  tiles \(kernel)\(thin ? " thin" : ""): " + candidates.compactMap { t in measured.seconds[t].map { String(format: "%@ %.3f ms (%+.1f %%)", t.description, $0 * 1e3, 100 * (gaps[t] ?? 0)) } }
                    .joined(separator: " · ") + " → \(chosen)")
            tensor.tiles.use(chosen, for: kernel, thin: thin)
            return chosen
        }
        let wide = decide([sizes.solo[sizes.solo.count - 1]], thin: false)
        m.tiles = (m.tiles ?? [:]).merging([kernel.description: wide.description]) { $1 }
        let thinShapes = sizes.thin.filter { CapacityModel.isThin(rows: $0.m, k: $0.k, n: $0.n) }
        if !thinShapes.isEmpty {
            let thin = decide(thinShapes, thin: true)
            m.thinTiles = (m.thinTiles ?? [:]).merging([kernel.description: thin.description]) { $1 }
        }
    }

    /// **The products with a short side** (`Sizes.thin`), each kernel the solver can give such a product alone: the fp32 routes of the GPU, the
    /// AMX and the ANE, the GPU's bf16 weights, then the kernels that stage their weight after the fp32 routes (the rounded activations, int8,
    /// q8). Each is a steady figure (a few milliseconds a run; the ANE's cost is its programs' compile, one per shape); the ANE takes the shapes
    /// of 256 rows and more (its unit). Kept apart from the solos: the big products' lines never see them.
    func thinSolos(_ m: inout Fit.Measurements) {
        guard !sizes.thin.isEmpty else { return }
        var kept: [Fit.Measurements.Solo] = []
        for probe in [Probe.gpu_fp32, .amx_fp32, .ane_fp16, .gpu_bf16, .gpu_bf16act, .gpu_int8, .gpu_q8, .gpu_int8act, .gpu_q8act] {
            for (rows, k, n) in sizes.thin.map({ ($0.m, $0.k, $0.n) }) {
                if probe == .ane_fp16, rows < NeuralEngine.rowUnit { continue }
                guard let s = solo(probe, m: rows, k: k, n: n) else { continue }
                if s.error != nil, s.seconds == 0 {
                    m.wrong.append(probe.kernel.description)
                    log?(String(format: "  %@ %dx%dx%d  WRONG, error %.3e", probe.kernel.description, rows, k, n, s.error ?? 0))
                    continue
                }
                kept.append(s)
                log?(String(format: "  thin %@ %dx%dx%d  %.2f ms  %.2f TFLOP/s", probe.kernel.description, rows, k, n, s.seconds * 1e3, s.flop / s.seconds * 1e-12))
            }
        }
        m.thin = kept
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
        // Judged at the bar of a half-precision route (1.25 × bf16's own error on the product, invariant 3) rather than the fp32 one:
        // the ANE's, and the 8-bit products on a rounded activation — whose operand is rounded after the rotation, where the fp64
        // truth of the rounded operand would round the other side of a tie now and then.
        var half = false
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
        case .gpu_int8, .gpu_q8, .gpu_int8act, .gpu_q8act:
            let isInt8 = probe == .gpu_int8 || probe == .gpu_int8act
            // The rounded ones run on `alone`, which compiled the rounding kernels; the fp32 ones on `both`, as a render beside the AMX.
            let rounded = probe == .gpu_int8act || probe == .gpu_q8act
            let gemm = rounded ? alone : both, buffers = rounded ? aloneB : bothB
            guard isInt8 ? gemm.tensorInt8 : gemm.tensorQ8, !rounded || (gemm.bf16Activations && (isInt8 || gemm.tensor?.roundsBlocks == true)),
                  (k * n) % 2 == 0, isInt8 ? k % Diagnostic.int8Group == 0 : k % Diagnostic.q8Block == 0,
                  let plan = WeightProducts.int8Plan(k: k, n: n, capacity: k * n, perColumn: false, group: isInt8 ? Diagnostic.int8Group : nil,
                                                     amxColumns: 0, block: isInt8 ? nil : Diagnostic.q8Block, rounded: rounded) else { return nil }
            truthKind = isInt8 ? .int8 : .q8
            half = rounded
            staged = { [self] in
                Widen.copyBfloat16(source: operands.q.base, destination: operands.b.base, count: k * n / 2)
                if isInt8 {
                    (operands.b.base + plan.scalesOffset).storeBytes(of: Diagnostic.int8Scale, as: Float.self)
                } else {
                    (operands.b.base + plan.scalesOffset).copyMemory(from: operands.q8Scales.base, byteCount: (k + Diagnostic.q8Block - 1) / Diagnostic.q8Block * n * 4)
                }
                dirty = max(dirty, k * n)
            }
            run = { WeightProducts.int8Product(gemm: gemm, plan: plan, reserve: buffers.b, a: buffers.a, c: buffers.c, m: m, k: k, n: n).status == 0 }
        case .gpu_q4kact:
            // Q4_K's blocks as a map stores them, in the reserve's head; the route's own function on them (`TensorGEMM.linear(a:kquant:…)`).
            guard alone.bf16Activations, alone.tensor?.readsKQuants == true,
                  let layout = TensorGEMM.kquantLayout(k: k, n: n, capacity: k * n * 4, five: false) else { return nil }
            truthKind = .q4k
            half = true
            staged = { [self] in
                operands.b.base.copyMemory(from: operands.kquant.base, byteCount: n * (k / 256) * 144)
                dirty = max(dirty, k * n)
            }
            run = { _ = self.alone.tensor!.linear(a: self.aloneB.a, kquant: self.aloneB.b, layout: layout, c: self.aloneB.c, m: m, k: k, n: n); return true }
        }
        // The truth first (it restores B for the fp32 ones), then the staged weight.
        prime(truthKind, ane: isANE || half, m: m, k: k, n: n)
        staged?()
        guard let verdict = judged(name, truthKind, ane: isANE || half, m: m, k: k, n: n, run) else { return nil }
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
        func solo(_ kernel: Kernel, at s: Int) -> Fit.Measurements.Solo? {
            measurements.solos.last { $0.kernel == kernel.description && $0.shape.first == s }
        }
        let floats = sizes.attentionRuns.max()! * h * d
        let pages = (q: Pages(bytes: floats * 4), k: Pages(bytes: floats * 4), v: Pages(bytes: floats * 4), out: Pages(bytes: floats * 4))
        Diagnostic.fill(pages.q.floats(), count: floats, scale: 3, seed: 11)
        Diagnostic.fill(pages.k.floats(), count: floats, scale: 3, seed: 12)
        Diagnostic.fill(pages.v.floats(), count: floats, scale: 1, seed: 13)
        let buffers = (q: try bench.buffer(pages.q), k: try bench.buffer(pages.k), v: try bench.buffer(pages.v), out: try bench.buffer(pages.out))
        let out = pages.out.floats()
        // The tier's longest sequence first: the half-precision kernels' form is chosen there (`chooseForm`), before any of their lines.
        // The others keep their order after it (a pair's sequence reads the tier's below it, measured by then).
        let longest = sizes.attention.max()
        let order = longest.map { l in [l] + sizes.attentionRuns.filter { $0 != l } } ?? sizes.attentionRuns
        var forms: [Kernel: AttentionTile?] = [:]
        for s in order {
            // The tier's sequences run every kernel; the pair's alone (`Sizes.pairSequences`) its two sides and the pair.
            let kernels = sizes.attention.contains(s)
            autoreleasepool {
                let rows = Diagnostic.sampleRows(s, count: Diagnostic.sampledQueries)
                let exact = Diagnostic.exactAttention(q: pages.q.floats(), k: pages.k.floats(), v: pages.v.floats(), sequence: s, heads: h, headDim: d, rows: rows)
                let flop = 4.0 * Double(s) * Double(s) * Double(h) * Double(d)
                func verdict(_ kernel: String, threshold: Double) -> (pass: Bool, error: Double) {
                    let width = h * d, vectors = rows.count * h
                    var got = [Float](repeating: 0, count: vectors * d)
                    for (i, row) in rows.enumerated() {
                        for head in 0..<h { for x in 0..<d { got[x * vectors + i * h + head] = out[row * width + head * d + x] } }
                    }
                    let judged = Diagnostic.judge(got, exact, columns: vectors, threshold: threshold)
                    keep(kernel, error: judged.error, threshold: threshold, pass: judged.pass)
                    return judged
                }
                func verdict(_ kernel: String) -> (pass: Bool, error: Double) { verdict(kernel, threshold: Diagnostic.fp32Threshold) }
                func poison() { for row in rows { (out + row * h * d).update(repeating: .nan, count: h * d) } }
                func record(_ kernel: Kernel, flop: Double, threshold: Double = Diagnostic.fp32Threshold, run: () -> Bool, shape: [Int]) {
                    poison()
                    guard run() else { return }
                    let v = verdict(kernel.description, threshold: threshold)
                    guard v.pass else { measurements.wrong.append(kernel.description); return }
                    guard let t = guarded("\(kernel.description) S=\(s)", { CapacityProbes.steady(run) }) else { return }
                    measurements.solos.append(.init(kernel: kernel.description, shape: shape, flop: flop, seconds: t, error: v.error))
                    log?(String(format: "  %@ S=%d  %.2f ms  %.2f TFLOP/s", kernel.description, s, t * 1e3, flop / t * 1e-12))
                }
                if kernels {
                    let graph = Attention(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d)
                    RouteBeacon.announce("gpu/graph")
                    record(Kernel(.gpu, .graph), flop: flop, run: { _ = graph.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out); return true }, shape: [s, h, d])
                }
                RouteBeacon.announce("gpu/flash")
                if kernels, !skipped.contains(Kernel(.gpu, .flash)), let flash = try? FlashMatrix(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d) {
                    record(Kernel(.gpu, .flash), flop: flop, run: { _ = flash.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out); return true }, shape: [s, h, d])
                }
                // The tiled kernels. fp32: the graph's bar. fp16 operands (Q, K, V, P rounded to half, fp32 sums): the bar of every fp16
                // route — 1.25 × what rounding the exact output to bf16 does (invariant 3; the ANE's, `aneThreshold`). bf16 operands: 1.25 ×
                // bf16's own error — the exact attention of the operands rounded to bf16, its output rounded too (what a bf16 reference
                // computes at best). Each only on a machine that runs it (the probe measures them everywhere: the permission is the
                // request's, at the solver). The half-precision ones run the form their map chose (three stages or fused, `chooseForm`).
                for (kernel, precision) in [(Kernel(.gpu, .tiles), TiledAttention.Precision.fp32), (Kernel(.gpu, .tiles16), .fp16), (Kernel(.gpu, .tilesbf16), .bf16)] {
                    RouteBeacon.announce(kernel.description)
                    guard kernels, !skipped.contains(kernel) else { continue }
                    var bar = Diagnostic.fp32Threshold
                    let bf16 = Diagnostic.Half.bf16
                    switch precision {
                    case .fp32: break
                    case .fp16:
                        let floor = Diagnostic.judge(exact.map { bf16.round(Float($0)) }, exact, columns: rows.count * h, threshold: 1).error
                        bar = max(Diagnostic.fp32Threshold, Diagnostic.floorMargin * floor)
                    case .bf16:
                        let n = s * h * d
                        let rounded = [pages.q.floats(), pages.k.floats(), pages.v.floats()].map { p in (0..<n).map { bf16.round(p[$0]) } }
                        let own = rounded[0].withUnsafeBufferPointer { q in rounded[1].withUnsafeBufferPointer { k in rounded[2].withUnsafeBufferPointer { v in
                            Diagnostic.exactAttention(q: q.baseAddress!, k: k.baseAddress!, v: v.baseAddress!, sequence: s, heads: h, headDim: d, rows: rows)
                        } } }
                        let floor = Diagnostic.judge(own.map { bf16.round(Float($0)) }, exact, columns: rows.count * h, threshold: 1).error
                        bar = max(Diagnostic.fp32Threshold, Diagnostic.floorMargin * floor)
                    }
                    func build(_ tile: AttentionTile?) -> TiledAttention? {
                        try? TiledAttention(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d, precision: precision, tile: tile)
                    }
                    if precision != .fp32, forms[kernel] == nil, s == longest {
                        forms[kernel] = chooseForm(kernel, &measurements, flop: flop, shape: [s, h, d], threshold: bar, build: build,
                                                   run: { _ = $0.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out) },
                                                   poison: poison, verdict: verdict)
                    }
                    let form = forms[kernel] ?? nil
                    guard let tiled = build(form) ?? (form != nil ? build(nil) : nil) else { continue }
                    record(kernel, flop: flop, threshold: bar,
                           run: { _ = tiled.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out); return true }, shape: [s, h, d])
                }
                // The CPU's heads: all but one of them, the figure being the CPU part's own wall.
                RouteBeacon.announce("cpu/heads")
                if kernels, !skipped.contains(Kernel(.cpu, .heads)), let split = try? HeadSplitAttention(device: bench.device, queue: bench.queue, heads: h, sequence: s, headDim: d,
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
                            if Sizes.pairSequences.contains(s), let graph = solo(Kernel(.gpu, .graph), at: s) {
                                attentionSet(&measurements, sequence: s, cpuSolo: (split.cpuHeads, t), graph: graph.seconds, pages: pages, buffers: buffers,
                                             verdict: verdict, poison: poison)
                            }
                        }
                    }
                }
                // A sequence only the pair runs at (`Sizes.pairSequences`, above the tier's): its sides alone are the tier's longest scaled by
                // the work (S²; both lines have no t₀). Only the balance needs them — the fit reads each side through its line (`Fit.overlap`).
                if !kernels, Sizes.pairSequences.contains(s), !skipped.contains(Kernel(.cpu, .heads)), let s0 = sizes.attention.filter({ $0 < s }).max(),
                   let graph = solo(Kernel(.gpu, .graph), at: s0), let cpu = solo(Kernel(.cpu, .heads), at: s0), cpu.shape.count == 4 {
                    let scale = Double(s) * Double(s) / (Double(s0) * Double(s0))
                    attentionSet(&measurements, sequence: s, cpuSolo: (cpu.shape[3], cpu.seconds * scale), graph: graph.seconds * scale,
                                 pages: pages, buffers: buffers, verdict: verdict, poison: poison)
                }
            }
        }
        // The GPU's wake (`CapacityProbesWake.swift`), on the half-precision kernel in the form its map chose.
        let tiles16 = Kernel(.gpu, .tiles16)
        if let short = sizes.attention.min(), let long = sizes.attention.max(), short < long, !skipped.contains(tiles16),
           !measurements.wrong.contains(tiles16.description),
           let busy = try? TiledAttention(device: bench.device, queue: bench.queue, heads: h, sequence: long, headDim: d, precision: .fp16, tile: forms[tiles16] ?? nil),
           let timed = try? TiledAttention(device: bench.device, queue: bench.queue, heads: h, sequence: short, headDim: d, precision: .fp16, tile: forms[tiles16] ?? nil) {
            _ = guarded("gpu wake") { () -> Bool? in wake(&measurements, busy: busy, timed: timed, sequence: short, buffers: buffers); return true }
        }
    }

    /// **The attention as a set of two engines**: the system graph on the GPU's heads beside the CPU's, at the heads the two solo lines
    /// balance, each side by the clock the split gives it (`HeadSplitAttention.run`: the GPU part's, the CPU part's wall). A side alone is its
    /// solo line scaled to its heads (both are proportional to them: the graph's and the heads' t₀ are nil). The same fit as the products'
    /// sets (`Fit.sharing`): a trial of `heads` rows of `2·S²·Dh` each.
    func attentionSet(_ measurements: inout Fit.Measurements, sequence s: Int, cpuSolo: (heads: Int, seconds: Double), graph: Double,
                      pages: (q: Pages, k: Pages, v: Pages, out: Pages),
                      buffers: (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, out: MTLBuffer),
                      verdict: (String) -> (pass: Bool, error: Double), poison: () -> Void) {
        let (h, d) = (Sizes.attentionHeads, Sizes.attentionHeadDim)
        let key = CapacityModel.key([.gpu, .cpu])
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
}
