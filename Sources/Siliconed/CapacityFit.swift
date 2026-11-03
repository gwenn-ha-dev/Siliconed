import Foundation

/// **From the probes' raw measurements to the `CapacityModel`: pure arithmetic, no Metal, no clock.** The machine layer
/// (`CapacityProbes`) fills a `CapacityMeasurements`; this file reads it. Kept apart so that every fit is tested against
/// synthetic data whose answer is known, and so that a report from another Mac can be fitted again offline.
///
///     solos      one kernel, one product or attention, seconds          →  t(W) = t₀ + W / rate     weighted least squares
///     stairs     the ANE alone at several row counts, one k × n         →  the unit u and t = t₀ + c · ⌈rows / u⌉ · u
///     trials     a set of engines on one product: each side alone at the rows it takes, then the set
///                                                                       →  keep[engine], overhead
///     witnesses  the same product before and after                      →  heat = after / before
///     chains     a set of engines back to back on one product, each side's own clock, the lines as the reference
///                                                                       →  κ per engine (the speed inside the set over alone), overhead
///
/// What sets it apart — what a fit written by analogy with "a line through the points" would get wrong:
///
///   - **a line is fitted on the relative error** (weights 1/t²): the 300 GFLOP point is a hundred times the 4 GFLOP one in
///     seconds, and an unweighted fit would be the big one's alone — the small one is where t₀ lives. A negative t₀ is a
///     fit that reached past its data: it is clamped to 0 and the rate refitted through the origin.
///   - **a disturbed figure is measured again, never left out by the fit** (`suspects`): a point more than the regime's spread (15 %)
///     above the line of the others is measured once more by the probes and the faster kept; the line is then fitted on every point.
///     A kernel's own shape effects (tile counts against the GPU's cores) put genuine points 20–60 % off any line: a trim by distance
///     cannot tell them from disturbances, a second measure can.
///   - **the map's noise is the rounds' own** (`noise`): the median, over the tiles' and attention forms' candidates, of the spread of
///     their repeated rounds — what a figure of this map is sure to; the solver's margin is twice it.
///   - **the ANE's unit is read, not assumed**: the candidates are fitted on the same rows and the *largest* unit that
///     explains them as well as the best (within half a percent) wins — samples all at multiples of 256 cannot tell 256 from
///     128, and the coarser is the one the model's rounding needs; an odd multiple of 256 in the samples is what rules 512 out.
///   - **keep and overhead are separated by the sides' own clocks, not by the wall**: each side's seconds in the set, against
///     the same rows alone, give `keep` (the share of its solo rate it keeps: (solo − t₀) / (side − t₀)); what the wall
///     adds beyond the slowest side is the `overhead` (the rendezvous, the join, the launch that no side owns). From the wall
///     alone the two cannot be told apart at one size, and with two sizes only when the sides are balanced; without a
///     side's clock, that side is taken to end with the set (keep from the wall, no overhead for it).
///   - **a set's fit is judged on the walls it did not use for keep**: `residual` is the model's `seconds` on the trial's
///     own rows against the wall, relative — the number that says whether "keep per engine + one fixed overhead" holds.
package enum CapacityFit {
    package typealias Kernel = CapacityModel.Kernel

    // MARK: - Lines

    package struct Point: Codable, Sendable, Equatable {
        package var flop: Double
        package var seconds: Double
        package init(flop: Double, seconds: Double) { self.flop = flop; self.seconds = seconds }
    }

    package static func median(_ values: [Double]) -> Double {
        let s = values.sorted()
        guard !s.isEmpty else { return .nan }
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    /// `t = a + b · x`, weights 1/t² (relative error); `nil` when the points do not give a positive slope.
    private static func affine(_ points: [(x: Double, t: Double)]) -> (a: Double, b: Double)? {
        let p = points.filter { $0.t > 0 && $0.x > 0 }
        guard !p.isEmpty else { return nil }
        if p.count == 1 { return (0, p[0].t / p[0].x) }
        var (s0, sx, sy, sxx, sxy) = (0.0, 0.0, 0.0, 0.0, 0.0)
        for (x, t) in p {
            let w = 1 / (t * t)
            s0 += w; sx += w * x; sy += w * t; sxx += w * x * x; sxy += w * x * t
        }
        let d = s0 * sxx - sx * sx
        guard d.magnitude > 1e-300 else { return nil }
        var b = (s0 * sxy - sx * sy) / d
        var a = (sy - b * sx) / s0
        if a < 0 || b <= 0 {
            a = 0
            b = sxy / sxx
        }
        return b > 0 ? (a, b) : nil
    }

    /// **A kernel's line: the weighted least squares on every point.** None is left out: a point far from the line is either a figure
    /// the Mac disturbed — measured again by the probes before the fit (`suspects`), the faster kept — or the kernel's own shape (a tile
    /// count that fills the GPU's cores badly: on an M5 Pro the rounded bf16 kernel runs 1 024 × 3 840² at 11 TFLOP/s and 2 048 × 2 048 ×
    /// 4 096 at 20, map after map), which a line must average, not drop. Leaving out what lies above a lower envelope dropped exactly those:
    /// one map kept the small product and lost the 4 096-row ones the renders run (+21 and +34 % above its line), the next kept them and
    /// lost a small product disturbed 2.2× — two lines 16 % apart from the same silicon.
    package static func line(_ points: [Point]) -> CapacityModel.Line? {
        affine(points.filter { $0.seconds > 0 && $0.flop > 0 }.map { ($0.flop, $0.seconds) }).map { .init(t0: $0.a, rate: 1 / $0.b) }
    }

    /// **The points to measure again before a line is fitted**: those more than `Regime.Rules.spread` above the line the *other* points
    /// give (leave one out) — a busy Mac only adds time. The probes measure each once more and keep the faster: a disturbed figure comes
    /// back to the line, a shape the kernel is slow at stays where it is. Under three points none can be told: none.
    package static func suspects(_ points: [Point]) -> [Int] {
        guard points.count >= 3 else { return [] }
        return points.indices.filter { i in
            let others = points.enumerated().filter { $0.offset != i }.map(\.element)
            guard let l = line(others), points[i].seconds > 0 else { return false }
            return points[i].seconds > (l.t0 + points[i].flop / l.rate) * (1 + Diagnostic.Regime.Rules.spread)
        }
    }

    /// **The intercept the points give before it is clamped at 0**: the fit's own, on the same weights. A line through two points
    /// of a kernel that is nearly proportional to W comes out a hair under 0 (the system graph, the CPU's heads: −0.6 to 0 ms against
    /// points of 9 to 12 ms); a line whose intercept is a quarter of its smallest point below 0 is not a line the points sit on.
    /// `nil` under two points.
    package static func rawIntercept(_ points: [Point]) -> Double? {
        let p = points.filter { $0.seconds > 0 && $0.flop > 0 }
        guard p.count >= 2 else { return nil }
        var (s0, sx, sy, sxx) = (0.0, 0.0, 0.0, 0.0)
        for q in p {
            let w = 1 / (q.seconds * q.seconds)
            s0 += w; sx += w * q.flop; sy += w * q.seconds; sxx += w * q.flop * q.flop
        }
        var sxy = 0.0
        for q in p { sxy += q.flop * q.seconds / (q.seconds * q.seconds) }
        let d = s0 * sxx - sx * sx
        guard d.magnitude > 1e-300 else { return nil }
        let b = (s0 * sxy - sx * sy) / d
        return (sy - b * sx) / s0
    }

    /// The line's relative error on its own points: median and worst.
    package static func relativeError(_ line: CapacityModel.Line, _ points: [Point]) -> (median: Double, worst: Double) {
        let e = points.map { abs(line.seconds($0.flop) - $0.seconds) / $0.seconds }
        return (median(e), e.max() ?? .nan)
    }

    // MARK: - The ANE's staircase

    package struct Stairs: Codable, Sendable, Equatable {
        package var unit: Int
        /// Seconds fixed, and seconds per row (of the rows rounded up to `unit`).
        package var t0: Double
        package var perRow: Double
        /// Relative rms of the fit per candidate unit (the evidence).
        package var rms: [Int: Double]
    }

    package static func stairs(_ samples: [(rows: Int, seconds: Double)], candidates: [Int] = [64, 128, 256, 512, 1024, 2048]) -> Stairs? {
        var fits: [(unit: Int, a: Double, b: Double, rms: Double)] = []
        for u in candidates {
            let xs = samples.map { (x: Double(($0.rows + u - 1) / u * u), t: $0.seconds) }
            guard let f = affine(xs) else { continue }
            let rms = (xs.map { pow((f.a + f.b * $0.x - $0.t) / $0.t, 2) }.reduce(0, +) / Double(xs.count)).squareRoot()
            fits.append((u, f.a, f.b, rms))
        }
        guard let best = fits.map(\.rms).min() else { return nil }
        guard let pick = fits.filter({ $0.rms <= best + 0.005 }).max(by: { $0.unit < $1.unit }) else { return nil }
        return Stairs(unit: pick.unit, t0: pick.a, perRow: pick.b, rms: Dictionary(uniqueKeysWithValues: fits.map { ($0.unit, $0.rms) }))
    }

    // MARK: - Measurements

    /// The raw numbers of a run of the probes: what the report carries, and what `model(from:)` reads.
    package struct Measurements: Codable, Sendable, Equatable {
        /// One kernel on one product (`shape` = [m, k, n]) or attention (`shape` = [sequence, heads, headDim], or the CPU
        /// heads' count last), `flop` its work, `seconds` the steady run's, `error` the judge's worst channel error.
        package struct Solo: Codable, Sendable, Equatable {
            package var kernel: String
            package var shape: [Int]
            package var flop: Double
            package var seconds: Double
            package var error: Double?
            package init(kernel: String, shape: [Int], flop: Double, seconds: Double, error: Double? = nil) {
                self.kernel = kernel; self.shape = shape; self.flop = flop; self.seconds = seconds; self.error = error
            }
        }
        package struct Stair: Codable, Sendable, Equatable {
            package var rows: Int
            package var seconds: Double
            package init(rows: Int, seconds: Double) { self.rows = rows; self.seconds = seconds }
        }
        /// One side of a set: the rows it took, its seconds alone at those rows, and its seconds within the set (its own clock).
        package struct Side: Codable, Sendable, Equatable {
            package var kernel: String
            package var rows: Int
            package var solo: Double
            package var within: Double?
            package init(kernel: String, rows: Int, solo: Double, within: Double? = nil) {
                self.kernel = kernel; self.rows = rows; self.solo = solo; self.within = within
            }
        }
        package struct Trial: Codable, Sendable, Equatable {
            /// `gpu+ane`, in `CapacityModel.key` order.
            package var set: String
            package var shape: [Int]
            /// 1: the rows the solo lines (no sharing) balance; 2: the rows the first pass's keep balances; 3: a long k on pass 2's;
            /// `again` (4): the short shape a third time, where the first two disagreed (`CapacityFit.needsAgain`).
            package var pass: Int
            package static let again = 4
            package var sides: [Side]
            package var wall: Double
            package var error: Double?
            package init(set: String, shape: [Int], pass: Int = 1, sides: [Side], wall: Double, error: Double? = nil) {
                self.set = set; self.shape = shape; self.pass = pass; self.sides = sides; self.wall = wall; self.error = error
            }
        }
        package struct Witness: Codable, Sendable, Equatable {
            package var engine: String
            package var before: Double
            package var after: Double
            package init(engine: String, before: Double, after: Double) { self.engine = engine; self.before = before; self.after = after }
        }

        package var solos: [Solo] = []
        /// **Products with a short side** (`CapacityModel.thinSide`), taken apart from the `solos` so that the big products' lines never see them:
        /// each kernel at the shapes of `Sizes.thin`. `nil` in a report that predates them.
        package var thin: [Solo]?
        package var stairShape: [Int] = []
        package var stairs: [Stair] = []
        package var trials: [Trial] = []
        package var witnesses: [Witness] = []
        package var memoryBandwidth: Double?
        /// Bytes per second the weight stream's own read gets from the library's volume (`CapacityDisk.measure`).
        package var diskBandwidth: Double?
        /// The same read at each class of service of the stream's threads (`CapacityDisk.measureClasses`), by class name.
        package var diskClasses: [String: CapacityDisk.Reading] = [:]
        /// Seconds per value to bring a weight to the form a kernel reads, by kind then by path (`"int8": {"widened": …}`).
        package var preparation: [String: [String: Double]] = [:]
        /// Kernels the judge refused (a wrong route fills nothing).
        package var wrong: [String] = []
        /// **The tile chosen per kernel** (`"gpu/bf16act": "128x128/8"`, `MatmulTile`) for its wide products and for its thin ones, and each
        /// candidate's solos that chose them (`kernel` = `"gpu/bf16act 128x128/8"`, one per round and shape). `nil` in a report that predates them.
        package var tiles: [String: String]?
        package var thinTiles: [String: String]?
        /// The form each half-precision attention kernel runs (`AttentionTile` text), measured before its line; absent: the three stages.
        package var attentionTiles: [String: String]?
        package var tileTrials: [Solo]?
        /// **The GPU's wake** (`Context.wake`): the tier's shortest tiled attention timed after the longest and an idle, one per idle (its
        /// microseconds last in `shape`). `nil` in a report that predates it.
        package var wake: [Solo]?
        package init() {}

        /// **The same numbers to `digits` significant digits** — what a report keeps (an exact FLOP count stays exact). The model of a
        /// report is fitted on these, so that anyone replaying the fit offline obtains it to the last digit (`Report.refit`).
        package func rounded(_ digits: Int = 5) -> Measurements {
            func r(_ x: Double) -> Double { Diagnostic.significant(x, digits) }
            var m = self
            m.solos = solos.map { Solo(kernel: $0.kernel, shape: $0.shape, flop: $0.flop, seconds: r($0.seconds),
                                       error: $0.error.map { Diagnostic.significant($0, 3) }) }
            m.thin = thin?.map { Solo(kernel: $0.kernel, shape: $0.shape, flop: $0.flop, seconds: r($0.seconds), error: $0.error.map { Diagnostic.significant($0, 3) }) }
            m.tileTrials = tileTrials?.map { Solo(kernel: $0.kernel, shape: $0.shape, flop: $0.flop, seconds: r($0.seconds), error: $0.error.map { Diagnostic.significant($0, 3) }) }
            m.stairs = stairs.map { Stair(rows: $0.rows, seconds: r($0.seconds)) }
            m.trials = trials.map { t in
                Trial(set: t.set, shape: t.shape, pass: t.pass,
                      sides: t.sides.map { Side(kernel: $0.kernel, rows: $0.rows, solo: r($0.solo), within: $0.within.map(r)) },
                      wall: r(t.wall), error: t.error.map { Diagnostic.significant($0, 3) })
            }
            m.witnesses = witnesses.map { Witness(engine: $0.engine, before: r($0.before), after: r($0.after)) }
            m.memoryBandwidth = memoryBandwidth.map(r)
            m.diskBandwidth = diskBandwidth.map(r)
            m.diskClasses = diskClasses.mapValues { .init(rate: r($0.rate), mean: r($0.mean)) }
            m.preparation = preparation.mapValues { $0.mapValues(r) }
            m.wake = wake?.map { Solo(kernel: $0.kernel, shape: $0.shape, flop: $0.flop, seconds: r($0.seconds)) }
            return m
        }
    }

    // MARK: - Sharing

    package struct SharingFit: Codable, Sendable, Equatable {
        package var sharing: CapacityModel.Sharing
        /// Per trial: each side's keep, the overhead, and the model's relative error on the wall.
        package var keeps: [[Double?]]
        package var overheads: [Double]
        package var residuals: [Double]
    }

    /// Keeps and overhead of one set from its trials, on `lines` (the solo lines, whose t₀ the sides' fixed part comes from).
    package static func sharing(set: String, trials: [Measurements.Trial], lines: [Kernel: CapacityModel.Line],
                                using passes: Set<Int>? = nil) -> SharingFit? {
        let used = trials.filter { $0.set == set && !$0.sides.isEmpty && (passes?.contains($0.pass) ?? true) }
        guard !used.isEmpty else { return nil }
        var keepsByEngine: [CapacityModel.Engine: [Double]] = [:]
        var keeps: [[Double?]] = [], overheads: [Double] = []
        for trial in used {
            var row: [Double?] = []
            var slowest = 0.0
            for side in trial.sides {
                guard let kernel = Kernel(side.kernel) else { row.append(nil); continue }
                let within = side.within ?? trial.wall
                slowest = max(slowest, within)
                let t0 = lines[kernel]?.t0 ?? 0
                let paid = within - t0, alone = side.solo - t0
                guard paid > 0, alone > 0 else { row.append(nil); continue }
                let keep = min(1.5, max(0.05, alone / paid))
                row.append(keep)
                keepsByEngine[kernel.engine, default: []].append(keep)
            }
            keeps.append(row)
            overheads.append(max(0, trial.wall - slowest))
        }
        var fit = CapacityModel.Sharing(overhead: median(overheads), keep: keepsByEngine.mapValues(median))
        fit.overhead = fit.overhead.isFinite ? fit.overhead : 0
        // The model's own time on each trial's rows, against the wall.
        var model = CapacityModel()
        model.lines = lines
        model.sharing[set] = fit
        var residuals: [Double] = []
        for trial in used {
            var rows: [Kernel: Int] = [:]
            for s in trial.sides { if let k = Kernel(s.kernel) { rows[k] = s.rows } }
            if trial.shape.count == 3, let t = model.seconds(m: trial.shape[0], k: trial.shape[1], n: trial.shape[2], rows: rows) {
                residuals.append((t - trial.wall) / trial.wall)
            }
        }
        return SharingFit(sharing: fit, keeps: keeps, overheads: overheads, residuals: residuals)
    }

    // MARK: - Lines with a shape

    /// Solves `A x = b` (normal equations, Gaussian elimination with partial pivoting); `nil` when singular.
    private static func solve(_ a: [[Double]], _ b: [Double]) -> [Double]? {
        var m = zip(a, b).map { $0 + [$1] }
        let n = b.count
        for c in 0..<n {
            guard let p = (c..<n).max(by: { abs(m[$0][c]) < abs(m[$1][c]) }), abs(m[p][c]) > 1e-300 else { return nil }
            m.swapAt(c, p)
            for r in (c + 1)..<max(c + 1, n) {
                let f = m[r][c] / m[c][c]
                for j in c...n { m[r][j] -= f * m[c][j] }
            }
        }
        var x = [Double](repeating: 0, count: n)
        for r in stride(from: n - 1, through: 0, by: -1) {
            x[r] = (m[r][n] - ((r + 1)..<max(r + 1, n)).reduce(0) { $0 + m[r][$1] * x[$1] }) / m[r][r]
        }
        return x
    }

    /// Weighted least squares of `t ≈ Σ xⱼ·fⱼ`, the weights `1/t²` (the relative error, as `affine`).
    private static func regress(_ rows: [(features: [Double], seconds: Double)]) -> [Double]? {
        guard let width = rows.first?.features.count, rows.count >= width else { return nil }
        var a = [[Double]](repeating: [Double](repeating: 0, count: width), count: width), b = [Double](repeating: 0, count: width)
        for (f, t) in rows where t > 0 {
            let w = 1 / (t * t)
            for i in 0..<width { b[i] += w * f[i] * t; for j in 0..<width { a[i][j] += w * f[i] * f[j] } }
        }
        return solve(a, b)
    }

    /// The same with every coefficient kept non-negative: the term that would go negative is dropped and the rest solved again (a small
    /// active set — the fits here have three or four terms).
    private static func regressNonNegative(_ rows: [(features: [Double], seconds: Double)]) -> [Double]? {
        guard let width = rows.first?.features.count else { return nil }
        var active = Array(0..<width)
        while !active.isEmpty {
            let sub = rows.map { row in (features: active.map { row.features[$0] }, seconds: row.seconds) }
            guard let x = regress(sub) else { return nil }
            if let worst = x.indices.min(by: { x[$0] < x[$1] }), x[worst] < 0 { active.remove(at: worst); continue }
            var full = [Double](repeating: 0, count: width)
            for (i, j) in active.enumerated() { full[j] = x[i] }
            return full
        }
        return nil
    }

    /// **The ANE's line with the host's passes in it**: `t = t₀ + w·k·n + h·rows·(k·r + n·parts) + 2·rows·k·n / R`, `h` the seconds per element of the
    /// two passes the host makes over a call's rows (the activation — rotated twice where K is cut, for a convrot weight —, the output), `parts` the
    /// ANE's own cut of K. `nil` unless the solos tell the terms apart (four distinct shapes at least, the host term and the rate both positive).
    private static func hostedLine(kernel: Kernel, solos used: [Measurements.Solo]) -> CapacityModel.Line? {
        guard used.count >= 4, Set(used.map { $0.shape }).count >= 4 else { return nil }
        let int8 = kernel.form == .int8
        func elements(_ s: Measurements.Solo) -> Double {
            let (m, k, n) = (s.shape[0], s.shape[1], s.shape[2])
            let p = NeuralEngine.parts(k: k, n: n, int8: int8, alignment: int8 ? Widen.largestRotationGroup : 64)
            return Double(m) * (Double(k) * (int8 && p > 1 ? 2 : 1) + Double(n) * Double(p))
        }
        // With the weight's lift when there are enough shapes to tell it from the host passes, without it otherwise.
        let withLift = used.count >= 6
        let rows = used.map { s -> (features: [Double], seconds: Double) in
            var f: [Double] = [1]
            if withLift { f.append(Double(s.shape[1] * s.shape[2]) * 1e-9) }
            f += [elements(s) * 1e-9, s.flop * 1e-12]
            return (f, s.seconds)
        }
        guard let x = regressNonNegative(rows), x[x.count - 1] > 0, x[x.count - 2] > 0 else { return nil }
        return .init(t0: x[0], rate: 1e12 / x[x.count - 1], perWeight: withLift && x[1] > 0 ? x[1] * 1e-9 : nil, perRow: x[x.count - 2] * 1e-9)
    }

    /// **The line of a kernel that reads a product, with the shapes the solos were taken at.** Alone, `t₀ + W / rate` is blind to the two
    /// things a family's shapes change: the ANE pays the weight's lift at every call (`perWeight`: `t = t₀ + w·k·n + W / rate`, which
    /// takes 3 products of different weight sizes at a given W apart where a line cannot), and the AMX's rate falls with k (`rateByK`:
    /// the seconds per FLOP linear in ln k between the k's the solos came at). A fit with a negative term, or too few distinct shapes
    /// to tell the terms apart, is the plain line.
    package static func productLine(kernel: Kernel, solos: [Measurements.Solo]) -> CapacityModel.Line? {
        var line = shapedLine(kernel: kernel, solos: solos)
        // The domain in rows (`Line.rowsFrom`): the AMX and the ANE are measured from the rows of the smallest solo, and asked for no fewer than half.
        if kernel.engine == .amx || kernel.engine == .ane, let smallest = solos.filter({ $0.shape.count == 3 && $0.seconds > 0 }).map({ $0.shape[0] }).min() {
            line?.rowsFrom = max(1, smallest / 2)
        }
        return line
    }

    private static func shapedLine(kernel: Kernel, solos: [Measurements.Solo]) -> CapacityModel.Line? {
        let used = solos.filter { $0.shape.count == 3 && $0.seconds > 0 && $0.flop > 0 }
        let plain = line(used.map { Point(flop: $0.flop, seconds: $0.seconds) })
        guard used.count >= 3 else { return plain }
        switch kernel.engine {
        case .ane:
            if let hosted = hostedLine(kernel: kernel, solos: used) { return hosted }
            let weights = Set(used.map { $0.shape[1] * $0.shape[2] })
            guard weights.count >= 2,
                  let x = regress(used.map { ([1, Double($0.shape[1] * $0.shape[2]) * 1e-9, $0.flop * 1e-12], $0.seconds) }),
                  x[0] >= 0, x[1] > 0, x[2] > 0 else { return plain }
            return .init(t0: x[0], rate: 1e12 / x[2], perWeight: x[1] * 1e-9)
        case .amx:
            let knots = Array(Set(used.map { $0.shape[1] })).sorted()
            guard knots.count >= 2 else { return plain }
            // Hat functions of ln k at the knots: the seconds per FLOP between two are the two knots' blend.
            func basis(_ k: Int) -> [Double] {
                var h = [Double](repeating: 0, count: knots.count)
                if k <= knots[0] { h[0] = 1 } else if k >= knots[knots.count - 1] { h[knots.count - 1] = 1 } else {
                    for i in 0..<(knots.count - 1) where knots[i] <= k && k <= knots[i + 1] {
                        let f = (log(Double(k)) - log(Double(knots[i]))) / (log(Double(knots[i + 1])) - log(Double(knots[i])))
                        h[i] = 1 - f; h[i + 1] = f
                    }
                }
                return h
            }
            // t₀ held at zero where the free fit would make it negative: a fixed cost within its error of zero is zero, not a reason to
            // drop the knots (the AMX's line changed form between two maps of one Mac, its t₀ 0.20 or 0.68 ms with it).
            guard let x = regressNonNegative(used.map { solo in ([1] + basis(solo.shape[1]).map { $0 * solo.flop * 1e-12 }, solo.seconds) }),
                  x.dropFirst().allSatisfy({ $0 > 0 }) else { return plain }
            let rates = x.dropFirst().map { 1e12 / $0 }
            return .init(t0: x[0], rate: rates[0], rateByK: zip(knots, rates).map { .init(k: $0, rate: $1) })
        default:
            return plain
        }
    }

    /// **The line of a kernel on the products with a short side**: `t = t₀ + s·(Bw·k·n + Ba·M·(k + n)) / BW + 2·M·k·n / R` — the weight read once
    /// (`Bw` bytes an element, the form's), the activation and the output streamed once (`Ba` bytes an element: fp32 each, plus the passes the form
    /// makes over them), at the memory's bandwidth `BW`, then the arithmetic. `M` is the rows the kernel *computes*, not the rows it is given
    /// (`CapacityModel.computedRows`: a GPU kernel works in tiles of 64 rows, so a prompt of 19 rows costs 64).
    ///
    /// Two numbers are fitted, `t₀` and the bandwidth's share `s` (1 when the kernel streams at the memory's rate; 0.9 to 1.25 for the GPU's forms on
    /// the M1 Pro); the bytes are the forms', the rate `R` the big products' (the engine's ceiling). The earlier forms of this line fitted
    /// the streaming terms free — four terms on six shapes, then three — and reproduced the probed shapes with a rate twice the silicon's, which
    /// missed a LoRA's `4 117 × 3 072 · 128` by a third; what the two numbers here cannot hold is the kernel's, not the line's. Fitted on the
    /// relative error (a 1 ms shape counts as much as a 12 ms one), a shape that comes out at over twice its time is left out of the second pass —
    /// the GPU's int8 and rounded-activation kernels run 8 times slower at `n = 128` (a cliff of the kernel, which no class falls in).
    ///
    /// `nil` without the big products' line or the memory's bandwidth, or unless the shapes tell the two apart (three distinct at least).
    package static func thinLine(kernel: Kernel, solos: [Measurements.Solo], arithmetic: CapacityModel.Line?, bandwidth: Double?) -> CapacityModel.Line? {
        thinFit(kernel: kernel, solos: solos, arithmetic: arithmetic, bandwidth: bandwidth)?.line
    }

    /// A thin solo's two terms on `arithmetic`'s rate: the bytes streamed at the bandwidth, and the arithmetic.
    private static func thinTerms(_ kernel: Kernel, _ s: Measurements.Solo, arithmetic: CapacityModel.Line, bandwidth: Double) -> (stream: Double, arithmetic: Double) {
        let (weightBytes, activationBytes) = CapacityModel.streamedBytes(kernel.form)
        let (k, n) = (Double(s.shape[1]), Double(s.shape[2]))
        let rows = Double(CapacityModel.computedRows(kernel, rows: s.shape[0]))
        return ((weightBytes * k * n + activationBytes * rows * (k + n)) / bandwidth, 2 * rows * k * n / arithmetic.rate(atK: s.shape[1]))
    }

    /// `thinLine`, with the solos its second pass kept.
    private static func thinFit(kernel: Kernel, solos: [Measurements.Solo], arithmetic: CapacityModel.Line?,
                                bandwidth: Double?) -> (line: CapacityModel.Line, kept: [Measurements.Solo])? {
        guard let arithmetic, arithmetic.rate > 0, let bandwidth, bandwidth > 0 else { return nil }
        let used = solos.filter { $0.shape.count == 3 && $0.seconds > 0 && $0.flop > 0 }
        guard used.count >= 3, Set(used.map { $0.shape }).count >= 3 else { return nil }
        let (weightBytes, activationBytes) = CapacityModel.streamedBytes(kernel.form)
        func terms(_ s: Measurements.Solo) -> (stream: Double, arithmetic: Double) { thinTerms(kernel, s, arithmetic: arithmetic, bandwidth: bandwidth) }
        func solve(_ kept: [Measurements.Solo]) -> (t0: Double, share: Double)? {
            let rows = kept.map { s -> (features: [Double], seconds: Double) in
                let t = terms(s)
                return ([1 / s.seconds, t.stream / s.seconds], (s.seconds - t.arithmetic) / s.seconds)
            }
            // A share the non-negative fit sets to zero is a share of zero — the stream hidden under the arithmetic and the call's fixed
            // cost —, not a missing line: the fit is continuous in the solos where the share crosses zero.
            guard let x = regressNonNegative(rows) else { return nil }
            return (x[0], x[1])
        }
        guard var fit = solve(used) else { return nil }
        func time(_ s: Measurements.Solo) -> Double { fit.t0 + fit.share * terms(s).stream + terms(s).arithmetic }
        var kept = used.filter { $0.seconds < 2 * time($0) }
        if kept.count < used.count, kept.count >= 3, let again = solve(kept) { fit = again } else { kept = used }
        var line = CapacityModel.Line(t0: fit.t0, rate: arithmetic.rate, perWeight: fit.share * weightBytes / bandwidth, rateByK: arithmetic.rateByK,
                                      perRow: fit.share * activationBytes / bandwidth)
        if kernel.engine == .amx || kernel.engine == .ane, let fewest = used.map({ $0.shape[0] }).min() { line.rowsFrom = max(1, fewest / 2) }
        return (line, kept)
    }

    /// **The thin lines of a map, the GPU's on one fixed cost.** Each kernel's line as `thinLine` fits it; then the GPU's kernels again
    /// together, one `t₀` for all and each its own share of the bandwidth, on the solos each one's own fit kept. A thin product's seconds are
    /// mostly that fixed cost — the call's submission and its completion reaching the CPU, the GPU's whatever the kernel —, and seven
    /// kernels each fitting their own read it 0.09 to 0.18 ms apart in no order a second map kept (two maps of an M5 Pro: `gpu/fp32` 0.175
    /// and 0.142 ms, `gpu/bf16act` 0.159 and 0.172): the choice between two GPU forms of a thin product, a gap of a few hundredths of a
    /// millisecond, followed that noise across the thin margin (`gpu/bf16act` 19 % under `gpu/bf16` in one map, 11 % in the next). Read
    /// once from every GPU solo, the fixed cost is common to the forms, and what tells them apart is what they stream and compute.
    package static func thinLines(_ groups: [Kernel: [Measurements.Solo]], lines: [Kernel: CapacityModel.Line],
                                  bandwidth: Double?) -> [Kernel: CapacityModel.Line] {
        var out: [Kernel: CapacityModel.Line] = [:], kept: [Kernel: [Measurements.Solo]] = [:]
        for (kernel, solos) in groups {
            guard let f = thinFit(kernel: kernel, solos: solos, arithmetic: lines[kernel], bandwidth: bandwidth) else { continue }
            out[kernel] = f.line; kept[kernel] = f.kept
        }
        let gpu = kept.keys.filter { $0.engine == .gpu }.sorted { $0.description < $1.description }
        guard gpu.count >= 2, let bandwidth else { return out }
        var rows: [(features: [Double], seconds: Double)] = []
        for (j, kernel) in gpu.enumerated() {
            guard let arithmetic = lines[kernel] else { continue }
            for s in kept[kernel] ?? [] {
                let t = thinTerms(kernel, s, arithmetic: arithmetic, bandwidth: bandwidth)
                var f = [Double](repeating: 0, count: gpu.count + 1)
                f[0] = 1 / s.seconds; f[j + 1] = t.stream / s.seconds
                rows.append((f, (s.seconds - t.arithmetic) / s.seconds))
            }
        }
        guard let x = regressNonNegative(rows) else { return out }
        for (j, kernel) in gpu.enumerated() {
            guard var line = out[kernel] else { continue }
            let (weightBytes, activationBytes) = CapacityModel.streamedBytes(kernel.form)
            line.t0 = x[0]; line.perWeight = x[j + 1] * weightBytes / bandwidth; line.perRow = x[j + 1] * activationBytes / bandwidth
            out[kernel] = line
        }
        return out
    }

    // MARK: - Sharing as an overlap

    /// **Each engine's κ in a set from one chain** (the speed inside the set over its speed alone, the speed alone read off the line at
    /// the rows it took): the first side to end overlapped the whole time, `κ = s / w`; each later one did the work its line says in
    /// `κ·w₁` plus, phase after phase, what the smaller set it was left in gave it, and the rest alone at speed 1. `sub` answers
    /// for the sets that remain (their κ were fitted first). `nil` when a side has no clock or no line.
    package static func overlapKeeps(sides: [(kernel: Kernel, solo: Double, within: Double)], k: Int,
                                     sub: (_ remaining: [Kernel]) -> CapacityModel.Sharing?) -> [Kernel: Double]? {
        let sorted = sides.sorted { $0.within < $1.within }
        guard sorted.count >= 2, sorted.allSatisfy({ $0.solo > 0 && $0.within > 0 }) else { return nil }
        let first = sorted[0].within
        var keeps: [Kernel: Double] = [sorted[0].kernel: sorted[0].solo / first]
        for (j, side) in sorted.enumerated() where j > 0 {
            // Phases 2…: from the end of side p−1 to the end of side p, the sides p … remain.
            var done = 0.0
            for p in 1...j {
                let remaining = sorted[p...].map(\.kernel)
                let duration = sorted[p].within - sorted[p - 1].within
                if p <= j {
                    let speed = remaining.count == 1 ? 1 : (sub(remaining)?.keep(side.kernel.engine, k: k) ?? 1)
                    done += speed * duration
                }
            }
            keeps[side.kernel] = (side.solo - done) / first
        }
        return keeps.mapValues { min(1.5, max(0.05, $0)) }
    }

    /// **The sharing of a set from its chains**, on the model's lines (`model.lines`, the ANE's unit): κ per engine the median over the
    /// trials, the overhead what the wall adds beyond the last side. The sub-sets' fits are read in `model.sharing`.
    package static func overlap(set: String, trials: [Measurements.Trial], model: CapacityModel) -> SharingFit? {
        let used = trials.filter { $0.set == set && $0.sides.count >= 2 && $0.shape.count == 3 && $0.wall > 0 }
        guard !used.isEmpty else { return nil }
        // What a side pays per call in this set beyond its line, known before the chains (`lift(set:model:)`): out of the κ, into the solo.
        let perWeight = lift(set: set, model: model)
        let paid = CapacityModel.Sharing(perWeight: perWeight)
        var byEngine: [CapacityModel.Engine: [Double]] = [:]
        var byK: [Int: [CapacityModel.Engine: [Double]]] = [:]
        var keeps: [[Double?]] = [], overheads: [Double] = [], residuals: [Double] = []
        for trial in used {
            let (m, k, n) = (trial.shape[0], trial.shape[1], trial.shape[2])
            var sides: [(kernel: Kernel, solo: Double, within: Double)] = []
            for side in trial.sides {
                guard let kernel = Kernel(side.kernel), let s = model.alone(kernel, rows: side.rows, k: k, n: n) else { continue }
                sides.append((kernel, s + paid.lift(kernel.engine, k: k, n: n), side.within ?? trial.wall))
            }
            guard sides.count == trial.sides.count, let ks = overlapKeeps(sides: sides, k: k, sub: { model.sharing($0) }) else { keeps.append([]); continue }
            keeps.append(trial.sides.map { Kernel($0.kernel).flatMap { ks[$0] } })
            for (kernel, keep) in ks { byEngine[kernel.engine, default: []].append(keep); byK[k, default: [:]][kernel.engine, default: []].append(keep) }
            overheads.append(max(0, trial.wall - (sides.map(\.within).max() ?? trial.wall)))
        }
        guard !byEngine.isEmpty else { return nil }
        var fit = CapacityModel.Sharing(overhead: median(overheads), keep: byEngine.mapValues(median), perWeight: perWeight)
        fit.overhead = fit.overhead.isFinite ? fit.overhead : 0
        // κ by k where the chains came at more than one: each engine's median at each k.
        if let low = byK.keys.min(), let high = byK.keys.max() { fit.kFrom = max(1, low / 2); fit.kTo = high * 2 }
        // **The attention's pair has no ceiling.** Its k is a head's work (2·S·keys·Dh), which no probe can follow to the formats (a 1536 × 1024
        // render is S = 6 144, k = 9.7·10⁹, past the 8.6·10⁹ of the longest S measured; a 2048² one 16 384): a ceiling there takes the cut by
        // heads off every large format. Above it, the κ is the longest sequence's (`Sharing.keep`): the sides stream no weight — the reason the
        // products' sets are bounded, where the AMX's κ halves as the weight grows —, and the κ measured rises with S (M1 Pro: 0.94–0.95 at
        // S = 1 024, 0.96–1.0 at 4 096): held at the longest, it errs on the safe side.
        if set == CapacityModel.key([.gpu, .cpu]) { fit.kTo = nil }
        if byK.count >= 2 {
            fit.keepByK = byK.keys.sorted().map { k in
                .init(k: k, keep: Dictionary(uniqueKeysWithValues: byK[k]!.map { ($0.key.rawValue, median($0.value)) }))
            }
        }
        var check = model
        check.sharing[set] = fit
        for trial in used {
            var rows: [Kernel: Int] = [:]
            for side in trial.sides { if let kernel = Kernel(side.kernel) { rows[kernel] = side.rows } }
            if let t = check.seconds(m: trial.shape[0], k: trial.shape[1], n: trial.shape[2], rows: rows) { residuals.append((t - trial.wall) / trial.wall) }
        }
        return SharingFit(sharing: fit, keeps: keeps, overheads: overheads, residuals: residuals)
    }

    /// **The chains a set's fit reads, and the error of its figure** (`Sharing.noise`). The passes from the second (pass 1 only balances
    /// on the lines; a set with no long k is read on its last), except where a shape was chained three times or more — the third chain
    /// is taken only when the first two disagree beyond what the map's noise allows (`needsAgain`) —: there, the median chain alone,
    /// each wall over the seconds the model fitted on the usual passes gives its rows (two passes balanced at different rows compare).
    ///
    /// Two chains are one degree of freedom: their gap is the estimate a single disturbed chain decides — the rounded pair's 13 % and
    /// the integers' rounded pair's 1.9 % in one map, 32 % in the next (7.01 against 5.32 ms at the same rows, the ANE side 7.0 ms inside
    /// the pair against 5.2). Three chains have a median one disturbance cannot move and a spread it cannot fake: σ = 1.4826 · MAD over
    /// the median (the median absolute deviation of the ratios, 1.4826 its factor to a normal's σ), and the median of n chains has the
    /// standard error 1.2533 · σ / √n (its efficiency against the mean, π/2 in variance). `noise` is that standard error, relative; `nil`
    /// where no shape was chained three times — two chains that agree within the map's noise carry the map's.
    package static func chains(set: String, trials: [Measurements.Trial], model: CapacityModel) -> (used: [Measurements.Trial], noise: Double?) {
        let mine = trials.filter { $0.set == set && $0.sides.count >= 2 && $0.shape.count == 3 && $0.wall > 0 }
        let passes = Set(mine.map(\.pass)).subtracting([Measurements.Trial.again])
        let from = passes.contains(3) ? 2 : (passes.max() ?? 0)
        var used = mine.filter { $0.pass >= from && $0.pass != Measurements.Trial.again }
        var reference = model
        if let f = overlap(set: set, trials: used, model: model) { reference.sharing[set] = f.sharing }
        var noise: Double?
        for (shape, group) in Dictionary(grouping: mine, by: \.shape) where group.count >= 3 {
            let ratios = group.compactMap { t -> (Measurements.Trial, Double)? in
                guard let r = ratio(t, model: reference) else { return nil }
                return (t, r)
            }.sorted { $0.1 < $1.1 }
            guard ratios.count >= 3 else { continue }
            let mid = ratios[ratios.count / 2]
            let m = median(ratios.map(\.1)), mad = median(ratios.map { abs($0.1 - m) })
            let se = 1.2533 * (1.4826 * mad / m) / Double(ratios.count).squareRoot()
            noise = max(noise ?? 0, se)
            used.removeAll { $0.shape == shape }
            used.append(mid.0)
        }
        return (used, noise)
    }

    /// A chain's wall over the seconds `model` gives its rows.
    static func ratio(_ trial: Measurements.Trial, model: CapacityModel) -> Double? {
        var rows: [Kernel: Int] = [:]
        for side in trial.sides { if let kernel = Kernel(side.kernel) { rows[kernel] = side.rows } }
        guard let t = model.seconds(m: trial.shape[0], k: trial.shape[1], n: trial.shape[2], rows: rows), t > 0 else { return nil }
        return trial.wall / t
    }

    /// **A figure's own σ, from the map's noise** (`CapacityModel.noise`): the median over the candidates of the slowest of three alternated
    /// rounds over the fastest, less one — a range of three, whose expectation is d₂(3) = 1.693 σ. `nil`: a map without rounds.
    package static func sigma(mapNoise: Double?) -> Double? { mapNoise.map { $0 / 1.693 } }

    /// **Whether a set's two chains at one shape call for a third** (`Measurements.Trial.again`): their ratios (wall over the model's
    /// seconds at their rows) apart by more than twice the σ of a difference of two figures, 2√2 σ — under it, the gap is the map's own
    /// noise; over it, one chain met what the other did not (a disturbance only adds time, and a pair's ANE side can run a third slower in
    /// one chain), and only a third tells which. Without the map's noise, always.
    package static func needsAgain(set: String, shape: [Int], trials: [Measurements.Trial], model: CapacityModel) -> Bool {
        let ratios = trials.filter { $0.set == set && $0.shape == shape && $0.sides.count >= 2 && $0.wall > 0 }.compactMap { ratio($0, model: model) }
        guard ratios.count == 2, let low = ratios.min(), let high = ratios.max() else { return false }
        guard let sigma = sigma(mapNoise: model.noise) else { return true }
        return high / low - 1 > 2 * 2.squareRoot() * sigma
    }

    /// **What a side of `set` pays per call beyond its line, per weight value** (`Sharing.perWeight`), from what the map measured before
    /// any chain: the rounded bf16 pair's ANE (`gpu/bf16act+ane/fp16`) reads the bf16 copy as published, and its program widens every value
    /// of it at every call (the lift reads each row widened, the conversion widens it again where the widened weight's is read as it is:
    /// one widening more than the line's weight, the shift that `preparation[bf16][widened]` times per value). `nil` for every other set:
    /// its sides read the weight their lines were measured on (the integers' pairs read the integers alone as in the set).
    package static func lift(set: String, model: CapacityModel) -> [CapacityModel.Engine: Double]? {
        guard set == CapacityModel.key([Kernel(.gpu, .bf16act), Kernel(.ane, .fp16)]),
              let widening = model.preparation[.bf16]?[.widened], widening > 0 else { return nil }
        return [.ane: widening]
    }

    /// **How far the map's identical calls disagreed** (`CapacityModel.noise`): for every candidate timed in alternated rounds on a wide product
    /// (no short side: a thin shape's first round pays what the next do not) or an attention, its slowest round over its fastest, less one;
    /// the median of those. `nil` with no such rounds (a report that predates them).
    package static func noise(_ trials: [Measurements.Solo]) -> Double? {
        var rounds: [String: [Double]] = [:]
        for t in trials where t.seconds > 0 && t.shape.count == 3 {
            guard let name = t.kernel.split(separator: " ").first, let kernel = Kernel(String(name)) else { continue }
            if kernel.form.isProduct, CapacityModel.isThin(rows: t.shape[0], k: t.shape[1], n: t.shape[2]) { continue }
            rounds["\(t.kernel) \(t.shape)", default: []].append(t.seconds)
        }
        let spreads = rounds.values.filter { $0.count >= 2 }.map { $0.max()! / $0.min()! - 1 }
        return spreads.isEmpty ? nil : median(spreads)
    }

    // MARK: - The model

    /// **The `CapacityModel` the measurements give** (`allocationUnit`: the rows an engine takes come in whole units of it, `NeuralEngine.rowUnit`): the lines, the ANE's unit, the sets' sharing (from the second pass
    /// when there is one, whose sides are balanced), the heat, the memory, the preparation.
    package static func model(from m: Measurements, allocationUnit: Int = 256) -> CapacityModel {
        var model = CapacityModel()
        var points: [Kernel: [Point]] = [:]
        for (name, text) in m.tiles ?? [:] {
            if let k = Kernel(name), let t = MatmulTile(text) { model.tiles[k] = t }
        }
        for (name, text) in m.thinTiles ?? [:] {
            if let k = Kernel(name), let t = MatmulTile(text) { model.thinTiles[k] = t }
        }
        for (name, text) in m.attentionTiles ?? [:] {
            if let k = Kernel(name), AttentionTile.kernels.contains(k), let t = AttentionTile(text) { model.attentionTiles[k] = t }
        }
        for s in m.solos {
            guard !m.wrong.contains(s.kernel), let k = Kernel(s.kernel), s.seconds > 0 else { continue }
            points[k, default: []].append(Point(flop: s.flop, seconds: s.seconds))
        }
        for (k, p) in points {
            if k.form.isProduct, let l = productLine(kernel: k, solos: m.solos.filter { $0.kernel == k.description && !m.wrong.contains($0.kernel) }) {
                model.lines[k] = l
            } else if let l = line(p) { model.lines[k] = l }
        }
        var thinGroups: [Kernel: [Measurements.Solo]] = [:]
        for (name, group) in Dictionary(grouping: (m.thin ?? []).filter({ !m.wrong.contains($0.kernel) && $0.seconds > 0 }), by: \.kernel) {
            if let kernel = Kernel(name) { thinGroups[kernel] = group }
        }
        model.thin = thinLines(thinGroups, lines: model.lines, bandwidth: m.memoryBandwidth)
        // The unit the ANE *charges* can be finer than the one an engine *allocates* (rows are given in whole
        // `allocationUnit`s): the model's rounding is the larger.
        if let st = stairs(m.stairs.map { ($0.rows, $0.seconds) }) { model.aneRowUnit = max(st.unit, allocationUnit) }
        // The preparation before the sets: a set's side may pay a widening its line does not hold (`lift(set:model:)`).
        for (kindName, paths) in m.preparation {
            if kindName == CapacityModel.kquantKind {
                model.kquantPreparation = paths[WeightPath.widened.rawValue].flatMap { $0 > 0 ? $0 : nil }
                model.kquantStaging = paths[WeightPath.native.rawValue].flatMap { $0 > 0 ? $0 : nil }
                continue
            }
            guard let kind = WeightKind(rawValue: kindName) else { continue }
            for (pathName, seconds) in paths { if let path = WeightPath(rawValue: pathName) { model.preparation[kind, default: [:]][path] = seconds } }
        }
        // The sets from the smallest: a set's κ is read through the sets that remain when its first side has ended.
        func size(_ set: String) -> Int { set.split(separator: "+").count }
        let order: [String] = Set(m.trials.map(\.set)).sorted { (a: String, b: String) -> Bool in size(a) != size(b) ? size(a) < size(b) : a < b }
        for set in order {
            // The passes from the second were balanced on a κ measured (the first, on the lines, only gives it); a set with two passes only
            // is read on the second.
            let chosen = chains(set: set, trials: m.trials, model: model)
            if let f = overlap(set: set, trials: chosen.used, model: model) {
                model.sharing[set] = f.sharing
                model.sharing[set]?.noise = chosen.noise
            }
        }
        for w in m.witnesses {
            if let e = CapacityModel.Engine(rawValue: w.engine), w.before > 0, w.after > 0 { model.heat[e] = max(1, w.after / w.before) }
        }
        model.noise = m.tileTrials.flatMap(noise)
        if let w = wake(m.wake) { model.wake = w.seconds; model.wakeIdle = w.idle }
        model.memoryBandwidth = m.memoryBandwidth
        model.diskBandwidth = m.diskBandwidth
        model.diskClasses = m.diskClasses
        return model
    }
}
