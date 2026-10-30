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
///
/// What sets it apart — what a fit written by analogy with "a line through the points" would get wrong:
///
///   - **a line is fitted on the relative error** (weights 1/t²): the 300 GFLOP point is a hundred times the 4 GFLOP one in
///     seconds, and an unweighted fit would be the big one's alone — the small one is where t₀ lives. A negative t₀ is a
///     fit that reached past its data: it is clamped to 0 and the rate refitted through the origin.
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

    package static func line(_ points: [Point]) -> CapacityModel.Line? {
        affine(points.map { ($0.flop, $0.seconds) }).map { .init(t0: $0.a, rate: 1 / $0.b) }
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
            /// 1: the rows the solo lines (no sharing) balance; 2: the rows the first pass's keep balances.
            package var pass: Int
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
        package var stairShape: [Int] = []
        package var stairs: [Stair] = []
        package var trials: [Trial] = []
        package var witnesses: [Witness] = []
        package var memoryBandwidth: Double?
        package var preparation: [RouteTable.WeightKind: [RouteTable.WeightPath: Double]] = [:]
        /// Kernels the judge refused (a wrong route fills nothing).
        package var wrong: [String] = []
        /// Seconds of each probe.
        package var budget: [String: Double] = [:]
        package init() {}
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
        let used = trials.filter { $0.set == set && (passes?.contains($0.pass) ?? true) }
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

    // MARK: - The model

    /// **The `CapacityModel` the measurements give** (`allocationUnit`: the rows an engine takes come in whole units of it, `NeuralEngine.rowUnit`): the lines, the ANE's unit, the sets' sharing (from the second pass
    /// when there is one, whose sides are balanced), the heat, the memory, the preparation.
    package static func model(from m: Measurements, allocationUnit: Int = 256) -> CapacityModel {
        var model = CapacityModel()
        var points: [Kernel: [Point]] = [:]
        for s in m.solos {
            guard !m.wrong.contains(s.kernel), let k = Kernel(s.kernel), s.seconds > 0 else { continue }
            points[k, default: []].append(Point(flop: s.flop, seconds: s.seconds))
        }
        for (k, p) in points { if let l = line(p) { model.lines[k] = l } }
        // The unit the ANE *charges* can be finer than the one an engine *allocates* (rows are given in whole
        // `allocationUnit`s): the model's rounding is the larger.
        if let st = stairs(m.stairs.map { ($0.rows, $0.seconds) }) { model.aneRowUnit = max(st.unit, allocationUnit) }
        for set in Set(m.trials.map(\.set)) {
            let passes = Set(m.trials.filter { $0.set == set }.map(\.pass))
            let last = passes.max().map { Set([$0]) }
            if let f = sharing(set: set, trials: m.trials, lines: model.lines, using: last) { model.sharing[set] = f.sharing }
        }
        for w in m.witnesses {
            if let e = CapacityModel.Engine(rawValue: w.engine), w.before > 0, w.after > 0 { model.heat[e] = max(1, w.after / w.before) }
        }
        model.memoryBandwidth = m.memoryBandwidth
        model.preparation = m.preparation
        return model
    }
}
