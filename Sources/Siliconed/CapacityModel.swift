import Foundation

/// **The machine as the capacity map knows it: a few numbers per engine, measured, from which the time of any block
/// is computed — instead of a table of routes timed on a grid.** Pure: no Metal, no clock. The probes fill it
/// (`CapacityProbes`), the solver reads it (`CapacitySolver`), the report carries it.
///
///     read (inventory)      crest[engine]           FLOP/s the silicon announces, nil where nothing says it
///     measured (the map)    lines[kernel]           t(W) = t₀ + W / rate          one kernel = an engine × what it reads
///                           aneRowUnit              the ANE's rows come in units: an odd unit costs a whole one
///                           sharing[engines]        each engine keeps `keep` of its own rate, the set pays `overhead` once
///                           heat[engine]            sustained over cold time (≥ 1): the fanless Mac's late minute
///                           memory, disk            bytes per second
///                           preparation             seconds per value to bring a weight to the form a kernel reads
///
///     a product m×k×n on engines E, rows rᵢ (Σ rᵢ = m), W = 2·k·n·m:
///         tᵢ = t₀ᵢ + 2·k·n·rᵢ / (rateᵢ · keepᵢ)          (rᵢ rounded up to `aneRowUnit` on the ANE)
///         T  = max tᵢ + overhead(E)
///     balanced (every side ends together), R′ᵢ = rateᵢ · keepᵢ:
///         T* = (W + Σ t₀ᵢ R′ᵢ) / Σ R′ᵢ          rᵢ = (T* − t₀ᵢ) · R′ᵢ / (2·k·n)
///         an engine whose rᵢ ≤ 0 leaves the set, and the rest is solved again
///
/// What sets it apart — what a model written by analogy with "add the engines' rates" would get wrong:
///
///   - **engines share, they do not add** (measured on nine Macs, six generations): GPU + AMX keep 0.74–0.88 of the sum of their solo
///     rates, GPU + ANE 0.49–0.72, the three 0.25–0.42. `keep` is per engine *within a set*: the GPU beside the ANE
///     is not the GPU beside the AMX. A set's time is never read off its members' solo lines.
///   - **the loss is two things the model keeps apart**: what does not depend on the size (`overhead`: launching,
///     synchronizing, a weight's preparation counted in the route — our code, a fix holds on every Mac) and what does
///     (`keep`: power and memory this Mac shares between its engines — physics, taken as measured). The loss is larger
///     at 1 024 rows than at 4 096: a single factor would hide which one it is.
///   - **the ANE is a staircase, not a curve**: its rows run in units of 256 (`NeuralEngine.rowUnit`), so 1 792 and
///     2 048 rows cost the same; a balance read off a smooth line lands on the edge of a step about half the time.
///     The shares are solved on the line, then the ANE's rows are tried at the units on both sides.
///   - **no chip name anywhere**: `crest` is the denominator of "reached / crest", read from the inventory; every
///     number the solver uses is measured on the Mac it runs on.
package struct CapacityModel: Codable, Sendable, Equatable {

    // MARK: - Kernels

    package enum Engine: String, Codable, Sendable, CaseIterable, Comparable {
        case gpu, amx, ane, cpu
        package static func < (a: Engine, b: Engine) -> Bool { allCases.firstIndex(of: a)! < allCases.firstIndex(of: b)! }
    }

    /// What a kernel reads (a product's weight as it reaches the engine) or which kernel it is (an attention's).
    package enum Form: String, Codable, Sendable, CaseIterable {
        /// Products: the weight widened to fp32; the bf16 weight as published, its activation in fp32 or rounded to
        /// bf16; the 8-bit weight as published; fp16 (the ANE's programs, from the widened weight).
        case fp32, bf16, bf16act, int8, q8, fp16
        /// Attentions: the MPSGraph graph, the flash kernel, a share of the heads on the CPU.
        case graph, flash, heads
    }

    /// One engine computing one form: `gpu/fp32`, `ane/fp16`, `cpu/heads`.
    package struct Kernel: Hashable, Codable, Sendable, CustomStringConvertible {
        package var engine: Engine
        package var form: Form
        package init(_ engine: Engine, _ form: Form) { self.engine = engine; self.form = form }
        package var description: String { "\(engine.rawValue)/\(form.rawValue)" }
        package init?(_ text: String) {
            let parts = text.split(separator: "/")
            guard parts.count == 2, let e = Engine(rawValue: String(parts[0])), let f = Form(rawValue: String(parts[1]))
            else { return nil }
            self.init(e, f)
        }
        package init(from decoder: any Decoder) throws {
            let text = try decoder.singleValueContainer().decode(String.self)
            guard let k = Kernel(text) else {
                throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "kernel « \(text) »"))
            }
            self = k
        }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.singleValueContainer(); try c.encode(description)
        }
    }

    /// `t(W) = t₀ + W / rate`: seconds for W FLOP. Affine holds a GPU alone within 1.7–2.9 % (median) from 4 to
    /// 825 GFLOP on 28 shapes; a constant rate misses by 6–7 %.
    package struct Line: Codable, Sendable, Equatable {
        package var t0: Double
        package var rate: Double
        package init(t0: Double, rate: Double) { self.t0 = t0; self.rate = rate }
        package func seconds(_ flop: Double) -> Double { t0 + flop / rate }
    }

    /// What engines lose when they run one block together: each keeps `keep[e]` of its solo rate, the set pays
    /// `overhead` seconds once.
    package struct Sharing: Codable, Sendable, Equatable {
        package var overhead: Double
        package var keep: [Engine: Double]
        package init(overhead: Double = 0, keep: [Engine: Double] = [:]) { self.overhead = overhead; self.keep = keep }
    }

    package var crest: [Engine: Double] = [:]
    package var lines: [Kernel: Line] = [:]
    package var aneRowUnit = 256
    /// Keyed by the set's engines in order, joined by `+` (`gpu+ane`): a dictionary keyed by a set does not survive JSON.
    package var sharing: [String: Sharing] = [:]
    package var heat: [Engine: Double] = [:]
    package var memoryBandwidth: Double?
    package var diskBandwidth: Double?
    /// Seconds per value to bring a weight of `kind` to the form a kernel reads (`RouteTable.WeightPath`).
    package var preparation: [RouteTable.WeightKind: [RouteTable.WeightPath: Double]] = [:]

    package init() {}

    package static func key(_ engines: [Engine]) -> String { engines.sorted().map(\.rawValue).joined(separator: "+") }
    package func sharing(_ kernels: [Kernel]) -> Sharing {
        kernels.count < 2 ? Sharing() : sharing[Self.key(kernels.map(\.engine))] ?? Sharing()
    }

    /// The rate a kernel keeps in a set, sustained (`heat`) when asked.
    func rate(_ k: Kernel, in set: Sharing, sustained: Bool) -> Double? {
        guard let line = lines[k], line.rate > 0 else { return nil }
        let keep = set.keep[k.engine] ?? 1
        let slow = sustained ? max(1, heat[k.engine] ?? 1) : 1
        return line.rate * keep / slow
    }

    // MARK: - A product

    /// One product's plan on a set of kernels: the rows each takes and the seconds.
    package struct Split: Sendable, Equatable {
        package var rows: [Kernel: Int]
        package var seconds: Double
    }

    /// **The time of `m×k×n` on `kernels` with these rows** (Σ = m), the ANE's rounded up to its unit.
    package func seconds(m: Int, k: Int, n: Int, rows: [Kernel: Int], sustained: Bool = false) -> Double? {
        let kernels = rows.keys.filter { rows[$0]! > 0 }
        guard !kernels.isEmpty else { return nil }
        let set = sharing(Array(kernels))
        var worst = 0.0
        for kernel in kernels {
            guard let r = rate(kernel, in: set, sustained: sustained), let t0 = lines[kernel]?.t0 else { return nil }
            var rowsHere = rows[kernel]!
            if kernel.engine == .ane { rowsHere = (rowsHere + aneRowUnit - 1) / aneRowUnit * aneRowUnit }
            worst = max(worst, t0 + 2 * Double(k) * Double(n) * Double(rowsHere) / r)
        }
        return worst + (kernels.count > 1 ? set.overhead : 0)
    }

    /// **The balanced split of `m×k×n` on `kernels`**: shares solved on the lines so that every side ends together,
    /// an engine with no positive share left out, the ANE's rows tried at the units on both sides of the balance.
    /// `nil` when a kernel has no line.
    package func split(m: Int, k: Int, n: Int, kernels: [Kernel], sustained: Bool = false) -> Split? {
        var active = kernels
        let perRow = 2 * Double(k) * Double(n)
        while !active.isEmpty {
            let set = sharing(active)
            var rates: [Kernel: Double] = [:]
            for kernel in active {
                guard let r = rate(kernel, in: set, sustained: sustained) else { return nil }
                rates[kernel] = r
            }
            let w = perRow * Double(m)
            let total = rates.values.reduce(0, +)
            let tStar = (w + active.reduce(0) { $0 + lines[$1]!.t0 * rates[$1]! }) / total
            let share = Dictionary(uniqueKeysWithValues: active.map { ($0, (tStar - lines[$0]!.t0) * rates[$0]! / perRow) })
            if let gone = active.first(where: { share[$0]! <= 0 }) {
                active.removeAll { $0 == gone }
                continue
            }
            if active.count == 1 {
                let only = active[0]
                return seconds(m: m, k: k, n: n, rows: [only: m], sustained: sustained).map { Split(rows: [only: m], seconds: $0) }
            }
            // Integer rows: the ANE's at the units around its share (or none: it may not pay its overhead), the rest
            // to the others by their shares.
            let ane = active.first { $0.engine == .ane }
            let others = active.filter { $0.engine != .ane }
            var aneOptions = [0]
            if let a = ane {
                let low = Int(share[a]!) / aneRowUnit * aneRowUnit
                aneOptions += [low, low + aneRowUnit].filter { $0 > 0 && $0 < m }
            }
            var candidates: [[Kernel: Int]] = []
            for aneRows in Set(aneOptions) {
                var rows: [Kernel: Int] = [:]
                if let a = ane { rows[a] = aneRows }
                let rest = m - aneRows
                var left = rest
                let othersShare = others.reduce(0) { $0 + share[$1]! }
                for (i, kernel) in others.enumerated() {
                    let r = i == others.count - 1 ? left : min(left, Int((Double(rest) * share[kernel]! / othersShare).rounded()))
                    rows[kernel] = r; left -= r
                }
                candidates.append(rows)
            }
            let timed = candidates.compactMap { rows in seconds(m: m, k: k, n: n, rows: rows, sustained: sustained).map { (rows, $0) } }
            guard let best = timed.min(by: { $0.1 < $1.1 }) else { return nil }
            return Split(rows: best.0.filter { $0.value > 0 }, seconds: best.1)
        }
        return nil
    }
}
