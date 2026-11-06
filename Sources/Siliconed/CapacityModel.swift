import Foundation

/// **The machine as the capacity map knows it: a few numbers per engine, measured, from which the time of any block
/// is computed — instead of a table of routes timed on a grid.** Pure: no Metal, no clock. The probes fill it
/// (`CapacityProbes`), the solver reads it (`CapacitySolver`), the report carries it.
///
///     read (inventory)      crest[engine]           FLOP/s the silicon announces, nil where nothing says it
///     measured (the map)    lines[kernel]           t(W) = t₀ + W / rate          one kernel = an engine × what it reads
///                           aneRowUnit              the ANE's rows come in units: an odd unit costs a whole one
///                           sharing[engines]        each engine keeps `keep` of its own rate, the set pays `overhead` once;
///                                                   a side may pay `perWeight`·k·n more inside the set than alone (`lift`)
///                           noise                   the map's own repeatability: under twice it, two plans are one
///                           heat[engine]            sustained over cold time (≥ 1): the fanless Mac's late minute
///                           memory, disk            bytes per second
///                           preparation             seconds per value to bring a weight to the form a kernel reads
///
///     a product m×k×n on engines E, rows rᵢ (Σ rᵢ = m), W = 2·k·n·m:
///         sᵢ = t₀ᵢ + wᵢ·k·n + 2·k·n·rᵢ / rateᵢ(k)       its seconds alone (rᵢ rounded up to `aneRowUnit` on the ANE)
///                                                      the ANE's rate carries the host's passes over the rows (`Line.perRow`):
///                                                      1/rate = 1/R + h·(k·r + n·parts) / (2·k·n)
///         the engines run **overlapped**: while the set S is active, each progresses at κᵢˢ of its speed alone, and at 1 once
///         it is alone; T = the end of the last + overhead(E)
///     balanced (every side ends together), κᵢ the full set's:
///         sᵢ = κᵢ · T*        T* = (W + Σ fᵢ Rᵢ) / Σ κᵢ Rᵢ        fᵢ = t₀ᵢ + wᵢ·k·n        rᵢ = (κᵢT* − fᵢ) · Rᵢ / (2·k·n)
///         an engine whose rᵢ ≤ 0 leaves the set, and the rest is solved again
///     inside a set whose side pays per call for the set's own operand (the ANE's fp16 program widening the GPU's bf16 copy:
///     `gpu/bf16act+ane/fp16`), fᵢ += λᵢ·k·n, λᵢ = `Sharing.perWeight[i]` — a cost in k·n, not in rows: it does not shrink
///     with the ANE's share, so at few rows (512² images: m ≈ 1 024) the ANE's balanced share falls to nothing and the pair
///     leaves the candidates, while at m ≈ 4 096 it is a few percent of the ANE's side
///
/// What sets it apart — what a model written by analogy with "add the engines' rates" would get wrong:
///
///   - **engines share, they do not add** (measured on nine Macs, six generations): GPU + AMX keep 0.74–0.88 of the sum of their solo
///     rates, GPU + ANE 0.49–0.72, the three 0.25–0.42. `keep` is per engine *within a set*: the GPU beside the ANE
///     is not the GPU beside the AMX. A set's time is never read off its members' solo lines.
///   - **the loss lasts as long as the partner runs**: an engine that ends first frees the others, which go on at their full
///     speed (a side's seconds past the balance are not its seconds inside the set). The `keep` of a side is its speed inside
///     the set over its speed alone — taken on the *line's* seconds at the rows the chain gave it, so that the line's own
///     bias at those rows is in the keep and not in the next product — and constant over the splits of one product
///     (GPU + ANE: 0.92 and 0.75 at 2 048, 2 304 and 2 560 ANE rows alike), which a `keep` read as "seconds inside over seconds
///     alone, all along" is not.
///   - **a kernel's seconds are not `t₀ + W / rate` on every shape**: the ANE lifts the whole weight to its input at every call
///     (`perWeight`, ∝ k·n: 2.6 of 15 ms at 3 840²), and the AMX's rate falls with k (`rateByK`: 2.3 TFLOP/s at 3 840, 1.8 at
///     12 288), where a line through three products of one family reads t₀ off a shape that is not the next one's.
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
        /// bf16; the 8-bit weight as published (`gpu/int8`; `ane/int8`: the ANE's program that takes the integers as its
        /// input, scale per column or for the tensor, the rotation on the activation); fp16 (the ANE's programs, from the
        /// widened weight); the 8-bit weight as published beside an activation rounded to bf16 (`gpu/int8act`: after its
        /// rotation, `bfloat × int8`, the scale after the sums; `gpu/q8act`: a GGUF Q8_0, each block's scale applied in fp32
        /// to its partial sums) — the integers are exact in bf16, so only the operand is rounded, as for `bf16act`; a K-quant
        /// (`gpu/q4kact`: Q4_K, Q5_K's `sc·q` exact in fp16, `bfloat × half` per super-block, its `d` in fp32 on the partial sums,
        /// the mins apart — `KQuantKernels`); the int8 weight as published beside the activation quantized to int8 (`gpu/int8x8`: W8A8,
        /// ComfyUI's `int8_linear` — after its rotation, per row, `int8 × int8 → int32`, `s_x · s_w` after the sums: `QuantizedKernels`).
        case fp32, bf16, bf16act, int8, q8, fp16, int8act, q8act, q4kact, int8x8
        /// Attentions: the MPSGraph graph, the flash kernel, a share of the heads on the CPU, the three stages by tiles of query rows
        /// (`tiles`: fp32 throughout; `tiles16`: fp16 operands, fp32 sums; `tilesbf16`: bf16 operands, fp32 sums) — the last two in
        /// the form their map chose (`attentionTiles`).
        case graph, flash, heads, tiles, tiles16, tilesbf16

        /// A product's form (its seconds a line of `k`, `n` and rows), not an attention's.
        package var isProduct: Bool {
            switch self {
            case .fp32, .bf16, .bf16act, .int8, .q8, .fp16, .int8act, .q8act, .q4kact, .int8x8: return true
            case .graph, .flash, .heads, .tiles, .tiles16, .tilesbf16: return false
            }
        }
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

    /// **A kernel's seconds on a product `m×k×n`** (rows `r` of it): `t₀ + perWeight·k·n + 2·r·k·n / rate(k)`. `t(W) = t₀ + W / rate`
    /// alone holds a GPU within 1.7–2.9 % (median) from 4 to 825 GFLOP on 28 shapes; the two terms added make the ANE (the lift of
    /// the weight at every call) and the AMX (the rate by k) fit as well.
    package struct Line: Codable, Sendable, Equatable {
        package var t0: Double
        package var rate: Double
        /// Seconds per weight value paid once per call, whatever the rows: the ANE's lift of the fp32 weight to its fp16 input.
        package var perWeight: Double?
        /// The rate by k (`RateAt`, ascending in k): linear in the seconds per FLOP against ln k between two knots, the nearest knot's
        /// beyond the ends. `nil`: `rate` at every k.
        package var rateByK: [RateAt]?
        /// **The line's domain in rows**: the fewest rows the kernel is asked for — half of the smallest the probes measured it at. Below it the line is an
        /// extrapolation, and a plan that relies on it is a guess (the AMX at 288 rows ran 11 % under the line fitted from 1 024 rows up, and its κ
        /// followed): the model gives no seconds there, and the solver offers no assignment that gives the kernel fewer. `nil`: no bound.
        package var rowsFrom: Int?
        /// **The ANE's host passes, per row**: seconds per element the host runs over for each row it hands the engine — the activation (`k`
        /// values, converted to fp16, and rotated first for a convrot weight: twice where K is cut in parts, once for the peak and once per part)
        /// and the output (`n` values a part, scaled and summed in fp32). `nil`: none, and the rate carries them. Thin products (n = 1 536,
        /// 3 072) spend more of their call there than their FLOP say: Krea 2's k and v, MOSS-Image's `3072×3072` ran 50–60 % over a line fitted
        /// on wide ones.
        package var perRow: Double?
        package struct RateAt: Codable, Sendable, Equatable {
            package var k: Int
            package var rate: Double
            package init(k: Int, rate: Double) { self.k = k; self.rate = rate }
        }
        package init(t0: Double, rate: Double, perWeight: Double? = nil, rateByK: [RateAt]? = nil, rowsFrom: Int? = nil, perRow: Double? = nil) {
            self.t0 = t0; self.rate = rate; self.perWeight = perWeight; self.rateByK = rateByK; self.rowsFrom = rowsFrom; self.perRow = perRow
        }
        package func seconds(_ flop: Double) -> Double { t0 + flop / rate }
        /// The seconds a call pays whatever its rows: `t₀` and the weight's lift.
        package func fixed(k: Int, n: Int) -> Double { t0 + (perWeight ?? 0) * Double(k) * Double(n) }
        /// The rate at `k`.
        package func rate(atK k: Int) -> Double {
            guard let knots = rateByK, let first = knots.first, let last = knots.last else { return rate }
            if k <= first.k { return first.rate }
            if k >= last.k { return last.rate }
            for (lo, hi) in zip(knots, knots.dropFirst()) where k <= hi.k {
                let f = (log(Double(k)) - log(Double(lo.k))) / (log(Double(hi.k)) - log(Double(lo.k)))
                return 1 / ((1 - f) / lo.rate + f / hi.rate)
            }
            return last.rate
        }
        /// The rate at `k` with the host's passes added to the engine's time: `1 / rate` plus the passes' seconds per FLOP (`parts`: the ANE's cut of
        /// K, `rotated`: the activation is rotated before it, as a convrot weight's is).
        package func rate(atK k: Int, n: Int, parts: Int = 1, rotated: Bool = false) -> Double {
            guard let perRow else { return rate(atK: k) }
            let elements = Double(k) * (rotated && parts > 1 ? 2 : 1) + Double(n) * Double(parts)
            return 1 / (1 / rate(atK: k) + perRow * elements / (2 * Double(k) * Double(n)))
        }
        /// Seconds of `rows` rows of a `k×n` weight (the caller rounds the ANE's up to its unit).
        package func seconds(rows: Int, k: Int, n: Int, parts: Int = 1, rotated: Bool = false) -> Double {
            fixed(k: k, n: n) + 2 * Double(rows) * Double(k) * Double(n) / rate(atK: k, n: n, parts: parts, rotated: rotated)
        }
    }

    /// What engines lose when they run one block together: while the set is active each keeps `keep[e]` of its speed alone (κ), the
    /// set pays `overhead` seconds once. `sustained`: the set's wall once it has run for tens of seconds over its wall in a chain of
    /// half a second (≥ 1; nil is 1) — what the minute cannot read (the SoC's power settles after ≈ 10 s), carried so that a plan's
    /// prediction can be asked at the regime of a render, and corrected by the render's own loop.
    package struct Sharing: Codable, Sendable, Equatable {
        package var overhead: Double
        package var keep: [Engine: Double]
        package var sustained: Double?
        /// κ measured at other k's, ascending (`KeepAt`): what the weight's length does to the loss — the AMX beside the GPU keeps 0.9 at
        /// k = 3 840 and half of that at 12 288, where both stream a weight of 200 MB. Between two knots the κ is linear in ln k, beyond the
        /// ends the nearest knot's; `keep` is the value for an engine no knot gives.
        package var keepByK: [KeepAt]?
        /// **The set's domain in k**: half of the smallest k and twice the largest the chains came at. Beyond it the κ would be an extrapolation (the
        /// triple's AMX keeps 0.73 at k = 3 840 and a third of it at 12 288: a κ read at one k and clamped over the next class chose a set that ran 20 %
        /// over its seconds), so the model gives no seconds to the set there. `nil`: no bound — the attention's pair has no ceiling
        /// (`CapacityFit.overlap`: it streams no weight, and no probe reaches the sequences of the large formats).
        package var kFrom: Int?, kTo: Int?
        /// **Seconds per weight value a side pays at every call in this set, beyond its own line** (keyed by engine): what the side's line,
        /// measured on another form of the weight, does not hold. The ANE's fp16 program beside the GPU's rounded rows reads the bf16 copy
        /// as published (`gpu/bf16act+ane/fp16`): every call widens the whole copy once more before its lift and conversion than the
        /// widened weight its line was measured on — `k·n` values whatever its rows, the bf16 widening's own rate (`preparation`). Read
        /// into the κ of a chain of 1 024 ANE rows it cost each of them a quarter of a call; at 256 rows (a 512² render's half grid) the
        /// same κ hid four times as much: the pair taken there ran 22–52 % over the GPU alone. `nil`: none.
        package var perWeight: [Engine: Double]?
        /// **The standard error of the set's figure** (relative), where its chains at one shape were measured three times because the
        /// first two disagreed beyond the map's noise (`CapacityFit.needsAgain`): 1.2533 · 1.4826 · MAD / median / √n over the chains'
        /// ratios to the model (`CapacityFit.chains`). The map's noise is read on single engines in alternated rounds; a pair's chains,
        /// two engines and the CPU's host passes beside them, can disagree more — the rounded pair 13 % in one map, the integers' rounded
        /// pair 1.9 % in one and 32 % in the next at the same rows. The solver keeps the set only where its gain passes twice the error of
        /// the difference (`CapacitySolver`'s `passingTheirError`). `nil`: two chains that agreed, a figure as sure as the map's.
        package var noise: Double?
        package struct KeepAt: Codable, Sendable, Equatable {
            package var k: Int
            /// Keyed by the engine's name (`"gpu"`).
            package var keep: [String: Double]
            package init(k: Int, keep: [String: Double]) { self.k = k; self.keep = keep }
        }
        package init(overhead: Double = 0, keep: [Engine: Double] = [:], sustained: Double? = nil, keepByK: [KeepAt]? = nil, kFrom: Int? = nil, kTo: Int? = nil,
                     perWeight: [Engine: Double]? = nil, noise: Double? = nil) {
            self.overhead = overhead; self.keep = keep; self.sustained = sustained; self.keepByK = keepByK; self.kFrom = kFrom; self.kTo = kTo
            self.perWeight = perWeight; self.noise = noise
        }
        /// The seconds `engine` pays per call in this set beyond its line, on a `k×n` weight (`perWeight`).
        package func lift(_ engine: Engine, k: Int, n: Int) -> Double { (perWeight?[engine] ?? 0) * Double(k) * Double(n) }
        /// Whether a weight of length `k` is inside the set's domain.
        package func covers(k: Int) -> Bool { k >= (kFrom ?? 0) && k <= (kTo ?? Int.max) }
        /// The κ of `engine` in this set on a weight of length `k`.
        package func keep(_ engine: Engine, k: Int) -> Double {
            let points = (keepByK ?? []).compactMap { p in p.keep[engine.rawValue].map { (k: p.k, keep: $0) } }
            guard let first = points.first, let last = points.last else { return keep[engine] ?? 1 }
            if k <= first.k { return first.keep }
            if k >= last.k { return last.keep }
            for (lo, hi) in zip(points, points.dropFirst()) where k <= hi.k {
                let f = (log(Double(k)) - log(Double(lo.k))) / (log(Double(hi.k)) - log(Double(lo.k)))
                return lo.keep * (1 - f) + hi.keep * f
            }
            return last.keep
        }

        private enum CodingKeys: String, CodingKey { case overhead, keep, sustained, keepByK, kFrom, kTo, perWeight, noise }
        /// `keep` keyed by the engine's name (`{"gpu": 0.95}`), as a person reads it.
        package init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            overhead = try c.decode(Double.self, forKey: .overhead)
            keep = try CapacityModel.engines(try c.decode([String: Double].self, forKey: .keep), codingPath: c.codingPath + [CodingKeys.keep])
            sustained = try c.decodeIfPresent(Double.self, forKey: .sustained)
            keepByK = try c.decodeIfPresent([KeepAt].self, forKey: .keepByK)
            kFrom = try c.decodeIfPresent(Int.self, forKey: .kFrom)
            kTo = try c.decodeIfPresent(Int.self, forKey: .kTo)
            perWeight = try c.decodeIfPresent([String: Double].self, forKey: .perWeight)
                .map { try CapacityModel.engines($0, codingPath: c.codingPath + [CodingKeys.perWeight]) }
            noise = try c.decodeIfPresent(Double.self, forKey: .noise)
        }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(overhead, forKey: .overhead)
            try c.encode(Dictionary(uniqueKeysWithValues: keep.map { ($0.key.rawValue, $0.value) }), forKey: .keep)
            try c.encodeIfPresent(sustained, forKey: .sustained)
            try c.encodeIfPresent(keepByK, forKey: .keepByK)
            try c.encodeIfPresent(kFrom, forKey: .kFrom)
            try c.encodeIfPresent(kTo, forKey: .kTo)
            try c.encodeIfPresent(perWeight.map { Dictionary(uniqueKeysWithValues: $0.map { ($0.key.rawValue, $0.value) }) }, forKey: .perWeight)
            try c.encodeIfPresent(noise, forKey: .noise)
        }
    }

    package var crest: [Engine: Double] = [:]
    package var lines: [Kernel: Line] = [:]
    /// **The lines of the products with a short side** (`thinSide`): a kernel's seconds on a product of at most `thinSide` rows, or whose `k` or `n`
    /// is, are not the big products' line extrapolated — a LoRA's `m × 3072 · 128`, a prompt of 19 rows through a 4096 × 12288 weight. They stream
    /// the activation and the output (or the whole weight) more than they compute: `t = t₀ + w·k·n + h·M·(k + n) + 2·M·k·n / R`, `M` the rows computed, `R` the big
    /// products' rate (the engine's ceiling), the streaming terms the forms' bytes at the memory's bandwidth, scaled by what the probes of that family of shapes measured (`CapacityFit.thinLine`), the rows those of the tile (`computedRows`). A kernel
    /// without one is timed by its `lines` everywhere, as before.
    package var thin: [Kernel: Line] = [:]
    /// The longest short side the `thin` lines speak for.
    package static let thinSide = 256
    package var aneRowUnit = 256
    /// Keyed by the set's engines in order, joined by `+` (`gpu+ane`), or by its kernels (`gpu/int8+ane/int8`) where the form
    /// changes the loss: a dictionary keyed by a set does not survive JSON.
    package var sharing: [String: Sharing] = [:]
    package var heat: [Engine: Double] = [:]
    package var memoryBandwidth: Double?
    package var diskBandwidth: Double?
    /// The same read at each class of service of the stream's threads, by class name (`"background"`, `"utility"`, `"default"`):
    /// the fastest pass and the slowest (`CapacityDisk.Reading`). Empty on a report that did not measure them.
    package var diskClasses: [String: CapacityDisk.Reading] = [:]
    /// The readings by class, for `CapacityDisk.choose`.
    package var diskReadings: [StreamClass: CapacityDisk.Reading] {
        Dictionary(uniqueKeysWithValues: diskClasses.compactMap { name, r in StreamClass(rawValue: name).map { ($0, r) } })
    }
    /// Seconds per value to bring a weight of `kind` to the form a kernel reads (`WeightPath`).
    package var preparation: [WeightKind: [WeightPath: Double]] = [:]
    /// Seconds per value to widen a K-quant (Q4_K to Q6_K) weight: it has no `WeightKind`, so `preparation` cannot say it,
    /// and a super-block's unpacking costs more than a bf16 shift. `nil`: the solver falls back on `preparation[.bf16]`.
    package var kquantPreparation: Double?
    /// Seconds per value to copy a K-quant's blocks as they are into the reserve (`Artifact.stagePacked`, the `gpu/q4kact` route: the GPU
    /// unpacks them). `nil`: a map before the route, priced as a Q8_0's copy.
    package var kquantStaging: Double?
    /// The name the probes' preparation gives the K-quant's two figures (`Fit.Measurements.preparation`), beside the `WeightKind`s.
    package static let kquantKind = "q4k"
    /// **The `matmul2d` tile each kernel runs** (`MatmulTile`): the fastest of its candidates on this Mac, measured before its line (which is
    /// the line of that tile). A kernel absent runs its standard tile — what every map before tiles were measured planned with.
    package var tiles: [Kernel: MatmulTile] = [:]
    /// ... and on its thin products (`isThin`), chosen on `Sizes.thin`: the tile that times its `thin` line.
    package var thinTiles: [Kernel: MatmulTile] = [:]
    /// **The form of each half-precision attention kernel** (`AttentionTile`, `gpu/tiles16` and `gpu/tilesbf16`): fused at this tile,
    /// the fastest of the candidates on this Mac, measured before its line (which is the line of that form). A kernel absent runs the
    /// three stages — what every map before the fused form planned with.
    package var attentionTiles: [Kernel: AttentionTile] = [:]
    /// **How far this map's identical calls disagreed** (relative): the median, over every candidate the map timed in alternated rounds on
    /// a wide product or an attention (`tileTrials`), of its slowest round over its fastest, less one (`CapacityFit.noise`). The solver
    /// reads it as the width under which two assignments are one (`CapacitySolver`: twice it, the difference of two such figures) and
    /// takes the first in its written order there — a choice the noise cannot flip from one map to the next. `nil`: a map without
    /// rounds, whose choices are the fastest's (a part in a billion).
    package var noise: Double?
    /// **What a self-attention pays beyond its line, after the CPU left the GPU idle** (seconds per call, `Context.wake`): the CPU's QK-norm
    /// and RoPE run before every self-attention of the five denoisers, and the GPU's governor makes the call after an idle past `wakeIdle`
    /// pay its clock back. `nil`: a map without the probe, priced as before.
    package var wake: Double?
    /// The idle from which the wake is paid (seconds): the report's.
    package var wakeIdle: Double?
    /// **The QK-norm and RoPE before a self-attention, on either side** (`Context.normRope`), each a line in the values it runs over (Q's and
    /// K's, `S · (H + H_kv) · Dh`): what the solver weighs against each other, the CPU's pass paying the wake besides. `nil`: a map without
    /// the probe — the CPU's pass, priced as before (at nothing but the wake).
    package var normRope: NormRope?

    /// The two routes of the QK-norm and RoPE, `t = t₀ + values / rate`.
    package struct NormRope: Codable, Sendable, Equatable {
        /// The CPU's pass (`Ops.rmsNormRope`), its wall.
        package var cpu: Line
        /// The GPU's kernel queued before the attention (`QKNormRope`, `wait: false`): the GPU's own seconds, what the chain adds.
        package var gpu: Line
        /// The same kernel waited for — what a cut by heads pays, its CPU heads reading Q and K after it: its wall, the completion included.
        package var gpuWaited: Line
        package init(cpu: Line, gpu: Line, gpuWaited: Line) { self.cpu = cpu; self.gpu = gpu; self.gpuWaited = gpuWaited }
    }

    /// **The decoders' 3×3 convolution, on fp32 operands and on bf16 ones** (`Context.convolution`): MPSGraph's, each a line in its
    /// FLOP. What the plan reads to run a decoder whose weights are published in bf16 in bf16 (`CapacityPlan.Resources.halfDecoder`).
    /// `nil`: a map without the probe, or whose bf16 convolution the judge refused — the decoders in fp32, as before.
    package var convolution: Convolution?

    /// The two lines of the decoders' convolution, `t = t₀ + FLOP / rate`, and the work they are compared at.
    package struct Convolution: Codable, Sendable, Equatable {
        package var fp32: Line
        package var bf16: Line
        /// The FLOP of the probe's largest convolution: a decoder's convolutions are of that order (a band's), never thin.
        package var work: Double
        package init(fp32: Line, bf16: Line, work: Double) { self.fp32 = fp32; self.bf16 = bf16; self.work = work }
        /// **bf16 is the faster at the decoders' work**: what the plan takes, where half precision is allowed.
        package var prefersHalf: Bool {
            func seconds(_ l: Line) -> Double { l.rate > 0 ? l.t0 + work / l.rate : .infinity }
            return seconds(bf16) < seconds(fp32)
        }
    }

    package init() {}

    // MARK: - JSON

    private enum CodingKeys: String, CodingKey {
        case crest, lines, thin, aneRowUnit, sharing, heat, memoryBandwidth, diskBandwidth, diskClasses, preparation, kquantPreparation, kquantStaging, tiles, thinTiles, attentionTiles, noise, wake, wakeIdle, normRope, convolution
    }

    /// The engines of a dictionary keyed by their names.
    fileprivate static func engines(_ named: [String: Double], codingPath: [any CodingKey]) throws -> [Engine: Double] {
        var out: [Engine: Double] = [:]
        for (name, value) in named {
            guard let e = Engine(rawValue: name) else {
                throw DecodingError.dataCorrupted(.init(codingPath: codingPath, debugDescription: "engine « \(name) »"))
            }
            out[e] = value
        }
        return out
    }

    /// **Written as a person reads it**: every dictionary keyed by the name of what it holds (`"gpu/fp32": {"rate": …, "t0": …}`,
    /// `"gpu+ane": {"keep": {"gpu": 0.87, "ane": 0.7}, "overhead": …}`) — a default encoding of a dictionary keyed by an enum is
    /// a flat array of alternating keys and values, which a shared report would not be.
    package init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        crest = try Self.engines(try c.decodeIfPresent([String: Double].self, forKey: .crest) ?? [:], codingPath: c.codingPath + [CodingKeys.crest])
        let named = try c.decodeIfPresent([String: Line].self, forKey: .lines) ?? [:]
        for (name, line) in named {
            guard let kernel = Kernel(name) else {
                throw DecodingError.dataCorruptedError(forKey: .lines, in: c, debugDescription: "kernel « \(name) »")
            }
            lines[kernel] = line
        }
        for (name, line) in try c.decodeIfPresent([String: Line].self, forKey: .thin) ?? [:] {
            guard let kernel = Kernel(name) else {
                throw DecodingError.dataCorruptedError(forKey: .thin, in: c, debugDescription: "kernel « \(name) »")
            }
            thin[kernel] = line
        }
        aneRowUnit = try c.decodeIfPresent(Int.self, forKey: .aneRowUnit) ?? 256
        sharing = try c.decodeIfPresent([String: Sharing].self, forKey: .sharing) ?? [:]
        heat = try Self.engines(try c.decodeIfPresent([String: Double].self, forKey: .heat) ?? [:], codingPath: c.codingPath + [CodingKeys.heat])
        memoryBandwidth = try c.decodeIfPresent(Double.self, forKey: .memoryBandwidth)
        diskBandwidth = try c.decodeIfPresent(Double.self, forKey: .diskBandwidth)
        diskClasses = try c.decodeIfPresent([String: CapacityDisk.Reading].self, forKey: .diskClasses) ?? [:]
        wake = try c.decodeIfPresent(Double.self, forKey: .wake)
        wakeIdle = try c.decodeIfPresent(Double.self, forKey: .wakeIdle)
        normRope = try c.decodeIfPresent(NormRope.self, forKey: .normRope)
        convolution = try c.decodeIfPresent(Convolution.self, forKey: .convolution)
        let prepared = try c.decodeIfPresent([String: [String: Double]].self, forKey: .preparation) ?? [:]
        for (kindName, paths) in prepared {
            guard let kind = WeightKind(rawValue: kindName) else {
                throw DecodingError.dataCorruptedError(forKey: .preparation, in: c, debugDescription: "weight kind « \(kindName) »")
            }
            for (pathName, seconds) in paths {
                guard let path = WeightPath(rawValue: pathName) else {
                    throw DecodingError.dataCorruptedError(forKey: .preparation, in: c, debugDescription: "weight path « \(pathName) »")
                }
                preparation[kind, default: [:]][path] = seconds
            }
        }
        kquantPreparation = try c.decodeIfPresent(Double.self, forKey: .kquantPreparation)
        kquantStaging = try c.decodeIfPresent(Double.self, forKey: .kquantStaging)
        for (name, tile) in try c.decodeIfPresent([String: MatmulTile].self, forKey: .tiles) ?? [:] {
            guard let kernel = Kernel(name) else {
                throw DecodingError.dataCorruptedError(forKey: .tiles, in: c, debugDescription: "kernel « \(name) »")
            }
            tiles[kernel] = tile
        }
        for (name, tile) in try c.decodeIfPresent([String: MatmulTile].self, forKey: .thinTiles) ?? [:] {
            guard let kernel = Kernel(name) else {
                throw DecodingError.dataCorruptedError(forKey: .thinTiles, in: c, debugDescription: "kernel « \(name) »")
            }
            thinTiles[kernel] = tile
        }
        for (name, tile) in try c.decodeIfPresent([String: AttentionTile].self, forKey: .attentionTiles) ?? [:] {
            guard let kernel = Kernel(name) else {
                throw DecodingError.dataCorruptedError(forKey: .attentionTiles, in: c, debugDescription: "kernel « \(name) »")
            }
            attentionTiles[kernel] = tile
        }
        noise = try c.decodeIfPresent(Double.self, forKey: .noise)
    }

    package func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(Dictionary(uniqueKeysWithValues: crest.map { ($0.key.rawValue, $0.value) }), forKey: .crest)
        try c.encode(Dictionary(uniqueKeysWithValues: lines.map { ($0.key.description, $0.value) }), forKey: .lines)
        if !thin.isEmpty { try c.encode(Dictionary(uniqueKeysWithValues: thin.map { ($0.key.description, $0.value) }), forKey: .thin) }
        try c.encode(aneRowUnit, forKey: .aneRowUnit)
        try c.encode(sharing, forKey: .sharing)
        try c.encode(Dictionary(uniqueKeysWithValues: heat.map { ($0.key.rawValue, $0.value) }), forKey: .heat)
        try c.encodeIfPresent(memoryBandwidth, forKey: .memoryBandwidth)
        try c.encodeIfPresent(diskBandwidth, forKey: .diskBandwidth)
        if !diskClasses.isEmpty { try c.encode(diskClasses, forKey: .diskClasses) }
        try c.encodeIfPresent(wake, forKey: .wake)
        try c.encodeIfPresent(wakeIdle, forKey: .wakeIdle)
        try c.encodeIfPresent(normRope, forKey: .normRope)
        try c.encodeIfPresent(convolution, forKey: .convolution)
        try c.encode(Dictionary(uniqueKeysWithValues: preparation.map { kind, paths in
            (kind.rawValue, Dictionary(uniqueKeysWithValues: paths.map { ($0.key.rawValue, $0.value) }))
        }), forKey: .preparation)
        try c.encodeIfPresent(kquantPreparation, forKey: .kquantPreparation)
        try c.encodeIfPresent(kquantStaging, forKey: .kquantStaging)
        if !tiles.isEmpty {
            try c.encode(Dictionary(uniqueKeysWithValues: tiles.map { ($0.key.description, $0.value) }), forKey: .tiles)
        }
        if !thinTiles.isEmpty {
            try c.encode(Dictionary(uniqueKeysWithValues: thinTiles.map { ($0.key.description, $0.value) }), forKey: .thinTiles)
        }
        if !attentionTiles.isEmpty {
            try c.encode(Dictionary(uniqueKeysWithValues: attentionTiles.map { ($0.key.description, $0.value) }), forKey: .attentionTiles)
        }
        try c.encodeIfPresent(noise, forKey: .noise)
    }

    /// **Every number to `digits` significant digits**: what a report keeps (`Diagnostic.significant`), so that a model read back
    /// from a report is the one the engine planned with — the plan's fingerprint hashes the numbers, and two readings of one
    /// report must be one plan.
    package func rounded(_ digits: Int = 6) -> CapacityModel {
        func r(_ x: Double) -> Double { Diagnostic.significant(x, digits) }
        var m = self
        m.crest = crest.mapValues(r)
        m.lines = lines.mapValues { Line(t0: r($0.t0), rate: r($0.rate), perWeight: $0.perWeight.map(r), rateByK: $0.rateByK?.map { .init(k: $0.k, rate: r($0.rate)) }, rowsFrom: $0.rowsFrom, perRow: $0.perRow.map(r)) }
        m.thin = thin.mapValues { Line(t0: r($0.t0), rate: r($0.rate), perWeight: $0.perWeight.map(r), rateByK: $0.rateByK?.map { .init(k: $0.k, rate: r($0.rate)) }, rowsFrom: $0.rowsFrom, perRow: $0.perRow.map(r)) }
        m.sharing = sharing.mapValues { Sharing(overhead: r($0.overhead), keep: $0.keep.mapValues(r), sustained: $0.sustained.map(r), keepByK: $0.keepByK?.map { .init(k: $0.k, keep: $0.keep.mapValues(r)) }, kFrom: $0.kFrom, kTo: $0.kTo, perWeight: $0.perWeight?.mapValues(r), noise: $0.noise.map(r)) }
        m.heat = heat.mapValues(r)
        m.memoryBandwidth = memoryBandwidth.map(r); m.diskBandwidth = diskBandwidth.map(r)
        m.diskClasses = diskClasses.mapValues { .init(rate: r($0.rate), mean: r($0.mean)) }
        m.preparation = preparation.mapValues { $0.mapValues(r) }
        m.kquantPreparation = kquantPreparation.map(r)
        m.kquantStaging = kquantStaging.map(r)
        m.noise = noise.map(r)
        m.wake = wake.map(r); m.wakeIdle = wakeIdle.map(r)
        func line(_ l: Line) -> Line { Line(t0: r(l.t0), rate: r(l.rate)) }
        m.normRope = normRope.map { NormRope(cpu: line($0.cpu), gpu: line($0.gpu), gpuWaited: line($0.gpuWaited)) }
        m.convolution = convolution.map { Convolution(fp32: line($0.fp32), bf16: line($0.bf16), work: r($0.work)) }
        return m
    }

    package static func key(_ engines: [Engine]) -> String { engines.sorted().map(\.rawValue).joined(separator: "+") }
    /// **Whether a set of engines was measured**: an engine alone always; two or more only with a sharing fitted from their probes. The
    /// solver offers no set that was not (`CapacitySolver`): taken at keep 1 it was the best plan, and cost a render a quarter more.
    package func measures(_ engines: [Engine]) -> Bool { engines.count < 2 || sharing[Self.key(engines)] != nil }

    /// **The key of a set of kernels** (`gpu/int8+ane/int8`): a set whose kernels read another form than the pair's usual
    /// (`gpu/fp32`, `ane/fp16`) keeps its own sharing, measured on its own trials — the ANE reading the integers costs the GPU
    /// beside it what the fp16 program does not, and the other way round.
    package static func key(_ kernels: [Kernel]) -> String { kernels.sorted { $0.engine < $1.engine }.map(\.description).joined(separator: "+") }
    /// The set's own sharing when the probes measured one (`key(kernels)`), else its engines' (`key(engines)`).
    package func sharing(_ kernels: [Kernel]) -> Sharing {
        kernels.count < 2 ? Sharing()
            : sharing[Self.key(kernels)] ?? sharing[Self.key(kernels.map(\.engine))] ?? Sharing()
    }

    // MARK: - A product

    /// One product's plan on a set of kernels: the rows each takes and the seconds.
    package struct Split: Sendable, Equatable {
        package var rows: [Kernel: Int]
        package var seconds: Double
    }

    /// **The rate of `kernel` on a `k×n` weight** with the host's passes in it (`Line.perRow`): the ANE's parts of K are the engine's own cut
    /// (`NeuralEngine.parts`), its int8 program reading convrot groups of 256 and rotating the activation.
    package func rate(_ kernel: Kernel, line: Line, k: Int, n: Int, short: Bool = false) -> Double {
        // A thin line's host term is the activation and the output streamed once (`k + n` per row), whatever the engine.
        if short { return line.rate(atK: k, n: n) }
        guard line.perRow != nil, kernel.engine == .ane else { return line.rate(atK: k) }
        let int8 = kernel.form == .int8
        return line.rate(atK: k, n: n, parts: NeuralEngine.parts(k: k, n: n, int8: int8, alignment: int8 ? Widen.largestRotationGroup : 64), rotated: int8)
    }

    /// **A kernel's seconds alone on `rows` rows of a `k×n` weight** (the ANE's rounded up to its unit), at the sustained rate
    /// (`heat`, on the rate: the lift of the weight and the launch do not slow) when asked. `nil` where the kernel has no line.
    package func alone(_ kernel: Kernel, rows: Int, k: Int, n: Int, sustained: Bool = false) -> Double? {
        guard let (line, short) = line(for: kernel, rows: rows, k: k, n: n), line.rate > 0, rows > 0, rows >= (line.rowsFrom ?? 0) else { return nil }
        var here = rows
        if kernel.engine == .ane { here = (here + aneRowUnit - 1) / aneRowUnit * aneRowUnit }
        if short { here = Self.computedRows(kernel, rows: here) }
        let slow = sustained ? max(1, heat[kernel.engine] ?? 1) : 1
        return line.fixed(k: k, n: n) + 2 * Double(here) * Double(k) * Double(n) / (rate(kernel, line: line, k: k, n: n, short: short) / slow)
    }

    /// **The rows a kernel computes when it is given `rows`**: the GPU's work in tiles of `GEMM.rowQuantum` rows (MPS's, and the 64-row tile of
    /// `matmul2d`), so a prompt of 19 rows costs 64 — the figure a short product's seconds are made of (a 19-row product through a 4 096 × 12 288
    /// int8 weight took 3.1 ms, the 64-row arithmetic 1.9 and the weight's pass the rest, against 0.8 for 19 rows' arithmetic). The AMX and the
    /// ANE (whose unit `alone` applies) compute what they are given.
    package static func computedRows(_ kernel: Kernel, rows: Int) -> Int {
        guard kernel.engine == .gpu, rows > 0 else { return rows }
        return (rows + GEMM.rowQuantum - 1) / GEMM.rowQuantum * GEMM.rowQuantum
    }

    /// **The bytes a form moves, an element each**: `(weight, activation)` — the weight as the kernel reads it (fp32, bf16 as published, int8 with its
    /// scales a rounding error beside it, Q8_0's fp32 scale per block of 64 rows of K), the activation in and the output out (fp32 each)
    /// beside the passes the form adds: the int8 route rotates the activation and scales the output (a read and a write of fp32 each), a rounded
    /// activation is read and written once before the product (half of that on each side of `k + n`).
    package static func streamedBytes(_ form: Form) -> (weight: Double, activation: Double) {
        switch form {
        case .bf16, .fp16: return (2, 4)
        case .bf16act: return (2, 8)
        case .int8: return (1, 12)
        case .q8: return (1 + 4.0 / 64, 4)
        // The rounded copy of the activation besides (read once in fp32, written and read in bf16).
        case .int8act: return (1, 16)
        case .q8act: return (1 + 4.0 / 64, 8)
        // The blocks (0.56 B), unpacked into `sc·q` (half: written, then read) and the mins; the activation's rounded copy and sums.
        case .q4kact: return (0.5625 + 4 + 4.0 / 32, 9)
        // The activation read once in fp32 and written in int8, read again in int8; the sums written in int32, then read and written
        // in fp32 by the scales' pass.
        case .int8x8: return (1, 18)
        default: return (4, 4)
        }
    }

    /// Whether a product of `rows` rows of a `k×n` weight has a short side: what the `thin` lines speak for.
    package static func isThin(rows: Int, k: Int, n: Int) -> Bool { min(rows, k, n) <= thinSide }

    /// **The line that times `kernel` on this product**: its `thin` line where the product has a short side and the probes measured one, else
    /// its `lines`. `short`: it is the thin one.
    package func line(for kernel: Kernel, rows: Int, k: Int, n: Int) -> (Line, short: Bool)? {
        if Self.isThin(rows: rows, k: k, n: n), let t = thin[kernel], t.rate > 0 { return (t, true) }
        return lines[kernel].map { ($0, false) }
    }

    /// The speed of a kernel inside the set that is active, over its speed alone.
    private func speed(_ kernel: Kernel, in set: Sharing, k: Int) -> Double { max(0.05, set.keep(kernel.engine, k: k)) }

    /// **The overlap: when each kernel ends**, from its seconds alone. While the set of kernels still running is `S`, each goes at
    /// the speed `sharing(S)` keeps it; the first to end frees the others, which go on as the smaller set they now are, and the last
    /// at its full speed. The ends are the sides' own clocks.
    package func overlap(_ solo: [Kernel: Double], k: Int) -> [Kernel: Double] {
        var remaining = solo.filter { $0.value > 0 }
        var ends: [Kernel: Double] = [:]
        var now = 0.0
        while remaining.count > 1 {
            let set = sharing(Array(remaining.keys))
            let (first, step) = remaining.map { ($0.key, $0.value / speed($0.key, in: set, k: k)) }.min { $0.1 < $1.1 }!
            now += step
            for (kernel, left) in remaining {
                if kernel == first { ends[kernel] = now; remaining[kernel] = nil } else { remaining[kernel] = max(0, left - step * speed(kernel, in: set, k: k)) }
            }
        }
        if let (kernel, left) = remaining.first { ends[kernel] = now + left }
        return ends
    }

    /// **Seconds each kernel is busy** in a product `m×k×n` with these rows (the ANE's rounded up to its unit): the sides' clocks
    /// of the overlap, before the set's overhead.
    package func busy(m: Int, k: Int, n: Int, rows: [Kernel: Int], sustained: Bool = false) -> [Kernel: Double] {
        let running = rows.filter { $0.value > 0 }.map(\.key)
        let set = running.count > 1 ? sharing(running) : Sharing()
        if running.count > 1, !set.covers(k: k) { return [:] }
        var solo: [Kernel: Double] = [:]
        for (kernel, r) in rows where r > 0 {
            guard let t = alone(kernel, rows: r, k: k, n: n, sustained: sustained) else { return [:] }
            solo[kernel] = t + set.lift(kernel.engine, k: k, n: n)
        }
        return overlap(solo, k: k)
    }

    /// **The time of `m×k×n` on `kernels` with these rows** (Σ = m): the last end of the overlap, times the set's `sustained` factor
    /// when asked, plus its overhead.
    package func seconds(m: Int, k: Int, n: Int, rows: [Kernel: Int], sustained: Bool = false) -> Double? {
        let kernels = rows.keys.filter { rows[$0]! > 0 }
        guard !kernels.isEmpty else { return nil }
        let ends = busy(m: m, k: k, n: n, rows: rows, sustained: sustained)
        guard ends.count == kernels.count, let last = ends.values.max() else { return nil }
        guard kernels.count > 1 else { return last }
        let set = sharing(Array(kernels))
        return last * (sustained ? max(1, set.sustained ?? 1) : 1) + set.overhead
    }

    /// **The balanced split of `m×k×n` on `kernels`**: shares solved so that every side ends together — each alone seconds `sᵢ = κᵢ·T`,
    /// `κ` the full set's —, an engine with no positive share left out, the ANE's rows tried at the units on both sides of the
    /// balance, and the candidates timed by the overlap (`seconds`). `nil` when a kernel has no line.
    package func split(m: Int, k: Int, n: Int, kernels: [Kernel], sustained: Bool = false) -> Split? {
        var active = kernels
        let perRow = 2 * Double(k) * Double(n)
        while !active.isEmpty {
            if active.count == 1 {
                let only = active[0]
                return seconds(m: m, k: k, n: n, rows: [only: m], sustained: sustained).map { Split(rows: [only: m], seconds: $0) }
            }
            let set = sharing(active)
            var rate: [Kernel: Double] = [:], fixed: [Kernel: Double] = [:], kappa: [Kernel: Double] = [:]
            for kernel in active {
                guard let (line, short) = self.line(for: kernel, rows: m, k: k, n: n), line.rate > 0 else { return nil }
                let slow = sustained ? max(1, heat[kernel.engine] ?? 1) : 1
                rate[kernel] = self.rate(kernel, line: line, k: k, n: n, short: short) / slow
                fixed[kernel] = line.fixed(k: k, n: n) + set.lift(kernel.engine, k: k, n: n)
                kappa[kernel] = speed(kernel, in: set, k: k)
            }
            let w = perRow * Double(m)
            let tStar = (w + active.reduce(0) { $0 + fixed[$1]! * rate[$1]! }) / active.reduce(0) { $0 + kappa[$1]! * rate[$1]! }
            let share = Dictionary(uniqueKeysWithValues: active.map { ($0, (kappa[$0]! * tStar - fixed[$0]!) * rate[$0]! / perRow) })
            if let gone = active.first(where: { share[$0]! <= 0 }) {
                active.removeAll { $0 == gone }
                continue
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
