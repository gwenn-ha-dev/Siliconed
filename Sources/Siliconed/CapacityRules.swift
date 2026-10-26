import Foundation

/// **The rules: what the capacity map decides about the engine's compute branches** — each one a rule on
/// measured capacities (a throughput, a correctness check), never on a chip's name. One place: the app
/// reads a map, these rules turn it into settings (`EngineSettings.load(from:map:)`), and the report says
/// what they chose (`Report.decisions`), so a shared JSON tells what that Mac uses.
///
///     a map (Diagnostic.Report)
///        │  margin = max(5 %, the drift witness's worst |1 − after / before|, GPU and AMX)
///        │
///        ├─ products, m = 1024 and 4096        g_m = Σ t(GPU alone) / Σ t(GPU + AMX)   ─▶ amx
///        │                                     share = R_gpu / (R_gpu + R_amx)          ─▶ amx_fraction
///        │  cuts, k and n axes at both m         ρ = (1 − T*) / T*, T* balanced side by side ─▶ amx_shares, T(k, n, m)
///        │                                     g_m = Σ t(GPU alone) / Σ t(GPU + ANE)    ─▶ gemm_ane (beside amx, shape by shape)
///        │                                     share = Σ f · T*_ane / Σ f               ─▶ ane_fraction
///        │  ANE cuts, the families' shapes       ρ = (1 − T*) / T* where its pair wins, 0 where the AMX's does ─▶ ane_shares
///        │                                     t(GPU + AMX + ANE) < min(both pairs) · (1 − margin)      ─▶ ane_amx
///        │                                     g_m = Σ [t(fp32 path) + t(widen k·n)]
///        │                                         / Σ [t(bf16 weight) + t(copy k·n)]    ─▶ gemm_tensor
///        │                                     g_m = Σ [t(fp32 path) + t(widen k·n)]
///        │                                         / Σ [t(bf16 weight, bf16 activation) + t(copy k·n)]    ─▶ gemm_bf16_act
///        │                                     g_m = Σ [t(fp32 path) + t(widen convrot k·n)]
///        │                                         / Σ [t(int8 weight) + t(copy k·n)]    ─▶ gemm_int8
///        │                                     g_m = Σ [t(fp32 path) + t(widen Q8_0 k·n)]
///        │                                         / Σ [t(Q8_0 weight) + t(copy k·n)]    ─▶ gemm_q8
///        ├─ attention, S = 1024 and 4096       g_S = Σ t(MPSGraph) / Σ t(flash)          ─▶ flash
///        │                                     g_S = Σ t(MPSGraph) / Σ t(head split)     ─▶ head_split
///        └─ a branch is on iff   every engine it would use passed its check (none missing)
///                           and  gain = geometric mean of the g over the two sizes  > 1 + margin
///                           and  no size loses beyond the noise: min g > 1 − margin
///
/// `t = flops / rate` for each item; a `g` sums the items of one size — both models' blocks, up and
/// down, so a size counts as the products one of its DiT blocks runs. The two sizes are a 512² and a
/// 1024² image's tokens, and **each weighs the same** in the geometric mean: the app renders both, and
/// the 4096 items, four times the work, would otherwise decide for the 512² render too. The `min`
/// clause keeps a gain at 1024² from buying a loss at 512².
///
/// What sets these rules apart — what rules written by analogy with the defaults would get wrong:
///
///   - **the margin is the map's own noise, not a constant.** A map taken on a hot or battery-throttled
///     machine drifts (this M1 Pro on battery, 14/10/2026: the GPU ×0.84 over 16 s); 5 % is only the
///     floor. Within the margin, a branch stays on the side the GPU alone stands — MPSGraph, no cut:
///     the path with fewer moving parts. Flash on this M1 Pro is such a tie, and stays off;
///   - **`amx_fraction` is the share that balances the whole set of products**: `R` is each engine's
///     total work over its total time (all eight items, each weighted by its flops), the GPU taking
///     `R_gpu / (R_gpu + R_amx)` of the rows — **only without `cuts`**: with them, every product's share is
///     computed from its `k`, `n` and `m` (`Conductor.Shares`; one frozen share left the GPU waiting 2.4 s of a
///     1024² Krea 2 evaluation), at quantum 64. Both move the frozen cut, hence the last bits of a
///     render — like `amx` itself; neither re-associates a sum (one engine per output row);
///   - **`gemm_tensor` is judged against the path it replaces, preparation included**: the fp32 path
///     is the GPU alone, or GPU + AMX when `amx` went on, each weight widened first; the bf16 path
///     copies the weight instead (`Artifact.stage`) — both at the map's own CPU rates (`widening`
///     "bf16" and "bf16 copy"). It has no AMX: when it goes on, `amx` goes off in the settings (the
///     decision still says what co-execution measured);
///   - **`gemm_bf16_act` against the same fp32 path, and rounding is its price**: the candidate is
///     `matmul2d` `bfloat × bfloat → float` with the activation rounded just before (the rounding's pass is in
///     its time), the weight copied as `gemm_tensor`'s. It is a branch of its own, not `gemm_tensor`'s: the
///     two read the same weight, but this one changes the numbers (an operand rounded to bf16, P20 allows it,
///     the check judges it against diffusers in bf16), and it is the only one that wins where `matmul2d`
///     runs on a bf16 matrix unit and not on fp32 (M5 Pro: 15.3 TFLOP/s bf16 × bf16, 6.0 fp32 × bf16, the
///     GPU's fp32 5.8–6.0). On a Mac without that unit it loses by the rounding (M1 Pro, ×0.5) and stays off.
///     The AMX and the ANE are untouched in the settings: they serve the products the path leaves widened
///     (the text encoder, a product under `bf16_act_min` rows). Only the bf16 weights are measured: the
///     8-bit ones round only if `bf16_act_8bit` says so, which no map decides;
///   - **`gemm_int8` likewise, against what an int8 map's default path does**: the convrot widening
///     ("int8 convrot") then the fp32 path, against the bytes copied ("int8 copy") then the int8 product —
///     beside the AMX on its columns when the AMX stays on (`amx` on, `gemm_tensor` off), alone otherwise.
///     It keeps the AMX: it reads its own widened columns. A map measures it on synthetic convrot weights
///     whatever is installed; on a bf16 map the setting does nothing;
///   - **`gemm_q8` the same way, on its own**: a GGUF Q8_0 map (Anima Compact) widens with its block scales
///     ("q8_0"), the candidate copies the bytes and the scales ("q8_0 copy"). A separate decision, not
///     `gemm_int8`'s: on the M1 Pro the int8 product wins and the Q8_0 one loses (×0.29–0.67: the
///     operand written into threadgroup memory runs `matmul2d` at ~2.3 TFLOP/s) — one setting for both
///     would have bought one path's gain with the other's loss;
///   - **`gemm_ane` and `amx` share the shapes, never a product**: both take rows off the GPU, and side by
///     side they slow each other (the M1 Pro's Anima Light 1024²: GPU + ANE 44.7 s, GPU + AMX 46.8 s, all three
///     46.2 s, 22/10/2026), so a product the ANE takes has the GPU alone beside it. Each is judged against the
///     GPU alone; **with both on, each shape of the ANE's grid goes to the ANE where its pair beats the GPU
///     alone and keeps 90 % of the AMX's speed** (`aneTolerance`), to the AMX elsewhere: thermal efficiency —
///     the ANE does the work at a fraction of the CPU cores' power. One engine for every shape was wrong both
///     ways on this M1 Pro (23/10/2026): Z-Image 1024² under the ANE 4 % slower for 10 % less energy, Krea 2 22 %
///     slower and 10 % MORE energy, its down projection (K = 16 384) 317 ms beside the ANE, 183 beside the AMX.
///     **And all three where they add up** (`ane_amx`): a shape where the three engines at once beat both pairs by
///     more than the margin takes them all — the ANE on its rows, the GPU beside the AMX on the rest —, whichever
///     pair it would otherwise have gone to. The three at once were once ruled out on one whole render (the line
///     above); one product at a time they add up on some shapes and not on others (M1 Pro, MOSS-Image's block at
///     4 096 rows, 26/10/2026: 3072 → 16384 6.92 TFLOP/s against GPU + ANE 6.34, 8192 → 3072 5.77 against 6.56),
///     so it is the grid's to say, on each Mac. The split is not swept: each engine takes rows in proportion to its
///     rate, which both pairs give (`threeWayShare`). The ANE keeps its 10 % against the AMX's pair; the three
///     have no such favour — they add the CPU's power and heat to the ANE's pair. The ANE's
///     products are judged against bf16's error + 25 % (invariant 3: it computes in fp16), not the fp32
///     threshold; the paths that replace the fp32 one (`gemm_tensor`, `gemm_int8`, `gemm_q8`) are then
///     judged against GPU + ANE. `ane_fraction` is each product's balanced share weighted by its flops;
///   - **flash and the head split are not exclusive**: Z-Image's blocks take flash before the split
///     (`Block`), Qwen-Image-2.1's target only has the split. So flash goes on only if it also beats the
///     split where the split goes on — otherwise Z-Image would trade a faster path for a slower one;
///   - **what the map does not decide stays at its default**, by rule, not by omission:
///       · the head split's `head_split_fraction` (0.4) moves only where the map's cut call (each part timed)
///         predicts another share faster by more than the margin (`HeadSplitAttention.balance`);
///       · fp16 *activations*: P20 (never under the published weights nor under bf16, sums in fp32)
///         stands — the fp16 figures answer a fleet question, they choose nothing (`gemm_tensor` rounds
///         nothing: its activations stay fp32; `gemm_bf16_act` rounds to bf16, and only to bf16);
///       · `bf16_act_min` (256 rows) and `bf16_act_8bit`;
///       · `amx_quantum` (64 under the grid, MPS's row tile), `amx_min`, `parallel`: nothing measures them.
extension Diagnostic {
    /// **One branch, decided**: on or off, and the figures behind it.
    package struct Decision: Codable, Sendable, Equatable {
        /// The branch is taken.
        package var on: Bool
        /// Every engine the branch would use ran and computed right (within its threshold).
        package var passed: Bool
        /// The geometric mean of `gains`: what is compared to `1 + margin`.
        package var gain: Double
        /// Per size (`Diagnostic.heights`, 1024 then 4096): the reference's summed time over the branch's.
        package var gains: [Double]

        package init(on: Bool, passed: Bool, gain: Double, gains: [Double]) {
            self.on = on; self.passed = passed
            self.gain = Diagnostic.significant(gain, 3); self.gains = gains.map { Diagnostic.significant($0, 3) }
        }
    }

    /// **What a map makes the engine use** — the rules above applied to one report.
    package struct Decisions: Codable, Sendable, Equatable {
        /// The noise a gain must clear: `max(minimumMargin, the drift witness's worst deviation)`.
        package var margin: Double
        /// GPU + AMX co-execution of the DiT products (`amx`).
        package var amx: Decision
        /// The GPU's share of the rows (`amx_fraction`) when `amx` is on; `nil` otherwise.
        package var amxFraction: Double?
        /// The GPU's share per weight shape and height (`amx_shares`, `Conductor.Shares`' text) when `amx` is
        /// on and the report measured `cuts`; `nil` otherwise — the shapes then take `amxFraction`.
        package var amxShares: String?
        /// The in-house flash kernel instead of MPSGraph (`flash`).
        package var flash: Decision
        /// The SDPA's heads cut between the GPU and the CPU from S = 1024 on (`head_split`).
        package var headSplit: Decision
        /// The share of heads the CPU takes (`head_split_fraction`) when the cut is on and the report's figures
        /// say another share than 0.4 beats it by more than the margin; `nil` otherwise (and before schema 11).
        package var headSplitFraction: Double?
        /// The DiT's products on the bf16 weights (`gemm_tensor`, `TensorGEMM`); `nil` in a decision
        /// stored by a report of schema 5.
        package var tensor: Decision?
        /// The DiT's products on an activation rounded to bf16 (`gemm_bf16_act`); `nil` before schema 11.
        package var bf16Act: Decision?
        /// The products on the published int8 weights (`gemm_int8`); `nil` before schema 8.
        package var int8: Decision?
        /// The products on GGUF Q8_0 weights (`gemm_q8`); `nil` before schema 9.
        package var q8: Decision?
        /// The Neural Engine beside the GPU on the fp32 products (`gemm_ane`); `nil` before schema 10.
        package var ane: Decision?
        /// The GPU's share of the rows beside the ANE (`ane_fraction`) when `ane` is on; `nil` otherwise.
        package var aneFraction: Double?
        /// The GPU's share beside the ANE per weight shape and height (`ane_shares`) when `ane` is on and the
        /// report measured `aneCuts`.
        package var aneShares: String?
        /// The shapes the three engines take at once (`ane_amx`: ρ 1 there, 0 elsewhere) — only when some do.
        package var aneAMX: String?

        package init(margin: Double, amx: Decision, amxFraction: Double?, flash: Decision, headSplit: Decision,
                    tensor: Decision? = nil, amxShares: String? = nil, int8: Decision? = nil, q8: Decision? = nil,
                    bf16Act: Decision? = nil, ane: Decision? = nil, aneFraction: Double? = nil, aneShares: String? = nil,
                    headSplitFraction: Double? = nil, aneAMX: String? = nil) {
            self.margin = Diagnostic.significant(margin, 3); self.amx = amx
            self.amxFraction = amxFraction.map { Diagnostic.significant($0, 3) }
            self.amxShares = amxShares
            self.flash = flash; self.headSplit = headSplit; self.tensor = tensor; self.int8 = int8
            self.bf16Act = bf16Act
            self.q8 = q8; self.ane = ane
            self.aneFraction = aneFraction.map { Diagnostic.significant($0, 3) }
            self.aneShares = aneShares; self.aneAMX = aneAMX
            self.headSplitFraction = headSplitFraction.map { Diagnostic.significant($0, 3) }
        }

        private enum CodingKeys: String, CodingKey { case margin, amx, amxFraction, amxShares, flash, headSplit, tensor, bf16Act, int8, q8, ane, aneFraction, aneShares, aneAMX, headSplitFraction }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(margin, forKey: .margin); try c.encode(amx, forKey: .amx)
            try c.encode(amxFraction, forKey: .amxFraction)
            try c.encode(amxShares, forKey: .amxShares)
            try c.encode(flash, forKey: .flash); try c.encode(headSplit, forKey: .headSplit)
            try c.encode(tensor, forKey: .tensor); try c.encode(int8, forKey: .int8)
            try c.encode(q8, forKey: .q8)
            try c.encodeIfPresent(bf16Act, forKey: .bf16Act)
            try c.encodeIfPresent(ane, forKey: .ane); try c.encodeIfPresent(aneFraction, forKey: .aneFraction)
            try c.encodeIfPresent(aneShares, forKey: .aneShares)
            try c.encodeIfPresent(aneAMX, forKey: .aneAMX)
            try c.encodeIfPresent(headSplitFraction, forKey: .headSplitFraction)
        }

        /// The settings they write, under `EngineSettings`' keys (those of `store/profil.json`).
        package var settings: [String: Any] {
            let tensorOn = tensor?.on == true
            let aneOn = ane?.on == true && !tensorOn
            var values: [String: Any] = ["amx": amx.on && !tensorOn, "flash": flash.on, "head_split": headSplit.on,
                                         "gemm_tensor": tensorOn, "gemm_int8": int8?.on == true,
                                         "gemm_q8": q8?.on == true, "gemm_ane": aneOn,
                                         "gemm_bf16_act": bf16Act?.on == true]
            if headSplit.on, let headSplitFraction { values["head_split_fraction"] = headSplitFraction }
            if aneOn, let aneFraction { values["ane_fraction"] = aneFraction }
            if aneOn, let aneShares { values["ane_shares"] = aneShares }
            if aneOn, amx.on, let aneAMX { values["ane_amx"] = aneAMX }
            if amx.on, !tensorOn, let amxFraction { values["amx_fraction"] = amxFraction }
            if amx.on, !tensorOn, let amxShares { values["amx_shares"] = amxShares; values["amx_quantum"] = 64 }
            return values
        }
    }

    /// **The three engines' split from their two pairs**: each engine takes rows in proportion to its rate, and the
    /// pairs give the rates' ratios — ρ_ane = (1 − T_ane) / T_ane beside the GPU (`aneShare`), ρ_amx likewise
    /// (`amxShare`, the AMX's grid). The GPU's side (GPU + AMX) keeps `(1 + ρ_amx) / (1 + ρ_ane + ρ_amx)` of the
    /// rows, the ANE the rest; the conductor then cuts the GPU's side at its own share. M1 Pro, 3072 → 16384 at
    /// 4 096 rows (26/10/2026): pairs 0.45 and 0.65 give the ANE 44 %, the AMX 20 % — the best of a sweep was 41 %
    /// and 19 %.
    package static func threeWayShare(ane aneShare: Double, amx amxShare: Double) -> Double {
        let (ane, amx) = ((1 - aneShare) / aneShare, (1 - amxShare) / amxShare)
        return (1 + amx) / (1 + ane + amx)
    }

    /// The margin's floor: below it, two runs of one machine already disagree (invariant 6: 4 to 13 %
    /// over a few renders; the map's own witness ×0.990–0.997 on a quiet M1 Pro).
    package static let minimumMargin = 0.05

    /// The share of the AMX's gain the ANE needs to take the rows from it: below, the AMX is clearly faster;
    /// above, the ANE's efficiency decides (a choice of the product, not a measure).
    package static let aneTolerance = 0.9

    /// **The rules.** Pure: the same report always gives the same decisions.
    package static func decide(products: [Product], attention: [AttentionCapacity], widening: [Widening],
                              witness: Witness, cuts: [Cut] = [], aneCuts: [Cut] = []) -> Decisions {
        var drift = abs(1 - witness.gpuAfter / witness.gpuBefore)
        if let before = witness.amxBefore, let after = witness.amxAfter, before > 0 { drift = max(drift, abs(1 - after / before)) }
        let margin = max(minimumMargin, drift.isFinite ? drift : 1)

        /// `on`, from the per-size gains and the checks.
        func decision(passed: Bool, gains: [Double]) -> Decision {
            let usable = passed && !gains.isEmpty && gains.allSatisfy { $0.isFinite && $0 > 0 }
            let gain = usable ? exp(gains.map { log($0) }.reduce(0, +) / Double(gains.count)) : 0
            let on = usable && gain > 1 + margin && gains.min()! > 1 - margin
            return Decision(on: on, passed: passed, gain: gain, gains: usable ? gains : [])
        }
        /// Per size: Σ flops / rate of the reference over that of the candidate.
        func gains<T>(_ items: [T], size: (T) -> Int, flops: (T) -> Double, reference: (T) -> Measure?,
                      candidate: (T) -> Measure?) -> [Double] {
            heights.compactMap { h in
                let these = items.filter { size($0) == h }
                guard !these.isEmpty else { return nil }
                var (tReference, tCandidate) = (0.0, 0.0)
                for item in these {
                    guard let r = reference(item), let c = candidate(item), r.rate > 0, c.rate > 0 else { return .nan }
                    tReference += flops(item) / r.rate; tCandidate += flops(item) / c.rate
                }
                return tReference / tCandidate
            }
        }

        // ── amx: the co-execution against the GPU alone ──
        let productFlops = { (p: Product) in 2.0 * Double(p.m) * Double(p.k) * Double(p.n) }
        let amxPassed = !products.isEmpty && products.allSatisfy { p in
            p.amx?.pass == true && p.coexecution?.pass == true
        }
        var amx = decision(passed: amxPassed, gains: gains(products, size: \.m, flops: productFlops,
                                                           reference: { $0.gpu }, candidate: \.coexecution))

        // ── gemm_ane: the GPU beside the Neural Engine, against the GPU alone; beside amx, shape by shape ──
        let anePassed = !products.isEmpty && products.allSatisfy { $0.ane?.pass == true }
        var ane = decision(passed: anePassed, gains: gains(products, size: \.m, flops: productFlops,
                                                           reference: { $0.gpu }, candidate: \.ane))
        // Each shape of the ANE's grid to the ANE where its pair beats the GPU alone and keeps `aneTolerance`
        // of the AMX's speed — the same work at a fraction of the CPU's power and heat (a choice of the
        // product, 22/10/2026) —, to the AMX elsewhere: Krea 2's down projection is the AMX's, its other
        // products the ANE's. A grid without walls (before 23/10/2026): one engine for every shape.
        let routed = !aneCuts.isEmpty && aneCuts.allSatisfy { $0.ane != nil && $0.gpu != nil }
        func aneTakes(_ cut: Cut) -> Bool {
            guard let paired = cut.ane, let alone = cut.gpu, paired < alone else { return false }
            guard amx.on, let beside = cut.amx else { return true }
            return paired * aneTolerance <= beside
        }
        // The AMX's grid (`amx_shares`), which the GPU's side follows when the three engines take a shape.
        let amxGrid = amx.on && !cuts.isEmpty
            ? Conductor.Shares(cuts.map { .init(k: $0.k, n: $0.n, m: $0.m, ratio: (1 - $0.share) / $0.share) }) : nil
        /// The ANE's share of the GPU's side when all three take the shape, or `nil` when they do not: they must beat
        /// both pairs beyond the noise.
        func threeShare(_ cut: Cut) -> Double? {
            guard ane.passed, amx.on, let three = cut.three, let paired = cut.ane, let beside = cut.amx,
                  three < min(paired, beside) * (1 - margin),
                  let amxShare = amxGrid?.share(k: cut.k, n: cut.n, m: cut.m) else { return nil }
            return threeWayShare(ane: cut.share, amx: amxShare)
        }
        if ane.on && routed && !aneCuts.contains(where: { aneTakes($0) || threeShare($0) != nil }) { ane.on = false }
        if ane.on && amx.on && !routed { if ane.gain >= aneTolerance * amx.gain { amx.on = false } else { ane.on = false } }
        // The grid the GEMM reads: ρ where the ANE takes the shape, 0 (the GPU's whole share) where it does not.
        func aneRatio(_ cut: Cut) -> Double {
            if routed, let three = threeShare(cut) { return (1 - three) / three }
            return !routed || aneTakes(cut) ? (1 - cut.share) / cut.share : 0
        }
        let aneGrid = ane.on && !aneCuts.isEmpty
            ? Conductor.Shares(aneCuts.map { .init(k: $0.k, n: $0.n, m: $0.m, ratio: aneRatio($0)) })
            : nil
        let threeGrid = ane.on && routed && aneCuts.contains(where: { threeShare($0) != nil })
            ? Conductor.Shares(aneCuts.map { .init(k: $0.k, n: $0.n, m: $0.m, ratio: threeShare($0) != nil ? 1 : 0) }).description
            : nil
        var aneFraction: Double?
        if ane.on {
            let work = products.map(productFlops).reduce(0, +)
            aneFraction = products.map { productFlops($0) * $0.aneGPUShare! }.reduce(0, +) / work
        }
        /// Whether the product's shape goes to the ANE (its grid's share under 1).
        func aneRoutes(_ p: Product) -> Bool {
            ane.on && (aneGrid.flatMap { $0.share(k: p.k, n: p.n, m: p.m, range: NeuralEngine.shareRange) }.map { $0 < 1 } ?? true)
        }
        /// The fp32 path the bf16 and 8-bit paths replace: what `amx` and `gemm_ane` chose.
        func fp32Path(_ p: Product) -> Measure? { aneRoutes(p) ? p.ane : amx.on ? p.coexecution : p.gpu }
        var fraction: Double?
        if amx.on {
            // Each engine's total work over its total time: the share that ends both together.
            var (work, tGPU, tAMX) = (0.0, 0.0, 0.0)
            for p in products {
                let f = productFlops(p)
                work += f; tGPU += f / p.gpu.rate; tAMX += f / p.amx!.rate
            }
            let (gpu, cpu) = (work / tGPU, work / tAMX)
            fraction = min(Conductor.fractionRange.upperBound, max(Conductor.fractionRange.lowerBound, gpu / (gpu + cpu)))
        }

        // ── gemm_tensor: the bf16-weight product against the fp32 path it replaces, preparations included ──
        // Seconds, not ratios of rates: a product's flops at TFLOP/s and a weight's values at G values/s
        // add up in one sum. The fp32 path is the one `amx` chose.
        let widen = widening.first { $0.kind == "bf16" }?.cpu, copy = widening.first { $0.kind == "bf16 copy" }?.cpu
        let tensorPassed = !products.isEmpty && widen?.pass == true && copy?.pass == true
            && products.allSatisfy { $0.tensor?.pass == true && fp32Path($0)?.pass == true }
        let tensorGains: [Double] = heights.compactMap { h in
            let these = products.filter { $0.m == h }
            guard !these.isEmpty else { return nil }
            var (tReference, tCandidate) = (0.0, 0.0)
            for p in these {
                guard let r = fp32Path(p)?.rate, let t = p.tensor?.rate, let w = widen?.rate, let c = copy?.rate,
                      r > 0, t > 0, w > 0, c > 0 else { return .nan }
                let (flops, values) = (productFlops(p) * 1e-12, Double(p.k * p.n) * 1e-9)
                tReference += flops / r + values / w
                tCandidate += flops / t + values / c
            }
            return tReference / tCandidate
        }
        let tensor = decision(passed: tensorPassed, gains: tensorGains)

        // ── gemm_bf16_act: the same bf16 weight, the activation rounded, against the same fp32 path ──
        // Absent from a map before schema 11 (no `bf16Act` measure): no decision, the branch off.
        let bf16Act: Decision? = products.contains { $0.bf16Act != nil } ? {
            let passed = !products.isEmpty && widen?.pass == true && copy?.pass == true
                && products.allSatisfy { $0.bf16Act?.pass == true && fp32Path($0)?.pass == true }
            let gains: [Double] = heights.compactMap { h in
                let these = products.filter { $0.m == h }
                guard !these.isEmpty else { return nil }
                var (tReference, tCandidate) = (0.0, 0.0)
                for p in these {
                    guard let r = fp32Path(p)?.rate, let t = p.bf16Act?.rate, let w = widen?.rate, let c = copy?.rate,
                          r > 0, t > 0, w > 0, c > 0 else { return .nan }
                    let (flops, values) = (productFlops(p) * 1e-12, Double(p.k * p.n) * 1e-9)
                    tReference += flops / r + values / w
                    tCandidate += flops / t + values / c
                }
                return tReference / tCandidate
            }
            return decision(passed: passed, gains: gains)
        }() : nil

        // ── gemm_int8 and gemm_q8: the 8-bit product against the widening + the fp32 path, preparations included ──
        // The AMX runs beside it when it stays on in the settings (`amx` on, `gemm_tensor` off).
        let amxRuns = amx.on && !tensor.on
        func eightBit(widened: String, copied: String, alone: (Product) -> Measure?, beside: (Product) -> Measure?) -> Decision {
            let widen = widening.first { $0.kind == widened }?.cpu, copy = widening.first { $0.kind == copied }?.cpu
            func candidate(_ p: Product) -> Measure? { amxRuns ? beside(p) : alone(p) }
            let passed = !products.isEmpty && widen?.pass == true && copy?.pass == true
                && products.allSatisfy { p in alone(p)?.pass == true && candidate(p)?.pass == true
                                              && fp32Path(p)?.pass == true }
            let gains: [Double] = heights.compactMap { h in
                let these = products.filter { $0.m == h }
                guard !these.isEmpty else { return nil }
                var (tReference, tCandidate) = (0.0, 0.0)
                for p in these {
                    // With `gemm_tensor` on, the AMX does not run: the GPU alone, or beside the ANE.
                    let reference = aneRoutes(p) ? p.ane : amxRuns ? p.coexecution : p.gpu
                    guard let r = reference?.rate, let t = candidate(p)?.rate, let w = widen?.rate, let c = copy?.rate,
                          r > 0, t > 0, w > 0, c > 0 else { return .nan }
                    let (flops, values) = (productFlops(p) * 1e-12, Double(p.k * p.n) * 1e-9)
                    tReference += flops / r + values / w
                    tCandidate += flops / t + values / c
                }
                return tReference / tCandidate
            }
            return decision(passed: passed, gains: gains)
        }
        let int8 = eightBit(widened: "int8 convrot", copied: "int8 copy", alone: \.int8, beside: \.int8Coexecution)
        let q8 = eightBit(widened: "q8_0", copied: "q8_0 copy", alone: \.q8, beside: \.q8Coexecution)

        // ── the attention: flash and the head split against MPSGraph ──
        let attentionFlops = { (a: AttentionCapacity) in
            4.0 * Double(a.sequence) * Double(a.sequence) * Double(a.heads) * Double(a.headDim)
        }
        let graphPassed = attention.allSatisfy { $0.graph?.pass == true }
        let flashPassed = !attention.isEmpty && graphPassed && attention.allSatisfy { $0.flash?.pass == true }
        let splitPassed = !attention.isEmpty && graphPassed && attention.allSatisfy { $0.headSplit?.pass == true }
        let split = decision(passed: splitPassed, gains: gains(attention, size: \.sequence, flops: attentionFlops,
                                                               reference: \.graph, candidate: \.headSplit))
        // The share of heads: from the largest size's cut call (both parts timed) and its graph alone.
        var headSplitFraction: Double?
        if split.on, let size = heights.last {
            let these = attention.filter { $0.sequence == size }
            let items = these.compactMap { a -> HeadSplitAttention.Balance? in
                guard let graph = a.graph, graph.rate > 0, let parts = a.splitParts else { return nil }
                return .init(heads: a.heads, graphSeconds: attentionFlops(a) / (graph.rate * 1e12), cpuHeads: parts.cpuHeads,
                             gpuSeconds: parts.gpuSeconds, cpuSeconds: parts.cpuSeconds)
            }
            if items.count == these.count { headSplitFraction = HeadSplitAttention.balance(items, margin: margin) }
        }
        var flash = decision(passed: flashPassed, gains: gains(attention, size: \.sequence, flops: attentionFlops,
                                                               reference: \.graph, candidate: \.flash))
        // Z-Image takes flash before the split: flash only where it is also the faster of the two.
        if flash.on && split.on && flash.gain <= split.gain { flash.on = false }

        // ── amx_shares: ρ = (1 − T) / T at each measured point; `Conductor.Shares` computes every product's T ──
        let shares = amx.on && !cuts.isEmpty
            ? Conductor.Shares(cuts.map { .init(k: $0.k, n: $0.n, m: $0.m, ratio: (1 - $0.share) / $0.share) }).description : nil
        // ── ane_shares: the same, on the ANE's grid ──
        let aneShares = aneGrid?.description
        return Decisions(margin: margin, amx: amx, amxFraction: fraction, flash: flash, headSplit: split, tensor: tensor,
                         amxShares: shares, int8: int8, q8: q8, bf16Act: bf16Act, ane: products.contains { $0.ane != nil } ? ane : nil,
                         aneFraction: aneFraction, aneShares: aneShares, headSplitFraction: headSplitFraction,
                         aneAMX: threeGrid)
    }

    /// The rules applied to `report`'s measures (never to its stored `decisions`: the rules live in the code).
    package static func decide(_ report: Report) -> Decisions {
        decide(products: report.products, attention: report.attention, widening: report.widening, witness: report.witness,
               cuts: report.cuts ?? [], aneCuts: report.aneCuts ?? [])
    }
}
