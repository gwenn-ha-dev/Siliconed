import Foundation

/// **More detail** — ComfyUI's Detail Daemon (Jonseed, `detail_daemon_node.py` at `3394e44`, MIT,
/// itself after muerrilla's `sd-webui-detail-daemon`), reduced to one named choice.
///
///     the sampler's σᵢ ──────────────────────────────────────────▶ x += (σᵢ₊₁ − σᵢ)·v     unchanged
///            └─ × (1 − 0.1·mᵢ) ─▶ σ'ᵢ ─▶ DiT(x, σ'ᵢ) ─▶ v
///
///     mᵢ: a bell over the steps,  0 ─ rises (start → mid) ─ amount ─ falls (mid → end) ─ 0
///
/// **The model is told less noise remains than really does**, in the middle of the trajectory only:
/// it removes less, and what it leaves is fine texture. The Euler step keeps the true σ, so the
/// trajectory still lands at σ = 0. Nothing else changes: same evaluations, same cost — the only
/// work added is `steps` multiplications on the CPU.
///
/// What the node does that a port by reading its README would miss:
///
///   - the bell is laid over **the steps the sampler receives** (`len(sigmas) − 1`), so over
///     Z-Image's null last step too (diffusers' schedule ends `…, 0, 0`), and in img2img over the
///     tail that runs, not the full schedule (ComfyUI hands the sampler the truncated σs);
///   - the indices are `int(round(x·(steps − 1)))` with **Python's `round`, half to even**: at 6
///     steps `mid·5 = 2.5` rounds to 2, and the bell peaks at the third step, not the fourth —
///     on a turbo's 6 to 9 steps it touches 1 to 4 evaluations, no more;
///   - `start_values.any()` guards the scaling: a rising edge of a single point stays at **0**, not
///     at `start_offset`;
///   - the multiplier is ×0.1 and ×CFG (`1 − 0.1·m·cfg`, floored at 10⁻⁶). The turbos here run
///     **without CFG**, so `cfg = 1` — the node would read 1 from them too;
///   - the schedule is stored in **float32** (`torch.tensor(…, dtype=float32)`), and σ' is the
///     float32 σ times the factor rounded to float32;
///   - a σ ≤ 0 is passed unchanged (`get_dd_schedule` returns 0 there);
///   - **the bell is in step indices, not in σ**: on a hard-shifted schedule (Qwen-Image-2.1) the
///     node's default lands where the composition is decided, and changes the picture instead of
///     its texture. Hence one bell per family (`Bell`), hidden like the rest.
///
/// Every denoiser honours it — Z-Image through `Sampler`, Anima, Krea 2, FLUX.2 [klein] and ERNIE
/// through `euler` (their modulation tables tabulated at σ'), Qwen-Image-2.1 through its own
/// sampler (late bell). The preview, the spectral schedule and img2img's start read the true σ.
///
/// **`normal` is the identity, to the bit**: `modelSigmas` returns its argument, and every sampler
/// passes the same σ it passed before. Judged on the products' md5.
public enum Detail: String, Sendable, Hashable, CaseIterable, Codable {
    case normal, more, most

    /// The node's knobs. None of them reaches the app: `more` and `most` fix them (`schedule`).
    package struct Schedule: Equatable, Sendable {
        package var amount: Double
        package var start = 0.2, end = 0.8, bias = 0.5, exponent = 1.0
        package var startOffset = 0.0, endOffset = 0.0, fade = 0.0
        package var smooth = true
        package init(amount: Double, start: Double = 0.2, end: Double = 0.8, bias: Double = 0.5,
                     exponent: Double = 1, startOffset: Double = 0, endOffset: Double = 0,
                     fade: Double = 0, smooth: Bool = true) {
            self.amount = amount; self.start = start; self.end = end; self.bias = bias
            self.exponent = exponent; self.startOffset = startOffset; self.endOffset = endOffset
            self.fade = fade; self.smooth = smooth
        }
    }

    /// **Where a denoiser's bell sits.** The node lays it over step *indices*, so where it falls in
    /// σ depends on the schedule — and one shape does not fit both visible families:
    ///
    ///   - `standard`, the node's defaults (start 0.2, end 0.8, bias 0.5): on Z-Image's 8 steps it
    ///     peaks at σ 0.69 and touches 0.88 → 0.55. Clean up to amount 2; at 3, confetti.
    ///   - `late` (start 0.6, end 1.0): Qwen-Image-2.1's schedule is shifted hard (512²: σ = 1,
    ///     0.96, 0.92, 0.84, 0.63, 0.36), and the standard bell lands on σ 0.92 and 0.84, where the
    ///     composition is still being decided: amount 0.5 already changed the woman and her clothes,
    ///     1 grew a bird on branches. The late bell touches only its 5th step (σ 0.63 at 512²) —
    ///     same picture, finer texture; up to 1.3 clean, at 1.6 white crackle lines.
    package enum Bell: Sendable { case standard, late }

    /// **The calibrated settings**, by eye at 512², seed 42, the library prompt, on the two
    /// visible families at their default steps (Z-Image 8, Qwen-Image-2.1 6), checked on a second
    /// seed. `more` ≈ +10 % of high-frequency energy (mean |Laplacian| of the luminance) and the same
    /// picture; `most` ≈ +20 %, the strongest that stayed clean, a step short of where artifacts
    /// begin. The four hidden families take the standard bell, unjudged.
    package func schedule(_ bell: Bell = .standard) -> Schedule? {
        switch (self, bell) {
        case (.normal, _): return nil
        case (.more, .standard): return Schedule(amount: 1)
        case (.most, .standard): return Schedule(amount: 2)
        case (.more, .late): return Schedule(amount: 1, start: 0.6, end: 1)
        case (.most, .late): return Schedule(amount: 1.3, start: 0.6, end: 1)
        }
    }

    /// **The amount follows the image's side**: calibrated at 512², it is multiplied by
    /// `512 / √(width·height)` — the same lie paints four times the tokens at 1024².
    ///
    ///   - Z-Image (standard bell): at 1024², More's amount 1 gave Most's +20 % and Most's 2 confetti
    ///     on skin and walls; scaled, More 0.5 and Most 1 (+8–12 %, +19–24 %, clean; 1.5 already
    ///     flecks the hair); at 1024×1536 Most is 0.82 (+16 %, clean).
    ///   - Qwen-Image-2.1 (late bell): at 1024², More's 1 already strews white flecks, Most's 1.3
    ///     everywhere; 0.65 is the last clean amount — exactly Most scaled. More is 0.5.
    ///
    /// Never more than at 512².
    package static func scale(width: Int, height: Int) -> Double {
        min(1, 512 / Double(width * height).squareRoot())
    }

    /// **The σ each evaluation hands the model**, for a full schedule (`steps + 1` values, ending in
    /// 0) of which steps `start…` run. `normal`: `sigmas` itself. `scale`: `Detail.scale` of the
    /// render's size (1 at 512²).
    package func modelSigmas(_ sigmas: [Float], start: Int = 0, bell: Bell = .standard, scale: Double = 1) -> [Float] {
        guard var schedule = schedule(bell), sigmas.count - start >= 2 else { return sigmas }
        schedule.amount *= scale
        return Detail.adjusted(sigmas, start: start, schedule)
    }

    /// `make_detail_daemon_schedule`, line for line, in double.
    package static func multipliers(steps: Int, _ s: Schedule) -> [Double] {
        guard steps > 0 else { return [] }
        let start = min(s.start, s.end)
        let mid = start + s.bias * (s.end - start)
        func index(_ x: Double) -> Int { Int((x * Double(steps - 1)).rounded(.toNearestOrEven)) }
        let (startIndex, midIndex, endIndex) = (index(start), index(mid), index(s.end))
        var multipliers = [Double](repeating: 0, count: steps)

        func edge(from a: Double, to b: Double, count: Int, offset: Double) -> [Double] {
            var values = linspace(a, b, count)
            if s.smooth { values = values.map { 0.5 * (1 - cos($0 * Double.pi)) } }
            values = values.map { pow($0, s.exponent) }
            if values.contains(where: { $0 != 0 }) {
                values = values.map { $0 * (s.amount - offset) + offset }
            }
            return values
        }
        let rising = edge(from: 0, to: 1, count: midIndex - startIndex + 1, offset: s.startOffset)
        let falling = edge(from: 1, to: 0, count: endIndex - midIndex + 1, offset: s.endOffset)
        // NumPy's slice assignments, in the node's order (the falling edge overwrites `mid`).
        func assign(_ values: [Double], at lower: Int) {
            for (k, v) in values.enumerated() where lower + k >= 0 && lower + k < steps { multipliers[lower + k] = v }
        }
        assign(rising, at: startIndex)
        assign(falling, at: midIndex)
        for i in 0..<min(max(startIndex, 0), steps) { multipliers[i] = s.startOffset }
        if endIndex + 1 < steps { for i in (endIndex + 1)..<steps { multipliers[i] = s.endOffset } }
        return multipliers.map { $0 * (1 - s.fade) }
    }

    /// `np.linspace(a, b, n)`: `i·step + a`, the last point set to `b`.
    private static func linspace(_ a: Double, _ b: Double, _ n: Int) -> [Double] {
        guard n > 1 else { return n == 1 ? [a] : [] }
        let step = (b - a) / Double(n - 1)
        var y = (0..<n).map { Double($0) * step + a }
        y[n - 1] = b
        return y
    }

    /// The sampler wrapper: the schedule over the steps that run, in float32, then
    /// `σ·max(10⁻⁶, 1 − 0.1·m·cfg)` for each σ > 0. Steps before `start` are left as they are.
    package static func adjusted(_ sigmas: [Float], start: Int, _ s: Schedule, cfgScale: Double = 1) -> [Float] {
        let m = multipliers(steps: sigmas.count - 1 - start, s).map { Float($0) }
        var out = sigmas
        for (k, multiplier) in m.enumerated() where sigmas[start + k] > 0 {
            let factor = Float(max(1e-6, 1 - Double(multiplier) * 0.1 * cfgScale))
            out[start + k] = sigmas[start + k] * factor
        }
        return out
    }
}
