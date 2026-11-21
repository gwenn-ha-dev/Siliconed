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
///   - the bell is laid over **the steps the sampler receives** (`len(sigmas) − 1`) — on Z-Image
///     since 0.8.2 every one is an evaluation —, and in img2img over the
///     tail that runs, not the full schedule (ComfyUI hands the sampler the truncated σs);
///   - the indices are `int(round(x·(steps − 1)))` with **Python's `round`, half to even**: at 6
///     steps `mid·5 = 2.5` rounds to 2, and the bell peaks at the third step, not the fourth —
///     on a turbo's 8 steps the node's bell touches 4 evaluations (Z-Image's bell 2, the late bell 1);
///   - `start_values.any()` guards the scaling: a rising edge of a single point stays at **0**, not
///     at `start_offset`;
///   - the multiplier is ×0.1 and ×CFG (`1 − 0.1·m·cfg`, floored at 10⁻⁶). The turbos here run
///     **without CFG**, so `cfg = 1` — the node would read 1 from them too;
///   - the schedule is stored in **float32** (`torch.tensor(…, dtype=float32)`), and σ' is the
///     float32 σ times the factor rounded to float32;
///   - a σ ≤ 0 is passed unchanged (`get_dd_schedule` returns 0 there);
///   - **the bell is in step indices, not in σ**: on a schedule that lingers near σ = 1
///     (Qwen-Image-2.1) the node's default lands where the composition is decided, and changes the picture instead of
///     its texture. Hence one bell per family (`Bell`), hidden like the rest.
///
/// Every family honours it in its own render, on its own bell (`DenoisingModule.detailBell`). The preview and
/// img2img's start read the true σ.
///
/// **`normal` is the identity, to the bit**: `modelSigmas` returns its argument, and every sampler
/// passes the same σ it passed before. Judged on the products' md5.
package enum Detail: String, Sendable, Hashable, CaseIterable, Codable {
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
    /// σ depends on the schedule — and one shape fits no two families:
    ///
    ///   - no bell (`nil`): a denoiser that offers no Detail is told its σ themselves.
    ///     The node's defaults (start 0.2, end 0.8, bias 0.5, `Schedule`'s) are no family's any more.
    ///     They were Z-Image's while Z-Image ran 7 evaluations (σ 1, 0.947, 0.882,
    ///     0.8, 0.692, 0.545, 0.333): peak at σ 0.69, clean up to amount 2. On the 8 of the
    ///     official recipe (0.8.2) the same indices fall on σ 0.9 ×¼, 0.833 ×¾, 0.75 ×1, 0.643 ×½,
    ///     and More already moved the pose; at 1.5, the framing.
    ///   - `zImage` (start 0.4, end 0.9), Z-Image's since 0.8.2: σ 0.75 ×½ and **0.643 ×1, nothing
    ///     else**. Between two walls: a lie at σ ≥ 0.75 moves the pose and the framing, a lie at
    ///     σ 0.5 sows specks for little sharpness (start 0.5, end 1, all on 0.643 and 0.5: +10 % at
    ///     amount 3, +10 % at 4, the flat areas +50 %). The node's later bell (start 0.4, end 1,
    ///     σ 0.5 ×½ too) sharpens more per amount but specks at 2.5 already.
    ///   - `late` (start 0.857, end 1, bias 0.75), Qwen-Image-2.1's: **the last evaluation alone**
    ///     (σ 0.415) of the Turbo's 8 fixed steps (1, 0.978, 0.954, 0.927, 0.895, 0.845, 0.705, 0.415).
    ///     The Turbo still decides the face at σ 0.705: the bell the family had on the third-party turbo
    ///     LoRA it once carried (start 0.6 — on that LoRA's 6 steps, σ 0.63 alone, clean),
    ///     on the Turbo's 8 steps ½ at σ 0.845 and 1 at σ 0.705, gave seeds 7 and 1000 another face at
    ///     512² from amount 0.65; that 7th step alone, from amount 1. On the 8th alone the person stays.
    ///
    /// **Anima answers the lie turned round** (512², seeds 0, 7, 42). Told less noise, it does not give
    /// finer texture: at Z-Image's place nothing measurable (−4 to +1 %), at σ 0.5 **mosaic blocks** — flat squares of
    /// one colour, 16 to 32 pixels —, on the last step (σ 0.3) a softer picture (−2 to −14 % from 1 to 4) and specks
    /// from 2. Told MORE noise at Z-Image's place (a negative amount, σ' = σ·(1 + 0.1·|m|)) it draws more on the same
    /// person and pose, clean: −1 → +4 / +4 / +5 % (25 to 28 dB to Normal: small things change, a necklace, a trim),
    /// −1.5 → +6 / +7 / +10 %; −2 (+7 to +12 %, 23 dB) starts to move the framing. On the last steps the same lie does
    /// nothing (±1 %).
    ///
    /// **Krea 2's sits on its last two steps** (σ 1, 0.957, 0.905, 0.840, 0.760, 0.655, 0.513, 0.311). The node's
    /// default bell lands on σ 0.84–0.66, where the picture is still being decided: More kept the woman and added
    /// jewels and no texture, Most changed her face (16.6 dB to Normal). On σ 0.513 ×½ and 0.311 ×1 the face stays:
    /// 1 → +3 / +3 / +2 % (37 dB), 2 → +4 / +5 / +3 % (32 dB), clean; the gain stops there (2.5: the same) and 3 strews
    /// white specks. The smallest gain of the five.
    ///
    /// **MOSS-Image's is its seventh step alone** (σ 1, 0.968, 0.926, 0.871, 0.791, 0.670, 0.463, 0.024). The eighth
    /// is too late (amount 4: +0.2 %); the sixth with the seventh moves the picture sooner (29 dB at +25 to +39 %).
    /// On the seventh: 2 → +9 / +6 / +8 % (37 to 39 dB), 3 → +20 / +17 / +18 % (33 to 35 dB), same picture, sharper
    /// skin and hair; 4 (+30 %) starts to age the skin, 6 carves it.
    package struct Bell: Sendable, Equatable {
        /// Where the bell sits over the steps that run (the node's `start`, `end`, `bias`).
        package var start, end, bias: Double
        /// The amounts of More and Most at 512² (`scale`).
        package var more, most: Double
        /// Whether the amounts follow the image's side (`scale`). Not MOSS-Image's: at 1024² the scaled amounts give
        /// +3 and +7 % where the 512² ones give what they give at 512² (+11 to +14 %, +21 to +26 %, seeds 7 and 42,
        /// 37 to 40 dB to Normal, clean).
        package var followsSide: Bool
        package init(start: Double, end: Double, bias: Double = 0.5, more: Double, most: Double, followsSide: Bool = true) {
            self.start = start; self.end = end; self.bias = bias; self.more = more; self.most = most
            self.followsSide = followsSide
        }
        /// The middle of the schedule (Z-Image's).
        package static let middle = Bell(start: 0.4, end: 0.9, more: 2, most: 2.5)
        /// The last step alone (Qwen-Image-2.1's).
        package static let late = Bell(start: 0.857, end: 1, bias: 0.75, more: 1.3, most: 2)
    }

    /// **The calibrated settings**, by eye at 512², the library prompt, on the families that offer it
    /// at their default steps. `more` ≈ +10 % of high-frequency energy (mean |Laplacian| of the
    /// luminance) and the same picture; `most` the strongest that stays clean, a step short of where
    /// artifacts begin.
    ///
    ///   - Z-Image (8 evaluations, seeds 42, 7, 0): More 2 → +9 / +8 / +11 %, same pose;
    ///     Most 2.5 → +11 / +10 / +15 %, clean (seed 42's hand goes to the hip, as Most's did on
    ///     7 evaluations); 3 streaks seed 0's hair blue. Less headroom than on 7 evaluations, where
    ///     Most gave +20 %: the σ that sharpened then sat between two of today's.
    ///   - Qwen-Image-2.1 (the late bell, seeds 42, 7, 1000): More 1.3 → +5 / +3 / +5 %, Most 2 →
    ///     +7 / +4 / +6 %, same person; 3 cracks the skin and frizzes the hair, 4 breaks it. Less
    ///     than Z-Image's +10 %: the only step that sharpens without redrawing is the last.
    /// No bell, no schedule: a family without one refuses Detail before any computation.
    package func schedule(_ bell: Bell?) -> Schedule? {
        guard let bell, self != .normal else { return nil }
        return Schedule(amount: self == .more ? bell.more : bell.most, start: bell.start, end: bell.end, bias: bell.bias)
    }

    /// **The amount follows the image's side**: calibrated at 512², it is multiplied by
    /// `512 / √(width·height)` — the same lie paints four times the tokens at 1024².
    ///
    ///   - Z-Image: on 7 evaluations and the node's bell, at 1024² More's amount 1 gave Most's
    ///     +20 % and Most's 2 confetti on skin and walls; scaled, More 0.5 and Most 1 clean. Its own
    ///     bell on the 8 (seeds 42, 7, 0, against Normal): More 1 → +7 / +8 / +6 %, Most 1.25 → +9 / +10 / +8 %, same picture,
    ///     clean; 1.5 already flecks seed 42's hair. The rule holds.
    ///   - Qwen-Image-2.1 (late bell, seeds 42, 7): at 1024² the scaled amounts 1, 1.5 and 2 stay clean,
    ///     same face, +6 / +8 / +9 %: Most's 1 and More's 0.65 with room to spare. (On the third-party
    ///     turbo's 6 steps, the rule's origin: 1 strewed white flecks, 0.65 was the last clean.)
    ///   - Anima and Krea 2 (seeds 7, 42): at 1024² the scaled amounts gain little (Anima +1 to +2 %, Krea 2 +1.5 %),
    ///     and the 512² ones gain no more while the picture moves (Anima 23 to 26 dB to Normal, Krea 2 29 dB). The
    ///     rule holds; their Detail is a 512² thing more than a 1024² one.
    ///   - MOSS-Image does not follow it (`Bell.followsSide`).
    ///
    /// Never more than at 512².
    package static func scale(width: Int, height: Int) -> Double {
        min(1, 512 / Double(width * height).squareRoot())
    }

    /// **The σ each evaluation hands the model**, for a full schedule (`steps + 1` values, ending in
    /// 0) of which steps `start…` run. `normal`, or no bell: `sigmas` itself. `scale`: `Detail.scale` of the
    /// render's size (1 at 512²).
    package func modelSigmas(_ sigmas: [Float], start: Int = 0, bell: Bell?, scale: Double = 1) -> [Float] {
        guard let bell, var schedule = schedule(bell), sigmas.count - start >= 2 else { return sigmas }
        if bell.followsSide { schedule.amount *= scale }
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
            var values = NumPyRamp.linspace(a, b, count)
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

    /// The sampler wrapper: the schedule over the steps that run, in float32, then
    /// `σ·max(10⁻⁶, 1 − 0.1·m·cfg)` for each σ > 0. Steps before `start` are left as they are.
    package static func adjusted(_ sigmas: [Float], start: Int, _ s: Schedule, cfgScale: Double = 1) -> [Float] {
        let m = multipliers(steps: sigmas.count - 1 - start, s).map { Float($0) }
        var out = sigmas
        for (k, multiplier) in m.enumerated() where sigmas[start + k] > 0 {
            let factor = Float(max(1e-6, 1 - Double(multiplier) * 0.1 * cfgScale))
            // A negative amount tells more noise than there is (Anima's): never more than all of it.
            out[start + k] = min(sigmas[start + k] * factor, max(1, sigmas[start + k]))
        }
        return out
    }
}

/// **NumPy's `linspace(start, stop, num)`** in double, as `function_base.linspace` computes it:
/// `step = (stop − start)/(num − 1)`, `i·step + start`, the last value set to `stop`. Anima's and
/// Krea 2's pipelines hand it to the scheduler, which casts it to float32; Detail's bell is drawn on it.
package enum NumPyRamp {
    package static func linspace(_ start: Double, _ stop: Double, _ count: Int) -> [Double] {
        guard count > 1 else { return count == 1 ? [start] : [] }
        let step = (stop - start) / Double(count - 1)
        var values = (0..<count).map { Double($0) * step + start }
        values[count - 1] = stop
        return values
    }
}
