import Foundation

/// **Green or orange, and what exactly is missing or aberrant** — the verdict on a report's measures. Pure: the machine's inventory, the
/// model, the raw measurements, the judge, the memory, the two readings of the conditions and the seconds go in; a list of defects comes
/// out. Orange is a defect of the *measure*, said precisely (which kernel, which set, which stage's figure): never the hour, never an
/// engine that computed wrong (that one is excluded, `Report.excluded`).
///
///     green  =  every engine the inventory announces has its lines         (a kernel without a line must be one the judge excluded)
///            ∧  every set of engines that exists has its sharing            (a keep for each member, an overhead — all the sets the solver may offer)
///            ∧  nothing aberrant                                             (rules below, each with the figures it stands on)
///            ∧  the Mac was not busy while it was measured                    (what the guard could not measure again: `Regime.disturbances`)
///            ∧  peak ≤ base + need ∧ base + need ≤ B                          (the memory held)
///            ∧  swap in use did not grow                                     (the probes pushed nothing out)
///            ∧  seconds ≤ the minute + 2 %
///
/// The aberrant-measure rules, and what they were set from (the M1 Pro in three runs of the probes; the nine Macs of the fleet):
///
///   - **A line's intercept t₀ not under −¼ of the smallest point it was fitted on.** The fit clamps a negative t₀ to 0 and refits through
///     the origin, so the clamped line never shows it; the raw intercept does. Healthy lines of two points that are nearly proportional to
///     W come out at −0.05 to 0 of their smallest point (the system graph +1 to 0 % and the CPU's heads −5 to −1 %, in 12 ms), the
///     others at +0.08 to +0.61: a quarter below zero is five times the worst healthy figure, and a point is no longer on an affine line.
///   - **A rate within [0.05, 1.2] × the peak the inventory read, for an engine that has one** (the GPU's fp32, from its cores and clock).
///     The GPU's fp32 kernels reached 0.74–0.76 of it here and 0.55–0.80 on the fleet (0.35 on a throttled M3 Max): under 0.05 a kernel
///     did not run on the silicon, over 1.2 the clock read is wrong or the timing is. The 1.2 holds for the kernels that compute with
///     the shader cores' fp32 FMA alone (`fp32`, the system graph, the flash kernel); the tensor forms (bf16, int8, Q8_0, rounded
///     activations) may run on matrix units of their own, above the fp32 peak, and only the lower bound applies. An engine with no peak
///     read (the AMX, the ANE, the CPU: the inventory has none for them) has no bound — the ANE's fp16 reached 1.06 × the GPU's fp32
///     peak here, a different silicon.
///   - **A keep within [0.1, 1.25]**: the fleet's GPU kept 0.34–1.03 of its solo rate in a set (0.48–0.95 here; the AMX beside the ANE 0.77–0.95); the fit clamps at
///     [0.05, 1.5], and a keep at its clamp is a side that was not timed, not a measure.
///   - **The Mac was busy, and the minute could not redo it** (`disturbed`): after a step a pulse asks whether as many threads as the
///     performance cores still get a core each (the median thread over one thread alone taken at the same moment: 1.04 on a free M1 Pro, 1.08 at
///     most over 725 pulses, 1.34–1.47 beside four and ten busy loops), whether an empty submission still reaches the GPU promptly (0.22 ms,
///     0.29 at most; 0.55 ms beside a GPU loop of another process, 1.6–10 ms beside ten busy loops), and whether identical calls agree (a
///     chain's p90 − p10 over its median: 0.02–0.06 on a free Mac, now and then 0.16–0.22 on an AMX chain at a depth of 12 288, 0.27–1.07 beside
///     four loops, a GPU loop or a burst). The step is measured again, after a wait for a quiet pulse; the defect is what stayed busy at the third
///     time or when the minute had no seconds left. It names what it saw — the threads, the GPU's wait, the calls — and the steps; the cores the
///     others used are written beside it (`Regime.foreign`) for whoever reads the report, never judged: a load is not an effect (a compilation
///     of two or three cores moved the AMX line by +11 and +17 % with every reading at its free value, and one core by nothing).
///   - **A staircase unit of the ANE not over four of the engine's own (1 024 rows).** The fit takes the *largest* unit that explains the
///     stairs as well as the best; on stairs read through a disturbed minute it came out at 2 048 — the whole range sampled — where it
///     reads 256 (or 128) on a quiet one. A unit that large rounds every row count the model asks of the ANE up to the same step.
extension Diagnostic {
    /// **The verdict**: green, or the list of defects (see the file's header for the rules and the figures they stand on).
    package struct Verdict: Codable, Sendable, Equatable {
        package var green: Bool
        package var defects: [Defect]

        /// One thing the measure lacks or got aberrant, said precisely.
        package struct Defect: Codable, Sendable, Equatable {
            package enum Kind: String, Codable, Sendable { case line, sharing, aberrant, memory, swap, time, disturbed }
            package var kind: Kind
            /// The kernel (`ane/fp16`), the set (`gpu+ane`) or the quantity (`peak`, `swap`, `seconds`) concerned.
            package var subject: String
            package var detail: String
            package init(_ kind: Kind, _ subject: String, _ detail: String) { self.kind = kind; self.subject = subject; self.detail = detail }
            package var text: String { "\(subject) \(detail)" }
        }

        package init(green: Bool, defects: [Defect]) { self.green = green; self.defects = defects }

        /// The thresholds of the rules (the header says what each stands on).
        package enum Rules {
            package static let interceptFloor = -0.25
            package static let rateFloor = 0.05, rateCeiling = 1.2
            package static let keepRange = 0.1...1.25
            /// The ANE's staircase unit, in rows: past four of the engine's own units (1 024) the fit has read noise, not steps.
            package static let maximumANEUnit = 4 * NeuralEngine.rowUnit
            /// The minute, with the 2 % the clock and the report's own assembly take.
            package static let timeAllowance = 1.02
        }

        /// **A disturbance in words**: what the guard saw, how many steps, and the worst of them — for the report.
        static func defect(_ d: Regime.Disturbance) -> Defect {
            let steps = d.steps.count == 1 ? "1 step" : "\(d.steps.count) steps"
            switch d.kind {
            case .threads:
                return Defect(.disturbed, "threads", String(format: "got %.0f %% of their cores at ", 100 / d.worst) + "\(steps) (worst: \(d.worstStep), the median thread \(Diagnostic.format(d.worst)) × a thread alone)")
            case .gpu:
                return Defect(.disturbed, "gpu", String(format: "kept an empty submission waiting %.2f ms at ", d.worst * 1e3) + "\(steps) (worst: \(d.worstStep))")
            case .calls:
                return Defect(.disturbed, "calls", String(format: "of identical work differed by %.0f %% at ", d.worst * 100)
                              + "\(steps) (worst: \(d.worstStep), over \(Int(Regime.Rules.spread * 100)) %)")
            }
        }

        /// What the app says: « The measure is not complete: ane/fp16 has no line, swap grew. »
        package var reasons: [String] { defects.map(\.text) }

        // MARK: What the inventory announces

        /// **The engines the inventory announces**: the AMX where the CPU has a performance level, the Neural Engine where `ane0` is
        /// in the registry. The GPU and the CPU's heads are always there (a Metal device is what the probes run on).
        package static func announces(_ machine: Machine) -> (amx: Bool, ane: Bool) {
            (amx: machine.inventory?.cpuLevels?.contains { $0.cores > 0 } == true, ane: machine.inventory?.ane != nil)
        }

        /// The product kernels and attention kernels without which the plan has nothing to plan with, for the engines announced.
        package static func required(_ machine: Machine) -> [CapacityModel.Kernel] {
            let a = announces(machine)
            var kernels = [CapacityModel.Kernel(.gpu, .fp32), CapacityModel.Kernel(.gpu, .graph), CapacityModel.Kernel(.cpu, .heads)]
            if a.amx { kernels.append(CapacityModel.Kernel(.amx, .fp32)) }
            if a.ane { kernels.append(CapacityModel.Kernel(.ane, .fp16)) }
            return kernels
        }

        /// **Every set of two engines or more the solver may offer**, in `CapacityModel.key` order: the announced product engines in every
        /// combination (the AMX beside the ANE without the GPU included), and the attention's graph beside the CPU's heads.
        package static func sets(_ machine: Machine) -> [[CapacityModel.Engine]] {
            let a = announces(machine)
            var sets: [[CapacityModel.Engine]] = []
            if a.amx { sets.append([.gpu, .amx]) }
            if a.ane { sets.append([.gpu, .ane]) }
            if a.amx && a.ane { sets.append([.gpu, .amx, .ane]); sets.append([.amx, .ane]) }
            sets.append([.gpu, .cpu])
            return sets
        }

        // MARK: The rules

        /// **The verdict on a measure.**
        package init(model: CapacityModel, measurements: CapacityFit.Measurements, judge: [String: Judge], machine: Machine,
                     memory: Memory, before: Conditions, after: Conditions, seconds: Double, regime: Regime? = nil) {
            let excluded = Diagnostic.excluded(judge)
            var defects: [Defect] = []

            // Every announced engine has its lines.
            let reference: Set<CapacityModel.Kernel> = [CapacityModel.Kernel(.gpu, .fp32), CapacityModel.Kernel(.gpu, .graph)]
            for kernel in Self.required(machine) {
                let name = kernel.description
                if model.lines[kernel] != nil { continue }
                if reference.contains(kernel) {
                    defects.append(Defect(.line, name, excluded.contains(kernel) ? "was judged wrong: nothing can be planned without it" : "has no line"))
                } else if !excluded.contains(kernel) {
                    defects.append(Defect(.line, name, "has no line"))
                }
            }
            // Every set that exists has its sharing.
            for set in Self.sets(machine) {
                let kernels = Diagnostic.kernels(of: set)
                guard kernels.allSatisfy({ model.lines[$0] != nil && !excluded.contains($0) }) else { continue }
                let key = CapacityModel.key(set)
                guard let sharing = model.sharing[key], set.allSatisfy({ sharing.keep[$0] != nil }) else {
                    defects.append(Defect(.sharing, key, "has no sharing"))
                    continue
                }
                for engine in set {
                    if let keep = sharing.keep[engine], !Rules.keepRange.contains(keep) {
                        defects.append(Defect(.aberrant, key, "keeps \(Diagnostic.format(keep)) of the \(engine.rawValue)'s rate"))
                    }
                }
            }
            // Nothing aberrant.
            for (kernel, line) in model.lines.sorted(by: { $0.key.description < $1.key.description }) {
                let name = kernel.description
                if let peak = model.crest[kernel.engine], peak > 0 {
                    let ratio = line.rate / peak
                    let fmaOnly = kernel.engine == .gpu && [.fp32, .graph, .flash].contains(kernel.form)
                    if ratio < Rules.rateFloor {
                        defects.append(Defect(.aberrant, name, "reaches \(Diagnostic.format(ratio)) of the peak read (under \(Rules.rateFloor))"))
                    } else if fmaOnly && ratio > Rules.rateCeiling {
                        defects.append(Defect(.aberrant, name, "reaches \(Diagnostic.format(ratio)) of the peak read (over \(Rules.rateCeiling))"))
                    }
                }
                let points = measurements.solos.filter { $0.kernel == name && $0.seconds > 0 }
                    .map { CapacityFit.Point(flop: $0.flop, seconds: $0.seconds) }
                if let raw = CapacityFit.rawIntercept(points), let smallest = points.map(\.seconds).min(),
                   raw < Rules.interceptFloor * smallest {
                    defects.append(Defect(.aberrant, name, "has a fixed cost of \(String(format: "%.2f", raw * 1e3)) ms, below zero"))
                }
            }
            if model.lines[CapacityModel.Kernel(.ane, .fp16)] != nil, model.aneRowUnit > Rules.maximumANEUnit {
                defects.append(Defect(.aberrant, "ane/fp16", "has a staircase of \(model.aneRowUnit) rows (over \(Rules.maximumANEUnit))"))
            }
            // The Mac was not busy while it measured — what was measured again is not a defect, what stayed busy is.
            for d in regime?.disturbances ?? [] { defects.append(Self.defect(d)) }
            // The memory held, no swap, the minute.
            if !memory.held {
                if memory.peak > memory.base + memory.need {
                    defects.append(Defect(.memory, "peak", "\(Diagnostic.megabytes(memory.peak)) is over the need \(Diagnostic.megabytes(memory.base + memory.need))"))
                } else {
                    defects.append(Defect(.memory, "need", "\(Diagnostic.megabytes(memory.base + memory.need)) is over B \(Diagnostic.megabytes(memory.budget))"))
                }
            }
            if after.swapUsedBytes > before.swapUsedBytes {
                defects.append(Defect(.swap, "swap", "grew from \(Diagnostic.megabytes(before.swapUsedBytes)) to \(Diagnostic.megabytes(after.swapUsedBytes))"))
            }
            if seconds > Diagnostic.minute * Rules.timeAllowance {
                defects.append(Defect(.time, "seconds", "took \(String(format: "%.1f", seconds)) s of \(Int(Diagnostic.minute))"))
            }
            self.init(green: defects.isEmpty, defects: defects)
        }
    }

    /// **The kernels a plan leaves out**, from the judge: those it refused, and — for a set of engines it refused as a whole — the engines
    /// beside the GPU (the plan would take the pair for free: no sharing was fitted from a set that computed wrong).
    package static func excluded(_ judge: [String: Judge]) -> Set<CapacityModel.Kernel> {
        var kernels = Set<CapacityModel.Kernel>()
        for (name, judged) in judge where judged.excluded {
            if let kernel = CapacityModel.Kernel(name) { kernels.insert(kernel); continue }
            let set = name.split(separator: "+").compactMap { CapacityModel.Engine(rawValue: String($0)) }
            kernels.formUnion(Diagnostic.kernels(of: set).filter { $0.engine != .gpu })
        }
        return kernels
    }

    /// **The kernels a set of engines runs**: the products' (fp32 on the GPU and the AMX, fp16 on the ANE), or — with the CPU — the attention's
    /// (the graph beside the heads).
    package static func kernels(of set: [CapacityModel.Engine]) -> [CapacityModel.Kernel] {
        set.map { engine in
            switch engine {
            case .gpu: return CapacityModel.Kernel(.gpu, set.contains(.cpu) ? .graph : .fp32)
            case .amx: return CapacityModel.Kernel(.amx, .fp32)
            case .ane: return CapacityModel.Kernel(.ane, .fp16)
            case .cpu: return CapacityModel.Kernel(.cpu, .heads)
            }
        }
    }

    static func format(_ x: Double) -> String { String(format: "%.2f", x) }
    static func megabytes(_ bytes: Int) -> String { String(format: "%.0f MB", Double(bytes) / 1e6) }
}
