import CryptoKit
import Foundation
import IOKit
import IOKit.ps
import Metal

/// **The diagnostic — the Mac's capacity model, measured once per version, in about a minute**: voluntary, not one byte of weights, a JSON a user can
/// share. It reads what the silicon announces, measures only what the system does not say, and writes the numbers the engine
/// plans with (`CapacityModel`): the engine reads **no table of routes**, it asks the plan for exact rows
/// at every product and every attention (`CapacityPlan`), solved from this model.
///
///     inventory      DiagnosticInventory     GPU cores and clock, CPU levels and clusters, the Neural Engine's generation:
///        │                                   read, and the denominator of "reached / peak" — never a decision
///        ├─ probes     CapacityProbes         each engine alone at three sizes, the ANE's rows, attention, the engines in sets,
///        │                                   the weights' preparation, the heat, the memory plan — each route judged against
///        │                                   the exact result (fp64) by the checks' rule
///        ├─ fit        CapacityFit            lines t(W) = t₀ + W / rate, the sets' keep and overhead (pure, replayable offline)
///        └─ findings   DiagnosticFindings     what was missing, aberrant or uncertain, and what the plan does with it
///
/// What sets it apart — what a diagnostic written by analogy with a table of routes timed against each other would get wrong:
///
///   - **It judges the chip, never the weights, and knows no model.** The operands are synthetic, the same on every machine
///     (xorshift by blocks of 2¹⁶): installing, removing or adding a model never changes the report, and a Mac with none
///     measures it whole;
///   - **It measures parameters, not winners.** No route is timed against another to decide a class: the model gives the time of
///     any block on any set of engines, and the solver (`CapacitySolver`) chooses at the shape of each product the render
///     really makes. The Mac need not be at rest for the lines to be right, and a Mac nobody has measured is served by the same
///     lines;
///   - **A route that computes wrong is excluded, and the map stays a map**: against fp64, never against another fp32, with the
///     checks' rule (the fp32 threshold; the ANE within 1.25 × bf16's own rounding — invariant 3). `excluded` names it, the plan
///     never offers it, and the report says which and by how much;
///   - **Every output is poisoned before each engine** (NaN on the sampled rows): a submission that fails silently leaves the
///     previous engine's right answer, which would pass;
///   - **The sizes follow the machine, not its name** (`CapacityProbes.Sizes`): chosen from the inventory's peak, then from the
///     first witnesses' own rates, so that the slowest Mac of the fleet holds the ceiling; each stage's projection is written
///     beside its seconds (`stages`);
///   - **A finished measure is always a map**: applied as it stands, never refused, never asked again. What the probes could not give
///     is said and the plan does without it (a route or a set that is not offered, like one the judge excluded); what is aberrant is
///     bounded or left out, named, with its effect; heat, the battery, Low Power Mode, another process's swap and the cores the others
///     used are facts of the regime, written beside the model (`Findings`);
///   - **It knows how sure each figure is, and reads the quick side of a noisy one** (`Regime`): after each step a pulse of a few
///     milliseconds asks whether as many threads as performance cores still get a core each (against a thread alone, measured at the
///     same moment: a clock that went down is the Mac's, a core that went to another process is not) and whether an empty submission
///     still reaches the GPU promptly, and the calls of a chain are compared with one another. A step that fails is measured again, after
///     a wait, while the ceiling leaves seconds (`Sizes.retryCap`) — and then it is kept: **a disturbance only adds time**, so a chain whose
///     calls disagree is read on its fast side (`CapacityProbes.figure`), how far they disagreed is written beside each engine and set
///     (`Report.uncertainty`), and the Mac need not be at rest: a moving mouse, a browser, Activity Monitor are the Mac's life;
///   - **It runs in a process of its own** (`Child`), because a program the Neural Engine refuses may kill the process that loads
///     it: the child says each stage and each route before it touches it, and a supervisor that sees it die by a crash leaves that
///     route out and measures again — the report says `died`, and no render loads a route the report has not seen load;
///   - **It holds the machine's render lock** (`RenderLock.machine`) and runs on the engine's queue: never beside a render, of
///     this process or another.
///
/// **The disk, as the weight stream reads it** (`CapacityDisk`, 1 to 2 s: in `CapacityModel.diskBandwidth` and the report) — read by the
/// product at one decision, the class of service the stream's readers run at in each phase (`CapacityDisk.streamClass`, from the rates
/// per class `diskClasses`); the stream's overlap is planned by `CapacityStream` and judged by developers'
/// benches (E5). **Deliberately not measured**: a model's step (its weights, its plan and the page cache decide
/// it, and a render's own figures already say it); the memory (B is a function of the RAM alone, ¾ · RAM − 2 GiB); an SDPA in fp16
/// (P20: never under bf16, sums in fp32 — a precision the engine refuses is no capacity).
///
/// **The engine reads it in the app, and only there**: the app measures before its first render when it holds no current report
/// — none yet, one of another `Fingerprint` (hardware, chip, macOS build, the app's build: an update measures again) — and whenever
/// the user asks (« Measure This Mac »). It keeps the report in its preferences and hands it to
/// `EngineSettings.load(from:map:)`, which designates the plan of its model — a later measure replaces it between two renders
/// (`EngineSettings.adopt(map:)`), never during one. The command line, the checks and the close-out
/// never designate one unless a developer names it (`SILICONED_MAP`): today's defaults.
package enum Diagnostic {
    /// The JSON's version: the script that aggregates the reports reads it. **22** (3/11/2026): each stage names the bias its seconds
    /// remove (`StageTime.removes`); the rounded pair carries the ANE's per-call widening of its bf16 copy (`Sharing.perWeight`); the
    /// model carries the map's noise (`CapacityModel.noise`), under which the plan settles by a written order; the rounded solos take
    /// eight shapes; a render's note counts its fallbacks (`RenderLoop.Note.fallbacks`). **21** (3/11/2026): the Standard's own pair — the GPU's
    /// rounded rows on a bf16 weight read as published beside the ANE's fp16 program on the same copy — keeps its own sharing
    /// (`gpu/bf16act+ane/fp16`, stage `sharing rounded`); **20** (3/11/2026): the 8-bit products on an activation
    /// rounded to bf16 (`gpu/int8act`, its tile among candidates; `gpu/q8act`) are measured and judged at the half-precision bar, and
    /// the ANE's integers beside the rounded one keep their own sharing (`gpu/int8act+ane/int8`); **19** (2/11/2026): the attention on bf16 operands
    /// (`gpu/tilesbf16`) is measured beside the others, and each half-precision attention kernel's form — three stages or fused at a
    /// tile — is chosen among candidates and kept in the model (`attentionTiles`, `AttentionTile`); **18** (2/11/2026): the `matmul2d` kernels' tile is measured
    /// among candidates and kept in the model (`tiles`, `MatmulTile`), each candidate's solo in the measurements; **17** (31/10/2026): the attention's
    /// two kernels by tiles (`gpu/tiles`, `gpu/tiles16`) are measured and judged beside the graph, flash and the CPU's heads; **16**: no verdict — `findings`, each with its
    /// effect on the plan, replace the green flag and the defects, and the fit's aberrant figures are bounded; 15 (30/10/2026) was the
    /// capacity model — inventory, the probes' raw measurements, the model they fit, the judge per kernel, the memory, the stages and
    /// a verdict of defects; 14 and before were a table of routes timed against each other (grid of classes, pairs A B A B), which the
    /// engine no longer reads. A report of a schema under 22 is not a map: the app measures again (`isCurrent`).
    package static let schema = 22
    package static let repository = "https://github.com/gwenn-ha-dev/Siliconed"
    /// **Wall seconds the whole diagnostic may take, inventory and witnesses included, on any Mac: a ceiling, not a target** (the
    /// user, 1/11/2026: a Mac pays once per version rather than a few percent at every render). The duration is a result: about forty
    /// seconds on an M1 Pro at the first tier. Under a one-minute budget the tier was chosen by time, and the tier stepped down on the
    /// noise of the first witnesses — the same Mac measured at one tier, then at another, and planned two ways; a Mac small or hot
    /// enough measured less than the plan needs. Now the memory chooses the tier (`Plan.choose`), the time only past the ceiling.
    package static let ceiling = 300.0

    /// **The fp32 threshold**: the AMX check's (`siliconed-dev amx`), a row of an fp32 product or
    /// attention no further than 10⁻⁵ from the exact. Measured on the M1 Pro (14/10/2026): the
    /// products 0.80–2.1·10⁻⁶ (the worst at k = 12 288, GPU and the cut alike), the attention
    /// 2.0–4.4·10⁻⁶ (flash the closest) — 2.3× under it at the worst, and a fault is orders above.
    package static let fp32Threshold = 1e-5
    /// A half-precision output passes within this factor of its rounding floor (invariant 3).
    package static let floorMargin = 1.25

    // ── the run ─────────────────────────────────────────────────────────────────────────────

    /// **The stage the diagnostic is at**, as `run` reports it to `onProgress` — an app says it in its
    /// own language; `description` is the English the command line prints.
    package enum Stage: String, Sendable, CaseIterable, CustomStringConvertible {
        case widening = "weight preparation"
        case witness = "the engines' first measurement"
        case engines = "each engine alone"
        case sharing = "the engines together"
        case forms = "the weight forms"
        case regime = "the engines again"
        package var description: String { rawValue }
        /// Its key on the child process's standard output: short.
        package var key: String {
            switch self {
            case .widening: return "widening"
            case .witness: return "witness"
            case .engines: return "engines"
            case .sharing: return "sharing"
            case .forms: return "forms"
            case .regime: return "regime"
            }
        }
    }

    /// **Runs the diagnostic**: under the ceiling on any Mac (about a minute), no weights, its peak under B (`Plan`). Synchronous and
    /// heavy: call it off the main thread, and never from the engine's queue.
    ///
    /// **It passes the doors a render passes**: it waits its turn on the engine's queue (`Engine.file`:
    /// never beside a render of this process) and holds the machine's render lock (a render of
    /// another process refuses it: `renderAlreadyRunning`). `cancellation` is checked between two
    /// measurements (`cancelled`).
    ///
    /// It knows no model: the same probes on every Mac, whatever is installed. `validate` (a developer's, outside the ceiling) adds the
    /// model's seconds on products it never saw, against a long measure — in `measure`'s probes, not in the report.
    package static func run(cancellation: Cancellation? = nil, validate: Bool = false, skipping: Set<CapacityModel.Kernel> = [],
                           log: ((String) -> Void)? = nil,
                           onProgress: (@Sendable (_ stage: Stage) -> Void)? = nil) throws(EngineError) -> Report {
        try measure(cancellation: cancellation, validate: validate, skipping: skipping, log: log, onProgress: onProgress).report
    }

    /// The same, with the probes' own result beside the report (the fits' details and the validation, which a report does not carry).
    package static func measure(cancellation: Cancellation? = nil, validate: Bool = false, skipping: Set<CapacityModel.Kernel> = [],
                                log: ((String) -> Void)? = nil, onProgress: (@Sendable (_ stage: Stage) -> Void)? = nil)
        throws(EngineError) -> (report: Report, probes: CapacityProbes.Result) {
        dispatchPrecondition(condition: .notOnQueue(Engine.file))
        let result = Engine.file.sync {
            Result { () throws(EngineError) -> (report: Report, probes: CapacityProbes.Result) in
                try runQueued(cancellation: cancellation, validate: validate, skipping: skipping, log: log, onProgress: onProgress)
            }
        }
        return try result.get()
    }

    /// **The probes' kernels chained at the shapes asked** (`CapacityProbes.measureKernels`), on the engine's queue: a bench's entry.
    package static func measureKernels(_ kernels: [String], shapes: [CapacityProbes.Sizes.Shape], chained: Bool = true,
                                       log: ((String) -> Void)? = nil) throws(EngineError) -> [CapacityFit.Measurements.Solo] {
        dispatchPrecondition(condition: .notOnQueue(Engine.file))
        let result = Engine.file.sync {
            Result { () throws(EngineError) -> [CapacityFit.Measurements.Solo] in
                try CapacityProbes.measureKernels(machine: machine(), kernels: kernels, shapes: shapes, chained: chained, log: log)
            }
        }
        return try result.get()
    }

    private static func runQueued(cancellation: Cancellation?, validate: Bool, skipping: Set<CapacityModel.Kernel>, log: ((String) -> Void)?,
                                  onProgress: (@Sendable (_ stage: Stage) -> Void)?)
        throws(EngineError) -> (report: Report, probes: CapacityProbes.Result) {
        let started = Date()
        let machine = machine()
        let before = conditions()
        let probes = try CapacityProbes.run(machine: machine, cancellation: cancellation, validate: validate, skipping: skipping, log: log,
                                            onStage: { onProgress?($0) })
        let after = conditions()
        var forced: [String: String] = [:]
        for row in EngineSettings.effective.rows where row.source != .byDefault { forced[row.key] = "\(row.value) (\(row.source.rawValue))" }
        let report = Report(date: ISO8601DateFormatter().string(from: started), seconds: Date().timeIntervalSince(started),
                            machine: machine, cpuFeatures: cpuFeatures(), settings: forced, before: before, after: after, probes: probes)
        return (report, probes)
    }
}
