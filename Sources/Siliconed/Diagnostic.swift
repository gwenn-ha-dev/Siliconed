import Accelerate
import CryptoKit
import Foundation
import IOKit
import IOKit.ps
import Metal

/// **The diagnostic** — what any user runs from the app, in ~15 s per model, and whose JSON fills the
/// table of measurements by chip (M1 → M5, Pro/Max/Ultra, 16 → 192 GB) that only users can fill.
///
///     for each installed model passed (the CLI: all of them; the app: the visible ones), at 512²:
///
///     prompt ─ text encoder ─ conditioning ─────────────────────────────┐            (timed, cold: no RenderCache)
///     x_t.0, σ₀ ─ DiT ─ model_out.0   1st evaluation: builds the DiT, the LoRA, the prefill — the warm-up
///     x_t.1, σ₁ ─ DiT ─ model_out.1   2nd evaluation: the step in steady state ─▶ channelError vs golden
///     x̂₀ = the one-step estimate ─ VAE decoder ─ image                              (timed, then dropped)
///
///     estimated render = encoder + 1st + (E − 1) · 2nd + decoder,   E = the plan's evaluations at the default steps
///
///     and the machine: chip, P/E cores, GPU cores, memory, macOS, binary version;
///     the peak `phys_footprint` sampled every 10 ms, the `vm_stat` swap-outs before/after.
///
/// **Why two evaluations, not one.** The first pays what a render pays once: the DiT's arenas, the
/// LoRA stack opened, Qwen-Image-2.1's conditions prefilled inside the first step
/// (`QwenImage21DiT.prefillPending`), the map's head entering the page cache, the GPU pipelines
/// compiled. The second is what every later step costs. One alone would mix the two; their ratio is
/// itself a figure of the machine (an 8 GB Mac re-reading its map at each step shows it here).
///
/// **Why the full render is derived, not measured.** At 512², a Qwen render is 6 evaluations and a
/// Z-Image one 7: running them would triple the diagnostic for figures the derivation gives within the
/// noise of a run (M1 Pro: a step varies by a few %, the machine drifts 4–13 % over a few renders).
/// The JSON says it: `estimatedRenderSeconds`, never "render seconds".
/// The step count is the model's default (`defaultSteps`); `E` comes from the denoiser's own plan,
/// so Z-Image's null last step (8 steps = 7 evaluations) is not counted.
///
/// **What is judged.** The second `model_out`, channel by channel (`channelError`, the same measure
/// as every check of the repository), against a **reduced golden embedded as a resource** — the DiT's
/// input and output at two steps of the oracle's trajectory, ~1 MB, not the oracle's 104 MB. The
/// conditioning is NOT embedded: it is recomputed by the user's encoder, so the deviation is that of
/// the whole text → DiT path at real inputs, larger than `qwen21-forward`'s at the oracle's inputs.
/// A model without a golden (the four families the diagnostic does not drive, every imported
/// DiT) still runs — timings, machine, peak, swap — and says `"golden": null`, `"deviation": null`.
///
/// **The golden of Qwen-Image-2.1** (`Resources/diagnostic/qwen-image-2.1.safetensors`, 1.05 MB):
///   - from the oracle's `goldens-qwen21-trajectory-512.safetensors` (sha256 `bf46a1089c03340e…`,
///     written on 03/10/2026: diffusers fp32 on CPU, base `d26bb612`, Viggle turbo
///     `009a44a8` at strength 1, prompt « a 30 year old woman posing in a library », seed 0, 6 steps);
///   - keys `latent_init` → `x_t.0`, `velocity_0` → `model_out.0`, `latent_1` → `x_t.1`, `velocity_1` →
///     `model_out.1` (the transformer's output on the target rows, packed `[1024, 64]`, stored planar
///     `[64, 32, 32]`), `sigmas[0:2]` → `sigma` = [1.0, 0.96255648] (the scheduler's σ, which
///     `QwenImage21DiT.forward` receives);
///   - step 0 is bit for bit `goldens-qwen21-dit-lora-512`'s `model_out` (the input of the turbo LoRA check at 512²);
///   - regenerated on 04/10/2026 from the oracle's goldens.
///
/// **The golden of Z-Image Turbo** (`Resources/diagnostic/z-image.safetensors`, 1.05 MB):
///   - from the oracle's `goldens-trajectory-512.safetensors` (sha256 `c3157e2da4be5b0a…`, written
///     on 04/10/2026: diffusers fp32 on CPU, the `goldens-text`
///     `cap_feats` of « a 30 year old woman posing in a library », seed 42, 8 steps);
///   - keys `latent_init` → `x_t.0`, `latent_1` → `x_t.1`, and **`−v_0` → `model_out.0`, `−v_1` →
///     `model_out.1`**: the oracle stores the pipeline's `noise_pred = −transformer(…)`, `DiT.forward`
///     returns the transformer's raw output;
///   - `sigma` = [0, 0.05263153]: the **reversed time** the oracle's transformer received,
///     `(1000 − t)/1000` in fp32 with `t = scheduler.timesteps` — not fp32 `1 − σ` (0.05263156), which
///     is what the product's `FlowMatchSchedule.modelTime` passes; 2.6·10⁻⁸ apart, read the oracle's;
///   - regenerated the same way.
///
/// **The threshold** (`threshold(for:)`): see there — measured on this machine at real inputs, and
/// wide enough for another chip's GPU/AMX cut, narrow enough that a wrong kernel cannot hide.
///
/// **Every field is there to decide something in this repository** (schema 2, 08/10/2026) — a report
/// from a machine we have never touched, read without Xcode on it, must say which line of code to change:
///   - `memory` per model: the budget at the launch, the floor, and **the memory plan's decisions**. The
///     diagnostic runs inside `MemoryPlan.during` like a render, so an 8 GB Mac takes the lean plan here
///     exactly where its renders would — before schema 2 it ran outside any render, on the default plan
///     a real render there would not have taken, and its estimate was another machine's. A refusal
///     carries the same figures: the 8 GB reserve (`MemoryBudget.reserve`) is judged on them;
///   - `diskReadBytes` per model: the map read from the disk or from the page cache — the 11–12 % measured
///     on a loaded desktop was that, and nothing else in a report says it;
///   - `witness`: an fp32 GEMM `[511 × 3840] × [3840 × 3840]` on the GPU alone and through `cblas_sgemm`
///     (the AMX), **before the first model and after the last**. Their ratio is what decides whether
///     `EngineSettings.amx` goes on for this chip and at which `amxFraction` (it is on for `referenceChip`
///     only); the GPU's rows against `cblas` check a new GPU family's fp32 without one byte of weights;
///     the before/after pair is the thermal witness that frames any timing, on a machine nobody watches;
///   - `conditions` before/after: thermal state, Low Power Mode, battery, reclaimable memory, the
///     compressor's room, the swap in use — a fanless Mac or a loaded one says so instead of a slow chip;
///   - `settings`: every setting that is not the reference's default, with its provenance, and `amx`
///     always — a profile or an environment variable on the user's side must not pass for the chip;
///   - `machine.gpu*`: the Metal device, its highest `apple` family and its working set — the kernels'
///     choices (flash, fp32 accumulators) are properties of the family, not of the core count.
///
/// What the diagnostic does NOT do: read or write the render cache (a kept conditioning would hide
/// the encoder), apply a user LoRA, run at the product's 1024² (the 512² step is what every machine
/// can afford in ~30 s; 1024² is ~4× the tokens). The OS page cache
/// is out of its hands: a model just used renders its encoder warmer than a cold boot would.
public enum Diagnostic {
    public static let resolution = 512
    public static let prompt = "a 30 year old woman posing in a library"
    public static let seed: UInt64 = 42
    /// The JSON's version: the script that aggregates the reports reads it.
    public static let schema = 2
    public static let repository = "https://github.com/gwenn-ha-dev/Siliconed"

    // ── the report ──────────────────────────────────────────────────────────────────────────

    public struct Machine: Codable, Sendable, Equatable {
        /// `machdep.cpu.brand_string`: "Apple M1 Pro".
        public var chip: String
        /// `hw.model`: "MacBookPro18,3".
        public var hardwareModel: String
        public var performanceCores: Int
        public var efficiencyCores: Int
        /// IORegistry's `gpu-core-count` (`AGXAccelerator`); 0 if unreadable.
        public var gpuCores: Int
        public var memoryBytes: Int
        /// "26.0.1 (25A…)".
        public var macOS: String
        /// `CFBundleShortVersionString (CFBundleVersion)` of the app, else the git commit of the
        /// repository the binary was built in (`-dirty` if modified), else "unknown".
        public var version: String
        /// The executable's fingerprint: the first 12 hex digits of the SHA-256 of `RenderCache.build`
        /// (path, size, date) — two builds of one commit differ here, and no local path goes public.
        public var build: String
        /// `MTLDevice.name`: "Apple M1 Pro".
        public var gpu: String
        /// The highest `MTLGPUFamily.appleN` the device supports, "apple7" on the M1 family; "" if none.
        public var gpuFamily: String
        /// `MTLDevice.recommendedMaxWorkingSetSize`: what Metal lets the process keep resident.
        public var gpuWorkingSetBytes: Int

        public init(chip: String, hardwareModel: String, performanceCores: Int, efficiencyCores: Int,
                    gpuCores: Int, memoryBytes: Int, macOS: String, version: String, build: String,
                    gpu: String = "", gpuFamily: String = "", gpuWorkingSetBytes: Int = 0) {
            self.chip = chip; self.hardwareModel = hardwareModel; self.performanceCores = performanceCores
            self.efficiencyCores = efficiencyCores; self.gpuCores = gpuCores; self.memoryBytes = memoryBytes
            self.macOS = macOS; self.version = version; self.build = build
            self.gpu = gpu; self.gpuFamily = gpuFamily; self.gpuWorkingSetBytes = gpuWorkingSetBytes
        }

        /// "16 GB": the spec sheet's figure (2³⁰), what a reader of the table compares.
        public var memoryLabel: String { "\(Int((Double(memoryBytes) / 1_073_741_824).rounded())) GB" }
    }

    /// Wall-clock seconds of each stage.
    public struct Seconds: Codable, Sendable, Equatable {
        /// Tokenizer + text encoder, cold.
        public var encoder: Double
        /// The first evaluation, with everything it builds (DiT, LoRA, prefill).
        public var firstEvaluation: Double
        /// The second: the step in steady state.
        public var steadyEvaluation: Double
        public var decoding: Double
        /// The model's whole diagnostic, building the model included.
        public var total: Double

        public init(encoder: Double, firstEvaluation: Double, steadyEvaluation: Double, decoding: Double, total: Double) {
            self.encoder = encoder; self.firstEvaluation = firstEvaluation; self.steadyEvaluation = steadyEvaluation
            self.decoding = decoding; self.total = total
        }
    }

    /// Where the golden comes from, as its file says.
    public struct GoldenInfo: Codable, Sendable, Equatable {
        public var file: String
        public var source: String
        public var sourceSHA256: String
        public var generated: String
        public init(file: String, source: String, sourceSHA256: String, generated: String) {
            self.file = file; self.source = source; self.sourceSHA256 = sourceSHA256; self.generated = generated
        }
    }

    /// The second `model_out` against the golden, channel by channel (`channelError`).
    public struct Deviation: Codable, Sendable, Equatable {
        public var worst: Double
        public var channel: Int
        public var median: Double
        public var threshold: Double
        public var pass: Bool
        /// The first evaluation's worst channel — for information, not judged.
        public var firstEvaluationWorst: Double
        public init(worst: Double, channel: Int, median: Double, threshold: Double, pass: Bool, firstEvaluationWorst: Double) {
            self.worst = worst; self.channel = channel; self.median = median; self.threshold = threshold
            self.pass = pass; self.firstEvaluationWorst = firstEvaluationWorst
        }
    }

    /// **The memory plan a render would have taken**, and what it decided (`MemoryPlan.Render`).
    public struct MemoryReport: Codable, Sendable, Equatable {
        /// `MemoryBudget` at the launch: what the machine could give back, and that minus the reserve.
        public var reclaimableBytes: Int
        public var availableBytes: Int
        public var reserveBytes: Int
        /// `ModelCard.memoryNeed` / `memoryComfort` at the diagnostic's format.
        public var floorBytes: Int
        public var comfortableBytes: Int
        /// The lean plan was taken (at the launch, or when the compressor ran short).
        public var lean: Bool
        /// Each site's decision, in order — the lines the developer's command line prints.
        public var decisions: [String]
        public init(reclaimableBytes: Int, availableBytes: Int, reserveBytes: Int, floorBytes: Int,
                    comfortableBytes: Int, lean: Bool, decisions: [String]) {
            self.reclaimableBytes = reclaimableBytes; self.availableBytes = availableBytes; self.reserveBytes = reserveBytes
            self.floorBytes = floorBytes; self.comfortableBytes = comfortableBytes; self.lean = lean; self.decisions = decisions
        }
    }

    /// **The state of the machine**, read before the first model and after the last.
    public struct Conditions: Codable, Sendable, Equatable {
        /// `ProcessInfo.thermalState`: nominal, fair, serious, critical. A fanless Mac throttles from fair.
        public var thermal: String
        public var lowPowerMode: Bool
        /// The machine runs on its battery; `nil` when the power source does not say.
        public var onBattery: Bool?
        /// `MemoryBudget.reclaimable`, the process's own footprint deducted.
        public var reclaimableBytes: Int
        /// `MemoryPressure.headroom`: the compressor's room before the kernel swaps; `nil` if unreadable.
        public var compressorHeadroomBytes: Int?
        /// `vm.swapusage`'s used bytes: swap another process left, not ours (ours is `swapouts`).
        public var swapUsedBytes: Int
        public init(thermal: String, lowPowerMode: Bool, onBattery: Bool?, reclaimableBytes: Int,
                    compressorHeadroomBytes: Int?, swapUsedBytes: Int) {
            self.thermal = thermal; self.lowPowerMode = lowPowerMode; self.onBattery = onBattery
            self.reclaimableBytes = reclaimableBytes; self.compressorHeadroomBytes = compressorHeadroomBytes
            self.swapUsedBytes = swapUsedBytes
        }
    }

    /// **The chip's fp32 GEMM, GPU and AMX, without a model** (`witness()`).
    public struct Witness: Codable, Sendable, Equatable {
        /// MPS, the GPU alone (below the conductor's `amx_min`), median of five.
        public var gpuTFLOPS: Double
        /// `cblas_sgemm`, Accelerate's own threads — the AMX blocks; median of five.
        public var amxTFLOPS: Double
        /// The GPU's worst row against `cblas` (relative): ≤ 10⁻⁵ on the M1 Pro (the AMX check's threshold).
        public var gpuWorstRowError: Double
        public init(gpuTFLOPS: Double, amxTFLOPS: Double, gpuWorstRowError: Double) {
            self.gpuTFLOPS = gpuTFLOPS; self.amxTFLOPS = amxTFLOPS; self.gpuWorstRowError = gpuWorstRowError
        }
    }

    public struct ModelReport: Codable, Sendable, Equatable {
        public var model: String
        public var name: String
        public var family: String
        /// The model's default step count, and the DiT evaluations they make (`DenoisingPlan`).
        public var steps: Int
        public var evaluations: Int
        public var seconds: Seconds?
        /// `encoder + firstEvaluation + (evaluations − 1) · steadyEvaluation + decoding` — derived, not measured.
        public var estimatedRenderSeconds: Double?
        /// The peak `phys_footprint` of the process during this model's diagnostic.
        public var peakFootprintBytes: Int
        /// `vm_stat` swap-outs (pages) during this model's diagnostic: zero, or a defect.
        public var swapouts: Int
        /// `ri_diskio_bytesread` during this model: its maps read from the disk, not from the page cache.
        public var diskReadBytes: Int = 0
        /// Standard or Compact (8-bit) — `nil` for an imported DiT.
        public var variant: String? = nil
        /// The budget, the floor and the plan; `nil` when the model did not get as far as reading them.
        public var memory: MemoryReport? = nil
        /// `ProcessInfo.thermalState` once this model is done.
        public var thermalAfter: String = ""
        public var golden: GoldenInfo?
        public var deviation: Deviation?
        /// Why the model did not run to the end; `nil` when it did.
        public var error: String?
        /// The same failure as the engine's case, for an app to say it in its own words (`error` is
        /// the technical text an issue quotes). Not in the JSON: `nil` in a report read back.
        public var failure: EngineError? = nil

        private enum CodingKeys: String, CodingKey {
            case model, name, family, steps, evaluations, seconds, estimatedRenderSeconds, peakFootprintBytes,
                 swapouts, diskReadBytes, variant, memory, thermalAfter, golden, deviation, error
        }

        public init(model: String, name: String, family: String, steps: Int, evaluations: Int, seconds: Seconds?,
                    estimatedRenderSeconds: Double?, peakFootprintBytes: Int, swapouts: Int,
                    golden: GoldenInfo?, deviation: Deviation?, error: String?, failure: EngineError? = nil) {
            self.model = model; self.name = name; self.family = family; self.steps = steps
            self.evaluations = evaluations; self.seconds = seconds; self.estimatedRenderSeconds = estimatedRenderSeconds
            self.peakFootprintBytes = peakFootprintBytes; self.swapouts = swapouts; self.golden = golden
            self.deviation = deviation; self.error = error; self.failure = failure
        }

        /// **Explicit `null`s**: the synthesized encoder omits a `nil`, and a reader (the aggregation
        /// script, a human) must tell "no golden" from "an older schema without the field".
        public func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(model, forKey: .model); try c.encode(name, forKey: .name); try c.encode(family, forKey: .family)
            try c.encode(steps, forKey: .steps); try c.encode(evaluations, forKey: .evaluations)
            try c.encode(seconds, forKey: .seconds); try c.encode(estimatedRenderSeconds, forKey: .estimatedRenderSeconds)
            try c.encode(peakFootprintBytes, forKey: .peakFootprintBytes); try c.encode(swapouts, forKey: .swapouts)
            try c.encode(diskReadBytes, forKey: .diskReadBytes); try c.encode(variant, forKey: .variant)
            try c.encode(memory, forKey: .memory); try c.encode(thermalAfter, forKey: .thermalAfter)
            try c.encode(golden, forKey: .golden); try c.encode(deviation, forKey: .deviation)
            try c.encode(error, forKey: .error)
        }
    }

    public struct Report: Codable, Sendable, Equatable {
        public var schema: Int
        /// ISO 8601, UTC.
        public var date: String
        public var resolution: Int
        public var prompt: String
        public var seed: UInt64
        public var machine: Machine
        public var models: [ModelReport]
        /// The whole diagnostic, every model.
        public var totalSeconds: Double
        public var peakFootprintBytes: Int
        public var swapouts: Int
        /// The machine before the first model and after the last.
        public var before: Conditions?
        public var after: Conditions?
        /// The chip's GEMM before the first model and after the last: their ratio is the thermal drift.
        public var witnessBefore: Witness?
        public var witnessAfter: Witness?
        /// Every setting that is not the default, `"value (provenance)"`, and `amx` always.
        public var settings: [String: String]
        /// The profile read, refused, or none (`EngineSettings.profileStatus`), the home folder elided.
        public var profile: String
        /// The library sits on the internal disk; `nil` when the volume does not say.
        public var libraryOnInternalDisk: Bool?

        public init(schema: Int = Diagnostic.schema, date: String, resolution: Int = Diagnostic.resolution,
                    prompt: String = Diagnostic.prompt, seed: UInt64 = Diagnostic.seed, machine: Machine,
                    models: [ModelReport], totalSeconds: Double, peakFootprintBytes: Int, swapouts: Int,
                    before: Conditions? = nil, after: Conditions? = nil, witnessBefore: Witness? = nil,
                    witnessAfter: Witness? = nil, settings: [String: String] = [:], profile: String = "",
                    libraryOnInternalDisk: Bool? = nil) {
            self.schema = schema; self.date = date; self.resolution = resolution; self.prompt = prompt
            self.seed = seed; self.machine = machine; self.models = models; self.totalSeconds = totalSeconds
            self.peakFootprintBytes = peakFootprintBytes; self.swapouts = swapouts
            self.before = before; self.after = after; self.witnessBefore = witnessBefore; self.witnessAfter = witnessAfter
            self.settings = settings; self.profile = profile; self.libraryOnInternalDisk = libraryOnInternalDisk
        }

        /// **Stable JSON**: sorted keys, so two reports diff line by line. `pretty: false` for the
        /// issue URL, where every byte counts.
        public func json(pretty: Bool = true) throws -> Data {
            let encoder = JSONEncoder()
            encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
                                              : [.sortedKeys, .withoutEscapingSlashes]
            return try encoder.encode(self)
        }

        /// **"Report my configuration"**: a new GitHub issue from the `diagnostic.yml` form, its fields
        /// prefilled by their ids — `chip`, `memory`, `macos`, `version` and `report`, the whole JSON,
        /// which the issue's body then carries (the script that aggregates the reports reads it back from there).
        /// `includingReport: false` leaves `report` empty, the other fields filled: for a JSON too long
        /// for a link (browsers and GitHub cut beyond ~8 000 characters), pasted by the user instead.
        public func issueURL(includingReport: Bool = true) throws -> URL {
            var fields = [
                ("template", "diagnostic.yml"),
                ("labels", "diagnostic"),
                ("title", "Diagnostic: \(machine.chip) · \(machine.memoryLabel) · macOS \(machine.macOS)"),
                ("chip", machine.chip + (machine.gpuCores > 0 ? " (\(machine.gpuCores)-core GPU)" : "")),
                ("memory", machine.memoryLabel),
                ("macos", machine.macOS),
                ("version", machine.version),
            ]
            if includingReport { fields.append(("report", String(decoding: try json(pretty: false), as: UTF8.self))) }
            return Diagnostic.issueURL(fields: fields)
        }
    }

    /// The URL of a new issue with `fields` as its query, **every reserved byte escaped**: `URLComponents`
    /// leaves `+` alone, which GitHub reads back as a space — in a JSON exponent (`1e+20`) or a chip name.
    package static func issueURL(fields: [(String, String)]) -> URL {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        let query = fields.map { key, value in
            "\(key)=\(value.addingPercentEncoding(withAllowedCharacters: allowed) ?? "")"
        }.joined(separator: "&")
        return URL(string: "\(repository)/issues/new?\(query)")!
    }

    /// `encoder + first + (E − 1) · steady + decoding`: one render's evaluations are the first (which
    /// builds everything) and `E − 1` like the second.
    package static func estimatedRenderSeconds(encoder: Double, first: Double, steady: Double, decoding: Double,
                                               evaluations: Int) -> Double {
        encoder + first + Double(max(0, evaluations - 1)) * steady + decoding
    }

    /// **The threshold on the second `model_out`'s worst channel**, at the user's own conditioning.
    ///
    /// Qwen-Image-2.1, measured on the M1 Pro on 04/10/2026 (the diagnostic, also run on the oracle's conditioning):
    ///
    ///                                        step 0 (σ = 1)    step 1 (σ = 0.9626, judged)
    ///     at the oracle's conditioning        3.65·10⁻⁶         8.22·10⁻⁶      ← step 0 is the turbo LoRA check's figure
    ///     at the engine's own conditioning    3.70·10⁻⁶         9.28·10⁻⁶      ← what a user reads (+13 %)
    ///
    /// The encoder's states sit at 2.2·10⁻⁵ of the oracle's (worst channel): the DiT damps them. There
    /// is no fp64 witness of this step under the turbo (`witness-qwen21-lora-f64-512` holds `temb` and
    /// the modulation only); the oracle's own distance to the exact is 5.3·10⁻⁶ without LoRA.
    /// The threshold is the repository's DiT threshold, **5·10⁻⁴** (`kleinVerdict`, every
    /// `qwen21-*` check), so a user's figure reads like a check's: 54× what this machine reads, room for
    /// another chip's GPU/AMX cut and summation order; and still ~300× under what a real fault moves —
    /// the turbo LoRA lost moves `model_out` by 0.43 (worst channel, `goldens-qwen21-dit-512` vs
    /// `-dit-lora-512`), a rank-16 user LoRA by 0.15.
    ///
    /// Z-Image Turbo, measured on the M1 Pro on 04/10/2026 (the same way):
    ///
    ///                                        step 0 (t = 0)    step 1 (t = 0.0526, judged)
    ///     at the oracle's conditioning        9.08·10⁻⁶         9.13·10⁻⁶      ← the trajectory check's 9.3·10⁻⁶
    ///     at the engine's own conditioning    9.26·10⁻⁶         9.51·10⁻⁶      ← what a user reads (+4 %)
    ///
    /// The encoder's states sit at 1.9·10⁻⁵ of `cap_feats` (worst channel). Same rule, same **5·10⁻⁴**:
    /// 53× what this machine reads, as for Qwen. A golden perturbed by 10⁻³ on one channel of
    /// `model_out.1` fails it, so the margin does not swallow a fault confined to a channel.
    package static func threshold(for family: Family) -> Double { 5e-4 }

    // ── the run ─────────────────────────────────────────────────────────────────────────────

    /// **The stage a model's diagnostic is at**, as `run` reports it to `onProgress` — an app says it
    /// in its own language; `description` is the English the command line prints.
    public enum Stage: String, Sendable, CaseIterable, CustomStringConvertible {
        /// Tokenizer and text encoder, cold.
        case textEncoder = "text encoder"
        /// The two DiT evaluations, the second judged.
        case denoiser = "two DiT evaluations"
        /// The VAE decoder, on the one-step estimate.
        case decoder = "VAE decoder"
        public var description: String { rawValue }
    }

    /// **Runs the diagnostic** on the cards that are installed in `library` (the others are skipped
    /// silently: `ModelCard.missing(in:)` says why). Never throws: a model that fails is reported with
    /// its `error`, the others still run. Synchronous and heavy (~15 s and up to ~5 GB per model): call
    /// it off the main thread, and never from the engine's queue.
    ///
    /// **It passes the same doors as a render**: it waits its turn on the engine's queue
    /// (`Engine.file`: never beside a render of this process), and each model takes the preflights
    /// (`Preflight.run`: license, memory at the diagnostic's format, the machine's render lock — a
    /// render of another process refuses it). A model refused there reports the refusal as its `error`.
    public static func run(_ cards: [ModelCard], in library: Library, cancellation: Cancellation? = nil,
                           onProgress: (@Sendable (_ model: ModelCard, _ stage: Stage) -> Void)? = nil) -> Report {
        dispatchPrecondition(condition: .notOnQueue(Engine.file))
        return Engine.file.sync { runQueued(cards, in: library, cancellation: cancellation, onProgress: onProgress) }
    }

    private static func runQueued(_ cards: [ModelCard], in library: Library, cancellation: Cancellation?,
                                  onProgress: (@Sendable (_ model: ModelCard, _ stage: Stage) -> Void)?) -> Report {
        let started = Date()
        let peak = FootprintPeak()
        let swapBefore = swapouts()
        let before = conditions()
        let witnessBefore = witness()
        var models: [ModelReport] = []
        for card in cards {
            if cancellation?.isCancelled == true { break }
            guard card.missing(in: library) == nil else { continue }
            models.append(diagnose(card, in: library, cancellation: cancellation, onProgress: onProgress))
        }
        let witnessAfter = models.isEmpty ? nil : witness()
        let after = conditions()
        let settings = EngineSettings.effective
        var forced: [String: String] = [:]
        for row in settings.rows where row.source != .byDefault || row.key == "amx" {
            forced[row.key] = "\(row.value) (\(row.source.rawValue))"
        }
        let onInternal = try? library.root.resourceValues(forKeys: [.volumeIsInternalKey]).volumeIsInternal
        return Report(date: ISO8601DateFormatter().string(from: started), machine: machine(), models: models,
                      totalSeconds: Date().timeIntervalSince(started), peakFootprintBytes: peak.stop(),
                      swapouts: max(0, swapouts() - swapBefore), before: before, after: after,
                      witnessBefore: witnessBefore, witnessAfter: witnessAfter, settings: forced,
                      profile: Library.withoutHome(settings.profileStatus), libraryOnInternalDisk: onInternal ?? nil)
    }

    /// One model: its report, with `error` set if anything threw.
    package static func diagnose(_ card: ModelCard, in library: Library, cancellation: Cancellation?,
                                 onProgress: (@Sendable (_ model: ModelCard, _ stage: Stage) -> Void)?) -> ModelReport {
        let started = Date()
        let peak = FootprintPeak()
        let swapBefore = swapouts()
        let diskBefore = diskBytesRead()
        var report = ModelReport(model: card.id, name: card.name, family: card.family.rawValue,
                                 steps: card.defaultSteps, evaluations: card.defaultSteps, seconds: nil,
                                 estimatedRenderSeconds: nil, peakFootprintBytes: 0, swapouts: 0,
                                 golden: nil, deviation: nil, error: nil)
        report.variant = card.isImported ? nil : library.installedVariant(of: card.family)?.rawValue
        defer { Arena.releaseToSystem() }
        let gpuFailures = Attention.failures
        var planned = false
        do {
            let model = try Model.named(card.id, in: library)
            // **As a render reads it** (`Chain.execute`): the budget once, at the launch, judged by the
            // preflight and handed to the memory plan. Written before the preflight: a refusal is the
            // figure an 8 GB report exists for.
            let budget = MemoryBudget.current()
            let floor = card.memoryNeed(width: resolution, height: resolution)
            let comfortable = card.memoryComfort(width: resolution, height: resolution)
            report.memory = MemoryReport(reclaimableBytes: budget.reclaimable, availableBytes: budget.available,
                                         reserveBytes: budget.reserve, floorBytes: floor, comfortableBytes: comfortable,
                                         lean: false, decisions: [])
            // Installed, license, memory at the diagnostic's format, no other render on the machine.
            let lock = try Preflight.run(model, width: resolution, height: resolution, budget: budget)
            defer { withExtendedLifetime(lock) {} }
            try MemoryPlan.during(budget: budget, floor: floor, comfortable: comfortable) {
                planned = true
                try measure(model, card: card, into: &report, started: started, cancellation: cancellation,
                            onProgress: onProgress)
            }
        } catch {
            report.error = Library.withoutHome("\(error)")
            report.failure = EngineError(error)
        }
        if planned, let decisions = MemoryPlan.lastRender?.decisions {
            report.memory?.decisions = decisions
            report.memory?.lean = decisions.contains { $0.contains("lean") }
        }
        // A failed submission leaves its output as it was: the figures above did not come from a
        // computation, and the report says so where a reader looks.
        if let failed = Attention.summarize(since: gpuFailures), report.error == nil { report.error = failed }
        report.peakFootprintBytes = peak.stop()
        report.swapouts = max(0, swapouts() - swapBefore)
        report.diskReadBytes = max(0, diskBytesRead() - diskBefore)
        report.thermalAfter = thermalLabel(ProcessInfo.processInfo.thermalState)
        return report
    }

    /// The three stages, timed, inside the render's memory plan.
    private static func measure(_ model: Model, card: ModelCard, into report: inout ModelReport, started: Date,
                                cancellation: Cancellation?,
                                onProgress: (@Sendable (_ model: ModelCard, _ stage: Stage) -> Void)?) throws {
        let denoising = model.chain.denoising
        let (height, width) = (resolution / denoising.space.factor, resolution / denoising.space.factor)
        report.evaluations = denoising.plan(height: height, width: width, steps: denoising.defaultSteps,
                                            start: 0, withLoRA: false).evaluations
        report.steps = denoising.defaultSteps
        // A golden belongs to the published model: an imported DiT of the family has other weights.
        let golden = card.isImported ? nil : try EmbeddedGolden.load(card.id)
        if let golden, golden.channels != denoising.space.channels || golden.height != height || golden.width != width {
            throw Artifact.Failure.misuse("diagnostic golden \(golden.file): [\(golden.channels), \(golden.height), "
                                             + "\(golden.width)], the model's latent is [\(denoising.space.channels), \(height), \(width)]")
        }
        report.golden = golden?.info
        let context = Context(reproducible: EngineSettings.effective.frozenCut, cancellation: cancellation,
                              onProgress: nil)

        // ── the text, cold ──
        onProgress?(card, .textEncoder)
        var begin = Date()
        let text = try (model.chain.text as? any ImageTextModule)?.encoder(prompt, images: [], context: context)
            ?? model.chain.text.encoder(prompt, context: context)
        let encoderSeconds = Date().timeIntervalSince(begin)
        Arena.releaseToSystem()

        // ── two evaluations ──
        onProgress?(card, .denoiser)
        let inputs = try golden.map { ($0.inputs, $0.sigmas) } ?? noGoldenInputs(model, height: height, width: width)
        begin = Date()
        let out = try evaluate(model, text: text, inputs: inputs.0, sigmas: inputs.1, height: height, width: width,
                               context: context)
        let denoisingSeconds = Date().timeIntervalSince(begin)
        Arena.releaseToSystem()
        if let golden {
            let threshold = threshold(for: card.family)
            let second = channelError(planar: out.second, golden.outputs[1], channels: golden.channels)
            let first = channelError(planar: out.first, golden.outputs[0], channels: golden.channels)
            report.deviation = Deviation(worst: second.worst, channel: second.channel, median: second.median,
                                         threshold: threshold, pass: second.worst <= threshold,
                                         firstEvaluationWorst: first.worst)
        }

        // ── the decoder, on the one-step estimate of x₀ ──
        onProgress?(card, .decoder)
        MemoryPlan.readForDecoding()   // after the DiT is gone, as `Chain.execute` does
        let estimate = try oneStepEstimate(model, x: inputs.0[1], output: out.second, sigma: inputs.1[1])
        begin = Date()
        _ = try model.chain.decoding.decode(Latent(space: denoising.space, height: height, width: width, values: estimate),
                                            context: context)
        let decodingSeconds = Date().timeIntervalSince(begin)
        Arena.releaseToSystem()

        let first = denoisingSeconds - out.secondSeconds
        report.seconds = Seconds(encoder: encoderSeconds, firstEvaluation: first, steadyEvaluation: out.secondSeconds,
                                 decoding: decodingSeconds, total: Date().timeIntervalSince(started))
        report.estimatedRenderSeconds = estimatedRenderSeconds(encoder: encoderSeconds, first: first,
                                                               steady: out.secondSeconds, decoding: decodingSeconds,
                                                               evaluations: report.evaluations)
    }

    /// Two evaluations of the model's DiT, as its product denoiser runs them. `secondSeconds`: the
    /// second `forward` alone; the first's time is the caller's wall clock minus it (construction included).
    package struct Evaluations { package var first: [Float]; package var second: [Float]; package var secondSeconds: Double }

    package static func evaluate(_ model: Model, text: Conditioning, inputs: [[Float]], sigmas: [Float],
                                 height: Int, width: Int, context: Context) throws -> Evaluations {
        switch model.chain.denoising {
        case let qwen as QwenImage21DenoisingModule:
            return try evaluateQwen(qwen, text: text, inputs: inputs, sigmas: sigmas, height: height, width: width,
                                    context: context)
        case let zImage as ZImageDenoisingModule:
            // `ZImageDenoisingModule.denoise`'s DiT, without the sampler: no LoRA, the module's caps drop.
            let dit = try DiT(artifact: try Artifact(path: zImage.map), latentHeight: height, latentWidth: width,
                              capLength: text.rows, freezeCut: context.reproducible, lora: nil)
            dit.capsDrop = zImage.capsDrop
            dit.cancellation = context.cancellation
            return try text.values.withUnsafeBufferPointer { caps in
                let first = try inputs[0].withUnsafeBufferPointer {
                    try dit.forward(latent: $0.baseAddress!, caps: caps.baseAddress!, sigma: sigmas[0]) }
                try context.check()
                let begin = Date()
                let second = try inputs[1].withUnsafeBufferPointer {
                    try dit.forward(latent: $0.baseAddress!, caps: caps.baseAddress!, sigma: sigmas[1]) }
                return Evaluations(first: first, second: second, secondSeconds: Date().timeIntervalSince(begin))
            }
        default:
            throw Failure.noProbe(model.family.rawValue)
        }
    }

    /// **`QwenImage21Sampler.run`'s turbo phase, two steps of it, on the given latents**: the same
    /// LoRA stack (the turbo alone, `LoRA(layers:)`), the same cache policy, the prefill folded into the
    /// first step. The sampler itself is not used because it chains its own latent (`x + Δσ·v`): the
    /// second evaluation must read the oracle's `x_t.1`, not ours. `--same-bits` checks the two agree.
    package static func evaluateQwen(_ module: QwenImage21DenoisingModule, text: Conditioning, inputs: [[Float]],
                                     sigmas: [Float], height: Int, width: Int, context: Context) throws -> Evaluations {
        guard text.format == module.entry, text.imageSlots.count == text.rows else {
            throw Chain.Failure.incompatibleText(output: text.format, entry: module.entry)
        }
        let turbo = try LoRA(layers: try LoRA(paths: [(path: module.turbo, strength: 1)]).layers)
        let sequence = try QwenImage21Sequence(slots: text.imageSlots + [Bool](repeating: true, count: height * width / 4),
                                               images: [.init(height: height, width: width)])
        let dit = try QwenImage21DiT(artifact: try Artifact(path: module.map), sequence: sequence, lora: turbo,
                                     storage: module.storage ?? QwenImage21CachePolicy.storage(prefix: sequence.prefix),
                                     kept: nil, freezeCut: context.reproducible)
        dit.cancellation = context.cancellation
        try dit.tabulateModulation(sigmas: sigmas)
        try text.values.withUnsafeBufferPointer {
            try dit.prefill(hidden: $0.baseAddress!, references: [], intoFirstStep: true)
        }
        let first = try inputs[0].withUnsafeBufferPointer { try dit.forward(latent: $0.baseAddress!, sigma: sigmas[0]) }
        try context.check()
        let begin = Date()
        let second = try inputs[1].withUnsafeBufferPointer { try dit.forward(latent: $0.baseAddress!, sigma: sigmas[1]) }
        return Evaluations(first: first, second: second, secondSeconds: Date().timeIntervalSince(begin))
    }

    /// Without a golden: the seed's noise at the schedule's first two σ (in what the DiT receives).
    /// Only the time counts, so both evaluations read the same latent.
    private static func noGoldenInputs(_ model: Model, height: Int, width: Int) throws -> ([[Float]], [Float]) {
        let denoising = model.chain.denoising
        let noise = Latent.noise(denoising.space, height: height, width: width, seed: seed).values
        let σ = denoising.sigmas(steps: denoising.defaultSteps, height: height, width: width)
        switch denoising {
        case is ZImageDenoisingModule:
            let schedule = FlowMatchSchedule(steps: denoising.defaultSteps)
            return ([noise, noise], [schedule.modelTime(at: 0), schedule.modelTime(at: 1)])
        case is QwenImage21DenoisingModule:
            return ([noise, noise], [σ[0], σ[1]])
        default:
            throw Failure.noProbe(model.family.rawValue)
        }
    }

    /// x̂₀ from one evaluation — what the decoder gets, so it decodes an image and not noise (its time
    /// does not depend on it). Qwen-Image-2.1: `x − σ·v`. Z-Image: the DiT reads `t = 1 − σ` and returns
    /// `−v` (`Sampler`), hence `x + (1 − t)·out`.
    private static func oneStepEstimate(_ model: Model, x: [Float], output: [Float], sigma: Float) throws -> [Float] {
        switch model.chain.denoising {
        case is QwenImage21DenoisingModule: return zip(x, output).map { $0 - sigma * $1 }
        case is ZImageDenoisingModule: return zip(x, output).map { $0 + (1 - sigma) * $1 }
        default: throw Failure.noProbe(model.family.rawValue)
        }
    }

    package enum Failure: Error, CustomStringConvertible, Equatable {
        case noProbe(String)
        package var description: String {
            switch self {
            case .noProbe(let family):
                return "the diagnostic does not drive the \(family) DiT yet (Z-Image and Qwen-Image-2.1 only)"
            }
        }
    }

    // ── the embedded golden ─────────────────────────────────────────────────────────────────

    /// `Resources/diagnostic/<model>.safetensors` — see the header.
    package struct EmbeddedGolden {
        package let file: String
        package let metadata: [String: String]
        package let channels, height, width: Int
        /// `x_t.0`, `x_t.1`, planar `[C, H, W]`.
        package let inputs: [[Float]]
        /// `model_out.0`, `model_out.1`, planar.
        package let outputs: [[Float]]
        /// What the DiT's `forward` receives for each.
        package let sigmas: [Float]

        package var info: GoldenInfo {
            GoldenInfo(file: file, source: metadata["source"] ?? "?", sourceSHA256: metadata["source_sha256"] ?? "?",
                       generated: metadata["generated"] ?? "?")
        }

        /// `nil` when the model has none; throws when it has one that does not read.
        package static func load(_ model: String) throws -> EmbeddedGolden? {
            guard let url = resource("diagnostic/\(model).safetensors") else { return nil }
            return try read(url)
        }

        package static func read(_ url: URL) throws -> EmbeddedGolden {
            let file = try Safetensors(path: url.path)
            func tensor(_ name: String) throws -> (values: [Float], shape: [Int]) {
                guard let e = file.entries[name], e.dtype == "F32", let values = file.materialize(name) else {
                    throw Safetensors.Failure.missingTensor(file: url.lastPathComponent, name: name)
                }
                return (values, e.shape)
            }
            let x0 = try tensor("x_t.0"), x1 = try tensor("x_t.1")
            let y0 = try tensor("model_out.0"), y1 = try tensor("model_out.1"), sigma = try tensor("sigma")
            guard x0.shape.count == 3, [x1.shape, y0.shape, y1.shape].allSatisfy({ $0 == x0.shape }), sigma.values.count == 2 else {
                throw Safetensors.Failure.badHeader("\(url.lastPathComponent): shapes \(x0.shape) \(x1.shape) \(y0.shape) "
                                                    + "\(y1.shape) \(sigma.shape)")
            }
            return EmbeddedGolden(file: "diagnostic/" + url.lastPathComponent, metadata: file.metadata,
                                  channels: x0.shape[0], height: x0.shape[1], width: x0.shape[2],
                                  inputs: [x0.values, x1.values], outputs: [y0.values, y1.values], sigmas: sigma.values)
        }
    }

    /// **A file of the library's resource bundle, found without `Bundle.module`**: its generated accessor
    /// calls `fatalError` when the bundle is not where SwiftPM left it — an app that forgot to copy it
    /// would crash on the diagnostic button instead of reporting `"golden": null`. Looked for beside the
    /// executable (the CLI, `swift test`), in the main bundle's resources (an app), and beside the code.
    package static func resource(_ relative: String) -> URL? {
        let name = "Siliconed_Siliconed.bundle"
        var folders: [URL] = []
        if let resources = Bundle.main.resourceURL { folders.append(resources) }
        folders.append(Bundle.main.bundleURL)
        if let executable = Bundle.main.executableURL?.resolvingSymlinksInPath() {
            let folder = executable.deletingLastPathComponent()
            // The developer's command line in `Contents/Helpers` shares the app's `Contents/Resources` (tools/app.sh).
            folders += [folder, folder.deletingLastPathComponent().appendingPathComponent("Resources")]
        }
        let code = Bundle(for: BundleMarker.self).bundleURL
        folders += [code, code.deletingLastPathComponent()]
        let fm = FileManager.default
        for folder in folders {
            for bundle in [folder.appendingPathComponent(name), folder.appendingPathComponent("Contents/Resources/\(name)")] {
                for candidate in [bundle.appendingPathComponent("Resources/\(relative)"),
                                  bundle.appendingPathComponent("Contents/Resources/Resources/\(relative)")]
                where fm.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
        }
        return nil
    }
    private final class BundleMarker {}

    // ── the machine ─────────────────────────────────────────────────────────────────────────

    public static func machine() -> Machine {
        let device = MTLCreateSystemDefaultDevice()
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let osBuild = sysctlString("kern.osversion")
        return Machine(chip: sysctlString("machdep.cpu.brand_string"), hardwareModel: sysctlString("hw.model"),
                       performanceCores: sysctlInteger("hw.perflevel0.physicalcpu"),
                       efficiencyCores: sysctlInteger("hw.perflevel1.physicalcpu"),
                       gpuCores: gpuCoreCount(), memoryBytes: sysctlInteger("hw.memsize"),
                       macOS: "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)" + (osBuild == "unknown" ? "" : " (\(osBuild))"),
                       version: binaryVersion(),
                       build: SHA256.hash(data: Data(RenderCache.build.utf8)).prefix(6).map { String(format: "%02x", $0) }.joined(),
                       gpu: device?.name ?? "none", gpuFamily: device.map(appleFamily) ?? "",
                       gpuWorkingSetBytes: Int(device?.recommendedMaxWorkingSetSize ?? 0))
    }

    /// The highest `appleN` family: `MTLGPUFamily.apple1` is 1001, and the next ones follow.
    private static func appleFamily(_ device: MTLDevice) -> String {
        for n in stride(from: 16, through: 1, by: -1) {
            if let family = MTLGPUFamily(rawValue: 1000 + n), device.supportsFamily(family) { return "apple\(n)" }
        }
        return ""
    }

    // ── the conditions ──────────────────────────────────────────────────────────────────────

    package static func conditions() -> Conditions {
        var usage = xsw_usage(), size = MemoryLayout<xsw_usage>.size
        let swap = sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 ? Int(usage.xsu_used) : 0
        return Conditions(thermal: thermalLabel(ProcessInfo.processInfo.thermalState),
                          lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, onBattery: onBattery(),
                          reclaimableBytes: MemoryBudget.current().reclaimable,
                          compressorHeadroomBytes: MemoryPressure.current()?.headroom, swapUsedBytes: swap)
    }

    package static func thermalLabel(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// The source the machine draws from: `nil` when IOKit does not say (a desktop says "AC Power").
    private static func onBattery() -> Bool? {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() else { return nil }
        return (type as String) == kIOPSBatteryPowerValue
    }

    /// Bytes this process has read from the disk (`ri_diskio_bytesread`): a map read from the page
    /// cache does not count.
    package static func diskBytesRead() -> Int {
        var info = rusage_info_v4()
        let ok = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        return ok == 0 ? Int(info.ri_diskio_bytesread) : 0
    }

    // ── the witness ─────────────────────────────────────────────────────────────────────────

    /// **The chip's fp32 GEMM, GPU against AMX** — `C = A × B`, `A [511 × 3840]`, `B [3840 × 3840]`: the
    /// Z-Image DiT's attention projections at the row count just under `amx_min` (512), so the GPU runs
    /// alone even where the conductor is on (the AMX check's "GPU-alone reference"). The AMX side is
    /// `cblas_sgemm` on the same operands, which is also the truth the GPU's rows are judged against.
    /// One warm-up each, then the median of five. ~0.2 s and 83 MB on the M1 Pro; `nil` without Metal.
    package static func witness() -> Witness? {
        let m = Conductor.minimumRows - 1, k = 3840, n = 3840
        guard m > 0, let gemm = try? GEMM(freezeCut: true),
              let arena = try? Arena(capacity: (m * k + k * n + 2 * m * n) * 4 + (32 << 20)),
              let a = try? arena.reserve("witness.a", bytes: m * k * 4).assumingMemoryBound(to: Float.self),
              let b = try? arena.reserve("witness.b", bytes: k * n * 4).assumingMemoryBound(to: Float.self),
              let c = try? arena.reserve("witness.c", bytes: m * n * 4).assumingMemoryBound(to: Float.self),
              let truth = try? arena.reserve("witness.truth", bytes: m * n * 4).assumingMemoryBound(to: Float.self)
        else { return nil }
        defer { withExtendedLifetime(arena) {} }
        var state: UInt64 = 0xA11CE
        func noise() -> Float {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return Float(Int32(truncatingIfNeeded: state)) / Float(Int32.max)
        }
        for i in 0..<(m * k) { a[i] = noise() }
        for i in 0..<(k * n) { b[i] = noise() }
        guard let ab = try? gemm.wrap(UnsafeMutableRawPointer(a), bytes: m * k * 4, name: "witness.a"),
              let bb = try? gemm.wrap(UnsafeMutableRawPointer(b), bytes: k * n * 4, name: "witness.b"),
              let cb = try? gemm.wrap(UnsafeMutableRawPointer(c), bytes: m * n * 4, name: "witness.c") else { return nil }
        func median(_ run: () -> Void) -> Double {
            run()
            let times = (0..<5).map { _ -> Double in
                let begin = DispatchTime.now().uptimeNanoseconds
                run()
                return Double(DispatchTime.now().uptimeNanoseconds - begin) * 1e-9
            }.sorted()
            return times[2]
        }
        let amx = median {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k),
                        1, a, Int32(k), b, Int32(n), 0, truth, Int32(n))
        }
        let gpu = median { _ = gemm.linear(a: ab, b: bb, c: cb, m: m, k: k, n: n, weightIsTransposed: true) }
        var worst = 0.0
        for row in 0..<m {
            var numerator = 0.0, denominator = 0.0
            for column in 0..<n {
                let i = row * n + column
                let d = Double(c[i]) - Double(truth[i])
                numerator += d * d
                denominator += Double(truth[i]) * Double(truth[i])
            }
            let relative = denominator > 0 ? (numerator / denominator).squareRoot() : 0
            worst = max(worst, relative.isNaN ? .infinity : relative)
        }
        let flops = 2.0 * Double(m) * Double(k) * Double(n)
        return Witness(gpuTFLOPS: flops / gpu / 1e12, amxTFLOPS: flops / amx / 1e12, gpuWorstRowError: worst)
    }

    private static func sysctlString(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return "unknown" }
        return String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    private static func sysctlInteger(_ name: String) -> Int {
        var value: Int64 = 0
        var size = MemoryLayout<Int64>.size
        if sysctlbyname(name, &value, &size, nil, 0) == 0 { return size == 4 ? Int(Int32(truncatingIfNeeded: value)) : Int(value) }
        return 0
    }

    /// The `AGXAccelerator`'s `gpu-core-count` in the IORegistry — Metal does not say it.
    private static func gpuCoreCount() -> Int {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iterator) == KERN_SUCCESS else {
            return 0
        }
        defer { IOObjectRelease(iterator) }
        var service = IOIteratorNext(iterator)
        while service != 0 {
            defer { IOObjectRelease(service); service = IOIteratorNext(iterator) }
            if let value = IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? Int, value > 0 {
                return value
            }
        }
        return 0
    }

    /// The app's version; else, for a binary built from a clone, its commit — read with `git` only when a
    /// `.git` lies above the executable (a Mac without the developer tools would otherwise be asked to
    /// install them by a dialog).
    private static func binaryVersion() -> String {
        var bundles = [Bundle.main]
        if let path = Bundle.main.executableURL?.resolvingSymlinksInPath().path, let range = path.range(of: ".app/Contents/") {
            if let app = Bundle(path: String(path[..<range.lowerBound]) + ".app") { bundles.append(app) }
        }
        for bundle in bundles {
            if let short = bundle.infoDictionary?["CFBundleShortVersionString"] as? String {
                let build = bundle.infoDictionary?["CFBundleVersion"] as? String
                return short + (build.map { " (\($0))" } ?? "")
            }
        }
        guard var folder = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent() else { return "unknown" }
        while folder.path != "/" {
            if FileManager.default.fileExists(atPath: folder.appendingPathComponent(".git").path) {
                guard let commit = git(["rev-parse", "--short=10", "HEAD"], in: folder), !commit.isEmpty else { return "unknown" }
                let dirty = !(git(["status", "--porcelain", "--untracked-files=no"], in: folder) ?? "").isEmpty
                return "git " + commit + (dirty ? "-dirty" : "")
            }
            folder = folder.deletingLastPathComponent()
        }
        return "unknown"
    }

    private static func git(_ arguments: [String], in folder: URL) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", folder.path] + arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// `vm_stat`'s "Swapouts" (pages): the machine's, not the process's — what the zero-swap rule reads.
    package static func swapouts() -> Int {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        return result == KERN_SUCCESS ? Int(stats.swapouts) : 0
    }

    /// `channelError` on planar `[C, H, W]` tensors: the channel is the latent's.
    package static func channelError(planar got: [Float], _ expected: [Float], channels: Int)
        -> (worst: Double, channel: Int, median: Double) {
        let g = channelsLast(got, channels: channels), e = channelsLast(expected, channels: channels)
        return e.withUnsafeBufferPointer { Siliconed.channelError(g, $0, columns: channels) }
    }

    /// `[C, h, w]` → `[h·w, C]`.
    package static func channelsLast(_ v: [Float], channels: Int) -> [Float] {
        let count = v.count / channels
        var out = [Float](repeating: 0, count: v.count)
        for ch in 0..<channels { for t in 0..<count { out[t * channels + ch] = v[ch * count + t] } }
        return out
    }
}

// ── the common measures — shared by the diagnostic and every check of the CLI ──────────────────

/// Relative error channel by channel, each normalized by its own norm — **bounded from below**.
///
/// The denominator went wrong twice, in both directions. Too large at first: a global
/// `‖a−b‖/‖b‖` is a test on channel 85, which carries 94 to 99.7 % of the energy, and it
/// underestimates a fault elsewhere by 373×. Too small afterwards: a channel worth 3.5·10⁻⁵ when the
/// median is 9·10⁻² turns 7.45·10⁻⁹ of absolute error — the noise of a cancellation over 256
/// terms — into 2.1·10⁻⁴ relative, and cries foul.
///
/// Hence the floor at 1 % of the median norm: a channel weaker than that carries less than 10⁻⁴ of
/// the energy, and its relative precision decides nothing. Sensitivity remains whole over four
/// orders of magnitude, ample to cover a massive activation, which is **above** the
/// median and not below it.
///
/// A library function (`package`), shared by the diagnostic and every check, so a user's figure
/// reads exactly like a check's.
package func channelError(_ got: [Float], _ expected: UnsafeBufferPointer<Float>, columns: Int)
    -> (worst: Double, channel: Int, median: Double) {
    let rows = expected.count / columns
    var numerator = [Double](repeating: 0, count: columns)
    var denominator = [Double](repeating: 0, count: columns)
    for row in 0..<rows {
        for column in 0..<columns {
            let i = row * columns + column
            let d = Double(got[i]) - Double(expected[i])
            numerator[column] += d * d
            denominator[column] += Double(expected[i]) * Double(expected[i])
        }
    }
    var norms = denominator.filter { $0 > 0 }.map { $0.squareRoot() }
    guard !norms.isEmpty else { return (0, 0, 0) }
    norms.sort()
    let floor = 0.01 * norms[norms.count / 2]

    // **`r > worst` is FALSE for a `NaN`.** A boundary that was entirely non-finite therefore came
    // out as "0.000e+00", median `nan`, verdict ✓ — in the project's most used comparison
    // function. A non-finite is not a small deviation: it is worth infinity, and it must make any
    // threshold fail.
    var worst = 0.0, channel = 0, live: [Double] = []
    for column in 0..<columns where denominator[column] > 0 {
        var r = numerator[column].squareRoot() / max(denominator[column].squareRoot(), floor)
        if r.isNaN { r = .infinity }
        live.append(r)
        if r > worst { worst = r; channel = column }
    }
    live.sort()
    return (worst, channel, live[live.count / 2])
}

/// **Which error judges a stage: the cumulative one, or its own.**
///
/// A check compares each boundary to the oracle's: the error it reads is **cumulative**, the
/// stage's own plus what it does to the error it receives. On a well-conditioned stage the second
/// term stays the size of the first, and the cumulative error is the right judge. On a badly
/// conditioned one it does not: the Flux encoder's one-head `mid_block` attention (logits up to 435)
/// takes 1.2·10⁻⁵ of fp32 rounding at its input to 6·10⁻⁴ at its output, where torch fp32 on MPS
/// lands at 8.6·10⁻⁴ and torch on CPU at 1.8·10⁻⁴ — the oracle is one draw among the fp32
/// implementations, and a ✗ there says nothing about the engine.
///
/// Hence the rule: the **amplification** is measured — the stage's output error over its input
/// error, both against the oracle — and above `amplificationLimit` the stage is judged on its
/// **local** error: fed with the oracle's input, its output against the oracle's, at the same
/// threshold. A fault inside the stage shows there whole; only the inherited error is set aside, and
/// the stages after it stay judged cumulatively. Never excused: an input identical to the oracle's
/// (the amplification is undefined, and the cumulative error already is the local one), and a
/// non-finite output.
package enum StageJudgement {
    /// Above this measured amplification, a stage is judged on its local error.
    package static let amplificationLimit = 10.0

    /// `output / input`, or `nil` when the input carries no error (or the figures are not finite).
    package static func amplification(inputError: Double, outputError: Double) -> Double? {
        guard inputError > 0, inputError.isFinite, outputError.isFinite else { return nil }
        return outputError / inputError
    }

    /// Whether the stage is judged on its local error rather than the cumulative one.
    package static func judgesLocally(inputError: Double, outputError: Double) -> Bool {
        guard let a = amplification(inputError: inputError, outputError: outputError) else { return false }
        return a > amplificationLimit
    }
}

/// **The peak `phys_footprint`, sampled** every 10 ms on a thread of its own — shared by
/// the diagnostic and the checks that report a peak.
package final class FootprintPeak: @unchecked Sendable {
    private let lock = NSLock()
    private var peak = 0, running = true
    package init() {
        Thread.detachNewThread { [self] in
            while true {
                let now = Arena.processFootprint()
                lock.lock()
                peak = max(peak, now)
                let go = running
                lock.unlock()
                if !go { return }
                usleep(10_000)
            }
        }
    }
    package var value: Int { lock.lock(); defer { lock.unlock() }; return max(peak, Arena.processFootprint()) }
    /// Stops the sampling and returns the peak.
    @discardableResult package func stop() -> Int {
        lock.lock(); running = false; lock.unlock()
        return value
    }
}
