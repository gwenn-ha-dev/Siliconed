// The diagnostic's report: the JSON types (machine, conditions, judge, memory, stages), the report itself of schema 22 and the
// issue link. Split from `Diagnostic.swift`, which says what the diagnostic is.

import CryptoKit
import Foundation
import IOKit
import IOKit.ps
import Metal

extension Diagnostic {
    // ── the machine ─────────────────────────────────────────────────────────────────────────

    package struct Machine: Codable, Sendable, Equatable {
        /// `machdep.cpu.brand_string`: "Apple M1 Pro".
        package var chip: String
        /// `hw.model`: "MacBookPro18,3".
        package var hardwareModel: String
        package var performanceCores: Int
        package var efficiencyCores: Int
        /// IORegistry's `gpu-core-count` (`AGXAccelerator`); 0 if unreadable.
        package var gpuCores: Int
        package var memoryBytes: Int
        /// "26.0.1 (25A…)".
        package var macOS: String
        /// `CFBundleShortVersionString (CFBundleVersion)` of the app, else the git commit of the
        /// repository the binary was built in (`-dirty` if modified), else "unknown".
        package var version: String
        /// The executable's fingerprint: the first 12 hex digits of the SHA-256 of `RenderCache.build`
        /// (path, size, date) — two builds of one commit differ here, and no local path goes public.
        package var build: String
        /// `MTLDevice.name`: "Apple M1 Pro".
        package var gpu: String
        /// The highest `MTLGPUFamily.appleN` the device supports, "apple7" on the M1 family; "" if none.
        package var gpuFamily: String
        /// `MTLDevice.recommendedMaxWorkingSetSize`: what Metal lets the process keep resident.
        package var gpuWorkingSetBytes: Int
        /// **What the registry and the kernel say about the silicon** (`Inventory`), read, never decided on;
        /// `nil` in a report that predates it (the schema stays: nothing reads it to decide).
        package var inventory: Inventory?

        package init(chip: String, hardwareModel: String, performanceCores: Int, efficiencyCores: Int,
                    gpuCores: Int, memoryBytes: Int, macOS: String, version: String, build: String,
                    gpu: String = "", gpuFamily: String = "", gpuWorkingSetBytes: Int = 0, inventory: Inventory? = nil) {
            self.chip = chip; self.hardwareModel = hardwareModel; self.performanceCores = performanceCores
            self.efficiencyCores = efficiencyCores; self.gpuCores = gpuCores; self.memoryBytes = memoryBytes
            self.macOS = macOS; self.version = version; self.build = build
            self.gpu = gpu; self.gpuFamily = gpuFamily; self.gpuWorkingSetBytes = gpuWorkingSetBytes
            self.inventory = inventory
        }

        /// The GPU's nominal fp32 peak, FLOP/s (`Inventory.gpuFP32Peak`); `nil` without a clock.
        package var gpuFP32Peak: Double? { inventory?.gpuFP32Peak(enabledCores: gpuCores) }

        /// "16 GB": the spec sheet's figure (2³⁰), what a reader of the table compares.
        package var memoryLabel: String { "\(Int((Double(memoryBytes) / 1_073_741_824).rounded())) GB" }
    }

    /// **The state of the machine**, read before the first measurement and after the last.
    package struct Conditions: Codable, Sendable, Equatable {
        /// `ProcessInfo.thermalState`: nominal, fair, serious, critical. A fanless Mac throttles from fair.
        package var thermal: String
        package var lowPowerMode: Bool
        /// The machine runs on its battery; `nil` when the power source does not say.
        package var onBattery: Bool?
        /// `vm.swapusage`'s used bytes: another process's swap, and memory this one may lack.
        package var swapUsedBytes: Int
        package init(thermal: String, lowPowerMode: Bool, onBattery: Bool?, swapUsedBytes: Int) {
            self.thermal = thermal; self.lowPowerMode = lowPowerMode; self.onBattery = onBattery
            self.swapUsedBytes = swapUsedBytes
        }

        /// **A machine at rest, as far as one reading says**: thermal state nominal, Low Power Mode off. The
        /// battery is not held against it: it did not throttle the maps taken on it (M5, M5 Pro: witness
        /// ×1.00). Swap is judged between two readings (`Report.isProvisional`), never from one.
        package var isNominal: Bool { thermal == "nominal" && !lowPowerMode }

        private enum CodingKeys: String, CodingKey { case thermal, lowPowerMode, onBattery, swapUsedBytes }
        /// Explicit `null`s, as everywhere in the report: "unknown" is said, not dropped.
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(thermal, forKey: .thermal); try c.encode(lowPowerMode, forKey: .lowPowerMode)
            try c.encode(onBattery, forKey: .onBattery); try c.encode(swapUsedBytes, forKey: .swapUsedBytes)
        }
    }

    // ── what the probes judged, held and spent ───────────────────────────────────────────────

    /// **What the judge said of one kernel (or of a set of engines)**: the worst channel error it was seen with, the threshold of that
    /// product, and whether it is left out of the plan. Judged against the exact result (fp64) by the checks' rule — the fp32
    /// threshold, the ANE within 1.25 × bf16's own rounding (invariant 3). A kernel is excluded; the map stays a map.
    package struct Judge: Codable, Sendable, Equatable {
        package var error: Double
        package var threshold: Double
        package var excluded: Bool
        /// Why it is left out, when the judge did not compute a wrong result: `died`, the process was gone on it.
        package var reason: Reason?
        package enum Reason: String, Codable, Sendable {
            /// The process that loaded this route died, by a signal; the measure was taken again without it (`Child`).
            case died
        }
        package init(error: Double, threshold: Double, excluded: Bool, reason: Reason? = nil) {
            self.error = Diagnostic.significant(error, 3); self.threshold = Diagnostic.significant(threshold, 3); self.excluded = excluded
            self.reason = reason
        }
    }

    /// **What the probes were allowed, computed, and took**: B, the process's base when it started, the need the memory plan counted
    /// before the first product (bytes, without the base) and the peak `phys_footprint` read (with it, the kernel's interval maximum).
    /// A peak above `base + need` is a defect of the code, written as such (`Findings`).
    package struct Memory: Codable, Sendable, Equatable {
        package var budget: Int
        package var base: Int
        package var need: Int
        package var peak: Int
        package init(budget: Int, base: Int, need: Int, peak: Int) {
            self.budget = budget; self.base = base; self.need = need; self.peak = peak
        }
        /// **The memory held**: the peak under the base and the need, and both under B.
        package var held: Bool { peak <= base + need && base + need <= budget }
    }

    /// **A stage's seconds, what the sizes and the first witnesses projected for it** (`CapacityProbes.Sizes.projection`), **and the
    /// bias those seconds remove** (`CapacityProbes.biases`): every second of the map is spent against an error the plan would make
    /// without it, named in the report a user shares.
    package struct StageTime: Codable, Sendable, Equatable {
        package var seconds: Double
        package var projected: Double?
        package var removes: String?
        package init(seconds: Double, projected: Double? = nil, removes: String? = nil) {
            self.seconds = Diagnostic.significant(seconds, 3); self.projected = projected.map { Diagnostic.significant($0, 3) }
            self.removes = removes
        }
    }

    // ── the report ──────────────────────────────────────────────────────────────────────────

    /// **The report of schema 22**: the machine and what it announces, the model the probes fitted — what the engine plans with —,
    /// the raw measurements it was fitted on (so that anyone can replay the fit: `refit`), what the judge said of each kernel, the heat
    /// and the memory, each stage's seconds beside its projection, and what the measure found (`findings`, `uncertainty`). A finished
    /// report is always a map: it is applied as it stands.
    ///
    /// **A sober JSON**: it is the one a user shares. Keys sorted, numbers to five or six significant digits, each dictionary keyed by
    /// what it holds (`"gpu/fp32"`, `"gpu+ane"`).
    package struct Report: Codable, Sendable, Equatable {
        package var schema: Int
        /// ISO 8601, UTC.
        package var date: String
        /// The whole diagnostic, wall clock.
        package var seconds: Double
        /// The machine and its inventory (`Machine.inventory`): read, the denominator of « reached / peak », never a decision.
        package var machine: Machine
        /// The CPU's `hw.optional.arm.*` flags that bear on the products (`FEAT_BF16`, `FEAT_I8MM`,
        /// `FEAT_SME`, `FEAT_SME2`…) and `sme_max_svl_b`, as `sysctl` returns them; a flag this macOS does
        /// not know is absent.
        package var cpuFeatures: [String: Int]
        /// Every engine setting that is not its default, `"value (provenance)"`: a profile or an
        /// environment variable must not pass for the chip.
        package var settings: [String: String]
        package var before: Conditions
        package var after: Conditions
        /// **The model the engine plans with** (`CapacityPlan(model:)`): the lines, the ANE's row unit, the sets' sharing, the heat
        /// (sustained over cold), the memory bandwidth, the weights' preparation, the peak the inventory read.
        package var model: CapacityModel
        /// **What the fit read**, to the digits it was fitted on: `Fit.model(from: measurements)` is `model` (`refit`).
        package var measurements: CapacityFit.Measurements
        /// Per kernel, and per set of engines judged wrong as a whole: the judge's words (`excluded` left out of the plan).
        package var judge: [String: Judge]
        package var memory: Memory
        /// Seconds of each stage, with its projection.
        package var stages: [String: StageTime]
        /// **What the measure found**, in words and with its effect on the plan (`Findings`): never a reason to refuse the map.
        package var findings: [Finding]
        /// **The last render against this model's prediction** (`RenderLoop.Note`): the report an app sends carries it; the
        /// model itself is kept without it.
        package var render: RenderLoop.Note?
        /// **What the measure says of the Mac it ran on** (`Regime`): the others' CPU, the witnesses along the way, the dispersion of the chains.
        package var regime: Regime?

        package init(schema: Int = Diagnostic.schema, date: String, seconds: Double, machine: Machine, cpuFeatures: [String: Int],
                    settings: [String: String] = [:], before: Conditions, after: Conditions, model: CapacityModel,
                    measurements: CapacityFit.Measurements, judge: [String: Judge], memory: Memory, stages: [String: StageTime],
                    findings: [Finding], render: RenderLoop.Note? = nil, regime: Regime? = nil) {
            self.schema = schema; self.date = date; self.seconds = Diagnostic.significant(seconds, 3)
            self.machine = machine; self.cpuFeatures = cpuFeatures; self.settings = settings; self.before = before; self.after = after
            self.model = model; self.measurements = measurements; self.judge = judge; self.memory = memory; self.stages = stages
            self.findings = findings; self.render = render; self.regime = regime
        }

        /// **A report from the probes' run**: the measurements rounded, the model fitted on them (so that `refit` gives it back), what
        /// the measure found said of the fit as the probes gave it, before its aberrant figures are bounded.
        package init(date: String, seconds: Double, machine: Machine, cpuFeatures: [String: Int], settings: [String: String],
                    before: Conditions, after: Conditions, probes: CapacityProbes.Result) {
            let measurements = probes.measurements.rounded()
            let model = Self.fit(measurements, machine: machine)
            let found = Findings.of(model: Self.fit(measurements, machine: machine, bounded: false), measurements: measurements,
                                    judge: probes.judge, machine: machine, memory: probes.memory, before: before, after: after,
                                    seconds: seconds, regime: probes.regime)
            self.init(date: date, seconds: seconds, machine: machine, cpuFeatures: cpuFeatures, settings: settings, before: before,
                      after: after, model: model, measurements: measurements, judge: probes.judge, memory: probes.memory,
                      stages: probes.stages, findings: found, regime: probes.regime)
        }

        /// The model a set of measurements gives on this machine: the fit, the peak the inventory read, its aberrant figures bounded
        /// (`Findings.bounded`), to six digits.
        package static func fit(_ measurements: CapacityFit.Measurements, machine: Machine, bounded: Bool = true) -> CapacityModel {
            var model = CapacityFit.model(from: measurements, allocationUnit: NeuralEngine.rowUnit)
            if let peak = machine.gpuFP32Peak { model.crest[.gpu] = peak }
            return (bounded ? Findings.bounded(model) : model).rounded()
        }

        /// **The fit replayed from the measurements** — what an offline reading of a fleet's reports does: equal to `model` for a
        /// report this code wrote.
        package var refit: CapacityModel { Self.fit(measurements, machine: machine) }

        /// What the report is valid for: the machine and the binary that measured it.
        package var fingerprint: Fingerprint { Fingerprint(machine) }
        /// **The report is a map of this process's machine and build**: the current schema and the same fingerprint — what an app checks before
        /// designating its plan; when it does not, the app offers to measure again.
        package var isCurrent: Bool { schema == Diagnostic.schema && fingerprint == .current }

        /// The kernels the plan leaves out: those the judge refused, the engines beside the GPU of a set that computed wrong (the plan
        /// has no sharing for it, and would take the pair for free), and a kernel whose rate was hopelessly under the peak read.
        package var excluded: Set<CapacityModel.Kernel> {
            var kernels = Diagnostic.excluded(judge)
            for f in findings where f.kind == .aberrant && f.effect == .leftOut { if let k = f.kernel { kernels.insert(k) } }
            return kernels
        }

        /// **How far each kernel's and each set's calls agreed**, as they were measured: (p90 − p10) over the median of the worst of its
        /// chains — 0.02–0.06 on a free Mac. It is the uncertainty of the figure, not a reason to doubt it: a chain that disagreed more
        /// than the rule is read on its fast side. Only what was read: a step whose calls were too few or too long to compare (a spread of 0)
        /// and a step that is neither a kernel nor a set (the staircase) say nothing. Sorted by subject (`ane/fp16`, `gpu+ane`).
        package var uncertainty: [Uncertainty] {
            var worst: [String: Double] = [:]
            for (step, spread) in regime?.spreads ?? [:] where spread > 0 {
                let subject = String(step.prefix { $0 != " " })
                guard subject.split(separator: "+").allSatisfy({ CapacityModel.Kernel(String($0)) != nil || CapacityModel.Engine(rawValue: String($0)) != nil })
                else { continue }
                worst[subject] = max(worst[subject] ?? 0, spread)
            }
            return worst.map { Uncertainty(subject: $0.key, spread: $0.value) }.sorted { $0.subject < $1.subject }
        }
        package struct Uncertainty: Sendable, Equatable {
            /// A kernel (`ane/fp16`) or a set of engines or kernels (`gpu+ane`, `gpu/int8+ane/int8`).
            package var subject: String
            package var spread: Double
            /// The reading's plus-or-minus: half the spread.
            package var plusMinus: Double { spread / 2 }
        }

        /// **The cores the other programs used over the measure**, from the system's ticks; `nil` when the report does not say.
        package var foreignCores: Double? { regime?.foreign["run"] }

        /// **The plan of this Mac** (`CapacityPlan(model:)`): what the app designates (`EngineSettings.designate(plan:)`).
        package func plan(allowsANE: Bool = true) -> CapacityPlan {
            CapacityPlan(model: model, excluded: excluded, allowsANE: allowsANE)
        }

        /// **What each kernel reaches**, against the peak the inventory read where there is one (the GPU's): rate in FLOP/s, the
        /// fraction of the peak. Sorted by engine then form.
        package struct Reach: Sendable, Equatable {
            package var kernel: CapacityModel.Kernel
            package var rate: Double
            package var crest: Double?
            package var fraction: Double? { crest.map { rate / $0 } }
        }
        package var reached: [Reach] {
            model.lines.filter { $0.value.rate > 0 }.map { Reach(kernel: $0.key, rate: $0.value.rate, crest: model.crest[$0.key.engine]) }
                .sorted { ($0.kernel.engine, $0.kernel.form.rawValue) < ($1.kernel.engine, $1.kernel.form.rawValue) }
        }

        /// Every kernel judged right ran within its threshold: nothing excluded.
        package var passes: Bool { !judge.values.contains { $0.excluded } }

        /// **Reads a stored or received report**, `nil` for anything that is not a report of schema 22 — an older schema is not a map.
        package static func read(_ data: Data) -> Report? {
            struct Schema: Decodable { var schema: Int }
            guard let s = try? JSONDecoder().decode(Schema.self, from: data), s.schema == Diagnostic.schema else { return nil }
            return try? JSONDecoder().decode(Report.self, from: data)
        }

        /// **Stable JSON**: sorted keys, so two reports diff line by line.
        package func json(pretty: Bool = true) throws -> Data {
            let encoder = JSONEncoder()
            encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
                                              : [.sortedKeys, .withoutEscapingSlashes]
            return try encoder.encode(self)
        }

        /// **"Report my configuration"**: a new GitHub issue from the `diagnostic.yml` form, its fields prefilled by their ids —
        /// `chip`, `memory`, `macos` and `version`; the `report` field is the JSON, which the user pastes (it outgrew a link's
        /// ~8 000 characters), and the issue's body then carries it.
        package func issueURL() throws -> URL {
            Diagnostic.issueURL(fields: [
                ("template", "diagnostic.yml"),
                ("labels", "diagnostic"),
                ("title", "Diagnostic: \(machine.chip) · \(machine.memoryLabel) · macOS \(machine.macOS)"),
                ("chip", machine.chip + (machine.gpuCores > 0 ? " (\(machine.gpuCores)-core GPU)" : "")),
                ("memory", machine.memoryLabel),
                ("macos", machine.macOS),
                ("version", machine.version),
            ])
        }
    }

    // ── a finding, apart from where it is ───────────────────────────────────────────────────

    /// What kind of thing a finding is, once the kernel or the set it concerns is set aside — so that an app says each in
    /// its own language, with the name of the engine beside it, without reading the English of `detail`.
    package enum Cause: Sendable, Equatable {
        /// A kernel the inventory announces has no line (the probe did not run, or gave nothing): the plan does not offer it.
        case noFigure
        /// The reference engine's kernel computed wrong: the plan has no model of it, and the engine's defaults answer.
        case computedWrong
        /// A set of engines was not measured together: the plan does not offer the set.
        case notShared
        /// A figure that cannot be a measure of the silicon: a rate beyond the peak read or hopelessly under it, a fixed cost
        /// below zero, a keep on the fit's clamp, a staircase of the whole range. It is bounded, or the kernel is not offered, as the
        /// finding says; the usual cause is a Mac that was busy.
        case implausible
        /// The probes' peak went over the need the memory plan counted for them.
        case memoryOverNeed
        /// The need itself is over B.
        case memoryOverBudget
        case swapGrew
        case tookTooLong
        /// Something else used the Mac while it was measured and the ceiling could not measure the steps again: their figures are read on the
        /// fast side.
        case disturbed
    }

    /// The URL of a new issue with `fields` as its query, **every byte that delimits a field escaped**:
    /// `URLComponents` leaves `+` alone, which GitHub reads back as a space — in a JSON exponent (`1e+20`)
    /// or a chip name. `,` and `:` stay as they are: a query may hold them (RFC 3986, `pchar`), they
    /// delimit nothing in a form's value, and escaped they were a quarter of a report's link (`%2C`, `%3A`:
    /// ~1 200 of 7 700 bytes for a schema 6 map). `;` stays escaped: some servers split a query on it.
    package static func issueURL(fields: [(String, String)]) -> URL {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~,:")
        let query = fields.map { key, value in
            "\(key)=\(value.addingPercentEncoding(withAllowedCharacters: allowed) ?? "")"
        }.joined(separator: "&")
        return URL(string: "\(repository)/issues/new?\(query)")!
    }

    /// `x` to `digits` significant digits, so the JSON says `3.312`, not `3.3123456789012`; a
    /// non-finite value becomes the largest finite double (JSON has no infinity).
    package static func significant(_ x: Double, _ digits: Int) -> Double {
        guard x.isFinite else { return x.isNaN || x > 0 ? .greatestFiniteMagnitude : -.greatestFiniteMagnitude }
        guard x != 0 else { return 0 }
        return Double(String(format: "%.\(digits - 1)e", x)) ?? x
    }
}

extension Diagnostic.Finding {
    /// The kernel this finding concerns (`ane/fp16`), if it concerns one.
    package var kernel: CapacityModel.Kernel? { CapacityModel.Kernel(subject) }

    /// The engines this finding concerns: the kernel's, or those of the set (`gpu+ane`); none for the memory, the swap or the ceiling.
    package var engines: [CapacityModel.Engine] {
        if let kernel { return [kernel.engine] }
        return subject.split(separator: "+").compactMap { CapacityModel.Engine(rawValue: String($0)) }
    }

    /// **What kind of finding this is**, from the kind and the subject the measure wrote and the kernels the judge excluded
    /// (`Report.excluded`): a missing line of the reference engine is « computed wrong » when the judge left it out.
    package func cause(excluding excluded: Set<CapacityModel.Kernel>) -> Diagnostic.Cause {
        switch kind {
        case .line: return kernel.map(excluded.contains) == true ? .computedWrong : .noFigure
        case .sharing: return .notShared
        case .aberrant: return .implausible
        case .memory: return subject == "peak" ? .memoryOverNeed : .memoryOverBudget
        case .swap: return .swapGrew
        case .time: return .tookTooLong
        case .disturbed: return .disturbed
        }
    }
}
