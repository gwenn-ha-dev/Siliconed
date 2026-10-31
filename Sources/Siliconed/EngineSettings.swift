import Foundation
import Synchronization

/// **The engine's settings, in one place, and driven by the machine.**
///
/// This machine — M1 Pro, 16 GB, 14 GPU cores, 2 P clusters — is not the be-all and end-all of the
/// targets. More RAM changes what is kept resident; an M4/M5 Pro changes where the GPU/AMX split
/// falls; a slower SSD changes what the prefetcher must anticipate. A path that loses here may
/// win elsewhere, and **that is a reason to keep it as a conditional branch, not to throw it
/// away**: code that is not optimal *on this machine* is parameterized code, not dead code.
///
/// Hence four rules:
///
/// 1. **These settings are what a Mac's measure cannot say**: the spinning flag's fault injector,
///    the spectral schedule, the tiles, the page cache. Every *route* — which engine takes which rows of which product, which
///    kernel runs which attention — is the plan's (`plan`, `CapacityPlan`): exact rows asked at every product, never a
///    switch read once. A path that is dead (read by nothing) or wrong (a probe that renders a false image) has no reason
///    to stay, and is removed.
/// 2. **The default is the engine's own**, so that nothing moves unless asked: with no profile, no environment variable and no
///    model, the plan is `CapacityPlan.defaults` — the GPU alone on the widened weights, the attention cut by heads from 1 024 rows.
/// 3. **The priority is: default → profile → environment**, the plan apart: the model of this Mac is the plan of the report an app keeps
///    (`load(from:map:)`, before the first read, a report of this Mac and this build only), or a check's strictest case
///    (`designate(plan:)`); a profile is what a bench measured here (the developer's bench writes it); the environment keeps the last
///    word, because a measurement is taken by forcing a setting, never by editing a file.
/// 4. **Outside an app a plan is designated only when asked** — a developer's timing, which must be the app's: the command line,
///    the checks and the close-out otherwise run on the defaults, so their bits and the md5 judges do not depend on a
///    measurement of the day. `SILICONED_MAP=<report.json>` names a report for a timing, and takes the app's path.
///
/// The profile carries the fingerprint of the machine that produced it. **It is refused if it does not match** — a profile
/// measured on an M4 Pro applied here would pass its constants off as ours, and that is exactly the confusion this file exists
/// to prevent.
package struct EngineSettings: Sendable {

    // ── what is tunable ────────────────────────────────────────────────────────────────────

    /// The spinning flag, which returns **wrong** results. It serves only to make the check
    /// fail, and it has no place in a profile.
    package var amxSpin = false
    /// **The plan the process runs under** (`CapacityPlan`): what the engine asks, product by product and attention by attention —
    /// exact rows per engine and the kernel of each attention. The engine's defaults without a model; an app designates the model of
    /// its Mac (`designate(plan:)`) before anything reads the settings, and a check designates its strictest case the same way.
    package fileprivate(set) var plan = CapacityPlan.defaults

    /// The half-size start, **when forced** (`SILICONED_SPECTRAL`): `k` half-size evaluations of
    /// Z-Image, Qwen-Image-2.1, Anima or Krea 2, 0 = full schedule, within each plan's rules (`Spectral.forced`).
    /// Not forced, the product decides (`Spectral.halfSizeStart`). A measurement's
    /// setting: it changes the image, so a profile may not write it.
    package var spectral = 0

    /// **Splitting the VAE's attention into query blocks.** 0 = whole matrix.
    /// The bottleneck's `S×S` matrix weighs 2,520 MB at 1024²; the split is a **null**
    /// re-association (measured: worst channel 9.4·10⁻⁸) and trades it for a little time.
    package var vaeRequestBlock = 0

    /// **Tiled decoding: a tile's side, in LATENT pixels.** 0 = no tiles.
    ///
    /// It was the answer to the decoder's intermediates ∝ pixels (4,283 MB), until the decoder
    /// ran by bands, **exactly** — the same bits as one graph, 2.9 GB at 1024×1536: tiles no
    /// longer buy room, only another image.
    ///
    /// ⚠️ **But it is not exact, and that is why it does not turn on by itself.** The decoder's
    /// `GroupNorm`s normalize over the **whole plane**: cut up, each tile has its own statistics,
    /// and the bottleneck's attention is global too. The overlap glues the edges back together, it
    /// does not yield the same image — and a profile, by the rule of profiles, tunes speed and room,
    /// **never the image**. `vae_tuile` is therefore refused in a profile: it must be asked for
    /// explicitly, via `SILICONED_VAE_TILE`.
    package var vaeTile = 0
    /// The overlap between tiles, in latent pixels. One latent pixel is worth eight image pixels:
    /// eight here make sixty-four pixels of blend.
    package var vaeOverlap = 8

    /// **The share of a DiT's layers whose pages are given back after use**, 0…1, counted
    /// from the end of the map. A map larger than what the page cache holds, swept in file order at
    /// every evaluation, misses everything under LRU: each evaluation reads it all again from disk.
    /// Handing the tail's pages back as soon as they are widened leaves the cache to the head, which
    /// then survives from one evaluation to the next. `madvise`/`msync` hints do not do it (the pages
    /// stay resident, the head goes): the tail is read with `pread` + `F_NOCACHE` into staging
    /// buffers (`TailStream.unitSlots` of them, ~1 GB of anonymous memory during the denoising).
    ///
    /// 0.5, measured: Z-Image 512² 42.4 → 38.0 s, disk per evaluation 12.33 → 5.43 GB; 0.4
    /// thrashes at 1024² (the head no longer fits beside the activations), 0.6 and 1.0 gain less.
    /// Qwen-Image-2.1 512² 46.7 → 43.9 s. Same bits. 0 = everything through the map, as before.
    ///
    /// **In a render, this is the cap of the map's head**: the head is what B leaves beside the
    /// denoising's need, never more than this split keeps — less on a smaller Mac, the whole map
    /// cached when it fits (`MemoryPlan.released`, `ditTail`). Outside a render (the checks) it serves
    /// as is; `SILICONED_MAP_TAIL` set in the environment forces it in a render too, at every size — a
    /// forcing of measurement, journaled.
    package var mapTail = 0.5
    /// The engine's thread count; 0 lets `Parallel` decide.
    package var parallel = 0

    /// **The render cache** (`RenderCache`, `SILICONED_CACHE`): what does not depend on the seed — the
    /// text stage's output, the image encoder's latents, Qwen-Image-2.1's K/V of the last edit — kept
    /// under `<root>/cache/` and read back, bit for bit, by the next render that needs it. A setting
    /// of time and disk, never of the image: a profile may write it.
    package var cache = true

    // ── where each value comes from ───────────────────────────────────────────────────────────

    package enum Provenance: String, Sendable { case byDefault = "default", map = "map", profile = "profile", environment = "env" }
    /// By setting name, where the active value comes from. It is what the developer's command line
    /// displays — a profile one cannot read cannot be debugged.
    package private(set) var provenance: [String: Provenance] = [:]
    /// The path of the retained profile, if there is one.
    package private(set) var profileRead: String? = nil
    /// What the profile says about the machine that measured it, if it mentions it.
    package private(set) var machineProfile: String? = nil
    /// The date of the report whose model is the plan, if one was designated and is this Mac's, this build's (`load(from:map:)`).
    package fileprivate(set) var mapRead: String? = nil
    /// A check designated its plan, or the environment forced one: a measure does not replace it.
    fileprivate var planFixed = false
    /// For the tests: a plan a check designated is not replaced by a measure.
    mutating func planFixedForTesting() { planFixed = true }
    /// **What the reading refused** — a profile from another machine, a setting a profile is not
    /// allowed to write. The library writes nothing to standard error: the caller displays them
    /// (the CLI at startup, an app wherever it likes).
    package fileprivate(set) var warnings: [String] = []

    // ── the machine ─────────────────────────────────────────────────────────────────────────

    package struct Machine: Sendable {
        /// `machdep.cpu.brand_string`: "Apple M1 Pro".
        package let chip: String
        package let hardwareModel: String
        package let ramBytes: Int
        package let cores: Int
        package let performanceCores: Int

        package static func detected() -> Machine {
            func sysctlString(_ name: String) -> String {
                var size = 0
                sysctlbyname(name, nil, &size, nil, 0)
                guard size > 0 else { return "unknown" }
                var buffer = [CChar](repeating: 0, count: size)
                sysctlbyname(name, &buffer, &size, nil, 0)
                return String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
            }
            func integer(_ name: String) -> Int {
                var value: Int64 = 0
                var size = MemoryLayout<Int64>.size
                if sysctlbyname(name, &value, &size, nil, 0) == 0 { return Int(value) }
                var small: Int32 = 0
                size = MemoryLayout<Int32>.size
                if sysctlbyname(name, &small, &size, nil, 0) == 0 { return Int(small) }
                return 0
            }
            return Machine(chip: sysctlString("machdep.cpu.brand_string"), hardwareModel: sysctlString("hw.model"),
                           ramBytes: integer("hw.memsize"),
                           cores: integer("hw.ncpu"),
                           performanceCores: integer("hw.perflevel0.logicalcpu"))
        }

        /// **In MiB, not MB.** The 16 GB the spec sheet announces are 16,384 MiB; dividing by
        /// 10⁶ would display 17,179 and suggest a machine nobody has.
        package var ramMiB: Int { ramBytes / 1024 / 1024 }
        package var summary: String {
            "\(hardwareModel) · \(ramMiB) MiB · \(cores) cores (\(performanceCores) P + "
                + "\(cores - performanceCores) E)"
        }
    }

    package let machine: Machine

    // ── reading ─────────────────────────────────────────────────────────────────────────

    /// The process's active settings, read on first access. The whole engine goes through here. **Only the plan can change afterwards**
    /// (`adopt(map:)`, between two renders, never during one); the profile and the environment are read once.
    package static var effective: EngineSettings {
        #if DEBUG
        if readsForbidden { forbiddenReads.add(1, ordering: .relaxed) }
        #endif
        return live.current
    }
    private static let live = SettingsStore(EngineSettings())

    /// **A finished measure applies to the next render** (`AppState.keep`): the report's plan replaces the process's, now when no render
    /// runs, else at `applyPendingMap()` — which the app calls before it starts a job. A render reads one plan from its first line to
    /// its last: `Engine.execute` counts itself in (`renderBegan`), and nothing is replaced while a count is open.
    @discardableResult
    package static func adopt(map report: Diagnostic.Report) -> Adoption { live.adopt(report) }
    /// Applies the measure `adopt` deferred, if no render runs. Returns whether the plan changed.
    @discardableResult
    package static func applyPendingMap() -> Bool { live.applyPending() }
    /// A render begins / ends (`Engine.execute`): the plan is not replaced in between.
    package static func renderBegan() { live.renderBegan() }
    package static func renderEnded() { live.renderEnded() }
    /// Counts each replacement of the plan: what the app watches to recompute what it showed (needs, the Neural Engine's offer).
    package static var generation: Int { live.generation }

    package enum Adoption: Sendable, Equatable {
        /// The plan is the report's from now on.
        case applied
        /// A render runs: the report waits for `applyPendingMap()`.
        case deferred
        /// Nothing to replace: the plan was designated or forced (a check, `SILICONED_PLAN`), or the report is not this Mac's.
        case unchanged
    }

    /// The prefix of the warning for a report that is not this Mac's: dropped when a later report replaces it.
    static let staleMapWarning = "⚠ the capacity model of"

    #if DEBUG
    /// **A tripwire for what must not read the settings** (debug builds only): inside
    /// `$readsForbidden.withValue(true)`, each read of `effective` is counted. A plan asked before the
    /// capacity map froze the process on the defaults once (the map then refused, the AMX off for the
    /// session: the first read designates the profile and the first plan, and a map measured later can only replace the plan between two renders);
    /// the first read cannot be watched in a test process where others already read, every read can.
    @TaskLocal package static var readsForbidden = false
    package static let forbiddenReads = Atomic<Int>(0)
    #endif

    /// What an app or a check designated before the first access (`load(from:map:)`, `designate(plan:)`): its profile, its report,
    /// the plan the process runs under.
    fileprivate struct Designation { var profile: String?; var map: Diagnostic.Report?; var plan: CapacityPlan? }
    fileprivate static let designated = Mutex(Designation())

    /// The profile's path: `SILICONED_PROFILE`, otherwise the one an app designated
    /// (`load(from:)`), otherwise the process library's `store/profil.json`
    /// (`Library.processWide`) — not the current directory's, which is `/` in an app.
    package static var profilePath: String {
        ProcessInfo.processInfo.environment["SILICONED_PROFILE"]
            ?? designated.withLock { $0.profile } ?? Library.processWide.profile
    }

    /// What `load` refuses.
    package enum Failure: Error, CustomStringConvertible, Equatable {
        /// The settings were already read, on another profile or another map: designating this one
        /// would change nothing.
        case alreadyRead(profileRead: String, requested: String)
        package var description: String {
            switch self {
            case let .alreadyRead(loaded, requested):
                return "settings already read (profile \(loaded)): \(requested) comes too late — "
                    + "call EngineSettings.load(from:) at launch, before the first render"
            }
        }
    }

    /// **For an app: read the machine's profile from ITS library, and run on the model its report holds.**
    ///
    /// The settings are a property of the machine, read for the whole process (`effective`) — the engine reads them
    /// deep in its loops, so a render must see one set from its first line to its last: **only the plan is replaced
    /// afterwards, between two renders** (`adopt(map:)`). **Nothing reads them before a render** but a render's memory need: neither building a
    /// `Request`, nor a `Model`, nor listing the catalog — a render, `ModelCard.memoryNeed`,
    /// `effective` and this `load` do. An app therefore calls this at launch, before it shows a need or
    /// renders. **Too late, it throws** (`EngineError.settingsAlreadyLoaded`) instead of silently doing
    /// nothing — unless the profile and the map already read are these. Without this call, the profile
    /// sought is `Library.processWide`'s, which the CLI follows and a sandboxed app would not find.
    ///
    /// `map`: the diagnostic's report this app kept (`Diagnostic.run`). **Refused with a warning when it is not
    /// `isCurrent`** (an older schema, another machine, another macOS, another build): the defaults then apply, and the
    /// app measures again. Taken, its `model` is the plan of the process (`Report.plan`); a process without the ANE restricts it
    /// (`CapacityPlan.restricted(allowingANE:)`). **It does not replace settings already read** — a later measure goes through
    /// `adopt(map:)`, between two renders —: an app without a current report defers this call until its measure, taken in
    /// another process, so that nothing here reads the settings before it.
    package static func load(from library: Library, map: Diagnostic.Report? = nil) throws(EngineError) {
        let path = library.profile
        // Designate and read under the same lock: a render that read `effective` between the two
        // cannot make the result lie.
        let refusal: Failure? = designated.withLock { designated in
            if alreadyRead.load(ordering: .acquiring) {
                let loaded = ProcessInfo.processInfo.environment["SILICONED_PROFILE"]
                    ?? designated.profile ?? Library.processWide.profile
                if loaded != path { return .alreadyRead(profileRead: loaded, requested: path) }
                if designated.map != map {
                    func named(_ m: Diagnostic.Report?) -> String { m.map { ", map of \($0.date)" } ?? ", no map" }
                    return .alreadyRead(profileRead: loaded + named(designated.map), requested: path + named(map))
                }
                return nil
            }
            designated.profile = path
            designated.map = map
            return nil
        }
        if let refusal { throw EngineError(refusal) }
        _ = effective   // read now: the warnings (profile from another machine, stale map) are ready
    }

    /// **Designate the plan the process runs under** — the model of this Mac (`CapacityPlan(model:)`), or a check's strictest case
    /// (`CapacityPlan.forcing`) — before anything reads the settings. **Too late, it throws**, unless the plan already read is this one.
    package static func designate(plan: CapacityPlan) throws(EngineError) {
        let late: Bool = designated.withLock { designated in
            if alreadyRead.load(ordering: .acquiring) { return true }
            designated.plan = plan
            return false
        }
        if late {
            // Compared outside the lock: the first read of `effective` takes it.
            let loaded = live.current.plan
            if loaded !== plan && loaded.identity != plan.identity {
                throw EngineError(Failure.alreadyRead(profileRead: "plan \(loaded.identity)", requested: "plan \(plan.identity)"))
            }
        }
        _ = effective
    }

    /// Set on the first read of `effective`: after it, designating a profile no longer serves any purpose.
    private static let alreadyRead = Atomic<Bool>(false)

    /// **What a report designates**: the plan of its model when it is a report of this Mac and this build (`Report.isCurrent`), else a
    /// warning and nothing — the defaults apply. Pure: no process state.
    static func designation(of report: Diagnostic.Report) -> (plan: CapacityPlan?, warning: String?) {
        if report.isCurrent { return (report.plan(), nil) }
        return (nil, "\(staleMapWarning) \(report.date) (schema \(report.schema)) was measured on \(report.fingerprint), this is "
                     + "\(Diagnostic.Fingerprint.current) — ignored, the defaults apply.")
    }

    private init() {
        EngineSettings.alreadyRead.store(true, ordering: .releasing)
        machine = Machine.detected()
        var refusal: String? = nil

        // ── the capacity map, if an app designated one and it is this machine's, this build's ────────
        let designation = EngineSettings.designated.withLock { $0 }
        var chosen = designation.plan != nil          // a plan was designated, or the environment forces one: a model does not replace it
        if let plan = designation.plan { self.plan = plan }
        var report = designation.map
        // A developer's timing names a report: the app's path, a file instead of its preferences (`SILICONED_MAP=1` is the command line's
        // own way to measure and keep one, `CapacityMapCLI.swift`).
        if let named = ProcessInfo.processInfo.environment["SILICONED_MAP"], named != "1", !named.isEmpty {
            if let data = FileManager.default.contents(atPath: named), let read = Diagnostic.Report.read(data) { report = read }
            else { warnings.append("⚠ SILICONED_MAP=\(named) is not a report of schema \(Diagnostic.schema) — the defaults apply.") }
        }
        // The environment keeps the last word: a developer's check or judge names the route it forces.
        if let named = ProcessInfo.processInfo.environment["SILICONED_PLAN"] {
            if let forced = CapacityPlan.forcing(named: named) { self.plan = forced; chosen = true }
            else if !named.isEmpty {
                warnings.append("⚠ SILICONED_PLAN=\(named) not read (amx, flash, ane, aneint8, int8, bf16act, comma-separated): the plan stays.")
            }
        }
        planFixed = chosen
        if let report {
            let taken = EngineSettings.designation(of: report)
            if let warning = taken.warning { warnings.append(warning) }
            if let plan = taken.plan {
                mapRead = report.date
                // The model's plan, unless a check designated or forced its own (which is stricter).
                if !chosen { self.plan = plan }
            }
        }

        // ── the profile, if it exists and if it speaks of this machine ──────────────────────────
        let path = EngineSettings.profilePath
        if let data = FileManager.default.contents(atPath: path),
           let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            // on-disk keys (`machine.modele`, `reglages`, …), kept for existing profiles
            let described = (root["machine"] as? [String: Any])?["modele"] as? String
            machineProfile = described
            if let described, described != machine.hardwareModel {
                refusal = "⚠ \(path) was measured on \(described), the machine is \(machine.hardwareModel)"
                       + " — profile ignored."
            } else if let values = root["reglages"] as? [String: Any] {
                profileRead = path
                apply(values, .profile)
            }
        }

        // ── the environment, which always has the last word ─────────────────────────────────
        let env = ProcessInfo.processInfo.environment
        var fromEnv: [String: Any] = [:]
        func read(_ variable: String, _ key: String) {
            if let raw = env[variable] { fromEnv[key] = raw }
        }
        // The keys (second argument) are those of `store/profil.json`: on-disk keys, kept for existing profiles.
        read("SILICONED_AMX_SPIN", "amx_spin")
        read("SILICONED_SPECTRAL", "spectral")
        read("SILICONED_VAE_BLOCK", "vae_bloc_requetes")
        read("SILICONED_VAE_TILE", "vae_tuile")
        read("SILICONED_VAE_OVERLAP", "vae_recouvrement")
        read("SILICONED_PARALLEL", "parallel")
        read("SILICONED_CACHE", "cache")
        read("SILICONED_MAP_TAIL", "map_tail")
        apply(fromEnv, .environment)

        if let refusal { warnings.insert(refusal, at: 0) }
    }

    /// **Settings from values alone** — the defaults, then `values`: no environment, no profile, no process-wide state. What the
    /// pure tests read.
    package init(applying values: [String: Any]) {
        machine = Machine.detected()
        apply(values, .map)
    }

    /// **A flag is read by its presence, and taken down with `=0`.** Historically the engine
    /// tested `!= nil`: setting the variable, whatever its value, turned the path on. That
    /// remains true — except that `=0`, `=no` and `=false` now turn it off, which is necessary as
    /// soon as a profile can turn something on: without it, one could not contradict it.
    private static func flag(_ raw: Any) -> Bool {
        if let b = raw as? Bool { return b }
        if let n = raw as? Int { return n != 0 }
        let text = "\(raw)".lowercased()
        return !["0", "no", "false", "off", ""].contains(text)
    }

    private static func integer(_ raw: Any) -> Int? {
        if let n = raw as? Int { return n }
        if let d = raw as? Double { return Int(d) }
        return Int("\(raw)")
    }

    private static func real(_ raw: Any) -> Double? {
        if let d = raw as? Double { return d }
        if let n = raw as? Int { return Double(n) }
        return Double("\(raw)")
    }

    /// **What a profile is not allowed to write.**
    ///
    /// A profile says what a *machine* can do fast and where it has room. It does not say which
    /// image is wanted: the spectral schedule and tiled decoding change the
    /// **result**, not just the path. Letting them into a profile would mean rendering a different
    /// image on two machines without anyone having asked for it — and a check that inherits the
    /// profile stops checking anything at all.
    ///
    /// The environment, for its part, can always set them: that is how a path is measured.
    /// Internal and not private: the test target verifies that every setting that changes the
    /// image is listed here. A rule that cannot be queried gets bypassed by oversight.
    static let outsideProfile: Set<String> = [
        "spectral",             // product choice: judged on the image
        "vae_tuile",            // changes the image: per-tile GroupNorm, per-tile attention
        "vae_recouvrement",
        "amx_spin",             // fault injector: the spinning flag's counter-proof
    ]

    private mutating func apply(_ values: [String: Any], _ source: Provenance) {
        var values = values
        if source == .profile {
            for key in values.keys where EngineSettings.outsideProfile.contains(key) {
                values.removeValue(forKey: key)
                warnings.append("⚠ « \(key) » ignored in the profile: a profile tunes speed "
                                      + "and room, never the image.")
            }
        }
        func flag(_ key: String, _ target: inout Bool) {
            guard let raw = values[key] else { return }
            target = EngineSettings.flag(raw); provenance[key] = source
        }
        func integer(_ key: String, _ target: inout Int) {
            guard let raw = values[key], let n = EngineSettings.integer(raw) else { return }
            target = n; provenance[key] = source
        }
        func real(_ key: String, _ target: inout Double) {
            guard let raw = values[key], let d = EngineSettings.real(raw) else { return }
            target = d; provenance[key] = source
        }
        flag("amx_spin", &amxSpin)
        integer("spectral", &spectral)
        integer("vae_bloc_requetes", &vaeRequestBlock)
        integer("vae_tuile", &vaeTile)
        integer("vae_recouvrement", &vaeOverlap)
        integer("parallel", &parallel)
        flag("cache", &cache)
        real("map_tail", &mapTail)
        mapTail = min(max(mapTail, 0), 1)
    }

    // ── what it looks like on screen, and on disk ────────────────────────────────────────

    /// The active settings, one per line, with their provenance. Default values are left out
    /// unless asked: what one wants to see in a second is **what is not the default**.
    package func report(all: Bool = false) -> String {
        var output = "machine : \(machine.summary)\n"
        if let mapRead { output += "map     : \(mapRead)\n" }
        output += "plan    : \(plan.identity)\n"
        output += "profile : \(profileStatus)\n"
        for row in rows where all || row.source != .byDefault {
            // `%s` wants a `char*` and would yield gibberish on non-ASCII text: we pad in Swift.
            output += "  " + row.key.padding(toLength: 20, withPad: " ", startingAt: 0)
                    + row.value.padding(toLength: 12, withPad: " ", startingAt: 0)
                    + row.source.rawValue + "\n"
        }
        if !all && provenance.isEmpty { output += "  (nothing is forced)\n" }
        return output
    }

    /// Which profile was read, refused, or none — what `report` and the diagnostic say.
    package var profileStatus: String {
        if let profileRead { return profileRead }
        if let machineProfile { return "refused (measured on \(machineProfile))" }
        return "none (\(EngineSettings.profilePath) missing) — everything is at the default measured on the reference"
    }

    /// Every setting, its value as `report` prints it, and where it comes from.
    package var rows: [(key: String, value: String, source: Provenance)] {
        let rows: [(String, String)] = [
            ("amx_spin", "\(amxSpin)"),
            ("spectral", "\(spectral)"),
            ("vae_bloc_requetes", "\(vaeRequestBlock)"),
            ("vae_tuile", vaeTile == 0 ? "no" : "\(vaeTile) (overlap \(vaeOverlap))"),
            ("map_tail", String(format: "%.2f", mapTail)),
            ("parallel", parallel == 0 ? "auto (2¹⁸)" : "\(parallel)"),
            ("cache", "\(cache)"),
        ]
        return rows.map { (key: $0.0, value: $0.1, source: provenance[$0.0] ?? .byDefault) }
    }

    /// The dictionary of active values, to write into a profile.
    package var values: [String: Any] {
        [
            "spectral": spectral, "vae_bloc_requetes": vaeRequestBlock,
            "parallel": parallel, "map_tail": mapTail,
        ]
    }

    /// Writes a profile. `measurements` is what the bench recorded — a profile without its measurements
    /// is a file of constants nobody will dare contradict.
    package static func writeProfile(_ values: [String: Any], measurements: [String: Any],
                                    to path: String) throws {
        let machine = Machine.detected()
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        // on-disk keys, kept for existing maps/profiles (read back by `init`)
        let root: [String: Any] = [
            "machine": ["modele": machine.hardwareModel, "ram_mio": machine.ramMiB,
                        "coeurs": machine.cores, "coeurs_p": machine.performanceCores],
            "mesure": ["date": formatter.string(from: Date()), "banc": "siliconed-dev bench"],
            "reglages": values,
            "mesures": measurements,
        ]
        let data = try JSONSerialization.data(withJSONObject: root,
                                                 options: [.prettyPrinted, .sortedKeys])
        try FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent,
                                                withIntermediateDirectories: true)
        try data.write(to: URL(fileURLWithPath: path))
    }
}

/// **The settings of the process and the turn of their replacement.** A render counts itself in; the plan of a new measure is put in
/// place only when no render is counted, so that one render reads one plan whatever the clock says. What a plan fixes and survives
/// a render (`RenderCache` keys carry the plan's identity; a model, a pipeline, a GEMM and its Neural Engine programs are built
/// per render) follows without a purge. A class and not a global, so that a test builds its own.
final class SettingsStore: Sendable {
    private struct Turn { var renders = 0; var pending: Diagnostic.Report?; var generation = 0 }
    private let settings: Mutex<EngineSettings>
    private let turn = Mutex(Turn())

    init(_ initial: EngineSettings) { settings = Mutex(initial) }

    var current: EngineSettings { settings.withLock { $0 } }
    var generation: Int { turn.withLock { $0.generation } }
    var rendering: Bool { turn.withLock { $0.renders > 0 } }

    func renderBegan() { turn.withLock { $0.renders += 1 } }
    func renderEnded() { turn.withLock { $0.renders = max(0, $0.renders - 1) } }

    func adopt(_ report: Diagnostic.Report) -> EngineSettings.Adoption {
        turn.withLock { turn in
            if turn.renders > 0 { turn.pending = report; return .deferred }
            turn.pending = nil
            return replace(with: report, in: &turn) ? .applied : .unchanged
        }
    }

    @discardableResult
    func applyPending() -> Bool {
        turn.withLock { turn in
            guard turn.renders == 0, let report = turn.pending else { return false }
            turn.pending = nil
            return replace(with: report, in: &turn)
        }
    }

    /// Under the turn's lock. The plan and the date of the report it came from; the profile's values and the environment's stay.
    private func replace(with report: Diagnostic.Report, in turn: inout Turn) -> Bool {
        let taken = EngineSettings.designation(of: report)
        let changed = settings.withLock { s -> Bool in
            s.warnings.removeAll { $0.hasPrefix(EngineSettings.staleMapWarning) }
            if let warning = taken.warning { s.warnings.append(warning) }
            guard !s.planFixed, let plan = taken.plan else { return false }
            let changed = s.plan.identity != plan.identity || s.mapRead != report.date
            s.plan = plan; s.mapRead = report.date
            return changed
        }
        if changed {
            EngineSettings.designated.withLock { $0.map = report }
            turn.generation += 1
        }
        return changed
    }
}
