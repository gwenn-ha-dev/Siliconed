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
/// Hence three rules:
///
/// 1. **These settings are compute branches kept for other chips** (M2–M5) that the reference
///    machine could not decide between: each one has its default measured here, and the other
///    branch stays because another chip may reverse the verdict. A path that is dead (read by
///    nothing) or wrong (a probe that renders a false image) has no such reason, and is removed.
/// 2. **The default is the reference machine's**, so that nothing moves unless asked: with no
///    profile and no environment variable, the engine takes the paths measured on the M1 Pro.
/// 3. **The priority is: default → profile → environment.** The profile is what a bench measured
///    on *this* machine (the developer's bench writes it); the environment keeps the last word,
///    because a measurement is taken by forcing a setting, never by editing a file.
///
/// The profile carries the fingerprint of the machine that produced it. **It is refused if the
/// model does not match** — a profile measured on an M4 Pro applied here would pass its constants
/// off as ours, and that is exactly the confusion this file exists to prevent.
public struct EngineSettings: Sendable {

    // ── what is tunable ────────────────────────────────────────────────────────────────────

    /// GPU + AMX co-execution (`SILICONED_AMX`). It gains 22 % on this machine — 512² 41.3 →
    /// 31.0 s — so **it is on by default on the chip it was measured on** (`referenceChip`):
    /// an app never runs the bench, and a first launch would otherwise render a third
    /// slower than the published timings. Elsewhere it stays off until the bench measures it: the
    /// frozen split (`amxFraction`) is this GPU's against this AMX, and a larger GPU would wait on
    /// the AMX. It changes the last bits, so the developer checks force it off.
    package var amx = false
    /// The chip `amx` and `amxFraction` were measured on (`machdep.cpu.brand_string`).
    package static let referenceChip = "Apple M1 Pro"
    /// The spinning flag, which returns **wrong** results. It serves only to make the check
    /// fail, and it has no place in a profile.
    package var amxSpin = false
    /// Two threads: one AMX block per P cluster on this machine (measured). A three-cluster chip would
    /// ask for three — it is a machine setting, not a law.
    package var amxThreads = 2
    /// The split's step in tokens. 256 leaves sixteen servo points at 4,128 rows.
    package var amxQuantum = 256
    /// Below that, the split costs more than the computation handed over (the text blocks run at m = 32).
    package var amxMinimumRows = 512
    /// **The fraction of rows handed to the GPU** — the servo's starting point, and the **frozen**
    /// value when the split is not servo-controlled (`frozenCut`, or a render's `reproducible`).
    ///
    /// 0.631 is the measured throughput ratio (3.364 / 5.333). It is a *machine* setting — it says
    /// where the split falls between this GPU and this AMX —, so a profile may write it. What it
    /// never changes is **the image**: each row of the output is computed entirely by a single
    /// engine, so moving the split redistributes the work without ever re-associating a sum. That
    /// is what makes the reproducible mode possible without turning the AMX off.
    package var amxFraction = 0.631
    /// **The frozen split** (`SILICONED_REPRODUCIBLE`): no more servo, `T = amxFraction` from the
    /// first GEMM to the last. Two renders of the same seed then yield the same bits.
    ///
    /// **And it is a profile setting, which calls for a justification** — the rule of profiles says a
    /// profile tunes speed and room, never the image. This setting does change the image's last
    /// bits… exactly like `amx` itself, which has been in the profile from the start. The
    /// forbidden family is that of the **approximations** — `spectral`, `vae_tuile`, `sdpa_fp16` —
    /// which change what one sees. Here the deviation is that of fp32 summation order, 2.5·10⁻⁵,
    /// and freezing it makes the image **stable** rather than making it different: what is
    /// unstable is the servo.
    ///
    /// ⚠️ Without this setting, `amxFraction` means nothing: the servo corrects it within a few
    /// blocks. The two are therefore swept together, or not at all.
    ///
    /// ## Default **true**, and a measurement decided it
    ///
    /// An A/B/A/B barrage on an entire 1024² render: `servo 114.4 / 116.6 s`, `frozen
    /// 116.2 / 116.4 s`. **+0.7 %** — when the gap between the two *servo* renders is **1.9 %** by
    /// itself. The cost is smaller than the variance of what it is compared to.
    ///
    /// What is bought for that price: the two frozen renders yield a PNG **identical byte for
    /// byte** (`md5 1fdd37f0…` twice), the two servo ones do not. A render becomes replayable
    /// again — hence verifiable by someone else, on the same machine, without taking our word.
    ///
    /// The servo is not buried for all that: `SILICONED_REPRODUCIBLE=0` turns it back on, and it
    /// keeps its reason for being — absorbing a thermal drift that 116 s do not show, and serving
    /// a machine whose throughput ratio is not this one's (rule 1 above).
    public var frozenCut = true

    /// The in-house matrix flash kernel (`FlashMatrix.swift`). **A tie** on this machine:
    /// 1.673 against 1.667 TFLOP/s, −132 MB. A chip with a larger fp32 accumulator budget would
    /// make it win — the measured cliff is a property of the chip.
    package var flash = false
    /// The number of rows per tile of the flash kernel; `nil` lets the kernel decide.
    package var flashRows: Int? = nil
    /// SDPA in half precision: 3.02 against 1.665 TFLOP/s, ~9 s on a render, and an error of
    /// 1.26·10⁻³ on `model_out`. **Set aside by decision, not by measurement**.
    package var sdpaFp16 = false

    /// Widening bf16 → fp32 on the GPU: +20 % in the engine here, because the widening is
    /// queued behind the GEMM, which is the bottleneck. On a machine where the CPU is the
    /// bottleneck — fewer P cores, or a more heavily used AMX — the verdict reverses.
    package var widenGPU = false
    /// The widening fused into the block's kernel.
    package var widenFused = false
    /// Keep the map's `MTLBuffer` wrappers: ~12 GB of footprint against ~112 ms per render.
    /// Ruinous on 16 GB, **probably a win at 32 or 64** — hence the setting.
    package var widenKeep = false

    /// The spectral schedule, **when forced**: `k` half-size evaluations, 0 = full schedule. Not
    /// forced, the product's rule decides (`Spectral.steps`). `k = 3` renders in 98.9 s instead of
    /// 140.1 at 1024², but free draws showed debris at `k = 3`: judged on one draw,
    /// the earlier verdict did not hold. A profile may not write it.
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

    /// The map's prefetcher. Turning it off takes the widening to 18.4 s.
    package var prefetch = true
    /// **The share of a DiT's layers whose pages are given back after use**, 0…1, counted
    /// from the end of the map. A map larger than what the page cache holds, swept in file order at
    /// every evaluation, misses everything under LRU: each evaluation reads it all again from disk.
    /// Handing the tail's pages back as soon as they are widened leaves the cache to the head, which
    /// then survives from one evaluation to the next. `madvise`/`msync` hints do not do it (the pages
    /// stay resident, the head goes): the tail is read with `pread` + `F_NOCACHE` into two staging
    /// buffers (`TailStream`, ~0.7–0.9 GB of anonymous memory during the denoising).
    ///
    /// 0.5, measured: Z-Image 512² 42.4 → 38.0 s, disk per evaluation 12.33 → 5.43 GB; 0.4
    /// thrashes at 1024² (the head no longer fits beside the activations), 0.6 and 1.0 gain less.
    /// Qwen-Image-2.1 512² 46.7 → 43.9 s. Same bits. 0 = everything through the map, as before.
    ///
    /// **In a render, this is the default plan's fraction**: the lean plan (a short budget at the
    /// launch, a short compressor during the render) streams the whole map instead (`MemoryPlan.tail`,
    /// `ditTail`). Outside a render (the checks) it serves as is; `SILICONED_MAP_TAIL` set in the
    /// environment forces it in a render too, at every size — a forcing of measurement, journaled.
    package var mapTail = 0.5
    /// The engine's thread count; 0 lets `Parallel` decide.
    package var parallel = 0

    /// **The render cache** (`RenderCache`, `SILICONED_CACHE`): what does not depend on the seed — the
    /// text stage's output, the image encoder's latents, Qwen-Image-2.1's K/V of the last edit — kept
    /// under `<racine>/cache/` and read back, bit for bit, by the next render that needs it. A setting
    /// of time and disk, never of the image: a profile may write it.
    public var cache = true

    // ── where each value comes from ───────────────────────────────────────────────────────────

    package enum Provenance: String, Sendable { case byDefault = "default", profile = "profile", environment = "env" }
    /// By setting name, where the active value comes from. It is what the developer's command line
    /// displays — a profile one cannot read cannot be debugged.
    package private(set) var provenance: [String: Provenance] = [:]
    /// The path of the retained profile, if there is one.
    package private(set) var profileRead: String? = nil
    /// What the profile says about the machine that measured it, if it mentions it.
    package private(set) var machineProfile: String? = nil
    /// **What the reading refused** — a profile from another machine, a setting a profile is not
    /// allowed to write. The library writes nothing to standard error: the caller displays them
    /// (the CLI at startup, an app wherever it likes).
    public private(set) var warnings: [String] = []

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

    /// The process's active settings, read once, on first access. The whole engine goes through here.
    public static let effective: EngineSettings = EngineSettings()

    /// The profile an app designated before the first access (`load(profile:)`).
    private static let designatedProfile = Mutex<String?>(nil)

    /// The profile's path: `SILICONED_PROFILE`, otherwise the one an app designated
    /// (`load(from:)`), otherwise the process library's `store/profil.json`
    /// (`Library.processWide`) — not the current directory's, which is `/` in an app.
    package static var profilePath: String {
        ProcessInfo.processInfo.environment["SILICONED_PROFILE"]
            ?? designatedProfile.withLock { $0 } ?? Library.processWide.profile
    }

    /// What `load` refuses.
    package enum Failure: Error, CustomStringConvertible, Equatable {
        /// The settings were already read, on another profile: designating this one would change nothing.
        case alreadyRead(profileRead: String, requested: String)
        package var description: String {
            switch self {
            case let .alreadyRead(loaded, requested):
                return "settings already read (profile \(loaded)): \(requested) comes too late — "
                    + "call EngineSettings.load(from:) at launch, before the first render"
            }
        }
    }

    /// **For an app: read the machine's profile from ITS library.**
    ///
    /// The settings are a property of the machine, read once for the whole process (`effective`) —
    /// there is no sense in changing them along the way, and the engine reads them deep in its
    /// loops. **Nothing reads them before a render**: neither building a `Request`, nor a
    /// `Model`, nor listing the catalog — only a render, `effective` and this `load` do. An app
    /// therefore calls this at launch, or at least before its first render. **Too late, it
    /// throws** (`EngineError.settingsAlreadyLoaded`) instead of silently doing nothing — unless the profile already
    /// read is this one. Without this call, the profile sought is `Library.processWide`'s,
    /// which the CLI follows and a sandboxed app would not find.
    public static func load(from library: Library) throws(EngineError) {
        let path = library.profile
        // Designate and read under the same lock: a render that read `effective` between the two
        // cannot make the result lie.
        let refusal: Failure? = designatedProfile.withLock { designated in
            if alreadyRead.load(ordering: .acquiring) {
                let loaded = ProcessInfo.processInfo.environment["SILICONED_PROFILE"]
                    ?? designated ?? Library.processWide.profile
                return loaded == path ? nil : .alreadyRead(profileRead: loaded, requested: path)
            }
            designated = path
            return nil
        }
        if let refusal { throw EngineError(refusal) }
        _ = effective   // read now: the warnings (profile from another machine) are ready
    }

    /// Set on the first read of `effective`: after it, designating a profile no longer serves any purpose.
    private static let alreadyRead = Atomic<Bool>(false)

    private init() {
        EngineSettings.alreadyRead.store(true, ordering: .releasing)
        machine = Machine.detected()
        amx = machine.chip == EngineSettings.referenceChip
        var refusal: String? = nil

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
        read("SILICONED_AMX", "amx")
        read("SILICONED_AMX_SPIN", "amx_spin")
        read("SILICONED_AMX_THREADS", "amx_threads")
        read("SILICONED_AMX_QUANTUM", "amx_quantum")
        read("SILICONED_AMX_MIN", "amx_min")
        read("SILICONED_AMX_FRACTION", "amx_fraction")
        read("SILICONED_REPRODUCIBLE", "coupe_figee")
        read("SILICONED_FLASH", "flash")
        read("SILICONED_FLASH_ROWS", "flash_rows")
        read("SILICONED_SDPA_FP16", "sdpa_fp16")
        read("SILICONED_WIDEN_GPU", "widen_gpu")
        read("SILICONED_WIDEN_FUSED", "widen_fused")
        read("SILICONED_WIDEN_KEEP", "widen_keep")
        read("SILICONED_SPECTRAL", "spectral")
        read("SILICONED_VAE_BLOCK", "vae_bloc_requetes")
        read("SILICONED_VAE_TILE", "vae_tuile")
        read("SILICONED_VAE_OVERLAP", "vae_recouvrement")
        read("SILICONED_PARALLEL", "parallel")
        read("SILICONED_CACHE", "cache")
        read("SILICONED_MAP_TAIL", "map_tail")
        if env["SILICONED_NO_PREFETCH"] != nil { fromEnv["prefetch"] = "0" }
        apply(fromEnv, .environment)

        if let refusal { warnings.insert(refusal, at: 0) }
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
    /// image is wanted: the spectral schedule, half precision and tiled decoding change the
    /// **result**, not just the path. Letting them into a profile would mean rendering a different
    /// image on two machines without anyone having asked for it — and a check that inherits the
    /// profile stops checking anything at all.
    ///
    /// The environment, for its part, can always set them: that is how a path is measured.
    /// Internal and not private: the test target verifies that every setting that changes the
    /// image is listed here. A rule that cannot be queried gets bypassed by oversight.
    static let outsideProfile: Set<String> = [
        "spectral",             // product choice: judged on the image, against the spectral oracle
        "sdpa_fp16",            // set aside by decision, not by measurement
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
        func optionalInteger(_ key: String, _ target: inout Int?) {
            guard let raw = values[key], let n = EngineSettings.integer(raw) else { return }
            target = n; provenance[key] = source
        }
        func real(_ key: String, _ target: inout Double) {
            guard let raw = values[key], let d = EngineSettings.real(raw) else { return }
            target = d; provenance[key] = source
        }
        flag("amx", &amx)
        flag("amx_spin", &amxSpin)
        integer("amx_threads", &amxThreads)
        integer("amx_quantum", &amxQuantum)
        integer("amx_min", &amxMinimumRows)
        real("amx_fraction", &amxFraction)
        flag("coupe_figee", &frozenCut)
        flag("flash", &flash)
        optionalInteger("flash_rows", &flashRows)
        flag("sdpa_fp16", &sdpaFp16)
        flag("widen_gpu", &widenGPU)
        flag("widen_fused", &widenFused)
        flag("widen_keep", &widenKeep)
        integer("spectral", &spectral)
        integer("vae_bloc_requetes", &vaeRequestBlock)
        integer("vae_tuile", &vaeTile)
        integer("vae_recouvrement", &vaeOverlap)
        flag("prefetch", &prefetch)
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
            ("amx", "\(amx)"), ("amx_threads", "\(amxThreads)"), ("amx_quantum", "\(amxQuantum)"),
            ("amx_min", "\(amxMinimumRows)"), ("amx_spin", "\(amxSpin)"),
            ("amx_fraction", String(format: "%.3f", amxFraction)),
            ("coupe_figee", "\(frozenCut)"),
            ("flash", "\(flash)"), ("flash_rows", flashRows.map { "\($0)" } ?? "auto"),
            ("sdpa_fp16", "\(sdpaFp16)"),
            ("widen_gpu", "\(widenGPU)"), ("widen_fused", "\(widenFused)"), ("widen_keep", "\(widenKeep)"),
            ("spectral", "\(spectral)"),
            ("vae_bloc_requetes", "\(vaeRequestBlock)"),
            ("vae_tuile", vaeTile == 0 ? "no" : "\(vaeTile) (overlap \(vaeOverlap))"),
            ("prefetch", "\(prefetch)"),
            ("map_tail", String(format: "%.2f", mapTail)),
            ("parallel", parallel == 0 ? "auto (2¹⁸)" : "\(parallel)"),
            ("cache", "\(cache)"),
        ]
        return rows.map { (key: $0.0, value: $0.1, source: provenance[$0.0] ?? .byDefault) }
    }

    /// The dictionary of active values, to write into a profile.
    package var values: [String: Any] {
        [
            "amx": amx, "amx_threads": amxThreads, "amx_quantum": amxQuantum,
            "amx_min": amxMinimumRows, "amx_fraction": amxFraction,
            "coupe_figee": frozenCut,
            "flash": flash, "sdpa_fp16": sdpaFp16,
            "widen_gpu": widenGPU, "widen_fused": widenFused, "widen_keep": widenKeep,
            "spectral": spectral, "vae_bloc_requetes": vaeRequestBlock,
            "prefetch": prefetch, "parallel": parallel, "map_tail": mapTail,
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
