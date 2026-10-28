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
/// 1. **These settings are compute branches kept for other chips** (M2–M5) that the reference
///    machine could not decide between: each one has its default measured here, and the other
///    branch stays because another chip may reverse the verdict. A path that is dead (read by
///    nothing) or wrong (a probe that renders a false image) has no such reason, and is removed.
/// 2. **The default is the reference machine's**, so that nothing moves unless asked: with no
///    profile and no environment variable, the engine takes the paths measured on the M1 Pro —
///    except `amx`, off everywhere: the GPU/AMX balance is the one thing a Mac's measure must say.
/// 3. **The priority is: default → map → profile → environment.** The map is the diagnostic's
///    capacity map of *this* machine, which only an app designates (`load(from:map:)`) and the rules
///    of `CapacityRules.swift` turn into `amx`, `amx_fraction`, `flash`, `head_split`, `gemm_tensor`; the profile is
///    what a bench measured here (the developer's bench writes it); the environment keeps the last
///    word, because a measurement is taken by forcing a setting, never by editing a file.
/// 4. **Outside an app a map is designated only when asked** — a developer's timing, which must be the
///    app's: the command line, the checks and the close-out otherwise run on the defaults, so their bits and
///    the md5 judges do not depend on a measurement of the day.
///
/// The profile carries the fingerprint of the machine that produced it, the map a finer one
/// (`Diagnostic.Fingerprint`). **Each is refused if it does not match** — a profile measured on an
/// M4 Pro applied here would pass its constants off as ours, and that is exactly the confusion this
/// file exists to prevent.
package struct EngineSettings: Sendable {

    // ── what is tunable ────────────────────────────────────────────────────────────────────

    /// GPU + AMX co-execution (`SILICONED_AMX`). It gains 22 % on the M1 Pro — 512² 41.3 → 31.0 s —
    /// and **it is off by default on every chip, the M1 Pro included**: whether it pays is the GPU's
    /// throughput against the AMX's, which only a measure of the Mac says — in the app the capacity map
    /// (`CapacityRules.swift`, which also sets `amxFraction`), taken before the first render; for a
    /// developer `SILICONED_AMX=1` or the bench's profile. Never a chip's name: a frozen split measured
    /// on one GPU makes a larger one wait on the AMX. It changes the last bits, so the developer checks
    /// force it off.
    package var amx = false
    /// The spinning flag, which returns **wrong** results. It serves only to make the check
    /// fail, and it has no place in a profile.
    package var amxSpin = false
    /// The split's step in tokens. 256 leaves sixteen servo points at 4,128 rows. Under a map's grid
    /// (`amxShares`) the map sets 64, MPS's row tile: at a 512² image's ~1 040 rows, 256 only offers the GPU
    /// 0.49 or 0.74 of the rows, and the computed shares fall between.
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
    /// **The frozen cut computed per product** (`Conductor.Shares`: `T = 1 / (1 + ρ(k, n, m))`, `ρ` measured by
    /// the map on this Mac); none by default — no machine's figures in the code —, and then `amxFraction`.
    /// `SILICONED_AMX_SHARES=<k×n@m=ρ,…>` hands the command line a map's grid.
    package var amxShares = Conductor.Shares.none
    /// **The frozen split** (`SILICONED_REPRODUCIBLE`): no more servo, `T = amxFraction` from the
    /// first GEMM to the last. Two renders of the same seed then yield the same bits.
    ///
    /// **And it is a profile setting, which calls for a justification** — the rule of profiles says a
    /// profile tunes speed and room, never the image. This setting does change the image's last
    /// bits… exactly like `amx` itself, which has been in the profile from the start. The
    /// forbidden family is that of the **approximations** — `spectral`, `vae_tuile` —
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
    package var frozenCut = true

    /// The in-house matrix flash kernel (`FlashMatrix.swift`). **A tie** on this machine:
    /// 1.673 against 1.667 TFLOP/s, −132 MB. A chip with a larger fp32 accumulator budget would
    /// make it win — the measured cliff is a property of the chip.
    package var flash = false
    /// **The SDPA cut by heads between the GPU and the CPU** (`HeadSplitAttention`) from S = 1024 on;
    /// off, those attentions run in one MPSGraph SDPA, as below 1024. ×1.5 on this machine (the map,
    /// 14/10/2026: 2.33–2.44 against 1.57–1.63 TFLOP/s), hence on by default; a chip whose GPU leaves
    /// the CPU far behind would reverse it.
    package var headSplit = true
    /// **The share of heads the CPU takes** under the cut (`head_split_fraction`): 0.4 as swept on the M1 Pro
    /// (Z-Image 12 of 30, Anima 6 of 16, Qwen 13 of 32, Krea 2 5 of 12 groups), the map's where its figures say
    /// another share beats it by more than the margin (`HeadSplitAttention.balance`). Frozen, not servoed: a share
    /// decides which engine computes a head, hence the bits.
    package var headSplitFraction = HeadSplitAttention.referenceFraction
    /// **The DiT's products read the bf16 weights unwidened** (`TensorGEMM`: Metal 4's `matmul2d`,
    /// fp32 activations × bf16 weights → fp32), the AMX off. Off by default: slower than MPS on this
    /// machine (2.36 against 3.46 TFLOP/s); the map turns it on where it measures it faster.
    package var gemmTensor = false
    /// **The products read the published int8 weights as they are** (`TensorGEMM`, `float × int8_t`, the
    /// scale after the sums, a convrot rotation moved to the activation), the AMX off on them. Off by
    /// default, as `gemm_tensor`: a path the map turns on where it measures it faster.
    package var gemmInt8 = false
    /// **The products read GGUF Q8_0 as published** (`TensorGEMM.linear_q8`: q · s per block of 32 rows
    /// into threadgroup memory, then `matmul2d`), beside the AMX on its columns. Off: on the M1 Pro,
    /// ×0.29 the widened path's speed at 4 096 rows — the threadgroup fp32 operand runs at half MPS's
    /// rate and its fill costs as much again. A branch the map turns on where it measures it faster
    /// (schema 9), apart from `gemm_int8`: on the M1 Pro the one wins and the other loses.
    package var gemmQ8 = false
    /// **The DiT's products take their activation rounded to bf16** (`BFloatKernels`: `matmul2d`
    /// `bfloat × bfloat → float` on the published bf16 weights, `bfloat × int8_t` on the 8-bit ones; sums and
    /// outputs fp32). The weights are read as published, as `gemm_tensor`, and **the operand of a product is
    /// all that is rounded**: norms, softmax, the residual stream and the modulation stay fp32. P20 allows
    /// bf16 between two operations. Off by default: a Mac without a bf16 matrix unit loses by it (the M1 Pro,
    /// ×0.5); the map turns it on where it measures it faster, and `siliconed check … --bf16-activations`
    /// judges it against diffusers in bf16.
    package var gemmBF16Activations = false
    /// Below this height a product keeps its fp32 activation: the modulation and the embedders (a handful
    /// of rows) gain nothing, and are the operands the error would be most amplified through.
    package var bf16ActivationMinimumRows = 256
    /// The published 8-bit weights (`gemm_int8`, `gemm_q8`) take the rounded activation too (`matmul2d`
    /// `bfloat × int8_t`). Off, and **decided by no map**: it measures the bf16 weights only. Forced by a check.
    package var bf16ActivationEightBit = false
    /// **The Neural Engine beside the GPU on the widened products** (`NeuralEngine`: the last rows in fp16,
    /// the weight's columns lifted by powers of two, a non-finite sent back to the GPU). Off by default, as
    /// every compute branch: the map turns it on where the pair beats the GPU alone and the ANE's rows hold
    /// the half-precision threshold. Only on the widened path: a weight read as published (`gemm_tensor`,
    /// `gemm_int8`, `gemm_q8`) keeps its own product.
    package var gemmANE = false
    /// The GPU's share of the rows when `gemm_ane` is on; the ANE takes the rest by 256 rows. A machine
    /// setting, frozen: each row is one engine's, the same seed gives the same bits.
    package var aneFraction = 0.5
    /// Below this height, the GPU alone: a program's fixed cost buys nothing on so few rows. Lowered only
    /// by a check that must put every product on the ANE (`siliconed check … --neural-engine`).
    package var aneMinimumRows = 1024
    /// The GPU's share beside the ANE per weight shape and height (`ane_shares`, `Conductor.Shares`' text),
    /// set by the map from its grid; with it, `aneFraction` only serves the shapes it cannot place.
    package var aneShares = Conductor.Shares.none
    /// The shapes where the AMX also takes part of the GPU's side beside the ANE (`ane_amx`, ρ 1 or 0 per point),
    /// set by the map where the three engines at once beat both pairs (`NeuralEngine.besideAMX`).
    package var aneAMX = Conductor.Shares.none

    /// **The route table these settings make** (`RouteTable`): what the engine consults, product by product and
    /// attention by attention, instead of the switches above. Derived once, after the last override (map,
    /// profile, environment) — a forced setting is a setting, hence a table. The switches stay the input
    /// layer: checks, bench and the environment force them as they always did.
    package private(set) var routes = RouteTable()

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
    /// The date of the capacity map applied, if one was (`load(from:map:)`).
    package private(set) var mapRead: String? = nil
    /// What the rules made of it.
    package private(set) var mapDecisions: Diagnostic.Decisions? = nil
    /// **What the reading refused** — a profile from another machine, a setting a profile is not
    /// allowed to write. The library writes nothing to standard error: the caller displays them
    /// (the CLI at startup, an app wherever it likes).
    package private(set) var warnings: [String] = []

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
    package static var effective: EngineSettings {
        #if DEBUG
        if readsForbidden { forbiddenReads.add(1, ordering: .relaxed) }
        #endif
        return read
    }
    private static let read = EngineSettings()

    #if DEBUG
    /// **A tripwire for what must not read the settings** (debug builds only): inside
    /// `$readsForbidden.withValue(true)`, each read of `effective` is counted. A plan asked before the
    /// capacity map froze the process on the defaults once (the map then refused, the AMX off for the
    /// session); the first read cannot be watched in a test process where others already read, every
    /// read can.
    @TaskLocal package static var readsForbidden = false
    package static let forbiddenReads = Atomic<Int>(0)
    #endif

    /// What an app designated before the first access (`load(from:map:)`): its profile, its map.
    private static let designated = Mutex<(profile: String?, map: Diagnostic.Report?)>((nil, nil))

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

    /// **For an app: read the machine's profile from ITS library, and its capacity map.**
    ///
    /// The settings are a property of the machine, read once for the whole process (`effective`) —
    /// there is no sense in changing them along the way, and the engine reads them deep in its
    /// loops. **Nothing reads them before a render** but a render's memory need: neither building a
    /// `Request`, nor a `Model`, nor listing the catalog — a render, `ModelCard.memoryNeed`,
    /// `effective` and this `load` do. An app therefore calls this at launch, before it shows a need or
    /// renders. **Too late, it throws** (`EngineError.settingsAlreadyLoaded`) instead of silently doing
    /// nothing — unless the profile and the map already read are these. Without this call, the profile
    /// sought is `Library.processWide`'s, which the CLI follows and a sandboxed app would not find.
    ///
    /// `map`: the diagnostic's report this app kept (`Diagnostic.run`), applied by the rules of
    /// `CapacityRules.swift` between the defaults and the profile. **Refused with a warning when it is not
    /// `isCurrent`** (another machine, another macOS, another build): the defaults then apply, and the
    /// app measures again. **It never changes settings already read**: an app without a current map defers this
    /// call until its measure — taken in another process, so that nothing here reads the settings before it.
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
            designated = (path, map)
            return nil
        }
        if let refusal { throw EngineError(refusal) }
        _ = effective   // read now: the warnings (profile from another machine, stale map) are ready
    }

    /// Set on the first read of `effective`: after it, designating a profile no longer serves any purpose.
    private static let alreadyRead = Atomic<Bool>(false)

    private init() {
        EngineSettings.alreadyRead.store(true, ordering: .releasing)
        machine = Machine.detected()
        var refusal: String? = nil

        // ── the capacity map, if an app designated one and it is this machine's, this build's ────────
        if let map = EngineSettings.designated.withLock({ $0.map }) {
            if map.isCurrent {
                let decisions = Diagnostic.decide(map)
                mapRead = map.date
                mapDecisions = decisions
                apply(decisions.settings, .map)
            } else {
                warnings.append("⚠ the capacity map of \(map.date) was measured on \(map.fingerprint), this is "
                                + "\(Diagnostic.Fingerprint.current) — map ignored, the defaults apply.")
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
        read("SILICONED_AMX", "amx")
        read("SILICONED_AMX_SPIN", "amx_spin")
        read("SILICONED_AMX_QUANTUM", "amx_quantum")
        read("SILICONED_AMX_MIN", "amx_min")
        read("SILICONED_AMX_FRACTION", "amx_fraction")
        read("SILICONED_AMX_SHARES", "amx_shares")
        read("SILICONED_REPRODUCIBLE", "coupe_figee")
        read("SILICONED_FLASH", "flash")
        read("SILICONED_HEAD_SPLIT", "head_split")
        read("SILICONED_HEAD_SPLIT_FRACTION", "head_split_fraction")
        read("SILICONED_GEMM_TENSOR", "gemm_tensor")
        read("SILICONED_GEMM_INT8", "gemm_int8")
        read("SILICONED_GEMM_Q8", "gemm_q8")
        read("SILICONED_GEMM_BF16_ACT", "gemm_bf16_act")
        read("SILICONED_BF16_ACT_MIN", "bf16_act_min")
        read("SILICONED_BF16_ACT_8BIT", "bf16_act_8bit")
        read("SILICONED_GEMM_ANE", "gemm_ane")
        read("SILICONED_ANE_FRACTION", "ane_fraction")
        read("SILICONED_ANE_MIN", "ane_min")
        read("SILICONED_ANE_SHARES", "ane_shares")
        read("SILICONED_ANE_AMX", "ane_amx")
        read("SILICONED_SPECTRAL", "spectral")
        read("SILICONED_VAE_BLOCK", "vae_bloc_requetes")
        read("SILICONED_VAE_TILE", "vae_tuile")
        read("SILICONED_VAE_OVERLAP", "vae_recouvrement")
        read("SILICONED_PARALLEL", "parallel")
        read("SILICONED_CACHE", "cache")
        read("SILICONED_MAP_TAIL", "map_tail")
        apply(fromEnv, .environment)

        if let refusal { warnings.insert(refusal, at: 0) }
        routes = RouteTable(settings: self)
    }

    /// **Settings from values alone** — the defaults, then `values` as a map's decisions write them
    /// (`Decisions.settings`): no environment, no profile, no process-wide state. What the pure tests and
    /// `RouteTable(decisions:)` read.
    package init(applying values: [String: Any]) {
        machine = Machine.detected()
        apply(values, .map)
        routes = RouteTable(settings: self)
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
        flag("amx", &amx)
        flag("amx_spin", &amxSpin)
        integer("amx_quantum", &amxQuantum)
        integer("amx_min", &amxMinimumRows)
        real("amx_fraction", &amxFraction)
        if let raw = values["amx_shares"] {
            if let shares = Conductor.Shares(text: "\(raw)") { amxShares = shares; provenance["amx_shares"] = source }
            else { warnings.append("⚠ « amx_shares » unreadable (\(raw)): the scalar share applies.") }
        }
        flag("coupe_figee", &frozenCut)
        flag("flash", &flash)
        flag("head_split", &headSplit)
        real("head_split_fraction", &headSplitFraction)
        headSplitFraction = min(max(headSplitFraction, 0.05), 0.8)
        flag("gemm_tensor", &gemmTensor)
        flag("gemm_int8", &gemmInt8)
        flag("gemm_q8", &gemmQ8)
        flag("gemm_bf16_act", &gemmBF16Activations)
        integer("bf16_act_min", &bf16ActivationMinimumRows)
        flag("bf16_act_8bit", &bf16ActivationEightBit)
        flag("gemm_ane", &gemmANE)
        real("ane_fraction", &aneFraction)
        aneFraction = min(max(aneFraction, 0), 1)
        integer("ane_min", &aneMinimumRows)
        if let raw = values["ane_shares"] {
            if let shares = Conductor.Shares(text: "\(raw)") { aneShares = shares; provenance["ane_shares"] = source }
            else { warnings.append("⚠ « ane_shares » unreadable (\(raw)): the scalar share applies.") }
        }
        if let raw = values["ane_amx"] {
            if let shares = Conductor.Shares(text: "\(raw)") { aneAMX = shares; provenance["ane_amx"] = source }
            else { warnings.append("⚠ « ane_amx » unreadable (\(raw)): the GPU alone beside the ANE.") }
        }
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
            ("amx", "\(amx)"), ("amx_quantum", "\(amxQuantum)"),
            ("amx_min", "\(amxMinimumRows)"), ("amx_spin", "\(amxSpin)"),
            ("amx_fraction", String(format: "%.3f", amxFraction)),
            ("amx_shares", amxShares.isEmpty ? "none" : "\(amxShares.points.count) points"),
            ("coupe_figee", "\(frozenCut)"),
            ("flash", "\(flash)"),
            ("head_split", "\(headSplit)"),
            ("head_split_fraction", String(format: "%.3f", headSplitFraction)),
            ("gemm_tensor", "\(gemmTensor)"),
            ("gemm_int8", "\(gemmInt8)"),
            ("gemm_q8", "\(gemmQ8)"),
            ("gemm_bf16_act", "\(gemmBF16Activations)"), ("bf16_act_min", "\(bf16ActivationMinimumRows)"),
            ("bf16_act_8bit", "\(bf16ActivationEightBit)"),
            ("gemm_ane", "\(gemmANE)"),
            ("ane_fraction", String(format: "%.3f", aneFraction)), ("ane_min", "\(aneMinimumRows)"),
            ("ane_shares", aneShares.isEmpty ? "none" : "\(aneShares.points.count) points"),
            ("ane_amx", aneAMX.isEmpty ? "none" : "\(aneAMX.points.filter { $0.ratio > 0 }.count) of \(aneAMX.points.count) points"),
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
            "amx": amx, "amx_quantum": amxQuantum,
            "amx_min": amxMinimumRows, "amx_fraction": amxFraction, "amx_shares": amxShares.description,
            "coupe_figee": frozenCut,
            "flash": flash, "head_split": headSplit, "head_split_fraction": headSplitFraction,
            "gemm_tensor": gemmTensor, "gemm_int8": gemmInt8, "gemm_q8": gemmQ8,
            "gemm_bf16_act": gemmBF16Activations, "bf16_act_min": bf16ActivationMinimumRows,
            "bf16_act_8bit": bf16ActivationEightBit,
            "gemm_ane": gemmANE, "ane_fraction": aneFraction, "ane_min": aneMinimumRows, "ane_shares": aneShares.description, "ane_amx": aneAMX.description,
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
