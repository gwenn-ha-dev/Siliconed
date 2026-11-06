import Foundation
import Synchronization

/// **The plan: what the engine asks, at every product and every attention, and what it is told — exact rows, no ratio.**
/// Pure: no Metal, no weights, no clock. The engine (`GEMM`, `WeightProducts`, the attention call sites, `MemoryNeed`) holds a
/// `CapacityPlan` and puts one question to it per computation; the answer is an *assignment*: which form of the weight is read,
/// and how many whole rows (or columns) each engine takes.
///
///     engine, at a product              ProductKey(k, n, m, weight format, phase)  ──▶  ProductAssignment
///       GEMM.linear, WeightProducts                                                      path   widened · bf16 · native, the activation rounded or not
///       MemoryNeed (before the first                                                     rows   gpu · amx · ane   (Σ = m; the ANE's in units of 256)
///        computation, same question)                                                     or, beside a native weight, the AMX's last columns
///     engine, at an attention           AttentionKey(S, keys, heads, kv heads, dim, cross)  ──▶  AttentionAssignment
///       Block, Anima, Krea 2, MOSS,                                                      kernel graph · flash · split, and the heads the CPU takes
///        Qwen-Image-2.1, MemoryNeed
///
///     plan = CapacityModel of this Mac + CapacitySolver     resolved when first asked, kept by the exact key
///                                                            (a shape never seen is solved then, by the same solver)
///
/// What sets it apart — what a plan written by analogy with the route table it replaces would get wrong:
///
///   - **The key is the class, exactly.** `(k, n, m, weight format, phase)`: the same product in the encoder and in the
///     denoiser, in bf16 and in int8, are four questions with four answers — no two classes share a ρ, and no answer is read from
///     the nearest point that was measured. An attention is `(S, keys, heads, kv heads, dim, cross)`: a self- and a
///     cross-attention of one S are two keys.
///   - **An answer is rows, not a share.** The engine does not round a ratio to its quantum and clamp it to a band: the solver
///     wrote the integers it will time (the ANE's in units of 256, the GPU's on MPS's 64-row tile, a native weight's GPU columns
///     on 64), and the engine runs those. Alone is an answer: `amx`/`ane` with no GPU rows when the model says it is faster.
///   - **What the engine imposes is hardware or precision, and the assignment says it is respected** (`isRunnable`): the ANE's
///     rows in units of 256 (or the whole product), no ANE on an encoder (P20: only the denoiser's fp16 rows), a bf16 weight read
///     as published goes whole to the GPU (`TensorGEMM`), a native 8-bit weight beside the AMX cuts columns on 64 (the GPU's tiles
///     write whole 64 × 64 blocks).
///   - **A process without the ANE gets a plan computed without it** (`restricted(allowingANE:)`): an encoder, a render on a Mac
///     whose map gives the ANE nothing. There is no fallback route behind a rule that names the ANE; the solver never named it.
///   - **A render with the accelerators off gets a plan computed in fp32** (`restricted(accelerators:)`, the app's « Accelerators »
///     switch): no ANE (fp16, int8), no operand rounded on the GPU's tensor units (`bf16act`, `tiles16`, `tilesbf16`). What stays is
///     exact against the fp32 product on the published weights widened: a bf16 weight read as published beside an fp32 activation
///     (`float × bfloat`, the widening is a shift), an 8-bit weight read as published (`float × int8`, `× s` after; Q8_0's `q · s`
///     in fp32) — one fp32 rounding where the widened weight had one, in another order, as the AMX's share moves the last bits.
///   - **No map, the engine's defaults** (`defaults`): the GPU alone on the widened weight, the attention cut by heads from 1 024
///     rows at the reference share — what every process ran before a plan existed, the same bits.
///   - **A probe or a check forces an answer** (`pin`, `forcing`): one key, or a rule over all of them; the engines the plan
///     brings up (`resources`) are those the forced answers need.
package final class CapacityPlan: Sendable {

    // MARK: - What a plan brings up

    /// **The engines and kernels this process builds**: those some answer of the plan may use. An engine not built cannot be
    /// asked for; one built and never used holds nothing (the conductor's workspace and the ANE's surfaces grow at first use).
    package struct Resources: Equatable, Sendable, Codable {
        package var amx = false
        package var ane = false
        package var bf16Weights = false
        package var int8Weights = false
        package var q8Weights = false
        package var roundedActivations = false
        /// **The ANE's int8 program** may be asked for: the ANE reads a published int8 weight as it is (`dequantize`), beside the
        /// GPU's int8 kernel. It needs both engines — the ANE built, and `TensorGEMM` with its int8 kernel, which reads the same
        /// reserve — so it is never set without `ane` and `int8Weights` (the initializer enforces it).
        package var aneInt8 = false
        /// **The attention may run in half-precision operands** (`tiles16`, `tilesbf16`): the denoiser's attention rounds Q, K, V and
        /// the probabilities to half or bf16.
        package var halfAttention = false
        /// **The decoders compute in bf16** where their weights are published in bf16 (`VAEPrecision`): activations and weights bf16
        /// between two operations, the norms' statistics and every sum in fp32 — the map's convolution faster in bf16
        /// (`CapacityModel.Convolution.prefersHalf`), or a check's forcing. Never under `restricted(accelerators: false)`. Not in
        /// `halfPrecision`: the denoiser's checks are not the decoder's (each decoder stage asks `VAEPrecision` itself).
        package var halfDecoder = false
        /// **A published int8 weight may be multiplied by the activation quantized to int8** (W8A8, `gpu/int8x8`, `QuantizedKernels`): the
        /// quant's author's own forward. Never under `restricted(accelerators: false)`, nor where the W8A8 switch is off
        /// (`CapacityPlan.allowsW8A8`). Not in `halfPrecision`: its checks judge against Comfy's forward, not the reference in bf16.
        package var quantizedActivations = false
        package init(amx: Bool = false, ane: Bool = false, bf16Weights: Bool = false, int8Weights: Bool = false,
                     q8Weights: Bool = false, roundedActivations: Bool = false, aneInt8: Bool = false, halfAttention: Bool = false,
                     quantizedActivations: Bool = false) {
            self.halfAttention = halfAttention
            self.quantizedActivations = quantizedActivations && int8Weights
            self.amx = amx; self.ane = ane; self.bf16Weights = bf16Weights; self.int8Weights = int8Weights
            self.q8Weights = q8Weights; self.roundedActivations = roundedActivations
            self.aneInt8 = aneInt8 && ane && int8Weights
        }
        private enum CodingKeys: String, CodingKey {
            case amx, ane, bf16Weights, int8Weights, q8Weights, roundedActivations, aneInt8, halfAttention, halfDecoder, quantizedActivations
        }
        package init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            func flag(_ k: CodingKeys) throws -> Bool { try c.decodeIfPresent(Bool.self, forKey: k) ?? false }
            amx = try flag(.amx); ane = try flag(.ane); bf16Weights = try flag(.bf16Weights); int8Weights = try flag(.int8Weights)
            q8Weights = try flag(.q8Weights); roundedActivations = try flag(.roundedActivations); aneInt8 = try flag(.aneInt8)
            halfAttention = try flag(.halfAttention); halfDecoder = try flag(.halfDecoder)
            quantizedActivations = try flag(.quantizedActivations)
        }
        /// `TensorGEMM` is built when any native form is.
        package var usesTensor: Bool { bf16Weights || int8Weights || q8Weights || roundedActivations || quantizedActivations }
        /// Some product may compute in half precision (the ANE's fp16, the activation rounded to bf16): the checks then
        /// judge against the reference run in bf16 (`HalfPrecisionReference`). False under `restricted(accelerators: false)`.
        package var halfPrecision: Bool { ane || roundedActivations || halfAttention }
    }

    /// A rule that forces answers: given a key, an assignment, or `nil` to let the plan answer.
    package typealias Override = @Sendable (ProductKey) -> ProductAssignment?
    /// The same for an attention (the key, and whether the caller tries the flash kernel first).
    package typealias AttentionOverride = @Sendable (AttentionKey, Bool) -> AttentionAssignment?

    package let resources: Resources
    /// A line naming the plan for the render cache's key and the report: `defaults`, a model's fingerprint, a forcing's label.
    package let identity: String
    package let model: CapacityModel?
    package let allowsANE: Bool
    /// **The GPU may compute on operands rounded below fp32**: the activation rounded to bf16 (`bf16act`), the attention on fp16 or
    /// bf16 operands (`tiles16`, `tilesbf16`) — the GPU's tensor units in half precision. False: fp32 throughout on the GPU
    /// (`restricted(accelerators: false)`).
    package let allowsHalfPrecision: Bool
    /// **The W8A8 switch** (`allowsW8A8`, the activation quantized to int8 beside a published int8 weight, `gpu/int8x8`): on, the map may
    /// choose it wherever it is faster; off, the integers are read as before it (`gpu/int8act`, `gpu/int8`), the same bits as then.
    package let allowsQuantizing: Bool
    /// **The process's W8A8 switch**: on, unless `SILICONED_W8A8=0` (a judge of the route before it, on a map that has its line).
    package static let allowsW8A8 = ProcessInfo.processInfo.environment["SILICONED_W8A8"] != "0"
    package var usesAMX: Bool { resources.amx }
    package var usesANE: Bool { resources.ane }
    /// **The height under which no product has a row on the AMX or the ANE**: the model's domain (`CapacitySolver.Limits`) for a
    /// plan solved from a model; 1 for a plan that forces or pins (any height may carry rows). What the memory need's sweep over
    /// the heights a prompt may give (`MemoryNeed.gemm`) starts from.
    package var engineFloor: Int {
        guard model != nil else { return 1 }
        return min(resources.amx ? limits.amxRows : .max, resources.ane ? limits.aneRows : .max)
    }

    /// **What the solver was given to answer for this plan** — the same limits, sustained rate, refused kernels, ANE and rounding
    /// permissions, fp16 derived from both (`Constraints.allowsHalf(ane:rounding:)`, at every version): what a render's prediction
    /// times the plan's answers with too (`RenderLoop.predictions`).
    package var constraints: CapacitySolver.Constraints {
        CapacitySolver.Constraints(limits: limits, sustained: sustained, excluded: excluded, allowsANE: allowsANE,
                                   allowsRounding: allowsHalfPrecision, allowsQuantizing: allowsHalfPrecision && allowsQuantizing)
    }

    /// fp16 operands may be asked for under this plan: the one rule (`Constraints.allowsHalf(ane:rounding:)`).
    private var allowsHalf: Bool { CapacitySolver.Constraints.allowsHalf(ane: allowsANE, rounding: allowsHalfPrecision) }

    /// **The half precisions the GPU's tensor units may compute in under this plan**, in the order a sentence names them: `bf16`
    /// (an activation rounded, the attention's bf16 operands), `fp16` (the attention's fp16 operands, `allowsHalf`). Empty:
    /// the GPU computes in fp32. What the app says a render may use (`AcceleratorOffer`); the solver still chooses class by class.
    package var gpuHalfPrecisions: [String] {
        guard allowsHalfPrecision else { return [] }
        func has(_ f: CapacityModel.Form) -> Bool {
            let k = CapacityModel.Kernel(.gpu, f)
            return (model?.lines[k]?.rate ?? 0) > 0 && !excluded.contains(k)
        }
        guard model != nil else { return (resources.roundedActivations ? ["bf16"] : []) + (resources.quantizedActivations ? ["int8"] : []) }
        var out: [String] = []
        if resources.roundedActivations || has(.tilesbf16) { out.append("bf16") }
        if allowsHalf && has(.tiles16) { out.append("fp16") }
        if resources.quantizedActivations { out.append("int8") }
        return out
    }

    private let limits: CapacitySolver.Limits
    private let sustained: Bool
    private let excluded: Set<CapacityModel.Kernel>
    private let override: Override?
    private let attentionOverride: AttentionOverride?

    private struct State {
        var products: [ProductKey: ProductAssignment] = [:]
        var attentions: [Memo: AttentionAssignment] = [:]
        var pins: [ProductKey: ProductAssignment] = [:]
        /// The products the solver had no answer for, and why (`ProductLedger.Reason`): the GPU alone, named rather than silent.
        var unsolved: [ProductKey: ProductLedger.Reason] = [:]
        /// The K-quant classes' answers without their native route (`assignWidened`).
        var widened: [ProductKey: ProductAssignment] = [:]
    }
    private struct Memo: Hashable { var key: AttentionKey; var flash: Bool }
    private let state = Mutex(State())
    /// The plan of a process without the ANE, made once: its memo is as valuable as this one's.
    private let withoutANE = Mutex<CapacityPlan?>(nil)
    /// The plan of a render with the accelerators off (fp32 throughout), made once.
    private let inFP32 = Mutex<CapacityPlan?>(nil)

    // MARK: - Making a plan

    /// **The engine's defaults**: the GPU alone, the widened weight, the head split from 1 024 rows at the reference share.
    package static let defaults = CapacityPlan()

    private init() {
        resources = Resources(); identity = "default"; model = nil; allowsANE = false; allowsHalfPrecision = true; allowsQuantizing = true
        limits = CapacitySolver.Limits(); sustained = true; excluded = []; override = nil; attentionOverride = nil
    }

    /// **The plan of a Mac whose model was measured**: the solver answers from it. The engines it brings up are those the model
    /// has a line for; the ANE only if `allowsANE`, the rounded operands only if `allowsHalfPrecision`.
    package init(model: CapacityModel, limits: CapacitySolver.Limits = .init(), sustained: Bool = true,
                 excluded: Set<CapacityModel.Kernel> = [], allowsANE: Bool = true, allowsHalfPrecision: Bool = true,
                 allowsQuantizing: Bool = CapacityPlan.allowsW8A8) {
        func has(_ e: CapacityModel.Engine, _ f: CapacityModel.Form) -> Bool {
            let k = CapacityModel.Kernel(e, f)
            return (model.lines[k]?.rate ?? 0) > 0 && !excluded.contains(k)
        }
        self.model = model; self.limits = limits; self.sustained = sustained; self.excluded = excluded
        self.allowsANE = allowsANE; self.allowsHalfPrecision = allowsHalfPrecision; self.allowsQuantizing = allowsQuantizing
        // An 8-bit weight is read as published by the GPU's kernel on the fp32 activation or, where operands may be rounded, on the
        // activation rounded to bf16 (`int8act`, `q8act`): either line brings the integers' kernels up.
        // A K-quant read as published (`q4kact`) rounds its activation too: its line brings the rounding's library up, the K-quants' with it.
        let rounds = allowsHalfPrecision && (has(.gpu, .bf16act) || has(.gpu, .int8act) || has(.gpu, .q8act) || has(.gpu, .q4kact))
        var resources = Resources(amx: has(.amx, .fp32), ane: allowsANE && has(.ane, .fp16),
                              bf16Weights: has(.gpu, .bf16), int8Weights: has(.gpu, .int8) || (allowsHalfPrecision && has(.gpu, .int8act)),
                              q8Weights: has(.gpu, .q8) || (allowsHalfPrecision && has(.gpu, .q8act)),
                              roundedActivations: rounds,
                              aneInt8: allowsANE && has(.ane, .fp16) && has(.ane, .int8),
                              halfAttention: (CapacitySolver.Constraints.allowsHalf(ane: allowsANE, rounding: allowsHalfPrecision) && has(.gpu, .tiles16))
                                  || (allowsHalfPrecision && has(.gpu, .tilesbf16)))
        resources.halfDecoder = allowsHalfPrecision && (model.convolution?.prefersHalf ?? false)
        // W8A8 (`gpu/int8x8`): its own library, where the activation may be quantized and the map measured the kernel.
        resources.quantizedActivations = allowsHalfPrecision && allowsQuantizing && resources.int8Weights && has(.gpu, .int8x8)
        self.resources = resources
        override = nil; attentionOverride = nil
        // The switch off names itself only where it changes something: a map without the W8A8 line keeps its identity either way.
        identity = "model " + Self.fingerprint(model) + (allowsANE ? "" : " no-ane") + (allowsHalfPrecision ? "" : " fp32")
            + (!allowsQuantizing && has(.gpu, .int8x8) ? " no-w8a8" : "")
    }

    /// **A plan that forces its answers**: a check's strictest case, a probe's route. `resources` are the engines it brings up;
    /// the key it does not speak of gets the engine's defaults.
    package init(resources: Resources, label: String, allowsANE: Bool = true, allowsHalfPrecision: Bool = true,
                 override: Override? = nil, attentionOverride: AttentionOverride? = nil) {
        var resources = resources
        if !allowsHalfPrecision {
            resources.roundedActivations = false; resources.halfAttention = false; resources.halfDecoder = false
            resources.quantizedActivations = false
        }
        self.resources = resources; identity = label; model = nil; self.allowsANE = allowsANE && resources.ane
        self.allowsHalfPrecision = allowsHalfPrecision; allowsQuantizing = resources.quantizedActivations
        limits = CapacitySolver.Limits(); sustained = true; excluded = []; self.override = override
        self.attentionOverride = attentionOverride
    }

    /// **A probe's plan** (a probe's solos, a bench's solos): the engines it builds, every product on the GPU alone
    /// until a key is `pin`ned to the exact rows to measure.
    package static func probing(amx: Bool = false, ane: Bool = false, bf16Weights: Bool = false, int8Weights: Bool = false,
                                q8Weights: Bool = false, roundedActivations: Bool = false, quantizedActivations: Bool = false) -> CapacityPlan {
        CapacityPlan(resources: Resources(amx: amx, ane: ane, bf16Weights: bf16Weights, int8Weights: int8Weights,
                                          q8Weights: q8Weights, roundedActivations: roundedActivations,
                                          quantizedActivations: quantizedActivations), label: "probing")
    }

    /// The same model and rules for a process that may not bring up the ANE (`allowingANE: false`): its own answers, its own memo.
    package func restricted(allowingANE allowed: Bool) -> CapacityPlan {
        if allowed || !allowsANE { return self }
        return withoutANE.withLock { kept in
            if let kept { return kept }
            var r = resources; r.ane = false; r.aneInt8 = false
            let made: CapacityPlan
            if let model {
                made = CapacityPlan(model: model, limits: limits, sustained: sustained, excluded: excluded, allowsANE: false,
                                    allowsHalfPrecision: allowsHalfPrecision, allowsQuantizing: allowsQuantizing)
            } else {
                made = CapacityPlan(resources: r, label: identity + " no-ane", allowsANE: false,
                                    allowsHalfPrecision: allowsHalfPrecision, override: override, attentionOverride: attentionOverride)
            }
            kept = made
            return made
        }
    }

    /// **The same model and rules with the accelerators off** (`accelerators: false`, the app's switch): fp32 throughout — no ANE,
    /// no operand rounded on the GPU (`allowsHalfPrecision` false). Its own answers, its own memo, its own identity (`… fp32`):
    /// the render cache never mistakes one for the other. `true` is this plan.
    package func restricted(accelerators on: Bool) -> CapacityPlan {
        if on || (!allowsANE && !allowsHalfPrecision) { return self }
        return inFP32.withLock { kept in
            if let kept { return kept }
            var r = resources; r.ane = false; r.aneInt8 = false
            let made: CapacityPlan
            if let model {
                made = CapacityPlan(model: model, limits: limits, sustained: sustained, excluded: excluded, allowsANE: false,
                                    allowsHalfPrecision: false, allowsQuantizing: allowsQuantizing)
            } else {
                made = CapacityPlan(resources: r, label: identity + (allowsANE ? " no-ane" : "") + " fp32", allowsANE: false,
                                    allowsHalfPrecision: false, override: override, attentionOverride: attentionOverride)
            }
            kept = made
            return made
        }
    }

    /// **The plan of a render's denoiser** under its resolved switch (`Accelerators`): this plan restricted to fp32 when they are
    /// off, without the ANE when the render does not bring it up — the one `GEMM`, `MemoryNeed` and the render cache all ask.
    package func forDenoiser(_ accelerators: Accelerators) -> CapacityPlan {
        switch accelerators {
        case .off: return restricted(accelerators: false)
        case .on(let ane): return restricted(allowingANE: ane)
        }
    }

    /// A hash of every number of the model, in a fixed order (a dictionary keyed by a kernel or an engine encodes in no order).
    private static func fingerprint(_ model: CapacityModel) -> String {
        var parts: [String] = []
        for (k, l) in model.lines.sorted(by: { $0.key.description < $1.key.description }) { parts.append("\(k) \(l.t0) \(l.rate)" + (l.perWeight.map { " w\($0)" } ?? "") + (l.perRow.map { " h\($0)" } ?? "") + (l.rateByK?.map { " \($0.k):\($0.rate)" }.joined() ?? "") + (l.rowsFrom.map { " r\($0)" } ?? "")) }
        for (k, l) in model.thin.sorted(by: { $0.key.description < $1.key.description }) { parts.append("thin \(k) \(l.t0) \(l.rate) \(l.perWeight ?? 0) \(l.perRow ?? 0) \(l.rowsFrom ?? 0)" + (l.rateByK?.map { " \($0.k):\($0.rate)" }.joined() ?? "")) }
        parts.append("unit \(model.aneRowUnit)")
        for (k, s) in model.sharing.sorted(by: { $0.key < $1.key }) {
            parts.append("\(k) \(s.overhead) \(s.sustained.map { "\($0)" } ?? "-") \(s.kFrom ?? 0)-\(s.kTo ?? 0) " + (s.keepByK?.map { "\($0.k):" + $0.keep.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ",") }.joined(separator: ";") ?? "") + " " + s.keep.sorted { $0.key < $1.key }.map { "\($0.key.rawValue)=\($0.value)" }.joined(separator: ",")
                         + (s.perWeight.map { " lift " + $0.sorted { $0.key < $1.key }.map { "\($0.key.rawValue)=\($0.value)" }.joined(separator: ",") } ?? "")
                         + (s.noise.map { " noise \($0)" } ?? ""))
        }
        for (e, h) in model.heat.sorted(by: { $0.key < $1.key }) { parts.append("heat \(e.rawValue) \(h)") }
        for (k, t) in model.tiles.sorted(by: { $0.key.description < $1.key.description }) { parts.append("tile \(k) \(t)") }
        for (k, t) in model.thinTiles.sorted(by: { $0.key.description < $1.key.description }) { parts.append("thin tile \(k) \(t)") }
        for (k, t) in model.attentionTiles.sorted(by: { $0.key.description < $1.key.description }) { parts.append("attention tile \(k) \(t)") }
        for (e, c) in model.crest.sorted(by: { $0.key < $1.key }) { parts.append("crest \(e.rawValue) \(c)") }
        parts.append("memory \(model.memoryBandwidth.map { "\($0)" } ?? "-") disk \(model.diskBandwidth.map { "\($0)" } ?? "-")")
        for (c, r) in model.diskClasses.sorted(by: { $0.key < $1.key }) { parts.append("diskclass \(c) \(r.rate) \(r.mean)") }
        for (kind, paths) in model.preparation.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            parts.append("prep \(kind.rawValue) " + paths.sorted { $0.key.rawValue < $1.key.rawValue }.map { "\($0.key.rawValue)=\($0.value)" }.joined(separator: ","))
        }
        parts.append("kquant \(model.kquantPreparation.map { "\($0)" } ?? "-")")
        if let staging = model.kquantStaging { parts.append("kquant copy \(staging)") }
        parts.append("noise \(model.noise.map { "\($0)" } ?? "-")")
        // Only where the map has it: a map before the probe keeps its fingerprint (its plan is the same).
        if let n = model.normRope { parts.append("normrope \(n.cpu.t0) \(n.cpu.rate) \(n.gpu.t0) \(n.gpu.rate) \(n.gpuWaited.t0) \(n.gpuWaited.rate)") }
        if let c = model.convolution { parts.append("conv \(c.fp32.t0) \(c.fp32.rate) \(c.bf16.t0) \(c.bf16.rate) \(c.work)") }
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in parts.joined(separator: "\n").utf8 { hash = (hash ^ UInt64(byte)) &* 0x100000001b3 }
        return String(hash, radix: 16)
    }

    // MARK: - Asking

    /// **The assignment of a product.** A pinned key first, then the forcing rule, then the model's solver; without a model, the
    /// GPU alone. Folded onto the GPU when this plan has no ANE (a forcing that names it is told to do without).
    package func assign(_ key: ProductKey) -> ProductAssignment {
        let known: ProductAssignment? = state.withLock { $0.pins[key] ?? $0.products[key] }
        if let known { return known }
        var a: ProductAssignment
        if let forced = override?(key) { a = forced } else {
            let (solved, reason) = CapacityPlan.solved(self, key)
            a = solved
            if let reason { state.withLock { $0.unsolved[key] = reason } }
        }
        // A forcing that names the ANE is told to do without: the rows to the GPU, and a weight the ANE would have read as published
        // is the widened one again (the GPU's int8 kernel rounds in another order — not the route a process without the ANE ran).
        if a.ane > 0 && !allowsANE {
            if a.path == .native { a = .gpuAlone(key.m) } else { a.gpu += a.ane; a.ane = 0 }
        }
        // A forcing that rounds the activation, in a plan computed in fp32: the same weight read as published, the operand fp32.
        if a.roundsActivation && !allowsHalfPrecision { a.roundsActivation = false }
        // ... or quantizes it where this plan may not: the same weight read as published, on the fp32 operand.
        if a.quantizesActivation && !(allowsHalfPrecision && resources.quantizedActivations) { a.quantizesActivation = false }
        let kept = a
        state.withLock { $0.products[key] = kept }
        return kept
    }

    /// **The assignment of a K-quant weight the native route cannot read** (a Q6_K, a legacy type, a reserve too small for its layout)
    /// in a class the plan gave that route (`gpu/q4kact`): what the plan answers without it — the solver with that kernel excluded, so
    /// the widened route of a plan before it, at the bit; under a forcing or without a model, the GPU alone. Any other answer is `assign`'s.
    /// The classes do not know which K-quant each module publishes (a Q4_K_M mixes Q4_K, Q5_K and Q6_K in one shape): the engine asks
    /// here per tensor, and the render's record names the fallback (`ProductLedger.Reason.kquantUnread`).
    package func assignWidened(_ key: ProductKey) -> ProductAssignment {
        let planned = assign(key)
        guard planned.path == .native, key.weight == .q4k else { return planned }
        if let known = state.withLock({ $0.widened[key] }) { return known }
        var a = ProductAssignment.gpuAlone(key.m)
        if override == nil, let model {
            var c = constraints
            c.excluded.insert(CapacityModel.Kernel(.gpu, .q4kact))
            if let solved = CapacitySolver.product(key, model: model, constraints: c)?.assignment { a = solved }
        }
        if a.ane > 0 && !allowsANE { a.gpu += a.ane; a.ane = 0 }
        if a.roundsActivation && !allowsHalfPrecision { a.roundsActivation = false }
        let kept = a
        state.withLock { $0.widened[key] = kept }
        return kept
    }

    /// The solver's answer, or the GPU alone and the reason it is the GPU alone.
    private static func solved(_ plan: CapacityPlan, _ key: ProductKey) -> (ProductAssignment, ProductLedger.Reason?) {
        guard let model = plan.model else { return (.gpuAlone(key.m), .noModel) }
        guard let a = CapacitySolver.product(key, model: model, constraints: plan.constraints)?.assignment else { return (.gpuAlone(key.m), .unsolved) }
        return (a, nil)
    }

    /// **Why `key` is on the GPU alone**, when the solver gave it no answer (`assign` asked first); `nil` when it was solved or forced.
    package func fallback(_ key: ProductKey) -> ProductLedger.Reason? {
        state.withLock { $0.unsolved[key] }
    }

    /// **The assignment of an attention.** `flash`: the caller's code tries the flash kernel before the split (Z-Image's and
    /// Anima's self-attentions); a caller that does not implement it never gets it.
    package func assign(_ key: AttentionKey, flash: Bool = false) -> AttentionAssignment {
        let memo = Memo(key: key, flash: flash)
        if let known = state.withLock({ $0.attentions[memo] }) { return known }
        var a: AttentionAssignment
        if let forced = attentionOverride?(key, flash) {
            a = forced
            // A forcing on half-precision operands, in a plan computed in fp32: the same tiles in fp32 (three stages).
            if !allowsHalfPrecision {
                if a.kernel == .tiles16 || a.kernel == .tilesbf16 { a.kernel = .tiles; a.tile = nil }
                if a.splits && (a.gpuKernel == .tiles16 || a.gpuKernel == .tilesbf16) { a.gpuKernel = .tiles; a.tile = nil }
            }
        } else if let model {
            a = CapacitySolver.attention(key, flash: flash, model: model, constraints: constraints)?.assignment ?? .defaults(key)
        } else {
            a = .defaults(key)
        }
        state.withLock { $0.attentions[memo] = a }
        return a
    }

    /// **Force an answer** (a probe measuring one route at exact rows): the key then answers `assignment`, whatever the plan
    /// would say, until `unpin`. The assignment must be one the engines can run for the key.
    package func pin(_ key: ProductKey, _ assignment: ProductAssignment) {
        precondition(assignment.isRunnable(for: key), "pinned assignment \(assignment) cannot run \(key)")
        state.withLock { $0.pins[key] = assignment }
    }
    package func unpin(_ key: ProductKey) { _ = state.withLock { $0.pins.removeValue(forKey: key) } }
    package func unpinAll() { state.withLock { $0.pins.removeAll() } }

    /// The products this plan was asked about so far, and what it answered: what a report shows of a render.
    package var answered: [(ProductKey, ProductAssignment)] {
        state.withLock { $0.products.map { ($0.key, $0.value) } }.sorted { ($0.0.phase.rawValue, $0.0.m, $0.0.k, $0.0.n) < ($1.0.phase.rawValue, $1.0.m, $1.0.k, $1.0.n) }
    }
}

// MARK: - A render's accelerators

/// **What a render's denoiser may compute on beyond fp32** — the app's « Accelerators » switch (`Request.accelerators`), resolved
/// against this Mac's plan once, at admission (`Admission`). The encoders take none of it in either case.
///
///     on(neuralEngine: true)    the plan as measured: the ANE's rows (fp16; int8 as published) and the GPU's tensor units on
///                               rounded operands (`bf16act`, `tiles16`, `tilesbf16`), where the map chooses them
///     on(neuralEngine: false)   the same without the ANE: this Mac's map gives it nothing (`usesANE` false)
///     off                       fp32 throughout (`CapacityPlan.restricted(accelerators: false)`), the image the fp32 checks compare
package enum Accelerators: Sendable, Hashable {
    case on(neuralEngine: Bool)
    case off

    /// The switch as a request carries it, resolved: `on` brings the ANE up where the plan uses it.
    package init(requested: Bool, usesANE: Bool) { self = requested ? .on(neuralEngine: usesANE) : .off }
    /// What a render not yet chosen may do: everything the process's plan offers.
    package static var process: Accelerators { .on(neuralEngine: EngineSettings.effective.plan.usesANE) }
    /// The denoiser's products may take the Neural Engine.
    package var neuralEngine: Bool { if case .on(true) = self { true } else { false } }
    package var isOn: Bool { self != .off }
}

/// **What the accelerators do on this Mac, for a render's denoiser** — what the app's sentences say (« the Neural Engine (fp16)
/// and the GPU's tensor units (bf16) »): read from the plan the render runs under (`CapacityPlan.forDenoiser`), never from the chip's
/// name. `none`: fp32 throughout — the switch is off, or this Mac's map chooses no accelerator.
package struct AcceleratorOffer: Sendable, Hashable, Codable {
    /// The Neural Engine takes rows of the denoiser's products (fp16).
    package var neuralEngine: Bool
    /// The half precisions the GPU's tensor units may compute in (`CapacityPlan.gpuHalfPrecisions`): `["bf16"]`, `["bf16", "fp16"]`…
    package var tensorUnits: [String]
    package init(neuralEngine: Bool = false, tensorUnits: [String] = []) {
        self.neuralEngine = neuralEngine; self.tensorUnits = tensorUnits
    }
    package static let none = AcceleratorOffer()
    /// What `plan` (a render's denoiser's) may use.
    package init(plan: CapacityPlan) { self.init(neuralEngine: plan.usesANE, tensorUnits: plan.gpuHalfPrecisions) }
    /// fp32 throughout.
    package var isNone: Bool { !neuralEngine && tensorUnits.isEmpty }
    /// The tensor units' precisions as a sentence writes them: `bf16`, `bf16, fp16`.
    package var tensorUnitsText: String { tensorUnits.joined(separator: ", ") }
}

// MARK: - The weight's form

/// What a weight is, for the plan: the form it is published in. (A K-quant is none of these: `CapacityClasses.WeightFormat.q4k`
/// is widened, or read as published on a rounded activation alone — `gpu/q4kact` —, its preparation `CapacityModel.kquant…`.)
package enum WeightKind: String, Codable, Sendable, CaseIterable { case bf16, int8, q8 }

/// How a product reads its weight.
package enum WeightPath: String, Codable, Sendable {
    /// Widened to fp32 on the CPU (the default path: MPS, the AMX, the ANE read it).
    case widened
    /// The bf16 weight as published (`TensorGEMM`).
    case bf16
    /// The 8-bit weight as published (int8 convrot, Q8_0), or a K-quant's blocks (Q4_K, Q5_K: `KQuantKernels`).
    case native
}

// MARK: - A product

/// **A product's class, exactly**: its shape and height, the form its weight is published in, and the phase it belongs to.
package struct ProductKey: Hashable, Sendable, Codable, CustomStringConvertible {
    package var k: Int
    package var n: Int
    package var m: Int
    package var weight: CapacityClasses.WeightFormat
    package var phase: CapacityClasses.Phase
    package init(k: Int, n: Int, m: Int, weight: CapacityClasses.WeightFormat = .bf16, phase: CapacityClasses.Phase = .denoising) {
        self.k = k; self.n = n; self.m = m; self.weight = weight; self.phase = phase
    }
    package var description: String { "\(phase.rawValue) \(k)x\(n)@\(m)~\(weight.rawValue)" }
}

/// **What the engines do with a product**: the weight's form, and the whole rows each takes. The first `gpu` rows are the GPU's,
/// then `amx`, then the last `ane` (the order the engine lays them in the output).
package struct ProductAssignment: Equatable, Sendable, Codable, CustomStringConvertible {
    package var path: WeightPath
    /// The activation is rounded to bf16 before the product (`gemm_bf16_act`): beside a bf16 weight read as published (`gpu/bf16act`),
    /// or the GPU's rows beside an 8-bit one (`gpu/int8act`, `gpu/q8act`).
    package var roundsActivation: Bool
    /// **The activation is quantized to int8 for the product** (W8A8, `gpu/int8x8`): beside a published int8 weight read as published,
    /// the GPU's whole product — ComfyUI's `int8_linear`. Never with `roundsActivation`.
    package var quantizesActivation: Bool
    package var gpu: Int
    package var amx: Int
    package var ane: Int
    /// Beside a native 8-bit weight the AMX takes its last columns, for all the rows (the GPU the others, `gpu` = m): the
    /// columns it takes, 0 for none. Its widening is done for them alone.
    package var amxColumns: Int

    package init(path: WeightPath = .widened, roundsActivation: Bool = false, quantizesActivation: Bool = false, gpu: Int, amx: Int = 0,
                 ane: Int = 0, amxColumns: Int = 0) {
        self.path = path; self.roundsActivation = roundsActivation; self.quantizesActivation = quantizesActivation
        self.gpu = gpu; self.amx = amx; self.ane = ane
        self.amxColumns = amxColumns
    }

    private enum CodingKeys: String, CodingKey { case path, roundsActivation, quantizesActivation, gpu, amx, ane, amxColumns }
    /// An assignment written before W8A8 reads as one that does not quantize.
    package init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        path = try c.decode(WeightPath.self, forKey: .path)
        roundsActivation = try c.decode(Bool.self, forKey: .roundsActivation)
        quantizesActivation = try c.decodeIfPresent(Bool.self, forKey: .quantizesActivation) ?? false
        gpu = try c.decode(Int.self, forKey: .gpu); amx = try c.decode(Int.self, forKey: .amx); ane = try c.decode(Int.self, forKey: .ane)
        amxColumns = try c.decode(Int.self, forKey: .amxColumns)
    }

    /// The default: the whole product on the GPU, the weight widened.
    package static func gpuAlone(_ m: Int, path: WeightPath = .widened, rounds: Bool = false) -> ProductAssignment {
        ProductAssignment(path: path, roundsActivation: rounds, gpu: m)
    }

    package var rows: Int { gpu + amx + ane }
    package var usesAMX: Bool { amx > 0 || amxColumns > 0 }
    package var usesANE: Bool { ane > 0 }
    package var gpuRows: Int { gpu }

    /// The engines, in the ledger's order (`gpu+ane+amx`): `gpu`, `gpu+amx`, `gpu+ane`, `gpu+ane+amx`; `amx`, `ane`, `ane+amx` alone.
    package var engines: String {
        var parts: [String] = []
        if gpu > 0 { parts.append("gpu") }
        if ane > 0 { parts.append("ane") }
        if usesAMX { parts.append("amx") }
        return parts.joined(separator: "+")
    }

    /// The route as the render's record (`ProductLedger`) names it: the engines for a widened weight, `gpu/bf16`, `gpu/bf16act`,
    /// `gpu+ane/bf16act`, `gpu/int8`, `gpu/int8act`, `gpu+ane/int8act`, `gpu+amx/q8`… for a weight read as published (`act`: the GPU's
    /// rows on the rounded activation).
    package func route(weight: CapacityClasses.WeightFormat) -> String {
        switch path {
        case .widened: return engines
        case .bf16: return roundsActivation ? (ane > 0 ? "gpu+ane/bf16act" : "gpu/bf16act") : "gpu/bf16"
        case .native:
            // The ANE beside the GPU's int8 kernel reads the integers (`ane > 0`): `gpu+ane/int8`, `ane/int8` alone.
            let engines = amxColumns > 0 ? "gpu+amx" : ane > 0 ? (gpu > 0 ? "gpu+ane" : "ane") : "gpu"
            if quantizesActivation { return engines + "/int8x8" }
            return engines + "/" + (weight == .int8 ? "int8" : weight == .q4k ? "q4k" : "q8") + (roundsActivation ? "act" : "")
        }
    }

    /// **Whether the engines can run it for `key`** — the hardware and precision limits, nothing else: no negative rows, rows
    /// that add up to the height; the ANE's in units of 256 or the whole product, and never on an encoder; a bf16 weight read
    /// as published whole on the GPU — or, its GPU rows on the rounded activation, with the ANE's last rows beside them (its fp16
    /// program reads the bf16 copy, lifted by powers of two as the widened weight; no AMX, which would need the weight widened); a native 8-bit weight with its AMX columns beside a GPU side of whole 64-column tiles (the
    /// kernel writes whole tiles, an edge inside one would overwrite the AMX's first columns) — or, if it is int8 with a scale per
    /// column or for the tensor, with the ANE's last rows beside the GPU's (the program reads the integers: `readsInt8`, no AMX). A
    /// rounded activation is the GPU's: it has rows, and no AMX columns beside it (the AMX reads the fp32 activation — one rounding for
    /// some columns only); the ANE's rows beside it read the fp32 activation as always (its program lifts each row into fp16's range).
    package func isRunnable(for key: ProductKey) -> Bool {
        guard gpu >= 0, amx >= 0, ane >= 0, amxColumns >= 0, rows == key.m, key.m > 0 else { return false }
        if ane > 0 {
            guard ane % NeuralEngine.rowUnit == 0 || ane == key.m, key.phase == .denoising else { return false }
        }
        if roundsActivation && (path == .widened || key.phase == .encoding) { return false }
        // W8A8: a published int8 weight (a scale per column or for the tensor — `Int8Plan` refuses Q8_0's blocks — and a rotation group
        // the kernel takes), the denoiser's, the GPU's whole product: no AMX (it reads the fp32 activation), no ANE (its int8 × int8 is
        // not built), never with the rounding.
        if quantizesActivation {
            return path == .native && key.weight == .int8 && key.phase == .denoising && !roundsActivation && gpu == key.m && amx == 0
                && ane == 0 && amxColumns == 0 && W8A8.stages(group: key.weight.rotationGroup) != nil
        }
        switch path {
        case .widened:
            return amxColumns == 0
        case .bf16:
            guard amx == 0, amxColumns == 0, key.weight == .bf16 else { return false }
            return ane == 0 || (roundsActivation && gpu > 0)
        case .native:
            // A K-quant's blocks: the GPU's rounded kernel, whole — no fp32-activation kernel (it would compute `float × half` at the
            // shader cores' rate), no ANE (its programs read a widened weight or per-column integers, never K-quant blocks), no AMX.
            if key.weight == .q4k { return roundsActivation && gpu == key.m && ane == 0 && amxColumns == 0 && amx == 0 }
            guard amx == 0, key.weight == .int8 || key.weight == .q8 else { return false }
            if roundsActivation && (gpu == 0 || amxColumns > 0) { return false }
            // The ANE beside the GPU's int8 kernel, both reading the integers: the program takes a scale per column or for the tensor
            // (never Q8_0's blocks), no AMX columns, and a weight whose rotation it can cut K for.
            if ane > 0 {
                return key.weight == .int8 && amxColumns == 0
                    && NeuralEngine.readsInt8(k: key.k, n: key.n, group: key.weight.rotationGroup)
            }
            if amxColumns == 0 { return true }
            let gpuColumns = key.n - amxColumns
            return key.n >= 128 && gpuColumns >= 64 && gpuColumns % 64 == 0
        }
    }

    package var description: String {
        "\(path.rawValue)\(roundsActivation ? "◐" : "")\(quantizesActivation ? "◑" : "") gpu \(gpu) amx \(amx)\(amxColumns > 0 ? " (+\(amxColumns) columns)" : "") ane \(ane)"
    }
}

// MARK: - An attention

/// **An attention's class, exactly**: the query rows, the key rows, the heads and their k/v heads, the head dimension, and whether
/// it reads another sequence's keys.
package struct AttentionKey: Hashable, Sendable, Codable, CustomStringConvertible {
    package var sequence: Int
    package var keys: Int
    package var heads: Int
    package var kvHeads: Int
    package var headDim: Int
    package var cross: Bool
    package init(sequence: Int, keys: Int? = nil, heads: Int, kvHeads: Int? = nil, headDim: Int, cross: Bool = false) {
        self.sequence = sequence; self.keys = keys ?? sequence; self.heads = heads; self.kvHeads = kvHeads ?? heads
        self.headDim = headDim; self.cross = cross
    }
    /// The heads that read one k/v head: the CPU takes whole groups.
    package var group: Int { max(1, heads / max(1, kvHeads)) }
    package var description: String { "attention \(sequence)/\(keys) \(heads)/\(kvHeads)x\(headDim)\(cross ? " cross" : "")" }
}

/// **What runs an attention**: the kernel, and for the split the heads the CPU takes (whole kv groups).
package struct AttentionAssignment: Equatable, Sendable, Codable, CustomStringConvertible {
    /// `tiles`: the three stages by tiles of query rows, fp32 throughout (`TiledAttention`); `tiles16`: fp16 operands and fp32 sums —
    /// `Constraints.allowsHalf`, at every version; `tilesbf16`: bf16 operands and fp32 sums — the rounding a request may allow
    /// (`Constraints.allowsRounding`, at every version too). The last two run three stages or fused, as their map chose (`tile`).
    package enum Kernel: String, Codable, Sendable { case graph, flash, split, tiles, tiles16, tilesbf16
        /// The three stages or the fused form (`TiledAttention`), in any precision.
        var isTiled: Bool { self == .tiles || self == .tiles16 || self == .tilesbf16 }
        var precision: TiledAttention.Precision { switch self { case .tiles16: .fp16; case .tilesbf16: .bf16; default: .fp32 } }
    }
    package var kernel: Kernel
    /// The CPU's heads under `split`; 0 otherwise.
    package var cpuHeads: Int
    /// **The kernel of the GPU's heads under `split`** — graph, flash, tiles or tiles16: the head split composes with the GPU's best
    /// kernel (each engine its own best); `graph` (what the split always ran) otherwise.
    package var gpuKernel: Kernel
    /// **The fused form's tile** of the tiled kernel that runs (the whole attention's, or the GPU's heads' under `split`): the map's
    /// choice for that kernel (`CapacityModel.attentionTiles`), attached by the solver; `nil`: the three stages.
    package var tile: AttentionTile?
    /// **The QK-norm and the RoPE before a self-attention on the GPU** (`QKNormRope`), queued after the products that wrote Q and K, instead
    /// of the CPU's pass that leaves the GPU idle and the call paying its wake: the map times both, the solver chooses (`CapacitySolver`).
    /// Never on a cross-attention. `false`: the CPU's pass, what every plan before it ran.
    package var normRopeOnGPU: Bool
    package init(kernel: Kernel, cpuHeads: Int = 0, gpuKernel: Kernel = .graph, tile: AttentionTile? = nil, normRopeOnGPU: Bool = false) {
        self.kernel = kernel; self.cpuHeads = cpuHeads; self.gpuKernel = gpuKernel; self.tile = tile; self.normRopeOnGPU = normRopeOnGPU
    }
    private enum CodingKeys: String, CodingKey { case kernel, cpuHeads, gpuKernel, tile, normRopeOnGPU }
    /// An assignment written before the route reads as the CPU's pass.
    package init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kernel = try c.decode(Kernel.self, forKey: .kernel); cpuHeads = try c.decode(Int.self, forKey: .cpuHeads)
        gpuKernel = try c.decode(Kernel.self, forKey: .gpuKernel); tile = try c.decodeIfPresent(AttentionTile.self, forKey: .tile)
        normRopeOnGPU = try c.decodeIfPresent(Bool.self, forKey: .normRopeOnGPU) ?? false
    }
    package func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(kernel, forKey: .kernel); try c.encode(cpuHeads, forKey: .cpuHeads); try c.encode(gpuKernel, forKey: .gpuKernel)
        try c.encodeIfPresent(tile, forKey: .tile)
        if normRopeOnGPU { try c.encode(true, forKey: .normRopeOnGPU) }
    }
    /// The same assignment with the QK-norm and RoPE where `onGPU` says: what a family's own rule (`QwenImage21DiT.attentionAssignment`…)
    /// keeps when it replaces the kernel.
    package func normRope(onGPU: Bool) -> AttentionAssignment { var a = self; a.normRopeOnGPU = onGPU; return a }
    package var gpuUsesTiles: Bool { splits && gpuKernel.isTiled }
    package var gpuUsesFlash: Bool { splits && gpuKernel == .flash }
    package var gpuTilePrecision: TiledAttention.Precision { gpuKernel.precision }

    package static let graph = AttentionAssignment(kernel: .graph)
    package static let flash = AttentionAssignment(kernel: .flash)
    package static let tiles = AttentionAssignment(kernel: .tiles)
    package static let tiles16 = AttentionAssignment(kernel: .tiles16)
    package static let tilesbf16 = AttentionAssignment(kernel: .tilesbf16)
    package var splits: Bool { kernel == .split }
    package var usesFlash: Bool { kernel == .flash }
    /// The three stages by tiles, in either precision: what the caller builds a `TiledAttention` for.
    package var usesTiles: Bool { kernel.isTiled }
    /// Whether it rounds the attention's operands to bf16 (`tilesbf16`, alone or for the GPU's heads of a split).
    package var roundsToBF16: Bool { kernel == .tilesbf16 || (splits && gpuKernel == .tilesbf16) }
    package var tilePrecision: TiledAttention.Precision { kernel.precision }
    package var description: String {
        let form = tile.map { " \($0)" } ?? ""
        let prologue = normRopeOnGPU ? " · norm+rope gpu" : ""
        return (kernel == .split ? "split \(cpuHeads)" + (gpuKernel == .graph ? "" : " + \(gpuKernel.rawValue)\(form)") : kernel.rawValue + form) + prologue
    }

    /// **The engine's defaults**: the cut by heads from `HeadSplitAttention.minimumSequence` rows at the reference share, the
    /// GPU keeping at least one group; the MPSGraph SDPA below. What every process ran before a plan existed.
    package static func defaults(_ key: AttentionKey) -> AttentionAssignment {
        guard key.sequence >= HeadSplitAttention.minimumSequence else { return .graph }
        let cpu = HeadSplitAttention.partition(heads: key.heads, fraction: HeadSplitAttention.referenceFraction, group: key.group).cpu.count
        return AttentionAssignment(kernel: .split, cpuHeads: cpu)
    }

    /// Whether the engines can run it for `key`: the CPU takes whole groups and the GPU keeps one at least.
    package func isRunnable(for key: AttentionKey) -> Bool {
        switch kernel {
        case .graph, .flash: return cpuHeads == 0 && gpuKernel == .graph
        case .tiles, .tiles16, .tilesbf16: return cpuHeads == 0 && gpuKernel == .graph && TiledAttention.runs(headDim: key.headDim, heads: key.heads, kvHeads: key.kvHeads)
        case .split:
            guard cpuHeads >= 0 && cpuHeads % key.group == 0 && cpuHeads < key.heads else { return false }
            switch gpuKernel {
            case .graph: return true
            case .flash: return key.headDim == FlashMatrix.headDim && key.kvHeads == key.heads
            case .tiles, .tiles16, .tilesbf16: return TiledAttention.runs(headDim: key.headDim, heads: key.heads, kvHeads: key.kvHeads)
            case .split: return false
            }
        }
    }
}

// MARK: - A check's strictest case

extension CapacityPlan {
    /// **A route forced on any Mac, whatever its map says** — a check's strictest case, a judge's reference. Each is a rule over the
    /// answers; the engines it needs are brought up; the rest is the engine's defaults.
    ///
    ///   - `neuralEngine` (`--neural-engine`): the denoiser's widened products of `aneMinimumRows` rows or more give half their
    ///     rows (rounded to the unit) to the ANE, the GPU beside it alone;
    ///   - `bf16Activations` (`--bf16-activations`): the denoiser's bf16 weights of `roundingMinimumRows` rows or more are read as
    ///     published, the activation rounded — and with `neuralEngine`, from `aneMinimumRows`, half their rows (rounded to the unit) go
    ///     to the ANE's fp16 program on the same bf16 copy, beside the GPU's rounded rows (`gpu+ane/bf16act`, the pair a Standard's map
    ///     may choose); so are its 8-bit ones (`gpu/int8act`, `gpu/q8act`: a rounded activation has no other way
    ///     to a published 8-bit weight), the ANE's integers beside them under `aneInt8`;
    ///   - `int8`, `q8`: a published int8 (Q8_0) weight is read as published by the GPU's kernel wherever no other rule speaks of it, in
    ///     every phase — on the fp32 activation, or the rounded one where `bf16Activations` says;
    ///   - `int8Activations` (`--int8-activations`, `int8x8`): every product of the denoiser on a published int8 weight, whatever its
    ///     height, multiplies the activation quantized to int8 (W8A8, `gpu/int8x8`) — as ComfyUI's `int8_linear` runs every quantized
    ///     linear, the modulation's included: the forward the check compares to Comfy's. It has the last word on those products;
    ///   - `amx`: the AMX beside the GPU on every widened product of 512 rows or more, the GPU keeping 0.631 of the rows on 256-row
    ///     tiles — **the split the judges' md5s were taken with**;
    ///   - `flash`: the flash kernel for every attention whose caller tries it (it falls back by itself on a length it does not
    ///     tile);
    ///   - `normRope` (`normrope`): every self-attention's QK-norm and RoPE on the GPU (`QKNormRope`), whatever the kernel — the route
    ///     the map may choose, forced where a check judges it against the reference;
    ///   - `decoderBF16` (`vaebf16`): every decoder whose weights are published in bf16 computes in bf16 (`VAEPrecision`) — the route
    ///     the map's convolution may choose, judged by the decoder stages against the reference decoded in bf16;
    ///   - `tiles`, `tiles16`, `tilesbf16`: the three stages by tiles of query rows for every attention the plan speaks of (fp32; fp16
    ///     or bf16 operands with fp32 sums, whose checks judge against the half-precision reference like the ANE's); `fused`: the
    ///     half-precision one in the fused form, at the first candidate tile (`AttentionTile.candidates`) — a developer's forcing, the
    ///     map choosing otherwise.
    package static func forcing(neuralEngine: Bool = false, bf16Activations: Bool = false, amx: Bool = false, flash: Bool = false,
                                tiles: Bool = false, tiles16: Bool = false, tilesbf16: Bool = false, fused: Bool = false, cut: Bool = false, aneInt8: Bool = false, int8: Bool = false, q8: Bool = false, normRope: Bool = false, decoderBF16: Bool = false,
                                int8Activations: Bool = false, aneMinimumRows: Int = 2 * NeuralEngine.rowUnit,
                                roundingMinimumRows: Int = 256) -> CapacityPlan {
        let neuralEngine = neuralEngine || aneInt8
        var resources = Resources(amx: amx, ane: neuralEngine, bf16Weights: bf16Activations,
                                  int8Weights: aneInt8 || int8 || bf16Activations || int8Activations,
                                  q8Weights: q8 || bf16Activations,
                                  roundedActivations: bf16Activations, aneInt8: aneInt8, halfAttention: tiles16 || tilesbf16,
                                  quantizedActivations: int8Activations)
        resources.halfDecoder = decoderBF16
        let label = "forcing" + (neuralEngine ? " ane≥\(aneMinimumRows)" : "") + (aneInt8 ? " aneint8" : "") + (int8 ? " int8" : "") + (q8 ? " q8" : "")
            + (bf16Activations ? " bf16act≥\(roundingMinimumRows)" : "")
            + (amx ? " amx 0.631" : "") + (flash ? " flash" : "")
            + (tiles16 ? " tiles16" : tilesbf16 ? " tilesbf16" : tiles ? " tiles" : "") + (fused && (tiles16 || tilesbf16) ? " fused" : "")
            + (cut && (tiles || tiles16 || tilesbf16) ? " cut" : "") + (normRope ? " normrope" : "") + (decoderBF16 ? " vaebf16" : "")
            + (int8Activations ? " int8x8" : "")
        var attention: AttentionOverride?
        if tiles16 || tilesbf16 || tiles {
            // Every attention the plan speaks of (the encoders', which no caller asks it, stay the graph's); a flash caller takes tiles too.
            var named = tiles16 ? AttentionAssignment.tiles16 : tilesbf16 ? AttentionAssignment.tilesbf16 : AttentionAssignment.tiles
            if fused, named.kernel != .tiles { named.tile = AttentionTile.candidates[0] }
            let forced = named
            // `cut`: the engine's default cut by heads where it applies, its GPU heads on the forced kernel (the composition).
            attention = { (key: AttentionKey, _: Bool) -> AttentionAssignment? in
                if cut {
                    var split = AttentionAssignment.defaults(key)
                    if split.splits {
                        split.gpuKernel = forced.kernel
                        split.tile = forced.tile
                        return split.isRunnable(for: key) ? split : nil
                    }
                }
                return forced.isRunnable(for: key) ? forced : nil
            }
        } else if flash { attention = { (_: AttentionKey, tries: Bool) -> AttentionAssignment? in tries ? AttentionAssignment.flash : nil } }
        if normRope {
            // The kernel the other words force, or the engine's defaults (what `assign` answers without a rule), with the pass on the GPU.
            let kernel = attention
            attention = { (key: AttentionKey, tries: Bool) -> AttentionAssignment? in
                guard !key.cross else { return kernel?(key, tries) }
                return (kernel?(key, tries) ?? .defaults(key)).normRope(onGPU: true)
            }
        }
        return CapacityPlan(resources: resources, label: label, override: { (key: ProductKey) -> ProductAssignment? in
            // W8A8 on every int8 product of the denoiser it can run (Comfy's forward has no floor of rows).
            if int8Activations, key.weight == .int8, key.phase == .denoising {
                let quantized = ProductAssignment(path: .native, quantizesActivation: true, gpu: key.m)
                if quantized.isRunnable(for: key) { return quantized }
            }
            // The activation rounded where the request rounds it: the denoiser's products of enough rows.
            let rounds = bf16Activations && key.phase == .denoising && key.m >= roundingMinimumRows
            let eightBit = key.weight == .int8 || key.weight == .q8
            // A K-quant on a rounded activation is read as published too (`gpu/q4kact`), the GPU alone.
            let published = eightBit || key.weight == .q4k
            if key.phase == .denoising {
                if rounds, key.weight == .bf16 {
                    if neuralEngine, key.m >= aneMinimumRows {
                        let unit = NeuralEngine.rowUnit
                        let rows = Int((Double(key.m) * 0.5 / Double(unit)).rounded()) * unit
                        let pair = ProductAssignment(path: .bf16, roundsActivation: true, gpu: key.m - rows, ane: rows)
                        if rows > 0 && rows < key.m && pair.isRunnable(for: key) { return pair }
                    }
                    return ProductAssignment(path: .bf16, roundsActivation: true, gpu: key.m)
                }
                if neuralEngine, key.m >= aneMinimumRows {
                    let unit = NeuralEngine.rowUnit
                    let rows = Int((Double(key.m) * 0.5 / Double(unit)).rounded()) * unit
                    if rows > 0 && rows < key.m {
                        // `aneint8`: a published int8 weight is read as it is by both, the GPU's int8 kernel and the ANE's program,
                        // where the engines can run it; the others keep the widened weight.
                        if aneInt8, key.weight == .int8 {
                            let native = ProductAssignment(path: .native, roundsActivation: rounds, gpu: key.m - rows, ane: rows)
                            if native.isRunnable(for: key) { return native }
                        }
                        // A rounded activation has no widened route: the 8-bit weight to the GPU's rounded kernel, whole.
                        if !(rounds && published) { return ProductAssignment(gpu: key.m - rows, ane: rows) }
                    }
                }
                if rounds, published { return ProductAssignment(path: .native, roundsActivation: true, gpu: key.m) }
            }
            // `int8`, `q8`: a published 8-bit weight is read as published by the GPU's kernel wherever the ANE takes none of its rows,
            // in every phase (what the switches `gemm_int8` and `gemm_q8` were) — the AMX leaves it.
            if int8, key.weight == .int8 { return ProductAssignment(path: .native, gpu: key.m) }
            if q8, key.weight == .q8 { return ProductAssignment(path: .native, gpu: key.m) }
            if amx, key.m >= CapacitySolver.Limits.defaultAMXRows {
                let tile = 256
                let gpu = max(tile, min(key.m - tile, Int((Double(key.m) * 0.631 / Double(tile)).rounded()) * tile))
                return ProductAssignment(gpu: gpu, amx: key.m - gpu)
            }
            return nil
        }, attentionOverride: attention)
    }

    /// **The forcing a developer names** (`SILICONED_PLAN`, a comma list of `amx`, `flash`, `tiles`, `tiles16`, `tilesbf16`, `fused`, `ane`,
    /// `aneint8`, `int8`, `q8`, `bf16act`, `normrope`, `vaebf16`, `int8x8`; `tiles16` before `tilesbf16` before `tiles` when several are named; `fused` with a half-precision one; `cut` with one of them: the default cut by heads, its GPU heads on that kernel): the
    /// previous switches of the checks and the judges, in one place. `aneint8` is `ane` with the published int8 weights read as they
    /// are by the ANE's int8 program beside the GPU's int8 kernel; `int8` reads them as published by the GPU's kernel everywhere
    /// else (`aneint8,int8` is the integers' route at every height, as the former `gemm_int8` + `gemm_ane` + `ane_int8` ran it).
    /// `nil` for an empty list; a word it does not know makes the whole list unread.
    package static func forcing(named list: String) -> CapacityPlan? {
        let words = Set(list.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces).lowercased() }.filter { !$0.isEmpty })
        guard !words.isEmpty, words.isSubset(of: ["amx", "flash", "tiles", "tiles16", "tilesbf16", "fused", "cut", "ane", "aneint8", "int8", "q8", "bf16act", "normrope",
                                                "vaebf16", "int8x8"]) else { return nil }
        return forcing(neuralEngine: words.contains("ane"), bf16Activations: words.contains("bf16act"),
                       amx: words.contains("amx"), flash: words.contains("flash"), tiles: words.contains("tiles"),
                       tiles16: words.contains("tiles16"), tilesbf16: words.contains("tilesbf16"), fused: words.contains("fused"), cut: words.contains("cut"), aneInt8: words.contains("aneint8"),
                       int8: words.contains("int8"), q8: words.contains("q8"), normRope: words.contains("normrope"),
                       decoderBF16: words.contains("vaebf16"), int8Activations: words.contains("int8x8"))
    }
}
