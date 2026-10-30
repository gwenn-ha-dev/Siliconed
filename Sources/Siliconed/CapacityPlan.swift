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
///   - **A process without the ANE gets a plan computed without it** (`restricted(allowingANE:)`): the Standard, an encoder, a
///     request that refused it. There is no fallback route behind a rule that names the ANE; the solver never named it.
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
        package init(amx: Bool = false, ane: Bool = false, bf16Weights: Bool = false, int8Weights: Bool = false,
                     q8Weights: Bool = false, roundedActivations: Bool = false) {
            self.amx = amx; self.ane = ane; self.bf16Weights = bf16Weights; self.int8Weights = int8Weights
            self.q8Weights = q8Weights; self.roundedActivations = roundedActivations
        }
        /// `TensorGEMM` is built when any native form is.
        package var usesTensor: Bool { bf16Weights || int8Weights || q8Weights || roundedActivations }
        /// Some product may compute in half precision (the ANE's fp16, the activation rounded to bf16): the checks then
        /// judge against the reference run in bf16 (`HalfPrecisionReference`).
        package var halfPrecision: Bool { ane || roundedActivations }
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
    package var usesAMX: Bool { resources.amx }
    package var usesANE: Bool { resources.ane }
    /// **The height under which no product has a row on the AMX or the ANE**: the model's domain (`CapacitySolver.Limits`) for a
    /// plan solved from a model; 1 for a plan that forces or pins (any height may carry rows). What the memory need's sweep over
    /// the heights a prompt may give (`MemoryNeed.gemm`) starts from.
    package var engineFloor: Int {
        guard model != nil else { return 1 }
        return min(resources.amx ? limits.amxRows : .max, resources.ane ? limits.aneRows : .max)
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
    }
    private struct Memo: Hashable { var key: AttentionKey; var flash: Bool }
    private let state = Mutex(State())
    /// The plan of a process without the ANE, made once: its memo is as valuable as this one's.
    private let withoutANE = Mutex<CapacityPlan?>(nil)

    // MARK: - Making a plan

    /// **The engine's defaults**: the GPU alone, the widened weight, the head split from 1 024 rows at the reference share.
    package static let defaults = CapacityPlan()

    private init() {
        resources = Resources(); identity = "default"; model = nil; allowsANE = false
        limits = CapacitySolver.Limits(); sustained = true; excluded = []; override = nil; attentionOverride = nil
    }

    /// **The plan of a Mac whose model was measured**: the solver answers from it. The engines it brings up are those the model
    /// has a line for; the ANE only if `allowsANE` (a Standard does not).
    package init(model: CapacityModel, limits: CapacitySolver.Limits = .init(), sustained: Bool = true,
                 excluded: Set<CapacityModel.Kernel> = [], allowsANE: Bool = true) {
        func has(_ e: CapacityModel.Engine, _ f: CapacityModel.Form) -> Bool {
            let k = CapacityModel.Kernel(e, f)
            return (model.lines[k]?.rate ?? 0) > 0 && !excluded.contains(k)
        }
        self.model = model; self.limits = limits; self.sustained = sustained; self.excluded = excluded
        self.allowsANE = allowsANE
        resources = Resources(amx: has(.amx, .fp32), ane: allowsANE && has(.ane, .fp16),
                              bf16Weights: has(.gpu, .bf16), int8Weights: has(.gpu, .int8), q8Weights: has(.gpu, .q8),
                              roundedActivations: has(.gpu, .bf16act))
        override = nil; attentionOverride = nil
        identity = "model " + Self.fingerprint(model) + (allowsANE ? "" : " no-ane")
    }

    /// **A plan that forces its answers**: a check's strictest case, a probe's route. `resources` are the engines it brings up;
    /// the key it does not speak of gets the engine's defaults.
    package init(resources: Resources, label: String, allowsANE: Bool = true, override: Override? = nil,
                 attentionOverride: AttentionOverride? = nil) {
        self.resources = resources; identity = label; model = nil; self.allowsANE = allowsANE && resources.ane
        limits = CapacitySolver.Limits(); sustained = true; excluded = []; self.override = override
        self.attentionOverride = attentionOverride
    }

    /// **A probe's plan** (the previous map's balancing, a bench's solos): the engines it builds, every product on the GPU alone
    /// until a key is `pin`ned to the exact rows to measure.
    package static func probing(amx: Bool = false, ane: Bool = false, bf16Weights: Bool = false, int8Weights: Bool = false,
                                q8Weights: Bool = false, roundedActivations: Bool = false) -> CapacityPlan {
        CapacityPlan(resources: Resources(amx: amx, ane: ane, bf16Weights: bf16Weights, int8Weights: int8Weights,
                                          q8Weights: q8Weights, roundedActivations: roundedActivations), label: "probing")
    }

    /// The same model and rules for a process that may not bring up the ANE (`allowingANE: false`): its own answers, its own memo.
    package func restricted(allowingANE allowed: Bool) -> CapacityPlan {
        if allowed || !allowsANE { return self }
        return withoutANE.withLock { kept in
            if let kept { return kept }
            var r = resources; r.ane = false
            let made: CapacityPlan
            if let model {
                made = CapacityPlan(model: model, limits: limits, sustained: sustained, excluded: excluded, allowsANE: false)
            } else {
                made = CapacityPlan(resources: r, label: identity + " no-ane", allowsANE: false, override: override,
                                    attentionOverride: attentionOverride)
            }
            kept = made
            return made
        }
    }

    /// A hash of every number of the model, in a fixed order (a dictionary keyed by a kernel or an engine encodes in no order).
    private static func fingerprint(_ model: CapacityModel) -> String {
        var parts: [String] = []
        for (k, l) in model.lines.sorted(by: { $0.key.description < $1.key.description }) { parts.append("\(k) \(l.t0) \(l.rate)") }
        parts.append("unit \(model.aneRowUnit)")
        for (k, s) in model.sharing.sorted(by: { $0.key < $1.key }) {
            parts.append("\(k) \(s.overhead) " + s.keep.sorted { $0.key < $1.key }.map { "\($0.key.rawValue)=\($0.value)" }.joined(separator: ","))
        }
        for (e, h) in model.heat.sorted(by: { $0.key < $1.key }) { parts.append("heat \(e.rawValue) \(h)") }
        for (e, c) in model.crest.sorted(by: { $0.key < $1.key }) { parts.append("crest \(e.rawValue) \(c)") }
        parts.append("memory \(model.memoryBandwidth.map { "\($0)" } ?? "-") disk \(model.diskBandwidth.map { "\($0)" } ?? "-")")
        for (kind, paths) in model.preparation.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            parts.append("prep \(kind.rawValue) " + paths.sorted { $0.key.rawValue < $1.key.rawValue }.map { "\($0.key.rawValue)=\($0.value)" }.joined(separator: ","))
        }
        parts.append("kquant \(model.kquantPreparation.map { "\($0)" } ?? "-")")
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
        var a = override?(key) ?? CapacityPlan.solved(self, key)
        if a.ane > 0 && !allowsANE { a.gpu += a.ane; a.ane = 0 }
        let kept = a
        state.withLock { $0.products[key] = kept }
        return kept
    }

    private static func solved(_ plan: CapacityPlan, _ key: ProductKey) -> ProductAssignment {
        guard let model = plan.model else { return .gpuAlone(key.m) }
        let c = CapacitySolver.Constraints(limits: plan.limits, sustained: plan.sustained, excluded: plan.excluded,
                                           allowsANE: plan.allowsANE)
        return CapacitySolver.product(key, model: model, constraints: c)?.assignment ?? .gpuAlone(key.m)
    }

    /// **The assignment of an attention.** `flash`: the caller's code tries the flash kernel before the split (Z-Image's and
    /// Anima's self-attentions); a caller that does not implement it never gets it.
    package func assign(_ key: AttentionKey, flash: Bool = false) -> AttentionAssignment {
        let memo = Memo(key: key, flash: flash)
        if let known = state.withLock({ $0.attentions[memo] }) { return known }
        let a: AttentionAssignment
        if let forced = attentionOverride?(key, flash) {
            a = forced
        } else if let model {
            let c = CapacitySolver.Constraints(limits: limits, sustained: sustained, excluded: excluded, allowsANE: allowsANE)
            a = CapacitySolver.attention(key, flash: flash, model: model, constraints: c)?.assignment ?? .defaults(key)
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

// MARK: - The weight's form

/// What a weight is, for the plan: the form it is published in. (A K-quant is none of these: `CapacityClasses.WeightFormat.q4k`
/// has no native form, it widens.)
package enum WeightKind: String, Codable, Sendable, CaseIterable { case bf16, int8, q8 }

/// How a product reads its weight.
package enum WeightPath: String, Codable, Sendable {
    /// Widened to fp32 on the CPU (the default path: MPS, the AMX, the ANE read it).
    case widened
    /// The bf16 weight as published (`TensorGEMM`).
    case bf16
    /// The 8-bit weight as published (int8 convrot, Q8_0).
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
    /// The activation is rounded to bf16 before the product (`gemm_bf16_act`).
    package var roundsActivation: Bool
    package var gpu: Int
    package var amx: Int
    package var ane: Int
    /// Beside a native 8-bit weight the AMX takes its last columns, for all the rows (the GPU the others, `gpu` = m): the
    /// columns it takes, 0 for none. Its widening is done for them alone.
    package var amxColumns: Int

    package init(path: WeightPath = .widened, roundsActivation: Bool = false, gpu: Int, amx: Int = 0, ane: Int = 0,
                 amxColumns: Int = 0) {
        self.path = path; self.roundsActivation = roundsActivation; self.gpu = gpu; self.amx = amx; self.ane = ane
        self.amxColumns = amxColumns
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
    /// `gpu/int8`, `gpu+amx/q8`… for a weight read as published.
    package func route(weight: CapacityClasses.WeightFormat) -> String {
        switch path {
        case .widened: return engines
        case .bf16: return roundsActivation ? "gpu/bf16act" : "gpu/bf16"
        case .native: return (amxColumns > 0 ? "gpu+amx/" : "gpu/") + (weight == .int8 ? "int8" : "q8")
        }
    }

    /// **Whether the engines can run it for `key`** — the hardware and precision limits, nothing else: no negative rows, rows
    /// that add up to the height; the ANE's in units of 256 or the whole product, and never on an encoder; a bf16 weight read
    /// as published whole on the GPU; a native 8-bit weight with no ANE, its AMX columns beside a GPU side of whole 64-column
    /// tiles (the kernel writes whole tiles, an edge inside one would overwrite the AMX's first columns).
    package func isRunnable(for key: ProductKey) -> Bool {
        guard gpu >= 0, amx >= 0, ane >= 0, amxColumns >= 0, rows == key.m, key.m > 0 else { return false }
        if ane > 0 {
            guard ane % NeuralEngine.rowUnit == 0 || ane == key.m, key.phase == .denoising, path == .widened else { return false }
        }
        if roundsActivation && (path == .widened || key.phase == .encoding) { return false }
        switch path {
        case .widened:
            return amxColumns == 0
        case .bf16:
            return amx == 0 && ane == 0 && amxColumns == 0 && key.weight == .bf16
        case .native:
            guard amx == 0, ane == 0, key.weight == .int8 || key.weight == .q8 else { return false }
            if amxColumns == 0 { return true }
            let gpuColumns = key.n - amxColumns
            return key.n >= 128 && gpuColumns >= 64 && gpuColumns % 64 == 0
        }
    }

    package var description: String {
        "\(path.rawValue)\(roundsActivation ? "◐" : "") gpu \(gpu) amx \(amx)\(amxColumns > 0 ? " (+\(amxColumns) columns)" : "") ane \(ane)"
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
    package enum Kernel: String, Codable, Sendable { case graph, flash, split }
    package var kernel: Kernel
    /// The CPU's heads under `split`; 0 otherwise.
    package var cpuHeads: Int
    package init(kernel: Kernel, cpuHeads: Int = 0) { self.kernel = kernel; self.cpuHeads = cpuHeads }

    package static let graph = AttentionAssignment(kernel: .graph)
    package static let flash = AttentionAssignment(kernel: .flash)
    package var splits: Bool { kernel == .split }
    package var usesFlash: Bool { kernel == .flash }
    package var description: String { kernel == .split ? "split \(cpuHeads)" : kernel.rawValue }

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
        case .graph, .flash: return cpuHeads == 0
        case .split: return cpuHeads >= 0 && cpuHeads % key.group == 0 && cpuHeads < key.heads
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
    ///     published, the activation rounded;
    ///   - `amx`: the AMX beside the GPU on every widened product of 512 rows or more, the GPU keeping 0.631 of the rows on 256-row
    ///     tiles — **the split the judges' md5s were taken with**;
    ///   - `flash`: the flash kernel for every attention whose caller tries it (it falls back by itself on a length it does not
    ///     tile).
    package static func forcing(neuralEngine: Bool = false, bf16Activations: Bool = false, amx: Bool = false, flash: Bool = false,
                                aneMinimumRows: Int = 2 * NeuralEngine.rowUnit, roundingMinimumRows: Int = 256) -> CapacityPlan {
        let resources = Resources(amx: amx, ane: neuralEngine, bf16Weights: bf16Activations, roundedActivations: bf16Activations)
        let label = "forcing" + (neuralEngine ? " ane≥\(aneMinimumRows)" : "") + (bf16Activations ? " bf16act≥\(roundingMinimumRows)" : "")
            + (amx ? " amx 0.631" : "") + (flash ? " flash" : "")
        var attention: AttentionOverride?
        if flash { attention = { (_: AttentionKey, tries: Bool) -> AttentionAssignment? in tries ? AttentionAssignment.flash : nil } }
        return CapacityPlan(resources: resources, label: label, override: { (key: ProductKey) -> ProductAssignment? in
            if key.phase == .denoising {
                if bf16Activations, key.weight == .bf16, key.m >= roundingMinimumRows {
                    return ProductAssignment(path: .bf16, roundsActivation: true, gpu: key.m)
                }
                if neuralEngine, key.m >= aneMinimumRows {
                    let unit = NeuralEngine.rowUnit
                    let rows = Int((Double(key.m) * 0.5 / Double(unit)).rounded()) * unit
                    if rows > 0 && rows < key.m { return ProductAssignment(gpu: key.m - rows, ane: rows) }
                }
            }
            if amx, key.m >= CapacitySolver.Limits.defaultAMXRows {
                let tile = 256
                let gpu = max(tile, min(key.m - tile, Int((Double(key.m) * 0.631 / Double(tile)).rounded()) * tile))
                return ProductAssignment(gpu: gpu, amx: key.m - gpu)
            }
            return nil
        }, attentionOverride: attention)
    }

    /// **The forcing a developer names** (`SILICONED_PLAN`, a comma list of `amx`, `flash`, `ane`, `bf16act`): the previous switches
    /// of the checks and the judges, in one place. `nil` for an empty list; a word it does not know makes the whole list unread.
    package static func forcing(named list: String) -> CapacityPlan? {
        let words = Set(list.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces).lowercased() }.filter { !$0.isEmpty })
        guard !words.isEmpty, words.isSubset(of: ["amx", "flash", "ane", "bf16act"]) else { return nil }
        return forcing(neuralEngine: words.contains("ane"), bf16Activations: words.contains("bf16act"),
                       amx: words.contains("amx"), flash: words.contains("flash"))
    }
}
