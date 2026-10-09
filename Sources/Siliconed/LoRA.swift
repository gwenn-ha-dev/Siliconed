import Foundation

/// **A LoRA stack, as a request carries it.**
///
/// It computes nothing: it keeps the maps open and knows how to **materialize** a module's stack
/// into two buffers the caller gives it. It is `Block` that does the two GEMMs.
///
/// ## The product, and why it is concatenated
///
/// `x(W + Σ αᵢ BᵢAᵢ) = xW + Σ αᵢ (xBᵢ)Aᵢ`. Applying each LoRA separately means **two MPS
/// submissions per LoRA and per GEMM**. A measurement found what that costs: a floor per
/// submission, **constant with the rank**, of 1.5 to 3 points. Three rank-32 LoRAs applied
/// separately would pay three floors — ~10% — where the same thing concatenated into a rank 96
/// pays only **one, 6.5%**.
///
/// The concatenation is exact, and it is algebra, not an approximation:
///
///     Σ αᵢ BᵢAᵢ = [α₁B₁ | α₂B₂ | α₃B₃] · [A₁ ; A₂ ; A₃]
///
/// In the forge's layout — `down = Aᵀ [k, r]`, `up = Bᵀ [r, n]` — this becomes a
/// concatenation **by columns** of `down` and **by rows** of `up`. The strength αᵢ is folded
/// into `down`, once, at materialization time: it cannot be a kernel scalar,
/// since a single GEMM now carries the whole stack.
///
/// ## What it does not do, and this is deliberate
///
/// **It never materializes the whole stack itself.** The maps stay mapped — clean,
/// disposable pages, never swapped. Keeping the expanded stack from one evaluation to the next is
/// the business of `CacheLoRA`, which the DiT owns and which dies with it: ~460 MB for Incase on
/// Krea 2, paid during denoising only, against 2.3 s of materialization per evaluation.
/// Z-Image had only 16 to 68 MB of free pages in a measured render: that is why it was not done
/// there, and why each model measures it.
public final class LoRA {

    /// A forged map and the strength applied to it.
    package struct Layer {
        package let artifact: Artifact
        package let strength: Float
        package let name: String
        package init(artifact: Artifact, strength: Float, name: String) {
            self.artifact = artifact; self.strength = strength; self.name = name
        }
    }

    package enum Failure: Error, CustomStringConvertible {
        case notALoRA(String, String)
        case inconsistentRank(target: String, expected: Int, found: Int)
        case inconsistentShape(target: String, reason: String)
        case missingTarget(String)
        case wrongTarget(path: String, name: String, forgedFor: String, model: String)
        package var description: String {
            switch self {
            case .notALoRA(let path, let kind):
                return "\(path): `kind` is “\(kind)”, not “lora” — it is a model map, "
                     + "not an adapter. Import the LoRA's .safetensors instead."
            case .inconsistentRank(let target, let expected, let found):
                return "\(target): rank \(found) instead of \(expected) — the two halves of one "
                     + "LoRA do not agree"
            case .inconsistentShape(let target, let reason):
                return "\(target): \(reason)"
            case .missingTarget(let path):
                return "\(path): the header does not say which model the LoRA was forged for — "
                     + "re-import it, which writes its target"
            case let .wrongTarget(path, name, forgedFor, model):
                return "LoRA « \(name) » (\(path)) forged for \(forgedFor), the model is "
                     + "\(model): it would touch none of its modules and the image would come out "
                     + "without it — refused before any computation"
            }
        }
    }

    package let layers: [Layer]

    /// **The cumulative rank** — the sum of the ranks of the stack, for a given module.
    ///
    /// It determines the size of the arena slices, and it is known **at request time**:
    /// the engine builds the DiT *after* receiving the stack, so there is no maximum to fix
    /// in advance and nothing to reload when the stack changes. One request, one stack, arenas
    /// sized for it.
    ///
    /// A LoRA that does not touch a module does not contribute to it: the cumulative rank is **per module**,
    /// and `maxRank` is what must be reserved.
    package private(set) var rankPerTarget: [String: Int] = [:]
    package private(set) var maxRank = 0
    /// The largest dimensions encountered, to size the slices only once.
    package private(set) var maxEntry = 0
    package private(set) var maxOutput = 0

    package init(layers: [Layer]) throws {
        self.layers = layers
        for layerPass in layers {
            let kind = layerPass.artifact.header["kind"] as? String ?? "(absent)"
            guard kind == "lora" else { throw Failure.notALoRA(layerPass.artifact.path, kind) }
            for name in layerPass.artifact.order where name.hasSuffix(".down") {
                let target = String(name.dropLast(5))
                guard let down = layerPass.artifact.tensors[name],
                      let up = layerPass.artifact.tensors[target + ".up"] else {
                    throw Failure.inconsistentShape(target: target, reason: "`.up` missing opposite `.down`")
                }
                guard down.shape.count == 2, up.shape.count == 2,
                      down.shape[1] == up.shape[0] else {
                    throw Failure.inconsistentShape(
                        target: target, reason: "down \(down.shape) and up \(up.shape) do not compose")
                }
                rankPerTarget[target, default: 0] += down.shape[1]
                maxEntry = max(maxEntry, down.shape[0])
                maxOutput = max(maxOutput, up.shape[1])
            }
        }
        maxRank = rankPerTarget.values.max() ?? 0
    }

    /// Opens a stack from paths and strengths. An unreadable map **throws** instead of being
    /// skipped: a LoRA believed applied that is not yields a plausible and wrong image,
    /// which is the most expensive failure mode in the repository.
    package convenience init(paths: [(path: String, strength: Float)]) throws {
        var layers: [Layer] = []
        for (path, strength) in paths {
            let artifact = try Artifact(path: path)
            let name = (artifact.header["nom"] as? String)
                ?? (path as NSString).lastPathComponent
            layers.append(Layer(artifact: artifact, strength: strength, name: name))
        }
        try self.init(layers: layers)
    }

    /// **For which model a map was forged**: `cible.modele` of its header, which
    /// `ForgeLoRA` writes from the family's DiT against which it checked every module and
    /// every shape — not from `ss_base_model_version`, which is what the trainer declares.
    ///
    /// Pure function, on the header alone: it is what the test target judges. A map without a
    /// target is refused, not guessed — a single version of the forge lives.
    package static func checkTarget(_ header: [String: Any], path: String,
                                     model: String) throws {
        // on-disk key, kept for existing maps/profiles (.silicon header)
        guard let target = (header["cible"] as? [String: Any])?["modele"] as? String else {
            throw Failure.missingTarget(path)
        }
        guard target == model else {
            throw Failure.wrongTarget(path: path,
                                      name: header["nom"] as? String ?? (path as NSString).lastPathComponent,
                                      forgedFor: target, model: model)
        }
    }

    /// The whole stack must target `model` (the denoiser's identifier) — the first one that does not
    /// target it throws.
    package func checkTarget(model: String) throws {
        for layerPass in layers {
            try LoRA.checkTarget(layerPass.artifact.header, path: layerPass.artifact.path, model: model)
        }
    }

    /// The cumulative rank that applies to this module — `0` if no LoRA touches it.
    package func rank(_ target: String) -> Int { rankPerTarget[target] ?? 0 }

    /// **Materializes a module's stack into two of the caller's buffers.**
    ///
    /// - Parameters:
    ///   - target: the name of the weight **without** `.weight` — `layers.0.attention.to_q`.
    ///   - down: receives `[k, Σr]`, the `down`s concatenated **by columns**, strength folded in.
    ///   - up: receives `[Σr, n]`, the `up`s stacked **by rows**.
    ///   - k, n: the dimensions of the main weight, as the DiT's map lays them out.
    /// - Returns: the cumulative rank written, or `0` if no LoRA touches this module.
    ///
    /// The strength is folded into `down` and not `up` because `down` is the smaller of the
    /// two when `n > k` — the case of `w1`/`w3`, which are two thirds of the work.
    @discardableResult
    package func materialize(_ target: String, k: Int, n: Int,
                             down: UnsafeMutablePointer<Float>,
                             up: UnsafeMutablePointer<Float>) throws -> Int {
        let total = rank(target)
        guard total > 0 else { return 0 }
        var offset = 0
        for layerPass in layers {
            let downName = target + ".down", upName = target + ".up"
            guard let t = layerPass.artifact.tensors[downName],
                  let u = layerPass.artifact.tensors[upName] else { continue }
            guard t.shape.count == 2, u.shape.count == 2 else {
                throw Failure.inconsistentShape(target: target,
                                                reason: "LoRA “\(layerPass.name)”: down \(t.shape), up \(u.shape)")
            }
            let r = t.shape[1]
            guard t.shape[0] == k, u.shape[1] == n else {
                throw Failure.inconsistentShape(
                    target: target,
                    reason: "LoRA “\(layerPass.name)” expects [\(t.shape[0]), \(u.shape[1])] "
                          + "and the weight is [\(k), \(n)]")
            }
            // `down` is `[k, r]` and the destination `[k, Σr]`: we copy **row by row**,
            // each row going into its slice of columns. This is the concatenation by columns.
            // A block copy would write the `r` values of row 0 straddling rows 0
            // and 1 of the destination — a fault that does not crash and yields a plausible image.
            //
            // **And the copy fits in ONE SINGLE `withUnsafeMutableBufferPointer`.** A pointer
            // taken outside its scope is only valid for the duration of the call — it is the fault that
            // `Sampler.swift` documents having paid for on the spectral descent.
            var buffer = [Float](repeating: 0, count: k * r)
            let strength = layerPass.strength
            try buffer.withUnsafeMutableBufferPointer { raw in
                try layerPass.artifact.materialize(downName, into: raw)
                let source = raw.baseAddress!
                for rowLine in 0..<k {
                    let from = source + rowLine * r
                    let to = down + rowLine * total + offset
                    for j in 0..<r { to[j] = from[j] * strength }
                }
            }
            // `up` is `[r, n]` and the destination `[Σr, n]`: the rows stack as they are,
            // so a single contiguous copy at the right height.
            try layerPass.artifact.materialize(upName, into: up + offset * n, capacity: (total - offset) * n)
            offset += r
        }
        return total
    }

    /// **Merge the stack into the widened weight, or apply it after the GEMM?** — by FLOP count, for
    /// one GEMM of `rows` rows on a `[k, n]` weight at cumulative rank `r` (which cancels out):
    /// applying it is `x·down` then `·up`, `2·rows·r·(k + n)`; merging it is `W += down·up` in the
    /// fp32 widening buffer, `2·k·r·n`, whatever the rows. Merging wins beyond `k·n / (k + n)` rows —
    /// 2 048 for Qwen-Image-2.1's `[4096, 4096]`, 3 072 for its MLP's `[4096, 12288]` and
    /// `[12288, 4096]`. **Not the same bits**: `x·(W + ΔW)` rounds `W + ΔW` once per weight, the
    /// application rounds `x·W` and `(x·down)·up` separately — the algebra is exact (measured: 5.3·10⁻⁷).
    /// The merge is in fp32: nothing like a merge into a bf16 weight, whose rounding swallows the delta.
    package static func merges(rows: Int, k: Int, n: Int) -> Bool {
        rows > 0 && k > 0 && n > 0 && rows * (k + n) > k * n
    }

    /// What the stack does, in one line — to display next to the render.
    package var summary: String {
        guard !layers.isEmpty else { return "no LoRA" }
        let detail = layers.map { "\($0.name) ×\(String(format: "%.2f", $0.strength))" }
            .joined(separator: ", ")
        return "\(layers.count) LoRA (cumulative rank \(maxRank)): \(detail)"
    }
}

/// **The expanded weights of a stack, kept from one evaluation to the next.**
///
/// `LoRA.materialize` redoes, at every GEMM of every evaluation, the same thing: read `down` and
/// `up` from the map, widen them to fp32, fold in the strength. The result depends neither on σ nor on the
/// latent — only on the stack and the module. The cache does it **once per module**, at the
/// first evaluation, in an arena of its own sized for the whole stack, and then returns the same
/// addresses: a GEMM that wraps them (`GEMM.wrap`, memoized by address) wraps them
/// only once.
///
/// The price is space: `Σ (k + n)·r` floats — ~460 MB for Incase on Krea 2 (rank 32,
/// 224 modules; +426 MB at the measured peak). It is paid **during denoising only**: the cache belongs to the DiT and
/// dies with it. The values are those of `materialize`, to the bit: it is the same function,
/// called once instead of eight times.
package final class CacheLoRA {
    package let lora: LoRA
    /// The reserved bytes, for all the retained modules.
    package let bytes: Int
    private let arena: Arena
    private var modules: [String: (down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>, rank: Int)] = [:]

    /// - Parameter retain: the modules this cache serves (the DiT's, not those of another stage
    ///   that the same stack touches — Krea 2's text fusion, for example).
    package init(lora: LoRA, retain: (String) -> Bool = { _ in true }) throws {
        self.lora = lora
        var shapes: [String: (k: Int, n: Int)] = [:]
        for layerPass in lora.layers {
            for name in layerPass.artifact.order where name.hasSuffix(".down") {
                let target = String(name.dropLast(5))
                guard retain(target), let down = layerPass.artifact.tensors[name],
                      let up = layerPass.artifact.tensors[target + ".up"] else { continue }
                shapes[target] = (down.shape[0], up.shape[1])
            }
        }
        let page = Arena.alignment
        func rounding(_ bytes: Int) -> Int { (bytes + page - 1) / page * page }
        let total = shapes.reduce(0) { sum, entry in
            let r = lora.rank(entry.key)
            return sum + rounding(entry.value.k * r * 4) + rounding(r * entry.value.n * 4)
        }
        bytes = total
        arena = try Arena(capacity: max(total, page))
    }

    /// The stack of module `target` (without `.weight`), expanded — on the first call only. `nil` if
    /// no LoRA of the stack touches it.
    package func module(_ target: String, k: Int, n: Int) throws
        -> (down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>, rank: Int)? {
        if let ready = modules[target] { return ready }
        let r = lora.rank(target)
        guard r > 0 else { return nil }
        let down = try arena.reserve(target + ".bas", bytes: k * r * 4).assumingMemoryBound(to: Float.self)
        let up = try arena.reserve(target + ".haut", bytes: r * n * 4).assumingMemoryBound(to: Float.self)
        let rank = try lora.materialize(target, k: k, n: n, down: down, up: up)
        modules[target] = (down, up, rank)
        return (down, up, rank)
    }
}

