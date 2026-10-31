import Accelerate
import Foundation

/// **The LoRA stack expanded one layer ahead, not all at once.** `CacheLoRA` keeps the whole stack in
/// fp32 for the render: a rank-256 LoRA on the 224 modules of the blocks is 2.68 GB resident, more than
/// any other item of the denoising stage. Here only two layers live (2 × 84 MB): layer `l` being read by the GPU
/// and layer `l + 1` being expanded on a background thread meanwhile — the same `LoRA.materialize`, so
/// the same bits, redone at each evaluation (~5 s of CPU per evaluation, hidden behind the GEMMs).
///
/// The two slots are **fixed arenas with a fixed layout** (every layer has the same modules and
/// shapes): a module's address is that of its slot, so `GEMM.wrap`'s memo by address stays valid —
/// freeing and remapping a slot would hand the GPU a stale mapping. Slot `l mod 2` is refilled for
/// layer `l + 2` only once layer `l + 1` starts, when the GPU has finished with layer `l` (every GEMM
/// waits for its command buffer). The modules outside the blocks (`txt_in`, `img_in`, `proj_out`,
/// read once per evaluation) stay in a small `CacheLoRA`.
package final class StreamedLoRA {
    package typealias Module = (down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>, rank: Int)
    private let lora: LoRA
    private let layers: Int
    private let blockPrefix = "transformer_blocks."
    /// Per module suffix (`attn.to_q`): `(down offset, up offset, k, n)` in bytes within a slot.
    private var layout: [String: (down: Int, up: Int, k: Int, n: Int)] = [:]
    private var slots: [Arena] = []
    private var filled: [Int?] = [nil, nil]
    private var pending: [DispatchGroup?] = [nil, nil]
    private var failures: [Error?] = [nil, nil]
    private let others: CacheLoRA
    package let bytes: Int

    /// - Parameter retain: the modules served (the exact ones are computed elsewhere, in double).
    package init(lora: LoRA, layers: Int, retain: @escaping (String) -> Bool) throws {
        self.lora = lora; self.layers = layers
        var shapes: [String: (k: Int, n: Int)] = [:]
        for layer in lora.layers {
            for name in layer.artifact.order where name.hasSuffix(".down") {
                let target = String(name.dropLast(5))
                guard target.hasPrefix(blockPrefix + "0."), retain(target), let down = layer.artifact.tensors[name],
                      let up = layer.artifact.tensors[target + ".up"] else { continue }
                shapes[String(target.dropFirst((blockPrefix + "0.").count))] = (down.shape[0], up.shape[1])
            }
        }
        var modules: [String: (k: Int, n: Int, rank: Int)] = [:]
        for (suffix, shape) in shapes { modules[suffix] = (shape.k, shape.n, lora.rank(blockPrefix + "0." + suffix)) }
        let laid = StreamedLoRA.layout(modules)
        layout = laid.layout
        let offset = laid.bytes
        // Every block must carry the same modules at the same rank as block 0.
        for target in lora.rankPerTarget.keys where target.hasPrefix(blockPrefix) && retain(target) {
            let parts = target.dropFirst(blockPrefix.count).split(separator: ".", maxSplits: 1)
            guard parts.count == 2, let suffix = parts.last.map(String.init), layout[suffix] != nil,
                  lora.rank(target) == lora.rank(blockPrefix + "0." + suffix) else {
                throw LoRA.Failure.inconsistentShape(target: target, reason: "a block whose LoRA modules differ from block 0's")
            }
        }
        bytes = offset
        if offset > 0 { slots = [try Arena(capacity: offset), try Arena(capacity: offset)] }
        others = try CacheLoRA(lora: lora, retain: { !$0.hasPrefix("transformer_blocks.") && retain($0) })
    }

    /// **One slot's layout** — each module's `down` then `up`, page-rounded, in suffix order — and its
    /// bytes. Two slots of it live during the denoising; `MemoryNeed.qwenImage21` charges them from here.
    package static func layout(_ modules: [String: (k: Int, n: Int, rank: Int)])
            -> (layout: [String: (down: Int, up: Int, k: Int, n: Int)], bytes: Int) {
        let page = Arena.alignment
        func rounded(_ b: Int) -> Int { (b + page - 1) / page * page }
        var layout: [String: (down: Int, up: Int, k: Int, n: Int)] = [:]
        var offset = 0
        for suffix in modules.keys.sorted() {
            let (k, n, r) = modules[suffix]!
            layout[suffix] = (offset, offset + rounded(k * r * 4), k, n)
            offset += rounded(k * r * 4) + rounded(r * n * 4)
        }
        return (layout, offset)
    }

    /// The stack of module `target`, expanded — `nil` if no LoRA touches it.
    package func module(_ target: String, k: Int, n: Int) throws -> Module? {
        guard target.hasPrefix(blockPrefix) else { return try others.module(target, k: k, n: n) }
        let parts = target.dropFirst(blockPrefix.count).split(separator: ".", maxSplits: 1)
        guard parts.count == 2, let layer = Int(parts[0]), let place = layout[String(parts[1])] else {
            return lora.rank(target) > 0 ? try others.module(target, k: k, n: n) : nil
        }
        precondition(place.k == k && place.n == n, "StreamedLoRA: \(target) is [\(k), \(n)], laid out [\(place.k), \(place.n)]")
        let slot = layer % 2
        if filled[slot] != layer { try fill(layer, sync: true) }
        try collect(slot)
        // The other slot held layer − 1, which the GPU has finished: refill it with layer + 1.
        let next = (layer + 1) % layers
        if filled[next % 2] != next { try fill(next, sync: false) }
        let base = slots[slot].pointer("slot")!
        return ((base + place.down).assumingMemoryBound(to: Float.self), (base + place.up).assumingMemoryBound(to: Float.self),
                lora.rank(target))
    }

    private func collect(_ slot: Int) throws {
        if let group = pending[slot] { group.wait(); pending[slot] = nil }
        if let failure = failures[slot] { failures[slot] = nil; filled[slot] = nil; throw failure }
    }

    /// Expands every module of `layer` into its slot — now, or on a background thread.
    private func fill(_ layer: Int, sync: Bool) throws {
        let slot = layer % 2
        if let group = pending[slot] { group.wait(); pending[slot] = nil }
        if slots[slot].pointer("slot") == nil { _ = try slots[slot].reserve("slot", bytes: bytes) }
        filled[slot] = layer
        failures[slot] = nil
        let base = slots[slot].pointer("slot")!
        let work = { [lora, layout, blockPrefix] () -> Error? in
            do {
                for (suffix, place) in layout {
                    _ = try lora.materialize(blockPrefix + "\(layer)." + suffix, k: place.k, n: place.n,
                                             down: (base + place.down).assumingMemoryBound(to: Float.self),
                                             up: (base + place.up).assumingMemoryBound(to: Float.self))
                }
                return nil
            } catch { return error }
        }
        if sync {
            failures[slot] = work()
            return
        }
        let group = DispatchGroup()
        nonisolated(unsafe) let task = work
        nonisolated(unsafe) let owner = self
        DispatchQueue.global(qos: .userInitiated).async(group: group) { owner.failures[slot] = task() }
        pending[slot] = group
    }

    deinit { for group in pending { group?.wait() } }
}
