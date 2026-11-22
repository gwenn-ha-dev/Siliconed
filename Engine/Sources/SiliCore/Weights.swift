import CoreAI
import Foundation

/// **A file's tensors into the arena**: their layout from its header alone, their reading, a product's parts, its
/// scales and targets.
public enum Weights {
    public typealias Entry = (name: String, key: String, recipe: Recipe)
    public typealias Slot = (name: String, shape: [Int], type: NDArray.ScalarType, bytes: Int)

    /// The tensors `entries` become in the arena, from the file's header alone: each argument's name, its shape and
    /// element as the functions read it, its bytes.
    public static func layout(_ entries: [Entry], in file: SafeTensors) throws -> [Slot] {
        try entries.flatMap { e -> [Slot] in
            switch e.recipe {
            case .codes, .product:
                let (_, out, inn, int8) = try product(e, in: file)
                guard int8 else { return [(e.name, [out, inn], .bfloat16, 2 * out * inn)] }
                let (count, piece) = try pieces(inn)
                return [(e.name, [count, out, piece], .int8, out * inn), (e.name + "_scale", [out], .float16, out * 2)]
            case .half, .float:
                let t = try file.real(e.key), n = t.shape.reduce(1, *)
                return e.recipe == .half ? [(e.name, t.shape, .float16, n * 2)] : [(e.name, t.shape, .float32, n * 4)]
            case .halfLast(let axis):
                let t = try file.real(e.key)
                guard axis < t.shape.count else { throw Failure("\(e.key): \(t.shape) has no axis \(axis)") }
                var shape = t.shape
                shape.remove(at: axis)
                return [(e.name, shape, .float16, shape.reduce(1, *) * 2)]
            case .target:
                return [(e.name, [1, 1, 1], .float16, 2)]
            case .bfloat16(let joined):
                let ts = try ([e.key] + joined).map { try file.real($0) }, n = ts.reduce(0) { $0 + $1.shape.reduce(1, *) }
                guard ts.dropFirst().allSatisfy({ Array($0.shape.dropFirst()) == Array(ts[0].shape.dropFirst()) }) else {
                    throw Failure("\(e.name): \(([e.key] + joined).joined(separator: ", ")) do not join along their first axis")
                }
                return [(e.name, [ts.reduce(0) { $0 + ($1.shape.first ?? 1) }] + ts[0].shape.dropFirst(), .bfloat16, n * 2)]
            case .halfTransposed, .floatTransposed:
                let t = try file.real(e.key)
                let half = e.recipe == .halfTransposed
                return [(e.name, [t.shape[1], t.shape[0]], half ? .float16 : .float32, t.shape[0] * t.shape[1] * (half ? 2 : 4))]
            case .bfloatTransposed:
                let t = try file.real(e.key)
                return [(e.name, [t.shape[1], t.shape[0]], .bfloat16, t.shape[0] * t.shape[1] * 2)]
            }
        }
    }

    /// What `entries` take in the arena.
    public static func size(_ entries: [Entry], in file: SafeTensors) throws -> Int {
        try layout(entries, in: file).reduce(0) { $0 + Arena.aligned($1.bytes) }
    }

    /// **The widest input a product on codes reads at once**: k cut in `count` equal pieces of `piece` inputs, at most
    /// 4096 and a multiple of 64 each (one piece where k is no wider). The ANE's rate falls past it (measured by the former
    /// engine, then on 20/11/2026 on the M1 Pro, a block of width 6144 at 1024 rows on the ANE alone: its feed-forward
    /// 194 ms read whole, 82 cut in two, 3.2 → 7.5 TFLOP/s; the GPU's the same either way), and Core AI's int8 × int8 on a
    /// GPU without int8 units falls with k (22/11 on the M1 Pro, 4096 rows: 16 384 → 6144 at 2.17 TFLOP/s whole, 3.96 in
    /// pieces, q64 3.94; 10 240 → 3840 3.56 and 3.90; the M5 Pro 42.0 and 44.9 in pieces). The codes lie by these pieces
    /// (`layout`), each whole: x's pieces read as slices keep Core AI's fusion, the codes' as slices of their rows lose it
    /// (≈ 24 TFLOP/s against 45 on the M5 Pro). A file's codes of a k no such pieces cut are refused, not read: a model
    /// of other widths than its client's is a file's fault, said, never the worker stopped.
    public static func pieces(_ k: Int) throws -> (count: Int, piece: Int) {
        guard let p = cut(k) else { throw Failure("codes of \(k) inputs: no equal pieces of at most \(widest), a multiple of 64 each") }
        return p
    }

    /// **The same for a function's arguments** (`codesShape`, the products' graphs), written at its client's own widths:
    /// a k no pieces cut there is the client's fault, not a file's.
    static func written(_ k: Int) -> (count: Int, piece: Int) {
        guard let p = cut(k) else { preconditionFailure("a function of \(k) inputs: no equal pieces of at most \(widest), a multiple of 64 each") }
        return p
    }

    private static func cut(_ k: Int) -> (count: Int, piece: Int)? {
        guard k > widest else { return (1, k) }
        return ((k + widest - 1) / widest...(k / 64)).first(where: { k % $0 == 0 && (k / $0) % 64 == 0 }).map { ($0, k / $0) }
    }

    static let widest = 4096

    /// The shape a product's codes `[out, in]` take in the arena, and their argument: `[p, out, in / p]` (`pieces`).
    public static func codesShape(inn: Int, out: Int) -> [Int] {
        let (count, piece) = written(inn)
        return [count, out, piece]
    }

    /// What reading `entries` passes through before its recipe: the largest tensor converted (an f32 read as it is goes
    /// straight to its place).
    public static func staging(_ entries: [Entry], in file: SafeTensors) throws -> Int {
        try entries.map { e in
            switch e.recipe {
            case .codes, .product:
                let (parts, _, inn, int8) = try product(e, in: file)
                // A part cut in pieces through the staging; the scales through it.
                let transposed = try int8 && pieces(inn).count > 1
                return try parts.map { key, t in
                    let s = int8 ? try scales(key, in: file, out: t.shape[0]) : nil
                    return max(transposed ? t.end - t.start : 0, s.map { $0.end - $0.start } ?? 0)
                }.max()!
            case .float: let t = try file.real(e.key); return t.dtype == "F32" ? 0 : t.end - t.start
            case .target: return 0
            case .bfloat16(let joined):
                return try ([e.key] + joined).map { k in let t = try file.real(k); return t.dtype == "F32" ? t.end - t.start : 0 }.max()!
            default: let t = try file.real(e.key); return t.end - t.start
            }
        }.max() ?? 0
    }

    /// Every tensor of `layout` taken in `arena` (in its heap `heap`), by its argument's name; `surfaces`: each a surface
    /// of its own, at its shape (what the ANE reads, `Arena`).
    public static func place(_ layout: [Slot], in arena: Arena, heap: Int = 0, surfaces: Bool = false) -> [String: Placed] {
        Dictionary(uniqueKeysWithValues: layout.map {
            let key = surfaces ? arena.take(surface: $0.shape, element: $0.bytes / $0.shape.reduce(1, *)) : arena.take($0.bytes, in: heap)
            arena.wire(key, bytes: $0.bytes)
            return ($0.name, Placed(offset: key, shape: $0.shape, type: $0.type))
        })
    }

    /// Each entry of `file` read into its place in `placed` (taken before, by `place` on the same layout or another
    /// block's of the same shapes), through `staging`.
    public static func read(_ entries: [Entry], from file: SafeTensors, into placed: [String: Placed], arena: Arena,
                            staging: UnsafeMutableRawPointer) throws {
        func place(_ name: String, _ shape: [Int]) -> UnsafeMutableRawPointer {
            guard let p = placed[name] else { preconditionFailure("no place for \(name)") }
            precondition(p.shape == shape, "\(name): \(shape) into a place of \(p.shape)")
            return arena.pointer(p.offset)
        }
        for e in entries {
            switch e.recipe {
            case .codes, .product:
                let (parts, total, inn, int8) = try product(e, in: file)
                let (count, piece) = int8 ? try pieces(inn) : (1, inn)
                let to = place(e.name, int8 ? [count, total, piece] : [total, inn])
                var column = 0
                for (key, t) in parts {
                    let out = t.shape[0], size = int8 ? 1 : 2
                    if !int8 || count == 1 {
                        // The file's rows as they lie, each part's after the one before.
                        try file.read(t, into: to + column * inn * size)
                    } else {
                        // Each row cut in its pieces, piece i of every row after piece i − 1's: copies, on every core.
                        try file.read(t, into: staging)
                        nonisolated(unsafe) let (from, codes) = (UnsafeRawPointer(staging), to)
                        let rows = column
                        DispatchQueue.concurrentPerform(iterations: count) { i in
                            for o in 0..<out {
                                (codes + (i * total + rows + o) * piece).copyMemory(from: from + o * inn + i * piece, byteCount: piece)
                            }
                        }
                    }
                    if int8 {
                        let s = try scales(key, in: file, out: out)
                        let sto = place(e.name + "_scale", [total]) + 2 * column
                        try file.read(s, into: staging)
                        if s.dtype == "F32" { half(staging, out, into: sto) } else { bf16(staging, out, into: sto.bindMemory(to: Float16.self, capacity: out)) { Float16($0) } }
                    }
                    column += out
                }
            case .half:
                let t = try file.real(e.key), n = t.shape.reduce(1, *)
                let to = place(e.name, t.shape)
                try file.read(t, into: staging)
                if t.dtype == "F32" { half(staging, n, into: to) } else { bf16(staging, n, into: to.bindMemory(to: Float16.self, capacity: n)) { Float16($0) } }
            case .halfLast(let axis):
                let t = try file.real(e.key), (outer, steps) = (t.shape[..<axis].reduce(1, *), t.shape[axis])
                let inner = t.shape[(axis + 1)...].reduce(1, *), size = t.dtype == "F32" ? 4 : 2
                var shape = t.shape
                shape.remove(at: axis)
                let to = place(e.name, shape)
                try file.read(t, into: staging)
                for o in 0..<outer {
                    let from = staging + ((o * steps + steps - 1) * inner) * size, into = to + o * inner * 2
                    if t.dtype == "F32" { half(from, inner, into: into) } else { bf16(from, inner, into: into.bindMemory(to: Float16.self, capacity: inner)) { Float16($0) } }
                }
            case .float:
                let t = try file.real(e.key), n = t.shape.reduce(1, *)
                let to = place(e.name, t.shape)
                if t.dtype == "F32" { try file.read(t, into: to) }
                else {
                    try file.read(t, into: staging)
                    bf16(staging, n, into: to.bindMemory(to: Float.self, capacity: n)) { $0 }
                }
            case .target(let products):
                let e0 = products.map { name -> Double in
                    guard let w = placed[name] else { preconditionFailure("\(e.name): no \(name) read before") }
                    guard let scales = placed[name + "_scale"] else {
                        precondition(w.type == .float16 && w.shape.count == 2, "\(e.name): \(name) neither codes nor f16 [in, out]")
                        return target16(weights: UnsafePointer(arena.pointer(w.offset).assumingMemoryBound(to: Float16.self)), count: w.shape.reduce(1, *), k: w.shape[0])
                    }
                    let s = UnsafePointer(arena.pointer(scales.offset).assumingMemoryBound(to: Float16.self))
                    return target16(scales: s, count: scales.shape.reduce(1, *), k: w.shape[0] * w.shape[2])
                }.min()!
                place(e.name, [1, 1, 1]).assumingMemoryBound(to: Float16.self).pointee = Float16(e0)
            case .bfloatTransposed:
                let t = try file.real(e.key)
                let (out, inn) = (t.shape[0], t.shape[1])
                let to = place(e.name, [inn, out]).assumingMemoryBound(to: UInt16.self)
                try file.read(t, into: staging)
                if t.dtype == "BF16" {
                    let from = UnsafePointer(staging.assumingMemoryBound(to: UInt16.self))
                    if out % 8 == 0 && inn % 8 == 0 { transposedHalves(from, out, inn, into: to, rounding: false) } else { transposed(from, out, inn, into: to) { $0 } }
                } else {
                    transposed(UnsafePointer(staging.assumingMemoryBound(to: Float.self)), out, inn, into: to) { BFloat16Bits.rounded($0) }
                }
            case .bfloat16(let joined):
                guard let p = placed[e.name] else { preconditionFailure("no place for \(e.name)") }
                var at = arena.pointer(p.offset)
                for key in [e.key] + joined {
                    at += 2 * (try file.readBFloat16(key, into: at, staging: staging))
                }
                precondition(at == arena.pointer(p.offset) + 2 * p.shape.reduce(1, *), "\(e.name): read into a place of \(p.shape)")
            case .halfTransposed, .floatTransposed:
                let t = try file.real(e.key)
                let (out, inn) = (t.shape[0], t.shape[1])
                let to = place(e.name, [inn, out])
                try file.read(t, into: staging)
                if t.dtype == "F32" {
                    let from = UnsafePointer(staging.assumingMemoryBound(to: Float.self))
                    if e.recipe == .halfTransposed { transposed(from, out, inn, into: to.assumingMemoryBound(to: Float16.self)) { Float16($0) } }
                    else { transposed(from, out, inn, into: to.assumingMemoryBound(to: Float.self)) { $0 } }
                } else {
                    let from = UnsafePointer(staging.assumingMemoryBound(to: UInt16.self))
                    func wide(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }
                    if e.recipe == .halfTransposed {
                        let h = to.assumingMemoryBound(to: UInt16.self)
                        if out % 8 == 0 && inn % 8 == 0 { transposedHalves(from, out, inn, into: h, rounding: true) }
                        else { transposed(from, out, inn, into: h) { Float16(wide($0)).bitPattern } }
                        // f16's largest finite magnitude is 0x7bff: a bf16 past 65 504 rounds to ∞ (or was NaN).
                        guard largestMagnitude(UnsafePointer(h), count: out * inn) < 0x7c00 else {
                            throw Failure("\(e.key): values past f16's range (65 504), which the ANE's f16 cannot hold")
                        }
                    }
                    else { transposed(from, out, inn, into: to.assumingMemoryBound(to: Float.self)) { wide($0) } }
                }
            }
        }
    }

    /// A product's parts (`Recipe.codes`, `Recipe.product`): each part's key and `<key>.weight`, the summed `out`, the
    /// shared `in`, whether int8 codes (else bf16).
    static func product(_ e: Entry, in file: SafeTensors) throws -> (parts: [(key: String, tensor: SafeTensors.Tensor)], out: Int, inn: Int, int8: Bool) {
        let keys: [String]
        switch e.recipe {
        case .codes: keys = [e.key]
        case .product(let joined): keys = [e.key] + joined
        default: preconditionFailure("\(e.name): not a product")
        }
        let parts = try keys.map { k -> (key: String, tensor: SafeTensors.Tensor) in
            guard let t = file.tensors[k + ".weight"] else { throw Failure("\(file.path): no tensor \(k).weight") }
            return (k, t)
        }
        let first = parts[0].tensor, int8 = first.dtype == "I8"
        guard first.shape.count == 2, int8 || first.dtype == "BF16", e.recipe != .codes || int8 else {
            throw Failure("\(e.key).weight: \(first.dtype) \(first.shape), not a product's int8 codes or bf16 weight")
        }
        for (k, t) in parts where t.dtype != first.dtype || t.shape.count != 2 || t.shape[1] != first.shape[1] {
            throw Failure("\(e.name): \(k).weight \(t.dtype) \(t.shape) does not join \(first.dtype) \(first.shape)")
        }
        return (parts, parts.reduce(0) { $0 + $1.tensor.shape[0] }, first.shape[1], int8)
    }

    /// The scales of `key`'s `out` rows of codes: ComfyUI's `<key>.weight_scale` (f32) or SDNQ's `<key>.scale` (bf16),
    /// `[out, 1]` or `[out]`.
    public static func scales(_ key: String, in file: SafeTensors, out: Int) throws -> SafeTensors.Tensor {
        for suffix in [".weight_scale", ".scale"] {
            guard let t = file.tensors[key + suffix] else { continue }
            guard t.dtype == "F32" || t.dtype == "BF16", t.shape.reduce(1, *) == out, t.shape.count <= 2 else {
                throw Failure("\(key)\(suffix): \(t.dtype) \(t.shape) for \(out) rows")
            }
            return t
        }
        throw Failure("\(key): int8 codes without their scales (\(key).weight_scale, \(key).scale)")
    }

    /// `target16` of one product on the CPU, k its inputs: ⌊log₂(65 504 / 128 k)⌋ − m − 1 for max |s| in (2^m, 2^(m+1)]
    /// (2⁻¹⁴ at least, as `peak`), capped at ⌊log₂(65 504 / 2 k)⌋ — the same as the graph's former `target16` on the GPU,
    /// for every block of the model it was measured on (18/11: the GPU's update unchanged).
    public static func target16(scales s: UnsafePointer<Float16>, count: Int, k: Int) -> Double {
        let top = max((0..<count).reduce(0.0) { max($0, abs(Double(s[$1]))) }, 0x1p-14), m = Foundation.log2(top).rounded(.up) - 1
        let k = Double(k)
        return min(Foundation.log2(65504 / (128 * k)).rounded(.down) - m - 1, Foundation.log2(65504 / (2 * k)).rounded(.down))
    }

    /// `target16` of one product of f16 weights `[k, n]` on the CPU: the largest e₀ with k · max|w| · 2^e₀ ≤ 65 504 (the
    /// guard's `target(weights:)`, the product's sums its result), max|w| f16's smallest normal at least.
    public static func target16(weights w: UnsafePointer<Float16>, count: Int, k: Int) -> Double {
        let top = max(Double(Float16(bitPattern: largestMagnitude(UnsafeRawPointer(w).assumingMemoryBound(to: UInt16.self), count: count))), 0x1p-14)
        return Foundation.log2(65504 / (Double(k) * top)).rounded(.down)
    }

    /// The largest of `count` f16 magnitudes, as bits (sign cleared; past 0x7bff, an ∞ or a NaN), on every core.
    static func largestMagnitude(_ h: UnsafePointer<UInt16>, count: Int) -> UInt16 {
        let chunk = 1 << 20, chunks = (count + chunk - 1) / chunk
        nonisolated(unsafe) let p = h
        let tops = UnsafeMutableBufferPointer<UInt16>.allocate(capacity: max(chunks, 1))
        defer { tops.deallocate() }
        tops.initialize(repeating: 0)
        nonisolated(unsafe) let t = tops
        DispatchQueue.concurrentPerform(iterations: chunks) { c in
            let (i0, i1) = (c * chunk, min((c + 1) * chunk, count))
            var top = SIMD8<UInt16>(repeating: 0), i = i0
            while i + 8 <= i1 {
                top = pointwiseMax(top, UnsafeRawPointer(p + i).loadUnaligned(as: SIMD8<UInt16>.self) & 0x7fff)
                i += 8
            }
            var m = top.max()
            while i < i1 { m = max(m, p[i] & 0x7fff); i += 1 }
            t[c] = m
        }
        return tops.max() ?? 0
    }
}
