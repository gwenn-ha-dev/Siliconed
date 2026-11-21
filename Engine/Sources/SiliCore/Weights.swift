import CoreAI
import Foundation

/// A tensor placed in the arena: where it starts, its shape and element as the functions read it.
public struct Placed: Sendable {
    public let offset: Int
    public let shape: [Int]
    public let type: NDArray.ScalarType
    public init(offset: Int, shape: [Int], type: NDArray.ScalarType) { self.offset = offset; self.shape = shape; self.type = type }
}

/// **How a tensor of a file enters the arena.** A family's recipe names, for each argument of its functions, the
/// file's key and one of these; the loader knows no model.
public enum Recipe: Equatable, Sendable {
    /// int8 codes `[out, in]` (`<key>.weight`) in the stage's codes layout (`CodesLayout`, `inOut`
    /// wherever a stage is planned today), given to `Weights` by the stage's plan. The file's codes given as a transposed view (strides `[1, in]`) are copied
    /// at every call (277 ms on the M1 Pro; 145 ms against 2.4 on the M5 Pro; `RunTests`:
    /// `aTransposedViewIsCopiedAtEveryCall`): a layout is written into the arena, never viewed. Their scales rounded once
    /// to f16 `[out]`, under the argument's name + `_scale`: ComfyUI's `<key>.weight_scale` (f32 `[out, 1]`) or SDNQ's
    /// `<key>.scale` (bf16 `[out, 1]`, exact in f16 within its normals).
    case codes
    /// **A product's weight, its parts side by side along `out`**: `<key>.weight` then each of `joined`'s, `[out, in]` in
    /// the file, fused into one product of the summed `out` (a transformer's q, k and v; its gate and up). int8 codes
    /// with their scales as `codes` reads them, in the plan's codes layout; bf16 as it lies, in the plan's bf16 layout
    /// (`CodesLayout`).
    case product(joined: [String])
    /// f32 or bf16, rounded once to f16, its shape kept (a norm's gain, a bias the f16 path adds, a convolution's kernel).
    case half
    /// The same at the last index of `axis`, that axis dropped: a causal kernel `[O, I, T, kh, kw]` read on one step of
    /// its third axis sees only its last slice (`axis` 2: `[O, I, kh, kw]`).
    case halfLast(axis: Int)
    /// f32 or bf16 `[out, in]` rounded once to f16 as `[in, out]`: a float weight a product reads — the Standard's
    /// products beside the ANE (`SharedWeights.half`). bf16 is transposed and rounded in one pass on every core
    /// (`transposedHalves(_:_:_:into:rounding:)`); a value past f16's range is refused, not made infinite.
    case halfTransposed
    /// f32 as it is, or bf16 widened (exact): a bias added in f32, a padding row, what a one-row function computes in f32.
    case float
    /// f32 `[out, in]` as `[in, out]`, in f32.
    case floatTransposed
    /// bf16 `[out, in]` as `[in, out]`, kept bf16 (f32 rounded once): the bf16 path's weights, laid out as the codes are,
    /// what Core AI's product reads as it lies.
    case bfloatTransposed
    /// The exponent e₀ the q64 products of these codes (entries before it, by their arguments' names) read their one
    /// input under, f16 `[1, 1, 1]`: `target16`'s rule (`target16(scales:count:k:)`), computed here from the scales once read, not in
    /// the function's graph; of f16 weights (`SharedWeights.half`, no scales), from their largest magnitude
    /// (`target16(weights:count:k:)`). Measured 18/11/2026 on the M5 Pro: in a transformer's shared feed-forward function, the ANE gave
    /// `target16` −65 504 where the GPU gave 2 for the layers whose largest w₁ scale lies just under 2⁻⁷ (1, 2, 7) — the
    /// input of w₁ crushed under f16's subnormals, the update 0.24 from the CPU's —, while the same `target16` alone, of the
    /// same scales, was right on both units; given as an argument, the ANE's function is within the GPU's.
    case target([String])
    /// The bf16 path's: bf16 as it lies (an f32 rounded to nearest even once, `SafeTensors.readBFloat16`), its shape kept;
    /// the tensors of `joined` after `key`'s, side by side along the first axis — a fused product's parts `[out, in]`
    /// laid one after the other as the file lays each: one read after the other, no staging.
    case bfloat16(joined: [String])
}

/// **How a product's weight lies in the arena**, its int8 codes (`Recipe.codes`, `Recipe.product`) or its bf16 values
/// (`Recipe.product`), each element its own choice. One arithmetic, two writings, and which one is faster is the Mac's:
/// measured 17/11 on the M1 Pro, 1024 × 3840 × 10240, `matmul(x, dequantize(q))` over `[in, out]` at 24.5 ms (1.07 ×
/// the same product with an f16 weight), `matmul(x, transpose(dequantize(q)))` over the file's `[out, in]` at 159 ms;
/// on the M5 Pro (18/11, 512 rows) the file's layout the faster, 1.92 ms against 2.41. No Mac measures it: the codes
/// are `inOut`, the layout whose loss is the smaller where it is the slower; bf16 `outIn`, the file's, as coreai-torch
/// writes `linear`.
/// The ANE's products (`q64`) read `[in, out]` whatever the GPU prefers: a stage whose weights are surfaces takes it.
public enum CodesLayout: String, Codable, Sendable {
    /// `[in, out]`: the codes transposed once as they are read, what Core AI's product reads as it lies.
    case inOut = "in-out"
    /// `[out, in]`: the file's codes as they lie, transposed inside the function.
    case outIn = "out-in"
}

public enum Weights {
    public typealias Entry = (name: String, key: String, recipe: Recipe)
    public typealias Slot = (name: String, shape: [Int], type: NDArray.ScalarType, bytes: Int)

    /// The tensors `entries` become in the arena, from the file's header alone: each argument's name, its shape and
    /// element as the functions read it, its bytes.
    public static func layout(_ entries: [Entry], in file: SafeTensors, codes: CodesLayout = .inOut,
                              bfloat16: CodesLayout = .outIn) throws -> [Slot] {
        try entries.flatMap { e -> [Slot] in
            switch e.recipe {
            case .codes, .product:
                let (_, out, inn, int8) = try product(e, in: file)
                let shape = (int8 ? codes : bfloat16) == .inOut ? [inn, out] : [out, inn]
                return int8 ? [(e.name, shape, .int8, out * inn), (e.name + "_scale", [out], .float16, out * 2)] : [(e.name, shape, .bfloat16, 2 * out * inn)]
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
    public static func size(_ entries: [Entry], in file: SafeTensors, codes: CodesLayout = .inOut, bfloat16: CodesLayout = .outIn) throws -> Int {
        try layout(entries, in: file, codes: codes, bfloat16: bfloat16).reduce(0) { $0 + Arena.aligned($1.bytes) }
    }

    /// What reading `entries` passes through before its recipe: the largest tensor converted (an f32 read as it is goes
    /// straight to its place).
    public static func staging(_ entries: [Entry], in file: SafeTensors, codes: CodesLayout = .inOut, bfloat16: CodesLayout = .outIn) throws -> Int {
        try entries.map { e in
            switch e.recipe {
            case .codes, .product:
                let (parts, _, _, int8) = try product(e, in: file)
                // A transposed part through the staging; the scales through it, whatever the layout.
                let transposed = (int8 ? codes : bfloat16) == .inOut
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
                            staging: UnsafeMutableRawPointer, codes layout: CodesLayout = .inOut,
                            bfloat16: CodesLayout = .outIn) throws {
        func place(_ name: String, _ shape: [Int]) -> UnsafeMutableRawPointer {
            guard let p = placed[name] else { preconditionFailure("no place for \(name)") }
            precondition(p.shape == shape, "\(name): \(shape) into a place of \(p.shape)")
            return arena.pointer(p.offset)
        }
        for e in entries {
            switch e.recipe {
            case .codes, .product:
                let (parts, total, inn, int8) = try product(e, in: file)
                let lies = (int8 ? layout : bfloat16) == .outIn
                let to = place(e.name, lies ? [total, inn] : [inn, total])
                var column = 0
                for (key, t) in parts {
                    let out = t.shape[0], size = int8 ? 1 : 2
                    if lies {
                        // The file's rows as they lie, each part's after the one before.
                        try file.read(t, into: to + column * inn * size)
                    } else {
                        try file.read(t, into: staging)
                        if int8 {
                            // On every core, by NEON tiles where the shapes are multiples of 16 (`transposedTiles`, the
                            // encoder's: 3.53 GB in 0.19 s, the same to the bit as the byte loop): 6 GB of codes took ~10 s
                            // on one core at 0.7 GB/s.
                            let from = UnsafePointer(staging.assumingMemoryBound(to: Int8.self)), codes = to.assumingMemoryBound(to: Int8.self) + column
                            if out % 16 == 0 && inn % 16 == 0 { transposedTiles(from, out, inn, into: codes, rowLength: total) }
                            else { transposedCodes(from, out, inn, into: codes, rowLength: total) }
                        } else {
                            transposedHalves(UnsafePointer(staging.assumingMemoryBound(to: UInt16.self)), out, inn,
                                             into: to.assumingMemoryBound(to: UInt16.self) + column, rowLength: total)
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
                    return target16(scales: s, count: scales.shape.reduce(1, *), k: w.shape[layout == .inOut ? 0 : 1])
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

    /// Each entry read from `file` into `arena`, by its argument's name: placed, then read through a staging of its own,
    /// given back after — in `Pages`, not `malloc`: freed, a `malloc` block stays in the footprint on macOS 27 (measured
    /// 18/11: a transformer's blocks read through a 39 MB staging, 47 to 80 MB of « Malloc Large » left after the stage
    /// let go, no live allocation in it).
    public static func load(_ entries: [Entry], from file: SafeTensors, into arena: Arena, surfaces: Bool = false,
                            codes: CodesLayout = .inOut, bfloat16: CodesLayout = .outIn) throws -> [String: Placed] {
        let placed = place(try layout(entries, in: file, codes: codes, bfloat16: bfloat16), in: arena, surfaces: surfaces)
        let staging = Pages(bytes: try Self.staging(entries, in: file, codes: codes, bfloat16: bfloat16))
        _ = try withExtendedLifetime(staging) {
            try read(entries, from: file, into: placed, arena: arena, staging: staging.base, codes: codes, bfloat16: bfloat16)
        }
        return placed
    }

    /// `count` bf16 widened to f32 (exact), each through `convert` (to f16: rounded to nearest once).
    static func bf16<B>(_ from: UnsafeRawPointer, _ count: Int, into to: UnsafeMutablePointer<B>, _ convert: (Float) -> B) {
        let b = from.assumingMemoryBound(to: UInt16.self)
        for i in 0..<count { to[i] = convert(Float(bitPattern: UInt32(b[i]) << 16)) }
    }

    /// `[out, in]` 16-bit values into `[in, out]`, **the rows of the result shared among the cores** by blocks of 64, each
    /// block by tiles of 8 × 8 in NEON registers (`transposedTiles`' way, at 16 bits): the bf16 path's weights, 14 GB of a
    /// transformer, which one core's 64 × 64 loop takes tens of seconds over. `out` and `inn` multiples of 8.
    /// `rounding`: each bf16 rounded to nearest in f16 as its tile is stored (`halves`), in the same pass.
    public static func transposedHalves(_ from: UnsafePointer<UInt16>, _ out: Int, _ inn: Int, into to: UnsafeMutablePointer<UInt16>,
                                        rounding: Bool) {
        precondition(out % 8 == 0 && inn % 8 == 0, "[\(out), \(inn)]: tiles of 8")
        nonisolated(unsafe) let (f, t) = (from, to)
        DispatchQueue.concurrentPerform(iterations: (inn + 63) / 64) { block in
            let i0 = block * 64, i1 = min(i0 + 64, inn)
            withUnsafeTemporaryAllocation(of: SIMD8<UInt16>.self, capacity: 16) { buffer in
                let base = buffer.baseAddress!
                for o0 in stride(from: 0, to: out, by: 8) {
                    for i in stride(from: i0, to: i1, by: 8) {
                        var a = base, b = base + 8
                        for m in 0..<8 { a[m] = UnsafeRawPointer(f + (o0 + m) * inn + i).loadUnaligned(as: SIMD8<UInt16>.self) }
                        // Three rounds of even/odd unzips swap row and column in an 8 × 8 tile (an index's six bits
                        // rotate by one each).
                        for _ in 0..<3 {
                            for m in 0..<4 {
                                let w = SIMD16(lowHalf: a[2 * m], highHalf: a[2 * m + 1])
                                b[m] = w.evenHalf; b[m + 4] = w.oddHalf
                            }
                            swap(&a, &b)
                        }
                        for m in 0..<8 { UnsafeMutableRawPointer(t + (i + m) * out + o0).storeBytes(of: rounding ? halves(a[m]) : a[m], as: SIMD8<UInt16>.self) }
                    }
                }
            }
        }
    }

    /// Eight bf16 rounded to nearest in f16 (widened to f32, exact, then narrowed: one `fcvtn`), as bits.
    @inline(__always) static func halves(_ b: SIMD8<UInt16>) -> SIMD8<UInt16> {
        let wide = unsafeBitCast(SIMD8<UInt32>(truncatingIfNeeded: b) &<< 16, to: SIMD8<Float>.self)
        return unsafeBitCast(SIMD8<Float16>(wide), to: SIMD8<UInt16>.self)
    }

    /// `count` f32 rounded to nearest in f16.
    static func half(_ from: UnsafeRawPointer, _ count: Int, into to: UnsafeMutableRawPointer) {
        let f = from.assumingMemoryBound(to: Float.self), h = to.bindMemory(to: Float16.self, capacity: count)
        for i in 0..<count { h[i] = Float16(f[i]) }
    }

    /// The same by tiles of 16 × 16 in NEON registers: 16 rows loaded, four unzips of the 256 bytes (each takes the even
    /// elements then the odd: an index's bits rotate by one, four rotations swap row and column), 16 rows stored; the
    /// rows of the result shared among the cores by blocks of 64, as `transposedCodes`. `out` and `inn` multiples of 16.
    public static func transposedTiles(_ from: UnsafePointer<Int8>, _ out: Int, _ inn: Int, into to: UnsafeMutablePointer<Int8>, rowLength: Int? = nil) {
        precondition(out % 16 == 0 && inn % 16 == 0, "[\(out), \(inn)]: tiles of 16")
        nonisolated(unsafe) let (f, t) = (UnsafeRawPointer(from), UnsafeMutableRawPointer(to))
        let ld = rowLength ?? out
        DispatchQueue.concurrentPerform(iterations: (inn + 63) / 64) { block in
            let i0 = block * 64, i1 = min(i0 + 64, inn)
            withUnsafeTemporaryAllocation(of: SIMD16<UInt8>.self, capacity: 32) { buffer in
                let base = buffer.baseAddress!
                for o0 in stride(from: 0, to: out, by: 16) {
                    for i in stride(from: i0, to: i1, by: 16) {
                        var a = base, b = base + 16
                        for m in 0..<16 { a[m] = (f + (o0 + m) * inn + i).loadUnaligned(as: SIMD16<UInt8>.self) }
                        for _ in 0..<4 {
                            for m in 0..<8 {
                                let w = SIMD32(lowHalf: a[2 * m], highHalf: a[2 * m + 1])
                                b[m] = w.evenHalf; b[m + 8] = w.oddHalf
                            }
                            swap(&a, &b)
                        }
                        for m in 0..<16 { (t + (i + m) * ld + o0).storeBytes(of: a[m], as: SIMD16<UInt8>.self) }
                    }
                }
            }
        }
    }

    /// int8 codes `[out, in]` into `[in, out]`, **the rows of the result shared among the cores** by blocks of 64, each
    /// block by tiles of 64 × 64. By rows and not by columns: each core writes pages of its own, which matters on an
    /// arena's fresh pages (each first write a fault, and faults on one page wait on each other).
    /// `rowLength`: the result's row length, when it lies beside others in a wider one (the products a function fuses).
    static func transposedCodes(_ from: UnsafePointer<Int8>, _ out: Int, _ inn: Int, into to: UnsafeMutablePointer<Int8>, rowLength: Int? = nil) {
        nonisolated(unsafe) let (f, t) = (from, to)
        let ld = rowLength ?? out
        DispatchQueue.concurrentPerform(iterations: (inn + 63) / 64) { block in
            let i0 = block * 64, i1 = min(i0 + 64, inn)
            for o0 in stride(from: 0, to: out, by: 64) {
                for o in o0..<min(o0 + 64, out) {
                    let row = f + o * inn
                    for i in i0..<i1 { t[i * ld + o] = row[i] }
                }
            }
        }
    }

    /// bf16 `[out, in]` into `[in, out]`, as `transposedCodes` (two bytes an element, the rows of the result shared among
    /// the cores). `rowLength`: the result's row, when it lies beside others in a wider one.
    static func transposedHalves(_ from: UnsafePointer<UInt16>, _ out: Int, _ inn: Int, into to: UnsafeMutablePointer<UInt16>, rowLength: Int? = nil) {
        nonisolated(unsafe) let (f, t) = (from, to)
        let ld = rowLength ?? out
        DispatchQueue.concurrentPerform(iterations: (inn + 63) / 64) { block in
            let i0 = block * 64, i1 = min(i0 + 64, inn)
            for o0 in stride(from: 0, to: out, by: 64) {
                for o in o0..<min(o0 + 64, out) {
                    let row = f + o * inn
                    for i in i0..<i1 { t[i * ld + o] = row[i] }
                }
            }
        }
    }

    /// `[out, in]` into `[in, out]`, by tiles of 64 × 64 (what the caches hold), each element through `convert`.
    static func transposed<A, B>(_ from: UnsafePointer<A>, _ out: Int, _ inn: Int, into to: UnsafeMutablePointer<B>, _ convert: (A) -> B) {
        for o0 in stride(from: 0, to: out, by: 64) {
            for i0 in stride(from: 0, to: inn, by: 64) {
                for o in o0..<min(o0 + 64, out) { for i in i0..<min(i0 + 64, inn) { to[i * out + o] = convert(from[o * inn + i]) } }
            }
        }
    }
}

/// **The rows a call covers** (`Block.calls`): its first row and how many, each height a function's fixed shape.
public enum Tiles {
    public struct Tile: Equatable, Sendable {
        public let start: Int
        public let rows: Int
        public init(start: Int, rows: Int) { self.start = start; self.rows = rows }
    }
}
