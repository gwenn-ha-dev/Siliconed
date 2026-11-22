import CoreAI
import Foundation

// How a tensor of a file enters the arena: where it is placed (`Placed`), its recipe.

/// A tensor placed in the arena: where it starts, its shape and element as the functions read it.
public struct Placed: Sendable {
    public let offset: Int
    public let shape: [Int]
    public let type: NDArray.ScalarType
    public init(offset: Int, shape: [Int], type: NDArray.ScalarType) { self.offset = offset; self.shape = shape; self.type = type }
}

/// **How a tensor of a file enters the arena.** A client's recipe names, for each argument of its functions, the
/// file's key and one of these; the loader knows no model.
public enum Recipe: Equatable, Sendable {
    /// int8 codes `[out, in]` (`<key>.weight`) laid `[p, out, in / p]`, the file's rows by pieces of k (`Weights.pieces`;
    /// the file's own layout where k is one piece). The file's codes given as a transposed view (strides `[1, in]`) are copied
    /// at every call (277 ms on the M1 Pro; 145 ms against 2.4 on the M5 Pro; `RunTests`:
    /// `aTransposedViewIsCopiedAtEveryCall`): a layout is written into the arena, never viewed. Their scales rounded once
    /// to f16 `[out]`, under the argument's name + `_scale`: ComfyUI's `<key>.weight_scale` (f32 `[out, 1]`) or SDNQ's
    /// `<key>.scale` (bf16 `[out, 1]`, exact in f16 within its normals).
    case codes
    /// **A product's weight, its parts side by side along `out`**: `<key>.weight` then each of `joined`'s, `[out, in]` in
    /// the file, fused into one product of the summed `out` (a transformer's q, k and v; its gate and up). int8 codes
    /// with their scales as `codes` reads them, by pieces of k; bf16 as it lies, `[out, in]`,
    /// as coreai-torch writes `linear`.
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
    /// bf16 `[out, in]` as `[in, out]`, kept bf16 (f32 rounded once): the weight of a product written `x · W` in bf16
    /// (a one-row function's, a client's function in bf16 that reads W `[in, out]`), laid out as that product reads it —
    /// Core AI copies a transposed view at every call (`codes`), so the layout is written once, here.
    case bfloatTransposed
    /// The exponent e₀ the products of these codes (q64 on the ANE, int8 × int8 on the GPU; entries before it, by their
    /// arguments' names) read their one input under, f16 `[1, 1, 1]`: `target16`'s rule (`target16(scales:count:k:)`), computed here from the scales once read, not in
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
