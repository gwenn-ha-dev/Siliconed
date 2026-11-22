import CoreAI
import Foundation

// How a tensor of a file enters the arena: where it is placed (`Placed`), its recipe, the codes' layout.

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
    /// int8 codes `[out, in]` (`<key>.weight`) in the stage's codes layout (`CodesLayout`: `inOut` for q64, `outIn` as the
    /// file lays them for the products int8 × int8, `SharedWeights.layout`), given to `Weights` by the stage's plan. The
    /// file's codes given as a transposed view (strides `[1, in]`) are copied
    /// at every call (277 ms on the M1 Pro; 145 ms against 2.4 on the M5 Pro; `RunTests`:
    /// `aTransposedViewIsCopiedAtEveryCall`): a layout is written into the arena, never viewed. Their scales rounded once
    /// to f16 `[out]`, under the argument's name + `_scale`: ComfyUI's `<key>.weight_scale` (f32 `[out, 1]`) or SDNQ's
    /// `<key>.scale` (bf16 `[out, 1]`, exact in f16 within its normals).
    case codes
    /// **A product's weight, its parts side by side along `out`**: `<key>.weight` then each of `joined`'s, `[out, in]` in
    /// the file, fused into one product of the summed `out` (a transformer's q, k and v; its gate and up). int8 codes
    /// with their scales as `codes` reads them, in the plan's codes layout (`CodesLayout`); bf16 as it lies, `[out, in]`,
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

/// **How a product's int8 codes lie in the arena** (`Recipe.codes`, `Recipe.product`; its bf16 values lie as the file
/// lays them, `[out, in]`). One arithmetic, two writings, and which one is faster is the Mac's:
/// measured 17/11 on the M1 Pro, 1024 × 3840 × 10240, `matmul(x, dequantize(q))` over `[in, out]` at 24.5 ms (1.07 ×
/// the same product with an f16 weight), `matmul(x, transpose(dequantize(q)))` over the file's `[out, in]` at 159 ms;
/// on the M5 Pro (18/11, 512 rows) the file's layout the faster, 1.92 ms against 2.41. No Mac measures it: the
/// declaration says (`Block.Declaration.codes`, `SharedWeights.layout`) — `inOut` for q64, the layout whose loss is the
/// smaller where it is the slower; `outIn` for the products int8 × int8, which Core AI's GPU takes NT only.
/// The ANE's products (`q64`) read `[in, out]` as they lie, `[out, in]` transposed in the function (`SharedWeights.integer`,
/// 21/11/2026 on the M5 Pro: 15.8 TFLOP/s against 17.8): the declaration says which (`Block.Declaration.codes`).
public enum CodesLayout: String, Codable, Sendable {
    /// `[in, out]`: the codes transposed once as they are read, what Core AI's product reads as it lies.
    case inOut = "in-out"
    /// `[out, in]`: the file's codes as they lie, transposed inside the function.
    case outIn = "out-in"
}
