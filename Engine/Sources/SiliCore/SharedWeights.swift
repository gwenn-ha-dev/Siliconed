import CoreAI
import Foundation
import Metal

/// **What the shared forms' products read**: ConvRot's int8 codes and their scales, the input rotated by groups of 256
/// (the Compact: `[in, out]` for `q64`, `codes`; `[out, in]` as the file lays them for `w8a8` and the ANE's `q64` beside
/// it, `integer`), or f16 weights `[in, out]` (the Standard beside the ANE: the publisher's bf16 rounded once to f16 as
/// they are read, `Recipe.halfTransposed`; the input as it is, `product16`). The ANE takes no bf16 at its
/// borders, so the Standard's rows there are f16 under the guard, the GPU's beside them the
/// same functions on the same bytes: one copy of the weights. **bf16 → f16 is exact** within f16's normals (8 bits of
/// significand into 11); measured 20/11/2026 on five published transformers (2 to 12.5 billion weights each): no weight
/// past 65 504 (the largest 123), 0.03 to 0.72 % under 2⁻¹⁴, each rounded to f16's subnormals within 2⁻²⁵. The ANE
/// flushes those: at most 5 · 10⁻⁶ of a product's energy (the product with the most under 2⁻¹⁴: 3.7 % of its weights).
///
/// **`integer`: the Compact's codes as the file lays them, `[1, out, in]`**, for a GPU whose tensor units multiply int8
/// into int32 (`offered`): its products int8 × int8 (`w8a8`), the ANE's `q64` on the same codes transposed in the
/// function. Measured 21/11/2026 on the M5 Pro, a product of 4128 rows: 4128 × 3840 → 10 240, 44.4 TFLOP/s against q64's
/// 24.7; 4128 × 10 240 → 3840, 39.6 against 24.3; the ANE's q64 on the codes transposed 15.8 against 17.8 as they lie
/// `[in, out]` (1024 rows × 3840 → 10 240). The codes in `[in, out]`, the same W8A8 runs at q64's rate: Core AI's GPU
/// takes the integer product NT only.
public enum SharedWeights: Sendable, Hashable {
    case codes, integer, half

    /// The weights are the Compact's int8 codes (`codes` or `integer`).
    public var isCodes: Bool { self != .half }

    /// How the plan lays a product's codes (`Block.Declaration.codes`).
    public var layout: CodesLayout { self == .integer ? .outIn : .inOut }

    /// **The weights this Mac's GPU is offered for the Compact's codes**, the first tried first: `integer` where its
    /// product int8 × int8 was measured faster than `codes`' (`integerTrial`, kept by Mac and system:
    /// `ThisMac.integerProducts`), then `codes`; `codes` alone elsewhere. Which of the two a block runs is then timed on
    /// this Mac (`Block.choose`). No chip is named: Core AI's int8 kernel is fast where the GPU's tensor units multiply
    /// int8 (the M5 Pro's, 21/11/2026), and may be on a GPU not known today, or be slow on one (MPS's fallback ran at
    /// 0.3 TFLOP/s against 49, scales in f32); a block offered a slow product would spend a whole run finding out.
    public static func offered(_ device: MTLDevice) async throws -> [SharedWeights] {
        let ratio: Double
        if let kept = ThisMac.current.integerProducts { ratio = kept } else {
            // A trial that fails (Core AI refusing the product on a GPU of its own) is kept as a product never faster,
            // so that the Compact renders on `codes` and the trial is not made again under this system. Kept as the
            // largest finite value: `JSONEncoder` refuses an infinite one, and a field of its own would say no more.
            ratio = (try? await integerTrial(device)) ?? .greatestFiniteMagnitude
            try ThisMac.update { $0.integerProducts = ratio }
        }
        return ratio < 1 ? [.integer, .codes] : [.codes]
    }

    /// **The trial**: one product of the shared forms, 2048 × 3840 → 3840 in f16 on the GPU, as `integer` (`w8a8`) and as
    /// `codes` (`q64`), each called 2 times cold then 9 timed; the ratio of their medians, `integer`'s over `codes'`.
    /// One square product: the same bytes serve both layouts, and what is timed is the kernel, not the values. The
    /// shape: a call long enough on every GPU that its launch does not weigh (on the M5 Pro, about 1.3 ms against 2.5),
    /// and an arena of 46 MiB taken and given back before any stage opens — **before the two functions**: a heap let go
    /// after the Core AI functions that read it stays in the process 12 to 36 s, let go before them 0.06 s (measured
    /// 14–15/11/2026 on the M1 Pro). The functions are loaded here and outlive `timed`, which holds the arena and its views.
    static func integerTrial(_ device: MTLDevice) async throws -> Double {
        let (m, k) = (2048, 3840)
        var functions: [InferenceFunction] = []
        for integer in [true, false] {
            var mod = Aicode.Module(arguments: [("x", Aicode.tensor([1, m, k], .f16)), ("w", Aicode.tensor([1, k, k], .si8)),
                                                ("w_scale", Aicode.tensor([1, 1, k], .f16))])
            var c = Aicode.ScalingConstants()
            let b = mod.body, (xa, wa, sa) = (mod.argument("x"), mod.argument("w"), mod.argument("w_scale"))
            mod.outputs = [("y", integer ? b.w8a8(xa, codes: wa, scale: sa, constants: &c) : b.q64(xa, codes: wa, scale: sa, constants: &c))]
            functions.append(try await Functions.load(try Functions.asset(mod, "integer-trial-\(integer ? "w8a8" : "q64")-\(m)x\(k)x\(k)")))
        }
        let medians = try await timed(device, functions, m: m, k: k)
        withExtendedLifetime(functions) {}
        return medians[0] / medians[1]
    }

    /// The trial's calls of each of `functions` on an arena of its own, the median of each: the arena, its views, the
    /// queue and the stream all let go as this returns, the functions still held by the caller.
    private static func timed(_ device: MTLDevice, _ functions: [InferenceFunction], m: Int, k: Int) async throws -> [Double] {
        guard let queue = device.makeCommandQueue() else { throw Failure("no command queue") }
        let stream = ComputeStream(commandQueue: queue)
        let arena = try Arena(device, bytes: 2 * Arena.aligned(m * k * 2) + Arena.aligned(k * k) + Arena.aligned(k * 2))
        let (xk, wk, sk, yk) = (arena.take(m * k * 2), arena.take(k * k), arena.take(k * 2), arena.take(m * k * 2))
        // x = 1, the codes 1, their scales 1/1024: every sum 3840, nothing out of f16's range.
        let x = arena.pointer(xk).bindMemory(to: UInt16.self, capacity: m * k), s = arena.pointer(sk).bindMemory(to: UInt16.self, capacity: k)
        for i in 0..<m * k { x[i] = Float16(1).bitPattern }
        for i in 0..<k { s[i] = Float16(1.0 / 1024).bitPattern }
        memset(arena.pointer(wk), 1, k * k)
        func view(_ key: Int, _ type: NDArray.ScalarType, _ shape: [Int]) -> View { View(buffer: arena.buffer(key), type: type, shape: shape) }
        let inputs = ["x": view(xk, .float16, [1, m, k]), "w": view(wk, .int8, [1, k, k]), "w_scale": view(sk, .float16, [1, 1, k])]
        let y = view(yk, .float16, [1, m, k])
        var medians: [Double] = []
        for f in functions {
            var times: [Double] = []
            for _ in 0..<11 {
                let t = DispatchTime.now().uptimeNanoseconds
                try await Run.call(f, stream, inputs: inputs, outputs: [("y", y)])
                times.append(Functions.seconds(since: t))
            }
            medians.append(times.dropFirst(2).sorted()[4])
        }
        return medians
    }

    /// A product's arguments: the codes `[1, in, out]` (`[1, out, in]` for `integer`) and their scales `[1, 1, out]`,
    /// or the f16 weight `[1, in, out]`.
    public func arguments(_ name: String, inn: Int, out: Int) -> [(name: String, shape: [Int], element: Aicode.Element)] {
        switch self {
        case .codes: [(name, [1, inn, out], .si8), (name + "_scale", [1, 1, out], .f16)]
        case .integer: [(name, [1, out, inn], .si8), (name + "_scale", [1, 1, out], .f16)]
        case .half: [(name, [1, inn, out], .f16)]
        }
    }

    /// A product's entry from its file's key (`<key>.weight`, `[out, in]`): its codes and scales (`Recipe.codes`, laid
    /// by the plan's `layout`), or the weight rounded to f16 `[in, out]` (`Recipe.halfTransposed`).
    public func entry(_ name: String, key: String) -> Weights.Entry {
        isCodes ? (name, key, .codes) : (name, key + ".weight", .halfTransposed)
    }
}
