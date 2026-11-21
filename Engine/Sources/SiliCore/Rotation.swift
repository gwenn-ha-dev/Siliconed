/// The rotation ComfyUI's `convrot` codes carry, as a product of Core AI: `W = (q · s) · R` with R the Hadamard by 256 of
/// the regular order 4 (`H₄ ⊗ H₄ ⊗ H₄ ⊗ H₄`, normalized), so `x · Wᵀ = (x · R) · (q · s)ᵀ` (R is symmetric). Recopied
/// from SiliCore (`CoreAIAttention.swift`'s `lowerHadamard`, aec8f3c), its product form along the last axis.
///
///     x [rows, n]  reshape [rows · n/256, 256] · matmul(·, Kᵀ) · reshape [rows, n]        K = H₂₅₆ / 16
///
/// What a port by analogy would miss (SiliCore's findings):
///   - **1/√256 is folded into the matrix**: 256 = 4⁴, so every entry ±2⁻⁴ is exact in f16 and bf16, and no scaling op
///     follows the product;
///   - **the constants are created in the reverse of their order in the block** (the last shape, the matrix, the first
///     shape), and the matrix is a composite's own constant: dense in the IR, shared by every rotation of a module (the
///     writer keeps one attribute for equal entries).

extension Aicode {
    /// The regular Hadamard matrix of order 4 (constant row sums: all ones but −1 on the anti-diagonal), ConvRot's.
    static let regular: [[Int8]] = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]

    /// `F ⊗ … ⊗ F` (`count` factors), row-major, the last factor on the fastest digit.
    static func kronecker(_ f: [[Int8]], _ count: Int) -> (order: Int, entries: [Int8]) {
        var order = 1, entries: [Int8] = [1]
        let n = f.count
        for _ in 0..<count {
            let next = order * n
            var out = [Int8](repeating: 0, count: next * next)
            for i in 0..<order {
                for k in 0..<n {
                    for j in 0..<order {
                        for l in 0..<n { out[(i * n + k) * next + j * n + l] = entries[i * order + j] * f[k][l] }
                    }
                }
            }
            order = next; entries = out
        }
        return (order, entries)
    }

    /// ConvRot's H₂₅₆ (entries ±1, unnormalized): what `convrot` multiplies by, for a reference to compute with.
    public static func convrotMatrix() -> (order: Int, entries: [Int8]) { kronecker(regular, 4) }

    /// A number in the element's bytes, rounded to nearest once from f32 (as torch converts a Python scalar).
    static func roundedScalar(_ x: Double, _ e: Element) -> [UInt8] {
        let f = Float(x)
        switch e {
        case .f16: return withUnsafeBytes(of: Float16(f).bitPattern.littleEndian) { Array($0) }
        case .bf16: return withUnsafeBytes(of: UInt16(f.bitPattern >> 16).littleEndian) { Array($0) }  // ±2⁻⁴: exact
        default: return withUnsafeBytes(of: f.bitPattern.littleEndian) { Array($0) }
        }
    }
}

extension Aicode.Block {
    /// `x` rotated along its last axis by groups of 256: ConvRot's Hadamard, as a product of Core AI.
    public func convrot(_ x: Int) -> Int {
        let (shape, element) = Aicode.shape(types[x])
        let m = 4, g = 1 << (2 * m)  // 4⁴ = 256
        let (order, entries) = Aicode.kronecker(Aicode.regular, m)
        precondition(order == g && shape.last! % g == 0, "convrot: \(shape) is not by groups of \(g)")
        let groups = shape.last! / g
        let dims = groups > 1 ? [shape.reduce(1, *) / g, g] : shape

        // K transposed (y = x · Kᵀ), entries ±2⁻ᵐ.
        let one = Aicode.roundedScalar(Double(sign: .plus, exponent: -m, significand: 1), element)
        let minus = Aicode.roundedScalar(-Double(sign: .plus, exponent: -m, significand: 1), element)
        let bytes = (0..<g * g).flatMap { i in entries[(i % g) * g + i / g] > 0 ? one : minus }
        func indices(_ v: [Int]) -> Int { constant(Aicode.dense(v.map { UInt32($0) }), Aicode.tensor([v.count], .ui32)) }
        let shapeOut = groups > 1 ? indices(shape) : nil
        let matrix = constant(Aicode.dense(bytes, splat: false), Aicode.tensor([g, g], element))
        let shapeIn = groups > 1 ? indices(dims) : nil

        var v = x
        if let shapeIn { v = add(Aicode.Op("primitive.reshape_v2", operands: [v, shapeIn], results: [Aicode.tensor(dims, element)]))[0] }
        precondition(dims.count == 2, "convrot: a rank-2 product only")
        v = matmul(v, matrix, result: Aicode.tensor(dims, element))
        if let shapeOut { v = add(Aicode.Op("primitive.reshape_v2", operands: [v, shapeOut], results: [Aicode.tensor(shape, element)]))[0] }
        return v
    }
}
