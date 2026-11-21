import Foundation
import Testing
@testable import SiliCore

// MARK: - Attention and the rotation, byte for byte against coreai-torch's

private func expected(_ name: String) -> [UInt8] { [UInt8](Data(base64Encoded: coreAIAttentionReferences[name]!)!) }

private func differs(_ name: String, _ written: [UInt8]) {
    let want = expected(name)
    let first = (0..<min(want.count, written.count)).first { want[$0] != written[$0] } ?? min(want.count, written.count)
    #expect(written == want, "\(name): \(written.count) bytes against \(want.count), first difference at \(first)")
}

// MARK: The rotation

/// ConvRot's matrix is the Kronecker product of four regular factors of order 4, the last on the fastest digit.
@Test func convrotMatrixIsTheKroneckerProduct() {
    let (order, entries) = Aicode.convrotMatrix()
    #expect(order == 256)
    let f = Aicode.regular
    for r in stride(from: 0, to: 256, by: 7) {
        for c in stride(from: 0, to: 256, by: 5) {
            let want = (0..<4).reduce(Int8(1)) { p, i in
                let shift = 2 * (3 - i)
                return p * f[(r >> shift) & 3][(c >> shift) & 3]
            }
            #expect(entries[r * 256 + c] == want)
        }
    }
    // Orthogonal up to 256: H Hᵀ = 256 I.
    for (a, b) in [(0, 0), (0, 1), (17, 17), (17, 200)] {
        let dot = (0..<256).reduce(0) { $0 + Int(entries[a * 256 + $1]) * Int(entries[b * 256 + $1]) }
        #expect(dot == (a == b ? 256 : 0))
    }
}

// MARK: Attention

/// coreai-torch's random symbol for the composite's graph (`sdpa_maskless_fwbumfvj`), read from the reference.
private func symbol(_ bytes: [UInt8]) -> String {
    let start = (0..<bytes.count - 5).first { bytes[$0..<$0 + 5].elementsEqual(Array("sdpa_".utf8)) }!
    return String(decoding: bytes[start...].prefix { $0 != 0 }, as: UTF8.self)
}

/// torch's default scale for a head of 8.
private let headScale = 1 / 8.0.squareRoot()

private typealias AttentionCase = (name: String, q: [Int], k: [Int], element: Aicode.Element, mask: [Int]?, scale: Double)

/// Bidirectional only: the engine writes no causal mask (a mask is an input).
private let attentionCases: [AttentionCase] = [
    ("sdpa_f16_1_2_5_8", [1, 2, 5, 8], [1, 2, 5, 8], .f16, nil, headScale),
    ("sdpa_bf16_1_2_5_8", [1, 2, 5, 8], [1, 2, 5, 8], .bf16, nil, headScale),
    ("sdpa_scale_f16_1_2_5_8", [1, 2, 5, 8], [1, 2, 5, 8], .f16, nil, 0.25),
    ("sdpa_scale_bf16_1_2_5_8", [1, 2, 5, 8], [1, 2, 5, 8], .bf16, nil, 0.25),
    ("sdpa_f16_2_2_3_5_8", [2, 2, 3, 8], [2, 2, 5, 8], .f16, nil, headScale),
    ("sdpa_mask_f16_1_2_5_8", [1, 2, 5, 8], [1, 2, 5, 8], .f16, [1, 1, 5, 5], headScale),
    ("sdpa_mask_full_f16_1_2_5_8", [1, 2, 5, 8], [1, 2, 5, 8], .f16, [1, 2, 5, 5], headScale),
    ("sdpa_f32_1_2_5_8", [1, 2, 5, 8], [1, 2, 5, 8], .f32, nil, headScale),
    ("sdpa_scale_f32_1_2_5_8", [1, 2, 5, 8], [1, 2, 5, 8], .f32, nil, 0.25),
]

/// The composite, its call and the module around them, written with the block's `attention`.
private func attentionModule(_ c: AttentionCase, symbol: String) -> Aicode.Module {
    let T = Aicode.tensor
    var arguments = [("q", T(c.q, c.element)), ("k", T(c.k, c.element)), ("v", T(c.k, c.element))]
    if let mask = c.mask { arguments.append(("m", T(mask, c.element))) }
    var m = Aicode.Module(arguments: arguments)
    m.outputs = [("scaled_dot_product_attention", m.body.attention(Array(0..<arguments.count), scale: c.scale, symbol: symbol))]
    return m
}

@Test(arguments: attentionCases.map(\.name))
func coreAIAttentionEqualsTheReference(_ name: String) {
    let c = attentionCases.first { $0.name == name }!
    differs(name, attentionModule(c, symbol: symbol(expected(name))).bytecode())
}
