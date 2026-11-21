import Foundation
import Testing
@testable import SiliCore

// MARK: - Convolutions, byte for byte against coreai-torch's (`tools/coreai/references_convolution.py`)

private struct ConvolutionCase: Sendable {
    var x: [Int], w: [Int], element: Aicode.Element = .f16
    var stride = [1], padding = [0], dilation = [1], groups = 1
}

/// The cases of the script, by name (the shapes and arguments it passes to `F.conv*`).
private let convolutionCases: [String: ConvolutionCase] = [
    "conv2d_f16": .init(x: [1, 4, 5, 6], w: [8, 4, 3, 3]),
    "conv2d_pad1": .init(x: [1, 4, 5, 6], w: [8, 4, 3, 3], padding: [1]),
    "conv2d_pad_hw": .init(x: [1, 4, 5, 6], w: [8, 4, 3, 3], padding: [1, 2]),
    "conv2d_stride2_pad1": .init(x: [1, 4, 5, 6], w: [8, 4, 3, 3], stride: [2], padding: [1]),
    "conv2d_stride_hw": .init(x: [1, 4, 5, 6], w: [8, 4, 3, 3], stride: [1, 2]),
    "conv2d_dilation2": .init(x: [1, 4, 7, 8], w: [8, 4, 3, 3], dilation: [2]),
    "conv2d_dilation_pad": .init(x: [1, 4, 7, 8], w: [8, 4, 3, 3], padding: [2], dilation: [2]),
    "conv2d_groups2": .init(x: [1, 4, 5, 6], w: [8, 2, 3, 3], padding: [1], groups: 2),
    "conv2d_depthwise": .init(x: [1, 4, 5, 6], w: [4, 1, 3, 3], padding: [1], groups: 4),
    "conv2d_1x1": .init(x: [1, 4, 5, 6], w: [8, 4, 1, 1]),
    "conv2d_batch2": .init(x: [2, 4, 5, 6], w: [8, 4, 3, 3], padding: [1]),
    "conv2d_f32": .init(x: [1, 4, 5, 6], w: [8, 4, 3, 3], element: .f32, padding: [1]),
    "conv2d_bf16": .init(x: [1, 4, 5, 6], w: [8, 4, 3, 3], element: .bf16, padding: [1]),
    "conv2d_pad1_f16_big": .init(x: [1, 16, 32, 32], w: [16, 16, 3, 3], padding: [1]),
    "conv1d_f16": .init(x: [1, 4, 9], w: [8, 4, 3]),
    "conv1d_pad1": .init(x: [1, 4, 9], w: [8, 4, 3], padding: [1]),
    "conv1d_stride2_pad1": .init(x: [1, 4, 9], w: [8, 4, 3], stride: [2], padding: [1]),
    "conv1d_groups2_dil2": .init(x: [1, 4, 9], w: [8, 2, 3], padding: [2], dilation: [2], groups: 2),
    "conv3d_f16": .init(x: [1, 4, 4, 5, 6], w: [8, 4, 3, 3, 3]),
    "conv3d_pad1": .init(x: [1, 4, 4, 5, 6], w: [8, 4, 3, 3, 3], padding: [1]),
    "conv3d_stride2_pad1": .init(x: [1, 4, 4, 5, 6], w: [8, 4, 3, 3, 3], stride: [2], padding: [1]),
    "conv3d_groups2_dil2": .init(x: [1, 4, 6, 7, 8], w: [8, 2, 3, 3, 3], padding: [2], dilation: [2], groups: 2),
    "conv2d_stride2": .init(x: [1, 4, 7, 8], w: [8, 4, 3, 3], stride: [2]),
    "conv2d_dilation2_nopad": .init(x: [1, 4, 7, 8], w: [8, 4, 3, 3], dilation: [2]),
    "conv2d_groups2_nopad": .init(x: [1, 4, 5, 6], w: [8, 2, 3, 3], groups: 2),
    "conv2d_all_nopad": .init(x: [1, 4, 9, 10], w: [8, 2, 3, 3], stride: [2, 1], dilation: [1, 2], groups: 2),
    "conv1d_stride2": .init(x: [1, 4, 9], w: [8, 4, 3], stride: [2]),
    "conv1d_dil2_groups2": .init(x: [1, 4, 9], w: [8, 2, 3], dilation: [2], groups: 2),
    "conv3d_stride2": .init(x: [1, 4, 6, 7, 8], w: [8, 4, 3, 3, 3], stride: [2]),
    "conv3d_all_nopad": .init(x: [1, 4, 6, 7, 8], w: [8, 2, 3, 3, 3], stride: [2, 1, 1], dilation: [1, 2, 1], groups: 2),
    "conv2d_pad_asym_stride": .init(x: [1, 4, 9, 10], w: [8, 2, 3, 3], stride: [2, 1], padding: [1, 0], dilation: [1, 2], groups: 2),
]

private func expected(_ name: String) -> [UInt8] { [UInt8](Data(base64Encoded: coreAIConvolutionReferences[name]!)!) }

@Test(arguments: coreAIConvolutionReferences.keys.sorted())
func coreAIConvolutionBytecodeEqualsTheReference(_ name: String) throws {
    let c = convolutionCases[name]!
    // The script's arguments spread over the axes as torch does; the result's shape as torch computes it.
    let spatial = c.x.count - 2
    func spread(_ a: [Int]) -> [Int] { a.count == 1 ? Array(repeating: a[0], count: spatial) : a }
    let (s, p, d) = (spread(c.stride), spread(c.padding), spread(c.dilation))
    let out = [c.x[0], c.w[0]] + (0..<spatial).map { i in (c.x[2 + i] + 2 * p[i] - d[i] * (c.w[2 + i] - 1) - 1) / s[i] + 1 }
    var m = Aicode.Module(arguments: [("x", Aicode.tensor(c.x, c.element)), ("w", Aicode.tensor(c.w, c.element))])
    m.outputs = [("convolution", m.body.convolution(0, 1, stride: s, padding: p, dilation: d, groups: c.groups, result: Aicode.tensor(out, c.element)))]
    let written = m.bytecode(), reference = expected(name)
    let first = (0..<min(reference.count, written.count)).first { reference[$0] != written[$0] } ?? min(reference.count, written.count)
    #expect(written == reference, "\(name): \(written.count) bytes against \(reference.count), first difference at \(first)")
}
