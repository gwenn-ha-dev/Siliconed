import Foundation
import Metal
import MetalPerformanceShadersGraph

// MARK: - The decoders' convolution, in fp32 and in bf16

/// **MPSGraph's 3×3 convolution, timed on fp32 operands and on bf16 ones** — what the plan reads to run a decoder published in bf16
/// in bf16 (`CapacityModel.Convolution`, `CapacityPlan.Resources.halfDecoder`). A decoder is ~90 % convolutions by its FLOP (Z-Image's
/// at 1024²: 10.4 TFLOP), all of them wide: `C → C` channels at a band of rows, never thin. Two such convolutions, `C = 512` at
/// `rows × 128` pixels (`Sizes.convolutionRows`), each in both formats, one graph per format and shape:
///
///     gpu/conv        fp32 operands, fp32 output                 (the decoders' route since their first line)
///     gpu/convbf16    bf16 operands and output, fp32 sums        (the route of a decoder in half precision, `VAEPrecision`)
///
/// Judged first at the smaller: the bf16 output against the fp32 one on the same operands (rounded to bf16 for both), channel by
/// channel, at `threshold` — an error of the kernel, not of the format (bf16's output rounding is ~2·10⁻³). Refused, the line is
/// not kept and the decoders stay in fp32. What it costs the map: a few tenths of a second. What it removes: a decoder computed in
/// fp32 on a GPU whose bf16 convolution is 2.5 times faster (an M5 Pro: 9.6 against 24.8 TFLOP/s), or in bf16 on one where it is not.
extension Diagnostic.CapacityProbes.Context {
    /// The bf16 convolution's worst channel against the fp32 one beyond which the kernel is refused.
    static let convolutionThreshold = 3e-2

    func convolution(_ measurements: inout Fit.Measurements) {
        let c = Sizes.convolutionChannels, w = Sizes.convolutionWidth
        let rows = Sizes.convolutionRows
        guard let largest = rows.max(), let smallest = rows.min() else { return }
        let device = bench.device
        let (inputCount, weightCount) = (c * largest * w, c * c * 9)
        let source = Pages(bytes: 4 * max(inputCount, weightCount))
        Diagnostic.fill(source.floats(), count: max(inputCount, weightCount), scale: 1, seed: 21)
        // The operands: values of bf16, held in both formats — the two kernels read the same numbers.
        guard let x32 = device.makeBuffer(length: 4 * inputCount, options: .storageModeShared),
              let x16 = device.makeBuffer(length: 2 * inputCount, options: .storageModeShared),
              let w32 = device.makeBuffer(length: 4 * weightCount, options: .storageModeShared),
              let w16 = device.makeBuffer(length: 2 * weightCount, options: .storageModeShared),
              let y32 = device.makeBuffer(length: 4 * inputCount, options: .storageModeShared),
              let y16 = device.makeBuffer(length: 2 * inputCount, options: .storageModeShared) else { return }
        Widen.float32ToBfloat16(source: source.floats(), destination: x16.contents(), count: inputCount)
        Widen.bfloat16ToFloat32(source: x16.contents(), destination: x32.contents(), count: inputCount)
        // The weights a tenth of the activations, as a decoder's 3×3 kernels are (~1/√(9C)).
        let scaled = (0..<weightCount).map { source.floats()[$0] * 0.03 }
        scaled.withUnsafeBufferPointer { Widen.float32ToBfloat16(source: $0.baseAddress!, destination: w16.contents(), count: weightCount) }
        Widen.bfloat16ToFloat32(source: w16.contents(), destination: w32.contents(), count: weightCount)

        let queue = bench.queue
        func graph(_ type: MPSDataType, rows h: Int) -> () -> Bool {
            let g = MPSGraph()
            let shape = [1, c, h, w].map { NSNumber(value: $0) }, kernel = [c, c, 3, 3].map { NSNumber(value: $0) }
            let x = g.placeholder(shape: shape, dataType: type, name: nil), k = g.placeholder(shape: kernel, dataType: type, name: nil)
            let d = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1,
                                                      paddingLeft: 1, paddingRight: 1, paddingTop: 1, paddingBottom: 1,
                                                      paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
            let y = g.convolution2D(x, weights: k, descriptor: d, name: nil)
            let half = type == .bFloat16
            let feeds = [x: MPSGraphTensorData(half ? x16 : x32, shape: shape, dataType: type),
                         k: MPSGraphTensorData(half ? w16 : w32, shape: kernel, dataType: type)]
            let results = [y: MPSGraphTensorData(half ? y16 : y32, shape: shape, dataType: type)]
            return {
                autoreleasepool {
                    let commands = MPSCommandBuffer(from: queue)
                    g.encode(to: commands, feeds: feeds, targetOperations: nil, resultsDictionary: results, executionDescriptor: nil)
                    commands.commit()
                    commands.waitUntilCompleted()
                    return commands.status != .error && commands.error == nil
                }
            }
        }
        // The judge: both formats at the smallest shape, the bf16 output widened, channel by channel (a channel is a row of pixels).
        let fp32Small = graph(.float32, rows: smallest), bf16Small = graph(.bFloat16, rows: smallest)
        guard fp32Small(), bf16Small() else { measurements.wrong.append(Fit.Measurements.convolutionBF16); return }
        let plane = smallest * w
        var got = [Float](repeating: 0, count: c * plane)
        got.withUnsafeMutableBytes { Widen.bfloat16ToFloat32(source: y16.contents(), destination: $0.baseAddress!, count: c * plane) }
        let expected = y32.contents().assumingMemoryBound(to: Float.self)
        var error = 0.0
        for ch in 0..<c {
            var difference = 0.0, norm = 0.0
            for i in (ch * plane)..<((ch + 1) * plane) {
                let e = Double(expected[i]), d = Double(got[i]) - e
                difference += d * d; norm += e * e
            }
            error = max(error, (difference / max(norm, 1e-30)).squareRoot())
        }
        let bf16Right = error.isFinite && error <= Self.convolutionThreshold
        if !bf16Right { measurements.wrong.append(Fit.Measurements.convolutionBF16) }
        var solos: [Fit.Measurements.Solo] = []
        for h in rows.sorted() {
            let flop = Double(2 * c * c * 9 * h * w), shape = [c, c, h, w]
            let fp32 = h == smallest ? fp32Small : graph(.float32, rows: h)
            if let t = guarded("gpu/conv \(h)x\(w)", { CapacityProbes.steady(fp32) }) {
                solos.append(.init(kernel: Fit.Measurements.convolutionFP32, shape: shape, flop: flop, seconds: t))
            }
            if bf16Right {
                let bf16 = h == smallest ? bf16Small : graph(.bFloat16, rows: h)
                if let t = guarded("gpu/convbf16 \(h)x\(w)", { CapacityProbes.steady(bf16) }) {
                    solos.append(.init(kernel: Fit.Measurements.convolutionBF16, shape: shape, flop: flop, seconds: t, error: error))
                }
            }
            let line = solos.filter { $0.shape == shape }.map { "\($0.kernel) " + String(format: "%.3f ms", $0.seconds * 1e3) }
            log?("  conv \(c)→\(c) \(h)×\(w)  " + line.joined(separator: "  "))
        }
        measurements.convolution = solos
        withExtendedLifetime(source) {}
    }

    /// **What the probe holds** (`CapacityProbes.need`): its fp32 source, the operands in both formats and both outputs at the largest
    /// shape, and as much again for what MPSGraph's convolution keeps of its own (reformatted weights, its temporaries).
    static func convolutionBytes() -> Int {
        let c = Sizes.convolutionChannels, n = c * (Sizes.convolutionRows.max() ?? 0) * Sizes.convolutionWidth, k = c * c * 9
        let ours = [4 * max(n, k), 4 * n, 2 * n, 4 * k, 2 * k, 4 * n, 2 * n].map { Arena.capacity(for: $0) }.reduce(0, +)
        return 2 * ours
    }
}

extension Diagnostic.CapacityProbes.Sizes {
    /// The decoders' convolution probe (`Context.convolution`): `C → C` channels, `rows × width` pixels — a decoder's bottleneck
    /// width (Z-Image's 512 channels) on a band of it: the FLOP of a band (9.7 and 19.3 G), in 97 MB at most with what MPSGraph keeps —
    /// the memory a Neo's second tier still holds (`CapacityProbes.need`), where 128 channels on as many FLOP held three times more.
    static let convolutionChannels = 512, convolutionWidth = 128
    static let convolutionRows = [16, 32]
}

extension CapacityFit {
    /// **The decoders' convolution, its two lines from the probe** (`Context.convolution`), compared at the largest convolution measured;
    /// `nil` without the probe, with the bf16 kernel refused, or without a line for each format.
    package static func convolution(_ solos: [Measurements.Solo]?, wrong: [String] = []) -> CapacityModel.Convolution? {
        guard let solos, !wrong.contains(Measurements.convolutionBF16) else { return nil }
        func fitted(_ name: String) -> CapacityModel.Line? {
            line(solos.filter { $0.kernel == name }.map { Point(flop: $0.flop, seconds: $0.seconds) })
        }
        guard let fp32 = fitted(Measurements.convolutionFP32), let bf16 = fitted(Measurements.convolutionBF16),
              let work = solos.map(\.flop).max() else { return nil }
        return .init(fp32: fp32, bf16: bf16, work: work)
    }
}

extension CapacityFit.Measurements {
    package static let convolutionFP32 = "gpu/conv", convolutionBF16 = "gpu/convbf16"
}
