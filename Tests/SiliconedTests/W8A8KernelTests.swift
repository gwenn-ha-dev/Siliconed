import Metal
import XCTest
@testable import Siliconed

/// **W8A8 on the GPU against ComfyUI's recipe on the CPU, to the bit** (`QuantizedKernels`, `TensorGEMM.submitQuantized`): the activation
/// rotated by groups (the regular Hadamard, `rotate_groups`'s butterflies) and quantized per row (`W8A8.quantize`: max|y| / 127, `rint` of the
/// correctly rounded quotient, clamp), the int8 × int8 product exact in int32, then `float(acc) · (s_x · s_w)` — every step a single rounding the
/// CPU makes the same, so the output is compared for equality, not to a bar. Rows cut in chunks with a remainder, a row of 40 chunks of 256
/// (two per simdgroup), a group of 64, no rotation on a `k` that is no whole chunk, a scale per column and one for the tensor, a row of zeros
/// (its scale the recipe's floor, 1e-30), a massive column. Skipped without the Metal 4 tensor kernels.
final class W8A8KernelTests: XCTestCase {
    private struct Case { let m, k, n: Int; let group: Int?; let perColumn: Bool; let chunk: Int; var inMap = false }

    private func run(_ t: Case) throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        guard let gemm = TensorGEMM(device: device, queue: queue, quantizingActivations: true), gemm.quantizesActivations else {
            throw XCTSkip("no Metal 4 tensor kernels for W8A8 on this device")
        }
        let (m, k, n) = (t.m, t.k, t.n)
        var state: UInt64 = 0x5EED ^ UInt64(m * 131 + k * 7 + n)
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(UInt32(truncatingIfNeeded: state >> 32)) / Float(UInt32.max) - 0.5
        }
        var a = (0..<(m * k)).map { _ in next() * 6 }
        for i in 0..<k { a[3 * k + i] = 0 }                        // a row of zeros
        for r in 0..<m { a[r * k + k / 3] *= 300 }                 // a massive column
        let q = (0..<(k * n)).map { _ in Int8(Int((next() * 254).rounded())) }
        let sw = (0..<(t.perColumn ? n : 1)).map { _ in (abs(next()) + 0.01) / 64 }
        // The reserve as `WeightProducts.int8Plan` lays it: the integers, their scales, the codes and the activation's scales — or, the
        // weight read in place (`inMap`), the integers and their scale where a map holds them (a page in, the scale on the next 16 bytes
        // past the values, as `Artifact.Scale` lays it), and the reserve the activation's alone from its second page.
        func page(_ b: Int) -> Int { (b + 16383) / 16384 * 16384 }
        let weightOffset = t.inMap ? 16384 : 0
        let scalesOffset = t.inMap ? weightOffset + (k * n + 15) / 16 * 16 : page(k * n)
        let codesOffset = t.inMap ? 16384 : page(scalesOffset + sw.count * 4), activationScales = page(codesOffset + t.chunk * k)
        let reserve = try XCTUnwrap(device.makeBuffer(length: activationScales + page(t.chunk * 4), options: .storageModeShared))
        let weight = t.inMap ? try XCTUnwrap(device.makeBuffer(length: page(scalesOffset + sw.count * 4), options: .storageModeShared)) : reserve
        q.withUnsafeBytes { (weight.contents() + weightOffset).copyMemory(from: $0.baseAddress!, byteCount: q.count) }
        sw.withUnsafeBytes { (weight.contents() + scalesOffset).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        let input = try XCTUnwrap(device.makeBuffer(bytes: a, length: 4 * a.count, options: .storageModeShared))
        let output = try XCTUnwrap(device.makeBuffer(length: 4 * m * n, options: .storageModeShared))
        let commands = try XCTUnwrap(gemm.submitQuantized(a: input, int8: weight, weightOffset: weightOffset, scales: weight,
                                                          scalesOffset: scalesOffset,
                                                          perColumn: t.perColumn, group: t.group,
                                                          quantizing: (reserve, codesOffset, activationScales, t.chunk),
                                                          c: output, m: m, k: k, n: n))
        commands.waitUntilCompleted()
        XCTAssertNil(commands.error)
        let c = output.contents().assumingMemoryBound(to: Float.self)
        let codes = (reserve.contents() + codesOffset).assumingMemoryBound(to: Int8.self)
        let scales = (reserve.contents() + activationScales).assumingMemoryBound(to: Float.self)
        let lastChunk = (m - 1) / t.chunk * t.chunk
        var unequal = 0, codesOff = 0
        for r in 0..<m {
            let (x, sx) = a.withUnsafeBufferPointer { W8A8.quantize($0.baseAddress! + r * k, k: k, group: t.group) }
            if r == 3 { XCTAssertEqual(sx, 1e-30, "a row of zeros keeps the recipe's floor") }
            // The slot holds the last chunk's codes and scales.
            if r >= lastChunk {
                XCTAssertEqual(scales[r - lastChunk], sx, "row \(r): the activation's scale")
                for i in 0..<k where codes[(r - lastChunk) * k + i] != x[i] { codesOff += 1 }
            }
            for j in 0..<n {
                var acc: Int32 = 0
                for i in 0..<k { acc &+= Int32(x[i]) * Int32(q[i * n + j]) }
                let expected = Float(acc) * (sx * sw[t.perColumn ? j : 0])
                if c[r * n + j].bitPattern != expected.bitPattern { unequal += 1 }
            }
        }
        XCTAssertEqual(codesOff, 0, "codes unlike the CPU's recipe")
        XCTAssertEqual(unequal, 0, "outputs unlike the CPU's recipe, of \(m * n)")
    }

    func testRotatedRowsInChunksAgainstTheRecipe() throws { try run(Case(m: 150, k: 768, n: 96, group: 256, perColumn: true, chunk: 64)) }
    func testALongRowTwoChunksASimdgroup() throws { try run(Case(m: 70, k: 10240, n: 72, group: 256, perColumn: true, chunk: 64)) }
    func testAGroupOf64AndATensorScale() throws { try run(Case(m: 65, k: 448, n: 130, group: 64, perColumn: false, chunk: 128)) }
    func testNoRotationOnAPartialChunk() throws { try run(Case(m: 33, k: 520, n: 40, group: nil, perColumn: true, chunk: 64)) }
    /// The weight read in place (`WeightProducts.inPlaceInt8`): the integers at a page of another buffer, their scale right past them,
    /// the activation's codes from the reserve's second page — the recipe's bits, as from the reserve.
    func testTheWeightReadInPlaceInTheMap() throws {
        try run(Case(m: 150, k: 768, n: 96, group: 256, perColumn: true, chunk: 64, inMap: true))
        try run(Case(m: 65, k: 448, n: 130, group: 64, perColumn: false, chunk: 128, inMap: true))
    }

    // MARK: The operand quantized by its producer

    /// **The norm that writes its output quantized too, against the norm then `quantize_rows`, to the bit** (`norm_scale_quantized`,
    /// `WeightProducts.RoundedOperand.int8`): its fp32 output is `norm_scale`'s, its codes and scales are those `quantize_rows` writes
    /// from that output (and the CPU's recipe), and the product that reads them without a pass of its own (`prequantized`) writes the
    /// outputs of the product that quantizes for itself. Z-Image's row (3 840, groups of 256), a partial chunk without rotation and
    /// without the modulation, the longest row the norm holds (8 192, groups of 16); a row of zeros, a massive column.
    private func runFused(rows m: Int, k: Int, n: Int, group: Int?, scaled: Bool) throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        guard let gemm = TensorGEMM(device: device, queue: queue, quantizingActivations: true), gemm.quantizesActivations else {
            throw XCTSkip("no Metal 4 tensor kernels for W8A8 on this device")
        }
        let elementwise = try ElementwiseGPU(device: device, queue: queue)
        var state: UInt64 = 0xF05E ^ UInt64(m * 131 + k * 7 + n)
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(UInt32(truncatingIfNeeded: state >> 32)) / Float(UInt32.max) - 0.5
        }
        var x = (0..<(m * k)).map { _ in next() * 4 }
        for i in 0..<k { x[2 * k + i] = 0 }                        // a row of zeros
        for r in 0..<m { x[r * k + k / 5] *= 200 }                 // a massive column
        let weight = (0..<k).map { _ in 1 + next() }, modulation = (0..<k).map { _ in 1 + next() / 2 }
        let q = (0..<(k * n)).map { _ in Int8(Int((next() * 254).rounded())) }
        let sw = (0..<n).map { _ in (abs(next()) + 0.01) / 64 }
        func page(_ b: Int) -> Int { (b + 16383) / 16384 * 16384 }
        let scalesOffset = page(k * n), codesOffset = page(scalesOffset + sw.count * 4), activationScales = page(codesOffset + m * k)
        func reserve() throws -> MTLBuffer {
            let r = try XCTUnwrap(device.makeBuffer(length: activationScales + page(m * 4), options: .storageModeShared))
            q.withUnsafeBytes { r.contents().copyMemory(from: $0.baseAddress!, byteCount: q.count) }
            sw.withUnsafeBytes { (r.contents() + scalesOffset).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            return r
        }
        let (fused, separate) = (try reserve(), try reserve())
        func buffer(_ v: [Float]) throws -> MTLBuffer { try XCTUnwrap(device.makeBuffer(bytes: v, length: 4 * v.count, options: .storageModeShared)) }
        let (xb, wb, sb) = (try buffer(x), try buffer(weight), try buffer(modulation))
        let plain = try XCTUnwrap(device.makeBuffer(length: 4 * m * k, options: .storageModeShared))
        let normed = try XCTUnwrap(device.makeBuffer(length: 4 * m * k, options: .storageModeShared))
        let scale: ElementwiseGPU.BufferSlice? = scaled ? (sb, 0) : nil
        elementwise.normScale((xb, 0), weight: (wb, 0), scale: scale, into: (plain, 0), rows: m, columns: k, eps: 1e-6)
        let slot = WeightProducts.RoundedOperand.Quantized(buffer: fused, codesOffset: codesOffset, scalesOffset: activationScales,
                                                           group: group, stages: try XCTUnwrap(W8A8.stages(group: group)), byProducer: true)
        elementwise.normScale((xb, 0), weight: (wb, 0), scale: scale, into: (normed, 0), rows: m, columns: k, eps: 1e-6, quantized: slot)
        let (p0, p1) = (plain.contents().assumingMemoryBound(to: Float.self), normed.contents().assumingMemoryBound(to: Float.self))
        XCTAssertEqual((0..<(m * k)).filter { p0[$0].bitPattern != p1[$0].bitPattern }.count, 0, "the fused norm's fp32 unlike norm_scale's")

        // The product that quantizes for itself, on the norm's output; the one that reads the norm's codes.
        let (c0, c1) = (try XCTUnwrap(device.makeBuffer(length: 4 * m * n, options: .storageModeShared)),
                        try XCTUnwrap(device.makeBuffer(length: 4 * m * n, options: .storageModeShared)))
        for (reserve, c, prequantized) in [(separate, c0, false), (fused, c1, true)] {
            let commands = try XCTUnwrap(gemm.submitQuantized(a: plain, int8: reserve, scales: reserve, scalesOffset: scalesOffset,
                                                              perColumn: true, group: group,
                                                              quantizing: (reserve, codesOffset, activationScales, m),
                                                              prequantized: prequantized, c: c, m: m, k: k, n: n))
            commands.waitUntilCompleted()
            XCTAssertNil(commands.error)
        }
        let (codesF, codesS) = ((fused.contents() + codesOffset).assumingMemoryBound(to: Int8.self),
                                (separate.contents() + codesOffset).assumingMemoryBound(to: Int8.self))
        let (scalesF, scalesS) = ((fused.contents() + activationScales).assumingMemoryBound(to: Float.self),
                                  (separate.contents() + activationScales).assumingMemoryBound(to: Float.self))
        var codesOff = 0, scalesOff = 0, recipeOff = 0
        for r in 0..<m {
            let (codes, sx) = W8A8.quantize(p0 + r * k, k: k, group: group)
            if scalesF[r].bitPattern != scalesS[r].bitPattern { scalesOff += 1 }
            if scalesF[r].bitPattern != sx.bitPattern { recipeOff += 1 }
            for i in 0..<k {
                if codesF[r * k + i] != codesS[r * k + i] { codesOff += 1 }
                if codesF[r * k + i] != codes[i] { recipeOff += 1 }
            }
        }
        XCTAssertEqual(codesOff, 0, "the norm's codes unlike quantize_rows'")
        XCTAssertEqual(scalesOff, 0, "the norm's scales unlike quantize_rows'")
        XCTAssertEqual(recipeOff, 0, "the norm's quantization unlike the CPU's recipe")
        XCTAssertEqual(scalesF[2], 1e-30, "a row of zeros keeps the recipe's floor")
        let (o0, o1) = (c0.contents().assumingMemoryBound(to: Float.self), c1.contents().assumingMemoryBound(to: Float.self))
        XCTAssertEqual((0..<(m * n)).filter { o0[$0].bitPattern != o1[$0].bitPattern }.count, 0,
                       "the product on the norm's codes unlike the one that quantizes for itself")
    }

    func testTheNormQuantizesZImagesRowAsTheSeparatePass() throws { try runFused(rows: 37, k: 3840, n: 64, group: 256, scaled: true) }
    func testTheNormQuantizesAPartialChunkUnrotated() throws { try runFused(rows: 19, k: 520, n: 40, group: nil, scaled: false) }
    func testTheNormQuantizesTheLongestRowItHolds() throws {
        try runFused(rows: 9, k: ElementwiseGPU.quantizedColumns, n: 32, group: 16, scaled: true)
    }

    /// **Each family's norm that quantizes as it writes, against itself without, to the bit** (`layer_norm_scale_quantized`,
    /// `norm_modulate_quantized` ×2, `residual_modulate_quantized`): the same fp32 output (and Anima's residual, written in place), and
    /// the codes and scales of the CPU's recipe on that output (`W8A8.quantize`, which `quantize_rows` writes bit for bit). Every input
    /// `[rows, k]` (a per-column one reads its first row); a row of zeros where the input is the row itself.
    private func checkProducer(rows m: Int, k: Int, group: Int?, inputs: Int,
                               run: (MTLDevice, MTLCommandQueue, [MTLBuffer], MTLBuffer, WeightProducts.RoundedOperand?) throws -> Void) throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        var state: UInt64 = 0xA11 ^ UInt64(m * 131 + k * 7 + inputs)
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(UInt32(truncatingIfNeeded: state >> 32)) / Float(UInt32.max) - 0.5
        }
        let values: [[Float]] = (0..<inputs).map { i in
            var v = (0..<(m * k)).map { _ in i == 0 ? next() * 4 : 1 + next() / 2 }
            if i == 0 { for j in 0..<k { v[j] = 0 } }               // a row of zeros
            return v
        }
        func fresh() throws -> [MTLBuffer] {
            try values.map { try XCTUnwrap(device.makeBuffer(bytes: $0, length: 4 * $0.count, options: .storageModeShared)) }
        }
        let (plainInputs, fusedInputs) = (try fresh(), try fresh())
        let plain = try XCTUnwrap(device.makeBuffer(length: 4 * m * k, options: .storageModeShared))
        let fused = try XCTUnwrap(device.makeBuffer(length: 4 * m * k, options: .storageModeShared))
        let slot = try XCTUnwrap(device.makeBuffer(length: 16384 + m * k + 16384, options: .storageModeShared))
        let scalesOffset = (m * k + 16383) / 16384 * 16384
        let q = WeightProducts.RoundedOperand.Quantized(buffer: slot, codesOffset: 0, scalesOffset: scalesOffset, group: group,
                                                        stages: try XCTUnwrap(W8A8.stages(group: group)), byProducer: true)
        try run(device, queue, plainInputs, plain, nil)
        try run(device, queue, fusedInputs, fused, WeightProducts.RoundedOperand(producing: q, k: k, rows: m))
        func bits(_ b: MTLBuffer, _ count: Int) -> [UInt32] { Array(UnsafeBufferPointer(start: b.contents().assumingMemoryBound(to: UInt32.self), count: count)) }
        XCTAssertEqual(bits(plain, m * k), bits(fused, m * k), "the fused norm's fp32 unlike the plain one's")
        for (a, b) in zip(plainInputs, fusedInputs) { XCTAssertEqual(bits(a, m * k), bits(b, m * k), "an input written otherwise") }
        let out = plain.contents().assumingMemoryBound(to: Float.self)
        let codes = slot.contents().assumingMemoryBound(to: Int8.self), scales = (slot.contents() + scalesOffset).assumingMemoryBound(to: Float.self)
        var off = 0
        for r in 0..<m {
            let (x, sx) = W8A8.quantize(out + r * k, k: k, group: group)
            if scales[r].bitPattern != sx.bitPattern { off += 1 }
            for i in 0..<k where codes[r * k + i] != x[i] { off += 1 }
        }
        XCTAssertEqual(off, 0, "the norm's quantization unlike the CPU's recipe")
    }

    private typealias Slice = ElementwiseGPU.BufferSlice

    func testQwensLayerNormQuantizesAsItWrites() throws {
        try checkProducer(rows: 21, k: 3072, group: 256, inputs: 2) { device, queue, b, out, operand in
            try ModulatedLayerNormGPU(device: device, queue: queue).run((b[0], 0), scale: (b[1], 0), into: (out, 0), rows: 21, columns: 3072,
                                                                     eps: 1e-6, rounded: operand)
        }
    }

    func testKrea2sNormQuantizesAsItWrites() throws {
        try checkProducer(rows: 13, k: 3072, group: 64, inputs: 4) { device, queue, b, out, operand in
            try ModulatedNormGPU(device: device, queue: queue).run((b[0], 0), weight: (b[1], 0), factor: (b[2], 0), shift: (b[3], 0),
                                                                   into: (out, 0), rows: 13, columns: 3072, eps: 1e-6, rounded: operand)
        }
    }

    func testMossNormQuantizesAsItWrites() throws {
        try checkProducer(rows: 11, k: 1280, group: 256, inputs: 4) { device, queue, b, out, operand in
            try MossImageNormGPU(device: device, queue: queue).normModulate((b[0], 0), weight: (b[1], 0), scale: (b[2], 0), shift: (b[3], 0),
                                                                            into: (out, 0), rows: 11, columns: 1280, eps: 1e-6, rounded: operand)
        }
    }

    func testAnimasResidualAndNormQuantizeAsTheyWrite() throws {
        for residual in [true, false] {
            try checkProducer(rows: 17, k: 2048, group: 256, inputs: 5) { device, queue, b, out, operand in
                let s: (Int) -> Slice = { (b[$0], 0) }
                try AnimaElementwiseGPU(device: device, queue: queue).run(s(0), residual: residual ? (y: s(1), gate: s(2)) : nil,
                                                                          modulate: (shift: s(3), scale: s(4), out: (out, 0)),
                                                                          rows: 17, columns: 2048, eps: 1e-6, rounded: operand)
            }
        }
    }

    /// The route's own plan: the codes and their scales past the integers, in the reserve a model holds, chunked by what is left.
    func testThePlanLaysTheQuantizedActivationInTheReserve() throws {
        let plan = try XCTUnwrap(WeightProducts.int8Plan(k: 3840, n: 11520, capacity: 3840 * 11520, perColumn: true, group: 256,
                                                         amxColumns: 0, quantized: true))
        XCTAssertGreaterThan(plan.quantizedOffset, 0)
        XCTAssertGreaterThanOrEqual(plan.quantizedOffset, plan.scalesOffset + 11520 * 4)
        XCTAssertGreaterThanOrEqual(plan.quantizedScalesOffset, plan.quantizedOffset + plan.chunk * 3840)
        XCTAssertLessThanOrEqual(plan.quantizedScalesOffset + plan.chunk * 4, 3840 * 11520 * 4)
        XCTAssertGreaterThanOrEqual(plan.chunk, 4128, "Z-Image's 1024² rows in one chunk")
        XCTAssertEqual(plan.roundedOffset, 0)
        XCTAssertNil(WeightProducts.int8Plan(k: 3840, n: 3840, capacity: 3840 * 3840, perColumn: true, group: 512, amxColumns: 0, quantized: true),
                     "a rotation group the kernel does not take")
        XCTAssertNil(WeightProducts.int8Plan(k: 4096, n: 64, capacity: 4096 * 64, perColumn: false, group: nil, amxColumns: 0, block: 32,
                                             quantized: true), "Q8_0's blocks cut the integers' sum")
    }

    /// **The weight read in place leaves the reserve to the activation** (`WeightProducts.inPlaceInt8`): the same route from the reserve's
    /// second page (an offset of 0 is the plan's "none"), more rows a chunk than beside the copy, in the reserve still; no AMX columns.
    func testThePlanOfAWeightReadInPlace() throws {
        let (k, n) = (10240, 3840), capacity = 10240 * 3840
        for (rounded, quantized) in [(false, true), (true, false)] {
            let copied = try XCTUnwrap(WeightProducts.int8Plan(k: k, n: n, capacity: capacity, perColumn: true, group: 256, amxColumns: 0,
                                                               rounded: rounded, quantized: quantized))
            let inPlace = try XCTUnwrap(WeightProducts.int8Plan(k: k, n: n, capacity: capacity, perColumn: true, group: 256, amxColumns: 0,
                                                                rounded: rounded, quantized: quantized, weightInReserve: false))
            XCTAssertEqual(inPlace.rotatedOffset, 16384)
            XCTAssertEqual(inPlace.quantizedOffset > 0, quantized)
            XCTAssertEqual(inPlace.roundedOffset > 0, rounded)
            XCTAssertGreaterThan(inPlace.chunk, copied.chunk)
            if quantized {
                XCTAssertEqual(inPlace.quantizedOffset, 16384)
                XCTAssertGreaterThanOrEqual(inPlace.quantizedScalesOffset, inPlace.quantizedOffset + inPlace.chunk * k)
                XCTAssertLessThanOrEqual(inPlace.quantizedScalesOffset + inPlace.chunk * 4, capacity * 4)
            } else {
                XCTAssertGreaterThanOrEqual(inPlace.roundedOffset, inPlace.rotatedOffset + inPlace.chunk * k * 4)
                XCTAssertLessThanOrEqual(inPlace.roundedOffset + inPlace.chunk * k * 2, capacity * 4)
            }
        }
        XCTAssertNil(WeightProducts.int8Plan(k: k, n: n, capacity: capacity, perColumn: true, group: 256, amxColumns: 512,
                                             weightInReserve: false), "the AMX's columns are widened from the copy")
    }
}
