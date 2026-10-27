// The half-precision products (fp16, bf16: MPSGraph, BNNS, Metal 4 matmul2d) and the
// fp32 attention. Split from `DiagnosticProbes.swift`.

import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

extension Diagnostic {
    // ── fp16 and bf16 products ──────────────────────────────────────────────────────────────

    /// A half-precision type, the way the four parties name and round it.
    enum Half: String, CaseIterable {
        case fp16, bf16
        var graph: MPSDataType { self == .fp16 ? .float16 : .bFloat16 }
        var bnns: BNNSDataType { self == .fp16 ? .float16 : BNNSDataTypeBFloat16 }
        /// The `matmul2d` kernel of operands of this type, fp32 result.
        var kernel: String { self == .fp16 ? "product_half" : "product_bfloat" }
        /// `x` rounded to the nearest of the type, to nearest even, as an fp32.
        func round(_ x: Float) -> Float { self == .fp16 ? Float(Float16(x)) : widen(narrow(x)) }
        func narrow(_ x: Float) -> UInt16 {
            if self == .fp16 { return Float16(x).bitPattern }
            let bits = x.bitPattern
            return UInt16(truncatingIfNeeded: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) >> 16)
        }
        func widen(_ h: UInt16) -> Float { self == .fp16 ? Float(Float16(bitPattern: h)) : Float(bitPattern: UInt32(h) << 16) }
    }

    /// **Half precision at the first product's shape**, three engines per type. The fp32 operands are
    /// rounded in place to the type first — the exact is then the product of exactly the values the
    /// engines read (fp16 before bf16: a value rounded to fp16, then to bf16, is a bf16).
    static func measureLowPrecision(_ o: Operands, _ bench: Bench) throws -> [LowPrecision] {
        let s = shapes[0]
        let (m, k, n) = (s.m, s.k, s.n)
        let a = o.a.floats(), b = o.b.floats(), c = o.c.floats()
        let (aHalf, bHalf, cHalf) = (Pages(bytes: m * k * 2), Pages(bytes: k * n * 2), Pages(bytes: m * n * 2))
        let (aBuffer, bBuffer, cBuffer) = (try bench.buffer(aHalf), try bench.buffer(bHalf), try bench.buffer(cHalf))
        let cFloat = try bench.buffer(o.c)
        let tensors = TensorProducts(bench)
        let flops = 2.0 * Double(m) * Double(k) * Double(n)
        var workspace: Pages?
        var results: [LowPrecision] = []
        for type in Half.allCases {
            try bench.check()
            for (values, half, count) in [(a, aHalf, m * k), (b, bHalf, k * n)] {
                nonisolated(unsafe) let (values, out) = (values, half.base.assumingMemoryBound(to: UInt16.self))
                DispatchQueue.concurrentPerform(iterations: (count + 65_535) / 65_536) { j in
                    for i in (j * 65_536)..<min(count, (j + 1) * 65_536) {
                        values[i] = type.round(values[i])
                        out[i] = type.narrow(values[i])
                    }
                }
            }
            let rows = sampleRows(m, count: sampledRows)
            let exact = exactRows(rows, a: a, b: b, k: k, n: n)
            // The floors: the exact rounded to the output's type, against the exact.
            let floorHalf = judge(exact.map { type.round(Float($0)) }, exact, columns: rows.count, rate: 0, threshold: 1).error
            let floorFloat = judge(exact.map { Float($0) }, exact, columns: rows.count, rate: 0, threshold: 1).error
            func threshold(_ floor: Double) -> Double { max(fp32Threshold, floorMargin * floor) }

            // MPSGraph: operands and result in the type.
            let h = cHalf.base.assumingMemoryBound(to: UInt16.self)
            for row in rows { (h + row * n).update(repeating: type.narrow(.nan), count: n) }
            let graph = MPSGraph()
            let x = graph.placeholder(shape: [m, k].map { NSNumber(value: $0) }, dataType: type.graph, name: nil)
            let y = graph.placeholder(shape: [k, n].map { NSNumber(value: $0) }, dataType: type.graph, name: nil)
            let z = graph.matrixMultiplication(primary: x, secondary: y, name: nil)
            let feeds = [x: MPSGraphTensorData(aBuffer, shape: [m, k].map { NSNumber(value: $0) }, dataType: type.graph),
                         y: MPSGraphTensorData(bBuffer, shape: [k, n].map { NSNumber(value: $0) }, dataType: type.graph)]
            let result = [z: MPSGraphTensorData(cBuffer, shape: [m, n].map { NSNumber(value: $0) }, dataType: type.graph)]
            let tGraph = seconds { graph.run(with: bench.queue, feeds: feeds, targetOperations: nil, resultsDictionary: result) }
            var got = [Float](repeating: 0, count: rows.count * n)
            for (i, row) in rows.enumerated() { for j in 0..<n { got[j * rows.count + i] = type.widen(h[row * n + j]) } }
            results.append(LowPrecision(engine: "mpsgraph", type: type.rawValue, output: type.rawValue, m: m, k: k, n: n,
                                        measure: judge(got, exact, columns: rows.count, rate: flops / tGraph / 1e12,
                                                       threshold: threshold(floorHalf)),
                                        floor: floorHalf, threshold: threshold(floorHalf)))

            // BNNS: operands in the type, result in fp32.
            try bench.check()
            var measure: Measure?
            if let run = bnnsProduct(type, a: aHalf.base, b: bHalf.base, c: o.c.base, m: m, k: k, n: n, workspace: &workspace) {
                poison(c, rows: rows, n: n)
                let t = seconds(run)
                measure = judge(gather(c, rows: rows, n: n), exact, columns: rows.count, rate: flops / t / 1e12,
                                threshold: threshold(floorFloat))
            }
            results.append(LowPrecision(engine: "bnns", type: type.rawValue, output: "fp32", m: m, k: k, n: n,
                                        measure: measure, floor: floorFloat, threshold: threshold(floorFloat)))

            // Metal 4's matmul2d: operands in the type, result in fp32.
            try bench.check()
            measure = nil
            if let run = tensors?.product(type.kernel, a: aBuffer, b: bBuffer, c: cFloat, m: m, k: k, n: n) {
                poison(c, rows: rows, n: n)
                let t = seconds(run)
                measure = judge(gather(c, rows: rows, n: n), exact, columns: rows.count, rate: flops / t / 1e12,
                                threshold: threshold(floorFloat))
            }
            results.append(LowPrecision(engine: "matmul2d", type: type.rawValue, output: "fp32", m: m, k: k, n: n,
                                        measure: measure, floor: floorFloat, threshold: threshold(floorFloat)))
        }
        return results
    }

    /// `BNNSMatMul` on operands of `type`, fp32 result, as `Conductor.multiply` calls it otherwise;
    /// `nil` when BNNS refuses the types.
    private static func bnnsProduct(_ type: Half, a: UnsafeMutableRawPointer, b: UnsafeMutableRawPointer,
                                    c: UnsafeMutableRawPointer, m: Int, k: Int, n: Int,
                                    workspace: inout Pages?) -> (() -> Void)? {
        func nd(_ p: UnsafeMutableRawPointer, _ rows: Int, _ columns: Int, _ t: BNNSDataType) -> BNNSNDArrayDescriptor {
            var d = BNNSNDArrayDescriptor()
            d.layout = BNNSDataLayoutRowMajorMatrix
            d.size = (columns, rows, 0, 0, 0, 0, 0, 0)
            d.data = p
            d.data_type = t
            d.data_scale = 1
            d.data_bias = 0
            return d
        }
        var (A, B, C) = (nd(a, m, k, type.bnns), nd(b, k, n, type.bnns), nd(c, m, n, .float))
        var params = BNNSFilterParameters()
        let size = amx_matmul_workspace(false, false, 1, &A, &B, &C, &params)
        guard size >= 0 else { return nil }
        if size > (workspace?.capacity ?? 0) { workspace = Pages(bytes: size) }
        nonisolated(unsafe) let space = size > 0 ? workspace?.base : nil
        return { _ = amx_matmul(false, false, 1, &A, &B, &C, space, &params) }
    }

    /// **Metal 4's tensor product** (`MetalPerformancePrimitives`, `matmul2d`): the kernel of Apple's
    /// header, 64 × 32 tiles over four simdgroups, its destination a cooperative tensor written once
    /// (`run` accumulates into a device tensor). On an M5 it runs on the GPU's neural accelerators;
    /// before, on the shader cores. Not tuned: it answers "is there a matrix engine", not "how fast can
    /// the best kernel go". `nil` on a device without Metal 4.
    private final class TensorProducts {
        private let queue: MTLCommandQueue
        private let pipelines: [String: MTLComputePipelineState]

        static let source = """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;
        using namespace mpp::tensor_ops;

        template <typename T>
        inline void product(device T *a, device T *b, device float *c, constant uint3 &mkn, uint2 tgid) {
            const int M = mkn.x, K = mkn.y, N = mkn.z;
            // Extents innermost first: A [M, K] row-major is (K, M).
            tensor<device T, dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
            tensor<device T, dextents<int32_t, 2>, tensor_inline> B(b, dextents<int32_t, 2>(N, K));
            tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
            constexpr auto descriptor = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
            matmul2d<descriptor, execution_simdgroups<4>> op;
            auto mA = A.slice(0, tgid.y * 64);
            auto mB = B.slice(tgid.x * 32, 0);
            auto mC = C.slice(tgid.x * 32, tgid.y * 64);
            // The header's example says `get_mask` and `__remove_addrspace_t`; this SDK's compiler
            // knows neither: `is_valid_element`, and the slices' own types.
            auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; }
            op.run(mA, mB, cT);
            cT.store(mC);
        }

        kernel void product_half(device half *a [[buffer(0)]], device half *b [[buffer(1)]],
                                 device float *c [[buffer(2)]], constant uint3 &mkn [[buffer(3)]],
                                 uint2 tgid [[threadgroup_position_in_grid]]) { product(a, b, c, mkn, tgid); }
        kernel void product_bfloat(device bfloat *a [[buffer(0)]], device bfloat *b [[buffer(1)]],
                                   device float *c [[buffer(2)]], constant uint3 &mkn [[buffer(3)]],
                                   uint2 tgid [[threadgroup_position_in_grid]]) { product(a, b, c, mkn, tgid); }
        """

        init?(_ bench: Bench) {
            guard bench.device.supportsFamily(.metal4) else { return nil }
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            guard let library = try? bench.device.makeLibrary(source: TensorProducts.source, options: options) else { return nil }
            var pipelines: [String: MTLComputePipelineState] = [:]
            for name in ["product_half", "product_bfloat"] {
                if let f = library.makeFunction(name: name), let p = try? bench.device.makeComputePipelineState(function: f) {
                    pipelines[name] = p
                }
            }
            queue = bench.queue
            self.pipelines = pipelines
        }

        /// `C [m, n] = A [m, k] · B [k, n]`, `m` a multiple of 64 and `n` of 32 (no edge tile).
        func product(_ kernel: String, a: MTLBuffer, b: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int) -> (() -> Void)? {
            guard let pipeline = pipelines[kernel], m % 64 == 0, n % 32 == 0 else { return nil }
            let queue = self.queue
            return {
                guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else { return }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(a, offset: 0, index: 0)
                encoder.setBuffer(b, offset: 0, index: 1)
                encoder.setBuffer(c, offset: 0, index: 2)
                var mkn = SIMD3<UInt32>(UInt32(m), UInt32(k), UInt32(n))
                encoder.setBytes(&mkn, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 3)
                encoder.dispatchThreadgroups(MTLSize(width: n / 32, height: m / 64, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                                                            height: 1, depth: 1))
                encoder.endEncoding()
                commands.commit()
                commands.waitUntilCompleted()
                _ = Attention.verdict(commands, what: "matmul2d \(m)×\(k)×\(n)")
            }
        }
    }

    // ── the attention ───────────────────────────────────────────────────────────────────────

    /// **Each model's heads, at both sequences, through the engine's three SDPAs.** `q` and `k` uniform
    /// in ±3, so the logits spread over ±12 and the softmax is peaked as a model's is (uniform in ±1, it
    /// would be flat, and an attention that ignored its weights would pass); `v` in ±1.
    static func measureAttention(_ bench: Bench) throws -> [AttentionCapacity] {
        var results: [AttentionCapacity] = []
        for block in blocks {
            for sequence in heights {
                try bench.check()
                let (h, d) = (block.heads, block.headDim)
                let floats = sequence * h * d
                let (q, k, v, out) = (Pages(bytes: floats * 4), Pages(bytes: floats * 4), Pages(bytes: floats * 4),
                                      Pages(bytes: floats * 4))
                fill(q.floats(), count: floats, scale: 3, seed: 11)
                fill(k.floats(), count: floats, scale: 3, seed: 12)
                fill(v.floats(), count: floats, scale: 1, seed: 13)
                let (qb, kb, vb, ob) = (try bench.buffer(q), try bench.buffer(k), try bench.buffer(v), try bench.buffer(out))
                let rows = sampleRows(sequence, count: sampledQueries)
                let exact = exactAttention(q: q.floats(), k: k.floats(), v: v.floats(), sequence: sequence, heads: h,
                                           headDim: d, rows: rows)
                let o = out.floats(), vectors = rows.count * h
                let flops = 4.0 * Double(sequence) * Double(sequence) * Double(h) * Double(d)
                func measure(_ run: () -> Void) -> Measure {
                    for row in rows { (o + row * h * d).update(repeating: .nan, count: h * d) }
                    let t = seconds(run)
                    var got = [Float](repeating: 0, count: vectors * d)
                    for (i, row) in rows.enumerated() {
                        for head in 0..<h { for x in 0..<d { got[x * vectors + i * h + head] = o[row * h * d + head * d + x] } }
                    }
                    return judge(got, exact, columns: vectors, rate: flops / t / 1e12, threshold: fp32Threshold)
                }
                let graph: Measure = try {
                    let attention = Attention(device: bench.device, queue: bench.queue, heads: h, sequence: sequence, headDim: d)
                    try bench.check()
                    return measure { _ = attention.run(q: qb, k: kb, v: vb, into: ob) }
                }()
                try bench.check()
                let flash = (try? FlashMatrix(device: bench.device, queue: bench.queue, heads: h, sequence: sequence,
                                              headDim: d)).map { kernel in measure { kernel.run(q: qb, k: kb, v: vb, into: ob) } }
                try bench.check()
                // At the reference share, not at whatever the settings say: the map must measure the same cut
                // whatever it decided last time.
                var parts: AttentionCapacity.SplitParts?
                let split = (try? HeadSplitAttention(device: bench.device, queue: bench.queue, heads: h, sequence: sequence,
                                                     headDim: d, cpuFraction: HeadSplitAttention.referenceFraction)).map { cut in
                    var best = (gpu: Double.infinity, cpu: Double.infinity, wall: Double.infinity)
                    let measured = measure {
                        let t = DispatchTime.now().uptimeNanoseconds
                        let r = cut.run(q: qb, k: kb, v: vb, into: ob, q: q.floats(), k: k.floats(), v: v.floats(), out: o)
                        let wall = Double(DispatchTime.now().uptimeNanoseconds - t) * 1e-9
                        if wall < best.wall { best = (r.gpu, r.cpu, wall) }
                    }
                    if best.wall.isFinite, cut.cpuHeads > 0 {
                        parts = .init(cpuHeads: cut.cpuHeads, gpuSeconds: best.gpu, cpuSeconds: best.cpu)
                    }
                    return measured
                }
                results.append(AttentionCapacity(model: block.model, heads: h, headDim: d, sequence: sequence,
                                                 graph: graph, flash: flash, headSplit: split, splitParts: parts))
            }
        }
        return results
    }

    /// **The exact attention of `rows` queries of every head**, fp64 on the CPU — `softmax(q·kᵀ/√Dh)·v`
    /// as the reference writes it, over all `S` keys. Laid out `[Dh, rows·H]`, the vector of query `i`
    /// at head `h` in column `i·H + h`.
    package static func exactAttention(q: UnsafePointer<Float>, k: UnsafePointer<Float>, v: UnsafePointer<Float>,
                                       sequence: Int, heads: Int, headDim: Int, rows: [Int]) -> [Double] {
        let vectors = rows.count * heads, width = heads * headDim
        let result = Pages(bytes: vectors * headDim * 8)
        nonisolated(unsafe) let (q, k, v, o) = (q, k, v, result.base.assumingMemoryBound(to: Double.self))
        let scale = 1 / Double(headDim).squareRoot()
        DispatchQueue.concurrentPerform(iterations: vectors) { c in
            let row = rows[c / heads], head = c % heads
            withUnsafeTemporaryAllocation(of: Double.self, capacity: sequence + headDim) { buffer in
                let scores = buffer.baseAddress!, sum = scores + sequence
                let query = q + row * width + head * headDim
                var maximum = -Double.infinity
                for j in 0..<sequence {
                    let key = k + j * width + head * headDim
                    var s = 0.0
                    for x in 0..<headDim { s += Double(query[x]) * Double(key[x]) }
                    scores[j] = s * scale
                    maximum = max(maximum, scores[j])
                }
                var total = 0.0
                for x in 0..<headDim { sum[x] = 0 }
                for j in 0..<sequence {
                    let p = Foundation.exp(scores[j] - maximum)
                    total += p
                    let value = v + j * width + head * headDim
                    for x in 0..<headDim { sum[x] += p * Double(value[x]) }
                }
                for x in 0..<headDim { o[x * vectors + c] = sum[x] / total }
            }
        }
        return Array(UnsafeBufferPointer(start: o, count: vectors * headDim))
    }
}
