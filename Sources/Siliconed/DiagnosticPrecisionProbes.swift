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
