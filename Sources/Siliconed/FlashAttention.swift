import Foundation
import Metal

/// **The SDPA that never materializes its attention matrix.**
///
/// This is the **only place in the project where a homegrown kernel has a
/// reason to exist** — we know because the two neighbouring avenues were closed cleanly by
/// measurement: the homegrown GEMM (register cliff, structural), and the API-level split of the SDPA
/// (exact, and 20 to 26 % slower).
///
/// ## Why `MPSGraph` tops out at 1.65 TFLOP/s
///
/// ```
/// one DiT SDPA at 1024²: S = 4128, 30 heads, Dh = 128
///   compute         4·S²·Dh·H                        = 262 GFLOP  → 75 ms at 3.5 TFLOP/s
///   traffic IF the S×S matrix is materialized
///                   S²·H·4 B written, re-read by the softmax,
///                   re-read by the product with V    ≈ 6.1 GB     → ~61 ms at 100 GB/s
/// ```
///
/// The two terms are of the same order, and that is exactly the factor separating the measured
/// 1.665 from a GEMM's 3.5. The proof that it really is the traffic: in fp16, where it is halved,
/// the throughput rises to 3.02 — even though the fp16 issue ceiling is only 8 % above fp32 (measured).
///
/// ## What this kernel does instead
///
/// A block of queries is processed once, and the keys scroll by: `K` and `V` come in by tiles
/// into threadgroup memory, the scores live in registers for the duration of a block, and the
/// softmax is held **online** — running maximum and running sum, rescaled at each tile. The
/// `S×S` matrix is never written anywhere.
///
/// ## The partitioning, and why this one
///
/// - **`Br ≈ 128` query rows per group.** This is what amortizes re-reading `K` and `V`:
///   at `Br = 8` the 4.2 MB of a head would be re-read 516 times, i.e. 65 GB per call — more than
///   what we are trying to eliminate. At 128, it is 32 times, and the 4.2 MB fit in the 24 MB of
///   last-level cache.
/// - **Four threads per row**, each carrying 32 of the 128 dimensions. The dot product ends
///   with two `simd_shuffle_xor` — the four lanes are consecutive, hence in the same simdgroup.
/// - **`Bc = 16` keys per tile**, which puts `K` and `V` at 8 KiB each: 16 KiB of the 32
///   available, hence two groups per core.
/// - **No `simdgroup_matrix`.** The register cliff was measured at **eight** fp32 accumulators;
///   a `Br × Dh` tiling would hold many more, and fp16 on the operands — which pushes it back to
///   sixteen — would cost precision we wanted to avoid *until we have tried without*.
///
/// ## `R` — the register blocks, and what we expect of them *before* measuring
///
/// A first version established that the kernel's slowness is no longer traffic — it has none left — but the
/// **issue ratio**: in the two hot loops, one threadgroup-memory read for one vector
/// FMA. At best half the instructions compute, and 0.771 TFLOP/s on a chip that
/// issues ~4.6 in fp32 is exactly what that ratio predicts.
///
/// The countermeasure is to make **one thread carry `R` query rows**: the `kt[j][d]` tile is
/// then read once and serves `R` dot products, `vt[j][d]` once and serves `R` FMAs.
///
/// ```
///   R = 1   1 read + 1 FMA          50 % of instructions compute
///   R = 2   1 read + 2 FMA          67 %
///   R = 4   1 read + 4 FMA          80 %
/// ```
///
/// **What we expected, written down before the measurement so that the measurement can refute it**:
/// the gain tops out at ×1.33 at `R = 2` and ×1.6 at `R = 4`, i.e. ~1.03 then ~1.23 TFLOP/s —
/// enough to get closer to `MPSGraph`'s 1.667, not to beat it.
///
/// ## What was measured, and why the default is `R=2, Bc=8, L=8`
///
/// ```
///   R=1, Bc=16, L=4     0.771 TFLOP/s      R=2, Bc=16, L=4   0.387   ← spills
///   R=1, Bc=8,  L=4           0.781              R=4, Bc=16, L=4   0.122   ← spills
///   R=2, Bc=8,  L=8   ← chosen  0.849            R=4, Bc=8,  L=8   0.291   ← spills
/// ```
///
/// **+11 %, and on one condition: the per-thread register budget must not move.** Two rows
/// of sixteen dimensions (`L = 8`) cost what one row of thirty-two cost. As soon as a
/// variant asks for more, the compiler hands `qreg` and `acc` back to private memory and the
/// throughput drops by a factor of 2 to 6.
///
/// **The spill is read off `maxTotalThreadsPerThreadgroup`, and it is read backwards**: at
/// `R=4, Bc=4, L=4` the pipeline grants **832** threads per group where this kernel only gets
/// 448. *A kernel that is suddenly granted more threads has not become frugal — it has
/// spilled.* It is the only register gauge Metal exposes, and it is printed with the
/// tiling for that reason.
///
/// **And the cause named by the first version is not the one that caps.** Two probes (`Probe`) say so:
/// removing *all* threadgroup-memory reads only returns **6 %**, removing the
/// cross-lane reduction 5 %. The "one read per FMA" ratio explained a factor of 2; a factor of
/// four is missing. This kernel reaches **24 % of the issue ceiling that its own instruction
/// count gives it**, and what remains to be explained is a latency, not a throughput.
/// The next step is `simdgroup_matrix`.
package final class FlashAttention {
    /// `Dh = 128` is fixed: it is the DiT's *and* Qwen3's. Making it variable would cost
    /// dynamically sized arrays in registers, that is, a spill to memory.
    package static let headDim = 128
    /// The tiling chosen by measurement — `R=2, Bc=8, L=8`, 0.849 TFLOP/s against 0.771 for the first kernel.
    package static let defaultRows = 2, defaultKeysPerTile = 8, defaultLanes = 8
    /// **`Br` is not fixed: the pipeline says how many threads it tolerates.**
    ///
    /// With 80 floats in registers per thread, the compiler only grants **448** threads per group
    /// and not 512 — asking for more makes the construction fail. So we take the largest `Br`
    /// that is a multiple of eight and fits, and the kernel reads its own group size to spread
    /// the tile loading. *A hard-coded tiling constant is a hypothesis about the
    /// compiler; this one is asked for.*
    private var rowsPerGroup = 0
    private var threadsPerGroup = 0
    /// Query rows carried by a single thread. `1` is the first kernel.
    package let rowsPerThread: Int
    /// Keys per threadgroup-memory tile (`Bc`) and lanes per row (`L`). `16` and `4` are the first kernel's.
    package let keysPerTile: Int
    package let lanes: Int

    /// The source is **generated**, not hard-coded: `R` enters it by interpolation. It is the same
    /// decision as for the non-linearity — what varies from one variant to
    /// the next is a generation parameter, never a runtime branch.
    package static func source(rowsPerThread R: Int, unrollTiles: Bool,
                              keysPerTile BC: Int = 16, lanes L: Int = 4,
                              binaryExponential: Bool = false, probe: Probe = .none) -> String {
        // `%%` marks the TILE loops (`j` over the 16 keys, `i` over the 8 `float4`s);
        // the ROW loops, for their part, always carry their unrolling — without it, `qreg` and
        // `acc` are dynamically indexed and leave the registers.
        let tile = unrollTiles ? "#pragma clang loop unroll(full)" : ""
        // **The dot-product reduction is generated, not written.** The `L` lanes of a row
        // are consecutive threads, hence in the same simdgroup: `log2(L)` `simd_shuffle_xor`s
        // suffice, and their number follows `L`. Hard-coding it would have frozen `L = 4`.
        // **`exp2` instead of `exp`.** The softmax is invariant under a change of base: setting
        // `s' = s·log₂e` and taking `2^(s'−m')` returns exactly `e^(s−m)`, up to rounding.
        // The gain is not in the formula but in the hardware — `exp2` is an instruction of the
        // special-function unit, `exp` a sequence. And there are `Bc` of them per tile **and per
        // lane**: the `L` lanes of a row recompute the same weights, since each needs them
        // for its slice of the accumulator.
        let e = binaryExponential ? "exp2" : "exp"
        // `kt[j][…]` → `kt[0][…]`: the read leaves the loops, the arithmetic stays.
        let key = probe == .withoutTileLoads ? "0" : "j"
        let log2e = binaryExponential ? " * 1.44269504088896340736f" : ""
        let fold = probe == .withoutFold ? "" : (0..<Int(log2(Double(L)))).map {
            "                        t += simd_shuffle_xor(t, \(1 << $0));"
        }.joined(separator: "\n")
        return """
        #include <metal_stdlib>
        using namespace metal;

        #define DH   128
        #define DIMS  (DH / LANES)
        #define BC    \(BC)
        #define LANES \(L)
        #define VEC   (DIMS / 4)
        #define R     \(R)

        struct Params { uint sequence; uint heads; float scale; };

        kernel void flash_sdpa(device const float *q      [[buffer(0)]],
                               device const float *k      [[buffer(1)]],
                               device const float *v      [[buffer(2)]],
                               device float       *o      [[buffer(3)]],
                               constant Params    &p      [[buffer(4)]],
                               // MSL requires position attributes to be ALL scalars or
                               // all vectors of the same width: `uint2` and `uint` cannot coexist.
                               uint2 group     [[threadgroup_position_in_grid]],
                               uint2 local     [[thread_position_in_threadgroup]],
                               uint2 groupSize [[threads_per_threadgroup]]) {
            // **In `float4`.** The scalar version consumed one threadgroup-memory read per
            // FMA and topped out at 0.707 TFLOP/s; here each instruction carries four, and the
            // address computation is divided accordingly.
            threadgroup float4 kt[BC][DH / 4];
            threadgroup float4 vt[BC][DH / 4];

            const uint S = p.sequence;
            const uint stride = p.heads * DH;
            const uint head = group.y;
            const uint tid = local.x;
            const uint width = groupSize.x;            // Br/R · LANES, decided at construction
            const uint threadRows = width / LANES;     // rows led abreast by the group
            const uint row = tid / LANES;              // 0…threadRows-1
            const uint part = tid % LANES;             // 0…3, its 32 dimensions
            const uint base4 = part * VEC;             // its 8 `float4`s
            // **A thread's `R` rows are spaced `threadRows` apart, not consecutive.** Each
            // sub-read then keeps exactly the coalesced pattern of the R = 1 version: neighbouring
            // threads read neighbouring addresses.
            const uint first = group.x * (threadRows * R) + row;

            float4 qreg[R][VEC];
            bool live[R];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < R; ++r) {
                uint s = first + r * threadRows;
                live[r] = s < S;
                device const float4 *src = (device const float4 *)
                    (q + (live[r] ? s : S - 1) * stride + head * DH) + base4;
                \(tile)
                for (uint i = 0; i < VEC; ++i) qreg[r][i] = src[i];
            }

            float4 acc[R][VEC];
            float runningMax[R], runningSum[R];
            #pragma clang loop unroll(full)
            for (uint r = 0; r < R; ++r) {
                \(tile)
                for (uint i = 0; i < VEC; ++i) acc[r][i] = 0;
                runningMax[r] = -INFINITY;
                runningSum[r] = 0;
            }

            for (uint tile = 0; tile < S; tile += BC) {
                // Tiles are loaded by several threads: 2048 floats per tensor, `width` threads.
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint i = tid; i < BC * (DH / 4); i += width) {
                    uint kr = i / (DH / 4), kd = i % (DH / 4);
                    uint g = tile + kr;
                    bool ok = g < S;
                    device const float4 *ks = (device const float4 *)(k + g * stride + head * DH);
                    device const float4 *vs = (device const float4 *)(v + g * stride + head * DH);
                    kt[kr][kd] = ok ? ks[kd] : float4(0);
                    vt[kr][kd] = ok ? vs[kd] : float4(0);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // The tile's scores. **The key is read once and serves the `R` rows** —
                // that is the whole point of the register block. The four lanes of the same row
                // are then summed without going through memory.
                float scores[R][BC];
                \(tile)
                for (uint j = 0; j < BC; ++j) {
                    float partial[R];
                    #pragma clang loop unroll(full)
                    for (uint r = 0; r < R; ++r) partial[r] = 0;
                    \(tile)
                    for (uint i = 0; i < VEC; ++i) {
                        float4 key = kt[\(key)][base4 + i];
                        #pragma clang loop unroll(full)
                        for (uint r = 0; r < R; ++r) partial[r] += dot(qreg[r][i], key);
                    }
                    bool inside = (tile + j) < S;
                    #pragma clang loop unroll(full)
                    for (uint r = 0; r < R; ++r) {
                        float t = partial[r];
        \(fold)
                        scores[r][j] = inside ? t * p.scale\(log2e) : -INFINITY;
                    }
                }

                // The online softmax: a single rescale per tile and per row, not one per key.
                #pragma clang loop unroll(full)
                for (uint r = 0; r < R; ++r) {
                    float tileMax = runningMax[r];
                    \(tile)
                    for (uint j = 0; j < BC; ++j) tileMax = max(tileMax, scores[r][j]);
                    float correction = \(e)(runningMax[r] - tileMax);
                    float sum = runningSum[r] * correction;
                    \(tile)
                    for (uint j = 0; j < BC; ++j) {
                        float weight = \(e)(scores[r][j] - tileMax);
                        scores[r][j] = weight;
                        sum += weight;
                    }
                    \(tile)
                    for (uint i = 0; i < VEC; ++i) acc[r][i] *= correction;
                    runningMax[r] = tileMax;
                    runningSum[r] = sum;
                }

                // **`vt[j][d]` read once, `R` FMAs.** The loop order is this one and no
                // other: `j` then `d` outside, the rows inside.
                \(tile)
                for (uint j = 0; j < BC; ++j) {
                    \(tile)
                    for (uint i = 0; i < VEC; ++i) {
                        float4 value = vt[\(key)][base4 + i];
                        #pragma clang loop unroll(full)
                        for (uint r = 0; r < R; ++r) acc[r][i] += scores[r][j] * value;
                    }
                }
            }

            #pragma clang loop unroll(full)
            for (uint r = 0; r < R; ++r) {
                if (!live[r]) continue;
                uint s = first + r * threadRows;
                device float4 *dst = (device float4 *)(o + s * stride + head * DH) + base4;
                float inverse = 1.0f / runningSum[r];
                \(tile)
                for (uint i = 0; i < VEC; ++i) dst[i] = acc[r][i] * inverse;
            }
        }
        """
    }

    /// **The probes do not compute the SDPA, and that is their reason for being.**
    ///
    /// A kernel at 24 % of its issue ceiling has a cost that no instruction count
    /// explains. To find out which, we remove one thing at a time and watch the clock — the
    /// output becomes wrong, and the bench excludes them from its verdict instead of pretending.
    /// It is the same family as `SILICONED_AMX_SPIN`: deliberately wrong code, kept because
    /// it answers a question that the right code cannot ask.
    package enum Probe {
        /// The real kernel.
        case none
        /// **The tiles are no longer re-read at each key**: the value of key 0 serves for
        /// all of them. Same arithmetic instruction count, eight times fewer threadgroup-memory
        /// addresses. If the clock collapses, it is the reads that cost.
        case withoutTileLoads
        /// **The cross-lane reduction is skipped.** Each lane keeps its partial product. If the clock
        /// collapses, it is the `simd_shuffle_xor`s that cost.
        case withoutFold
    }

    private struct Params { var sequence: UInt32; var heads: UInt32; var scale: Float }

    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    package let heads: Int
    package let sequence: Int
    /// What the pipeline actually granted — to be displayed, because it is the effective tiling.
    package var tiling: String {
        "R=\(rowsPerThread), Bc=\(keysPerTile), L=\(lanes), Br=\(rowsPerGroup), "
        + "\(threadsPerGroup) threads/group"
    }

    package enum Failure: Error, CustomStringConvertible {
        case unsupported(String)
        package var description: String {
            switch self { case .unsupported(let why): return "flash SDPA: \(why)" }
        }
    }

    package init(device: MTLDevice, queue: MTLCommandQueue, heads: Int, sequence: Int,
                headDim: Int, rowsPerThread: Int? = nil,
                keysPerTile: Int = FlashAttention.defaultKeysPerTile,
                lanes: Int = FlashAttention.defaultLanes,
                unrollTiles: Bool = false) throws {
        guard headDim == FlashAttention.headDim else {
            throw Failure.unsupported("Dh = \(headDim), the kernel is written for \(FlashAttention.headDim)")
        }
        let R = rowsPerThread ?? EngineSettings.effective.flashRows ?? FlashAttention.defaultRows
        guard R >= 1, R <= 8 else {
            throw Failure.unsupported("R = \(R) outside 1…8")
        }
        guard [4, 8, 16].contains(lanes), FlashAttention.headDim % (4 * lanes) == 0 else {
            throw Failure.unsupported("L = \(lanes): needs 4, 8 or 16 lanes, and `Dh/L` a multiple of 4")
        }
        self.rowsPerThread = R
        self.keysPerTile = keysPerTile
        self.lanes = lanes
        self.queue = queue
        self.heads = heads
        self.sequence = sequence
        let library = try device.makeLibrary(source: FlashAttention.source(rowsPerThread: R, unrollTiles: unrollTiles,
                                                 keysPerTile: keysPerTile, lanes: lanes),
                                             options: nil)
        guard let function = library.makeFunction(name: "flash_sdpa") else {
            throw Failure.unsupported("flash_sdpa not found")
        }
        pipeline = try device.makeComputePipelineState(function: function)
        // **We aim for `Br ≈ 128` whatever `R` is**, because it is `Br` — not `R` — that fixes
        // how many times `K` and `V` are re-read from DRAM. At large `R`, the group is therefore
        // narrower, not wider. The pipeline has the last word: if it grants fewer threads
        // than requested, `Br` goes down, and the printed tiling says so.
        let permitted = pipeline.maxTotalThreadsPerThreadgroup
        let wanted = max(lanes, (128 / R) * lanes)
        let width = min(permitted, wanted) / lanes * lanes
        threadsPerGroup = width
        rowsPerGroup = (width / lanes) * R
        guard width >= lanes else {
            throw Failure.unsupported("\(permitted) threads per group are not enough")
        }
    }

    /// Inputs and output in **fp32**, `[S, H·Dh]` — the format the engine already holds, hence no
    /// transposition either way. Returns the GPU time.
    @discardableResult
    package func run(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, into out: MTLBuffer) -> Double {
        let commands = queue.makeCommandBuffer()!
        let encoder = commands.makeComputeCommandEncoder()!
        encoder.label = "flash_sdpa"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        var params = Params(sequence: UInt32(sequence), heads: UInt32(heads),
                            scale: 1 / Float(FlashAttention.headDim).squareRoot())
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 4)
        let blocks = (sequence + rowsPerGroup - 1) / rowsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: blocks, height: heads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup,
                                                                    height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "flash S=\(sequence) H=\(heads) R=\(rowsPerThread)")
    }
}
