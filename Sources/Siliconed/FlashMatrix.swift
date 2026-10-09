import Foundation
import Metal

/// **The flash SDPA on the matrix units.** This is the countermeasure named by the tiling measurement of
/// `FlashAttention`, and the only one that attacks what its probes left unexplained.
///
/// ## What the tiling measurement left, and why the instruction family has to change
///
/// The scalar kernel (`FlashAttention`) reaches **0.849 TFLOP/s** against `MPSGraph`'s 1.667, and its
/// two probes refuted the cause it was blamed on: removing *all* its threadgroup-memory reads
/// only returns 6 %. It reaches 24 % of the issue ceiling that its own instruction
/// count gives it, and what is missing is a latency that no tiling setting
/// recovers. `simdgroup_matrix`es are not a setting: they replace eight scalar FMAs
/// with one instruction, and they have their own data path.
///
/// ## The hard constraint, and it comes from a measurement — not from an article
///
/// ```
///   tile         accumulators      fp32          verdict
///   16×16/sg          4           1.698
///   16×32/sg          8           2.440   ← the best fp32 tiling measured
///   16×48/sg         12           0.141   ← the cliff, and it falls at TWELVE
///   32×32/sg         16           0.075
/// ```
///
/// **Eight accumulators, not one more.** A `simdgroup_float8x8` takes two registers per lane;
/// beyond eight, the compiler spills and the throughput is divided by twenty. Yet a flash tile
/// needs its output accumulators **permanently** — they span the whole key loop — and it needs
/// others for the scores. Hence everything that follows.
///
/// ## The partitioning that comes out of it, and there are not many options
///
/// A simdgroup carries `SR` query rows × `DS` output dimensions, i.e. `SR·DS/64`
/// accumulators — so we take `8×64` or `16×32`, the only two that give exactly eight.
/// It then holds only a **slice** of the 128 dimensions: it can only compute a **partial**
/// score. The `128/DS` simdgroups of the same row block exchange their partials through
/// threadgroup memory and the total is reassembled there — that is the price of the register budget, and it
/// is small: `SR × Bc` floats per simdgroup and per tile.
///
/// **`Br` is measured, and the reasoning that predicted it was wrong.** It was expected to be large: at
/// `Br = 128`, `K` and `V` are only re-read 32 times per head (~4 GB per call) against 65 times at
/// `Br = 64` (8.2 GB, i.e. *more* than `MPSGraph`'s 6.1 GB). Swept, the optimum is nonetheless
/// **`Br = 64`** — 156 ms against 175 at 96 rows and 209 at 112. This kernel is therefore not limited by
/// DRAM but by occupancy and barriers: sixteen simdgroups per group is the point where the
/// two balance out, and beyond it the barrier costs more than the re-reading saves.
///
/// The tiling chosen, measured among twenty: **`8×64/sg`, `Bc = 16`, 8 blocks, `K` and `V` in tiles** —
/// 512 threads, `Br = 64`, **1.673 TFLOP/s**.
///
/// ## The softmax rescaling, which would cost half the kernel if it were unconditional
///
/// When the running maximum rises, the accumulators have to be rescaled. On scalar
/// registers that is a multiplication; on a `simdgroup_matrix`, whose distribution
/// of elements among lanes is unspecified, one has to **multiply by a diagonal** —
/// eight more matrix products per tile, against sixteen useful ones. **50 % overhead.**
///
/// The countermeasure is that the maximum almost never rises after the first tiles: we test it,
/// and only rescale if it moved. The flag is written to threadgroup memory by the thread that
/// does the softmax, hence read uniformly by the whole simdgroup — no divergence.
///
/// ## The sequence tail, handled by overlap rather than by mask
///
/// `S = 4128` is not a multiple of 128. Rather than masking — which would require putting
/// `Q` in threadgroup memory so as to fill it with zeros, and that does not fit — the last block
/// is **brought back within bounds** (`min(wanted, S - Br)`). A few rows are then computed
/// twice, by two different groups, and written twice. This is harmless **because
/// a row's computation does not depend on the block carrying it**: same tiles, same order, same
/// bits. The overhead is 2.3 % at 1024².
package final class FlashMatrix {
    package static let headDim = 128

    package let rowsPerSimdgroup: Int      // SR
    package let dimsPerSimdgroup: Int      // DS
    package let keysPerTile: Int           // Bc
    package let blocksPerGroup: Int
    package let stageKeys: Bool
    package let stageValues: Bool
    /// Operands in `half`, accumulators in fp32. The `q`, `k`, `v` buffers must then hold
    /// half-precision values — it is up to the caller to convert them, once, off the hot path.
    package let mixed: Bool
    package let heads: Int
    package let sequence: Int

    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let threadsPerGroup: Int
    private let rowsPerGroup: Int
    private let permitted: Int

    package var tiling: String {
        "\(rowsPerSimdgroup)×\(dimsPerSimdgroup)/sg, \(rowsPerSimdgroup * dimsPerSimdgroup / 64) acc, "
        + "Bc=\(keysPerTile), Br=\(rowsPerGroup), \(threadsPerGroup)/\(permitted) threads"
        + ", tiles " + (stageKeys ? "K" : "-") + (stageValues ? "V" : "-")
        + (mixed ? ", half operands" : "")
    }

    package enum Failure: Error, CustomStringConvertible {
        case unsupported(String)
        package var description: String {
            switch self { case .unsupported(let why): return "matrix SDPA: \(why)" }
        }
    }

    /// **Operands in half precision, accumulators in fp32 — the `mixed`.**
    ///
    /// It is the only door the half-precision decision left open, and a measurement found its hinge on this
    /// machine: with `half` operands, the register cliff no longer falls at twelve
    /// accumulators but at thirty-two, and the raw throughput goes from **2.44 to 3.27 TFLOP/s**.
    ///
    /// **This is not the compromise of `SILICONED_SDPA_FP16`**, which is ruled out by decision: there,
    /// `MPSGraph` accumulates *also* in fp16 and `model_out` comes out at 1.26·10⁻³. Here accumulation
    /// stays exact — only `Q`, `K`, `V` and the softmax weights go through `half` before
    /// entering the matrix unit, and everything that is summed stays in fp32.
    ///
    /// What it costs in accuracy is therefore the rounding of the operands, and nothing else. What it
    /// brings is twofold: half the operand traffic — on an engine limited by DRAM —
    /// and half the registers, hence a wider tiling.
    package static func source(rowsPerSimdgroup SR: Int, dimsPerSimdgroup DS: Int,
                              keysPerTile BC: Int, blocksPerGroup BLOCKS: Int,
                              stageKeys: Bool, stageValues: Bool, mixed: Bool = false) -> String {
        let typeOp = mixed ? "half" : "float"
        let matOp = mixed ? "simdgroup_half8x8" : "simdgroup_float8x8"
        let halves = FlashMatrix.headDim / DS      // simdgroups per row block
        let rowTiles = SR / 8                      // RT
        let dimTiles = DS / 8                      // DT
        let keyTiles = BC / 8                      // JT
        let simdgroups = BLOCKS * halves
        let width = simdgroups * 32
        // One thread per cell when possible, otherwise several cells per thread. The two
        // tilings that fit in registers — `8×64` and `16×32` — land exactly on one.
        let cells = max(1, (SR * BC) / (halves * 32))
        // **`simdgroup_load` reads DRAM as well as threadgroup memory**, and the scalar
        // kernel only put `K` and `V` in tiles because it re-read them lane by lane.
        // Removing them eliminates two barriers per tile and frees `Bc × 1 KiB` of threadgroup
        // memory — that is, what capped `Bc`.
        // **`K` and `V` are not decided together.** Putting both in tiles costs
        // `2 × Bc` KiB of threadgroup memory, and that is what caps the number of blocks — hence
        // occupancy. Removing both is slower (measured). Separating them lets us choose.
        let tiles = (stageKeys ? "            threadgroup \(typeOp) kt[BC][DH];\n" : "")
                  + (stageValues ? "            threadgroup \(typeOp) vt[BC][DH];\n" : "")
        let lines = (stageKeys ? "                    kt[r][c] = g < S ? k[g * stride + head * DH + c] : 0;\n" : "")
                  + (stageValues ? "                    vt[r][c] = g < S ? v[g * stride + head * DH + c] : 0;\n" : "")
        let load = lines.isEmpty ? "" : """
                for (uint i = tid; i < BC * DH; i += WIDTH) {
                    uint r = i / DH, c = i % DH;
                    uint g = tile + r;
        \(lines)        }
                threadgroup_barrier(mem_flags::mem_threadgroup);
        """
        // Without tiles, the tail is handled like the rows': the key block is brought back within
        // bounds. The tail's scores are masked to `-INFINITY` by the softmax, so their
        // weight is zero — but the read has to be **valid**, otherwise a non-finite pattern
        // multiplied by zero would return a `NaN`.
        let keyMat = stageKeys
            ? "simdgroup_load(km, &kt[j * 8][dim0 + d], DH, ulong2(0, 0), true);"
            : "simdgroup_load(km, k + min(tile + j * 8, S - 8) * stride + head * DH + dim0 + d, "
              + "stride, ulong2(0, 0), true);"
        let valueMat = stageValues
            ? "simdgroup_load(vm, &vt[j * 8][dim0 + t * 8], DH);"
            : "simdgroup_load(vm, v + min(tile + j * 8, S - 8) * stride + head * DH + dim0 + t * 8, "
              + "stride);"
        let trailing = lines.isEmpty ? "" : "threadgroup_barrier(mem_flags::mem_threadgroup);"
        return """
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;

        #define DH      128
        #define DS      \(DS)
        #define SR      \(SR)
        #define BC      \(BC)
        #define HALVES  \(halves)
        #define BLOCKS  \(BLOCKS)
        #define RT      \(rowTiles)
        #define DT      \(dimTiles)
        #define JT      \(keyTiles)
        #define WIDTH   \(width)
        #define BR      (SR * BLOCKS)
        // Cells (row, key) per thread, and lanes carrying the same row.
        #define CELLS   \(cells)
        #define ROWLANES (BC / CELLS)

        struct Params { uint sequence; uint heads; float scale; };

        kernel void flash_sdpa(device const \(typeOp) *q      [[buffer(0)]],
                               device const \(typeOp) *k      [[buffer(1)]],
                               device const \(typeOp) *v      [[buffer(2)]],
                               device float       *o      [[buffer(3)]],
                               constant Params    &p      [[buffer(4)]],
                               uint2 group  [[threadgroup_position_in_grid]],
                               uint  tid    [[thread_index_in_threadgroup]],
                               uint  sgid   [[simdgroup_index_in_threadgroup]],
                               uint  lane   [[thread_index_in_simdgroup]]) {
            \(tiles)
            // The partial scores, one slice of dimensions per simdgroup, stitched together by the softmax.
            // **The weights are written over the partial scores.** Each cell (row,
            // key) belongs to one thread and only one: it reads the `HALVES` partials of ITS cell,
            // then writes the weight at the same place. No thread reads another's cell, so
            // the aliasing needs no barrier — and it gives back `BLOCKS × SR × Bc` floats, which
            // are exactly what capped the number of blocks, hence occupancy.
            threadgroup float partial[BLOCKS][HALVES][SR][BC];
            \(mixed ? "threadgroup half wt[BLOCKS][SR][BC];\n            #define weights(b) wt[b]"
                     : "#define weights(b) partial[b][0]")
            // The rescaling diagonal, one 8×8 matrix per row tile.
            threadgroup float diagonal[BLOCKS][RT][64];
            threadgroup float runMax[BLOCKS][SR];
            threadgroup float runSum[BLOCKS][SR];
            threadgroup uint  rescale[BLOCKS];

            const uint S = p.sequence;
            const uint stride = p.heads * DH;
            const uint head = group.y;
            const uint block = sgid / HALVES;
            const uint slice = sgid % HALVES;
            const uint dim0 = slice * DS;
            // The tail: the block is brought back within bounds rather than masked (see the header).
            const uint wanted = group.x * BR + block * SR;
            const uint rowBase = min(wanted, S - SR);

            for (uint i = tid; i < BLOCKS * SR; i += WIDTH) {
                runMax[i / SR][i % SR] = -INFINITY;
                runSum[i / SR][i % SR] = 0;
            }
            // The diagonal is zero everywhere except on its diagonal, and only that will be rewritten.
            for (uint i = tid; i < BLOCKS * RT * 64; i += WIDTH) {
                diagonal[i / (RT * 64)][(i / 64) % RT][i % 64] = 0;
            }

            simdgroup_float8x8 acc[RT][DT];
            for (uint r = 0; r < RT; ++r)
                for (uint t = 0; t < DT; ++t)
                    acc[r][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint tile = 0; tile < S; tile += BC) {
                if (tid < BLOCKS) rescale[tid] = 0;
                \(load)

                // ── the scores, over the simdgroup's slice of dimensions ─────────────────────
                simdgroup_float8x8 s[RT][JT];
                for (uint r = 0; r < RT; ++r)
                    for (uint j = 0; j < JT; ++j)
                        s[r][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
                // **Do not hoist the loads out of the loops.** A tile takes 56
                // matrix loads for 32 products, and it is tempting to pull `Q` and `P` out
                // of the inner loops. Measured: **156.6 → 221.1 ms**, and the pipeline drops from
                // 1,024 to 896 threads. Keeping them alive costs more than re-reading them — it is
                // the same lesson again, on a kernel with not a register left to give.
                for (uint d = 0; d < DS; d += 8) {
                    for (uint j = 0; j < JT; ++j) {
                        // `Kᵀ`: the transpose is a load flag, not a copy.
                        \(matOp) km;
                        \(keyMat)
                        for (uint r = 0; r < RT; ++r) {
                            \(matOp) qm;
                            simdgroup_load(qm, q + (rowBase + r * 8) * stride + head * DH + dim0 + d,
                                           stride);
                            simdgroup_multiply_accumulate(s[r][j], qm, km, s[r][j]);
                        }
                    }
                }
                for (uint r = 0; r < RT; ++r)
                    for (uint j = 0; j < JT; ++j)
                        simdgroup_store(s[r][j], &partial[block][slice][r * 8][j * 8], BC);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // ── the softmax, ONE THREAD PER CELL (row, key) ──────────────────────────────
                //
                // The first version gave one per **row**: eight threads worked and
                // five hundred and four waited at the barrier, for scalar work comparable to the
                // matrix path framing it. It was half the critical path.
                //
                // Here each thread carries `CELLS` cells, the `BC/CELLS` lanes of a row are
                // consecutive — hence in the same simdgroup — and the softmax's two reductions
                // are done by `simd_shuffle_xor`, without going back through memory.
                {
                    const uint cell = (slice * 32 + lane) * CELLS;
                    const uint r = cell / BC;
                    const uint j0 = cell % BC;
                    float sc[CELLS];
                    for (uint c = 0; c < CELLS; ++c) {
                        float t = 0;
                        for (uint h = 0; h < HALVES; ++h) t += partial[block][h][r][j0 + c];
                        sc[c] = (tile + j0 + c < S) ? t * p.scale : -INFINITY;
                    }
                    float m = sc[0];
                    for (uint c = 1; c < CELLS; ++c) m = max(m, sc[c]);
                    for (uint d = 1; d < ROWLANES; d <<= 1) m = max(m, simd_shuffle_xor(m, d));
                    const float previous = runMax[block][r];
                    const float tileMax = max(previous, m);
                    const float correction = exp(previous - tileMax);
                    float sum = 0;
                    for (uint c = 0; c < CELLS; ++c) {
                        float w = exp(sc[c] - tileMax);
                        weights(block)[r][j0 + c] = w;
                        sum += w;
                    }
                    for (uint d = 1; d < ROWLANES; d <<= 1) sum += simd_shuffle_xor(sum, d);
                    if (j0 == 0) {
                        runMax[block][r] = tileMax;
                        runSum[block][r] = runSum[block][r] * correction + sum;
                        diagonal[block][r / 8][(r % 8) * 9] = correction;
                        // **The rescaling is only paid for when it is warranted.** Unconditional,
                        // it would cost as much as the useful computation — eight more matrix products
                        // per tile, against sixteen useful ones. In practice the maximum freezes after
                        // a few tiles. Several threads may arm the flag: they write
                        // the same value to it.
                        if (correction != 1.0f) rescale[block] = 1;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (rescale[block]) {
                    for (uint r = 0; r < RT; ++r) {
                        simdgroup_float8x8 dm;
                        simdgroup_load(dm, &diagonal[block][r][0], 8);
                        for (uint t = 0; t < DT; ++t) {
                            simdgroup_float8x8 z = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
                            simdgroup_multiply_accumulate(z, dm, acc[r][t], z);
                            acc[r][t] = z;
                        }
                    }
                }

                // ── O += P · V, over the simdgroup's slice of dimensions ─────────────────────
                for (uint t = 0; t < DT; ++t) {
                    for (uint j = 0; j < JT; ++j) {
                        \(matOp) vm;
                        \(valueMat)
                        for (uint r = 0; r < RT; ++r) {
                            \(matOp) pm;
                            simdgroup_load(pm, &weights(block)[r * 8][j * 8], BC);
                            simdgroup_multiply_accumulate(acc[r][t], pm, vm, acc[r][t]);
                        }
                    }
                }
                \(trailing)
            }

            // ── the normalization, by the same diagonal ──────────────────────────────────────
            if (slice == 0 && lane < SR) {
                uint r = lane;
                diagonal[block][r / 8][(r % 8) * 9] = 1.0f / runSum[block][r];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < RT; ++r) {
                simdgroup_float8x8 dm;
                simdgroup_load(dm, &diagonal[block][r][0], 8);
                for (uint t = 0; t < DT; ++t) {
                    simdgroup_float8x8 z = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
                    simdgroup_multiply_accumulate(z, dm, acc[r][t], z);
                    simdgroup_store(z, o + (rowBase + r * 8) * stride + head * DH + dim0 + t * 8,
                                    stride);
                }
            }
        }
        """
    }

    private struct Params { var sequence: UInt32; var heads: UInt32; var scale: Float }

    package init(device: MTLDevice, queue: MTLCommandQueue, heads: Int, sequence: Int, headDim: Int,
                rowsPerSimdgroup: Int = 8, dimsPerSimdgroup: Int = 64,
                keysPerTile: Int = 16, blocksPerGroup: Int = 8,
                stageKeys: Bool = true, stageValues: Bool = true, mixed: Bool = false) throws {
        guard headDim == FlashMatrix.headDim else {
            throw Failure.unsupported("Dh = \(headDim), the kernel is written for 128")
        }
        guard rowsPerSimdgroup % 8 == 0, dimsPerSimdgroup % 8 == 0, keysPerTile % 8 == 0,
              FlashMatrix.headDim % dimsPerSimdgroup == 0 else {
            throw Failure.unsupported("the tiling must be in multiples of 8, and `Dh/DS` an integer")
        }
        let accumulators = rowsPerSimdgroup * dimsPerSimdgroup / 64
        // Measured: the cliff falls at **twelve** accumulators when the operands are fp32, and at
        // **thirty-two** when they are half — because a `simdgroup_float8x8` takes two
        // registers per lane where a `half8x8` takes one. The limit is therefore not the same on
        // both sides, and we refuse out loud rather than return 0.141 TFLOP/s for no
        // visible reason.
        let ceiling = mixed ? 16 : 8
        guard accumulators <= ceiling else {
            throw Failure.unsupported(
                "\(accumulators) accumulators — the limit is \(ceiling) with "
                    + (mixed ? "half operands" : "fp32") + " (M14)")
        }
        self.rowsPerSimdgroup = rowsPerSimdgroup
        self.dimsPerSimdgroup = dimsPerSimdgroup
        self.keysPerTile = keysPerTile
        self.blocksPerGroup = blocksPerGroup
        self.stageKeys = stageKeys
        self.stageValues = stageValues
        self.mixed = mixed
        self.heads = heads
        self.sequence = sequence
        self.queue = queue
        let simdgroups = blocksPerGroup * (FlashMatrix.headDim / dimsPerSimdgroup)
        threadsPerGroup = simdgroups * 32
        rowsPerGroup = rowsPerSimdgroup * blocksPerGroup
        guard threadsPerGroup <= 1024 else {
            throw Failure.unsupported("\(threadsPerGroup) threads requested, the maximum is 1024")
        }
        guard sequence >= rowsPerGroup else {
            throw Failure.unsupported("S = \(sequence) < Br = \(rowsPerGroup): the tail cannot be brought back")
        }
        let library = try device.makeLibrary(
            source: FlashMatrix.source(rowsPerSimdgroup: rowsPerSimdgroup,
                                       dimsPerSimdgroup: dimsPerSimdgroup,
                                       keysPerTile: keysPerTile, blocksPerGroup: blocksPerGroup,
                                       stageKeys: stageKeys, stageValues: stageValues,
                                       mixed: mixed),
            options: nil)
        guard let function = library.makeFunction(name: "flash_sdpa") else {
            throw Failure.unsupported("flash_sdpa not found")
        }
        pipeline = try device.makeComputePipelineState(function: function)
        permitted = pipeline.maxTotalThreadsPerThreadgroup
        guard permitted >= threadsPerGroup else {
            throw Failure.unsupported(
                "the pipeline only grants \(permitted) threads, \(threadsPerGroup) are needed")
        }
    }

    @discardableResult
    package func run(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, into out: MTLBuffer) -> Double {
        let commands = queue.makeCommandBuffer()!
        let encoder = commands.makeComputeCommandEncoder()!
        encoder.label = "flash_sdpa_matrix"
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        var params = Params(sequence: UInt32(sequence), heads: UInt32(heads),
                            scale: 1 / Float(FlashMatrix.headDim).squareRoot())
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 4)
        let blocks = (sequence + rowsPerGroup - 1) / rowsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: blocks, height: heads, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup,
                                                                    height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "matrix flash S=\(sequence) H=\(heads)")
    }
}
