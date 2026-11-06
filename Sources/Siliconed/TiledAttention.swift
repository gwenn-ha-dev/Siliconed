import Foundation
import Metal
import Synchronization

/// **The attention in three stages, by tiles of query rows — or fused, one pass by simdgroup** — three candidate kernels of the capacity
/// map (`AttentionAssignment.Kernel` `tiles`, `tiles16`, `tilesbf16`), none ever the default: the map times them on each Mac and the
/// solver picks; for the two half-precision ones the map also picks the form (three stages, or fused at a measured `AttentionTile`).
///
///     per pass of `headsPerPass` heads and block of `blockRows` query rows (scores [heads, rows, keys] kept under `scoreBytes`):
///       scores  = Q_blk · K_hᵀ                 matmul2d, strided views of the callers' [S, H·Dh] arrays — no copy, no transposition
///       P       = softmax(scale · scores)      one simdgroup a row, sums in fp32, in place (fp32) or written as half (fp16)
///       O_blk   = P · V_h                      matmul2d, written at the head's column offset, row stride H·Dh
///
/// What sets it apart — what an attention written by analogy with the flash kernel or the graph would get wrong:
///
///   - **The products are Metal 4's `matmul2d`** (as `TensorGEMM`), so that where the GPU has matrix accelerators the two products
///     go through them, which `MPSMatrixMultiplication` does not promise; on a GPU without, they run on the shader cores — slower or
///     not, the map says. Only the `slice` API and `tensor_inline` views with strides, the forms every compiler takes.
///   - **The scores are materialized, but one block at a time**: a block of `blockRows × keys` per head (16 MiB by default) stays in the
///     GPU's last-level cache between the three stages; the graph's whole `S² · H` matrix (2 GB at 1 024²) does not. The scale is
///     applied in the softmax, where the products are plain `Q·Kᵀ`.
///   - **fp32 changes nothing but the order of the sums** (`tiles`): the same operands, the same fp32 sums, the softmax in fp32 with
///     `precise::exp`. The fp16 variant (`tilesHalf`) rounds Q, K, V and P to half — once each, nothing else — and sums, scales and
///     normalizes in fp32: operands as the ANE's fp16, sums as P20 requires. Whether a render may run it is one rule
///     (`CapacitySolver.Constraints.allowsHalf`: the ANE allowed and the accelerators on, at every version, the Standard's included —
///     decided 3/11/2026, checked against the reference in bf16 by `--fp16-attention`), whether it should is the map's.
///   - **The bf16 variant** (`tilesbf16`) is the fp16 one with bfloat operands: the range of fp32, three fewer bits — the rounding the
///     Standard may take (`Constraints.allowsRounding`, P20 on bf16 weights), judged against the reference in bf16 like `bf16act`.
///   - **The fused form** (`AttentionTile`, half and bfloat only) keeps nothing between the stages: per simdgroup a block of `rows` query
///     rows, per chunk of `keys` keys S = Q·Kᵀ into a cooperative fp32 tensor, the online softmax in the registers (`reduce_rows` max
///     and sum, fp32), P rounded to the operand type and handed to the second product as its cooperative left input, O += P·V
///     multiply-accumulated in fp32; O is rescaled by the row's factor through threadgroup memory (the two products' layouts differ).
///     Where the GPU has matrix accelerators both products reach them (M5: 18 TFLOP/s against 7 for the three stages); the tile that
///     fits the registers is the machine's (M5: 16 rows — the 16 × 128 accumulator; 32 rows halves the speed), hence measured.
///     P is written at S's own index: the two destinations of one descriptor share the layout (`map_iterator` maps only into a
///     reduction), which the check against fp64 proves on every Mac (`KernelStages`). O's rescale is skipped when no row's maximum
///     moved (every factor exactly 1: same bits), and only the last chunk asks for each score's index. Where the time goes (M5 Pro,
///     S = 4 128, 30 heads): the two products alone 12.0 ms, the softmax ~1.2 ms more, the rounding of Q, K, V before the kernel
///     ~1.1 ms — where the QK-norm and RoPE pass precedes the attention it writes Q and K rounded into `operands` itself (Z, Krea 2,
///     MOSS, Anima), and only V is converted here: its product has too many routes (ANE rows, LoRA added after) to write it rounded. Mapping O's iterator straight into the factor's reduction (no threadgroup memory), P handed through threadgroup
///     memory, two chunks a pass, `exp2` on a folded scale, 8 or 16 simdgroups: all as fast or slower there.
///   - **The output is the next product's operand** (`to_out`, `proj.o`): where that product rounds its activation to bf16
///     (`BFloatKernels`), the last pass — the fused kernel's epilogue, or the three stages' second product — also writes O rounded into
///     the slot the product reads (`run(…rounded:)`, `WeightProducts.RoundedOperand`), with `round_bf16`'s integer sequence on the values
///     it stores: the product encodes no rounding pass, the same bits. The fp32 O is written as before (the ANE's rows, a LoRA read it).
///   - **Any length**: the tensors carry the real extents, so a block of 100 rows or a key length that is no multiple of 32 needs no
///     padding rows. Cross-attention (`keys ≠ sequence`) and grouped k/v heads are the same code; a causal mask is not (the callers
///     that need one never ask).
package final class TiledAttention {
    package enum Precision: Sendable { case fp32, fp16, bf16
        /// The operand type in the kernels' source.
        var metal: String { switch self { case .fp32: "float"; case .fp16: "half"; case .bf16: "bfloat" } }
        var suffix: String { switch self { case .fp32: "f32"; case .fp16: "f16"; case .bf16: "bf16" } }
    }
    package enum Failure: Error, CustomStringConvertible {
        case unsupported(String)
        package var description: String { switch self { case .unsupported(let why): return "tiled attention: \(why)" } }
    }

    /// The scores one pass keeps (fp32, per head and block): sized for the last-level cache of the smallest Mac we map.
    package static let scoreBudget = 16 << 20

    package let heads: Int, kvHeads: Int, sequence: Int, keys: Int, headDim: Int
    /// The heads this instance computes: the first `computed` of `heads` (all of them alone; the GPU's share when the head split composes
    /// with it — the CPU's heads are the last, and their columns are not written here).
    package let computed: Int
    package let precision: Precision
    package let blockRows: Int
    package let headsPerPass: Int
    /// The fused form's tile; `nil`: the three stages.
    package let tile: AttentionTile?

    private let queue: MTLCommandQueue
    private let scoresPipeline, softmaxPipeline, productPipeline: MTLComputePipelineState?
    private let fusedPipeline: MTLComputePipelineState?
    private let halfPipeline: MTLComputePipelineState
    private let scores: MTLBuffer?
    private let probabilities: MTLBuffer?
    private let half: (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer)?

    /// Whether the kernel takes this shape: the product writes the head's columns by tiles of 32, and the grouped k/v heads divide the
    /// q heads. The plan offers nothing else (`AttentionAssignment.isRunnable`).
    package static func runs(headDim: Int, heads: Int, kvHeads: Int) -> Bool {
        headDim >= 32 && headDim % 32 == 0 && kvHeads > 0 && heads % kvHeads == 0
    }

    /// The block and the heads of a pass for a length: whole sequence and several heads when it is short, else a block of the budget.
    package static func tiling(heads: Int, sequence: Int, keys: Int, budget: Int = scoreBudget) -> (rows: Int, heads: Int) {
        let rows = max(1, budget / (4 * max(1, keys)))
        if rows >= sequence { return (sequence, max(1, min(heads, rows / max(1, sequence)))) }
        return (max(64, rows / 64 * 64), 1)
    }

    /// The bytes the instance holds (private buffers), by the same function the memory need reads.
    /// The bytes for the `computed` first heads of `heads` (`nil`: all), grouped k/v heads following (`kvHeads` of `heads`).
    package struct Bytes: Equatable {
        /// The scores of a pass (fp32), the probabilities (half and bf16 only), the half copies of q, k and v (half and bf16 only);
        /// the fused form holds the copies alone.
        package var scores: Int, probabilities: Int, operands: Int
        package var asTotal: Int { scores + probabilities + operands }
    }
    package static func bytes(heads: Int, kvHeads: Int, sequence: Int, keys: Int, headDim: Int, precision: Precision,
                              computed: Int? = nil, tile: AttentionTile? = nil) -> Bytes {
        let computed = computed ?? heads
        let t = tiling(heads: computed, sequence: sequence, keys: keys)
        let group = max(1, heads / max(1, kvHeads))
        let s = Arena.capacity(for: t.heads * t.rows * keys * 4)
        switch precision {
        case .fp32: return Bytes(scores: s, probabilities: 0, operands: 0)
        case .fp16, .bf16:
            let operands = Arena.capacity(for: sequence * computed * headDim * 2)
                + 2 * Arena.capacity(for: keys * max(1, computed / group) * headDim * 2)
            if tile != nil { return Bytes(scores: 0, probabilities: 0, operands: operands) }
            return Bytes(scores: s, probabilities: Arena.capacity(for: t.heads * t.rows * keys * 2), operands: operands)
        }
    }

    package init(device: MTLDevice, queue: MTLCommandQueue, heads: Int, sequence: Int, headDim: Int, keys: Int? = nil, kvHeads: Int? = nil,
                 precision: Precision, computed: Int? = nil, blockRows: Int? = nil, headsPerPass: Int? = nil,
                 tile: AttentionTile? = nil) throws {
        let keys = keys ?? sequence, kvHeads = kvHeads ?? heads
        let computed = computed ?? heads
        guard computed >= 1, computed <= heads, computed % max(1, heads / max(1, kvHeads)) == 0 else {
            throw Failure.unsupported("\(computed) of \(heads) heads")
        }
        guard device.supportsFamily(.metal4) else { throw Failure.unsupported("no Metal 4") }
        guard TiledAttention.runs(headDim: headDim, heads: heads, kvHeads: kvHeads), sequence > 0, keys > 0 else {
            throw Failure.unsupported("head dimension \(headDim), \(heads)/\(kvHeads) heads")
        }
        let library = try TiledAttention.library(device)
        let suffix = precision.suffix
        func pipeline(_ name: String, from library: MTLLibrary = library) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw Failure.unsupported("no function \(name)") }
            return try device.makeComputePipelineState(function: f)
        }
        let tile = precision == .fp32 ? nil : tile
        self.tile = tile
        if let tile {
            fusedPipeline = try pipeline("fused", from: TiledAttention.fusedLibrary(device, precision: precision, tile: tile, headDim: headDim))
            guard fusedPipeline!.maxTotalThreadsPerThreadgroup >= 32 * tile.simdgroups else {
                throw Failure.unsupported("the fused kernel at \(tile) takes \(fusedPipeline!.maxTotalThreadsPerThreadgroup) threads")
            }
            (scoresPipeline, softmaxPipeline, productPipeline) = (nil, nil, nil)
        } else {
            fusedPipeline = nil
            scoresPipeline = try pipeline("tiles_scores_" + suffix)
            softmaxPipeline = try pipeline("tiles_softmax_" + suffix)
            productPipeline = try pipeline("tiles_product_" + suffix)
        }
        halfPipeline = try pipeline(precision == .bf16 ? "tiles_to_bf16" : "tiles_to_half")
        let t = TiledAttention.tiling(heads: computed, sequence: sequence, keys: keys)
        self.computed = computed
        self.blockRows = blockRows ?? t.rows
        self.headsPerPass = headsPerPass ?? t.heads
        self.heads = heads; self.kvHeads = kvHeads; self.sequence = sequence; self.keys = keys; self.headDim = headDim
        self.precision = precision; self.queue = queue
        func buffer(_ n: Int) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: n, options: .storageModePrivate) else { throw Failure.unsupported("no \(n) bytes") }
            return b
        }
        scores = tile == nil ? try buffer(self.headsPerPass * self.blockRows * keys * 4) : nil
        if precision != .fp32 {
            probabilities = tile == nil ? try buffer(self.headsPerPass * self.blockRows * keys * 2) : nil
            let kvComputed = computed / (heads / kvHeads)
            half = (try buffer(sequence * computed * headDim * 2), try buffer(keys * kvComputed * headDim * 2),
                    try buffer(keys * kvComputed * headDim * 2))
        } else { probabilities = nil; half = nil }
    }

    /// **The instance a caller runs an assignment with**, `nil` (said once) when this machine's compiler or the shape refuses it: the
    /// caller then keeps the graph — the plan asked for a kernel the map measured, so a refusal here is a defect worth a line.
    package static func make(_ a: AttentionAssignment, device: MTLDevice, queue: MTLCommandQueue, heads: Int, sequence: Int, headDim: Int,
                             keys: Int? = nil, kvHeads: Int? = nil) -> TiledAttention? {
        do {
            return try TiledAttention(device: device, queue: queue, heads: heads, sequence: sequence, headDim: headDim, keys: keys,
                                      kvHeads: kvHeads, precision: a.tilePrecision, tile: a.tile)
        } catch {
            Warnings.emit("⚠ tiled attention: falling back to MPSGraph at S=\(sequence) — \(error)")
            return nil
        }
    }

    private struct Params {
        var rows: UInt32, keys: UInt32, qWidth: UInt32, kvWidth: UInt32, headDim: UInt32, group: UInt32
        var headBase: UInt32, firstRow: UInt32, count: UInt32, scale: Float, outWidth: UInt32
        /// 1: the output also written rounded to bf16 (`run(…rounded:)`).
        var rounds: UInt32
    }
    private struct FusedParams {
        var sequence: UInt32, keys: UInt32, qWidth: UInt32, kvWidth: UInt32, outWidth: UInt32, group: UInt32; var scale: Float
        var rounds: UInt32
    }

    /// The attentions run on bf16 operands since the process started: the check under `--bf16-attention` says how many it judged.
    package static let bf16Runs = Mutex<Int>(0)
    /// The same on fp16 operands, for `--fp16-attention`.
    package static let fp16Runs = Mutex<Int>(0)

    /// The route the ledger and the logs name.
    package var route: String {
        switch precision { case .fp32: "tiles"; case .fp16: "tiles16"; case .bf16: "tilesbf16" }
    }

    /// `SILICONED_ATTN_ROUND_FUSED=0`: every operand converted before the kernel again, as before their producers wrote them rounded
    /// (a forcing for a measurement; the same bits either way).
    package static let fusesRounding = ProcessInfo.processInfo.environment["SILICONED_ATTN_ROUND_FUSED"] != "0"

    /// The operands an attention may be handed already rounded (`Prerounded`): written by their producer into `operands`.
    package struct Prerounded: OptionSet, Sendable {
        package let rawValue: Int
        package init(rawValue: Int) { self.rawValue = rawValue }
        package static let q = Prerounded(rawValue: 1), k = Prerounded(rawValue: 2), v = Prerounded(rawValue: 4)
    }

    /// **The half-precision copies the kernel reads**, where a producer may write them itself (the QK-norm and RoPE pass,
    /// `QKNormRope.Operand.rounded`) and spare the conversion before the kernel: `q [S, H·Dh]`, `k` and `v [keys, KVH·Dh]`, the callers'
    /// own layouts — so only when this instance computes every head (a head split's GPU share keeps a compact copy of its columns).
    /// `nil` in fp32 or under a split. The same rounding of the same fp32 values either way: the same bits.
    package var operands: (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer)? { computed == heads ? half : nil }

    /// **The output written rounded to bf16 too**, as the next product reads its operand (`run(…rounded:)`,
    /// `WeightProducts.RoundedOperand`): only where this instance computes every head — a head split's GPU share writes some columns.
    package var writesRounded: Bool { computed == heads }

    /// **Runs the attention** over `q [S, H·Dh]`, `k` and `v [keys, KVH·Dh]` (fp32, offset 0), writing `out [S, H·Dh]`: one command
    /// buffer. Returns the GPU seconds. `prerounded`: the operands already in `operands`, whose fp32 source is then not read.
    /// `rounded`: the output also written there rounded to bf16 (`[S, H·Dh]`, the same row stride; `writesRounded` only) — by the
    /// last pass, from the values it stores, with the rounding of `round_bf16` (`BFloatKernels.roundingSource`): the operand the
    /// projection that reads `out` would have rounded itself, bit for bit; the fp32 output is written as before.
    package func run(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, into out: MTLBuffer, prerounded: Prerounded = [],
                     rounded: (buffer: MTLBuffer, offset: Int)? = nil) -> Double {
        let clock = ProductLedger.attention(sequence: sequence, heads: heads, headDim: headDim, route: route, stream: queue.stream)
        defer { clock?.finish() }
        let commands = submit(q: q, k: k, v: v, into: out, prerounded: operands == nil ? [] : prerounded, rounded: rounded)
        let what = "tiled attention S=\(sequence) H=\(computed)/\(heads) \(precision)"
        return queue.stream.settle(commands, label: "sdpa GPU", what: { what })
    }

    /// The same, **committed and not awaited**: the head split computes the CPU's heads meanwhile, then waits on the buffer.
    package func submit(q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, into out: MTLBuffer, prerounded: Prerounded = [],
                        rounded: (buffer: MTLBuffer, offset: Int)? = nil) -> MTLCommandBuffer {
        precondition(rounded == nil || writesRounded, "the output rounded where only some heads are computed")
        if precision == .bf16 { Self.bf16Runs.withLock { $0 += 1 } }
        if precision == .fp16 { Self.fp16Runs.withLock { $0 += 1 } }
        let commands = queue.checkedCommandBuffer()
        let e = commands.checkedEncoder()
        e.label = tile != nil ? "fused_sdpa_" + precision.suffix : precision == .fp32 ? "tiles_sdpa" : "tiles_sdpa_" + precision.suffix
        let outWidth = heads * headDim
        let group = heads / kvHeads
        // The widths the kernels read: the callers' (all heads), or — in half — the compact copy of the computed heads' columns.
        var qWidth = heads * headDim, kvWidth = kvHeads * headDim
        var qs = q, ks = k, vs = v
        if let half {
            let kvComputed = computed / group
            e.setComputePipelineState(halfPipeline)
            // (source, copy, rows, columns converted, source row stride): the first columns of each row, the computed heads'.
            for (operand, from, to, rows, columns, stride) in [(Prerounded.q, q, half.q, sequence, computed * headDim, qWidth),
                                                               (.k, k, half.k, keys, kvComputed * headDim, kvWidth),
                                                               (.v, v, half.v, keys, kvComputed * headDim, kvWidth)]
                where !prerounded.contains(operand) {
                var shape = SIMD3<UInt32>(UInt32(columns), UInt32(stride), UInt32(columns))
                e.setBuffer(from, offset: 0, index: 0); e.setBuffer(to, offset: 0, index: 1)
                e.setBytes(&shape, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 2)
                e.dispatchThreads(MTLSize(width: columns, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            (qs, ks, vs) = (half.q, half.k, half.v)
            (qWidth, kvWidth) = (computed * headDim, kvComputed * headDim)
        }
        if let tile, let fusedPipeline {
            var p = FusedParams(sequence: UInt32(sequence), keys: UInt32(keys), qWidth: UInt32(qWidth), kvWidth: UInt32(kvWidth),
                                outWidth: UInt32(outWidth), group: UInt32(group), scale: 1 / Float(headDim).squareRoot(),
                                rounds: rounded == nil ? 0 : 1)
            e.setComputePipelineState(fusedPipeline)
            e.setBuffer(qs, offset: 0, index: 0); e.setBuffer(ks, offset: 0, index: 1); e.setBuffer(vs, offset: 0, index: 2)
            e.setBuffer(out, offset: 0, index: 3)
            e.setBytes(&p, length: MemoryLayout<FusedParams>.stride, index: 4)
            e.setBuffer(rounded?.buffer ?? out, offset: rounded?.offset ?? 0, index: 5)  // read only where `rounds`
            e.setThreadgroupMemoryLength(tile.simdgroups * tile.rows * 4, index: 0)
            let blocks = (sequence + tile.rows * tile.simdgroups - 1) / (tile.rows * tile.simdgroups)
            e.dispatchThreadgroups(MTLSize(width: 1, height: blocks, depth: computed),
                                   threadsPerThreadgroup: MTLSize(width: 32 * tile.simdgroups, height: 1, depth: 1))
            e.endEncoding()
            commands.commit()
            return commands
        }
        guard let scores, let scoresPipeline, let softmaxPipeline, let productPipeline else { preconditionFailure("three stages unbuilt") }
        var first = 0
        while first < sequence {
            let rows = min(blockRows, sequence - first)
            var head = 0
            while head < computed {
                let count = min(headsPerPass, computed - head)
                var p = Params(rows: UInt32(rows), keys: UInt32(keys), qWidth: UInt32(qWidth), kvWidth: UInt32(kvWidth),
                               headDim: UInt32(headDim), group: UInt32(heads / kvHeads), headBase: UInt32(head),
                               firstRow: UInt32(first), count: UInt32(count), scale: 1 / Float(headDim).squareRoot(),
                               outWidth: UInt32(outWidth), rounds: rounded == nil ? 0 : 1)
                e.setComputePipelineState(scoresPipeline)
                e.setBuffer(qs, offset: 0, index: 0); e.setBuffer(ks, offset: 0, index: 1); e.setBuffer(scores, offset: 0, index: 2)
                e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
                e.dispatchThreadgroups(MTLSize(width: (keys + 31) / 32, height: (rows + 63) / 64, depth: count),
                                       threadsPerThreadgroup: MTLSize(width: scoresPipeline.threadExecutionWidth * 4, height: 1, depth: 1))
                e.setComputePipelineState(softmaxPipeline)
                e.setBuffer(scores, offset: 0, index: 0)
                e.setBuffer(probabilities ?? scores, offset: 0, index: 1)
                e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
                e.dispatchThreadgroups(MTLSize(width: (rows * count + 7) / 8, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                e.setComputePipelineState(productPipeline)
                e.setBuffer(probabilities ?? scores, offset: 0, index: 0); e.setBuffer(vs, offset: 0, index: 1)
                e.setBuffer(out, offset: 0, index: 2)
                e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 3)
                e.setBuffer(rounded?.buffer ?? out, offset: rounded?.offset ?? 0, index: 4)  // read only where `rounds`
                e.dispatchThreadgroups(MTLSize(width: headDim / 32, height: (rows + 63) / 64, depth: count),
                                       threadsPerThreadgroup: MTLSize(width: productPipeline.threadExecutionWidth * 4, height: 1, depth: 1))
                head += count
            }
            first += rows
        }
        e.endEncoding()
        commands.commit()
        return commands
    }

    // MARK: The kernels

    private static let libraries = Mutex<[UInt64: MTLLibrary]>([:])
    private static func library(_ device: MTLDevice) throws -> MTLLibrary {
        if let known = libraries.withLock({ $0[device.registryID] }) { return known }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        options.mathMode = .safe
        let built: MTLLibrary
        do { built = try device.makeLibrary(source: source, options: options) } catch {
            throw Failure.unsupported("the compiler refused the source — \(error.localizedDescription)")
        }
        libraries.withLock { $0[device.registryID] = built }
        return built
    }

    static let source = """
    #include <metal_stdlib>
    #include <metal_tensor>
    #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    using namespace metal;
    using namespace mpp::tensor_ops;

    struct Params {
        uint rows, keys, qWidth, kvWidth, headDim, group, headBase, firstRow, count;
        float scale;
        uint outWidth;
        uint rounds;
    };

    \(BFloatKernels.roundingSource)

    template <typename Op, typename SA, typename SB, typename SC>
    inline void tile(Op op, SA a, SB b, SC c) {
        auto t = op.template get_destination_cooperative_tensor<SA, SB, float>();
        for (uint16_t i = 0; i < t.get_capacity(); ++i) { if (t.is_valid_element(i)) t[i] = 0; }
        op.run(a, b, t);
        t.store(c);
    }

    // scores[z, r, j] = Σ_d Q[first + r, head·Dh + d] · K[j, kvhead·Dh + d]. Extents innermost first.
    template <typename T>
    inline void scores_body(device T *q, device T *k, device float *s, constant Params &p, uint3 g) {
        const int head = int(p.headBase + g.z), kvHead = head / int(p.group);
        const int rows = int(p.rows), keys = int(p.keys), dim = int(p.headDim);
        tensor<device T, dextents<int32_t, 2>, tensor_inline> A(q + ulong(p.firstRow) * p.qWidth + ulong(head) * dim,
                                                               dextents<int32_t, 2>(dim, rows), array<int32_t, 2>({1, int(p.qWidth)}));
        tensor<device T, dextents<int32_t, 2>, tensor_inline> B(k + ulong(kvHead) * dim,
                                                               dextents<int32_t, 2>(dim, keys), array<int32_t, 2>({1, int(p.kvWidth)}));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(s + ulong(g.z) * rows * keys, dextents<int32_t, 2>(keys, rows));
        constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 32, y = g.y * 64;
        tile(op, A.slice(0, y), B.slice(0, x), C.slice(x, y));
    }
    kernel void tiles_scores_f32(device float *q [[buffer(0)]], device float *k [[buffer(1)]], device float *s [[buffer(2)]],
                                 constant Params &p [[buffer(3)]], uint3 g [[threadgroup_position_in_grid]]) { scores_body<float>(q, k, s, p, g); }
    kernel void tiles_scores_f16(device half *q [[buffer(0)]], device half *k [[buffer(1)]], device float *s [[buffer(2)]],
                                 constant Params &p [[buffer(3)]], uint3 g [[threadgroup_position_in_grid]]) { scores_body<half>(q, k, s, p, g); }
    kernel void tiles_scores_bf16(device bfloat *q [[buffer(0)]], device bfloat *k [[buffer(1)]], device float *s [[buffer(2)]],
                                  constant Params &p [[buffer(3)]], uint3 g [[threadgroup_position_in_grid]]) { scores_body<bfloat>(q, k, s, p, g); }

    // One simdgroup a row of the pass's [count · rows, keys] scores: softmax(scale · s), sums in fp32.
    kernel void tiles_softmax_f32(device float *s [[buffer(0)]], device float *unused [[buffer(1)]], constant Params &p [[buffer(3)]],
                                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
        const uint row = tg * 8 + sg;
        if (row >= p.rows * p.count) return;
        device float *x = s + ulong(row) * p.keys;
        float m = -INFINITY;
        for (uint c = lane; c < p.keys; c += 32) m = max(m, x[c] * p.scale);
        m = simd_max(m);
        float sum = 0.0f;
        for (uint c = lane; c < p.keys; c += 32) { const float e = precise::exp(x[c] * p.scale - m); x[c] = e; sum += e; }
        const float inverse = 1.0f / simd_sum(sum);
        for (uint c = lane; c < p.keys; c += 32) x[c] *= inverse;
    }
    template <typename T>
    inline void softmax_rounded(device const float *s, device T *out, constant Params &p, uint tg, uint sg, uint lane) {
        const uint row = tg * 8 + sg;
        if (row >= p.rows * p.count) return;
        const ulong base = ulong(row) * p.keys;
        float m = -INFINITY;
        for (uint c = lane; c < p.keys; c += 32) m = max(m, s[base + c] * p.scale);
        m = simd_max(m);
        float sum = 0.0f;
        for (uint c = lane; c < p.keys; c += 32) sum += precise::exp(s[base + c] * p.scale - m);
        const float inverse = 1.0f / simd_sum(sum);
        for (uint c = lane; c < p.keys; c += 32) out[base + c] = T(precise::exp(s[base + c] * p.scale - m) * inverse);
    }
    kernel void tiles_softmax_f16(device const float *s [[buffer(0)]], device half *out [[buffer(1)]], constant Params &p [[buffer(3)]],
                                  uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) { softmax_rounded<half>(s, out, p, tg, sg, lane); }
    kernel void tiles_softmax_bf16(device const float *s [[buffer(0)]], device bfloat *out [[buffer(1)]], constant Params &p [[buffer(3)]],
                                   uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) { softmax_rounded<bfloat>(s, out, p, tg, sg, lane); }

    // O[first + r, head·Dh + n] = Σ_j P[z, r, j] · V[j, kvhead·Dh + n], written where the head's columns are — and, where `rounds`,
    // rounded to bf16 into `r` at the same index, from the values stored.
    template <typename T>
    inline void product_body(device T *pr, device T *v, device float *o, device ushort *r, constant Params &p, uint3 g) {
        const int head = int(p.headBase + g.z), kvHead = head / int(p.group);
        const int rows = int(p.rows), keys = int(p.keys), dim = int(p.headDim);
        tensor<device T, dextents<int32_t, 2>, tensor_inline> A(pr + ulong(g.z) * rows * keys, dextents<int32_t, 2>(keys, rows));
        tensor<device T, dextents<int32_t, 2>, tensor_inline> B(v + ulong(kvHead) * dim,
                                                               dextents<int32_t, 2>(dim, keys), array<int32_t, 2>({1, int(p.kvWidth)}));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> C(o + ulong(p.firstRow) * p.outWidth + ulong(head) * dim,
                                                                   dextents<int32_t, 2>(dim, rows), array<int32_t, 2>({1, int(p.outWidth)}));
        constexpr auto d = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<d, execution_simdgroups<4>> op;
        const int x = g.x * 32, y = g.y * 64;
        auto As = A.slice(0, y);
        auto Bs = B.slice(x, 0);
        auto t = op.template get_destination_cooperative_tensor<decltype(As), decltype(Bs), float>();
        for (uint16_t i = 0; i < t.get_capacity(); ++i) { if (t.is_valid_element(i)) t[i] = 0; }
        op.run(As, Bs, t);
        t.store(C.slice(x, y));
        if (p.rounds) {
            tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> R((device bfloat *)r + ulong(p.firstRow) * p.outWidth + ulong(head) * dim,
                                                                       dextents<int32_t, 2>(dim, rows), array<int32_t, 2>({1, int(p.outWidth)}));
            auto u = op.template get_destination_cooperative_tensor<decltype(As), decltype(Bs), bfloat>();
            for (uint16_t i = 0; i < t.get_capacity(); ++i) { if (t.is_valid_element(i)) u[i] = as_type<bfloat>(ushort(rounded(t[i]))); }
            u.store(R.slice(x, y));
        }
    }
    kernel void tiles_product_f32(device float *pr [[buffer(0)]], device float *v [[buffer(1)]], device float *o [[buffer(2)]],
                                  constant Params &p [[buffer(3)]], device ushort *r [[buffer(4)]],
                                  uint3 g [[threadgroup_position_in_grid]]) { product_body<float>(pr, v, o, r, p, g); }
    kernel void tiles_product_f16(device half *pr [[buffer(0)]], device half *v [[buffer(1)]], device float *o [[buffer(2)]],
                                  constant Params &p [[buffer(3)]], device ushort *r [[buffer(4)]],
                                  uint3 g [[threadgroup_position_in_grid]]) { product_body<half>(pr, v, o, r, p, g); }
    kernel void tiles_product_bf16(device bfloat *pr [[buffer(0)]], device bfloat *v [[buffer(1)]], device float *o [[buffer(2)]],
                                   constant Params &p [[buffer(3)]], device ushort *r [[buffer(4)]],
                                   uint3 g [[threadgroup_position_in_grid]]) { product_body<bfloat>(pr, v, o, r, p, g); }

    // The first `columns` of each row of a source with row stride `stride`, into a compact copy (x: column, y: row).
    kernel void tiles_to_half(device const float *x [[buffer(0)]], device half *y [[buffer(1)]], constant uint3 &shape [[buffer(2)]],
                              uint2 i [[thread_position_in_grid]]) {
        y[ulong(i.y) * shape.z + i.x] = half(x[ulong(i.y) * shape.y + i.x]);
    }
    kernel void tiles_to_bf16(device const float *x [[buffer(0)]], device bfloat *y [[buffer(1)]], constant uint3 &shape [[buffer(2)]],
                              uint2 i [[thread_position_in_grid]]) {
        y[ulong(i.y) * shape.z + i.x] = bfloat(x[ulong(i.y) * shape.y + i.x]);
    }
    """

    // MARK: The fused form

    private static let fusedLibraries = Mutex<[String: MTLLibrary]>([:])
    /// The fused kernel for an operand type, a tile and a head dimension (all compile-time: the cooperative tensors' extents), once per
    /// device and combination.
    static func fusedLibrary(_ device: MTLDevice, precision: Precision, tile: AttentionTile, headDim: Int) throws -> MTLLibrary {
        let key = "\(device.registryID) \(precision.metal) \(tile) \(headDim)"
        if let known = fusedLibraries.withLock({ $0[key] }) { return known }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        options.mathMode = .safe
        let built: MTLLibrary
        do { built = try device.makeLibrary(source: fusedSource(type: precision.metal, tile: tile, headDim: headDim), options: options) } catch {
            throw Failure.unsupported("the compiler refused the fused kernel at \(tile) — \(error.localizedDescription)")
        }
        fusedLibraries.withLock { $0[key] = built }
        return built
    }

    /// The fused kernel's source. One simdgroup a block of `BR` query rows of one head (the grid's z: the computed heads; y: the blocks
    /// of `simdgroups` simdgroups); the softmax is the online one, every figure of it fp32.
    static func fusedSource(type: String, tile: AttentionTile, headDim: Int) -> String {
        """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;
        using namespace mpp::tensor_ops;
        typedef \(type) T;
        constant constexpr int BR = \(tile.rows), BC = \(tile.keys), D = \(headDim);
        struct FusedParams { uint sequence, keys, qWidth, kvWidth, outWidth, group; float scale; uint rounds; };

        \(BFloatKernels.roundingSource)

        kernel void fused(device T *q [[buffer(0)]], device T *k [[buffer(1)]], device T *v [[buffer(2)]], device float *o [[buffer(3)]],
                          constant FusedParams &p [[buffer(4)]], device ushort *r [[buffer(5)]],
                          threadgroup float *shared [[threadgroup(0)]],
                          uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint groups [[simdgroups_per_threadgroup]]) {
            const int head = int(tg.z), kvHead = head / int(p.group);
            const int row0 = int(tg.y * groups + sg) * BR;
            if (row0 >= int(p.sequence)) return;
            const int keys = int(p.keys);
            threadgroup float *factor = shared + sg * BR;

            tensor<device T, dextents<int32_t, 2>, tensor_inline> Q(q + head * D, dextents<int32_t, 2>(D, int(p.sequence)), array<int32_t, 2>({1, int(p.qWidth)}));
            tensor<device T, dextents<int32_t, 2>, tensor_inline> K(k + kvHead * D, dextents<int32_t, 2>(D, keys), array<int32_t, 2>({1, int(p.kvWidth)}));
            tensor<device T, dextents<int32_t, 2>, tensor_inline> V(v + kvHead * D, dextents<int32_t, 2>(D, keys), array<int32_t, 2>({1, int(p.kvWidth)}));
            tensor<device float, dextents<int32_t, 2>, tensor_inline> O(o + head * D, dextents<int32_t, 2>(D, int(p.sequence)), array<int32_t, 2>({1, int(p.outWidth)}));

            constexpr auto ds = matmul2d_descriptor(BR, BC, D, false, true, false);
            matmul2d<ds, execution_simdgroup> scores;
            constexpr auto dv = matmul2d_descriptor(BR, D, BC, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
            matmul2d<dv, execution_simdgroup> product;

            auto Qb = Q.slice(0, row0);
            auto K0 = K.slice(0, 0);
            auto V0 = V.slice(0, 0);
            using QT = decltype(Qb); using KT = decltype(K0); using VT = decltype(V0);
            using PT = matmul2d<dv, execution_simdgroup>::cooperative_tensor_left_input_t<T, T, float>;
            auto Oc = product.get_destination_cooperative_tensor<PT, VT, float>();
            #pragma unroll
            for (uint16_t i = 0; i < Oc.get_capacity(); ++i) { if (Oc.is_valid_element(i)) Oc[i] = 0; }
            auto m = scores.get_row_reduction_destination_cooperative_tensor<QT, KT, float>();
            auto l = scores.get_row_reduction_destination_cooperative_tensor<QT, KT, float>();
            #pragma unroll
            for (uint16_t i = 0; i < m.get_capacity(); ++i) { if (m.is_valid_element(i)) { m[i] = -INFINITY; l[i] = 0; } }

            for (int j0 = 0; j0 < keys; j0 += BC) {
                auto Kc = K.slice(0, j0);
                auto Sc = scores.get_destination_cooperative_tensor<QT, KT, float>();
                #pragma unroll
            for (uint16_t i = 0; i < Sc.get_capacity(); ++i) { if (Sc.is_valid_element(i)) Sc[i] = 0; }
                scores.run(Qb, Kc, Sc);
                // The keys past the length (the last chunk's) leave the softmax: −∞, hence P = 0 (V's rows there read as zeros).
                // Only the last chunk can hold such keys: the others scale without asking for each score's index.
                const int valid = keys - j0;
                if (valid < BC) {
                    #pragma unroll
                    for (uint16_t i = 0; i < Sc.get_capacity(); ++i) {
                        if (Sc.is_valid_element(i)) { auto at = Sc.get_multidimensional_index(i); Sc[i] = at[0] < valid ? Sc[i] * p.scale : -INFINITY; }
                    }
                } else {
                    #pragma unroll
                    for (uint16_t i = 0; i < Sc.get_capacity(); ++i) { if (Sc.is_valid_element(i)) Sc[i] *= p.scale; }
                }
                auto mx = scores.get_row_reduction_destination_cooperative_tensor<QT, KT, float>();
                reduce_rows(Sc, mx, reduction_operation::max, -INFINITY);
                bool unchanged = true;
                #pragma unroll
            for (uint16_t i = 0; i < mx.get_capacity(); ++i) {
                    if (mx.is_valid_element(i)) {
                        const float mn = max(m[i], mx[i]);
                        const float alpha = precise::exp(m[i] - mn);
                        m[i] = mn; mx[i] = alpha;
                        unchanged = unchanged && alpha == 1.0f;
                        factor[mx.get_multidimensional_index(i)[0]] = alpha;
                    }
                }
                auto P = scores.get_destination_cooperative_tensor<QT, KT, T>();
                #pragma unroll
            for (uint16_t i = 0; i < Sc.get_capacity(); ++i) {
                    if (Sc.is_valid_element(i)) {
                        const float e = precise::exp(Sc[i] - *m.map_iterator(Sc.get_iterator(i)));
                        Sc[i] = e;
                        P[i] = T(e);
                    }
                }
                auto sum = scores.get_row_reduction_destination_cooperative_tensor<QT, KT, float>();
                reduce_rows(Sc, sum, reduction_operation::sum, 0.0f);
                #pragma unroll
            for (uint16_t i = 0; i < l.get_capacity(); ++i) { if (l.is_valid_element(i)) l[i] = l[i] * mx[i] + sum[i]; }
                // No row's maximum moved (every factor exactly 1, uniform over the simdgroup): rescaling O would change no bit — skipped.
                if (!simd_all(unchanged)) {
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    #pragma unroll
                    for (uint16_t i = 0; i < Oc.get_capacity(); ++i) {
                        if (Oc.is_valid_element(i)) { auto at = Oc.get_multidimensional_index(i); Oc[i] *= factor[at[1]]; }
                    }
                }
                auto Pin = product.get_left_input_cooperative_tensor<T, T, float>(P);
                auto Vc = V.slice(0, j0);
                product.run(Pin, Vc, Oc);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            #pragma unroll
            for (uint16_t i = 0; i < l.get_capacity(); ++i) { if (l.is_valid_element(i)) factor[l.get_multidimensional_index(i)[0]] = 1.0f / l[i]; }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint16_t i = 0; i < Oc.get_capacity(); ++i) {
                if (Oc.is_valid_element(i)) { auto at = Oc.get_multidimensional_index(i); Oc[i] *= factor[at[1]]; }
            }
            auto Ob = O.slice(0, row0);
            Oc.store(Ob);
            // The output rounded to bf16 too, where the next product reads its operand so: from the values just stored, into a
            // cooperative tensor of O's layout, stored as O is (element by element, the writes scatter and cost more than the pass).
            if (p.rounds) {
                tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> R((device bfloat *)r + head * D, dextents<int32_t, 2>(D, int(p.sequence)),
                                                                           array<int32_t, 2>({1, int(p.outWidth)}));
                auto Rc = product.get_destination_cooperative_tensor<PT, VT, bfloat>();
                #pragma unroll
                for (uint16_t i = 0; i < Oc.get_capacity(); ++i) { if (Oc.is_valid_element(i)) Rc[i] = as_type<bfloat>(ushort(rounded(Oc[i]))); }
                Rc.store(R.slice(0, row0));
            }
        }
        """
    }
}

/// **The fused attention's tile** (`TiledAttention`'s fused form): `rows` query rows a simdgroup, `keys` keys a chunk, `simdgroups` a
/// threadgroup — written `16x32/4`. Which one a Mac runs, if any, is its map's (`CapacityModel.attentionTiles`, per half-precision
/// kernel): the same candidates on every Mac (`candidates`), the three stages (`nil`) first, the choice by `MatmulTile.choose`'s rule.
package struct AttentionTile: Hashable, Sendable, Codable, CustomStringConvertible {
    package let rows: Int, keys: Int, simdgroups: Int
    package init(_ rows: Int, _ keys: Int, simdgroups: Int) { self.rows = rows; self.keys = keys; self.simdgroups = simdgroups }
    package init?(_ text: String) {
        let parts = text.split(separator: "/")
        guard parts.count == 2, let g = Int(parts[1]), g > 0 else { return nil }
        let dims = parts[0].split(separator: "x").compactMap { Int($0) }
        guard dims.count == 2, dims[0] > 0, dims[0] % 8 == 0, dims[1] > 0, dims[1] % 16 == 0 else { return nil }
        self.init(dims[0], dims[1], simdgroups: g)
    }
    package var description: String { "\(rows)x\(keys)/\(simdgroups)" }
    package init(from decoder: Decoder) throws {
        let text = try decoder.singleValueContainer().decode(String.self)
        guard let t = AttentionTile(text) else { throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: text)) }
        self = t
    }
    package func encode(to encoder: Encoder) throws { var c = encoder.singleValueContainer(); try c.encode(description) }

    /// The kernels the map measures a form for: the two whose operands are half-precision (the fused form has no fp32 variant: on an
    /// fp32 operand `matmul2d` stays on the shader cores, and there the graph is faster).
    package static let kernels = [CapacityModel.Kernel(.gpu, .tiles16), CapacityModel.Kernel(.gpu, .tilesbf16)]
    /// The fused tiles every Mac's map times beside the three stages. An accumulator of `rows × headDim` fp32 has to fit a simdgroup's
    /// registers, hence few rows; the chunk of keys trades the number of passes against the scores' tile.
    package static let candidates = [AttentionTile(16, 32, simdgroups: 4), AttentionTile(16, 64, simdgroups: 4), AttentionTile(32, 32, simdgroups: 4)]
    /// Alternated rounds of the choice, one run of the tier's longest attention each (11 to 60 ms on an M5 Pro): the least of three.
    /// Not the shortest sequence, though 16 times cheaper: its millisecond runs are bimodal — on an M5 Pro one bf16 tile ran 1.11 ms in
    /// one map and 1.60 in the next, and the choice flipped to a tile 30 % slower at 4 096.
    package static let rounds = 3
}
