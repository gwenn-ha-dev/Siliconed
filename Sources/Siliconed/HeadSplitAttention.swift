import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph
import Synchronization

/// **The SDPA cut by heads between the GPU and the AMX** — Z-Image's only SDPA path at
/// S ≥ 1024 (judged at threshold on its goldens), and
/// Qwen-Image-2.1's for its target's queries on all the keys (judged at the fp64 witness).
///
///     q [S, H·Dh], k, v [K, H·Dh] fp32, as the GEMMs wrote them (unified memory, no copy)
///        │
///        ├─ heads [0, g)  GPU   MPSGraph: slice heads → [g, S, Dh] → SDPA → [S, g, Dh] → pad `a` heads of zeros ─► out [S, H·Dh]
///        │
///        └─ heads [g, H)  CPU   per block of `rows` queries of one head, Q_b, K_h, V_h packed:    ─► spill [S, a·Dh]
///                                 scores = Q_b · K_hᵀ                  BNNSMatMul, 1 thread (AMX)
///                                 p = exp(scale·s − scale·max s), Σp    vDSP_maxv, then `exponentiate` (one pass)
///                                 O_b = (P · V) / Σp                   BNNSMatMul, then vDSP_vsdiv on rows × Dh only
///        then, both engines done: spill → the CPU heads' columns of out
///
/// During the SDPA the AMX and the P cores are idle, and `MPSGraph`'s SDPA is bound by issue, not by
/// DRAM. Measured on 30 heads × 128 at S = 4128: GPU alone 152–174 ms, cut `a = 13` on
/// six threads **87.7 ms**, the GPU heads bit-identical to the GPU alone, the CPU heads at
/// 5.4·10⁻⁶ of an fp64 attention (the GPU's own: 5.4·10⁻⁶).
///
/// What a port by analogy with `Conductor` would get wrong:
///
/// - **The cut is by head, never by query row.** A row split would also be exact, but every
///   engine would then read all of `K`/`V` for all heads; by head, each engine reads only its own.
///   Each head is computed entirely by one engine, so no sum is re-associated: the fraction can be
///   frozen and the render stays reproducible bit for bit (as for the frozen GPU/AMX cut).
/// - **The CPU's block boundaries are fixed by position** (`rows`-multiples), not by thread, and each
///   thread takes a contiguous run of blocks (bounds by endpoints): the bits do not depend on which
///   thread computed a block, and a thread packs each head's `K_h`/`V_h` once for its run.
/// - **The operands are packed, never read in place.** In place, a row of one head is `Dh` floats every
///   `H·Dh`; at Qwen's 32 × 128 that stride is 16 KB, a power of two, and every row lands in the same
///   cache sets: a CPU head cost 15 ms at 32 heads against 7.9 at 30 (same S and K) — the "~14 ms,
///   cause unknown" that once kept Qwen off this path. Packed: 8.5 ms at both, and the copy is
///   `rows·Dh` floats for `Q_b`, `keys·Dh` for `K_h` and `V_h` once per head of a thread's run.
/// - **`MPSGraph` cannot write a strided result**, so the GPU writes the whole `[S, H·Dh]` (its heads,
///   then zeros) and the CPU heads wait in a spill until the GPU is done. The copy is `a/H` of the
///   output, ~27 MB at 1024² — a few ms against the ~60 ms saved.
/// - **The softmax is not free on the CPU.** It is 17 M `exp` per head at S = 4128 — half of a head's
///   time on one thread (GEMMs 9.9 ms, softmax 10.5 ms); it is why the CPU part wants all the
///   P cores and not only one thread per AMX block, unlike the GEMM cut.
/// - **GQA cuts by group, never inside one** (Krea 2, 48 `q` heads over 12 `k`/`v`): the `g` `q` heads that read one
///   `k`/`v` head stay on one engine, so the GPU's graph slices whole groups and broadcasts them as `Attention`
///   does, and a CPU thread packs a group's `K`/`V` once for its `g` heads. `kvHeads == heads` is the old path,
///   graph and bits.
/// - **BNNS on one thread per call.** Several threads share one AMX block per cluster; the work
///   is parallelized here, by blocks, and `n_threads = 1` keeps BNNS from spawning its own.
///   Not `cblas_sgemm`: its threading switch (`BLASSetThreading`) is process-wide, and the engine
///   has other `cblas` users.
package final class HeadSplitAttention {
    package let heads: Int
    package let sequence: Int
    /// The rows of `k` and `v`: `sequence` for a self-attention (Z-Image), more for a query block
    /// that also reads a prefix (Qwen-Image-2.1's target on the conditions' keys).
    package let keys: Int
    package let headDim: Int
    /// The `k`/`v` heads (GQA): `heads` without grouping; `q` head `h` reads `k`/`v` head `h / (heads / kvHeads)`.
    package let kvHeads: Int
    private var group: Int { heads / kvHeads }
    /// Heads `[0, gpuHeads)` go to the GPU, `[gpuHeads, heads)` to the CPU.
    package let gpuHeads: Int
    package var cpuHeads: Int { heads - gpuHeads }

    /// The query rows of one CPU work item: the `rows × S` scores of one head must stay in the
    /// cluster's L2 (12 MB on the M1 Pro, shared by three threads). A sweep of 64 to 512: flat
    /// within 5 % on one thread, 128 and 256 ahead.
    package static let blockRows = 128

    /// Below this, the cut is not taken: the text blocks run at 32 tokens, where a CPU head is
    /// microseconds and the rendezvous would dominate.
    package static let minimumSequence = 1024

    /// **Whether an attention of `sequence` query rows (of `heads` of `headDim`, when the caller knows) is cut by
    /// heads** — the one condition the blocks, Qwen-Image-2.1's target and the memory formulas share: the route
    /// table's (`RouteTable.Attention`; today's `head_split` setting and `minimumSequence` as its rules). Off,
    /// the attention runs in one MPSGraph SDPA.
    package static func applies(sequence: Int, heads: Int? = nil, headDim: Int? = nil) -> Bool {
        EngineSettings.effective.routes.attention.route(sequence: sequence, heads: heads, headDim: headDim).splits
    }

    /// **Whether the flash kernel is allowed for an attention of `sequence` rows** (the route table's; a model that
    /// implements it takes it before the split).
    package static func allowsFlash(sequence: Int, heads: Int? = nil, headDim: Int? = nil) -> Bool {
        EngineSettings.effective.routes.attention.route(sequence: sequence, heads: heads, headDim: headDim).flash
    }

    /// **Which heads go where, defined once.** `fraction` is the share of heads handed to the CPU,
    /// rounded to the nearest head — to the nearest `group` of heads under GQA; the GPU keeps at least one
    /// group, the CPU may get none.
    package static func partition(heads: Int, fraction: Double, group: Int = 1) -> (gpu: Range<Int>, cpu: Range<Int>) {
        let groups = heads / group
        let cpu = group * max(0, min(groups - 1, Int((Double(groups) * fraction).rounded())))
        return (0..<(heads - cpu), (heads - cpu)..<heads)
    }

    /// The CPU work items: `(head, first row, rows)`, every row of every CPU head exactly once, in a
    /// fixed order. The bits of a block depend on its position, never on the thread that computes it.
    package static func items(cpu: Range<Int>, sequence: Int, rows: Int) -> [(head: Int, row: Int, rows: Int)] {
        var result: [(head: Int, row: Int, rows: Int)] = []
        for head in cpu {
            var row = 0
            while row < sequence { result.append((head, row, min(rows, sequence - row))); row += rows }
        }
        return result
    }

    private let queue: MTLCommandQueue
    private let graph: MPSGraph
    private let q, k, v, out: MPSGraphTensor
    private let threads: Int
    private let work: [(head: Int, row: Int, rows: Int)]
    /// The CPU heads' output, `[S, a·Dh]`, copied into `out` once the GPU is done.
    private let spill: UnsafeMutablePointer<Float>
    private let scratch: [Scratch]
    /// **Every host byte of the instance, in one arena of `hostSlices`**. `malloc`ed, the spill
    /// and the scratch outlived the instance in `phys_footprint` on macOS 27 (with the rest of the DiT's
    /// `malloc`, 384 MB between Z-Image's DiT and its decoder at 1024²); unmapped, they leave with it.
    private let host: Arena

    /// One thread's scores block, BNNS workspace, and its **packed** operands: the queries of its block
    /// and the keys and values of the head it is on, each contiguous at `Dh` floats a row. Slices of
    /// the instance's arena.
    private final class Scratch {
        let scores: UnsafeMutablePointer<Float>
        let sums: UnsafeMutablePointer<Float>
        let packedQ, packedK, packedV: UnsafeMutablePointer<Float>
        /// Sized for the largest of the instance's products (`workspaceBytes`), `nil` when BNNS asks for none.
        let workspace: UnsafeMutableRawPointer?
        let workspaceBytes: Int
        /// The `k`/`v` head whose keys and values `packedK`/`packedV` hold, `-1` for none (set per `run`).
        var packedHead = -1
        var failures = 0
        init(_ arena: Arena, thread t: Int, rows: Int, keys: Int, headDim: Int, workspace: Int) throws {
            func floats(_ name: String, _ count: Int) throws -> UnsafeMutablePointer<Float> {
                try arena.reserve("\(name).\(t)", bytes: count * 4).assumingMemoryBound(to: Float.self)
            }
            scores = try floats("scores", rows * keys)
            sums = try floats("sums", rows)
            packedQ = try floats("q", rows * headDim)
            packedK = try floats("k", keys * headDim)
            packedV = try floats("v", keys * headDim)
            self.workspace = workspace > 0 ? try arena.reserve("workspace.\(t)", bytes: workspace) : nil
            workspaceBytes = workspace
        }
    }

    /// **The share of heads handed to the CPU: 0.4, the M1 Pro's, unless the capacity map found another** (the
    /// `head_split_fraction` setting, `balance`). Frozen, not servoed: a share decides which engine computes a
    /// head, hence the bits. It was swept in the engine at S = 4128 — 12 of Z-Image's 30 heads, 18 GPU ahead
    /// (155.7 → 95.9 ms) — and judged at threshold on every Z-Image golden, then adopted. Thread count, by
    /// contrast, changes the speed and never the bits.
    ///
    /// **It does not depend on the sequence.** Swept again at 2,048 to 24,576 tokens (Anima's 16 heads, Z-Image's
    /// 30): both sides cost `S²` per head — the GPU's `4·S²·Dh` FLOP at ~1.6 TFLOP/s, the CPU's the same through the
    /// AMX plus `S²` exponentials —, so the ratio of the two per-head times, which sets the share, stays within
    /// 1.7–2.4 and the best count of heads sits at 6 of 16 (5 at 12,288, ahead by 5 %) and 12 of 30 at every
    /// length, the call time flat within a head either way. What moves with the length is the GPU's penalty
    /// for the CPU running beside it (+5 % at 8,192 tokens, +15 % at 4,096, taken into `balance`).
    package static var cpuFraction: Double { EngineSettings.effective.routes.attention.defaultSplitFraction }

    /// The share of heads the CPU takes for an attention of this class: the route table's rule, the default
    /// share where none splits.
    package static func cpuFraction(sequence: Int, heads: Int? = nil, headDim: Int? = nil) -> Double {
        EngineSettings.effective.routes.attention.route(sequence: sequence, heads: heads, headDim: headDim).splitFraction
            ?? cpuFraction
    }

    /// The share the capacity map measures the cut at, and the default of `head_split_fraction`.
    package static let referenceFraction = 0.4

    /// **One attention's cut call, as the map measured it** (`Diagnostic.AttentionCapacity`).
    package struct Balance: Sendable {
        package var heads, group: Int
        /// The whole call on the GPU alone, seconds.
        package var graphSeconds: Double
        /// The cut call's CPU heads and the two parts' wall times, run at the same time.
        package var cpuHeads: Int
        package var gpuSeconds, cpuSeconds: Double

        package init(heads: Int, group: Int = 1, graphSeconds: Double, cpuHeads: Int, gpuSeconds: Double, cpuSeconds: Double) {
            self.heads = heads; self.group = group; self.graphSeconds = graphSeconds
            self.cpuHeads = cpuHeads; self.gpuSeconds = gpuSeconds; self.cpuSeconds = cpuSeconds
        }

        /// **The call's time at a share**, from the three figures. A GPU head costs `b₀ = graph / H` alone and
        /// `b₀·(1 + κφ)` while the CPU takes a share φ (the contention for memory and power, fitted on the
        /// measured point: κ = (gpu / (H − c) / b₀ − 1) / φ₀); a CPU head costs `τ = cpu / c`. The call ends
        /// when both do: `max((H − c)·b₀·(1 + κφ), c·τ)` at `c` heads, the count `partition` gives.
        /// `nil` if the figures are unusable.
        package func seconds(fraction: Double) -> Double? {
            let gpuHeads = heads - cpuHeads
            guard heads > 0, cpuHeads > 0, gpuHeads > 0, graphSeconds > 0, gpuSeconds > 0, cpuSeconds > 0 else { return nil }
            let b0 = graphSeconds / Double(heads), tau = cpuSeconds / Double(cpuHeads)
            let share0 = Double(cpuHeads) / Double(heads)
            let kappa = max(0, (gpuSeconds / Double(gpuHeads) / b0 - 1) / share0)
            let c = Double(HeadSplitAttention.partition(heads: heads, fraction: fraction, group: group).cpu.count)
            let phi = c / Double(heads)
            return max((Double(heads) - c) * b0 * (1 + kappa * phi), c * tau)
        }
    }

    /// **The share of heads the CPU should take on this machine**, from the map's attentions: the one among
    /// `candidates` that minimizes the summed predicted time, if it beats the reference share's by more than
    /// `margin` — otherwise `nil`, the reference stays (a gain inside the noise is not worth moving every
    /// bit). Only the attention's cost shapes enter: the optimum does not depend on the sequence.
    package static func balance(_ items: [Balance], margin: Double,
                                candidates: [Double] = stride(from: 0.1, through: 0.7, by: 0.05).map { $0 }) -> Double? {
        guard !items.isEmpty else { return nil }
        func total(_ fraction: Double) -> Double? {
            var sum = 0.0
            for item in items { guard let t = item.seconds(fraction: fraction) else { return nil }; sum += t }
            return sum
        }
        guard let reference = total(referenceFraction) else { return nil }
        var best: (fraction: Double, seconds: Double)?
        for fraction in candidates {
            if let t = total(fraction), t < (best?.seconds ?? .infinity) { best = (fraction, t) }
        }
        guard let best, abs(best.fraction - referenceFraction) > 1e-9, reference / best.seconds > 1 + margin else { return nil }
        return best.fraction
    }

    /// - Parameter cpuFraction: the share of heads handed to the CPU; the product's is `cpuFraction`,
    ///   other values serve the `sdpa` check's sweep.
    /// - Parameter keys: the rows of `k`/`v`, `nil` for `sequence`.
    /// - Parameter kvHeads: the `k`/`v` heads under GQA, `nil` for `heads`.
    /// - Parameter matrixLimit: the bound the GPU heads' query slices keep under, `Attention.maxMatrix` —
    ///   lower only in the `sdpa` check, which slices a graph that fits whole to compare the two.
    package init(device: MTLDevice, queue: MTLCommandQueue, heads: Int, sequence: Int, headDim: Int,
                 keys: Int? = nil, kvHeads: Int? = nil, cpuFraction: Double? = nil,
                 threads: Int = EngineSettings.effective.machine.performanceCores,
                 matrixLimit: Int = Attention.maxMatrix) throws {
        let cpuFraction = cpuFraction ?? HeadSplitAttention.cpuFraction(sequence: sequence, heads: heads, headDim: headDim)
        let kvHeads = kvHeads ?? heads
        precondition(heads % kvHeads == 0, "\(heads) heads over \(kvHeads) k/v heads")
        let group = heads / kvHeads
        let (gpu, cpu) = HeadSplitAttention.partition(heads: heads, fraction: cpuFraction, group: group)
        self.queue = queue
        let keys = keys ?? sequence
        self.heads = heads; self.sequence = sequence; self.keys = keys; self.headDim = headDim; self.kvHeads = kvHeads
        self.gpuHeads = gpu.count
        self.threads = max(1, threads)
        work = HeadSplitAttention.items(cpu: cpu, sequence: sequence, rows: HeadSplitAttention.blockRows)
        let host = try Arena(capacity: HeadSplitAttention.hostBytes(heads: heads, sequence: sequence, keys: keys,
                                                                    headDim: headDim, group: group,
                                                                    cpuFraction: cpuFraction, threads: threads))
        self.host = host
        spill = try host.reserve("spill", bytes: 4 * HeadSplitAttention.spillFloats(sequence: sequence, cpuHeads: cpu.count,
                                                                                    headDim: headDim))
            .assumingMemoryBound(to: Float.self)
        let workspace = HeadSplitAttention.workspaceBytes(sequence: sequence, keys: keys, headDim: headDim, cpuHeads: cpu.count)
        scratch = try (0..<max(1, threads)).map {
            try Scratch(host, thread: $0, rows: HeadSplitAttention.blockRows, keys: keys, headDim: headDim, workspace: workspace)
        }

        let graph = MPSGraph()
        self.graph = graph
        let n = { (x: Int) in NSNumber(value: x) }
        let shape: [NSNumber] = [1, n(sequence), n(heads), n(headDim)]
        let kvShape: [NSNumber] = [1, n(keys), n(kvHeads), n(headDim)]
        let q = graph.placeholder(shape: shape, dataType: .float32, name: "q")
        let k = graph.placeholder(shape: kvShape, dataType: .float32, name: "k")
        let v = graph.placeholder(shape: kvShape, dataType: .float32, name: "v")
        self.q = q; self.k = k; self.v = v
        func gpuPart(_ t: MPSGraphTensor) -> MPSGraphTensor {
            let sliced = cpu.isEmpty ? t : graph.sliceTensor(t, dimension: 2, start: 0, length: gpu.count, name: nil)
            return graph.transposeTensor(sliced, dimension: 1, withDimension: 2, name: nil)
        }
        let back: MPSGraphTensor
        if group == 1 {
            // **Past `Attention.maxMatrix` the queries go in slices**, as `Attention`'s own graphs do: the
            // GPU's heads over a long sequence would otherwise be one SDPA whose matrix MPSGraph splits by
            // itself (Z-Image from about 1024×1660: 18 heads × 6,800²; at 2048²: 19.6 GB). Each output row
            // reads its own query row and every key, so the slices reassociate nothing; at or below the
            // bound there is one slice, the graph of a single SDPA, bits included.
            let slices = Attention.groupedSlices(heads: gpu.count, rows: sequence, keys: keys, limit: matrixLimit)
            let (qh, kh, vh) = (gpuPart(q), gpuPart(k), gpuPart(v))
            let scale = 1 / Float(headDim).squareRoot()
            let attended: MPSGraphTensor
            if slices == 1 {
                attended = graph.scaledDotProductAttention(query: qh, key: kh, value: vh, mask: nil, scale: scale, name: "sdpa")
            } else {
                let length = (sequence + slices - 1) / slices
                var pieces: [MPSGraphTensor] = []
                var begin = 0
                while begin < sequence {
                    let l = min(length, sequence - begin)
                    pieces.append(graph.scaledDotProductAttention(
                        query: graph.sliceTensor(qh, dimension: 2, start: begin, length: l, name: nil),
                        key: kh, value: vh, mask: nil, scale: scale, name: nil))
                    begin += l
                }
                attended = graph.concatTensors(pieces, dimension: 2, name: nil)
            }
            back = graph.transposeTensor(attended, dimension: 1, withDimension: 2, name: nil)
        } else {
            // As `Attention`'s GQA graph, on the GPU's groups: `q` [1, Gk, g, S, Dh], `k`/`v` [1, Gk, 1, K, Dh]
            // broadcast over `g`, in query slices that keep each matrix under `Attention.maxMatrix`.
            let kvGPU = gpu.count / group
            func kvPart(_ t: MPSGraphTensor) -> MPSGraphTensor {
                let sliced = cpu.isEmpty ? t : graph.sliceTensor(t, dimension: 2, start: 0, length: kvGPU, name: nil)
                return graph.reshape(graph.transposeTensor(sliced, dimension: 1, withDimension: 2, name: nil),
                                     shape: [1, n(kvGPU), 1, n(keys), n(headDim)], name: nil)
            }
            let qg = graph.reshape(gpuPart(q), shape: [1, n(kvGPU), n(group), n(sequence), n(headDim)], name: nil)
            let (kh, vh) = (kvPart(k), kvPart(v))
            let slices = Attention.groupedSlices(heads: gpu.count, rows: sequence, keys: keys, limit: matrixLimit)
            let length = (sequence + slices - 1) / max(1, slices)
            var pieces: [MPSGraphTensor] = []
            var begin = 0
            while begin < sequence {
                let l = min(length, sequence - begin)
                let slice = slices > 1 ? graph.sliceTensor(qg, dimension: 3, start: begin, length: l, name: nil) : qg
                pieces.append(graph.scaledDotProductAttention(query: slice, key: kh, value: vh, mask: nil,
                                                              scale: 1 / Float(headDim).squareRoot(), name: nil))
                begin += l
            }
            let grouped = pieces.count > 1 ? graph.concatTensors(pieces, dimension: 3, name: nil) : pieces[0]
            let merged = graph.reshape(grouped, shape: [1, n(gpu.count), n(sequence), n(headDim)], name: nil)
            back = graph.transposeTensor(merged, dimension: 1, withDimension: 2, name: nil)
        }
        out = cpu.isEmpty ? back : graph.padTensor(back, with: .constant, leftPadding: [0, 0, 0, 0],
                                                   rightPadding: [0, 0, n(cpu.count), 0], constantValue: 0, name: nil)
    }

    /// The CPU heads' spill, `[S, a·Dh]` floats (one at least).
    package static func spillFloats(sequence: Int, cpuHeads: Int, headDim: Int) -> Int {
        max(1, sequence * cpuHeads * headDim)
    }

    /// **The slices of the instance's arena, in the order `init` reserves them**, bytes — the spill,
    /// then per thread a `Scratch`: the scores block `[rows, K]`, its sums, the packed `Q_b`, `K_h`,
    /// `V_h`, and the BNNS workspace (`workspaceBytes`; no slice when BNNS asks for none).
    package static func hostSlices(heads: Int, sequence: Int, keys: Int? = nil, headDim: Int, group: Int = 1,
                                   cpuFraction: Double? = nil,
                                   threads: Int = EngineSettings.effective.machine.performanceCores) -> [Int] {
        let cpuFraction = cpuFraction ?? HeadSplitAttention.cpuFraction(sequence: sequence, heads: heads, headDim: headDim)
        let cpu = partition(heads: heads, fraction: cpuFraction, group: group).cpu.count, k = keys ?? sequence, rows = blockRows
        let workspace = workspaceBytes(sequence: sequence, keys: k, headDim: headDim, cpuHeads: cpu)
        let perThread = [rows * k, rows, rows * headDim, k * headDim, k * headDim].map { $0 * 4 }
            + (workspace > 0 ? [workspace] : [])
        return [4 * spillFloats(sequence: sequence, cpuHeads: cpu, headDim: headDim)]
            + Array(repeating: perThread, count: max(1, threads)).flatMap { $0 }
    }

    /// **The host memory one instance holds**: its arena, `hostSlices` each rounded to the page — all
    /// of it, mapped by `init` and unmapped by `deinit`. The memory need charges it whole, as the
    /// head split's term beside the GPU heads' query copy (`OperatorFootprint.sdpaQueries`).
    package static func hostBytes(heads: Int, sequence: Int, keys: Int? = nil, headDim: Int, group: Int = 1,
                                  cpuFraction: Double? = nil,
                                  threads: Int = EngineSettings.effective.machine.performanceCores) -> Int {
        Arena.capacity(slices: hostSlices(heads: heads, sequence: sequence, keys: keys, headDim: headDim, group: group,
                                          cpuFraction: cpuFraction, threads: threads))
    }

    /// **One thread's BNNS workspace**: the largest `BNNSMatMul` asks for over the instance's products —
    /// `Q_b · K_hᵀ` and `P · V_h`, at `blockRows` rows and at the last block's — asked of BNNS with the
    /// descriptors `multiply` passes (`descriptors`). BNNS answers from the shapes alone, not the data.
    package static func workspaceBytes(sequence: Int, keys: Int, headDim: Int, cpuHeads: Int) -> Int {
        guard cpuHeads > 0 else { return 0 }
        let rowCounts = Set([min(blockRows, sequence), sequence % blockRows].filter { $0 > 0 })
        return rowCounts.map { rows in
            let scores = descriptors(a: nil, aRows: rows, aColumns: headDim, aStride: headDim,
                                     b: nil, bRows: keys, bColumns: headDim, bStride: headDim,
                                     c: nil, cColumns: keys, cStride: keys)
            let values = descriptors(a: nil, aRows: rows, aColumns: keys, aStride: keys,
                                     b: nil, bRows: keys, bColumns: headDim, bStride: headDim,
                                     c: nil, cColumns: headDim, cStride: cpuHeads * headDim)
            return max(workspace(scores, transposeB: true), workspace(values, transposeB: false), 0)
        }.max() ?? 0
    }

    private static func workspace(_ d: (a: BNNSNDArrayDescriptor, b: BNNSNDArrayDescriptor, c: BNNSNDArrayDescriptor),
                                  transposeB: Bool) -> Int {
        var (A, B, C) = d
        var params = BNNSFilterParameters()
        params.n_threads = 1
        return amx_matmul_workspace(false, transposeB, 1, &A, &B, &C, &params)
    }

    /// Inputs and output fp32 `[S, H·Dh]`: the buffers for the GPU, the same memory as pointers for
    /// the CPU. Returns the GPU time and the CPU part's wall time.
    package func run(q qBuffer: MTLBuffer, k kBuffer: MTLBuffer, v vBuffer: MTLBuffer, into outBuffer: MTLBuffer,
                     q qPointer: UnsafePointer<Float>, k kPointer: UnsafePointer<Float>, v vPointer: UnsafePointer<Float>,
                     out outPointer: UnsafeMutablePointer<Float>) -> (gpu: Double, cpu: Double) {
        let n = { (x: Int) in NSNumber(value: x) }
        let shape: [NSNumber] = [1, n(sequence), n(heads), n(headDim)]
        let kvShape: [NSNumber] = [1, n(keys), n(kvHeads), n(headDim)]
        func data(_ b: MTLBuffer, _ s: [NSNumber]) -> MPSGraphTensorData { MPSGraphTensorData(b, shape: s, dataType: .float32) }
        // **The graph may commit by itself** (`commitAndContinue`, as in `Attention.run`): at
        // 1024×1536 it does, and committing a wrapped `MTLCommandBuffer` afterwards was the fatal
        // "commit an already committed command buffer". The `MPSCommandBuffer` is committed and
        // waited on as such: it follows whichever buffer the graph left current. Same graph, same bits.
        // Our own buffer then carries no GPU time, so a failing buffer's "nothing ran" is read on the output:
        // a sentinel in the last row of head 0, which the graph overwrites whenever it runs.
        let probe = (sequence - 1) * heads * headDim, sentinel: UInt32 = 0x7FA5_A5A5
        outPointer[probe] = Float(bitPattern: sentinel)
        let commands = MPSCommandBuffer(from: queue)
        graph.encode(to: commands,
                     feeds: [q: data(qBuffer, shape), k: data(kBuffer, kvShape), v: data(vBuffer, kvShape)],
                     targetOperations: nil, resultsDictionary: [out: data(outBuffer, shape)], executionDescriptor: nil)
        // The GPU part's time is the wall time to the completion of the buffer the graph left current:
        // the root's own GPU time covers only its first segment when the graph committed by itself
        // (Krea 2 at 1024²: 0.96 s reported for ~4 s of work).
        let finished = Atomic<UInt64>(0), done = DispatchSemaphore(value: 0)
        commands.commandBuffer.addCompletedHandler { _ in
            finished.store(DispatchTime.now().uptimeNanoseconds, ordering: .releasing)
            done.signal()
        }
        let started = DispatchTime.now().uptimeNanoseconds
        commands.commit()

        if !work.isEmpty {
            // A thread takes a contiguous run of items (bounds by endpoints), hence one or two heads,
            // each packed once; a free-for-all would repack a head at nearly every item.
            HeadSplitAttention.perform(threads) { t in
                let s = scratch[t]
                s.packedHead = -1
                for i in (t * work.count / threads)..<((t + 1) * work.count / threads) {
                    block(work[i], s, q: qPointer, k: kPointer, v: vPointer)
                }
            }
        }
        let cpu = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9

        commands.waitUntilCompleted()
        let root = commands.rootCommandBuffer
        root.waitUntilCompleted()
        let what = "SDPA (head split) S=\(sequence) H=\(gpuHeads)/\(heads)"
        if root.error != nil || root.gpuEndTime > root.gpuStartTime {
            _ = Attention.verdict(root, what: what)
        } else if outPointer[probe].bitPattern == sentinel {
            // Committed by the graph itself (1024×1536): the work ran in buffers MPS made, whose
            // verdict we do not hold — the sentinel says whether it ran.
            Attention.report("\(what): the output was never written")
        }
        done.wait()
        let gpu = Double(finished.load(ordering: .acquiring) - started) * 1e-9
        if cpuHeads > 0 {
            let width = heads * headDim, spilled = cpuHeads * headDim, column = gpuHeads * headDim
            HeadSplitAttention.perform(threads) { t in
                for row in stride(from: t, to: sequence, by: threads) {
                    (outPointer + row * width + column).update(from: spill + row * spilled, count: spilled)
                }
            }
            let failures = scratch.reduce(0) { $0 + $1.failures }
            if failures > 0 {
                // Through `Attention.report`, like a GPU refusal: counted, so the render's summary says it.
                Attention.report("SDPA head split: BNNSMatMul refused \(failures) block(s); the CPU heads are wrong")
                scratch.forEach { $0.failures = 0 }
            }
        }
        return (gpu, cpu)
    }

    /// `body(0..<n)` on `n` threads, returning when all are done. Each thread owns its scratch and
    /// its work items, so nothing is written twice — the same pattern as `Parallel.rows`.
    private static func perform(_ n: Int, _ body: (Int) -> Void) {
        withoutActuallyEscaping(body) { body in
            nonisolated(unsafe) let body = body
            DispatchQueue.concurrentPerform(iterations: n) { body($0) }
        }
    }

    /// `rows` query rows of one CPU head, into the spill.
    private func block(_ item: (head: Int, row: Int, rows: Int), _ s: Scratch,
                       q: UnsafePointer<Float>, k: UnsafePointer<Float>, v: UnsafePointer<Float>) {
        let width = heads * headDim, column = item.head * headDim
        let kvHead = item.head / group, kvWidth = kvHeads * headDim, kvColumn = kvHead * headDim
        let spilledWidth = cpuHeads * headDim, spilledColumn = (item.head - gpuHeads) * headDim
        let o = spill + item.row * spilledWidth + spilledColumn
        // Packed, not read in place: at Qwen's 32 × 128 the row stride is 16 KB, and in place every row
        // of `K_h` falls in the same cache sets (see the header).
        func pack(_ source: UnsafePointer<Float>, rows: Int, into destination: UnsafeMutablePointer<Float>) {
            for r in 0..<rows { (destination + r * headDim).update(from: source + r * width, count: headDim) }
        }
        func packKV(_ source: UnsafePointer<Float>, into destination: UnsafeMutablePointer<Float>) {
            for r in 0..<keys { (destination + r * headDim).update(from: source + r * kvWidth, count: headDim) }
        }
        pack(q + item.row * width + column, rows: item.rows, into: s.packedQ)
        if s.packedHead != kvHead {
            packKV(k + kvColumn, into: s.packedK)
            packKV(v + kvColumn, into: s.packedV)
            s.packedHead = kvHead
        }
        // scores [rows, K] = Q_b [rows, Dh] · K_hᵀ
        multiply(s, a: s.packedQ, aRows: item.rows, aColumns: headDim, aStride: headDim,
                 b: s.packedK, bRows: keys, bColumns: headDim, bStride: headDim, transposeB: true,
                 c: s.scores, cColumns: keys, cStride: keys)
        let scale = 1 / Float(headDim).squareRoot()
        for r in 0..<item.rows {
            let row = s.scores + r * keys
            var maximum: Float = 0
            vDSP_maxv(row, 1, &maximum, vDSP_Length(keys))
            // One pass over the row for the scale, the exponential and the sum (see `exponentiate`).
            s.sums[r] = HeadSplitAttention.exponentiate(row, count: keys, scale: scale, maximum: maximum)
        }
        // O_b [rows, Dh] = P [rows, K] · V_h [K, Dh], then each row divided by its Σp
        multiply(s, a: s.scores, aRows: item.rows, aColumns: keys, aStride: keys,
                 b: s.packedV, bRows: keys, bColumns: headDim, bStride: headDim, transposeB: false,
                 c: o, cColumns: headDim, cStride: spilledWidth)
        for r in 0..<item.rows {
            vDSP_vsdiv(o + r * spilledWidth, 1, s.sums + r, o + r * spilledWidth, 1, vDSP_Length(headDim))
        }
    }

    /// **The softmax's numerator and denominator in one pass**: `row[j] = exp(scale · (row[j] − maximum))` in place, and
    /// the sum of the row returned (eight fp32 partial sums, added at the end). In C (`AMXLegacy`): the same code in Swift's SIMD types ran five times slower (the fused multiply-adds went through the scalar library). `maximum` is the row's, so every
    /// argument is ≤ 0 and `exp` is in (0, 1].
    ///
    /// Four passes over a row (`vDSP_vsmsa`, `vvexpf`, `vDSP_sve` after the maximum) cost as much as the
    /// products on the CPU — 10.5 ms of a head's 20 at 4,128 tokens —, and the softmax is the part of the
    /// head that the AMX does not do: the faster it is, the more heads the P cores can take from the GPU.
    /// Cephes' `expf` (`x = n·ln 2 + r`, |r| ≤ ln 2 / 2, a degree-7 polynomial on `r`, `2ⁿ` by the
    /// exponent field), fused multiply-adds throughout, 4 lanes × 2. Its relative error is under 4·10⁻⁷ with the argument's own fp32 rounding (a few
    /// ulp; `vvexpf`'s is one), which the sum's own fp32 rounding (~10⁻⁶ over 16,384 terms) and the GPU
    /// heads' (5·10⁻⁶ of fp64) bury. Below `−87.3365` the argument is held there: `exp` is 1.2·10⁻³⁸ against a
    /// sum of at least 1 (the maximum's own term), and no result is subnormal.
    package static func exponentiate(_ row: UnsafeMutablePointer<Float>, count: Int, scale: Float, maximum: Float) -> Float {
        amx_softmax_exponentiate(row, count, scale, -scale * maximum)
    }

    /// `C = A · op(B)` on one thread, row-major, with row strides. `B` is described as stored
    /// (`bRows × bColumns`); `transposeB` multiplies by its transpose.
    private func multiply(_ s: Scratch, a: UnsafeMutablePointer<Float>, aRows: Int, aColumns: Int, aStride: Int,
                          b: UnsafeMutablePointer<Float>, bRows: Int, bColumns: Int, bStride: Int, transposeB: Bool,
                          c: UnsafeMutablePointer<Float>, cColumns: Int, cStride: Int) {
        var (A, B, C) = HeadSplitAttention.descriptors(a: a, aRows: aRows, aColumns: aColumns, aStride: aStride,
                                                       b: b, bRows: bRows, bColumns: bColumns, bStride: bStride,
                                                       c: c, cColumns: cColumns, cStride: cStride)
        var params = BNNSFilterParameters()
        params.n_threads = 1
        let size = amx_matmul_workspace(false, transposeB, 1, &A, &B, &C, &params)
        // More than `workspaceBytes` foresaw would be a write past the slice: refused, and counted.
        guard size >= 0, size <= s.workspaceBytes else { s.failures += 1; return }
        if amx_matmul(false, transposeB, 1, &A, &B, &C, size > 0 ? s.workspace : nil, &params) != 0 { s.failures += 1 }
    }

    /// Row-major `A [aRows, aColumns]`, `B` as stored, `C [aRows, cColumns]`, with row strides — one
    /// source for `multiply` and `workspaceBytes`, so the size asked of BNNS is the product's.
    private static func descriptors(a: UnsafeMutablePointer<Float>?, aRows: Int, aColumns: Int, aStride: Int,
                                    b: UnsafeMutablePointer<Float>?, bRows: Int, bColumns: Int, bStride: Int,
                                    c: UnsafeMutablePointer<Float>?, cColumns: Int, cStride: Int)
        -> (a: BNNSNDArrayDescriptor, b: BNNSNDArrayDescriptor, c: BNNSNDArrayDescriptor) {
        func nd(_ p: UnsafeMutablePointer<Float>?, _ rows: Int, _ columns: Int, _ rowStride: Int) -> BNNSNDArrayDescriptor {
            var d = BNNSNDArrayDescriptor()
            d.layout = BNNSDataLayoutRowMajorMatrix
            d.size = (columns, rows, 0, 0, 0, 0, 0, 0)
            d.stride = (1, rowStride, 0, 0, 0, 0, 0, 0)
            d.data = p.map { UnsafeMutableRawPointer($0) }
            d.data_type = .float
            d.data_scale = 1
            d.data_bias = 0
            return d
        }
        return (nd(a, aRows, aColumns, aStride), nd(b, bRows, bColumns, bStride), nd(c, aRows, cColumns, cStride))
    }
}
