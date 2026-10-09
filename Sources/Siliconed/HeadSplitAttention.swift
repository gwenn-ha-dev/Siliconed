import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph
import Synchronization

/// **The SDPA cut by heads between the GPU and the AMX** — Z-Image's only SDPA path at
/// S ≥ 1024 (judged at threshold on its goldens; md5 of the 512² seed-42 render `798d5497`).
///
///     q, k, v   [S, H·Dh] fp32, as the GEMMs wrote them (unified memory, no copy)
///        │
///        ├─ heads [0, g)  GPU   MPSGraph: slice heads → [g, S, Dh] → SDPA → [S, g, Dh] → pad a zero heads ─► out [S, H·Dh]
///        │
///        └─ heads [g, H)  CPU   per block of `rows` queries of one head, any thread:              ─► spill [S, a·Dh]
///                                 scores = Q_b · Kᵀ                    BNNSMatMul, 1 thread (AMX)
///                                 p = exp(scale·s − scale·max s)       vDSP_maxv, vDSP_vsmsa, vvexpf
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
/// - **The CPU's block boundaries are fixed by position** (`rows`-multiples), not by thread: a block
///   is handed to whichever thread is free, and the bits do not depend on which one took it.
/// - **`MPSGraph` cannot write a strided result**, so the GPU writes the whole `[S, H·Dh]` (its heads,
///   then zeros) and the CPU heads wait in a spill until the GPU is done. The copy is `a/H` of the
///   output, ~27 MB at 1024² — a few ms against the ~60 ms saved.
/// - **The softmax is not free on the CPU.** It is 17 M `exp` per head at S = 4128 — half of a head's
///   time on one thread (GEMMs 9.9 ms, softmax 10.5 ms); it is why the CPU part wants all the
///   P cores and not only one thread per AMX block, unlike the GEMM cut.
/// - **BNNS on one thread per call.** Several threads share one AMX block per cluster; the work
///   is parallelized here, by blocks, and `n_threads = 1` keeps BNNS from spawning its own.
///   Not `cblas_sgemm`: its threading switch (`BLASSetThreading`) is process-wide, and the engine
///   has other `cblas` users.
package final class HeadSplitAttention {
    package let heads: Int
    package let sequence: Int
    package let headDim: Int
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

    /// **Which heads go where, defined once.** `fraction` is the share of heads handed to the CPU,
    /// rounded to the nearest head; the GPU keeps at least one head, the CPU may get none.
    package static func partition(heads: Int, fraction: Double) -> (gpu: Range<Int>, cpu: Range<Int>) {
        let cpu = max(0, min(heads - 1, Int((Double(heads) * fraction).rounded())))
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
    private let next = Atomic<Int>(0)
    /// The CPU heads' output, `[S, a·Dh]`, copied into `out` once the GPU is done.
    private let spill: UnsafeMutablePointer<Float>
    private let scratch: [Scratch]

    /// One thread's scores block and BNNS workspace.
    private final class Scratch {
        let scores: UnsafeMutablePointer<Float>
        let sums: UnsafeMutablePointer<Float>
        var workspace: UnsafeMutableRawPointer?
        var workspaceBytes = 0
        var failures = 0
        init(rows: Int, keys: Int) {
            scores = .allocate(capacity: rows * keys)
            sums = .allocate(capacity: rows)
        }
        deinit { scores.deallocate(); sums.deallocate(); workspace?.deallocate() }
        func space(_ bytes: Int) -> UnsafeMutableRawPointer? {
            if bytes > workspaceBytes {
                workspace?.deallocate()
                workspace = .allocate(byteCount: bytes, alignment: 128); workspaceBytes = bytes
            }
            return bytes > 0 ? workspace : nil
        }
    }

    /// **The share of heads handed to the CPU, frozen: 12 of Z-Image's 30.** Not a setting:
    /// a fraction decides the bits, and one version lives. It was swept in the engine at S = 4128 —
    /// 18 GPU / 12 CPU ahead (155.7 → 95.9 ms) — and judged at threshold on every Z-Image
    /// golden, then adopted it. Thread count, by contrast, changes the speed and never the bits.
    package static let cpuFraction = 0.4

    /// - Parameter cpuFraction: the share of heads handed to the CPU; the product's is `cpuFraction`,
    ///   other values serve the `sdpa` check's sweep.
    package init(device: MTLDevice, queue: MTLCommandQueue, heads: Int, sequence: Int, headDim: Int,
                 cpuFraction: Double = HeadSplitAttention.cpuFraction,
                 threads: Int = EngineSettings.effective.machine.performanceCores) {
        let (gpu, cpu) = HeadSplitAttention.partition(heads: heads, fraction: cpuFraction)
        self.queue = queue
        self.heads = heads; self.sequence = sequence; self.headDim = headDim
        self.gpuHeads = gpu.count
        self.threads = max(1, threads)
        work = HeadSplitAttention.items(cpu: cpu, sequence: sequence, rows: HeadSplitAttention.blockRows)
        spill = .allocate(capacity: max(1, sequence * cpu.count * headDim))
        scratch = (0..<max(1, threads)).map { _ in Scratch(rows: HeadSplitAttention.blockRows, keys: sequence) }

        let graph = MPSGraph()
        self.graph = graph
        let n = { (x: Int) in NSNumber(value: x) }
        let shape: [NSNumber] = [1, n(sequence), n(heads), n(headDim)]
        let q = graph.placeholder(shape: shape, dataType: .float32, name: "q")
        let k = graph.placeholder(shape: shape, dataType: .float32, name: "k")
        let v = graph.placeholder(shape: shape, dataType: .float32, name: "v")
        self.q = q; self.k = k; self.v = v
        func gpuPart(_ t: MPSGraphTensor) -> MPSGraphTensor {
            let sliced = cpu.isEmpty ? t : graph.sliceTensor(t, dimension: 2, start: 0, length: gpu.count, name: nil)
            return graph.transposeTensor(sliced, dimension: 1, withDimension: 2, name: nil)
        }
        let attended = graph.scaledDotProductAttention(query: gpuPart(q), key: gpuPart(k), value: gpuPart(v),
                                                       mask: nil, scale: 1 / Float(headDim).squareRoot(), name: "sdpa")
        let back = graph.transposeTensor(attended, dimension: 1, withDimension: 2, name: nil)
        out = cpu.isEmpty ? back : graph.padTensor(back, with: .constant, leftPadding: [0, 0, 0, 0],
                                                   rightPadding: [0, 0, n(cpu.count), 0], constantValue: 0, name: nil)
    }

    deinit { spill.deallocate() }

    /// Inputs and output fp32 `[S, H·Dh]`: the buffers for the GPU, the same memory as pointers for
    /// the CPU. Returns the GPU time and the CPU part's wall time.
    package func run(q qBuffer: MTLBuffer, k kBuffer: MTLBuffer, v vBuffer: MTLBuffer, into outBuffer: MTLBuffer,
                     q qPointer: UnsafePointer<Float>, k kPointer: UnsafePointer<Float>, v vPointer: UnsafePointer<Float>,
                     out outPointer: UnsafeMutablePointer<Float>) -> (gpu: Double, cpu: Double) {
        let n = { (x: Int) in NSNumber(value: x) }
        let shape: [NSNumber] = [1, n(sequence), n(heads), n(headDim)]
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
                     feeds: [q: data(qBuffer, shape), k: data(kBuffer, shape), v: data(vBuffer, shape)],
                     targetOperations: nil, resultsDictionary: [out: data(outBuffer, shape)], executionDescriptor: nil)
        commands.commit()

        let started = DispatchTime.now().uptimeNanoseconds
        if !work.isEmpty {
            next.store(0, ordering: .relaxed)
            HeadSplitAttention.perform(threads) { t in
                let s = scratch[t]
                while true {
                    let i = next.wrappingAdd(1, ordering: .relaxed).oldValue
                    guard i < work.count else { return }
                    block(work[i], s, q: qPointer, k: kPointer, v: vPointer)
                }
            }
        }
        let cpu = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9

        commands.waitUntilCompleted()
        let root = commands.rootCommandBuffer
        root.waitUntilCompleted()
        let what = "SDPA (head split) S=\(sequence) H=\(gpuHeads)/\(heads)"
        let gpu: Double
        if root.error != nil || root.gpuEndTime > root.gpuStartTime {
            gpu = Attention.verdict(root, what: what)
        } else {
            // Committed by the graph itself (1024×1536): the work ran in buffers MPS made, whose GPU
            // time we do not hold — wall time, as `Attention` does for its split graphs. The image
            // stays at 77.9 dB of the graph alone, as at 512².
            if outPointer[probe].bitPattern == sentinel { Attention.report("\(what): the output was never written") }
            gpu = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        }
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
        let spilledWidth = cpuHeads * headDim, spilledColumn = (item.head - gpuHeads) * headDim
        let qBlock = UnsafeMutablePointer(mutating: q) + item.row * width + column
        let kHead = UnsafeMutablePointer(mutating: k) + column
        let vHead = UnsafeMutablePointer(mutating: v) + column
        let o = spill + item.row * spilledWidth + spilledColumn
        // scores [rows, S] = Q_b [rows, Dh] · K_hᵀ, K_h being [S, Dh] at a row stride of H·Dh
        multiply(s, a: qBlock, aRows: item.rows, aColumns: headDim, aStride: width,
                 b: kHead, bRows: sequence, bColumns: headDim, bStride: width, transposeB: true,
                 c: s.scores, cColumns: sequence, cStride: sequence)
        let scale = 1 / Float(headDim).squareRoot()
        var count = Int32(sequence)
        for r in 0..<item.rows {
            let row = s.scores + r * sequence
            var maximum: Float = 0
            vDSP_maxv(row, 1, &maximum, vDSP_Length(sequence))
            var a = scale, b = -scale * maximum
            vDSP_vsmsa(row, 1, &a, &b, row, 1, vDSP_Length(sequence))
            vvexpf(row, row, &count)
            vDSP_sve(row, 1, s.sums + r, vDSP_Length(sequence))
        }
        // O_b [rows, Dh] = P [rows, S] · V_h [S, Dh], then each row divided by its Σp
        multiply(s, a: s.scores, aRows: item.rows, aColumns: sequence, aStride: sequence,
                 b: vHead, bRows: sequence, bColumns: headDim, bStride: width, transposeB: false,
                 c: o, cColumns: headDim, cStride: spilledWidth)
        for r in 0..<item.rows {
            vDSP_vsdiv(o + r * spilledWidth, 1, s.sums + r, o + r * spilledWidth, 1, vDSP_Length(headDim))
        }
    }

    /// `C = A · op(B)` on one thread, row-major, with row strides. `B` is described as stored
    /// (`bRows × bColumns`); `transposeB` multiplies by its transpose.
    private func multiply(_ s: Scratch, a: UnsafeMutablePointer<Float>, aRows: Int, aColumns: Int, aStride: Int,
                          b: UnsafeMutablePointer<Float>, bRows: Int, bColumns: Int, bStride: Int, transposeB: Bool,
                          c: UnsafeMutablePointer<Float>, cColumns: Int, cStride: Int) {
        func nd(_ p: UnsafeMutablePointer<Float>, _ rows: Int, _ columns: Int, _ rowStride: Int) -> BNNSNDArrayDescriptor {
            var d = BNNSNDArrayDescriptor()
            d.layout = BNNSDataLayoutRowMajorMatrix
            d.size = (columns, rows, 0, 0, 0, 0, 0, 0)
            d.stride = (1, rowStride, 0, 0, 0, 0, 0, 0)
            d.data = UnsafeMutableRawPointer(p)
            d.data_type = .float
            d.data_scale = 1
            d.data_bias = 0
            return d
        }
        var A = nd(a, aRows, aColumns, aStride)
        var B = nd(b, bRows, bColumns, bStride)
        var C = nd(c, aRows, cColumns, cStride)
        var params = BNNSFilterParameters()
        params.n_threads = 1
        let size = amx_matmul_workspace(false, transposeB, 1, &A, &B, &C, &params)
        guard size >= 0 else { s.failures += 1; return }
        if amx_matmul(false, transposeB, 1, &A, &B, &C, s.space(size), &params) != 0 { s.failures += 1 }
    }
}
