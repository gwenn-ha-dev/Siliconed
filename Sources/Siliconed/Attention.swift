import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Synchronization

/// The SDPA through `MPSGraph`, which is a system API — hence zero MSL: we exhaust the API first.
///
/// It cannot be written with `MPSMatrixMultiplication`: at 4128 tokens and 30 heads, the `QKᵀ`
/// matrix weighs 4128² × 30 × 4 bytes = **2.0 GB**. A fused attention is needed, not three GEMMs.
///
/// The mask is passed as `nil`: at 1024² the image tokens number 4096, already a multiple of 32, and
/// the text padding tokens are **semantic and visible** (pitfall 3.8, verified — the reference's
/// mask has an L2 of √S, so it is all ones). The caller must confirm this.
package final class Attention {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let graph: MPSGraph
    private let q: MPSGraphTensor
    private let k: MPSGraphTensor
    private let v: MPSGraphTensor
    private let out: MPSGraphTensor

    package let heads: Int
    /// The heads of `k`/`v`: `heads` without GQA, fewer with it (Krea 2: 48 `q` heads for 12).
    package let kvHeads: Int
    /// Is the graph split into several SDPAs (GQA beyond `maxMatrix`)?
    private let isSplit: Bool
    package let sequence: Int
    /// The length of the keys and values. Equal to `sequence` for a self-attention; different
    /// for a **cross**-attention — Anima's, where 4,096 image tokens query the
    /// 512 text tokens.
    package let keySequence: Int
    package let headDim: Int
    /// The split into query blocks, `0` for the original graph.
    package let queryChunk: Int

    /// Half precision is a **trade**, not a gain, and it is therefore optional.
    ///
    /// The **range** is safe: on the golden tensors, normalized `q` and `k` peak at 8 and 11,
    /// `v` at 185, the output at 149, the maximum score before softmax at 1009 — margins of ×65 to
    /// ×8000. The DiT's overflow (pitfall 3.11) is in the FFN, not here.
    ///
    /// It is the **precision** that is paid for. Measured end to end at 1024², same session:
    ///
    ///                  model_out    7-eval render   GPU SDPA   image vs ref.   worst byte
    ///     fp32         3.14·10⁻⁵      180.8 s        34.5 s      83.74 dB         3/255
    ///     fp16         1.54·10⁻³      156.9 s        19.4 s      51.03 dB        74/255
    ///
    /// The second leaderboard is decided on the image — and **on the image, fp16
    /// passes**: at 1:1 on the face, eyes included, it is indistinguishable from the reference.
    /// Only 53 pixels in a million deviate by 32/255 or more.
    ///
    /// **Yet it is not for the image that we keep the exact path by default, it is for the
    /// instrument.** In fp16, `model_out` rises to 1.54·10⁻³: the bench's noise floor is
    /// multiplied by 50, and `sample`'s guided pass (threshold 5·10⁻⁴) can no longer tell a real
    /// regression from the approximation we have just accepted. We would lose the only thing
    /// that lets us assert the rest is right. fp16 is therefore a *render* option, to be
    /// taken up when the exact levers are exhausted — **never a verification option.**
    /// Read once from the environment: constant for the duration of the process, hence no
    /// shared mutable state to protect.
    package static let useHalfPrecision = EngineSettings.effective.sdpaFp16
    private let precision: MPSDataType

    /// **The transposition is in the graph, not on the CPU.** The engine holds `q`, `k` and `v` as
    /// `[S, H·Dh]` — the layout the GEMM produces. The SDPA wants them as `[H, S, Dh]`. We long
    /// paid for this rearrangement with CPU copies: at 1024², 30 heads × 4,128 tokens = 123,840
    /// copies of 512 bytes per tensor, four tensors, thirty-two blocks — **~16 GB moved per
    /// evaluation at 11.5 GB/s**, on a single thread and with a stride that massacres the cache. Hence a
    /// **superlinear** cost in `S` (×6.2 for ×3.9 tokens) and 9.9 s on a render.
    ///
    /// The buffers are already visible to the GPU (`GEMM.wrap`, unified memory). The rearrangement
    /// therefore has no reason to go through the CPU: `transposeTensor` does it inside the graph,
    /// where it blends into the work the GPU is already doing.
    ///
    /// The conversion to half precision follows the same path: a `cast` in the graph replaces the
    /// fp32 → fp16 copy that the CPU did during the transposition. The caller is left with a
    /// single format to know — **fp32, `[S, H·Dh]`, both ways** — and four arena
    /// slices disappear along with the intermediate buffers.
    /// - Parameter causal: an additive triangular mask, **built once in the graph**.
    ///   The DiT does not need one — its tokens all attend to each other — but the text encoder is a
    ///   `Qwen3ForCausalLM` and does not exist without. The default is `false`, and the graph then builds
    ///   exactly the previous one: the DiT's verified path does not move by a single operation.
    ///
    ///   The mask is **additive** (`0` below the diagonal, `-inf` above) and not
    ///   boolean: `MPSGraph` adds it to the scores before the softmax, which is the reference's
    ///   convention. A boolean mask would be interpreted there as `0`/`1` to be added, hence
    ///   as a tiny bias — wrong, and perfectly plausible to the eye.
    /// - Parameter queryChunk: **the split that does not change the result.**
    ///
    ///   `MPSGraph.scaledDotProductAttention` **materializes** its attention matrix: at 4,128
    ///   tokens and 30 heads, `S²·H·4` = **2.04 GB**, written then read twice — 6.1 GB of traffic
    ///   per call, i.e. ~61 ms at 100 GB/s against ~75 ms of compute. That is exactly the factor
    ///   that separates the measured 1.665 TFLOP/s from a GEMM's 3.5.
    ///
    ///   Now **the softmax is per row**: splitting the queries into blocks, and the heads one by one,
    ///   gives the *same* result — each output row depends only on its own query
    ///   row and on all of `K`/`V`. It is therefore not an approximation, it is a null
    ///   reassociation. And a `Bq × S` slice of a single head weighs `Bq·S·4` bytes: at `Bq = 1024` and
    ///   `S = 4128`, **16.9 MB**, which fit in the M1 Pro's 24 MB of last-level cache.
    ///   The round trips then stop going to DRAM.
    ///
    ///   **Zero MSL**: we exhaust the API before writing a kernel. A measurement settled
    ///   it: the split is exact (worst channel 9.4·10⁻⁸) but **slower, ×0.74 to ×0.81** — slicing the
    ///   heads of a transposed tensor gives a non-contiguous view that MPS materializes. The API path
    ///   is exhausted. This parameter only serves the `sdpa` check's sweep now; splitting for
    ///   **room** lives elsewhere (`maxMatrix` here; the VAE's `vae_bloc_requetes`).
    ///
    ///   `0` keeps the original graph, with a single SDPA. The split is refused with a mask —
    ///   the text encoder is causal and its mask is built for the whole sequence.
    /// - Parameter keySequence: the length of `k` and `v`, `nil` for `sequence`. A cross
    ///   attention is not split here (refused), which leaves a self-attention's graph
    ///   **identical** to the previous one. It may be `causal`: the queries are then **the last
    ///   `sequence` rows of the keys** — query `i` sees the keys `0…keys − sequence + i` —, the
    ///   text segment of a block-causal prefix that attends to everything before it plus its own
    ///   causal triangle (Qwen-Image-2.1, `QwenImage21AttnProcessor`). With `keys == sequence`
    ///   it is the previous causal mask, bit for bit.
    /// - Parameter kvHeads: **GQA without repetition.** `nil`: as many `k`/`v` heads as `q`.
    ///   Otherwise `k` and `v` arrive as the GEMM produces them, `[S, kvHeads·Dh]`, and the SDPA
    ///   receives rank-5 tensors that it **broadcasts**: `q` as `[1, kvHeads, g, S, Dh]` (a
    ///   `reshape` of `[1, H, S, Dh]`, the `g = heads / kvHeads` heads of a group being
    ///   contiguous), `k`/`v` as `[1, kvHeads, 1, S, Dh]`. This is `repeat_interleave(g)` without the
    ///   copy — the `q` head of index `h` reads head `h / g` — and **the same bits** as the
    ///   SDPA on repeated `k`/`v`. Two attempts discarded: laying the `g` heads out on the
    ///   query axis (`[1, kvHeads, g·S, Dh]`), by permutation (+1 GB of footprint at 1024²) or by
    ///   `reshape` after the transposition (+300 MB) — `MPSGraph` materializes an intermediate there.
    ///   Beyond `maxMatrix`, the queries are split into slices (see `isSplit`).
    /// - Parameter visibleKeys: with `causal`, **the keys beyond it are masked for all
    ///   queries** — the right padding of an encoder that keeps it (FLUX.2 [klein] re-reads its 512
    ///   positions). A padding query then sees only the real tokens, never itself:
    ///   it is `transformers`' `causal ∧ attention_mask` mask. `nil`: the previous mask.
    package init(device: MTLDevice, queue: MTLCommandQueue, heads: Int, sequence: Int, headDim: Int,
                causal: Bool = false, queryChunk: Int = 0, keySequence: Int? = nil, kvHeads: Int? = nil,
                visibleKeys: Int? = nil) {
        let keys = keySequence ?? sequence
        let kvHeads = kvHeads ?? heads
        precondition(keys == sequence || queryChunk == 0, "cross-attention: no split")
        precondition(keys >= sequence || !causal, "causal cross-attention: the queries are the last keys")
        precondition(kvHeads == heads || (!causal && queryChunk == 0 && keys == sequence
                                           && heads % kvHeads == 0),
                     "GQA: self-attention without a mask or an imposed split, divisible heads")
        // LOCAL bindings, as in `VAE.swift`: a nested function cannot capture
        // `self` until all properties are initialized, and `out` is the result.
        let precision: MPSDataType = Attention.useHalfPrecision ? .float16 : .float32
        let graph = MPSGraph()
        self.graph = graph
        self.precision = precision
        self.device = device
        self.queue = queue
        self.heads = heads
        self.kvHeads = kvHeads
        self.sequence = sequence
        self.keySequence = keys
        self.headDim = headDim
        // The mask only exists in the causal case, and it is built for the whole sequence.
        self.queryChunk = causal ? 0 : queryChunk
        // What the caller has in memory: `[1, S, H, Dh]`, contiguous.
        let tokenShape: [NSNumber] = [1, NSNumber(value: sequence),
                                      NSNumber(value: heads), NSNumber(value: headDim)]
        let keyShape: [NSNumber] = [1, NSNumber(value: keys),
                                    NSNumber(value: kvHeads), NSNumber(value: headDim)]
        let q = graph.placeholder(shape: tokenShape, dataType: .float32, name: "q")
        let k = graph.placeholder(shape: keyShape, dataType: .float32, name: "k")
        let v = graph.placeholder(shape: keyShape, dataType: .float32, name: "v")
        self.q = q; self.k = k; self.v = v

        /// `[1, S, H, Dh]` → `[1, H, S, Dh]`, then the requested precision.
        func headsFirst(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
            let moved = graph.transposeTensor(tensor, dimension: 1, withDimension: 2, name: nil)
            return precision == .float32 ? moved : graph.cast(moved, to: precision, name: nil)
        }
        // `scale = None` in the reference, hence 1/√headDim.
        var mask: MPSGraphTensor? = nil
        if causal {
            var values = [Float](repeating: 0, count: sequence * keys)
            let visible = min(visibleKeys ?? keys, keys), offset = keys - sequence
            for row in 0..<sequence {
                for column in min(offset + row + 1, visible)..<keys { values[row * keys + column] = -.infinity }
            }
            mask = values.withUnsafeBufferPointer {
                graph.constant(Data(buffer: $0), shape: [1, 1, NSNumber(value: sequence),
                                                         NSNumber(value: keys)],
                               dataType: .float32)
            }
            if precision != .float32 { mask = graph.cast(mask!, to: precision, name: nil) }
        }
        let scale = 1 / Float(headDim).squareRoot()
        let attended: MPSGraphTensor
        var isSplit = false
        if kvHeads < heads {
            // [1, S, H, Dh] → [1, H, S, Dh] (the common transposition) → [1, Hkv, g, S, Dh]: the g
            // heads of a group are contiguous, it is a simple `reshape`; `k`/`v` as
            // [1, Hkv, 1, S, Dh], which the SDPA broadcasts over the g axis.
            let g = heads / kvHeads
            let n = { (x: Int) in NSNumber(value: x) }
            let qg = graph.reshape(headsFirst(q), shape: [1, n(kvHeads), n(g), n(sequence), n(headDim)], name: nil)
            let kh = graph.reshape(headsFirst(k), shape: [1, n(kvHeads), 1, n(keys), n(headDim)], name: nil)
            let vh = graph.reshape(headsFirst(v), shape: [1, n(kvHeads), 1, n(keys), n(headDim)], name: nil)
            // The query rows per SDPA: all of them, as long as the attention matrix fits under
            // `maxMatrix`; otherwise equal slices that fit.
            let rows = sequence
            let slices = Int((Double(heads * rows * keys) * 4 / Double(Attention.maxMatrix))
                .rounded(.up))
            var pieceTensors: [MPSGraphTensor] = []
            var begin = 0
            let length = (rows + slices - 1) / max(1, slices)
            while begin < rows {
                let l = min(length, rows - begin)
                let slice = slices > 1 ? graph.sliceTensor(qg, dimension: 3, start: begin, length: l, name: nil) : qg
                pieceTensors.append(graph.scaledDotProductAttention(query: slice, key: kh, value: vh, mask: nil,
                                                              scale: scale, name: nil))
                begin += l
            }
            isSplit = pieceTensors.count > 1
            let grouped = isSplit ? graph.concatTensors(pieceTensors, dimension: 3, name: nil) : pieceTensors[0]
            // [1, Hkv, g, S, Dh] → [1, H, S, Dh] → [1, S, H, Dh], the common transposition.
            let mergedHeads = graph.reshape(grouped, shape: [1, n(heads), n(sequence), n(headDim)], name: nil)
            let returned = graph.transposeTensor(mergedHeads, dimension: 1, withDimension: 2, name: nil)
            self.isSplit = isSplit
            out = precision == .float32 ? returned : graph.cast(returned, to: .float32, name: nil)
            return
        } else if queryChunk > 0, mask == nil {
            let qh = headsFirst(q), kh = headsFirst(k), vh = headsFirst(v)   // [1, H, S, Dh]
            var perHead: [MPSGraphTensor] = []
            for head in 0..<heads {
                let kOne = graph.sliceTensor(kh, dimension: 1, start: head, length: 1, name: nil)
                let vOne = graph.sliceTensor(vh, dimension: 1, start: head, length: 1, name: nil)
                let qOne = graph.sliceTensor(qh, dimension: 1, start: head, length: 1, name: nil)
                var pieces: [MPSGraphTensor] = []
                var start = 0
                while start < sequence {
                    let length = min(queryChunk, sequence - start)
                    let slice = graph.sliceTensor(qOne, dimension: 2, start: start,
                                                  length: length, name: nil)
                    pieces.append(graph.scaledDotProductAttention(
                        query: slice, key: kOne, value: vOne, mask: nil, scale: scale, name: nil))
                    start += length
                }
                perHead.append(pieces.count == 1 ? pieces[0]
                                                 : graph.concatTensors(pieces, dimension: 2, name: nil))
            }
            attended = perHead.count == 1 ? perHead[0]
                                          : graph.concatTensors(perHead, dimension: 1, name: nil)
        } else {
            attended = graph.scaledDotProductAttention(query: headsFirst(q), key: headsFirst(k),
                                                       value: headsFirst(v), mask: mask,
                                                       scale: scale, name: "sdpa")
        }
        let back = graph.transposeTensor(attended, dimension: 1, withDimension: 2, name: nil)
        out = precision == .float32 ? back : graph.cast(back, to: .float32, name: nil)
        self.isSplit = isSplit
    }

    /// **The largest attention matrix that a single SDPA receives**, in bytes — Krea 2's
    /// at 1024² (48 heads × 4,115², 3.25 GB), the largest measured case that fits in one call. Beyond
    /// (1024×1536: 7.3 GB), `MPSGraph` split on its own and committed into our buffer:
    /// "commit an already committed command buffer". The query slices are exact
    /// up to reassociation (measured: 9.4·10⁻⁸), and are only used beyond: below this threshold, the
    /// graph is that of a single SDPA, bits included.
    package static let maxMatrix = 3_300_000_000

    /// Inputs and output in **fp32**, `[S, H·Dh]`, in buffers that the caller owns.
    package func run(q qBuffer: MTLBuffer, k kBuffer: MTLBuffer, v vBuffer: MTLBuffer,
                    into outBuffer: MTLBuffer) -> Double {
        func data(_ buffer: MTLBuffer, _ length: Int) -> MPSGraphTensorData {
            MPSGraphTensorData(buffer, shape: [1, NSNumber(value: length),
                                               NSNumber(value: heads), NSNumber(value: headDim)],
                               dataType: .float32)
        }
        func kv(_ buffer: MTLBuffer) -> MPSGraphTensorData {
            MPSGraphTensorData(buffer, shape: [1, NSNumber(value: keySequence),
                                               NSNumber(value: kvHeads), NSNumber(value: headDim)],
                               dataType: .float32)
        }
        let feeds = [q: data(qBuffer, sequence), k: kv(kBuffer), v: kv(vBuffer)]
        // **A split graph commits by itself.** With a hundred and twenty SDPAs, `MPSGraph` calls
        // `commitAndContinue`: it replaces the command buffer from under our feet, and the caller's
        // `commit()` then commits an already committed one — a fatal driver assertion. So we go
        // through the synchronous API, which owns its buffers from start to finish.
        //
        // The price is that the returned time becomes **wall-clock** and not GPU. That is accepted: for a
        // graph that splits into a hundred and twenty submissions, the GPU time of a single buffer no
        // longer means anything, and it is the wall-clock time that decides anyway.
        //
        // **And splitting is not the only graph that commits by itself**: the
        // half-precision one does too, because of its `cast`s. The fp16 path therefore crashed on the
        // first SDPA — on `forward 512` as on a render — and nothing caught it, because
        // no close-out check exercises it. *A flag that no check goes through is
        // dead code that looks alive.*
        if queryChunk > 0 || isSplit || Attention.useHalfPrecision {
            let started = DispatchTime.now().uptimeNanoseconds
            graph.run(with: queue, feeds: feeds, targetOperations: nil,
                      resultsDictionary: [out: data(outBuffer, sequence)])
            return Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        }
        let commands = queue.makeCommandBuffer()!
        graph.encode(to: MPSCommandBuffer(commandBuffer: commands),
                     feeds: feeds,
                     targetOperations: nil,
                     resultsDictionary: [out: data(outBuffer, sequence)],
                     executionDescriptor: nil)
        commands.commit()
        commands.waitUntilCompleted()
        return Attention.verdict(commands, what: "SDPA S=\(sequence) H=\(heads)")
    }

    /// **A failing `MTLCommandBuffer` throws nothing**: it leaves the output buffer
    /// as it was — hence full of whatever was there — and returns `gpuEndTime == gpuStartTime`. The
    /// engine then reads values that come from no computation, and the defect shows up three
    /// layers further on as an entirely non-finite image.
    ///
    /// The project had drawn this lesson on a bench kernel; it had never been applied
    /// here nor in `GEMM`, that is, to the **only two submissions of the render path**.
    ///
    /// We do not throw: these calls are deep inside non-throwing loops. We count, and say so
    /// loudly — once per cause, so as not to drown the output under thirty-two repetitions.
    static func verdict(_ commands: MTLCommandBuffer, what: @autoclosure () -> String) -> Double {
        let elapsed = commands.gpuEndTime - commands.gpuStartTime
        if let error = commands.error {
            Attention.report("\(what()): the GPU refused — \(error.localizedDescription)")
        } else if elapsed <= 0 {
            Attention.report("\(what()): zero GPU time, so nothing ran")
        }
        return elapsed
    }

    /// The failures counted, and the causes already reported so as to mention each only once. Under
    /// a lock: a single queue renders, but a check can call from elsewhere, and a bare counter
    /// shared between threads is a data race even when it is rare.
    private static let state = Mutex<(reported: Set<String>, failures: Int)>(([], 0))
    package static var failures: Int { state.withLock { $0.failures } }
    static func report(_ message: String) {
        let key = String(message.prefix(60))
        let isNew = state.withLock { item -> Bool in
            item.failures += 1
            return item.reported.insert(key).inserted
        }
        if isNew { Warnings.emit("✗ \(message)") }
    }

    /// **The count, read at the end of a render** (`Engine.execute`, `Diagnostic`): the causes are said
    /// once per process, so a second render failing the same way would otherwise say nothing. Here
    /// every render that saw a failure ends on one line in its journal; `nil` when none failed.
    @discardableResult
    package static func summarize(since before: Int) -> String? {
        let failed = failures - before
        guard failed > 0 else { return nil }
        let message = "\(failed) GPU or AMX submission(s) failed during this run: its output did not come from a computation"
        Warnings.emit("✗ \(message)")
        return message
    }
}
