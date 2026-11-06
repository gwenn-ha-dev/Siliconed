import Foundation
import Metal

/// **One decoded token on the GPU, from the weights where they lie**: the step of `Qwen3VLGenerator` that reads every
/// weight once for one new row (`DecodeKernels` says how), as a few command buffers instead of 36 layers of CPU
/// calls around products that first copy their weight (7 GB of int8 a token) into a reserve.
///
///     hidden (the row's embedding) ─ 36 × layer ─ final RMSNorm ─ lm_head ─ argmax ─▶ Pick (id, top 1, top 2)
///     layer: norm+rotate ─ q, k, v ─ QK-norm, RoPE, K/V appended to the cache ─ attention over it ─ o ─ norm+rotate ─ gate, up ─ SwiGLU ─ down
///
/// The layers' tensors are read where they lie and wrapped as `MTLBuffer`s without a copy (one per tensor: a page-aligned
/// base and length, each below `maxBufferLength`): the map's own pages, or the anonymous pages of `Artifact.keepResident`
/// (tests and checks). A mapped tensor's pages are wired by Metal once the GPU has read them, until the buffer goes (measured:
/// the file cache shrinks by as much as the wired memory grows, and the release frees them rather than handing them back), so
/// the decoder is memory for as long as it lives: the writer builds one per run (`PromptWriter`). The layers from `Tail.first` on are **streamed**:
/// each is read with `pread` + `F_NOCACHE` into one of a few staging buffers while the GPU computes the layer before it
/// (`TailFeeder`), so the cache keeps the head of the map and a token never reads the tail through it. Resident
/// anonymous buffers are held for the GPU by a `MTLResidencySet` on the decoder's own queue — Metal does not re-validate
/// 250 buffers at each command buffer; the map's pages are not put in it (`Tuning.pinsWeights`: pinning would wire them,
/// and a wired page is no longer the cache's to drop). The K/V cache is the generator's, wrapped the same way (`attach`). What the decoder owns is small: its scratch arena
/// (`scratchBytes`, charged by `PromptWriterNeed`) — the row, the rotated copies, the projections' partial sums,
/// the logits, the norm weights widened once (bf16 → fp32 is exact).
///
/// What a port by analogy would miss:
///
///   - **Only whole-product precision is kept, not the CPU's order of sums**: the products accumulate in fp32 over a
///     K range per threadgroup and the ranges are added in a fixed order — the widened path's arithmetic class
///     (integers exact, fp32 sums), other last bits. A greedy token can differ from the CPU path only where two logits are
///     within rounding (the gap is reported: `Pick.gap`).
///   - **A token is a few command buffers, committed as soon as they are encoded**: the encoding of 470 dispatches costs a few
///     ms of CPU, which the GPU would otherwise wait for; the first buffers run while the next are encoded
///     (`layersPerCommand`). The queue is serial, so the order is the commit order.
///   - **Independent products run concurrently** (`MTLDispatchType.concurrent`, q/k/v and gate/up), every dependency
///     is an explicit barrier; the residual adds are fused into the next norm, so no kernel reads a half-written row.
///   - **The embedding row and the RoPE row are the CPU's** (the table is the map's mapped file, one row a token; the
///     table of frequencies is `TextEncoder.writeRope`'s) — written into `hidden` and `frequencies` before a step.
///   - **Head dimension 128 and a Metal device**: anything else, a weight the kernels do not read (a 4-bit, a fp8, a
///     per-tensor scale, mixed rotation groups on products that share an input) makes `make` return `nil`, and the generator
///     keeps its CPU step.
package final class Qwen3VLDecoder {
    /// The shapes of the model, all a decode step needs of its config.
    package struct Shape: Equatable {
        package let hidden, layers, heads, kvHeads, headDim, intermediate, vocabulary: Int
        package let eps: Float
        package init(hidden: Int, layers: Int, heads: Int, kvHeads: Int, headDim: Int, intermediate: Int, vocabulary: Int, eps: Float) {
            self.hidden = hidden; self.layers = layers; self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
            self.intermediate = intermediate; self.vocabulary = vocabulary; self.eps = eps
        }
        package init(_ config: TextEncoder.Config, vocabulary: Int) {
            self.init(hidden: config.hidden, layers: config.layers, heads: config.heads, kvHeads: config.kvHeads,
                      headDim: config.headDim, intermediate: config.intermediate, vocabulary: vocabulary, eps: config.eps)
        }
        var attentionWidth: Int { heads * headDim }
        var kvWidth: Int { kvHeads * headDim }
    }

    /// A layer's products, in the order they are applied.
    package struct Layer {
        package var q, k, v, o, gate, up, down: DecodeKernels.Matrix
        package init(q: DecodeKernels.Matrix, k: DecodeKernels.Matrix, v: DecodeKernels.Matrix, o: DecodeKernels.Matrix,
                     gate: DecodeKernels.Matrix, up: DecodeKernels.Matrix, down: DecodeKernels.Matrix) {
            self.q = q; self.k = k; self.v = v; self.o = o; self.gate = gate; self.up = up; self.down = down
        }
    }

    /// **The next token**: the first maximum of the logits (`torch.argmax`) and the two largest values.
    package struct Pick: Equatable {
        package let id: Int
        package let value: Float, second: Float
        /// How far greedy was from another choice; 0 on a tie.
        package var gap: Float { value - second }
    }

    package enum Failure: Error, CustomStringConvertible {
        case unsupported(String)
        case gpu(String)
        package var description: String {
            switch self {
            case .unsupported(let why): return "GPU decoding: \(why)"
            case .gpu(let why): return "GPU decoding: \(why)"
            }
        }
    }

    /// What can be tuned without touching the arithmetic of a step's result beyond the order of the K-range sums.
    package struct Tuning {
        /// Threadgroups a product is cut in, at least (`DecodeKernels.splits`). 48 to 192 measured within the noise
        /// (58–63 ms a token on synthetic weights of the real shapes); 288 and over was slower.
        package var targetGroups = 96
        /// Layers encoded into one command buffer; the buffer is committed as soon as it is encoded.
        package var layersPerCommand = 12
        /// Whether the weights' buffers are made resident for the GPU (`MTLResidencySet.requestResidency`): right for pages the
        /// process owns, wrong for the map's — it would wire the cache's pages.
        package var pinsWeights = true
        package init() {}
    }

    /// **The streamed layers**: `first` to the last are read from `path` into `slots` staging buffers (`TailFeeder`).
    /// `runs[l - first]` is layer `l`'s run in the file (page-rounded, `Artifact.stagingRun`); `slotBytes` is the largest.
    package struct Tail {
        package let path: String
        package let first: Int
        package let runs: [(start: Int, end: Int)]
        package let slots: Int
        package var slotBytes: Int { runs.map { $0.end - $0.start }.max() ?? 0 }
        package init(path: String, first: Int, runs: [(start: Int, end: Int)], slots: Int) {
            self.path = path; self.first = first; self.runs = runs; self.slots = slots
        }
    }

    package let shape: Shape
    package let tuning: Tuning
    package let kernels: DecodeKernels
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let layers: [Layer]
    private let feeder: TailFeeder?
    private let head: MTLBuffer
    private let arena: Arena
    private let residency: MTLResidencySet?

    // The scratch, by the name `slices` gives it.
    private let xBuffer, freqsBuffer, rotHidden, rotAttended, rotIntermediate, qBuffer, attended, norms, logits, result: MTLBuffer
    private let partsQ, partsK, partsV, partsO, partsGate, partsUp, partsDown: MTLBuffer
    private let splitsQ, splitsK, splitsV, splitsO, splitsGate, splitsUp, splitsDown: Int
    private let normBase: UnsafeMutablePointer<Float>

    /// The row a step starts from (the embedding) and, after it, the residual stream; `[hidden]`.
    package let hidden: UnsafeMutablePointer<Float>
    /// The RoPE row of the step, `[headDim]` as (cos, sin) pairs (`TextEncoder.writeRope` with one position).
    package let frequencies: UnsafeMutablePointer<Float>

    private var cache: (buffer: MTLBuffer, layerStride: Int)?
    /// The last step's wall seconds and the GPU's (the sum over its command buffers).
    package private(set) var lastStep: (wall: Double, gpu: Double) = (0, 0)

    // MARK: - Memory

    /// **The scratch arena's slices, in floats**, in the order `init` reserves them — one function for the allocation and
    /// the memory need. The partial sums are `splits × n` of each product (`DecodeKernels.splits`).
    package static func slices(_ s: Shape, tuning: Tuning = Tuning()) -> [(name: String, count: Int)] {
        let (h, a, kv, i) = (s.hidden, s.attentionWidth, s.kvWidth, s.intermediate)
        func parts(_ k: Int, _ n: Int) -> Int { DecodeKernels.splits(k: k, n: n, target: tuning.targetGroups) * n }
        return [("x", h), ("freqs", s.headDim), ("rotHidden", h), ("rotAttended", a), ("rotIntermediate", i), ("q", a),
                ("attended", a), ("norms", normCount(s)), ("logits", s.vocabulary), ("result", 4),
                ("partsQ", parts(h, a)), ("partsK", parts(h, kv)), ("partsV", parts(h, kv)), ("partsO", parts(a, h)),
                ("partsGate", parts(h, i)), ("partsUp", parts(h, i)), ("partsDown", parts(i, h))]
    }

    /// The decoder's own memory, page-rounded as `Arena` maps it.
    package static func scratchBytes(_ s: Shape, tuning: Tuning = Tuning()) -> Int {
        Arena.capacity(slices: slices(s, tuning: tuning).map { $0.count * 4 })
    }

    /// The norm weights, widened once: per layer `input_layernorm`, `post_attention_layernorm` (hidden each), `q_norm`,
    /// `k_norm` (head dimension each), then the final norm.
    package static func normCount(_ s: Shape) -> Int { s.layers * (2 * s.hidden + 2 * s.headDim) + s.hidden }

    /// Where a layer's norm weights start in the table (floats). `part`: 0 input, 1 post-attention, 2 q-norm, 3 k-norm.
    package static func normOffset(_ s: Shape, layer: Int, part: Int) -> Int {
        let base = layer * (2 * s.hidden + 2 * s.headDim)
        switch part {
        case 0: return base
        case 1: return base + s.hidden
        case 2: return base + 2 * s.hidden
        default: return base + 2 * s.hidden + s.headDim
        }
    }
    package static func finalNormOffset(_ s: Shape) -> Int { s.layers * (2 * s.hidden + 2 * s.headDim) }

    // MARK: - Construction

    /// - Parameters:
    ///   - head: `lm_head` `[vocabulary, hidden]` bf16, as published, page-aligned.
    ///   - fillNorms: writes the norm table (`normCount` floats, laid out by `normOffset`) — the layers' widened
    ///     `*_layernorm`, `q_norm` and `k_norm`, then the final norm.
    package init(device: MTLDevice, shape: Shape, layers: [Layer], head: MTLBuffer, tuning: Tuning = Tuning(),
                 feeder: TailFeeder? = nil, fillNorms: (UnsafeMutablePointer<Float>) throws -> Void) throws {
        guard shape.headDim == 128 else { throw Failure.unsupported("head dimension \(shape.headDim), 128 expected") }
        guard layers.count == shape.layers else { throw Failure.unsupported("\(layers.count) layers of \(shape.layers)") }
        guard shape.hidden % 512 == 0, shape.hidden <= 8192, shape.attentionWidth % 4 == 0 else {
            throw Failure.unsupported("hidden width \(shape.hidden)")
        }
        let (h, a, kv, i) = (shape.hidden, shape.attentionWidth, shape.kvWidth, shape.intermediate)
        for (n, l) in layers.enumerated() {
            let want = [(l.q, h, a), (l.k, h, kv), (l.v, h, kv), (l.o, a, h), (l.gate, h, i), (l.up, h, i), (l.down, i, h)]
            for (m, k, width) in want where m.k != k || m.n != width || !m.isReadable {
                throw Failure.unsupported("layer \(n): a [\(m.k), \(m.n)] product where [\(k), \(width)] is read")
            }
            // q, k, v read one rotated row, gate and up another: their groups must agree.
            guard l.q.group == l.k.group, l.q.group == l.v.group, l.gate.group == l.up.group else {
                throw Failure.unsupported("layer \(n): products sharing an input rotate it by different groups")
            }
        }
        self.device = device; self.shape = shape; self.tuning = tuning; self.layers = layers; self.head = head; self.feeder = feeder
        guard let queue = device.makeCommandQueue() else { throw Failure.gpu("no command queue") }
        self.queue = queue
        kernels = try DecodeKernels(device: device)

        let plan = Self.slices(shape, tuning: tuning)
        let arena = try Arena(capacity: Arena.capacity(slices: plan.map { $0.count * 4 }))
        self.arena = arena
        var pointers: [String: UnsafeMutableRawPointer] = [:], buffers: [String: MTLBuffer] = [:]
        for (name, count) in plan {
            pointers[name] = try arena.reserve(name, bytes: count * 4)
            buffers[name] = try arena.buffer(name, device: device)
        }
        xBuffer = buffers["x"]!; freqsBuffer = buffers["freqs"]!; rotHidden = buffers["rotHidden"]!
        rotAttended = buffers["rotAttended"]!; rotIntermediate = buffers["rotIntermediate"]!; qBuffer = buffers["q"]!
        attended = buffers["attended"]!; norms = buffers["norms"]!; logits = buffers["logits"]!; result = buffers["result"]!
        partsQ = buffers["partsQ"]!; partsK = buffers["partsK"]!; partsV = buffers["partsV"]!; partsO = buffers["partsO"]!
        partsGate = buffers["partsGate"]!; partsUp = buffers["partsUp"]!; partsDown = buffers["partsDown"]!
        hidden = pointers["x"]!.assumingMemoryBound(to: Float.self)
        frequencies = pointers["freqs"]!.assumingMemoryBound(to: Float.self)
        normBase = pointers["norms"]!.assumingMemoryBound(to: Float.self)
        let target = tuning.targetGroups
        splitsQ = DecodeKernels.splits(k: h, n: a, target: target); splitsK = DecodeKernels.splits(k: h, n: kv, target: target)
        splitsV = splitsK; splitsO = DecodeKernels.splits(k: a, n: h, target: target)
        splitsGate = DecodeKernels.splits(k: h, n: i, target: target); splitsUp = splitsGate
        splitsDown = DecodeKernels.splits(k: i, n: h, target: target)
        try fillNorms(normBase)

        // Everything the GPU reads stays resident for it, once, on the decoder's queue.
        let descriptor = MTLResidencySetDescriptor()
        descriptor.initialCapacity = 7 * layers.count + buffers.count + 2
        if let set = try? device.makeResidencySet(descriptor: descriptor) {
            var seen = Set<ObjectIdentifier>()
            func add(_ b: MTLBuffer) { if seen.insert(ObjectIdentifier(b)).inserted { set.addAllocation(b) } }
            if tuning.pinsWeights {
                for l in layers { for m in [l.q, l.k, l.v, l.o, l.gate, l.up, l.down] { add(m.buffer); if let s = m.scales { add(s) } } }
                add(head)
            }
            // The staging buffers are anonymous and few: always pinned.
            for slot in feeder?.buffers ?? [] { add(slot) }
            for b in buffers.values { add(b) }
            set.commit(); set.requestResidency()
            queue.addResidencySet(set)
            residency = set
        } else {
            residency = nil
        }
    }

    deinit { if let residency { queue.removeResidencySet(residency) } }

    /// **Binds the sequence's K/V cache** (`Pages`: keys of layer `l` at `l · layerStride` floats, values at
    /// `(layers + l) · layerStride`), wrapped without a copy. Replaces the previous sequence's.
    package func attach(cache pages: Pages, layerStride: Int) throws {
        if let old = cache?.buffer { residency?.removeAllocation(old) }
        guard let buffer = device.makeBuffer(bytesNoCopy: pages.base, length: pages.capacity, options: .storageModeShared,
                                             deallocator: nil) else { throw Failure.gpu("the K/V cache cannot be wrapped") }
        buffer.label = "K/V cache"
        MetalWraps.track(buffer)
        residency?.addAllocation(buffer); residency?.commit()
        cache = (buffer, layerStride)
    }

    /// Lets go of the cache's view before its pages are given back (a buffer over freed pages must not stay in the
    /// residency set).
    package func detachCache() {
        if let old = cache?.buffer { residency?.removeAllocation(old); residency?.commit() }
        cache = nil
    }

    // MARK: - Steps

    /// **One decoded token**: `hidden` (the embedding of the token at cache row `row`) and `frequencies` (its RoPE row)
    /// are written; runs the layers (K and V appended at `row`, attention over rows `0...row`), the final norm and the head.
    /// On return `hidden` is the residual stream after the last layer.
    package func step(row: Int) throws -> Pick {
        try run(row: row, layers: true)
    }

    /// **The head alone** on `hidden` as it stands: the first token, once the prompt has been read (the prefill's last row).
    package func pick() throws -> Pick {
        try run(row: 0, layers: false)
    }

    private func run(row: Int, layers withLayers: Bool) throws -> Pick {
        let s = shape, d = kernels
        let (h, kv) = (s.hidden, s.kvWidth)
        let started = DispatchTime.now().uptimeNanoseconds
        var buffers: [MTLCommandBuffer] = []
        func encoder() throws -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
            guard let commands = queue.makeCommandBuffer(), let e = commands.makeComputeCommandEncoder(dispatchType: .concurrent) else {
                throw Failure.gpu("no command buffer")
            }
            return (commands, e)
        }
        func barrier(_ e: MTLComputeCommandEncoder) { e.memoryBarrier(scope: .buffers) }
        func submit(_ commands: MTLCommandBuffer, _ e: MTLComputeCommandEncoder, then: (() -> Void)? = nil) {
            e.endEncoding()
            if let then { commands.addCompletedHandler { _ in then() } }
            commands.commit(); buffers.append(commands)
        }
        // The streamed tail's reads start with the step: the first layers are read while the cached ones compute. A step that
        // ends early (an error) lets the reads go (`abort`) so that nothing waits on a layer no one will compute.
        let pass = withLayers ? feeder?.begin() : nil
        defer { pass?.abort() }

        var (commands, e) = try encoder()
        if withLayers {
            guard let cache else { throw Failure.gpu("no K/V cache attached") }
            let length = row + 1
            let layerBytes = cache.layerStride * 4
            for l in 0..<s.layers {
                let w = layers[l]
                let slot = feeder.flatMap { l >= $0.first ? (l - $0.first) % $0.slotCount : nil }
                if slot != nil, let pass {
                    // What the GPU has been given so far goes first; the wait for the read is the CPU's, not the GPU's.
                    if l == feeder!.first && l > 0 { submit(commands, e); (commands, e) = try encoder() }
                    try pass.wait(layer: l)
                }
                let rotation = w.q.group, post = w.gate.group
                // The residual of the previous layer's down projection is added by this norm.
                d.normRotate(e, x: xBuffer, parts: l == 0 ? nil : partsDown, count: splitsDown, stride: h,
                             weight: norms, weightOffset: 4 * Self.normOffset(s, layer: l, part: 0), out: rotHidden, n: h,
                             group: rotation, eps: s.eps)
                barrier(e)
                d.gemv(e, w.q, x: rotHidden, parts: partsQ, splits: splitsQ)
                d.gemv(e, w.k, x: rotHidden, parts: partsK, splits: splitsK)
                d.gemv(e, w.v, x: rotHidden, parts: partsV, splits: splitsV)
                barrier(e)
                let keysAt = l * layerBytes + row * kv * 4, valuesAt = (s.layers + l) * layerBytes + row * kv * 4
                d.post(e, q: (partsQ, splitsQ), k: (partsK, splitsK), v: (partsV, splitsV), qNorm: norms, kNorm: norms,
                       normOffsets: (4 * Self.normOffset(s, layer: l, part: 2), 4 * Self.normOffset(s, layer: l, part: 3)),
                       freqs: freqsBuffer, out: qBuffer, keys: (cache.buffer, keysAt), values: (cache.buffer, valuesAt),
                       heads: s.heads, kvHeads: s.kvHeads, headDim: s.headDim, eps: s.eps)
                barrier(e)
                d.attend(e, q: qBuffer, keys: (cache.buffer, l * layerBytes), values: (cache.buffer, (s.layers + l) * layerBytes),
                         out: attended, heads: s.heads, kvHeads: s.kvHeads, length: length)
                barrier(e)
                var input = attended
                if w.o.group > 0 {
                    d.rotate(e, attended, out: rotAttended, n: s.attentionWidth, group: w.o.group)
                    barrier(e)
                    input = rotAttended
                }
                d.gemv(e, w.o, x: input, parts: partsO, splits: splitsO)
                barrier(e)
                d.normRotate(e, x: xBuffer, parts: partsO, count: splitsO, stride: h,
                             weight: norms, weightOffset: 4 * Self.normOffset(s, layer: l, part: 1), out: rotHidden, n: h,
                             group: post, eps: s.eps)
                barrier(e)
                d.gemv(e, w.gate, x: rotHidden, parts: partsGate, splits: splitsGate)
                d.gemv(e, w.up, x: rotHidden, parts: partsUp, splits: splitsUp)
                barrier(e)
                d.swiglu(e, gate: partsGate, gateParts: splitsGate, up: partsUp, upParts: splitsUp, out: rotIntermediate,
                         n: s.intermediate, group: w.down.group)
                barrier(e)
                d.gemv(e, w.down, x: rotIntermediate, parts: partsDown, splits: splitsDown)
                barrier(e)
                if let slot, let pass {
                    // One command buffer a streamed layer: its completion gives the staging buffer back to the reads.
                    submit(commands, e) { pass.release(slot: slot) }
                    (commands, e) = try encoder()
                } else if (l + 1) % tuning.layersPerCommand == 0 && l + 1 < s.layers {
                    submit(commands, e)
                    (commands, e) = try encoder()
                }
            }
        }
        // The final norm (with the last layer's residual), the head, the argmax.
        d.normRotate(e, x: xBuffer, parts: withLayers ? partsDown : nil, count: splitsDown, stride: h,
                     weight: norms, weightOffset: 4 * Self.finalNormOffset(s), out: rotHidden, n: h, group: 0, eps: s.eps)
        barrier(e)
        d.headProduct(e, head: head, rows: s.vocabulary, k: h, x: rotHidden, logits: logits)
        barrier(e)
        d.argmaxTop2(e, logits: logits, rows: s.vocabulary, result: result)
        submit(commands, e)
        commands.waitUntilCompleted()
        var gpu = 0.0
        for c in buffers {
            if let error = c.error { throw Failure.gpu(error.localizedDescription) }
            gpu += c.gpuEndTime - c.gpuStartTime
        }
        lastStep = (Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9, gpu)
        let words = result.contents().assumingMemoryBound(to: UInt32.self)
        return Pick(id: Int(words[0]), value: Float(bitPattern: words[1]), second: Float(bitPattern: words[2]))
    }

    /// The logits of the last head product (`vocabulary` floats): for the tests and the checks, never read in a step.
    package var logitValues: UnsafePointer<Float> { UnsafePointer(logits.contents().assumingMemoryBound(to: Float.self)) }
}

// MARK: - From the maps

extension Qwen3VLDecoder {
    /// **The decoder over the tensors of an encoder map and a head map, where they lie**: the pages the map is mapped on (or
    /// `Artifact.keepResident`'s, if the caller kept them), and, from `tail.first`, the staging buffers the streamed layers are
    /// read into. Throws `Failure.unsupported` (naming the first thing) when this Mac or these maps are not what the kernels read:
    /// no GPU, a tensor not page-aligned, not int8 with per-column scales (and a convrot or none) or bf16, a shape other than the
    /// config's `[k, n]`. Nothing is read from the disk here but the norm weights, which are widened.
    ///
    /// The arithmetic does not depend on where the bytes come from: the same kernels read the same values.
    package static func make(encoder: Artifact, head: Artifact, config: TextEncoder.Config, vocabulary: Int, tail: Tail? = nil,
                             device: MTLDevice? = MTLCreateSystemDefaultDevice(), tuning: Tuning = Tuning()) throws -> Qwen3VLDecoder {
        guard let device else { throw Failure.unsupported("no Metal device") }
        let shape = Shape(config, vocabulary: vocabulary)
        let mapped = (encoder.residentCapacity == 0 || head.residentCapacity == 0)
        var tuning = tuning
        // The map's pages are the cache's: the GPU is not asked to keep them.
        if mapped { tuning.pinsWeights = false }
        let feeder = try tail.map { try TailFeeder(tail: $0, layers: shape.layers, device: device) }
        func wrap(_ artifact: Artifact, _ name: String) throws -> (MTLBuffer, Artifact.Tensor) {
            guard let tensor = artifact.tensors[name], let pointer = artifact.pointer(name) else {
                throw Failure.unsupported("\(name) is not in the map")
            }
            guard Int(bitPattern: pointer) % Arena.alignment == 0 else { throw Failure.unsupported("\(name) is not page-aligned") }
            let length = (tensor.bytes + Arena.alignment - 1) / Arena.alignment * Arena.alignment
            // A mapped tensor's last page is the file's: a buffer longer than the mapping would read past it.
            guard artifact.residentCapacity > 0 || tensor.offset + length <= (artifact.size + Arena.alignment - 1) / Arena.alignment * Arena.alignment else {
                throw Failure.unsupported("\(name): \(length) bytes from \(tensor.offset) end beyond the map")
            }
            guard let buffer = device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: pointer), length: length,
                                                 options: .storageModeShared, deallocator: nil) else {
                throw Failure.unsupported("\(name): the GPU refuses its \(tensor.bytes) bytes")
            }
            buffer.label = name
            MetalWraps.track(buffer)
            return (buffer, tensor)
        }
        /// `at`: the tensor's place in a streamed layer's staging buffer (`buffer` is then the slot's), 0 where it has its own.
        func matrix(_ name: String, k: Int, n: Int, layer: Int) throws -> DecodeKernels.Matrix {
            var buffer: MTLBuffer, t: Artifact.Tensor, at = 0
            if let tail, layer >= tail.first, let feeder {
                guard let tensor = encoder.tensors[name] else { throw Failure.unsupported("\(name) is not in the map") }
                let run = tail.runs[layer - tail.first]
                guard tensor.offset >= run.start, tensor.offset + tensor.bytes <= run.end else {
                    throw Failure.unsupported("\(name) lies outside layer \(layer)'s run")
                }
                buffer = feeder.buffers[(layer - tail.first) % feeder.slotCount]; t = tensor; at = tensor.offset - run.start
            } else {
                (buffer, t) = try wrap(encoder, name)
            }
            guard t.shape == [k, n] else { throw Failure.unsupported("\(name) is \(t.shape), [\(k), \(n)] expected") }
            switch t.dtype {
            case .bfloat16:
                guard t.rotation == nil else { throw Failure.unsupported("\(name): a rotated bf16 weight") }
                return DecodeKernels.Matrix(buffer: buffer, format: .bfloat16, k: k, n: n, offset: at)
            case .int8:
                guard let scale = t.scale, scale.block == nil, scale.shape == [n] else {
                    throw Failure.unsupported("\(name): scales other than one per column")
                }
                let type: DecodeKernels.Matrix.ScaleType
                switch scale.dtype {
                case .float32: type = .float32
                case .bfloat16: type = .bfloat16
                case .float16: type = .float16
                default: throw Failure.unsupported("\(name): scales in \(scale.dtype.rawValue)")
                }
                return DecodeKernels.Matrix(buffer: buffer, scales: buffer, scalesOffset: at + scale.offset, scaleType: type,
                                            format: .int8, k: k, n: n, group: t.rotation?.group ?? 0, offset: at)
            default: throw Failure.unsupported("\(name) is \(t.dtype.rawValue)")
            }
        }
        var layers: [Layer] = []
        let (h, a, kv, i) = (shape.hidden, shape.attentionWidth, shape.kvWidth, shape.intermediate)
        for l in 0..<shape.layers {
            let p = "layers.\(l)."
            layers.append(Layer(q: try matrix(p + "self_attn.q_proj.weight", k: h, n: a, layer: l), k: try matrix(p + "self_attn.k_proj.weight", k: h, n: kv, layer: l),
                                v: try matrix(p + "self_attn.v_proj.weight", k: h, n: kv, layer: l), o: try matrix(p + "self_attn.o_proj.weight", k: a, n: h, layer: l),
                                gate: try matrix(p + "mlp.gate_proj.weight", k: h, n: i, layer: l), up: try matrix(p + "mlp.up_proj.weight", k: h, n: i, layer: l),
                                down: try matrix(p + "mlp.down_proj.weight", k: i, n: h, layer: l)))
        }
        let (headBuffer, headTensor) = try wrap(head, "lm_head.weight")
        guard headTensor.dtype == .bfloat16, headTensor.shape == [vocabulary, h], h % 512 == 0 else {
            throw Failure.unsupported("lm_head is \(headTensor.dtype.rawValue) \(headTensor.shape)")
        }
        return try Qwen3VLDecoder(device: device, shape: shape, layers: layers, head: headBuffer, tuning: tuning, feeder: feeder) { table in
            func widen(_ artifact: Artifact, _ name: String, _ offset: Int, _ count: Int) throws {
                let got = try artifact.materialize(name, into: table + offset, capacity: count)
                guard got == count else { throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected") }
            }
            for l in 0..<shape.layers {
                let p = "layers.\(l)."
                try widen(encoder, p + "input_layernorm.weight", normOffset(shape, layer: l, part: 0), h)
                try widen(encoder, p + "post_attention_layernorm.weight", normOffset(shape, layer: l, part: 1), h)
                try widen(encoder, p + "self_attn.q_norm.weight", normOffset(shape, layer: l, part: 2), shape.headDim)
                try widen(encoder, p + "self_attn.k_norm.weight", normOffset(shape, layer: l, part: 3), shape.headDim)
            }
            try widen(head, "norm.weight", finalNormOffset(shape), h)
        }
    }

    /// **A layer's run in the file, from the header** — what the streamed layers are read as (`Artifact.stagingRun`).
    package static func runs(of encoder: Artifact, layers: Range<Int>) throws -> [(start: Int, end: Int)] {
        try layers.map { l in
            let prefix = "layers.\(l)."
            let tensors = encoder.tensors.filter { $0.key.hasPrefix(prefix) }.map { (offset: $0.value.offset, bytes: $0.value.bytes) }
            guard let run = Artifact.stagingRun(tensors, page: encoder.page, fileSize: encoder.size) else {
                throw Failure.unsupported("layer \(l) has no tensor")
            }
            return run
        }
    }
}

/// **The streamed layers' reads**: `slotCount` anonymous staging buffers, each one layer's run, filled in the order the layers
/// are computed by `pread` + `F_NOCACHE` (the stream's own lanes: `CapacityDisk.read`), so that the file's pages never enter the
/// cache and the head of the map stays in it. A read of layer `l` waits for the GPU to be done with layer `l − slotCount`, which
/// used the same buffer (`Pass.release`); the CPU waits for the read before it encodes the layer (`Pass.wait`).
package final class TailFeeder: @unchecked Sendable {
    package let first: Int, slotCount: Int, layerCount: Int
    package let buffers: [MTLBuffer]
    private let pages: [Pages]
    private let runs: [(start: Int, end: Int)]
    private let descriptor: Int32
    private let issue = DispatchQueue(label: "siliconed.writer.tail")

    init(tail: Qwen3VLDecoder.Tail, layers: Int, device: MTLDevice) throws {
        guard tail.first < layers, tail.runs.count == layers - tail.first, tail.slots > 0 else {
            throw Qwen3VLDecoder.Failure.unsupported("a tail from layer \(tail.first) of \(layers)")
        }
        let fd = open(tail.path, O_RDONLY)
        guard fd >= 0 else { throw Qwen3VLDecoder.Failure.unsupported("\(tail.path) cannot be opened (\(errno))") }
        _ = fcntl(fd, F_NOCACHE, 1)
        descriptor = fd; first = tail.first; slotCount = tail.slots; layerCount = layers; runs = tail.runs
        let slots = (0..<tail.slots).map { _ in Pages(bytes: tail.slotBytes) }
        pages = slots
        var made: [MTLBuffer] = []
        for p in slots {
            guard let b = device.makeBuffer(bytesNoCopy: p.base, length: p.capacity, options: .storageModeShared, deallocator: nil) else {
                close(fd)
                throw Qwen3VLDecoder.Failure.unsupported("the GPU refuses a staging buffer of \(p.capacity) bytes")
            }
            b.label = "tail staging"
            MetalWraps.track(b)
            made.append(b)
        }
        buffers = made
    }

    deinit { close(descriptor) }   // the queued reads hold `self`: none is left to run

    /// One step's reads: every streamed layer, in order, as the buffers are given back.
    func begin() -> Pass {
        let pass = Pass(feeder: self)
        for index in 0..<runs.count {
            issue.async {
                pass.free[index % self.slotCount].wait()
                if !pass.isAborted {
                    let run = self.runs[index]
                    if !CapacityDisk.read(self.descriptor, from: run.start, to: run.end, into: self.pages[index % self.slotCount].base) {
                        pass.fail()
                    }
                }
                pass.ready[index].signal()
            }
        }
        return pass
    }

    final class Pass: @unchecked Sendable {
        fileprivate let free: [DispatchSemaphore], ready: [DispatchSemaphore]
        private let feeder: TailFeeder
        private let lock = NSLock()
        private var aborted = false, failed = false

        fileprivate init(feeder: TailFeeder) {
            self.feeder = feeder
            free = (0..<feeder.slotCount).map { _ in DispatchSemaphore(value: 1) }
            ready = (0..<(feeder.layerCount - feeder.first)).map { _ in DispatchSemaphore(value: 0) }
        }
        fileprivate var isAborted: Bool { lock.withLock { aborted } }
        fileprivate func fail() { lock.withLock { failed = true } }

        /// Blocks until layer `l` is in its buffer.
        func wait(layer l: Int) throws {
            ready[l - feeder.first].wait()
            if lock.withLock({ failed || aborted }) { throw Qwen3VLDecoder.Failure.gpu("the tail's read of layer \(l) failed") }
        }
        /// The GPU is done with the slot's buffer.
        func release(slot: Int) { free[slot].signal() }
        /// No more reads: the ones still waiting are let through without reading. The extra signals also keep a semaphore's value
        /// at least its initial one, which libdispatch requires of one it frees.
        func abort() {
            lock.withLock { aborted = true }
            for s in free { for _ in 0..<ready.count { s.signal() } }
        }
    }
}
