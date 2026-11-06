import Darwin
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// **A VAE's memory: every tensor in a heap of ours, placed before the first graph runs**.
///
///     registers   one placement heap (shared): every tensor that lives between two GPU passes — the
///                 input, each operation's output, the band windows, the weights — at an offset the plan
///                 computed in advance from the lifetimes of all of them (`VAELayout.place`)
///     workspace   one heap (private) that MPS receives for every command buffer (`MPSHeapProvider`):
///                 what a graph would ask for inside itself — nothing, for the one-operation graphs
///                 the VAEs run since the cut into passes
///
/// Each VAE has a static plan function that walks its own code in planning mode (no device, no
/// weight, no graph: the same calls to `register`, `free` and `graph`) and returns the `VAELayout` its
/// constructor then allocates: one source, the equality by construction. The run then asks for the
/// same slots in the same order, and each gets the offset the plan gave it — a run that leaves its
/// plan (another size, another lifetime) stops rather than overlap two live tensors.
///
/// What it replaces, and why — what a port that "just frees what it allocates" would get wrong:
///
///   - **Metal returns GPU memory late.** A buffer or a heap the GPU has touched leaves the process's
///     footprint 0.05 s to more than 2 s after its last release, and nothing public hurries it
///     (purgeable `.empty`, unretained references, a completion handler, an empty command buffer:
///     all tried). A VAE that allocated each register and let MPS allocate each graph's
///     temporaries left behind it everything it freed in its last second, and every stage carried the
///     stages before it, still pending. So nothing is freed inside a VAE: the two heaps are made once
///     and carved;
///   - **our own pages wrapped `bytesNoCopy` do not escape it**: `munmap` takes them out of
///     `phys_footprint` at once, but they stay **wired** until Metal drops its mapping: refuted;
///   - **a graph of several operations allocates its intermediates itself.** MPSGraph takes from the
///     heap provider only one block, the size of its largest tensor, and puts every other intermediate
///     in Metal memory of its own, which no API directs elsewhere: 1.76 GB beyond our heaps in the
///     Qwen-Image-2.1 decoder's tail at 1024², 1.6 GB in its encoder's first block. A larger
///     provider heap changes nothing (3 GB offered: the same 1,758 MB beside it). Hence the VAEs
///     run graphs of **one kernel** each — a convolution and its bias (and a residual sum), a norm and
///     its SiLU — whose only tensors are their inputs and their output, all registers here; and the
///     pure copies MPSGraph materializes twice over (the ×2 `broadcast`, `DupUp3D`'s gather, `AvgDown3D`'s
///     permutes) are kernels of ours (`VAEKernels`). Cut where MPSGraph fused nothing across the cut,
///     the same bits;
///   - **the placement is computed, not found.** A first fit at run time fragments: the heap reserved
///     whole held ~60 MB of holes at the Flux decoder's peak. The plan knows every slot's size,
///     birth and death before the first byte: it places the largest first, each at the lowest offset no
///     slot alive at the same time occupies, and the heap is the highest end — **equal to the largest sum
///     of slots alive at once** (`VAELayout.peak`, tested on every VAE at three formats).
package struct VAELayout: Sendable, Equatable {
    /// Every slot starts on a page: a placed buffer's offset must meet the heap's alignment
    /// (`heapBufferSizeAndAlign`, checked at each placement), and a page meets it on Apple GPUs.
    package static let alignment = 1 << 14

    /// A slot of the walk: its bytes (whole pages) and its lifetime, in events (an allocation or a
    /// free is one event): alive from `birth` to `death`, `Int.max` when never freed.
    package struct Slot: Sendable, Equatable {
        package var bytes: Int
        package var birth: Int
        package var death = Int.max
    }

    package private(set) var slots: [Slot] = []
    /// Each slot's offset in the placement heap, once `place` ran.
    package private(set) var offsets: [Int] = []
    private var event = 0
    /// The placement heap's bytes: the highest end of a placed slot.
    package private(set) var registers = 0
    /// The workspace heap's bytes: what MPS asks for the largest graph the VAE runs.
    package private(set) var workspace = 0
    /// The host (`[Float]`) copies the VAE makes beside its heaps: its result, a de-normalized latent.
    package private(set) var host = 0
    /// **What stays MPSGraph's, beside both heaps, at most**: its SDPA's own copies of the attention's
    /// operands, `4·S·C` floats for one pass (`sdpaOutside`) — the one graph whose cut changes the
    /// bits. Not in `bytes`: a term of its own, computed from the shapes like the rest.
    package private(set) var outside = 0

    package init() {}

    /// The largest single slot, in bytes (whole pages): the widest register of the walk.
    package var largestSlot: Int { slots.map(\.bytes).max() ?? 0 }

    /// Everything: both heaps and the host copies.
    package var bytes: Int { registers + workspace + host }

    package static func rounded(_ bytes: Int) -> Int { (max(bytes, 1) + alignment - 1) / alignment * alignment }

    /// A new slot of `bytes`: its index, the order of the walk.
    package mutating func allocate(_ bytes: Int) -> Int {
        slots.append(Slot(bytes: VAELayout.rounded(bytes), birth: event))
        event += 1
        return slots.count - 1
    }

    /// The event of this free: the walk's clock, for the run to check it frees as its plan did.
    @discardableResult
    package mutating func free(_ slot: Int) -> Int {
        precondition(slots[slot].death == Int.max, "VAELayout: freeing a slot twice")
        slots[slot].death = event
        event += 1
        return slots[slot].death
    }

    /// **What MPS asks of the workspace for a one-kernel graph** (read on every pass of the Qwen-Image-2.1
    /// VAE): 2 MiB for a reduction or a convolution — and, for a stride-2 or 1×1 convolution, its
    /// weights reformatted, rounded up to 2 MiB (`reformattedWeights`); MPSGraph's SDPA, a copy of its
    /// queries. Declarations, not a law: an ask beyond them is granted beside the heap (`Workspace`), a defect.
    package static let minimumAsk = 2 << 20

    /// A convolution's `floats` of weights, as MPS reformats them in the workspace.
    package static func reformattedWeights(_ floats: Int) -> Int {
        (4 * floats + minimumAsk - 1) / minimumAsk * minimumAsk
    }

    /// A graph whose workspace request is `elements` floats (0: none).
    package mutating func graph(largest elements: Int) {
        if elements > 0 { workspace = max(workspace, VAELayout.rounded(4 * elements)) }
    }

    package mutating func host(_ bytes: Int) { host += bytes }

    package mutating func outside(_ bytes: Int) { outside = max(outside, bytes) }

    /// **What MPSGraph's `scaledDotProductAttention` allocates itself** for `S` tokens of `C` channels, one
    /// head: four `S·C` copies (q, k, v and its output) beside the one it asks of the workspace — 72 MiB
    /// for Qwen-Image-2.1's decoder at 1024² (`S` 4096, `C` 1152), 48 MiB for its encoder (`C` 768),
    /// 128 MiB for Flux's (`S` 16,384, `C` 512): read on all three. Decomposing it into our
    /// registers (`QKᵀ`, softmax, `PV`) moves the decoder's output by 1.35·10⁻⁵ (worst channel): refused
    /// for now, the bits first.
    package static func sdpaOutside(tokens s: Int, channels c: Int) -> Int { 4 * 4 * s * c }

    /// **What MPSGraph's SDPA asks of the workspace**, in floats: the copy of its queries, `S·C` floats,
    /// asked like every MPS request in whole `minimumAsk`s — 4.72 MB of queries at Qwen-Image-2.1's
    /// decoder at 512² (`S` 1,024, `C` 1,152) ask 6 MiB. Declared at `S·C` unrounded, the heap held 4.5
    /// MiB and MPS was granted its 6 MiB beside it, outside the plan.
    package static func sdpaAsk(tokens s: Int, channels c: Int) -> Int {
        (4 * s * c + minimumAsk - 1) / minimumAsk * minimumAsk / 4
    }

    /// Nothing live: every slot freed (the run's end).
    package var empty: Bool { slots.allSatisfy { $0.death != Int.max } }

    /// **The occupancy at its largest**: the most bytes of slots alive at the same event — what no
    /// placement can go under, and what `place` reaches.
    package var peak: Int {
        var changes: [(event: Int, bytes: Int)] = []
        for s in slots {
            changes.append((s.birth, s.bytes))
            if s.death != Int.max { changes.append((s.death, -s.bytes)) }
        }
        var live = 0, peak = 0
        for c in changes.sorted(by: { $0.event < $1.event }) { live += c.bytes; peak = max(peak, live) }
        return peak
    }

    /// **The offsets, from the lifetimes** — an arena laid out in advance. Largest first (ties: the
    /// earliest born), each at the lowest offset where it overlaps no placed slot that is alive at the
    /// same time. When that order leaves a hole the peak does not need, the earliest-born order is tried
    /// too, and the lower heap kept.
    package mutating func place() {
        // Flat buffers and plain loops: Qwen's decoder at its smallest band has ~2,000 slots, each
        // scanning every slot placed before it (160 ms in a debug build with closures, the first form).
        // The order of equal starts in `busy` does not change the offset found.
        let bytes = slots.map(\.bytes), births = slots.map(\.birth), deaths = slots.map(\.death)
        func layout(_ order: [Int]) -> (offsets: [Int], end: Int) {
            let n = bytes.count
            var offsets = [Int](repeating: 0, count: n)
            var end = 0
            bytes.withUnsafeBufferPointer { bytes in births.withUnsafeBufferPointer { births in
            deaths.withUnsafeBufferPointer { deaths in offsets.withUnsafeMutableBufferPointer { offsets in
                let placed = UnsafeMutableBufferPointer<Int>.allocate(capacity: n)
                // The busy intervals, `start` and `stop` interleaved, sorted by `start` as they come.
                let busy = UnsafeMutableBufferPointer<Int>.allocate(capacity: 2 * n)
                defer { placed.deallocate(); busy.deallocate() }
                var count = 0
                for i in order {
                    let (size, birth, death) = (bytes[i], births[i], deaths[i])
                    var b = 0
                    var k = 0
                    while k < count {
                        let j = placed[k]
                        if births[j] < death && birth < deaths[j] {
                            let (start, stop) = (offsets[j], offsets[j] + bytes[j])
                            var at = b
                            while at > 0 && busy[2 * (at - 1)] > start {
                                busy[2 * at] = busy[2 * (at - 1)]; busy[2 * at + 1] = busy[2 * (at - 1) + 1]
                                at -= 1
                            }
                            busy[2 * at] = start; busy[2 * at + 1] = stop
                            b += 1
                        }
                        k += 1
                    }
                    var offset = 0
                    var m = 0
                    while m < b {
                        if offset + size <= busy[2 * m] { break }
                        offset = max(offset, busy[2 * m + 1])
                        m += 1
                    }
                    offsets[i] = offset
                    placed[count] = i
                    count += 1
                    end = max(end, offset + size)
                }
            } } } }
            return (offsets, end)
        }
        let indices = Array(slots.indices)
        var best = layout(indices.sorted { (slots[$0].bytes, -slots[$0].birth) > (slots[$1].bytes, -slots[$1].birth) })
        let target = peak
        if best.end > target {
            let byBirth = layout(indices)
            if byBirth.end < best.end { best = byBirth }
        }
        if best.end > target {
            let byLength = layout(indices.sorted {
                let a = slots[$0], b = slots[$1]
                return (min(a.death, event) - a.birth, a.bytes) > (min(b.death, event) - b.birth, b.bytes)
            })
            if byLength.end < best.end { best = byLength }
        }
        offsets = best.offsets
        registers = best.end
    }
}

/// A tensor between two GPU passes: a slot of the placement heap (`nil` buffer while planning), in fp32 — or in bf16, the
/// activations and weights of a decoder computing in half precision (`VAEMemory.half`).
final class VAERegister {
    let shape: [Int]
    /// The slot's index in the walk (`VAELayout.slots`); a view shares its slot's.
    let slot: Int
    /// The slot's bytes: a view (`VAEMemory.view`) keeps its slot's.
    let bytes: Int
    let buffer: MTLBuffer?
    let data: MPSGraphTensorData?
    /// `.float32`, or `.bFloat16`.
    let dataType: MPSDataType

    init(shape: [Int], slot: Int, bytes: Int, buffer: MTLBuffer?, dataType: MPSDataType = .float32) {
        self.shape = shape; self.slot = slot; self.bytes = bytes; self.buffer = buffer; self.dataType = dataType
        data = buffer.map { MPSGraphTensorData($0, shape: shape.map { NSNumber(value: $0) }, dataType: dataType) }
    }

    var count: Int { shape.reduce(1, *) }
    var nsShape: [NSNumber] { shape.map { NSNumber(value: $0) } }
    var isHalf: Bool { dataType == .bFloat16 }
    /// The bytes of one element: 4, or 2 in bf16.
    var elementSize: Int { VAERegister.size(dataType) }
    static func size(_ type: MPSDataType) -> Int { type == .bFloat16 ? 2 : 4 }
    /// The floats of an fp32 register.
    var values: UnsafeMutablePointer<Float> {
        precondition(!isHalf, "VAERegister: a bf16 register read as floats")
        return buffer!.contents().assumingMemoryBound(to: Float.self)
    }
    var raw: UnsafeMutableRawPointer { buffer!.contents() }

    /// `count` floats written from `source` — rounded to the nearest bf16 (ties to even) in a bf16 register.
    func store(_ source: UnsafePointer<Float>, count: Int) {
        if !isHalf { values.update(from: source, count: count); return }
        Widen.float32ToBfloat16(source: source, destination: raw, count: count)
    }

    /// Its values as floats (a bf16 register widened, exactly).
    func floats() -> [Float] {
        if !isHalf { return Array(UnsafeBufferPointer(start: values, count: count)) }
        var out = [Float](repeating: 0, count: count)
        out.withUnsafeMutableBytes { Widen.bfloat16ToFloat32(source: UnsafeRawPointer(raw), destination: $0.baseAddress!, count: count) }
        return out
    }
}

/// The two heaps, the walk over them, and the command buffers that run the VAE's graphs and kernels.
/// `VAEMemory()` plans; `VAEMemory(plan:…)` runs.
package final class VAEMemory {
    /// The walk so far (planning: the plan itself, placed when read through `layout`).
    private var walk = VAELayout()
    let device: MTLDevice?
    /// Our own queue, released with the heaps: MPS's command buffers keep their heap provider until
    /// their queue lets them go (the VAE's own queue held the workspace 3 s past `finish`).
    private var queue: MTLCommandQueue?
    private var heap: MTLHeap?
    private var workspace: Workspace?
    private var kernels: VAEKernels?
    private let plan: VAELayout?
    /// The graphics ledger before the heaps were made: what `finish` waits for Metal to come back to.
    private var baseline = 0
    /// **The decoder computes in bf16** (`VAEPrecision`): its activations and weights in bf16 registers, the statistics of its
    /// norms and the latent it reads in fp32. Planning and running alike: a plan in bf16 is the walk of a run in bf16.
    let half: Bool
    /// The activations' type: bf16 when `half`, fp32 otherwise.
    var activation: MPSDataType { half ? .bFloat16 : .float32 }

    var isLive: Bool { heap != nil }

    /// The plan: the walk, placed.
    var layout: VAELayout { var l = walk; l.place(); return l }

    /// A run that does not follow its plan: a bug of the plan, said rather than run past the heap.
    enum Failure: Error, CustomStringConvertible {
        case plan(String)
        var description: String { if case .plan(let what) = self { return "VAE memory: \(what)" }; return "" }
    }

    /// Planning: the same calls, no device. The layout it ends with is the plan.
    init(half: Bool = false) { device = nil; queue = nil; plan = nil; self.half = half }

    /// Running: the two heaps, at the plan's sizes.
    init(plan: VAELayout, device: MTLDevice, queue: MTLCommandQueue, half: Bool = false) throws {
        precondition(plan.offsets.count == plan.slots.count, "VAEMemory: a plan that was not placed")
        self.plan = plan; self.device = device; self.half = half
        baseline = VAEMemory.graphicsFootprint()
        guard let own = device.makeCommandQueue() else { throw GEMM.Failure.noDevice }
        self.queue = own
        kernels = try VAEKernels.shared(device)
        let registers = MTLHeapDescriptor()
        registers.type = .placement
        registers.storageMode = .shared
        registers.hazardTrackingMode = .untracked
        registers.size = max(plan.registers, VAELayout.alignment)
        // MPS's own request (as measured): private, untracked, automatic — returned as asked.
        let scratch = MTLHeapDescriptor()
        scratch.type = .automatic
        scratch.storageMode = .private
        scratch.hazardTrackingMode = .untracked
        scratch.size = max(plan.workspace, VAELayout.alignment)
        guard let heap = device.makeHeap(descriptor: registers), let work = device.makeHeap(descriptor: scratch) else {
            throw GEMM.Failure.noDevice
        }
        heap.label = "VAE registers"; work.label = "VAE workspace"
        self.heap = heap
        workspace = Workspace(heap: work, device: device)
    }

    // ── registers ──────────────────────────────────────────────────────────────────────────

    /// A register of the activations' type (`activation`), or of `type`.
    func register(_ shape: [Int], _ type: MPSDataType? = nil) throws -> VAERegister {
        let type = type ?? activation
        let bytes = VAERegister.size(type) * shape.reduce(1, *)
        let slot = walk.allocate(bytes)
        let length = VAELayout.rounded(bytes)
        guard let heap, let device, let plan else {
            return VAERegister(shape: shape, slot: slot, bytes: length, buffer: nil, dataType: type)
        }
        guard slot < plan.slots.count, plan.slots[slot].bytes == length, plan.slots[slot].birth == walk.slots[slot].birth else {
            throw Failure.plan("slot \(slot) of \(length) bytes is not the one the plan placed")
        }
        let options: MTLResourceOptions = [.storageModeShared, .hazardTrackingModeUntracked]
        let need = device.heapBufferSizeAndAlign(length: length, options: options)
        precondition(VAELayout.alignment % need.align == 0 && need.size <= length,
                     "VAEMemory: this GPU places buffers at \(need.align) bytes, not \(VAELayout.alignment)")
        guard let buffer = heap.makeBuffer(length: length, options: options, offset: plan.offsets[slot]) else {
            throw GEMM.Failure.noDevice
        }
        return VAERegister(shape: shape, slot: slot, bytes: length, buffer: buffer, dataType: type)
    }

    /// A register holding `values` (copied when running; rounded to bf16 in a bf16 register).
    func register(_ shape: [Int], from values: UnsafePointer<Float>) throws -> VAERegister {
        let r = try register(shape)
        if isLive { r.store(values, count: r.count) }
        return r
    }

    /// The same slot under another shape, no larger than the slot: a layer that writes its output
    /// over its input. Freed with the slot, once.
    func view(_ r: VAERegister, as shape: [Int]) throws -> VAERegister {
        let length = VAELayout.rounded(r.elementSize * shape.reduce(1, *))
        precondition(length <= r.bytes, "VAEMemory: a view larger than its slot")
        guard let heap, let plan else {
            return VAERegister(shape: shape, slot: r.slot, bytes: r.bytes, buffer: nil, dataType: r.dataType)
        }
        let options: MTLResourceOptions = [.storageModeShared, .hazardTrackingModeUntracked]
        guard let buffer = heap.makeBuffer(length: length, options: options, offset: plan.offsets[r.slot]) else {
            throw GEMM.Failure.noDevice
        }
        return VAERegister(shape: shape, slot: r.slot, bytes: r.bytes, buffer: buffer, dataType: r.dataType)
    }

    func free(_ r: VAERegister) {
        let event = walk.free(r.slot)
        if let plan {
            precondition(plan.slots[r.slot].death == event, "VAEMemory: slot \(r.slot) freed out of its plan")
        }
    }

    func free(_ registers: [VAERegister]) { for r in registers.reversed() { free(r) } }

    /// **The weights of a graph, in registers of ours**: one per weight, `declared` their floats in the
    /// order the graph reads them (`WeightList`), widened from `file` (running) — or, in bf16 (`half`), copied as
    /// published, bit for bit — and fed to the graph's placeholders. `fed`: what the graph built recorded (`nil` while planning, or when nothing was
    /// built) — checked against `declared`, element by element.
    func weights(_ declared: [Int], _ fed: FedWeights?, file: Safetensors?, vae: String) throws
            -> (registers: [VAERegister], feeds: [MPSGraphTensor: MPSGraphTensorData]) {
        if let fed {
            guard fed.items.map({ $0.shape.reduce(1, *) }) == declared else {
                throw Failure.plan("\(vae) reads \(fed.items.map(\.name)) — not the weights its plan charges")
            }
        }
        var registers: [VAERegister] = [], feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
        for (k, floats) in declared.enumerated() {
            let item = fed?.items[k]
            let r = try register(item?.shape ?? [floats])
            registers.append(r)
            if let item, let file, isLive {
                if half { try VAEBuildingBlocks.copyHalf(file, item.name, rows: item.rows, vae: vae, into: r.raw) }
                else { try VAEBuildingBlocks.widen(file, item.name, rows: item.rows, vae: vae, into: r.values) }
                feeds[item.placeholder] = r.data!
            }
        }
        return (registers, feeds)
    }

    /// A host copy beside the heaps (a `[Float]` result): charged to the plan, not placed.
    func host(_ bytes: Int) { walk.host(bytes) }

    /// What a pass leaves MPSGraph to allocate beside the heaps (`VAELayout.outside`).
    func outside(_ bytes: Int) { walk.outside(bytes) }

    // ── graphs and kernels ─────────────────────────────────────────────────────────────────

    /// **A graph whose workspace request is `largest` floats** (0: a one-kernel graph, which asks
    /// none) — declared before it is built, from the shapes alone. Planning: recorded, `nil` returned
    /// (nothing is built). Running: `build` runs. A plan that under-declares makes MPS ask beyond the
    /// workspace heap, on any machine.
    func graph<T>(largest: Int, _ build: () throws -> T) rethrows -> T? {
        walk.graph(largest: largest)
        guard isLive else { return nil }
        return try build()
    }

    /// **`graph.encode` on a command buffer whose MPS temporaries come from our workspace heap**,
    /// without MPS's heap cache (it would keep them for 3–15 s), and waited for.
    func execute(_ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData],
                 results: [MPSGraphTensor: MPSGraphTensorData]) {
        guard let queue, let workspace else { return }
        autoreleasepool {
            let commands = MPSCommandBuffer(from: queue)
            MPSSetHeapCacheDuration(commands, 0)
            commands.heapProvider = workspace
            graph.encode(to: commands, feeds: feeds, targetOperations: nil, resultsDictionary: results, executionDescriptor: nil)
            commands.commit()
            commands.waitUntilCompleted()
            // The VAEs' graphs were the one GPU submission of a render whose failure went unseen: a
            // refused command buffer writes nothing, its result register keeps whatever its heap held,
            // and the latent of an edit's reference is then cached for every later render
            // (`RenderCache`). Counted and said like the GEMMs' and the SDPA's (`Attention.verdict`);
            // only the error, never the GPU time — MPS may commit and continue, leaving this one empty.
            if commands.status == .error || commands.error != nil {
                Attention.report("VAE graph: the GPU refused — \(commands.error?.localizedDescription ?? "status error")")
            }
        }
        // What the graph's compilation freed goes back before the next pass (`Arena.returnFreedPages`):
        // ~100 graphs a decoder, each one's mlir churn otherwise billed until the phase ends.
        Arena.returnFreedPages()
    }

    /// One of our kernels over registers (`VAEKernels`), waited for — its bf16 variant when the first register is bf16.
    /// Planning: nothing.
    func dispatch(_ kernel: VAEKernels.Kernel, _ registers: [VAERegister], constants: [UInt32], grid: MTLSize) {
        guard let queue, let kernels else { return }
        autoreleasepool {
            kernels.dispatch(kernel, on: queue, registers.map { $0.buffer! }, constants: constants, grid: grid,
                             half: registers[0].isHalf)
        }
    }

    // ── the norms of a decoder in bf16 (`half`): kernels of ours, fp32 inside ──────────────────

    /// **Weights read by a kernel**, as published (bf16), one register each, in `names`' order; freed by the caller.
    func kernelWeights(_ names: [String], counts: [Int], file: Safetensors?, vae: String) throws -> [VAERegister] {
        try zip(names, counts).map { name, count in
            let r = try register([count], .bFloat16)
            if let file, isLive {
                guard file.entries[name]?.count == count else { throw Safetensors.Failure.missingTensor(file: vae, name: name) }
                try VAEBuildingBlocks.copyHalf(file, name, rows: nil, vae: vae, into: r.raw)
            }
            return r
        }
    }

    /// The threads per group of `vae_group_partials`: as many as the group has elements, up to 4,096.
    static func partialThreads(_ perGroup: Int) -> Int { min(4096, perGroup) }

    /// **A `GroupNorm`'s mean and variance per group, over the whole bf16 `x`** (`[1, C, H, W]`, `groups` contiguous runs),
    /// in two fp32 registers `[1, groups, 1]`: partial means and squared deviations on the GPU, merged here in double (Chan).
    /// The variance is the population's, as `GroupNorm`'s.
    func groupStatistics(_ x: VAERegister, groups: Int) throws -> (mean: VAERegister, variance: VAERegister) {
        let perGroup = x.count / groups, threads = VAEMemory.partialThreads(perGroup)
        let m = try register([1, groups, 1], .float32), v = try register([1, groups, 1], .float32)
        let partials = try register([groups, threads, 2], .float32)
        if isLive {
            dispatch(.groupPartials, [x, partials], constants: [UInt32(perGroup), UInt32(threads)],
                     grid: MTLSize(width: threads, height: groups, depth: 1))
            let p = partials.values
            for g in 0..<groups {
                var n = 0.0, mean = 0.0, m2 = 0.0
                for t in 0..<threads {
                    let count = Double((perGroup - t + threads - 1) / threads)
                    guard count > 0 else { continue }
                    let (mt, m2t) = (Double(p[2 * (g * threads + t)]), Double(p[2 * (g * threads + t) + 1]))
                    let total = n + count, delta = mt - mean
                    mean += delta * count / total
                    m2 += m2t + delta * delta * n * count / total
                    n = total
                }
                m.values[g] = Float(mean); v.values[g] = Float(m2 / n)
            }
        }
        free(partials)
        return (m, v)
    }

    /// `act(GroupNorm(x))` into `y` (same shape, bf16): `stats` from `groupStatistics`, `affine` γ and β (`groupNormWeights`).
    func groupNorm(_ x: VAERegister, into y: VAERegister, stats: (mean: VAERegister, variance: VAERegister), groups: Int,
                   affine: [VAERegister], silu: Bool, epsilon: Float) {
        let c = x.shape[1], plane = x.count / c
        dispatch(.groupNorm, [x, stats.mean, stats.variance, affine[0], affine[1], y],
                 constants: [UInt32(c), UInt32(plane), UInt32(groups), silu ? 1 : 0, epsilon.bitPattern],
                 grid: MTLSize(width: plane, height: c, depth: 1))
    }

    /// γ and β of the `GroupNorm` at `norm`, as published, for `groupNorm`.
    func groupNormWeights(_ norm: String, channels c: Int, file: Safetensors?, vae: String) throws -> [VAERegister] {
        try kernelWeights([norm + ".weight", norm + ".bias"], counts: [c, c], file: file, vae: vae)
    }

    /// `act(RMSNorm(x))` into `y` (same shape, bf16): Qwen-Image's `F.normalize · √C · γ`, `gamma` as published (`kernelWeights`).
    func rmsNorm(_ x: VAERegister, into y: VAERegister, gamma: VAERegister, silu: Bool) {
        let c = x.shape[1], plane = x.count / c
        dispatch(.rmsNorm, [x, gamma, y], constants: [UInt32(c), UInt32(plane), silu ? 1 : 0],
                 grid: MTLSize(width: plane, height: 1, depth: 1))
    }

    // ── the end ────────────────────────────────────────────────────────────────────────────

    /// **Releases both heaps, and returns once Metal has taken them back.** Every register must be
    /// dead: a placed buffer keeps its heap alive. Metal takes the heaps back on its own time — under
    /// 0.1 s since the VAEs run in single-kernel passes —, and the memory need says a VAE hands
    /// everything back before it returns (`MemoryNeed`): without the wait, the next phase started beside
    /// what was not yet returned (+35 MB between two Qwen-Image-2.1 references), uncharged. So the
    /// graphics ledger is polled until **our two heaps** have left it — down by their size from its level
    /// just before they are released, or back to `baseline` if that comes first, within `releaseTolerance` —
    /// for `releaseWait` at most; past it, what is left is the check's to report.
    ///
    /// **The heaps, not the baseline**: after a render's denoising, ~50 MB of Metal's own that the first
    /// graphs allocated beside the heaps stays in the ledger ~1.2 s after they are gone (Z-Image 1024²:
    /// 1173 → 61 MB in 27 ms, then 61 → 11–19 MB at +1.2 s; never in the decoder alone, where it leaves
    /// with them), and the ledger lands a few MB either side of a baseline of 14 MB. Polled back to the
    /// baseline, every render's decoding waited 1.2 s, or the full `releaseWait` when it landed above
    /// `baseline + releaseTolerance` (2.33 s of decoding for 1.1 s of passes). Never later than the
    /// baseline: what Metal keeps of its own it returns on its own time, inside the phase's charge.
    func finish() {
        // Their sizes only: a reference held here would keep them alive through the wait.
        guard let heapBytes = heap?.size, let workspaceBytes = workspace?.heap.size else { return }
        precondition(walk.empty, "VAEMemory: registers still live at the end")
        let returned = VAEMemory.graphicsFootprint() - heapBytes - workspaceBytes
        heap = nil; workspace = nil; queue = nil
        let target = max(baseline, returned) + VAEMemory.releaseTolerance
        let deadline = Date().addingTimeInterval(VAEMemory.releaseWait)
        while VAEMemory.graphicsFootprint() > target, Date() < deadline { usleep(1000) }
    }

    /// How long `finish` waits for Metal at most, and what it leaves above the baseline: measured
    /// up to ~2 s before the passes, under 0.1 s after; 4 MiB is the ledger's noise at rest.
    static let releaseWait: TimeInterval = 2
    static let releaseTolerance = 4 << 20

    /// The process's graphics ledger: Metal's buffers and heaps, ours and MPS's.
    static func graphicsFootprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.ledger_tag_graphics_footprint) : 0
    }

    /// **MPS's temporaries, in our heap.** MPS asks a heap per command buffer (`MPSCommandBuffer`'s
    /// `heapProvider`) and retires it after; we hand it the same heap every time. A request our heap
    /// cannot hold — another system planning otherwise than the plan declares — is granted as asked, never
    /// refused: the render's phase peak, checked against its need, is where it shows.
    private final class Workspace: NSObject, MPSHeapProvider {
        let heap: MTLHeap
        let device: MTLDevice

        init(heap: MTLHeap, device: MTLDevice) { self.heap = heap; self.device = device }

        func newHeap(with descriptor: MTLHeapDescriptor) -> MTLHeap? {
            if descriptor.storageMode == heap.storageMode, descriptor.type == heap.type,
               descriptor.size <= heap.maxAvailableSize(alignment: VAELayout.alignment) {
                return heap
            }
            return device.makeHeap(descriptor: descriptor)
        }

        func retire(_ heap: MTLHeap, cacheDelay seconds: Double) {}
    }
}

/// **The VAEs' pure copies, as kernels of ours** — what MPSGraph materialized beside our heaps,
/// up to twice over (a `broadcast` then its reshape: 5 tensors' worth beyond its output for the ×2 alone):
///
///   - `double`: `nearest-exact` ×2, each pixel its 2×2 block — `QwenImage21VAE.double`, and Flux's in `VAE.Bands`;
///   - `dup_up_add`: `main + DupUp3D(skip)` in place over `main` — the gather of `QwenImage21VAE.dupUpSources`
///     and one fp32 sum, as the graph it replaced laid it;
///   - `avg_down_add`: `main + AvgDown3D(x)` in place over `main` — the group mean of
///     `QwenImage21VAE.avgDownGroups` (members of the zero frame add 0), summed in member order then
///     divided by the group size, as the replaced graph's `mean` (a power of two: the division is exact).
///
/// Copies and single sums round once, as the graph did: the same bits by construction; the mean's
/// order is the graph's (the encoder's md5). No fast math.
final class VAEKernels {
    enum Kernel: String, CaseIterable {
        case double = "vae_double", dupUpAdd = "vae_dup_up_add", avgDownAdd = "vae_avg_down_add"
        case tokens = "vae_tokens", untokensAdd = "vae_untokens_add"
        /// bf16 only (`VAEMemory.half`): a `GroupNorm`'s partial statistics, the norm with its affine (and SiLU), the RMSNorm.
        case groupPartials = "vae_group_partials", groupNorm = "vae_group_norm", rmsNorm = "vae_rms_norm"

        /// The kernels with a bf16 variant (`<name>_bf16`), taken when the operands are bf16.
        var halfName: String { rawValue + "_bf16" }
        static let halfOnly: Set<Kernel> = [.groupPartials, .groupNorm, .rmsNorm]
        static let fp32Only: Set<Kernel> = [.avgDownAdd]
    }

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // y [C, 2h, 2w] ← x [C, h, w], each pixel its 2×2 block. d = (h, w).
    kernel void vae_double(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                           constant uint4 &d [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
        uint h = d.x, w = d.y, H = 2 * h, W = 2 * w;
        if (p.x >= W || p.y >= H) return;
        y[(p.z * H + p.y) * W + p.x] = x[(p.z * h + p.y / 2) * w + p.x / 2];
    }

    // m [O, 2h, 2w] += DupUp3D(s [I, h, w]): output o, row offset i reads the input channel
    // ((o·ft + ft − 1)·2 + i)·2 / r (the column offset never matters: checked by the caller).
    // d = (h, w, ft, r).
    kernel void vae_dup_up_add(device float *m [[buffer(0)]], device const float *s [[buffer(1)]],
                               constant uint4 &d [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
        uint h = d.x, w = d.y, ft = d.z, r = d.w, H = 2 * h, W = 2 * w;
        if (p.x >= W || p.y >= H) return;
        uint c = (((p.z * ft + ft - 1) * 2 + p.y % 2) * 2) / r;
        uint k = (p.z * H + p.y) * W + p.x;
        m[k] = m[k] + s[(c * h + p.y / 2) * w + p.x / 2];
    }

    // m [O, h, w] += AvgDown3D(x [I, s·h, s·w]): g members per output channel, member k of o is flat
    // o·g + k of the reference's view, in channel c = flat / (ft·s²), time t = (flat % (ft·s²)) / s²
    // (t < ft − 1: the zero frame in front), offset q = flat % s². d = (h, w, s, ft), e = (g).
    kernel void vae_avg_down_add(device float *m [[buffer(0)]], device const float *x [[buffer(1)]],
                                 constant uint4 &d [[buffer(2)]], constant uint4 &e [[buffer(3)]],
                                 uint3 p [[thread_position_in_grid]]) {
        uint h = d.x, w = d.y, s = d.z, ft = d.w, g = e.x, factor = ft * s * s;
        if (p.x >= w || p.y >= h) return;
        float sum = 0.0f;
        for (uint k = 0; k < g; k++) {
            uint flat = p.z * g + k, c = flat / factor, r = flat % factor;
            float v = 0.0f;
            if (r / (s * s) + 1 >= ft) {
                uint q = r % (s * s);
                v = x[(c * s * h + s * p.y + q / s) * (s * w) + s * p.x + q % s];
            }
            sum = k == 0 ? v : sum + v;
        }
        uint k = (p.z * h + p.y) * w + p.x;
        m[k] = m[k] + sum / float(g);
    }

    // The attention's tokens: t [S, C] <- x [C, S]. d = (C, S).
    kernel void vae_tokens(device const float *x [[buffer(0)]], device float *t [[buffer(1)]],
                           constant uint4 &d [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
        if (p.x >= d.x || p.y >= d.y) return;
        t[p.y * d.x + p.x] = x[p.x * d.y + p.y];
    }

    // The attention's residual: x [C, S] += o [S, C]. d = (C, S).
    kernel void vae_untokens_add(device float *x [[buffer(0)]], device const float *o [[buffer(1)]],
                                 constant uint4 &d [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
        if (p.x >= d.y || p.y >= d.x) return;
        uint k = p.y * d.y + p.x;
        x[k] = x[k] + o[p.x * d.x + p.y];
    }

    // ── bf16 (a decoder in half precision): copies move the 16 bits; a sum is computed in fp32 and rounded once. ──

    kernel void vae_double_bf16(device const ushort *x [[buffer(0)]], device ushort *y [[buffer(1)]],
                                constant uint4 &d [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
        uint h = d.x, w = d.y, H = 2 * h, W = 2 * w;
        if (p.x >= W || p.y >= H) return;
        y[(p.z * H + p.y) * W + p.x] = x[(p.z * h + p.y / 2) * w + p.x / 2];
    }

    kernel void vae_dup_up_add_bf16(device bfloat *m [[buffer(0)]], device const bfloat *s [[buffer(1)]],
                                    constant uint4 &d [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
        uint h = d.x, w = d.y, ft = d.z, r = d.w, H = 2 * h, W = 2 * w;
        if (p.x >= W || p.y >= H) return;
        uint c = (((p.z * ft + ft - 1) * 2 + p.y % 2) * 2) / r;
        uint k = (p.z * H + p.y) * W + p.x;
        m[k] = bfloat(float(m[k]) + float(s[(c * h + p.y / 2) * w + p.x / 2]));
    }

    kernel void vae_tokens_bf16(device const ushort *x [[buffer(0)]], device ushort *t [[buffer(1)]],
                                constant uint4 &d [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
        if (p.x >= d.x || p.y >= d.y) return;
        t[p.y * d.x + p.x] = x[p.x * d.y + p.y];
    }

    kernel void vae_untokens_add_bf16(device bfloat *x [[buffer(0)]], device const bfloat *o [[buffer(1)]],
                                      constant uint4 &d [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
        if (p.x >= d.y || p.y >= d.x) return;
        uint k = p.y * d.y + p.x;
        x[k] = bfloat(float(x[k]) + float(o[p.x * d.x + p.y]));
    }

    // A `GroupNorm`'s statistics, first half: group g is the contiguous run x[g·n, (g+1)·n) of [C, H, W]; thread t of it
    // takes the elements t, t + T, t + 2T… and writes their mean and the sum of their squared deviations from it (two passes
    // over its own elements, fp32), merged on the host (Chan's formula, in double). d = (n, T).
    kernel void vae_group_partials_bf16(device const bfloat *x [[buffer(0)]], device float2 *partial [[buffer(1)]],
                                        constant uint4 &d [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
        uint n = d.x, T = d.y, t = p.x, g = p.y;
        if (t >= T) return;
        device const bfloat *run = x + ulong(g) * n;
        float sum = 0.0f; uint count = 0;
        for (uint i = t; i < n; i += T) { sum += float(run[i]); count++; }
        float mean = count > 0 ? sum / float(count) : 0.0f, m2 = 0.0f;
        for (uint i = t; i < n; i += T) { float e = float(run[i]) - mean; m2 += e * e; }
        partial[g * T + t] = float2(mean, m2);
    }

    // y = act((x − mean_g) / √(var_g + ε) · γ_c + β_c) on [C, P] — `GroupNorm` with its affine, then the SiLU if d.w; fp32
    // inside, one rounding. d = (C, P, groups, silu), e = (bits of ε).
    kernel void vae_group_norm_bf16(device const bfloat *x [[buffer(0)]], device const float *mean [[buffer(1)]],
                                    device const float *variance [[buffer(2)]], device const bfloat *gamma [[buffer(3)]],
                                    device const bfloat *beta [[buffer(4)]], device bfloat *y [[buffer(5)]],
                                    constant uint4 &d [[buffer(6)]], constant uint4 &e [[buffer(7)]],
                                    uint3 p [[thread_position_in_grid]]) {
        uint C = d.x, P = d.y, c = p.y;
        if (p.x >= P || c >= C) return;
        uint g = c / (C / d.z);
        ulong k = ulong(c) * P + p.x;
        float v = (float(x[k]) - mean[g]) / sqrt(variance[g] + as_type<float>(e.x)) * float(gamma[c]) + float(beta[c]);
        if (d.w != 0) v = v / (1.0f + exp(-v));
        y[k] = bfloat(v);
    }

    // `F.normalize(x, dim=1) · √C · γ` on [C, P], one pixel per thread (its norm over the channels in fp32, floored at 1e-12),
    // then the SiLU if d.z. d = (C, P, silu).
    kernel void vae_rms_norm_bf16(device const bfloat *x [[buffer(0)]], device const bfloat *gamma [[buffer(1)]],
                                  device bfloat *y [[buffer(2)]], constant uint4 &d [[buffer(3)]],
                                  uint3 p [[thread_position_in_grid]]) {
        uint C = d.x, P = d.y;
        if (p.x >= P) return;
        float squares = 0.0f;
        for (uint c = 0; c < C; c++) { float v = float(x[ulong(c) * P + p.x]); squares += v * v; }
        float norm = max(sqrt(squares), 1e-12f), scale = sqrt(float(C));
        for (uint c = 0; c < C; c++) {
            ulong k = ulong(c) * P + p.x;
            float v = float(x[k]) / norm * (scale * float(gamma[c]));
            if (d.z != 0) v = v / (1.0f + exp(-v));
            y[k] = bfloat(v);
        }
    }
    """

    private let pipelines: [Kernel: MTLComputePipelineState]
    private let halfPipelines: [Kernel: MTLComputePipelineState]

    private init(device: MTLDevice) throws {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let library = try device.makeLibrary(source: Self.source, options: options)
        var pipelines: [Kernel: MTLComputePipelineState] = [:], halfPipelines: [Kernel: MTLComputePipelineState] = [:]
        for k in Kernel.allCases {
            if !Kernel.halfOnly.contains(k) {
                guard let f = library.makeFunction(name: k.rawValue) else { throw GEMM.Failure.noDevice }
                pipelines[k] = try device.makeComputePipelineState(function: f)
            }
            if !Kernel.fp32Only.contains(k) {
                guard let f = library.makeFunction(name: k.halfName) else { throw GEMM.Failure.noDevice }
                halfPipelines[k] = try device.makeComputePipelineState(function: f)
            }
        }
        self.pipelines = pipelines; self.halfPipelines = halfPipelines
    }

    nonisolated(unsafe) private static var cache: VAEKernels?
    private static let lock = NSLock()

    /// Compiled once per process (one GPU).
    static func shared(_ device: MTLDevice) throws -> VAEKernels {
        lock.lock(); defer { lock.unlock() }
        if let cache { return cache }
        let k = try VAEKernels(device: device)
        cache = k
        return k
    }

    /// `constants`: up to eight, as two `uint4` (buffers after the operands). `half`: the kernel's bf16 variant.
    func dispatch(_ kernel: Kernel, on queue: MTLCommandQueue, _ buffers: [MTLBuffer], constants: [UInt32], grid: MTLSize,
                  half: Bool = false) {
        guard let commands = queue.makeCommandBuffer(), let encoder = commands.makeComputeCommandEncoder() else {
            preconditionFailure("VAEKernels: no command buffer")
        }
        guard let pipeline = half ? halfPipelines[kernel] : pipelines[kernel] else {
            preconditionFailure("VAEKernels: \(kernel.rawValue) has no \(half ? "bf16" : "fp32") variant")
        }
        encoder.setComputePipelineState(pipeline)
        for (i, b) in buffers.enumerated() { encoder.setBuffer(b, offset: 0, index: i) }
        var c = constants + [UInt32](repeating: 0, count: 8 - constants.count)
        encoder.setBytes(&c, length: 16, index: buffers.count)
        c.withUnsafeBytes { encoder.setBytes($0.baseAddress! + 16, length: 16, index: buffers.count + 1) }
        encoder.dispatchThreads(grid, threadsPerThreadgroup: MTLSize(width: min(grid.width, 64), height: 4, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        precondition(commands.status == .completed, "VAEKernels: \(kernel.rawValue) failed: \(String(describing: commands.error))")
    }
}
