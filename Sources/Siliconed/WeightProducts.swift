import Accelerate
import Metal

/// **A product by a published weight, whatever the model**: `C[m, n] = A[m, k] · W`, plus the weight's
/// LoRA stack — the one place where the engine chooses how a weight is read and multiplied. A model
/// says what it computes (the weight's name, the shapes, its buffers); the plan (`CapacityPlan`, asked at every call with the
/// product's exact class) says how: the form the weight is read in, the rows each engine takes.
///
///     name ─▶ the weight into the reserve, by the shortest path the settings allow
///              ├─ path bf16, a bf16 weight     copied as it is (`Artifact.stage`)       ─▶ TensorGEMM
///              │     rounding                  the same, the activation rounded to bf16 first ─▶ TensorGEMM
///              │                               (`roundsActivations`, `BFloatKernels`)
///              │     + the ANE's last rows     the same bf16 copy, read in place            ─▶ the ANE's fp16 program
///              │                               (`ProductAssignment.ane` on a rounded bf16 weight) beside the GPU's rows
///              ├─ path native, a published int8  copied as it is, scale fp32 after it   ─▶ TensorGEMM, × s after the
///              │                                (convrot: x · R by chunks of rows in the reserve's tail)  sums
///              │     read in place              rounded or W8A8, the GPU's whole product, scale fp32 as published, on the stream:
///              │                                integers and scale read where the map holds them (`Artifact.wrappedMap`) —
///              │                                no copy, no drain; the whole reserve for the activation's chunks
///              │     or a GGUF Q8_0             (scales per 32 rows: q · s in threadgroup memory)
///              │     + the AMX's columns        widened alone, while the GPU               ─▶ BNNS on them
///              │                                computes the others
///              │     + the ANE's last rows      the same integers and scales, read in place ─▶ the ANE's int8 program,
///              │                                (`ProductAssignment.ane` on a native weight)   beside the GPU's rows
///              │     or W8A8                    the activation rotated and quantized to   ─▶ matmul2d int8 × int8 → int32,
///              │                                int8 per row in the reserve's tail          × s_x · s_w after (`QuantizedKernels`)
///              │                                — by its producer, a norm (`RoundedOperand.int8`), or once by the first product
///              │                                that reads it: the next ones read the same codes
///              │     or a K-quant (Q4_K, Q5_K)  its blocks copied as stored, unpacked on  ─▶ matmul2d bfloat × half per
///              │                                the GPU into sc·q, d, dmin·m (`KQuantKernels`)  super-block, the activation rounded
///              └─ path widened                 widened on the CPU (`Artifact.materialize`) ─▶ MPS (+ AMX, + ANE rows)
///          ─▶ C = A · W
///          ─▶ + the LoRA stack:  mid = A · down, then C += mid · up      after the product, each one a product of the plan (`lora`), or
///                                W += down · up, then C = A · W  merged, where it costs fewer FLOPs (the GPU's)
///
/// Every path computes in fp32 from the published values (a bf16 widening is a shift, an 8-bit one a
/// single multiplication): they differ in the order of the sums, hence the last bits, never in what is
/// computed. The default plan's path is the one every model ran before this module, bit for bit.
///
/// What sets it apart — what a model writing its own product would get wrong:
///
///   - **the LoRA stack is added after the product, whichever path made it, as two products of the plan**: the classes
///     `loraDown` (`m × k · r`) and `loraUp` (`m × r · n`, added to the output), cut by rows between the GPU, the AMX and the ANE's fp16
///     like any widened product, in fp32 beside the weights (its small deltas would be lost otherwise). The CPU engines' rows of
///     the second go to the reserve, dead once the base product is made, and are added to the output in one pass. A bf16-weight product
///     takes a LoRA like any other (the guard that kept Krea 2 under a LoRA widened was such a
///     local exception). The merge (`mergesLoRA`, Qwen-Image-2.1's: a rank-256 stack after the GEMM cost
///     2.0 s per 1024² evaluation, ∝ rows) needs an fp32 weight to add into: under `gemm_tensor` it falls
///     back to after the product;
///   - **the reserve is the model's, sized by its memory need** (`MemoryNeed`): this module wraps it, it
///     never allocates — a bf16 weight takes its first half, an int8 one its first quarter, then its
///     scales, then the rotated activation by as many rows as the rest holds (`int8Plan`). A weight read
///     in place (`inPlaceRounding`, `inPlaceInt8`) leaves it all to the activation: more rows a chunk, the same bits (the rows
///     are independent). Read in place, the map is charged in B whole (`MemoryNeed.inPlaceBytes`), or the plan does not read it so;
///   - **an int8 weight's rotation goes to the activation**: `W = R · q · s` (`Widen.dequantizeRotated`)
///     gives `x · W = (x · R) · q · s` — `R` is applied to `x` in fp32, the integers are never touched;
///   - **the ANE reads the integers where the plan gives it rows of a native weight** (`aneInt8Rows`): nothing is widened, nothing
///     copied for it but its surface; the GPU's rows run the int8 kernel on the same reserve, each row one engine's. A program the
///     process cannot run for the weight (`NeuralEngine.accepts`: its rotation's groups, a refusal) hands its rows to the GPU;
///   - **the ANE reads the bf16 copy where the plan gives it rows beside the GPU's rounded ones** (`bf16Product`): its program lifts and
///     converts the bf16 values as they are (a widening is a shift: the fp16 operand of the widened route, bit for bit), its rows read
///     the fp32 activation; the GPU's rows round theirs into the reserve's tail, past the weight the ANE reads. A program the process
///     cannot run (a refusal, a weight laid out `[n, k]`) or a non-finite output hands the rows back to the GPU's rounded kernel;
///   - **an operand quantized ahead survives the next int8 weights' staging** (`quantizedSlot`): its codes lie past the integers,
///     scales and AMX columns of the weight that asked for it (`Int8Plan.rotatedOffset`), so k and v, staged after q, still read
///     the codes the norm wrote; any write that may reach them (another path, a rotation, a LoRA, a pass of a product's own) drops it;
///   - **a rounded activation rounds the operand of a product and nothing else** (`roundsActivations`, which a
///     DiT sets and a text encoder does not): the caller's activation stays fp32 — the LoRA stack reads it —,
///     the rounded copy lives in the reserve's tail beside the weight (`roundingPlan`), the sums and the
///     output are fp32. Under the plan's rows (the modulation, the embedders) the product
///     keeps its fp32 operand: nothing to win there, and the most amplified error;
///   - **the timings keep their names** (`staging`, `widening`, `gemm wall`, `gemm GPU`, `lora weights`,
///     `lora wall`, `lora GPU`): the models add them to theirs, so a budget reads the same as before.
package final class WeightProducts {
    private let gemm: GEMM
    private let artifact: Artifact
    private let reserve: UnsafeMutablePointer<Float>
    /// The reserve's floats: a bf16 weight may take twice as many values.
    private let capacity: Int
    private let transposed: Bool
    private let lora: (any LoRAStacks)?
    private let mid: UnsafeMutablePointer<Float>?
    private let midFloats: Int
    private let mergesLoRA: Bool
    /// The operand of a product is rounded to bf16 — or quantized to int8 beside an int8 weight (W8A8) — where the plan, the settings and
    /// the machine allow it (`gemm_bf16_act`, `QuantizedKernels`): the DiT's products, never an encoder's.
    private let roundsActivations: Bool
    /// Every write into the reserve this module makes (a staging, a widening, a rounding pass, a LoRA's scratch), counted: an
    /// operand a producer rounded into it (`RoundedOperand`) is read only while nothing has written over it since.
    private var reserveWrites: UInt64 = 0
    /// The int8 operand handed out last (`roundedOperand`, W8A8): its generation, how far into the reserve the writes since reached —
    /// an int8 weight staged below its codes leaves them whole, so q, k and v each stage theirs and still read them —, and whether
    /// they are written yet (by the producer, or by the first product that read the operand). `nil`: overwritten, or none.
    private var quantizedSlot: (generation: UInt64, reach: Int, filled: Bool)?

    /// A write into the reserve that may reach anywhere in it: no operand rounded or quantized there before survives it.
    private func wroteReserve() {
        reserveWrites &+= 1
        quantizedSlot = nil
    }
    /// The last int8 or rounded bf16 product had rows on the ANE, and it computed them (`classification`).
    private var aneTook = false
    /// ... and it took every row of the product: the ledger names the route `ane/int8`, as the plan does (`ProductAssignment.route`).
    private var aneAlone = false

    /// Per phase, accumulated since the last `resetTimings`.
    package private(set) var timings: [String: Double] = [:]
    package func resetTimings() { timings.removeAll() }

    /// - Parameters:
    ///   - reserve: the model's weight reserve, `capacity` floats (the largest weight it multiplies by).
    ///   - transposed: the weights are laid out `[k, n]` by the forge (`Artifact.linearWeightsTransposed`).
    ///   - lora: the stack per module, `nil` without a LoRA; `mid`, its `[rows, rank]` scratch of `midFloats`.
    ///   - mergesLoRA: the stack may be merged into the widened weight where it costs fewer FLOPs.
    package init(gemm: GEMM, artifact: Artifact, reserve: UnsafeMutablePointer<Float>, capacity: Int,
                 transposed: Bool, lora: (any LoRAStacks)? = nil, mid: UnsafeMutablePointer<Float>? = nil,
                 midFloats: Int = 0, mergesLoRA: Bool = false, roundsActivations: Bool = false) {
        self.gemm = gemm; self.artifact = artifact; self.reserve = reserve; self.capacity = capacity
        self.transposed = transposed; self.lora = lora; self.mid = mid; self.midFloats = midFloats
        self.mergesLoRA = mergesLoRA; self.roundsActivations = roundsActivations
    }

    private func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        let result = try body()
        timings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        return result
    }

    // MARK: The operand rounded or quantized by its producer

    /// **Where the kernel that makes a product's operand may also write it rounded to bf16**, so the product does not round it
    /// again (`TensorGEMM.linear(…rounded:)`): the reserve's head, as `round_bf16` would have written it — the operand of a bf16 weight
    /// read in place on the stream, all its `m` rows in one chunk. The producer writes the fp32 operand too (the ANE's rows, the
    /// LoRA stack and any product that does not take this path read it), so a product that finds the reserve written since
    /// (`reserveWrites`) or another path rounds it again from there: the same bits either way — **unless the plan says no reader
    /// reads it** (`writesFP32`, `roundedOperand(…readers:)`): then the producer writes the rounded copy alone.
    ///
    /// **Or quantized to int8, for W8A8** (`int8`): the codes and the rows' scales where the product's plan puts them
    /// (`Int8Plan.quantizedOffset`), all `m` rows in one chunk, as `quantize_rows` writes them. A producer that holds the whole row
    /// writes them as it writes the fp32 operand (`ElementwiseGPU.normScale`, `byProducer`); any other leaves them to the first
    /// product that reads the operand, and the next ones read them — one pass per operand, not one per product (q, k and v; w1 and
    /// w3). A producer that only rounds reads `bf16`, `nil` here: the int8 slot is never its to write.
    package struct RoundedOperand {
        /// The bf16 slot (offset in bytes, 16-aligned), `nil` for an int8 operand.
        package let bf16: (buffer: MTLBuffer, offset: Int)?
        /// The int8 slot, `nil` for a bf16 operand.
        package let int8: Quantized?
        fileprivate let k: Int, rows: Int, generation: UInt64
        /// The producer writes the fp32 operand beside the bf16 copy. `false` only where its caller named every product that reads
        /// the buffer and the plan has each of them read the copy alone (`roundedOperand(…readers:)`): the fp32 write would be dead.
        package let writesFP32: Bool
        /// The operand's row length: its product's `k`.
        package var columns: Int { k }

        /// Where an int8 operand lies: its codes `[rows, k]` and its rows' scales fp32, in bytes, page-aligned at the first row; the
        /// rotation the product's weight carries (`stages`, `norm` as `quantize_rows` takes them).
        package struct Quantized {
            package let buffer: MTLBuffer
            package let codesOffset: Int, scalesOffset: Int
            package let group: Int?
            package let stages: Int
            package var norm: Float { group.map { 1 / Float($0).squareRoot() } ?? 1 }
            /// The producer writes the codes (`ElementwiseGPU.normScale`); else the first product that reads the operand does.
            package let byProducer: Bool

            /// The slot as a producer's quantizing kernel takes it — the codes at `index`, the scales at `index + 1`, the rotation
            /// (`uint2`: its stages, its norm's bits) at `index + 2` — counted (`GEMM.producerQuantized`).
            package func bind(_ encoder: MTLComputeCommandEncoder, at index: Int) {
                encoder.setBuffer(buffer, offset: codesOffset, index: index)
                encoder.setBuffer(buffer, offset: scalesOffset, index: index + 1)
                var rotation = SIMD2<UInt32>(UInt32(stages), norm.bitPattern)
                encoder.setBytes(&rotation, length: MemoryLayout<SIMD2<UInt32>>.stride, index: index + 2)
                GEMM.producerQuantized.withLock { $0 += 1 }
            }
        }

        /// An int8 operand no product reads (a producer's kernel run alone, the tests): never a live slot.
        package init(producing q: Quantized, k: Int, rows: Int) {
            self.init(bf16: nil, int8: q, k: k, rows: rows, generation: 0)
        }

        fileprivate init(bf16: (buffer: MTLBuffer, offset: Int)?, int8: Quantized?, k: Int, rows: Int, generation: UInt64,
                         writesFP32: Bool = true) {
            self.bf16 = bf16; self.int8 = int8; self.k = k; self.rows = rows; self.generation = generation
            self.writesFP32 = writesFP32
        }

        /// The same operand from its row `first` on: a product over a slice of the rows (a chunk of an MLP), a producer by bands.
        package func rows(from first: Int) -> RoundedOperand {
            RoundedOperand(bf16: bf16.map { ($0.buffer, $0.offset + first * k * 2) },
                           int8: int8.map { Quantized(buffer: $0.buffer, codesOffset: $0.codesOffset + first * k,
                                                      scalesOffset: $0.scalesOffset + first * 4, group: $0.group, stages: $0.stages,
                                                      byProducer: $0.byProducer) },
                           k: k, rows: rows - first, generation: generation, writesFP32: writesFP32)
        }
    }

    /// The slot for `name`'s operand (`[m, k]`), or `nil` where its product would not read it (another path, rows in several
    /// chunks, the stream off for a bf16 one, `SILICONED_ROUND_FUSED=0`). Asked just before the producer is encoded: nothing but the
    /// products that read the operand may write the reserve until they are encoded. `quantizes`: the producer can write an int8
    /// operand too, a row of up to `holding` columns (`ElementwiseGPU.normScale`'s `quantizedColumns`, `swiglu`'s
    /// `swigluQuantizedColumns`); else it is the first product's. `readers`: every product that reads the buffer the producer writes
    /// (`name`'s among them), each weight with its `n`, where nothing else reads it before it is written again — the caller's
    /// knowledge; the plan says the rest (`fp32Unread`). `nil`: not said, the fp32 operand written.
    package func roundedOperand(_ name: String, m: Int, k: Int, n: Int, quantizes: Bool = false,
                                holding: Int = ElementwiseGPU.quantizedColumns,
                                readers: [(name: String, n: Int)]? = nil) throws -> RoundedOperand? {
        guard Self.fusesRounding else { return nil }
        let planned = gemm.assignment(k: k, n: n, m: m, weight: format(of: name))
        let w = try gemm.wrap(UnsafeMutableRawPointer(reserve), bytes: capacity * 4, name: "w")
        if let (_, _, rounding) = inPlaceRounding(name, k: k, n: n, planned: planned), rounding.rows >= m {
            let writesFP32 = readers.map { !fp32Unread(by: $0, m: m, k: k) } ?? true
            if !writesFP32 { GEMM.fp32Unwritten.withLock { $0 += 1 } }
            wroteReserve()                         // the producer writes the slot: any operand rounded there before is gone
            return RoundedOperand(bf16: (w, rounding.offset), int8: nil, k: k, rows: m, generation: reserveWrites,
                                  writesFP32: writesFP32)
        }
        // The slot where the product will look for it: the in-place plan's (the whole reserve to the codes) or the copy's.
        let inPlacePlan = inPlaceInt8(name, k: k, n: n, m: m, planned: planned).flatMap { $0.plan.quantizedOffset > 0 ? $0.plan : nil }
        guard Self.sharesQuantized != .off, let plan = inPlacePlan ?? quantizedPlan(name, k: k, n: n, planned: planned), plan.chunk >= m,
              aneInt8Rows(planned, plan, m: m, k: k, n: n) == 0, let stages = W8A8.stages(group: plan.group) else { return nil }
        let byProducer = quantizes && Self.sharesQuantized == .producer && k <= holding
        wroteReserve()
        quantizedSlot = (reserveWrites, 0, byProducer)
        return RoundedOperand(bf16: nil, int8: .init(buffer: w, codesOffset: plan.quantizedOffset, scalesOffset: plan.quantizedScalesOffset,
                                                     group: plan.group, stages: stages, byProducer: byProducer),
                              k: k, rows: m, generation: reserveWrites)
    }

    /// **No reader reads the fp32 operand**: each product of `readers` (`[m, k]` on its weight) takes the in-place path on the rounded
    /// copy its producer wrote — `linear`'s reuse, one chunk at the reserve's head —, with no ANE rows (they read the fp32 operand,
    /// `aneRowsBesideRounded`) and no LoRA stack (it reads it too: any stack keeps it, whether or not it has the module). Read in place,
    /// a product writes nothing into the reserve (`prepare`), so the copy is still whole for the next reader. `linear` checks it holds.
    private func fp32Unread(by readers: [(name: String, n: Int)], m: Int, k: Int) -> Bool {
        guard lora == nil, !readers.isEmpty else { return false }
        return readers.allSatisfy { reader in
            let planned = gemm.assignment(k: k, n: reader.n, m: m, weight: format(of: reader.name))
            guard let (_, _, rounding) = inPlaceRounding(reader.name, k: k, n: reader.n, planned: planned),
                  rounding.offset == 0, rounding.rows >= m else { return false }
            return aneRowsBesideRounded(planned, m: m) == 0
        }
    }

    /// The ANE's rows beside the GPU's rounded ones on a bf16 weight (`linear`): the plan's last rows, where it gives some but not all.
    private func aneRowsBesideRounded(_ planned: ProductAssignment, m: Int) -> Int {
        planned.path == .bf16 && planned.ane > 0 && planned.ane < m && transposed ? planned.ane : 0
    }

    /// `SILICONED_ROUND_FUSED=0`: every product rounds or quantizes its own operand again, as before the producers wrote it (a forcing
    /// of measurement: the same bits).
    package static let fusesRounding = ProcessInfo.processInfo.environment["SILICONED_ROUND_FUSED"] != "0"

    /// **Who quantizes a W8A8 operand that several products read** (`SILICONED_QUANT_FUSED`, a forcing of measurement: the same bits):
    /// `producer`, the default — the producer that can (`roundedOperand(…quantizes:)`), else the first product; `pass` (`pass`) — the
    /// first product always, the next ones read its codes; `off` (`0`) — every product its own, as before.
    package enum QuantizedSharing { case off, pass, producer }
    package static let sharesQuantized: QuantizedSharing = switch ProcessInfo.processInfo.environment["SILICONED_QUANT_FUSED"] {
    case "0", "off": .off
    case "pass": .pass
    default: .producer
    }

    /// The int8 slot of `operand` where this W8A8 product of `plan` may read it: handed out last, nothing written over its codes since
    /// (`quantizedSlot`, this product's staging included), the same rotation, all the rows. With whether its codes are written yet.
    private func sharedQuantized(_ operand: RoundedOperand?, plan: Int8Plan, w: MTLBuffer, m: Int, k: Int)
        -> (quantizing: TensorGEMM.Quantizing, written: Bool)? {
        guard let operand, let q = operand.int8, let slot = quantizedSlot, slot.generation == operand.generation,
              slot.reach <= q.codesOffset, plan.quantizedOffset > 0, plan.group == q.group, operand.k == k, operand.rows >= m,
              q.buffer === w else { return nil }
        return ((w, q.codesOffset, q.scalesOffset, operand.rows), slot.filled)
    }

    /// **`C = A · W` (+ the stack of `target`)**, `W` the map's `name`. `a` and `c` are the model's
    /// buffers, wrapped at their reserved size (`GEMM.wrap` memoizes by address). `target`: the module
    /// the LoRA stack is keyed by — `name` without `.weight` unless said. `rounded`: `a` already rounded by its producer
    /// (`roundedOperand`), read from there where this product takes the path it was given for.
    package func linear(_ name: String, a: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                        target: String? = nil, rounded operand: RoundedOperand? = nil) throws {
        let w = try gemm.wrap(UnsafeMutableRawPointer(reserve), bytes: capacity * 4, name: "w")
        let (path, planned, skipped) = try prepare(name, count: k * n, m: m, n: n)
        let staged = path != .widened
        let target = target ?? String(name.dropLast(".weight".count))
        // An operand whose fp32 the producer did not write (`RoundedOperand.writesFP32`): only the in-place rounded path reads it.
        let fp32Written = operand?.writesFP32 ?? true
        precondition(fp32Written || (inPlace != nil && lora == nil), "\(name): reads an fp32 operand its producer did not write")
        var stack: (down: MTLBuffer, up: MTLBuffer, rank: Int)?
        if let lora, let module = try timed("lora weights", { try lora.module(target, k: k, n: n) }) {
            stack = (try gemm.wrap(UnsafeMutableRawPointer(module.down), bytes: k * module.rank * 4, name: "loraDown"),
                     try gemm.wrap(UnsafeMutableRawPointer(module.up), bytes: module.rank * n * 4, name: "loraUp"),
                     module.rank)
        }
        let merged = stack != nil && mergesLoRA && !staged && LoRA.merges(rows: m, k: k, n: n)
        if merged, let stack {
            timed("lora wall") {
                timings["lora GPU", default: 0] += gemm.accumulatedLinear(a: stack.down, b: stack.up, c: w,
                                                                         m: k, k: stack.rank, n: n)
            }
        }
        var failure: String?
        // The render's check (`ProductLedger`): this product's wall seconds, as the class and the route it is.
        let ledger = ProductLedger.current
        let began = ledger == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        // On a stream that queues, the call only encodes: its seconds are the buffers' and the ANE's (`ProductLedger.Charge`).
        let stream = gemm.queue.stream
        let charge = ledger.flatMap { stream.isOn ? ProductLedger.Charge(ledger: $0) : nil }
        let previousCharge = charge == nil ? nil : stream.charge
        if let charge { stream.charge = charge }
        // The ANE's rows the plan gave and the process did not hand it (`ProductLedger.Reason.aneNotAccepted`).
        var aneDeclined = false
        let givenBackBefore = planned.ane > 0 && ledger != nil ? GEMM.neuralEngineTally.withLock { $0.givenBack } : 0
        timed("gemm wall") {
            switch path {
            case .bf16(let rounding?):
                // The ANE beside the GPU's rounded rows: the plan's last rows, read from the bf16 copy (`bf16Product`), where the process
                // can run them.
                let aneRows = aneRowsBesideRounded(planned, m: m)
                if let (map, offset) = inPlace {
                    // The operand its producer rounded, where nothing wrote over it since; else this product's own pass writes the
                    // reserve.
                    let slot = operand?.bf16
                    let reuse = operand.map { $0.generation == reserveWrites && $0.k == k && $0.rows >= m && slot?.buffer === w
                                              && rounding.offset == 0 && rounding.rows >= m } ?? false
                    if reuse { GEMM.producerRounded.withLock { $0 += 1 } } else { wroteReserve() }
                    // Its producer left the fp32 operand unwritten (`fp32Unread`): this product must read the copy alone, all its rows.
                    precondition(fp32Written || (reuse && aneRows == 0), "\(name): reads an fp32 operand its producer did not write")
                    let rounding = reuse ? Rounding(offset: slot!.offset, rows: rounding.rows) : rounding
                    // On the stream: the ANE's rows beside the queued GPU work, waited for on the GPU (`GPUStream.neuralEngine`).
                    if aneRows > 0, GPUStream.policy == .on, let ane = gemm.neuralEngine, ane.accepts(rows: aneRows, k: k, n: n),
                       let weight = artifact.pointer(name) {
                        let first = m - aneRows
                        nonisolated(unsafe) let (pa, pc) = (a.contents().assumingMemoryBound(to: Float.self),
                                                            c.contents().assumingMemoryBound(to: Float.self))
                        nonisolated(unsafe) let pw = weight.assumingMemoryBound(to: UInt16.self)
                        let stream = gemm.queue.stream
                        let done = stream.neuralEngine(on: gemm.queue, work: {
                            ane.product(a: pa, bf16: pw, c: pc, first: first, rows: aneRows, k: k, n: n)
                        }, fallback: {
                            Self.cpuRows(a: pa, bf16: pw, c: pc, first: first, rows: aneRows, k: k, n: n)
                        })
                        if first > 0 {
                            timings["gemm GPU", default: 0] += gemm.linear(a: a, rounding: (w, rounding.offset, rounding.rows), bf16: map,
                                                                          weightOffset: offset, c: c, m: first, k: k, n: n,
                                                                          weightIsTransposed: transposed, rounded: reuse)
                        }
                        stream.neuralEngineQueued(done)
                        aneTook = true
                    } else {
                        aneTook = false
                        aneDeclined = planned.ane > 0
                        timings["gemm GPU", default: 0] += gemm.linear(a: a, rounding: (w, rounding.offset, rounding.rows), bf16: map,
                                                                      weightOffset: offset, c: c, m: m, k: k, n: n,
                                                                      weightIsTransposed: transposed, rounded: reuse)
                    }
                } else if aneRows > 0, let ane = gemm.neuralEngine, ane.accepts(rows: aneRows, k: k, n: n) {
                    let r = Self.bf16Product(gemm: gemm, rounding: (w, rounding.offset, rounding.rows), reserve: w, a: a, c: c,
                                             m: m, k: k, n: n, ane: ane, aneRows: aneRows)
                    timings["gemm GPU", default: 0] += r.gpu
                    aneTook = r.computed
                } else {
                    aneTook = false
                    aneDeclined = planned.ane > 0
                    timings["gemm GPU", default: 0] += gemm.linear(a: a, rounding: (w, rounding.offset, rounding.rows), bf16: w,
                                                                  c: c, m: m, k: k, n: n, weightIsTransposed: transposed)
                }
            case .bf16(nil):
                timings["gemm GPU", default: 0] += gemm.linear(a: a, bf16: w, c: c, m: m, k: k, n: n,
                                                              weightIsTransposed: transposed)
            case .int8(let plan):
                GEMM.int8Products.withLock { $0 += 1 }
                // The ANE beside the GPU on the integers: the plan's last rows (`ProductAssignment.ane` on a native weight), read from
                // the reserve beside the GPU's int8 kernel. The engine takes them only if it can run them (`aneInt8Rows`). A weight read
                // in place has none (`inPlaceInt8`).
                let aneRows = int8InPlace == nil ? aneInt8Rows(planned, plan, m: m, k: k, n: n) : 0
                if aneRows > 0, let ane = gemm.neuralEngine {
                    quantizedSlot = nil                    // the GPU's rows quantize theirs into the tail
                    let r = Self.int8Product(gemm: gemm, plan: plan, reserve: w, a: a, c: c, m: m, k: k, n: n, ane: ane, aneRows: aneRows)
                    timings["gemm GPU", default: 0] += r.gpu
                    aneTook = r.computed
                    aneAlone = r.computed && aneRows == m
                } else {
                    aneTook = false; aneAlone = false
                    aneDeclined = planned.ane > 0
                    // W8A8 on an operand quantized ahead (`RoundedOperand.int8`): read where its producer or the product before wrote
                    // it, or written there by this one for the next. Anything else writes the reserve's tail (its own pass, a rotation).
                    let shared = sharedQuantized(operand, plan: plan, w: w, m: m, k: k)
                    if let shared, shared.written { GEMM.prequantizedProducts.withLock { $0 += 1 } }
                    let r = Self.int8Product(gemm: gemm, plan: plan, reserve: w, a: a, c: c, m: m, k: k, n: n, weight: int8InPlace,
                                             quantized: shared)
                    if shared != nil { quantizedSlot?.filled = true } else { quantizedSlot = nil }
                    timings["gemm GPU", default: 0] += r.gpu
                    timings["widening", default: 0] += r.widening
                    if r.status != 0 { failure = "\(name): the AMX refused its columns (BNNS \(r.status))" }
                }
            case .kquant(let layout):
                GEMM.roundedProducts.withLock { $0 += 1 }
                GEMM.kquantProducts.withLock { $0 += 1 }
                aneTook = false; aneAlone = false
                timings["gemm GPU", default: 0] += gemm.tensor!.linear(a: a, kquant: w, layout: layout, c: c, m: m, k: k, n: n)
            case .widened:
                timings["gemm GPU", default: 0] += gemm.linear(a: a, b: w, c: c, m: m, k: k, n: n,
                                                              weightIsTransposed: transposed, weight: format(of: name),
                                                              assignment: planned)
            }
        }
        if let ledger {
            let (weight, route) = classification(of: name, path)
            let wall = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
            if let charge {
                stream.charge = previousCharge
                charge.finish(.product(k: k, n: n, m: m, weight: weight), route: route, wall: wall)
            } else {
                ledger.add(.product(k: k, n: n, m: m, weight: weight), route: route, seconds: wall)
            }
            // What ran otherwise than planned, and why: the plan's own fallback first, then the path skipped, then the ANE's rows.
            let key = ProductKey(k: k, n: n, m: m, weight: format(of: name), phase: gemm.phase)
            let reason: ProductLedger.Reason?
            if let unsolved = gemm.plan.fallback(key) { reason = unsolved }
            else if let skipped { reason = skipped }
            else if planned.ane > 0 && !route.contains("ane") {
                let givenBack = GEMM.neuralEngineTally.withLock { $0.givenBack } > givenBackBefore
                reason = aneDeclined || !givenBack ? .aneNotAccepted : .aneGivenBack
            } else { reason = nil }
            if let reason { ledger.fellBack(.product(k: k, n: n, m: m, weight: weight), planned: planned.route(weight: key.weight), reason: reason) }
        }
        // A refused slice leaves its columns as they were: a plausible image, wrong exactly there.
        if let failure { throw Artifact.Failure.badHeader(failure) }
        // The stack writes the reserve where the CPU's engines take rows of its second product (`GEMM.accumulatedLinear`'s scratch),
        // or where it was merged into the weight there; wholly the GPU's, it does not.
        if let stack, merged || !(gemm.runsOnGPUAlone(k: k, n: stack.rank, m: m, weight: .loraDown)
                                  && gemm.runsOnGPUAlone(k: stack.rank, n: n, m: m, weight: .loraUp)) { wroteReserve() }
        if !merged, let stack, let mid {
            let mb = try gemm.wrap(UnsafeMutableRawPointer(mid), bytes: midFloats * 4, name: "loraMid")
            lora(x: a, stack: stack, mid: mb, c: c, scratch: w, m: m, k: k, n: n)
        }
    }

    /// **The stack's two products, each one the plan's**: `mid = x · down` (`loraDown`, written) then `C += mid · up` (`loraUp`,
    /// accumulated: the GPU's rows inside its kernel, the CPU engines' through the reserve the product has just left, `GEMM.accumulatedLinear`).
    /// Each is a class of the render, in the ledger with the route it took.
    private func lora(x: MTLBuffer, stack: (down: MTLBuffer, up: MTLBuffer, rank: Int), mid: MTLBuffer, c: MTLBuffer,
                      scratch: MTLBuffer, m: Int, k: Int, n: Int) {
        let ledger = ProductLedger.current
        func record(_ weight: CapacityClasses.WeightFormat, k: Int, n: Int, began: UInt64) {
            ledger?.add(.product(k: k, n: n, m: m, weight: weight), route: gemm.lastEngines,
                        seconds: Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9)
        }
        // Both products wholly the GPU's: one submission, as before the stack was the plan's (the same bits, the same cost). The ledger
        // shares its seconds between the two classes by their FLOP.
        if gemm.runsOnGPUAlone(k: k, n: stack.rank, m: m, weight: .loraDown), gemm.runsOnGPUAlone(k: stack.rank, n: n, m: m, weight: .loraUp) {
            timed("lora wall") {
                let began = DispatchTime.now().uptimeNanoseconds
                timings["lora GPU", default: 0] += gemm.lora(x: x, down: stack.down, mid: mid, up: stack.up, c: c,
                                                            m: m, k: k, r: stack.rank, n: n)
                if let ledger {
                    let seconds = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9, share = Double(k) / Double(k + n)
                    ledger.add(.product(k: k, n: stack.rank, m: m, weight: .loraDown), route: "gpu", seconds: seconds * share)
                    ledger.add(.product(k: stack.rank, n: n, m: m, weight: .loraUp), route: "gpu", seconds: seconds * (1 - share))
                }
            }
            return
        }
        timed("lora wall") {
            var began = DispatchTime.now().uptimeNanoseconds
            timings["lora GPU", default: 0] += gemm.linear(a: x, b: stack.down, c: mid, m: m, k: k, n: stack.rank,
                                                          weightIsTransposed: true, weight: .loraDown)
            record(.loraDown, k: k, n: stack.rank, began: began)
            began = DispatchTime.now().uptimeNanoseconds
            timings["lora GPU", default: 0] += gemm.accumulatedLinear(a: mid, b: stack.up, c: c, m: m, k: stack.rank, n: n,
                                                                     weight: .loraUp, scratch: scratch)
            record(.loraUp, k: stack.rank, n: n, began: began)
        }
    }

    /// **The class of the product that just ran, as the plan names it**: the weight's format (as published) and the route taken: the
    /// engines the widened product ran on (`GEMM.lastEngines`), or the weight's own form (`gpu/bf16`, `gpu/bf16act`, `gpu/int8`,
    /// `gpu+amx/int8`, `gpu/q8`, `gpu+amx/q8`, and `…act` where the GPU's rows rounded their activation — what ran: a rounding the
    /// reserve could not hold is the fp32 kernel's name).
    private func classification(of name: String, _ path: Path) -> (CapacityClasses.WeightFormat, String) {
        switch path {
        case .bf16(let rounding): return (.bf16, rounding == nil ? "gpu/bf16" : aneTook ? "gpu+ane/bf16act" : "gpu/bf16act")
        case .int8(let plan):
            let weight: CapacityClasses.WeightFormat = plan.block == nil ? .int8 : .q8
            if plan.quantizedOffset > 0 { return (weight, "gpu/int8x8") }
            let act = plan.roundedOffset > 0 ? "act" : ""
            if aneTook { return (weight, aneAlone ? "ane/" + weight.rawValue : "gpu+ane/" + weight.rawValue + act) }
            return (weight, (plan.amxColumns > 0 && gemm.conductor != nil ? "gpu+amx/" : "gpu/") + weight.rawValue + act)
        case .kquant:
            return (.q4k, "gpu/q4kact")
        case .widened:
            return (format(of: name), gemm.lastEngines)
        }
    }

    /// The published weight's format: an 8-bit one's (`int8`, or `q8` for blocks of scales), `q4k` for a GGUF-packed one (they
    /// widen), `bf16` for the rest.
    private func format(of name: String) -> CapacityClasses.WeightFormat {
        guard let tensor = artifact.tensors[name] else { return .bf16 }
        if tensor.dtype == .int8 { return tensor.scale?.block == nil ? .int8 : .q8 }
        if tensor.dtype.sqBits != nil { return .int8 }          // decoded to int8 for its product (`prepareSQ`)
        return tensor.dtype.isPacked ? .q4k : .bf16
    }

    /// How the weight lies in the reserve, hence which product reads it.
    private enum Path: Equatable {
        case widened
        /// A bf16 weight staged as published; its activation rounded into the reserve's tail, or fp32.
        case bf16(Rounding?)
        case int8(Int8Plan)
        /// A K-quant's blocks staged as stored, unpacked and multiplied on the GPU on the rounded activation (`gpu/q4kact`).
        case kquant(TensorGEMM.KQuantLayout)
    }

    /// Where the rounded activation lies in the reserve and how many rows at a time (a multiple of 64).
    package struct Rounding: Equatable {
        package let offset, rows: Int
    }

    /// **Where a bf16 activation fits beside a bf16 weight of `k · n` values** in a reserve of `capacity`
    /// floats: after the weight's pages, `rows` rows of `k` bf16 values, or `nil` when fewer than 64 rows
    /// would (the reserve is the model's, never grown for this path) or `k` is odd.
    package static func roundingPlan(k: Int, n: Int, capacity: Int, weightInReserve: Bool = true) -> Rounding? {
        guard k % 2 == 0 else { return nil }
        // A weight read in place (`inPlace`) leaves the whole reserve to the operand: at 1 024² Z-Image's `w2` (k 10 240) is then
        // one chunk of 4 128 rows instead of 3 840 + 288 — the same bits (rows are independent), one dispatch.
        let offset = weightInReserve ? (2 * k * n + 16383) / 16384 * 16384 : 0
        let rows = (capacity * 4 - offset) / (2 * k) / 64 * 64
        return rows >= 64 ? Rounding(offset: offset, rows: rows) : nil
    }

    /// An int8 weight in the reserve: `q` at 0, its fp32 scales at `scalesOffset`, the AMX's last
    /// `amxColumns` widened at `sliceOffset`, the rotated activation (convrot of `group`) at
    /// `rotatedOffset`, `chunk` rows at a time. Offsets in bytes, page-aligned. A weight read in place (`Int8Source`) leaves `q`
    /// and its scales out: the activation from the second page (`int8Plan(…weightInReserve:)`).
    package struct Int8Plan: Equatable {
        package let scalesOffset, sliceOffset, rotatedOffset, chunk, amxColumns: Int
        package let perColumn: Bool
        package let group: Int?
        /// Scales per block of this many rows and column (GGUF Q8_0); `nil`: per column or tensor.
        package var block: Int? = nil
        /// The activation rounded to bf16 (`gemm_bf16_act`) at this offset, `chunk` rows at a time; 0: fp32.
        package var roundedOffset = 0
        /// The activation quantized to int8 (W8A8, `QuantizedKernels`): its codes at this offset, `chunk` rows at a time, their scales
        /// at `quantizedScalesOffset`; 0: not quantized.
        package var quantizedOffset = 0, quantizedScalesOffset = 0
        package var layout: Widen.ScaleLayout {
            if let block { return .blocks(columns: 0, rows: block) }
            return perColumn ? .columns(0) : .tensor
        }
    }

    /// Where an int8 `[k, n]` weight, the AMX's widened columns and the rotated activation fit in
    /// `capacity` floats, or `nil` when fewer than 64 rows would: the reserve is the model's, never
    /// grown for this path.
    ///
    /// `quantized` (W8A8): the activation's int8 codes (`k` bytes a row) then their fp32 scales (4 bytes a row) past the weight's scales —
    /// no fp32 scratch, the kernel rotates in registers —, never beside the AMX's columns or Q8_0's blocks, for a rotation group the kernel
    /// takes. All of it lies in the reserve the model's memory need already holds (the largest weight in fp32: `4 k n` bytes, of which the
    /// integers take `k n`): nothing is charged for it, and a reserve that leaves fewer than 64 rows refuses the route.
    ///
    /// `weightInReserve` false: the integers and their scales are read in place in the map (`inPlaceInt8`), the reserve holds the
    /// activation alone, from its second page — an offset of 0 is the plan's "none" (`roundedOffset`, `quantizedOffset`); `scalesOffset`
    /// and `sliceOffset` are then 0 and unread, and no AMX columns (they are widened from the reserve's copy).
    package static func int8Plan(k: Int, n: Int, capacity: Int, perColumn: Bool, group: Int?, amxColumns: Int,
                                 block: Int? = nil, rounded: Bool = false, quantized: Bool = false,
                                 weightInReserve: Bool = true) -> Int8Plan? {
        func page(_ bytes: Int) -> Int { (bytes + 16383) / 16384 * 16384 }
        guard weightInReserve || amxColumns == 0 else { return nil }
        let scales = block.map { (k + $0 - 1) / $0 * n } ?? (perColumn ? n : 1)
        let scalesOffset = weightInReserve ? page(k * n) : 0
        let sliceOffset = weightInReserve ? page(scalesOffset + scales * 4) : 0
        let rotatedOffset = weightInReserve ? page(sliceOffset + k * amxColumns * 4) : 16384
        guard rotatedOffset <= capacity * 4 else { return nil }
        if quantized {
            guard block == nil, amxColumns == 0, W8A8.stages(group: group) != nil else { return nil }
            // The codes, then a page boundary at most, then the scales.
            let rows = (capacity * 4 - rotatedOffset - 16384) / (k + 4) / 64 * 64
            guard rows >= 64 else { return nil }
            let scalesAt = page(rotatedOffset + rows * k)
            guard scalesAt + rows * 4 <= capacity * 4 else { return nil }
            return Int8Plan(scalesOffset: scalesOffset, sliceOffset: sliceOffset, rotatedOffset: rotatedOffset, chunk: rows,
                            amxColumns: 0, perColumn: perColumn, group: group, block: nil,
                            quantizedOffset: rotatedOffset, quantizedScalesOffset: scalesAt)
        }
        if rounded {
            // The rotated rows (fp32, only under a rotation) then the same rows rounded (bf16): 6 k bytes a row.
            guard k % 2 == 0 else { return nil }
            let rows = (capacity * 4 - rotatedOffset) / (k * (group == nil ? 2 : 6)) / 64 * 64
            guard rows >= 64 else { return nil }
            let roundedOffset = group == nil ? rotatedOffset : (rotatedOffset + rows * k * 4 + 16383) / 16384 * 16384
            guard roundedOffset + rows * k * 2 <= capacity * 4 else { return nil }
            return Int8Plan(scalesOffset: scalesOffset, sliceOffset: sliceOffset, rotatedOffset: rotatedOffset,
                            chunk: rows, amxColumns: amxColumns, perColumn: perColumn, group: group, block: block,
                            roundedOffset: roundedOffset)
        }
        let rows = (capacity * 4 - rotatedOffset) / (k * 4) / 64 * 64
        guard group == nil || rows >= 64 else { return nil }
        return Int8Plan(scalesOffset: scalesOffset, sliceOffset: sliceOffset, rotatedOffset: rotatedOffset,
                        chunk: rows, amxColumns: amxColumns, perColumn: perColumn, group: group, block: block)
    }

    /// **Where an int8 product reads its integers and their fp32 scales outside the reserve**: the wrapped map, in place
    /// (`inPlaceInt8`) — `offset` the tensor's, `scalesOffset` its scale's, both 16-aligned in `buffer`.
    package struct Int8Source {
        package let buffer: MTLBuffer
        package let offset, scalesOffset: Int
    }

    /// **The int8 product of a plan, the weight already in `reserve`** (`q`, its fp32 scales) — or read from `weight` in place, the
    /// activation's rows alone in the reserve, the product queued on the stream (the GPU alone: `plan` has no AMX columns): the GPU on
    /// the first columns — all of them without a conductor or AMX columns —, the CPU widening the last ones
    /// meanwhile, the AMX on them. What the map measures (`Diagnostic`) is this function. Returns the GPU
    /// seconds (0 queued), the widening's, BNNS's status (0, or its refusal: those columns are then unwritten).
    ///
    /// `quantized` (W8A8): the activation quantized into an operand's slot instead of the plan's (`RoundedOperand.int8`) — read from
    /// there if `written`, else written there in one chunk.
    package static func int8Product(gemm: GEMM, plan: Int8Plan, reserve w: MTLBuffer, a: MTLBuffer, c: MTLBuffer,
                                    m: Int, k: Int, n: Int, weight: Int8Source? = nil,
                                    quantized: (quantizing: TensorGEMM.Quantizing, written: Bool)? = nil)
        -> (gpu: Double, widening: Double, status: Int32) {
        let rotation = plan.group.map { (group: $0, scratch: w, offset: plan.rotatedOffset, chunk: plan.chunk) }
        let rounding = plan.roundedOffset > 0 ? (buffer: w, offset: plan.roundedOffset, rows: plan.chunk) : nil
        let quantizing = quantized?.quantizing ?? (plan.quantizedOffset > 0
            ? (buffer: w, codesOffset: plan.quantizedOffset, scalesOffset: plan.quantizedScalesOffset, rows: plan.chunk) : nil)
        if rounding != nil { GEMM.roundedProducts.withLock { $0 += 1 } }
        guard plan.amxColumns > 0, let conductor = gemm.conductor else {
            let (q, qOffset, sOffset) = weight.map { ($0.buffer, $0.offset, $0.scalesOffset) } ?? (w, 0, plan.scalesOffset)
            return (gemm.tensor!.linear(a: a, int8: q, weightOffset: qOffset, scales: q, scalesOffset: sOffset, perColumn: plan.perColumn,
                                        block: plan.block, rotation: rotation, rounding: rounding, quantizing: quantizing,
                                        prequantized: quantized?.written ?? false, c: c, m: m, k: k, n: n, settled: weight != nil), 0, 0)
        }
        precondition(weight == nil, "the AMX's columns are widened from the reserve's copy")
        let g = n - plan.amxColumns
        let commands = gemm.tensor!.submit(a: a, int8: w, scales: w, scalesOffset: plan.scalesOffset, perColumn: plan.perColumn,
                                           block: plan.block, rotation: rotation, c: c, m: m, k: k, n: n, columns: g)
        let raw = w.contents(), slice = raw + plan.sliceOffset, scales = raw + plan.scalesOffset
        let started = DispatchTime.now().uptimeNanoseconds
        if let group = plan.group {
            Widen.dequantizeRotated(raw, rows: k, columns: n, group: group, scale: scales,
                                    scaleType: .float32, layout: plan.perColumn ? .columns(n) : .tensor,
                                    columns: g..<n, into: slice.assumingMemoryBound(to: Float.self))
        } else {
            Widen.dequantizeColumns(raw, rows: k, columns: n, scales: scales.assumingMemoryBound(to: Float.self),
                                    layout: plan.layout, columns: g..<n, into: slice.assumingMemoryBound(to: Float.self))
        }
        let widening = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
        let status = conductor.columns(a: a.contents(), slice: slice, c: c.contents(), m: m, k: k, n: n,
                                       first: g, width: plan.amxColumns)
        commands.waitUntilCompleted()
        return (Attention.verdict(commands, what: "matmul2d float × int8 \(m)×\(k)×\(n)"), widening, status)
    }

    /// **The ANE's rows of an int8 product read as published**: those the plan gave (`planned.ane`, the last ones), if this process
    /// can run them — the ANE built, the weight the program's kind (scales per column or for the tensor, never Q8_0's blocks; no AMX
    /// columns), and a program that accepts its rotation's `group` — else 0, and the GPU takes the whole product. The GPU's rows may
    /// round their activation (its copy lies in the reserve's tail, past the integers and scales the ANE reads); the ANE's read the
    /// fp32 one. The plan's own runnability (`ProductAssignment.isRunnable`) is the hardware's; this is the process's.
    private func aneInt8Rows(_ planned: ProductAssignment, _ plan: Int8Plan, m: Int, k: Int, n: Int) -> Int {
        guard planned.path == .native, planned.ane > 0, plan.block == nil, plan.amxColumns == 0,
              let ane = gemm.neuralEngine, planned.ane <= m,
              ane.accepts(rows: planned.ane, k: k, n: n, int8Group: plan.group) else { return 0 }
        return planned.ane
    }

    /// **The int8 product cut between the GPU and the ANE**, at the same time: the first `m − aneRows` rows on the GPU's int8
    /// kernel (`int8Product`, the plan's rotation of its own rows in the reserve's tail), the last `aneRows` on the ANE's int8
    /// program, which reads the integers and their scales from the same reserve — nothing widened, nothing copied for it but
    /// the surface. Each row is one engine's. The ANE's rows refused or non-finite: the whole product again on the GPU alone.
    /// `gpu`: the GPU's seconds (its kernel's, and the whole product's if it had to be redone); `gpuWall`, `aneWall`: each side's
    /// wall from the start, as `GEMM.lastSides`; `computed`: the ANE's rows were its own.
    package static func int8Product(gemm: GEMM, plan: Int8Plan, reserve w: MTLBuffer, a: MTLBuffer, c: MTLBuffer, m: Int, k: Int, n: Int,
                                    ane: NeuralEngine, aneRows: Int) -> (gpu: Double, computed: Bool, gpuWall: Double, aneWall: Double) {
        let first = m - aneRows
        nonisolated(unsafe) let (pa, pc) = (a.contents().assumingMemoryBound(to: Float.self), c.contents().assumingMemoryBound(to: Float.self))
        let raw = w.contents()
        let weight = NeuralEngine.Int8Weight(values: raw.assumingMemoryBound(to: Int8.self),
                                             scales: (raw + plan.scalesOffset).assumingMemoryBound(to: Float.self),
                                             perColumn: plan.perColumn, group: plan.group)
        nonisolated(unsafe) var computed = false
        nonisolated(unsafe) var aneWall = 0.0
        let began = DispatchTime.now().uptimeNanoseconds
        let group = DispatchGroup()
        DispatchQueue.global(qos: .userInitiated).async(group: group) {
            computed = ane.product(a: pa, int8: weight, c: pc, first: first, rows: aneRows, k: k, n: n)
            aneWall = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
        }
        var gpu = first > 0 ? int8Product(gemm: gemm, plan: plan, reserve: w, a: a, c: c, m: first, k: k, n: n).gpu : 0
        let gpuWall = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
        group.wait()
        GEMM.neuralEngineTally.withLock { $0.handed += 1; if !computed { $0.givenBack += 1 } }
        if !computed { gpu += int8Product(gemm: gemm, plan: plan, reserve: w, a: a, c: c, m: m, k: k, n: n).gpu }
        return (gpu, computed, gpuWall, aneWall)
    }

    /// **The ANE's rows refused, computed on the CPU** — the stream's fallback (`GPUStream.neuralEngine`): the GPU already queued what
    /// reads them, so they are written before the ANE's signal, never left undefined. The operand rounded to bf16 as the GPU's route
    /// rounds it, the weight widened, sums in fp32 (BLAS): close to the GPU's rows, not their bits — a refusal is rare and counted.
    static func cpuRows(a: UnsafePointer<Float>, bf16 w: UnsafePointer<UInt16>, c: UnsafeMutablePointer<Float>,
                        first: Int, rows: Int, k: Int, n: Int) {
        var operand = [Float](repeating: 0, count: rows * k), weight = [Float](repeating: 0, count: n * k)
        for i in 0..<(rows * k) {
            let bits = a[first * k + i].bitPattern
            operand[i] = Float(bitPattern: (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000)
        }
        for i in 0..<(n * k) { weight[i] = Float(bitPattern: UInt32(w[i]) << 16) }
        // `w` is `[k, n]`, the forge's layout (`transposed`, the only one the ANE takes): C = A · W.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rows), Int32(n), Int32(k), 1, operand, Int32(k), weight, Int32(n),
                    0, c + first * n, Int32(n))
    }

    /// **The rounded bf16 product cut between the GPU and the ANE**, at the same time: the first `m − aneRows` rows on the GPU's
    /// kernel on the activation rounded to bf16 (`GEMM.linear(a:rounding:bf16:…)`, its rounded rows in the reserve's tail), the last
    /// `aneRows` on the ANE's fp16 program, which reads the same bf16 copy at the reserve's head (`NeuralEngine.product(a:bf16:…)`) and
    /// the fp32 activation. Each row is one engine's. The ANE's rows refused or non-finite: the whole product again on the GPU's rounded
    /// kernel. What the map measures for the pair (`gpu/bf16act+ane/fp16`) where the ANE is not streamed. The weight is `[k, n]`. Returns as
    /// `int8Product(…ane:)`.
    package static func bf16Product(gemm: GEMM, rounding: TensorGEMM.Rounding, reserve w: MTLBuffer, a: MTLBuffer, c: MTLBuffer,
                                    m: Int, k: Int, n: Int, ane: NeuralEngine, aneRows: Int)
        -> (gpu: Double, computed: Bool, gpuWall: Double, aneWall: Double) {
        let first = m - aneRows
        nonisolated(unsafe) let (pa, pc) = (a.contents().assumingMemoryBound(to: Float.self), c.contents().assumingMemoryBound(to: Float.self))
        nonisolated(unsafe) let weight = UnsafePointer(w.contents().assumingMemoryBound(to: UInt16.self))
        nonisolated(unsafe) var computed = false
        nonisolated(unsafe) var aneWall = 0.0
        let began = DispatchTime.now().uptimeNanoseconds
        let group = DispatchGroup()
        DispatchQueue.global(qos: .userInitiated).async(group: group) {
            computed = ane.product(a: pa, bf16: weight, c: pc, first: first, rows: aneRows, k: k, n: n)
            aneWall = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
        }
        var gpu = first > 0 ? gemm.linear(a: a, rounding: rounding, bf16: w, c: c, m: first, k: k, n: n, weightIsTransposed: true) : 0
        let gpuWall = Double(DispatchTime.now().uptimeNanoseconds - began) * 1e-9
        group.wait()
        GEMM.neuralEngineTally.withLock { $0.handed += 1; if !computed { $0.givenBack += 1 } }
        if !computed { gpu += gemm.linear(a: a, rounding: rounding, bf16: w, c: c, m: m, k: k, n: n, weightIsTransposed: true) }
        return (gpu, computed, gpuWall, aneWall)
    }

    /// **The same pair as the stream runs it** (`GPUStream.neuralEngine`, the render's in-place path): the ANE's rows on its thread,
    /// started by the GPU's signal, the GPU's rows queued beside them, one wait at the end (the stream's drain) instead of a rendezvous
    /// on the CPU. A refusal is computed on the CPU (`cpuRows`) before the signal, as in a render. `gpuWall`: when the GPU's rows
    /// completed (a marker buffer behind them); `aneWall`: when the ANE's thread returned; both from the call's start.
    package static func bf16ProductStreamed(gemm: GEMM, rounding: TensorGEMM.Rounding, reserve w: MTLBuffer, a: MTLBuffer, c: MTLBuffer,
                                            m: Int, k: Int, n: Int, ane: NeuralEngine, aneRows: Int)
        -> (gpu: Double, computed: Bool, gpuWall: Double, aneWall: Double) {
        let first = m - aneRows
        nonisolated(unsafe) let (pa, pc) = (a.contents().assumingMemoryBound(to: Float.self), c.contents().assumingMemoryBound(to: Float.self))
        nonisolated(unsafe) let weight = UnsafePointer(w.contents().assumingMemoryBound(to: UInt16.self))
        final class Ends: @unchecked Sendable {   // under `lock`
            let lock = NSLock(); var gpu: UInt64 = 0, ane: UInt64 = 0, computed = false
            func set(_ body: (Ends) -> Void) { lock.lock(); body(self); lock.unlock() }
        }
        let ends = Ends()
        let stream = gemm.queue.stream
        let wasOn = stream.isOn
        stream.isOn = true
        let began = DispatchTime.now().uptimeNanoseconds
        let done = stream.neuralEngine(on: gemm.queue, work: {
            let computed = ane.product(a: pa, bf16: weight, c: pc, first: first, rows: aneRows, k: k, n: n)
            ends.set { $0.ane = DispatchTime.now().uptimeNanoseconds; $0.computed = computed }
            return computed
        }, fallback: {
            cpuRows(a: pa, bf16: weight, c: pc, first: first, rows: aneRows, k: k, n: n)
        })
        if first > 0 { _ = gemm.linear(a: a, rounding: rounding, bf16: w, c: c, m: first, k: k, n: n, weightIsTransposed: true) }
        if let marker = gemm.queue.makeCommandBuffer() {
            marker.addCompletedHandler { _ in ends.set { $0.gpu = DispatchTime.now().uptimeNanoseconds } }
            marker.commit()
            stream.neuralEngineQueued(done)
            stream.drain()
            marker.waitUntilCompleted()
        } else { stream.neuralEngineQueued(done); stream.drain() }
        let gpu = stream.gpuSeconds().values.reduce(0, +)
        if !wasOn { stream.isOn = false }
        ends.lock.lock(); defer { ends.lock.unlock() }
        return (gpu, ends.computed, Double(ends.gpu &- began) * 1e-9, Double(ends.ane &- began) * 1e-9)
    }

    /// The weight into the reserve, by the path the plan assigned, and the assignment the widened product then runs under. A path
    /// this process cannot run (no `TensorGEMM`, no Q8_0 reading, no rounding kernels, a reserve too small for the rounding) is
    /// skipped, not refused: the plain form of the same weight, down to the widened one on the GPU — and the skip is named
    /// (`skipped`, into the render's ledger), never silent.
    /// The weight of the product being prepared, where it lies in the wrapped map (`prepare`'s in-place path), or `nil`.
    private var inPlace: (buffer: MTLBuffer, offset: Int)?
    /// ... the same for an int8 weight and its scale (`inPlaceInt8`).
    private var int8InPlace: Int8Source?

    /// The in-place path's condition (`prepare`), without its side effects: the weight where the map holds it and the rounded
    /// operand from the reserve's head.
    private func inPlaceRounding(_ name: String, k: Int, n: Int, planned: ProductAssignment)
        -> (map: MTLBuffer, offset: Int, rounding: Rounding)? {
        guard gemm.queue.stream.isOn, MemoryPlan.mapInPlace, planned.path == .bf16, planned.roundsActivation, roundsActivations,
              gemm.bf16Activations, artifact.tensors[name]?.dtype == .bfloat16,
              let offset = artifact.mappedOffset(name), offset % 16 == 0, let map = artifact.wrappedMap(on: gemm.device),
              let rounding = Self.roundingPlan(k: k, n: n, capacity: capacity, weightInReserve: false) else { return nil }
        return (map, offset, rounding)
    }

    /// The published int8 weight `name` as this process reads it — `[k, n]` (the forge's layout), its scale per column, for the
    /// tensor or by Q8_0's blocks, the kernel for it built —, where the plan reads it as published; else `nil`.
    private func readableInt8(_ name: String, k: Int, n: Int, planned: ProductAssignment) -> (Artifact.Tensor, Artifact.Scale)? {
        guard planned.path == .native, transposed, let tensor = artifact.tensors[name], tensor.dtype == .int8,
              let scale = tensor.scale, tensor.shape == [k, n],
              scale.block == nil ? scale.shape.count <= 1 : scale.shape.count == 2,
              gemm.tensor != nil, scale.block == nil ? gemm.tensorInt8 : gemm.tensorQ8 else { return nil }
        return (tensor, scale)
    }

    /// **The int8 plan the assignment asks for, its weight copied into the reserve**: W8A8, the rounded activation, the fp32 one
    /// beside the AMX's columns, the GPU alone — the first the reserve holds —, and the reason a route asked for is not the one
    /// taken; `nil` when none fits.
    private func plannedInt8(_ tensor: Artifact.Tensor, _ scale: Artifact.Scale, planned: ProductAssignment, k: Int, n: Int,
                             amx: Bool = true) -> (plan: Int8Plan, unavailable: ProductLedger.Reason?)? {
        let (perColumn, group) = (scale.shape.count == 1, tensor.rotation?.group)
        let roundsHere = planned.roundsActivation && roundsActivations && gemm.bf16Activations
            && (scale.block == nil || gemm.tensor?.roundsBlocks == true)
        // W8A8: the activation quantized where the plan says, this caller quantizes (the DiT's), the kernels are built and take
        // the row; else the integers on the fp32 activation, said in the record.
        let quantizesHere = planned.quantizesActivation && roundsActivations && gemm.int8Activations && scale.block == nil
            && gemm.tensor?.quantized?.simdgroups(k: k) != nil
        // The AMX's columns the plan gave: widened alone beside the GPU's. Rounded, the GPU alone: the AMX reads the fp32
        // activation, one rounding for some columns only.
        let amxColumns = gemm.conductor == nil || !amx ? 0 : planned.amxColumns
        guard let plan = (quantizesHere ? Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: perColumn, group: group,
                                                        amxColumns: 0, quantized: true) : nil)
                ?? (roundsHere ? Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: perColumn, group: group,
                                                     amxColumns: 0, block: scale.block, rounded: true) : nil)
                ?? Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: perColumn, group: group,
                                 amxColumns: roundsHere ? 0 : amxColumns, block: scale.block)
                ?? Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: perColumn, group: group, amxColumns: 0,
                                 block: scale.block)
        else { return nil }
        let unavailable: ProductLedger.Reason? = planned.quantizesActivation && plan.quantizedOffset == 0 ? .quantizingUnavailable
            : planned.roundsActivation && plan.roundedOffset == 0 ? .roundingUnavailable : nil
        return (plan, unavailable)
    }

    /// **The in-place path's condition for an int8 weight** (`prepare`), without its side effects: on the stream, where the render's
    /// plan charges the map, a weight whose copied plan (`plannedInt8`) rounds or quantizes its activation on the GPU alone — no AMX
    /// columns, no ANE rows (both read the reserve's copy) — and whose scale is fp32 per column or for the tensor (every Compact's: the
    /// copy would have taken it as it is; a bf16 or fp16 scale, or Q8_0's blocks, keep the copy that widens them). The fp32 route keeps
    /// it too: as the bf16 weight under no rounding, its product is waited for where it runs. Returns the same route's plan with the whole
    /// reserve to the activation (more rows a chunk: the same bits, the rows are independent), and where the integers and the scale lie
    /// in the wrapped map.
    private func inPlaceInt8(_ name: String, k: Int, n: Int, m: Int, planned: ProductAssignment)
        -> (plan: Int8Plan, source: Int8Source, unavailable: ProductLedger.Reason?)? {
        guard gemm.queue.stream.isOn, MemoryPlan.mapInPlace, let (tensor, scale) = readableInt8(name, k: k, n: n, planned: planned),
              scale.block == nil, scale.dtype == .float32,
              let (copied, unavailable) = plannedInt8(tensor, scale, planned: planned, k: k, n: n),
              copied.amxColumns == 0, copied.quantizedOffset > 0 || copied.roundedOffset > 0,
              aneInt8Rows(planned, copied, m: m, k: k, n: n) == 0,
              let offset = artifact.mappedOffset(name), offset % 16 == 0, (offset + scale.offset) % 16 == 0,
              let map = artifact.wrappedMap(on: gemm.device),
              let plan = Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: copied.perColumn, group: copied.group, amxColumns: 0,
                                       rounded: copied.roundedOffset > 0, quantized: copied.quantizedOffset > 0, weightInReserve: false),
              (plan.quantizedOffset > 0) == (copied.quantizedOffset > 0), (plan.roundedOffset > 0) == (copied.roundedOffset > 0)
        else { return nil }
        return (plan, Int8Source(buffer: map, offset: offset, scalesOffset: offset + scale.offset), unavailable)
    }

    /// **The W8A8 plan `prepare` gives `name`'s product**, without staging anything (`roundedOperand` asks it ahead): a published
    /// int8 weight scaled per column or for the tensor, the activation quantized where the plan says, this caller quantizes (the
    /// DiT's), the kernels are built and take the row. `nil`: another path.
    private func quantizedPlan(_ name: String, k: Int, n: Int, planned: ProductAssignment) -> Int8Plan? {
        guard planned.path == .native, planned.quantizesActivation, roundsActivations, gemm.int8Activations, transposed,
              let tensor = artifact.tensors[name], tensor.dtype == .int8 || tensor.dtype.sqBits != nil, let scale = tensor.scale,
              scale.block == nil, scale.shape.count <= 1, tensor.shape == [k, n], gemm.tensorInt8,
              gemm.tensor?.quantized?.simdgroups(k: k) != nil
        else { return nil }
        return Self.int8Plan(k: k, n: n, capacity: capacity, perColumn: scale.shape.count == 1, group: tensor.rotation?.group,
                             amxColumns: 0, quantized: true)
    }

    private func prepare(_ name: String, count: Int, m: Int, n: Int) throws -> (Path, ProductAssignment, skipped: ProductLedger.Reason?) {
        func expect(_ got: Int) throws {
            guard got == count else { throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected") }
        }
        let k = count / n
        let planned = gemm.assignment(k: k, n: n, m: m, weight: format(of: name))
        // **In place, on the stream**: a bf16 weight read where the map holds it (`Artifact.wrappedMap`), no copy into the reserve, so
        // nothing for the CPU to write while the queued products read — the product queues behind the others. Only the GPU's whole
        // product (no ANE rows, which read the copy), on a rounded activation. Only where the render's plan charges the map
        // (`MemoryPlan.mapInPlace`): read in place, its pages count in the footprint. An int8 weight likewise (`inPlaceInt8`).
        inPlace = nil
        int8InPlace = nil
        let stream = gemm.queue.stream
        if let (map, offset, rounding) = inPlaceRounding(name, k: k, n: n, planned: planned) {
            inPlace = (map, offset)
            return (.bf16(rounding), planned, nil)
        }
        // The same for a published int8 weight and its fp32 scale, on a rounded or quantized activation (`inPlaceInt8`): the GPU
        // writes the activation's rows into the reserve, after the queued work that reads it (the reserve is one tracked buffer).
        // Nothing is staged: an int8 operand's codes ahead stay whole for W8A8 (the product reads them or clears the slot, `linear`);
        // a rounded activation is written over them.
        if let (plan, source, unavailable) = inPlaceInt8(name, k: k, n: n, m: m, planned: planned) {
            int8InPlace = source
            if plan.quantizedOffset > 0 { reserveWrites &+= 1 } else { wroteReserve() }
            return (.int8(plan), planned, unavailable)
        }
        // An SQ weight: decoded into the reserve, then the int8 product (`prepareSQ`).
        if let tensor = artifact.tensors[name], tensor.dtype.sqBits != nil {
            return try prepareSQ(name, tensor, count: count, k: k, m: m, n: n, planned: planned)
        }
        // Anything else writes the reserve on the CPU: the queued work that reads it first. An int8 operand's codes survive an int8
        // weight staged below them (`quantizedSlot`).
        let slot = quantizedSlot
        wroteReserve()
        stream.drain()
        // How this product reads its weight (the plan's): the bf16 weight as published, its operand rounded to bf16 where the
        // plan says it gains, the 8-bit weight as published, or widened.
        if planned.path == .bf16, let tensor = artifact.tensors[name], tensor.dtype == .bfloat16, gemm.tensorBF16 || (planned.roundsActivation && gemm.bf16Activations) {
            var rounding: Rounding?
            if planned.roundsActivation && roundsActivations && gemm.bf16Activations {
                rounding = Self.roundingPlan(k: k, n: n, capacity: capacity)
            }
            // The rounding does not fit the reserve, or this caller does not round: the plain bf16 weight, if the process reads it.
            if (rounding != nil || gemm.tensorBF16),
               let got = try timed("staging", { try artifact.stage(name, into: UnsafeMutableRawPointer(reserve), capacity: 2 * capacity) }) {
                try expect(got)
                return (.bf16(rounding), planned, planned.roundsActivation && rounding == nil ? .roundingUnavailable : nil)
            }
        }
        if let (tensor, scale) = readableInt8(name, k: k, n: n, planned: planned) {
            guard let (plan, unavailable) = plannedInt8(tensor, scale, planned: planned, k: k, n: n)
            else { return (try widen(name, count: count), .gpuAlone(m), .int8Unavailable) }
            try expect(tensor.count)
            try timed("staging") { try artifact.stageInt8(name, into: UnsafeMutableRawPointer(reserve),
                                                          scalesAt: plan.scalesOffset) }
            // The staging and the AMX's columns write below `rotatedOffset`: an int8 operand's codes past it are whole.
            if let slot { quantizedSlot = (slot.generation, max(slot.reach, plan.rotatedOffset), slot.filled) }
            return (.int8(plan), planned, unavailable)
        }
        if planned.path == .native, planned.roundsActivation, let tensor = artifact.tensors[name], tensor.dtype.isPacked {
            // A K-quant's blocks as published, where this process reads their type (Q4_K, Q5_K) on a rounded activation and the
            // reserve holds the layout; else the plan's answer without the route (a Q6_K beside Q4_K in one class keeps its widened
            // route, the ANE's rows included), named in the render's record.
            if transposed, KQuantKernels.reads(tensor.dtype), tensor.shape == [k, n], roundsActivations, gemm.bf16Activations,
               gemm.tensor?.readsKQuants == true,
               let layout = TensorGEMM.kquantLayout(k: k, n: n, capacity: capacity * 4, five: tensor.dtype == .q5_k) {
                _ = try timed("staging") { try artifact.stagePacked(name, into: UnsafeMutableRawPointer(reserve), capacity: layout.integers) }
                return (.kquant(layout), planned, nil)
            }
            let widened = gemm.plan.assignWidened(ProductKey(k: k, n: n, m: m, weight: .q4k, phase: gemm.phase))
            return (try widen(name, count: count), widened, .kquantUnread)
        }
        let skipped: ProductLedger.Reason? = switch planned.path {
        case .widened: nil
        case .bf16: .bf16Unavailable
        case .native: .int8Unavailable
        }
        return (try widen(name, count: count), planned.path == .widened ? planned : .gpuAlone(m), skipped)
    }

    /// **An SQ weight's product** (`SQ`): its integers and scales decoded where the Compact's int8 copy lies in the reserve
    /// (`Int8Plan`, the weight in the reserve), then the Compact's int8 product on them — `float × int8` or W8A8, as the plan asks —,
    /// the GPU's whole (no AMX columns, no ANE rows: both would read the map's bytes as integers). The decoding runs on the GPU from
    /// the wrapped map, queued behind the work in flight (the reserve is one tracked buffer), where the render's plan charges the map
    /// (`MemoryPlan.mapInPlace`); else on the CPU, after a drain, from the map's pages (`Artifact.stageSQ`): the same bytes either
    /// way. The product reads the reserve as a weight read in place (`int8InPlace`): queued on the stream. A plan that does not give
    /// the int8 route, or a reserve that does not hold it: widened on the CPU.
    private func prepareSQ(_ name: String, _ tensor: Artifact.Tensor, count: Int, k: Int, m: Int, n: Int,
                           planned: ProductAssignment) throws -> (Path, ProductAssignment, skipped: ProductLedger.Reason?) {
        let stream = gemm.queue.stream
        guard planned.path == .native, transposed, let bits = tensor.dtype.sqBits, let scale = tensor.scale, scale.dtype == .float32,
              scale.shape == [n], tensor.shape == [k, n], gemm.tensor != nil, gemm.tensorInt8,
              let (plan, unavailable) = plannedInt8(tensor, scale, planned: planned, k: k, n: n, amx: false), plan.amxColumns == 0
        else {
            wroteReserve()
            stream.drain()
            return (try widen(name, count: count), planned.path == .widened ? planned : .gpuAlone(m),
                    planned.path == .native ? .int8Unavailable : nil)
        }
        let slot = quantizedSlot
        wroteReserve()
        let w = try gemm.wrap(UnsafeMutableRawPointer(reserve), bytes: capacity * 4, name: "w")
        if MemoryPlan.mapInPlace, let kernels = SQKernels.of(gemm.device), let offset = artifact.mappedOffset(name), offset % 16 == 0,
           let map = artifact.wrappedMap(on: gemm.device) {
            let commands = gemm.queue.checkedCommandBuffer()
            kernels.encode(commands, source: map, offset: offset, scalesOffset: offset + scale.offset, into: w,
                           scalesAt: plan.scalesOffset, k: k, n: n, bits: bits)
            commands.commit()
            SQKernels.decodes.withLock { $0 += 1 }
            timings["gemm GPU", default: 0] += stream.settle(commands, label: "sq decode") { "SQ\(bits) decoding \(k)×\(n)" }
        } else {
            stream.drain()
            try timed("staging") { try artifact.stageSQ(name, into: UnsafeMutableRawPointer(reserve), scalesAt: plan.scalesOffset) }
        }
        // The decoding writes below `rotatedOffset`: an int8 operand's codes past it are whole, as after a staging.
        if let slot { quantizedSlot = (slot.generation, max(slot.reach, plan.rotatedOffset), slot.filled) }
        int8InPlace = Int8Source(buffer: w, offset: 0, scalesOffset: plan.scalesOffset)
        return (.int8(plan), planned, unavailable)
    }

    /// The weight widened to fp32 into the reserve, on the CPU.
    private func widen(_ name: String, count: Int) throws -> Path {
        let got = try timed("widening") { try artifact.materialize(name, into: reserve, capacity: capacity) }
        guard got == count else { throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected") }
        return .widened
    }
}

/// **The LoRA stack of one module, expanded**: `down [k, r]` and `up [r, n]`, or `nil` when no LoRA of
/// the stack touches it. `CacheLoRA` keeps them from one evaluation to the next, `StreamedLoRA` streams
/// Qwen-Image-2.1's by layer, `ExpandedLoRA` expands them at each call into two fixed slices.
package protocol LoRAStacks: AnyObject {
    func module(_ target: String, k: Int, n: Int) throws
        -> (down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>, rank: Int)?
}

extension CacheLoRA: LoRAStacks {}
extension StreamedLoRA: LoRAStacks {}

/// The stack expanded afresh at every call into `down` and `up` (sized for the largest module), as
/// Anima and Krea 2's text fusion do: no cache to charge, the expansion paid at each product.
package final class ExpandedLoRA: LoRAStacks {
    private let lora: LoRA
    private let down, up: UnsafeMutablePointer<Float>
    package init(lora: LoRA, down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>) {
        self.lora = lora; self.down = down; self.up = up
    }
    package func module(_ target: String, k: Int, n: Int) throws
        -> (down: UnsafeMutablePointer<Float>, up: UnsafeMutablePointer<Float>, rank: Int)? {
        guard lora.rank(target) > 0 else { return nil }
        return (down, up, try lora.materialize(target, k: k, n: n, down: down, up: up))
    }
}
