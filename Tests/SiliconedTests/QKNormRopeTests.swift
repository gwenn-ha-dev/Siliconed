import Foundation
import Metal
import Testing
@testable import Siliconed

/// **The QK-norm and RoPE before a self-attention, on either side** (`QKNormRope`, `Ops.rmsNormRope`): the CPU's one pass gives the bits of
/// the two passes it replaces (Z-Image, Krea 2, MOSS-Image ran those), the GPU's kernel is the CPU's within fp32 channel by channel (its
/// rotation to the bit), at the families' shapes; the map's lines, the solver's choice, the plan's forcing. Random tensors, no weights.
@Suite struct QKNormRopeTests {
    typealias M = CapacityModel

    /// A family's shape: rows, Q's heads, K's heads, the layout of its table, its ε.
    struct Shape: CustomStringConvertible {
        var name: String, rows: Int, heads: Int, kvHeads: Int, layout: QKNormRope.Layout, eps: Float
        var description: String { name }
    }
    static let shapes: [Shape] = [
        .init(name: "z-image", rows: 77, heads: 30, kvHeads: 30, layout: .interleaved, eps: 1e-5),
        .init(name: "qwen", rows: 64, heads: 32, kvHeads: 32, layout: .interleaved, eps: 1e-6),
        .init(name: "anima", rows: 50, heads: 16, kvHeads: 16, layout: .interleaved, eps: 1e-5),
        .init(name: "krea2", rows: 45, heads: 48, kvHeads: 12, layout: .interleaved, eps: 1e-6),
        .init(name: "moss", rows: 33, heads: 24, kvHeads: 24, layout: .halfSplit, eps: 1e-6),
    ]
    static let headDim = 128

    func random(_ count: Int, seed: UInt64, scale: Float) -> [Float] {
        var state = seed &* 0x9E3779B97F4A7C15 | 1
        return (0..<count).map { _ in
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return (Float(state >> 40) / Float(1 << 24) * 2 - 1) * scale
        }
    }
    /// A table of `[rows, Dh]` the way each family holds it: `(cos, sin)` pairs, or `[sin | cos]`.
    func table(rows: Int, layout: QKNormRope.Layout) -> [Float] {
        let angles = random(rows * Self.headDim / 2, seed: 99, scale: 40)
        var t = [Float](repeating: 0, count: rows * Self.headDim)
        let half = Self.headDim / 2
        for r in 0..<rows {
            for j in 0..<half {
                let a = angles[r * half + j]
                if layout == .interleaved { t[r * Self.headDim + 2 * j] = cosf(a); t[r * Self.headDim + 2 * j + 1] = sinf(a) }
                else { t[r * Self.headDim + j] = sinf(a); t[r * Self.headDim + half + j] = cosf(a) }
            }
        }
        return t
    }

    /// `Ops.rmsNorm` in place, as the families call it.
    func norm(_ x: inout [Float], _ w: [Float], rows: Int, eps: Float) {
        x.withUnsafeMutableBufferPointer { Ops.rmsNorm($0.baseAddress!, weight: w, into: $0.baseAddress!, rows: rows, columns: Self.headDim, eps: eps) }
    }

    @Test(arguments: shapes) func theCPUsOnePassIsTheTwoPassesToTheBit(_ shape: Shape) {
        let d = Self.headDim
        for (heads, seed) in [(shape.heads, UInt64(1)), (shape.kvHeads, 2)] {
            let x = random(shape.rows * heads * d, seed: seed, scale: 30)
            let w = random(d, seed: 3, scale: 2)
            let f = table(rows: shape.rows, layout: shape.layout)
            var two = x, one = x
            norm(&two, w, rows: shape.rows * heads, eps: shape.eps)
            if shape.layout == .halfSplit { MossImageRope.apply(&two, freqs: f, sequence: shape.rows, heads: heads, headDim: d) }
            else { Ops.rope(&two, freqs: f, sequence: shape.rows, heads: heads, headDim: d) }
            Ops.rmsNormRope(&one, weight: w, freqs: f, sequence: shape.rows, heads: heads, headDim: d, eps: shape.eps,
                            halfSplit: shape.layout == .halfSplit)
            #expect(one.map(\.bitPattern) == two.map(\.bitPattern), "\(shape) heads \(heads)")
        }
    }

    /// Q and K wrapped in page-aligned pages, the table read from a row that is not the first (Qwen's `first`, Z-Image's caption block).
    @Test(arguments: shapes) func theGPUsKernelIsTheCPUsPassWithinFP32ItsRotationToTheBit(_ shape: Shape) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return }
        let gpu = try QKNormRope(device: device, queue: queue)
        let d = Self.headDim, skipped = 5
        let f = table(rows: shape.rows + skipped, layout: shape.layout)
        let table = Pages(bytes: f.count * 4)
        table.floats().update(from: f, count: f.count)
        let tableBuffer = try #require(device.makeBuffer(bytesNoCopy: table.base, length: table.capacity, options: .storageModeShared, deallocator: nil))
        for (heads, seed) in [(shape.heads, UInt64(1)), (shape.kvHeads, 2)] {
            let n = shape.rows * heads * d
            let x = random(n, seed: seed, scale: 30)
            let w = random(d, seed: 3, scale: 2)
            let pages = Pages(bytes: n * 4)
            let buffer = try #require(device.makeBuffer(bytesNoCopy: pages.base, length: pages.capacity, options: .storageModeShared, deallocator: nil))
            let freqs = (tableBuffer, skipped * d * 4)
            let offset = Array(f[(skipped * d)...])
            // norm + rope against the CPU's pass
            pages.floats().update(from: x, count: n)
            gpu.run([.init(buffer, heads: heads, weight: w)], rows: shape.rows, headDim: d, eps: shape.eps, freqs: freqs, layout: shape.layout)
            var cpu = x
            Ops.rmsNormRope(&cpu, weight: w, freqs: offset, sequence: shape.rows, heads: heads, headDim: d, eps: shape.eps,
                            halfSplit: shape.layout == .halfSplit)
            let got = Array(UnsafeBufferPointer(start: pages.floats(), count: n))
            let e = cpu.withUnsafeBufferPointer { channelError(got, $0, columns: d) }
            #expect(e.worst < 1e-6, "\(shape) heads \(heads): \(e.worst) at channel \(e.channel)")
            // the rotation alone, to the bit: the same products and sums, no contraction
            pages.floats().update(from: x, count: n)
            gpu.run([.init(buffer, heads: heads, weight: nil)], rows: shape.rows, headDim: d, eps: shape.eps, freqs: freqs, layout: shape.layout,
                    wait: false)
            var rotated = x
            if shape.layout == .halfSplit { MossImageRope.apply(&rotated, freqs: offset, sequence: shape.rows, heads: heads, headDim: d) }
            else { Ops.rope(&rotated, freqs: offset, sequence: shape.rows, heads: heads, headDim: d) }
            // the queued pass is waited for by the next command buffer of the queue
            queue.makeCommandBuffer().map { $0.commit(); $0.waitUntilCompleted() }
            #expect(Array(UnsafeBufferPointer(start: pages.floats(), count: n)).map(\.bitPattern) == rotated.map(\.bitPattern), "\(shape) rope")
            // the norm alone: the CPU's `rmsNorm` within fp32
            pages.floats().update(from: x, count: n)
            gpu.run([.init(buffer, heads: heads, weight: w)], rows: shape.rows, headDim: d, eps: shape.eps, freqs: nil, layout: shape.layout)
            var normed = x
            norm(&normed, w, rows: shape.rows * heads, eps: shape.eps)
            let gotNormed = Array(UnsafeBufferPointer(start: pages.floats(), count: n))
            #expect(normed.withUnsafeBufferPointer { channelError(gotNormed, $0, columns: d).worst } < 1e-6, "\(shape) norm")
        }
    }

    /// **The pass writing Q and K rounded into the attention's copies** (`Operand.rounded`, `TiledAttention.operands`) gives the
    /// attention the operands its own conversion would: the same output to the bit, fp32 Q and K left unwritten and unread.
    @Test(arguments: [TiledAttention.Precision.fp16, .bf16]) func theRoundedCopyIsTheAttentionsOwnConversionToTheBit(_ precision: TiledAttention.Precision) throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.metal4), let queue = device.makeCommandQueue() else { return }
        let gpu = try QKNormRope(device: device, queue: queue)
        let d = Self.headDim, rows = 77, heads = 4, kvHeads = 2
        let f = table(rows: rows, layout: .interleaved)
        func buffer(_ values: [Float]) throws -> MTLBuffer {
            try #require(device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared))
        }
        let x = random(rows * heads * d, seed: 1, scale: 30), y = random(rows * kvHeads * d, seed: 2, scale: 30)
        let w = random(d, seed: 3, scale: 2), table = try buffer(f), v = try buffer(random(rows * kvHeads * d, seed: 4, scale: 50))
        func attention(rounded: Bool) throws -> [UInt32] {
            let q = try buffer(x), k = try buffer(y), out = try buffer([Float](repeating: 0, count: rows * heads * d))
            let tiled = try TiledAttention(device: device, queue: queue, heads: heads, sequence: rows, headDim: d, kvHeads: kvHeads,
                                           precision: precision, tile: AttentionTile.candidates[0])
            let copies = rounded ? try #require(tiled.operands) : nil
            gpu.run([.init(q, heads: heads, weight: w, rounded: copies.map { ($0.q, precision) }, keepsFP32: false),
                     .init(k, heads: kvHeads, weight: w, rounded: copies.map { ($0.k, precision) }, keepsFP32: false)],
                    rows: rows, headDim: d, eps: 1e-5, freqs: (table, 0), layout: .interleaved)
            if rounded {
                #expect(Array(UnsafeBufferPointer(start: q.contents().assumingMemoryBound(to: Float.self), count: x.count)) == x,
                        "the fp32 Q left as it was")
            }
            _ = tiled.run(q: q, k: k, v: v, into: out, prerounded: rounded ? [.q, .k] : [])
            return Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: UInt32.self), count: rows * heads * d))
        }
        #expect(try attention(rounded: true) == attention(rounded: false), "\(precision)")
    }

    /// **The attention writing its output rounded for the next product** (`TiledAttention.run(…rounded:)`): the slot holds `round_bf16`
    /// of the fp32 output to the bit, at the slot's offset and row stride, the rows past the last tile left untouched; the fp32 output is
    /// the one written without it. The fused form and the three stages, each operand type, a sequence no multiple of a tile.
    @Test(arguments: [(TiledAttention.Precision.fp32, false), (.fp16, false), (.fp16, true), (.bf16, false), (.bf16, true)])
    func theOutputRoundedIsTheProductsOwnRoundingToTheBit(_ form: (precision: TiledAttention.Precision, fused: Bool)) throws {
        guard let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.metal4), let queue = device.makeCommandQueue() else { return }
        let d = Self.headDim, rows = 77, heads = 4, kvHeads = 2, width = heads * d, offset = 256
        func buffer(_ values: [Float]) throws -> MTLBuffer {
            try #require(device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared))
        }
        let x = random(rows * width, seed: 5, scale: 3), y = random(rows * kvHeads * d, seed: 6, scale: 3)
        let z = random(rows * kvHeads * d, seed: 7, scale: 50)
        let tiled = try TiledAttention(device: device, queue: queue, heads: heads, sequence: rows, headDim: d, kvHeads: kvHeads,
                                       precision: form.precision, tile: form.fused ? AttentionTile.candidates[0] : nil)
        #expect(tiled.writesRounded)
        func attention(rounded slot: MTLBuffer?) throws -> [UInt32] {
            let out = try buffer([Float](repeating: 0, count: rows * width))
            _ = tiled.run(q: try buffer(x), k: try buffer(y), v: try buffer(z), into: out, rounded: slot.map { ($0, offset) })
            return Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: UInt32.self), count: rows * width))
        }
        let sentinel: UInt16 = 0xABCD, slotCount = offset / 2 + rows * width + 64
        let slot = try #require(device.makeBuffer(bytes: [UInt16](repeating: sentinel, count: slotCount), length: slotCount * 2,
                                                  options: .storageModeShared))
        let withSlot = try attention(rounded: slot), without = try attention(rounded: nil)
        #expect(withSlot == without, "the fp32 output as without the slot")
        let written = Array(UnsafeBufferPointer(start: slot.contents().assumingMemoryBound(to: UInt16.self), count: slotCount))
        let expected = withSlot.map { u -> UInt16 in        // `BFloatKernels.roundingSource`, as `round_bf16` writes it
            (u & 0x7fff_ffff) > 0x7f80_0000 ? UInt16((u >> 16) | 0x40) : UInt16(truncatingIfNeeded: (u &+ 0x7fff &+ ((u >> 16) & 1)) >> 16)
        }
        #expect(Array(written[(offset / 2)..<(offset / 2 + rows * width)]) == expected, "\(form)")
        #expect(written[..<(offset / 2)].allSatisfy { $0 == sentinel } && written[(offset / 2 + rows * width)...].allSatisfy { $0 == sentinel },
                "nothing written outside the slot")
    }

    // MARK: - The map, the solver, the plan

    func model(cpuRate: Double = 4e9, gpuRate: Double = 40e9, wake: Double? = 0.00085) -> M {
        var m = M()
        m.lines[M.Kernel(.gpu, .fp32)] = .init(t0: 0, rate: 3e12)
        m.lines[M.Kernel(.gpu, .graph)] = .init(t0: 0, rate: 1.5e12)
        m.wake = wake
        m.normRope = .init(cpu: .init(t0: 2e-5, rate: cpuRate), gpu: .init(t0: 1e-5, rate: gpuRate), gpuWaited: .init(t0: 2e-4, rate: gpuRate))
        return m
    }
    let constraints = CapacitySolver.Constraints(sustained: false, allowsANE: false)
    let own = AttentionKey(sequence: 4096, heads: 24, headDim: 128)

    @Test func theSolverTakesTheGPUsPassWhereTheCallPaysLessAndKeepsItsKernel() throws {
        let m = model()
        let s = try #require(CapacitySolver.attention(own, flash: false, model: m, constraints: constraints))
        #expect(s.assignment.normRopeOnGPU)
        var without = m; without.normRope = nil
        let before = try #require(CapacitySolver.attention(own, flash: false, model: without, constraints: constraints))
        #expect(s.assignment.normRope(onGPU: false) == before.assignment, "the pass changes no kernel")
        // the GPU's own seconds, no wake: the call's line plus the queued kernel
        let values = 4096.0 * 48 * 128
        #expect(abs(s.compute - (before.compute - 0.00085 + 1e-5 + values / 40e9)) < 1e-12)
        #expect(!before.assignment.normRopeOnGPU, "a map without the probe: the CPU's pass, as before")
    }

    @Test func theCPUsPassStaysWithinTheNoiseAndWithoutTheWake() throws {
        // no wake, a GPU as fast as the CPU: nothing past the noise, the CPU's pass (the written order)
        var m = model(cpuRate: 40e9, gpuRate: 40e9, wake: nil)
        m.normRope!.gpu.t0 = 2e-5
        m.noise = 0.02
        let s = try #require(CapacitySolver.attention(own, flash: false, model: m, constraints: constraints))
        #expect(!s.assignment.normRopeOnGPU)
        // the CPU's pass is now priced: its seconds in the call
        var without = m; without.normRope = nil
        let before = try #require(CapacitySolver.attention(own, flash: false, model: without, constraints: constraints))
        #expect(abs(s.compute - (before.compute + 2e-5 + 4096.0 * 48 * 128 / 40e9)) < 1e-12)
    }

    @Test func aCrossAttentionHasNoPassAndACutWaitsForIt() throws {
        let m = model()
        let cross = AttentionKey(sequence: 4096, keys: 512, heads: 24, headDim: 128, cross: true)
        let c = try #require(CapacitySolver.attention(cross, flash: false, model: m, constraints: constraints))
        #expect(!c.assignment.normRopeOnGPU)
        var without = m; without.normRope = nil
        #expect(c.compute == CapacitySolver.attention(cross, flash: false, model: without, constraints: constraints)?.compute)
        // a cut by heads pays the waited kernel's wall
        var split = m
        split.lines[M.Kernel(.cpu, .heads)] = .init(t0: 0, rate: 0.5e12)
        split.sharing[M.key([.gpu, .cpu])] = .init()
        let a = AttentionAssignment(kernel: .split, cpuHeads: 8, normRopeOnGPU: true)
        let queued = AttentionAssignment(kernel: .split, cpuHeads: 8)
        let t = try #require(CapacitySolver.attention(assigned: a, key: own, model: split, constraints: constraints))
        let u = try #require(CapacitySolver.attention(assigned: queued, key: own, model: split, constraints: constraints))
        let values = 4096.0 * 48 * 128
        // The GPU's route pays no wake and the waited kernel's wall: the cut's own time without a wake, the CPU's pass swapped for it.
        var calm = split; calm.wake = nil
        let v = try #require(CapacitySolver.attention(assigned: queued, key: own, model: calm, constraints: constraints))
        #expect(abs(t.compute - (v.compute - (2e-5 + values / 4e9) + 2e-4 + values / 40e9)) < 1e-9)
        // The CPU's pass leaves the GPU idle: under the cut, the wake moves only the GPU's side's end, the call's where that side is
        // the last (`CapacitySolver.timed`) — never earlier.
        #expect(u.compute >= v.compute)
    }

    @Test func theMapKeepsItsLinesAndAnOldMapReadsWithout() throws {
        let m = model()
        let back = try JSONDecoder().decode(M.self, from: JSONEncoder().encode(m))
        #expect(back.normRope == m.normRope)
        var none = m; none.normRope = nil
        #expect(try JSONDecoder().decode(M.self, from: JSONEncoder().encode(none)).normRope == nil)
        // the plan's fingerprint moves with the lines, and not without them
        #expect(CapacityPlan(model: m).identity != CapacityPlan(model: none).identity)
        var other = none; other.wake = 0.00085
        #expect(CapacityPlan(model: other).identity == CapacityPlan(model: none).identity)
    }

    @Test func theFitReadsThreeLinesAndARefusedGPUGivesNone() throws {
        typealias S = CapacityFit.Measurements.Solo
        let me = CapacityFit.Measurements.self
        let solos = [1024, 4096].flatMap { s -> [S] in
            let v = Double(2 * s * 24 * 128)
            return [S(kernel: me.normRopeCPU, shape: [s, 24, 128], flop: v, seconds: 2e-5 + v / 4e9),
                    S(kernel: me.normRopeGPU, shape: [s, 24, 128], flop: v, seconds: 1e-5 + v / 40e9),
                    S(kernel: me.normRopeGPUWaited, shape: [s, 24, 128], flop: v, seconds: 2e-4 + v / 40e9)]
        }
        let n = try #require(CapacityFit.normRope(solos))
        #expect(abs(n.cpu.rate / 4e9 - 1) < 1e-9 && abs(n.gpu.t0 - 1e-5) < 1e-12 && abs(n.gpuWaited.t0 - 2e-4) < 1e-12)
        #expect(CapacityFit.normRope(solos, wrong: [me.normRopeGPU]) == nil)
        #expect(CapacityFit.normRope(nil) == nil)
    }

    @Test func anAssignmentWrittenBeforeReadsAsTheCPUsPassAndTheForcingPutsItOnTheGPU() throws {
        let old = #"{"kernel":"split","cpuHeads":6,"gpuKernel":"graph"}"#
        let a = try JSONDecoder().decode(AttentionAssignment.self, from: Data(old.utf8))
        #expect(a == AttentionAssignment(kernel: .split, cpuHeads: 6) && !a.normRopeOnGPU)
        let g = AttentionAssignment.tiles16.normRope(onGPU: true)
        #expect(try JSONDecoder().decode(AttentionAssignment.self, from: JSONEncoder().encode(g)) == g)
        let plan = try #require(CapacityPlan.forcing(named: "normrope"))
        #expect(plan.assign(own).normRopeOnGPU)
        #expect(plan.assign(own).normRope(onGPU: false) == .defaults(own))
        #expect(!plan.assign(AttentionKey(sequence: 4096, keys: 512, heads: 24, headDim: 128, cross: true)).normRopeOnGPU)
        let tiles = try #require(CapacityPlan.forcing(named: "tiles16,normrope"))
        #expect(tiles.assign(own) == AttentionAssignment.tiles16.normRope(onGPU: true))
        #expect(!CapacityPlan.defaults.assign(own).normRopeOnGPU)
    }

    /// The plan prices the pass with its attention: the render's class measures it there too, with no route and no execution of its own.
    @Test func theRendersClassMeasuresThePassWithItsAttentionAndNoRoute() {
        let work = ProductLedger.Work.attention(sequence: 4096, heads: 30, headDim: 128)
        let predicted = RenderLoop.Predicted(work: work, route: "flash", executions: 10, seconds: 1.1)
        let tallies = [ProductLedger.Tally(work: work, route: "flash", executions: 10, seconds: 1.0),
                       ProductLedger.Tally(work: work, route: ProductLedger.passRoute, executions: 10, seconds: 0.1)]
        let row = try! #require(RenderLoop.compare(tallies, predicted: [predicted]).rows.first)
        #expect(row.routes == ["flash"] && row.executions == 10 && abs(row.measured - 1.1) < 1e-12 && row.routeAgrees)
    }
}
