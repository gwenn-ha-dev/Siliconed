import Foundation
import Metal

// MARK: - The GPU's wake: what a call pays after the CPU left the GPU idle

/// **The attention as a render calls it: after the CPU's own pass on its operands.** Every self-attention of the five denoisers follows the
/// CPU's QK-norm and RoPE on S·H·Dh values (`Ops.rope`, `Ops.rmsNormRope`), the GPU idle meanwhile; the probes time their kernels back
/// to back, the GPU never idle. On one M5 Pro the difference is a step, not a slope: with the GPU kept busy before (the render's duty,
/// the products holding it), an idle of 1 ms costs the next call nothing and one of 2 ms or more about 0.85 ms, at S = 1 024 as at 4 096 —
/// the clock the GPU's governor lowered and raises again. A fixed cost per call, so a constant the attention's line does not hold (its t₀ is
/// read on calls back to back).
///
///     per idle g ∈ `idles`:  rounds × [ the tier's longest attention (busy) · g of idle · the tier's shortest, timed ]  → its median
///     wake = the median of the two longest idles' medians − the median at g = 0           (≥ 0)
///     idle = the shortest g whose median pays half the wake or more                         (what the report shows; the solver reads `wake`)
///
/// Measured where the half-precision tiled kernel runs, the kernel every family's attention takes on a GPU with tensor units; the wake is
/// the GPU's, not the kernel's. What it costs the map: about half a second.
extension Diagnostic.CapacityProbes.Context {
    /// The idles, in seconds, from none to well past any governor's step seen (2 ms on an M5 Pro).
    static let wakeIdles: [Double] = [0, 0.0005, 0.001, 0.002, 0.004, 0.008]
    static let wakeRounds = 5

    func wake(_ measurements: inout Fit.Measurements, busy: TiledAttention, timed: TiledAttention, sequence: Int,
              buffers: (q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, out: MTLBuffer)) {
        let run = { (a: TiledAttention) in _ = a.run(q: buffers.q, k: buffers.k, v: buffers.v, into: buffers.out) }
        run(busy); run(timed)
        var solos: [Fit.Measurements.Solo] = []
        for idle in Self.wakeIdles {
            var times: [Double] = []
            for _ in 0..<Self.wakeRounds {
                run(busy)
                if idle > 0 { usleep(useconds_t(idle * 1e6)) }
                let start = CapacityProbes.now()
                run(timed)
                times.append(CapacityProbes.now() - start)
            }
            let t = Fit.median(times)
            solos.append(.init(kernel: Kernel(.gpu, .tiles16).description, shape: [sequence, Sizes.attentionHeads, Sizes.attentionHeadDim,
                                                                                   Int((idle * 1e6).rounded())], flop: 0, seconds: t))
        }
        measurements.wake = solos
        if let w = Fit.wake(solos) {
            log?(String(format: "  gpu wake  %.2f ms after %.1f ms of idle", w.seconds * 1e3, w.idle * 1e3))
        }
    }
}

extension CapacityFit {
    /// **The wake from its probe** (`Context.wake`): the plateau's median less the call back to back, never below 0, and the first idle
    /// that pays half of it. `nil` without the probe, or without a call back to back and two idles to read the plateau on.
    package static func wake(_ solos: [Measurements.Solo]?) -> (seconds: Double, idle: Double)? {
        guard let solos, let warm = solos.first(where: { $0.shape.last == 0 })?.seconds else { return nil }
        let idled = solos.filter { ($0.shape.last ?? 0) > 0 }.sorted { $0.shape.last! < $1.shape.last! }
        guard idled.count >= 2 else { return nil }
        let plateau = median(idled.suffix(2).map(\.seconds)) - warm
        let w = max(0, plateau)
        let idle = idled.first { $0.seconds - warm >= w / 2 }.map { Double($0.shape.last!) * 1e-6 } ?? 0
        return (w, w > 0 ? idle : 0)
    }
}

// MARK: - The QK-norm and RoPE, on either side

/// **The pass before a self-attention, timed on the CPU and on the GPU** — what the solver weighs (`CapacitySolver`: the CPU's pass and
/// the wake its idle costs, against the GPU's kernel chained to the attention). At each of the tier's attention sequences, on the probe's
/// own Q and K (`[S, 24, 128]` each, the values `2 · S · 24 · 128`):
///
///     cpu/normrope        `Ops.rmsNormRope` on Q then K, its wall (a steady figure: the CPU's walls are not bimodal)
///     gpu/normrope        `QKNormRope` on both in one command buffer, the GPU's own seconds (what the chain adds: no completion paid)
///     gpu/normrope-wait   the same, its wall, waited for (what a cut by heads pays) — the **median of every call**, not the fast side:
///                         a completion lands in one of two modes, and a render waiting once per attention pays the mix
///
/// **Each GPU call right after a long attention** (`busy`, the tier's longest), as a render's follows the products that wrote Q and K: back
/// to back with nothing before, the governor clocks the GPU down between calls, and one M5 Pro read the kernel at 0.58 ms on S = 1 024
/// where the CPU took 0.42 — a GPU no render runs at.
///
/// Judged first at the shortest sequence: the GPU's output against the CPU's, channel by channel (`fp32Threshold`); refused, neither GPU
/// line is kept. What it costs the map: about a quarter of a second. What it removes: the CPU's pass the plan never priced (a render paid
/// it before every self-attention, its prediction did not), and a route no map could offer.
extension Diagnostic.CapacityProbes.Context {
    static let normRopeCalls = 9

    func normRope(_ measurements: inout Fit.Measurements, sequences: [Int], q: Pages, k: Pages, buffers: (q: MTLBuffer, k: MTLBuffer),
                  busy: (() -> Void)?) {
        let (h, d) = (Sizes.attentionHeads, Sizes.attentionHeadDim)
        guard let longest = sequences.max(), let shortest = sequences.min(),
              let kernel = QKNormRope.shared(device: bench.device, queue: bench.queue) else { return }
        let table = Pages(bytes: longest * d * 4)
        Diagnostic.fill(table.floats(), count: longest * d, scale: 1, seed: 14)
        guard let freqs = try? bench.buffer(table) else { return }
        let weight = (0..<d).map { 0.75 + Float($0 % 7) * 0.05 }
        let eps: Float = 1e-6
        nonisolated(unsafe) let (qp, kp, fp) = (q.floats(), k.floats(), table.floats())
        func cpu(_ s: Int) {
            weight.withUnsafeBufferPointer { w in
                Ops.rmsNormRope(qp, weight: w.baseAddress!, freqs: fp, sequence: s, heads: h, headDim: d, eps: eps)
                Ops.rmsNormRope(kp, weight: w.baseAddress!, freqs: fp, sequence: s, heads: h, headDim: d, eps: eps)
            }
        }
        func gpu(_ s: Int) -> Double {
            kernel.run([.init(buffers.q, heads: h, weight: weight), .init(buffers.k, heads: h, weight: weight)],
                       rows: s, headDim: d, eps: eps, freqs: (freqs, 0), layout: .interleaved)
        }
        // The judge: the CPU's pass on a copy of Q, the GPU's on Q itself.
        let n = shortest * h * d
        var expected = Array(UnsafeBufferPointer(start: qp, count: n))
        expected.withUnsafeMutableBufferPointer { e in
            weight.withUnsafeBufferPointer { w in
                Ops.rmsNormRope(e.baseAddress!, weight: w.baseAddress!, freqs: fp, sequence: shortest, heads: h, headDim: d, eps: eps)
            }
        }
        _ = gpu(shortest)
        let got = Array(UnsafeBufferPointer(start: qp, count: n))
        let error = expected.withUnsafeBufferPointer { channelError(got, $0, columns: d).worst }
        let gpuRight = error.isFinite && error <= Diagnostic.fp32Threshold
        if !gpuRight { measurements.wrong.append(Fit.Measurements.normRopeGPU); measurements.wrong.append(Fit.Measurements.normRopeGPUWaited) }
        var solos: [Fit.Measurements.Solo] = []
        for s in sequences.sorted() {
            let values = Double(2 * s * h * d), shape = [s, h, d]
            if let t = guarded("cpu/normrope S=\(s)", { CapacityProbes.steady { cpu(s); return true } }) {
                solos.append(.init(kernel: Fit.Measurements.normRopeCPU, shape: shape, flop: values, seconds: t))
            }
            guard gpuRight else { continue }
            let figures = guarded("gpu/normrope S=\(s)") { () -> (own: Double, wall: Double)? in
                _ = gpu(s); _ = gpu(s)
                var own: [Double] = [], wall: [Double] = []
                for _ in 0..<Self.normRopeCalls {
                    busy?()
                    let start = CapacityProbes.now()
                    own.append(gpu(s))
                    wall.append(CapacityProbes.now() - start)
                }
                return (Fit.median(own), Fit.median(wall))
            }
            if let f = figures, f.own > 0 {
                solos.append(.init(kernel: Fit.Measurements.normRopeGPU, shape: shape, flop: values, seconds: f.own, error: error))
                solos.append(.init(kernel: Fit.Measurements.normRopeGPUWaited, shape: shape, flop: values, seconds: f.wall, error: error))
            }
            let line = solos.filter { $0.shape == shape }.map { "\($0.kernel) " + String(format: "%.3f ms", $0.seconds * 1e3) }.joined(separator: "  ")
            log?("  S=\(s)  " + line)
        }
        measurements.normRope = solos
    }
}
