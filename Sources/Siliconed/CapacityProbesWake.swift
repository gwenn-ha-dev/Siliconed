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
