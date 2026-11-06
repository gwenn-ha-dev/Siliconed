import Foundation
import Metal
import MetalPerformanceShadersGraph

/// **What is still outside our hand** — the bytes of a render no constructor of ours sizes.
///
/// Everything else of a render's memory is ours and exact by construction: arenas, `Pages`, the VAEs'
/// two heaps, the GEMM's padding tiers, the tokenizer's tables, each sized by the shape function its
/// constructor calls (`MemoryNeed`). The per-operator laws once fitted on this M1 Pro — `a + b·elements`
/// per operator — are gone, and **so are their constants**: a figure no shape gives is not folded into
/// the formula, it is removed by the structure or left to the check, which reports it (`MemoryCheck`).
///
/// **Charged, because a shape gives it** (each one copy of a tensor whose shape we hand in):
///
///     sdpaQueries   MPSGraph's SDPA transposes the queries into [H, S, Dh] before its kernel: one fp32
///                   copy, rows·heads·headDim·4 (measured: 4.03 B per element on six shapes: the copy,
///                   nothing else grows with the shape). The matrix H·S·K is never materialized on
///                   macOS 27 (measured: 41 MB where it would be 3.3 GB)
///     VAELayout.sdpaOutside  the VAEs' one-head SDPA: its q, k, v and output copied, 4 · S·C fp32
///
/// **Not charged — the frameworks' constants, no shape gives them** (12/10/2026: taken out of the
/// formula, where they had been folded in after the check reported them). Removing them means leaving
/// MPS or MPSGraph, which changes bits (a decomposed VAE SDPA moved its output by 1.35·10⁻⁵); so
/// they stay, the check reports them, and they are listed among the open items:
///
///     MPSGraph's compiled SDPA   ~7.4 MB per SDPA graph alive in a phase (measured alone, six shapes, constant)
///     MPS's GEMM state           ~3.8 MB per `MPSMatrixMultiplication` model (measured alone, four shapes, constant)
///     Core ML's programs         ~0.4 MB per ANE program loaded (`gemm_ane`: 8 MB beyond its surfaces over five
///                                shapes, ~6 of them the framework's first use, which `bringUp` moves into the base)
///     Metal's first use          the device, its queue, the compiled kernels of a process's first render
///                                when the base was read before them (12.0–13.1 MB for a fresh CLI)
///     a phase's leftovers        `malloc`'s small regions kept dirty by MPSGraph's compilation churn,
///                                graphics Metal frees late (~18 MB at Qwen 512²'s decoder)
///
/// **The process's base is read, not charged** (`processBase()`): whatever the process holds when the
/// render starts — the app's interface and history, the command line's runtime — is its footprint
/// then, and B covers it with the render.
package enum OperatorFootprint {

    /// **The queries' fp32 copy MPSGraph's SDPA makes** (the transposition to heads-first), for one
    /// cached `Attention` or the GPU part of a `HeadSplitAttention`: `rows · heads · headDim · 4` bytes.
    package static func sdpaQueries(rows: Int, heads: Int, headDim: Int) -> Int { rows * heads * headDim * 4 }

    /// **The process's footprint now** (`phys_footprint`, `Arena.processFootprint`): what a render about
    /// to start finds already held — the app's or the command line's own. Read from this process alone,
    /// never from the kernel's state of the others. **Less the text encoder kept** (`KeptEncoder.heldBytes`): a plan charges it
    /// itself, or gives it back before its first computation.
    package static func processBase() -> Int { max(0, Arena.processFootprint() - KeptEncoder.heldBytes) }

    /// **The frameworks a render runs on, brought up before its base is read** — once per process: the
    /// Metal device and a queue, and MPSGraph's own start (one graph of one addition, run). What they
    /// hold — ~2 MB for the device, ~7 MB for MPSGraph's first graph (mlir's context, its uniquers,
    /// measured alone) — is the process's from then on, every render after the first finds it in its
    /// base; the first found it nowhere, brought up in its text phase after the base was read, and
    /// carried it uncharged into every phase after. Brought up first, the base reads it.
    /// The same work a render did anyway, moved before it: no time added.
    /// Under a plan with the ANE, Core ML's too (`NeuralEngine.bringUp`).
    /// The queue dies with the bring-up, never kept: the Metal driver holds its command-buffer memory
    /// (tens of MB) while any queue that ever submitted lives, and gives it back at once when the last
    /// one goes — a queue kept here would leave a released model's share to the idle trim, a second later.
    package static func bringUp() {
        _ = frameworks
        if EngineSettings.effective.plan.usesANE { _ = NeuralEngine.bringUp }
    }

    private static let frameworks: MTLDevice? = {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { return nil }
        autoreleasepool {
            let graph = MPSGraph()
            let x = graph.placeholder(shape: [4], dataType: .float32, name: nil)
            let y = graph.addition(x, x, name: nil)
            if let buffer = device.makeBuffer(length: 16, options: .storageModeShared) {
                _ = graph.run(with: queue, feeds: [x: MPSGraphTensorData(buffer, shape: [4], dataType: .float32)],
                              targetTensors: [y], targetOperations: nil)
            }
        }
        return device
    }()
}
