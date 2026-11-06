import Metal
import ObjectiveC
import Synchronization

// **Metal objects the engine cannot go on without.** `makeCommandBuffer()`, `makeComputeCommandEncoder()` and `makeBuffer(length:)`
// answer `nil` when the queue was invalidated or saturated, or the GPU's memory is out: a bare `!` there crashes with
// "unexpectedly found nil" and no cause. These say which object, and on which queue or size, in the crash report. The callers
// are inside non-throwing loops (see `Attention.verdict`): a refusal here is a breakdown, not a state to compute through.
extension MTLCommandQueue {
    func checkedCommandBuffer() -> MTLCommandBuffer {
        guard let commands = makeCommandBuffer() else {
            fatalError("Metal refused a command buffer on queue \(label ?? "unnamed") (invalidated or saturated)")
        }
        stream.awaitNeuralEngine(in: commands)
        return commands
    }
}

extension MTLCommandBuffer {
    func checkedEncoder() -> MTLComputeCommandEncoder {
        guard let encoder = makeComputeCommandEncoder() else {
            fatalError("Metal refused a compute encoder on command buffer \(label ?? "unnamed") (status \(status.rawValue))")
        }
        return encoder
    }
}

extension MTLDevice {
    func checkedBuffer(length: Int, options: MTLResourceOptions = .storageModeShared) -> MTLBuffer {
        guard let buffer = makeBuffer(length: length, options: options) else {
            fatalError("Metal refused a buffer of \(length) bytes on \(name)")
        }
        return buffer
    }
}

/// **The bytes of the Metal buffers made over our own memory, alive** (`makeBuffer(bytesNoCopy:)`: the arenas, the maps, the pages
/// wrapped for a graph), each counted from its creation to its object's death. The device's `currentAllocatedSize` counts them
/// beside Metal's own allocations — ours and the frameworks' (MPS's, MPSGraph's) —, while the graphics ledger holds only the latter
/// (a wrap's pages are anonymous memory or a file's, in the footprint): `PhaseBoundary.returnedDenoiser()` tells them apart by this.
/// A wrap made without `track` would read as Metal's own allocation: the boundary's floor that much higher, its wait that much
/// shorter. Every `bytesNoCopy` alive at that boundary tracks.
package enum MetalWraps {
    private static let bytes = Atomic<Int>(0)
    nonisolated(unsafe) private static let key = UnsafeMutableRawPointer.allocate(byteCount: 1, alignment: 1)

    package static var live: Int { bytes.load(ordering: .relaxed) }

    /// `buffer` counted until it dies: its size rides on it (an associated object), subtracted when it goes.
    static func track(_ buffer: MTLBuffer) {
        let size = buffer.allocatedSize
        bytes.add(size, ordering: .relaxed)
        objc_setAssociatedObject(buffer, key, Death(size), .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
    }

    private final class Death {
        let size: Int
        init(_ size: Int) { self.size = size }
        deinit { MetalWraps.bytes.subtract(size, ordering: .relaxed) }
    }
}

enum SystemGPU {
    /// **The system's GPU and a command queue on it**, or `GEMM.Failure.noDevice`: the opening of every model that computes on the GPU.
    static func open() throws -> (device: MTLDevice, queue: MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GEMM.Failure.noDevice
        }
        return (device, queue)
    }
}
