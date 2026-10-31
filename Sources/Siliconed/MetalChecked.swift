import Metal

// **Metal objects the engine cannot go on without.** `makeCommandBuffer()`, `makeComputeCommandEncoder()` and `makeBuffer(length:)`
// answer `nil` when the queue was invalidated or saturated, or the GPU's memory is out: a bare `!` there crashes with
// "unexpectedly found nil" and no cause. These say which object, and on which queue or size, in the crash report. The callers
// are inside non-throwing loops (see `Attention.verdict`): a refusal here is a breakdown, not a state to compute through.
extension MTLCommandQueue {
    func checkedCommandBuffer() -> MTLCommandBuffer {
        guard let commands = makeCommandBuffer() else {
            fatalError("Metal refused a command buffer on queue \(label ?? "unnamed") (invalidated or saturated)")
        }
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

enum SystemGPU {
    /// **The system's GPU and a command queue on it**, or `GEMM.Failure.noDevice`: the opening of every model that computes on the GPU.
    static func open() throws -> (device: MTLDevice, queue: MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw GEMM.Failure.noDevice
        }
        return (device, queue)
    }
}
