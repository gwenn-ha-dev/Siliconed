import CoreAI
import Metal

extension Stage {
    /// **A stage planned under `budget` and opened**: planned on the process's footprint, refused at once where its socle
    /// and the share its functions were last read at on this Mac pass B (`estimates`, `choose`: a set the fuse stopped is
    /// refused too); never read, opened, the fuse behind.
    public static func open(_ device: MTLDevice, _ declaration: Declaration, files: [SafeTensors], budget: Int, depth: Int = 2,
                            measured: ThisMac = .current) async throws -> Stage {
        let p = try plan(declaration, files: files, budget: budget, floor: Memory.footprint(), depth: depth)
        _ = try choose([p], estimates([(key(p.declaration), 0)], measured: measured))
        return try await Stage(device, plan: p, files: files, measured: measured)
    }
}
