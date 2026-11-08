import Foundation

// **What every model of the engine does to be checked and timed, written once.** A check reads a model's intermediate
// tensors by name (`boundaries`, filled when `recordBoundaries` is set) and its time by phase (`timings`); the model records
// and times through these two defaults instead of carrying a copy each. Pure bookkeeping: no computation here.

/// A model that can hand its intermediate tensors to a check.
protocol BoundaryRecording: AnyObject {
    var recordBoundaries: Bool { get }
    var boundaries: [String: [Float]] { get set }
}

extension BoundaryRecording {
    /// Keeps a copy of `n` floats at `p` under `name`, when a check asked for them.
    func record(_ name: String, _ p: UnsafePointer<Float>, _ n: Int) {
        guard recordBoundaries else { return }
        boundaries[name] = Array(UnsafeBufferPointer(start: p, count: n))
    }
}

/// A model that keeps its own seconds by phase (the weight products' are added by `timings`).
protocol PhaseTiming: AnyObject {
    var ownTimings: [String: Double] { get set }
}

extension PhaseTiming {
    /// The body's result, which a caller timing a GPU launch for its phase alone may drop (the launch's own seconds).
    @discardableResult
    func timed<T>(_ phase: String, _ body: () throws -> T) rethrows -> T {
        let started = DispatchTime.now().uptimeNanoseconds
        defer { ownTimings[phase, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9 }
        return try body()
    }
}
