import Foundation

/// **The stages of a process as they open and close, told to whoever listens**: what a client says of a piece of work —
/// each stage's P and B, how long it lived — without its own code naming them. An
/// observer is called on the thread that opens or closes the stage, and must return at once.
public enum Stages {
    public enum Event: Sendable {
        /// A stage allocates: its plan (`p`, `budget`).
        case opened(Stage.Plan)
        /// A stage let go (`Stage.close`): its plan, the seconds it lived, the groups it kept (from the start and in
        /// flight).
        case closed(Stage.Plan, seconds: Double, kept: Int = 0)
        /// A stage's reader kept one more group in flight (`Ring`): the groups the open stage holds now, those its plan
        /// kept from the start included — told as each is kept, for a reader of the stage while it lives.
        case kept(Int)
        /// The share of a stage's work the ANE computes, 0 to 1, as its client counts it (`computed`).
        case neural(Double)
        /// Core AI begins to specialize a function its cache does not hold (`Functions.load`), and has done so, in
        /// `seconds`: a compilation, told whether a stage is open or not.
        case compiling
        case compiled(seconds: Double)
        /// A cycle of a block, done (`Block.run`): told at the next cycle's first group, or at the block's close.
        case cycle(Cycle)
        /// A block's ANE unit decided (`Block.settle`).
        case weighed(Weighed)
    }

    /// **A cycle of a block, as it ran**: the ANE's rows over it at its unit (0: the GPU alone) of the rows it could
    /// take; its seconds from its first group to the next cycle's first (the last cycle's, to the block's close); what
    /// its start loaded, in seconds — the GPU's functions of a share moved (`moved`), of taller calls (`climbed`); nil,
    /// nothing; its calls' seconds, where they were timed (a share in flight, every cycle but the last).
    public struct Cycle: Sendable, Equatable {
        public let block: String
        public let unit, rows, shareable: Int
        public let seconds: Double
        public let moved, climbed: Double?
        public let calls: Calls?
    }

    /// **A cycle's calls, their seconds summed**: the ANE's; the GPU's beside them (past the ANE's rows); the GPU's
    /// alone (its own steps, every ANE call before them done).
    public struct Calls: Sendable, Equatable {
        public let ane, beside, alone: Double
    }

    /// **A block's ANE unit decided**: for each unit tried, the seconds of a cycle at its best share, as predicted from its
    /// measured cycle, and that share's ANE rows (none: decided without a trial); the unit kept on this Mac.
    public struct Weighed: Sendable, Equatable {
        public let block: String
        public let seconds: [Int: Double]
        public let rows: [Int: Int]
        public let kept: Int
    }

    /// One observer, heard until it is cancelled.
    public final class Observation: Sendable {
        let id: Int
        init(_ id: Int) { self.id = id }
        public func cancel() { Stages.lock.withLock { _ = Stages.observers.removeValue(forKey: id) } }
        deinit { cancel() }
    }

    nonisolated(unsafe) private static var observers: [Int: @Sendable (Event) -> Void] = [:]
    nonisolated(unsafe) private static var next = 0
    private static let lock = NSLock()

    /// `observer` hears every stage from now until the observation is cancelled (or let go).
    public static func observe(_ observer: @escaping @Sendable (Event) -> Void) -> Observation {
        lock.withLock {
            next += 1
            observers[next] = observer
            return Observation(next)
        }
    }

    static func tell(_ e: Event) {
        for o in lock.withLock({ observers.sorted { $0.key < $1.key }.map(\.value) }) { o(e) }
    }

    /// **The share of the open stage's work the ANE computes** (its rows over the stage's): the
    /// client's to say, once it runs a share — the engine counts no unit of its work.
    public static func computed(neuralEngine share: Double) { tell(.neural(share)) }
}
