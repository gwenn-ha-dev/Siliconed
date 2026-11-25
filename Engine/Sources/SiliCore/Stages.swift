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
        /// kept from the start included, and the bytes of weights it holds now (`Stage.Plan.weights` and the groups kept
        /// since) — told as each is kept, for a reader of the stage while it lives.
        case kept(Int, weights: Int)
        /// The share of a stage's work the ANE computes, 0 to 1, as its client counts it (`computed`).
        case neural(Double)
        /// Core AI begins to specialize a function its cache does not hold (`Functions.load`), and has done so, in
        /// `seconds`: a compilation, told whether a stage is open or not.
        case compiling
        case compiled(seconds: Double)
        /// A cycle of a block, done (`Block.run`): told at the next cycle's first group, or at the block's close.
        case cycle(Cycle)
        /// **What a stage's calls executed, over their span** (`Work`), in every run: a block's at its close (every call
        /// since its opening), a stage of bands' at each run's end.
        case work(Work)
        /// **What a stage opened under the trace recorded** (`tracing`, `Trace`): told by such a stage alone.
        case traced(Trace)
    }

    /// **What a stage's calls executed, over their span**: their FLOP by unit (`Aicode.Module.flops`), counted as each
    /// call is made, from the first call put on the stream to the last done — what the loop waited for the ring and its
    /// client's own work between two calls inside the span, as in a cycle's seconds. Its end, `DispatchTime` nanoseconds.
    public struct Work: Sendable, Equatable {
        public let stage: String
        public let flops: Flops
        public let seconds: Double
        public let end: UInt64
        /// TFLOP/s over the span.
        public var rate: Double { seconds > 0 ? Double(flops.ane + flops.gpu) / seconds / 1e12 : 0 }
    }

    /// **The trace's switch**: a stage opened while it is on records what it does call by call (`Event.traced`), read
    /// once at its opening and never at a call; a stage opened while it is off records nothing — a test a call. Off
    /// until a client says otherwise.
    public static var tracing: Bool {
        get { lock.withLock { _tracing } }
        set { lock.withLock { _tracing = newValue } }
    }
    nonisolated(unsafe) private static var _tracing = false

    /// **A cycle of a block, as it ran**: the ANE's rows over it at its unit (0: the GPU alone) of the rows it could
    /// take, whether it visits the share of none (`Block.zero`: where this Mac's κ is unknown, or predicted the fastest); its seconds from
    /// its first group to the next cycle's first (the last cycle's, to the block's close); what
    /// its start loaded, in seconds — the GPU's functions of a share moved (`moved`), of taller calls (`climbed`); nil,
    /// nothing; its calls' seconds, where they were timed (a share in flight, every cycle but the last); whether its
    /// start's move to the share of none was refused under B (`Block.zero`: κ measured at a later opening); what its
    /// calls executed (`flops`), and the rates that gives (`rates`); when it ended (`end`, `DispatchTime` nanoseconds), for a
    /// client that lays the cycles on its own clock.
    public struct Cycle: Sendable, Equatable {
        public let block: String
        public let unit, rows, shareable: Int
        public let trying: Bool
        public let seconds: Double
        public let end: UInt64
        public let moved, climbed: Double?
        public let calls: Calls?
        public let refused: Bool
        public let flops: Flops

        /// **Its rates, TFLOP/s**: the block's over its own seconds (what its start loaded left out); each unit's over its
        /// calls' seconds, where they were timed.
        public var rates: Rates {
            let own = seconds - (moved ?? 0) - (climbed ?? 0), gpu = calls.map { $0.beside + $0.alone } ?? 0
            return Rates(block: own > 0 ? Double(flops.ane + flops.gpu) / own / 1e12 : 0,
                         ane: (calls?.ane ?? 0) > 0 ? Double(flops.ane) / calls!.ane / 1e12 : nil,
                         gpu: gpu > 0 ? Double(flops.gpu) / gpu / 1e12 : nil)
        }
    }

    /// **What a cycle's calls executed**, by unit (`Aicode.Module.flops`): counted as each function is written, added up
    /// as its calls are made — no timing, no wait of their own.
    public struct Flops: Sendable, Equatable {
        public internal(set) var ane = 0, gpu = 0
        init(ane: Int = 0, gpu: Int = 0) { (self.ane, self.gpu) = (ane, gpu) }
    }

    /// **A block's rates, TFLOP/s**: over a cycle (`block`), and each unit's over its own calls (nil: not timed — the
    /// GPU alone, the last cycle).
    public struct Rates: Codable, Sendable, Equatable {
        public let block: Double
        public let ane, gpu: Double?

        /// Whether `other` moved by 1 % or more from these, any of the three.
        func moved(_ other: Rates) -> Bool {
            func far(_ a: Double?, _ b: Double?) -> Bool {
                guard let a, let b else { return (a == nil) != (b == nil) }
                return abs(a - b) >= 0.01 * max(abs(a), abs(b))
            }
            return far(block, other.block) || far(ane, other.ane) || far(gpu, other.gpu)
        }
    }

    /// **A cycle's calls, their seconds summed**: the ANE's; the GPU's beside them (past the ANE's rows); the GPU's
    /// alone (its own steps, every ANE call before them done); and each step's on each unit (`steps`).
    public struct Calls: Sendable, Equatable {
        public let ane, beside, alone: Double
        public let steps: [Step]

        /// **A step's calls on one unit over a cycle** (`ane`, `beside`, `alone`): how many, their rows, seconds and FLOP
        /// summed — what tells a call's fixed cost from its rate, step by step.
        public struct Step: Sendable, Equatable {
            public let name, unit: String
            public internal(set) var calls = 0, rows = 0, flops = 0
            public internal(set) var seconds = 0.0
            init(name: String, unit: String) { (self.name, self.unit) = (name, unit) }
            mutating func add(rows r: Int, seconds s: Double, flops f: Int) { calls += 1; rows += r; seconds += s; flops += f }
        }
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
