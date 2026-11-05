import Foundation
import Metal

/// **What the measure says of the Mac it ran on, from the inside**: was something else using the Mac while it measured? Three
/// readings, none of them a load average, written beside the model in the report and read by its findings (`DiagnosticFindings`):
///
///     pulse    after every step of the probes, a few milliseconds: do as many threads as the performance cores still get a
///              core each (a fixed loop, the median thread's time over one thread alone), and how long does an empty submission
///              wait before the GPU starts it                                              (`Pulse`, `CapacityProbes.Pulser`)
///     spread   (p90 − p10) / median of the calls of a chain, and of the runs of a steady figure: identical work that takes
///              different times                                                             (`Regime.spreads`)
///     system   the host's busy ticks minus this process's own CPU, over the run and per stage: the others' cores — the *cause*,
///              written for whoever reads the report, not what the findings stand on       (`LoadWatch`)
///
/// A step whose pulse or spread says the Mac was busy is measured again (`CapacityProbes.Context.guarded`), after a wait for a
/// quiet pulse, while the ceiling leaves time; what it could not redo is kept, its chains read on their fast side
/// (`CapacityProbes.figure`), and said by the findings as disturbed — an uncertainty, never a reason to refuse the map.
///
/// What sets it apart — what a "was the Mac busy?" written by analogy with a load average would get wrong:
///
///   - **A load is not an effect.** The others' CPU at rest is 0.1 to 1.2 cores depending on the apps open, 1 to 1.3 cores of
///     background work moved a step by 0.4 to 1.3 %, and one core on the performance cluster by 6.6 % (30/10/2026); a
///     GPU shared with another process loses half its rate while the CPU sees 0.5 core. The effect is read on identical work
///     (the spread) and on what the work needs (a core each, a prompt GPU), never on the cores the others took.
///   - **A constant disturbance has no start and no end**: ten busy loops beside the whole measure leave the first witness and
///     the last equal to each other and 40 % under the Mac's own. Comparing the measure with itself in time cannot see it; a
///     quantity that is 1 on a free Mac (a crowd of threads against one thread) can, and so can the dispersion of identical calls.
///   - **The host's tick counters are read once a second**: a half-second window of the system's account reads 0, 0, 0, then
///     four times too much. Windows here are seconds.
extension Diagnostic {
    package struct Regime: Codable, Sendable, Equatable {
        /// **The state of the Mac at the end of a step**: a crowd of threads against one thread alone, and the wait of an empty GPU
        /// submission. `attempt` counts the times the step was measured (a step measured again has several pulses).
        package struct Pulse: Codable, Sendable, Equatable {
            package var at: Double
            package var step: String
            package var attempt: Int
            /// The median thread's seconds over one thread's alone, for as many threads as the performance cores: 1 on a free Mac.
            package var crowd: Double
            /// One thread's seconds on the loop, alone, just now: the clock the core runs at, and whether anything shares it.
            package var alone: Double
            /// The median wait of an empty submission to the GPU, seconds; and the longest.
            package var wait: Double
            package var late: Double
            package init(at: Double, step: String, attempt: Int, crowd: Double, alone: Double = 0, wait: Double, late: Double) {
                self.at = at; self.step = step; self.attempt = attempt; self.crowd = crowd; self.alone = alone; self.wait = wait; self.late = late
            }

            private enum CodingKeys: String, CodingKey { case at, step, attempt, crowd, alone, wait, late }
            /// `alone` is a late field: the pulses recorded before it (the cards the rules were set on) read as 0.
            package init(from decoder: any Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                at = try c.decode(Double.self, forKey: .at); step = try c.decode(String.self, forKey: .step)
                attempt = try c.decode(Int.self, forKey: .attempt); crowd = try c.decode(Double.self, forKey: .crowd)
                alone = try c.decodeIfPresent(Double.self, forKey: .alone) ?? 0
                wait = try c.decode(Double.self, forKey: .wait); late = try c.decode(Double.self, forKey: .late)
            }
        }

        /// The pulses in the order they were taken.
        package var pulses: [Pulse] = []
        /// (p90 − p10) / median of the calls of each chain, or (max − min) / median of the runs of a steady figure, keyed by step; the
        /// last measure of a step is the one kept.
        package var spreads: [String: Double] = [:]
        /// Cores the other processes used over the run (`run`) and per stage, from the system's ticks.
        package var foreign: [String: Double] = [:]
        /// The same per second of the run, in order.
        package var windows: [Double] = []
        /// How many steps were measured again, and the seconds the guard spent on it (the waits included).
        package var repeated = 0
        package var repeatedSeconds = 0.0
        /// **Seconds one pulse took before the projection** (`Sizes.Rates.pulse`), the fast side of two: what the projection counted per
        /// step. `nil` in a report that predates it (under schema 23).
        package var pulseSeconds: Double?
        package init() {}

        // MARK: The rules

        /// **The thresholds** (30/10/2026, the M1 Pro: quiet cards, and cards beside ten, four, two and one busy loops, a compilation,
        /// and a GPU product loop in another process).
        package enum Rules {
            /// The median thread of a crowd of as many threads as performance cores takes this much over one thread alone.
            package static let crowd = 1.25
            /// An empty submission waits this many times the quietest wait of the run (and not less than `waitFloor`: a Mac whose quietest
            /// is 0.12 ms is not busy at 0.25), or this long in all (`waitCeiling`: a jam that has lasted the whole run has no quiet to compare with).
            package static let waitRatio = 2.0, waitFloor = 0.35e-3, waitCeiling = 1.5e-3
            /// Identical calls differ by this much (p90 − p10 over the median). A free Mac's chains differ by 0.02–0.06, and now and then two AMX chains
            /// at a depth of 12 288 by 0.16–0.22 (their rate there is unsteady by nature): the step is measured again and the second time is
            /// the free one. Beside one or two busy loops or a compilation they differ by 0.08–0.20 with a line off by 8 % at most; beside four
            /// loops, a GPU loop or a burst, by 0.27–1.07, and the step stays so when it is measured again.
            package static let spread = 0.15
        }

        /// The quietest wait of the run, the tenth percentile of its pulses (not the smallest: one lucky pulse would make a wait of every
        /// other pulse twice too long): what an empty submission costs on this Mac when nothing else asks for the GPU.
        package var quietestWait: Double? {
            let waits = pulses.map(\.wait).sorted()
            return waits.isEmpty ? nil : waits[Int(Double(waits.count - 1) * 0.1)]
        }

        /// Whether this pulse says the Mac was busy: the crowd lost cores, or the GPU kept a submission waiting.
        package func busy(_ pulse: Pulse) -> Bool {
            pulse.crowd > Rules.crowd || late(pulse.wait, over: quietestWait)
        }

        private func late(_ wait: Double, over quietest: Double?) -> Bool {
            wait > Rules.waitCeiling || (quietest.map { wait > max(Rules.waitRatio * $0, Rules.waitFloor) } ?? false)
        }

        /// **What the measure kept of a busy Mac**: the pulses that closed a step for the last time and say it was busy, the
        /// spreads of the steps that are over the rule. Empty when every step was measured on a free Mac — or measured again until it was.
        package struct Disturbance: Equatable, Sendable {
            package enum Kind: String, Sendable { case threads, gpu, calls }
            package var kind: Kind
            package var steps: [String]
            /// The worst figure: the crowd's ratio, the wait in seconds, the spread.
            package var worst: Double
            package var worstStep: String
        }

        package var disturbances: [Disturbance] {
            var last: [String: Pulse] = [:]
            for p in pulses { last[p.step] = p }
            let kept = pulses.filter { last[$0.step] == $0 }
            var out: [Disturbance] = []
            let threads = kept.filter { $0.crowd > Rules.crowd }
            if let worst = threads.max(by: { $0.crowd < $1.crowd }) {
                out.append(Disturbance(kind: .threads, steps: threads.map(\.step), worst: worst.crowd, worstStep: worst.step))
            }
            let floor = quietestWait
            let slow = kept.filter { late($0.wait, over: floor) }
            if let worst = slow.max(by: { $0.wait < $1.wait }) {
                out.append(Disturbance(kind: .gpu, steps: slow.map(\.step), worst: worst.wait, worstStep: worst.step))
            }
            let wide = spreads.filter { $0.value > Rules.spread }.sorted { $0.key < $1.key }
            if let worst = wide.max(by: { $0.value < $1.value }) {
                out.append(Disturbance(kind: .calls, steps: wide.map(\.key), worst: worst.value, worstStep: worst.key))
            }
            return out
        }

        func rounded() -> Regime {
            func r(_ x: Double) -> Double { Diagnostic.significant(x, 4) }
            var out = self
            out.pulses = pulses.map { Pulse(at: r($0.at), step: $0.step, attempt: $0.attempt, crowd: r($0.crowd), alone: r($0.alone), wait: r($0.wait), late: r($0.late)) }
            out.spreads = spreads.mapValues(r)
            out.foreign = foreign.mapValues(r)
            out.windows = windows.map(r)
            out.repeatedSeconds = r(repeatedSeconds)
            return out
        }
    }
}

// MARK: - The machine layer

extension Diagnostic.CapacityProbes {
    /// **Watches the system while the probes run**: the host's busy ticks minus this process's own CPU seconds — the others' cores —
    /// read at the edges of each stage, and every quarter of a second from a thread of its own (summed to seconds at the end).
    final class LoadWatch: @unchecked Sendable {
        static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) * 1e-9 }

        /// Busy ticks of every core (user + system + nice), in 1/100 s of one core.
        static func hostBusy() -> Double {
            var info = host_cpu_load_info()
            var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
            let status = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count) }
            }
            guard status == KERN_SUCCESS else { return .nan }
            let t = info.cpu_ticks
            return Double(t.0) + Double(t.1) + Double(t.3)
        }

        /// This process's CPU seconds, all its threads.
        static func ownCPU() -> Double {
            var u = rusage()
            getrusage(RUSAGE_SELF, &u)
            return Double(u.ru_utime.tv_sec) + Double(u.ru_utime.tv_usec) * 1e-6 + Double(u.ru_stime.tv_sec) + Double(u.ru_stime.tv_usec) * 1e-6
        }

        /// A reading of the system, to be compared with a later one.
        struct Reading {
            let host = LoadWatch.hostBusy(), own = LoadWatch.ownCPU(), at = LoadWatch.now()
            /// Cores the others used since the reading.
            func foreign() -> Double {
                let dt = LoadWatch.now() - at
                guard dt > 0 else { return 0 }
                return max(0, (LoadWatch.hostBusy() - host) / 100 - (LoadWatch.ownCPU() - own)) / dt
            }
        }

        private static let window = 0.25
        private let lock = NSLock()
        private var running = true
        private var recorded: [Double] = []
        private var edges: [String: Reading] = [:]
        private var stages: [String: Double] = [:]
        private let whole = Reading()

        init() {
            Thread.detachNewThread { [self] in
                while true {
                    let reading = Reading()
                    Thread.sleep(forTimeInterval: Self.window)
                    lock.lock()
                    let go = running
                    if go { recorded.append(reading.foreign()) }
                    lock.unlock()
                    if !go { return }
                }
            }
        }

        func begin(_ stage: String) { lock.lock(); edges[stage] = Reading(); lock.unlock() }

        func end(_ stage: String) {
            lock.lock()
            if let start = edges[stage] { stages[stage] = start.foreign() }
            lock.unlock()
        }

        /// Stops the watch and writes what it saw into `regime`.
        func finish(into regime: inout Diagnostic.Regime) {
            lock.lock(); running = false
            regime.windows = recorded.count >= 4 ? stride(from: 0, to: recorded.count - 3, by: 4).map { recorded[$0..<$0 + 4].reduce(0, +) / 4 } : []
            regime.foreign = stages
            regime.foreign["run"] = whole.foreign()
            lock.unlock()
        }
    }

    // MARK: The pulse

    /// A loop of dependent fused multiply-adds: no memory, no branch, the same time for the same core at the same clock.
    @inline(never) static func burn(_ iterations: Int, into sink: UnsafeMutablePointer<Float>) {
        var a = Float(1), b = Float(1.1), c = Float(1.2), d = Float(1.3)
        for _ in 0..<iterations {
            a = a * 0.99991 + 0.00013; b = b * 0.99992 + 0.00014; c = c * 0.99993 + 0.00015; d = d * 0.99994 + 0.00016
        }
        // Written out: a call whose result nobody reads is a call the compiler removes.
        sink.pointee = a + b + c + d
    }

    /// **The pulse's instruments**, made once: the loop sized for a few milliseconds alone, the empty kernel the GPU is asked to run.
    final class Pulser: @unchecked Sendable {
        let threads: Int
        private let sinks = UnsafeMutablePointer<Float>.allocate(capacity: 64 * 16)
        private var iterations = 1_000_000
        /// One thread's seconds on the loop, alone.
        private(set) var solo = 0.0
        private let queue: MTLCommandQueue
        private let pipeline: MTLComputePipelineState?
        private let buffer: MTLBuffer?

        /// `threads`: the performance cores. `target`: seconds the loop takes alone.
        init(threads: Int, device: MTLDevice, queue: MTLCommandQueue, target: Double = 0.004) {
            self.threads = max(threads, 1)
            self.queue = queue
            sinks.initialize(repeating: 0, count: 64 * 16)
            let source = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void nothing(device float* d [[buffer(0)]], uint i [[thread_position_in_grid]]) { d[i] = d[i] + 1.0; }
            """
            pipeline = (try? device.makeLibrary(source: source, options: nil)).flatMap { $0.makeFunction(name: "nothing") }
                .flatMap { try? device.makeComputePipelineState(function: $0) }
            buffer = device.makeBuffer(length: 4096, options: .storageModePrivate)
            var t0 = LoadWatch.now()
            Diagnostic.CapacityProbes.burn(iterations, into: sinks)
            iterations = max(1000, Int(Double(iterations) * target / max(LoadWatch.now() - t0, 1e-6)))
            var alone: [Double] = []
            for _ in 0..<5 {
                t0 = LoadWatch.now()
                Diagnostic.CapacityProbes.burn(iterations, into: sinks)
                alone.append(LoadWatch.now() - t0)
            }
            solo = CapacityFit.median(alone)
            _ = crowd(); _ = crowd()   // the pool of threads awake
        }

        deinit { sinks.deallocate() }

        /// **The median thread's seconds over one thread's alone, both taken now**, the best of two runs each: the cores the crowd gets,
        /// whatever the clock they run at. A cluster that a long GPU + Neural Engine run has left at a lower clock slows the crowd and the
        /// thread alone alike (the crowd over a thread alone *measured long before* read 1.5–1.9 after such a set on a Mac whose
        /// others used 0.2 core); a neighbour that holds cores slows the crowd and not the thread alone. A burst is the Mac's, a lull is
        /// not a measure: the best of two.
        func crowd() -> Double { crowdAndAlone().crowd }

        /// The same, with the thread alone's seconds it was taken against.
        func crowdAndAlone() -> (crowd: Double, alone: Double) {
            guard threads > 1, solo > 0 else { return (1, solo) }
            nonisolated(unsafe) let sink = sinks
            let n = iterations, count = threads
            func one() -> Double {
                let start = LoadWatch.now()
                Diagnostic.CapacityProbes.burn(n, into: sink)
                return LoadWatch.now() - start
            }
            let alone = min(one(), one())
            func run() -> Double {
                let times = UnsafeMutablePointer<Double>.allocate(capacity: count)
                times.initialize(repeating: 0, count: count)
                defer { times.deallocate() }
                nonisolated(unsafe) let out = times
                DispatchQueue.concurrentPerform(iterations: count) { i in
                    let start = LoadWatch.now()
                    Diagnostic.CapacityProbes.burn(n, into: sink + 16 * (i % 64))
                    out[i] = LoadWatch.now() - start
                }
                return CapacityFit.median((0..<count).map { times[$0] }) / alone
            }
            return (min(run(), run()) , alone)
        }

        /// The wait of an empty submission to the GPU: median and longest of twelve command buffers committed and waited for one by one.
        func wait() -> (median: Double, longest: Double) {
            guard let pipeline, let buffer else { return (0, 0) }
            var waits: [Double] = []
            for i in 0..<14 {
                guard let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { return (0, 0) }
                e.setComputePipelineState(pipeline)
                e.setBuffer(buffer, offset: 0, index: 0)
                e.dispatchThreads(MTLSize(width: 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                e.endEncoding()
                let start = LoadWatch.now()
                cb.commit(); cb.waitUntilCompleted()
                if i >= 2 { waits.append(LoadWatch.now() - start) }
            }
            return (CapacityFit.median(waits), waits.max() ?? 0)
        }
    }

    // MARK: Spread

    /// (p90 − p10) / median of a chain's calls — six at least: over fewer, the percentiles are the extremes, and a free Mac's extremes
    /// differ by 20 % (the three or four runs of a steady figure say nothing of the Mac).
    static func spread(_ times: [Double]) -> Double {
        guard times.count >= 6 else { return 0 }
        let s = times.sorted()
        let median = CapacityFit.median(s)
        guard median > 0 else { return 0 }
        return (s[Int(Double(s.count - 1) * 0.9)] - s[Int(Double(s.count - 1) * 0.1)]) / median
    }

    /// **A chain's figure: the median of its calls, or — when they disagree — the median of its quickest third.** Identical calls on a
    /// Mac nobody else uses take the same time (0.02–0.06 apart, p90 − p10 over the median), and the median is their figure. When they
    /// differ by more than the rule (`Regime.Rules.spread`) something else was using the Mac, and **a disturbance only adds time**: the
    /// quick calls are the engine's own, the slow ones are the company it kept. A chain that agrees reads as it always did, bit for bit.
    static func figure(_ times: [Double]) -> Double {
        let used = quick(times.map { (wall: $0, within: [String: Double]()) }).map(\.wall)
        return CapacityFit.median(used)
    }

    /// The calls a chain's figure is read from: all of them when they agree, the quickest third (three at least) when they do not.
    /// The sides of a set are read from the same calls as its wall, so that the balance and the keeps stay one measure.
    static func quick(_ samples: [(wall: Double, within: [String: Double])]) -> [(wall: Double, within: [String: Double])] {
        guard spread(samples.map(\.wall)) > Diagnostic.Regime.Rules.spread else { return samples }
        let fastest = samples.sorted { $0.wall < $1.wall }
        return Array(fastest.prefix(max(3, samples.count / 3)))
    }
}
