import Darwin
import Foundation
import Synchronization

/// **Says which route the process is about to use**, for whoever watches it from outside. The measure runs in a process of its own
/// (`Diagnostic.Child`) because some programs the Neural Engine is given kill the process that loads them rather than return an
/// error: the beacon is the last word of a process that dies. Nothing listens in a render — the call is a lock and a nil.
package enum RouteBeacon {
    private static let listener = Mutex<(@Sendable (String) -> Void)?>(nil)

    /// Who is told of each route (`nil`: nobody).
    package static func listen(_ body: (@Sendable (String) -> Void)?) { listener.withLock { $0 = body } }

    /// `route` is a kernel's name (`ane/fp16`, `ane/int8`, `gpu/q8`…).
    package static func announce(_ route: String) {
        let body = listener.withLock { $0 }
        body?(route)
    }
}

/// **The measure in a process of its own, and the process that watches it.**
///
///     the app / the command line                          the child (`Siliconed --measure-capacity`)
///       Supervisor ── launches ──────────────────────────▶ Child.serve ─▶ Diagnostic.run
///           ▲                                                  │
///           │   stage <key>       the stage it is in           │
///           │   route <kernel>    the route it is about to use │   (`RouteBeacon`, written before the route is touched)
///           └── report <json> | error <code> <detail> ◀────────┘
///
///     the child dies by a signal  ─▶  the last route it named is excluded, and the measure starts again without it
///
/// What sets it apart — what a supervisor written by analogy with "run it again" would get wrong:
///
///   - **A route that kills the process is not a wrong route, it is a route not seen**: the judge excludes a route that computed
///     wrong; here nothing was computed, the process was gone at the load. The route is excluded all the same — *a render never loads a
///     route the map has not seen load* — and the report says why (`Judge.Reason.died`), so that the one who reads it knows the
///     Mac was not slow, the route is not available.
///   - **Only a crash is a verdict on a route.** A signal of a crash (`SIGSEGV`, `SIGBUS`, `SIGILL`, `SIGABRT`, `SIGTRAP`, `SIGFPE`);
///     never `SIGKILL` or `SIGTERM` (the user stopped it, the system needed the memory), never an exit status (the child said what
///     failed), and never a route that was already excluded (that would loop).
///   - **The reference engine cannot be excluded**: the GPU's product and its attention are what everything is measured against, and
///     nothing is planned without them; their death is a failure said as such.
///   - **The child declares its role** (`claimRole`): launched by an app or a launchd job, a process that says nothing about itself reads
///     the pair of the GPU and the Neural Engine 20–50 % noisier than the same executable launched from a terminal; as a foreground
///     application, like the render it measures for, it reads the same.
///   - **The restart is bounded**: one restart per route that died, three at most — a Mac whose every route kills is a Mac to
///     tell the user about, not to loop on.
extension Diagnostic {
    package enum Child {
        /// The argument of the measuring process: `Siliconed --measure-capacity [--skip-route <kernel>]…`.
        package static let argument = "--measure-capacity"
        /// A route the measure leaves out (its process died on it).
        package static let skip = "--skip-route"
        /// The most restarts one measure makes.
        package static let maximumDeaths = 3
        /// The signals of a crash.
        package static let crashes: Set<Int32> = [SIGSEGV, SIGBUS, SIGILL, SIGABRT, SIGTRAP, SIGFPE]
        /// The kernels everything is measured against: never excluded for having died.
        package static let references: Set<CapacityModel.Kernel> = [CapacityModel.Kernel(.gpu, .fp32), CapacityModel.Kernel(.gpu, .graph)]

        // MARK: What the child says

        package enum Line {
            case stage(Stage)
            case route(String)
            case report(Report)
            case failure(code: String, detail: String)

            package init?(_ line: String) {
                let (word, rest) = line.firstIndex(of: " ").map { (line[..<$0], String(line[line.index(after: $0)...])) } ?? (line[...], "")
                switch word {
                case "stage":
                    guard let stage = Stage.allCases.first(where: { $0.key == rest }) else { return nil }
                    self = .stage(stage)
                case "route":
                    self = .route(rest)
                case "report":
                    guard let report = Report.read(Data(rest.utf8)) else { return nil }
                    self = .report(report)
                case "error":
                    let code = rest.prefix { $0 != " " }
                    self = .failure(code: String(code), detail: String(rest.dropFirst(code.count)).trimmingCharacters(in: .whitespaces))
                default: return nil
                }
            }
        }

        // MARK: The child's side

        /// **Measures and says so on the standard output**, then exits: `0` with the report, `1` with an `error` line. `onRoute`
        /// is told each route after the line is written (a test's crash switch hangs there); `skipping` is the routes to leave out;
        /// `claimingRole` is `false` for a bench that measures what the launcher's role does (`claimRole`).
        package static func serve(skipping skipped: Set<CapacityModel.Kernel>, claimingRole: Bool = true,
                                  onRoute: (@Sendable (String) -> Void)? = nil) -> Never {
            setvbuf(stdout, nil, _IOLBF, 0)   // a line reaches the other side as it is written, and the last one survives a crash
            if claimingRole { Self.claimRole() }
            let last = Mutex("")
            RouteBeacon.listen { route in
                let isNew = last.withLock { old -> Bool in
                    defer { old = route }
                    return old != route
                }
                if isNew { print("route " + route) }
                onRoute?(route)
            }
            do {
                let report = try Diagnostic.run(skipping: skipped, onProgress: { stage in print("stage " + stage.key) })
                let json = try report.json(pretty: false)
                print("report " + String(decoding: json, as: UTF8.self))
                exit(0)
            } catch {
                let e = EngineError(error)
                print("error \(e.code) " + e.description.replacingOccurrences(of: "\n", with: " "))
                exit(1)
            }
        }

        /// **The child is a foreground application, whoever launched it.** A process that sets no role takes the one of where it was
        /// launched: a child of the terminal runs as the terminal's, a child of an app opened through LaunchServices or of a launchd job gets
        /// less of the chip when the GPU and the Neural Engine work at once — the same executable on the same Mac, identical calls of that
        /// pair 20–50 % apart instead of 1–8 %, the ANE's side of a set 20 % slower. The role the app itself holds while it renders for a
        /// user who waits is the foreground's; the measure takes it, so that it reads the silicon a render reads and not the launcher's
        /// company. Another role (the default application's, `taskpolicy -a`, the throughput tiers, QoS classes, an activity held for
        /// the measure) did not move it. Returns whether the system took it.
        @discardableResult
        package static func claimRole() -> Bool {
            var policy = task_category_policy(role: TASK_FOREGROUND_APPLICATION)
            return withUnsafeMutablePointer(to: &policy) {
                $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                    task_policy_set(mach_task_self_, task_policy_flavor_t(TASK_CATEGORY_POLICY), $0, 1)
                }
            } == KERN_SUCCESS
        }

        /// The task role this process holds now (`TASK_FOREGROUND_APPLICATION` is 1), or `nil` when the system does not say.
        package static var role: Int32? {
            var policy = task_category_policy(role: TASK_UNSPECIFIED)
            var count = mach_msg_type_number_t(1)
            var isDefault: boolean_t = 0
            let status = withUnsafeMutablePointer(to: &policy) {
                $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                    task_policy_get(mach_task_self_, task_policy_flavor_t(TASK_CATEGORY_POLICY), $0, &count, &isDefault)
                }
            }
            return status == KERN_SUCCESS ? Int32(policy.role.rawValue) : nil
        }

        /// The routes a process was asked to leave out (`--skip-route <kernel>`, repeated).
        package static func skipped(in arguments: [String]) -> Set<CapacityModel.Kernel> {
            var kernels = Set<CapacityModel.Kernel>()
            for (i, a) in arguments.enumerated() where a == skip && i + 1 < arguments.count {
                if let kernel = CapacityModel.Kernel(arguments[i + 1]) { kernels.insert(kernel) }
            }
            return kernels
        }

        // MARK: The supervisor's side

        /// **How the child ended**, and what it had last said.
        package struct Death: Sendable, Equatable {
            /// The signal that ended it, or its exit status.
            package var status: Int32
            package var signaled: Bool
            package var stage: Stage?
            package var route: String?

            package init(status: Int32, signaled: Bool, stage: Stage? = nil, route: String? = nil) {
                self.status = status; self.signaled = signaled; self.stage = stage; self.route = route
            }

            /// "ended by SIGSEGV in “the engines together”, on ane/int8".
            package var description: String {
                (signaled ? "ended by signal \(status)" : "ended with status \(status)") + (stage.map { " in “\($0)”" } ?? "")
                    + (route.map { ", on \($0)" } ?? "")
            }
        }

        /// **The route a death condemns**, or `nil`: a crash (never a kill, never an exit status), on a route named and not yet
        /// excluded, that is not a reference.
        package static func culprit(of death: Death, skipped: Set<String>) -> String? {
            guard death.signaled, crashes.contains(death.status), let route = death.route, !skipped.contains(route),
                  let kernel = CapacityModel.Kernel(route), !references.contains(kernel) else { return nil }
            return route
        }

        /// What a supervised measure came to.
        package struct Outcome: Sendable {
            package var report: Report?
            /// The child's own failure (`error <code> <detail>`), or a death nothing could be done about.
            package var failure: (code: String, detail: String)?
            package var deaths: [Death] = []
            /// The routes left out, in the order their deaths came.
            package var skipped: [String] = []
            package var stopped = false
            package init() {}
        }

        /// **Runs the measuring process, and again without a route when a crash on it ended the last one.** Synchronous: call it off the
        /// main thread. `stop()` ends the running child and the restarts.
        package final class Supervisor: @unchecked Sendable {
            package let executable: URL
            package let arguments: [String]
            private let state = Mutex<(process: Process?, stopped: Bool)>((nil, false))

            package init(executable: URL, arguments: [String] = [Child.argument]) {
                self.executable = executable; self.arguments = arguments
            }

            package func stop() {
                let process = state.withLock { s -> Process? in s.stopped = true; return s.process }
                // `terminate()` on a process that never launched raises: only a running one is ended here (`run` ends one that
                // started after the request, below).
                if let process, process.isRunning { process.terminate() }
            }

            package func run(onStage: (Stage) -> Void = { _ in }, onRoute: (String) -> Void = { _ in },
                     onRestart: (_ skipped: [String]) -> Void = { _ in }) -> Outcome {
                var outcome = Outcome()
                while true {
                    let process = Process()
                    process.executableURL = executable
                    process.arguments = arguments + outcome.skipped.flatMap { [Child.skip, $0] }
                    let pipe = Pipe()
                    process.standardOutput = pipe
                    process.standardError = FileHandle.nullDevice
                    guard !state.withLock({ $0.stopped }) else { outcome.stopped = true; return outcome }
                    do { try process.run() } catch {
                        outcome.failure = ("internal", "the measuring process did not start: \(error.localizedDescription)")
                        return outcome
                    }
                    // Registered only once launched, so that `stop()` never meets a process that did not start; a stop that came
                    // between the check above and here ends this one now.
                    let stoppedMeanwhile = state.withLock { s -> Bool in s.process = process; return s.stopped }
                    if stoppedMeanwhile { process.terminate() }
                    var stage: Stage?, route: String?
                    var report: Report?, failure: (String, String)?
                    var pending = Data()
                    func take(_ line: String) {
                        switch Line(line) {
                        case .stage(let s): stage = s; onStage(s)
                        case .route(let r): route = r; onRoute(r)
                        case .report(let r): report = r
                        case .failure(let code, let detail): failure = (code, detail)
                        case nil: break
                        }
                    }
                    while true {
                        let chunk = pipe.fileHandleForReading.availableData
                        if chunk.isEmpty { break }
                        pending.append(chunk)
                        while let newline = pending.firstIndex(of: 0x0a) {
                            take(String(decoding: pending[pending.startIndex..<newline], as: UTF8.self))
                            pending.removeSubrange(pending.startIndex...newline)
                        }
                    }
                    if !pending.isEmpty { take(String(decoding: pending, as: UTF8.self)) }
                    process.waitUntilExit()
                    let stopped = state.withLock { s -> Bool in s.process = nil; return s.stopped }
                    if stopped { outcome.stopped = true; return outcome }
                    if let report, process.terminationStatus == 0 { outcome.report = report; return outcome }
                    if let failure { outcome.failure = failure; return outcome }
                    let death = Death(status: process.terminationStatus, signaled: process.terminationReason == .uncaughtSignal,
                                      stage: stage, route: route)
                    outcome.deaths.append(death)
                    guard outcome.deaths.count <= Child.maximumDeaths,
                          let culprit = Child.culprit(of: death, skipped: Set(outcome.skipped)) else {
                        outcome.failure = ("internal", "the measuring process " + death.description)
                        return outcome
                    }
                    outcome.skipped.append(culprit)
                    onRestart(outcome.skipped)
                }
            }
        }
    }
}

