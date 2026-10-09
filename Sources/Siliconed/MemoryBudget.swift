import Foundation
import Darwin
import Synchronization

/// **The memory a render may count on, read from the machine's real state at its launch** — and the
/// execution settings that derive from it.
///
///     host_statistics64(HOST_VM_INFO64)                 phys_footprint (this process)
///       free − speculative                                       │
///     + purgeable                                                │
///     + external − executable  ── × page ── reclaimable ◀────────┘ (+)
///                                              │
///                                         − reserve ── available ─┬─ Preflight: floor ≤ available, or refused
///                                                                 ├─ the DiT: map tail, staging buffers (#1–#4)
///                                                                 ├─ the text encoder: staging buffers (#5)
///                                   re-read after the denoising ──┴─ the decoders: band size (#10, #11)
///
/// **What swaps is anonymous memory, never the map**: the map is clean and
/// file-backed, the kernel drops its pages without writing them. So what the render may count on is
/// the pages the kernel can hand it without swapping anyone: the free ones, the purgeable ones, and the
/// file-backed ones (others' caches and our own map's), minus the executable ones (the running code,
/// which would be paged straight back in). Plus our own footprint: the measured peaks are a process's
/// whole `phys_footprint`, which includes what this process already holds.
///
/// What a reading by analogy would get wrong:
///
///   - **`free_count` includes `speculative_count`, and so does `external_page_count`.** `vm_stat`
///     prints "Pages free" as `free_count − speculative_count`, and on this machine
///     `File-backed + Anonymous = active + inactive + speculative` to the page: the speculative pages
///     are file read-ahead, counted on both sides. Summing `free + external` counts them twice.
///   - **"free" alone means nothing** (measured: 16 to 68 MB during a render that never swapped); the
///     compressor and the other processes' anonymous pages are never counted.
///   - `os_proc_available_memory` does not exist on macOS (`API_UNAVAILABLE(macos)`), and
///     `kern.memorystatus_level` counts the others' active anonymous memory as available.
///   - **The page cache is not protected by the reserve.** The reserve guards anonymous memory (what
///     can swap). The map's head lives in clean cache: evicted, it costs a re-read, never a swap. So
///     the head is sized on `reclaimable`, the anonymous buffers on `available`.
///
/// **No setting derived here changes a bit of the image** (inventory: proved or same bytes by
/// construction). The ones that do — `mlpBudget`, the prefill's groups, `maxMatrix`,
/// `vaeRequestBlock`, `vaeTile`, the decoder's band alignment — are not read here and stay fixed.
public struct MemoryBudget: Sendable, Equatable {

    /// The bytes the state reading found the render can have: the machine's reclaimable pages plus
    /// this process's own footprint, **before** the reserve.
    public let reclaimable: Int
    /// What is held back for the system and the other applications.
    public let reserve: Int
    /// `true` when `SILICONED_MEMORY_AVAILABLE_GB` replaced the reading.
    public let forced: Bool

    /// **What the render's anonymous memory may reach**: `reclaimable − reserve`.
    public var available: Int { max(0, reclaimable - reserve) }

    /// **The reserve: 1.6 GB** — set so that no render observed without swap is refused: the
    /// smallest `reclaimable` read at the launch of six normal renders (6.12 GB) minus the largest need
    /// measured (4.51 GB, Qwen-Image-2.1's 1024² edit with 3 references), rounded down.
    ///
    /// Measured since (a freshly restarted machine, the refusal judged on the economical
    /// floor): an ordinary desktop at rest (idle apps compressing 3×, 4.05 GB reclaimable) renders
    /// Z-Image 1216×832 without swap. **Its known limit**: under 5–6 GiB of incompressible, active memory
    /// held elsewhere, accepted renders still swap (6.69 GB reclaimable: 570 MB; 6.19 GB: 279 MB, during
    /// step 1 with our footprint at 1.30 GB — the map's page cache against the others, not our memory).
    /// No fixed reserve separates the two (the desktop wants ≤ ~1.97 GB, refusing 6.69 GB wants
    /// ≥ ~3.5): swap comes from the compressor reaching its limit, which this reading does not see.
    package static let reserve = 1_600_000_000

    package init(reclaimable: Int, reserve: Int = MemoryBudget.reserve, forced: Bool = false) {
        self.reclaimable = max(0, reclaimable)
        self.reserve = reserve
        self.forced = forced
    }

    // ── the reading ────────────────────────────────────────────────────────────────────

    /// The page counts the reading uses, from `vm_statistics64`.
    package struct Pages: Sendable, Equatable {
        package var free = 0, speculative = 0, purgeable = 0, external = 0, executable = 0
        package init(free: Int = 0, speculative: Int = 0, purgeable: Int = 0, external: Int = 0, executable: Int = 0) {
            self.free = free; self.speculative = speculative; self.purgeable = purgeable
            self.external = external; self.executable = executable
        }
    }

    /// **The formula**, pure: `page × (free − speculative + purgeable + external − executable) + own`.
    package static func reclaimable(_ p: Pages, pageSize: Int, ownFootprint: Int) -> Int {
        let pages = max(0, p.free - p.speculative) + p.purgeable + max(0, p.external - p.executable)
        return pageSize * pages + ownFootprint
    }

    /// **The executable pages a kernel that does not count them is assumed to hold**: 2 GiB, above
    /// the 1.5 GB measured on the M1 Pro under macOS 27 with an IDE and a browser open, and never more
    /// than the file-backed pages. `executable_count` arrived with `HOST_VM_INFO64` rev5 (the macOS 27
    /// kernel and SDK); before it the term would read 0 and the running code would count as
    /// reclaimable — a reading too generous, which is the one thing the reserve cannot absorb. Too
    /// high an estimate only selects a leaner plan: time, never swap.
    package static let assumedExecutableBytes = 2 << 30

    /// **`vm_statistics64` as the kernel returned it**, read as raw integers: the fields at their
    /// offsets in the structure (they never move from one revision to the next, revisions only append),
    /// so it compiles against any SDK. `count` is what the kernel wrote, in `integer_t`s; a field past
    /// it was not written.
    package static func pages(raw: [integer_t], count: Int, pageSize: Int) -> Pages {
        func u32(_ offset: Int) -> Int { Int(UInt32(bitPattern: raw[offset / 4])) }
        func u64(_ offset: Int) -> Int {
            Int(UInt64(UInt32(bitPattern: raw[offset / 4])) | UInt64(UInt32(bitPattern: raw[offset / 4 + 1])) << 32)
        }
        var p = Pages(free: u32(0), speculative: u32(92), purgeable: u32(88), external: u32(136))
        let executableOffset = 280   // `executable_count`, rev5
        p.executable = count >= executableOffset / 4 + 2 ? u64(executableOffset)
                                                         : min(p.external, assumedExecutableBytes / pageSize)
        return p
    }

    /// The machine's page counts now, or `nil` if the kernel does not answer.
    package static func pages() -> (pages: Pages, pageSize: Int)? {
        var raw = [integer_t](repeating: 0, count: 256)
        var count = mach_msg_type_number_t(raw.count)
        let result = raw.withUnsafeMutableBufferPointer {
            host_statistics64(mach_host_self(), HOST_VM_INFO64, $0.baseAddress!, &count)
        }
        var pageSize: vm_size_t = 0
        guard result == KERN_SUCCESS, host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else { return nil }
        return (pages(raw: raw, count: Int(count), pageSize: Int(pageSize)), Int(pageSize))
    }

    /// **The forcing** (`SILICONED_MEMORY_AVAILABLE_GB`, GiB): what the reading would find —
    /// `reclaimable`, before the reserve — on another machine or in another state. It replaces the
    /// reading itself, not the machine's RAM: "8" simulates a render launched with 8 GiB reclaimable,
    /// hence 8 GiB − 1.6 GB available after the reserve. A tool of measurement and of the tests, read in the
    /// engine and in the app alike.
    package static let forcingVariable = "SILICONED_MEMORY_AVAILABLE_GB"

    package static func forcedReclaimable(_ environment: [String: String]) -> Int? {
        guard let text = environment[forcingVariable],
              let gib = Double(text.replacingOccurrences(of: ",", with: ".")), gib >= 0 else { return nil }
        return Int(gib * 1_073_741_824)
    }

    /// **A forcing of measurement for the reserve** (`SILICONED_MEMORY_RESERVE_GB`, GiB): what 5(e)
    /// renders with to put the machine where the reserve is measured — never a setting.
    package static let reserveVariable = "SILICONED_MEMORY_RESERVE_GB"

    /// The process's environment, read live (`getenv`: a test may set it), for these variables only.
    package static func liveEnvironment(_ names: [String]) -> [String: String] {
        var environment: [String: String] = [:]
        for name in names { if let value = getenv(name) { environment[name] = String(cString: value) } }
        return environment
    }

    /// **The budget now**: the forcing if set, otherwise the machine's state and this process's footprint.
    /// `environment` is the process's own by default, read live.
    public static func current(environment: [String: String]? = nil) -> MemoryBudget {
        let environment = environment ?? liveEnvironment([forcingVariable, reserveVariable])
        let reserve = environment[reserveVariable].flatMap { Double($0.replacingOccurrences(of: ",", with: ".")) }
            .map { Int($0 * 1_073_741_824) } ?? MemoryBudget.reserve
        if let forced = forcedReclaimable(environment) {
            return MemoryBudget(reclaimable: forced, reserve: reserve, forced: true)
        }
        guard let (pages, pageSize) = pages() else {
            // The kernel did not answer: the physical memory, as the preflight used to, is the only figure left.
            return MemoryBudget(reclaimable: Int(ProcessInfo.processInfo.physicalMemory), reserve: reserve)
        }
        return MemoryBudget(reclaimable: reclaimable(pages, pageSize: pageSize, ownFootprint: Arena.processFootprint()),
                            reserve: reserve)
    }

    package var summary: String {
        String(format: "%.2f GB available (%.2f reclaimable − %.2f reserve%@)", Double(available) / 1e9,
               Double(reclaimable) / 1e9, Double(reserve) / 1e9, (forced ? ", forced" : "") as NSString)
    }
}

// ── the pressure, read during the render ─────────────────────────────────────────────────────

/// **How close the compressor is to the point where the kernel starts swapping** — read while the
/// render runs, where the launch reading is blind.
///
/// What the measurements established: the kernel swaps when the compressor's occupied memory passes
/// ≈ 12/22 × (physical − wired) (deduced from the measurements, not read in XNU; −0.45 to +0.24 GB on
/// the sample). And no reading at the launch separates an ordinary desktop (idle apps compress ~3×,
/// renders without swap) from an incompressible ballast (which swaps): incompressibility is only visible
/// once the kernel has compressed. **During a render it has** — it compresses nearly all the others'
/// anonymous memory in the first evaluations. So the render reads how much room the compressor has
/// left, continuously (`MemoryPlan.watchInterval`), and when that room runs out goes lean at its next
/// safe point (`MemoryPlan.relieved`): slower, never other bits.
///
/// This is not an "early stop" (refuted: the compressor can climb 1.4 GB in a second, no stop
/// guarantees zero swap). Nothing stops; the render changes how it reads its map and decodes.
package struct MemoryPressure: Sendable, Equatable {
    /// The compressor's occupied memory (`compressor_page_count`), bytes.
    package var compressor: Int
    /// The machine's wired memory (`wire_count`), bytes — ours included (arenas wrapped for the GEMMs,
    /// MPSGraph's allocations).
    package var wired: Int
    package var physical: Int

    package init(compressor: Int, wired: Int, physical: Int) {
        self.compressor = compressor; self.wired = wired; self.physical = physical
    }

    /// **The measured limit**: 12/22 × (physical − wired).
    package var limit: Int { (physical - wired) / 22 * 12 }
    /// The room the compressor has left before the kernel swaps; negative past the limit.
    package var headroom: Int { limit - compressor }

    /// **Below this room, the render goes lean.** 1.5 GB: the largest measured error on the limit (0.45 GB)
    /// plus one second of the fastest measured climb (6.25 → 7.64 GB). **Not a guard by itself**:
    /// on the ordinary desktop it never fires; under an incompressible ballast, read only at the
    /// evaluations' starts, it fired too late (step 3) or never — the room swings by several GB within a
    /// second between two readings. Hence the watcher (`MemoryPlan.watchInterval`), which latches the
    /// first short reading, and a lean plan that takes the whole map out of the cache instead of
    /// handing back the buffers (~0.7 GB, less than a first full-size step adds, ~1.8 GB — and what
    /// put the whole map back into the cache).
    package static let margin = 1_500_000_000

    /// **A forcing of measurement** (`SILICONED_MEMORY_HEADROOM_GB`, GiB): the room the reading would
    /// find. `GIB@SECONDS` forces it only from that many seconds into the render (the machine's reading
    /// before): how a lean plan triggered in the middle of a render is provoked. Never a setting.
    package static let forcingVariable = "SILICONED_MEMORY_HEADROOM_GB"

    /// The forced room in bytes, `nil` when the forcing is absent, malformed, or not yet due at
    /// `elapsed` seconds into the render (a delayed forcing outside a render is never due).
    package static func forcedRoom(_ text: String?, elapsed: Double?) -> Int? {
        guard let text else { return nil }
        let parts = text.split(separator: "@", maxSplits: 1).map { $0.replacingOccurrences(of: ",", with: ".") }
        guard let gib = Double(parts[0]) else { return nil }
        if parts.count == 2 {
            guard let from = Double(parts[1]), let elapsed, elapsed >= from else { return nil }
        }
        return Int(gib * 1_073_741_824)
    }

    /// **The pressure now**: the forcing if set; `nil` under a forced budget (a measurement simulating
    /// another machine reads no real pressure) or if the kernel does not answer — then nothing changes.
    /// `elapsed`: seconds since the render's start, for a delayed forcing.
    package static func current(environment: [String: String]? = nil, elapsed: Double? = nil) -> MemoryPressure? {
        let environment = environment ?? MemoryBudget.liveEnvironment([forcingVariable, MemoryBudget.forcingVariable])
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        if let room = forcedRoom(environment[forcingVariable], elapsed: elapsed) {
            // A machine with nothing wired whose compressor sits `room` below the limit.
            let limit = MemoryPressure(compressor: 0, wired: 0, physical: physical).limit
            return MemoryPressure(compressor: limit - room, wired: 0, physical: physical)
        }
        guard environment[MemoryBudget.forcingVariable] == nil else { return nil }
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        var pageSize: vm_size_t = 0
        guard result == KERN_SUCCESS, host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else { return nil }
        return MemoryPressure(compressor: Int(stats.compressor_page_count) * Int(pageSize),
                              wired: Int(stats.wire_count) * Int(pageSize), physical: physical)
    }
}

// ── the settings derived from the budget ─────────────────────────────────────────────────────

/// **The execution settings of one render, derived from its budget** — beside `EngineSettings.effective`,
/// which is read once per process. Pure functions (the tests drive them with simulated budgets), and
/// the state of the render in progress (`during`), which the DiTs, the text encoder and the decoders
/// read where they decide.
///
/// Outside a render (a check, a bench) there is no budget: each site keeps the reference values it had
/// before this file — the measurement tools stay what they were. Inside one, `SILICONED_MAP_TAIL` set in
/// the environment is still a forcing of measurement: the DiT then streams that fraction with two
/// buffers, at every size, whatever the budget and the pressure.
package enum MemoryPlan {

    // #1–#4 · the map's tail, and its staging buffers

    /// The DiT's plan for its map: the share of its layers streamed around the page cache, and how
    /// many staging buffers of its largest block.
    package struct Tail: Sendable, Equatable {
        package var fraction: Double
        package var slots: Int
        package static let none = Tail(fraction: 0, slots: 0)
        /// **The lean plan's map: all of it beside the cache, with its two buffers**.
        package static let lean = Tail(fraction: 1, slots: 2)
    }

    /// **The tail, from the budget.** Measured mechanism: a map larger than the page cache
    /// misses every page under LRU; streaming its tail leaves the cache to its head.
    ///
    /// - **the default plan**: the measured fraction (0.5, `EngineSettings.mapTail`) with two staging
    ///   buffers, when the room above the floor holds them (`available − floor`, one block each). The
    ///   fraction is not derived: a head sized on the reading (`reclaimable − floor − buffers`) gave
    ///   0.43–0.73 on this machine and read 6.6–8.6 GB per evaluation against ~5.4 at 0.5 — the page
    ///   cache holds less of the head than the reading says. Deriving it again waits for a timed
    ///   comparison in series; so does turning the tail off for a map that fits whole (8-bit maps);
    /// - **otherwise the lean plan: the whole map streamed, still with its two buffers** (`Tail.lean`),
    ///   no longer none. A budget too short for the buffers is a page cache too short for the head:
    ///   kept, the head is evicted by the others and read back through the cache at every evaluation.
    ///   And the map through the cache is what swaps: 12.33 GB per step of file pages
    ///   against incompressible anonymous memory elsewhere, the kernel compresses the others rather
    ///   than drop the cache — 234.7 GB through the compressor in a swapping render, 13.1
    ///   with the whole map streamed, which rendered without swap where the default swapped. The
    ///   two buffers (~0.72 GB at Z-Image) cost less than the cache they spare; without a ballast the
    ///   whole map costs nothing measurable at 1024² (84.6 against 85.8 s), but 512² read more slowly
    ///   at 1 than at 0.5 (40.5 against 38.1 s) — hence not the default.
    ///   **Never one buffer**: measured, one serializes the reads behind the computation
    ///   (Z-Image 512² 14.4 s per step against 4.1 with two; Qwen 512 20 s against 6), the same bits;
    /// - `.none` only when the fraction is 0 (a `map_tail = 0` setting).
    ///
    /// Same bits whatever the fraction and the buffers (`9beccc5b`, `078b9f41` on the whole sweep).
    package static func tail(budget: MemoryBudget, floor: Int, blockBytes: Int,
                             fraction: Double = EngineSettings.effective.mapTail) -> Tail {
        guard blockBytes > 0, fraction > 0 else { return .none }
        let slots = 2
        guard budget.available - floor >= slots * blockBytes else { return .lean }
        return Tail(fraction: fraction, slots: slots)
    }

    // #5 · the text encoder's staging buffers

    /// The text encoder's reference: four buffers (three layers ahead).
    package static let encoderSlots = 4

    /// **The text encoder's buffers**: four when the room left holds them, otherwise none —
    /// every layer through the map, as before the buffers (Qwen3-VL text 4.63 s against 3.1–3.4). Nothing in
    /// between: one buffer was measured worse than none on the DiT (`tail`), two and three never.
    /// `footprint` is the process's, measured as the encoder starts — its arena already counted. Same
    /// bytes read either way.
    package static func encoderSlots(budget: MemoryBudget, footprint: Int, blockBytes: Int) -> Int {
        guard blockBytes > 0 else { return encoderSlots }
        return budget.available - footprint >= encoderSlots * blockBytes ? encoderSlots : 0
    }

    // #10, #11 · the decoders' bands

    /// **A decoder's band, from the budget read after the denoising**: the measured size (FLUX's 2²³
    /// elements; Qwen's 2¹⁸ pixels) when that budget covers the default plan's peak
    /// (`floor` here is `Render.comfortable`: at the economical floor the band is the smallest), the smallest
    /// proved size otherwise (FLUX 2²⁰; Qwen: every size down to 2 rows). Never larger:
    /// 2²⁴ costs 140 MB more for the same time. The band's alignment is never derived (measured:
    /// an odd window changes the bits).
    package static func band(budget: MemoryBudget, floor: Int, standard: Int, minimum: Int) -> Int {
        budget.available >= floor ? standard : min(standard, minimum)
    }

    /// FLUX's decoder (Z-Image, klein, ERNIE): the smallest band proved and timed (7.3 s at 1024² against 4.4).
    package static let fluxMinimumBand = 1 << 20
    /// Qwen-Image-2.1's decoder: a quarter of the measured band (bits proved down to 2 rows; time not measured).
    package static let qwenMinimumBand = 1 << 16

    // ── the render in progress ─────────────────────────────────────────────────────────────

    /// The budget of the render in progress, its floor, and what was decided from them.
    package struct Render: Sendable {
        package var budget: MemoryBudget
        package var floor: Int
        /// The default plan's peak (`ModelCard.memoryComfort`): the decoders' measured band needs it.
        package var comfortable: Int
        /// The budget read again after the denoising, for the decoding.
        package var decoding: MemoryBudget?
        /// What each site decided, in order, for the report (the developer's command line prints it).
        package var decisions: [String] = []
        /// **Once the compressor ran short of room, the render stays lean to its end** (`relieved`):
        /// a plan that comes back at the next evaluation would put the map back into the cache.
        package var relieved = false
        /// Where the pressure is read: the machine's by default, injected by the tests.
        package var pressure: @Sendable () -> MemoryPressure? = { MemoryPressure.current() }
        /// Which render this is: the watcher of a render that ended never latches the next one.
        fileprivate var generation: UInt64 = 0
        fileprivate var start = Date()
    }

    private static let state = Mutex<Render?>(nil)
    private static let last = Mutex<Render?>(nil)
    private static let generations = Atomic<UInt64>(0)

    /// **The watcher's period: 0.25 s** — the pressure read on a thread of its own for the whole
    /// render, not only where a site decides. Measured: read at the evaluations' starts, the room had
    /// gone under the margin 11.7 s before step 3 read it, or swung back up between two readings while
    /// the kernel swapped; the compressor was seen climbing 1.4 GB in a second. The first short reading is
    /// latched (`Render.relieved`), the plan changes at the next safe point (an evaluation's start, the
    /// encoder's, the decoding's). The period of `SILICONED_MEMORY_TRACE` and of the measuring probe (0.25–0.5 s);
    /// a reading is one `host_statistics64`, a few microseconds.
    package static let watchInterval = 0.25

    /// The render in progress, or `nil` outside one.
    package static var current: Render? { state.withLock { $0 } }
    /// The last render's plan, once it is over (or while it runs).
    package static var lastRender: Render? { last.withLock { $0 } ?? current }

    /// Wires a render's budget in for the duration of `body` — the engine, around a render (renders are
    /// serialized: the engine queue and `RenderLock.machine`).
    /// `watch`: the watcher's period (`watchInterval`; the tests shorten it).
    package static func during<T>(budget: MemoryBudget, floor: Int, comfortable: Int? = nil,
                                  pressure: (@Sendable () -> MemoryPressure?)? = nil,
                                  watch: Double = MemoryPlan.watchInterval,
                                  _ body: () throws -> T) rethrows -> T {
        var render = Render(budget: budget, floor: floor, comfortable: max(floor, comfortable ?? floor))
        let start = Date()
        render.start = start
        render.pressure = pressure ?? { MemoryPressure.current(elapsed: Date().timeIntervalSince(start)) }
        render.generation = generations.add(1, ordering: .relaxed).newValue
        let generation = render.generation
        state.withLock { $0 = render }
        let watcher = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        watcher.schedule(deadline: .now() + watch, repeating: watch, leeway: .milliseconds(Int(watch * 200)))
        watcher.setEventHandler { _ = relieved(nil, generation: generation) }
        watcher.resume()
        defer {
            watcher.cancel()
            let ended = state.withLock { s -> Render? in let r = s; s = nil; return r }
            last.withLock { $0 = ended }
        }
        return try body()
    }

    /// **The decoding's budget: read again after the denoising** — the best point (inventory): the DiT
    /// is gone, its arenas returned, and the machine has had a whole denoising to change.
    package static func readForDecoding(environment: [String: String]? = nil) {
        state.withLock { $0?.decoding = MemoryBudget.current(environment: environment) }
    }

    /// **The lean plan, forced** (`SILICONED_MEMORY_PLAN=economical`): the whole map streamed with its
    /// two buffers (`Tail.lean`), no encoder buffer, the smallest decoder band — what a short budget at
    /// the launch or a short compressor during the render leads to. A measuring tool; the bits are the
    /// same. With `SILICONED_MAP_TAIL=0` beside it, the economical plan (no buffer at all, the map through the
    /// cache), on which the floors of `ModelCard.measuredPeaks` were taken.
    package static let planVariable = "SILICONED_MEMORY_PLAN"
    package static var economical: Bool {
        MemoryBudget.liveEnvironment([planVariable])[planVariable] == "economical"
    }

    package static func note(_ decision: String) {
        state.withLock { $0?.decisions.append(decision) }
    }

    /// `note`, unless it is already the last decision — a site asked at every evaluation.
    package static func noteOnce(_ decision: String) {
        state.withLock { s in if s?.decisions.last != decision { s?.decisions.append(decision) } }
    }

    /// **Has the compressor run short of room during this render?** (`MemoryPressure`). Asked where a
    /// site can still change its plan — every DiT evaluation, the encoder's start, the decoding — and
    /// read meanwhile by the watcher (`watchInterval`, `site` nil); sticky: once short, the rest of the
    /// render is lean. Outside a render, `false`.
    package static func relieved(_ site: String) -> Bool { relieved(site, generation: nil) }

    private static func relieved(_ site: String?, generation: UInt64?) -> Bool {
        guard let render = current, generation == nil || render.generation == generation else { return false }
        if render.relieved { return true }
        guard let pressure = render.pressure(), pressure.headroom < MemoryPressure.margin else { return false }
        let latched = state.withLock { s -> Bool in
            guard s?.generation == render.generation, s?.relieved == false else { return false }
            s?.relieved = true
            return true
        }
        guard latched else { return current?.relieved ?? false }
        let at = site.map { "at the \($0)" }
            ?? String(format: "%.1f s into the render", Date().timeIntervalSince(render.start))
        note(String(format: "compressor short of room %@ (%.2f GB left before its limit of %.2f): the render goes lean",
                    at as NSString, Double(pressure.headroom) / 1e9, Double(pressure.limit) / 1e9))
        return true
    }

    // ── what the sites ask ─────────────────────────────────────────────────────────────────

    /// **The DiT's tail**, where it decides (`DiT.forward`, `QwenImage21DiT.pass`), at every evaluation:
    /// the first released layer and the staging buffers. The default plan's half (`tail`) while the
    /// budget holds the buffers and the compressor has room; **the whole map with its two buffers
    /// once either runs short** (`Tail.lean`) — at the launch, or between two evaluations,
    /// which changes which bytes go through the cache and never the bytes. Outside a render, or under
    /// `SILICONED_MAP_TAIL` (a forcing of measurement): that fraction, two buffers, at every size
    /// (the former guard, a tail only up to the area of a 1024², is lifted: its 1024×1536 swap was the
    /// decoder's). The decision is journaled in every case, the forcing's included.
    package static func ditTail(artifact: Artifact, layerBlocks: [[String]]) -> (released: Int, slots: Int) {
        let layers = layerBlocks.count
        let block = layerBlocks.map { artifact.span(of: $0) }.max() ?? 0
        func decide(_ tail: Tail, _ why: String) -> (released: Int, slots: Int) {
            let released = tail.slots == 0 ? layers : Prefetcher.firstReleasedLayer(of: layers, tail: tail.fraction)
            let streamed = released < layers ? tail.slots : 0
            noteOnce(String(format: "map tail %.2f (%d of %d layers), %d buffer(s) of %.0f MB%@", tail.fraction,
                            layers - released, layers, streamed, Double(block) / 1e6, why as NSString))
            return (released, tail.slots)
        }
        let settings = EngineSettings.effective
        if settings.provenance["map_tail"] == .environment {
            return decide(Tail(fraction: settings.mapTail, slots: 2), " (forced: SILICONED_MAP_TAIL)")
        }
        guard let render = current else { return decide(Tail(fraction: settings.mapTail, slots: 2), "") }
        if economical { return decide(.lean, " (lean plan, forced)") }
        if relieved("denoising") { return decide(.lean, " (lean plan: compressor short of room)") }
        let tail = tail(budget: render.budget, floor: render.floor, blockBytes: block)
        return decide(tail, tail == .lean ? " (lean plan: no room for the head)" : "")
    }

    /// **The text encoder's buffers**, as it starts. Outside a render, the reference (4).
    package static func encoderSlots(artifact: Artifact, layerBlocks: [[String]]) -> Int {
        guard let render = current else { return encoderSlots }
        let block = layerBlocks.map { artifact.span(of: $0) }.max() ?? 0
        let slots = economical || relieved("text encoding") ? 0 : encoderSlots(budget: render.budget, footprint: Arena.processFootprint(), blockBytes: block)
        note(String(format: "text encoder %d buffer(s) of %.0f MB", slots, Double(block) / 1e6))
        return slots
    }

    /// **A decoder's band**, as it starts decoding. Outside a render (or before the budget was read
    /// again), `standard` — the static a check may have set.
    package static func decoderBand(standard: Int, minimum: Int, unit: String) -> Int {
        guard let render = current, let decoding = render.decoding else { return standard }
        // The decoding holds the peak of most formats: the last place the render can give back.
        let band = economical || relieved("decoding") ? min(standard, minimum)
            : band(budget: decoding, floor: render.comfortable, standard: standard, minimum: minimum)
        note("decoder bands of 2^\(band.trailingZeroBitCount) \(unit) (budget at decoding: \(decoding.summary))")
        return band
    }
}

extension Artifact {
    /// The bytes a run of tensors spans in the file — a streamed block's staging size.
    package func span(of names: [String]) -> Int {
        let ranges = names.compactMap { tensors[$0] }.map { ($0.offset, $0.offset + $0.bytes) }
        guard let first = ranges.map(\.0).min(), let end = ranges.map(\.1).max() else { return 0 }
        return end - first
    }
}
