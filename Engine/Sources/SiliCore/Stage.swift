import CoreAI
import Foundation
import Metal

/// **A stage: a loop of fixed functions over an arena, planned under B before anything is allocated.**
///
///     the client    declares its stage (`Declaration`): its tensors, its weights by groups, the order its loop uses
///                   them in, its functions
///     `plan`        P = the process's floor + the socle's arena (the heap to the byte, what is apart, the surfaces:
///                   the tensors, the residents, the groups the ring cannot hold, the ring's slots), refused where
///                   P + 64 MiB passes B. Core AI's share is not on paper: it is read in flight (`signal`)
///     `choose`      of a client's forms (taller calls or bands first), each planned, the first whose P and what Core
///                   AI will hold under its functions on this Mac (read, or bounded from their kin's readings) hold
///                   under B; none, the soberest on paper where nothing bounds it from below, else refused
///     `Stage`       the arena allocated, the residents read, the ring's reader started (the groups kept from the start
///                   read ahead in the loop's order, the others through the slots), the functions loaded; views by
///                   name, groups acquired and released in the loop's order; past the loop's signal (`signal`), the
///                   cycle's groups kept in flight while what Core AI is read to hold leaves room under B; `close` lets
///                   everything go, waits for the footprint and keeps what this stage's functions were read to hold,
///                   if called
///
/// The engine knows tensors, groups, functions and bytes; what they compute is the client's.
public final class Stage: @unchecked Sendable {
    public let plan: Plan
    public let device: MTLDevice
    /// The stage's own queue and stream: let go with it (`close`).
    public private(set) var queue: MTLCommandQueue!
    public private(set) var stream: SerialStream!
    public private(set) var arena: Arena!
    /// The functions loaded, by the declaration's names.
    public private(set) var functions: [String: InferenceFunction] = [:]
    /// The residents, by their arguments' names.
    public private(set) var residents: [String: Placed] = [:]

    private var at: [String: Int] = [:]
    var ring: Ring?
    /// **What this stage's functions hold, as read**: this Mac's key for them (`key`), what it last read past the first
    /// signal (`reserve`; nil, never), the footprint as the stage reads it (the process's, unless a test says otherwise).
    public let key: String
    public internal(set) var reserve: Int?
    let footprint: @Sendable () -> Int
    let measured: ThisMac
    /// Under `lock`: Core AI's share read so far at its largest — the footprint past the floor and the arena —, and at
    /// the first signal; whether the reader keeps (`signal`).
    var largest = Int.min, first: Int?, keeping = false
    /// Under `lock`, past a climb (`rekey`): the turn its functions are first called in, the share then, and the growth
    /// it expects.
    var climbed: (turn: Int, share: Int, expects: Int)?
    /// Under `lock`: the groups the reader decided to keep whose heap is not allocated yet (`keeps`, `Arena.grow`) —
    /// counted by every decision against B meanwhile, a climb's as the reader's.
    var promised = 0
    /// Under `lock`: the stage opened whole (its functions loaded); a stage whose opening threw keeps no reading.
    private var ready = false
    /// **The share at its peak, read every millisecond**: a call's working memory is taken and given back within the
    /// call, between two of the stage's readings (20/11: the share read at the uses 1 826 MiB at most, the fuse's
    /// footprint 156 MiB above it under the same arena, at B = 4). Under `lock`, the loads under way (`loading`): what a
    /// function's loading takes is the load's, not the calls', and none is read or kept meanwhile.
    var watch: FootprintWatch?
    var loads = 0
    /// Under `lock`: the key the stage's reading is kept under (its own, or the functions a client loaded in flight:
    /// `rekey`), what those functions hold (`Extent`, kept beside the reading), and the share at its largest since.
    public internal(set) var readingKey: String
    var readingExtent: Extent?
    var keyed = 0
    /// The reader's staging in the arena.
    private var staging: Int?
    var use = 0
    /// What its close waited for, once closed (`Stages` told once): a stage may be closed again by its owner's `deinit`.
    private var settled: Settled?
    /// Its stream's identity, which a call on it marks (`called`).
    var streamID: ObjectIdentifier?
    let lock = NSLock()
    /// The footprint before the stage allocated: what `close` waits for.
    public let before: Int
    /// When it began allocating (`Stages`).
    private let began = DispatchTime.now().uptimeNanoseconds

    /// **The stage of `plan`**: the arena allocated (its tensors zeroed), the residents read, the groups kept from the
    /// start placed and read ahead in the loop's order by the ring's reader (which reads the others into its slots), the
    /// functions loaded while it reads. `files`: the plan's, opened again past the system's
    /// cache for the reader. `measured`: where the reserve is read (`ThisMac.reserves`).
    public init(_ device: MTLDevice, plan: Plan, files: [SafeTensors], measured: ThisMac = .current,
                footprint: @escaping @Sendable () -> Int = Memory.footprint) async throws {
        self.plan = plan
        self.device = device
        self.footprint = footprint
        self.measured = measured
        key = Self.key(plan.declaration)
        readingKey = key
        readingExtent = plan.declaration.extent
        reserve = measured.reserves?[key]
        before = footprint()
        loads = 1
        defer { lock.withLock { loads -= 1 } }
        opened(key)
        Stages.tell(.opened(plan))
        let d = plan.declaration
        guard let queue = device.makeCommandQueue() else { throw Failure("no command queue") }
        self.queue = queue
        stream = SerialStream(commandQueue: queue)
        listen(stream)
        let arena = try Arena(device, bytes: plan.heap + plan.apart, apart: plan.apart)
        self.arena = arena
        for t in d.tensors {
            let key = switch t.place {
            case .heap: arena.take(t.bytes)
            case .apart: arena.take(apart: t.bytes)
            case .surface(let shape, let element): arena.take(surface: shape, element: element)
            case .units(let width, let height, let element): arena.take(units: t.bytes, width: width, height: height, element: element)
            }
            at[t.name] = key
            memset(arena.pointer(key), 0, t.size)
        }
        let surfaces = d.surfaces, sizes = plan.sizes
        // The staging: one, the reader's, the residents read through it before the reader starts.
        let staging = sizes.staging
        let stagingKey = staging > 0 ? arena.take(staging) : nil
        self.staging = stagingKey
        for (file, entries) in try Self.byFile(d.residents, files) {
            let placed = Weights.place(try Weights.layout(entries, in: file), in: arena, surfaces: surfaces)
            try Weights.read(entries, from: file, into: placed, arena: arena, staging: arena.pointer(stagingKey!))
            residents.merge(placed) { $1 }
        }
        // The kept groups' places, the ring's slots.
        var groupFile: [String: Int] = [:]
        for (g, entries) in d.groups {
            let f = try Self.file(of: entries, g, files)
            groupFile[g] = files.firstIndex { $0 === f }!
        }
        var layouts: [String: [Weights.Slot]] = [:]
        for (g, entries) in d.groups { layouts[g] = try Weights.layout(entries, in: files[groupFile[g]!]) }
        var bound: [String: [String: Placed]] = [:]
        for g in plan.kept { bound[g] = Weights.place(layouts[g]!, in: arena, surfaces: surfaces) }
        var slots: [[String: Placed]] = []
        if plan.depth > 0, let s = d.slot { slots = (0..<plan.depth).map { _ in Weights.place(layouts[s]!, in: arena, surfaces: surfaces) } }
        if !plan.kept.isEmpty || !slots.isEmpty {
            let uncached = try files.map { try $0.reopened(uncached: true) }
            let groups = d.groups, fileOf = groupFile, laid = layouts
            // Past the signal, a group of the cycle that rides the ring may be kept: in a heap of its own, or its own
            // surfaces (`Arena.grow`).
            let keeps = slots.isEmpty ? [] : Set(d.cycle.filter { !plan.kept.contains($0) && sizes.rides($0) })
            let slotBytes = slots.isEmpty ? 0 : sizes.slot
            ring = Ring(slots: slots, once: d.once, cycle: d.cycle, bound: bound, keeps: keeps, place: { [weak self] g in
                defer { self?.delivered(sizes.groups[g]!) }
                let heap = surfaces ? 0 : try arena.grow(bytes: sizes.groups[g]!)
                return Weights.place(laid[g]!, in: arena, heap: heap, surfaces: surfaces)
            }, keep: { [weak self] g, unwritten in
                self?.keeps(sizes.groups[g]!, pending: unwritten * slotBytes, allocated: arena.allocated) ?? false
            }, kept: { [fromStart = plan.kept.count] in Stages.tell(.kept(fromStart + $0)) }, fill: { g, place in
                try Weights.read(groups[g]!, from: uncached[fileOf[g]!], into: place, arena: arena, staging: arena.pointer(stagingKey!))
            })
        }
        let floor = plan.floor
        watch = FootprintWatch { [weak arena] in footprint() - floor - (arena?.allocated ?? 0) }
        for (name, f) in d.functions.sorted(by: { $0.key < $1.key }) {
            functions[name] = try await Functions.load(try Functions.asset(f.asset, f.build), unit: f.unit)
        }
        watch?.reset()
        lock.withLock { ready = true }
    }

    /// **Functions loaded in flight** (`body`: a client's taller calls, another share of the ANE): nothing read or kept
    /// meanwhile, the peak read from its end.
    public func loading<T>(_ body: () async throws -> T) async rethrows -> T {
        lock.withLock { loads += 1 }
        defer {
            watch?.reset()
            lock.withLock { loads -= 1 }
        }
        return try await body()
    }

    /// A stage let go by its owner without `close` (a stage of bands', with its pass) keeps its readings all the same.
    deinit {
        ring?.stop()
        watch?.stop()
        opened(nil)
        if lock.withLock({ settled == nil && ready }) { keep() }
        listen(nil)
    }

    /// **What this stage read, kept on this Mac**, at its largest: the share read since its key, the reading of the
    /// functions it names (`ThisMac.readings`); what Core AI's share grew past the first signal, the reserve of its
    /// functions (`ThisMac.reserves`) — not past a climb: a reserve counts from a stage's first signal on its functions,
    /// where a stage opened on them gives it (20/11: read past a climb from a signal of the main pass, 31 MiB; opened on
    /// the same functions, the first signal in a pass of keys before it, the share grew 194 MiB past the margin and the
    /// fuse stopped a run at B = 10). At the close or the stage let go, and wherever its client knows every function was
    /// called (`Bands`, a pass done: a client may hold its stage until the process ends). A stage none of whose functions
    /// was called since its opening or its last climb keeps nothing (`called`).
    public func keep() {
        guard called() else { return }
        let (share, _, _) = read()
        let (keyed, readingKey, extent, first, climbed) = lock.withLock { (self.keyed, self.readingKey, self.readingExtent, self.first, self.climbed != nil) }
        Self.keep(reading: keyed, under: readingKey, extent: extent, budget: plan.budget)
        if let first, !climbed {
            let reading = max(share - first, 0), key = readingKey
            try? ThisMac.update { $0.reserves = ($0.reserves ?? [:]).merging([key: reading]) { max($0, $1) } }
        }
    }

    /// Functions let go before the stage is: what their client no longer calls (Core AI's memory of a function goes
    /// with it).
    public func drop(_ names: [String]) { for n in names { functions[n] = nil } }

    /// The tensor `name` from byte `offset` as a function reads it: its surface (what the ANE reads) only whole, from
    /// its first byte; past it, the GPU's buffer over the same bytes. Part of the tensor from a page, a buffer of its own
    /// over its bytes, to its last element's end (`Arena.part`).
    public func view(_ name: String, _ shape: [Int], _ type: NDArray.ScalarType, strides: [Int] = [], bytes offset: Int = 0) -> View {
        guard let key = at[name] else { preconditionFailure("\(plan.declaration.name): no tensor \(name)") }
        let buffer = arena.buffer(key), steps = strides.isEmpty ? Layout.rowMajor(shape) : strides
        // A surface's rows lie packed (`Arena.take(surface:)`): a view of it reads them so or not at all.
        precondition(arena.surface(key) == nil || steps == Layout.rowMajor(shape), "\(name): a surface viewed with strides \(steps)")
        // Its rows padded, the last one's padding too: what Core AI asks of it (`minimumByteCount`).
        let extent = max((zip(shape, steps).map { ($0 - 1) * $1 }.reduce(0, +) + 1), (shape.first ?? 1) * (steps.first ?? 1)) * Layout.element(type)
        // A surface's part (`Arena.part`) keeps the surface from its start: the GPU reads the part, the ANE the surface.
        if offset + extent < buffer.length || offset > 0, let part = arena.part(key, from: offset, bytes: extent) {
            return View(buffer: part, type: type, shape: shape, strides: strides, surface: offset == 0 ? arena.surface(key) : nil)
        }
        return View(buffer: buffer, offset: offset, type: type, shape: shape, strides: strides, surface: offset == 0 ? arena.surface(key) : nil)
    }

    /// The tensor `name` as the ANE reads it: its surface from byte `offset` (a unit of `Arena.take(units:)`, or a
    /// surface whole from 0).
    public func neuralView(_ name: String, _ shape: [Int], _ type: NDArray.ScalarType, bytes offset: Int = 0) -> View {
        guard let key = at[name], let s = arena.surface(key) else { preconditionFailure("\(plan.declaration.name): \(name) is no surface") }
        return View(buffer: arena.buffer(key), offset: offset, type: type, shape: shape, surface: s)
    }

    /// A weight placed (a resident, a group's) as a function reads it, at `shape` if given (the same bytes at another
    /// rank: `[k, n]` codes read as `[1, k, n]`).
    public func view(_ p: Placed, _ shape: [Int]? = nil) -> View {
        precondition(shape == nil || shape!.reduce(1, *) == p.shape.reduce(1, *), "a weight of \(p.shape) viewed as \(shape!)")
        return View(buffer: arena.buffer(p.offset), type: p.type, shape: shape ?? p.shape, surface: arena.surface(p.offset))
    }

    /// The bytes of the tensor `name`, for the CPU to write or read.
    public func pointer(_ name: String) -> UnsafeMutableRawPointer {
        guard let key = at[name] else { preconditionFailure("\(plan.declaration.name): no tensor \(name)") }
        return arena.pointer(key)
    }

    /// A group's weights while the loop holds them: kept (`use` nil) or a slot of the ring; `turn`, the loop's use.
    public struct Held: @unchecked Sendable {
        public let group: String
        public let weights: [String: Placed]
        public let use: Int?
        public let turn: Int
    }

    /// **The group's weights, once read**: kept, or the ring's next use, which must be the loop's next (the order
    /// declared: `once`, then `cycle` over and over). Released (`release`) once the last call that reads it is done.
    public func acquire(_ group: String) throws -> Held {
        guard let ring else { preconditionFailure("\(plan.declaration.name): \(group) neither kept nor in a ring") }
        let m = lock.withLock { () -> Int in defer { use += 1 }; return use }
        read()
        let (weights, n) = try ring.acquire(m, group)
        return Held(group: group, weights: weights, use: n, turn: m)
    }

    /// **The loop's use `turn`, ahead of its acquisition**: its group and its weights once read (`Ring.peek`), the loop's
    /// own order untouched — what the CPU prepares for a unit while the group before computes. Nil past the loop, or
    /// once `stopPeeking`.
    public func upcoming(_ turn: Int) -> (group: String, weights: [String: Placed])? {
        guard let ring, let g = ring.group(turn), let w = ring.peek(turn) else { return nil }
        return (g, w)
    }

    /// Every `upcoming` under way returns nil, and none waits again: what prepares ahead lets the stage close.
    public func stopPeeking() { ring?.stopPeeking() }

    /// The held group done: its slot may take the ring's next read (a kept group stays).
    public func release(_ held: Held) {
        if let n = held.use { ring?.release(n) }
    }

    /// The groups kept so far, from the start or past the signal, in the order read.
    public var kept: [String] { ring?.keptGroups ?? plan.kept }

    /// What a stage let go left: the footprint against what it was before the stage allocated, the seconds waited, the
    /// arenas still alive and their heaps still held.
    public struct Settled: Sendable {
        public let footprint, before: Int
        public let seconds: Double
        public let alive, held: Int
        public var line: String {
            (footprint >= before ? "+" : "") + Memory.mib(footprint - before) + String(format: " after %.2f s", seconds)
                + (alive + held == 0 ? "" : ", arenas alive \(Memory.mib(alive)), held \(Memory.mib(held))")
        }
    }

    /// **Everything let go** — the reader stopped, the arena, then the functions, the queue and the stream — then the
    /// wait until the footprint is back within 64 MiB of what it was before the stage, 2 s at most (measured 18/11 on the
    /// M1 Pro, a run's stages each let go: the heap leaves within 0.05 to 0.5 s, then 100 to 160 MiB of Core AI's or
    /// Metal's 1 to 2 s after; a function holds its last call's buffers until it is let go, `Arena.take(apart:)`).
    /// **The arena before the functions that read it**: a heap let go after them stays billed to the process 12 to 36 s,
    /// before them 0.06 s (14–15/11 on the M1 Pro, the graphics ledger; the int8 trial's arena likewise). Past the wait it
    /// would lie in the next stage's floor, and leave while that stage reads Core AI's share, its reading short by as
    /// much. On the M5 Pro either order leaves within the wait (21/11/2026).
    @discardableResult
    public func close() -> Settled {
        // Closed again (its owner's `deinit`): what the first close waited for, not a second wait.
        lock.lock(); let done = settled; lock.unlock()
        if let done { return done }
        let kept = self.kept.count
        ring?.stop()
        ring = nil
        opened(nil)
        keep()
        listen(nil)
        watch?.stop()
        watch = nil
        residents = [:]
        arena = nil
        functions = [:]
        stream = nil
        queue = nil
        let t = DispatchTime.now().uptimeNanoseconds
        var f = Memory.footprint()
        while f > before + Stage.margin && Functions.seconds(since: t) < 2 { usleep(2000); f = Memory.footprint() }
        let settled = Settled(footprint: f, before: before, seconds: Functions.seconds(since: t), alive: Arena.alive, held: Arena.lingering)
        Stages.tell(.closed(plan, seconds: Functions.seconds(since: began), kept: kept))
        lock.lock(); self.settled = settled; lock.unlock()
        return settled
    }
}
