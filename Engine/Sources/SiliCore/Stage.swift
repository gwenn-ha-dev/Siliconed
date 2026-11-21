import CoreAI
import CryptoKit
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
///                   AI was read to hold under its functions on this Mac hold under B; none read, the soberest
///     `Stage`       the arena allocated, the residents read, the ring's reader started (the groups kept from the start
///                   read ahead in the loop's order, the others through the slots), the functions loaded; views by
///                   name, groups acquired and released in the loop's order; past the loop's signal (`signal`), the
///                   cycle's groups kept in flight while what Core AI is read to hold leaves room under B; `close` lets
///                   everything go, waits for the footprint and keeps what this stage's functions were read to hold
///
/// The engine knows tensors, groups, functions and bytes; what they compute is the client's.
public final class Stage: @unchecked Sendable {
    // MARK: - The declaration

    /// Where a tensor lies: in the heap; apart, in a buffer of its own (a stage's exit, what its last call writes:
    /// `Arena.take(apart:)`); a surface of `shape`, `element` bytes each, for the ANE (`Arena.take(surface:element:)`).
    public enum Place: Sendable, Equatable {
        case heap
        case apart
        case surface(shape: [Int], element: Int)
        /// A per-row tensor the ANE takes a unit of `height` rows at a time, `width` values a row (`Arena.take(units:)`).
        case units(width: Int, height: Int, element: Int)
    }

    /// A tensor of the stage that no file holds: its name, its bytes, where it lies.
    public struct Tensor: Sendable {
        public var name: String
        public var bytes: Int
        public var place: Place
        public init(_ name: String, bytes: Int, _ place: Place = .heap) { self.name = name; self.bytes = bytes; self.place = place }
        public init(_ name: String, surface shape: [Int], element: Int) {
            self.init(name, bytes: shape.reduce(1, *) * element, .surface(shape: shape, element: element))
        }
        /// What it takes in the arena: its pages.
        var size: Int { Arena.aligned(bytes) }
    }

    /// A function the stage loads: its asset's name (from what it computes: the key of its share on this Mac), the unit
    /// it is specialized for, its module, built only when its asset must be written (`Functions.asset`).
    public struct Function: Sendable {
        public var asset: String
        public var unit: ComputeUnitKind
        public var build: @Sendable () throws -> Aicode.Module
        public init(_ asset: String, unit: ComputeUnitKind = .gpu, build: @escaping @Sendable () throws -> Aicode.Module) {
            self.asset = asset; self.unit = unit; self.build = build
        }
    }

    /// **What a stage needs**, as its client declares it.
    public struct Declaration: Sendable {
        /// A name for what the engine says of it.
        public var name: String
        public var tensors: [Tensor] = []
        /// Weights read once and held all along.
        public var residents: [Weights.Entry] = []
        /// Weights the stage may keep or read into the ring, by group, in no order (the loop's is `once` and `cycle`).
        public var groups: [String: [Weights.Entry]] = [:]
        /// The group whose layout the ring's slots take: every group read into the ring lays out as it does.
        public var slot: String? = nil
        /// The loop's use of the groups: each of `once` in turn, then `cycle` over and over.
        public var once: [String] = []
        public var cycle: [String] = []
        public var functions: [String: Function] = [:]
        /// The weights (residents, kept, slots) as surfaces, for the ANE, laid as `codes` says (the ANE reads `[out, in]`
        /// codes transposed in its function as well, 21/11/2026: `SharedWeights.integer`).
        public var surfaces = false
        /// The layouts the functions are written for, of the products' int8 codes and of their bf16 weights: the plan lays
        /// the weights so and reads no measure of this Mac (`Block` reads it once, as it writes its functions). Read twice,
        /// a measure kept between the two reads handed the functions weights laid otherwise, and a precondition stopped
        /// the worker (19/11).
        public var codes: CodesLayout = .inOut
        public var bfloat16: CodesLayout = .outIn
        public init(_ name: String) { self.name = name }
    }

    // MARK: - The plan

    /// **A plan under B**: the declaration taken, the groups kept from the start, the ring's depth, the arena's terms.
    public struct Plan: Sendable {
        public let declaration: Declaration
        /// The groups kept from the start, in the loop's order (those the ring cannot hold; every group where none rides
        /// it); the ring's slots (0: no ring).
        public let kept: [String]
        public let depth: Int
        /// The layouts this stage's product weights take, and its functions are built for: its int8 codes', its bf16
        /// weights' (`Recipe.product`).
        public let codes: CodesLayout
        public let bfloat16: CodesLayout
        /// What the arena asks: the heap (its tensors, its weights, the staging) and what lies apart from it.
        public let heap: Int, apart: Int
        public let floor: Int, budget: Int

        /// What Metal allocates for the arena (`Arena.allocation`) and what lies apart.
        public var arena: Int { Arena.allocation(bytes: heap) + apart }
        /// P = floor + arena: what the stage allocates; Core AI's share is read in flight (`signal`, `estimates`).
        public var p: Int { floor + arena }
        /// P's terms, as a sentence.
        public var line: String {
            "\(declaration.name): P = floor \(Memory.mib(floor)) + arena \(Memory.mib(arena)) = \(Memory.mib(p)) under B = \(Memory.mib(budget))"
                + (depth == 0 ? ", every group kept" : ", a ring of \(depth)" + (kept.isEmpty ? "" : ", \(kept.count) groups kept from the start"))
        }
    }

    /// What P keeps clear under B: the footprint's sampling and Core AI's small allocations.
    public static let margin = 64 << 20

    /// **The plan of `d` under `budget`**, its weights in `files`. `floor`: the process's footprint before the stage;
    /// `depth`: the ring's slots. **The socle alone**: a group that rides the ring rides it from the start, and what of the
    /// cycle the stage keeps is decided in flight, once Core AI's share is read (`signal`): no share is known before the
    /// first call, and a share measured apart was off by 0.3 to 3 GiB a stage (19/11). A socle over B: `OverBudget`, by
    /// how much it passes B. A client with several forms plans each and chooses among those that hold (`choose`).
    public static func plan(_ d: Declaration, files: [SafeTensors], budget: Int, floor: Int, depth: Int = 2) throws -> Plan {
        precondition(depth >= 1, "a ring of a slot at least")
        let codes = d.codes, bfloat16 = d.bfloat16
        let sizes = try Sizes(d, files: files, codes: codes, bfloat16: bfloat16)
        func plan(_ kept: [String], _ depth: Int) -> Plan {
            let (heap, apart) = sizes.arena(kept: kept, depth: depth)
            return Plan(declaration: d, kept: kept, depth: depth, codes: codes, bfloat16: bfloat16, heap: heap, apart: apart,
                        floor: floor, budget: budget)
        }
        let order = sizes.order
        // The groups that cannot ride the ring are kept whatever it costs; where none rides, every group is, and there is
        // no ring.
        let bound = order.filter { !sizes.rides($0) }
        let socle = bound.count == order.count ? plan(order, 0) : plan(bound, depth)
        if socle.p + margin <= budget { return socle }
        let why = socle.depth == 0 ? "every group kept" : bound.isEmpty ? "a ring of \(depth)" : "a ring of \(depth) and \(bound.count) groups the ring cannot hold"
        throw OverBudget(needed: socle.p + margin, budget: budget,
                         description: "\(d.name): at its finest (\(why)) P = \(Memory.mib(socle.p)) + \(Memory.mib(margin)) passes B = "
                             + "\(Memory.mib(budget)) by \(Memory.mib(socle.p + margin - budget))")
    }

    /// A declaration's sizes, from its files' headers alone.
    struct Sizes {
        let declaration: Declaration
        /// The heap's tensors, those apart and the surfaces (`aligned` each).
        let heapTensors: Int, apartTensors: Int
        let residents: Int
        let groups: [String: Int]
        let slot: Int
        let staging: Int
        /// The groups in the loop's order, each once.
        let order: [String]
        private let riding: Set<String>

        init(_ d: Declaration, files: [SafeTensors], codes: CodesLayout, bfloat16: CodesLayout) throws {
            declaration = d
            heapTensors = d.tensors.filter { $0.place == .heap }.reduce(0) { $0 + $1.size }
            apartTensors = d.tensors.filter { $0.place != .heap }.reduce(0) { $0 + $1.size }
            var staging = 0, residents = 0
            for (file, entries) in try Stage.byFile(d.residents, files) {
                residents += try Weights.size(entries, in: file, codes: codes, bfloat16: bfloat16)
                staging = max(staging, try Weights.staging(entries, in: file, codes: codes, bfloat16: bfloat16))
            }
            var groups: [String: Int] = [:], layouts: [String: [String]] = [:]
            for (g, entries) in d.groups {
                let file = try Stage.file(of: entries, g, files)
                groups[g] = try Weights.size(entries, in: file, codes: codes, bfloat16: bfloat16)
                staging = max(staging, try Weights.staging(entries, in: file, codes: codes, bfloat16: bfloat16))
                layouts[g] = try Weights.layout(entries, in: file, codes: codes, bfloat16: bfloat16).map { "\($0.name) \($0.shape) \($0.type)" }.sorted()
            }
            var seen = Set<String>(), order: [String] = []
            for g in d.once + d.cycle where seen.insert(g).inserted {
                guard groups[g] != nil else { throw Failure("\(d.name): the loop uses \(g), no group of that name") }
                order.append(g)
            }
            if let missing = Set(groups.keys).subtracting(seen).sorted().first { throw Failure("\(d.name): the group \(missing) is never used") }
            if let s = d.slot, groups[s] == nil { throw Failure("\(d.name): the slots' group \(s) is not a group") }
            self.residents = residents
            self.groups = groups
            self.order = order
            // A page at least where anything is read: a recipe that reads straight into its place (bf16) passes none through
            // it, and the reader is given one all the same.
            let reads = !d.residents.isEmpty || !d.groups.isEmpty
            self.staging = staging == 0 ? (reads ? Arena.page : 0) : Arena.aligned(staging)
            slot = d.slot.map { groups[$0]! } ?? 0
            riding = Set(groups.keys.filter { g in d.slot.map { layouts[g] == layouts[$0] } ?? false })
        }

        /// Whether a group can be read into the ring's slots: it lays out as the slots' group.
        func rides(_ g: String) -> Bool { riding.contains(g) }

        /// The heap and what lies apart, `kept` kept and a ring of `depth`.
        func arena(kept: [String], depth: Int) -> (heap: Int, apart: Int) {
            let weights = residents + kept.reduce(0) { $0 + groups[$1]! } + depth * slot
            return declaration.surfaces ? (heapTensors + staging, apartTensors + weights) : (heapTensors + weights + staging, apartTensors)
        }
    }

    /// The file that holds a weight: its key, or a product's parts (`Recipe.codes`, `Recipe.product`).
    static func holds(_ e: Weights.Entry, _ file: SafeTensors) -> Bool {
        switch e.recipe {
        case .codes: file.tensors[e.key + ".weight"] != nil
        case .product(let joined): ([e.key] + joined).allSatisfy { file.tensors[$0 + ".weight"] != nil }
        case .target: true
        default: file.tensors[e.key] != nil
        }
    }

    /// The one file of `files` that holds every entry of a group.
    static func file(of entries: [Weights.Entry], _ group: String, _ files: [SafeTensors]) throws -> SafeTensors {
        guard let f = files.first(where: { f in entries.allSatisfy { holds($0, f) } }) else {
            throw Failure("the group \(group): no file holds all its weights")
        }
        return f
    }

    /// `entries` by the file that holds each, in the files' order (a `target` with the codes it reads).
    static func byFile(_ entries: [Weights.Entry], _ files: [SafeTensors]) throws -> [(SafeTensors, [Weights.Entry])] {
        var at: [Int: [Weights.Entry]] = [:]
        var last = 0
        for e in entries {
            if case .target = e.recipe { at[last, default: []].append(e); continue }
            guard let i = files.firstIndex(where: { holds(e, $0) }) else { throw Failure("no file holds \(e.key)") }
            at[i, default: []].append(e)
            last = i
        }
        return at.keys.sorted().map { (files[$0], at[$0]!) }
    }

    // MARK: - The live stage

    public let plan: Plan
    public let device: MTLDevice
    /// The stage's own queue and stream: let go with it (`close`).
    public private(set) var queue: MTLCommandQueue!
    public private(set) var stream: ComputeStream!
    public private(set) var arena: Arena!
    /// The functions loaded, by the declaration's names.
    public private(set) var functions: [String: InferenceFunction] = [:]
    /// The residents, by their arguments' names.
    public private(set) var residents: [String: Placed] = [:]

    private var at: [String: Int] = [:]
    private var ring: Ring?
    /// **What this stage's functions hold, as read**: this Mac's key for them (`key`), what it last read past the first
    /// signal (`reserve`; nil, never), the footprint as the stage reads it (the process's, unless a test says otherwise).
    public let key: String
    public private(set) var reserve: Int?
    private let footprint: @Sendable () -> Int
    private let measured: ThisMac
    /// Under `lock`: Core AI's share read so far at its largest — the footprint past the floor and the arena —, and at
    /// the first signal; whether the reader keeps (`signal`).
    private var largest = Int.min, first: Int?, keeping = false
    /// Under `lock`, past a climb (`rekey`): the turn its functions are first called in, the share then, and the growth
    /// it expects.
    private var climbed: (turn: Int, share: Int, expects: Int)?
    /// Under `lock`: the groups the reader decided to keep whose heap is not allocated yet (`keeps`, `Arena.grow`) —
    /// counted by every decision against B meanwhile, a climb's as the reader's.
    private var promised = 0
    /// Under `lock`: the stage opened whole (its functions loaded); a stage whose opening threw keeps no reading.
    private var ready = false
    /// **The share at its peak, read every millisecond**: a call's working memory is taken and given back within the
    /// call, between two of the stage's readings (20/11: the share read at the uses 1 826 MiB at most, the fuse's
    /// footprint 156 MiB above it under the same arena, at B = 4). Under `lock`, the loads under way (`loading`): what a
    /// function's loading takes is the load's, not the calls', and none is read or kept meanwhile.
    private var watch: FootprintWatch?
    private var loads = 0
    /// Under `lock`: the key the stage's reading is kept under (its own, or the functions a client loaded in flight:
    /// `rekey`), and the share at its largest since.
    public private(set) var readingKey: String
    private var keyed = 0
    /// The reader's staging in the arena.
    private var staging: Int?
    private var use = 0
    /// What its close waited for, once closed (`Stages` told once): a stage may be closed again by its owner's `deinit`.
    private var settled: Settled?
    private let lock = NSLock()
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
        reserve = measured.reserves?[key]
        before = footprint()
        loads = 1
        defer { lock.withLock { loads -= 1 } }
        opened(key)
        Stages.tell(.opened(plan))
        let d = plan.declaration
        guard let queue = device.makeCommandQueue() else { throw Failure("no command queue") }
        self.queue = queue
        stream = ComputeStream(commandQueue: queue)
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
        let codes = plan.codes, bfloat16 = plan.bfloat16, surfaces = d.surfaces
        // The staging: one, the reader's, the residents read through it before the reader starts.
        let staging = try Sizes(d, files: files, codes: codes, bfloat16: bfloat16).staging
        let stagingKey = staging > 0 ? arena.take(staging) : nil
        self.staging = stagingKey
        for (file, entries) in try Self.byFile(d.residents, files) {
            let placed = Weights.place(try Weights.layout(entries, in: file, codes: codes, bfloat16: bfloat16), in: arena, surfaces: surfaces)
            try Weights.read(entries, from: file, into: placed, arena: arena, staging: arena.pointer(stagingKey!), codes: codes, bfloat16: bfloat16)
            residents.merge(placed) { $1 }
        }
        // The kept groups' places, the ring's slots.
        var groupFile: [String: Int] = [:]
        for (g, entries) in d.groups {
            let f = try Self.file(of: entries, g, files)
            groupFile[g] = files.firstIndex { $0 === f }!
        }
        var layouts: [String: [Weights.Slot]] = [:]
        for (g, entries) in d.groups { layouts[g] = try Weights.layout(entries, in: files[groupFile[g]!], codes: codes, bfloat16: bfloat16) }
        var bound: [String: [String: Placed]] = [:]
        for g in plan.kept { bound[g] = Weights.place(layouts[g]!, in: arena, surfaces: surfaces) }
        var slots: [[String: Placed]] = []
        if plan.depth > 0, let s = d.slot { slots = (0..<plan.depth).map { _ in Weights.place(layouts[s]!, in: arena, surfaces: surfaces) } }
        if !plan.kept.isEmpty || !slots.isEmpty {
            let uncached = try files.map { try $0.reopened(uncached: true) }
            let groups = d.groups, fileOf = groupFile, laid = layouts
            let sizes = try Sizes(d, files: files, codes: codes, bfloat16: bfloat16)
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
                try Weights.read(groups[g]!, from: uncached[fileOf[g]!], into: place, arena: arena, staging: arena.pointer(stagingKey!), codes: codes, bfloat16: bfloat16)
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

    /// A stage let go by its owner without `close` (a decoder's, with its pass) keeps its readings all the same.
    deinit {
        ring?.stop()
        watch?.stop()
        opened(nil)
        if lock.withLock({ settled == nil && ready }) { keep() }
    }

    /// **What this stage read, kept on this Mac**, at its largest: the share read since its key, the reading of the
    /// functions it names (`ThisMac.readings`); what Core AI's share grew past the first signal, the reserve of its
    /// functions (`ThisMac.reserves`) — not past a climb: a reserve counts from a stage's first signal on its functions,
    /// where a stage opened on them gives it (20/11: read past a climb from a signal of the main pass, 31 MiB; opened on
    /// the same functions, the first signal in a pass of keys before it, the share grew 194 MiB past the margin and the
    /// fuse stopped a render at B = 10). At the close or the stage let go, and wherever its client knows every function was
    /// called (`Bands`, a pass done: a client may hold its stage until the process ends). A stage that loads no function
    /// keeps nothing.
    public func keep() {
        let (share, _, _) = read()
        let (keyed, readingKey, first, climbed) = lock.withLock { (self.keyed, self.readingKey, self.first, self.climbed != nil) }
        guard !plan.declaration.functions.isEmpty else { return }
        Self.keep(reading: keyed, under: readingKey)
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

    /// A group's weights while the loop holds them: kept (`use` nil) or a slot of the ring.
    public struct Held: @unchecked Sendable {
        public let group: String
        public let weights: [String: Placed]
        public let use: Int?
    }

    /// **The group's weights, once read**: kept, or the ring's next use, which must be the loop's next (the order
    /// declared: `once`, then `cycle` over and over). Released (`release`) once the last call that reads it is done.
    public func acquire(_ group: String) throws -> Held {
        guard let ring else { preconditionFailure("\(plan.declaration.name): \(group) neither kept nor in a ring") }
        let m = lock.withLock { () -> Int in defer { use += 1 }; return use }
        read()
        let (weights, n) = try ring.acquire(m, group)
        return Held(group: group, weights: weights, use: n)
    }

    /// The held group done: its slot may take the ring's next read (a kept group stays).
    public func release(_ held: Held) {
        if let n = held.use { ring?.release(n) }
    }

    /// Every group kept from the start read.
    public func filled() throws { try ring?.ready(plan.kept) }

    /// The groups kept so far, from the start or past the signal, in the order read.
    public var kept: [String] { ring?.keptGroups ?? plan.kept }

    /// **This Mac's key for a stage's functions**: the hash of the assets it loads, each with its unit (a reading holds
    /// for these functions on this system: `ThisMac` is kept by system).
    public static func key(_ d: Declaration) -> String { key(d.functions.values.map { ($0.asset, $0.unit) }) }

    /// The same key, of functions a client will load (`Block`'s, at another height or share of the ANE).
    public static func key(_ functions: [(asset: String, unit: ComputeUnitKind)]) -> String {
        let lines = functions.map { ThisMac.shareKey(asset: $0.asset, unit: $0.unit) }.sorted().joined(separator: "\n")
        return "stage " + Data(SHA256.hash(data: Data(lines.utf8))).prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    /// Core AI's share at its largest so far, the footprint now, and the share now (the footprint past the floor and the
    /// arena).
    @discardableResult
    private func read(allocated: Int? = nil) -> (largest: Int, footprint: Int, share: Int) {
        let now = footprint(), held = allocated ?? arena?.allocated ?? 0, share = now - plan.floor - held
        let largest = lock.withLock { () -> Int in
            // The watch's peak read under the lock: a `rekey` resets it there, and no peak of the functions before it
            // reaches the new key's reading.
            let peak = max(share, watch?.peak ?? share)
            if loads == 0 {
                largest = max(largest, peak)
                keyed = max(keyed, peak)
            }
            return max(largest, share)
        }
        return (largest, now, share)
    }

    /// **Core AI's share at its peak under the functions loaded now** (since the last `rekey`): what a client scales
    /// to estimate another set's (`Block`).
    public func observe() -> Int {
        read()
        return lock.withLock { keyed }
    }

    /// Under `lock`: what is left to come of Core AI's share at `share` — of the reserve, what it has not grown since
    /// the first signal; past a climb, of the growth it expects, what it has not grown since.
    private func left(_ share: Int) -> Int {
        max((reserve ?? 0) - (share - (first ?? share)), 0) + (climbed.map { max($0.expects - (share - $0.share), 0) } ?? 0)
    }

    /// The cycle's turn of the use before the next (−1 in `once`); `next`, of the next use.
    private func turn(next: Bool = false) -> Int {
        let d = plan.declaration, m = next ? use : use - 1
        return m < d.once.count || d.cycle.isEmpty ? -1 : (m - d.once.count) / d.cycle.count
    }

    /// **The reading kept under `key` from now on, if what it `expects` holds under B** — the functions a client will load
    /// in flight (`Block`'s taller calls, another share of the ANE), `expects` the growth of Core AI's share it estimated
    /// for them: the footprint at the share's peak, the growth, what is left of the reserve and the margin beside, under
    /// B (`keeps`'s rule), decided under the lock the reader decides under, so that neither takes the room the other
    /// just took. Then the reading so far kept under the key before; from the climb, the stage reads as if opened: the
    /// reserve the new functions' (or none), counted from the next signal, the growth expected counted as still to come
    /// until the share has grown by as much; never read, the reader keeps from the first signal two turns past the one
    /// the new functions are first called in (`signal`), not on the lower functions' numbers before the taller ones took
    /// their working memory. `false`: nothing changed.
    public func rekey(_ key: String, expects: Int) -> Bool {
        let (share, now, current) = read()
        let reserve = measured.reserves?[key], extra = max(expects, 0)
        let old = lock.withLock { () -> (key: String, reading: Int)? in
            guard now + (share - current) + promised + extra + left(share) + max(Stage.margin, (share + extra) / 20) <= plan.budget
            else { return nil }
            let old = (readingKey, keyed)
            watch?.reset()
            (readingKey, keyed, self.reserve, first) = (key, 0, reserve, nil)
            keeping = keeping && reserve != nil
            climbed = (turn(next: true), share, extra)
            return old
        }
        guard let old else { return false }
        Self.keep(reading: old.reading, under: old.key)
        opened(key)
        return true
    }

    /// A kept group's heap allocated (or its place failed): no longer promised.
    private func delivered(_ bytes: Int) { lock.withLock { promised -= Arena.allocation(bytes: bytes) } }

    /// A reading kept at its largest (a stage fused, −1, is kept as it is: `fused`).
    static func keep(reading share: Int, under key: String) {
        try? ThisMac.update { m in
            let old = m.readings?[key]
            m.readings = (m.readings ?? [:]).merging([key: old == -1 ? -1 : max(old ?? 0, max(share, 0))]) { $1 }
        }
    }

    /// The open stages' reading keys and names, by stage: what the fuse marks and says (`fused`).
    nonisolated(unsafe) private static var open: [ObjectIdentifier: (key: String, name: String, order: Int)] = [:]
    nonisolated(unsafe) private static var openings = 0
    private static let openLock = NSLock()
    private func opened(_ key: String?) {
        let name = plan.declaration.name
        Self.openLock.withLock {
            Self.openings += 1
            Self.open[ObjectIdentifier(self)] = key.map { ($0, name, Self.openings) }
        }
    }

    /// **The fuse stopped the process** (`FootprintWatch`): the functions of the stage opened (or rekeyed) last marked −1
    /// on this system, never taken again (`estimates`), and named. Not every open stage's: one opened before it held its
    /// place under B until it opened (a decoder its client holds idle until the process ends, `Bands`), and marking it
    /// would refuse it for good for another's fault.
    static func fused() -> [String] {
        guard let last = openLock.withLock({ open.values.max { $0.order < $1.order } }) else { return [] }
        try? ThisMac.update { m in m.readings = (m.readings ?? [:]).merging([last.key: -1]) { $1 } }
        return [last.name]
    }

    /// **What a set of functions will hold, before its first call**: its own reading on this Mac; never read, the
    /// reading of the tallest of the others read, scaled by the rows a call covers (a function's working memory is its
    /// intermediates, as tall as its rows: 1.1 GiB a convolution head at 256 × 256, 0.9 to 1.07 a layer's function over
    /// 4608 rows, tens of MiB over a few hundred; measured 19/11 on the mini); none read, unknown; fused, never taken.
    public enum Estimate: Sendable, Equatable {
        case read(Int), scaled(Int), unknown, fused
        public var bytes: Int? { switch self { case .read(let b), .scaled(let b): b; default: nil } }
    }

    /// The estimate of each of `forms` (a key and the rows its tallest call covers), from `measured`.
    public static func estimates(_ forms: [(key: String, rows: Int)], measured: ThisMac = .current) -> [Estimate] {
        let readings = forms.map { measured.readings?[$0.key] }
        let reference = zip(forms, readings).filter { ($0.1 ?? -1) >= 0 }.max { $0.0.rows < $1.0.rows }
        return zip(forms, readings).map { f, r in
            if r == -1 { return .fused }
            if let r { return .read(r) }
            guard let (ref, bytes) = reference else { return .unknown }
            return .scaled(Int((Double(bytes!) * Double(f.rows) / Double(max(ref.rows, 1))).rounded(.up)))
        }
    }

    /// **The form to take of `plans` (the tallest first), their `estimates` beside**: the first whose P, its estimate
    /// counted, holds under B with the margin; none known, the soberest whose socle holds (the last); one known and none
    /// that holds, refused with the number; every form fused, refused with the soberest's P, the one number known.
    public static func choose(_ plans: [Plan], _ estimates: [Estimate]) throws -> Int {
        precondition(!plans.isEmpty && plans.count == estimates.count)
        if let i = plans.indices.first(where: { i in estimates[i].bytes.map { plans[i].p + $0 + margin <= plans[i].budget } ?? false }) { return i }
        if estimates.last == .unknown { return plans.count - 1 }
        // None read, some fused: the soberest form never tried.
        if estimates.allSatisfy({ $0 == .unknown || $0 == .fused }), let i = estimates.lastIndex(of: .unknown) { return i }
        let i = plans.indices.filter { estimates[$0].bytes != nil }.min { plans[$0].p + estimates[$0].bytes! < plans[$1].p + estimates[$1].bytes! }
        guard let i else {
            // What Core AI held past the soberest's socle was never read, only that the process passed B under it (the
            // mark outlives the B it was taken under): its P is the number known.
            let soberest = plans.last!
            throw OverBudget(needed: soberest.p + margin, budget: soberest.budget,
                             description: "\(soberest.declaration.name): every form stopped the fuse on this Mac; at its soberest P = "
                                 + "\(Memory.mib(soberest.p)) + \(Memory.mib(margin)), under B = \(Memory.mib(soberest.budget))")
        }
        let p = plans[i].p + estimates[i].bytes!
        throw OverBudget(needed: p + margin, budget: plans[i].budget,
                         description: "\(plans[i].declaration.name): its socle and what Core AI will hold, \(Memory.mib(p)) + "
                             + "\(Memory.mib(margin)), pass B = \(Memory.mib(plans[i].budget)) by \(Memory.mib(p + margin - plans[i].budget))")
    }

    /// **Whether the reader keeps a group of `bytes` now**: past the signal, if the footprint, what is allocated and not
    /// yet written (`pending`: the slots never read into), the group, what is left of the reserve (the reserve less
    /// what Core AI's share grew since the first signal) and a margin hold under B — the margin the larger of
    /// `Stage.margin` and 5 % of the share read —, and if the process may wire it beside what it wired (`Arena.wire`: a
    /// kept group is wired, or macOS compresses it between two reads).
    private func keeps(_ bytes: Int, pending: Int, allocated: Int) -> Bool {
        let (share, now, current) = read(allocated: allocated)
        let wired = Arena.wiredBytes
        return lock.withLock {
            let group = Arena.allocation(bytes: bytes)
            guard keeping && loads == 0,
                  now + (share - current) + promised + pending + group + left(share) + max(Stage.margin, share / 20) <= plan.budget,
                  wired + promised + group <= Memory.wireLimit
            else { return false }
            promised += group
            return true
        }
    }

    /// Whether the reader keeps the cycle's groups past a signal (`signal`): a loop that gives its signal only when it
    /// may change something stops once it holds (`Layers.through`).
    public var keeps: Bool { lock.withLock { keeping } }

    /// **The loop's signal: what Core AI holds is read**, the second group of a pass done, every call before it done (a
    /// function's working memory is taken at its second call, not its first: 982 then 1 318 MiB, a layer of 60, 19/11).
    /// The first signal marks what the reserve counts from; from it, the reader keeps the cycle's groups while B holds
    /// them with what is left of the reserve beside (`keeps`). **The reserve never read on this Mac** (a set of functions
    /// first met): read in this stage — the reader keeps from the first signal of the cycle's third turn, every function
    /// then seen called twice, a function called once a turn too (the reserve 0; the margin and the fuse beside), and
    /// this stage's reading kept at its close (20/11: kept from the second turn, the share grew 156 MiB past a margin of
    /// 91 as a step's edges took their working memory, and the fuse stopped a render at B = 4). Given, never waited for.
    public func signal() {
        let (share, _, _) = read()
        let on = lock.withLock { () -> Bool in
            if first == nil { first = share }
            if reserve != nil || turn() >= (climbed?.turn ?? 0) + 2 { keeping = true }
            return keeping
        }
        if on { ring?.signal(); ring?.wake() }
    }

    /// What a stage let go left: the footprint against what it was before the stage allocated, the seconds waited, the
    /// arenas still alive and their heaps still held.
    public struct Settled: Sendable {
        public let footprint, before: Int
        public let seconds: Double
        public let alive, held: Int
        public var within: Bool { footprint <= before + Stage.margin && alive == 0 && held == 0 }
        public var line: String {
            (footprint >= before ? "+" : "") + Memory.mib(footprint - before) + String(format: " after %.2f s", seconds)
                + (alive + held == 0 ? "" : ", arenas alive \(Memory.mib(alive)), held \(Memory.mib(held))")
        }
    }

    /// **Everything let go** — the reader stopped, the functions, the arena, the queue and the stream — then the wait
    /// until the footprint is back within 64 MiB of what it was before the stage, 2 s at most (measured 18/11 on the M1
    /// Pro, a render's stages each let go: the heap leaves within 0.05 to 0.5 s, then 100 to 160 MiB of Core AI's or
    /// Metal's 1 to 2 s after; a function holds its last call's buffers until it is let go, `Arena.take(apart:)`).
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
        watch?.stop()
        watch = nil
        functions = [:]
        residents = [:]
        arena = nil
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
