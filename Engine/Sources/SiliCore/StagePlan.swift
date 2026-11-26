import CoreAI
import Foundation
import Metal

/// **What a stage needs and its plan under B**: the client's declaration, the plan of its socle from its files' headers
/// alone (`Sizes`), the files that hold its weights; a stage planned and opened.
extension Stage {
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
        /// The weights (residents, kept, slots) as surfaces, for the ANE.
        public var surfaces = false
        /// **The calls of one function its client keeps in flight at once** (`BandStage.inFlight`): each has its own
        /// intermediates in Core AI's share, so a reading taken at one count is not the reading at another (`key`).
        public var inFlight = 1
        /// **What its functions are apart from their shapes, and what each call holds** (`Bands`): a reading of the same
        /// functions at other shapes bounds what these will hold (`estimates(extents:)`); kept beside its reading.
        public var extent: Extent? = nil
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
        /// What the arena asks: the heap (its tensors, its weights, the staging) and what lies apart from it.
        public let heap: Int, apart: Int
        public let floor: Int, budget: Int
        /// Its declaration's sizes, from its files' headers: read once, for the plan and the stage it opens.
        let sizes: Sizes

        /// What Metal allocates for the arena (`Arena.allocation`) and what lies apart.
        public var arena: Int { Arena.allocation(bytes: heap) + apart }
        /// **The weights its arena holds from the start**: the residents, the groups kept from the start, the ring's slots
        /// (the groups that ride it pass through them) — in bytes, as read from the files' headers. A group kept in
        /// flight adds its own (`Stages.Event.kept`).
        public var weights: Int { sizes.residents + kept.reduce(0) { $0 + sizes.groups[$1]! } + depth * sizes.slot }
        /// P = floor + arena: what the stage allocates; Core AI's share is read in flight (`signal`, `estimates`).
        public var p: Int { floor + arena }
        /// **The same socle on `floor`**: refused (`OverBudget`, by how much it passes B) where P and the margin pass B.
        func on(floor: Int) throws -> Plan {
            let p = Plan(declaration: declaration, kept: kept, depth: depth, heap: heap, apart: apart, floor: floor, budget: budget, sizes: sizes)
            if p.p + margin <= budget { return p }
            let why = depth == 0 ? "every group kept" : kept.isEmpty ? "a ring of \(depth)" : "a ring of \(depth) and \(kept.count) groups the ring cannot hold"
            throw OverBudget(needed: p.p + margin, budget: budget,
                             description: "\(declaration.name): at its finest (\(why)) P = \(Memory.mib(p.p)) + \(Memory.mib(margin)) passes B = "
                                 + "\(Memory.mib(budget)) by \(Memory.mib(p.p + margin - budget))")
        }
        /// P's terms, as a sentence.
        public var line: String {
            "\(declaration.name): P = floor \(Memory.mib(floor)) + arena \(Memory.mib(arena)) = \(Memory.mib(p)) under B = \(Memory.mib(budget))"
                + (depth == 0 ? ", every group kept" : ", a ring of \(depth)" + (kept.isEmpty ? "" : ", \(kept.count) groups kept from the start"))
        }
    }

    /// What P keeps clear under B: the footprint's sampling and Core AI's small allocations.
    public static let margin = 64 << 20

    /// **What the groups kept past the signal leave clear under B, for the rest of the Mac**: a kept group is wired, so a
    /// stage whose weights pass B kept groups up to B less `margin` and held the footprint there, wired, the whole run;
    /// macOS then swapped other processes' pages out (a 16 GB M1 Pro, B = 10 GiB, 9.75 GiB: 4 to 283 pages a run,
    /// 21–22/11/2026). Keeping is optional: the same step there with 2, 10 or 13 of 28 groups kept (22/11/2026).
    public static let keepClear = 512 << 20

    /// **The plan of `d` under `budget`**, its weights in `files`. `floor`: the process's footprint before the stage;
    /// `depth`: the ring's slots. **The socle alone**: a group that rides the ring rides it from the start, and what of the
    /// cycle the stage keeps is decided in flight, once Core AI's share is read (`signal`): no share is known before the
    /// first call, and a share measured apart was off by 0.3 to 3 GiB a stage (19/11). A socle over B: `OverBudget`, by
    /// how much it passes B. A client with several forms plans each and chooses among those that hold (`choose`).
    public static func plan(_ d: Declaration, files: [SafeTensors], budget: Int, floor: Int, depth: Int = 2) throws -> Plan {
        try socle(d, files: files, budget: budget, depth: depth).on(floor: floor)
    }

    /// **The socle of `d` under `budget`, on no floor yet**: what does not change with the footprint, planned once and
    /// judged on a floor by `Plan.on` (`planned` judges it on two).
    static func socle(_ d: Declaration, files: [SafeTensors], budget: Int, depth: Int = 2) throws -> Plan {
        precondition(depth >= 1, "a ring of a slot at least")
        let sizes = try Sizes(d, files: files)
        let order = sizes.order
        // The groups that cannot ride the ring are kept whatever it costs; where none rides, every group is, and there is
        // no ring.
        let bound = order.filter { !sizes.rides($0) }
        let (kept, ring) = bound.count == order.count ? (order, 0) : (bound, depth)
        let (heap, apart) = sizes.arena(kept: kept, depth: ring)
        return Plan(declaration: d, kept: kept, depth: ring, heap: heap, apart: apart, floor: 0, budget: budget, sizes: sizes)
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

        init(_ d: Declaration, files: [SafeTensors]) throws {
            declaration = d
            heapTensors = d.tensors.filter { $0.place == .heap }.reduce(0) { $0 + $1.size }
            apartTensors = d.tensors.filter { $0.place != .heap }.reduce(0) { $0 + $1.size }
            var staging = 0, residents = 0
            for (file, entries) in try Stage.byFile(d.residents, files) {
                residents += try Weights.size(entries, in: file)
                staging = max(staging, try Weights.staging(entries, in: file))
            }
            var groups: [String: Int] = [:], layouts: [String: [String]] = [:]
            for (g, entries) in d.groups {
                let file = try Stage.file(of: entries, g, files)
                groups[g] = try Weights.size(entries, in: file)
                staging = max(staging, try Weights.staging(entries, in: file))
                layouts[g] = try Weights.layout(entries, in: file).map { "\($0.name) \($0.shape) \($0.type)" }.sorted()
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

    /// **A stage planned under `budget` and opened**: planned on the process's footprint, refused at once where its socle
    /// and the share its functions were last read at on this Mac pass B (`estimates`, `choose`: a set the fuse stopped is
    /// refused too); never read, opened, the fuse behind.
    public static func open(_ device: MTLDevice, _ declaration: Declaration, files: [SafeTensors], budget: Int, depth: Int = 2,
                            measured: ThisMac = .current) async throws -> Stage {
        let choice = try chosen(declaration, files: files, budget: budget, depth: depth, measured: measured)
        let p = try planned(measured: measured) { floor in try ((), choice(floor)) }
        return try await Stage(device, plan: p.plan, files: files, measured: measured, opening: p.opening)
    }

    /// **What `open` would specialize**, its assets written: the same plan and choice, from the files' headers, nothing
    /// allocated or read (`specializations`), on the floor the process comes back to (`restingFloor`).
    public static func prepare(_ declaration: Declaration, files: [SafeTensors], budget: Int, depth: Int = 2,
                               measured: ThisMac = .current) throws -> [Functions.Specialization] {
        let choice = try chosen(declaration, files: files, budget: budget, depth: depth, measured: measured)
        return try specializations(choice(restingFloor()).declaration.functions.values)
    }

    /// The plan `open` takes on a floor, refused where its socle or the share its functions were read at passes B.
    private static func chosen(_ declaration: Declaration, files: [SafeTensors], budget: Int, depth: Int,
                               measured: ThisMac) throws -> (Int) throws -> Plan {
        let socle = try socle(declaration, files: files, budget: budget, depth: depth)
        let estimates = estimates(extents: [(key(declaration), declaration.extent)], measured: measured, budget: budget)
        return { floor in
            let p = try socle.on(floor: floor)
            _ = try choose([p], estimates)
            return p
        }
    }

    /// **`functions` as a stage loads them, to specialize before it opens** (`Functions.specialize`): each asset written,
    /// each function once, the ANE's first — 0.4 to 2.3 s each to compile on the M1 Pro (25/11/2026), 0.3 to 1.1 on the
    /// M5 Pro, a GPU's ≈ 35 ms (24/11): a preparation cut short has done the most of it.
    public static func specializations(_ functions: some Sequence<Function>) throws -> [Functions.Specialization] {
        // Once by its name and unit: an asset's URL ends in a slash once its folder exists, so the same function met
        // before and after it was written would give two.
        var seen = Set<String>(), all: [(name: String, s: Functions.Specialization)] = []
        for f in functions where seen.insert(ThisMac.shareKey(asset: f.asset, unit: f.unit)).inserted {
            all.append((f.asset, Functions.Specialization(try Functions.asset(f.asset, f.build), unit: f.unit)))
        }
        return all.sorted { ($0.s.unit == .neuralEngine ? 0 : 1, $0.name) < ($1.s.unit == .neuralEngine ? 0 : 1, $1.name) }.map(\.s)
    }
}
