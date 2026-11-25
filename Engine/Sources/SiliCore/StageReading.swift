import CoreAI
import Foundation

/// **What a stage's functions hold, read in flight and kept on this Mac**: their key and extent, the share read, the
/// climb (`rekey`), the reader's keeping and the loop's signal, the fuse's mark, the estimates and the form chosen.
extension Stage {
    /// **A stage's functions apart from their shapes** (`kin`: the client's functions and what each computes, how many
    /// calls it keeps in flight) **and the elements they hold** (`elements`, by a name the same at any shapes:
    /// a stage of bands' arguments, `Bands.extent`; a block's functions' results, `Block.extent`). Core AI's share under
    /// functions written alike grows with what their calls hold, never faster, but where its kernels change with the
    /// shapes (`estimates(extents:)`).
    public struct Extent: Codable, Sendable, Equatable {
        public let kin: String
        public let elements: [String: Int]
        public init(kin: String, elements: [String: Int]) { (self.kin, self.elements) = (kin, elements) }

        /// The kin of `lines` (what names the functions apart from their shapes, their ops among it: `Block.walk`),
        /// launched `inFlight` at a time. Not the engine that wrote them: a new binary that writes the same ops keeps its
        /// kin's readings, and one that writes them otherwise is another kin by its ops (decided 22/11/2026: with the
        /// engine in its kin, every build lost every reading, and each stage took its soberest form once more).
        public init(_ lines: [String], inFlight: Int, elements: [String: Int]) {
            let all = (["in flight \(inFlight)"] + lines.sorted()).joined(separator: "\n")
            self.init(kin: "kin " + Block.digest(Data(all.utf8)), elements: elements)
        }

        /// **How much more this holds than `read`**: of its arguments, the largest ratio of their elements — nil where
        /// their arguments are not the same.
        public func ratio(over read: Extent) -> Double? {
            guard kin == read.kin, Set(elements.keys) == Set(read.elements.keys), read.elements.values.allSatisfy({ $0 > 0 }) else { return nil }
            return elements.map { Double($0.value) / Double(read.elements[$0.key]!) }.max() ?? 1
        }
    }

    /// **This Mac's key for a stage's functions**: the hash of the assets it loads, each with its unit, and of the calls
    /// it keeps in flight past one (a reading holds for these functions so launched, on this system: `ThisMac` is kept
    /// by system). Two calls of a function in flight hold two sets of its intermediates: bands of 2¹⁴ positions, +80 to
    /// +170 MiB at the peak on the M5 Pro (21/11/2026), past the margin a reading taken one call at a time leaves.
    public static func key(_ d: Declaration) -> String { key(d.functions.values.map { ($0.asset, $0.unit) }, inFlight: d.inFlight) }

    /// The same key, of functions a client will load (`Block`'s, at another height or share of the ANE).
    public static func key(_ functions: [(asset: String, unit: ComputeUnitKind)], inFlight: Int = 1) -> String {
        let lines = (functions.map { ThisMac.shareKey(asset: $0.asset, unit: $0.unit) }.sorted() + (inFlight > 1 ? ["in flight \(inFlight)"] : []))
            .joined(separator: "\n")
        return "stage " + Block.digest(Data(lines.utf8))
    }

    /// Core AI's share at its largest so far, the footprint now, and the share now (the footprint past the floor and the
    /// arena).
    @discardableResult
    func read(allocated: Int? = nil) -> (largest: Int, footprint: Int, share: Int) {
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
    /// their working memory. `extent`: what the new functions hold, kept beside their reading. `false`: nothing changed.
    public func rekey(_ key: String, expects: Int, extent: Extent? = nil) -> Bool {
        guard holds(expects: expects) else { return false }
        return rekeyed(key, expects: expects, extent: extent)
    }

    /// **Whether functions whose share grows by `expects` would hold** under B beside what is kept now: `rekey`'s rule,
    /// nothing changed — what a block weighs its shares by before it moves (`Block.admit`).
    public func holds(expects: Int) -> Bool {
        let (share, now, current) = read()
        let extra = max(expects, 0)
        return lock.withLock { now + (share - current) + promised + extra + left(share) + max(Stage.margin, (share + extra) / 20) <= plan.budget }
    }

    private func rekeyed(_ key: String, expects: Int, extent: Extent?) -> Bool {
        let (share, now, current) = read()
        let reserve = measured.reserves?[key], extra = max(expects, 0)
        let old = lock.withLock { () -> (key: String, reading: Int, extent: Extent?)? in
            guard now + (share - current) + promised + extra + left(share) + max(Stage.margin, (share + extra) / 20) <= plan.budget
            else { return nil }
            let old = (readingKey, keyed, readingExtent)
            watch?.reset()
            (readingKey, readingExtent, keyed, self.reserve, first) = (key, extent, 0, reserve, nil)
            keeping = keeping && reserve != nil
            climbed = (turn(next: true), share, extra)
            return old
        }
        guard let old else { return false }
        // The old key's reading, if its functions were called; the new key's needs calls of its own (`called`).
        if called(again: true) { Self.keep(reading: old.reading, under: old.key, extent: old.extent, budget: plan.budget) }
        opened(key)
        return true
    }

    /// A kept group's heap allocated (or its place failed): no longer promised.
    func delivered(_ bytes: Int) { lock.withLock { promised -= Arena.allocation(bytes: bytes) } }

    /// A reading kept at its largest, of a stage run under `budget` (`ThisMac.keep`: a mark −1 of this engine that holds
    /// under it kept as it is, another's or one taken under a smaller B replaced), and what its functions held beside it
    /// (`ThisMac.extents`).
    static func keep(reading share: Int, under key: String, extent: Extent? = nil, budget: Int) {
        try? ThisMac.update { m in
            m.keep(reading: share, under: key, budget: budget)
            if let extent { m.extents = (m.extents ?? [:]).merging([key: extent]) { $1 } }
        }
    }

    /// The open stages' reading keys and names, by stage: what the fuse marks and says (`fused`).
    nonisolated(unsafe) private static var open: [ObjectIdentifier: (key: String, name: String, order: Int, budget: Int)] = [:]
    nonisolated(unsafe) private static var openings = 0
    private static let openLock = NSLock()
    func opened(_ key: String?) {
        let (name, budget) = (plan.declaration.name, plan.budget)
        Self.openLock.withLock {
            Self.openings += 1
            Self.open[ObjectIdentifier(self)] = key.map { ($0, name, Self.openings, budget) }
        }
    }

    /// **The fuse stopped the process** (`FootprintWatch`): the functions of the stage opened (or rekeyed) last marked −1
    /// on this system under the B it ran under, never taken again by this engine under a B no larger (`estimates`,
    /// `ThisMac.fuse`), and named. Not every open stage's:
    /// one opened before it held its place under B until it opened (a stage of bands its client holds idle until the
    /// process ends, `Bands`), and marking it
    /// would refuse it for good for another's fault.
    static func fused() -> [String] {
        guard let last = openLock.withLock({ open.values.max { $0.order < $1.order } }) else { return [] }
        try? ThisMac.update { $0.fuse(last.key, budget: last.budget) }
        return [last.name]
    }

    /// **The open stages' streams, each with whether a call was put on it** since its stage opened or was rekeyed
    /// (`Run.encode`): a reading is kept of functions called only. The share read under functions loaded and never
    /// called is next to nothing; kept, it became the reading of a key never read and, `estimates` taking the least
    /// bound, the bound of its whole kin (a stage closed after its loading, its client stopped: a block whose tallest
    /// calls hold 1589 MiB would have kept ≈ 0, and the next opening taken them). A stage calls the ANE (`Run.neural`,
    /// no stream) beside the GPU's calls on its stream, never alone.
    nonisolated(unsafe) private static var calls: [ObjectIdentifier: Bool] = [:]

    /// A call put on `stream`: its stage's, if a stage's.
    static func call(on stream: SerialStream) {
        let id = ObjectIdentifier(stream)
        openLock.withLock { if calls[id] != nil { calls[id] = true } }
    }

    /// The stage's stream listened to (at its opening), or no longer (`nil`, at its close).
    func listen(_ stream: SerialStream?) {
        Self.openLock.withLock {
            if let stream { Self.calls[ObjectIdentifier(stream)] = false } else if let id = streamID { Self.calls[id] = nil }
        }
        if let stream { streamID = ObjectIdentifier(stream) }
    }

    /// Whether a call was put on the stage's stream since it opened or was rekeyed; `again`, from now on as if none.
    func called(again: Bool = false) -> Bool {
        guard let id = streamID else { return false }
        return Self.openLock.withLock {
            defer { if again && Self.calls[id] != nil { Self.calls[id] = false } }
            return Self.calls[id] ?? false
        }
    }

    /// **What a set of functions will hold, before its first call**: its own reading on this Mac; never read, a bound from
    /// its kin's readings at other shapes (`estimates`: a block's other heights and shares of the ANE, a stage's other
    /// bands, at any size) — `scaled` from a form whose calls hold no more, the reading times their ratio, a prediction
    /// (bands of 16 384 positions from a reading of 14 336, 10 % over what they held); `capped` by a form whose calls
    /// hold more, the reading itself, which says what it holds at most and nothing of how much less; none read,
    /// unknown; fused by this engine, never taken (another engine's mark is no reading: `ThisMac.reading`).
    public enum Estimate: Sendable, Equatable {
        case read(Int), scaled(Int), capped(Int), unknown, fused
        public var bytes: Int? { switch self { case .read(let b), .scaled(let b), .capped(let b): b; default: nil } }
    }

    /// **What a reading moves by with what ran before it in its process** (the same form's share read alone and after a
    /// process's other stages: −1 % to +9 %, and a form's reading times its ratio up to 4 % short of another's, 22/11/2026
    /// on the M5 Pro): a bound from another form's reading counts it (`estimates`).
    public static let swing = 1.125

    /// **What each of a stage's forms will hold** (`Extent`: the same functions at other shapes — a block's calls at
    /// another height or beside another share of the ANE, bands of another height, at any size): its own
    /// reading on this Mac; never read, the least of the bounds each reading of its kin gives — the reading times how
    /// much more its calls hold (`Extent.ratio`), never less than the reading itself (what Core AI holds beside the
    /// calls' intermediates, its functions' code, does not shrink with them), by `swing`, and `margin` beside: `scaled`
    /// where the least comes from a form that holds no more, `capped` from one that holds more (`Estimate`; a tie,
    /// `scaled`: a prediction says that number); none, or a form that says no extent, unknown; fused by this engine,
    /// never taken.
    ///
    /// Measured 22/11/2026 on the M5 Pro. Four stages of bands at ten sizes and four heights each (3024 pairs of a
    /// reading and another form's, each read in a process of its own; then 1656, read after the process's other stages):
    /// the reading times its ratio fell short of the other form's by 7 MiB at most alone, by 145 MiB after the other stages
    /// (2979 MiB read at 896 × 1152, bands of 14 336 positions, times 1.14, against 3549 read at 1024², of 16 384) — the
    /// swing and the margin left 122 MiB at the least over every pair. Three transformers' blocks, their products int8
    /// and bf16, on the GPU alone and beside the ANE, at every height over passes of two sizes (4096 to 4160 rows, 1024
    /// to 1088), each read in a process of its own (266 pairs of one kin): 41 MiB left at the least but once, 32 MiB
    /// short — a bf16 block over calls of 1024 rows, 868 MiB read, from 686 read over calls of 2048 (Core AI's graphics
    /// memory owned and unmapped, 534 MB against 246, its heaps 232 MiB against 419: what its kernels take at that
    /// height, not what the calls hold), within the margin `choose` counts beside. The scaling by rows this replaces
    /// bounded a shorter form from a taller one below its own reading: bands of 2¹³ from 2¹⁴, 721 MiB, 914 read; a
    /// block's calls of 2112 rows from a call a pass of 4160, 987 MiB, 1589 read (int8, 4608 keys); 68 of the blocks'
    /// 136 pairs of one size below their reading.
    /// `budget`: the B the forms are planned under, under which a mark −1 taken under a smaller one is no reading
    /// (`ThisMac.reading`); nil, every mark holds.
    public static func estimates(extents forms: [(key: String, extent: Extent?)], measured: ThisMac = .current, budget: Int? = nil) -> [Estimate] {
        forms.map { f in
            if let r = measured.reading(f.key, budget: budget) { return r == -1 ? .fused : .read(r) }
            guard let extent = f.extent else { return .unknown }
            let bounds = (measured.extents ?? [:]).compactMap { key, read -> (bytes: Int, capped: Int)? in
                guard let r = measured.reading(key), r >= 0, let ratio = extent.ratio(over: read) else { return nil }
                return (Int((Double(r) * max(ratio, 1) * swing).rounded(.up)) + margin, ratio < 1 ? 1 : 0)
            }
            guard let least = bounds.min(by: { ($0.bytes, $0.capped) < ($1.bytes, $1.capped) }) else { return .unknown }
            return least.capped == 1 ? .capped(least.bytes) : .scaled(least.bytes)
        }
    }

    /// **The form to take of `plans` (the tallest first), their `estimates` beside**: the first whose P, its estimate
    /// counted, holds under B with the margin. None holds: a form taller than one read or fused holds more than it, and
    /// is never taken; of the forms soberer than every one read or fused, the soberest goes on paper, under the fuse,
    /// where nothing bounds it from below — nothing of its kin read (`unknown`: a size never met, no one can know), or
    /// only a form that holds more (`capped`: a B lowered under every form read, decided 22/11/2026, goes as on a Mac
    /// with no reading) —, and is refused where a form that holds no more predicts it past B (`scaled`: a size larger
    /// than any this Mac read, refused before the computation whose result it reads, `BandStage.admit`). Refused, the
    /// number: the soberest fused, its P, the one number known; else the least P its estimate counted.
    public static func choose(_ plans: [Plan], _ estimates: [Estimate]) throws -> Int {
        precondition(!plans.isEmpty && plans.count == estimates.count)
        if let i = plans.indices.first(where: { i in estimates[i].bytes.map { plans[i].p + $0 + margin <= plans[i].budget } ?? false }) { return i }
        let past = (estimates.lastIndex { switch $0 { case .read, .fused: true; default: false } } ?? -1) + 1, last = plans.count - 1
        if past <= last { switch estimates[last] { case .unknown, .capped: return last; default: break } }
        guard estimates[last] != .fused else {
            // What Core AI held past the soberest's socle was never read, only that the process passed B under it (the
            // mark outlives the B it was taken under): its P is the number known.
            let soberest = plans[last]
            throw OverBudget(needed: soberest.p + margin, budget: soberest.budget,
                             description: "\(soberest.declaration.name): its soberest form stopped the fuse on this Mac, and a taller one "
                                 + "holds more; at its soberest P = \(Memory.mib(soberest.p)) + \(Memory.mib(margin)), under B = "
                                 + "\(Memory.mib(soberest.budget))")
        }
        // The soberest read or predicted: an estimate at least.
        let i = plans.indices.filter { estimates[$0].bytes != nil }.min { plans[$0].p + estimates[$0].bytes! < plans[$1].p + estimates[$1].bytes! }!
        let p = plans[i].p + estimates[i].bytes!
        throw OverBudget(needed: p + margin, budget: plans[i].budget,
                         description: "\(plans[i].declaration.name): its socle and what Core AI will hold, \(Memory.mib(p)) + "
                             + "\(Memory.mib(margin)), pass B = \(Memory.mib(plans[i].budget)) by \(Memory.mib(p + margin - plans[i].budget))")
    }

    /// **Whether the reader keeps a group of `bytes` now**: past the signal, if the footprint, what is allocated and not
    /// yet written (`pending`: the slots never read into), the group, what is left of the reserve (the reserve less
    /// what Core AI's share grew since the first signal) and a margin hold under B — the margin the larger of
    /// `Stage.keepClear` and 5 % of the share read —, and if the process may wire it beside what it wired (`Arena.wire`: a
    /// kept group is wired, or macOS compresses it between two reads).
    func keeps(_ bytes: Int, pending: Int, allocated: Int) -> Bool {
        let (share, now, current) = read(allocated: allocated)
        let wired = Arena.wiredBytes
        return lock.withLock {
            let group = Arena.allocation(bytes: bytes)
            guard keeping && loads == 0,
                  now + (share - current) + promised + pending + group + left(share) + max(Stage.keepClear, share / 20) <= plan.budget,
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
    /// 91 as a step's edges took their working memory, and the fuse stopped a run at B = 4). Given, never waited for.
    public func signal() {
        let (share, _, _) = read()
        let on = lock.withLock { () -> Bool in
            if first == nil { first = share }
            if reserve != nil || turn() >= (climbed?.turn ?? 0) + 2 { keeping = true }
            return keeping
        }
        if on { ring?.signal(); ring?.wake() }
    }
}
