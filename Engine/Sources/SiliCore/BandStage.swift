import CoreAI
import Foundation
import Metal

/// **A stage of bands, its arena filled and its functions loaded**: the plan of the first candidate whose P holds under
/// B (bands of fewer positions, the shorter, the later), each distinct function once (the same module, the same asset:
/// two layers of one shape share it), every weight a resident; its calls on the stream in their order, two in flight
/// (`inFlight`).
public final class BandStage: @unchecked Sendable {
    public let plan: Bands
    public let stage: Stage
    /// The rank of the candidate taken (0: the preferred).
    public let candidate: Int
    /// **Whether B made the band shorter than the preferred**: a taller one's socle, or its estimate, over B. Not when no
    /// candidate was ever read on this Mac and the soberest is taken for want of a reading (`Stage.choose`): that is a
    /// first run at these shapes, not this Mac's limit (seen 21/11 under B = 46 GiB: the shortest band at a stage's first
    /// run, the tallest at its second, and its client said B had cut it).
    public let narrowed: Bool
    private let functions: [Bands.Shaped: InferenceFunction]
    private let signatures: [Bands.Shaped: Block.Signature]

    private init(_ plan: Bands, _ stage: Stage, candidate: Int, narrowed: Bool, signatures: [Bands.Shaped: Block.Signature],
                 assets: [Bands.Shaped: String]) {
        (self.plan, self.stage, self.candidate, self.narrowed, self.signatures) = (plan, stage, candidate, narrowed, signatures)
        var loaded: [Bands.Shaped: InferenceFunction] = [:]
        for (key, asset) in assets { loaded[key] = stage.functions[asset] }
        functions = loaded
        for (name, m) in plan.masks {
            let p = stage.pointer(name).assumingMemoryBound(to: Float.self)
            for i in 0..<m.length { p[i] = m.inside.contains(i) ? 1 : 0 }
        }
    }

    /// **A stage of bands' candidates, built once** (`candidates`): the plan of each band's positions (the tallest first),
    /// its functions built at its shapes and named by their bytes (`prefix-<hash>`), its socle from the files' headers —
    /// what `admit` judges before the computation whose result the stage reads and `open` opens after it, judged on the
    /// floor each meets. Built once, not at each (26/11/2026, the M1 Pro: a stage's four candidates built and named
    /// before the transformer before it, then again after it, 0.34 to 0.40 s the second time, the trace's gap before
    /// the stage).
    public struct Candidates: @unchecked Sendable {
        public let name: String
        let files: [SafeTensors]
        let plans: [Bands]
        let socles: [Stage.Plan]
        let named: [[Bands.Shaped: String]]
        let modules: [String: Aicode.Module]
    }

    /// **The candidates of a stage of bands** (`plan` writes the plan of a band's positions), under `budget`, their
    /// residents read from `files`. A socle that cannot be planned for another reason than B (a tensor the file lacks)
    /// goes up as it is.
    public static func candidates(_ name: String, prefix: String, candidates: [Int], files: [SafeTensors], residents: [Weights.Entry],
                                  budget: Int, plan: (Int) -> Bands) throws -> Candidates {
        let plans = candidates.map(plan)
        var modules: [String: Aicode.Module] = [:]
        var socles: [Stage.Plan] = [], named: [[Bands.Shaped: String]] = []
        for p in plans {
            var d = Stage.Declaration("\(name), bands of \(p.positions) positions")
            d.tensors = p.slabs.enumerated().map { Stage.Tensor("slab\($0.offset)", bytes: $0.element) }
                + p.smalls.map { Stage.Tensor($0.name, bytes: $0.bytes, $0.apart ? .apart : .heap) }
            d.residents = residents
            d.inFlight = inFlight
            d.extent = try p.extent(prefix, inFlight: inFlight)
            var assets: [Bands.Shaped: String] = [:]
            for (key, module) in try p.modules() {
                let asset = Functions.name(module, prefix: prefix)
                assets[key] = asset
                modules[asset] = module
                nonisolated(unsafe) let m = module
                d.functions[asset] = Stage.Function(asset, unit: .gpu) { m }
            }
            socles.append(try Stage.socle(d, files: files, budget: budget))
            named.append(assets)
        }
        return Candidates(name: name, files: files, plans: plans, socles: socles, named: named, modules: modules)
    }

    /// **The stage of the first of the candidates whose socle and estimate hold under B** on the footprint it opens on
    /// (`Stage.planned`), its residents read: what Core AI will hold under each estimated from this Mac's readings
    /// (`Stage.estimates(extents:)`: its own, or a bound from a reading of the same functions at another height or
    /// size); none holding, the soberest on paper where nothing of its kin was read or only bands that hold more bound
    /// it, refused where a reading of bands that hold no more predicts it past B — a size larger than any read, refused
    /// before the computation `admit` judges for (`Stage.choose`). A stage's reading is kept at its close, once its
    /// functions were called: the next stage of its kin climbs on it.
    public static func open(_ c: Candidates) async throws -> BandStage {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("no Metal device") }
        let began = DispatchTime.now().uptimeNanoseconds
        var p = try Stage.planned { floor in
            let choice = try choice(c, floor: floor)
            return (choice, choice.plan)
        }
        // The trace's plan from the judgment on (the candidates were built before: `candidates`).
        p.opening.start = began
        let stage = try await Stage(device, plan: p.plan, files: c.files, measured: .current, opening: p.opening)
        let chosen = p.value
        return BandStage(chosen.bands, stage, candidate: chosen.rank, narrowed: chosen.narrowed,
                         signatures: chosen.assets.mapValues { Block.Signature(c.modules[$0]!) }, assets: chosen.assets)
    }

    /// The stage of `candidates` built (`candidates`) and opened (`open`): what a stage opened at once takes.
    public static func open(_ name: String, prefix: String, candidates: [Int], files: [SafeTensors], residents: [Weights.Entry], budget: Int,
                            plan: (Int) -> Bands) async throws -> BandStage {
        try await open(Self.candidates(name, prefix: prefix, candidates: candidates, files: files, residents: residents, budget: budget, plan: plan))
    }

    /// **What `open` would specialize**, its assets written: every candidate that holds on the floor the process comes
    /// back to (`Stage.restingFloor`), nothing allocated or read (`Stage.specializations`). Every one that holds, not
    /// the one taken alone: with nothing of its kin read on this Mac `open` takes the finest (`Stage.choose`), once read a
    /// taller one; and it opens later on a higher floor — a candidate refused here is refused there (25/11/2026, the M1
    /// Pro, a stage of 4 candidates: opened at the finest the first time, prepared, at the tallest the next, never
    /// prepared, and compiled it).
    public static func prepare(_ c: Candidates) throws -> [Functions.Specialization] {
        try Stage.specializations(choice(c, floor: Stage.restingFloor()).holding.flatMap(\.declaration.functions.values))
    }

    /// **Whether `open` would hold over `floor`**, nothing allocated: the candidate it would take, or `OverBudget` with
    /// the number — what a client judges before the computation whose result the stage reads (refused after it, that
    /// computation's minutes were paid for nothing). `floor`: the footprint the stage will open over — where the process
    /// comes back to once the stages before it gave back what they took (`Stage.restingFloor`).
    @discardableResult
    public static func admit(_ c: Candidates, floor: Int = Stage.restingFloor()) throws -> Int {
        try choice(c, floor: floor).rank
    }

    private struct Choice {
        let rank: Int, bands: Bands, plan: Stage.Plan, narrowed: Bool
        let assets: [Bands.Shaped: String]
        /// The plan of every candidate that holds, the chosen among them.
        let holding: [Stage.Plan]
    }

    /// The candidate `open` takes and `admit` judges on `floor`, its plan and its functions.
    private static func choice(_ c: Candidates, floor: Int) throws -> Choice {
        // A socle over B leaves the shorter bands to try (`Plan.on` refuses with `OverBudget`); none holds, the finest's
        // number.
        var holding: [(rank: Int, plan: Stage.Plan)] = [], refusal: OverBudget?
        for (rank, socle) in c.socles.enumerated() {
            do { holding.append((rank, try socle.on(floor: floor))) }
            catch let o as OverBudget { refusal = o }
        }
        guard let budget = holding.first?.plan.budget else { throw refusal! }
        let estimates = Stage.estimates(extents: holding.map { (Stage.key($0.plan.declaration), $0.plan.declaration.extent!) }, budget: budget)
        // Every form fused: `OverBudget` too, the finest's P its number (`Stage.choose`).
        let chosen = holding[try Stage.choose(holding.map(\.plan), estimates)].rank
        let unread = holding.count == c.socles.count && estimates.allSatisfy { $0 == .unknown }
        return Choice(rank: chosen, bands: c.plans[chosen], plan: holding.first { $0.rank == chosen }!.plan, narrowed: chosen > 0 && !unread,
                      assets: c.named[chosen], holding: holding.map(\.plan))
    }

    private func view(_ s: Bands.Spot) -> View {
        stage.view(s.tensor, s.shape, s.type, strides: Layout.padded(s.shape, s.type), bytes: s.offset)
    }

    /// **The calls a run keeps on the stream at once**: a call is encoded while the one before runs. Waited for one by
    /// one, the GPU stood idle while the CPU encoded each call (≈ 0.5 ms, a function of 40 to 90 ops) and while the end
    /// of each was signalled: measured 21/11 on the M5 Pro, a network of 512 to 128 channels, 659 calls of bands of 2¹⁴
    /// positions, 1.25 s, of which 0.36 s encoding; two in flight 0.90 s, three 0.88; a network of 1152 to 144 channels
    /// 1.67 → 1.27 s, one of 384 to 96 0.77 → 0.47 s; the same outputs to the bit. The stream orders the calls by the
    /// buffers they read and write (each band on pages of its own, `Arena.part`): a call reads what the one before
    /// wrote. Two calls of one function in flight (a layer's bands follow each other) each have their intermediates:
    /// Core AI allocates a second set (+80 to +170 MiB at the peak on those three), read in flight like the rest of its
    /// share (`Stage.keep`).
    static let inFlight = 2

    /// Every call, in order, `inFlight` on the stream; the marks `keeping` names read back as their calls are done,
    /// `[C, rows, W]` f32. What the run executed told at its end (`Stages.Work`); under the trace (`Stage.traced`), each
    /// call's start and end, from the ends its waits give (`Stages.Trace.calls`).
    public func run(keeping: Set<String> = []) async throws -> [String: [Float]] {
        var kept: [String: [Float]] = [:]
        let marks = plan.marks.filter { keeping.contains($0.name) }
        let stream = stage.stream!
        var flight: [Task<UInt64, Never>] = []
        let traced = stage.traced
        var calls: [Stages.Called] = [], encoded: [UInt64] = [], ends: [UInt64] = []
        var flops = 0, first: UInt64?
        func landed(_ end: UInt64) { if traced { ends.append(end) } }
        for (i, call) in plan.calls.enumerated() {
            let sig = signatures[call.function]!
            func bind(_ arg: String, _ shape: [Int], _ type: NDArray.ScalarType) -> View {
                if let s = call.spots[arg] { return view(s) }
                if let key = call.weights[arg] {
                    guard let p = stage.residents[key] else { preconditionFailure("\(call.function): no resident \(key) for \(arg)") }
                    return stage.view(p, shape)
                }
                if let t = call.tensors[arg] { return stage.view(t, shape, type, strides: Layout.padded(shape, type)) }
                preconditionFailure("\(call.function): nothing bound to \(arg)")
            }
            var inputs: [String: View] = [:]
            for (n, s, t) in sig.inputs { inputs[n] = bind(n, s, t) }
            let outputs = sig.outputs.map { n, s, t in (n, bind(n, s, t)) }
            if flight.count == Self.inFlight { landed(await flight.removeFirst().value) }
            let left = try Run.encode(functions[call.function]!, stream, inputs: inputs, outputs: outputs)
            precondition(left.isEmpty, "\(call.function.key): outputs no view holds, \(left.keys.sorted())")
            let at = DispatchTime.now().uptimeNanoseconds
            if first == nil { first = at }
            flops += sig.flops
            flight.append(stream.done())
            if traced {
                calls.append(Stages.Called(kind: .fixed, step: call.function.key, group: 0, place: 1, flops: sig.flops))
                encoded.append(at)
            }
            // A boundary read back: every call so far done.
            if marks.contains(where: { $0.call == i }) { await Run.finish(stream) }
            for m in marks where m.call == i { kept[m.name] = values(m.plane) }
        }
        await Run.finish(stream)
        // The last calls in flight, done with the stream: their ends read, no wait.
        var end = DispatchTime.now().uptimeNanoseconds
        for t in flight { let e = await t.value; landed(e); end = e }
        if let first {
            Stages.tell(.work(Stages.Work(stage: stage.plan.declaration.name, flops: Stages.Flops(gpu: flops),
                                          seconds: Double(max(end, first) - first) / 1e9, end: max(end, first))))
        }
        if traced, let first, ends.count == calls.count {
            // On one stream, a call starts at its encoding or at the end of the one before, the later.
            for j in calls.indices { (calls[j].start, calls[j].end) = (max(encoded[j], j > 0 ? ends[j - 1] : 0), ends[j]) }
            Stages.tell(.traced(.calls(stage: stage.plan.declaration.name, cycle: nil, Stages.occupancy(calls, start: first, end: max(end, first)))))
        }
        // Every function called: the reading kept now (a client may hold its stage until its process ends).
        stage.keep()
        return kept
    }

    /// The bytes of the stage's tensor `name`.
    public func pointer(_ name: String) -> UnsafeMutableRawPointer { stage.pointer(name) }

    /// **A plane's values**, `[C, rows, W]` f32, its rows' factors multiplied out under the guard.
    public func values(_ t: Bands.Plane) -> [Float] {
        let base = UnsafeRawPointer(stage.pointer(t.slab.map { "slab\($0)" } ?? "t\(t.id)"))
        let factors = t.factors ? UnsafeRawPointer(stage.pointer("f\(t.id)")) : nil
        let row = Layout.padded([1, t.channels, t.rho, t.width], t.element)[2], half = t.element == .float16
        var out = [Float](repeating: 0, count: t.channels * t.rows * t.width)
        out.withUnsafeMutableBufferPointer { o in
            for c in 0..<t.channels {
                for y in 0..<t.rows {
                    let (j, i) = (y / t.rho, y % t.rho), at = base + j * t.bandBytes + (c * t.rho + i) * row * Layout.element(t.element)
                    let f = factors.map { $0.advanced(by: j * Arena.page).assumingMemoryBound(to: Float.self)[i] } ?? 1
                    let to = o.baseAddress! + (c * t.rows + y) * t.width
                    if half { let h = at.assumingMemoryBound(to: Float16.self); for x in 0..<t.width { to[x] = Float(h[x]) * f } }
                    else { let s = at.assumingMemoryBound(to: Float.self); for x in 0..<t.width { to[x] = s[x] * f } }
                }
            }
        }
        return out
    }

    /// **`values` (`[C, rows, W]`) written into a plane**, under factors of 1, its carry `carry` if it has one (what a
    /// plane the CPU writes carries to the layers that read it).
    public func write(_ t: Bands.Plane, _ values: [Float], carry: [Float] = []) {
        precondition(values.count == t.channels * t.rows * t.width, "\(values.count) values for a plane of \(t.channels) × \(t.rows) × \(t.width)")
        let base = stage.pointer(t.slab.map { "slab\($0)" } ?? "t\(t.id)")
        let row = Layout.padded([1, t.channels, t.rho, t.width], t.element)[2], half = t.element == .float16
        values.withUnsafeBufferPointer { v in
            for c in 0..<t.channels {
                for y in 0..<t.rows {
                    let (j, i) = (y / t.rho, y % t.rho), at = base + j * t.bandBytes + (c * t.rho + i) * row * Layout.element(t.element)
                    let from = v.baseAddress! + (c * t.rows + y) * t.width
                    if half { let h = at.assumingMemoryBound(to: Float16.self); for x in 0..<t.width { h[x] = Float16(from[x]) } }
                    else { let s = at.assumingMemoryBound(to: Float.self); for x in 0..<t.width { s[x] = from[x] } }
                }
            }
        }
        if t.factors {
            let f = stage.pointer("f\(t.id)")
            for j in 0..<t.count { let p = (f + j * Arena.page).assumingMemoryBound(to: Float.self); for i in 0..<t.rho { p[i] = 1 } }
        }
        if !t.carry.isEmpty {
            precondition(carry.count == t.carry.reduce(1, *), "a carry of \(carry.count) values for \(t.carry)")
            // Read as the carry of a plane no call wrote: its first buffer.
            let p = stage.pointer("c\(t.id)#0").assumingMemoryBound(to: Float.self)
            let row = Layout.padded(t.carry, .float32)[max(t.carry.count - 2, 0)], width = t.carry.last!
            for (n, x) in carry.enumerated() { p[t.carry.count == 1 ? n : (n / width) * row + n % width] = x }
        }
    }
}
