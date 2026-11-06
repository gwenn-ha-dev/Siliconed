import Foundation
import Testing
@testable import Siliconed

/// **The coverage guard: every precision P20 permits has a route the solver can offer, or a written reason why not — and no route
/// offers one it forbids.** Pure: no GPU, no weights, no Mac in particular. The M5 Pro's lesson (2/11/2026): P20
/// permitted bf16 on bf16 weights and no route of the Standard offered it — neither the map nor a real render measures a route that
/// does not exist, so only a table read from the code can see the hole.
///
///     cell = version × the weight's published form (as the version's classes carry it, per phase) × the engines' lanes
///            lane = an engine × the operand its rows compute on — fp32 · bf16 (rounded) · fp16 · int8 (an activation quantized)
///            one lane alone, or a pair, or the three (and the CPU's heads beside the GPU for an attention)
///
///     routed     some candidate of the solver (`CapacitySolver.candidates`) under a render's constraints, on a map holding every
///                line the probes measure, for some class of the version (`CapacityClasses`, the five families, 1024²)
///     permitted  P20 (`p20` below), as the decisions of 2/11 and 3/11/2026 read it
///     listed     `gaps`: each permitted cell without a route matches the first entry that names it — hardware, a decision, or a
///                lot still open (R7…) — with its reason written
///
///     fails      a permitted cell neither routed nor listed (a new hole) · a routed cell P20 forbids · an entry that names a routed
///                cell (a hole filled: narrow the entry) or none (stale) · a route that reads a line no probe measures, unless listed
///
/// What sets it apart — what a coverage test written by analogy with "each route has a test" would get wrong:
///
///   - **It reads the code's own enumerations**, not a copy: the versions (`Variant`), the families and their classes
///     (`CapacityClasses.classes`), the engines (`CapacityModel.Engine`), the forms of a kernel (an exhaustive switch: a new form
///     does not compile until this file says what its rows compute on), the lines a map can hold (`CapacityProbes.Context.Probe`),
///     what the solver weighs (`candidates`, not only the one it keeps).
///   - **Both directions.** A route P20 forbids fails as surely as a permission without a route.
///   - **A pair is its own cell.** The ×1.52 once claimed for GPU + ANE was read off the members' lines, while the pair of the
///     Standard's own route (`gpu·bf16 + ane·fp16`) had no route — it has one since 3/11/2026, with its own sharing in the map. A set
///     is listed only where each of its lanes has a route alone; a set with a lane that has none is that lane's hole.
///   - **A map is the probes'**: the solver's lines are those the probes can measure. A route that reads another is offered on no Mac:
///     listed in `deadRoutes` (empty since the AMX's columns beside a native weight, which read `amx/int8` and `amx/q8`, left the
///     solver on 3/11/2026).
@Suite struct CoverageGuardTests {
    typealias M = CapacityModel
    typealias Engine = CapacityModel.Engine
    typealias Format = CapacityClasses.WeightFormat
    typealias Phase = CapacityClasses.Phase

    // MARK: - The table's axes

    /// What an engine's rows compute on: the activation's precision as it enters the product (the weight is as published or wider;
    /// every sum fp32). `int8`: the activation quantized for the product (W8A8).
    enum Operand: String, CaseIterable, Comparable {
        case fp32, bf16, fp16, int8
        static func < (a: Operand, b: Operand) -> Bool { allCases.firstIndex(of: a)! < allCases.firstIndex(of: b)! }
    }

    struct Lane: Hashable, Comparable, CustomStringConvertible {
        var engine: Engine
        var operand: Operand
        init(_ engine: Engine, _ operand: Operand) { self.engine = engine; self.operand = operand }
        static func < (a: Lane, b: Lane) -> Bool { (a.engine, a.operand) < (b.engine, b.operand) }
        var description: String { "\(engine.rawValue)·\(operand.rawValue)" }
    }

    enum Subject: Hashable, CustomStringConvertible {
        case product(Phase, Format)
        case attention
        var description: String {
            switch self {
            case let .product(phase, format): return "\(phase == .encoding ? "encoder" : "denoiser") \(format.rawValue)"
            case .attention: return "denoiser attention"
            }
        }
        var phase: Phase { if case let .product(p, _) = self { p } else { .denoising } }
        var format: Format? { if case let .product(_, f) = self { f } else { nil } }
    }

    struct Cell: Hashable, CustomStringConvertible {
        var variant: Variant
        var subject: Subject
        var lanes: Set<Lane>
        var description: String { "\(variant.rawValue) · \(subject) · " + lanes.sorted().map(\.description).joined(separator: " + ") }
    }

    /// **What the rows of a kernel compute on.** Exhaustive over `CapacityModel.Form`: a new form is not a silent cell.
    static func lane(_ k: M.Kernel) -> Lane {
        let operand: Operand
        switch k.form {
        // The weight widened, or a bf16 weight read as published beside an fp32 activation (`float × bfloat`: the widening is a shift).
        case .fp32, .bf16: operand = .fp32
        // A bf16, 8-bit or K-quant weight read as published beside the activation rounded to bf16 (the integers are exact in bf16; a
        // K-quant's `sc·q` in fp16, its scales applied in fp32).
        case .bf16act, .int8act, .q8act, .q4kact: operand = .bf16
        // An 8-bit weight read as published: beside the GPU's fp32 activation; the ANE's int8 program takes the integers as its fp16 input.
        case .int8, .q8: operand = k.engine == .ane ? .fp16 : .fp32
        case .fp16: operand = .fp16
        case .graph, .flash, .heads, .tiles: operand = .fp32
        case .tiles16: operand = .fp16
        case .tilesbf16: operand = .bf16
        // W8A8: the activation rotated and quantized per row, the int8 × int8 product in int32.
        case .int8x8: operand = .int8
        }
        return Lane(k.engine, operand)
    }

    // MARK: - P20

    /// **What P20 permits** (P20 and P17; decisions of 2/11 and 3/11/2026): never below the published weight, sums fp32.
    ///
    ///   - fp32: always.
    ///   - bf16: a bf16 weight — bf16 between two operations, as a GPU in bf16; a LoRA — the forge stores it in bf16, its files are fp16
    ///     or bf16; an 8- or 4-bit weight — its integers are exact in bf16, its scales applied in fp32, the rest at least bf16.
    ///   - fp16: only where bf16's range is kept (P17: the residual stream reaches 6.6·10⁵, ten times fp16's ceiling) — the ANE's rows
    ///     (each weight column and activation row lifted by a power of two into fp16's range, a row that overflows handed back to the
    ///     GPU: the ANE at every version, 2/11/2026), and the attention's operands (Q and K out of the norms, V a projection of a normed
    ///     input, P in [0, 1]: `tiles16` at every version, 3/11/2026). A product on plain fp16 operands, on any weight, is not.
    ///   - int8 (the activation quantized for the product): only beside a published 8- or 4-bit weight ("the multiplications may be
    ///     done in the weight's format"); never beside bf16, whose mantissa int8 loses.
    static func p20(_ subject: Subject, _ lane: Lane) -> Bool {
        switch lane.operand {
        case .fp32, .bf16: return true
        case .fp16: return lane.engine == .ane || subject == .attention
        case .int8: return subject.format.map { $0 == .int8 || $0 == .q8 || $0 == .q4k } ?? false
        }
    }

    // MARK: - The listed gaps

    struct Gap: Sendable {
        enum Kind: String, Sendable { case hardware, decided, open }
        let label: String
        let kind: Kind
        let reason: String
        let matches: @Sendable (Cell) -> Bool
        /// An assignment that stands for the gap, and whether the engines run it today (`ProductAssignment.isRunnable`): « present,
        /// never planned » against « the engine would have to change too ». `nil`: no product assignment stands for it.
        let probe: (key: ProductKey, assignment: ProductAssignment, runs: Bool)?
        init(_ label: String, _ kind: Kind, _ reason: String, probe: (ProductKey, ProductAssignment, Bool)? = nil,
             _ matches: @escaping @Sendable (Cell) -> Bool) {
            self.label = label; self.kind = kind; self.reason = reason; self.matches = matches
            self.probe = probe.map { (key: $0.0, assignment: $0.1, runs: $0.2) }
        }
    }

    static func has(_ c: Cell, _ engine: Engine, _ operand: Operand? = nil) -> Bool {
        c.lanes.contains { $0.engine == engine && (operand == nil || $0.operand == operand) }
    }
    static func isProduct(_ c: Cell, _ formats: Set<Format>, _ phase: Phase = .denoising) -> Bool {
        if case let .product(p, f) = c.subject { return p == phase && formats.contains(f) } else { return false }
    }

    /// **Every permitted cell without a route, by its reason**: the first entry that matches a cell names it. Each entry must name a
    /// cell (else it is stale) and no routed one (else the hole was filled: narrow the entry).
    static let gaps: [Gap] = [
        // Hardware: what no engine of any Mac computes.
        Gap("ane-operand", .hardware,
            "the ANE's programs compute on fp16 (an int8 weight dequantized into fp16): no fp32 or bf16 operand (`NeuralEngine`); "
                + "a bf16 ANE was never verified, Core AI set aside") { c in
            c.lanes.contains { $0.engine == .ane && ($0.operand == .fp32 || $0.operand == .bf16) }
        },
        Gap("cpu-products", .hardware,
            "a product on the CPU is the AMX's (BNNS, `amx/fp32`): the CPU's own cores carry no product kernel; the CPU's line is the "
                + "attention's heads") { c in c.subject != .attention && has(c, .cpu) },
        // Decided: a route that could exist and is not wanted.
        Gap("attention-cpu-alone", .decided,
            "the cut by heads keeps one kv group on the GPU at least (`AttentionAssignment.isRunnable`): the CPU alone never takes an "
                + "attention") { c in c.subject == .attention && c.lanes == [Lane(.cpu, .fp32)] },
        Gap("attention-cpu-half", .decided,
            "the CPU's heads compute in fp32 (`HeadSplitAttention`, cblas): no half-precision kernel on the CPU, none measured") { c in
            c.subject == .attention && c.lanes.contains { $0.engine == .cpu && $0.operand != .fp32 }
        },
        Gap("attention-off-gpu", .open,
            "R7: no attention on the AMX or the ANE — their routes are products by a published weight (`Conductor`, `NeuralEngine`), "
                + "an attention has none; never a candidate of the map") { c in c.subject == .attention && (has(c, .amx) || has(c, .ane)) },
        // The encoder (R7, R3d).
        Gap("encoder", .open,
            "R7 (R3d), the encoder: never rounded, never on the ANE (`ProductAssignment.isRunnable`, `widened`) — its products are thin "
                + "(m = the prompt's tokens, under `Limits.roundingRows` 256 and `aneRows` 1 024 for most prompts, below the map's domain) "
                + "and its output conditions every evaluation of the DiT",
            probe: (ProductKey(k: 4096, n: 4096, m: 1024, weight: .bf16, phase: .encoding),
                    .gpuAlone(1024, path: .bf16, rounds: true), false)) { c in
            c.subject.phase == .encoding && c.lanes.contains { $0.operand != .fp32 }
        },
        // W8A8 off the GPU, on a weight in blocks, beside another engine. The GPU's on an int8 weight is a route (`gpu/int8x8`, 6/11/2026).
        Gap("w8a8-off-gpu", .open,
            "an activation quantized to int8 on the ANE (its int8 × int8, M3/M4, not built: its int8 program takes the integers as an "
                + "fp16 input) or the AMX (BNNS reads the fp32 activation): only the GPU's `gpu/int8x8`") { c in
            c.lanes.contains { $0.operand == .int8 && $0.engine != .gpu }
        },
        Gap("w8a8-blocks", .open,
            "W8A8 on a weight in blocks (Q8_0's 32 rows, a K-quant's super-blocks): a scale per block cuts the integers' sum along K, so "
                + "the product is no single int32 accumulation and `Int8Plan` refuses it; the GPU's rounded kernels take them",
            probe: (ProductKey(k: 3072, n: 3072, m: 4096, weight: .q4k),
                    ProductAssignment(path: .native, quantizesActivation: true, gpu: 4096), false)) { c in
            isProduct(c, [.q8, .q4k]) && has(c, .gpu, .int8)
        },
        Gap("w8a8-sets", .open,
            "W8A8 beside another engine: the ANE's rows read the fp32 activation, and no probe measures the pair's sharing — the GPU "
                + "takes the whole product (`ProductAssignment.isRunnable`)",
            probe: (ProductKey(k: 3072, n: 3072, m: 4096, weight: .int8),
                    ProductAssignment(path: .native, quantizesActivation: true, gpu: 2048, ane: 2048), false)) { c in
            isProduct(c, [.int8]) && has(c, .gpu, .int8) && c.lanes.count >= 2
        },
        Gap("amx-bf16", .open,
            "no AMX kernel on bf16 operands (BFMMLA/SME): the probes measure `amx/fp32` alone, so the map cannot say whether it pays") { c in
            c.lanes.contains { $0.engine == .amx && $0.operand == .bf16 }
        },
        Gap("kquant-ane", .hardware,
            "the ANE beside the GPU's K-quant kernel: its int8 program takes a scale per column or for the tensor, never a K-quant's "
                + "super-blocks, and its fp16 program reads a widened weight, which the native route never makes (the reserve holds the "
                + "blocks and their unpacked `sc·q`, whose `d` changes every 256 rows of K); widened, the ANE's rows stay (`gpu+ane`)",
            probe: (ProductKey(k: 3072, n: 3072, m: 4096, weight: .q4k),
                    ProductAssignment(path: .native, roundsActivation: true, gpu: 2048, ane: 2048), false)) { c in
            isProduct(c, [.q4k]) && has(c, .gpu, .bf16) && has(c, .ane)
        },
        Gap("lora-bf16", .open,
            "a LoRA's two products: their operand is the stack's fp32 expansion (`CacheLoRA`), widened only — thin products (the rank, "
                + "≤ `CapacityModel.thinSide`), a few per cent of a render; no bf16 copy of the stack",
            probe: (ProductKey(k: 3072, n: 64, m: 4096, weight: .loraDown), .gpuAlone(4096, path: .bf16, rounds: true), false)) { c in
            isProduct(c, [.loraDown, .loraUp]) && has(c, .gpu, .bf16)
        },
        // The products on a rounded activation, beside another engine.
        Gap("rounded-bf16-amx", .decided,
            "the AMX beside the GPU's rounded rows on a bf16 weight read as published: its rows need the weight widened to fp32 (no AMX "
                + "kernel reads bf16, `amx-bf16`) and the fp32 activation — a widening of the whole weight at every call, beside a reserve "
                + "that holds the bf16 copy and the rounded rows and has no room for it — for what the members' lines gave the AMX on the "
                + "M5 Pro, ~2 % beside GPU + ANE; `isRunnable` refuses AMX rows on a bf16 path (decided 3/11/2026)",
            probe: (ProductKey(k: 3072, n: 3072, m: 4096, weight: .bf16),
                    ProductAssignment(path: .bf16, roundsActivation: true, gpu: 2048, amx: 1024, ane: 1024), false)) { c in
            isProduct(c, [.bf16]) && has(c, .gpu, .bf16) && has(c, .amx)
        },
        Gap("rounded-8bit-amx", .decided,
            "the AMX beside the GPU's rounded rows on an 8-bit or K-quant weight: the AMX reads the fp32 activation (BNNS, `amx/fp32`), so "
                + "one product would round some of its rows or columns and not the others, and its columns beside a native weight left the "
                + "solver (no probe times them; a K-quant's would need its blocks widened beside); `isRunnable` refuses a rounded "
                + "assignment with AMX columns, and any AMX beside a K-quant's blocks",
            probe: (ProductKey(k: 3072, n: 3072, m: 4096, weight: .int8),
                    ProductAssignment(path: .native, roundsActivation: true, gpu: 4096, amxColumns: 1024), false)) { c in
            isProduct(c, [.int8, .q8, .q4k]) && has(c, .gpu, .bf16) && has(c, .amx)
        },
        // No « q8-ane » entry since Z-Image's and Anima's Compacts became int8 convrot: no version of the catalogue publishes Q8_0, so
        // no cell names it. The fact stays for an imported GGUF: the ANE's int8 program takes a scale per column or for the tensor,
        // never Q8_0's blocks (`NeuralEngine.readsInt8`, `ProductAssignment.isRunnable`, `CapacityPlanTests`).
        Gap("lora-sets", .decided,
            "a LoRA's products are thin (`CapacityModel.isThin`): a set's loss was measured on products with no short side, none is "
                + "offered on one — each engine alone is, by its thin line") { c in
            isProduct(c, [.loraDown, .loraUp]) && c.lanes.count >= 2
        },
    ]

    /// **Routes the solver can name that no map can offer**: they read a line no probe measures. Each with its reason.
    static let deadRoutes: [String: String] = [:]

    /// **What no class of the plan covers**, hence no cell: a stage of the chain with a single route.
    static let unplanned: [String: String] = [
        "the VAE": "one route, the GPU in fp32 (`CapacityClasses`: « what is not a class »); R7 (R3d)",
    ]

    // MARK: - The maps

    /// The lines the probes measure: every product probe (`CapacityProbes.Context.Probe`) and the attention kernels they time
    /// (`CapacityProbesSolos`: the graph, flash, the three tiled kernels, the CPU's heads). The bf16 attention is offered only where it
    /// beats fp16 by the margin (`MatmulTile.margin`): its line is faster, so that both are candidates.
    static let probed: Set<M.Kernel> = Set(Diagnostic.CapacityProbes.Context.Probe.allCases.map(\.kernel))
        .union([.init(.gpu, .graph), .init(.gpu, .flash), .init(.gpu, .tiles), .init(.cpu, .heads)])
        .union(AttentionTile.kernels)

    static func model(lines: Set<M.Kernel>) -> M {
        var m = M()
        for k in lines { m.lines[k] = .init(t0: 0, rate: k.form == .tilesbf16 ? 2e12 : 1e12) }
        let engines = Engine.allCases
        for mask in 1..<(1 << engines.count) {
            let set = engines.indices.filter { mask & (1 << $0) != 0 }.map { engines[$0] }
            if set.count >= 2 { m.sharing[M.key(set)] = .init() }
        }
        // The pairs whose sharing the probes measure by kernels (the ANE's integers beside each GPU kernel that reads them, the rounded
        // bf16 pair).
        for pair in Diagnostic.CapacityProbes.Context.integerPairs.map({ [M.Kernel(.gpu, $0), M.Kernel(.ane, .int8)] })
            + [Diagnostic.CapacityProbes.Context.roundedPair] where pair.allSatisfy(lines.contains) {
            m.sharing[M.key(pair)] = .init()
        }
        return m
    }

    /// Every class of every family at 1024², for a short and the longest prompt (the encoder's heights), a LoRA of rank 32.
    static func classes(_ variant: Variant) -> [(Family, CapacityClasses.Class)] {
        Family.allCases.flatMap { family -> [(Family, CapacityClasses.Class)] in
            [11, Request.maximumPromptTokens].flatMap { tokens -> [(Family, CapacityClasses.Class)] in
                let r = CapacityClasses.Request(family: family, variant: variant, promptTokens: tokens, loraRank: 32)
                return (CapacityClasses.classes(for: r) ?? []).map { (family, $0) }
            }
        }
    }

    // MARK: - What the solver offers

    struct Offer {
        var routed: Set<Cell> = []
        var routes: Set<String> = []
        var present: [Variant: Set<Subject>] = [:]
        /// Every candidate, with the key it answers: what the resources and the switch are checked on.
        var products: [(ProductKey, CapacitySolver.ProductSolution)] = []
        var attentions: [CapacitySolver.AttentionSolution] = []
    }

    /// The candidates of every class under `plan` (the denoiser's, and the encoder's: without the ANE, as `GEMM` builds it).
    static func offer(_ model: M, plan: (CapacityPlan) -> CapacityPlan = { $0.forDenoiser(.on(neuralEngine: true)) }) -> Offer {
        let base = CapacityPlan(model: model, sustained: false)
        let denoiser = plan(base).constraints, encoder = plan(base).restricted(allowingANE: false).constraints
        var o = Offer()
        for variant in Variant.allCases {
            for (family, cl) in classes(variant) {
                switch cl.shape {
                case let .product(k, n, rows, format):
                    let key = ProductKey(k: k, n: n, m: rows, weight: format, phase: cl.phase)
                    let subject = Subject.product(cl.phase, format)
                    o.present[variant, default: []].insert(subject)
                    for s in CapacitySolver.candidates(key, model: model, constraints: cl.phase == .encoding ? encoder : denoiser) {
                        let lanes = Set(s.rows.filter { $0.value > 0 }.keys.map { lane($0) })
                        o.routed.insert(Cell(variant: variant, subject: subject, lanes: lanes))
                        o.routes.insert(s.route)
                        o.products.append((key, s))
                    }
                case let .attention(s, keys, heads, kv, dim, cross):
                    let key = AttentionKey(sequence: s, keys: keys, heads: heads, kvHeads: kv, headDim: dim, cross: cross)
                    o.present[variant, default: []].insert(.attention)
                    for a in CapacitySolver.candidates(key, flash: CapacitySolver.triesFlash(family: family, cross: cross), model: model,
                                                       constraints: denoiser) {
                        let lanes = Set(a.rows.filter { $0.value > 0 }.keys.map { lane($0) })
                        o.routed.insert(Cell(variant: variant, subject: .attention, lanes: lanes))
                        o.routes.insert(a.route)
                        o.attentions.append(a)
                    }
                }
            }
        }
        return o
    }

    // MARK: - The table

    struct Verdict {
        var routed: [Cell] = []
        var listed: [(Cell, Gap)] = []
        var holes: [Cell] = []
        var forbidden: [Cell] = []
        var hits: [String: Int] = [:]
    }

    /// Every cell of every version and subject the classes carry: each lane alone, then each set of two or three engines whose lanes
    /// have a route alone; routed, listed, or a hole; and every routed cell against P20.
    static func judge(_ o: Offer) -> Verdict {
        var v = Verdict()
        func listed(_ c: Cell) -> Gap? { gaps.first { $0.matches(c) } }
        for variant in Variant.allCases {
            for subject in (o.present[variant] ?? []).sorted(by: { $0.description < $1.description }) {
                let lanes = Engine.allCases.flatMap { e in Operand.allCases.map { Lane(e, $0) } }
                var alone: [Lane] = []
                var cells: [Cell] = lanes.map { Cell(variant: variant, subject: subject, lanes: [$0]) }
                for c in cells where o.routed.contains(c) { alone.append(c.lanes.first!) }
                // The sets: one lane per engine, among those with a route alone.
                let byEngine = Dictionary(grouping: alone, by: \.engine)
                var sets: [Set<Lane>] = [[]]
                for e in Engine.allCases {
                    sets += sets.flatMap { s in (byEngine[e] ?? []).map { s.union([$0]) } }
                }
                cells += sets.filter { $0.count >= 2 }.map { Cell(variant: variant, subject: subject, lanes: $0) }
                // Routed sets whose lanes are not all routed alone are judged too (P20).
                cells += o.routed.filter { $0.variant == variant && $0.subject == subject && $0.lanes.count >= 2 && !cells.contains($0) }
                for c in cells {
                    let permitted = c.lanes.allSatisfy { p20(subject, $0) }
                    if o.routed.contains(c) {
                        if permitted { v.routed.append(c) } else { v.forbidden.append(c) }
                    } else if permitted {
                        if let g = listed(c) { v.listed.append((c, g)); v.hits[g.label, default: 0] += 1 } else { v.holes.append(c) }
                    }
                }
            }
        }
        return v
    }

    static func table(_ v: Verdict) -> String {
        var out = ["routed:"] + v.routed.map { "  ✓ \($0)" }
        out += ["listed:"] + v.listed.map { "  · \($0.0)  [\($0.1.label), \($0.1.kind.rawValue)]" }
        out += ["gaps:"] + gaps.map { "  \($0.label) (\($0.kind.rawValue), \(v.hits[$0.label] ?? 0) cells): \($0.reason)" }
        out += ["dead routes:"] + deadRoutes.sorted { $0.key < $1.key }.map { "  \($0.key): \($0.value)" }
        out += ["unplanned:"] + unplanned.map { "  \($0.key): \($0.value)" }
        return out.joined(separator: "\n")
    }

    // MARK: - The guard

    @Test func everyPermittedCellIsRoutedOrListedAndNoRouteIsForbidden() {
        let v = Self.judge(Self.offer(Self.model(lines: Self.probed)))
        if ProcessInfo.processInfo.environment["SILICONED_COVERAGE_TABLE"] != nil { print(Self.table(v)) }
        #expect(v.holes.isEmpty, "a permitted cell has no route and no reason: \(v.holes.map(\.description).joined(separator: "; "))")
        #expect(v.forbidden.isEmpty, "a route P20 forbids: \(v.forbidden.map(\.description).joined(separator: "; "))")
        for g in Self.gaps {
            #expect((v.hits[g.label] ?? 0) > 0, "gap \(g.label) names no cell: stale, or filled — update the list")
            for c in v.routed where g.matches(c) {
                Issue.record("gap \(g.label) names a routed cell, \(c): the hole was filled — narrow the entry")
            }
        }
        #expect(!v.routed.isEmpty)
    }

    /// The Standard takes what the decisions gave it: bf16 on its own weights, the ANE beside the GPU on the widened ones and beside
    /// its rounded rows on the bf16 copy, the attention on fp16 and bf16 operands — the permissions of 2/11 and 3/11/2026, seen from
    /// the table.
    @Test func theStandardsDecisionsAreRoutes() {
        let routed = Self.offer(Self.model(lines: Self.probed)).routed
        let bf16 = Subject.product(.denoising, .bf16)
        for lanes: Set<Lane> in [[Lane(.gpu, .bf16)], [Lane(.gpu, .fp32), Lane(.ane, .fp16)], [Lane(.gpu, .fp32), Lane(.amx, .fp32), Lane(.ane, .fp16)],
                                 [Lane(.gpu, .bf16), Lane(.ane, .fp16)]] {
            #expect(routed.contains(Cell(variant: .standard, subject: bf16, lanes: lanes)), "\(lanes)")
        }
        for lanes: Set<Lane> in [[Lane(.gpu, .fp16)], [Lane(.gpu, .bf16)], [Lane(.gpu, .fp16), Lane(.cpu, .fp32)]] {
            #expect(routed.contains(Cell(variant: .standard, subject: .attention, lanes: lanes)), "attention \(lanes)")
        }
        // The rounded pair only where the map measured its own sharing: the engines' `gpu+ane`, measured on `gpu/fp32`, is not its κ.
        var borrowed = Self.model(lines: Self.probed)
        borrowed.sharing[M.key(Diagnostic.CapacityProbes.Context.roundedPair)] = nil
        #expect(!Self.offer(borrowed).routed.contains(Cell(variant: .standard, subject: bf16, lanes: [Lane(.gpu, .bf16), Lane(.ane, .fp16)])))
    }

    /// The 8-bit products on a rounded activation (3/11/2026): the GPU's rounded kernel alone on an int8 and on a Q8_0 weight, and beside
    /// the ANE's integers on an int8 one; a Light's K-quant blocks on the GPU's rounded kernel alone — in the denoiser of the versions
    /// that publish them, and never in the encoder.
    @Test func theRoundedEightBitProductsAreRoutes() {
        let routed = Self.offer(Self.model(lines: Self.probed)).routed
        let rounded = Lane(.gpu, .bf16)
        for variant in Variant.allCases {
            for (format, sets) in [(Format.int8, [[rounded], [rounded, Lane(.ane, .fp16)]]), (Format.q8, [[rounded]]), (Format.q4k, [[rounded]])] as [(Format, [Set<Lane>])] {
                let subject = Subject.product(.denoising, format)
                guard Self.classes(variant).contains(where: { if case let .product(_, _, _, f) = $0.1.shape { f == format && $0.1.phase == .denoising } else { false } })
                else { continue }
                for lanes in sets { #expect(routed.contains(Cell(variant: variant, subject: subject, lanes: lanes)), "\(variant) \(format) \(lanes)") }
            }
            #expect(!routed.contains { $0.variant == variant && $0.subject.phase == .encoding && $0.lanes.contains(rounded) })
        }
        #expect(routed.contains { $0.subject == .product(.denoising, .int8) }, "no version publishes int8 weights any more: update the test")
        // No version of the catalogue publishes Q8_0 weights since Z-Image's and Anima's Compacts became int8 convrot: the Q8_0
        // route stays for an imported GGUF, judged above wherever a version's classes hold it.
        #expect(routed.contains { $0.variant == .light && $0.subject == .product(.denoising, .q4k) && $0.lanes == [rounded] },
                "the Light's K-quants on a rounded activation (`gpu/q4kact`)")
    }

    @Test func aRouteReadsOnlyLinesAProbeMeasures() {
        let every = Set(Engine.allCases.flatMap { e in M.Form.allCases.map { M.Kernel(e, $0) } })
        let dead = Self.offer(Self.model(lines: every)).routes.subtracting(Self.offer(Self.model(lines: Self.probed)).routes)
        #expect(dead == Set(Self.deadRoutes.keys), "routes no map can offer: \(dead.sorted()) — listed: \(Self.deadRoutes.keys.sorted())")
        // Every attention form is one the probes time: none is read by a route the map never measures.
        let attentionForms: Set<M.Form> = [.graph, .flash, .heads, .tiles, .tiles16, .tilesbf16]
        #expect(attentionForms.isSubset(of: Set(Self.probed.map(\.form))))
        #expect(Set(M.Form.allCases).subtracting(attentionForms) == [.fp32, .bf16, .bf16act, .int8, .q8, .fp16, .int8act, .q8act, .q4kact, .int8x8],
                "a new form of kernel: say whether it is a product's or an attention's")
        #expect(M.Form.allCases.allSatisfy { $0.isProduct != attentionForms.contains($0) })
    }

    @Test func eachGapsProbeSaysWhetherTheEngineRunsIt() {
        for g in Self.gaps {
            guard let p = g.probe else { continue }
            #expect(p.assignment.isRunnable(for: p.key) == p.runs, "\(g.label): the engines \(p.runs ? "ran" : "did not run") it — update the entry")
        }
    }

    /// The engines a candidate needs are those the render's plan brings up (`CapacityPlan.Resources`), and the switch off leaves fp32.
    @Test func theCandidatesNeedOnlyWhatThePlanBuildsAndTheSwitchOffIsFP32() {
        let model = Self.model(lines: Self.probed)
        let r = CapacityPlan(model: model, sustained: false).forDenoiser(.on(neuralEngine: true)).resources
        let o = Self.offer(model)
        for (key, s) in o.products {
            let a = s.assignment
            if a.usesAMX { #expect(r.amx, "\(key) \(s.route)") }
            if a.usesANE { #expect(r.ane && (a.path != .native || r.aneInt8), "\(key) \(s.route)") }
            if a.path == .bf16 { #expect(a.roundsActivation ? r.roundedActivations : r.bf16Weights, "\(key) \(s.route)") }
            if a.path == .native {
                #expect(key.weight == .int8 ? r.int8Weights : key.weight == .q4k ? r.roundedActivations : r.q8Weights, "\(key) \(s.route)")
            }
            if a.roundsActivation { #expect(r.roundedActivations, "\(key) \(s.route)") }
            if a.quantizesActivation { #expect(r.quantizedActivations && key.weight == .int8, "\(key) \(s.route)") }
        }
        for a in o.attentions where a.assignment.usesTiles && a.assignment.tilePrecision != .fp32 || a.assignment.gpuUsesTiles && a.assignment.gpuTilePrecision != .fp32 {
            #expect(r.halfAttention, "\(a.route)")
        }
        let off = Self.offer(model, plan: { $0.restricted(accelerators: false) })
        let below = off.routed.filter { $0.lanes.contains { $0.operand != .fp32 } }
        #expect(below.isEmpty, "the accelerators off offer a precision below fp32: \(below.map(\.description))")
    }

    /// **W8A8 on the Compact's int8 weights** (6/11/2026): the GPU's int8 × int8 alone on a denoiser's int8 product, never in the
    /// encoder; the switch off (`allowsQuantizing`) takes it away and leaves the routes it had.
    @Test func theW8A8IsARouteOnInt8AndTheSwitchTakesItAway() {
        let model = Self.model(lines: Self.probed)
        let quantized = Lane(.gpu, .int8), subject = Subject.product(.denoising, .int8)
        let on = Self.offer(model).routed
        #expect(on.contains { $0.subject == subject && $0.lanes == [quantized] }, "gpu/int8x8 on the Compact")
        #expect(!on.contains { $0.subject.phase == .encoding && $0.lanes.contains(quantized) })
        let off = Self.offer(model, plan: { _ in
            CapacityPlan(model: model, sustained: false, allowsQuantizing: false).forDenoiser(.on(neuralEngine: true))
        }).routed
        #expect(!off.contains { $0.lanes.contains(quantized) })
        #expect(on.subtracting(off).allSatisfy { $0.lanes.contains(quantized) }, "the switch takes W8A8 away, nothing else")
    }

    /// The phases are the plan's: a new one is a new row of the table, and the VAE stays outside it.
    @Test func thePhasesAreTheTablesRows() {
        #expect(Phase.allCases == [.encoding, .denoising], "a new phase: give it its cells, or list it in `unplanned`")
        #expect(!Self.unplanned.isEmpty)
    }
}
