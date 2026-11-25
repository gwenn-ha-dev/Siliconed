import Foundation
@testable import SiliCore
import Testing

/// **A block's binding and its tensors' places**, on a declaration written by hand: a stream `x` (its factors beside),
/// a per-row `q`, a tensor `g` and a mask, a group `L` of two weights (`w`, `qkv`) and its own factor `L.w_down`; a form
/// of three steps (`pre` either unit's, `heads` the GPU's, an attention across rows), a pass the ANE may share.
@Suite struct Blocks {
    static var declaration: Block.Declaration {
        var d = Block.Declaration("a block", rows: 16)
        d.perRow = [.init("x", width: 4, .float16, next: "y", factor: "x_factor"), .init("x_factor", width: 1, .float16, next: "y_factor"),
                    .init("q", width: 4, .float16), .init("t", width: 4, .float16, rows: 8)]
        d.tensors = [Stage.Tensor("g", bytes: 8), Stage.Tensor("mask-a", bytes: 32), Stage.Tensor("mask-b", bytes: 32), Stage.Tensor("L.w_down", bytes: 32)]
        d.groups = [Block.Group("L", form: "s", weights: [])]
        d.cycle = ["L"]
        d.forms = [Block.Form("s", steps: [.init("pre"), .init("heads", on: .gpu, bind: ["qkv": .rows("q")]), .init("attention", on: .gpu, whole: ["q"])],
                              bind: ["mask": .tensor("mask-a")]) { _, b in module(rows: b.rows) }]
        d.passes = [Block.Pass("p", rows: 0..<16, shareable: 0..<16), Block.Pass("other", rows: 0..<16, bind: ["mask": .tensor("mask-b")])]
        return d
    }
    static var resolver: Block.Resolver {
        let d = declaration
        return Block.Resolver(d, forms: ["s": d.forms[0]], weights: ["L": ["w", "qkv"], "": ["emb"]])
    }

    /// The ANE's step: the stream and its factors in, the next out; `g` and the weight `w` read.
    static func module(rows: Int) -> Aicode.Module {
        var m = Aicode.Module(arguments: [("x", Aicode.tensor([1, rows, 4], .f16)), ("x_factor", Aicode.tensor([1, rows, 1], .f16)),
                                          ("g", Aicode.tensor([1, 1, 4], .f16)), ("w", Aicode.tensor([4, 4], .si8)), ("w_down", Aicode.tensor([4, 2], .f16))])
        let b = m.body
        m.outputs = [("y", b.binary(.mul, m.argument("x"), b.scalar(2, .f16))), ("y_factor", b.binary(.mul, m.argument("x_factor"), b.scalar(1, .f16)))]
        return m
    }

    /// **A block's arithmetic** (`ThisMac.capacities`): the codes where a group reads them, f16 otherwise.
    @Test func anArithmeticIsWhatTheANEReads() {
        var d = Self.declaration
        #expect(Block.arithmetic(d) == "f16")
        d.groups[0].weights = [("w", "w.weight", .halfTransposed)]
        #expect(Block.arithmetic(d) == "f16")
        d.groups[0].weights = [("w", "w", .codes)]
        #expect(Block.arithmetic(d) == "codes")
    }

    /// **A function's FLOP, counted by hand** (`Aicode.Module.flops`): `linear` x [1, 64, 128] · wᵀ, w [256, 128] —
    /// 2·64·256·128; a batched product [2, 64, 32] · [2, 32, 48] — 2·2·64·48·32; a convolution x [1, 8, 16, 16] by a kernel
    /// [4, 2, 3, 3] in 4 groups, padded to the same size — 2·(4·16·16)·(2·3·3); the elementwise op between them, none.
    @Test func aFunctionCountsItsProductsFLOP() {
        var m = Aicode.Module(arguments: [("x", Aicode.tensor([1, 64, 128], .f16)), ("w", Aicode.tensor([256, 128], .f16)),
                                          ("a", Aicode.tensor([2, 64, 32], .f16)), ("b", Aicode.tensor([2, 32, 48], .f16)),
                                          ("i", Aicode.tensor([1, 8, 16, 16], .f16)), ("k", Aicode.tensor([4, 2, 3, 3], .f16))])
        let body = m.body
        let y = body.linear(m.argument("x"), m.argument("w"))
        let z = body.matmul(m.argument("a"), m.argument("b"), result: Aicode.tensor([2, 64, 48], .f16))
        let c = body.convolution(m.argument("i"), m.argument("k"), stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 4,
                                 result: Aicode.tensor([1, 4, 16, 16], .f16))
        m.outputs = [("y", body.binary(.mul, y, y)), ("z", z), ("c", c)]
        #expect(m.flops == 2 * 64 * 256 * 128 + 2 * 2 * 64 * 48 * 32 + 2 * (4 * 16 * 16) * (2 * 3 * 3))
        #expect(m.trimmed().flops == m.flops)
    }

    /// **The ANE's unit, by arithmetic**: 512 on the codes, 1024 on f16, 512 for both under two units of 1024.
    @Test func theUnitIsTheArithmetics() {
        #expect(Block.neuralUnit("codes", shareable: 4096) == 512 && Block.neuralUnit("f16", shareable: 4096) == 1024)
        #expect(Block.neuralUnit("f16", shareable: 2048) == 1024 && Block.neuralUnit("f16", shareable: 2047) == 512)
        #expect(Block.neuralUnit("codes", shareable: 1024) == 512)
    }

    @Test func anArgumentIsBoundByItsNameFirstMatch() {
        let r = Self.resolver, d = Self.declaration
        let (g, p, other) = (d.groups[0], d.passes[0], d.passes[1])
        let (pre, heads, attention) = (d.forms[0].steps[0], d.forms[0].steps[1], d.forms[0].steps[2])
        func at(_ a: String, input: Bool = true, pass: Block.Pass? = p, step: Block.Step? = pre, call: [String: Block.Bind] = [:]) -> Block.Target {
            r.resolve(a, input: input, group: g, pass: pass, step: step, call: call)
        }
        #expect(at("x") == .rows("x", next: false, whole: false))
        #expect(at("y", input: false) == .rows("x", next: true, whole: false))
        #expect(at("y_factor", input: false) == .rows("x_factor", next: true, whole: false))
        // A weight's name for an input; the step's binding before it.
        #expect(at("qkv") == .weight("qkv"))
        #expect(at("qkv", step: heads) == .rows("q", next: false, whole: false))
        #expect(at("emb") == .weight("emb"))
        #expect(at("q", step: attention) == .rows("q", next: false, whole: true))
        // The group's own tensor before a tensor of the same name.
        #expect(at("w_down") == .tensor("L.w_down", row: nil))
        // The call's binding, then the step's, the pass's, the form's.
        #expect(at("mask") == .tensor("mask-a", row: nil))
        #expect(at("mask", pass: other) == .tensor("mask-b", row: nil))
        #expect(at("mask", pass: other, call: ["mask": .row("g", 2)]) == .tensor("g", row: 2))
    }

    /// A group's held weights bound by their own names, under the group's prefix, after its other weights: in a run's
    /// calls and in an edge's outside one (`call(_:group:)`); another group's never.
    @Test func aGroupsHeldWeightIsBoundUnderItsName() {
        let d = Self.declaration, r = Block.Resolver(d, forms: ["s": d.forms[0]], weights: ["L": ["w", "qkv"], "": ["emb"]], held: ["L": ["ck", "w"]])
        let other = Block.Group("M", form: "s", weights: [])
        #expect(r.resolve("ck", input: true, group: d.groups[0], pass: nil, step: nil, call: [:]) == .weight("L.ck"))
        #expect(r.resolve("w", input: true, group: d.groups[0], pass: nil, step: nil, call: [:]) == .weight("w"))
        #expect(r.resolve("emb", input: true, group: d.groups[0], pass: nil, step: nil, call: [:]) == .weight("emb"))
        #expect(r.resolve("emb", input: true, group: other, pass: nil, step: nil, call: [:]) == .weight("emb"))
    }

    /// The ring's slots lay out as most groups do, the first such in the loop's order: two groups of float weights
    /// before twenty of codes keep the slots the codes'.
    @Test func theSlotsAreTheCommonestLayout() {
        let floats = ["w [4, 4] float16"], codes = ["w [4, 4] int8", "w_scale [4] float16"]
        var layouts = ["b0": floats, "b1": floats]
        for i in 2..<22 { layouts["b\(i)"] = codes }
        let order = (0..<22).map { "b\($0)" }
        #expect(Block.slot(layouts, order: order) == "b2")
        #expect(Block.slot(layouts.mapValues { _ in floats }, order: order) == "b0")
        #expect(Block.slot([:], order: []) == nil)
    }

    @Test func whatTheANEReadsIsASurface() {
        let d = Self.declaration, forms = ["s": d.forms[0]]
        let signatures = [Block.neural("s", "pre", 8): Block.Signature(Self.module(rows: 8))]
        let beside = Dictionary(uniqueKeysWithValues: Block.places(d, forms: forms, signatures: signatures, resolver: Self.resolver, unit: 8).map { ($0.name, $0) })
        #expect(beside["x#0"]?.place == .units(width: 4, height: 8, element: 2) && beside["x#1"]?.place == .units(width: 4, height: 8, element: 2))
        #expect(beside["x_factor#1"]?.place == .units(width: 8, height: 1, element: 2))
        #expect(beside["x#0"]?.bytes == 16 * 4 * 2)
        #expect(beside["q"]?.place == .heap && beside["mask-a"]?.place == .heap)
        #expect(beside["t"]?.bytes == 8 * 4 * 2)
        #expect(beside["g"]?.place == .surface(shape: [1, 1, 4], element: 2))
        #expect(beside["L.w_down"]?.place == .surface(shape: [4, 2], element: 2))
        let alone = Block.places(d, forms: forms, signatures: signatures, resolver: Self.resolver, unit: nil)
        #expect(alone.allSatisfy { $0.place == .heap })
        #expect(Set(alone.map(\.name)) == ["x#0", "x#1", "x_factor#0", "x_factor#1", "q", "t", "g", "mask-a", "mask-b", "L.w_down"])
    }

    /// A copy reads and writes rows of one width and type, within each tensor's rows (a short one's, the block's).
    @Test func aCopyIsRowsOfOneWidthWithinBoth() {
        let d = Self.declaration
        let (x, q, t) = (d.perRow[0], d.perRow[2], d.perRow[3])
        let a = Block.copied(t, rows: 0..<6, to: q, at: 10, block: d.rows)
        #expect(a.from == 0..<48 && a.into == 80..<128)
        let b = Block.copied(x, rows: 14..<16, to: t, at: 6, block: d.rows)
        #expect(b.from == 112..<128 && b.into == 48..<64)
    }

    /// A call's height: its rows rounded up to 64, never past the per-row tensors' rows; under a cut, equal calls.
    @Test func aCallIsItsRowsRoundedTo64() {
        typealias T = Block.Tile
        #expect(Block.calls(0..<4144, cut: nil, limit: 4608) == [T(start: 0, rows: 4160)])
        #expect(Block.calls(2048..<4144, cut: nil, limit: 4608) == [T(start: 2048, rows: 2112)])
        #expect(Block.calls(4096..<4144, cut: nil, limit: 4608) == [T(start: 4096, rows: 64)])
        #expect(Block.calls(0..<4096, cut: 4096, limit: 4608) == [T(start: 0, rows: 4096)])
        #expect(Block.calls(0..<4144, cut: 2048, limit: 4608) == [T(start: 0, rows: 1408), T(start: 1408, rows: 1408), T(start: 2816, rows: 1408)])
        #expect(Block.calls(0..<1056, cut: nil, limit: 1056) == [T(start: 0, rows: 1056)])
        #expect(Block.calls(0..<1100, cut: 1024, limit: 1120) == [T(start: 0, rows: 576), T(start: 576, rows: 544)])
        #expect(Block.calls(64..<64, cut: nil, limit: 128).isEmpty)
    }

    /// A form is built at the calls of the passes that serve it alone.
    @Test func aFormIsBuiltForThePassesItRunsOver() {
        var d = Block.Declaration("two passes", rows: 256)
        d.forms = Self.declaration.forms
        d.passes = [Block.Pass("long", rows: 0..<256), Block.Pass("short", rows: 0..<64, forms: ["another"])]
        #expect(Set(Block.steps(d, d.forms, neural: 0, cut: nil).keys) == Set(["pre", "heads", "attention"].map { Block.gpu("s", $0, 256) }))
        d.passes[1].forms = nil
        #expect(Block.steps(d, d.forms, neural: 0, cut: nil).count == 6)
    }

    /// **The best share is on its unit's grid**: the cycle ran at 3072 rows; 1024's grid has 3.0 s at best, 512's 2.5 s
    /// at 2560 rows — a share the taller unit cannot take (MOSS-Image on the M1 Pro, 25/11/2026: 8.55 s a step at 2560
    /// against 9.15 at 3072).
    @Test func theBestShareIsOnItsUnitsGrid() {
        typealias C = Block.Timed
        let cycle: [[C]] = [[C(kind: .neural, step: "pre", phase: 0, rows: 3072, seconds: 3.0), C(kind: .past, step: "pre", phase: 0, rows: 1024, seconds: 1.5)]]
        let tall = Block.bestShare(cycle, measured: 3072, unit: 1024, shareable: 4096)
        let fine = Block.bestShare(cycle, measured: 3072, unit: 512, shareable: 4096)
        #expect(abs(tall.seconds - 3.0) < 1e-9 && tall.rows % 1024 == 0)
        #expect(abs(fine.seconds - 2.5) < 1e-9 && fine.rows == 2560)
    }

    /// The share kept crosses into units in rows, rounded, the GPU keeping a unit.
    @Test func aShareIsCountedInUnits() {
        #expect(Block.count(rows: 2048, of: 512, shareable: 4096) == 4)
        #expect(Block.count(rows: 1536, of: 1024, shareable: 4096) == 2)
        #expect(Block.count(rows: 3584, of: 1024, shareable: 4096) == 3)
        #expect(Block.count(rows: 100, of: 1024, shareable: 4096) == 1)
        #expect(Block.count(rows: 4096, of: 512, shareable: 4096) == 7)
    }

    /// The entries of the rules before (`units`, `neuralUnits`: a unit by key; `capacities`' `unit`: a unit by
    /// arithmetic, tried; `paces`; `zeros`: a block's share of none, timed by key) are no decision: the unit is the
    /// arithmetic's (`neuralUnit`), and the share of none is predicted from κ, measured once by arithmetic.
    @Test func anOldEntryDecidesNothing() throws {
        let old = #"{"units": {"block k": 256}, "neuralUnits": {"block k": 512}, "paces": {"pace p": 1.5}, "zeros": {"block k": 5.6}, "flight": {"block k": 1024}}"#
        let m = try JSONDecoder().decode(ThisMac.self, from: Data(old.utf8))
        #expect(m.capacities == nil && m.flight?["block k"] == 1024)
        let tried = #"{"capacities": {"codes": {"unit": 1024, "ratio": 0.37}}}"#
        #expect(try JSONDecoder().decode(ThisMac.self, from: Data(tried.utf8)).capacities == ["codes": ThisMac.Capacity(ratio: 0.37)])
    }

    /// **A move pays** where its gain a cycle passes the noise, and over the cycles left its cost and the noise; never
    /// said, over one cycle.
    @Test func aMovePaysOverTheCyclesLeft() {
        func pays(_ gain: Double, left: Int?) -> Bool { Block.pays(gain, noise: 0.02, cost: 0.3, left: left) }
        #expect(!pays(0.01, left: 9))       // under the noise
        #expect(!pays(0.1, left: nil))      // one cycle: 0.1 < 0.32
        #expect(pays(0.4, left: nil))
        #expect(pays(0.1, left: 5))         // 0.5 > 0.32
        #expect(!pays(0.1, left: 0))        // the last: one cycle at most
    }

    /// **A value smoothed** at a half-life of two cycles: the first taken as it is.
    @Test func aValueIsSmoothedOverTwoCycles() {
        #expect(Block.smoothing(nil, 1) == 1)
        let twice = Block.smoothing(Block.smoothing(1, 0), 0)
        #expect(abs(twice - 0.5) < 0.01)
    }

    /// **The best share among those admitted**: one B refuses is never predicted, even the fastest.
    @Test func theBestShareIsOneBAdmits() {
        typealias T = Block.Timed
        let cycle = [[T(kind: .neural, step: "s", phase: 0, rows: 1024, seconds: 1.0),
                      T(kind: .past, step: "s", phase: 0, rows: 3072, seconds: 3.0)]]
        // A row 1 ms on both: the balance at half.
        #expect(Block.bestShare(cycle, measured: 1024, unit: 1024, shareable: 4096).rows == 2048)
        #expect(Block.bestShare(cycle, measured: 1024, unit: 1024, shareable: 4096, admitted: [1, 3]).rows == 1024)
    }

    /// **A group's calls summed by step**: what a share's count of calls changes, its steps do not.
    @Test func callsAreSummedByStep() {
        typealias T = Block.Timed
        let s = T.summed([T(kind: .neural, step: "a", phase: 0, rows: 512, seconds: 1, flops: 10),
                          T(kind: .neural, step: "a", phase: 0, rows: 512, seconds: 2, flops: 10),
                          T(kind: .past, step: "a", phase: 0, rows: 3072, seconds: 4)])
        #expect(s.count == 2 && s[0].rows == 1024 && s[0].seconds == 3 && s[0].flops == 20 && s[1].kind == .past)
    }

    /// A phase lasts the longer of its units; a `gpu` step what it did. The GPU's past rows shrink as the ANE's grow.
    @Test func aShareIsPredictedByPhase() {
        typealias C = Block.Timed
        let pass: [C] = [C(kind: .neural, step: "pre", phase: 0, rows: 256, seconds: 1), C(kind: .past, step: "pre", phase: 0, rows: 768, seconds: 1.5),
                         C(kind: .fixed, step: "heads", phase: 1, rows: 1024, seconds: 0.5),
                         C(kind: .neural, step: "out", phase: 1, rows: 256, seconds: 0.25), C(kind: .past, step: "out", phase: 1, rows: 768, seconds: 0.75)]
        #expect(Block.predicted([pass], measured: 256, rows: 256) == 0.5 + 1.5 + 0.75)
        #expect(Block.predicted([pass], measured: 256, rows: 512) == 0.5 + 2 + max(0.5, 0.5))
        #expect(Block.predicted([pass, pass], measured: 256, rows: 256, neural: 2) == 2 * (0.5 + 2 + 0.75))
        let f = Self.declaration.forms[0]
        #expect(f.steps.indices.map { Block.phase(f, $0) } == [0, 0, 1])
    }

    /// **The share of none, predicted from a cycle at a share and κ** (`ThisMac.Capacity.alone`): the GPU's past rows over
    /// every row, κ times faster alone, its `gpu` steps as they were; and κ read back from the cycle at 0 it predicts.
    /// Z-Image Compact 1024² on the M1 Pro (25/11/2026): the GPU beside the ANE at 2048 rows 6.60 s for 2108 rows a layer,
    /// its `gpu` steps 3.15 s; at 0, 16.14 s: κ 1.00.
    @Test func theShareOfNoneIsPredictedByKappa() {
        typealias C = Block.Timed
        let cycle: [[C]] = [[C(kind: .neural, step: "ffn", phase: 0, rows: 2048, seconds: 3.84),
                             C(kind: .past, step: "ffn", phase: 0, rows: 2108, seconds: 6.60),
                             C(kind: .fixed, step: "heads", phase: 1, rows: 4156, seconds: 3.15)]]
        let beside = 6.60 / 2108 * 4156
        #expect(abs(Block.predicted(cycle, measured: 2048, rows: 0) - (beside + 3.15)) < 1e-9)
        #expect(abs(Block.predicted(cycle, measured: 2048, rows: 0, alone: 1.24) - (beside / 1.24 + 3.15)) < 1e-9)
        // κ leaves the share measured, and those past it, as measured.
        #expect(Block.predicted(cycle, measured: 2048, rows: 2048, alone: 1.24) == Block.predicted(cycle, measured: 2048, rows: 2048))
        #expect(Block.predicted(cycle, measured: 2048, rows: 3072, alone: 1.24) == Block.predicted(cycle, measured: 2048, rows: 3072))
        #expect(Block.aloneRate(cycle, measured: 2048, zero: beside / 1.24 + 3.15).map { abs($0 - 1.24) < 1e-9 } == true)
        #expect(abs(Block.aloneRate(cycle, measured: 2048, zero: 16.14)! - 1.0) < 0.01)
        // A cycle at 0 no longer than the `gpu` steps tells nothing.
        #expect(Block.aloneRate(cycle, measured: 2048, zero: 3.0) == nil)
    }

    /// **Between none and the share measured, the GPU's seconds a row on a straight line** from 1/κ of them to them: the
    /// GPU slows as the ANE beside it takes more. Krea 2 Standard 1024² on the M5 Pro (25/11/2026): at 2048 rows the ANE
    /// 6.55 s, the GPU beside 2.93 for 2048 rows, its `gpu` steps 1.00; at 0, 5.40 s; 1024 rows told took 4.82. The
    /// constant predicted 1024 at 5.40, no faster than none, and the share of none stayed.
    @Test func theGPUSlowsAsTheANETakesMore() {
        typealias C = Block.Timed
        let cycle: [[C]] = [[C(kind: .neural, step: "ffn", phase: 0, rows: 2048, seconds: 6.55),
                             C(kind: .past, step: "ffn", phase: 0, rows: 2048, seconds: 2.93),
                             C(kind: .fixed, step: "heads", phase: 1, rows: 4096, seconds: 1.00)]]
        let kappa = Block.aloneRate(cycle, measured: 2048, zero: 5.40)!
        #expect(abs(kappa - 1.332) < 0.001)
        #expect(abs(Block.predicted(cycle, measured: 2048, rows: 0, alone: kappa) - 5.40) < 1e-9)
        #expect(abs(Block.predicted(cycle, measured: 2048, rows: 1024) - 5.395) < 0.001)
        #expect(abs(Block.predicted(cycle, measured: 2048, rows: 1024, alone: kappa) - 4.848) < 0.001)
        let best = Block.bestShare(cycle, measured: 2048, unit: 1024, shareable: 4096, alone: kappa)
        #expect(best.rows == 1024 && best.seconds < 5.40 - 0.5)
    }

    /// A tapped run is the GPU's alone over every row: a block declared tapped has the GPU's functions of the whole pass
    /// beside those past the ANE's, a step that taps (`Step.taps`) in its tapped form under a name of its own.
    @Test func aTappedRunHasItsOwnFunctions() {
        var d = Self.declaration
        d.forms[0].steps[1].taps = true
        d.passes = [d.passes[0]]
        let used = [d.forms[0]]
        let plain = Block.steps(d, used, neural: 8, cut: nil)
        #expect(plain.keys.sorted() == ["s.attention-16", "s.heads-16", "s.pre-8"] && plain.values.allSatisfy { !$0.tapped })
        d.tapped = true
        let tapped = Block.steps(d, used, neural: 8, cut: nil)
        #expect(tapped.keys.sorted() == ["s.attention-16", "s.heads+taps-16", "s.heads-16", "s.pre-16", "s.pre-8"])
        #expect(tapped["s.heads+taps-16"]?.tapped == true && tapped["s.heads-16"]?.tapped == false && tapped["s.pre-16"]?.tapped == false)
        #expect(Block.gpu("s", "heads", 16, tapped: true) == "s.heads+taps-16" && Block.gpu("s", "heads", 16) == "s.heads-16")
        #expect(Block.Build(rows: 16).tapped == false)
    }

    /// An edge of the ANE's that reads codes is refused at the opening, a declaration's error (the ANE reads codes only
    /// from a shared form's copy); the same edge the GPU's is built.
    @Test func anEdgeOfTheANEsReadsNoCodes() throws {
        var d = Self.declaration
        d.edges = ["head": { _ in Self.module(rows: 1) }]
        #expect(try Block.edges(d)[Block.edge("head")]?.unit == .gpu)
        d.neuralEdges = ["head"]
        #expect(throws: Failure.self) { try Block.edges(d) }
    }

    @Test func aSignatureIsTheModulesShapes() {
        let s = Block.Signature(Self.module(rows: 8))
        #expect(s.inputs.map(\.0) == ["x", "x_factor", "g", "w", "w_down"])
        #expect(s.inputs[0].1 == [1, 8, 4] && s.inputs[0].2 == .float16 && s.inputs[3].2 == .int8)
        #expect(s.outputs.map(\.0) == ["y", "y_factor"] && s.outputs[1].1 == [1, 8, 1])
    }

    /// A per-row tensor of each group, of its own rows: `<group>.<name>`, named or bound by its name, off the ANE's surfaces; a pass's calls stop at its
    /// limit; a run's steps stop at `until`.
    @Test func aTensorOfEachGroupAndAPassOfItsOwnLimit() {
        var d = Self.declaration
        d.groups.append(Block.Group("M", form: "s", weights: []))
        d.perRow += ["L", "M"].map { .init("\($0).kp", width: 4, .float16, rows: 6) }
        d.tensors.append(Stage.Tensor("M.w_down", bytes: 32))
        d.passes[1].limit = 12
        let forms = ["s": d.forms[0]]
        let signatures = [Block.neural("s", "pre", 8): Block.Signature(Self.module(rows: 8))]
        let r = Block.Resolver(d, forms: forms, weights: ["L": ["w", "qkv"], "M": ["w", "qkv"], "": ["emb"]])
        let placed = Dictionary(uniqueKeysWithValues: Block.places(d, forms: forms, signatures: signatures, resolver: r, unit: 8).map { ($0.name, $0) })
        #expect(placed["L.kp"]?.bytes == 6 * 4 * 2 && placed["M.kp"]?.place == .heap && placed["kp"] == nil)
        #expect(r.resolve("kp", input: true, group: d.groups[1], pass: d.passes[1], step: d.forms[0].steps[2], call: [:]) == .rows("M.kp", next: false, whole: false))
        #expect(r.resolve("k", input: true, group: d.groups[1], pass: d.passes[1], step: nil, call: ["k": .rows("kp")]) == .rows("M.kp", next: false, whole: false))
        let called = Block.steps(d, [d.forms[0]], neural: 0, cut: nil)
        #expect(called[Block.gpu("s", "heads", 16)] != nil && called[Block.gpu("s", "heads", 12)] != nil)
        #expect(Block.calls(0..<10, cut: nil, limit: 12) == [Block.Tile(start: 0, rows: 12)] && Block.calls(0..<10, cut: nil, limit: 10) == [Block.Tile(start: 0, rows: 10)])
        #expect(Block.steps(of: d.forms[0], until: "heads").map(\.element.name) == ["pre", "heads"])
        #expect(Block.steps(of: d.forms[0], until: nil).count == 3)
    }
}

/// **What a block's functions hold** (`Block.extent`): a step's functions named apart from their rows, their results'
/// elements summed by name; the same kin at any height, another for other ops.
@Suite struct BlockExtents {
    /// A step over `rows` rows of 8: `x · w`, then its SiLU — results of `[rows, 16]` each.
    static func step(_ rows: Int, silu: Bool = true) -> Stage.Function {
        let T = Aicode.tensor
        var m = Aicode.Module(arguments: [("x", T([rows, 8], .f16)), ("w", T([8, 16], .f16))])
        let product = m.body.matmul(0, 1, result: T([rows, 16], .f16))
        m.outputs = [("y", silu ? m.body.silu(product) : product)]
        nonisolated(unsafe) let module = m
        return Stage.Function("asset-\(rows)") { module }
    }

    @Test func aStepIsNamedApartFromItsRows() {
        #expect(Block.unsized("layer.ffn-4160") == "layer.ffn" && Block.unsized("layer.ffn+taps-64") == "layer.ffn+taps")
        #expect(Block.unsized("edge-embed-2") == "edge-embed-2" && Block.unsized("ane-layer.ffn@512") == "ane-layer.ffn@512")
    }

    /// Two calls' heights over one pass (2048, the last 1024 rows shorter): their elements summed — twice the functions
    /// hold more than one of 2048 rows, and the kin is the same.
    @Test func theCallsOfAHeightAreSummedByName() throws {
        let one = try Block.extent(["f.s-2048": Self.step(2048)], inFlight: 1)
        let two = try Block.extent(["f.s-2048": Self.step(2048), "f.s-1024": Self.step(1024)], inFlight: 1)
        let pass = try Block.extent(["f.s-4096": Self.step(4096)], inFlight: 1)
        #expect(one.kin == two.kin && two.kin == pass.kin)
        // The largest of the ratios: the products' results are as tall as their rows; a second function brings its
        // own constants (the SiLU's), a few elements more.
        #expect(two.ratio(over: one)! >= 1.5 && two.ratio(over: one)! < 1.001 * 1.5 && pass.ratio(over: one) == 2)
        #expect(one.ratio(over: pass)! > 0.5 && one.ratio(over: pass)! < 1)
        #expect(one.elements["f.s results"]! > 2048 * 16 && one.elements["f.s largest"] == 2048 * 16)
    }

    /// Other ops are another kin: no reading of one bounds the other.
    @Test func otherOpsAreAnotherKin() throws {
        let a = try Block.extent(["f.s-64": Self.step(64)], inFlight: 1), b = try Block.extent(["f.s-64": Self.step(64, silu: false)], inFlight: 1)
        #expect(a.kin != b.kin && a.ratio(over: b) == nil)
    }
}

/// **A block never run on this Mac starts where its ANE and GPU finish together** (`Block.startRows`): S · ρ / (1 + ρ);
/// half where no ratio was measured.
@Test func aNewBlockStartsWhereBothUnitsFinishTogether() {
    #expect(Block.startRows(ratio: nil, shareable: 4096) == 2048)
    #expect(Block.startRows(ratio: 1.0 / 3, shareable: 4096) == 1024)
    #expect(Block.startRows(ratio: 0.37, shareable: 4096) == 1106)
}
