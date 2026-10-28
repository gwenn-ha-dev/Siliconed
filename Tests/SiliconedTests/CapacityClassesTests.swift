import Foundation
import XCTest
@testable import Siliconed

/// **The catalogue of a render's computations** (`CapacityClasses.swift`), the prediction built on it, and the route
/// table completed to heights nobody measured (`RouteTableHeights.swift`). No GPU, no weights: the families' `Config`
/// defaults, the formats, and synthetic plans.
final class CapacityClassesTests: XCTestCase {
    typealias CC = CapacityClasses
    typealias CM = CapacityMeasure

    // MARK: helpers

    private func classes(_ family: Family, _ variant: Variant = .standard, w: Int = 1024, h: Int = 1024, faster: Bool = true,
                         encoder: Bool = false, prompt: Int = 11) -> [CC.Class] {
        CC.classes(for: .init(family: family, variant: variant, width: w, height: h, fasterRendering: faster,
                              promptTokens: prompt, includesEncoder: encoder))!
    }

    private func products(_ cs: [CC.Class], phase: CC.Phase = .denoising) -> [CC.Class] {
        cs.filter { if case .product = $0.shape { return $0.phase == phase } else { return false } }
    }

    private func executions(_ cs: [CC.Class], _ family: Family, k: Int, n: Int, rows: Int) -> Int {
        cs.filter { if case let .product(kk, nn, r, _) = $0.shape { return kk == k && nn == n && r == rows } else { return false } }
            .reduce(0) { $0 + ($1.executions[family] ?? 0) }
    }

    private func attentions(_ cs: [CC.Class]) -> [CC.Class] {
        cs.filter { if case .attention = $0.shape { return true } else { return false } }
    }

    private func rows(_ cs: [CC.Class]) -> Set<Int> {
        Set(products(cs).compactMap { if case let .product(_, _, r, _) = $0.shape { return r } else { return nil } })
    }

    // MARK: the published lists

    func testZImageAtFullSizeHasTheInventorysProducts() {
        let cs = classes(.zImage, faster: false)
        // 128 projections + 64 + 32 SwiGLU + 14 context refiners + 3 embedders, per evaluation, 8 evaluations.
        XCTAssertEqual(products(cs).reduce(0) { $0 + ($1.executions[.zImage] ?? 0) }, 241 * 8)
        XCTAssertEqual(rows(cs), [4128, 4096, 32])
        XCTAssertEqual(executions(cs, .zImage, k: 3840, n: 3840, rows: 4128), 120 * 8)
        XCTAssertEqual(executions(cs, .zImage, k: 3840, n: 3840, rows: 4096), 8 * 8)
        XCTAssertEqual(executions(cs, .zImage, k: 3840, n: 3840, rows: 32), 8 * 8)
        XCTAssertEqual(executions(cs, .zImage, k: 3840, n: 10240, rows: 4128), 60 * 8)
        XCTAssertEqual(executions(cs, .zImage, k: 10240, n: 3840, rows: 4128), 30 * 8)
        XCTAssertEqual(executions(cs, .zImage, k: 64, n: 3840, rows: 4096), 8)
        XCTAssertEqual(executions(cs, .zImage, k: 2560, n: 3840, rows: 32), 8)
        // The attentions: 30 layers over the joint sequence, 2 image refiners; the context refiners' 32 rows are no class.
        let a = attentions(cs)
        XCTAssertEqual(a.count, 2)
        XCTAssertEqual(a.first { $0.shape == .attention(sequence: 4128, keys: 4128, heads: 30, kvHeads: 30, headDim: 128, cross: false) }?.executions[.zImage], 30 * 8)
        XCTAssertEqual(a.first { $0.shape == .attention(sequence: 4096, keys: 4096, heads: 30, kvHeads: 30, headDim: 128, cross: false) }?.executions[.zImage], 2 * 8)
    }

    func testFasterRenderingAddsTheHalfGrid() {
        // Z-Image 1024²: 3 evaluations at 1024 + 32 rows, 5 at 4096 + 32.
        let z = classes(.zImage)
        XCTAssertEqual(rows(z), [1056, 1024, 4128, 4096, 32])
        XCTAssertEqual(executions(z, .zImage, k: 3840, n: 3840, rows: 1056), 120 * 3)
        XCTAssertEqual(executions(z, .zImage, k: 3840, n: 3840, rows: 4128), 120 * 5)
        XCTAssertEqual(executions(z, .zImage, k: 3840, n: 3840, rows: 32), 8 * 8, "the context refiners run at 32 rows at every evaluation")
        // 512²: no half grid (the 512-px floor); 1024×1536: the half grid is 512×768, 1 536 image rows.
        XCTAssertEqual(rows(classes(.zImage, w: 512, h: 512)), [1056, 1024, 32])
        XCTAssertEqual(rows(classes(.zImage, w: 1024, h: 1536)), [1568, 1536, 6176, 6144, 32])
        // The text rows and counts of the other families, from the families' configs.
        XCTAssertEqual(rows(classes(.krea2)).intersection([1040, 4112]), [1040, 4112])
        XCTAssertEqual(executions(classes(.krea2), .krea2, k: 6144, n: 6144, rows: 1040), 3 * 28 * 4, "Krea 2: 4 half-size evaluations")
        XCTAssertEqual(executions(classes(.krea2), .krea2, k: 6144, n: 6144, rows: 4112), 3 * 28 * 4)
        XCTAssertEqual(executions(classes(.mossImage), .mossImage, k: 3072, n: 3072, rows: 1045), 42, "MOSS-Image: one half-size evaluation")
        XCTAssertEqual(executions(classes(.mossImage), .mossImage, k: 3072, n: 3072, rows: 4117), 42 * 7)
        XCTAssertEqual(executions(classes(.qwenImage21), .qwenImage21, k: 4096, n: 4096, rows: 1024), 4 * 32 * 3)
        XCTAssertEqual(executions(classes(.anima), .anima, k: 2048, n: 2048, rows: 1024), 6 * 28 * 3)
        // Not at 6 144+ rows of Krea 2 and MOSS at 512²: nothing under the floor.
        XCTAssertEqual(rows(classes(.mossImage, w: 512, h: 512)).intersection([1045]), [1045])
        XCTAssertEqual(executions(classes(.mossImage, w: 512, h: 512), .mossImage, k: 3072, n: 3072, rows: 1045), 42 * 8)
        XCTAssertFalse(classes(.zImage, faster: false).contains { if case .product(_, _, 1056, _) = $0.shape { return true } else { return false } })
    }

    func testQwenMLPRunsByChunksAndAnimaHasItsCrossAttention() {
        // 1024×1536 full grid: 6 144 rows → two MLP chunks of 3 072; one attention over 6 144 query rows and 6 144 + 19 keys.
        let q = classes(.qwenImage21, w: 1024, h: 1536)
        XCTAssertEqual(executions(q, .qwenImage21, k: 4096, n: 12288, rows: 3072), 2 * 2 * 32 * 5, "gate and proj, two chunks, 5 full evaluations")
        XCTAssertEqual(executions(q, .qwenImage21, k: 12288, n: 4096, rows: 3072), 2 * 32 * 5)
        XCTAssertEqual(executions(q, .qwenImage21, k: 4096, n: 12288, rows: 4096) + executions(q, .qwenImage21, k: 4096, n: 12288, rows: 6144), 0)
        XCTAssertNotNil(q.first { $0.shape == .attention(sequence: 6144, keys: 6144 + 19, heads: 32, kvHeads: 32, headDim: 128, cross: false) })
        // The prefill: 19 rows through every layer, the last one only for its keys and values.
        XCTAssertEqual(executions(q, .qwenImage21, k: 4096, n: 4096, rows: 19), 4 * 31 + 2)
        // Anima: cross-attention keys are the 512 padded text rows, its k/v products run once per render at 512 rows.
        let a = classes(.anima, faster: false)
        XCTAssertEqual(executions(a, .anima, k: 1024, n: 2048, rows: 512), 56)
        let cross = a.first { $0.shape == .attention(sequence: 4096, keys: 512, heads: 16, kvHeads: 16, headDim: 128, cross: true) }
        XCTAssertEqual(cross?.executions[.anima], 28 * 8)
        // Krea 2: GQA 48/12.
        XCTAssertNotNil(classes(.krea2, faster: false).first { $0.shape == .attention(sequence: 4112, keys: 4112, heads: 48, kvHeads: 12, headDim: 128, cross: false) })
        XCTAssertEqual(executions(classes(.krea2, faster: false), .krea2, k: 6144, n: 1536, rows: 4112), 2 * 28 * 8)
        // MOSS-Image: fused qkv and gate/up, the two text blocks at 21 rows.
        let m = classes(.mossImage, faster: false)
        XCTAssertEqual(executions(m, .mossImage, k: 3072, n: 9216, rows: 4117), 42 * 8)
        XCTAssertEqual(executions(m, .mossImage, k: 3072, n: 16384, rows: 21), 2 * 8)
        XCTAssertEqual(executions(m, .mossImage, k: 4096, n: 3072, rows: 21), 8)
    }

    func testTheEncoderAtSmallM() {
        let z = CC.classes(for: .init(family: .zImage, includesEncoder: true))!
        let enc = z.filter { $0.phase == .encoding }
        XCTAssertEqual(Set(enc.compactMap { c -> Int? in if case let .product(_, _, r, _) = c.shape { return r } else { return nil } }), [19])
        XCTAssertEqual(enc.reduce(0) { $0 + ($1.executions[.zImage] ?? 0) }, 35 * 7)
        XCTAssertEqual(executions(enc, .zImage, k: 2560, n: 9728, rows: 19), 2 * 35)
        let moss = CC.classes(for: .init(family: .mossImage, includesEncoder: true))!.filter { $0.phase == .encoding }
        XCTAssertEqual(Set(moss.compactMap { c -> Int? in if case let .product(_, _, r, _) = c.shape { return r } else { return nil } }), [21])
        XCTAssertEqual(executions(CC.classes(for: .init(family: .qwenImage21))!, .qwenImage21, k: 4096, n: 12288, rows: 33), 2 * 36)
        XCTAssertEqual(executions(CC.classes(for: .init(family: .krea2))!, .krea2, k: 2560, n: 9728, rows: 50), 2 * 36)
        XCTAssertEqual(executions(CC.classes(for: .init(family: .anima))!, .anima, k: 1024, n: 3072, rows: 11), 2 * 28)
        XCTAssertTrue(classes(.zImage).allSatisfy { $0.phase == .denoising }, "an « image suivante » does not run the encoder")
    }

    func testVersionsAndTheirFormats() {
        XCTAssertEqual(CC.weightFormat(.zImage, .standard), .bf16)
        XCTAssertEqual(CC.weightFormat(.zImage, .compact), .q8)
        XCTAssertEqual(CC.weightFormat(.anima, .compact), .q8)
        XCTAssertEqual(CC.weightFormat(.krea2, .compact), .int8)
        XCTAssertEqual(CC.weightFormat(.qwenImage21, .light), .q4k)
        XCTAssertNil(CC.weightFormat(.mossImage, .light))
        XCTAssertNil(CC.classes(for: .init(family: .mossImage, variant: .light)))
        XCTAssertNil(CC.WeightFormat.q4k.kind)
        XCTAssertEqual(CC.encoderFormat(.anima, .compact), .bf16)
        XCTAssertEqual(CC.encoderFormat(.zImage, .compact), .int8)
        // The format is in the shape: the same k×n of two versions are two classes.
        let std = classes(.zImage), cmp = classes(.zImage, .compact)
        XCTAssertEqual(std.count, cmp.count)
        XCTAssertTrue(Set(std.map(\.shape)).isDisjoint(with: Set(cmp.filter { if case .product = $0.shape { return true } else { return false } }.map(\.shape))))
        XCTAssertEqual(cmp.compactMap(\.weightKind).first, .q8)
    }

    // MARK: fusion and weights

    func testIdenticalClassesMergeAcrossFamiliesAndKeepTheirCountsApart() {
        // Z-Image's encoder at 50 rows (prompt 42) and Krea 2's (prompt 11): the same Qwen3 text layers.
        let z = CC.classes(for: .init(family: .zImage, promptTokens: 42))!
        let k = CC.classes(for: .init(family: .krea2, promptTokens: 11))!
        let merged = CC.merge([z, k])
        XCTAssertLessThan(merged.count, z.count + k.count)
        let both = merged.first { c in
            if case .product(2560, 9728, 50, .bf16) = c.shape, c.phase == .encoding { return true } else { return false }
        }
        XCTAssertEqual(both?.executions, [.zImage: 2 * 35, .krea2: 2 * 36])
        // Merging changes no family's list of executions.
        for (family, own) in [(Family.zImage, z), (.krea2, k)] {
            for c in own {
                XCTAssertEqual(merged.first { $0.phase == c.phase && $0.shape == c.shape }?.executions[family], c.executions[family])
            }
        }
        // Merging a family with itself adds its executions.
        let twice = CC.merge([z, z])
        XCTAssertEqual(twice.count, z.count)
        XCTAssertEqual(twice.map { $0.executions[.zImage]! }.reduce(0, +), 2 * z.map { $0.executions[.zImage]! }.reduce(0, +))
        // Deterministic order.
        XCTAssertEqual(CC.merge([k, z]).map(\.shape), merged.map(\.shape))
    }

    func testWeightsSumToOneAndFollowTheFLOPWhenRatesAreNeutral() {
        let all = Family.allCases.map { CC.classes(for: .init(family: $0))! }
        let merged = CC.merge(all)
        for family in Family.allCases {
            let w = CC.weights(merged, family: family)
            XCTAssertEqual(w.reduce(0, +), 1, accuracy: 1e-12, family.name)
            XCTAssertTrue(w.allSatisfy { $0 >= 0 })
            for (c, x) in zip(merged, w) where c.executions[family] == nil { XCTAssertEqual(x, 0) }
        }
        let z = classes(.zImage)
        let w = CC.weights(z, family: .zImage)
        let total = z.reduce(0) { $0 + $1.totalFLOP(.zImage) }
        for (c, x) in zip(z, w) { XCTAssertEqual(x, c.totalFLOP(.zImage) / total, accuracy: 1e-12) }
        // A family's weights do not move when other families join (the merge only adds classes the family does not run).
        let zWhole = CC.classes(for: .init(family: .zImage))!
        let alone = CC.weights(zWhole, family: .zImage)
        let joined = CC.weights(merged, family: .zImage)
        for (c, x) in zip(zWhole, alone) {
            let i = merged.firstIndex { $0.phase == c.phase && $0.shape == c.shape }!
            XCTAssertEqual(joined[i], x, accuracy: 1e-12)
        }
        // Per class and family.
        let per = CC.weights(merged)
        XCTAssertEqual(per.count, merged.count)
        XCTAssertEqual(per.compactMap { $0[.krea2] }.reduce(0, +), 1, accuracy: 1e-12)
        // The product shares of a 1024² Z-Image: attention a bit over a tenth of the FLOP.
        let attention = zip(z, w).filter { if case .attention = $0.0.shape { return true } else { return false } }.reduce(0) { $0 + $1.1 }
        XCTAssertGreaterThan(attention, 0.10)
        XCTAssertLessThan(attention, 0.20)
    }

    func testRatesMoveTheWeights() {
        let z = classes(.zImage)
        let slowAttention = CC.Rates(attention: ["graph": 2e11])         // 5× slower than the products' neutral 1e12
        let a = CC.weights(z, family: .zImage)
        let b = CC.weights(z, family: .zImage, rates: slowAttention)
        func share(_ w: [Double]) -> Double { zip(z, w).filter { if case .attention = $0.0.shape { return true } else { return false } }.reduce(0) { $0 + $1.1 } }
        XCTAssertGreaterThan(share(b), share(a))
        XCTAssertEqual(b.reduce(0, +), 1, accuracy: 1e-12)
        // A measured rate for one class wins over the default.
        let key = z[0].routeKey.id
        let r = CC.Rates(measured: [key: ["gpu": 5e11]])
        XCTAssertEqual(r.rate(z[0], route: "gpu"), 5e11)
        XCTAssertEqual(r.rate(z[1], route: "gpu"), 1e12)
    }

    // MARK: the prediction

    private func amxTable() -> RouteTable {
        RouteTable(products: .init(amx: .init(enabled: true)))
    }

    func testPredictionIsAdditiveOverClasses() {
        let cs = classes(.zImage, encoder: false)
        let table = amxTable()
        let rates = CC.Rates(product: ["gpu": 3e12, "gpu+amx": 4.6e12], attention: ["graph": 2e12])
        let whole = CC.predict(cs, family: .zImage, table: table, rates: rates)
        let (left, right) = (Array(cs[..<5]), Array(cs[5...]))
        let sum = CC.predict(left, family: .zImage, table: table, rates: rates).seconds + CC.predict(right, family: .zImage, table: table, rates: rates).seconds
        XCTAssertEqual(whole.seconds, sum, accuracy: 1e-9 * whole.seconds)
        XCTAssertEqual(whole.seconds, whole.parts.reduce(0) { $0 + $1.seconds }, accuracy: 1e-9)
        XCTAssertEqual(whole.parts.count, cs.count)
        // By hand for one class: executions · FLOP / rate.
        let c = cs[0]
        let id = CC.route(of: c, family: .zImage, table: table)
        XCTAssertEqual(id, "gpu+amx")
        XCTAssertEqual(CC.predict([c], family: .zImage, table: table, rates: rates).seconds, c.totalFLOP(.zImage) / 4.6e12, accuracy: 1e-9)
        // The AMX takes nothing under 512 rows; the encoder's products never reach the ANE.
        let enc = CC.classes(for: .init(family: .zImage))!.first { $0.phase == .encoding }!
        XCTAssertEqual(CC.route(of: enc, family: .zImage, table: table), "gpu")
        var withANE = table
        withANE.products.ane = .init(enabled: true, minimumRows: 1)
        XCTAssertEqual(CC.route(of: enc, family: .zImage, table: withANE), "gpu")
        // The default phase is the denoising.
        let full = CC.classes(for: .init(family: .zImage))!
        XCTAssertEqual(CC.predict(full, family: .zImage, table: table).seconds, CC.predict(cs, family: .zImage, table: table).seconds, accuracy: 1e-9)
        XCTAssertGreaterThan(CC.predict(full, family: .zImage, table: table, phases: [.denoising, .encoding]).seconds,
                             CC.predict(full, family: .zImage, table: table).seconds)
        // A family that does not run a class pays nothing for it.
        XCTAssertEqual(CC.predict(CC.merge([cs, classes(.krea2)]), family: .zImage, table: table, rates: rates).seconds, whole.seconds, accuracy: 1e-9 * whole.seconds)
    }

    func testPredictionIsMonotoneInRatesAndHeat() {
        let cs = classes(.qwenImage21)
        let table = amxTable()
        var rates = CC.Rates(product: ["gpu": 3e12, "gpu+amx": 4e12], attention: ["graph": 1.5e12, "split": 2e12])
        var last = CC.predict(cs, family: .qwenImage21, table: table, rates: rates).seconds
        for factor in [1.1, 1.5, 2.0] {
            rates.product["gpu+amx"]! *= factor
            rates.attention["graph"]! *= factor
            let t = CC.predict(cs, family: .qwenImage21, table: table, rates: rates).seconds
            XCTAssertLessThan(t, last)
            last = t
        }
        // Heat: an engine that takes twice as long makes every route that keeps it busy take twice as long; no other.
        let cold = CC.predict(cs, family: .qwenImage21, table: table, rates: rates)
        let hot = CC.predict(cs, family: .qwenImage21, table: table, rates: rates, heat: .init(["amx": 2]))
        XCTAssertGreaterThan(hot.seconds, cold.seconds)
        for (c, h) in zip(cold.parts, hot.parts) {
            XCTAssertEqual(h.seconds, c.seconds * (c.route.contains("amx") ? 2 : 1), accuracy: 1e-9 * c.seconds)
        }
        // A split attention is held back by the slower of the GPU and the CPU.
        var splitTable = table
        splitTable.attention = .init(rules: [.init(splitFraction: 0.4)])
        let split = CC.predict(cs, family: .qwenImage21, table: splitTable, rates: rates, heat: .init(["cpu": 3]))
        XCTAssertTrue(split.parts.contains { $0.route == "split" })
        XCTAssertGreaterThan(split.seconds, CC.predict(cs, family: .qwenImage21, table: splitTable, rates: rates).seconds)
        // Z-Image and Anima try flash before the split.
        let flashTable = RouteTable(attention: .init(rules: [.init(flash: true, splitFraction: 0.4)]))
        let za = classes(.zImage).first { if case .attention = $0.shape { return true } else { return false } }!
        XCTAssertEqual(CC.route(of: za, family: .zImage, table: flashTable), "flash")
        XCTAssertEqual(CC.route(of: za, family: .krea2, table: flashTable), "split")
    }

    // MARK: the plan, at heights nobody measured

    private func product(_ k: Int, _ n: Int, _ m: Int) -> RouteTable.ClassKey { .product(k: k, n: n, m: m) }

    private func decision(_ key: RouteTable.ClassKey, _ chosen: String?, _ ratios: [Double?], status: CM.Status = .closed) -> CM.Decision {
        CM.Decision(classID: key.id, chosen: chosen, status: status, loss: 0, hotRatios: ratios, runs: 0)
    }

    private func plan(_ decisions: [CM.Decision]) -> CM.Plan { CM.Plan(decisions: decisions, residualLoss: 0, spent: 1, exhausted: false) }

    private func computeClass(_ key: RouteTable.ClassKey, _ routes: [CM.Route]) -> CM.ComputeClass { CM.ComputeClass(key.id, weight: 1, routes: routes) }

    private func render(_ keys: [RouteTable.ClassKey]) -> [CC.Class] {
        keys.map { key in
            switch key {
            case let .product(k, n, m): return CC.Class(phase: .denoising, shape: .product(k: k, n: n, rows: m, format: .bf16), executions: [.zImage: 1])
            case let .attention(s, h, d): return CC.Class(phase: .denoising, shape: .attention(sequence: s, keys: s, heads: h, kvHeads: h, headDim: d, cross: false), executions: [.zImage: 1])
            }
        }
    }

    func testAnEfficientRouteIsTakenBetweenTheMeasuredHeightsWhereItWins() {
        // The ANE loses at 1 024 rows (×1.3) and wins at 4 096 (×0.7); the line crosses 1 in between.
        let routes = [CM.Route("gpu"), CM.Route("gpu+ane", efficient: true, engines: ["gpu", "ane"])]
        let lo = product(3840, 3840, 1024), hi = product(3840, 3840, 4096)
        let compute = [computeClass(lo, routes), computeClass(hi, routes)]
        let p = plan([decision(lo, "gpu", [1, 1.3]), decision(hi, "gpu+ane", [1, 0.7])])
        let targets = [1056, 1568, 2048, 4128, 6176].map { product(3840, 3840, $0) }
        let table = RouteTable(plan: p, classes: compute, renderClasses: render(targets))
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 1024), .gpu)
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 4096), .ane)
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 1056), .gpu)        // ρ ≈ 1.29
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 1568), .gpu)        // ρ ≈ 1.075
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 2048), .ane)        // ρ ≈ 0.954
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 4128), .ane)
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 6176), .ane)        // the line goes on: ρ ≈ 0.58
        XCTAssertTrue(table.products.usesANE)
        // Not asked for: no rule. Another shape, with no anchor: nothing invented.
        XCTAssertNil(table.products.forced(k: 3840, n: 3840, m: 3000))
        let other = RouteTable(plan: p, classes: compute, renderClasses: render([product(4096, 4096, 2048)]))
        XCTAssertNil(other.products.forced(k: 4096, n: 4096, m: 2048))
        // Without render classes the table is the plan's alone, as before.
        XCTAssertNil(RouteTable(plan: p, classes: compute).products.forced(k: 3840, n: 3840, m: 2048))
    }

    func testTheLineGoesPastTheLastHeightInsteadOfStoppingThere() {
        // The AMX wins at both measured heights (×0.8, ×0.95) by a shrinking margin: its line reaches 1 at ~6 144 rows,
        // where a clamp at 4 096 would still send it to the AMX.
        let routes = [CM.Route("gpu"), CM.Route("gpu+amx", engines: ["gpu", "amx"])]
        let lo = product(3840, 10240, 1024), hi = product(3840, 10240, 4096)
        let compute = [computeClass(lo, routes), computeClass(hi, routes)]
        let p = plan([decision(lo, "gpu+amx", [1, 0.8]), decision(hi, "gpu+amx", [1, 0.95])])
        let table = RouteTable(plan: p, classes: compute, renderClasses: render([2048, 3000, 6144].map { product(3840, 10240, $0) }))
        XCTAssertEqual(table.products.forced(k: 3840, n: 10240, m: 2048), .amx)
        XCTAssertEqual(table.products.forced(k: 3840, n: 10240, m: 3000), .amx)
        XCTAssertEqual(table.products.forced(k: 3840, n: 10240, m: 6144), .gpu)
        // Below the first height too (the line runs backwards): the margin grows.
        let low = RouteTable(plan: p, classes: compute, renderClasses: render([product(3840, 10240, 512)]))
        XCTAssertEqual(low.products.forced(k: 3840, n: 10240, m: 512), .amx)
    }

    func testOneMeasuredHeightStandsAlone() {
        let routes = [CM.Route("gpu"), CM.Route("gpu+amx", engines: ["gpu", "amx"])]
        let only = product(2048, 2048, 4096)
        let table = RouteTable(plan: plan([decision(only, "gpu+amx", [1, 0.8])]), classes: [computeClass(only, routes)],
                               renderClasses: render([product(2048, 2048, 6144)]))
        XCTAssertEqual(table.products.forced(k: 2048, n: 2048, m: 6144), .amx)
    }

    func testAHoleAtOneAnchorIsNotACandidateBetween() {
        let routes = [CM.Route("gpu"), CM.Route("gpu+amx", engines: ["gpu", "amx"]), CM.Route("gpu+ane", efficient: true, engines: ["gpu", "ane"])]
        let lo = product(3840, 3840, 1024), hi = product(3840, 3840, 4096)
        let p = plan([decision(lo, "gpu+amx", [1, 0.9, nil]), decision(hi, "gpu+ane", [1, 0.9, 0.5])])
        let table = RouteTable(plan: p, classes: [computeClass(lo, routes), computeClass(hi, routes)],
                               renderClasses: render([product(3840, 3840, 2048)]))
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 2048), .amx)
    }

    func testAnExactDecisionIsNeverOverwritten() {
        let routes = [CM.Route("gpu"), CM.Route("gpu+amx", engines: ["gpu", "amx"])]
        let a = product(3840, 3840, 1024), b = product(3840, 3840, 2048), c = product(3840, 3840, 4096)
        let p = plan([decision(a, "gpu+amx", [1, 0.5]), decision(b, "gpu", [1, 1.2]), decision(c, "gpu+amx", [1, 0.5])])
        let table = RouteTable(plan: p, classes: [a, b, c].map { computeClass($0, routes) }, renderClasses: render([b]))
        XCTAssertEqual(table.products.classes.filter { $0.minRows == 2048 }.count, 1)
        XCTAssertEqual(table.products.forced(k: 3840, n: 3840, m: 2048), .gpu)
    }

    func testAnAttentionAtAnUnmeasuredSequence() {
        let routes = [CM.Route("graph"), CM.Route("flash"), CM.Route("split")]
        let lo = RouteTable.ClassKey.attention(sequence: 1024, heads: 30, headDim: 128)
        let hi = RouteTable.ClassKey.attention(sequence: 4096, heads: 30, headDim: 128)
        let p = plan([decision(lo, "flash", [1, 0.9, 1.2]), decision(hi, "split", [1, 1.1, 0.6])])
        let compute = [computeClass(lo, routes), computeClass(hi, routes)]
        let mid = RouteTable.ClassKey.attention(sequence: 2048, heads: 30, headDim: 128)
        let table = RouteTable(plan: p, classes: compute, renderClasses: render([mid, .attention(sequence: 2048, heads: 32, headDim: 128)]))
        XCTAssertTrue(table.attention.route(sequence: 2048, heads: 30, headDim: 128).splits)
        XCTAssertFalse(table.attention.route(sequence: 2048, heads: 32, headDim: 128).splits, "another head count has no anchor")
        XCTAssertFalse(table.attention.route(sequence: 2048, heads: 32, headDim: 128).flash)
        XCTAssertTrue(table.attention.route(sequence: 4096, heads: 30, headDim: 128).splits)
        XCTAssertTrue(table.attention.route(sequence: 1024, heads: 30, headDim: 128).flash)
    }

    func testTheRuleIsTheMapsOwn() {
        // CapacityMeasure.rule on ratios = what `plan` chose from the same figures.
        let routes = [CM.Route("gpu"), CM.Route("gpu+amx", engines: ["gpu", "amx"]), CM.Route("gpu+ane", efficient: true, engines: ["gpu", "ane"])]
        XCTAssertEqual(CM.rule(ratios: [1, 0.8, 0.85], efficient: routes.map(\.efficient)), 2, "the ANE within 90 % of the best")
        XCTAssertEqual(CM.rule(ratios: [1, 0.8, 0.95], efficient: routes.map(\.efficient)), 1)
        XCTAssertEqual(CM.rule(ratios: [1, 0.995, nil], efficient: routes.map(\.efficient)), 0, "a tie goes to the simplest")
        XCTAssertNil(CM.rule(ratios: [nil, nil], efficient: [false, false]))
    }

    // MARK: one form per prepared weight

    private func int8Classes(rowsAndCounts: [(Int, Int)]) -> [CC.Class] {
        rowsAndCounts.map { CC.Class(phase: .denoising, shape: .product(k: 3840, n: 10240, rows: $0.0, format: .int8), executions: [.zImage: $0.1]) }
    }

    func testAWeightKeepsOneFormAtEveryHeight() {
        // A rule that keeps the 8-bit weight as published only from 2 048 rows: widened at 1 056, native at 4 128.
        var base = RouteTable()
        base.products.weights = [.init(kind: .int8, path: .native, minRows: 2048)]
        let cs = int8Classes(rowsAndCounts: [(1056, 3), (4128, 5)])
        let conflicts = base.weightPathConflicts(for: cs)
        XCTAssertEqual(conflicts.count, 1)
        XCTAssertEqual(conflicts[0].paths, [1056: .widened, 4128: .native])

        let empty = CM.Plan(decisions: [], residualLoss: 0, spent: 0, exhausted: false)
        let table = RouteTable(plan: empty, classes: [], base: base, renderClasses: cs)
        XCTAssertTrue(table.weightPathConflicts(for: cs).isEmpty)
        for m in [1056, 4128] { XCTAssertEqual(table.products.weight(.int8, k: 3840, n: 10240, m: m).path, .native, "the heavy height's form") }
        // Other shapes of the same kind are untouched.
        XCTAssertEqual(table.products.weight(.int8, k: 4096, n: 4096, m: 1056).path, .widened)

        // Heavy at the small height: the other way round.
        let light = int8Classes(rowsAndCounts: [(1056, 400), (4128, 1)])
        let back = RouteTable(plan: empty, classes: [], base: base, renderClasses: light)
        for m in [1056, 4128] { XCTAssertEqual(back.products.weight(.int8, k: 3840, n: 10240, m: m).path, .widened) }
        XCTAssertTrue(back.weightPathConflicts(for: light).isEmpty)
    }

    func testNoConflictLeavesTheWeightRulesAlone() {
        let rules = RouteTable.Products.weightRules(bf16: true, int8: true, q8: true, rounded: true, minRows: 256, eightBit: false)
        var base = RouteTable()
        base.products.weights = rules
        // bf16 weights rounded their activation from 256 rows, widened never: the path is `bf16` at every height.
        let cs = Family.allCases.flatMap { classes($0) }
        XCTAssertTrue(base.weightPathConflicts(for: cs).isEmpty)
        let empty = CM.Plan(decisions: [], residualLoss: 0, spent: 0, exhausted: false)
        XCTAssertEqual(RouteTable(plan: empty, classes: [], base: base, renderClasses: cs).products.weights, rules)
        // The rounding stays where the table put it when its path is the pinned one.
        var mixed = base
        mixed.products.weights = [.init(kind: .bf16, path: .bf16, roundsActivation: true, minRows: 256), .init(kind: .bf16, path: .widened, minRows: 0, maxRows: 100)]
        let small = [CC.Class(phase: .denoising, shape: .product(k: 64, n: 3840, rows: 32, format: .bf16), executions: [.zImage: 1]),
                     CC.Class(phase: .denoising, shape: .product(k: 64, n: 3840, rows: 4096, format: .bf16), executions: [.zImage: 100])]
        XCTAssertEqual(mixed.weightPathConflicts(for: small).count, 1)
        var fixed = mixed
        fixed.unifyWeightPaths(for: small)
        XCTAssertEqual(fixed.products.weight(.bf16, k: 64, n: 3840, m: 32).path, .bf16)
        XCTAssertFalse(fixed.products.weight(.bf16, k: 64, n: 3840, m: 32).roundsActivation, "32 rows never rounded")
        XCTAssertTrue(fixed.products.weight(.bf16, k: 64, n: 3840, m: 4096).roundsActivation)
    }

    func testAPlanThatInterpolatesAndUnifiesInOneInit() {
        let routes = [CM.Route("gpu"), CM.Route("gpu+ane", efficient: true, engines: ["gpu", "ane"])]
        let lo = product(3840, 10240, 1024), hi = product(3840, 10240, 4096)
        let p = plan([decision(lo, "gpu", [1, 1.3]), decision(hi, "gpu+ane", [1, 0.7])])
        var base = RouteTable()
        base.products.weights = [.init(kind: .int8, path: .native, minRows: 2048)]
        let cs = int8Classes(rowsAndCounts: [(1056, 3), (2048, 1), (4128, 5)])
        let table = RouteTable(plan: p, classes: [computeClass(lo, routes), computeClass(hi, routes)], base: base, renderClasses: cs)
        XCTAssertEqual(table.products.forced(k: 3840, n: 10240, m: 2048), .ane)
        XCTAssertEqual(table.products.forced(k: 3840, n: 10240, m: 1056), .gpu)
        XCTAssertTrue(table.weightPathConflicts(for: cs).isEmpty)
        let data = try? JSONEncoder().encode(table)
        XCTAssertEqual(data.flatMap { try? JSONDecoder().decode(RouteTable.self, from: $0) }, table)
    }
}
