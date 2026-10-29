import Foundation
import XCTest
@testable import Siliconed

/// **The diagnostic without a GPU and without a byte of weights**: the JSON a report becomes (stable,
/// schema 6, measures as triples, explicit `null`s, read back whole), the issue URL and its length at
/// the full size of a report, the shapes it measures, and its CPU references against naive loops.
final class DiagnosticTests: XCTestCase {
    /// A report of the full size — every product, attention, widening and half-precision item a run
    /// writes — with figures of the M1 Pro's order, and one engine missing of each kind.
    private func sample() -> Diagnostic.Report {
        let machine = Diagnostic.Machine(chip: "Apple M1 Pro", hardwareModel: "MacBookPro18,3", performanceCores: 6,
                                         efficiencyCores: 2, gpuCores: 14, memoryBytes: 17_179_869_184,
                                         macOS: "27.0.1 (26A434)", version: "git 0123456789-dirty", build: "x+y",
                                         gpu: "Apple M1 Pro", gpuFamily: "apple7", gpuWorkingSetBytes: 12_713_115_648)
        func measure(_ rate: Double, _ error: Double, threshold: Double = 1e-5) -> Diagnostic.Measure {
            .init(rate: rate * 1.000123456, error: error * 1.0123456, threshold: threshold)
        }
        let products = Diagnostic.shapes.enumerated().map { i, s in
            Diagnostic.Product(model: s.model, product: s.product, m: s.m, k: s.k, n: s.n, gpu: measure(3.461, 1.1e-6),
                               amx: i == 7 ? nil : measure(2.223, 8.0e-7), gpuShare: i == 7 ? nil : 0.6091234,
                               coexecution: i == 7 ? nil : measure(5.497, 1.1e-6), coexecutionGPURows: i == 7 ? nil : 2560,
                               tensor: i == 6 ? nil : measure(2.362, 1.1e-6), int8: measure(2.871, 1.6e-6),
                               int8Coexecution: i == 7 ? nil : measure(4.912, 1.4e-6), int8GPUColumns: i == 7 ? nil : 6656,
                               q8: measure(1.234, 1.5e-6), q8Coexecution: i == 7 ? nil : measure(2.871, 1.3e-6),
                               q8GPUColumns: i == 7 ? nil : 6656)
        }
        let lowPrecision = ["fp16", "bf16"].flatMap { t in
            [Diagnostic.LowPrecision(engine: "mpsgraph", type: t, output: t, m: 1024, k: 3840, n: 10240,
                                     measure: measure(3.425, 2.1e-4, threshold: 2.654e-4), floor: 2.1234e-4, threshold: 2.654e-4),
             Diagnostic.LowPrecision(engine: "bnns", type: t, output: "fp32", m: 1024, k: 3840, n: 10240,
                                     measure: measure(2.487, 8.0e-7), floor: 2.57e-8, threshold: 1e-5),
             Diagnostic.LowPrecision(engine: "matmul2d", type: t, output: "fp32", m: 1024, k: 3840, n: 10240,
                                     measure: t == "fp16" ? measure(1.234, 9e-7) : nil, floor: 2.57e-8, threshold: 1e-5)]
        }
        let attention = Diagnostic.blocks.flatMap { b in
            Diagnostic.heights.map { s in
                Diagnostic.AttentionCapacity(model: b.model, heads: b.heads, headDim: b.headDim, sequence: s,
                                             graph: measure(1.637, 4.4e-6), flash: s == 1024 ? nil : measure(1.686, 2.8e-6),
                                             headSplit: measure(2.497, 4.2e-6))
            }
        }
        let widening = [Diagnostic.Widening(kind: "bf16", values: 39_321_600, cpu: measure(21.26, 0)),
                        Diagnostic.Widening(kind: "bf16 copy", values: 39_321_600, cpu: measure(41.37, 0)),
                        Diagnostic.Widening(kind: "int8", values: 39_321_600, cpu: measure(23.07, 0)),
                        Diagnostic.Widening(kind: "int8 convrot", values: 39_321_600, cpu: measure(6.12, 0)),
                        Diagnostic.Widening(kind: "int8 copy", values: 39_321_600, cpu: measure(48.3, 0)),
                        Diagnostic.Widening(kind: "q8_0", values: 39_321_600, cpu: measure(22.91, 0)),
                        Diagnostic.Widening(kind: "q8_0 copy", values: 39_321_600, cpu: measure(69.4, 0))]
        let stages = Dictionary(uniqueKeysWithValues: Diagnostic.Stage.allCases.enumerated().map { ($1.key, 1.234567 * Double($0 + 1)) })
        let conditions = Diagnostic.Conditions(thermal: "nominal", lowPowerMode: false, onBattery: nil, swapUsedBytes: 0)
        var features: [String: Int] = ["sme_max_svl_b": 0]
        for name in Diagnostic.featureNames { features[name] = name.contains("FP16") || name.contains("DotProd") ? 1 : 0 }
        return Diagnostic.Report(date: "2026-10-14T10:00:00Z", seconds: 14.04, machine: machine, cpuFeatures: features,
                                 settings: ["flash": "true (env)"], before: conditions, after: conditions,
                                 products: products, lowPrecision: lowPrecision, attention: attention, widening: widening,
                                 witness: .init(gpuBefore: 3.461, gpuAfter: 3.4721, amxBefore: 2.223, amxAfter: 2.149),
                                 stages: stages,
                                 cuts: Diagnostic.cutDepths.flatMap { k in Diagnostic.heights.map { m in
                                     Diagnostic.Cut(k: k, n: Diagnostic.cutWidth, m: m, share: k < 8192 ? 0.6234 : 0.6876) } })
    }

    // ── off rest ────────────────────────────────────────────────────────────────────────────

    /// A map's conditions are facts of its regime, never a reason for it to be orange (`Report.isGreen`, `CapacityMapTests`):
    /// `isNominal` only says what the machine was doing.
    func testConditionsAreFactsOfTheRegime() {
        func conditions(_ thermal: String = "nominal", lowPower: Bool = false, battery: Bool? = nil, swap: Int = 0) -> Diagnostic.Conditions {
            Diagnostic.Conditions(thermal: thermal, lowPowerMode: lowPower, onBattery: battery, swapUsedBytes: swap)
        }
        var warm = sample(); warm.before = conditions(); warm.after = conditions("serious", lowPower: true, swap: 7_000_000_000)
        XCTAssertFalse(warm.isGreen, "a legacy report holds no verdict")
        XCTAssertTrue(conditions(battery: true).isNominal)
        XCTAssertFalse(conditions("fair").isNominal)
        XCTAssertFalse(conditions(lowPower: true).isNominal)
    }

    // ── the JSON ────────────────────────────────────────────────────────────────────────────

    func testJSONIsStableAndReadsBack() throws {
        let report = sample()
        let a = try report.json(), b = try report.json()
        XCTAssertEqual(a, b)
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Report.self, from: a), report)
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Report.self, from: try report.json(pretty: false)), report)
        let text = String(decoding: a, as: UTF8.self)
        // Sorted keys: `attention` before `machine` before `products` before `schema` before `widening`.
        let order = ["\"attention\"", "\"machine\"", "\"products\"", "\"schema\"", "\"widening\""].map { text.range(of: $0)!.lowerBound }
        XCTAssertEqual(order, order.sorted())
        XCTAssertTrue(text.contains("\"schema\" : 14"))
        XCTAssertEqual(Diagnostic.schema, 14)
        XCTAssertEqual(String(decoding: try JSONEncoder().encode(Diagnostic.Cut(k: 16384, n: 6144, m: 4096, share: 0.68764)), as: UTF8.self),
                       "[16384,6144,4096,0.688]", "a cut is a quadruple, its share rounded")
    }

    /// **Schema 6 is the capacity map**: nothing of the model-rendering diagnostic survives in it.
    func testSchemaSixHoldsNoModelRun() throws {
        let text = String(decoding: try sample().json(), as: UTF8.self)
        for gone in ["\"models\"", "golden", "deviation", "estimatedRenderSeconds", "peakFootprintBytes", "swapouts",
                     "diskReadBytes", "needBytes", "\"prompt\"", "\"seed\"", "\"resolution\"", "witnessBefore"] {
            XCTAssertFalse(text.contains(gone), gone)
        }
        for key in ["products", "lowPrecision", "attention", "widening", "witness", "thresholds", "cpuFeatures",
                    "gpuShare", "coexecution", "headSplit", "floor", "tensor", "stages", "bf16 copy", "cuts"] {
            XCTAssertTrue(text.contains("\"\(key)\""), key)
        }
    }

    /// A measure is `[rate, error, pass]`, its figures rounded: the issue link's bytes are counted.
    func testMeasureIsATripleOfRoundedFigures() throws {
        let m = Diagnostic.Measure(rate: 3.461234567, error: 1.123456e-6, threshold: 1e-5)
        XCTAssertEqual(m.rate, 3.461)
        XCTAssertEqual(m.error, 1.12e-6)
        XCTAssertTrue(m.pass)
        let text = String(decoding: try JSONEncoder().encode(m), as: UTF8.self)
        XCTAssertEqual(text, "[3.461,1.12e-06,true]")
        XCTAssertEqual(try JSONDecoder().decode(Diagnostic.Measure.self, from: Data(text.utf8)), m)
        // Judged before the rounding: 1.0004e-5 rounds to 1e-5 and still fails.
        XCTAssertFalse(Diagnostic.Measure(rate: 1, error: 1.0004e-5, threshold: 1e-5).pass)
        XCTAssertTrue(Diagnostic.Measure(rate: 1, error: 0, threshold: 0).pass)
    }

    /// A NaN output is an infinite error: it fails, and JSON — which has no infinity — still writes it.
    func testNonFiniteErrorFailsAndEncodes() throws {
        for bad in [Double.nan, .infinity] {
            let m = Diagnostic.Measure(rate: .infinity, error: bad, threshold: 1)
            XCTAssertFalse(m.pass)
            XCTAssertEqual(m.error, .greatestFiniteMagnitude)
            XCTAssertEqual(m.rate, .greatestFiniteMagnitude)
            XCTAssertNoThrow(try JSONEncoder().encode(m))
        }
        XCTAssertEqual(Diagnostic.significant(0, 3), 0)
        XCTAssertEqual(Diagnostic.significant(-2.71828, 2), -2.7)
    }

    /// An engine that did not run says `null`, it does not drop the field.
    func testMissingEnginesAreExplicitNulls() throws {
        let compact = String(decoding: try sample().json(pretty: false), as: UTF8.self)
        XCTAssertFalse(compact.contains("\n"))
        for null in ["\"amx\":null", "\"gpuShare\":null", "\"coexecution\":null", "\"coexecutionGPURows\":null",
                     "\"flash\":null", "\"measure\":null", "\"onBattery\":null"] {
            XCTAssertTrue(compact.contains(null), null)
        }
        let report = sample()
        XCTAssertTrue(report.passes)
        var failing = report
        failing.widening[0].cpu = .init(rate: 19, error: 1e-7, threshold: 0)
        XCTAssertFalse(failing.passes)
    }

    /// A widening written by an earlier build still holds the GPU kernel's measure: it reads, the measure
    /// dropped — a map kept by the app is not refused for a field nothing decides by any more.
    func testAnEarlierWideningWithItsGPUMeasureStillReads() throws {
        let json = #"{"kind":"bf16","values":39321600,"cpu":[21.26,0,true],"gpu":[19.48,0,true]}"#
        let read = try JSONDecoder().decode(Diagnostic.Widening.self, from: Data(json.utf8))
        XCTAssertEqual(read, Diagnostic.Widening(kind: "bf16", values: 39_321_600,
                                                 cpu: .init(rate: 21.26, error: 0, threshold: 0)))
        XCTAssertFalse(String(decoding: try JSONEncoder().encode(read), as: UTF8.self).contains("gpu"))
    }

    /// The machine's state and flags read without a GPU.
    func testConditionsAndFeaturesRead() {
        let c = Diagnostic.conditions()
        XCTAssertTrue(["nominal", "fair", "serious", "critical"].contains(c.thermal))
        XCTAssertGreaterThanOrEqual(c.swapUsedBytes, 0)
        let flags = Diagnostic.cpuFeatures()
        for (name, value) in flags where name.hasPrefix("FEAT_") { XCTAssertTrue([0, 1].contains(value), name) }
        XCTAssertTrue(Set(flags.keys).isSubset(of: Set(Diagnostic.featureNames + ["sme_max_svl_b"])))
    }

    // ── the issue URL ───────────────────────────────────────────────────────────────────────

    func testIssueURLCarriesTheFieldsAndTheWholeJSON() throws {
        let report = sample()
        let url = try report.issueURL()
        XCTAssertTrue(url.absoluteString.hasPrefix("https://github.com/gwenn-ha-dev/Siliconed/issues/new?template=diagnostic.yml&"))
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems!
        func field(_ name: String) -> String? { items.first { $0.name == name }?.value }
        XCTAssertEqual(field("chip"), "Apple M1 Pro (14-core GPU)")
        XCTAssertEqual(field("memory"), "16 GB")
        XCTAssertEqual(field("macos"), "27.0.1 (26A434)")
        XCTAssertEqual(field("version"), "git 0123456789-dirty")
        XCTAssertEqual(field("labels"), "diagnostic")
        let json = try XCTUnwrap(field("report"))
        // The measures whole; the decisions left out (the rules give them back from the measures).
        var measures = report
        measures.decisions = nil
        let read = try JSONDecoder().decode(Diagnostic.Report.self, from: Data(json.utf8))
        XCTAssertEqual(read, measures)
        XCTAssertEqual(Diagnostic.decide(read), report.decisions)
        // `+` is escaped: GitHub reads a bare one back as a space.
        XCTAssertFalse(url.absoluteString.contains("+"))
        XCTAssertTrue(url.absoluteString.contains("x%2By"))
    }

    /// **The link the app opens stays short, whatever the report holds**: browsers and GitHub cut beyond
    /// ~8 000 characters. A whole report outgrew it (schema 8: 7 938; schema 9's Q8_0 products: 8 903), so the
    /// app leaves `report` empty and the user pastes the JSON (`openIssue`); the command line only prints the
    /// whole link's length.
    func testTheAppsIssueLinkStaysShort() throws {
        var report = sample()
        report.machine.chip = "Apple M5 Max"
        report.settings = ["flash": "true (profile)", "amx": "true (profile)", "gemm_tensor": "true (env)"]
        let length = try report.issueURL(includingReport: false).absoluteString.utf8.count
        XCTAssertLessThanOrEqual(length, 1_000, "\(length) bytes")
    }

    /// What delimits a field is escaped, what does not is left as it is: a value holding `&`, `=`, `+`,
    /// `#`, `;` or `%` reads back whole, and the JSON's `,` and `:` cost one byte each, not three.
    func testIssueURLEscapesOnlyTheDelimiters() throws {
        let value = "a&b=c+d#e;f%g,h:i [j] {\"k\"}"
        let url = Diagnostic.issueURL(fields: [("report", value), ("chip", "x")])
        XCTAssertEqual(URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems!.first { $0.name == "report" }?.value, value)
        XCTAssertEqual(url.query?.filter { $0 == "&" }.count, 1)
        XCTAssertTrue(url.absoluteString.contains(",h:i"))
        XCTAssertFalse(url.absoluteString.contains("+"))
        XCTAssertFalse(url.absoluteString.contains(";"))
    }

    /// Too long for a link: the same form, every field but the report, which the user pastes.
    func testIssueURLWithoutTheReport() throws {
        let items = URLComponents(url: try sample().issueURL(includingReport: false), resolvingAgainstBaseURL: false)!.queryItems!
        XCTAssertNil(items.first { $0.name == "report" })
        XCTAssertEqual(items.first { $0.name == "chip" }?.value, "Apple M1 Pro (14-core GPU)")
        XCTAssertEqual(items.first { $0.name == "template" }?.value, "diagnostic.yml")
    }

    /// No account name in what the app shows or a report publishes.
    func testTheHomeFolderIsWrittenTilde() {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        XCTAssertEqual(Library.withoutHome("file missing: \(home)/Library/x.silicon"), "file missing: ~/Library/x.silicon")
        XCTAssertEqual(Library(root: URL(fileURLWithPath: home + "/Library/Application Support/Siliconed")).displayPath,
                       "~/Library/Application Support/Siliconed")
        XCTAssertEqual(Library.withoutHome("/opt/x"), "/opt/x")
    }

    // ── what it measures ────────────────────────────────────────────────────────────────────

    /// The blocks' shapes, from the models' defaults, at heights `GEMM` submits whole: no padded tail.
    func testShapesAreTheBlocksAtWholeHeights() {
        let shapes = Diagnostic.shapes
        XCTAssertEqual(shapes.count, 8)
        XCTAssertEqual(shapes[0], .init(model: "z-image", product: "up", m: 1024, k: 3840, n: 10240))
        XCTAssertTrue(shapes.contains(.init(model: "qwen-image-2.1", product: "down", m: 4096, k: 12288, n: 4096)))
        for s in shapes {
            XCTAssertEqual(GEMM.rowSplit(s.m).tail, 0, "\(s)")
            XCTAssertFalse(GEMM.goesThroughTiers(s.m))
            XCTAssertGreaterThanOrEqual(s.m, Conductor.minimumRows)
        }
        XCTAssertEqual(Diagnostic.blocks.map(\.heads), [30, 32])
        XCTAssertEqual(Diagnostic.blocks.map(\.headDim), [128, 128])
    }

    /// The sampled rows are distinct, in range, and visit every quantum of the cut.
    func testSampledRowsCoverTheCut() {
        for m in Diagnostic.heights {
            let rows = Diagnostic.sampleRows(m, count: Diagnostic.sampledRows)
            XCTAssertEqual(Set(rows).count, rows.count)
            XCTAssertTrue(rows.allSatisfy { (0..<m).contains($0) })
            XCTAssertEqual(Set(rows.map { $0 / Conductor.quantum }).count, m / Conductor.quantum)
        }
    }

    /// The fill is the same values whatever the length asked: a shorter operand is a prefix.
    func testFillIsAPrefixAndBounded() {
        let long = UnsafeMutablePointer<Float>.allocate(capacity: 200_000), short = UnsafeMutablePointer<Float>.allocate(capacity: 70_000)
        defer { long.deallocate(); short.deallocate() }
        Diagnostic.fill(long, count: 200_000, scale: 3, seed: 7)
        Diagnostic.fill(short, count: 70_000, scale: 3, seed: 7)
        XCTAssertTrue((0..<70_000).allSatisfy { long[$0] == short[$0] })
        XCTAssertTrue((0..<200_000).allSatisfy { abs(long[$0]) <= 3 })
        XCTAssertGreaterThan((0..<200_000).map { abs(long[$0]) }.max()!, 2.9)
    }

    /// `exactRows` (`cblas_dgemm` by slices of `B`) is the naive fp64 product, laid out `[n, rows]`.
    func testExactRowsAreTheNaiveProduct() {
        let (m, k, n) = (9, 700, 5)
        var a = [Float](repeating: 0, count: m * k), b = [Float](repeating: 0, count: k * n)
        Diagnostic.fill(&a, count: a.count, scale: 1, seed: 3)
        Diagnostic.fill(&b, count: b.count, scale: 1, seed: 4)
        let rows = [0, 4, 8]
        let exact = Diagnostic.exactRows(rows, a: a, b: b, k: k, n: n)
        for (i, row) in rows.enumerated() {
            for j in 0..<n {
                var s = 0.0
                for x in 0..<k { s += Double(a[row * k + x]) * Double(b[x * n + j]) }
                XCTAssertEqual(exact[j * rows.count + i], s, accuracy: 1e-12)
            }
        }
    }

    /// `exactAttention` is softmax(q·kᵀ/√Dh)·v, laid out `[Dh, rows·H]`; a row of equal keys averages `v`.
    func testExactAttentionIsTheReferenceFormula() {
        let (s, h, d) = (6, 2, 4)
        var q = [Float](repeating: 0, count: s * h * d), k = q, v = q
        Diagnostic.fill(&q, count: q.count, scale: 2, seed: 5)
        Diagnostic.fill(&k, count: k.count, scale: 2, seed: 6)
        Diagnostic.fill(&v, count: v.count, scale: 1, seed: 8)
        let rows = [1, 5]
        let exact = Diagnostic.exactAttention(q: q, k: k, v: v, sequence: s, heads: h, headDim: d, rows: rows)
        for (i, row) in rows.enumerated() {
            for head in 0..<h {
                var scores = (0..<s).map { j in
                    (0..<d).reduce(0.0) { $0 + Double(q[row * h * d + head * d + $1]) * Double(k[j * h * d + head * d + $1]) } / 2
                }
                let top = scores.max()!
                scores = scores.map { Foundation.exp($0 - top) }
                let total = scores.reduce(0, +)
                for x in 0..<d {
                    let o = (0..<s).reduce(0.0) { $0 + scores[$1] * Double(v[$1 * h * d + head * d + x]) } / total
                    XCTAssertEqual(exact[x * rows.count * h + i * h + head], o, accuracy: 1e-12)
                }
            }
        }
    }

    /// The fp64 `channelError` reads like the fp32 one, and sees a fault confined to one row.
    func testChannelErrorAgainstDoubles() {
        let expected: [Double] = (0..<600).map { Double($0 % 7) - 3.25 }
        let got = expected.map { Float($0) }
        let float = expected.map { Float($0) }
        let e64 = expected.withUnsafeBufferPointer { channelError(got, $0, columns: 3) }
        let e32 = float.withUnsafeBufferPointer { channelError(got, $0, columns: 3) }
        XCTAssertEqual(e64.worst, 0)
        XCTAssertEqual(e32.worst, 0)
        var moved = got
        for r in 0..<200 { moved[r * 3 + 1] *= 1.001 }
        let e = expected.withUnsafeBufferPointer { channelError(moved, $0, columns: 3) }
        XCTAssertEqual(e.channel, 1)
        XCTAssertEqual(e.worst, 1e-3, accuracy: 1e-6)
        var nan = got
        nan[4] = .nan
        XCTAssertEqual(expected.withUnsafeBufferPointer { channelError(nan, $0, columns: 3) }.worst, .infinity)
    }
}
