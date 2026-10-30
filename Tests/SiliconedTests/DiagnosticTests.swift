import Foundation
import XCTest
@testable import Siliconed

/// **The diagnostic's means without a GPU and without a byte of weights**: the conditions and flags it reads, the issue link, the
/// blocks it widens, and its CPU references against naive loops. The report itself, its JSON and its verdict are
/// `DiagnosticReportTests` and `DiagnosticVerdictTests`.
final class DiagnosticTests: XCTestCase {
    /// The machine's state and flags read without a GPU.
    func testConditionsAndFeaturesRead() {
        let c = Diagnostic.conditions()
        XCTAssertTrue(["nominal", "fair", "serious", "critical"].contains(c.thermal))
        XCTAssertGreaterThanOrEqual(c.swapUsedBytes, 0)
        let flags = Diagnostic.cpuFeatures()
        for (name, value) in flags where name.hasPrefix("FEAT_") { XCTAssertTrue([0, 1].contains(value), name) }
        XCTAssertTrue(Set(flags.keys).isSubset(of: Set(Diagnostic.featureNames + ["sme_max_svl_b"])))
    }

    /// A machine at rest is a fact the report states, not a condition of its verdict.
    func testNominalIsWhatTheMachineWasDoing() {
        func conditions(_ thermal: String = "nominal", lowPower: Bool = false, battery: Bool? = nil, swap: Int = 0) -> Diagnostic.Conditions {
            Diagnostic.Conditions(thermal: thermal, lowPowerMode: lowPower, onBattery: battery, swapUsedBytes: swap)
        }
        XCTAssertTrue(conditions(battery: true).isNominal)
        XCTAssertFalse(conditions("fair").isNominal)
        XCTAssertFalse(conditions(lowPower: true).isNominal)
        let encoded = String(decoding: try! JSONEncoder().encode(conditions()), as: UTF8.self)
        XCTAssertTrue(encoded.contains("\"onBattery\":null"), "unknown is said, not dropped")
    }

    // ── the issue URL ───────────────────────────────────────────────────────────────────────

    /// The link carries the machine's fields; the report is pasted (it outgrew a link's ~8 000 characters).
    func testIssueURLCarriesTheFieldsAndNotTheReport() throws {
        let report = ReportFixtures.report()
        let url = try report.issueURL()
        XCTAssertTrue(url.absoluteString.hasPrefix("https://github.com/gwenn-ha-dev/Siliconed/issues/new?template=diagnostic.yml&"))
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems!
        func field(_ name: String) -> String? { items.first { $0.name == name }?.value }
        XCTAssertEqual(field("chip"), "Apple M1 Pro (14-core GPU)")
        XCTAssertEqual(field("memory"), "16 GB")
        XCTAssertEqual(field("macos"), "27.0.1 (26A434)")
        XCTAssertEqual(field("version"), "git 0123456789-dirty")
        XCTAssertEqual(field("labels"), "diagnostic")
        XCTAssertEqual(field("template"), "diagnostic.yml")
        XCTAssertNil(field("report"))
        XCTAssertLessThanOrEqual(url.absoluteString.utf8.count, 1_000)
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

    /// No account name in what the app shows or a report publishes.
    func testTheHomeFolderIsWrittenTilde() {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        XCTAssertEqual(Library.withoutHome("file missing: \(home)/Library/x.silicon"), "file missing: ~/Library/x.silicon")
        XCTAssertEqual(Library(root: URL(fileURLWithPath: home + "/Library/Application Support/Siliconed")).displayPath,
                       "~/Library/Application Support/Siliconed")
        XCTAssertEqual(Library.withoutHome("/opt/x"), "/opt/x")
    }

    /// A figure to `digits` significant digits, a non-finite one the largest finite double (JSON has no infinity).
    func testSignificantDigits() {
        XCTAssertEqual(Diagnostic.significant(3.461234567, 4), 3.461)
        XCTAssertEqual(Diagnostic.significant(0, 3), 0)
        XCTAssertEqual(Diagnostic.significant(-2.71828, 2), -2.7)
        for bad in [Double.nan, .infinity] {
            XCTAssertEqual(Diagnostic.significant(bad, 3), .greatestFiniteMagnitude)
            XCTAssertNoThrow(try JSONEncoder().encode(Diagnostic.significant(bad, 3)))
        }
    }

    // ── what it widens and judges by ────────────────────────────────────────────────────────

    /// The blocks the widening probes size on, from the models' defaults.
    func testTheBlocksAreTheModelsDefaults() {
        XCTAssertEqual(Diagnostic.blocks.map(\.heads), [30, 32])
        XCTAssertEqual(Diagnostic.blocks.map(\.headDim), [128, 128])
        XCTAssertEqual(Diagnostic.blocks.map(\.model), ["z-image", "qwen-image-2.1"])
    }

    /// The sampled rows are distinct, in range, and visit every quantum of the cut.
    func testSampledRowsCoverTheCut() {
        for m in [1024, 4096] {
            let rows = Diagnostic.sampleRows(m, count: Diagnostic.sampledRows)
            XCTAssertEqual(Set(rows).count, rows.count)
            XCTAssertTrue(rows.allSatisfy { (0..<m).contains($0) })
            XCTAssertEqual(Set(rows.map { $0 / 256 }).count, m / 256)
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
