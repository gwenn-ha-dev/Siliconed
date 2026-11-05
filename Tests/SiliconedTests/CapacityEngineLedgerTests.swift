import Foundation
import XCTest
@testable import Siliconed

/// **The catalogue against what the engine executed** (`CapacityClasses.swift` against `ProductLedger`): for each of the five families, the classes a
/// Compact render at 1 024² of the benchmark prompt recorded — the product or attention, its executions, heights merged within `RenderLoop.heightGap`
/// rows as the comparison does — and the catalogue's own list for the same request. A class the engine runs that the catalogue does not
/// carry (MOSS-Image's four apart projections taken for one fused, Z-Image's refiners taken for 8-bit, Krea 2's tokens counted twice) is a
/// render no prediction can judge: the fixtures are the engine's, taken from renders, not written from the code.
final class CapacityEngineLedgerTests: XCTestCase {
    typealias CC = CapacityClasses

    // MARK: the engine's ledgers (`id executions`)

    static let zImage: [String] = [
        "product:3840x10240@4128~int8 320",
        "product:3840x3840@4128~int8 640",
        "product:10240x3840@4128~int8 160",
        "attention:4128/30x128 150",
        "product:3840x3840@1056~int8 384",
        "product:3840x10240@1056~int8 192",
        "product:10240x3840@1056~int8 96",
        "attention:1056/30x128 90",
        "attention:4096/30x128 10",
        "product:3840x3840@32~int8 64",
        "product:3840x10240@32~int8 32",
        "product:10240x3840@32~int8 16",
        "attention:1024/30x128 6",
        "product:2560x3840@32 8",
        "product:3840x64@4128 5",
        "product:64x3840@4096 5",
        "product:64x3840@1024 3",
        "product:3840x64@1056 3",
    ]
    static let qwenImage21: [String] = [
        "attention:4096/32x128 160",
        "product:4096x12288@4096~int8 320",
        "product:4096x4096@4096~int8 640",
        "product:12288x4096@4096~int8 160",
        "product:4096x4096@1024~int8 384",
        "product:4096x12288@1024~int8 192",
        "product:12288x4096@1024~int8 96",
        "attention:1024/32x128 96",
        "product:4096x12288@19~int8 124",
        "product:4096x4096@19~int8 252",
        "product:12288x4096@19~int8 62",
        "product:4096x4096@19 4",
        "product:64x4096@4096 5",
        "product:4096x64@4096 5",
        "product:64x4096@1024 3",
        "product:4096x64@1024 3"
    ]
    static let anima: [String] = [
        "product:2048x2048@4096~int8 780",
        "attention:4096/16x128 280",
        "product:8192x2048@4096~int8 115",
        "product:2048x8192@4096~int8 115",
        "product:2048x2048@1024~int8 468",
        "attention:1024/16x128 168",
        "product:2048x8192@4096 25",
        "product:8192x2048@4096 25",
        "product:8192x2048@1024~int8 69",
        "product:2048x8192@1024~int8 69",
        "product:2048x2048@4096 60",
        "product:1024x2048@512~int8 104",
        "product:2048x2048@1024 36",
        "product:8192x2048@1024 15",
        "product:2048x8192@1024 15",
        "product:68x2048@4096 5",
        "product:1024x2048@512 8",
        "product:2048x64@4096 5",
        "product:68x2048@1024 3",
        "product:2048x64@1024 3",
    ]
    static let krea2: [String] = [
        "product:6144x16384@4112~int8 280",
        "attention:4112/48x128 140",
        "product:16384x6144@4112~int8 140",
        "product:6144x6144@4112~int8 420",
        "product:6144x16384@1040~int8 168",
        "product:6144x1536@4112~int8 280",
        "product:16384x6144@1040~int8 84",
        "product:6144x6144@1040~int8 252",
        "attention:1040/48x128 84",
        "product:6144x1536@1040~int8 168",
        "product:64x6144@4096 5",
        "product:2560x2560@192 10",
        "product:64x6144@1024 3",
        "product:2560x6912@192 4",
        "product:6144x64@4096 5",
        "product:2560x2560@16 10",
        "product:6912x2560@192 2",
        "product:2560x6912@16 4",
        "product:6144x64@1024 3",
        "product:6144x6144@16 1",
        "product:6912x2560@16 2",
        "product:2560x6144@16 1"
    ]
    static let mossImage: [String] = [
        "attention:4117/24x128 294",
        "product:8192x3072@4117~int8 294",
        "product:3072x3072@4117~int8 1176",
        "product:3072x16384@4117~int8 294",
        "product:3072x3072@1045~int8 168",
        "product:8192x3072@1045~int8 42",
        "product:3072x16384@1045~int8 42",
        "attention:1045/24x128 42",
        "product:3072x8192@21 32",
        "product:3072x9216@21 16",
        "product:8192x3072@21 16",
        "product:4096x3072@21 8",
        "product:3072x3072@21 16",
        "product:128x3072@4096 7",
        "product:3072x128@4096 7",
        "product:3072x128@1024 1",
        "product:128x3072@1024 1",
        // The DMD LoRA's two products on each of the 252 modules, every evaluation (`WeightProducts.lora`, since the stack is the plan's):
        // q, k, v, o and the fused gate+up read 3 072 columns, the down projection 8 192; the fused up writes 16 384, the others 3 072.
        "product:3072x128@4117~loraDown 1470",
        "product:8192x128@4117~loraDown 294",
        "product:128x3072@4117~loraUp 1470",
        "product:128x16384@4117~loraUp 294",
        "product:3072x128@1045~loraDown 210",
        "product:8192x128@1045~loraDown 42",
        "product:128x3072@1045~loraUp 210",
        "product:128x16384@1045~loraUp 42"
    ]

    // MARK: reading both sides

    /// A class as the comparison merges it: the shape without its height, the heights within the gap chained into one, the executions added.
    private struct Cluster: Hashable { var shape: String; var low: Int; var high: Int }

    private func clusters(_ items: [(shape: String, rows: Int, executions: Int)], merging: Bool) -> [Cluster: Int] {
        var by: [String: [(rows: Int, executions: Int)]] = [:]
        for i in items { by[i.shape, default: []].append((i.rows, i.executions)) }
        var out: [Cluster: Int] = [:]
        for (shape, heights) in by {
            var chains: [[(rows: Int, executions: Int)]] = []
            for h in heights.sorted(by: { $0.rows < $1.rows }) {
                if merging, let last = chains.last?.last, h.rows - last.rows <= RenderLoop.heightGap { chains[chains.count - 1].append(h) } else { chains.append([h]) }
            }
            for c in chains { out[Cluster(shape: shape, low: c.first!.rows, high: c.last!.rows), default: 0] += c.reduce(0) { $0 + $1.executions } }
        }
        return out
    }

    private func engine(_ lines: [String]) -> [Cluster: Int] {
        var items: [(shape: String, rows: Int, executions: Int)] = []
        for line in lines {
            let parts = line.split(separator: " "), id = String(parts[0]), n = Int(parts[1])!
            if id.hasPrefix("attention:") {
                // attention:S/HxD — exact in S, so a shape of its own per S.
                let body = id.dropFirst("attention:".count).split(separator: "/")
                items.append((shape: "attention " + body[1], rows: Int(body[0])!, executions: n))
            } else {
                // product:KxN@M[~format]
                var body = String(id.dropFirst("product:".count)), format = "bf16"
                if let tilde = body.firstIndex(of: "~") { format = String(body[body.index(after: tilde)...]); body = String(body[..<tilde]) }
                let at = body.split(separator: "@")
                items.append((shape: "product \(at[0]) \(format)", rows: Int(at[1])!, executions: n))
            }
        }
        // An attention is exact: its heights are never chained.
        let products = items.filter { $0.shape.hasPrefix("product") }, attentions = items.filter { $0.shape.hasPrefix("attention") }
        return clusters(products, merging: true).merging(clusters(attentions.map { (shape: $0.shape + " @\($0.rows)", rows: $0.rows, executions: $0.executions) }, merging: false)) { $0 + $1 }
    }

    private func catalogue(_ family: Family, renderTokens: Int, halfSteps: Int) -> [Cluster: Int] {
        let request = CC.Request(family: family, variant: .compact, width: 1024, height: 1024, fasterRendering: true,
                                 promptTokens: CC.promptTokens(family, renderTokens: renderTokens), steps: 8, includesEncoder: false, halfSteps: halfSteps)
        var items: [(shape: String, rows: Int, executions: Int)] = []
        var attentions: [(shape: String, rows: Int, executions: Int)] = []
        for c in CC.classes(for: request)! where c.phase == .denoising {
            let n = c.executions[family] ?? 0
            switch c.shape {
            case let .product(k, nn, rows, format): items.append((shape: "product \(k)x\(nn) \(format == .bf16 ? "bf16" : format.rawValue)", rows: rows, executions: n))
            case let .attention(s, _, heads, _, dim, _): attentions.append((shape: "attention \(heads)x\(dim) @\(s)", rows: s, executions: n))
            }
        }
        return clusters(items, merging: true).merging(clusters(attentions, merging: false)) { $0 + $1 }
    }

    private func assertSame(_ family: Family, _ ledger: [String], renderTokens: Int, halfSteps: Int, file: StaticString = #filePath, line: UInt = #line) {
        let ran = engine(ledger), carried = catalogue(family, renderTokens: renderTokens, halfSteps: halfSteps)
        // Same classes by shape and by height (a cluster's heights may differ by the gap between the engine's and the catalogue's own: the prompt's tokens)
        func key(_ c: Cluster) -> String { c.shape }
        var unmatched = carried
        for (cluster, n) in ran.sorted(by: { ($0.key.shape, $0.key.low) < ($1.key.shape, $1.key.low) }) {
            let near = unmatched.keys.filter { key($0) == key(cluster) && $0.low - RenderLoop.heightGap <= cluster.high && cluster.low <= $0.high + RenderLoop.heightGap }
            guard let hit = near.first else { XCTFail("the engine ran \(cluster) × \(n) and the catalogue does not carry it", file: file, line: line); continue }
            XCTAssertEqual(unmatched[hit], n, "\(cluster)", file: file, line: line)
            unmatched[hit] = nil
        }
        XCTAssertTrue(unmatched.isEmpty, "the catalogue carries what the engine did not run: \(unmatched.keys.sorted { $0.shape < $1.shape })", file: file, line: line)
    }

    // MARK: one family each

    func testZImage() { assertSame(.zImage, Self.zImage, renderTokens: 19, halfSteps: 3) }
    func testQwenImage21() { assertSame(.qwenImage21, Self.qwenImage21, renderTokens: 19, halfSteps: 3) }
    func testAnima() { assertSame(.anima, Self.anima, renderTokens: 11, halfSteps: 3) }
    func testKrea2() { assertSame(.krea2, Self.krea2, renderTokens: 16, halfSteps: 3) }
    func testMossImage() { assertSame(.mossImage, Self.mossImage, renderTokens: 11, halfSteps: 1) }

    // MARK: the tokens

    /// `Render.tokens` is each family's own count « for display »; the catalogue counts the prompt's.
    func testTheRenderTokensAreReadPerFamily() {
        for (family, tokens) in [(Family.zImage, 19), (.qwenImage21, 19), (.krea2, 16), (.mossImage, 11), (.anima, 11)] {
            let p = CC.promptTokens(family, renderTokens: tokens)
            XCTAssertEqual(CC.textRows(family, promptTokens: p), family == .zImage ? 32 : family == .anima ? 0 : family == .krea2 ? 16 : family == .mossImage ? 21 : 19,
                           "\(family): the rows the engine joins to the image's")
        }
    }
}
