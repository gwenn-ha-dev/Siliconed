import CryptoKit
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import Measurement

/// **`silicontrol measure`** without an app: the options, the TSV, the summary, the pixel digest and the
/// procedure itself, driven by a fake app that answers on the injected transport.
final class MeasurementTests: XCTestCase {

    func testOptionsDefaultsAndPassthrough() throws {
        let d = try MeasureOptions.parse([])
        XCTAssertEqual(d.size, "1024x1024"); XCTAssertEqual(d.renders, 3); XCTAssertEqual(d.seed, 42)
        XCTAssertEqual(d.prompt(forRender: 1), "a 30 year old woman posing in a library")
        let o = try MeasureOptions.parse(["--model", "anima", "--size", "512", "--renders", "2", "--prompt", "a", "--prompt", "b",
                                          "--", "--steps", "6", "--lora", "flat"])
        XCTAssertEqual(o.size, "512x512")
        XCTAssertEqual([o.prompt(forRender: 1), o.prompt(forRender: 2), o.prompt(forRender: 3)], ["a", "b", "a"])
        XCTAssertEqual(o.extra, ["--steps", "6", "--lora", "flat"])
    }

    /// The series writes the header, then one line per render as it ends: the rows measured before a failure stay.
    func testRunnerWritesEachRenderAndKeepsTheRowsBeforeAFailure() throws {
        for failing in [false, true] {
            var added = 0
            let send: Runner.Send = { argv, onLine in
                switch argv[0] {
                case "status": return (["ok": true, "running": NSNull(), "queue": []], getpid())
                case "models": return (["ok": true, "models": [["id": "other", "ready": true],
                                                               ["id": "fake", "ready": true, "variant": "standard"]]], getpid())
                default:
                    added += 1
                    if failing, added == 3 { return (["ok": false, "error": "render", "message": "boom"], getpid()) }
                    let dir = URL(fileURLWithPath: argv[argv.firstIndex(of: "--out")! + 1], isDirectory: true)
                    let path = dir.appendingPathComponent("f-\(added).png")
                    self.writePNG(path) { x, y in (UInt8(x), UInt8(y), 7) }
                    onLine(["type": "image", "seconds": 0.5, "steps": 8, "path": path.path])
                    return (["ok": true], getpid())
                }
            }
            let o = try MeasureOptions.parse(["--model", "fake", "--renders", "1", "--settle", "0"])
            var lines: [String] = []
            if failing {
                XCTAssertThrowsError(try Runner.run(o, send: send, log: { _ in }, emit: { _ in }, append: { lines.append($0) })) {
                    XCTAssertEqual(($0 as? Failure)?.code, "render")
                }
            } else {
                _ = try Runner.run(o, send: send, log: { _ in }, emit: { _ in }, append: { lines.append($0) })
            }
            // The header, then the load, the two witnesses and the render — or the two rows before the failure.
            XCTAssertEqual(lines.count, failing ? 3 : 5, "failing: \(failing)")
            XCTAssertEqual(lines.first, TSV.header.joined(separator: "\t"))
            let column = TSV.header.firstIndex(of: "model")!
            for line in lines.dropFirst() { XCTAssertEqual(line.split(separator: "\t", omittingEmptySubsequences: false)[column], "fake") }
        }
    }

    func testOptionsRefuseWhatBreaksTheMeasure() {
        for argv in [["--size", "256"], ["--size", "1000x1024"], ["--renders", "0"], ["--seed", "x"], ["--bogus"], ["--model"],
                     ["--", "--batch", "2"], ["--", "--out", "x"], ["--", "--wait"]] {
            XCTAssertThrowsError(try MeasureOptions.parse(argv), "\(argv)")
        }
    }

    private func row(_ n: Int, _ role: Role, wall: Double, md5: String = "aa", prompt: String = "p", swaps: UInt64 = 0, peak: UInt64 = 1 << 30) -> Row {
        Row(n: n, role: role, model: "m", variant: "standard", size: "512x512", steps: 8, seed: 42, prompt: prompt,
            wallSeconds: wall, engineSeconds: wall - 0.1, startFootprint: 1 << 29,
            peakSampled: peak, swapouts: swaps, diskRead: 2_000_000_000, md5: md5)
    }

    func testSummaryMedianWitnessDriftAndNotes() {
        let rows = [row(0, .load, wall: 60), row(1, .witnessHead, wall: 40), row(2, .render, wall: 41), row(3, .render, wall: 43),
                    row(4, .render, wall: 42), row(5, .witnessTail, wall: 42)]
        let s = Summary(rows, gpuAlone: true)
        XCTAssertEqual(s.median, 42); XCTAssertEqual(s.fastest, 41); XCTAssertEqual(s.slowest, 43)
        XCTAssertEqual(s.drift!, 0.05, accuracy: 1e-12)
        XCTAssertTrue(s.bitsStable); XCTAssertTrue(s.notes.isEmpty)
        XCTAssertEqual(Summary.median([4, 1, 3, 2]), 2.5)

        var bad = rows
        bad[5].wallSeconds = 50; bad[3].md5 = "bb"; bad[4].swapouts = 7
        let t = Summary(bad, power: ["battery"], gpuAlone: true)
        XCTAssertFalse(t.bitsStable)
        XCTAssertEqual(t.notes.count, 4)   // drift, swap, battery, bits
        XCTAssertTrue(t.notes[3].contains("report it"))
        // Under ane the Neural Engine's share moves between renders: said, not a defect to report.
        let u = Summary(bad, power: ["battery"], gpuAlone: false)
        XCTAssertFalse(u.notes[3].contains("report it"))
    }

    func testTSVLineHasOneCellPerColumnAndNoTab() {
        let r = row(2, .render, wall: 12.345, prompt: "a\tb\nc")
        let c = Context(date: "2026-10-25T10:00:00+02:00", tag: "x", app: "0.15", chip: "Apple M1 Pro", memoryGiB: 16, macOS: "26.0 (25A1)")
        let cells = TSV.line(r, c, power: "AC", thermal: "nominal").split(separator: "\t", omittingEmptySubsequences: false)
        XCTAssertEqual(cells.count, TSV.header.count)
        XCTAssertEqual(cells[TSV.header.firstIndex(of: "wall_s")!], "12.35")
        XCTAssertEqual(cells[TSV.header.firstIndex(of: "peak_GiB")!], "1.000")
        XCTAssertEqual(cells[TSV.header.firstIndex(of: "prompt")!], "a b c")
    }

    /// Writes a w×h PNG whose pixels are `value(x, y)`; returns the RGB bytes it holds.
    @discardableResult
    private func writePNG(_ url: URL, w: Int = 8, h: Int = 5, alpha: Bool = false, value: (Int, Int) -> (UInt8, UInt8, UInt8)) -> [UInt8] {
        var data = [UInt8](repeating: 255, count: w * h * 4), rgb: [UInt8] = []
        for y in 0..<h { for x in 0..<w {
            let (r, g, b) = value(x, y), i = 4 * (y * w + x)
            data[i] = r; data[i + 1] = g; data[i + 2] = b
            rgb += [r, g, b]
        } }
        let info = alpha ? CGImageAlphaInfo.premultipliedLast : CGImageAlphaInfo.noneSkipLast
        let context = data.withUnsafeMutableBytes {
            CGContext(data: $0.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 4 * w,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: info.rawValue)!
        }
        let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, context.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return rgb
    }

    /// The digest is the MD5 of the decoded RGB bytes: PIL's `convert("RGB").tobytes()`.
    func testPixelDigestIsTheMD5OfTheDecodedRGB() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("measure-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        let rgb = writePNG(url) { x, y in (UInt8(x * 30), UInt8(y * 50), UInt8(x * y)) }
        let expected = Insecure.MD5.hash(data: rgb).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(Probes.pixelDigest(png: url), expected)
        XCTAssertNil(Probes.pixelDigest(png: url.appendingPathExtension("missing")))
    }

    /// The procedure, against a fake app: busy refused, then the plan (load, witness, renders, witness)
    /// with one TSV line each and a verdict that carries the summary.
    func testRunnerFollowsThePlanAgainstAFakeApp() throws {
        var added = 0, busy = false
        let send: Runner.Send = { argv, onLine in
            switch argv[0] {
            case "status":
                return (["ok": true, "running": busy ? ["job": "t1"] : NSNull(), "queue": []], getpid())
            case "models":
                return (["ok": true, "models": [["id": "fake", "ready": true, "variant": "standard"]]], getpid())
            default:
                added += 1
                let dir = URL(fileURLWithPath: argv[argv.firstIndex(of: "--out")! + 1], isDirectory: true)
                let path = dir.appendingPathComponent("fake-\(added).png")
                self.writePNG(path) { x, y in (UInt8(x), UInt8(y), 7) }
                onLine(["type": "image", "seconds": 0.5, "steps": 8, "path": path.path])
                return (["ok": true], getpid())
            }
        }
        var o = try MeasureOptions.parse(["--model", "fake", "--size", "512", "--renders", "2", "--settle", "0"])
        var lines: [String] = [], emitted = 0
        let verdict = try Runner.run(o, send: send, log: { _ in }, emit: { _ in emitted += 1 }, append: { lines.append($0) })
        XCTAssertEqual(added, 5); XCTAssertEqual(emitted, 5)
        XCTAssertEqual(lines.count, 6)   // header + five renders
        XCTAssertEqual(lines[0], TSV.header.joined(separator: "\t"))
        XCTAssertEqual(lines.dropFirst().map { $0.split(separator: "\t", omittingEmptySubsequences: false)[TSV.header.firstIndex(of: "role")!] }, ["load", "witness-head", "render", "render", "witness-tail"])
        let summary = verdict["summary"] as? [String: Any]
        XCTAssertEqual(summary?["renders"] as? Int, 2)
        XCTAssertEqual(summary?["bits_stable"] as? Bool, true)

        busy = true
        XCTAssertThrowsError(try Runner.run(o, send: send, log: { _ in }, emit: { _ in }, append: { _ in })) {
            XCTAssertEqual(($0 as? Failure)?.code, "busy")
        }
        busy = false
        // Without --model, the app's default; this fake app marks none.
        o.model = nil
        XCTAssertThrowsError(try Runner.run(o, send: send, log: { _ in }, emit: { _ in }, append: { _ in })) {
            XCTAssertEqual(($0 as? Failure)?.code, "model")
        }
        o.model = "nope"
        XCTAssertThrowsError(try Runner.run(o, send: send, log: { _ in }, emit: { _ in }, append: { _ in })) {
            XCTAssertEqual(($0 as? Failure)?.code, "model")
        }
    }

    func testProbesReadThisProcess() {
        let u = Probes.usage(of: getpid())
        XCTAssertNotNil(u)
        XCTAssertGreaterThan(u?.footprint ?? 0, 1 << 20)
        let sampler = Probes.PeakSampler(pid: getpid(), interval: 0.01)
        Thread.sleep(forTimeInterval: 0.05)
        XCTAssertGreaterThanOrEqual(sampler.stop().peak, 1 << 20)
        XCTAssertGreaterThan(Probes.Host.read().memoryGiB, 0)
        XCTAssertFalse(Probes.powerSource().isEmpty)
    }
}
