// **The shape of a measured series**: its options, one row per render, the TSV they make and the
// summary read from them. Pure — nothing here touches the app, the machine or a file — so the numbers
// a series prints can be tested without a weight.

import Foundation

package struct MeasureOptions: Equatable, Sendable {
    package static let defaultPrompt = "a 30 year old woman posing in a library"

    package var model = "z-image"
    package var size = "1024x1024"
    package var renders = 3
    package var seed = 42
    package var prompts: [String] = []
    package var tsv: String?
    package var out: String?
    package var settle = 15
    package var tag = ""
    /// `--faster-rendering on|off`: the setting for the series, put back at its end; `nil`, the app's as it is.
    package var fasterRendering: Bool?
    /// Passed to `add` as given, after `--`: `--steps`, `--accelerators`, `--no-accelerators`, `--lora`…
    package var extra: [String] = []

    package struct Failure: Error, Equatable { package let message: String }

    /// Options the series owns: a `--` passthrough may not change what is measured or how it ends.
    private static let reserved: Set<String> = ["--wait", "--out", "--batch", "--ref", "--model", "--preview", "--no-preview"]

    package static func parse(_ argv: [String]) throws -> MeasureOptions {
        var o = MeasureOptions()
        var i = 0
        func value(_ flag: String) throws -> String {
            i += 1
            guard i < argv.count else { throw Failure(message: "\(flag) needs a value") }
            return argv[i]
        }
        func number(_ flag: String, min: Int) throws -> Int {
            guard let n = Int(try value(flag)), n >= min else { throw Failure(message: "\(flag) needs an integer ≥ \(min)") }
            return n
        }
        while i < argv.count {
            let a = argv[i]
            switch a {
            case "--model": o.model = try value(a)
            case "--size":
                let v = try value(a)
                o.size = v == "512" ? "512x512" : v == "1024" ? "1024x1024" : v
                guard let (w, h) = dimensions(o.size) else { throw Failure(message: "--size: 512, 1024 or WxH") }
                guard min(w, h) >= 512, w % 16 == 0, h % 16 == 0 else {
                    throw Failure(message: "--size: each side ≥ 512 and a multiple of 16")
                }
            case "--renders": o.renders = try number(a, min: 1)
            case "--seed": o.seed = try number(a, min: 0)
            case "--settle": o.settle = try number(a, min: 0)
            case "--prompt": o.prompts.append(try value(a))
            case "--tsv": o.tsv = try value(a)
            case "--out": o.out = try value(a)
            case "--tag": o.tag = try value(a)
            case "--faster-rendering":
                switch try value(a) {
                case "on": o.fasterRendering = true
                case "off": o.fasterRendering = false
                default: throw Failure(message: "--faster-rendering: on or off")
                }
            case "--":
                o.extra = Array(argv[(i + 1)...])
                i = argv.count
            default: throw Failure(message: "unknown option \(a)")
            }
            i += 1
        }
        if let bad = o.extra.first(where: { reserved.contains($0) }) {
            throw Failure(message: "\(bad) belongs to the measure itself and cannot follow --")
        }
        return o
    }

    package static func dimensions(_ size: String) -> (Int, Int)? {
        let p = size.split(separator: "x").compactMap { Int($0) }
        return p.count == 2 && size.split(separator: "x").count == 2 ? (p[0], p[1]) : nil
    }

    /// The prompt of render `n` (1-based): the list in turn, the default prompt when none was given.
    package func prompt(forRender n: Int) -> String {
        prompts.isEmpty ? Self.defaultPrompt : prompts[(n - 1) % prompts.count]
    }
}

package enum Role: String, Sendable { case load, witnessHead = "witness-head", render, witnessTail = "witness-tail" }

package struct Row: Sendable, Equatable {
    package var n: Int
    package var role: Role
    package var model: String, variant: String, size: String
    package var steps: Int?, seed: Int
    package var prompt: String
    package var fasterRendering: Bool?
    /// The « Accelerators » switch the render took (`--accelerators`, the model's choice); the column keeps its name, `neural_engine`.
    package var neuralEngine: Bool?
    package var wallSeconds: Double, engineSeconds: Double?
    package var startFootprint: UInt64, peakSampled: UInt64
    package var swapouts: UInt64, diskRead: UInt64
    package var md5: String
}

/// What is the same on every row of a series: when, where, with which build.
package struct Context: Sendable, Equatable {
    package var date: String, tag: String, app: String
    package var chip: String, memoryGiB: Int, macOS: String
}

package enum TSV {
    package static let header = ["date", "tag", "n", "role", "model", "variant", "size", "steps", "seed", "prompt",
                                 "faster_rendering", "neural_engine", "wall_s", "engine_s", "start_GiB", "peak_GiB",
                                 "swapouts", "disk_read_GB", "md5", "app", "chip", "memory_GiB", "macos",
                                 "power", "thermal"]

    private static func gib(_ b: UInt64) -> String { String(format: "%.3f", Double(b) / 1_073_741_824) }
    private static func flag(_ b: Bool?) -> String { b.map { $0 ? "on" : "off" } ?? "-" }
    private static func clean(_ s: String) -> String {
        s.map { $0 == "\t" || $0 == "\n" || $0 == "\r" ? " " : $0 }.reduce(into: "") { $0.append($1) }
    }

    /// `power` and `thermal` are read at each render: a series that left the adapter shows where.
    package static func line(_ r: Row, _ c: Context, power: String, thermal: String) -> String {
        [c.date, clean(c.tag), String(r.n), r.role.rawValue, r.model, r.variant, r.size, r.steps.map(String.init) ?? "-",
         String(r.seed), clean(r.prompt), flag(r.fasterRendering), flag(r.neuralEngine),
         String(format: "%.2f", r.wallSeconds), r.engineSeconds.map { String(format: "%.2f", $0) } ?? "-",
         gib(r.startFootprint), gib(r.peakSampled), String(r.swapouts),
         String(format: "%.2f", Double(r.diskRead) / 1e9), r.md5, c.app, c.chip, String(c.memoryGiB), c.macOS,
         power, thermal].joined(separator: "\t")
    }
}

package struct Summary: Sendable, Equatable {
    package var renders: Int
    package var median: Double?, fastest: Double?, slowest: Double?
    package var witnessHead: Double?, witnessTail: Double?
    /// (tail − head) / head: the machine's drift over the series, the margin any comparison is read with.
    package var drift: Double?
    package var load: Double?
    package var peak: UInt64
    package var swapouts: UInt64
    package var diskRead: UInt64
    /// Same prompt and seed must give the same pixels; `false` is a defect to report, not a measure.
    package var bitsStable: Bool
    package var distinctPictures: Int
    package var notes: [String]

    /// Past this, the machine moved more than the 4 to 13 % that series have shown: the series says nothing.
    package static let driftLimit = 0.13

    package static func median(_ x: [Double]) -> Double? {
        guard !x.isEmpty else { return nil }
        let s = x.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    package init(_ rows: [Row], power: Set<String> = []) {
        let renders = rows.filter { $0.role == .render }
        self.renders = renders.count
        median = Self.median(renders.map(\.wallSeconds))
        fastest = renders.map(\.wallSeconds).min()
        slowest = renders.map(\.wallSeconds).max()
        let head = rows.first { $0.role == .witnessHead }?.wallSeconds
        let tail = rows.first { $0.role == .witnessTail }?.wallSeconds
        witnessHead = head
        witnessTail = tail
        load = rows.first { $0.role == .load }?.wallSeconds
        drift = head.flatMap { h in tail.map { ($0 - h) / h } }
        peak = rows.map(\.peakSampled).max() ?? 0
        swapouts = rows.reduce(0) { $0 + $1.swapouts }
        diskRead = rows.reduce(0) { $0 + $1.diskRead }
        // One picture per (prompt, seed): every row of a group must carry the same pixels.
        var groups: [String: Set<String>] = [:]
        for r in rows { groups["\(r.prompt)\u{1}\(r.seed)\u{1}\(r.size)", default: []].insert(r.md5) }
        bitsStable = groups.values.allSatisfy { $0.count == 1 }
        distinctPictures = Set(rows.map(\.md5)).count
        var notes: [String] = []
        if let d = drift, abs(d) > Self.driftLimit {
            notes.append(String(format: "the witness moved %+.1f %% over the series (beyond %.0f %%): do not compare it to another series", d * 100, Self.driftLimit * 100))
        }
        if swapouts > 0 { notes.append("\(swapouts) pages were swapped out during the series: its times include the swap") }
        if power.contains("battery") { notes.append("part of the series ran on the battery") }
        if !bitsStable { notes.append("the same prompt and seed gave different pixels: report it, this is not a measure") }
        self.notes = notes
    }
}
