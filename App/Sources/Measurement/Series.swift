// **The shape of a measured series**: its options, one row per render, the TSV they make and the
// summary read from them. Pure — nothing here touches the app, the machine or a file — so the numbers
// a series prints can be tested without a weight.

import Foundation

package struct MeasureOptions: Equatable, Sendable {
    package static let defaultPrompt = "a 30 year old woman posing in a library"

    /// `--model ID`; `nil`, the app's default (the entry « models » marks `"default": true`).
    package var model: String?
    package var size = "1024x1024"
    package var renders = 3
    package var seed = 42
    package var prompts: [String] = []
    package var tsv: String?
    package var out: String?
    package var settle = 15
    package var tag = ""
    /// Passed to `add` as given, after `--`: `--steps`, `--lora`…
    package var extra: [String] = []

    package struct Failure: Error, Equatable { package let message: String }

    /// Options the series owns: a `--` passthrough may not change what is measured or how it ends.
    private static let reserved: Set<String> = ["--wait", "--out", "--batch", "--ref", "--model", "--preview", "--no-preview"]

    private static let valued: Set<String> = ["--model", "--size", "--renders", "--seed", "--settle", "--prompt", "--tsv", "--out", "--tag"]

    /// The size is read as the app reads one (`SizeSyntax`: `1024`, `832x1216`, `832 × 1216`) and handed to `add` as
    /// `WxH`; whether the model renders it is the app's to say, at the first render.
    package static func parse(_ argv: [String]) throws -> MeasureOptions {
        let flags: Flags
        switch Flags.scan(argv, valued: valued) {
        case .success(let f): flags = f
        case .failure(let failure): throw Failure(message: failure.message)
        }
        if let word = flags.words.first { throw Failure(message: "unexpected argument \"\(word)\": the options of add go after --") }
        func number(_ flag: String, min: Int) throws -> Int? {
            guard let v = flags.value(flag) else { return nil }
            guard let n = Int(v), n >= min else { throw Failure(message: "\(flag) needs an integer ≥ \(min)") }
            return n
        }
        var o = MeasureOptions()
        o.model = flags.value("--model")
        if let v = flags.value("--size") {
            guard let (w, h) = SizeSyntax.parse(v) else { throw Failure(message: "--size: 512, 1024 or WxH (832x1216)") }
            o.size = "\(w)x\(h)"
        }
        o.renders = try number("--renders", min: 1) ?? o.renders
        o.seed = try number("--seed", min: 0) ?? o.seed
        o.settle = try number("--settle", min: 0) ?? o.settle
        o.prompts = flags.values["--prompt"] ?? []
        o.tsv = flags.value("--tsv")
        o.out = flags.value("--out")
        o.tag = flags.value("--tag") ?? o.tag
        o.extra = flags.rest
        if let bad = o.extra.first(where: { reserved.contains($0) }) {
            throw Failure(message: "\(bad) belongs to the measure itself and cannot follow --")
        }
        return o
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
    package var wallSeconds: Double, engineSeconds: Double?
    package var startFootprint: UInt64, peakSampled: UInt64
    package var swapouts: UInt64, diskRead: UInt64
    package var md5: String
    /// The app's Neural Engine switch and its B when the render was asked (`status`); `nil` when the app did not say.
    package var neuralEngine: Bool? = nil
    package var budget: Int? = nil
}

/// What is the same on every row of a series: when, where, with which build.
package struct Context: Sendable, Equatable {
    package var date: String, tag: String, app: String
    package var chip: String, memoryGiB: Int, macOS: String
}

package enum TSV {
    package static let header = ["date", "tag", "n", "role", "model", "variant", "size", "steps", "seed", "prompt",
                                 "wall_s", "engine_s", "start_GiB", "peak_GiB",
                                 "swapouts", "disk_read_GB", "md5", "app", "chip", "memory_GiB", "macos",
                                 "power", "thermal", "neural_engine", "budget"]

    private static func gib(_ b: UInt64) -> String { String(format: "%.3f", Double(b) / 1_073_741_824) }
    private static func clean(_ s: String) -> String {
        s.map { $0 == "\t" || $0 == "\n" || $0 == "\r" ? " " : $0 }.reduce(into: "") { $0.append($1) }
    }

    /// `on` or `off`, `-` when the app did not say.
    package static func switchText(_ on: Bool?) -> String { on.map { $0 ? "on" : "off" } ?? "-" }

    /// `power` and `thermal` are read at each render: a series that left the adapter shows where. `budget` is B in GiB.
    package static func line(_ r: Row, _ c: Context, power: String, thermal: String) -> String {
        [c.date, clean(c.tag), String(r.n), r.role.rawValue, r.model, r.variant, r.size, r.steps.map(String.init) ?? "-",
         String(r.seed), clean(r.prompt),
         String(format: "%.2f", r.wallSeconds), r.engineSeconds.map { String(format: "%.2f", $0) } ?? "-",
         gib(r.startFootprint), gib(r.peakSampled), String(r.swapouts),
         String(format: "%.2f", Double(r.diskRead) / 1e9), r.md5, c.app, c.chip, String(c.memoryGiB), c.macOS,
         power, thermal, switchText(r.neuralEngine), r.budget.map { gib(UInt64($0)) } ?? "-"].joined(separator: "\t")
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
    /// Same prompt and seed gave the same pixels.
    package var bitsStable: Bool
    /// The series ran with the Neural Engine off: the GPU alone, one plan per render, so unstable bits are a defect. With
    /// it on the engine moves work to the Neural Engine from one render to the next, and the pixels may move with it.
    package var gpuAlone: Bool
    package var distinctPictures: Int
    /// The Neural Engine switch and B of the series' first render; a change during the series is a note.
    package var neuralEngine: Bool?
    package var budget: Int?
    package var notes: [String]

    /// Past this, the machine moved more than the 4 to 13 % that series have shown: the series says nothing.
    package static let driftLimit = 0.13

    package static func median(_ x: [Double]) -> Double? {
        guard !x.isEmpty else { return nil }
        let s = x.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    package init(_ rows: [Row], power: Set<String> = [], gpuAlone: Bool) {
        self.gpuAlone = gpuAlone
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
        neuralEngine = rows.first?.neuralEngine
        budget = rows.first?.budget
        var notes: [String] = []
        if Set(rows.map { TSV.switchText($0.neuralEngine) }).count > 1 || Set(rows.map { $0.budget ?? -1 }).count > 1 {
            notes.append("the Neural Engine switch or B changed during the series: its renders were not made under one setting")
        }
        if let d = drift, abs(d) > Self.driftLimit {
            notes.append(String(format: "the witness moved %+.1f %% over the series (beyond %.0f %%): do not compare it to another series", d * 100, Self.driftLimit * 100))
        }
        if swapouts > 0 { notes.append("\(swapouts) pages were swapped out during the series: its times include the swap") }
        if power.contains("battery") { notes.append("part of the series ran on the battery") }
        if !bitsStable {
            notes.append(gpuAlone ? "the same prompt and seed gave different pixels with the Neural Engine off: report it, this is not a measure"
                                  : "the same prompt and seed gave different pixels: with the Neural Engine on its share moves between renders")
        }
        self.notes = notes
    }
}
