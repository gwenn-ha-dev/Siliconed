import Foundation

// **The registry: every stage that is verified, in one list.** `siliconed check list` prints it, the
// help is read from it, the verification guide's table is its Markdown, and the maintainer's closing run
// walks it. A stage that is not here is not verified.
//
//     siliconed check list [--markdown]
//     siliconed check kernels [NAME]                 the stages without weights, all or one
//     siliconed check <model> <stage> [options]      one stage (Invocation.swift: the options)

package enum Registry {
    /// In the order a reader meets them: the kernels, then each model from its text to its image.
    package static let stages: [Stage] = KernelStages.all + ZImageStages.all + QwenImageStages.all + AnimaStages.all + Krea2Stages.all + MossImageStages.all

    package static func stage(_ model: String, _ name: String) -> Stage? {
        stages.first { $0.model == model && $0.name == name }
    }
}

/// `siliconed check …`: the dispatch. Returns the process's exit code. `stages`: the registry, or the
/// registry and a maintainer's own stages beside it.
package enum CheckCommand {
    package static func run(_ arguments: [String], stages all: [Stage] = Registry.stages) -> Int32 {
        guard let first = arguments.first else { print(usage); return 64 }
        switch first {
        case "list":
            print(arguments.dropFirst().first == "--markdown" ? markdown(all) : table(all))
            return 0
        case "kernels":
            let kernels = all.filter { $0.model == "kernels" }
            if let name = arguments.dropFirst().first, !name.hasPrefix("--") {
                guard let stage = kernels.first(where: { $0.name == name }) else {
                    return unknown("kernel '\(name)'", among: kernels.map(\.name))
                }
                return stage.execute(Array(arguments.dropFirst(2))).exitCode
            }
            return inTurn(kernels.map { ($0.name, $0, Array(arguments.dropFirst())) })
        default:
            let stages = all.filter { $0.model == first }
            guard !stages.isEmpty else {
                return unknown("model '\(first)'", among: all.reduce(into: []) { if !$0.contains($1.model) { $0.append($1.model) } })
            }
            guard let name = arguments.dropFirst().first, !name.hasPrefix("--") else {
                return inTurn(everyStage(stages, Array(arguments.dropFirst())))
            }
            guard let stage = stages.first(where: { $0.name == name }) else {
                return unknown("stage '\(name)' of \(first)", among: stages.map(\.name))
            }
            return stage.execute(Array(arguments.dropFirst(2))).exitCode
        }
    }

    static let usage = """
        siliconed check list [--markdown]              every verified stage: what it proves, its rule, its oracle
        siliconed check kernels [NAME]                 the kernels against their definition, no weights
        siliconed check <model> [--size S] [--goldens DIR]   every stage, at every size the model's oracle `all` writes
        siliconed check <model> <stage> [--size 512|1024|WxH] [--goldens DIR] …
        """

    /// `check <model>`: each stage at each of its sizes (or at `--size` alone, where the stage has it) and
    /// in each of its variants — what the oracle's `all` writes. `--long` and `--neural-engine` stay per
    /// stage, as their goldens stay out of `all`.
    static func everyStage(_ stages: [Stage], _ arguments: [String]) -> [(String, Stage, [String])] {
        let size = arguments.firstIndex(of: "--size").flatMap { arguments.indices.contains($0 + 1) ? arguments[$0 + 1] : nil }
        return stages.flatMap { stage in
            (size.map { stage.sizes.contains($0) ? [$0] : [] } ?? stage.sizes).flatMap { s in
                stage.variants.map { variant in
                    let sized = size == nil && stage.sizes.count > 1 ? ["--size", s] : []
                    let title = ["\(stage.model) \(stage.name)"] + (stage.sizes.count > 1 ? [s] : []) + variant
                    return (title.joined(separator: " "), stage, sized + variant + arguments)
                }
            }
        }
    }

    /// Runs one after the other. A failure outranks a skip that comes after it: 64, then 1, then 2, then 0.
    private static func inTurn(_ runs: [(title: String, stage: Stage, arguments: [String])]) -> Int32 {
        var codes: Set<Int32> = []
        for run in runs {
            print("\n── \(run.title)")
            codes.insert(run.stage.execute(run.arguments).exitCode)
        }
        return [64, 1, 2].first(where: codes.contains) ?? 0
    }

    private static func unknown(_ what: String, among: [String]) -> Int32 {
        FileHandle.standardError.write("✗ unknown \(what): \(among.joined(separator: " · "))\n".data(using: .utf8)!)
        return 64
    }

    /// The registry for a terminal: one stage per block.
    static func table(_ stages: [Stage]) -> String {
        stages.map { s in
            """
            \(s.model) \(s.name)\(s.sizes.count > 1 || s.sizes != ["512"] ? "  [" + s.sizes.joined(separator: " · ") + "]" : "")
                \(s.proves)
                rule: \(s.rule)
                oracle: \(s.oracle ?? "none — judged against its definition")
                weights: \(s.weights.isEmpty ? "none" : s.weights.joined(separator: ", "))
            """
        }.joined(separator: "\n")
    }

    /// The registry as the verification guide's table.
    static func markdown(_ stages: [Stage]) -> String {
        var lines = ["| Stage | Proves | Rule | Oracle | Weights |", "|---|---|---|---|---|"]
        for s in stages {
            // A `|` inside a cell (`max |Δ|`) would end it.
            let cell = { (text: String) in text.replacingOccurrences(of: "|", with: "\\|") }
            lines.append("| `\(s.model) \(s.name)` | \(cell(s.proves)) | \(cell(s.rule)) | "
                         + (s.oracle.map { "`\($0)`" } ?? "—") + " | "
                         + (s.weights.isEmpty ? "—" : s.weights.joined(separator: ", ")) + " |")
        }
        return lines.joined(separator: "\n")
    }
}
