import Foundation
import Siliconed

// **A stage: one thing the engine computes, checked against the reference.** The registry
// (`Registry.swift`) lists them all; the help, `check list`, the verification guide's table and the
// maintainer's closing run are read from it, so that a stage cannot exist in one and not the others.

package struct Stage: Sendable {
    /// The app's model identifier of a family (`z-image`, `moss-image`…).
    package let model: String
    /// `text`, `dit`, `trajectory`, `vae`… the same word for the same stage of every model.
    package let name: String
    /// What a pass proves, in one line.
    package let proves: String
    /// The sizes the oracle writes goldens for; the first is the default.
    package let sizes: [String]
    /// Options beyond `Invocation.common`.
    package let options: Set<String>
    /// The argument sets `check <model>` runs at each size, one per golden the oracle's `all` writes
    /// (Qwen-Image-2.1's `--edit 1`, `--edit 2`); `[[]]`, the stage as it stands.
    package let variants: [[String]]
    /// The script that writes the stage's goldens and its arguments, `nil` for a stage judged against
    /// a definition (a bit-exact identity).
    package let oracle: String?
    /// What the stage loads from the installed library, `[]` for none.
    package let weights: [String]
    /// How the stage is judged, one line: `Judge` rule with its floor, bit-exact, a bound.
    package let rule: String
    /// The stage's run: it prints its measures and returns the report holding them.
    package let run: @Sendable (Invocation) throws -> Report

    package init(model: String, name: String, proves: String, sizes: [String] = ["512"], options: Set<String> = [],
                 variants: [[String]] = [[]], oracle: String?, weights: [String], rule: String, run: @escaping @Sendable (Invocation) throws -> Report) {
        self.model = model
        self.name = name
        self.proves = proves
        self.sizes = sizes
        self.options = options
        self.variants = variants
        self.oracle = oracle
        self.weights = weights
        self.rule = rule
        self.run = run
    }

    /// The stages whose own subject is memory: they watch the engine's stages themselves.
    static let judgesMemory: Set<String> = ["memory", "writer-memory", "vae-bands", "pace"]

    /// `siliconed check <model> <stage>`.
    package var command: String { "siliconed check \(model) \(name)" }

    /// Runs the stage on `arguments` and returns how it ended; the verdict line is the last printed.
    package func execute(_ arguments: [String]) -> Outcome {
        let title = "\(model) \(name)"
        do {
            let invocation = try Invocation(arguments, accepted: options, defaultSize: sizes.first ?? "512")
            // The memory rule, on every stage: what each engine stage it opens holds against its P, and no swap. The
            // stages that judge memory themselves watch on their own.
            let watch = Self.judgesMemory.contains(name) ? nil : StageWatch()
            let result = Result { try run(invocation) }
            watch?.stop()
            let report = try result.get()
            let heard = watch?.heard ?? []
            // Said, not judged: what the Neural Engine computed of the engine stages that said it.
            let shared = heard.filter { $0.neuralEngine != nil }
            if !shared.isEmpty {
                print("neural engine · " + shared.map { String(format: "%@ %.0f %%", $0.name, 100 * $0.neuralEngine!) }.joined(separator: ", "))
            }
            for measure in MemoryStages.rule(heard) { report.add(measure) }
            return report.verdict()
        } catch let skip as Skip {
            print("— \(title) · skipped: \(skip.reason)")
            return .skipped(skip.reason)
        } catch let usage as UsageError {
            FileHandle.standardError.write("✗ \(usage.message)\n".data(using: .utf8)!)
            exit(64)
        } catch {
            print("✗ \(title) · \(error)")
            return .failed
        }
    }
}
