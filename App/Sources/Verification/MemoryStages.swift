import CryptoKit
import Foundation
import Siliconed
import SiliCore
import Synchronization

// **The memory of a render, stage by stage** — `siliconed check <model> memory [--size 1024]`.
//
// A render is a few of the engine's stages, one after the other: the text, [the image,] the DiT, the decoder. The
// engine plans each under B before it allocates (P = the floor + the arena, `SiliCore.Stage.plan`) and reads Core AI's
// share while it runs, keeping weights in memory while B holds them with a margin beside (`Stage.signal`); this stage
// renders one picture and reads what the process really held while each lived (`StageWatch`).
//
//     a stage      the footprint at its most           what the stage kept in flight left the margin it counts
//                  ≤ B − the margin (64 MiB)
//     the render   the footprint at its most ≤ B       the budget, on this Mac
//     the swap     no page swapped out                 a render under B that swaps is not under B
//
// No reference: the stage is judged against the plan's own word.

enum MemoryStages {
    static let all: [Stage] = ModelCard.allCards.map(stage) + Family.allCases.filter { $0.traits.writer != nil }.map(writer)

    /// The prompt of every memory check.
    static let prompt = "a lighthouse on a cliff at dusk, waves breaking on the rocks below, a warm light in its window"

    static func stage(_ card: ModelCard) -> Stage {
        Stage(
            model: card.id, name: "memory",
            proves: "one render's memory, stage by stage (text, DiT, decoder): the footprint at its most while each lived against B less the margin its stage keeps, the process's footprint against B, no page swapped out",
            sizes: ["512", "1024"], oracle: nil, weights: ["text encoder", "DiT", "VAE"],
            rule: "each stage's footprint stays under B less 64 MiB; the process's under B (¾ · RAM − 2 GiB); no swap-out"
        ) { invocation in
            let report = Report(title: "\(card.id) memory \(invocation.size)")
            let model = try Model.named(card.id, in: Library.processWide)
            let asked = ProcessInfo.processInfo.environment["SILICONED_MEMORY_PROMPT"]
            let request = Request(asked ?? prompt, width: invocation.width, height: invocation.height)
            let watch = StageWatch()
            let render: Engine.Render
            do { render = try Engine().render(request, model: model) } catch { watch.stop(); throw error }
            watch.stop()
            // The picture's digest: two renders of one plan give the same one.
            let md5 = render.image.pixels.withUnsafeBytes { Insecure.MD5.hash(data: $0) }.map { String(format: "%02x", $0) }.joined()
            print("image md5 \(md5.prefix(12))")
            let t = render.timings
            print(String(format: "timings: text %.1f s, denoising %.1f s, decoding %.1f s, total %.1f s; ANE %.0f %%",
                         t.text, t.denoising, t.decoding, t.total, 100 * (render.footprints.neuralEngine ?? 0)))
            // What Core AI specialized for the render, stage by stage: the first render of a plan's functions pays it.
            print("compiled: " + render.footprints.stages.map { "\($0.stage.rawValue) \(String(format: "%.1f", $0.compilation)) s" }.joined(separator: ", "))
            flight(watch.heard)
            judge(watch.heard, report)
            return report
        }
    }

    /// **The prompt writer's memory**, the same judgment on one writing: a rewrite, or with `--describe <image>` a
    /// description, under the product's own limits (the caches are planned for the longest answer).
    static func writer(_ family: Family) -> Stage {
        Stage(
            model: family.rawValue, name: "writer-memory",
            proves: "one writing's memory (a rewrite; with --describe <image>, a description): the footprint at its most while each of its stages lived against B less the margin, the process's footprint against B, no page swapped out",
            options: ["--describe"], oracle: nil, weights: ["text encoder"],
            rule: "each stage's footprint stays under B less 64 MiB; the process's under B (¾ · RAM − 2 GiB); no swap-out"
        ) { invocation in
            let report = Report(title: "\(family.rawValue) writer-memory")
            guard let writer = PromptWriter(library: Library.processWide) else { throw Skip("no writer installed") }
            let request: PromptWriter.Request
            if let path = invocation.extras["--describe"] {
                let image = try PromptWriterImage.rgb8(from: Data(contentsOf: URL(fileURLWithPath: path)))
                request = .describe(image: image, instruction: PromptWriterInstructions.instruction(for: family, .describe), ask: "")
            } else {
                request = .rewrite(text: prompt, instruction: PromptWriterInstructions.instruction(for: family, .rewrite))
            }
            let watch = StageWatch()
            let stream = writer.write(request)
            let failure = Mutex<Error?>(nil), tokens = Mutex(0), done = DispatchSemaphore(value: 0)
            Task.detached {
                do { for try await event in stream { if case .finished(_, let n) = event { tokens.withLock { $0 = n } } } }
                catch { failure.withLock { $0 = error } }
                done.signal()
            }
            done.wait()
            watch.stop()
            if let error = failure.withLock({ $0 }) { throw error }
            print("wrote \(tokens.withLock { $0 }) tokens")
            judge(watch.heard, report)
            return report
        }
    }

    static func gib(_ bytes: Int) -> Double { Double(bytes) / Double(1 << 30) }

    /// **The ANE in flight, cycle by cycle**, for each engine stage whose block could share its rows: the ANE's rows of
    /// each cycle and its seconds, what its start loaded (a share moved, taller calls), and the unit decided.
    static func flight(_ heard: [StageWatch.Entry]) {
        for e in heard {
            guard let first = e.cycles.first, first.shareable > 0 else { continue }
            let cycles = e.cycles.map { c in
                String(format: "%d %.2f s", c.rows, c.seconds)
                    + (c.moved.map { String(format: " (moved, %.2f s)", $0) } ?? "")
                    + (c.climbed.map { String(format: " (climbed, %.2f s)", $0) } ?? "")
                    + (c.calls.map { String(format: " [ANE %.2f, GPU beside %.2f, alone %.2f]", $0.ane, $0.beside, $0.alone) } ?? "")
            }
            print("flight \(e.name): the ANE's rows of \(first.shareable) by \(first.unit), each cycle: " + cycles.joined(separator: ", "))
            if let w = e.weighed {
                let costs = w.seconds.keys.sorted(by: >).map { String(format: "%d: %.2f s at %d rows", $0, w.seconds[$0]!, w.rows[$0] ?? 0) }.joined(separator: ", ")
                print("flight \(e.name): unit decided, a cycle at each unit's best share: \(costs.isEmpty ? "none, a single unit" : costs); kept \(w.kept)")
            }
        }
    }

    /// What an engine stage is held to: its B less the margin its stage keeps (`Stage.margin`). P, the socle it allocated,
    /// is said beside: what it kept in flight past P is the stage's to decide under B (`Stage.signal`).
    static func bar(_ e: StageWatch.Entry) -> (bytes: Int, unit: String) { (e.budget - SiliCore.Stage.margin, "GiB, B − 64 MiB") }

    /// Each engine stage's footprint at its most against its B less the margin, to the MiB, what its plan counted
    /// said beside; then the process against B and the swap. Nothing heard: the stage opened no engine stage, and says so.
    static func judge(_ heard: [StageWatch.Entry], _ report: Report) {
        guard !heard.isEmpty else { return report.add(Judge.holds("an engine stage", false, note: "the render opened none")) }
        for e in heard {
            let note = String(format: "%+d MiB; P %.2f GiB over a floor of %.2f; %@; %.1f s",
                              (e.peak - bar(e).bytes) / (1 << 20), gib(e.p), gib(e.floor),
                              e.ring > 0 ? "a ring of \(e.ring), \(e.kept) groups kept" : "every group kept",
                              e.seconds)
            report.add(Judge.bound(e.name, gib(e.peak), gib(bar(e).bytes), unit: bar(e).unit, note: note))
        }
        report.add(Judge.bound("the process", gib(heard.map(\.peak).max() ?? 0), gib(MemoryBudget.machineCeiling()), unit: "GiB, B"))
        report.add(swap(heard))
    }

    /// No page swapped out while the stages lived (the Mac's counter: nothing else computes under the token).
    static func swap(_ heard: [StageWatch.Entry]) -> Measure {
        Judge.bound("swap", Double(heard.reduce(0) { $0 + $1.swapouts }), 0, unit: "pages swapped out")
    }

    /// **The memory rule on any stage's engine stages** (`Stage.execute`): each holds no more than its B less the margin,
    /// whatever the control proves — a stage past it is found by the first control that opens it, not by the one written
    /// to look. Two lines: the engine stage nearest its bar (`bar`) or furthest past it, and the swap. Nothing where the control
    /// opened no engine stage.
    static func rule(_ heard: [StageWatch.Entry]) -> [Measure] {
        guard let worst = heard.max(by: { $0.peak - bar($0).bytes < $1.peak - bar($1).bytes }) else { return [] }
        let note = String(format: "%@, %+d MiB; %d engine stage%@", worst.name, (worst.peak - bar(worst).bytes) / (1 << 20),
                          heard.count, heard.count == 1 ? "" : "s")
        return [Judge.bound("memory", gib(worst.peak), gib(bar(worst).bytes), unit: bar(worst).unit, note: note), swap(heard)]
    }
}
