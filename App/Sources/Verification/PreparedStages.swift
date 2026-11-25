import Foundation
import Siliconed
import SiliCore

// **A render prepared compiles nothing** — `siliconed check <model> prepared [--size 1024]`.
//
// What this process specialized erased (`Functions.eraseSpecializations`: the command line's alone, never the app's
// engine's), the render's preparation (`Engine.prepare`) specialized function by function, then the render itself: each
// of its stages must say no second of Core AI's compilation (`Engine.Held.compilation`). What a stage still compiles is a
// shape the preparation did not foresee — a share move, a palier, a decoder's band.
//
//     the preparation   its specializations, the ANE's first, and their seconds
//     each stage        0 s compiled at the render                       its function was prepared
enum PreparedStages {
    static let all: [Stage] = ModelCard.allCards.map(stage)

    static func stage(_ card: ModelCard) -> Stage {
        Stage(
            model: card.id, name: "prepared",
            proves: "a render prepared ahead compiles nothing: this process's specializations erased, the render's preparation specialized, then the render's stages each at 0 s of Core AI compilation",
            sizes: ["512", "1024"], oracle: nil, weights: ["text encoder", "DiT", "VAE"],
            rule: "every stage of the render compiles 0 s once its preparation is specialized"
        ) { invocation in
            let report = Report(title: "\(card.id) prepared \(invocation.size)")
            let model = try Model.named(card.id, in: Library.processWide)
            let request = Request(MemoryStages.prompt, width: invocation.width, height: invocation.height)
            Functions.eraseSpecializations(of: [ProcessInfo.processInfo.processName])
            let began = Date()
            let all = try Engine().prepare(request, model: model)
            let planned = Date().timeIntervalSince(began)
            let units = all.map { $0.unit == .neuralEngine ? "ane" : "gpu" }
            for s in all { try blocking { try await Functions.specialize(s) } }
            let specialized = Date().timeIntervalSince(began) - planned
            print(String(format: "prepared: %d of the ANE's, %d of the GPU's; planned in %.1f s, specialized in %.1f s",
                         units.filter { $0 == "ane" }.count, units.filter { $0 == "gpu" }.count, planned, specialized))
            report.add(Judge.holds("the ANE's first", units == units.sorted(), note: units.joined(separator: " ")))
            let render = try Engine().render(request, model: model)
            print(String(format: "render: %.1f s", render.timings.total))
            for h in render.footprints.stages {
                report.add(Judge.bound("\(h.stage.rawValue) compiled", h.compilation, 0, unit: "s"))
            }
            return report
        }
    }
}
