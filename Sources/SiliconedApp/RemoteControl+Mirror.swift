// **`silicontrol` follows the switch** (Render › Compute On): it talks only to this Mac's socket, never
// to the network, but once the window shows another Mac, the commands act on what it shows — the
// other Mac's queue and history (`AppState+Mirror.swift`), as the buttons do. Split from `RemoteControl.swift`.
//
// - `add` is asked of the other Mac, which numbers the jobs and answers each (`RemoteAnswer`): the
//   answer comes before the verdict, so a refusal is the command's own. An edit with no format typed
//   leaves its size to the other Mac (image 1's, by its denoiser's rule).
// - `wait` and `follow` read the signals the mirror gives from what the other Mac publishes.
// - What acts on this Mac — `install`, `diagnose`, `set` — refuses: the window says where it computes.
//   `grid`, `vary`, `enlarge` and `subjects` refuse too, for now.

import AppKit
import Foundation
import Siliconed

extension RemoteControl {
    /// The commands a mirror does not take, with the sentence that says why.
    func refuseOnMirror(_ command: String) throws {
        guard let mirror = app.mirror else { return }
        let name = mirror.macName
        switch command {
        case "install", "diagnose", "set":
            throw Refusal("remote", String(localized: "\(command) acts on this Mac, and the window computes on “\(name)”"),
                          help: String(localized: "Render › Compute On › This Mac, then retry"))
        case "grid", "vary", "enlarge", "subjects":
            throw Refusal("remote", String(localized: "\(command) does not work on another Mac yet: the window computes on “\(name)”"),
                          help: String(localized: "Render › Compute On › This Mac, then retry"))
        default:
            break
        }
    }

    /// **`models` on a mirror**: the other Mac's ready models, as it published them (`MirrorCatalog`).
    func mirroredModels() -> [String: Any] {
        let catalog = app.mirrorCatalog
        let list = app.visibleCards.map { f -> [String: Any] in
            var o: [String: Any] = ["id": f.id, "name": f.name, "default_steps": f.defaultSteps,
                                    "formats": f.formats.map { "\($0.width)x\($0.height)" },
                                    "commercial_license": f.license.commercial, "license": f.license.text,
                                    "default": f.id == "z-image", "ready": app.readySet.contains(f.id)]
            if let t = catalog?.traits[f.id] {
                o["edit"] = t.maxReferences > 0
                o["max_references"] = t.maxReferences
                o["loras"] = (catalog?.loras[f.id] ?? []).map { l in
                    ["lora": Self.shortName(l.path), "name": catalog?.loraNames[l.path] ?? Self.shortName(l.path)]
                }
            }
            return o
        }
        return ["ok": true, "models": list, "computing_on": app.mirror?.macName ?? NSNull()]
    }

    /// **`add` on a mirror**: the request read as `add` reads it, judged with what the other Mac
    /// published, then asked of it — one job per alternative of the prompt, at the same seeds.
    func addOnMirror(_ mirror: MirrorLink, _ arguments: [String], cwd: String, _ c: Connection) throws {
        let d = try parse(arguments)
        if d.sketch || d.x != nil || d.y != nil { try refuseOnMirror("grid") }
        guard app.readySet.contains(d.model), let card = app.cards.first(where: { $0.id == d.model }) else {
            let ready = app.visibleReady.map(\.id).joined(separator: ", ")
            throw Refusal("model", String(localized: "model \"\(d.model)\" is not ready on “\(mirror.macName)”"),
                          help: String(localized: "ready: \(ready) — silicontrol models"))
        }
        let traits = app.mirrorCatalog?.traits[d.model]
        guard let prompt = d.prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Refusal("usage", String(localized: "empty prompt"))
        }
        let alternatives: PromptAlternatives
        do { alternatives = try PromptAlternatives(prompt) } catch {
            let p = Problem(error)
            throw Refusal("prompt", p.title, help: p.suggestion)
        }

        // The references: files here, or images of the other Mac's history (`i7`), sent by their ordinal.
        var references: [ReferenceImage] = []
        if !d.references.isEmpty {
            let max = traits?.maxReferences ?? 0
            guard max > 0 else {
                throw Refusal("edit", String(localized: "\(card.name) does no editing by reference (--ref)"),
                              help: String(localized: "models that do have \"edit\": true in \"silicontrol models\""))
            }
            guard d.references.count <= max else {
                let n = d.references.count
                throw Refusal("edit", String(localized: "\(n) references: \(card.name) reads \(max) at most"),
                              help: String(localized: "\"max_references\" in \"silicontrol models\""))
            }
            references = try d.references.map { r -> ReferenceImage in
                if let entry = image(r), let reference = app.fromHistory(entry) { return reference }
                let url = URL(fileURLWithPath: absolute(r, cwd))
                guard let vignette = NSImage(contentsOf: url), vignette.isValid, let size = ReferenceImage.size(of: url) else {
                    let path = url.path
                    throw Refusal("ref", String(localized: "image unreadable or missing: \(path)"))
                }
                return ReferenceImage(source: .file(url), vignette: vignette, name: url.lastPathComponent,
                                      width: size.width, height: size.height)
            }
        }
        // An edit with no format typed: image 1's, derived on the other Mac.
        let typed = d.format.map { RecommendedFormat($0.0, $0.1) }
        let format = typed ?? RecommendedFormat(1024, 1024)
        if typed != nil || references.isEmpty {
            do { try Format.check(width: format.width, height: format.height, family: card.family) } catch {
                let p = Problem(error)
                throw Refusal("format", p.title, help: p.suggestion)
            }
        }
        let steps = d.steps ?? card.defaultSteps
        if let t = traits, !t.steps.contains(steps) {
            let accepted = t.steps.map(String.init).formatted(.list(type: .or))
            throw Refusal("steps", String(localized: "This model has no schedule for \(String(steps)) steps: \(accepted)."),
                          help: "--steps \(card.defaultSteps) (the model's \"default_steps\")")
        }
        guard d.loras.count <= AppState.maxLoRA else {
            let n = d.loras.count, max = AppState.maxLoRA
            throw Refusal("lora", String(localized: "\(n) LoRAs: the app stacks \(max) at most"))
        }
        let forged = app.mirrorCatalog?.loras[d.model] ?? []
        let stack = try d.loras.map { entry -> LoRASlot in
            let names = app.mirrorCatalog?.loraNames ?? [:]
            guard let l = forged.first(where: { Self.shortName($0.path) == entry.name || $0.path == entry.name || names[$0.path] == entry.name }) else {
                let name = entry.name, list = forged.map { Self.shortName($0.path) }.joined(separator: ", ")
                throw Refusal("lora", String(localized: "LoRA \"\(name)\" not found for \(d.model)"),
                              help: forged.isEmpty ? String(localized: "no LoRA installed for this model")
                                                   : String(localized: "compatible: \(list)"))
            }
            return LoRASlot(path: l.path, strength: entry.strength)
        }
        var folder: String?
        if let raw = d.folder {
            folder = absolute(raw, cwd)
            try createFolder(folder!)
        }

        let firstNumber = d.seed ?? UInt64.random(in: 0...UInt64(UInt32.max))
        let variants = alternatives.variants
        let jobs = variants.map { v -> Job in
            let settings = RenderSettings(identifier: card.id, modelName: card.name, prompt: v.prompt, format: format,
                                          steps: steps, loras: stack, references: references.count, detail: d.detail,
                                          solver: d.solver, schedule: d.schedule)
            var job = Job(ordinal: 0, settings: settings, seeds: (0..<d.batch).map { firstNumber &+ UInt64($0) },
                          references: references, previews: d.previews ?? app.previews, defaultSteps: card.defaultSteps, plan: nil)
            job.neuralEngine = d.neuralEngine
            return job
        }

        // Each job is answered by the other Mac; the command answers once all are.
        var answers: [RemoteAnswer?] = Array(repeating: nil, count: jobs.count)
        func answered() {
            guard answers.allSatisfy({ $0 != nil }) else { return }
            let refused = answers.compactMap { a -> String? in if case .refused(let t)? = a { t } else { nil } }
            if let first = refused.first {
                finish(c, Response.failure("refused", first))
                return
            }
            let added = zip(answers, variants).map { a, v -> [String: Any] in
                guard case let .queued(o, estimate, _, _)? = a else { return [:] }
                var j = mirror.jobs[o].map { json($0) } ?? ["job": "t\(o)"]
                j["estimate_s"] = estimate.map { Response.number($0, 1) } ?? NSNull()
                if alternatives.hasAlternatives { j["choices"] = v.choices }
                j["computing_on"] = mirror.macName
                return j
            }
            let ordinals = answers.compactMap { a -> Int? in if case let .queued(o, _, _, _)? = a { o } else { nil } }
            if d.wait {
                for var o in added { o["type"] = "added"; emit(c, o) }
                followToEnd(ordinals, folder: folder, c)
            } else if added.count == 1 {
                var o = added[0]
                o["ok"] = true
                finish(c, o)
            } else {
                finish(c, ["ok": true, "jobs": added, "computing_on": mirror.macName])
            }
        }
        for (i, job) in jobs.enumerated() {
            do {
                try mirror.render(job, format: typed != nil || references.isEmpty) { a in
                    answers[i] = a
                    answered()
                }
            } catch {
                throw Refusal("ref", Problem(error).text)
            }
        }
    }
}
