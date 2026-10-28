// **The other Mac's state, shown here** — the borrowing side of the remote Mac (`OtherMacs.swift`). A
// second `AppState`, `init(mirroring:)`, beside this Mac's own, which it never touches: back in local,
// everything is where it was.
//
//     the mini publishes (`Mirror.swift`)        the mirror keeps
//       catalog        ready models, traits, B     readySet, compatibleLoras, missing, memoryCeiling
//       queue          job running, waiting        currentJob, file, fraction, stage, steps, problem
//       entry + PNG    a history image             history (the thumbnail made here, from the PNG)
//       preview        a small PNG per step        preview
//
// - **It computes nothing**: its derived values (`busy`, `canRender`, `memoryRefusal`…) are the same
//   code reading what was published — B is the mini's, and the mini checks every job again before
//   queueing it.
// - **Its actions are sent** (`enqueue`, `cancel`, `remove`, `stopAll`, `clearQueue`, `delete`: each
//   starts with `if let mirror`). A job keeps the form's every choice, its seeds included: the mini
//   renders the same bits as if it had been asked on its own screen.
// - **Ordinals are the identities** (`t3`, `i7`), as for `silicontrol`: a job or an image keeps its
//   `UUID` here from one snapshot to the next, so that the selection holds.
// - Not yet: edits with references (`maxReferences` is 0), grids and explorations, variations,
//   enlarging, the statistics, the license sheet, ⌘Z after a removal.

import AppKit
import Foundation
import ImageIO
import Siliconed

/// What the mirror sends through, and what it remembers between two snapshots.
@MainActor
final class MirrorLink {
    let macName: String
    private let sender: (SessionMessage) -> Void
    /// Jobs and images already built here, by ordinal: their `UUID` stays.
    var jobs: [Int: Job] = [:]
    var entries: [Int: Entry] = [:]
    /// An entry waits for its PNG, the blob that follows it.
    var pending: [UInt32: RemoteEntry] = [:]
    var previewBlob: UInt32?

    init(macName: String, send: @escaping (SessionMessage) -> Void) {
        self.macName = macName; self.sender = send
    }

    func send(_ m: SessionMessage) { sender(m) }

    func render(_ job: Job) { sender(.render(MirrorProjection.remote(job))) }
}

/// Between the app's types and what crosses (`Mirror.swift`), both ways.
enum MirrorProjection {
    static func remote(_ j: Job) -> RemoteJob {
        remote(j.settings, ordinal: j.ordinal, seeds: j.seeds, previews: j.previews, defaultSteps: j.defaultSteps,
               neuralEngine: j.neuralEngine)
    }

    static func remote(_ s: RenderSettings, ordinal: Int, seeds: [UInt64], previews: Bool, defaultSteps: Int?,
                       neuralEngine: Bool?) -> RemoteJob {
        RemoteJob(ordinal: ordinal, model: s.identifier, modelName: s.modelName, prompt: s.prompt,
                  width: s.format.width, height: s.format.height, steps: s.steps, defaultSteps: defaultSteps,
                  loras: s.loras.map { RemoteLoRA(path: $0.path, strength: $0.strength) }, seeds: seeds,
                  detail: s.detail, solver: s.solver, schedule: s.schedule, previews: previews, neuralEngine: neuralEngine)
    }

    static func settings(_ r: RemoteJob) -> RenderSettings {
        var s = RenderSettings(identifier: r.model, modelName: r.modelName, prompt: r.prompt,
                               format: RecommendedFormat(r.width, r.height), steps: r.steps,
                               loras: r.loras.map { LoRASlot(path: $0.path, strength: $0.strength) },
                               detail: r.detail, solver: r.solver, schedule: r.schedule)
        s.neuralEngine = r.neuralEngine ?? false
        return s
    }

    static func job(_ r: RemoteJob, ordinal: Int, plan: DenoisingPlan? = nil) -> Job {
        var j = Job(ordinal: ordinal, settings: settings(r), seeds: r.seeds, references: [], previews: r.previews,
                    defaultSteps: r.defaultSteps, plan: plan)
        j.neuralEngine = r.neuralEngine
        return j
    }

    static let stageStates: [StageState: String] = [.idle: "idle", .active: "active", .done: "done", .failed: "failed"]
}

extension AppState {
    /// The chosen model's traits, as the other Mac published them.
    var traits: RemoteTraits? { mirrorCatalog?.traits[identifier] }

    /// The rack's three module names: from the chain here, from the traits in a mirror.
    var moduleNames: (text: String?, encoding: String?, decoding: String?) {
        if mirror != nil { return (traits?.text, traits?.encoding, traits?.decoding) }
        return (chain?.text.name, chain?.encoding?.name, chain?.decoding.name)
    }

    func readMirroredCatalog() {
        guard let c = mirrorCatalog else { compatibleLoras = []; missing = nil; return }
        compatibleLoras = (c.loras[identifier] ?? []).compactMap { l in
            LoRACard(header: ["kind": "lora", "cible": ["modele": identifier], "nom": c.loraNames[l.path] ?? l.path],
                     path: l.path)
        }
        missing = readySet.contains(identifier) ? nil : String(localized: "Not installed on \(mirror?.macName ?? "")")
    }

    // MARK: - What the other Mac publishes

    func receive(_ m: SessionMessage) {
        guard let link = mirror else { return }
        switch m {
        case .catalog(let c):
            mirrorCatalog = c
            readySet = Set(c.ready)
            memoryCeiling = c.ceiling
            // Not `fasterRendering`: its setter writes this Mac's preference and the engine's switch.
            settingsRead = c.settingsRead
            if !isVisible(identifier) || !readySet.contains(identifier),
               let first = visibleReady.first?.id { identifier = first }
            readMirroredCatalog()
        case .queue(let q):
            func job(_ r: RemoteJob) -> Job {
                if let j = link.jobs[r.ordinal] { return j }
                let j = MirrorProjection.job(r, ordinal: r.ordinal)
                link.jobs[r.ordinal] = j
                return j
            }
            let running = q.current.map(job)
            if running?.ordinal != currentJob?.ordinal { batchSize = running?.seeds.count ?? 1 }
            currentJob = running
            file = q.waiting.map(job)
            awaitingLicense = q.awaitingLicense.map(job)
            let kept = Set(([q.current].compactMap { $0 } + q.waiting + q.awaitingLicense).map(\.ordinal))
            link.jobs = link.jobs.filter { kept.contains($0.key) }
            fraction = q.fraction
            stage = q.stage.flatMap(Engine.Stage.init(rawValue:))
            currentStep = q.currentStep; totalSteps = q.totalSteps
            batchImage = q.batchImage; batchSize = q.batchSize
            let states = Dictionary(uniqueKeysWithValues: MirrorProjection.stageStates.map { ($1, $0) })
            stageStates = Dictionary(uniqueKeysWithValues: q.stageStates.compactMap { k, v in
                Engine.Stage(rawValue: k).flatMap { s in states[v].map { (s, $0) } }
            })
            stageSeconds = Dictionary(uniqueKeysWithValues: q.stageSeconds.compactMap { k, v in Engine.Stage(rawValue: k).map { ($0, v) } })
            stageModel = q.current?.model
            estimatedRemaining = q.remaining
            remainingInstant = Date()
            problem = q.problem.map { Problem($0) }
            if running == nil { preview = nil }
        case .entry(let e):
            link.pending[e.png] = e
        case .removed(let ordinals):
            let gone = Set(ordinals)
            let ids = Set(history.filter { gone.contains($0.ordinal) }.map(\.id))
            guard !ids.isEmpty else { return }
            let first = history.firstIndex { ids.contains($0.id) } ?? 0
            history.removeAll { ids.contains($0.id) }
            for o in ordinals { link.entries[o] = nil }
            marked.subtract(ids)
            if let s = selection, ids.contains(s) {
                selection = history.isEmpty ? nil : history[min(first, history.count - 1)].id
                anchor = selection
                loadSelectedImage()
            }
        case .preview(let id):
            link.previewBlob = id
            if id == nil { preview = nil }
        case .refused(let text):
            problem = Problem(text)
            flash(text)
        default:
            break
        }
    }

    func receive(blob id: UInt32, _ data: Data) {
        guard let link = mirror else { return }
        if id == link.previewBlob {
            preview = Self.decode(data)
            return
        }
        guard let e = link.pending.removeValue(forKey: id), link.entries[e.ordinal] == nil,
              let image = Self.decode(data) else { return }
        var settings = MirrorProjection.settings(e.job)
        settings.neuralEngine = e.neuralEngine
        let entry = Entry(ordinal: e.ordinal, job: e.job.ordinal, png: data,
                          vignette: FinishedImage.reduce(image, side: 240) ?? image, width: e.width, height: e.height,
                          seed: e.seed, timings: e.timings, footprints: e.footprints, evaluations: e.evaluations,
                          settings: settings, metadata: e.metadata)
        link.entries[e.ordinal] = entry
        // The history is newest first; entries arrive in their order, the first connection's oldest first.
        let at = history.firstIndex { $0.ordinal < e.ordinal } ?? history.count
        history.insert(entry, at: at)
        lastImage = max(lastImage, e.ordinal)
        if at == 0, followsRender {
            selectedImage = (entry.id, image)
            selection = entry.id
            marked = []; anchor = entry.id
        }
    }

    nonisolated static func decode(_ png: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(png as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}

extension AppState {
    /// The session with the other Mac changed: said where the problems are said; joined again, the
    /// problem goes.
    func linkChanged(_ s: RemoteLink.State, name: String) {
        switch s {
        case .connected: if problem?.code == "remote" { problem = nil }
        case .connecting, .off: break
        case .waiting: problem = { var p = Problem(String(localized: "Lost “\(name)”: trying again…")); p.code = "remote"; return p }()
        case .refused: problem = { var p = Problem(OtherMacsSection.state(s, found: true)); p.code = "remote"; return p }()
        }
    }
}
