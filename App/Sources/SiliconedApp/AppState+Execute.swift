// `AppState`: a job run on the engine, and what each of its events moves.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    func execute(_ job: Job) async -> Issue {
        let settings = job.settings
        let loraEntries = settings.loras.map { LoRAEntry($0.path, strength: $0.strength) }
        do {
            try library.checkReachable()
            // The references are decoded off the main thread (3072 px at most), in their order; the
            // engine then brings each to its size: the full-size photos do not live during the render.
            let sources = job.references
            let references = try await Task.detached(priority: .userInitiated) {
                try sources.map { try $0.imageRGB() }
            }.value
            // The starting image (img2img), decoded off the main thread too, as `siliconed-dev render --image` reads it; the
            // request fills and crops it to the format at once (`Request.image`).
            let startSource = settings.strength == nil ? nil : job.start
            let start = try await Task.detached(priority: .userInitiated) { try startSource?.imageRGB() }.value
            var request = Request(settings.modelPrompt ?? settings.prompt, width: settings.format.width,
                                  height: settings.format.height, seed: job.seeds[0],
                                  steps: settings.steps == job.defaultSteps ? nil : settings.steps,
                                  loras: loraEntries, image: start, strength: settings.strength ?? Strength.defaultValue,
                                  previews: job.previews)
            request.references = references
            request.detail = settings.detail
            request.solver = settings.solver
            request.variations = settings.variations

            var progress = Engine.Progress()
            // A train: its jobs' seeds, one image each, in order.
            let together = train.count > 1
            // When the denoising under way began: its box says the seconds from there to the decoding, its DiT's plan
            // included.
            var denoisingBegan: Date?
            // The engine renders in its own process (`EngineWorker`): the model is read there, under B.
            for try await item in EngineWorker.shared.render(request, seeds: together ? train.map { $0.seeds[0] } : job.seeds,
                                                            model: settings.identifier, library: library) {
                progress.receive(item)
                measurements.receive(item)
                if progress.fraction != fraction { DockTile.show(fraction: progress.fraction, waiting: file.count) }
                fraction = progress.fraction
                if progress.estimatedRemaining != estimatedRemaining {
                    estimatedRemaining = progress.estimatedRemaining
                    remainingInstant = Date()
                }
                switch item {
                case .stage(let e, let image):
                    // The stage that was working is done; this one lights up.
                    if let previous = stage, stageStates[previous] == .active { stageStates[previous] = .done }
                    // The denoising ends where the decoding begins.
                    if e == .denoising { denoisingBegan = Date() }
                    if e == .decoding, stage == .denoising, let began = denoisingBegan {
                        stageSeconds[.denoising] = Date().timeIntervalSince(began)
                    }
                    stageStates[e] = .active
                    stage = e
                    batchImage = image
                    if let rider = turn(to: image) { signal(.stage(rider, e, image: together ? 0 : image)) }
                case let .text(_, tokenizer, encoder, _):
                    stageSeconds[.text] = tokenizer + encoder
                case .encoding(let s):
                    stageSeconds[.image] = s
                case .decoding(let s):
                    stageSeconds[.decoding] = s
                case let .step(image, index, total, _, seconds, _, _):
                    batchImage = image; currentStep = index; totalSteps = total
                    if let rider = turn(to: image) {
                        signal(.step(rider, image: together ? 0 : image, index: index, total: total, seconds: seconds))
                    }
                case .preview(let a):
                    // The image's own job: in a train, the cell it is of, whichever the running one is now.
                    let of = together && train.indices.contains(a.image) ? train[a.image] : job
                    prerenders[of.ordinal] = a.cgImage()
                case .image(let image, let renderResult):
                    stageStates[.decoding] = .done
                    guard let job = turn(to: image) else { break }
                    // The PNG and the thumbnail are made off the main thread; the engine has already
                    // moved on to the next image of the batch.
                    // A job `silicontrol add "@name"` translated: the PNG keeps the prompt typed, the model's beside it.
                    // Any other job (the rack's, which reads what it holds): the engine's metadata, untouched.
                    let named = settings.modelPrompt.map {
                        NamedSubjects.metadata(renderResult.metadata, typed: settings.prompt, modelPrompt: $0)
                    }
                    let finished = try await Task.detached(priority: .userInitiated) {
                        try FinishedImage(renderResult, metadata: named)
                    }.value
                    lastImage += 1
                    let entry = Entry(ordinal: lastImage, job: job.ordinal, png: finished.png, vignette: finished.vignette,
                                        width: renderResult.image.width, height: renderResult.image.height,
                                        seed: renderResult.seed, timings: renderResult.timings, footprints: renderResult.footprints,
                                        evaluations: renderResult.evaluations,
                                        settings: settings,
                                        metadata: named ?? renderResult.metadata, references: job.references,
                                        start: start == nil ? nil : job.start, owner: job.owner)
                    admit(entry, image: finished.image, shows: followsRender)
                    prerenders[job.ordinal] = nil
                    sessionImages += 1
                    sessionSeconds += renderResult.timings.total
                    jobImages.append(entry.ordinal)
                    signal(.image(job, entry))
                    // In a train, a job ends with its image: the call goes on for the others.
                    if together {
                        delivered.insert(job.id)
                        complete(job, .finished, images: [entry.ordinal], warnings: journal)
                    }
                case let .unusable(image, lost):
                    // In a train, a job whose image is lost ends with it: the call goes on for the others. Alone, its
                    // call ends in `nonFiniteImage`, below.
                    guard together, let job = turn(to: image) else { break }
                    delivered.insert(job.id)
                    let p = Problem(EngineError.nonFiniteImage(stage: lost))
                    problem = p
                    complete(job, .failed(p), warnings: journal)
                case .warning(let message):
                    journal.append(message)
                default:
                    break
                }
            }
            // What this render teaches about speed: its own pace (the first render on a Mac finds the Neural Engine's share
            // in flight, its pace within 3 % of the next, measured 20/11 on the mini).
            if var v = measurements.speed {
                // A cached prompt teaches nothing about the encoder: the cost learned before stays.
                if v.text == nil { v.text = speeds[job.speedKey]?.text }
                speeds[job.speedKey] = v
            }
            // Cancelled, the `Task` leaves the loop WITHOUT an error (AsyncThrowingStream).
            if Task.isCancelled {
                for (s, v) in stageStates where v == .active { stageStates[s] = .idle }
                flash(String(localized: "Render stopped"))
                return .stopped
            }
            return .finished
        } catch {
            // Every refusal of the engine is one `EngineError` case (`EngineError(_:)` folds the rest).
            let e = EngineError(error)
            if let s = stage, stageStates[s] == .active { stageStates[s] = e == .cancelled ? .idle : .failed }
            if e == .cancelled {
                flash(String(localized: "Render stopped"))
                return .stopped
            }
            let p = Problem(e)
            problem = p
            return .failed(p)
        }
    }
}
