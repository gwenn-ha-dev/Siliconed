// A render's figures, drawn from its events (`Measures`), and the speed learned from them (`Speed`).

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

/// **The learned speed** of a model at a format, drawn from a finished render: each stage in
/// seconds, `full` the average of a DiT evaluation.
struct Speed {
    var full: Double
    /// `nil` while the text encoder has only been read back from the cache: ~0 s for that prompt,
    /// nothing learned for the next one.
    var text: Double?
    var image: Double
    var decoding: Double

    /// One key per number of references: a speed learned with one reference says nothing of three
    /// (the sequence, the encoder's images, the condition cache all grow with each).
    static func key(_ identifier: String, _ format: RecommendedFormat, references: Int, lora: Bool) -> String {
        "\(identifier)|\(format.width)×\(format.height)\(references > 0 ? "|edit\(references)" : "")\(lora ? "|LoRA" : "")"
    }

    /// The duration of a render of `images` images on this plan. `image` (encoding the references)
    /// was measured with the same number of references: the key says so.
    func duration(_ plan: DenoisingPlan, images: Int) -> Double {
        let perImage = Double(plan.evaluations) * full + decoding
        return (text ?? 0) + image + Double(images) * perImage
    }
}

/// **The figures of a render**, drawn from its events — what the Chain inspector
/// shows. In memory, like the rest.
struct Measures {
    private(set) var job: Job?
    private(set) var begin: Date?
    private(set) var plan: Engine.Plan?
    private(set) var tokens: Int?
    private(set) var tokenizer: Double?, encoder: Double?
    private(set) var encoding: Double?
    /// The conditioning came from `RenderCache`: its time is not the encoder's cost.
    private(set) var textCached = false
    /// Per image of the batch, the duration of each DiT evaluation.
    private(set) var stepSeconds: [[Double]] = []
    private(set) var decodings: [Double] = []
    /// The timers and footprints of the last finished image.
    private(set) var timings: Engine.Timings?
    private(set) var footprints: Engine.Footprints?
    /// What the stages held so far, as the engine's ledger read it (`Engine.Event.held`); the last image's once it is out.
    private(set) var held: [Engine.Held] = []
    /// The footprint over the render's whole time (`Engine.Event.memory`): the render under way's, then the last one's
    /// until the next begins (`beginJob` clears it).
    private(set) var memory = Engine.MemorySeries()
    /// A mirror has no `plan`: the other Mac's evaluations per image (`RemoteMeasures`).
    private(set) var remoteEvaluations: Int?
    /// When each stage was last heard compiling (`compiling(_:at:)`).
    private(set) var compiledAt: [Engine.Stage: Date] = [:]

    /// **A burst of compilations said as one**: a stage compiles while the engine says so, and for `burst` seconds past
    /// the last time it did. A model's first render compiles in bursts — the decoder a function per band, a block its
    /// functions one after the other — and a word that came and went with each one flickered on every line that says it.
    static let burst: TimeInterval = 1.5
    func compiling(_ stage: Engine.Stage, at now: Date = Date()) -> Bool {
        held.first { $0.stage == stage }?.compiling == true || compiledAt[stage].map { now.timeIntervalSince($0) < Self.burst } == true
    }
    func compiling(at now: Date = Date()) -> Bool { held.contains { compiling($0.stage, at: now) } }

    /// **The DiT's cycles of image `image`** (`Engine.Held.cycles`): one block for the batch, its evaluations in turn.
    func cycles(image: Int) -> ArraySlice<Engine.Cycle> {
        let all = held.first { $0.stage == .denoising }?.cycles ?? []
        guard planned > 0 else { return all[...] }
        let from = min(max(image, 0) * planned, all.count)
        return all[from..<min(from + planned, all.count)]
    }

    init() {}

    /// What the other Mac measured for this Mac's render, dated here: `begin` is `elapsed` ago.
    init(remote r: RemoteMeasures, job: Job, now: Date = Date()) {
        self.job = job
        begin = now.addingTimeInterval(-r.elapsed)
        remoteEvaluations = r.evaluations
        tokens = r.tokens; tokenizer = r.tokenizer; encoder = r.encoder; encoding = r.encoding
        textCached = r.textCached; stepSeconds = r.stepSeconds; decodings = r.decodings
        timings = r.timings; footprints = r.footprints
        held = r.footprints?.stages ?? []
        memory = r.memory ?? Engine.MemorySeries()
    }

    /// The evaluations planned per image, here or on the Mac that renders.
    var planned: Int { plan?.denoising.evaluations ?? remoteEvaluations ?? 0 }

    mutating func beginJob(_ t: Job) {
        self = Measures()
        job = t
        begin = Date()
    }

    mutating func receive(_ item: Engine.Event) {
        switch item {
        case .start(let p):
            plan = p
            stepSeconds = Array(repeating: [], count: p.seeds.count)
        case let .text(j, t, e, cached):
            tokens = j; tokenizer = t; encoder = e; textCached = cached
        case .encoding(let s):
            encoding = s
        case let .step(image, _, _, _, seconds, _, _) where image < stepSeconds.count:
            stepSeconds[image].append(seconds)
        case .decoding(let s):
            decodings.append(s)
        case .image(_, let renderResult):
            timings = renderResult.timings
            footprints = renderResult.footprints
            held = renderResult.footprints.stages
        case .memory(let delta):
            memory.append(delta)
        case .held(let stages):
            // Heard compiling, or done compiling since the last event: the burst's last moment so far.
            for s in stages where s.compiling || held.first(where: { $0.stage == s.stage })?.compiling == true { compiledAt[s.stage] = Date() }
            held = stages
        default:
            break
        }
    }

    /// This render's pace; `nil` without a step.
    var speed: Speed? {
        func mean(_ v: [Double]) -> Double? { v.isEmpty ? nil : v.reduce(0, +) / Double(v.count) }
        guard let full = mean(stepSeconds.flatMap { $0 }) else { return nil }
        return Speed(full: full, text: textCached ? nil : (tokenizer ?? 0) + (encoder ?? 0),
                       image: encoding ?? 0, decoding: mean(decodings) ?? Engine.Progress.decodingWeights * full)
    }
}
