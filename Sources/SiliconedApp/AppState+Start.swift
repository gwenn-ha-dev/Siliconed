// **`AppState`: the starting image** (img2img) — carried from the Prompt Lab or by `silicontrol start-image`, shown in the
// rack's Images box with its strength, read by the render. The rules are the library's (`StartImage`, `Strength`); the state
// itself is in `AppState.swift`.
//
//     Prompt Lab ── a picture made « Starting image », its strength ──▶ Carry to the Rack
//                                                                        │ startImage, startStrength
//                                                                        │ Size ← the recommended format nearest its proportions
//     rack:  [ Images ▣ 75 % ━━●━━ ] ── right rail ──▶ Image encoder ── latent ──▶ DiT    (Request.image, Request.strength)
//
// - **Only where the chain starts from an image** (`startsFromImage`: an image encoder and a denoiser that accepts a start —
//   Z-Image, Anima, Krea 2). Qwen-Image-2.1 edits by reference instead: its image 1 stays the rack's references, untouched here.
//   MOSS-Image has no image encoder yet: the role and the box appear by themselves once its chain offers the entry.
// - **The image is read as the file is** (`ReferenceImage.Source.bytes` for a picture of the lab or of the history, `.file` for a
//   path): the bytes ImageIO decodes for `siliconed-dev render --image`, so that the app and the command line start from the same
//   pixels — the same seed and strength then give the same image.
// - **Incognito**: the image lives in memory, as the references do; nothing is written.

import AppKit
import ImageIO
import Siliconed
import SwiftUI

extension AppState {
    /// **The strength a starting image comes with**: the rack's family's (`StartImage.defaultStrength`).
    var defaultStartStrength: Double { StartImage.defaultStrength(for: card?.family) }

    /// Whether the rack's model starts from an image: its chain offers the image entry (`Chain.entries`). Never in a mirror: the
    /// other Mac's traits do not say it, and its jobs carry no starting image.
    var startsFromImage: Bool {
        guard mirror == nil, let chain else { return false }
        return chain.entries.contains { $0.kind == .image }
    }

    /// The starting image the next render reads: the one carried, where the model starts from one.
    var usedStart: ReferenceImage? { startsFromImage ? startImage : nil }

    /// The images the next render encodes: the edit's references, or the starting image — what its memory need counts
    /// (`MemoryNeed.Request.references`: a start at the render's format).
    var sentImages: Int { references.count + (usedStart == nil ? 0 : 1) }

    /// **The steps the strength leaves to the render**, of all those of the schedule — the plain words under the slider.
    /// `nil` without a starting image, or before the model is read.
    var startRedraws: (redrawn: Int, of: Int)? {
        guard usedStart != nil, let denoising = chain?.denoising else { return nil }
        return Self.redraws(denoising, steps: steps, strength: startStrength)
    }

    static func redraws(_ denoising: any DenoisingModule, steps: Int, strength: Double) -> (redrawn: Int, of: Int)? {
        guard steps >= 1 else { return nil }
        let start = Strength.startStep(steps: steps, strength: strength)
        let all = denoising.plan(height: 64, width: 64, steps: steps, start: 0, withLoRA: true, faster: false).evaluations
        let left = start < steps ? denoising.plan(height: 64, width: 64, steps: steps, start: start, withLoRA: true, faster: false).evaluations : 0
        return (max(0, left), all)
    }

    /// Why « Generate » waits on the starting image: a strength that leaves no step — the image would come back as it is.
    var startBlocker: String? {
        guard let r = startRedraws, r.redrawn == 0 else { return nil }
        return String(localized: "At this strength the image would come back unchanged: raise it.")
    }

    /// **Whether the strength ignores the image**: it redraws the whole schedule, from σ = 1 — pure noise, the image weighs
    /// nothing (at 8 steps, above 7/8). Rendered all the same, as diffusers does; said under the slider.
    var startIgnored: Bool {
        guard usedStart != nil, steps >= 1, let denoising = chain?.denoising,
              Strength.startStep(steps: steps, strength: startStrength) == 0 else { return false }
        return (denoising.sigmas(steps: steps).first ?? 0) >= 1
    }

    /// What the rack says under the slider: why « Generate » waits, that the image is ignored, or what a start does.
    var startCaption: (text: String, tone: StartTone) {
        if let blocker = startBlocker { return (blocker, .failure) }
        if startIgnored {
            return (String(localized: "At this strength the image is ignored: the render starts from noise alone. Lower it to keep its composition."), .warning)
        }
        return (String(localized: "The render starts from this image and keeps its composition. The size can differ: the centre is kept."), .plain)
    }

    enum StartTone { case plain, warning, failure }

    // MARK: Carrying, retouching, removing

    /// **A starting image comes into the rack** — from the Prompt Lab or the remote control: with its strength (the rack's when
    /// `nil`), and the size set to the model's recommended format nearest its proportions (`StartImage.recommendedFormat`), at
    /// the scale the rack had and among the sizes this Mac renders. The size stays free afterwards: the engine crops the centre.
    func carryStart(_ image: ReferenceImage, strength: Double? = nil) {
        guard startsFromImage else { return }
        setStart(image)
        if let strength { startStrength = StartImage.snapped(strength) }
        if let f = recommendedFormat(width: image.width, height: image.height), f != format { useFormat(f) }
    }

    /// The format a starting image of this size recommends, among the model's sizes this Mac renders.
    func recommendedFormat(width: Int, height: Int) -> RecommendedFormat? {
        let fitting = formats.filter { macNeeded($0, images: 1) == nil }
        return StartImage.recommendedFormat(width: width, height: height, among: fitting.isEmpty ? formats : fitting,
                                            current: format)
    }

    func removeStart() { setStart(nil) }

    /// The strength, as the slider and the remote control set it: in hundredths.
    func setStartStrength(_ s: Double) { startStrength = StartImage.snapped(s) }

    // MARK: Where a starting image comes from

    /// **An image of the history as a starting image**: its PNG's bytes, as `siliconed-dev render --image` would read the file.
    func startImage(from entry: Entry) -> ReferenceImage {
        ReferenceImage(source: .bytes(entry.png), vignette: NSImage(cgImage: entry.vignette, size: .zero),
                       name: String(localized: "image from history (seed \(String(entry.seed)))"),
                       width: entry.width, height: entry.height, ordinal: entry.ordinal)
    }

    /// **A file as a starting image**, or `nil` (and the problem said) when ImageIO cannot read it.
    func startImage(file url: URL) -> ReferenceImage? {
        guard let vignette = NSImage(contentsOf: url), vignette.isValid, let size = ReferenceImage.size(of: url) else {
            problem = Problem(EngineError.imageUnreadable(file: url.path))
            return nil
        }
        return ReferenceImage(source: .file(url), vignette: vignette, name: url.lastPathComponent,
                              width: size.width, height: size.height)
    }

    /// **Bytes as a starting image** — a picture of the Prompt Lab, kept as it was given, never re-encoded.
    static func startImage(bytes data: Data, thumbnail: NSImage, width: Int, height: Int, name: String) -> ReferenceImage {
        ReferenceImage(source: .bytes(data), vignette: thumbnail, name: name, width: width, height: height)
    }

    /// **PNG bytes as a starting image** (⌘V on the rack), its thumbnail reduced once; `nil` when ImageIO cannot read them.
    static func startImage(png data: Data, name: String) -> ReferenceImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let vignette = FinishedImage.reduce(cg, side: 240) ?? cg
        return startImage(bytes: data, thumbnail: NSImage(cgImage: vignette, size: .zero), width: cg.width, height: cg.height,
                          name: name)
    }

}

/// **The starting image in words**: its strength as a percentage, and what it means.
enum StartWords {
    /// `65 %` — the strength as the slider shows it.
    static func percent(_ s: Double) -> String { Figures.percent(Int((s * 100).rounded())) }

    /// The history caption's « from an image, 65 % ».
    static func legend(_ s: Double) -> String { String(localized: "from an image, \(percent(s))") }

    /// The slider's tooltip: the strength, then what it does here — the steps it leaves.
    static func explanation(_ redraws: (redrawn: Int, of: Int)?) -> String {
        let base = String(localized: "How far the render moves away from the starting image: low keeps its composition and colours, \(percent(1)) ignores it.")
        guard let r = redraws else { return base }
        return base + "\n" + String(localized: "Here: \(String(r.redrawn)) of the \(String(r.of)) steps are redrawn.")
    }
}
