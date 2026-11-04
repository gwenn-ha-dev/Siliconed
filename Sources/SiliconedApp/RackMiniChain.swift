// **The chain in miniature, pinned at the head of the rack** once its boxes have scrolled away (`Sidebar`): the prompt, the
// text encoder, the DiT, the VAE decoder — and the image encoder under the text's, only when images go.
//
//     ❝ ─▶ Text encoder ─╮
//          ≈ 1.2 s       ├─▶ DiT ─▶ VAE decoder
//     ▣ ─▶ Image encoder ╯   ≈ 38 s  ≈ 2.1 s
//
// For seeing, not for wiring: the stage at work lit (the accent, as its box), the wires of their true type (`WireKind`), under each
// stage its time — what it just took, else the estimate learned on this Mac (`AppState.stageCost`, the boxes' own). Settings stay
// in the boxes below: a click on the strip brings them back. Its words are the boxes' titles, so it says nothing they do not.

import Siliconed
import SwiftUI

struct MiniChain: View {
    let app: AppState

    var body: some View {
        let images = sendsImages
        HStack(spacing: 0) {
            // The two encoders as wide as the wider, so that both wires leave from the same edge.
            VStack(alignment: .leading, spacing: Self.rowGap) {
                feed(symbol: "text.quote", kind: .text, stage: .text, title: "Text encoder")
                if images { feed(symbol: "photo", kind: .image, stage: .image, title: "Image encoder") }
            }
            .fixedSize(horizontal: true, vertical: false)
            Merge(rows: images ? [(.text, app.state(of: .denoising)), (.latent, app.state(of: .denoising))]
                               : [(.text, app.state(of: .denoising))], gap: Self.rowGap)
                .frame(width: 16).frame(maxHeight: .infinity)
            stage(.denoising, title: "DiT")
            wire(.latent, app.state(of: .decoding)).frame(width: 14)
            stage(.decoding, title: "VAE decoder")
        }
        .fixedSize()
        .animation(Motion.pick(.easeInOut(duration: 0.25)), value: app.state(of: .denoising))
    }

    private static let rowGap: CGFloat = 4

    /// Images go to the image encoder: Qwen-Image-2.1 edits with references, or a model starts from an image (img2img) — what the
    /// rack's wires carry, not what the model could take.
    private var sendsImages: Bool {
        app.editingPossible ? app.chain?.encoding != nil && !app.references.isEmpty : app.usedStart != nil
    }

    /// A source (the prompt, the images), its wire, and the encoder that reads it.
    private func feed(symbol: String, kind: WireKind, stage s: Engine.Stage, title: LocalizedStringKey) -> some View {
        HStack(spacing: 0) {
            Image(systemName: symbol).font(.caption2).foregroundStyle(kind.color).frame(width: 14)
            wire(kind, app.state(of: s)).frame(width: 10)
            stage(s, title: title, fill: true)
        }
    }

    private func stage(_ s: Engine.Stage, title: LocalizedStringKey, fill: Bool = false) -> some View {
        let state = app.state(of: s)
        let cost = s == .image && app.sentImages == 0 ? (seconds: nil, measured: false) : app.stageCost(s)
        let shape = RoundedRectangle(cornerRadius: 7, style: .continuous)
        return VStack(spacing: 0) {
            Text(title).font(.caption2.weight(.semibold))
                .foregroundStyle(state == .active ? Palette.live : state == .failed ? Palette.failure : .primary)
            Text(cost.seconds.map { (cost.measured ? "" : "≈ ") + Self.short($0) } ?? "—")
                .font(.caption2).monospacedDigit().foregroundStyle(.secondary)
        }
        .lineLimit(1)
        .padding(.horizontal, 7).padding(.vertical, 3)
        .frame(maxWidth: fill ? .infinity : nil)
        .background(shape.fill(state == .active ? Palette.live.opacity(0.18) : Palette.surface))
        .overlay(shape.strokeBorder(state == .active ? Palette.live : state == .failed ? Palette.failure : Palette.hairline,
                                    lineWidth: state == .active || state == .failed ? 1.5 : 1))
    }

    /// The boxes' wire, in small: its colour, muted at rest, full and wider while the stage that moves it runs.
    private func wire(_ kind: WireKind, _ state: StageState) -> some View {
        Capsule().fill(state == .idle ? kind.color.opacity(0.6) : kind.color)
            .frame(height: state == .active ? 3 : 2)
    }

    /// `2.4 s`, `38 s`: the boxes' figure, a decimal shorter.
    private static func short(_ s: Double) -> String {
        Measurement(value: s, unit: UnitDuration.seconds)
            .formatted(.measurement(width: .abbreviated, usage: .asProvided,
                                    numberFormatStyle: .number.precision(.fractionLength(s < 10 ? 1 : 0))))
    }
}

/// The encoders' wires joining at the DiT: from the middle of each row to the middle of the strip.
private struct Merge: View {
    let rows: [(kind: WireKind, state: StageState)]
    let gap: CGFloat

    var body: some View {
        Canvas { c, size in
            let n = CGFloat(rows.count), row = (size.height - (n - 1) * gap) / n
            let end = CGPoint(x: size.width, y: size.height / 2)
            for (i, r) in rows.enumerated() {
                let start = CGPoint(x: 0, y: CGFloat(i) * (row + gap) + row / 2)
                var p = Path()
                p.move(to: start)
                p.addCurve(to: end, control1: CGPoint(x: size.width * 0.6, y: start.y), control2: CGPoint(x: size.width * 0.4, y: end.y))
                c.stroke(p, with: .color(r.state == .idle ? r.kind.color.opacity(0.6) : r.kind.color),
                         style: StrokeStyle(lineWidth: r.state == .active ? 3 : 2, lineCap: .round))
            }
        }
        .allowsHitTesting(false)
    }
}
