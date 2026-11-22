import Foundation
import SiliCore

/// **A VAE of Flux's kind and its pass**: its networks written as layers over planes (`Bands`), the decoder (a latent to
/// the picture) and the encoder (a picture to its latent), every weight a resident of the arena.
///
/// **The bands** (`Bands.height`): a layer's band is the tallest power of two whose output holds at most `positions`
/// positions — one function a layer whatever the plane's rows, a taller picture more calls, its last band past the
/// plane's end. `positions` sets how much of Core AI's memory a function holds (its intermediates scale with its band)
/// against how many calls a pass makes. Measured on the M1 Pro, Flux's decoder at 1024², the kernels constants (18/11:
/// 35 functions, one a layer; the arena 1044 MiB, the kernels 92 MiB): at 2¹⁴, a decoding 4.71 – 4.81 s, the peak
/// 3485 MiB; at 2¹⁷, the peak 10 040 MiB and the M1 Pro swaps.
///
/// **The planes live in slabs**, each as its bands: a resnet reads x and writes h, then reads h and x and writes x′; an
/// ×2 or a ↓ reads x and writes y. Three slabs hold them in turn, each as large as the largest plane it ever holds.
package struct FluxVAEPlan {
    package enum Direction: Sendable { case decode, encode }

    package let vae: FluxVAE
    package let direction: Direction
    /// The latent's rows and columns (decoding), or the picture's (encoding).
    package let height, width: Int
    package let bands: Bands
    /// What the pass is given and gives back: the latent and the picture, by direction.
    let input, output: Bands.Plane
    /// The most positions a band's output holds.
    package var positions: Int { bands.positions }

    /// The pass planned before anything is allocated. `height`, `width`: the latent's (`decode`), the picture's
    /// (`encode`, each a multiple of 8 · the cell).
    package init(_ vae: FluxVAE, _ direction: Direction, height: Int, width: Int, positions: Int) {
        (self.vae, self.direction, self.height, self.width) = (vae, direction, height, width)
        var ends: (Bands.Plane, Bands.Plane)!
        bands = Bands.surveyed(positions: positions, element: vae.element, factors: vae.half) { p in
            ends = direction == .decode ? Self.decoder(vae, &p, height, width) : Self.encoder(vae, &p, height, width)
        }
        (input, output) = ends
    }

    /// A layer of `kind` reading `reads` and writing `y`, its weights those of `layer`.
    private static func layer(_ vae: FluxVAE, _ p: inout Bands, _ kind: FluxVAE.Kind, _ layer: String, cin: Int, cout: Int, reads: [Bands.Read], _ y: Bands.Plane,
                              whole: Bool = false) {
        let weights = Dictionary(uniqueKeysWithValues: vae.weights(kind, layer: layer, cin: cin, cout: cout).map { ($0.name, $0.key) })
        p.layer(vae.function(kind, cin: cin, cout: cout), reads: reads, writes: [Bands.Write("y", y)], weights: weights, whole: whole)
    }

    /// A plane of `c` channels whose writer carries its GroupNorm's statistics.
    private static func plane(_ p: inout Bands, _ c: Int, _ h: Int, _ w: Int) -> Bands.Plane { p.plane(c, rows: h, width: w, carry: FluxVAE.statistics) }

    /// A resnet of `layer` on `x` (`cin` channels) to `cout`: conv1 by bands into h, conv2 by bands into x′.
    private static func resnet(_ vae: FluxVAE, _ p: inout Bands, _ x: Bands.Plane, _ cout: Int, _ name: String) -> Bands.Plane {
        let h = plane(&p, cout, x.rows, x.width)
        layer(vae, &p, .conv1, name, cin: x.channels, cout: cout, reads: [Bands.Read("x", x, before: true, after: true, carry: "x_stats")], h)
        let next = plane(&p, cout, x.rows, x.width)
        layer(vae, &p, .conv2, name, cin: x.channels, cout: cout, reads: [Bands.Read("h", h, before: true, after: true, carry: "h_stats"), Bands.Read("x", x)], next)
        p.release(h); p.release(x)
        return next
    }

    /// **The decoder's network**: the head whole, each level's resnets and ×2 by bands, `out` to the picture's planes.
    /// Its boundaries marked, diffusers' names: `decoder_mid`, the trunk's end; `decoder_up<i>`, the i-th ×2's output.
    private static func decoder(_ vae: FluxVAE, _ p: inout Bands, _ height: Int, _ width: Int) -> (Bands.Plane, Bands.Plane) {
        let latent = p.plane(vae.latentChannels, rows: height, width: width, element: .float32, factors: false)
        var x = plane(&p, vae.trunk, vae.cell * height, vae.cell * width)
        layer(vae, &p, .head, "decoder", cin: vae.latentChannels, cout: vae.trunk, reads: [Bands.Read("latent", latent)], x, whole: true)
        p.mark("decoder_mid", x)
        for (level, cout) in vae.decoderWidths.enumerated() {
            for i in 0..<3 { x = resnet(vae, &p, x, cout, "decoder.up_blocks.\(level).resnets.\(i)") }
            if level < vae.decoderWidths.count - 1 {
                let y = plane(&p, x.channels, 2 * x.rows, 2 * x.width)
                layer(vae, &p, .up, "decoder.up_blocks.\(level)", cin: x.channels, cout: x.channels, reads: [Bands.Read("x", x, over: 2, before: true, after: true)], y)
                p.mark("decoder_up\(level)", y)
                p.release(x)
                x = y
            }
        }
        let pixels = p.plane(3, rows: x.rows, width: x.width, element: .float32, factors: false, apart: true)
        layer(vae, &p, .out, "decoder", cin: x.channels, cout: 3, reads: [Bands.Read("x", x, before: true, after: true, carry: "x_stats")], pixels)
        return (latent, pixels)
    }

    /// **The encoder's network**: conv_in on the picture and each level's resnets and ↓ by bands, the tail whole.
    private static func encoder(_ vae: FluxVAE, _ p: inout Bands, _ height: Int, _ width: Int) -> (Bands.Plane, Bands.Plane) {
        let picture = p.plane(3, rows: height, width: width, element: .float32, factors: false)
        var x = plane(&p, vae.encoderWidths[0], height, width)
        layer(vae, &p, .first, "encoder", cin: 3, cout: vae.encoderWidths[0], reads: [Bands.Read("x", picture, before: true, after: true)], x)
        for (level, cout) in vae.encoderWidths.enumerated() {
            for i in 0..<2 { x = resnet(vae, &p, x, cout, "encoder.down_blocks.\(level).resnets.\(i)") }
            if level < vae.encoderWidths.count - 1 {
                // A band of r output rows reads the input rows 2s ..< 2s + 2r + 1: the last of the last band is past the
                // plane, the ↓'s zero padding (its mask's 0).
                let y = plane(&p, x.channels, x.rows / 2, x.width / 2)
                layer(vae, &p, .down, "encoder.down_blocks.\(level).downsamplers.0.conv", cin: x.channels, cout: x.channels,
                      reads: [Bands.Read("x", x, per: 2, after: true)], y)
                p.release(x)
                x = y
            }
        }
        let latent = p.plane(vae.latentChannels, rows: x.rows / vae.cell, width: x.width / vae.cell, element: .float32, factors: false, apart: true)
        layer(vae, &p, .tail, "encoder", cin: x.channels, cout: vae.latentChannels, reads: [Bands.Read("x", x)], latent, whole: true)
        return (picture, latent)
    }

    /// The weights the plan's calls read, each with its recipe.
    func residents(_ file: SafeTensors) -> [Weights.Entry] { bands.weights.map { ($0, $0, vae.recipe($0, in: file)) } }
}

/// **A pass of a VAE of Flux's kind, its arena filled and its functions loaded**: what a render opens once and runs
/// for each latent (or picture) of its batch.
package final class FluxVAEPass: @unchecked Sendable {
    package let plan: FluxVAEPlan
    let bands: BandStage
    /// The stage that holds the arena, the weights and the functions, each loaded before the first pass and kept.
    package var stage: SiliCore.Stage { bands.stage }

    init(_ plan: FluxVAEPlan, _ bands: BandStage) { (self.plan, self.bands) = (plan, bands) }

    /// The latent (`[latentChannels, h, w]`, f32) to the picture's planes `[3, H, W]`, f32, around [−1, 1].
    package func decode(_ latent: [Float]) async throws -> [Float] { try await decode(latent, keeping: []).pixels }

    /// The same, and the boundaries `keeping` names (`decoder_mid`, `decoder_up<i>`), each `[C, H, W]` f32.
    package func decode(_ latent: [Float], keeping: Set<String>) async throws -> (pixels: [Float], boundaries: [String: [Float]]) {
        precondition(plan.direction == .decode, "an encoder's pass")
        bands.write(plan.input, latent)
        let kept = try await bands.run(keeping: keeping)
        return (bands.values(plan.output), kept)
    }

    /// The picture's planes `[3, H, W]` in [−1, 1] to what the encoder's tail gives (`FluxVAE.Latent`).
    package func encode(_ image: [Float]) async throws -> [Float] {
        precondition(plan.direction == .encode, "a decoder's pass")
        bands.write(plan.input, image)
        _ = try await bands.run()
        return bands.values(plan.output)
    }
}

extension FluxVAE {
    /// **The bands a pass is planned among**, in positions, from the one it prefers. 2¹⁴ first: the decoder at 1024² is
    /// both faster and lighter there than at any taller band (measured 18/11 on the M1 Pro, `FluxVAEBands.swift`); the
    /// shorter ones make room under a small B.
    package static let candidates = [1 << 14, 1 << 13, 1 << 12, 1 << 11]

    /// **A pass of this VAE at `path` opened under `budget`**: one plan a candidate band (`positions`: that one alone),
    /// the first whose P holds taken (`BandStage.open`). `narrowed`: B took a band shorter than the preferred
    /// (`BandStage.narrowed`).
    package func open(_ direction: FluxVAEPlan.Direction, path: String, height: Int, width: Int, budget: Int,
                      positions: Int? = nil) async throws -> (pass: FluxVAEPass, narrowed: Bool) {
        let file = try SafeTensors(path)
        let candidates = positions.map { [$0] } ?? Self.candidates
        let plans = candidates.map { FluxVAEPlan(self, direction, height: height, width: width, positions: $0) }
        let side = direction == .decode ? "decoder" : "encoder"
        let stage = try await BandStage.open("\(name) \(side) \(height)x\(width)", prefix: "\(name)-\(side)", candidates: candidates, files: [file],
                                             residents: plans[0].residents(file), budget: budget) { p in plans[candidates.firstIndex(of: p)!].bands }
        return (FluxVAEPass(plans[stage.candidate], stage), stage.narrowed)
    }

    /// Whether `open` would hold under `budget`, nothing allocated (`BandStage.admit`).
    package func admit(_ direction: FluxVAEPlan.Direction, path: String, height: Int, width: Int, budget: Int) throws {
        let file = try SafeTensors(path)
        let plans = Self.candidates.map { FluxVAEPlan(self, direction, height: height, width: width, positions: $0) }
        let side = direction == .decode ? "decoder" : "encoder"
        try BandStage.admit("\(name) \(side) \(height)x\(width)", prefix: "\(name)-\(side)", candidates: Self.candidates, files: [file],
                            residents: plans[0].residents(file), budget: budget) { p in plans[Self.candidates.firstIndex(of: p)!].bands }
    }
}
