import Foundation
import Siliconed
import SiliCore

// **A decoder by bands is the decoder** — `siliconed check <model> vae-bands [--size 1024] [--budget GiB]`.
//
// A decoder whose norms read the whole plane (Flux's two: `FluxVAE`) always decodes layer by layer and by bands of
// rows; B only chooses how short (`FluxVAE.candidates`). Its stage decodes the reference's latent by much shorter bands
// than it prefers and judges it as the `vae` stage judges the decoder, on its own reference; at a size no reference is
// written at, the latent repeated over the grid, the memory alone is judged: what the process held against the plan's P.
//
// A decoder whose norms read no more than a position (Wan's, Qwen-Image's VAE and its kin: its RMS norms over the
// channels; its one attention, at the latent's size, is computed whole) always decodes layer by layer and by bands of
// rows too, each band reading its neighbours' rows as its margins; B only chooses how short (`QwenImageVAE.positions`).
// Its stage decodes one latent twice — under this Mac's B, then under a small one — and compares the two pictures value
// by value: the bands' factors are powers of two and their rows' own, so how short they are changes no bit.
//
//     the latent    the `vae` stage's at 512², repeated over the grid of the size asked
//     the small B   `--budget`, else three quarters of the P of the plan this Mac's B takes: a B its bands do not fit

enum BandStages {
    static let all: [Stage] = [
        stage("anima", sizes: ["1024"], QwenImageVAEDecoder.self, latent: { try plain($0, "anima") }, path: { try Library.processWide.component(.anima, "vae.safetensors") }),
        stage("krea2", sizes: ["1024"], QwenImageVAEDecoder.self, latent: { try plain($0, "krea2") }, path: { try Library.processWide.component(.krea2, "vae.safetensors") }),
        stage("qwen-image-2.1", sizes: ["1024"], QwenImage21VAEDecoder.self, latent: packed, path: { Library.processWide.map(QwenImage21Stages.vaeMap) }),
        layers("z-image", sizes: ["768x512", "2048"], reference: "768x512", goldens: "goldens-vae-{size}", witness: "witness-vae-f64-{size}",
               half: ("witness-bf16-vae-{size}", "tools/oracle/z-image.py vae-bf16 --size {size}"),
               vae: FluxVAEDecoder.vae, path: { try Library.processWide.component(.zImage, "vae.safetensors") }),
        layers("moss-image", sizes: ["512", "2048"], reference: "512", goldens: "moss/goldens-moss-vae-{size}", witness: "moss/witness-moss-vae-f64-{size}",
               half: nil, vae: Flux2VAEDecoder.vae, path: { try Library.processWide.component(.mossImage, "vae.safetensors") }),
    ]

    /// A decoder by layers and bands: at the reference's size its own latent, decoded by bands of `--elements`
    /// positions (by default the shortest candidate, 8 times shorter than the preferred), or under `--budget`, judged on
    /// the reference by the `vae` stage's rule; at another size the same latent repeated over the grid, under `--budget`
    /// (by default 10 GiB, a Mac of 16 GB's), the memory judged.
    static func layers(_ model: String, sizes: [String], reference size: String, goldens: String, witness: String, half: (file: String, script: String)?,
                       vae: FluxVAE, path: @escaping @Sendable () throws -> String) -> Stage {
        Stage(
            model: model, name: "vae-bands",
            proves: "the decoder layer by layer and by bands of rows, at bands much shorter than it prefers, on the vae stage's reference: the norms' statistics combined from the bands' parts, the margins masked at the plane's edges; and at a size no reference reaches, what it held against its plan",
            sizes: sizes, options: ["--budget", "--elements"], oracle: nil, weights: ["VAE"],
            rule: "at the reference's size, per plane, at the fp64 witness: no further from it than the reference in the decoder's arithmetic, + 25 % (the vae stage's rule); at any size, the footprint's peak no higher than the plan's P + 64 MiB, and P under B"
        ) { invocation in
            let report = Report(title: "\(model) vae-bands \(invocation.size)")
            let named = { (stem: String) in stem.replacingOccurrences(of: "{size}", with: size) }
            let file = try invocation.goldens.open(named(goldens) + ".safetensors")
            let fp64 = invocation.goldens.optional(named(witness) + ".safetensors")
            let reference = try half.map { try invocation.goldens.half(named($0.file) + ".safetensors", script: named($0.script)) }
            guard let entry = file.entries["latent"], let latent = file.float32("latent"), let expected = file.float32("image") else {
                throw Skip("\(file.path): latent or image missing")
            }
            let (h, w) = (invocation.height / vae.factor, invocation.width / vae.factor), judged = [h, w] == Array(entry.shape[1...])
            let (c, h0, w0) = (entry.shape[0], entry.shape[1], entry.shape[2])
            var values = [Float](repeating: 0, count: c * h * w)
            for k in 0..<c { for y in 0..<h { for x in 0..<w { values[(k * h + y) * w + x] = latent[(k * h0 + y % h0) * w0 + x % w0] } } }
            let budget: Int
            if let text = invocation.extras["--budget"] {
                guard let gib = Double(text), gib > 0 else { throw UsageError("--budget \(text): GiB") }
                budget = Int(gib * Double(1 << 30))
            } else { budget = judged ? MemoryBudget.machineCeiling() : 10 << 30 }
            let forced = try invocation.integer("--elements", judged && invocation.extras["--budget"] == nil ? FluxVAE.candidates.last! : 0)
            let vaePath = try path()

            let watch = FootprintWatch()
            watch.reset()
            let start = Date()
            let (image, plan, p) = try blocking { () async throws -> ([Float], FluxVAEPlan, SiliCore.Stage.Plan) in
                let (pass, _) = try await vae.open(.decode, path: vaePath, height: h, width: w, budget: budget, positions: forced > 0 ? forced : nil)
                defer { pass.stage.close() }
                return (try await pass.decode(values), pass.plan, pass.stage.plan)
            }
            let seconds = Date().timeIntervalSince(start), peak = watch.peak
            watch.stop()
            let gib = { (bytes: Int) in Double(bytes) / Double(1 << 30) }
            report.add(Judge.holds("the plan cuts", plan.positions < FluxVAE.candidates[0] || !judged,
                                   note: String(format: "bands of at most %d positions, %d functions, %d calls; B %.2f GiB",
                                                plan.positions, plan.bands.functionCount, plan.bands.callCount, gib(budget))))
            if judged {
                report.add(Judge.planes("image RGB", engine: image, oracle: expected, witness: fp64?.float32("image"),
                                        reference: reference?.float32("image").map(Array.init), planes: 3))
            }
            report.add(Judge.bound("the footprint's peak", gib(peak), gib(p.p + SiliCore.Stage.margin), unit: "GiB",
                                   note: String(format: "P %.2f GiB (floor %.2f, arena %.2f) under B %.2f; decoded in %.2f s, the stage opened and closed",
                                                gib(p.p), gib(p.floor), gib(p.arena), gib(budget), seconds)))
            return report
        }
    }

    typealias Latent = (shape: [Int], values: [Float])

    /// Anima's and Krea 2's final latent, `[16, h, w]`, normalized as their renders give it to the decoder.
    static func plain(_ goldens: Goldens, _ family: String) throws -> Latent {
        let file = try goldens.open("\(family)/goldens-\(family)-vae-512.safetensors")
        guard let entry = file.entries["latent_final"], let latent = file.float32("latent_final") else { throw Skip("\(file.path): latent_final missing") }
        return (entry.shape, QwenImageVAEDecoder.normalized(Array(latent)))
    }

    /// Qwen-Image 2.1's, `[h·w, 64]` tokens unpacked to planes and de-normalized.
    static func packed(_ goldens: Goldens) throws -> Latent {
        let file = try goldens.open("qwen21/goldens-qwen21-vae-512.safetensors")
        guard let entry = file.entries["vae_in"], let tokens = file.float32("latent_final") else { throw Skip("\(file.path): latent_final missing") }
        let (channels, area) = (entry.shape[0], entry.shape[1] * entry.shape[2])
        var latent = [Float](repeating: 0, count: tokens.count)
        for t in 0..<area { for c in 0..<channels { latent[c * area + t] = tokens[t * channels + c] } }
        return (entry.shape, QwenImage21VAEDecoder.denormalized(latent))
    }

    /// A decoder by bands of rows against the whole one, bit for bit, under a small B (`--budget`).
    static func stage<Decoder: BandedDecoder & SendableMetatype>(_ model: String, sizes: [String], _: Decoder.Type, latent: @escaping @Sendable (Goldens) throws -> Latent,
                                              path: @escaping @Sendable () throws -> String) -> Stage {
        Stage(
            model: model, name: "vae-bands",
            proves: "the decoder the plan takes by bands of rows under a B its whole session does not fit: the same picture as the whole decoder's, and what the device held against the plan's need",
            sizes: sizes, options: ["--budget"], oracle: nil, weights: ["VAE"],
            rule: "under the small B the plan cuts; every value of the picture is the whole decoder's, bit for bit; the banded session holds no more than its plan's need"
        ) { invocation in
            let report = Report(title: "\(model) vae-bands \(invocation.size)")
            let seed = try latent(invocation.goldens)
            let (c, h0, w0) = (seed.shape[0], seed.shape[1], seed.shape[2])
            let (h, w) = (invocation.height / Decoder.factor, invocation.width / Decoder.factor)
            var values = [Float](repeating: 0, count: c * h * w)
            for k in 0..<c { for y in 0..<h { for x in 0..<w { values[(k * h + y) * w + x] = seed.values[(k * h0 + y % h0) * w0 + x % w0] } } }
            let given = try invocation.extras["--budget"].map { text -> Int in
                guard let gib = Double(text), gib > 0 else { throw UsageError("--budget \(text): GiB") }
                return Int(gib * Double(1 << 30))
            }
            let machine = MemoryBudget.machineCeiling()
            let gib = { (bytes: Int) in Double(bytes) / Double(1 << 30) }
            let (input, vae) = (values, try path())

            var start = Date()
            let admitted = try Decoder.admit(path: vae, latentHeight: h, latentWidth: w, budget: machine)
            let (whole, preferred, wholeP) = try blocking { () async throws -> ([Float], Int, Int) in
                let run = try await Decoder.open(admitted)
                defer { run.stage.close() }
                return (try await run.pixels(input), run.positions, run.stage.plan.p)
            }
            let wholeSeconds = Date().timeIntervalSince(start), budget = given ?? wholeP / 4 * 3

            // Then the bands, under the small B: the peak is theirs, its floor counted in their plan's P.
            let watch = FootprintWatch()
            watch.reset()
            start = Date()
            let small = try Decoder.admit(path: vae, latentHeight: h, latentWidth: w, budget: budget)
            let (banded, positions, p) = try blocking { () async throws -> ([Float], Int, SiliCore.Stage.Plan) in
                let run = try await Decoder.open(small)
                defer { run.stage.close() }
                return (try await run.pixels(input), run.positions, run.stage.plan)
            }
            let bandedSeconds = Date().timeIntervalSince(start), peak = watch.peak
            watch.stop()

            report.add(Judge.holds("the plan cuts", positions < preferred,
                                   note: String(format: "B %.2f GiB: bands of at most %d positions; %d under this Mac's B %.2f GiB, its P %.2f GiB",
                                                gib(budget), positions, preferred, gib(machine), gib(wholeP))))
            var differing = 0, worst: Float = 0
            for (a, b) in zip(banded, whole) where a != b { differing += 1; worst = max(worst, abs(a - b)) }
            if differing > 0 { print("  worst |Δ| \(worst)") }
            report.add(Judge.identity("pixels, bands against whole", differing: banded.count == whole.count ? differing : -1, of: whole.count, against: banded.count))
            report.add(Judge.bound("the footprint's peak", gib(peak), gib(p.p + SiliCore.Stage.margin), unit: "GiB",
                                   note: String(format: "P %.2f GiB (floor %.2f, arena %.2f) under B %.2f; decoded in %.2f s, the whole one in %.2f s, each stage opened and closed",
                                                gib(p.p), gib(p.floor), gib(p.arena), gib(budget), bandedSeconds, wholeSeconds)))
            return report
        }
    }
}
