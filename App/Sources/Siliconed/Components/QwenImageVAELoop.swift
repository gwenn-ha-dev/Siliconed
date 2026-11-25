import Foundation
import SiliCore

/// **A Wan VAE's plan and its run** (the functions of `QwenImageVAE`): its network written as layers over planes
/// (`Bands`), every weight a resident of the arena, its calls one after the other.
///
/// **The bands** (`Bands.height`): a layer's band is the tallest power of two whose output holds at most `positions`
/// positions (an ×2's output twice as tall and twice as wide as its input band, a ↓'s half as wide) — one function a
/// layer whatever the plane's rows, a taller picture more calls, its last band past the plane's end. A resnet reads x
/// and writes h, then reads h and x and writes x′; an ×2 reads x and writes y, and Qwen-Image-2.1's blocks read their
/// input again at their ×2 or ↓ (`entry`), which keeps its slab meanwhile.
extension QwenImageVAE {
    /// **The run planned before anything is allocated**: a network writes it with the steps below
    /// (`QwenImageVAEDecoder.plan`, `QwenImageVAEEncoder.plan`, Qwen-Image-2.1's).
    struct Plan {
        let network: String
        fileprivate(set) var bands: Bands
        /// What the run is given (the decoder's latent, the encoder's picture) and gives back (the pixels, the mean).
        private(set) var input: Bands.Plane?
        private(set) var result: Bands.Plane?
        var positions: Int { bands.positions }

        /// **The plan the network's `steps` write**, run twice (`Bands.surveyed`).
        static func surveyed(_ network: String, positions: Int, _ steps: (inout Plan) -> Void) -> Plan {
            var ends: (Bands.Plane?, Bands.Plane?) = (nil, nil)
            let bands = Bands.surveyed(positions: positions, element: .float16, factors: true) { b in
                var plan = Plan(network: network, bands: b)
                steps(&plan)
                b = plan.bands
                ends = (plan.input, plan.result)
            }
            var plan = Plan(network: network, bands: bands)
            (plan.input, plan.result) = ends
            return plan
        }

        /// A stored plane, its writer carrying its largest row factor.
        private mutating func plane(_ c: Int, _ h: Int, _ w: Int) -> Bands.Plane { bands.plane(c, rows: h, width: w, carry: QwenImageVAE.top) }
        mutating func release(_ t: Bands.Plane) { bands.release(t) }

        private mutating func layer(_ kind: Kind, _ layer: String, reads: [Bands.Read], _ y: Bands.Plane, whole: Bool = false, positions: Int? = nil) {
            let weights = Dictionary(uniqueKeysWithValues: QwenImageVAE.weights(kind, layer: layer).map { ($0.name, $0.key) })
            bands.layer(QwenImageVAE.function(network, kind), reads: reads, writes: [Bands.Write("y", y)], weights: weights, positions: positions, whole: whole)
        }

        // MARK: The steps

        /// The decoder's head: the latent `[c, h, w]` f32 through `steps`, whole, to a plane of `channels`.
        mutating func head(_ steps: [Step], latent c: Int, h: Int, w: Int, channels: Int) -> Bands.Plane {
            let latent = bands.plane(c, rows: h, width: w, element: .float32, factors: false)
            input = latent
            let x = plane(channels, h, w)
            layer(.head(steps: steps), "", reads: [Bands.Read("latent", latent)], x, whole: true)
            return x
        }

        /// The encoder's picture `[c, h, w]`, a stored plane its run writes (f16, its factors 1).
        mutating func picture(c: Int, h: Int, w: Int) -> Bands.Plane {
            let t = plane(c, h, w)
            input = t
            return t
        }

        /// The encoder's tail: `x` whole through `steps`, to the mean `[channels, h, w]` f32.
        mutating func tail(_ steps: [Step], _ x: Bands.Plane, channels: Int) {
            let mean = bands.plane(channels, rows: x.rows, width: x.width, element: .float32, factors: false, apart: true)
            result = mean
            layer(.tail(steps: steps), "", reads: [Bands.Read("x", x)], mean, whole: true)
        }

        /// A 3×3 of `key` over `x` with no norm, by bands, to `channels`. `x` released.
        mutating func first(_ x: Bands.Plane, key: String, channels: Int) -> Bands.Plane {
            let y = plane(channels, x.rows, x.width)
            layer(.first(cin: x.channels, cout: channels), key, reads: [Bands.Read("x", x, before: true, after: true, carry: "x_top")], y)
            release(x)
            return y
        }

        /// A resnet of `layer` over `x`, to `channels`, by bands: conv1 then conv2. `x` released unless `keeping` (a
        /// block's input read again).
        mutating func resnet(_ x: Bands.Plane, layer name: String, channels: Int, keeping: Bool = false) -> Bands.Plane {
            let hidden = plane(channels, x.rows, x.width)
            layer(.conv1(cin: x.channels, cout: channels), name, reads: [Bands.Read("x", x, before: true, after: true)], hidden)
            let next = plane(channels, x.rows, x.width)
            let shortcut = Bands.Read("x", x, carry: x.channels == channels ? nil : "x_top")
            layer(.conv2(cin: x.channels, cout: channels), name, reads: [Bands.Read("h", hidden, before: true, after: true), shortcut], next)
            release(hidden)
            if !keeping { release(x) }
            return next
        }

        /// Nearest ×2 then the 3×3 of `key` over `x`, to `channels`, by bands of input rows; with `dup`, the DupUp3D of
        /// the block's input `entry` added. `x` released (not `entry`).
        mutating func up(_ x: Bands.Plane, key: String, channels: Int, dup: (entry: Bands.Plane, temporal: Int)? = nil) -> Bands.Plane {
            let y = plane(channels, 2 * x.rows, 2 * x.width)
            var reads = [Bands.Read("x", x, over: 2, before: true, after: true, carry: "x_top")]
            if let dup { reads.append(Bands.Read("entry", dup.entry, over: 2)) }
            layer(.up(cin: x.channels, cout: channels, dup: dup.map { Dup(input: $0.entry.channels, temporal: $0.temporal) }), key, reads: reads, y)
            release(x)
            return y
        }

        /// The zero padding right and bottom, then the 3×3 of `key` at stride 2 over `x`, by bands of output rows; with
        /// `avg`, the AvgDown3D of the block's input `entry` added. `x` released (not `entry`).
        mutating func down(_ x: Bands.Plane, key: String, avg: (entry: Bands.Plane, kind: Avg)? = nil) -> Bands.Plane {
            let y = plane(x.channels, x.rows / 2, x.width / 2)
            // Output rows s ..< s + r read input rows 2s ..< 2s + 2r + 1: for the last band, the last past the plane (masked).
            var reads = [Bands.Read("x", x, per: 2, after: true, carry: "x_top")]
            if let avg { reads.append(Bands.Read("entry", avg.entry, per: 2)) }
            layer(.down(c: x.channels, avg: avg?.kind), key, reads: reads, y)
            release(x)
            return y
        }

        /// RMS · SiLU · `conv_out` of `layer` over `x`, clamped, by bands: the pixels `[planes, H, W]` f32. Its bands are
        /// the tallest candidate's in every plan: a convolution to 3 or 4 planes is computed otherwise as its band is
        /// taller (measured 19/11: 96 → 3 planes, bands of 16 rows against 4, 11 453 of 12 288 values differ, none
        /// rounded to its f16), and its share is small beside the trunk's.
        mutating func out(_ x: Bands.Plane, layer name: String, planes: Int) {
            let pixels = bands.plane(planes, rows: x.rows, width: x.width, element: .float32, factors: false, apart: true)
            result = pixels
            layer(.out(cin: x.channels, planes: planes), name, reads: [Bands.Read("x", x, before: true, after: true)], pixels,
                  positions: QwenImageVAE.positions[0])
            release(x)
        }

        /// The weights the plan's calls read, each with its recipe.
        func residents(_ file: SafeTensors) -> [Weights.Entry] { bands.weights.map { ($0, $0, QwenImageVAE.recipe($0, in: file)) } }
    }
}

/// **A decoder by bands of rows** (`QwenImageVAERun`'s): how short B made its bands, its stage, and how many pixels a
/// latent position makes on a side (`vae-bands`).
package protocol BandedDecoder: LatentDecoder {
    static var factor: Int { get }
    var positions: Int { get }
    var stage: SiliCore.Stage { get }
}

/// **A Wan VAE's run**: the stage of the first of its plans (by bands from the tallest) whose P holds under B — its
/// arena allocated, its weights read, its functions loaded — then its calls one after the other.
final class QwenImageVAERun {
    typealias Plan = QwenImageVAE.Plan

    let plan: Plan
    let bands: BandStage
    var stage: SiliCore.Stage { bands.stage }

    private init(_ plan: Plan, _ bands: BandStage) { (self.plan, self.bands) = (plan, bands) }

    /// **What a run opens and `admit` judges, written once**: the network's plan for each band's size
    /// (`QwenImageVAE.positions`, from the tallest), from the VAE `file` — so that what a render judges before its DiT
    /// is what it opens after.
    struct Plans {
        let network: String
        let file: SafeTensors
        let plans: [Plan]
        /// `plan` writes the network's plan for a band's size.
        init(_ network: String, file: SafeTensors, plan: (Int) -> Plan) {
            (self.network, self.file, plans) = (network, file, QwenImageVAE.positions.map(plan))
        }
        var residents: [Weights.Entry] { plans[0].residents(file) }
        func bands(_ positions: Int) -> Bands { plans[QwenImageVAE.positions.firstIndex(of: positions)!].bands }
    }

    /// **The run of the first of `plans` that holds under `budget`** (`BandStage.open`).
    static func open(_ plans: Plans, budget: Int) async throws -> QwenImageVAERun {
        let stage = try await BandStage.open(plans.network, prefix: plans.network, candidates: QwenImageVAE.positions, files: [plans.file],
                                             residents: plans.residents, budget: budget, plan: plans.bands)
        return QwenImageVAERun(plans.plans[stage.candidate], stage)
    }

    /// Whether `open` would hold under `budget`, nothing allocated (`BandStage.admit`).
    static func admit(_ plans: Plans, budget: Int) throws {
        try BandStage.admit(plans.network, prefix: plans.network, candidates: QwenImageVAE.positions, files: [plans.file],
                            residents: plans.residents, budget: budget, plan: plans.bands)
    }

    /// What `open` would specialize, nothing allocated (`BandStage.prepare`).
    static func prepare(_ plans: Plans, budget: Int) throws -> [Functions.Specialization] {
        try BandStage.prepare(plans.network, prefix: plans.network, candidates: QwenImageVAE.positions, files: [plans.file],
                              residents: plans.residents, budget: budget, plan: plans.bands)
    }

    /// What the run is given — the latent `[c, h, w]` f32, the picture `[c, H, W]` in [−1, 1] (f16 under factors of 1,
    /// its largest 1) — run, and what it gives back, `[C, H, W]` f32.
    func run(_ input: [Float]) async throws -> [Float] {
        let t = plan.input!
        bands.write(t, input, carry: t.carry.isEmpty ? [] : [1])
        _ = try await bands.run()
        return bands.values(plan.result!)
    }
}
