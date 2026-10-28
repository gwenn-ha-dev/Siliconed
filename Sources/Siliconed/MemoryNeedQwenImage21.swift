import Foundation

/// **Qwen-Image-2.1's need** — the render of `Chain.render` for this family,
/// phase by phase, each charged with what its constructors allocate: the same static functions they
/// call, on the configs and tensor tables of the installed maps (`QwenImage21Maps`).
///
///     references (originals) ─ Lanczos ~R² ─ prepared [3, Hᵢ, Wᵢ] ──────────────────────────────── crossing
///          │                                                                                       │
///          ├─ vision  : Qwen3VLVision arena (largest image's patches), every pixel_values, the    ┐
///          │            features done, the tokenizer's tables + the tower's SDPA (queries' copy)  │
///          ├─ text    : TextEncoder arena (S = prompt + image tokens), its staging layers, the    │ one at a
///          │            tokenizer's tables, pixel_values, features + causal SDPA (mask S²) · GEMM │ time:
///          ├─ refs    : the RGBA handed in + the encoder's two heaps (largest reference)          ┤ peak =
///          ├─ denoise : DiT arena (P + S rows, scratch, MLP chunk), K/V arena or file, staging     │ max +
///          │            blocks (K/V in memory only), LoRA slots, reference copies                 │ crossing
///          │            + every SDPA the DiT keeps (masks, head splits, queries' copies) · GEMM   │
///          └─ decode  : the decoder's two heaps and its RGBA, then the RGB copy                    ┘
///                       (a batch: the images already decoded, and every latent crossing until then)
///
/// **What crosses** (alive from its birth to the end of the render, charged to every phase — the
/// earlier phases are overcharged by a few tens of MB, never undercharged): the references as handed
/// and as prepared (`Chain.render` keeps `prepared` to its end), the encoded text (`text`, read by the
/// result), the reference latents, the target latent (and the batch's other latents). **The K/V do
/// not cross here**: the prefill runs inside the first step (`QwenImage21DiT.prefillPending`) and the
/// cache dies with the DiT, before the decoding (`Context.keptDenoiser = nil`); in memory they are the
/// denoising's, on disk nobody's.
///
/// What the formula has to know of the code that an estimate by analogy would miss:
///
///   - **the K/V's place follows the prefix, not the operation** (`QwenImage21CachePolicy.storage`, the
///     one rule the sampler applies too): in memory up to 256 MB (P ≤ 256 rows), on disk beyond. A long
///     prompt (P > 256 text rows) moves a *generation's* K/V to the file — and with it its staging
///     blocks go (below);
///   - **the prompt's tokens are only bounded** (`Request.promptTokens`: its bytes, before any
///     tokenizer): the bound may say « file » for a prompt whose real prefix stays in memory, where the
///     K/V arena and the staging blocks are more than the longer arena the file needs. So **while
///     the real prefix may fall on either side, both placements are charged and the larger kept**
///     (`placements`): the memory one at the longest prefix it holds, the file one at the bound — the
///     need never falls when the prompt grows;
///   - **the staging blocks of the map's tail exist only with the K/V in memory**
///     (`QwenImage21DiT.body`): an edit — or a long generation — streams nothing, whatever the plan's
///     tail says; a generation takes the plan's tail (`Request.tail`), in `TailStream.unitSlots` buffers of the
///     largest tensor's run in the map (`QwenImage21Maps.ditUnitStaging`);
///   - **the scratch is not the prefix**: the prefill runs by groups of segments
///     (`QwenImage21DiT.layout`), so `normed`, `o`, the MLP chunk scale with the largest group; `x`,
///     `keys`, `values` and the RoPE table scale with the whole sequence `P + S`;
///   - **the text rows are the encoder's rows minus the 14 of the system turn** (`_drop_idx`), each
///     image slot is 4 rows of the DiT (one 2×2 group of latent tokens);
///   - **the DiT keeps every SDPA it built** (`attentions`, `headSplits`, by `rows×visible` key): each
///     causal text segment with its mask, each reference segment, each slice of the target cut by heads;
///   - **the modulation is tabulated before the first attention exists** (`tabulateModulation`):
///     `modulation.1` widened to fp32 (`Krea2Exact.widenedBytes`, `d × 4d`) is alive beside the arenas,
///     not beside the attentions — the larger of the two moments is charged;
///   - **the tokenizer lives through the vision and text phases** (`Qwen3VLEncoder` holds it): its two
///     tables (`Tokenizer.tableBytes(file:)`, ~20 MB, from the installed file's counts);
///   - **a generation's half-size DiT** (the spectral schedule, `QwenImage21Sampler`) is built and
///     released before the full one: never both, the full one is the larger — and a batch's next image
///     does not keep the previous full DiT when it builds a half-size one first
///     (`QwenImage21DenoisingModule.keepsDenoiser`);
///   - **the VAEs hand everything back before they return** (`VAEMemory.finish` waits for Metal's
///     ledger): nothing of the encoder is carried into the denoising, and the decoder's RGBA outlives
///     its heaps only by the RGB copy.
extension MemoryNeed {
    static func qwenImage21(_ request: Request) -> MemoryNeed? {
        guard let maps = request.qwenMaps ?? QwenImage21Maps(card: request.card, library: request.library) else { return nil }
        return QwenImage21Need(request, maps: maps).need
    }
}

/// **What Qwen-Image-2.1's formula reads beside the request**: the DiT's and the encoder's configs and
/// tensor tables (their headers, never a tensor), and the tokenizer's table sizes (its file scanned,
/// `Tokenizer.tableBytes(file:)`) — each file read once, by its stamp. A map whose header does not
/// parse makes no maps: the need is unknown and the render refused (`EngineError.memoryNeedUnknown`).
package struct QwenImage21Maps: Sendable {
    package let dit: QwenImage21DiT.Config
    package let encoder: TextEncoder.Config
    package let vision: Qwen3VLVision.Config
    /// The largest layer's staging run in the DiT's map (`Artifact.stagingBytes`): the map's type, read by the tests.
    package let ditLayerStaging: Int
    /// The largest tensor's staging run in the DiT's map (`Artifact.largestRun`): what one buffer of its stream holds.
    package let ditUnitStaging: Int
    /// The largest layer's staging run in the encoder's map: what one of its buffers holds.
    package let encoderLayerStaging: Int
    /// What the tokenizer's two tables map (`Tokenizer.tableBytes`).
    package let tokenizerTables: Int
    /// What widening the largest rotated (int8 convrot) weight allocates beside the arenas
    /// (`Widen.rotationScratchBytes`): the Compact's DiT and encoder, the Light's encoder; 0 otherwise.
    package let encoderRotationScratch: Int, ditRotationScratch: Int

    /// From the headers and the tokenizer's count — pure: the tests hand it headers written in memory.
    /// `nil` when a config does not parse or a map has no tensor table.
    package init?(dit: [String: Any], ditSize: Int = .max, encoder: [String: Any], encoderSize: Int = .max,
                  tokenizerTables: Int) {
        guard let encoderConfig = try? TextEncoder.Config(header: encoder),
              let vision = try? Qwen3VLVision.Config(header: encoder) else { return nil }
        let ditConfig = QwenImage21DiT.Config(header: dit)
        let ditStaging = Artifact.stagingBytes(header: dit, prefixes: (0..<ditConfig.layers).map(QwenImage21DiT.layerPrefix),
                                               fileSize: ditSize)
        let encoderStaging = Artifact.largestRun(header: encoder, prefixes: TextEncoder.layerPrefixes(encoderConfig),
                                                   fileSize: encoderSize)
        guard ditStaging > 0, encoderStaging > 0 else { return nil }
        self.dit = ditConfig; self.encoder = encoderConfig; self.vision = vision
        ditUnitStaging = Artifact.largestRun(header: dit, prefixes: (0..<ditConfig.layers).map(QwenImage21DiT.layerPrefix),
                                             fileSize: ditSize)
        ditLayerStaging = ditStaging; encoderLayerStaging = encoderStaging; self.tokenizerTables = tokenizerTables
        encoderRotationScratch = Widen.rotationScratchBytes(header: encoder)
        ditRotationScratch = Widen.rotationScratchBytes(header: dit, prefixes: (0..<ditConfig.layers).map(QwenImage21DiT.layerPrefix))
    }

    /// The maps this card renders with: its own DiT (an imported one) or the library's installed
    /// version, that version's encoder, the family's tokenizer. `nil` when one cannot be read.
    package init?(card: ModelCard, library: Library = .processWide) {
        let ditPath = card.map ?? library.map(library.ditMap(.qwenImage21))
        let encoderPath = library.map(library.encoderMap(.qwenImage21))
        guard let processor = try? library.component(.qwenImage21, "processor") else { return nil }
        let tokenizerPath = processor + "/tokenizer.json"
        let maps = QwenImage21Maps.installed(MemoryNeed.InstalledFiles(paths: [ditPath, encoderPath, tokenizerPath])) {
            () -> QwenImage21Maps? in
            guard let dit = try? Artifact.header(ditPath), let encoder = try? Artifact.header(encoderPath),
                  let tables = try? Tokenizer.tableBytes(file: tokenizerPath) else { return nil }
            return QwenImage21Maps(dit: dit, ditSize: MemoryNeed.InstalledFiles.size(ditPath), encoder: encoder,
                                   encoderSize: MemoryNeed.InstalledFiles.size(encoderPath), tokenizerTables: tables)
        }
        guard let maps else { return nil }
        self = maps
    }

    /// The maps read, by their files' stamps.
    private static let installed = ShapeMemo<MemoryNeed.InstalledFiles, QwenImage21Maps?>(capacity: 8)
}

/// **The shapes of one Qwen-Image-2.1 render**, derived from a `MemoryNeed.Request` as the engine
/// derives them (`Qwen3VLImages.conditionSize`, the template, the joint sequence), then charged.
package struct QwenImage21Need {
    typealias Term = MemoryNeed.Term

    /// One reference: as handed, as prepared (`QwenImage21Pipeline.reference`), and what it becomes.
    package struct Reference {
        package let original: (width: Int, height: Int)
        package let prepared: (width: Int, height: Int)
        /// The latent grid (`/16`) — also the vision tower's patch grid (patch 16).
        package var latent: (height: Int, width: Int) { (prepared.height / 16, prepared.width / 16) }
        package var patches: Int { latent.height * latent.width }
        /// `<|image_pad|>` tokens: one per 2×2 group of patches.
        package var slots: Int { patches / 4 }
    }

    /// `_drop_idx`: the system turn's tokens (`Qwen3VLPrompt.systemTurn` tokenized), not passed to the DiT.
    package static let dropCount = 14
    /// The template's text tokens around an image: `<imageN><|vision_start|>` before (the separating
    /// space merged in), `<|vision_end|>` after — the oracle's edit-2 layout has 8 text rows before
    /// image 1 (3 of `<|im_start|>user\n`) and 6 between. What `promptTokens` includes per reference.
    package static let tokensBefore = 5, tokensAfter = 1

    package let request: MemoryNeed.Request
    package let maps: QwenImage21Maps
    package let references: [Reference]
    /// The encoder's sequence: the prompt with its template, plus every image token.
    package let encoderTokens: Int
    /// The DiT's text rows: the encoder's rows past the system turn, image slots excluded — at most
    /// (the request's tokens are a bound).
    package let textRows: Int
    /// The fewest text rows any prompt gives: the template with an empty prompt.
    package let fewestTextRows: Int
    package let target: (height: Int, width: Int)
    /// The DiT's joint sequence at `textRows`: text and references in the template's order, the target last.
    package let sequence: QwenImage21Sequence
    package var dit: QwenImage21DiT.Config { maps.dit }
    package let vae = QwenImage21VAE.Config()
    package var vision: Qwen3VLVision.Config { maps.vision }
    /// The P cores the head split runs on.
    package var threads: Int { request.threads }

    /// - `request.promptTokens`: the encoder's tokens **without the image slots** — the template
    ///   (system turn included), the prompt, each image's tags; at least what the template needs.
    package init(_ request: MemoryNeed.Request, maps: QwenImage21Maps) {
        self.request = request
        self.maps = maps
        references = request.references.map { r in
            Reference(original: (r.width, r.height),
                      prepared: Qwen3VLImages.conditionSize(width: r.width, height: r.height,
                                                            resolution: QwenImage21Pipeline.outputResolution))
        }
        let n = references.count
        // `<|im_start|>user\n` (3), each image's tags, and after the last image at least the closing
        // `<|im_end|>\n<|im_start|>assistant\n` (5): what the template has with an empty prompt.
        fewestTextRows = 3 + n * (QwenImage21Need.tokensBefore + QwenImage21Need.tokensAfter) + 5
        textRows = max(request.promptTokens - QwenImage21Need.dropCount, fewestTextRows)
        encoderTokens = textRows + QwenImage21Need.dropCount + references.reduce(0) { $0 + $1.slots }
        target = (request.height / 16, request.width / 16)
        sequence = QwenImage21Need.jointSequence(textRows: textRows, references: references, target: target)
    }

    /// **The joint sequence's layout** for `textRows` text rows: the user header, each image between its
    /// tags, the rest of the text after the last image, then the target's slots.
    package static func jointSequence(textRows: Int, references: [Reference],
                                 target: (height: Int, width: Int)) -> QwenImage21Sequence {
        // Its inputs: the text rows, each reference's latent grid, the target's (`sequences`).
        let key = [textRows, target.height, target.width] + references.flatMap { [$0.latent.height, $0.latent.width] }
        return sequences(key) {
            var slots: [Bool] = []
            func text(_ count: Int) { slots += [Bool](repeating: false, count: count) }
            text(3)
            for (i, reference) in references.enumerated() {
                text(QwenImage21Need.tokensBefore)
                slots += [Bool](repeating: true, count: reference.slots)
                if i < references.count - 1 { text(QwenImage21Need.tokensAfter) }
            }
            text(textRows - slots.filter { !$0 }.count)
            slots += [Bool](repeating: true, count: target.height * target.width / 4)
            return try! QwenImage21Sequence(
                slots: slots, images: references.map { .init(height: $0.latent.height, width: $0.latent.width) }
                    + [.init(height: target.height, width: target.width)])
        }
    }

    /// The joint sequences built, by their layout's inputs: `QwenImage21Sequence` lays out every row
    /// and its RoPE position (a three-reference edit at 1024×1536, 12 ms in a debug build), and the
    /// need reads only its segments and counts.
    private static let sequences = ShapeMemo<[Int], QwenImage21Sequence>()

    // ── the phases ───────────────────────────────────────────────────────────────────────────

    /// The vision tower, one image at a time in an arena sized for the largest (its `Attention` built
    /// per image and dropped with it); every image's `pixel_values` and the features of those done are
    /// held (they feed the language model), and the tokenizer the encoder holds.
    package var visionPhase: MemoryNeed.Phase? {
        guard let largest = references.map(\.patches).max() else { return nil }
        let (d, i, m, o) = (vision.hidden, vision.intermediate, vision.merged, vision.output)
        return .init(kind: .vision, terms: [
            Term("vision arena", Qwen3VLVision.arenaBytes(config: vision, patches: largest)),
            Term("pixel_values", pixelValues), Term("vision features", features),
            Term("tokenizer tables", maps.tokenizerTables),
            Term("encoder rotation scratch", maps.encoderRotationScratch),
        ] + MemoryNeed.attention("vision", heads: vision.heads, rows: largest, keys: largest, headDim: vision.headDim,
                                 causal: false, split: false, threads: threads)
          + MemoryNeed.gemm("vision", products: [(largest, vision.patchValues, d), (largest, d, 3 * d), (largest, d, d),
                                                 (largest, d, i), (largest, i, d), (largest / 4, m, m), (largest / 4, m, o)]))
    }

    /// The language model on the whole sequence, with its staging layers (`Request.encoderSlots`).
    package var textPhase: MemoryNeed.Phase {
        let s = encoderTokens, e = maps.encoder
        let (h, a, kv, i) = (e.hidden, e.attentionWidth, e.kvWidth, e.intermediate)
        return .init(kind: .text, terms: [
            Term("encoder arena", TextEncoder.arenaBytes(config: e, sequence: s)),
            Term("encoder staging", TextEncoder.stagingSlots(granted: request.encoderSlots) * maps.encoderLayerStaging),
            Term("pixel_values", pixelValues), Term("vision features", features),
            Term("tokenizer tables", maps.tokenizerTables),
            Term("encoder rotation scratch", maps.encoderRotationScratch),
        ] + MemoryNeed.attention("encoder", heads: e.heads, rows: s, keys: s, headDim: e.headDim, causal: true,
                                 split: false, threads: threads)
          + MemoryNeed.gemm("encoder", bounded: [(s, h, a), (s, h, kv), (s, a, h), (s, h, i), (s, i, h)]))
    }

    /// The pixels of the largest prepared reference: what the encoder is planned for.
    private var largestReference: (width: Int, height: Int)? {
        references.map(\.prepared).max { $0.width * $0.height < $1.width * $1.height }
    }

    /// The VAE encoder, one reference at a time (`QwenImage21EncodingModule`, which dies on return):
    /// the RGBA array the module builds and hands in (`rgba`, `[4, H, W]`), and the encoder's two heaps
    /// and host copies (`QwenImage21VAEEncoder.plan`).
    package var referencesPhase: MemoryNeed.Phase? {
        guard let largest = largestReference else { return nil }
        return .init(kind: .references, terms: [Term("rgba", vae.inChannels * largest.width * largest.height * 4)]
            + MemoryNeed.vae("Qwen encoder", QwenImage21VAEEncoder.plan(config: vae, height: largest.height,
                                                                         width: largest.width)))
    }

    /// Where the conditions' K/V live at the bound (`QwenImage21CachePolicy`, the sampler's rule).
    package var storage: QwenImage21DiT.CacheStorage { QwenImage21CachePolicy.storage(prefix: sequence.prefix) }

    /// **Every placement the real prefix may take, with the text rows charged for it**: the bound's,
    /// and — when the bound's prefix is on disk but the fewest text rows' would stay in memory — the
    /// memory one at the longest prefix it holds. The real token count lies between the two ends; each
    /// placement's need grows with the rows, so its longest case bounds it.
    package var placements: [(storage: QwenImage21DiT.CacheStorage, textRows: Int)] {
        var cases = [(storage, textRows)]
        let imageRows = sequence.prefix - textRows
        let longest = QwenImage21CachePolicy.largestMemoryPrefix - imageRows
        if storage == .file, longest >= fewestTextRows,
           QwenImage21CachePolicy.storage(prefix: longest + imageRows) == .memory {
            cases.append((.memory, min(longest, textRows)))
        }
        return cases
    }

    /// The full-size DiT (the half-size one, a generation's first steps, is gone before it is built):
    /// the larger over the placements the prefix may take (`placements`).
    package var denoisingPhase: MemoryNeed.Phase {
        placements.map { placement in
            denoisingPhase(sequence: placement.textRows == textRows ? sequence
                                : QwenImage21Need.jointSequence(textRows: placement.textRows, references: references, target: target),
                           textRows: placement.textRows, storage: placement.storage)
        }.max { $0.bytes < $1.bytes }!
    }

    /// The full-size DiT on `sequence`, its K/V `storage`.
    package func denoisingPhase(sequence: QwenImage21Sequence, textRows: Int,
                                storage: QwenImage21DiT.CacheStorage) -> MemoryNeed.Phase {
        let p = sequence.prefix, s = sequence.target, all = sequence.count
        let (_, scratch) = QwenImage21DiT.layout(sequence, fullPass: storage == .recompute)
        let rank = request.loraRanks.reduce(0, +)
        let largestImage = sequence.images.map { $0.height * $0.width }.max() ?? 0
        let (d, h) = (dit.dim, dit.hidden)
        var terms = [
            Term("DiT arena", QwenImage21DiT.arenaBytes(config: dit, rows: all, prefix: p, scratch: scratch,
                                                        largestImage: largestImage, rank: rank,
                                                        fullPass: storage == .recompute)),
            Term("K/V arena", p > 0 && storage == .memory ? QwenImage21DiT.cacheArenaBytes(config: dit, prefix: p) : 0),
            // The map's stream buffers: only with the K/V in memory (`QwenImage21DiT.body`).
            Term("map staging", storage == .memory && request.tail.fraction > 0
                ? TailStream.unitSlots * maps.ditUnitStaging : 0),
            Term("DiT rotation scratch", maps.ditRotationScratch),
        ]
        // Two slots of the user's stack, on the 7 Linears of a block (`StreamedLoRA`), and, with more
        // than one LoRA, the `down` staging of the module a slot fills (`LoRA.downBytes`).
        if rank > 0 {
            let modules: [String: (k: Int, n: Int, rank: Int)] = [
                "attn.to_q": (d, d, rank), "attn.to_k": (d, d, rank), "attn.to_v": (d, d, rank),
                "attn.to_out.0": (d, d, rank), "img_mlp.gate_layer": (d, h, rank), "img_mlp.proj": (d, h, rank),
                "img_mlp.out": (h, d, rank)]
            terms.append(Term("LoRA slots", 2 * StreamedLoRA.layout(modules).bytes))
            if request.loraRanks.count > 1 {
                terms.append(Term("LoRA down staging", LoRA.downBytes(k: h, rank: request.loraRanks.max() ?? 0)))
            }
        }
        // The sampler's copies of the reference latents (`Pages`), the velocity, the modulation table.
        terms.append(Term("sampler", references.reduce(0) { $0 + Arena.capacity(for: vae.zDim * $1.patches * 4) }
            + vae.zDim * s * 4 + QwenImage21Pipeline.defaultSteps * dit.modulation * 4
            // The solver's history (`Solver.history`): latents of the target's size.
            + request.solver.historyBytes(count: vae.zDim * s)))
        // The two moments past the arenas: the modulation tabulated (`modulation.1` widened, and under a
        // LoRA its `down`/`up` in `Pages`), then every attention and the GEMM. The larger is charged.
        let attentions = ditAttentions(sequence: sequence, scratch: scratch)
        let mlpRows = QwenImage21DiT.mlpRows(scratch: scratch, hidden: h)
        // The images' rows are exact; the scratch (the prefill's groups hold text) and the text's follow
        // the tokens' bound.
        let running = attentions + MemoryNeed.gemm("DiT", products: [(largestImage, dit.channels, d)],
                                                   bounded: [(scratch, d, d), (mlpRows, d, h), (mlpRows, h, d),
                                                             (textRows, dit.context, d)],
                                                   rank: rank, ane: request.neuralEngine)
        let widened = Krea2Exact.widenedBytes(inputs: d, outputs: 4 * d)
            + (rank > 0 ? Arena.capacity(for: d * rank * 4) + Arena.capacity(for: rank * 4 * d * 4) : 0)
        let runningBytes = running.reduce(0) { $0 + $1.bytes }
        terms += running
        terms.append(Term("modulation widening beyond the attentions", max(0, widened - runningBytes)))
        return .init(kind: .denoising, terms: terms)
    }

    /// The decoder's two heaps and the RGBA it returns (`QwenImage21VAEDecoder.plan`); then, its heaps
    /// handed back, the RGB copy `QwenImage21DecodingModule` takes of the RGBA (`Array(rgba.prefix(3·H·W))`)
    /// — the larger of the two moments; and a batch's images already decoded.
    package var decodingPhase: MemoryNeed.Phase {
        let (H, W) = (vae.factor * target.height, vae.factor * target.width)
        let band = request.minimumBands ? min(QwenImage21VAEDecoder.bandPixels, MemoryPlan.qwenMinimumBand)
                                        : QwenImage21VAEDecoder.bandPixels
        let plan = QwenImage21VAEDecoder.plan(config: vae, latentHeight: target.height, latentWidth: target.width,
                                              bandPixels: band)
        let rgb = 3 * H * W * 4
        return .init(kind: .decoding, terms: MemoryNeed.vae("Qwen decoder", plan)
            + [Term("RGB copy beyond the heaps", max(0, rgb - plan.registers - plan.workspace)),
               MemoryNeed.batchImages(request)])
    }

    /// The references as handed and as prepared, the encoded text, the reference latents, the target
    /// latent — and the batch's other targets, waiting for their decoding. Each image is charged its three
    /// fp32 planes **and its straight RGBA** (`ImageRGB.rgba`, 4 bytes a pixel: one plane more), which only
    /// an image with transparency carries — the request does not say which, so every one is charged it.
    package var crossing: Int {
        let images = references.reduce(0) {
            $0 + (3 + 1) * ($1.original.width * $1.original.height + $1.prepared.width * $1.prepared.height)
        }
        let rows = encoderTokens - QwenImage21Need.dropCount
        let targetLatent = vae.zDim * target.height * target.width
        let latents = references.reduce(0) { $0 + vae.zDim * $1.patches } + targetLatent
        return (images + rows * maps.encoder.hidden + latents) * 4 + rows
            + MemoryNeed.batchLatents(request, latent: targetLatent * 4)
    }

    package var need: MemoryNeed {
        MemoryNeed(phases: [visionPhase, textPhase, referencesPhase, denoisingPhase, decodingPhase].compactMap { $0 },
                   crossing: crossing, base: request.base)
    }

    // ── the pieces ───────────────────────────────────────────────────────────────────────────

    /// `pixel_values` of every reference (`Qwen3VLImages.Pixels`, in `Pages`).
    private var pixelValues: Int {
        references.reduce(0) { $0 + Qwen3VLImages.Pixels.valueBytes(gridHeight: $1.latent.height, gridWidth: $1.latent.width) }
    }
    /// The tower's output per image (`Qwen3VLVision.Features`, in `Pages`): the merger's and the
    /// deepstack mergers' `[slots, output]`.
    private var features: Int {
        references.reduce(0) { $0 + Qwen3VLVision.featureBytes(tokens: $1.slots, output: vision.output,
                                                               levels: vision.deepstack.count) }
    }

    /// **Every SDPA the DiT keeps** (`attentions`, `headSplits`: one per cache key, alive to the end of
    /// the denoising), enumerated as `QwenImage21DiT.attend` builds them — each slice's rows and visible
    /// keys, the causal text segments with their mask, the target cut by heads.
    private func ditAttentions(sequence: QwenImage21Sequence, scratch: Int) -> [Term] {
        let heads = dit.heads, headDim = dit.headDim, p = sequence.prefix, all = sequence.count
        var terms: [Term] = []
        var built = Set<String>()
        func attend(count: Int, keys: Int, causal: Bool, target: Bool) {
            let slices = QwenImage21DiT.attentionSlices(heads: heads, queries: count, keys: keys)
            for piece in 0..<slices {
                let end = (piece + 1) * count / slices, rows = end - piece * count / slices
                let visible = causal ? keys - count + end : keys
                let split = target && HeadSplitAttention.applies(sequence: rows, heads: heads, headDim: headDim)
                // `attentions` and `headSplits` are two dictionaries: the same key may live in both.
                let key = "\(rows)×\(visible)\(causal ? "c" : "")"
                guard built.insert((split ? "s" : "a") + key).inserted else { continue }
                terms += MemoryNeed.attention("DiT \(key)", heads: heads, rows: rows, keys: visible, headDim: headDim,
                                              causal: causal, split: split, threads: threads)
            }
        }
        for segment in sequence.segments {
            attend(count: segment.end - segment.start, keys: segment.end, causal: segment.isText, target: false)
        }
        attend(count: all - p, keys: all, causal: false, target: true)
        return terms
    }
}
