import Foundation

/// **The classes of computation of a render, their executions and their FLOP** — what the solver is given to plan (`CapacitySolver.solve`)
/// and what the render's predicted-versus-measured check compares with (`RenderLoop`). Pure: it reads the five families' `Config`
/// defaults (no header, no weights), a version (the weights' format), a format (the pixels) and "Faster Rendering".
///
///     Request (family, version, W×H, Faster Rendering, prompt tokens, steps)
///        │
///        ├─ passes      k evaluations on the half grid, then steps − k on the full one (k = the family's
///        │              `halfSizeSteps`, only where `Spectral.divisible` allows the half grid); rows of a pass
///        │              = image tokens (W/16 · H/16) [+ the text rows the family joins to them]
///        ├─ denoising   per evaluation: every product k×n at its real height, every attention (S, keys, heads,
///        │              kv heads, head dim, crossed or not); once per render: the products of the conditions
///        │              (Anima's cross k/v at 512 rows, Qwen's prefill at the prefix's rows)
///        └─ encoding    the text encoder's products at m = the prompt's tokens (+ the template's): once per render
///
///     class = (phase, shape)          shape = product(k, n, rows, weight format) | attention(S, keys, heads, kv, dim, cross)
///     executions[family]              how many times a render of that family runs it
///     FLOP                            product 2·k·n·m ;  attention 4·S·keys·heads·dim  (QKᵀ and PV)
///
/// What sets it apart — what a catalogue written by analogy with "one product per layer" would get wrong:
///
///   - **The heights are the render's, not a grid's**: the image rows plus each family's own text rows —
///     Z-Image 19 template + prompt tokens rounded up to 32 (1 056 · 4 128 · 6 176 at 512² · 1024² · 1024×1536),
///     Krea 2 +16 (1 040 · 4 112 · 6 160), MOSS-Image +21 (1 045 · 4 117 · 6 165), Qwen-Image-2.1 and Anima the
///     image alone (the text is a prefix of the keys, or a cross-attention). Faster Rendering adds the half grid's
///     (about 1 050 · 1 560 rows), at the family's own count of evaluations (Z-Image, Qwen, Anima 3; Krea 2 4; MOSS 1).
///   - **The 8-bit blocks are not the whole DiT**: a Compact quantizes the layers' projections, and leaves as published (bf16, or fp32: the engine's record
///     says `bf16` for both) the embedders, Z-Image's two refiners and context refiners, Krea 2's text fusion, MOSS-Image's text blocks. Those are
///     classes of the Standard's format inside a Compact render — 4 % of Z-Image's seconds, which a catalogue that reads the version's format everywhere
///     sees as « not carried ».
///   - **MOSS-Image's layers run q, k, v and o apart** (four products of 3 072², not a fused QKV of 9 216 — that one is the text blocks'), its gate and
///     up fused (3 072 × 16 384), the text blocks' up and gate apart.
///   - **What a DiT computes from the conditions runs once per grid**, not once per render: each grid has its own DiT, built and run apart, so Anima's
///     cross k/v and Qwen-Image-2.1's prefill (19 rows through the 32 layers, `txt_in` first) run twice under Faster Rendering. Krea 2's text
///     fusion (twelve hidden states per token, then `txt_in`) runs once.
///   - **`Render.tokens` is the family's own count** (`promptTokens(_:renderTokens:)`): the rows joined to the image for Z-Image and Qwen-Image-2.1, the
///     rows once the template is removed for Krea 2, the prompt's for the others.
///   - **Qwen-Image-2.1's MLP runs by row chunks** (`QwenImage21DiT.mlpRows`): a 1024×1536 target is two products of
///     3 072 rows, not one of 6 144 — the class is the chunk's.
///   - **A weight format is part of a product's shape**: the same k×n in bf16 (Standard) and in int8 (Compact) are two
///     classes, so a merge across families of different versions never adds what no route could treat alike. K-quants
///     (Light) have a single route, the widening: they carry no `WeightKind`.
///   - **Merged across families, executions stay per family**: a class common to two families (the same k×n at the same
///     height) is one class with two executions counts.
///   - **A LoRA's two products are classes** (`loraDown`, `loraUp`: `x · down` at `k → r`, `mid · up` at `r → n`, fp32, the stack's
///     expansion), on every layer projection the stack touches — MOSS-Image's DMD always (`mossTurboRank`), a request's LoRAs
///     (`Request.loraRank`) on Z-Image, Anima and Krea 2; Qwen-Image-2.1's stack, merged where that is cheaper, is not carried.
///   - **What is not a class**: the modulation and time-embedding gemv (m = 1, the CPU), the norms and elementwise
///     kernels, the VAE (one route), the weights' widening and reading, attentions over fewer than 64 rows (no route
///     decides them), the encoder's attention and DeltaNet recurrence (m < 64), the embedders other than Z-Image's, Anima's adapter and Krea 2's text fusion, W_b/W_a of the
///     MOSS encoder. They are small (each under 1 % of a render), or have no route to choose among.
///   - **Unread from code**: Krea 2's text encoder runs up to its last tap (36 layers is the Qwen3-VL-4B text model's
///     count, an upper bound); MOSS's encoder FFN width (12 288) is the Qwen3.5-9B card's, not read in the code. Which weights a Light leaves
///     as published is not read either: the Light keeps its version's format everywhere.
///   - **The id of a class is the render's and the ledger's** (`product:K×N@M`, `~int8`, `~q8`, `~q4k` for the weight's form;
///     `attention:S/H×D`): a self- and a cross-attention of the same S, heads and head dim share one — what the ledger can tell
///     them by — and the catalogue keeps both so that their executions add.
package enum CapacityClasses {
    package enum Phase: String, Sendable, Hashable, CaseIterable, Codable {
        /// The text encoder, once per render.
        case encoding
        /// The DiT's evaluations, and what it computes once from the conditions.
        case denoising
    }

    /// The format a weight is published in, as the version carries it.
    package enum WeightFormat: String, Sendable, Hashable, CaseIterable, Codable {
        case bf16, int8, q8
        /// Q4_K to Q6_K: a single route (widening on the CPU).
        case q4k
        /// **A LoRA's two products are classes of the plan like any other**, their operand not a published weight but the stack's
        /// fp32 expansion (`CacheLoRA`): `loraDown` is `mid = x · down` (`m × k · r`, written), `loraUp` is `C += mid · up`
        /// (`m × r · n`, accumulated into the base product's output — what the CPU engines' rows pay a pass of memory for). Nothing
        /// to prepare, a single route (fp32, widened): the GPU, the AMX, the ANE's fp16 where the version allows it.
        case loraDown, loraUp

        /// **The convrot group of the format's rotation**, `nil` for none: every int8 DiT of the catalogue is Comfy's `int8_tensorwise`
        /// convrot of group 256 (the headers of the five Compact maps, checked), so the plan reasons on 256 — what the ANE's int8
        /// program needs to know to cut K (`NeuralEngine.readsInt8`) and the memory need to size its rotated row. A tensor of
        /// another group is not refused here: the engine reads the tensor's own and declines what its program cannot cut.
        package var rotationGroup: Int? { self == .int8 ? Widen.largestRotationGroup : nil }

        /// The weight's kind, `nil` for the K-quants (no native form).
        package var kind: WeightKind? {
            switch self {
            case .bf16: return .bf16
            case .int8: return .int8
            case .q8: return .q8
            case .q4k, .loraDown, .loraUp: return nil
            }
        }

        /// The operand of a LoRA's product (`loraDown`, `loraUp`): fp32 as the stack expands it, no preparation.
        package var isLoRA: Bool { self == .loraDown || self == .loraUp }
    }

    // MARK: - Versions

    /// **The DiT's weight format of a version**: Standard bf16; Compact int8 convrot (every family: Z-Image's and Anima's were
    /// GGUF Q8_0 until 0.20); Light Q4_K (no Light for MOSS-Image: `nil`).
    package static func weightFormat(_ family: Family, _ variant: Variant) -> WeightFormat? {
        switch (variant, family) {
        case (.standard, _): return .bf16
        case (.compact, _): return .int8
        case (.light, .mossImage): return nil
        case (.light, _): return .q4k
        }
    }

    /// **The text encoder's**: Standard bf16; the Compact's and the Light's int8 convrot, Anima's staying bf16.
    package static func encoderFormat(_ family: Family, _ variant: Variant) -> WeightFormat? {
        guard weightFormat(family, variant) != nil else { return nil }
        return variant == .standard || family == .anima ? .bf16 : .int8
    }

    // MARK: - Request

    package struct Request: Sendable, Equatable {
        package var family: Family
        package var variant: Variant
        /// Pixels.
        package var width: Int
        package var height: Int
        /// « Faster Rendering » (`Request.fasterRendering`): the family's first evaluations on the half grid.
        package var fasterRendering: Bool
        /// The prompt's tokens (the benchmarks' prompt: 11).
        package var promptTokens: Int
        package var steps: Int
        /// Include the text encoder (a new prompt); an « image suivante » does not run it.
        package var includesEncoder: Bool
        /// The half-grid evaluations a render actually did, when it is not the family's rule (`SILICONED_SPECTRAL`);
        /// `nil`: the family's own count.
        package var halfSteps: Int?
        /// **The cumulative rank of the LoRAs the request adds** to the family's own (MOSS-Image's DMD is always there, `mossTurboRank`):
        /// each layer projection then carries the two products of the stack at this rank (`loraDown`, `loraUp`). 0: none.
        package var loraRank: Int
        /// **The format the installed DiT map really holds** (`Artifact.weightFormat(header:)`), when it is known; `nil`: the version's
        /// own (`weightFormat(_:_:)`). The classes follow the tensors, never the name of the version: our 4-bit Light (SQ) is
        /// decoded to int8 at each product and is the Compact's classes, the GGUF Light's Q4_K is its own.
        package var ditFormat: WeightFormat?

        package init(family: Family, variant: Variant = .standard, width: Int = 1024, height: Int = 1024,
                     fasterRendering: Bool = true, promptTokens: Int = 11, steps: Int = 8, includesEncoder: Bool = true,
                     halfSteps: Int? = nil, loraRank: Int = 0, ditFormat: WeightFormat? = nil) {
            self.ditFormat = ditFormat
            self.family = family; self.variant = variant; self.width = width; self.height = height
            self.fasterRendering = fasterRendering; self.promptTokens = promptTokens; self.steps = steps
            self.includesEncoder = includesEncoder; self.halfSteps = halfSteps; self.loraRank = loraRank
        }

        /// The request with the version the installed tensors make (`ditFormat`): the variant is what the classes are written by
        /// (the refiners left as published, the encoder's form), so a map that is not its version's usual form is read as the
        /// version that has its form. A family without that version stays as it is.
        fileprivate var byTensors: Request {
            guard let ditFormat else { return self }
            var r = self
            switch ditFormat {
            case .bf16: r.variant = .standard
            case .q4k: r.variant = .light
            case .int8, .q8: r.variant = .compact
            case .loraDown, .loraUp: break
            }
            return r
        }
    }

    // MARK: - Classes

    /// **The rank of MOSS-Image's DMD LoRA** (`lodestones/MOSS-Image-SFT-8-steps-dmd`, 252 modules), applied at every evaluation of every
    /// render: its two products are classes of the family whatever the request carries. (The map's own rank is read by the memory
    /// need, `MossMaps.turboRank`; the catalogue reads no header.)
    package static let mossTurboRank = 128

    /// **The rank of the stack a family's DiT runs under** in this request: MOSS-Image's always-on LoRA plus the request's; the
    /// request's alone elsewhere. Qwen-Image-2.1 counts none: its stack is merged into the widened weight where that costs fewer
    /// FLOPs (`LoRA.merges`, by rows) and is a product of the plan only after the GEMM, a split of the classes by height that the
    /// catalogue does not carry.
    package static func stackRank(_ r: Request) -> Int {
        switch r.family {
        case .mossImage: return mossTurboRank + r.loraRank
        case .qwenImage21: return 0
        default: return r.loraRank
        }
    }

    /// `product:KxN@M`, then `~int8`, `~q8`, `~q4k`, `~loraDown`, `~loraUp` for a weight not in bf16.
    package static func productID(k: Int, n: Int, m: Int, format: WeightFormat) -> String {
        "product:\(k)x\(n)@\(m)" + (format == .bf16 ? "" : "~" + format.rawValue)
    }

    /// `attention:S/HxD`.
    package static func attentionID(sequence: Int, heads: Int, headDim: Int) -> String { "attention:\(sequence)/\(heads)x\(headDim)" }

    package enum Shape: Hashable, Sendable {
        case product(k: Int, n: Int, rows: Int, format: WeightFormat)
        case attention(sequence: Int, keys: Int, heads: Int, kvHeads: Int, headDim: Int, cross: Bool)

        /// FLOP of one execution.
        package var flop: Double {
            switch self {
            case let .product(k, n, rows, _): return 2 * Double(k) * Double(n) * Double(rows)
            case let .attention(s, keys, heads, _, dim, _): return 4 * Double(s) * Double(keys) * Double(heads) * Double(dim)
            }
        }
    }

    package struct Class: Sendable, Equatable {
        package var phase: Phase
        package var shape: Shape
        /// Executions per render, by family.
        package var executions: [Family: Int]

        package init(phase: Phase, shape: Shape, executions: [Family: Int]) {
            self.phase = phase; self.shape = shape; self.executions = executions
        }

        package var flop: Double { shape.flop }

        /// The class's name in a render's record and its prediction.
        package var id: String {
            switch shape {
            case let .product(k, n, rows, format): return CapacityClasses.productID(k: k, n: n, m: rows, format: format)
            case let .attention(s, _, heads, _, dim, _): return CapacityClasses.attentionID(sequence: s, heads: heads, headDim: dim)
            }
        }
    }

    private struct Key: Hashable { var phase: Phase; var shape: Shape }

    /// Accumulates the executions of one family's render.
    private struct Builder {
        var order: [Key] = []
        var counts: [Key: Int] = [:]

        mutating func add(_ phase: Phase, _ shape: Shape, _ n: Int) {
            guard n > 0 else { return }
            let key = Key(phase: phase, shape: shape)
            if counts[key] == nil { order.append(key) }
            counts[key, default: 0] += n
        }
        mutating func product(_ phase: Phase, _ k: Int, _ n: Int, _ rows: Int, _ format: WeightFormat, times: Int) {
            add(phase, .product(k: k, n: n, rows: rows, format: format), times)
        }
        /// Attentions over fewer than 64 rows are no class.
        mutating func attention(_ s: Int, keys: Int, heads: Int, kv: Int, dim: Int, cross: Bool = false, times: Int) {
            guard s >= 64 else { return }
            add(.denoising, .attention(sequence: s, keys: keys, heads: heads, kvHeads: kv, headDim: dim, cross: cross), times)
        }
    }

    // MARK: - Rows of a render

    /// Image tokens of a `width × height` render: 16 × 16 pixels each, for the five families.
    package static func imageTokens(width: Int, height: Int) -> Int { (width / 16) * (height / 16) }

    /// Pixels per latent cell: 8 for the 8-px VAEs, 16 for Qwen-Image-2.1's and MOSS-Image's.
    static func latentFactor(_ family: Family) -> Int {
        switch family {
        case .qwenImage21, .mossImage: return 16
        default: return 8
        }
    }

    /// The first evaluations of Faster Rendering on the half grid, per family (`…DenoisingModule.halfSizeSteps`).
    package static func halfSizeSteps(_ family: Family) -> Int {
        switch family {
        case .zImage: return ZImageDenoisingModule.halfSizeSteps
        case .qwenImage21: return QwenImage21DenoisingModule.halfSizeSteps
        case .anima: return AnimaDenoisingModule.halfSizeSteps
        case .krea2: return Krea2DenoisingModule.halfSizeSteps
        case .mossImage: return MossImageDenoisingModule.halfSizeSteps
        }
    }

    /// **The passes of a render**: `(image tokens, evaluations)`, the half grid's first, the full grid's last.
    package static func passes(_ r: Request) -> [(tokens: Int, evaluations: Int)] {
        let factor = latentFactor(r.family)
        let (height, width, floor) = (r.height / factor, r.width / factor, Spectral.floor(factor: factor))
        // A render that did its half-size evaluations says how many (`halfSteps`); a catalogue asked for a request applies the
        // engine's own rule (`Spectral.reducedEvaluations`) — its step count taken as the family's recipe.
        let k = r.halfSteps.map { r.fasterRendering && Spectral.divisible(height: height, width: width, by: 2, floor: floor)
                ? min($0, max(r.steps - 1, 0)) : 0 }
            ?? Spectral.reducedEvaluations(faster: r.fasterRendering, height: height, width: width, steps: r.steps, start: 0,
                                           withLoRA: false, defaultSteps: r.steps, halfSizeSteps: halfSizeSteps(r.family),
                                           evaluations: max(r.steps, 1), floor: floor)
        var result: [(Int, Int)] = []
        if k > 0 { result.append((imageTokens(width: r.width / 2, height: r.height / 2), k)) }
        result.append((imageTokens(width: r.width, height: r.height), r.steps - k))
        return result
    }

    /// The text rows the family joins to the image's in the DiT's products (or keeps as a prefix: Qwen-Image-2.1).
    package static func textRows(_ family: Family, promptTokens p: Int) -> Int {
        switch family {
        case .zImage: return ((p + 8) + 31) / 32 * 32          // 19 rows, padded to a multiple of `DiT.Config.seqMultiple`
        case .qwenImage21: return p + 8                         // the prefix, in the keys
        case .anima: return 0                                   // cross-attention over 512 padded rows
        case .krea2: return p + 5                               // 16 rows once the template's 34 are removed
        case .mossImage: return p + 10                          // 21 rows
        }
    }

    /// **The prompt's tokens as this catalogue counts them, from a render's `Render.tokens`** — which each family's text module reports in its
    /// own sense (« for display »): Z-Image's and Qwen-Image-2.1's are the rows the DiT joins (template included, `p + 8`), Krea 2's the rows
    /// once the template's are removed (`p + 5`), MOSS-Image's and Anima's the prompt's. Asked as the render's own, the catalogue put Krea 2's
    /// attention at 4 117 rows when the engine ran it at 4 112.
    package static func promptTokens(_ family: Family, renderTokens t: Int) -> Int {
        switch family {
        case .zImage, .qwenImage21: return max(1, t - 8)
        case .krea2: return max(1, t - 5)
        case .anima, .mossImage: return t
        }
    }

    /// **The prompt's tokens that give `rows` text rows** (the exact inverse of `textRows`, for a DiT that knows its rows and not the prompt).
    package static func promptTokens(_ family: Family, textRows rows: Int) -> Int {
        switch family {
        case .zImage, .qwenImage21: return max(1, rows - 8)
        case .krea2: return max(1, rows - 5)
        case .mossImage: return max(1, rows - 10)
        case .anima: return 1
        }
    }

    // MARK: - Encoders

    /// A text encoder's dimensions (published; the maps carry them in their headers).
    private struct Encoder {
        var hidden, heads, kvHeads, headDim, intermediate, layers, template: Int
    }

    private static func encoder(_ family: Family) -> Encoder? {
        switch family {
        case .zImage: return Encoder(hidden: 2560, heads: 32, kvHeads: 8, headDim: 128, intermediate: 9728, layers: 35, template: 8)
        case .qwenImage21: return Encoder(hidden: 4096, heads: 32, kvHeads: 8, headDim: 128, intermediate: 12288, layers: 36, template: 22)
        case .anima: return Encoder(hidden: 1024, heads: 16, kvHeads: 8, headDim: 128, intermediate: 3072, layers: 28, template: 0)
        case .krea2: return Encoder(hidden: 2560, heads: 32, kvHeads: 8, headDim: 128, intermediate: 9728, layers: 36, template: 39)
        case .mossImage: return nil
        }
    }

    private static func encoding(_ b: inout Builder, _ r: Request, _ format: WeightFormat) {
        func layerProducts(_ b: inout Builder, rows: Int, hidden: Int, qWidth: Int, kvWidth: Int, ffn: Int, layers: Int) {
            b.product(.encoding, hidden, qWidth, rows, format, times: layers)
            b.product(.encoding, hidden, kvWidth, rows, format, times: 2 * layers)
            b.product(.encoding, qWidth, hidden, rows, format, times: layers)
            b.product(.encoding, hidden, ffn, rows, format, times: 2 * layers)
            b.product(.encoding, ffn, hidden, rows, format, times: layers)
        }
        if let e = encoder(r.family) {
            layerProducts(&b, rows: r.promptTokens + e.template, hidden: e.hidden, qWidth: e.heads * e.headDim,
                          kvWidth: e.kvHeads * e.headDim, ffn: e.intermediate, layers: e.layers)
            return
        }
        // MOSS-Image's Qwen3.5-9B: 24 Gated DeltaNet layers and 8 full attentions, h 4096, FFN 12288.
        let rows = r.promptTokens + 10, h = 4096, ffn = 12288
        b.product(.encoding, h, 2 * h, rows, format, times: 24)              // W_qkv
        b.product(.encoding, h, h, rows, format, times: 24 * 2 + 8)          // W_z, W_out ; full attention's o
        b.product(.encoding, h, 2 * h, rows, format, times: 8)               // q with its gate
        b.product(.encoding, h, h / 4, rows, format, times: 2 * 8)           // k, v
        b.product(.encoding, h, ffn, rows, format, times: 2 * 32)
        b.product(.encoding, ffn, h, rows, format, times: 32)
    }

    // MARK: - DiTs

    /// The format of the weights a version leaves as published, beside the 8-bit blocks (the embedders, the refiners, Krea 2's text
    /// fusion, MOSS-Image's text blocks: bf16 or fp32 in the Compact maps, read as `bf16` by the engine's record). Read from the five
    /// Compact maps' headers; a Light's (Q4_K from a GGUF) is not read and stays the version's own.
    private static func plainFormat(_ r: Request, _ f: WeightFormat) -> WeightFormat { r.variant == .compact ? .bf16 : f }

    /// One evaluation's classes at `tokens` image rows, `times` evaluations.
    private static func evaluation(_ b: inout Builder, _ r: Request, _ f: WeightFormat, tokens t: Int, times: Int) {
        let text = textRows(r.family, promptTokens: r.promptTokens)
        let plain = plainFormat(r, f)
        let rank = stackRank(r)
        /// The two products of the LoRA stack on a projection (`k → n`), after its product.
        func lora(_ k: Int, _ n: Int, _ rows: Int, _ count: Int) {
            guard rank > 0 else { return }
            b.product(.denoising, k, rank, rows, .loraDown, times: count * times)
            b.product(.denoising, rank, n, rows, .loraUp, times: count * times)
        }
        /// A product of the version's 8-bit blocks.
        func p(_ k: Int, _ n: Int, _ rows: Int, _ count: Int) {
            b.product(.denoising, k, n, rows, f, times: count * times); lora(k, n, rows, count)
        }
        /// A product by a weight the version leaves as published.
        func q(_ k: Int, _ n: Int, _ rows: Int, _ count: Int) { b.product(.denoising, k, n, rows, plain, times: count * times) }
        switch r.family {
        case .zImage:
            let c = DiT.Config(header: [:]), m = t + text, hd = c.dim / c.heads
            func block(_ rows: Int, _ layers: Int, _ format: WeightFormat, stack: Bool) {
                b.product(.denoising, c.dim, c.dim, rows, format, times: 4 * layers * times)
                b.product(.denoising, c.dim, c.hidden, rows, format, times: 2 * layers * times)
                b.product(.denoising, c.hidden, c.dim, rows, format, times: layers * times)
                if stack { lora(c.dim, c.dim, rows, 4 * layers); lora(c.dim, c.hidden, rows, 2 * layers); lora(c.hidden, c.dim, rows, layers) }
            }
            block(m, c.layers, f, stack: true)                                   // the layers
            // The noise and context refiners: int8 in the Compact (Comfy-Org's convrot quantizes them with the layers), as published
            // in the Light; no LoRA on them.
            let refiners = r.variant == .compact ? f : plain
            block(t, c.refiners, refiners, stack: false); block(text, c.refiners, refiners, stack: false)
            let latent = c.patch * c.patch * c.channels
            q(latent, c.dim, t, 1); q(c.capDim, c.dim, text, 1); q(c.dim, latent, m, 1)   // x_embedder, cap_embedder, the final linear
            b.attention(m, keys: m, heads: c.heads, kv: c.heads, dim: hd, times: c.layers * times)
            b.attention(t, keys: t, heads: c.heads, kv: c.heads, dim: hd, times: c.refiners * times)
        case .qwenImage21:
            let c = QwenImage21DiT.Config(header: [:])
            let chunk = QwenImage21DiT.mlpRows(scratch: t, hidden: c.hidden)
            p(c.dim, c.dim, t, 4 * c.layers)
            var left = t
            while left > 0 {
                let rows = min(chunk, left); left -= rows
                p(c.dim, c.hidden, rows, 2 * c.layers); p(c.hidden, c.dim, rows, c.layers)
            }
            q(c.channels, c.dim, t, 1); q(c.dim, c.channels, t, 1)               // img_in, proj_out
            b.attention(t, keys: t + text, heads: c.heads, kv: c.heads, dim: c.headDim, times: c.layers * times)
        case .anima:
            let c = AnimaDiT.Config(header: [:])
            // The Compact's blocks left in bf16 by its publisher (`animaCompactPlain`), with the stack like the others.
            let (attention, mlp) = r.variant == .compact ? animaCompactPlain : (0, 0)
            func kept(_ k: Int, _ n: Int, _ count: Int) { q(k, n, t, count); lora(k, n, t, count) }
            p(c.dim, c.dim, t, 6 * (c.layers - attention)); kept(c.dim, c.dim, 6 * attention)   // self q k v o, cross q o
            p(c.dim, c.hidden, t, c.layers - mlp); p(c.hidden, c.dim, t, c.layers - mlp)
            kept(c.dim, c.hidden, mlp); kept(c.hidden, c.dim, mlp)
            q((c.channels + 1) * c.patch * c.patch, c.dim, t, 1); q(c.dim, c.channels * c.patch * c.patch, t, 1)   // patch_embed, proj_out
            b.attention(t, keys: t, heads: c.heads, kv: c.heads, dim: c.headDim, times: c.layers * times)
            b.attention(t, keys: Self.animaTextRows, heads: c.heads, kv: c.heads, dim: c.headDim, cross: true, times: c.layers * times)
        case .krea2:
            let c = Krea2DiT.Config(header: [:]), m = t + text
            p(c.dim, c.dim, m, 3 * c.layers); p(c.dim, c.kvWidth, m, 2 * c.layers)   // q, gate, out ; k, v
            p(c.dim, c.hidden, m, 2 * c.layers); p(c.hidden, c.dim, m, c.layers)
            q(c.channels, c.dim, t, 1); q(c.dim, c.channels, t, 1)               // img_in, final_layer
            b.attention(m, keys: m, heads: c.heads, kv: c.kvHeads, dim: c.headDim, times: c.layers * times)
        case .mossImage:
            let c = MossImageDiT.Config(header: [:]), m = t + text
            p(c.dim, c.dim, m, 4 * c.layers)                                   // q, k, v, o: four products, not one fused QKV
            p(c.dim, 2 * c.hidden, m, c.layers); p(c.hidden, c.dim, m, c.layers)
            // The two text blocks (bf16): fused QKV, the output, up and gate apart, down — and the embedders.
            q(c.dim, 3 * c.dim, text, c.textBlocks); q(c.dim, c.dim, text, c.textBlocks)
            q(c.dim, c.hidden, text, 2 * c.textBlocks); q(c.hidden, c.dim, text, c.textBlocks)
            q(c.textDim, c.dim, text, 1); q(c.channels, c.dim, t, 1); q(c.dim, c.channels, t, 1)
            b.attention(m, keys: m, heads: c.heads, kv: c.heads, dim: c.headDim, times: c.layers * times)
        }
    }

    /// Anima's text, padded to 512 rows: the cross-attention's keys, and its k/v products' height.
    static let animaTextRows = 512

    /// **The blocks Anima's Compact leaves in bf16** (Bedovyy's int8 convrot, its header checked): the attention of 2 blocks (0 and
    /// 27: self q k v o, cross q k v o) and the MLP of 5 (0 to 3, and 27); every adaLN too, products of one row, no class.
    static let animaCompactPlain = (attention: 2, mlp: 5)

    /// **What the DiT computes from the conditions**, once per pass (each grid has its own DiT, built and run apart): Anima's cross k/v
    /// and Qwen-Image-2.1's prefill. Krea 2's text fusion runs once per render.
    private static func conditions(_ b: inout Builder, _ r: Request, _ f: WeightFormat, passes: Int) {
        let plain = plainFormat(r, f)
        switch r.family {
        case .anima:
            let c = AnimaDiT.Config(header: [:])
            let attention = r.variant == .compact ? animaCompactPlain.attention : 0
            b.product(.denoising, c.textDim, c.dim, animaTextRows, f, times: 2 * (c.layers - attention) * passes)
            b.product(.denoising, c.textDim, c.dim, animaTextRows, plain, times: 2 * attention * passes)
        case .qwenImage21:
            // The prefill: the conditions through every layer, the last one only for its keys and values; txt_in's two products first.
            let c = QwenImage21DiT.Config(header: [:]), rows = textRows(.qwenImage21, promptTokens: r.promptTokens)
            let chunk = QwenImage21DiT.mlpRows(scratch: rows, hidden: c.hidden)
            let full = c.layers - 1
            b.product(.denoising, c.context, c.dim, rows, plain, times: passes)
            b.product(.denoising, c.dim, c.dim, rows, plain, times: passes)
            b.product(.denoising, c.dim, c.dim, rows, f, times: (4 * full + 2) * passes)
            var left = rows
            while left > 0 {
                let n = min(chunk, left); left -= n
                b.product(.denoising, c.dim, c.hidden, n, f, times: 2 * full * passes)
                b.product(.denoising, c.hidden, c.dim, n, f, times: full * passes)
            }
        case .krea2:
            // The fusion (bf16): two blocks over 12 hidden states per token, two over the tokens, then txt_in. Once per render.
            let d = Krea2TextFusion.Shape.dim, h = Krea2TextFusion.Shape.hidden, t = textRows(.krea2, promptTokens: r.promptTokens)
            let c = Krea2DiT.Config(header: [:])
            for rows in [12 * t, t] {
                b.product(.denoising, d, d, rows, plain, times: 2 * 5)
                b.product(.denoising, d, h, rows, plain, times: 2 * 2)
                b.product(.denoising, h, d, rows, plain, times: 2)
            }
            b.product(.denoising, d, c.dim, t, plain, times: 1)
            b.product(.denoising, c.dim, c.dim, t, plain, times: 1)
        default: break
        }
    }

    // MARK: - The catalogue

    /// **The classes of one render**, `nil` when the version does not exist for the family (MOSS-Image has no Light).
    /// Sorted: encoding first, then by decreasing FLOP of the render, then by name — deterministic.
    package static func classes(for original: Request) -> [Class]? {
        let r = original.byTensors
        guard let format = weightFormat(r.family, r.variant), let encoderFormat = encoderFormat(r.family, r.variant) else { return nil }
        var b = Builder()
        if r.includesEncoder { encoding(&b, r, encoderFormat) }
        let passes = passes(r)
        for pass in passes { evaluation(&b, r, format, tokens: pass.tokens, times: pass.evaluations) }
        conditions(&b, r, format, passes: passes.count)
        return sorted(b.order.map { Class(phase: $0.phase, shape: $0.shape, executions: [r.family: b.counts[$0]!]) })
    }

    /// **The classes of one pass of the render alone** (`passes(r)[index]`: the half grid's, the full grid's): the denoising's evaluations at
    /// that pass's rows, without the encoder and without what is computed once — what the weight stream's read runs beside, evaluation by
    /// evaluation (`CapacityStream`). `nil` for a version that does not exist or a pass that does not.
    package static func classes(for original: Request, pass index: Int) -> [Class]? {
        let r = original.byTensors
        guard let format = weightFormat(r.family, r.variant) else { return nil }
        let all = passes(r)
        guard all.indices.contains(index) else { return nil }
        var b = Builder()
        evaluation(&b, r, format, tokens: all[index].tokens, times: all[index].evaluations)
        return sorted(b.order.map { Class(phase: $0.phase, shape: $0.shape, executions: [r.family: b.counts[$0]!]) })
    }

    /// **The classes of one denoising evaluation at `tokens` image rows** (a pass's per-evaluation work), for a weight in `format` — what the
    /// engine knows at a forward (the map's format, the grid, the prompt), without a `Request`'s width or step count. `nil` for a format no
    /// version of the family has.
    package static func evaluationClasses(family: Family, format: WeightFormat, tokens: Int, promptTokens: Int) -> [Class]? {
        let variant: Variant = format == .bf16 ? .standard : (format == .q4k ? .light : .compact)
        guard weightFormat(family, variant) == format else { return nil }
        let r = Request(family: family, variant: variant, promptTokens: promptTokens)
        var b = Builder()
        evaluation(&b, r, format, tokens: tokens, times: 1)
        return sorted(b.order.map { Class(phase: $0.phase, shape: $0.shape, executions: [family: b.counts[$0]!]) })
    }

    /// **Identical classes of different families, merged** (same phase, same shape: the same k×n, weight format and
    /// height; the same attention), their executions kept per family.
    package static func merge(_ lists: [[Class]]) -> [Class] {
        var byKey: [Key: Class] = [:]
        var order: [Key] = []
        for c in lists.joined() {
            let key = Key(phase: c.phase, shape: c.shape)
            if var kept = byKey[key] {
                for (family, n) in c.executions { kept.executions[family, default: 0] += n }
                byKey[key] = kept
            } else { byKey[key] = c; order.append(key) }
        }
        return sorted(order.map { byKey[$0]! })
    }

    private static func sorted(_ classes: [Class]) -> [Class] {
        func total(_ c: Class) -> Double { c.executions.values.reduce(0) { $0 + Double($1) } * c.flop }
        return classes.sorted { a, b in
            if a.phase != b.phase { return a.phase == .encoding }
            if total(a) != total(b) { return total(a) > total(b) }
            return a.id + String(describing: a.shape) < b.id + String(describing: b.shape)
        }
    }
}
