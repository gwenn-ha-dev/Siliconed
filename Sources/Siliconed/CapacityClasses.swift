import Foundation

/// **The classes of computation of a render, their executions, their weights, and the time they are predicted to
/// take** — the catalogue the capacity map decides over and the render's predicted-versus-measured check
/// compares with. Pure: it reads the five families' `Config` defaults (no header, no weights), a version (the
/// weights' format), a format (the pixels) and "Faster Rendering".
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
///     weight[family] (c)              executions · FLOP / rate(c, reference route)   ÷   Σ over the family's classes
///     time[family] (c, route)         executions · FLOP / rate(c, route) · heat(route)        (seconds)
///     prediction(family, table)       Σ over the phases asked (the denoising by default) of time(c, table's route for c)
///
/// What sets it apart — what a catalogue written by analogy with "one product per layer" would get wrong:
///
///   - **The heights are the render's, not the map's two**: the image rows plus each family's own text rows —
///     Z-Image 19 template + prompt tokens rounded up to 32 (1 056 · 4 128 · 6 176 at 512² · 1024² · 1024×1536),
///     Krea 2 +16 (1 040 · 4 112 · 6 160), MOSS-Image +21 (1 045 · 4 117 · 6 165), Qwen-Image-2.1 and Anima the
///     image alone (the text is a prefix of the keys, or a cross-attention). Faster Rendering adds the half grid's
///     (about 1 050 · 1 560 rows), at the family's own count of evaluations (Z-Image, Qwen, Anima 3; Krea 2 4; MOSS 1).
///   - **Qwen-Image-2.1's MLP runs by row chunks** (`QwenImage21DiT.mlpRows`): a 1024×1536 target is two products of
///     3 072 rows, not one of 6 144 — the class is the chunk's.
///   - **A class has the key of the table**: `routeKey` is the `RouteTable.ClassKey` a plan names (`product:kxn@m`,
///     `attention:S/headsxdim`); a self- and a cross-attention of the same S, heads and dim share one — the table
///     cannot tell them apart, and the catalogue keeps both so that their weights add.
///   - **A weight format is part of a product's shape**: the same k×n in bf16 (Standard) and in int8 (Compact) are two
///     classes, so a merge across families of different versions never adds what no route could treat alike. K-quants
///     (Light) have a single route, the widening: they carry no `RouteTable.WeightKind`.
///   - **Merged across families, weights stay per family**: a class common to two families (the same k×n at the same
///     height) is one class with two executions counts; its weight is asked for a family, and sums to 1 over that family.
///   - **What is not a class**: the modulation and time-embedding gemv (m = 1, the CPU), the norms and elementwise
///     kernels, the VAE (one route), the weights' widening and reading, attentions over fewer than 64 rows (no route
///     decides them), the encoder's attention and DeltaNet recurrence (m < 64), MOSS's LoRA (rank 128, fp32, applied
///     after the product), the embedders other than Z-Image's, Anima's adapter and Krea 2's text fusion, W_b/W_a of the
///     MOSS encoder. They are small (each under 1 % of a render), or have no route to choose among.
///   - **Unread from code**: Krea 2's text encoder runs up to its last tap (36 layers is the Qwen3-VL-4B text model's
///     count, an upper bound); MOSS's encoder FFN width (12 288) is the Qwen3.5-9B card's, not read in the code.
///   - **The prediction is cold rates times a heat factor**, and nothing else: no overlap between phases, no cost for
///     widening or reading (the table's `weights` rules say how a weight is read; the rates, once measured, carry what
///     it costs). Neutral rates give the FLOP ranking; the map's measured rates replace them class by class.
package enum CapacityClasses {
    package enum Phase: String, Sendable, Hashable, CaseIterable {
        /// The text encoder, once per render.
        case encoding
        /// The DiT's evaluations, and what it computes once from the conditions.
        case denoising
    }

    /// The format a weight is published in, as the version carries it.
    package enum WeightFormat: String, Sendable, Hashable, CaseIterable {
        case bf16, int8, q8
        /// Q4_K to Q6_K: a single route (widening on the CPU).
        case q4k

        /// The route table's kind, `nil` for the K-quants.
        package var kind: RouteTable.WeightKind? {
            switch self {
            case .bf16: return .bf16
            case .int8: return .int8
            case .q8: return .q8
            case .q4k: return nil
            }
        }
    }

    // MARK: - Versions

    /// **The DiT's weight format of a version**: Standard bf16; Compact Q8_0 (Z-Image, Anima) or int8 convrot (Qwen-Image-2.1,
    /// Krea 2, MOSS-Image); Light Q4_K (no Light for MOSS-Image: `nil`).
    package static func weightFormat(_ family: Family, _ variant: Variant) -> WeightFormat? {
        switch (variant, family) {
        case (.standard, _): return .bf16
        case (.compact, .zImage), (.compact, .anima): return .q8
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
        /// « Faster Rendering » (`Spectral.halfSizeStart`): the family's first evaluations on the half grid.
        package var fasterRendering: Bool
        /// The prompt's tokens (the benchmarks' prompt: 11).
        package var promptTokens: Int
        package var steps: Int
        /// Include the text encoder (a new prompt); an « image suivante » does not run it.
        package var includesEncoder: Bool
        /// The half-grid evaluations a render actually did, when it is not the family's rule (`SILICONED_SPECTRAL`);
        /// `nil`: the family's own count.
        package var halfSteps: Int?

        package init(family: Family, variant: Variant = .standard, width: Int = 1024, height: Int = 1024,
                     fasterRendering: Bool = true, promptTokens: Int = 11, steps: Int = 8, includesEncoder: Bool = true,
                     halfSteps: Int? = nil) {
            self.family = family; self.variant = variant; self.width = width; self.height = height
            self.fasterRendering = fasterRendering; self.promptTokens = promptTokens; self.steps = steps
            self.includesEncoder = includesEncoder; self.halfSteps = halfSteps
        }
    }

    // MARK: - Classes

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

        /// The name a plan gives the class and the table reads (`RouteTable.ClassKey`).
        package var routeKey: RouteTable.ClassKey {
            switch shape {
            case let .product(k, n, rows, _): return .product(k: k, n: n, m: rows)
            case let .attention(s, _, heads, _, dim, _): return .attention(sequence: s, heads: heads, headDim: dim)
            }
        }

        /// The weight's kind (products only; `nil`: a K-quant, or an attention).
        package var weightKind: RouteTable.WeightKind? {
            if case let .product(_, _, _, format) = shape { return format.kind }
            return nil
        }

        /// FLOP of all the executions of a render of `family`.
        package func totalFLOP(_ family: Family) -> Double { Double(executions[family] ?? 0) * flop }
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
        let halvable = Spectral.divisible(height: r.height / factor, width: r.width / factor, by: 2,
                                          floor: Spectral.floor(factor: factor))
        let k = r.fasterRendering && halvable ? min(r.halfSteps ?? halfSizeSteps(r.family), max(r.steps - 1, 0)) : 0
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

    /// One evaluation's classes at `tokens` image rows, `times` evaluations.
    private static func evaluation(_ b: inout Builder, _ r: Request, _ f: WeightFormat, tokens t: Int, times: Int) {
        let text = textRows(r.family, promptTokens: r.promptTokens)
        func p(_ k: Int, _ n: Int, _ rows: Int, _ count: Int) { b.product(.denoising, k, n, rows, f, times: count * times) }
        switch r.family {
        case .zImage:
            let c = DiT.Config(header: [:]), m = t + text, hd = c.dim / c.heads
            func block(_ rows: Int, _ layers: Int) {
                p(c.dim, c.dim, rows, 4 * layers); p(c.dim, c.hidden, rows, 2 * layers); p(c.hidden, c.dim, rows, layers)
            }
            block(m, c.layers); block(t, c.refiners); block(text, c.refiners)   // layers, noise refiner, context refiner
            let latent = c.patch * c.patch * c.channels
            p(latent, c.dim, t, 1); p(c.capDim, c.dim, text, 1); p(c.dim, latent, t, 1)
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
            b.attention(t, keys: t + text, heads: c.heads, kv: c.heads, dim: c.headDim, times: c.layers * times)
        case .anima:
            let c = AnimaDiT.Config(header: [:])
            p(c.dim, c.dim, t, 6 * c.layers)                                   // self q k v o, cross q o
            p(c.dim, c.hidden, t, c.layers); p(c.hidden, c.dim, t, c.layers)
            b.attention(t, keys: t, heads: c.heads, kv: c.heads, dim: c.headDim, times: c.layers * times)
            b.attention(t, keys: Self.animaTextRows, heads: c.heads, kv: c.heads, dim: c.headDim, cross: true, times: c.layers * times)
        case .krea2:
            let c = Krea2DiT.Config(header: [:]), m = t + text
            p(c.dim, c.dim, m, 3 * c.layers); p(c.dim, c.kvWidth, m, 2 * c.layers)
            p(c.dim, c.hidden, m, 2 * c.layers); p(c.hidden, c.dim, m, c.layers)
            b.attention(m, keys: m, heads: c.heads, kv: c.kvHeads, dim: c.headDim, times: c.layers * times)
        case .mossImage:
            let c = MossImageDiT.Config(header: [:]), m = t + text
            func block(_ rows: Int, _ layers: Int) {
                p(c.dim, 3 * c.dim, rows, layers); p(c.dim, c.dim, rows, layers)
                p(c.dim, 2 * c.hidden, rows, layers); p(c.hidden, c.dim, rows, layers)
            }
            block(m, c.layers); block(text, c.textBlocks)
            p(c.textDim, c.dim, text, 1)
            b.attention(m, keys: m, heads: c.heads, kv: c.heads, dim: c.headDim, times: c.layers * times)
        }
    }

    /// Anima's text, padded to 512 rows: the cross-attention's keys, and its k/v products' height.
    static let animaTextRows = 512

    /// What the DiT computes once from the conditions, per render.
    private static func conditions(_ b: inout Builder, _ r: Request, _ f: WeightFormat) {
        switch r.family {
        case .anima:
            let c = AnimaDiT.Config(header: [:])
            b.product(.denoising, c.textDim, c.dim, animaTextRows, f, times: 2 * c.layers)
        case .qwenImage21:
            // The prefill: the conditions through every layer, the last one only for its keys and values.
            let c = QwenImage21DiT.Config(header: [:]), rows = textRows(.qwenImage21, promptTokens: r.promptTokens)
            let chunk = QwenImage21DiT.mlpRows(scratch: rows, hidden: c.hidden)
            let full = c.layers - 1
            b.product(.denoising, c.dim, c.dim, rows, f, times: 4 * full + 2)
            var left = rows
            while left > 0 {
                let n = min(chunk, left); left -= n
                b.product(.denoising, c.dim, c.hidden, n, f, times: 2 * full)
                b.product(.denoising, c.hidden, c.dim, n, f, times: full)
            }
        default: break
        }
    }

    // MARK: - The catalogue

    /// **The classes of one render**, `nil` when the version does not exist for the family (MOSS-Image has no Light).
    /// Sorted: encoding first, then by decreasing FLOP of the render, then by name — deterministic.
    package static func classes(for r: Request) -> [Class]? {
        guard let format = weightFormat(r.family, r.variant), let encoderFormat = encoderFormat(r.family, r.variant) else { return nil }
        var b = Builder()
        if r.includesEncoder { encoding(&b, r, encoderFormat) }
        for pass in passes(r) { evaluation(&b, r, format, tokens: pass.tokens, times: pass.evaluations) }
        conditions(&b, r, format)
        return sorted(b.order.map { Class(phase: $0.phase, shape: $0.shape, executions: [r.family: b.counts[$0]!]) })
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
            return a.routeKey.id + String(describing: a.shape) < b.routeKey.id + String(describing: b.shape)
        }
    }

    // MARK: - Rates and weights

    /// The route names of a class: engines for a product, kernels for an attention.
    package static let productRoutes = ["gpu", "gpu+amx", "gpu+ane", "gpu+ane+amx"]
    package static let attentionRoutes = ["graph", "flash", "split"]

    /// The engines a route keeps busy (the keys of `CapacityMeasure.Heat.factors`): `gpu+ane+amx` → gpu, ane, amx;
    /// `split` → gpu, cpu (the heads' share on the CPU); `graph` and `flash`, the GPU's kernels → gpu (as `CapacityMap.attentionRoutes`
    /// declares them); the weight's form after a `/` (`gpu+amx/int8`) is no engine.
    package static func engines(ofRoute id: String) -> [String] {
        switch id {
        case "split": return ["gpu", "cpu"]
        case "graph", "flash": return ["gpu"]
        default: return id.split(separator: "/")[0].split(separator: "+").map(String.init)
        }
    }

    /// **Cold rates, FLOP per second, by route**: a default per route and kind of class, and the figures measured
    /// per class (keyed by `routeKey.id`, then by route). Neutral: one rate for every route — the weights are then
    /// the FLOP shares and the prediction cannot tell routes apart.
    package struct Rates: Sendable, Equatable {
        package var product: [String: Double]
        package var attention: [String: Double]
        package var measured: [String: [String: Double]]
        package var fallback: Double

        package init(product: [String: Double] = [:], attention: [String: Double] = [:],
                     measured: [String: [String: Double]] = [:], fallback: Double = 1e12) {
            self.product = product; self.attention = attention; self.measured = measured; self.fallback = fallback
        }

        package static let neutral = Rates()

        package func rate(_ c: Class, route: String) -> Double {
            if let r = measured[c.routeKey.id]?[route], r > 0 { return r }
            switch c.shape {
            case .product: return product[route] ?? fallback
            case .attention: return attention[route] ?? fallback
            }
        }

        /// The route a weight is computed on: the simplest (GPU alone, the graph).
        func reference(_ c: Class) -> String {
            switch c.shape {
            case .product: return CapacityClasses.productRoutes[0]
            case .attention: return CapacityClasses.attentionRoutes[0]
            }
        }
    }

    /// **The weight of each class for `family`**: its share of the family's render, `executions · FLOP / rate` on the
    /// reference route, over the sum on the classes of the family (all phases, or `phases`). Zero for a class the
    /// family does not run; the weights of a family sum to 1.
    package static func weights(_ classes: [Class], family: Family, rates: Rates = .neutral,
                                phases: Set<Phase> = Set(Phase.allCases)) -> [Double] {
        let raw = classes.map { c -> Double in
            guard phases.contains(c.phase) else { return 0 }
            return c.totalFLOP(family) / rates.rate(c, route: rates.reference(c))
        }
        let total = raw.reduce(0, +)
        return total > 0 ? raw.map { $0 / total } : raw
    }

    /// The same, per family: `weights[class][family]`, only the families that run the class.
    package static func weights(_ classes: [Class], rates: Rates = .neutral) -> [[Family: Double]] {
        var result = classes.map { _ in [Family: Double]() }
        for family in Family.allCases {
            let w = weights(classes, family: family, rates: rates)
            for i in classes.indices where classes[i].executions[family] != nil { result[i][family] = w[i] }
        }
        return result
    }

    // MARK: - The prediction

    package struct Prediction: Sendable, Equatable {
        package struct Part: Sendable, Equatable {
            package var classID: String
            package var phase: Phase
            package var route: String
            package var seconds: Double
        }
        package var seconds: Double
        package var parts: [Part]
    }

    /// **The route the table gives a class** for `family`: a product's engines (`RouteTable.Products.route`, the AMX only
    /// from `amxMinimumRows` — the engine's `amx_min` —, never the ANE on an encoder, nor in a process that did not bring
    /// it up: `aneBroughtUp`), an attention's kernel (`flash` where the family's code tries it before the split — Z-Image
    /// and Anima's self-attention, not Anima's cross-attention —, `split` where the table splits, else `graph`).
    ///
    /// `weightForms`: the route as the map names it for a weight read as published — the weight rule of the table for the
    /// class's kind says `gpu/bf16`, `gpu/bf16act`, `gpu/int8`, `gpu+amx/int8`, `gpu/q8`, `gpu+amx/q8` (no ANE and, for a
    /// bf16 weight, no AMX: `TensorGEMM` takes the product whole) —, which is what the rates of the map are keyed by and
    /// what the ledger records. Off by default: the engines alone.
    package static func route(of c: Class, family: Family, table: RouteTable,
                              amxMinimumRows: Int = EngineSettings.defaultAMXMinimumRows,
                              aneBroughtUp: Bool = true, weightForms: Bool = false) -> String {
        switch c.shape {
        case let .product(k, n, rows, _):
            var route = table.products.route(k: k, n: n, m: rows, aneBroughtUp: aneBroughtUp)
            if c.phase == .encoding { route.aneRows = 0; route.amxBesideANE = false }
            if rows < amxMinimumRows, route.aneRows == 0 { route.amx = false }
            let engines = route.engines
            // A K-quant has no native form (`weightKind` nil): it widens.
            guard weightForms, let kind = c.weightKind else { return engines.rawValue }
            let weight = table.products.weight(kind, k: k, n: n, m: rows)
            switch weight.path {
            case .widened: return engines.rawValue
            case .bf16: return weight.roundsActivation ? "gpu/bf16act" : "gpu/bf16"
            case .native:
                let withAMX = (engines == .amx || engines == .aneAMX) && n >= 128 && rows >= amxMinimumRows
                return (withAMX ? "gpu+amx/" : "gpu/") + kind.rawValue
            }
        case let .attention(s, _, heads, _, dim, cross):
            let a = table.attention.route(sequence: s, heads: heads, headDim: dim)
            if a.flash, !cross, family == .zImage || family == .anima { return "flash" }
            return a.splits ? "split" : "graph"
        }
    }

    /// **The predicted seconds of a render of `family`** over `phases` (the denoising by default): for each class,
    /// `executions · FLOP / rate(route) · heat(route)` with the route the table gives it, summed. Additive over classes,
    /// decreasing in every rate, increasing in every heat factor.
    package static func predict(_ classes: [Class], family: Family, table: RouteTable, rates: Rates = .neutral,
                                heat: CapacityMeasure.Heat = CapacityMeasure.Heat(), phases: Set<Phase> = [.denoising],
                                amxMinimumRows: Int = EngineSettings.defaultAMXMinimumRows,
                                aneBroughtUp: Bool = true, weightForms: Bool = false) -> Prediction {
        var parts: [Prediction.Part] = []
        for c in classes where phases.contains(c.phase) && (c.executions[family] ?? 0) > 0 {
            let id = route(of: c, family: family, table: table, amxMinimumRows: amxMinimumRows,
                           aneBroughtUp: aneBroughtUp, weightForms: weightForms)
            let factor = heat.factor(CapacityMeasure.Route(id, engines: engines(ofRoute: id)))
            parts.append(.init(classID: c.routeKey.id, phase: c.phase, route: id,
                               seconds: c.totalFLOP(family) / rates.rate(c, route: id) * factor))
        }
        return Prediction(seconds: parts.reduce(0) { $0 + $1.seconds }, parts: parts)
    }
}
