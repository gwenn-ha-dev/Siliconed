import Foundation

/// **What the lab's writer needs, computed from shapes before its first token** — `MemoryNeed`'s rule: every
/// term is a function its allocator calls (`Qwen3VLGenerator.arenaBytes`, `cacheBytes`, `Qwen3VLVision.arenaBytes`…),
/// never measured, never fitted.
///
///     anonymous (the writer's own, all through its life)
///         GPU decoder scratch
///         + the tail's staging buffers  (`PromptWriterPlan.stagingSlots` × the largest layer's run; none when nothing streams)
///     + the larger of
///         vision    (Describe only)   tower arena, pixel_values, features, attention, GEMM tiers
///         language                    arena (≤ 512 rows), K/V cache for prompt + new tokens, attention scores,
///                                     features until the prompt is read, GEMM tiers
///     + the process's base (the app's, read from the process: `Request.base`)
///
///     the cache's (clean pages of the maps, nobody's footprint — what B leaves)
///         the head: lm_head 1.24 GB + the final norm       always, a token cannot do without it
///         + the layers before `streamedFrom`               the rest of the layers are read around the cache
///
/// **A plan, chosen in B before the first token** (`PromptWriterNeed.chosen`, `MemoryPlan`'s rule): from every layer in the
/// cache (`streamedFrom` = 36: the Compact on a 16 GB Mac) down to the head alone (`streamedFrom` = 0: the layers all
/// streamed, two staging buffers), the first whose anonymous peak and cache together fit in B. The cache is a share of B,
/// sized as the DiT's head is (`MemoryPlan.released`): the writer does not leave the cache more than B has left, so a
/// plan that fits is a plan whose pages can all stay.
///
/// **Why the weights are no longer the writer's own pages.** Decoding reads every weight at every token. Held in a copy of
/// the process's own (`Artifact.keepResident`) they were 8.2 GB of anonymous memory at the Compact's width — 85 % of B on a
/// 16 GB Mac, and what the kernel compresses and swaps when the apps around want room. The map's pages are clean and
/// file-backed: under pressure the kernel drops them and reads them back, it never swaps them. What does not fit B's cache
/// is read from the disk at each token, and the card says what that costs (`tokenSeconds`: bytes in the cache over the
/// memory's bandwidth, plus bytes streamed over the disk's) — the sentence « a streamed decode is too slow to be a
/// product » is a number now, not a premise.
///
/// The writer is offered when a plan fits B (the head and two staging buffers beside the larger phase); « needs a Mac with N
/// GB » only says which Mac holds the smallest plan. The writer runs alone (the lab is cancelled and the weights released
/// when a render starts), so B is the writer's.
///
/// What stays outside the formula, as for the renders: the mapped `embed_tokens` (one row a token, file-backed,
/// clean) and the vision weights (`visual.*`, read once from the map), which are the page cache's.
package struct PromptWriterMaps: Sendable {
    package let encoder: TextEncoder.Config
    package let vision: Qwen3VLVision.Config
    /// The bytes of the layers and of the head in their maps (each tensor page-rounded, from the headers): what the cache holds
    /// of them when all are cached.
    package let residentLayers: Int, residentHead: Int
    /// Each layer's run in the encoder map (`Artifact.layerSpans`): what a streamed layer reads at each token, and what
    /// its staging buffer holds (the largest).
    package let layerSpans: [Int]
    /// What widening the largest rotated (int8 convrot) weight allocates beside the arenas; 0 for bf16.
    package let rotationScratch: Int
    package let format: CapacityClasses.WeightFormat
    /// The logits' rows (`lm_head.weight`'s): the identifiers the detokenization table covers.
    package let vocabulary: Int
    /// The tokenizer's tables (`Tokenizer.tableBytes`) and the bytes-of-token table (`Tokenizer.detokenizationBytes`).
    package let tokenizerTables: Int, detokenization: Int

    /// From the headers and the tokenizer's counts — pure: the tests hand it headers written in memory.
    /// `nil` when a config does not parse, the head is not a head, or a map has no tensor table.
    package init?(encoder: [String: Any], head: [String: Any], tokenizerTables: Int, detokenization: Int) {
        guard let config = try? TextEncoder.Config(header: encoder), let vision = try? Qwen3VLVision.Config(header: encoder),
              head["kind"] as? String == "qwen3-vl-lm-head",
              let rows = ((head["tensors"] as? [String: [String: Any]])?["lm_head.weight"]?["shape"] as? [Int])?.first else { return nil }
        let prefixes = Qwen3VLGenerator.residentPrefixes(config: config)
        residentLayers = Artifact.residentBytes(header: encoder, prefixes: prefixes.encoder)
        residentHead = Artifact.residentBytes(header: head, names: prefixes.head)
        layerSpans = Artifact.layerSpans(header: encoder, prefixes: prefixes.encoder, fileSize: .max)
        guard residentLayers > 0, residentHead > 0, layerSpans.count == config.layers, layerSpans.allSatisfy({ $0 > 0 }) else { return nil }
        self.encoder = config; self.vision = vision; vocabulary = rows
        rotationScratch = Widen.rotationScratchBytes(header: encoder, prefixes: prefixes.encoder)
        format = Artifact.weightFormat(header: encoder)
        self.tokenizerTables = tokenizerTables; self.detokenization = detokenization
    }

    /// The maps the library has installed for Qwen-Image-2.1 (its encoder, its writer's head, its tokenizer);
    /// `nil` when one is missing or unreadable.
    package init?(library: Library = .processWide) {
        guard let headPath = library.writerHeadMap(.qwenImage21), let processor = try? library.component(.qwenImage21, "processor"),
              let encoder = try? Artifact.header(library.map(library.encoderMap(.qwenImage21))),
              let head = try? Artifact.header(headPath),
              let rows = ((head["tensors"] as? [String: [String: Any]])?["lm_head.weight"]?["shape"] as? [Int])?.first,
              let tables = try? Tokenizer.tableBytes(file: processor + "/tokenizer.json"),
              let detokenization = try? Tokenizer.detokenizationBytes(file: processor + "/tokenizer.json", rows: rows),
              let maps = PromptWriterMaps(encoder: encoder, head: head, tokenizerTables: tables, detokenization: detokenization)
        else { return nil }
        self = maps
    }
}

/// **Where the writer's layers are read from at each token**: the first `streamedFrom` from the cache, the others around it
/// (`pread` + `F_NOCACHE` into `stagingSlots` buffers, the way the DiT's tail is). The head is always the cache's.
package struct PromptWriterPlan: Sendable, Equatable {
    /// The first streamed layer; the layer count: none.
    package let streamedFrom: Int
    package let layers: Int
    package init(streamedFrom: Int, layers: Int) { self.streamedFrom = min(max(0, streamedFrom), layers); self.layers = layers }
    package static func cached(layers: Int) -> PromptWriterPlan { .init(streamedFrom: layers, layers: layers) }
    package var streams: Bool { streamedFrom < layers }
    package var cachedLayers: Int { streamedFrom }
    package var streamedLayers: Int { layers - streamedFrom }
    package static var stagingSlots: Int { Qwen3VLGenerator.stagingSlots }
}

package struct PromptWriterNeed: Sendable {
    /// What a request may ask: the app's limits, which the availability of the feature is judged at.
    package struct Limits: Sendable, Equatable {
        /// The prompt's tokens, template, instruction and image tokens included.
        package var promptTokens = 2048
        /// The tokens the answer may take.
        package var newTokens = 768
        /// The image's patches (16 × 16 pixels): `Qwen3VLChat.maxPixels` is 4 096 of them.
        package var imagePatches = Qwen3VLChat.maxPixels / 256
        package init() {}
        package init(promptTokens: Int, newTokens: Int, imagePatches: Int = Qwen3VLChat.maxPixels / 256) {
            self.promptTokens = promptTokens; self.newTokens = newTokens; self.imagePatches = imagePatches
        }
    }

    package enum Availability: Equatable, Sendable {
        /// A plan fits B: its anonymous peak (the cache's share is in `PromptWriterNeed.cacheBytes`).
        case available(need: Int)
        /// No plan fits B (not even the head and two staging buffers): the smallest Mac size (GB) whose B covers the smallest plan.
        case needsMac(gigabytes: Int, need: Int)
    }

    package let maps: PromptWriterMaps
    /// The prompt's tokens and the answer's bound; `patches`: the image's, `nil` for a text rewrite.
    package let promptTokens: Int, newTokens: Int, patches: Int?
    package private(set) var resident: [MemoryNeed.Term]
    package let phases: [MemoryNeed.Phase]
    package var base = 0
    /// Where the layers are read from (the head is always the cache's).
    package private(set) var placement: PromptWriterPlan

    /// - Parameters:
    ///   - plan: the encoder's plan (`CapacityPlan.restricted(allowingANE: false)` of the process's): which engine takes
    ///     rows of which product; `.defaults` (the GPU alone) in the tests.
    ///   - threads: the performance cores the vision attention's head split runs on.
    package init(maps: PromptWriterMaps, promptTokens: Int, newTokens: Int, patches: Int? = nil, streamedFrom: Int? = nil,
                 plan: CapacityPlan = EngineSettings.effective.plan.restricted(allowingANE: false),
                 threads: Int = EngineSettings.effective.machine.performanceCores) {
        self.maps = maps; self.promptTokens = promptTokens; self.newTokens = newTokens; self.patches = patches
        let e = maps.encoder, v = maps.vision
        let (h, a, kv, i) = (e.hidden, e.attentionWidth, e.kvWidth, e.intermediate)
        let pixels = patches.map { Arena.capacity(for: $0 * v.patchValues * 4) } ?? 0
        let features = patches.map { Qwen3VLVision.featureBytes(tokens: $0 / (v.merge * v.merge), output: v.output,
                                                                levels: v.deepstack.count) } ?? 0
        resident = []
        placement = .cached(layers: e.layers)
        var list: [MemoryNeed.Phase] = []
        if let largest = patches {
            let (d, vi, m, o) = (v.hidden, v.intermediate, v.merged, v.output)
            list.append(.init(kind: .vision, terms: [
                MemoryNeed.Term("vision arena", Qwen3VLVision.arenaBytes(config: v, patches: largest)),
                MemoryNeed.Term("pixel_values", pixels), MemoryNeed.Term("vision features", features),
                MemoryNeed.Term("tokenizer tables", maps.tokenizerTables), MemoryNeed.Term("detokenization table", maps.detokenization),
                MemoryNeed.Term("rotation scratch", maps.rotationScratch),
            ] + MemoryNeed.attention("vision", heads: v.heads, rows: largest, keys: largest, headDim: v.headDim,
                                     causal: false, threads: threads)
              + MemoryNeed.gemm("vision", products: [(largest, v.patchValues, d), (largest, d, 3 * d), (largest, d, d),
                                                     (largest, d, vi), (largest, vi, d), (largest / 4, m, m), (largest / 4, m, o)],
                                plan: plan, weight: maps.format, phase: .encoding)))
        }
        let rows = Qwen3VLGenerator.rows(prompt: promptTokens)
        list.append(.init(kind: .text, terms: [
            MemoryNeed.Term("writer arena", Qwen3VLGenerator.arenaBytes(config: e, prompt: promptTokens)),
            MemoryNeed.Term("K/V cache", Qwen3VLGenerator.cacheBytes(config: e, length: promptTokens + newTokens)),
            MemoryNeed.Term("attention scores", Qwen3VLGenerator.attentionScratchBytes(longest: promptTokens + newTokens)),
            MemoryNeed.Term("pixel_values", pixels), MemoryNeed.Term("vision features", features),
            MemoryNeed.Term("tokenizer tables", maps.tokenizerTables), MemoryNeed.Term("detokenization table", maps.detokenization),
            MemoryNeed.Term("rotation scratch", maps.rotationScratch),
        ] + MemoryNeed.gemm("writer", bounded: [(rows, h, a), (rows, h, kv), (rows, a, h), (rows, h, i), (rows, i, h)],
                            spread: rows - 1, plan: plan, weight: maps.format, phase: .encoding)))
        phases = list
        place(streamedFrom: streamedFrom ?? e.layers)
    }

    /// The writer's own memory through its life, for a placement: the decoder is built at `load`, whatever the prompt, and lives as
    /// long as the weights; the staging buffers exist only when a tail streams. Each is the allocator's own function.
    private mutating func place(streamedFrom: Int) {
        let e = maps.encoder
        placement = PromptWriterPlan(streamedFrom: streamedFrom, layers: e.layers)
        resident = [MemoryNeed.Term("GPU decoder scratch", Qwen3VLGenerator.decoderBytes(config: e, vocabulary: maps.vocabulary))]
        if placement.streams {
            resident.append(MemoryNeed.Term("tail staging", PromptWriterPlan.stagingSlots * Arena.capacity(for: maps.layerSpans.max() ?? 0)))
        }
    }

    /// The same request on another placement of the layers.
    package func placed(streamedFrom: Int) -> PromptWriterNeed {
        var other = self
        other.place(streamedFrom: streamedFrom)
        return other
    }

    /// **What the cache holds of the maps** in this placement: the layers before the stream, and the head.
    package var cacheBytes: Int { maps.layerSpans.prefix(placement.streamedFrom).reduce(0, +) + maps.residentHead }

    /// **What a token reads around the cache**: the streamed layers' runs.
    package var streamedBytes: Int { maps.layerSpans.suffix(placement.streamedLayers).reduce(0, +) }

    /// The anonymous peak and the cache together: what B must hold for every page to stay where it is read from.
    package var total: Int { peak + cacheBytes }

    /// **A token's seconds, predicted from the card**: the bytes in the cache over the memory's bandwidth, plus the streamed
    /// ones over the disk's (`diskBandwidth`, the stream's own rate). A sum, not the larger of the two: the disk's reads and
    /// the memory's overlap in part, so it is an upper bound where they do. `nil` when the card lacks a figure it needs
    /// (the memory's always; the disk's only for a plan that streams).
    package func tokenSeconds(model: CapacityModel? = EngineSettings.effective.plan.model) -> Double? {
        guard let memory = model?.memoryBandwidth, memory > 0 else { return nil }
        var seconds = Double(cacheBytes) / memory
        if streamedBytes > 0 {
            guard let disk = model?.diskBandwidth, disk > 0 else { return nil }
            seconds += Double(streamedBytes) / disk
        }
        return seconds
    }

    /// The resident weights.
    package var residentBytes: Int { resident.reduce(0) { $0 + $1.bytes } }

    /// **The peak**: the weights, the larger phase, the process's base.
    package var peak: Int { residentBytes + (phases.map(\.bytes).max() ?? 0) + base }

    /// Whether B covers this placement — or the Mac that does. `ceiling`: B (`MemoryBudget.machineCeiling()`).
    package func availability(ceiling: Int) -> Availability {
        total <= ceiling ? .available(need: peak) : .needsMac(gigabytes: MemoryBudget.macNeeded(for: total), need: total)
    }

    /// **The plan, chosen in B**: the placement with the most layers in the cache whose anonymous peak and cache fit `ceiling`; when
    /// none does, the smallest (the head alone in the cache), which `availability` then refuses with its Mac. Nothing is read from
    /// the machine, and the same request on the same Mac takes the same plan.
    package func chosen(ceiling: Int) -> PromptWriterNeed {
        let layers = maps.encoder.layers
        for first in stride(from: layers, through: 0, by: -1) {
            let candidate = placed(streamedFrom: first)
            if candidate.total <= ceiling { return candidate }
        }
        return placed(streamedFrom: 0)
    }

    /// **The need of the feature at its limits** (`Limits`), an image included, on every layer in the cache — what the plan
    /// starts from (`chosen`). Use `atLimits(…, ceiling:)` for the plan the lab judges.
    package static func atLimits(maps: PromptWriterMaps, limits: Limits = Limits(), base: Int = 0,
                                 plan: CapacityPlan = EngineSettings.effective.plan.restricted(allowingANE: false),
                                 threads: Int = EngineSettings.effective.machine.performanceCores) -> PromptWriterNeed {
        var need = PromptWriterNeed(maps: maps, promptTokens: limits.promptTokens, newTokens: limits.newTokens,
                                    patches: limits.imagePatches, plan: plan, threads: threads)
        need.base = base
        return need
    }

    /// **The plan the lab judges before any request**: the need at the limits, placed in `ceiling` (`chosen`).
    package static func atLimits(maps: PromptWriterMaps, limits: Limits = Limits(), base: Int = 0, ceiling: Int,
                                 plan: CapacityPlan = EngineSettings.effective.plan.restricted(allowingANE: false),
                                 threads: Int = EngineSettings.effective.machine.performanceCores) -> PromptWriterNeed {
        atLimits(maps: maps, limits: limits, base: base, plan: plan, threads: threads).chosen(ceiling: ceiling)
    }
}
