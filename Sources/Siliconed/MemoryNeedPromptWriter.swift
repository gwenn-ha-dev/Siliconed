import Foundation

/// **What the lab's writer needs, computed from shapes before its first token** — `MemoryNeed`'s rule: every
/// term is a function its allocator calls (`Qwen3VLGenerator.arenaBytes`, `cacheBytes`, `Artifact.residentBytes`,
/// `Qwen3VLVision.arenaBytes`…), never measured, never fitted.
///
///     resident (all through the writer's life)        ┬─ the 36 layers at the map's width (int8: 6.95 GB, bf16: 13.9 GB)
///       + the head, bf16 as published                 ┘   lm_head 1.24 GB + the final norm
///     + the larger of
///         vision    (Describe only)   tower arena, pixel_values, features, attention, GEMM tiers
///         language                    arena (≤ 512 rows), K/V cache for prompt + new tokens, attention scores,
///                                     features until the prompt is read, GEMM tiers
///     + the process's base (the app's, read from the process: `Request.base`)
///
/// **Why the layers are charged whole.** A render streams its encoder once and the pages are the cache's, nobody's
/// footprint; decoding reads every weight at every token, so the writer keeps them in its own pages
/// (`Artifact.keepResident`) — read once, counted once, and the kernel cannot evict them to read them back at
/// ~3 GB/s, which would make a token cost seconds. The writer runs alone (the lab is cancelled and the weights
/// released when a render starts), so B is the writer's. If the need is above B the writer is unavailable and the
/// answer is a Mac size (`MemoryBudget.macNeeded`): there is no streamed fallback to plan, a streamed decode being
/// too slow to be a product.
///
/// What stays outside the formula, as for the renders: the mapped `embed_tokens` (one row a token, file-backed,
/// clean) and the vision weights (`visual.*`, read once from the map), which are the page cache's.
package struct PromptWriterMaps: Sendable {
    package let encoder: TextEncoder.Config
    package let vision: Qwen3VLVision.Config
    /// The bytes `keepResident` maps for the layers and for the head (page-rounded, from the headers).
    package let residentLayers: Int, residentHead: Int
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
        guard residentLayers > 0, residentHead > 0 else { return nil }
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
        case available(need: Int)
        /// The need is above B: the smallest Mac size (GB) whose B covers it.
        case needsMac(gigabytes: Int, need: Int)
    }

    package let maps: PromptWriterMaps
    /// The prompt's tokens and the answer's bound; `patches`: the image's, `nil` for a text rewrite.
    package let promptTokens: Int, newTokens: Int, patches: Int?
    package let resident: [MemoryNeed.Term]
    package let phases: [MemoryNeed.Phase]
    package var base = 0

    /// - Parameters:
    ///   - plan: the encoder's plan (`CapacityPlan.restricted(allowingANE: false)` of the process's): which engine takes
    ///     rows of which product; `.defaults` (the GPU alone) in the tests.
    ///   - threads: the performance cores the vision attention's head split runs on.
    package init(maps: PromptWriterMaps, promptTokens: Int, newTokens: Int, patches: Int? = nil,
                 plan: CapacityPlan = EngineSettings.effective.plan.restricted(allowingANE: false),
                 threads: Int = EngineSettings.effective.machine.performanceCores) {
        self.maps = maps; self.promptTokens = promptTokens; self.newTokens = newTokens; self.patches = patches
        let e = maps.encoder, v = maps.vision
        let (h, a, kv, i) = (e.hidden, e.attentionWidth, e.kvWidth, e.intermediate)
        let pixels = patches.map { Arena.capacity(for: $0 * v.patchValues * 4) } ?? 0
        let features = patches.map { Qwen3VLVision.featureBytes(tokens: $0 / (v.merge * v.merge), output: v.output,
                                                                levels: v.deepstack.count) } ?? 0
        resident = [MemoryNeed.Term("resident layers", maps.residentLayers), MemoryNeed.Term("resident head", maps.residentHead)]
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
    }

    /// The resident weights.
    package var residentBytes: Int { resident.reduce(0) { $0 + $1.bytes } }

    /// **The peak**: the weights, the larger phase, the process's base.
    package var peak: Int { residentBytes + (phases.map(\.bytes).max() ?? 0) + base }

    /// Whether B covers the need — or the Mac that does. `ceiling`: B (`MemoryBudget.machineCeiling()`).
    package func availability(ceiling: Int) -> Availability {
        peak <= ceiling ? .available(need: peak) : .needsMac(gigabytes: MemoryBudget.macNeeded(for: peak), need: peak)
    }

    /// **The need of the feature at its limits** (`Limits`), an image included — what the lab judges before any request.
    package static func atLimits(maps: PromptWriterMaps, limits: Limits = Limits(), base: Int = 0,
                                 plan: CapacityPlan = EngineSettings.effective.plan.restricted(allowingANE: false),
                                 threads: Int = EngineSettings.effective.machine.performanceCores) -> PromptWriterNeed {
        var need = PromptWriterNeed(maps: maps, promptTokens: limits.promptTokens, newTokens: limits.newTokens,
                                    patches: limits.imagePatches, plan: plan, threads: threads)
        need.base = base
        return need
    }
}
