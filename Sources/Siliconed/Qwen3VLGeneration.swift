import Accelerate
import Foundation
import Metal

/// **Qwen3-VL-8B that writes**: greedy decoding with a K/V cache — `Qwen3VLForConditionalGeneration.generate`
/// with `do_sample=False` (transformers 5.19), in fp32 from the installed encoder map, whatever its width
/// (bf16 Standard, int8 convrot Compact). What the lab's writer runs (`PromptWriter`): one sequence, a
/// system turn and a user turn (a text, or an image then a text) in, the assistant's answer out.
///
///     chat template ─ tokenizer ─ ids (each <|image_pad|> × gh·gw/4) ─ 3D positions (get_rope_index)
///     image ─ smart_resize(32, ≤ 2²⁰ px) ─ Lanczos (PIL) ─ processor ─ vision tower ─ tokens + 3 deepstack
///                                                                          │
///     ┌─ a pass: the pending rows of the sequence (the prompt in chunks of ≤ `prefillRows`, then one row)
///     │    embed_tokens(ids), image rows overwritten ◀─────────────────────────┘
///     │    36 layers, each: q, k, v ─ QK-Norm, mRoPE ─ k, v APPENDED to the sequence's cache
///     │                     attention of the rows over the cache (causal from their offset)
///     │                     after layers 0, 1, 2: + deepstack on the prompt's image rows
///     │    the last row, once the whole prompt has been read:
///     │        final RMSNorm ─ lm_head ─ argmax (first maximum) ─ the next token, at the next text position
///     └─ until <|im_end|> or <|endoftext|> (the stop token is kept), or `maxNewTokens`
///
/// A layer is the encoder's (`TextEncoder`, same weights, same map, read through the same `WeightProducts`):
/// `x += W_o·SDPA(RoPE(norm_h(W_q·n)), RoPE(norm_h(W_k·n)), W_v·n)` with `n = RMSNorm(x)`, then
/// `x += W_down·(silu(W_gate·n') ⊙ W_up·n')`. The encoder's code is not touched: Qwen-Image-2.1 reads it, and its
/// path stays the same to the bit. What this file adds around it is the cache, the head and the loop.
///
/// ## What sets decoding apart, and what a port by analogy would miss
///
/// Read in `modeling_qwen3_vl.py` and `generation/utils.py` (transformers 5.19), and in the Instruct's files:
///
///   - **The template is the Instruct's `apply_chat_template`, not the pipeline's** (`Qwen3VLPrompt` is the
///     pipeline's, with its own system turn and `<imageN>` tags): a system turn exactly when the first
///     message is `system`, then `<|im_start|>user\n`, the image slot `<|vision_start|><|image_pad|><|vision_end|>`
///     with nothing between it and the text, `<|im_end|>\n`, and the generation prompt `<|im_start|>assistant\n`.
///   - **The final RMSNorm and `lm_head` are read** — the two tensors the encoder map leaves out
///     (Qwen-Image reads the state before the norm, and its map must not change by a byte). They come
///     from a map of their own, `ForgeText.forgeHead`, in bf16 as published.
///   - **`lm_head` is untied** (`tie_word_embeddings: false`): 151 936 × 4096 of its own, not
///     `embed_tokens` transposed. Read by rows, like a table, and widened by blocks: never whole in fp32.
///   - **A generated token sits at `max(prompt positions) + 1 + k` on all three axes** (`rope_deltas =
///     max + 1 − len`): after an image the positions run SHORTER than the cache — the k-th new token is
///     at cache row `len + k` but position `max + 1 + k`. On equal axes the interleaved mRoPE is the 1D one.
///   - **Deepstack is added on the prompt's image rows only**, after layers 0, 1, 2; decode steps add
///     nothing (their rows are text).
///   - **Two stop tokens**: `generation_config.eos_token_id` is `[<|im_end|>, <|endoftext|>]`, kept in the
///     head map's header; `generate` keeps the stop token in its output, so does `Output.tokens`; the
///     text drops it (`skip_special_tokens`).
///   - **Greedy ties go to the lowest id** (`torch.argmax` returns the first maximum): a strict `>`, scanned
///     in id order across the blocks of `lm_head`.
///   - **The published `generation_config.json` samples** (`do_sample: true`, temperature 0.7, top-k 20,
///     top-p 0.8): greedy is the engine's choice, checked against `do_sample=False`; it can repeat itself
///     on a long answer where the sampled run would not.
///   - **A byte-level BPE token may hold half a character**: the text is detokenized incrementally
///     (`UTF8Stream`), never one identifier at a time.
///   - **Decoding reads every weight at every token**: 7 GB of int8 (13.9 GB of bf16) and `lm_head`'s 1.24 GB.
///     Whatever does not stay in the page cache is read again from the disk (~3 GB/s) at every token, so the writer's plan
///     (`PromptWriterNeed.chosen`) sizes what B leaves to the cache and streams the rest, and predicts a token's seconds from
///     the card. The weights are the map's clean pages, never a copy of them in the process's own: the kernel can drop a
///     page and read it back, it cannot swap it.
///   - **A step reads them where they lie** (`Qwen3VLDecoder`, `DecodeKernels`): the mapped pages are wrapped as GPU
///     buffers without a copy, and a token is a few command buffers — the product kernels widen the int8 in registers
///     and accumulate in fp32, the rotation goes to the activation, `lm_head` is a bf16 × fp32 GEMV on the GPU with the
///     argmax. The prompt (many rows, compute-bound) keeps the passes below, and so does a map the kernels do not read
///     (`decoder` is `nil`) or `decodesOnGPU = false`.
package final class Qwen3VLGenerator {
    /// The sequence to continue: the prompt's identifiers and 3D positions (`Qwen3VLChat.prepare`) and
    /// its images, in slot order.
    package struct Request {
        package let ids: [Int]
        package let positions: [[Int]]
        package let images: [Qwen3VLImages.Pixels]
        package init(ids: [Int], positions: [[Int]], images: [Qwen3VLImages.Pixels]) {
            self.ids = ids; self.positions = positions; self.images = images
        }
    }

    /// What the sequence wrote.
    package struct Output {
        /// The generated identifiers, the stop token included when one was reached.
        package let tokens: [Int]
        package let promptTokens: Int
        /// Whether a stop token ended it (otherwise `maxNewTokens`).
        package let stopped: Bool
        /// Seconds from the start of `generate` to the last token.
        package let seconds: Double
    }

    package enum Failure: Error, CustomStringConvertible {
        case notAHead(String)
        case incomplete(String)
        package var description: String {
            switch self {
            case .notAHead(let m): return "Qwen3-VL head map: \(m)"
            case .incomplete(let m): return "Qwen3-VL generation: \(m)"
            }
        }
    }

    package let encoder: Artifact
    package let head: Artifact
    package let config: TextEncoder.Config
    package let vision: Qwen3VLVision.Config
    package let tokens: Qwen3VLPrompt.Tokens
    /// The head map's `eos_token_ids` (`generation_config.eos_token_id`).
    package let stops: Set<Int>
    package let vocabulary: Int

    package var cancellation: Cancellation?
    /// A decode step runs on the GPU from the resident weights when `load` could build the decoder; the CPU's passes
    /// otherwise. `SILICONED_WRITER_CPU_DECODE=1` keeps the CPU's (to compare the two on the same prompt).
    package var decodesOnGPU = ProcessInfo.processInfo.environment["SILICONED_WRITER_CPU_DECODE"] != "1"
    /// The GPU decoder over the resident tensors, built by `load` (`nil`: not resident, or a map it does not read).
    package private(set) var decoder: Qwen3VLDecoder?
    /// Why `load` left `decoder` unset (the maps the kernels do not read, no GPU), `nil` when it is built.
    package private(set) var decoderNote: String?
    /// The top-1 / top-2 logit gap of each token the last `generate` chose, in order: 0 is a tie, a small one is where
    /// two engines' roundings could choose differently.
    package private(set) var gaps: [Float] = []

    /// **Teacher forcing, for the check against the reference**: the sequence continues with `tokens` (the
    /// reference's) whatever the engine would choose, and `observe` hears each step's engine choice (its
    /// argmax, the first maximum) and the whole row of logits, before the next pass. The engine's tokens equal
    /// the reference's exactly when every step's choice does: a free run never leaves the forced path before
    /// the first difference, so the one run proves both the tokens and the logits. Nil: a free run.
    package struct Teacher {
        package let tokens: [Int]
        package let observe: (_ step: Int, _ choice: Int, _ logits: UnsafeBufferPointer<Float>) -> Void
        package init(tokens: [Int], observe: @escaping (_ step: Int, _ choice: Int, _ logits: UnsafeBufferPointer<Float>) -> Void) {
            self.tokens = tokens; self.observe = observe
        }
    }
    package var teacher: Teacher?
    /// Seconds, by phase: `vision`, `prefill`, `decode`, `head` (`lm_head`, inside the other two).
    package private(set) var timings: [String: Double] = [:]
    /// Passes over the 36 layers: each reads every layer once.
    package private(set) var passes = 0

    /// The prompt rows a pass computes at most: 512 rows of arena are 122 MB, where 2048 would be 488 MB, and a
    /// prompt of 1 500 tokens costs three passes over resident weights instead of one.
    package static let prefillRows = 512
    /// `lm_head` is widened and multiplied by blocks of this many rows (134 MB in fp32).
    package static let headBlock = 8192
    /// The query rows an attention task takes at once.
    package static let attentionBlock = 64

    package init(encoder: Artifact, head: Artifact, tokenizer: Tokenizer) throws {
        let config = try TextEncoder.Config(header: encoder.header)
        guard config.mropeSections != nil else { throw Failure.incomplete("the encoder map declares no mRoPE: not Qwen3-VL") }
        guard encoder.header["final_norm"] as? Bool != true else {
            throw Failure.incomplete("the encoder map applies the final norm itself: not the Qwen3-VL encoder map")
        }
        let published = (encoder.header["config"] as? [String: Any])?["num_hidden_layers"] as? Int
        guard published == config.layers else {
            throw Failure.incomplete("the map holds \(config.layers) layers of \(published ?? 0): generation needs them all")
        }
        guard head.header["kind"] as? String == "qwen3-vl-lm-head" else {
            throw Failure.notAHead("kind \(head.header["kind"] ?? "none")")
        }
        guard let lm = head.tensors["lm_head.weight"], lm.shape.count == 2, lm.shape[1] == config.hidden, lm.dtype == .bfloat16,
              let norm = head.tensors["norm.weight"], norm.shape == [config.hidden] else {
            throw Failure.notAHead("bf16 lm_head.weight [V, \(config.hidden)] and norm.weight [\(config.hidden)] expected")
        }
        guard let eos = head.header["eos_token_ids"] as? [Int], !eos.isEmpty else { throw Failure.notAHead("eos_token_ids missing") }
        guard let table = encoder.tensors["embed_tokens.weight"], table.shape[0] == lm.shape[0] else {
            throw Failure.notAHead("\(lm.shape[0]) logits for an embedding table of \(encoder.tensors["embed_tokens.weight"]?.shape[0] ?? 0)")
        }
        self.encoder = encoder; self.head = head; self.config = config
        self.vision = try Qwen3VLVision.Config(header: encoder.header)
        self.tokens = try Qwen3VLPrompt.Tokens(header: encoder.header, tokenizer: tokenizer)
        self.stops = Set(eos); self.vocabulary = lm.shape[0]
    }

    // MARK: - Residency

    /// **The tensors a decoding step reads, by name**: the 36 layers of the encoder map and the head's two.
    /// (`embed_tokens` is one row a token, left in the map; the vision tower runs once, from the map.)
    package static func residentPrefixes(config: TextEncoder.Config) -> (encoder: [String], head: [String]) {
        (TextEncoder.layerPrefixes(config), ["lm_head.weight", "norm.weight"])
    }

    /// **Takes the weights where they lie** and builds the GPU decoder over them. Nothing is copied or read here: the layers
    /// before `streamedFrom` and the head are the map's own pages (the cache's: clean, dropped and read again under pressure,
    /// never swapped), the layers from `streamedFrom` on are read at each token into staging buffers (`TailFeeder`).
    /// `streamedFrom` ≥ the layer count: nothing streamed. Called again with another plan, it rebuilds the decoder — the writer
    /// chooses its plan at every run, in B, and a longer prompt can leave the cache less room.
    ///
    /// `progress` is called with 1 once the decoder is built (there is no longer a long read to report).
    package func load(streamedFrom: Int? = nil, progress: ((Double) -> Void)? = nil) throws {
        try cancellation.check()
        decoder = nil; decoderNote = nil
        let tailFrom = min(streamedFrom ?? config.layers, config.layers)
        loadedStreamedFrom = tailFrom
        progress?(1)
        guard decodesOnGPU else { decoderNote = "GPU decoding switched off"; return }
        do {
            let tail: Qwen3VLDecoder.Tail? = tailFrom < config.layers
                ? .init(path: encoder.path, first: tailFrom, runs: try Qwen3VLDecoder.runs(of: encoder, layers: tailFrom..<config.layers),
                        slots: Self.stagingSlots)
                : nil
            decoder = try Qwen3VLDecoder.make(encoder: encoder, head: head, config: config, vocabulary: vocabulary, tail: tail)
        } catch {
            decoderNote = "\(error)"   // the CPU's passes decode instead, slowly: said, not hidden
        }
    }

    /// The first streamed layer of the last `load` (the layer count: none); `nil` before any.
    package private(set) var loadedStreamedFrom: Int?

    /// The staging buffers a streamed tail reads through: one layer being computed, one being read. (The disk is the pace of a
    /// streamed token, so a third would only be memory.)
    package static let stagingSlots = 2

    /// Gives the decoder back (the GPU's views of the pages first).
    package func unload() { decoder = nil; loadedStreamedFrom = nil; encoder.releaseResident(); head.releaseResident() }

    // MARK: - Memory

    /// **The language phase's arena**, in floats, in the order `generate` reserves it: the activations
    /// of `rows` rows (`x`, `normed`, `projected` at `hidden`; `q`, `attended` at `heads·headDim`; `k`, `v`
    /// at `kvHeads·headDim`; `gate`, `up`, `swiglu` at `intermediate`; the RoPE rows), the norm weights'
    /// scratch, the reserve of the largest weight (`hidden × intermediate`, as `TextEncoder`), the last row,
    /// the final norm's weight, one block of `lm_head` widened and its logits.
    package static func slices(config: TextEncoder.Config, rows r: Int) -> [(name: String, count: Int)] {
        let h = config.hidden, a = config.attentionWidth, kv = config.kvWidth, i = config.intermediate
        return [("x", r * h), ("normed", r * h), ("projected", r * h), ("q", r * a), ("k", r * kv), ("v", r * kv),
                ("attended", r * a), ("gate", r * i), ("up", r * i), ("swiglu", r * i), ("freqs", r * config.headDim),
                ("scratch", max(h, config.headDim)), ("finalNorm", h), ("reserve", h * i),
                ("last", h), ("headRows", headBlock * h), ("logits", headBlock)]
    }

    /// The GPU decoder's scratch (`Qwen3VLDecoder.scratchBytes`): alive from `load` to `unload`, whatever the prompt.
    package static func decoderBytes(config: TextEncoder.Config, vocabulary: Int) -> Int {
        Qwen3VLDecoder.scratchBytes(.init(config, vocabulary: vocabulary))
    }

    /// The arena `generate` maps for a prompt of `prompt` tokens.
    package static func arenaBytes(config: TextEncoder.Config, prompt: Int) -> Int {
        Arena.capacity(slices: slices(config: config, rows: rows(prompt: prompt)).map { $0.count * 4 })
    }

    /// The rows of the widest pass: the prompt's chunk, or the one row of a step.
    package static func rows(prompt: Int) -> Int { max(1, min(prefillRows, prompt)) }

    /// The sequence's K/V cache: keys and values of every layer for its prompt and the tokens it may write.
    package static func cacheBytes(config: TextEncoder.Config, length: Int) -> Int {
        Arena.capacity(for: 2 * config.layers * length * config.kvWidth * 4)
    }

    /// The attention's scores, one block of query rows by the longest sequence, per worker.
    package static func attentionScratchBytes(longest: Int, workers: Int = Parallel.threads) -> Int {
        Arena.capacity(for: workers * attentionBlock * longest * 4)
    }

    // MARK: - Generation

    /// **Greedy continuation**: each pass computes the pending rows of the sequence (the prompt, cut where
    /// `prefillRows` falls, then the last token written), and, once it has read all of them, the next token.
    /// `onToken` hears each identifier as it is chosen (the stop token included), before the next pass.
    package func generate(_ request: Request, maxNewTokens: Int,
                          onToken: ((Int) throws -> Void)? = nil) throws -> Output {
        timings = [:]; passes = 0; gaps = []
        let started = Date()
        let h = config.hidden, a = config.attentionWidth, kv = config.kvWidth, i = config.intermediate
        let dh = config.headDim
        let count = request.ids.count
        guard request.positions.count == 3, request.positions.allSatisfy({ $0.count == count }), count > 0 else {
            throw Failure.incomplete("three axes of \(count) positions are needed")
        }

        // ── the vision tower, image by image; gone before the language model allocates ──────────
        var features: [Qwen3VLVision.Features] = []
        if let largest = request.images.map(\.patches).max() {
            let t0 = Date()
            let tower = try Qwen3VLVision(artifact: encoder, config: vision, maximumPatches: largest)
            tower.cancellation = cancellation
            features = try request.images.map { try tower.encode($0) }
            timings["vision"] = Date().timeIntervalSince(t0)
            PhaseBoundary.mark("vision")
        }
        // Token index → (image, merged row) for the prompt's `<|image_pad|>` rows, in sequence order.
        var imageSource: [Int: (image: Int, row: Int)] = [:]
        var next = 0, row = 0
        for (index, id) in request.ids.enumerated() where id == tokens.image {
            while next < features.count && row == features[next].tokens { next += 1; row = 0 }
            guard next < features.count else { throw Failure.incomplete("more <|image_pad|> rows than image features") }
            imageSource[index] = (next, row); row += 1
        }
        guard imageSource.count == features.reduce(0, { $0 + $1.tokens }) else {
            throw Failure.incomplete("\(imageSource.count) <|image_pad|> rows for \(features.reduce(0) { $0 + $1.tokens }) image features")
        }
        guard maxNewTokens > 0 else { return Output(tokens: [], promptTokens: count, stopped: false, seconds: 0) }

        // ── the language phase's arena and buffers ───────────────────────────────────────────────
        let rows = Self.rows(prompt: count)
        let plan = Self.slices(config: config, rows: rows)
        let arena = try Arena(capacity: Arena.capacity(slices: plan.map { $0.count * 4 }))
        var slice: [String: UnsafeMutablePointer<Float>] = [:]
        for (name, n) in plan { slice[name] = try arena.reserve(name, bytes: n * 4).assumingMemoryBound(to: Float.self) }
        let x = slice["x"]!, normed = slice["normed"]!, projected = slice["projected"]!
        let q = slice["q"]!, k = slice["k"]!, v = slice["v"]!, attended = slice["attended"]!
        let gate = slice["gate"]!, up = slice["up"]!, swiglu = slice["swiglu"]!, freqs = slice["freqs"]!
        let scratch = slice["scratch"]!, finalNorm = slice["finalNorm"]!, reserve = slice["reserve"]!
        let last = slice["last"]!, headRows = slice["headRows"]!, logits = slice["logits"]!
        // The encoder's own GEMM phase: the plan asks the encoder's questions of the encoder's products.
        let gemm = try GEMM(phase: .encoding)
        let normedBuf = try gemm.wrap(normed, bytes: rows * h * 4, name: "normed")
        let projectedBuf = try gemm.wrap(projected, bytes: rows * h * 4, name: "projected")
        let qBuf = try gemm.wrap(q, bytes: rows * a * 4, name: "q")
        let kBuf = try gemm.wrap(k, bytes: rows * kv * 4, name: "k")
        let vBuf = try gemm.wrap(v, bytes: rows * kv * 4, name: "v")
        let attendedBuf = try gemm.wrap(attended, bytes: rows * a * 4, name: "attended")
        let gateBuf = try gemm.wrap(gate, bytes: rows * i * 4, name: "gate")
        let upBuf = try gemm.wrap(up, bytes: rows * i * 4, name: "up")
        let products = WeightProducts(gemm: gemm, artifact: encoder, reserve: reserve, capacity: h * i, transposed: true)
        let capacity = count + maxNewTokens
        let cache = Pages(bytes: Self.cacheBytes(config: config, length: capacity))
        let layerStride = capacity * kv
        func keys(_ layer: Int) -> UnsafeMutablePointer<Float> { cache.floats() + layer * layerStride }
        func values(_ layer: Int) -> UnsafeMutablePointer<Float> { cache.floats() + (config.layers + layer) * layerStride }
        let scores = Pages(bytes: Self.attentionScratchBytes(longest: capacity))
        let gpu = decodesOnGPU ? decoder : nil
        try gpu?.attach(cache: cache, layerStride: layerStride)
        defer { gpu?.detachCache() }   // before `cache` is given back
        try widenExactly(head, "norm.weight", count: h, into: finalNorm)
        let embedRows = encoder.tensors["embed_tokens.weight"]!.shape[0]

        var ids = request.ids, positions = request.positions
        var nextPosition = Qwen3VLChat.firstGeneratedPosition(request.positions)
        var processed = 0
        var generated: [Int] = [], stopped = false
        let rowOfLogits: UnsafeMutablePointer<Float>? = teacher.map { _ in .allocate(capacity: vocabulary) }
        defer { rowOfLogits?.deallocate() }

        /// The token chosen, appended where the next pass reads it.
        func accept(_ pick: Int, gap: Float) throws {
            generated.append(pick)
            gaps.append(gap)
            ids.append(pick)
            for axis in 0..<3 { positions[axis].append(nextPosition) }
            nextPosition += 1
            stopped = stops.contains(pick)
            try onToken?(pick)
        }

        /// With a teacher, the engine's choice and its row of logits are heard, and the reference's token goes on;
        /// on either path, the GPU's logits being the decoder's own buffer (`logitValues`).
        func taught(_ choice: Int, logits row: UnsafePointer<Float>?) -> Int {
            guard let teacher, let row else { return choice }
            teacher.observe(generated.count, choice, UnsafeBufferPointer(start: row, count: vocabulary))
            return generated.count < teacher.tokens.count ? teacher.tokens[generated.count] : choice
        }

        while !stopped && generated.count < maxNewTokens {
            try cancellation.check()
            // ── the pass's rows ──────────────────────────────────────────────────────────────────
            let start = processed, used = min(ids.count - processed, rows)
            let prefill = start < count
            let passStarted = Date()

            // ── a decode step, on the GPU from the resident weights: the layers, the head and the argmax ──
            if let gpu, !prefill {
                let id = ids[start]
                guard id >= 0, id < embedRows else { throw Failure.incomplete("token \(id) outside the embedding table") }
                try encoder.materializeRows("embed_tokens.weight", first: id, count: 1, into: gpu.hidden)
                TextEncoder.writeRope(into: gpu.frequencies, positions: (0..<3).map { [positions[$0][start]] }, count: 1,
                                      theta: config.theta, headDim: dh, sections: config.mropeSections)
                let pick = try gpu.step(row: start)
                timings["gpu", default: 0] += gpu.lastStep.gpu
                passes += 1
                processed += 1
                try accept(taught(pick.id, logits: gpu.logitValues), gap: pick.gap)
                timings["decode", default: 0] += Date().timeIntervalSince(passStarted)
                continue
            }
            var axes: [[Int]] = [[], [], []]
            for r in 0..<used {
                let t = start + r, destination = x + r * h
                for axis in 0..<3 { axes[axis].append(positions[axis][t]) }
                if let source = imageSource[t] {
                    destination.update(from: features[source.image].embeddings + source.row * h, count: h)
                } else {
                    let id = ids[t]
                    guard id >= 0, id < embedRows else { throw Failure.incomplete("token \(id) outside the embedding table") }
                    try encoder.materializeRows("embed_tokens.weight", first: id, count: 1, into: destination)
                }
            }
            TextEncoder.writeRope(into: freqs, positions: axes, count: used, theta: config.theta, headDim: dh,
                                  sections: config.mropeSections)

            for layer in 0..<config.layers {
                try cancellation.check()
                let p = "layers.\(layer)."

                // ── attention ──────────────────────────────────────────────────────────────────
                try widenExactly(encoder, p + "input_layernorm.weight", count: h, into: scratch)
                Ops.rmsNorm(x, weight: scratch, into: normed, rows: used, columns: h, eps: config.eps)
                try products.linear(p + "self_attn.q_proj.weight", a: normedBuf, c: qBuf, m: used, k: h, n: a)
                try products.linear(p + "self_attn.k_proj.weight", a: normedBuf, c: kBuf, m: used, k: h, n: kv)
                try products.linear(p + "self_attn.v_proj.weight", a: normedBuf, c: vBuf, m: used, k: h, n: kv)
                try widenExactly(encoder, p + "self_attn.q_norm.weight", count: dh, into: scratch)
                Ops.rmsNormRope(q, weight: scratch, freqs: freqs, sequence: used, heads: config.heads, headDim: dh, eps: config.eps)
                try widenExactly(encoder, p + "self_attn.k_norm.weight", count: dh, into: scratch)
                Ops.rmsNormRope(k, weight: scratch, freqs: freqs, sequence: used, heads: config.kvHeads, headDim: dh, eps: config.eps)
                (keys(layer) + start * kv).update(from: k, count: used * kv)
                (values(layer) + start * kv).update(from: v, count: used * kv)
                Self.cachedAttention(q: q, rows: used, keys: keys(layer), values: values(layer), offset: start,
                                     heads: config.heads, kvHeads: config.kvHeads, headDim: dh, into: attended,
                                     scratch: scores.floats(), scratchFloats: scores.capacity / 4)
                try products.linear(p + "self_attn.o_proj.weight", a: attendedBuf, c: projectedBuf, m: used, k: a, n: h)
                vDSP_vadd(x, 1, projected, 1, x, 1, vDSP_Length(used * h))

                // ── feed-forward network ───────────────────────────────────────────────────────
                try widenExactly(encoder, p + "post_attention_layernorm.weight", count: h, into: scratch)
                Ops.rmsNorm(x, weight: scratch, into: normed, rows: used, columns: h, eps: config.eps)
                try products.linear(p + "mlp.gate_proj.weight", a: normedBuf, c: gateBuf, m: used, k: h, n: i)
                try products.linear(p + "mlp.up_proj.weight", a: normedBuf, c: upBuf, m: used, k: h, n: i)
                Ops.siluGate(gate, up, into: gate, count: used * i, scratch: swiglu)
                try products.linear(p + "mlp.down_proj.weight", a: gateBuf, c: projectedBuf, m: used, k: i, n: h)
                vDSP_vadd(x, 1, projected, 1, x, 1, vDSP_Length(used * h))

                // `_deepstack_process`: the image rows of the prompt, after layers 0, 1, 2.
                if layer < vision.deepstack.count {
                    for r in 0..<used {
                        guard let source = imageSource[start + r] else { continue }
                        let destination = x + r * h
                        vDSP_vadd(destination, 1, features[source.image].deepstack(layer) + source.row * h, 1,
                                  destination, 1, vDSP_Length(h))
                    }
                }
            }
            passes += 1
            processed += used

            // ── the next token, once every row has been read ─────────────────────────────────────
            if processed == ids.count {
                let headStarted = Date()
                let best: Top2
                if let gpu {
                    // The prompt's last row: the final norm, `lm_head` and the argmax on the GPU.
                    gpu.hidden.update(from: x + (used - 1) * h, count: h)
                    let pick = try gpu.pick()
                    best = Top2(id: pick.id, value: pick.value, second: pick.second)
                } else {
                    last.update(from: x + (used - 1) * h, count: h)
                    Ops.rmsNorm(last, weight: finalNorm, into: last, rows: 1, columns: h, eps: config.eps)
                    best = try greedy(last, headRows: headRows, logits: logits, row: rowOfLogits)
                }
                timings["head", default: 0] += Date().timeIntervalSince(headStarted)
                try accept(taught(best.id, logits: gpu?.logitValues ?? rowOfLogits.map { UnsafePointer($0) }),
                           gap: best.value - best.second)
            }
            timings[prefill ? "prefill" : "decode", default: 0] += Date().timeIntervalSince(passStarted)
            if prefill && processed == count { features = [] }   // read by the prompt's passes, not by a step
        }
        return Output(tokens: generated, promptTokens: count, stopped: stopped, seconds: Date().timeIntervalSince(started))
    }

    private func widenExactly(_ artifact: Artifact, _ name: String, count: Int, into destination: UnsafeMutablePointer<Float>) throws {
        let got = try artifact.materialize(name, into: destination, capacity: count)
        guard got == count else { throw Artifact.Failure.badHeader("\(name): \(got) values, \(count) expected") }
    }

    /// `lm_head` on the normalized row, block of rows by block of rows (widened, then one product),
    /// and the argmax — the first maximum, as `torch.argmax`.
    private func greedy(_ hidden: UnsafePointer<Float>, headRows: UnsafeMutablePointer<Float>,
                        logits: UnsafeMutablePointer<Float>, row: UnsafeMutablePointer<Float>? = nil) throws -> Top2 {
        let h = config.hidden
        var best = Top2()
        var first = 0
        while first < vocabulary {
            let n = min(Self.headBlock, vocabulary - first)
            try head.materializeRows("lm_head.weight", first: first, count: n, into: headRows)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, Int32(n), Int32(h), 1,
                        hidden, Int32(h), headRows, Int32(h), 0, logits, Int32(n))
            best.scan(logits, count: n, base: first)
            if let row { (row + first).update(from: logits, count: n) }
            first += n
        }
        return best
    }

    /// **The largest logit and the runner-up**, scanned block after block in id order: a strict `>`
    /// keeps the first maximum (`torch.argmax`). The runner-up says how close greedy was to another choice.
    package struct Top2 {
        package var id = -1
        package var value = -Float.infinity
        package var second = -Float.infinity
        package init() {}
        package init(id: Int, value: Float, second: Float) { self.id = id; self.value = value; self.second = second }
        package mutating func scan(_ values: UnsafePointer<Float>, count: Int, base: Int) {
            for j in 0..<count {
                let x = values[j]
                if x > value { second = value; value = x; id = base + j } else if x > second { second = x }
            }
        }
    }

    /// **Causal attention of `rows` new query rows over a sequence's cache**: row `r` sits at cache row
    /// `offset + r` and attends to rows `0 ... offset + r` (the cache already holds the new rows' keys
    /// and values). `q`, `out`: `[rows, heads, headDim]`; `keys`, `values`: `[offset + rows, kvHeads,
    /// headDim]`; query head `h` reads key head `h / (heads / kvHeads)` (`repeat_kv`). Scores scaled by
    /// `1/√headDim`, softmax in fp32 — `F.scaled_dot_product_attention`'s arithmetic, summed in another order.
    /// Tasks (a head × a block of `attentionBlock` rows) spread over the workers; `scratch` holds
    /// `workers × attentionBlock × (offset + rows)` floats at least.
    package static func cachedAttention(q: UnsafePointer<Float>, rows: Int, keys: UnsafePointer<Float>,
                                        values: UnsafePointer<Float>, offset: Int, heads: Int, kvHeads: Int,
                                        headDim: Int, into out: UnsafeMutablePointer<Float>,
                                        scratch: UnsafeMutablePointer<Float>, scratchFloats: Int) {
        let block = attentionBlock, blocks = (rows + block - 1) / block, tasks = heads * blocks
        let length = offset + rows
        let workers = max(1, min(tasks, scratchFloats / (block * length)))
        precondition(workers >= 1 && block * length <= scratchFloats, "attention scratch: \(scratchFloats) floats for \(block) × \(length)")
        let group = heads / kvHeads, scale = 1 / Float(headDim).squareRoot()
        let qStride = heads * headDim, kvStride = kvHeads * headDim
        nonisolated(unsafe) let q = q
        nonisolated(unsafe) let keys = keys
        nonisolated(unsafe) let values = values
        nonisolated(unsafe) let out = out
        nonisolated(unsafe) let scratch = scratch
        DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let s = scratch + worker * block * length
            var task = worker
            while task < tasks {
                let head = task % heads, b = task / heads
                let r0 = b * block, n = min(block, rows - r0), width = offset + r0 + n
                let kv = head / group
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(n), Int32(width), Int32(headDim), scale,
                            q + r0 * qStride + head * headDim, Int32(qStride), keys + kv * headDim, Int32(kvStride),
                            0, s, Int32(width))
                for r in 0..<n {
                    let row = s + r * width, valid = offset + r0 + r + 1
                    var top: Float = 0
                    vDSP_maxv(row, 1, &top, vDSP_Length(valid))
                    var negative = -top
                    vDSP_vsadd(row, 1, &negative, row, 1, vDSP_Length(valid))
                    var count = Int32(valid)
                    vvexpf(row, row, &count)
                    var sum: Float = 0
                    vDSP_sve(row, 1, &sum, vDSP_Length(valid))
                    vDSP_vsdiv(row, 1, &sum, row, 1, vDSP_Length(valid))
                    if valid < width { (row + valid).update(repeating: 0, count: width - valid) }
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(n), Int32(headDim), Int32(width), 1,
                            s, Int32(width), values + kv * headDim, Int32(kvStride), 0,
                            out + r0 * qStride + head * headDim, Int32(qStride))
                task += workers
            }
        }
    }
}

/// **The prompt side of generation**: the chat template, the image's size, the position of the first
/// generated token. Pure functions.
package enum Qwen3VLChat {
    package static let imageSlot = "<|vision_start|><|image_pad|><|vision_end|>"

    /// `processor.apply_chat_template(messages, add_generation_prompt=True)` with the Instruct's chat template,
    /// `messages` = an optional `system` turn, then one `user` turn: the image slots one after the other, the
    /// text right after them. A system turn is written exactly when one is given (the template writes one
    /// only if the first message is `system`, whatever its content).
    package static func template(system: String? = nil, user: String, images: Int) -> String {
        (system.map { "<|im_start|>system\n" + $0 + "<|im_end|>\n" } ?? "")
            + "<|im_start|>user\n" + String(repeating: imageSlot, count: images) + user
            + "<|im_end|>\n<|im_start|>assistant\n"
    }

    /// The same, for the instruction of a caption (no system turn).
    package static func template(instruction: String, images: Int) -> String {
        template(system: nil, user: instruction, images: images)
    }

    package enum Failure: Error, CustomStringConvertible, Equatable {
        /// A text spells one of the template's control tokens: the tokenizer would read it
        /// as the token, not as text.
        case reservedText(String)
        package var description: String {
            switch self { case .reservedText(let t): return "the text contains \(t), a control token of Qwen3-VL's chat template" }
        }
    }

    /// The control tokens of the template that a text must not spell.
    package static let reserved = ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
                                   "<|im_start|>", "<|im_end|>", "<|endoftext|>"]

    /// The template tokenized, each `<|image_pad|>` expanded to its image's merged tokens, and the 3D
    /// positions `get_rope_index` gives it. `grids`: each image's `(gh, gw)`.
    package static func prepare(system: String? = nil, user: String, grids: [(Int, Int)], tokenizer: Tokenizer,
                                tokens: Qwen3VLPrompt.Tokens, merge: Int = 2) throws -> (ids: [Int], positions: [[Int]]) {
        for text in reserved where user.contains(text) || (system?.contains(text) ?? false) { throw Failure.reservedText(text) }
        let raw = tokenizer.encode(template(system: system, user: user, images: grids.count))
        let ids = try Qwen3VLPrompt.expand(raw, image: tokens.image, counts: grids.map { $0.0 * $0.1 / (merge * merge) })
        return (ids, Qwen3VLPrompt.positions(ids: ids, image: tokens.image, grids: grids, merge: merge))
    }

    /// **At most 2²⁰ pixels an image**, so at most 1 024 image tokens — the largest condition
    /// Qwen-Image-2.1 hands this tower (1024²). The processor's own bound (2²⁴) would let a 4K scan
    /// cost 16 384 tokens of vision attention and cache; a description does not need them.
    package static let maxPixels = 1 << 20

    /// `smart_resize(h, w, factor = 32, max_pixels = maxPixels)`: each side to the nearest multiple of 32,
    /// scaled down under `maxPixels`. The image is brought there by PIL's Lanczos (`pixels`), so that the
    /// processor's own resize (bicubic, not ported) is the identity — the oracle does the same.
    package static func captionSize(width: Int, height: Int, maxPixels: Int = maxPixels) -> (width: Int, height: Int) {
        let size = Qwen3VLImages.smartResize(height: height, width: width, factor: 32, maxPixels: maxPixels)
        return (size.width, size.height)
    }

    /// An image as the writer reads it: Lanczos to `captionSize`, then the processor.
    package static func pixels(_ image: Qwen3VLImages.RGB8, maxPixels: Int = maxPixels) throws
        -> (resized: Qwen3VLImages.RGB8, pixels: Qwen3VLImages.Pixels) {
        let size = captionSize(width: image.width, height: image.height, maxPixels: maxPixels)
        let resized = PillowResample.lanczos(image, width: size.width, height: size.height)
        return (resized, try Qwen3VLImages.pixels(resized))
    }

    /// The position of the first generated token: `max(positions) + 1` (`rope_deltas` of
    /// `get_rope_index`, applied to the cache's length), the same on the three axes.
    package static func firstGeneratedPosition(_ positions: [[Int]]) -> Int {
        (positions.flatMap { $0 }.max() ?? -1) + 1
    }

    /// The generated text, as `batch_decode(skip_special_tokens=True)` gives it: the special tokens left out.
    package static func text(_ generated: [Int], tokenizer: Tokenizer) -> String {
        tokenizer.decode(generated.filter { !tokenizer.specialIdentifiers.contains($0) })
    }
}

/// **An incremental UTF-8 decoder**: bytes in, the text they complete out. A byte-level BPE token can end in
/// the middle of a character (an accent is two bytes, a kanji three, an emoji four, and the tokens cut
/// where the merges fell), so the bytes of an incomplete tail wait for the next token; `finish` turns a tail
/// that never completed into U+FFFD, as `String(decoding:)` (`errors="replace"`) does. Everything else is
/// decoded by `String(decoding:)` itself, so the concatenation of what `push` and `finish` return is
/// that call's result on all the bytes, whatever the cuts.
package struct UTF8Stream {
    private var pending: [UInt8] = []

    package init() {}

    /// The text completed by `bytes`; an incomplete character at their end is kept for the next call.
    package mutating func push(_ bytes: [UInt8]) -> String {
        pending += bytes
        let keep = UTF8Stream.incompleteTail(pending)
        let head = pending.count - keep
        guard head > 0 else { return "" }
        let text = String(decoding: pending[0..<head], as: UTF8.self)
        pending.removeFirst(head)
        return text
    }

    /// What is left: a character cut short, replaced.
    package mutating func finish() -> String {
        defer { pending = [] }
        return pending.isEmpty ? "" : String(decoding: pending, as: UTF8.self)
    }

    /// The length of the longest suffix that is a proper prefix of a well-formed UTF-8 sequence (Unicode
    /// Table 3-7: `E0` wants `A0…BF` next, `ED` `80…9F`, `F0` `90…BF`, `F4` `80…8F`).
    package static func incompleteTail(_ bytes: [UInt8]) -> Int {
        for back in 1...min(3, bytes.count) {
            let lead = bytes[bytes.count - back]
            let needed: Int
            switch lead {
            case 0xC2...0xDF: needed = 2
            case 0xE0...0xEF: needed = 3
            case 0xF0...0xF4: needed = 4
            case 0x80...0xBF: continue          // a continuation byte: look further back for its lead
            default: return 0                   // ASCII, or a byte no sequence starts with
            }
            guard back < needed else { return 0 }   // complete (or longer than needed: invalid, decoded now)
            let tail = bytes[(bytes.count - back + 1)...]
            for (offset, byte) in tail.enumerated() {
                var range: ClosedRange<UInt8> = 0x80...0xBF
                if offset == 0 {
                    switch lead {
                    case 0xE0: range = 0xA0...0xBF
                    case 0xED: range = 0x80...0x9F
                    case 0xF0: range = 0x90...0xBF
                    case 0xF4: range = 0x80...0x8F
                    default: break
                    }
                }
                guard range.contains(byte) else { return 0 }
            }
            return back
        }
        return 0
    }
}
