import Foundation
import os
import Synchronization

/// **The memory a render needs, computed before its first computation from its inputs** — never read
/// from the machine, never fitted on it, never looked up in a table of measured renders.
///
///     Request ── family's phases ──┬─ text        ┐
///      (family, W×H, references,   ├─ vision      │  peak = max over phases of
///       prompt tokens, LoRA ranks, ├─ references  │        (what is alive at the phase's peak)
///       batch, maps, plan)         ├─ denoising   │        + what crosses the phases
///                                  └─ decoding    ┘        + the process's base
///
/// **Every term is a shape function its allocator calls** — `DiT.arenaBytes`, `TextEncoder.arenaBytes`,
/// `HeadSplitAttention.hostBytes`, `Attention.maskBytes`, `Conductor.workspaceBytes`, `GEMM.tierBytes`,
/// `Tokenizer.tableBytes`, `VAE.plan`, … — so the formula equals the allocation by construction: an
/// arena, a `Pages`, a VAE heap is mapped at the size the formula charges. What no constructor of ours
/// sizes but a shape still gives: MPSGraph's copy of the queries (`OperatorFootprint.sdpaQueries`) and
/// its VAE SDPA's copies (`VAELayout.sdpaOutside`).
///
/// **What B covers** (`MemoryBudget`): the whole process — the base it already holds when the render
/// starts (`Request.base`: the app's interface and history, or the command line's runtime, read from
/// the process itself, never from the kernel's state of the others), plus this formula. **What it does
/// not charge**: the frameworks' constants no shape gives — MPSGraph's compiled SDPA, MPS's GEMM state,
/// Metal's first use in a process, what a phase leaves to the next (`OperatorFootprint`). Removed where
/// the structure allows, never folded in as figures: the check reports what is left of them.
///
/// A phase charges **only what is alive at its peak**, following the pipeline's real order: a
/// tokenizer dropped before its encoder maps its arena, a half-size DiT released before the full one
/// is built, a VAE's heaps released before the next phase are not charged beside what follows them.
///
/// **The verification is the product's** (`MemoryCheck`): every render reads its own peak
/// (`ri_interval_max_phys_footprint`) and compares it with this; a peak above the need is a defect of
/// the code — an allocation outside our hand found —, never a figure to fold back in here.
package struct MemoryNeed: Sendable, Equatable {

    /// **What a render's need depends on** — every input that makes it larger.
    package struct Request: Sendable {
        package var card: ModelCard
        package var width: Int
        package var height: Int
        /// Each reference image at its own size (named subjects' included; Z-Image's img2img start),
        /// in pixels as handed to the engine.
        package var references: [(width: Int, height: Int)] = []
        /// The prompt's tokens (template included, as the encoder sees them) — or an upper bound of
        /// them (`promptTokens(_:family:references:)`).
        package var promptTokens: Int
        /// The rank of each LoRA of the stack (`LoRA.layerRanks`): their sum bounds the cumulative
        /// rank the arenas reserve, their count says whether a `down` is staged (two LoRAs or more).
        package var loraRanks: [Int] = []
        /// The DiT's map plan (tail fraction and staging buffers).
        package var tail: MemoryPlan.Tail = .init(fraction: 0.5, streamed: true)
        /// The text encoder's staging buffers (`MemoryPlan.encoderSlots`; 0 on the lean plan).
        package var encoderSlots: Int = MemoryPlan.encoderSlots
        /// The decoders' smallest bands (`MemoryPlan.fluxMinimumBand`, `qwenMinimumBand`), as the lean
        /// plan takes them; otherwise the standard ones.
        package var minimumBands = false
        /// The update rule (`Solver.history`): the latents its walk keeps between two steps.
        package var solver: Solver = .euler
        /// **The denoiser may take the Neural Engine** (`Request.neuralEngine` against the map's `gemm_ane`):
        /// its surfaces are charged to the phases whose products it takes — the DiT's, Krea 2's fusion —, never
        /// to an encoder's, which never takes it. The map's by default: what a render not yet chosen may do.
        package var neuralEngine = EngineSettings.effective.gemmANE
        /// The images of the batch (`Engine.renderBatch`): every latent waits for the decodings, every
        /// image decoded waits for the batch's end.
        package var batch = 1
        /// The P cores the head split's threads run on (`HeadSplitAttention`: a scratch per thread).
        /// The machine's by default; the tests pass their own, so that no figure depends on the Mac.
        package var threads = EngineSettings.effective.machine.performanceCores
        /// **The process's footprint when the render starts** (`OperatorFootprint.processBase()`): what
        /// the process already holds — the app's interface and history, the command line's runtime —,
        /// read from the process itself. B covers it with the render. `ModelCard.memoryRequest` reads
        /// it; 0 here, for a need of shapes alone (the tests).
        package var base = 0
        /// The library whose installed maps the need reads — the model's own, never another root's.
        package var library: Library = .processWide
        /// The family's maps, given rather than read from the installed ones (the tests); `nil`: the
        /// library's (`ZImageMaps(card:library:)`, `QwenImage21Maps(card:library:)`, `AnimaMaps(card:library:)`).
        package var zImageMaps: ZImageMaps? = nil
        package var qwenMaps: QwenImage21Maps? = nil
        package var animaMaps: AnimaMaps? = nil
        package var krea2Maps: Krea2Maps? = nil
        package var mossMaps: MossMaps? = nil

        package init(card: ModelCard, width: Int, height: Int, references: [(width: Int, height: Int)] = [],
                     promptTokens: Int, loraRanks: [Int] = [], tail: MemoryPlan.Tail = .init(fraction: 0.5, streamed: true)) {
            self.card = card; self.width = width; self.height = height; self.references = references
            self.promptTokens = promptTokens; self.loraRanks = loraRanks; self.tail = tail
        }

        /// **The most economical plan** (`MemoryPlan`'s lean one): the whole map streamed through its two
        /// buffers, no encoder buffer, the decoders' smallest bands — the floor a render can go down to.
        package var lean: Request {
            var r = self
            r.tail = .lean; r.encoderSlots = 0; r.minimumBands = true
            return r
        }

        /// **An upper bound of the encoder's tokens for `prompt`**, without the tokenizer: a BPE token is
        /// at least one UTF-8 byte, so the bytes of the template and the prompt bound the tokens, capped
        /// where the encoder refuses (`Request.admitPrompt`). Z-Image: the chat template around the
        /// prompt; Qwen-Image-2.1: the system turn, the user header, each reference's tags and the
        /// closing turn (`QwenImage21Need`); Anima: the prompt alone (`AnimaTextModule` encodes it without
        /// a template, refused past 512); Krea 2: its 34 template tokens, the prompt capped where
        /// `Krea2Text.admitted` refuses, the 5 of the suffix.
        package static func promptTokens(_ prompt: String, family: Family, references: Int = 0) -> Int {
            promptTokens(text: prompt.utf8.count, family: family, references: references)
        }

        /// The same for `text` tokens of prompt (or bytes, their bound): the template around them added.
        package static func promptTokens(text: Int, family: Family, references: Int = 0) -> Int {
            switch family {
            case .qwenImage21:
                return QwenImage21Need.dropCount + 3 + references * (QwenImage21Need.tokensBefore + QwenImage21Need.tokensAfter)
                    + 5 + min(text, Qwen3VLPrompt.maxPromptTokens)
            case .anima:
                return min(max(1, text), Siliconed.Request.maximumPromptTokens)
            case .krea2:
                return min(Krea2Text.templateTokens + text, Krea2Text.maximumIdentifiers - Krea2Text.suffixTokens)
                    + Krea2Text.suffixTokens
            case .zImage:
                return Tokenizer.chatPrefix.utf8.count + min(text, Siliconed.Request.maximumPromptTokens)
                    + Tokenizer.chatSuffix.utf8.count
            case .mossImage:
                // The template's bytes bound its tokens (`Qwen35Prompt.template`); the prompt is admitted up to the bound.
                return min(Qwen35Prompt.template("").utf8.count + min(text, Siliconed.Request.maximumPromptTokens),
                           Qwen35Prompt.maxTokens)
            }
        }

        /// The prompt the app and the diagnostic assume before a prompt is known: 128 tokens of text.
        /// A figure for a display, never the render's own (the engine passes `promptTokens(_:…)`).
        package static let typicalPromptTokens = 128
    }

    /// **One allocation alive at a phase's peak**: its name (the function it comes from), its bytes,
    /// and whether it is ours (sized by our shape function) or outside our hand (`OperatorFootprint`).
    package struct Term: Sendable, Equatable {
        package var name: String
        package var bytes: Int
        package var outside: Bool
        /// The largest single buffer the term holds, in bytes (0: not a limit's concern) — a VAE's
        /// largest register. MPSGraph and our kernels index it in 32 bits (`MemoryNeed.limit`).
        package var largest: Int
        package init(_ name: String, _ bytes: Int, outside: Bool = false, largest: Int = 0) {
            self.name = name; self.bytes = bytes; self.outside = outside; self.largest = largest
        }
    }

    /// One phase of a render: what is alive at its peak.
    package struct Phase: Sendable, Equatable {
        package enum Kind: String, Sendable, CaseIterable, Codable { case text, vision, references, denoising, decoding }
        package var kind: Kind
        package var terms: [Term]
        package init(kind: Kind, terms: [Term]) { self.kind = kind; self.terms = terms.filter { $0.bytes != 0 } }
        /// Ours: arenas, `Pages`, VAE heaps, host copies — each from its shape function.
        package var own: Int { terms.filter { !$0.outside }.reduce(0) { $0 + $1.bytes } }
        /// Outside our hand: `OperatorFootprint`'s named terms.
        package var outside: Int { terms.filter(\.outside).reduce(0) { $0 + $1.bytes } }
        package var bytes: Int { own + outside }
        /// The bytes of the terms whose name starts with `prefix`.
        package func bytes(_ prefix: String) -> Int { terms.filter { $0.name.hasPrefix(prefix) }.reduce(0) { $0 + $1.bytes } }
    }

    package var phases: [Phase]
    /// What crosses the phases (latents, encoded text, images handed in), held through all of them.
    package var crossing: Int
    /// The process's footprint when the render starts (`Request.base`).
    package var base: Int

    /// **The largest single buffer any phase holds**, in bytes — a VAE's widest register, which at a large
    /// format is the one tensor whose element count nears the limit of the 32-bit indexes (`limit`).
    package var largestBuffer: Int { phases.flatMap(\.terms).map(\.largest).max() ?? 0 }

    /// **The engine's own limit on one buffer: `Int32.max` elements.** The VAEs' tensors are indexed in
    /// 32 bits — MPSGraph's shapes, and our kernels' `uint` offsets (`VAEKernels`) —, and a register past
    /// 2³¹ − 1 floats is not tried: a format whose widest register passes it is refused
    /// (`EngineError.formatRefused`, `.tooLarge`) rather than left to fail in a pass. Conservative: the
    /// kernels alone would hold 2³² (not measured above 2³¹ − 1 on any Mac yet). The Flux decoder's widest register
    /// is 256 channels at the image's resolution, so 8.39 Mpx at most (2048×4096 and 3072² are refused,
    /// 2048×3072 is not).
    package static let limit = Int(Int32.max)

    /// **The largest area, in pixels, this format's tensors let the engine address** — `nil` when its
    /// widest buffer is within `limit`. The widest register grows with the pixels, so the area at which
    /// it reaches the limit is the format's own count of elements per pixel, divided out.
    package func addressableArea(width: Int, height: Int) -> Int? {
        let elements = largestBuffer / MemoryLayout<Float>.size
        guard elements > MemoryNeed.limit else { return nil }
        let pixels = max(1, width * height)
        return MemoryNeed.limit / max(1, (elements + pixels - 1) / pixels)
    }

    /// **The peak**: the largest phase plus what crosses them, plus the base.
    package var peak: Int { (phases.map(\.bytes).max() ?? 0) + crossing + base }

    /// **A phase's own need**: its terms, what crosses, the base — what that phase's measured peak
    /// is compared with.
    package func need(of kind: Phase.Kind) -> Int? {
        phases.filter { $0.kind == kind }.map(\.bytes).max().map { $0 + crossing + base }
    }

    /// **The need of a request**, every family computed from its shapes. `nil` when the maps' headers
    /// cannot be read — **the render is then refused** (`EngineError.memoryNeedUnknown`), never planned
    /// nor accepted on a need of 0.
    package static func of(_ request: Request) -> MemoryNeed? {
        switch request.card.family {
        case .zImage: zImage(request)
        case .qwenImage21: qwenImage21(request)
        case .anima: anima(request)
        case .krea2: krea2(request)
        case .mossImage: mossImage(request)
        }
    }

    // ── the pieces every formula takes, from the constructors' functions ─────────────────────

    /// **One cached attention**, as `Block`, `TextEncoder`, `QwenImage21DiT` and the vision tower build
    /// it: cut by heads (`HeadSplitAttention`: its host arena, ours, and MPSGraph's copy of the GPU
    /// heads' queries) or in one MPSGraph SDPA (its queries' copy, and its causal mask in our pages).
    package static func attention(_ name: String, heads: Int, rows: Int, keys: Int, headDim: Int, causal: Bool,
                                  split: Bool, threads: Int, group: Int = 1) -> [Term] {
        // The GPU's heads: all of them in one MPSGraph SDPA, or what the head split leaves it.
        let gpu = split ? HeadSplitAttention.partition(heads: heads, fraction: HeadSplitAttention.cpuFraction(sequence: rows, heads: heads, headDim: headDim), group: group).gpu.count
            : heads
        // Past `Attention.maxMatrix` the unmasked graph runs in query slices, which MPSGraph concatenates:
        // one more copy of the GPU heads' output (`Attention.init`, `HeadSplitAttention.init`). A causal
        // graph is never sliced (its mask is built whole).
        let sliced = !causal && Attention.groupedSlices(heads: gpu, rows: rows, keys: keys) > 1
        let concatenation = sliced ? [Term("\(name) SDPA concatenation", rows * gpu * headDim * 4, outside: true)] : []
        if split {
            return [Term("\(name) head split", HeadSplitAttention.hostBytes(heads: heads, sequence: rows, keys: keys,
                                                                           headDim: headDim, group: group, threads: threads)),
                    Term("\(name) SDPA queries", OperatorFootprint.sdpaQueries(rows: rows, heads: gpu, headDim: headDim),
                         outside: true)] + concatenation
        }
        return [Term("\(name) SDPA queries", OperatorFootprint.sdpaQueries(rows: rows, heads: heads, headDim: headDim),
                     outside: true)]
            + (causal ? [Term("\(name) causal mask", Attention.maskBytes(sequence: rows, keys: keys))] : []) + concatenation
    }

    /// **One model's `GEMM` over a phase**: its padding tiers (`GEMM.tierBytes`: what a height of 8 to
    /// 32 rows, or one that is not a multiple of 64, sends through them, at the widest such product;
    /// `rank`: a LoRA's, whose `mid` goes through them too), and the conductor's BNNS workspace — the
    /// largest it can grow to over the products the phase runs (`Conductor.workspaceBytes` of the AMX
    /// slice, either weight layout), when the cut applies. The slice is at most `m − cut(m, T)` with the
    /// servo's `T` held in `Conductor.fractionRange`: the bound is its lower end (and the frozen cut's).
    /// MPS's own state is not charged: no shape gives it (`OperatorFootprint`). `bounded`: the products
    /// whose height follows the prompt's tokens — a bound, not the height (`GEMM.tierBytes`). Under
    /// `int8` (`gemm_int8` or `gemm_q8`) the AMX may take a product's last columns instead, all `m` rows
    /// (`Conductor.columnWorkspaceBytes`, larger: 151 against 109 MB at Krea 2's 4096 × 16384 × 6144),
    /// at the same bound on the share — counted for every product, the map's dtype unknown here. Under
    /// `ane` (`Request.neuralEngine`: a denoiser's phase only), the Neural Engine's fp16 operands and its rows
    /// (`NeuralEngine.bytes`).
    package static func gemm(_ name: String, products: [(m: Int, k: Int, n: Int)] = [],
                             bounded: [(m: Int, k: Int, n: Int)] = [], rank: Int = 0,
                             amx: Bool = EngineSettings.effective.routes.products.usesAMX,
                             int8: Bool = EngineSettings.effective.routes.products.usesInt8Weights
                                 || EngineSettings.effective.routes.products.usesQ8Weights,
                             ane: Bool = false) -> [Term] {
        var workspace = 0
        if amx {
            for p in products + bounded where p.m >= Conductor.minimumRows {
                let slices = Set([p.m - Conductor.cut(p.m, Conductor.fractionRange.lowerBound),
                                  p.m - Conductor.cut(p.m, EngineSettings.effective.amxFraction)])
                for rows in slices where rows > 0 {
                    for transposed in [false, true] {
                        workspace = max(workspace, Conductor.workspaceBytes(rows: rows, k: p.k, n: p.n,
                                                                            weightIsTransposed: transposed))
                    }
                }
                if int8, p.n >= 128 {
                    for T in [Conductor.fractionRange.lowerBound, EngineSettings.effective.amxFraction] {
                        workspace = max(workspace, Conductor.columnWorkspaceBytes(
                            m: p.m, k: p.k, n: p.n, width: WeightProducts.amxColumns(n: p.n, share: T)))
                    }
                }
            }
        }
        let tiers = GEMM.tierBytes(products: products, bounded: bounded, rank: rank)
        let neural = ane ? NeuralEngine.bytes(products: products + bounded, gpuShare: EngineSettings.effective.routes.products.ane.gpuShare,
                                                shares: EngineSettings.effective.routes.products.ane.grid.shares) : 0
        return (tiers > 0 ? [Term("\(name) GEMM tiers", tiers)] : [])
            + (workspace > 0 ? [Term("\(name) conductor workspace", Arena.capacity(for: workspace))] : [])
            + (neural > 0 ? [Term("\(name) Neural Engine", neural)] : [])
    }

    /// **A VAE's two heaps and its host copies** (`VAELayout.bytes`), and what its graphs still
    /// allocate beside them: MPSGraph's SDPA, the one pass left whole (`VAELayout.outside`).
    package static func vae(_ name: String, _ layout: VAELayout) -> [Term] {
        [Term("\(name) registers", layout.registers, largest: layout.largestSlot), Term("\(name) workspace", layout.workspace),
         Term("\(name) host", layout.host), Term("\(name) SDPA outside", layout.outside, outside: true)]
    }

    /// **What the rest of a batch holds** (`Engine.renderBatch`: every image denoised, then every image
    /// decoded): the other images' latents, waiting for their decoding — `latent` bytes each, crossing
    /// every phase —, and at the last decoding the images already decoded, `ImageRGB`'s three fp32
    /// planes each, returned together. One image: nothing.
    package static func batchLatents(_ request: Request, latent: Int) -> Int { max(0, request.batch - 1) * latent }
    package static func batchImages(_ request: Request) -> Term {
        Term("earlier images of the batch", max(0, request.batch - 1) * 3 * request.width * request.height * 4)
    }
}

// ── the product's own verification ───────────────────────────────────────────────────────────

/// **How a render's peaks were read** (`FootprintInterval`).
package enum PeakMethod: String, Codable, Sendable, Equatable {
    /// The kernel's `ri_interval_max_phys_footprint`, reset at each boundary: exact.
    case kernelInterval
    /// `phys_footprint` sampled every 10 ms, the kernel's reset missing or failing: a lower bound.
    case sampled10ms
}

/// **What a render needed, and what it took** — computed at its start (`MemoryNeed.of`), read at its
/// end (`ri_interval_max_phys_footprint`, phase by phase, or the 10 ms sampler: `peakMethod`). A phase
/// whose measured peak passes its need is a **defect of the code** (an allocation outside our hand
/// found), logged under the `memory` category and carried by the render's result; never a figure to
/// fold back into the formula.
package struct MemoryCheck: Codable, Sendable, Equatable {
    /// One phase of the render, as `PhaseBoundary` marked it.
    package struct Phase: Codable, Sendable, Equatable {
        /// The boundary's name (`text`, `vision`, `reference 1`, `denoising`, `decoding`…).
        package var name: String
        /// The formula's need for this phase, crossing and base included; `nil` for a boundary the
        /// formula has no phase for (`start`, `denoiser released`: charged to their neighbours).
        package var needBytes: Int?
        /// `phys_footprint`'s maximum over the phase (the kernel's interval maximum).
        package var measuredBytes: Int
        /// Measured minus need: positive is a defect.
        package var gapBytes: Int?
        package init(name: String, needBytes: Int?, measuredBytes: Int) {
            self.name = name; self.needBytes = needBytes; self.measuredBytes = measuredBytes
            gapBytes = needBytes.map { measuredBytes - $0 }
        }
        private enum CodingKeys: String, CodingKey { case name, needBytes, measuredBytes, gapBytes }
        /// Explicit `null`s, as the diagnostic's JSON writes them: "no need for this boundary" is said.
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(name, forKey: .name); try c.encode(needBytes, forKey: .needBytes)
            try c.encode(measuredBytes, forKey: .measuredBytes); try c.encode(gapBytes, forKey: .gapBytes)
        }
    }
    /// The formula's peak (`MemoryNeed.peak`).
    package var needBytes: Int
    /// The render's own peak: the largest phase's.
    package var measuredBytes: Int
    package var phases: [Phase]
    /// Measured minus need, over the whole render.
    package var gapBytes: Int
    /// How the phases' peaks were read (`FootprintInterval`): exact, or sampled every 10 ms.
    package var peakMethod: PeakMethod
    /// **The process's footprint when the render started** (`MemoryNeed.Request.base`), inside every
    /// need above: the peaks are the process's, the app's interface and history or the command line's
    /// runtime included, and so are the needs. `nil` in a report written before it was carried.
    package var baseBytes: Int?
    package init(needBytes: Int, measuredBytes: Int, phases: [Phase],
                peakMethod: PeakMethod = .kernelInterval, baseBytes: Int? = nil) {
        self.needBytes = needBytes; self.measuredBytes = measuredBytes; self.phases = phases
        self.peakMethod = peakMethod; self.baseBytes = baseBytes
        gapBytes = measuredBytes - needBytes
    }
    /// **The phases above their need** — each one a defect.
    package var exceeded: [Phase] {
        phases.filter { ($0.gapBytes ?? 0) > 0 } + (gapBytes > 0 && !phases.contains { ($0.gapBytes ?? 0) > 0 }
            ? [Phase(name: "render", needBytes: needBytes, measuredBytes: measuredBytes)] : [])
    }

    /// The engine's memory log: `log stream --predicate 'subsystem == "dev.gwenn-ha.siliconed" && category == "memory"'`.
    package static let log = Logger(subsystem: "dev.gwenn-ha.siliconed", category: "memory")

    /// The `PhaseBoundary` names, mapped to the formula's phases.
    package static func kind(of boundary: String) -> MemoryNeed.Phase.Kind? {
        switch boundary {
        case "vision": .vision
        case "text": .text
        case "image": .references
        case "denoising", "denoiser released": .denoising
        case "decoding": .decoding
        default: boundary.hasPrefix("reference") ? .references : nil
        }
    }

    /// **The check of one render**: each marked phase against its own need.
    package init(need: MemoryNeed, marks: [(name: String, peak: Int)],
                 peakMethod: PeakMethod = FootprintInterval.method) {
        let phases = marks.map { mark in
            Phase(name: mark.name, needBytes: MemoryCheck.kind(of: mark.name).flatMap { need.need(of: $0) },
                  measuredBytes: mark.peak)
        }
        self.init(needBytes: need.peak, measuredBytes: marks.map(\.peak).max() ?? 0, phases: phases,
                  peakMethod: peakMethod, baseBytes: need.base)
    }

    /// One line per defect, for the log and the CLI.
    package var defects: [String] {
        exceeded.map {
            String(format: "memory defect: %@ peaked at %.0f MB, its need is %.0f MB (+%.0f MB)", $0.name as NSString,
                   Double($0.measuredBytes) / 1e6, Double($0.needBytes ?? needBytes) / 1e6,
                   Double($0.gapBytes ?? gapBytes) / 1e6)
        }
    }

    /// **The phases below their need**, one line each — reported, never a failure: a check that only
    /// looks up is blind to an over-charge, and an over-charge can hide an under-charge in the same
    /// phase (two terms compensating). The slack is what the formula charges that the render did not take.
    package var slack: [String] {
        phases.compactMap { phase -> String? in
            guard let need = phase.needBytes, let gap = phase.gapBytes, gap < 0 else { return nil }
            return String(format: "memory slack: %@ peaked at %.0f MB, its need is %.0f MB (%.0f MB)", phase.name as NSString,
                          Double(phase.measuredBytes) / 1e6, Double(need) / 1e6, Double(gap) / 1e6)
        }
    }
}

// ── the shape functions' memory ──────────────────────────────────────────────────────────────

/// **A shape function's results, kept by its inputs.** The need is asked at every preflight and by the
/// app for each format it greys out, and a VAE's plan walks every pass and places every slot (a Qwen
/// decoder 16 ms, three references' sequence 12 ms, in a debug build): the same inputs give the same
/// bytes, so each is walked once per process. A pure function only — its key holds every input it
/// reads; the value is what the function returned, never a copy rebuilt. Bounded: past `capacity`
/// keys, everything is dropped and walked again (a format grid, not a growing log).
package final class ShapeMemo<Key: Hashable & Sendable, Value: Sendable>: Sendable {
    private let table = Mutex<[Key: Value]>([:])
    private let capacity: Int

    package init(capacity: Int = 64) { self.capacity = capacity }

    /// The value of `key`, computed by `compute` the first time (outside the lock: two threads may both
    /// walk it, and keep the same bytes).
    package func callAsFunction(_ key: Key, _ compute: () -> Value) -> Value {
        if let value = table.withLock({ $0[key] }) { return value }
        let value = compute()
        table.withLock { table in
            if table.count >= capacity { table.removeAll(keepingCapacity: true) }
            table[key] = value
        }
        return value
    }
}

/// The key of a VAE's plan: its config and the integers of its shape, in the function's order.
package struct VAEPlanKey<Config: Hashable & Sendable>: Hashable, Sendable {
    package var config: Config
    package var shape: [Int]
    package init(_ config: Config, _ shape: [Int]) { self.config = config; self.shape = shape }
}
