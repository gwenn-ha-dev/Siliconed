English · [Français](ENGINE.fr.md)

# The Siliconed engine

What Siliconed computes on, how it stays within its memory budget and how it proves its results. For what the app does, see [The Siliconed app](APP.md).

## Core AI

**One runtime: Core AI, Apple's own, which arrived with macOS 27.** The engine, `SiliCore` (`Engine/`), writes Core AI functions of fixed shapes op by op, in Swift — products, attentions, norms, convolutions — and calls them in a loop that is the model's own, also in Swift. A function holds no weight: weights are its arguments, views of an arena that the engine reads them into. Core AI compiles each function once per Mac and runs it on the GPU or on the Neural Engine; what the chip prefers is measured on the chip, never written in a table.

**The frontier.** The engine knows tensors, weights by groups, functions, bytes and seconds; each model (a *family*, in `App/`) declares what it needs in those words and keeps its own. The engine holds no word of the image domain, and a test fails if it learns one.

**The arithmetic is the family's**, written in its functions: int8 codes multiplied int8 × int8 on the GPU (W8A8) and dequantized inside the function on the Neural Engine (W8A16, on a copy of the codes laid as it reads them), bf16 weights rounded once to f16 as they are read where the Standard shares its rows with the Neural Engine, f16 kept in range by explicit factors, f32 where a family asks for it. The starting noise is `torch.randn`'s, bit for bit.

**Compilation.** Core AI compiles a function for its shapes, and keeps it. The first render of a model at a new size, in a version (Standard or Compact) not yet run, or with a LoRA of a new rank compiles first and pays for it once (its time in brackets under [Numbers](#numbers)). Every render after it, at that size, starts at once.

**The bits belong to the system.** The computation is Core AI's: a macOS update can move a rounding. The same seed gives the same picture to the eye, not necessarily the same file byte for byte from one macOS to the next. What Siliconed guarantees is the deviation from the reference, stage by stage, and you can measure it again at home ([VERIFY.md](VERIFY.md)).

## The interface

A family uses a handful of types, all public, each with its tests:

| | |
|---|---|
| `Aicode` | the op-by-op writer of a function's bytecode, and the forms both the GPU and the Neural Engine take (`Aicode.Block`: the products, the norm, the rotation) |
| `SharedWeights` | what those forms' products read: the Compact's int8 codes, or the Standard's weights in f16 where it shares its rows with the Neural Engine |
| `Functions` | a function's asset, written when its bytes change, and loaded for the GPU or the Neural Engine |
| `Arena` | one Metal heap per stage, every tensor on a page; a buffer apart for a stage's exit; surfaces for the Neural Engine |
| `Weights`, `Recipe`, `SafeTensors` | the published files (one, or the shards of an index) read as they are, each tensor converted once as it enters the arena; a product's parts joined |
| `Stage` | a stage declared, planned under the budget, then run: its arena, its weights kept or read ahead, its functions |
| `ThisMac` | what this Mac measured, kept by Mac and by system |
| `Block` | a transformer's loop as its family declares it, run by the engine: its calls put ahead of the GPU, its rows shared with the Neural Engine |
| `Bands`, `BandStage` | a network of layers over planes (a decoder, an encoder), each layer one function called band after band down its plane's rows, two calls in flight, the bands sized and the planes laid out by the engine |
| `Layers` | a stack of layers (a text encoder, the Prompt Lab's writer), over the whole prompt at once or one token at a time through every layer |
| `Memory`, `FootprintWatch` | the process's footprint, read as the system counts it; the fuse |

**A stage** is declared by its family: its tensors (in the heap, apart, or as surfaces), the weights it holds all along, the weights it can keep or read in turn by groups, the order its loop uses the groups in, the functions it loads. A stage may come in several forms, the tallest calls or bands first (`Block`'s heights, `Bands`' positions): each is planned, and the engine takes the first whose P and what Core AI will hold under its functions on this Mac (read before, or bounded from what was read: below) hold under the budget (`Stage.choose`).

## Memory and the budget

Siliconed never grants itself more than its budget: ¾ of the Mac's memory minus 2 GB by default, 4 GB on an 8 GB Mac, 10 GB on 16 GB, 46 GB on 64 GB, adjustable in the app. What other applications hold is theirs.

<p align="center"><img src="images/budget.svg" width="900" alt="A stage under the budget: the floor, the arena, the ring fed from the published files, the weights kept, Core AI's share read in flight, all under B; a block's rows shared between the Neural Engine and the GPU, the GPU alone for what reads across rows."></p>

**Every stage is planned before anything is allocated**, to the byte: P = the process's footprint before it + the arena as Metal will allocate it (its tensors, its resident weights, the slots of a ring). The weights a loop repeats are read from disk into that ring, on a thread of their own, while the loop computes on the others; what of them the stage keeps in memory is decided during the first passes, once Core AI's share is read. The footprint is read once the pages `malloc` has freed are handed back to the kernel (`Memory.floor`): macOS bills a freed page until the kernel reclaims it, and a floor that counted them would grow from one render to the next. What no plan can fit is refused up front, with the number it is short by. **A render's decoder is judged before its DiT starts** (`BandStage.admit`), over the floor the DiT will give back: a decoding that cannot fit is refused before the DiT's minutes, not after them.

**What Core AI holds for its functions cannot be counted on paper**, so it is read while the stage runs: the footprint past the floor and the arena, sampled every millisecond (a call's working memory comes and goes within the call). From the loop's signal, the reader keeps a group of weights in a heap of its own while the footprint at that share's peak, the group, what is still to come of the share (what it grew past the signal last time) and a margin hold under the budget, and while the system lets the process wire it; otherwise the group rides the ring. Every weight placed is wired (`mlock`) until its stage closes, wherever the system lets the process wire it (past its limit, a weight stays unwired): unwired, a weight the loop reads once a step is a page left alone in between, which macOS compresses under pressure, and every call that reads it then pays the decompression (Z-Image Compact at 1024² on a 16 GB M1 Pro, a browser open: 83 to 86 s with 6 GiB kept, 79 s wired). Each set of functions' largest share is kept on the Mac, under a key that carries its functions, their unit and how many calls of one function are in flight (`Stage.key`), and only if they were called: it chooses the next time how tall a stage's calls or bands are, and a share of the Neural Engine moves only where its functions hold.

**A form never read is bounded from its kin** — the same functions at another size, height or share of the Neural Engine, whatever build of the engine wrote them, as long as they write the same ops (`Stage.Extent`, `Block.extent`): the least of their readings times how much more its calls hold, never below the reading itself, with a margin for what ran before in the process (`Stage.estimates(extents:)`). Bounded from a form that holds less (a size larger than any this Mac rendered), the bound is a prediction, about 10 % over what the calls hold: past the budget, the form is refused before anything runs. Bounded only by a form that holds more (a budget lowered under every form read), the bound says nothing of how much less: the form goes as on a Mac with no reading. With nothing read of its kin, a stage takes its soberest form, on paper.

**The fuse.** A watch reads the footprint every millisecond and stops the process the moment it passes B (`FootprintWatch`). It is a last guard, which a plan written right never sounds: B is never passed, and a render the fuse stopped is a fault whose cause is traced, not a way for the engine to learn. The functions opened last when it sounded are marked on the Mac, and never taken again by the engine that took the mark (its binary) under a budget no larger than the one it ran under: a new version, or a larger budget, tries them once more. A form taller than one marked holds more, and is never taken. If the fuse still stops a decoding, the latents its DiT left are kept: the render is handed once to a new worker, which decodes them alone, without running the DiT again.

**What the engine learned of Core AI shapes the arena**: a tensor off a page is copied at every call, a transposed view as well, and a function holds its last call's buffers until it is called again or let go. So every tensor starts on a page, a product's weights are written once in the layout Core AI's product reads as it lies (the Compact's codes `[pieces, out, in / pieces]` for the GPU's integer product, bf16 weights as the file has them, or `[in, out]` where a product is written x · W), a stage's exit lies apart from its heap, and a stage let go lets its arena go before the functions that read it, then waits until the footprint has come down before the next one starts. A block waits for a call before calling its function again: a second call in flight takes a second working memory. A stage of bands keeps two calls on the stream, each encoded while the one before runs (one at a time, the GPU stood idle while the CPU encoded each); their second set of intermediates is part of Core AI's share, and the key a reading is kept under says so. Core AI's stream is touched by one thread at a time: two waits for it at once can lower its count of sync points, and the next wait stops the process (the 0.32 at the second render of Krea 2 and Qwen-Image 2.1 on a 16 GB M1 Pro), so the calls are put on it under a lock and its waits run on a queue of their own that holds it (`SerialStream`). Each of these facts is a test run on the Mac's GPU (`RunTests`).

**The Neural Engine beside the GPU.** A stage can give part of its work to the Neural Engine, its weights then on surfaces both units read: the first rows of a block, in groups, for the steps that compute each row alone, while the GPU takes the rows past them and the steps that read across rows (an attention) on every row; the engine orders the two units' calls, each waiting only for the rows it reads. The share is followed in flight, at every step: each step's calls are timed and smoothed (a half-life of two steps, so that heat and battery are followed), and at the next step the share moves to the one predicted to finish first, phase by phase as the calls are ordered, among the shares known to hold under the budget, where what it gains over the steps left (the render loop tells them) pays for the move and passes the noise between two steps. The Neural Engine's rate against the GPU's, FLOP for FLOP, is kept by arithmetic: a block never run on the Mac starts where both units finish together, a block run before where the last render left it. The share of none (the GPU alone, in the same arithmetic) is measured once per block in its first steps, and kept where it wins. The rows a Neural Engine call takes (its unit) are decided once per arithmetic on the Mac, tried at its first run, and never change after.

**One arithmetic for each switch position.** With the Neural Engine on, a DiT computes in f16 end to end, its functions shared by the GPU and the Neural Engine: the Compact on its int8 codes, the Standard on its bf16 weights rounded once to f16. With it off, each computes in its own format on the GPU alone (the Standard in bf16). No other way is tried. The Neural Engine's share runs from none to all but one unit, and none is a share like any other: the first render of a block on a Mac also runs two steps on the GPU alone, with the same functions, because the GPU computes a row slower beside the Neural Engine than alone and the calls beside it do not predict that. The share of none is kept where it wins (`Block.zero`, `ThisMac.flight`). If B refuses it, the step says so (`Stages.Cycle.refused`) and the next render measures it again. The price is known: on a GPU that computes bf16 as fast as f16 (the Mac mini M5 Pro at 1024²), the Standard runs a step 1 % slower than in bf16 on the GPU alone on Krea 2, 12 % slower on Anima.

**The Compact on one path.** Its codes lie as the file lays their rows, cut along their inputs into equal pieces of at most 4096, multiples of 64 (`Weights.pieces`). The GPU multiplies them int8 × int8 (W8A8, `w8a8`): each piece of the input brought to int8 under its row's scale, the sums in int32 by Core AI's integer kernel, the scales applied after, in an order that keeps every sum within f16 — the recipe of the Compact's int8 reference, activations quantized per row. The Neural Engine reads its codes only as `[1, in, out]`, its own int8 weights dequantized in its function (W8A16, `q64`): the CPU writes each code into one copy of that layout, at the GPU's step before the Neural Engine's steps that read it, every Neural Engine call before it done, so that the copy never runs beside the Neural Engine nor changes its surfaces; that copy is a surface in P, only where the Neural Engine takes rows.

**Kept by Mac and by system.** What a Mac measured — the shares Core AI was read at and what their calls held, the fuse's marks with the engine and the budget they were taken under, the Neural Engine's share of each block and its share of none, its unit and its rate against the GPU's by arithmetic, each block's rates in TFLOP/s — is kept under the Mac's model and the system's build, and measured again after a macOS update. The app keeps it in its own space on the Mac (`engine/this-mac.json`, which macOS never purges and which stays on the Mac when the library moves); the functions' assets, which can always be written again, live in the user's caches.

## Numbers

The five models at 1024², 8 steps, seed 42, the app's default settings, under the default budget. The time is the render's, the model loaded; the first render at a size also compiles (above), its time in brackets where it was taken.

| | Mac mini M5 Pro, 64 GB · Standard | Mac mini M5 Pro, 64 GB · Compact | M1 Pro, 16 GB · Compact, 0.30 |
|---|---:|---:|---:|
| Z-Image Turbo | 21.7 s | 16.2 s | 75.9 s (93.5) |
| Anima Turbo | 7.7 s | 8.2 s | 33.6 s (45.8) |
| Krea 2 Turbo | 41.1 s | 27.8 s | 128.1 s (149.6) |
| MOSS-Image | 21.7 s | 18.1 s | 76.4 s (99.3) |
| Qwen-Image-2.1 Turbo | 29.4 s | 21.0 s | 97.7 s (120.7) |

The M1 Pro's column is still 0.30's, until it is measured again.

**The Compact reads half the bytes of the Standard**, from disk and in memory, and where the GPU multiplies int8 into int32 it computes faster as well: on the Mac mini, a step of its DiT at 1024² is 10 to 40 % faster than in 0.30, its int8 × int8 against 0.30's W8A16. Where the budget cannot hold the Standard's weights, it is the version that reads the least.

Other Macs, the 8 GB ones first, await reports from the machines themselves; `silicontrol measure` takes your own ([REMOTE-CONTROL.md](REMOTE-CONTROL.md#what-gets-measured)), and [this form](https://github.com/gwenn-ha-dev/Siliconed/issues/new?template=times.yml) brings them here.

## Next to other tools

- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [diffusers](https://github.com/huggingface/diffusers) in Python**: the same models, the same files, with no Python environment and no node graph to wire. Siliconed implements them natively and checks itself against diffusers and [transformers](https://github.com/huggingface/transformers); `silicontrol` covers scripts and automation.

## Verification

Every stage (tokenizer, text encoder, denoiser, VAE) is compared with the reference implementation, diffusers and transformers, on the same input. The deviation is measured channel by channel, each channel relative to its own norm, so a fault confined to one channel cannot hide in a global figure.

- **The Standard is judged against the reference run in bf16**, the arithmetic of its weights. Where an fp64 computation of the same stage exists, it is the witness: the engine must be no further from it than the bf16 reference, plus 25 %.
- **The Compact is judged at the denoiser, against ComfyUI's int8 products** on the same weights. Its render is checked as a finite image, its deviation from the fp32 image shown and not judged: eight free steps in 8 bits fork, and judging the fork would say nothing about the computation.
- **What is exact stays exact**: the tokenizer, the seed's noise, the σ schedule, bit for bit.

**These models live at the edge of bf16, and the checks show it.** Two faithful bf16 computations deviate from each other as much as each does from the fp32 reference, and a step's worst channel varies from 0.7 to 2 times from one draw to the next. So a few measurements pass their bar by a few percent, and we do not move the bar to let them through: they are listed, with their numbers, in [VERIFY.md](VERIFY.md). What would stop a release: a global error clearly above the bf16 reference's, an exact stage that no longer is, a wrong image.

The comparison ships: the oracle scripts in `App/tools/oracle/` and the `siliconed check` command redo it on your Mac, step by step, with the fingerprints of our references to compare yours with; [VERIFY.md](VERIFY.md) says how, and what it costs. Memory is held to the same discipline: each check also says whether the session stayed within what its plan announced. The Prompt Lab's writer is checked likewise: its logits against transformers' on the same tokens, then its decoding (`siliconed check qwen-image-2.1 writer`).

## Building from source

macOS 27 or later, Apple silicon, Xcode 27 or later. The Command Line Tools alone are not enough: their SDK lacks SwiftUI's macros, and the app's phrases are compiled by `xcstringstool`, which only Xcode ships (`tools/app.sh` says so and stops).

The repository holds two packages: `Engine/`, the engine, and `App/`, the app, which depends on it.

```
cd Engine && swift build && swift test     the engine and its pure tests
cd App && swift build -c release           the app, silicontrol and siliconed
cd App && swift test                       the app's pure tests, with no weights at all
cd App && tools/app.sh                     builds .build/Siliconed.app and opens it
cd App && tools/app.sh --no-open           builds only
cd App && tools/dmg.sh                     packages the built app as .build/Siliconed-<version>.dmg
```

`swift test` needs nothing else: no weights, no Python. The engine's tests that go through this Mac's GPU are apart: `cd Engine && SILICORE_METAL=1 swift test -c release --filter RunTests`.

By default the app is signed ad hoc: it runs on the Mac that built it. To distribute it, sign it with your Developer ID:

```
cd App && tools/app.sh --sign "Developer ID Application: <name> (<team>)"
```

If `SILICONED_NOTARY_PROFILE` names a `notarytool` keychain profile, the app is also notarized and stapled; `tools/dmg.sh` then signs and notarizes the disk image with the same identity.
