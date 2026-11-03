English · [Français](ENGINE.fr.md)

# The Siliconed engine

How Siliconed measures your Mac, picks its kernels, stays within its memory budget and proves its results, with the numbers. For what the app does, see [The Siliconed app](APP.md).

## An engine measured on your Mac

**No two Macs compute alike, so Siliconed neither guesses nor looks a Mac up in a table.** What it knows about yours it reads or measures, and every decision of a render is worked out from those numbers. Four steps:

1. **Read.** In milliseconds, what the silicon announces: the GPU's cores and clock, the CPU's clusters, the Neural Engine's generation, the memory. These facts decide nothing: they say which engines exist, and they are the denominator of “reached / peak”. No chip name appears anywhere in the engine, so a Mac that does not exist yet is served by the same code.
2. **Measure.** Before the first render after each install or update — and whenever the Mac or macOS changes —, in about a minute (five at most) and without any model, only what the system does not say: each engine alone — the GPU, the CPU's matrix unit (AMX), the Neural Engine — on matrix products of the models' shapes (a fixed cost, a rate, and for the Neural Engine what lifting the weight costs and for the AMX how the rate depends on the depth of the product; the AMX and the Neural Engine in chains of back-to-back calls, because the second call of a pair is the cluster still warming up), the Neural Engine's steps by rows, attention through its kernels (for those on half-precision operands, in the form this Mac runs fastest: three stages, or fused at one of the same tiles on every Mac), products read straight from published 8-bit weights, bf16 products on the GPU's tensor units, each at the tile this Mac runs fastest among the same candidates on every Mac (for the large products and for the thin ones); then every set of engines together — chains of the set at the rows the model balances, each side timed by its own clock, because engines share power and memory and their rates do not add up: what each keeps of its rate is read as an overlap, the side that finishes first having run beside the other all along —, the heat, the memory bandwidth and the cost of preparing a weight. Every result is checked against the exact one, computed in double precision on the CPU, and a route that computed wrong is excluded. The probes' sizes follow the machine, so that the slowest Mac holds the minute too, and their memory is counted before the first product, under the budget. 

   **The measure also knows when the Mac is not free.** After each step it asks whether as many threads as there are performance cores still get a core each, whether an empty submission still reaches the GPU promptly, and whether identical calls take the same time; a step that fails any of the three is measured again after a wait, while the minute leaves seconds, and what stays busy is kept, never refused: a disturbance only adds time, so a chain whose identical calls disagree is read on its quick side, how far they disagreed is written beside each engine and set, and the Mac does not have to be at rest (a moving mouse, a browser, Activity Monitor are the Mac's life) — a load is not what is read (a core of background work moves a figure by a few percent, a program using the GPU halves it), the effect is. 

   **A finished measure is always the map**, applied as it stands; what it could not give is a route or a set the plan does not offer, what looked aberrant is bounded or left out and said, and nothing asks to measure again.

   The measure runs in a process of its own, because a program the Neural Engine refuses can kill the process that loads it: when that happens the route is left out, the measure starts again without it, and the report says the route died — no render loads a route the measure has not seen load. That process declares itself a foreground application, as a render in front of its user is: left to the role its launcher gave it, an app opened through LaunchServices read the pair of the GPU and the Neural Engine 20–50 % noisier than the same executable launched from a terminal.
3. **Model.** The probes are fitted into **this Mac's model**: a few numbers per engine — for each kernel a fixed cost and a rate, the Neural Engine's step, what each engine keeps of its rate beside the others and what the set costs to use, what heat takes off over a long render, the memory bandwidth, what preparing a weight costs. The report keeps the model and the raw measurements it was fitted on, so that anyone can replay the fit.
4. **Solve, then loop.** Before each matrix product and each attention of a render, a solver takes that block — its shape, the form of its weight, its phase —, times it with the model on every set of engines, and keeps the set and the whole rows each engine takes that finish soonest, under the memory budget and the precision rules (the Standard never computes below its published weights: they are read as published, in bf16, and where this Mac's measure finds it faster the activations and the attention's operands are rounded to bf16 between operations, every sum in fp32; the Neural Engine, which computes in fp16, joins only where you leave it on, below). That answer is the **plan**: exact rows per engine and the path of the weight, solved once per kind of block and kept. Each render then compares the time the plan predicted with the time it measured, class by class, and reports the gap; it never changes the plan or the image — a model that has drifted is a reason to measure again, not a factor to apply. When a render ran more than 10 % off its prediction, the app says so under the Generate button and offers to measure again, once for a given measure; the new one applies to the next render, never to one under way: a render reads one plan from its first product to its last, and two renders of one seed under one measure are one image.

The measure buys speed and nothing else: it never changes the image beyond what the arithmetic it checked allows. The same seed gives the same pixels from one render to the next on one Mac with one measure; another measure, or another Mac, may plan other rows, and other rows sum in another order ([VERIFY](VERIFY.md#timings-and-peak-memory)). The starting noise is `torch.randn`'s, to the bit, on all five models: a seed gives diffusers' noise ([VERIFY](VERIFY.md#timings-and-peak-memory)). The Neural Engine's fp16 is your choice, per model (below). Help › **Measure This Mac…** shows the map — how sure each engine's figure is, and what the measure found, in plain sentences —, what each engine reaches against the peak the silicon announces, the memory the measure held and how long it took, and lets you share the report ([Community diagnostic](#community-diagnostic)).

### The Neural Engine

The Mac's Neural Engine computes in fp16. Where this Mac's plan gives it rows, it takes part of the DiT's matrix products beside the GPU, block by block: each weight column and each activation row is lifted by a power of two into fp16's range and brought back in fp32, and rows that overflow go back to the GPU. On int8 weights (the Compact of Krea 2, Qwen-Image-2.1 and MOSS-Image) it reads the weights as published, in 8 bits, and widens them itself; each column's scale is applied afterwards, in fp32. On the M1 Pro those three Compacts take 24 to 37 % less time than their Standard ([Numbers](#numbers)). The text encoders and the VAE never use it.

**The GPU's tensor units.** Where this Mac's measure finds it faster — an M5's GPU computes bf16 several times faster than fp32, an M1 Pro's does not —, the DiT rounds an activation to bf16 before a product on a bf16 weight and runs its attention on bf16 or fp16 operands, the sums in fp32: the precision of a GPU computing in bf16, checked against diffusers run in bf16 (the same rule as the Neural Engine's: its own error + 25 %). The text encoders and the VAE never do.

**One switch, Accelerators**, in the DiT's box of the rack, covers both, per model, and the choice is kept: **on by default on every version**, the Standard included (a Neural Engine turned off before 0.19 is Accelerators off). Off, the DiT is in fp32 throughout — no product in fp16 on the Neural Engine, no operand rounded on the GPU — and checked against the reference in fp32; weights published in bf16 or 8 bits are still read as published, beside fp32 activations, a product that rounds nothing more than fp32's. The label beside it says what this Mac does with them: **computes fp32 · ANE**, **computes fp32 · GPU bf16** (`bf16, fp16` where the attention may take fp16 too), both, or a plain **computes fp32**; an image computed with the Neural Engine carries `neural-engine` in its metadata, one computed with the switch off `accelerators = off`. On a Mac whose measure offers neither (no Neural Engine found or its results were wrong, and a GPU on which half precision is no faster), the switch is greyed, with that reason: there is nothing to choose.

These energy and time figures were taken on 23/10/2026 with app 0.14, before 0.18's capacity model, and have not been retaken since (0.18's times are under [Numbers](#numbers)). The M1 Pro, 1024², Standard, Faster Rendering on, seed 42, the library prompt (“a 30 year old woman posing in a library”):

| | Krea 2 Turbo | Z-Image Turbo | Qwen-Image-2.1 Turbo |
|---|---|---|---|
| time in the app, off / on | 144.6–144.9 s / 135.4–137.4 s | 86.8–93.9 s / 83.4–84.0 s | 103.6 s / 99.9–100.2 s |
| power draw of the whole machine, off / on | 43.9 W / 35.3 W | 38.0 W / 31.2 W | 45.4 W / 36.9 W |
| energy per image, on against off | −26 % | −16 % | −20 % |
| the image, on against off | PSNR 35.9 dB: the same scene, face and light, the finest details shifted (book spines) | 55.9 dB, a tenth of a level on average | 56.3 dB, a tenth of a level on average |
| deviation of the denoiser's output from the exact fp64 result, with half of every product's rows on the Neural Engine | 4.8·10⁻³ (512²) | 3.6·10⁻³ (1024²) | 4.4·10⁻³ (512²) |
| …as a share of the publisher's bf16 deviation | 0.14 | 0.06 | 0.31 |

Times: renders queued in the app with `silicontrol`, each in a newly launched app, 0 swap-out; off, three Z-Image renders in a row took 93.9, 89.0 and 86.8 s with the same bits, the first after the library had been moved. These are 0.14's times and the library prompt; 0.18's are under [Numbers](#numbers). Power and energy: the same comparison through the command line, four renders per model alternating on and off, the whole machine's power draw (display included) read from the power adapter's telemetry. The deviation is that of one evaluation of the denoiser with more of the Neural Engine than any plan gives (half of every product's rows), as printed by `siliconed check <model> dit --neural-engine` ([VERIFY.md](VERIFY.md)), 24/10/2026; the bf16 one is diffusers' deviation with the publisher's bf16 weights, which is how most tools compute.

## Numbers

Reference machine: MacBook Pro M1 Pro (8 cores, 14 GPU cores), 16 GB. App 0.18, 1/11/2026, on the capacity map the app measured, seed 42, 8 steps, the prompt “a 30 year old woman posing in a library”. The first two lines compute every evaluation at full size: the publisher's recipe, the image checked against the reference. Faster Rendering ([below](#faster-rendering)) computes the first ones at half size and gives another image.

| | Krea 2 Turbo | Z-Image Turbo | Qwen-Image-2.1 Turbo | Anima Turbo | MOSS-Image |
|---|---|---|---|---|---|
| 1024², Compact (8 bits, the default version), Neural Engine on | **140.9 s** | **101.9 s** | **85.8 s** | **40.1 s** | **85.0 s** |
| 1024², Standard (the publisher's bf16, computed in fp32) | **213.3 s** | **113.1 s** | **135.6 s** | **40.9 s** | **111.7 s** |
| 1024², Compact, with Faster Rendering (measured with it on; a Beta option, off by default) | 97.0 s | 80.8 s | 71.4 s | 32.4 s | 78.9 s |
| peak memory of the app, 1024², Compact / Standard | 1.7 / 2.1 GiB | 1.9 / 1.9 GiB | 1.5 / 1.5 GiB | 1.3 / 1.3 GiB | 2.2 / 2.4 GiB |
| read from disk per image, 1024², Compact / Standard | 98 / 195 GB | 0.1 / 44 GB | 0 / 121 GB | 0 / 0 GB | 0.1 / 76 GB |
| deviation of the denoiser's output (`model_out`, worst channel) from the fp32 reference · from the fp64 exact (the reference's own) | 2.6·10⁻⁴ · 3.3·10⁻⁴ (3.1·10⁻⁴), 1024² | 9.8·10⁻⁶ · 1.2·10⁻⁵ (1.3·10⁻⁵), 1024² | 3.1·10⁻⁶ · 2.6·10⁻⁶ (2.5·10⁻⁶), 512² | 1.4·10⁻⁵ · per latent plane 2.9·10⁻⁶ (3.8·10⁻⁶), 1024² | per latent plane 1.2·10⁻⁶ · 1.3·10⁻⁶ (1.4·10⁻⁶), 512² |

**The Compact is faster; the Standard is the image checked against the reference.** The Standard reads the publisher's bf16 weights and, on the M1 Pro of these tables, computes them in fp32, the precision the reference is checked in (on a GPU that computes bf16 faster, part of the DiT in bf16 with the accelerators on: [The Neural Engine](#the-neural-engine)); the Compact reads 8-bit weights published by third parties and computes in fp32 too, and the capacity map sends part of its products to the Neural Engine where that is faster ([The Neural Engine](#the-neural-engine)). On Krea 2, Qwen-Image-2.1 and MOSS-Image, whose Compact weights are int8, the Compact takes 24 to 37 % less time than the Standard. On Z-Image and Anima, whose Compact weights are Q8_0, it takes 10 % and 2 % less. At full precision the weights of Krea 2, Qwen-Image-2.1 and MOSS-Image do not stay in a 16 GB Mac's memory and are read again from disk at each step, 195, 121 and 76 GB per image at 1024² (Krea 2's Compact too, 98 GB); the reading runs under the computation. That is also why every peak above stays under 3 GiB: weights mapped from disk are not counted in `phys_footprint`, since the system can drop them at any moment. What the Standard costs is disk space, twice the Compact's, energy and time, from 2 % on Anima to 37 % on Qwen-Image-2.1.

Each line is a `silicontrol measure` series in a freshly opened app, the series anyone can repeat ([REMOTE-CONTROL.md](REMOTE-CONTROL.md)): a load render, which encodes the prompt, then a witness, the render and a witness again. The prompt being the witness's, these three are the same image, identical to the bit, and the time given is their median, from the request to the PNG, the prompt already encoded; within a series they stayed within 3.9 % of each other, 0 swap-out. The capacity map was measured in the app before each version's series, in 45.6 s before the Standard's and 41.0 s before the Compact's. Peak memory is the app's `phys_footprint`, sampled every 100 ms. The deviations are what `siliconed check <model> dit --size <side>` printed on the same Mac on 24/10/2026, against goldens written by `tools/oracle/`; on yours they differ in the last digits ([VERIFY.md](VERIFY.md), [Verification](#verification)).

### On a MacBook Neo

An 8 GB Mac: a MacBook Neo (A18 Pro), on AC power. App 0.18 as published, 1/11/2026, the series above run on the Neo itself, on the capacity map the app had just measured (45.2 s), with the version installed there, the Compact, Neural Engine on.

| 1024², Compact | Z-Image Turbo | Anima Turbo |
|---|---|---|
| every evaluation at full size | **246.5 s** | **91.5 s** |
| with Faster Rendering | 188.7 s | 67.7 s |
| peak memory of the app | 2.0 GiB | 1.4 GiB |
| read from disk per image | 44 GB | 0 GB |

The budget of an 8 GB Mac is 4 GB (¾ of 8 GB minus 2 GB, [Memory and the budget](#memory-and-the-budget)): each peak stays at half of it at most, and no series swapped out a page. Z-Image's Compact weights do not stay in the Neo's memory and are read again from disk at each step, 44 GB per image. That is 2.1 to 2.4 times the reference machine's time on the same lines. Each series gave a single image, identical to the bit from render to render. Z-Image at full size is the mean of its series' first two renders, 247.0 and 246.1 s: the third, the closing witness, took 312.2 s, 26 % more than the first, beyond the 13 % within which `silicontrol measure` holds a series comparable; the other three series kept their renders within 7 % of each other.

### Faster Rendering

A Beta option (Settings › Advanced › Beta), off by default: it adds a switch to the DiT's box, next to Accelerators, and with the option off no render is a faster one, whatever was chosen. Once on, a choice per model, off for every version (the full-size recipe, the image checked against the reference), for all five models; the choices are kept when the option is turned off and come back with it. The switch is greyed, with its reason, when it would do nothing for the render being set up. The choice is saved with the image, so Render Again gives the same one. From 1024 px on each side (1024², 1024×1536, 1536×1024…), the first evaluations run on the half-size grid — 3 of 8 for Z-Image, Qwen-Image-2.1 and Anima, 4 for Krea 2, whose schedule keeps its first noise levels higher, 1 for MOSS-Image, which starts losing fine detail from the second — and the rest at full size. Still 8 evaluations; one at half size costs about a quarter of a full one. No usual portrait or landscape format qualifies: the half of 832×1216 would fall under 512 px on one side. At 1024² on the reference machine's Compact it takes 31 % off a render on Krea 2, 21 % on Z-Image, 17 % on Qwen-Image-2.1, 19 % on Anima, and 7 % on MOSS-Image, whose single half-size evaluation is all it allows ([Numbers](#numbers)).

The small grid is fed the averaged state plus the low band of the starting noise, so it holds a clean, correctly noised picture; the velocity brought back to full size is the one that leads the full state to the enlarged clean estimate, so it also takes away the fine noise a half grid cannot see, at the pace a full step would. The half-size steps dropped in 0.8.2 had neither: their enlarged velocity left that fine noise in place (sand on skin and walls), and their small grid was a plain decimation, whose aliasing the model read as texture (cracks). Each model's number of half-size evaluations was judged by eye on two prompts, a portrait and the library: no sand, no cracks, freckles, hair and book spines intact; one step more, and fine detail starts to go.

It produces a different image from the full-size recipe with the same seed (the composition is decided in the first steps: the same scene, sometimes another person), not a degraded copy of it. It does not apply with a LoRA, Detail, an edit, another number of steps or another schedule. Off, every evaluation is at full size and the image is the same, to the bit, as before: the one checked against diffusers, to which the deviations above apply.

### Next to Draw Things

[Draw Things](https://drawthings.ai) is the reference on the Mac; here, its community engine, [draw-things-community](https://github.com/drawthingsai/draw-things-community) at `bc3970c`, with the model's defaults (UniPC trailing, CFG 1). The same machine, freshly restarted, one engine loaded at a time; Z-Image Turbo, 1024², 8 steps (8 evaluations on each side), seed 42, three renders on each side over three variants of the same prompt. Draw Things was measured on 13/10/2026 (a 70-minute session) and nothing changed on its side since; Siliconed at full size on 14/10/2026, after the fix, and with Faster Rendering on 19/10/2026 (app 0.12, three renders of the library prompt each, every one in a freshly opened app). Siliconed's peaks in this table are 0.12's; they are lower since ([Numbers](#numbers)).

| | weights | time | peak memory (GiB) | read from disk per render |
|---|---|---|---|---|
| Siliconed, Standard | the publisher's bf16, computed in fp32 | 113–119 s | 2.6 | 51–58 GB |
| Siliconed, Light | GGUF Q4_K_M | 118–120 s | 2.0 | 3.5–8.7 GB |
| Siliconed, Standard, Faster Rendering on | the publisher's bf16, computed in fp32 | 85–86 s | 2.8 | 51–58 GB |
| Siliconed, Light, Faster Rendering on | GGUF Q4_K_M | 90–91 s | 2.1 | 3.5–8.7 GB |
| Draw Things, Exact | f16 | 167–179 s | 5.3 | 99 GB |
| Draw Things, default | 8 bits (`q8p`) | 142–164 s | 3.5–4.3 | 11 GB |
| Draw Things, 6-bit | 6 bits (`q6p`) | 136–138 s | 3.5 | 6–14 GB |

The comparison is about wall-clock time, not identical arithmetic: the samplers and the weights differ, so do the images. Peak memory does not count the weights mapped from disk, on either side. Siliconed makes the machine run hotter, which follows from computing in fp32. Siliconed's first two lines are every evaluation at full size, like Draw Things'; with Faster Rendering, the first 3 of its 8 evaluations run on a half-size grid, which Draw Things does not do: less work, a different image ([how](#faster-rendering)).

8-bit against 8-bit, on 23/10/2026 (app 0.14): Siliconed's Compact (the third party's GGUF Q8_0, kept as published) against Draw Things' default `q8p`, in its community engine, updated since to `6c1dd09`; Z-Image Turbo, 1024², 8 steps, seed 42, the library prompt; each render in a freshly started process on each side (the app, the engine's server), three on each side, then Siliconed again as the control.

| | weights | time | peak memory (GiB) | read from disk per render |
|---|---|---|---|---|
| Siliconed, Compact, Faster Rendering and the Neural Engine (both on for these series) | GGUF Q8_0 | 81.1–83.2 s (control 81.3 s) | 2.6 | 3.7–11 GB |
| Draw Things, default | 8 bits (`q8p`) | 135.0–144.2 s | 4.3 | 10–18 GB |

## Memory and the budget

The largest model, Krea 2 Turbo in Standard (sizes: [Model management](APP.md#model-management)), runs on a 16 GB Mac within a fixed budget. Weights are read from disk as they are needed. Siliconed never takes more than ¾ of the Mac's memory minus 2 GB (4 GB on an 8 GB Mac, 10 GB on 16 GB, 22 GB on 32 GB): each render computes what it needs before it starts, takes the fastest plan that fits, and checks its own peak against that figure. What other applications hold stays theirs.

**Within the budget, measured.** Each model's peak `phys_footprint` and its swap-outs (`vm_stat` delta) are measured on a 16 GB Mac with nothing else running. With 0.18 at 1024², every model peaks between 1.3 and 2.4 GiB, Standard or Compact, with 0 swap-out ([Numbers](#numbers)); with 0.15 at 1024×1536, between 1.7 and 2.8 GiB; Anima Turbo at 2048² peaks at 5.4 GB; Qwen-Image-2.1 at 1024×1536 with 3 references, 0 swap-out (0.14). Z-Image's VAE decodes large images in bands, to the same bits.

## How it compares

- **[Draw Things](https://drawthings.ai)**, the reference native app on the Mac: its engine is [measured above](#next-to-draw-things) on the same machine. Siliconed computes in fp32 — with the accelerators on, part of the DiT on the Neural Engine and the GPU's tensor units where this Mac's measure finds it faster — and is checked stage by stage against the reference implementation, in fp32, and in bf16 for the accelerators.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [diffusers](https://github.com/huggingface/diffusers) in Python**: the same models, without a Python environment or a node graph to wire. Siliconed implements them natively and checks itself against diffusers and [transformers](https://github.com/huggingface/transformers); `silicontrol` covers scripting and automation.

## Community diagnostic

One machine is not a benchmark. **Measure This Mac…** (Help menu) shows the measure this Mac took before its first render, in about a minute and without any model. First the map: a finished measure is always the map of this Mac, applied as it stands. Under it, how sure each engine's and each set's figure is (a plus-or-minus, from how far their identical calls agreed) and the cores the other programs used while it measured; then what it found, each thing in a sentence with what the plan does with it — an engine with no figure or a set not measured together (the plan does not offer it), a figure that looks wrong (bounded, or left out), a Mac that was busy (the figures that differed are read on their quick side), a swap, a minute overrun. A route that computed wrong, or whose load killed the measuring process, is left out of the plan and marked ✗. Then what each engine reaches, alone, one titled line per route measured — the GPU, the CPU's matrix unit and the Neural Engine, on matrix products of the models' sizes and on attention, bf16 products, products on published int8 (GPU and Neural Engine) and Q8_0 weights — its speed in TFLOP/s, its share of the GPU's peak as a bar (the GPU's matrix products only: the peak is the one its cores and clock announce, and a share of it would mislead on an attention or on another engine), and an “Exact” column, ✓ where the result matched the exact one computed in double precision; then the engines together, what each keeps of its speed alone when they share a product and what the set reaches against the sum of its engines alone; the memory the measure held, its duration and, if you have rendered since, how far the plan's prediction was from your last render. As an option, **Share… ▸ Open a GitHub Issue** copies the JSON report and opens a GitHub issue form to paste it into (**Copy JSON** alone, or you send it another way): chip, CPU and GPU cores, what the registry says of the silicon (the GPU's clock, the CPU's clusters, the Neural Engine's generation), memory, macOS, the CPU's feature flags, the model and the raw measurements it was fitted on (for each engine and kernel its fixed cost and its rate, how much each engine keeps beside the others, the heat), whether each computed right, the memory it held, the seconds of each stage, what the measure found (its findings, each with its effect on the plan) and how far identical calls agreed, the last render's predicted and measured times by class, the thermal state and power source before and after. Nothing is sent unless you submit the issue.

We read the reports back, one line per kind of Mac — each engine's rate and the peak announced, what each keeps beside the others, the duration, the verdict —, to check that one model describes every Mac, those we never held included. A report never adds a rule for a chip: there are none.

Render times on the reference machine are under [Numbers](#numbers).

## Verification

Each stage (text encoder, denoiser, VAE) is compared with the reference implementation, diffusers and transformers in fp32 on CPU, on the same input. The deviation is measured channel by channel, each channel against its own norm, so that a fault confined to one channel cannot hide in a global figure. Where the fp32 reference carries its own rounding error, both are judged against an fp64 computation of the same stage, and the engine may be no further from it than the reference plus 25 %. Since 0.12 the arithmetic rounds where the reference rounds. The comparison with the reference is included: the oracle scripts in `tools/oracle/` and the `siliconed check` command run it on your Mac, stage by stage, with the fingerprints of our goldens to compare yours against; [VERIFY.md](VERIFY.md) says how, and what it costs. The measurement journal is not part of this repository. The Prompt Lab's writer is held to the same standard: its greedy decoding gives transformers' tokens, and its logits match when the reference's own tokens are forced (`siliconed check qwen-image-2.1 writer`). The diagnostic in the app checks a machine against exact results computed on its own CPU.

## Build from source

macOS 26 or later, Apple silicon, Xcode 26 or later. The Command Line Tools alone are not enough: their SDK lacks SwiftUI's macros, and the app's phrases are compiled by `xcstringstool`, which only Xcode ships (`tools/app.sh` says so and stops).

```
swift build                 the app and silicontrol
swift test                  the pure unit tests, no weights needed
tools/app.sh                builds Siliconed.app at the repository root and opens it
tools/app.sh --no-open      builds only
tools/dmg.sh                packages the built app as .build/Siliconed-<version>.dmg
```

`swift test` needs nothing else: no weights, no Python. Siliconed ships an app, not a library: the engine's declarations are internal to the package.

By default the app is signed ad hoc: it runs on the Mac that built it. To distribute it, sign it with your Developer ID:

```
tools/app.sh --sign "Developer ID Application: <name> (<team>)"
```

If `SILICONED_NOTARY_PROFILE` names a `notarytool` keychain profile, the app is also notarized and stapled; `tools/dmg.sh` then signs and notarizes the disk image with the same identity.
