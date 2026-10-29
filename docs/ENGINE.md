English · [Français](ENGINE.fr.md)

# The Siliconed engine

How Siliconed measures your Mac, picks its kernels, stays within its memory budget and proves its results, with the numbers. For what the app does, see [The Siliconed app](APP.md).

## An engine measured on your Mac

**No two Macs compute alike, so Siliconed does not guess.** Before the first render after each install or update — and whenever the Mac or macOS changes — it measures the Mac it runs on, in about a minute and without any model. It times a fixed grid of matrix products spanning the shapes of every model's DiT blocks — the same grid whatever is installed — on the GPU, on the CPU's matrix unit (AMX), on the Neural Engine and on the GPU beside each, attention through three kernels, products read straight from published int8 and Q8_0 weights, and bf16 products on the GPU's tensor units, and checks every output against the exact result. That report becomes **this Mac's map**, and the map picks the kernels: one turns on only if it computed right and beats the default by at least 5 % (more if the machine drifted while it was being measured). No kernel is switched on by chip name.

What the map decided on the M1 Pro of the numbers below, 16 GB, on 24/10/2026 (app 0.15):

| Kernel | Decision | Against the default |
|---|---|---|
| GPU + CPU matrix unit (AMX) co-execution | **on** | ×1.25 |
| the Neural Engine beside the GPU | **on** | ×1.39 |
| attention split by heads between the GPU and the CPU | **on** | ×1.53 |
| flash attention | off, inside the margin | ×1.10 |
| products on published int8 weights | off | ×0.98 |
| products on published Q8_0 weights | off | ×0.38 |
| bf16 products on the GPU's tensor units | off | ×0.48 |

The products on published int8 weights (the Compact of Krea 2 and Qwen-Image-2.1) won on this chip in the measurement made with 0.12 (18/10/2026); with 0.15 they tie with the default and stay off. The AMX and the Neural Engine share the products shape by shape: the Neural Engine takes a shape wherever it keeps 90 % of the AMX's speed there, because that saves energy, and where all three engines at once beat both pairs, the GPU, the AMX and the Neural Engine share the same product (on this M1 Pro, the square and widening products at 1024², 5 to 10 % faster than the best pair; never the narrowing ones, where the AMX slows the others down). This is one Mac: another chip measures its own and decides otherwise, and its report can fill a line of the [Community diagnostic](#community-diagnostic) table. The map buys speed and nothing else: it never changes the image beyond what the arithmetic it checked allows. The Neural Engine's fp16 is your choice, per model (below). Help → **Measure This Mac** shows the map and lets you share it.

### The Neural Engine

The Mac's Neural Engine computes in fp16. Where this Mac's map turned it on, it takes part of the DiT's matrix products beside the GPU, shape by shape: each weight column and each activation row is lifted by a power of two into fp16's range and brought back in fp32, and rows that overflow go back to the GPU. The text encoders and the VAE never use it. The **fp32** tag in the DiT module of the rack switches it on or off, per model: **off by default on the Standard**, which stays on the publisher's weights computed in fp32, with an image verified against the reference; **on by default on Compact and Light**, whose 8- and 4-bit weights already produce a different image from the Standard's. The tag reads **fp32 · ANE** when it is on, and an image computed with it carries `neural-engine` in its metadata. Where the map left the Neural Engine off, the tag is a plain fp32 and there is nothing to choose.

On the M1 Pro, 1024², Standard, Faster Rendering, seed 42, the library prompt (“a 30 year old woman posing in a library”), 23/10/2026, app 0.14:

| | Krea 2 Turbo | Z-Image Turbo | Qwen-Image-2.1 Turbo |
|---|---|---|---|
| time in the app, off / on | 144.6–144.9 s / 135.4–137.4 s | 86.8–93.9 s / 83.4–84.0 s | 103.6 s / 99.9–100.2 s |
| power draw of the whole machine, off / on | 43.9 W / 35.3 W | 38.0 W / 31.2 W | 45.4 W / 36.9 W |
| energy per image, on against off | −26 % | −16 % | −20 % |
| the image, on against off | PSNR 35.9 dB: the same scene, face and light, the finest details shifted (book spines) | 55.9 dB, a tenth of a level on average | 56.3 dB, a tenth of a level on average |
| deviation of the denoiser's output from the exact fp64 result, with half of every product's rows on the Neural Engine | 4.8·10⁻³ (512²) | 3.6·10⁻³ (1024²) | 4.4·10⁻³ (512²) |
| …as a share of the publisher's bf16 deviation | 0.14 | 0.06 | 0.31 |

Times: renders queued in the app with `silicontrol`, each in a newly launched app, 0 swap-out; off, three Z-Image renders in a row took 93.9, 89.0 and 86.8 s with the same bits, the first after the library had been moved. These are 0.14's times and the library prompt; 0.15's, with lower peaks, are under [Numbers](#numbers). Power and energy: the same comparison through the command line, four renders per model alternating on and off, the whole machine's power draw (display included) read from the power adapter's telemetry. The deviation is that of one evaluation of the denoiser with more of the Neural Engine than any map gives (half of every product's rows), as printed by `siliconed check <model> dit --neural-engine` ([VERIFY.md](VERIFY.md)), 24/10/2026; the bf16 one is diffusers' deviation with the publisher's bf16 weights, which is how most tools compute.

## Numbers

Reference machine: MacBook Pro M1 Pro (8 cores, 14 GPU cores), 16 GB. App 0.15, 25/10/2026 (MOSS-Image: app 0.16, 26/10/2026), seed 42, 8 steps, Faster Rendering on (the default), the prompt “Photo of a young Irish woman with wavy auburn hair and freckles, in a shorty wetsuit, holding her surfboard on a wild beach in the west of Ireland, Atlantic waves behind her, 35 mm photograph.”

| | Krea 2 Turbo | Z-Image Turbo | Qwen-Image-2.1 Turbo | Anima Turbo | MOSS-Image |
|---|---|---|---|---|---|
| 1024², Compact (8 bits, the default), Neural Engine on | **135.8 s** | **83.1 s** | **100.2 s** | **33.1 s** | **107.7 s** |
| 1024², Standard (the publisher's bf16, computed in fp32) | 147.8 s | 82.4 s | 110.4 s | 34.3 s | 107.3 s |
| 1024², Light (4 to 6 bits), Neural Engine on | 145.0 s | 87.5 s | 103.7 s | 34.7 s | — (no Light) |
| 1024×1536, Compact / Standard | 207.7 s / 223.1 s | 130.2 s / 133.1 s | 161.1 s / 170.9 s | 53.6 s / 53.7 s | 174.2 s / 174.2 s |
| the first image of a new prompt (its text encoding included), 1024², Compact / Standard | 136.7 s / 150.2 s | 84.2 s / 84.9 s | 105.7 s / 118.4 s | 33.8 s / 33.9 s | 109.9 s / 109.7 s |
| peak memory of the app, 1024² · 1024×1536, Compact / Standard | 2.1 · 2.7 / 2.1 · 2.6 GiB | 1.9 · 2.8 / 1.9 · 2.8 GiB | 1.5 · 1.7 / 1.5 · 1.7 GiB | 1.3 · 2.0 / 1.3 · 2.0 GiB | 2.4 · 2.8 / 2.4 · 2.8 GiB |
| read from disk per image, 1024², Compact / Standard | 98 / 195 GB | 0 / 44 GB | 0.2 / 135 GB | 0 / 0 GB | 0.1 / 76 GB |
| 1024², Standard, every evaluation at full size (0.12) | 221.1 s | 117.7 s | 142.7 s | 44.4 s | 116.9 s (0.16) |
| other, Standard | | | an edit with 1 image 211 s, with 2 images 297 s | 512² 12.1 s |  |
| deviation of the denoiser's output (`model_out`, worst channel) from the fp32 reference · from the fp64 exact (the reference's own) | 2.6·10⁻⁴ · 3.3·10⁻⁴ (3.1·10⁻⁴), 1024² | 9.8·10⁻⁶ · 1.2·10⁻⁵ (1.3·10⁻⁵), 1024² | 3.1·10⁻⁶ · 2.6·10⁻⁶ (2.5·10⁻⁶), 512² | 1.4·10⁻⁵ · per latent plane 2.9·10⁻⁶ (3.8·10⁻⁶), 1024² | per latent plane 1.2·10⁻⁶ · 1.3·10⁻⁶ (1.4·10⁻⁶), 512² |

**Full precision costs almost nothing.** The Standard computes the publisher's bf16 weights in fp32, the precision the reference is checked in; the Compact reads 8-bit weights published by third parties and computes in fp32 too. On Z-Image and Anima the Standard takes the Compact's time within 4 %. On Krea 2 and Qwen-Image-2.1 it takes 6 to 10 % more: at full precision their weights do not stay in a 16 GB Mac's memory and are read again from disk at each step, 195 GB and 135 GB per image at 1024². MOSS-Image's Standard reads 76 GB per image and still takes the Compact's time: its steps are long enough for the disk to keep up. That is also why every peak above stays under 3 GiB: weights mapped from disk are not counted in `phys_footprint`, since the system can drop them at any moment. What the Standard does cost is disk space, twice the Compact's, and energy, the Neural Engine being off on it by default ([The Neural Engine](#the-neural-engine)).

Each line is a `silicontrol measure` series in a freshly opened app, the series anyone can repeat ([REMOTE-CONTROL.md](REMOTE-CONTROL.md)): a load render, a witness (the library prompt, “a 30 year old woman posing in a library”), two renders of the prompt above, a witness again. A time is the second render's, from the request to the PNG, the prompt already encoded; the first image of a new prompt adds its text encoding. The two witnesses of each series stayed within 1.4 % of each other, 0 swap-out, and each image came out identical, to the bit, from one render to the next. Peak memory is the app's `phys_footprint`, sampled every 100 ms. Every evaluation at full size: 18/10/2026, app 0.12, 0 swap-out. The other times: Qwen-Image-2.1 on 15/10/2026 with 0.10, Anima Turbo's 512² with the engine alone (14–15/10/2026). The deviations are what `siliconed check <model> dit --size <side>` printed on the same Mac on 24/10/2026, against goldens written by `tools/oracle/`; on yours they differ in the last digits ([VERIFY.md](VERIFY.md), [Verification](#verification)).

### Faster Rendering

On by default, at the head of Settings, for all five models. From 1024 px on each side (1024², 1024×1536, 1536×1024…), the first evaluations run on the half-size grid — 3 of 8 for Z-Image, Qwen-Image-2.1 and Anima, 4 for Krea 2, whose schedule keeps its first noise levels higher, 1 for MOSS-Image, which starts losing fine detail from the second — and the rest at full size. Still 8 evaluations; one at half size costs about a quarter of a full one. No usual portrait or landscape format qualifies: the half of 832×1216 would fall under 512 px on one side.

The small grid is fed the averaged state plus the low band of the starting noise, so it holds a clean, correctly noised picture; the velocity brought back to full size is the one that leads the full state to the enlarged clean estimate, so it also takes away the fine noise a half grid cannot see, at the pace a full step would. The half-size steps dropped in 0.8.2 had neither: their enlarged velocity left that fine noise in place (sand on skin and walls), and their small grid was a plain decimation, whose aliasing the model read as texture (cracks). Each model's number of half-size evaluations was judged by eye on two prompts, a portrait and the library: no sand, no cracks, freckles, hair and book spines intact; one step more, and fine detail starts to go.

It produces a different image from the full-size recipe with the same seed (the composition is decided in the first steps: the same scene, sometimes another person), not a degraded copy of it. It does not apply with a LoRA, Detail, an edit, another number of steps or another schedule. Off, every evaluation is at full size and the image is the same, to the bit, as before: the one checked against diffusers, to which the deviations above apply.

### Next to Draw Things

[Draw Things](https://drawthings.ai) is the reference on the Mac; here, its community engine, [draw-things-community](https://github.com/drawthingsai/draw-things-community) at `bc3970c`, with the model's defaults (UniPC trailing, CFG 1). The same machine, freshly restarted, one engine loaded at a time; Z-Image Turbo, 1024², 8 steps (8 evaluations on each side), seed 42, three renders on each side over three variants of the same prompt. Draw Things was measured on 13/10/2026 (a 70-minute session) and nothing changed on its side since; Siliconed at full size on 14/10/2026, after the fix, and with Faster Rendering on 19/10/2026 (app 0.12, three renders of the library prompt each, every one in a freshly opened app). Siliconed's peaks in this table are 0.12's; 0.15 lowered them ([Numbers](#numbers)).

| | weights | time | peak memory (GiB) | read from disk per render |
|---|---|---|---|---|
| Siliconed, Standard | the publisher's bf16, computed in fp32 | 113–119 s | 2.6 | 51–58 GB |
| Siliconed, Light | GGUF Q4_K_M | 118–120 s | 2.0 | 3.5–8.7 GB |
| Siliconed, Standard, Faster Rendering (default) | the publisher's bf16, computed in fp32 | 85–86 s | 2.8 | 51–58 GB |
| Siliconed, Light, Faster Rendering (default) | GGUF Q4_K_M | 90–91 s | 2.1 | 3.5–8.7 GB |
| Draw Things, Exact | f16 | 167–179 s | 5.3 | 99 GB |
| Draw Things, default | 8 bits (`q8p`) | 142–164 s | 3.5–4.3 | 11 GB |
| Draw Things, 6-bit | 6 bits (`q6p`) | 136–138 s | 3.5 | 6–14 GB |

The comparison is about wall-clock time, not identical arithmetic: the samplers and the weights differ, so do the images. Peak memory does not count the weights mapped from disk, on either side. Siliconed makes the machine run hotter, which follows from computing in fp32. Siliconed's first two lines are every evaluation at full size, like Draw Things'; with Faster Rendering, the first 3 of its 8 evaluations run on a half-size grid, which Draw Things does not do: less work, a different image ([how](#faster-rendering)).

8-bit against 8-bit, on 23/10/2026 (app 0.14): Siliconed's Compact (the third party's GGUF Q8_0, kept as published) against Draw Things' default `q8p`, in its community engine, updated since to `6c1dd09`; Z-Image Turbo, 1024², 8 steps, seed 42, the library prompt; each render in a freshly started process on each side (the app, the engine's server), three on each side, then Siliconed again as the control.

| | weights | time | peak memory (GiB) | read from disk per render |
|---|---|---|---|---|
| Siliconed, Compact, Faster Rendering and the Neural Engine (both by default) | GGUF Q8_0 | 81.1–83.2 s (control 81.3 s) | 2.6 | 3.7–11 GB |
| Draw Things, default | 8 bits (`q8p`) | 135.0–144.2 s | 4.3 | 10–18 GB |

## Memory and the budget

The largest model, Krea 2 Turbo in Standard (sizes: [Model management](APP.md#model-management)), runs on a 16 GB Mac within a fixed budget. Weights are read from disk as they are needed. Siliconed never takes more than ¾ of the Mac's memory minus 2 GB (4 GB on an 8 GB Mac, 10 GB on 16 GB, 22 GB on 32 GB): each render computes what it needs before it starts, takes the fastest plan that fits, and checks its own peak against that figure. What other applications hold stays theirs.

**Within the budget, measured.** Each model's peak `phys_footprint` and its swap-outs (`vm_stat` delta) are measured on a 16 GB Mac with nothing else running. With 0.15 at 1024×1536, every model peaks between 1.7 and 2.8 GiB, Standard or Compact, with 0 swap-out ([Numbers](#numbers)); Anima Turbo at 2048² peaks at 5.4 GB; Qwen-Image-2.1 at 1024×1536 with 3 references, 0 swap-out (0.14). Z-Image's VAE decodes large images in bands, to the same bits.

## How it compares

- **[Draw Things](https://drawthings.ai)**, the reference native app on the Mac: its engine is [measured above](#next-to-draw-things) on the same machine. Siliconed computes in fp32 and is checked stage by stage against the reference implementation.
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI) or [diffusers](https://github.com/huggingface/diffusers) in Python**: the same models, without a Python environment or a node graph to wire. Siliconed implements them natively and checks itself against diffusers and [transformers](https://github.com/huggingface/transformers); `silicontrol` covers scripting and automation.

## Community diagnostic

One machine is not a benchmark. **Measure This Mac** (Help menu) shows the map this Mac measured before its first render, in about a minute and without any model: the matrix products of the models' blocks on the GPU, on the CPU's matrix unit, on the Neural Engine and on the GPU beside each, fp16 and bf16 products, products on published int8 and Q8_0 weights, attention through the engine's three kernels, weight widening on the CPU and on the GPU — each result checked against the exact one computed in double precision — and what the map decided from them. It copies the JSON report and opens a GitHub issue form to paste it into (or you send it another way): chip, CPU and GPU cores, memory, macOS, the CPU's feature flags, each throughput and whether it computed right, the thermal state and power source before and after. Nothing is sent unless you submit the issue.

The reports fill this table, one line per chip (TFLOP/s, fp32 unless noted):

| Chip | CPU cores | GPU cores | Memory | GPU | CPU matrix unit | Both | Attention | GPU fp16 | Source |
|---|---|---|---|---|---|---|---|---|---|
| M1 Pro | 8 | 14 | 16 GB | 3.5 | 2.2 | 5.1 | 2.5 | 3.4 | reference machine |
| *your line here* | | | | | | | | | [report](https://github.com/gwenn-ha-dev/Siliconed/issues/new?template=diagnostic.yml) |

Render times on the reference machine are under [Numbers](#numbers).

## Verification

Each stage (text encoder, denoiser, VAE) is compared with the reference implementation, diffusers and transformers in fp32 on CPU, on the same input. The deviation is measured channel by channel, each channel against its own norm, so that a fault confined to one channel cannot hide in a global figure. Where the fp32 reference carries its own rounding error, both are judged against an fp64 computation of the same stage, and the engine may be no further from it than the reference plus 25 %. Since 0.12 the arithmetic rounds where the reference rounds. The comparison with the reference is included: the oracle scripts in `tools/oracle/` and the `siliconed check` command run it on your Mac, stage by stage, with the fingerprints of our goldens to compare yours against; [VERIFY.md](VERIFY.md) says how, and what it costs. The measurement journal is not part of this repository. The diagnostic in the app checks a machine against exact results computed on its own CPU.

## Build from source

macOS 26 or later, Apple silicon, Xcode 26 or later.

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
