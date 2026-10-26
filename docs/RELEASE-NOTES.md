English · [Français](RELEASE-NOTES.fr.md)

# Siliconed — release notes

## 0.16

### A fifth model: MOSS-Image

- **[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)**, by the OpenMOSS team, joins
  Krea 2, Z-Image, Qwen-Image-2.1 and Anima. A single-stream DiT of 5.1 billion parameters that reads
  the prompt with Qwen3.5-9B, a hybrid language model (three layers in four of linear attention,
  Gated DeltaNet, new in the engine), and decodes with FLUX.2's VAE.
- **MOSS-Image is a beta**, and the app says so (a “Beta” badge on its card and in Models): its 8-step
  recipe is the community's LoRA, not its publisher's. It leaves beta when OpenMOSS releases its own
  fast version.
- **Its recipe is the community's**: its publisher released no fast version, so Siliconed runs
  MOSS-Image-SFT under [lodestones' DMD LoRA](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd)
  at strength 1: 8 evaluations, no CFG. The LoRA is kept beside the weights in fp32, never merged:
  merged into the bf16 weights, its deltas would be lost to rounding. Text-to-image; no Detail, no
  edit.
- **Your LoRAs** stack on top of the 8-step one: ComfyUI's and ai-toolkit's layout (the attention's
  fused `qkv`, split exactly), kohya's and diffusers'. LoRAs trained on MOSS-Image Base apply as the
  community uses them. Each imported module's ΔW equals PyTorch's from the published file.
- **Times**: on an M1 Pro with 16 GB, seed 42, 8 evaluations, queued in the app: 1024² in 107.7 s
  in Compact, 107.3 s in Standard; 1024×1536 in 174.2 s in both.
- **Faster Rendering takes one half-size evaluation**, not three: from the second, MOSS-Image
  smooths skin and softens fine detail where the other models keep it. 8 % less time at 1024²
  (116.9 s with every evaluation at full size).
- **The preview shows during the render**, as with the other models: a projection of the FLUX.2
  latent's 128 channels to colour, fitted on twelve renders.
- **Checked against the reference**, stage by stage: the tokenizer and the schedule to the bit, the
  text encoder's 32 layers, the denoiser with and without the LoRA, the 8-step trajectory and the
  decoder, each at the fp64 witness where the reference carries its own error. At 512², the
  denoiser's output is 1.3·10⁻⁶ from the exact result, the reference's own 1.4·10⁻⁶. Under the
  Neural Engine, both versions' denoiser stays within bf16's own error plus 25 %, as for the other
  models.
- **Standard or Compact**: 27.5 GB (the publisher's bf16, the LoRA in fp32) or 17.9 GB (the DiT and
  the text encoder in int8 convrot, as Comfy-Org publishes them); the Compact is preselected, as for
  the other models, with the Neural Engine on by default. No Light yet: a 4-bit DiT is published,
  not yet checked.
- **Within the budget**: its memory need is computed from the model's shapes before the first
  computation. The app's peak: 2.4 GiB at 1024², 2.8 GiB at 1024×1536, in both versions.
- **License**: Apache 2.0, the LoRA included. Commercial use allowed.

### Every Mac

- **The first launch after an update no longer runs slower**: when the first gesture was an
  exploration, a grid or `silicontrol add`, the session ignored this Mac's measure and ran on the
  defaults — the AMX off, about 20 % per step on an M1 Pro. `silicontrol status` now says which
  measure the session uses.
- **A measure taken off rest is taken again**: measured while the Mac was warm, in Low Power Mode, or
  with its swap growing (another app pressing on memory), the choices still serve, the Chain's chip
  stays orange, and the next launch at rest measures again before its first render — once.
- **An update that leaves the engine untouched keeps this Mac's measure**: the measure is tied to the
  engine's sources and compiler, no longer to the app's build; one that moves the engine measures again.
- **macOS 26 or later**: the only version Siliconed was ever measured and checked on. The app no
  longer installs on macOS 15.

## 0.15

### Compact by default

- **The Compact is now preselected, and marked recommended,** when you install a model on a Mac with
  more than 8 GB (the Light stays preselected on 8 GB). On an M1 Pro it renders as fast as the Standard or faster, takes half its disk space, and has the Neural Engine on by default. A model already
  installed keeps its version; the three stay one click apart.
- **Full precision costs almost nothing.** The Standard, the publisher's bf16 weights computed in
  fp32, takes the Compact's time within 4 % on Z-Image and Anima, and 6 to 10 % more on Krea 2 and
  Qwen-Image-2.1, whose weights do not fit in memory at full precision and are read again from disk
  at each step. Every figure: [Numbers](ENGINE.md#numbers).

### Larger images, lower peaks

- **No area ceiling any more: this Mac's memory budget decides.** The size picker offers 1536² and
  2048², greyed with the Mac they need where the budget refuses them. Anima Turbo renders 2048² on a
  16 GB Mac, peaking at 5.4 GB.
- **Lower peaks, the same pixels**: weights are now read from disk a few tensors at a time instead
  of a layer at a time — 0.8 to 1.4 GB less for Krea 2, 1.1 GB for Z-Image, 0.5 GB for
  Qwen-Image-2.1.
- **Fixed: Krea 2's memory crept up during the denoising** (objects freed only at the end of the
  render).

### Check our numbers yourself

- **`siliconed check <model>`** compares every stage of a model with the reference implementation,
  on your Mac, against goldens you write with the scripts of `tools/oracle/` and whose fingerprints
  we publish. `--neural-engine` judges the Neural Engine's arithmetic, `--variant compact|light`
  the Compact and the Light against the reference run on the third party's published weights. The
  command ships inside the app (`Contents/Helpers`): it checks the very engine the app renders
  with, without Xcode. How, and what it costs: [VERIFY.md](VERIFY.md).
- **`silicontrol measure`** repeats our timings and peaks with the app alone: a series of renders
  between two witnesses, the app's sampled peak memory, swap-outs, disk reads and each image's
  checksum, as a TSV. `silicontrol set faster-rendering on|off` switches Faster Rendering like
  Settings does.

### Crop and save

- **Crop** a rendered image or an edit reference: C to enter, corner and side handles, rule of thirds
  while dragging, Return to accept, Escape to cancel; never under 512 px a side. The crop is a new
  history entry; the original stays.
- **Save as PNG, JPEG or HEIC**; the last format is remembered. JPEG and HEIC carry no metadata.

### Engine

- **Large formats split their attention** between the GPU and the CPU at the share this Mac's
  measure finds fastest, the CPU heads' softmax in one NEON pass.
- **A bf16-activation path** for the denoiser's products, chosen per Mac by its measure, judged
  against diffusers in bf16; off on the M1 Pro, where it does not win.
- **This Mac is measured again** before the first render after the update, about a minute.

## 0.14

### The Neural Engine

- **The Mac's Neural Engine joins the GPU** in the image model's matrix products, where this Mac's
  measure finds the pair faster than the GPU alone. Beside the CPU's matrix unit, it takes a
  product's shape wherever it keeps 90 % of that unit's speed, for the energy it saves. It computes
  in fp16: each weight column and each activation row is lifted by a power of two into fp16's
  range, then brought back in fp32, and rows that overflow go back to the GPU. The text encoders
  and the VAE never use it.
- **Your choice, per model**: the **fp32** tag of the rack's DiT module switches it. **Off by
  default on the Standard**, which stays the publisher's weights computed in fp32, its image checked
  against the reference; **on by default on Compact and Light**, whose 8- and 4-bit weights already
  make another image. The tag reads **fp32 · ANE** when it is on, and an image computed with it
  carries `neural-engine` in its metadata. Where the measure left the Neural Engine off, there is
  nothing to choose.
- **On an M1 Pro**, 1024², Standard, Faster Rendering: 16 to 26 % less energy per image, and as
  fast or faster — Krea 2 135.4–137.4 s instead of 144.6–144.9 s, Z-Image 83.4–84.0 s instead of
  86.8–93.9 s, Qwen-Image-2.1 99.9–100.2 s instead of 103.6 s. The image moves by a tenth of a
  level on average on Z-Image and Qwen-Image-2.1; Krea 2 keeps its scene, face and light, and its
  finest details move. Every figure: [The Neural Engine](ENGINE.md#the-neural-engine).
- **Eight bits against eight bits**: Z-Image Compact at 1024², with Faster Rendering and the Neural
  Engine (both by default), renders in 81.1–83.2 s; Draw Things' default 8-bit `q8p`, 135.0–144.2 s
  on the same Mac ([Next to Draw Things](ENGINE.md#next-to-draw-things)).
- **This Mac is measured again** before the first render after the update, about a minute, as after
  every version: the measure now includes the Neural Engine, on each model's own products.
- **The remote control**: `silicontrol add --ane` or `--no-ane`, like a click on the tag;
  `neural_engine` in `models`, in jobs and in images.

### Open a Hugging Face model in the app

- **The app opens `siliconed://hf` links**, relayed by a page of the site that offers the .dmg
  when the app is not installed: what Hugging Face's « Use this model » button opens, once Hugging
  Face lists Siliconed. A family's own repository opens its installation; any other repository
  lists the `.safetensors` and `.gguf` files the app could import — each judged on its header
  before anything downloads —, with their size and license. The file chosen is downloaded, verified,
  then imported like a dropped file. Nothing in the link is trusted.
- **More LoRAs import**: an integer `alpha` (as sd-scripts saves it) is read instead of refused; a
  LoRA that touches only the attention finds its family from the installed models' shapes, when
  only one takes it all; a LyCORIS file or a text-encoder LoRA is refused with its own message
  instead of being taken for something else.

### Fewer downloads

- **A text encoder or a VAE you already have is taken instead of downloaded.** Dropped on the app,
  a file equal in size and SHA-256 to one an installation fetches — the 8-bit encoders of Z-Image,
  Qwen-Image-2.1 and Krea 2, Krea 2's bf16 encoder, the Standard encoders, every family's VAE — is
  copied where the download would have put it, then verified like a download.

### Longer prompts

- **Every model reads up to 1,024 tokens of prompt**, and refuses a longer one with a message. Until
  0.13, Z-Image, Anima and Krea 2 silently ignored everything past 512 tokens, as the reference
  pipelines do by default, and Qwen-Image-2.1 refused past 512. A prompt under 512 tokens gives the
  same image as before.

### Fixed

- **A LoRA trained with a different rank per layer showed « rank 0 »** in the Models window and in
  the rack's tooltip; it now shows its range, « ranks 2–57 » for inCase on Anima.

## 0.13

### Your library on another disk

- **Settings moves the models** to an external disk or a network share, opens a library already
  there, or brings it back to the Mac. Each file is copied, read back and compared (SHA-256) before
  anything switches; the old place is emptied last. Stopped or failed, the library is unchanged, and
  a new attempt keeps the files already verified. A FAT disk, a read-only folder or another library
  is refused before the first byte. ([#2](https://github.com/gwenn-ha-dev/Siliconed/issues/2))
- **Renders never read the other disk.** As soon as a model is chosen, what it reads is copied to
  the Mac for the session (`~/Library/Caches/Siliconed`; several models kept, the least recently
  used evicted, 5 GB always left free), and removed when the app quits — after a crash, at the next
  launch. The same seed gives the same image as from the Mac's own disk.
- **What belongs to the Mac stays on it**: the licenses you accepted, this Mac's measure, the render
  cache. Unplugged, the disk is flagged: nothing renders until it is back, and an empty library never
  takes its place. The remote control and the command line follow a moved library.

### A model imported before its family

- **Dropping a fine-tune before its family is installed** (a Civitai or Hugging Face checkpoint of
  Z-Image, Qwen-Image-2.1, Anima or Krea 2) now opens that family's installation, the same panel as
  in the Models window: the license in full, the space, the text encoder and the VAE — never the
  family's own DiT, since the file brings its own. Once installed, the file is imported, and the
  license accepted there covers it: no second sheet at its first render. Until 0.12 the app only
  said to install the family first. ([#5](https://github.com/gwenn-ha-dev/Siliconed/issues/5))
- **A download refused by Hugging Face** (HTTP 401 or 403) no longer sends you to a terminal to sign
  in: no repository the app installs from asks for an account. The message now points at what
  refuses — a proxy, a VPN, a firewall.

### Fixed

- **A render could come out entirely non-finite** (Z-Image at 1024² under a LoRA, at certain prompt
  lengths): Apple's fast fp32 matrix product sometimes writes invalid values into some output
  buffers. Each buffer and shape is now checked once, and recomputed in a fresh buffer when wrong;
  the same bits everywhere else, no measurable cost.
- **Detail on Qwen-Image-2.1**: More and Most no longer change the person. Calibrated on the 6-step
  LoRA the family used before 0.10, they weighed on steps where the Turbo still decides the face;
  they now touch the last evaluation alone (More 1.3, Most 2): the same person, finer texture.
  Normal is unchanged, to the bit.
- **The license sheet of an imported model** says the image model is the imported file, and links
  the licenses of the installed version's encoder and VAE — a Compact's or a Light's come from third
  parties.
- **Two conversions of the same model at once** (the app and the command line, say) no longer empty
  each other's file in progress: the second one is refused.
- **Krea 2 renders' metadata** name the Civitai page that carries the same bf16 file, so Civitai
  recognizes the model.
- Two engine settings never chosen by any Mac's measure (half-precision attention, GPU widening) are
  removed; the default paths are unchanged, to the bit.

## 0.12

### A fourth model: Krea 2 Turbo

- **[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo)**, by Krea, joins Z-Image Turbo,
  Qwen-Image-2.1 and Anima in the app, without Developer Mode. A 12.9-billion-parameter DiT, the
  largest of the four and the slowest, that reads the prompt with Qwen3-VL-4B (twelve of its layers,
  merged by a learned fusion) and decodes with the Qwen-Image VAE. Aesthetic images, from
  photography to painting.
- **Its recipe**: 8 evaluations, no CFG. Text-to-image, the rectangular formats, the four solvers
  in Developer Mode, each checked against diffusers on Krea's own schedule; LoRAs (ai-toolkit
  layout, read as diffusers and PEFT read it; ComfyUI gives the same update), applied whole on the
  Compact and the Light and judged against the reference, the time modules included;
  no editing by instruction, and no Detail: on this model its bell falls where the picture is still
  being decided, and redraws it instead of sharpening it. Subtle variations use 0.05 (0.1 changes
  the person).
- **Times**: on an M1 Pro with 16 GB, Standard, seed 42, 8 evaluations, queued in the app: 1024² in
  145.6 s, 1024×1536 in 216.9 s with Faster Rendering (below, on by default); 221.1 s and 355.3 s
  with every evaluation at full size.
- **Checked against the reference**: at 1024², the denoiser's output is 1.44·10⁻⁴ from diffusers
  in fp32.
- **Standard, Compact or Light**: 34.4 GB (the publisher's bf16), 19.5 GB (the DiT and the text
  encoder in int8 convrot, as Comfy-Org publishes them) or 12.8 GB (the DiT in GGUF Q4_K_M, as
  vantagewithai publishes it, with the Compact's text encoder). The Compact and the Light are
  checked to the bit against their reference dequantization.
- **Nothing comes from Krea's own repository**: it asks for a Hugging Face account, and the app has
  no sign-in. Each file comes from an open copy proven identical: the DiT and the text encoder from
  Comfy-Org (the DiT the very same file, the encoder equal tensor for tensor), the VAE from
  Qwen-Image (Krea's fp32 VAE is its bf16 one widened, value for value), the tokenizer from
  Qwen-Image-2.1-Turbo (the same file); Krea's configuration files are embedded as published. The
  license sheet says so, and links the card of every repository downloaded from.
- **Within the budget**: its memory need is computed from the model's shapes before the first
  computation, as for the other three. Peaks at 1024², the engine alone: 3.07 GB in Standard
  (3.51 GB at 1024×1536), 2.20 GB in Compact, 1.35 GB in Light.
- **License**: Krea 2 Community. Commercial use under $1M of annual revenue; a content filter is
  mandatory when the model is deployed. The app shows the license in full before the first byte is
  downloaded, with the attribution it asks for.

### Faster Rendering, on by default

- **A new setting, at the head of Settings, on by default, for all four models.** From 1024 px
  on each side (1024², 1024×1536, 1536×1024…), the first evaluations run on the half-size grid — 3 of 8 for Z-Image,
  Qwen-Image-2.1 and Anima, 4 for Krea 2, whose shifted schedule keeps its first noise levels
  higher — then the rest at full size. Still 8 evaluations; one at half size costs about a quarter
  of a full one. No usual portrait or landscape format qualifies: the half of 832×1216 would fall
  under 512 px on one side.
- **Times**, on an M1 Pro with 16 GB, Standard, seed 42, queued in the app (on / off): at 1024²,
  Z-Image 84.5 / 117.7 s, Qwen-Image-2.1 103.7 / 142.7 s, Anima 32.7 / 44.4 s, Krea 2 145.6 /
  221.1 s; at 1024×1536, on: 131.0 s, 161.7 s, 52.5 s and 216.9 s (Krea 2 off: 355.3 s). In
  Light, on, at 1024²: Z-Image 90–91 s, Qwen-Image-2.1 111.0 s, Anima 35.4 s, Krea 2 153.8 s — the
  Light saves disk and memory (Krea 2's peak in the app: 1.55 GB), not time.
- **Why it stays clean this time.** The half-size steps dropped in 0.8.2 decimated the state and
  enlarged the velocity: that velocity carried no high band, so the fine noise of the state was
  never taken away while they ran (sand on skin and walls), and the decimation's aliasing reached
  the model as texture (cracks). Now the half grid is fed the 2×2 mean of the state plus the low
  band of the starting noise — antialiased, at the right noise level —, and the velocity brought
  back to full size is the one that leads the full state to the enlarged clean estimate: it takes
  the fine noise away at the pace a full step would, and the state stays exactly on the recipe's
  path. Each model's number of half-size evaluations was judged by eye on two prompts, a portrait and the library, at the
  same seed: no sand, no cracks, freckles, hair and book
  spines intact; one step more, and the freckles fade, the shelves turn to blocks.
- **Another image, not a degraded one.** With the same seed, the picture is not the full-size
  recipe's: the composition is decided in the first steps (the same scene, sometimes another
  person). Not with a LoRA, Detail, an edit, another number of steps or another schedule.
- **Off, nothing moves**: every evaluation at full size, the same bits as before, the image checked
  against diffusers. The deviations from the reference published for each model are for that
  image.

### The engine measures your Mac before its first render

- **No render without a map of this Mac.** Before the very first render, and again whenever the
  Mac, macOS or the app changes, the app measures the chip, in about 30 to 40 seconds and without any
  model, then renders with what it found; the sidebar says so while it runs. Until 0.11, the map
  was optional, and the defaults were those of an M1 Pro.
- **What it measures**: the real matrix products of the four DiTs' blocks on the GPU, on the CPU's
  matrix unit (AMX) and on both at once, at several depths and widths; attention through three
  kernels; products read straight from published int8 and GGUF Q8_0 weights; bf16 products on the
  GPU's tensor units (macOS 26, Metal 4). Every output is checked against the exact result, computed
  in double precision.
- **What it decides**: a kernel turns on only if it computed right and beats the default by at
  least 5 %, or by the drift the measure itself saw if that is larger, at both a 512² and a 1024²
  image's sizes. The GPU + AMX co-execution, which turned on by chip name until 0.11, is now off
  everywhere unless measured faster, and the share each engine takes is computed for each product's
  shape instead of one frozen share. On the M1 Pro (18/10/2026): co-execution on (×1.30), attention
  split by heads between the GPU and the CPU on (×1.49, now also for Krea 2's grouped heads), flash
  attention off (×1.05, inside the margin), the int8 product on, the Q8_0 product off (×0.43), the
  bf16 tensor product off (×0.53). Another chip decides from its own measurements.
- **Published 8-bit weights read as they are**: where the map turns them on, the int8 weights of
  Compact versions (such as Krea 2's and Qwen-Image-2.1's) and the GGUF Q8_0 weights (Anima's Compact) are
  multiplied as published, instead of widened to fp32 first; the activations and the sums stay
  fp32. Where the GPU's tensor units multiply faster, the publisher's bf16 weights are read the same
  way. Each path is checked against the reference.
- **The map changes the speed, not the picture** beyond the arithmetic it checked; the same Mac
  with the same map still gives the same bits.
- **« Chain », at the head of the rack, opens the measure**: orange, with its chip, until this Mac is
  measured, a green chip after. **Measure This Mac** (Help menu) shows the map and what it
  decided; « Open the Issue » now copies the report and opens a short GitHub form to paste it into,
  since the report no longer fits in a link; « Copy JSON » serves whoever sends it another way.

### Closer to the reference, to the rounding

- **The arithmetic rounds where the reference rounds**: Euler steps, UniPC's corrector (solved as
  LAPACK does), the integer timesteps diffusers gives the DiT under DPM++ 2M and UniPC, the schedules
  of Z-Image, Anima and Krea 2, every model's rotary position tables, and the gated residuals and
  modulations (a·b + c in two roundings, as torch computes it). **Every picture changes by a few
  units in the last place**, and each model stays within the reference's own distance of an fp64
  computation: an image made with 0.11 does not come out again to the bit.
- **Z-Image at 832×1216 and 1216×832 changes image**: its image tokens are now padded to a multiple of
  32 as diffusers does; those two formats were 16 tokens short.
- **LoRAs reach every module the reference applies them to**: on Z-Image the outer modules (patch
  and caption embedders, time embedder, final layer), on Krea 2 the time embedding and modulation,
  which were imported and then ignored. A LoRA aiming at a module the engine does not apply is
  refused before any computation, with the reason.
- **Qwen-Image-2.1 edits keep a reference's transparency** as diffusers does: the VAE reads the
  image with its alpha, only the copy the encoder sees is pasted on white.

### Sturdier

- **A broken render is no longer delivered.** If a render comes out with non-finite values (a
  black image), the app says so and keeps nothing, instead of adding it to the history as a
  success; in a batch, only that image is dropped, the others are still delivered. Nothing non-finite is carried from one render to the next: the render cache neither
  keeps nor reads back such a value, and a Qwen-Image-2.1 edit keeps its attention keys and values
  only behind a finite first step (one bad value there used to blacken every later edit of the same
  image for the session); a VAE pass the GPU refuses is reported instead of ignored. `silicontrol` reports it as `"error"` with the code
  `non_finite_image`, and the verdict of `wait` (and the `end` line of `follow`) now carries the
  engine's `"warnings"`.
- **Formats per model**: Qwen-Image-2.1 wants sides in multiples of 32, Anima at most 2048 px; both
  are refused before any computation, and the size field says the chosen model's rule.
- **The render cache lasts one session**: emptied at launch and at quit, it no longer counts in the
  Models window's used space.
- **Importing never downloads**: a LoRA or a fine-tune whose model is not installed is refused, and
  the app says which model to install first; real errors (a full disk) are no longer swallowed.
- **Hostile files are refused**: a LoRA whose alpha pattern would make the matcher backtrack without
  end, non-finite alphas or strengths, rank 0, a GGUF Q8_0 block with a non-finite scale.
- **The memory need counts what it missed**: the scratch of rotated int8 weights (Krea 2's
  Compact overran its text stage by 7 MB), Z-Image's padded tokens. A long Krea 2 prompt, past about
  170 tokens, no longer stops with « arena exhausted ».
- **Attention without the head split is sliced by queries** where it would exceed the GPU's matrix
  limit (Z-Image at 1024×1536 on a Mac whose map turns the split off).
- The flash attention kernel computes in safe math and lost two branches that were never used and
  could misread a key; the int8 and Q8_0 products no longer write past the GPU's columns; a damaged
  download registry is ignored; the queue waits for an enlargement or a grid sheet to finish;
  `silicontrol` requests have a deadline.

### Interface

- **A LoRA takes its file's name**: an imported LoRA is called by its file, no longer by the name
  written in its metadata, often a trainer's default (« my first lora v1 copy copy »). The
  trainer's name, and the trigger words when the file states them beyond doubt, show in a tooltip on
  the LoRA, in the rack and in the Models window. LoRAs imported before keep their name.
- Krea 2's license sheet shows the whole license; hairlines and fills a little stronger, secondary
  text where it informs, one export symbol; the Settings window opens centered; the import dialog's
  sentence is shorter, so the Finder's sidebar keeps its width.
- For developers: Siliconed ships an app, not a library. The library product and its API
  documentation are gone; `swift build` builds the app and `silicontrol`.

Timings are measured on a MacBook Pro M1 Pro with 16 GB.

## 0.11

- **Enlarge ×4** (the button under the image, its context menu, the Image menu): macOS's own
  super-resolution, the kind of network Pixelmator uses. It sees only the image — no prompt —, so it
  sharpens edges, hair and texture without adding anything: 1024 × 1024 becomes 4096 × 4096 in a second
  or two, beside the original, which stays. The model is Apple's: macOS downloads it once, after
  asking. Needs macOS 26; `silicontrol enlarge i7` does the same.
- **The curtain compares an image with its enlargement**, and any two marked images of the same
  proportions; **zooming goes toward the pointer**, on the canvas and under the curtain, to slide
  the line over a detail.
- **Paste an image** with ⌘V (outside the prompt): copied in Procreate, Photoshop, Preview or the
  Finder, it goes where a drop on the Images box would — an image to edit, or a subject to name.
- **LoRA in the open**: the stack sits right under the model; « Add » is a menu of the model's
  LoRAs; each one shows its strength's slider, its figure (double-click: back to 0.8) and a remove
  button — no more popover to open and close.
- **Explore stands beside Generate**: the same gesture, many small images instead of one.
- **The rack says more, in fewer places**: each box shows what its stage took as soon as it ends;
  hovering a stage's name says what it does (text encoder, DiT, VAE…); the memory a render needs is
  said once, under the rack; what keeps Generate greyed (« Write a prompt »…) is said above it.
- **The queue scrolls** past a few renders and shows the running one at its head: its image k/N in a
  batch, the time left, a stop that lets the rest go on. The diagnostic leaves the toolbar (Settings
  and the Help menu keep it).

### Fixed

- A crash when marking images to compare, most likely with a large image still being read.
- PNG metadata always writes `CFG scale: 1`: Civitai's « Remix » no longer starts from its own
  default scale, which burnt the image.
- The Dock icon no longer bounces when the queue empties.

## 0.10.2

- **A Settings window** (Siliconed → Settings…, ⌘,) replaces the switches of the app menu: Developer
  Mode, Experimental (Named Subjects, Negative LoRA Strengths), the comparison's curtain or side by
  side, this Mac's capacity map, the silicontrol command (installed or not, and where) and the models
  folder, shown in the Finder; each with a sentence saying what it does.
- **Negative LoRA strengths** (Settings → Experimental): the LoRA sliders go from −1.5
  to 1.5, in the panel and in the exploration. A negative strength pushes the image away from what
  the LoRA learned: it is meant for « slider » LoRAs (age, detail…), trained to be used from −x to
  +x; on an ordinary LoRA it mostly damages the image, hence the switch. Off, a negative value is
  refused, with what to turn on. `silicontrol add --lora name:-0.5` follows the same switch.
- **A solver axis in the exploration** (Developer Mode): Euler, Euler a, DPM++ 2M and UniPC side by
  side, same seed, same steps; `silicontrol grid --x solver=euler,dpmpp-2m` does the same. The
  sheet's subtitle names the solver and the ComfyUI schedule when they are not the defaults.

### Fixed

- **LoRA strengths typed with a point in the exploration** (`0.4, 0.8`) were unreadable on a Mac
  set to a language that writes decimals with a comma, and « Explore » stayed greyed out. Both
  forms now read.

## 0.10.1

- **Anima images credit Anima on Civitai.** Drop an Anima render on Civitai and it links the model
  page, as Z-Image and Qwen-Image-2.1 renders already did: the PNG names Anima Turbo v1.1
  (circlestone_labs' own page), whose file is byte for byte the one the app installs. Compact and
  Light credit the same page.

## 0.10

### A third model: Anima Turbo v1.1

- **[Anima](https://huggingface.co/circlestone-labs/Anima) Turbo v1.1**, by CircleStone Labs and
  Comfy Org, joins Z-Image Turbo and Qwen-Image-2.1 in the app, without Developer Mode. A
  2-billion-parameter DiT (Cosmos-Predict2) that reads the prompt with Qwen3-0.6B and an adapter, and
  decodes with the Qwen-Image VAE. It makes illustration, painting and anime; **no photorealism**, by its
  publisher's choice. It knows artists by name, written with `@` in front (`@name`), as its model
  card says.
- **Its recipe**: 8 evaluations, CFG 1. Text-to-image, image-to-image, LoRAs (kohya and ComfyUI
  layouts), the rectangular formats, the four solvers in Developer Mode; no editing by instruction,
  and no Detail: on this model it makes mosaic blocks, or nothing visible.
- **Times**: on an M1 Pro with 16 GB, Standard, seed 42, 8 evaluations, 512² in 12.1 s, 1024² in
  43.7 s, 1024×1536 in 75.6 s.
- **Standard, Compact or Light**: 5.65 GB, 4.10 GB (the DiT in GGUF Q8_0) or 3.31 GB (the DiT in
  GGUF Q4_K_M), the DiTs as vanes430 publishes them. The text encoder stays the publisher's bf16 in
  all three: no 8-bit version of it is published.
- **Checked against the reference**, stage by stage: at 1024², the denoiser's output is 1.43·10⁻⁵
  from an fp64 computation of the same evaluation, the fp32 reference 1.47·10⁻⁵; the VAE closer
  to the fp64 than the reference; a LoRA against ComfyUI's own merge.
- **License**: CircleStone Labs Non-Commercial for the model, which derives from NVIDIA's
  Cosmos-Predict2 (NVIDIA Open Model License). **The images it makes may be used commercially**,
  as its publisher's model card says. The app shows the license before the first byte is downloaded.

### Qwen-Image-2.1: the recipe as published

- **Every step at full size.** Until now a 1024² generation took its first two steps at half size,
  a shortcut of ours; it is gone: the image is the Turbo's 8 published steps and nothing else. A
  1024² takes 134 s in the app (142 s from the command line) instead of ~120 s on an M1 Pro with 16 GB;
  1024×1536 232 s instead of 190 s. Edits are unchanged: they never took a step at half size.
- **ComfyUI's schedule, to compare** (Developer Mode): a « ComfyUI schedule » switch walks the noise
  levels ComfyUI uses for this model — the base model's, not the Turbo's — same seed, same steps.
  `silicontrol add --schedule comfyui` does the same.

### Fixed

- **An artist written `@name` in an Anima prompt** was refused by `silicontrol` with Named Subjects on
  (« No subject is called @name »): a model that reads no images now keeps its `@` as text, as the
  rack already did.
- **The text encoder's time** in the rack could read « ≈ 0.00 s » after a prompt read back from the
  render cache; only a prompt actually encoded now teaches it.
- **The time left** counted the running render as nothing until its first full step: the pace learned
  for that model and size now stands in.
- **The rack's seed** kept showing an old fixed seed after a `silicontrol` render with a random one; it
  now says Random.

## 0.9

### Report My Configuration now measures the chip

- **The diagnostic no longer renders a model.** In about 15 seconds, without any model, it maps what
  your Mac's chip computes: the matrix products of the models' blocks on the GPU, on the CPU's matrix
  unit and on both at once, fp16 and bf16 products, attention through the engine's three kernels,
  weight widening on the CPU and on the GPU — each result checked against the exact one, computed in
  double precision. The JSON report says what the chip computes, and whether it computes it right.
- **Fixed: with the Compact install, the diagnostic always said « differs from the reference ».** It
  compared the 8-bit weights' output with the full-precision reference: the weights differed, not the
  chip.

### The engine follows the map

- **Once you have run it, Siliconed uses what it measured, from the next launch:** the GPU and the
  CPU's matrix unit together on the large products, and at which share; the attention's heads split
  between the GPU and the CPU; the in-house attention kernel — each only if it is faster on this Mac,
  beyond the measurement's own noise, and only if it computed right. Until you run it, nothing
  changes. The sheet says what this Mac uses, and whether it applies now or at the next launch.
- After a macOS or Siliconed update, the map is set aside until you run it again.
- With a map, the same seed may give slightly different bits than without; the same Mac with the same
  map still gives the same bits.

## 0.8.2

### Fixed: Z-Image Turbo follows its recipe

- **8 steps are now 8 full-size evaluations, as the model card says.** Until 0.8.1, Siliconed ran
  7 (it followed an older diffusers schedule whose last step moved nothing), and at 1024² it
  computed the first two at half size. Both are gone; checked again against diffusers, stage by
  stage. The images come out cleaner, with finer detail.
- **Z-Image Turbo 1024² now takes 113–119 s** (118–120 s in Light), 512² ~34 s. The times
  published until 0.8.1 (82–85 s at 1024²) were for the shorter schedule; Siliconed's figures in the
  README are remeasured; its comparison with Draw Things now sets 8 evaluations against 8, Draw
  Things' figures being those measured on 13/10/2026.
- A Z-Image image made with 0.8.1 or earlier no longer comes out again to the bit; the same version
  on the same Mac still gives the same bits. Qwen-Image-2.1 is unchanged.

Timings are measured on a MacBook Pro M1 Pro with 16 GB.

## 0.8.1

### Fixed

- **The rack's last rows take the click again.** Just above the column's footer, the LoRA row,
  **+ LoRA** and **Import…** were drawn but did not respond until the column was scrolled. The
  footer now sits under the rack instead of over it.

### PNG metadata Civitai reads

- **Upload an image to Civitai and its settings show up.** Beside its own keys, each PNG now carries
  the text Civitai reads: the prompt with a `<lora:name:weight>` tag per LoRA, then the steps, the
  solver, the seed, the size and the model. Z-Image Turbo and Qwen-Image-2.1 renders credit their
  Civitai page with no manual entry.
- **LoRAs are credited too**, by the sha256 of the file as it was published, kept when you import
  it. A LoRA imported with an earlier version is named but not credited: import it again to credit it.

## 0.8

### Check for Updates

- **In the app menu, on demand only.** Siliconed asks GitHub for its latest release and compares it
  with the version running; if a newer one exists, it opens its page. Nothing is checked in the
  background, nothing is installed for you.

### Solvers (Developer Mode)

- **Euler ancestral, DPM++ 2M and UniPC**, besides Euler, on Z-Image Turbo and Qwen-Image-2.1: the
  same steps, the same σ, another walk between them. Each one is checked against diffusers'
  scheduler of the same name. Euler, the default, gives the same images as before, bit for bit.
- **Another image, the same time.** UniPC may recompose a Qwen-Image-2.1 picture; the ancestral
  solver may flatten a background. Edits keep image 1 under all four. Previews and sketches show
  each solver's own estimate. In the rack, the history caption, the PNG and `silicontrol add|grid
  --solver`.

### The numbers

- **Remeasured through the app**, every render queued by `silicontrol`: Z-Image Turbo 1024² in 82 to
  85 s, Qwen-Image-2.1 in 118 s, an edit with one image in 212 s, with two in 285 s; 0 swap-out.
  The README sets them next to the Draw Things engine on the same Mac. *Note (0.8.2): Z-Image ran 7 of the 8 evaluations of its recipe until 0.8.1; these Z-Image times are for 7 (see 0.8.2).*

Timings and memory are measured on a MacBook Pro M1 Pro with 16 GB.

## 0.7.1

0.7 was never released: 0.7.1 carries it, with the memory computation made exact in the cases it
missed.

### Memory: a fixed budget

- **Siliconed never takes more than its budget**: three quarters of the Mac's memory minus 2 GB
  (10 GB on a 16 GB Mac, 4 on an 8 GB one). It no longer reads how busy the Mac is: the same render
  on the same Mac always gets the same answer, whatever else is open. What the other apps hold is
  theirs.
- **Known before the first step.** Each render computes the memory it needs from its format, its
  prompt, its images and its LoRAs, takes the fastest plan that fits in the budget, and checks its
  own peak against that figure. Nothing switches plans halfway through any more. The figure is an
  upper bound: a long prompt on Qwen-Image-2.1, a batch of several images and the app's own memory
  are counted, and a render whose need cannot be computed is refused rather than started blind.
- **Greyed out ahead.** A format or a model beyond this Mac's budget is greyed out before you click,
  with the Mac it would need: “needs a Mac with 24 GB”.
- **Batches of Qwen-Image-2.1** no longer keep two copies of the model at once.
- Same images, bit for bit. Measured on the M1 Pro: Z-Image Turbo 1024² in 81 s, Qwen-Image-2.1
  1024² in 116 s; the largest edit (1024×1536, three images) peaks at 3.2 GB. *Note (0.8.2): Z-Image ran 7 of the 8 evaluations of its recipe until 0.8.1; these Z-Image times are for 7 (see 0.8.2).*
- The diagnostic report (schema 4) gives the budget, the need of each stage and the peak measured
  against it.

### Named Subjects (Experimental)

- **Nothing kept.** Subjects now live in memory only, like the session's images: nothing is written
  to disk, they go when the app quits. A subjects folder written by 0.6 is deleted at launch.
- **Images or Subjects.** The box above the prompt holds either the images of an edit (1 to 3) or
  your subjects (as many as you like, the prompt citing 3 at most) — one or the other, never both.
- **The name never reaches the model.** “@alice” becomes “the person in image 1”: a name read as
  content gave Alice in Wonderland.

Timings and memory are measured on a MacBook Pro M1 Pro with 16 GB.

## 0.6

### One Images box, above the prompt

- **The images come first.** On Qwen-Image-2.1, the box that holds the images to edit is now called
  Images and sits above the prompt: you lay the picture down, then say what to make of it. Its wires
  run past the prompt to the encoders that read them.
- **One budget, counted once.** With Named Subjects on (Experimental), the subjects live in the same
  box, under the edit's images: to the model, a subject you cite is one more image. The box's title
  counts both against the model's 3 images, and turns red past them.

### Memory

- **An idle Mac no longer refuses Qwen-Image-2.1 at 1024².** A plain generation was charged the
  memory of an edit with 3 images, and the safety margin ignored what macOS frees by compressing the
  other apps' memory. Now a render is charged the peak measured for its own number of images, and
  the margin counts that compression: on the Mac of the report, from 2.9 to 4.5 GB available for
  the same desktop. The no-swap rule is unchanged: everything is still measured at the worst case.

### Named Subjects (Experimental)

- **Compare with the subject.** “@ana as a western comic” keeps the picture's framing, as an edit
  does: the curtain under the image now compares the result with its image 1, whether it came from
  the Images box or from a subject (its frame, as it was sent).

### Under the hood

- Superseded kernels and settings removed (a scalar attention kernel, a half-precision mode): one
  version of each computation lives, and it is the one that is checked. Nothing changes in the
  images.

Timings and memory are measured on a MacBook Pro M1 Pro with 16 GB.

## 0.5

### Qwen-Image-2.1 is now Qwen's official Turbo

- **One model, from its publisher.** Qwen-Image-2.1 now runs Qwen's own distilled model,
  [Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo). Until 0.4, the app ran
  the base model under a third party's turbo LoRA (Viggle's); that LoRA and the base model are
  gone. Text-to-image, editing by instruction with 1 to 3 images and your own LoRAs work as before.
- **8 steps, on the schedule Qwen publishes.** The step count is fixed at 8: the 9-step mode is
  gone, and any other count is refused.
- **Compact and Light follow.** The Compact takes Comfy-Org's 8-bit (int8 convrot) Turbo, the
  Light unsloth's GGUF Q4_K_M Turbo; the Compact's 8-bit text encoder is unchanged. On disk:
  31.2 GB in Standard, 17.3 GB in Compact, 14.2 GB in Light (33.2, 19.4 and 16.2 GB before: the
  turbo LoRA is no longer stored).
- **An existing installation is recognized.** The Models window asks you to “Finish
  Installing…”: the old transformer and small files are removed, the Turbo's are downloaded in
  the same version (Standard, Compact or Light). The text encoder is kept: the Turbo's is the base
  model's, bit for bit, so it is not downloaded again. Fine-tunes of Qwen-Image-2.1 imported for
  the base model can no longer render and are removed.
- **Same license**: Qwen Research, non-commercial — the Turbo carries the base model's text.
- **Checked like the rest.** Every stage is compared with Qwen's reference implementation; the
  engine stays within 3.0·10⁻⁶ of the exact (fp64) computation.

### Faster than 0.4 for an image, slower for an edit

The Turbo takes 8 steps where the third-party LoRA took 6. We chose it anyway: it is the quality Qwen
publishes and checks, with no third-party piece in between that could change or disappear. Then we
made its steps cheaper, without lowering the precision: everything is still computed in fp32 and
checked against the exact computation.

- **The first two steps at half size.** At 1024² and above, a generation computes its first two
  steps at 512², then enlarges, as Z-Image already does. The image changes (the same seed gives
  another composition) but stays clean. Not under a LoRA or Detail, nor in an edit.
- **Attention shared between the GPU and the CPU.** In every layer, 13 of the 32 attention heads
  go to the CPU's matrix unit while the GPU computes the others.
- **Smaller savings**, with the same bits: the model is read in one stream, and normalizations are
  fused.

On the reference Mac:

- 1024² generation: ~120 s (~134 s in 0.4, ~160 s for the Turbo without these savings);
- an edit with 1 reference: ~210 s (~190 s), with 2 references ~285 s;
- the worst case, 1024×1536 with 3 references: ~515 s, 4.7 GB of memory at the peak, without any
  swap;
- the Light takes the same time as the Standard.

### Installation

- **Faster, and lighter on memory.** Preparing a downloaded model for the engine now takes seconds
  instead of minutes, and the next file downloads while the previous one is prepared. On a fast
  connection (85 MB/s), Qwen-Image-2.1 installs in ~6½ min in Standard (~8 min in 0.4), ~3½ min in
  Compact (~5½) and ~3 min in Light (~4); the network is now what sets the time. The installed
  model is the same, byte for byte.
- **No swap while installing.** The installation's memory peak goes from 4.6 GB to 2.2–2.5 GB, and
  freshly downloaded files no longer crowd the rest of your Mac out of memory: an installation
  that could swap on a busy Mac no longer does.

### Batches

- **Batches no longer swap.** Each image of a batch used to rebuild the model, and on a busy Mac a
  batch could end up swapping. Now every image is computed with the same model, then every image
  is decoded: the same images, with no swap, on Z-Image and Qwen-Image-2.1.
- **Including with your own LoRA.** On Qwen-Image-2.1, a batch with a LoRA of yours keeps the
  model, LoRA applied, from one image to the next.

Timings are measured on a MacBook Pro M1 Pro with 16 GB.

## 0.4

### Installation

- **Standard, Compact or Light.** Z-Image Turbo and Qwen-Image-2.1 install with the publisher's
  weights (Standard), with 8-bit weights published by third parties for the transformer and the text
  encoder (Compact: 11.7 GB instead of 20.4 for Z-Image, 19.4 GB instead of 33.2 for
  Qwen-Image-2.1), or with the transformer in 4 to 6 bits (GGUF Q4_K_M) and the Compact's text
  encoder (Light: 9.5 GB and 16.2 GB). The Light is preselected on a Mac with 8 GB of memory. The
  installed version keeps rendering until the new one is complete.
- **Downloads that resume.** Large files come in chunks, several at a time; a stalled chunk is asked
  again, and a cut download resumes where it stopped. Every file is still checked by sha256.
- **Signed and notarized.** The app and the disk image are signed with a Developer ID and notarized
  by Apple: they open with a double click.

### Editing

- **Edit from anywhere.** Right-click an image in the history and choose « Edit with
  Qwen-Image-2.1 », even when the rack is set to a model that does not edit: the rack switches to
  the editor and the image becomes image 1. A thumbnail dragged from the history onto the Edit box
  becomes a reference.
- **Original / edited curtain.** An edited image has a button that lays it over its original, with a
  line to drag across to see exactly what changed. Escape leaves it.

### Images

- **Detail at 1024².** « Detail: Normal · More · Most » now gives clean results at 1024² too, on
  Z-Image and Qwen-Image-2.1: its strength follows the size of the image, and it no longer leaves
  a grainy texture at high resolution.
- **Variations follow « Images ».** Subtle or strong variations of an image now make as many images
  as the rack's « Images » field (1 by default), instead of always four.

### Exploration

- **Pick the model in the panel.** The exploration panel has its own « Model » menu.
- **One status line.** The panel says where it is in a single line: « Sketch at n steps of N », or
  « Finished images, N steps » for Qwen-Image-2.1, which now explores with finished images (its
  early sketches were blurry).

### Imported models

- **8-bit stays 8-bit.** A fine-tune imported in 8 bits — fp8, int8, GGUF Q8_0 or int8 convrot — is
  kept in 8 bits on disk, about half the size of the 16-bit file (Z-Image: 6.2 GB instead of 12.3
  GB). The computation is still fp32, and every weight is checked bit for bit against the
  publisher's own conversion. The app imports `.gguf` files as well as `.safetensors`.
- **GGUF down to 4 bits.** A `.gguf` fine-tune in Q4_0 to Q6_K is kept as published, its blocks
  copied whole (Z-Image: about 5 GB in Q4_K_M instead of 12.3 GB). Files below 4 bits, and 4-bit
  files outside GGUF, are refused, with the reason.

### Speed and memory

- **Z-Image is faster at 1024²**: about 82 s instead of about 107 s, and 153 s instead of 168 s at
  1024×1536, still without any swap. *Note (0.8.2): Z-Image ran 7 of the 8 evaluations of its recipe until 0.8.1; these Z-Image times are for 7 (see 0.8.2).*
- **A memory check on the machine's real state**: before each render, the app reads the memory it can
  really have at that moment (not just the Mac's total) and refuses, with the figures, a render that
  would not fit — instead of letting it swap. If memory gets short during a render, the engine
  switches to a leaner plan: the same image, a little slower, rather than swapping.

### Languages

- **German, Spanish and Italian**, besides English and French: the app follows the system language.

Timings are measured on a MacBook Pro M1 Pro with 16 GB.
