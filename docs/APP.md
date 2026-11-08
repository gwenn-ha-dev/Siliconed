English · [Français](APP.fr.md)

# The Siliconed app

`Siliconed.app` turns a prompt into an image, edits images when you tell it what to change, and keeps a queue and a history of it all. No account, no cloud, no spinner that waits for a server somewhere. Just your Mac, working hard and telling you what it is doing. It follows the system language (English, French, German, Spanish, Italian, Japanese or Simplified Chinese).

![The window: the rack on the left (text, text encoder, DiT with Krea 2 Turbo, VAE decoder, each with its settings and time), a Krea 2 Turbo photo at 1248×832 in the middle — a laughing red-haired Irish woman wading out of the shore break, her surfboard under her arm, backlit by a low sun —, the Prompt Lab folded to one line beneath it, the history on the right](images/window.jpg)

*The window, with the Krea 2 Turbo photo from the [Models](#models) section: the rack shows the chain, each stage's settings and time; the Prompt Lab, folded, under the image; the history on the right.*

The engine behind it, with the measurements: [The Siliconed engine](ENGINE.md).

## Create

- **Five models.** Krea 2 Turbo, Z-Image Turbo, Qwen-Image-2.1 Turbo, Anima Turbo and MOSS-Image, each under its own license ([Models](#models)). All five read the prompt with a language model.
- **The chain, visible.** No hidden pipeline: the rack shows it top to bottom (text, encoders, denoiser, decoder), with typed wires and the active stage lit; each stage holds its own settings and shows its time. The text and the images sit on top, side by side, each above the encoder it feeds: read only, in outline, they are written and chosen in the Prompt Lab under the image. Every model renders in the 8 steps it is distilled for; a Steps line shows in the denoiser's box only when something else set another count (a PNG's settings reused, `silicontrol add --steps`), with a ✕ back to 8; with the Beta option Choose Steps on (Settings › Advanced › Beta), the line is always there, for a LoRA distilled to fewer steps. Scrolled down to the denoiser's settings, the chain stays at the top in small, lit the same way, each stage's time under it; a click brings its boxes back. Under the title, two small labels say what is stored and what computes: `weights int8` (or `Q8_0`, `Q4_K_M`, `bf16`…, read from what is installed) beside `computes fp32`. On the denoiser, the label says what this Mac's accelerators do: `computes fp32 · ANE` (part of its products on the Neural Engine, in fp16), `computes fp32 · GPU bf16` (part of it on the GPU's tensor units, in bf16), both, or a plain `computes fp32`. The **Accelerators** switch in the denoiser's box turns them on or off — off, the DiT is in fp32 throughout —, and is greyed, with its reason, on a Mac whose measure gives them nothing to do ([how](ENGINE.md#the-neural-engine)).
- **More detail** with Z-Image and Qwen-Image-2.1: “Detail: Normal · More · Most” adds fine texture, at no extra time.
- **Faster Rendering**: from 1024 px on each side, the first evaluations on a half-size grid, the rest at full size. 17 to 31 % less time on the Compact at 1024² (7 % for MOSS-Image; a MacBook Pro M1 Pro, app 0.18), still 8 evaluations, a slightly different image. It is a Beta option (Settings › Advanced › Beta), off by default, and it adds a switch to the denoiser's box, next to Accelerators: a choice per model, off for every version unless you turn it on, so the images stay those checked against the reference. With the option off, no render is a faster one; your choices come back when you turn it on again. The switch is greyed, with its reason, when it would do nothing for the size set (a side under 1024 px), a LoRA, Detail or an edit. The choice is saved with the image, so Render Again gives the same one ([how](ENGINE.md#faster-rendering)).
- **Variations**: cousins of an image, subtle or strong, with the same settings (the button under the image, or its context menu). **Render Again (Same Seed)**, ⌘R, is the other direction: the same image, to check that you are not dreaming. A seed is diffusers' seed on all five models (the same starting noise, bit for bit). How many you get at once is the rack's **Batch**.
- **Enlarge ×4** with macOS's own super-resolution (macOS 26): no prompt, nothing added, edges and texture sharpened; 1024² becomes 4096² in a second or two once macOS's model is warm. It takes images up to 1920 px a side, and an enlarged image is not enlarged again. The original stays in the history.
- **LoRAs** on all five models, in the denoiser's box: Add lists the model's LoRAs, then Import… for a file from Civitai. Each one is a card: a box to switch it off without losing it, its strength (a slider, or the figure typed), and its trigger words, when its file states them, as a button that puts them at the head of the prompt. Right-click: Replace With, Reset to 0.8, Remove. Three at most.
- **Reproducible to the bit.** Same prompt, seed, settings and version on the same Mac, same image, byte for byte. Two things can change the pixels: a new version of the app (0.12 did, by a few units in the last place, once its arithmetic rounded as the reference does), and a new measure of this Mac, which may split the work in other rows ([Measure This Mac](#measure-this-mac)).

![A Krea 2 Turbo render under way: the DiT's box lit in mint at step 5 of 8 with 19 seconds left, the image still blurred as it forms, the Prompt Lab unfolded with its text, Add to Queue at its foot, the job in the queue under the rack](images/live.jpg)

*A render under way, Preview Every Step on: the active stage is lit, its steps and the time left under it; the image forms in place.*

![The rack scrolled down to the LoRA: the chain pinned at its top in small (text encoder, DiT lit, VAE decoder), Anima Turbo with the inCase LoRA at 0.80 and its bracketed values, the DiT at step 2 of 8](images/chain.jpg)

*Scrolled down to the denoiser's settings, the chain stays at the top in small, lit the same way.*

Each side must be at least 512 and a multiple of 16; there is no fixed ceiling on the area: the memory budget decides. What a format costs is computed from the model's shapes before the first computation and judged against the Mac's memory budget ([how](ENGINE.md#memory-and-the-budget)): a Mac that cannot hold it is refused, and told which one can. The size picker lists 1536² and 2048² beside 1024², and greys what this Mac's budget refuses; any other size is typed. Formats past 1024×1536 are priced by the same formulas; Anima Turbo at 2048² was rendered on a 16 GB Mac, peaking at 5.4 GB. Below 512² the models are outside their domain. Qwen-Image-2.1 needs multiples of 32 (its VAE divides by 16), and Anima at most 2048 px on a side (the positions its reference computes); both are refused before any computation, and so is any side past 8192 px.

**Size in Two Scales** (Settings › Advanced › Beta, off by default) splits the rack's Size menu in two: a proportion (1:1, 4:5, 3:4, 2:3, 9:16 and, turned, 5:4, 4:3, 3:2, 16:9), then a resolution, 1K (about 1 megapixel) or 2K (about 4); the app works out the pixels, always multiples of 32 so that every model takes them. The pencil, the memory check and each model's limits are unchanged, and the default menu stays the default.

| Proportion | 1K | 2K |
|---|---|---|
| 1:1 | 1024 × 1024 | 2048 × 2048 |
| 4:5 | 896 × 1120 | 1792 × 2240 |
| 3:4 | 864 × 1152 | 1728 × 2304 |
| 2:3 | 832 × 1248 | 1664 × 2496 |
| 9:16 | 768 × 1344 ≈ | 1440 × 2560 |

The landscapes are the same sizes turned (3:2 is 1248 × 832 at 1K). The ≈: at 1K an exact 9:16 is out of reach: an exact 9:16 in multiples of 32 comes only in steps of 288 × 512 (576 × 1024, 864 × 1536), so 1K keeps 768 × 1344, which is really a 4:7 (1.6 % off) and is shown “≈ 9:16”; the same goes for 16:9 lying down. At 2K, 1440 × 2560 is an exact 9:16. Anima stops at 2048 px on a side, so its 2K is the square alone and the other 2K sizes are greyed with that reason. `silicontrol add` and `grid` take the same sizes as `3:4@2K`, Beta option on or off ([Remote control](REMOTE-CONTROL.md)).

## Edit

- **Editing by instruction** with Qwen-Image-2.1 Turbo: give it 1 to 3 images and a sentence (“replace the cloudy sky with a blue sky”); its encoder sees the images. In the app an edit is limited to an area of 1024².
- **Three ways to compare** two images, an edit or an enlargement and its original: pick Curtain, Switch or Side by Side in the canvas toolbar.
  - **Curtain**: drag a line across the images to see what changed. Zoom in toward the pointer to look at a detail: pinch, or a mouse's wheel (a horizontal wheel still moves the image).
  - **Switch**: the same two images one at a time, at the same place and zoom; click the image, or press ← for the first and → for the second.
- **Named subjects**: name a person, an object or a style from a photo, then cite it as `@name` in the Prompt Lab's text (see Prepare, below); the render receives the prompt translated and the images numbered.
- **A history like Photos**, in a column on the right (the slider at its foot sets the thumbnails' size): arrows, ⌫, the ✕ of a hovered thumbnail and ⌘Z, multiple selection, export, drag and drop, Share. From a terminal: `silicontrol remove`, `undo`, `crop`, `again`, `reuse`, `save` (PNG, JPEG, HEIC).

![An edit by Qwen-Image-2.1 in the window: the Krea 2 photo of the surfer as image 1, “Image to Edit”, in the Prompt Lab, the instruction “Convert this photo to a western comic book style with cel shading”, the cel-shaded result in the canvas, the image ports of the rack numbered](images/edit.jpg)

*Qwen-Image-2.1 Turbo, Standard: “Convert this photo to a western comic book style with cel shading”, the Krea 2 photo as image 1, 1248×832, seed 42.*

![An edit by Qwen-Image-2.1, under the sliding curtain: the Krea 2 surfer wading out of the surf in her wetsuit, then in a white linen blouse, her face, hair, surfboard and the sea unchanged](images/outfit.gif)

*An edit that changes only what it is told: “Replace her wetsuit with a white linen blouse. Keep her face, hair, surfboard, pose and the beach unchanged.”, the Krea 2 photo as image 1, seed 42.*

![Subjects cited by name: the surfer as image 1, three subjects named from another photo (a coat, a dog, a woman), the text “she wears @coat over her wetsuit, and @dog runs beside her through the shallow water” with its two names as chips, the result in the canvas: the surfer in the camel coat, a border collie running beside her in the water](images/mentions.jpg)

*Subjects: a camel coat, a border collie and a woman, named from one photo (a Z-Image Turbo render on a quay in Brittany), then cited by name. Qwen-Image-2.1 Turbo, Standard, the surfer as image 1, seed 42.*

![A new composition from two subjects alone, no image edited: “@maelle walks @dog along a wild beach at sunset”, a woman with short dark curly hair in a long camel coat walking a border collie on a leash along a beach at sunset](images/e3-new.jpg)

*The same subjects without an image to edit: a new picture, 832×1248, seed 42.*

## Prepare (and let a model do the typing)

The **Prompt Lab** is the bar under the image, where the next render's text and images are prepared. Folded, it is one line to read: the images that go, as small thumbnails, the text, and how many images the render will not read; nothing on it starts a render or a rewrite, so that a click to take the text back only unfolds it. **Generate** is ⌘↩ from anywhere in the window; when something stops it, ⌘↩ unfolds the bar, where the reason is said. The window opens with it unfolded (folded while no model is installed, the models to download in sight); click the text, or press ⌥⌘L (View › Show Prompt Lab), and it unfolds as a glass plate over the image, its edges fading, the image keeping its size; a click on the image, or Esc, folds it, the text kept. The button beside its chevron detaches the unfolded plate into a window of its own, resizable, its text growing with it; closing that window folds the lab back to its line. Its foot holds what varies and Generate; what stops Generate is said there, with the largest size this Mac renders when the memory is short. What the render costs (size, memory, time) is the chain's, under it in the rack. The queue, the measure of this Mac and what went wrong stay under the rack. Its text is the rack's prompt and its images are the rack's inputs: nothing to copy or carry, and the rack shows what will go.

- **Images**: drop them on the bar or the canvas, paste one (⌘V), open them (+), or send one from the history (its menu: Send to the Prompt Lab, or Edit with Qwen-Image-2.1, which sends it as the image to edit). Each image has a role, in its menu. With Qwen-Image-2.1 an image takes the next place: the first is the **Image to Edit** (image 1, which gives the format), the next ones are **References**, three images at most with the subjects the text cites; full, a newcomer comes in with **No Role** and says so. With Z-Image, Anima, Krea 2 and MOSS-Image an image comes in with no role and two buttons: **Describe**, and **Start From This Image** (img2img). The render then starts from it instead of noise and keeps its composition, and its **strength** (0 to 100 %; to begin with 65 % on Krea 2, 75 % on Z-Image and Anima, 80 % on MOSS-Image), set on its tile and read in the rack, says how far it moves away; the size takes the format nearest its proportions (any other is still yours: the centre is kept). The PNG writes the strength, not the image.
- **A change of model never gives a role, nor turns one into another**: a reference keeps its role under Z-Image, without effect (its tile says so), and works again back on Qwen-Image-2.1; the image Qwen edits never becomes the start of an img2img. Beside Generate, “1 image of the Prompt Lab not used” counts what the next render will not read. Right-click an image: Crop…, Replace…, Name…, Reuse These Settings (for a PNG Siliconed wrote: dropping one no longer changes the rack by itself), Remove.
- **Subjects**: Name… gives an image a name, a kind and a frame, and the text cites it as `@name` (completed as you type). A subject goes to the render only when the text cites it, after the images the roles send, in the order the text names them. The text keeps its `@names`: they are translated into “image N” when the render is queued, and the PNG keeps both. Subjects live in memory only (nothing is written to disk, as in the history). `silicontrol subjects` acts on the same ones.
- **What is greyed does not go.** Under a model that reads no image, a subject's `@name` is greyed and left out of what the model reads (any other `@` stays text: Anima's artists); so is a reference to an image the render does not carry (`<image3>`, `image 3` or `Picture 3` with two images). The hover says why; the text keeps them for a model that reads them.
- **Text**: one text, always yours to edit; the ✕ in its corner clears it (⌘Z brings it back). **Rewrite** turns it into a text written for the rack's model, named in the bar's title, and keeps your `@names` (one it loses is said, never dropped in silence); the additions and removals are marked, and a click in the text puts the marks away. **Describe**, on an image, writes what it shows into the text: it fills an empty text, otherwise it goes where the caret is or replaces the selection (at the end when the text is not on screen). Both write into the text as the writer goes. ⌘Z undoes any gesture of the model. A `@name` cannot sit inside a `{a|b}` group. `silicontrol rewrite` and `silicontrol describe` are the same two gestures for a terminal (same writer, same instructions, same protection of the `@names`, same waits); the text comes back on the socket; `silicontrol lab rewrite` and `lab describe N` write it into the bar's text, marks included, as the buttons do. `silicontrol lab` reads and changes the bar: its images, their roles, folded or not.
- **Following another Mac** (see below): Rewrite and Describe are off and say so; the images, the roles and the subjects fill the rack you see, which sends its references to the Mac that renders.
- **Rewrite and Describe are the only gestures that need the writer** (below): during a render, or without the writer, only they wait, and the rest of the bar works.
- **The writer is a model you already have**: Qwen-Image-2.1's text encoder, Qwen3-VL-8B, made to write. Only its output layer is missing from the installed model, and “Install the Writer” (or `silicontrol install writer`) downloads it from the same repository (1.25 GB, by byte range; no license of its own, it is Qwen-Image-2.1's). It needs Qwen-Image-2.1 installed, and reads its layers in a plan chosen within the memory budget before the first word: on a 16 GB Mac the Compact's all from memory; on an 8 GB Mac, or with the Standard on 16 GB, part of them from the disk at each token, more slowly. The window tells you what is missing.
- **Our instructions, one per model**, drawn from what each publisher says of its prompts. Every element of your prompt is kept, quoted text word for word; nothing is removed, softened or moralized.
- **Never during a render**: an image that starts stops the writer, and Rewrite and Describe wait. On an M1 Pro with the Compact, the first word comes after about 10 s for Rewrite and 25 s for Describe, which reads the image first, then about 12 tokens a second, on the GPU. Its greedy output matches transformers' token for token ([VERIFY](VERIFY.md)).
- **Incognito**: texts, images and subjects stay in memory, with no file and no history; the writer gives its weights back after every gesture, so the Mac's memory is free between two.

![The Prompt Lab unfolded as a glass plate over a Z-Image Turbo photo, in the light appearance: a young red-haired Irish woman in a cream Aran sweater riding a green bicycle down a lane between dry-stone walls and red fuchsia hedges, the prompt in the bar, Generate at its foot](images/window-light.jpg)

*The Prompt Lab unfolded over the image, in the light appearance: Z-Image Turbo, Standard, 832×1248, seed 7001.*

![Rewrite for Z-Image Turbo: a short idea turned into a long description of the cyclist photo, the additions marked in peach, the title “Write for Z-Image Turbo”](images/rewrite.jpg)

*Rewrite: a one-line idea written out for Z-Image Turbo, what was added marked in the text.*

![Describe at work: a photo of a woman hugging a border collie on a quay as the only image of the Prompt Lab, with Start From This Image and Describe, the writer filling the empty text as it goes, “Writing…” under it](images/describe.jpg)

*Describe: the writer reads the image and writes what it shows into the text as it goes.*

## Series (bracket, then choose)

Anything that varies makes a series, and everything that can vary carries the same yellow ± at its right: outlined it offers, filled it varies, its values written under it with the arrow of their axis (↔ columns, ↕ rows). A `{red|blue}` group in the prompt is one, drawn as a yellow chip with a dot between its words: a click on it opens its bubble, and the ± at the right of the text, under its ✕ — or Vary This Word… at the head of the right-click menu — turns the selected word, or the word at the caret, into one. The Batch is another: consecutive seeds from the rack's, its ± on the Seed line. The knobs are the rest, by their ± or right-click › Bracket…: a LoRA's strength, LoRAs compared (the ± of the LoRA title, or Add › Compare LoRAs…, where the model has two LoRAs or more: one added per image, and the image without), the starting image's strength, Detail and, with More Solvers on, the solver. The bubble offers three values around the knob's (a LoRA ±0.2, the starting image ±15 points), a step and a count to make them again, and the axis they take, columns or rows; placed where another axis is, it replaces it, and says which before you click. A series is two values or more. Two axes at most, the Batch one of them: beside Generate, a line per axis with its ✕, and ⇄ to swap them; a third thing that varies keeps Generate off and says why. Generate then queues one finished image per cell, the very image it would render alone, into the history one by one; the canvas lays them out as a grid while they render and until you choose one, and Add Sheet to History draws the sheet when you want it. Models, steps and each image of an edit vary from a terminal: `silicontrol grid` (its axes become the window's series where the window has them) and `silicontrol sheet g2`.

![A series filling in the canvas: Anima Turbo with the inCase LoRA, a laughing red-haired Irish woman before sea cliffs, the wetsuit's colour across (teal, coral red, mustard yellow) and inCase's strength down (0, 0.5, 1), one cell after another, until the nine are there](images/series.gif)

*A series with Anima Turbo and the inCase LoRA: the wetsuit's colour written as a group in the prompt (`{teal|coral red|mustard yellow}`) across, inCase bracketed at 0, 0.5 and 1 down; 832×1248, seed 4001, nine finished images.*

![The series done, the Prompt Lab unfolded: the yellow chip “teal · coral red · mustard yellow” in the text, the inCase LoRA's ± filled with its values written under it, a line per axis beside “Generate 9 Images”, the nine images on the canvas behind](images/series-set.jpg)

*What varies, where it varies: the group in the text, inCase's ± in the rack, one line per axis beside Generate.*

![The bracket bubble open on inCase's strength: the values 0.00, 0.50 and 1.00 with a ✕ each and a +, a step of 0.20 and a count of 3, the axis set to Rows, Remove, Cancel and Bracket](images/bracket.jpg)

*The bubble of a knob: its values, a step and a count to make them again, and the axis they take.*

## Model management

- **Nothing is preinstalled.** You pick a model; before the first byte is downloaded the app shows its license and the disk space it takes, and asks you to accept. Every file is pinned to a revision and checked by sha256.
- **Standard, Compact or Light.** Each model installs with the publisher's weights (Standard), with 8-bit weights published by third parties (Compact, the default and the one we recommend: 19.5 GB instead of 34.4 for Krea 2, 11.7 GB instead of 20.4 for Z-Image, 17.3 GB instead of 31.2 for Qwen-Image-2.1, 4.10 GB instead of 5.65 for Anima, 17.9 GB instead of 27.5 for MOSS-Image; a slightly different image), or with the transformer in 4 to 8 bits, quantized by Siliconed itself (SQ, published on Hugging Face; Light: 12.9 GB, 9.5 GB, 14.4 GB and 3.15 GB, which reads the least from disk at each step; another set of weights, an image that may compose differently). For Krea 2, Z-Image, Qwen-Image-2.1 and MOSS-Image the Compact's text encoder is 8-bit too, and the Light uses it; Anima's text encoder stays the publisher's bf16 in all three, since no 8-bit version of it is published. MOSS-Image has no Light. The Light is preselected on a Mac with 8 GB of memory, the Compact elsewhere; you choose before installing. One version at a time: switching replaces the other, which keeps rendering until the new one is complete, then frees its space; if the disk cannot hold both, the app says so before you accept.
- **Civitai and Hugging Face imports.** LoRAs (Civitai, Comfy, kohya, diffusers layouts) and a fine-tune's DiT (`.safetensors`, or `.gguf` from Q4 to Q8_0), by drag and drop or ⌘O; a Civitai checkpoint of a known family imports as is and borrows its family's encoder and VAE. Dropped before its family is installed, it opens the family's installation — license, encoder and VAE, never the family's own DiT — and is imported once that is done. A quantized file stays as published on disk, never re-quantized (Z-Image: 6.2 GB in 8 bits, about 5 GB in GGUF Q4_K_M, instead of 12.3 GB; each step a little slower, the computation stays fp32); below 4 bits, the file is refused. An imported LoRA takes its file's name; the trainer's name and trigger words show in a tooltip.
- **Statistics.** A floating palette (⌥⌘I, Window › Statistics, or the toolbar button): where the render stands, what each stage cost, what remains for it and for the queue, and what the last render weighed. Nothing is estimated from a constant: a model or format never rendered here has no estimate, and the palette says so.

### Measure This Mac

Before its first render, and again after an update or a new macOS, Siliconed times what each engine of your Mac computes. The first render starts it by itself and waits for it; better with the Mac at rest — no export, compilation or game meanwhile —, because what another app takes from the GPU, the CPU or the Neural Engine is counted as missing from your Mac. It takes about a minute, five at most, needs no model, and gives a map every time: nothing is ever refused, and how sure the measure is of each figure is written beside it. You can open it whenever you like: Help › Measure This Mac…, the Chain title and the Mac's icon at the head of the rack (greyed until this Mac is measured), or Settings › Macs › This Mac. It is not in the Models window: installing a model and measuring a Mac are two different jobs.

The sheet shows, with its columns titled: each route's speed, its share of the GPU's peak, whether its result matched the exact one (Exact ✓), what each engine keeps of its speed when they work together, the memory the measure held and how long it took. What it found is said in plain sentences, each with its effect on the plan. Sharing is an option, never the default: Share… ▸ Open a GitHub Issue, or Copy JSON.

A measure you ask for applies to the **next** render, never to one under way, so from then on one seed can give a slightly different image from before the measure. When a render ran more than 10 % off what the measure predicted, the sidebar says so, once, beside the button that measures again.

Models live in `~/Library/Application Support/Siliconed` (override with `SILICONED_ROOT`). To free the internal disk, Settings › Move Library… moves them to another one — an external disk, a network share: every file is copied, checked, then removed from the Mac. Siliconed then copies to the Mac only the models you render, for the session, and removes them when it quits; the licenses you accepted and this Mac's measurements stay on the Mac. Unplugged, the disk is flagged: nothing renders until it is back. From a terminal: `silicontrol library move DIR` (without `--yes`, it says what would be copied and the room there is).

![The Models window: the five publishers' models installed in Standard, each with its license and its size on disk (Z-Image Turbo 20.36 GB, Qwen-Image-2.1 Turbo 32.46 GB, Anima Turbo 5.65 GB, Krea 2 Turbo 55.97 GB, MOSS-Image 27.5 GB, marked beta), MOSS-Image's versions open with the Compact chosen and the room it would free, the disk used and free at the bottom](images/models.jpg)

## Models

Only recent models whose text encoder is a language model — no Stable Diffusion, SDXL or Flux.1.

| Model | Publisher | What it does | License |
|---|---|---|---|
| [Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) | Krea | text-to-image (aesthetic images, from photography to painting), LoRA | Krea 2 Community: commercial use under $1M annual revenue, a content filter required when deployed |
| [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Tongyi-MAI | text-to-image, LoRA | Apache 2.0 |
| [Qwen-Image-2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | Qwen | text-to-image, editing by instruction with 1 to 3 images, LoRA | Qwen Research (non-commercial) |
| [Anima Turbo v1.1](https://huggingface.co/circlestone-labs/Anima) | CircleStone Labs × Comfy Org | text-to-image (illustration, painting, anime; no photorealism, by its publisher's choice), LoRA | CircleStone Labs Non-Commercial; its images may be used commercially |
| [MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | OpenMOSS, under lodestones' 8-step LoRA | text-to-image, LoRA; beta | Apache 2.0 |

### Krea 2 Turbo, the largest of the five

[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo), by Krea, is the fourth model, added in 0.12: a 12.9-billion-parameter DiT, the largest of the five, for aesthetic images from photography to painting. It reads the prompt with Qwen3-VL-4B and runs its whole 8-evaluation recipe in fp32 on the publisher's bf16 weights, with [Faster Rendering](ENGINE.md#faster-rendering) as a Beta option (times and deviations: [Numbers](ENGINE.md#numbers)), its weights streamed from disk within the memory budget. It installs as Standard, Compact or Light (sizes: [Model management](#model-management)), from open copies proven to be identical to Krea's files (see below), under the Krea 2 Community license (table above).

![A Krea 2 Turbo photo at 1248×832, from the water at wave height: a young Irish woman with long wavy auburn hair and freckles, in a short-sleeved shorty wetsuit, wading out of the shore break laughing with her surfboard under her arm, the low sun behind her, rim light in her wet hair, backlit spray, the horizon slightly tilted](images/krea2.jpg)

*Krea 2 Turbo, Standard, 1248×832, 8 steps, seed 3303: “Surf photograph shot from the water at wave height: a young Irish woman with long wavy auburn red hair, pale freckled skin and green eyes, in a short-sleeved shorty wetsuit, wading out of the shore break and laughing, her surfboard under her arm, on a rare sunny evening in the west of Ireland. Hard low sun behind her, rim light in her wet hair, the spray backlit like sparks, deep shadows, horizon slightly tilted, one drop of water on the lens in a corner, Kodak Portra 400 grain, 35 mm.”*

Krea 2 Turbo downloads nothing from [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo): that repository asks for a Hugging Face account, and the app has no sign-in. Each of its files comes from an open copy proven identical to Krea's: the DiT and the Qwen3-VL-4B text encoder in bf16 from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) (the DiT the very same file, the encoder equal tensor for tensor), the VAE from [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) (Krea's fp32 VAE is this bf16 one widened, value for value), the tokenizer from [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (the same file); Krea's own configuration files are embedded as published. Its Compact takes the DiT and the text encoder in int8 convrot from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2); its Light, our 4-bit file of [gwenn-ha-dev/Krea-2-Turbo-Siliconed-SQ4](https://huggingface.co/gwenn-ha-dev/Krea-2-Turbo-Siliconed-SQ4) with the Compact's text encoder.

### MOSS-Image, distilled by its community

[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), by the OpenMOSS team, is the fifth model, added in 0.16: a single-stream DiT of 5.1 billion parameters that reads the prompt with Qwen3.5-9B — a hybrid language model, three layers in four of linear attention (Gated DeltaNet) — and decodes with FLUX.2's VAE. Its publisher released no fast version; the community did: Siliconed runs MOSS-Image-SFT under [lodestones' DMD LoRA](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) at strength 1, 8 evaluations, no CFG, the LoRA kept beside the weights in fp32 (merged into the bf16 weights, its small deltas would be lost to rounding). Both are checked against the reference, the LoRA included ([VERIFY.md](VERIFY.md)). Text to image, and your LoRAs on top of the 8-step one — ComfyUI's and ai-toolkit's fused `qkv` is split exactly, and LoRAs trained on MOSS-Image Base apply too, as the community uses them; no Detail, no edit. [Faster Rendering](ENGINE.md#faster-rendering) takes a single half-size evaluation: from the second, MOSS-Image loses fine detail (skin, book spines) where the other models keep it. It is marked **beta** in the app, because its 8-step recipe is the community's, not the publisher's; it leaves beta when OpenMOSS releases its own fast version. It installs as Standard or Compact (sizes: [Model management](#model-management)), under Apache 2.0, the LoRA included.

![MOSS-Image starting from the Krea 2 photo: the rack's Starting Image port reads it at 80 %, the Prompt Lab holds it as “Starting Image” with its strength, and the canvas shows the same shot with a laughing grey-haired Irish grandmother in her seventies in the shorty wetsuit, her surfboard under her arm](images/start.jpg)

*MOSS-Image, Standard, starting from the Krea 2 photo (Start From This Image, strength 80 %), 1248×832, 8 steps, seed 9001: the same prompt, the young woman replaced by “a grey-haired Irish grandmother in her seventies”. The composition holds; the person changes.*

The Standard takes the DiT, the Qwen3.5-9B text encoder (its language model alone, the vision tower left out), the VAE and the tokenizer from [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), and the LoRA from [lodestones/MOSS-Image-SFT-8-steps-dmd](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd). Its Compact takes the DiT and the text encoder in int8 convrot from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image), with the same LoRA, VAE and tokenizer. No Light yet: a 4-bit DiT is published but not yet checked; on a Mac with 8 GB the Compact is preselected, and its peak stays within that Mac's budget.

### Where the files come from

The other Compact versions use 8-bit weights published by third parties: [Comfy-Org/Z-Image-Turbo](https://huggingface.co/Comfy-Org/Z-Image-Turbo), [Disty0/Z-Image-Turbo-SDNQ-int8](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-int8), [unsloth/Qwen-Image-2.1-FP8](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8), [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1); the Light versions, our own 4-bit files, [gwenn-ha-dev/Z-Image-Turbo-Siliconed-SQ4](https://huggingface.co/gwenn-ha-dev/Z-Image-Turbo-Siliconed-SQ4) and [gwenn-ha-dev/Qwen-Image-2.1-Turbo-Siliconed-SQ4](https://huggingface.co/gwenn-ha-dev/Qwen-Image-2.1-Turbo-Siliconed-SQ4), with the Compact's text encoder. Anima's Compact takes the DiT in int8 convrot from [Bedovyy/Anima-INT8](https://huggingface.co/Bedovyy/Anima-INT8), its Light is our 4-bit file [gwenn-ha-dev/Anima-Turbo-Siliconed-SQ4](https://huggingface.co/gwenn-ha-dev/Anima-Turbo-Siliconed-SQ4); both keep the publisher's text encoder in bf16. The revisions and sha256 sums are in `Sources/Siliconed/Forge/Installation.swift`.

## Render on another Mac

- **One Mac renders for another on the same network.** The MacBook in the living room computes on the Mac mini in the next room: **Render › Compute On** moves the whole window there — its queue, progress, preview, history, models, library, licenses and measure are the other Mac's — while the form, your selection, exports and the clipboard stay on yours. An “On:” menu beside Chain says where you compute, and the rack wears a blue edge while connected; one menu brings you back. Never two views at once. From a terminal: `silicontrol macs` and `silicontrol compute NAME|this`.
- **Pair once.** On the Mac that lends itself: Settings › Macs › Lend This Mac, then Pair a Mac…. On yours: Settings › Macs › Pair With a Mac…, and choose it. Both screens show six digits; accept on the lending Mac only if they are the same. Both Macs must run the same version of Siliconed. Nothing connects before a pairing is accepted, and Forget undoes one on either side.
- **Encrypted, nothing in clear.** TLS 1.2 with a key per paired Mac, kept in the Keychain. macOS asks once for the local network, the first time you lend this Mac or pair it; a Mac that never does either is never asked.
- **Everything works there**: renders, edits with your references, variations, grids and series, enlarging and cropping, undo; the Prompt Lab fills the rack you see (images, subjects and Carry; Rewrite and Describe stay off); installing a model or importing a LoRA on the lending Mac, measuring it. Removing a model waits for a click on that Mac. Each Mac sees only its own jobs and images, and the lending Mac stays within its own memory budget. `silicontrol` follows Compute On ([REMOTE-CONTROL.md](REMOTE-CONTROL.md)).

## Settings (few, on purpose)

Siliconed › Settings… (⌘,), four tabs chosen by their icons; each setting says in one line what it does or where it stands, and the whole sentence on hover.

- **General**: where the models live, what they weigh and what their disk has left, with Show in Finder, Move Library…, Open Library… and Move Back to This Mac. From a terminal: `silicontrol import`, `uninstall` (with `--yes`) and `library`. Then **Save Every Image To…**, off by default: the history stays in memory, as everywhere else in the app. Turned on, it asks for a folder, and each new image (a render, a variation, an edit, a crop, an enlargement, a grid's sheet) is also saved there as a PNG with its prompt, seed and model, under the name Save… gives it, never over another file. A failure to write is said like a failed Save…. From a terminal: `silicontrol set save-images FOLDER|off`.
- **Macs**: this Mac, its chip and memory, and its measure (Measure This Mac…); then the other Macs: lend this one, pair with another, each paired Mac with its state ([Render on another Mac](#render-on-another-mac)).
- **Prompt Lab**: the instructions the writer is sent before each Rewrite and each Describe, one per model and gesture, ten in all, in English as they are sent. Choose the model and the gesture, and rewrite the text as you like: yours is sent in place of Siliconed's for that model and gesture only; Restore brings Siliconed's back, and what you leave untouched follows Siliconed's from version to version. When the model the Prompt Lab writes for has one of yours, its bar says so beside its name (Your Instruction), and a click opens it here. From a terminal: `silicontrol instructions`.
- **Advanced**, in this order: Command Line (installs `silicontrol`, below); and **Beta**, off by default and liable to change: More Solvers (adds the solvers Euler a, DPM++ 2M and UniPC to the denoiser's box and to the series; Euler, the default, is the one the models were distilled for), Choose Steps (adds the step count to the denoiser's box, for a LoRA distilled to fewer steps; off, every render takes the model's own count), Faster Rendering (adds its switch to the denoiser's box), Size in Two Scales (the rack's Size becomes a proportion and a resolution, below), Negative LoRA Strengths (strengths from −1.5 to 1.5, for “slider” LoRAs; an ordinary LoRA is damaged by a negative strength). `silicontrol set beta` flips them like the switches.

While you compute on another Mac, the measure is that Mac's, and the window says so.

![Settings, General: the models On This Mac with what they weigh and the disk's free space, Show in Finder, Move Library… and Open Library…, then Save Every Image To…, off](images/settings.jpg)

![Settings, Prompt Lab: Krea 2's Rewrite instruction replaced by one of yours, “Yours: sent instead of Siliconed's.”, and Restore](images/settings-lab.jpg)

*Settings › Prompt Lab: an instruction of your own for Krea 2's Rewrite (a photographer's brief), sent in place of Siliconed's; Restore brings ours back.*

## Keyboard

Lost? The Help menu holds Siliconed Help (this guide), Remote Control Help (the `silicontrol` guide), Report a Problem… and Measure This Mac…; Window › Show Siliconed brings the window back if you closed it. The menus hold the full list of keys; these are the ones worth knowing. ⌘ is Command, ⌥ Option, ⌃ Control.

| Keys | Does |
|---|---|
| ⌘↩ | Generate |
| ⌘R | Render Again (Same Seed) |
| ⌘. · ⌥⌘. | Stop Render · Clear Queue and Stop |
| ⌘S · ⇧⌘E | Save Image… · Export Images… |
| ⇧⌘C | Copy Image |
| ⌘O · ⇧⌘M | Import a Model or LoRA… · Models… |
| ⌘+ · ⌘− · ⌘0 | Zoom In · Zoom Out · Zoom to Fit |
| ⌃⌘F | Full Screen (Esc, Space or a click leaves; ← → move between images) |
| ⌥⌘← · ⌥⌘→ | Previous · Next Image, from anywhere |
| ⌥⌘A | Select All Images |
| ⌥⌘I · ⌥⌘L | Statistics · Show, Hide Prompt Lab |
| ⌘, · ⌘W | Settings · Close Window |

And without ⌘, once nothing is being typed in a field and the history is not empty: **← →** walk the history (with ⇧ they extend the selection; with the curtain shown they move its line, with Switch they show A or B), **Home** and **End** jump to the newest and the oldest image, **Space** opens full screen, **⌫** removes the selected images from the history, **C** opens the crop (Return crops, Esc leaves), **Esc** leaves a comparison or clears a multiple selection, ⌘A selects all, ⌘V pastes an image into the Prompt Lab. Esc also leaves the field you are typing in; in the Prompt Lab's, it folds the bar too, and so it does when there is nothing else to leave. A slider, a stepper or a button you reached with Tab keeps its own keys.

## Private by construction (nobody's looking)

Rendering happens on your Mac. The only network traffic is the model download you accepted, the one request to GitHub for the latest version number when you choose Siliconed › Check for Updates…, and, if you turn it on (Settings › Macs), the encrypted link to another Mac of your network; the diagnostic copies its report and opens a GitHub issue form that you choose to submit or not. The render cache lasts one session: it is emptied at launch and at quit. Before each render the engine checks the model, its license, the format and the memory it needs, and refuses up front a render beyond this Mac's budget. The app censors no prompt and adds no content filter; each model's license still applies, and Krea 2 Community's asks whoever deploys the model for one.

## For scripts and agents: silicontrol

Drives the running app from a terminal. It ships in `Siliconed.app/Contents/Helpers/silicontrol`; Settings › Advanced › Command Line links it into `/usr/local/bin`. What you add appears in the app's window, and there is one queue, never two renders at once. Edit › Copy as Terminal Command puts on the pasteboard the `silicontrol` command for the form as it stands: handy for turning a render you like into a script.

```
silicontrol help                 the full reference (always start here)
silicontrol models
silicontrol add --wait --out out "a woman posing in a library"
silicontrol status
```

It does what the window does, by the same functions as its buttons: it flips a model's Faster Rendering or Accelerators (`silicontrol set faster-rendering on|off --model krea2`, `set accelerators …`) and the Beta options (`silicontrol set beta`), rewrites and describes with the Prompt Lab's writer (`silicontrol rewrite`, `describe`, `install writer`), and repeats a timed series of renders on your Mac (`silicontrol measure`); it removes and undoes, crops, renders an image again at its seed (even from a PNG of another day), adds a series' sheet to the history (`silicontrol remove`, `undo`, `crop`, `again`, `sheet`), imports and uninstalls models and LoRAs, switches Compute On to a paired Mac and moves the library (`silicontrol import`, `uninstall`, `compute`, `library`; what removes or moves needs `--yes`); it grids, varies, enlarges, follows a render step by step, and more: `silicontrol help all` lists every command. Output is JSON lines; the last one carries `"ok"`. How it works inside: [REMOTE-CONTROL.md](REMOTE-CONTROL.md).
