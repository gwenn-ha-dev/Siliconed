English · [Français](API.fr.md)

# Siliconed's API

For whoever writes a macOS app (SwiftUI, “in the manner of Draw Things”) on top of the `Siliconed`
library. State as of 2026-10-09. A complete SwiftUI app, **Siliconed.app**, is in [`Sources/SiliconedApp/`](../Sources/SiliconedApp/) (`App.swift`: the app, its
menus and its window; `AppState.swift`: the state and every call to the library;
`MainView.swift`: the view; `Statistics.swift`: the statistics palette; `ModelsSheet.swift`:
the “Models” sheet; `RemoteControl.swift`: the socket that `silicontrol …` drives, see
[`REMOTE-CONTROL.md`](REMOTE-CONTROL.md);
`Localizable.xcstrings`: its phrases in English, French, German, Spanish and Italian). It is **compiled by the package**
(executable target `SiliconedApp`, as a pure client, no `@testable`), so it cannot drift from the API:
when in doubt, it is the reference example.
`tools/app.sh` builds it into `Siliconed.app` (translations and icon included) and opens it;
`SILICONED_ROOT=<folder> swift run SiliconedApp` does too, in the system's language.

Package: `Package.swift`, product `.library(name: "Siliconed")`, **macOS 15+**, Swift 6.

---

## 1. In 30 seconds

```swift
import Foundation
import Siliconed

let library = Library(root: URL(fileURLWithPath: "/Users/me/Siliconed"))
try EngineSettings.load(from: library)   // at launch, before the first render (see §2.1)

let model = try Model.zImage(in: library)     // checks every file, computes nothing
let request = Request("a 30 year old woman posing in a library", resolution: 1024, seed: 42)

var progress = Engine.Progress()
for try await event in Engine().events(request, model: model) {
    progress.receive(event)
    switch event {
    case let .step(_, index, total, _, seconds, _, _):
        print("step \(index)/\(total) in \(seconds) s — \(Int(progress.fraction * 100)) %"
              + (progress.estimatedRemaining.map { ", ~\(Int($0)) s left" } ?? ""))
    case .warning(let message):
        print("⚠ \(message)")
    case .image(_, let renderResult):
        try PNG.write(try renderResult.png(), to: "/Users/me/library.png")   // sRGB PNG + metadata
    default:
        break
    }
}
```

The render runs on the engine's queue, not on the cooperative pool. If the iterating `Task` is
cancelled, or if the loop exits, the render is cancelled and its memory returned.

---

## 2. Concepts

**Conventions**, one way to do each thing:

- **Names**: what the engine *emits* is nested in `Engine` (`Render`, `Event`, `Plan`, `Stage`,
  `Preview`, `Progress`, `Timings`, `Footprints`); what the app *builds or plugs in* is at the top
  level (`Request`, `Model`, `Chain`, the modules, `ImageRGB`, `Library`, `Cancellation`). **One
  error for everything: `EngineError`**, a closed enumeration (§5); every public function that
  throws is `throws(EngineError)`.
- **Labels**: the library goes under `in:`; `render`, `renderBatch`, `events` take
  `(request, [seeds:], model:, …)` in that order; a strength is a `Double`.
- **Surface**: what only the package's own tools and tests use is `package` — invisible to an app.

### 2.1 `Library`: the models' folder

```swift
public struct Library: Sendable {
    public let root: URL
    public init(root: URL)
}
```

Every path derives from `root`. (The on-disk names `composants`, `importes`, `telechargements`
and `profil.json` are kept from the original layout, so existing libraries stay valid.)

| folder | content | size (Standard installation, `Family.installedSize`) |
|---|---|---|
| `<root>/store/` | forged maps (`*.silicon`: DiT, text encoders), forged LoRAs (`*.lora.silicon`), `profil.json` | Z-Image: 12.3 GB DiT + 7.8 GB encoder; Qwen-Image-2.1: 14.2 GB DiT + 16.3 GB encoder (Compact: 11.7 and 19.4 GB in all; Light: 9.5 and 16.2 GB, §3.11) |
| `<root>/store/composants/<family>/` | a family's small published files, read as is: tokenizers, VAE, `transformer.json`, `dit.json` (names and shapes of the publisher's DiT, against which an import is checked) | 0.18 GB for Z-Image, 2.7 GB for Qwen-Image-2.1 (its turbo LoRA, 1.36 GB, included) |
| `<root>/store/importes/` | imported DiTs (`<name>.silicon`) | the size of one DiT of the family |
| `<root>/telechargements/` | an installation's sources, for the time it takes to forge them | empty at rest |
| `<root>/cache/` | **the render cache**: what renders keep between them so as not to redo it (below) | up to ~0.5 GB, plus one Qwen-Image-2.1 edit's K/V (2 to 6.4 GB) |

**Nothing else is read at render time, and everything can be redone**: `store/` rebuilds from an
empty root (`library.install(_:)`, §3.11), which downloads from each publisher, at a pinned
revision checked by sha256, then forges the same maps bit for bit. An imported file can be thrown
away once imported. `cache/` only saves time: deleting it changes no image.

**The render cache** (`library.cacheFolder`). Re-rendering the same prompt with another seed, or the
same edit (same instruction, same images) again, does not redo what does not depend on the seed:

- `cache/conditioning/`: the text encoder's output, for every model, and the latents the VAE encodes
  (an img2img start image, editing references). Bounded at 512 MB; the least recently used entries
  go first. Z-Image's text stage goes from 2.8 s to 0.01 s;
- `cache/kv/`: Qwen-Image-2.1's conditions (instruction and references) as its DiT reads them at
  every step — the keys and values its first step computes. **One edit only** (the last one, both
  phases in the 9-step mode), 2 to 6.4 GB on disk (6.4 GB at 1024×1536 with three references); it is
  not kept when it would leave less than 8 GB free on the volume. A generation (no reference) keeps
  nothing here: its conditions cost almost nothing.

What comes back is **the very floats** that were computed: the image is the same, bit for bit, as
without the cache. Each entry's key names everything that determines it: the model's files (path,
size, date: a reinstall or a new import is another key), the prompt, every byte of every image,
the LoRA stack for the K/V, the settings that change bits, and the engine binary itself (a new
version of the app starts from an empty cache). The cache is used by the models built from a
library (`Model.named(_:in:)` and the like); a hand-composed `Model(card:chain:)` has none.

- `library.occupancy().cache` counts it (and `total` includes it); `try library.emptyCache()`
  gives it back — a render in progress keeps the file it has open.
- `SILICONED_CACHE=0` in the environment, or `"cache": false` in the profile's settings, turns it
  off (`EngineSettings.effective.cache`).

- **Where to put it in an app**: the library searches for nothing by itself. The app builds
  `Library(root:)`, either on `~/Library/Application Support/Siliconed/` (`Library.standard`) or on
  a folder chosen by the user (in a sandboxed app: a security-scoped bookmark). The maps are
  projected into memory (mmap) and re-read at every evaluation: put them on the internal SSD. No
  external drive has been measured.
- **What to show of it**: `library.displayPath` (`~/Library/Application Support/Siliconed`), never
  `root.path` — a screenshot would carry the account's name. `Library.withoutHome(_:)` writes the
  home folder `~` anywhere in a text (an error message quoting a path); the diagnostic's `error`
  goes through it before it can be published.
- `Library.processWide` is the package's fallback when no library is designated (`SILICONED_ROOT`,
  then the current directory if it has a `store/`, then the repository deduced from the binary, then
  `Library.standard`). It is `package`: an app has no use for it.
- `try EngineSettings.load(from: library)` designates the library's `store/profil.json`, which
  holds the machine's settings (AMX co-execution, VAE tiles). The settings are read **once per
  process**, and **nothing reads them before a render**: neither building a `Request` (its
  `reproducible` is `nil`, “the machine's setting”, resolved at render time), nor a `Model`, nor
  `library.models()`. Only a render, `EngineSettings.effective` and `load` itself read them. So
  call `load` at launch (at the latest before the first render); **too late, it throws
  `EngineError.settingsAlreadyLoaded`** instead of silently doing nothing (unless the profile
  already read is the same one). A profile measured on another machine is ignored, with a warning
  in `EngineSettings.effective.warnings`, and the defaults apply.
- Low-level access: `map(_ name: String) -> String` (`<root>/store/<name>`), `profile`. The rest
  (components, DiT reference, downloads) is internal to the package.

### 2.2 `Model` and the catalog

A **model** groups a preset chain, a card and a license. Building it checks that each of its files
exists. If one is missing, the error gives its full path. No computation is started.

```swift
public static func zImage(in b: Library, denoising: ZImageDenoisingModule? = nil) throws(EngineError) -> Model
public static func qwenImage21(in b: Library) throws(EngineError) -> Model
public static func named(_ identifier: String, in b: Library) throws(EngineError) -> Model
public init(card: ModelCard, chain: Chain)        // a hand-composed chain (§2.4)
// model.identifier ("z-image" | "qwen-image-2.1" | "<family>/<name>" imported), .name, .family, .license, .card, .chain
```

The library is always passed under the label `in:`, as in `card.missing(in:)`.

The **catalog** describes the models without opening anything heavy, and it never throws:

```swift
let readySet: [ModelCard] = library.models()    // the ready models (all files present)
let allCards: [ModelCard] = library.cards()   // the repository's models, then the imported ones, ready or not
let missingList: String? = allCards[1].missing(in: library)   // what is missing, or nil
let model: Model = try Model.named(readySet[0].id, in: library)   // a card's model
let loras: [LoRACard] = library.loras(for: "z-image")  // store/*.lora.silicon, read from their header, sorted by name
```

`ModelCard` (`Identifiable`, `Hashable`: usable as is as a `Picker`'s `tag`): `id`, `name`,
`family: Family`, `isImported`, `license: License` (`text`, `commercial`, `requiredFilter`, `urls`),
`defaultSteps` (8 for Z-Image, 6 for Qwen-Image-2.1), `space`, `formats: [RecommendedFormat]`. An imported
model inherits from its family the architecture, the formats and the publisher's license text (to
be checked on the model's page).
`LoRACard`: `path`, `name` (readable), `rawName`, `target` (the family aimed at), `rank`,
`resolution?`, `trainedOn?`, `compatible(with:)`, `entry(strength:) -> LoRAEntry`.

| identifier | name | license | denoiser | VAE |
|---|---|---|---|---|
| `z-image` | Z-Image Turbo | Apache 2.0 | DiT S3, 8 steps, spectral schedule at 1024² | Flux |
| `qwen-image-2.1` | Qwen-Image-2.1 Turbo (§3.12) | **non-commercial** (Qwen Research) | Qwen-Image-2.1 DiT (32 single-stream layers, block-causal attention) + Viggle's turbo LoRA, 6 steps; editing with 1 to 3 images (§3.12) | Qwen-Image-2.1 (16×, 64 channels) |

### 2.3 `Request`: everything that distinguishes a render

```swift
public init(_ prompt: String, width: Int, height: Int, seed: UInt64 = 42, steps: Int? = nil,
            loras: [LoRAEntry] = [], image: ImageRGB? = nil, strength: Double = Strength.defaultValue,
            previews: Bool = false, reproducible: Bool? = nil)
public init(_ prompt: String, resolution: Int = 1024, seed: UInt64 = 42, steps: Int? = nil, …)  // the square
```

Building a request **reads nothing** (no file, no profile): it can live in an app's state before
anything else.

- **Prompt**: an empty or blank prompt is refused by every model, before any computation
  (`EngineError.emptyPrompt`).

- **Format**: `width` × `height` in pixels (in that order, like Draw Things: `832x1216` is a
  portrait). The rules are in §4.
- **Seed**: `UInt64`. The default is 42; for a random seed, the app draws `UInt64.random(in:)`
  itself.
- **Steps**: `nil` gives the denoiser's default (8 for Z-Image Turbo, 6 for Qwen-Image-2.1 Turbo;
  both are distilled for that number). Another value is accepted (Qwen-Image-2.1: only 5, 7 or 9, §3.12), but
  its quality has not been judged. A count outside `1...Request.maximumSteps` (50) is refused
  before any computation (`EngineError.invalidSteps`, `allowed` empty: “N steps: a render takes
  between 1 and 50.”).
- **LoRA**: `[LoRAEntry]`, a stack. `LoRAEntry(_ path: String, strength: Double = 1)`, or
  `loraCard.entry(strength: 0.8)`. The whole stack must target the model, otherwise the render is
  refused. A strength is a `Double` everywhere in the API (LoRA as well as img2img).
- **Image and strength** (img2img): `image: ImageRGB?` of any size. **The request scales it to
  fill and center-crops it to its format as soon as it receives it** (at init, on assignment, and
  when `width`/`height` change): it never keeps a full-size photo. Changing the format afterwards
  readjusts the image *already fitted*; to avoid losing edges, set the source image again after
  the format. The engine does not stretch it. `strength` has exactly the meaning of diffusers'
  `strength`, in `]0 ; 1]`, 0.6 by default (`Strength.defaultValue`). It is **quantized in steps
  of 1/N** and amounts to txt2img beyond `1 − 1/N`. Without `image`, `strength` is ignored:
  txt2img and img2img go through the same API.
- **References** (editing): `var references: [ImageRGB]`, empty by default. Unlike `image`, a
  reference **is not re-noised**: the denoiser reads it at every layer and the generated image
  starts from noise; the prompt says what changes (“change her jacket to bright red”). Each
  reference keeps **its own** proportions, at the size the denoiser gives it
  (`DenoisingModule.preparedReferences`: Qwen-Image-2.1 at ~1024² at multiples of 32), whatever the
  request's format — giving the request the format of image 1 (`editFormat`, §3.12) keeps the
  framing. Only the denoisers that announce it read them (`DenoisingModule.maxReferences`: 3 for
  Qwen-Image-2.1, 0 for Z-Image); beyond that, or on another model, the render is refused before
  any computation (`EngineError.tooManyReferences`). Cost: the sequence grows by as many tokens
  (Qwen-Image-2.1: a 1024² generation 132–136 s, an edit at 1248×832 with one reference ~190 s).
  **Editing by instruction** — the order of the images, how the prompt names them, the
  output's size: §3.12.

```swift
let qwen = try Model.qwenImage21(in: library)
let photo = try ImageRGB(contentsOf: chosenURL)
let (w, h) = qwen.chain.denoising.editFormat(referenceWidth: photo.width, referenceHeight: photo.height)
var request = Request("change her jacket to bright red", width: w, height: h)
request.references = [photo]                          // image 1, the one edited
let edited = try await Engine().render(request, model: qwen)
```
- **`reproducible`**: frozen GPU/AMX split, i.e. the same bits for the same seed (§4). `nil` by
  default: the machine's setting (`frozenCut`, true by default), read at render time.
- **`previews`**: a small image per step (§3.6). False by default; when false, nothing is
  computed.
- **`sketch`**: `Int?`, `nil` by default — stop after this many evaluations and decode the model's
  estimate of the final image instead of the image (§3.16). Set as a property, not in `init`.
- **`detail`**: `Detail` (`.normal`, `.more`, `.most`), `.normal` by default — finer texture at
  the same cost; `.normal` is the render without it, to the bit (§3.17). Set as a property.
- **`variations`**: `[Variation]`, empty by default — the seed's starting noise turned towards other
  seeds', in order: cousins of the image the seed gives (§3.18). Empty: the seed's noise, to the
  bit. In a batch, each seed is turned by the same variations. Set as a property.

### 2.4 `Chain` and its wires

A render always chains the same stages:

```
TextModule ─ Conditioning ─┐
                           ├─ DenoisingModule ─ Latent ─ DecodingModule ─ ImageRGB
[ImageEncodingModule: ImageRGB → Latent] ─┘   (img2img only)
```

The stages are connected by typed wires: `TextFormat` (`.zImage`, `.qwenImage21`) and
`LatentSpace` (`.flux` — 16 channels, one cell per 8 pixels —, `.qwenImage21` — 64 channels, one
cell per 16 pixels). `Chain` checks
the connections **at construction**, and a wrong connection throws `EngineError.incompatibleChain`. A format that
**reads the images** (`TextFormat.readsImages`: Qwen-Image-2.1's) must come from an
`ImageTextModule`, which receives the references with the prompt — otherwise
`EngineError.incompatibleChain`.

```swift
public init(text: any TextModule, denoising: any DenoisingModule,
            decoding: any DecodingModule, encoding: (any ImageEncodingModule)? = nil) throws(EngineError)
```

| model | text | image (img2img) | denoising | decoding |
|---|---|---|---|---|
| Z-Image | `ZImageTextModule` | `FluxEncodingModule` | `ZImageDenoisingModule` | `FluxDecodingModule` |
| Qwen-Image-2.1 Turbo | `QwenImage21TextModule` (an `ImageTextModule`) | `QwenImage21EncodingModule` (the references only: img2img is refused) | `QwenImage21DenoisingModule` | `QwenImage21DecodingModule` |

The system is **closed**: the protocols are public so that a chain can be composed from the
repository's modules, not so that new ones can be written (their `Context` has nothing usable
outside the package). An app almost never needs an explicit `Chain`: `model.chain` is enough. The
only useful setting is Z-Image's denoiser:

```swift
let withoutSpectral = try Model.zImage(in: library, denoising: ZImageDenoisingModule(
    map: library.map("z-image-turbo-dit.v1.silicon"), spectral: 0))
```

A hand-composed chain renders like the others, under a card:
`Model(card: .zImage, chain: myChain)` — `render`, `renderBatch` and `events` only take a `Model`.

A wrong connection is refused before any computation:

```swift
let z = try Model.zImage(in: b), q = try Model.qwenImage21(in: b)
_ = try Chain(text: z.chain.text, denoising: q.chain.denoising, decoding: q.chain.decoding)
// ✗ EngineError.incompatibleChain(output: "qwen3-4b · …", input: "qwen3-vl-8b · prompt + images · …")
```

Each stage builds its model, computes, then **returns its memory before the next stage**: that is
what lets it fit on a 16 GB machine. The trade-off: the weights are re-read at every render, and
no DiT stays loaded between two renders.

### 2.5 `Engine`: one render at a time

```swift
public final class Engine: Sendable { public init() }
```

An `Engine` holds nothing. All of the process's renders go through **a single static queue**
(`siliconed.engine`, QoS `.userInitiated`), whatever the number of `Engine()` created: two views
that each start a render will see them run one after the other. The rule has two reasons:

1. the engine is not reentrant (`GEMM` memoizations, `Conductor` state, arenas). Two simultaneous
   renders would not crash, they would silently corrupt their data;
2. there is only one GPU, two AMX blocks and 16 GB. Two renders in parallel would double the
   memory peak and gain nothing.

There are two ways to call the engine, with **the same parameters in the same order**
(`request`, [`seeds:`], `model:`, [`onProgress:`]). They run the same code (`execute`).

```swift
// async: what an app calls
public func render(_ request: Request, model: Model,
                   onProgress: (@Sendable (Event) -> Void)? = nil) async throws(EngineError) -> Render
public func renderBatch(_ request: Request, seeds: [UInt64], model: Model,
                      onProgress: (@Sendable (Event) -> Void)? = nil) async throws(EngineError) -> [Render]
public func events(_ request: Request, seeds: [UInt64]? = nil,
                       model: Model) -> AsyncThrowingStream<Event, Error>

// blocking: a command-line tool (cancellation goes through an explicit token)
public func render(_ request: Request, model: Model, cancellation: Cancellation? = nil,
                   onProgress: (@Sendable (Event) -> Void)? = nil) throws(EngineError) -> Render
public func renderBatch(_ request: Request, seeds: [UInt64], model: Model, cancellation: Cancellation? = nil,
                      onProgress: (@Sendable (Event) -> Void)? = nil) throws(EngineError) -> [Render]
```

**Which to prefer**: a UI takes the `events` stream, which delivers the events in order, where one
iterates. The `onProgress` closure is called **on the render thread**, synchronously; it must
return quickly. Without `onProgress`, a render's warnings go to `Warnings.outsideRender` (they are
no longer lost).

**Order of events** of a successful render: `start(Plan)`, `stage(.text)`, `text`,
[`stage(.image)`, `encoding`], then for each image: `stage(.denoising)`, `step` × N (each followed
by its `preview` if requested), `stage(.decoding)`, `decoding`, `image(index:, Render)`.
A `warning` can arrive at any time.

| event | use in the UI |
|---|---|
| `start(Plan)` | `plan.denoising.evaluations`, `plan.seeds`, `plan.stages`, `plan.denoising.startStep` / `startSigma` (img2img), `plan.denoising.reduced` (spectral) |
| `stage(Stage, image:)` | label “text… / image… / denoising… / decoding…”; emitted *before* the stage |
| `text(tokens:tokenizer:encoder:)`, `encoding(seconds:)`, `decoding(seconds:)` | timers |
| `step(image:index:total:sigma:seconds:latentGridHeight:latentGridWidth:)` | `index` runs from 1 to `total`, numbered the same way for every model |
| `preview(Preview)` | the step's thumbnail (if `request.previews`) |
| `image(index:, Render)` | the result (the last event of each image) |
| `warning(String)` | to log or display discreetly |

`Engine.Progress` draws from these events a `fraction` (in `[0, 1]`, which never goes backwards
during a render) and an `estimatedRemaining` in seconds (`nil` before the first step), computed
from the duration of the steps already done. The unit is the DiT evaluation. Each `start` resets
it: a single `Progress` can serve the app's whole life.

### 2.6 `Engine.Render`: what comes out

A `Render` is an image with its trace. **The engine writes no file.**

- what produced it: `model`, `prompt`, `seed`, `steps`, `loras`, `loraSummary` (the stack in one
  line), `strength?` (`nil` in txt2img), `startStep`, `startSigma`, `reproducible` (resolved),
  `detail`, `variations` (those of null strength left out);
- the image: `image: ImageRGB` — `pixels` (`[3, height, width]`, planar, fp32 around `[-1, 1]`,
  not clamped: the decoder outputs up to ±1.14; `rgba8()` clamps), `height`, `width`;
- the cost: `evaluations` (for a sketch, those it stopped after), `sketch` (`nil` unless the
  render stopped as a sketch, §3.16), `spectral` (the `k` applied), `timings` (`text`, `encoding`,
  `denoising`, `decoding`, which partition `total`), `footprints` (memory at the checkpoints),
  `tokens`;
- the outputs: `metadata: [String: String]` and `png() throws(EngineError) -> Data`.

At 1024², `renderResult.image.pixels` weighs 12 MB. For a history, better keep
`renderResult.image.cgImage()` or `rgba8()`.

---

## 3. Recipes

### 3.1 txt2img

```swift
let renderResult = try await Engine().render(Request("woman posing in a library", width: 832, height: 1216,
                                              seed: 7), model: try Model.qwenImage21(in: library))
```

### 3.2 img2img from a file

```swift
let entry = try ImageRGB(contentsOf: chosenURL)   // PNG, JPEG, HEIC, TIFF… ; EXIF applied ; sRGB ; alpha over white
let request = Request("a watercolor painting of a woman reading in a library", resolution: 1024, seed: 42,
                      image: entry, strength: 0.6)       // the image is fitted to 1024² right here
let renderResult = try await Engine().render(request, model: try Model.zImage(in: library))
// renderResult.startStep, renderResult.startSigma: what the strength gave FOR THIS MODEL
```

- The crop the engine will apply can be shown in advance: `ImageRGB.crop(source:target:)`
  computes the geometry without touching the pixels.
- Decoding a large file is **downsampled by ImageIO**: `ImageRGB(contentsOf:maxSide:)`, at most
  3072 px on the long side by default (`ImageRGB.maximumDecodedSide`, the longest side an accepted
  format can have). A 48 Mpx photo never lives at full size.
- An **in-memory** image (drag and drop, pasteboard): `try ImageRGB(cgImage: cg)` (sRGB, alpha over
  white; no EXIF orientation, a `CGImage` has none), bounded the same way: a longer side is drawn
  straight to `maxSide`, never held at full size.
- Bounds of the strength slider: between 0.2 and 0.8 in practice. For Z-Image, the strength must
  exceed 1/N, i.e. 1/8 at 8 steps (`EngineError.strengthTooLow` gives the `floor`).
  `Strength.startStep(steps:strength:)` gives the starting step, to display it before the render.
- Z-Image does not change *style* in img2img, even at 0.85.
- Qwen-Image-2.1 does no img2img (`EngineError.imageToImageUnsupported`): pass the image as
  image 1 of an edit (§3.12).

### 3.3 LoRA, and the refusal of another model's LoRA

```swift
let model = try Model.zImage(in: library)
let compatible = library.loras(for: model.identifier)     // show only those
let request = Request("portrait, studio light", resolution: 1024,
                      loras: compatible.prefix(1).map { $0.entry(strength: 0.8) })
let renderResult = try await Engine().render(request, model: model)
```

Each forged LoRA declares its family (`cible.modele` in its header — an on-disk key kept from the
original layout — written by the forge from the family's DiT, against which it checked every
module and every shape). If a LoRA in the stack targets another model, the render throws
`EngineError.loraForOtherModel` **before any computation** (~0.3 s). Without this refusal, the LoRA
would touch no module and the image would come out without it, with no message at all. A map
forged without a target throws `.unsupportedLoRA(file:reason: .noTarget)`: it must be reforged.

A LoRA is **imported** as published (§3.11): ai-toolkit, PEFT, diffusers, kohya (`lora_unet_…` and
`alpha`), ComfyUI. It targets a family: it also applies to the imported models of that family.

### 3.4 Batch of seeds

```swift
let renderResults = try await Engine().renderBatch(request, seeds: [42, 43, 44, 45], model: model)
// or: Engine().events(request, seeds: [42, 43, 44, 45], model: model)
```

`request.seed` is then ignored. The text (and the input image) are encoded **only once**, and
image n of the batch is **bit-identical** to the isolated render of its seed (verified). The
gain is limited to the text, ~3 s per image, because the DiT is rebuilt for each image. In the
events, `image` / `index` give the position in the batch.

### 3.5 Cancellation (`Task.cancel`)

```swift
let task = Task { try await Engine().render(request, model: model) }
// … Stop button:
task.cancel()
// → `try await task.value` throws EngineError.cancelled, 0.03 to 0.41 s later, memory returned
```

- With the async `render` / `renderBatch`: `EngineError.cancelled` is thrown.
- With `events`: cancelling the iterating `Task`, or leaving the loop, cancels the render. **The
  loop then ends without throwing** (that is `AsyncThrowingStream`'s behavior): test
  `Task.isCancelled` after the loop.
- With the blocking API: `let a = Cancellation()`, pass `cancellation: a`, then `a.cancel()` from
  any thread — from a `SIGINT` handler, for a command-line tool's Ctrl-C.
- The token is checked between stages, between steps and between the layers of the DiTs and the
  encoders, never in the middle of a GEMM. A render started right after a cancellation therefore
  waits on the queue until the previous one has reached the next layer.

### 3.6 Per-step preview

```swift
let request = Request(prompt, resolution: 1024, previews: true)
// in the stream: case .preview(let p): thumbnail = p.cgImage()   // p.width × p.height = the latent grid
```

The preview is x̂₀ = x − σ·v, the prediction of the final image (not the noisy state), projected
from 16 channels to RGB. It is one pixel per latent cell: 64×64 at 512², 128×128 at 1024². It
must be enlarged for display (`.interpolation(.none)` or smoothed). Quality: 13 to 19 dB at the
first step, 25 to 30 dB at the last. Cost: negligible next to a step.

### 3.7 SwiftUI progress bar: `AsyncThrowingStream`

```swift
@MainActor @Observable final class ProgressIndicator {
    // One property per line: `@Observable` refuses `var a = 0, b = 1`.
    var fraction = 0.0
    var remaining: Double?
    var stage: Engine.Stage?
    var preview: CGImage?
    var image: CGImage?
    var errorMessage: String?
    private var task: Task<Void, Never>?

    func launch(_ request: Request, _ model: Model) {
        task?.cancel()
        task = Task {
            var progress = Engine.Progress()
            do {
                for try await item in Engine().events(request, model: model) {
                    progress.receive(item)
                    fraction = progress.fraction; remaining = progress.estimatedRemaining
                    switch item {
                    case .stage(let e, _): stage = e
                    case .preview(let a): preview = a.cgImage()
                    case .image(_, let renderResult): image = renderResult.image.cgImage()
                    default: break
                    }
                }
                if Task.isCancelled { return }            // cancelled: the loop ended without throwing
            } catch EngineError.cancelled {
            } catch {
                errorMessage = error.localizedDescription        // an EngineError; its `code` keys the translation
            }
        }
    }
    func cancel() { task?.cancel() }
}
// View: ProgressView(value: p.fraction) ; Image(decorative: cg, scale: 1)
```

The complete version is the app ([`Sources/SiliconedApp/`](../Sources/SiliconedApp/)).

**The rack, on the left**: the chain top to bottom, in the direction of the signal, **each module
with its settings in its box** — there is no other form. First the inputs, as the chain
declares them (`chain.entries`: the text, mandatory, typed in the prompt's box; “Edit”, the references, if the denoiser reads
some — numbered thumbnails up to `maxReferences`, chosen, dropped onto the box or the canvas, or
“Edit This Image” from the history, which makes it image 1; the output takes image 1's format,
§3.12). **An img2img's starting image (`.image`) is not there**: re-noised, it does
not follow an instruction prompt; `Request.image` stays in the API. Then the DiT (model, license and precision as labels,
orientation and recommended format, fixed or drawn seed, number of images, steps, a stack of at
most three compatible LoRAs, import; while it runs, its steps), then the VAE —
the latent is not a box, nor is the image: it is the result, on the canvas. The column follows the window: narrower, its boxes narrow and
their rows of settings wrap onto two lines. **Between two modules, the wire** that connects them, of its
real type: `chain.text.output` (the `TextFormat`) to the DiT, one more latent when images are set
for editing, `chain.denoising.space` (the `LatentSpace`, its channels) to the VAE; it is highlighted
while the module it feeds is working. Each module's name comes from the chain; the light of the
working box turns on (`.stage` events), red if the error fell there. Under the rack:
“Generate” with **its cost** as soon as this model has rendered at this format in the session (the
learned speed × `plan(…).evaluations`, never a constant), and **the queue**: “Generate” during a
render becomes “Add to Queue” — the job takes the settings of the moment; each waiting job can be
removed, “Stop All” empties the queue.

**The canvas, on the right**: during the render, the frame already has the shape of the coming
image and each step's preview is drawn in it, with the stage, a segment per evaluation, the time
remaining and “Stop”; then the image (pinch to zoom, two fingers to move it in its zone), its prompt and caption, “Redo (Same Seed)”,
and **its details**: settings, time and memory peak of each stage (`Render.timings`,
`Render.footprints`). The history as a strip of thumbnails (menu: reuse the settings, edit,
remove), **browsable during a render**: the running render heads the strip, its preview and progress
on a live thumbnail; a history image chosen meanwhile keeps the canvas, the live thumbnail brings it
back. **A PNG Siliconed wrote, dropped on the canvas, restores its settings** (model, prompt, seed
fixed, steps, size, LoRAs found by file name) through the same path as “Reuse These Settings”; what
cannot come back (model not installed, LoRA not found, an img2img's source image) is said, the rest
is restored. Any other image dropped there keeps its meaning (an edit's reference). **The menus** carry every shortcut: Generate ⌘↩, Redo ⌘R, Stop ⌘., Stop All ⌥⌘., Save
⌘S (PNG with metadata or JPEG), copy the image ⇧⌘C, full screen ⌘F, neighboring images ⌥⌘← →,
import ⌘O, statistics ⌥⌘I.

**The statistics palette** (⌥⌘I, or its toolbar button) floats above: the current render (elapsed,
remaining, end time), each stage (tokens, tokenizer and encoder, the duration of each DiT step as
bars, the VAE), the queue (estimated end at the pace learned in the session for each model and
format — a job never measured says so instead of inventing), the last image (time and memory peaks
per stage) and the session.

**The main thread does not compute**: decoding the reference image, encoding the PNG and reducing
the thumbnail happen on a detached task; the history keeps the PNG and a thumbnail, and only the
chosen image is decoded at full size; the catalog (LoRAs, what is missing) is re-read when the
model changes, not at every redraw. **Five languages**: English, French, German, Spanish and
Italian, following the system; `tools/translations.sh` extracts the phrases from the code,
synchronizes them into `Localizable.xcstrings` and refuses a phrase missing in any of them. **Incognito**: the history lives in
memory, and nothing is written without an explicit gesture — no temporary file, no window position,
no restored state. **For a script**: no launch argument fills the form or renders; `silicontrol add` queues a
render in the open app (`docs/REMOTE-CONTROL.md`, `silicontrol help add`).

### 3.8 Writing a PNG with metadata, getting a `CGImage`

```swift
let cg: CGImage = renderResult.image.cgImage()           // sRGB, 8 bits
let bytes: [UInt8] = renderResult.image.rgba8()         // RGBA row by row, opaque alpha
let png: Data = try renderResult.png()                   // sRGB + iTXt chunks, no date: same bytes for the same render
try PNG.write(png, to: url.path)               // atomic; throws EngineError.imageWriteFailed
let custom = try renderResult.image.png(metadata: renderResult.metadata.merging(["note": "favorite"]) { $1 })
let reread: [String: String] = PNG.text(try Data(contentsOf: url))   // read the parameters back
```

The keys written are `prompt`, `seed`, `model`, `steps`, `format` (`WxH`), `reproducible`,
`Software`, plus `strength` and `image` (`source not included`) in img2img, and `lora`
(`file.lora.silicon:0.80,…`, **file name only**) when there are any, and `sketch` (the evaluations
it stopped after) for a sketch (§3.16), `detail` (`more` or `most`; **absent for `normal`**, as on
every render before it, §3.17) and `variation` (`seed:strength,…`, in the order applied; `seed`
stays the origin's, §3.18). They suffice to redo a **txt2img** (same bits if
`reproducible`, same machine, same version); **not an img2img**: the source image is not in the
PNG, and the `image` key says so. A key must be ASCII and at most 79 bytes, otherwise it is
ignored.

`Engine.Render.Recipe(metadata: PNG.text(data))` reads them back as values (`model`, `prompt`,
`seed`, `steps`, `width`/`height`, `loras`, `strength`, `sourceImageMissing`, `detail` (`normal`
when the key is absent), `variations` (empty when absent) — not `sketch`: a sketch's
recipe redoes the finished image), `nil` unless
`Software` is `Siliconed`; a key present but unreadable is `nil` and listed in `unreadable`. Pure:
it does not say whether the model or a LoRA is installed, and `model` is the family (`z-image`),
not an imported fine-tune's card.

### 3.9 Listing the models and the compatible LoRAs

```swift
let readySet = library.models()                                   // [ModelCard]
for card in ModelCard.allCards where !readySet.contains(where: { $0.id == card.id }) {
    print(card.name, "unavailable:", card.missing(in: library) ?? "")
}
let loras = library.loras(for: "qwen-image-2.1")                // [LoRACard], sorted by name
```

`library.models()` builds each model to check its files: it is fast (no weights read), and it does
not read the machine's profile (§2.1).

### 3.10 Recommended formats

`ModelCard.recommendedFormats` (and `card.formats`) are the Draw Things formats that fit under the
ceiling: **1024², 896×1152, 832×1216, 768×1344**, their transposes, and **512²** for fast
iteration. `RecommendedFormat` has `width`, `height`, `orientation` (`.square`, `.portrait`,
`.landscape`) and a displayable `description` (`832×1216 (portrait)`). 1024×1536 is accepted but
not recommended (on Z-Image it takes ~153 s against ~82 s for a 1024²). For a free-form field,
`Format.parse("832x1216")` then `Format.check(width:height:)`.
When the memory refuses a format (`insufficientMemory`), `RecommendedFormat.largestFitting(card.formats,
available:current:need:)` gives the largest one that fits a budget read once — the chosen orientation
first, never under 512 on a side — or `nil`: what the app offers instead of a dead end.

---

### 3.11 Importing a `.safetensors` or a `.gguf`, installing a family

A user brings a `.safetensors` as Civitai or Hugging Face publishes it: a LoRA, or a model's DiT,
**without encoder or VAE** — or a DiT as a `.gguf`. The import recognizes what it is and for which
family, installs from the publisher what the family lacks (tokenizers, text encoder, VAE), and
forges. Original names (ComfyUI, `net.`, fused qkv) as well as diffusers names are read; bf16,
fp16, fp32, the 8-bit formats and the GGUF types from 4 to 6 bits below too. Every name and every shape is checked against the
publisher's DiT: a file of another architecture throws, it is not guessed.

**A map is never wider nor narrower than the file given.** The published bytes and their scales
are copied as they are, only transposed; fp16 stays fp16, and an fp32 value that bf16 cannot hold
exactly stays fp32 (a fine-tune published in fp32 can thus give a whole fp32 map, ~24 GB for
Z-Image). The 8-bit formats kept as published:

| Format | Weight | As published by |
|---|---|---|
| fp8 E4M3, one scale per tensor | `w = f8 · s` | ComfyUI “scaled” (checked on synthetic files only) |
| fp8 E4M3, one scale per row | `w = f8 · s[n]` | torchao, SDNQ |
| int8, one scale per row | `w = q · s[n]` | SDNQ, torchao, ComfyUI `int8_tensorwise` |
| int8 convrot | `w = (q · s[n]) · R`, R a Hadamard rotation over groups of 256 inputs | ComfyUI |
| GGUF Q8_0 (`.gguf`) | blocks of 32 inputs, `w = d · q`, d fp16 | unsloth |

**Below 8 bits, GGUF only, down to 4 bits.** The `.gguf` files that unsloth, jayn7, leejet
(stable-diffusion.cpp) and Abiray publish for Z-Image Turbo and Qwen-Image-2.1 are read in these
types, mixed tensor by tensor as the publisher chose (a `Q4_K_M` file holds Q4_K, Q5_K, Q6_K and
sometimes Q8_0 tensors):

| Type | Bits per weight | Weight |
|---|---|---|
| Q6_K | 6.56 | super-blocks of 256 inputs, `w = (d · sc) · q`, d fp16, sc int8 per 16 |
| Q5_K, Q4_K | 5.5, 4.5 | super-blocks of 256 inputs, `w = (d · sc) · q − dmin · m`, 6-bit sc and m per 32 |
| Q5_0, Q4_0 | 5.5, 4.5 | blocks of 32 inputs, `w = d · (q − 16)` or `d · (q − 8)`, d fp16 |
| Q5_1, Q4_1 | 6, 5 | blocks of 32 inputs, `w = d · q + m`, d and m fp16 |

**Kept as published** means the blocks are copied whole into the map, bytes unchanged: never
unpacked to 8 bits (the map would grow), never re-quantized by us. **The floor is 4 bits**: Q3_K,
Q2_K, the I-quants (IQ…), TQ, MXFP4 and Q8_1 refuse the file entirely, even when a single tensor
carries them — so does a `Q4_K_S` that mixes in Q3_K tensors (unsloth's for Qwen-Image-2.1), with a
message that names the type.

The engine dequantizes each weight when it loads it and **computes in fp32**, as with any map: no
int8 matrix product, activations never go down to 8 bits (`input_scale` is ignored). Each fp8,
int8 and GGUF weight, once dequantized, is **bit-identical to the publisher's own fp32 dequantization** (torchao, SDNQ,
diffusers' GGUF code, and gguf-py for the types below 8 bits), checked on pieces of published
Z-Image and Qwen-Image-2.1 files and on whole maps (Z-Image Q8_0, Q6_K, Q4_K_M; Qwen-Image-2.1 Q4_K_M), on CPU and GPU. Int8 convrot, whose reference rounds twice, is exact to the fp32
rounding of the fp64 result, 10× closer to it than ComfyUI's own fp32.

Measured on Z-Image Turbo (M1 Pro, 16 GB):

| | bf16 map | SDNQ int8 | int8 convrot | GGUF Q8_0 |
|---|---|---|---|---|
| map on disk | 12.3 GB | 6.2 GB | 6.2 GB | 7.25 GB (refiners published in bf16) |
| import | — | 11 s | 31 s | 18 s |
| time per step, 512² | 3.9–4.3 s | +15 to +19 % | not timed | +8 to +17 % |
| time per step, 1024² | 15.5–16.1 s | +2 to +5 % | not timed | +3 to +6 % |
| 1024×1536 | 0 swap | 0 swap | 0 swap | 0 swap |

An 8-bit file is another set of weights: same composition, a different image from the bf16 one
(PSNR 23 to 31 dB at 512²).

The GGUF files below 8 bits buy a smaller map and fewer bytes read from disk at each step — what
counts on a Mac with 8 GB, where the DiT map cannot stay in memory between steps. Measured on Z-Image
Turbo (M1 Pro, 16 GB):

| | bf16 map | GGUF Q8_0 | GGUF Q6_K | GGUF Q4_K_M |
|---|---|---|---|---|
| map on disk | 12.3 GB | 7.25 GB | 5.93 GB | 5.04 GB |
| read from disk per step, 16 GB Mac | 5.43 GB | 2.89 GB | 2.23 GB | 1.78 GB |
| 1024×1536 | 0 swap | 0 swap | 0 swap, 3.35 GB peak | 0 swap, 3.35 GB peak |

On this machine a render is also a little faster than with Q8_0. A file below 8 bits is yet another set of weights: not degraded to the eye, but not the same
image either — at 1024², a Q4_K_M file can compose a different image from the same seed.

**Refused**, file and all, with a message that says why: GGUF below 4 bits (above); 4-bit outside
GGUF (nvfp4, SDNQ uint4, Nunchaku) — a file that mixes a single such layer with 8-bit ones is
refused entirely; mxfp8; fp8 E5M2; asymmetric int8 (a zero point that is not null).

```swift
// In a detached task: an import reads and writes GBs, an installation downloads some.
let r = try ModelImport.importFile("/Users/me/Downloads/my-model.safetensors",
                                 name: "My model", in: library) { rowLine in print(rowLine) }
// r.kind (.lora | .model), r.family, r.name, r.path, r.journal
if let id = r.identifier {                           // "z-image/my-model"
    let model = try Model.named(id, in: library)
    _ = try await Engine().render(Request("a 30 year old woman posing in a library"), model: model)
}

// An absent family, or redoing everything from an empty root:
try library.install(.qwenImage21) { print($0) } // baseDiT: false → without the publisher's DiT
let ready: [Family] = library.readyFamilies() // enough to render an imported model

// Z-Image and Qwen-Image-2.1 also install in a Compact or a Light version (below):
try library.install(.zImage, variant: .compact) { print($0) }
library.variant(of: .zImage)                   // .compact — what the render reads
Family.zImage.preselectedVariant()             // .light on a Mac of 8 GB, .standard elsewhere
```

**Standard, Compact or Light.** `Family.variants` lists the versions a family installs in: every family
has its **Standard** (`Variant.standard`, the publisher's weights as published — the default);
Z-Image and Qwen-Image-2.1 also have a **Compact** (`Variant.compact`): 8-bit weights published by
third parties, for the DiT **and** the text encoder, kept 8-bit in the maps — about 40 % smaller on
disk, an image slightly different from the Standard's, about as fast (the weights are widened at
each evaluation, and less is read from disk). Tokenizers, VAE and Qwen-Image-2.1's turbo LoRA are
the Standard's. They also have a **Light** (`Variant.light`): the DiT as unsloth publishes it in
GGUF Q4_K_M — 4- to 6-bit blocks kept whole, as any GGUF import (below), Qwen's with four Q8_0
tensors — and **the Compact's text encoder**, the same map (no text encoder is published below
8 bits in a form the forge reads, and Siliconed never quantizes one itself). The smallest on disk
and the least read from it at each step (Z-Image: 1.78 GB per step instead of 2.89); another set
of weights, so an image that may compose differently. On an M1 Pro with 16 GB it renders in
the Compact's time (Z-Image 512² −5 to −7 %, 1024² and Qwen-Image-2.1 within ±1.5 %). Switching
between Compact and Light replaces the DiT alone. `family.preselectedVariant()` is the version an
app offers first: the Standard, the Light on a Mac with 8 GB of memory — preselected, never imposed.

| family | Compact DiT | Compact text encoder | on disk, Standard → Compact |
|---|---|---|---|
| Z-Image | `unsloth/Z-Image-Turbo-GGUF` `z-image-turbo-Q8_0.gguf` (GGUF Q8_0) | `Disty0/Z-Image-Turbo-SDNQ-int8` `text_encoder/model.safetensors` (SDNQ int8), with the publisher's `config.json` | 20.4 → 11.7 GB |
| Qwen-Image-2.1 | `Comfy-Org/Qwen-Image-2.1` `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` (int8 convrot) | `unsloth/Qwen-Image-2.1-FP8` `Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors` (int8 convrot), with the publisher's `config.json` | 33.2 → 19.4 GB |

| family | Light DiT | Light text encoder | on disk, Standard → Light |
|---|---|---|---|
| Z-Image | `unsloth/Z-Image-Turbo-GGUF` `z-image-turbo-Q4_K_M.gguf` (Q4_K, Q5_K, Q6_K) | the Compact's | 20.4 → 9.5 GB |
| Qwen-Image-2.1 | `unsloth/Qwen-Image-2.1-GGUF` `qwen-image-2.1-Q4_K_M.gguf` (Q4_K, Q5_K, Q6_K, Q8_0) | the Compact's | 33.2 → 16.2 GB |

Each file is pinned (revision and sha256). **One version of a family is installed at a time, and
never none**: `install(_:variant:)` with another version forges the new one beside the installed
one, which keeps rendering — `library.variant(of:)` is the version whose maps are complete — and
removes the old one's DiT and encoder **once the new one is complete** (the encoder only if
neither the new version nor another family reads it). A full disk, a cut, a stop or a refused file leave the installed version whole. Only
when the disk cannot hold both does the old one go before the first byte, after a check that
counts the space it gives back: `library.installPlan(_:variant:baseDiT:)` says so beforehand
(`removesFirst`), with what the installation adds (`added`), frees (`freed`) and needs at its peak
(`peak`) — show it before the user accepts. `variant: nil` (the default) keeps the version already
there, Standard if none: a call that names no version never switches. `baseDiT: false` (what an
imported model lacks) never switches either: another version than the one installed is refused.
`library.installedVariant(of:)` is `nil` while nothing of the family is there. The render needs
nothing more: `Model.named` reads the maps of `library.variant(of:)`.
`family.licenseURLs(variant)` lists the license files an installation of that version links —
`ModelCard.license.urls`, then each third party's model card.

```swift
let plan = try library.installPlan(.zImage, variant: .compact)
if plan.removesFirst { print("not enough room for both: the \(plan.replacing!) version goes first") }
```

A gated repository asks you to accept its license on huggingface.co, then to set `HF_TOKEN` (or
`hf auth login`, whose token is re-read); Z-Image and Qwen-Image-2.1 are open. To show what an installation costs and what is installed:
`library.occupancy()` (each family's `variant`), `Family.installedSize(_:)`, `library.uninstall(_:)`
(every version).

### 3.12 Editing by instruction

The prompt is an **instruction** (“replace the cloudy sky with a blue sky”, “remove the second
person from image 1 and put the dog of image 2 in her place”), and the images it speaks of travel
in `Request.references`. A model that edits says so in its chain: `chain.entries` holds a
`.reference` entry, and `chain.denoising.maxReferences` is how many images it reads
(Qwen-Image-2.1: 3; 0 for a model without editing, Z-Image). Nothing is re-noised: the images
are read, the new image starts from noise. No mask, no inpainting: the prompt alone says what
changes.

**Qwen-Image-2.1**, every stage checked against the reference, renders its worst case (1024×1536
with three references) without a byte of swap (4.58 GB peak). It takes 6 steps (Viggle's 5 and 7, and a 9-step mode,
are accepted; any other count is refused before any computation), and does **no img2img**: a
`Request.image` is refused — pass the image as image 1. Ask `chain.denoising.check(steps:startImage:)`
to know beforehand.

**The order is the numbering.** `references[0]` is **image 1, the one edited**; `references[1]`
is image 2, `references[2]` image 3 — what the prompt borrows from. Qwen-Image-2.1's encoder
(Qwen3-VL-8B) **sees** the images: its template puts `<image1>`, `<image2>`… with each image's
vision tokens in front of the prompt, so the prompt can name them (“image 1”, “image 2”).
`chain.text.output.readsImages` (`TextFormat.readsImages`) says that the encoder sees the images:
it is what an app reads before suggesting “image 1”, “image 2” in the prompt.

**Each reference is resized by the engine** (`chain.denoising.preparedReferences`), whatever the
request's format, its proportions kept. Qwen-Image-2.1: to about `output_resolution²` (1024², ~1 MP),
each side to the nearest multiple of 32 (ties to even), by Pillow's Lanczos to the bit; the same
resized image feeds the encoder and the VAE.

**The output's size follows image 1.** The request's `width` × `height` is the output's; for an
edit, give it image 1's format by this rule:

- the area of `output_resolution²` (1024²) at image 1's aspect ratio,
- each side a **multiple of 32**, and **at least 512** (`Format.minimumSide`) — the short side raised
  keeping the ratio,
- the area **under `Format.maxSurface`** (1024×1536): the longest side gives way.

diffusers' pipeline, without `height`/`width`, takes the size of the *last* image; Siliconed takes
image 1's, the one being edited, so that the framing is kept. The rule belongs to the denoiser:
`chain.denoising.editFormat(referenceWidth:referenceHeight:)`, which the app and `silicontrol add --ref`
both call when no format is typed — a command and a click give the
same image.

```swift
let photoURL = URL(fileURLWithPath: "photo.jpg"), dogURL = URL(fileURLWithPath: "dog.jpg")
let photo = try ImageRGB(contentsOf: photoURL), dog = try ImageRGB(contentsOf: dogURL)
let qwen = try Model.named("qwen-image-2.1", in: library)
let (w, h) = qwen.chain.denoising.editFormat(referenceWidth: photo.width, referenceHeight: photo.height)
var request = Request("remove the second person from image 1 and put the dog of image 2 in her place",
                      width: w, height: h)
request.references = [photo, dog]                      // image 1 (edited), image 2
let edited = try await Engine().render(request, model: qwen)
```

**Cost**: each reference lengthens the render — its tokens join the sequence. Qwen-Image-2.1's
conditions (text and references) do not see the image being generated: their keys and values are
computed **once per render**, then re-read at every step. Measured in series: a 1024²
generation takes 132–136 s, an edit at 1248×832 ~190 s with one reference, 262–264 s with two;
one DiT evaluation at 1024² ~20 s under the turbo. An app learns its own cost per model, format **and number of references** (the app does),
never from a constant.

Beyond `maxReferences`, or on a model without editing, the render is refused before any
computation (`EngineError.tooManyReferences`).

**In the app**, the “Edit” box shows the references as numbered thumbnails (1, 2, 3), at most
the model's `maxReferences`: added by its button, by dropping images onto the box or the canvas, or
by “Edit This Image” on a history image (which becomes image 1); removed and reordered by dragging
or from a thumbnail's menu. Image 1 is marked as the one edited, the size row shows the output's
format (“follows image 1”), and a discreet line reminds that the prompt can say “image 1 / image 2”.

### 3.13 Diagnostic: what this machine does, in a JSON

`Diagnostic.run` measures the installed models on **this** Mac and returns a `Codable` report — what
the app's “Report My Configuration” button sends, and what fills the README's table by chip. For
each model passed that is installed (`ModelCard.missing(in:) == nil`; the others are skipped), at
512², prompt "a 30 year old woman posing in a library", seed 42: the text encoder (cold, no render
cache), **two DiT evaluations** (the first pays the construction and the warm-up, the second is the
step in steady state), the VAE decoder. A whole render is **derived, not run**:
`estimatedRenderSeconds = encoder + first + (E − 1) · steady + decoder`, `E` the evaluations of the
model's default steps. Beside the timings: the machine (chip, P/E cores, GPU cores, memory, macOS,
version), the peak `phys_footprint` sampled every 10 ms, the `vm_stat` swap-outs before and after,
and — for a model whose small reference output is embedded in the library (Z-Image and
Qwen-Image-2.1) — the second `model_out`'s deviation from the fp32 reference, channel by channel, with
its threshold. A model without one reports `"golden": null, "deviation": null`; a model that fails reports its `error`, and the others
still run: `run` never throws.

Each model runs under the memory plan a render would take (`memory`: the budget at the launch, the
floor, `lean`, and the plan's `decisions`; filled even when the preflight refuses), and reports the
bytes it read from the disk (`diskReadBytes`: a map read from the page cache does not count). The
report adds the machine's `before`/`after` conditions (thermal state, Low Power Mode, battery,
reclaimable memory, the compressor's room, swap in use), an fp32 GEMM `witnessBefore`/`witnessAfter`
on the GPU and through `cblas_sgemm` (TFLOP/s, and the GPU's worst row against `cblas`), the
`settings` that are not the defaults (and `amx`, always), the `profile` read, and the GPU's Metal
name, family and working set in `machine`. `schema` is 2.

```swift
let cards = library.models().filter { ["z-image", "qwen-image-2.1"].contains($0.id) }   // the visible ones
let report = await Task.detached {
    Diagnostic.run(cards, in: library) { card, stage in   // stage: .textEncoder, .denoiser, .decoder
        print("\(card.name) · \(stage)")                  // on the diagnostic's thread: hop to the UI's
    }
}.value                                                   // ~15 s per model
let json = try report.json()                  // stable: sorted keys, explicit nulls
let issue = try report.issueURL()            // a prefilled GitHub issue: NSWorkspace.shared.open(issue)
let form = try report.issueURL(includingReport: false)   // the same form without the JSON, to paste
```

A link beyond ~8 000 characters is cut by browsers and GitHub: the app then copies `json()` to the
clipboard and opens `issueURL(includingReport: false)`. `Diagnostic.Stage` is what `onProgress`
reports, in English as `description`, for an app to say in its own language.

It is synchronous and heavy (one model's weights at a time, 2.9 GB at the peak for Qwen-Image-2.1):
run it off the main thread, and **never during a render** — the figures would measure both. Its
`cancellation` is checked between stages.

### 3.14 Alternatives in a prompt: `{a|b}`

`PromptAlternatives` turns one prompt with groups of alternatives into the prompts it stands for —
**combinatorial, never random**: several groups make their cartesian product, in reading order,
the first group varying the slowest. The engine itself never sees a brace: a `Request` takes one
expanded prompt, and an app queues one render per prompt, **at the same seed** (a series compares
prompts, nothing else may differ). That is what the app's “Generate” and `silicontrol add` do.

```swift
let series = try PromptAlternatives("woman posing in a {library|greenhouse} at {dawn|dusk}")
series.count                          // 4 — known before anything is built
series.groups                         // [["library", "greenhouse"], ["dawn", "dusk"]]
for v in series.variants {            // library·dawn, library·dusk, greenhouse·dawn, greenhouse·dusk
    let r = try await Engine().render(Request(v.prompt, resolution: 512, seed: 42), model: model)
    print(v.choices, r.metadata["prompt"] ?? "")   // ["library", "dawn"] woman posing in a library at dawn
}
```

- **The syntax.** `{` opens a group, `|` separates its alternatives, `}` closes it. An empty
  alternative is allowed (`{|red }dress` → `dress`, `red dress`) and nothing is trimmed: the spaces
  belong to the alternative that carries them. `\{`, `\}`, `\|` are the characters themselves;
  outside a group `|` is an ordinary character, so a prompt without braces or backslashes is
  returned as is, character for character.
- **The refusals**, before anything is built: `promptSyntax(position:reason:)` for a `{` never
  closed, a `}` that closes nothing, a group inside a group (`PromptRefusal`: `unclosed_group`,
  `unmatched_close`, `nested_group`; `position` counts characters from 1), and
  `tooManyPromptVariants(count:max:)` beyond `PromptAlternatives.maxCombinations` (64) — or the
  `limit:` passed to `init`.
- **For a grid**: `groups` are the axes' labels and `variants[i].choices` the label taken in each
  group by prompt `i`, as typed (escapes resolved).
- `PromptAlternatives.escaping(_:)` writes an expanded prompt back so that it parses to itself (its
  braces escaped): what a form shows when it reopens a history image, so that “Redo” renders that
  one prompt and not the series again.

### 3.15 An XY grid: `RenderGrid`

`RenderGrid` sweeps one or two settings — a `{…}` group of the prompt, the seed, the step count,
one LoRA's strength, the model, the edit's images — and lists **one ordinary render per cell**, row by row, X varying the fastest
(the order a sheet is read, and the order to queue them). The engine never hears of a grid: each cell
is a `Request`. Then `sheet(imageWidth:imageHeight:)` says where everything goes on the comparison
sheet — pure geometry, the drawing is the app's. That is what the app's « Exploration » window (its
cells are sketches, §3.16) and `silicontrol grid` (finished images) do.

```swift
let grid = try RenderGrid(.init(prompt: "woman posing in a {library|greenhouse}", seed: 42, steps: 8),
                          x: .prompt(group: 0), y: .consecutiveSeeds(count: 2))
grid.count                            // 4 = grid.columns × grid.rows — known before anything renders
for cell in grid.cells {              // library·42, greenhouse·42, library·43, greenhouse·43
    let r = try await Engine().render(Request(cell.prompt, resolution: 512, seed: cell.seed, steps: cell.steps),
                                      model: model)
    print(cell.column, cell.row, r.seed)
}
let sheet = grid.sheet(imageWidth: 512, imageHeight: 512)
print(sheet.width, sheet.height, sheet.cell(column: 1, row: 0))   // top-left origin, in pixels
let axis = try RenderGrid.axis("lora:flat=0.4,0.8", loras: ["flat"])   // .loraStrength(slot: 0, values: [0.4, 0.8])
_ = axis
```

- **The axes** (`RenderGrid.Axis`): `.prompt(group:)` (its alternatives, as typed),
  `.seeds([…])`, `.consecutiveSeeds(count:)` (from the base seed, like a batch), `.steps([…])`,
  `.loraStrength(slot:values:)` (the other LoRAs keep theirs), `.models([…])` (by identifier,
  `z-image`, `qwen-image-2.1`…: `Cell.model`), `.images` (each of the base's `images` edit images
  alone, with the same instruction: `Cell.image`, its index from 0 — without it, an edit reads all
  its images together), `.addedLoRAs([…], strength:)` (one LoRA added to the stack per cell, all at
  one strength — `""` for the cell without: `Cell.addedLoRA`, `Cell.addedStrength`; the strings
  are whatever the caller keeps, paths in the app). `values(of:)` gives an axis's values
  as `RenderGrid.Value`, for an app to label in its language; `title` is the prompt with its groups
  written back as `{a|b}`.
- **A model axis gives each cell its model's own step count**: `Cell.steps` is `nil` (the model's
  default — 8 for Z-Image, 6 for Qwen-Image-2.1) unless steps are the other
  axis; without a model axis, it is the base's or the axis's. `Cell.model` is `nil` without a model
  axis (the base's model), `Cell.image` without an image axis (all the images).
- **The refusals**, before any render: `gridRefused(reason:)` (`GridRefusal`: `empty_axis` — an
  image axis without edit images is one —, `same_axis_twice` — the seed is one setting however it
  is listed —, `no_such_group`, `no_such_lora`, `group_not_on_axis`, `invalid_value`,
  `unreadable_axis`, `lora_across_models` — a model axis with a LoRA stack or a LoRA axis: a LoRA
  is made for one model) and
  `tooManyGridCells(count:max:)` beyond `RenderGrid.maxCells` (64, the app's queue) or `limit:`.
  **Every group of the prompt must be an axis**: a group left out would stack several images in a
  cell — refused rather than multiplied as `PromptAlternatives` alone would.
- **The sheet** (`RenderGrid.Sheet`): each cell reduced to 512 px on its long side at most, all the
  cells together to 12 Mpx (64 cells of 512² come out at 432 px: a sheet of ~14 Mpx, ~55 MB as RGBA
  while it is drawn), font sizes that follow the cells, a row-label column only with a Y axis.
- `RenderGrid.axis(_:loras:)` reads an axis written `kind[:which][=values]` (`prompt:2`,
  `seed=42,7`, `seeds=4`, `steps=6,8`, `lora[:NAME]=0.4,0.8`, `loras[:STRENGTH]=none,flat,ink`
  (strength 1 if not given), `model=z-image,qwen-image-2.1`, `image`, decimals with a point): the syntax of `silicontrol grid`.

### 3.16 A sketch: stopping where the image is readable

`Request.sketch = n` stops the render after `n` evaluations and decodes the denoiser's estimate of
the final image, x̂₀ = x − σ·v, instead of the image. The estimate costs nothing (the evaluation
computed `v` anyway) and speaks early. The evaluations done are **the first ones of the whole
render, bit for bit**: a sketch is a prefix, not another render — the same request without
`sketch` finishes the image it showed. Under 1, one evaluation; at or past the plan's count, the
whole render (`Render.sketch` is then `nil`).

```swift
let model = try Model.zImage(in: library)
var request = Request("woman posing in a library", resolution: 512, seed: 42)
request.sketch = model.sketchEvaluations(steps: nil, width: 512, height: 512)   // 3 (of 7)
let rough = try await Engine().render(request, model: model)                    // rough.sketch == 3
request.sketch = nil
let finished = try await Engine().render(request, model: model)                 // the image it announced
```

- **Where to stop** (`Sketch`): after the first evaluation that brings σ to `Sketch.threshold`
  (0.8) or below — a σ, not a step count: Qwen-Image-2.1's schedule keeps σ high longer.
  `Sketch.evaluations(sigmas:)` reads it from a schedule (null steps are not evaluations),
  `Model.sketchEvaluations(steps:width:height:)` from a model at a format (`steps` nil: its default),
  and `DenoisingPlan.sketched(_:)` cuts a plan to it, to estimate a sketch's duration before it runs.
- **What it is worth** (512², estimate against the final image reduced ×4, as on a sheet):
  Z-Image stops after 3 of 7 evaluations — ~16.7 s instead of ~32.5 s, ~21 dB, pose, clothes and
  light already the final image's; Qwen-Image-2.1 after 4 of 6 — ~20 dB, the composition, soft, which is why it no longer sketches
  (`DenoisingModule.sketches`: its cells are finished images). A
  sketch shows where the image goes; the finished render is the judge (under a LoRA, Z-Image's pose
  still moved between 3 and 4).
- **Events**: `start` carries the whole render's plan; the `step` events count to the sketch
  (`total` = `n`), then decoding and `image` as usual. The PNG carries a `sketch` key (§3.8).

### 3.17 More detail: `Detail`

`Request.detail = .more` (or `.most`) tells the denoiser, in the middle of the trajectory, that a
little less noise remains than really does: it removes less, and what it leaves is fine texture.
The Euler step keeps the true σ, so the trajectory still lands at σ = 0. It is ComfyUI's Detail
Daemon (Jonseed, MIT), ported line by line and reduced to one named choice: none of its ten knobs
is exposed. **Same evaluations, same cost**; `.normal` is the render without it, to the bit.

```swift
var request = Request("woman posing in a library", resolution: 512, seed: 42)
request.detail = .more                                       // .normal · .more · .most
let detailed = try await Engine().render(request, model: try Model.zImage(in: library))
print(detailed.detail, detailed.metadata["detail"] ?? "")   // more more
```

- **What it is worth** (512², default steps, by eye and by the mean |Laplacian| of the
  luminance): `.more` ≈ +10 % of fine detail and the same picture, `.most` ≈ +20 %, the strongest
  that stayed clean. Z-Image takes the node's bell; on Qwen-Image-2.1, whose shifted schedule
  decides the composition early, that bell changed the picture, so it gets a late bell on its 5th
  step only.
- **The bell is laid over step indices**, not over σ: another step count than the default, or
  1024², has not been judged. In img2img it covers the steps that run.
- The PNG carries `detail` unless `.normal` (§3.8); the app shows it as « Detail: Normal · More ·
  Most » in the DiT box.

### 3.18 Variations: `Variation`

`Request.variations` turns the seed's starting noise towards another seed's: the image keeps its
composition in proportion to `strength` — Draw Things' and AUTOMATIC1111's “variation seed”. Only
the noise changes: the text, the images and the render cache's keys do not see it.

```swift
let model = try Model.zImage(in: library)
var request = Request("woman posing in a library", resolution: 512, seed: 42)
let subtle = Variation.Amount.subtle.strength(for: model.family)    // 0.1 (Qwen-Image-2.1: 0.05)
request.variations = [Variation(seed: UInt64.random(in: 0...UInt64.max), strength: subtle)]
let cousin = try await Engine().render(request, model: model)       // seed 42, its noise turned
request.variations.append(Variation(seed: 7, strength: 0.5))        // a variation of THAT image
```

- **The mix is a rotation**: noise ← cos θ·a + sin θ·b, θ = `strength`·π/2, in fp64 rounded once
  to fp32. For two independent standard normal draws it is *exactly* a standard normal again — the
  noise the denoiser was trained on — where AUTOMATIC1111's slerp is so only on average.
  `strength` in `[0, 1]` (clamped outside, 0 if not finite): 0 is the seed's noise to the bit, 1
  the variation seed's alone (a re-roll).
- **Variations chain**, applied in order: a variation of a variation turns *that* image's noise,
  so its cousins stay close to it, not to its parent.
- **The second noise comes from the same generator, at the same shape**: every model, editing and
  img2img alike. A variation is defined on the latent: the same pair of seeds at another
  format is another image.
- **`Variation.Amount`**, the app's two choices, never shown as numbers: `.strong` 0.5 everywhere
  (the same idea, pose, clothes and framing that move); `.subtle` 0.1 (same framing, pose and
  person), **0.05 on Qwen-Image-2.1**, whose first steps decide who is in the image. Calibrated
  on Z-Image and Qwen-Image-2.1 at 512² only.
  `Variation.Amount.named(_:)` names a strength read back from a PNG.
- The PNG carries `variation` (§3.8), and `Recipe.variations` reads it back: a render redone
  from its PNG gives the same bits.

## 4. Limits and contracts

| rule | value | in the code |
|---|---|---|
| floor | **each side ≥ 512 px** (latent 64). No evaluation happens below 512: under it, the models are outside their domain | `Format.minimumSide` |
| multiple | each side a multiple of **16** (VAE ×8, patch ×2) | `Format.multiple` |
| ceiling | area ≤ **1024×1536** (1,572,864 px), the last format measured | `Format.maxSurface` |
| memory | **16 GB** machine (MacBookPro18,3). Measured peaks (`phys_footprint`): 1.7 GB for Z-Image and 2.9 GB for Qwen-Image-2.1 at 512², 2.9 GB for Z-Image and 3.6 GB for Qwen-Image-2.1 at 1024², **3.8 GB** for Z-Image at 1024×1536, **4.58 GB** for Qwen-Image-2.1 at 1024×1536 with three references; 0 swap-out at each worst case. Before a render, the preflight checks the need against the memory actually available (§5) | the per-stage scope (`Chain.swift`), `MemoryBudget` |
| precision | **fp32** compute everywhere; the weights may be stored in bf16, fp16, 8 bits or GGUF 4 to 6 bits (§3.11), then widened. No compute in fp16/bf16 | `Widen` (the weights widened to fp32) |
| determinism | with `reproducible` (true by default, frozen GPU/AMX split): **same seed, same request, same machine and same version → same bits**, and non-regression is judged by md5. Image n of a batch = the isolated render of its seed. `Noise` is **not** `torch.randn`: a seed does not give back diffusers' image nor Draw Things'. From one version to the next, pixels may change, and from one machine to another, nothing has been measured | `Request.reproducible` (`nil`: `EngineSettings.effective.frozenCut`) |
| spectral schedule (Z-Image) | only when **both sides are ≥ 1024**: 1024², 1024×1536 and its transpose (`k = 2`). The usual portraits and landscapes pay their 7 full evaluations. It is off in img2img; forced below the floor, it is refused before any computation | `ZImageDenoisingModule.plan` |

**Indicative times**: MacBookPro18,3 (M1 Pro, 16 GB), machine profile loaded, default steps (8 for
Z-Image, 6 for Qwen-Image-2.1), seed 42, “a 30 year old woman posing in a library”, render cache
off, full render (text + denoising + decoding). They depend on heat and on the page cache (the first render re-reads the maps):
the UI must rely on `Progress.estimatedRemaining`, not on this table.

| model | 512² | 1024² | 1216×832 | 1024×1536 |
|---|---|---|---|---|
| Z-Image Turbo (8 steps) | **~29 s** | **~82 s** (spectral k = 2) | 110 s | ~153 s |
| Qwen-Image-2.1 Turbo (6 steps) | **~38 s** | **132–136 s** | — | — |

Qwen-Image-2.1 editing: at 1248×832, ~190 s with one reference, 262–264 s with two;
1024×1536 with three references ~523 s. A merged LoRA adds 0.2 to 0.4 s per step to an edit. On
Z-Image, a LoRA's cache adds about +340 MB to the denoising peak.

**Licenses**: `model.license` is displayed to the user, who accepts it: **a render of a model whose
license is not accepted in its library is refused** (`EngineError.licenseNotAccepted`, §5). The app
records the acceptance with `library.acceptLicense(card)` (`<root>/accepted-licenses.json`, the
text accepted kept: a license that changes asks again). Siliconed ships no weights: it is
only compatible, and everyone reads the license of the model they import.

`license.urls` says where to read it: the license file(s) on Hugging Face **at the revision the
installation downloads**, the model's first, then each component under a license of its own. An
imported model gets its family's.

```swift
let licenseURLs: [URL] = model.license.urls   // https://huggingface.co/<repository>/blob/<40-hex sha>/<file>
```

| model | `urls` (each at its pinned revision) |
|---|---|
| Z-Image Turbo | `Tongyi-MAI/Z-Image-Turbo` `README.md` — no license file; the model card declares Apache 2.0 |
| Qwen-Image-2.1 Turbo | `Qwen/Qwen-Image-2.1` `LICENSE`, then `Viggle/Qwen-Image-2.1-viggle-turbo` `LICENSE` (the turbo LoRA, same Qwen Research text) |

A Compact (§3.11) is under the same license; its installation also links each third party's model
card, which declares it (`family.licenseURLs(.compact)`): `unsloth/Z-Image-Turbo-GGUF` and
`Disty0/Z-Image-Turbo-SDNQ-int8` (Apache 2.0), `Comfy-Org/Qwen-Image-2.1` and
`unsloth/Qwen-Image-2.1-FP8` (Qwen Research). A Light links its DiT's card, then its encoder's:
`unsloth/Z-Image-Turbo-GGUF` and `Disty0/Z-Image-Turbo-SDNQ-int8`, `unsloth/Qwen-Image-2.1-GGUF` and
`unsloth/Qwen-Image-2.1-FP8`.

- **Z-Image Turbo**: Apache 2.0.
- **Qwen-Image-2.1 Turbo**: Qwen Research, non-commercial.
- Each LoRA has its own license, which the library does not read.

---

## 5. Errors

**One public error: `EngineError`**, a closed enumeration (`Sources/Siliconed/Errors.swift`). Every
public function that throws is `throws(EngineError)`; `events` ends its stream with an `EngineError`.
Each case carries its values, never a ready-made sentence:

- `code` — a stable key (`"disk_full"`, `"license_not_accepted"`…), what the app looks its
  translated sentence up by (its `Localizable.xcstrings`). Never renamed once published.
- `errorDescription` — the English source sentence; `recoverySuggestion` — what the user can do.
- `FormatRefusal`, `LoRARefusal`, `PromptRefusal` and `GridRefusal` refine `format_refused`,
  `unsupported_lora`, `prompt_syntax` and `grid_refused` (`format_refused.too_large`…).
- `internalFailure(component:detail:)` and `importRefused(file:detail:)` carry a technical
  diagnostic in English, to show under the sentence, never in it.

The readers and kernels keep their own errors inside the package; `EngineError(_:)` folds each into
its case at the door. The module protocols (`TextModule.encoder`, `DecodingModule.decode`…) are
public in name only (§2.4) and throw untyped; `render` folds what they throw.

**The preflights.** Before any computation a render checks, in this order, and throws the first
that fails: the model is installed (`card.missing(in:)`) → its license is accepted
(`library.isLicenseAccepted(card)`) → the format (`Format.check`) → the memory
(`card.checkMemory(width:height:)`: the model's floor at this format, `card.memoryNeed(width:height:)`,
against `MemoryBudget.current().available` — the machine's state read at the render's launch: free,
purgeable and file-backed pages, plus the process's own footprint, minus a reserve; not its RAM,
and not its "free" memory alone, which means nothing under the page cache) → no other process renders on
this machine, whatever its library or user (`/private/tmp/siliconed-render.lock`). Then the request's own refusals, still before any computation.
An installation checks the disk **before its first byte** (`diskFull`). A hand-composed chain
(`Model(card:chain:)`) has no library: the first two and the lock do not apply to it.

| case | `code` | when | what the app does with it |
|---|---|---|---|
| `modelNotInstalled(model:)` | `model_not_installed` | `Model.named`, a render after an uninstallation | offer `library.install(card.family)` |
| `unknownModel(model:)` | `unknown_model` | `Model.named` | identifier outside `library.cards()` |
| `licenseNotAccepted(model:)` | `license_not_accepted` | render | show `card.license`, then `library.acceptLicense(card)` |
| `fileMissing(path:)`, `fileUnreadable(path:)` | `file_missing`, `file_unreadable` | a LoRA or an image moved, permissions | choose the file again |
| `corruptMap(file:)` | `corrupt_map` | a map or published file that does not read (signature, size, header, another version); `file` may be `nil` | reinstall the model, re-import |
| `diskFull(needed:available:)` | `disk_full` | `install`, `importFile`, the K/V cache of an edit | make room (`occupancy()`, `uninstall`, `emptyCache`); downloads resume |
| `downloadInterrupted(file:)`, `downloadRefused(file:status:)`, `downloadCorrupt(file:)` | `download_interrupted`, `download_refused`, `download_corrupt` | `install` | retry; 401/403: gated repository, `HF_TOKEN` |
| `importRefused(file:detail:)`, `notAnImportedModel(model:)` | `import_refused`, `not_an_imported_model` | `ModelImport` | not a LoRA nor a DiT of a known family, shape mismatch, below 4 bits or another unread format |
| `emptyPrompt` | `empty_prompt` | render | grey out “Generate” if the prompt is blank |
| `invalidSteps(steps:allowed:)` | `invalid_steps` | render | `allowed` (Qwen-Image-2.1: 5, 6, 7, 9), or 1 to `Request.maximumSteps` (50) when empty |
| `tooManyReferences(count:max:)` | `too_many_references` | render | `max` 0: the model does not edit by reference |
| `imageTooSmall(side:minimum:)` | `image_too_small` | `Format.check`, render | offer `card.formats` |
| `formatRefused(width:height:reason:)`, `formatUnreadable(text:)` | `format_refused`, `format_unreadable` | `Format.check`, render | offer `card.formats` |
| `promptSyntax(position:reason:)`, `tooManyPromptVariants(count:max:)` | `prompt_syntax`, `too_many_prompt_variants` | `PromptAlternatives(_:)` (§3.14) | grey out “Generate” and say where; the count before the click |
| `promptReservedText(text:)`, `promptTooLong(tokens:max:)` | `prompt_reserved_text`, `prompt_too_long` | render, before any weight is read (Qwen-Image-2.1: `<\|image_pad\|>` typed in the prompt; more than 512 tokens) | say what to remove; `tokens` and `max` give the count and the limit |
| `gridRefused(reason:)`, `tooManyGridCells(count:max:)` | `grid_refused`, `too_many_grid_cells` | `RenderGrid(_:x:y:limit:)`, `RenderGrid.axis(_:loras:)` (§3.15) | grey out “Generate”; the cells before the click |
| `strengthOutOfRange(strength:)`, `strengthTooLow(strength:steps:floor:)` | `strength_out_of_range`, `strength_too_low` | render | bound the slider to `]floor ; 1]` |
| `imageToImageUnsupported(model:)` | `image_to_image_unsupported` | render | Qwen-Image-2.1: pass the image as reference 1 |
| `imageUnreadable(file:)`, `imageEncodingFailed`, `imageWriteFailed(file:)` | `image_unreadable`, `image_encoding_failed`, `image_write_failed` | `ImageRGB(contentsOf:)`, `png()`, `PNG.write` | refuse the image; save elsewhere |
| `unsupportedLoRA(file:reason:)`, `loraForOtherModel(lora:target:model:)` | `unsupported_lora`, `lora_for_other_model` | render (~0.3 s) | offer only `loras(for:)`; otherwise re-import |
| `incompatibleChain(output:input:)` | `incompatible_chain` | `Chain(…)` | programming bug: does not happen with `model.chain` |
| `insufficientMemory(needed:available:)` | `insufficient_memory` | render; `card.checkMemory` before the click | a smaller format, a lighter model, quitting other apps |
| `renderAlreadyRunning` | `render_already_running` | render, while another process renders on the machine | wait, or queue |
| `cancelled` | `cancelled` | the user cancelled (a cancelled `Task` too) | nothing: return to the ready state |
| `settingsAlreadyLoaded(loaded:requested:)` | `settings_already_loaded` | `EngineSettings.load(from:)` after a render | call `load` at launch |
| `internalFailure(component:detail:)` | `internal_failure` | a breakdown deep in the engine (GPU, kernels, arenas, forge) | display, log, retry; an issue if it persists |

The engine **prints nothing**. Its warnings arrive as `.warning` events during a render, or through
`Warnings.outsideRender` outside one. At startup, the app should also display
`EngineSettings.effective.warnings` (an ignored profile, for example). `SILICONED_MEMORY_AVAILABLE_GB`
(GiB) replaces the reading of the machine's state (what it would find before the reserve: `8` leaves
7.0 GB available, the reserve being 1.6 GB), in the engine and in `checkMemory` alike: a tool of measurement and of tests, to
reach `insufficientMemory` or render under another machine's budget — the image stays the same, only
the time changes.

---

## 6. What the API does not do yet

- **CFG and negative prompt**: both models are distilled (Z-Image: a single evaluation per step;
  Qwen-Image-2.1: its turbo LoRA, no CFG). There is no `negative` field.
- **Inpainting / outpainting / masks**, **ControlNet**, IP-Adapter, reference image.
- **Upscaling** (upscaler, *hires fix*); formats beyond 1024×1536.
- **Choice of sampler or schedule**: each model has its own (Euler flow matching; `Sampler` for
  Z-Image).
- Prompt weighting, textual inversion. A batch varies the seed only; different **prompts** are a
  series of renders (`PromptAlternatives`, §3.14), not one batch.
- **Keeping the DiT loaded** between two images or two renders: it is rebuilt every time, to fit in
  16 GB and guarantee bit identity.
- **Importing a non-distilled checkpoint** and rendering it well: the engine has neither CFG nor
  negative prompt. A base (non-Turbo) checkpoint of a known family imports and renders, but badly;
  fine-tunes of the Turbos are the target.
- **GGUF below 4 bits** (Q3_K, Q2_K, I-quants), **4-bit outside GGUF** (nvfp4, SDNQ uint4,
  Nunchaku), mxfp8, fp8 E5M2, asymmetric int8, and LyCORIS / DoRA / `diff` LoRAs: refused at import,
  with the message saying why (§3.11). A quantized file stays as published on disk, but the
  computation is fp32: no int8 or 4-bit matrix product.
- Changing machine profile mid-process (`EngineSettings` is read once; `load` says so by throwing).
- Re-reading a PNG to **replay** an img2img: the source image is not in the metadata (the `image`
  key says so), and the LoRAs appear there only by file name.
- Cancelling **in the middle** of a VAE (encoding, decoding): cancellation waits for the end of the stage (a few seconds at most).
- Not yet checked against the reference: a whole trajectory at a rectangular format; Qwen-Image-2.1
  at 1024² and with two references.
