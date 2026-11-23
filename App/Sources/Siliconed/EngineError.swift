import Foundation
import SiliCore

// **One error leaves the engine: `EngineError`, a closed enumeration.**
//
// Everything that can refuse or fail in the engine is a named case, and every function the app calls that
// throws throws it (`throws(EngineError)`: the compiler holds the door). An app translates it case by
// case: `code` is the stable key of the sentence (`"disk_full"`, …), the associated values are its
// parameters, `errorDescription` is the English source, `recoverySuggestion` what the user can do.
// No case carries a pre-formatted sentence; the only free text is `internalFailure.detail` and
// `importRefused.detail`, technical diagnostics in English, shown under the sentence, never in it.
//
// Inside the package, the readers and kernels keep their own `Failure`s (`Safetensors`, `Tokenizer`…): they say *where* things broke, which the checks and the
// CLI print. They are `package`, and `EngineError(_:)` folds each into its `EngineError` case at the door
// (`EngineError.boundary`). A new internal failure that reaches the door unmapped becomes
// `internalFailure` — a breakdown, never a silent "operation couldn't be completed".

/// **Everything the engine refuses or fails on, by name.**
package enum EngineError: Error, Equatable, Sendable, Codable, LocalizedError, CustomStringConvertible {

    // ── the library: models, files, installation ──────────────────────────────────────────

    /// A model whose maps or published files are not all in the library (`ModelCard.missing(in:)`).
    case modelNotInstalled(model: String)
    /// An identifier that is neither a catalog model nor an imported one.
    case unknownModel(model: String)
    /// The model's license has not been accepted in this library (`Library.acceptLicense(_:)`).
    case licenseNotAccepted(model: String)
    /// A file the engine needs and that is not there (a LoRA removed, an image moved).
    case fileMissing(path: String)
    /// A file that is there and cannot be opened (permissions, a volume gone).
    case fileUnreadable(path: String)
    /// **A map or a published file that does not read**: wrong signature, truncated, header invalid,
    /// sizes that disagree, a map from another version. `file` is `nil` when the reader that refused
    /// did not know it (a shape checked deep in a model's loader). The sha256 is checked at download;
    /// this is what a damaged or stale file looks like afterwards.
    case corruptMap(file: String?)
    /// **Not enough room on a disk**: the library's volume, judged **before the first byte** of an installation or a move;
    /// a download or an import that ran out midway; an image written out (`writing: .image`, `ImagePNG.write`). Bytes; `nil`
    /// when the system did not say. `writing` is what the user can act on: what to free and what to do again differ.
    case diskFull(needed: Int?, available: Int?, writing: DiskWrite = .library)
    /// A download cut midway (network, size received differs from the announced one), or a write that failed on
    /// the way. The files already complete are kept; the next attempt resumes a large file from the chunks it
    /// already wrote (`Downloader.downloadInChunks`), and starts a small one over.
    case downloadInterrupted(file: String)
    /// **Another download is writing the same file** (another installation, the command line): its `.partiel` and
    /// its ledger are locked. Two writers would clobber each other's chunks.
    case downloadInProgress(file: String)
    /// The server refused the file. No repository the app reads is gated and no request carries a token: a 401 or
    /// 403 comes from the way to the Hub (a proxy, a VPN, a firewall), anything else is the server's.
    case downloadRefused(file: String, status: Int)
    /// A download whose sha256 differs from the one the Hub announced: thrown away, start again.
    case downloadCorrupt(file: String)
    /// `ModelImport.importFile` refused the file: neither a LoRA nor a DiT of a known family, a
    /// shape that does not match, a quantized format, an empty name. `detail`: the reader's diagnostic.
    case importRefused(file: String, detail: String)
    /// `ModelImport.remove` on a model that was not imported.
    case notAnImportedModel(model: String)
    /// **An import downloads nothing**: `file` belongs to the family of `model` (an identifier), whose
    /// components — the published DiT's names and shapes, its config, encoder and VAE — are not in the
    /// library. Installing the family (license accepted, size shown) comes first.
    case importNeedsFamily(file: String, model: String)
    /// **A publisher's base model where the family renders its Turbo**: `file` is, byte for byte, a
    /// published base checkpoint (`FamilyTraits.baseCheckpoints`) of the family of `model` (an identifier). It has the
    /// Turbo's names and shapes, so the import would take it — and the Turbo's few steps would ruin its image.
    case importBaseModel(file: String, model: String)
    /// **A model deeper than its family, at the same width**: `file` holds every tensor the family of `model` reads, and
    /// `holds` layers where it reads `reads` (`ModelImport.deeper`). Rendered on its first layers, its image would be
    /// wrong; a model's depth read from its file is to come.
    case importDeeperModel(file: String, model: String, holds: Int, reads: Int)
    /// **The library is on a disk that is not there** — unplugged, a share not mounted (`Library.isReachable`).
    /// Nothing renders, installs or imports until it is back: never another library in its place.
    /// `volume`: the disk's name as the Finder shows it, `nil` when the path does not say.
    case libraryAbsent(path: String, volume: String?)
    /// Where the library cannot go, or a folder that is not one (`LibraryMove`, `Library.opening`).
    case libraryRefused(path: String, reason: LibraryRefusal)

    // ── the request ───────────────────────────────────────────────────────────────────────

    case emptyPrompt
    /// A step count the model refuses. `allowed`: the counts it accepts, empty when any count ≥ 1 is.
    case invalidSteps(steps: Int, allowed: [Int])
    /// More reference images than the model reads (`max` 0: it does no reference editing).
    case tooManyReferences(count: Int, max: Int)
    /// A side under `Format.minimumSide` — the models are out of their domain below. `side`: the
    /// image's smaller side in pixels.
    case imageTooSmall(side: Int, minimum: Int)
    /// A format the engine, or this model, does not render (see `FormatRefusal`). `bound`: the rule
    /// broken, in its own unit — the multiple each side must be, the longest side in pixels, the
    /// largest area in pixels.
    case formatRefused(width: Int, height: Int, reason: FormatRefusal, bound: Int)
    /// A format typed by hand that does not parse (`512`, `1024`, `832x1216`).
    case formatUnreadable(text: String)
    /// A prompt whose alternatives (`{a|b}`, `PromptAlternatives`) do not parse (see `PromptRefusal`).
    /// `position`: the offending brace, in characters from 1.
    case promptSyntax(position: Int, reason: PromptRefusal)
    /// A prompt whose alternatives make more prompts than `max` (`PromptAlternatives.maxCombinations`).
    case tooManyPromptVariants(count: Int, max: Int)
    /// A prompt that spells a text the model's encoder reserves (`text`: `<|image_pad|>`, the place of a
    /// reference image, for an encoder that reads pictures), refused before any weight is read.
    case promptReservedText(text: String)
    /// A prompt longer than the model reads: `tokens` counted, `max` at most (`Request.maximumPromptTokens`).
    case promptTooLong(tokens: Int, max: Int)
    /// An XY grid (`RenderGrid`) whose axes do not make a grid (see `GridRefusal`).
    case gridRefused(reason: GridRefusal)
    /// An XY grid of more cells than `max` (`RenderGrid.maxCells`): one render per cell.
    case tooManyGridCells(count: Int, max: Int)
    /// An img2img strength outside `]0, 1]`.
    case strengthOutOfRange(strength: Double)
    /// An img2img strength that leaves no evaluation: it must exceed `floor`.
    case strengthTooLow(strength: Double, steps: Int, floor: Double)
    /// A starting image given to a model without img2img (`DenoisingModule.acceptsStartImage`).
    case imageToImageUnsupported(model: String)
    /// A `Solver` the model's sampler does not walk (`DenoisingModule.solvers`).
    case solverUnsupported(solver: String, model: String)
    /// More or Most detail asked of a model that does not offer it (`DenoisingModule.acceptsDetail`).
    case detailUnsupported(model: String)
    /// An image ImageIO cannot read.
    case imageUnreadable(file: String)
    /// ImageIO refused to encode the PNG.
    case imageEncodingFailed
    /// The PNG could not be written there.
    case imageWriteFailed(file: String)
    /// A LoRA map the engine cannot apply (see `LoRARefusal`).
    case unsupportedLoRA(file: String, reason: LoRARefusal)
    /// A LoRA made for another model: `target` is the one it was made for.
    case loraForOtherModel(lora: String, target: String, model: String)
    /// A hand-composed `Chain` whose wires do not match: `output` comes out where `input` is expected.
    case incompatibleChain(output: String, input: String)
    /// **The model is not translated for the engine yet** (no family of its own): installed, recognized, and not rendered.
    case notOnEngine(model: String)

    // ── the machine ───────────────────────────────────────────────────────────────────────

    /// **Beyond this Mac's budget**: the engine found no plan of the render, or of the writer, under B (¾ · RAM − 2 GiB by default).
    /// The same request on the same Mac is always refused. Bytes.
    case insufficientMemory(needed: Int, budget: Int)
    /// Another process is rendering on this machine (the app, the CLI, whatever their library or
    /// user — `RenderLock.machine`): two renders would each be planned under the whole budget. Within one
    /// process, renders queue instead.
    case renderAlreadyRunning
    /// **The engine's process stopped** (`EngineWorker`) in the middle of a render or a writer's gesture: killed past B
    /// (`budget`: B, in bytes), or stopped by anything else (`nil`). `detail` is technical: who stopped it, what it held.
    /// The app lives on; the next render starts another engine.
    case engineStopped(budget: Int?, detail: String)
    /// **The engine's process is of another build** (`WorkerHello`): `app`, the protocol the app speaks; `engine`, the
    /// worker's, `nil` when it said none (a build from before the hello). Refused before any call.
    case engineMismatch(app: Int, engine: Int?)
    /// **The engine's process said nothing for `seconds`** with a call in hand (`EngineWorker.silence`): killed, the
    /// next call starts another.
    case engineSilent(seconds: Int)
    /// The render, the installation or the download was cancelled.
    case cancelled
    /// **The render went wrong and its image is unusable**: a non-finite value (NaN, ±∞) in the final latent
    /// (`stage` `.denoising`: the DiT, or what it read) or in the decoded image (`.decoding`: the VAE) — an image of
    /// that kind comes out black or blank. Thrown instead of delivering it as done; nothing of it is kept. Not a
    /// misuse: the same request rendered again is the remedy, and an issue if it comes back. The stage is said
    /// because the user sees it: a NaN latent once showed as the VAE's failure (issue 17), the decoding having
    /// been announced before the latent was checked.
    case nonFiniteImage(stage: Engine.Stage)
    /// A breakdown deep in the engine's numerics: not a misuse.
    /// `component` names where, `detail` is the technical message.
    case internalFailure(component: String, detail: String)

    /// Why a format is refused — `code` refines into `format_refused.<rawValue>`.
    package enum FormatRefusal: String, Sendable, Equatable, Codable, CaseIterable {
        /// A side that is not a multiple of the model's `Format.multiple(for:)` (its VAE's factor times its patch).
        case notMultiple = "not_multiple"
        /// A side longer than the model computes (`Format.maximumSide(for:)`: a position table's reach).
        case sideTooLong = "side_too_long"
    }

    /// Why a prompt's alternatives do not parse — `code` refines into `prompt_syntax.<rawValue>`.
    package enum PromptRefusal: String, Sendable, Equatable, Codable, CaseIterable {
        /// A `{` that no `}` closes.
        case unclosedGroup = "unclosed_group"
        /// A `}` that closes no group.
        case unmatchedClose = "unmatched_close"
        /// A `{` inside a group: groups do not nest.
        case nestedGroup = "nested_group"
    }

    /// Why an XY grid is refused — `code` refines into `grid_refused.<rawValue>`.
    package enum GridRefusal: String, Sendable, Equatable, Codable, CaseIterable {
        /// An axis without a value.
        case emptyAxis = "empty_axis"
        /// X and Y vary the same setting (the seed twice, the same group, the same LoRA).
        case sameAxisTwice = "same_axis_twice"
        /// An axis names a group of alternatives the prompt does not have.
        case noSuchGroup = "no_such_group"
        /// An axis names a LoRA the stack does not have.
        case noSuchLoRA = "no_such_lora"
        /// A group of alternatives of the prompt is on neither axis: it would stack several images
        /// in each cell.
        case groupNotOnAxis = "group_not_on_axis"
        /// A value no render takes: steps under 1, a strength that is not a number, no seed.
        case invalidValue = "invalid_value"
        /// An axis written in a way `RenderGrid.axis(_:loras:)` does not read.
        case unreadableAxis = "unreadable_axis"
        /// A model axis under a LoRA stack: a LoRA is made for one model.
        case loraAcrossModels = "lora_across_models"
        /// A start strength axis without a starting image.
        case noStartImage = "no_start_image"
    }

    /// Why a library cannot go somewhere, or be opened — `code` refines into `library_refused.<rawValue>`.
    package enum LibraryRefusal: String, Sendable, Equatable, Codable, CaseIterable {
        /// An MS-DOS (FAT32) disk: no file over 4 GB, and a model's map is tens of GB. exFAT is fine.
        case fileSizeLimit = "file_size_limit"
        /// A disk, or a folder, that cannot be written to.
        case readOnly = "read_only"
        /// « Open a library »: the folder holds no `store/`.
        case notALibrary = "not_a_library"
        /// « Move the library »: the folder already holds another library's models — to be opened
        /// instead, never merged.
        case alreadyALibrary = "already_a_library"
        /// The folder is the library itself, inside it, or holds it.
        case sameLibrary = "same_library"
        /// A file read back from the new place differs from the original: nothing was switched, the
        /// original is untouched.
        case copyDamaged = "copy_damaged"
    }

    /// What a full disk stopped (`diskFull`) — the code stays `disk_full`: the sentence is the same, the way out is not.
    package enum DiskWrite: String, Sendable, Equatable, Codable, CaseIterable {
        /// Files into the library: an installation, a download, an import, a move.
        case library
        /// An image out of the app: a PNG saved, exported, or written by Save Every Image To….
        case image
    }

    /// Why a LoRA is refused — `code` refines into `unsupported_lora.<rawValue>`.
    package enum LoRARefusal: String, Sendable, Equatable, Codable, CaseIterable {
        /// The map is a model, not an adapter.
        case notALoRA = "not_a_lora"
        /// No family's graph has every module the file adapts, at its shape (recognition is by tensors).
        case noTarget = "no_target"
        /// Its two halves do not compose (rank or shape): damaged.
        case inconsistent = "inconsistent"
        /// A kind of adapter the graph does not apply: LoKr, LoHa, DoRA, rsLoRA, a text encoder's, PEFT's ranks or
        /// alphas a module (`LoRAFile`).
        case unsupportedKind = "unsupported_kind"
    }
}
