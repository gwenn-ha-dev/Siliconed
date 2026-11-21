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

    // ── what an app reads ─────────────────────────────────────────────────────────────────

    /// **The stable key of the case** — what an app looks its translated sentence up by. Never
    /// renamed once published; a new case takes a new code.
    package var code: String {
        switch self {
        case .modelNotInstalled: return "model_not_installed"
        case .unknownModel: return "unknown_model"
        case .licenseNotAccepted: return "license_not_accepted"
        case .fileMissing: return "file_missing"
        case .fileUnreadable: return "file_unreadable"
        case .corruptMap: return "corrupt_map"
        case .diskFull: return "disk_full"
        case .downloadInterrupted: return "download_interrupted"
        case .downloadRefused: return "download_refused"
        case .downloadCorrupt: return "download_corrupt"
        case .downloadInProgress: return "download_in_progress"
        case .importRefused: return "import_refused"
        case .notAnImportedModel: return "not_an_imported_model"
        case .importNeedsFamily: return "import_needs_family"
        case .importBaseModel: return "import_base_model"
        case .libraryAbsent: return "library_absent"
        case .libraryRefused: return "library_refused"
        case .emptyPrompt: return "empty_prompt"
        case .invalidSteps: return "invalid_steps"
        case .tooManyReferences: return "too_many_references"
        case .imageTooSmall: return "image_too_small"
        case .formatRefused: return "format_refused"
        case .formatUnreadable: return "format_unreadable"
        case .promptSyntax: return "prompt_syntax"
        case .tooManyPromptVariants: return "too_many_prompt_variants"
        case .promptReservedText: return "prompt_reserved_text"
        case .promptTooLong: return "prompt_too_long"
        case .gridRefused: return "grid_refused"
        case .tooManyGridCells: return "too_many_grid_cells"
        case .strengthOutOfRange: return "strength_out_of_range"
        case .strengthTooLow: return "strength_too_low"
        case .imageToImageUnsupported: return "image_to_image_unsupported"
        case .solverUnsupported: return "solver_unsupported"
        case .detailUnsupported: return "detail_unsupported"
        case .imageUnreadable: return "image_unreadable"
        case .imageEncodingFailed: return "image_encoding_failed"
        case .imageWriteFailed: return "image_write_failed"
        case .unsupportedLoRA: return "unsupported_lora"
        case .loraForOtherModel: return "lora_for_other_model"
        case .incompatibleChain: return "incompatible_chain"
        case .notOnEngine: return "not_on_engine"
        case .insufficientMemory: return "insufficient_memory"
        case .renderAlreadyRunning: return "render_already_running"
        case .engineStopped: return "engine_stopped"
        case .engineMismatch: return "engine_mismatch"
        case .engineSilent: return "engine_silent"
        case .cancelled: return "cancelled"
        case .nonFiniteImage: return "non_finite_image"
        case .internalFailure: return "internal_failure"
        }
    }

    package var errorDescription: String? {
        switch self {
        case .modelNotInstalled(let model): return "The model \(model) is not installed."
        case .unknownModel(let model): return "There is no model called \(model)."
        case .licenseNotAccepted(let model): return "The license of \(model) has not been accepted."
        case .fileMissing(let path): return "A file is missing: \(path)."
        case .fileUnreadable(let path): return "A file cannot be opened: \(path)."
        case .corruptMap(let file):
            return "A model file is damaged or from another version" + (file.map { ": \($0)." } ?? ".")
        case let .diskFull(needed, available, _):
            switch (needed, available) {
            case let (n?, a?): return "Not enough disk space: \(gigabytes(n)) needed, \(gigabytes(a)) available."
            case let (n?, nil): return "Not enough disk space: \(gigabytes(n)) needed."
            default: return "The disk is full."
            }
        case .downloadInterrupted(let file): return "The download of \(file) was interrupted."
        case let .downloadRefused(file, status):
            return "The download of \(file) was refused by the server (HTTP \(status))."
        case .downloadCorrupt(let file): return "The download of \(file) is corrupt (sha256 mismatch)."
        case .downloadInProgress(let file): return "\(file) is already being downloaded by another installation on this Mac."
        case .importRefused(let file, _): return "This file cannot be imported: \(file)."
        case .notAnImportedModel(let model): return "\(model) is not an imported model."
        case let .importNeedsFamily(file, model): return "To import \(file), install \(model) first."
        case let .importBaseModel(file, model):
            let card = ModelCard.withID(model)
            return "\(file) is the base model of \(card?.family.name ?? model): Siliconed renders \(card?.name ?? model)."
        case let .libraryAbsent(path, volume):
            return volume.map { "The library is on \($0), which is not connected." } ?? "The library cannot be found: \(path)."
        case let .libraryRefused(path, reason):
            switch reason {
            case .fileSizeLimit: return "This disk cannot hold the models: its format (MS-DOS FAT) stops at 4 GB per file."
            case .readOnly: return "This folder cannot be written to: \(path)."
            case .notALibrary: return "This folder is not a Siliconed library: \(path)."
            case .alreadyALibrary: return "This folder already holds a Siliconed library: \(path)."
            case .sameLibrary: return "This folder is the library itself, or inside it: \(path)."
            case .copyDamaged: return "A file read back from the new place differs from the original: \(path)."
            }
        case .emptyPrompt: return "The prompt is empty."
        case let .invalidSteps(steps, allowed):
            return allowed.isEmpty ? "\(steps) steps: a render takes between 1 and \(Request.maximumSteps)."
                : "This model does not render in \(steps) steps."
        case let .tooManyReferences(count, max):
            return max == 0 ? "This model does not edit from reference images."
                : "\(count) reference images: this model reads at most \(max)."
        case let .imageTooSmall(side, minimum):
            return "The image is too small: \(side) px on a side, \(minimum) px at least."
        case let .formatRefused(width, height, reason, bound):
            switch reason {
            case .notMultiple: return "\(width)×\(height): each side must be a multiple of \(bound)."
            case .sideTooLong: return "\(width)×\(height): this model draws \(bound) px at most on a side."
            }
        case .formatUnreadable(let text): return "\(text) is not a format."
        case let .promptSyntax(position, reason):
            switch reason {
            case .unclosedGroup: return "The prompt opens a group of alternatives that is never closed (character \(position))."
            case .unmatchedClose: return "The prompt closes a group of alternatives that was never opened (character \(position))."
            case .nestedGroup: return "Groups of alternatives cannot be nested (character \(position))."
            }
        case let .tooManyPromptVariants(count, max):
            return "The prompt's alternatives make \(count) prompts: \(max) at most."
        case .promptReservedText(let text):
            return "The prompt contains \(text), a control token of the model's text encoder."
        case let .promptTooLong(tokens, max):
            return "The prompt is \(tokens) tokens long: this model reads at most \(max)."
        case .gridRefused(let reason):
            switch reason {
            case .emptyAxis: return "An axis of the grid has no value."
            case .sameAxisTwice: return "The grid's two axes vary the same setting."
            case .noSuchGroup: return "An axis of the grid names a group of alternatives the prompt does not have."
            case .noSuchLoRA: return "An axis of the grid names a LoRA that is not in the stack."
            case .groupNotOnAxis: return "The prompt has a group of alternatives that is not an axis of the grid."
            case .invalidValue: return "An axis of the grid has a value no render takes."
            case .unreadableAxis: return "An axis of the grid cannot be read."
            case .loraAcrossModels: return "A grid that varies the model cannot take a LoRA: each LoRA is made for one model."
            case .noStartImage: return "A grid that varies the starting image's strength needs a starting image."
            }
        case let .tooManyGridCells(count, max):
            return "The grid makes \(count) cells: \(max) at most."
        case .strengthOutOfRange(let strength):
            return "The strength must be above 0 and at most 1 (\(strengthText(strength)))."
        case let .strengthTooLow(strength, steps, floor):
            return "A strength of \(strengthText(strength)) leaves no step out of \(steps): "
                + "it must be above \(String(format: "%.3f", floor))."
        case .imageToImageUnsupported(let model): return "\(model) does not start from an image."
        case let .solverUnsupported(solver, model): return "\(model) does not offer the \(solver) solver."
        case .detailUnsupported(let model): return "\(model) does not offer Detail."
        case .imageUnreadable(let file): return "This image cannot be read: \(file)."
        case .imageEncodingFailed: return "The image could not be encoded as PNG."
        case .imageWriteFailed(let file): return "The image could not be saved to \(file)."
        case let .unsupportedLoRA(file, reason):
            switch reason {
            case .notALoRA: return "\(file) is a model, not a LoRA."
            case .noTarget: return "The LoRA \(file) adapts modules that no model of this app has."
            case .inconsistent: return "The LoRA \(file) is damaged."
            case .unsupportedKind:
                return "The LoRA \(file) is of a kind this app does not apply (LoKr, LoHa, DoRA, rsLoRA, a text encoder's, or ranks that vary by module)."
            }
        case let .loraForOtherModel(lora, target, model):
            return "The LoRA \(lora) is for \(target), not for \(model)."
        case let .incompatibleChain(output, input):
            return "These modules do not connect: \(output) comes out where \(input) is expected."
        case .notOnEngine(let model):
            return "\(model) does not run on this engine yet."
        case let .insufficientMemory(needed, budget):
            // Said of a render and of a rewrite alike: the sentence names neither. `needed` within the budget: every form
            // stopped the fuse (`Stage.choose`), and the one number known is the soberest's socle — not what it needs.
            if needed <= budget { return "Past this Mac's memory limit of \(gigabytes(budget)) every time it was tried: refused." }
            return "Beyond this Mac's memory limit: \(gigabytes(needed)) needed, this Mac offers \(gigabytes(budget))."
        case .renderAlreadyRunning: return "Another render is already running on this machine."
        case let .engineStopped(budget?, _): return "The engine went past this Mac's memory limit of \(gigabytes(budget)) and was stopped."
        case .engineStopped: return "The engine stopped unexpectedly."
        case let .engineMismatch(app, engine):
            return "The engine is from another build than the app (protocol \(engine.map(String.init) ?? "none"), the app's \(app))."
        case .engineSilent(let seconds): return "The engine stopped answering for \(seconds / 60) minutes and was stopped."
        case .cancelled: return "Cancelled."
        case .nonFiniteImage(.decoding): return "This render went wrong in the VAE and its image is unusable: it was not kept."
        case .nonFiniteImage: return "This render went wrong in the DiT and its image is unusable: it was not kept."
        case let .internalFailure(component, detail): return "Internal engine failure (\(component)): \(detail)"
        }
    }

    package var recoverySuggestion: String? {
        switch self {
        case .modelNotInstalled: return "Install the model, or choose the library it is installed in."
        case .unknownModel: return "Choose a model from the list of installed models, or install it first."
        case .licenseNotAccepted: return "Read and accept the model's license before rendering."
        case .fileMissing: return "Choose the file again, or reinstall the model it belongs to."
        case .fileUnreadable: return "Check the file's permissions and that its volume is connected."
        case .corruptMap: return "Reinstall the model, or import it again."
        case .diskFull(_, _, .image): return "Make room on the disk (uninstall a model, or delete images from the folder they are saved to), then render again."
        case .diskFull: return "Make room (uninstall a model, or discard the interrupted downloads), then start again: the files already downloaded are kept, and a large file resumes where it stopped."
        case .downloadInterrupted: return "Check the connection and start again: the files already downloaded are kept, and a large file resumes where it stopped."
        case let .downloadRefused(_, status) where status == 401 || status == 403:
            return "Check that huggingface.co opens in a browser on this Mac (a proxy, a VPN or a firewall can refuse it), then install again: the files already downloaded are kept."
        case .downloadRefused: return "Try again later."
        case .downloadCorrupt: return "Start again: the damaged file was thrown away."
        case .downloadInProgress: return "Wait for that download to finish, then start again."
        case .importRefused: return "Import a LoRA or a full model of a supported family, as .safetensors."
        case .notAnImportedModel: return nil
        case .importNeedsFamily: return "Install the model, then import the file again: an import downloads nothing."
        case .importBaseModel(_, let model): return "Install \(ModelCard.withID(model)?.name ?? model), or import a model made from it."
        case .libraryAbsent: return "Connect the disk, or go back to this Mac's library in the settings."
        case .libraryRefused(_, .fileSizeLimit): return "Choose a disk formatted APFS, Mac OS Extended or exFAT."
        case .libraryRefused(_, .readOnly): return "Choose a folder you can write to."
        case .libraryRefused(_, .notALibrary): return "Choose the folder that holds the library's store folder, or move the library there."
        case .libraryRefused(_, .alreadyALibrary): return "Open it instead, or choose an empty folder."
        case .libraryRefused(_, .sameLibrary): return "Choose a folder outside the library."
        case .libraryRefused(_, .copyDamaged): return "Check the disk and its cable, then start again: the files already checked are kept."
        case .emptyPrompt: return "Write a prompt."
        case let .invalidSteps(_, allowed) where !allowed.isEmpty:
            return "Choose " + allowed.map(String.init).joined(separator: ", ") + " steps."
        case .invalidSteps: return "Leave the number of steps at the model's default."
        case .tooManyReferences(_, let max) where max == 0:
            return "Choose a model that edits by reference."
        case .tooManyReferences(_, let max): return "Keep at most \(max) reference images."
        case .promptSyntax:
            return "Write alternatives as {a|b|c}, without nesting; \\{, \\} and \\| are the characters themselves."
        case .tooManyPromptVariants: return "Remove a group, or some alternatives."
        case .promptReservedText(let text): return "Remove \(text) from the prompt."
        case .promptTooLong: return "Shorten the prompt."
        case .gridRefused(let reason):
            switch reason {
            case .groupNotOnAxis: return "Make every group an axis, or write its braces as \\{ and \\}."
            case .sameAxisTwice: return "Choose another setting for one of the axes."
            case .invalidValue: return "Steps are integers from 1, strengths numbers, seeds at least one."
            case .unreadableAxis: return "Write an axis as prompt[:N], seed=42,43, seeds=4, steps=6,8, lora[:NAME]=0.4,0.8, loras[:STRENGTH]=none,flat,ink, model=" + Family.allCases.prefix(2).map(\.rawValue).joined(separator: ",") + ", image, solver=euler,dpmpp-2m, start=0.45,0.6 or detail=normal,more."
            case .emptyAxis, .noSuchGroup, .noSuchLoRA: return "Give each axis at least one value, from the prompt and the stack."
            case .loraAcrossModels: return "Remove the LoRAs, or vary something other than the model."
            case .noStartImage: return "Start from an image (--image), or vary something else."
            }
        case .tooManyGridCells: return "Remove values from an axis."
        case .imageTooSmall, .formatRefused, .formatUnreadable:
            return "Choose one of the formats the model offers, for example 1024×1024 or 832×1216."
        case .strengthOutOfRange: return "A strength between 0.2 and 0.8 keeps the image while transforming it; 0.6 by default."
        case .strengthTooLow: return "Increase the strength, or the number of steps."
        case .imageToImageUnsupported: return "Pass the image as reference 1 and describe the change."
        case .solverUnsupported: return "Choose Euler."
        case .detailUnsupported: return "Choose Normal."
        case .imageUnreadable: return "Choose a PNG, JPEG or HEIC image."
        case .imageEncodingFailed: return nil
        case .imageWriteFailed: return "Check that the folder exists and that you can write to it."
        case .unsupportedLoRA(_, .notALoRA): return "Import it as a model instead."
        case .unsupportedLoRA(_, .noTarget): return "Choose a LoRA made for one of the app's models."
        case .unsupportedLoRA(_, .unsupportedKind): return "Choose a plain LoRA (lora_A and lora_B, or lora_down and lora_up) made for the model's DiT."
        case .unsupportedLoRA: return "Import the LoRA again from its .safetensors."
        case .loraForOtherModel: return "Choose a LoRA made for this model, or remove this one from the stack."
        case .incompatibleChain: return "Render with the model as installed, rather than modules assembled by hand."
        case .notOnEngine: return "Choose another model."
        case .insufficientMemory: return "Choose a smaller format, fewer reference images or a lighter model, or raise the memory limit in Settings › Macs."
        case .renderAlreadyRunning: return "Wait for it to finish, or add the request to the app's queue."
        case .engineStopped(_?, _): return "Render again: the engine starts afresh. If it stops again, choose a smaller size or a lighter model."
        case .engineStopped: return "Render again: the engine starts afresh. If it happens again, open an issue with this message."
        case .engineMismatch: return "Install Siliconed again: the app and its engine come from the same build."
        case .engineSilent: return "Render again: the engine starts afresh. If it happens again, open an issue with this message."
        case .cancelled: return nil
        case .nonFiniteImage: return "Render it again; if it happens again, open an issue."
        case .internalFailure: return "Try again; if it persists, open an issue with this message."
        }
    }

    /// The sentence and its suggestion — what the CLI prints; an import's diagnostic follows it.
    package var description: String {
        let text = [errorDescription, recoverySuggestion].compactMap { $0 }.joined(separator: " ")
        if case let .importRefused(_, detail) = self { return text + " (\(detail))" }
        if case let .engineStopped(_, detail) = self { return text + " (\(detail))" }
        return text
    }

    // ── the door: every internal failure, folded into its case ────────────────────────────

    /// **The fold.** An `EngineError` comes back as is; each internal failure becomes the case an
    /// app can name; anything else is an `internalFailure`.
    /// **A write that ran out of room**: the disk, said as such (`diskFull`), never a write refused.
    static func ranOutOfRoom(_ error: any Error) -> Bool {
        let ns = error as NSError
        return (ns.domain == NSPOSIXErrorDomain && ns.code == Int(ENOSPC))
            || (ns.domain == NSCocoaErrorDomain && ns.code == NSFileWriteOutOfSpaceError)
    }

    package init(_ error: any Error) {
        switch error {
        case let e as EngineError: self = e
        case is CancellationError: self = .cancelled
        // The engine's refusal under B, before it allocated: a lack of memory, with its numbers.
        case let e as OverBudget: self = .insufficientMemory(needed: e.needed, budget: e.budget)
        case let e as MissingFile: self = .fileMissing(path: e.path)
        case let e as Request.Failure:
            switch e {
            case .emptyPrompt: self = .emptyPrompt
            case .steps(let n): self = .invalidSteps(steps: n, allowed: [])
            case let .references(n, max): self = .tooManyReferences(count: n, max: max)
            case let .solver(solver, model): self = .solverUnsupported(solver: solver.rawValue, model: model)
            case .detail(let model): self = .detailUnsupported(model: model)
            case let .promptTooLong(tokens, max): self = .promptTooLong(tokens: tokens, max: max)
            }
        case let e as Strength.Failure:
            switch e {
            case .outOfBounds(let s): self = .strengthOutOfRange(strength: s)
            case let .noEvaluation(s, steps, floor): self = .strengthTooLow(strength: s, steps: steps, floor: floor)
            case .withoutEncoder(let model): self = .imageToImageUnsupported(model: model)
            }
        case let e as Model.Failure:
            switch e { case .unknown(let name): self = .unknownModel(model: name) }
        case let e as ImageRGB.Failure:
            switch e {
            case .unreadable(let file): self = .imageUnreadable(file: file)
            }
        case let e as Chain.Failure:
            switch e {
            case let .incompatibleText(output, entry): self = .incompatibleChain(output: "\(output)", input: "\(entry)")
            case let .latentIncompatible(output, entry), let .imageIncompatible(output, entry):
                self = .incompatibleChain(output: "\(output)", input: "\(entry)")
            case .textWithoutImages(let format): self = .incompatibleChain(output: "text", input: "\(format)")
            }
        case let e as Safetensors.Failure:
            switch e {
            case let .cannotOpen(path, code): self = code == ENOENT ? .fileMissing(path: path) : .fileUnreadable(path: path)
            case .cannotStat(let path, _), .cannotMap(let path, _): self = .fileUnreadable(path: path)
            case .missingTensor(let file, _): self = .corruptMap(file: file)
            case .badHeader, .unknownDType: self = .corruptMap(file: nil)
            }
        case let e as Tokenizer.Failure:
            switch e {
            case .fileNotFound(let path): self = .fileMissing(path: path)
            case .malformed, .unsupported: self = .corruptMap(file: nil)
            }
        case let e as Qwen3VLChat.Failure:
            switch e { case .reservedText(let text): self = .promptReservedText(text: text) }
        case is T5Tokenizer.Failure, is OrderedJSON.ParseError: self = .corruptMap(file: nil)
        default:
            if Self.ranOutOfRoom(error) {
                self = .diskFull(needed: nil, available: nil)
            } else if let url = error as? URLError {
                self = .downloadInterrupted(file: url.failingURL?.lastPathComponent ?? "?")
            } else {
                self = .internalFailure(component: "\(type(of: error))", detail: "\(error)")
            }
        }
    }

    /// **The door**: runs `body` and throws only `EngineError`. Every function the app calls that throws
    /// passes through it (or throws an `EngineError` itself).
    package static func boundary<T>(_ body: () throws -> T) throws(EngineError) -> T {
        do { return try body() } catch { throw EngineError(error) }
    }

    /// Every case once, with plausible values — the test target checks that the codes are unique
    /// and that each case has an English sentence. **A new case is added here too.**
    package static let samples: [EngineError] = [
        .modelNotInstalled(model: "z-image"), .unknownModel(model: "sd15"), .licenseNotAccepted(model: "qwen-image-2.1"),
        .fileMissing(path: "/a.safetensors"), .fileUnreadable(path: "/a.safetensors"), .corruptMap(file: "/a.safetensors"),
        .diskFull(needed: 30_000_000_000, available: 1_000_000_000), .downloadInterrupted(file: "vae.safetensors"),
        .downloadRefused(file: "vae.safetensors", status: 403), .downloadCorrupt(file: "vae.safetensors"),
        .downloadInProgress(file: "vae.safetensors"),
        .importRefused(file: "x.safetensors", detail: "not recognized"), .notAnImportedModel(model: "Z-Image Turbo"),
        .importNeedsFamily(file: "x.safetensors", model: "anima"),
        .importBaseModel(file: "qwen_image_2.1_bf16.safetensors", model: "qwen-image-2.1"),
        .libraryAbsent(path: "/Volumes/Models/Siliconed", volume: "Models"),
        .libraryRefused(path: "/Volumes/KEY/Siliconed", reason: .fileSizeLimit),
        .emptyPrompt, .invalidSteps(steps: 8, allowed: [5, 6, 7, 9]), .tooManyReferences(count: 4, max: 3),
        .imageTooSmall(side: 256, minimum: 512), .formatRefused(width: 1000, height: 1000, reason: .notMultiple, bound: 16),
        .formatUnreadable(text: "big"), .promptSyntax(position: 12, reason: .unclosedGroup),
        .tooManyPromptVariants(count: 128, max: 64), .promptReservedText(text: "<|image_pad|>"),
        .promptTooLong(tokens: 600, max: 512), .gridRefused(reason: .groupNotOnAxis),
        .tooManyGridCells(count: 81, max: 64), .strengthOutOfRange(strength: 1.5),
        .strengthTooLow(strength: 0.1, steps: 8, floor: 0.125), .imageToImageUnsupported(model: "qwen-image-2.1"),
        .solverUnsupported(solver: "unipc", model: "qwen-image-2.1"),
        .detailUnsupported(model: "anima"),
        .imageUnreadable(file: "/a.png"), .imageEncodingFailed, .imageWriteFailed(file: "/a.png"),
        .unsupportedLoRA(file: "x.safetensors", reason: .noTarget),
        .loraForOtherModel(lora: "PopArt", target: "z-image", model: "krea2"),
        .incompatibleChain(output: "flux", input: "qwen-image"), .notOnEngine(model: "anima"),
        .insufficientMemory(needed: 7_800_000_000, budget: 8_589_934_592), 
        .renderAlreadyRunning, .engineStopped(budget: 4 << 30, detail: "silicore-worker held 4.12 GiB, past B = 4.00 GiB"),
        .engineMismatch(app: 2, engine: 1), .engineSilent(seconds: 600),
        .cancelled, .nonFiniteImage(stage: .decoding),
        .internalFailure(component: "engine", detail: "plan refused"),
    ]
}

private func gigabytes(_ bytes: Int) -> String { String(format: "%.1f GB", Double(bytes) / 1e9) }

/// A strength as typed: enough digits that a value just inside or outside the bounds does not
/// print as one of them (`0.001` is not `0.00`), without trailing zeros.
private func strengthText(_ value: Double) -> String {
    guard value.isFinite else { return "\(value)" }
    var text = String(format: "%.6f", value)
    while text.hasSuffix("0") { text.removeLast() }
    if text.hasSuffix(".") { text.removeLast() }
    return text == "-0" ? "0" : text
}

/// **A missing file — internal to the package**: a weights file from `store/`, a component
/// installed under `store/composants/` (tokenizer, VAE, configs). Distinct from a file that is
/// present but unreadable or damaged, which throws its reader's error. At the door it becomes
/// `EngineError.fileMissing`, or `modelNotInstalled` when it is a model being built (`Model.named`).
package struct MissingFile: Error, CustomStringConvertible, Equatable {
    package enum Nature: Sendable, Equatable {
        /// Under `store/`: a weights file — it is installed.
        case map
        /// Under `store/composants/`: a published file of a family (tokenizer, VAE) — it is installed.
        case published
        /// A golden tensor of a check (`goldens-*.safetensors`, beside the oracles): it is not
        /// installed, the reference writes it, by its oracle.
        case golden
    }
    package let path: String
    package let nature: Nature
    /// A golden is recognized by its name whoever reads it (`Safetensors` opens goldens and
    /// published files alike, and only knows it was asked for a file).
    package init(_ path: String, _ nature: Nature) {
        self.path = path
        self.nature = Self.isGolden(path) ? .golden : nature
    }

    package static func isGolden(_ path: String) -> Bool {
        let name = (path as NSString).lastPathComponent
        return name.hasPrefix("goldens-") && name.hasSuffix(".safetensors")
    }

    package var description: String {
        switch nature {
        case .map: return "map missing: \(path) — install the model or import it"
        case .published: return "component missing: \(path) — install the model"
        case .golden:
            return "golden missing: \(path) — goldens are written by the reference oracle, not installed "
                + "(the .json of the same name records the command that made it)"
        }
    }
}
