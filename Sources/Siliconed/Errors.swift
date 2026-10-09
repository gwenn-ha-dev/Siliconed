import Foundation

// **One public error: `EngineError`, a closed enumeration.**
//
// Everything that can refuse or fail in the engine is a named case, and every public function that
// throws throws it (`throws(EngineError)`: the compiler holds the door). An app translates it case by
// case: `code` is the stable key of the sentence (`"disk_full"`, …), the associated values are its
// parameters, `errorDescription` is the English source, `recoverySuggestion` what the user can do.
// No case carries a pre-formatted sentence; the only free text is `internalFailure.detail` and
// `importRefused.detail`, technical diagnostics in English, shown under the sentence, never in it.
//
// Inside the package, the readers and kernels keep their own `Failure`s (`Artifact`, `Safetensors`,
// `GEMM`, `Arena`, the forge's `Numerics`…): they say *where* things broke, which the checks and the
// CLI print. They are `package`, and `EngineError(_:)` folds each into its public case at the door
// (`EngineError.boundary`). A new internal failure that reaches the door unmapped becomes
// `internalFailure` — a breakdown, never a silent "operation couldn't be completed".

/// **Everything the engine refuses or fails on, by name.**
public enum EngineError: Error, Equatable, Sendable, LocalizedError, CustomStringConvertible {

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
    /// Not enough room on the library's volume, judged **before the first byte** of an installation
    /// (or by the K/V cache of an edit). Bytes; `nil` when the system did not say.
    case diskFull(needed: Int?, available: Int?)
    /// A download cut midway (network, size received differs from the announced one). The files
    /// already complete are kept; the interrupted one starts over on the next attempt.
    case downloadInterrupted(file: String)
    /// The server refused the file: `status` 401/403 is a gated repository (accept its license on
    /// huggingface.co and set `HF_TOKEN`), anything else is the server's.
    case downloadRefused(file: String, status: Int)
    /// A download whose sha256 differs from the one the Hub announced: thrown away, start again.
    case downloadCorrupt(file: String)
    /// `ModelImport.importFile` refused the file: neither a LoRA nor a DiT of a known family, a
    /// shape that does not match, a quantized format, an empty name. `detail`: the forge's diagnostic.
    case importRefused(file: String, detail: String)
    /// `ModelImport.remove` on a model that was not imported.
    case notAnImportedModel(model: String)

    // ── the request ───────────────────────────────────────────────────────────────────────

    case emptyPrompt
    /// A step count the model refuses. `allowed`: the counts it accepts, empty when any count ≥ 1 is.
    case invalidSteps(steps: Int, allowed: [Int])
    /// More reference images than the model reads (`max` 0: it does no reference editing).
    case tooManyReferences(count: Int, max: Int)
    /// A side under `Format.minimumSide` — the models are out of their domain below. `side`: the
    /// smaller side in pixels (of the image, or of an evaluation a forced spectral schedule would make).
    case imageTooSmall(side: Int, minimum: Int)
    /// A format the engine does not render (see `FormatRefusal`).
    case formatRefused(width: Int, height: Int, reason: FormatRefusal)
    /// A format typed by hand that does not parse (`512`, `1024`, `832x1216`).
    case formatUnreadable(text: String)
    /// A prompt whose alternatives (`{a|b}`, `PromptAlternatives`) do not parse (see `PromptRefusal`).
    /// `position`: the offending brace, in characters from 1.
    case promptSyntax(position: Int, reason: PromptRefusal)
    /// A prompt whose alternatives make more prompts than `max` (`PromptAlternatives.maxCombinations`).
    case tooManyPromptVariants(count: Int, max: Int)
    /// A prompt that spells a text the model's encoder reserves (`text`; Qwen-Image-2.1:
    /// `<|image_pad|>`, the place of a reference image), refused before any weight is read.
    case promptReservedText(text: String)
    /// A prompt longer than the model reads: `tokens` counted, `max` at most (Qwen-Image-2.1: 512).
    case promptTooLong(tokens: Int, max: Int)
    /// An XY grid (`RenderGrid`) whose axes do not make a grid (see `GridRefusal`).
    case gridRefused(reason: GridRefusal)
    /// An XY grid of more cells than `max` (`RenderGrid.maxCells`): one render per cell.
    case tooManyGridCells(count: Int, max: Int)
    /// An img2img strength outside `]0, 1]`.
    case strengthOutOfRange(strength: Double)
    /// An img2img strength that leaves no evaluation: it must exceed `floor`.
    case strengthTooLow(strength: Double, steps: Int, floor: Double)
    /// A starting image given to a model without img2img (Qwen-Image-2.1 edits by reference instead).
    case imageToImageUnsupported(model: String)
    /// An image ImageIO cannot read.
    case imageUnreadable(file: String)
    /// ImageIO refused to encode the PNG.
    case imageEncodingFailed
    /// The PNG could not be written there.
    case imageWriteFailed(file: String)
    /// A LoRA map the engine cannot apply (see `LoRARefusal`).
    case unsupportedLoRA(file: String, reason: LoRARefusal)
    /// A LoRA forged for another model: `target` is the one it was forged for.
    case loraForOtherModel(lora: String, target: String, model: String)
    /// A hand-composed `Chain` whose wires do not match: `output` comes out where `input` is expected.
    case incompatibleChain(output: String, input: String)

    // ── the machine ───────────────────────────────────────────────────────────────────────

    /// The memory available now (`MemoryBudget.available`: the machine's state at the render's launch,
    /// minus a reserve) is below the model's floor at this format (`ModelCard.memoryNeed`). Bytes.
    case insufficientMemory(needed: Int, available: Int)
    /// Another process is rendering on this machine (the app, the CLI, whatever their library or
    /// user — `RenderLock.machine`): two models do not fit
    /// in 16 GB. Within one process, renders queue instead.
    case renderAlreadyRunning
    /// The render, the installation or the download was cancelled.
    case cancelled
    /// `EngineSettings.load(from:)` after the settings were read from another profile.
    case settingsAlreadyLoaded(loaded: String, requested: String)
    /// A breakdown deep in the engine (GPU, kernels, arenas, forge numerics): not a misuse.
    /// `component` names where, `detail` is the technical message.
    case internalFailure(component: String, detail: String)

    /// Why a format is refused — `code` refines into `format_refused.<rawValue>`.
    public enum FormatRefusal: String, Sendable, Equatable, CaseIterable {
        /// A side that is not a multiple of `Format.multiple` (16: VAE ×8, patch ×2).
        case notMultiple = "not_multiple"
        /// An area beyond `Format.maxSurface` (1024×1536, the last format measured).
        case tooLarge = "too_large"
    }

    /// Why a prompt's alternatives do not parse — `code` refines into `prompt_syntax.<rawValue>`.
    public enum PromptRefusal: String, Sendable, Equatable, CaseIterable {
        /// A `{` that no `}` closes.
        case unclosedGroup = "unclosed_group"
        /// A `}` that closes no group.
        case unmatchedClose = "unmatched_close"
        /// A `{` inside a group: groups do not nest.
        case nestedGroup = "nested_group"
    }

    /// Why an XY grid is refused — `code` refines into `grid_refused.<rawValue>`.
    public enum GridRefusal: String, Sendable, Equatable, CaseIterable {
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
        /// A model axis under a LoRA stack: a LoRA is forged for one model.
        case loraAcrossModels = "lora_across_models"
    }

    /// Why a LoRA is refused — `code` refines into `unsupported_lora.<rawValue>`.
    public enum LoRARefusal: String, Sendable, Equatable, CaseIterable {
        /// The map is a model, not an adapter.
        case notALoRA = "not_a_lora"
        /// The map does not say which model it was forged for: forge it again.
        case noTarget = "no_target"
        /// Its two halves do not compose (rank or shape): damaged, forge it again.
        case inconsistent = "inconsistent"
    }

    // ── what an app reads ─────────────────────────────────────────────────────────────────

    /// **The stable key of the case** — what an app looks its translated sentence up by. Never
    /// renamed once published; a new case takes a new code.
    public var code: String {
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
        case .importRefused: return "import_refused"
        case .notAnImportedModel: return "not_an_imported_model"
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
        case .imageUnreadable: return "image_unreadable"
        case .imageEncodingFailed: return "image_encoding_failed"
        case .imageWriteFailed: return "image_write_failed"
        case .unsupportedLoRA: return "unsupported_lora"
        case .loraForOtherModel: return "lora_for_other_model"
        case .incompatibleChain: return "incompatible_chain"
        case .insufficientMemory: return "insufficient_memory"
        case .renderAlreadyRunning: return "render_already_running"
        case .cancelled: return "cancelled"
        case .settingsAlreadyLoaded: return "settings_already_loaded"
        case .internalFailure: return "internal_failure"
        }
    }

    public var errorDescription: String? {
        switch self {
        case .modelNotInstalled(let model): return "The model \(model) is not installed."
        case .unknownModel(let model): return "There is no model called \(model)."
        case .licenseNotAccepted(let model): return "The license of \(model) has not been accepted."
        case .fileMissing(let path): return "A file is missing: \(path)."
        case .fileUnreadable(let path): return "A file cannot be opened: \(path)."
        case .corruptMap(let file):
            return "A model file is damaged or from another version" + (file.map { ": \($0)." } ?? ".")
        case let .diskFull(needed, available):
            switch (needed, available) {
            case let (n?, a?): return "Not enough disk space: \(gigabytes(n)) needed, \(gigabytes(a)) available."
            case let (n?, nil): return "Not enough disk space: \(gigabytes(n)) needed."
            default: return "The disk is full."
            }
        case .downloadInterrupted(let file): return "The download of \(file) was interrupted."
        case let .downloadRefused(file, status):
            return status == 401 || status == 403
                ? "The download of \(file) was refused: the repository is gated (HTTP \(status))."
                : "The download of \(file) was refused by the server (HTTP \(status))."
        case .downloadCorrupt(let file): return "The download of \(file) is corrupt (sha256 mismatch)."
        case .importRefused(let file, _): return "This file cannot be imported: \(file)."
        case .notAnImportedModel(let model): return "\(model) is not an imported model."
        case .emptyPrompt: return "The prompt is empty."
        case let .invalidSteps(steps, allowed):
            return allowed.isEmpty ? "\(steps) steps: a render takes between 1 and \(Request.maximumSteps)."
                : "This model does not render in \(steps) steps."
        case let .tooManyReferences(count, max):
            return max == 0 ? "This model does not edit from reference images."
                : "\(count) reference images: this model reads at most \(max)."
        case let .imageTooSmall(side, minimum):
            return "The image is too small: \(side) px on a side, \(minimum) px at least."
        case let .formatRefused(width, height, reason):
            switch reason {
            case .notMultiple: return "\(width)×\(height): each side must be a multiple of \(Format.multiple)."
            case .tooLarge:
                return "\(width)×\(height) (\(megapixels(width * height))) is larger than the largest format: "
                    + "\(megapixels(Format.maxSurface)) at most."
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
            }
        case let .tooManyGridCells(count, max):
            return "The grid makes \(count) cells: \(max) at most."
        case .strengthOutOfRange(let strength):
            return "The strength must be above 0 and at most 1 (\(strengthText(strength)))."
        case let .strengthTooLow(strength, steps, floor):
            return "A strength of \(strengthText(strength)) leaves no step out of \(steps): "
                + "it must be above \(String(format: "%.3f", floor))."
        case .imageToImageUnsupported(let model): return "\(model) does not start from an image."
        case .imageUnreadable(let file): return "This image cannot be read: \(file)."
        case .imageEncodingFailed: return "The image could not be encoded as PNG."
        case .imageWriteFailed(let file): return "The image could not be saved to \(file)."
        case let .unsupportedLoRA(file, reason):
            switch reason {
            case .notALoRA: return "\(file) is a model, not a LoRA."
            case .noTarget: return "The LoRA \(file) does not say which model it is for."
            case .inconsistent: return "The LoRA \(file) is damaged."
            }
        case let .loraForOtherModel(lora, target, model):
            return "The LoRA \(lora) is for \(target), not for \(model)."
        case let .incompatibleChain(output, input):
            return "These modules do not connect: \(output) comes out where \(input) is expected."
        case let .insufficientMemory(needed, available):
            return "Not enough memory: this render needs \(gigabytes(needed)), \(gigabytes(available)) is available now."
        case .renderAlreadyRunning: return "Another render is already running on this machine."
        case .cancelled: return "Cancelled."
        case .settingsAlreadyLoaded: return "The engine settings were already loaded from another profile."
        case let .internalFailure(component, detail): return "Internal engine failure (\(component)): \(detail)"
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .modelNotInstalled: return "Install the model, or choose the library it is installed in."
        case .unknownModel: return "Choose a model from the list of installed models, or install it first."
        case .licenseNotAccepted: return "Read and accept the model's license before rendering."
        case .fileMissing: return "Choose the file again, or reinstall the model it belongs to."
        case .fileUnreadable: return "Check the file's permissions and that its volume is connected."
        case .corruptMap: return "Reinstall the model, or import it again."
        case .diskFull: return "Make room (uninstall a model, empty the cache), then start again: the files already downloaded are kept, the interrupted one starts over."
        case .downloadInterrupted: return "Check the connection and start again: the files already downloaded are kept, the interrupted one starts over."
        case let .downloadRefused(_, status) where status == 401 || status == 403:
            return "Accept the model's license on huggingface.co, then set HF_TOKEN (or log in with `hf auth login`)."
        case .downloadRefused: return "Try again later."
        case .downloadCorrupt: return "Start again: the damaged file was thrown away."
        case .importRefused: return "Import a LoRA or a full model of a supported family, as .safetensors (not quantized)."
        case .notAnImportedModel: return nil
        case .emptyPrompt: return "Write a prompt."
        case let .invalidSteps(_, allowed) where !allowed.isEmpty:
            return "Choose " + allowed.map(String.init).joined(separator: ", ") + " steps."
        case .invalidSteps: return "Leave the number of steps at the model's default."
        case .tooManyReferences(_, let max) where max == 0:
            return "Choose a model that edits by reference (Qwen-Image-2.1, FLUX.2 [klein])."
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
            case .unreadableAxis: return "Write an axis as prompt[:N], seed=42,43, seeds=4, steps=6,8, lora[:NAME]=0.4,0.8, loras[:STRENGTH]=none,flat,ink, model=z-image,qwen-image-2.1 or image."
            case .emptyAxis, .noSuchGroup, .noSuchLoRA: return "Give each axis at least one value, from the prompt and the stack."
            case .loraAcrossModels: return "Remove the LoRAs, or vary something other than the model."
            }
        case .tooManyGridCells: return "Remove values from an axis."
        case .imageTooSmall, .formatRefused, .formatUnreadable:
            return "Choose one of the formats the model offers, for example 1024×1024 or 832×1216."
        case .strengthOutOfRange: return "A strength between 0.2 and 0.8 keeps the image while transforming it; 0.6 by default."
        case .strengthTooLow: return "Increase the strength, or the number of steps."
        case .imageToImageUnsupported: return "Pass the image as reference 1 and describe the change."
        case .imageUnreadable: return "Choose a PNG, JPEG or HEIC image."
        case .imageEncodingFailed: return nil
        case .imageWriteFailed: return "Check that the folder exists and that you can write to it."
        case .unsupportedLoRA(_, .notALoRA): return "Import it as a model instead."
        case .unsupportedLoRA: return "Import the LoRA again from its .safetensors."
        case .loraForOtherModel: return "Choose a LoRA made for this model, or remove this one from the stack."
        case .incompatibleChain: return "Render with the model as installed, rather than modules assembled by hand."
        case .insufficientMemory: return "Choose a smaller format or a lighter model, or close other applications."
        case .renderAlreadyRunning: return "Wait for it to finish, or add the request to the app's queue."
        case .cancelled: return nil
        case .settingsAlreadyLoaded: return "Load the settings at launch, before the first render."
        case .internalFailure: return "Try again; if it persists, open an issue with this message."
        }
    }

    /// The sentence and its suggestion — what the CLI prints; an import's diagnostic follows it.
    public var description: String {
        let text = [errorDescription, recoverySuggestion].compactMap { $0 }.joined(separator: " ")
        if case let .importRefused(_, detail) = self { return text + " (\(detail))" }
        return text
    }

    // ── the door: every internal failure, folded into its case ────────────────────────────

    /// **The fold.** An `EngineError` comes back as is; each internal failure becomes the case an
    /// app can name; anything else is an `internalFailure`.
    public init(_ error: any Error) {
        switch error {
        case let e as EngineError: self = e
        case is CancellationError: self = .cancelled
        case let e as MissingFile: self = .fileMissing(path: e.path)
        case let e as Request.Failure:
            switch e {
            case .emptyPrompt: self = .emptyPrompt
            case .steps(let n): self = .invalidSteps(steps: n, allowed: [])
            case let .references(n, max): self = .tooManyReferences(count: n, max: max)
            }
        case let e as Strength.Failure:
            switch e {
            case .outOfBounds(let s): self = .strengthOutOfRange(strength: s)
            case let .noEvaluation(s, steps, floor): self = .strengthTooLow(strength: s, steps: steps, floor: floor)
            case .withoutEncoder(let model): self = .imageToImageUnsupported(model: model)
            }
        case let e as QwenImage21Pipeline.Failure:
            switch e {
            case .steps(let n): self = .invalidSteps(steps: n, allowed: QwenImage21Pipeline.acceptedSteps)
            case .imageToImage: self = .imageToImageUnsupported(model: ModelCard.qwenImage21.id)
            }
        case let e as Model.Failure:
            switch e { case .unknown(let name): self = .unknownModel(model: name) }
        case let e as ImageRGB.Failure:
            switch e {
            case .unreadable(let file): self = .imageUnreadable(file: file)
            case .encoding: self = .imageEncodingFailed
            case .writing(let file): self = .imageWriteFailed(file: file)
            }
        case let e as LoRA.Failure:
            switch e {
            case let .wrongTarget(_, name, forgedFor, model):
                self = .loraForOtherModel(lora: name, target: forgedFor, model: model)
            case let .notALoRA(path, _): self = .unsupportedLoRA(file: path, reason: .notALoRA)
            case .missingTarget(let path): self = .unsupportedLoRA(file: path, reason: .noTarget)
            case .inconsistentRank(let target, _, _), .inconsistentShape(let target, _):
                self = .unsupportedLoRA(file: target, reason: .inconsistent)
            }
        case let e as Chain.Failure:
            switch e {
            case let .incompatibleText(output, entry): self = .incompatibleChain(output: "\(output)", input: "\(entry)")
            case let .latentIncompatible(output, entry), let .imageIncompatible(output, entry):
                self = .incompatibleChain(output: "\(output)", input: "\(entry)")
            case .textWithoutImages(let format): self = .incompatibleChain(output: "text", input: "\(format)")
            }
        case let e as Sampler.Failure:
            switch e {
            case .belowFloor(let side): self = .imageTooSmall(side: side * 8, minimum: Format.minimumSide)
            case .missingSide, .indivisible: self = .internalFailure(component: "sampler", detail: e.description)
            }
        case let e as Qwen3VLPrompt.Failure:
            switch e {
            case .reservedText(let text): self = .promptReservedText(text: text)
            case let .tooLong(tokens, max): self = .promptTooLong(tokens: tokens, max: max)
            case .slots: self = .internalFailure(component: "prompt", detail: e.description)
            }
        case let e as EngineSettings.Failure:
            switch e { case let .alreadyRead(loaded, requested): self = .settingsAlreadyLoaded(loaded: loaded, requested: requested) }
        case let e as Artifact.Failure:
            switch e {
            case let .cannotOpen(path, code): self = code == ENOENT ? .fileMissing(path: path) : .fileUnreadable(path: path)
            case .inFile(let path, _): self = .corruptMap(file: path)
            case .badMagic, .truncated, .badHeader, .unaligned, .doesNotFit: self = .corruptMap(file: nil)
            case .misuse: self = .internalFailure(component: "engine", detail: e.description)
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
        case is T5Tokenizer.Failure, is OrderedJSON.ParseError: self = .corruptMap(file: nil)
        case let e as Arena.Failure: self = .internalFailure(component: "arena", detail: e.description)
        case let e as GEMM.Failure: self = .internalFailure(component: "gpu", detail: e.description)
        case let e as Conductor.Failure: self = .internalFailure(component: "co-execution", detail: e.description)
        case let e as FlashMatrix.Failure: self = .internalFailure(component: "flash", detail: e.description)
        case let e as FlashAttention.Failure: self = .internalFailure(component: "flash", detail: e.description)
        case let e as WidenGPU.Failure: self = .internalFailure(component: "widen", detail: e.description)
        case let e as Numerics.Failure: self = .internalFailure(component: "forge", detail: e.description)
        default:
            let ns = error as NSError
            // A write that ran out of room mid-forge, mid-cache: the disk, said as such.
            if (ns.domain == NSPOSIXErrorDomain && ns.code == Int(ENOSPC))
                || (ns.domain == NSCocoaErrorDomain && ns.code == NSFileWriteOutOfSpaceError) {
                self = .diskFull(needed: nil, available: nil)
            } else if let url = error as? URLError {
                self = .downloadInterrupted(file: url.failingURL?.lastPathComponent ?? "?")
            } else {
                self = .internalFailure(component: "\(type(of: error))", detail: "\(error)")
            }
        }
    }

    /// **The door**: runs `body` and throws only `EngineError`. Every public function that throws
    /// passes through it (or throws an `EngineError` itself).
    package static func boundary<T>(_ body: () throws -> T) throws(EngineError) -> T {
        do { return try body() } catch { throw EngineError(error) }
    }

    /// Every case once, with plausible values — the test target checks that the codes are unique
    /// and that each case has an English sentence. **A new case is added here too.**
    package static let samples: [EngineError] = [
        .modelNotInstalled(model: "z-image"), .unknownModel(model: "sd15"), .licenseNotAccepted(model: "qwen-image-2.1"),
        .fileMissing(path: "/a.silicon"), .fileUnreadable(path: "/a.silicon"), .corruptMap(file: "/a.silicon"),
        .diskFull(needed: 30_000_000_000, available: 1_000_000_000), .downloadInterrupted(file: "vae.safetensors"),
        .downloadRefused(file: "vae.safetensors", status: 403), .downloadCorrupt(file: "vae.safetensors"),
        .importRefused(file: "x.safetensors", detail: "not recognized"), .notAnImportedModel(model: "Z-Image Turbo"),
        .emptyPrompt, .invalidSteps(steps: 8, allowed: [5, 6, 7, 9]), .tooManyReferences(count: 4, max: 3),
        .imageTooSmall(side: 256, minimum: 512), .formatRefused(width: 1000, height: 1000, reason: .notMultiple),
        .formatUnreadable(text: "big"), .promptSyntax(position: 12, reason: .unclosedGroup),
        .tooManyPromptVariants(count: 128, max: 64), .promptReservedText(text: "<|image_pad|>"),
        .promptTooLong(tokens: 600, max: 512), .gridRefused(reason: .groupNotOnAxis),
        .tooManyGridCells(count: 81, max: 64), .strengthOutOfRange(strength: 1.5),
        .strengthTooLow(strength: 0.1, steps: 8, floor: 0.125), .imageToImageUnsupported(model: "qwen-image-2.1"),
        .imageUnreadable(file: "/a.png"), .imageEncodingFailed, .imageWriteFailed(file: "/a.png"),
        .unsupportedLoRA(file: "x.lora.silicon", reason: .noTarget),
        .loraForOtherModel(lora: "PopArt", target: "z-image", model: "krea2"),
        .incompatibleChain(output: "flux", input: "qwen-image"),
        .insufficientMemory(needed: 7_800_000_000, available: 8_589_934_592), .renderAlreadyRunning, .cancelled,
        .settingsAlreadyLoaded(loaded: "/a", requested: "/b"), .internalFailure(component: "gpu", detail: "no Metal device"),
    ]
}

private func gigabytes(_ bytes: Int) -> String { String(format: "%.1f GB", Double(bytes) / 1e9) }
private func megapixels(_ pixels: Int) -> String { String(format: "%.2f Mpx", Double(pixels) / 1e6) }

/// A strength as typed: enough digits that a value just inside or outside the bounds does not
/// print as one of them (`0.001` is not `0.00`), without trailing zeros.
private func strengthText(_ value: Double) -> String {
    guard value.isFinite else { return "\(value)" }
    var text = String(format: "%.6f", value)
    while text.hasSuffix("0") { text.removeLast() }
    if text.hasSuffix(".") { text.removeLast() }
    return text == "-0" ? "0" : text
}

/// **A missing file — internal to the package**: a forged map or LoRA from `store/`, a component
/// installed under `store/composants/` (tokenizer, VAE, configs). Distinct from a file that is
/// present but unreadable or damaged, which throws its reader's error. At the door it becomes
/// `EngineError.fileMissing`, or `modelNotInstalled` when it is a model being built (`Model.named`).
package struct MissingFile: Error, CustomStringConvertible, Equatable {
    package enum Nature: Sendable, Equatable {
        /// Under `store/`: a forged map or LoRA — it is installed or imported (`Installer`, `ModelImport`).
        case map
        /// Under `store/composants/`: a published file of a family (tokenizer, VAE) — it is installed.
        case published
        /// A golden tensor of a check (`goldens-*.safetensors`, beside the oracles): it is not
        /// installed, the reference writes it — its oracle, run under the machine token.
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

    /// **The oracle that writes a golden**: the family's own oracle for a family's folder, the
    /// stage's oracle for Z-Image's (named by the words of the golden's name before the first number).
    package static func oracle(forGolden path: String) -> String {
        let folder = ((path as NSString).deletingLastPathComponent as NSString).lastPathComponent
        if folder != "oracle", !folder.isEmpty, folder != "." {
            return "bench/oracle/\(folder)/oracle_\(folder).py"
        }
        let stem = ((path as NSString).lastPathComponent as NSString).deletingPathExtension
        let words = stem.split(separator: "-").dropFirst().prefix { !($0.first?.isNumber ?? true) }
        return "bench/oracle/oracle_\(words.isEmpty ? "<stage>" : words.joined(separator: "_")).py"
    }

    package var description: String {
        switch nature {
        case .map: return "map missing: \(path) — install the model or import it"
        case .published: return "component missing: \(path) — install the model"
        case .golden:
            return "golden missing: \(path) — goldens are written by the reference, not installed: "
                + "tools/machine.sh tools/.venv/bin/python \(Self.oracle(forGolden: path)) … "
                + "(the .json of the same name records how it was made)"
        }
    }
}
