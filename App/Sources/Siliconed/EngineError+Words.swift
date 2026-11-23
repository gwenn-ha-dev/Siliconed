// What an app reads of an `EngineError`: its code, its English sentence, its suggestion.

import Foundation
import SiliCore

extension EngineError {
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
        case .importDeeperModel: return "import_deeper_model"
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
        case let .importDeeperModel(file, model, holds, reads):
            return "\(file) has \(holds) layers where \(ModelCard.withID(model)?.name ?? model) has \(reads): not supported yet, coming soon."
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
            // Said of a render and of a rewrite alike: the sentence names neither. `needed` within the budget: the soberest
            // form stopped the fuse (`Stage.choose`), and the one number known is the soberest's socle — not what it needs.
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
        case .notAnImportedModel, .importDeeperModel: return nil
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
