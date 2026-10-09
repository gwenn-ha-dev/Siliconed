// **What the app says when something refuses or fails** — one sentence per `EngineError.code`,
// translated, and what the user can do on a second line.
//
// The engine throws a closed enumeration (`EngineError`, 38 cases, `docs/API.md` §5): each case
// carries its values, never a ready-made sentence. Here each case becomes the app's sentence, in the
// system language (`Localizable.xcstrings`, English source — the engine's `errorDescription`
// reworded for a person rather than a program —, every language required by `tools/translations.sh`). The
// switch is exhaustive: a new case in the engine does not compile here until it has its sentence.
// The only free text is the technical detail of `internalFailure` and `importRefused`, shown under
// the sentence in a smaller font, never in it.

import Foundation
import Siliconed

/// **A problem to show**: its sentence, what to do, and the engine's technical detail if any.
struct Problem: Equatable, Sendable {
    /// `EngineError.code` (`"disk_full"`…), `nil` for what the app refuses by itself.
    var code: String?
    var title: String
    var suggestion: String?
    var detail: String?

    /// The sentence and its suggestion, one per line — what the remote control returns as "message".
    var text: String { [title, suggestion].compactMap { $0 }.joined(separator: "\n") }

    /// A refusal of the app itself (an unreadable image chosen, a JPEG that does not encode).
    init(_ title: String, suggestion: String? = nil) {
        self.title = title
        self.suggestion = suggestion
    }

    /// Anything thrown: folded at the engine's door into its case (`EngineError(_:)`).
    init(_ error: any Error) { self.init(EngineError(error)) }

    init(_ e: EngineError) {
        code = e.code
        title = Self.sentence(e)
        suggestion = Self.suggestion(e)
        switch e {
        case let .internalFailure(_, detail), let .importRefused(_, detail): self.detail = Library.withoutHome(detail)
        default: detail = nil
        }
    }

    /// A model's name as the user knows it, from its identifier.
    private static func name(_ id: String) -> String { ModelCard.withID(id)?.name ?? id }

    /// A file's name, not its path: the user chose the file and knows where it is.
    private static func file(_ path: String) -> String { (path as NSString).lastPathComponent }

    // ── one sentence per code ──

    private static func sentence(_ e: EngineError) -> String {
        switch e {
        case .modelNotInstalled(let m):                              // model_not_installed
            return String(localized: "\(name(m)) isn't installed.")
        case .unknownModel(let m):                                   // unknown_model
            return String(localized: "There is no model called “\(m)”.")
        case .licenseNotAccepted(let m):                             // license_not_accepted
            return String(localized: "The license of \(name(m)) hasn't been accepted.")
        case .fileMissing(let p):                                    // file_missing
            return String(localized: "A file is missing: \(file(p)).")
        case .fileUnreadable(let p):                                 // file_unreadable
            return String(localized: "A file can't be opened: \(file(p)).")
        case .corruptMap(let f):                                     // corrupt_map
            if let f { return String(localized: "A model file is damaged or from another version: \(file(f)).") }
            return String(localized: "A model file is damaged or from another version.")
        case let .diskFull(needed, available):                       // disk_full
            switch (needed, available) {
            case let (n?, a?):
                return String(localized: "Not enough disk space: \(Figures.disk(n)) needed, \(Figures.disk(a)) available.")
            case let (n?, nil):
                return String(localized: "Not enough disk space: \(Figures.disk(n)) needed.")
            default:
                return String(localized: "The disk is full.")
            }
        case .downloadInterrupted(let f):                            // download_interrupted
            return String(localized: "The download of \(f) was interrupted.")
        case let .downloadRefused(f, status):                        // download_refused
            if status == 401 || status == 403 {
                return String(localized: "The download of \(f) was refused: the repository is gated (HTTP \(String(status))).")
            }
            return String(localized: "The download of \(f) was refused by the server (HTTP \(String(status))).")
        case .downloadCorrupt(let f):                                // download_corrupt
            return String(localized: "The download of \(f) is corrupt: its checksum doesn't match.")
        case .importRefused(let f, _):                               // import_refused
            return String(localized: "This file can't be imported: \(file(f)).")
        case .notAnImportedModel(let m):                             // not_an_imported_model
            return String(localized: "\(name(m)) wasn't imported: uninstall it from the Models sheet.")
        case .emptyPrompt:                                           // empty_prompt
            return String(localized: "The prompt is empty.")
        case let .invalidSteps(steps, allowed):                      // invalid_steps
            return allowed.isEmpty ? String(localized: "\(steps) steps: a render takes between 1 and \(Request.maximumSteps).")
                : String(localized: "This model doesn't render in \(steps) steps.")
        case let .tooManyReferences(count, max):                     // too_many_references
            return max == 0 ? String(localized: "This model doesn't edit images.")
                : String(localized: "\(count) images: this model reads \(max) at most.")
        case let .imageTooSmall(side, minimum):                      // image_too_small
            return String(localized: "Too small: \(String(side)) px on a side, \(String(minimum)) px at least.")
        case let .formatRefused(w, h, reason):                       // format_refused
            switch reason {
            case .notMultiple:
                return String(localized: "\(String(w)) × \(String(h)): each side must be a multiple of \(String(Format.multiple)).")
            case .tooLarge:
                return String(localized: "\(String(w)) × \(String(h)) is larger than the largest size, 1024 × 1536.")
            }
        case .formatUnreadable(let t):                               // format_unreadable
            return String(localized: "“\(t)” isn't a size: write it as width × height, like 832x1216.")
        case let .promptSyntax(position, reason):                    // prompt_syntax
            switch reason {
            case .unclosedGroup:
                return String(localized: "The prompt opens a group of alternatives that is never closed (character \(String(position))).")
            case .unmatchedClose:
                return String(localized: "The prompt closes a group of alternatives that was never opened (character \(String(position))).")
            case .nestedGroup:
                return String(localized: "Groups of alternatives can't be nested (character \(String(position))).")
            }
        case let .tooManyPromptVariants(count, max):                 // too_many_prompt_variants
            return String(localized: "These alternatives make \(String(count)) prompts: \(String(max)) at most.")
        case .promptReservedText(let text):                          // prompt_reserved_text
            return String(localized: "The prompt contains \(text), a text the model reserves for its reference images.")
        case let .promptTooLong(tokens, max):                        // prompt_too_long
            return String(localized: "The prompt is too long: \(String(tokens)) tokens, this model reads \(String(max)) at most.")
        case .gridRefused(let reason):                               // grid_refused
            switch reason {
            case .emptyAxis: return String(localized: "An axis of the grid has no value.")
            case .sameAxisTwice: return String(localized: "The grid's two axes vary the same setting.")
            case .noSuchGroup: return String(localized: "An axis of the grid names a group of alternatives the prompt doesn't have.")
            case .noSuchLoRA: return String(localized: "An axis of the grid names a LoRA that isn't in the stack.")
            case .groupNotOnAxis: return String(localized: "The prompt has a group of alternatives that isn't an axis of the grid.")
            case .invalidValue: return String(localized: "An axis of the grid has a value no render takes.")
            case .unreadableAxis: return String(localized: "An axis of the grid can't be read.")
            case .loraAcrossModels: return String(localized: "A grid that varies the model can't take a LoRA: each LoRA is made for one model.")
            }
        case let .tooManyGridCells(count, max):                      // too_many_grid_cells
            return String(localized: "This grid makes \(String(count)) cells: \(String(max)) at most.")
        case .strengthOutOfRange(let s):                             // strength_out_of_range
            return String(localized: "The strength must be between 0 and 1 (\(Figures.strength(s))).")
        case let .strengthTooLow(s, steps, _):                       // strength_too_low
            return String(localized: "A strength of \(Figures.strength(s)) leaves no step out of \(steps).")
        case .imageToImageUnsupported(let m):                        // image_to_image_unsupported
            return String(localized: "\(name(m)) doesn't start from an image.")
        case .imageUnreadable(let f):                                // image_unreadable
            return String(localized: "This image can't be read: \(file(f)).")
        case .imageEncodingFailed:                                   // image_encoding_failed
            return String(localized: "The image couldn't be encoded as PNG.")
        case .imageWriteFailed(let f):                               // image_write_failed
            return String(localized: "The image couldn't be saved to \(f).")
        case let .unsupportedLoRA(f, reason):                        // unsupported_lora
            switch reason {
            case .notALoRA: return String(localized: "\(file(f)) is a model, not a LoRA.")
            case .noTarget: return String(localized: "The LoRA \(file(f)) doesn't say which model it's for.")
            case .inconsistent: return String(localized: "The LoRA \(file(f)) is damaged.")
            }
        case let .loraForOtherModel(lora, target, model):            // lora_for_other_model
            return String(localized: "The LoRA \(lora) is for \(name(target)), not for \(name(model)).")
        case let .incompatibleChain(output, input):                  // incompatible_chain
            return String(localized: "These modules don't connect: \(output) comes out where \(input) is expected.")
        case let .insufficientMemory(needed, available):             // insufficient_memory
            return String(localized: "This size needs \(Figures.memory(needed)) of memory, and \(Figures.memory(available)) is free right now.")
        case .renderAlreadyRunning:                                  // render_already_running
            return String(localized: "Another program is already rendering on this library.")
        case .cancelled:                                             // cancelled
            return String(localized: "Stopped.")
        case .settingsAlreadyLoaded:                                 // settings_already_loaded
            return String(localized: "The engine settings were already loaded from another profile.")
        case .internalFailure(let component, _):                     // internal_failure
            return String(localized: "The engine failed (\(component)).")
        }
    }

    // ── what the user can do ──

    private static func suggestion(_ e: EngineError) -> String? {
        switch e {
        case .modelNotInstalled:
            return String(localized: "Install it from the Models sheet (⇧⌘M).")
        case .unknownModel:
            return String(localized: "Choose a model from the list.")
        case .licenseNotAccepted:
            return String(localized: "Read its license and accept it, then render again.")
        case .fileMissing:
            return String(localized: "Choose the file again, or reinstall the model it belongs to.")
        case .fileUnreadable:
            return String(localized: "Check the file's permissions, and that its disk is connected.")
        case .corruptMap:
            return String(localized: "Reinstall the model from the Models sheet, or import it again.")
        case .diskFull:
            return String(localized: "Make room — uninstall a model, or discard the interrupted downloads in the Models sheet — then start again: the files already downloaded are kept.")
        case .downloadInterrupted:
            return String(localized: "Check the connection and start again: the files already downloaded are kept, the interrupted one starts over.")
        case let .downloadRefused(_, status) where status == 401 || status == 403:
            // The downloader reads `~/.cache/huggingface/token`, which `hf auth login` writes;
            // an environment variable cannot be given to an app opened from the Finder.
            return String(localized: "Accept the model's license on huggingface.co, then sign in on this Mac: “hf auth login” in a terminal, and install again.")
        case .downloadRefused:
            return String(localized: "Try again later.")
        case .downloadCorrupt:
            return String(localized: "Start again: the damaged file was thrown away.")
        case .importRefused:
            return String(localized: "Import a LoRA or a full model of a supported family, as .safetensors (at least 8 bits) or .gguf (at least 4 bits).")
        case .notAnImportedModel, .imageEncodingFailed, .cancelled:
            return nil
        case .emptyPrompt:
            return String(localized: "Write a prompt.")
        case let .invalidSteps(_, allowed) where !allowed.isEmpty:
            return String(localized: "Choose \(allowed.map(String.init).formatted(.list(type: .or))) steps.")
        case .invalidSteps:
            return String(localized: "Leave the number of steps at the model's default.")
        case .tooManyReferences(_, let max) where max == 0:
            return String(localized: "Choose a model that edits, like Qwen-Image-2.1.")
        case .tooManyReferences(_, let max):
            return String(localized: "Keep \(max) images at most.")
        case .promptSyntax:
            return String(localized: "Write alternatives as {a|b|c}, without nesting; \\{, \\} and \\| are the characters themselves.")
        case .tooManyPromptVariants:
            return String(localized: "Remove a group, or some alternatives.")
        case .promptReservedText(let text):
            return String(localized: "Remove \(text) from the prompt.")
        case .promptTooLong:
            return String(localized: "Shorten the prompt.")
        case .gridRefused(let reason):
            switch reason {
            case .groupNotOnAxis:
                return String(localized: "Make each group an axis, or write its braces as \\{ and \\}.")
            case .sameAxisTwice: return String(localized: "Choose another setting for one of the axes.")
            case .invalidValue: return String(localized: "Steps are whole numbers from 1, and a grid needs at least one seed.")
            case .unreadableAxis, .emptyAxis, .noSuchGroup, .noSuchLoRA:
                return String(localized: "Give each axis its values: a group of the prompt, seeds, steps, a LoRA of the stack, models or the edit's images.")
            case .loraAcrossModels: return String(localized: "Remove the LoRAs, or vary something other than the model.")
            }
        case .tooManyGridCells:
            return String(localized: "Remove values from an axis.")
        case .imageTooSmall, .formatRefused, .formatUnreadable:
            return String(localized: "Choose one of the model's sizes, like 1024 × 1024 or 832 × 1216.")
        case .strengthOutOfRange, .strengthTooLow:
            return String(localized: "Increase the strength, or the number of steps.")
        case .imageToImageUnsupported:
            return String(localized: "Give the image as image 1 of an edit, and say what changes.")
        case .imageUnreadable:
            return String(localized: "Choose a PNG, JPEG or HEIC image.")
        case .imageWriteFailed:
            return String(localized: "Check that the folder exists and that you can write to it.")
        case .unsupportedLoRA(_, .notALoRA):
            return String(localized: "Import it as a model instead.")
        case .unsupportedLoRA:
            return String(localized: "Import the LoRA again from its .safetensors.")
        case .loraForOtherModel:
            return String(localized: "Choose a LoRA made for this model.")
        case .incompatibleChain:
            return String(localized: "Report it: the app built a chain the engine refuses.")
        case .insufficientMemory:
            return String(localized: "Try a smaller size, or render again once other apps have given memory back.")
        case .renderAlreadyRunning:
            return String(localized: "Wait for it to finish, then render again.")
        case .settingsAlreadyLoaded:
            return String(localized: "Quit and reopen the app.")
        case .internalFailure:
            return String(localized: "Try again; if it persists, report it (Help ▸ Report My Configuration…).")
        }
    }
}

// MARK: - The engine's other texts, said by the app

// The engine also writes English text that is not an error: a license's name, the forge's journal,
// the warnings of a render or of the machine profile. What is a closed set is translated here; what
// is free (a developer's warning, a forge's row) is never shown raw in a sentence of the app — a
// localized line says what happened, and the engine's text stays in its tooltip, like a problem's
// detail.

extension ModelCard {
    /// The license as the app says it (`License.text` is the engine's, English, for the JSON).
    var licenseName: String {
        guard isImported else { return Self.licenseName(license.text) }
        let f = ModelCard.of(family)
        let architecture = f.name, inherited = Self.licenseName(f.license.text)
        return String(localized: "that of the imported model (architecture \(architecture): \(inherited))")
    }

    private static func licenseName(_ text: String) -> String {
        switch text {
        case "Apache 2.0": text
        case "non-commercial (circlestone-labs)": String(localized: "non-commercial (circlestone-labs)")
        case "Krea 2 Community (commercial < $1M, mandatory content filter)":
            String(localized: "Krea 2 Community (commercial under $1M, mandatory content filter)")
        case "Qwen Research (non-commercial)": String(localized: "Qwen Research (non-commercial)")
        default: text
        }
    }
}

/// **The forge's journal, as a line of the app**: the engine writes rows for a terminal (`↓ file
/// (1.23 Go)`, `  … 40 %`, `forging the DiT → name`); the app shows what they mean.
enum ForgeProgress {
    static func line(_ journal: [String]) -> String? {
        var said: String?, downloading: String?
        for row in journal {
            let t = row.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("↓ ") {
                let label = String(t.dropFirst(2).split(separator: " (").first ?? "")
                let file = (label as NSString).lastPathComponent
                downloading = file
                said = String(localized: "Downloading \(file)")
            } else if t.hasPrefix("… "), let n = Int(t.dropFirst(2).prefix { $0.isNumber }) {
                let p = Figures.percent(n)
                said = downloading.map { String(localized: "Downloading \($0): \(p)") } ?? String(localized: "Converting: \(p)")
            } else if t.hasPrefix("✓ "), t.hasSuffix(" installed") {
                said = String(localized: "Installed")
            } else if t.hasPrefix("✓ ") {
                downloading = nil
            } else if t.hasPrefix("forging "), let arrow = t.range(of: "→ ") {
                downloading = nil
                let name = String(t[arrow.upperBound...])
                said = String(localized: "Preparing \(name)")
            }
        }
        return said
    }
}

extension AppState {
    /// The warnings of the last render, said by the app; the engine's own rows are its tooltip.
    var journalLine: String? {
        journal.isEmpty ? nil : String(localized: "The engine noted something unusual during this render.")
    }

    /// The machine profile's warnings, said by the app (`EngineSettings.warnings`: one known form).
    var profileLines: [String] {
        profileWarnings.map { w in
            if let open = w.range(of: "« "), let close = w.range(of: " » ignored in the profile"), open.upperBound <= close.lowerBound {
                let key = String(w[open.upperBound..<close.lowerBound])
                return String(localized: "The machine profile sets « \(key) », which it may not: a profile tunes speed and room, never the image.")
            }
            return String(localized: "The machine profile couldn't be read as it is.")
        }
    }
}
