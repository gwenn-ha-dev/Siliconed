// **What the app says when something refuses or fails** — one sentence per `EngineError.code`,
// translated, and what the user can do on a second line.
//
// The engine throws a closed enumeration (`EngineError`, `Errors.swift`): each case
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
    /// The error itself, when there was one: what crosses to a borrowing Mac, which says it in its language.
    var engine: EngineError?

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
        engine = e
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
            return String(localized: "The download of \(f) was refused by the server (HTTP \(String(status))).")
        case .downloadCorrupt(let f):                                // download_corrupt
            return String(localized: "The download of \(f) is corrupt: its checksum doesn't match.")
        case .importRefused(let f, _):                               // import_refused
            return String(localized: "This file can't be imported: \(file(f)).")
        case let .importNeedsFamily(f, m):                           // import_needs_family
            return String(localized: "To import \(file(f)), install \(name(m)) first.")
        case .notAnImportedModel(let m):                             // not_an_imported_model
            return String(localized: "\(name(m)) wasn't imported: uninstall it from the Models sheet.")
        case let .libraryAbsent(p, volume):                          // library_absent
            if let volume { return String(localized: "The library is on “\(volume)”, which isn't connected.") }
            return String(localized: "The library can't be found: \(Library.withoutHome(p)).")
        case let .libraryRefused(p, reason):                         // library_refused
            let folder = Library.withoutHome(p)
            switch reason {
            case .fileSizeLimit: return String(localized: "This disk can't hold the models: its format (MS-DOS FAT) stops at 4 GB per file.")
            case .readOnly: return String(localized: "This folder can't be written to: \(folder).")
            case .notALibrary: return String(localized: "This folder isn't a Siliconed library: \(folder).")
            case .alreadyALibrary: return String(localized: "This folder already holds a Siliconed library: \(folder).")
            case .sameLibrary: return String(localized: "This folder is the library itself, or inside it.")
            case .copyDamaged: return String(localized: "A file read back from the new place differs from the original: \(file(p)).")
            }
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
        case let .formatRefused(w, h, reason, bound):                // format_refused
            switch reason {
            case .notMultiple:
                return String(localized: "\(String(w)) × \(String(h)): each side must be a multiple of \(String(bound)).")
            case .sideTooLong:
                return String(localized: "\(String(w)) × \(String(h)): this model draws \(String(bound)) px at most on a side.")
            case .tooLarge:
                return String(localized: "\(String(w)) × \(String(h)) is larger than the engine addresses, whatever the memory: \(String(bound / 1_000_000)) megapixels at most.")
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
        case let .solverUnsupported(s, m):                           // solver_unsupported
            return String(localized: "\(name(m)) doesn't offer the \(s) solver.")
        case .detailUnsupported(let m):                              // detail_unsupported
            return String(localized: "\(name(m)) doesn't offer Detail.")
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
        case let .insufficientMemory(needed, _):                     // insufficient_memory
            return AppState.needsMac(needed)
        case .memoryNeedUnknown(let m):                              // memory_need_unknown
            return String(localized: "The memory \(name(m)) needs can't be computed: its files can't be read.")
        case .renderAlreadyRunning:                                  // render_already_running
            return String(localized: "Another program is already rendering on this library.")
        case .cancelled:                                             // cancelled
            return String(localized: "Stopped.")
        case .nonFiniteImage:                                        // non_finite_image
            return String(localized: "This render went wrong and its image is unusable: it wasn't kept.")
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
        case .corruptMap, .memoryNeedUnknown:
            return String(localized: "Reinstall the model from the Models sheet, or import it again.")
        case .diskFull:
            return String(localized: "Make room — uninstall a model, or discard the interrupted downloads in the Models sheet — then start again: the files already downloaded are kept.")
        case .downloadInterrupted:
            return String(localized: "Check the connection and start again: the files already downloaded are kept, the interrupted one starts over.")
        case let .downloadRefused(_, status) where status == 401 || status == 403:
            // No repository an installation reads is gated (`Installation.swift`), and the Hub serves
            // them even to an expired token: a refusal comes from the way to it, not from a license.
            return String(localized: "Check that huggingface.co opens in a browser on this Mac — a proxy, a VPN or a firewall can refuse it — then install again: the files already downloaded are kept.")
        case .downloadRefused:
            return String(localized: "Try again later.")
        case .downloadCorrupt:
            return String(localized: "Start again: the damaged file was thrown away.")
        case .importRefused:
            return String(localized: "Import a LoRA or a full model of a supported family, as .safetensors (at least 8 bits) or .gguf (at least 4 bits).")
        case .importNeedsFamily:
            return String(localized: "Install it from the Models sheet (⇧⌘M), then import the file again: an import downloads nothing.")
        case .libraryAbsent:
            return String(localized: "Connect the disk — or, in Settings, go back to this Mac's library.")
        case .libraryRefused(_, .fileSizeLimit):
            return String(localized: "Choose a disk formatted APFS, Mac OS Extended or exFAT.")
        case .libraryRefused(_, .readOnly):
            return String(localized: "Choose a folder you can write to.")
        case .libraryRefused(_, .notALibrary):
            return String(localized: "Choose the folder that holds the library (its “store” folder), or move the library there.")
        case .libraryRefused(_, .alreadyALibrary):
            return String(localized: "Open it instead, or choose an empty folder.")
        case .libraryRefused(_, .sameLibrary):
            return String(localized: "Choose a folder outside the library.")
        case .libraryRefused(_, .copyDamaged):
            return String(localized: "Check the disk and its cable, then start again: nothing was switched, and the files already checked are kept.")
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
            case .invalidValue: return String(localized: "Steps are whole numbers from 1, a grid needs at least one seed, and a negative LoRA strength needs Experimental → Negative LoRA Strengths.")
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
        case .solverUnsupported:
            return String(localized: "Choose Euler.")
        case .detailUnsupported:
            return String(localized: "Choose Normal.")
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
            return String(localized: "Choose a smaller size or fewer images.")
        case .renderAlreadyRunning:
            return String(localized: "Wait for it to finish, then render again.")
        case .settingsAlreadyLoaded:
            return String(localized: "Quit and reopen the app.")
        case .internalFailure, .nonFiniteImage:
            return String(localized: "Try again; if it persists, report it (Help ▸ Measure This Mac…, then Open the Issue).")
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

    /// **Whether the images it makes may be used commercially** — a license may say yes to its
    /// outputs and no to the model: CircleStone's (Anima) grants its « Outputs » any use, commercial
    /// included (§ 2.e). An imported model: its family's, like its license's name.
    var imagesCommercial: Bool {
        let text = isImported ? ModelCard.of(family).license.text : license.text
        return license.commercial || text == "CircleStone Labs Non-Commercial"
    }

    private static func licenseName(_ text: String) -> String {
        switch text {
        case "Apache 2.0": text
        case "CircleStone Labs Non-Commercial": String(localized: "CircleStone Labs Non-Commercial")
        case "Krea 2 Community (commercial < $1M, mandatory content filter)":
            String(localized: "Krea 2 Community (commercial under $1M, mandatory content filter)")
        case "Qwen Research (non-commercial)": String(localized: "Qwen Research (non-commercial)")
        default: text
        }
    }
}

/// **The forge's journal, as a line of the app**: the engine writes rows for a terminal (`↓ file
/// (1.23 Go)`, `  … 40 %`, `forging the DiT → name`, `  ✓ forged name in 9 s`); the app shows what they
/// mean. Two steps can run at once — the DiT downloads while the encoder is forged
/// (`Installer.install`) — and the line then says both; the forge says no percentage then, so a
/// `… %` after a `↓` is always the download's.
enum ForgeProgress {
    static func line(_ journal: [String]) -> String? {
        var said: String?
        var downloading: String?, received: String?, preparing: String?, converted: String?
        for row in journal {
            let t = row.trimmingCharacters(in: .whitespaces)
            // A library being moved (`AppState.moveLibrary`): « ⇢ <percent> <file> ».
            if t.hasPrefix("⇢ ") {
                let parts = t.dropFirst(2).split(separator: " ", maxSplits: 1)
                if parts.count == 2, let n = Int(parts[0]) {
                    said = String(localized: "\(Figures.percent(n)) — \(String(parts[1]))")
                }
                continue
            }
            if t.hasPrefix("↓ ") {
                let label = String(t.dropFirst(2).split(separator: " (").first ?? "")
                downloading = (label as NSString).lastPathComponent
                received = nil
            } else if t.hasPrefix("… "), let n = Int(t.dropFirst(2).prefix { $0.isNumber }) {
                if downloading != nil { received = Figures.percent(n) } else { converted = Figures.percent(n) }
            } else if t.hasPrefix("✓ "), t.hasSuffix(" installed") {
                said = String(localized: "Installed")
                continue
            } else if t.hasPrefix("✓ forged ") {
                (preparing, converted) = (nil, nil)
            } else if t.hasPrefix("✓ ") {
                (downloading, received) = (nil, nil)
            } else if t.hasPrefix("forging "), let arrow = t.range(of: "→ ") {
                (preparing, converted) = (String(t[arrow.upperBound...]), nil)
            } else {
                continue
            }
            switch (downloading, received, preparing) {
            case let (file?, p?, name?): said = String(localized: "Downloading \(file): \(p) — preparing \(name)")
            case let (file?, nil, name?): said = String(localized: "Downloading \(file) — preparing \(name)")
            case let (file?, p?, nil): said = String(localized: "Downloading \(file): \(p)")
            case let (file?, nil, nil): said = String(localized: "Downloading \(file)")
            case let (nil, _, name?): said = converted.map { String(localized: "Converting: \($0)") } ?? String(localized: "Preparing \(name)")
            case (nil, _, nil): break
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
