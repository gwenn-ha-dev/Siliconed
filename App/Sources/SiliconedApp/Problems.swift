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
        // The component that failed is a word of the code (`engine`, `gpu`): the detail says it, the sentence does not.
        case let .internalFailure(component, detail):
            self.detail = detail.isEmpty ? component : "\(component): \(Library.withoutHome(detail))"
        case let .importRefused(_, detail), let .engineStopped(_, detail):
            self.detail = Library.withoutHome(detail)
        case let .engineMismatch(app, engine):
            detail = "the app's protocol \(app), the engine's \(engine.map(String.init) ?? "none")"
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
            return String(localized: "\(name(m)) isn’t installed.")
        case .unknownModel(let m):                                   // unknown_model
            return String(localized: "There is no model called “\(m)”.")
        case .licenseNotAccepted(let m):                             // license_not_accepted
            return String(localized: "The license of \(name(m)) hasn’t been accepted.")
        case .fileMissing(let p):                                    // file_missing
            return String(localized: "A file is missing: \(file(p)).")
        case .fileUnreadable(let p):                                 // file_unreadable
            return String(localized: "A file can’t be opened: \(file(p)).")
        case .corruptMap(let f):                                     // corrupt_map
            if let f { return String(localized: "A model file is damaged or from another version: \(file(f)).") }
            return String(localized: "A model file is damaged or from another version.")
        case let .diskFull(needed, available, _):                    // disk_full
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
            return String(localized: "The download of \(f) is corrupt: its checksum doesn’t match.")
        case .downloadInProgress(let f):                             // download_in_progress
            return String(localized: "\(f) is already being downloaded by another installation on this Mac.")
        case .importRefused(let f, _):                               // import_refused
            return String(localized: "This file can’t be imported: \(file(f)).")
        case let .importNeedsFamily(f, m):                           // import_needs_family
            return String(localized: "To import \(file(f)), install \(name(m)) first.")
        case let .importBaseModel(f, m):                             // import_base_model
            return String(localized: "\(file(f)) is the base model of \(ModelCard.withID(m)?.family.name ?? m): Siliconed renders \(name(m)).")
        case let .importDeeperModel(f, m, holds, reads):             // import_deeper_model
            return String(localized: "\(file(f)) has \(holds) layers where \(name(m)) has \(reads): not supported yet, coming soon.")
        case .notAnImportedModel(let m):                             // not_an_imported_model
            return String(localized: "\(name(m)) wasn’t imported: uninstall it from the Models sheet.")
        case let .libraryAbsent(p, volume):                          // library_absent
            if let volume { return String(localized: "The library is on “\(volume)”, which isn’t connected.") }
            return String(localized: "The library can’t be found: \(Library.withoutHome(p)).")
        case let .libraryRefused(p, reason):                         // library_refused
            let folder = Library.withoutHome(p)
            switch reason {
            case .fileSizeLimit: return String(localized: "This disk can’t hold the models: its format (MS-DOS FAT) stops at 4 GB per file.")
            case .readOnly: return String(localized: "This folder can’t be written to: \(folder).")
            case .notALibrary: return String(localized: "This folder isn’t a Siliconed library: \(folder).")
            case .alreadyALibrary: return String(localized: "This folder already holds a Siliconed library: \(folder).")
            case .sameLibrary: return String(localized: "This folder is the library itself, or inside it.")
            case .copyDamaged: return String(localized: "A file read back from the new place differs from the original: \(file(p)).")
            }
        case .emptyPrompt:                                           // empty_prompt
            return String(localized: "The prompt is empty.")
        case let .invalidSteps(steps, allowed):                      // invalid_steps
            return allowed.isEmpty ? String(localized: "\(steps) steps: a render takes between 1 and \(Request.maximumSteps).")
                : String(localized: "This model doesn’t render in \(steps) steps.")
        case let .tooManyReferences(count, max):                     // too_many_references
            return max == 0 ? String(localized: "This model doesn’t edit images.")
                : String(localized: "\(count) images: this model reads \(max) at most.")
        case let .imageTooSmall(side, minimum):                      // image_too_small
            return String(localized: "Too small: \(String(side)) px on a side, \(String(minimum)) px at least.")
        case let .formatRefused(w, h, reason, bound):                // format_refused
            switch reason {
            case .notMultiple:
                return String(localized: "\(Figures.size(w, h)): each side must be a multiple of \(String(bound)).")
            case .sideTooLong:
                return String(localized: "\(Figures.size(w, h)): this model draws \(String(bound)) px at most on a side.")
            }
        case .formatUnreadable(let t):                               // format_unreadable
            return String(localized: "“\(t)” isn’t a size: write it as width × height, like 832x1216.")
        case let .promptSyntax(position, reason):                    // prompt_syntax
            switch reason {
            case .unclosedGroup:
                return String(localized: "The prompt opens a group of alternatives that is never closed (character \(String(position))).")
            case .unmatchedClose:
                return String(localized: "The prompt closes a group of alternatives that was never opened (character \(String(position))).")
            case .nestedGroup:
                return String(localized: "Groups of alternatives can’t be nested (character \(String(position))).")
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
            case .sameAxisTwice: return String(localized: "The grid’s two axes vary the same setting.")
            case .noSuchGroup: return String(localized: "An axis of the grid names a group of alternatives the prompt doesn’t have.")
            case .noSuchLoRA: return String(localized: "An axis of the grid names a LoRA that isn’t in the stack.")
            case .groupNotOnAxis: return String(localized: "The prompt has a group of alternatives that isn’t an axis of the grid.")
            case .invalidValue: return String(localized: "An axis of the grid has a value no render takes.")
            case .unreadableAxis: return String(localized: "An axis of the grid can’t be read.")
            case .loraAcrossModels: return String(localized: "A grid that varies the model can’t take a LoRA: each LoRA is made for one model.")
            case .noStartImage: return String(localized: "A grid that varies the starting image’s strength needs a starting image.")
            }
        case let .tooManyGridCells(count, max):                      // too_many_grid_cells
            return String(localized: "This grid makes \(String(count)) cells: \(String(max)) at most.")
        case .strengthOutOfRange(let s):                             // strength_out_of_range
            return String(localized: "The strength must be between 0 and 1 (\(Figures.strength(s))).")
        case let .strengthTooLow(s, steps, _):                       // strength_too_low
            return String(localized: "A strength of \(Figures.strength(s)) leaves no step out of \(steps).")
        case .imageToImageUnsupported(let m):                        // image_to_image_unsupported
            return String(localized: "\(name(m)) doesn’t start from an image.")
        case let .solverUnsupported(s, m):                           // solver_unsupported
            // The solver's name as the rack shows it (« DPM++ 2M »), not its raw value (« dpmpp-2m »).
            return String(localized: "\(name(m)) doesn’t offer the \(Solver(rawValue: s)?.choice ?? s) solver.")
        case .detailUnsupported(let m):                              // detail_unsupported
            return String(localized: "\(name(m)) doesn’t offer Detail.")
        case .imageUnreadable(let f):                                // image_unreadable
            return String(localized: "This image can’t be read: \(file(f)).")
        case .imageEncodingFailed:                                   // image_encoding_failed
            return String(localized: "The image couldn’t be encoded as PNG.")
        case .imageWriteFailed(let f):                               // image_write_failed
            return String(localized: "The image couldn’t be saved to \(f).")
        case let .unsupportedLoRA(f, reason):                        // unsupported_lora
            switch reason {
            case .notALoRA: return String(localized: "\(file(f)) is a model, not a LoRA.")
            case .noTarget: return String(localized: "The LoRA \(file(f)) adapts modules that none of Siliconed’s models has.")
            case .inconsistent: return String(localized: "The LoRA \(file(f)) is damaged.")
            case .unsupportedKind:
                return String(localized: "The LoRA \(file(f)) is of a kind Siliconed doesn’t apply: LoKr, LoHa, DoRA, rsLoRA, a text encoder’s, or ranks that vary by module.")
            }
        case let .loraForOtherModel(lora, target, model):            // lora_for_other_model
            return String(localized: "The LoRA \(lora) is for \(name(target)), not for \(name(model)).")
        case let .incompatibleChain(output, input):                  // incompatible_chain
            return String(localized: "These modules don’t connect: \(output) comes out where \(input) is expected.")
        case .notOnEngine(let m):                                    // not_on_engine
            return String(localized: "\(name(m)) doesn’t run on this engine yet.")
        case let .insufficientMemory(needed, budget):                // insufficient_memory
            // Within the limit: the soberest form stopped the fuse (`Stage.choose`), the number known is the soberest's socle, not a need.
            if needed <= budget {
                return String(localized: "This render went past this Mac’s memory limit of \(Figures.memory(budget)) every time it was tried: it is refused.")
            }
            return String(localized: "This render needs \(Figures.memory(needed)) of memory, more than this Mac’s limit of \(Figures.memory(budget)).")
        case .renderAlreadyRunning:                                  // render_already_running
            // The lock is the machine's (`RenderLock.machine`), whatever the library.
            return String(localized: "Another program is already rendering on this Mac.")
        case .engineStopped(let budget?, _):                         // engine_stopped
            return String(localized: "The engine went past this Mac’s memory limit of \(Figures.memory(budget)) and was stopped.")
        case .engineStopped:
            return String(localized: "The engine stopped unexpectedly.")
        case .engineMismatch:                                        // engine_mismatch
            return String(localized: "Siliconed’s engine comes from another build than the app: it was refused.")
        case .engineSilent(let seconds):                             // engine_silent
            return String(localized: "The engine stopped answering for \(seconds / 60) minutes and was stopped.")
        case .cancelled:                                             // cancelled
            return String(localized: "Stopped.")
        case .nonFiniteImage(.decoding):                             // non_finite_image
            return String(localized: "This render went wrong in the VAE and its image is unusable: it wasn’t kept.")
        case .nonFiniteImage:
            return String(localized: "This render went wrong in the DiT and its image is unusable: it wasn’t kept.")
        case .internalFailure:                                       // internal_failure
            return String(localized: "The engine failed.")
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
            return String(localized: "Check the file’s permissions, and that its disk is connected.")
        case .corruptMap:
            return String(localized: "Reinstall the model from the Models sheet, or import it again.")
        case .diskFull(_, _, .image):
            return String(localized: "Make room on the disk — uninstall a model in the Models sheet, or delete images from the folder they are saved to — then render again.")
        case .diskFull:
            return String(localized: "Make room — uninstall a model, or discard the interrupted downloads in the Models sheet — then start again: the files already downloaded are kept.")
        case .downloadInterrupted:
            return String(localized: "Check the connection and start again: the files already downloaded are kept, and a large file resumes where it stopped.")
        case let .downloadRefused(_, status) where status == 401 || status == 403:
            // No repository the app reads is gated and no request carries a token (`Downloader`): a refusal comes
            // from the way to the Hub, not from a license nor an account.
            return String(localized: "Check that huggingface.co opens in a browser on this Mac — a proxy, a VPN or a firewall can refuse it — then install again: the files already downloaded are kept.")
        case .downloadRefused:
            return String(localized: "Try again later.")
        case .downloadCorrupt:
            return String(localized: "Start again: the damaged file was thrown away.")
        case .downloadInProgress:
            return String(localized: "Wait for that download to finish, then start again.")
        case .importRefused:
            return String(localized: "Import a LoRA or a full model of a supported family, as .safetensors.")
        case .importNeedsFamily:
            return String(localized: "Install it from the Models sheet (⇧⌘M), then import the file again: an import downloads nothing.")
        case .importBaseModel(_, let m):
            return String(localized: "Install \(name(m)) from the Models sheet (⇧⌘M), or import a model made from it.")
        case .libraryAbsent:
            return String(localized: "Connect the disk — or, in Settings, go back to this Mac’s library.")
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
        case .notAnImportedModel, .importDeeperModel, .imageEncodingFailed, .cancelled:
            return nil
        case .emptyPrompt:
            return String(localized: "Write a prompt.")
        case let .invalidSteps(_, allowed) where !allowed.isEmpty:
            return String(localized: "Choose \(allowed.map(String.init).formatted(.list(type: .or))) steps.")
        case .invalidSteps:
            return String(localized: "Leave the number of steps at the model’s default.")
        case .tooManyReferences(_, let max) where max == 0:
            return String(localized: "Choose a model that edits images.")
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
            case .invalidValue: return String(localized: "Steps are whole numbers from 1, a grid needs at least one seed, and a negative LoRA strength needs Settings › Advanced › Negative LoRA Strengths.")
            case .unreadableAxis, .emptyAxis, .noSuchGroup, .noSuchLoRA:
                return String(localized: "Give each axis its values: a group of the prompt, seeds, steps, a LoRA of the stack, models or the edit’s images.")
            case .loraAcrossModels: return String(localized: "Remove the LoRAs, or vary something other than the model.")
            case .noStartImage: return String(localized: "Start from an image, or vary something else.")
            }
        case .tooManyGridCells:
            return String(localized: "Remove values from an axis.")
        case .imageTooSmall, .formatRefused, .formatUnreadable:
            return String(localized: "Choose one of the model’s sizes, like 1024 × 1024 or 832 × 1216.")
        case .strengthOutOfRange:
            return String(localized: "Choose a strength between 0 and 1.")
        case .strengthTooLow:
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
        case .unsupportedLoRA(_, .noTarget):
            return String(localized: "Choose a LoRA made for one of Siliconed’s models.")
        case .unsupportedLoRA(_, .unsupportedKind):
            return String(localized: "Choose a plain LoRA made for the model’s DiT.")
        case .unsupportedLoRA:
            return String(localized: "Import the LoRA again from its .safetensors.")
        case .loraForOtherModel:
            return String(localized: "Choose a LoRA made for this model.")
        case .incompatibleChain:
            return String(localized: "Report it: the app built a chain the engine refuses.")
        case .notOnEngine:
            return String(localized: "Choose another model.")
        case .insufficientMemory:
            return String(localized: "Choose a smaller size, fewer images or a lighter model, or raise the memory limit in Settings › Macs.")
        case .renderAlreadyRunning:
            return String(localized: "Wait for it to finish, then render again.")
        case .engineStopped(_?, _):
            return String(localized: "What waits goes on, in a new engine, which plans this render more narrowly now. If it stops again, choose a smaller size or a lighter model.")
        case .engineStopped:
            return String(localized: "What waits goes on, in a new engine. If it happens again, report it (Help › Report a Problem…).")
        case .engineMismatch:
            return String(localized: "Install Siliconed again: the app and its engine must come from the same build.")
        case .engineSilent:
            return String(localized: "What waits goes on, in a new engine. If it happens again, report it (Help › Report a Problem…).")
        case .internalFailure, .nonFiniteImage:
            return String(localized: "Try again; if it persists, report it (Help › Report a Problem…).")
        }
    }
}

// MARK: - The engine's other texts, said by the app

// The engine also writes English text that is not an error: a license's name, the warnings of a render. A license's name is
// its family's to say (`FamilyPresentation.licenseName`); what is free (a developer's warning) is never shown raw in a
// sentence of the app — a localized line says what happened, and the engine's text stays in its tooltip, like a problem's
// detail.

extension ModelCard {
    /// The license as the app says it: its family's name for it (`FamilyPresentation.licenseName`), else the engine's text
    /// (`License.text`, English, for the JSON) — a name that stays the same in every language.
    var licenseName: String {
        guard isImported else { return Self.licenseName(of: family, license) }
        let f = ModelCard.of(family)
        let architecture = f.name, inherited = Self.licenseName(of: family, f.license)
        return String(localized: "that of the imported model (architecture \(architecture): \(inherited))")
    }

    /// **Whether the images it makes may be used commercially** — a license may say yes to its outputs and no to the model
    /// (`FamilyTraits.commercialImages`). An imported model: its family's, like its license's name.
    var imagesCommercial: Bool { license.commercial || family.traits.commercialImages }

    private static func licenseName(of family: Family, _ license: License) -> String {
        family.traits.presentation.licenseName.map { String(localized: $0) } ?? license.text
    }
}

/// **A job on the library, as a line of the app**: an installation says where it is (`Installer.Progress`), a move of
/// the library writes rows for a terminal (`⇢ <percent> <file>`); the app shows what they mean.
enum LibraryJobProgress {
    /// An installation: the file coming, and how far the whole installation is.
    static func line(_ p: Installer.Progress) -> String {
        line(file: (p.file as NSString).lastPathComponent, received: p.received, total: p.total)
    }

    static func line(file: String, received: Int, total: Int) -> String {
        guard total > 0 else { return String(localized: "Downloading \(file)") }
        let percent = Figures.percent(min(100, received * 100 / total))
        return String(localized: "Downloading \(file): \(percent)")
    }

    /// A library being moved (`AppState.moveLibrary`): its last « ⇢ <percent> <file> ».
    /// A file going to another Mac, or coming from one (`UploadProgress`).
    static func line(_ journal: [String]) -> String? {
        if let upload = UploadProgress.line(journal) { return upload }
        for row in journal.reversed() {
            let t = row.trimmingCharacters(in: .whitespaces)
            guard t.hasPrefix("⇢ ") else { continue }
            let parts = t.dropFirst(2).split(separator: " ", maxSplits: 1)
            if parts.count == 2, let n = Int(parts[0]) { return String(localized: "\(Figures.percent(n)) — \(String(parts[1]))") }
        }
        return nil
    }
}

extension AppState {
    /// The warnings of the last render, said by the app; the engine's own rows are its tooltip.
    var journalLine: String? {
        // A lost image is the problem's to say (`execute`), not the engine's noting something unusual.
        let journal = journal.filter { !LostImage.isWarning($0) }
        if journal.isEmpty { return nil }
        // The one warning that is no anomaly: the plan chose a slower decoding, and the render says so.
        if journal.allSatisfy({ $0 == bandDecodingNotice }) {
            return String(localized: "Decoded in bands to stay within this Mac’s memory limit: the same image, a longer decoding.")
        }
        // A shared file an older app left was not the published bytes: removed before the render, which it refused.
        if journal.contains(where: { $0.hasPrefix(sharedFileRemovedNotice) }) {
            return String(localized: "A file of this model wasn’t the published one: it was removed, and installing the model again downloads it alone.")
        }
        return String(localized: "The engine noted something unusual during this render.")
    }
}
