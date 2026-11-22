// What `silicontrol` asks of the app for the window's other gestures — the history's (remove, undo, again, reuse, crop), the series'
// (its sheet, its removal, its settings reused, a knob's bubble), the models' (import, uninstall), Compute On, the library, the Prompt Lab's bar, the rack's scroll
// and the Settings window — kept here, pure, so that the tests read the
// arguments, the way a cell is named, a crop's frame and the `--yes` guard without an app.
//
// The gestures themselves are the app's (`AppState`, `MainWindow`): a command calls the function the button calls.

import Measurement

// MARK: - Naming a cell

/// **A cell of a grid**, as the answers name it: `g2c3` — the grid and the cell's rank in the sheet's reading order, from 0 (row by
/// row: `row · columns + column`, the same numbers the job's `column` and `row` give). Every cell is also an ordinary job, `t5`.
///
/// One letter and a number for each kind of thing (`t3` a job, `i7` an image, `g2` a grid), and a cell is a part of a grid: `g2`, then
/// `c` and its rank. A pair of coordinates (`g2:1,0`) would need quotes in some shells and a sheet's width to be read right.
package enum CellName {
    package static func name(grid: Int, cell: Int) -> String { "g\(grid)c\(cell)" }
}

// MARK: - A crop

/// **What `crop` is asked**: a frame in pixels of the image (`x,y,w,h`, the origin at the top left), or a proportion — the largest
/// frame of that proportion, centered, which is what the editor's proportion menu starts from.
package enum CropSpec: Equatable, Sendable {
    case box(Crop.Box)
    case ratio(Crop.Ratio)

    package enum Failure: Error, Equatable, Sendable {
        /// Smaller than the image can give: both sides at `Format.minimumSide` or more, one of them above it.
        case cannotCrop(minimum: Int)
        case outside
        case belowFloor(minimum: Int)
        /// The whole image is not a crop (`commitCrop` does nothing with it).
        case whole
        case unsupportedRatio
    }

    /// **The frame, or why not** — with the rules the editor's own frame follows (`Crop.Frame`).
    package func resolve(width: Int, height: Int) -> Result<Crop.Box, Failure> {
        let frame = Crop.Frame(width: width, height: height)
        guard frame.canCrop else { return .failure(.cannotCrop(minimum: frame.minimum)) }
        let box: Crop.Box
        switch self {
        case .box(let b):
            guard b.x >= 0, b.y >= 0, b.width > 0, b.height > 0, b.maxX <= width, b.maxY <= height else { return .failure(.outside) }
            guard b.width >= frame.minimum, b.height >= frame.minimum else { return .failure(.belowFloor(minimum: frame.minimum)) }
            box = b
        case .ratio(let r):
            guard frame.supports(.fixed(r)), let fitted = frame.fit(frame.whole, to: r) else { return .failure(.unsupportedRatio) }
            box = fitted
        }
        return box == frame.whole ? .failure(.whole) : .success(box)
    }
}

// MARK: - The orders

/// **The arguments of the gestures that take few**, read. One parse for them all; the app resolves what the words name (an image of the
/// history, a job, a Mac) and applies the rules the window applies.
package enum GestureOrder: Equatable, Sendable {
    /// `remove i7 [i8 …]`: Remove from History.
    case remove([String])
    /// `remove g2`: a series' tile › « Remove Series from History ».
    case removeSeries(grid: Int)
    /// `undo [--redo]`: ⌘Z, or ⇧⌘Z.
    case undo(redo: Bool)
    /// `again <i7|file.png>`: Render Again (Same Seed), from an image or from a Siliconed PNG.
    case again(String)
    /// `reuse <i7|file.png>`: Reuse These Settings, without rendering.
    case reuse(String)
    /// `reuse g2`: a series' tile › « Reuse This Grid ».
    case reuseSeries(grid: Int)
    /// `crop i7 --box X,Y,W,H | --ratio P:Q`.
    case crop(image: String, CropSpec)
    /// `sheet g2 <file.png|folder>`: Export Sheet — the series' images side by side, labelled, written there.
    case sheet(grid: Int, to: String)
    /// `uninstall …`: Uninstall, Remove (a model, a LoRA), Discard (the interrupted downloads). `confirmed`: `--yes`.
    case uninstall(Removal, confirmed: Bool)
    /// `macs`, `macs connect NAME`, `macs disconnect NAME`.
    case macs(MacsAction)
    /// `compute NAME|this`: Render › Compute On.
    case compute(ComputeTarget)
    /// `library …`.
    case library(LibraryAction, confirmed: Bool)
    /// `start-image …`: the rack's starting image (img2img) — an image of the Prompt Lab given the start's role, and its strength.
    case startImage(StartImageAction)
    /// `lab …`: the Prompt Lab, the window's first column — what it holds, folded or unfolded, an image in, a role, an image out, and its
    /// two buttons, Rewrite and Describe, written in its text.
    case lab(LabAction)
    /// `settings …`: Siliconed › Settings… (⌘,), on a pane — and, on Prompt Lab, on one instruction.
    case settings(SettingsAction)
    /// `bracket KNOB` · `bracket close`: a knob's ± clicked (or right-click › « Bracket… »), its bubble open — or closed.
    case bracket(BracketAction)
    /// `rack top|dit|fold|unfold|detach|dock`: the rack's column folded away or unfolded, or scrolled to its head, or to the DiT's settings
    /// (the chain then pinned in small at the top), or put in a window of its own and back into the main one.
    case rack(RackPlace)
    /// `snapshot FILE.png [--window WORD|N] [--appearance light|dark]`: one of the app's windows, drawn by the app into a PNG.
    case snapshot(file: String, window: String?, appearance: SnapshotAppearance?)

    package enum Removal: Equatable, Sendable {
        /// A model's id: its family is uninstalled — one of its versions (`standard`, `compact`, as typed) or all of it;
        /// an imported model's: its file alone.
        case model(String, version: String?)
        case downloads
        /// The Models sheet's « Free… »: what version 0.21 left in the library and this one never reads.
        case legacy
    }
    package enum MacsAction: Equatable, Sendable { case list, connect(String), disconnect(String) }
    package enum ComputeTarget: Equatable, Sendable { case thisMac, mac(String) }
    package enum LibraryAction: Equatable, Sendable {
        case status
        /// Move Library…: the folder chosen (a `Siliconed` folder is made in it unless it is one).
        case move(String)
        /// Move Back to This Mac.
        case home
        /// Open Library…: a folder that already holds one.
        case open(String)
        /// Use This Mac's Library.
        case local
    }

    /// What `start-image` does: say what the rack holds, carry an image (a file, or `i7` of the history) with its strength or the
    /// rack's, set the strength of the one carried, or take it out.
    package enum StartImageAction: Equatable, Sendable {
        case status
        case carry(String, strength: Double?)
        case strength(Double)
        case remove
    }

    /// What `lab` does. The images are numbered as the bar shows them, from 1.
    package enum LabAction: Equatable, Sendable {
        case status
        /// The text, as typed in the field: the rack's prompt.
        case text(String)
        /// « Generate » clicked: what the rack and the lab hold goes to the queue.
        case generate
        /// Files, or `i7` of the history: each takes the role a newcomer takes.
        case add([String])
        case role(image: Int, LabInputs.Role)
        case remove(image: Int)
        /// The bar's « Rewrite », written in its text. `wait`: the answer comes when the text is written, not when it starts.
        case rewrite(wait: Bool)
        /// Image N's « Describe », written in the text at the caret.
        case describe(image: Int, wait: Bool)

        /// A role as the command names it: `edited`, `reference`, `start`, `none`.
        package static func role(named word: String) -> LabInputs.Role? {
            switch word.lowercased() {
            case "edited": .edited
            case "reference": .reference
            case "start": .start
            case "none": .neutral
            default: nil
            }
        }

        package static func name(_ role: LabInputs.Role) -> String { role == .neutral ? "none" : role.rawValue }
    }

    /// **A knob of the rack, as `bracket` names it** — the ± the bubble opens from. The text's groups have their own bubble, over
    /// the words (`Vary This Word…`), and no name here.
    package enum BracketKnob: Equatable, Sendable {
        /// The Seed line's ±: the lot, the Seeds count.
        case seeds
        case detail
        case solver
        /// The starting image's strength.
        case start
        /// The LoRA stack's ±: « Compare LoRAs… ».
        case compareLoRAs
        /// One LoRA's strength, by its name in the stack (as `lab` and `status` print it), or its path.
        case lora(String)

        package static func named(_ word: String) -> BracketKnob? {
            switch word.lowercased() {
            case "seed", "seeds", "batch": .seeds
            case "detail": .detail
            case "solver": .solver
            case "start", "strength": .start
            case "loras", "compare": .compareLoRAs
            default: nil
            }
        }
    }
    package enum BracketAction: Equatable, Sendable { case open(BracketKnob), close }
    package enum RackPlace: String, Equatable, Sendable { case top, dit, fold, unfold, detach, dock }
    package enum SnapshotAppearance: String, Equatable, Sendable { case light, dark }

    /// What `settings` opens: a pane, or Settings › Prompt Lab on one instruction — the model and the gesture its two menus show,
    /// each left as it is when not named.
    package enum SettingsAction: Equatable, Sendable {
        /// `nil`: the pane last shown, as ⌘, opens it.
        case show(SettingsPane?)
        /// `target`: a model's id or a family, as typed (the app resolves it).
        case instruction(PromptWriterInstructions.Gesture?, target: String?)
    }

    package enum Failure: Error, Equatable, Sendable {
        /// Not the syntax; the associated value is the usage line.
        case usage(String)
        case unknownFlag(String)
        case flagWithoutValue(String)
        /// `(flag, value)`: a value that does not read.
        case badValue(String, String)
    }
}

// MARK: - The panes of Settings

/// **The Settings window's four panes**, in the order of its tabs, by the word `settings` takes — the window draws them
/// (`SettingsWindow.Pane`) from this one list.
package enum SettingsPane: String, CaseIterable, Sendable {
    case general, macs, lab, advanced
}

// MARK: - A Mac by name

/// **Which paired Mac a word names**: its name (case and diacritics aside) or its id. `nil` when none, or when several carry the same
/// name — a name two Macs share is an order the command cannot guess.
package enum MacName {
    package static func match(_ word: String, among macs: [(id: String, name: String)]) -> (id: String, name: String)? {
        let key = word.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
        if let byID = macs.first(where: { $0.id == word }) { return byID }
        let named = macs.filter { $0.name.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil) == key }
        return named.count == 1 ? named[0] : nil
    }
}
