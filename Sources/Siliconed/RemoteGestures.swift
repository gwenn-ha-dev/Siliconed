// What `silicontrol` asks of the app for the window's other gestures — the history's (remove, undo, again, reuse, crop), the series'
// (add its sheet, a knob's bubble), the models' (import, uninstall), Compute On, the library, the Prompt Lab's bar, the rack's scroll
// and the Settings window — kept here, pure, so that the tests read the
// arguments, the way a cell is named, a crop's frame and the `--yes` guard without an app.
//
// The gestures themselves are the app's (`AppState`, `MainWindow`): a command calls the function the button calls.

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
    /// `undo [--redo]`: ⌘Z, or ⇧⌘Z.
    case undo(redo: Bool)
    /// `again <i7|file.png>`: Render Again (Same Seed), from an image or from a Siliconed PNG.
    case again(String)
    /// `reuse <i7|file.png>`: Reuse These Settings, without rendering.
    case reuse(String)
    /// `crop i7 --box X,Y,W,H | --ratio P:Q`.
    case crop(image: String, CropSpec)
    /// `sheet g2`: Add Sheet to History.
    case sheet(grid: Int)
    /// `import FILE …`: Import a Model or LoRA.
    case importFiles([String])
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
    /// `lab …`: the Prompt Lab, the bar under the image — what it holds, folded or unfolded, an image in, a role, an image out, and its
    /// two buttons, Rewrite and Describe, written in its text.
    case lab(LabAction)
    /// `settings …`: Siliconed › Settings… (⌘,), on a pane — and, on Prompt Lab, on one instruction.
    case settings(SettingsAction)
    /// `bracket KNOB` · `bracket close`: a knob's ± clicked (or right-click › « Bracket… »), its bubble open — or closed.
    case bracket(BracketAction)
    /// `rack top|dit`: the rack's column scrolled to its head, or to the DiT's settings (the chain then pinned in small at the top).
    case rack(RackPlace)

    package enum Removal: Equatable, Sendable {
        /// A model's id: a publisher's family is uninstalled, an imported model removed.
        case model(String)
        case lora(String)
        case downloads
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
        case unfold, fold
        /// The unfolded plate in a window of its own (`detach`), or back over the image (`attach`).
        case detach, attach
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
        /// The Seed line's ±: the lot.
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
    package enum RackPlace: String, Equatable, Sendable { case top, dit }

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

    package static let verbs: Set<String> = ["remove", "undo", "again", "reuse", "crop", "sheet", "import", "uninstall",
                                             "macs", "compute", "library", "start-image", "lab", "settings", "bracket", "rack"]

    package static func usage(_ command: String) -> String {
        switch command {
        case "remove": "remove <i7> [i8 …]"
        case "undo": "undo [--redo]"
        case "again": "again <i7|file.png>"
        case "reuse": "reuse <i7|file.png>"
        case "crop": "crop <i7> --box X,Y,W,H | --ratio P:Q"
        case "sheet": "sheet <g2>"
        case "import": "import <file.safetensors|file.gguf> …"
        case "uninstall": "uninstall <model-id> [--yes] · uninstall lora <name> [--yes] · uninstall downloads [--yes]"
        case "macs": "macs [connect|disconnect NAME]"
        case "compute": "compute <NAME|this>"
        case "start-image": "start-image [<image|i7>] [--strength 0.65] · start-image --remove"
        case "lab": "lab [open|close] · lab add <image|i7> … · lab role <n> edited|reference|start|none · lab remove <n> · lab rewrite [--wait] · lab describe <n> [--wait]"
        case "settings": "settings [general|macs|lab|advanced] · settings lab [rewrite|describe] [--for ID]"
        case "bracket": "bracket seed|detail|solver|start|loras · bracket lora <name> · bracket close"
        case "rack": "rack top|dit"
        default: "library [move DIR|home|open DIR|local] [--yes]"
        }
    }

    /// **The guard**: `--yes` is the only thing that makes a destructive command act; without it the app says what it would do.
    /// Only the commands that take it accept it — to the others it is an unknown flag, never silently ignored.
    package static func takesYes(_ command: String, _ arguments: [String]) -> Bool {
        switch command {
        case "uninstall": true
        case "library": ["move", "home"].contains(arguments.first { !$0.hasPrefix("--") })
        default: false
        }
    }

    package static func parse(_ command: String, _ arguments: [String]) -> Result<GestureOrder, Failure> {
        let line = usage(command)
        var words: [String] = []
        var yes = false
        var values: [String: String] = [:]
        let valued: Set<String> = switch command {
        case "crop": ["--box", "--ratio"]
        case "start-image": ["--strength"]
        case "settings": ["--for"]
        default: []
        }
        var i = 0
        while i < arguments.count {
            let a = arguments[i]
            if valued.contains(a) {
                guard i + 1 < arguments.count else { return .failure(.flagWithoutValue(a)) }
                guard values[a] == nil else { return .failure(.usage(line)) }
                values[a] = arguments[i + 1]
                i += 2
                continue
            }
            if a == "--yes", takesYes(command, arguments) { yes = true; i += 1; continue }
            if command == "undo", a == "--redo" { values[a] = ""; i += 1; continue }
            if command == "start-image", a == "--remove" { values[a] = ""; i += 1; continue }
            if command == "lab", a == "--wait" { values[a] = ""; i += 1; continue }
            if a.hasPrefix("--"), a.count > 2, !a.contains(" ") { return .failure(.unknownFlag(a)) }
            words.append(a)
            i += 1
        }
        func one() -> String? { words.count == 1 ? words[0] : nil }

        switch command {
        case "remove":
            return words.isEmpty ? .failure(.usage(line)) : .success(.remove(words))
        case "undo":
            return words.isEmpty ? .success(.undo(redo: values["--redo"] != nil)) : .failure(.usage(line))
        case "again": return one().map { .success(.again($0)) } ?? .failure(.usage(line))
        case "reuse": return one().map { .success(.reuse($0)) } ?? .failure(.usage(line))
        case "crop":
            guard let image = one(), values.count == 1 else { return .failure(.usage(line)) }
            if let v = values["--box"] {
                let n = v.split(separator: ",", omittingEmptySubsequences: false).map { Int($0.trimmingCharacters(in: .whitespaces)) }
                guard n.count == 4, !n.contains(where: { $0 == nil }) else { return .failure(.badValue("--box", v)) }
                return .success(.crop(image: image, .box(Crop.Box(x: n[0]!, y: n[1]!, width: n[2]!, height: n[3]!))))
            }
            let v = values["--ratio"] ?? ""
            let sides = v.split(separator: ":", omittingEmptySubsequences: false).map { Int($0) }
            guard sides.count == 2, let w = sides[0], let h = sides[1], w > 0, h > 0 else { return .failure(.badValue("--ratio", v)) }
            return .success(.crop(image: image, .ratio(Crop.Ratio(w, h))))
        case "sheet":
            guard let word = one() else { return .failure(.usage(line)) }
            guard word.first == "g", let n = Int(word.dropFirst()), n >= 0 else { return .failure(.badValue("grid", word)) }
            return .success(.sheet(grid: n))
        case "import":
            return words.isEmpty ? .failure(.usage(line)) : .success(.importFiles(words))
        case "uninstall":
            if words == ["downloads"] { return .success(.uninstall(.downloads, confirmed: yes)) }
            if words.count == 2, words[0] == "lora" { return .success(.uninstall(.lora(words[1]), confirmed: yes)) }
            if words.count == 1, words[0] != "lora" { return .success(.uninstall(.model(words[0]), confirmed: yes)) }
            return .failure(.usage(line))
        case "macs":
            if words.isEmpty { return .success(.macs(.list)) }
            guard words.count == 2, ["connect", "disconnect"].contains(words[0]) else { return .failure(.usage(line)) }
            return .success(.macs(words[0] == "connect" ? .connect(words[1]) : .disconnect(words[1])))
        case "compute":
            guard let word = one() else { return .failure(.usage(line)) }
            return .success(.compute(word.lowercased() == "this" ? .thisMac : .mac(word)))
        case "start-image":
            var strength: Double?
            if let v = values["--strength"] {
                guard let s = StartImage.strength(v) else { return .failure(.badValue("--strength", v)) }
                strength = s
            }
            if values["--remove"] != nil {
                return words.isEmpty && strength == nil ? .success(.startImage(.remove)) : .failure(.usage(line))
            }
            switch (words.count, strength) {
            case (0, nil): return .success(.startImage(.status))
            case (0, let s?): return .success(.startImage(.strength(s)))
            case (1, let s): return .success(.startImage(.carry(words[0], strength: s)))
            default: return .failure(.usage(line))
            }
        case "lab":
            let rest = Array(words.dropFirst())
            func number(_ word: String) -> Int? { Int(word).flatMap { $0 >= 1 ? $0 : nil } }
            // `--wait` belongs to the two gestures that write; anywhere else it is a mistake, never ignored.
            let wait = values["--wait"] != nil
            if wait, !["rewrite", "describe"].contains(words.first ?? "") { return .failure(.usage(line)) }
            switch words.first {
            case nil: return .success(.lab(.status))
            case "open"?: return rest.isEmpty ? .success(.lab(.unfold)) : .failure(.usage(line))
            case "close"?: return rest.isEmpty ? .success(.lab(.fold)) : .failure(.usage(line))
            case "detach"?: return rest.isEmpty ? .success(.lab(.detach)) : .failure(.usage(line))
            case "attach"?: return rest.isEmpty ? .success(.lab(.attach)) : .failure(.usage(line))
            case "add"?: return rest.isEmpty ? .failure(.usage(line)) : .success(.lab(.add(rest)))
            case "role"?:
                guard rest.count == 2 else { return .failure(.usage(line)) }
                guard let n = number(rest[0]) else { return .failure(.badValue("image", rest[0])) }
                guard let role = LabAction.role(named: rest[1]) else { return .failure(.badValue("role", rest[1])) }
                return .success(.lab(.role(image: n, role)))
            case "remove"?:
                guard rest.count == 1 else { return .failure(.usage(line)) }
                return number(rest[0]).map { .success(.lab(.remove(image: $0))) } ?? .failure(.badValue("image", rest[0]))
            case "rewrite"?: return rest.isEmpty ? .success(.lab(.rewrite(wait: wait))) : .failure(.usage(line))
            case "describe"?:
                guard rest.count == 1 else { return .failure(.usage(line)) }
                return number(rest[0]).map { .success(.lab(.describe(image: $0, wait: wait))) } ?? .failure(.badValue("image", rest[0]))
            default: return .failure(.usage(line))
            }
        case "settings":
            guard words.count <= 2 else { return .failure(.usage(line)) }
            var pane: SettingsPane?
            if let word = words.first {
                guard let p = SettingsPane(rawValue: word.lowercased()) else { return .failure(.badValue("pane", word)) }
                pane = p
            }
            let target = values["--for"]
            guard words.count == 2 || target != nil else { return .success(.settings(.show(pane))) }
            // A gesture and a model are the Prompt Lab pane's two menus: no other pane has them.
            guard pane == .lab else { return .failure(.usage(line)) }
            var gesture: PromptWriterInstructions.Gesture?
            if words.count == 2 {
                guard let g = PromptWriterInstructions.Gesture(rawValue: words[1].lowercased()) else { return .failure(.badValue("gesture", words[1])) }
                gesture = g
            }
            return .success(.settings(.instruction(gesture, target: target)))
        case "bracket":
            switch words.first?.lowercased() {
            case "close"?: return words.count == 1 ? .success(.bracket(.close)) : .failure(.usage(line))
            case "lora"?: return words.count == 2 ? .success(.bracket(.open(.lora(words[1])))) : .failure(.usage(line))
            case let w?:
                guard words.count == 1 else { return .failure(.usage(line)) }
                return BracketKnob.named(w).map { .success(.bracket(.open($0))) } ?? .failure(.badValue("knob", words[0]))
            case nil: return .failure(.usage(line))
            }
        case "rack":
            guard words.count == 1 else { return .failure(.usage(line)) }
            return RackPlace(rawValue: words[0].lowercased()).map { .success(.rack($0)) } ?? .failure(.badValue("place", words[0]))
        default:
            switch words.first {
            case nil: return .success(.library(.status, confirmed: false))
            case "move"?: return words.count == 2 ? .success(.library(.move(words[1]), confirmed: yes)) : .failure(.usage(line))
            case "home"?: return words.count == 1 ? .success(.library(.home, confirmed: yes)) : .failure(.usage(line))
            case "open"?: return words.count == 2 ? .success(.library(.open(words[1]), confirmed: false)) : .failure(.usage(line))
            case "local"?: return words.count == 1 ? .success(.library(.local, confirmed: false)) : .failure(.usage(line))
            default: return .failure(.usage(line))
            }
        }
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
