// What `silicontrol` asks of the app for the window's other gestures — the history's (remove, undo, again, reuse, crop), the exploration's
// (finish a cell, add the sheet), the models' (import, uninstall), Compute On and the library — kept here, pure, so that the tests read
// the arguments, the way a cell is named, a crop's frame and the `--yes` guard without an app.
//
// The gestures themselves are the app's (`AppState`, `MainWindow`): a command calls the function the button calls.

// MARK: - Naming a cell

/// **A cell of an exploration**, as `finish` names it: `g2c3` — the grid and the cell's rank in the sheet's reading order, from 0
/// (row by row: `row · columns + column`, the same numbers the job's `column` and `row` give) — or `t5`, the job that renders it:
/// every cell is an ordinary job, so the name `wait` and `cancel` already know is one more way to say it.
///
/// One letter and a number for each kind of thing (`t3` a job, `i7` an image, `g2` a grid), and a cell is a part of a grid: `g2`, then
/// `c` and its rank. A pair of coordinates (`g2:1,0`) would need quotes in some shells and a sheet's width to be read right.
package enum CellName: Equatable, Sendable {
    case cell(grid: Int, index: Int)
    case job(Int)

    package init?(_ word: String) {
        if let rest = word.first == "g" ? word.dropFirst() : nil, let c = rest.firstIndex(of: "c"),
           let grid = Int(rest[..<c]), let index = Int(rest[rest.index(after: c)...]), grid >= 0, index >= 0,
           rest.allSatisfy({ $0.isNumber || $0 == "c" }) {
            self = .cell(grid: grid, index: index)
        } else if word.first == "t", let n = Int(word.dropFirst()), n >= 0 {
            self = .job(n)
        } else {
            return nil
        }
    }

    /// How the answers say it.
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
    /// `finish g2c3|t5`: Finish This Image.
    case finish(CellName)
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
    /// `start-image …`: the rack's starting image (img2img), as the Prompt Lab carries it and the Images box retouches it.
    case startImage(StartImageAction)

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

    package enum Failure: Error, Equatable, Sendable {
        /// Not the syntax; the associated value is the usage line.
        case usage(String)
        case unknownFlag(String)
        case flagWithoutValue(String)
        /// `(flag, value)`: a value that does not read.
        case badValue(String, String)
    }

    package static let verbs: Set<String> = ["remove", "undo", "again", "reuse", "crop", "finish", "sheet", "import", "uninstall",
                                             "macs", "compute", "library", "start-image"]

    package static func usage(_ command: String) -> String {
        switch command {
        case "remove": "remove <i7> [i8 …]"
        case "undo": "undo [--redo]"
        case "again": "again <i7|file.png>"
        case "reuse": "reuse <i7|file.png>"
        case "crop": "crop <i7> --box X,Y,W,H | --ratio P:Q"
        case "finish": "finish <g2c3|t5>"
        case "sheet": "sheet <g2>"
        case "import": "import <file.safetensors|file.gguf> …"
        case "uninstall": "uninstall <model-id> [--yes] · uninstall lora <name> [--yes] · uninstall downloads [--yes]"
        case "macs": "macs [connect|disconnect NAME]"
        case "compute": "compute <NAME|this>"
        case "start-image": "start-image [<image|i7>] [--strength 0.65] · start-image --remove"
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
        let valued: Set<String> = command == "crop" ? ["--box", "--ratio"] : command == "start-image" ? ["--strength"] : []
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
        case "finish":
            guard let word = one() else { return .failure(.usage(line)) }
            return CellName(word).map { .success(.finish($0)) } ?? .failure(.badValue("cell", word))
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
