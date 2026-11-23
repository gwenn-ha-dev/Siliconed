// A gesture's command line read into its `GestureOrder`: the verbs, their usage lines, the `--yes` guard.

import Measurement

extension GestureOrder {
    package static let verbs: Set<String> = ["remove", "undo", "again", "reuse", "crop", "sheet", "uninstall",
                                             "macs", "compute", "library", "start-image", "lab", "settings", "bracket", "rack", "zoom", "snapshot"]

    package static func usage(_ command: String) -> String {
        switch command {
        case "remove": "remove <i7> [i8 …] · remove <g2>"
        case "undo": "undo [--redo]"
        case "again": "again <i7|file.png>"
        case "reuse": "reuse <i7|g2|file.png>"
        case "crop": "crop <i7> --box X,Y,W,H | --ratio P:Q"
        case "sheet": "sheet <g2> <file.png|folder>"
        case "uninstall": "uninstall <model-id> [--variant standard|compact] [--yes] · uninstall downloads|legacy [--yes]"
        case "macs": "macs [connect|disconnect NAME]"
        case "compute": "compute <NAME|this>"
        case "start-image": "start-image [<image|i7>] [--strength 0.65] · start-image --remove"
        case "lab": "lab text <text> · lab generate · lab add <image|i7> … · lab role <n> edited|reference|start|none · lab remove <n> · lab rewrite [--wait] · lab describe <n> [--wait]"
        case "settings": "settings [general|macs|lab|advanced] · settings lab [rewrite|describe] [--for ID]"
        case "bracket": "bracket seed|detail|solver|start|loras · bracket lora <name> · bracket close"
        case "rack": "rack top|dit|fold|unfold|detach|dock"
        case "zoom": "zoom in|out|fit"
        case "snapshot": "snapshot <file.png> [--window WORD|N] [--appearance light|dark]"
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
        let valued: Set<String> = switch command {
        case "crop": ["--box", "--ratio"]
        case "start-image": ["--strength"]
        case "uninstall": ["--variant"]
        case "settings": ["--for"]
        case "snapshot": ["--window", "--appearance"]
        default: []
        }
        var switches: Set<String> = switch command {
        case "undo": ["--redo"]
        case "start-image": ["--remove"]
        case "lab": ["--wait"]
        default: []
        }
        if takesYes(command, arguments) { switches.insert("--yes") }
        let flags: Flags
        switch Flags.scan(arguments, valued: valued, switches: switches) {
        case .success(let f): flags = f
        case .failure(.unknownFlag(let a)): return .failure(.unknownFlag(a))
        case .failure(.flagWithoutValue(let a)): return .failure(.flagWithoutValue(a))
        }
        // A valued flag is given once.
        guard flags.values.values.allSatisfy({ $0.count == 1 }) else { return .failure(.usage(line)) }
        let words = flags.allWords, yes = flags.has("--yes")
        var values = flags.values.compactMapValues(\.last)
        for s in flags.switches where s != "--yes" { values[s] = "" }
        func one() -> String? { words.count == 1 ? words[0] : nil }
        /// `g2`: a series of the session, as the answers name it.
        func grid(_ word: String) -> Int? { word.first == "g" ? Int(word.dropFirst()).flatMap { $0 >= 0 ? $0 : nil } : nil }

        switch command {
        case "remove":
            // A series is removed alone: its tile's gesture, not a selection of images.
            if let n = words.lazy.compactMap(grid).first { return words.count == 1 ? .success(.removeSeries(grid: n)) : .failure(.usage(line)) }
            return words.isEmpty ? .failure(.usage(line)) : .success(.remove(words))
        case "undo":
            return words.isEmpty ? .success(.undo(redo: values["--redo"] != nil)) : .failure(.usage(line))
        case "again": return one().map { .success(.again($0)) } ?? .failure(.usage(line))
        case "reuse":
            guard let word = one() else { return .failure(.usage(line)) }
            return .success(grid(word).map { .reuseSeries(grid: $0) } ?? .reuse(word))
        case "crop":
            guard let image = one(), values.count == 1 else { return .failure(.usage(line)) }
            if let v = values["--box"] {
                let n = v.split(separator: ",", omittingEmptySubsequences: false).map { Int($0.trimmingCharacters(in: .whitespaces)) }
                guard n.count == 4, !n.contains(where: { $0 == nil }) else { return .failure(.badValue("--box", v)) }
                return .success(.crop(image: image, .box(Crop.Box(x: n[0]!, y: n[1]!, width: n[2]!, height: n[3]!))))
            }
            let v = values["--ratio"] ?? ""
            let sides = v.split(separator: ":", omittingEmptySubsequences: false).map { Int($0) }
            // Bounded as a side is: the crop's products of a side and a ratio trap on overflow.
            guard sides.count == 2, let w = sides[0], let h = sides[1], (1...Format.maximumSide).contains(w), (1...Format.maximumSide).contains(h)
            else { return .failure(.badValue("--ratio", v)) }
            return .success(.crop(image: image, .ratio(Crop.Ratio(w, h))))
        case "sheet":
            guard words.count == 2 else { return .failure(.usage(line)) }
            guard let n = grid(words[0]) else { return .failure(.badValue("grid", words[0])) }
            return .success(.sheet(grid: n, to: words[1]))
        case "uninstall":
            let version = values["--variant"]
            if words == ["downloads"], version == nil { return .success(.uninstall(.downloads, confirmed: yes)) }
            if words == ["legacy"], version == nil { return .success(.uninstall(.legacy, confirmed: yes)) }
            if words.count == 1, !["downloads", "legacy"].contains(words[0]) { return .success(.uninstall(.model(words[0], version: version), confirmed: yes)) }
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
            case "text"?: return rest.count == 1 ? .success(.lab(.text(rest[0]))) : .failure(.usage(line))
            case "generate"?: return rest.isEmpty ? .success(.lab(.generate)) : .failure(.usage(line))
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
        case "zoom":
            guard words.count == 1 else { return .failure(.usage(line)) }
            return ZoomStep(rawValue: words[0].lowercased()).map { .success(.zoom($0)) } ?? .failure(.badValue("step", words[0]))
        case "snapshot":
            guard let file = one() else { return .failure(.usage(line)) }
            var appearance: SnapshotAppearance?
            if let word = values["--appearance"] {
                guard let a = SnapshotAppearance(rawValue: word.lowercased()) else { return .failure(.badValue("--appearance", word)) }
                appearance = a
            }
            return .success(.snapshot(file: file, window: values["--window"], appearance: appearance))
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
