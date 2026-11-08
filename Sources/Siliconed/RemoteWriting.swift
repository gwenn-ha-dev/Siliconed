// What `silicontrol` asks of the app that is a decision, not a gesture — kept here, pure, so that the tests read it
// and the window and the command cannot disagree: the arguments of `rewrite`, `describe` and `set`, the Beta options
// by name, the rule that makes the Prompt Lab's writer wait, and the model the lab writes for by default.
//
// The gestures themselves are the app's (`LabModel`, `AppState`): a command calls the function the button calls.

// MARK: - The writer waits

/// **Why Rewrite and Describe wait**, from facts: the Prompt Lab says it in a sentence under its buttons, `silicontrol` in a
/// stable code. One rule for both (`decide`), so the order of the reasons is the same in the window and in the terminal.
package enum WriterPause: Equatable, Sendable {
    /// The window computes on another Mac: the writer is this Mac's, and is off.
    case following
    /// An image renders or waits: the render always has priority.
    case rendering
    /// No installed model carries a writer.
    case notInstalled
    /// One is installed and does not fit this Mac: `gigabytes` of memory would.
    case needsMemory(gigabytes: Int)
    /// The encoder is installed, the writer's head is not: `bytes` is what installing it downloads.
    case needsHead(bytes: Int64)
    /// Being downloaded or converted.
    case preparing

    /// The `error` of the remote control's refusal.
    package var code: String {
        switch self {
        case .following: "remote"
        case .rendering: "busy"
        case .notInstalled: "writer_absent"
        case .needsMemory: "writer_memory"
        case .needsHead: "writer_head"
        case .preparing: "writer_preparing"
        }
    }

    /// The reasons in order: the other Mac first (nothing here is the writer's to ask), then the render, then the disk.
    /// `installed`: what the writer's own state says (`nil`: ready); `hasTarget`: some installed model to write for.
    package static func decide(following: Bool, rendering: Bool, installed: WriterPause?, hasTarget: Bool) -> WriterPause? {
        if following { return .following }
        if rendering { return .rendering }
        if let installed { return installed }
        return hasTarget ? nil : .notInstalled
    }
}

// MARK: - The model the lab writes for

package enum WriterTarget {
    /// **The family a writer writes for** when nobody named one: the one chosen by hand if it is still installed, else the
    /// rack's, else the first installed (the catalog's order). The lab's rule, which the command follows.
    package static func pick(chosen: Family?, rack: Family?, installed: [Family]) -> Family? {
        if let chosen, installed.contains(chosen) { return chosen }
        if let rack, installed.contains(rack) { return rack }
        return installed.first
    }

    /// What `--for` names: a model's id or a family's (they are the same words for the models in front), among the families
    /// installed. `nil` when it names none of them.
    package static func named(_ word: String, installed: [Family], modelFamily: (String) -> Family?) -> Family? {
        let family = Family(rawValue: word) ?? modelFamily(word)
        return family.flatMap { installed.contains($0) ? $0 : nil }
    }
}

// MARK: - The arguments of rewrite and describe

/// **`rewrite "<text>" [--for ID]` and `describe <image> [--for ID]`**, read. The subject is the text (one argument, in quotes,
/// like `add`'s prompt) or the image (a history name `i7`, or a path); what it names is the app's to resolve.
package struct WriterOrder: Equatable, Sendable {
    package enum Gesture: String, Sendable { case rewrite, describe }

    package enum Failure: Error, Equatable, Sendable {
        /// Not the syntax; the associated value is the usage line.
        case usage(String)
        case unknownFlag(String)
        case flagWithoutValue(String)
        case emptyText
    }

    package var gesture: Gesture
    package var subject: String
    /// `--for`: a model id or a family, as typed.
    package var target: String?

    package static func usage(_ gesture: Gesture) -> String {
        switch gesture {
        case .rewrite: "rewrite \"<text>\" [--for ID]"
        case .describe: "describe <image|i7> [--for ID]"
        }
    }

    package static func parse(_ gesture: Gesture, _ arguments: [String]) -> Result<WriterOrder, Failure> {
        var subject: String?, target: String?
        var i = 0
        while i < arguments.count {
            let a = arguments[i]
            if a == "--for" {
                guard i + 1 < arguments.count else { return .failure(.flagWithoutValue(a)) }
                guard target == nil else { return .failure(.usage(usage(gesture))) }
                target = arguments[i + 1]
                i += 2
                continue
            }
            // A word shaped like a flag is one; a text with a space in it ("--ar 16:9 a cat") is a text.
            if a.hasPrefix("--"), a.count > 2, !a.contains(" ") { return .failure(.unknownFlag(a)) }
            guard subject == nil else { return .failure(.usage(usage(gesture))) }
            subject = a
            i += 1
        }
        guard let subject else { return .failure(.usage(usage(gesture))) }
        if gesture == .rewrite, subject.allSatisfy(\.isWhitespace) { return .failure(.emptyText) }
        return .success(WriterOrder(gesture: gesture, subject: subject, target: target))
    }
}

// MARK: - The arguments of instructions

/// **`instructions [rewrite|describe] [--for ID] [--set "<text>" | --file PATH | --restore]`**, read: Settings › Prompt Lab
/// from the terminal. Without a gesture, the list (of one family with `--for`); with one, that instruction, shown or
/// changed — a change names its gesture, as the pane's field is one gesture's. `--for` names any family, installed or not:
/// the instructions are the app's, not the library's.
package struct InstructionOrder: Equatable, Sendable {
    package enum Change: Equatable, Sendable {
        case set(String)
        /// A path as typed: the app reads it against the command's directory.
        case file(String)
        case restore
    }

    package var gesture: PromptWriterInstructions.Gesture?
    package var target: String?
    package var change: Change?

    package static let usage = "instructions [rewrite|describe] [--for ID] [--set \"<text>\" | --file PATH | --restore]"

    package static func parse(_ arguments: [String]) -> Result<InstructionOrder, WriterOrder.Failure> {
        var order = InstructionOrder()
        var i = 0
        while i < arguments.count {
            let a = arguments[i]
            switch a {
            case "--for", "--set", "--file":
                guard i + 1 < arguments.count else { return .failure(.flagWithoutValue(a)) }
                let value = arguments[i + 1]
                if a == "--for" {
                    guard order.target == nil else { return .failure(.usage(usage)) }
                    order.target = value
                } else {
                    guard order.change == nil else { return .failure(.usage(usage)) }
                    order.change = a == "--set" ? .set(value) : .file(value)
                }
                i += 2
            case "--restore":
                guard order.change == nil else { return .failure(.usage(usage)) }
                order.change = .restore
                i += 1
            default:
                if a.hasPrefix("--") { return .failure(.unknownFlag(a)) }
                guard order.gesture == nil, let gesture = PromptWriterInstructions.Gesture(rawValue: a) else { return .failure(.usage(usage)) }
                order.gesture = gesture
                i += 1
            }
        }
        if order.change != nil, order.gesture == nil { return .failure(.usage(usage)) }
        return .success(order)
    }
}

// MARK: - The Beta options

/// **The four switches of Settings › Advanced › Beta**, by the name `silicontrol set beta` gives them.
package enum BetaOption: String, CaseIterable, Sendable {
    case moreSolvers = "more-solvers"
    case steps = "steps"
    case fasterRendering = "faster-rendering"
    case sizeScales = "size-scales"
    case negativeLoRA = "negative-lora"

    /// The name as typed: the canonical one, or the setting's own label slugged (`size-in-two-scales`).
    package init?(argument: String) {
        let word = argument.lowercased()
        switch word {
        case "size-in-two-scales": self = .sizeScales
        case "negative-lora-strengths": self = .negativeLoRA
        case "choose-steps": self = .steps
        default: self.init(rawValue: word)
        }
    }

    /// The key of the option in the remote control's JSON (`more_solvers`).
    package var key: String { rawValue.replacingOccurrences(of: "-", with: "_") }

    package static var names: String { allCases.map(\.rawValue).joined(separator: ", ") }
}

// MARK: - The arguments of set

/// **`set …`**, read: a model's switch (`faster-rendering`, `accelerators` — `neural-engine`, its name before it covered the GPU's
/// tensor units, still read), the Beta options, or Save Every Image To… (`save-images`: a folder, or `off`; a folder named
/// `off` is `./off`).
package enum SetOrder: Equatable, Sendable {
    package enum Switch: String, Sendable {
        case fasterRendering = "faster-rendering", accelerators = "accelerators"
        /// The switch's names, the former one included.
        init?(argument: String) {
            if argument == "neural-engine" { self = .accelerators } else { self.init(rawValue: argument) }
        }
    }

    case perModel(Switch, on: Bool, model: String?)
    /// `set beta OPTION on|off`.
    case beta(BetaOption, on: Bool)
    /// `set beta`: reads them all.
    case betaList
    /// `set save-images FOLDER|off`: the folder as typed (relative to the caller's directory), `nil` for off.
    case saveImages(folder: String?)
    /// `set save-images`: reads it.
    case saveImagesState

    package enum Failure: Error, Equatable, Sendable {
        case usage(String)
        case unknownBeta(String)
    }

    package static let usage = "set faster-rendering|accelerators on|off [--model ID] · set beta [OPTION on|off] · set save-images [FOLDER|off]"

    package static func parse(_ arguments: [String]) -> Result<SetOrder, Failure> {
        guard let what = arguments.first else { return .failure(.usage(usage)) }
        if what == "beta" {
            if arguments.count == 1 { return .success(.betaList) }
            guard arguments.count == 3, ["on", "off"].contains(arguments[2]) else {
                return .failure(.usage("set beta [OPTION on|off]"))
            }
            guard let option = BetaOption(argument: arguments[1]) else { return .failure(.unknownBeta(arguments[1])) }
            return .success(.beta(option, on: arguments[2] == "on"))
        }
        if what == "save-images" {
            switch arguments.count {
            case 1: return .success(.saveImagesState)
            case 2: return .success(.saveImages(folder: arguments[1] == "off" ? nil : arguments[1]))
            default: return .failure(.usage("set save-images [FOLDER|off]"))
            }
        }
        guard let kind = Switch(argument: what) else { return .failure(.usage(usage)) }
        let line = "set \(what) on|off [--model ID]"
        guard arguments.count == 2 || (arguments.count == 4 && arguments[2] == "--model"),
              ["on", "off"].contains(arguments[1]) else { return .failure(.usage(line)) }
        return .success(.perModel(kind, on: arguments[1] == "on", model: arguments.count == 4 ? arguments[3] : nil))
    }
}
