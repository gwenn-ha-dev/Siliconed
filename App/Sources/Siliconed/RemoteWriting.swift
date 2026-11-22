// What `silicontrol` asks of the app that is a decision, not a gesture — kept here, pure, so that the tests read it
// and the window and the command cannot disagree: the arguments of `rewrite`, `describe`, `instructions` and `set`, the Beta options
// by name, the rule that makes the Prompt Lab's writer wait, and the model the lab writes for by default.
//
// The gestures themselves are the app's (`LabModel`, `AppState`): a command calls the function the button calls.

import Measurement

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

    /// The `error` of the remote control's refusal.
    package var code: String {
        switch self {
        case .following: "remote"
        case .rendering: "busy"
        case .notInstalled: "writer_absent"
        }
    }

    /// The reasons in order: the other Mac first (nothing here is the writer's to ask), then the render, then the disk.
    /// `installed`: some installed model carries a writer; `hasTarget`: some installed model to write for.
    package static func decide(following: Bool, rendering: Bool, installed: Bool, hasTarget: Bool) -> WriterPause? {
        if following { return .following }
        if rendering { return .rendering }
        return installed && hasTarget ? nil : .notInstalled
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

        init(_ failure: Flags.Failure) {
            switch failure {
            case .unknownFlag(let flag): self = .unknownFlag(flag)
            case .flagWithoutValue(let flag): self = .flagWithoutValue(flag)
            }
        }
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

    /// A text that begins with `--` (`--ar 16:9 a cat`) is written after a lone `--` (`Flags`).
    package static func parse(_ gesture: Gesture, _ arguments: [String]) -> Result<WriterOrder, Failure> {
        let flags: Flags
        switch Flags.scan(arguments, valued: ["--for"]) {
        case .success(let f): flags = f
        case .failure(let failure): return .failure(Failure(failure))
        }
        guard (flags.values["--for"]?.count ?? 0) <= 1, flags.allWords.count == 1, let subject = flags.allWords.first else {
            return .failure(.usage(usage(gesture)))
        }
        let target = flags.value("--for")
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
        let flags: Flags
        switch Flags.scan(arguments, valued: ["--for", "--set", "--file"], switches: ["--restore"]) {
        case .success(let f): flags = f
        case .failure(let failure): return .failure(WriterOrder.Failure(failure))
        }
        // One target, one change at most (`--set`, `--file` or `--restore`), one gesture at most.
        let changes = (flags.values["--set"] ?? []).map(Change.set) + (flags.values["--file"] ?? []).map(Change.file)
            + flags.switches.map { _ in Change.restore }
        let words = flags.allWords
        guard (flags.values["--for"]?.count ?? 0) <= 1, changes.count <= 1, words.count <= 1 else { return .failure(.usage(usage)) }
        var order = InstructionOrder(target: flags.value("--for"), change: changes.first)
        if let word = words.first {
            guard let gesture = PromptWriterInstructions.Gesture(rawValue: word) else { return .failure(.usage(usage)) }
            order.gesture = gesture
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

/// **`set …`**, read, every subcommand: the Beta options, Save Every Image To… (`save-images`: a folder, or `off`; a folder
/// named `off` is `./off`), the memory limit, and this Mac's switches. Alone, a subcommand reads what it sets.
package enum SetOrder: Equatable, Sendable {
    /// `set beta OPTION on|off`.
    case beta(BetaOption, on: Bool)
    /// `set beta`: reads them all.
    case betaList
    /// `set save-images FOLDER|off`: the folder as typed (relative to the caller's directory), `nil` for off.
    case saveImages(folder: String?)
    /// `set save-images`: reads it.
    case saveImagesState
    /// `set memory [GB|default]`: B as typed, in GiB, or the default; `nil` reads it. This Mac's range is the app's to judge.
    case memory(Memory?)
    /// `set neural-engine|glass|updates|preview [on|off]`: one of this Mac's switches; `nil` reads it.
    case toggle(Toggle, on: Bool?)

    package enum Memory: Equatable, Sendable {
        case standard
        case gigabytes(Double)
    }

    /// The switches `set` turns on and off, by the name it gives them.
    package enum Toggle: String, CaseIterable, Sendable {
        case neuralEngine = "neural-engine", glass, updates, preview
    }

    package enum Failure: Error, Equatable, Sendable {
        /// The associated value is the usage line of the subcommand, or all of `set`'s.
        case usage(String)
        case unknownBeta(String)
        /// `set` takes no flag: a word of two dashes is refused, by name — a folder whose name begins with `--` goes after a
        /// lone `--` (`set save-images -- --renders`).
        case unknownFlag(String)
    }

    /// Each subcommand's usage, in the order `help set` gives them.
    package static let usages: [(name: String, usage: String)] = ["beta", "neural-engine", "save-images", "glass", "updates", "memory", "preview"].map { name in
        let argument = switch name {
        case "beta": "[OPTION on|off]"
        case "save-images": "[FOLDER|off]"
        case "memory": "[GB|default]"
        default: "[on|off]"
        }
        return (name, "set \(name) \(argument)")
    }

    /// All of `set`.
    package static let usage = usages.map(\.usage).joined(separator: " · ")

    package static func usage(of what: String) -> String { usages.first { $0.name == what }?.usage ?? usage }

    /// `on` or `off`, the word every switch takes; `nil` for any other.
    package static func onOff(_ word: String) -> Bool? {
        switch word {
        case "on": true
        case "off": false
        default: nil
        }
    }

    package static func parse(_ arguments: [String]) -> Result<SetOrder, Failure> {
        let words: [String]
        switch Flags.scan(arguments) {
        case .success(let f): words = f.allWords
        case .failure(.unknownFlag(let flag)), .failure(.flagWithoutValue(let flag)): return .failure(.unknownFlag(flag))
        }
        guard let what = words.first else { return .failure(.usage(usage)) }
        let rest = Array(words.dropFirst())
        let refused = Result<SetOrder, Failure>.failure(.usage(usage(of: what)))
        switch what {
        case "beta":
            if rest.isEmpty { return .success(.betaList) }
            guard rest.count == 2, let on = onOff(rest[1]) else { return refused }
            guard let option = BetaOption(argument: rest[0]) else { return .failure(.unknownBeta(rest[0])) }
            return .success(.beta(option, on: on))
        case "save-images":
            switch rest.count {
            case 0: return .success(.saveImagesState)
            case 1: return .success(.saveImages(folder: rest[0] == "off" ? nil : rest[0]))
            default: return refused
            }
        case "memory":
            switch rest.count {
            case 0: return .success(.memory(nil))
            case 1 where rest[0] == "default": return .success(.memory(.standard))
            case 1:
                guard let gb = Double(rest[0]), gb.isFinite, gb > 0 else { return refused }
                return .success(.memory(.gigabytes(gb)))
            default: return refused
            }
        default:
            guard let toggle = Toggle(rawValue: what) else { return .failure(.usage(usage)) }
            switch rest.count {
            case 0: return .success(.toggle(toggle, on: nil))
            case 1:
                guard let on = onOff(rest[0]) else { return refused }
                return .success(.toggle(toggle, on: on))
            default: return refused
            }
        }
    }
}
