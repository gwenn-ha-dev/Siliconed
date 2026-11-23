// **The remote control's help: the complete reference** — its routing here, its pages by subject in `Help….swift`.
//
// `silicontrol help`, `silicontrol help <topic>`, `silicontrol help all` and `silicontrol <command> --help`
// are answered with one line, `{"ok": true, "help": "<text>"}` (the client prints the field raw). An
// AI told "run `silicontrol help`" must be able to do everything without another document: what the
// help says, the server does.
//
// A module of its own, Foundation only, shared by the two sides: `silicontrol` answers the help
// itself — reading a manual must not launch the app — and the app answers it on its socket (`nc -U`
// gets the same pages). One text, one routing (`answer`), so the two cannot diverge.
//
// Two versions of the same pages — English, and a faithful French one, picked by the user's language
// order among the two (`preferred`). The protocol itself (commands, flags, JSON keys, error codes) is
// English in both: only the prose is translated.

import Foundation

// This module is no part of the engine: only the app and `silicontrol` link it.
package enum RemoteControlHelp {
    /// One language's pages: the overview, and a page per topic — and the refusal of an unknown topic.
    struct Pages: Sendable {
        let noHelp: @Sendable (_ topic: String) -> String
        let topics: @Sendable (_ list: String) -> String
        let overview: String
        let pages: [String: String]
    }

    /// The topics in reading order (`help all` follows it).
    static let order = ["contract", "open", "models", "install", "import", "add", "grid", "edit", "wait", "status", "follow",
                        "cancel", "clear", "set", "compiled", "history", "save", "vary", "enlarge", "crop", "again", "remove", "undo", "sheet", "rewrite", "describe",
                        "instructions", "settings", "start-image", "lab", "bracket", "rack", "zoom", "snapshot", "subjects", "uninstall", "macs", "library", "measure", "errors", "recipes"]
    static let aliases = ["examples": "recipes", "reuse": "again", "compute": "macs"]

    /// French when the user puts it before English. Not `Bundle.main.preferredLocalizations`: the app's
    /// bundle has French phrases, `silicontrol` (a bare tool in `Contents/Helpers`) has none, and both
    /// must answer in the same language. `forPreferences: nil` is the user's order: without the
    /// argument, a bare tool gets `["en"]` whatever the user's language.
    static var preferred: Pages {
        Bundle.preferredLocalizations(from: ["en", "fr"], forPreferences: nil).first == "fr" ? french : english
    }

    /// **The answer to a help request**, or `nil` if `argv` is not one: `help [topic]`, `-h`, `--help`,
    /// `<command> --help` (or `-h`) before a lone `--` — after it, they are words, a prompt's (`add -- -h`) —, and nothing
    /// at all (the overview). A page is
    /// `{"ok": true, "help": …}`; an unknown topic, the usage failure (`"error": "usage"`, with the
    /// topics as `"hint"`).
    package static func answer(_ argv: [String]) -> [String: Any]? { answer(argv, preferred) }

    static func answer(_ argv: [String], _ pages: Pages) -> [String: Any]? {
        let subject: String?
        switch argv.first {
        case nil: subject = nil
        case "help", "-h", "--help": subject = argv.dropFirst().first
        case let command? where argv.dropFirst().prefix(while: { $0 != "--" }).contains(where: { $0 == "--help" || $0 == "-h" }):
            subject = command
        default: return nil
        }
        guard let text = text(for: subject, pages) else {
            return ["ok": false, "error": "usage", "message": pages.noHelp(subject ?? ""),
                    "hint": pages.topics(topicList)]
        }
        return ["ok": true, "help": text]
    }

    /// The text for `topic` (`nil`: the overview), or `nil` if there is no such topic.
    static func text(for topic: String?, _ pages: Pages = preferred) -> String? {
        guard let topic else { return pages.overview }
        let key = aliases[topic] ?? topic
        if key == "all" {
            return ([pages.overview] + order.compactMap { pages.pages[$0] })
                .joined(separator: "\n" + String(repeating: "─", count: 78) + "\n\n")
        }
        return pages.pages[key]
    }

    static var topicList: String { (order + ["all"]).joined(separator: ", ") }

    /// **One page in its two languages**: a page is written in both or in neither.
    struct Page: Sendable {
        let en: String
        let fr: String
    }

    /// Every topic's page, gathered from the five files that hold them by subject (`HelpContract.swift`, `HelpRender.swift`,
    /// `HelpLibrary.swift`, `HelpWriter.swift`, `HelpSettings.swift`); a topic in two of them stops the process at its first help.
    static let all: [String: Page] = [contractPages, renderPages, libraryPages, writerPages, settingsPages]
        .reduce(into: [:]) { all, group in all.merge(group) { _, _ in preconditionFailure("a help topic written twice") } }

    static let english = Pages(noHelp: { "no help for \"\($0)\"" }, topics: { "topics: \($0)" }, overview: overview.en, pages: all.mapValues(\.en))
    static let french = Pages(noHelp: { "pas d’aide pour « \($0) »" }, topics: { "sujets : \($0)" }, overview: overview.fr, pages: all.mapValues(\.fr))
}
