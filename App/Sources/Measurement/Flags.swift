// **What every command line of the app reads the same way**: its flags (`Flags.scan`), a size (`SizeSyntax`), bytes in
// hexadecimal, the app's bundle. Here, the lowest module, so that `silicontrol`, the app, the library and the checks share
// one reading — eight loops had drifted apart (`--ar 16:9 a cat` was a prompt to one command, a flag to another).

import Foundation

/// **A command's arguments, read once**: the flags it knows, with their values, and its words in order.
///
/// - A flag is a word of two dashes and more (`--seed`); one the command does not know is refused, by name, wherever it is.
/// - A valued flag takes the next argument as it is, whatever it looks like (`--prompt "--ar 16:9"`).
/// - A lone `--` ends the flags: what follows is words (`rest`), however they begin — a prompt that starts with `--` is
///   written after it.
package struct Flags: Equatable, Sendable {
    /// Each valued flag's values, in the order given (a flag given twice has two).
    package private(set) var values: [String: [String]] = [:]
    /// The switches given, in the order given.
    package private(set) var switches: [String] = []
    /// The words before a lone `--`, in order.
    package private(set) var words: [String] = []
    /// The arguments after a lone `--`, as given; empty without one.
    package private(set) var rest: [String] = []

    package enum Failure: Error, Equatable, Sendable {
        case unknownFlag(String)
        case flagWithoutValue(String)

        /// The English sentence, for a command without a catalog of its own.
        package var message: String {
            switch self {
            case .unknownFlag(let flag): "unknown flag: \(flag)"
            case .flagWithoutValue(let flag): "\(flag) expects a value"
            }
        }
    }

    /// A word that reads as a flag: two dashes and something after them.
    package static func isFlag(_ word: String) -> Bool { word.hasPrefix("--") && word.count > 2 }

    package static func scan(_ arguments: [String], valued: Set<String> = [], switches known: Set<String> = []) -> Result<Flags, Failure> {
        var flags = Flags()
        var i = 0
        while i < arguments.count {
            let a = arguments[i]
            if a == "--" {
                flags.rest = Array(arguments[(i + 1)...])
                break
            }
            if valued.contains(a) {
                guard i + 1 < arguments.count else { return .failure(.flagWithoutValue(a)) }
                flags.values[a, default: []].append(arguments[i + 1])
                i += 2
                continue
            }
            if known.contains(a) {
                flags.switches.append(a)
            } else if isFlag(a) {
                return .failure(.unknownFlag(a))
            } else {
                flags.words.append(a)
            }
            i += 1
        }
        return .success(flags)
    }

    /// The last value of `flag`, `nil` when absent.
    package func value(_ flag: String) -> String? { values[flag]?.last }

    package func has(_ flag: String) -> Bool { switches.contains(flag) || values[flag] != nil }

    /// Which of `flags` (switches that undo each other, `--preview` and `--no-preview`) was given last; `nil` for none.
    package func last(of flags: Set<String>) -> String? { switches.last { flags.contains($0) } }

    /// The words before the separator, then those after it: a command whose words are all of one kind.
    package var allWords: [String] { words + rest }
}

/// **A size as a command types it: `WxH`**, width × height, the order of screens (`832x1216` is a portrait). `×` counts as
/// `x`, spaces around it nothing (`832 × 1216`, as the app writes it); a lone integer is a square. The syntax only: what a
/// model renders is the app's to judge.
package enum SizeSyntax {
    package static func parse(_ text: String) -> (width: Int, height: Int)? {
        let pieces = text.lowercased().replacingOccurrences(of: "×", with: "x")
            .split(separator: "x", omittingEmptySubsequences: false).map { $0.trimmingCharacters(in: .whitespaces) }
        switch pieces.count {
        case 1: return Int(pieces[0]).map { ($0, $0) }
        case 2:
            guard let w = Int(pieces[0]), let h = Int(pieces[1]) else { return nil }
            return (w, h)
        default: return nil
        }
    }
}

extension Sequence where Element == UInt8 {
    /// Two lowercase hexadecimal digits a byte: a digest as files and the wire write it.
    package var hexadecimal: String {
        let digits = Array("0123456789abcdef".utf8)
        var out: [UInt8] = []
        for byte in self { out.append(digits[Int(byte >> 4)]); out.append(digits[Int(byte & 0x0F)]) }
        return String(decoding: out, as: UTF8.self)
    }
}

/// The app's bundle, as macOS knows it: what `open -b` launches, whose preferences and caches are the app's.
package let siliconedBundleIdentifier = "dev.gwenn-ha.siliconed"
