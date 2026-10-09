import Foundation

// **Alternatives in a prompt: `woman posing in a {library|greenhouse}` is two prompts.**
//
//     "a {red|blue} dress in a {library|greenhouse}"
//        │ parse: text · group · text · group
//        ▼
//     pieces  [.text("a "), .alternatives(["red", "blue"]), .text(" dress in a "), .alternatives(["library", "greenhouse"])]
//        │ expand: cartesian product, in reading order — the FIRST group varies the slowest
//        ▼
//     "a red dress in a library"        choices ["red", "library"]
//     "a red dress in a greenhouse"     choices ["red", "greenhouse"]
//     "a blue dress in a library"       choices ["blue", "library"]
//     "a blue dress in a greenhouse"    choices ["blue", "greenhouse"]
//
// The idea is ComfyUI's « Dynamic Prompts », with two deliberate differences an analogy would miss:
//
// - **Combinatorial, never random.** Dynamic Prompts draws one alternative per group by default;
//   here every combination is a render, at the same seed: the point is to compare, and a draw
//   compares nothing. The order is the reading order (an odometer whose leftmost wheel is the
//   slowest), so the n-th image of a series is the same combination from one run to the next.
// - **Nothing is trimmed.** `{|red }dress` gives `dress` and `red dress`: the spaces belong to
//   the alternative that carries them, and an empty alternative is the way to say « or nothing ».
//   `{a | b}` is therefore `a ` and ` b` — what was typed.
//
// The syntax, and what it refuses before any render (`EngineError.promptSyntax`):
// - `{` opens a group, `|` separates its alternatives, `}` closes it. Outside a group `|` is an
//   ordinary character: a prompt without braces or backslashes is the prompt, character for character.
// - `\{`, `\}`, `\|` are the literal characters, inside a group or out. Any other backslash stays
//   as typed (`\n` is two characters, `\\` too).
// - a `{` left open, a `}` that closes nothing, a `{` inside a group (nesting is not supported:
//   its meaning — a product inside an alternative — is a grid the user would not see coming).
// - more than `maxCombinations` prompts (`EngineError.tooManyPromptVariants`): the product grows
//   fast (six groups of two is 64 renders, ~35 min at 512² for Z-Image) and one typo should not
//   queue an afternoon.
//
// The positions an error names count `Character`s from 1, as a person counts the prompt they see.
//
// **What a grid will reuse**: `groups` (the alternatives of each group) and `variants` (the
// prompts, each with the label chosen in each group, `choices`), so that an XY axis is a group and
// its labels, without parsing the prompt a second time.

/// **A prompt with alternatives**, parsed: its pieces, its groups, and the prompts it stands for.
public struct PromptAlternatives: Equatable, Sendable {

    /// A run of literal text (escapes already resolved), or a group of alternatives.
    public enum Piece: Equatable, Sendable {
        case text(String)
        case alternatives([String])
    }

    /// One prompt of the expansion, and what was chosen in each group to make it, in reading order.
    public struct Variant: Equatable, Sendable {
        public let prompt: String
        /// The alternative taken in each group, as typed (escapes resolved, spaces kept; `""` for an
        /// empty alternative). Empty for a prompt without groups.
        public let choices: [String]
    }

    /// **The most prompts one expansion may make.** The app's queue holds that many renders.
    public static let maxCombinations = 64

    public let pieces: [Piece]

    /// Parses `prompt`, and refuses a syntax error or an expansion beyond `limit` prompts.
    public init(_ prompt: String, limit: Int = maxCombinations) throws(EngineError) {
        pieces = try Self.parse(prompt)
        let n = count
        guard n <= limit else { throw .tooManyPromptVariants(count: n, max: limit) }
    }

    /// The alternatives of each group, in reading order.
    public var groups: [[String]] {
        pieces.compactMap { if case .alternatives(let a) = $0 { return a } else { return nil } }
    }

    /// How many prompts the expansion makes: the product of the groups' sizes, 1 without a group.
    /// Saturates at `Int.max` rather than overflow: a count that large is refused anyway.
    public var count: Int {
        groups.reduce(1) { n, g in
            let (p, overflow) = n.multipliedReportingOverflow(by: g.count)
            return overflow ? .max : p
        }
    }

    /// The prompt has at least one group: rendering it is a series, not one render.
    public var hasAlternatives: Bool { !groups.isEmpty }

    /// **The prompts, in reading order** — the first group varies the slowest.
    public var variants: [Variant] {
        var result = [Variant(prompt: "", choices: [])]
        for piece in pieces {
            switch piece {
            case .text(let t):
                result = result.map { Variant(prompt: $0.prompt + t, choices: $0.choices) }
            case .alternatives(let options):
                // Each prompt so far, extended by every option: the earlier groups stay outermost.
                result = result.flatMap { v in options.map { Variant(prompt: v.prompt + $0, choices: v.choices + [$0]) } }
            }
        }
        return result
    }

    /// The expanded prompts alone.
    public var prompts: [String] { variants.map(\.prompt) }

    /// **`text` written to parse back to itself**: its braces escaped. What a form shows when it
    /// reopens an expanded prompt (a history image's), so that « Redo » renders that one prompt and
    /// not a series. `|` needs nothing: outside a group it is a character. The one prompt this
    /// cannot write is a backslash right before a brace — the syntax has no escape for `\` itself.
    public static func escaping(_ text: String) -> String {
        guard text.contains(where: { $0 == "{" || $0 == "}" }) else { return text }
        var s = ""
        for c in text {
            if c == "{" || c == "}" { s.append("\\") }
            s.append(c)
        }
        return s
    }

    // ── the parser ──

    private static func parse(_ prompt: String) throws(EngineError) -> [Piece] {
        var pieces: [Piece] = []
        var text = ""                 // the current literal run, or the current alternative
        var options: [String]?        // non-nil inside a group
        var opening = 0               // where the open group started, for the error
        var position = 0
        var iterator = Array(prompt).makeIterator()
        while let c = iterator.next() {
            position += 1
            switch c {
            case "\\":
                // Only the three syntax characters are escaped; any other backslash stays as typed.
                guard let next = iterator.next() else { text.append(c); continue }
                position += 1
                if next == "{" || next == "}" || next == "|" { text.append(next) } else { text.append(c); text.append(next) }
            case "{":
                guard options == nil else { throw .promptSyntax(position: position, reason: .nestedGroup) }
                if !text.isEmpty { pieces.append(.text(text)) }
                text = ""; options = []; opening = position
            case "|" where options != nil:
                options!.append(text); text = ""
            case "}":
                guard var o = options else { throw .promptSyntax(position: position, reason: .unmatchedClose) }
                o.append(text)
                pieces.append(.alternatives(o))
                text = ""; options = nil
            default:
                text.append(c)
            }
        }
        guard options == nil else { throw .promptSyntax(position: opening, reason: .unclosedGroup) }
        if !text.isEmpty { pieces.append(.text(text)) }
        return pieces
    }
}
