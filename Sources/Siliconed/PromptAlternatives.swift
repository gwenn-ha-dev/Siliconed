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
package struct PromptAlternatives: Equatable, Sendable {

    /// A run of literal text (escapes already resolved), or a group of alternatives.
    package enum Piece: Equatable, Sendable {
        case text(String)
        case alternatives([String])
    }

    /// One prompt of the expansion, and what was chosen in each group to make it, in reading order.
    package struct Variant: Equatable, Sendable {
        package let prompt: String
        /// The alternative taken in each group, as typed (escapes resolved, spaces kept; `""` for an
        /// empty alternative). Empty for a prompt without groups.
        package let choices: [String]
    }

    /// **The most prompts one expansion may make.** The app's queue holds that many renders.
    package static let maxCombinations = 64

    package let pieces: [Piece]

    /// Parses `prompt`, and refuses a syntax error or an expansion beyond `limit` prompts.
    package init(_ prompt: String, limit: Int = maxCombinations) throws(EngineError) {
        pieces = try Self.parse(prompt)
        let n = count
        guard n <= limit else { throw .tooManyPromptVariants(count: n, max: limit) }
    }

    /// The alternatives of each group, in reading order.
    package var groups: [[String]] {
        pieces.compactMap { if case .alternatives(let a) = $0 { return a } else { return nil } }
    }

    /// How many prompts the expansion makes: the product of the groups' sizes, 1 without a group.
    /// Saturates at `Int.max` rather than overflow: a count that large is refused anyway.
    package var count: Int {
        groups.reduce(1) { n, g in
            let (p, overflow) = n.multipliedReportingOverflow(by: g.count)
            return overflow ? .max : p
        }
    }

    /// The prompt has at least one group: rendering it is a series, not one render.
    package var hasAlternatives: Bool { !groups.isEmpty }

    /// **The prompts, in reading order** — the first group varies the slowest.
    package var variants: [Variant] {
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
    package var prompts: [String] { variants.map(\.prompt) }

    /// **`text` written to parse back to itself**: its braces escaped. What a form shows when it
    /// reopens an expanded prompt (a history image's), so that « Redo » renders that one prompt and
    /// not a series. `|` needs nothing: outside a group it is a character. The one prompt this
    /// cannot write is a backslash right before a brace — the syntax has no escape for `\` itself.
    package static func escaping(_ text: String) -> String {
        guard text.contains(where: { $0 == "{" || $0 == "}" }) else { return text }
        var s = ""
        for c in text {
            if c == "{" || c == "}" { s.append("\\") }
            s.append(c)
        }
        return s
    }

    // ── the groups in the text as typed ──

    /// **Where a group is in the text as typed**: from its `{` to its `}` included, and each alternative's characters.
    package struct Span: Equatable, Sendable {
        package let range: Range<String.Index>
        package let alternatives: [Range<String.Index>]
    }

    /// **The groups of `prompt` where they are typed**, in reading order — what a field draws as a chip and a gesture
    /// rewrites. `[]` when the text does not parse: a chip is never drawn over a syntax error.
    package static func spans(_ prompt: String) -> [Span] {
        guard (try? PromptAlternatives(prompt, limit: .max)) != nil else { return [] }
        var spans: [Span] = []
        var open: String.Index?
        var start = prompt.startIndex
        var alternatives: [Range<String.Index>] = []
        var i = prompt.startIndex
        while i < prompt.endIndex {
            let c = prompt[i]
            let next = prompt.index(after: i)
            switch c {
            case "\\":
                // As the parser: the character after a backslash is consumed with it, whatever it is.
                i = next < prompt.endIndex ? prompt.index(after: next) : next
                continue
            case "{":
                open = i; start = next; alternatives = []
            case "|" where open != nil:
                alternatives.append(start..<i); start = next
            case "}":
                if let o = open {
                    alternatives.append(start..<i)
                    spans.append(Span(range: o..<next, alternatives: alternatives))
                    open = nil
                }
            default:
                break
            }
            i = next
        }
        return spans
    }

    /// **An alternative written to read back as itself inside a group**: its `{`, `}` and `|` escaped.
    package static func escapingAlternative(_ text: String) -> String {
        var s = ""
        for c in text {
            if c == "{" || c == "}" || c == "|" { s.append("\\") }
            s.append(c)
        }
        return s
    }

    /// **A group as typed**: `{red|blue|green}`, each alternative escaped — what « Vary This Word… » writes.
    package static func group(_ alternatives: [String]) -> String {
        "{" + alternatives.map(escapingAlternative).joined(separator: "|") + "}"
    }

    /// **The `g`-th group of `prompt` replaced by `text`**, written as it should be typed (another group, or a word
    /// `escaping` made safe). The prompt unchanged when it has no such group.
    package static func replacing(group g: Int, in prompt: String, with text: String) -> String {
        let spans = spans(prompt)
        guard spans.indices.contains(g) else { return prompt }
        var s = prompt
        s.replaceSubrange(spans[g].range, with: text)
        return s
    }

    /// **The `g`-th group brought back to its first alternative** — the text a series no longer varies there.
    package static func collapsing(group g: Int, in prompt: String) -> String {
        guard let p = try? PromptAlternatives(prompt, limit: .max), p.groups.indices.contains(g) else { return prompt }
        return replacing(group: g, in: prompt, with: escaping(p.groups[g].first ?? ""))
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
