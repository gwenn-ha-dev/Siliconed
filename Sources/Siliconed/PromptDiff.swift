// **What a rewrite changed, word by word** — the Prompt Lab shows the writer's text against the
// user's, additions and removals marked, spaces and punctuation as typed.
//
//     old   a red car, parked      new   a blue car parked in the rain
//     same "a"  removed " red"  added " blue"  same " car"  removed ","  same " parked"  added " in the rain"
//
// What a plain split on spaces would miss:
//
// - **The unit is a word, a punctuation mark, a CJK character or one emoji** (a grapheme cluster, so a
//   ZWJ sequence or an accented letter in either normalization form stays whole). Punctuation is its
//   own unit: « rouge, » → « rouge » changes the comma, not the word. An apostrophe between two letters
//   belongs to the word (« l’image »).
// - **Chinese and Japanese have no spaces**: each Han, kana or full-width character is a unit, so a
//   changed word shows as the characters that differ.
// - **Spaces are not compared.** Each unit carries the white space that preceded it, so the pieces of
//   `same` and `added`, joined, are exactly the new text; a removed piece keeps the old text's spacing.

import Foundation

package enum PromptDiff {
    package enum Kind: Sendable, Equatable { case same, added, removed }

    package struct Piece: Sendable, Equatable {
        package let kind: Kind
        package let text: String
        package init(_ kind: Kind, _ text: String) { self.kind = kind; self.text = text }
    }

    /// The pieces turning `old` into `new`, neighbours of one kind merged; in a changed stretch the
    /// removal comes before the addition. White space at the very end belongs to the new text.
    package static func pieces(from old: String, to new: String) -> [Piece] {
        if old == new { return new.isEmpty ? [] : [Piece(.same, new)] }
        let a = units(old), b = units(new)
        var out: [Piece] = []
        func emit(_ kind: Kind, _ text: String) {
            guard !text.isEmpty else { return }
            if let last = out.last, last.kind == kind { out[out.count - 1] = Piece(kind, last.text + text) }
            else { out.append(Piece(kind, text)) }
        }
        // The common head and tail first: a rewrite often keeps its opening and its end, and the
        // table below is then only as big as the middle.
        var head = 0
        while head < a.units.count, head < b.units.count, a.units[head].word == b.units[head].word { head += 1 }
        var tail = 0
        while tail < a.units.count - head, tail < b.units.count - head,
              a.units[a.units.count - 1 - tail].word == b.units[b.units.count - 1 - tail].word { tail += 1 }
        for u in b.units[..<head] { emit(.same, u.lead + u.word) }

        let x = Array(a.units[head..<(a.units.count - tail)]), y = Array(b.units[head..<(b.units.count - tail)])
        for step in align(x.map(\.word), y.map(\.word)) {
            switch step {
            case .same(_, let j): emit(.same, y[j].lead + y[j].word)
            case .removed(let i): emit(.removed, x[i].lead + x[i].word)
            case .added(let j): emit(.added, y[j].lead + y[j].word)
            }
        }
        for u in b.units[(b.units.count - tail)...] { emit(.same, u.lead + u.word) }
        emit(.same, b.trailing)
        return out
    }

    // MARK: Units

    private struct Unit { let lead: String; let word: String }

    private static func units(_ text: String) -> (units: [Unit], trailing: String) {
        var units: [Unit] = [], lead = ""
        let chars = Array(text)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c.isWhitespace { lead.append(c); i += 1; continue }
            var word = String(c)
            i += 1
            if isWordCharacter(c) {
                while i < chars.count {
                    let n = chars[i]
                    if isWordCharacter(n) { word.append(n); i += 1 }
                    else if n == "'" || n == "\u{2019}", i + 1 < chars.count, isWordCharacter(chars[i + 1]) {
                        word.append(n); i += 1
                    } else { break }
                }
            }
            units.append(Unit(lead: lead, word: word))
            lead = ""
        }
        return (units, lead)
    }

    private static func isWordCharacter(_ c: Character) -> Bool {
        (c.isLetter || c.isNumber) && !isIdeographicOrKana(c)
    }

    private static func isIdeographicOrKana(_ c: Character) -> Bool {
        guard let v = c.unicodeScalars.first?.value else { return false }
        switch v {
        case 0x3040...0x30FF, 0x31F0...0x31FF, 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF,
             0xFF66...0xFF9F, 0x20000...0x2FA1F, 0x30000...0x3134F: return true
        default: return false
        }
    }

    // MARK: Alignment

    private enum Step { case same(Int, Int), removed(Int), added(Int) }

    /// The longest common subsequence. Beyond 4 M cells (a prompt is a few hundred units) the middle
    /// is given as one removal and one addition, which is true if coarse.
    private static func align(_ x: [String], _ y: [String]) -> [Step] {
        let n = x.count, m = y.count
        if n == 0 { return (0..<m).map { .added($0) } }
        if m == 0 { return (0..<n).map { .removed($0) } }
        if n * m > 4_000_000 { return (0..<n).map { .removed($0) } + (0..<m).map { .added($0) } }
        // table[i][j]: the common length of x[i...] and y[j...].
        var table = [Int32](repeating: 0, count: (n + 1) * (m + 1))
        let w = m + 1
        for i in stride(from: n - 1, through: 0, by: -1) {
            for j in stride(from: m - 1, through: 0, by: -1) {
                table[i * w + j] = x[i] == y[j] ? table[(i + 1) * w + j + 1] + 1
                                                 : max(table[(i + 1) * w + j], table[i * w + j + 1])
            }
        }
        var steps: [Step] = [], i = 0, j = 0
        while i < n, j < m {
            if x[i] == y[j] { steps.append(.same(i, j)); i += 1; j += 1 }
            else if table[(i + 1) * w + j] >= table[i * w + j + 1] { steps.append(.removed(i)); i += 1 }
            else { steps.append(.added(j)); j += 1 }
        }
        while i < n { steps.append(.removed(i)); i += 1 }
        while j < m { steps.append(.added(j)); j += 1 }
        // A change reads removal first, then addition.
        var ordered: [Step] = [], pendingAdds: [Step] = []
        for s in steps {
            switch s {
            case .added: pendingAdds.append(s)
            case .removed: ordered.append(s)
            case .same: ordered += pendingAdds; pendingAdds = []; ordered.append(s)
            }
        }
        return ordered + pendingAdds
    }
}
