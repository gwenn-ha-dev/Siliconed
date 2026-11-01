/// **The writer's loop guard**: greedy decoding can fall into a cycle it never leaves (a tag list that repeats
/// "solo, solo, solo, …" until `Limits.newTokens`, pages of it on screen). The decode itself is the transcription of
/// transformers and stays as it is; this is the product's answer, on the identifiers alone.
///
/// A loop is **exact**: the last `period` tokens repeat, token for token, at least `minimumRepeats` times over at least
/// `minimumSpan` tokens. The thresholds are what keeps a normal text out of it: "ha ha ha" is three repeats, a tag
/// list shares its commas but not a whole period, and a refrain a writer means (a line said four times) is short of
/// the span. The loop seen (a one-word tag every other token) is hundreds of repeats, so the margin is wide.
///
/// A fuzzy loop (a sentence that drifts at each turn) is not seen here, by design: nothing approximate stops a text.
package struct RepetitionGuard {
    package static let maximumPeriod = 64
    package static let minimumRepeats = 4
    package static let minimumSpan = 32

    package init() {}

    /// Where `tokens` should end, or nil when its tail is no loop: the first occurrence of the cycle is kept, the rest dropped.
    /// Asked after each token, so the cost stays at `maximumPeriod` comparisons in the common case.
    package static func cut(_ tokens: [Int]) -> Int? {
        let n = tokens.count
        guard n >= minimumSpan else { return nil }
        for period in 1...min(maximumPeriod, n / minimumRepeats) {
            // How far back the sequence is periodic with this period, from the end.
            var run = 0
            while run < n - period && tokens[n - 1 - run] == tokens[n - 1 - run - period] { run += 1 }
            let span = run + period
            guard span >= minimumSpan, span / period >= minimumRepeats else { continue }
            // `run` equal pairs from the end: the periodic stretch starts at n - span, one occurrence is kept.
            return n - span + period
        }
        return nil
    }
}
