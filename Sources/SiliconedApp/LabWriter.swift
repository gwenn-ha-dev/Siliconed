// **What the Prompt Lab asks of an engine that writes** (`Lab.swift` is the window). The window knows no
// model: it hands a text or an image and the family it writes for, and reads back events. The engine
// side — the best installed language model that fits in memory, its loading, its sampling — implements
// `LabWriter` and is set in `LabModel.shared.writer`; until then `FakeLabWriter` lets the window be
// seen and judged.
//
//     window ──write(input, target)──▶ writer      loading(progress) · text(so far) · finished(text) · cancelled
//     window ──release()──────────────▶ writer      a render starts: give the memory back, now
//
// - **`text` carries the whole text so far, not a delta**: the window shows what it is given and
//   cannot drift from the engine when a token is retracted (a detokenizer revises its last word).
// - **Cancelling is the consumer's**: the window cancels the task reading the stream, and the stream's
//   `onTermination` stops the sampling. `cancelled` is for a writer that stops by itself.
// - **`release()` may be called at any time**, loaded or not, mid-write or not; when it returns, the
//   writer holds no weight. The next `write` loads again.

import Foundation
import Siliconed

/// What the window gives the writer.
enum LabInput: Sendable, Equatable {
    /// A prompt to improve.
    case text(String)
    /// An image to describe: the file's bytes as they were dropped or opened (any format the system reads).
    case image(Data)
}

enum LabEvent: Sendable, Equatable {
    /// The writer is being loaded; `nil` while the fraction is not known.
    case loading(Double?)
    /// Everything written so far.
    case text(String)
    case finished(String)
    case cancelled
}

/// **Whether a writer can write now**, and if not, why — in the terms the window says to the user.
enum LabAvailability: Equatable, Sendable {
    /// `size`: the writer's weights, in bytes.
    case ready(name: String, size: Int64)
    case unavailable(LabUnavailable)
}

enum LabUnavailable: Equatable, Sendable {
    /// No installed model carries a language model that writes.
    case notInstalled
    /// One is installed, and does not fit this Mac: `gigabytes` of RAM would.
    case needsMemory(gigabytes: Int)
    /// Being downloaded or converted.
    case preparing
}

protocol LabWriter: AnyObject, Sendable {
    /// Read when the window opens, when it comes forward and when a render ends — cheap, no loading.
    var availability: LabAvailability { get }
    /// `target` writes for that family (its publisher's guidance decides how). The stream ends after
    /// `finished` or `cancelled`, or throws.
    func write(_ input: LabInput, target: Family) -> AsyncThrowingStream<LabEvent, Error>
    func release() async
}

/// **A stand-in that streams a canned text at ~25 tokens/s**, so the window can be judged before the
/// engine is wired. It touches nothing on disk and loads nothing.
final class FakeLabWriter: LabWriter {
    var availability: LabAvailability { .ready(name: "Sample writer", size: 16_000_000_000) }

    func write(_ input: LabInput, target: Family) -> AsyncThrowingStream<LabEvent, Error> {
        let result = Self.canned(for: input, target: target)
        return AsyncThrowingStream { continuation in
            let producer = Task {
                for step in 1...6 {
                    try? await Task.sleep(for: .milliseconds(100))
                    if Task.isCancelled { continuation.yield(.cancelled); continuation.finish(); return }
                    continuation.yield(.loading(Double(step) / 6))
                }
                // Words with their trailing space stand for tokens.
                var tokens: [Substring] = [], start = result.startIndex
                for i in result.indices where result[i] == " " {
                    let next = result.index(after: i)
                    tokens.append(result[start..<next]); start = next
                }
                if start < result.endIndex { tokens.append(result[start...]) }
                var soFar = ""
                for token in tokens {
                    try? await Task.sleep(for: .milliseconds(40))
                    if Task.isCancelled { continuation.yield(.cancelled); continuation.finish(); return }
                    soFar += token
                    continuation.yield(.text(soFar))
                }
                continuation.yield(.finished(result))
                continuation.finish()
            }
            continuation.onTermination = { _ in producer.cancel() }
        }
    }

    func release() async {}

    private static func canned(for input: LabInput, target: Family) -> String {
        switch input {
        case .text(let text):
            let base = text.trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "posing", with: "standing")
                .trimmingCharacters(in: CharacterSet(charactersIn: ".,"))
            return "A candid photograph of \(base), soft window light, shallow depth of field, warm tones"
        case .image:
            return "A woman in a long green coat stands between tall wooden shelves in an old library, a book open in her hands, warm lamplight on her face, shallow depth of field, photographed on 35 mm film"
        }
    }
}
