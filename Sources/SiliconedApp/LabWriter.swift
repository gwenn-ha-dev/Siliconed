// **What the Prompt Lab asks of an engine that writes** (`Lab.swift` is the window). The window knows no
// model: it hands a text or an image and the family it writes for, and reads back events. The engine
// side — the best installed language model that fits in memory, its loading, its sampling — implements
// `LabWriter` (`EngineLabWriter.swift`) and is set in `LabModel.shared.writer`.
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
    /// The whole text, and the tokens it took (the stop token not counted).
    case finished(String, tokens: Int)
    case cancelled
}

/// What went wrong while writing: one sentence in the user's language, and the engine's stable `code` when it has one
/// (`render_already_running`…) — the window shows the sentence, `silicontrol` the two.
struct LabWriterFailure: LocalizedError {
    let errorDescription: String?
    let code: String?
    init(_ sentence: String, code: String? = nil) { errorDescription = sentence; self.code = code }
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
    /// The encoder is installed, the writer's head is not: `bytes` is what installing it downloads.
    case needsHead(bytes: Int64)
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
