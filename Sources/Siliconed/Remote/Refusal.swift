import Foundation

/// **A refusal or a failure as it crosses**, so that it is said in the language of the Mac that reads it,
/// not of the one that wrote it. The engine's own errors cross as themselves (`EngineError` is `Codable`:
/// its case and its parameters, translated by `Problem(EngineError)` on the borrowing side); the app's own
/// sentences as `AppRefusal`, a code with its parameters; `text` is what has neither — the lending Mac's
/// words, shown as they come.
package enum RemoteRefusal: Codable, Equatable, Sendable, ExpressibleByStringLiteral {
    case engine(EngineError)
    case app(AppRefusal)
    case text(String)

    package init(stringLiteral text: String) { self = .text(text) }
}

/// **The sentences only the app says**, by code. Parameters are identifiers or numbers, never a phrase of
/// the sender's language: a model by its id (its name is read where the sentence is made).
package enum AppRefusal: Codable, Equatable, Sendable {
    /// An image of the history asked for by its `i` ordinal, no longer there.
    case imageGone(Int)
    case tooManyReferences(count: Int, model: String, max: Int)
    case editSurface(width: Int, height: Int)
    /// `name`: the LoRA's file name without its path or suffix.
    case loraNotInstalled(name: String)
    case seedCount
    case enlargeModelDownload, enlargeFailed
    case gridNotCropped, sketchNotCropped
    case cropNeeds(side: Int), cropSmallest(side: Int)
    /// A restore asked for images the session's trash no longer holds (it holds 64 per Mac, and empties when this Mac quits).
    case imagesNotRestorable
    /// A license accepted from a text that is not this Mac's own (`LicenseDigest`).
    case licenseMismatch(model: String)
    /// A render whose named subjects came without their rewritten prompt, or without their images.
    case subjectsIncomplete
}
