import Foundation

// **A model: a preset chain, a name and a license.** Each family's folder holds its modules, its card and the
// constructor of its chain; `construct` below reaches each through its family's traits, naming none.

/// **A model is a preset chain, a name and a license.** Its files are read from a `Library`
/// and checked at construction: a missing file throws here, before a minute of computation, with
/// its full path.
///
/// The constructors and `named` take the library under the same label, `in:` — that of
/// `ModelCard.missing(in:)`. A custom chain becomes a model through
/// `Model(card:chain:)`: it is the only thing `render` and `renderBatch` take.
package struct Model: Sendable {
    /// The name, the license, the recommended formats: what the catalog knows without opening anything.
    package let card: ModelCard
    package let chain: Chain
    /// The library it was built from, every file checked there (`Model.named`): whose licenses a render reads, and
    /// whose installation it holds the render lock of (`Preflight.run`). `nil` for a hand-composed chain.
    package let library: Library?
    /// **How the installed weights are stored** (`WeightFormats`), read once with the model; empty for a hand-composed chain.
    package var formats = WeightFormats()

    /// A hand-composed chain, under the sheet of the model whose denoiser it holds.
    package init(card: ModelCard, chain: Chain) { self.card = card; self.chain = chain; self.library = nil }

    package init(card: ModelCard, library: Library, chain: Chain) {
        self.card = card; self.chain = chain; self.library = library
    }

    /// `z-image` — what `--model` accepts; an imported model is
    /// called `<family>/<name>` (`z-image/my-model`).
    package var identifier: String { card.id }
    package var name: String { card.name }
    /// The architecture — the target a LoRA declares.
    package var family: Family { card.family }
    /// To be recalled to anyone distributing an image or the model.
    package var license: License { card.license }

    package static let identifiers = ModelCard.allCards.map(\.id)

    package enum Failure: Error, CustomStringConvertible {
        case unknown(String)
        package var description: String {
            switch self {
            case .unknown(let name):
                return "unknown model '\(name)' — \(Model.identifiers.joined(separator: ", "))"
            }
        }
    }

    /// The model `identifier` of the library — `EngineError.modelNotInstalled` if a file is missing
    /// (`ModelCard.missing(in:)` says which), `unknownModel` if there is no such model.
    package static func named(_ identifier: String, in b: Library) throws(EngineError) -> Model {
        try building(identifier) { try make(identifier, in: b) }
    }

    /// **The door of a model's construction**: a missing file is a model not installed.
    static func building(_ identifier: String, _ body: () throws -> Model) throws(EngineError) -> Model {
        do { return try body() } catch is MissingFile {
            throw .modelNotInstalled(model: identifier)
        } catch {
            throw EngineError(error)
        }
    }

    /// `named`, with the internal failure that says which file is missing.
    package static func make(_ identifier: String, in b: Library) throws -> Model {
        guard let card = b.card(identifier) else { throw Failure.unknown(identifier) }
        var model = try construct(card, in: b)
        model.formats = WeightFormats.read(card: card, in: b)
        return model
    }

    /// The card's chain over the library's files, each checked: what `missing(in:)` asks, without the formats.
    static func construct(_ card: ModelCard, in b: Library) throws -> Model {
        try card.family.traits.model(b, card, card.map ?? b.ditMap(card.family))
    }

    /// The library's `store/<name>` map, if it is there (`MissingFile` otherwise, folded into
    /// `modelNotInstalled` by `building`).
    static func map(_ name: String, _ b: Library) throws -> String {
        let path = name.hasPrefix("/") ? name : b.map(name)
        guard FileManager.default.fileExists(atPath: path) else { throw MissingFile(path, .map) }
        return path
    }
}
