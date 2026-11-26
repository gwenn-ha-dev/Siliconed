import Foundation
import SiliCore

// **What every render runs alike**: its decoder, and the check on what it hands back — pieces of `RenderLoop`, the
// decoder's also opened alone by the checks.

/// **A VAE decoder a render runs**: a latent to the picture's planes, on the engine's functions (`FluxVAEDecoder`,
/// `QwenImageVAEDecoder`, …). It is opened once for a batch — its plan drawn from B, its arena taken,
/// its functions loaded — then run for each latent.
package protocol LatentDecoder: SendableMetatype {
    /// **What `admit` judged and `open` opens**: the decoder's candidates, their functions built and named once
    /// (`BandStage.candidates`).
    associatedtype Admitted: Sendable
    /// The decoder `admit` judged, opened.
    static func open(_ admitted: Admitted) async throws -> Self
    /// **Whether the decoder of latents `[C, latentHeight, latentWidth]` of the VAE at `path` holds under `budget`**
    /// (bytes), judged now, nothing allocated (`BandStage.admit`): `OverBudget` otherwise; what `open` opens. A render
    /// asks it before its DiT, over the floor its DiT will give back, and opens what it judged after it.
    static func admit(path: String, latentHeight: Int, latentWidth: Int, budget: Int) throws -> Admitted
    /// **What `open` would specialize** under `budget`, its assets written, nothing allocated (`BandStage.prepare`).
    static func prepare(path: String, latentHeight: Int, latentWidth: Int, budget: Int) throws -> [Preparation]
    /// Whether B made its plan cut finer than the one it prefers (`bandDecodingNotice`).
    var narrowed: Bool { get }
    /// A latent, as the decoder reads it, to the picture's planes `[3, H, W]` in `[-1, 1]`.
    func pixels(_ latent: [Float]) async throws -> [Float]
}

/// A decoder that never cuts its plan finer than it prefers.
extension LatentDecoder {
    package var narrowed: Bool { false }
}

/// **The decoder of a render**, opened once for a batch: beside the DiT's last evaluation where B holds both at their
/// peaks (`Stage.beside`), after the DiT's close otherwise.
package final class DecoderSession<Decoder: LatentDecoder> {
    private let decoder: UncheckedSendable<Decoder>

    /// What `admit` judged, opened.
    package init(_ admitted: Decoder.Admitted) throws {
        decoder = UncheckedSendable(try blocking { try await Decoder.open(admitted) })
    }

    /// Judged under the machine's B (`admit`), then opened.
    package convenience init(path: String, latentHeight: Int, latentWidth: Int) throws {
        try self.init(try Decoder.admit(path: path, latentHeight: latentHeight, latentWidth: latentWidth, budget: MemoryBudget.machineCeiling()))
    }

    /// What the render says of the decoding's cut: `bandDecodingNotice` when B made it finer than the decoder prefers.
    package var warning: String? { decoder.value.narrowed ? bandDecodingNotice : nil }

    /// A latent, as the decoder reads it, to the picture's planes `[3, H, W]` in `[-1, 1]`.
    package func pixels(_ latent: [Float]) throws -> [Float] {
        let decoder = decoder
        return try blocking { try await decoder.value.pixels(latent) }
    }
}

/// **The latents a render's DiT left, its decoder's to decode** (`RenderLoop.keeping`): each image's latent, its
/// timings and its seconds so far, the prompt's tokens, and how many images the decoder already handed over. One file a
/// render, written once its DiT is done (a few MiB: 1 MiB a 1024² latent of 16 channels), each image gone from it
/// once handed over — measured 22/11/2026 on the M5 Pro: 6.1 ms written, 0.5 ms read back, four latents of 1 MiB.
package struct KeptLatents: Codable, Sendable, Equatable {
    package struct Image: Codable, Sendable, Equatable {
        package let index: Int
        package let latent: [Float]
        package let timings: Engine.Timings
        package let elapsed: TimeInterval

        package init(index: Int, latent: [Float], timings: Engine.Timings, elapsed: TimeInterval) {
            (self.index, self.latent, self.timings, self.elapsed) = (index, latent, timings, elapsed)
        }

        private enum CodingKeys: String, CodingKey { case index, latent, timings, elapsed }
        // The floats as their bytes: as JSON numbers, a latent of a million floats is 10 MB and 40 ms to write.
        package init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let bytes = try c.decode(Data.self, forKey: .latent)
            guard bytes.count % 4 == 0 else { throw DecodingError.dataCorruptedError(forKey: .latent, in: c, debugDescription: "not floats") }
            latent = bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            (index, timings, elapsed) = (try c.decode(Int.self, forKey: .index), try c.decode(Engine.Timings.self, forKey: .timings),
                                         try c.decode(TimeInterval.self, forKey: .elapsed))
        }
        package func encode(to encoder: any Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(index, forKey: .index)
            try c.encode(latent.withUnsafeBytes { Data($0) }, forKey: .latent)
            try c.encode(timings, forKey: .timings)
            try c.encode(elapsed, forKey: .elapsed)
        }
    }

    package let tokens: Int
    package private(set) var images: [Image]
    package private(set) var handed = 0

    package init(tokens: Int, images: [Image]) { (self.tokens, self.images) = (tokens, images) }

    /// The file at `url`, nil when there is none or what it holds is not a render's latents.
    package static func read(_ url: URL) -> KeptLatents? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? PropertyListDecoder().decode(KeptLatents.self, from: data)
    }

    package func write(to url: URL) throws {
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        try encoder.encode(self).write(to: url, options: .atomic)
    }

    /// Image `index` handed over (or lost): gone from the file at `url`, written again.
    package mutating func hand(_ index: Int, to url: URL?) {
        images.removeAll { $0.index == index }
        handed += 1
        guard let url else { return }
        try? write(to: url)
    }
}

/// What a render says when B made its decoder cut finer than it prefers, one of the closed set the app says in the
/// user's language (`AppState.journalLine`): more calls of smaller bands, the same image, a longer decoding.
package let bandDecodingNotice = "Decoding by bands to stay within this Mac's memory limit: the same image, a longer decoding"

/// A latent or an image with a non-finite value is not delivered: the image is lost (`Engine.Event.unusable`), and a
/// batch left with none fails (`EngineError.nonFiniteImage`).
enum FiniteImage {
    static func isFinite(_ values: [Float]) -> Bool { values.allSatisfy(\.isFinite) }
}
