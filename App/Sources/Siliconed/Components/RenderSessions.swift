import Foundation

// **What every render runs alike**: its decoder, and the check on what it hands back — pieces of `RenderLoop`, the
// decoder's also opened alone by the checks.

/// **A VAE decoder a render runs**: a latent to the picture's planes, on the engine's functions (`FluxVAEDecoder`,
/// `QwenImageVAEDecoder`, …). It is opened once for a batch — its plan drawn from B, its arena taken,
/// its functions loaded — then run for each latent.
package protocol LatentDecoder {
    /// The decoder of latents `[C, latentHeight, latentWidth]` of the VAE at `path`, under `budget` (bytes).
    static func open(path: String, latentHeight: Int, latentWidth: Int, budget: Int) async throws -> Self
    /// Whether B made its plan cut finer than the one it prefers (`bandDecodingNotice`).
    var narrowed: Bool { get }
    /// A latent, as the decoder reads it, to the picture's planes `[3, H, W]` in `[-1, 1]`.
    func pixels(_ latent: [Float]) async throws -> [Float]
}

/// A decoder that never cuts its plan finer than it prefers.
extension LatentDecoder {
    package var narrowed: Bool { false }
}

/// **The decoder of a render**, opened once for a batch. A render lets its DiT go before it opens this one: each stage
/// is held alone under B, never two models at once.
package final class DecoderSession<Decoder: LatentDecoder> {
    private let decoder: UncheckedSendable<Decoder>

    /// Under the machine's B.
    package init(path: String, latentHeight: Int, latentWidth: Int) throws {
        let budget = MemoryBudget.machineCeiling()
        decoder = UncheckedSendable(try blocking {
            try await Decoder.open(path: path, latentHeight: latentHeight, latentWidth: latentWidth, budget: budget)
        })
    }

    /// What the render says of the decoding's cut: `bandDecodingNotice` when B made it finer than the decoder prefers.
    package var warning: String? { decoder.value.narrowed ? bandDecodingNotice : nil }

    /// A latent, as the decoder reads it, to the picture's planes `[3, H, W]` in `[-1, 1]`.
    package func pixels(_ latent: [Float]) throws -> [Float] {
        let decoder = decoder
        return try blocking { try await decoder.value.pixels(latent) }
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
