// An image of a job lost while the others of its render came: what the job's answer says of it.

/// **An image the engine lost** (`Engine.Event.unusable`: its latent or its pixels not finite) in a job whose other seeds
/// rendered. The job keeps the images that came and ends done; each lost one is a line of its `"warnings"`, by its seed and
/// its stage, in the engine's English as its other warnings are (`EngineError.nonFiniteImage`'s sentence, after its code).
/// The window says it in its own words (`Problem`). A job whose images are all lost is no warning: its render fails with
/// `nonFiniteImage`, and the job with it.
package enum LostImage {
    package static func warning(seed: UInt64, stage: Engine.Stage) -> String {
        let e = EngineError.nonFiniteImage(stage: stage)
        return "\(e.code): seed \(seed): \(e.errorDescription ?? "")"
    }

    /// Whether a warning is a lost image's: the window does not say it twice, its problem says it.
    package static func isWarning(_ line: String) -> Bool {
        line.hasPrefix(EngineError.nonFiniteImage(stage: .denoising).code + ": ")
    }
}
