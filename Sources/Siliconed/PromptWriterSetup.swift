import CoreGraphics
import Foundation

/// **What the lab needs around the writer, decided without a weight**: the image the user drops, what is
/// installed and what is not.
///
///     status:  encoder + tokenizer installed? ─ no ─▶ notInstalled
///              head installed?                ─ no ─▶ needsHead(download size)      one button in the lab
///              need from shapes ≤ B?          ─ no ─▶ needsMac(GB)                  `PromptWriterNeed.Availability`
///                                                └ yes ▶ ready(weights on disk)
package enum PromptWriterStatus: Equatable, Sendable {
    case notInstalled
    /// The encoder is there, the head (`Family.writerHead`) is not: `downloadBytes` is what installing it fetches.
    case needsHead(downloadBytes: Int)
    case needsMac(gigabytes: Int)
    /// `weightBytes`: the encoder's and the head's maps on disk.
    case ready(weightBytes: Int)

    /// The decision, from facts; `fit` is the writer's own verdict (`PromptWriter.availability`), asked only once
    /// everything is installed. Pure: the tests hand it every combination.
    package static func resolve(encoderInstalled: Bool, headInstalled: Bool, fit: () -> PromptWriterNeed.Availability?,
                                headDownloadBytes: Int, weightBytes: Int) -> PromptWriterStatus {
        guard encoderInstalled else { return .notInstalled }
        guard headInstalled else { return .needsHead(downloadBytes: headDownloadBytes) }
        switch fit() {
        case .needsMac(let gigabytes, _)?: return .needsMac(gigabytes: gigabytes)
        case .available?, nil: return .ready(weightBytes: weightBytes)
        }
    }
}

extension Library {
    /// The family whose encoder writes (`PromptWriter(library:)`).
    package static let writerFamily = Family.qwenImage21

    /// Its encoder map and its tokenizer are installed — the head is a separate question (`writerHeadMap`).
    package var writerEncoderInstalled: Bool {
        Self.exists(map(encoderMap(Self.writerFamily)))
            && ((try? component(Self.writerFamily, "processor")).map { Self.exists($0 + "/tokenizer.json") } ?? false)
    }

    /// What installing the writer's head downloads: two byte ranges of the encoder's file (the configs are a few KB).
    package static var writerHeadDownloadBytes: Int { Installer.writerHeadDownloadBytes }

    /// The encoder's and the head's maps on disk.
    package var writerWeightBytes: Int {
        Self.size(map(encoderMap(Self.writerFamily))) + (writerHeadMap(Self.writerFamily).map { Self.size($0) } ?? 0)
    }

    /// Where the writer stands on this disk; `writer` is asked for the memory verdict once everything is there.
    package func writerStatus(_ writer: () -> PromptWriter?) -> PromptWriterStatus {
        PromptWriterStatus.resolve(encoderInstalled: writerEncoderInstalled,
                                   headInstalled: writerHeadMap(Self.writerFamily) != nil,
                                   fit: { writer()?.availability() },
                                   headDownloadBytes: Self.writerHeadDownloadBytes, weightBytes: writerWeightBytes)
    }
}

/// **A dropped image, as the writer reads it**: ImageIO's decoding, the EXIF orientation applied, transparency over
/// white — and for an image that has transparency, the rule Qwen-Image-2.1's own reference images follow
/// (`QwenImage21Pipeline.preparedReference`: Pillow's Lanczos on the premultiplied copy, then `white.paste(img, mask=A)`;
/// compositing before resizing is what an analogy would do). An opaque one goes in as decoded: `PromptWriter` brings
/// it to its caption size itself, once.
package enum PromptWriterImage {
    package static func rgb8(from data: Data) throws(EngineError) -> Qwen3VLImages.RGB8 {
        let image = try ImageRGB(data: data, name: "image")
        return QwenImage21Pipeline.rgb8(image.rgba == nil ? image : QwenImage21Pipeline.preparedReference(image))
    }
}
