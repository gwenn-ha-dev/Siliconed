// How a figure is written in the user's language (`Figures`), and the labels of a stage and an orientation.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

/// Durations, seconds, strengths and memory, formatted according to the system language (`0,85 s` in
/// French, `0.85 s` in English). The developer's command line keeps its decimal point: it is a
/// syntax, not a display.
enum Figures {
    /// `0.85 sec`, `12.3 sec` (`0,85 s` in French, `0.85秒` in Japanese) — the precision of the engine's timers.
    static func seconds(_ s: Double) -> String {
        Measurement(value: s, unit: UnitDuration.seconds)
            .formatted(.measurement(width: .abbreviated, usage: .asProvided,
                                    numberFormatStyle: .number.precision(.fractionLength(s < 10 ? 2 : 1))))
    }

    /// `42 s`, `1 min 5 s`, `1 h 2 min` — a waiting time ("42 sec", "1 min, 5 sec" in English).
    static func duration(_ s: Double) -> String {
        let d = Duration.seconds(max(0, s.rounded()))
        return s < 3600
            ? d.formatted(.units(allowed: [.minutes, .seconds], width: .abbreviated))
            : d.formatted(.units(allowed: [.hours, .minutes], width: .abbreviated))
    }

    /// `1024 × 1024`, width first — every size the app shows, in every language. Its digits are never grouped (`1024`,
    /// not `1 024`): a size is read as a pair of pixel counts, not as quantities.
    static func size(_ width: Int, _ height: Int) -> String { "\(String(width)) × \(String(height))" }

    static func strength(_ f: Double) -> String { f.formatted(.number.precision(.fractionLength(2))) }

    /// `6.1`, `0.84` — TFLOP/s, the unit said by its column or its sentence: a tenth from 1, a hundredth under.
    static func tflops(_ x: Double) -> String { x.formatted(.number.precision(.fractionLength(x < 1 ? 2 : 1))) }

    /// `40 %` in French, `40%` in English — `n` out of 100.
    static func percent(_ n: Int) -> String { (Double(n) / 100).formatted(.percent.precision(.fractionLength(0))) }

    static func hour(_ d: Date) -> String { d.formatted(date: .omitted, time: .shortened) }

    static func memory(_ bytes: Int) -> String { Int64(bytes).formatted(.byteCount(style: .memory)) }

    /// The space on disk, as the Finder counts it (`20.4 GB`).
    static func disk(_ bytes: Int) -> String { Int64(bytes).formatted(.byteCount(style: .file)) }
}

extension Engine.Stage {
    var label: String {
        switch self {
        case .text: String(localized: "Encoding the prompt")
        case .image: String(localized: "Encoding the image")
        case .denoising: String(localized: "Denoising")
        case .decoding: String(localized: "Decoding")
        }
    }
}

extension RecommendedFormat.Orientation {
    var label: String {
        switch self {
        case .square: String(localized: "Square")
        case .portrait: String(localized: "Portrait")
        case .landscape: String(localized: "Landscape")
        }
    }
    /// A symbol whose silhouette follows the aspect ratio.
    var symbol: String {
        switch self {
        case .square: "square"
        case .portrait: "rectangle.portrait"
        case .landscape: "rectangle"
        }
    }
}
