import Foundation

/// **What an older version left in the preferences**: carried, once, into what this one reads, then deleted — at launch,
/// before the state reads them.
///
/// - 0.30's Precision (`precision`): « Exact » kept the model's own arithmetic on the GPU alone, what the Neural Engine
///   switch off is now; « Neural Engine » (`ane`) is the switch on, its default. Carried only where the switch was never
///   set: a choice made since stands.
/// - 0.30's height of the Prompt Lab's image box (`labImagesBoxHeight`): the box now keeps a number of rows
///   (`labImagesRows`), and a height has none to give.
package enum RetiredPreferences {
    static let precision = "precision", labImagesBoxHeight = "labImagesBoxHeight"

    /// `neuralEngineKey`: where the app keeps the Neural Engine switch.
    package static func migrate(_ defaults: UserDefaults, neuralEngineKey: String) {
        if defaults.string(forKey: precision) == "exact", defaults.object(forKey: neuralEngineKey) == nil {
            defaults.set(false, forKey: neuralEngineKey)
        }
        defaults.removeObject(forKey: precision)
        defaults.removeObject(forKey: labImagesBoxHeight)
    }
}
