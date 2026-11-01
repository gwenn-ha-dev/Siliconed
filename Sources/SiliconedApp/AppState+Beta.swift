// The four Beta options (Settings › Advanced › Beta) by name: what `silicontrol set beta` reads and writes. Each is the property
// its switch is bound to, so setting one here is a click there — the same `didSet`, the same effects.

import Siliconed

extension AppState {
    func beta(_ option: BetaOption) -> Bool {
        switch option {
        case .moreSolvers: developerMode
        case .fasterRendering: fasterRenderingBeta
        case .sizeScales: sizeScalesBeta
        case .negativeLoRA: negativeLoRAStrengths
        }
    }

    func setBeta(_ option: BetaOption, _ on: Bool) {
        switch option {
        case .moreSolvers: developerMode = on
        case .fasterRendering: fasterRenderingBeta = on
        case .sizeScales: sizeScalesBeta = on
        case .negativeLoRA: negativeLoRAStrengths = on
        }
    }
}
