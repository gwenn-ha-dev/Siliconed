// `set`: Settings' switches and the memory budget from the terminal.

import AppKit
import Foundation
import Measurement
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    /// **`set …`: Settings' switches, and the DiT box's « Preview every step », from the terminal** (`SetOrder` reads every
    /// subcommand). Each goes through the property its switch is bound to, so the choice is written and the window follows,
    /// exactly as a click. Setting the value it already has is not a change. `set save-images FOLDER|off` is Settings ›
    /// General's Save Every Image To… (`EveryImage`), this Mac's: an existing folder, never created, as the panel only offers one.
    func set(_ arguments: [String], cwd: String) throws -> [String: Any] {
        let order: SetOrder
        switch SetOrder.parse(arguments) {
        case .success(let parsed): order = parsed
        case .failure(.usage(let line)): throw Refusal("usage", line, help: "silicontrol help set")
        case .failure(.unknownFlag(let flag)): throw Self.unknownFlag(flag, text: false)
        case .failure(.unknownBeta(let name)):
            throw Refusal("usage", String(localized: "unknown Beta option: \"\(name)\""),
                          help: String(localized: "Beta options: \(BetaOption.names)"))
        }
        switch order {
        case .saveImagesState: return ["ok": true, "save_images": EveryImage.shared.folder?.path ?? NSNull()]
        case .saveImages(let typed):
            let previous: Any = EveryImage.shared.folder?.path ?? NSNull()
            if let typed {
                let path = absolute(typed, cwd)
                var isFolder: ObjCBool = false
                guard FileManager.default.fileExists(atPath: path, isDirectory: &isFolder), isFolder.boolValue else {
                    throw Refusal("write", String(localized: "not a folder: \(path)"), help: String(localized: "give an existing folder"))
                }
                EveryImage.shared.folder = URL(fileURLWithPath: path, isDirectory: true)
            } else {
                EveryImage.shared.folder = nil
            }
            return ["ok": true, "save_images": EveryImage.shared.folder?.path ?? NSNull(), "previous": previous]
        case .betaList: return ["ok": true, "beta": betaState()]
        case .beta(let option, let on): return setBeta(option, on)
        case .memory(let typed): return try setMemory(typed)
        case .toggle(let toggle, let on): return setToggle(toggle, on)
        }
    }

    /// **`set neural-engine|glass|updates|preview [on|off]`**, as a click on its switch; alone, it is read.
    /// - `neural-engine`: Settings › General's Neural Engine, through the property it is bound to (it writes the choice and
    ///   tells the engine). This Mac's preference, for what this Mac computes: a render it lends follows it, one it borrows
    ///   follows the lender's.
    /// - `glass`: Settings › General's Glass. `updates`: Settings › General's Check at Launch.
    /// - `preview`: the DiT box's « Preview every step » — the window's state, this Mac's or the one it computes on, whose
    ///   renders it changes from the next one on (`add --preview` and `--no-preview` still decide for one job).
    /// - `debug-live`: the Machine window's Debug Live, this Mac's (`AppState.debugLive`): the renders asked from then on
    ///   record their trace, which `trace` reads; off at every launch.
    func setToggle(_ toggle: SetOrder.Toggle, _ on: Bool?) -> [String: Any] {
        let (key, read, write): (String, () -> Bool, (Bool) -> Void) = switch toggle {
        case .neuralEngine: ("neural_engine", { self.local.neuralEngine }, { self.local.neuralEngine = $0 })
        case .glass: ("glass", { self.local.glass }, { self.local.glass = $0 })
        case .updates: ("updates", { Updates.shared.atLaunch }, { Updates.shared.atLaunch = $0 })
        case .preview: ("preview", { self.app.previews }, { self.app.previews = $0 })
        case .debugLive: ("debug_live", { self.local.debugLive }, { self.local.debugLive = $0 })
        }
        let previous = read()
        if let on { write(on) }
        return ["ok": true, key: read(), "previous": previous]
    }

    /// **`set memory [GB|default]`**: Settings › Macs › This Mac's Memory Limit — B, in gigabytes (GiB), within this
    /// Mac's range (`MemoryBudget.range`), through the property its slider is bound to; alone, it is read. The engine's
    /// next process takes it.
    func setMemory(_ typed: SetOrder.Memory?) throws -> [String: Any] {
        let previous = memoryState()
        switch typed {
        case nil: break
        case .standard?: local.memoryBudget = nil
        case .gigabytes(let gb)?:
            let range = MemoryBudget.range, bytes = gb * Double(1 << 30)
            guard bytes >= Double(range.lowerBound), bytes <= Double(range.upperBound) else {
                throw Refusal("usage", String(localized: "set memory [GB|default]: from \(range.lowerBound >> 30) to \(range.upperBound >> 30) on this Mac"),
                              help: "silicontrol help set")
            }
            local.memoryBudget = Int(bytes)
        }
        return ["ok": true, "memory": memoryState(), "previous": previous]
    }

    /// B as Settings shows it: what it is, whether the user chose it, the default and the range.
    func memoryState() -> [String: Any] {
        let range = MemoryBudget.range
        return ["budget_bytes": MemoryBudget.budget, "chosen": local.memoryBudget != nil,
                "default_bytes": MemoryBudget.standard, "range_bytes": [range.lowerBound, range.upperBound]]
    }

    /// The four Beta options as the window's switches hold them (this Mac's preferences, shown by Settings whichever Mac computes).
    func betaState() -> [String: Any] {
        var o: [String: Any] = [:]
        // JSON keys are snake_case; the name typed after `set beta` keeps its hyphens.
        for option in BetaOption.allCases { o[option.key] = local.beta(option) }
        return o
    }

    /// **`set beta OPTION on|off`**: the property the switch is bound to, on this Mac's state — and on the one the window shows,
    /// when it computes elsewhere, whose copy of the preference would otherwise stay as it was at the switch (the rack there
    /// reads it). Each option's own effect is its property's (`AppState.developerMode`'s `didSet` takes a solver the models no
    /// longer offer out of the form; `negativeLoRAStrengths`' brings the strengths back to 0).
    func setBeta(_ option: BetaOption, _ on: Bool) -> [String: Any] {
        let previous = local.beta(option)
        local.setBeta(option, on)
        if app !== local { app.setBeta(option, on) }
        return ["ok": true, "option": option.rawValue, "enabled": local.beta(option), "previous": previous, "beta": betaState()]
    }
}
