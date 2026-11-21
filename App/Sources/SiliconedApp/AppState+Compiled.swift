// **This Mac's compilations**: the functions' assets and what Core AI specialized of them (`CoreAI.compiled`), which
// Settings ▸ Macs ▸ This Mac weighs and erases (`CompiledRow`), and `silicontrol compiled` as well.
//
// - **The engine erases them** (`EngineWorker.erase`), in the order of its calls: after the work before, before the work
//   after, which compiles again what it needs. Nothing waits for it here; a render asked meanwhile waits there.
// - **An update erases them by itself**, at launch (`AppState.compiledAtLaunch`): an asset is named by its bytes, and the
//   version's change stands for what no plan of the new one names.

import Foundation
import Siliconed

extension AppState {
    /// **What the compilations weigh**, read off the main thread and kept (`compiledBytes`): the pane, `silicontrol
    /// compiled`, and the end of an erasure read it here.
    @discardableResult
    func weighCompiled() async -> Int {
        let bytes = await Task.detached(priority: .utility) { CoreAI.compiled() }.value
        compiledBytes = bytes
        return bytes
    }

    /// **The compilations erased by the engine**; `done` handed what it freed, or why it failed, once the new weight is
    /// read — without it, the window says it. A request while one is under way is dropped (`erasingCompiled` greys
    /// « Erase… », `silicontrol` refuses it).
    func eraseCompiled(_ done: (@MainActor (Result<Int, any Error>) -> Void)? = nil) {
        guard !erasingCompiled else { return }
        erasingCompiled = true
        EngineWorker.shared.erase { result in
            Task { @MainActor [weak self] in await self?.erased(result, done) }
        }
    }

    private func erased(_ result: Result<Int, any Error>, _ done: (@MainActor (Result<Int, any Error>) -> Void)?) async {
        if case .success = result { CoreAI.markCompiled(for: CoreAI.thisApp) }
        await weighCompiled()
        erasingCompiled = false
        if let done { return done(result) }
        switch result {
        case .success(let freed): flash(String(localized: "\(Figures.disk(freed)) freed"))
        case .failure(let e): problem = Problem(e)
        }
    }

    /// **At launch**: what the engines before this one left erased off the main thread (`CoreAI.sweep`: up to 10 GB that
    /// no engine reads), and the compilations of another installation erased by the engine, before the first call it
    /// runs. A failure leaves the mark unwritten: the next launch tries again.
    func compiledAtLaunch() {
        let library = library
        Task.detached(priority: .utility) { _ = CoreAI.sweep(library) }
        guard CoreAI.due(installed: CoreAI.thisApp) else { return }
        eraseCompiled { result in
            if case .failure(let e) = result {
                FileHandle.standardError.write(Data("siliconed: the compilations of another installation stay: \(e)\n".utf8))
            }
        }
    }
}
