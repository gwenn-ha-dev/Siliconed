// **`AppState`: the compilations made ahead** (`EngineWorker.prepare`) — what Core AI would compile at a render, compiled
// before it comes. The engine specializes between its calls, one function at a time: a render asked meanwhile waits for
// the one under way, then the rest waits for the render.
//
// - **The form's key** (the model, the size, the LoRA, the Neural Engine, a start image, an edit's images, B, the
//   catalog's reading — another version of a model, its other functions): once it rests
//   (`pause`), its render's specializations replace what the form's last key left, first in line. The prompt is the
//   form's (a stand-in while it is empty): it sets only the text's palier.
// - **In the background**, after the form's: the default key (1024², no LoRA) of a model that becomes ready — an
//   installation, an import, another library —, and of every installed model when this app or this macOS is not the one
//   they were last prepared for (`CoreAI.preparationDue`): an update compiles again, on battery too, shown as any
//   compilation (`compilingCoreAI`): the window's band says what, and how far (`CompilationMark`).

import Foundation
import Siliconed

/// What a render compiles for, as the form sets it.
struct PreparationKey: Equatable {
    let identifier: String
    let width, height: Int
    let loras: [String]
    let neuralEngine, start: Bool
    let references: [[Int]]
    let budget: Int?
    let catalog: Int
}

extension AppState {
    /// The rest before the form's key is prepared: a key changed meanwhile is the one prepared.
    static let pause = Duration.seconds(1.5)
    /// The prompt prepared while the form's is empty.
    static let standInPrompt = "a photograph"

    /// The form's key, `nil` when this Mac does not render it (another Mac's, a model not installed, no library).
    var preparationKey: PreparationKey? {
        guard mirror == nil, libraryReachable, readySet.contains(identifier) else { return nil }
        let f = outputFormat
        return PreparationKey(identifier: identifier, width: f.width, height: f.height, loras: activeLoRAs.map(\.path),
                              neuralEngine: neuralEngine, start: usedStart != nil,
                              references: renderReferences.map { [$0.width, $0.height] }, budget: memoryBudget, catalog: libraryRead)
    }

    /// **At launch, after the compilations of another installation are erased** (`compiledAtLaunch`, the engine's calls
    /// in their order): every installed model in the background when due, then the form's key for as long as the app lives.
    func prepareAtLaunch() {
        guard mirror == nil else { return }
        EngineWorker.shared.preparing = { what in
            DispatchQueue.main.async { MainActor.assumeIsolated { MainWindow.shared.local.preparationTold(what) } }
        }
        if libraryReachable, CoreAI.preparationDue() {
            prepareInBackground(cards.map(\.id).filter(readySet.contains))
            CoreAI.markPrepared()
        }
        watchPreparation()
        keyChanged()
    }

    /// **A preparation's compilations said as one burst** (`Measures.burst`, as a render's): shown at the first, its count
    /// moving with each, hidden `burst` seconds past the last. Said function by function, the word came and went several
    /// times a second, and every view that reads `compilingCoreAI` with it, while the machine compiled.
    func preparationTold(_ what: WorkerPreparation?) {
        preparationEnd?.cancel()
        if let what {
            if compilingAhead != what { compilingAhead = what }
            return
        }
        preparationEnd = Task { [weak self] in
            try? await Task.sleep(for: .seconds(Measures.burst))
            guard !Task.isCancelled, let self, self.compilingAhead != nil else { return }
            self.compilingAhead = nil
        }
    }

    /// **The default key of each of `models`, after the form's** (1024², no LoRA, the Neural Engine as chosen).
    func prepareInBackground(_ models: [String]) {
        guard mirror == nil, libraryReachable else { return }
        let format = ModelCard.recommendedFormats[0]
        for model in models {
            let request = Request(preparedPrompt, width: format.width, height: format.height)
            EngineWorker.shared.prepare(request, model: model, library: library, background: true)
        }
    }

    private var preparedPrompt: String {
        let typed = (try? alternatives.get().prompts.first) ?? prompt
        return typed.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Self.standInPrompt : typed
    }

    private func watchPreparation() {
        withObservationTracking { _ = preparationKey } onChange: { [weak self] in
            Task { @MainActor in
                self?.keyChanged()
                self?.watchPreparation()
            }
        }
    }

    /// The form's key changed: prepared once it rests.
    private func keyChanged() {
        preparation?.cancel()
        guard let key = preparationKey, key != preparedKey else { return }
        preparation = Task { [weak self] in
            try? await Task.sleep(for: Self.pause)
            guard !Task.isCancelled, let self, self.preparationKey == key else { return }
            await self.prepare(key)
        }
    }

    /// **The form's render at `key`** sent to the engine, its images decoded off the main thread as `execute` decodes them.
    private func prepare(_ key: PreparationKey) async {
        let start = usedStart, references = renderReferences, strength = startStrength
        let loras = activeLoRAs.map { LoRAEntry($0.path, strength: $0.strength) }
        let prompt = preparedPrompt
        guard let images = try? await Task.detached(priority: .utility, operation: {
            (try start?.imageRGB(), try references.map { try $0.imageRGB() })
        }).value, !Task.isCancelled, preparationKey == key else { return }
        var request = Request(prompt, width: key.width, height: key.height, loras: loras, image: images.0,
                              strength: start == nil ? Strength.defaultValue : strength)
        request.references = images.1
        preparedKey = key
        EngineWorker.shared.prepare(request, model: key.identifier, library: library)
    }
}
