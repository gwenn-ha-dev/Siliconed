// **`AppState`: the diagnostic** (« Measure This Mac »), and `unmount` when the window closes.
// The state itself is in `AppState.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

extension AppState {
    // MARK: Diagnostic

    /// Why the diagnostic cannot start now, or `nil`. It occupies the GPU and the CPU like a render —
    /// measured beside one, it would measure both.
    var diagnosticBlocker: String? { diagnosticBlock?.sentence }

    /// The same, as a code: in a mirror, the lending Mac's (it renders for others too, which this one does not see).
    var diagnosticBlock: RemoteBlocker? {
        if mirror != nil { return mirrorMachine?.blocker }
        if diagnosticRunning { return .measuring }
        if busy { return .rendering }
        if forgeInProgress != nil { return .installing }
        return nil
    }

    func openDiagnostic() {
        managementOpen = false
        switch diagnostic {
        case .finished, .running: break
        case .ready, .failed: diagnostic = .ready
        }
        diagnosticOpen = true
    }

    /// **Runs `Diagnostic.run`** in a child process (`Diagnostic.Child`, about a minute, five at most), the queue held meanwhile.
    func startDiagnostic() {
        guard diagnosticBlocker == nil else { return }
        // In a mirror the lending Mac measures itself; the sheet shows it starting until its word comes.
        if let mirror {
            diagnostic = .running(done: 0, total: Diagnostic.Stage.allCases.count, line: Self.diagnosticLine(0))
            mirror.send(.measure)
            return
        }
        measure()
    }

    /// **The measure, in a child process**: this app's executable with `Diagnostic.Child.argument`. The
    /// diagnostic reads the engine's settings, and the settings are read once per process: measured here, before the
    /// first render, it would freeze this process on the defaults and the model could no longer apply. The child
    /// measures on the defaults, whatever this process uses — a measure taken again is not skewed by the model in use —,
    /// and frees its memory on exit. **A route that kills the child is left out and the measure starts again**
    /// (`Diagnostic.Child.Supervisor`): the report says which route and why, and no render loads a route the report has not seen load.
    /// Its report applies from the next render (`keep`). `next` calls this past its
    /// own guards, a job waiting; the sheet and the remote control through `startDiagnostic`.
    func measure() {
        guard let executable = Bundle.main.executableURL else { return }
        let supervisor = Diagnostic.Child.Supervisor(executable: executable)
        diagnosticSupervisor = supervisor
        diagnosticStopped = false
        diagnostic = .running(done: 0, total: Diagnostic.Stage.allCases.count, line: String(localized: "Preparing…"))
        // A thread of its own: the supervisor waits on the child's pipe for the whole measure, which a task of the pool should not.
        Thread.detachNewThread { [weak self] in
            let outcome = supervisor.run(
                onStage: { stage in Task { @MainActor in self?.diagnosticProgress(stage) } },
                onRestart: { _ in Task { @MainActor in self?.diagnosticRestarted() } })
            Task { @MainActor in self?.diagnosticFinished(outcome) }
        }
    }

    /// The child's word came: the report, a failure, or the stop.
    private func diagnosticFinished(_ outcome: Diagnostic.Child.Outcome) {
        diagnosticSupervisor = nil
        if diagnosticStopped || outcome.stopped {
            diagnostic = .ready
        } else if let report = outcome.report {
            diagnostic = .finished(report)
            keep(report)   // what this Mac plans with: from the next render — whatever it found
        } else {
            // The one refusal the app says in its own words: another process renders.
            let failure = outcome.failure ?? ("internal", "the measuring process did not finish")
            let problem = failure.code == EngineError.renderAlreadyRunning.code
                ? Problem(EngineError.renderAlreadyRunning)
                : Problem(EngineError.internalFailure(component: "diagnostic", detail: failure.detail))
            diagnostic = .failed(problem)
        }
        next()
    }

    /// A route killed the child: the measure starts again, and the progress with it.
    private func diagnosticRestarted() {
        guard case .running = diagnostic else { return }
        diagnostic = .running(done: 0, total: Diagnostic.Stage.allCases.count, line: String(localized: "Preparing…"))
    }

    func cancelDiagnostic() {
        if let mirror { if diagnosticRunning { mirror.send(.stopMeasure) }; return }
        guard let supervisor = diagnosticSupervisor else { return }
        diagnosticStopped = true
        supervisor.stop()
    }

    /// The library names the stage (`Diagnostic.Stage`): the app says it in the user's language.
    private func diagnosticProgress(_ stage: Diagnostic.Stage) {
        guard case .running = diagnostic, let i = Diagnostic.Stage.allCases.firstIndex(of: stage) else { return }
        diagnostic = .running(done: i, total: Diagnostic.Stage.allCases.count, line: Self.diagnosticLine(stage))
    }

    static func diagnosticLine(_ stage: Diagnostic.Stage) -> String {
        switch stage {
        case .widening: String(localized: "weight preparation")
        case .witness: String(localized: "the engines’ first measurement")
        case .engines: String(localized: "each engine alone")
        case .sharing: String(localized: "the engines together")
        case .forms: String(localized: "the weight forms")
        case .regime: String(localized: "the engines again")
        }
    }

    /// **« Open a GitHub Issue »**: the report goes to the clipboard, and GitHub's form opens with every field
    /// prefilled but the report's, which the user pastes. Never the report in the link: signed out, GitHub
    /// sends a new issue's URL through its login page inside `return_to`, escaped once more, and a link
    /// carrying ~6 KB of JSON comes out of it broken — the short one survives the round trip.
    /// Returns what to tell the user.
    func openIssue(_ report: Diagnostic.Report) -> String? {
        do {
            copyJSON(report)
            NSWorkspace.shared.open(try report.with(render: reportNote(for: report)).issueURL())
            return String(localized: "The report is in the clipboard: paste it into the form’s “Report” field.")
        } catch {
            return Problem(error).text
        }
    }

    func copyJSON(_ report: Diagnostic.Report) {
        guard let data = try? report.with(render: reportNote(for: report)).json() else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(String(decoding: data, as: UTF8.self), forType: .string)
        flash(String(localized: "Report copied (JSON)"))
    }

    /// **The last render against this report's model** (`RenderLoop.Note`, ~1 KB), if there was one: it travels with the report and is
    /// shown by the sheet. A render made under another model — a measure taken since, which applies from the next render — is not
    /// this report's: its gap would say something about a model the report does not hold.
    func reportNote(for report: Diagnostic.Report) -> RenderLoop.Note? {
        renderNote.flatMap { $0.map == report.date ? $0 : nil }
    }

    /// The window closes: nothing must keep occupying the GPU with no view to show it.
    func unmount() {
        toastTask?.cancel()
        clearQueue()
        task?.cancel()
        // A mirror's window closing does not stop the measure of the Mac it shows.
        if mirror == nil { cancelDiagnostic() }
    }
}
