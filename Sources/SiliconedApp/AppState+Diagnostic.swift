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

    /// **Runs `Diagnostic.run`** in a child process (`CapacityMeasure`, ~1 min), the queue held meanwhile.
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

    /// **The measure, in a child process**: this app's executable with `CapacityMeasure.argument`. The
    /// diagnostic reads the engine's settings (`Conductor`'s split, its quantum), and the settings are read
    /// once per process: measured here, before the first render, it would freeze this process on the
    /// defaults and the map could no longer apply. The child measures on the defaults, whatever this
    /// process uses — a measure taken again is not skewed by the map in use —, and frees its 1.4 GB on exit.
    /// Its report applies at once when nothing read the settings yet (`keep`). `next` calls this past its
    /// own guards, a job waiting; the sheet and the remote control through `startDiagnostic`.
    func measure() {
        guard let executable = Bundle.main.executableURL else { return }
        let process = Process()
        process.executableURL = executable
        // The families installed: the minute goes to the classes of the models this Mac will render with (none installed: all five).
        process.arguments = [CapacityMeasure.argument] + (installedFamilies.isEmpty ? [] : [installedFamilies.map(\.rawValue).sorted().joined(separator: ",")])
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        diagnosticProcess = process
        diagnosticStopped = false
        diagnostic = .running(done: 0, total: Diagnostic.Stage.allCases.count, line: String(localized: "Preparing…"))
        Task { [weak self] in
            var report: Diagnostic.Report?, failure: Problem?
            do {
                try process.run()
                for try await line in output.fileHandleForReading.bytes.lines {
                    switch CapacityMeasure.Line(line) {
                    case .stage(let stage): self?.diagnosticProgress(stage)
                    case .report(let r): report = r
                    case .failure(let p): failure = p
                    case nil: break
                    }
                }
                while process.isRunning { try? await Task.sleep(for: .milliseconds(20)) }
            } catch {
                failure = Problem(error)
            }
            guard let self else { return }
            self.diagnosticProcess = nil
            if self.diagnosticStopped {
                self.diagnostic = .ready
                self.fallBackOnStoredMap()
            } else if let report, process.terminationStatus == 0 {
                self.diagnostic = .finished(report)
                self.keep(report)   // what this Mac uses: from now if nothing read the settings yet, else from the next launch
            } else {
                let problem = failure ?? Problem(EngineError.internalFailure(
                    component: "diagnostic", detail: "the measuring process ended with status \(process.terminationStatus)"))
                self.diagnostic = .failed(problem)
                self.fallBackOnStoredMap()
                // A job waits for this measure: the failure is said where the render's would be.
                if !self.settingsRead, !self.file.isEmpty { self.problem = problem }
            }
            self.next()
        }
    }

    func cancelDiagnostic() {
        if let mirror { if diagnosticRunning { mirror.send(.stopMeasure) }; return }
        guard let process = diagnosticProcess else { return }
        diagnosticStopped = true
        process.terminate()
    }

    /// The library names the stage (`Diagnostic.Stage`): the app says it in the user's language.
    private func diagnosticProgress(_ stage: Diagnostic.Stage) {
        guard case .running = diagnostic, let i = Diagnostic.Stage.allCases.firstIndex(of: stage) else { return }
        diagnostic = .running(done: i, total: Diagnostic.Stage.allCases.count, line: Self.diagnosticLine(stage))
    }

    static func diagnosticLine(_ stage: Diagnostic.Stage) -> String {
        switch stage {
        case .widening: String(localized: "weight preparation")
        case .witness: String(localized: "the engines' first measurement")
        case .routes: String(localized: "compute routes")
        case .regime: String(localized: "the engines again")
        }
    }

    /// **« Open the issue »**: the report goes to the clipboard, and GitHub's form opens with every field
    /// prefilled but the report's, which the user pastes. Never the report in the link: signed out, GitHub
    /// sends a new issue's URL through its login page inside `return_to`, escaped once more, and a link
    /// carrying ~6 KB of JSON comes out of it broken — the short one survives the round trip.
    /// Returns what to tell the user.
    func openIssue(_ report: Diagnostic.Report) -> String? {
        do {
            copyJSON(report)
            NSWorkspace.shared.open(try report.with(render: renderNote).issueURL(includingReport: false))
            return String(localized: "The report is in the clipboard: paste it into the form's “Report” field.")
        } catch {
            return Problem(error).text
        }
    }

    func copyJSON(_ report: Diagnostic.Report) {
        // The last render against the map in use travels with the report (`RenderLoop.Note`, ~1 KB).
        guard let data = try? report.with(render: renderNote).json() else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(String(decoding: data, as: UTF8.self), forType: .string)
        flash(String(localized: "Report copied (JSON)"))
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

/// **The capacity measure's child process**: `Siliconed --measure-capacity` runs `Diagnostic.run` before
/// anything else (`SiliconedApplication.init`), says each stage and then the report on its standard
/// output, one line each, and exits — no window, no Dock icon: it never reaches `NSApplication.run`.
enum CapacityMeasure {
    /// `--measure-capacity [family,family,…]`: the families the map decides for.
    static let argument = "--measure-capacity"

    /// **The child's side**: does nothing unless the process was launched with `argument`; then never returns.
    static func runIfAsked() {
        guard CommandLine.arguments.dropFirst().first == argument else { return }
        setvbuf(stdout, nil, _IOLBF, 0)   // a stage line reaches the app as it is written
        do {
            let named = CommandLine.arguments.dropFirst(2).first.map { $0.split(separator: ",").compactMap { Family(rawValue: String($0)) } } ?? []
            let report = try Diagnostic.run(families: named.isEmpty ? Family.allCases : named) { stage in print("stage " + stage.key) }
            let json = try report.json(pretty: false)
            print("report " + String(decoding: json, as: UTF8.self))
            exit(0)
        } catch {
            let e = EngineError(error)
            print("error \(e.code) " + e.description.replacingOccurrences(of: "\n", with: " "))
            exit(1)
        }
    }

    /// One line of the child, read by the app.
    enum Line {
        case stage(Diagnostic.Stage)
        case report(Diagnostic.Report)
        case failure(Problem)

        init?(_ line: String) {
            let (word, rest) = line.firstIndex(of: " ").map { (line[..<$0], String(line[line.index(after: $0)...])) }
                ?? (line[...], "")
            switch word {
            case "stage":
                guard let stage = Diagnostic.Stage.allCases.first(where: { $0.key == rest }) else { return nil }
                self = .stage(stage)
            case "report":
                guard let report = try? JSONDecoder().decode(Diagnostic.Report.self, from: Data(rest.utf8)) else { return nil }
                self = .report(report)
            case "error":
                // The one refusal the app says in its own words: another process renders.
                let code = rest.prefix { $0 != " " }
                let detail = String(rest.dropFirst(code.count)).trimmingCharacters(in: .whitespaces)
                self = .failure(code == EngineError.renderAlreadyRunning.code
                    ? Problem(EngineError.renderAlreadyRunning)
                    : Problem(EngineError.internalFailure(component: "diagnostic", detail: detail)))
            default: return nil
            }
        }
    }
}
