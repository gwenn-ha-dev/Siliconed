// **`AppState`: where the library is** — moving it to another disk, opening one there, bringing it
// back to this Mac; and the copy renders read while it is elsewhere (`WorkingCopy`). The state
// itself is in `AppState.swift`.
//
// - **The library elsewhere is the library**: its disk unplugged, nothing renders, installs or
//   imports (`libraryBlocker`, `managementPossible`) — never this Mac's in its place; going back to
//   this Mac's is a gesture of the settings (`useLocalLibrary`).
// - **The copy starts as soon as a model is chosen** (`warm`): at launch for the model shown, at each
//   choice, for every model of an exploration as it enters the queue — while the prompt is typed. A
//   render prepares again before reading (`prepareCopy`, nothing to do when it is done), so it never
//   reads the other disk. It is removed when the app quits (`App.swift`).

import AppKit
import Siliconed
import SwiftUI
import Synchronization

/// A copy to this Mac under way: what is copied, and how far.
struct CopyProgress: Equatable {
    var title: String
    var fraction: Double
}

extension AppState {
    // MARK: The working copy

    /// The copy a library needs, if it needs one; `nil` when the folder cannot be made — the render
    /// then reads the library where it is, as before.
    static func makeWorkingCopy(_ library: Library) -> WorkingCopy? {
        guard WorkingCopy.isNeeded(for: library) else { return nil }
        return try? WorkingCopy(source: library)
    }

    /// Why nothing renders: the library's disk is not there — the sentence alone; what to do is said
    /// where the gesture is (the canvas, the remote control's hint).
    var libraryBlocker: String? {
        libraryReachable ? nil : Problem(EngineError.libraryAbsent(path: library.root.path, volume: library.volumeName)).title
    }

    /// **Starts copying what the chosen model reads** — or `models` (an exploration's), with their
    /// LoRAs. A new choice stops the previous copy; a failure other than a stop is said.
    func warm(_ models: [String]? = nil, loras: [String]? = nil) {
        guard workingCopy != nil, libraryReachable, forgeInProgress == nil else { return }
        let models = models ?? (readySet.contains(identifier) ? [identifier] : [])
        guard !models.isEmpty else { return }
        let loras = loras ?? self.loras.map(\.path).filter { !$0.isEmpty }
        warmCancellation?.cancel()
        let cancellation = Cancellation()
        warmCancellation = cancellation
        Task {
            do { try await prepareCopy(models: models, loras: loras, cancellation: cancellation) } catch {
                let e = EngineError(error)
                if e != .cancelled, currentJob == nil { problem = Problem(e) }
            }
        }
    }

    /// **Brings the copy up to date for `models`** off the main thread, its progress in `copying`.
    func prepareCopy(models: [String], loras: [String], cancellation: Cancellation?) async throws {
        guard let copy = workingCopy else { return }
        let names = models.map { id in cards.first { $0.id == id }?.name ?? id }.formatted(.list(type: .and))
        copiesInFlight += 1
        defer {
            copiesInFlight -= 1
            if copiesInFlight == 0 { copying = nil }
        }
        let shown = Mutex(-1)
        try await Task.detached(priority: .userInitiated) { [weak self] in
            try copy.prepare(models: models, loras: loras, cancellation: cancellation) { p in
                let percent = p.total > 0 ? p.done * 100 / p.total : 0
                guard shown.withLock({ old in defer { old = percent }; return old != percent }) else { return }
                let title = String(localized: "Copying \(names) to this Mac")
                Task { @MainActor in self?.copying = CopyProgress(title: title, fraction: Double(percent) / 100) }
            }
        }.value
    }

    /// Removes the copy — when the app quits.
    func removeWorkingCopy() { workingCopy?.remove() }

    /// **A disk came or went**: the library may be back, or gone — the catalog and the copy follow.
    func watchVolumes() {
        let center = NSWorkspace.shared.notificationCenter
        for name in [NSWorkspace.didMountNotification, NSWorkspace.didUnmountNotification, NSWorkspace.didRenameVolumeNotification] {
            volumeObservers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.volumesChanged() }
            })
        }
    }

    /// At exit: the observers `watchVolumes` added.
    func stopWatchingVolumes() {
        for observer in volumeObservers { NSWorkspace.shared.notificationCenter.removeObserver(observer) }
        volumeObservers = []
    }

    private func volumesChanged() {
        let reachable = library.isReachable
        guard reachable != libraryReachable else { return }
        libraryReachable = reachable
        if !reachable { warmCancellation?.cancel() }
        if reachable, workingCopy == nil { workingCopy = Self.makeWorkingCopy(library) }
        refreshCatalog()
        // Absent, the library showed nothing installed and the form fell back on the default model:
        // back, the first model it renders is chosen again.
        if reachable, !readySet.contains(identifier), let first = visibleReady.first {
            identifier = first.id                   // which starts its copy (`modelChanged`)
        } else if reachable {
            warm()
        }
    }

    // MARK: Moving, opening

    /// The library is on another disk than this Mac's own space.
    var libraryIsElsewhere: Bool { library.isElsewhere }

    /// « Move Library… »: a folder on another disk (or a share), then the move, verified.
    func chooseLibraryDestination() {
        guard managementPossible else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = String(localized: "Move Here")
        panel.message = String(localized: "Where the models go — an external disk, for instance. A “Siliconed” folder is made in it.")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        moveLibrary(to: LibraryMove.destination(in: url))
    }

    /// « Move Back to This Mac »: the library returns to the app's own space.
    func moveLibraryHome() { moveLibrary(to: library.local) }

    /// **Moves the library**, verified file by file, the old place emptied last (`LibraryMove.move`).
    func moveLibrary(to destination: URL) {
        guard managementPossible else { return }
        let source = library
        let place = destination.path == source.local.path ? String(localized: "this Mac")
            : Library(root: destination, local: source.local).volumeName.map { "“\($0)”" } ?? Library.withoutHome(destination.path)
        workingCopy?.remove()
        launchForge(String(localized: "Moving the library to \(place)"),
                    stopped: String(localized: "Stopped — the library stays where it was, untouched. The files already copied are kept for the next attempt.")) { _, cancellation, journal in
            let said = Mutex(-1)
            return try LibraryMove.move(source, to: destination, cancellation: cancellation) { p in
                let percent = p.total > 0 ? p.done * 100 / p.total : 0
                guard said.withLock({ old in defer { old = percent }; return old != percent }) else { return }
                journal("⇢ \(percent) \(p.file)")
            }
        } end: { [weak self] (moved: Library) in
            guard let self else { return }
            switchLibrary(to: moved)
            flash(String(localized: "Library moved to \(place)"))
        }
    }

    /// « Open Library… »: a folder that already holds a library (another disk's, another Mac's).
    func chooseLibraryToOpen() {
        guard forgeInProgress == nil, !busy, !diagnosticRunning else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = String(localized: "Open")
        panel.message = String(localized: "A folder holding a Siliconed library. Nothing is moved: this library stays where it is.")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        openLibrary(at: url)
    }

    /// The folder chosen, or named by `silicontrol library open`. The problem, if the folder holds no library — also said in the window.
    @discardableResult
    func openLibrary(at url: URL) -> Problem? {
        guard forgeInProgress == nil, !busy, !diagnosticRunning else { return nil }
        do { switchLibrary(to: try library.opening(url)) } catch { problem = Problem(error); return problem }
        return nil
    }

    /// « Use This Mac’s Library »: the disk is absent, or the user wants this Mac's — nothing moved.
    @discardableResult
    func useLocalLibrary() -> Problem? {
        guard forgeInProgress == nil, !busy, !diagnosticRunning else { return nil }
        do { switchLibrary(to: try library.useLocal()) } catch { problem = Problem(error); return problem }
        return nil
    }

    /// **The app reads another library from now on**: its catalog, its copy; the form keeps what it can.
    private func switchLibrary(to new: Library) {
        warmCancellation?.cancel()
        workingCopy?.remove()
        library = new
        if !new.isElsewhere { try? FileManager.default.createDirectory(atPath: new.map(""), withIntermediateDirectories: true) }
        workingCopy = Self.makeWorkingCopy(new)
        libraryReachable = new.isReachable
        refreshCatalog()
        let valid = Set(compatibleLoras.map(\.path))
        loras.removeAll { !$0.path.isEmpty && !valid.contains($0.path) }
        if !readySet.contains(identifier), let first = visibleReady.first { identifier = first.id }
        warm()
    }
}
