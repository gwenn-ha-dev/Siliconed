// **`AppState`: where the library is** — moving it to another disk, opening one there, bringing it
// back to this Mac. The state itself is in `AppState.swift`.
//
// - **The library elsewhere is the library**: its disk unplugged, nothing renders, installs or
//   imports (`libraryBlocker`, `managementPossible`) — never this Mac's in its place; going back to
//   this Mac's is a gesture of the settings (`useLocalLibrary`).

import AppKit
import Siliconed
import SwiftUI
import Synchronization

extension AppState {
    /// Why nothing renders: the library's disk is not there — the sentence alone; what to do is said
    /// where the gesture is (the canvas, the remote control's hint).
    var libraryBlocker: String? {
        libraryReachable ? nil : Problem(EngineError.libraryAbsent(path: library.root.path, volume: library.volumeName)).title
    }

    /// **A disk came or went**: the library may be back, or gone — the catalog follows.
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
        refreshCatalog()
        // Absent, the library showed nothing installed and the form fell back on the default model:
        // back, the first model it renders is chosen again.
        if reachable, !readySet.contains(identifier), let first = visibleReady.first {
            identifier = first.id
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
        launchLibraryJob(String(localized: "Moving the library to \(place)"),
                    stopped: String(localized: "Stopped — the library stays where it was, untouched. The files already copied are kept for the next attempt.")) { _, cancellation, journal, _ in
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
        guard libraryJob == nil, !busy else { return }
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
        guard libraryJob == nil, !busy else { return nil }
        do { switchLibrary(to: try library.opening(url)) } catch { problem = Problem(error); return problem }
        return nil
    }

    /// « Use This Mac’s Library »: the disk is absent, or the user wants this Mac's — nothing moved.
    @discardableResult
    func useLocalLibrary() -> Problem? {
        guard libraryJob == nil, !busy else { return nil }
        do { switchLibrary(to: try library.useLocal()) } catch { problem = Problem(error); return problem }
        return nil
    }

    /// **The app reads another library from now on**: its catalog; the form keeps what it can.
    private func switchLibrary(to new: Library) {
        library = new
        if !new.isElsewhere { try? FileManager.default.createDirectory(atPath: new.map(""), withIntermediateDirectories: true) }
        libraryReachable = new.isReachable
        refreshCatalog()
        let valid = Set(compatibleLoras.map(\.path))
        loras.removeAll { !$0.path.isEmpty && !valid.contains($0.path) }
        if !readySet.contains(identifier), let first = visibleReady.first { identifier = first.id }
    }
}
