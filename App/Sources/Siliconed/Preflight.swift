import Foundation
import Synchronization

// **The preflights: what a render checks before its first computation, in this order.**
//
//     1. the model is installed          its maps and published files are all there: checked by its construction
//                                        (`Model.named`), the only way a model gets a library
//     2. its license is accepted         `<local>/accepted-licenses.json`, written by the app
//     3. the format                      side ≥ 512, multiple of 16, and the family's own rules (its traits'
//                                        multiple and longest side, `Format.check(…, family:)`)
//     4. no other render                 `RenderLock.machine`, one for the machine, held for the whole render
//     5. its shared files hashed once    those an older app left (`Installer.checkShared`): after the refusals that cost
//                                        nothing, under the lock; one that is not the pinned bytes is removed, said
//                                        (`sharedFileRemovedNotice`) and refused (`corruptMap`)
//
// The first that fails throws its `EngineError`. The memory is not among them: the engine plans the render under B before
// its first computation, and refuses it there when no plan fits.

package enum Preflight {

    /// **The checks 2 and 3, pure** — what the test target drives with an injected catalog.
    package static func check(card: ModelCard, width: Int, height: Int, licenseAccepted: Bool) throws(EngineError) {
        guard licenseAccepted else { throw .licenseNotAccepted(model: card.id) }
        try Format.check(width: width, height: height, family: card.family)
    }

    /// **All five, for a model about to render**: the lock comes back held, released when dropped; `warn` hears why the
    /// render runs unguarded when the system refuses it, and which shared file was removed. A hand-composed chain (no
    /// library) has neither registry nor lock: its files are checked by its own constructors.
    package static func run(_ model: Model, width: Int, height: Int, cancellation: Cancellation? = nil,
                            warn: (String) -> Void) throws(EngineError) -> RenderLock? {
        guard let library = model.library else {
            try check(card: model.card, width: width, height: height, licenseAccepted: true)
            return nil
        }
        try check(card: model.card, width: width, height: height,
                  licenseAccepted: Licenses.implicit || library.isLicenseAccepted(model.card))
        let lock = try RenderLock.acquire(RenderLock.machine, warn: warn)
        var installer = Installer(library: library)
        installer.journal = { _ in }
        installer.cancellation = cancellation
        do {
            try installer.checkShared(model.card.family)
        } catch {
            if case .corruptMap(let file?) = error { warn("\(sharedFileRemovedNotice): \(file)") }
            throw error
        }
        return lock
    }
}


/// What a render says of a shared file removed before it (`Preflight.run`), followed by the file's path: the app says it
/// in its own words (`AppState.journalLine`).
package let sharedFileRemovedNotice = "A shared file of the model was not the published bytes: it was removed, and installing the model again downloads it alone"


// ── 2 · the accepted licenses ────────────────────────────────────────────────────────────

/// The developer CLI accepts the licenses implicitly (it says so when it renders); an app never does.
package enum Licenses {
    private static let implicitFlag = Atomic<Bool>(false)
    package static var implicit: Bool {
        get { implicitFlag.load(ordering: .relaxed) }
        set { implicitFlag.store(newValue, ordering: .relaxed) }
    }
}

extension Library {
    /// `<local>/accepted-licenses.json` — `{ model identifier: license text accepted }`. The text is
    /// kept so that a license that changes asks again. This Mac's user accepted them: they stay on
    /// the Mac when the library moves.
    package var acceptedLicensesFile: String { local.appendingPathComponent("accepted-licenses.json").path }

    /// The models whose license was accepted, with the text accepted. Empty if the file is absent
    /// or unreadable — nothing is accepted by default.
    package func acceptedLicenses() -> [String: String] {
        guard let data = FileManager.default.contents(atPath: acceptedLicensesFile),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: String] else { return [:] }
        return object
    }

    /// Accepted, and for the license the card states today.
    package func isLicenseAccepted(_ card: ModelCard) -> Bool {
        acceptedLicenses()[card.id] == card.license.text
    }

    /// **Records that the user accepted `card`'s license** — what the app writes after showing it.
    package func acceptLicense(_ card: ModelCard) throws(EngineError) {
        var accepted = acceptedLicenses()
        accepted[card.id] = card.license.text
        try writeAcceptedLicenses(accepted)
    }

    private func writeAcceptedLicenses(_ accepted: [String: String]) throws(EngineError) {
        try EngineError.boundary {
            try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: accepted, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: acceptedLicensesFile), options: .atomic)
        }
    }
}

// ── 4 · one render on the machine ────────────────────────────────────────────────────────

/// **One render on the machine, enforced across processes**: each plans under the whole budget, so two at once would
/// overrun it. Within a process the engine
/// queue already serializes renders (`Engine.file`); between processes — the app and the CLI, two
/// libraries, two users — an advisory `flock` on `RenderLock.machine`, taken without waiting: the
/// second one is refused (`renderAlreadyRunning`) instead of swapping both. The CLI also asks the
/// app whether its queue is busy; this lock is what the engine itself holds.
///
/// The kernel releases it when the process dies: a crash never leaves the machine locked.
package final class RenderLock {
    private let descriptor: Int32

    private init(_ descriptor: Int32) { self.descriptor = descriptor }

    /// **`/private/tmp/siliconed-render.lock` — one lock for the machine, every user.**
    ///
    /// Not `<root>/render.lock`, one per library: two processes on two libraries (the app
    /// and a CLI given `--library`, or two users) would render side by side, while what the lock protects
    /// — the memory — is the machine's, not the library's nor the user's (fast user switching keeps
    /// both sessions' processes running). `/private/tmp` is the one directory every user can create in
    /// and nobody else can delete from (sticky bit), and it is emptied at boot, where no lock can
    /// outlive its holder anyway. Another user's file is opened read-only: `flock` does not need to write.
    package static let machine = "/private/tmp/siliconed-render.lock"

    /// The lock, held; `nil` if the file can neither be created nor opened, or the system refuses `flock` (a machine without a
    /// writable `/tmp` renders unguarded, as before the lock existed — and the second case says so to `warn`, the
    /// render's warning).
    package static func acquire(_ path: String, warn: (String) -> Void) throws(EngineError) -> RenderLock? {
        // 0o644 and read-only: the file another user created stays theirs, and still locks for us.
        let fd = open(path, O_RDONLY | O_CREAT, 0o644)
        guard fd >= 0 else { return nil }
        var locked = flock(fd, LOCK_EX | LOCK_NB)
        while locked != 0, errno == EINTR { locked = flock(fd, LOCK_EX | LOCK_NB) }
        guard locked == 0 else {
            let error = errno
            close(fd)
            if error == EWOULDBLOCK { throw .renderAlreadyRunning }
            // The wall « one render on the machine » stands open for this render: said, not silent.
            warn("The machine's render lock could not be taken (errno \(error)): this render runs unguarded")
            return nil
        }
        return RenderLock(fd)
    }

    deinit {
        flock(descriptor, LOCK_UN)
        close(descriptor)
    }
}
