import Foundation
import Synchronization

// **The preflights: what a render checks before its first computation, in this order.**
//
//     1. the model is installed          its maps and published files are all there (`ModelCard.missing`)
//     2. its license is accepted         `<root>/accepted-licenses.json`, written by the app
//     3. the format                      side ≥ 512, multiple of 16, and the family's own rules:
//                                        Qwen-Image-2.1 ×32, Anima ≤ 2048 px a side (`Format.check(…, family:)`);
//                                        no area ceiling: the memory (4) decides
//     4. the memory                      the request's computed floor ≤ B, this Mac's budget (`MemoryBudget`);
//                                        a floor that cannot be computed is refused too, and a format whose
//                                        widest tensor the engine cannot index (`MemoryNeed.limit`)
//     5. no other render                 `RenderLock.machine`, one for the machine, held for the whole render
//
// The first that fails throws its `EngineError`. 1, 2 and 5 need the model's library: a
// hand-composed chain (`Model(card:chain:)`, no library) has neither registry nor lock, and its
// files are checked by its own constructors. The request's own refusals (empty prompt, steps,
// strength, references, solver, detail) come before these, in `Engine.execute`: a request the model
// cannot render is refused for what it is, not for its memory.
//
// **The memory is B, fixed by this Mac's RAM and nothing else** (`MemoryBudget.ceiling(physical:)`,
// ¾ · RAM − 2 GiB): no reading of the kernel's state decides — the same render on the same Mac gets
// the same answer, always. Against it, the **floor**: the need of the most economical plan,
// computed from the request's shapes (`MemoryNeed`, `ModelCard.memoryNeed`), plus what the process
// already holds (the app's interface and history: B covers the whole process) — what remains above it
// buys time (`MemoryPlan`), never another image. The app asks the same function ahead, to grey out
// what this Mac cannot do (`ModelCard.checkMemory`).
//
// The plan is then chosen in the same B (`MemoryPlan.choose`): the fastest whose need ≤ B.

package enum Preflight {

    /// **The checks 1 to 4, pure** — what the test target drives with an injected machine and catalog.
    /// `memory`: the request's own inputs (prompt, references' sizes); without it, the references
    /// at the output's size and a typical prompt. `checksMemory`: false for a hand-composed chain,
    /// whose modules are its caller's — the need describes the published families' constructors only.
    package static func check(card: ModelCard, width: Int, height: Int, references: Int = 0, installed: Bool,
                              licenseAccepted: Bool, ceiling: Int,
                              memory: MemoryNeed.Request? = nil, checksMemory: Bool = true) throws(EngineError) {
        guard installed else { throw .modelNotInstalled(model: card.id) }
        guard licenseAccepted else { throw .licenseNotAccepted(model: card.id) }
        try Format.check(width: width, height: height, family: card.family)
        guard checksMemory else { return }
        if let memory {
            try card.checkMemory(memory, ceiling: ceiling)
        } else {
            try card.checkMemory(width: width, height: height, references: references, ceiling: ceiling)
        }
    }

    /// **All five, for a model about to render**: the lock comes back held, released when dropped.
    package static func run(_ model: Model, width: Int, height: Int, references: Int = 0,
                            ceiling: Int = MemoryBudget.machineCeiling(),
                            memory: MemoryNeed.Request? = nil) throws(EngineError) -> RenderLock? {
        let library = model.library
        try check(card: model.card, width: width, height: height, references: references,
                  installed: library.map { model.card.missing(in: $0) == nil } ?? true,
                  licenseAccepted: library.map { Licenses.implicit || $0.isLicenseAccepted(model.card) } ?? true,
                  ceiling: ceiling, memory: memory, checksMemory: library != nil)
        return library == nil ? nil : try RenderLock.acquire(RenderLock.machine)
    }
}

// ── 4 · the memory a render needs ────────────────────────────────────────────────────────

extension ModelCard {
    /// **The request the need is computed on** (`MemoryNeed.Request`): the format, each reference at
    /// its size, the prompt's tokens (or `MemoryNeed.Request.typicalPromptTokens` when no prompt is known
    /// yet — the app's display), the LoRA ranks, and **the process's footprint now** (`base`: what the
    /// app or the command line already holds — B covers the whole process; a test passes its own).
    package func memoryRequest(width: Int, height: Int, references: [(width: Int, height: Int)] = [],
                               promptTokens: Int? = nil, loraRanks: [Int] = [], base: Int? = nil,
                               library: Library = .processWide) -> MemoryNeed.Request {
        let tokens = promptTokens ?? MemoryNeed.Request.promptTokens(text: MemoryNeed.Request.typicalPromptTokens,
                                                                     family: family, references: references.count)
        var request = MemoryNeed.Request(card: self, width: width, height: height, references: references,
                                         promptTokens: tokens, loraRanks: loraRanks)
        request.base = base ?? OperatorFootprint.processBase()
        request.library = library
        return request
    }

    /// **Refuses a render whose floor exceeds B** — the one memory check, the engine's preflight and the
    /// app's greying out alike. `ceiling`: B, this Mac's by default (`MemoryBudget.machineCeiling()`,
    /// `SILICONED_MEMORY_BUDGET_GB` included — no reading of the kernel). `references`: the edit's
    /// images, named subjects' included (0: a generation), taken at the output's size.
    package func checkMemory(width: Int, height: Int, references: Int = 0,
                            ceiling: Int = MemoryBudget.machineCeiling()) throws(EngineError) {
        try checkMemory(memoryRequest(width: width, height: height,
                                      references: Array(repeating: (width, height), count: references)), ceiling: ceiling)
    }

    /// **The same check on a whole request** — what the engine asks, with the prompt, the references'
    /// own sizes and the LoRA ranks: the lean plan's need against B, nothing else. **A need that cannot
    /// be computed is a refusal** (`memoryNeedUnknown`: maps missing, or a header that does not parse),
    /// never a need of 0 that every B would hold.
    package func checkMemory(_ request: MemoryNeed.Request, ceiling: Int = MemoryBudget.machineCeiling()) throws(EngineError) {
        guard let need = MemoryNeed.of(request.lean) else { throw .memoryNeedUnknown(model: id) }
        // The engine's own limit comes before B: no Mac's RAM raises it (`MemoryNeed.limit`).
        if let area = need.addressableArea(width: request.width, height: request.height) {
            throw .formatRefused(width: request.width, height: request.height, reason: .tooLarge, bound: area)
        }
        guard need.peak <= ceiling else { throw .insufficientMemory(needed: need.peak, budget: ceiling) }
    }

    /// **The memory a render of this model needs at `width × height`**, in bytes: its floor — the peak
    /// of the most economical plan (`MemoryNeed.Request.lean`), computed from the shapes
    /// (`MemoryNeed.of`), with what this process holds now. Above the floor the plan buys time
    /// (`MemoryPlan.choose`); at the floor the render is slower, never another image. `nil` when it
    /// cannot be computed: the model's maps are not installed, or their headers do not parse.
    package func memoryNeed(width: Int, height: Int, references: Int = 0) -> Int? {
        memoryNeed(memoryRequest(width: width, height: height, references: Array(repeating: (width, height), count: references)))
    }

    package func memoryNeed(_ request: MemoryNeed.Request) -> Int? { MemoryNeed.of(request.lean)?.peak }
}

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

// ── 5 · one render on the machine ────────────────────────────────────────────────────────

/// **"Two engines do not fit in 16 GB", enforced across processes.** Within a process the engine
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
    /// It used to be `<root>/render.lock`, one per library: two processes on two libraries (the app
    /// and a CLI given `--library`, or two users) rendered side by side, while what the lock protects
    /// — the memory — is the machine's, not the library's nor the user's (fast user switching keeps
    /// both sessions' processes running). `/private/tmp` is the one directory every user can create in
    /// and nobody else can delete from (sticky bit), and it is emptied at boot, where no lock can
    /// outlive its holder anyway. Another user's file is opened read-only: `flock` does not need to write.
    package static let machine = "/private/tmp/siliconed-render.lock"

    /// The lock, held; `nil` if the file can neither be created nor opened, or the system refuses `flock` (a machine without a
    /// writable `/tmp` renders unguarded, as before the lock existed — and the second case says so as a warning).
    package static func acquire(_ path: String) throws(EngineError) -> RenderLock? {
        // 0o644 and read-only: the file another user created stays theirs, and still locks for us.
        let fd = open(path, O_RDONLY | O_CREAT, 0o644)
        guard fd >= 0 else { return nil }
        var locked = flock(fd, LOCK_EX | LOCK_NB)
        while locked != 0, errno == EINTR { locked = flock(fd, LOCK_EX | LOCK_NB) }
        guard locked == 0 else {
            let busy = errno == EWOULDBLOCK
            close(fd)
            if busy { throw .renderAlreadyRunning }
            // The wall « one render on the machine » stands open for this render: said, not silent.
            Warnings.emit("⚠ the machine's render lock could not be taken (errno \(errno)): this render runs unguarded")
            return nil
        }
        return RenderLock(fd)
    }

    deinit {
        flock(descriptor, LOCK_UN)
        close(descriptor)
    }
}
