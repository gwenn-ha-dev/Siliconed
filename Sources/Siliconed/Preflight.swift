import Foundation
import Synchronization

// **The preflights: what a render checks before its first computation, in this order.**
//
//     1. the model is installed          its maps and published files are all there (`ModelCard.missing`)
//     2. its license is accepted         `<racine>/accepted-licenses.json`, written by the app
//     3. the format                      side ≥ 512, multiple of 16, area ≤ 1024×1536 (`Format.check`)
//     4. the memory                      the model's floor at this format ≤ the budget read now (`MemoryBudget`)
//     5. no other render                 `RenderLock.machine`, one for the machine, held for the whole render
//
// The first that fails throws its `EngineError`. 1, 2 and 5 need the model's library: a
// hand-composed chain (`Model(card:chain:)`, no library) has neither registry nor lock, and its
// files are checked by its own constructors. Then the request's own refusals follow (empty prompt,
// steps, references, strength) in `Engine.execute`, still before any computation.
//
// **The memory is the machine's state at the render's launch, not its RAM, nor its "free" memory**
// the pages the kernel can hand over without swapping anyone, plus the process's
// own footprint, minus a reserve (`MemoryBudget`). Against it, the **floor**: the anonymous memory of
// the most economical plan at that format (`ModelCard.memoryNeed`) — what remains above it buys time
// (`MemoryPlan`), never another image. The app asks the same function (`ModelCard.checkMemory`).

package enum Preflight {

    /// **The checks 1 to 4, pure** — what the test target drives with an injected machine and catalog.
    package static func check(card: ModelCard, width: Int, height: Int, installed: Bool,
                              licenseAccepted: Bool, budget: MemoryBudget) throws(EngineError) {
        guard installed else { throw .modelNotInstalled(model: card.id) }
        guard licenseAccepted else { throw .licenseNotAccepted(model: card.id) }
        try Format.check(width: width, height: height)
        try card.checkMemory(width: width, height: height, budget: budget)
    }

    /// **All five, for a model about to render**: the lock comes back held, released when dropped.
    package static func run(_ model: Model, width: Int, height: Int,
                            budget: MemoryBudget = .current()) throws(EngineError) -> RenderLock? {
        let library = model.library
        try check(card: model.card, width: width, height: height,
                  installed: library.map { model.card.missing(in: $0) == nil } ?? true,
                  licenseAccepted: library.map { Licenses.implicit || $0.isLicenseAccepted(model.card) } ?? true,
                  budget: budget)
        return library == nil ? nil : try RenderLock.acquire(RenderLock.machine)
    }
}

// ── 4 · the memory a model declares ──────────────────────────────────────────────────────

/// **One measured peak**: the `phys_footprint` maximum of a whole render at a format, and the
/// measurement it comes from.
package struct MeasuredPeak: Sendable {
    /// What the figure is worth as a **floor** — the anonymous memory of the most economical plan.
    package enum Bound: Sendable {
        /// Measured on the economical plan: no staging buffer anywhere, the map through the cache, the smallest
        /// bands. **No render takes that plan any more**: the lean plan streams the whole map with
        /// two buffers, which peaks where the default plan does during the denoising
        /// (Z-Image 1024² 2 933 MB, 1024×1536 3 771 MB, against floors of 2 214 and 3 200). The
        /// refusal stays judged here on purpose — see `memoryNeed`.
        case floor
        /// A peak measured at the default plan (streamed tail, staging buffers): **above** the floor,
        /// by what the plan's buffers cost. A floor nobody has measured yet (lot 5(e)).
        case majorant
    }
    package let width: Int, height: Int
    package let bytes: Int
    package let bound: Bound
    package let source: String
    package var area: Int { width * height }
}

extension ModelCard {
    /// **The peaks measured for this model's family** (`phys_footprint`, M1 Pro 16 GB), whole render.
    /// Only measurements — a format nobody measured is not filled by a guess (see `memoryNeed`).
    package var measuredPeaks: [MeasuredPeak] {
        func p(_ w: Int, _ h: Int, _ gb: Double, _ source: String) -> MeasuredPeak {
            MeasuredPeak(width: w, height: h, bytes: Int(gb * 1e9), bound: .majorant, source: source)
        }
        func floor(_ w: Int, _ h: Int, _ gb: Double, _ source: String) -> MeasuredPeak {
            MeasuredPeak(width: w, height: h, bytes: Int(gb * 1e9), bound: .floor, source: source)
        }
        switch family {
        case .zImage:
            // The SDPA head split adds its spill and scratch to the denoising: +40 MB at 512²,
            // +114 MB at 1024²; at 1024×1536 the banded decoding's peak stays the larger, and there the
            // decoding holds the floor (the denoising without the tail is below it).
            return [p(512, 512, 1.695, "M120: seed 42, denoising peak 1 695 MB, tail streamed"),
                    // The banded decoder: the 832×1216 measured at 4.8 GB went with the one-graph decoder.
                    p(1024, 1024, 2.936, "M120, denoising with the tail (decoding 2.36)"),
                    // The former 3.358 GB (its decoding) predates the head split: the denoising with it now peaks above it.
                    p(1024, 1536, 3.774, "M123, denoising with the tail, sampled at 10 ms (decoding 3.34)"),
                    // The economical plan (no tail, no encoder buffer, bands of 2^20): same pixels, 3–7 s more. The
                    // lean plan peaks two buffers above at the denoising (`Bound.floor`, `memoryNeed`).
                    floor(1024, 1024, 2.214, "M133, denoising (decoding 2 207 MB)"),
                    floor(832, 1216, 2.077, "M133, decoding (default plan 2 764 MB)"),
                    floor(1024, 1536, 3.200, "M133, decoding (denoising 3 051 MB)")]
        case .anima:
            return [p(832, 1216, 5.3, "M67"), p(1024, 1024, 5.2, "M55"), p(1024, 1536, 7.6, "M74")]
        case .krea2:
            return [p(512, 512, 0.984, "M64"), p(832, 1216, 5.2, "M67"), p(1024, 1024, 5.1, "M64"),
                    p(1024, 1536, 7.8, "M74")]
        case .klein4b:
            return [p(1024, 1024, 5.1, "M86")]
        case .ernie:
            return [p(1024, 1024, 4.9, "M88")]
        case .qwenImage21:
            // Measured with binary dde99e2, after the decoder's banded tail: the earlier 4.68 GB came from the decoder
            // before its tail was banded. The 1024² tier is an edit's: 3 references peak 0.9 GB above the generation.
            return [p(1248, 832, 4.412, "M107, edit with 2 references (1 reference: 4 141 MB)"),
                    p(1024, 1024, 4.510, "M123, edit with 3 references, sampled at 10 ms (1 reference 4.08, generation 3.63)"),
                    p(1024, 1536, 4.74, "M96, worst case: 3 references")]
        }
    }

    /// **Refuses a render whose floor exceeds the budget** — the one memory check, the engine's
    /// preflight and the app's "Generate" button alike. `budget` is read now by default
    /// (`MemoryBudget.current()`, `SILICONED_MEMORY_AVAILABLE_GB` included).
    public func checkMemory(width: Int, height: Int, budget: MemoryBudget = .current()) throws(EngineError) {
        let needed = memoryNeed(width: width, height: height)
        guard needed <= budget.available else {
            throw .insufficientMemory(needed: needed, available: budget.available)
        }
    }

    /// **The memory a render of this model needs at `width × height`**, in bytes: its floor — the
    /// peak of the most economical plan where it was measured (`MeasuredPeak.Bound.floor`), otherwise
    /// a peak measured at the default plan, which lies above it. Above the floor the plan buys time
    /// (`MemoryPlan`); at the floor the render is slower, never another image (measured: same pixels).
    ///
    /// The format falls in the smallest measured area that holds it, and takes the largest figure
    /// measured up to that area (so the need never decreases with the area: Anima's 832×1216 peaks
    /// above its 1024²). A format beyond the family's measurements takes the largest peak any family
    /// measured at or beyond it — klein and ERNIE at 1024×1536 take Krea 2's 7.8 GB. The references of
    /// an edit are not a separate axis: Qwen-Image-2.1's worst case with 3 references is in the table.
    ///
    /// **Below the lean plan's own peak, by its two buffers** (Z-Image: 2 × 362 MB, the whole map
    /// streamed). A render accepted within 0.72 GB of the floor may take that much of the
    /// reserve during its denoising. Kept so: the floor with the buffers (2.93 GB at 1024²) would refuse
    /// two measured desktops again (2.63 and 2.45 GB available; 1216×832 would ask ~2.77 GB,
    /// the economical plan's 2 043 MB of denoising plus the buffers), which rendered without swap; and what swapped under a ballast was the map through the
    /// cache, which the buffers avoid, not our anonymous memory (at 1.30 GB of footprint). The
    /// swap judge of that band decides it.
    public func memoryNeed(width: Int, height: Int) -> Int {
        need(width: width, height: height, floors: true)
    }

    /// **The default plan's peak at `width × height`**: what the render needs to take every buffer
    /// that buys time. The decoders take their measured band only above it (`MemoryPlan.decoderBand`).
    package func memoryComfort(width: Int, height: Int) -> Int {
        need(width: width, height: height, floors: false)
    }

    private func need(width: Int, height: Int, floors: Bool) -> Int {
        let area = width * height
        // One figure per measured format: its floor where there is one and it is asked for, otherwise
        // its default plan's peak. A format with only a floor is no tier for the default plan's peak.
        func figures(_ peaks: [MeasuredPeak]) -> [(area: Int, bytes: Int)] {
            Dictionary(grouping: peaks, by: \.area).compactMap { area, peaks in
                let floor = floors ? peaks.filter { $0.bound == .floor }.map(\.bytes).min() : nil
                return (floor ?? peaks.filter { $0.bound == .majorant }.map(\.bytes).max()).map { (area, $0) }
            }
        }
        let own = figures(measuredPeaks)
        if let tier = own.map(\.area).filter({ $0 >= area }).min() {
            return own.filter { $0.area <= tier }.map(\.bytes).max()!
        }
        let beyond = ModelCard.allCards.flatMap { figures($0.measuredPeaks) }.filter { $0.area >= area }.map(\.bytes)
        return max(beyond.max() ?? 0, own.map(\.bytes).max() ?? 0)
    }
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
    /// `<racine>/accepted-licenses.json` — `{ model identifier: license text accepted }`. The text is
    /// kept so that a license that changes asks again.
    package var acceptedLicensesFile: String { root.appendingPathComponent("accepted-licenses.json").path }

    /// The models whose license was accepted, with the text accepted. Empty if the file is absent
    /// or unreadable — nothing is accepted by default.
    public func acceptedLicenses() -> [String: String] {
        guard let data = FileManager.default.contents(atPath: acceptedLicensesFile),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: String] else { return [:] }
        return object
    }

    /// Accepted, and for the license the card states today.
    public func isLicenseAccepted(_ card: ModelCard) -> Bool {
        acceptedLicenses()[card.id] == card.license.text
    }

    /// **Records that the user accepted `card`'s license** — what the app writes after showing it.
    public func acceptLicense(_ card: ModelCard) throws(EngineError) {
        var accepted = acceptedLicenses()
        accepted[card.id] = card.license.text
        try writeAcceptedLicenses(accepted)
    }

    /// Withdraws an acceptance: the next render of `card` is refused until it is accepted again.
    package func revokeLicense(_ card: ModelCard) throws(EngineError) {
        var accepted = acceptedLicenses()
        guard accepted.removeValue(forKey: card.id) != nil else { return }
        try writeAcceptedLicenses(accepted)
    }

    private func writeAcceptedLicenses(_ accepted: [String: String]) throws(EngineError) {
        try EngineError.boundary {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
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
/// app whether its queue is busy (`appBusy`); this lock is what the engine itself holds.
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

    /// The lock, held; `nil` if the file can neither be created nor opened (a machine without a
    /// writable `/tmp` renders unguarded, as before the lock existed).
    package static func acquire(_ path: String) throws(EngineError) -> RenderLock? {
        // 0o644 and read-only: the file another user created stays theirs, and still locks for us.
        let fd = open(path, O_RDONLY | O_CREAT, 0o644)
        guard fd >= 0 else { return nil }
        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            let busy = errno == EWOULDBLOCK
            close(fd)
            if busy { throw .renderAlreadyRunning }
            return nil
        }
        return RenderLock(fd)
    }

    deinit {
        flock(descriptor, LOCK_UN)
        close(descriptor)
    }
}
