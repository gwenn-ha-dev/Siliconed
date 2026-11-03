// **The Prompt Lab: where the inputs of a render are prepared, and an installed language model helps write them.**
//
//     Window ▸ Prompt Lab (⌥⌘L)
//     ┌ Prompt Lab ───────────────────────────── Write for [Qwen-Image-2.1 ▾] ┐
//     │ Images                                                                │   a library, as long as one wants
//     │ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────┐                   │   each image its role:
//     │ │ 1 · Start ▾│ │ 2 · @alice │ │ Describe   │ │ +  │  drop, open,      │   start · reference · @name · describe only
//     │ └────────────┘ └────────────┘ │ only      ▾│ └────┘  or from history   │
//     │ Text                           └────────────┘      Use the Rack’s Prompt│
//     │ ┌───────────────────────────────────────────────────────────────────┐ │   one text, always editable by hand
//     │ │ (▣ alice) on a station platform at night, neon, fine rain…        │ │   `@name` drawn as a chip
//     │ └───────────────────────────────────────────────────────────────────┘ │
//     │ [ Rewrite  Beta ]                                                     │
//     │ The rack will receive: « …the person in <image2>… »  1 ▣  2 ▣  1024×1536│
//     │ Replaces the rack’s prompt (⌘Z brings it back) and its images.        │
//     │                                              [ Carry to the Rack ]    │
//     └───────────────────────────────────────────────────────────────────────┘
//
// - **The lab prepares, the rack shows what the render reads.** « Carry to the Rack » fills the boxes the rack has
//   always had: its Prompt (the text translated, the `@names` turned into « image N »; ⌘Z in the field brings the old
//   prompt back) and its Images box (1 to 3 numbered images, which have no undo — the line says so). The rack does not
//   know the subjects. The rules are `LabInputs` (`Sources/Siliconed`), pure and tested.
// - **The starting image of an img2img** (Z-Image, Anima, Krea 2: a model that reads no image but starts from one) is a role
//   too, chosen by a gesture — « Start From This Image », or the role menu — never by default: a picture dropped to be described
//   must not turn the next render into an img2img. Its tile then carries the strength (0 to 100 %, `StartImage`), and « Carry »
//   puts both in the rack's Images box (`AppState.carryStart`), the size set to the format nearest the image's proportions.
// - **No tabs, gestures that mix, written in the field as they come.** Describe writes into the text: it fills it when empty,
//   otherwise it inserts at the caret or replaces the selection. Rewrite works on all of it. Both draw their marks token by
//   token over the field (additions in the accent, removals struck; « Edit » puts them away), and the text lands in the field
//   once whole. **⌘Z undoes any gesture of the language model**: the text is written through the text system (`PromptTextView.replace`).
// - **Availability is judged gesture by gesture.** While an image renders, or without an installed writer, only Rewrite
//   and Describe wait, under one sentence; naming, framing, choosing a role and carrying need no model. **The render always
//   has priority** (`AppState.busy`): a write under way is cancelled — said once, quietly — and the writer gives its memory back.
// - **Mentions are protected in a rewrite** (`LabInputs.MentionGuard`): each `@name` is put aside as a neutral token and put
//   back; one the writer lost or deformed is told, never dropped in silence.
// - **Incognito**: the text, the images, the subjects live in memory; no file, no default, no history. Closing the window cancels
//   and releases; the texts stay for the session.
// - **Following another Mac** (`Carrier.mirror`): Rewrite and Describe are off and say so (the writer is this Mac's); images, roles,
//   subjects and « Carry » fill the rack that is shown, which sends its references to the Mac that renders.
// - The engine is behind `LabWriter` (`LabWriter.swift`).

import AppKit
import Siliconed
import SwiftUI

// MARK: - The state

@MainActor @Observable
final class LabModel {
    static let shared = LabModel()

    enum Gesture: Equatable { case rewrite, describe }
    enum Phase: Equatable { case idle, loading(Double?), writing }

    /// An image of the zone: its bytes as given, never re-encoded, and what the render makes of it.
    struct Picture: Identifiable {
        let id = UUID()
        let data: Data
        let thumbnail: NSImage
        /// Upright, in pixels: image 1's give the format.
        let width: Int, height: Int
        var role: LabInputs.Role
    }

    /// **The strength the starting image is carried with** (img2img), set on its tile — the rack's own is retouched there afterwards.
    /// The aimed family's default (`StartImage.defaultStrength`) until one is chosen.
    var startStrength: Double { chosenStartStrength ?? defaultStartStrength }
    var defaultStartStrength: Double { StartImage.defaultStrength(for: target) }
    private var chosenStartStrength: Double?

    /// What a rewrite changed, to draw its marks while the text is still the one it wrote.
    struct Changes: Equatable { var before: String; var after: String }

    /// The engine's writer; nothing is read from disk until the first `availability`, and no weight until the first gesture.
    var writer: any LabWriter = EngineLabWriter(library: MainWindow.shared.local.library) { didSet { refresh() } }
    private(set) var availability: LabAvailability = .unavailable(.preparing)

    var text = "" {
        // The marks belong to the text the writer wrote: any other text puts them away.
        didSet { if let c = changes, text != c.after { changes = nil } }
    }
    var pictures: [Picture] = []
    /// The family chosen by hand; `nil` follows the rack's.
    var chosenTarget: Family?
    private(set) var changes: Changes?

    private(set) var phase = Phase.idle
    /// The gesture under way, and the image it describes.
    private(set) var running: Gesture?
    private(set) var describing: Picture.ID?
    private(set) var stopped = false
    private(set) var failure: String?
    /// What a rewrite did not keep of the mentions, in one sentence.
    private(set) var notice: String?
    private(set) var carried = false
    /// Whether a render is running or queued on this Mac: the writer waits.
    private(set) var rendering = false

    /// The sheet that names or edits a subject, while it is open — and the image it names.
    var draft: SubjectDraft?
    @ObservationIgnored private var naming: (picture: Picture.ID, known: Int)?

    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var watching = false

    /// A gesture of the window, or one `silicontrol` asked: there is one writer, and it writes one text at a time.
    var isWriting: Bool { phase != .idle || remote != nil }

    /// **The gesture a command asked** (`LabRemote.swift`), if one is under way: it holds the writer like the window's does, and
    /// a render starting stops it (`stop` is what says so to the command).
    @ObservationIgnored var remote: (task: Task<Void, Never>, stop: @MainActor () -> Void)?

    func refresh() {
        (writer as? EngineLabWriter)?.library = MainWindow.shared.local.library
        availability = writer.availability
    }

    // MARK: Where it writes for

    /// The rack the lab fills: the one the window shows — this Mac's, or the other Mac's mirror.
    var rack: AppState { MainWindow.shared.app }
    /// Whether the window follows another Mac.
    var following: Bool { MainWindow.shared.carrier.mirror != nil }
    var subjects: SubjectLibrary { MainWindow.shared.local.subjects }

    /// The families the lab prepares for: those installed here, in the catalog's order — or, in a mirror, the model
    /// the other Mac's rack has chosen.
    var families: [Family] {
        if following { return rack.card.map { [$0.family] } ?? [] }
        let ready = Set(rack.cards.filter { rack.readySet.contains($0.id) }.map(\.family))
        return Family.allCases.filter(ready.contains)
    }

    var target: Family? { WriterTarget.pick(chosen: chosenTarget, rack: rack.card?.family, installed: families) }

    /// The installed card the rack takes to render for `family`: its own when it is of that family.
    func card(for family: Family) -> ModelCard? {
        if let current = rack.card, current.family == family { return current }
        return rack.cards.first { $0.family == family && rack.readySet.contains($0.id) }
    }

    /// **How many images the model aimed at reads** (0: none) — from its denoiser here, from the traits the other Mac
    /// published in a mirror.
    var capacity: Int { capacity(for: target) }

    /// The same, for the model a gesture writes for (`silicontrol rewrite --for`).
    func capacity(for family: Family?) -> Int {
        guard let family else { return 0 }
        if following { return rack.promptNamesImages ? rack.maxReferences : 0 }
        guard let card = card(for: family), let model = rack.model(card.id) else { return 0 }
        return model.chain.text.output.readsImages ? model.chain.denoising.maxReferences : 0
    }

    /// **Whether the model aimed at starts from an image** (img2img): its chain offers the image entry. Never in a mirror — the
    /// other Mac's traits do not say it, and its jobs carry no starting image.
    var startsFromImage: Bool { startsFromImage(for: target) }

    func startsFromImage(for family: Family?) -> Bool {
        guard !following, let family, let card = card(for: family), let model = rack.model(card.id) else { return false }
        return model.chain.entries.contains { $0.kind == .image }
    }

    /// The model's start is an img2img's — not Qwen-Image-2.1's image 1, which the model reads as a reference.
    var startIsImg2Img: Bool { capacity == 0 && startsFromImage }

    var offeredRoles: [LabInputs.Role] { LabInputs.roles(capacity: capacity, startsFromImage: startsFromImage) }

    /// What the roles depend on: a change of either conforms them.
    var rolesKey: String { "\(capacity)·\(startsFromImage)" }

    // MARK: The render's priority

    /// Follows `AppState.busy` for as long as the process lives — not through a view, which SwiftUI may
    /// stop updating once its window is hidden.
    func watchRender() {
        guard !watching else { return }
        watching = true
        rendering = MainWindow.shared.local.busy
        observeRender()
    }

    private func observeRender() {
        withObservationTracking {
            _ = MainWindow.shared.local.busy
        } onChange: { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.renderChanged(MainWindow.shared.local.busy)
                self.observeRender()
            }
        }
    }

    private func renderChanged(_ busy: Bool) {
        guard busy != rendering else { return }
        rendering = busy
        if busy {
            // The memory goes back even when nothing was being written: the writer may still be loaded.
            if phase != .idle { stop(becauseOfRender: true) }
            remote?.stop()
            Task { await writer.release() }
        } else {
            refresh()
        }
    }

    // MARK: The writer's head

    /// The head is being downloaded and forged (`Installer.installWriterHead`), through the app's forge: one job at a
    /// time, no render meanwhile, Stop as for any installation.
    private(set) var installingHead = false
    /// The head's installation failed: one sentence.
    private(set) var headFailure: String?

    private enum HeadOutcome: Sendable { case done, stopped, failed(String) }

    /// **One button**: the encoder's family is installed, so its licence was accepted when it was; the head comes from the
    /// same repository (two byte ranges of the encoder's file, checked by sha256), and the button says its size.
    func installHead(_ app: AppState) {
        guard !installingHead, app.managementPossible, case .unavailable(.needsHead) = availability else { return }
        installingHead = true; headFailure = nil
        app.launchForge(String(localized: "Installing the writer"), task: .other) { library, cancellation, journal -> HeadOutcome in
            var installer = Installer(library: library)
            installer.cancellation = cancellation; installer.journal = journal
            do {
                try installer.installWriterHead(Library.writerFamily)
                return .done
            } catch {
                if EngineError(error) == .cancelled { return .stopped }
                return .failed(Problem(error).title)
            }
        } end: { [self] outcome in
            installingHead = false
            switch outcome {
            case .done: app.flash(String(localized: "The writer is installed"))
            case .stopped: app.flash(String(localized: "Stopped — the files already downloaded are kept; the one in progress will start over"))
            case .failed(let why): headFailure = why
            }
            refresh()
        }
    }

    // MARK: What waits, gesture by gesture

    /// **Why Rewrite and Describe wait**, in one sentence; `nil` when they can run. They are the only gestures that need the
    /// language model: everything else in the lab works without it, during a render too.
    var writerPause: (symbol: String, sentence: String)? { pause(rendering: rendering).map(Self.presentation) }

    /// The decision (`WriterPause.decide`), for the window and for `silicontrol rewrite` and `describe`: the render is passed in
    /// because the command reads the queue itself, not the copy this model keeps up to date for the window.
    func pause(rendering: Bool) -> WriterPause? {
        let installed: WriterPause?
        switch availability {
        case .ready: installed = nil
        case .unavailable(.notInstalled): installed = .notInstalled
        case .unavailable(.needsMemory(let n)): installed = .needsMemory(gigabytes: n)
        case .unavailable(.preparing): installed = .preparing
        case .unavailable(.needsHead(let bytes)): installed = .needsHead(bytes: bytes)
        }
        return WriterPause.decide(following: following, rendering: rendering, installed: installed, hasTarget: target != nil)
    }

    static func presentation(_ pause: WriterPause) -> (symbol: String, sentence: String) {
        switch pause {
        case .following: ("macbook", String(localized: "Rewrite and Describe are off while the window follows another Mac."))
        case .rendering: ("pause.circle", String(localized: "Rewrite and Describe wait while an image renders."))
        case .notInstalled: ("shippingbox", String(localized: "No writer is installed yet. Install a model to have one."))
        case .needsMemory(let n): ("memorychip", String(localized: "The writer needs a Mac with \(n) GB of memory."))
        case .preparing: ("hourglass", String(localized: "The writer is being prepared."))
        case .needsHead(let bytes):
            ("arrow.down.circle", String(localized: "The writer needs one more download from Qwen-Image-2.1’s repository: \(Figures.disk(Int(bytes))).")) 
        }
    }

    var canRewrite: Bool {
        writerPause == nil && !isWriting && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var canDescribe: Bool { writerPause == nil && !isWriting }

    // MARK: Writing

    /// **Rewrite**: the whole text, its mentions put aside and put back. It is written **in the field, as it comes**: the marks
    /// (additions in the accent, removals struck) follow each token, and stay over the text it leaves.
    func rewrite() {
        guard canRewrite, let target else { return }
        let typed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let (input, guarded) = Self.rewriteInput(typed, capacity: capacity)
        begin(input, for: target, gesture: .rewrite, picture: nil, progress: { [self] written in
            changes = Changes(before: typed, after: guarded?.restore(written).text ?? written)
        }) { [self] written in
            let back = Self.rewritten(written, guarded: guarded)
            notice = back.notice
            place(Self.splice(back.text, into: text, at: nil))
            changes = Changes(before: typed, after: text)
        }
    }

    /// **What a rewrite sends the writer** — the lab's and `silicontrol rewrite`'s alike. `@` is a subject's only for a model that
    /// reads images (`capacity` > 0); elsewhere it is the model's own (Anima's artists), and the text goes as it is.
    static func rewriteInput(_ typed: String, capacity: Int) -> (input: LabInput, guarded: LabInputs.MentionGuard?) {
        let guarded = capacity > 0 ? LabInputs.MentionGuard(typed) : nil
        return (.text(guarded?.masked ?? typed), guarded)
    }

    /// **What a rewrite gives back**: the mentions put back, and the sentence that says which one the writer lost or deformed.
    static func rewritten(_ written: String, guarded: LabInputs.MentionGuard?) -> (text: String, lost: [String], notice: String?) {
        guard let guarded else { return (written, [], nil) }
        let back = guarded.restore(written)
        return (back.text, back.lost, back.intact ? nil : mentionNotice(lost: back.lost))
    }

    /// **Describe**: an image of the zone, or a subject's first picture.
    func describe(_ picture: Picture) { describe(data: picture.data, id: picture.id) }

    func describe(_ subject: NamedSubjects.Subject) {
        guard let first = subject.pictures.first, let data = subjects.data(subject, first) else { return }
        describe(data: data, id: nil)
    }

    /// Written in the field as it comes, where the caret was (or over the selection) when the gesture began — the field takes no
    /// key while the writer writes —, marked as an addition.
    private func describe(data: Data, id: Picture.ID?) {
        guard canDescribe, let target else { return }
        let before = text
        // The field off screen: after the text, as a caret at its end would.
        let selection = PromptTextView.current.flatMap { $0.window == nil ? nil : $0.selectedRange() }
            ?? NSRange(location: (before as NSString).length, length: 0)
        begin(.image(data), for: target, gesture: .describe, picture: id, progress: { [self] written in
            changes = Changes(before: before, after: Self.splice(written, into: before, at: selection).text)
        }) { [self] written in
            place(Self.splice(written, into: text, at: selection))
            changes = Changes(before: before, after: text)
        }
    }

    private func begin(_ input: LabInput, for target: Family, gesture: Gesture, picture: Picture.ID?,
                       progress: @escaping @MainActor (String) -> Void, finished: @escaping @MainActor (String) -> Void) {
        generation += 1
        let mine = generation
        phase = .loading(nil); stopped = false; failure = nil; notice = nil; carried = false; changes = nil
        running = gesture; describing = picture
        task = Task { [writer] in
            do {
                var last = ""
                for try await event in writer.write(input, target: target) {
                    guard mine == self.generation else { return }
                    switch event {
                    case .loading(let f): phase = .loading(f)
                    case .text(let t): phase = .writing; last = t; progress(t)
                    case .finished(let t, _): last = t; end(last, finished)
                    case .cancelled: self.stop(becauseOfRender: false); return
                    }
                }
                guard mine == self.generation else { return }
                // A stream that ended without saying so: what it wrote is the result.
                if phase != .idle { end(last, finished) }
            } catch {
                guard mine == self.generation else { return }
                phase = .idle; running = nil; describing = nil; changes = nil
                failure = error.localizedDescription
            }
        }
    }

    private func end(_ written: String, _ finished: @MainActor (String) -> Void) {
        let clean = written.trimmingCharacters(in: .whitespacesAndNewlines)
        phase = .idle; running = nil; describing = nil
        // The marks drawn as it came belong to a text not yet in the field: the gesture's own replace them.
        changes = nil
        guard !clean.isEmpty else { return }
        finished(clean)
    }

    /// **Where a written text goes, and with what around it**: the whole text (`range` nil and a rewrite, or an empty text),
    /// else the caret or the selection, the words not glued to the text around. Pure: the live marks and the final placing
    /// compute the same text.
    struct Splice: Equatable { var text: String; var range: NSRange; var inserted: String }

    static func splice(_ written: String, into current: String, at selection: NSRange?) -> Splice {
        let ns = current as NSString
        guard let selection, ns.length > 0 else {
            return Splice(text: written, range: NSRange(location: 0, length: ns.length), inserted: written)
        }
        let location = min(selection.location, ns.length)
        let range = NSRange(location: location, length: min(selection.length, ns.length - location))
        var inserted = written
        if range.location > 0, !ns.substring(with: NSRange(location: range.location - 1, length: 1)).first!.isWhitespace {
            inserted = " " + inserted
        }
        let end = range.location + range.length
        if end < ns.length, !ns.substring(with: NSRange(location: end, length: 1)).first!.isWhitespace { inserted += " " }
        return Splice(text: ns.replacingCharacters(in: range, with: inserted), range: range, inserted: inserted)
    }

    /// The text changes through the text system when the field is on screen: **one undo step per gesture**, which is how ⌘Z
    /// undoes the writer.
    ///
    /// The field is still locked when the gesture ends (SwiftUI unlocks it at its next update), and a locked `NSTextView` refuses
    /// `shouldChangeText`: `replace` would then set `string`, which tells no delegate, the binding would keep the old text and
    /// put it back at the next update — the writer's text lost. So the field is unlocked first.
    private func place(_ splice: Splice) {
        guard let view = PromptTextView.current, view.window != nil, view.string == text else {
            text = splice.text
            return
        }
        if !view.isEditable { view.isEditable = true }
        view.replace(splice.range, with: splice.inserted)
        view.window?.makeFirstResponder(view)
    }

    private static func mentionNotice(lost: [String]) -> String {
        if lost.isEmpty { return String(localized: "The rewrite deformed a mention. Check the @names in the text; ⌘Z brings your text back.") }
        return String(localized: "The rewrite lost \(lost.joined(separator: ", ")). Cite it again, or press ⌘Z to bring your text back.")
    }

    /// The previous text stays: half a prompt is not offered to the field.
    func stop(becauseOfRender: Bool = false) {
        generation += 1
        task?.cancel(); task = nil
        // Marks still coming are half a text's: the field keeps the one it had.
        if running != nil { changes = nil }
        phase = .idle; running = nil; describing = nil
        stopped = becauseOfRender
    }

    /// Puts the marks of the last rewrite away; the text stays.
    func dismissChanges() { changes = nil }

    /// The window closes: nothing keeps writing, nothing keeps the memory.
    func windowClosed() {
        if phase != .idle { stop() }
        // A command's gesture is not the window's: closing the window does not take its writer away.
        guard remote == nil else { return }
        Task { await writer.release() }
    }

    // MARK: The images

    /// A role for each image of the zone, as the rules see them.
    private var slots: [LabInputs.Slot<Picture.ID>] { pictures.map { LabInputs.Slot(id: $0.id, role: $0.role) } }

    /// After a change of the model aimed at: an image whose role it no longer offers is « describe only ».
    func conformRoles() {
        let kept = LabInputs.conforming(slots, capacity: capacity, startsFromImage: startsFromImage)
        for (i, slot) in kept.enumerated() where pictures[i].role != slot.role { pictures[i].role = slot.role }
    }

    /// « Start From This Image »: the gesture that makes an img2img's start — the previous start, if any, goes back to « describe only ».
    func startFrom(_ picture: Picture) { setRole(.start, of: picture) }

    func setStartStrength(_ s: Double) { chosenStartStrength = StartImage.snapped(s) }

    func setRole(_ role: LabInputs.Role, of picture: Picture) {
        for slot in LabInputs.assigning(role, to: picture.id, in: slots, capacity: capacity) {
            if let i = pictures.firstIndex(where: { $0.id == slot.id }), pictures[i].role != slot.role { pictures[i].role = slot.role }
        }
    }

    func remove(_ picture: Picture) {
        if describing == picture.id { stop() }
        pictures.removeAll { $0.id == picture.id }
    }

    /// An image from bytes: its thumbnail and size are read upright; it takes the role the zone gives a newcomer.
    func load(image data: Data) {
        guard let upright = SubjectPictures.upright(data), upright.width > 0 else {
            failure = String(localized: "This image could not be read.")
            return
        }
        let thumbnail = FinishedImage.reduce(upright, side: 360).map { NSImage(cgImage: $0, size: .zero) } ?? NSImage(cgImage: upright, size: .zero)
        pictures.append(Picture(data: data, thumbnail: thumbnail, width: upright.width, height: upright.height,
                                role: LabInputs.defaultRole(adding: slots, capacity: capacity)))
        failure = nil
    }

    func load(_ dropped: DroppedImage) {
        switch dropped {
        case .png(let data): load(image: data)
        case .file(let url):
            guard let data = try? Data(contentsOf: url) else {
                failure = String(localized: "This image could not be read.")
                return
            }
            load(image: data)
        }
    }

    func chooseImages() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.prompt = String(localized: "Choose")
        panel.message = String(localized: "Images to describe, to start from, or to name.")
        guard panel.runModal() == .OK else { return }
        for url in panel.urls { load(.file(url)) }
    }

    /// “Use the Rack’s Prompt”.
    func useRackPrompt() { text = rack.prompt }

    // MARK: The subjects

    /// The sheet opens on an image of the zone: once the subject is created, the image becomes it.
    func name(_ picture: Picture) {
        guard let draft = SubjectDraft(new: .png(picture.data), taken: subjects.all.map(\.name)) else {
            failure = String(localized: "This image could not be read.")
            return
        }
        naming = (picture.id, subjects.all.count)
        self.draft = draft
    }

    /// The open panel, then the sheet: an image to name that is not in the zone.
    func chooseImageToName() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.prompt = String(localized: "Choose")
        panel.message = String(localized: "An image to name: a person, a garment, an object, a place or a style.")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard let draft = SubjectDraft(new: .file(url), taken: subjects.all.map(\.name)) else {
            failure = String(localized: "This image could not be read.")
            return
        }
        naming = nil
        self.draft = draft
    }

    func edit(_ s: NamedSubjects.Subject) {
        naming = nil
        draft = SubjectDraft(editing: s, library: subjects)
    }

    /// The sheet is closed: an image it named leaves the zone — it is a subject now.
    func sheetClosed() {
        defer { naming = nil }
        guard let naming, subjects.all.count > naming.known else { return }
        pictures.removeAll { $0.id == naming.picture }
    }

    /// Asked first: texts that cite it are refused afterwards.
    func confirmDeletion(_ s: NamedSubjects.Subject) {
        let alert = NSAlert()
        alert.messageText = String(localized: "Delete @\(s.name)?")
        alert.informativeText = String(localized: "Prompts that cite it will be refused.")
        alert.addButton(withTitle: String(localized: "Delete"))
        alert.addButton(withTitle: String(localized: "Cancel"))
        alert.buttons[0].hasDestructiveAction = true
        if alert.runModal() == .alertFirstButtonReturn { deleteSubject(s) }
    }

    func deleteSubject(_ s: NamedSubjects.Subject) { subjects.delete(s) }

    /// `@name` inserted where the caret is (a click on a subject) — appended when the field is not on screen. A space
    /// before it when it would glue to a word.
    func cite(_ s: NamedSubjects.Subject) {
        if let field = PromptTextView.current, field.window != nil {
            field.insertMention(s.name)
            return
        }
        let glue = text.last.map { !$0.isWhitespace } ?? false
        text += (glue ? " " : "") + "@\(s.name) "
    }

    /// A subject renamed: the text's mentions of it follow.
    func renameMentions(_ old: String, to new: String) {
        var renamed = text
        for m in NamedSubjects.mentions(in: text).reversed() where NamedSubjects.key(m.name) == NamedSubjects.key(old) {
            if let r = Range(NSRange(m.range, in: text), in: renamed) { renamed.replaceSubrange(r, with: "@" + new) }
        }
        if renamed != text { text = renamed }
    }

    // MARK: What the rack will receive

    /// **The line under the lab** (`LabInputs.receipt`): the translated text, the images numbered, and what carrying replaces.
    var receipt: LabInputs.Receipt<Picture.ID> {
        let family = target
        let changesModel = !following && family != nil && rack.card?.family != family
        let img2img = startIsImg2Img
        return LabInputs.receipt(text: text, slots: slots, subjects: subjects.all, capacity: capacity, startsFromImage: img2img,
                                 rackPrompt: rack.prompt,
                                 rackImages: img2img ? (rack.startImage == nil ? 0 : 1) : rack.references.count,
                                 changesModel: changesModel)
    }

    /// The picture behind an image of the carried list.
    func picture(_ id: Picture.ID) -> Picture? { pictures.first { $0.id == id } }

    /// The size of an image of the carried list: image 1's gives the format.
    func size(of image: LabInputs.Image<Picture.ID>) -> (width: Int, height: Int)? {
        switch image {
        case .plain(let id, _): picture(id).map { ($0.width, $0.height) }
        case .subject(let p): subjects.framedSize(p.subject, p.picture)
        }
    }

    /// **The format that will come out**: the one the model derives from image 1 (`DenoisingModule.editFormat`), as the
    /// rack shows it. `nil` in a mirror (the other Mac computes it) or without an image.
    var outputFormat: RecommendedFormat? {
        guard !following, let first = receipt.first, let size = size(of: first), let family = target,
              let card = card(for: family), let denoising = rack.model(card.id)?.chain.denoising else { return nil }
        // An img2img's size follows its starting image too, but among the model's sizes: the nearest proportions, at the rack's scale.
        if startIsImg2Img {
            return rack.identifier == card.id
                ? rack.recommendedFormat(width: size.width, height: size.height)
                : StartImage.recommendedFormat(width: size.width, height: size.height, among: card.formats, current: rack.format)
        }
        let f = denoising.editFormat(referenceWidth: size.width, referenceHeight: size.height)
        return AppState.cappedForEditing(RecommendedFormat(f.width, f.height))
    }

    /// **« Carry to the Rack »**: the rack takes the lab's model if it was on another family, its Images box takes the
    /// numbered images (when the lab sends any), its Prompt the translated text — the old one kept by the window's undo
    /// manager, so that ⌘Z in the field brings it back (`MainWindow.setPrompt`).
    func carry() {
        let receipt = receipt
        guard case .carries(let carry) = receipt.outcome, let family = target else { return }
        let app = rack
        let img2img = startIsImg2Img
        if !following, receipt.changesModel, let wanted = card(for: family) { app.identifier = wanted.id }
        if img2img, case .plain(let id, _)? = carry.images.first, let picture = picture(id) {
            app.carryStart(AppState.startImage(bytes: picture.data, thumbnail: picture.thumbnail, width: picture.width,
                                               height: picture.height, name: String(localized: "image from the lab")),
                           strength: startStrength)
        } else if !carry.images.isEmpty {
            app.references = references(of: carry)
        }
        if receipt.replacesPrompt { MainWindow.shared.setPrompt(carry.prompt, of: app) }
        self.carried = true
        MainWindow.shared.show()
        Task {
            try? await Task.sleep(for: .seconds(2))
            self.carried = false
        }
    }

    /// The rack's references: the images of the zone as they are (upright, whole), the subjects' pictures cropped to their frame.
    private func references(of carry: LabInputs.Carry<Picture.ID>) -> [ReferenceImage] {
        carry.images.map { image in
            switch image {
            case .subject(let placed):
                return subjects.references([placed])[0]
            case .plain(let id, _):
                let picture = self.picture(id)
                let data = picture?.data ?? Data()
                guard let cg = SubjectPictures.upright(data) else {
                    return ReferenceImage(source: .bytes(data), vignette: picture?.thumbnail ?? NSImage(),
                                          name: String(localized: "image from the lab"), width: picture?.width ?? 1, height: picture?.height ?? 1)
                }
                let vignette = FinishedImage.reduce(cg, side: 240) ?? cg
                return ReferenceImage(source: .image(cg), vignette: NSImage(cgImage: vignette, size: .zero),
                                      name: String(localized: "image from the lab"), width: cg.width, height: cg.height)
            }
        }
    }
}

// MARK: - The window

@MainActor
final class LabWindow: NSObject, NSWindowDelegate {
    static let shared = LabWindow()
    private var window: NSWindow?

    func show() {
        let model = LabModel.shared
        model.watchRender()
        model.refresh()
        if let window {
            NSApp.activate()
            window.makeKeyAndOrderFront(nil)
            return
        }
        let host = NSHostingController(rootView: LabView(model: model))
        host.sizingOptions = []
        let w = NSWindow(contentViewController: host)
        w.title = String(localized: "Prompt Lab")
        w.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        w.isReleasedWhenClosed = false
        // Incognito: no frame kept, nothing restored.
        w.isRestorable = false
        w.delegate = self
        w.setContentSize(NSSize(width: 640, height: 780))
        w.center()
        window = w
        NSApp.activate()
        w.makeKeyAndOrderFront(nil)
    }

    /// **The smallest the Lab gets** (`MinimumContent`, issue #7): its pieces with the text at about twice its own minimum (110 pt),
    /// and room for the status lines a rewrite adds.
    static let minimumContent = NSSize(width: 560, height: 640)

    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
        MinimumContent.clamp(frameSize, of: sender, to: Self.minimumContent)
    }

    func windowWillClose(_ notification: Notification) { LabModel.shared.windowClosed() }
    func windowDidBecomeKey(_ notification: Notification) { LabModel.shared.refresh() }
}
