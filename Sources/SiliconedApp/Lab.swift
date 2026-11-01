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
// - **No tabs, gestures that mix.** Describe writes into the text: it fills it when empty, otherwise it inserts at the
//   caret or replaces the selection. Rewrite works on all of it, additions and removals marked (« Edit » puts the marks
//   away). **⌘Z undoes any gesture of the language model**: the text is written through the text system (`PromptTextView.replace`).
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
    private(set) var streaming = ""
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

    var isWriting: Bool { phase != .idle }

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

    var target: Family? {
        if let c = chosenTarget, families.contains(c) { return c }
        if let f = rack.card?.family, families.contains(f) { return f }
        return families.first
    }

    /// The installed card the rack takes to render for `family`: its own when it is of that family.
    private func card(for family: Family) -> ModelCard? {
        if let current = rack.card, current.family == family { return current }
        return rack.cards.first { $0.family == family && rack.readySet.contains($0.id) }
    }

    /// **How many images the model aimed at reads** (0: none) — from its denoiser here, from the traits the other Mac
    /// published in a mirror.
    var capacity: Int {
        guard let family = target else { return 0 }
        if following { return rack.promptNamesImages ? rack.maxReferences : 0 }
        guard let card = card(for: family), let model = rack.model(card.id) else { return 0 }
        return model.chain.text.output.readsImages ? model.chain.denoising.maxReferences : 0
    }

    var offeredRoles: [LabInputs.Role] { LabInputs.roles(capacity: capacity) }

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
            if isWriting { stop(becauseOfRender: true) }
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
    var writerPause: (symbol: String, sentence: String)? {
        if following { return ("macbook", String(localized: "Rewrite and Describe are off while the window follows another Mac.")) }
        if rendering { return ("pause.circle", String(localized: "Rewrite and Describe wait while an image renders.")) }
        if case .unavailable(let why) = availability {
            switch why {
            case .notInstalled: return ("shippingbox", String(localized: "No writer is installed yet. Install a model to have one."))
            case .needsMemory(let n): return ("memorychip", String(localized: "The writer needs a Mac with \(n) GB of memory."))
            case .preparing: return ("hourglass", String(localized: "The writer is being prepared."))
            case .needsHead(let bytes):
                return ("arrow.down.circle", String(localized: "The writer needs one more download from Qwen-Image-2.1’s repository: \(Figures.disk(Int(bytes))).")) 
            }
        }
        if target == nil { return ("shippingbox", String(localized: "No writer is installed yet. Install a model to have one.")) }
        return nil
    }

    var canRewrite: Bool {
        writerPause == nil && !isWriting && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var canDescribe: Bool { writerPause == nil && !isWriting }

    // MARK: Writing

    /// **Rewrite**: the whole text, its mentions put aside and put back.
    func rewrite() {
        guard canRewrite, let target else { return }
        let typed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // `@` is a subject's only for a model that reads images; elsewhere it is the model's own (Anima's artists).
        let guarded = capacity > 0 ? LabInputs.MentionGuard(typed) : nil
        let input = LabInput.text(guarded?.masked ?? typed)
        begin(input, for: target, gesture: .rewrite, picture: nil) { [self] written in
            var final = written
            if let guarded {
                let back = guarded.restore(written)
                final = back.text
                if !back.intact { notice = Self.mentionNotice(lost: back.lost) }
            }
            place(final, replacingAll: true)
            changes = Changes(before: typed, after: text)
        }
    }

    /// **Describe**: an image of the zone, or a subject's first picture.
    func describe(_ picture: Picture) { describe(data: picture.data, id: picture.id) }

    func describe(_ subject: NamedSubjects.Subject) {
        guard let first = subject.pictures.first, let data = subjects.data(subject, first) else { return }
        describe(data: data, id: nil)
    }

    private func describe(data: Data, id: Picture.ID?) {
        guard canDescribe, let target else { return }
        begin(.image(data), for: target, gesture: .describe, picture: id) { [self] written in
            place(written, replacingAll: false)
        }
    }

    private func begin(_ input: LabInput, for target: Family, gesture: Gesture, picture: Picture.ID?,
                       finished: @escaping @MainActor (String) -> Void) {
        generation += 1
        let mine = generation
        phase = .loading(nil); streaming = ""; stopped = false; failure = nil; notice = nil; carried = false
        running = gesture; describing = picture
        task = Task { [writer] in
            do {
                var last = ""
                for try await event in writer.write(input, target: target) {
                    guard mine == self.generation else { return }
                    switch event {
                    case .loading(let f): phase = .loading(f)
                    case .text(let t): phase = .writing; streaming = t; last = t
                    case .finished(let t): last = t; end(last, finished)
                    case .cancelled: self.stop(becauseOfRender: false); return
                    }
                }
                guard mine == self.generation else { return }
                // A stream that ended without saying so: what it wrote is the result.
                if phase != .idle { end(last, finished) }
            } catch {
                guard mine == self.generation else { return }
                phase = .idle; streaming = ""; running = nil; describing = nil
                failure = error.localizedDescription
            }
        }
    }

    private func end(_ written: String, _ finished: @MainActor (String) -> Void) {
        let clean = written.trimmingCharacters(in: .whitespacesAndNewlines)
        phase = .idle; streaming = ""; running = nil; describing = nil
        guard !clean.isEmpty else { return }
        finished(clean)
    }

    /// The text changes through the text system when the field is on screen: **one undo step per gesture**, which is how ⌘Z
    /// undoes the writer. A Describe goes to the caret (or replaces the selection) and fills an empty text.
    private func place(_ written: String, replacingAll: Bool) {
        guard let view = PromptTextView.current, view.window != nil else {
            if replacingAll || text.isEmpty { text = written } else { text += (text.last?.isWhitespace == true ? "" : " ") + written }
            return
        }
        let current = view.string as NSString
        var range = (replacingAll || current.length == 0) ? NSRange(location: 0, length: current.length) : view.selectedRange()
        var inserted = written
        if !replacingAll, current.length > 0 {
            // Words are not glued to the text around.
            if range.location > 0, !current.substring(with: NSRange(location: range.location - 1, length: 1)).first!.isWhitespace {
                inserted = " " + inserted
            }
            let end = range.location + range.length
            if end < current.length, !current.substring(with: NSRange(location: end, length: 1)).first!.isWhitespace {
                inserted += " "
            }
        }
        if range.location > current.length { range = NSRange(location: current.length, length: 0) }
        view.replace(range, with: inserted)
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
        phase = .idle; streaming = ""; running = nil; describing = nil
        stopped = becauseOfRender
    }

    /// Puts the marks of the last rewrite away; the text stays.
    func dismissChanges() { changes = nil }

    /// The window closes: nothing keeps writing, nothing keeps the memory.
    func windowClosed() {
        if isWriting { stop() }
        Task { await writer.release() }
    }

    // MARK: The images

    /// A role for each image of the zone, as the rules see them.
    private var slots: [LabInputs.Slot<Picture.ID>] { pictures.map { LabInputs.Slot(id: $0.id, role: $0.role) } }

    /// After a change of the model aimed at: an image whose role it no longer offers is « describe only ».
    func conformRoles() {
        let kept = LabInputs.conforming(slots, capacity: capacity)
        for (i, slot) in kept.enumerated() where pictures[i].role != slot.role { pictures[i].role = slot.role }
    }

    func setRole(_ role: LabInputs.Role, of picture: Picture) {
        for slot in LabInputs.assigning(role, to: picture.id, in: slots) {
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

    /// « Prendre le prompt du rack ».
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
        return LabInputs.receipt(text: text, slots: slots, subjects: subjects.all, capacity: capacity,
                                 rackPrompt: rack.prompt, rackImages: rack.references.count, changesModel: changesModel)
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
        if !following, receipt.changesModel, let wanted = card(for: family) { app.identifier = wanted.id }
        if !carry.images.isEmpty { app.references = references(of: carry) }
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
        w.contentMinSize = NSSize(width: 560, height: 640)
        w.center()
        window = w
        NSApp.activate()
        w.makeKeyAndOrderFront(nil)
    }

    func windowWillClose(_ notification: Notification) { LabModel.shared.windowClosed() }
    func windowDidBecomeKey(_ notification: Notification) { LabModel.shared.refresh() }
}
