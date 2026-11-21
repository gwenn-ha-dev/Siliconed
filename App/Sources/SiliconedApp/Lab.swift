// **The Prompt Lab: where the inputs of a render are prepared, and an installed language model helps write them** — the
// Text and Images boxes at the head of the window's column (`LabBoxes.swift`).
//
//     ┌ Text ─────────────────────────────── for Qwen-Image-2.1 ● ┐
//     │                                                [✦ Rewrite] │
//     │ put (▣ alice) on the platform of image 1, fine rain, neon… │   the rack's prompt
//     └────────────────────────────────────────────────────────────┘
//     ┌ Images ───────────────────────────────────────────── 3/3 ● ┐
//     │ ▣ 1 · Image to Edit   ▣ 2 · Reference   ▣ 3 · @alice   +   │   the rack's images
//     └────────────────────────────────────────────────────────────┘
//
// - **One text, one set of images, nothing to carry.** The Lab edits the rack's own prompt (`AppState.prompt`) and images
//   (`AppState.inputs`): the rack shows what will go, the render translates the `@names` when it is queued
//   (`AppState.namedJob`). The rules are `LabInputs` (`Sources/Siliconed`), pure and tested.
// - **The starting image of an img2img** (Z-Image, Anima, Krea 2: a model that reads no image but starts from one) is a role
//   too, chosen by a gesture — « Start From This Image », or the role menu — never by default: a picture dropped to be described
//   must not turn the next render into an img2img. Its tile then carries the rack's strength (0 to 100 %, `StartImage`).
// - **No tabs, gestures that mix, written in the field as they come.** Describe writes into the text: it fills it when empty,
//   otherwise it inserts at the caret or replaces the selection. Rewrite works on all of it. Both draw their marks token by
//   token over the field (additions in the accent, removals struck; « Edit » puts them away), and the text lands in the field
//   once whole. **⌘Z undoes any gesture of the language model**: the text is written through the text system (`PromptTextView.replace`).
// - **Availability is judged gesture by gesture.** While an image renders, or without an installed writer, only Rewrite
//   and Describe wait, under one sentence; naming, framing and choosing a role need no model. **The render always
//   has priority** (`AppState.busy`): a write under way is cancelled, said once, quietly.
// - **Mentions are protected in a rewrite** (`LabInputs.MentionGuard`): each `@name` is put aside as a neutral token and put
//   back; one the writer lost or deformed is told, never dropped in silence.
// - **Incognito**: the text, the images, the subjects live in memory; no file, no default, no history. Folding the column
//   cancels a gesture under way (`RackColumnModel`).
// - **Following another Mac** (`Carrier.mirror`): Rewrite and Describe are off and say so (the writer is this Mac's); the
//   text, images, roles and subjects are the rack's that is shown, which sends its references to the Mac that renders.
// - The engine is behind `LabWriter` (`LabWriter.swift`): nothing of it stays in memory between two gestures.

import AppKit
import Siliconed
import SwiftUI

// MARK: - The state

@MainActor @Observable
final class LabModel {
    static let shared = LabModel()

    enum Gesture: Equatable { case rewrite, describe }
    enum Phase: Equatable { case idle, loading(Double?), writing }

    /// What a rewrite changed, to draw its marks while the text is still the one it wrote.
    struct Changes: Equatable { var before: String; var after: String }

    /// The engine's writer, over the library this Mac renders from (it can move).
    var writer: LabWriter { LabWriter(library: MainWindow.shared.local.library) }
    /// Some installed model carries a writer: read when the Text box appears, when a render ends and before a command.
    private(set) var installed = false

    /// **The text is the rack's prompt**: one text, typed here or in the rack.
    var text: String {
        get { rack.prompt }
        set {
            rack.prompt = newValue
            // The marks belong to the text the writer wrote: any other text puts them away.
            if let c = changes, newValue != c.after { changes = nil }
        }
    }
    /// **The images are the rack's** (`AppState.inputs`), each with its role.
    var pictures: [InputImage] { rack.inputs }
    private(set) var changes: Changes?

    private(set) var phase = Phase.idle
    /// The gesture under way, and the image it describes.
    private(set) var running: Gesture?
    private(set) var describing: InputImage.ID?
    private(set) var stopped = false
    private(set) var failure: String?
    /// The engine's stable code of that failure, when it has one (`prompt_too_long`…): what `silicontrol lab rewrite --wait` says.
    private(set) var failureCode: String?
    /// What a rewrite did not keep of the mentions, in one sentence.
    private(set) var notice: String?
    /// Whether a render is running or queued on this Mac: the writer waits.
    private(set) var rendering = false

    /// The sheet that names or edits a subject, while it is open — and the image it names.
    var draft: SubjectDraft?
    @ObservationIgnored private var naming: (picture: InputImage.ID, known: Int)?

    @ObservationIgnored private var task: Task<Void, Never>?
    /// One more at each gesture begun or stopped: a gesture that ends with the same number ended by itself (`lab rewrite --wait`).
    @ObservationIgnored private(set) var generation = 0
    @ObservationIgnored private var watching = false

    /// A gesture of the window, or one `silicontrol` asked: there is one writer, and it writes one text at a time.
    var isWriting: Bool { phase != .idle || remote != nil }

    /// **The gesture a command asked** (`LabRemote.swift`), if one is under way: it holds the writer like the window's does, and
    /// a render starting stops it (`stop` is what says so to the command).
    @ObservationIgnored var remote: (task: Task<Void, Never>, stop: @MainActor (RemoteOutcome) -> Void)?

    func refresh() { installed = writer.isInstalled }

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

    /// The model the Lab writes for: the rack's — one setting, chosen in the rack.
    var target: Family? { WriterTarget.pick(chosen: nil, rack: rack.card?.family, installed: families) }

    /// The installed card the rack takes to render for `family`: its own when it is of that family.
    func card(for family: Family) -> ModelCard? {
        if let current = rack.card, current.family == family { return current }
        return rack.cards.first { $0.family == family && rack.readySet.contains($0.id) }
    }

    /// **How many images the rack's model reads** (0: none) — from its denoiser here, from the traits the other Mac
    /// published in a mirror.
    var capacity: Int { rack.imageCapacity }

    /// The same, for the model a gesture writes for (`silicontrol rewrite --for`).
    func capacity(for family: Family?) -> Int {
        guard let family else { return 0 }
        if following { return rack.promptNamesImages ? rack.maxReferences : 0 }
        guard let card = card(for: family), let model = rack.model(card.id) else { return 0 }
        return model.chain.text.output.readsImages ? model.chain.denoising.maxReferences : 0
    }

    /// **Whether the rack's model starts from an image** (img2img). Never in a mirror — the other Mac's traits do not say it,
    /// and its jobs carry no starting image.
    var startsFromImage: Bool { rack.startsFromImage }

    /// The model's start is an img2img's.
    var startIsImg2Img: Bool { capacity == 0 && startsFromImage }

    var offeredRoles: [LabInputs.Role] { LabInputs.roles(capacity: capacity, startsFromImage: startsFromImage) }

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
            if phase != .idle { stop(becauseOfRender: true) }
            remote?.stop(.stoppedByRender)
        } else {
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
        return WriterPause.decide(following: following, rendering: rendering, installed: installed, hasTarget: target != nil)
    }

    static func presentation(_ pause: WriterPause) -> (symbol: String, sentence: String) {
        switch pause {
        case .following: ("macbook", String(localized: "Rewrite and Describe are off while the window follows another Mac."))
        case .rendering: ("pause.circle", String(localized: "Rewrite and Describe wait while an image renders."))
        case .notInstalled: ("shippingbox", String(localized: "No writer is installed yet. Install a model to have one."))
        }
    }

    var canRewrite: Bool { rewriteRefusal == nil }

    var canDescribe: Bool { describeRefusal == nil }

    /// **Why Rewrite is greyed**, its hover; `nil` when it can write.
    var rewriteRefusal: String? {
        if let p = writerPause { return p.sentence }
        if isWriting { return String(localized: "The writer is writing: wait for it, or stop it") }
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return String(localized: "Write a prompt first: Rewrite starts from it") }
        return nil
    }

    /// **Why Describe is greyed**, its hover; `nil` when it can write.
    var describeRefusal: String? {
        if let p = writerPause { return p.sentence }
        return isWriting ? String(localized: "The writer is writing: wait for it, or stop it") : nil
    }

    // MARK: Writing

    /// **Rewrite**: the whole text, its mentions put aside and put back. It is written **in the field, as it comes**: the marks
    /// (additions in the accent, removals struck) follow each token, and stay over the text it leaves.
    /// `false` when it could not start (the button is greyed then): `silicontrol lab rewrite` presses it too.
    @discardableResult
    func rewrite() -> Bool {
        guard canRewrite, let target else { return false }
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
        return true
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

    /// **Describe**: an image of the rack, or a subject's first picture. `false` when it could not start: `silicontrol lab describe`
    /// presses it too.
    @discardableResult
    func describe(_ picture: InputImage) -> Bool {
        guard let data = Self.bytes(of: picture) else { failure = String(localized: "This image could not be read."); return false }
        return describe(data: data, id: picture.id)
    }

    /// An image's bytes as it came — or, for pixels without bytes (a crop, a history image taken whole), a PNG of them.
    static func bytes(of picture: InputImage) -> Data? {
        if let bytes = picture.bytes { return bytes }
        switch picture.shown.source {
        case .file(let url): return try? Data(contentsOf: url)
        case .bytes(let data): return data
        case .image(let cg): return try? PNG.data(cg)
        case .subject(let cg, _): return cg.flatMap { try? PNG.data($0) }
        case .remote: return nil
        }
    }

    func describe(_ subject: NamedSubjects.Subject) {
        guard let first = subject.pictures.first, let data = subjects.data(subject, first) else { return }
        describe(data: data, id: nil)
    }

    /// Written in the field as it comes, where the caret was (or over the selection) when the gesture began — the field takes no
    /// key while the writer writes —, marked as an addition.
    @discardableResult
    private func describe(data: Data, id: InputImage.ID?) -> Bool {
        guard canDescribe, let target else { return false }
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
        return true
    }

    private func begin(_ input: LabInput, for target: Family, gesture: Gesture, picture: InputImage.ID?,
                       progress: @escaping @MainActor (String) -> Void, finished: @escaping @MainActor (String) -> Void) {
        generation += 1
        let mine = generation
        phase = .loading(nil); stopped = false; failure = nil; failureCode = nil; notice = nil; changes = nil
        running = gesture; describing = picture
        task = Task { [writer] in
            do {
                var last = ""
                for try await event in writer.write(input, instruction: instruction(for: target, input)) {
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
                failureCode = (error as? LabWriterFailure)?.code
            }
        }
    }

    private func end(_ written: String, _ finished: @MainActor (String) -> Void) {
        let clean = written.trimmingCharacters(in: .whitespacesAndNewlines)
        phase = .idle; running = nil; describing = nil
        // The marks drawn as it came belong to a text not yet in the field: the gesture's own replace them.
        changes = nil
        // Said, as a command hears it (`LabRemote`): the marks went, the text stayed, and nothing told why.
        guard !clean.isEmpty else {
            failure = String(localized: "The writer wrote nothing.")
            failureCode = "writer_failed"
            return
        }
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
        // A command's gesture holds the writer too (`isWriting`): the window's « Stop » ends it, and the command hears why.
        if !becauseOfRender { remote?.stop(.failed(code: "stopped", message: String(localized: "Stopped in the Prompt Lab."))) }
        generation += 1
        task?.cancel(); task = nil
        // Marks still coming are half a text's: the field keeps the one it had.
        if running != nil { changes = nil }
        phase = .idle; running = nil; describing = nil
        stopped = becauseOfRender
    }

    /// Puts the marks of the last rewrite away; the text stays.
    func dismissChanges() { changes = nil }

    // MARK: The instructions

    /// **What the writer is told**, family by family and gesture by gesture: ours, or the user's own written in Settings ›
    /// Prompt Lab (`PromptWriterInstructions.Edits`), kept in this Mac's preferences — the writer is this Mac's.
    var instructions = PromptWriterInstructions.Edits(UserDefaults.standard.dictionary(forKey: LabModel.instructionsKey) as? [String: String] ?? [:]) {
        didSet { UserDefaults.standard.set(instructions.texts, forKey: Self.instructionsKey) }
    }
    static let instructionsKey = "writerInstructions"

    /// The system turn for a gesture: Rewrite reads a text, Describe an image.
    func instruction(for family: Family, _ input: LabInput) -> String {
        instructions.instruction(for: family, input.gesture)
    }

    /// The gestures whose instruction for the family the lab writes for is the user's, in the order of the bar.
    var editedInstructions: [PromptWriterInstructions.Gesture] {
        guard let target else { return [] }
        return PromptWriterInstructions.Gesture.allCases.filter { instructions.sent(target, $0) != nil }
    }

    /// **« Send to the Prompt Lab »**: a history image comes in with the role a newcomer takes, and the column unfolds on it.
    func send(_ entry: Entry) {
        arrive {
            guard let image = await self.rack.decodedFromHistory(entry) else { return false }
            self.rack.addInput(image, bytes: entry.png)
            RackColumnModel.shared.unfold()
            return true
        }
    }

    // MARK: The images

    /// « Start From This Image »: the gesture that makes an img2img's start — the previous start, if any, goes back to neutral.
    func startFrom(_ picture: InputImage) { setRole(.start, of: picture) }

    func setRole(_ role: LabInputs.Role, of picture: InputImage) { rack.setRole(role, of: picture.id) }

    func remove(_ picture: InputImage) {
        if describing == picture.id { stop() }
        rack.removeInput(picture.id)
    }

    /// **An image comes in as bytes** (a drop of pixels, ⌘V): a history image (dragged from the column) as « Send to the Prompt
    /// Lab » brings it — its pixels, its ordinal for a mirror —, any other decoded upright: its pixels for an edit, its bytes
    /// for a start. It takes the role a newcomer takes. Decoded off the main thread, with its vignette: 51 ms for a 2048² PNG
    /// on an M5 Pro.
    func load(image data: Data, name: String = String(localized: "dropped image")) {
        arrive {
            if let entry = self.rack.history.first(where: { $0.png.count == data.count && $0.png == data }),
               let image = self.rack.fromHistory(entry) {
                self.rack.addInput(image, bytes: entry.png)
                return true
            }
            let decoded = await Task.detached(priority: .userInitiated) { () -> (image: CGImage, vignette: CGImage)? in
                guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                      let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
                return (cg, FinishedImage.reduce(cg, side: 240) ?? cg)
            }.value
            guard let decoded else { return false }
            self.rack.addInput(ReferenceImage(source: .image(decoded.image), vignette: NSImage(cgImage: decoded.vignette, size: .zero),
                                              name: name, width: decoded.image.width, height: decoded.image.height), bytes: data)
            return true
        }
    }

    /// The image of a gesture: bytes as above, a file as the render reads a file (`AppState.read`) — the same pixels for an
    /// edit and for a start (`ImageRGB(data:)` reads bytes as the file).
    func load(_ dropped: DroppedImage) {
        switch dropped {
        case .png(let data): load(image: data)
        case .file(let url):
            arrive {
                guard let image = await self.rack.read(url) else { return false }
                self.rack.addInput(image)
                return true
            }
        }
    }

    /// The images read off the main thread come in **in the order they were given** — several files dropped at once keep their
    /// order, and with it their roles —, each after the one before; one unreadable is said, the others still come.
    @ObservationIgnored private var arriving: Task<Void, Never>?

    private func arrive(_ read: @escaping @MainActor () async -> Bool) {
        let before = arriving
        arriving = Task {
            await before?.value
            failure = await read() ? nil : String(localized: "This image could not be read.")
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

    // MARK: The subjects

    /// The sheet opens on an image of the rack: once the subject is created, the image becomes it.
    func name(_ picture: InputImage) {
        guard let data = Self.bytes(of: picture), let draft = SubjectDraft(new: .png(data), taken: subjects.all.map(\.name)) else {
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
        rack.removeInput(naming.picture)
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
}
