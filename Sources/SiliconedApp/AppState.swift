// **The app's state**: what the user sets, what the render shows, the history.
//
// **Incognito by default**: the history lives in memory and dies with the
// process. Nothing is written to disk without an explicit gesture — "Save", "Copy" —,
// no temporary file, no preference, no log: nothing that Spotlight or recent items
// could find. **The exceptions**, each a gesture of the user: the preferences (UserDefaults) — the « More
// Solvers » switch (`developerModeKey`), « Faster Rendering », the comparison's curtain, switch or side by side
// (`Curtain.mode`), the beta « Negative LoRA Strengths » switch
// (`negativeLoRAKey`), the file format of the last export (`exportFormatKey`), the Neural Engine's choice per
// model (`choiceKey`), the Macs paired and the lending switch (`OtherMacs`), and this Mac's capacity map
// with the last render's note against it (`capacityMapKey`, `renderNoteKey`), measured when
// the user asks, like the system language the app follows —, the licenses accepted, which the library keeps itself (`<local>/accepted-licenses.json`) because its
// engine refuses a render without them, and where the library is when the user moved it (`<local>/library-location`,
// `AppState+Place.swift`). A library on another disk is rendered through a copy on this Mac, removed when the
// app quits (`WorkingCopy`). The named subjects (`Subjects.swift`) live in memory, like
// the history.
//
// **Five models in front**, in this order: Z-Image Turbo, Qwen-Image-2.1 Turbo, Anima Turbo, Krea 2
// Turbo, MOSS-Image (`frontFamilies`) — every family the library knows, so nothing is filtered: the order
// is all `frontFamilies` sets. « More Solvers » (`developerMode`, a key older than its name) offers the
// solvers on those models (`offeredSolvers`): Euler, the distillation's own, is every other render's.
//
// **The main thread does no computing**: the engine runs on its queue, and what the app does with
// the pixels — decoding the reference image, encoding the PNG, reducing the thumbnail — goes to a
// detached task. The catalog (ready models, compatible LoRAs, what is missing) is read from disk when
// it changes, never at each drawing of the view.
//
// **All the phrases go through `String(localized:)`** or through a `Text` key: the table is
// `Localizable.xcstrings` (English source, then fr de es it ja zh-Hans), which `tools/app.sh` synchronizes and compiles.
// The engine's refusals are translated case by case in `Problems.swift`.
//
// A client of `Siliconed` from the same package (`import`, never `@testable`): it reaches the library's
// `package` symbols, never its internals. The view is in `MainView.swift`, the window and the app in `App.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

@MainActor @Observable
final class AppState {
    /// The library: this Mac's, or the one the user moved or opened elsewhere (`AppState+Place.swift`).
    var library: Library
    /// **The copy renders read** when the library is on another disk (`WorkingCopy.isNeeded`), `nil` otherwise.
    @ObservationIgnored var workingCopy: WorkingCopy?
    /// The library's disk is connected (re-read when a volume comes or goes: `watchVolumes`).
    var libraryReachable = true
    /// A copy to this Mac under way, and how far (`prepareCopy`).
    var copying: CopyProgress?
    @ObservationIgnored var copiesInFlight = 0
    /// The token of the copy the model's choice started: a new choice stops it (`warm`).
    @ObservationIgnored var warmCancellation: Cancellation?
    @ObservationIgnored var volumeObservers: [NSObjectProtocol] = []
    /// The warnings of the engine's settings (a map measured on another Mac or build, for example).
    /// (Set by a mirror from what the lending Mac published.)
    var settingsWarnings: [String] = []
    /// All the known models — the publisher's, then the imported ones (`Library.cards()`).
    var cards: [ModelCard] = []
    /// The ready models (all their files present).
    var readySet: Set<String> = []

    // ── What the app shows ──
    /// **The families in front**: the five the product is, **in the order every list shows them**
    /// (the rack's picker, the Models sheet, the welcome, `silicontrol models`) — not the engine's
    /// `Family.allCases`. A family added to the engine and not here comes after them (`displayRank`).
    static let frontFamilies: [Family] = [.zImage, .qwenImage21, .anima, .krea2, .mossImage]
    private static let developerModeKey = "developerMode"
    /// **More Solvers** (Settings ▸ Advanced ▸ Beta): the solvers (`offeredSolvers`). The property and its
    /// preference key keep the name they had when this was « Developer Mode ».
    /// One of the app's preferences (with `Curtain.mode` and the capacity
    /// map) — written without a render gesture, because the user switched it.
    var developerMode = UserDefaults.standard.bool(forKey: AppState.developerModeKey) {
        didSet {
            guard developerMode != oldValue else { return }
            UserDefaults.standard.set(developerMode, forKey: Self.developerModeKey)
            // A solver the models no longer offer leaves the form.
            if !developerMode {
                solver = .euler
                // The solver axis is the rack's choice swept: it leaves with it.
                if gridX.kind == .solver { gridX.kind = .seed }
                if gridY.kind == .solver { gridY.kind = .none }
            }
        }
    }
    private static let negativeLoRAKey = "negativeLoRAStrengths"
    /// **Negative LoRA strengths** (Experimental): W + s·ΔW with s < 0 pushes the image away from what
    /// the LoRA learned — meant for « slider » LoRAs, trained as a difference to be used from −x to +x;
    /// on an ordinary LoRA it mostly damages the image, hence the switch. The engine takes any finite
    /// strength (`LoRA.materialize` folds it into `down`, sign included): the gate is the app's, for
    /// the rack, the exploration and `silicontrol` alike. Off, the strengths already set come back to 0.
    var negativeLoRAStrengths = UserDefaults.standard.bool(forKey: AppState.negativeLoRAKey) {
        didSet {
            guard negativeLoRAStrengths != oldValue else { return }
            UserDefaults.standard.set(negativeLoRAStrengths, forKey: Self.negativeLoRAKey)
            if !negativeLoRAStrengths {
                loras = loras.map(allowedStrength)
                gridX.addedStrength = max(0, gridX.addedStrength)
                gridY.addedStrength = max(0, gridY.addedStrength)
            }
        }
    }
    private static let fasterRenderingKey = "fasterRenderingChoices"
    /// **Faster Rendering, per model and version** (Settings): at 1024² and above, the first evaluations of the family on the
    /// half grid (`Request.fasterRendering`; three of eight, Krea 2 four) — a quarter to a third faster, another image than the
    /// recipe at full size, judged clean by eye per family. Not with a LoRA, Detail or an edit (each plan's rules). What the user
    /// chose; absent, the version's default (`Variant.fasterRenderingByDefault`: off on every version since it became a
    /// Beta option, which gates it). A new key: the single switch of earlier builds is read once, at the first launch after
    /// the update (`adoptEarlierFasterRendering`), then removed.
    /// Every request carries the value and every history image keeps it (`RenderSettings.fasterRendering`): a redo is the same
    /// image whatever the switch says by then.
    var fasterRenderingChoices = UserDefaults.standard.dictionary(forKey: AppState.fasterRenderingKey) as? [String: Bool] ?? [:] {
        didSet { UserDefaults.standard.set(fasterRenderingChoices, forKey: Self.fasterRenderingKey) }
    }
    /// **Faster Rendering a render of `id` takes**: nothing without the Beta option, whatever the choice kept for the model;
    /// else its choice, else its version's default (`DiTSwitches.fasterRendering`). The one reading every render path uses.
    func fasterRendering(for id: String) -> Bool {
        DiTSwitches.fasterRendering(beta: fasterRenderingBeta, choice: fasterRenderingChoice(for: id), versionDefault: fasterRenderingDefault(for: id))
    }
    /// What the user chose for `id` (nil: nothing yet) — what the switch of the DiT box shows, Beta on or off.
    func fasterRenderingChoice(for id: String) -> Bool? {
        guard let card = cards.first(where: { $0.id == id }) else { return nil }
        return fasterRenderingChoices[choiceKey(card)]
    }
    private func fasterRenderingDefault(for id: String) -> Bool {
        guard let card = cards.first(where: { $0.id == id }), !card.isImported else { return false }
        return library.variant(of: card.family).fasterRenderingByDefault
    }
    /// What the switch of the DiT box shows: the model's choice, else the default — Beta aside.
    func fasterRenderingShown(for id: String) -> Bool { fasterRenderingChoice(for: id) ?? fasterRenderingDefault(for: id) }
    private static let fasterRenderingBetaKey = "fasterRenderingBeta"
    /// **Faster Rendering, the Beta option** (Settings ▸ Advanced ▸ Beta): off by default. Off, no render is a faster one and the
    /// DiT box shows no switch; the per-model choices stay in `fasterRenderingChoices` and come back with the option.
    var fasterRenderingBeta = UserDefaults.standard.bool(forKey: AppState.fasterRenderingBetaKey) {
        didSet { UserDefaults.standard.set(fasterRenderingBeta, forKey: Self.fasterRenderingBetaKey) }
    }
    private static let sizeScalesBetaKey = "sizeScalesBeta"
    /// **Size in two scales, the Beta option** (Settings ▸ Advanced ▸ Beta): off by default. On, the rack's Size menu becomes a
    /// proportion and a resolution (`SizeScales`). Only the rack changes: the sizes, the pencil, `silicontrol` are the same.
    var sizeScalesBeta = UserDefaults.standard.bool(forKey: AppState.sizeScalesBetaKey) {
        didSet { UserDefaults.standard.set(sizeScalesBeta, forKey: Self.sizeScalesBetaKey) }
    }
    /// Why Faster Rendering would do nothing for the render being set up in the form (nil: it applies).
    var fasterRenderingUnavailable: DiTSwitchUnavailable? {
        DiTSwitches.fasterRenderingUnavailable(width: format.width, height: format.height,
                                               withLoRA: !loras.isEmpty || detail != .normal, edit: !references.isEmpty,
                                               atRecipeSteps: steps == (card?.defaultSteps ?? 8))
    }
    func setFasterRendering(_ on: Bool, for id: String) {
        if let card = cards.first(where: { $0.id == id }) { fasterRenderingChoices[choiceKey(card)] = on }
    }
    /// **The single switch of the builds before the per-model choice** (« fasterRendering », on by default): a user who turned it off
    /// asked for the recipe at full size, and an update must not turn it back on behind them — each installed model that would default
    /// to on gets the choice off. On (the old default, or chosen) says nothing more than the versions' defaults do. Read once, removed.
    private static let earlierFasterRenderingKey = "fasterRendering"
    private func adoptEarlierFasterRendering() {
        let defaults = UserDefaults.standard
        guard let earlier = defaults.object(forKey: Self.earlierFasterRenderingKey) as? Bool else { return }
        defaults.removeObject(forKey: Self.earlierFasterRenderingKey)
        guard !earlier else { return }
        var choices = fasterRenderingChoices
        for card in cards where !card.isImported && choices[choiceKey(card)] == nil { choices[choiceKey(card)] = false }
        fasterRenderingChoices = choices
    }
    private static let exportFormatKey = "exportFormat"
    /// **The file format the last export used** (PNG, JPEG or HEIC): the next Save and Export open on it.
    var exportFormat = ExportFormat(rawValue: UserDefaults.standard.string(forKey: AppState.exportFormatKey) ?? "") ?? .png {
        didSet { UserDefaults.standard.set(exportFormat.rawValue, forKey: Self.exportFormatKey) }
    }
    /// The defaults key of the « Accelerators » choices: the Neural Engine's switch's, kept when it came to cover the GPU's tensor units
    /// too (2/11/2026) — a model whose Neural Engine was turned off has its accelerators off, nothing turned back on behind the user.
    private static let acceleratorsKey = "neuralEngine"
    /// **The accelerators, per model and version** (the DiT box's switch, `Request.accelerators`): what the user chose; absent, the
    /// version's default (`Variant.acceleratorsByDefault`, on everywhere).
    var acceleratorChoices = UserDefaults.standard.dictionary(forKey: AppState.acceleratorsKey) as? [String: Bool] ?? [:] {
        didSet { UserDefaults.standard.set(acceleratorChoices, forKey: Self.acceleratorsKey) }
    }
    /// **What the accelerators do on this Mac** (`AcceleratorOffer`), for a render whose switch is on: the Neural Engine where the map
    /// gives it rows, the GPU's tensor units where it chooses them. `nil` before the settings are read: `effective` read at launch would
    /// freeze the defaults and the first measure could no longer designate its plan (`EngineSettings.load` refuses it). Follows a new
    /// measure (`settingsEpoch`).
    var acceleratorOffer: AcceleratorOffer? {
        _ = settingsEpoch
        guard settingsRead else { return nil }
        let plan = EngineSettings.effective.plan
        return AcceleratorOffer(plan: plan.forDenoiser(.on(neuralEngine: plan.usesANE)))
    }
    /// This Mac's map chooses some accelerator: otherwise there is nothing to choose, the denoiser is fp32.
    var acceleratorsOffered: Bool { acceleratorOffer.map { !$0.isNone } ?? false }
    /// The key of a choice: the model and its installed version — another version starts from its own default.
    /// (Also the key of Faster Rendering's choice.)
    private func choiceKey(_ card: ModelCard) -> String {
        card.isImported ? card.id : "\(card.id) \(library.variant(of: card.family).rawValue)"
    }
    /// Whether `id`'s denoiser may use the accelerators: the user's choice, else its version's default — on everywhere, an imported
    /// model too.
    func accelerators(for id: String) -> Bool {
        guard let card = cards.first(where: { $0.id == id }) else { return false }
        return acceleratorChoices[choiceKey(card)] ?? (card.isImported || library.variant(of: card.family).acceleratorsByDefault)
    }
    func setAccelerators(_ on: Bool, for id: String) {
        if let card = cards.first(where: { $0.id == id }) { acceleratorChoices[choiceKey(card)] = on }
    }
    /// What the accelerators do for the model in the form, as its render will be told: `none` with its switch off.
    var acceleratorsInEffect: AcceleratorOffer {
        guard accelerators(for: identifier), let offer = acceleratorOffer else { return .none }
        return offer
    }
    /// The range of a LoRA's strength slider.
    var loraStrengthRange: ClosedRange<Double> { negativeLoRAStrengths ? -1.5...1.5 : 0...1.5 }
    /// `slot` with its strength at 0 if negative while the switch is off.
    func allowedStrength(_ slot: LoRASlot) -> LoRASlot {
        var s = slot
        if !negativeLoRAStrengths { s.strength = max(0, s.strength) }
        return s
    }
    /// A grid that sets a negative strength while the switch is off.
    func refusesNegativeStrength(_ g: RenderGrid) -> Bool {
        !negativeLoRAStrengths && g.cells.contains { $0.loraStrengths.contains { $0 < 0 } || ($0.addedLoRA != nil && $0.addedStrength < 0) }
    }
    /// Where a family comes in a list: the front ones in `frontFamilies`' order, the others after.
    static func displayRank(_ family: Family) -> Int {
        frontFamilies.firstIndex(of: family) ?? frontFamilies.count + (Family.allCases.firstIndex(of: family) ?? 0)
    }
    /// `items` in the display order of their family, each family's own order kept (the publisher's
    /// model before its imported fine-tunes).
    static func inDisplayOrder<T>(_ items: [T], family: (T) -> Family) -> [T] {
        items.enumerated()
            .sorted { (displayRank(family($0.element)), $0.offset) < (displayRank(family($1.element)), $1.offset) }
            .map(\.element)
    }
    /// The models the list offers: the front families' (and their imported fine-tunes), or all of them.
    var visibleCards: [ModelCard] { Self.inDisplayOrder(cards, family: \.family) }
    /// The visible models ready to render — empty at first launch: the canvas welcomes (`Welcome`).
    var visibleReady: [ModelCard] { visibleCards.filter { readySet.contains($0.id) } }
    /// Z-Image first (Apache 2.0); a non-commercial model is never the default unless it is the only one.
    private var defaultIdentifier: String { visibleReady.first?.id ?? ModelCard.zImage.id }

    // ── What the user sets ──
    var identifier = "" { didSet { if identifier != oldValue { modelChanged() } } }
    var prompt = ""
    var orientation = RecommendedFormat.Orientation.square { didSet { orientationChanged() } }
    var format = ModelCard.recommendedFormats[0]
    var loras: [LoRASlot] = []
    /// The edit's references, numbered from 1 in this order: **image 1 is the one edited** (it sets
    /// the output's format), the others are what the prompt borrows from (« the dog of image 2 »).
    /// At most the denoiser's `maxReferences`. Added by the box, a drop, « Edit This Image », or carried by
    /// the Prompt Lab (the starting image, the references, the subjects the text cites: `Lab.swift`).
    var references: [ReferenceImage] = []
    var steps = 8
    /// **More detail** (`Detail`): Normal is the render as the model makes it. What a render takes is
    /// `effectiveDetail`: on a model that offers none (Anima, Krea 2: `DenoisingModule.acceptsDetail`), Normal.
    var detail = Detail.normal
    /// Whether the form's model offers Detail — the rack hides the choice otherwise.
    var offersDetail: Bool { mirror != nil ? traits?.acceptsDetail ?? true : model(identifier)?.chain.denoising.acceptsDetail ?? true }
    var effectiveDetail: Detail { offersDetail ? detail : .normal }
    /// **The solver** (`Solver`), offered with More Solvers on the models in front
    /// (`offeredSolvers`). What a render takes is `effectiveSolver`: a choice the form's model does
    /// not offer is Euler.
    var solver = Solver.euler
    /// The families whose samplers walk every solver (`DenoisingModule.solvers`: Z-Image's,
    /// Qwen-Image-2.1's, Anima's since 0.10, Krea 2's since it is in front).
    static let solverFamilies: Set<Family> = [.zImage, .qwenImage21, .anima, .krea2]
    /// The solvers the rack offers: all four with More Solvers on those families, Euler alone otherwise.
    var offeredSolvers: [Solver] {
        developerMode && card.map({ Self.solverFamilies.contains($0.family) }) == true ? Solver.allCases : [.euler]
    }
    var effectiveSolver: Solver { offeredSolvers.contains(solver) ? solver : .euler }
    /// What turns the half-size first steps off besides a LoRA (`DenoisingModule.plan`).
    var plainSampling: Bool { effectiveDetail == .normal }
    /// Fixed seed (the field) or drawn at random at each render, like Draw Things. A variation
    /// belongs to its seed: changing the seed, or letting it be drawn, drops it.
    var fixedSeed = false { didSet { if !fixedSeed { variations = [] } } }
    var seed: UInt64 = 42 { didSet { if seed != oldValue { variations = [] } } }
    /// **The variations the next render takes** (`Request.variations`) — only ever set by reusing a
    /// variation's settings (its menu, a dropped PNG), shown in the rack with a button that removes them.
    var variations: [Variation] = []
    var batch = 1
    var previews = true
    /// **The named subjects** (`Subjects.swift`): the Prompt Lab's library, in memory. The rack does not read them.
    let subjects: SubjectLibrary
    /// **The exploration's form** (`Exploration.swift`): its own prompt, a small format, the two axes.
    var explorePrompt = ""
    var exploreShape = ExploreShape.square
    var gridX = GridAxisForm(kind: .seed)
    var gridY = GridAxisForm(kind: .none)
    /// **The exploration shown**: its cells as they arrive, kept once ended — until the next one.
    var explored: GridRun?
    /// The canvas shows the exploration's grid rather than an image (`ExplorationCanvas`).
    var showsExploration = false
    /// A grid cell's long side on the canvas, in points: the slider, the pinch, ⌘+ ⌘−.
    var exploreCellSide: CGFloat = 220
    /// **Where the exploration's sketches stop**: 0, where the image is readable (`Sketch`); n, after n
    /// evaluations — more for a LoRA, whose style can settle late; at the plan's count, finished images.
    var exploreStop = 0
    static let cellSides: ClosedRange<CGFloat> = 96...1024
    static func clampedCellSide(_ s: CGFloat) -> CGFloat { min(max(s, cellSides.lowerBound), cellSides.upperBound) }

    // ── The chosen model's catalog: read from disk when it changes, not at each drawing ──
    /// The chosen model's chain — what the boxes draw. `nil` if it is not installed.
    private(set) var chain: Chain?
    /// The LoRAs forged for the chosen model.
    var compatibleLoras: [LoRACard] = []
    /// What the chosen model is missing (the library's message), or `nil` if it is ready.
    var missing: String?

    /// Re-reads from disk what depends on the chosen model: its chain, its LoRAs, what it is missing.
    func readModelCatalog() {
        if mirror != nil { readMirroredCatalog(); return }
        chain = model(identifier)?.chain
        compatibleLoras = library.loras(for: identifier)
        missing = readySet.contains(identifier) ? nil : card?.missing(in: library)
    }

    // ── What the render shows ──
    /// The running render, then those waiting — in the order they were requested.
    var currentJob: Job?
    /// The figures of the running render and of the last finished one — the statistics window.
    var measurements = Measures()
    /// **Seconds per evaluation**, per model and format, learned in this session (never
    /// written): that is what estimates the queue and the "Generate" button. A pair never rendered has
    /// no estimate.
    var speeds: [String: Speed] = [:]
    /// What the session cost: images, and seconds of computing.
    var sessionImages = 0
    var sessionSeconds = 0.0
    var file: [Job] = []
    var inProgress: Bool { currentJob != nil }
    /// A render is running or waiting: forges (several GB) and removing a model wait.
    var busy: Bool { inProgress || !file.isEmpty }
    var fraction = 0.0
    /// The time remaining at the last step or decoding, and when: between two events (a Krea 2
    /// step lasts 35 s), `remaining(at:)` counts it down instead of letting the predicted end slide.
    var estimatedRemaining: Double?
    var remainingInstant = Date()
    var stage: Engine.Stage?
    var currentStep = 0
    var totalSteps = 0
    var batchImage = 0
    var batchSize = 1
    var preview: CGImage?
    var history: [Entry] = []
    var selection: Entry.ID? {
        didSet {
            guard selection != oldValue else { return }
            loadSelectedImage()
            // An image chosen (or a finished render followed): the canvas shows it, not the grid.
            if selection != nil { showsExploration = false; cropping = nil }
        }
    }
    /// **The images marked in the strip**, beyond the one shown: ⌘-click adds or removes one, ⇧-click
    /// and ⇧-arrow mark a run, ⌘A marks all. Meaningful from two on — what Export, Copy, Share and
    /// Remove then act on (`targets`); a plain click or a new image clears it.
    var marked: Set<Entry.ID> = [] { didSet { if marked != oldValue { loadCompared() } } }
    /// **Two to four images marked are shown side by side** on the canvas — to choose between seeds —,
    /// decoded once per marking, like the chosen image.
    var compared: [(entry: Entry, image: CGImage)] = []
    static let maxCompared = 4
    /// **An image against its image 1** (the bar's curtain button): the original under the output, at
    /// the output's size. Stays on from one such image to the next; an image made from none shows alone.
    var comparesOriginal = false { didSet { loadOriginal() } }
    var original: (id: Entry.ID, image: CGImage)?
    var showsOriginal: Bool {
        comparesOriginal && marked.count <= 1 && original != nil && original?.id == selectedEntry?.id
    }
    /// Any image the model read can be the original — an edit's or a named subject's, framed as it was
    /// sent: « @ana as a western comic » keeps the picture's framing as an edit does.
    /// An enlargement's original is the image it was made from, stretched to its size.
    /// In a mirror, an edit's images stayed on the other Mac (`.remote`): no original to lay under it.
    func canCompareOriginal(_ e: Entry) -> Bool {
        e.grid == nil && (e.references.contains { if case .remote = $0.source { false } else { true } } || e.enlargedFrom != nil)
    }
    /// The thumbnails' height in the strip (the slider at its end); in memory, like the rest.
    var thumbnailSide: Double = 64
    /// Where a ⇧-run starts: the last image clicked without ⇧.
    var anchor: Entry.ID?
    /// **The images that left the app** — saved, exported, written by `silicontrol save`. The others
    /// exist only in this process: quitting asks before losing them.
    var saved: Set<Entry.ID> = []
    var unsaved: [Entry] { history.filter { !saved.contains($0.id) } }
    /// The window's undo manager: removing images from the history is undone with ⌘Z.
    weak var undoManager: UndoManager?
    /// The canvas's zoom on the chosen image (1: fitted) — the pinch, and ⌘+ ⌘− ⌘0.
    var zoom: CGFloat = 1
    /// **The canvas follows the running render** — its preview, then each image it makes —, until a
    /// history image is chosen during it (`show`): the history is browsed while the render goes on, its
    /// images join the strip without taking the canvas. « Generate » and the live thumbnail follow it
    /// again.
    var followsRender = true
    /// The chosen image, decoded from its PNG — the only full-size image the app keeps.
    var selectedImage: (id: Entry.ID, image: CGImage)?
    // **The decodings are off the main thread** (`Entry.decoded`): each loader keeps its task, a
    // newer request cancels the older, and a result lands only if it is still the one asked for. The
    // canvas keeps what it showed until then — tens of milliseconds.
    @ObservationIgnored var comparedLoad: Task<Void, Never>?
    @ObservationIgnored var selectedLoad: Task<Void, Never>?
    @ObservationIgnored var originalLoad: Task<Void, Never>?
    var journal: [String] = []
    /// The last refusal or failure, translated (`Problems.swift`); closed by the user.
    var problem: Problem?
    /// **Each stage of the last render, as the engine's events moved it** — the rack's lights, for the
    /// model of that render. Reset when a job starts.
    var stageStates: [Engine.Stage: StageState] = [:]
    /// **What each finished stage of that render took**, set the moment it ends — the box shows it
    /// instead of the estimate, without waiting for the whole render (`stageCost`).
    var stageSeconds: [Engine.Stage: Double] = [:]
    /// The model the lights belong to: those of another model, chosen since, stay off.
    var stageModel: String?
    /// **The license the engine asked for** — the sheet that shows it (`LicenseSheet.swift`).
    var licenseRequest: LicenseRequest?
    /// The jobs whose model's license the engine refused: they wait for the user's answer — out of the
    /// queue, so `silicontrol status` lists them apart (`"waiting_for_license"`).
    var awaitingLicense: [Job] = []
    /// **The family whose installation the Models sheet offers**: its license and its place on disk
    /// shown, waiting for « Accept and Install » — nothing is downloaded before.
    var installOffer: InstallOffer?
    struct InstallOffer: Equatable {
        let family: Family
        /// False: only what an imported model of the family lacks (encoder, VAE), not its DiT.
        var baseDiT = true
        /// The version to install, chosen in the panel: the one installed (else the preselected one)
        /// unless the user picks another — never another one when `baseDiT` is false (`offerInstall`).
        var variant: Variant = .standard
        /// The files dropped before their family was there: imported once this installation is done.
        /// Cancelling the offer, or stopping the installation, forgets them with it.
        var thenImport: [URL] = []
        /// In a mirror: the LoRA of the Hub the lender imports once the family is installed there.
        var thenHub: RemoteHubFile?
    }
    /// **The « Use this model » link being answered** (`HubSheet`): the repository, its files judged.
    var hubOffer: HubOffer?
    /// The diagnostic sheet, and where the diagnostic stands.
    var diagnosticOpen = false
    var diagnostic = DiagnosticPhase.ready
    /// The supervisor of the measuring child (`Diagnostic.Child`), while it runs; `diagnosticStopped` when Stop ended it.
    var diagnosticSupervisor: Diagnostic.Child.Supervisor?
    var diagnosticStopped = false
    var diagnosticRunning: Bool { if case .running = diagnostic { true } else { false } }
    /// The chosen image full screen, in the window (never a file opened elsewhere).
    var fullScreen = false
    var toast: String?
    /// An installation or an import in progress (several minutes, several GB) — and what it says.
    var forgeInProgress: String?
    var forgeJournal: [String] = []
    /// What the forge in progress is, by identifiers: what a borrowing Mac is told (`RemoteForge`).
    var forgeTask: RemoteForge.Task = .other
    /// In a mirror: the other Mac's library and the installation under way there (`MirrorLibrary.swift`).
    var remoteLibrary: RemoteLibrary?
    var remoteForge: RemoteForge?
    /// In a mirror: a removal asked waits for a click on the lender (`RemoteLibraryNews.confirming`).
    var remoteConfirming = false
    /// The token of the forge in progress: "Stop" cuts a download in the middle.
    var cancellationForge: Cancellation?
    /// What the library occupies — the "Models" sheet; re-read with the catalog.
    var occupancy: Library.Occupancy?
    /// The "Models" sheet is open (⇧⌘M, or an installation offered: `offerInstall`).
    var managementOpen = false

    var task: Task<Void, Never>?
    var toastTask: Task<Void, Never>?
    /// **No App Nap during a render**: in the background (another app in the foreground, or
    /// the app opened by `silicontrol open`), macOS throttles the app — 64 s instead of 44
    /// for a measured 512². Held from the first job launched until the queue is empty.
    var activity: NSObjectProtocol?

    // ── What the remote control reads (`RemoteControl.swift`) ──
    private(set) var lastJob = 0
    var lastImage = 0
    /// **« Enlarge »** (`Enlarge.swift`): the image being enlarged, the model's download (0…1) while
    /// macOS fetches it, and the seconds per input megapixel learned this session.
    var enlarging: Entry.ID?
    var enlargeDownload: Double?
    var enlargeSpeed: Double?
    /// **The crop being made** (`CropEditor.swift`): the frame over an image of the history or an edit's reference.
    var cropping: CropSession?
    /// The session's finished jobs, their outcome, their images (even removed from the history
    /// since) and the engine's warnings during their render: `wait t3` afterwards still returns the answer.
    private(set) var completed: [Int: (job: Job, issue: Issue, images: [Int], warnings: [String])] = [:]
    var jobImages: [Int] = []
    /// **The grids whose cells are still on their way**, and which cell each job is.
    var gridRuns: [Int: GridRun] = [:]
    var gridCells: [Int: (grid: Int, cell: Int)] = [:]
    /// The session's ended grids: their cells' jobs, and their sheet (`i9`), if one was drawn.
    var endedGrids: [Int: (jobs: [Int], sheet: Int?)] = [:]
    /// The grids whose cells have all ended and whose sheet is being drawn, with their cells' jobs:
    /// out of `gridRuns`, not yet in `endedGrids` — `wait g2` and `follow` must still see them.
    var drawingSheets: [Int: [Int]] = [:]
    var lastGrid = 0
    /// Who listens to the signals, by token. Called on the main thread, in order.
    private var subscribers: [UUID: (Signal) -> Void] = [:]

    func subscribe(_ listener: @escaping (Signal) -> Void) -> UUID {
        let token = UUID()
        subscribers[token] = listener
        return token
    }
    func unsubscribe(_ token: UUID) { subscribers[token] = nil }
    func signal(_ s: Signal) { for listener in subscribers.values { listener(s) } }

    /// In a mirror whose lender restarted: the ends it told were numbered by a process that is gone.
    func forgetCompletedJobs() { completed = [:] }

    func complete(_ t: Job, _ issue: Issue, images: [Int] = [], warnings: [String] = []) {
        completed[t.ordinal] = (t, issue, images, warnings.map(Library.withoutHome))
        signal(.finished(t, issue))
        if let at = gridCells.removeValue(forKey: t.ordinal), gridRuns[at.grid] != nil {
            gridRuns[at.grid]!.issues[at.cell] = issue
            if explored?.ordinal == at.grid { explored = gridRuns[at.grid] }
            if gridRuns[at.grid]!.isComplete { finishGrid(at.grid) }
        }
    }

    func nextJobNumber() -> Int { lastJob += 1; return lastJob }

    static let maxLoRA = 3
    /// A prompt's alternatives queue one render each, up to `PromptAlternatives.maxCombinations`:
    /// the queue holds that many, so that the largest series fits in an empty queue.
    static let maxQueueLength = PromptAlternatives.maxCombinations

    // ── A mirror (`AppState+Mirror.swift`) ──
    /// Set on the instance that shows another Mac: its actions go there instead of running here.
    @ObservationIgnored var mirror: MirrorLink?
    /// What the other Mac published of its models.
    var mirrorCatalog: MirrorCatalog?
    /// The other Mac's chip, map and diagnostic (`MirrorMachine.swift`).
    var mirrorMachine: RemoteMachine?
    /// What the other Mac renders for others (itself, other Macs): how many, and for how long (`nil`: unknown).
    var othersQueued = 0
    var othersSeconds: Double?
    /// **Why nothing can be sent now**, in a mirror: the session with the other Mac is not joined (connecting,
    /// lost and tried again, refused). A request made then would be dropped without a word (`RemoteLink.send`),
    /// so « Generate » waits for the link (`formBlocker`). `nil`: joined.
    var linkBlocker: String?
    /// What the other Mac answered to the form's questions (`quoted`).
    var mirrorQuotes: [RemoteQuote: RemoteQuoteAnswer] = [:]
    /// What the other Mac estimates each waiting job of this Mac's takes, by ordinal (`RemoteStatistics`):
    /// the speeds learned there are not a plan's to compute here.
    var mirrorEstimates: [Int: Double] = [:]

    /// **The state of another Mac**, shown here (`AppState+Mirror.swift`): nothing read from this Mac's
    /// library but the named subjects, which stay with the user (this Mac's own library, shared: `silicontrol
    /// subjects` and the chips act on the same ones in either view); no capacity map, no copy, no warm-up.
    init(mirroring link: MirrorLink, library: Library, subjects: SubjectLibrary) {
        self.library = library
        self.subjects = subjects
        mirror = link
        // Joined only when its `connected` comes (`linkChanged`).
        linkBlocker = String(localized: "Connecting to “\(link.macName)”…")
        link.down = linkBlocker
        cards = ModelCard.allCards
        identifier = defaultIdentifier
    }

    init(library: Library) {
        self.library = library
        libraryReachable = library.isReachable
        WorkingCopy.removeAbandoned()
        workingCopy = Self.makeWorkingCopy(library)
        subjects = SubjectLibrary(library: library)
        cards = library.cards()
        readySet = Set(library.models().map(\.id))
        occupancy = library.occupancy()
        identifier = defaultIdentifier
        adoptEarlierFasterRendering()
        readEditors()
        // The engine's settings, read with the capacity map — once per process, and only
        // with a current map: the app builds only one `AppState` (see `App.swift`). Without one (none yet,
        // or one of another Mac, macOS or build: an update), the settings stay unread until the measure
        // the first render starts (`next`). **A map of this Mac and this build is read as it stands**: what the measure could not
        // give is said in its findings and the plan does without it, how far its identical calls agreed is written beside each
        // engine, and nothing asks to measure again. A render's gap to the prediction never asks for a new one either: the plan
        // stays what the model made it, and the bits of an image with it.
        let stored = Self.readStoredMap()
        storedMap = stored
        if let stored, stored.isCurrent {
            readSettings(map: stored)
            // The last render's note against this very map travels with the next report.
            renderNote = Self.readRenderNote(of: stored)
        }
        watchVolumes()
        warm()
    }

    // MARK: Capacity map

    /// **The capacity model, in the app's preferences** (`Diagnostic.Report`, its JSON): what the engine plans with (`Report.plan`).
    /// **No render without one** (the user, 18/10/2026): the defaults are the M1 Pro's, and only a measure says what another
    /// Mac's engines are worth. So the first render measures first — when none is kept, or the one kept is of another schema or
    /// another `fingerprint` (another Mac, macOS build or app build) —, said in the sidebar, never in the background; the user can
    /// also measure again (« Measure This Mac »), which then applies to the next render.
    private static let capacityMapKey = "capacityMap"

    /// The stored report; an older schema is not a map (`Report.read`).
    private static func readStoredMap() -> Diagnostic.Report? {
        UserDefaults.standard.data(forKey: capacityMapKey).flatMap { Diagnostic.Report.read($0) }
    }

    /// The last render's note, if it was made against this very report (a new measure drops it).
    private static func readRenderNote(of report: Diagnostic.Report) -> RenderLoop.Note? {
        UserDefaults.standard.data(forKey: renderNoteKey).flatMap { try? JSONDecoder().decode(RenderLoop.Note.self, from: $0) }
            .flatMap { $0.map == report.date ? $0 : nil }
    }

    /// **Checks a finished render against the model it ran under**: its note is shown by the report and kept beside it.
    /// Never changes the plan itself: a new measure, taken by the user, does — from the next render.
    func checkMap(_ render: Engine.Render, model: Model, references: Int) {
        guard let map = mapInUse,
              let check = RenderLoop.check(render, variant: model.library?.variant(of: model.family) ?? .standard,
                                           references: references, plan: EngineSettings.effective.plan, mapDate: map.date,
                                           thresholds: .init(uncertainty: map)) else { return }
        renderNote = check.note
        if let data = try? JSONEncoder().encode(check.note) { UserDefaults.standard.set(data, forKey: Self.renderNoteKey) }
    }

    /// The map kept in the preferences, current or not.
    private(set) var storedMap: Diagnostic.Report?
    /// The map this process's plan comes from — `nil`: not read yet, or read without one (a refusal, `settingsWarnings`).
    /// It changes only between two renders (`EngineSettings.adopt`): a render reads one plan from start to end, and the
    /// next one may differ from the last — another measure of the same Mac can give an image a little different from one seed.
    private(set) var mapInUse: Diagnostic.Report?
    /// Counts the plan's replacements: what the needs, the offer of the Neural Engine and the catalog are recomputed from.
    private(set) var settingsEpoch = 0
    /// **The engine's settings were read** (`EngineSettings.load`), the map with them: renders may run,
    /// and the memory needs be computed — they read the settings (`amx` adds the AMX's operands).
    var settingsRead = false
    /// **The last render against the model it ran under** (`RenderLoop`): what « Report my configuration » carries. It acts on nothing.
    @ObservationIgnored private(set) var renderNote: RenderLoop.Note?
    private static let renderNoteKey = "capacityMapRenderNote"

    /// Keeps `report` as this Mac's map, and the engine's plan from the next render (at once when no render runs).
    func keep(_ report: Diagnostic.Report) {
        guard let data = try? report.json(pretty: false) else { return }
        UserDefaults.standard.set(data, forKey: Self.capacityMapKey)
        if report.date != renderNote?.map { renderNote = nil; UserDefaults.standard.removeObject(forKey: Self.renderNoteKey) }
        storedMap = report
        if !settingsRead { readSettings(map: report) } else { EngineSettings.adopt(map: report); syncPlan() }
    }

    /// **Before a job starts** (`next`): a measure that finished during a render becomes the plan now, so that the render that
    /// follows reads it whole.
    func applyPendingMap() {
        guard settingsRead, currentJob == nil else { return }
        if EngineSettings.applyPendingMap() { syncPlan() }
    }

    /// The map in use is the one the engine's plan comes from; what showed a need follows.
    private func syncPlan() {
        let date = EngineSettings.effective.mapRead
        if let date, let stored = storedMap, stored.date == date, mapInUse?.date != date { mapInUse = stored }
        if EngineSettings.generation != planGeneration { planGeneration = EngineSettings.generation; settingsEpoch += 1 }
        settingsWarnings = EngineSettings.effective.warnings.map(Library.withoutHome)
    }
    @ObservationIgnored private var planGeneration = 0

    /// **Reads the engine's settings**, once: `map`, and what it warns of.
    private func readSettings(map: Diagnostic.Report) {
        do {
            try EngineSettings.load(map: map)
            settingsWarnings = EngineSettings.effective.warnings.map(Library.withoutHome)
            mapInUse = map
            planGeneration = EngineSettings.generation
        } catch {
            settingsWarnings = [Library.withoutHome(error.localizedDescription)]
        }
        settingsRead = true
    }

    // MARK: Catalog

    /// The installed models that edit, read once per catalog: where « Edit with… » sends an image.
    private(set) var editingModels: Set<String> = []
    func readEditors() {
        editingModels = readySet.filter { (model($0)?.chain.denoising.maxReferences ?? 0) > 0 }
    }

    /// **The models, read once per catalog**: `Model.named` reads the map headers on disk, and the
    /// exploration's footer asks for each cell's model at every drawing. Emptied by `refreshCatalog`.
    @ObservationIgnored var modelsRead: [String: Model?] = [:]

    /// **B, this Mac's memory budget** (`MemoryBudget.machineCeiling()`: ¾ · RAM − 2 GiB, or
    /// `SILICONED_MEMORY_BUDGET_GB`), read once: it depends on the RAM and nothing else, so what this
    /// Mac cannot do is known ahead and greyed out where it is chosen — the size, the images — never
    /// discovered at the click, and never different from one minute to the next.
    /// B: this Mac's, or the lending Mac's in a mirror (`AppState+Mirror.swift`).
    var memoryCeiling = MemoryBudget.machineCeiling()
}
