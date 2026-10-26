// **The app's state**: what the user sets, what the render shows, the history.
//
// **Incognito by default**: the history lives in memory and dies with the
// process. Nothing is written to disk without an explicit gesture — "Save", "Copy" —,
// no temporary file, no preference, no log: nothing that Spotlight or recent items
// could find. **The exceptions**, each a gesture of the user: five preferences — the « Developer
// Mode » switch, the comparison's curtain or side by side (`Curtain.preferred`), the experimental
// « Named Subjects » and « Negative LoRA Strengths » switches (`SubjectLibrary.enabledKey`,
// `negativeLoRAKey`) and this Mac's capacity map, measured when
// the user asks (`capacityMapKey`), like the system language the app follows —, the licenses accepted, which the library keeps itself (`<local>/accepted-licenses.json`) because its
// engine refuses a render without them, and where the library is when the user moved it (`<local>/library-location`,
// `AppState+Place.swift`). A library on another disk is rendered through a copy on this Mac, removed when the
// app quits (`WorkingCopy`). The named subjects (`Subjects.swift`) live in memory, like
// the history.
//
// **Four models in front**, in this order: Z-Image Turbo, Qwen-Image-2.1 Turbo, Anima Turbo, Krea 2
// Turbo (`frontFamilies`) — every family the library knows since Krea 2 joined them. The filter
// stays here, in the app (the library is the same): a family left out of `frontFamilies` would show
// in the list, the Models sheet and `silicontrol models` only in Developer Mode. Developer Mode offers
// the solvers on the models in front (`offeredSolvers`): Euler, the distillation's own, is every
// other render's.
//
// **The main thread does no computing**: the engine runs on its queue, and what the app does with
// the pixels — decoding the reference image, encoding the PNG, reducing the thumbnail — goes to a
// detached task. The catalog (ready models, compatible LoRAs, what is missing) is read from disk when
// it changes, never at each drawing of the view.
//
// **All the phrases go through `String(localized:)`** or through a `Text` key: the table is
// `Localizable.xcstrings` (English source, fr de es it), which `tools/app.sh` synchronizes and compiles.
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
    /// The warnings of the machine profile (a profile measured elsewhere, for example).
    private(set) var profileWarnings: [String] = []
    /// All the known models — the publisher's, then the imported ones (`Library.cards()`).
    var cards: [ModelCard] = []
    /// The ready models (all their files present).
    var readySet: Set<String> = []

    // ── What the app shows ──
    /// **The families in front**: the four the product is, **in the order every list shows them**
    /// (the rack's picker, the Models sheet, the welcome, `silicontrol models`) — not the engine's
    /// `Family.allCases`. The others need Developer Mode, and come after them.
    static let frontFamilies: [Family] = [.zImage, .qwenImage21, .anima, .krea2, .mossImage]
    private static let developerModeKey = "developerMode"
    /// **Developer Mode**: the solvers (`offeredSolvers`), ComfyUI's schedule — and a family not in
    /// front, were there one (none since Krea 2 joined them).
    /// One of the app's preferences (with `Curtain.preferred`, the named subjects' switch and the capacity
    /// map) — written without a render gesture, because the user switched it.
    var developerMode = UserDefaults.standard.bool(forKey: AppState.developerModeKey) {
        didSet {
            guard developerMode != oldValue else { return }
            UserDefaults.standard.set(developerMode, forKey: Self.developerModeKey)
            // A model that leaves the list leaves the form too, and so does a solver.
            if !isVisible(identifier) { identifier = defaultIdentifier }
            if !developerMode {
                solver = .euler; comfySchedule = false
                // The solver axis is the rack's choice swept: it leaves with it.
                if gridX.kind == .solver { gridX.kind = .seed }
                if gridY.kind == .solver { gridY.kind = .none }
            }
        }
    }
    func isVisible(_ family: Family) -> Bool { developerMode || Self.frontFamilies.contains(family) }
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
    private static let fasterRenderingKey = "fasterRendering"
    /// **Faster Rendering** (Settings, on by default): at 1024² and above, the first evaluations of every
    /// family on the half grid (`Spectral.halfSizeStart`; three of eight, Krea 2 four) — a quarter to a
    /// third faster, another image than the recipe at full size, judged clean by eye per family. Not
    /// with a LoRA, Detail or an edit (each plan's rules). The engine reads it at every plan, so time
    /// estimates follow at once.
    var fasterRendering = UserDefaults.standard.object(forKey: AppState.fasterRenderingKey) as? Bool ?? true {
        didSet {
            guard fasterRendering != oldValue else { return }
            UserDefaults.standard.set(fasterRendering, forKey: Self.fasterRenderingKey)
            Spectral.halfSizeStart = fasterRendering
        }
    }
    private static let exportFormatKey = "exportFormat"
    /// **The file format the last export used** (PNG, JPEG or HEIC): the next Save and Export open on it.
    var exportFormat = ExportFormat(rawValue: UserDefaults.standard.string(forKey: AppState.exportFormatKey) ?? "") ?? .png {
        didSet { UserDefaults.standard.set(exportFormat.rawValue, forKey: Self.exportFormatKey) }
    }
    private static let neuralEngineKey = "neuralEngine"
    /// **The Neural Engine, per model and version** (the rack's fp32 tag, `Request.neuralEngine`): what the
    /// user chose; absent, the version's default (`Variant.neuralEngineByDefault`).
    var neuralEngineChoices = UserDefaults.standard.dictionary(forKey: AppState.neuralEngineKey) as? [String: Bool] ?? [:] {
        didSet { UserDefaults.standard.set(neuralEngineChoices, forKey: Self.neuralEngineKey) }
    }
    /// This Mac's map turned the Neural Engine on: otherwise there is nothing to choose, the denoiser is fp32.
    /// Not before the settings are read: `effective` read at launch would freeze the defaults, and the measure
    /// the first render starts would apply only at the next launch (`EngineSettings.load` refuses it).
    var neuralEngineOffered: Bool { settingsRead && EngineSettings.effective.gemmANE }
    /// The key of a choice: the model and its installed version — another version starts from its own default.
    private func neuralEngineKey(_ card: ModelCard) -> String {
        card.isImported ? card.id : "\(card.id) \(library.variant(of: card.family).rawValue)"
    }
    /// Whether `id`'s denoiser may take the Neural Engine: the user's choice, else its version's default;
    /// an imported model is the Standard's case.
    func neuralEngine(for id: String) -> Bool {
        guard let card = cards.first(where: { $0.id == id }) else { return false }
        return neuralEngineChoices[neuralEngineKey(card)] ?? (!card.isImported && library.variant(of: card.family).neuralEngineByDefault)
    }
    func setNeuralEngine(_ on: Bool, for id: String) {
        if let card = cards.first(where: { $0.id == id }) { neuralEngineChoices[neuralEngineKey(card)] = on }
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
    func isVisible(_ id: String) -> Bool { cards.first { $0.id == id }.map { isVisible($0.family) } ?? false }
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
    var visibleCards: [ModelCard] { Self.inDisplayOrder(cards.filter { isVisible($0.family) }, family: \.family) }
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
    /// At most the denoiser's `maxReferences`. An image added while the box shows the subjects brings
    /// it back to its images (`referenceMode`): « Edit This Image », a drop on the canvas.
    var references: [ReferenceImage] = [] {
        didSet { if !references.isEmpty && referenceMode == .subjects { stashedReferences = []; referenceMode = .images } }
    }
    /// **The Images box holds images or subjects, never both** (`Subjects.swift`): numbered images —
    /// an edit —, or the named subjects the prompt cites by `@name`. Switching to the subjects puts the
    /// images aside, switching back brings them again; a render sends only what the box shows.
    enum ReferenceMode { case images, subjects }
    var referenceMode = ReferenceMode.images {
        didSet {
            guard referenceMode != oldValue else { return }
            if referenceMode == .subjects {
                stashedReferences = references
                references = []
            } else if references.isEmpty {
                references = stashedReferences
                stashedReferences = []
            }
        }
    }
    @ObservationIgnored private var stashedReferences: [ReferenceImage] = []
    var steps = 8
    /// **More detail** (`Detail`): Normal is the render as the model makes it. What a render takes is
    /// `effectiveDetail`: on a model that offers none (Anima, Krea 2: `DenoisingModule.acceptsDetail`), Normal.
    var detail = Detail.normal
    /// Whether the form's model offers Detail — the rack hides the choice otherwise.
    var offersDetail: Bool { model(identifier)?.chain.denoising.acceptsDetail ?? true }
    var effectiveDetail: Detail { offersDetail ? detail : .normal }
    /// **The solver** (`Solver`), offered in Developer Mode on the models in front
    /// (`offeredSolvers`). What a render takes is `effectiveSolver`: a choice the form's model does
    /// not offer is Euler.
    var solver = Solver.euler
    /// The families whose samplers walk every solver (`DenoisingModule.solvers`: Z-Image's,
    /// Qwen-Image-2.1's, Anima's since 0.10, Krea 2's since it is in front).
    static let solverFamilies: Set<Family> = [.zImage, .qwenImage21, .anima, .krea2]
    /// The solvers the rack offers: all four in Developer Mode on those families, Euler alone otherwise.
    var offeredSolvers: [Solver] {
        developerMode && card.map({ Self.solverFamilies.contains($0.family) }) == true ? Solver.allCases : [.euler]
    }
    var effectiveSolver: Solver { offeredSolvers.contains(solver) ? solver : .euler }
    /// **ComfyUI's σ instead of the official ones** (`SigmaSchedule`), to compare: a switch in Developer
    /// Mode, on the models whose `DenoisingModule.schedules` offer it (Qwen-Image-2.1). What a render
    /// takes is `effectiveSchedule`.
    var comfySchedule = false
    var offersComfySchedule: Bool {
        developerMode && model(identifier)?.chain.denoising.schedules.contains(.comfyUI) == true
    }
    var effectiveSchedule: SigmaSchedule { offersComfySchedule && comfySchedule ? .comfyUI : .official }
    /// What turns the half-size first steps off besides a LoRA (`DenoisingModule.plan`).
    var plainSampling: Bool { effectiveDetail == .normal && effectiveSchedule == .official }
    /// Fixed seed (the field) or drawn at random at each render, like Draw Things. A variation
    /// belongs to its seed: changing the seed, or letting it be drawn, drops it.
    var fixedSeed = false { didSet { if !fixedSeed { variations = [] } } }
    var seed: UInt64 = 42 { didSet { if seed != oldValue { variations = [] } } }
    /// **The variations the next render takes** (`Request.variations`) — only ever set by reusing a
    /// variation's settings (its menu, a dropped PNG), shown in the rack with a button that removes them.
    var variations: [Variation] = []
    var batch = 1
    var previews = true
    /// **The named subjects** (`Subjects.swift`, experimental): the library, its switch — and the sheet
    /// that names or edits one, while it is open.
    let subjects: SubjectLibrary
    var subjectDraft: SubjectDraft?
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
    private(set) var compatibleLoras: [LoRACard] = []
    /// What the chosen model is missing (the library's message), or `nil` if it is ready.
    private(set) var missing: String?

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
    func canCompareOriginal(_ e: Entry) -> Bool { e.grid == nil && (!e.references.isEmpty || e.enlargedFrom != nil) }
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
    }
    /// **The « Use this model » link being answered** (`HubSheet`): the repository, its files judged.
    var hubOffer: HubOffer?
    /// The diagnostic sheet, and where the diagnostic stands.
    var diagnosticOpen = false
    var diagnostic = DiagnosticPhase.ready
    /// The measuring child (`CapacityMeasure`), while it runs; `diagnosticStopped` when Stop ended it.
    var diagnosticProcess: Process?
    var diagnosticStopped = false
    var diagnosticRunning: Bool { if case .running = diagnostic { true } else { false } }
    /// The chosen image full screen, in the window (never a file opened elsewhere).
    var fullScreen = false
    var toast: String?
    /// An installation or an import in progress (several minutes, several GB) — and what it says.
    var forgeInProgress: String?
    var forgeJournal: [String] = []
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
    private var lastJob = 0
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
        Spectral.halfSizeStart = fasterRendering
        readEditors()
        // The machine profile, read in THIS library, with the capacity map — once per process, and only
        // with a current map: the app builds only one `AppState` (see `App.swift`). Without one (none yet,
        // or one of another Mac, macOS or build: an update), the settings stay unread until the measure
        // the first render starts (`next`).
        let stored = Self.readStoredMap()
        storedMap = stored
        if let stored, stored.isCurrent { readSettings(map: stored) }
        watchVolumes()
        warm()
    }

    // MARK: Capacity map

    /// **The capacity map, in the app's preferences** (`Diagnostic.Report`, its JSON): what the engine
    /// branches on (`CapacityRules.swift`). **No render without one** (the user, 18/10/2026): the defaults
    /// are the M1 Pro's, and only a measure says what another Mac's GPU and AMX are worth. So the first
    /// render measures first — when none is kept, or the one kept has another `fingerprint` (another
    /// Mac, macOS build or app build) —, said in the sidebar, never in the background; the user can
    /// also measure again (« Measure This Mac »), which then applies from the next launch.
    private static let capacityMapKey = "capacityMap"

    private static func readStoredMap() -> Diagnostic.Report? {
        UserDefaults.standard.data(forKey: capacityMapKey).flatMap { try? JSONDecoder().decode(Diagnostic.Report.self, from: $0) }
    }

    /// The map kept in the preferences, current or not.
    private(set) var storedMap: Diagnostic.Report?
    /// The map this process's settings were read with — `nil`: not read yet, or read without one (a
    /// refusal, `profileWarnings`). It never changes once set: the settings are read once per process,
    /// and a render's bits must not move mid-session.
    private(set) var mapInUse: Diagnostic.Report?
    /// **The engine's settings were read** (`EngineSettings.load`), the map with them: renders may run,
    /// and the memory needs be computed — they read the settings (`amx` adds the AMX's operands).
    private(set) var settingsRead = false

    /// Keeps `report` as the map of the next launches — and of this one, when nothing read the settings yet.
    func keep(_ report: Diagnostic.Report) {
        guard let data = try? report.json(pretty: false) else { return }
        UserDefaults.standard.set(data, forKey: Self.capacityMapKey)
        storedMap = report
        if !settingsRead { readSettings(map: report) }
    }

    /// **Reads the engine's settings**, once: the profile of THIS library and `map`.
    private func readSettings(map: Diagnostic.Report) {
        do {
            try EngineSettings.load(from: library, map: map)
            profileWarnings = EngineSettings.effective.warnings.map(Library.withoutHome)
            mapInUse = map
        } catch {
            profileWarnings = [Library.withoutHome(error.localizedDescription)]
        }
        settingsRead = true
    }

    // MARK: Catalog

    var card: ModelCard? { cards.first { $0.id == identifier } }
    var formats: [RecommendedFormat] { card?.formats ?? ModelCard.recommendedFormats }
    var orientationFormats: [RecommendedFormat] { formats.filter { $0.orientation == orientation } }
    func loraCard(_ path: String) -> LoRACard? { compatibleLoras.first { $0.path == path } }
    var canAddLoRA: Bool { loras.count < Self.maxLoRA && !compatibleLoras.isEmpty }
    /// How many references the chosen model reads (0: it does no editing).
    var maxReferences: Int { chain?.denoising.maxReferences ?? 0 }
    var editingPossible: Bool { maxReferences > 0 }
    /// The installed models that edit, read once per catalog: where « Edit with… » sends an image.
    private(set) var editingModels: Set<String> = []
    var editors: [ModelCard] { visibleReady.filter { editingModels.contains($0.id) } }
    func readEditors() {
        editingModels = readySet.filter { (model($0)?.chain.denoising.maxReferences ?? 0) > 0 }
    }

    /// **The models, read once per catalog**: `Model.named` reads the map headers on disk, and the
    /// exploration's footer asks for each cell's model at every drawing. Emptied by `refreshCatalog`.
    @ObservationIgnored var modelsRead: [String: Model?] = [:]
    func model(_ id: String) -> Model? {
        if let known = modelsRead[id] { return known }
        let m = try? Model.named(id, in: library)
        modelsRead[id] = .some(m)
        return m
    }
    var canAddReference: Bool { references.count < maxReferences }
    /// The model's encoder sees the images (`TextFormat.readsImages`), so the prompt can name them
    /// (`<image1>`, `<image2>`… in Qwen-Image-2.1's template).
    var promptNamesImages: Bool { chain?.text.output.readsImages ?? false }

    /// **The format the render will have**: when editing, the one the denoiser derives from image 1
    /// (`DenoisingModule.editFormat`, the rule the developer's command line applies too), the chosen one
    /// otherwise.
    var outputFormat: RecommendedFormat {
        guard let first = references.first, let denoising = chain?.denoising else { return format }
        let f = denoising.editFormat(referenceWidth: first.width, referenceHeight: first.height)
        return Self.cappedForEditing(RecommendedFormat(f.width, f.height))
    }

    /// **An edit's area, in the app: 1024² at most**. The engine renders up
    /// to 1024×1536 with three references, but that is 523 s: it stays with the developer's command line.
    /// The denoiser's own rule already lands near 1024² (about 1 megapixel at image 1's proportions);
    /// only an image elongated enough that its short side is raised to 512 goes beyond — its long side
    /// then gives way, to a multiple of 32, the short side kept.
    static let editingSurface = 1024 * 1024

    static func cappedForEditing(_ f: RecommendedFormat) -> RecommendedFormat {
        guard f.width * f.height > editingSurface else { return f }
        if f.width >= f.height { return RecommendedFormat(editingSurface / f.height / 32 * 32, f.height) }
        return RecommendedFormat(f.width, editingSurface / f.width / 32 * 32)
    }

    /// **The memory this render needs**: its floor computed from the request's shapes, with what the
    /// app already holds (`ModelCard.memoryNeed`) — what the engine's preflight compares to B. `nil` when it cannot be computed.
    var memoryNeed: Int? {
        guard settingsRead else { return nil }
        let f = outputFormat
        return card.flatMap { $0.memoryNeed(width: f.width, height: f.height, references: references.count + subjectImageCount) }
    }

    /// Why "Generate" is grayed out, or `nil`: the app does not try what the request would refuse.
    /// The footer reads its two halves once per drawing (`formBlocker`, then `memoryBlocker`).
    var blocker: String? { formBlocker ?? memoryBlocker }

    /// What the form itself refuses: everything but the memory.
    var formBlocker: String? {
        if let reason = libraryBlocker { return reason }
        if let reason = installBlocker { return reason }
        if missing != nil { return String(localized: "This model isn't installed.") }
        if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return String(localized: "Write a prompt.") }
        let n: Int
        switch alternatives {
        case .failure(let e): return Problem(e).text
        case .success(let a): n = a.count
        }
        if let refusal = subjectsBlocker { return refusal }
        if file.count >= Self.maxQueueLength { return String(localized: "The queue is full.") }
        if file.count + n > Self.maxQueueLength {
            let room = Self.maxQueueLength - file.count
            return String(localized: "These alternatives make \(String(n)) renders: the queue has room for \(String(room)).")
        }
        if let denoising = chain?.denoising, (try? denoising.check(steps: steps, startImage: false)) == nil {
            let accepted = acceptedSteps.map(String.init).formatted(.list(type: .or))
            return String(localized: "This model has no schedule for \(String(steps)) steps: \(accepted).")
        }
        return nil
    }

    /// Rule 7: a render that cannot fit is refused before it starts — by the engine's own check
    /// (`ModelCard.checkMemory`, against B), not after the click.
    var memoryBlocker: String? { memoryRefusal?.text }

    /// **B, this Mac's memory budget** (`MemoryBudget.machineCeiling()`: ¾ · RAM − 2 GiB, or
    /// `SILICONED_MEMORY_BUDGET_GB`), read once: it depends on the RAM and nothing else, so what this
    /// Mac cannot do is known ahead and greyed out where it is chosen — the size, the images — never
    /// discovered at the click, and never different from one minute to the next.
    let memoryCeiling = MemoryBudget.machineCeiling()

    /// **The smallest Mac that renders `format` with `images` references**, in GB — `nil` when this
    /// one does. The engine's own check (`ModelCard.checkMemory`), against B.
    func macNeeded(_ format: RecommendedFormat, images: Int) -> Int? {
        guard settingsRead, let card else { return nil }
        do { try card.checkMemory(width: format.width, height: format.height, references: images, ceiling: memoryCeiling) }
        catch {
            guard case let .insufficientMemory(needed, _) = error else { return nil }
            return MemoryBudget.macNeeded(for: needed)
        }
        return nil
    }

    /// « Needs a Mac with 24 GB of memory. » — the one sentence of a memory refusal, wherever it shows.
    nonisolated static func needsMac(_ needed: Int) -> String {
        String(localized: "Needs a Mac with \(String(MemoryBudget.macNeeded(for: needed))) GB of memory.")
    }

    /// **A memory refusal, never a dead end**: what greys « Generate » out, and the way forward — the
    /// largest of the model's formats this Mac renders. Reached only by what the pickers cannot grey
    /// out ahead: a size typed by hand before a model change, subjects cited by the prompt.
    struct MemoryRefusal {
        let text: String
        /// The largest format that fits in B (`RecommendedFormat.largestFitting`), `nil` when none
        /// does — or when editing, where image 1 sets the size and the picker does not.
        let smaller: RecommendedFormat?
    }

    var memoryRefusal: MemoryRefusal? {
        guard settingsRead, let card else { return nil }
        let f = outputFormat
        let images = references.count + subjectImageCount
        do { try card.checkMemory(width: f.width, height: f.height, references: images, ceiling: memoryCeiling) } catch {
            // A need that cannot be computed refuses too, with its own sentence (`memoryNeedUnknown`).
            guard case let .insufficientMemory(needed, budget) = error else { return MemoryRefusal(text: Problem(error).text, smaller: nil) }
            let smaller = references.isEmpty
                ? RecommendedFormat.largestFitting(formats, budget: budget, current: f) {
                    card.memoryNeed(width: $0.width, height: $0.height, references: images) ?? .max
                  }
                : nil
            return MemoryRefusal(text: Self.needsMac(needed), smaller: smaller)
        }
        return nil
    }

    /// **What one more image would need** (after `adding` others): `nil` when this Mac renders it,
    /// else the sentence. Image 1 sets an edit's size (about a megapixel, 1024² at most): before it,
    /// judged at 1024².
    func nextReferenceRefusal(adding: Int = 0) -> String? {
        let f = references.isEmpty ? RecommendedFormat(1024, 1024) : outputFormat
        let images = references.count + adding + 1
        guard let card, macNeeded(f, images: images) != nil else { return nil }
        return card.memoryNeed(width: f.width, height: f.height, references: images).map(Self.needsMac)
    }

    /// The way forward the footer offers: the size that fits, chosen as the picker would.
    func useFormat(_ f: RecommendedFormat) {
        orientation = f.orientation
        format = f
    }

    /// **The step counts the chosen model has a schedule for**, asked of its denoiser
    /// (`DenoisingModule.check`) rather than known here: all of them for most, 8 alone for
    /// Qwen-Image-2.1-Turbo. Read when "Generate" is greyed out for that reason, never at each drawing.
    var acceptedSteps: [Int] {
        guard let denoising = chain?.denoising else { return [] }
        return (1...50).filter { (try? denoising.check(steps: $0, startImage: false)) != nil }
    }
    var canRender: Bool { blocker == nil }

    /// **The prompt's alternatives** (`{library|greenhouse}`), parsed at each reading: a few
    /// microseconds, and the form never holds a stale parse. A prompt without a group is one prompt.
    var alternatives: Result<PromptAlternatives, EngineError> {
        Result { () throws(EngineError) in try PromptAlternatives(prompt) }
    }

    /// How many renders "Generate" queues: one per expanded prompt (1 when the prompt does not parse —
    /// the button is greyed out then, and `blocker` says why).
    var variantCount: Int { (try? alternatives.get())?.count ?? 1 }

    // ── The XY grid ──

    /// A model as the user knows it: its card's name, or its identifier.
    func modelName(_ identifier: String) -> String { cards.first { $0.id == identifier }?.name ?? identifier }

    /// The LoRAs the render will take, in the stack's order — what a LoRA axis indexes.
    var activeLoRAs: [LoRASlot] { loras.filter { !$0.path.isEmpty } }

    /// The exploration prompt's `{…}` groups — what the grid's axes offer (none when it does not parse).
    var promptGroups: [[String]] { (try? PromptAlternatives(explorePrompt, limit: .max).groups) ?? [] }

    /// A LoRA as the user knows it: its card's name, or its short name.
    func loraName(_ path: String) -> String { loraCard(path)?.name ?? RemoteControl.shortName(path) }

    /// **The grid the exploration's form describes**, checked by the library. `seed`: the base seed
    /// (the rack's; the one drawn at the click for a random seed). A model axis drops the stack: a
    /// LoRA is forged for one model (`loraAcrossModels` otherwise).
    func grid(seed: UInt64) -> Result<RenderGrid, EngineError> {
        let models = [gridX, gridY].contains { $0.kind == .model }
        let stack = models ? [] : activeLoRAs
        return Result { () throws(EngineError) in
            guard let x = try gridX.axis(stack: stack) else { throw .gridRefused(reason: .emptyAxis) }
            let g = try RenderGrid(.init(prompt: explorePrompt, seed: seed, steps: steps,
                                         loraStrengths: stack.map(\.strength), images: references.count),
                                   x: x, y: try gridY.axis(stack: stack))
            guard !refusesNegativeStrength(g) else { throw .gridRefused(reason: .invalidValue) }
            // A solver the form's model does not offer (Developer Mode off) renders nothing new.
            guard g.cells.allSatisfy({ $0.solver.map(offeredSolvers.contains) ?? true }) else { throw .gridRefused(reason: .invalidValue) }
            return g
        }
    }
    var grid: Result<RenderGrid, EngineError> { grid(seed: seed) }

    /// Axes that make sense for this prompt — its groups first, the seed otherwise.
    func gridDefaults() {
        guard (try? grid.get()) == nil else { return }
        let groups = promptGroups.count
        gridX = GridAxisForm(kind: groups > 0 ? .prompt(0) : .seed)
        gridY = GridAxisForm(kind: groups > 1 ? .prompt(1) : groups == 1 ? .seed : .none)
        if groups == 1 { gridY.seedCount = 2 }
    }

    /// **The exploration's form takes a grid** — a sheet's settings reused, or a grid the remote
    /// control queued: its prompt as typed, its axes.
    func setGridForm(_ g: RenderGrid, prompt typed: String) {
        let stack = activeLoRAs
        explorePrompt = typed
        gridX = GridAxisForm(g.x, stack: stack)
        gridY = GridAxisForm(g.y, stack: stack)
    }

    /// The series, one expanded prompt per line — the tooltip of the render count.
    var alternativesHelp: String {
        guard let a = try? alternatives.get(), a.hasAlternatives else { return "" }
        let list = a.prompts.prefix(12).joined(separator: "\n")
        return a.count > 12 ? list + "\n…" : list
    }

    /// **What the render will cost as it is set**, if it has already been measured in the session at
    /// this format: never a constant, never another machine.
    var plannedCost: Double? {
        let withLoRA = loras.contains { !$0.path.isEmpty }
        let format = outputFormat
        let images = references.count + subjectImageCount
        guard let v = speeds[Speed.key(identifier, format, references: images, lora: withLoRA)] else { return nil }
        guard let plan = expectedPlan(steps: steps, format: format, withLoRA: withLoRA || !plainSampling,
                                      references: images) else { return nil }
        // One render per alternative, each the cost of one: same model, format, plan and batch.
        return Double(variantCount) * v.duration(plan, images: max(1, batch))
    }

    /// **One stage's share of that cost**, per image — what the rack writes under each box. Learned
    /// like the whole (`speeds`): `nil` until this model has rendered at this size in the session.
    func stageEstimate(_ stage: Engine.Stage) -> Double? {
        let withLoRA = loras.contains { !$0.path.isEmpty }
        let format = outputFormat
        let images = references.count + subjectImageCount
        guard let v = speeds[Speed.key(identifier, format, references: images, lora: withLoRA)] else { return nil }
        switch stage {
        case .text: return v.text
        case .image: return v.image
        case .decoding: return v.decoding
        case .denoising:
            guard let plan = expectedPlan(steps: steps, format: format, withLoRA: withLoRA || !plainSampling,
                                      references: images) else { return nil }
            let reduced = min(plan.reduced, plan.evaluations)
            return Double(plan.evaluations - reduced) * v.full + Double(reduced) * (v.reducedAverage ?? v.full)
        }
    }

    /// **Under a box, its time**: what the stage just took when it is done in the render on screen;
    /// otherwise the estimate learned on this Mac. `measured` says which.
    func stageCost(_ stage: Engine.Stage) -> (seconds: Double?, measured: Bool) {
        if state(of: stage) == .done, let s = stageSeconds[stage] { return (s, true) }
        return (stageEstimate(stage), false)
    }

    /// The rack's light for a stage of the chosen model.
    func state(of stage: Engine.Stage) -> StageState {
        stageModel == identifier ? stageStates[stage] ?? .idle : .idle
    }

    private func modelChanged() {
        readModelCatalog()
        warm()
        // A LoRA from another model would be refused at render (`LoRA.Failure.wrongTarget`).
        let valid = Set(compatibleLoras.map(\.path))
        loras.removeAll { !$0.path.isEmpty && !valid.contains($0.path) }
        steps = card?.defaultSteps ?? 8
        if references.count > maxReferences { references = Array(references.prefix(maxReferences)) }
        // A size typed by hand stays while the new model takes it (`Format.check(…, family:)`: Qwen-Image-2.1
        // wants multiples of 32, Anima 2048 px a side at most) — and this Mac's memory, or the footer
        // says so and offers the largest that fits.
        if (try? checkFormat(width: format.width, height: format.height)) == nil {
            format = formats[0]; orientation = format.orientation
        }
    }

    /// **The chosen model's format rules** (`Format.check(…, family:)`, the engine's preflight); the
    /// general ones when no model is chosen.
    func checkFormat(width: Int, height: Int) throws(EngineError) {
        if let family = card?.family { try Format.check(width: width, height: height, family: family) }
        else { try Format.check(width: width, height: height) }
    }

    /// The multiple a typed side must be, and the longest side, for the chosen model — what the size
    /// field announces, so that it never suggests a size the model refuses.
    var formatMultiple: Int { card.map { Format.multiple(for: $0.family) } ?? Format.multiple }
    var formatMaximumSide: Int { card.map { Format.maximumSide(for: $0.family) } ?? Format.maximumSide }

    /// **A size typed by hand** (`832x1216`, `1024`): read by `Format.parse`, judged by `Format.check` —
    /// the engine's own rules (sides ≥ 512, multiples of 16, at most 8192 px a side; no area ceiling, B decides), and the
    /// model's (Qwen-Image-2.1 multiples of 32, Anima 2048 px a side at most). The refusal, translated.
    func setCustomFormat(_ text: String) -> Problem? {
        guard let (w, h) = Format.parse(text.trimmingCharacters(in: .whitespaces)) else {
            return Problem(EngineError.formatUnreadable(text: text))
        }
        do { try checkFormat(width: w, height: h) } catch { return Problem(error) }
        // Beyond B, refused here as the picker greys it out: the same check as the engine's.
        if settingsRead {
            do { try card?.checkMemory(width: w, height: h, references: subjectImageCount, ceiling: memoryCeiling) }
            catch { return Problem(error) }
        }
        let f = RecommendedFormat(w, h)
        orientation = f.orientation
        format = f
        return nil
    }

    /// The sizes the picker offers: the model's for this orientation, and the one typed by hand.
    var pickerFormats: [RecommendedFormat] {
        orientationFormats.contains(format) ? orientationFormats : orientationFormats + [format]
    }

    /// Re-reads from disk what depends on the chosen model: its chain, its LoRAs, what it is missing.
    func readModelCatalog() {
        chain = model(identifier)?.chain
        compatibleLoras = library.loras(for: identifier)
        missing = readySet.contains(identifier) ? nil : card?.missing(in: library)
    }

    /// The orientation's first size this Mac renders (`macNeeded`), its first at all when none does —
    /// the footer then says why.
    private func orientationChanged() {
        guard format.orientation != orientation, let first = orientationFormats.first else { return }
        format = orientationFormats.first { macNeeded($0, images: subjectImageCount) == nil } ?? first
    }
}
