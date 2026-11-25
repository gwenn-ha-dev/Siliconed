// **The app's state**: what the user sets, what the render shows, the history.
//
// **Incognito by default**: the history lives in memory and dies with the
// process. Nothing is written to disk without an explicit gesture — "Save", "Copy" —,
// no temporary file, no preference, no log: nothing that Spotlight or recent items
// could find. **The exceptions**, each a gesture of the user: the preferences (UserDefaults) — the « More
// Solvers » switch (`developerModeKey`), the comparison's curtain, switch or side by side
// (`Curtain.mode`), the beta « Negative LoRA Strengths » switch
// (`negativeLoRAKey`), the file format of the last export (`exportFormatKey`), the folder of « Save Every Image To… », off
// unless the user turns it on — each new image is then written there (`EveryImage`) —, the frames of the main window and of the Prompt Lab's (a place on the screen, nothing of what was made), the Macs paired and the lending switch (`OtherMacs`) — like the system language the app follows —, the licenses accepted, which the library keeps itself (`<local>/accepted-licenses.json`) because its
// engine refuses a render without them, and where the library is when the user moved it (`<local>/library-location`,
// `AppState+Place.swift`). The named subjects (`Subjects.swift`) live in memory, like
// the history.
//
// **The models in front** are the library's families, in the order it declares them (`frontFamilies`): nothing is
// filtered. « More Solvers » (`developerMode`, a key older than its name) offers the
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
// `package` symbols, never its internals. The view is in `MainView.swift`, the window in `MainWindow.swift`, the app in `App.swift`.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

@MainActor @Observable
final class AppState {
    /// The library: this Mac's, or the one the user moved or opened elsewhere (`AppState+Place.swift`).
    var library: Library
    /// The library's disk is connected (re-read when a volume comes or goes: `watchVolumes`).
    var libraryReachable = true
    @ObservationIgnored var volumeObservers: [NSObjectProtocol] = []
    /// All the known models — the publisher's, then the imported ones (`Library.cards()`).
    var cards: [ModelCard] = []
    /// The ready models (all their files present).
    var readySet: Set<String> = []
    /// Counts the readings of the library (`refreshCatalog`): what follows the library without being one of the
    /// values above — a lender's catalog of LoRAs — observes it, since a value set equal to itself tells no one.
    var libraryRead = 0

    // ── What the app shows ──
    /// **The families in front**, **in the order every list shows them** (the rack's picker, the Models sheet, the
    /// welcome, `silicontrol models`): the library's, as it declares them.
    static let frontFamilies = Family.allCases
    private static let developerModeKey = "developerMode"
    /// **More Solvers** (Settings ▸ Advanced ▸ Beta): the solvers (`offeredSolvers`). The property and its
    /// preference key keep the name they had when this was « Developer Mode ».
    /// One of the app's preferences (with `Curtain.mode`) — written without a render gesture, because the user switched it.
    var developerMode = UserDefaults.standard.bool(forKey: AppState.developerModeKey) {
        didSet {
            guard developerMode != oldValue else { return }
            UserDefaults.standard.set(developerMode, forKey: Self.developerModeKey)
            // A solver the models no longer offer leaves the form.
            if !developerMode {
                solver = .euler
            }
        }
    }
    private static let stepsBetaKey = "stepsBeta"
    /// **Choose Steps** (Settings ▸ Advanced ▸ Beta, issue 16): off by default, the steps are the model's recipe — the five are
    /// distilled for 8, and another count only costs. On, the DiT box offers them, for a step-distilled LoRA (Krea 2 at 4)
    /// that brings a recipe of its own. Off again, the form goes back to the recipe, as More Solvers off goes back to Euler.
    /// `silicontrol add --steps` is not gated by it: the command line always could, and a PNG's settings reused still bring theirs.
    var stepsBeta = UserDefaults.standard.bool(forKey: AppState.stepsBetaKey) {
        didSet {
            guard stepsBeta != oldValue else { return }
            UserDefaults.standard.set(stepsBeta, forKey: Self.stepsBetaKey)
            if !stepsBeta { steps = card?.defaultSteps ?? 8 }
        }
    }
    private static let negativeLoRAKey = "negativeLoRAStrengths"
    /// **Negative LoRA strengths** (Experimental): W + s·ΔW with s < 0 pushes the image away from what
    /// the LoRA learned — meant for « slider » LoRAs, trained as a difference to be used from −x to +x;
    /// on an ordinary LoRA it mostly damages the image, hence the switch. The engine takes any finite
    /// strength (`LoRA.materialize` folds it into `down`, sign included): the gate is the app's, for
    /// the rack, a series and `silicontrol` alike. Off, the strengths already set come back to 0.
    var negativeLoRAStrengths = UserDefaults.standard.bool(forKey: AppState.negativeLoRAKey) {
        didSet {
            guard negativeLoRAStrengths != oldValue else { return }
            UserDefaults.standard.set(negativeLoRAStrengths, forKey: Self.negativeLoRAKey)
            if !negativeLoRAStrengths {
                loras = loras.map(allowedStrength)
                brackets = brackets.map { b in
                    var b = b
                    b.numbers = b.key == .startStrength ? b.numbers : b.numbers.map { max(0, $0) }
                    b.strength = max(0, b.strength)
                    return b
                }
            }
        }
    }
    private static let sizeScalesBetaKey = "sizeScalesBeta"
    /// **Size in two scales, the Beta option** (Settings ▸ Advanced ▸ Beta): off by default. On, the rack's Size menu becomes a
    /// proportion and a resolution (`SizeScales`). Only the rack changes: the sizes, the pencil, `silicontrol` are the same.
    var sizeScalesBeta = UserDefaults.standard.bool(forKey: AppState.sizeScalesBetaKey) {
        didSet { UserDefaults.standard.set(sizeScalesBeta, forKey: Self.sizeScalesBetaKey) }
    }
    private static let exportFormatKey = "exportFormat"
    /// **The file format the last export used** (PNG, JPEG or HEIC): the next Save and Export open on it.
    var exportFormat = ExportFormat(rawValue: UserDefaults.standard.string(forKey: AppState.exportFormatKey) ?? "") ?? .png {
        didSet { UserDefaults.standard.set(exportFormat.rawValue, forKey: Self.exportFormatKey) }
    }
    static let neuralEngineKey = "neuralEngine"
    private static let glassKey = "glass"
    /// **Whether the engine may give work to the Neural Engine** (Settings ▸ General ▸ Neural Engine), on by default: the
    /// engine sizes its share and takes it back where it does not help. Pushed to the engine (`CoreAI.neuralEngine`) at
    /// creation and at each change, before any plan reads it.
    var neuralEngine = UserDefaults.standard.object(forKey: AppState.neuralEngineKey) as? Bool ?? true {
        didSet {
            UserDefaults.standard.set(neuralEngine, forKey: Self.neuralEngineKey)
            CoreAI.neuralEngine = neuralEngine
        }
    }
    /// **The columns' glass** (Settings ▸ General): on, a zoomed image shows through the chain and the history; off, they
    /// are opaque. The glass composes again whenever the image moves under it: measured on an M5 Pro at +0.4 % of a 1024²
    /// render (median of three) with a zoomed image shown, more on a small GPU.
    var glass = UserDefaults.standard.object(forKey: AppState.glassKey) as? Bool ?? true {
        didSet { UserDefaults.standard.set(glass, forKey: Self.glassKey) }
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
    /// Where a family comes in a list.
    static func displayRank(_ family: Family) -> Int { frontFamilies.firstIndex(of: family) ?? frontFamilies.count }
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
    /// The first ready model in the lists' order; none ready, the product's default (`Family.default`).
    private var defaultIdentifier: String { visibleReady.first?.id ?? ModelCard.of(.default).id }

    // ── What the user sets ──
    var identifier = "" {
        didSet {
            guard identifier != oldValue else { return }
            // Each family has its own default strength: a strength chosen for one does not follow to another.
            if cards.first(where: { $0.id == oldValue })?.family != card?.family { chosenStartStrength = nil }
            modelChanged()
        }
    }
    var prompt = ""
    var orientation = RecommendedFormat.Orientation.square { didSet { orientationChanged() } }
    var format = ModelCard.recommendedFormats[0]
    var loras: [LoRASlot] = []
    /// **The render's images, each with its role** (`AppState+Inputs.swift`): the image edited and the references
    /// (a model that reads images), the image an img2img starts from (a model that starts from one), the neutral ones waiting
    /// for a role. Brought by the Prompt Lab, a drop, « Edit This Image », ⌘V or the remote control; `references` and
    /// `startImage` read them.
    var inputs: [InputImage] = []
    /// **The start's strength** (diffusers' `strength`, `StartImage`): how far the render moves from the image — retouched in the rack
    /// between two renders. In hundredths. The family's default (`StartImage.defaultStrength`) until one is chosen; forgotten
    /// when the rack changes family.
    var startStrength: Double {
        get { chosenStartStrength ?? defaultStartStrength }
        set { chosenStartStrength = newValue }
    }
    var chosenStartStrength: Double?
    var steps = 8
    /// **More detail** (`Detail`): Normal is the render as the model makes it. What a render takes is
    /// `effectiveDetail`: on a model that offers none (`DenoisingModule.acceptsDetail`), Normal.
    var detail = Detail.normal
    /// Whether the form's model offers Detail — the rack hides the choice otherwise.
    var offersDetail: Bool { mirror != nil ? traits?.acceptsDetail ?? true : model(identifier)?.chain.denoising.acceptsDetail ?? true }
    var effectiveDetail: Detail { offersDetail ? detail : .normal }
    /// **The solver** (`Solver`), offered with More Solvers on the models in front
    /// (`offeredSolvers`). What a render takes is `effectiveSolver`: a choice the form's model does
    /// not offer is Euler.
    var solver = Solver.euler
    /// The families whose samplers walk every solver (`FamilyTraits.everySolver`).
    static let solverFamilies = Set(Family.allCases.filter(\.traits.everySolver))
    /// The solvers the rack offers: all four with More Solvers on those families, Euler alone otherwise.
    var offeredSolvers: [Solver] {
        developerMode && card.map({ Self.solverFamilies.contains($0.family) }) == true ? Solver.allCases : [.euler]
    }
    var effectiveSolver: Solver { offeredSolvers.contains(solver) ? solver : .euler }
    /// Fixed seed (the field) or drawn at random at each render, like Draw Things. A variation
    /// belongs to its seed: changing the seed, or letting it be drawn, drops it.
    var fixedSeed = false { didSet { if !fixedSeed { variations = [] } } }
    var seed: UInt64 = 42 { didSet { seedShown = true; if seed != oldValue { variations = [] } } }
    /// Whether the rack's field has a seed to show: none before the first render draws one (or a seed is set: the dice,
    /// a reused image, `silicontrol`) — the greyed field says « — », not the 42 nothing drew. Set by `seed`, and carried
    /// across a switch of Mac (`copyForm`).
    var seedShown = false
    /// **The variations the next render takes** (`Request.variations`) — only ever set by reusing a
    /// variation's settings (its menu, a dropped PNG), shown in the rack with a button that removes them.
    var variations: [Variation] = []
    var batch = 1
    var previews = true
    /// **The named subjects** (`Subjects.swift`): the Prompt Lab's library, in memory. The rack does not read them.
    let subjects: SubjectLibrary
    /// **The knobs bracketed** (`Series.swift`), in the order bracketed — the words are the text's groups, the seeds the lot.
    var brackets: [Bracket] = []
    /// Where the hand put the series' axes, `[columns, rows]` (`SeriesLayout`).
    var seriesPlaced: [SeriesKey?] = [nil, nil]
    /// The knob whose « Bracket… » bubble is open.
    var bracketing: SeriesKey?
    /// **Where `silicontrol rack` asks the rack's column to scroll**, and a count so that the same place asked twice scrolls again;
    /// `nil` once the column has scrolled — a folded column, unfolded for it, scrolls as soon as it is drawn (`Sidebar`).
    var rackScroll: (place: GestureOrder.RackPlace, count: Int)?
    /// **The series whose grid the canvas shows** rather than an image (`SeriesCanvas`): its tile's click, « Generate »
    /// with something bracketed, `silicontrol grid`. Any image chosen takes the canvas back. Leaving the grid loads the
    /// image the strip marks: a series renders its cells without choosing one, and the canvas would come back empty.
    var shownSeries: Int? {
        didSet {
            // A grid opened fills the canvas, whatever size the hand gave its cells the last time it was shown.
            if shownSeries != oldValue { seriesFit = nil }
            if shownSeries == nil, oldValue != nil { loadSelectedImage() }
        }
    }
    /// The series whose cells the history folds under their tile — the arrows skip them.
    var foldedSeries: Set<Int> = []
    /// A grid cell's long side on the canvas, in points: the slider, the pinch, ⌘+ ⌘−.
    var exploreCellSide: CGFloat = 220
    /// **How the shown series was last fitted to the canvas** (`SeriesFit`): whether its cells follow the window, and the
    /// side ⌘0 gives back.
    var seriesFit: SeriesFit?
    static let cellSides = SeriesFit.sides
    static func clampedCellSide(_ s: CGFloat) -> CGFloat { SeriesFit.clamped(s) }

    // ── The chosen model's catalog: read from disk when it changes, not at each drawing ──
    /// The chosen model's chain — what the boxes draw. `nil` if it is not installed.
    private(set) var chain: Chain?
    /// The library's LoRAs for the chosen model.
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
    /// The running render; those waiting are `file`, in the order they were requested.
    var currentJob: Job?
    /// The figures of the running render and of the last finished one — the Chain inspector.
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
    /// **A preparation compiles now** (`AppState+Prepare.swift`): outside any render, or before the render that waits for it.
    var preparingCoreAI = false
    /// The form's key last prepared, and its pause under way (`AppState+Prepare.swift`).
    @ObservationIgnored var preparedKey: PreparationKey?
    @ObservationIgnored var preparation: Task<Void, Never>?
    /// The end of a preparation's burst, said once it lasted (`preparationTold`).
    @ObservationIgnored var preparationEnd: Task<Void, Never>?
    /// A render is running or waiting: installations (several GB) and removing a model wait.
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
    /// **The last pre-render of each job under way**, by job: in a train of seeds (a series' cells rendered in one call) the
    /// engine passes from one cell to the next before the first's image is decoded — each cell keeps its own pre-render
    /// in its place until its image comes, never the next cell's.
    var prerenders: [Int: CGImage] = [:]
    /// The running job's pre-render.
    var preview: CGImage? { currentJob.flatMap { prerenders[$0.ordinal] } }
    var history: [Entry] = []
    var selection: Entry.ID? {
        didSet {
            guard selection != oldValue else { return }
            loadSelectedImage()
            // An image chosen (or a finished render followed): the canvas shows it, not the grid.
            if selection != nil { shownSeries = nil; cropping = nil }
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
        (e.references.contains { if case .remote = $0.source { false } else { true } } || e.enlargedFrom != nil)
    }
    /// The thumbnails' side in the history column (the slider at its foot); in memory, like the rest.
    var thumbnailSide: Double = 96
    /// **The history widened into a grid over the canvas** (`HistoryGrid`): the same list, the same selection, the same keys;
    /// it narrows back to the column to show the image.
    var historyGrid = false
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
    /// **How far the zoom goes, in the image's pixels**: up to `Zoom.pointsPerPixel` points of the screen for one of
    /// its pixels, whatever the window — set by the view that shows it (`Zoom.limit`).
    var zoomLimit: CGFloat = Zoom.pointsPerPixel
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
        /// The version to install, chosen in the panel: the one rendered (else the preselected one) unless the
        /// user picks another.
        var variant: Variant = .standard
    }
    /// **The « Use this model » link being answered** (`HubSheet`): the repository, its files judged.
    var hubOffer: HubOffer?
    /// **Space moves the components aside** — the two columns, the bar under the image —: the image alone
    /// in the window, never a second viewer. Space or Esc brings them back.
    var componentsAway = false
    /// **The image's details**, the popover of the ⓘ under it (⌘I).
    var imageDetails = false
    var toast: String?
    /// An installation or a move of the library in progress (several minutes, several GB): its title, and what it says.
    var libraryJob: String?
    var libraryJobJournal: [String] = []
    /// Where an installation in progress is (`Installer.Progress`); `nil` before its first file, and for a move.
    var libraryJobProgress: Installer.Progress?
    /// What the job in progress is, by identifiers: what a borrowing Mac is told (`RemoteLibraryJob`).
    var libraryJobTask: RemoteLibraryJob.Task = .other
    /// In a mirror: the other Mac's library and the installation under way there (`MirrorLibrary.swift`).
    var remoteLibrary: RemoteLibrary?
    var remoteLibraryJob: RemoteLibraryJob?
    /// In a mirror: a removal asked waits for a click on the lender (`RemoteLibraryNews.confirming`).
    var remoteConfirming = false
    /// The token of the job in progress: "Stop" cuts a download in the middle.
    var libraryJobCancellation: Cancellation?
    /// **What this Mac's compilations weigh** (`CoreAI.compiled`), Settings ▸ Macs ▸ This Mac's; `nil` before the first
    /// read (`weighCompiled`).
    var compiledBytes: Int?
    /// An erasure of the compilations sent to the engine and not yet over (`eraseCompiled`).
    var erasingCompiled = false
    /// What the library occupies — the "Models" sheet; re-read with the catalog, off the main thread (`readOccupancy`). Shown,
    /// and published to a borrowing Mac: never what a decision reads, which looks at the disk itself (`Library.part`).
    var occupancy: Library.Occupancy?
    /// The latest read of `occupancy` asked: an earlier one that returns after it is dropped.
    @ObservationIgnored var occupancyReads = 0
    /// The LoRAs the library holds, each with its file's bytes — the "Models" sheet's; re-read with the catalog. Empty in a
    /// mirror: the other Mac's are its own to remove.
    var installedLoRAs: [(card: LoRACard, bytes: Int)] = []
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
    /// **The jobs the running engine call renders, one image each, in its images' order** — the running job alone, or
    /// with the cells of its series that follow it in the queue and are the same render under another seed
    /// (`riders`): the text once, one DiT session, then one decoder session, as a lot. Measured on Z-Image at 512²,
    /// four seeds, text already kept: 6.13 s a cell alone, 5.0–5.1 s an image of a lot (M5 Pro).
    /// **Each stays its own job**: the one whose image the engine works on is `currentJob`, the others wait in `file`
    /// (`turn`) — a job is never out of both before it ends, which is what a mirror reads.
    var train: [Job] = []
    /// Of `train`, the jobs whose image came (they have ended) and those told as started.
    var delivered: Set<Job.ID> = []
    var started: Set<Job.ID> = []
    /// As many images as a lot (the rack's stepper): no image comes before the last denoising of a train.
    static let maxTrain = 8
    /// **The session's series**, ended or not, and which cell each of their jobs is — an image of the history is a
    /// cell by its job (`cell(of:)`).
    var series: [Int: GridRun] = [:]
    var gridCells: [Int: (grid: Int, cell: Int)] = [:]
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
        if let at = gridCells[t.ordinal], let run = series[at.grid], run.issues[at.cell] == nil {
            series[at.grid]!.issues[at.cell] = issue
            if series[at.grid]!.isComplete { signal(.gridEnded(series[at.grid]!)) }
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
    /// The other Mac's chip, memory and ready models (`MirrorMachine.swift`).
    var mirrorMachine: RemoteMachine?
    /// What the other Mac renders for others (itself, other Macs): how many, and for how long (`nil`: unknown).
    var othersQueued = 0
    var othersSeconds: Double?
    /// **Why nothing can be sent now**, in a mirror: the session with the other Mac is not joined (connecting,
    /// lost and tried again, refused). A request made then would be dropped without a word (`RemoteLink.send`),
    /// so « Generate » waits for the link (`blocker`). `nil`: joined.
    var linkBlocker: String?
    /// What the other Mac answered to the form's questions (`quoted`).
    var mirrorQuotes: [RemoteQuote: RemoteQuoteAnswer] = [:]
    /// What the other Mac estimates each waiting job of this Mac's takes, by ordinal (`RemoteStatistics`):
    /// the speeds learned there are not a plan's to compute here.
    var mirrorEstimates: [Int: Double] = [:]

    /// **The state of another Mac**, shown here (`AppState+Mirror.swift`): nothing read from this Mac's
    /// library but the named subjects, which stay with the user (this Mac's own library, shared: `silicontrol
    /// subjects` and the chips act on the same ones in either view); no warm-up.
    init(mirroring link: MirrorLink, library: Library, subjects: SubjectLibrary) {
        self.library = library
        self.subjects = subjects
        mirror = link
        // Joined only when its `connected` comes (`linkChanged`).
        linkBlocker = String(localized: "Connecting to “\(link.macName)”…")
        link.down = linkBlocker
        cards = ModelCard.allCards
        identifier = defaultIdentifier
        CoreAI.neuralEngine = neuralEngine
    }

    init(library: Library) {
        self.library = library
        libraryReachable = library.isReachable
        subjects = SubjectLibrary(library: library)
        cards = library.cards()
        readySet = Set(library.models().map(\.id))
        identifier = defaultIdentifier
        CoreAI.neuralEngine = neuralEngine
        readEditors()
        watchVolumes()
        readOccupancy()
    }

    // MARK: Catalog

    /// The installed models that edit, read once per catalog: where « Edit with… » sends an image.
    private(set) var editingModels: Set<String> = []
    func readEditors() {
        editingModels = readySet.filter { (model($0)?.chain.denoising.maxReferences ?? 0) > 0 }
    }

    /// **The models, read once per catalog**: `Model.named` reads the map headers on disk, and the
    /// views ask for the chosen model at every drawing (`offersDetail`, the formats). Emptied by `refreshCatalog`.
    @ObservationIgnored var modelsRead: [String: Model?] = [:]

    /// **B, this Mac's memory budget** (`MemoryBudget.budget`: ¾ · RAM − 2 GiB, or
    /// `SILICONED_MEMORY_BUDGET_GB`), read once: it depends on the RAM and nothing else, so what this
    /// Mac cannot do is known ahead and greyed out where it is chosen — the size, the images — never
    /// discovered at the click, and never different from one minute to the next.
    /// B: this Mac's, or the lending Mac's in a mirror (`AppState+Mirror.swift`). This Mac's is the user's, set in
    /// Settings (`memoryBudget`), else the default.
    var memoryCeiling = MemoryBudget.budget

    // MARK: B (`AppState+Budget.swift`)

    /// Counts the changes of the user's B: the slider follows.
    var memoryBudgetChanges = 0
}
