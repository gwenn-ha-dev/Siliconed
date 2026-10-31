// **The Prompt Lab: a window where an installed language model writes a better prompt.**
//
//     Window ▸ Prompt Lab (⌥⌘L)
//     ┌──────────────────────────────────────────────────┐
//     │ Write for [Z-Image ▾]         Writer: Qwen3-VL-8B │
//     │            ( Rewrite | Describe )                │
//     │ ┌──────────────────────────────────────────────┐ │
//     │ │ woman posing in a library                    │ │   Rewrite: a text, typed, pasted
//     │ │                    Use the Rack's Prompt     │ │   or taken from the rack
//     │ └──────────────────────────────────────────────┘ │   Describe: an image, dropped or opened
//     │ [                  Rewrite                     ] │
//     │ ┌ Result ─────────────────────────── Edit ─────┐ │
//     │ │ A candid photograph of woman standing …      │ │   additions in the accent, removals struck
//     │ └──────────────────────────────────────────────┘ │
//     │                                  [ Send to Rack ]│
//     └──────────────────────────────────────────────────┘
//
// - **The render always has priority.** When one starts (`AppState.busy`: running or queued), a write
//   under way is cancelled — the window says so once, quietly —, the writer is told to release its memory,
//   and the two buttons grey out under one sentence until the render ends. Hand-editing stays.
// - **Incognito**: the text, the image, the results live in `LabModel`, in memory; no file, no default,
//   no history. Closing the window cancels and releases; the texts stay for the session.
// - **This Mac only**: while the window follows another Mac (`Carrier.mirror`), the lab says so and
//   does nothing — the writer is this Mac's, the rack it would fill is not.
// - **« Send to Rack »** sets the rack's prompt; the prompt field applies it through the text system, so
//   ⌘Z in that field brings the old prompt back (`PromptField.updateNSView`).
// - The engine is behind `LabWriter` (`LabWriter.swift`).

import AppKit
import Siliconed
import SwiftUI

// MARK: - The state

@MainActor @Observable
final class LabModel {
    static let shared = LabModel()

    enum Gesture: String, CaseIterable, Identifiable { case rewrite, describe; var id: String { rawValue } }
    enum Phase: Equatable { case idle, loading(Double?), writing }

    /// A finished result: `original` is the text it rewrites (the diff's other side), `nil` when there is none.
    struct Result: Equatable { var text: String; var original: String? }

    /// Swapped by the engine's wiring; the sample writer until then.
    var writer: any LabWriter = FakeLabWriter() { didSet { refresh() } }
    private(set) var availability: LabAvailability = .unavailable(.preparing)

    var gesture = Gesture.rewrite
    var text = ""
    var image: Data?
    var thumbnail: NSImage?
    /// The family chosen by hand; `nil` follows the rack's.
    var chosenTarget: Family?

    private(set) var phase = Phase.idle
    private(set) var streaming = ""
    var results: [Gesture: Result] = [:]
    /// Rewrite: whether the result is shown as edited by hand rather than as a diff.
    var editing = false
    private(set) var stopped = false
    private(set) var failure: String?
    private(set) var sent = false
    /// Whether a render is running or queued on this Mac: the lab is paused.
    private(set) var rendering = false

    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var watching = false

    var isWriting: Bool { phase != .idle }

    func refresh() { availability = writer.availability }

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

    // MARK: Gestures

    func run(target: Family) {
        guard !isWriting, !rendering else { return }
        let input: LabInput, g = gesture
        switch g {
        case .rewrite:
            let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !t.isEmpty else { return }
            input = .text(t)
        case .describe:
            guard let image else { return }
            input = .image(image)
        }
        let original: String? = { if case .text(let t) = input { t } else { nil } }()
        generation += 1
        let mine = generation
        phase = .loading(nil); streaming = ""; stopped = false; failure = nil; sent = false
        task = Task { [writer] in
            do {
                var last = ""
                for try await event in writer.write(input, target: target) {
                    guard mine == self.generation else { return }
                    switch event {
                    case .loading(let f): phase = .loading(f)
                    case .text(let t): phase = .writing; streaming = t; last = t
                    case .finished(let t): last = t; finish(t, original: original, as: g)
                    case .cancelled: self.stop(becauseOfRender: false); return
                    }
                }
                guard mine == self.generation else { return }
                // A stream that ended without saying so: what it wrote is the result.
                if phase != .idle { finish(last, original: original, as: g) }
            } catch {
                guard mine == self.generation else { return }
                phase = .idle; streaming = ""
                failure = error.localizedDescription
            }
        }
    }

    private func finish(_ text: String, original: String?, as g: Gesture) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        phase = .idle; streaming = ""
        guard !clean.isEmpty else { return }
        results[g] = Result(text: clean, original: original)
        editing = false
    }

    /// The previous result stays: half a prompt is not offered to the rack.
    func stop(becauseOfRender: Bool = false) {
        generation += 1
        task?.cancel(); task = nil
        phase = .idle; streaming = ""
        stopped = becauseOfRender
    }

    /// The window closes: nothing keeps writing, nothing keeps the memory.
    func windowClosed() {
        if isWriting { stop() }
        Task { await writer.release() }
    }

    // MARK: Inputs

    func useRackPrompt(_ app: AppState) {
        text = app.prompt
        gesture = .rewrite
    }

    func load(image data: Data) {
        guard let picture = NSImage(data: data), picture.isValid else {
            failure = String(localized: "This image could not be read.")
            return
        }
        image = data; thumbnail = picture; gesture = .describe; failure = nil
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

    func chooseImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.prompt = String(localized: "Choose")
        panel.message = String(localized: "The image to describe.")
        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(.file(url))
    }

    func send(_ result: String, to app: AppState) {
        app.prompt = result
        sent = true
        MainWindow.shared.show()
        Task {
            try? await Task.sleep(for: .seconds(2))
            sent = false
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
        w.setContentSize(NSSize(width: 560, height: 700))
        w.contentMinSize = NSSize(width: 480, height: 580)
        w.center()
        window = w
        NSApp.activate()
        w.makeKeyAndOrderFront(nil)
    }

    func windowWillClose(_ notification: Notification) { LabModel.shared.windowClosed() }
    func windowDidBecomeKey(_ notification: Notification) { LabModel.shared.refresh() }
}

// MARK: - The view

struct LabView: View {
    @Bindable var model: LabModel
    @FocusState private var typing: Bool
    @State private var targeted = false

    private var rack: AppState { MainWindow.shared.local }
    private var followingAnotherMac: Bool { MainWindow.shared.carrier.mirror != nil }

    /// The families installed here, in the catalog's order: the lab writes for what can render.
    private var families: [Family] {
        let ready = Set(rack.cards.filter { rack.readySet.contains($0.id) }.map(\.family))
        return Family.allCases.filter(ready.contains)
    }

    private var target: Family? {
        if let c = model.chosenTarget, families.contains(c) { return c }
        if let f = rack.card?.family, families.contains(f) { return f }
        return families.first
    }

    /// Why the gestures are off, in one sentence; `nil` when they are on.
    private var pause: (symbol: String, sentence: String)? {
        if followingAnotherMac { return ("macbook", String(localized: "The lab works on this Mac only.")) }
        if model.rendering { return ("pause.circle", String(localized: "Paused while an image renders.")) }
        if case .unavailable(let why) = model.availability {
            switch why {
            case .notInstalled: return ("shippingbox", String(localized: "No writer is installed yet. Install a model to have one."))
            case .needsMemory(let n): return ("memorychip", String(localized: "The writer needs a Mac with \(n) GB of memory."))
            case .preparing: return ("hourglass", String(localized: "The writer is being prepared."))
            }
        }
        if target == nil { return ("shippingbox", String(localized: "No writer is installed yet. Install a model to have one.")) }
        return nil
    }

    private var canRun: Bool {
        guard pause == nil else { return false }
        switch model.gesture {
        case .rewrite: return !model.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case .describe: return model.image != nil
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Picker("Action", selection: $model.gesture) {
                Text("Rewrite").tag(LabModel.Gesture.rewrite)
                Text("Describe").tag(LabModel.Gesture.describe)
            }
            .pickerStyle(.segmented).labelsHidden()
            .disabled(model.isWriting)
            input
            action
            result
            footer
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .dropDestination(for: DroppedImage.self) { items, _ in
            guard let first = items.first, !model.isWriting else { return false }
            model.load(first)
            return true
        } isTargeted: { targeted = $0 }
        .overlay {
            if targeted {
                RoundedRectangle(cornerRadius: 12).strokeBorder(Color.primary.opacity(0.35), style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                    .padding(6).allowsHitTesting(false)
            }
        }
    }

    // MARK: Pieces

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            if families.count > 1, let target {
                Picker("Write for", selection: Binding(get: { target }, set: { model.chosenTarget = $0 })) {
                    ForEach(families, id: \.self) { Text(verbatim: $0.name).tag($0) }
                }
                .fixedSize()
                .disabled(model.isWriting)
            } else if let target {
                Text("Write for \(target.name)").foregroundStyle(.secondary)
            }
            Spacer()
            if case .ready(let name, _) = model.availability {
                Text("Writer: \(name)").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var input: some View {
        switch model.gesture {
        case .rewrite: rewriteInput
        case .describe: describeInput
        }
    }

    private var rewriteInput: some View {
        VStack(alignment: .trailing, spacing: 4) {
            TextEditor(text: $model.text)
                .font(.body)
                .scrollContentBackground(.hidden)
                .focused($typing)
                .padding(6)
                .frame(minHeight: 96, maxHeight: 150)
                .background(card)
                .overlay(alignment: .topLeading) {
                    if model.text.isEmpty {
                        Text("Type or paste a prompt to improve")
                            .foregroundStyle(.tertiary).padding(.horizontal, 11).padding(.vertical, 6)
                            .allowsHitTesting(false)
                    }
                }
                .disabled(model.isWriting)
            Button("Use the Rack's Prompt") { model.useRackPrompt(rack) }
                .buttonStyle(.link).font(.callout)
                .disabled(rack.prompt.isEmpty || model.isWriting)
        }
    }

    private var describeInput: some View {
        VStack(spacing: 8) {
            if let picture = model.thumbnail {
                Image(nsImage: picture).resizable().scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: 190)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .accessibilityLabel(Text("The image to describe"))
            } else {
                VStack(spacing: 6) {
                    Image(systemName: "photo.badge.arrow.down").font(.system(size: 28, weight: .light)).foregroundStyle(.secondary)
                    Text("Drop an image here").foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 130)
            }
            Button(model.image == nil ? "Choose an Image…" : "Choose Another…") { model.chooseImage() }
                .buttonStyle(.link).font(.callout)
                .disabled(model.isWriting)
        }
        .padding(10)
        .frame(maxWidth: .infinity)
        .background(card)
    }

    private var card: some View {
        RoundedRectangle(cornerRadius: 10).fill(Palette.surface)
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Palette.hairline, lineWidth: 0.5))
    }

    private var action: some View {
        VStack(spacing: 6) {
            if model.isWriting {
                Button { model.stop() } label: { Text("Stop").frame(maxWidth: .infinity) }
                    .controlSize(.large)
                    .keyboardShortcut(.cancelAction)
            } else {
                Button { if let target { model.run(target: target) } } label: {
                    Text(model.gesture == .rewrite ? "Rewrite" : "Describe").fontWeight(.semibold).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!canRun)
            }
            if let pause {
                Label(pause.sentence, systemImage: pause.symbol).font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var current: LabModel.Result? { model.results[model.gesture] }

    private var result: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Result").font(.callout.weight(.semibold))
                Spacer()
                if model.gesture == .rewrite, !model.isWriting, let current, current.original != nil {
                    Button(model.editing ? "Show Changes" : "Edit") { model.editing.toggle() }
                        .buttonStyle(.link).font(.callout)
                }
            }
            Group {
                switch model.phase {
                case .loading(let fraction): loading(fraction)
                case .writing: writing
                case .idle: finished
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .background(card)
            if let failure = model.failure {
                Text(failure).font(.caption).foregroundStyle(Palette.failure)
            } else if model.stopped {
                Text("Stopped: an image started rendering.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func loading(_ fraction: Double?) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Preparing the writer…").foregroundStyle(.secondary)
            if let fraction { ProgressView(value: fraction) } else { ProgressView().controlSize(.small) }
        }
        .padding(12)
    }

    private var writing: some View {
        ScrollView {
            Text(model.streaming).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(12)
        }
        .overlay(alignment: .bottomTrailing) {
            ProgressView().controlSize(.small).padding(10)
        }
    }

    @ViewBuilder private var finished: some View {
        if let current {
            if model.gesture == .rewrite, let original = current.original, !model.editing {
                ScrollView {
                    Text(Self.shown(PromptDiff.pieces(from: original, to: current.text)))
                        .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(12)
                }
            } else {
                TextEditor(text: Binding(get: { model.results[model.gesture]?.text ?? "" },
                                         set: { model.results[model.gesture]?.text = $0 }))
                    .font(.body).scrollContentBackground(.hidden).padding(6)
            }
        } else {
            Text("The result appears here.").foregroundStyle(.tertiary).padding(12)
        }
    }

    private var footer: some View {
        HStack {
            Spacer()
            if model.sent {
                Label("Sent", systemImage: "checkmark").foregroundStyle(.secondary)
            }
            Button("Send to Rack") {
                if let current { model.send(current.text, to: rack) }
            }
            .controlSize(.large)
            .disabled(current?.text.isEmpty != false || model.isWriting || followingAnotherMac)
            .help("Replaces the rack's prompt. ⌘Z in the prompt field brings the old one back.")
        }
    }

    // MARK: The diff, drawn

    /// Additions in the accent on a faint wash (a colour alone would not do for everyone), removals
    /// struck through and quiet. A removed piece's leading space stays outside the strike.
    static func shown(_ pieces: [PromptDiff.Piece]) -> AttributedString {
        var out = AttributedString()
        for piece in pieces {
            switch piece.kind {
            case .same:
                out += AttributedString(piece.text)
            case .added:
                var a = AttributedString(piece.text)
                a.foregroundColor = Palette.live
                a.backgroundColor = Palette.live.opacity(0.12)
                out += a
            case .removed:
                let lead = piece.text.prefix { $0.isWhitespace }
                out += AttributedString(String(lead))
                var a = AttributedString(String(piece.text.dropFirst(lead.count)))
                a.strikethroughStyle = .single
                a.foregroundColor = Color.secondary
                out += a
            }
        }
        return out
    }
}
