// The rack's box furniture (title, ports, fields, tags, cost) and the two input nodes,
// prompt and images. Split from `Rack.swift`, whose header says how the boxes behave.

import AppKit
import Siliconed
import SwiftUI

// MARK: - A box

/// **A stage's box**: its symbol and name, its state on the right, then what fills it. Idle: a quiet
/// card. Working: the accent's edge and a faint wash of it. Done: a discreet check. Refused: red. It
/// takes the width it is given.
struct StageBox<Content: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let state: StageState
    /// What the stage does, in plain words: shown when the pointer rests on its name.
    let explanation: LocalizedStringKey
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            BoxTitle(title: title, symbol: symbol, tint: state == .active ? Palette.live : .secondary,
                     explanation: explanation) {
                switch state {
                case .idle: EmptyView()
                case .active: ProgressView().controlSize(.mini)
                case .done: Image(systemName: "checkmark").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
                case .failed: Image(systemName: "exclamationmark.triangle.fill").font(.caption).foregroundStyle(Palette.failure)
                }
            }
            content
        }
        .boxed(edge: edge, width: state == .idle || state == .done ? 1 : 1.5,
               wash: state == .active ? Palette.live : state == .failed ? Palette.failure : nil)
        .animation(.easeInOut(duration: 0.25), value: state)
    }

    private var edge: Color {
        switch state {
        case .idle, .done: Palette.hairline
        case .active: Palette.live
        case .failed: Palette.failure
        }
    }
}

/// A box's first line: a small tinted chip with its symbol, its name, and what it says on the right.
private struct BoxTitle<Trailing: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let tint: Color
    var explanation: LocalizedStringKey? = nil
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 7) {
            HStack(spacing: 7) {
                Image(systemName: symbol)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(tint)
                    .frame(width: 20, height: 20)
                    .background(RoundedRectangle(cornerRadius: 5, style: .continuous).fill(tint.opacity(0.16)))
                Text(title).font(.callout.weight(.semibold))
                    .lineLimit(2).fixedSize(horizontal: false, vertical: true)
            }
            .contentShape(Rectangle())
            .help(explanation.map { Text($0) } ?? Text(verbatim: ""))
            Spacer(minLength: 4)
            trailing
        }
    }
}

extension View {
    /// The box: padding, the card's fill, its edge — and, while it runs or failed, a faint wash of
    /// that color.
    fileprivate func boxed(edge: Color, width: CGFloat = 1, wash: Color? = nil) -> some View {
        padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(boxShape.fill(Palette.surface))
            .background(boxShape.fill(Color(nsColor: .windowBackgroundColor)))
            .overlay { if let wash { boxShape.fill(wash.opacity(0.06)).allowsHitTesting(false) } }
            .overlay(boxShape.strokeBorder(edge, lineWidth: width))
            .shadow(color: .black.opacity(0.08), radius: 2, y: 1)
    }
}

/// **A box the signal starts in** — the prompt, an edit's images: its chip in the color of the wire
/// it sends; its edge takes the accent while it is being typed in (`focused`).
struct PortBox<Trailing: View, Content: View>: View {
    let title: LocalizedStringKey
    let symbol: String
    let kind: WireKind
    var focused = false
    @ViewBuilder var trailing: Trailing
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            BoxTitle(title: title, symbol: symbol, tint: kind.color) {
                trailing.font(.caption).monospacedDigit().foregroundStyle(.secondary)
            }
            content
        }
        .boxed(edge: focused ? Palette.live : Palette.hairline, width: focused ? 1.5 : 1)
        .animation(.easeInOut(duration: 0.15), value: focused)
    }
}

extension PortBox where Trailing == EmptyView {
    init(title: LocalizedStringKey, symbol: String, kind: WireKind, focused: Bool = false, @ViewBuilder content: () -> Content) {
        self.init(title: title, symbol: symbol, kind: kind, focused: focused, trailing: { EmptyView() }, content: content)
    }
}

/// **A setting in a box**: its name on the left, its controls on the right — or under it, when the
/// box is too narrow for both on one line.
struct Field<Controls: View>: View {
    let title: LocalizedStringKey
    @ViewBuilder var controls: Controls

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
                Text(title).foregroundStyle(.secondary)
                Spacer(minLength: 8)
                controls
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(title).foregroundStyle(.secondary)
                HStack(spacing: 8) { controls }
            }
        }
        .font(.callout)
    }
}

/// A variation's strength, in words: the app's two, or a percentage for one set elsewhere.
enum VariationWords {
    static func label(_ strength: Double) -> String {
        switch Variation.Amount.named(strength) {
        case .subtle: return String(localized: "Subtle")
        case .strong: return String(localized: "Strong")
        case nil: break
        }
        return Figures.percent(Int((strength * 100).rounded()))
    }
}

/// **A module the family sets**: its name in a slot, locked. The slot is where a choice will be
/// offered once there is one to make — not an empty menu now.
struct ModuleSlot: View {
    let name: String?
    var body: some View {
        HStack(spacing: 5) {
            Text(name ?? "—").lineLimit(1).truncationMode(.middle)
            Spacer(minLength: 2)
            Image(systemName: "lock.fill").imageScale(.small).foregroundStyle(.secondary)
        }
        .font(.callout)
        .padding(.horizontal, 8).padding(.vertical, 5)
        .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Palette.well))
        .help("Set by the model's family")
    }
}

/// A label: a small capsule, neutral — or with a dot of color (the latent's wire).
struct Tag: View {
    let text: Text
    var symbol: String? = nil
    var dot: Color? = nil
    var body: some View {
        HStack(spacing: 4) {
            if let dot { Circle().fill(dot).frame(width: 6, height: 6) }
            if let symbol { Image(systemName: symbol).imageScale(.small) }
            text
        }
        .font(.caption2.weight(.medium))
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .padding(.horizontal, 7).padding(.vertical, 2.5)
        .background(Capsule().fill(Palette.well))
    }
}

/// The time a stage costs: what it just took in the render on screen (no « ≈ »), or the estimate
/// learned on this Mac — or nothing yet.
struct Cost: View {
    let seconds: Double?
    let measured: Bool
    init(_ cost: (seconds: Double?, measured: Bool)) { seconds = cost.seconds; measured = cost.measured }
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "clock").imageScale(.small)
            Text(seconds.map { (measured ? "" : "≈ ") + Figures.seconds($0) } ?? "—").monospacedDigit()
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .help(measured ? "What this stage just took"
                       : "Learned on this Mac: the first render at this size, with this model, sets the estimate")
    }
}

// MARK: - The inputs

/// **The prompt is typed in its box**: what to make — or, when editing, what changes. A real text view
/// (`TextEditor`, an `NSTextView`): ⌘A, ⌘C, ⌘Z, the spelling, a long prompt that scrolls. The grip
/// under it gives it the height one wants; Esc leaves it (the arrows then walk the strip).
struct PromptNode: View {
    @Bindable var app: AppState
    @FocusState private var typing: Bool
    /// The named subjects' field says itself when it has the keyboard (`PromptField`).
    @State private var fieldTyping = false
    @State private var height: CGFloat = 72
    @GestureState private var stretch: CGFloat = 0
    private static let heights: ClosedRange<CGFloat> = 44...480

    var body: some View {
        PortBox(title: "Prompt", symbol: "text.quote", kind: .text, focused: typing || fieldTyping) {
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    // Named subjects on: the field that draws `@name` as chips. Off: the app as it was.
                    if app.subjectsActive {
                        PromptField(app: app, focused: $fieldTyping)
                    } else {
                        TextEditor(text: $app.prompt)
                            .font(.body)
                            .scrollContentBackground(.hidden)
                            .scrollIndicators(.automatic)
                            .focused($typing)
                    }
                    if app.prompt.isEmpty {
                        Text(app.references.isEmpty ? "Describe the image…" : "Say what changes…")
                            .font(.body).foregroundStyle(.tertiary)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
                .frame(height: min(max(height + stretch, Self.heights.lowerBound), Self.heights.upperBound))
                grip
                if app.subjectsActive, let rewrite = app.shownRewrite { ModelPromptRow(app: app, rewrite: rewrite) }
            }
        }
    }

    /// The handle: dragged, the field grows or shrinks; its cursor says so.
    private var grip: some View {
        Capsule().fill(Color.secondary.opacity(0.5)).frame(width: 34, height: 4)
            .frame(maxWidth: .infinity).frame(height: 12)
            .contentShape(Rectangle())
            .onHover { inside in if inside { NSCursor.resizeUpDown.push() } else { NSCursor.pop() } }
            .gesture(DragGesture(minimumDistance: 1)
                .updating($stretch) { v, s, _ in s = v.translation.height }
                .onEnded { v in
                    height = min(max(height + v.translation.height, Self.heights.lowerBound), Self.heights.upperBound)
                })
            .help("Drag to resize the prompt")
            .padding(.bottom, -6)
    }
}

/// **The images the prompt works on, numbered as it names them**: image 1 is the one edited — the
/// output takes its format —, the others are what the prompt borrows from. Added by the button, by a
/// drop here or on the canvas (there, a PNG Siliconed wrote restores its settings instead), or by
/// « Edit This Image » in the history; reordered by dragging a
/// thumbnail onto another, or from its menu. Only for a model that edits (rule 2: the inputs follow
/// the model). **With Named Subjects on, the box is images OR subjects** (`AppState.referenceMode`):
/// numbered images, or a library of named subjects — as many as one wants — that the prompt cites by
/// `@name`, 3 at most. The title counts what the next render sends.
struct ImagesNode: View {
    @Bindable var app: AppState
    private var subjects: Bool { app.subjectsActive }
    private var used: Int { subjects ? app.subjectImageCount : app.references.count }
    var body: some View {
        PortBox(title: "Images", symbol: "photo.on.rectangle", kind: .image) {
            Text(used == 0 ? String(localized: "up to \(app.maxReferences) images") : "\(used)/\(app.maxReferences)")
                .foregroundStyle(used > app.maxReferences ? Palette.failure : .secondary)
        } content: {
            if app.subjectsAvailable {
                Picker("Images", selection: $app.referenceMode) {
                    Text("Images").tag(AppState.ReferenceMode.images)
                    Text("Subjects").tag(AppState.ReferenceMode.subjects)
                }
                .pickerStyle(.segmented).labelsHidden()
                .padding(.bottom, 2)
            }
            if subjects { SubjectShelf(app: app) } else { images }
        }
        .dropDestination(for: DroppedImage.self) { items, _ in
            if subjects {
                // Here an image is named: the sheet opens on the first one.
                switch items.first {
                case .file(let url)? where !AppState.importable(url): app.nameSubject(.file(url))
                case .png(let data)?: app.nameSubject(.png(data))
                default: return false
                }
                return true
            }
            var files: [URL] = []
            for item in items {
                switch item {
                case .file(let url) where !AppState.importable(url): files.append(url)
                case .file: break
                case .png(let data): app.addReference(png: data)
                }
            }
            app.addReferences(files)
            return items.contains { if case .file(let u) = $0 { !AppState.importable(u) } else { true } }
        }
    }

    @ViewBuilder private var images: some View {
        // One more image beyond this Mac's memory (B): the « + » greyed out, the sentence below.
        let refusal = app.canAddReference ? app.nextReferenceRefusal() : nil
        HStack(alignment: .top, spacing: 6) {
            ForEach(Array(app.references.enumerated()), id: \.element.id) { n, reference in
                ReferenceThumbnail(app: app, reference: reference, number: n + 1,
                                   last: n == app.references.count - 1)
            }
            if app.canAddReference {
                Button { app.chooseReferences() } label: {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .strokeBorder(style: StrokeStyle(lineWidth: 1.25, dash: [4, 3]))
                        .foregroundStyle(WireKind.image.color.opacity(0.7))
                        .overlay(Image(systemName: "plus").foregroundStyle(WireKind.image.color))
                        .frame(width: ReferenceThumbnail.side, height: ReferenceThumbnail.side)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(refusal != nil).opacity(refusal == nil ? 1 : 0.35)
                .help(app.references.isEmpty ? "The image to edit — or drop or paste it here (⌘V)"
                                             : "Add an image the prompt can draw from — or drop or paste it here (⌘V)")
            }
            Spacer(minLength: 0)
        }
        Group {
            if let refusal {
                Text(refusal)
            } else if app.references.isEmpty {
                Text("Optional. Add an image and the prompt becomes an instruction: “change her jacket to red”.")
            } else if app.promptNamesImages && app.maxReferences > 1 {
                Text("Image 1 is the one edited. The prompt can name them: “put the dog of image 2 next to the woman in image 1”.")
                    .help("The model reads them as <image1>, <image2>… in front of the prompt")
            } else {
                Text("Image 1 is the one edited: the prompt says what changes.")
            }
        }
        .font(.caption).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
}
