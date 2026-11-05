// **Settings › Prompt Lab: the writer's instructions, the user's to read and to rewrite** (4/11) — the system turn
// sent to Qwen3-VL before each Rewrite and each Describe, one per family and gesture, ten in all.
//
//     INSTRUCTIONS
//     ┌───────────────────────────────────────────────────────┐
//     │ [Krea 2 ▾]                        [Rewrite|Describe]  │
//     │ ┌───────────────────────────────────────────────────┐ │
//     │ │ You rewrite a text-to-image prompt for Krea 2.    │ │
//     │ │ Keep every element of the user's text: …          │ │
//     │ └───────────────────────────────────────────────────┘ │
//     │ Yours: sent instead of ours.               [Restore]  │
//     └───────────────────────────────────────────────────────┘
//     Sent in English, as written, …
//
// - **The whole text, as it is sent** (his choice, 4/11): our three parts (the rules every family shares, the family's
//   style, the answer's contract) are one text here, and changing a shared rule changes it for this family alone.
// - **What is not touched follows ours** from version to version; « Restore » brings ours back
//   (`PromptWriterInstructions.Edits`, whose rules are the engine's and tested there).
// - **Shown in English whatever the app's language**: it is what the writer reads.
// - **The Prompt Lab says it** when an instruction of the model it writes for is the user's (`LabView`'s header), and
//   opens this pane on it.
// - **An AppKit text view, not a `TextEditor`**: the prompt's caret and selection (`Palette.ink`), which SwiftUI's editor
//   leaves in the system's blue.

import AppKit
import Siliconed
import SwiftUI

/// The pane: one instruction at a time, chosen by its model and its gesture.
struct LabInstructionsPane: View {
    @Bindable var selection: SettingsSelection
    let lab = LabModel.shared

    private var family: Family { selection.instructionFamily }
    private var gesture: PromptWriterInstructions.Gesture { selection.instructionGesture }

    var body: some View {
        SettingsGroup("Instructions",
                      footer: "Sent in English, as written, to the writer (Qwen3-VL) before each Rewrite or Describe for this model. What you leave untouched follows Siliconed’s own instructions from version to version.") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Picker("Model", selection: $selection.instructionFamily) {
                        ForEach(Family.allCases, id: \.self) { f in
                            // A dot after a model with an instruction of the user's.
                            let edited = lab.instructions.isEdited(f, .rewrite) || lab.instructions.isEdited(f, .describe)
                            Text(verbatim: edited ? "\(f.name)  •" : f.name).tag(f)
                        }
                    }
                    .labelsHidden().fixedSize()
                    Spacer()
                    Picker("Gesture", selection: $selection.instructionGesture) {
                        Text("Rewrite").tag(PromptWriterInstructions.Gesture.rewrite)
                        Text("Describe").tag(PromptWriterInstructions.Gesture.describe)
                    }
                    .pickerStyle(.segmented).labelsHidden().fixedSize().tint(Palette.control)
                }
                InstructionEditor(text: Binding(get: { lab.instructions.text(family, gesture) },
                                                set: { lab.instructions.set($0, for: family, gesture) }))
                    .padding(6)
                    .frame(height: 260)
                    .background(Palette.well, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                HStack(alignment: .firstTextBaseline) {
                    state.font(.caption).foregroundStyle(lab.instructions.isEdited(family, gesture) ? Palette.link : .secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                    Button("Restore") { lab.instructions.set(nil, for: family, gesture) }
                        .disabled(!lab.instructions.isEdited(family, gesture))
                        .help("Brings back Siliconed’s instruction for this model and gesture.")
                }
            }
            .padding(12)
        }
    }

    /// Whose text the field holds, and what is sent.
    private var state: Text {
        guard lab.instructions.isEdited(family, gesture) else { return Text("Siliconed’s instruction, as this version sends it.") }
        guard lab.instructions.sent(family, gesture) != nil else { return Text("Empty: Siliconed’s instruction is sent instead.") }
        return Text("Yours: sent instead of Siliconed’s.")
    }
}

/// **The instruction's field**: plain text in the callout font, the chart's caret and selection. A text set from outside
/// (another model or gesture, « Restore ») starts a new undo history: ⌘Z never brings back another instruction's text.
private struct InstructionEditor: NSViewRepresentable {
    @Binding var text: String

    func makeCoordinator() -> Coordinator { Coordinator(text: $text) }

    func makeNSView(context: NSViewRepresentableContext<Self>) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        let view = scroll.documentView as! NSTextView
        view.isRichText = false
        view.importsGraphics = false
        view.allowsUndo = true
        view.drawsBackground = false
        view.font = .preferredFont(forTextStyle: .callout)
        view.textContainerInset = .zero
        Palette.ink(view)
        view.string = text
        view.delegate = context.coordinator
        view.setAccessibilityLabel(String(localized: "Instruction"))
        scroll.drawsBackground = false
        scroll.borderType = .noBorder
        scroll.autohidesScrollers = true
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: NSViewRepresentableContext<Self>) {
        context.coordinator.text = $text
        guard let view = scroll.documentView as? NSTextView, view.string != text, !view.hasMarkedText() else { return }
        view.string = text
        context.coordinator.undo.removeAllActions()
    }

    @MainActor final class Coordinator: NSObject, NSTextViewDelegate {
        var text: Binding<String>
        /// The field's own history, not the Settings window's.
        let undo = UndoManager()
        init(text: Binding<String>) { self.text = text }

        func undoManager(for view: NSTextView) -> UndoManager? { undo }

        func textDidChange(_ notification: Notification) {
            guard let view = notification.object as? NSTextView else { return }
            text.wrappedValue = view.string
        }
    }
}
