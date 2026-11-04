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
                TextEditor(text: Binding(get: { lab.instructions.text(family, gesture) },
                                         set: { lab.instructions.set($0, for: family, gesture) }))
                    .font(.callout)
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .frame(height: 260)
                    .background(Palette.well, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                    .accessibilityLabel(Text("Instruction"))
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
