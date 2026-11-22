// **Settings ▸ Macs ▸ This Mac: its B and its compilations.**
//
//     THIS MAC
//     ┌─────────────────────────────────────────────────────────┐
//     │ [▣] Mac mini                                            │
//     │     M1 Pro · 16 GB                                      │
//     │ [▤] Memory for Renders                        [Default] │
//     │     12 GB, chosen; the default is 10 GB.                │
//     │     4 GB ────────────────────●──────── 12 GB            │
//     │ [▤] Compiled for This Mac                      [Erase…] │
//     │     1.2 GB · compiled again by each model's next render │
//     └─────────────────────────────────────────────────────────┘
//
// - **B** (`AppState.memoryBudget`): from 4 GB to the RAM less 4 (`MemoryBudget.range`), a step a gigabyte; on a Mac of
//   8 GB, no slider: there is no choice. The engine's next process takes it, a render under way keeps its own.
// - **The compilations** (`AppState.compiledBytes`, `eraseCompiled`): what Core AI compiled of the engine's functions
//   for this Mac, weighed when the pane opens; erased by the engine between two pieces of work, after a confirmation.

import Siliconed
import SwiftUI

/// **B, the memory a render may hold**, set by the user within this Mac's range.
struct MemoryBudgetRow: View {
    @Bindable var app: AppState
    private static let gib = Double(1 << 30)

    var body: some View {
        let range = MemoryBudget.range, chosen = app.memoryBudget, current = chosen ?? MemoryBudget.standard
        let fixed = range.lowerBound == range.upperBound
        SettingRow(Text("Memory Limit"), subtitle(chosen: chosen, current: current, fixed: fixed),
                   detail: "The most the engine holds while it renders. More lets larger images and models run whole; less leaves more to the other apps. A render never goes past it; one under way keeps the memory it began with.") {
            IconTile(symbol: "memorychip", color: Palette.sky, side: 26)
        } trailing: {
            if chosen != nil { Button("Default") { app.memoryBudget = nil }.controlSize(.small) }
        } below: {
            if !fixed {
                HStack(spacing: 8) {
                    Text(verbatim: Figures.memory(range.lowerBound)).font(.caption).foregroundStyle(Palette.secondary)
                    Slider(value: Binding(get: { Double(current) / Self.gib },
                                          set: { app.memoryBudget = Int($0.rounded() * Self.gib) }),
                           in: Double(range.lowerBound) / Self.gib...Double(range.upperBound) / Self.gib, step: 1) {
                        Text("Memory Limit")
                    }
                    .labelsHidden().controlSize(.small).tint(Palette.control)
                    Text(verbatim: Figures.memory(range.upperBound)).font(.caption).foregroundStyle(Palette.secondary)
                }
            }
        }
    }

    private func subtitle(chosen: Int?, current: Int, fixed: Bool) -> Text {
        if fixed { return Text("\(Figures.memory(current)): this Mac's memory leaves no choice.") }
        if chosen == nil { return Text("\(Figures.memory(current)), the default: three quarters of the memory, less 2 GB.") }
        return Text("\(Figures.memory(current)), chosen; the default is \(Figures.memory(MemoryBudget.standard)).")
    }
}

/// **What Core AI compiled for this Mac**: its weight, and « Erase… ».
struct CompiledRow: View {
    let app: AppState
    @State private var confirming = false

    var body: some View {
        SettingRow(Text("Compiled for This Mac"), subtitle,
                   detail: "Core AI compiles each of the engine’s functions for this Mac and keeps it, so that the next render starts at once. Erasing frees the disk: each model’s next render compiles again what it needs, from a few seconds to a minute. Siliconed erases them by itself after an update.") {
            IconTile(symbol: "cpu", color: Palette.peach, side: 26)
        } trailing: {
            Button("Erase…") { confirming = true }
                .controlSize(.small)
                .unavailable(app.erasingCompiled ? String(localized: "Erasing…")
                             : app.compiledBytes == 0 ? String(localized: "Nothing compiled yet.") : nil)
        } below: {
            EmptyView()
        }
        .task { await app.weighCompiled() }
        .confirmationDialog("Erase what was compiled for this Mac?", isPresented: $confirming) {
            Button("Erase", role: .destructive) { app.eraseCompiled() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Each model’s next render compiles again what it needs, from a few seconds to a minute. A render under way finishes first.")
        }
    }

    private var subtitle: Text? {
        if app.erasingCompiled { return Text("Erasing…") }
        guard let bytes = app.compiledBytes else { return nil }
        if bytes == 0 { return Text("Nothing compiled yet.") }
        return Text("\(Figures.disk(bytes)) · compiled again by each model’s next render")
    }
}
