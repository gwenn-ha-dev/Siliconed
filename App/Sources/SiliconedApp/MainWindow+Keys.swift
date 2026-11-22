// `MainWindow`: the keys of the strip and of the crop frame.

import AppKit
import Siliconed
import SilicontrolHelp
import SwiftUI

extension MainWindow {
    /// The modifiers that make a shortcut: Caps Lock, the numeric pad and the function flag are states
    /// of the keyboard, not of the chord.
    private static func chord(_ event: NSEvent) -> NSEvent.ModifierFlags {
        event.modifierFlags.intersection(.deviceIndependentFlagsMask).subtracting([.capsLock, .numericPad, .function])
    }

    func watchKeys() {
        guard keyMonitor == nil else { return }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self, let w = window, event.window === w, w.attachedSheet == nil else { return event }
            // Esc leaves the field being typed in: the keys go back to the strip, and nothing folds. The prompt's list of
            // subjects, shown, is closed first, by the field.
            if w.firstResponder is NSText {
                guard event.keyCode == 53, Self.chord(event).isEmpty else { return event }
                if (w.firstResponder as? PromptTextView)?.listShown?() == true { return event }
                w.makeFirstResponder(nil)
                return nil
            }
            if app.cropping != nil { return cropKey(event) ? nil : event }
            // ⌘V by its character (AZERTY): an image on the pasteboard goes to the Images box.
            if Self.chord(event) == .command,
               event.charactersIgnoringModifiers?.lowercased() == "v" {
                return app.pasteImage() ? nil : event
            }
            // A control that has the keyboard (a slider, a stepper, a menu, a button reached by Tab) keeps its
            // keys: ← → adjust it, Space presses it, ⌫ is not the history's.
            if w.firstResponder is NSControl { return event }
            // The grid emptied (its images all removed) still closes by Esc, as its help says.
            if app.history.isEmpty, app.historyGrid, event.keyCode == 53, Self.chord(event).isEmpty {
                withAnimation(HistoryGrid.motion) { app.historyGrid = false }
                return nil
            }
            guard !app.history.isEmpty else { return event }
            // C opens the crop, by its character: on an AZERTY keyboard the key code of C is another letter.
            if Self.chord(event).isEmpty,
               event.charactersIgnoringModifiers?.lowercased() == "c" {
                app.beginCropOfShownImage()
                return nil
            }
            return key(event) ? nil : event
        }
    }

    /// **The keys while the crop frame is up**: Return (or Enter) crops, Esc leaves; the arrows, which
    /// would walk the strip, and Space, Delete and Home/End stay out of it. ⌘-keys go on to the menu.
    private func cropKey(_ event: NSEvent) -> Bool {
        let mods = Self.chord(event)
        guard mods.isEmpty else { return false }
        switch Int(event.keyCode) {
        case 36, 76: app.commitCrop()                                   // Return, Enter
        case 53: app.cancelCrop()                                       // Esc
        case 123, 124, 125, 126, 115, 119, 49, 51, 117: break            // ← → ↓ ↑ Home End Space ⌫ ⌦
        default: return event.charactersIgnoringModifiers?.lowercased() == "c"
        }
        return true
    }

    private func key(_ event: NSEvent) -> Bool {
        let mods = Self.chord(event)
        let shift = mods == .shift
        switch Int(event.keyCode) {
        // The curtain shown, ← → move its line — the switch shown, show A or B (`Curtain.swift`); ⇧ still marks a run, Esc leaves.
        case 123 where mods.isEmpty && Curtain.shown(app): Curtain.shared.nudge(-1)
        case 124 where mods.isEmpty && Curtain.shown(app): Curtain.shared.nudge(1)
        case 123 where mods.isEmpty || shift: app.chooseAdjacent(-1, extending: shift)  // ←
        case 124 where mods.isEmpty || shift: app.chooseAdjacent(1, extending: shift)   // →
        case 115 where mods.isEmpty: app.chooseEnd(oldest: false)                       // Home
        case 119 where mods.isEmpty: app.chooseEnd(oldest: true)                        // End
        case 49 where mods.isEmpty:                                                     // Space
            withAnimation(MainView.away) { app.componentsAway.toggle() }
        case 51 where mods.isEmpty, 117 where mods.isEmpty:                             // ⌫, ⌦
            guard !(app.inProgress && app.followsRender) else { return false }
            app.delete(app.targets)
        case 53 where mods.isEmpty:                                                     // Esc
            if app.historyGrid { withAnimation(HistoryGrid.motion) { app.historyGrid = false }; return true }
            if app.componentsAway { withAnimation(MainView.away) { app.componentsAway = false }; return true }
            if app.showsOriginal { app.comparesOriginal = false; return true }
            guard app.marked.count > 1 else { return false }
            app.unmark()
        // ⌘A by its character, not its key code: code 0 is A on a QWERTY keyboard but Q on an AZERTY
        // one — read as a position, it took ⌘Q from the menu.
        case _ where mods == .command && event.charactersIgnoringModifiers?.lowercased() == "a": app.markAll()
        default: return false
        }
        return true
    }
}
