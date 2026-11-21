// **The Mac that computes, drawn as macOS draws it in About This Mac** — beside « Chain », where the rack says whose
// measure plans the render.
//
//     hw.model "MacBookPro18,3"  ──tag of class com.apple.device-model-code──▶  UTType com.apple.macbookpro-14-2021
//                                                                                     │ NSWorkspace.icon(for:)
//                                                                                     ▼
//                                                                         the icon CoreTypes holds for that model
//
// - **A Mac newer than its macOS has no type yet**: the tag then resolves to a dynamic type (`dyn.…`, `isDynamic`),
//   whose icon is a blank document — the app's own icon stands in for it, never that blank.
// - **In a mirror it is the lender's model** (`RemoteMachine.hardwareModel`), the Mac whose measure plans the render;
//   a lender older than the field sends none, and the app's icon stands in again.
// - **The icon has the room of a square around it**: a laptop fills its width and half its height — the frame is
//   sized for that, not for a symbol.

import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct MacIcon: View {
    /// `hw.model` of the Mac to draw; `nil`, or unknown to this macOS: the app's icon.
    let hardwareModel: String?

    var body: some View {
        Image(nsImage: Self.image(hardwareModel)).resizable().interpolation(.high).aspectRatio(contentMode: .fit)
    }

    private static var cache: [String: NSImage] = [:]

    static func image(_ hardwareModel: String?) -> NSImage {
        guard let model = hardwareModel else { return NSApp.applicationIconImage }
        if let image = cache[model] { return image }
        let image: NSImage
        if let type = UTType(tag: model, tagClass: UTTagClass(rawValue: "com.apple.device-model-code"), conformingTo: nil),
           !type.isDynamic {
            image = NSWorkspace.shared.icon(for: type)
        } else {
            image = NSApp.applicationIconImage
        }
        cache[model] = image
        return image
    }
}
