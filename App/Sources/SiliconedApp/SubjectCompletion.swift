// The completion at `@`: the list of subjects under the caret.

import AppKit
import Siliconed
import SwiftUI

/// **The list under the caret**: the subjects whose name starts with what follows the `@`, each with
/// its thumbnail and kind. A borderless child window that never takes the keyboard: the text view
/// keeps it, and forwards ↑ ↓ Return Tab Esc.
@MainActor
final class SubjectCompletion {
    private var panel: NSPanel?
    private let model = Model()

    @MainActor @Observable
    final class Model {
        var items: [(subject: NamedSubjects.Subject, thumbnail: NSImage?)] = []
        var selected = 0
        var take: ((NamedSubjects.Subject) -> Void)?
    }

    var isShown: Bool { panel?.isVisible ?? false }

    /// The list's measures (`CompletionList`), in points: its width, a subject's row, the space between two, the margin
    /// around them — the panel is sized from them.
    static let width: CGFloat = 240, row: CGFloat = 28, spacing: CGFloat = 2, inset: CGFloat = 5

    func show(_ items: [(NamedSubjects.Subject, NSImage?)], under caret: NSRect, parent: NSWindow,
              take: @escaping (NamedSubjects.Subject) -> Void) {
        let previous = model.items.map(\.subject.id)
        model.items = items.map { (subject: $0.0, thumbnail: $0.1) }
        if previous != model.items.map(\.subject.id) { model.selected = 0 }
        model.take = take
        let p = panel ?? makePanel()
        let height = CGFloat(items.count) * (Self.row + Self.spacing) + 2 * Self.inset
        p.setContentSize(NSSize(width: Self.width, height: height))
        p.setFrameTopLeftPoint(NSPoint(x: caret.minX - 6, y: caret.minY - 4))
        if p.parent == nil { parent.addChildWindow(p, ordered: .above) }
        p.orderFront(nil)
    }

    func move(_ by: Int) {
        guard !model.items.isEmpty else { return }
        model.selected = (model.selected + by + model.items.count) % model.items.count
    }

    func accept() {
        guard model.items.indices.contains(model.selected) else { close(); return }
        model.take?(model.items[model.selected].subject)
    }

    func close() {
        if let p = panel {
            p.parent?.removeChildWindow(p)
            p.orderOut(nil)
        }
    }

    private func makePanel() -> NSPanel {
        let p = NSPanel(contentRect: .zero, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        p.isFloatingPanel = true
        p.hasShadow = true
        p.backgroundColor = .clear
        p.isOpaque = false
        p.isReleasedWhenClosed = false
        p.contentView = NSHostingView(rootView: CompletionList(model: model))
        panel = p
        return p
    }
}

private struct CompletionList: View {
    let model: SubjectCompletion.Model

    var body: some View {
        VStack(alignment: .leading, spacing: SubjectCompletion.spacing) {
            ForEach(Array(model.items.enumerated()), id: \.element.subject.id) { i, item in
                HStack(spacing: 8) {
                    SubjectChip(name: item.subject.name, kind: item.subject.kind, thumbnail: item.thumbnail,
                                pictures: item.subject.pictures.count)
                    Spacer(minLength: 4)
                    Text(item.subject.kind.label).font(.caption).foregroundStyle(Palette.secondary)
                }
                .padding(.horizontal, 6).frame(height: SubjectCompletion.row)
                .background(RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(i == model.selected ? Palette.live.opacity(0.22) : .clear))
                .contentShape(Rectangle())
                .onTapGesture { model.take?(item.subject) }
                .onHover { if $0 { model.selected = i } }
            }
        }
        .padding(SubjectCompletion.inset)
        .frame(width: SubjectCompletion.width, alignment: .topLeading)
        .background(VisualEffect().clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous)))
        .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(Palette.hairline))
    }
}

/// The menu's material behind the list.
private struct VisualEffect: NSViewRepresentable {
    func makeNSView(context: NSViewRepresentableContext<Self>) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = .menu
        v.blendingMode = .behindWindow
        v.state = .active
        return v
    }
    func updateNSView(_ nsView: NSVisualEffectView, context: NSViewRepresentableContext<Self>) {}
}
