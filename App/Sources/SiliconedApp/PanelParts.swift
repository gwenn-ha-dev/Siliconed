// What the two floating panels (`Inspectors.swift`) are made of: a section on its card, a `label ……… value` line.

import SwiftUI

/// A section title, a complement on the right, then its rows — on a card.
struct PanelSection<Content: View>: View {
    let title: LocalizedStringKey
    var right: String? = nil
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(title).textCase(.uppercase).font(.caption.weight(.semibold)).tracking(0.7).foregroundStyle(Palette.secondary)
                Spacer()
                if let right { Text(right).font(.caption).foregroundStyle(Palette.secondary).lineLimit(1) }
            }
            content
        }
        .padding(11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor).opacity(0.8)))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Palette.hairline))
    }
}

/// `label ……… value`, and a discreet detail below.
struct StatLine: View {
    let label: Text, value: String
    var detail: String?
    init(_ label: LocalizedStringKey, _ value: String, detail: String? = nil) {
        self.label = Text(label); self.value = value; self.detail = detail
    }
    init(verbatim label: String, _ value: String, detail: String? = nil) {
        self.label = Text(verbatim: label); self.value = value; self.detail = detail
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(alignment: .firstTextBaseline) {
                label.foregroundStyle(Palette.secondary)
                Spacer(minLength: 12)
                Text(value).monospacedDigit().multilineTextAlignment(.trailing)
            }
            if let detail {
                Text(detail).font(.caption2).foregroundStyle(Palette.secondary).frame(maxWidth: .infinity, alignment: .trailing)
            }
        }
        .font(.callout)
    }
}
