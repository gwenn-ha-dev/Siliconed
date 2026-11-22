// The layout manager that draws a subject's chip and a series' group in the prompt.

import AppKit
import Siliconed
import SwiftUI

/// **Draws the chips**: the fill behind the text, the thumbnail and the edge above it — the edge after
/// the selection, so a selected chip stays recognizable.
final class SubjectLayoutManager: NSLayoutManager {
    var thumbnail: ((String) -> NSImage?)?

    override func drawBackground(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
        chips(in: glyphsToShow, at: origin) { rect, _, color, _, _ in
            color.withAlphaComponent(0.14).setFill()
            NSBezierPath(roundedRect: rect, xRadius: rect.height / 2, yRadius: rect.height / 2).fill()
        }
        groups(in: glyphsToShow, at: origin) { rect in
            Palette.bracketWashInk.setFill()
            NSBezierPath(roundedRect: rect, xRadius: 5, yRadius: 5).fill()
        }
        super.drawBackground(forGlyphRange: glyphsToShow, at: origin)
    }

    override func drawGlyphs(forGlyphRange glyphsToShow: NSRange, at origin: NSPoint) {
        super.drawGlyphs(forGlyphRange: glyphsToShow, at: origin)
        let accent = Palette.bracketInk
        groups(in: glyphsToShow, at: origin) { rect in
            let edge = NSBezierPath(roundedRect: rect.insetBy(dx: 0.5, dy: 0.5), xRadius: 4.5, yRadius: 4.5)
            edge.lineWidth = 1
            accent.withAlphaComponent(0.7).setStroke()
            edge.stroke()
        }
        dots(in: glyphsToShow, at: origin) { center in
            accent.setFill()
            NSBezierPath(ovalIn: NSRect(x: center.x - 1.75, y: center.y - 1.75, width: 3.5, height: 3.5)).fill()
        }
        chips(in: glyphsToShow, at: origin) { rect, key, color, first, inert in
            let edge = NSBezierPath(roundedRect: rect.insetBy(dx: 0.5, dy: 0.5), xRadius: rect.height / 2 - 0.5, yRadius: rect.height / 2 - 0.5)
            edge.lineWidth = 1
            color.setStroke()
            edge.stroke()
            guard first else { return }
            let side = PromptStyle.thumbnail
            let box = NSRect(x: rect.minX + PromptStyle.leading, y: rect.midY - side / 2, width: side, height: side)
            NSGraphicsContext.saveGraphicsState()
            NSBezierPath(roundedRect: box, xRadius: 3.5, yRadius: 3.5).addClip()
            if let image = self.thumbnail?(key) {
                // Aspect fill: the frame's middle, the thumbnail's square.
                let s = max(side / image.size.width, side / image.size.height)
                let w = image.size.width * s, h = image.size.height * s
                image.draw(in: NSRect(x: box.midX - w / 2, y: box.midY - h / 2, width: w, height: h),
                           from: .zero, operation: .sourceOver, fraction: inert ? 0.35 : 1, respectFlipped: true, hints: nil)
            } else {
                color.withAlphaComponent(0.5).setFill()
                box.fill()
            }
            NSGraphicsContext.restoreGraphicsState()
        }
    }

    /// Each chip's rectangles in the glyphs shown (a chip that wraps has one per line), centered on
    /// the text's line whatever the line spacing.
    private func chips(in glyphs: NSRange, at origin: NSPoint, _ draw: (NSRect, String, NSColor, Bool, Bool) -> Void) {
        guard let storage = textStorage, let container = textContainers.first else { return }
        let characters = characterRange(forGlyphRange: glyphs, actualGlyphRange: nil)
        let font = PromptStyle.font
        var found: [(NSRect, String, NSColor, Bool, Bool)] = []
        storage.enumerateAttribute(.subjectChip, in: characters) { value, range, _ in
            guard let tag = value as? String, let colon = tag.firstIndex(of: ":") else { return }
            let key = String(tag[tag.index(after: colon)...])
            let inert = storage.attribute(.subjectChipInert, at: range.location, effectiveRange: nil) != nil
            let color = inert ? Palette.secondaryInk : (NamedSubjects.Kind(rawValue: String(tag[..<colon])) ?? .person).nsColor
            let glyphRange = glyphRange(forCharacterRange: range, actualCharacterRange: nil)
            var first = true
            enumerateLineFragments(forGlyphRange: glyphRange) { lineRect, _, _, lineGlyphs, _ in
                let part = NSIntersectionRange(lineGlyphs, glyphRange)
                guard part.length > 0 else { return }
                let bounds = self.boundingRect(forGlyphRange: part, in: container)
                let baseline = lineRect.minY + self.location(forGlyphAt: part.location).y
                let top = baseline - font.ascender - 2, bottom = baseline - font.descender + 2
                let rect = NSRect(x: bounds.minX + origin.x, y: top + origin.y,
                                  width: bounds.width, height: bottom - top)
                found.append((rect, key, color, first, inert))
                first = false
            }
        }
        for (rect, key, color, first, inert) in found { draw(rect, key, color, first, inert) }
    }

    /// Each group's rectangles in the glyphs shown, one per line it spans — the line's text height and a point around.
    private func groups(in glyphs: NSRange, at origin: NSPoint, _ draw: (NSRect) -> Void) {
        guard let storage = textStorage, let container = textContainers.first else { return }
        let characters = characterRange(forGlyphRange: glyphs, actualGlyphRange: nil)
        let font = PromptStyle.font
        var found: [NSRect] = []
        storage.enumerateAttribute(.seriesGroup, in: characters) { value, range, _ in
            guard value != nil else { return }
            let glyphRange = glyphRange(forCharacterRange: range, actualCharacterRange: nil)
            enumerateLineFragments(forGlyphRange: glyphRange) { lineRect, _, _, lineGlyphs, _ in
                let part = NSIntersectionRange(lineGlyphs, glyphRange)
                guard part.length > 0 else { return }
                let bounds = self.boundingRect(forGlyphRange: part, in: container)
                let baseline = lineRect.minY + self.location(forGlyphAt: part.location).y
                let top = baseline - font.ascender - 1, bottom = baseline - font.descender + 1
                found.append(NSRect(x: bounds.minX + origin.x, y: top + origin.y, width: bounds.width, height: bottom - top))
            }
        }
        for rect in found { draw(rect) }
    }

    /// The centre of each group's `|` shown: where its dot goes, at the words' middle height.
    private func dots(in glyphs: NSRange, at origin: NSPoint, _ draw: (NSPoint) -> Void) {
        guard let storage = textStorage, let container = textContainers.first else { return }
        let characters = characterRange(forGlyphRange: glyphs, actualGlyphRange: nil)
        let font = PromptStyle.font
        var found: [NSPoint] = []
        storage.enumerateAttribute(.seriesSeparator, in: characters) { value, range, _ in
            guard value != nil else { return }
            for i in range.location..<NSMaxRange(range) {
                let glyph = glyphIndexForCharacter(at: i)
                let bounds = boundingRect(forGlyphRange: NSRange(location: glyph, length: 1), in: container)
                let line = lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil)
                let baseline = line.minY + location(forGlyphAt: glyph).y
                found.append(NSPoint(x: bounds.midX + origin.x, y: baseline - font.xHeight / 2 + origin.y))
            }
        }
        for p in found { draw(p) }
    }
}
