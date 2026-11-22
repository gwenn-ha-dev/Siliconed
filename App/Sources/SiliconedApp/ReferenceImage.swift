// An image that comes into the app (an edit's reference, a starting image, an image dropped) or leaves it by a drag.

import AppKit
import ImageIO
import Siliconed
import SwiftUI
import UniformTypeIdentifiers

/// An edit's reference: a chosen or dragged file, or an image from the history. We keep the source,
/// not an `ImageRGB` (75 MB of floats for a 3072 px photo): it is decoded off the main thread at
/// render time. Its size in pixels (EXIF orientation applied, as `ImageRGB(contentsOf:)` does) is
/// read from the header alone: image 1 sets the output's format (`DenoisingModule.editFormat`).
///
/// `@unchecked Sendable`: immutable; the thumbnail (`NSImage`) is read only by the view.
struct ReferenceImage: Identifiable, @unchecked Sendable {
    /// `subject`: a named subject's picture, cropped to its frame (`Subjects.swift`) — `nil` if its
    /// bytes did not decode, which fails the render naming it. `bytes`: a file's bytes sent by another
    /// Mac (`MirrorServer`), decoded as the file was there. `remote`: an image that stays on the other
    /// Mac, named only (a mirrored history image's edit images) — nothing to show or decode here.
    enum Source { case file(URL), image(CGImage), subject(CGImage?, name: String), bytes(Data), remote(RemoteReference.Origin) }
    /// Given only to a subject the prompt cites (`AppState.references`): the same id at every reading, for the rack's rows.
    var id = UUID()
    let source: Source
    let vignette: NSImage
    let name: String
    let width: Int, height: Int
    /// Taken from the history: its `i` ordinal — what a mirror sends instead of the bytes.
    var ordinal: Int? = nil

    /// The same picture as `other`: the same file, or the same image of the history.
    func same(_ other: ReferenceImage) -> Bool {
        if let a = ordinal, let b = other.ordinal { return a == b }
        if case .file(let a) = source, case .file(let b) = other.source { return a.standardizedFileURL == b.standardizedFileURL }
        return false
    }

    /// Decoded for display, upright, at `width × height` — the original under an edit's curtain.
    func displayed(width: Int, height: Int) -> CGImage? {
        let image: CGImage?
        switch source {
        case .image(let cg): image = cg
        case .subject(let cg, _): image = cg
        case .remote: image = nil
        case .file, .bytes:
            let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                            kCGImageSourceThumbnailMaxPixelSize: max(self.width, self.height),
                                            kCGImageSourceCreateThumbnailWithTransform: true]
            image = imageSource.flatMap { CGImageSourceCreateThumbnailAtIndex($0, 0, options as CFDictionary) }
        }
        guard let image, let c = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                           space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        c.interpolationQuality = .high
        c.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return c.makeImage()
    }

    func imageRGB() throws -> ImageRGB {
        switch source {
        case .file(let url): return try ImageRGB(contentsOf: url)
        case .bytes(let data): return try ImageRGB(data: data, name: name)
        case .remote: throw EngineError.imageUnreadable(file: name)
        case .image(let cg): return try ImageRGB(cgImage: cg)
        case .subject(let cg, let name):
            guard let cg else { throw EngineError.imageUnreadable(file: name) }
            return try ImageRGB(cgImage: cg)
        }
    }

    private var imageSource: CGImageSource? {
        switch source {
        case .file(let url): CGImageSourceCreateWithURL(url as CFURL, nil)
        case .bytes(let data): CGImageSourceCreateWithData(data as CFData, nil)
        case .image, .subject, .remote: nil
        }
    }

    /// **A file as the render reads it** — the file itself (`ImageRGB(contentsOf:)`, upright), its vignette and size; `nil` if
    /// ImageIO cannot read it. Its bytes come from the file's disk: 4.4 ms for a 12 MB PNG (2048²) on an M5 Pro's SSD, as long
    /// as the disk takes from a network share — the window reads it off the main thread (`AppState.read`).
    nonisolated static func file(_ url: URL) -> ReferenceImage? {
        guard let vignette = NSImage(contentsOf: url), vignette.isValid, let size = size(of: url) else { return nil }
        return ReferenceImage(source: .file(url), vignette: vignette, name: url.lastPathComponent,
                              width: size.width, height: size.height)
    }

    /// A file's size as it will be decoded, upright — or `nil` if ImageIO cannot read its header.
    static func size(of url: URL) -> (width: Int, height: Int)? {
        CGImageSourceCreateWithURL(url as CFURL, nil).flatMap(size)
    }

    static func size(of data: Data) -> (width: Int, height: Int)? {
        CGImageSourceCreateWithData(data as CFData, nil).flatMap(size)
    }

    private static func size(_ source: CGImageSource) -> (width: Int, height: Int)? {
        guard let p = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = p[kCGImagePropertyPixelWidth] as? Int, let h = p[kCGImagePropertyPixelHeight] as? Int,
              w > 0, h > 0 else { return nil }
        // Orientations 5 to 8 turn the image a quarter: the upright image has its sides swapped.
        let quarter = ((p[kCGImagePropertyOrientation] as? Int) ?? 1) >= 5
        return quarter ? (h, w) : (w, h)
    }
}

/// **An image as it leaves the app** by a drag or Share: its PNG bytes, with the name the receiver
/// gives the file. Only data: no temporary file is written on the way (incognito).
struct PNGImage: Transferable {
    let data: Data
    let name: String
    static var transferRepresentation: some TransferRepresentation {
        DataRepresentation(exportedContentType: .png) { $0.data }
            .suggestedFileName { $0.name }
    }
}

/// **What the Images box accepts**: a file (the Finder), or PNG bytes — a strip thumbnail, which leaves
/// as data only (`PNGImage`). The file first: a Finder PNG is read from its place, not copied.
enum DroppedImage: Transferable {
    case file(URL), png(Data)
    static var transferRepresentation: some TransferRepresentation {
        ProxyRepresentation(importing: { (url: URL) in DroppedImage.file(url) })
        DataRepresentation(importedContentType: .png) { DroppedImage.png($0) }
    }
}
