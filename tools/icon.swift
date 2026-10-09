// **Renders the icon's SVG source to a PNG, alpha kept**: `swift tools/icon.swift <in.svg> <out.png> [side]`.
// `tools/app.sh` calls it, then `iconutil`. `NSImage` reads SVG natively (macOS 14+), so nothing
// is installed; `qlmanage -t` would do it too, but flattens the transparent corners onto white.
import AppKit

let arguments = CommandLine.arguments
guard arguments.count >= 3, let image = NSImage(contentsOf: URL(fileURLWithPath: arguments[1])) else {
    FileHandle.standardError.write(Data("usage: swift tools/icon.swift <in.svg> <out.png> [side]\n".utf8))
    exit(1)
}
let side = arguments.count > 3 ? Int(arguments[3]) ?? 1024 : 1024
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: side, pixelsHigh: side, bitsPerSample: 8,
                           samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                           bytesPerRow: 0, bitsPerPixel: 0)!
rep.size = NSSize(width: side, height: side)
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
NSGraphicsContext.current?.imageInterpolation = .high
image.draw(in: NSRect(x: 0, y: 0, width: side, height: side))
NSGraphicsContext.restoreGraphicsState()
try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: arguments[2]))
