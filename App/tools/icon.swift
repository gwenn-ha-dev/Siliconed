// **Renders an SVG source to a PNG, alpha kept**: `swift tools/icon.swift <in.svg> <out.png> [side | WxH]`.
// `tools/app.sh` calls it for the icon, then `iconutil`; `tools/dmg.sh` for the disk image's background. `NSImage` reads SVG natively (macOS 14+), so nothing
// is installed; `qlmanage -t` would do it too, but flattens the transparent corners onto white.
import AppKit

let arguments = CommandLine.arguments
guard arguments.count >= 3, let image = NSImage(contentsOf: URL(fileURLWithPath: arguments[1])) else {
    FileHandle.standardError.write(Data("usage: swift tools/icon.swift <in.svg> <out.png> [side | WxH]\n".utf8))
    exit(1)
}
let size = (arguments.count > 3 ? arguments[3] : "1024").split(separator: "x").compactMap { Int($0) }
let (width, height) = (size.first ?? 1024, size.count > 1 ? size[1] : size.first ?? 1024)
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8,
                           samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                           bytesPerRow: 0, bitsPerPixel: 0)!
rep.size = NSSize(width: width, height: height)
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
NSGraphicsContext.current?.imageInterpolation = .high
image.draw(in: NSRect(x: 0, y: 0, width: width, height: height))
NSGraphicsContext.restoreGraphicsState()
try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: arguments[2]))
