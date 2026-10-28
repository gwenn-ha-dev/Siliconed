// swift-tools-version:6.0
import Foundation
import PackageDescription

let package = Package(
    name: "Siliconed",
    // macOS 26: the only one the engine was ever measured and checked on (every map received ran 26.7 or 27);
    // MPSGraph's SDPA needs 15, Metal 4's matmul2d and the super-resolution 26.
    platforms: [.macOS("26.0")],
    products: [
        .executable(name: "SiliconedApp", targets: ["SiliconedApp"]),
        .executable(name: "silicontrol", targets: ["Silicontrol"]),
        .executable(name: "siliconed", targets: ["SiliconedCommand"]),
    ],
    targets: [
        // The current CBLAS headers of Accelerate (the old ones are deprecated since macOS 13.3).
        // Same routines, same 32-bit integers: ILP64 is not requested.
        .target(name: "Siliconed", dependencies: ["AMXLegacy"], cSettings: [.define("ACCELERATE_NEW_LAPACK")]),
        // `BNNSMatMul`, deprecated and kept: isolated in C, where the deprecation is acknowledged once.
        .target(name: "AMXLegacy", linkerSettings: [.linkedFramework("Accelerate")]),
        // **The app, `Siliconed.app` — the product.** The library is not a product of its own: the app
        // reaches its `package` symbols from this package (no `@testable`, so never its internals), and
        // `tools/translations.sh`, which compiles the app's files outside SwiftPM, passes the package's
        // name to do the same. The target is `SiliconedApp`, not
        // `Siliconed`: the library holds that name, and the file system does not tell it from
        // `siliconed`. `tools/app.sh` copies its binary to `Siliconed.app/Contents/MacOS/Siliconed`.
        // Its phrases (`Localizable.xcstrings`, English source, fr de es it translations) are not a resource of it:
        // `tools/app.sh` compiles them into `Siliconed.app/Contents/Resources`, where `Text("…")`
        // looks for them (the main bundle). Under `swift run`, the app speaks the source language.
        // `InfoPlist.xcstrings`: the Info.plist's one phrase (the local network's), compiled beside them.
        .executableTarget(name: "SiliconedApp", dependencies: ["Siliconed", "SilicontrolHelp"],
                          exclude: ["Localizable.xcstrings", "InfoPlist.xcstrings"]),
        // **`silicontrol`, the app's remote control**, shipped as
        // `Siliconed.app/Contents/Helpers/silicontrol`: a pipe to the running app's socket,
        // system frameworks only — it does not link the engine.
        .executableTarget(name: "Silicontrol", dependencies: ["SilicontrolHelp", "Measurement"]),
        // **What `silicontrol measure` reads and how a series goes**: the app's footprint (`proc_pid_rusage`),
        // swap-outs, pixels (ImageIO), power source (IOKit) — system calls only, no engine, so that anyone
        // with the app alone can repeat our timings and peaks (`docs/VERIFY.md`).
        .target(name: "Measurement", linkerSettings: [.linkedFramework("IOKit"), .linkedFramework("ImageIO")]),
        // **`siliconed`, the command line**: `siliconed check` runs every stage of every model against the
        // reference implementation's goldens (`docs/VERIFY.md`). The stages live in `Verification`, which the
        // maintainer's tool links too; the executable target is named apart because the file system does
        // not tell `Siliconed` from `siliconed`.
        .target(name: "Verification", dependencies: ["Siliconed", "AMXLegacy"],
                cSettings: [.define("ACCELERATE_NEW_LAPACK")]),
        .executableTarget(name: "SiliconedCommand", dependencies: ["Verification"]),
        // **The remote control's help**, Foundation only: one text for both sides — `silicontrol`
        // answers `help` without launching the app, the app answers it on its socket.
        .target(name: "SilicontrolHelp"),
        // **The fast target.** It touches neither a model's map nor a golden tensor: it exercises
        // pure functions and small synthetic files, in a few seconds and without a byte of
        // weights. A few tests do touch Metal — the GPU dequantization of synthetic 8-bit maps on
        // the Metal device (`GGUFTests`, `QuantizedMapTests`), an `MPSGraph` built to fail on
        // purpose (`TokenizerAndVAETests`) — on kilobytes, in milliseconds.
        .testTarget(name: "SiliconedTests", dependencies: ["Siliconed", "SilicontrolHelp", "Verification", "Measurement"],
                    exclude: ["Fixtures"]),
    ]
)

// **A maintainer's command line, when present.** The checks against the reference implementations
// (diffusers and transformers in fp32) live outside this repository; when their sources sit in
// `Sources/CLI`, they build with the package as `siliconed-dev`, reaching its `package` symbols.
// A clone without them builds the app and `silicontrol` alone (and the library they link).
if FileManager.default.fileExists(atPath: Context.packageDirectory + "/Sources/CLI") {
    package.products.append(.executable(name: "siliconed-dev", targets: ["CLI"]))
    package.targets.append(.executableTarget(name: "CLI", dependencies: ["Siliconed", "AMXLegacy", "Verification"],
                                             cSettings: [.define("ACCELERATE_NEW_LAPACK")]))
}
