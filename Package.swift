// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "Siliconed",
    platforms: [.macOS(.v15)],   // MPSGraph.scaledDotProductAttention : macOS 15+
    products: [
        .library(name: "Siliconed", targets: ["Siliconed"]),
        .executable(name: "SiliconedApp", targets: ["SiliconedApp"]),
        .executable(name: "silicontrol", targets: ["Silicontrol"]),
    ],
    targets: [
        // The current CBLAS headers of Accelerate (the old ones are deprecated since macOS 13.3).
        // Same routines, same 32-bit integers: ILP64 is not requested.
        .target(name: "Siliconed", dependencies: ["AMXLegacy"], resources: [.copy("Resources")], cSettings: [.define("ACCELERATE_NEW_LAPACK")]),
        // `BNNSMatMul`, deprecated and kept: isolated in C, where the deprecation is acknowledged once.
        .target(name: "AMXLegacy", linkerSettings: [.linkedFramework("Accelerate")]),
        // **The app, `Siliconed.app` — the product.** It is also the SwiftUI example of
        // `docs/API.md`, compiled: a client of the library as any app would be (no `@testable`). If
        // it no longer compiles, the API changed without its doc. The target is `SiliconedApp`, not
        // `Siliconed`: the library holds that name, and the file system does not tell it from
        // `siliconed`. `tools/app.sh` copies its binary to `Siliconed.app/Contents/MacOS/Siliconed`.
        // Its phrases (`Localizable.xcstrings`, English source, fr de es it translations) are not a resource of it:
        // `tools/app.sh` compiles them into `Siliconed.app/Contents/Resources`, where `Text("…")`
        // looks for them (the main bundle). Under `swift run`, the app speaks the source language.
        .executableTarget(name: "SiliconedApp", dependencies: ["Siliconed", "SilicontrolHelp"],
                          exclude: ["Localizable.xcstrings"]),
        // **`silicontrol`, the app's remote control**, shipped as
        // `Siliconed.app/Contents/Helpers/silicontrol`: a pipe to the running app's socket,
        // Foundation only — it does not link the engine.
        .executableTarget(name: "Silicontrol", dependencies: ["SilicontrolHelp"]),
        // **The remote control's help**, Foundation only: one text for both sides — `silicontrol`
        // answers `help` without launching the app, the app answers it on its socket.
        .target(name: "SilicontrolHelp"),
        // **The fast target.** It touches neither a model's map nor a golden tensor: it exercises
        // pure functions and small synthetic files, in a few seconds and without a byte of
        // weights. A few tests do touch Metal — the GPU dequantization of synthetic 8-bit maps on
        // the Metal device (`GGUFTests`, `QuantizedMapTests`), an `MPSGraph` built to fail on
        // purpose (`TokenizerAndVAETests`) — on kilobytes, in milliseconds.
        .testTarget(name: "SiliconedTests", dependencies: ["Siliconed", "SilicontrolHelp"],
                    exclude: ["Fixtures"]),
    ]
)
