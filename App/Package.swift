// swift-tools-version: 6.0
import PackageDescription

// The app: the engine's client. The model families, the render, the checks against the reference implementations.
// macOS 27: the engine runs its programs through Core AI.
let package = Package(
    name: "Siliconed",
    platforms: [.macOS("27.0")],
    products: [
        .executable(name: "SiliconedApp", targets: ["SiliconedApp"]),
        .executable(name: "silicontrol", targets: ["Silicontrol"]),
        .executable(name: "siliconed", targets: ["SiliconedCommand"]),
        // The app's engine, in a process of its own (`WorkerServer`): the app's renders and writer run there, under B,
        // and the app kills it past B. Its name is stable: Core AI files its compilations by process name.
        .executable(name: "silicore-worker", targets: ["SiliconedWorker"]),
    ],
    dependencies: [.package(name: "SiliCore", path: "../Engine")],
    targets: [
        // `Measurement`: the engine's process's footprint, read from the app (`EngineWorker`).
        .target(name: "Siliconed", dependencies: ["SiliCore", "Measurement"]),
        // The app, `Siliconed.app`. The target is `SiliconedApp`: the library holds `Siliconed`, and the file system
        // does not tell it from `siliconed`. Its phrases are not a resource of the target: `tools/app.sh` compiles
        // them into the bundle, where `Text("…")` looks for them; under `swift run` the app speaks English.
        .executableTarget(name: "SiliconedApp", dependencies: ["Siliconed", "SilicontrolHelp"],
                          exclude: ["Localizable.xcstrings", "InfoPlist.xcstrings"]),
        // `silicontrol`, the running app's remote control: a pipe to its socket, system frameworks only.
        .executableTarget(name: "Silicontrol", dependencies: ["SilicontrolHelp", "Measurement"]),
        // What `silicontrol measure` reads: the app's footprint, swap-outs, pixels, power source. System calls only.
        .target(name: "Measurement", linkerSettings: [.linkedFramework("IOKit"), .linkedFramework("ImageIO")]),
        // The remote control's help, Foundation only: one text for `silicontrol help` and for the app's socket.
        .target(name: "SilicontrolHelp"),
        // `siliconed check`: every stage of every model against the reference implementation's goldens. The executable
        // target is named apart because the file system does not tell `Siliconed` from `siliconed`.
        .target(name: "Verification", dependencies: ["Siliconed", "SiliCore"]),
        .executableTarget(name: "SiliconedCommand", dependencies: ["Verification"]),
        .executableTarget(name: "SiliconedWorker", dependencies: ["Siliconed"]),
        // The fast target: pure functions and small synthetic files, no map and no golden tensor.
        .testTarget(name: "SiliconedTests", dependencies: ["Siliconed", "SilicontrolHelp", "Verification", "Measurement", "SiliCore"]),
    ]
)
