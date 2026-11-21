// swift-tools-version: 6.2
import PackageDescription

// The engine alone: Core AI functions of fixed shapes written op by op, their weights given as views of an arena we
// hold, the memory and the units its own. It knows no word of any workload (`engineKnowsNoWorkload` fails `swift test`
// otherwise): the app beside it (`../App`) writes its functions with these tools and holds its loop.
let package = Package(
    name: "SiliCore",
    platforms: [.macOS("27.0")],
    products: [.library(name: "SiliCore", targets: ["SiliCore"])],
    targets: [
        .target(name: "SiliCore"),
        // The fast target: no GPU, no weights.
        .testTarget(name: "SiliCoreTests", dependencies: ["SiliCore"]),
        // What runs on this Mac's GPU and ANE through Core AI: off unless `SILICORE_METAL` is set, in release.
        .testTarget(name: "RunTests", dependencies: ["SiliCore"]),
    ]
)
