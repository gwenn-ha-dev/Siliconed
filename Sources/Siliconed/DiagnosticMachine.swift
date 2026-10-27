// What the diagnostic reads rather than measures: the machine, its fingerprint, the CPU
// flags and the conditions (thermal, power, swap). Split from `Diagnostic.swift`.

import CryptoKit
import Foundation
import IOKit
import IOKit.ps
import Metal

extension Diagnostic {
    // ── the machine ─────────────────────────────────────────────────────────────────────────

    package static func machine() -> Machine {
        let device = MTLCreateSystemDefaultDevice()
        return Machine(chip: sysctlString("machdep.cpu.brand_string"), hardwareModel: sysctlString("hw.model"),
                       performanceCores: sysctlInteger("hw.perflevel0.physicalcpu") ?? 0,
                       efficiencyCores: sysctlInteger("hw.perflevel1.physicalcpu") ?? 0,
                       gpuCores: gpuCoreCount(), memoryBytes: sysctlInteger("hw.memsize") ?? 0,
                       macOS: macOSLabel(), version: binaryVersion(), build: buildStamp(),
                       gpu: device?.name ?? "none", gpuFamily: device.map(appleFamily) ?? "",
                       gpuWorkingSetBytes: Int(device?.recommendedMaxWorkingSetSize ?? 0))
    }

    /// "26.0.1 (25A…)": the version and, when the kernel says it, the build.
    private static func macOSLabel() -> String {
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let osBuild = sysctlString("kern.osversion")
        return "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)" + (osBuild == "unknown" ? "" : " (\(osBuild))")
    }

    /// The first 12 hex digits of the SHA-256 of `RenderCache.build` (the executable's path, size, date).
    /// **The engine that measured**: in an app, its Info.plist's `SiliconedEngine` — the digest `tools/app.sh`
    /// takes of the engine's sources (`Sources/Siliconed`, `Sources/AMXLegacy`, `Package.swift`) and of the
    /// compiler, so that an update moving only the app, its phrases or its docs keeps the map; elsewhere
    /// (the command line, a test), the executable's own stamp: every build measures again.
    private static func buildStamp() -> String {
        for bundle in enclosingBundles() {
            if let engine = bundle.infoDictionary?["SiliconedEngine"] as? String { return "engine " + engine }
        }
        return SHA256.hash(data: Data(RenderCache.build.utf8)).prefix(6).map { String(format: "%02x", $0) }.joined()
    }

    /// This executable's bundle, then the `.app` it sits in (a helper's `Contents/Helpers`).
    private static func enclosingBundles() -> [Bundle] {
        var bundles = [Bundle.main]
        if let path = Bundle.main.executableURL?.resolvingSymlinksInPath().path, let range = path.range(of: ".app/Contents/") {
            if let app = Bundle(path: String(path[..<range.lowerBound]) + ".app") { bundles.append(app) }
        }
        return bundles
    }

    /// **What a map is valid for**: the hardware model, the chip, macOS with its build, and the engine
    /// that measured it (`build`). Any of them changing — a new Mac restored from a backup, a macOS
    /// update that moves MPS, an update that moves a kernel — makes the map stale: the stale one is not
    /// applied, and the app measures again (~1 min). **The app's version is said, not compared**: the
    /// map measures the engine, and an update of the app alone does not move it.
    package struct Fingerprint: Codable, Sendable, Equatable, CustomStringConvertible {
        package var hardwareModel, chip, macOS, version, build: String

        package init(hardwareModel: String, chip: String, macOS: String, version: String, build: String) {
            self.hardwareModel = hardwareModel; self.chip = chip; self.macOS = macOS; self.version = version
            self.build = build
        }

        package init(_ machine: Machine) {
            self.init(hardwareModel: machine.hardwareModel, chip: machine.chip, macOS: machine.macOS,
                      version: machine.version, build: machine.build)
        }

        package static func == (a: Fingerprint, b: Fingerprint) -> Bool {
            (a.hardwareModel, a.chip, a.macOS, a.build) == (b.hardwareModel, b.chip, b.macOS, b.build)
        }

        /// This process's, read without Metal (`sysctl`, the bundle, the executable's stamp).
        package static var current: Fingerprint {
            Fingerprint(hardwareModel: sysctlString("hw.model"), chip: sysctlString("machdep.cpu.brand_string"),
                        macOS: macOSLabel(), version: binaryVersion(), build: buildStamp())
        }

        package var description: String { "\(chip) (\(hardwareModel)) · macOS \(macOS) · \(version) · \(build)" }
    }

    /// **The CPU flags a product can use**, read: half-precision arithmetic (`FEAT_FP16`, `FEAT_FHM`),
    /// bf16 (`FEAT_BF16`, `FEAT_EBF16`), the int8 matrix and dot products (`FEAT_I8MM`, `FEAT_DotProd`),
    /// and SME — the M4's matrix unit made architectural — with its outer products by type and its
    /// vector length. Not measured: BNNS decides what it runs on, and its products are (`LowPrecision`).
    package static let featureNames = [
        "FEAT_FP16", "FEAT_FHM", "FEAT_BF16", "FEAT_EBF16", "FEAT_I8MM", "FEAT_DotProd",
        "FEAT_SME", "FEAT_SME2", "SME_F32F32", "SME_F16F32", "SME_B16F32", "SME_I8I32",
        "FEAT_SME_F16F16", "FEAT_SME_B16B16",
    ]

    package static func cpuFeatures() -> [String: Int] {
        var flags: [String: Int] = [:]
        for name in featureNames { flags[name] = sysctlInteger("hw.optional.arm.\(name)") }
        flags["sme_max_svl_b"] = sysctlInteger("hw.optional.arm.sme_max_svl_b")
        return flags
    }

    /// The highest `appleN` family: `MTLGPUFamily.apple1` is 1001, and the next ones follow.
    private static func appleFamily(_ device: MTLDevice) -> String {
        for n in stride(from: 16, through: 1, by: -1) {
            if let family = MTLGPUFamily(rawValue: 1000 + n), device.supportsFamily(family) { return "apple\(n)" }
        }
        return ""
    }

    // ── the conditions ──────────────────────────────────────────────────────────────────────

    package static func conditions() -> Conditions {
        var usage = xsw_usage(), size = MemoryLayout<xsw_usage>.size
        let swap = sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 ? Int(usage.xsu_used) : 0
        return Conditions(thermal: thermalLabel(ProcessInfo.processInfo.thermalState),
                          lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled, onBattery: onBattery(),
                          swapUsedBytes: swap)
    }

    package static func thermalLabel(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// The source the machine draws from: `nil` when IOKit does not say (a desktop says "AC Power").
    private static func onBattery() -> Bool? {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() else { return nil }
        return (type as String) == kIOPSBatteryPowerValue
    }

    private static func sysctlString(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return "unknown" }
        return String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// `nil` when the kernel does not know the name.
    private static func sysctlInteger(_ name: String) -> Int? {
        var value: Int64 = 0
        var size = MemoryLayout<Int64>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
        return size == 4 ? Int(Int32(truncatingIfNeeded: value)) : Int(value)
    }

    /// The `AGXAccelerator`'s `gpu-core-count` in the IORegistry — Metal does not say it.
    private static func gpuCoreCount() -> Int {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iterator) == KERN_SUCCESS else {
            return 0
        }
        defer { IOObjectRelease(iterator) }
        var service = IOIteratorNext(iterator)
        while service != 0 {
            defer { IOObjectRelease(service); service = IOIteratorNext(iterator) }
            if let value = IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? Int, value > 0 {
                return value
            }
        }
        return 0
    }

    /// The app's version; else, for a binary built from a clone, its commit — read with `git` only when a
    /// `.git` lies above the executable (a Mac without the developer tools would otherwise be asked to
    /// install them by a dialog).
    private static func binaryVersion() -> String {
        for bundle in enclosingBundles() {
            if let short = bundle.infoDictionary?["CFBundleShortVersionString"] as? String {
                let build = bundle.infoDictionary?["CFBundleVersion"] as? String
                return short + (build.map { " (\($0))" } ?? "")
            }
        }
        guard var folder = Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent() else { return "unknown" }
        while folder.path != "/" {
            if FileManager.default.fileExists(atPath: folder.appendingPathComponent(".git").path) {
                guard let commit = git(["rev-parse", "--short=10", "HEAD"], in: folder), !commit.isEmpty else { return "unknown" }
                let dirty = !(git(["status", "--porcelain", "--untracked-files=no"], in: folder) ?? "").isEmpty
                return "git " + commit + (dirty ? "-dirty" : "")
            }
            folder = folder.deletingLastPathComponent()
        }
        return "unknown"
    }

    private static func git(_ arguments: [String], in folder: URL) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", folder.path] + arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
