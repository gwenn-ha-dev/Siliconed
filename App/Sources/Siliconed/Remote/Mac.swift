import Foundation

/// **A Mac, as another one is told of it**: its chip, its model, its memory, its system and the app's version — read
/// from `sysctl` and the bundle, never measured.
package struct Mac: Codable, Sendable, Equatable {
    /// `machdep.cpu.brand_string`: "Apple M1 Pro".
    package var chip: String
    /// `hw.model`: "MacBookPro18,3".
    package var hardwareModel: String
    package var memoryBytes: Int
    /// "27.0.1 (26A…)".
    package var macOS: String
    /// `CFBundleShortVersionString (CFBundleVersion)` of the app, else the git commit of the repository the binary was
    /// built in (`-dirty` if modified), else "unknown".
    package var version: String

    package init(chip: String, hardwareModel: String, memoryBytes: Int, macOS: String, version: String) {
        self.chip = chip; self.hardwareModel = hardwareModel; self.memoryBytes = memoryBytes
        self.macOS = macOS; self.version = version
    }

    /// This one.
    package static func current() -> Mac {
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let osBuild = sysctlString("kern.osversion")
        let macOS = "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)" + (osBuild == "unknown" ? "" : " (\(osBuild))")
        return Mac(chip: sysctlString("machdep.cpu.brand_string"), hardwareModel: sysctlString("hw.model"),
                   memoryBytes: Int(ProcessInfo.processInfo.physicalMemory), macOS: macOS, version: binaryVersion())
    }

    private static func sysctlString(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return "unknown" }
        return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// The app's version; else, for a binary built from a clone, its commit — read with `git` only when a
    /// `.git` lies above the executable (a Mac without the developer tools would otherwise be asked to
    /// install them by a dialog).
    private static func binaryVersion() -> String {
        var bundles = [Bundle.main]
        if let path = Bundle.main.executableURL?.resolvingSymlinksInPath().path, let range = path.range(of: ".app/Contents/"),
           let app = Bundle(path: String(path[..<range.lowerBound]) + ".app") {
            bundles.append(app)
        }
        for bundle in bundles {
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
