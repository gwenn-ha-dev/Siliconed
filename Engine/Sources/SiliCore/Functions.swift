import CoreAI
import CryptoKit
import Foundation
import Metal

/// **Writing, specializing and loading a function's asset.** An asset is written under `folder`, `name.aimodel`, only
/// when its bytes changed (a rewritten asset is a new source to Core AI's cache, specialized again).
public enum Functions {
    /// Where the assets are written: the client's to choose (the default is the user's caches).
    nonisolated(unsafe) public static var folder: URL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("SiliCore/functions", isDirectory: true)

    /// The asset `name`, `build` called only when it must be written or checked (not named by its bytes in this process:
    /// `named`).
    public static func asset(_ name: String, _ build: () throws -> Aicode.Module) throws -> URL {
        let url = folder.appendingPathComponent("\(name).aimodel")
        if lock.withLock({ hashed.contains(name) }) { return url }
        return try asset(try build(), name)
    }

    /// The asset of `m` as `name`, written only when its bytes changed.
    public static func asset(_ m: Aicode.Module, _ name: String) throws -> URL {
        let url = folder.appendingPathComponent("\(name).aimodel")
        let bytes = m.trimmed().bytecode()
        if (try? Data(contentsOf: url.appendingPathComponent("main.hash"))) == Data(SHA256.hash(data: bytes)) { return url }
        try Aicode.write(bytes, to: url, producer: "SiliCore")
        return url
    }

    /// **The asset of `m` named by its bytes**: `prefix-` and the first 16 bytes of the SHA-256 of its trimmed bytecode,
    /// written only when absent — the same module, the same asset, whatever its client calls it; another, another name
    /// (a share measured on this Mac, `ThisMac`, is that of these bytes).
    public static func named(_ m: Aicode.Module, prefix: String) throws -> String {
        let bytes = m.trimmed().bytecode(), digest = Data(SHA256.hash(data: bytes))
        let name = name(digest, prefix)
        let url = folder.appendingPathComponent("\(name).aimodel")
        if (try? Data(contentsOf: url.appendingPathComponent("main.hash"))) != digest { try Aicode.write(bytes, to: url, producer: "SiliCore") }
        lock.withLock { _ = hashed.insert(name) }
        return name
    }

    /// The name `named` gives `m`, nothing written: a plan names the assets of every candidate, the stage writes the
    /// one it loads (`asset`).
    public static func name(_ m: Aicode.Module, prefix: String) -> String { name(Data(SHA256.hash(data: m.trimmed().bytecode())), prefix) }
    private static func name(_ digest: Data, _ prefix: String) -> String { prefix + "-" + digest.prefix(16).map { String(format: "%02x", $0) }.joined() }

    /// **Every asset of `folder` erased**, and what this process wrote of them forgotten (`named`, `asset`: a name it
    /// wrote is not looked for again, and would be handed back gone). An asset is named by its bytes, so what no plan of
    /// a version names is known only by building every plan at every shape: a client erases them all — on an update, or
    /// when its user asks — and the next load writes again what it names, once. Run with nothing loaded from `folder`.
    /// What it erased, in bytes.
    @discardableResult
    public static func erase(_ folder: URL = folder) -> Int {
        let fm = FileManager.default
        var bytes = 0
        lock.withLock {
            for url in (try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [] where url.pathExtension == "aimodel" {
                let size = allocated(under: url)
                if (try? fm.removeItem(at: url)) != nil { bytes += size }
            }
            hashed.removeAll()
        }
        return bytes
    }

    /// **Where Core AI keeps what it specialized for the process `name`**: one folder per build of macOS,
    /// `coreai-cache/<build>/<name>` under `root` (the user's caches) — the layout Core AI's documentation gives
    /// (`AIModelCache`) for a process outside a sandbox under the default policy, whose specializations are purgeable.
    /// A process is filed by its name: what one specializes, another specializes again.
    public static func specializations(of name: String, under root: URL = caches) -> [URL] {
        let fm = FileManager.default, cache = root.appendingPathComponent("coreai-cache", isDirectory: true)
        return ((try? fm.contentsOfDirectory(at: cache, includingPropertiesForKeys: nil)) ?? [])
            .map { $0.appendingPathComponent(name, isDirectory: true) }
            .filter { fm.fileExists(atPath: $0.path) }
    }

    /// **What Core AI specialized for the processes `names` erased**, folder by folder — this process's own included, with
    /// nothing of it loaded: it specializes again what it loads next (`RunTests`). A process that no longer runs leaves
    /// its folder behind until macOS updates. Not `AIModelCache.deleteAll`, whose documentation speaks of the cache's
    /// entries: run on 20/11 from one process, it emptied every process's folder of the user's cache, other apps'
    /// included. What it erased, in bytes.
    @discardableResult
    public static func eraseSpecializations(of names: [String], under root: URL = caches) -> Int {
        var bytes = 0
        for url in names.flatMap({ specializations(of: $0, under: root) }) {
            let size = allocated(under: url)
            if (try? FileManager.default.removeItem(at: url)) != nil { bytes += size }
        }
        return bytes
    }

    /// What the assets of `folder` and the specializations of the processes `names` weigh on the disk, in bytes.
    public static func weight(_ folder: URL = folder, specializationsOf names: [String], under root: URL = caches) -> Int {
        let fm = FileManager.default
        let assets = ((try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []).filter { $0.pathExtension == "aimodel" }
        return (assets + names.flatMap { specializations(of: $0, under: root) }).reduce(0) { $0 + allocated(under: $1) }
    }

    /// The user's caches, where Core AI files its specializations.
    public static let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]

    /// What the files under `url` hold on the disk, in bytes: walked once, each file's two values read from what the
    /// walk prefetched (thousands of entries under a process's folder).
    private static func allocated(under url: URL) -> Int {
        let keys: Set<URLResourceKey> = [.totalFileAllocatedSizeKey, .isRegularFileKey]
        guard let walk = FileManager.default.enumerator(at: url, includingPropertiesForKeys: Array(keys)) else { return 0 }
        var bytes = 0
        for case let file as URL in walk {
            guard let values = try? file.resourceValues(forKeys: keys), values.isRegularFile == true else { continue }
            bytes += values.totalFileAllocatedSize ?? 0
        }
        return bytes
    }
    nonisolated(unsafe) private static var hashed: Set<String> = []
    private static let lock = NSLock()

    /// The function `main` of the asset at `url`, specialized for `unit` (`nil`, Core AI's default: every unit allowed,
    /// the combination it finds fastest). Core AI falls back on another unit for what the preferred one cannot run, and
    /// says nothing of where it placed what (measured 12/11): a placement shows in the values, not in the API.
    ///
    /// **A compilation is told** (`Stages.Event.compiling`, `compiled`): an asset Core AI's cache does not hold for these
    /// options (`AIModelCache.model(for:options:)`) is specialized here and its function loaded, seconds where a cached
    /// one loads in milliseconds. What its first call still pays is not counted: 47 ms against 0.5 for a trivial function
    /// on 4096 values, its later calls and a cached one's first alike (20/11, the mini).
    public static func load(_ url: URL, unit: ComputeUnitKind? = .gpu) async throws -> InferenceFunction {
        let options = unit.map { SpecializationOptions(preferredComputeUnitKind: $0) } ?? .default
        if let cached = try? AIModelCache.default.model(for: url, options: options) { return try main(cached, url) }
        let t = DispatchTime.now().uptimeNanoseconds
        Stages.tell(.compiling)
        defer { Stages.tell(.compiled(seconds: seconds(since: t))) }
        return try main(try await AIModel(contentsOf: url, options: options), url)
    }

    private static func main(_ model: AIModel, _ url: URL) throws -> InferenceFunction {
        guard let f = try model.loadFunction(named: "main") else { throw Failure("\(url.lastPathComponent): no function main") }
        return f
    }

    /// **The process's first call of Core AI, paid once, before any floor is read**: `y = x + x` on 64 × 64 f16 loaded and
    /// called on the GPU, then on the ANE, and let go. Called at a process's start (a client's worker, its command line, the
    /// tests on the chip), it lies in every stage's floor and in no term of P. Once a process: a second call waits for the first's.
    ///
    /// Measured 19/11 on the mini, a trivial GPU function on 128 × 256 f16, each in a fresh process: the first +61.5 MiB,
    /// the next +0.7 and +1.9; past the witness (+58.7), the first +1.5. **Of that first cost ≈ 22 MiB stay and ≈ 36 come
    /// and go**: idle 2.5 s, the footprint gives back 36.2 MiB, and the next call of any function takes them again (+37.7),
    /// the witness's functions held or not — Core AI's own, while it computes.
    ///
    /// **The witness waits for them to leave** (`settle`): the driver's memory for its command buffers, in the graphics
    /// ledger and in no allocation — 36 to 44 MiB, given back 1.0 to 1.5 s after the witness's last command buffer
    /// (21/11/2026 on the M5 Pro, the ledger 44.8 MiB, then 0.5 to 8.5, Metal's allocations 0.4). Read at once, the
    /// first stage's floor held them and no later floor did: of three runs of a block in one process, the first's floor
    /// and peak 31 to 46 MiB above the two others', for each of five clients' blocks. Paid once a process.
    public static func witness(_ device: MTLDevice) async throws {
        let task = lock.withLock { () -> Task<Void, Error> in
            if let witnessed { return witnessed }
            let t = Task {
                try await call(device)
                settle(device)
            }
            witnessed = t
            return t
        }
        try await task.value

        @Sendable func call(_ device: MTLDevice) async throws {
            let shape = [64, 64], bytes = 64 * 64 * 2
            var m = Aicode.Module(arguments: [("x", Aicode.tensor(shape, .f16))])
            m.outputs = [("y", m.body.binary(.add, m.argument("x"), m.argument("x")))]
            let asset = folder.appendingPathComponent("\(try named(m, prefix: "witness")).aimodel")
            guard let queue = device.makeCommandQueue() else { throw Failure("no command queue") }
            let stream = SerialStream(commandQueue: queue)
            // The GPU's operands on the heap, the ANE's surfaces of their own (`Arena.take(surface:)`).
            let arena = try Arena(device, bytes: 4 * Arena.aligned(bytes), apart: 2 * Arena.aligned(bytes))
            let (x, y) = (arena.take(bytes), arena.take(bytes))
            let (sx, sy) = (arena.take(surface: shape, element: 2), arena.take(surface: shape, element: 2))
            for k in [x, sx] { memset(arena.pointer(k), 0, bytes) }
            func view(_ k: Int) -> View { View(buffer: arena.buffer(k), type: .float16, shape: shape, surface: arena.surface(k)) }
            let gpu = try await load(asset, unit: .gpu)
            try await Run.call(gpu, stream, inputs: ["x": view(x)], outputs: [("y", view(y))])
            let ane = try await load(asset, unit: .neuralEngine)
            try await Run.neural(ane, inputs: ["x": view(sx)], outputs: [("y", view(sy))])
            withExtendedLifetime((gpu, ane, stream, arena)) {}
        }

        /// The driver's memory for the witness's command buffers, which no allocation holds, given back before any floor
        /// is read: the graphics ledger within 16 MiB of what Metal still has allocated, 2 s at most.
        @Sendable func settle(_ device: MTLDevice) {
            let t = DispatchTime.now().uptimeNanoseconds
            while Memory.graphics() > device.currentAllocatedSize + (16 << 20) && seconds(since: t) < 2 { usleep(2000) }
        }
    }
    nonisolated(unsafe) private static var witnessed: Task<Void, Error>?

    /// Seconds since `t` (`DispatchTime.now().uptimeNanoseconds`).
    public static func seconds(since t: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - t) / 1e9 }
}
