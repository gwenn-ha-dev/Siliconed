import CoreAI
import Foundation
import Measurement
import Metal
import SiliCore

// **What every family shares**: a map's tensors by name and shape, whether the user lets a render take the Neural
// Engine, the bridge from `async` work to the render queue.
//
// **A publisher's int8 safetensors is read as it is published** (ComfyUI's convention, the same at Comfy-Org and
// Bedovyy): `<module>.weight` I8 `[out, in]`, its scales a row in `<module>.weight_scale` F32 `[out, 1]`, and what
// they mean in `<module>.comfy_quant`, a JSON text stored as bytes — `{"format": "int8_tensorwise", "convrot": true,
// "convrot_groupsize": 256}`. Any other format is refused: codes read as another quantization give a plausible,
// wrong weight.

/// **A map's tensors, by the names a family gives them**: a safetensors, or the shards behind an index. Names and shapes
/// only — what an import is recognized by (`ModelImport`, `LoRAFile`); the values are the engine's to read.
package final class MapWeights {
    package let path: String
    /// Each file of the map, and the file each key lies in.
    private let files: [Safetensors]
    private let place: [String: Int]

    /// What a DiT's tensors may lie under: nothing in most publishers' files, a checkpoint's prefix in ComfyUI's,
    /// `net.` in Cosmos-Predict2's own files (Anima's base and what is trained from it).
    package static let checkpointPrefixes = ["model.diffusion_model.", "diffusion_model.", "net."]

    /// **The prefix `file` keeps a DiT's tensors under**: "" when none, else the first of `checkpointPrefixes` under which
    /// `marker`, a key of that DiT alone, lies; nil when it lies under none.
    package static func checkpointPrefix(of file: SafeTensors, marker: String) -> String? {
        ([""] + checkpointPrefixes).first { file.tensors[$0 + marker] != nil }
    }

    /// **A DiT's file read under its prefix** (`checkpointPrefix`, `SafeTensors(_:under:)`): its tensors by the names the
    /// family gives them, whether a publisher's file or a whole checkpoint keeps them. `marker` under no prefix: not the
    /// family's DiT.
    package static func dit(_ path: String, marker: String) throws -> SafeTensors {
        guard let prefix = checkpointPrefix(of: try SafeTensors(path), marker: marker) else { throw EngineError.corruptMap(file: path) }
        return try SafeTensors(path, under: prefix)
    }
    /// The prefixes a key is also looked up under, or without (none: the file is read as it is).
    private let prefixes: [String]

    /// **A DiT's file**: each tensor found by the name the family gives it, whether the file keeps it under that name,
    /// under a checkpoint's prefix, or without the prefix the publisher's own file carries.
    package convenience init(dit path: String) throws { try self.init(path, under: Self.checkpointPrefixes) }

    package init(_ path: String, under prefixes: [String] = []) throws {
        self.path = path
        self.prefixes = prefixes
        let paths = Self.files(path).filter { !$0.hasSuffix(".index.json") }
        if path.hasSuffix(".index.json"), paths.isEmpty { throw ProgramError("\(path): an index that names no shard") }
        files = try paths.map { try Safetensors(path: $0) }
        var place: [String: Int] = [:]
        for (i, file) in files.enumerated() { for key in file.entries.keys { place[key] = i } }
        self.place = place
        for key in keys where key.hasSuffix(".comfy_quant") {
            guard let text = entry(key), text.entry.dtype == "U8", let pointer = files[text.file].pointer(text.entry.name),
                  let said = try? JSONSerialization.jsonObject(with: Data(bytes: pointer, count: text.entry.bytes)) as? [String: Any],
                  said["format"] as? String == "int8_tensorwise" else {
                throw ProgramError("\(path): \(key) is not an int8 quantization this app reads")
            }
            guard said["convrot"] as? Bool == true else { continue }
            guard let group = said["convrot_groupsize"] as? Int, group > 0 else {
                throw ProgramError("\(path): \(key) rotates without a group size")
            }
        }
    }

    /// **The files a map is made of**: itself, and behind a shards' index (`….index.json`) each shard it names, beside
    /// it. What weighs, moves or is checked as one map.
    package static func files(_ path: String) -> [String] {
        guard path.hasSuffix(".index.json"), let data = FileManager.default.contents(atPath: path),
              let map = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["weight_map"] as? [String: String] else {
            return [path]
        }
        let folder = (path as NSString).deletingLastPathComponent
        return [path] + Set(map.values).sorted().map { (folder as NSString).appendingPathComponent($0) }
    }

    package var keys: [String] { Array(place.keys) }
    /// The file's own key for `key`: itself, else without the prefix it carries, else under one.
    private func fileKey(_ key: String) -> String {
        if prefixes.isEmpty || place[key] != nil { return key }
        let bare = prefixes.first(where: key.hasPrefix).map { String(key.dropFirst($0.count)) } ?? key
        return ([bare] + prefixes.map { $0 + bare }).first { place[$0] != nil } ?? key
    }
    private func entry(_ key: String) -> (file: Int, entry: Safetensors.Entry)? {
        let k = fileKey(key)
        guard let i = place[k], let e = files[i].entries[k] else { return nil }
        return (i, e)
    }
    package func has(_ key: String) -> Bool { entry(key) != nil }
    package func shape(_ key: String) -> [Int]? { entry(key)?.entry.shape }

    /// **A tensor's values in f32**, from f32, bf16 or f16 (exact): a LoRA's factors, as its trainer stored them. `nil`
    /// for another element.
    package func floats(_ key: String) -> [Float]? {
        guard let (i, e) = entry(key), let p = files[i].pointer(e.name) else { return nil }
        switch e.dtype {
        case "F32": return (0..<e.count).map { p.loadUnaligned(fromByteOffset: 4 * $0, as: Float.self) }
        case "F16": return (0..<e.count).map { Float(Float16(bitPattern: p.loadUnaligned(fromByteOffset: 2 * $0, as: UInt16.self))) }
        case "BF16": return (0..<e.count).map { Float(bitPattern: UInt32(p.loadUnaligned(fromByteOffset: 2 * $0, as: UInt16.self)) << 16) }
        default: return nil
        }
    }

    /// A scalar's value, whatever float it is stored in (a LoRA's `alpha`: f32, bf16 or f16 by the trainer).
    package func value(_ key: String) -> Double? {
        guard let (i, e) = entry(key), e.count == 1, let p = files[i].pointer(e.name) else { return nil }
        switch e.dtype {
        case "F64": return p.loadUnaligned(as: Double.self)
        case "F32": return Double(p.loadUnaligned(as: Float.self))
        case "F16": return Double(Float16(bitPattern: p.loadUnaligned(as: UInt16.self)))
        case "BF16": return Double(Float(bitPattern: UInt32(p.loadUnaligned(as: UInt16.self)) << 16))
        default: return nil
        }
    }
}

/// A translation that cannot be written against what it was given (a weight missing from a map, an unknown layout).
package struct ProgramError: Error, CustomStringConvertible {
    package let description: String
    package init(_ description: String) { self.description = description }
}

/// **What a DiT computes in**, the same rule for every family: its map's weights and the Neural Engine switch. The
/// Standard's own bf16 functions on the GPU alone (the switch off); the shared forms — the functions the GPU and the ANE
/// both take, f16 under the guard (`SharedWeights`) — on the Compact's codes whatever the switch, on the Standard's
/// weights rounded to f16 with it on (the ANE takes no bf16 at its borders); beside the ANE or not (`Block.Share`).
package enum Arithmetic: Equatable, Sendable, CustomStringConvertible {
    case standard
    case shared(SharedWeights, Block.Share)

    /// A map's with the Neural Engine or not: its products int8 codes (the Compact) or not. With it, the ANE's rows a
    /// control says (`CoreAI.neuralEngineRows`, 0 the GPU alone), else found in flight.
    package static func of(codes: Bool, neuralEngine: Bool) -> Arithmetic {
        let share: Block.Share = !neuralEngine ? .gpu : CoreAI.neuralEngineRows.map { .told($0) } ?? .inFlight
        if codes { return .shared(.codes, share) }
        return neuralEngine ? .shared(.half, share) : .standard
    }

    /// What the engine offers the Compact's codes on `device` (`SharedWeights.offered`, a trial once a system), `codes`
    /// alone for a map of other weights or where a control decides (`CoreAI.integerProducts`).
    package static func offered(codes: Bool, _ device: MTLDevice) async throws -> [SharedWeights] {
        codes && CoreAI.integerProducts == nil ? try await SharedWeights.offered(device) : [.codes]
    }

    /// The Compact's weights a control says (`--products int8` or `q64`), nil where the engine chooses.
    private static var toldWeights: SharedWeights? { CoreAI.integerProducts.map { $0 ? .integer : .codes } }

    /// **A check's**, where nothing is timed (`check z-image block`, `block-lora`): `of`'s, the Compact's weights a
    /// control's word (`--products`), else the first the engine offers this Mac (`offered`) — the one a render tries
    /// first, the faster where the trial measured it.
    package static func told(codes: Bool, neuralEngine: Bool, _ device: MTLDevice) async throws -> Arithmetic {
        let a = of(codes: codes, neuralEngine: neuralEngine)
        guard case .shared(.codes, let share) = a else { return a }
        if let told = toldWeights { return .shared(told, share) }
        return .shared(try await offered(codes: true, device)[0], share)
    }

    /// **A render's**, `of`'s — but for the Standard beside the ANE in flight, the engine's choice between it and the
    /// Standard's bf16 on the GPU alone, each `declaration` timed on this Mac (`Block.choose`). Measured 20/11/2026 on
    /// the mini at 1024², whose GPU runs bf16 as fast as f16: the shared forms cost its GPU alone 7 to 16 % over bf16,
    /// and the ANE won that back on one DiT of five (−15 %), two even, two still slower (+7 to +9 %). Where bf16 runs
    /// slower than f16 (the M1 Pro: 2.16 TFLOP/s against 3.50), the shared forms should win everywhere.
    ///
    /// **The Compact's codes**: the weights the engine offered this Mac's GPU (`offered`, `SharedWeights.offered`:
    /// `codes`, and `integer` where its product int8 × int8 was measured the faster), each timed by `Block.choose`; a
    /// control's word (`CoreAI.integerProducts`) in place of the choice. Beside the ANE in flight, each on the GPU alone
    /// too: the ANE kept only where the block is faster with it, as the Settings say. Measured 21/11/2026 on the mini at
    /// 1024², int8 × int8: Krea 2 and Anima a step 3 % slower beside the ANE than alone, at its least share.
    package static func opened(codes: Bool, neuralEngine: Bool, offered: [SharedWeights] = [.codes],
                               _ declaration: (Arithmetic) -> Block.Declaration) throws -> Arithmetic {
        let a = of(codes: codes, neuralEngine: neuralEngine)
        let alternatives: [Arithmetic]
        if case .shared(.codes, let share) = a {
            let told = toldWeights
            // A control's rows: nothing timed (no pace is kept under them), the weights offered first.
            if case .told = share { return .shared(told ?? offered[0], share) }
            let shares: [Block.Share] = share == .inFlight ? [.inFlight, .gpu] : [share]
            alternatives = (told.map { [$0] } ?? offered).flatMap { w in shares.map { Arithmetic.shared(w, $0) } }
        } else if a == .shared(.half, .inFlight) {
            alternatives = [a, .standard]
        } else {
            return a
        }
        return alternatives.count == 1 ? alternatives[0] : alternatives[try Block.choose(alternatives.map { (declaration($0), $0.share) })]
    }

    /// **Whether a DiT's file holds the Compact's codes**: its product `marker` (a key of the file, `.weight` included)
    /// stored I8.
    package static func codes(_ file: SafeTensors, marker: String) -> Bool { file.tensors[marker]?.dtype == "I8" }

    /// **A render's DiT, opened**: its arithmetic `opened` under the Neural Engine switch, from what the engine offers
    /// this Mac's GPU (`offered`), each alternative's `declaration` timed where there is a choice; then the block of that
    /// declaration under `budget` (`Block.open`), on the share the arithmetic says. `marker`: the product whose element
    /// says the Compact (`codes`).
    package static func block(file: SafeTensors, codes marker: String, budget: Int,
                              _ declaration: (Arithmetic) -> Block.Declaration) async throws -> (block: Block, arithmetic: Arithmetic) {
        guard let device = MTLCreateSystemDefaultDevice() else { throw ProgramError("no Metal device") }
        let codes = Self.codes(file, marker: marker)
        let offered = try await Self.offered(codes: codes, device)
        let arithmetic = try opened(codes: codes, neuralEngine: CoreAI.neuralEngineInForce, offered: offered, declaration)
        let block = try await Block.open(device, declaration(arithmetic), files: [file], budget: budget, share: arithmetic.share)
        return (block, arithmetic)
    }

    /// The shared forms, or the Standard's bf16.
    package var shared: Bool { self != .standard }
    /// The shared forms' weights (the Standard's: none, `codes` said).
    package var weights: SharedWeights {
        if case .shared(let w, _) = self { w } else { .codes }
    }
    package var share: Block.Share {
        if case .shared(_, let s) = self { s } else { .gpu }
    }
    /// The shared forms' weights, nil for the Standard's: what a block's declaration lays its codes by
    /// (`Block.Declaration.sharedWeights`).
    package var sharedWeights: SharedWeights? {
        if case .shared(let w, _) = self { w } else { nil }
    }
    /// The residual stream's element, and the low-rank factors': f16 in the shared forms, bf16 in the Standard's.
    package var element: NDArray.ScalarType { shared ? .float16 : .bfloat16 }

    package var description: String {
        switch self {
        case .standard: "standard"
        case .shared(let w, let s):
            (w == .codes ? "compact" : w == .integer ? "compact, int8 × int8 on the GPU" : "standard in f16") + (s == .gpu ? "" : s == .inFlight ? ", the ANE in flight" : ", the ANE told")
        }
    }
}

/// **Core AI, as Siliconed hands it every function**: whether a piece of work may take the Neural Engine, and where Core
/// AI's assets are written.
package enum CoreAI {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var chosen: (neuralEngine: Bool, rows: Int?, integer: Bool?) = (false, nil, nil)
    /// The switch of the work in hand (`under`), which the process's does not reach until it ends.
    nonisolated(unsafe) private static var held: Bool?

    /// **Whether every render may take the Neural Engine**, the process's switch: the app sets the user's (Settings), a
    /// command its `--neural-engine`. Off until someone says otherwise: a control is judged on the GPU alone, the model's
    /// own arithmetic.
    package static var neuralEngine: Bool {
        get { lock.withLock { chosen.neuralEngine } }
        set { lock.withLock { chosen.neuralEngine = newValue } }
    }

    /// **A control's word**: the rows of every function the Neural Engine is told to take, in place of the share the
    /// engine finds by itself (`--neural-engine`); `nil` leaves it to the engine.
    package static var neuralEngineRows: Int? {
        get { lock.withLock { chosen.rows } }
        set { lock.withLock { chosen.rows = newValue } }
    }

    /// **A control's word**: the Compact's products int8 × int8 on the GPU (`SharedWeights.integer`) or not, in place of
    /// the engine's choice between them (`--products int8` or `q64`); `nil` leaves it to the engine.
    package static var integerProducts: Bool? {
        get { lock.withLock { chosen.integer } }
        set { lock.withLock { chosen.integer = newValue } }
    }

    /// **One switch from end to end of a piece of work**: a render computes its text, its denoising and its decoder at
    /// different times, and a preference changed in between would make one image of two arithmetics. The engine queue
    /// runs one such work at a time, under the process's switch as it is when the work begins (or the work's around it).
    package static func under<T>(_ work: () throws -> T) rethrows -> T {
        let before = lock.withLock { () -> Bool? in
            let before = held
            held = held ?? chosen.neuralEngine
            return before
        }
        defer { lock.withLock { held = before } }
        return try work()
    }

    /// Whether the work computed now may take the Neural Engine: the work's in hand, else the process's switch.
    package static var neuralEngineInForce: Bool { lock.withLock { held ?? chosen.neuralEngine } }

    /// **Where the functions' assets are written** (`Functions.folder`): the user's caches, where everything is written
    /// again when missing — no weight in an asset, its bytes its name. What this Mac measured stays in the engine folder
    /// of `library`'s app space (`<local>/engine`, `ThisMac`, `this-mac.json`), which macOS never purges and which stays on
    /// this Mac when the library moves: in the caches, a purge sent every block back to its unit's trial. Set before a
    /// piece of work computes.
    package static func place(_ library: Library, functions: URL = functions) {
        Functions.folder = functions
        ThisMac.file = library.engineFolder.appendingPathComponent("this-mac.json")
    }

    /// **This app, as the compilations' mark names it** (`markCompiled`): its version and its build, inside the app's own
    /// bundle, and its engine's binary (`silicore-worker`'s UUID); `siliconed` for the command line. The binary because
    /// a worker does not read the specializations another binary of its name wrote — `nilError` at its first call,
    /// `_GenericObjCError` at its first stage, every render (20/11, the M1 Pro: two builds of one number, nothing erased,
    /// a family that rendered no longer did); a build number does not change between two builds of one commit.
    package static let thisApp: String = {
        guard Bundle.main.bundleIdentifier == siliconedBundleIdentifier, let info = Bundle.main.infoDictionary else { return "siliconed" }
        let version = "\(info["CFBundleShortVersionString"] as? String ?? "?") (\(info["CFBundleVersion"] as? String ?? "?"))"
        return EngineWorker.executable.flatMap(binaryUUID).map { "\(version) \($0)" } ?? version
    }()

    /// **The UUID a Mach-O binary carries** (`LC_UUID`), its first slice's in a universal one; nil when it has none.
    static func binaryUUID(_ url: URL) -> String? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        func read(_ offset: UInt64, _ count: Int) -> Data? {
            guard (try? h.seek(toOffset: offset)) != nil, let d = try? h.read(upToCount: count), d.count == count else { return nil }
            return d
        }
        func u32(_ d: Data, _ at: Int, big: Bool = false) -> UInt32 {
            let v = d.subdata(in: at..<at + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
            return big ? UInt32(bigEndian: v) : UInt32(littleEndian: v)
        }
        guard let magic = read(0, 8) else { return nil }
        var base: UInt64 = 0
        if u32(magic, 0, big: true) == 0xCAFE_BABE {  // universal: the first slice
            guard let arch = read(8, 20) else { return nil }
            base = UInt64(u32(arch, 8, big: true))
        }
        guard let header = read(base, 32), u32(header, 0) == 0xFEED_FACF else { return nil }
        let count = Int(u32(header, 16)), size = Int(u32(header, 20))
        guard let commands = read(base + 32, size) else { return nil }
        var at = 0
        for _ in 0..<count {
            guard at + 8 <= commands.count else { return nil }
            let (command, length) = (u32(commands, at), Int(u32(commands, at + 4)))
            if command == 0x1B, at + 24 <= commands.count {  // LC_UUID
                return commands.subdata(in: at + 8..<at + 24).map { String(format: "%02X", $0) }.joined()
            }
            guard length > 0 else { return nil }
            at += length
        }
        return nil
    }

    /// **The process's first call of Core AI, paid at its start** (`Functions.witness`): before any stage reads its
    /// floor, so that every floor holds it and no P counts it. Waited for on the calling thread (`blocking`); a failure
    /// said on the standard error, the work after it left to fail where it would.
    package static func witness() {
        do {
            guard let device = MTLCreateSystemDefaultDevice() else { throw ProgramError("no Metal device") }
            try blocking { try await Functions.witness(device) }
        } catch {
            FileHandle.standardError.write(Data("siliconed: Core AI's first call failed: \(error)\n".utf8))
        }
    }

    package static var functions: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("Siliconed/functions", isDirectory: true)
    }

    /// **The processes that specialize this app's functions**, as Core AI files them (`Functions.specializations`): the
    /// engine's (`silicore-worker`) and the command line's (`siliconed`, the same assets).
    package static let worker = "silicore-worker"
    package static let specializing = [worker, "siliconed"]

    /// **What this Mac's compilations weigh**: the functions' assets and what Core AI specialized of them for each
    /// process of the app, in bytes.
    package static func compiled(_ folder: URL = functions, under root: URL = Functions.caches) -> Int {
        Functions.weight(folder, specializationsOf: specializing, under: root)
    }

    /// **The compilations erased, in the engine's process, between two pieces of work** (`WorkerCall.erase`), nothing
    /// loaded: every asset (`Functions.erase`) and what Core AI specialized of them for each of the app's processes, by
    /// their folders — the app's alone, never another's. Every model's next render writes and specializes again what it
    /// needs. What it freed, in bytes.
    package static func erase(_ folder: URL = functions, under root: URL = Functions.caches) -> Int {
        Functions.erase(folder) + Functions.eraseSpecializations(of: specializing, under: root)
    }

    /// **Whether the compilations are another installation's**: `installed` (the app's version and build) is not the one
    /// they were last erased for (`markCompiled`). An asset is named by its bytes, so what no plan of this version names
    /// is known only by building every plan at every shape: the version's change stands for it.
    package static func due(installed: String, folder: URL = functions) -> Bool {
        (try? String(contentsOf: mark(folder), encoding: .utf8)) != installed
    }

    /// The compilations are now `installed`'s: written once the engine erased the others'.
    package static func markCompiled(for installed: String, folder: URL = functions) {
        try? FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? installed.write(to: mark(folder), atomically: true, encoding: .utf8)
    }

    /// Beside the functions' folder, not in it: erased with them, it would say nothing.
    private static func mark(_ folder: URL) -> URL { folder.deletingLastPathComponent().appendingPathComponent("compiled-for") }
}

/// **A tensor handed from one piece of a render to the next on the CPU** (a text's conditioning, a latent): its shape
/// and its values in fp32, which hold what an encoder writes (bf16, f16, f32) bit for bit.
package struct HostTensor: Sendable, Equatable {
    package var shape: [Int]
    package var floats: [Float]
    package init(_ shape: [Int], floats: [Float]) { (self.shape, self.floats) = (shape, floats) }
}

/// **Waits on the calling thread for `body`, run as a task**: the one bridge from `async` work (a URLSession request,
/// the engine's preparation) to a synchronous caller — the render queue, an installation, the command line.
/// The calling thread blocks on a semaphore: **never call it from Swift's cooperative pool** (a `Task`), whose few
/// threads the task it waits for may need; an `async` caller goes through `offThePool`.
package func blocking<T>(_ body: @escaping () async throws -> T) throws -> T {
    let box = BlockingOutcome<T>()
    let done = DispatchSemaphore(value: 0)
    let work = UncheckedSendable(body)
    Task.detached {
        do { box.result = .success(try await work.value()) } catch { box.result = .failure(error) }
        done.signal()
    }
    done.wait()
    return try box.result!.get()
}

private final class BlockingOutcome<T>: @unchecked Sendable { var result: Result<T, Error>? }

/// **Synchronous work that blocks (`blocking`), for an `async` caller**: run on a thread of Dispatch's, outside the
/// cooperative pool, the caller suspended meanwhile.
package func offThePool<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global(qos: .userInitiated).async { continuation.resume(with: Result { try work() }) }
    }
}

/// A value carried across an isolation boundary its type does not declare it may cross (a closure over the caller's
/// state): the caller waits for the other side, so the value is never used from two places at once.
package struct UncheckedSendable<T>: @unchecked Sendable {
    package let value: T
    package init(_ value: T) { self.value = value }
}
