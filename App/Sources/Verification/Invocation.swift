import Foundation
import Siliconed

// **One check's arguments, the same grammar for every stage**:
//
//     siliconed check <model> <stage> [--size 512|1024|WxH] [--edit N] [--goldens DIR] [--map PATH] [--long]
//                                     [--neural-engine off|on|ROWS]
//
// `--neural-engine` is the user's switch (`CoreAI.neuralEngine`): a control is judged on the GPU alone, the model's own
// arithmetic, unless it says otherwise — `on`, the engine gives the Neural Engine rows of the programs by itself; a
// count, as many rows as it says (0: the shared forms on the GPU alone). The stage prints what it gave.
//
// Named options only, no positional magic: `--size` is the only thing that names a size, so a number
// meant for something else is never taken for one. An option a stage does not accept is a usage
// error, not silently ignored.

/// The parsed options of one `check <model> <stage>` run.
package struct Invocation: Sendable {
    /// The size the golden tensors were written at, as their file names spell it: `512`, `768x512`.
    package let size: String
    package let width: Int
    package let height: Int
    /// `--edit N`: the number of reference images (Qwen-Image-2.1's editing).
    package let edit: Int?
    /// `--map PATH`: another DiT file than the installed Standard (a Compact).
    package let map: String?
    /// `--long`: the text stages on the oracle's long prompt (past 512 tokens), golden `…-text-long`.
    package let long: Bool
    package let goldens: Goldens
    /// A stage's own options (`--sequence 256`), by name, the last value given.
    package let extras: [String: String]

    /// The options every stage accepts.
    package static let common: Set<String> = ["--size", "--goldens", "--neural-engine", "--products"]

    /// Parses `arguments` (what follows `check <model> <stage>`), refusing any option outside `accepted`
    /// and `common`. `defaultSize`: the stage's own, when `--size` is absent.
    package init(_ arguments: [String], accepted: Set<String>, defaultSize: String) throws {
        var values: [String: [String]] = [:]
        var flags: Set<String> = []
        var i = 0
        while i < arguments.count {
            let option = arguments[i]
            guard option.hasPrefix("--") else { throw UsageError("unexpected argument '\(option)': options are named (--size 1024)") }
            guard accepted.contains(option) || Self.common.contains(option) else {
                let all = (accepted.union(Self.common)).sorted().joined(separator: " ")
                throw UsageError("this stage does not take \(option) (it takes: \(all))")
            }
            if Self.flags.contains(option) { flags.insert(option); i += 1; continue }
            guard i + 1 < arguments.count, !arguments[i + 1].hasPrefix("--") else { throw UsageError("\(option) expects a value") }
            values[option, default: []].append(arguments[i + 1])
            i += 2
        }
        let size = values["--size"]?.last ?? defaultSize
        guard let (w, h) = Format.parse(size), w >= Format.minimumSide, h >= Format.minimumSide else {
            throw UsageError("--size \(size): 512, 1024 or WxH (768x512)")
        }
        self.size = Format.fileLabel(width: w, height: h)
        (width, height) = (w, h)
        edit = try values["--edit"]?.last.map {
            guard let n = Int($0), n >= 1 else { throw UsageError("--edit \($0): a count of reference images, 1 or more") }
            return n
        }
        map = values["--map"]?.last
        // The process's switch, and a control's own share of the Neural Engine: every plan the stage makes reads them.
        if let word = values["--neural-engine"]?.last {
            switch word {
            case "off": (CoreAI.neuralEngine, CoreAI.neuralEngineRows) = (false, nil)
            case "on": (CoreAI.neuralEngine, CoreAI.neuralEngineRows) = (true, nil)
            default:
                guard let n = Int(word), n >= 0 else { throw UsageError("--neural-engine \(word): off, on or a count of rows") }
                (CoreAI.neuralEngine, CoreAI.neuralEngineRows) = (true, n)
            }
        }
        if let products = values["--products"]?.last {
            guard products == "int8" || products == "q64" else { throw UsageError("--products \(products): int8 or q64") }
            CoreAI.integerProducts = products == "int8"
        }
        long = flags.contains("--long")
        goldens = Goldens(directory: values["--goldens"]?.last)
        extras = values.compactMapValues(\.last).filter { !Self.known.contains($0.key) }
    }

    /// The options that take no value.
    private static let flags: Set<String> = ["--long"]

    /// The options parsed above; any other accepted option lands in `extras`.
    private static let known: Set<String> = ["--size", "--goldens", "--edit", "--map", "--neural-engine", "--products"]

    /// The DiT map: `--map`, else the installed one named `standard`; missing or of another variant than the golden's, the stage is skipped.
    /// `golden`: the variant the stage's golden was made for; another one is skipped (`MapCheck`).
    package func ditMap(_ standard: String, golden: MapVariant = .standard) throws -> String {
        let path = map ?? Library.processWide.map(standard)
        guard FileManager.default.fileExists(atPath: path) else { throw Skip("map missing: \(path) — siliconed install") }
        return try MapCheck.use(path, golden: golden)
    }

    /// `--variant compact`: the Compact's map and goldens; absent, the Standard's. Any other variant is a usage error.
    package func compact() throws -> Bool {
        switch extras["--variant"] {
        case nil, "standard": return false
        case "compact": return true
        case let other?: throw UsageError("--variant \(other): compact (or standard, the default)")
        }
    }

    /// A stage's own integer option, `fallback` when absent.
    package func integer(_ option: String, _ fallback: Int) throws -> Int {
        guard let text = extras[option] else { return fallback }
        guard let value = Int(text) else { throw UsageError("\(option) \(text): an integer") }
        return value
    }
}

/// **Where the golden tensors are.** `--goldens DIR`, else `SILICONED_GOLDENS`, else `goldens/` in the
/// current directory. They are not shipped: the reference's licenses do not allow it, and they weigh
/// gigabytes. The oracle scripts write them (`siliconed check list` names the script of each stage).
package struct Goldens: Sendable {
    package let directory: String
    /// Every golden opened, kept mapped as long as the invocation lives: a stage reads them through
    /// pointers (`float32`), and a release build may free a file right after its last named use.
    private let opened = Opened()

    private final class Opened: @unchecked Sendable {
        private let lock = NSLock()
        private var files: [Safetensors] = []
        func keep(_ file: Safetensors) -> Safetensors { lock.withLock { files.append(file) }; return file }
    }

    package init(directory: String?) {
        self.directory = directory ?? ProcessInfo.processInfo.environment["SILICONED_GOLDENS"] ?? "goldens"
    }

    package func path(_ file: String) -> String { (directory as NSString).appendingPathComponent(file) }

    package func exists(_ file: String) -> Bool { FileManager.default.fileExists(atPath: path(file)) }

    /// A golden the stage cannot run without: missing, the check is skipped and says which.
    package func open(_ file: String) throws -> Safetensors {
        let p = path(file)
        guard FileManager.default.fileExists(atPath: p) else {
            throw Skip("golden missing: \(p) — written by the stage's oracle script (`siliconed check list`)")
        }
        return opened.keep(try Safetensors(path: p))
    }

    /// An input file of the reference's (a test image), in `inputs/` beside the goldens.
    package func input(_ name: String) throws -> URL {
        let p = path("inputs/" + name)
        guard FileManager.default.fileExists(atPath: p) else { throw Skip("input missing: \(p)") }
        return URL(fileURLWithPath: p)
    }

    /// A golden's manifest, the `.json` the oracle writes beside it.
    package func manifest(_ stem: String) throws -> [String: Any] {
        let p = path(stem + ".json")
        guard FileManager.default.fileExists(atPath: p) else { throw Skip("manifest missing: \(p)") }
        return try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: p))) as? [String: Any] ?? [:]
    }

    /// **The reference run in bf16**, which a bf16 path is judged beside (`Judge`'s `half`): missing, the stage is
    /// skipped with the script that writes it — never judged against an f32 floor the path does not claim.
    package func half(_ file: String, script: String) throws -> Safetensors {
        guard exists(file) else { throw Skip("bf16 reference missing: \(path(file)) — \(script)") }
        return opened.keep(try Safetensors(path: path(file)))
    }

    /// **The reference run with the publisher's int8 products** (ComfyUI's `int8_linear`: the activation rotated,
    /// cast into int8 with a scale a row, int8 × int8 exact, the rest in bf16), which a Compact's DiT is judged beside
    /// in place of the bf16 one: its publisher's own arithmetic on the same codes, which the engine's W8A16 is held to.
    /// Missing, the stage is skipped.
    package func int8(_ file: String, script: String) throws -> Safetensors {
        guard exists(file) else { throw Skip("W8A8 reference missing: \(path(file)) — \(script)") }
        return opened.keep(try Safetensors(path: path(file)))
    }

    /// A golden the stage reads when present (a witness, a bf16 reference).
    package func optional(_ file: String) -> Safetensors? {
        exists(file) ? (try? Safetensors(path: path(file))).map(opened.keep) : nil
    }
}
