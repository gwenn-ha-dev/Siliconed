import Foundation
import Siliconed

// **One check's arguments, the same grammar for every stage**:
//
//     siliconed check <model> <stage> [--size 512|1024|WxH] [--edit N] [--goldens DIR] [--map PATH] [--profile]
//                                     [--neural-engine] [--bf16-activations] [--bf16-attention | --fp16-attention] [--long]
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
    /// `--map PATH`: another DiT map than the installed Standard (a Compact, a Light, an import).
    package let map: String?
    /// `--profile`: the evaluation's time budget after the verdict's figures.
    package let profile: Bool
    /// `--neural-engine`: the DiT's products shared with the Neural Engine (`NeuralEngineRun`).
    package let neuralEngine: Bool
    /// `--bf16-activations`: the DiT's products on an activation rounded to bf16 (`BF16ActivationRun`).
    package let bf16Activations: Bool
    /// `--bf16-attention`: the DiT's attention on bf16 operands, fused (`BF16AttentionRun`).
    package let bf16Attention: Bool
    /// `--fp16-attention`: the DiT's attention on fp16 operands, fused (`FP16AttentionRun`) — the `tiles16` a render takes where the map
    /// chooses it, at every version.
    package let fp16Attention: Bool
    /// Some half-precision flag is given: the stage is judged against the reference run in bf16.
    package var halfPrecision: Bool { neuralEngine || bf16Activations || bf16Attention || fp16Attention }
    /// `--no-accelerators`: this Mac's map (`SILICONED_MAP`) with the « Accelerators » switch off — fp32 throughout (`FP32Run`).
    package let noAccelerators: Bool
    /// `--long`: the text stages on the oracle's long prompt (past 512 tokens), golden `…-text-long`.
    package let long: Bool
    package let goldens: Goldens
    /// A stage's own options (`--sequence 256`), by name, the last value given.
    package let extras: [String: String]

    /// The options every stage accepts.
    package static let common: Set<String> = ["--size", "--goldens", "--profile"]

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
        profile = flags.contains("--profile")
        neuralEngine = flags.contains("--neural-engine")
        bf16Activations = flags.contains("--bf16-activations")
        bf16Attention = flags.contains("--bf16-attention")
        fp16Attention = flags.contains("--fp16-attention")
        noAccelerators = flags.contains("--no-accelerators")
        if bf16Attention && fp16Attention {
            throw UsageError("--bf16-attention and --fp16-attention: one precision of the attention at a time")
        }
        if noAccelerators && (neuralEngine || bf16Activations || bf16Attention || fp16Attention) {
            throw UsageError("--no-accelerators is fp32 throughout: not with --neural-engine, --bf16-activations, --bf16-attention or --fp16-attention")
        }
        if neuralEngine || bf16Activations || bf16Attention || fp16Attention {
            try HalfPrecisionRun.enable(neuralEngine: neuralEngine, bf16Activations: bf16Activations, bf16Attention: bf16Attention,
                                        fp16Attention: fp16Attention)
        }
        if noAccelerators { try FP32Run.enable() }
        long = flags.contains("--long")
        goldens = Goldens(directory: values["--goldens"]?.last)
        extras = values.compactMapValues(\.last).filter { !Self.known.contains($0.key) }
    }

    /// The options that take no value.
    private static let flags: Set<String> = ["--profile", "--neural-engine", "--bf16-activations", "--bf16-attention", "--fp16-attention",
                                             "--no-accelerators", "--long"]

    /// The options parsed above; any other accepted option lands in `extras`.
    private static let known: Set<String> = ["--size", "--goldens", "--edit", "--map"]

    /// The DiT map: `--map`, else the installed one named `standard`; missing or of another variant than the golden's, the stage is skipped.
    /// `golden`: the variant the stage's golden was made for; another one is skipped (`MapCheck`).
    package func ditMap(_ standard: String, golden: MapVariant = .standard) throws -> String {
        let path = map ?? Library.processWide.map(standard)
        guard FileManager.default.fileExists(atPath: path) else { throw Skip("map missing: \(path) — siliconed install") }
        return try MapCheck.use(path, golden: golden)
    }

    /// The bf16 reference beside `file` when the engine runs a half-precision path. Under `--neural-engine`, `--bf16-activations`,
    /// `--bf16-attention` or `--fp16-attention` it is required: the fp32 rule would fail the ANE's fp16 products for what they are, not for a fault.
    package func halfReference(_ file: String, script: String) throws -> HalfPrecisionReference? {
        let half = HalfPrecisionReference(goldens: goldens, file: file)
        if halfPrecision, half == nil { throw Skip("bf16 reference missing beside \(file) — \(script)") }
        return half
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

    /// A golden the stage reads when present (a witness, a bf16 reference).
    package func optional(_ file: String) -> Safetensors? {
        exists(file) ? (try? Safetensors(path: path(file))).map(opened.keep) : nil
    }
}
