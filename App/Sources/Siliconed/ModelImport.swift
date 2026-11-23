import Foundation
import os

// **Importing what a user brings**: a `.safetensors`, a LoRA or a model, as its author published it. Nothing is
// converted: the file is recognized, copied as it is into the library, and the render reads it there.
//
//     a LoRA    the family whose graph applies every module it adapts (`LoRAFile.family`) — `store/loras/<family>/`
//     a model   a DiT alone, without encoder or VAE: the family whose graph finds every tensor it reads in the file,
//               at its shape, at the file's root or under a checkpoint's prefix (`MapWeights(dit:)`) —
//               `store/importes/<family>/`. It renders with the family's installed text encoder and VAE, in the
//               arithmetic its file is stored in (ComfyUI's int8, else bf16: a weight stored in fp32, fp16 or ComfyUI's scaled fp8 is loaded in bf16)
//
// **Recognized by its tensors, never by its name or its metadata**: a fine-tune keeps its architecture's names and
// shapes, and a file is its family's when that family's graph can be built over it — the same graph that renders.
//
// **An import downloads nothing**: a model whose family is not installed is refused (`importNeedsFamily`), and the
// app offers the installation first.

/// A published file told by its bytes: where it was read (for whoever maintains the table), its size, its sha256.
package struct BaseCheckpoint: Sendable {
    package let source: String
    package let bytes: Int
    package let sha256: String
    package init(_ source: String, _ bytes: Int, _ sha256: String) { self.source = source; self.bytes = bytes; self.sha256 = sha256 }
}

package enum ModelImport {
    package enum Kind: String, Sendable { case lora, model }

    package struct Outcome: Sendable {
        package let kind: Kind
        package let family: Family
        /// The name shown: the file's, without its extension.
        package let name: String
        /// Where the library keeps the file.
        package let path: String
        /// A model's identifier (`<family>/<name>`); `nil` for a LoRA.
        package let identifier: String?
    }

    /// **The tensors of `family` the file lacks**, by the names its graph reads: absent, or of another shape.
    static func lacking(_ family: [String: [Int]], shape: (String) -> [Int]?) -> [String] {
        family.filter { shape($0.key) != $0.value }.keys.sorted()
    }

    /// **A model deeper than its family, at the same width**: a stack the family reads (`blocks.0` … `blocks.27`) that the
    /// file carries further (`blocks.28` …). Every tensor the family reads is there at its shape, so the family would take
    /// it, and render on its first layers only: a wrong image (Fn-Moment Anima V5, 23/11/2026: 40 blocks where Anima reads
    /// 28). Refused, said as coming, until a model's depth is read from its file. Tensors the family has no name for (a checkpoint's VAE and
    /// text encoder beside the DiT) are not depth. The first such stack: what the family reads of it, what the file holds.
    static func deeper(_ family: [String: [Int]], shape: (String) -> [Int]?) -> (reads: Int, holds: Int)? {
        // Each indexed name split at its first index: the stack before it, what follows; the family's count of each.
        var stacks: [String: (stack: String, rest: String, reads: Int)] = [:]
        for name in family.keys {
            let parts = name.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
            guard let i = parts.firstIndex(where: { Int($0) != nil }), let n = Int(parts[i]) else { continue }
            let stack = parts[..<i].joined(separator: "."), rest = parts[(i + 1)...].joined(separator: ".")
            let key = stack + "\u{0}" + rest
            if (stacks[key]?.reads ?? 0) < n + 1 { stacks[key] = (stack, rest, n + 1) }
        }
        func named(_ s: (stack: String, rest: String, reads: Int), _ n: Int) -> String { s.rest.isEmpty ? "\(s.stack).\(n)" : "\(s.stack).\(n).\(s.rest)" }
        for s in stacks.values.sorted(by: { ($0.stack, $0.rest) < ($1.stack, $1.rest) }) where shape(named(s, s.reads)) != nil {
            var holds = s.reads + 1
            while shape(named(s, holds)) != nil { holds += 1 }
            return (s.reads, holds)
        }
        return nil
    }

    /// A family's DiT tensors (`FamilyTraits.ditTensors`), its graph built once a process: a classification reads all
    /// five, and an import classifies twice (`recognize`, then `importFile`).
    static func tensors(of family: Family) -> [String: [Int]] {
        known.withLock { known in
            if let t = known[family] { return t }
            let t = family.traits.ditTensors()
            known[family] = t
            return t
        }
    }
    private static let known = OSAllocatedUnfairLock<[Family: [String: [Int]]]>(initialState: [:])

    /// **The family whose DiT the tensors are**: the one whose graph lacks none. `nil` when none does, or — never
    /// seen — when several do.
    static func family(shape: (String) -> [Int]?) -> Family? {
        let fitting = Family.allCases.filter { f in
            let tensors = tensors(of: f)
            return !tensors.isEmpty && lacking(tensors, shape: shape).isEmpty
        }
        return fitting.count == 1 ? fitting[0] : nil
    }

    /// **Refuses a publisher's undistilled model** (`FamilyTraits.baseCheckpoints`): `importBaseModel` when `file` is
    /// one of them, byte for byte. The size first; the hash — seconds on a large file — only when a size agrees.
    static func refuseBase(_ file: String, _ family: Family, hash: (String) throws -> String = Hashes.shared.sha256) throws {
        guard let bytes = Library.fileSize(file) else { return }
        let sized = family.traits.baseCheckpoints.filter { $0.bytes == bytes }
        guard !sized.isEmpty else { return }
        let digest = try hash(file)
        if sized.contains(where: { $0.sha256 == digest }) {
            throw EngineError.importBaseModel(file: (file as NSString).lastPathComponent, model: ModelCard.of(family).id)
        }
    }

    /// **One hash per file, not two**: the app asks `recognize` before `importFile`, on the same file, and a
    /// fine-tune saved at its base's size would be hashed twice. Kept while the file keeps its size, its
    /// modification date and its inode.
    final class Hashes: @unchecked Sendable {
        static let shared = Hashes()
        private let lock = NSLock()
        private var known: [String: String] = [:]

        func sha256(_ path: String) throws -> String {
            var info = stat()
            let stamp = stat(path, &info) == 0 ? "\(info.st_size) \(info.st_mtimespec.tv_sec).\(info.st_mtimespec.tv_nsec) \(info.st_ino)" : "-"
            let key = "\(path)|\(stamp)"
            if let sha = lock.withLock({ known[key] }) { return sha }
            let sha = try Downloader.sha256(path)
            lock.withLock {
                if known.count >= 16 { known.removeAll() }
                known[key] = sha
            }
            return sha
        }
    }

    /// **What the file is, read no further than its header**: a model of a family, or a LoRA of one. Refused by
    /// name otherwise (`importRefused`, the closest family's missing tensors said; a LoRA's own refusals).
    /// `sha256`: the file's digest when its publisher already said it (`HubImport.examine`, on a file whose data is
    /// not there to hash).
    package static func recognize(_ file: String, sha256: String? = nil) throws(EngineError) -> (Kind, Family) {
        try EngineError.boundary {
            let name = (file as NSString).lastPathComponent
            guard FileManager.default.fileExists(atPath: file) else { throw EngineError.fileMissing(path: file) }
            let map: MapWeights
            do { map = try MapWeights(dit: file) } catch { throw EngineError.importRefused(file: name, detail: "\(error)") }
            if let family = family(shape: map.shape) {
                if let d = deeper(tensors(of: family), shape: map.shape) {
                    throw EngineError.importDeeperModel(file: name, model: ModelCard.of(family).id, holds: d.holds, reads: d.reads)
                }
                try refuseBase(file, family, hash: sha256.map { said in { _ in said } } ?? Hashes.shared.sha256)
                return (.model, family)
            }
            if map.keys.contains(where: { key in LoRAFile.downSuffixes.contains(where: key.hasSuffix) }) {
                let lora = try LoRAFile(file)
                if let family = lora.family() { return (.lora, family) }
                throw EngineError.importRefused(file: name, detail: lora.refusal())
            }
            var closest: (family: Family, lacking: [String], of: Int)?
            for f in Family.allCases {
                let tensors = tensors(of: f)
                guard !tensors.isEmpty else { continue }
                let lacks = lacking(tensors, shape: map.shape)
                if closest.map({ lacks.count * $0.of < $0.lacking.count * tensors.count }) ?? true { closest = (f, lacks, tensors.count) }
            }
            // A file that shares tensors with a family is said how far from it it is; one that shares none, nothing more.
            let detail = closest.flatMap { c -> String? in
                guard c.lacking.count < c.of else { return nil }
                return "neither a LoRA nor a model: \(c.lacking.count) of \(c.family.rawValue)'s \(c.of) tensors are not in it, e.g. " + c.lacking.prefix(2).joined(separator: ", ")
            }
            throw EngineError.importRefused(file: name, detail: detail ?? "neither a LoRA nor a model of a family this app renders")
        }
    }

    /// **Imports a file**: recognized, then copied as it is into its family's folder. A file of the same name is
    /// replaced. A model whose family has no version installed is refused before a byte is copied. `cancellation`
    /// raised (« Stop »), the copy — a model's or a LoRA's — stops where it is and throws `cancelled`: the library is
    /// as it was before (`copyIn`).
    package static func importFile(_ file: String, in b: Library, cancellation: Cancellation? = nil) throws(EngineError) -> Outcome {
        try cancellation.check()
        let (kind, family) = try recognize(file)
        try cancellation.check()
        let name = (file as NSString).lastPathComponent, stem = (name as NSString).deletingPathExtension
        if kind == .lora {
            let card = try b.importLoRA(file, cancellation: cancellation)
            return Outcome(kind: .lora, family: family, name: card.name, path: card.path, identifier: nil)
        }
        guard !Installer(library: b).installed(family).isEmpty else {
            throw EngineError.importNeedsFamily(file: name, model: ModelCard.of(family).id)
        }
        return try EngineError.boundary {
            let folder = b.importedFolder(family), target = (folder as NSString).appendingPathComponent(stem + ".safetensors")
            try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
            try copyIn(file, to: target, cancellation: cancellation)
            return Outcome(kind: .model, family: family, name: stem, path: target, identifier: "\(family.rawValue)/\(stem)")
        }
    }

    /// **A file copied into the library whole, or not at all**: the file itself, not a link to it (what the library
    /// holds must outlive where it came from), copied beside `target` under a `.partiel` name, then renamed over it.
    /// A copy cut short — the app killed, the disk full — leaves the file `target` was, never a truncated one listed
    /// as a model or a LoRA; what it leaves is a `.partiel`, which `Installer.clean` throws away. Nothing is done when
    /// `file` already is `target`.
    ///
    /// **Stopped** (`cancellation` raised), the copy quits between two of its blocks — `copyfile`'s progress callback —, its
    /// `.partiel` is removed, and it throws `EngineError.cancelled`: `target` stays what it was, and `file`, the user's, is
    /// only ever read. `COPYFILE_CLONE` makes the copy a clone on the same APFS volume (instant, no space taken), as
    /// `FileManager.copyItem` did; a clone never calls back, so the token is checked once more before the rename.
    static func copyIn(_ file: String, to target: String, cancellation: Cancellation? = nil) throws {
        let fm = FileManager.default
        let source = URL(fileURLWithPath: file).resolvingSymlinksInPath().path
        guard source != URL(fileURLWithPath: target).resolvingSymlinksInPath().path else { return }
        let partial = Downloader.partial(target)
        try? fm.removeItem(atPath: partial)
        func fail(_ code: Int32) -> Error {
            try? fm.removeItem(atPath: partial)
            return code == ECANCELED ? EngineError.cancelled : NSError(domain: NSPOSIXErrorDomain, code: Int(code), userInfo: [NSFilePathErrorKey: target])
        }
        try cancellation.check()
        let state = copyfile_state_alloc()
        defer { copyfile_state_free(state) }
        if let cancellation {
            let callback: copyfile_callback_t = { what, stage, _, _, _, context in
                guard what == COPYFILE_COPY_DATA, stage == COPYFILE_PROGRESS, let context else { return COPYFILE_CONTINUE }
                return Unmanaged<Cancellation>.fromOpaque(context).takeUnretainedValue().isCancelled ? COPYFILE_QUIT : COPYFILE_CONTINUE
            }
            copyfile_state_set(state, UInt32(COPYFILE_STATE_STATUS_CB), unsafeBitCast(callback, to: UnsafeRawPointer.self))
            copyfile_state_set(state, UInt32(COPYFILE_STATE_STATUS_CTX), Unmanaged.passUnretained(cancellation).toOpaque())
        }
        // The token lives through the call (`withExtendedLifetime`): the callback reads it unretained.
        let copied = withExtendedLifetime(cancellation) { copyfile(source, partial, state, copyfile_flags_t(COPYFILE_CLONE)) }
        guard copied == 0 else { throw fail(errno) }
        guard cancellation?.isCancelled != true else { throw fail(ECANCELED) }
        guard rename(partial, target) == 0 else { throw fail(errno) }
    }

    /// **Removes an imported model**: its file. The family's own files stay.
    package static func remove(_ card: ModelCard, in b: Library) throws(EngineError) {
        guard let map = card.map else { throw EngineError.notAnImportedModel(model: card.id) }
        try EngineError.boundary { try FileManager.default.removeItem(atPath: b.map(map)) }
    }
}

extension LoRAFile {
    /// Why no family applies the file: the closest one's modules it does not reach — a family reaching none of them is
    /// not close (all missing all, the first one would be named: issue 17, a LoRA of one family refused as another's).
    func refusal() -> String {
        let closest = Family.allCases.map { ($0, unapplied(by: LoRAStack.modules(of: $0))) }.min { $0.1.count < $1.1.count }
        guard let closest, closest.1.count < modules.count else {
            return "none of its \(modules.count) modules is one a family adapts, e.g. \(modules.keys.sorted().prefix(2).joined(separator: ", "))"
        }
        return "\(closest.1.count) of its \(modules.count) modules are not \(closest.0.rawValue)'s, e.g. \(closest.1.prefix(2).joined(separator: ", "))"
    }
}

extension Library {
    /// Where the library keeps a family's imported models: `store/importes/<family>/`, each the file its author wrote.
    package func importedFolder(_ family: Family) -> String { map("importes/\(family.rawValue)") }

    /// **The imported models**, by family then by name: each a card of its family's — its license, its steps, its
    /// formats — under its file's name. Read from the folder's names alone; a file that no longer reads is said when
    /// the model is built (`ModelCard.missing(in:)`).
    package func importedCards() -> [ModelCard] {
        Family.allCases.flatMap { family -> [ModelCard] in
            let names = (try? FileManager.default.contentsOfDirectory(atPath: importedFolder(family))) ?? []
            let base = ModelCard.of(family)
            return names.filter { $0.hasSuffix(".safetensors") }.sorted().map { file in
                let stem = (file as NSString).deletingPathExtension
                return ModelCard(id: "\(family.rawValue)/\(stem)", name: stem.replacingOccurrences(of: "_", with: " "), family: family,
                                 map: "importes/\(family.rawValue)/\(file)", license: base.license, defaultSteps: base.defaultSteps,
                                 space: base.space, formats: base.formats)
            }
        }
    }

    /// The card of an identifier: a published model's, or an imported one's.
    package func card(_ id: String) -> ModelCard? { cards().first { $0.id == id } }
}
