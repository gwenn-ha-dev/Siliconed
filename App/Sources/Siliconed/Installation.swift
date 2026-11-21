import Foundation

/// **Installing a family is downloading its published files** — nothing is converted, nothing is written but what a
/// publisher published: each file a version declares (`Family.files(_:)`) goes from its repository, at its pinned
/// commit, to its place under `store/`, and bears its name only once its bytes have the pinned size and digest.
///
///     store/<published name>              a DiT, an encoder: one file each, as published
///     store/composants/<family>/…         tokenizers, VAE, shards behind their index, the family's own LoRA
///     store/composants/<family>/variant   the version the last installation aimed at (`Library.variant(of:)`)
///     store/composants/<family>/hashed    the shared files hashed in this library, and their digests (`checkShared`)
///
/// A download cut short keeps what it had (`….partiel` and its ledger, beside the final name) and resumes; a file
/// already there with the pinned size is not downloaded again (`verify` hashes what is there). A family's two versions
/// share its small files, and each keeps its own DiT and encoder: both can be installed, and the render reads the one
/// the last installation aimed at.
package struct Installer {
    package let library: Library
    package var journal: @Sendable (String) -> Void = { print($0) }
    /// Once raised, it stops the installation between two files or in the middle of one (`EngineError.cancelled`).
    /// What was downloaded stays, and the next attempt resumes from it.
    package var cancellation: Cancellation?
    /// Where the installation is, as it moves: what an app draws (the journal is a terminal's).
    package var progress: (@Sendable (Progress) -> Void)?

    package init(library: Library) { self.library = library }

    /// **Where an installation is**: the file coming, and the bytes of the whole installation on disk so far — those an
    /// interrupted attempt left included, so that a resumed installation starts where it stopped.
    package struct Progress: Sendable, Equatable {
        /// `Z-Image-Turbo/vae/diffusion_pytorch_model.safetensors` (`PublishedFile.label`).
        package let file: String
        /// The file's rank among those to download, from 1, and how many there are.
        package let index: Int, count: Int
        package let received: Int, total: Int
        package init(file: String, index: Int, count: Int, received: Int, total: Int) {
            (self.file, self.index, self.count, self.received, self.total) = (file, index, count, received, total)
        }
        package var fraction: Double { total > 0 ? min(1, Double(received) / Double(total)) : 1 }
    }

    // MARK: - What is there

    package func path(_ file: PublishedFile) -> String { library.map(file.place) }

    /// There, with the pinned size. The digest was checked when the file took its name (`Downloader.fetch`).
    package func isPresent(_ file: PublishedFile) -> Bool { Library.fileSize(path(file)) == file.bytes }

    /// The files of a version that are not there yet.
    package func missing(_ f: Family, _ v: Variant) -> [PublishedFile] { f.files(v).filter { !isPresent($0) } }

    package func isInstalled(_ f: Family, _ v: Variant) -> Bool { missing(f, v).isEmpty }

    /// The versions of a family that are whole on disk.
    package func installed(_ f: Family) -> [Variant] { f.variants.filter { isInstalled(f, $0) } }

    // MARK: - The plan

    /// **What an installation would do**, known before its first byte: the files to download, their bytes, what an
    /// interrupted attempt already holds of them, and the room on the volume.
    package struct Plan: Sendable {
        package let family: Family
        package let variant: Variant
        package let files: [PublishedFile]
        /// The bytes of `files`.
        package let bytes: Int
        /// The bytes of `files` an interrupted download already wrote (`.partiel`).
        package let resumed: Int
        package let free: Int?
        /// The family's other versions whole on disk, and the bytes their removal would give back: their files that
        /// this version does not read.
        package var others: [Variant] = []
        package var releasable = 0
        /// The bytes the volume still has to take.
        package var needed: Int { bytes - resumed }
        package var fits: Bool { free.map { needed <= $0 } ?? true }
        /// **The volume takes this version only in place of the others**: what an app offers when the room is short
        /// (`install(_:variant:replacing:)`). Until the installation is done the family then renders nothing.
        package var fitsReplacing: Bool { !fits && releasable > 0 && free.map { needed <= $0 + releasable } ?? false }
    }

    /// The files of the family's other installed versions that `v` does not read: what `uninstall(_:variant:keeping:)`
    /// removes to make room for it.
    func releasable(_ f: Family, for v: Variant) -> [PublishedFile] {
        let kept = Set(f.files(v).map(\.place))
        return Array(Set(f.variants.filter { $0 != v && isInstalled(f, $0) }.flatMap { f.files($0) }.filter { !kept.contains($0.place) }))
    }

    package func plan(_ f: Family, _ v: Variant) -> Plan {
        let files = missing(f, v)
        let resumed = files.reduce(0) { $0 + min($1.bytes, Library.reclaimable(Downloader.partial(path($1)))) }
        var plan = Plan(family: f, variant: v, files: files, bytes: files.reduce(0) { $0 + $1.bytes }, resumed: resumed,
                        free: library.freeSpace())
        plan.others = f.variants.filter { $0 != v && isInstalled(f, $0) }
        // A linked file gives back nothing of this volume: its link goes, its target stays.
        plan.releasable = releasable(f, for: v).reduce(0) { sum, file in
            (try? FileManager.default.destinationOfSymbolicLink(atPath: path(file))) == nil ? sum + file.bytes : sum
        }
        return plan
    }

    // MARK: - The license

    /// **What a user reads before the first byte**: the model's license files, then the card of each repository the
    /// version also downloads from — a third party publishes no license file, its card declares the publisher's and
    /// says whose bytes these are.
    package static func licensePages(_ f: Family, _ v: Variant) -> [URL] {
        var pages = f.traits.card.license.urls
        for repository in f.files(v).map(\.repository) {
            let card = repository.page("README.md")
            let named = pages.contains { $0.path.hasPrefix("/\(repository.name)/") }
            if !named && !pages.contains(card) { pages.append(card) }
        }
        return pages
    }

    // MARK: - Installing

    /// **The version a render reads, among those whole on disk** (`Library.variant(of:)`): what an installation writes
    /// before its first byte, and what an app writes when the user picks the other installed version.
    package func use(_ f: Family, variant v: Variant) throws(EngineError) {
        let marker = library.intendedVariantPath(f)
        try EngineError.boundary {
            try FileManager.default.createDirectory(atPath: (marker as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
            try Data(v.rawValue.utf8).write(to: URL(fileURLWithPath: marker), options: .atomic)
        }
    }

    /// **Installs a version of a family**: its missing files, one after the other, each verified. Refused before the
    /// first byte when the license is not accepted or the volume cannot take what is left to download.
    /// `replacing`: when the volume takes this version only in place of the family's others (`Plan.fitsReplacing`),
    /// they are removed first — the user was told, and the family renders nothing until this version is whole.
    package func install(_ f: Family, variant v: Variant = .standard, replacing: Bool = false) throws(EngineError) {
        guard f.variants.contains(v) else { throw .unknownModel(model: "\(f.rawValue) \(v.rawValue)") }
        let card = f.traits.card
        guard Licenses.implicit || library.isLicenseAccepted(card) else { throw .licenseNotAccepted(model: card.id) }
        var plan = plan(f, v)
        journal("── \(f.name) (\(v.rawValue))")
        try cancellation.check()
        if replacing && plan.fitsReplacing {
            for other in plan.others {
                let freed = try uninstall(f, variant: other, keeping: v)
                journal(String(format: "  removed the %@ version to make room: %.2f GB freed", other.rawValue, Double(freed) / 1e9))
            }
            plan = self.plan(f, v)
        }
        guard plan.fits else { throw .diskFull(needed: plan.needed, available: plan.free) }
        let files = plan.files, total = plan.bytes
        try use(f, variant: v)
        try EngineError.boundary {
            var before = 0
            for (k, file) in files.enumerated() {
                try cancellation.check()
                try Downloader.fetch(file, to: path(file), cancellation: cancellation,
                                     received: told(file, index: k + 1, count: files.count, before: before, total: total),
                                     journal: journal)
                before += file.bytes
                if shared(f).contains(file) { try record(f, file) }
            }
        }
        journal(files.isEmpty ? "  already installed" : String(format: "  installed: %d file%@, %.2f GB", files.count, files.count == 1 ? "" : "s", Double(total) / 1e9))
    }

    /// What `Downloader.fetch` tells of one file, as the progress of the whole installation: `before` bytes came with
    /// the files before it.
    private func told(_ file: PublishedFile, index: Int, count: Int, before: Int, total: Int) -> (@Sendable (Int) -> Void)? {
        guard let progress else { return nil }
        let label = file.label
        return { progress(Progress(file: label, index: index, count: count, received: before + $0, total: total)) }
    }

    /// **Hashes what is installed**: the files of the version whose bytes are not the pinned ones (absent ones
    /// included). Reads every byte: minutes for a large family.
    package func verify(_ f: Family, _ v: Variant) throws(EngineError) -> [PublishedFile] {
        var damaged: [PublishedFile] = []
        for file in f.files(v) {
            try cancellation.check()
            guard isPresent(file) else { journal("✗ \(file.place): \(Library.exists(path(file)) ? "wrong size" : "absent")"); damaged.append(file); continue }
            let same = try EngineError.boundary { try file.digest(of: path(file)) == file.digest }
            journal("\(same ? "✓" : "✗") \(file.place)")
            if !same { damaged.append(file) }
        }
        return damaged
    }

    // MARK: - The shared files, hashed once

    /// **The files every version of a family reads**: its tokenizers, its VAE, its own LoRA — 0.2 to 1.5 GB a family,
    /// against gigabytes for a DiT or an encoder, which each version has its own.
    func shared(_ f: Family) -> [PublishedFile] {
        let versions = f.variants.map { Set(f.files($0)) }
        return f.files(f.variants[0]).filter { file in versions.allSatisfy { $0.contains(file) } }
    }

    /// The record of the shared files hashed in this library: a line `<place>\t<digest>` each.
    private func hashedPath(_ f: Family) -> String { library.map("composants/\(f.rawValue)/hashed") }

    private func hashed(_ f: Family) -> Set<String> {
        guard let text = try? String(contentsOfFile: hashedPath(f), encoding: .utf8) else { return [] }
        return Set(text.split(separator: "\n").map(String.init))
    }

    private func line(_ file: PublishedFile) -> String { "\(file.place)\t\(file.digest.text)" }

    private func record(_ f: Family, _ file: PublishedFile) throws {
        let lines = hashed(f).union([line(file)]).sorted()
        try (lines.joined(separator: "\n") + "\n").write(toFile: hashedPath(f), atomically: true, encoding: .utf8)
    }

    /// **Hashes once the shared files of `f` this library never hashed** — those an older version of the app left,
    /// which this one takes on their size alone (`isPresent`). One whose bytes are not the pinned ones is removed and
    /// refused (`corruptMap`): the family is then not installed, and installing it again downloads that file alone.
    /// Under a second a family; afterwards, the read of a few lines. Issue 17: a Qwen whose renders all came out
    /// non-finite after the update from 0.21 was cured by removing its `composants/` folder, whose files are these.
    package func checkShared(_ f: Family) throws(EngineError) {
        let done = hashed(f)
        for file in shared(f) where !done.contains(line(file)) && isPresent(file) {
            try cancellation.check()
            try EngineError.boundary {
                guard try file.digest(of: path(file)) == file.digest else {
                    journal("✗ \(file.place): not the pinned bytes, removed")
                    try FileManager.default.removeItem(atPath: path(file))
                    throw EngineError.corruptMap(file: path(file))
                }
                try record(f, file)
            }
        }
    }

    // MARK: - Removing

    /// **Removes a version of a family** — or the whole family (`variant: nil`): its files, what an interrupted
    /// download left of them, and the family's folder once empty. A version leaves the files another installed version
    /// of the family still reads — and what `keeping` reads, installed or not: the version about to take its place.
    /// Returns the bytes freed. A linked file loses its link, never its target.
    @discardableResult
    package func uninstall(_ f: Family, variant: Variant? = nil, keeping: Variant? = nil) throws(EngineError) -> Int {
        let fm = FileManager.default
        let going = (variant.map { [$0] } ?? f.variants).flatMap { f.files($0) }
        let kept = variant == nil ? [] : Set(f.variants.filter { $0 != variant && (isInstalled(f, $0) || $0 == keeping) }
            .flatMap { f.files($0) }.map(\.place))
        var freed = 0
        try EngineError.boundary {
            for file in Set(going) where !kept.contains(file.place) {
                for p in [path(file), Downloader.partial(path(file)), Downloader.ledger(path(file))] where (try? fm.attributesOfItem(atPath: p)) != nil {
                    freed += Library.reclaimable(p)
                    try fm.removeItem(atPath: p)
                }
            }
            if installed(f).isEmpty {
                try? fm.removeItem(atPath: library.intendedVariantPath(f))
                try? fm.removeItem(atPath: hashedPath(f))
            }
            // The folders the files leave empty, deepest first; never one that still holds something.
            let folder = library.map("composants/\(f.rawValue)")
            let folders = (fm.enumerator(atPath: folder)?.allObjects as? [String] ?? []).filter { Library.isFolder(folder + "/" + $0) }
            for sub in folders.sorted(by: { $0.count > $1.count }) + [""] {
                let p = sub.isEmpty ? folder : folder + "/" + sub
                if (try? fm.contentsOfDirectory(atPath: p).filter { $0 != ".DS_Store" })?.isEmpty == true { try? fm.removeItem(atPath: p) }
            }
        }
        return freed
    }

    /// **Throws away what interrupted downloads left** under `store/` — never one being written: a download holds its
    /// `.partiel` locked. Returns the bytes freed.
    @discardableResult
    package func clean() throws(EngineError) -> Int {
        let store = library.root.appendingPathComponent("store").path
        var freed = 0
        try EngineError.boundary {
            for file in Library.regularFiles(under: store) where file.path.hasSuffix(".partiel") {
                let fd = open(file.path, O_RDONLY)
                guard fd >= 0 else { continue }
                defer { close(fd) }
                guard flock(fd, LOCK_EX | LOCK_NB) == 0 else { continue }      // being written
                freed += Library.reclaimable(file.path)
                try FileManager.default.removeItem(atPath: file.path)
            }
        }
        return freed
    }
}
