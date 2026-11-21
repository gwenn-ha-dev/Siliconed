import Foundation

// **What the library occupies, and what a family weighs before it is there.** An app that downloads tens of GB says
// where they are and how many, version by version; everything that goes can be downloaded again (`Installer`).

extension Library {
    /// **What the library occupies**, in bytes — sizes read from the disk, never a weight.
    package struct Occupancy: Sendable, Equatable {
        /// A family: its versions whole on disk, and what all its files there weigh.
        package struct FamilyPart: Sendable, Identifiable, Equatable {
            package var id: Family { family }
            package let family: Family
            /// The versions whole on disk (`Installer.installed`).
            package let installed: [Variant]
            /// The version a render reads (`Library.variant(of:)`); `nil` while none is whole.
            package let variant: Variant?
            /// The family's published files on disk, each counted once: its two versions share its small ones.
            package let bytes: Int
            package var isReady: Bool { variant != nil }
            /// Something of the family is on disk.
            package var present: Bool { bytes > 0 }

            package init(family: Family, installed: [Variant], variant: Variant?, bytes: Int) {
                self.family = family; self.installed = installed; self.variant = variant; self.bytes = bytes
            }
        }

        package let families: [FamilyPart]
        /// The models the user imported, each with its file's bytes.
        package struct ImportedPart: Sendable, Identifiable, Equatable {
            package var id: String { card.id }
            package let card: ModelCard
            package let bytes: Int
        }
        package let imported: [ImportedPart]
        /// What interrupted downloads left under `store/` (`.partiel`): resumed by the next installation, or given
        /// back by `emptyDownloads()`.
        package let downloads: Int
        /// What an earlier version of the app left under `store/` and this one never reads (`Library.legacyFiles`).
        package let legacy: Int
        /// The free space on the library's volume, if the system reports it.
        package let free: Int?

        package init(families: [FamilyPart], imported: [ImportedPart] = [], downloads: Int, legacy: Int = 0, free: Int?) {
            self.families = families; self.imported = imported; self.downloads = downloads; self.legacy = legacy
            self.free = free
        }

        package var total: Int { families.reduce(0) { $0 + $1.bytes } + imported.reduce(0) { $0 + $1.bytes } + downloads + legacy }
    }

    /// **The whole library**: a walk of `store/` (the downloads left, what 0.21 left), as long as its disk takes — the app
    /// reads it off the main thread. What a decision needs at once is read apart: `part(_:)`, `downloadsLeft()`.
    package func occupancy() -> Occupancy {
        let imported = importedCards().map { Occupancy.ImportedPart(card: $0, bytes: $0.map.flatMap { Self.fileSize(map($0)) } ?? 0) }
        return Occupancy(families: Family.allCases.map(part), imported: imported, downloads: downloadsLeft(),
                         legacy: legacyFiles().reduce(0) { $0 + $1.bytes }, free: freeSpace())
    }

    /// A family's part: its declared files, one look at each — no walk.
    package func part(_ f: Family) -> Occupancy.FamilyPart {
        let installed = Installer(library: self).installed(f)
        let places = Set(f.variants.flatMap { f.files($0) }.map(\.place))
        return .init(family: f, installed: installed, variant: installed.isEmpty ? nil : variant(of: f),
                     bytes: places.reduce(0) { $0 + (Self.fileSize(map($1)) ?? 0) })
    }

    /// What interrupted downloads left under `store/` (`.partiel`): a walk of it.
    package func downloadsLeft() -> Int {
        Self.regularFiles(under: root.appendingPathComponent("store").path).filter { $0.path.hasSuffix(".partiel") }
            .reduce(0) { $0 + Self.reclaimable($1.path) }
    }

    /// **What version 0.21 forged under `store/` and this app never reads**: its `.silicon` maps (models, encoders,
    /// LoRAs), its forged adapters (`….adaptateur.safetensors`), and the configurations it kept beside a family's
    /// components (`dit.json`, `transformer.json`, `vae.json`, `model_index.json`, `scheduler.json`, and the chat
    /// processor's folder — the last three seen on a real 0.21 library, 11 MB). Told by their names alone, and never a place a
    /// published file takes: this app's files are the publishers', and it migrates nothing — what it reads is
    /// downloaded as published. And its render cache, on this Mac (`<local>/cache/`: conditionings and K/V), which
    /// this one keeps in memory only.
    package func legacyFiles() -> [(path: String, bytes: Int)] {
        let store = root.appendingPathComponent("store").path
        let published = Set(Family.allCases.flatMap { f in f.variants.flatMap { f.files($0) } }.map { map($0.place) })
        let configurations: Set<String> = ["dit.json", "transformer.json", "vae.json", "model_index.json", "scheduler.json"]
        let components = Set(Family.allCases.map { store + "/composants/\($0.rawValue)" })
        let processors = Set(components.map { $0 + "/processor" })
        return Self.regularFiles(under: store).compactMap { file in
            let name = (file.path as NSString).lastPathComponent, folder = (file.path as NSString).deletingLastPathComponent
            let forged = name.hasSuffix(".silicon") || name.hasSuffix(".adaptateur.safetensors")
            guard forged || (configurations.contains(name) && components.contains(folder)) || processors.contains(folder),
                  !published.contains(file.path) else { return nil }
            return (file.path, Int(file.info.st_size))
        } + Self.regularFiles(under: local.appendingPathComponent("cache").path).map { ($0.path, Int($0.info.st_size)) }
    }

    /// Deletes `legacyFiles()`, and returns the bytes given back.
    package func releaseLegacy() throws(EngineError) -> Int {
        try EngineError.boundary {
            try legacyFiles().reduce(0) { freed, file in
                try FileManager.default.removeItem(atPath: file.path)
                return freed + file.bytes
            }
        }
    }

    /// Throws away what interrupted downloads left — never one being written (`Installer.clean`).
    package func emptyDownloads() throws(EngineError) { try Installer(library: self).clean() }
}

extension Family {
    /// **What a version takes on disk once installed**: the bytes of its published files.
    package func installedSize(_ v: Variant) -> Int { files(v).reduce(0) { $0 + $1.bytes } }

    /// **The version an installation offers first**: the Compact where one is published — smaller on disk and in
    /// memory, about as fast, and the target is the small Macs.
    package var preselectedVariant: Variant { variants.contains(.compact) ? .compact : .standard }

    /// A community recipe rather than its publisher's (`FamilyTraits.isBeta`).
    package var isBeta: Bool { traits.isBeta }
}
