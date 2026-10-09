import Foundation

// **Managing the library: what occupies the disk, and giving it back.** An app that installs tens
// of GB must say where they are, how many, and know how to give them back — family by family,
// imported model by imported model, LoRA by LoRA. Everything that goes here can be redone: a
// family is reinstalled from the vendor; an imported model or a LoRA is reimported from its
// `.safetensors`.

extension Library {
    /// **An app's library**: `~/Library/Application Support/Siliconed/` — the app's own space,
    /// outside any repository. It is also the command line's last resort (`processWide`): the app
    /// and the CLI then share the same maps, never two copies.
    public static var standard: Library {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return Library(root: support.appendingPathComponent("Siliconed"))
    }

    /// **What the library occupies**, in bytes — read from disk (sizes and headers), never a weight.
    public struct Occupancy: Sendable {
        /// A vendor family: its three parts, each absent (0) or not.
        public struct FamilyPart: Sendable, Identifiable {
            public var id: Family { family }
            public let family: Family
            /// Tokenizers, VAE, configs (`store/composants/<famille>/`).
            public let components: Int
            /// The text encoder map.
            public let encoder: Int
            /// The other families present that read the same encoder map (Z-Image and FLUX.2
            /// [klein]): uninstalling it does not remove the map as long as they are there. When
            /// absent, the family does not count this map (`encoder` is 0): it belongs to its
            /// neighbors.
            public let encoderSharedWith: [Family]
            /// The vendor's DiT (and Anima's adapter).
            public let dit: Int
            /// The version on disk (`nil`: nothing of the family's own is there).
            public let variant: Variant?
            /// Enough to render with a DiT, its own or an imported one.
            public let isReady: Bool
            public var total: Int { components + encoder + dit }
            /// Something of the family is on disk.
            public var present: Bool { total > 0 }
        }

        public let families: [FamilyPart]
        public let imported: [(card: ModelCard, bytes: Int)]
        public let loras: [(card: LoRACard, bytes: Int)]
        /// What an interrupted installation or import left behind: `telechargements/` (the files
        /// downloaded in full are reused by the next attempt, sha256 checked), and the `.partiel` a
        /// crash left in `store/` (`Library.partials`). `emptyDownloads()` gives it back.
        public let downloads: Int
        /// What renders keep between them (`Library.cacheFolder`): conditionings, and the K/V of
        /// Qwen-Image-2.1's last edit (up to 6.4 GB). `emptyCache()` gives it back.
        public let cache: Int
        /// The free space on the library's volume, if the system reports it.
        public let free: Int?

        /// A shared encoder map counts only once.
        public var total: Int {
            var encoders: [String: Int] = [:]
            for p in families {
                let name = p.family.encoderMap(p.variant ?? .standard)
                encoders[name] = max(encoders[name] ?? 0, p.encoder)
            }
            return families.reduce(0) { $0 + $1.components + $1.dit } + encoders.values.reduce(0, +)
                + imported.reduce(0) { $0 + $1.bytes } + loras.reduce(0) { $0 + $1.bytes } + downloads + cache
        }
    }

    public func occupancy() -> Occupancy {
        let i = Installer(library: self)
        let partials = self.partials()
        // A `.partiel` under a family's components counts once, with the downloads.
        func partialBytes(in folder: String) -> Int {
            partials.filter { $0.hasPrefix(folder + "/") }.reduce(0) { $0 + Self.size($1) }
        }
        let parts = Family.offered.map { f -> Occupancy.FamilyPart in
            let dit = map(ditMap(f))
            let neighbors = encoderNeighbors(f)
            // A shared encoder map belongs to the families present: the one that was uninstalled no
            // longer counts it, and does not look "half installed".
            let own = isOwn(f)
            // The other version's maps still there — a switch under way, or one stopped before the
            // new version was complete (`Installer.install` removes the old one only then): they
            // occupy the disk too. An encoder another family reads stays that family's.
            // The Compact and the Light share their encoder: counted once, and not at all when it
            // is the current version's.
            let current = variant(of: f)
            let other = f.variants.filter { $0 != current }
            let otherDiT = other.reduce(0) { $0 + Self.size(map(f.ditMap($1))) + Self.size(adapterPath(fromMap: map(f.ditMap($1)))) }
            let otherEncoder = Set(other.map { f.encoderMap($0) })
                .filter { $0 != f.encoderMap(current) && !isRead($0, byOtherThan: f) }
                .reduce(0) { $0 + Self.size(map($1)) }
            return .init(family: f, components: Self.size(i.componentsFolder(f)) - partialBytes(in: i.componentsFolder(f)),
                         encoder: (own || neighbors.isEmpty ? Self.size(map(encoderMap(f))) : 0) + otherEncoder,
                         encoderSharedWith: neighbors,
                         dit: Self.size(dit) + Self.size(adapterPath(fromMap: dit)) + otherDiT,
                         variant: own ? variant(of: f) : nil,
                         isReady: i.isReady(f))
        }
        return Occupancy(
            families: parts,
            imported: importedCards().map { f in
                (f, Self.size(f.map!) + Self.size(adapterPath(fromMap: f.map!)))
            },
            loras: loras().map { ($0, Self.size($0.path)) },
            downloads: Self.size(i.downloads) + partials.reduce(0) { $0 + Self.size($1) },
            cache: Self.size(cacheFolder.path),
            free: freeSpace())
    }

    /// **Uninstalls a vendor family**: its DiT, its components, and its encoder if no other family
    /// present reads it. The family's imported models and LoRAs stay — they belong to the user;
    /// they will only render once the family is reinstalled (`install(_:baseDiT: false)` is
    /// enough for an imported model).
    public func uninstall(_ f: Family) throws(EngineError) {
        let fm = FileManager.default
        var targets = [Installer(library: self).componentsFolder(f)]
        for v in f.variants {
            let dit = map(f.ditMap(v))
            targets += [dit, adapterPath(fromMap: dit)]
            let encoder = map(f.encoderMap(v))
            if !isRead(f.encoderMap(v), byOtherThan: f), !targets.contains(encoder) { targets.append(encoder) }
        }
        for c in targets where Self.exists(c) { try EngineError.boundary { try fm.removeItem(atPath: c) } }
    }

    // MARK: - Versions

    /// **The version of a family the render reads** (`ditMap(_:)`, `encoderMap(_:)`) — the one that
    /// is **complete** on disk: its encoder map, and its DiT map unless no version has one (an
    /// installation for imported models only, `baseDiT: false`). While a switch is under way the new
    /// version is incomplete and the old one still whole: the render keeps reading the old one until
    /// the new one is forged (`Installer.install` removes the old one only then). When both or
    /// neither are complete — the instant between the new one's last map and the old one's removal,
    /// or a switch stopped after the old one had to go first for lack of room — the version the last
    /// installation aimed at decides (`intendedVariant`); without it, the version other than the
    /// Standard whose DiT map is there (its alone), else the Compact if the 8-bit encoder it shares
    /// with the Light is there. Standard when nothing is installed (`installedVariant(of:)` says
    /// whether something is).
    public func variant(of f: Family) -> Variant {
        guard f.variants.count > 1 else { return .standard }
        let complete = f.variants.filter { isComplete(f, $0) }
        let candidates = complete.isEmpty ? f.variants : complete
        if candidates.count == 1 { return candidates[0] }
        if let aimed = intendedVariant(f), candidates.contains(aimed) { return aimed }
        let others = candidates.filter { $0 != .standard }
        if let v = others.first(where: { Self.exists(map(f.ditMap($0))) }) { return v }
        return others.first { Self.exists(map(f.encoderMap($0))) } ?? .standard
    }

    /// The version of the family's own installation; `nil` when nothing of it is on disk (FLUX.2
    /// [klein]'s encoder alone does not make Z-Image installed).
    public func installedVariant(of f: Family) -> Variant? { isOwn(f) ? variant(of: f) : nil }

    /// A version whose maps are all there: its encoder, and its DiT — unless no version has a DiT.
    func isComplete(_ f: Family, _ v: Variant) -> Bool {
        guard Self.exists(map(f.encoderMap(v))) else { return false }
        return Self.exists(map(f.ditMap(v))) || !f.variants.contains { $0 != v && Self.exists(map(f.ditMap($0))) }
    }

    /// `store/composants/<famille>/variant`: the version the last installation of the family aimed
    /// at, written before its first byte. Read only to break a tie (`variant(of:)`).
    func intendedVariantPath(_ f: Family) -> String { Installer(library: self).componentsFolder(f) + "/variant" }

    func intendedVariant(_ f: Family) -> Variant? {
        (try? String(contentsOfFile: intendedVariantPath(f), encoding: .utf8))
            .flatMap { Variant(rawValue: $0.trimmingCharacters(in: .whitespacesAndNewlines)) }
    }

    /// The DiT map the family renders with, by name in `store/` — that of its installed version.
    package func ditMap(_ f: Family) -> String { f.ditMap(variant(of: f)) }

    /// The encoder map the family reads, by name in `store/` — that of its installed version.
    package func encoderMap(_ f: Family) -> String { f.encoderMap(variant(of: f)) }

    /// Something of the family's own is on disk: its components, or one of its DiT maps.
    func isOwn(_ f: Family) -> Bool {
        Self.exists(Installer(library: self).componentsFolder(f)) || f.variants.contains { Self.exists(map(f.ditMap($0))) }
    }

    /// The other families present that read the same encoder map as `f` (Z-Image Standard and FLUX.2
    /// [klein] share the publisher's Qwen3-4B; Z-Image Compact shares nothing).
    func encoderNeighbors(_ f: Family) -> [Family] {
        let mine = encoderMap(f)
        return Family.allCases.filter { $0 != f && isOwn($0) && encoderMap($0) == mine }
    }

    /// Does a family present, other than `f`, read the encoder map `name`? (Z-Image's Standard
    /// encoder while FLUX.2 [klein] is there, whichever version Z-Image is in.)
    func isRead(_ name: String, byOtherThan f: Family) -> Bool {
        Family.allCases.contains { $0 != f && isOwn($0) && encoderMap($0) == name }
    }

    /// What `release(_:keeping:)` would remove: the other versions' DiT, and their encoder unless
    /// `kept` reads it too (the Compact's and the Light's are one map) or another family present
    /// does. The components are the same in every version: they stay.
    func releasable(_ f: Family, keeping kept: Variant) -> [String] {
        var targets: [String] = []
        for v in f.variants where v != kept {
            let dit = map(f.ditMap(v))
            targets += [dit, adapterPath(fromMap: dit)]
            let encoder = map(f.encoderMap(v))
            if f.encoderMap(v) != f.encoderMap(kept), !isRead(f.encoderMap(v), byOtherThan: f), !targets.contains(encoder) {
                targets.append(encoder)
            }
        }
        return targets.filter(Self.exists)
    }

    /// **Removes the versions of `f` other than `kept`** (`releasable`), and returns the bytes given
    /// back. `Installer.install` calls it **once `kept` is forged and complete** — or before its first
    /// byte only when the disk cannot hold both versions at once, which the user was told before
    /// accepting (`InstallPlan.removesFirst`).
    @discardableResult
    package func release(_ f: Family, keeping kept: Variant) throws -> Int {
        var freed = 0
        for c in releasable(f, keeping: kept) {
            freed += Self.size(c)
            try FileManager.default.removeItem(atPath: c)
        }
        return freed
    }

    /// **The version an installation installs**, from what is asked and what is on disk:
    /// - `requested` nil keeps the version installed, Standard if none: a command that names no
    ///   version never changes it;
    /// - `baseDiT` false (what an imported model lacks: encoder, VAE) **never** changes it either: the
    ///   version installed is imposed, Standard if none, and a `requested` that contradicts it is
    ///   refused with the reason — switching versions is the publisher's model's installation.
    package func resolvedVariant(_ f: Family, requested: Variant?, baseDiT: Bool) throws -> Variant {
        let installed = installedVariant(of: f)
        if !baseDiT {
            let imposed = installed ?? .standard
            if let r = requested, r != imposed {
                throw Numerics.Failure(description: "\(f.name): what an imported model needs installs in "
                    + (installed == nil ? "the default version (\(imposed.rawValue))" : "the version installed (\(imposed.rawValue))")
                    + ", not \(r.rawValue) — the version changes only with the publisher's model (an installation with its DiT)")
            }
            return imposed
        }
        let v = requested ?? installed ?? .standard
        guard f.variants.contains(v) else {
            throw Numerics.Failure(description: "\(f.name) has no \(v.rawValue) version (\(f.variants.map(\.rawValue).joined(separator: ", ")))")
        }
        return v
    }

    /// **What an installation will do, before its first byte** — what the app states above « Accept
    /// and Install », and what `install(_:variant:baseDiT:)` follows.
    public struct InstallPlan: Sendable, Equatable {
        /// The version installed (`variant: nil` keeps the one there, Standard if none; `baseDiT`
        /// false always keeps it).
        public let variant: Variant
        /// The other version on disk, replaced: removed **once the new one is complete** — the render
        /// reads it until then — or before the first byte if `removesFirst`.
        public let replacing: Variant?
        /// Bytes on disk the installation adds once done (`Family.installedSize` of the parts
        /// missing; an encoder already there, shared or not, counts for nothing).
        public let added: Int
        /// Bytes the replaced version gives back.
        public let freed: Int
        /// The peak while forging: the parts missing plus the sources of the largest (`spaceNeeded`).
        public let peak: Int
        /// **The disk cannot hold both versions at once**: the replaced one is removed before the
        /// first download, and a stop or a failure then leaves the family without a complete version
        /// until the installation is resumed. To be said before the user accepts.
        public let removesFirst: Bool
        /// The installation fits — the replaced version's space counted when it goes first.
        public let fits: Bool
    }

    /// The plan of `install(_:variant:baseDiT:)` against the free space now. Throws if the version
    /// asked does not exist, or contradicts the one installed when `baseDiT` is false.
    public func installPlan(_ f: Family, variant: Variant? = nil, baseDiT: Bool = true) throws(EngineError) -> InstallPlan {
        try EngineError.boundary { try installPlan(f, variant: variant, baseDiT: baseDiT, free: freeSpace()) }
    }

    /// `free`: the free bytes on the volume (`nil`: the system did not say — everything fits).
    func installPlan(_ f: Family, variant requested: Variant?, baseDiT: Bool, free: Int?) throws -> InstallPlan {
        let v = try resolvedVariant(f, requested: requested, baseDiT: baseDiT)
        let i = Installer(library: self)
        let size = f.installedSize(v)
        var added = 0
        if !i.presentComponents(f) { added += size.components }
        if !Self.exists(map(f.encoderMap(v))) { added += size.encoder }
        if baseDiT && !Self.exists(map(f.ditMap(v))) { added += size.dit }
        let old = releasable(f, keeping: v)
        let freed = old.reduce(0) { $0 + Self.size($1) }
        // The version on disk, else (a switch stopped midway) the one whose DiT is still there.
        let replacing = old.isEmpty ? nil
            : installedVariant(of: f).flatMap { $0 != v ? $0 : nil }
                ?? f.variants.first { $0 != v && Self.exists(map(f.ditMap($0))) }
                ?? f.variants.first { $0 != v }
        let peak = i.spaceNeeded(f, variant: v, baseDiT: baseDiT)
        // With the 2 % margin the exact checks add (`Installer.checkSpace(_:)`): a plan that keeps
        // both by a hair would fail at the first download.
        let margin = peak + peak / 50
        let both = free.map { margin <= $0 } ?? true
        let fits = both || replacing != nil && free.map { margin <= $0 + freed } ?? true
        return InstallPlan(variant: v, replacing: replacing, added: added, freed: freed, peak: peak,
                           removesFirst: replacing != nil && !both && fits, fits: fits)
    }

    /// Throws away what an interrupted installation left in `telechargements/`, and the `.partiel`
    /// a crash left in `store/` — never one being written: a forge (`MapWriter`) and a download
    /// hold theirs locked, and a locked one is left alone (the app also offers this only while
    /// nothing installs, `managementPossible`).
    public func emptyDownloads() throws(EngineError) {
        let t = Installer(library: self).downloads
        if Self.exists(t) { try EngineError.boundary { try FileManager.default.removeItem(atPath: t) } }
        for p in partials() {
            let fd = open(p, O_RDONLY)
            guard fd >= 0 else { continue }
            defer { close(fd) }
            guard flock(fd, LOCK_EX | LOCK_NB) == 0 else { continue }      // being written
            try EngineError.boundary { try FileManager.default.removeItem(atPath: p) }
        }
    }

    /// **The `.partiel` files under `store/`**: a map being forged (`MapWriter`), or one a crash
    /// interrupted; a component being downloaded, or one cut short. They live next to their final
    /// name — `store/`, `store/importes/`, `store/composants/<famille>/` and its subfolders.
    package func partials() -> [String] {
        let store = root.appendingPathComponent("store")
        guard let e = FileManager.default.enumerator(at: store, includingPropertiesForKeys: [.isRegularFileKey]) else { return [] }
        var found: [String] = []
        while let u = e.nextObject() as? URL {
            if u.pathExtension == "partiel" { found.append(u.standardizedFileURL.path) }
        }
        return found.sorted()
    }

    /// Locks a `.partiel` while its writer still needs it; the descriptor is closed by the caller
    /// (`-1` when it could not be opened — the file is then simply unprotected).
    package static func holdPartial(_ path: String) -> Int32 {
        let fd = open(path, O_RDONLY)
        if fd >= 0 { _ = flock(fd, LOCK_EX | LOCK_NB) }
        return fd
    }

    /// The free space on the library's volume (the folder, or its first existing parent).
    public func freeSpace() -> Int? {
        var u = root
        while !Self.exists(u.path), u.path != "/" { u = u.deletingLastPathComponent() }
        let v = try? u.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage.map(Int.init)
    }

    /// **Refuses a job that would not fit** — before having downloaded or written GB.
    package func checkSpace(_ bytes: Int) throws(EngineError) {
        guard let free = freeSpace(), bytes > free else { return }
        throw .diskFull(needed: bytes, available: free)
    }

    static func exists(_ path: String) -> Bool { FileManager.default.fileExists(atPath: path) }

    /// The size of a file, or of a whole folder; 0 if it does not exist.
    static func size(_ path: String) -> Int {
        let fm = FileManager.default
        var folder: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &folder) else { return 0 }
        guard folder.boolValue else { return (try? fm.attributesOfItem(atPath: path)[.size] as? Int) ?? 0 }
        var total = 0
        let e = fm.enumerator(at: URL(fileURLWithPath: path), includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
        while let u = e?.nextObject() as? URL {
            let v = try? u.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            if v?.isRegularFile == true { total += v?.fileSize ?? 0 }
        }
        return total
    }
}

extension ModelImport {
    /// Removes a forged LoRA (the original `.safetensors` was no longer read anyway).
    public static func remove(_ lora: LoRACard) throws(EngineError) {
        try EngineError.boundary { try FileManager.default.removeItem(atPath: lora.path) }
    }
}

extension Family {
    /// **What a family occupies once installed in `variant`**, in bytes — to state the price before
    /// downloading. Standard: measured on the maps (bf16, as published). Compact: the components are
    /// the Standard's; the maps measured as installed: Z-Image's DiT from
    /// unsloth's GGUF, its encoder from Disty0's; Qwen-Image-2.1's DiT from Comfy-Org's int8 convrot,
    /// its encoder from unsloth's. An installation needs more while forging: the sources, about as
    /// large as the map, then thrown away (`spaceNeeded`).
    public func installedSize(_ variant: Variant) -> (dit: Int, encoder: Int, components: Int) {
        if variant == .light {
            // The DiT maps measured as installed; the encoder is the Compact's.
            switch self {
            case .zImage: return (5_039_636_480, installedSize(.compact).encoder, installedSize(.standard).components)
            case .qwenImage21: return (4_167_073_792, installedSize(.compact).encoder, installedSize(.standard).components)
            default: break
            }
        }
        if variant == .compact {
            switch self {
            case .zImage: return (7_247_331_328, 4_317_841_408, installedSize(.standard).components)
            case .qwenImage21: return (7_257_833_472, 9_355_759_616, installedSize(.standard).components)
            default: break
            }
        }
        switch self {
        case .zImage: return (12_333_072_384, 7_845_068_800, 179_122_176)
        case .anima: return (4_184_088_192, 1_193_871_360, 267_694_080)
        case .krea2: return (26_284_982_272, 7_845_068_800, 531_697_664)
        case .klein4b: return (7_752_122_368, 7_845_068_800, 196_112_384)
        case .ernie: return (16_068_853_760, 6_625_787_904, 186_089_472)
        // Measured on the maps of 2026-10-03; the components include the turbo LoRA (1.36 GB).
        case .qwenImage21: return (14_231_355_392, 16_295_034_880, 2_721_627_752)
        }
    }
}
