import Foundation

/// **The lending Mac's library, shown and driven from the borrowing one** (the Models sheet and the Hub
/// sheet, in a mirror). The library is the lender's and common to all its clients: what is installed
/// there, the place it takes, an installation under way — published to every client alike —, and the
/// gestures that change it, each judged again on the lender (family known, disk, license digest, nothing
/// else running).
///
///     mini → Neo   library        families (parts on disk, version, license accepted, the plans of each
///                                 installation), imported models, LoRAs, the downloads left, free space
///                  forge          the installation or import under way: a task and a phase, structured —
///                                 the Neo says it in its language (`ForgeProgress`); `nil`: none
///                  hubVerdict     what an import of a file of a Hub repository would make of it
///     Neo → mini   install        a family in a version, with the digest of the license text shown
///                                 (`LicenseDigest`): the lender accepts only its own
///                  cancel, uninstall, removeModel, removeLoRA, discard
///                  hubJudge, hubInstall   a Hub file, by repository and path: judged, downloaded and
///                                 imported on the lender, whose maps alone say what a LoRA fits
///
/// **The Neo plans nothing from its own disk**: `Library.installPlan` reads the lender's maps, so the
/// lender sends each plan (`RemotePlan`) with its family. No byte of weight crosses; no path of the
/// lender's but a LoRA's, which the catalog already names.
package struct RemoteLibrary: Codable, Equatable, Sendable {
    package struct Part: Codable, Equatable, Sendable {
        package var family: Family
        package var components: Int, encoder: Int, dit: Int
        /// The version on disk (`FamilyPart.variant`).
        package var variant: Variant?
        package var isReady: Bool
        /// The family's license is accepted on the lender (`Library.isLicenseAccepted`).
        package var accepted: Bool
        /// `Library.installedVariant(of:)`.
        package var installed: Variant?
        /// Compact, or Light on a lender of 8 GB: the lender's memory decides, not the reader's.
        package var preselected: Variant
        /// What each installation would do, against the lender's disk now: every version with its DiT,
        /// and what is kept of the installed version without it.
        package var plans: [RemotePlan]
        package func plan(_ variant: Variant, baseDiT: Bool) -> RemotePlan? {
            plans.first { $0.baseDiT == baseDiT && (baseDiT ? $0.variant == variant : true) }
        }
    }
    package struct Imported: Codable, Equatable, Sendable {
        package var id: String, name: String
        package var family: Family
        package var bytes: Int
    }
    package struct LoRA: Codable, Equatable, Sendable {
        /// On the lender: what the catalog and a removal name it by.
        package var path: String
        package var rawName: String, target: String
        package var rank: Int, smallestRank: Int
        package var bytes: Int
    }

    package var parts: [Part]
    package var imported: [Imported]
    package var loras: [LoRA]
    package var downloads: Int, cache: Int
    package var free: Int?

    package init(parts: [Part], imported: [Imported], loras: [LoRA], downloads: Int, cache: Int, free: Int?) {
        self.parts = parts; self.imported = imported; self.loras = loras
        self.downloads = downloads; self.cache = cache; self.free = free
    }

    /// The lender's snapshot of what it holds. Reads the disk for the plans: when the library changes, not at each tick.
    package init(_ o: Library.Occupancy, in library: Library) {
        parts = o.families.map { p in
            let f = p.family
            var plans = f.variants.compactMap { v in (try? library.installPlan(f, variant: v, baseDiT: true)).map { RemotePlan($0, baseDiT: true) } }
            if let kept = try? library.installPlan(f, variant: nil, baseDiT: false) { plans.append(RemotePlan(kept, baseDiT: false)) }
            return Part(family: f, components: p.components, encoder: p.encoder, dit: p.dit, variant: p.variant, isReady: p.isReady,
                        accepted: library.isLicenseAccepted(ModelCard.of(f)), installed: library.installedVariant(of: f),
                        preselected: f.preselectedVariant(), plans: plans)
        }
        imported = o.imported.map { Imported(id: $0.card.id, name: $0.card.name, family: $0.card.family, bytes: $0.bytes) }
        loras = o.loras.map {
            LoRA(path: $0.card.path, rawName: $0.card.rawName, target: $0.card.target, rank: $0.card.rank,
                 smallestRank: $0.card.smallestRank, bytes: $0.bytes)
        }
        downloads = o.downloads; cache = o.cache; free = o.free
    }

    package func part(_ f: Family) -> Part? { parts.first { $0.family == f } }

    /// What the Models sheet reads, rebuilt from the lender's word. An imported model's map is the
    /// lender's: only that it is imported is kept (`ModelCard.isImported`).
    package func occupancy() -> Library.Occupancy {
        Library.Occupancy.assembled(
            families: parts.map { Library.Occupancy.FamilyPart.assembled(family: $0.family, components: $0.components, encoder: $0.encoder, dit: $0.dit,
                                                               variant: $0.variant, isReady: $0.isReady) },
            imported: imported.map {
                (ModelCard(isImported: $0.name, slug: String($0.id.drop { $0 != "/" }.dropFirst()), family: $0.family, map: ""), $0.bytes)
            },
            loras: loras.compactMap { l in
                LoRACard(header: ["kind": "lora", "cible": ["modele": l.target], "nom": l.rawName,
                                  "rang": l.rank == l.smallestRank ? l.rank as Any : [l.smallestRank, l.rank] as Any], path: l.path)
                    .map { ($0, l.bytes) }
            },
            downloads: downloads, cache: cache, free: free)
    }
}

// The memberwise initializers are the module's own: the app builds a mirror's occupancy through these.
extension Library.Occupancy {
    package static func assembled(families: [FamilyPart], imported: [(card: ModelCard, bytes: Int)],
                                  loras: [(card: LoRACard, bytes: Int)], downloads: Int, cache: Int, free: Int?) -> Self {
        Self(families: families, imported: imported, loras: loras, downloads: downloads, cache: cache, free: free)
    }
}

extension Library.Occupancy.FamilyPart {
    package static func assembled(family: Family, components: Int, encoder: Int, dit: Int, variant: Variant?, isReady: Bool) -> Self {
        Self(family: family, components: components, encoder: encoder, dit: dit, variant: variant, isReady: isReady)
    }
}

/// `Library.InstallPlan`, as it crosses.
package struct RemotePlan: Codable, Equatable, Sendable {
    package var variant: Variant
    /// `false`: what an imported model of the family lacks (encoder, VAE), never its DiT.
    package var baseDiT: Bool
    package var replacing: Variant?
    package var added: Int, freed: Int, peak: Int
    package var removesFirst: Bool, fits: Bool

    package init(variant: Variant, baseDiT: Bool, replacing: Variant?, added: Int, freed: Int, peak: Int, removesFirst: Bool, fits: Bool) {
        self.variant = variant; self.baseDiT = baseDiT; self.replacing = replacing
        self.added = added; self.freed = freed; self.peak = peak; self.removesFirst = removesFirst; self.fits = fits
    }

    /// **The installation takes a version off the disk**, once the new one is ready or before its first byte:
    /// the lender's user is asked, as for any removal (`RemoteLibraryRequest.deletes`).
    package var deletes: Bool { replacing != nil || removesFirst }

    package init(_ p: Library.InstallPlan, baseDiT: Bool) {
        variant = p.variant; self.baseDiT = baseDiT; replacing = p.replacing
        added = p.added; freed = p.freed; peak = p.peak; removesFirst = p.removesFirst; fits = p.fits
    }
}

// MARK: - Progress

/// **An installation or an import under way**, structured: the engine's journal is a terminal's rows
/// (`↓ file (1.23 GB)`, `  … 40 %`, `forging the DiT → name`); read here once, on the lender, into what
/// they mean, so that the borrowing Mac says it in its own language and sizes.
package struct RemoteForge: Codable, Equatable, Sendable {
    /// What was asked, by identifiers and file names — never a sentence.
    package enum Task: Codable, Equatable, Sendable {
        case installing(Family)
        case downloading(file: String)
        case importing(files: [String])
        /// A move of the library and the rest.
        case other
    }
    package enum Phase: Codable, Equatable, Sendable {
        case starting
        /// A file coming; `preparing` when a map is forged beside it (the DiT downloads while the encoder is forged).
        /// `bytes`: received so far, when the percent and the size are known; `total` the file's size.
        case downloading(file: String, percent: Int?, bytes: Int?, total: Int?, preparing: String?)
        /// A map forged (or an import converted: `converted` percent).
        case preparing(name: String, converted: Int?)
        case installed
    }
    package var task: Task
    package var phase: Phase

    package init(task: Task, phase: Phase) { self.task = task; self.phase = phase }
    package init(task: Task, journal: [String]) { self.init(task: task, phase: Phase(journal: journal)) }
}

extension RemoteForge.Phase {
    /// The journal read the way the lender's own sidebar reads it (`ForgeProgress.line`): two steps may
    /// run at once, and a `… %` after a `↓` is the download's.
    package init(journal: [String]) {
        var phase = Self.starting
        var downloading: String?, received: Int?, total: Int?, preparing: String?, converted: Int?
        for row in journal {
            let t = row.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("⇢ ") { continue }
            if t.hasPrefix("↓ ") {
                let rest = t.dropFirst(2)
                let label = String(rest.split(separator: " (").first ?? "")
                downloading = (label as NSString).lastPathComponent
                received = nil
                total = Self.size(in: String(rest))
            } else if t.hasPrefix("… "), let n = Int(t.dropFirst(2).prefix { $0.isNumber }) {
                if downloading != nil { received = n } else { converted = n }
            } else if t.hasPrefix("✓ "), t.hasSuffix(" installed") {
                phase = .installed
                continue
            } else if t.hasPrefix("✓ forged ") {
                (preparing, converted) = (nil, nil)
            } else if t.hasPrefix("✓ ") {
                (downloading, received, total) = (nil, nil, nil)
            } else if t.hasPrefix("forging "), let arrow = t.range(of: "→ ") {
                (preparing, converted) = (String(t[arrow.upperBound...]), nil)
            } else {
                continue
            }
            if let file = downloading {
                phase = .downloading(file: file, percent: received, bytes: received.flatMap { p in total.map { $0 / 100 * p } },
                                     total: total, preparing: preparing)
            } else if let name = preparing {
                phase = .preparing(name: name, converted: converted)
            }
        }
        self = phase
    }

    /// `(1.23 GB)` in a `↓` row, in bytes.
    static func size(in row: String) -> Int? {
        guard let open = row.range(of: " ("), let close = row.range(of: " GB)", range: open.upperBound..<row.endIndex),
              let gb = Double(row[open.upperBound..<close.lowerBound]) else { return nil }
        return Int(gb * 1e9)
    }
}

// MARK: - The gestures

/// A family installed from its publisher in a version. `digest`: the license text the Neo's sheet showed
/// (`LicenseDigest.of(card, variant:)`); the lender accepts only its own.
/// `thenHub`: a LoRA of the Hub that waited for this family (as `InstallOffer.thenImport` does here): the
/// lender imports it itself once the family is installed — it holds the file's name, the Neo need not
/// stay to click again, and a failed or stopped installation drops it with the installation.
package struct RemoteInstall: Codable, Equatable, Sendable {
    package var family: Family
    package var variant: Variant
    package var baseDiT: Bool
    package var digest: Data
    package var thenHub: RemoteHubFile?
    package init(family: Family, variant: Variant, baseDiT: Bool, digest: Data, thenHub: RemoteHubFile? = nil) {
        self.family = family; self.variant = variant; self.baseDiT = baseDiT; self.digest = digest; self.thenHub = thenHub
    }
}

/// A file of a Hugging Face repository. Only its names cross: the lender reads the Hub itself, at the
/// commit it finds, and checks the sha256 the Hub lists.
package struct RemoteHubFile: Codable, Equatable, Hashable, Sendable {
    package var repository: String
    package var path: String
    package init(repository: String, path: String) { self.repository = repository; self.path = path }
}

package enum RemoteHubVerdict: Codable, Equatable, Sendable {
    case lora(Family), model(Family), loraNeedsFamily(Family), unexamined
    case refused(RemoteRefusal)

    package init(_ v: HubImport.Verdict) {
        switch v {
        case .lora(let f): self = .lora(f)
        case .model(let f): self = .model(f)
        case .loraNeedsFamily(let f): self = .loraNeedsFamily(f)
        case .unexamined: self = .unexamined
        // The forge's reason, in the lender's words: it has no code.
        case .refused(let why): self = .refused(.text(why))
        }
    }
}

/// What a client asks of the library (Neo → mini).
package enum RemoteLibraryRequest: Codable, Equatable, Sendable {
    case install(RemoteInstall)
    /// Stops the installation or import under way: the files downloaded are kept.
    case cancel
    case uninstall(Family)
    /// An imported model, by its identifier.
    case removeModel(String)
    /// A forged LoRA, by its path on the lender (the one the catalog named).
    case removeLoRA(String)
    /// The interrupted downloads, thrown away.
    case discard
    case hubJudge(RemoteHubFile)
    case hubInstall(RemoteHubFile)

    /// Deletes files on the lender: it asks its user before (`MirrorLibraryHost`), whatever the borrower's
    /// own confirmation. Installing stays free. `downloadsLeft`: emptying nothing deletes nothing.
    package func deletes(downloadsLeft: Int) -> Bool {
        switch self {
        case .uninstall, .removeModel, .removeLoRA: true
        case .discard: downloadsLeft > 0
        case .install, .cancel, .hubJudge, .hubInstall: false
        }
    }
}

/// What the lender says of its library (mini → Neo).
package enum RemoteLibraryNews: Codable, Equatable, Sendable {
    case library(RemoteLibrary)
    case forge(RemoteForge?)
    case hubVerdict(RemoteHubFile, RemoteHubVerdict)
    /// To the client alone: its removal waits for a click on this Mac (`true`), or no longer does.
    case confirming(Bool)
}
