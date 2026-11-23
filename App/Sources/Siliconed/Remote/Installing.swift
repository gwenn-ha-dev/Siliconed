import Foundation

/// **The lending Mac's library, shown and driven from the borrowing one** (the Models sheet, in a mirror). The
/// library is the lender's and common to all its clients: what is installed there, the place it takes, an
/// installation under way — published to every client alike —, and the gestures that change it, each judged again on
/// the lender (family known, disk, license digest, nothing else running).
///
///     lender → borrower
///                  library        families (versions whole on disk, the one rendered, bytes, license accepted, the
///                                 plan of each version's installation), imported models, LoRAs, the downloads left,
///                                 free space
///                  job            the installation under way: a task and a phase, structured — the borrower says it in
///                                 its language; `nil`: none
///     borrower → lender
///                  install        a family in a version, with the digest of the license text shown
///                                 (`LicenseDigest`): the lender accepts only its own
///                  cancel, use, uninstall, removeModel, removeLoRA, discard
///
/// **The borrower plans nothing from its own disk**: `Installer.plan` reads the lender's store, so the lender sends each
/// plan (`RemotePlan`) with its family. No byte of weight crosses, and no path of the lender's but a LoRA's — the name
/// the catalog already gives it for a render (`MirrorCatalog.loras`), and the one a removal asks for.
package struct RemoteLibrary: Codable, Equatable, Sendable {
    package struct Part: Codable, Equatable, Sendable {
        package var family: Family
        /// `Library.Occupancy.FamilyPart`: the versions whole on disk, the one rendered, the bytes there.
        package var installed: [Variant]
        package var variant: Variant?
        package var bytes: Int
        /// The family's license is accepted on the lender (`Library.isLicenseAccepted`).
        package var accepted: Bool
        /// What installing each version would do, against the lender's disk now.
        package var plans: [RemotePlan]
        package func plan(_ variant: Variant) -> RemotePlan? { plans.first { $0.variant == variant } }
    }

    /// A model imported on the lender: what the borrower's lists say of it. Its file stays there.
    package struct Imported: Codable, Equatable, Sendable {
        package var id: String
        package var name: String
        package var family: Family
        package var bytes: Int
        package init(id: String, name: String, family: Family, bytes: Int) {
            self.id = id; self.name = name; self.family = family; self.bytes = bytes
        }
    }

    /// A LoRA of the lender's library: what its line in the borrower's Models sheet says. Its file stays there.
    package struct LoRA: Codable, Equatable, Sendable {
        package var path: String
        package var rawName: String
        package var target: String
        package var rank: Int, smallestRank: Int
        package var resolution: String?, trainedOn: String?, trainerName: String?
        package var triggers: [String]
        package var bytes: Int

        package init(_ card: LoRACard, bytes: Int) {
            path = card.path; rawName = card.rawName; target = card.target
            rank = card.rank; smallestRank = card.smallestRank
            resolution = card.resolution; trainedOn = card.trainedOn; trainerName = card.trainerName; triggers = card.triggers
            self.bytes = bytes
        }

        package var card: LoRACard {
            LoRACard(path: path, rawName: rawName, target: target, rank: rank, smallestRank: smallestRank, resolution: resolution,
                     trainedOn: trainedOn, trainerName: trainerName, triggers: triggers)
        }
    }

    package var parts: [Part]
    package var imported: [Imported]
    package var loras: [LoRA]
    package var downloads: Int
    package var free: Int?

    package init(parts: [Part], imported: [Imported] = [], loras: [LoRA] = [], downloads: Int, free: Int?) {
        self.parts = parts; self.imported = imported; self.loras = loras
        self.downloads = downloads; self.free = free
    }

    /// The lender's snapshot of what it holds. Reads the disk for the plans: when the library changes, not at each tick.
    /// `loras`: the library's LoRAs as the lender already read them (`AppState.installedLoRAs`).
    package init(_ o: Library.Occupancy, loras: [(card: LoRACard, bytes: Int)], in library: Library) {
        let installer = Installer(library: library)
        parts = o.families.map { p in
            Part(family: p.family, installed: p.installed, variant: p.variant, bytes: p.bytes,
                 accepted: library.isLicenseAccepted(ModelCard.of(p.family)),
                 plans: p.family.variants.map { RemotePlan(installer.plan(p.family, $0)) })
        }
        imported = o.imported.map { Imported(id: $0.card.id, name: $0.card.name, family: $0.card.family, bytes: $0.bytes) }
        self.loras = loras.map { LoRA($0.card, bytes: $0.bytes) }
        downloads = o.downloads; free = o.free
    }

    package func part(_ f: Family) -> Part? { parts.first { $0.family == f } }

    /// What the Models sheet reads, rebuilt from the lender's word. An imported model's file is the lender's: its
    /// card here is its family's under its own name, and its map says only that it is imported (`ModelCard.isImported`).
    package func occupancy() -> Library.Occupancy {
        Library.Occupancy(
            families: parts.map { .init(family: $0.family, installed: $0.installed, variant: $0.variant, bytes: $0.bytes) },
            imported: imported.map { i in
                let base = ModelCard.of(i.family)
                return .init(card: ModelCard(id: i.id, name: i.name, family: i.family, map: "", license: base.license,
                                             defaultSteps: base.defaultSteps, space: base.space, formats: base.formats),
                             bytes: i.bytes)
            },
            downloads: downloads, free: free)
    }
}

/// `Installer.Plan`, as it crosses — and as the Models sheet reads it on either Mac.
package struct RemotePlan: Codable, Equatable, Sendable {
    package var variant: Variant
    /// The bytes the volume still has to take (`Installer.Plan.needed`); 0: the version is whole.
    package var needed: Int
    /// The bytes still to download (`Installer.Plan.toDownload`): what the sheet announces.
    package var toDownload: Int
    package var fits: Bool
    /// The family's other versions on disk, what their removal gives back, and whether this version fits only in
    /// their place (`Installer.Plan.fitsReplacing`).
    package var others: [Variant]
    package var releasable: Int
    package var fitsReplacing: Bool

    package init(variant: Variant, needed: Int, toDownload: Int, fits: Bool, others: [Variant], releasable: Int, fitsReplacing: Bool) {
        self.variant = variant; self.needed = needed; self.toDownload = toDownload; self.fits = fits
        self.others = others; self.releasable = releasable; self.fitsReplacing = fitsReplacing
    }

    package init(_ p: Installer.Plan) {
        variant = p.variant; needed = p.needed; toDownload = p.toDownload; fits = p.fits
        others = p.others; releasable = p.releasable; fitsReplacing = p.fitsReplacing
    }
}

// MARK: - Progress

/// **An installation under way, or another job that holds the library** (a move), structured: the borrowing Mac says
/// it in its own language and sizes.
package struct RemoteLibraryJob: Codable, Equatable, Sendable {
    /// What was asked, by identifiers — never a sentence.
    package enum Task: Codable, Equatable, Sendable {
        case installing(Family)
        /// Files brought into the library, from its own disk or from a borrowing Mac (`Upload.swift`).
        case importing
        /// A move of the library and the rest.
        case other
    }
    package enum Phase: Codable, Equatable, Sendable {
        case starting
        /// A file coming, among `count`; `received` of `total`: the bytes of the whole installation (`Installer.Progress`).
        case downloading(file: String, index: Int, count: Int, received: Int, total: Int)
    }
    package var task: Task
    package var phase: Phase

    package init(task: Task, progress: Installer.Progress?) {
        self.task = task
        phase = progress.map {
            .downloading(file: ($0.file as NSString).lastPathComponent, index: $0.index, count: $0.count,
                         received: $0.received, total: $0.total)
        } ?? .starting
    }
}

// MARK: - The gestures

/// A family installed from its publisher in a version. `digest`: the license text the borrower's sheet showed
/// (`LicenseDigest.of(card, variant:)`); the lender accepts only its own. `replacing`: the family's other versions go
/// first when the lender's disk takes this one only in their place — the lender's user is asked.
package struct RemoteInstall: Codable, Equatable, Sendable {
    package var family: Family
    package var variant: Variant
    package var replacing: Bool
    package var digest: Data
    package init(family: Family, variant: Variant, replacing: Bool, digest: Data) {
        self.family = family; self.variant = variant; self.replacing = replacing; self.digest = digest
    }
}

/// What a client asks of the library (borrower → lender).
package enum RemoteLibraryRequest: Codable, Equatable, Sendable {
    case install(RemoteInstall)
    /// Stops the installation under way: what was downloaded is kept.
    case cancel
    /// Renders the family with this version, whole on the lender's disk.
    case use(Family, Variant)
    /// A version of a family, or the whole family (`nil`).
    case uninstall(Family, Variant?)
    /// An imported model, by its id (`RemoteLibrary.Imported.id`).
    case removeModel(String)
    /// A LoRA, by its path on the lender (`RemoteLibrary.LoRA.path`).
    case removeLoRA(String)
    /// The interrupted downloads, thrown away.
    case discard

    /// Deletes files on the lender: it asks its user before (`MirrorLibraryHost`), whatever the borrower's own
    /// confirmation. A plain installation stays free. `downloadsLeft`: emptying nothing deletes nothing.
    package func deletes(downloadsLeft: Int) -> Bool {
        switch self {
        case .uninstall, .removeModel, .removeLoRA: true
        case .install(let i): i.replacing
        case .discard: downloadsLeft > 0
        case .cancel, .use: false
        }
    }

    /// **What a removal names, among what the lender holds**: an imported model by its id, a LoRA by a path the
    /// lender listed itself — compared as written, so a path of the borrower's making (outside the LoRA folders, or
    /// led out of them by `..`) names nothing and deletes nothing.
    package static func importedModel(_ id: String, among cards: [ModelCard]) -> ModelCard? {
        cards.first { $0.id == id && $0.isImported }
    }

    package static func lora(_ path: String, among cards: [LoRACard]) -> LoRACard? {
        cards.first { $0.path == path }
    }
}

/// What the lender says of its library (lender → borrower).
package enum RemoteLibraryNews: Codable, Equatable, Sendable {
    case library(RemoteLibrary)
    case job(RemoteLibraryJob?)
    /// To the client alone: its removal waits for a click on this Mac (`true`), or no longer does.
    case confirming(Bool)
}
