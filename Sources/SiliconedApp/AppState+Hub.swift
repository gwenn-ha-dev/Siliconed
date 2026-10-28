// **« Use this model » on Hugging Face**: the link the site's relay page opens (`HubLink`), answered
// by a sheet (`HubSheet.swift`) — the repository read, each file judged on its header while the user
// looks, nothing downloaded before « Import ». A family's own repository opens its installation in
// the Models sheet instead. The download waits in `telechargements/hub/`, then goes through the
// import every dropped file goes through (`perform`), which throws it away.

import Foundation
import Siliconed

extension AppState {
    /// The files judged at most: a GGUF repository lists a dozen quants, a LoRA's dozens of epochs.
    nonisolated static let hubExamined = 24

    struct HubOffer: Identifiable {
        let id = UUID()
        let link: HubLink
        var repository: HubRepository?
        /// The candidates in the order offered, each with its verdict once judged.
        var files: [(file: HubRepository.File, verdict: HubImport.Verdict?)] = []
        var chosen: String?
        var problem: Problem?

        static func importable(_ v: HubImport.Verdict?) -> Bool {
            switch v {
            case .lora, .loraNeedsFamily, .model, .unexamined: true
            case .refused, nil: false
            }
        }
        var choice: (file: HubRepository.File, verdict: HubImport.Verdict?)? { files.first { $0.file.path == chosen } }
    }

    /// **A link arrives** (`Delegate.application(_:open:)`): the sheet opens at once, the Hub answers
    /// behind it.
    func open(_ link: HubLink) {
        if mirror != nil { openRemotely(link); return }
        let offer = HubOffer(link: link)
        hubOffer = offer
        let b = library
        Task.detached(priority: .userInitiated) { [weak self] in
            let r: HubRepository
            do { r = try HubRepository.fetch(link.repository) } catch {
                await MainActor.run { if self?.hubOffer?.id == offer.id { self?.hubOffer?.problem = Problem(error) } }
                return
            }
            let candidates = Array(r.candidates(preferring: link.file).prefix(Self.hubExamined))
            let isInstallation: Bool = await MainActor.run {
                guard let self, self.hubOffer?.id == offer.id else { return true }
                if let f = r.installs {
                    self.hubOffer = nil
                    self.offerInstall(f)
                    return true
                }
                self.hubOffer?.repository = r
                self.hubOffer?.files = candidates.map { ($0, nil) }
                return false
            }
            guard !isInstallation else { return }
            for f in candidates {
                let v: HubImport.Verdict
                do { v = try HubImport.examine(r, f, in: b) } catch { v = .refused(Problem(error).text) }
                let stillOpen: Bool = await MainActor.run {
                    guard let self, self.hubOffer?.id == offer.id, let k = self.hubOffer?.files.firstIndex(where: { $0.file == f }) else { return false }
                    self.hubOffer?.files[k].verdict = v
                    // The link's file if it is importable, else the first that is.
                    if self.hubOffer?.chosen == nil, HubOffer.importable(v), link.file == nil || link.file == f.path
                        || !candidates.contains(where: { $0.path == link.file }) {
                        self.hubOffer?.chosen = f.path
                    }
                    return true
                }
                guard stillOpen else { return }
            }
        }
    }

    /// **« Import »**: the file downloads at the commit read, its sha256 checked, then imports as a
    /// dropped file. A LoRA for a family not installed offers that family first, the import after it.
    func importFromHub() {
        if mirror != nil { importFromHubRemotely(); return }
        guard managementPossible, let offer = hubOffer, let r = offer.repository, let (f, judged) = offer.choice,
              let verdict = judged, HubOffer.importable(verdict) else { return }
        hubOffer = nil
        downloadFromHub(r, f, verdict: verdict)
    }

    /// The download and the import of a file judged — the user's here, or a borrowing Mac's (`MirrorLibraryHost`).
    func downloadFromHub(_ r: HubRepository, _ f: HubRepository.File, verdict: HubImport.Verdict) {
        let destination = HubImport.destination(r, f, in: library)
        let name = (f.path as NSString).lastPathComponent
        launchForge(String(localized: "Downloading \(name)"), task: .downloading(file: name)) { b, cancellation, journal in
            // The file, and its map beside it: never wider than the file (a LoRA in fp32 halves in bf16).
            try b.checkSpace(2 * f.size)
            try HubImport.download(r, f, to: destination, cancellation: cancellation, journal: journal)
        } end: { [weak self] (_: Void) in
            guard let self else { return }
            let url = URL(fileURLWithPath: destination)
            if case .loraNeedsFamily(let family) = verdict, !Installer(library: library).isReady(family) {
                offerInstall(family, thenImport: [url])
            } else {
                perform([url])
            }
        }
    }
}
