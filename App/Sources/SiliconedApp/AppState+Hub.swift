// **« Use this model » on Hugging Face**: the link the site's relay page opens (`HubLink`), answered by a sheet
// (`HubSheet.swift`) — the repository read, each file judged on its header while the user looks, nothing downloaded
// before « Import ». A family's own repository opens its installation in the Models sheet instead. The download
// waits in the library's `televersements`, then goes through the import every dropped file goes through (`perform`),
// which throws it away.

import Foundation
import Siliconed

extension AppState {
    /// The files judged at most: a LoRA's repository lists dozens of epochs.
    nonisolated static let hubExamined = 24

    struct HubOffer: Identifiable {
        let id = UUID()
        let link: HubLink
        var repository: HubRepository?
        /// The candidates in the order offered, each with its verdict once judged.
        var files: [(file: HubRepository.File, verdict: HubImport.Verdict?)] = []
        var chosen: String?
        var problem: Problem?

        var choice: (file: HubRepository.File, verdict: HubImport.Verdict?)? { files.first { $0.file.path == chosen } }
        /// A row that can be chosen: one « Import » downloads, or a model whose family « Install » offers first.
        static func choosable(_ v: HubImport.Verdict?) -> Bool {
            if case .modelNeedsFamily = v { return true }
            return v?.importable ?? false
        }
    }

    /// **A link arrives** (`Delegate.application(_:open:)`): the sheet opens at once, the Hub answers behind it.
    /// On a Mac that computes on another, the library is that Mac's: the link is refused, said why.
    func open(_ link: HubLink) {
        if let mirror {
            problem = Problem(String(localized: "A Hugging Face link imports into this Mac’s library, and the library in use is that of “\(mirror.macName)”."),
                              suggestion: String(localized: "Open the link on that Mac, or choose Render › Compute On › This Mac first."))
            return
        }
        let offer = HubOffer(link: link)
        hubOffer = offer
        let installed = Set(Family.allCases.filter { !Installer(library: library).installed($0).isEmpty })
        Task.detached(priority: .userInitiated) { [weak self] in
            let r: HubRepository
            do { r = try await offThePool { try HubRepository.fetch(link.repository) } } catch {
                await MainActor.run { if self?.hubOffer?.id == offer.id { self?.hubOffer?.problem = Problem(error) } }
                return
            }
            // A gated repository serves its files only under an account, and the app has none (`Downloader`): none
            // is offered, the sheet says how to bring one in.
            let candidates = r.gated ? [] : Array(r.candidates(preferring: link.file).prefix(Self.hubExamined))
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
                do { v = try await offThePool { try HubImport.examine(r, f, installed: installed) } } catch { v = .refused(Problem(error).text) }
                let stillOpen: Bool = await MainActor.run {
                    guard let self, self.hubOffer?.id == offer.id, let k = self.hubOffer?.files.firstIndex(where: { $0.file == f }) else { return false }
                    self.hubOffer?.files[k].verdict = v
                    // The link's file if it can be chosen, else the first that can.
                    if self.hubOffer?.chosen == nil, HubOffer.choosable(v), link.file == nil || link.file == f.path
                        || !candidates.contains(where: { $0.path == link.file }) {
                        self.hubOffer?.chosen = f.path
                    }
                    return true
                }
                guard stillOpen else { return }
            }
        }
    }

    /// **« Import »**: the file downloads at the commit read, its sha256 checked, then imports as a dropped file.
    /// A model whose family is not installed offers that family instead: nothing downloads for a file the import
    /// would refuse, and the link is opened again once the family is there.
    func importFromHub() {
        guard managementPossible, let offer = hubOffer, let r = offer.repository, let (f, judged) = offer.choice, let verdict = judged else { return }
        if case .modelNeedsFamily(let family) = verdict {
            hubOffer = nil
            offerInstall(family)
            return
        }
        guard verdict.importable else { return }
        hubOffer = nil
        let name = (f.path as NSString).lastPathComponent
        launchLibraryJob(String(localized: "Downloading \(name)"), task: .importing) { b, cancellation, journal, progress in
            try HubImport.download(r, f, in: b, cancellation: cancellation, received: { n in
                progress(Installer.Progress(file: name, index: 1, count: 1, received: n, total: f.size))
            }, journal: journal)
        } end: { [weak self] (path: String) in
            // Straight into the import, never through the gesture's checks: the download held the library, no render ran
            // meanwhile, and a job queued during it waits for this one too. A refusal there would leave the file in the
            // uploads; the import's job removes it whatever its outcome.
            self?.importFiles([URL(fileURLWithPath: path)])
        }
    }
}
