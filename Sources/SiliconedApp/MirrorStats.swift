// **The statistics and the license sheet, on the borrowing Mac** (`MirrorServer` publishes them): the
// Statistics window keeps reading `measurements`, `speeds` and the session where it always did — the
// mirror fills them from what the other Mac measured for this Mac's renders; the license sheet is the
// ordinary one, opened by the other Mac's request and answered by a message.

import Foundation
import Siliconed

extension AppState {
    // MARK: - What the other Mac publishes: statistics, licenses

    func receiveStatistics(_ s: RemoteStatistics) {
        guard let link = mirror else { return }
        measurements = s.measures.map { m in
            Measures(remote: m, job: link.jobs[m.job.ordinal] ?? MirrorProjection.job(m.job, ordinal: m.job.ordinal))
        } ?? Measures()
        speeds = s.speeds.mapValues {
            Speed(full: $0.full, reducedAverage: $0.reducedAverage, text: $0.text, image: $0.image, decoding: $0.decoding)
        }
        mirrorEstimates = s.estimates
        sessionImages = s.images; sessionSeconds = s.seconds
    }

    /// The other Mac asks for a license (`nil`: none, or answered elsewhere: the sheet closes).
    func receiveLicensing(_ l: RemoteLicense?) {
        guard let link = mirror else { return }
        guard let l, let card = cards.first(where: { $0.id == l.model }) else {
            link.licenseVariant = nil
            licenseRequest = nil
            return
        }
        link.licenseVariant = Variant(rawValue: l.variant)
        if licenseRequest?.card.id != card.id {
            managementOpen = false
            licenseRequest = LicenseRequest(card: card)
        }
    }

    /// The version whose license words the sheet shows: the other Mac's installed one, in a mirror.
    func licenseVariant(of card: ModelCard) -> Variant {
        if let mirror { return mirror.licenseVariant ?? .standard }
        return library.variant(of: card.family)
    }

    /// The click on the sheet, sent with the digest of the text it showed; the other Mac judges it.
    func answerLicense(_ card: ModelCard, accepted: Bool, via link: MirrorLink) {
        link.send(.answerLicense(RemoteLicenseAnswer(
            model: card.id, accepted: accepted,
            digest: accepted ? LicenseDigest.of(card, variant: licenseVariant(of: card)) : nil)))
        licenseRequest = nil
    }
}
