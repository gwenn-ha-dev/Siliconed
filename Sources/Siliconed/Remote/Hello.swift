import Foundation

/// **The first message of a session, each way**: who is there, on which machine, with which version.
/// One version lives: two Macs talk only when their apps are the same build, so that both ends of
/// every message were compiled together. A refusal says which side to update, never "try anyway".

package struct Hello: Codable, Equatable, Sendable {
    /// The shape of the frames and of `PairingMessage`; moves only when they do.
    package static let protocolVersion = 1

    package let protocolVersion: Int
    /// The Mac's name, as the user knows it ("Mac mini de Gwenn").
    package let name: String
    /// What the other side lists before choosing: chip, cores, memory, macOS, and the app's version
    /// (`Machine.version`, short version and build).
    package let machine: Diagnostic.Machine

    package init(name: String, machine: Diagnostic.Machine, protocolVersion: Int = Hello.protocolVersion) {
        self.protocolVersion = protocolVersion; self.name = name; self.machine = machine
    }

    package enum Mismatch: Error, Codable, Equatable, Sendable {
        case protocolVersion(mine: Int, theirs: Int)
        /// The two apps are different builds: the older one is to be updated.
        case appVersion(mine: String, theirs: String)
    }

    /// `nil` when the two can talk.
    package func mismatch(with theirs: Hello) -> Mismatch? {
        if protocolVersion != theirs.protocolVersion {
            return .protocolVersion(mine: protocolVersion, theirs: theirs.protocolVersion)
        }
        if machine.version != theirs.machine.version {
            return .appVersion(mine: machine.version, theirs: theirs.machine.version)
        }
        return nil
    }
}

/// **What a session says**, once the TLS handshake proved the client holds one of the server's keys.
///
///     server: challenge(c) ──▶                       ◀── client: join(hello, identity, HMAC(key, "session" ‖ c))
///     server: welcome(hello) ──▶     or     mismatch ──▶ and close
///
/// **Why a proof inside TLS.** The server's listener holds every paired Mac's key, and its TLS reads
/// back all of them, not the one the handshake used (`sec_protocol_metadata_access_pre_shared_keys`,
/// measured 28/10): it cannot tell which Mac came in. The client proves it here, with the key filed under
/// the identity it names — so one paired Mac cannot pass for another, and forgetting a Mac on the server
/// shuts out that Mac alone. After `welcome`, the mirror's messages.
package enum SessionMessage: Codable, Equatable, Sendable {
    case challenge(Data)
    case join(hello: Hello, identity: Data, proof: Data)
    case welcome(Hello)
    /// The two builds differ: said, then the server closes; the client does not retry.
    case mismatch(Hello.Mismatch)

    // The mirror (`Mirror.swift`): mini → Neo
    case catalog(MirrorCatalog)
    case queue(RemoteQueue)
    /// Its PNG is the blob `entry.png`, which follows.
    case entry(RemoteEntry)
    case removed([Int])
    /// The preview is the blob `id` (a PNG); `nil`: no preview now.
    case preview(UInt32?)
    /// A `render` the mini did not queue, in its words.
    case refused(RemoteRefusal)
    case quoted(RemoteQuoteAnswer)
    case answer(request: Int, RemoteAnswer)
    case ended(RemoteEnded)
    /// Enlarging this image needs Apple's model, not yet on the mini: the Neo asks the user.
    case enlargeNeedsDownload(Int)
    case explored(RemoteExploration?)
    /// One cell's image of the exploration `grid`; its PNG is the blob `entry.png`, which follows.
    case cellImage(grid: Int, cell: Int, RemoteEntry)
    /// What the Statistics window reads for this client (`Statistics.swift`).
    case statistics(RemoteStatistics)
    /// The license this client's waiting job needs, to be accepted on its screen; `nil`: none waits.
    case licensing(RemoteLicense?)
    /// First after `catalog` at each join (see `RemoteSync`).
    case synced(RemoteSync)

    // Neo → mini
    case render(RemoteJob)
    case quote(RemoteQuote)
    case enlarge(Int, download: Bool)
    case crop(Int, Crop.Box)
    case grid(RemoteGridRequest)
    case addSheet(Int)
    case cancel
    case removeJob(Int)
    case stopAll
    case clearQueue
    case deleteImages([Int])
    case answerLicense(RemoteLicenseAnswer)
    /// « Undo » of a deletion: the images (by `i` ordinal) back in the history, in their place.
    case restoreImages([Int])

    package static func proof(_ key: PairedKey, challenge: Data) -> Data {
        Pairing.proof(key.key, side: "session", transcript: challenge)
    }

    package static func checks(_ proof: Data, _ key: PairedKey, challenge: Data) -> Bool {
        Pairing.checks(proof, key.key, side: "session", transcript: challenge)
    }
}
