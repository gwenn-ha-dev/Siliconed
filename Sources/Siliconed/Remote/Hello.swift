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
    /// **The lender's launch** (in its `welcome`): a number drawn once per process, so that a borrower
    /// that finds another one knows the lender restarted — the ordinals it holds (`i7`, `t3`) were numbered
    /// by a process that is gone, and the new one starts again from 1. `nil` where it does not apply.
    package var launch: UInt64?
    /// **What the borrower still holds of that lender** (in its `join`): lets a rejoin skip the images it
    /// already has instead of sending the whole history again. `nil`: nothing, send everything.
    package var held: Held?

    package struct Held: Codable, Equatable, Sendable {
        /// The `launch` the ordinals below were numbered under; the lender trusts them only if it is its own.
        package var launch: UInt64
        /// The `i` ordinals of the history images whose PNG came whole.
        package var entries: [Int]
        package init(launch: UInt64, entries: [Int]) { self.launch = launch; self.entries = entries }
    }

    /// Drawn once per process; `OtherMacs` puts it in the Hello it gives both roles.
    package static let thisLaunch = UInt64.random(in: 1...UInt64.max)

    package init(name: String, machine: Diagnostic.Machine, protocolVersion: Int = Hello.protocolVersion,
                 launch: UInt64? = nil, held: Held? = nil) {
        self.protocolVersion = protocolVersion; self.name = name; self.machine = machine
        self.launch = launch; self.held = held
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
    /// The lender's machine, its map and its diagnostic (`Machine.swift`): at each join, then when one changes.
    case machine(RemoteMachine)
    /// The lender's library, an installation under way, a Hub file judged (`Installing.swift`).
    case libraryNews(RemoteLibraryNews)
    /// An upload offered (`Upload.swift`): taken or refused before its first byte, then verified or failed.
    case uploadAnswer(id: UInt32, UploadAnswer)
    case uploadResult(id: UInt32, UploadResult)

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
    /// « Measure This Mac » of the lender, asked from the borrower's sheet; `stopMeasure` stops it.
    case measure
    case stopMeasure
    /// The lender's « Faster Rendering » (`Spectral.halfSizeStart` is its process's); the new value shows in the catalog.
    case setFasterRendering(Bool)
    /// Install, remove, stop, judge a Hub file: the lender's library is common to its clients (`Installing.swift`).
    case libraryAsk(RemoteLibraryRequest)
    /// A file of the Neo to import on the mini: offered, then streamed (`Upload.swift`) if the answer is yes.
    case uploadOffer(UploadOffer)
    case uploadCancel(UInt32)

    package static func proof(_ key: PairedKey, challenge: Data) -> Data {
        Pairing.proof(key.key, side: "session", transcript: challenge)
    }

    package static func checks(_ proof: Data, _ key: PairedKey, challenge: Data) -> Bool {
        Pairing.checks(proof, key.key, side: "session", transcript: challenge)
    }
}
