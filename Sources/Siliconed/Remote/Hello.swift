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

    package enum Mismatch: Error, Equatable, Sendable {
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
