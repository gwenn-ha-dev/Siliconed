import CryptoKit
import Foundation

/// **Pairing a Mac with the Mac that will compute for it**, once, before any encrypted session: both
/// screens show the same six digits and the user accepts on the server, the Mac whose machine,
/// library and queue will be lent. What comes out is a 32-byte key and an identity, kept by both sides,
/// that the session's TLS uses as its pre-shared key.
///
///     client (the Neo)                                 server (the mini)
///       start: Pa, name ──────────────────────────▶
///                       ◀──────────────────────────── commit: Pb, Cb = H("commit" ‖ Pb ‖ Pa ‖ Nb)
///       nonce: Na ────────────────────────────────▶
///                       ◀──────────────────────────── reveal: Nb        (client checks Cb)
///       code = H("code" ‖ Pa ‖ Pb ‖ Na ‖ Nb) mod 10⁶, on both screens
///                       ◀──────────────────────────── accepted: identity, HMAC(key, "server" ‖ T)
///       confirm: HMAC(key, "client" ‖ T) ─────────▶
///
/// P-256 key agreement, ephemeral on both sides; key = HKDF-SHA256(shared secret, salt = H(T)), T the
/// transcript `Pa ‖ Pb ‖ Na ‖ Nb`. It is Bluetooth's numeric comparison (LE Secure Connections),
/// assembled from CryptoKit — which has no PAKE.
///
/// **What a short code can and cannot prove.** The code is computed from public values: anyone who
/// pairs with the server gets the same digits as the server shows, without seeing its screen. So
/// typing on the client a code the server displays would prove nothing to the server. The proof is
/// the user's eyes on two screens that agree: a man in the middle holds two key exchanges, one per
/// side, and the server commits to its nonce before seeing the client's, so he cannot steer both
/// codes to agree — one chance in 10⁶ per attempt, no offline search. Hence the acceptance on the
/// server, showing the client's name and the code, and one attempt per pairing window.

// MARK: - The messages

package enum PairingMessage: Codable, Equatable, Sendable {
    case start(publicKey: Data, name: String)
    case commit(publicKey: Data, commitment: Data)
    case nonce(Data)
    case reveal(Data)
    /// The server's user accepted: the identity under which the key is filed on both sides.
    case accepted(identity: Data, proof: Data)
    case confirm(proof: Data)
    /// The user declined, the window closed, or something did not check: the pairing ends.
    case refused
}

package enum PairingFailure: Error, Equatable, Sendable {
    /// A message out of order, or for another step.
    case unexpected
    /// A public key that is not a P-256 point.
    case badKey
    /// The revealed nonce does not open the commitment: someone stood in between.
    case commitmentBroken
    /// The proof of the key does not check: the two sides do not hold the same key.
    case badProof
    case refused
}

/// What a finished pairing leaves on each side.
package struct PairedKey: Equatable, Sendable {
    /// The TLS pre-shared key identity, chosen by the server.
    package let identity: Data
    package let key: SymmetricKey

    package static func == (a: PairedKey, b: PairedKey) -> Bool {
        a.identity == b.identity && a.key.withUnsafeBytes { x in b.key.withUnsafeBytes { y in x.elementsEqual(y) } }
    }
}

// MARK: - The shared arithmetic

package enum Pairing {
    package static let nonceSize = 32
    package static let identitySize = 16

    static func digest(_ label: String, _ parts: Data...) -> Data {
        var h = SHA256()
        h.update(data: Data(label.utf8))
        for p in parts { h.update(data: p) }
        return Data(h.finalize())
    }

    static func commitment(server: Data, client: Data, nonce: Data) -> Data {
        digest("siliconed pairing commit", server, client, nonce)
    }

    /// The six digits both screens show. 64 bits reduced modulo 10⁶: the bias is under 10⁻¹³.
    package static func code(client: Data, server: Data, clientNonce: Data, serverNonce: Data) -> String {
        let d = digest("siliconed pairing code", client, server, clientNonce, serverNonce)
        let n = d.prefix(8).reduce(UInt64(0)) { $0 << 8 | UInt64($1) } % 1_000_000
        let s = String(n)
        return String(repeating: "0", count: 6 - s.count) + s
    }

    package static let nameLimit = 64

    /// **A name that crossed, fit to be shown**: the other Mac chose it, and it goes into a sheet and an alert.
    /// Without control characters or line breaks, trimmed, at most `nameLimit` characters.
    package static func displayName(_ raw: String) -> String {
        let clean = String(String.UnicodeScalarView(raw.unicodeScalars.filter { !$0.properties.generalCategory.isControl }))
        let name = String(clean.trimmingCharacters(in: .whitespaces).prefix(nameLimit))
        return name.isEmpty ? "Mac" : name
    }

    /// `482 913`, as the screens write it.
    package static func spaced(_ code: String) -> String {
        "\(code.prefix(3)) \(code.suffix(3))"
    }

    static func key(_ secret: SharedSecret, transcript: Data) -> SymmetricKey {
        secret.hkdfDerivedSymmetricKey(using: SHA256.self, salt: digest("siliconed pairing salt", transcript),
                                       sharedInfo: Data("siliconed pairing key".utf8), outputByteCount: 32)
    }

    static func proof(_ key: SymmetricKey, side: String, transcript: Data) -> Data {
        Data(HMAC<SHA256>.authenticationCode(for: Data(side.utf8) + transcript, using: key))
    }

    /// In constant time.
    static func checks(_ proof: Data, _ key: SymmetricKey, side: String, transcript: Data) -> Bool {
        HMAC<SHA256>.isValidAuthenticationCode(proof, authenticating: Data(side.utf8) + transcript, using: key)
    }

    static func random(_ n: Int) -> Data {
        var g = SystemRandomNumberGenerator()
        return Data((0..<n).map { _ in UInt8.random(in: .min ... .max, using: &g) })
    }

    static func publicKey(_ raw: Data) throws(PairingFailure) -> P256.KeyAgreement.PublicKey {
        do { return try P256.KeyAgreement.PublicKey(x963Representation: raw) } catch { throw .badKey }
    }
}

// MARK: - The client

/// The side that asks to be lent a Mac. It shows the code; the user compares it with the server's.
package struct PairingClient: Sendable {
    private enum Step: Sendable {
        case started, nonceSent(server: Data, commitment: Data)
        case compared(server: Data, serverNonce: Data), done
    }

    private let secret: P256.KeyAgreement.PrivateKey
    private let nonce: Data
    private var step = Step.started
    package private(set) var code: String?

    package init() {
        secret = P256.KeyAgreement.PrivateKey()
        nonce = Pairing.random(Pairing.nonceSize)
    }

    /// For the tests: a fixed key and nonce.
    package init(secret: P256.KeyAgreement.PrivateKey, nonce: Data) {
        self.secret = secret; self.nonce = nonce
    }

    private var publicKey: Data { secret.publicKey.x963Representation }

    package func start(name: String) -> PairingMessage { .start(publicKey: publicKey, name: name) }

    /// The server's next message; returns what to send back, or `nil` when the pairing is done
    /// (the key is then in `key`).
    package mutating func receive(_ m: PairingMessage) throws(PairingFailure) -> PairingMessage? {
        switch (step, m) {
        case (_, .refused):
            step = .done
            throw .refused
        case (.started, .commit(let server, let commitment)):
            _ = try Pairing.publicKey(server)
            step = .nonceSent(server: server, commitment: commitment)
            return .nonce(nonce)
        case (.nonceSent(let server, let commitment), .reveal(let serverNonce)):
            guard Pairing.commitment(server: server, client: publicKey, nonce: serverNonce) == commitment else {
                step = .done
                throw .commitmentBroken
            }
            code = Pairing.code(client: publicKey, server: server, clientNonce: nonce, serverNonce: serverNonce)
            step = .compared(server: server, serverNonce: serverNonce)
            return nil
        case (.compared(let server, let serverNonce), .accepted(let identity, let proof)):
            step = .done
            let transcript = publicKey + server + nonce + serverNonce
            let shared: SharedSecret
            do { shared = try secret.sharedSecretFromKeyAgreement(with: try Pairing.publicKey(server)) } catch { throw .badKey }
            let k = Pairing.key(shared, transcript: transcript)
            guard identity.count == Pairing.identitySize,
                  Pairing.checks(proof, k, side: "server", transcript: transcript) else { throw .badProof }
            key = PairedKey(identity: identity, key: k)
            return .confirm(proof: Pairing.proof(k, side: "client", transcript: transcript))
        default:
            step = .done
            throw .unexpected
        }
    }

    package private(set) var key: PairedKey?
}

// MARK: - The server

/// The side that lends its Mac. It shows the code beside the client's name; only `accept()`, called
/// by the user's click, lets the pairing finish.
package struct PairingServer: Sendable {
    private enum Step: Sendable {
        case waiting, committed(client: Data), compared(client: Data, clientNonce: Data)
        case accepted(transcript: Data), done
    }

    private let secret: P256.KeyAgreement.PrivateKey
    private let nonce: Data
    private var step = Step.waiting
    package private(set) var clientName: String?
    package private(set) var code: String?
    package private(set) var key: PairedKey?
    /// Set once the client proved it holds the same key: the pairing can be filed.
    package private(set) var confirmed = false

    package init() {
        secret = P256.KeyAgreement.PrivateKey()
        nonce = Pairing.random(Pairing.nonceSize)
    }

    package init(secret: P256.KeyAgreement.PrivateKey, nonce: Data) {
        self.secret = secret; self.nonce = nonce
    }

    private var publicKey: Data { secret.publicKey.x963Representation }

    /// The client's next message; returns what to send back. After the nonce, the server shows
    /// `code` and waits for the user: `accept()` or `refuse()`.
    package mutating func receive(_ m: PairingMessage) throws(PairingFailure) -> PairingMessage? {
        switch (step, m) {
        case (_, .refused):
            step = .done
            throw .refused
        case (.waiting, .start(let client, let name)):
            _ = try Pairing.publicKey(client)
            clientName = Pairing.displayName(name)
            step = .committed(client: client)
            return .commit(publicKey: publicKey, commitment: Pairing.commitment(server: publicKey, client: client, nonce: nonce))
        case (.committed(let client), .nonce(let clientNonce)):
            guard clientNonce.count == Pairing.nonceSize else { step = .done; throw .unexpected }
            code = Pairing.code(client: client, server: publicKey, clientNonce: clientNonce, serverNonce: nonce)
            step = .compared(client: client, clientNonce: clientNonce)
            return .reveal(nonce)
        case (.accepted(let transcript), .confirm(let proof)):
            step = .done
            guard let k = key, Pairing.checks(proof, k.key, side: "client", transcript: transcript) else { key = nil; throw .badProof }
            confirmed = true
            return nil
        default:
            step = .done
            throw .unexpected
        }
    }

    /// The user saw the same code on both screens and clicked. Returns the message to send.
    package mutating func accept() throws(PairingFailure) -> PairingMessage {
        guard case .compared(let client, let clientNonce) = step else { throw .unexpected }
        let transcript = client + publicKey + clientNonce + nonce
        let shared: SharedSecret
        do { shared = try secret.sharedSecretFromKeyAgreement(with: try Pairing.publicKey(client)) } catch { throw .badKey }
        let k = Pairing.key(shared, transcript: transcript)
        let identity = Pairing.random(Pairing.identitySize)
        key = PairedKey(identity: identity, key: k)
        step = .accepted(transcript: transcript)
        return .accepted(identity: identity, proof: Pairing.proof(k, side: "server", transcript: transcript))
    }

    package mutating func refuse() -> PairingMessage {
        step = .done
        key = nil
        return .refused
    }
}

private extension Unicode.GeneralCategory {
    /// Controls, and the characters that break lines.
    var isControl: Bool {
        switch self {
        case .control, .lineSeparator, .paragraphSeparator: true
        default: false
        }
    }
}
