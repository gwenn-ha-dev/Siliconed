import Foundation
import Network
import Security

/// **How two Macs reach each other**: two Bonjour services on the local network, each a TCP listener of
/// the Mac that lends itself (the mini).
///
///     _siliconed-pair._tcp   plain TCP, open only while the mini's pairing window is: one attempt.
///                            Plain is enough: the pairing defends itself (`Pairing.swift`), and it
///                            carries nothing but public keys, nonces and the client's name.
///     _siliconed._tcp        TLS 1.2 with a pre-shared key, one per paired Mac, chosen by its
///                            identity: the sessions. TXT: the mini's `id` (stable), app `version`.
///
/// **What was measured on macOS 27 (28/10)**, and decides the shape here:
/// - TLS 1.3 with an external PSK fails (-9858); TLS 1.2 works. Hence no QUIC.
/// - The suite is forced to `ECDHE_PSK_WITH_CHACHA20_POLY1305_SHA256` (0xCCAC): the ephemeral ECDH
///   gives forward secrecy — a key stolen later does not open a recorded session. `tls_ciphersuite_t`
///   has no name for it; its raw value is accepted.
/// - One listener holds every paired Mac's key (`add_pre_shared_key` once per key): the server picks
///   by the identity the client sends. A wrong key under a known identity fails the handshake with
///   « bad MAC » (-9820 on the client, -9846 on the server); an unknown identity with -9864.
/// - **The identity goes as hex.** Sent raw, an identity holding a zero byte fails every handshake with
///   « unknown PSK identity » (-9864): somewhere it is read as a C string. Sixteen random bytes hold one
///   6 % of the time — 400 failures in 400 with a zero, none in 1,200 in hex (28/10).
/// - TCP keepalive (5 s idle, 2 s × 3 probes) is what notices a Mac gone from the network without a
///   word — a lid closed, Wi-Fi lost: about 11 s.
package enum RemoteTransport {
    package static let sessionService = "_siliconed._tcp"
    package static let pairingService = "_siliconed-pair._tcp"

    static let suite = tls_ciphersuite_t(rawValue: 0xCCAC)!

    /// The listener's side: every key it accepts.
    package static func server(keys: [PairedKey]) -> NWParameters {
        tls(keys: keys, hint: nil)
    }

    /// The client's side: its one key, sent under its identity.
    package static func client(key: PairedKey) -> NWParameters {
        tls(keys: [key], hint: key.identity)
    }

    /// Plain TCP, for the pairing alone.
    package static func pairing() -> NWParameters {
        let p = NWParameters(tls: nil, tcp: tcp())
        p.includePeerToPeer = false
        return p
    }

    private static func tls(keys: [PairedKey], hint: Data?) -> NWParameters {
        let tls = NWProtocolTLS.Options()
        let o = tls.securityProtocolOptions
        for k in keys {
            let secret = k.key.withUnsafeBytes { Data($0) }
            sec_protocol_options_add_pre_shared_key(o, dispatch(secret), dispatch(hex(k.identity)))
        }
        if let hint { sec_protocol_options_set_tls_pre_shared_key_identity_hint(o, dispatch(hex(hint))) }
        sec_protocol_options_set_min_tls_protocol_version(o, .TLSv12)
        sec_protocol_options_set_max_tls_protocol_version(o, .TLSv12)
        sec_protocol_options_append_tls_ciphersuite(o, suite)
        let p = NWParameters(tls: tls, tcp: tcp())
        p.includePeerToPeer = false
        return p
    }

    private static func tcp() -> NWProtocolTCP.Options {
        let t = NWProtocolTCP.Options()
        t.enableKeepalive = true
        t.keepaliveIdle = 5
        t.keepaliveInterval = 2
        t.keepaliveCount = 3
        t.noDelay = true
        return t
    }

    private static func hex(_ d: Data) -> Data {
        Data(d.map { String(format: "%02x", $0) }.joined().utf8)
    }

    private static func dispatch(_ d: Data) -> __DispatchData {
        d.withUnsafeBytes { DispatchData(bytes: $0) } as __DispatchData
    }

    /// The handshake said no: a key that is not the other side's (it forgot this Mac, or another
    /// pairing replaced it). Told apart from a network that failed, which is retried.
    package static func isRefusal(_ e: NWError) -> Bool {
        guard case .tls(let status) = e else { return false }
        return [-9820, -9846, -9864, errSSLPeerHandshakeFail, errSSLPeerAccessDenied].contains(status)
    }
}
