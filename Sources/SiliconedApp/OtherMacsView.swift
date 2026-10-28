// **Other Macs, in the Settings**: lend this Mac, pair a Mac, the Macs paired each way — and the two
// pairing sheets, one per side, where the same six digits are compared.
//
//     ┌ Other Macs ────────────────────────────────────────────────┐
//     │ Lend This Mac                                        [on]  │
//     │   Neo de Gwenn · connected                       Forget    │
//     │   [Pair a Mac…]                                            │
//     │ Compute on Another Mac                [Pair With a Mac…]   │
//     │   Mac mini de Gwenn · connected      Disconnect  Forget    │
//     └────────────────────────────────────────────────────────────┘
//
// The acceptance is on the lending side only (`Pairing.swift`: the code proves nothing typed on the
// other one); the borrowing sheet shows the code and says where to accept.

import Siliconed
import SwiftUI

struct OtherMacsSection: View {
    @Bindable var macs: OtherMacs
    @State private var hostSheet = false
    @State private var clientSheet = false

    var body: some View {
        Section {
            Toggle(isOn: $macs.lending) {
                titled("Lend This Mac", "Paired Macs render on this one, with its models. Nothing connects before you accept a pairing.")
            }
            if macs.lending {
                ForEach(macs.pairings?.lent ?? [], id: \.identity) { m in
                    LabeledContent {
                        Button("Forget") { macs.forget(lent: m) }
                    } label: {
                        Text(verbatim: m.name)
                        Text(macs.connected[m.identity] != nil ? "connected" : "not connected").font(.caption)
                    }
                }
                Button("Pair a Mac…") {
                    macs.openPairing()
                    hostSheet = true
                }
            }
            if let p = macs.lendingProblem {
                Text(verbatim: p).font(.caption).foregroundStyle(.orange)
            }
            LabeledContent {
                Button("Pair With a Mac…") {
                    macs.lookForPairable()
                    clientSheet = true
                }
            } label: {
                titled("Compute on Another Mac", "Pair once, with a code shown on both screens.")
            }
            ForEach(macs.pairings?.borrowed ?? [], id: \.id) { m in
                LabeledContent {
                    HStack {
                        if macs.links[m.id] == nil {
                            Button("Connect") { macs.connect(m) }
                        } else {
                            Button("Disconnect") { macs.disconnect(m.id) }
                        }
                        Button("Forget") { macs.forget(borrowed: m) }
                    }
                } label: {
                    Text(verbatim: m.name)
                    Text(Self.state(macs.links[m.id], found: macs.found[m.id] != nil)).font(.caption)
                    if let machine = macs.machines[m.id] { Text(verbatim: machine.summary).font(.caption) }
                }
            }
        } header: {
            titled("Other Macs", "One Mac renders for another on the same network.")
        }
        .onAppear { macs.load() }
        .sheet(isPresented: $hostSheet, onDismiss: macs.closePairing) {
            HostPairingSheet(macs: macs) { hostSheet = false }
        }
        .sheet(isPresented: $clientSheet, onDismiss: macs.stopLookingForPairable) {
            ClientPairingSheet(macs: macs) { clientSheet = false }
        }
    }

    static func state(_ s: RemoteLink.State?, found: Bool) -> String {
        switch s {
        case nil, .off?: found ? String(localized: "on the network") : String(localized: "not on the network")
        case .connecting?: String(localized: "connecting…")
        case .connected?: String(localized: "connected")
        case .waiting?: String(localized: "lost, trying again…")
        case .refused(.forgotten)?: String(localized: "refused: that Mac forgot this one; pair again")
        case .refused(.mismatch(let m))?: mismatch(m)
        }
    }

    static func mismatch(_ m: Hello.Mismatch) -> String {
        switch m {
        case .protocolVersion, .appVersion: String(localized: "refused: the two Macs must run the same version of Siliconed")
        }
    }

    private func titled(_ title: LocalizedStringKey, _ text: LocalizedStringKey) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
            Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// The six digits, large, as both screens write them.
private struct Code: View {
    let code: String
    var body: some View {
        Text(verbatim: Pairing.spaced(code)).font(.system(size: 40, weight: .semibold, design: .rounded)).monospacedDigit()
            .textSelection(.disabled)
    }
}

/// **The lending side**: waits for a Mac, shows its name and the code, and only here is it accepted.
private struct HostPairingSheet: View {
    let macs: OtherMacs
    let done: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            switch macs.window {
            case nil, .waiting?:
                ProgressView()
                Text("Waiting for a Mac…").font(.headline)
                Text("On the other Mac: Settings → Other Macs → Pair With a Mac…, then choose “\(macs.name)”.")
                    .multilineTextAlignment(.center)
                Button("Cancel", action: done).keyboardShortcut(.cancelAction)
            case .compare(let client, let code)?:
                Text("Does “\(client)” show this code?").font(.headline)
                Code(code: code)
                Text("Accept only if both screens show the same six digits.").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("Refuse") { macs.refusePairing() }.keyboardShortcut(.cancelAction)
                    // No default key: a Return typed while the other Mac's code comes up must not accept it.
                    Button("Accept") { macs.acceptPairing() }
                }
            case .paired(_, let client)?:
                Text("Paired with “\(client)”.").font(.headline)
                Button("Done", action: done).keyboardShortcut(.defaultAction)
            case .ended(let how)?:
                Text(Self.ended(how)).font(.headline).multilineTextAlignment(.center)
                Button("Close", action: done).keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 380)
    }

    static func ended(_ how: RemoteHost.PairingEnd) -> String {
        switch how {
        case .closed: String(localized: "The other Mac stopped the pairing.")
        case .timedOut: String(localized: "No Mac came in two minutes.")
        case .refused: String(localized: "Pairing refused.")
        case .failed: String(localized: "The pairing failed: nothing was kept.")
        }
    }
}

/// **The borrowing side**: the Macs whose pairing window is open, then the code to compare.
private struct ClientPairingSheet: View {
    let macs: OtherMacs
    let done: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            if let a = macs.attempt {
                switch a.step {
                case .connecting:
                    ProgressView()
                    Text("Connecting to “\(a.with.name)”…").font(.headline)
                case .compare(let code):
                    Text("Check that “\(a.with.name)” shows this code, and accept there.").font(.headline)
                        .multilineTextAlignment(.center)
                    Code(code: code)
                case .paired:
                    Text("Paired with “\(a.with.name)”.").font(.headline)
                    Button("Done", action: done).keyboardShortcut(.defaultAction)
                case .ended(let how):
                    Text(Self.ended(how, a.with.name)).font(.headline).multilineTextAlignment(.center)
                    Button("Close", action: done).keyboardShortcut(.defaultAction)
                }
                if case .paired = a.step {} else if case .ended = a.step {} else {
                    Button("Cancel", action: done).keyboardShortcut(.cancelAction)
                }
            } else {
                Text("Choose the Mac to compute on").font(.headline)
                if macs.pairable.isEmpty {
                    ProgressView()
                    Text("On that Mac: Settings → Other Macs → Lend This Mac, then Pair a Mac…")
                        .multilineTextAlignment(.center).font(.caption).foregroundStyle(.secondary)
                } else {
                    ForEach(macs.pairable) { m in
                        Button { macs.pair(with: m) } label: {
                            Text(verbatim: m.name).frame(maxWidth: .infinity)
                        }
                    }
                }
                Button("Cancel", action: done).keyboardShortcut(.cancelAction)
            }
        }
        .padding(24)
        .frame(width: 380)
    }

    static func ended(_ how: RemotePairingAttempt.PairingEnd, _ name: String) -> String {
        switch how {
        case .refused: String(localized: "“\(name)” refused the pairing.")
        case .failed: String(localized: "The pairing failed: nothing was kept.")
        case .unreachable: String(localized: "“\(name)” is no longer waiting for a Mac: open its pairing window again.")
        case .cancelled: String(localized: "Pairing cancelled.")
        }
    }
}
