// **A new version, said at launch** — or when asked, by Siliconed › Check for Updates….
//
//     ┌───────────────────────────────┐
//     │            [icon]             │
//     │        Siliconed 0.23         │
//     │  [Download]   [Website]       │
//     └───────────────────────────────┘
//
// - **One request, to GitHub** (`NewVersion.releases`), through an ephemeral session: no cookie, no cache, nothing of the
//   user's. At each launch unless Settings › General › Updates says otherwise (`atLaunch`, on by default); the menu asks
//   whatever the switch says.
// - **At launch, only a newer version speaks**: up to date, offline, GitHub unreachable — nothing is shown. Asked from
//   the menu, each outcome is said.
// - **The app does not update itself**: Download opens the release's `.dmg` in the browser, which downloads it; the
//   site has every version and its notes.

import AppKit
import Siliconed
import SwiftUI

@MainActor @Observable
final class Updates {
    static let shared = Updates()

    static let atLaunchKey = "checkForUpdatesAtLaunch"
    /// **Settings › General › Updates**: whether each launch asks GitHub for the latest version. On by default.
    var atLaunch = UserDefaults.standard.object(forKey: Updates.atLaunchKey) as? Bool ?? true {
        didSet { UserDefaults.standard.set(atLaunch, forKey: Self.atLaunchKey) }
    }

    @ObservationIgnored private var panel: NSPanel?

    /// At launch: silent unless a newer version is published. A build outside the app's bundle has no version to
    /// compare (`swift run`): it does not ask.
    func launch() {
        guard atLaunch, Infotext.appVersion != nil else { return }
        Task {
            if case .newer(let v) = await Self.ask() { show(v) }
        }
    }

    /// « Check for Updates… »: each outcome said.
    func check() {
        Task {
            let outcome = await Self.ask()
            if case .newer(let v) = outcome { show(v); return }
            let alert = NSAlert()
            if case .upToDate = outcome {
                alert.messageText = String(localized: "Siliconed is up to date.")
                alert.informativeText = String(localized: "Version \(Infotext.appVersion ?? "?") is the latest published.")
                alert.addButton(withTitle: String(localized: "OK"))
            } else {
                alert.messageText = String(localized: "The check did not succeed.")
                alert.informativeText = String(localized: "GitHub could not be reached. New versions are also listed on the project’s page.")
                alert.addButton(withTitle: String(localized: "Open the Page"))
                alert.addButton(withTitle: String(localized: "Cancel"))
            }
            NSApp.activate()
            if alert.runModal() == .alertFirstButtonReturn, alert.buttons.count == 2 {
                NSWorkspace.shared.open(MainWindow.website)
            }
        }
    }

    private enum Outcome { case newer(NewVersion), upToDate, unreachable }

    private static func ask() async -> Outcome {
        var request = URLRequest(url: NewVersion.releases)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 15
        guard let (data, response) = try? await URLSession(configuration: .ephemeral).data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let latest = NewVersion(releases: data) else { return .unreachable }
        return latest.isNewer(than: Infotext.appVersion) ? .newer(latest) : .upToDate
    }

    /// **The panel**: the icon, the number, the two links. One at a time; shown again, it says the latest.
    private func show(_ version: NewVersion) {
        panel?.close()
        let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 300, height: 260),
                        styleMask: [.titled, .closable, .fullSizeContentView], backing: .buffered, defer: true)
        p.titlebarAppearsTransparent = true
        p.titleVisibility = .hidden
        p.backgroundColor = Palette.columnInk
        p.isReleasedWhenClosed = false
        // Incognito: no frame kept, nothing restored.
        p.isRestorable = false
        p.hidesOnDeactivate = false
        let host = NSHostingController(rootView: NewVersionView(version: version) { [weak p] in p?.close() })
        // Sized once, from what the view needs (it is `.fixedSize()`), like every other window of the app. With
        // `.preferredContentSize` the window's constraint pass asked SwiftUI for its size, which asked for one more
        // pass: on an M1 Pro (macOS 27.2.1) AppKit threw at launch, each launch, as soon as a newer version was out.
        host.sizingOptions = []
        p.contentViewController = host
        p.setContentSize(host.view.fittingSize)
        p.center()
        panel = p
        NSApp.activate()
        p.makeKeyAndOrderFront(nil)
    }
}

/// What the panel shows: nothing but the icon, the version, and where to get it.
private struct NewVersionView: View {
    let version: NewVersion
    let close: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 96, height: 96)
            Text(verbatim: "Siliconed \(version.version)").font(.title2.weight(.semibold))
            HStack(spacing: 10) {
                Button("Download") { open(version.image ?? version.page) }
                    .keyboardShortcut(.defaultAction).tint(Palette.control)
                    .help(version.image == nil ? "Opens the version’s page, with its notes." : "Downloads the disk image, in the browser.")
                Button("Website") { open(MainWindow.website) }
                    .help(Text(verbatim: "gwenn-ha-dev.github.io/Siliconed"))
            }
            .controlSize(.large)
        }
        .padding(.horizontal, 36).padding(.top, 30).padding(.bottom, 24)
        .fixedSize()
        .background(Palette.column)
    }

    private func open(_ url: URL) {
        NSWorkspace.shared.open(url)
        close()
    }
}
