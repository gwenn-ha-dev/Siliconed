import Foundation
import Siliconed

/// **`siliconed models | install | verify | uninstall | clean | legacy | loras | import`**: the library from the command line —
/// what is there, the published files that come and go (`Installer`), and the LoRAs it holds (`Library.importLoRA`). The library is the process's (`Library.processWide`):
/// `SILICONED_ROOT`, else the app's.
package enum LibraryCommand {
    static let verbs = ["models", "install", "verify", "uninstall", "clean", "legacy", "loras", "import"]

    static let usage = """
        siliconed models                                           what is installed, and what each version weighs
        siliconed install <family> [--variant compact] [--accept-license]   download the version's published files
        siliconed verify <family> [--variant compact]              hash the installed files against their pinned digests
        siliconed uninstall <family> [--variant standard|compact] [--yes]   remove a version, or the whole family;
                                                                   without --yes, says what would go and removes nothing
        siliconed clean                                            throw away what interrupted downloads left
        siliconed loras                                            the LoRAs the library holds, by family
        siliconed legacy [--free]                                  what version 0.21 left that is no longer read; --free deletes it
        siliconed import <file.safetensors> …                      recognize a model or a LoRA and copy it into the library
        """

    package static func run(_ verb: String, _ arguments: [String]) -> Int32 {
        let library = Library.processWide
        let installer = Installer(library: library)
        var rest = arguments
        func flag(_ name: String) -> Bool {
            guard let i = rest.firstIndex(of: name) else { return false }
            rest.remove(at: i)
            return true
        }
        func value(_ name: String) -> String? {
            guard let i = rest.firstIndex(of: name), i + 1 < rest.count else { return nil }
            defer { rest.removeSubrange(i...i + 1) }
            return rest[i + 1]
        }
        let named = value("--variant")
        let accept = flag("--accept-license")
        // Removing acts only on `--yes`, as `silicontrol uninstall` and the Models sheet's confirmation.
        let confirmed = verb == "uninstall" && flag("--yes")
        if let named, Variant(rawValue: named) == nil { return refuse("unknown variant '\(named)' (standard, compact)") }
        let variant = named.flatMap(Variant.init(rawValue:))

        do {
            switch verb {
            case "models":
                guard rest.isEmpty else { return refuse("`models` takes no argument") }
                print("library \(Library.withoutHome(library.root.path))" + (library.freeSpace().map { " · \(gigabytes($0)) free" } ?? ""))
                for family in Family.allCases { print(sheet(family, installer)) }
                for card in library.importedCards() {
                    let bytes = card.map.flatMap { Library.fileSize(library.map($0)) } ?? 0
                    print("\(card.id) — imported, \(card.family.rawValue) architecture · \(gigabytes(bytes))" + (card.missing(in: library).map { _ in " · not ready" } ?? ""))
                }
                return 0
            case "loras":
                guard rest.isEmpty else { return refuse("`loras` takes no argument") }
                for card in library.loras() {
                    let rank = card.rank == card.smallestRank ? "rank \(card.rank)" : "ranks \(card.smallestRank)–\(card.rank)"
                    print("\(card.target)  \(card.rawName)  ·  \(rank)")
                }
                return 0
            case "import":
                guard !rest.isEmpty else { return refuse("`import` takes one file or more") }
                var failed = false
                for file in rest {
                    do {
                        let done = try ModelImport.importFile(file, in: library)
                        switch done.kind {
                        case .lora: print("✓ \(done.name) — a LoRA for \(done.family.rawValue)")
                        case .model: print("✓ \(done.name) — a model of \(done.family.rawValue): \(done.identifier ?? "")")
                        }
                    } catch {
                        failed = true
                        print("✗ \(error.description)")
                    }
                }
                return failed ? 1 : 0
            case "legacy":
                guard rest.isEmpty || rest == ["--free"] else { return refuse("`legacy` takes --free, or nothing") }
                let files = library.legacyFiles()
                for file in files { print("\(gigabytes(file.bytes))  \(Library.withoutHome(file.path))") }
                if files.isEmpty {
                    print("nothing of version 0.21 is left")
                } else if rest.isEmpty {
                    print("\(files.count) files of version 0.21, \(gigabytes(files.reduce(0) { $0 + $1.bytes })): `siliconed legacy --free` deletes them")
                } else {
                    print("\(gigabytes(try library.releaseLegacy())) freed")
                }
                return 0
            case "clean":
                guard rest.isEmpty else { return refuse("`clean` takes no argument") }
                print("\(gigabytes(try installer.clean())) freed")
                return 0
            default:
                guard let name = rest.first, rest.count == 1, let family = Family(rawValue: name) else {
                    return refuse("`\(verb)` takes one family (\(Family.allCases.map(\.rawValue).joined(separator: ", ")))")
                }
                switch verb {
                case "install":
                    let v = variant ?? .standard
                    let card = family.traits.card
                    if !library.isLicenseAccepted(card) {
                        print("\(card.name) — \(card.license.text)")
                        for page in Installer.licensePages(family, v) { print("  \(page.absoluteString)") }
                        guard accept else {
                            print("Read the license, then install with --accept-license.")
                            return 1
                        }
                        try library.acceptLicense(card)
                    }
                    let plan = installer.plan(family, v)
                    if !plan.files.isEmpty {
                        print("\(plan.files.count) file\(plan.files.count == 1 ? "" : "s"), \(gigabytes(plan.toDownload)) to download"
                              + (plan.resumed > 0 ? " (\(gigabytes(plan.resumed)) already there)" : "")
                              + (plan.free.map { " · \(gigabytes($0)) free" } ?? ""))
                    }
                    try installer.install(family, variant: v)
                    return 0
                case "verify":
                    let damaged = try installer.verify(family, variant ?? library.variant(of: family))
                    return damaged.isEmpty ? 0 : 1
                default:
                    guard confirmed else {
                        print(sheet(family, installer))
                        print("would remove \(variant.map { "the \($0.rawValue) version of " } ?? "")\(family.rawValue): nothing was removed, add --yes to remove it")
                        return 1
                    }
                    print("\(gigabytes(try installer.uninstall(family, variant: variant))) freed")
                    return 0
                }
            }
        } catch {
            FileHandle.standardError.write(Data("✗ \(error.localizedDescription)\n".utf8))
            if let advice = error.recoverySuggestion { FileHandle.standardError.write(Data("  \(advice)\n".utf8)) }
            return 1
        }
    }

    /// A family's lines: its license, then each version — whole, partly there, or absent — and its size.
    static func sheet(_ family: Family, _ installer: Installer) -> String {
        let card = family.traits.card, library = installer.library
        let whole = installer.installed(family)
        var lines = ["\(family.rawValue) — \(card.name) · \(card.license.text)" + (library.isLicenseAccepted(card) ? "" : " (not accepted)")]
        for v in family.variants {
            let files = family.files(v), missing = installer.missing(family, v)
            let bytes = files.reduce(0) { $0 + $1.bytes }
            let state: String
            if missing.isEmpty {
                state = "installed" + (whole.count > 1 && library.variant(of: family) == v ? " · the one rendered" : "")
            } else if missing.count == files.count {
                state = "absent"
            } else {
                state = "\(files.count - missing.count) of \(files.count) files · \(gigabytes(missing.reduce(0) { $0 + $1.bytes })) missing"
            }
            lines.append("  \(v.rawValue.padding(toLength: 9, withPad: " ", startingAt: 0)) \(gigabytes(bytes).padding(toLength: 9, withPad: " ", startingAt: 0))  \(state)")
        }
        return lines.joined(separator: "\n")
    }

    static func gigabytes(_ bytes: Int) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }

    static func refuse(_ why: String) -> Int32 {
        FileHandle.standardError.write(Data("✗ \(why)\n\(usage)\n".utf8))
        return 64
    }
}
