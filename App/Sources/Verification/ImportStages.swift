import Foundation
import Siliconed

// **A model brought by the user renders as its file says** — `siliconed check <model> import`.
//
// No fine-tune is pinned anywhere: the one file whose image is known is the publisher's own. The stage puts it where an
// import would (`store/importes/<family>/`, in a library of its own made of links to the process's), under another
// name, and renders it as the imported model it then is — recognized by its tensors, built from its card, read through
// `MapWeights(dit:)` — beside the published model, same request, same seed.
//
//     recognized   `ModelImport.recognize` says a model of the family
//     the image    above 50 dB against the published model's: the same graph over the same bytes
//
// What it does not prove: a file whose tensors lie under a checkpoint's prefix, or in another arithmetic than the
// installed version's (pure tests read those headers; no such file has been rendered).

enum ImportStages {
    static let all: [Stage] = ModelCard.allCards.map(stage)

    static func stage(_ card: ModelCard) -> Stage {
        Stage(
            model: card.id, name: "import",
            proves: "the installed DiT, placed in the library as an imported model under another name, is recognized as its family's and renders the published model's image",
            sizes: ["512"], oracle: nil, weights: ["text encoder", "DiT", "VAE"],
            rule: "the file is recognized as a model of the family; its image is the published model's, above 50 dB"
        ) { invocation in
            let report = Report(title: "\(card.id) import \(invocation.size)")
            let library = Library.processWide, family = card.family, fm = FileManager.default
            let published = try Model.named(card.id, in: library)
            let root = fm.temporaryDirectory.appendingPathComponent("siliconed-import-\(ProcessInfo.processInfo.processIdentifier)")
            defer { try? fm.removeItem(at: root) }
            let store = root.appendingPathComponent("store"), source = library.root.appendingPathComponent("store")
            try fm.createDirectory(at: store, withIntermediateDirectories: true)
            for name in try fm.contentsOfDirectory(atPath: source.path) where name != "importes" {
                try fm.createSymbolicLink(at: store.appendingPathComponent(name), withDestinationURL: source.appendingPathComponent(name))
            }
            // This Mac's own files (the accepted licenses) stay the process's: the trial library keeps its own, and
            // carries the family's acceptance to the model built on its encoder and VAE, as the app does at an import.
            let trial = Library(root: root)
            let folder = trial.importedFolder(family), file = folder + "/published_again.safetensors"
            try fm.createDirectory(atPath: folder, withIntermediateDirectories: true)
            try fm.createSymbolicLink(atPath: file, withDestinationPath: library.map(library.ditMap(family)))
            let (kind, recognized) = try ModelImport.recognize(file)
            report.add(Judge.holds("recognized", kind == .model && recognized == family, note: "\(kind.rawValue) of \(recognized.rawValue)"))
            let imported = try Model.named("\(family.rawValue)/published_again", in: trial)
            guard library.isLicenseAccepted(card) else { throw Skip("the license of \(card.id) is not accepted in this library") }
            try trial.acceptLicense(imported.card)
            let request = Request(MemoryStages.prompt, width: invocation.width, height: invocation.height)
            let expected = try Engine().render(request, model: published), render = try Engine().render(request, model: imported)
            let (psnr, _, worst, notFinite) = expected.image.pixels.withUnsafeBufferPointer { imageError(render.image.pixels, $0) }
            report.add(Judge.bound("image PSNR against the published model's", psnr, 50, unit: "dB", above: true,
                                   note: notFinite > 0 ? "\(notFinite) non-finite values" : String(format: "worst pixel %.3g", worst)))
            return report
        }
    }
}
