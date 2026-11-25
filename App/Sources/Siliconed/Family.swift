import Foundation
import SiliCore

/// A family of models. **Everything a family is lives in its folder**; the rest of the app reads only its `traits`.
/// Adding one: its folder, a case here and its line in `traits`, its stages in `Verification`'s registry.
package enum Family: String, CaseIterable, Sendable, Codable {
    case zImage = "z-image"
    case anima
    case krea2
    case mossImage = "moss-image"
    case qwenImage21 = "qwen-image-2.1"

    package var traits: FamilyTraits {
        switch self {
        case .zImage: return .zImage
        case .anima: return .anima
        case .krea2: return .krea2
        case .mossImage: return .mossImage
        case .qwenImage21: return .qwenImage21
        }
    }

    package var name: String { traits.name }

    /// The product's default: the family a request that names none goes to, the first declared.
    package static var `default`: Family { allCases[0] }

    package var variants: [Variant] { Variant.allCases.filter { traits.ditMaps[$0] != nil } }

    /// The DiT's map and the text encoder's, in `store/`.
    package func ditMap(_ v: Variant) -> String { traits.ditMaps[v]! }
    package func encoderMap(_ v: Variant) -> String { traits.encoderMaps[v]! }
}

/// What the app reads of a family, declared in the family's folder.
package struct FamilyTraits: Sendable {
    package let name: String
    package let ditMaps: [Variant: String]
    package let encoderMaps: [Variant: String]
    /// Every published file a version reads, pinned: what an installation downloads (`Installer`).
    package let files: [Variant: [PublishedFile]]
    /// The published model's sheet.
    package let card: ModelCard
    /// What the interface says of the family.
    package let presentation: FamilyPresentation
    /// How the prompt writer writes for the family, and tells an image for it (`PromptWriterInstructions`).
    package let writerStyle: PromptWriterInstructions.Style
    /// The version of the family's page on Civitai that its published DiT is a file of, written in an image's
    /// `parameters` (`Infotext`).
    package var civitaiModelVersion: Int? = nil
    /// The LoRA its Civitai page's recipe runs under, when its few steps are a LoRA's: without the credit, a Remix
    /// there runs the base alone.
    package var civitaiRecipeLoRA: Int? = nil
    /// The multiple each side must be, when coarser than the app's 16 px.
    package var sideMultiple: Int? = nil
    /// The longest side the model renders, when shorter than the app's.
    package var maximumSide: Int? = nil
    /// The strength of a subtle variation (`Variation.Amount`), when the family's schedule asks for less than 0.1.
    package var subtleVariation: Double = 0.1
    /// The fast recipe is a community's distillation, not the publisher's: said beside the model's name until the
    /// publisher releases its own.
    package var isBeta = false
    /// The license in full, when it asks whoever makes the model available for a copy of itself: shown whole where
    /// it is accepted, and part of what a borrower's Mac agrees to (`LicenseDigest`).
    package var agreement: Agreement? = nil
    /// **Its images may be used commercially, though the model may not be** — a license that says yes to its outputs and no
    /// to the model (`License.commercial` says the model's). An imported model of the family inherits it.
    package var commercialImages = false
    /// The strength a starting image comes with (`StartImage.defaultStrength`).
    package var startStrength = 0.65
    /// Its sampler walks every solver (`DenoisingModule.solvers`), not Euler alone: what More Solvers offers.
    package var everySolver = false
    /// **Every module a LoRA may adapt in the family's DiT**, by the name its published file gives the weight (without
    /// `.weight`), with the product's shape: said by the family's DiT. Empty: the family
    /// applies none yet.
    package var loraModules: @Sendable () -> [String: LoRAStack.Shape] = { [:] }
    /// **Every tensor the family's DiT reads**, by the name its published file gives it, with its shape, as the
    /// DiT says it: what a model brought by the user is recognized against (`ModelImport`).
    package var ditTensors: @Sendable () -> [String: [Int]] = { [:] }
    /// **The publisher's undistilled models of the family's architecture**, each file byte for byte: the same names
    /// and shapes as the model the family renders, and an image ruined by its few steps without guidance — refused
    /// at the import (`ModelImport.refuseBase`). Nothing in a header tells them apart; only the bytes do.
    package var baseCheckpoints: [BaseCheckpoint] = []
    /// **What the family's text encoder needs to write prompts** (`PromptWriter`), when it can: its files in
    /// `library`, thrown missing while they are not installed.
    package var writer: (@Sendable (_ library: Library) throws -> PromptWriter.Source)? = nil
    /// The model over a DiT map: the published one, or an imported one's.
    package let model: @Sendable (_ library: Library, _ card: ModelCard, _ dit: String) throws -> Model
    package let render: @Sendable (_ request: Request, _ admitted: Admission, _ seeds: [UInt64], _ model: Model,
                                   _ cancellation: Cancellation?, _ emit: (Engine.Event) -> Void) throws -> [Engine.Render]
    /// **What its render of a request would specialize** (`RenderLoop.preparation`), compiled before the render comes.
    package let prepare: @Sendable (_ request: Request, _ admitted: Admission, _ model: Model) throws -> [Functions.Specialization]
}

/// **What the interface says of a family**, declared in its folder: English sentences, the keys of the app's catalog
/// (`tools/translations.sh` extracts them from the library too). A sentence is looked up in the main bundle: the
/// app's translations; a tool without them prints the English.
package struct FamilyPresentation: Sendable {
    /// The SF Symbol of the family's card.
    package var symbol = "sparkles"
    /// What the model does, in one sentence — said, not sold.
    package let blurb: LocalizedStringResource
    /// What a version changes, where the general sentence is not true of this family.
    package var summaries: [Variant: LocalizedStringResource] = [:]
    /// Whose bytes a version is, in which format — for whoever wants to know.
    package var details: [Variant: LocalizedStringResource] = [:]
    /// What its license adds where it is accepted: a parent license, a duty it puts on whoever deploys the model.
    package var licenseNotes: [LocalizedStringResource] = []
    /// Where a version's files come from, when it is not simply from the publisher or, for a Compact, from third parties.
    package var sources: [Variant: LocalizedStringResource] = [:]
    /// The license's name as the interface says it, when its text (`License.text`, English, for the JSON) does not stay the
    /// same in every language.
    package var licenseName: LocalizedStringResource? = nil
}

/// A license quoted whole: its attribution notice, word for word, and its text.
package struct Agreement: Sendable, Equatable {
    package let notice: String
    package let text: String
    package init(notice: String, text: String) { self.notice = notice; self.text = text }
}

package enum Variant: String, CaseIterable, Sendable, Codable {
    /// The publisher's weights, as published.
    case standard
    /// 8-bit weights as published (int8, a scale a row), the DiT and the text encoder where one is published.
    case compact
}

extension Library {
    /// **The version of a family the render reads**: the one that is complete on disk (its encoder map and its DiT
    /// map). When several or none are, the version the last installation aimed at decides
    /// (`store/composants/<family>/variant`); without it, the version other than the Standard whose DiT map is there.
    package func variant(of f: Family) -> Variant {
        func there(_ name: String) -> Bool { FileManager.default.fileExists(atPath: map(name)) }
        let complete = f.variants.filter { there(f.encoderMap($0)) && there(f.ditMap($0)) }
        let candidates = complete.isEmpty ? f.variants : complete
        if candidates.count == 1 { return candidates[0] }
        let aimed = (try? String(contentsOfFile: intendedVariantPath(f), encoding: .utf8))
            .flatMap { Variant(rawValue: $0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        if let aimed, candidates.contains(aimed) { return aimed }
        return candidates.first { $0 != .standard && there(f.ditMap($0)) } ?? .standard
    }

    /// The version aimed at, written by an installation before its first byte: what breaks the tie when both versions
    /// are whole.
    func intendedVariantPath(_ f: Family) -> String { components(f) + "/variant" }

    /// The DiT map the family renders with, by name in `store/` — that of its installed version.
    package func ditMap(_ f: Family) -> String { f.ditMap(variant(of: f)) }

    /// The encoder map the family reads, by name in `store/` — that of its installed version.
    package func encoderMap(_ f: Family) -> String { f.encoderMap(variant(of: f)) }
}
