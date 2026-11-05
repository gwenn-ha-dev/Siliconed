// **`AppState`: the render's images, in one place** — each with its role, whatever brought it (the Prompt Lab, a drop,
// « Edit This Image », ⌘V, the remote control). The rules are the library's (`LabInputs`); the state is `AppState.inputs`.
//
//     inputs   ┌ edited ┐ ┌ reference ┐ ┌ neutral ┐ ┌ start ┐     one list, a role each (`LabInputs.Role`)
//              └───┬────┘ └─────┬─────┘ └─────────┘ └───┬───┘
//                  │ references (shown, counted)          │ startImage
//     prompt  "@alice on a quay" ──┐                      │
//                                  ▼                      ▼
//     render()   namedJob: the subjects' pictures after the edit's, the text translated for the model — at the queue
//
// - **One reserve, read twice.** `references` is what the rack shows and counts — the edit's images in their numbers, then
//   the subjects the text cites, as « Carry to the Rack » used to put them in the Images box: the format, the memory need,
//   the plan and the other Mac's quote read it unchanged. What the render reads is made at the queue (`renderReferences`).
// - **The bits of 0.19's rack**: an image is read as its Images box read it — a file as the file (its bytes, for a start,
//   decode the same: `ImageRGB(data:)`), pixels as pixels, a history image as its pixels (with its ordinal, for a mirror)
//   or as its PNG bytes for a start. Same image, same seed, same render.
// - **The text keeps its `@names`**: the translation is the queue's (`AppState.namedJob`), never written back.
// - **Incognito**: the images live in memory; nothing is written.

import AppKit
import CryptoKit
import Siliconed

/// **An image of the render's inputs, with its role.**
struct InputImage: Identifiable {
    let id = UUID()
    /// What the rack shows and counts: the size upright, the vignette, the name, a history image's ordinal.
    var shown: ReferenceImage
    /// The bytes it came as, when it came as bytes: what an img2img starts from (`siliconed-dev render --image` reads a
    /// file's bytes the same way).
    var bytes: Data? = nil
    var role: LabInputs.Role
}

extension AppState {
    // MARK: Reading

    /// **How many images the rack's model reads** (0: none) — `LabInputs`' capacity.
    var imageCapacity: Int { promptNamesImages ? maxReferences : 0 }

    var inputSlots: [LabInputs.Slot<InputImage.ID>] { inputs.map { LabInputs.Slot(id: $0.id, role: $0.role) } }

    /// **What the next render receives**, numbered (`LabInputs.carry`): the edited image, the references, the subjects the
    /// text cites — or why it can not be queued.
    var carried: Result<LabInputs.Carry<InputImage.ID>, LabInputs.Refusal> {
        LabInputs.carry(text: prompt, slots: inputSlots, subjects: subjects.all, capacity: imageCapacity,
                        startsFromImage: startsFromImage)
    }

    /// The edit's images in their numbers, without the subjects: what `namedJob` places the subjects after.
    var editInputs: [InputImage] {
        guard imageCapacity > 0 else { return [] }
        return inputs.filter { $0.role == .edited } + inputs.filter { $0.role == .reference }
    }

    /// **The images the rack shows and counts**: the edit's, then the subjects the text cites (their framed size and
    /// thumbnail, nothing decoded). A refused text counts the edit's alone.
    var references: [ReferenceImage] {
        let edit = editInputs.map(\.shown)
        guard case .success(let c) = carried else { return edit }
        return edit + c.images.compactMap { image in
            guard case .subject(let p) = image else { return nil }
            let size = subjects.framedSize(p.subject, p.picture) ?? (1024, 1024)
            let picture = p.subject.pictures.firstIndex(of: p.picture) ?? 0
            return ReferenceImage(id: Self.citedID(p.subject.id, picture), source: .subject(nil, name: "@" + p.subject.name),
                                  vignette: subjects.thumbnail(p.subject, picture: picture) ?? NSImage(),
                                  name: "@" + p.subject.name, width: size.width, height: size.height)
        }
    }

    /// A cited subject's id, the same at every reading: its row in the rack is not drawn anew at each keystroke.
    private static func citedID(_ subject: String, _ picture: Int) -> UUID {
        var bytes = Array(Insecure.MD5.hash(data: Data("\(subject)/\(picture)".utf8)))
        bytes[6] = bytes[6] & 0x0F | 0x30; bytes[8] = bytes[8] & 0x3F | 0x80
        return UUID(uuid: (bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                           bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]))
    }

    /// **The image the render starts from** (img2img), as the render reads it — `nil` without one. `usedStart` keeps it for
    /// a model that starts from one.
    var startImage: ReferenceImage? {
        inputs.first { $0.role == .start }.map(Self.asStart)
    }

    /// The edit's images as the render reads them — as shown (the subjects come from `namedJob`).
    var renderReferences: [ReferenceImage] { editInputs.map(\.shown) }

    static func asStart(_ input: InputImage) -> ReferenceImage {
        guard let bytes = input.bytes else { return input.shown }
        let s = input.shown
        return ReferenceImage(source: .bytes(bytes), vignette: s.vignette, name: s.name, width: s.width, height: s.height,
                              ordinal: s.ordinal)
    }

    func input(_ id: InputImage.ID) -> InputImage? { inputs.first { $0.id == id } }

    /// The input whose shown image is `id` — the rack's Images box speaks in its references' ids.
    func inputIndex(shown id: ReferenceImage.ID) -> Int? { inputs.firstIndex { $0.shown.id == id } }

    // MARK: Changing

    /// **An image comes in**: the role a newcomer takes (`LabInputs.defaultRole`: the next place under a model that reads
    /// images, neutral elsewhere), unless one is given.
    @discardableResult
    func addInput(_ shown: ReferenceImage, bytes: Data? = nil, role: LabInputs.Role? = nil) -> InputImage.ID {
        let input = InputImage(shown: shown, bytes: bytes, role: role ?? LabInputs.defaultRole(adding: inputSlots, capacity: imageCapacity))
        inputs.append(input)
        if role == .start || role == .edited { setRole(input.role, of: input.id) }
        return input.id
    }

    /// **A role chosen by a gesture** (`LabInputs.assigning`): the edited image and the start are unique. A start takes,
    /// as `carryStart` does, the model's size nearest its proportions.
    func setRole(_ role: LabInputs.Role, of id: InputImage.ID) {
        for slot in LabInputs.assigning(role, to: id, in: inputSlots) {
            if let i = inputs.firstIndex(where: { $0.id == slot.id }), inputs[i].role != slot.role { inputs[i].role = slot.role }
        }
        if role == .start, startsFromImage, let s = input(id)?.shown,
           let f = recommendedFormat(width: s.width, height: s.height), f != format { useFormat(f) }
    }

    /// **What the Prompt Lab holds that the next render does not read** — said beside « Generate », in the lab: its images without a role
    /// or with one the model lacks, and the subjects the text cites to a model that reads no image.
    var unusedInputs: Int {
        let images = LabInputs.unused(inputSlots, capacity: imageCapacity, startsFromImage: startsFromImage)
        guard imageCapacity == 0 else { return images }
        let cited = Set(NamedSubjects.mentions(in: prompt).map { NamedSubjects.key($0.name) })
        return images + subjects.all.filter { cited.contains($0.id) }.count
    }

    /// Whether an image's role does something under the rack's model.
    func isEffective(_ input: InputImage) -> Bool {
        LabInputs.isEffective(input.role, capacity: imageCapacity, startsFromImage: startsFromImage)
    }

    func removeInput(_ id: InputImage.ID) { inputs.removeAll { $0.id == id } }

    /// Another image in the same place, with the same role.
    func replaceInput(_ id: InputImage.ID, with shown: ReferenceImage, bytes: Data? = nil) {
        guard let i = inputs.firstIndex(where: { $0.id == id }) else { return }
        inputs[i] = InputImage(shown: shown, bytes: bytes, role: inputs[i].role)
    }

    /// **After a change of model** (`LabInputs.conforming`): each image keeps its role, without effect where the model lacks it.
    func conformInputs() {
        let kept = LabInputs.conforming(inputSlots, capacity: imageCapacity, startsFromImage: startsFromImage)
        for (i, slot) in kept.enumerated() where inputs[i].role != slot.role { inputs[i].role = slot.role }
    }

    /// **The edit's images set whole** — a command, a series' cell: they take the edited place and the references in
    /// their order, the neutral images and the start staying — but not twice: a neutral image of the same file or history
    /// image as one coming in leaves its place to it.
    func setEditImages(_ images: [ReferenceImage]) {
        inputs.removeAll { $0.role == .edited || $0.role == .reference || ($0.role == .neutral && images.contains(where: $0.shown.same)) }
        let placed = images.enumerated().map { InputImage(shown: $0.element, role: $0.offset == 0 ? .edited : .reference) }
        inputs.insert(contentsOf: placed, at: 0)
    }

    /// **The starting image set** (`nil`: none): it takes the place of the previous one, and of a neutral copy of itself.
    func setStart(_ image: ReferenceImage?, bytes: Data? = nil) {
        inputs.removeAll { $0.role == .start || ($0.role == .neutral && image.map($0.shown.same) == true) }
        if let image { inputs.append(InputImage(shown: image, bytes: bytes, role: .start)) }
    }

}
