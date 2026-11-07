import Foundation

// **The sizes of a map's streams, from its header alone.** What `TailStream` stages and what the memory need charges are the same
// functions of the table (`header["tensors"]`: offsets, bytes, page), written once here and called by both; nothing in this file
// reads a byte of the map.

// ── sizing: a streamed block's staging buffer, from the map's table ───────────────────────────

extension Artifact {
    /// **The run a streamed block occupies in the file** — from its first tensor's offset to its last
    /// one's end rounded up to the page, never past the file: what `TailStream` reads into a staging
    /// buffer, and what that buffer holds. `nil` for no tensor.
    package static func stagingRun(_ tensors: [(offset: Int, bytes: Int)], page: Int,
                                   fileSize: Int) -> (start: Int, end: Int)? {
        guard let start = tensors.map(\.offset).min(),
              let last = tensors.map({ $0.offset + $0.bytes }).max() else { return nil }
        return (start, min((last + page - 1) / page * page, fileSize))
    }

    /// **A stream's staging buffer, from a map's header alone** — the largest block's run
    /// (`stagingRun`), each block being the tensors whose name starts with its prefix, as
    /// `Prefetcher.namesOfBlock` gathers them. This is where the map's type shows: a bf16 layer of
    /// Z-Image runs 362 MB, an 8-bit one about half. 0 for a header without a tensor table.
    package static func stagingBytes(header: [String: Any], prefixes: [String], fileSize: Int) -> Int {
        stagingSpans(header: header, prefixes: prefixes, fileSize: fileSize).compactMap { $0 }.max() ?? 0
    }

    /// **Each layer's bytes in the map**, from the header alone: the run `stagingBytes` takes the largest of, one per prefix (0 for a layer
    /// without a tensor). What the stream reads of a layer at each evaluation, and what `MemoryPlan.released` sizes the head with.
    package static func layerSpans(header: [String: Any], prefixes: [String], fileSize: Int) -> [Int] {
        stagingSpans(header: header, prefixes: prefixes, fileSize: fileSize).map { $0 ?? 0 }
    }

    /// The bytes of every tensor of the map, from the header alone (what `MemoryPlan.released` compares with B).
    package static func mapBytes(header: [String: Any]) -> Int {
        (header["tensors"] as? [String: [String: Any]])?.values.reduce(0) { $0 + ($1["bytes"] as? Int ?? 0) } ?? 0
    }

    /// Each block's run alone (`nil`: no tensor, or no tensor table) — the header's table read once for
    /// all of them; `stagingBytes` is their largest.
    private static func stagingSpans(header: [String: Any], prefixes: [String], fileSize: Int) -> [Int?] {
        guard let table = header["tensors"] as? [String: [String: Any]],
              let page = header["page"] as? Int, page > 0 else { return prefixes.map { _ in nil } }
        let names = (header["order"] as? [String]) ?? table.keys.sorted()
        return prefixes.map { prefix -> Int? in
            let runs = names.filter { $0.hasPrefix(prefix) }.compactMap { name -> (offset: Int, bytes: Int)? in
                guard let entry = table[name], let offset = entry["offset"] as? Int,
                      let bytes = entry["bytes"] as? Int else { return nil }
                return (offset, bytes)
            }
            return stagingRun(runs, page: page, fileSize: fileSize).map { $0.end - $0.start }
        }
    }
}

// ── sizing: a stream of units, from the map's table ──────────────────────────────────────────

extension Artifact {
    /// **What one tensor occupies in a staging buffer**: from its offset to its end rounded up to the page,
    /// never past the file. Tensors start on a page, so this is a whole number of pages but for the file's last.
    package static func stagingLength(offset: Int, bytes: Int, page: Int, fileSize: Int) -> Int {
        min((offset + bytes + page - 1) / page * page, fileSize) - offset
    }

    /// The staging length of a named tensor (`stagingLength`); 0 for one the map does not have.
    package func stagingLength(of name: String) -> Int {
        tensors[name].map { Self.stagingLength(offset: $0.offset, bytes: $0.bytes, page: page, fileSize: size) } ?? 0
    }

    /// **The visits of a stream over these layers**: each layer's `names`, in the order the engine uses them
    /// (a name that is used several times appears several times), cut into runs of consecutive uses whose
    /// distinct tensors' staging lengths add up to at most `cap` (a tensor longer than it is a visit alone).
    /// A visit never spans two layers, so a layer whose last call is missed frees its buffers at its end.
    /// Names the map does not have are left out. The same cut serves the stream and the memory need.
    package func stagingUnits(layers: [[String]], cap: Int) -> [[String]] {
        layers.flatMap { layer -> [[String]] in
            var units: [[String]] = [], current: [String] = [], inside = Set<String>(), sum = 0
            for name in layer where tensors[name] != nil {
                let length = inside.contains(name) ? 0 : stagingLength(of: name)
                if !current.isEmpty, sum + length > cap {
                    units.append(current); current = []; inside = []; sum = 0
                }
                if inside.insert(name).inserted { sum += stagingLength(of: name) }
                current.append(name)
            }
            if !current.isEmpty { units.append(current) }
            return units
        }
    }

    /// **A layer's tensors in the order its code widens them** — the order a stream reads them in: those of
    /// `suffixes` (after `prefix`) present in `names`, then any other, in file order, so that none is left out.
    package static func inConsumptionOrder(_ names: [String], prefix: String, suffixes: [String]) -> [String] {
        let wanted = suffixes.map { prefix + $0 }, present = Set(names)
        return wanted.filter(present.contains) + names.filter { !wanted.contains($0) }
    }

    /// **Streams a tail by units** (`TailStream`, sequential): the layers `released...` of `layers`, each in
    /// the order of `suffixes`, in `TailStream.unitSlots` buffers of the largest tensor's size.
    package func streamUnits(layers: [[String]], from released: Int, prefix: (Int) -> String, suffixes: [String]) throws {
        try streamUnits(ordered: (released..<layers.count).map {
            Artifact.inConsumptionOrder(layers[$0], prefix: prefix($0), suffixes: suffixes)
        })
    }

    /// `streamUnits` for layers whose uses the caller has listed itself, repeats included.
    package func streamUnits(ordered streamed: [[String]]) throws {
        let cap = streamed.joined().map { stagingLength(of: $0) }.max() ?? 0
        try streamTail(blocks: stagingUnits(layers: streamed, cap: cap), slots: TailStream.unitSlots,
                       sequential: true, buffer: cap)
    }

    /// **The format a map's products are published in**: that of its largest two-dimensional tensor — `int8` (`q8` when its scales
    /// are per block of rows), a GGUF-packed type (`q4k`: it widens), `bf16` for the rest. What the memory need asks the plan about
    /// (`ProductKey.weight`) before any tensor is read by name.
    package static func weightFormat(header: [String: Any]) -> CapacityClasses.WeightFormat {
        guard let table = header["tensors"] as? [String: [String: Any]] else { return .bf16 }
        // An SQ map's products are int8 products (`SQ`: decoded into the reserve), whatever tensor is the largest in bytes (its bf16
        // refiners are).
        if table.values.contains(where: { ($0["dtype"] as? String).flatMap(DType.init(rawValue:))?.sqBits != nil }) { return .int8 }
        let largest = table.values.filter { ($0["shape"] as? [Int])?.count == 2 }
            .max { ($0["bytes"] as? Int ?? 0) < ($1["bytes"] as? Int ?? 0) }
        guard let entry = largest, let raw = entry["dtype"] as? String, let dtype = DType(rawValue: raw) else { return .bf16 }
        if dtype == .int8 { return ((entry["scale"] as? [String: Any])?["block"] as? Int) == nil ? .int8 : .q8 }
        return dtype.isPacked ? .q4k : .bf16
    }

    /// **The largest tensor's staging length** among the tensors whose name starts with one of the prefixes —
    /// from the header alone (the memory need) — the size of a stream's buffer. 0 without a tensor table.
    package static func largestRun(header: [String: Any], prefixes: [String], fileSize: Int) -> Int {
        guard let table = header["tensors"] as? [String: [String: Any]],
              let page = header["page"] as? Int, page > 0 else { return 0 }
        return table.compactMap { name, entry -> Int? in
            guard prefixes.contains(where: { name.hasPrefix($0) }), let offset = entry["offset"] as? Int,
                  let bytes = entry["bytes"] as? Int else { return nil }
            return stagingLength(offset: offset, bytes: bytes, page: page, fileSize: fileSize)
        }.max() ?? 0
    }
}
