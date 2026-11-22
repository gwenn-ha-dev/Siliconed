import Foundation

// The bytecode written: its numbers (`BytecodeEmitter`), then MLIR's numbering and writing (`Writer`).

/// MLIR's prefix varint: the trailing zeros of the first byte, plus one, are its length; nine bytes past 2⁵⁶.
struct BytecodeEmitter {
    var bytes: [UInt8] = []

    mutating func byte(_ b: UInt8) { bytes.append(b) }
    mutating func append(_ b: [UInt8]) { bytes += b }

    mutating func varint(_ v: UInt64) {
        for n in 1...8 where v < UInt64(1) << UInt64(7 * n) {
            let encoded = (v << UInt64(n)) | (UInt64(1) << UInt64(n - 1))
            for i in 0..<n { bytes.append(UInt8(truncatingIfNeeded: encoded >> UInt64(8 * i))) }
            return
        }
        bytes.append(0)
        for i in 0..<8 { bytes.append(UInt8(truncatingIfNeeded: v >> UInt64(8 * i))) }
    }
    mutating func varint(_ v: Int) { varint(UInt64(v)) }
    /// Zigzag, as MLIR's `emitSignedVarInt`.
    mutating func signed(_ v: Int64) { varint(UInt64(bitPattern: (v << 1) ^ (v >> 63))) }

    mutating func section(_ id: UInt8, _ contents: [UInt8]) {
        byte(id)
        varint(contents.count)
        append(contents)
    }
}

/// MLIR's `IRNumbering` and `BytecodeWriter`, for one dialect and isolated single-block regions.
struct Writer {
    let root: Aicode.Op
    let resources: [(key: String, bytes: [UInt8])]

    private struct Counted<Key: Hashable> {
        var first: [Key: Int] = [:]
        var refs: [Int] = []
        var keys: [Key] = []
        /// True when the key is new.
        mutating func count(_ k: Key) -> Bool {
            if let i = first[k] { refs[i] += 1; return false }
            first[k] = keys.count; keys.append(k); refs.append(1); return true
        }
        /// Most used first, ties in the order met.
        func ordered() -> (keys: [Key], index: [Key: Int]) {
            let order = keys.indices.sorted { refs[$0] != refs[$1] ? refs[$0] > refs[$1] : $0 < $1 }
            let keys = order.map { self.keys[$0] }
            return (keys, Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($1, $0) }))
        }
    }

    func bytes() -> [UInt8] {
        // Numbering: pre-order, isolated regions from a stack.
        var ops = Counted<String>(), attrs = Counted<Aicode.Entry>(), types = Counted<Aicode.Entry>()
        var used = Counted<String>()  // resources, in the order met (never sorted)
        func field(_ f: Aicode.Field) {
            switch f {
            case .attribute(let e): attribute(e)
            case .type(let e): type(e)
            case .resource(let k): _ = used.count(k)
            default: break
            }
        }
        func attribute(_ e: Aicode.Entry) { if attrs.count(e) { e.fields.forEach(field) } }
        func type(_ e: Aicode.Entry) { if types.count(e) { e.fields.forEach(field) } }
        func number(_ op: Aicode.Op) {
            _ = ops.count(op.name)
            op.results.forEach(type)
            for p in op.properties ?? [] {
                switch p {
                case .attribute(let e): attribute(e)
                case .optional(let e?): attribute(e)
                default: break
                }
            }
            attribute(Aicode.location)
        }
        number(root)
        var stack: [Aicode.Block] = root.body.map { [$0] } ?? []
        while let block = stack.popLast() {
            for t in block.arguments { attribute(Aicode.location); type(t) }
            block.ops.forEach(number)
            stack += block.ops.compactMap(\.body)
        }
        let (opOrder, opIndex) = ops.ordered()
        let (attrOrder, attrIndex) = attrs.ordered()
        let (typeOrder, typeIndex) = types.ordered()

        // Strings, as they are written.
        var strings: [String] = [], stringIndex: [String: Int] = [:]
        func str(_ s: String) -> Int {
            if let i = stringIndex[s] { return i }
            stringIndex[s] = strings.count; strings.append(s); return strings.count - 1
        }

        var dialect = BytecodeEmitter()
        dialect.varint(1)
        dialect.varint(str("aicode") << 1 | 1)
        var version = BytecodeEmitter()
        [1, 2, UInt64(opOrder.count)].forEach { version.varint($0) }
        dialect.varint(version.bytes.count); dialect.append(version.bytes)
        dialect.varint(0); dialect.varint(opOrder.count)
        for name in opOrder { dialect.varint(str(name) << 1 | 1) }

        func encode(_ e: Aicode.Entry) -> [UInt8] {
            var w = BytecodeEmitter()
            w.varint(e.kind)
            for f in e.fields {
                switch f {
                case .int(let v): w.varint(v)
                case .signed(let v): w.signed(v)
                case .byte(let b): w.byte(b)
                case .string(let s): w.varint(str(s))
                case .attribute(let a): w.varint(attrIndex[a]!)
                case .type(let t): w.varint(typeIndex[t]!)
                case .blob(let b): w.varint(b.count); w.append(b)
                case .resource(let k): w.varint(used.first[k]!)
                }
            }
            return w.bytes
        }
        var table = BytecodeEmitter(), offsets = BytecodeEmitter()
        offsets.varint(attrOrder.count); offsets.varint(typeOrder.count)
        for group in [attrOrder, typeOrder] where !group.isEmpty {
            offsets.varint(0); offsets.varint(group.count)
            for e in group { let b = encode(e); table.append(b); offsets.varint(b.count << 1 | 1) }
        }

        // IR, properties numbered as they are written (equal encodings shared).
        var properties: [[UInt8]] = [], propertyIndex: [[UInt8]: Int] = [:]
        let location = attrIndex[Aicode.location]!
        func write(_ op: Aicode.Op, _ w: inout BytecodeEmitter) {
            w.varint(opIndex[op.name]!)
            var mask: UInt8 = 0
            if !op.results.isEmpty { mask |= 0x02 }
            if !op.operands.isEmpty { mask |= 0x04 }
            if op.body != nil { mask |= 0x10 }
            if op.properties != nil { mask |= 0x40 }
            w.byte(mask)
            w.varint(location)
            if let props = op.properties {
                var p = BytecodeEmitter()
                for f in props {
                    switch f {
                    case .int(let v): p.varint(v)
                    case .attribute(let e): p.varint(attrIndex[e]!)
                    case .optional(let e): if let e { p.varint(attrIndex[e]! << 1 | 1) } else { p.varint(0) }
                    }
                }
                if propertyIndex[p.bytes] == nil { propertyIndex[p.bytes] = properties.count; properties.append(p.bytes) }
                w.varint(propertyIndex[p.bytes]!)
            }
            if !op.results.isEmpty { w.varint(op.results.count); op.results.forEach { w.varint(typeIndex[$0]!) } }
            if !op.operands.isEmpty { w.varint(op.operands.count); op.operands.forEach { w.varint($0) } }
            if let body = op.body {
                w.varint(1 << 1 | 1)
                var region = BytecodeEmitter()
                region.varint(1); region.varint(body.values)
                region.varint(body.ops.count << 1 | (body.arguments.isEmpty ? 0 : 1))
                if !body.arguments.isEmpty {
                    region.varint(body.arguments.count)
                    for t in body.arguments { region.varint(typeIndex[t]! << 1 | 1); region.varint(location) }
                    region.byte(0)
                }
                for o in body.ops { write(o, &region) }
                w.section(4, region.bytes)
            }
        }
        var ir = BytecodeEmitter()
        ir.varint(1 << 1)
        write(root, &ir)

        // Resources: each blob its alignment, its size, padding to 16 from the file's start, its bytes; the
        // section aligned the same way, so that a reader can map the data in place.
        let blobs = Dictionary(uniqueKeysWithValues: resources.map { ($0.key, $0.bytes) })
        precondition(Set(used.keys) == Set(blobs.keys), "resources given and used differ")
        var resourceOffsets = BytecodeEmitter(), resourceSection: [UInt8] = [], header = BytecodeEmitter(), length = 0
        resourceOffsets.varint(0)  // no external group
        if !used.keys.isEmpty {
            resourceOffsets.varint(0); resourceOffsets.varint(used.keys.count)
            for k in used.keys {
                let start = length
                header = BytecodeEmitter()
                header.varint(16); header.varint(blobs[k]!.count)
                while (length + header.bytes.count) % 16 != 0 { header.byte(0xcb) }
                resourceSection += header.bytes + blobs[k]!
                length += header.bytes.count + blobs[k]!.count
                resourceOffsets.varint(str(k)); resourceOffsets.varint(length - start); resourceOffsets.byte(0)
            }
        }

        var stringSection = BytecodeEmitter()
        stringSection.varint(strings.count)
        for s in strings.reversed() { stringSection.varint(s.utf8.count + 1) }
        for s in strings { stringSection.append(Array(s.utf8) + [0]) }
        var propertySection = BytecodeEmitter()
        propertySection.varint(properties.count)
        for p in properties { propertySection.varint(p.count); propertySection.append(p) }

        var file = BytecodeEmitter()
        file.append([0x4d, 0x4c, 0xef, 0x52]); file.varint(6); file.append(Array("MLIR22.0.0git".utf8) + [0])
        file.section(1, dialect.bytes)
        file.section(3, offsets.bytes)
        file.section(2, table.bytes)
        file.section(4, ir.bytes)
        file.section(6, resourceOffsets.bytes)
        if resourceSection.isEmpty { file.section(5, []) } else {
            file.byte(5 | 0x80); file.varint(length); file.varint(16)
            while file.bytes.count % 16 != 0 { file.byte(0xcb) }
            file.append(resourceSection)
        }
        file.section(0, stringSection.bytes)
        file.section(8, propertySection.bytes)
        return file.bytes
    }
}
