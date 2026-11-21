import Foundation
import SiliCore

/// **The tokenizer's two big tables, outside `malloc`**: the vocabulary (`symbol → id`) and
/// the merge ranks (`left \0 right → rank`), 151 k entries each for Qwen.
///
/// Built as Swift dictionaries from `JSONSerialization`'s object graph, they and the graph left
/// ~52 MB of freed blocks in `phys_footprint` after every render's text phase on macOS 27 — more
/// than the whole of what survived the DiT. Here the keys are read straight from the mapped
/// `tokenizer.json` (`TokenizerJSON`), unescaped into a byte heap, and indexed by an open-addressing
/// table, both in `Pages`: unmapped with the tokenizer.
///
/// Keys compare **byte for byte** (UTF-8), where a Swift `String` key compares by canonical
/// equivalence. The same thing for these keys: byte-level BPE symbols are the 256 one-scalar strings
/// of `Tokenizer.byteSymbols` — precomposed letters, no combining mark — so no two distinct byte
/// sequences of them are canonically equivalent. A duplicate key keeps the **last** value, as the
/// dictionaries did (`ranks[key] = rank`; `JSONSerialization` on a repeated object key).
struct ByteTable {
    private let heap: Pages
    private let slots: Pages
    private let mask: Int
    private(set) var count = 0
    private var used = 0

    /// One slot: the key's heap offset and length, its value, and the hash's high bits (0: empty).
    private struct Slot { var offset: Int32; var length: Int32; var value: Int32; var tag: UInt32 }

    /// For at most `entries` keys of at most `keyBytes` bytes in all.
    init(entries: Int, keyBytes: Int) {
        let capacity = ByteTable.slotCount(entries: entries)
        mask = capacity - 1
        heap = Pages(bytes: max(1, keyBytes))
        slots = Pages(bytes: capacity * MemoryLayout<Slot>.stride)   // zero-filled: every slot empty
    }

    /// The slots for `entries` keys: a power of two, at least twice the keys (and 16).
    private static func slotCount(entries: Int) -> Int {
        var capacity = 16
        while capacity < 2 * max(1, entries) { capacity *= 2 }
        return capacity
    }

    private var table: UnsafeMutablePointer<Slot> { slots.base.assumingMemoryBound(to: Slot.self) }
    private var bytes: UnsafeMutablePointer<UInt8> { heap.base.assumingMemoryBound(to: UInt8.self) }

    /// FNV-1a, 64 bits.
    private static func hash(_ key: UnsafeRawBufferPointer) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in key { h = (h ^ UInt64(b)) &* 0x0000_0100_0000_01b3 }
        return h
    }

    private func find(_ key: UnsafeRawBufferPointer) -> (index: Int, found: Bool) {
        let h = ByteTable.hash(key)
        let tag = UInt32(truncatingIfNeeded: h >> 32) | 1
        var i = Int(truncatingIfNeeded: h) & mask
        while true {
            let slot = table[i]
            if slot.tag == 0 { return (i, false) }
            if slot.tag == tag, Int(slot.length) == key.count,
               memcmp(bytes + Int(slot.offset), key.baseAddress ?? UnsafeRawPointer(bytes), key.count) == 0 {
                return (i, true)
            }
            i = (i + 1) & mask
        }
    }

    /// The key is the `length` bytes the caller has just written at `pendingKey` (the heap's end).
    mutating func commit(length: Int, value: Int) {
        let key = UnsafeRawBufferPointer(start: bytes + used, count: length)
        let (i, found) = find(key)
        if found { table[i].value = Int32(value); return }    // last wins, the key's bytes dropped
        let h = ByteTable.hash(key)
        table[i] = Slot(offset: Int32(used), length: Int32(length), value: Int32(value),
                        tag: UInt32(truncatingIfNeeded: h >> 32) | 1)
        used += length
        count += 1
    }

    /// Where the next key's bytes go; `commit` then indexes them.
    var pendingKey: UnsafeMutablePointer<UInt8> { bytes + used }

    func value(_ key: String) -> Int? {
        var key = key
        return key.withUTF8 { utf8 in
            let (i, found) = find(UnsafeRawBufferPointer(utf8))
            return found ? Int(table[i].value) : nil
        }
    }

    /// Every `(key, value)`, in slot order.
    func forEach(_ body: (String, Int) -> Void) {
        for i in 0...mask where table[i].tag != 0 {
            let s = table[i]
            body(String(decoding: UnsafeBufferPointer(start: bytes + Int(s.offset), count: Int(s.length)), as: UTF8.self),
                 Int(s.value))
        }
    }
}

/// **A reader of `tokenizer.json`'s raw bytes**, just enough to find `model.vocab` and `model.merges`
/// and to read their strings — the two values `JSONSerialization` must not see (`ByteTable`).
/// Strict where it reads, refusing what is not JSON, and silent on nothing.
struct TokenizerJSON {
    let bytes: UnsafeRawBufferPointer

    struct Malformed: Error { let at: Int }

    static func isSpace(_ b: UInt8) -> Bool { b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D }

    func space(_ i: Int) -> Int {
        var i = i
        while i < bytes.count, TokenizerJSON.isSpace(bytes[i]) { i += 1 }
        return i
    }

    func expect(_ byte: UInt8, _ i: Int) throws -> Int {
        let i = space(i)
        guard i < bytes.count, bytes[i] == byte else { throw Malformed(at: i) }
        return i + 1
    }

    /// Past the string that starts at `i` (`"`).
    func skipString(_ i: Int) throws -> Int {
        guard i < bytes.count, bytes[i] == 0x22 else { throw Malformed(at: i) }
        var i = i + 1
        while i < bytes.count {
            switch bytes[i] {
            case 0x5C: i += 2
            case 0x22: return i + 1
            default: i += 1
            }
        }
        throw Malformed(at: i)
    }

    /// Past the value that starts at `i` (after spaces).
    func skipValue(_ i: Int) throws -> Int {
        var i = space(i)
        guard i < bytes.count else { throw Malformed(at: i) }
        switch bytes[i] {
        case 0x22: return try skipString(i)
        case 0x7B, 0x5B:
            var depth = 0
            while i < bytes.count {
                switch bytes[i] {
                case 0x22: i = try skipString(i); continue
                case 0x7B, 0x5B: depth += 1
                case 0x7D, 0x5D:
                    depth -= 1
                    if depth == 0 { return i + 1 }
                default: break
                }
                i += 1
            }
            throw Malformed(at: i)
        default:
            while i < bytes.count, bytes[i] != 0x2C, bytes[i] != 0x7D, bytes[i] != 0x5D, !TokenizerJSON.isSpace(bytes[i]) { i += 1 }
            return i
        }
    }

    /// The string at `i` (after spaces), unescaped into `out` (room for its raw length at least:
    /// unescaping never lengthens). Returns its length and the index past it.
    func string(_ i: Int, into out: UnsafeMutablePointer<UInt8>) throws -> (length: Int, next: Int) {
        var i = space(i)
        guard i < bytes.count, bytes[i] == 0x22 else { throw Malformed(at: i) }
        i += 1
        var n = 0
        func put(scalar v: UInt32) throws {
            guard let s = Unicode.Scalar(v) else { throw Malformed(at: i) }
            for b in String(s).utf8 { out[n] = b; n += 1 }
        }
        func hex4(_ at: Int) throws -> UInt32 {
            guard at + 4 <= bytes.count else { throw Malformed(at: at) }
            var v: UInt32 = 0
            for k in at..<(at + 4) {
                let c = bytes[k]
                let d: UInt32
                switch c {
                case 0x30...0x39: d = UInt32(c - 0x30)
                case 0x41...0x46: d = UInt32(c - 0x41 + 10)
                case 0x61...0x66: d = UInt32(c - 0x61 + 10)
                default: throw Malformed(at: k)
                }
                v = v << 4 | d
            }
            return v
        }
        while i < bytes.count {
            let c = bytes[i]
            if c == 0x22 { return (n, i + 1) }
            if c != 0x5C { out[n] = c; n += 1; i += 1; continue }
            guard i + 1 < bytes.count else { throw Malformed(at: i) }
            switch bytes[i + 1] {
            case 0x22: out[n] = 0x22; n += 1; i += 2
            case 0x5C: out[n] = 0x5C; n += 1; i += 2
            case 0x2F: out[n] = 0x2F; n += 1; i += 2
            case 0x62: out[n] = 0x08; n += 1; i += 2
            case 0x66: out[n] = 0x0C; n += 1; i += 2
            case 0x6E: out[n] = 0x0A; n += 1; i += 2
            case 0x72: out[n] = 0x0D; n += 1; i += 2
            case 0x74: out[n] = 0x09; n += 1; i += 2
            case 0x75:
                let high = try hex4(i + 2)
                i += 6
                if (0xD800..<0xDC00).contains(high), i + 1 < bytes.count, bytes[i] == 0x5C, bytes[i + 1] == 0x75 {
                    let low = try hex4(i + 2)
                    guard (0xDC00..<0xE000).contains(low) else { throw Malformed(at: i) }
                    try put(scalar: 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
                    i += 6
                } else {
                    try put(scalar: high)      // a lone surrogate is refused by `Unicode.Scalar`
                }
            default: throw Malformed(at: i)
            }
        }
        throw Malformed(at: i)
    }

    /// The value ranges of the members `names` of the object at `i`, in one pass.
    func members(of i: Int, _ names: Set<String>) throws -> [String: Range<Int>] {
        var i = try expect(0x7B, i)
        var found: [String: Range<Int>] = [:]
        if bytes[space(i)] == 0x7D { return found }
        let scratch = UnsafeMutablePointer<UInt8>.allocate(capacity: 256)
        defer { scratch.deallocate() }
        while true {
            let keyStart = space(i), keyEnd = try skipString(keyStart)
            var name: String?
            if keyEnd - keyStart <= 256 {
                let (n, _) = try string(keyStart, into: scratch)
                name = String(decoding: UnsafeBufferPointer(start: scratch, count: n), as: UTF8.self)
            }
            i = try expect(0x3A, keyEnd)
            let valueStart = space(i), valueEnd = try skipValue(valueStart)
            if let name, names.contains(name) { found[name] = valueStart..<valueEnd }
            i = space(valueEnd)
            guard i < bytes.count else { throw Malformed(at: i) }
            if bytes[i] == 0x7D { return found }
            i = try expect(0x2C, i)
        }
    }

    /// Each element start of the array at `range`, or each `(key start, value start)` of the object.
    func elements(_ range: Range<Int>, _ body: (Int) throws -> Void) throws {
        let open = bytes[range.lowerBound], close: UInt8 = open == 0x5B ? 0x5D : 0x7D
        var i = space(range.lowerBound + 1)
        if bytes[i] == close { return }
        while true {
            try body(i)
            i = try skipValue(i)
            if open == 0x7B { i = try skipValue(try expect(0x3A, i)) }
            i = space(i)
            guard i < range.upperBound else { throw Malformed(at: i) }
            if bytes[i] == close { return }
            i = space(try expect(0x2C, i))
        }
    }

    /// The integer at `i` (after spaces).
    func integer(_ i: Int) throws -> Int {
        let start = space(i), end = try skipValue(start)
        guard end > start, let v = Int(String(decoding: UnsafeRawBufferPointer(rebasing: bytes[start..<end]), as: UTF8.self)) else {
            throw Malformed(at: start)
        }
        return v
    }
}
