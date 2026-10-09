import Foundation

/// **A JSON that keeps the order of its keys and the spelling of its numbers.**
///
/// `JSONSerialization` loses both: a dictionary has no order, and `1.0` becomes an `NSNumber` there
/// that can no longer be told apart from `1`. Yet a map's header copies the published model's
/// `config.json`, and the map forged in Swift must say **the same thing** as the one Python forged
/// (`json.dumps`) — same keys, in the same order, same numbers. A number therefore keeps its
/// original text; a number computed here is written the way `repr` would write it.
package indirect enum OrderedJSON: Sendable, Equatable {
    case null
    case boolean(Bool)
    /// The number's text, as read or as `repr` would write it.
    case number(String)
    case string(String)
    case list([OrderedJSON])
    case object([Pair])

    package struct Pair: Sendable, Equatable {
        package let key: String
        package let value: OrderedJSON
        package init(_ key: String, _ value: OrderedJSON) { self.key = key; self.value = value }
    }

    package static func integer(_ n: Int) -> OrderedJSON { .number(String(n)) }

    /// A float written the way Python would write it: `256.0`, `1e-05`, `3.3895e+38`.
    package static func real(_ x: Double) -> OrderedJSON {
        .number(OrderedJSON.reprPython(x))
    }

    static func reprPython(_ x: Double) -> String {
        if x.isNaN { return "NaN" }
        if x.isInfinite { return x > 0 ? "Infinity" : "-Infinity" }
        // Swift and Python both give the shortest decimal that reads back the same double; they
        // differ in notation. Python switches to the exponent below 1e-4 and from 1e16.
        let a = abs(x)
        if a != 0 && (a < 1e-4 || a >= 1e16) {
            var s = "\(x)"                                   // "1e-05", "3.3895e+38"
            if !s.contains("e") {                            // Swift may have written without an exponent
                s = String(format: "%.17g", x)
            }
            // Python always writes the exponent with at least two digits, with its sign.
            if let e = s.firstIndex(of: "e") {
                let mantissa = String(s[..<e])
                var exponent = String(s[s.index(after: e)...])
                let sign = exponent.hasPrefix("-") ? "-" : "+"
                exponent = exponent.trimmingCharacters(in: CharacterSet(charactersIn: "+-"))
                if exponent.count < 2 { exponent = "0" + exponent }
                return mantissa + "e" + sign + exponent
            }
            return s
        }
        let s = "\(x)"
        return s.contains(".") || s.contains("e") ? s : s + ".0"
    }

    // MARK: - Reading

    package subscript(_ key: String) -> OrderedJSON? {
        guard case .object(let pairs) = self else { return nil }
        return pairs.first { $0.key == key }?.value
    }

    package var text: String? { if case .string(let s) = self { return s }; return nil }
    package var integer: Int? { if case .number(let s) = self { return Int(s) }; return nil }
    package var double: Double? { if case .number(let s) = self { return Double(s) }; return nil }
    package var boolean: Bool? { if case .boolean(let b) = self { return b }; return nil }
    package var elements: [OrderedJSON]? { if case .list(let l) = self { return l }; return nil }
    package var pairs: [Pair]? { if case .object(let p) = self { return p }; return nil }

    /// The same object without the keys that start with `_` — what the forges kept of
    /// `config.json` (`{k: v for k, v in config.items() if not k.startswith("_")}`).
    package var withoutPrivateKeys: OrderedJSON {
        guard case .object(let p) = self else { return self }
        return .object(p.filter { !$0.key.hasPrefix("_") })
    }

    // MARK: - Parsing

    package struct ParseError: Error, CustomStringConvertible {
        package let description: String
    }

    package static func parse(_ data: Data) throws -> OrderedJSON {
        var l = Parser(bytes: [UInt8](data))
        l.blanks()
        let v = try l.value()
        l.blanks()
        guard l.i == l.bytes.count else { throw ParseError(description: "JSON: extra bytes at \(l.i)") }
        return v
    }

    package static func read(_ path: String) throws -> OrderedJSON {
        try parse(try Data(contentsOf: URL(fileURLWithPath: path)))
    }

    /// How deeply objects and lists may nest. A `config.json` nests three or four levels; the
    /// parser recurses once per level, so a hostile file of `[[[[…` would exhaust the stack.
    package static let maximumDepth = 64

    private struct Parser {
        let bytes: [UInt8]
        var i = 0
        var depth = 0

        mutating func blanks() {
            while i < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[i]) { i += 1 }
        }

        mutating func wait(_ c: UInt8) throws {
            guard i < bytes.count, bytes[i] == c else {
                throw ParseError(description: "JSON: \"\(Character(UnicodeScalar(c)))\" expected at \(i)")
            }
            i += 1
        }

        mutating func word(_ m: String) throws {
            for c in m.utf8 { try wait(c) }
        }

        mutating func value() throws -> OrderedJSON {
            guard i < bytes.count else { throw ParseError(description: "JSON: premature end") }
            switch bytes[i] {
            case UInt8(ascii: "{"):
                try enter(); defer { depth -= 1 }
                i += 1; blanks()
                var pairs: [Pair] = []
                if i < bytes.count, bytes[i] == UInt8(ascii: "}") { i += 1; return .object(pairs) }
                while true {
                    blanks()
                    let key = try chain()
                    blanks(); try wait(UInt8(ascii: ":")); blanks()
                    pairs.append(Pair(key, try value()))
                    blanks()
                    if i < bytes.count, bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                    try wait(UInt8(ascii: "}"))
                    return .object(pairs)
                }
            case UInt8(ascii: "["):
                try enter(); defer { depth -= 1 }
                i += 1; blanks()
                var list: [OrderedJSON] = []
                if i < bytes.count, bytes[i] == UInt8(ascii: "]") { i += 1; return .list(list) }
                while true {
                    blanks()
                    list.append(try value())
                    blanks()
                    if i < bytes.count, bytes[i] == UInt8(ascii: ",") { i += 1; continue }
                    try wait(UInt8(ascii: "]"))
                    return .list(list)
                }
            case UInt8(ascii: "\""): return .string(try chain())
            case UInt8(ascii: "t"): try word("true"); return .boolean(true)
            case UInt8(ascii: "f"): try word("false"); return .boolean(false)
            case UInt8(ascii: "n"): try word("null"); return .null
            case UInt8(ascii: "N"): try word("NaN"); return .number("NaN")
            case UInt8(ascii: "I"): try word("Infinity"); return .number("Infinity")
            default:
                let begin = i
                if bytes[i] == UInt8(ascii: "-") {
                    i += 1
                    if i < bytes.count, bytes[i] == UInt8(ascii: "I") { try word("Infinity"); return .number("-Infinity") }
                }
                while i < bytes.count, "0123456789.eE+-".utf8.contains(bytes[i]) { i += 1 }
                guard i > begin else { throw ParseError(description: "JSON: unexpected value at \(i)") }
                return .number(String(decoding: bytes[begin..<i], as: UTF8.self))
            }
        }

        mutating func enter() throws {
            guard depth < OrderedJSON.maximumDepth else {
                throw ParseError(description: "JSON: nested deeper than \(OrderedJSON.maximumDepth) levels at \(i)")
            }
            depth += 1
        }

        mutating func hex4() throws -> UInt32 {
            guard i + 4 <= bytes.count,
                  let v = UInt32(String(decoding: bytes[i..<i + 4], as: UTF8.self), radix: 16) else {
                throw ParseError(description: "JSON: invalid \\u escape at \(i)")
            }
            i += 4
            return v
        }

        mutating func chain() throws -> String {
            try wait(UInt8(ascii: "\""))
            var raw: [UInt8] = []
            while true {
                guard i < bytes.count else { throw ParseError(description: "JSON: unterminated string") }
                let c = bytes[i]; i += 1
                if c == UInt8(ascii: "\"") { break }
                if c != UInt8(ascii: "\\") { raw.append(c); continue }
                guard i < bytes.count else { throw ParseError(description: "JSON: truncated escape") }
                let e = bytes[i]; i += 1
                switch e {
                case UInt8(ascii: "n"): raw.append(0x0A)
                case UInt8(ascii: "t"): raw.append(0x09)
                case UInt8(ascii: "r"): raw.append(0x0D)
                case UInt8(ascii: "b"): raw.append(0x08)
                case UInt8(ascii: "f"): raw.append(0x0C)
                case UInt8(ascii: "u"):
                    var code = try hex4()
                    if (0xD800..<0xDC00).contains(code), i + 6 <= bytes.count,
                       bytes[i] == UInt8(ascii: "\\"), bytes[i + 1] == UInt8(ascii: "u") {
                        i += 2
                        let down = try hex4()
                        // A high half must be followed by a low half: anything else would make
                        // `down - 0xDC00` wrap (a trap) or build a scalar from garbage.
                        guard (0xDC00..<0xE000).contains(down) else {
                            throw ParseError(description: "JSON: \\u escape at \(i - 6) is not the low half of a surrogate pair")
                        }
                        code = 0x10000 + ((code - 0xD800) << 10) + (down - 0xDC00)
                    }
                    raw.append(contentsOf: Array(String(UnicodeScalar(code).map(Character.init) ?? "\u{FFFD}").utf8))
                default: raw.append(e)                       // \" \\ \/
                }
            }
            return String(decoding: raw, as: UTF8.self)
        }
    }

    // MARK: - Writing, the way `json.dumps` does

    /// `", "` and `": "` as separators, every non-ASCII character escaped as `\uXXXX`.
    package var jsonText: String {
        var s = ""
        write(into: &s)
        return s
    }

    private func write(into s: inout String) {
        switch self {
        case .null: s += "null"
        case .boolean(let b): s += b ? "true" : "false"
        case .number(let n): s += n
        case .string(let c): OrderedJSON.escape(c, into: &s)
        case .list(let l):
            s += "["
            for (k, v) in l.enumerated() {
                if k > 0 { s += ", " }
                v.write(into: &s)
            }
            s += "]"
        case .object(let p):
            s += "{"
            for (k, pair) in p.enumerated() {
                if k > 0 { s += ", " }
                OrderedJSON.escape(pair.key, into: &s)
                s += ": "
                pair.value.write(into: &s)
            }
            s += "}"
        }
    }

    private static func escape(_ c: String, into s: inout String) {
        s += "\""
        for u in c.utf16 {
            switch u {
            case 0x22: s += "\\\""
            case 0x5C: s += "\\\\"
            case 0x0A: s += "\\n"
            case 0x0D: s += "\\r"
            case 0x09: s += "\\t"
            case 0x08: s += "\\b"
            case 0x0C: s += "\\f"
            case 0x20..<0x7F: s.unicodeScalars.append(UnicodeScalar(UInt8(u)))
            default: s += String(format: "\\u%04x", u)
            }
        }
        s += "\""
    }
}

extension OrderedJSON: ExpressibleByStringLiteral, ExpressibleByBooleanLiteral, ExpressibleByIntegerLiteral {
    package init(stringLiteral value: String) { self = .string(value) }
    package init(booleanLiteral value: Bool) { self = .boolean(value) }
    package init(integerLiteral value: Int) { self = .integer(value) }
}
