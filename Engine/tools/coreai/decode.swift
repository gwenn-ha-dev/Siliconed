// Decodes an MLIR bytecode file (a Core AI asset's main.mlirb): sections, strings, dialects and op names, every
// attribute and type entry (custom-encoded or text), the IR and properties sections in hex. The map for the writer.
//   swiftc -O -o .build/scratch/decode tools/coreai/decode.swift && .build/scratch/decode ASSET/main.mlirb

import Foundation
let bytes = [UInt8](try! Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
struct R {
    let b: [UInt8]; var p: Int
    mutating func byte() -> UInt8 { defer { p += 1 }; return b[p] }
    mutating func varint() -> UInt64 {
        let first = b[p]
        if first == 0 { var v: UInt64 = 0; for i in 0..<8 { v |= UInt64(b[p + 1 + i]) << (8 * i) }; p += 9; return v }
        let n = first.trailingZeroBitCount + 1
        var v: UInt64 = 0; for i in 0..<n { v |= UInt64(b[p + i]) << (8 * i) }
        p += n; return v >> n
    }
    mutating func cstring() -> String { var s: [UInt8] = []; while b[p] != 0 { s.append(b[p]); p += 1 }; p += 1; return String(decoding: s, as: UTF8.self) }
}
var r = R(b: bytes, p: 4)
print("version", r.varint(), "producer", r.cstring())
var sections: [Int: [UInt8]] = [:]
while r.p < bytes.count {
    let code = r.byte(); let id = Int(code & 0x7f); let len = Int(r.varint())
    if code & 0x80 != 0 { let align = Int(r.varint()); while r.p % align != 0 { precondition(bytes[r.p] == 0xcb); r.p += 1 } }
    sections[id] = Array(bytes[r.p..<r.p + len]); print("section", id, "bytes", len); r.p += len
}
// Strings
var s = R(b: sections[0]!, p: 0); let count = Int(s.varint())
var sizes: [Int] = (0..<count).map { _ in Int(s.varint()) }; sizes.reverse()
var strings: [String] = []
for z in sizes { strings.append(String(decoding: s.b[s.p..<s.p + z - 1], as: UTF8.self)); s.p += z }
for (i, t) in strings.enumerated() { print(" str", i, t.debugDescription.prefix(70)) }
// Dialects
var d = R(b: sections[1]!, p: 0); let nd = Int(d.varint()); var dialects: [String] = []
for _ in 0..<nd {
    let v = d.varint(); let name = strings[Int(v >> 1)]; dialects.append(name)
    if v & 1 == 1 { let sz = Int(d.varint()); print(" dialect", name, "version bytes", Array(d.b[d.p..<d.p+sz])); d.p += sz } else { print(" dialect", name) }
}
var opNames: [String] = []
while d.p < d.b.count {
    let di = Int(d.varint()); let n = Int(d.varint())
    for _ in 0..<n { let v = d.varint(); let nm = dialects[di] + "." + strings[Int(v >> 1)]; opNames.append(nm); print(" op", opNames.count - 1, nm, v & 1 == 1 ? "registered" : "") }
}
// Attribute/type offsets
var o = R(b: sections[3]!, p: 0); let na = Int(o.varint()), nt = Int(o.varint())
print("attrs", na, "types", nt)
var attrType = R(b: sections[2]!, p: 0)
var index = 0
while o.p < o.b.count {
    let di = Int(o.varint()); let n = Int(o.varint())
    for _ in 0..<n {
        let v = o.varint(); let size = Int(v >> 1); let custom = v & 1 == 1
        let chunk = Array(attrType.b[attrType.p..<attrType.p + size]); attrType.p += size
        let kind = index < na ? "attr" : "type"
        let text = custom ? chunk.map { String(format: "%02x", $0) }.joined(separator: " ") : String(decoding: chunk.dropLast(), as: UTF8.self)
        print(" \(kind) \(index < na ? index : index - na) [\(dialects[di])] \(custom ? "custom" : "text  ") \(text.prefix(110))")
        index += 1
    }
}
func hex(_ a: [UInt8]) -> String { a.map { String(format: "%02x", $0) }.joined(separator: " ") }
print("IR", hex(sections[4]!))
print("props", hex(sections[8]!))
print("resource offsets", hex(sections[6]!))

// The IR as a tree: per op its name, mask, location, properties index, result types, operands, regions.
func ir(_ r: inout R, _ depth: Int) {
    let pad = String(repeating: "  ", count: depth)
    let blocks = Int(r.varint()); if blocks == 0 { print(pad + "(empty region)"); return }
    let values = Int(r.varint()); print(pad + "region: blocks \(blocks) values \(values)")
    for _ in 0..<blocks {
        let header = r.varint(); let ops = Int(header >> 1)
        var line = pad + "block: ops \(ops)"
        if header & 1 == 1 {
            let n = Int(r.varint()); var args: [String] = []
            for _ in 0..<n { let v = r.varint(); args.append("t\(v >> 1)" + (v & 1 == 1 ? "@a\(r.varint())" : "")) }
            line += " args " + args.joined(separator: " ")
            line += " uselist \(r.byte())"
        }
        print(line)
        for _ in 0..<ops {
            let name = opNames[Int(r.varint())]; let mask = r.byte()
            var s = pad + "  " + name + String(format: " mask %02x loc a%d", mask, r.varint())
            if mask & 0x01 != 0 { s += " attrs a\(r.varint())" }
            if mask & 0x40 != 0 { s += " props p\(r.varint())" }
            if mask & 0x02 != 0 { let n = Int(r.varint()); s += " results " + (0..<n).map { _ in "t\(r.varint())" }.joined(separator: ",") }
            if mask & 0x04 != 0 { let n = Int(r.varint()); s += " operands " + (0..<n).map { _ in "v\(r.varint())" }.joined(separator: ",") }
            if mask & 0x08 != 0 { let n = Int(r.varint()); s += " successors " + (0..<n).map { _ in "b\(r.varint())" }.joined(separator: ",") }
            if mask & 0x20 != 0 { s += " (use-list orders: not decoded)" }
            print(s)
            if mask & 0x10 != 0 {
                let v = r.varint(); let regions = Int(v >> 1); let isolated = v & 1 == 1
                for _ in 0..<regions {
                    if isolated {
                        let code = r.byte(); precondition(code & 0x7f == 4); let len = Int(r.varint())
                        var sub = R(b: Array(r.b[r.p..<r.p + len]), p: 0); r.p += len
                        ir(&sub, depth + 2)
                    } else { ir(&r, depth + 2) }
                }
            }
        }
    }
}
var top = R(b: sections[4]!, p: 0)
let topHeader = top.varint(); print("top block: ops \(topHeader >> 1)")
for _ in 0..<Int(topHeader >> 1) {
    let name = opNames[Int(top.varint())]; let mask = top.byte()
    var s = name + String(format: " mask %02x loc a%d", mask, top.varint())
    if mask & 0x01 != 0 { s += " attrs a\(top.varint())" }
    if mask & 0x40 != 0 { s += " props p\(top.varint())" }
    print(s)
    if mask & 0x10 != 0 {
        let v = top.varint()
        for _ in 0..<Int(v >> 1) {
            let code = top.byte(); precondition(code & 0x7f == 4); let len = Int(top.varint())
            var sub = R(b: Array(top.b[top.p..<top.p + len]), p: 0); top.p += len
            ir(&sub, 1)
        }
    }
}
var pr = R(b: sections[8]!, p: 0); let np = Int(pr.varint())
for i in 0..<np { let n = Int(pr.varint()); print(" p\(i)", hex(Array(pr.b[pr.p..<pr.p + n]))); pr.p += n }
