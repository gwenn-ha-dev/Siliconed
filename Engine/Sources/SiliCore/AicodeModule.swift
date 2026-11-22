import CryptoKit
import Foundation

/// **The module**: one function as coreai-torch writes it, its bytecode, its asset written.
extension Aicode {
    /// One function, `main`, as coreai-torch writes it: named arguments, named results.
    public struct Module {
        public var body: Block
        public var argumentNames: [String]
        public var outputs: [(name: String, value: Int)]
        /// The blobs of the constants of 256 elements or more, in the order their constants were added.
        public private(set) var resources: [(key: String, bytes: [UInt8])] = []

        public init(arguments: [(String, Entry)]) {
            body = Block(arguments.map(\.1))
            argumentNames = arguments.map(\.0)
            outputs = []
        }

        /// The value of the argument `name`.
        public func argument(_ name: String) -> Int {
            guard let i = argumentNames.firstIndex(of: name) else { preconditionFailure("no argument \(name)") }
            return i
        }

        /// **The module as the engine's functions declare it**: each tensor argument and result without its trailing
        /// unit axes (`Layout.trimmed`), reshaped back where it is read and before it is returned; the module itself when
        /// none has any. The same ops, renumbered (every region is isolated: only the top block's operands move).
        public func trimmed() -> Module {
            let declared = body.arguments.map(Aicode.trimmed)
            guard declared != body.arguments || outputs.contains(where: { Aicode.trimmed(body.types[$0.value]) != body.types[$0.value] }) else { return self }
            var m = Module(arguments: Array(zip(argumentNames, declared)))
            m.resources = resources
            let b = m.body
            var map = body.arguments.indices.map { declared[$0] == body.arguments[$0] ? $0 : b.reshape($0, Aicode.shape(body.arguments[$0]).shape) }
            for op in body.ops {
                map += b.add(Op(op.name, operands: op.operands.map { map[$0] }, results: op.results, properties: op.properties, body: op.body))
            }
            m.outputs = outputs.map { o in
                let v = map[o.value], t = Aicode.trimmed(b.types[v])
                return (o.name, t == b.types[v] ? v : b.reshape(v, Aicode.shape(t).shape))
            }
            return m
        }

        /// A constant the module computes with (a rotation's matrix; the weights are arguments): written in the IR under
        /// 256 elements, a splat when every element is the same, otherwise a blob of the resource section, 16-aligned,
        /// under `key` — coreai-torch's rule (`create_elements_attr`), the key ours: the same bytes from one run to the next.
        public mutating func constant(_ key: String, _ bytes: [UInt8], _ type: Entry) -> Int {
            let (shape, element) = Aicode.shape(type)
            let width = element.size
            precondition(bytes.count == element.bytes(shape.reduce(1, *)), "constant \(key): \(bytes.count) bytes for \(shape)")
            precondition(element != .si4 || shape.reduce(1, *) >= 256, "constant \(key): \(shape) int4 is under 256 codes")
            let data: Entry
            if shape.reduce(1, *) < 256 {
                let small = bytes, first = small.prefix(width)
                let splat = stride(from: 0, to: small.count, by: width).allSatisfy { small[$0..<$0 + width].elementsEqual(first) }
                data = Aicode.dense(splat ? Array(first) : small, splat: splat)
            } else {
                precondition(!resources.contains { $0.key == key }, "resource \(key) given twice")
                resources.append((key, bytes))
                data = Entry(7, [.resource(key)])
            }
            return body.constant(data, type)
        }

        /// The bytes of `main.mlirb`. A call that carries a composite (`Aicode.Block.attention`) has the composite's graph
        /// written before `main`, as coreai-torch's assets hold them, and keeps only its symbol.
        public func bytecode() -> [UInt8] {
            typealias Op = Aicode.Op
            var graphs: [Op] = [], calls: [(op: Op, body: Aicode.Block, properties: [Aicode.Property])] = []
            for op in body.ops where op.name == "dataflow.invoke_v1" {
                guard let inner = op.body, let properties = op.properties, properties.count == 4,
                      case .attribute(let callee) = properties[1], case .attribute(let reference) = callee.fields[0],
                      case .string(let name) = reference.fields[1], case .attribute(let declaration) = properties[3] else { continue }
                let results = inner.ops.last.map { $0.operands.map { inner.types[$0] } } ?? []
                let function = Aicode.function(inner.arguments, results)
                graphs.append(Op("dataflow.graph_v1", properties: [
                    .attribute(Aicode.names(inner.arguments.indices.map { ["q", "k", "v", "m"][$0] })), .optional(declaration),
                    .attribute(Aicode.no), .attribute(Aicode.typeAttribute(function)),
                    .attribute(Aicode.integer(1, bits: 1)), .attribute(Aicode.string(name)),
                    .attribute(Aicode.integer(1, bits: 1)), .attribute(Aicode.array([])),
                    .attribute(Aicode.array([Aicode.array([])])),
                ], body: inner))
                calls.append((op, inner, properties))
            }
            for c in calls { c.op.body = nil; c.op.properties = Array(c.properties.prefix(3)) }
            defer { for c in calls { c.op.body = c.body; c.op.properties = c.properties } }

            var all: [Op] = []
            func walk(_ b: Aicode.Block) { for op in b.ops { all.append(op); if let body = op.body { walk(body) } } }
            graphs.forEach { walk($0.body!) }
            walk(body)
            let date = all.map { Op.since($0.name) }.max() ?? Op.since("dataflow.graph_v1")
            let resultTypes = outputs.map { body.types[$0.value] }
            let main = Op("dataflow.graph_v1", properties: [
                .attribute(Aicode.names(argumentNames)), .int(0),
                .attribute(Aicode.no), .attribute(Aicode.typeAttribute(Aicode.function(body.arguments, resultTypes))),
                .attribute(Aicode.no), .attribute(Aicode.string("main")),
                .attribute(Aicode.no), .attribute(Aicode.array([])),
                .attribute(Aicode.names(outputs.map(\.name))),
            ], body: body)
            body.ops.append(Op("dataflow.output_v1", operands: outputs.map(\.value)))
            defer { body.ops.removeLast() }
            let inner = Aicode.Block([]); inner.ops = graphs + [main]
            let asset = Op("serialization.asset", properties: [.int(0), .optional(Aicode.integer(date, bits: 32))], body: inner)
            return Writer(root: asset, resources: resources).bytes()
        }

    }

    /// Writes `directory` (an `.aimodel`): `main.mlirb`, its SHA-256 in `main.hash`, and `metadata.json`.
    public static func write(_ bytecode: [UInt8], to directory: URL, producer: String) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(bytecode).write(to: directory.appendingPathComponent("main.mlirb"))
        try Data(SHA256.hash(data: bytecode)).write(to: directory.appendingPathComponent("main.hash"))
        let date = ISO8601DateFormatter()
        date.formatOptions = [.withYear, .withMonth, .withDay, .withTime, .withTimeZone]
        let stamp = date.string(from: Date()).replacingOccurrences(of: "-", with: "").replacingOccurrences(of: ":", with: "")
        let metadata = "{\n  \"producer\" : \"\(producer)\",\n  \"creationDate\" : \"\(stamp)\",\n  \"assetVersion\" : \"2.0\"\n}"
        try Data(metadata.utf8).write(to: directory.appendingPathComponent("metadata.json"))
    }
}
