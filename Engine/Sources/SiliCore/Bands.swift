import CoreAI
import Foundation
import Metal

/// **Bands: a stage of layers over planes, each layer a function called band after band down the plane's rows.** The
/// client writes its network as layers (`layer`): what each call reads (windows of planes, their rows, the row each side
/// of them), what it writes (rows of a plane), its function (`Function`, built at the shapes the plan gives) and its
/// weights. The engine sizes the bands, lays every plane out, names every function by its bytes, plans the stage under
/// B, binds each argument by its name and runs the calls.
///
///     a plane       a tensor [C, rows, width] lying as bands of ρ rows, each [1, C, ρ, W] on pages of its own — every
///                   argument a buffer whole, as Core AI asks (`Layout`) — with, under the guard, each band's rows'
///                   factors ([ρ] f32, a page a band), and what its layer carries from band to band (`carry`)
///     a layer       one function called once a band: its band's height the tallest power of two whose output holds at
///                   most `positions` positions (`height`) — **the same function whatever the plane's rows**: a taller
///                   plane is more calls, its last band past the plane's end (its rows computed, masked, never read as
///                   values)
///     ρ             each plane's bands the largest dividing every call that reads or writes it, from a survey of the
///                   network (`surveyed`: the steps run once on bands of a row, once on what they gave)
///     a slab        planes released are laid again in the free slab that holds them with the least room
///
/// **What a call is given, by name** (`Read`, `Write`): of a window `x`, its bands `x0`…, the band before's last row
/// `x_before` and the band after's first `x_after` (past the plane, the band itself), their factors `x0_factor`…, its
/// mask `x_mask` (1 for a row inside the plane, 0 past either end); its plane's carry, under the name the read gives it.
/// Of an output `y`, its bands `y0`… and their factors; its mask `y_valid` (a row past the plane's end must be written 0);
/// its carry: `y_carried` read (zeros at the layer's first band), `y_carry` written. Any other argument is a weight by
/// the layer's map, or a tensor of the stage.
///
/// **The plane's rows past its end are never read as values**: a window's mask multiplies them out, an output's
/// `valid` zeroes them, a whole read keeps the plane's rows alone (`join`). Measured 19/11 on the M5 Pro (a band 128 →
/// 128 channels of 16 × 1024, a 3×3): a window joined from three bands of their own 1.85 ms a call, the same window as a
/// strided view of the plane 2.46 ms (copied by Core AI at each call).
public struct Bands: Sendable {
    // MARK: - The network as the client writes it

    /// A plane of the stage: its channels, its rows and columns, its element and factors, what its writer carries.
    public struct Plane: Sendable, Hashable {
        public let id: Int
        public let channels, rows, width: Int
        public let element: NDArray.ScalarType
        public let factors: Bool
        /// The shape of what each band's call carries to the next (f32), empty for none.
        public let carry: [Int]
        /// Its bands' rows and their count (from the survey; 1 and its rows while surveyed).
        public let rho: Int
        public let count: Int
        /// Its slab, or nil: a tensor apart (a stage's exit, `apart`).
        let slab: Int?
        public var coverage: Int { rho * count }
        var bandBytes: Int { Arena.aligned(Layout.paddedBytes([1, channels, rho, width], element)) }
        var bytes: Int { count * bandBytes }
    }

    /// A window a call reads: `rows` rows of `plane` per row of its output band (`per`: 1; ½ an ×2's, `over` 2; a ↓'s 2),
    /// with the row before and the row after; `carry`, the argument its plane's carry is read as; `whole`, every row.
    public struct Read: Sendable {
        public let name: String
        public let plane: Plane
        public let per: Int, over: Int
        public let before: Bool, after: Bool
        public let carry: String?
        public init(_ name: String, _ plane: Plane, per: Int = 1, over: Int = 1, before: Bool = false, after: Bool = false, carry: String? = nil) {
            (self.name, self.plane, self.per, self.over, self.before, self.after, self.carry) = (name, plane, per, over, before, after, carry)
        }
    }

    /// What a call writes: its band's rows of `plane`, under `name`.
    public struct Write: Sendable {
        public let name: String
        public let plane: Plane
        public init(_ name: String, _ plane: Plane) { (self.name, self.plane) = (name, plane) }
    }

    /// **The shapes a function is built at**: its band's rows; of each window its channels, rows (the halo apart), columns,
    /// bands, element, factors and halo; of each output its channels, rows, columns, bands, element, factors and carry.
    public struct Geometry: Sendable, Hashable {
        public struct Window: Sendable, Hashable {
            public let channels, rows, width, bands: Int
            public let element: NDArray.ScalarType
            public let factors, before, after: Bool
            /// The rows its bands hold: past `rows` on a whole read, cut back to them (`join`).
            public let held: Int
            /// The argument its plane's carry is read as, and its shape.
            public let carry: String?
            public let carried: [Int]
        }
        public struct Output: Sendable, Hashable {
            public let channels, rows, width, bands: Int
            public let element: NDArray.ScalarType
            public let factors: Bool
            public let carry: [Int]
            /// The rows its bands hold: past `rows` on a whole write, zeros added (`split`).
            public let held: Int
            /// Whether a mask of its rows inside the plane is given (`valid`): every banded write.
            public let masked: Bool
        }
        public let rows: Int
        public let windows: [String: Window]
        public let outputs: [String: Output]
    }

    /// A function of the client: what tells it apart from another at the same shapes (`key`), its module at a geometry.
    public struct Function: Sendable {
        public let key: String
        public let build: @Sendable (Geometry) throws -> Aicode.Module
        public init(_ key: String, build: @escaping @Sendable (Geometry) throws -> Aicode.Module) { self.key = key; self.build = build }
    }

    // MARK: - The plan

    /// Where a call's argument lies: a tensor of the stage, from a byte, at a shape and element; its rows padded as Core
    /// AI's GPU asks.
    struct Spot: Sendable {
        let tensor: String, offset: Int, shape: [Int], type: NDArray.ScalarType
    }
    /// A function at its shapes: what one asset is built from.
    public struct Shaped: Sendable, Hashable { public let key: String; public let geometry: Geometry }
    struct Call: Sendable {
        let function: Shaped
        let geometry: Geometry
        let spots: [String: Spot]
        let weights: [String: String]
        let tensors: [String: String]
    }

    /// The most positions a band's output holds, the plan's.
    public let positions: Int
    public let element: NDArray.ScalarType
    public let factors: Bool
    private(set) var planes: [Plane] = []
    private(set) var slabs: [Int] = []
    private var free: [Int] = []
    private(set) var smalls: [(name: String, bytes: Int, apart: Bool)] = []
    private(set) var calls: [Call] = []
    private(set) var functions: [Shaped: Function] = [:]
    /// Boundaries the client reads: a plane whole once the call at its index is done.
    private(set) var marks: [(name: String, call: Int, plane: Plane)] = []
    /// The masks the stage fills: by name, their length, the rows inside the plane.
    private(set) var masks: [String: (length: Int, inside: Range<Int>)] = [:]
    /// Per plane: the call that last wrote its carry (the buffer it lies in), and how many calls wrote it.
    private var carried: [Int: Int] = [:]
    /// From the survey: each plane's bands' rows and the rows its calls reach; gathered while surveyed.
    private let surveyed: [Int: (rho: Int, reach: Int)]?
    private var survey: [Int: (rho: Int, reach: Int)] = [:]

    private init(positions: Int, element: NDArray.ScalarType, factors: Bool, surveyed: [Int: (rho: Int, reach: Int)]?) {
        (self.positions, self.element, self.factors, self.surveyed) = (positions, element, factors, surveyed)
    }

    /// **The plan `steps` write**, run twice: surveyed (each plane in bands of a row), then with each plane in bands of
    /// the largest rows dividing every call that reads or writes it. `element`, `factors`: the planes' unless one says.
    public static func surveyed(positions: Int, element: NDArray.ScalarType, factors: Bool, _ steps: (inout Bands) -> Void) -> Bands {
        var survey = Bands(positions: positions, element: element, factors: factors, surveyed: nil)
        steps(&survey)
        var plan = Bands(positions: positions, element: element, factors: factors, surveyed: survey.survey)
        steps(&plan)
        return plan
    }

    /// **A plane** of `channels` × `rows` × `width`, in the free slab that holds it with the least room (else the largest
    /// free one grown, else a new one), or `apart` (a stage's exit: its last call writes it). `carry`: what the layer
    /// that writes it carries from band to band.
    public mutating func plane(_ channels: Int, rows: Int, width: Int, element: NDArray.ScalarType? = nil, factors: Bool? = nil,
                               carry: [Int] = [], apart: Bool = false) -> Plane {
        let id = planes.count, e = element ?? self.element, f = factors ?? self.factors
        let (bands, reach) = surveyed.map { $0[id] ?? (rows, rows) } ?? (1, rows)
        // Bands no call bounds (a plane read and written whole): one band.
        let rho = bands == 0 ? rows : bands
        precondition(!f || rho * 4 <= Arena.page, "bands of \(rho) rows: their factors past a page")
        let count = (max(reach, rows) + rho - 1) / rho
        var p = Plane(id: id, channels: channels, rows: rows, width: width, element: e, factors: f, carry: carry, rho: rho, count: count, slab: nil)
        if apart {
            small("t\(id)", p.bytes, apart: true)
        } else {
            let need = p.bytes, slab: Int
            if let i = free.filter({ slabs[$0] >= need }).min(by: { slabs[$0] < slabs[$1] }) { slab = i }
            else if let i = free.max(by: { slabs[$0] < slabs[$1] }) { slab = i; slabs[i] = need }
            else { slab = slabs.count; slabs.append(need) }
            free.removeAll { $0 == slab }
            p = Plane(id: id, channels: channels, rows: rows, width: width, element: e, factors: f, carry: carry, rho: rho, count: count, slab: slab)
        }
        if f { small("f\(id)", count * Arena.page) }
        if !carry.isEmpty {
            let bytes = Layout.paddedBytes(carry, .float32)
            small("c\(id)#0", bytes); small("c\(id)#1", bytes); small("zeros-" + carry.map(String.init).joined(separator: "x"), bytes)
        }
        planes.append(p)
        return p
    }

    /// The plane's slab free again: a plane made after takes it.
    public mutating func release(_ p: Plane) { if let s = p.slab { free.append(s) } }

    /// A tensor of the stage the client names (`layer`'s `tensors`), zeroed; `apart`, a stage's exit.
    public mutating func tensor(_ name: String, bytes: Int, apart: Bool = false) { small(name, bytes, apart: apart) }

    /// The plane whole once the last call so far is done, read back under `name` (`BandStage.run(keeping:)`).
    public mutating func mark(_ name: String, _ plane: Plane) { marks.append((name, calls.count - 1, plane)) }

    private mutating func small(_ name: String, _ bytes: Int, apart: Bool = false) {
        if !smalls.contains(where: { $0.name == name }) { smalls.append((name, bytes, apart)) }
    }

    /// **The rows of a band** whose output, `width` wide, holds at most `positions`: a power of two, at least `least`,
    /// whatever the plane's rows, so that a taller plane is more calls of the same functions. The price: a plane
    /// shorter than its band is one band, its rows past the plane computed for nothing (on paper: a 64 × 64 plane at
    /// 2¹⁴ positions, a band of 256 rows, four times its work).
    public static func height(width: Int, positions: Int, least: Int = 1) -> Int {
        var r = 1
        while 2 * r * width <= positions { r *= 2 }
        return max(r, least)
    }

    /// **A layer**: `function` called once a band of its first output's rows, each band reading `reads` and writing
    /// `writes`, its weights by argument (`weights`, the residents' keys), its tensors by argument (`tensors`).
    /// `positions`: the layer's own bound on a band's output (a layer whose values move with its band's height: the
    /// tallest band of every plan); `whole`: one call over every row (a function of the plane's size).
    public mutating func layer(_ function: Function, reads: [Read], writes: [Write], weights: [String: String] = [:], tensors: [String: String] = [:],
                               positions: Int? = nil, whole: Bool = false) {
        let out = writes.first?.plane ?? reads[0].plane
        let least = reads.map(\.over).max() ?? 1
        let r = whole ? max(out.coverage, out.rows) : Self.height(width: out.width, positions: positions ?? self.positions, least: least)
        let n = whole ? 1 : (out.rows + r - 1) / r
        for b in 0..<n {
            var spots: [String: Spot] = [:]
            var windows: [String: Geometry.Window] = [:], outputs: [String: Geometry.Output] = [:]
            for read in reads {
                let t = read.plane
                let (first, rows) = whole ? (0, t.rows) : (b * r * read.per / read.over, r * read.per / read.over)
                let held = whole ? t.coverage : rows
                note(t, first, held, whole: whole)
                let j0 = first / t.rho, k = (held + t.rho - 1) / t.rho
                func put(_ name: String, _ j: Int) {
                    spots[name] = band(t, j)
                    if t.factors { spots["\(name)_factor"] = factors(t, j) }
                }
                for i in 0..<k { put("\(read.name)\(i)", j0 + i) }
                if read.before { put("\(read.name)_before", j0 > 0 ? j0 - 1 : j0) }
                if read.after { put("\(read.name)_after", j0 + k < t.count ? j0 + k : j0 + k - 1) }
                if read.before || read.after {
                    let start = first - (read.before ? 1 : 0), length = rows + (read.before ? 1 : 0) + (read.after ? 1 : 0)
                    spots["\(read.name)_mask"] = mask(start, length, t.rows)
                }
                if let c = read.carry, !t.carry.isEmpty { spots[c] = carry(t) }
                let carry = t.carry.isEmpty ? nil : read.carry
                windows[read.name] = Geometry.Window(channels: t.channels, rows: whole ? t.rows : rows, width: t.width, bands: k, element: t.element,
                                                     factors: t.factors, before: read.before, after: read.after, held: k * t.rho,
                                                     carry: carry, carried: carry == nil ? [] : t.carry)
            }
            for write in writes {
                let t = write.plane
                let (first, rows) = whole ? (0, t.rows) : (b * r * t.rows / out.rows, r * t.rows / out.rows)
                let held = whole ? t.coverage : rows
                note(t, first, held, whole: whole)
                let j0 = first / t.rho, k = (held + t.rho - 1) / t.rho
                for i in 0..<k {
                    spots["\(write.name)\(i)"] = band(t, j0 + i)
                    if t.factors { spots["\(write.name)\(i)_factor"] = factors(t, j0 + i) }
                }
                if !whole { spots["\(write.name)_valid"] = mask(first, rows, t.rows) }
                if !t.carry.isEmpty {
                    spots["\(write.name)_carried"] = b == 0 ? zeros(t) : carry(t)
                    carried[t.id] = (carried[t.id] ?? -1) + 1
                    spots["\(write.name)_carry"] = carry(t)
                }
                outputs[write.name] = Geometry.Output(channels: t.channels, rows: whole ? t.rows : rows, width: t.width, bands: k, element: t.element,
                                                      factors: t.factors, carry: t.carry, held: k * t.rho, masked: !whole)
            }
            let geometry = Geometry(rows: whole ? out.rows : r, windows: windows, outputs: outputs)
            let key = Shaped(key: function.key, geometry: geometry)
            if functions[key] == nil { functions[key] = function }
            calls.append(Call(function: key, geometry: geometry, spots: spots, weights: weights, tensors: tensors))
        }
    }

    /// While surveyed, the rows a call reaches of `t`; after, a check that they lie on its bands. A whole read or write
    /// takes the bands as they lie: it bounds no band.
    private mutating func note(_ t: Plane, _ first: Int, _ rows: Int, whole: Bool) {
        guard surveyed != nil else {
            func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
            let s = survey[t.id] ?? (0, 0)
            survey[t.id] = (whole ? s.rho : gcd(gcd(s.rho, rows), first), max(s.reach, first + rows))
            return
        }
        precondition(first % t.rho == 0, "rows from \(first) across bands of \(t.rho)")
    }

    private func band(_ t: Plane, _ j: Int) -> Spot {
        Spot(tensor: t.slab.map { "slab\($0)" } ?? "t\(t.id)", offset: j * t.bandBytes, shape: [1, t.channels, t.rho, t.width], type: t.element)
    }
    private func factors(_ t: Plane, _ j: Int) -> Spot { Spot(tensor: "f\(t.id)", offset: j * Arena.page, shape: [t.rho], type: .float32) }

    /// The buffer of `t`'s carry the last call that wrote it wrote, of the two in turn.
    private func carry(_ t: Plane) -> Spot { Spot(tensor: "c\(t.id)#\((carried[t.id] ?? 0) % 2)", offset: 0, shape: t.carry, type: .float32) }
    private func zeros(_ t: Plane) -> Spot { Spot(tensor: "zeros-" + t.carry.map(String.init).joined(separator: "x"), offset: 0, shape: t.carry, type: .float32) }

    /// A mask of `length` rows from row `start`: 1 for a row inside `0 ..< rows`, 0 past either end.
    private mutating func mask(_ start: Int, _ length: Int, _ rows: Int) -> Spot {
        let inside = max(0, -start) ..< max(max(0, -start), min(length, rows - start))
        let name = "mask-\(length)-\(inside.lowerBound)-\(inside.upperBound)"
        if masks[name] == nil { masks[name] = (length, inside); small(name, length * 4) }
        return Spot(tensor: name, offset: 0, shape: [length], type: .float32)
    }

    /// The residents every call names, sorted.
    public var weights: [String] { Array(Set(calls.flatMap(\.weights.values))).sorted() }

    /// The functions a plan builds: one a layer's kind and shapes.
    public var functionCount: Int { functions.count }
    /// Each function's module at its shapes, nothing named or written.
    public func modules() throws -> [Shaped: Aicode.Module] { try functions.reduce(into: [:]) { $0[$1.key] = try $1.value.build($1.key.geometry) } }
    public var callCount: Int { calls.count }

    /// **What its calls hold, by names the same at any size** (`Stage.Extent`): of each function in the order of its
    /// first call, numbered among those of its key, each window's elements (its rows each side counted) and each
    /// output's; its kin named from these names, their elements' type, what each function computes (`Block.walk`: the
    /// kinds of its ops, each once), `prefix` and the calls kept in flight. A plane of another size moves a function's
    /// widths and rows, not its place in the network nor its kinds of ops. Not its ops in order: a whole layer writes its
    /// output as the bands of the plane that holds it (`split`), one slice each, so that its ops in order were another
    /// kin at each band's height — no reading of one height bounded another, and a stage whose soberest form was read
    /// alone never took a taller one again (a network of bands on the M5 Pro, 24/11/2026: bands of 2048 positions read, those
    /// of 16 384 unknown, its run 2.64 s; its whole render 9.49 → 7.73 s once the kin held at every height).
    public func extent(_ prefix: String, inFlight: Int) throws -> Stage.Extent {
        var named: [Shaped: String] = [:], count: [String: Int] = [:], elements: [String: Int] = [:], lines = ["prefix \(prefix)"]
        for call in calls where named[call.function] == nil {
            let key = call.function.key, name = "\(key)#\(count[key, default: 0])"
            count[key, default: 0] += 1
            named[call.function] = name
            lines.append("\(name) ops \(Block.walk(try functions[call.function]!.build(call.function.geometry).body).kinds)")
            for (n, w) in call.geometry.windows {
                elements["\(name) \(n)"] = (w.held + (w.before ? 1 : 0) + (w.after ? 1 : 0)) * w.width * w.channels
                lines.append("\(name) \(n) \(w.element)")
            }
            for (n, o) in call.geometry.outputs {
                elements["\(name) \(n) out"] = o.held * o.width * o.channels
                lines.append("\(name) \(n) out \(o.element)")
            }
        }
        return Stage.Extent(lines, inFlight: inFlight, elements: elements)
    }
}
