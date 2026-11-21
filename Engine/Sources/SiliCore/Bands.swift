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

    /// The number of each function's calls, by its key.
    public var counts: [String: Int] { calls.reduce(into: [:]) { $0[$1.function.key, default: 0] += 1 } }
    /// The functions a plan builds: one a layer's kind and shapes.
    public var functionCount: Int { functions.count }
    public var shapes: Set<Shaped> { Set(functions.keys) }
    /// Each function's module at its shapes, nothing named or written.
    public func modules() throws -> [Shaped: Aicode.Module] { try functions.reduce(into: [:]) { $0[$1.key] = try $1.value.build($1.key.geometry) } }
    /// The boundaries marked, in order: each its name and the call after which it is whole.
    public var boundaries: [(name: String, call: Int)] { marks.map { ($0.name, $0.call) } }
    public var callCount: Int { calls.count }
}

// MARK: - Writing a band's function

extension Bands {
    /// **A function's arguments at `g`**, the weights aside, in one order (an asset is named by its bytes): each window's
    /// bands, the band before and after, their factors, its mask and its plane's carry; each output's mask and the carry
    /// it is given.
    public static func arguments(_ g: Geometry) -> [(String, Aicode.Entry)] {
        func element(_ t: NDArray.ScalarType) -> Aicode.Element { t == .float16 ? .f16 : .f32 }
        var out: [(String, Aicode.Entry)] = []
        for (name, w) in g.windows.sorted(by: { $0.key < $1.key }) {
            let rho = w.held / w.bands
            let names = (0..<w.bands).map { "\(name)\($0)" } + (w.before ? ["\(name)_before"] : []) + (w.after ? ["\(name)_after"] : [])
            for n in names {
                out.append((n, Aicode.tensor([1, w.channels, rho, w.width], element(w.element))))
                if w.factors { out.append(("\(n)_factor", Aicode.tensor([rho], .f32))) }
            }
            if w.before || w.after { out.append(("\(name)_mask", Aicode.tensor([w.rows + (w.before ? 1 : 0) + (w.after ? 1 : 0)], .f32))) }
            if let c = w.carry { out.append((c, Aicode.tensor(w.carried, .f32))) }
        }
        for (name, o) in g.outputs.sorted(by: { $0.key < $1.key }) {
            if o.masked { out.append(("\(name)_valid", Aicode.tensor([o.rows], .f32))) }
            if !o.carry.isEmpty { out.append(("\(name)_carried", Aicode.tensor(o.carry, .f32))) }
        }
        return out
    }

    /// **The shape a weight of `shape` is given as**: its first axis and the rest flat, `[O, C · kh · kw]` for a
    /// convolution's kernel, when a row of it is a multiple of 64 bytes; else rows of 64 bytes — Core AI's GPU copies an
    /// argument whose rows are not at every call (`Layout`). The function reshapes it back. Measured 19/11 on the mini, f16
    /// (3×3 128 → 128 at 128 × 1024, 3×3 and 1×1 512 → 512 at 64 × 256): a kernel given flat runs at 0.95 – 0.96 × the
    /// same kernel a constant of its asset, the same bits; given `[O, C, 3, 3]`, copied at every call, 0.77 × and 0.32 ×.
    public static func flat(_ shape: [Int], _ e: Aicode.Element) -> [Int] {
        let n = shape.reduce(1, *), rest = n / shape[0], row = 64 / e.size
        if rest * e.size % 64 == 0 { return [shape[0], rest] }
        precondition(n % row == 0, "a weight of \(shape) in no rows of 64 bytes")
        return [n / row, row]
    }

    /// **The window `name` joined** in `m` (`Geometry.Window`): its bands, the band before's last row and the band
    /// after's first, along the rows (`[1, C, rows + halo, W]`), and their factors (`[rows + halo]` f32) under the guard;
    /// a whole read cut back to the plane's rows. The mask is the client's to apply (`<name>_mask`): what it zeroes
    /// depends on what it reads (a norm's output is not zero where its input is).
    public static func join(_ m: Aicode.Module, _ name: String, _ w: Geometry.Window) -> (value: Int, factor: Int?) {
        let b = m.body
        func rows(_ suffix: String, _ axis: Int) -> Int {
            var parts: [Int] = []
            if w.before { parts.append(b.slice(m.argument("\(name)_before\(suffix)"), axis: axis, start: w.held / w.bands - 1, size: 1)) }
            parts += (0..<w.bands).map { m.argument("\(name)\($0)\(suffix)") }
            if w.after { parts.append(b.slice(m.argument("\(name)_after\(suffix)"), axis: axis, start: 0, size: 1)) }
            let joined = parts.count == 1 ? parts[0] : b.concat(parts, axis: axis)
            return w.held > w.rows && !w.before && !w.after ? b.slice(joined, axis: axis, start: 0, size: w.rows) : joined
        }
        return (rows("", 2), w.factors ? rows("_factor", 0) : nil)
    }

    /// **The output `name` as its bands** (`Geometry.Output`): `value` `[1, C, rows, W]` and its factors `[rows]` cut in
    /// `name0`, `name0_factor`, …; a whole write's rows past the plane's added, zeros and factors of 1.
    public static func split(_ b: Aicode.Block, _ name: String, value: Int, factor: Int?, _ o: Geometry.Output) -> [(String, Int)] {
        var (v, f) = (value, factor)
        if o.held > o.rows {
            let pad = o.held - o.rows, e = Aicode.shape(b.types[v]).element
            v = b.concat([v, b.constant(Aicode.dense(Aicode.scalar(0, e), splat: true), Aicode.tensor([1, o.channels, pad, o.width], e))], axis: 2)
            if let g = f { f = b.concat([g, b.constant(Aicode.dense(Aicode.scalar(1, .f32), splat: true), Aicode.tensor([pad], .f32))], axis: 0) }
        }
        let rho = o.held / o.bands
        var out: [(String, Int)] = []
        for i in 0..<o.bands {
            out.append(("\(name)\(i)", o.bands == 1 ? v : b.slice(v, axis: 2, start: i * rho, size: rho)))
            if let f { out.append(("\(name)\(i)_factor", o.bands == 1 ? f : b.slice(f, axis: 0, start: i * rho, size: rho))) }
        }
        return out
    }
}

// MARK: - The live stage

/// **A stage of bands, its arena filled and its functions loaded**: the plan of the first candidate whose P holds under
/// B (bands of fewer positions, the shorter, the later), each distinct function once (the same module, the same asset:
/// two layers of one shape share it), every weight a resident; its calls one after the other, each waited for.
public final class BandStage: @unchecked Sendable {
    public let plan: Bands
    public let stage: Stage
    /// The rank of the candidate taken (0: the preferred).
    public let candidate: Int
    /// **Whether B made the band shorter than the preferred**: a taller one's socle, or its estimate, over B. Not when no
    /// candidate was ever read on this Mac and the soberest is taken for want of a reading (`Stage.choose`): that is a
    /// first run at these shapes, not this Mac's limit (seen 21/11 under B = 46 GiB: the shortest band at a stage's first
    /// run, the tallest at its second, and its client said B had cut it).
    public let narrowed: Bool
    private let functions: [Bands.Shaped: InferenceFunction]
    private let signatures: [Bands.Shaped: Block.Signature]

    private init(_ plan: Bands, _ stage: Stage, candidate: Int, narrowed: Bool, signatures: [Bands.Shaped: Block.Signature],
                 assets: [Bands.Shaped: String]) {
        (self.plan, self.stage, self.candidate, self.narrowed, self.signatures) = (plan, stage, candidate, narrowed, signatures)
        var loaded: [Bands.Shaped: InferenceFunction] = [:]
        for (key, asset) in assets { loaded[key] = stage.functions[asset] }
        functions = loaded
        for (name, m) in plan.masks {
            let p = stage.pointer(name).assumingMemoryBound(to: Float.self)
            for i in 0..<m.length { p[i] = m.inside.contains(i) ? 1 : 0 }
        }
    }

    /// **The stage of the first of `candidates` whose socle and estimate hold under `budget`** (`plan` writes the plan of
    /// a band's positions, the tallest first), its residents read from `files`: the functions built at each plan's shapes
    /// and named by their bytes (`prefix-<hash>`), what Core AI will hold under each estimated from this Mac's readings
    /// (`Stage.estimates`: its own, or another's scaled by the positions a band covers), the soberest where none was read
    /// (`Stage.choose`). A stage's reading is kept at its close: the next decoding climbs on it.
    public static func open(_ name: String, prefix: String, candidates: [Int], files: [SafeTensors], residents: [Weights.Entry], budget: Int,
                            plan: (Int) -> Bands) async throws -> BandStage {
        guard let device = MTLCreateSystemDefaultDevice() else { throw Failure("no Metal device") }
        let plans = candidates.map(plan)
        var modules: [String: Aicode.Module] = [:]
        var declarations: [Stage.Declaration] = [], named: [[Bands.Shaped: String]] = []
        for p in plans {
            var d = Stage.Declaration("\(name), bands of \(p.positions) positions")
            d.tensors = p.slabs.enumerated().map { Stage.Tensor("slab\($0.offset)", bytes: $0.element) }
                + p.smalls.map { Stage.Tensor($0.name, bytes: $0.bytes, $0.apart ? .apart : .heap) }
            d.residents = residents
            var assets: [Bands.Shaped: String] = [:]
            for (key, module) in try p.modules() {
                let asset = Functions.name(module, prefix: prefix)
                assets[key] = asset
                modules[asset] = module
                nonisolated(unsafe) let m = module
                d.functions[asset] = Stage.Function(asset, unit: .gpu) { m }
            }
            declarations.append(d)
            named.append(assets)
        }
        // A socle over B leaves the shorter bands to try (`Stage.plan` refuses with `OverBudget`); none holds, the
        // finest's number. Any other error (a tensor the file lacks) is not a band too tall: it goes up as it is.
        var holding: [(rank: Int, plan: Stage.Plan)] = [], refusal: OverBudget?
        for (rank, d) in declarations.enumerated() {
            do { holding.append((rank, try Stage.plan(d, files: files, budget: budget, floor: Memory.footprint()))) }
            catch let o as OverBudget { refusal = o }
        }
        guard !holding.isEmpty else { throw refusal! }
        let estimates = Stage.estimates(holding.map { (Stage.key($0.plan.declaration), plans[$0.rank].positions) })
        // Every form fused: `OverBudget` too, the finest's P its number (`Stage.choose`).
        let chosen = holding[try Stage.choose(holding.map(\.plan), estimates)].rank
        let stage = try await Stage(device, plan: holding.first { $0.rank == chosen }!.plan, files: files)
        let assets = named[chosen]
        let unread = holding.count == declarations.count && estimates.allSatisfy { $0 == .unknown }
        return BandStage(plans[chosen], stage, candidate: chosen, narrowed: chosen > 0 && !unread,
                         signatures: assets.mapValues { Block.Signature(modules[$0]!) }, assets: assets)
    }

    private func view(_ s: Bands.Spot) -> View {
        stage.view(s.tensor, s.shape, s.type, strides: Layout.padded(s.shape, s.type), bytes: s.offset)
    }

    /// Every call, one after the other; the marks `keeping` names read back as their calls are done, `[C, rows, W]` f32.
    public func run(keeping: Set<String> = []) async throws -> [String: [Float]] {
        var kept: [String: [Float]] = [:]
        let marks = plan.marks.filter { keeping.contains($0.name) }
        for (i, call) in plan.calls.enumerated() {
            let sig = signatures[call.function]!
            func bind(_ arg: String, _ shape: [Int], _ type: NDArray.ScalarType) -> View {
                if let s = call.spots[arg] { return view(s) }
                if let key = call.weights[arg] {
                    guard let p = stage.residents[key] else { preconditionFailure("\(call.function): no resident \(key) for \(arg)") }
                    return stage.view(p, shape)
                }
                if let t = call.tensors[arg] { return stage.view(t, shape, type, strides: Layout.padded(shape, type)) }
                preconditionFailure("\(call.function): nothing bound to \(arg)")
            }
            var inputs: [String: View] = [:]
            for (n, s, t) in sig.inputs { inputs[n] = bind(n, s, t) }
            let outputs = sig.outputs.map { n, s, t in (n, bind(n, s, t)) }
            try await Run.call(functions[call.function]!, stage.stream, inputs: inputs, outputs: outputs)
            for m in marks where m.call == i { kept[m.name] = values(m.plane) }
        }
        // Every function called: the reading kept now (a client may hold its stage until its process ends).
        stage.keep()
        return kept
    }

    /// The bytes of the stage's tensor `name`.
    public func pointer(_ name: String) -> UnsafeMutableRawPointer { stage.pointer(name) }

    /// **A plane's values**, `[C, rows, W]` f32, its rows' factors multiplied out under the guard.
    public func values(_ t: Bands.Plane) -> [Float] {
        let base = UnsafeRawPointer(stage.pointer(t.slab.map { "slab\($0)" } ?? "t\(t.id)"))
        let factors = t.factors ? UnsafeRawPointer(stage.pointer("f\(t.id)")) : nil
        let row = Layout.padded([1, t.channels, t.rho, t.width], t.element)[2], half = t.element == .float16
        var out = [Float](repeating: 0, count: t.channels * t.rows * t.width)
        out.withUnsafeMutableBufferPointer { o in
            for c in 0..<t.channels {
                for y in 0..<t.rows {
                    let (j, i) = (y / t.rho, y % t.rho), at = base + j * t.bandBytes + (c * t.rho + i) * row * Layout.element(t.element)
                    let f = factors.map { $0.advanced(by: j * Arena.page).assumingMemoryBound(to: Float.self)[i] } ?? 1
                    let to = o.baseAddress! + (c * t.rows + y) * t.width
                    if half { let h = at.assumingMemoryBound(to: Float16.self); for x in 0..<t.width { to[x] = Float(h[x]) * f } }
                    else { let s = at.assumingMemoryBound(to: Float.self); for x in 0..<t.width { to[x] = s[x] * f } }
                }
            }
        }
        return out
    }

    /// **`values` (`[C, rows, W]`) written into a plane**, under factors of 1, its carry `carry` if it has one (what a
    /// plane the CPU writes carries to the layers that read it).
    public func write(_ t: Bands.Plane, _ values: [Float], carry: [Float] = []) {
        precondition(values.count == t.channels * t.rows * t.width, "\(values.count) values for a plane of \(t.channels) × \(t.rows) × \(t.width)")
        let base = stage.pointer(t.slab.map { "slab\($0)" } ?? "t\(t.id)")
        let row = Layout.padded([1, t.channels, t.rho, t.width], t.element)[2], half = t.element == .float16
        values.withUnsafeBufferPointer { v in
            for c in 0..<t.channels {
                for y in 0..<t.rows {
                    let (j, i) = (y / t.rho, y % t.rho), at = base + j * t.bandBytes + (c * t.rho + i) * row * Layout.element(t.element)
                    let from = v.baseAddress! + (c * t.rows + y) * t.width
                    if half { let h = at.assumingMemoryBound(to: Float16.self); for x in 0..<t.width { h[x] = Float16(from[x]) } }
                    else { let s = at.assumingMemoryBound(to: Float.self); for x in 0..<t.width { s[x] = from[x] } }
                }
            }
        }
        if t.factors {
            let f = stage.pointer("f\(t.id)")
            for j in 0..<t.count { let p = (f + j * Arena.page).assumingMemoryBound(to: Float.self); for i in 0..<t.rho { p[i] = 1 } }
        }
        if !t.carry.isEmpty {
            precondition(carry.count == t.carry.reduce(1, *), "a carry of \(carry.count) values for \(t.carry)")
            // Read as the carry of a plane no call wrote: its first buffer.
            let p = stage.pointer("c\(t.id)#0").assumingMemoryBound(to: Float.self)
            let row = Layout.padded(t.carry, .float32)[max(t.carry.count - 2, 0)], width = t.carry.last!
            for (n, x) in carry.enumerated() { p[t.carry.count == 1 ? n : (n / width) * row + n % width] = x }
        }
    }
}
