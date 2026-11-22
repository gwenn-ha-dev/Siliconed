import Foundation
@testable import SiliCore
import Testing

/// **The band planner, on a network written by hand**: a plane `x` read by a 3×3 into `h` (a window with its rows each
/// side, the plane's carry), `h` read by an ×2 into `y`; no byte of a weight, no function built.
@Suite struct BandPlans {
    static func plan(rows: Int, width: Int = 64, positions: Int = 1 << 11) -> (Bands, Bands.Plane, Bands.Plane, Bands.Plane) {
        var planes: (Bands.Plane, Bands.Plane, Bands.Plane)!
        let f = Bands.Function("conv") { _ in Aicode.Module(arguments: []) }, up = Bands.Function("up") { _ in Aicode.Module(arguments: []) }
        let plan = Bands.surveyed(positions: positions, element: .float16, factors: true) { p in
            let x = p.plane(8, rows: rows, width: width, carry: [3, 4])
            let h = p.plane(8, rows: rows, width: width, carry: [3, 4])
            p.layer(f, reads: [Bands.Read("x", x, before: true, after: true, carry: "x_stats")], writes: [Bands.Write("y", h)], weights: ["kernel": "k"])
            let y = p.plane(8, rows: 2 * rows, width: 2 * width)
            p.layer(up, reads: [Bands.Read("x", h, over: 2, before: true, after: true)], writes: [Bands.Write("y", y)], weights: ["kernel": "k2"])
            planes = (x, h, y)
        }
        return (plan, planes.0, planes.1, planes.2)
    }

    /// A band is the tallest power of two whose output holds the positions, whatever the plane's rows.
    @Test func theHeightIsAPowerOfTwo() {
        #expect(Bands.height(width: 128, positions: 1 << 14) == 128)
        #expect(Bands.height(width: 104, positions: 1 << 14) == 128)
        #expect(Bands.height(width: 1024, positions: 1 << 14) == 16)
        #expect(Bands.height(width: 64, positions: 1 << 14) == 256)
        #expect(Bands.height(width: 1 << 15, positions: 1 << 14, least: 2) == 2)
    }

    /// **One function a layer whatever the plane's rows**: a taller plane is more calls of the same functions, its last
    /// band past the plane's end.
    @Test func theFunctionsDoNotMoveWithTheRows() {
        let shapes = [96, 100, 128, 200].map { rows -> Set<Bands.Shaped> in Set(Self.plan(rows: rows).0.calls.map(\.function)) }
        #expect(shapes.allSatisfy { $0 == shapes[0] })
        #expect(shapes[0].count == 2)
        let (short, tall) = (Self.plan(rows: 96).0, Self.plan(rows: 200).0)
        #expect(tall.callCount > short.callCount)
    }

    /// **A plan's extent names its calls the same at any size and height** (`Bands.extent`): the planes taller, wider or
    /// in taller bands, the same kin and arguments, their elements those of the calls.
    @Test func theExtentIsTheSameNetworkAtAnySize() {
        let small = Self.plan(rows: 100).0.extent("p", inFlight: 2), wide = Self.plan(rows: 200, width: 96).0.extent("p", inFlight: 2)
        let tall = Self.plan(rows: 100, positions: 1 << 12).0.extent("p", inFlight: 2)
        #expect(small.kin == wide.kin && small.kin == tall.kin && small.kin != Self.plan(rows: 100).0.extent("q", inFlight: 2).kin)
        #expect(Set(small.elements.keys) == ["conv#0 x", "conv#0 y out", "up#0 x", "up#0 y out"])
        // 2¹¹ positions over 64 columns: bands of 32 rows, the window's one row each side.
        #expect(small.elements["conv#0 x"] == 34 * 64 * 8 && small.elements["conv#0 y out"] == 32 * 64 * 8)
        // Twice the positions: every output twice, the windows less (their rows each side the same); the other way, the
        // ×2's window the largest ratio (8 + 2 rows of 16 + 2).
        #expect(tall.ratio(over: small) == 2 && small.ratio(over: tall) == 10.0 / 18)
    }

    /// A band's window is masked past either end of the plane, and an output's rows past its end are given as not valid.
    @Test func theMasksCutThePlanesEnds() {
        let (p, _, _, _) = Self.plan(rows: 100)
        let first = p.calls[0], last = p.calls[3]
        let r = first.geometry.rows
        #expect(r == 32 && 100 % r != 0)
        func inside(_ c: Bands.Call, _ name: String) -> Range<Int> { p.masks[c.spots[name]!.tensor]!.inside }
        #expect(inside(first, "x_mask") == 1 ..< r + 2)
        #expect(inside(last, "x_mask") == 0 ..< 100 - 3 * r + 1)
        #expect(inside(first, "y_valid") == 0 ..< r)
        #expect(inside(last, "y_valid") == 0 ..< 100 - 3 * r)
    }

    /// A layer's carry goes from band to band through two buffers in turn, from zeros; its reader takes the last one
    /// written.
    @Test func theCarryAlternates() {
        let (p, _, _, _) = Self.plan(rows: 100)
        let layer = p.calls.filter { $0.function.key == "conv" }
        #expect(layer.count == 4)
        #expect(layer[0].spots["y_carried"]!.tensor.hasPrefix("zeros-"))
        for (a, b) in zip(layer, layer.dropFirst()) {
            #expect(b.spots["y_carried"]!.tensor == a.spots["y_carry"]!.tensor)
            #expect(a.spots["y_carry"]!.tensor != b.spots["y_carry"]!.tensor)
        }
    }

    /// An ×2 reads half its band's rows; every read and write lies on its plane's bands, inside its slab.
    @Test func everyViewLiesOnItsPlane() {
        let (p, x, h, y) = Self.plan(rows: 100)
        for t in [x, h, y] { #expect(t.coverage >= t.rows && t.coverage % t.rho == 0) }
        let up = p.calls.first { $0.function.key == "up" }!
        #expect(up.geometry.windows["x"]!.rows * 2 == up.geometry.rows)
        var sizes: [String: Int] = [:]
        for (i, bytes) in p.slabs.enumerated() { sizes["slab\(i)"] = bytes }
        for small in p.smalls { sizes[small.name] = small.bytes }
        for call in p.calls {
            for (_, s) in call.spots {
                let end = s.offset + Layout.paddedBytes(s.shape, s.type)
                #expect(end <= sizes[s.tensor]!, "\(s.tensor) to \(end)")
            }
        }
        #expect(p.weights == ["k", "k2"])
    }

    /// A function's arguments, its windows' and outputs' in one order: its asset's name holds from one plan to the next.
    @Test func theArgumentsAreInOneOrder() {
        let g = Self.plan(rows: 100).0.calls[0].geometry
        let names = Bands.arguments(g).map(\.0)
        #expect(names == ["x0", "x0_factor", "x_before", "x_before_factor", "x_after", "x_after_factor", "x_mask", "x_stats", "y_valid", "y_carried"])
        #expect(Bands.flat([128, 3, 3, 3], .f16) == [108, 32])
        #expect(Bands.flat([512, 128, 3, 3], .f16) == [512, 1152])
    }
}

/// **The erasure, on folders of its own**: every asset of the folder erased and nothing else, the specializations of
/// the processes named erased build by build, another process's left alone.
@Suite struct FunctionsErase {
    @Test func theAssetsAndTheNamedSpecializationsGo() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("erase-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        func file(_ path: String, _ bytes: Int) throws {
            let url = root.appendingPathComponent(path)
            try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(count: bytes).write(to: url)
        }
        let folder = root.appendingPathComponent("functions")
        try file("functions/a-00.aimodel/main.hash", 5000)
        try file("functions/b-01.aimodel/main.hash", 5000)
        try file("functions/notes.txt", 100)
        try file("coreai-cache/26A1/worker/h0/s0/model", 20000)
        try file("coreai-cache/26B2/worker/h1/s0/model", 20000)
        try file("coreai-cache/26B2/gone/h2/s0/model", 20000)
        try file("coreai-cache/26B2/another/h3/s0/model", 20000)
        #expect(Functions.specializations(of: "worker", under: root).count == 2)
        let before = Functions.weight(folder, specializationsOf: ["worker", "gone"], under: root)
        #expect(before >= 70000)
        let assets = Functions.erase(folder), specializations = Functions.eraseSpecializations(of: ["worker", "gone"], under: root)
        #expect(assets >= 10000 && specializations >= 60000 && assets + specializations == before)
        #expect(try fm.contentsOfDirectory(atPath: folder.path) == ["notes.txt"])
        #expect(Functions.weight(folder, specializationsOf: ["worker", "gone"], under: root) == 0)
        #expect(fm.fileExists(atPath: root.appendingPathComponent("coreai-cache/26B2/another/h3/s0/model").path))
    }
}
