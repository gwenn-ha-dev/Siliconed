import Foundation
import Testing
@testable import Siliconed

/// **The `matmul2d` tile the map chooses** (`MatmulTile`, `CapacityModel.tiles`): the same candidates on every Mac, the standard tile
/// first and what a map without tiles runs, the int8 candidates within the AMX's 64-column cut, and the tile carried from the probes'
/// measurements into the model, its JSON and the plan's identity. Pure: no GPU.
@Suite struct MatmulTileTests {
    typealias Kernel = CapacityModel.Kernel

    @Test func aTileReadsAsItIsWritten() throws {
        let t = MatmulTile(128, 64, simdgroups: 8)
        #expect(t.description == "128x64/8")
        #expect(MatmulTile("128x64/8") == t)
        #expect(MatmulTile("128x64") == nil && MatmulTile("0x64/4") == nil && MatmulTile("ax64/4") == nil)
        let data = try JSONEncoder().encode([t])
        #expect(String(decoding: data, as: UTF8.self) == "[\"128x64\\/8\"]")
        #expect(try JSONDecoder().decode([MatmulTile].self, from: data) == [t])
        let z = MatmulTile(128, 128, simdgroups: 8, morton: true)
        #expect(z.description == "128x128/8z" && MatmulTile("128x128/8z") == z && z != MatmulTile(128, 128, simdgroups: 8))
    }

    /// **Z order covers every tile once**: the line of codes holds the tiles' rows and columns in their bits, and the codes past C are left out
    /// by the kernel (the same decoding, here in Swift, as `zOrderSource`'s).
    @Test func zOrderCoversEveryTileOnce() {
        func compact(_ x: UInt32) -> UInt32 {
            var x = x & 0x5555_5555
            x = (x | (x >> 1)) & 0x3333_3333; x = (x | (x >> 2)) & 0x0f0f_0f0f
            x = (x | (x >> 4)) & 0x00ff_00ff; x = (x | (x >> 8)) & 0x0000_ffff
            return x
        }
        for (m, n) in [(4128, 3840), (4128, 10240), (128, 64), (300, 7000), (4096, 4096)] {
            let t = MatmulTile(128, 128, simdgroups: 8, morton: true)
            let (across, down) = ((n + 127) / 128, (m + 127) / 128)
            let xb = UInt32(MatmulTile.bits(across)), yb = UInt32(MatmulTile.bits(down)), paired = min(xb, yb)
            var seen = Set<[UInt32]>()
            for code in 0..<UInt32(t.grid(rows: m, columns: n).width) {
                let p = code & (paired == 0 ? 0 : (1 << (2 * paired)) - 1)
                var g = (compact(p), compact(p >> 1)), tail = code >> (2 * paired)
                if xb > paired { g.0 |= (tail & ((1 << (xb - paired)) - 1)) << paired; tail >>= xb - paired }
                if yb > paired { g.1 |= tail << paired }
                if g.0 < across && g.1 < down { #expect(seen.insert([g.0, g.1]).inserted) }
            }
            #expect(seen.count == across * down, "\(m)×\(n)")
        }
    }

    /// **Z order is a wide candidate of the tensor units' kernels only**, last, and never a thin one.
    @Test func zOrderIsTheLastWideCandidateOfTheTensorUnitsKernels() {
        for k in MatmulTile.kernels {
            let wide = MatmulTile.candidates(for: k), thin = MatmulTile.candidates(for: k, thin: true)
            #expect(thin.allSatisfy { !$0.morton })
            let units = [CapacityModel.Form.bf16act, .int8act, .int8x8].contains(k.form)
            #expect(wide == thin + (units ? [MatmulTile(128, 128, simdgroups: 8, morton: true)] : []), "\(k)")
        }
        let z = MatmulTile(128, 128, simdgroups: 8, morton: true)
        for k in [Kernel(.gpu, .bf16act), Kernel(.gpu, .int8x8)] {
            let source = z.source(for: k)
            #expect(source.contains("z_order(code, tiles)") && source.contains("[[buffer(7)]]"))
        }
        #expect(!MatmulTile(128, 128, simdgroups: 8).source(for: Kernel(.gpu, .bf16act)).contains("z_order"))
    }

    @Test func theStandardTileIsTheSourceOneAndTheFirstCandidate() {
        #expect(MatmulTile.standard(for: Kernel(.gpu, .bf16)) == MatmulTile(64, 32, simdgroups: 4))
        #expect(MatmulTile.standard(for: Kernel(.gpu, .bf16act)) == MatmulTile(64, 32, simdgroups: 4))
        #expect(MatmulTile.standard(for: Kernel(.gpu, .int8)) == MatmulTile(64, 64, simdgroups: 4))
        for k in MatmulTile.kernels {
            let c = MatmulTile.candidates(for: k)
            #expect(c.first == MatmulTile.standard(for: k))
            #expect(Set(c).count == c.count)
        }
    }

    /// **Three rounds on the wide product, two on the thin shapes**: the wide rounds are the map's noise, the thin ones only choose; a candidate
    /// in doubt is measured three more beside the fastest, whatever its class.
    @Test func theThinShapesTakeTwoRoundsAndTheWideThree() {
        #expect(MatmulTile.rounds(thin: false) == 3 && MatmulTile.rounds(thin: true) == 2 && MatmulTile.extraRounds == 3)
    }

    @Test func theInt8CandidatesStayWithinTheAMXsCut() {
        // Beside the AMX the GPU writes only its first columns: a wider tile would write into the AMX's.
        for t in MatmulTile.candidates(for: Kernel(.gpu, .int8)) { #expect(t.columns == 64) }
    }

    @Test func theSourceCarriesTheTile() {
        let t = MatmulTile(128, 128, simdgroups: 8)
        let bf16act = t.source(for: Kernel(.gpu, .bf16act))
        #expect(bf16act.contains("matmul2d_descriptor(128, 128,") && bf16act.contains("execution_simdgroups<8>"))
        #expect(bf16act.contains("device bfloat *a") && bf16act.contains("linear_kn") && bf16act.contains("linear_nk"))
        #expect(t.source(for: Kernel(.gpu, .bf16)).contains("device float *a"))
        let int8 = MatmulTile(128, 64, simdgroups: 4).source(for: Kernel(.gpu, .int8))
        #expect(int8.contains("linear_int8") && int8.contains("device int8_t *w") && !int8.contains("linear_kn"))
    }

    @Test func theModelTakesTheMeasuredTilesAndKeepsThemThroughItsJSON() throws {
        var m = CapacityFit.Measurements()
        m.tiles = ["gpu/bf16act": "128x128/8", "gpu/int8": "64x64/4", "gpu/bogus": "128x128/8", "gpu/bf16": "nonsense"]
        let model = CapacityFit.model(from: m)
        #expect(model.tiles == [Kernel(.gpu, .bf16act): MatmulTile(128, 128, simdgroups: 8), Kernel(.gpu, .int8): MatmulTile(64, 64, simdgroups: 4)])
        let back = try JSONDecoder().decode(CapacityModel.self, from: JSONEncoder().encode(model))
        #expect(back.tiles == model.tiles)
        #expect(back.rounded().tiles == model.tiles)
    }

    @Test func theChoiceIsTheFirstCandidateWithinTheMarginOfTheFastest() {
        let k = Kernel(.gpu, .bf16act)
        let c = MatmulTile.candidates(for: k), standard = MatmulTile.standard(for: k)
        let (a, b, d) = (c[0], c[1], c[2])
        // Nothing timed: the standard tile.
        #expect(MatmulTile.choose([:], among: c, standard: standard) == standard)
        // The standard within 5 % of the fastest stays: a tile is bits, and two times within the noise are one speed.
        #expect(MatmulTile.choose([a: 1.045, b: 1.0], among: c, standard: standard) == a)
        #expect(MatmulTile.choose([a: 1.06, b: 1.0], among: c, standard: standard) == b)
        // The thin class's margin is its own noise's.
        #expect(MatmulTile.choose([a: 1.12, b: 1.0], among: c, standard: standard, margin: MatmulTile.thinMargin) == a)
        // Beyond it, the fastest — or the first listed of those within the margin of it, whichever run is a hair faster.
        #expect(MatmulTile.choose([a: 2.0, b: 1.04, d: 1.0], among: c, standard: standard) == b)
        #expect(MatmulTile.choose([a: 2.0, b: 1.0, d: 1.02], among: c, standard: standard) == b)
        #expect(MatmulTile.choose([a: 2.0, d: 1.0], among: c, standard: standard) == d)
        // A refused candidate (judged wrong: seconds 0) is never chosen.
        #expect(MatmulTile.choose([a: 2.0, b: 0, d: 1.5], among: c, standard: standard) == d)
    }

    @Test func theThinTilesTravelBesideTheWideOnes() throws {
        var m = CapacityFit.Measurements()
        m.tiles = ["gpu/bf16act": "128x128/8"]
        m.thinTiles = ["gpu/bf16act": "64x32/4", "gpu/int8": "128x64/4"]
        let model = CapacityFit.model(from: m)
        #expect(model.thinTiles == [Kernel(.gpu, .bf16act): MatmulTile(64, 32, simdgroups: 4), Kernel(.gpu, .int8): MatmulTile(128, 64, simdgroups: 4)])
        let back = try JSONDecoder().decode(CapacityModel.self, from: JSONEncoder().encode(model))
        #expect(back.thinTiles == model.thinTiles && back.tiles == model.tiles)
        var other = model
        other.thinTiles[Kernel(.gpu, .bf16act)] = MatmulTile(128, 64, simdgroups: 4)
        #expect(CapacityPlan(model: model).identity != CapacityPlan(model: other).identity)
    }

    @Test func aTileIsPartOfThePlansIdentity() {
        var a = CapacityModel()
        a.lines[Kernel(.gpu, .fp32)] = .init(t0: 0.001, rate: 3e12)
        var b = a
        b.tiles[Kernel(.gpu, .bf16act)] = MatmulTile(128, 128, simdgroups: 8)
        #expect(CapacityPlan(model: a).identity != CapacityPlan(model: b).identity)
    }
}
