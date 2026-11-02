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
