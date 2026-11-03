import Foundation
import Metal
import Synchronization

/// **The tile of a `matmul2d` product: chosen by this Mac's map, never by a chip's name.** One threadgroup computes `rows × columns`
/// of C on `simdgroups` simdgroups, K read from device memory. The kernels `gpu/bf16` (fp32 activation × bf16 weight, `TensorGEMM`),
/// `gpu/bf16act` (rounded activation × bf16 weight, `BFloatKernels`), `gpu/int8` (fp32 activation × int8 weight) and `gpu/int8act`
/// (rounded activation × int8 weight) each take the fastest of `candidates(for:)`, measured by the map's probes on this Mac and judged
/// like every route (`CapacityModel.tiles`).
///
///     map:    for each kernel, each candidate on the largest solo product, three alternated rounds, the fastest → `choose` → `tiles`
///             and on the thin products (`CapacityModel.isThin`), the same, each shape's fastest summed → `choose` → `thinTiles`
///     engine: GEMM reads the report's tiles → TensorGEMM compiles each variant once → a product takes its class's tile
///
/// What sets it apart — what a tile chosen by analogy would get wrong:
///
///   - **the speed of a tile is the machine's**: on an M1 Pro `matmul2d` runs on the shader cores and 64 × 32 on four simdgroups
///     was within 7 % of the best; on a GPU with neural accelerators a 128 × 128 tile on eight simdgroups runs the rounded operand
///     at twice the rate of 64 × 32, and an fp32 operand gains nothing from any tile. So the candidates are measured, and the
///     default — the tile every map before this one was measured with — stays what a Mac without a measured tile runs: the same bits;
///   - **the int8 tile keeps 64 columns**: beside the AMX the GPU computes only the first columns of C, and a tile wider than the
///     AMX's cut (64) would write into the AMX's. Its candidates differ in rows and simdgroups only. The rounded one (`int8act`) never
///     has the AMX beside it (the AMX reads the fp32 activation): its candidates are the bf16 ones', its standard the 64 × 64 the
///     library carried;
///   - **a wide product and a thin one want different tiles**: a 128-row tile on 32 rows computes four times what it keeps, and a short `k`
///     or `n` leaves the large tile's threadgroups idle (`bf16act` on an M5 Pro: ×1.8 on 1 024 × 3 840², ×0.4 on 2 048 × 256 × 8 192). The
///     class boundary is the model's own (`isThin`), so a tile and the line that times it always speak of the same products;
///   - **a tile changes the order of the sums, not their precision**: the destination is fp32 and accumulated in fp32 in every
///     candidate (`relaxed_precision` false); the probe judges each before timing it, and a refused compilation is skipped.
package struct MatmulTile: Hashable, Sendable, Codable, CustomStringConvertible {
    package var rows: Int
    package var columns: Int
    package var simdgroups: Int
    package init(_ rows: Int, _ columns: Int, simdgroups: Int) {
        self.rows = rows; self.columns = columns; self.simdgroups = simdgroups
    }

    /// `128x64/4`.
    package var description: String { "\(rows)x\(columns)/\(simdgroups)" }
    package init?(_ text: String) {
        let parts = text.split(separator: "/")
        guard parts.count == 2, let s = Int(parts[1]) else { return nil }
        let dims = parts[0].split(separator: "x")
        guard dims.count == 2, let r = Int(dims[0]), let c = Int(dims[1]), r > 0, c > 0, s > 0 else { return nil }
        self.init(r, c, simdgroups: s)
    }
    package init(from decoder: any Decoder) throws {
        let text = try decoder.singleValueContainer().decode(String.self)
        guard let t = MatmulTile(text) else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "tile « \(text) »"))
        }
        self = t
    }
    package func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer(); try c.encode(description)
    }

    /// The kernels whose tile the map chooses.
    package static let kernels: [CapacityModel.Kernel] = [.init(.gpu, .bf16), .init(.gpu, .bf16act), .init(.gpu, .int8), .init(.gpu, .int8act)]

    /// **The tile a kernel runs when the map says nothing**: the one the source carried before tiles were measured.
    package static func standard(for kernel: CapacityModel.Kernel) -> MatmulTile {
        kernel.form == .int8 || kernel.form == .int8act ? MatmulTile(64, 64, simdgroups: 4) : MatmulTile(64, 32, simdgroups: 4)
    }

    /// **What the map measures for a kernel**, the standard first. The same list on every Mac.
    package static func candidates(for kernel: CapacityModel.Kernel) -> [MatmulTile] {
        switch kernel.form {
        case .int8:
            return [MatmulTile(64, 64, simdgroups: 4), MatmulTile(128, 64, simdgroups: 4), MatmulTile(128, 64, simdgroups: 8),
                    MatmulTile(256, 64, simdgroups: 8)]
        case .int8act:
            return [MatmulTile(64, 64, simdgroups: 4), MatmulTile(128, 64, simdgroups: 4), MatmulTile(128, 128, simdgroups: 8),
                    MatmulTile(64, 32, simdgroups: 4)]
        default:
            return [MatmulTile(64, 32, simdgroups: 4), MatmulTile(128, 64, simdgroups: 4), MatmulTile(128, 128, simdgroups: 8),
                    MatmulTile(64, 64, simdgroups: 4)]
        }
    }

    /// **What a candidate must win to be chosen over one listed before it**, by class: the noise of the class between two maps of one Mac
    /// (an M5 Pro, 2/11/2026: the wide product's fastest-of-three 2–4 % apart, three tiles within 4 % of each other; the thin shapes' sums
    /// 10–50 % apart). The standard tile is listed first: within the margin it stays, and with it the bits of every map measured before.
    package static let margin = 0.05
    package static let thinMargin = 0.15
    /// Alternated rounds per candidate; each shape keeps its fastest.
    package static let rounds = 3

    /// Rounds a candidate in doubt and the fastest are measured again, alternated (`ambiguous`): a candidate's figures that straddle the
    /// margin are a question three more paired rounds settle.
    package static let extraRounds = 3

    /// **The choice, from each candidate's seconds** (the fast side of alternated rounds, or its sum over the thin shapes): the first candidate,
    /// in the list's order, within the margin of the fastest. The same Mac measured twice chooses the same tile wherever two candidates are
    /// within the noise of each other — a tile is bits, not only speed. No candidate timed: the standard one.
    ///
    /// **The gap is read in pairs, the margin is one for all**: with `rounds` (each candidate's seconds round by round, the candidates
    /// alternated within a round), a candidate's gap is the median over the rounds of its seconds over the fastest's *in the same round* —
    /// a disturbance lasting a round slows both sides of the ratio, where fast sides taken rounds apart compare two moments. The margin is
    /// the floor (`margin`, `thinMargin`) for every candidate alike: symmetric, and a candidate whose rounds scatter gains nothing by its
    /// scatter (it is measured again instead, `ambiguous`). An M5 Pro under a video call read 128 × 64 at 3.4–14.3 % over 128 × 128 round
    /// by round in one map and 11.6–14.7 % in the next: medians 7.0 and 12.0 %, both past 5 % — where a margin widened by each
    /// candidate's own scatter (10.8 % in the first) called it a tie in one map and a gain in the other.
    package static func choose<Candidate: Hashable>(_ seconds: [Candidate: Double], among candidates: [Candidate], standard: Candidate,
                                                margin: Double = MatmulTile.margin, rounds: [Candidate: [Double]] = [:]) -> Candidate {
        guard let gaps = gaps(seconds, rounds: rounds) else { return standard }
        return candidates.first { c in gaps[c].map { $0 <= margin } ?? false } ?? standard
    }

    /// **Each candidate's gap to the fastest** (the least fast side): the median of the paired ratios where both have rounds, else the
    /// ratio of the fast sides. `nil` when nothing was timed.
    package static func gaps<Candidate: Hashable>(_ seconds: [Candidate: Double], rounds: [Candidate: [Double]] = [:]) -> [Candidate: Double]? {
        guard let (fastest, best) = seconds.filter({ $0.value > 0 }).min(by: { $0.value < $1.value }) else { return nil }
        var out: [Candidate: Double] = [:]
        for (c, s) in seconds where s > 0 {
            if let mine = rounds[c], let ref = rounds[fastest], !mine.isEmpty, !ref.isEmpty, mine.allSatisfy({ $0 > 0 }), ref.allSatisfy({ $0 > 0 }) {
                out[c] = CapacityFit.median(zip(mine, ref).map { $0 / $1 }) - 1
            } else {
                out[c] = s / best - 1
            }
        }
        return out
    }

    /// **The candidates whose choice is in doubt**: listed before the chosen one, or the chosen one itself, whose paired ratios to the
    /// fastest fall on both sides of the margin — the rounds disagree on the answer. Measured again with the fastest (`extraRounds`).
    package static func ambiguous<Candidate: Hashable>(_ seconds: [Candidate: Double], among candidates: [Candidate], standard: Candidate,
                                                   margin: Double = MatmulTile.margin, rounds: [Candidate: [Double]]) -> [Candidate] {
        guard let (fastest, _) = seconds.filter({ $0.value > 0 }).min(by: { $0.value < $1.value }), let ref = rounds[fastest] else { return [] }
        let chosen = choose(seconds, among: candidates, standard: standard, margin: margin, rounds: rounds)
        var out: [Candidate] = []
        for c in candidates {
            if c != fastest, let mine = rounds[c] {
                let ratios = zip(mine, ref).map { $0 / $1 - 1 }
                if let lo = ratios.min(), let hi = ratios.max(), lo <= margin, hi > margin { out.append(c) }
            }
            if c == chosen { break }
        }
        return out
    }

    /// The threadgroups that cover `rows × columns` of C.
    func grid(rows m: Int, columns n: Int) -> MTLSize {
        MTLSize(width: (n + columns - 1) / columns, height: (m + rows - 1) / rows, depth: 1)
    }
    func threads(_ pipeline: MTLComputePipelineState) -> MTLSize {
        MTLSize(width: pipeline.threadExecutionWidth * simdgroups, height: 1, depth: 1)
    }

    // MARK: - The variants' source

    /// **The kernels of `kernel` at this tile**: `linear_kn` and `linear_nk` for the bf16 weights (`A` fp32 for `gpu/bf16`, bfloat for
    /// `gpu/bf16act`), `linear_int8` for the int8 ones (`A` fp32 for `gpu/int8`, bfloat for `gpu/int8act`) — the default library's
    /// kernels with the tile as a parameter.
    func source(for kernel: CapacityModel.Kernel) -> String {
        let a = kernel.form == .bf16act || kernel.form == .int8act ? "bfloat" : "float"
        let head = """
        #include <metal_stdlib>
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace metal;
        using namespace mpp::tensor_ops;

        template <typename Op, typename SA, typename SB, typename SC>
        inline void tile(Op op, SA a, SB b, SC c) {
            auto t = op.template get_destination_cooperative_tensor<SA, SB, float>();
            for (uint16_t i = 0; i < t.get_capacity(); ++i) { if (t.is_valid_element(i)) t[i] = 0; }
            op.run(a, b, t);
            t.store(c);
        }

        """
        func product(_ name: String, weight: String, transposed: Bool) -> String {
            """
            kernel void \(name)(device \(a) *a [[buffer(0)]], device \(weight) *w [[buffer(1)]], device float *c [[buffer(2)]],
                                constant uint3 &mkn [[buffer(3)]], uint2 g [[threadgroup_position_in_grid]]) {
                const int M = mkn.x, K = mkn.y, N = mkn.z;
                tensor<device \(a), dextents<int32_t, 2>, tensor_inline> A(a, dextents<int32_t, 2>(K, M));
                tensor<device \(weight), dextents<int32_t, 2>, tensor_inline> B(w, dextents<int32_t, 2>(\(transposed ? "K, N" : "N, K")));
                tensor<device float, dextents<int32_t, 2>, tensor_inline> C(c, dextents<int32_t, 2>(N, M));
                constexpr auto d = matmul2d_descriptor(\(rows), \(columns), static_cast<int>(dynamic_extent), false, \(transposed), false);
                matmul2d<d, execution_simdgroups<\(simdgroups)>> op;
                const int x = g.x * \(columns), y = g.y * \(rows);
                tile(op, A.slice(0, y), B.slice(\(transposed ? "0, x" : "x, 0")), C.slice(x, y));
            }

            """
        }
        if kernel.form == .int8 || kernel.form == .int8act { return head + product("linear_int8", weight: "int8_t", transposed: false) }
        return head + product("linear_kn", weight: "bfloat", transposed: false) + product("linear_nk", weight: "bfloat", transposed: true)
    }
}

/// **The pipelines of one kernel at one tile**: `byColumns`/`byRows` for a bf16 weight (`[k, n]`, `[n, k]`), `int8` for the int8 one.
struct TiledProducts {
    let tile: MatmulTile
    let byColumns: MTLComputePipelineState?
    let byRows: MTLComputePipelineState?
    let int8: MTLComputePipelineState?
}

/// **The tile each kernel of a `TensorGEMM` runs, and the variants compiled so far.** Set from the map (`GEMM.init`), or by the probe
/// that measures the candidates (`use(_:for:)`). A kernel no tile was set for runs its standard one, from the default library.
final class TileTable: @unchecked Sendable {
    private let device: MTLDevice
    private struct Slot: Hashable { let kernel: CapacityModel.Kernel; let thin: Bool }
    private let state = Mutex<(chosen: [Slot: TiledProducts], compiled: [String: TiledProducts])>(([:], [:]))
    init(device: MTLDevice) { self.device = device }

    /// The variant `kernel` runs on a product of that shape (`CapacityModel.isThin`), `nil` for its standard one.
    func products(_ kernel: CapacityModel.Kernel, rows: Int, k: Int, n: Int) -> TiledProducts? {
        let slot = Slot(kernel: kernel, thin: CapacityModel.isThin(rows: rows, k: k, n: n))
        return state.withLock { $0.chosen[slot] }
    }

    /// **Runs `kernel` at `tile` on its wide products, or on its thin ones, from now on**: the standard one forgets the variant; another is
    /// compiled once. False when this macOS's compiler refuses it — the kernel keeps what it ran.
    @discardableResult
    func use(_ tile: MatmulTile, for kernel: CapacityModel.Kernel, thin: Bool = false) -> Bool {
        let slot = Slot(kernel: kernel, thin: thin)
        if tile == MatmulTile.standard(for: kernel) {
            _ = state.withLock { $0.chosen.removeValue(forKey: slot) }
            return true
        }
        let key = "\(kernel) \(tile)"
        if let known = state.withLock({ $0.compiled[key] }) {
            state.withLock { $0.chosen[slot] = known }
            return true
        }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        options.mathMode = .safe
        guard let library = MetalCompile.library(device, tile.source(for: kernel), options, what: "matmul2d \(kernel) \(tile)") else { return false }
        func pipeline(_ name: String) -> MTLComputePipelineState? {
            library.makeFunction(name: name).flatMap { MetalCompile.pipeline(device, $0, what: "matmul2d \(kernel) \(tile) \(name)") }
        }
        let products: TiledProducts
        if kernel.form == .int8 || kernel.form == .int8act {
            guard let p = pipeline("linear_int8"), p.maxTotalThreadsPerThreadgroup >= p.threadExecutionWidth * tile.simdgroups else { return false }
            products = TiledProducts(tile: tile, byColumns: nil, byRows: nil, int8: p)
        } else {
            guard let kn = pipeline("linear_kn"), let nk = pipeline("linear_nk"),
                  min(kn.maxTotalThreadsPerThreadgroup, nk.maxTotalThreadsPerThreadgroup) >= kn.threadExecutionWidth * tile.simdgroups
            else { return false }
            products = TiledProducts(tile: tile, byColumns: kn, byRows: nk, int8: nil)
        }
        state.withLock { $0.compiled[key] = products; $0.chosen[slot] = products }
        return true
    }
}
