import XCTest
@testable import Siliconed

/// **MOSS-Image's DiT, without a byte of weights**: the scheduler's sigmas against the reference's
/// (`MOSSImageScheduler.set_timesteps`, torch 2.14 on this machine), the four-axis RoPE's positions and
/// rotation, SITU against its definition (the kernel against double), and the DiT's arena and need against
/// their shape functions. The evaluation itself is judged against the oracle (`siliconed check moss-image dit`).
final class MossImageTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-moss-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    // MARK: - the sigmas

    /// The recipe's 8 steps are values of the training grid: 0.968, not the shifted `linspace`'s 0.9679.
    func testTheEightStepsAreTheTrainingGridsValues() {
        let sigmas = MossImageSchedule.sigmas(steps: 8)
        XCTAssertEqual(sigmas, [1, 0.968, 0.926, 0.871, 0.791, 0.671, 0.46300002932548523, 0.024, 0])
        // Timesteps are σ·1000 in fp32: 463.00003, not 463.
        XCTAssertEqual(MossImageSchedule.timesteps(sigmas), [1000, 968, 926, 871, 791, 671, 463.0000305175781, 24])
    }

    /// All of the 50-step schedule, to the bit (torch's values; the official recipe's).
    func testFiftyStepsToTheBit() {
        let reference: [Float] = [
            1.0, 0.9959999918937683, 0.9919999837875366, 0.9869999885559082, 0.9829999804496765, 0.9779999852180481,
            0.9729999899864197, 0.9679999947547913, 0.9629999995231628, 0.9570000171661377, 0.9520000219345093,
            0.9459999799728394, 0.9390000104904175, 0.9329999685287476, 0.9259999990463257, 0.918999969959259,
            0.9120000004768372, 0.9049999713897705, 0.8970000147819519, 0.8880000114440918, 0.8799999952316284,
            0.8709999918937683, 0.8610000014305115, 0.8510000109672546, 0.8399999737739563, 0.8289999961853027,
            0.8169999718666077, 0.8050000071525574, 0.7910000085830688, 0.7770000100135803, 0.7619999647140503,
            0.7459999918937683, 0.7289999723434448, 0.7109999656677246, 0.6919999718666077, 0.6710000038146973,
            0.6480000019073486, 0.6229999661445618, 0.597000002861023, 0.5679999589920044, 0.5359999537467957,
            0.5009999871253967, 0.46300002932548523, 0.42100003361701965, 0.37300002574920654, 0.32100000977516174,
            0.26100000739097595, 0.19300000369548798, 0.11500000208616257, 0.024000000208616257, 0.0]
        XCTAssertEqual(MossImageSchedule.sigmas(steps: 50).map(\.bitPattern), reference.map(\.bitPattern))
    }

    /// A dense schedule has fewer σ than steps: consecutive duplicates are removed after the snapping —
    /// 250 steps give 246 evaluations, 1000 give 615 (torch's counts).
    func testADenseScheduleLosesItsDuplicates() {
        for (steps, evaluations, middle) in [(250, 246, Float(0.8289999961853027)), (500, 415, 0.781000018119812),
                                             (1000, 615, 0.6919999718666077), (1500, 739, 0.6299999952316284)] {
            let sigmas = MossImageSchedule.sigmas(steps: steps)
            XCTAssertEqual(sigmas.count - 1, evaluations, "\(steps) steps")
            XCTAssertEqual(sigmas[(evaluations + 1) / 2], middle, "\(steps) steps")
            XCTAssertEqual(sigmas[evaluations - 1], 0.024)
            XCTAssertEqual(sigmas[1], 0.999)
        }
    }

    func testEverySigmaIsOnTheTrainingGridAndTheyDescend() {
        let grid = Set(MossImageSchedule.trainingGrid.map(\.bitPattern))
        for steps in [1, 2, 3, 5, 13, 28, 100] {
            let sigmas = MossImageSchedule.sigmas(steps: steps)
            XCTAssertEqual(sigmas.last, 0)
            XCTAssertEqual(sigmas.first, 1)
            XCTAssertEqual(sigmas, sigmas.sorted(by: >), "\(steps)")
            XCTAssertEqual(Set(sigmas).count, sigmas.count, "\(steps): duplicates")
            XCTAssert(sigmas.dropLast().allSatisfy { grid.contains($0.bitPattern) }, "\(steps)")
        }
        XCTAssertEqual(MossImageSchedule.sigmas(steps: 1), [1, 0])
        // 1000 on 0.001 … 1: the grid, 0.001 apart.
        XCTAssertEqual(MossImageSchedule.trainingGrid.count, 1000)
        XCTAssertEqual(MossImageSchedule.trainingGrid.first, 0.001)
        XCTAssertEqual(MossImageSchedule.trainingGrid.last, 1)
    }

    /// `torch.linspace` on this machine fuses each value's multiply-add: the unfused formula misses 239 of
    /// the training grid's 1000 values.
    func testLinspaceFusesItsMultiplyAdd() {
        let fused = MossImageSchedule.linspace(1, 0.001, 1000)
        let step = (Float(0.001) - 1) / 999
        let unfused = (0..<1000).map { $0 < 500 ? 1 + step * Float($0) : Float(0.001) - step * Float(999 - $0) }
        XCTAssertGreaterThan(zip(fused, unfused).filter { $0 != $1 }.count, 100)
        XCTAssertEqual(fused.first, 1); XCTAssertEqual(fused.last, 0.001)
    }

    // MARK: - the RoPE

    /// Text rows at `(0, i, 0, 0)`; the image at `(0, T, h − (H−1)/2, w − (W−1)/2)`, centred, its frame axis the
    /// text length.
    func testPositionsHaveFourAxesAndTheImageIsCentred() {
        let rows = MossImageRope.coordinates(textRows: 3, tilesHigh: 2, tilesWide: 4)
        XCTAssertEqual(rows.count, 3 + 8)
        XCTAssertEqual(rows[0], [0, 0, 0, 0]); XCTAssertEqual(rows[2], [0, 2, 0, 0])
        XCTAssertEqual(rows[3], [0, 3, -0.5, -1.5])
        XCTAssertEqual(rows[4], [0, 3, -0.5, -0.5])
        XCTAssertEqual(rows[10], [0, 3, 0.5, 1.5])
        // Odd grid: the centre cell is the origin.
        let odd = MossImageRope.coordinates(textRows: 1, tilesHigh: 3, tilesWide: 3)
        XCTAssertEqual(odd[1 + 4], [0, 1, 0, 0])
    }

    /// The same 16 bands for the four axes: `1 / θ^(i/16)` in fp32, from 1 down to θ^(−15/16).
    func testTheBandsAreSharedByTheFourAxes() {
        let bands = MossImageRope.bands(headDim: 128, base: 10000)
        XCTAssertEqual(bands.count, 16)
        XCTAssertEqual(bands[0], 1)
        XCTAssertEqual(Double(bands[8]), 0.01, accuracy: 1e-9)
        XCTAssertEqual(Double(bands[15]), pow(10000, -15.0 / 16), accuracy: 1e-10)
    }

    /// The table is `[sin(64) | cos(64)]`, entry `axis·16 + band`, and the rotation pairs `(j, j + 64)`.
    func testTheTableAndTheHalfSplitRotation() {
        let (t, h, w, dh) = (2, 2, 3, 128)
        let rows = t + h * w
        var table = [Float](repeating: 0, count: rows * dh)
        table.withUnsafeMutableBufferPointer {
            MossImageRope.write(into: $0.baseAddress!, textRows: t, tilesHigh: h, tilesWide: w, headDim: dh, base: 10000)
        }
        let coordinates = MossImageRope.coordinates(textRows: t, tilesHigh: h, tilesWide: w)
        let bands = MossImageRope.bands(headDim: dh, base: 10000)
        for row in [0, 1, 2, 4, 7] {
            for axis in 0..<4 {
                for band in [0, 5, 15] {
                    let angle = Double(coordinates[row][axis] * bands[band])
                    XCTAssertEqual(Double(table[row * dh + axis * 16 + band]), sin(angle), accuracy: 2e-7)
                    XCTAssertEqual(Double(table[row * dh + 64 + axis * 16 + band]), cos(angle), accuracy: 2e-7)
                }
            }
        }
        // Row 0 sits at the origin: sin 0, cos 1 — it does not rotate.
        XCTAssertEqual(Array(table[0..<64]), [Float](repeating: 0, count: 64))
        XCTAssertEqual(Array(table[64..<128]), [Float](repeating: 1, count: 64))
        // The rotation, against `x·cos + cat(−b, a)·sin` computed in double, on two heads.
        var generator = SystemRandomNumberGenerator()
        let heads = 2
        let x = (0..<(rows * heads * dh)).map { _ in Float.random(in: -2...2, using: &generator) }
        var rotated = x
        table.withUnsafeBufferPointer { f in
            rotated.withUnsafeMutableBufferPointer {
                MossImageRope.apply($0.baseAddress!, freqs: f.baseAddress!, sequence: rows, heads: heads, headDim: dh)
            }
        }
        for s in 0..<rows { for head in 0..<heads { for j in 0..<64 {
            let base = (s * heads + head) * dh
            let a = Double(x[base + j]), b = Double(x[base + 64 + j])
            let sn = Double(table[s * dh + j]), cs = Double(table[s * dh + 64 + j])
            XCTAssertEqual(Double(rotated[base + j]), a * cs - b * sn, accuracy: 1e-5)
            XCTAssertEqual(Double(rotated[base + 64 + j]), b * cs + a * sn, accuracy: 1e-5)
        } } }
    }

    // MARK: - SITU

    /// `4·tanh(g/4)·σ(g) · 25·tanh(h/25)`: zero when either side is, each side capped (gate ±4, value ±25).
    func testSituIsCappedOnBothSides() {
        XCTAssertEqual(MossImageExact.situ(hidden: 0, gate: 3), 0)
        XCTAssertEqual(MossImageExact.situ(hidden: 3, gate: 0), 0)
        XCTAssertEqual(MossImageExact.situ(hidden: 1e6, gate: 1e6), 4 * 25, accuracy: 1e-9)
        XCTAssertEqual(MossImageExact.situ(hidden: -1e6, gate: 1e6), -4 * 25, accuracy: 1e-9)
        // The gate is not a SiLU: very negative, it is 4·tanh(g/4)·σ(g) → 0 from below.
        XCTAssertLessThan(MossImageExact.situ(hidden: 1, gate: -3), 0)
        XCTAssertEqual(MossImageExact.situ(hidden: 1, gate: -60), 0, accuracy: 1e-20)
        // Small arguments: g·σ(g)≈g/2, h ≈ h.
        XCTAssertEqual(MossImageExact.situ(hidden: 0.01, gate: 0.01), 0.01 * 0.01 * 0.5, accuracy: 1e-6)
    }

    /// The kernel against the definition in double, on a fused `[hidden | gate]` row (the main blocks) and on
    /// two buffers with the output over `hidden` (the text blocks).
    func testTheKernelIsTheDefinition() throws {
        let gemm = try GEMM(freezeCut: false, neuralEngine: false)
        let kernels = try MossImageKernels(device: gemm.device, queue: gemm.queue)
        let (rows, columns) = (37, 300)
        let page = Arena.alignment / 4
        func buffer(_ floats: [Float]) throws -> (MTLBuffer, [Float]) {
            let count = (floats.count + page - 1) / page * page
            let pages = Pages(bytes: count * 4)
            let pointer = pages.floats()
            for i in 0..<count { pointer[i] = i < floats.count ? floats[i] : 0 }
            let b = try gemm.wrap(UnsafeMutableRawPointer(pointer), bytes: count * 4, name: "t")
            withExtendedLifetime(pages) {}
            keep.append(pages)
            return (b, floats)
        }
        var generator = SystemRandomNumberGenerator()
        let fused = (0..<(rows * 2 * columns)).map { _ in Float.random(in: -30...30, using: &generator) }
        let (fb, _) = try buffer(fused)
        let (ob, _) = try buffer([Float](repeating: 0, count: rows * columns))
        kernels.situ(hidden: (fb, 0), gate: (fb, columns), into: (ob, 0), rows: rows, columns: columns,
                     strides: (2 * columns, 2 * columns, columns))
        let out = UnsafeBufferPointer(start: ob.contents().assumingMemoryBound(to: Float.self), count: rows * columns)
        var worst = 0.0
        for r in 0..<rows { for c in 0..<columns {
            let h = Double(fused[r * 2 * columns + c]), g = Double(fused[r * 2 * columns + columns + c])
            let expected = MossImageExact.situ(hidden: h, gate: g)
            worst = max(worst, abs(Double(out[r * columns + c]) - expected) / max(abs(expected), 1e-3))
        } }
        XCTAssertLessThan(worst, 5e-6)

        // Two buffers, in place over `hidden`.
        let hidden = (0..<(rows * columns)).map { _ in Float.random(in: -30...30, using: &generator) }
        let gates = (0..<(rows * columns)).map { _ in Float.random(in: -30...30, using: &generator) }
        let (hb, _) = try buffer(hidden), (gb, _) = try buffer(gates)
        kernels.situ(hidden: (hb, 0), gate: (gb, 0), into: (hb, 0), rows: rows, columns: columns, strides: (columns, columns, columns))
        let inPlace = UnsafeBufferPointer(start: hb.contents().assumingMemoryBound(to: Float.self), count: rows * columns)
        for i in 0..<(rows * columns) {
            let expected = MossImageExact.situ(hidden: Double(hidden[i]), gate: Double(gates[i]))
            XCTAssertEqual(Double(inPlace[i]), expected, accuracy: 5e-6 * max(abs(expected), 1e-3), "\(i)")
        }
    }
    private var keep: [Pages] = []

    // MARK: - the time path

    /// `[cos | sin]` of `t · 10⁴^(−j/128)`, the timestep as fp32 the scheduler hands.
    func testTheSinusoidIsCosThenSin() {
        let e = MossImageExact.sinusoid(timestep: 1000, dim: 256)
        XCTAssertEqual(e.count, 256)
        XCTAssertEqual(Double(e[0]), cos(1000.0), accuracy: 1e-7)        // j = 0: frequency 1
        XCTAssertEqual(Double(e[128]), sin(1000.0), accuracy: 1e-7)
        XCTAssertEqual(Double(e[127]), cos(1000 * pow(10000.0, -127.0 / 128)), accuracy: 1e-7)
        XCTAssertEqual(Double(e[255]), sin(1000 * pow(10000.0, -127.0 / 128)), accuracy: 1e-7)
        // The fp32 value of the timestep is what is used: 463.00003, not 463.
        XCTAssertNotEqual(MossImageExact.sinusoid(timestep: 463.00003, dim: 256)[0], MossImageExact.sinusoid(timestep: 463, dim: 256)[0])
    }

    // MARK: - the LoRA's modules

    func testALoRAReachesTheBlocksLinearsAndTheTimePathOnly() {
        for module in ["layers.0.proj.q", "layers.41.proj.ffn_up_gate", "layers.7.proj.ffn_down", "layers.3.proj.o",
                       "time_embedding.0", "time_embedding.2", "shared_block_modulation", "video_head_modulation"] {
            XCTAssertTrue(MossImageDiT.appliesLoRA(module), module)
        }
        for module in ["layers.0.norm1", "layers.0.q_norm", "text_embedder.proj", "text_embedder.blocks.0.qkv",
                       "video_embedder", "final_linear_video", "layers.x.proj.q", "layers.0.proj.z", "proj.q"] {
            XCTAssertFalse(MossImageDiT.appliesLoRA(module), module)
        }
        let config = MossImageDiT.Config(header: [:])
        let targets = MossImageDiT.loraTargets(config: config)
        XCTAssertEqual(targets.count, 252)   // 42 × 6: the DMD LoRA's 252 modules
        XCTAssertEqual(targets.filter { $0.name.hasSuffix("ffn_up_gate") }.first.map { [$0.k, $0.n] }, [3072, 16384])
        XCTAssertEqual(targets.filter { $0.name.hasSuffix("ffn_down") }.first.map { [$0.k, $0.n] }, [8192, 3072])
        XCTAssert(targets.allSatisfy { MossImageDiT.appliesLoRA($0.name) })
    }

    // MARK: - the arena and the need

    private func map(_ header: [String: Any]) throws -> String {
        let page = 16, dataStart = 1 << 16
        var full = header
        full["page"] = page; full["tensors"] = [String: Any](); full["order"] = [String]()
        let json = try JSONSerialization.data(withJSONObject: full)
        var bytes = Array("SILICON".utf8) + [3]
        for i in 0..<8 { bytes.append(UInt8((UInt64(json.count) >> (8 * UInt64(i))) & 0xff)) }
        bytes += json
        bytes += [UInt8](repeating: 0, count: dataStart + 16 - bytes.count)
        let path = root.appendingPathComponent("m-\(UUID()).silicon").path
        try Data(bytes).write(to: URL(fileURLWithPath: path))
        return path
    }

    private var tinyHeader: [String: Any] {
        ["kind": MossImageDiT.kind, "linear_weights_transposed": true,
         "config": ["dim": 64, "num_heads": 4, "ffn_dim": 96, "num_layers": 2, "text_dim": 48, "freq_dim": 32,
                    "latent_channels": 16, "rope_base": 10000.0]]
    }

    /// The DiT's arena, built, is its shape function — without and with a LoRA stack.
    func testTheArenaIsTheFormula() throws {
        let artifact = try Artifact(path: try map(tinyHeader))
        let config = MossImageDiT.Config(header: artifact.header)
        XCTAssertEqual(config.headDim, 16); XCTAssertEqual(config.bands, 2)
        for (h, w, text) in [(4, 6, 7), (8, 8, 20)] {
            let dit = try MossImageDiT(artifact: artifact, gridHeight: h, gridWidth: w, maxText: text)
            XCTAssertEqual(dit.arenaCapacity, MossImageDiT.arenaBytes(config: config, gridHeight: h, gridWidth: w, maxText: text, rank: 0))
        }
    }

    /// The published shapes, laid out as the forge lays them (`[input, output]`), on 16 KB pages: the
    /// largest tensor of a layer is `ffn_up_gate`, 3072 × 16384 in bf16.
    private func publishedHeader() -> [String: Any] {
        let (d, f) = (3072, 8192)
        let layer: [(String, [Int])] = [
            ("k_norm.weight", [128]), ("norm1.weight", [d]), ("norm2.weight", [d]), ("proj.ffn_down.weight", [f, d]),
            ("proj.ffn_up_gate.weight", [d, 2 * f]), ("proj.k.weight", [d, d]), ("proj.o.weight", [d, d]),
            ("proj.q.weight", [d, d]), ("proj.v.weight", [d, d]), ("q_norm.weight", [128]),
        ]
        var header = MemoryFixtures.table(layers: 42, prefix: MossImageDiT.layerPrefix, tensors: layer, bytesPerValue: 2)
        header["config"] = [String: Any]()   // the published defaults
        return header
    }

    func testTheDenoisingPhaseIsComputedFromTheConstructorsFunctions() throws {
        let header = publishedHeader()
        let config = MossImageDiT.Config(header: header)
        XCTAssertEqual([config.dim, config.heads, config.headDim, config.hidden, config.layers, config.bands], [3072, 24, 128, 8192, 42, 16])
        XCTAssertEqual(Artifact.largestRun(header: header, prefixes: (0..<42).map(MossImageDiT.layerPrefix), fileSize: .max),
                       3072 * 16384 * 2)   // ffn_up_gate
        var peaks: [Int] = []
        for (w, h) in [(512, 512), (1024, 1024), (1024, 1536)] {
            var request = MemoryNeed.Request(card: .krea2, width: w, height: h, promptTokens: 25)
            request.threads = 6
            let phase = try XCTUnwrap(MemoryNeed.mossImageDenoising(request, textRows: 25, header: header))
            XCTAssertEqual(phase.kind, .denoising)
            XCTAssertEqual(phase.bytes("DiT arena"),
                           MossImageDiT.arenaBytes(config: config, gridHeight: h / 16, gridWidth: w / 16, maxText: 25, rank: 0))
            XCTAssertEqual(phase.bytes("map staging"), TailStream.unitSlots * 3072 * 16384 * 2)
            peaks.append(phase.bytes)
        }
        XCTAssertEqual(peaks, peaks.sorted()); XCTAssertEqual(Set(peaks).count, 3)
        // The arena at 1024² and 25 text rows, slice by slice: 4121 rows of 3072, the fused feed-forward 16384 wide.
        let n = 4096 + 25
        func pages(_ floats: [Int]) -> Int { floats.reduce(0) { $0 + Arena.capacity(for: $1 * 4) } }
        XCTAssertEqual(MossImageDiT.arenaBytes(config: config, gridHeight: 64, gridWidth: 64, maxText: 25, rank: 0),
                       pages(Array(repeating: n * 3072, count: 7) + [n * 16384, n * 8192, 8 * 3072, 3072 * 16384, 4096 * 128,
                                                                     4096 * 128, 25 * 4096, n * 128]))
    }

    /// **A LoRA costs three things in the DiT**: `mid [n, r]`, the cache of the 252 modules (the DMD LoRA's
    /// rank 128, fp32: 1.19 GB) and, with two LoRAs, one `down` staged.
    func testALoRAIsCharged() throws {
        let header = publishedHeader()
        var request = MemoryNeed.Request(card: .krea2, width: 1024, height: 1024, promptTokens: 25, loraRanks: [128])
        request.threads = 6
        let one = try XCTUnwrap(MemoryNeed.mossImageDenoising(request, textRows: 25, header: header))
        let none = try XCTUnwrap(MemoryNeed.mossImageDenoising(
            MemoryNeed.Request(card: .krea2, width: 1024, height: 1024, promptTokens: 25), textRows: 25, header: header))
        XCTAssertGreaterThan(one.bytes("LoRA cache"), 1_180_000_000)
        XCTAssertLessThan(one.bytes("LoRA cache"), 1_200_000_000)
        XCTAssertEqual(one.bytes("LoRA down staging"), 0)
        XCTAssertGreaterThan(one.bytes, none.bytes + one.bytes("LoRA cache"))
        request.loraRanks = [128, 32]
        let two = try XCTUnwrap(MemoryNeed.mossImageDenoising(request, textRows: 25, header: header))
        XCTAssertEqual(two.bytes("LoRA down staging"), LoRA.downBytes(k: 8192, rank: 128))
    }
}
