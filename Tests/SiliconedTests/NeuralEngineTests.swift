import Accelerate
import XCTest
@testable import Siliconed

/// **The ANE's programs, written without coremltools**: `coremldata.bin` byte for byte what coremltools 9
/// compiled for the same `matmul(x, w)` (fp16 in and out), at three shapes whose dimensions take
/// different varints; and the cut's rows.
final class NeuralEngineTests: XCTestCase {
    private func bytes(_ hex: String) -> Data {
        var data = Data()
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            data.append(UInt8(hex[index..<next], radix: 16)!)
            index = next
        }
        return data
    }

    private static let header = "f6010000090000000000000000000000000000000000000000000000070000000000000067656e6572696309"
        + "000000000000000000000000000000000000000000000000000000000000003c00000000000000"
    private static let trailer = "f6010000000000000000000000000000"

    func testDescriptionMatchesCoremltools() {
        let cases: [(rows: Int, k: Int, n: Int, payload: String)] = [
            (2048, 2048, 2048, "0a110a01781a0c2a0a0a0480108010109080040a110a01771a0c2a0a0a04801080101090800452110a01791a0c2a0a0a048010801010908004a20600"),
            (1536, 2048, 8192, "0a110a01781a0c2a0a0a04800c8010109080040a110a01771a0c2a0a0a04801080401090800452110a01791a0c2a0a0a04800c804010908004a20600"),
            (3072, 8192, 2048, "0a110a01781a0c2a0a0a0480188040109080040a110a01771a0c2a0a0a04804080101090800452110a01791a0c2a0a0a048018801010908004a20600"),
        ]
        for c in cases {
            XCTAssertEqual(NeuralEngine.description(rows: c.rows, k: c.k, n: c.n),
                           bytes(Self.header + c.payload + Self.trailer), "\(c.rows)×\(c.k)→\(c.n)")
        }
    }

    /// A one-byte dimension shortens the description: its length in the header follows.
    func testDescriptionLengthFollowsThePayload() {
        let data = NeuralEngine.description(rows: 64, k: 2048, n: 2048)
        let length = data[0x4b..<0x53].enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        XCTAssertEqual(length, data.count - 0x53 - 16)
        XCTAssertEqual(length, 58)
    }

    /// K cut within the ANE's limits: K ≤ 4 096 and a part's weight ≤ 1.5 · 2²⁴ values, in equal parts of a
    /// multiple of 64 when a few more give them.
    func testParts() {
        XCTAssertEqual(NeuralEngine.parts(k: 2048, n: 8192), 1)
        XCTAssertEqual(NeuralEngine.parts(k: 3840, n: 3840), 1)
        XCTAssertEqual(NeuralEngine.parts(k: 3840, n: 10240), 2, "39 M values: two of 1 920")
        XCTAssertEqual(NeuralEngine.parts(k: 6144, n: 6144), 2, "two parts of 3 072")
        XCTAssertEqual(NeuralEngine.parts(k: 8192, n: 2048), 2, "K ≤ 4 096")
        XCTAssertEqual(NeuralEngine.parts(k: 10240, n: 3840), 4, "three would be uneven: four of 2 560")
        XCTAssertEqual(NeuralEngine.parts(k: 12288, n: 4096), 3)
        XCTAssertEqual(NeuralEngine.parts(k: 16384, n: 6144), 4)
        XCTAssertEqual(NeuralEngine.parts(k: 4097, n: 64), 2, "no even split within reach: the fewest parts")
    }

    /// The engine holds what the plan gave it: the surfaces of the shapes whose rows it takes, a sum over the shapes, and nothing
    /// for a product it takes no row of.
    func testBytesFollowTheRowsThePlanGave() {
        XCTAssertEqual(NeuralEngine.bytes(products: []), 0)
        XCTAssertEqual(NeuralEngine.bytes(products: [(0, 2048, 2048)]), 0, "no ANE row: nothing held")
        let one = NeuralEngine.bytes(products: [(2048, 2048, 2048)])
        XCTAssertGreaterThan(one, 0)
        XCTAssertGreaterThan(NeuralEngine.bytes(products: [(2048, 2048, 2048), (1024, 2048, 2048)]), one, "a shape each: a sum")
        XCTAssertEqual(NeuralEngine.bytes(products: [(2048, 2048, 2048), (2048, 2048, 2048)]), one, "the same shape twice: one program")
        XCTAssertGreaterThan(NeuralEngine.bytes(products: [(4096, 2048, 2048)]), NeuralEngine.bytes(products: [(2048, 2048, 2048)]))
    }

    /// Every version takes the Neural Engine unless the user turns it off; the PNG says it only when the denoiser could take it,
    /// and a choice the map left without effect keys nothing.
    func testEveryVersionTakesTheEngineByDefaultAndThePNGSaysWhenItWasTaken() throws {
        for variant in Variant.allCases { XCTAssertTrue(variant.acceleratorsByDefault) }
        var render = Engine.Render(model: "krea2", prompt: "woman posing in a library", seed: 42, steps: 8, loras: [],
                                   loraSummary: nil, image: ImageRGB(pixels: [Float](repeating: 0, count: 3 * 4), height: 2, width: 2), evaluations: 8, spectral: 0, strength: nil,
                                   startStep: 0, startSigma: 1, timings: .init(), footprints: .init(), tokens: 1)
        XCTAssertNil(render.metadata["neural-engine"])
        XCTAssertEqual(Engine.Render.Recipe(metadata: render.metadata)?.neuralEngine, false)
        render.neuralEngine = true
        let recipe = try XCTUnwrap(Engine.Render.Recipe(metadata: PNG.text(try render.png())))
        XCTAssertTrue(recipe.neuralEngine)
        XCTAssertEqual(recipe.unreadable, [])
        XCTAssertFalse(EngineSettings.effective.plan.usesANE)
        XCTAssertEqual(RenderCache.numerics(.on(neuralEngine: true)), RenderCache.numerics())
    }

    // MARK: - the int8 program

    /// The int8 description is the fp16 one with the weight first and typed `INT8` (131 080: `88 80 08`), the program's own
    /// signature `(wq, x)`; its length in the header follows the payload.
    func testInt8DescriptionTypesTheWeightAsInt8() {
        let data = NeuralEngine.description(rows: 2048, k: 2048, n: 2048, int8: true)
        let bytes = [UInt8](data)
        func count(_ pattern: [UInt8]) -> Int {
            (0...(bytes.count - pattern.count)).filter { Array(bytes[$0..<($0 + pattern.count)]) == pattern }.count
        }
        XCTAssertEqual(count([0x88, 0x80, 0x08]), 1, "INT8, once: the weight")
        XCTAssertEqual(count([0x90, 0x80, 0x04]), 2, "FLOAT16, twice: the activation and the output")
        XCTAssertEqual(count(Array("wq".utf8)), 1)
        let length = bytes[0x4b..<0x53].enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        XCTAssertEqual(length, bytes.count - 0x53 - 16)
        XCTAssertNotEqual(data, NeuralEngine.description(rows: 2048, k: 2048, n: 2048))
        XCTAssertEqual(NeuralEngine.description(rows: 2048, k: 2048, n: 2048, int8: false), NeuralEngine.description(rows: 2048, k: 2048, n: 2048))
    }

    /// The text that was run on the M1 Pro: an int8 input, MIL's `dequantize` with the constant scalar 2⁻⁶, then the product.
    func testInt8ProgramDequantizesWithAConstantScale() {
        let text = NeuralEngine.programText(rows: 1024, k: 3840, n: 4096, int8: true)
        XCTAssertTrue(text.contains("tensor<int8, [3840, 4096]> wq"))
        XCTAssertTrue(text.contains("tensor<fp16, [1024, 3840]> x"))
        XCTAssertTrue(text.contains("fp16 sc = const()[name = string(\"sc\"), val = fp16(0.015625)]"))
        XCTAssertTrue(text.contains("dequantize(input = wq, scale = sc)"))
        XCTAssertTrue(text.contains("tensor<fp16, [1024, 4096]> y = matmul("))
        XCTAssertFalse(text.contains("cast("), "cast + mul runs at half the rate")
        XCTAssertEqual(NeuralEngine.programText(rows: 1024, k: 3840, n: 4096, int8: false), NeuralEngine.programText(rows: 1024, k: 3840, n: 4096))
    }

    /// The integers' limits: K ≤ 4 096 and 50 M values (50 MB) — the Compact maps' shapes (Qwen-Image-2.1 4 096 and 12 288,
    /// Krea 2 6 144 / 16 384, MOSS-Image 3 072 / 8 192) in the fewest equal parts of a multiple of the convrot group.
    func testInt8Parts() {
        XCTAssertEqual(NeuralEngine.maximumInt8Weight, 4096 * 12288)
        for (k, n, fp16, int8) in [(4096, 4096, 1, 1), (4096, 12288, 2, 1), (12288, 4096, 3, 3), (6144, 6144, 2, 2), (6144, 16384, 4, 2),
                                   (16384, 6144, 4, 4), (3072, 3072, 1, 1), (8192, 3072, 2, 2), (3072, 16384, 2, 1), (3840, 10240, 2, 1)] {
            XCTAssertEqual(NeuralEngine.parts(k: k, n: n), fp16, "fp16 \(k)×\(n)")
            XCTAssertEqual(NeuralEngine.parts(k: k, n: n, int8: true, alignment: 256), int8, "int8 \(k)×\(n)")
        }
        XCTAssertEqual(NeuralEngine.parts(k: 10240, n: 3840, int8: true, alignment: 256), 4, "three would be uneven: four of 2 560")
        XCTAssertEqual(NeuralEngine.parts(k: 4864, n: 64, int8: true, alignment: 256), 2, "no even cut on 256: the fewest parts (the engine declines)")
        XCTAssertFalse(NeuralEngine.readsInt8(k: 4864, n: 64, group: 256), "a part's bound inside a rotation group")
        XCTAssertFalse(NeuralEngine().accepts(rows: 256, k: 4864, n: 64, int8Group: 256))
        for (k, n) in [(4096, 4096), (4096, 12288), (12288, 4096), (6144, 6144), (6144, 16384), (16384, 6144), (3072, 3072), (8192, 3072), (3072, 16384)] {
            XCTAssertTrue(NeuralEngine.readsInt8(k: k, n: n, group: 256), "the Compact maps' \(k)×\(n)")
        }
        XCTAssertTrue(NeuralEngine().accepts(rows: 256, k: 4864, n: 64, int8Group: nil))
    }

    /// The memory need holds the integers' shapes for the products whose weight the ANE reads as published, and the fp16 ones for
    /// the others — exactly, not a bound over both: the int8 program's weight surface is a byte a value, its input surface the
    /// whole K when it is one part.
    func testBytesHoldTheShapesOfTheFormEachProductReads() {
        let rows = 2048, k = 4096, n = 12288
        let fp16 = NeuralEngine.bytes(products: [(rows, k, n)])
        let int8 = NeuralEngine.bytes(products: [], integers: [(rows, k, n, 256)])
        XCTAssertGreaterThan(int8, 0)
        XCTAssertNotEqual(int8, fp16, "two programs of different shapes")
        // Both forms of one product (the fp16 program for one weight, the integers' for another of its shape): the surfaces of each,
        // the per-column state once.
        let both = NeuralEngine.bytes(products: [(rows, k, n)], integers: [(rows, k, n, 256)])
        XCTAssertGreaterThan(both, max(fp16, int8))
        XCTAssertLessThanOrEqual(both, fp16 + int8)
        XCTAssertEqual(NeuralEngine.bytes(products: [], integers: [(0, k, n, 256)]), 0, "no ANE row: nothing held")
        // The integers' weight is a byte a value in one surface where the fp16 program holds two parts of one shape: the weight surface
        // is the same 50 MB, the input surface is the whole K — more by `rows · K/2 · 2` bytes, the rest equal.
        XCTAssertEqual(int8 - fp16, Arena.capacity(for: rows * k * 2) - Arena.capacity(for: rows * (k / 2) * 2))
    }

    /// `x · R` is the rotation `Widen.dequantizeRotated` applies to the weight, moved to the activation: for random integers,
    /// `(rotate(x) / √G) · (q · s)` equals `x · W` with `W` the widened weight, at every group.
    func testRotationMovesTheWeightsRotationToTheActivation() {
        for group in [4, 16, 64, 256] {
            let (k, n) = (512, 8)
            var state: UInt64 = 7
            func next() -> UInt64 { state = state &* 6364136223846793005 &+ 1442695040888963407; return state >> 33 }
            let q = (0..<(k * n)).map { _ in Int8(truncatingIfNeeded: Int(next() % 255) - 127) }
            let scales = (0..<n).map { Float(1 + $0) / 256 }
            let x = (0..<k).map { _ in Float(Int(next() % 2001) - 1000) / 500 }
            var w = [Float](repeating: 0, count: k * n)
            q.withUnsafeBytes { raw in
                scales.withUnsafeBytes { sc in
                    Widen.dequantizeRotated(raw.baseAddress!, rows: k, columns: n, group: group, scale: sc.baseAddress!, scaleType: .float32,
                                            layout: .columns(n), into: &w)
                }
            }
            var rotated = [Float](repeating: 0, count: k)
            x.withUnsafeBufferPointer { src in rotated.withUnsafeMutableBufferPointer { NeuralEngine.rotate(src.baseAddress!, into: $0.baseAddress!, count: k, group: group) } }
            let norm = 1 / Double(group).squareRoot()
            for j in 0..<n {
                var reference = 0.0, moved = 0.0
                for r in 0..<k {
                    reference += Double(x[r]) * Double(w[r * n + j])
                    moved += Double(rotated[r]) * norm * Double(q[r * n + j]) * Double(scales[j])
                }
                XCTAssertEqual(moved, reference, accuracy: 1e-4 * max(1, abs(reference)), "group \(group), column \(j)")
            }
        }
    }

    /// The route end to end, on a small product (a Gaussian activation, integers of the published weights' spread): the ANE's rows
    /// against the fp64 product of the activation with `(R·q)·s`, per sampled row, beside the fp16 route's on the widened weight —
    /// the integers' within 1.25 × the widened weight's, both under fp16's own rounding. Skipped where the ANE's program does not
    /// load (a Mac without one) and in a virtual machine: the CI's "Apple M1 (Virtual)" loads the program and runs it elsewhere than
    /// on a Neural Engine, 5.3·10⁻³ from the exact product on its fp16 route. The product is not smaller: under K = 1 024 the ANE's own sums show (4·10⁻³ on
    /// 256 × 512 × 256, fp16 route and int8 route alike).
    func testInt8RouteAgainstFloat64() throws {
        var virtual: Int32 = 0, size = MemoryLayout<Int32>.size
        try XCTSkipIf(sysctlbyname("kern.hv_vmm_present", &virtual, &size, nil, 0) == 0 && virtual == 1,
                      "a virtual machine has no Neural Engine of its own")
        let (rows, k, n, group) = (256, 2048, 512, 256)
        var state: UInt64 = 11
        func next() -> UInt64 { state = state &* 6364136223846793005 &+ 1442695040888963407; return state >> 33 }
        func unit() -> Double { (Double(next() % 1_000_000) + 0.5) / 1_000_000 }
        func gauss() -> Double { (-2 * log(unit())).squareRoot() * cos(2 * Double.pi * unit()) }
        let q = (0..<(k * n)).map { _ in Int8(max(-127, min(127, (gauss() * 34).rounded()))) }
        let scales = (0..<n).map { Float(1 + $0 % 7) / 1024 }
        let x = (0..<(rows * k)).map { _ in Float(gauss()) }
        let engine = NeuralEngine()
        var c8 = [Float](repeating: .nan, count: rows * n), c16 = c8
        let ok8 = q.withUnsafeBufferPointer { qp in scales.withUnsafeBufferPointer { sp in x.withUnsafeBufferPointer { xp in c8.withUnsafeMutableBufferPointer { cp in
            engine.product(a: xp.baseAddress!, int8: .init(values: qp.baseAddress!, scales: sp.baseAddress!, perColumn: true, group: group),
                           c: cp.baseAddress!, first: 0, rows: rows, k: k, n: n)
        } } } }
        try XCTSkipUnless(ok8, "the Neural Engine's int8 program did not run here (fallbacks \(engine.fallbacks))")
        var w = [Float](repeating: 0, count: k * n)
        q.withUnsafeBytes { raw in scales.withUnsafeBytes { sc in
            Widen.dequantizeRotated(raw.baseAddress!, rows: k, columns: n, group: group, scale: sc.baseAddress!, scaleType: .float32, layout: .columns(n), into: &w)
        } }
        let ok16 = x.withUnsafeBufferPointer { xp in w.withUnsafeBufferPointer { wp in c16.withUnsafeMutableBufferPointer { cp in
            engine.product(a: xp.baseAddress!, w: wp.baseAddress!, c: cp.baseAddress!, first: 0, rows: rows, k: k, n: n)
        } } }
        XCTAssertTrue(ok16)
        // The exact product of every 16th row, in fp64.
        let picked = Array(stride(from: 0, to: rows, by: 16))
        let xd = picked.flatMap { r in (0..<k).map { Double(x[r * k + $0]) } }, wd = w.map(Double.init)
        var exact = [Double](repeating: 0, count: picked.count * n)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(picked.count), Int32(n), Int32(k), 1, xd, Int32(k), wd, Int32(n), 0, &exact, Int32(n))
        func worst(_ c: [Float]) -> Double {
            picked.enumerated().map { i, r in
                var num = 0.0, den = 0.0
                for j in 0..<n { let e = exact[i * n + j], d = Double(c[r * n + j]) - e; num += d * d; den += e * e }
                return (num / den).squareRoot()
            }.max()!
        }
        let (e8, e16) = (worst(c8), worst(c16))
        XCTAssertLessThan(e16, 2e-3, "fp16 operands, sums of 2 048 terms")
        XCTAssertLessThan(e8, 1.25 * e16, "the integers are no further from the exact product than the fp16 copy of the widened weight")
        XCTAssertGreaterThan(NeuralEngine.int8Tally.withLock { $0 }, 0)
    }

    /// A non-finite in the activation sends the ANE's rows back (`false`, counted), on the integers' route as on the fp16 one:
    /// the rotation would spread it over its group, and the peak that sets σ is not a number.
    func testInt8RouteGivesBackANonFiniteActivation() throws {
        let (rows, k, n, group) = (256, 2048, 256, 256)
        let q = [Int8](repeating: 3, count: k * n), scales = [Float](repeating: 0.01, count: n)
        var x = [Float](repeating: 0.5, count: rows * k)
        let engine = NeuralEngine()
        var c = [Float](repeating: 0, count: rows * n)
        func run() -> Bool {
            q.withUnsafeBufferPointer { qp in scales.withUnsafeBufferPointer { sp in x.withUnsafeBufferPointer { xp in c.withUnsafeMutableBufferPointer { cp in
                engine.product(a: xp.baseAddress!, int8: .init(values: qp.baseAddress!, scales: sp.baseAddress!, perColumn: true, group: group),
                               c: cp.baseAddress!, first: 0, rows: rows, k: k, n: n)
            } } } }
        }
        try XCTSkipUnless(run(), "the Neural Engine's int8 program did not run here")
        let before = engine.fallbacks
        x[17 * k + 1000] = .infinity
        XCTAssertFalse(run())
        XCTAssertEqual(engine.fallbacks, before + 1)
        x[17 * k + 1000] = 0.5
        XCTAssertTrue(run(), "the engine is not left in a state")
    }
}
