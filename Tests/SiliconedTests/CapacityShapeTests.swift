import Foundation
import Testing
@testable import Siliconed

/// The shape of a kernel's seconds, the overlap of a set, and the fits that read them from chains: pure, with truths that are known.
@Suite struct CapacityShapeTests {
    typealias F = CapacityFit
    typealias M = CapacityModel
    let gpu = M.Kernel(.gpu, .fp32), amx = M.Kernel(.amx, .fp32), ane = M.Kernel(.ane, .fp16)
    typealias Shape = (m: Int, k: Int, n: Int)
    static let designed: [Shape] = [(2048, 2048, 4096), (2048, 3840, 3840), (4096, 3840, 3840), (1024, 12288, 4096)]

    /// The solos of `kernel` at `shapes`, with the seconds `time` gives each.
    func solos(_ kernel: M.Kernel, _ shapes: [Shape], _ time: (Shape) -> Double) -> [F.Measurements.Solo] {
        var out: [F.Measurements.Solo] = []
        for s in shapes {
            let flop: Double = 2.0 * Double(s.m) * Double(s.k) * Double(s.n)
            out.append(F.Measurements.Solo(kernel: kernel.description, shape: [s.m, s.k, s.n], flop: flop, seconds: time(s)))
        }
        return out
    }

    // MARK: A kernel's seconds with its shape

    @Test func theWeightIsLiftedOnceWhateverTheRows() {
        let line = M.Line(t0: 0.001, rate: 5e12, perWeight: 1.3e-10)
        let few = line.seconds(rows: 256, k: 4096, n: 4096), many = line.seconds(rows: 2304, k: 4096, n: 4096)
        // The difference is the rows' FLOP alone; the lift (2.2 ms here) is in both.
        #expect(abs((many - few) - 2 * 2048.0 * 4096 * 4096 / 5e12) < 1e-12)
        #expect(abs(line.fixed(k: 4096, n: 4096) - (0.001 + 1.3e-10 * 4096 * 4096)) < 1e-12)
    }

    @Test func theRateFallsWithKBetweenItsKnots() {
        let line = M.Line(t0: 0, rate: 2.3e12, rateByK: [.init(k: 3840, rate: 2.3e12), .init(k: 12288, rate: 1.6e12)])
        #expect(line.rate(atK: 1024) == 2.3e12 && line.rate(atK: 3840) == 2.3e12)
        #expect(line.rate(atK: 16384) == 1.6e12 && line.rate(atK: 12288) == 1.6e12)
        let middle = line.rate(atK: 6860)            // between: the seconds per FLOP are linear in ln k
        #expect(middle < 2.3e12 && middle > 1.6e12)
        let f = (log(6860.0) - log(3840.0)) / (log(12288.0) - log(3840.0))
        #expect(abs(1 / middle - ((1 - f) / 2.3e12 + f / 1.6e12)) < 1e-18)
    }

    @Test func aPlainLineIsStillThePlainLine() {
        let line = M.Line(t0: 0.0008, rate: 2.5e12)
        #expect(abs(line.seconds(rows: 512, k: 2048, n: 2048) - line.seconds(2 * 512.0 * 2048 * 2048)) < 1e-15)
    }

    // MARK: The host's passes of the ANE

    /// The ANE's seconds with a known truth: `t₀ + w·k·n + h·rows·(k·r + n·parts) + 2·rows·k·n / R`, `parts` the engine's own cut of K.
    func hosted(_ shape: Shape, int8: Bool, t0: Double = 0.0011, w: Double = 2e-11, h: Double = 2.4e-10, rate: Double = 9e12) -> Double {
        let p = NeuralEngine.parts(k: shape.k, n: shape.n, int8: int8, alignment: int8 ? Widen.largestRotationGroup : 64)
        let elements = Double(shape.m) * (Double(shape.k) * (int8 && p > 1 ? 2 : 1) + Double(shape.n) * Double(p))
        return t0 + w * Double(shape.k * shape.n) + h * elements + 2 * Double(shape.m) * Double(shape.k) * Double(shape.n) / rate
    }

    @Test func aThinProductCostsMoreThanItsFlopSay() {
        let line = M.Line(t0: 0.001, rate: 9e12, perRow: 2.4e-10)
        let plain = M.Line(t0: 0.001, rate: 9e12)
        // Rows of 4 096 on a wide weight (n = 12 288) and a thin one (n = 1 536): the same rate pays the same per FLOP, the host's passes do not.
        let wide = line.rate(atK: 4096, n: 12288), thin = line.rate(atK: 4096, n: 1536)
        #expect(thin < 0.75 * wide && wide < plain.rate)
        // Where K is cut and the rows rotated (an int8 weight), the activation pass is paid twice.
        #expect(line.rate(atK: 8192, n: 3072, parts: 2, rotated: true) < line.rate(atK: 8192, n: 3072, parts: 2, rotated: false))
        #expect(plain.rate(atK: 4096, n: 1536, parts: 3, rotated: true) == plain.rate)
    }

    @Test func theModelReadsTheCutOfKFromTheEnginesOwn() {
        var m = M()
        let a8 = M.Kernel(.ane, .int8)
        m.lines[a8] = .init(t0: 0.0011, rate: 9e12, perRow: 2.4e-10)
        // 8 192 × 3 072 of integers: two parts of 4 096, the rows rotated twice.
        let t = m.alone(a8, rows: 2048, k: 8192, n: 3072)!
        let p = NeuralEngine.parts(k: 8192, n: 3072, int8: true, alignment: 256)
        #expect(p == 2)
        let elements = 2048.0 * (8192.0 * 2 + 3072.0 * Double(p))
        #expect(abs(t - (0.0011 + 2.4e-10 * elements + 2 * 2048.0 * 8192 * 3072 / 9e12)) < 1e-9)
    }

    @Test func theHostPassesAreToldFromTheLiftAndTheRateByThinAndCutProducts() throws {
        for int8 in [false, true] {
            let kernel = M.Kernel(.ane, int8 ? .int8 : .fp16)
            // Four shapes tell the host passes and the rate from the fixed cost (the weight's lift is left to them); six tell the lift too.
            let four: [Shape] = [(2048, 2048, 4096), (2048, 3840, 3840), (1024, 12288, 4096), (2048, 8192, 1536)]
            let six = four + [(4096, 3840, 3840), (1024, 6144, 1536)]
            for (shapes, lift) in [(four, 0.0), (six, 2e-11)] {
                let line = try #require(F.productLine(kernel: kernel, solos: solos(kernel, shapes) { hosted($0, int8: int8, w: lift) }))
                let h = try #require(line.perRow)
                #expect(abs(h / 2.4e-10 - 1) < 0.01 && abs(line.rate / 9e12 - 1) < 0.01 && abs(line.t0 / 0.0011 - 1) < 0.01, "\(kernel): \(line)")
                #expect((line.perWeight == nil) == (lift == 0) || abs((line.perWeight ?? 0) / lift - 1) < 0.01)
                // A product it never saw, thin and cut: the form predicts it where a plain line would be 30 % off.
                let unseen: Shape = (4096, 6144, 1536)
                let truth = hosted(unseen, int8: int8, w: lift)
                var m = M(); m.lines[kernel] = line
                #expect(abs(m.alone(kernel, rows: unseen.m, k: unseen.k, n: unseen.n)! / truth - 1) < 0.01)
            }
        }
    }

    @Test func fewerThanFourShapesFallBackOnTheThreeTermLine() throws {
        let kernel = ane
        let shapes: [Shape] = [(2048, 2048, 4096), (2048, 3840, 3840), (1024, 12288, 4096)]
        let line = try #require(F.productLine(kernel: kernel, solos: solos(kernel, shapes) { hosted($0, int8: false) }))
        #expect(line.perRow == nil)
    }

    @Test func theKeepByKIsInterpolatedAndClamped() {
        let s = M.Sharing(keep: [.amx: 0.9], keepByK: [.init(k: 3840, keep: ["amx": 0.9, "gpu": 0.97]), .init(k: 12288, keep: ["amx": 0.5])])
        #expect(s.keep(.amx, k: 2048) == 0.9 && s.keep(.amx, k: 16384) == 0.5)
        #expect(s.keep(.gpu, k: 12288) == 0.97)            // one knot only for this engine: its value everywhere
        #expect(s.keep(.ane, k: 4096) == 1)                // none: no loss
        let mid = s.keep(.amx, k: 6860)
        #expect(mid < 0.9 && mid > 0.5)
    }

    @Test func theNewFieldsSurviveJSON() throws {
        var m = M()
        m.lines[ane] = .init(t0: 0.0016, rate: 5.9e12, perWeight: 1.4e-10)
        m.lines[amx] = .init(t0: 0.001, rate: 2.2e12, rateByK: [.init(k: 2048, rate: 2.2e12), .init(k: 12288, rate: 1.6e12)])
        m.sharing["gpu+ane"] = .init(overhead: 0.00002, keep: [.gpu: 0.91, .ane: 0.75], sustained: 1.014,
                                     keepByK: [.init(k: 3840, keep: ["gpu": 0.92, "ane": 0.76]), .init(k: 12288, keep: ["gpu": 0.9, "ane": 0.7])])
        let back = try JSONDecoder().decode(M.self, from: JSONEncoder().encode(m))
        #expect(back == m)
        #expect(back.rounded(5).lines[ane]?.perWeight != nil && back.rounded(5).sharing["gpu+ane"]?.keepByK?.count == 2)
    }

    // MARK: The overlap

    func pair(_ gpuKeep: Double, _ aneKeep: Double) -> M {
        var m = M()
        m.lines[gpu] = .init(t0: 0, rate: 3.5e12)
        m.lines[ane] = .init(t0: 0, rate: 5e12)
        m.sharing["gpu+ane"] = .init(keep: [.gpu: gpuKeep, .ane: aneKeep])
        return m
    }

    @Test func theFirstToEndFreesThePartner() {
        let m = pair(0.9, 0.75)
        // GPU alone 10 ms, ANE alone 30 ms: the GPU ends at 10/0.9, the ANE has done 0.75 of that by then, and finishes the rest alone.
        let ends = m.overlap([gpu: 0.010, ane: 0.030], k: 4096)
        let gpuEnd = 0.010 / 0.9
        #expect(abs(ends[gpu]! - gpuEnd) < 1e-12)
        #expect(abs(ends[ane]! - (gpuEnd + (0.030 - 0.75 * gpuEnd))) < 1e-12)
        // Never longer than "both slowed all along", never shorter than alone.
        #expect(ends[ane]! < 0.030 / 0.75 && ends[ane]! > 0.030)
    }

    @Test func anEngineThatIsAloneGoesAtItsFullSpeed() {
        let m = pair(0.5, 0.5)
        #expect(abs(m.overlap([gpu: 0.02], k: 4096)[gpu]! - 0.02) < 1e-15)
    }

    @Test func threeEnginesLeaveBySets() {
        var m = pair(0.9, 0.75)
        m.lines[amx] = .init(t0: 0, rate: 2.2e12)
        m.sharing["gpu+amx+ane"] = .init(keep: [.gpu: 0.8, .amx: 0.7, .ane: 0.65])
        m.sharing["amx+ane"] = .init(keep: [.amx: 0.85, .ane: 0.9])
        let ends = m.overlap([gpu: 0.010, amx: 0.020, ane: 0.030], k: 4096)
        let t1 = 0.010 / 0.8                                      // the GPU ends first, in the full set
        let amxLeft = 0.020 - 0.7 * t1, aneLeft = 0.030 - 0.65 * t1
        let t2 = t1 + min(amxLeft / 0.85, aneLeft / 0.9)           // then the AMX and the ANE, as their pair
        #expect(abs(ends[gpu]! - t1) < 1e-12)
        #expect(abs(ends[amx]! - t2) < 1e-12 || abs(ends[ane]! - t2) < 1e-12)
        #expect(ends.values.max()! > t2)
    }

    @Test func aBalancedSplitEndsTogetherAtTheClosedForm() throws {
        let m = pair(0.9, 0.75)
        let s = try #require(m.split(m: 4096, k: 3840, n: 3840, kernels: [gpu, ane]))
        let ends = m.busy(m: 4096, k: 3840, n: 3840, rows: s.rows)
        #expect(abs(ends[gpu]! - ends[ane]!) / s.seconds < 0.08)       // within a unit of the ANE's rows (256 of 4 096)
        let w = 2.0 * 4096 * 3840 * 3840
        let tStar = w / (0.9 * 3.5e12 + 0.75 * 5e12)
        #expect(abs(s.seconds - tStar) / tStar < 0.03)
    }

    @Test func theBalanceKeepsTheFixedPartInsideTheLoss() throws {
        // With a fixed part f the balanced split solves f + V = κ·T (the whole seconds alone, scaled), not V = κ·(T − f).
        var m = pair(0.8, 0.8)
        m.lines[ane] = .init(t0: 0.002, rate: 5e12, perWeight: 1e-10)
        let s = try #require(m.split(m: 4096, k: 4096, n: 4096, kernels: [gpu, ane]))
        let solo = m.alone(ane, rows: s.rows[ane] ?? 0, k: 4096, n: 4096)!
        #expect(abs(solo / 0.8 - s.seconds) / s.seconds < 0.03)
    }

    @Test func theSustainedFactorIsTheSetsOwn() throws {
        var m = pair(0.9, 0.75)
        m.sharing["gpu+ane"]!.sustained = 1.05
        let rows = [gpu: 2048, ane: 2048]
        let cold = try #require(m.seconds(m: 4096, k: 4096, n: 4096, rows: rows))
        let hot = try #require(m.seconds(m: 4096, k: 4096, n: 4096, rows: rows, sustained: true))
        #expect(abs(hot / cold - 1.05) < 1e-9)
        // A kernel alone has none.
        let alone = try #require(m.seconds(m: 4096, k: 4096, n: 4096, rows: [gpu: 4096], sustained: true))
        #expect(alone == m.seconds(m: 4096, k: 4096, n: 4096, rows: [gpu: 4096]))
    }

    // MARK: The domain

    @Test func aKernelIsNotAskedForFewerRowsThanHalfOfWhatItWasMeasuredAt() throws {
        var m = pair(0.9, 0.75)
        m.lines[amx] = .init(t0: 0.001, rate: 2.2e12, rowsFrom: 512)
        #expect(m.alone(amx, rows: 511, k: 4096, n: 4096) == nil)
        #expect(m.alone(amx, rows: 512, k: 4096, n: 4096) != nil)
        #expect(m.seconds(m: 1056, k: 3840, n: 3840, rows: [gpu: 768, amx: 288]) == nil)       // the set is not timed outside a line's domain
        // The fit sets it: half of the smallest solo's rows, for the AMX and the ANE only.
        let truthAMX = M.Line(t0: 0.0012, rate: 2.3e12)
        let line = try #require(F.productLine(kernel: amx, solos: solos(amx, Self.designed) { truthAMX.seconds(rows: $0.m, k: $0.k, n: $0.n) }))
        #expect(line.rowsFrom == 512)
        let gpuLine = try #require(F.productLine(kernel: gpu, solos: solos(gpu, Self.designed) { truthAMX.seconds(rows: $0.m, k: $0.k, n: $0.n) }))
        #expect(gpuLine.rowsFrom == nil)
        // The split leaves out a kernel whose balanced share falls outside its domain.
        var n = pair(0.9, 0.75)
        n.lines[ane] = .init(t0: 0.0015, rate: 5.8e12, perWeight: 1.4e-10, rowsFrom: 512)
        let s = try #require(n.split(m: 1056, k: 3840, n: 10240, kernels: [gpu, ane]))
        #expect((s.rows[ane] ?? 0) == 0 || (s.rows[ane] ?? 0) >= 512)
    }

    @Test func aSetIsNotTimedBeyondTheKsItWasMeasuredAt() throws {
        var m = pair(0.9, 0.75)
        m.sharing["gpu+ane"] = .init(keep: [.gpu: 0.9, .ane: 0.75], kFrom: 1920, kTo: 7680)
        #expect(m.seconds(m: 4096, k: 3840, n: 3840, rows: [gpu: 2048, ane: 2048]) != nil)
        #expect(m.seconds(m: 4096, k: 12288, n: 3840, rows: [gpu: 2048, ane: 2048]) == nil)       // 12 288 > 2 × 3 840
        #expect(m.seconds(m: 4096, k: 12288, n: 3840, rows: [gpu: 4096]) != nil)                  // a kernel alone has no set
        // The fit sets it from the chains' k's.
        var truth = pair(0.9, 0.75)
        var fitted = M(); fitted.lines = truth.lines
        let rows = try #require(truth.split(m: 4096, k: 3840, n: 3840, kernels: [gpu, ane])).rows
        truth.sharing["gpu+ane"] = .init(keep: [.gpu: 0.9, .ane: 0.75])
        let fit = try #require(F.overlap(set: "gpu+ane", trials: [chain(truth, truth, set: "gpu+ane", shape: (4096, 3840, 3840), rows: rows)], model: fitted))
        #expect(fit.sharing.kFrom == 1920 && fit.sharing.kTo == 7680 && fit.sharing.keepByK == nil)
    }

    // MARK: The fits

    @Test func theANELiftAndTheAMXRateByKAreRecovered() throws {
        let truthANE = M.Line(t0: 0.0014, rate: 5.6e12, perWeight: 1.3e-10)
        let line = try #require(F.productLine(kernel: ane, solos: solos(ane, Self.designed) { truthANE.seconds(rows: $0.m, k: $0.k, n: $0.n) }))
        #expect(abs(line.t0 - 0.0014) < 1e-9 && abs(line.rate / 5.6e12 - 1) < 1e-9 && abs((line.perWeight ?? 0) / 1.3e-10 - 1) < 1e-9)

        let truthAMX = M.Line(t0: 0.0012, rate: 2.3e12, rateByK: [.init(k: 2048, rate: 2.05e12), .init(k: 3840, rate: 2.3e12), .init(k: 12288, rate: 1.6e12)])
        let amxLine = try #require(F.productLine(kernel: amx, solos: solos(amx, Self.designed) { truthAMX.seconds(rows: $0.m, k: $0.k, n: $0.n) }))
        #expect(abs(amxLine.t0 - 0.0012) < 1e-9)
        #expect(amxLine.rateByK?.map(\.k) == [2048, 3840, 12288])
        for (found, truth) in zip(amxLine.rateByK ?? [], truthAMX.rateByK ?? []) { #expect(abs(found.rate / truth.rate - 1) < 1e-9) }
        // On a k it never saw, the interpolation is the truth's (the seconds per FLOP are linear in ln k between the knots).
        #expect(abs(amxLine.seconds(rows: 1536, k: 8192, n: 4096) / truthAMX.seconds(rows: 1536, k: 8192, n: 4096) - 1) < 1e-9)
    }

    @Test func aFitWithTooFewShapesIsThePlainLine() {
        let shapes: [Shape] = [(1024, 3840, 3840), (2048, 3840, 3840), (4096, 3840, 3840)]
        let list = solos(ane, shapes) { 0.002 + 2.0 * Double($0.m) * Double($0.k) * Double($0.n) / 5e12 }
        let line = F.productLine(kernel: ane, solos: list)
        #expect(line != nil && line?.perWeight == nil && line?.rateByK == nil)             // one weight size: the lift is t₀
    }

    /// A chain of `rows` on `truth`'s κ, as the machine would give it: the sides' own clocks.
    func chain(_ model: M, _ kappa: M, set: String, shape: (Int, Int, Int), rows: [M.Kernel: Int]) -> F.Measurements.Trial {
        let (m, k, n) = shape
        let ends = kappa.busy(m: m, k: k, n: n, rows: rows)
        let wall = ends.values.max()!
        let sides = rows.sorted { $0.key.engine < $1.key.engine }.map {
            F.Measurements.Side(kernel: $0.key.description, rows: $0.value, solo: model.alone($0.key, rows: $0.value, k: k, n: n)!, within: ends[$0.key]!)
        }
        return .init(set: set, shape: [m, k, n], pass: 2, sides: sides, wall: wall)
    }

    @Test func theOverlapIsReadBackFromChainsBalancedOrNot() throws {
        var truth = pair(0.91, 0.75)
        truth.lines[amx] = .init(t0: 0, rate: 2.2e12)
        truth.sharing["gpu+amx"] = .init(keep: [.gpu: 0.97, .amx: 0.9])
        truth.sharing["amx+ane"] = .init(keep: [.amx: 0.86, .ane: 0.92])
        truth.sharing["gpu+amx+ane"] = .init(keep: [.gpu: 0.8, .amx: 0.7, .ane: 0.67])
        var fitted = M()
        fitted.lines = truth.lines
        for (set, kernels) in [("amx+ane", [amx, ane]), ("gpu+amx", [gpu, amx]), ("gpu+ane", [gpu, ane]), ("gpu+amx+ane", [gpu, amx, ane])] {
            // Three splits around the balance, so that the sides do not end together.
            let balance = try #require(truth.split(m: 4096, k: 3840, n: 3840, kernels: kernels)).rows
            var trials: [F.Measurements.Trial] = []
            for shift in [-256, 0, 256] {
                var rows = balance
                if let a = rows[ane] { rows[ane] = a + shift; let other = kernels.first { $0 != ane }!; rows[other] = rows[other]! - shift }
                else { rows[kernels[0]] = rows[kernels[0]]! + shift; rows[kernels[1]] = rows[kernels[1]]! - shift }
                trials.append(chain(truth, truth, set: set, shape: (4096, 3840, 3840), rows: rows))
            }
            let fit = try #require(F.overlap(set: set, trials: trials, model: fitted))
            fitted.sharing[set] = fit.sharing
            let want = truth.sharing[set]!.keep
            for (engine, keep) in want { #expect(abs(fit.sharing.keep[engine]! - keep) < 1e-9, "\(set) \(engine)") }
            #expect(fit.residuals.allSatisfy { abs($0) < 1e-9 })
        }
    }

    @Test func theKeepIsReadAtEachKAndInterpolated() throws {
        var truth = pair(0.9, 0.75)
        truth.sharing["gpu+ane"] = .init(keepByK: [.init(k: 3840, keep: ["gpu": 0.92, "ane": 0.76]), .init(k: 12288, keep: ["gpu": 0.88, "ane": 0.68])])
        var fitted = M()
        fitted.lines = truth.lines
        var trials: [F.Measurements.Trial] = []
        for shape in [(4096, 3840, 3840), (1536, 12288, 4096)] {
            let rows = try #require(truth.split(m: shape.0, k: shape.1, n: shape.2, kernels: [gpu, ane])).rows
            trials.append(chain(truth, truth, set: "gpu+ane", shape: shape, rows: rows))
        }
        let fit = try #require(F.overlap(set: "gpu+ane", trials: trials, model: fitted))
        #expect(fit.sharing.keepByK?.map(\.k) == [3840, 12288])
        fitted.sharing["gpu+ane"] = fit.sharing
        #expect(abs(fitted.sharing["gpu+ane"]!.keep(.ane, k: 3840) - 0.76) < 1e-9 && abs(fitted.sharing["gpu+ane"]!.keep(.ane, k: 12288) - 0.68) < 1e-9)
    }

    @Test func theModelFromMeasurementsFitsTheSmallestSetsFirst() throws {
        var truth = pair(0.91, 0.75)
        truth.lines[amx] = .init(t0: 0, rate: 2.2e12)
        truth.sharing["gpu+amx"] = .init(keep: [.gpu: 0.97, .amx: 0.9])
        truth.sharing["amx+ane"] = .init(keep: [.amx: 0.86, .ane: 0.92])
        truth.sharing["gpu+amx+ane"] = .init(keep: [.gpu: 0.8, .amx: 0.7, .ane: 0.67])
        var m = F.Measurements()
        for (kernel, line) in truth.lines { m.solos += solos(kernel, Self.designed) { line.seconds(rows: $0.m, k: $0.k, n: $0.n) } }
        var base = M(); base.lines = truth.lines
        for (set, kernels) in [("gpu+amx+ane", [gpu, amx, ane]), ("gpu+ane", [gpu, ane]), ("amx+ane", [amx, ane]), ("gpu+amx", [gpu, amx])] {
            let rows = try #require(truth.split(m: 4096, k: 3840, n: 3840, kernels: kernels)).rows
            m.trials.append(chain(truth, truth, set: set, shape: (4096, 3840, 3840), rows: rows))
        }
        let model = F.model(from: m)
        for (set, sharing) in truth.sharing {
            for (engine, keep) in sharing.keep { #expect(abs(model.sharing[set]!.keep[engine]! - keep) < 1e-6, "\(set) \(engine)") }
        }
    }
}
