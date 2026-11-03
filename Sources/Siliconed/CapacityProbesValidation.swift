import Accelerate
internal import AMXLegacy
import Foundation
import Metal
import MetalPerformanceShadersGraph

// The model on products it never saw (`CapacityProbes.run(validate:)`): a probe judged against a long measure.

extension Diagnostic.CapacityProbes.Context {
    // MARK: Validation

    /// **The model on products it never saw**: each kernel alone, then each set at the rows the model's own split gives,
    /// against a long measure (fifteen runs after five warm-ups; a set's, the trial's steady median).
    func validate(_ model: CapacityModel) -> [Check] {
        restoreB()
        var checks: [Check] = []
        func long(_ body: () -> Bool) -> Double? {
            for _ in 0..<5 { guard body() else { return nil } }
            var t: [Double] = []
            for _ in 0..<15 { guard let s = CapacityProbes.timed(body) else { return nil }; t.append(s) }
            return Fit.median(t)
        }
        let g = Kernel(.gpu, .fp32), x = Kernel(.amx, .fp32), a = Kernel(.ane, .fp16), b = Kernel(.gpu, .bf16)
        for (m, k, n) in CapacityProbes.validationShapes.map({ ($0.m, $0.k, $0.n) }) {
            let flop = 2.0 * Double(m) * Double(k) * Double(n)
            var runs: [(Kernel, () -> Bool)] = [(g, { self.gpu(m, k, n) }), (x, { self.amx(m, k, n) })]
            if let e = neural.neuralEngine { runs.append((a, { self.ane(e, rows: m, k, n) })) }
            if alone.tensorBF16 {
                runs.append((b, { _ = self.alone.linear(a: self.aloneB.a, bf16: self.bf16, c: self.aloneB.c, m: m, k: k, n: n, weightIsTransposed: true); return true }))
            }
            for (kernel, body) in runs {
                guard let line = model.lines[kernel], let t = long(body) else { continue }
                checks.append(Check(what: kernel.description, shape: [m, k, n], rows: [kernel.description: m], predicted: model.alone(kernel, rows: m, k: k, n: n) ?? line.seconds(flop), measured: t))
                log?(String(format: "  validate %@ %dx%dx%d  model %.2f ms  long %.2f ms  %+.1f %%", kernel.description, m, k, n,
                            checks.last!.predicted * 1e3, t * 1e3, checks.last!.error * 100))
            }
        }
        for (m, k, n) in CapacityProbes.validationShapes.map({ ($0.m, $0.k, $0.n) }) {
            for set in [[g, x], [g, a], [g, x, a]] {
                let key = CapacityModel.key(set.map(\.engine))
                guard model.sharing[key] != nil, let t = trial(set, key: key, shape: (m, k, n), model: model, pass: 3),
                      !t.sides.isEmpty else { continue }
                var rows: [Kernel: Int] = [:]
                for s in t.sides { if let kernel = Kernel(s.kernel) { rows[kernel] = s.rows } }
                guard let p = model.seconds(m: m, k: k, n: n, rows: rows) else { continue }
                checks.append(Check(what: key, shape: [m, k, n], rows: Dictionary(uniqueKeysWithValues: t.sides.map { ($0.kernel, $0.rows) }),
                                    predicted: p, measured: t.wall))
                log?(String(format: "  validate %@ %dx%dx%d  model %.2f ms  measured %.2f ms  %+.1f %%", key, m, k, n, p * 1e3, t.wall * 1e3, checks.last!.error * 100))
            }
        }
        // The integers' route, on the same products: the ANE alone, then each GPU int8 kernel beside it.
        let a8 = Kernel(.ane, .int8)
        if let e = neural.neuralEngine, let line = model.lines[a8] {
            for (m, k, n) in CapacityProbes.validationShapes.map({ ($0.m, $0.k, $0.n) }) {
                guard let plan = int8Plan(k, n) else { continue }
                restoreB(); stageInt8(plan, k: k, n: n)
                guard let t = long({ self.aneInt8(e, plan, rows: m, k, n) }) else { continue }
                let flop = 2.0 * Double(m) * Double(k) * Double(n)
                checks.append(Check(what: a8.description, shape: [m, k, n], rows: [a8.description: m], predicted: model.alone(a8, rows: m, k: k, n: n) ?? line.seconds(flop), measured: t))
                log?(String(format: "  validate %@ %dx%dx%d  model %.2f ms  long %.2f ms  %+.1f %%", a8.description, m, k, n,
                            line.seconds(flop) * 1e3, t * 1e3, checks.last!.error * 100))
            }
            restoreB()
        }
        // The pairs on a weight read as published (the integers' beside each GPU kernel that reads them, the rounded bf16 one), each at the
        // rows the model's split gives.
        if neural.neuralEngine != nil {
            for pair in Self.integerPairs.map({ [Kernel(.gpu, $0), a8] }) + [Self.roundedPair] {
                let (g, e) = (pair[0], pair[1])
                let key = CapacityModel.key(pair)
                for (m, k, n) in CapacityProbes.validationShapes.map({ ($0.m, $0.k, $0.n) }) where model.sharing[key] != nil {
                    guard let t = trialPair(key: key, gpu: g, ane: e, shape: (m, k, n), model: model, pass: 3), !t.sides.isEmpty else { continue }
                    var rows: [Kernel: Int] = [:]
                    for s in t.sides { if let kernel = Kernel(s.kernel) { rows[kernel] = s.rows } }
                    guard let p = model.seconds(m: m, k: k, n: n, rows: rows) else { continue }
                    checks.append(Check(what: key, shape: [m, k, n], rows: Dictionary(uniqueKeysWithValues: t.sides.map { ($0.kernel, $0.rows) }),
                                        predicted: p, measured: t.wall))
                    log?(String(format: "  validate %@ %dx%dx%d  model %.2f ms  measured %.2f ms  %+.1f %%", key, m, k, n, p * 1e3, t.wall * 1e3, checks.last!.error * 100))
                }
            }
            restoreB()
        }
        return checks
    }
}
