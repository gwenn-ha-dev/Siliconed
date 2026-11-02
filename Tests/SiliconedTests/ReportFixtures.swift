import Foundation
@testable import Siliconed

/// **A report of schema 19 written in memory** — an M1 Pro's order of figures (the probes' three runs, `capacities`), no GPU, no clock:
/// measurements generated from lines whose answer is known, the model fitted on them by the code under test, the judge and the memory
/// of a clean run. The same figures on every Mac, the CI's included.
enum ReportFixtures {
    typealias Fit = CapacityFit
    typealias Kernel = CapacityModel.Kernel
    static let gpu = Kernel(.gpu, .fp32), amx = Kernel(.amx, .fp32), ane = Kernel(.ane, .fp16)

    /// The machine: an M1 Pro's inventory (14 of 16 GPU cores at 1 296 MHz → 4.645 TFLOP/s, 6 P + 2 E, `ane,t8020`).
    static func machine(inventory: Bool = true, build: String = "x+y") -> Diagnostic.Machine {
        Diagnostic.Machine(chip: "Apple M1 Pro", hardwareModel: "MacBookPro18,3", performanceCores: 6, efficiencyCores: 2, gpuCores: 14,
                           memoryBytes: 17_179_869_184, macOS: "27.0.1 (26A434)", version: "git 0123456789-dirty", build: build,
                           gpu: "Apple M1 Pro", gpuFamily: "apple7", gpuWorkingSetBytes: 12_713_115_648,
                           inventory: inventory ? .init(gpuSiliconCores: 16, gpuGen: 13, gpuCoreMask: [253, 253], gpuMaxMHz: 1296,
                                                        cpuLevels: [.init(name: "Performance", cores: 6, coresPerL2: 3),
                                                                    .init(name: "Efficiency", cores: 2, coresPerL2: 2)],
                                                        ane: .init(compatible: "ane,t8020", type: 96, subtype: 0)) : nil)
    }

    static func conditions(swap: Int = 0) -> Diagnostic.Conditions {
        .init(thermal: "nominal", lowPowerMode: false, onBattery: nil, swapUsedBytes: swap)
    }

    /// The lines the measurements are generated from (t₀ in seconds, rate in FLOP/s).
    static let truth: [Kernel: CapacityModel.Line] = [
        gpu: .init(t0: 0.0003, rate: 3.53e12), amx: .init(t0: 0.0009, rate: 2.3e12), ane: .init(t0: 0.00125, rate: 4.9e12),
        Kernel(.gpu, .bf16): .init(t0: 0.0002, rate: 2.18e12), Kernel(.gpu, .int8): .init(t0: 0.0004, rate: 3.4e12),
        Kernel(.gpu, .graph): .init(t0: 0.0001, rate: 1.4e12), Kernel(.gpu, .flash): .init(t0: 0.0007, rate: 1.6e12),
        Kernel(.cpu, .heads): .init(t0: 0.0001, rate: 1.0e12),
    ]

    /// Three solos per product kernel (≈ 4, 30 and 309 GFLOP), two per attention kernel, the stairs of the ANE, the sets' trials with the
    /// keep and overhead the generation of the sides gives, and the witnesses.
    static func measurements(wrong: [String] = [], dropping dropped: Set<String> = [], bandwidth: Double? = 187e9) -> Fit.Measurements {
        var m = Fit.Measurements()
        let shapes = [(512, 2048, 2048), (1024, 3840, 3840), (4096, 6144, 6144)]
        for (kernel, line) in truth.sorted(by: { $0.key.description < $1.key.description }) where !dropped.contains(kernel.description) {
            switch kernel.form {
            case .graph, .flash, .heads:
                for s in [1024, 4096] {
                    let flop = 4.0 * Double(s) * Double(s) * 24 * 128
                    m.solos.append(.init(kernel: kernel.description, shape: [s, 24, 128], flop: flop, seconds: line.seconds(flop), error: 3e-6))
                }
            default:
                for (rows, k, n) in shapes {
                    let flop = 2.0 * Double(rows) * Double(k) * Double(n)
                    m.solos.append(.init(kernel: kernel.description, shape: [rows, k, n], flop: flop, seconds: line.seconds(flop), error: 1e-6))
                }
            }
        }
        m.stairShape = [3840, 3840]
        for rows in [768, 1024, 1792, 2048, 2304] {
            m.stairs.append(.init(rows: rows, seconds: 0.00125 + 2.0 * Double(rows) * 3840 * 3840 / 4.9e12))
        }
        // A set: each side's rows, its seconds alone (the line) and within (the line over its keep); the wall is the slowest plus the overhead.
        func trial(_ engines: [CapacityModel.Engine], keeps: [CapacityModel.Engine: Double], overhead: Double, pass: Int, _ shape: (Int, Int, Int)) -> Fit.Measurements.Trial {
            let (rows, k, n) = shape
            let kernels = engines.map { Kernel($0, $0 == .ane ? .fp16 : .fp32) }
            let share = rows / kernels.count / 64 * 64
            var sides: [Fit.Measurements.Side] = []
            var slowest = 0.0
            for (i, kernel) in kernels.enumerated() {
                let r = i == kernels.count - 1 ? rows - share * (kernels.count - 1) : share
                let line = truth[kernel]!
                let flop = 2.0 * Double(r) * Double(k) * Double(n)
                let alone = line.seconds(flop)
                let within = line.t0 + flop / line.rate / keeps[kernel.engine]!
                slowest = max(slowest, within)
                sides.append(.init(kernel: kernel.description, rows: r, solo: alone, within: within))
            }
            return .init(set: CapacityModel.key(engines), shape: [rows, k, n], pass: pass, sides: sides, wall: slowest + overhead, error: 1e-6)
        }
        let sets: [([CapacityModel.Engine], [CapacityModel.Engine: Double], Double)] = [
            ([.gpu, .amx], [.gpu: 0.95, .amx: 0.88], 0.00001), ([.gpu, .ane], [.gpu: 0.87, .ane: 0.71], 0.00001),
            ([.gpu, .amx, .ane], [.gpu: 0.8, .amx: 0.65, .ane: 0.7], 0.00008), ([.amx, .ane], [.amx: 0.8, .ane: 0.93], 0.00002),
        ]
        for (engines, keeps, overhead) in sets {
            m.trials.append(trial(engines, keeps: keeps, overhead: overhead, pass: 1, (2048, 3840, 3840)))
            for shape in [(1024, 3840, 3840), (2048, 3840, 3840), (4096, 3840, 3840)] {
                m.trials.append(trial(engines, keeps: keeps, overhead: overhead, pass: 2, shape))
            }
        }
        // The attention as a set: the graph on the GPU's heads beside the CPU's, at the heads the two lines balance.
        for s in [1024, 4096] {
            let per = 2 * s * s * 128, graph = truth[Kernel(.gpu, .graph)]!, heads = truth[Kernel(.cpu, .heads)]!
            let (gpuHeads, cpuHeads) = (14, 10)
            let g = graph.t0 + Double(gpuHeads) * 2 * Double(per) / graph.rate, c = heads.t0 + Double(cpuHeads) * 2 * Double(per) / heads.rate
            m.trials.append(.init(set: "gpu+cpu", shape: [24, per, 1], pass: 2,
                                  sides: [.init(kernel: "gpu/graph", rows: gpuHeads, solo: g, within: g / 0.93),
                                          .init(kernel: "cpu/heads", rows: cpuHeads, solo: c, within: c / 0.88)],
                                  wall: max(g / 0.93, c / 0.88) + 0.00002, error: 3e-6))
        }
        m.witnesses = [.init(engine: "gpu", before: 0.0172, after: 0.0181), .init(engine: "amx", before: 0.0262, after: 0.0266),
                       .init(engine: "ane", before: 0.0124, after: 0.0125)]
        m.memoryBandwidth = bandwidth
        m.preparation = ["bf16": ["widened": 4.6e-11, "bf16": 2.6e-11], "int8": ["widened": 1.4e-10, "native": 1.2e-11],
                         "q8": ["widened": 4.1e-11, "native": 1.6e-11]]
        m.wrong = wrong
        return m
    }

    static func judge(excluded: Set<String> = []) -> [String: Diagnostic.Judge] {
        var out: [String: Diagnostic.Judge] = [:]
        for kernel in truth.keys.map(\.description) {
            out[kernel] = .init(error: kernel == "ane/fp16" ? 2.9e-4 : 1.2e-6, threshold: kernel == "ane/fp16" ? 3.1e-4 : 1e-5,
                                excluded: excluded.contains(kernel))
        }
        return out
    }

    static func memory(peak: Int = 2_400_000_000) -> Diagnostic.Memory {
        .init(budget: 10_737_418_240, base: 60_000_000, need: 2_970_000_000, peak: peak)
    }

    static func stages() -> [String: Diagnostic.StageTime] {
        ["setup": .init(seconds: 0.9), "preparation": .init(seconds: 0.18), "witnesses start": .init(seconds: 0.45),
         "solos fp32": .init(seconds: 2.26, projected: 2.1), "sharing": .init(seconds: 3.1, projected: 3.6)]
    }

    /// A report as `Diagnostic.run` makes it from these probes, 12.9 s of them.
    static func report(machine: Diagnostic.Machine = machine(), measurements: Fit.Measurements = measurements(),
                       judge: [String: Diagnostic.Judge] = judge(), memory: Diagnostic.Memory = memory(),
                       before: Diagnostic.Conditions = conditions(), after: Diagnostic.Conditions = conditions(),
                       seconds: Double = 15.2) -> Diagnostic.Report {
        let probes = Diagnostic.CapacityProbes.Result(date: "2026-10-30T10:00:00Z", seconds: seconds, tier: 0, measurements: measurements,
                                           model: CapacityFit.model(from: measurements), judge: judge, memory: memory,
                                           stages: stages(), details: .init())
        return Diagnostic.Report(date: "2026-10-30T10:00:00Z", seconds: seconds, machine: machine, cpuFeatures: ["FEAT_BF16": 1, "sme_max_svl_b": 0],
                                 settings: [:], before: before, after: after, probes: probes)
    }
}

extension CapacityModel {
    /// A copy in which every set of engines the solver may offer and that has no sharing is measured and free (keep 1, no overhead): the
    /// tests of the solver's arithmetic want the engines to share nothing, and say so.
    func withFreeSharing() -> CapacityModel {
        var m = self
        for set in [[Engine.gpu, .amx], [.gpu, .ane], [.gpu, .amx, .ane], [.amx, .ane], [.gpu, .cpu]] where m.sharing[Self.key(set)] == nil {
            m.sharing[Self.key(set)] = Sharing()
        }
        return m
    }
}
