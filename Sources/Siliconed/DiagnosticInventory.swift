// What the registry and the kernel say about the silicon, **read and never decided on**: the GPU's
// cores and clock, the CPU's performance levels and clusters, the Neural Engine's generation. Its uses
// are the denominator of "reached / peak" in a report and the list of probes worth running (an engine
// that is absent is not probed). No chip name enters: a field the machine does not give is `nil`, a
// value a layout check refuses is `nil`, and a Mac without an `AGXAccelerator` (a virtualised CI
// runner) reads all `nil`. Split from `DiagnosticMachine.swift`.
//
//     AGXAccelerator ── GPUConfigurationVariable {gpu_gen, num_cores, core_mask_list}
//          └ parent (the "gpu" node) ── perf-states, perf-state-count, perf-state-table-count,
//                                       gpu-num-perf-states ─────────────▶ clock table ─▶ max MHz
//     hw.perflevelN.{name, physicalcpu, cpusperl2} ───────────────────────▶ cores, clusters
//     ane0 ── compatible, ane-type, ane-subtype ──────────────────────────▶ generation
//
// What a reading by analogy would miss:
//   • **The GPU clock is on the GPU's own node**, not found by guessing which `voltage-states<N>` of
//     `pmgr` is the GPU's: `perf-states` there is the same table (M1 Pro: 7 entries, 0 Hz then 388.8 MHz
//     … 1 296 MHz; `pmgr`'s `voltage-states9` and `…14` hold the same frequencies) and says by itself
//     how it is cut — `perf-state-count` entries per table, `perf-state-table-count` tables (one per
//     voltage bin; `perf-states-sram` is the SRAM rail's voltages, its frequencies are not the clock's).
//     The table is accepted only if its byte length is exactly that cut and its non-zero entries number
//     `gpu-num-perf-states` (the idle entry is the zero): a layout that moved fails both tests and the
//     clock stays `nil`.
//   • **`num_cores` is the silicon's, `gpu-core-count` the enabled ones**: 16 against 14 on an M1 Pro.
//     `core_mask_list` is one mask per GPU partition (253, 253 there: two cores off, one in each).
//   • **The CPU tables of `pmgr` are not read.** They give periods (65 536 000 / value = MHz: 3 228 for
//     the P cluster of an M1 Pro, 2 064 for the E), and nothing on the cluster or the node says which
//     table is whose: matching them would be a guess by size. The P clock stays `nil`; the AMX/SME peak
//     is measured or comes with a clock from elsewhere.
//   • **The ANE's core count is not exposed** — only its generation (`ane,t8020`, type 0x60 on the M1
//     family). It is not inferred here.

import Foundation
import IOKit

extension Diagnostic {
    /// The inventory a report carries. Optional fields are absent from the JSON when unread (the link
    /// of an issue counts its bytes), and a report that predates it decodes with all of them `nil`.
    package struct Inventory: Codable, Sendable, Equatable {
        /// `GPUConfigurationVariable.num_cores`: the GPU's cores on the die (`Machine.gpuCores` is the enabled ones).
        package var gpuSiliconCores: Int?
        /// `GPUConfigurationVariable.gpu_gen`: 13 for the M1 family.
        package var gpuGen: Int?
        /// `GPUConfigurationVariable.core_mask_list`.
        package var gpuCoreMask: [Int]?
        /// The highest GPU clock in MHz, from the GPU node's `perf-states`; `nil` unless the table passed its checks.
        package var gpuMaxMHz: Double?
        /// `hw.perflevelN`, N = 0 the fastest.
        package var cpuLevels: [CPULevel]?
        /// `ane0`'s `compatible`, `ane-type` and `ane-subtype`.
        package var ane: ANE?

        package init(gpuSiliconCores: Int? = nil, gpuGen: Int? = nil, gpuCoreMask: [Int]? = nil, gpuMaxMHz: Double? = nil,
                    cpuLevels: [CPULevel]? = nil, ane: ANE? = nil) {
            self.gpuSiliconCores = gpuSiliconCores; self.gpuGen = gpuGen; self.gpuCoreMask = gpuCoreMask
            self.gpuMaxMHz = gpuMaxMHz; self.cpuLevels = cpuLevels; self.ane = ane
        }

        /// A performance level, written `[name, cores, cpusperl2]` (the link's bytes are counted).
        package struct CPULevel: Codable, Sendable, Equatable {
            package var name: String
            package var cores: Int
            /// Cores sharing one L2 — one cluster, so one matrix unit (AMX/SME) on the P level; `nil` if the kernel does not say.
            package var coresPerL2: Int?

            package init(name: String, cores: Int, coresPerL2: Int?) {
                self.name = name; self.cores = cores; self.coresPerL2 = coresPerL2
            }

            /// Rounded up: a cluster with a core fused off is still a cluster.
            package var clusters: Int? { coresPerL2.flatMap { $0 > 0 ? (cores + $0 - 1) / $0 : nil } }

            package init(from decoder: any Decoder) throws {
                var c = try decoder.unkeyedContainer()
                name = try c.decode(String.self); cores = try c.decode(Int.self)
                coresPerL2 = c.isAtEnd ? nil : try c.decode(Int?.self)
            }

            package func encode(to encoder: any Encoder) throws {
                var c = encoder.unkeyedContainer()
                try c.encode(name); try c.encode(cores)
                if let coresPerL2 { try c.encode(coresPerL2) }
            }
        }

        package struct ANE: Codable, Sendable, Equatable {
            /// "ane,t8020".
            package var compatible: String?
            package var type: Int?
            package var subtype: Int?

            package init(compatible: String? = nil, type: Int? = nil, subtype: Int? = nil) {
                self.compatible = compatible; self.type = type; self.subtype = subtype
            }
        }

        /// **The GPU's nominal fp32 peak in FLOP/s** for `enabledCores`: cores × 128 lanes × 2 (a fused
        /// multiply-add) × clock. 128 fp32 lanes per core is the Apple GPU architecture's constant, the same on
        /// the generations measured so far (M1 Pro: 14 × 256 × 1.296 GHz = 4.65 TFLOP/s, of which the
        /// engine reaches 3.5). A *nominal* figure, the denominator of an efficiency, not a prediction:
        /// the clock is the table's highest, which a loaded GPU does not hold. `nil` without cores or clock.
        package func gpuFP32Peak(enabledCores: Int) -> Double? {
            guard enabledCores > 0, let gpuMaxMHz else { return nil }
            return Double(enabledCores) * 128 * 2 * gpuMaxMHz * 1e6
        }

        // ── the analysis: pure, on what the registry returned ───────────────────────────────

        /// `GPUConfigurationVariable` as the registry returns it (a dictionary of numbers and a list).
        package static func gpuConfiguration(_ dictionary: [String: Any]?) -> (cores: Int?, gen: Int?, mask: [Int]?) {
            guard let dictionary else { return (nil, nil, nil) }
            func positive(_ key: String) -> Int? { (dictionary[key] as? NSNumber).map(\.intValue).flatMap { $0 > 0 ? $0 : nil } }
            let mask = (dictionary["core_mask_list"] as? [Any])?.compactMap { ($0 as? NSNumber)?.intValue }
            return (positive("num_cores"), positive("gpu_gen"), mask?.isEmpty == false ? mask : nil)
        }

        /// **The highest clock in MHz of a GPU node's `perf-states`**, or `nil` when anything disagrees with
        /// the cut the node declares (see the header). Entries are `(u32 Hz, u32 millivolt-ish)`, little-endian.
        package static func gpuMaxMHz(perfStates: Data?, stateCount: Int?, tableCount: Int?, activeStates: Int?) -> Double? {
            guard let perfStates, let stateCount, let tableCount, let activeStates,
                  stateCount > 0, tableCount > 0, activeStates > 0,
                  perfStates.count == stateCount * tableCount * 8 else { return nil }
            var maxima: [UInt32] = []
            for table in 0..<tableCount {
                let hertz = (0..<stateCount).map { littleEndian32(perfStates, at: (table * stateCount + $0) * 8) }
                guard hertz.filter({ $0 > 0 }).count == activeStates, let top = hertz.max() else { return nil }
                maxima.append(top)
            }
            // Bins of one die share their frequencies; two different tops would make "the" maximum a choice.
            guard let top = maxima.first, maxima.allSatisfy({ $0 == top }) else { return nil }
            let mhz = Double(top) / 1e6
            return (100...6000).contains(mhz) ? mhz : nil
        }

        /// A little-endian 32-bit integer from a registry property (`<06000000>`), `nil` unless it is 4 bytes.
        package static func integer32(_ data: Data?) -> Int? {
            guard let data, data.count == 4 else { return nil }
            return Int(littleEndian32(data, at: 0))
        }

        /// `compatible`'s first string ("ane,t8020"): the property is NUL-separated strings.
        package static func firstString(_ data: Data?) -> String? {
            guard let data else { return nil }
            let first = data.prefix { $0 != 0 }
            return first.isEmpty ? nil : String(decoding: first, as: UTF8.self)
        }

        private static func littleEndian32(_ data: Data, at offset: Int) -> UInt32 {
            let base = data.startIndex + offset
            return (0..<4).reduce(UInt32(0)) { $0 | UInt32(data[base + $1]) << (8 * UInt32($1)) }
        }

        // ── the reading: the registry and the kernel ────────────────────────────────────────

        /// All `nil` where there is no `AGXAccelerator` or no `ane0` (a virtualised runner, an Intel Mac).
        package static func read() -> Inventory {
            var inventory = Inventory()
            var iterator: io_iterator_t = 0
            if IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"), &iterator) == KERN_SUCCESS {
                defer { IOObjectRelease(iterator) }
                let service = IOIteratorNext(iterator)
                if service != 0 {
                    defer { IOObjectRelease(service) }
                    let configuration = gpuConfiguration(property(service, "GPUConfigurationVariable") as? [String: Any])
                    inventory.gpuSiliconCores = configuration.cores; inventory.gpuGen = configuration.gen
                    inventory.gpuCoreMask = configuration.mask
                    var parent: io_registry_entry_t = 0
                    if IORegistryEntryGetParentEntry(service, kIOServicePlane, &parent) == KERN_SUCCESS {
                        defer { IOObjectRelease(parent) }
                        inventory.gpuMaxMHz = gpuMaxMHz(
                            perfStates: property(parent, "perf-states") as? Data,
                            stateCount: integer32(property(parent, "perf-state-count") as? Data),
                            tableCount: integer32(property(parent, "perf-state-table-count") as? Data),
                            activeStates: integer32(property(parent, "gpu-num-perf-states") as? Data))
                    }
                }
            }
            inventory.cpuLevels = cpuLevels()
            inventory.ane = readANE()
            return inventory
        }

        private static func property(_ entry: io_registry_entry_t, _ key: String) -> Any? {
            IORegistryEntryCreateCFProperty(entry, key as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue()
        }

        private static func readANE() -> ANE? {
            var iterator: io_iterator_t = 0
            guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceNameMatching("ane0"), &iterator) == KERN_SUCCESS else {
                return nil
            }
            defer { IOObjectRelease(iterator) }
            let service = IOIteratorNext(iterator)
            guard service != 0 else { return nil }
            defer { IOObjectRelease(service) }
            let ane = ANE(compatible: firstString(property(service, "compatible") as? Data),
                          type: integer32(property(service, "ane-type") as? Data),
                          subtype: integer32(property(service, "ane-subtype") as? Data))
            return ane == ANE() ? nil : ane
        }

        private static func cpuLevels() -> [CPULevel]? {
            guard let count = Diagnostic.sysctlInteger("hw.nperflevels"), count > 0 else { return nil }
            let levels = (0..<count).compactMap { n -> CPULevel? in
                guard let cores = Diagnostic.sysctlInteger("hw.perflevel\(n).physicalcpu"), cores > 0 else { return nil }
                let name = Diagnostic.sysctlString("hw.perflevel\(n).name")
                return CPULevel(name: name == "unknown" ? "level\(n)" : name, cores: cores,
                                coresPerL2: Diagnostic.sysctlInteger("hw.perflevel\(n).cpusperl2"))
            }
            return levels.isEmpty ? nil : levels
        }
    }
}
