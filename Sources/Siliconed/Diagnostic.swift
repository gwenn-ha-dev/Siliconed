import CryptoKit
import Foundation
import IOKit
import IOKit.ps
import Metal

/// **The diagnostic — a capacity map of the machine**: one minute of wall on any Mac (`CapacityMap.budget`, settings and
/// witnesses included), voluntary, not one byte of weights, a JSON a user can share. It answers two things: what the fleet
/// has (one report per user; `tools/aggregate_diagnostics.py` makes the table by chip), and, per machine, **which route each
/// class of computation of a render takes** — the table the engine reads (`RouteTable`), nothing else of it is a switch.
///
///     the classes of a render   CapacityMap.catalogue      every product k×n at its real height, every attention, of the families asked,
///        │                                                  weight = share of the render, the maximum over the requests
///        ├─ widening     the preparation rates: widen, copy — bf16, int8, Q8_0                                   (0.2 s)
///        ├─ witnesses    one fixed product on the GPU, the AMX and the ANE, median of five                       (1 s, twice)
///        ├─ the plan     CapacityMeasure.plan: pairs A B A B of two routes of a class, the seconds go to the class whose
///        │               decision costs most if it is wrong; a probe (`MapRun.probe`) runs a route once, judged against fp64
///        │               the first time, the cut of a pair balanced when the pair is first asked
///        └─ the table    the choices as class rules over the grids of the cuts, completed to the heights nobody measured
///
/// What sets it apart — what a diagnostic written by analogy with the switches it replaces would get wrong:
///
///   - **it judges the chip, never the weights.** The operands are synthetic, the same on every machine (xorshift by blocks of
///     2¹⁶), and every product is a block's shape, read from the families' `Config` defaults (`CapacityClasses`). An 8-bit
///     install could only fail a map that ran its model;
///   - **a route is decided per class, from pairs measured in turn** (`CapacityMeasure`): the drift of a Mac that heats cancels
///     in the ratio, and the table is decided at the regime of the map's end, read from a witness per engine
///     (`CapacityMap.EngineWitness`) — not at the cold start, and not through a margin taken from the drift;
///   - **the AMX is the engine's BNNS, not `cblas`** (`Conductor.multiply`, the call the cut makes on its worker thread);
///   - **against fp64, never against another fp32**: `cblas` fp32 has its own error, of the size of the GPU's. A route that
///     computes wrong is excluded from its class (`MapResult.wrong`), nothing more;
///   - **a half-precision output is judged against its own rounding**: the ANE within 1.25 × the bf16 rounding of the exact
///     (invariant 3), never under the fp32 threshold;
///   - **every output is poisoned before each engine** (NaN on the sampled rows): a submission that fails silently leaves the
///     previous engine's right answer, which would pass;
///   - **the minute holds the whole map, and the memory holds B**: the plan of the map's memory is chosen from its classes
///     before the first product (`MemoryNeedCapacityMap.swift`), its peak is read by step (`Report.memory`), and green
///     (`CapacityMap.Verdict`) says the map is complete, judged, within its seconds and its memory — never that the Mac was
///     at rest;
///   - **it holds the machine's render lock** (`RenderLock.machine`) and runs on the engine's queue: never beside a render, of
///     this process or another.
///
/// **Deliberately not measured**: the disk (no decision reads its speed); a model's step (its weights, its plan and the page
/// cache decide it, and a render's own figures already say it); the memory (B is a function of the RAM alone, ¾ · RAM − 2 GiB);
/// an SDPA in fp16 (P20: never under bf16, sums in fp32 — a precision the engine refuses is no capacity); the half-precision
/// products of a fleet question (dropped from schema 13); a taller image than 1024² (the table extrapolates the heights).
///
/// **The engine reads it in the app, and only there**: the app measures before its first render when it holds no current map
/// — none yet, one of another `Fingerprint` (hardware, chip, macOS build, the app's build: an update measures again), one
/// that is not green or does not cover the families installed — and whenever the user asks (« Measure This Mac »). It keeps
/// the report in its preferences and hands it to `EngineSettings.load(from:map:)`, which installs its table. The command line,
/// the checks and the close-out never designate a map: today's defaults.
package enum Diagnostic {
    /// The JSON's version: the script that aggregates the reports reads it. **5** (14/10/2026): the
    /// capacity map; 4 and before rendered the installed models. `decisions` came into 5 later the same
    /// day, optional: a report without it is the same measures. **6** (17/10/2026): the bf16-weight
    /// products (`products[].tensor`), the `bf16 copy` widening and `stages`; the decisions add `tensor`.
    /// **7** (18/10/2026): `cuts`, the GPU's balanced share on a grid of shapes (`k` and `n` axes, both
    /// heights); the decisions add `amxShares`. **8** (18/10/2026): the int8-weight products
    /// (`products[].int8`, `int8Coexecution`, `int8GPUColumns`), the `int8 convrot` and `int8 copy`
    /// widenings; the decisions add `int8`. **9** (18/10/2026): the Q8_0-weight products (`products[].q8`,
    /// `q8Coexecution`, `q8GPUColumns`), the `q8_0` and `q8_0 copy` widenings; the decisions add `q8`.
    /// **10** (22/10/2026): the Neural Engine beside the GPU (`products[].ane`, `aneGPUShare`, and `aneCuts` on
    /// the cuts' grid); the decisions add `ane`, `aneFraction` and `aneShares`.
    /// **11** (25/10/2026): the products on an activation rounded to bf16 (`products[].bf16Act`); the decisions
    /// add `bf16Act`. **12** (26/10/2026): the three engines at once on the ANE's grid (`aneCuts`' eighth figure,
    /// `three`), and MOSS-Image's shapes in it; the decisions add `aneAMX`. **13** (29/10/2026): the map decides by
    /// classes, not by switches — `routes` (the table the engine reads), `map` (per class: the route chosen, its ratios
    /// and their dispersion, the loss that stays, the seconds spent), `regime` (a witness per engine at the start and at
    /// the end, the heat that came out), `verdict` (green or orange and why), `memory` declared; the stages are
    /// `widening`, `witness`, `routes`, `regime`, and `products`, `cuts`, `aneCuts`, `lowPrecision`, `attention` are empty.
    /// A report of an older schema is not read by the engine (`isCurrent`).
    package static let schema = 13
    package static let repository = "https://github.com/gwenn-ha-dev/Siliconed"

    /// **The fp32 threshold**: the AMX check's (`siliconed-dev amx`), a row of an fp32 product or
    /// attention no further than 10⁻⁵ from the exact. Measured on the M1 Pro (14/10/2026): the
    /// products 0.80–2.1·10⁻⁶ (the worst at k = 12 288, GPU and the cut alike), the attention
    /// 2.0–4.4·10⁻⁶ (flash the closest) — 2.3× under it at the worst, and a fault is orders above.
    package static let fp32Threshold = 1e-5
    /// A half-precision output passes within this factor of its rounding floor (invariant 3).
    package static let floorMargin = 1.25

    // ── the run ─────────────────────────────────────────────────────────────────────────────

    /// **The stage the diagnostic is at**, as `run` reports it to `onProgress` — an app says it in its
    /// own language; `description` is the English the command line prints.
    package enum Stage: String, Sendable, CaseIterable, CustomStringConvertible {
        case widening = "weight preparation"
        case witness = "the engines' first measurement"
        case routes = "compute routes"
        case regime = "the engines again"
        package var description: String { rawValue }
        /// Its key in the report's `stages`: short, the issue link's bytes are counted.
        package var key: String {
            switch self {
            case .widening: return "widening"
            case .witness: return "witness"
            case .routes: return "routes"
            case .regime: return "regime"
            }
        }
    }

    /// **Runs the diagnostic**: ~1 min on the M1 Pro, no weights, its peak under B (`Plan`). Synchronous and
    /// heavy: call it off the main thread, and never from the engine's queue.
    ///
    /// **It passes the doors a render passes**: it waits its turn on the engine's queue (`Engine.file`:
    /// never beside a render of this process) and holds the machine's render lock (a render of
    /// another process refuses it: `renderAlreadyRunning`). `cancellation` is checked between two
    /// measurements (`cancelled`).
    ///
    /// `families`: the families the map decides for (all five by default). The classes of what is not installed only
    /// make the minute thinner for what is.
    package static func run(families: [Family] = Family.allCases, cancellation: Cancellation? = nil,
                           onProgress: (@Sendable (_ stage: Stage) -> Void)? = nil) throws(EngineError) -> Report {
        dispatchPrecondition(condition: .notOnQueue(Engine.file))
        let result = Engine.file.sync {
            Result { () throws(EngineError) -> Report in
                try runQueued(families: families, cancellation: cancellation, onProgress: onProgress)
            }
        }
        return try result.get()
    }
}
