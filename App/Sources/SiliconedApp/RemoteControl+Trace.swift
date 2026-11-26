// `trace`: what the last render's engine stages recorded under Debug Live, as JSON.

import Foundation
import Siliconed

extension RemoteControl {
    /// **`trace`: the render under way's, or the last one's**, stage by stage: what each engine stage's calls executed
    /// over their spans (`Engine.Held.work`, in every render), and under Debug Live (`AppState.debugLive`, `set
    /// debug-live`) each DiT step's units — busy, idle and where, step by step (`Engine.Occupancy`) —, the runs of calls
    /// outside the steps (the text encoder's layers, the decoder's bands), each engine stage's opening and close piece by
    /// piece. Seconds on the render's clock (`MemorySeries`, from its start).
    func trace() -> [String: Any] {
        let held = app.measurements.held
        return ["ok": true, "debug_live": local.debugLive, "stages": held.map(Self.traceJSON)]
    }

    static func traceJSON(_ h: Engine.Held) -> [String: Any] {
        var o: [String: Any] = ["stage": stageName(h.stage), "compiled_s": Response.number(h.compilation, 3)]
        o["work"] = h.work.map { w -> [String: Any] in
            ["tflops": Response.number(w.tflops, 3), "seconds": Response.number(w.seconds, 4),
             "end": w.end.map { Response.number($0, 4) } ?? NSNull(), "ane_fraction": Response.number(w.aneFraction, 3)]
        }
        if !h.cycles.isEmpty {
            o["cycles"] = h.cycles.enumerated().map { n, c -> [String: Any] in
                var e: [String: Any] = ["index": n + 1, "ane_rows": c.rows, "rows": c.shareable, "seconds": Response.number(c.seconds, 4),
                                        "end": c.end.map { Response.number($0, 4) } ?? NSNull(), "tflops": Response.number(c.tflops, 3),
                                        "ane_tflops": c.neuralEngine.map { Response.number($0, 3) } ?? NSNull(),
                                        "gpu_tflops": c.gpu.map { Response.number($0, 3) } ?? NSNull()]
                e["occupancy"] = c.occupancy.map(occupancyJSON) ?? NSNull()
                return e
            }
        }
        if !h.traces.isEmpty {
            o["engine_stages"] = h.traces.map { t -> [String: Any] in
                func pieces(_ p: [Engine.StageTrace.Piece]) -> [[String: Any]] {
                    p.map { ["name": $0.name, "seconds": Response.number($0.seconds, 4)] }
                }
                return ["name": t.name, "opened": t.opened.map { Response.number($0, 4) } ?? NSNull(),
                        "closed": t.closed.map { Response.number($0, 4) } ?? NSNull(),
                        "opening": pieces(t.opening), "closing": pieces(t.closing), "runs": t.runs.map(occupancyJSON)]
            }
        }
        return o
    }

    /// An occupancy: its seconds, each unit's busy seconds, where each stood idle, each step's calls on each unit.
    static func occupancyJSON(_ x: Engine.Occupancy) -> [String: Any] {
        func idle(_ i: Engine.Occupancy.Idle) -> [String: Any] {
            ["across_alone": Response.number(i.acrossAlone, 4), "at_groups": Response.number(i.atGroups, 4),
             "between": Response.number(i.between, 4), "outside": Response.number(i.outside, 4)]
        }
        return ["seconds": Response.number(x.seconds, 4), "ane_busy": Response.number(x.ane, 4), "gpu_busy": Response.number(x.gpu, 4),
                "ane_idle": idle(x.aneIdle), "gpu_idle": idle(x.gpuIdle), "unread": x.unread,
                "steps": x.steps.map { s -> [String: Any] in
                    ["unit": s.unit, "name": s.name, "calls": s.calls, "rows": s.rows,
                     "seconds": Response.number(s.seconds, 5), "tflops": Response.number(s.tflops, 3)]
                }]
    }
}
