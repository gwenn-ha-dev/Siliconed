// **`silicontrol measure`'s procedure**, the one our own series follow: the app's queue must be idle;
// a first render carries the load; a witness opens the series and another closes it (the machine
// drifts by 4 to 13 % from one hour to the next, and a time without its witness is not a time);
// each render is `add --wait` on the app's socket, timed from the request to the PNG, while a
// thread samples the app's `phys_footprint` every 100 ms. The transport is injected: the tests drive
// the procedure with a fake app, `silicontrol` with the socket.

import Darwin
import Foundation

package struct Failure: Error { package let code: String, message: String }

package enum Runner {
    /// One command sent to the app: calls `onLine` for each JSON line, returns the last and the app's pid.
    package typealias Send = (_ argv: [String], _ onLine: ([String: Any]) -> Void) -> (last: [String: Any]?, peer: pid_t?)

    private static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

    private static func setFaster(_ on: Bool, of model: String, send: Send) throws {
        let (answer, _) = send(["set", "faster-rendering", on ? "on" : "off", "--model", model]) { _ in }
        guard answer?["ok"] as? Bool == true else {
            throw Failure(code: answer?["error"] as? String ?? "app_not_running",
                          message: answer?["message"] as? String ?? "the app does not answer")
        }
    }

    package static func run(_ o: MeasureOptions, send: Send, log: (String) -> Void,
                            emit: ([String: Any]) -> Void, append: (String) -> Void) throws -> [String: Any] {
        // The queue must be idle: a render queued by someone else shares the machine with ours.
        func idle() throws -> pid_t {
            let (status, peer) = send(["status"]) { _ in }
            guard let status, status["ok"] as? Bool == true, let peer else {
                throw Failure(code: "app_not_running", message: "the app does not answer")
            }
            if status["running"] is [String: Any] || !((status["queue"] as? [Any]) ?? []).isEmpty
                || status["installing"] is [String: Any] {
                throw Failure(code: "busy", message: "the app is rendering or installing: a series measured beside another job measures both")
            }
            return peer
        }
        var pid = try idle()
        let (models, _) = send(["models"]) { _ in }
        guard let list = models?["models"] as? [[String: Any]], let entry = list.first(where: { $0["id"] as? String == o.model }) else {
            throw Failure(code: "model", message: "\(o.model) is not a model the app shows (silicontrol models)")
        }
        guard entry["ready"] as? Bool == true else {
            throw Failure(code: "model", message: "\(o.model) is not installed (silicontrol install \(o.model))")
        }
        // Faster Rendering is chosen per model, in the app: read from the measured model's entry, changed
        // and given back through `set … --model`, so the app writes the choice and the switch follows. Not
        // from the preferences file, which an app that has not synchronized yet would give stale.
        let original = entry["faster_rendering"] as? Bool
        var changed = false
        if let wanted = o.fasterRendering {
            guard let original else {
                throw Failure(code: "unsupported", message: "this app does not report Faster Rendering per model: --faster-rendering needs the same build as silicontrol")
            }
            if wanted != original { try setFaster(wanted, of: o.model, send: send); changed = true }
        }
        // Whatever ends the series, the user's value comes back; a refusal there (someone queued a job
        // meanwhile) is told, not hidden.
        defer {
            if changed, let original {
                do { try setFaster(original, of: o.model, send: send) }
                catch let failure as Failure { log("measure: Faster Rendering could not be put back to \(original ? "on" : "off"): \(failure.message)") }
                catch {}
            }
        }
        let variant = entry["variant"] as? String ?? "-"
        // The « Accelerators » switch (`--ane`, `--no-ane` its former names; an app before it says "neural_engine").
        let on = ["--accelerators", "--ane"], off = ["--no-accelerators", "--no-ane"]
        let neuralOverride = o.extra.contains(where: on.contains) ? true : o.extra.contains(where: off.contains) ? false : nil
        let neural = neuralOverride ?? (entry["accelerators"] ?? entry["neural_engine"]) as? Bool
        let host = Probes.Host.read()
        let bundle = Probes.bundle(of: pid)
        let faster = o.fasterRendering ?? original
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZZZZZ"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let context = Context(date: formatter.string(from: Date()), tag: o.tag, app: bundle?.version ?? "?",
                              chip: host.chip, memoryGiB: host.memoryGiB, macOS: host.macOS)
        log("measure: \(o.model) (\(variant)) \(o.size), seed \(o.seed), \(o.renders) renders + load + 2 witnesses, "
            + "Faster Rendering \(faster.map { $0 ? "on" : "off" } ?? "unknown"), app \(context.app), \(host.chip) \(host.memoryGiB) GiB, \(Probes.powerSource())")
        if Probes.powerSource() == "battery" { log("measure: on the battery — plug the adapter in for a time comparable to ours") }

        append(TSV.header.joined(separator: "\t"))
        let outDirectory: URL
        if let out = o.out { outDirectory = URL(fileURLWithPath: out, isDirectory: true) }
        else { outDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("siliconed-measure-\(getpid())", isDirectory: true) }
        try? FileManager.default.createDirectory(at: outDirectory, withIntermediateDirectories: true)
        defer { if o.out == nil { try? FileManager.default.removeItem(at: outDirectory) } }

        if o.settle > 0 {
            sync()
            log("measure: settling \(o.settle) s")
            Thread.sleep(forTimeInterval: Double(o.settle))
        }

        // The witness is always the default prompt at seed 42: the same picture asked of every series.
        let plan: [(Role, String, Int)] =
            [(.load, MeasureOptions.defaultPrompt, 42), (.witnessHead, MeasureOptions.defaultPrompt, 42)]
            + (1...o.renders).map { (.render, o.prompt(forRender: $0), o.seed) }
            + [(.witnessTail, MeasureOptions.defaultPrompt, 42)]

        var rows: [Row] = [], powers = Set<String>()
        for (n, step) in plan.enumerated() {
            let (role, prompt, seed) = step
            pid = try idle()
            let before = Probes.usage(of: pid)
            let sampler = Probes.PeakSampler(pid: pid)
            let swap0 = Probes.swapouts()
            var imageAt: Double?, engine: Double?, steps: Int?, path: String?
            let t0 = now()
            let (last, _) = send(["add", "--wait", "--out", outDirectory.path, "--model", o.model] + o.extra + [prompt, o.size, String(seed)]) { line in
                if line["type"] as? String == "image", imageAt == nil {
                    imageAt = now()
                    engine = (line["seconds"] as? NSNumber)?.doubleValue
                    steps = (line["steps"] as? NSNumber)?.intValue
                    path = line["path"] as? String
                }
            }
            let (peak, disk) = sampler.stop()
            let swaps = Probes.swapouts() &- swap0
            guard let imageAt, last?["ok"] as? Bool == true else {
                throw Failure(code: last?["error"] as? String ?? "render",
                              message: last?["message"] as? String ?? "the render did not end")
            }
            guard let path, let md5 = Probes.pixelDigest(png: URL(fileURLWithPath: path)) else {
                throw Failure(code: "write", message: "no PNG to read the pixels from (\(path ?? "no path"))")
            }
            if o.out == nil { try? FileManager.default.removeItem(atPath: path) }
            let row = Row(n: n, role: role, model: o.model, variant: variant, size: o.size, steps: steps, seed: seed,
                          prompt: prompt, fasterRendering: faster, neuralEngine: neural, wallSeconds: imageAt - t0,
                          engineSeconds: engine, startFootprint: before?.footprint ?? 0, peakSampled: peak,
                          swapouts: swaps, diskRead: disk, md5: md5)
            rows.append(row)
            let power = Probes.powerSource(), thermal = Probes.thermal()
            powers.insert(power)
            append(TSV.line(row, context, power: power, thermal: thermal))
            log(String(format: "measure: %@ %.1f s (engine %@ s)  peak %.2f GiB  swapouts %d  disk read %.1f GB  %@  [%@, %@]",
                       role.rawValue.padding(toLength: 13, withPad: " ", startingAt: 0), row.wallSeconds,
                       engine.map { String(format: "%.1f", $0) } ?? "-", Double(peak) / 1_073_741_824, swaps,
                       Double(disk) / 1e9, String(md5.prefix(10)), power, thermal))
            emit(["type": "render", "n": n, "role": role.rawValue, "seconds": row.wallSeconds, "engine_seconds": engine as Any,
                  "peak_gib": Double(peak) / 1_073_741_824, "swapouts": swaps, "disk_read_bytes": disk, "md5": md5])
        }

        let s = Summary(rows, power: powers)
        func f(_ x: Double?, _ digits: Int = 1) -> String { x.map { String(format: "%.\(digits)f", $0) } ?? "-" }
        log("")
        log("summary  \(o.model) (\(variant)) \(o.size)  \(s.renders) renders")
        log("  median \(f(s.median)) s   fastest \(f(s.fastest)) s   slowest \(f(s.slowest)) s   (the load render: \(f(s.load)) s)")
        log("  witness \(f(s.witnessHead)) s -> \(f(s.witnessTail)) s   drift \(s.drift.map { String(format: "%+.1f %%", $0 * 100) } ?? "-")")
        log(String(format: "  peak %.2f GiB   swapouts %d   disk read %.1f GB   pixels %@", Double(s.peak) / 1_073_741_824, s.swapouts,
                   Double(s.diskRead) / 1e9, s.bitsStable ? "stable (\(s.distinctPictures) distinct)" : "NOT STABLE"))
        for note in s.notes { log("  ! " + note) }

        var summary: [String: Any] = ["renders": s.renders, "peak_gib": Double(s.peak) / 1_073_741_824, "swapouts": s.swapouts,
                                      "disk_read_bytes": s.diskRead, "bits_stable": s.bitsStable, "distinct_pictures": s.distinctPictures,
                                      "notes": s.notes]
        for (k, v) in [("median_seconds", s.median), ("fastest_seconds", s.fastest), ("slowest_seconds", s.slowest),
                       ("load_seconds", s.load), ("witness_head_seconds", s.witnessHead),
                       ("witness_tail_seconds", s.witnessTail), ("witness_drift", s.drift)] { if let v { summary[k] = v } }
        return ["ok": true, "summary": summary]
    }
}
