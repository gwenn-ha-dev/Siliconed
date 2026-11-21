import CryptoKit
import Foundation
import Siliconed

// **The DiT's pace, alone** — `siliconed check <model> pace [--size 1024] [--renders 3] [--evaluations 8]`.
//
// A render's denoising without its text or its decoder: the DiT as the render opens it, opened again for each render
// and evaluated as many times as the sampler would, on seeded inputs of the render's shapes. Each render says its
// evaluations and the digest of every velocity it computed: two renders of one plan give the same digest. No
// reference: the values are the `dit` stage's to judge.

enum Pace {
    /// `renders` DiTs opened by `session`, each evaluated `evaluations` times through `evaluate` (its index, the
    /// sampler's time for it, the velocity it returns); then the memory judged as `memory` judges a render: each engine
    /// stage opened against its P, the process against B, no swap.
    static func run(_ invocation: Invocation, title: String, session: () throws -> AnyObject,
                    evaluate: (AnyObject, Int, Float) throws -> [Float]) throws -> Report {
        let renders = try invocation.integer("--renders", 3)
        let evaluations = try invocation.integer("--evaluations", 8)
        guard renders >= 1, evaluations >= 1 else { throw UsageError("--renders and --evaluations: 1 or more") }
        let report = Report(title: title), watch = StageWatch()
        defer { watch.stop() }
        for r in 0..<renders {
            let planning = Date()
            let s = try session()
            let planned = Date().timeIntervalSince(planning)
            var seconds: [Double] = [], digest = Insecure.MD5()
            for e in 0..<evaluations {
                let start = Date()
                let velocity = try evaluate(s, e, 1 - Float(e) / Float(evaluations))
                seconds.append(Date().timeIntervalSince(start))
                velocity.withUnsafeBytes { digest.update(bufferPointer: $0) }
            }
            let rest = seconds.dropFirst().sorted()
            let md5 = digest.finalize().map { String(format: "%02x", $0) }.joined()
            print(String(format: "render %d: opened in %.2f s; evaluations %@ s; first %.2f, median of the others %.3f, all %.2f; md5 %@",
                         r + 1, planned, seconds.map { String(format: "%.2f", $0) }.joined(separator: " "), seconds[0],
                         rest.isEmpty ? 0 : rest[rest.count / 2], seconds.reduce(0, +), String(md5.prefix(12))))
        }
        watch.stop()
        MemoryStages.flight(watch.heard)
        MemoryStages.judge(watch.heard, report)
        return report
    }

    /// Seeded values in [-1, 1): the inputs of an evaluation whose values nothing judges.
    static func noise(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int64(bitPattern: state) >> 11) / Float(1 << 52)
        }
    }

    static let proves = "the DiT as a render opens it, alone: each render's evaluations on seeded inputs (--renders, 3; --evaluations, 8), the digest of its velocities, each engine stage's footprint against its P, the process against B, no swap"
    static let rule = "each stage's footprint stays under B less 64 MiB; the process's under B (¾ · RAM − 2 GiB); no swap-out"
}
