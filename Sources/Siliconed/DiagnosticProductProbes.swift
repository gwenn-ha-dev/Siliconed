// The ANE's balance beside the GPU on one product, which the map's probes call (`MapRun`).
// Split from `DiagnosticProbes.swift`.

import Foundation

extension Diagnostic {
    /// **The GPU's share beside the ANE that ends both sides together**, for `run` (one `m × k → n` product
    /// through `neural`). Each side's wall is affine in its rows — the ANE reads the whole weight at every
    /// call, whatever its rows —: three runs at ½ and three at ¾, the medians give both lines, and the share
    /// where they cross, in `NeuralEngine.shareRange`. And the GPU alone's wall on all `m` rows, read off the
    /// same fit (`gpuFixed + gpuSlope · m`) — no run of its own. `nil` when the ANE fell back.
    static func aneBalance(_ neural: GEMM, run: () -> Void, m: Int, k: Int, n: Int) -> (share: Double, alone: Double)? {
        guard let engine = neural.neuralEngine else { return nil }
        func sides(_ share: Double) -> (gpu: Double, gpuRows: Double, ane: Double, aneRows: Double)? {
            engine.gpuShare = share
            run()
            var walls: [(gpu: Double, ane: Double)] = []
            for _ in 0..<3 {
                let before = engine.fallbacks
                run()
                guard let last = neural.lastSides, engine.fallbacks == before else { return nil }
                walls.append((last.gpu, last.ane))
            }
            let rows = engine.rows(m, k: k, n: n)
            return (walls.map(\.gpu).sorted()[1], Double(m - rows), walls.map(\.ane).sorted()[1], Double(rows))
        }
        guard let half = sides(0.5), let quarter = sides(0.75), half.aneRows != quarter.aneRows else { return nil }
        let gpuSlope = (half.gpu - quarter.gpu) / (half.gpuRows - quarter.gpuRows)
        let aneSlope = (half.ane - quarter.ane) / (half.aneRows - quarter.aneRows)
        let (gpuFixed, aneFixed) = (half.gpu - gpuSlope * half.gpuRows, half.ane - aneSlope * half.aneRows)
        // gpuFixed + gpuSlope · (m − r) = aneFixed + aneSlope · r
        let r = (gpuFixed + gpuSlope * Double(m) - aneFixed) / (gpuSlope + aneSlope)
        let range = NeuralEngine.shareRange
        let alone = gpuFixed + gpuSlope * Double(m)
        return (r.isFinite ? min(range.upperBound, max(range.lowerBound, 1 - r / Double(m))) : range.upperBound, alone)
    }
}
