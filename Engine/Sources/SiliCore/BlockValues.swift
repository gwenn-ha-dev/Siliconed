import CoreAI
import Foundation

/// **What the CPU reads and writes**: a tensor's bytes, rows copied, values widened to f32; the block closed.
extension Block {
    /// The bytes of tensor `name`, every call done.
    public func pointer(_ name: String) async -> UnsafeMutableRawPointer {
        await settled()
        return stage.pointer(name)
    }

    /// The bytes of per-row tensor `name` from row `row` (a stream's current buffer, or its other), every call done.
    public func rows(_ name: String, from row: Int = 0, next: Bool = false) async -> UnsafeMutableRawPointer {
        await settled()
        return stage.pointer(tensor(name, next: next)) + row * perRow[name]!.rowBytes
    }

    /// The bytes of a row of per-row tensor `name`.
    public func rowBytes(_ name: String) -> Int { perRow[name]!.rowBytes }

    /// **Rows `range` of per-row tensor `name` copied into per-row tensor `to` from its row `row`** (a stream's current
    /// buffer each, `to`'s other under `next`), every call done: rows a pass computed set where another pass reads them.
    public func copy(_ name: String, rows range: Range<Int>, to: String, at row: Int, next: Bool = false) async {
        let (from, into) = Self.copied(perRow[name]!, rows: range, to: perRow[to]!, at: row, block: declaration.rows)
        await settled()
        memcpy(stage.pointer(tensor(to, next: next)) + into.lowerBound, stage.pointer(tensor(name, next: false)) + from.lowerBound, from.count)
    }

    /// The bytes a `copy` reads and writes: rows of the same width and type, within both tensors' rows.
    static func copied(_ a: Rows, rows range: Range<Int>, to b: Rows, at row: Int, block: Int) -> (from: Range<Int>, into: Range<Int>) {
        precondition(a.width == b.width && a.type == b.type, "\(a.name) to \(b.name): rows of another width or type")
        precondition(range.lowerBound >= 0 && range.upperBound <= a.rows ?? block && row >= 0 && row + range.count <= b.rows ?? block,
                     "\(a.name) \(range) to \(b.name) at \(row): past their rows")
        let n = range.count * a.rowBytes
        return (range.lowerBound * a.rowBytes ..< range.lowerBound * a.rowBytes + n, row * b.rowBytes ..< row * b.rowBytes + n)
    }

    /// **Rows `range` of per-row tensor `name` as true values**, in f32: its elements widened, times its factor's.
    public func values(_ name: String, rows range: Range<Int>) async -> [Float] {
        var out = [Float](repeating: 0, count: range.count * perRow[name]!.width)
        await values(name, rows: range, into: &out)
        return out
    }

    /// The same into `out`, `range.count · width` values: storage its client made before the stage was planned, so that
    /// its floor counts it (made while the stage lives, it is read as Core AI's share).
    public func values(_ name: String, rows range: Range<Int>, into out: inout [Float]) async {
        let (base, factor) = await sources(name, range, count: out.count)
        out.withUnsafeMutableBufferPointer { widen(name, base, factor, into: $0) }
    }

    /// The same into a buffer (a check's, part of one).
    public func values(_ name: String, rows range: Range<Int>, into out: UnsafeMutableBufferPointer<Float>) async {
        let (base, factor) = await sources(name, range, count: out.count)
        widen(name, base, factor, into: out)
    }

    /// The bytes of `range`'s rows of `name` and of its factor's, read into `count` values.
    private func sources(_ name: String, _ range: Range<Int>, count: Int) async -> (UnsafeMutableRawPointer, UnsafeMutableRawPointer?) {
        precondition(count == range.count * perRow[name]!.width, "\(name): \(range.count) rows into \(count) values")
        let base = await rows(name, from: range.lowerBound)
        guard let f = perRow[name]!.factor else { return (base, nil) }
        return (base, await rows(f, from: range.lowerBound))
    }

    private func widen(_ name: String, _ base: UnsafeMutableRawPointer, _ factor: UnsafeMutableRawPointer?, into out: UnsafeMutableBufferPointer<Float>) {
        let r = perRow[name]!
        func value(_ p: UnsafeMutableRawPointer, _ type: NDArray.ScalarType, _ i: Int) -> Float {
            switch type {
            case .float32: p.assumingMemoryBound(to: Float.self)[i]
            case .float16: Float(p.assumingMemoryBound(to: Float16.self)[i])
            default: Float(bitPattern: UInt32(p.assumingMemoryBound(to: UInt16.self)[i]) << 16)
            }
        }
        let ft = r.factor.map { perRow[$0]!.type }
        for i in 0..<out.count { out[i] = value(base, r.type, i) * (factor.map { value($0, ft!, i / r.width) } ?? 1) }
    }

    /// Everything let go, and the footprint waited for (`Stage.close`): the functions the block loaded itself first.
    @discardableResult
    public func close() -> Stage.Settled {
        // A copy prepared ahead writes into the arena: done, or its wait for the ring's next use given up, first.
        stage.stopPeeking()
        copier.sync {}
        let end = DispatchTime.now().uptimeNanoseconds
        // A run that ended on its measured cycle at 0 rows: decided here, the next opening at the share kept.
        if case .measured(let rows, let seconds) = zero, let b = begun {
            _ = zeroMeasured(at: Double(end &- b.at) * 1e-9 - (b.moved ?? 0) - (b.climbed ?? 0), share: rows, seconds)
        }
        cycled(until: end)
        functions = [:]
        return stage.close()
    }
}
