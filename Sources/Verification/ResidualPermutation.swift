import Darwin
import Foundation
import Siliconed

// **A summation order drawn, the engine's side** of `tools/oracle/krea2.py draw --permute`.
//
// The residual stream's channels are reordered everywhere at once: the input rows of every `Linear` that reads it,
// the output columns (and their scales) of every `Linear` that writes it, every per-channel vector (norm weights,
// modulation tables, the time embedding the final layer adds). The permuted model computes the same function; only
// the order of its sums over the hidden dimension changes. One draw of that order against another says how much a
// verdict at an ill-conditioned boundary owes to the arithmetic's order rather than to the route.
//
// The definition is the oracle's, to the bit, so that one seed gives one order on both sides: SplitMix64 drives a
// Fisher–Yates shuffle (`fisherYates`), `perm[c]` is the reference channel that permuted channel `c` carries
// (`x_permuted = x[..., perm]`), seed 0 is the identity.
//
// What a port by analogy would miss:
//   - a published convrot weight (int8, `W = (q·s)·R` along K, R = H/√G block-diagonal) can only have its K rows
//     permuted **by whole groups**: R is the same Hadamard in every group, so a permutation of groups commutes with
//     it, and nothing finer does. The group family is the only one an int8 map can draw; `PermutedMap` refuses any
//     other on a rotated weight rather than write a different model;
//   - the per-column scale of an int8 weight travels with its column when the output side is permuted, and stays put
//     when the input side is;
//   - the map's `Linear`s are transposed (`[input, output]`): the oracle's "rows" (torch's output axis) are the
//     map's columns, and the reverse.
//
// Nothing here is on a render's path: the permuted model is a separate map file, written for a check.

package enum ResidualPermutation {
    /// The convrot group of every int8 DiT of the catalogue, and the oracle's `PERMUTATION_GROUP`.
    package static let group = 256

    /// SplitMix64 (Steele, Lea & Flood, 2014): advances `state` and returns its output, modulo 2⁶⁴.
    package static func splitMix64(_ state: inout UInt64) -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// `[0, n)` shuffled: the state starts at `seed`; for `i = n−1` down to 1, `j = next mod (i + 1)`, `a[i]` and `a[j]`
    /// swapped (the modulo's bias is part of the definition) — the oracle's `fisher_yates`.
    package static func fisherYates(_ n: Int, seed: UInt64) -> [Int] {
        var a = Array(0..<n), state = seed
        var i = n - 1
        while i > 0 {
            let j = Int(splitMix64(&state) % UInt64(i + 1))
            a.swapAt(i, j)
            i -= 1
        }
        return a
    }

    /// The permutation of `seed`: whole groups of `group` channels shuffled (each keeps its inner order), or every
    /// channel when `group` is `nil`. Seed 0 is the identity — the oracle's `residual_permutation`.
    package static func residual(dim: Int, seed: UInt64, group: Int? = ResidualPermutation.group) -> [Int] {
        guard seed != 0 else { return Array(0..<dim) }
        guard let group else { return fisherYates(dim, seed: seed) }
        precondition(dim % group == 0, "\(dim) channels are not whole groups of \(group)")
        return fisherYates(dim / group, seed: seed).flatMap { g in (g * group)..<((g + 1) * group) }
    }

    /// `x` seen as rows of `perm.count`, its channels permuted: `out[r, c] = x[r, perm[c]]`.
    package static func permuted(_ x: UnsafeBufferPointer<Float>, _ perm: [Int]) -> [Float] {
        let d = perm.count
        precondition(x.count % d == 0, "\(x.count) values are not rows of \(d)")
        var out = [Float](repeating: 0, count: x.count)
        for r in 0..<(x.count / d) {
            for c in 0..<d { out[r * d + c] = x[r * d + perm[c]] }
        }
        return out
    }

    /// The inverse of `permuted`: a boundary of the permuted model put back in the reference's channel order,
    /// `out[r, perm[c]] = x[r, c]`.
    package static func restored(_ x: [Float], _ perm: [Int]) -> [Float] {
        let d = perm.count
        precondition(x.count % d == 0, "\(x.count) values are not rows of \(d)")
        var out = [Float](repeating: 0, count: x.count)
        for r in 0..<(x.count / d) {
            for c in 0..<d { out[r * d + perm[c]] = x[r * d + c] }
        }
        return out
    }

    /// Is `perm` a permutation of whole groups of `group` (what a convrot weight's rows admit)?
    package static func movesWholeGroups(_ perm: [Int], group: Int) -> Bool {
        guard perm.count % group == 0 else { return false }
        for start in stride(from: 0, to: perm.count, by: group) {
            let first = perm[start]
            guard first % group == 0 else { return false }
            for i in 1..<group where perm[start + i] != first + i { return false }
        }
        return true
    }
}

/// **A Krea 2 DiT map with its residual stream permuted**, written beside the published one: what each tensor does
/// with the residual stream is decided by its name (the oracle's `_ROWS`, `_COLUMNS`, `_TABLES`, `_UNTOUCHED`, in the
/// map's transposed layout); a tensor with no rule refuses the map — one forgotten would make another model, not
/// another rounding. The file is a clone of the source (`clonefile`, copy on write), each moved tensor rewritten in
/// place: the untouched ones (the text fusion, `time_embed.linear_1`) cost no disk.
package enum Krea2PermutedMap {
    /// How one tensor reads or writes the residual stream, in the map's layout `[rows, columns]` (a vector is one row).
    enum Move {
        case untouched
        /// `out[r, c] = in[rows?[r] ?? r, columns?[c] ?? c]`.
        case permute(rows: [Int]?, columns: [Int]?)
    }

    static func move(_ name: String, perm: [Int]) throws -> Move {
        let d = perm.count
        let key: String
        if name.hasPrefix("transformer_blocks."), let rest = name.split(separator: ".", maxSplits: 2).dropFirst(2).first {
            key = "block:" + rest
        } else {
            key = name
        }
        // The map's output axis (torch's rows): the Linears that write the stream, and the per-channel vectors.
        let writes: Set<String> = ["img_in.weight", "time_embed.linear_2.weight", "txt_in.linear_2.weight",
                                   "block:attn.to_out.0.weight", "block:ff.down.weight",
                                   "img_in.bias", "time_embed.linear_2.bias", "txt_in.linear_2.bias",
                                   "final_layer.norm.weight", "block:norm1.weight", "block:norm2.weight",
                                   "final_layer.scale_shift_table", "block:scale_shift_table"]
        // The map's input axis (torch's columns): the Linears that read the stream.
        let reads: Set<String> = ["final_layer.linear.weight", "block:attn.to_q.weight", "block:attn.to_k.weight",
                                  "block:attn.to_v.weight", "block:attn.to_gate.weight", "block:ff.gate.weight",
                                  "block:ff.up.weight"]
        let untouched: Set<String> = ["time_embed.linear_1.weight", "time_embed.linear_1.bias", "txt_in.norm.weight",
                                      "txt_in.linear_1.weight", "txt_in.linear_1.bias", "final_layer.linear.bias",
                                      "block:attn.norm_q.weight", "block:attn.norm_k.weight"]
        if writes.contains(key) { return .permute(rows: nil, columns: perm) }
        if reads.contains(key) { return .permute(rows: perm, columns: nil) }
        // `[d, 6d]`: reads temb (permuted), writes six per-channel vectors.
        let six = (0..<6).flatMap { j in perm.map { j * d + $0 } }
        if key == "time_mod_proj.weight" { return .permute(rows: perm, columns: six) }
        if key == "time_mod_proj.bias" { return .permute(rows: nil, columns: six) }
        if untouched.contains(key) || key.hasPrefix("text_fusion.") { return .untouched }
        throw Artifact.Failure.badHeader("\(name): no rule for the residual permutation")
    }

    /// **One draw of the summation order**: `draws` with a single request.
    package static func draw(source: String, seed: UInt64, text: UnsafeBufferPointer<Float>, directory: String? = nil,
                             evaluate: (Artifact, UnsafeBufferPointer<Float>) throws -> [String: [Float]]) throws -> [String: [Float]] {
        try draws(source: source, requests: [[seed]], text: text, directory: directory) { _, map, moved in try evaluate(map, moved) }[0][seed]!
    }

    /// **Draws of the summation order, seed outside** (`SeedOutside`): for each seed any request asks for, the map at
    /// `source` permuted by the group permutation of that seed is written **once** in `directory` (by default a temporary
    /// folder on the source's own volume, so that `clonefile` shares the untouched tensors), every request of that seed
    /// evaluated on it, then the map removed — at most one map per seed on the disk, and one write, however many
    /// requests (a trajectory's failing steps) share it. `text` (`[T, d]`, `txt_in`'s output) is permuted alike.
    /// `evaluate(request, map, text)` runs the permuted model and returns its boundaries by name, `model_out` included;
    /// every boundary but `model_out` (the latent, outside the residual stream) comes back in the reference's channel
    /// order. Seed 0 evaluates `source` itself. Returns, per request, its boundaries by seed.
    package static func draws(source: String, requests: [[UInt64]], text: UnsafeBufferPointer<Float>, directory: String? = nil,
                              evaluate: (Int, Artifact, UnsafeBufferPointer<Float>) throws -> [String: [Float]])
        throws -> [[UInt64: [String: [Float]]]] {
        guard requests.contains(where: { !$0.isEmpty }) else { return requests.map { _ in [:] } }
        let d = Krea2DiT.Config(header: try Artifact(path: source).header).dim
        typealias Map = (path: String, temporary: URL?, perm: [Int], text: [Float])
        return try SeedOutside.run(requests, prepare: { seed -> Map in
            let perm = ResidualPermutation.residual(dim: d, seed: seed)
            guard seed != 0 else { return (source, nil, perm, Array(text)) }
            let folder: URL, temporary: URL?
            if let directory {
                folder = URL(fileURLWithPath: directory); temporary = nil
            } else {
                folder = try FileManager.default.url(for: .itemReplacementDirectory, in: .userDomainMask,
                                                     appropriateFor: URL(fileURLWithPath: source), create: true)
                temporary = folder
            }
            let path = folder.appendingPathComponent("krea2-permgroup\(seed).silicon").path
            do { try write(source: source, destination: path, perm: perm) } catch {
                try? FileManager.default.removeItem(atPath: path)
                if let temporary { try? FileManager.default.removeItem(at: temporary) }
                throw error
            }
            return (path, temporary, perm, ResidualPermutation.permuted(text, perm))
        }, release: { map in
            guard map.path != source else { return }
            try? FileManager.default.removeItem(atPath: map.path)
            if let temporary = map.temporary { try? FileManager.default.removeItem(at: temporary) }
        }, evaluate: { request, _, map in
            let boundaries = try autoreleasepool {
                try map.text.withUnsafeBufferPointer { try evaluate(request, try Artifact(path: map.path), $0) }
            }
            var restored: [String: [Float]] = [:]
            for (name, values) in boundaries {
                restored[name] = name == "model_out" ? values : ResidualPermutation.restored(values, map.perm)
            }
            return restored
        })
    }

    /// Writes at `destination` the map at `source` with its residual stream permuted by `perm` (`ResidualPermutation`).
    /// Returns the bytes rewritten.
    @discardableResult
    package static func write(source: String, destination: String, perm: [Int]) throws -> Int {
        let artifact = try Artifact(path: source)
        guard artifact.header["kind"] as? String == "krea2-turbo-dit", artifact.linearWeightsTransposed else {
            throw Artifact.Failure.badHeader("\(source): a Krea 2 DiT map with transposed Linears expected")
        }
        let d = Krea2DiT.Config(header: artifact.header).dim
        guard perm.count == d, Set(perm).count == d, perm.allSatisfy({ (0..<d).contains($0) }) else {
            throw Artifact.Failure.misuse("a permutation of \(d) channels expected, \(perm.count) given")
        }
        // Every rule first: a map refused is refused before a byte is written.
        let moves = try artifact.order.map { ($0, try move($0, perm: perm)) }

        if FileManager.default.fileExists(atPath: destination) { try FileManager.default.removeItem(atPath: destination) }
        if clonefile(source, destination, 0) != 0 {
            try FileManager.default.copyItem(atPath: source, toPath: destination)
        }
        let input = open(source, O_RDONLY), output = open(destination, O_RDWR)
        defer { if input >= 0 { close(input) }; if output >= 0 { close(output) } }
        guard input >= 0, output >= 0 else { throw Artifact.Failure.cannotOpen(input < 0 ? source : destination, errno) }

        var written = 0
        for (name, m) in moves {
            guard case let .permute(rows, columns) = m else { continue }
            let t = artifact.tensors[name]!
            guard !t.dtype.isPacked, t.scale?.block == nil else {
                throw Artifact.Failure.badHeader("\(name): \(t.dtype.rawValue) in blocks — no residual permutation for it")
            }
            let (height, width) = t.shape.count == 1 ? (1, t.shape[0]) : (t.shape[0], t.shape[1])
            guard t.shape.count <= 2, height * width == t.count else {
                throw Artifact.Failure.badHeader("\(name): shape \(t.shape)")
            }
            if let rows, rows.count != height { throw Artifact.Failure.badHeader("\(name): \(height) rows, \(rows.count) permuted") }
            if let columns, columns.count != width { throw Artifact.Failure.badHeader("\(name): \(width) columns, \(columns.count) permuted") }
            if let rows, let rotation = t.rotation, !ResidualPermutation.movesWholeGroups(rows, group: rotation.group) {
                throw Artifact.Failure.badHeader("\(name): rotated by groups of \(rotation.group) — only whole groups may move")
            }
            var bytes = [UInt8](repeating: 0, count: t.bytes)
            try bytes.withUnsafeMutableBytes { try readFully(input, $0, at: t.offset, name) }
            var out = bytes
            bytes.withUnsafeBytes { from in
                out.withUnsafeMutableBytes { to in
                    func move<E>(_: E.Type) {
                        let source = from.baseAddress!.assumingMemoryBound(to: E.self)
                        let target = to.baseAddress!.assumingMemoryBound(to: E.self)
                        for r in 0..<height {
                            let s = source + (rows?[r] ?? r) * width, o = target + r * width
                            if let columns {
                                for c in 0..<width { o[c] = s[columns[c]] }
                            } else {
                                o.update(from: s, count: width)
                            }
                        }
                    }
                    switch t.dtype.size {
                    case 1: move(UInt8.self)
                    case 2: move(UInt16.self)
                    default: move(UInt32.self)
                    }
                    // A scale per column travels with its column; per tensor, or on the input side, it stays.
                    if let scale = t.scale, scale.shape.count == 1, let columns {
                        precondition(scale.shape[0] == width)
                        let size = scale.dtype.size
                        for c in 0..<width {
                            memcpy(to.baseAddress! + scale.offset + c * size, from.baseAddress! + scale.offset + columns[c] * size, size)
                        }
                    }
                }
            }
            try out.withUnsafeBytes { try writeFully(output, $0, at: t.offset, name) }
            written += t.bytes
        }
        return written
    }

    private static func readFully(_ fd: Int32, _ buffer: UnsafeMutableRawBufferPointer, at offset: Int, _ name: String) throws {
        var done = 0
        while done < buffer.count {
            let n = pread(fd, buffer.baseAddress! + done, buffer.count - done, off_t(offset + done))
            guard n > 0 else { throw Artifact.Failure.truncated("\(name): read \(done) of \(buffer.count) bytes") }
            done += n
        }
    }

    private static func writeFully(_ fd: Int32, _ buffer: UnsafeRawBufferPointer, at offset: Int, _ name: String) throws {
        var done = 0
        while done < buffer.count {
            let n = pwrite(fd, buffer.baseAddress! + done, buffer.count - done, off_t(offset + done))
            guard n > 0 else { throw Artifact.Failure.truncated("\(name): wrote \(done) of \(buffer.count) bytes") }
            done += n
        }
    }
}

/// **Seed outside, evaluations inside**: what a costly per-seed setup (a permuted map, ~26 GB written for Krea 2's
/// Standard) is shared by. `requests[i]` is the seeds request `i` wants; each seed of their union, in increasing order,
/// is prepared **once**, every request that wants it evaluated on it, then released before the next seed is prepared —
/// one setup alive at a time. Returns, per request, its results by seed.
package enum SeedOutside {
    package static func run<Map>(_ requests: [[UInt64]], prepare: (UInt64) throws -> Map, release: (Map) -> Void,
                                 evaluate: (Int, UInt64, Map) throws -> [String: [Float]]) throws -> [[UInt64: [String: [Float]]]] {
        var results = [[UInt64: [String: [Float]]]](repeating: [:], count: requests.count)
        for seed in Set(requests.joined()).sorted() {
            let map = try prepare(seed)
            defer { release(map) }
            for (i, seeds) in requests.enumerated() where seeds.contains(seed) {
                results[i][seed] = try evaluate(i, seed, map)
            }
        }
        return results
    }
}
