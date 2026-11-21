import CryptoKit
import Foundation
import Synchronization

/// **What a render keeps for the next one: the encoded text.**
///
/// A conditioning depends on the prompt's tokens and on the encoder's weights — not on the seed, the format, the steps,
/// the solver or a LoRA (a LoRA patches the DiT). A series over any of those encoded the same prompt once per cell:
/// measured on Z-Image at 512², four seeds, 6.75 s a cell against 4.9–5.0 s an image of a lot (M5 Pro).
///
/// **In this process's memory, nowhere else**: nothing of a prompt is written to disk, and a new build never reads what
/// an older encoder computed — a check (`siliconed check`), a process of its own, always runs its encoder. The values
/// are kept in fp32, which holds what an encoder writes bit for bit; the newest are kept, under `capacity`.
package enum RenderCache {
    /// What the kept conditionings may weigh together, in bytes; one that is larger alone is not kept.
    static let capacity = 64 << 20

    /// **Everything an encoding depends on, digested**: who encodes, the weights' files as they are on disk, the tokens,
    /// and whatever else that encoder reads (a picture's patches).
    package struct Key {
        private var hash = SHA256()

        /// `stage`: the encoder and its configuration — two families may read one file differently.
        /// The Neural Engine switch in force is part of it: the engine may compute an encoder otherwise with it on.
        package init(_ stage: String) { text(stage); text(CoreAI.neuralEngineInForce ? "neural engine" : "gpu") }

        /// The file as it is on disk: its path, its size, its modification date — an imported model replaced under
        /// the same name is another file.
        package mutating func file(_ path: String) throws {
            let attributes = try FileManager.default.attributesOfItem(atPath: (path as NSString).resolvingSymlinksInPath)
            text(path)
            integers([(attributes[.size] as? NSNumber)?.intValue ?? 0,
                      Int(((attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0) * 1000)])
        }

        package mutating func integers(_ values: [Int]) {
            count(values.count)
            values.withUnsafeBytes { hash.update(bufferPointer: $0) }
        }

        package mutating func floats(_ values: UnsafeMutableBufferPointer<Float>) {
            count(values.count)
            hash.update(bufferPointer: UnsafeRawBufferPointer(values))
        }

        private mutating func text(_ s: String) {
            let bytes = Data(s.utf8)
            count(bytes.count)
            hash.update(data: bytes)
        }

        /// Each field begins with its length: `[1, 2] [3]` and `[1] [2, 3]` are two keys.
        private mutating func count(_ n: Int) { withUnsafeBytes(of: n) { hash.update(bufferPointer: $0) } }

        var digest: Data { Data(hash.finalize()) }
    }

    private struct Kept {
        let key: Data
        let shape: [Int]
        let values: [Float]
        var bytes: Int { values.count * MemoryLayout<Float>.size }
    }

    /// The newest last.
    private static let kept = Mutex<[Kept]>([])

    /// **The conditioning under `key`**: read back if a render of this model kept it, encoded and kept otherwise.
    /// A hand-composed chain (`Model.library` is `nil`) keeps nothing.
    static func conditioning(_ key: Key, of model: Model, encode: () throws -> HostTensor) rethrows -> (values: HostTensor, cached: Bool) {
        guard model.library != nil else { return (try encode(), false) }
        let digest = key.digest
        let found = kept.withLock { all -> Kept? in
            guard let i = all.firstIndex(where: { $0.key == digest }) else { return nil }
            let entry = all.remove(at: i)
            all.append(entry)
            return entry
        }
        if let found { return (HostTensor(found.shape, floats: found.values), true) }
        let encoded = try encode()
        let entry = Kept(key: digest, shape: encoded.shape, values: encoded.floats)
        guard entry.bytes <= capacity else { return (encoded, false) }
        kept.withLock { all in
            all.append(entry)
            var total = all.reduce(0) { $0 + $1.bytes }
            while total > capacity { total -= all.removeFirst().bytes }
        }
        return (encoded, false)
    }
}
