import CryptoKit
import Foundation

// **What a family is made of on disk: published files, taken as they are.** Siliconed ships no weights and writes
// none: an installation downloads each file a family declares (`FamilyTraits.files`) from its publisher's repository,
// at a pinned commit, checks its size and its digest, and lays it down under `store/` — the very bytes a render reads
// and a check judges.

/// A repository on the Hub, at the commit an installation takes.
package struct Repository: Sendable, Hashable {
    package let name: String
    package let revision: String
    package init(_ name: String, _ revision: String) { (self.name, self.revision) = (name, revision) }

    private func address(_ kind: String, _ path: String) -> URL {
        URL(string: "https://huggingface.co/\(name)/\(kind)/\(revision)/\(path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)!)")!
    }
    /// The file's bytes.
    package func url(_ path: String) -> URL { address("resolve", path) }
    /// The file's page, for a human to read (a license, a model card).
    package func page(_ path: String) -> URL { address("blob", path) }
}

/// **A published file, pinned**: where it is published, where it lives under `store/`, and what its bytes must be.
/// The size and the digest are the recipe's, read once from the Hub at the pinned commit and found again on the files
/// the checks judged; a download that does not hash to it is thrown away.
package struct PublishedFile: Sendable, Hashable {
    package enum Digest: Sendable, Hashable {
        /// An LFS object: the sha256 of its bytes.
        case sha256(String)
        /// A file kept in git itself (a small JSON): its blob id, the SHA-1 of `blob <n>\0` and its bytes.
        case gitBlob(String)

        /// `sha256:<hex>` or `git:<hex>`: what a record keeps of it.
        package var text: String {
            switch self {
            case .sha256(let h): "sha256:\(h)"
            case .gitBlob(let h): "git:\(h)"
            }
        }
    }
    package let repository: Repository
    package let path: String
    /// Its place under `store/`: a distinctive published name at the top, a generic one in the family's folder.
    package let place: String
    package let bytes: Int
    package let digest: Digest

    package init(_ repository: Repository, _ path: String, at place: String, bytes: Int, sha256: String) {
        (self.repository, self.path, self.place, self.bytes, self.digest) = (repository, path, place, bytes, .sha256(sha256))
    }
    package init(_ repository: Repository, _ path: String, at place: String, bytes: Int, gitBlob: String) {
        (self.repository, self.path, self.place, self.bytes, self.digest) = (repository, path, place, bytes, .gitBlob(gitBlob))
    }

    /// `Z-Image-Turbo/vae/diffusion_pytorch_model.safetensors`: what a journal and an error name it by.
    package var label: String { "\(repository.name.split(separator: "/").last!)/\(path)" }

    /// The digest of the file at `path`, in this file's kind.
    package func digest(of path: String) throws -> Digest {
        switch digest {
        case .sha256: return .sha256(try Downloader.sha256(path))
        case .gitBlob:
            // Git keeps small text files only: read whole.
            guard let data = FileManager.default.contents(atPath: path) else { throw EngineError.fileUnreadable(path: path) }
            var hasher = Insecure.SHA1()
            hasher.update(data: Data("blob \(data.count)\u{0}".utf8))
            hasher.update(data: data)
            return .gitBlob(hasher.finalize().map { String(format: "%02x", $0) }.joined())
        }
    }
}

extension Family {
    /// Every file the version reads, the family's shared ones included.
    package func files(_ v: Variant) -> [PublishedFile] { traits.files[v] ?? [] }
}
