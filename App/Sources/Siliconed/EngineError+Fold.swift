// The door: every internal failure folded into its `EngineError` case, and the missing file it folds.

import Foundation
import SiliCore

extension EngineError {
    /// **A write that ran out of room**: the disk or the user's quota, said as such (`diskFull`), never a write refused.
    static func ranOutOfRoom(_ error: any Error) -> Bool {
        let ns = error as NSError
        return (ns.domain == NSPOSIXErrorDomain && ranOutOfRoom(errno: Int32(ns.code)))
            || (ns.domain == NSCocoaErrorDomain && ns.code == NSFileWriteOutOfSpaceError)
    }

    /// The same judge, for a system call's `errno`.
    static func ranOutOfRoom(errno code: Int32) -> Bool { code == ENOSPC || code == EDQUOT }

    /// **The fold.** An `EngineError` comes back as is; each internal failure becomes the case an
    /// app can name; anything else is an `internalFailure`.
    package init(_ error: any Error) {
        switch error {
        case let e as EngineError: self = e
        case is CancellationError: self = .cancelled
        // The engine's refusal under B, before it allocated: a lack of memory, with its numbers.
        case let e as OverBudget: self = .insufficientMemory(needed: e.needed, budget: e.budget)
        case let e as MissingFile: self = .fileMissing(path: e.path)
        case let e as Request.Failure:
            switch e {
            case .emptyPrompt: self = .emptyPrompt
            case .steps(let n): self = .invalidSteps(steps: n, allowed: [])
            case let .references(n, max): self = .tooManyReferences(count: n, max: max)
            case let .solver(solver, model): self = .solverUnsupported(solver: solver.rawValue, model: model)
            case .detail(let model): self = .detailUnsupported(model: model)
            case let .promptTooLong(tokens, max): self = .promptTooLong(tokens: tokens, max: max)
            }
        case let e as Strength.Failure:
            switch e {
            case .outOfBounds(let s): self = .strengthOutOfRange(strength: s)
            case let .noEvaluation(s, steps, floor): self = .strengthTooLow(strength: s, steps: steps, floor: floor)
            case .withoutEncoder(let model): self = .imageToImageUnsupported(model: model)
            }
        case let e as Model.Failure:
            switch e { case .unknown(let name): self = .unknownModel(model: name) }
        case let e as ImageRGB.Failure:
            switch e {
            case .unreadable(let file): self = .imageUnreadable(file: file)
            }
        case let e as Chain.Failure:
            switch e {
            case let .incompatibleText(output, entry): self = .incompatibleChain(output: "\(output)", input: "\(entry)")
            case let .latentIncompatible(output, entry), let .imageIncompatible(output, entry):
                self = .incompatibleChain(output: "\(output)", input: "\(entry)")
            case .textWithoutImages(let format): self = .incompatibleChain(output: "text", input: "\(format)")
            }
        case let e as Safetensors.Failure:
            switch e {
            case let .cannotOpen(path, code): self = code == ENOENT ? .fileMissing(path: path) : .fileUnreadable(path: path)
            case .cannotStat(let path, _), .cannotMap(let path, _): self = .fileUnreadable(path: path)
            case .missingTensor(let file, _): self = .corruptMap(file: file)
            case .badHeader, .unknownDType: self = .corruptMap(file: nil)
            }
        case let e as Tokenizer.Failure:
            switch e {
            case .fileNotFound(let path): self = .fileMissing(path: path)
            case .malformed, .unsupported: self = .corruptMap(file: nil)
            }
        case let e as Qwen3VLChat.Failure:
            switch e { case .reservedText(let text): self = .promptReservedText(text: text) }
        case is T5Tokenizer.Failure, is OrderedJSON.ParseError: self = .corruptMap(file: nil)
        default:
            if Self.ranOutOfRoom(error) {
                self = .diskFull(needed: nil, available: nil)
            } else if let url = error as? URLError {
                self = .downloadInterrupted(file: url.failingURL?.lastPathComponent ?? "?")
            } else {
                self = .internalFailure(component: "\(type(of: error))", detail: "\(error)")
            }
        }
    }

    /// **The door**: runs `body` and throws only `EngineError`. Every function the app calls that throws
    /// passes through it (or throws an `EngineError` itself).
    package static func boundary<T>(_ body: () throws -> T) throws(EngineError) -> T {
        do { return try body() } catch { throw EngineError(error) }
    }
}

/// **A missing file — internal to the package**: a weights file from `store/`, a component
/// installed under `store/composants/` (tokenizer, VAE, configs). Distinct from a file that is
/// present but unreadable or damaged, which throws its reader's error. At the door it becomes
/// `EngineError.fileMissing`, or `modelNotInstalled` when it is a model being built (`Model.named`).
package struct MissingFile: Error, CustomStringConvertible, Equatable {
    package enum Nature: Sendable, Equatable {
        /// Under `store/`: a weights file — it is installed.
        case map
        /// Under `store/composants/`: a published file of a family (tokenizer, VAE) — it is installed.
        case published
        /// A golden tensor of a check (`goldens-*.safetensors`, beside the oracles): it is not
        /// installed, the reference writes it, by its oracle.
        case golden
    }
    package let path: String
    package let nature: Nature
    /// A golden is recognized by its name whoever reads it (`Safetensors` opens goldens and
    /// published files alike, and only knows it was asked for a file).
    package init(_ path: String, _ nature: Nature) {
        self.path = path
        self.nature = Self.isGolden(path) ? .golden : nature
    }

    package static func isGolden(_ path: String) -> Bool {
        let name = (path as NSString).lastPathComponent
        return name.hasPrefix("goldens-") && name.hasSuffix(".safetensors")
    }

    package var description: String {
        switch nature {
        case .map: return "map missing: \(path) — install the model or import it"
        case .published: return "component missing: \(path) — install the model"
        case .golden:
            return "golden missing: \(path) — goldens are written by the reference oracle, not installed "
                + "(the .json of the same name records the command that made it)"
        }
    }
}
