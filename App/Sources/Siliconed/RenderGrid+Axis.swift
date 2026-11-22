// An axis as `silicontrol grid` writes it, read (`RenderGrid.axis`).

import CoreGraphics
import Foundation

extension RenderGrid {
    /// **An axis as `silicontrol grid` writes it**, `kind[:which][=values]`:
    ///
    ///     prompt        the prompt's first group          prompt:2      its second
    ///     seed=42,7     these seeds                       seeds=4       4 from the base seed
    ///     steps=6,8,10  these step counts
    ///     lora=0.4,0.8  the stack's first LoRA            lora:NAME=…   the LoRA named NAME in
    ///                                                     lora:2=…      `loras` (or the 2nd)
    ///     model=z-image,qwen-image-2.1                    image         each edit image alone
    ///     loras=none,flat,ink   one LoRA added per cell   loras:0.6=…   at that strength (1 otherwise)
    ///     solver=euler,dpmpp-2m these solvers, by `Solver.rawValue`
    ///     start=0.45,0.6,0.75   the starting image's strengths (img2img)
    ///     detail=normal,more,most   these levels of Detail, by `Detail.rawValue`
    ///
    /// Values are separated by commas, decimals written with a point: the syntax of a command line,
    /// not of a language. `loras`: the stack's names, in its order. What does not read is
    /// `gridRefused(.unreadableAxis)`; a LoRA not in `loras`, `gridRefused(.noSuchLoRA)`. The
    /// grid's own checks (`init`) come after.
    package static func axis(_ text: String, loras: [String] = []) throws(EngineError) -> Axis {
        let t = text.trimmingCharacters(in: .whitespaces)
        let (head, list): (Substring, Substring?) = {
            guard let eq = t.firstIndex(of: "=") else { return (t[...], nil) }
            return (t[..<eq], t[t.index(after: eq)...])
        }()
        let kind: Substring, which: Substring?
        if let colon = head.firstIndex(of: ":") {
            kind = head[..<colon]; which = head[head.index(after: colon)...]
        } else {
            kind = head; which = nil
        }
        func values<T>(_ parse: (String) -> T?) throws(EngineError) -> [T] {
            guard let list else { throw .gridRefused(reason: .unreadableAxis) }
            var out: [T] = []
            for token in list.split(separator: ",", omittingEmptySubsequences: false) {
                guard let v = parse(token.trimmingCharacters(in: .whitespaces)) else {
                    throw .gridRefused(reason: .unreadableAxis)
                }
                out.append(v)
            }
            return out
        }
        switch kind.lowercased() {
        case "prompt":
            guard list == nil else { throw .gridRefused(reason: .unreadableAxis) }
            guard let which else { return .prompt(group: 0) }
            guard let n = Int(which), n >= 1 else { throw .gridRefused(reason: .unreadableAxis) }
            return .prompt(group: n - 1)
        case "seed" where which == nil:
            return .seeds(try values { UInt64($0) })
        case "seeds" where which == nil:
            let n = try values { Int($0) }
            guard n.count == 1 else { throw .gridRefused(reason: .unreadableAxis) }
            return .consecutiveSeeds(count: n[0])
        case "steps" where which == nil:
            return .steps(try values { Int($0) })
        case "model" where which == nil, "models" where which == nil:
            return .models(try values { $0.isEmpty ? nil : $0 })
        case "image" where which == nil && list == nil, "images" where which == nil && list == nil:
            return .images
        case "solver" where which == nil, "solvers" where which == nil:
            return .solvers(try values { Solver(rawValue: $0.lowercased()) })
        case "start" where which == nil:
            return .startStrength(try values { Double($0).flatMap { $0.isFinite ? $0 : nil } })
        case "detail" where which == nil:
            return .detail(try values { Detail(rawValue: $0.lowercased()) })
        case "loras":
            let strength: Double
            if let which {
                guard let f = Double(which), f.isFinite else { throw .gridRefused(reason: .unreadableAxis) }
                strength = f
            } else {
                strength = 1
            }
            return .addedLoRAs(try values { $0.isEmpty ? nil : ($0.lowercased() == "none" ? "" : $0) }, strength: strength)
        case "lora":
            let strengths = try values { Double($0).flatMap { $0.isFinite ? $0 : nil } }
            guard let which else {
                guard !loras.isEmpty else { throw .gridRefused(reason: .noSuchLoRA) }
                return .loraStrength(slot: 0, values: strengths)
            }
            if let slot = loras.firstIndex(of: String(which)) { return .loraStrength(slot: slot, values: strengths) }
            if let n = Int(which), loras.indices.contains(n - 1) { return .loraStrength(slot: n - 1, values: strengths) }
            throw .gridRefused(reason: .noSuchLoRA)
        default:
            throw .gridRefused(reason: .unreadableAxis)
        }
    }
}
