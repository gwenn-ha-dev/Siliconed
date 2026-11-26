import CoreAI
import Foundation

/// **An argument bound by its name** (`Resolver`): a weight, a tensor or rows; a call's views over the arena.
extension Block {
    /// What an argument resolves to: a weight by its name; a tensor (a row of it); a per-row tensor, its stream's other
    /// buffer, from row 0.
    enum Target: Equatable { case weight(String), tensor(String, row: Int?), rows(String, next: Bool, whole: Bool) }

    struct Resolver: Sendable {
        let forms: [String: Form]
        let perRow: [String: Rows]
        let nextOf: [String: String]
        let tensorNames: Set<String>
        /// Each group's weights' names, the residents' under ""; each group's held weights' (`Group.held`).
        let weightNames: [String: Set<String>]
        let heldNames: [String: Set<String>]

        init(_ d: Declaration, forms: [String: Form], weights: [String: Set<String>], held: [String: Set<String>] = [:]) {
            self.forms = forms
            perRow = Dictionary(uniqueKeysWithValues: d.perRow.map { ($0.name, $0) })
            nextOf = Dictionary(uniqueKeysWithValues: d.perRow.compactMap { r in r.next.map { ($0, r.name) } })
            tensorNames = Set(d.tensors.map(\.name))
            weightNames = weights
            heldNames = held
        }

        /// `shadowed`: an input of the name of one of the function's outputs (a product's codes, and its result): the
        /// bindings are the output's. A per-row tensor named — by the argument, a binding or `whole` — is the group's own
        /// (`<group>.<name>`) where the group has one: a layer's cache, a layer's keys.
        func resolve(_ arg: String, input: Bool, shadowed: Bool = false, group: Group?, pass: Pass?, step: Step?, call: [String: Bind]) -> Target {
            let target = named(arg, input: input, shadowed: shadowed, group: group, pass: pass, step: step, call: call)
            if case .rows(let t, let next, let whole) = target, let g = group, perRow["\(g.name).\(t)"] != nil {
                return .rows("\(g.name).\(t)", next: next, whole: whole)
            }
            return target
        }

        private func named(_ arg: String, input: Bool, shadowed: Bool, group: Group?, pass: Pass?, step: Step?, call: [String: Bind]) -> Target {
            let form = group.map { forms[$0.form]! }
            if !shadowed, let b = call[arg] ?? step?.bind[arg] ?? pass?.bind[arg] ?? form?.bind[arg] {
                switch b {
                case .rows(let t): return .rows(t, next: false, whole: false)
                case .next(let t): return .rows(t, next: true, whole: false)
                case .whole(let t): return .rows(t, next: false, whole: true)
                case .tensor(let t): return .tensor(t, row: nil)
                case .row(let t, let i): return .tensor(t, row: i)
                }
            }
            if step?.whole.contains(arg) == true { return .rows(arg, next: false, whole: true) }
            if input, let g = group, weightNames[g.name]?.contains(arg) == true { return .weight(arg) }
            if input, let g = group, heldNames[g.name]?.contains(arg) == true { return .weight("\(g.name).\(arg)") }
            if input, weightNames[""]?.contains(arg) == true { return .weight(arg) }
            if perRow[arg] != nil || group.map({ perRow["\($0.name).\(arg)"] != nil }) == true { return .rows(arg, next: false, whole: false) }
            if let s = nextOf[arg] { return .rows(s, next: true, whole: false) }
            if let g = group, tensorNames.contains("\(g.name).\(arg)") { return .tensor("\(g.name).\(arg)", row: nil) }
            if tensorNames.contains(arg) { return .tensor(arg, row: nil) }
            preconditionFailure("no binding for \(arg)\(group.map { " in \($0.name)" } ?? "")\(step.map { ", step \($0.name)" } ?? "")")
        }
    }

    /// The stage's tensor of per-row tensor `name`: a stream's current buffer or its other.
    func tensor(_ name: String, next: Bool) -> String {
        guard let c = current[name] else { return name }
        return "\(name)#\(next ? 1 - c : c)"
    }

    /// The views of function `f`'s arguments and results over rows from `start` (the ANE's: surfaces), but those of `skip`.
    func views(_ f: String, start: Int, group: Group?, pass: Pass?, step: Step?, call: [String: Bind], neural: Bool,
               skip: Set<String> = [], written: inout Set<String>) -> (inputs: [String: View], outputs: [(String, View)]) {
        let sig = signatures[f]!, outputs = Set(sig.outputs.map(\.0))
        func view(_ arg: String, _ shape: [Int], _ type: NDArray.ScalarType, input: Bool) -> View {
            switch resolver.resolve(arg, input: input, shadowed: input && outputs.contains(arg), group: group, pass: pass, step: step, call: call) {
            case .weight(let w):
                // Codes the ANE reads from its copy; a weight of the same name in f16 (another form's) as it lies.
                if neural && type == .int8 && copies.contains(w) { return stage.neuralView(Self.copy(w), shape, type) }
                guard let p = held?.weights.weights[w] ?? stage.residents[w] else { preconditionFailure("\(f): no weight \(w)") }
                return stage.view(p, shape)
            case .tensor(let t, let row):
                let offset = row.map { $0 * Arena.aligned(shape.reduce(1, *) * Layout.element(type)) } ?? 0
                return neural ? stage.neuralView(t, shape, type, bytes: offset) : stage.view(t, shape, type, bytes: offset)
            case .rows(let t, let next, let whole):
                guard let r = perRow[t] else { preconditionFailure("\(f).\(arg): no per-row tensor \(t)") }
                precondition(r.type == type, "\(f).\(arg): \(t) is \(r.type), the function reads \(type)")
                if !input && current[t] != nil && next { written.insert(t) }
                let name = tensor(t, next: next), offset = whole ? 0 : start * r.rowBytes
                precondition(offset + shape.reduce(1, *) * Layout.element(type) <= (r.rows ?? declaration.rows) * r.rowBytes,
                             "\(f).\(arg): past the rows of \(t)")
                return neural ? stage.neuralView(name, shape, type, bytes: offset) : stage.view(name, shape, type, bytes: offset)
            }
        }
        var inputs: [String: View] = [:]
        for (n, s, t) in sig.inputs where !skip.contains(n) { inputs[n] = view(n, s, t, input: true) }
        return (inputs, sig.outputs.filter { !skip.contains($0.0) }.map { n, s, t in (n, view(n, s, t, input: false)) })
    }
}
