// What `add` and `grid` check alike, on this Mac and on another.

import AppKit
import Foundation
import Measurement
import Observation
import Siliconed
import SilicontrolHelp

extension RemoteControl {
    /// The flags of one command that the other does not take.
    func checkUsage(_ d: AddOrder, grid: Bool) throws {
        if !grid, d.x != nil || d.y != nil || !d.sheet {
            throw Refusal("usage", String(localized: "--x, --y and --no-sheet belong to “silicontrol grid”"), help: "silicontrol help grid")
        }
        if grid, d.x == nil {
            throw Refusal("usage", String(localized: "a grid needs --x (and optionally --y)"), help: "silicontrol help grid")
        }
        if grid, d.batchGiven {
            throw Refusal("usage", String(localized: "a grid’s cell is one image: --batch does not apply"),
                          help: String(localized: "--x seeds=N gives N seeds — silicontrol help grid"))
        }
    }

    /// `--solver` other than Euler: the rack's, so More Solvers on, and a model whose family walks the others
    /// (`AppState.solverFamilies`). `name`: the model's, as the refusal says it.
    func checkSolver(_ solver: Solver, family: Family?, name: String) throws {
        guard solver != .euler else { return }
        guard app.developerMode else {
            throw Refusal("solver", String(localized: "--solver \(solver.rawValue) is offered only when More Solvers is on"),
                          help: String(localized: "turn on More Solvers in Settings (⌘,)"))
        }
        guard family.map(AppState.solverFamilies.contains) == true else {
            throw Refusal("solver", String(localized: "\(name) doesn’t offer the \(solver.rawValue) solver"), help: solverHint)
        }
    }

    /// A solver axis is the rack's Solver swept, so as `--solver` — but with a model axis, a cell of a model walking
    /// Euler alone renders Euler, as in the app.
    func checkSolverAxis(_ g: RenderGrid, family: Family?, name: String) throws {
        guard g.cells.contains(where: { ($0.solver ?? .euler) != .euler }) else { return }
        guard app.developerMode else {
            throw Refusal("solver", String(localized: "a solver axis is offered only when More Solvers is on"),
                          help: String(localized: "turn on More Solvers in Settings (⌘,)"))
        }
        guard g.cells.contains(where: { $0.model != nil }) || family.map(AppState.solverFamilies.contains) == true else {
            throw Refusal("solver", String(localized: "\(name) walks Euler only"), help: solverHint)
        }
    }

    /// The models a solver other than Euler is offered on, as `--model` takes them: read from the families' traits.
    private var solverHint: String {
        let ids = Family.allCases.filter(AppState.solverFamilies.contains).map { ModelCard.of($0).id }.joined(separator: "|")
        return String(localized: "Euler only, or --model \(ids)")
    }

    /// Detail: not on a model that offers none — the rack hides it there.
    func checkDetail(_ detail: Detail, accepted: Bool, name: String) throws {
        guard detail != .normal, !accepted else { return }
        throw Refusal("detail", String(localized: "\(name) doesn’t offer Detail"), help: String(localized: "leave out --detail"))
    }

    /// The stack as the rack allows it: `AppState.maxLoRA` at most, a negative strength only with its Beta option.
    func checkLoRAs(_ d: AddOrder) throws {
        guard d.loras.count <= AppState.maxLoRA else {
            let n = d.loras.count, max = AppState.maxLoRA
            throw Refusal("lora", String(localized: "\(n) LoRAs: the app stacks \(max) at most"))
        }
        guard app.negativeLoRAStrengths || !d.loras.contains(where: { $0.strength < 0 }) else {
            throw Refusal("lora", String(localized: "negative LoRA strengths are off"),
                          help: String(localized: "turn on Settings › Advanced › Negative LoRA Strengths (⌘,)"))
        }
    }

    /// A step count with no schedule among `scheduled`, refused with the counts there are.
    func checkSteps(_ n: Int, scheduled: [Int], defaultSteps: Int?) throws {
        guard !scheduled.contains(n) else { return }
        let accepted = scheduled.map(String.init).formatted(.list(type: .or))
        throw Refusal("steps", String(localized: "This model has no schedule for \(String(n)) steps: \(accepted)."),
                      help: defaultSteps.map { "--steps \($0) (the model's \"default_steps\")" })
    }
}
