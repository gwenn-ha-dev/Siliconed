import CoreAI
import Foundation

/// **A block's declaration**: what its client writes (per-row tensors, bindings, units, steps, builds, forms,
/// groups, passes) and the ANE's share.
extension Block {
    /// A per-row tensor: `width` values of `type` a row, every row of the block. `next`: a stream, two buffers in turn, the
    /// outputs named `next` writing the other, swapped after a pass that wrote it. `factor`: the per-row tensor its true
    /// values are multiplied by (`values`). `rows`: fewer rows than the block's, for a pass over its first rows alone (its
    /// calls rounded up to 64 within them).
    public struct Rows: Sendable {
        public var name: String
        public var width: Int
        public var type: NDArray.ScalarType
        public var next: String?
        public var factor: String?
        /// Its own rows, not the block's (a cache's); `apart`: in buffers of its own (`Stage.Place.apart`), what a stage's
        /// last call writes.
        public var rows: Int?
        public var apart: Bool
        public init(_ name: String, width: Int, _ type: NDArray.ScalarType, next: String? = nil, factor: String? = nil, rows: Int? = nil,
                    apart: Bool = false) {
            self.name = name; self.width = width; self.type = type; self.next = next; self.factor = factor; self.rows = rows; self.apart = apart
        }
        var rowBytes: Int { width * Layout.element(type) }
    }

    /// What an argument is bound to: the call's rows of a per-row tensor, or of a stream's other buffer; a per-row tensor
    /// from row 0; a tensor whole; row `i` of a tensor, a row being the argument's bytes rounded to a page.
    public enum Bind: Sendable, Equatable {
        case rows(String), next(String), whole(String), tensor(String), row(String, Int)
    }

    /// Which unit computes a step: `either`, the ANE on its units and the GPU past them; `gpu`, the GPU on every row.
    public enum Unit: Sendable, Equatable {
        case either, gpu
    }

    /// A step: its function computes each row alone (`either` unit: the ANE on its units, the GPU past them), or is the
    /// GPU's on every row (it reads across rows, or what the ANE cannot run); every call of a step after the step before's.
    public struct Step: Sendable {
        public var name: String
        public var unit: Unit
        public var whole: Set<String>
        public var bind: [String: Bind]
        /// The step has a tapped form (`Build.tapped`): a function that writes more, for a check listening inside the block.
        public var taps: Bool
        public init(_ name: String, on unit: Unit = .either, whole: Set<String> = [], bind: [String: Bind] = [:], taps: Bool = false) {
            self.name = name; self.unit = unit; self.whole = whole; self.bind = bind; self.taps = taps
        }
    }

    /// What a function is built for: its rows (a call's, a unit's; an edge's 0), its unit.
    public struct Build: Sendable {
        public let rows: Int
        /// A tapped run's function of a step that taps (`Step.taps`): the module writes what the check listens to.
        public let tapped: Bool
        /// The ANE's function: a step's of either unit (`Unit.either`) at its unit's rows, or an edge the ANE computes
        /// (`Declaration.neuralEdges`); false, the GPU's. The two differ on the Compact's codes: the GPU's products int8 ×
        /// int8 on the codes by pieces, the ANE's q64 on its copy `[in, out]` (`SharedWeights`).
        public let neural: Bool
        public init(rows: Int, tapped: Bool = false, neural: Bool = false) {
            self.rows = rows; self.tapped = tapped; self.neural = neural
        }
    }

    public struct Form: Sendable {
        public var name: String
        public var steps: [Step]
        public var bind: [String: Bind]
        public var module: @Sendable (_ step: String, _ build: Build) throws -> Aicode.Module
        public init(_ name: String, steps: [Step], bind: [String: Bind] = [:], module: @escaping @Sendable (String, Build) throws -> Aicode.Module) {
            self.name = name; self.steps = steps; self.bind = bind; self.module = module
        }
        var shared: Bool { steps.contains { $0.unit == .either } }
    }

    /// `held`: weights of the group kept whatever B (residents named `<group>.<name>`), bound by their own names in the
    /// group's calls — a `run`'s, or an edge's outside one (`call(_:group:)`): what a loop reads once, before its cycles.
    public struct Group: Sendable {
        public var name: String
        public var form: String
        public var weights: [Weights.Entry]
        public var held: [Weights.Entry]
        public init(_ name: String, form: String, weights: [Weights.Entry], held: [Weights.Entry] = []) {
            self.name = name; self.form = form; self.weights = weights; self.held = held
        }
    }

    /// `forms`: the forms whose groups run over it (nil: every form), the only ones built at its calls' rows.
    public struct Pass: Sendable {
        public var name: String
        public var rows: Range<Int>
        public var shareable: Range<Int>?
        /// The rows past which no call of the pass reaches (the per-row tensors it touches hold that many): nil, the block's.
        public var limit: Int?
        public var bind: [String: Bind]
        public var forms: Set<String>?
        public init(_ name: String, rows: Range<Int>, shareable: Range<Int>? = nil, limit: Int? = nil, bind: [String: Bind] = [:],
                    forms: Set<String>? = nil) {
            precondition(shareable.map { $0.lowerBound == rows.lowerBound && $0.upperBound <= rows.upperBound } ?? true,
                         "\(name): the ANE's rows from the pass's first")
            precondition(limit.map { $0 >= rows.upperBound } ?? true, "\(name): a limit under the pass's rows")
            self.name = name; self.rows = rows; self.shareable = shareable; self.limit = limit; self.bind = bind; self.forms = forms
        }
        func serves(_ f: Form) -> Bool { forms?.contains(f.name) ?? true }
    }

    public struct Declaration: Sendable {
        public var name: String
        /// Every per-row tensor's rows: the calls' rows never pass them.
        public var rows: Int
        public var perRow: [Rows] = []
        /// The tensors no file holds and no row indexes; one an ANE's step reads becomes a surface at its signature's shape.
        public var tensors: [Stage.Tensor] = []
        public var residents: [Weights.Entry] = []
        public var groups: [Group] = []
        /// The loop's use of the groups: each of `once` in turn, then `cycle` over and over (`Stage`).
        public var once: [String] = []
        public var cycle: [String] = []
        /// The group whose layout the ring's slots take (`Stage.Declaration.slot`); nil, the layout most groups share (`slot`).
        public var slot: String? = nil
        /// Whether a pass may be cut into calls of fewer rows (B too tight for one): not where a function computes across
        /// every row of the pass it writes (a layer's own attention).
        public var cuts = true
        public var forms: [Form] = []
        public var passes: [Pass] = []
        public var edges: [String: @Sendable (Build) throws -> Aicode.Module] = [:]
        /// The edges the ANE computes (`call` runs them there, every GPU call before them done): their tensors surfaces
        /// as declared, the weights surfaces.
        public var neuralEdges: Set<String> = []
        /// Passes may be run tapped (`run`'s `tapped`): the functions of the steps that tap are built in both forms, and the
        /// calls over every row of a pass the GPU's alone, at their own heights.
        public var tapped = false
        public init(_ name: String, rows: Int) { self.name = name; self.rows = rows }

        /// The forms its groups run, by name: the only ones built.
        var used: [Form] {
            let byName = Dictionary(uniqueKeysWithValues: forms.map { ($0.name, $0) })
            return Set(groups.map(\.form)).sorted().map { byName[$0]! }
        }
        /// The rows the ANE may share: the tallest pass's shareable rows where a form used is shared, else 0.
        var shareable: Int { used.contains(where: \.shared) ? passes.compactMap(\.shareable).map(\.count).max() ?? 0 : 0 }
    }

    /// The ANE's share: none; a control's rows (0, the shared forms on the GPU alone); found in flight.
    public enum Share: Sendable, Equatable { case gpu, told(Int), inFlight }
}
