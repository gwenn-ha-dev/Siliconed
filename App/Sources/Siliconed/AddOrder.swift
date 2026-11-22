// The arguments of `silicontrol add` and `grid`, read: here, pure, so that the tests read a command line as the app does
// (`silicontrol measure` sends its renders as `add` lines, and a prompt it was given must arrive whole). What the order
// names — a model, a LoRA, an image — is the app's to resolve and judge (`RemoteControl.add`).

import Measurement

/// **`add [flags] "<prompt>" [size] [seed]`**, and `grid`'s, read: the flags first or anywhere, then the prompt, then a size
/// and a seed in any order, each once. A prompt that begins with `--` is written after a lone `--` (`Flags`), the size and
/// the seed after it too.
package struct AddOrder {
    package var model = ModelCard.of(.default).id
    package var prompt: String?
    package var format: (width: Int, height: Int)?
    package var seed: UInt64?
    package var batch = 1
    package var steps: Int?
    /// `--detail more|most`: the rack's Detail choice (`Request.detail`).
    package var detail = Detail.normal
    /// `--solver NAME`: the rack's Solver (`Request.solver`); whether it is offered is the app's to judge.
    package var solver = Solver.euler
    package var loras: [(name: String, strength: Double)] = []
    /// The `--ref`, in their order: the first is image 1, the one edited.
    package var references: [String] = []
    /// `--image IMAGE|i7`: the starting image of an img2img (a model with "start_image": true); `--strength`: its strength,
    /// diffusers' (the family's, `StartImage.defaultStrength`, when left out).
    package var image: String?
    package var strength: Double?
    /// `nil`: like the DiT box's « Preview every step » switch.
    package var previews: Bool?
    package var wait = false
    /// `grid --no-sheet`: a series as « Generate » renders it — the cells, the sheet only on `sheet g2`.
    package var sheet = true
    package var folder: String?
    /// `grid`: the axes as typed (`RenderGrid.axis`), and whether `--batch` was given (refused).
    package var x: String?, y: String?
    package var batchGiven = false

    package init() {}

    /// What the command line got wrong; the app says each in its words (`RemoteControl.refusal`).
    package enum Failure: Error, Equatable, Sendable {
        case unknownFlag(String)
        case flagWithoutValue(String)
        /// A flag's value it cannot take: the flag, and the value as typed.
        case badValue(Valued, String)
        case promptMissing
        /// A `P:Q@1K` size: the proportion, or the resolution, it does not know; or one it cannot read at all.
        case unknownRatio(String)
        case unknownResolution(String)
        case unreadableScale(String)
        /// A `WxH` size it cannot read.
        case unreadableFormat(String)
        /// A word after the prompt that is neither a size nor a seed, or a second one of either.
        case extraArgument(String)
    }

    /// The flags whose value is read here, not only carried: those a value can be refused for.
    package enum Valued: String, Sendable {
        case steps = "--steps", detail = "--detail", solver = "--solver", batch = "--batch", lora = "--lora", strength = "--strength"
    }

    /// The flags of `add` and `grid`, by kind: those that take a value, and the switches.
    package static let valued: Set<String> = ["--model", "--steps", "--detail", "--solver", "--batch", "--x", "--y", "--lora", "--ref",
                                              "--out", "--image", "--strength"]
    package static let switches: Set<String> = ["--preview", "--no-preview", "--wait", "--no-sheet"]

    /// `--batch`'s range: the rack's « Seeds » stepper's.
    package static let batches = 1...8

    package static func parse(_ arguments: [String]) -> Result<AddOrder, Failure> {
        do { return .success(try read(arguments)) } catch { return .failure(error) }
    }

    private static func read(_ arguments: [String]) throws(Failure) -> AddOrder {
        var d = AddOrder()
        let flags: Flags
        switch Flags.scan(arguments, valued: valued, switches: switches) {
        case .success(let f): flags = f
        case .failure(.unknownFlag(let a)): throw .unknownFlag(a)
        case .failure(.flagWithoutValue(let a)): throw .flagWithoutValue(a)
        }
        // A flag given twice: its last value, except those that add up (`--lora`, `--ref`).
        if let model = flags.value("--model") { d.model = model }
        if let v = flags.value("--steps") {
            guard let n = Int(v), (1...Request.maximumSteps).contains(n) else { throw .badValue(.steps, v) }
            d.steps = n
        }
        if let v = flags.value("--detail") {
            guard let level = Detail(rawValue: v.lowercased()) else { throw .badValue(.detail, v) }
            d.detail = level
        }
        if let v = flags.value("--solver") {
            guard let rule = Solver(rawValue: v.lowercased()) else { throw .badValue(.solver, v) }
            d.solver = rule
        }
        if let v = flags.value("--batch") {
            guard let n = Int(v), batches.contains(n) else { throw .badValue(.batch, v) }
            d.batch = n
            d.batchGiven = true
        }
        d.x = flags.value("--x")
        d.y = flags.value("--y")
        for v in flags.values["--lora"] ?? [] {
            // `name:strength` — the last colon separates the strength (a path has none).
            if let colonIndex = v.lastIndex(of: ":"), let f = Double(v[v.index(after: colonIndex)...].replacingOccurrences(of: ",", with: ".")) {
                // `Double` reads « nan » and « inf »: a strength that would make every weight NaN.
                guard f.isFinite else { throw .badValue(.lora, v) }
                d.loras.append((String(v[..<colonIndex]), f))
            } else {
                d.loras.append((v, 1))
            }
        }
        d.references = flags.values["--ref"] ?? []
        d.folder = flags.value("--out")
        if let last = flags.last(of: ["--preview", "--no-preview"]) { d.previews = last == "--preview" }
        if flags.has("--wait") { d.wait = true }
        if flags.has("--no-sheet") { d.sheet = false }
        d.image = flags.value("--image")
        if let v = flags.value("--strength") {
            guard let s = StartImage.strength(v) else { throw .badValue(.strength, v) }
            d.strength = s
        }
        // A prompt that begins with `--` is written after a lone `--`: everything there is a word.
        let positional = flags.allWords
        guard let prompt = positional.first else { throw .promptMissing }
        d.prompt = prompt
        for p in positional.dropFirst() {
            if d.format == nil, SizeScales.looksLikeScale(p) {
                // `3:4@2K`: the pixels of the rack's two-scale Size (`SizeScales`), whether or not that Beta option is on.
                // The model's own limits are the app's to judge, with `WxH`'s.
                do { let c = try SizeScales.parse(p); d.format = (c.width, c.height) } catch {
                    switch error {
                    case .unknownRatio(let r): throw .unknownRatio(r)
                    case .unknownResolution(let r): throw .unknownResolution(r)
                    case .unreadable: throw .unreadableScale(p)
                    }
                }
            } else if d.format == nil, p.lowercased().contains("x") || p.contains("×") {
                guard let f = Format.parse(p) else { throw .unreadableFormat(p) }
                d.format = (f.width, f.height)
            } else if d.format == nil, p == "512" || p == "1024" {
                d.format = (Int(p)!, Int(p)!)
            } else if d.seed == nil, let g = UInt64(p) {
                d.seed = g
            } else {
                throw .extraArgument(p)
            }
        }
        return d
    }
}
