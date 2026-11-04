import XCTest
@testable import Siliconed

/// **The solvers against diffusers' schedulers**, on Z-Image's, Qwen-Image-2.1's, Anima's and Krea 2's schedules and a synthetic model
/// `m = −0.75·x + b_k`. The final samples below were printed by diffusers 0.41.0 (`oracle_solvers.py`)
/// (torch fp32) on that model: `FlowMatchEulerDiscreteScheduler` (and its `stochastic_sampling`),
/// `DPMSolverMultistepScheduler` and `UniPCMultistepScheduler` in `flow_prediction`.
final class SolverTests: XCTestCase {

    /// `(solver, steps, start, the sample after the last step)`.
    private let oracle: [(String, Int, Int, [Float])] = [
        ("euler", 6, 0, [-1.1033508777618408, 1.7132935523986816, -1.9699612855911255, 1.333426594734192, -1.970532774925232, 0.22864893078804016, -2.4484200477600098, -0.02536393702030182]),
        ("euler", 8, 0, [-1.1106786727905273, 1.6444131135940552, -1.6489256620407104, 0.8760226368904114, -1.6430608034133911, 0.23633979260921478, -2.5730843544006348, 0.05320030078291893]),
        ("euler", 9, 0, [-0.9208986759185791, 1.56882643699646, -1.37153160572052, 0.9550409317016602, -1.942929744720459, 0.4314452111721039, -2.6684513092041016, 0.2718547582626343]),
        ("euler", 8, 3, [-0.6616952419281006, 1.4369295835494995, -1.6500006914138794, 0.9568566083908081, -1.7024173736572266, 0.5493326187133789, -2.275459051132202, 0.1985270380973816]),
        ("euler-ancestral", 6, 0, [0.9222689867019653, 0.6808022260665894, -2.111231803894043, 0.11638568341732025, 1.2595845460891724, 0.006280690431594849, -0.7522937655448914, -0.23731088638305664]),
        ("euler-ancestral", 8, 0, [-0.15103070437908173, 1.3915375471115112, -0.4829846918582916, -0.7958823442459106, 0.09408301115036011, 0.18237313628196716, 1.070834994316101, -1.0317296981811523]),
        ("euler-ancestral", 9, 0, [-0.6805782914161682, 0.7346886992454529, 0.5830985307693481, -0.09964106231927872, -0.8737106919288635, -0.05708365887403488, 0.9349198937416077, -0.05678915977478027]),
        ("euler-ancestral", 8, 3, [-0.1439712643623352, -0.3592674732208252, -0.7892913818359375, 0.9594094157218933, 1.3405905961990356, -1.1495860815048218, -1.1880152225494385, 0.7585704922676086]),
        ("dpmpp-2m", 6, 0, [-0.9123567342758179, 1.6385133266448975, -2.179819107055664, 1.3789942264556885, -2.1721620559692383, 0.43884313106536865, -2.594515323638916, 0.1888832449913025]),
        ("dpmpp-2m", 8, 0, [-1.1924399137496948, 1.8086758852005005, -1.643510103225708, 0.7150586843490601, -1.6277917623519897, 0.17756974697113037, -2.4942166805267334, 0.0006021708250045776]),
        ("dpmpp-2m", 9, 0, [-0.8817951679229736, 1.4450653791427612, -1.277989387512207, 0.8626824617385864, -1.839207649230957, 0.4943021535873413, -2.880439281463623, 0.34153133630752563]),
        ("dpmpp-2m", 8, 3, [-0.6871606111526489, 1.3288705348968506, -1.465782880783081, 0.8607569932937622, -1.5104215145111084, 0.542015790939331, -2.4509267807006836, 0.4008936285972595]),
        ("unipc", 6, 0, [-0.955305814743042, 1.652481198310852, -2.206585645675659, 1.4174067974090576, -2.1964921951293945, 0.40157854557037354, -2.6016600131988525, 0.26472049951553345]),
        ("unipc", 8, 0, [-1.2816945314407349, 1.8757015466690063, -1.5837924480438232, 0.6629835963249207, -1.5663479566574097, 0.09234323352575302, -2.4421517848968506, 0.0032588765025138855]),
        ("unipc", 9, 0, [-0.8977227210998535, 1.4219862222671509, -1.219353437423706, 0.829765260219574, -1.7791211605072021, 0.4817572236061096, -2.916081666946411, 0.4080297350883484]),
        ("unipc", 8, 3, [-0.657537579536438, 1.2877357006072998, -1.440201997756958, 0.8403950333595276, -1.48361074924469, 0.5745089650154114, -2.5027225017547607, 0.42934462428092957]),
        // Qwen-Image-2.1 Turbo's published schedule (`steps` 0).
        ("euler", 0, 0, [-1.2069569826126099, 1.617618441581726, -1.4886418581008911, 0.6948248147964478, -1.4939619302749634, 0.11396308243274689, -2.502941131591797, 0.019483402371406555]),
        ("euler-ancestral", 0, 0, [-0.7163186073303223, 1.3111557960510254, 0.5710384249687195, -0.9948921203613281, -0.3675674498081207, 0.04091916233301163, 1.7205355167388916, -1.1122753620147705]),
        ("dpmpp-2m", 0, 0, [-1.5139200687408447, 1.938751459121704, -1.3454623222351074, 0.4451591372489929, -1.3419554233551025, -0.17240361869335175, -2.2583086490631104, -0.2596471309661865]),
        ("unipc", 0, 0, [-1.49125075340271, 1.9566712379455566, -1.3006551265716553, 0.35510727763175964, -1.2963987588882446, -0.14798621833324432, -2.2468819618225098, -0.1913665384054184]),
    ]

    private let n = 8
    /// Qwen-Image-2.1 Turbo's `sample_sigmas`, then 0.
    private let qwen: [Float] = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0]
    private func xInit() -> [Float] { (0..<n).map { Float((($0 * 5 + 3) % 11) - 5) / 4 } }
    private func bias(_ k: Int) -> [Float] { (0..<n).map { Float((($0 * 7 + k * 13 + 1) % 17) - 8) / 8 } }
    private func eps(_ k: Int) -> [Float] { (0..<n).map { Float((($0 * 3 + k * 11 + 2) % 13) - 6) / 4 } }

    /// The walk as the sampler drives it: the model sees the sample, then the solver steps.
    private func walk(_ solver: Solver, steps: Int, start: Int, schedule: [Float]? = nil) -> [Float] {
        let walk = SolverWalk(solver: solver, schedule: schedule ?? (steps == 0 ? qwen : FlowMatchSchedule(steps: steps).sigmas), start: start,
                              count: n, noise: { k, e in for (i, v) in self.eps(k).enumerated() { e[i] = v } })
        var x = xInit()
        for k in 0..<(walk.sigmas.count - 1) {
            let b = bias(k)
            let m = (0..<n).map { -0.75 * x[$0] + b[$0] }
            x.withUnsafeMutableBufferPointer { xs in
                m.withUnsafeBufferPointer { walk.step(latent: xs.baseAddress!, derivative: $0.baseAddress!) }
            }
        }
        return x
    }

    func testTheSolversAreDiffusers() {
        for (name, steps, start, want) in oracle {
            let solver = Solver(rawValue: name)!
            let got = walk(solver, steps: steps, start: start)
            for (i, (g, w)) in zip(got, want).enumerated() {
                XCTAssertEqual(g, w, accuracy: 2e-6 * max(1, abs(w)), "\(name), \(steps) steps from \(start), value \(i)")
            }
        }
    }

    /// **Anima Turbo's schedule**: its pipeline hands `np.linspace(1, 1/N, N)` (float64) to the
    /// published scheduler, where Z-Image's hands a torch float32 ramp — and the engine computes Anima's
    /// in double (`AnimaDiT.sigmas`). At 8 steps diffusers' four schedulers land on the same samples as
    /// on Z-Image's σ (`oracle_solvers.py --anima` prints the rows above for 8 steps from 0 and from 3,
    /// to the last digit): the walk on Anima's own σ must too.
    func testTheSolversAreDiffusersOnAnimasSchedule() {
        for (name, steps, start, want) in oracle where steps == 8 {
            let got = walk(Solver(rawValue: name)!, steps: steps, start: start, schedule: AnimaDiT.sigmas(steps: steps))
            for (i, (g, w)) in zip(got, want).enumerated() {
                XCTAssertEqual(g, w, accuracy: 2e-6 * max(1, abs(w)), "Anima, \(name), from \(start), value \(i)")
            }
        }
    }

    /// **Krea 2 Turbo's schedule**: `linspace(1, 1/N, N)` through Krea's published scheduler, the
    /// exponential shift at the distilled `mu = 1.15` — another curve than the static shift 3, so other
    /// samples (`oracle_solvers.py --krea2`, diffusers 0.41.0). The engine's σ are computed in double
    /// (`Krea2DiT.sigmas`); the walk on them must land where diffusers' schedulers do.
    private let krea2Oracle: [(String, Int, Int, [Float])] = [
        ("euler", 8, 0, [-1.111696720123291, 1.6342649459838867, -1.6367683410644531, 0.8652126789093018, -1.6315077543258667, 0.23391211032867432, -2.5779964923858643, 0.0591563954949379]),
        ("euler", 8, 3, [-0.6572859883308411, 1.4371005296707153, -1.6597223281860352, 0.9587817192077637, -1.710408329963684, 0.5577795505523682, -2.290286064147949, 0.21600812673568726]),
        ("euler-ancestral", 8, 0, [-0.18864445388317108, 1.3944568634033203, -0.4186874032020569, -0.8052156567573547, 0.05443991720676422, 0.18304318189620972, 1.1168577671051025, -1.035529613494873]),
        ("euler-ancestral", 8, 3, [-0.07792258262634277, -0.3744056224822998, -0.8030372858047485, 0.9239850640296936, 1.3949041366577148, -1.1337108612060547, -1.2146718502044678, 0.7629859447479248]),
        ("dpmpp-2m", 8, 0, [-1.193876028060913, 1.8004076480865479, -1.6281157732009888, 0.6997604370117188, -1.6132315397262573, 0.17418678104877472, -2.4952540397644043, 0.005625583231449127]),
        ("dpmpp-2m", 8, 3, [-0.6785050630569458, 1.3229246139526367, -1.4744012355804443, 0.8631650805473328, -1.5174074172973633, 0.5544805526733398, -2.471022129058838, 0.4192492961883545]),
        ("unipc", 8, 0, [-1.27956223487854, 1.8659559488296509, -1.568372368812561, 0.6459725499153137, -1.551832675933838, 0.09236440062522888, -2.444056987762451, 0.007956631481647491]),
        ("unipc", 8, 3, [-0.6475751399993896, 1.2799315452575684, -1.4491088390350342, 0.843713641166687, -1.4909093379974365, 0.5882243514060974, -2.5244669914245605, 0.44735559821128845]),
    ]

    func testTheSolversAreDiffusersOnKrea2sSchedule() {
        for (name, steps, start, want) in krea2Oracle {
            let got = walk(Solver(rawValue: name)!, steps: steps, start: start, schedule: Krea2DiT.sigmas(steps: steps))
            for (i, (g, w)) in zip(got, want).enumerated() {
                XCTAssertEqual(g, w, accuracy: 2e-6 * max(1, abs(w)), "Krea 2, \(name), from \(start), value \(i)")
            }
        }
    }

    /// The walk is the schedule's tail: every Z-Image step moves (no final `0 → 0` since 0.8.2), and
    /// img2img starts inside the schedule.
    func testTheWalkIsTheScheduleTail() {
        let σ = FlowMatchSchedule(steps: 8).sigmas
        XCTAssertEqual(σ.count, 9)
        XCTAssertEqual(σ.last, 0)
        XCTAssertNotEqual(σ[7], 0)
        XCTAssertEqual(SolverWalk(solver: .euler, schedule: σ, start: 0, count: n).sigmas, σ)
        XCTAssertEqual(SolverWalk(solver: .euler, schedule: σ, start: 3, count: n).sigmas, Array(σ[3...]))
    }

    /// Euler's walk is the sampler's own update, `x + Δσ·m`, to the bit.
    func testEulerIsTheSampler() {
        let σ = FlowMatchSchedule(steps: 8).sigmas
        var x = xInit(), y = xInit()
        let walk = SolverWalk(solver: .euler, schedule: σ, start: 0, count: n)
        for k in 0..<8 {
            let m = bias(k)
            x.withUnsafeMutableBufferPointer { xs in
                m.withUnsafeBufferPointer { walk.step(latent: xs.baseAddress!, derivative: $0.baseAddress!) }
            }
            let delta = σ[k + 1] - σ[k]
            for i in 0..<n { y[i] = y[i] + delta * m[i] }   // two roundings, as torch
        }
        XCTAssertEqual(x, y)
    }

    /// The estimate a preview shows: `x − σ·m` on the sample the DiT saw, for every rule
    /// but Euler (whose `x' − σ'·m` is it already).
    func testTheEstimateIsTheLastLanding() {
        let σ = FlowMatchSchedule(steps: 8).sigmas
        for solver in Solver.allCases {
            let walk = SolverWalk(solver: solver, schedule: σ, start: 0, count: n, noise: { k, e in
                for (i, v) in self.eps(k).enumerated() { e[i] = v } })
            XCTAssertEqual(walk.estimate == nil, solver == .euler)
            var x = xInit()
            for k in 0..<3 {
                let seen = x, m = bias(k)
                x.withUnsafeMutableBufferPointer { xs in
                    m.withUnsafeBufferPointer { walk.step(latent: xs.baseAddress!, derivative: $0.baseAddress!) }
                }
                guard let estimate = walk.estimate else { continue }
                let s = solver == .unipc && k == 0 ? σ[0] - 1e-6 : σ[k]
                for i in 0..<n { XCTAssertEqual(estimate[i], seen[i] - s * m[i], accuracy: 1e-6, "\(solver), step \(k)") }
            }
        }
    }

    // MARK: - To the bit

    /// `(family, solver, start, the sample after the last step as bit patterns)`, printed by
    /// `oracle_solvers.py --bits` (diffusers 0.41.0, torch fp32 on the CPU) on a model whose values are
    /// thirds, sevenths and elevenths: every product rounds, so a fused `a·b + c` shows.
    private let bitsOracle: [(String, String, Int, [UInt32])] = [
        ("z-image", "euler", 0, [0xbfbb24b5, 0x400ac6fc, 0xc00b1b7a, 0x3f98d109, 0xc010d858, 0x3ebba05f, 0xc05b0953, 0x3d7909c2, 0x40416eeb, 0xbede8f90, 0x402bc221, 0xbfcc05da, 0x4006bd94, 0xbffc99d4, 0x3fb268e9, 0xc02a1e41, 0x3f76918e, 0xc05f12ba, 0x3e1469c2, 0x404e3ada, 0xbf545f6d, 0x40487dfe, 0xbfd418a9, 0x3ffc9a03, 0xbffb3870, 0x3f7fba2b, 0xc01d5252, 0x3f290700, 0xc067834b, 0x3da7a69e, 0x405b096c, 0xbf212fb2]),
        ("z-image", "euler", 3, [0xbf6d3e3e, 0x3ff2f096, 0xc0090df6, 0x3fa1ccab, 0xc012583e, 0x3f368b46, 0xc041b6ff, 0x3e685558, 0x4039d8b6, 0xbf259774, 0x40169532, 0xbfb1893f, 0x400d4aea, 0xc0015696, 0x3f7e3211, 0xc00aa0dd, 0x3e816e0a, 0xc02de461, 0x3ddc6f2e, 0x40287ee6, 0xbf06b9f2, 0x401f349d, 0xbf89e402, 0x3fdfabb5, 0xbfc81ef9, 0x3f8e87c8, 0xc01bfaae, 0x3f100181, 0xc04b5971, 0x3ed5b0c2, 0x40303644, 0xbf4c2137]),
        ("z-image", "euler-ancestral", 0, [0xbe860532, 0x3f2fefa8, 0xbead76ba, 0xbedeeb3a, 0x3e9b1fa4, 0xbe9169a7, 0x3ed9aaa8, 0xbe96aca6, 0xbf3523d0, 0x3ed57fdd, 0x3ec109e2, 0xbd86e940, 0xbf0e025d, 0x3ee3cc98, 0x3eca1b6a, 0xbdd4d653, 0x3e0e2495, 0xbeca6ce5, 0x3efd5a3f, 0x3e0aa304, 0xbf20b827, 0x3de81548, 0xbe900cac, 0x3e7120a8, 0x3e570d36, 0xbf63a8d4, 0x3e65f599, 0x3f1de62a, 0xbe843f5f, 0xbea02ad4, 0x3ea0cfa0, 0x3e4f2cb1]),
        ("z-image", "euler-ancestral", 3, [0xbcfd1628, 0x3e25c6b3, 0xbf0e88ac, 0x3ec67504, 0x3f176a9d, 0xbec94418, 0xbf2a0aa2, 0x3f019dcc, 0x3e947322, 0x3e863848, 0xbefc5d72, 0xbea8cfa7, 0x3f8b3429, 0xbf44048c, 0xbdcce9b0, 0xbea1f24c, 0x3b9d7dc0, 0x3e86ef78, 0xbe547cb2, 0xbe4ca789, 0x3c953538, 0x3f72c965, 0xbd86b5e2, 0xbe9d57b2, 0x3e0c33cd, 0x3eb498f1, 0xbddba5a8, 0xbedb202f, 0xbf32f8ac, 0x3f37d778, 0x3e82970c, 0xbef05c20]),
        ("z-image", "dpmpp-2m", 0, [0xbfc79599, 0x401726fb, 0xc00b1253, 0x3f81c012, 0xc010356f, 0x3e9a32d3, 0xc055de53, 0x3a346700, 0x4037f12f, 0xbec9224a, 0x402950e6, 0xbfbd9a58, 0x40053462, 0xbfe71ad2, 0x3fbbd9ba, 0xc03a373e, 0x3f8dc9c6, 0xc067d0eb, 0x3e9906e8, 0x4054fe01, 0xbf864c2f, 0x404fdae4, 0xbfe17f8d, 0x400a3204, 0xc003fada, 0x3f4fac39, 0xc01d2a6b, 0x3f152b0e, 0xc062d34c, 0x3c972194, 0x406c82ef, 0xbf186511]),
        ("z-image", "dpmpp-2m", 3, [0xbf7573bd, 0x3fe3b383, 0xbff7bac0, 0x3f941ee7, 0xc004ae4a, 0x3f34c858, 0xc04f05ed, 0x3eea947d, 0x40349e7a, 0xbed0fa27, 0x4017c11b, 0xbfc8edd2, 0x400ef032, 0xc00c4137, 0x3fa0df84, 0xc0151220, 0x3dc3039e, 0xc031ef7e, 0xbd2e334e, 0x403afec8, 0xbf2a0c6c, 0x40322ddd, 0xbf8ec0f1, 0x3fcfac72, 0xbfae7ffd, 0x3f8017d5, 0xc00eb1d4, 0x3f0cba34, 0xc0590977, 0x3ed2ed18, 0x402a9af0, 0xbf108b37]),
        ("z-image", "unipc", 0, [0xbfd4b9a1, 0x401c1e2e, 0xc006c44d, 0x3f746ee5, 0xc00bcc7d, 0x3e50fe8c, 0xc0522a5c, 0x3b741660, 0x40348872, 0xbea2fe0e, 0x40220816, 0xbfb57b42, 0x3ffee8eb, 0xbfe4c4ea, 0x3fc619f3, 0xc03ec844, 0x3f96a008, 0xc06ed416, 0x3ebd375f, 0x405a79b4, 0xbf8eb710, 0x40557186, 0xbfeeceb2, 0x400f13a7, 0xc0043a3a, 0x3f4044cd, 0xc018d703, 0x3eec86be, 0xc05f34e3, 0xbda05964, 0x406edb8e, 0xbf05a920]),
        ("z-image", "unipc", 3, [0xbf6ce7bc, 0x3fddc613, 0xbff413d2, 0x3f9133ba, 0xc002c7a1, 0x3f3e5962, 0xc052e2fa, 0x3efb3a59, 0x40336a25, 0xbebfbac0, 0x401a66a0, 0xbfd02562, 0x4011a8e8, 0xc00e5bdf, 0x3fa5b1e1, 0xc0171999, 0x3d2f65dc, 0xc0301d1a, 0xbdc00424, 0x403da93a, 0xbf31253d, 0x4034eb80, 0xbf8a99a7, 0x3fc9a04e, 0xbfaaf7c0, 0x3f7a1be4, 0xc00cda84, 0x3f160dd5, 0xc05cf5db, 0x3ee62de3, 0x4029573f, 0xbf0828ed]),
        ("qwen", "euler", 0, [0xbfc8ae9e, 0x4008688c, 0xbffdf30b, 0x3f7b876c, 0xc00564ea, 0x3e65db06, 0xc0553964, 0x3cb668c8, 0x403196da, 0xbe957f5a, 0x4023f467, 0xbfc508ce, 0x3ff54e93, 0xbfe1ec23, 0x3fc00a27, 0xc032767d, 0x3fa14046, 0xc062faa7, 0x3e1c5339, 0x4052ba12, 0xbf7f05f8, 0x404c4ead, 0xbfe08b51, 0x3ff8f464, 0xbffa6d0e, 0x3f4bce03, 0xc0115344, 0x3f154434, 0xc06127c0, 0xbd7f03b0, 0x4060bd85, 0xbef4f227]),
        ("qwen", "euler", 3, [0xbf5df922, 0x3fecc62f, 0xc008ec62, 0x3f96f8bb, 0xc010f164, 0x3f57180b, 0xc0540b6e, 0x3f131dd1, 0x4038c0a8, 0xbf0eb468, 0x40230a4e, 0xbfcd8e46, 0x401b054a, 0xc011addd, 0x3f932e62, 0xc019b2e1, 0x3dcfa4ee, 0xc02f693a, 0xbcc3edc4, 0x4036db7b, 0xbf31ba55, 0x402ed67a, 0xbf8449e1, 0x3fd778df, 0xbfa646fd, 0x3f81ab6b, 0xc01b980d, 0x3f2c7d69, 0xc05eb215, 0x3f0c695d, 0x402e1a00, 0xbf394f09]),
        ("qwen", "euler-ancestral", 0, [0xbf079430, 0x3f2cf7c9, 0x3e4dc714, 0xbf2b3956, 0x3e810bda, 0xbec523e3, 0x3f31966b, 0xbed2f0a5, 0xbf91b472, 0x3ed7c850, 0x3eb05416, 0x3e9a39dd, 0xbf619338, 0x3eb0386e, 0x3f10e667, 0xbd43a66a, 0x3f01253e, 0xbf3558a7, 0x3efc81bc, 0x3f158764, 0xbf59f042, 0x3d8e8af8, 0xbf09fa6d, 0x3f02e1de, 0x3ec3ae9e, 0xbfa90608, 0x3e74b4f4, 0x3ea245a3, 0x3df3300c, 0xbed7bb4f, 0x3e29687a, 0x3ec45ef9]),
        ("qwen", "euler-ancestral", 3, [0x3f1223de, 0xbe85d906, 0xbee32e12, 0x3cfbbf80, 0x3f8605a0, 0xbe5870e0, 0xbf4e0938, 0x3f41ed98, 0x3d8fa5fa, 0x3f2d0787, 0xbf28c23f, 0xbf1c42c2, 0x3f7c8fba, 0xbecb0b4a, 0xbdd9c470, 0xbebd4d65, 0xbed4f89d, 0x3f6e2fd2, 0xbeef557a, 0xbf0dde96, 0xbe3380f7, 0x3f6fc3b3, 0x3f0f2c19, 0xbf6a50c4, 0x3f0d2f5e, 0x3c9cc6e4, 0x3ef0929d, 0xbe644ff7, 0xbf5100fc, 0x3f47d182, 0x3d6fcfa1, 0xbe9fed8f]),
        ("qwen", "dpmpp-2m", 0, [0xbff60452, 0x40203845, 0xbfe9a661, 0x3f335841, 0xbff569a9, 0xbdcf1b6b, 0xc043de80, 0xbe97ee40, 0x40215f5b, 0xbdc5715a, 0x4016c7b8, 0xbf9eaa66, 0x3fd05d8f, 0xbfb8a490, 0x3fdcbb81, 0xc0489913, 0x3fcb4744, 0xc07be7fe, 0x3efa937c, 0x4062e70c, 0xbfa81f8e, 0x405d0566, 0xc0075eb2, 0x4013dbba, 0xc013e39e, 0x3f01e615, 0xc0071161, 0x3eaf0f09, 0xc0503b0a, 0xbec5b874, 0x407b67f1, 0xbe9440ad]),
        ("qwen", "dpmpp-2m", 3, [0xbf3496ab, 0x3fcdbeb2, 0xbfe84277, 0x3f5a8c9d, 0xbff74c52, 0x3f83a43e, 0xc0699037, 0x3f6d1866, 0x40259af1, 0xbe4ae5e8, 0x4030c9ea, 0xbff530dc, 0x402944fe, 0xc02ad4a0, 0x3fc492d7, 0xc032598d, 0xbe3a0a3a, 0xc0272a94, 0xbe992c82, 0x40514137, 0xbf86c388, 0x4049bc4a, 0xbf60cb2e, 0x3fb7a473, 0xbf7afb46, 0x3f2e581a, 0xc006b34a, 0x3f5b13fc, 0xc0749d56, 0x3f3d0048, 0x401a8dd0, 0xbebddbf8]),
        ("qwen", "unipc", 0, [0xbff2bcb3, 0x40218ed3, 0xbfe32662, 0x3f191233, 0xbfeed259, 0xbd95aa23, 0xc0431436, 0xbe5ff416, 0x401af595, 0xbd2e0880, 0x400ec352, 0xbf9cc65a, 0x3fd45b86, 0xbfb99a22, 0x3fe3f1d3, 0xc04f836c, 0x3fcd7ad2, 0xc07a7548, 0x3f01df96, 0x4066a9f4, 0xbfb5a4c2, 0x4060d3fa, 0xc005c440, 0x401528f0, 0xc00f1630, 0x3ecef545, 0xc003cf10, 0x3e5ac63d, 0xc04f7a1a, 0xbeb74966, 0x407b7014, 0xbe71e054]),
        ("qwen", "unipc", 3, [0xbf2ccf20, 0x3fc1d065, 0xbfe84b40, 0x3f6ab5b6, 0xbff7419c, 0x3f87ca4e, 0xc06ffc5c, 0x3f6d8b6a, 0x4029c65c, 0xbe487dd4, 0x4032fe17, 0xc000e36d, 0x402b82ea, 0xc0271e32, 0x3fc52224, 0xc02e995e, 0xbe8e53e8, 0xc02561a4, 0xbeca2d51, 0x4051aa01, 0xbf7e01bc, 0x404a2ed6, 0xbf5922d3, 0x3faba68a, 0xbf7b2c06, 0x3f3e6202, 0xc006b5bb, 0x3f6340e6, 0xc07b1148, 0x3f455432, 0x401eb170, 0xbebce654]),
        ("anima", "euler", 0, [0xbfbb24b5, 0x400ac6fc, 0xc00b1b7a, 0x3f98d109, 0xc010d858, 0x3ebba05f, 0xc05b0953, 0x3d7909c2, 0x40416eeb, 0xbede8f90, 0x402bc221, 0xbfcc05da, 0x4006bd94, 0xbffc99d4, 0x3fb268e9, 0xc02a1e41, 0x3f76918e, 0xc05f12ba, 0x3e1469c2, 0x404e3ada, 0xbf545f6d, 0x40487dfe, 0xbfd418a9, 0x3ffc9a03, 0xbffb3870, 0x3f7fba2b, 0xc01d5252, 0x3f290700, 0xc067834b, 0x3da7a69e, 0x405b096c, 0xbf212fb2]),
        ("anima", "euler", 3, [0xbf6d3e3e, 0x3ff2f096, 0xc0090df6, 0x3fa1ccab, 0xc012583e, 0x3f368b46, 0xc041b6ff, 0x3e685558, 0x4039d8b6, 0xbf259774, 0x40169532, 0xbfb1893f, 0x400d4aea, 0xc0015696, 0x3f7e3211, 0xc00aa0dd, 0x3e816e0a, 0xc02de461, 0x3ddc6f2e, 0x40287ee6, 0xbf06b9f2, 0x401f349d, 0xbf89e402, 0x3fdfabb5, 0xbfc81ef9, 0x3f8e87c8, 0xc01bfaae, 0x3f100181, 0xc04b5971, 0x3ed5b0c2, 0x40303644, 0xbf4c2137]),
        ("anima", "euler-ancestral", 0, [0xbe860532, 0x3f2fefa8, 0xbead76ba, 0xbedeeb3a, 0x3e9b1fa4, 0xbe9169a7, 0x3ed9aaa8, 0xbe96aca6, 0xbf3523d0, 0x3ed57fdd, 0x3ec109e2, 0xbd86e940, 0xbf0e025d, 0x3ee3cc98, 0x3eca1b6a, 0xbdd4d653, 0x3e0e2495, 0xbeca6ce5, 0x3efd5a3f, 0x3e0aa304, 0xbf20b827, 0x3de81548, 0xbe900cac, 0x3e7120a8, 0x3e570d36, 0xbf63a8d4, 0x3e65f599, 0x3f1de62a, 0xbe843f5f, 0xbea02ad4, 0x3ea0cfa0, 0x3e4f2cb1]),
        ("anima", "euler-ancestral", 3, [0xbcfd1628, 0x3e25c6b3, 0xbf0e88ac, 0x3ec67504, 0x3f176a9d, 0xbec94418, 0xbf2a0aa2, 0x3f019dcc, 0x3e947322, 0x3e863848, 0xbefc5d72, 0xbea8cfa7, 0x3f8b3429, 0xbf44048c, 0xbdcce9b0, 0xbea1f24c, 0x3b9d7dc0, 0x3e86ef78, 0xbe547cb2, 0xbe4ca789, 0x3c953538, 0x3f72c965, 0xbd86b5e2, 0xbe9d57b2, 0x3e0c33cd, 0x3eb498f1, 0xbddba5a8, 0xbedb202f, 0xbf32f8ac, 0x3f37d778, 0x3e82970c, 0xbef05c20]),
        ("anima", "dpmpp-2m", 0, [0xbfc79599, 0x401726fb, 0xc00b1253, 0x3f81c012, 0xc010356f, 0x3e9a32d3, 0xc055de53, 0x3a346700, 0x4037f12f, 0xbec9224a, 0x402950e6, 0xbfbd9a58, 0x40053462, 0xbfe71ad2, 0x3fbbd9ba, 0xc03a373e, 0x3f8dc9c6, 0xc067d0eb, 0x3e9906e8, 0x4054fe01, 0xbf864c2f, 0x404fdae4, 0xbfe17f8d, 0x400a3204, 0xc003fada, 0x3f4fac39, 0xc01d2a6b, 0x3f152b0e, 0xc062d34c, 0x3c972194, 0x406c82ef, 0xbf186511]),
        ("anima", "dpmpp-2m", 3, [0xbf7573bd, 0x3fe3b383, 0xbff7bac0, 0x3f941ee7, 0xc004ae4a, 0x3f34c858, 0xc04f05ed, 0x3eea947d, 0x40349e7a, 0xbed0fa27, 0x4017c11b, 0xbfc8edd2, 0x400ef032, 0xc00c4137, 0x3fa0df84, 0xc0151220, 0x3dc3039e, 0xc031ef7e, 0xbd2e334e, 0x403afec8, 0xbf2a0c6c, 0x40322ddd, 0xbf8ec0f1, 0x3fcfac72, 0xbfae7ffd, 0x3f8017d5, 0xc00eb1d4, 0x3f0cba34, 0xc0590977, 0x3ed2ed18, 0x402a9af0, 0xbf108b37]),
        ("anima", "unipc", 0, [0xbfd4b9a1, 0x401c1e2e, 0xc006c44d, 0x3f746ee5, 0xc00bcc7d, 0x3e50fe8c, 0xc0522a5c, 0x3b741660, 0x40348872, 0xbea2fe0e, 0x40220816, 0xbfb57b42, 0x3ffee8eb, 0xbfe4c4ea, 0x3fc619f3, 0xc03ec844, 0x3f96a008, 0xc06ed416, 0x3ebd375f, 0x405a79b4, 0xbf8eb710, 0x40557186, 0xbfeeceb2, 0x400f13a7, 0xc0043a3a, 0x3f4044cd, 0xc018d703, 0x3eec86be, 0xc05f34e3, 0xbda05964, 0x406edb8e, 0xbf05a920]),
        ("anima", "unipc", 3, [0xbf6ce7bc, 0x3fddc613, 0xbff413d2, 0x3f9133ba, 0xc002c7a1, 0x3f3e5962, 0xc052e2fa, 0x3efb3a59, 0x40336a25, 0xbebfbac0, 0x401a66a0, 0xbfd02562, 0x4011a8e8, 0xc00e5bdf, 0x3fa5b1e1, 0xc0171999, 0x3d2f65dc, 0xc0301d1a, 0xbdc00424, 0x403da93a, 0xbf31253d, 0x4034eb80, 0xbf8a99a7, 0x3fc9a04e, 0xbfaaf7c0, 0x3f7a1be4, 0xc00cda84, 0x3f160dd5, 0xc05cf5db, 0x3ee62de3, 0x4029573f, 0xbf0828ed]),
        ("krea2", "euler", 0, [0xbfbb434a, 0x400a0350, 0xc00a322f, 0x3f9734aa, 0xc00ff87a, 0x3eba259f, 0xc05b5bdf, 0x3d8a75b6, 0x404080ae, 0xbed89107, 0x402b724c, 0xbfcceb10, 0x400664c3, 0xbffc533e, 0x3fb3a871, 0xc02aa4cb, 0x3f7b0def, 0xc05efa6d, 0x3e0aa876, 0x404eba91, 0xbf56f9c8, 0x4048f447, 0xbfd42829, 0x3ffb21c2, 0xbffa0a33, 0x3f7c9f9c, 0xc01c6aea, 0x3f286609, 0xc067ce4e, 0x3da17f54, 0x405ac320, 0xbf1e123f]),
        ("krea2", "euler", 3, [0xbf6c1f36, 0x3ff31765, 0xc009d431, 0x3fa22a5b, 0xc0130378, 0x3f39198b, 0xc042e7a2, 0x3e7cca91, 0x403a635e, 0xbf26d522, 0x40179492, 0xbfb332e4, 0x400e654c, 0xc0020ff7, 0x3f7e639f, 0xc00b3f3e, 0x3e7b4be1, 0xc02e0e07, 0x3dd0aeda, 0x4028e717, 0xbf07c438, 0x401fb7d0, 0xbf897fb2, 0x3fdfa750, 0xbfc67326, 0x3f8eba44, 0xc01cbb84, 0x3f12395d, 0xc04c9fad, 0x3edaf881, 0x4030ab52, 0xbf4db54f]),
        ("krea2", "euler-ancestral", 0, [0xbe8f5822, 0x3f2ff48c, 0xbe9c955a, 0xbee55f0e, 0x3e977993, 0xbe93442b, 0x3ee24720, 0xbe9a76f0, 0xbf3cc37b, 0x3ed54922, 0x3ec2c54a, 0xbd3e0650, 0xbf14c8fa, 0x3ee10a75, 0x3ed15d1b, 0xbdcb7758, 0x3e2c4f29, 0xbed6df23, 0x3efdbd61, 0x3e29bb46, 0xbf23c564, 0x3dd9c17a, 0xbe990f94, 0x3e813dec, 0x3e60fd23, 0xbf6b89e0, 0x3e663acc, 0x3f1ac6a8, 0xbe73d906, 0xbea3c070, 0x3e9aa5fc, 0x3e5e62c0]),
        ("krea2", "euler-ancestral", 3, [0x3c222cc0, 0x3e085dbf, 0xbf0c0684, 0x3eb8aeca, 0x3f2102a7, 0xbec3a7b1, 0xbf2a39da, 0x3f077838, 0x3e882330, 0x3e93f13e, 0xbf039e66, 0xbeb00b8f, 0x3f89f252, 0xbf3d7f2c, 0xbdd31f6c, 0xbea116bf, 0xbcba3678, 0x3ea23039, 0xbe625d0c, 0xbe70b9ed, 0x3bd45930, 0x3f70daaf, 0xbcbe3d88, 0xbeb4c3a2, 0x3e332614, 0x3ea7b98a, 0xbd85b0a6, 0xbed49cf4, 0xbf32b47a, 0x3f3935f1, 0x3e6e5bdf, 0xbeea2543]),
        ("krea2", "dpmpp-2m", 0, [0xbfc7c0f4, 0x4016845a, 0xc009ea41, 0x3f7ef180, 0xc00f1a66, 0x3e98233a, 0xc055e4ba, 0x3bd2ac80, 0x4036a141, 0xbec1ac0d, 0x402864d2, 0xbfbdffaf, 0x4004b927, 0xbfe65a24, 0x3fbd5ec0, 0xc03af7f6, 0x3f906f6c, 0xc067afe9, 0x3e95a746, 0x40559443, 0xbf882625, 0x4050641f, 0xbfe19612, 0x400999cb, 0xc003661c, 0x3f4b4741, 0xc01c04f2, 0x3f125583, 0xc062cf4b, 0x3c69b60c, 0x406c5172, 0xbf148048]),
        ("krea2", "dpmpp-2m", 3, [0xbf73143a, 0x3fe2f36a, 0xbff91c14, 0x3f948d78, 0xc005457c, 0x3f388254, 0xc05097ab, 0x3ef5521b, 0x40352c5e, 0xbed2fec0, 0x40190638, 0xbfcb6414, 0x40104ec8, 0xc00ce503, 0x3fa10b95, 0xc0159c72, 0x3da67332, 0xc031c29b, 0xbd60f5d8, 0x403b6b6b, 0xbf2adb46, 0x4032b3fc, 0xbf8db9f1, 0x3fcec396, 0xbfaca693, 0x3f805da5, 0xc00f5d65, 0x3f1022ab, 0xc05aaf94, 0x3eda89cc, 0x402b1475, 0xbf11df0b]),
        ("krea2", "unipc", 0, [0xbfd45e80, 0x401b5f3a, 0xc0059b19, 0x3f6f5e3c, 0xc00ab165, 0x3e50d7b5, 0xc0523fe9, 0x3c14fc28, 0x40331448, 0xbe9ba547, 0x40212383, 0xbfb6065e, 0x3ffe67db, 0xbfe40138, 0x3fc79025, 0xc03fa122, 0x3f991870, 0xc06e6b3a, 0x3eb91a12, 0x405b04cc, 0xbf90c8d1, 0x4055ee82, 0xbfee5cf8, 0x400e5ffd, 0xc003a6fd, 0x3f3b614c, 0xc017b0a0, 0x3ee73c80, 0xc05f3f26, 0xbda1015a, 0x406e9f4c, 0xbf01cf8f]),
        ("krea2", "unipc", 3, [0xbf6a25bf, 0x3fdcbff2, 0xbff57f7a, 0x3f91c416, 0xc003645d, 0x3f4270ba, 0xc054933a, 0x3f02e22c, 0x403407a8, 0xbec1f5f8, 0x401bc1cd, 0xbfd2db3a, 0x40131d2c, 0xc00eeb8b, 0x3fa5cdc0, 0xc0179029, 0x3cdc00f0, 0xc02fd605, 0xbddd93b6, 0x403e0c7c, 0xbf31aa30, 0x403567df, 0xbf8960d0, 0x3fc87201, 0xbfa92816, 0x3f7aec4b, 0xc00d8b56, 0x3f19d4da, 0xc05eba32, 0x3eee84b9, 0x4029e0b0, 0xbf0996dc]),
    ]

    private let roughCount = 32
    private func roughX() -> [Float] { (0..<roughCount).map { Float((($0 * 5 + 3) % 11) - 5) / 3 } }
    private func roughBias(_ k: Int) -> [Float] { (0..<roughCount).map { Float((($0 * 7 + k * 13 + 1) % 17) - 8) / 7 } }
    private func roughEps(_ k: Int) -> [Float] { (0..<roughCount).map { Float((($0 * 3 + k * 11 + 2) % 13) - 6) / 11 } }

    /// **Every rule computes what torch computes, rounding for rounding**: `x + Δσ·m` is two roundings
    /// on the CPU (torch never fuses), so the walk must not fuse either — `vDSP_vsma` did, 18/10.
    func testTheSolversAreDiffusersToTheBit() {
        let schedules: [String: [Float]] = ["z-image": FlowMatchSchedule(steps: 8).sigmas, "qwen": qwen,
                                            "anima": AnimaDiT.sigmas(steps: 8), "krea2": Krea2DiT.sigmas(steps: 8)]
        for (family, name, start, want) in bitsOracle {
            let walk = SolverWalk(solver: Solver(rawValue: name)!, schedule: schedules[family]!, start: start,
                                  count: roughCount, noise: { k, e in for (i, v) in self.roughEps(k).enumerated() { e[i] = v } })
            var x = roughX()
            for k in 0..<(walk.sigmas.count - 1) {
                let b = roughBias(k)
                let m = (0..<roughCount).map { -0.75 * x[$0] + b[$0] }
                x.withUnsafeMutableBufferPointer { xs in
                    m.withUnsafeBufferPointer { walk.step(latent: xs.baseAddress!, derivative: $0.baseAddress!) }
                }
            }
            let apart = zip(x, want).filter { $0.bitPattern != $1 }.count
            let worst = zip(x, want).map { abs(Int64($0.bitPattern) - Int64($1)) }.max() ?? 0
            // **The one residue**: torch's `exp` is SLEEF's `expf4_u10`, not libm's — on Krea 2's σ₂ of the
            // DPM++ walk, `exp(−h)` is one bit apart (0x3f0e38e4 against 0x3f0e38e3), and one final value
            // with it. A coefficient, not the per-element arithmetic: bounded here.
            let allowed = family == "krea2" && name == "dpmpp-2m" && start == 0 ? 1 : 0
            XCTAssertLessThanOrEqual(apart, allowed, "\(family), \(name) from \(start): \(apart) of \(roughCount) values apart")
            XCTAssertLessThanOrEqual(worst, Int64(allowed), "\(family), \(name) from \(start): \(worst) ulp")
        }
    }

    /// Anima's and Krea 2's σ as `set_timesteps` leaves them (`oracle_solvers.py --bits`), N = 1…50:
    /// float32 operation by operation, not the double formula rounded once.
    private let scheduleOracle: [(String, Int, [UInt32])] = [
        ("anima", 1, [0x3f800000, 0x00000000]),
        ("anima", 2, [0x3f800000, 0x3f400000, 0x00000000]),
        ("anima", 4, [0x3f800000, 0x3f666666, 0x3f400000, 0x3f000000, 0x00000000]),
        ("anima", 6, [0x3f800000, 0x3f700001, 0x3f5b6db6, 0x3f400000, 0x3f199999, 0x3ec00000, 0x00000000]),
        ("anima", 8, [0x3f800000, 0x3f745d17, 0x3f666666, 0x3f555555, 0x3f400000, 0x3f249249, 0x3f000000, 0x3e99999a, 0x00000000]),
        ("anima", 9, [0x3f800000, 0x3f75c290, 0x3f69bd39, 0x3f5b6db6, 0x3f4a1af3, 0x3f34b4b5, 0x3f199999, 0x3eec4ec6, 0x3e8ba2e9, 0x00000000]),
        ("anima", 12, [0x3f800000, 0x3f787878, 0x3f700001, 0x3f666666, 0x3f5b6db6, 0x3f4ec4ed, 0x3f400000, 0x3f2e8ba3, 0x3f199999, 0x3f000000, 0x3ec00000, 0x3e5b6db7, 0x00000000]),
        ("anima", 20, [0x3f800000, 0x3f7b9611, 0x3f76db6d, 0x3f71c71d, 0x3f6c4ec6, 0x3f666666, 0x3f5fffff, 0x3f590b21, 0x3f51745d, 0x3f492494, 0x3f400000, 0x3f35e50d, 0x3f2aaaab, 0x3f1e1e1d, 0x3f100000, 0x3f000000, 0x3edb6db8, 0x3eb13b15, 0x3e800000, 0x3e0ba2e9, 0x00000000]),
        ("anima", 30, [0x3f800000, 0x3f7d1745, 0x3f7a0be7, 0x3f76db6d, 0x3f73831f, 0x3f700001, 0x3f6c4ec6, 0x3f686bca, 0x3f645307, 0x3f5fffff, 0x3f5b6db6, 0x3f569697, 0x3f51745d, 0x3f4c0001, 0x3f46318e, 0x3f400000, 0x3f39611a, 0x3f324924, 0x3f2aaaab, 0x3f227628, 0x3f199999, 0x3f100000, 0x3f0590b3, 0x3ef45d17, 0x3edb6db8, 0x3ec00000, 0x3ea1af2a, 0x3e800000, 0x3e34b4b6, 0x3dc00000, 0x00000000]),
        ("anima", 50, [0x3f800000, 0x3f7e4531, 0x3f7c7e3e, 0x3f7aaaaa, 0x3f78c9e9, 0x3f76db6d, 0x3f74de9b, 0x3f72d2d2, 0x3f70b768, 0x3f6e8ba4, 0x3f6c4ec6, 0x3f6a0000, 0x3f679e7a, 0x3f65294a, 0x3f629f7a, 0x3f5fffff, 0x3f5d49c2, 0x3f5a7b95, 0x3f579436, 0x3f549249, 0x3f51745d, 0x3f4e38e5, 0x3f4ade32, 0x3f476278, 0x3f43c3c4, 0x3f400000, 0x3f3c14e5, 0x3f37ffff, 0x3f33bea3, 0x3f2f4dea, 0x3f2aaaab, 0x3f25d174, 0x3f20be83, 0x3f1b6db6, 0x3f15da89, 0x3f100000, 0x3f09d89e, 0x3f035e51, 0x3ef914c1, 0x3eeaaaa9, 0x3edb6db8, 0x3ecb4b4c, 0x3eba2e8c, 0x3ea80001, 0x3e94a529, 0x3e800000, 0x3e53dcb1, 0x3e249249, 0x3de38e38, 0x3d6c4ec5, 0x00000000]),
        ("krea2", 1, [0x3f800000, 0x00000000]),
        ("krea2", 2, [0x3f800000, 0x3f426f4f, 0x00000000]),
        ("krea2", 4, [0x3f800000, 0x3f678f54, 0x3f426f4f, 0x3f0349c1, 0x00000000]),
        ("krea2", 6, [0x3f800000, 0x3f70c0f3, 0x3f5d0292, 0x3f426f4f, 0x3f1cbd8d, 0x3ec6347c, 0x00000000]),
        ("krea2", 8, [0x3f800000, 0x3f74ebd8, 0x3f678f54, 0x3f572119, 0x3f426f4f, 0x3f2791b0, 0x3f0349c1, 0x3e9f2e6d, 0x00000000]),
        ("krea2", 9, [0x3f800000, 0x3f7640de, 0x3f6ac2fc, 0x3f5d0292, 0x3f4c425c, 0x3f37686c, 0x3f1cbd8d, 0x3ef2dba0, 0x3e90ea5c, 0x00000000]),
        ("krea2", 12, [0x3f800000, 0x3f78d651, 0x3f70c0f3, 0x3f678f54, 0x3f5d0292, 0x3f50c7c4, 0x3f426f4f, 0x3f315f51, 0x3f1cbd8d, 0x3f0349c1, 0x3ec6347c, 0x3e646b12, 0x00000000]),
        ("krea2", 20, [0x3f800000, 0x3f7bcdbd, 0x3f774cae, 0x3f7273d5, 0x3f6d38c3, 0x3f678f54, 0x3f61694e, 0x3f5ab5e8, 0x3f53612f, 0x3f4b5344, 0x3f426f4f, 0x3f38922a, 0x3f2d908a, 0x3f21347d, 0x3f1339f2, 0x3f0349c1, 0x3ee1e4ef, 0x3eb73b56, 0x3e84ff0e, 0x3e11f337, 0x00000000]),
        ("krea2", 30, [0x3f800000, 0x3f7d3c2c, 0x3f7a5690, 0x3f774cae, 0x3f741bd1, 0x3f70c0f3, 0x3f6d38c3, 0x3f697f91, 0x3f659145, 0x3f61694e, 0x3f5d0292, 0x3f585759, 0x3f53612f, 0x3f4e18d0, 0x3f4875fd, 0x3f426f4f, 0x3f3bfa0b, 0x3f3509de, 0x3f2d908a, 0x3f257d87, 0x3f1cbd8d, 0x3f1339f2, 0x3f08d7eb, 0x3efaef15, 0x3ee1e4ef, 0x3ec6347c, 0x3ea76c0e, 0x3e84ff0e, 0x3e3c7b15, 0x3dc9214c, 0x00000000]),
        ("krea2", 50, [0x3f800000, 0x3f7e5b3a, 0x3f7caaa1, 0x3f7aedb4, 0x3f7923e8, 0x3f774cae, 0x3f75676d, 0x3f737381, 0x3f71703c, 0x3f6f5ce9, 0x3f6d38c3, 0x3f6b02fa, 0x3f68bab1, 0x3f665efc, 0x3f63eee0, 0x3f61694e, 0x3f5ecd26, 0x3f5c1934, 0x3f594c2b, 0x3f5664a9, 0x3f53612f, 0x3f504021, 0x3f4cffc4, 0x3f499e39, 0x3f461977, 0x3f426f4f, 0x3f3e9d5e, 0x3f3aa113, 0x3f367797, 0x3f321ddd, 0x3f2d908a, 0x3f28cbf0, 0x3f23cc0b, 0x3f1e8c6b, 0x3f19082f, 0x3f1339f2, 0x3f0d1bbc, 0x3f06a6ea, 0x3effa832, 0x3ef13610, 0x3ee1e4ef, 0x3ed1a003, 0x3ec04fd3, 0x3eadd9cc, 0x3e9a1fb6, 0x3e84ff0e, 0x3e5ca073, 0x3e2bcb25, 0x3dee28ce, 0x3d780391, 0x00000000]),
    ]

    func testAnimaAndKrea2SchedulesAreDiffusersToTheBit() {
        for (family, steps, want) in scheduleOracle {
            let got = family == "anima" ? AnimaDiT.sigmas(steps: steps) : Krea2DiT.sigmas(steps: steps)
            XCTAssertEqual(got.map(\.bitPattern), want, "\(family), \(steps) steps")
        }
    }

    /// What Z-Image's DiT is told: `(1000 − σ·1000)/1000` in float32, as `ZImagePipeline` computes it
    /// (diffusers 0.41.0, printed from its `scheduler.timesteps`) — `1 − σ` was 8 ulps off at step 1.
    func testZImageModelTimeIsThePipelines() {
        let want: [UInt32] = [0x0, 0x3d3a2e98, 0x3dcccccd, 0x3e2aaaac, 0x3e800000, 0x3eb6db6f, 0x3f000000, 0x3f333333]
        let s = FlowMatchSchedule(steps: 8)
        XCTAssertEqual((0..<8).map { FlowMatchSchedule.modelTime(sigma: s.sigmas[$0]).bitPattern }, want)
        XCTAssertNotEqual(1 - s.sigmas[1], FlowMatchSchedule.modelTime(sigma: s.sigmas[1]))
    }

    /// **What the DiT is told under DPM++ and UniPC**: diffusers' int64 `timesteps`, printed by the
    /// schedulers themselves (`oracle_solvers.py --timesteps`, diffusers 0.41.0) on each family's σ,
    /// N = 1…50, from step 0 and step 3. `(family, steps, start, solver, timesteps)`; Qwen's `steps` 0
    /// is its published schedule.
    func testTheMultistepDiTTimeIsDiffusersIntegerTimestep() {
        let rows: [(String, Int, Int, String, [Int])] = [
        ("z-image", 1, 0, "dpmpp-2m", [1000]),
        ("z-image", 1, 0, "unipc", [999]),
        ("z-image", 2, 0, "dpmpp-2m", [1000, 750]),
        ("z-image", 2, 0, "unipc", [999, 750]),
        ("z-image", 4, 0, "dpmpp-2m", [1000, 900, 750, 500]),
        ("z-image", 4, 0, "unipc", [999, 900, 750, 500]),
        ("z-image", 4, 3, "dpmpp-2m", [500]),
        ("z-image", 4, 3, "unipc", [500]),
        ("z-image", 6, 0, "dpmpp-2m", [1000, 937, 857, 750, 599, 375]),
        ("z-image", 6, 0, "unipc", [999, 937, 857, 750, 599, 375]),
        ("z-image", 6, 3, "dpmpp-2m", [750, 599, 375]),
        ("z-image", 6, 3, "unipc", [750, 599, 375]),
        ("z-image", 8, 0, "dpmpp-2m", [1000, 954, 900, 833, 750, 642, 500, 300]),
        ("z-image", 8, 0, "unipc", [999, 954, 900, 833, 750, 642, 500, 300]),
        ("z-image", 8, 3, "dpmpp-2m", [833, 750, 642, 500, 300]),
        ("z-image", 8, 3, "unipc", [833, 750, 642, 500, 300]),
        ("z-image", 9, 0, "dpmpp-2m", [1000, 960, 913, 857, 789, 705, 599, 461, 272]),
        ("z-image", 9, 0, "unipc", [999, 960, 913, 857, 789, 705, 599, 461, 272]),
        ("z-image", 9, 3, "dpmpp-2m", [857, 789, 705, 599, 461, 272]),
        ("z-image", 9, 3, "unipc", [857, 789, 705, 599, 461, 272]),
        ("z-image", 12, 0, "dpmpp-2m", [1000, 970, 937, 900, 857, 807, 750, 681, 599, 500, 375, 214]),
        ("z-image", 12, 0, "unipc", [999, 970, 937, 900, 857, 807, 750, 681, 599, 500, 375, 214]),
        ("z-image", 12, 3, "dpmpp-2m", [900, 857, 807, 750, 681, 599, 500, 375, 214]),
        ("z-image", 12, 3, "unipc", [900, 857, 807, 750, 681, 599, 500, 375, 214]),
        ("z-image", 20, 0, "dpmpp-2m", [1000, 982, 964, 944, 923, 900, 874, 847, 818, 785, 750, 710, 666, 617, 562, 500, 428, 346, 250, 136]),
        ("z-image", 20, 0, "unipc", [999, 982, 964, 944, 923, 900, 874, 847, 818, 785, 750, 710, 666, 617, 562, 500, 428, 346, 250, 136]),
        ("z-image", 20, 3, "dpmpp-2m", [944, 923, 900, 874, 847, 818, 785, 750, 710, 666, 617, 562, 500, 428, 346, 250, 136]),
        ("z-image", 20, 3, "unipc", [944, 923, 900, 874, 847, 818, 785, 750, 710, 666, 617, 562, 500, 428, 346, 250, 136]),
        ("z-image", 30, 0, "dpmpp-2m", [1000, 988, 976, 964, 951, 937, 923, 907, 891, 875, 857, 838, 818, 796, 774, 749, 724, 696, 666, 634, 599, 562, 521, 477, 428, 375, 315, 249, 176, 93]),
        ("z-image", 30, 0, "unipc", [999, 988, 976, 964, 951, 937, 923, 907, 891, 875, 857, 838, 818, 796, 774, 749, 724, 696, 666, 634, 599, 562, 521, 477, 428, 375, 315, 249, 176, 93]),
        ("z-image", 30, 3, "dpmpp-2m", [964, 951, 937, 923, 907, 891, 875, 857, 838, 818, 796, 774, 749, 724, 696, 666, 634, 599, 562, 521, 477, 428, 375, 315, 249, 176, 93]),
        ("z-image", 30, 3, "unipc", [964, 951, 937, 923, 907, 891, 875, 857, 838, 818, 796, 774, 749, 724, 696, 666, 634, 599, 562, 521, 477, 428, 375, 315, 249, 176, 93]),
        ("z-image", 50, 0, "dpmpp-2m", [1000, 993, 986, 979, 971, 964, 956, 948, 940, 931, 923, 914, 904, 895, 885, 874, 864, 853, 842, 830, 818, 805, 792, 778, 764, 750, 734, 718, 702, 684, 666, 647, 627, 607, 585, 562, 538, 513, 486, 458, 428, 397, 363, 328, 290, 249, 206, 160, 111, 57]),
        ("z-image", 50, 0, "unipc", [999, 993, 986, 979, 971, 964, 956, 948, 940, 931, 923, 914, 904, 895, 885, 874, 864, 853, 842, 830, 818, 805, 792, 778, 764, 750, 734, 718, 702, 684, 666, 647, 627, 607, 585, 562, 538, 513, 486, 458, 428, 397, 363, 328, 290, 249, 206, 160, 111, 57]),
        ("z-image", 50, 3, "dpmpp-2m", [979, 971, 964, 956, 948, 940, 931, 923, 914, 904, 895, 885, 874, 864, 853, 842, 830, 818, 805, 792, 778, 764, 750, 734, 718, 702, 684, 666, 647, 627, 607, 585, 562, 538, 513, 486, 458, 428, 397, 363, 328, 290, 249, 206, 160, 111, 57]),
        ("z-image", 50, 3, "unipc", [979, 971, 964, 956, 948, 940, 931, 923, 914, 904, 895, 885, 874, 864, 853, 842, 830, 818, 805, 792, 778, 764, 750, 734, 718, 702, 684, 666, 647, 627, 607, 585, 562, 538, 513, 486, 458, 428, 397, 363, 328, 290, 249, 206, 160, 111, 57]),
        ("anima", 1, 0, "dpmpp-2m", [1000]),
        ("anima", 1, 0, "unipc", [999]),
        ("anima", 2, 0, "dpmpp-2m", [1000, 750]),
        ("anima", 2, 0, "unipc", [999, 750]),
        ("anima", 4, 0, "dpmpp-2m", [1000, 900, 750, 500]),
        ("anima", 4, 0, "unipc", [999, 900, 750, 500]),
        ("anima", 4, 3, "dpmpp-2m", [500]),
        ("anima", 4, 3, "unipc", [500]),
        ("anima", 6, 0, "dpmpp-2m", [1000, 937, 857, 750, 599, 375]),
        ("anima", 6, 0, "unipc", [999, 937, 857, 750, 599, 375]),
        ("anima", 6, 3, "dpmpp-2m", [750, 599, 375]),
        ("anima", 6, 3, "unipc", [750, 599, 375]),
        ("anima", 8, 0, "dpmpp-2m", [1000, 954, 900, 833, 750, 642, 500, 300]),
        ("anima", 8, 0, "unipc", [999, 954, 900, 833, 750, 642, 500, 300]),
        ("anima", 8, 3, "dpmpp-2m", [833, 750, 642, 500, 300]),
        ("anima", 8, 3, "unipc", [833, 750, 642, 500, 300]),
        ("anima", 9, 0, "dpmpp-2m", [1000, 960, 913, 857, 789, 705, 599, 461, 272]),
        ("anima", 9, 0, "unipc", [999, 960, 913, 857, 789, 705, 599, 461, 272]),
        ("anima", 9, 3, "dpmpp-2m", [857, 789, 705, 599, 461, 272]),
        ("anima", 9, 3, "unipc", [857, 789, 705, 599, 461, 272]),
        ("anima", 12, 0, "dpmpp-2m", [1000, 970, 937, 900, 857, 807, 750, 681, 599, 500, 375, 214]),
        ("anima", 12, 0, "unipc", [999, 970, 937, 900, 857, 807, 750, 681, 599, 500, 375, 214]),
        ("anima", 12, 3, "dpmpp-2m", [900, 857, 807, 750, 681, 599, 500, 375, 214]),
        ("anima", 12, 3, "unipc", [900, 857, 807, 750, 681, 599, 500, 375, 214]),
        ("anima", 20, 0, "dpmpp-2m", [1000, 982, 964, 944, 923, 900, 874, 847, 818, 785, 750, 710, 666, 617, 562, 500, 428, 346, 250, 136]),
        ("anima", 20, 0, "unipc", [999, 982, 964, 944, 923, 900, 874, 847, 818, 785, 750, 710, 666, 617, 562, 500, 428, 346, 250, 136]),
        ("anima", 20, 3, "dpmpp-2m", [944, 923, 900, 874, 847, 818, 785, 750, 710, 666, 617, 562, 500, 428, 346, 250, 136]),
        ("anima", 20, 3, "unipc", [944, 923, 900, 874, 847, 818, 785, 750, 710, 666, 617, 562, 500, 428, 346, 250, 136]),
        ("anima", 30, 0, "dpmpp-2m", [1000, 988, 976, 964, 951, 937, 923, 907, 891, 874, 857, 838, 818, 796, 774, 750, 724, 696, 666, 634, 599, 562, 521, 477, 428, 375, 315, 250, 176, 93]),
        ("anima", 30, 0, "unipc", [999, 988, 976, 964, 951, 937, 923, 907, 891, 874, 857, 838, 818, 796, 774, 750, 724, 696, 666, 634, 599, 562, 521, 477, 428, 375, 315, 250, 176, 93]),
        ("anima", 30, 3, "dpmpp-2m", [964, 951, 937, 923, 907, 891, 874, 857, 838, 818, 796, 774, 750, 724, 696, 666, 634, 599, 562, 521, 477, 428, 375, 315, 250, 176, 93]),
        ("anima", 30, 3, "unipc", [964, 951, 937, 923, 907, 891, 874, 857, 838, 818, 796, 774, 750, 724, 696, 666, 634, 599, 562, 521, 477, 428, 375, 315, 250, 176, 93]),
        ("anima", 50, 0, "dpmpp-2m", [1000, 993, 986, 979, 971, 964, 956, 948, 940, 931, 923, 914, 904, 895, 885, 874, 864, 853, 842, 830, 818, 805, 792, 778, 764, 750, 734, 718, 702, 684, 666, 647, 627, 607, 585, 562, 538, 513, 486, 458, 428, 397, 363, 328, 290, 250, 206, 160, 111, 57]),
        ("anima", 50, 0, "unipc", [999, 993, 986, 979, 971, 964, 956, 948, 940, 931, 923, 914, 904, 895, 885, 874, 864, 853, 842, 830, 818, 805, 792, 778, 764, 750, 734, 718, 702, 684, 666, 647, 627, 607, 585, 562, 538, 513, 486, 458, 428, 397, 363, 328, 290, 250, 206, 160, 111, 57]),
        ("anima", 50, 3, "dpmpp-2m", [979, 971, 964, 956, 948, 940, 931, 923, 914, 904, 895, 885, 874, 864, 853, 842, 830, 818, 805, 792, 778, 764, 750, 734, 718, 702, 684, 666, 647, 627, 607, 585, 562, 538, 513, 486, 458, 428, 397, 363, 328, 290, 250, 206, 160, 111, 57]),
        ("anima", 50, 3, "unipc", [979, 971, 964, 956, 948, 940, 931, 923, 914, 904, 895, 885, 874, 864, 853, 842, 830, 818, 805, 792, 778, 764, 750, 734, 718, 702, 684, 666, 647, 627, 607, 585, 562, 538, 513, 486, 458, 428, 397, 363, 328, 290, 250, 206, 160, 111, 57]),
        ("krea2", 1, 0, "dpmpp-2m", [1000]),
        ("krea2", 1, 0, "unipc", [999]),
        ("krea2", 2, 0, "dpmpp-2m", [1000, 759]),
        ("krea2", 2, 0, "unipc", [999, 759]),
        ("krea2", 4, 0, "dpmpp-2m", [1000, 904, 759, 512]),
        ("krea2", 4, 0, "unipc", [999, 904, 759, 512]),
        ("krea2", 4, 3, "dpmpp-2m", [512]),
        ("krea2", 4, 3, "unipc", [512]),
        ("krea2", 6, 0, "dpmpp-2m", [1000, 940, 863, 759, 612, 387]),
        ("krea2", 6, 0, "unipc", [999, 940, 863, 759, 612, 387]),
        ("krea2", 6, 3, "dpmpp-2m", [759, 612, 387]),
        ("krea2", 6, 3, "unipc", [759, 612, 387]),
        ("krea2", 8, 0, "dpmpp-2m", [1000, 956, 904, 840, 759, 654, 512, 310]),
        ("krea2", 8, 0, "unipc", [999, 956, 904, 840, 759, 654, 512, 310]),
        ("krea2", 8, 3, "dpmpp-2m", [840, 759, 654, 512, 310]),
        ("krea2", 8, 3, "unipc", [840, 759, 654, 512, 310]),
        ("krea2", 9, 0, "dpmpp-2m", [1000, 961, 917, 863, 797, 716, 612, 474, 283]),
        ("krea2", 9, 0, "unipc", [999, 961, 917, 863, 797, 716, 612, 474, 283]),
        ("krea2", 9, 3, "dpmpp-2m", [863, 797, 716, 612, 474, 283]),
        ("krea2", 9, 3, "unipc", [863, 797, 716, 612, 474, 283]),
        ("krea2", 12, 0, "dpmpp-2m", [1000, 972, 940, 904, 863, 815, 759, 692, 612, 512, 387, 223]),
        ("krea2", 12, 0, "unipc", [999, 972, 940, 904, 863, 815, 759, 692, 612, 512, 387, 223]),
        ("krea2", 12, 3, "dpmpp-2m", [904, 863, 815, 759, 692, 612, 512, 387, 223]),
        ("krea2", 12, 3, "unipc", [904, 863, 815, 759, 692, 612, 512, 387, 223]),
        ("krea2", 20, 0, "dpmpp-2m", [1000, 983, 966, 947, 926, 904, 880, 854, 825, 794, 759, 720, 677, 629, 575, 512, 441, 357, 259, 142]),
        ("krea2", 20, 0, "unipc", [999, 983, 966, 947, 926, 904, 880, 854, 825, 794, 759, 720, 677, 629, 575, 512, 441, 357, 259, 142]),
        ("krea2", 20, 3, "dpmpp-2m", [947, 926, 904, 880, 854, 825, 794, 759, 720, 677, 629, 575, 512, 441, 357, 259, 142]),
        ("krea2", 20, 3, "unipc", [947, 926, 904, 880, 854, 825, 794, 759, 720, 677, 629, 575, 512, 441, 357, 259, 142]),
        ("krea2", 30, 0, "dpmpp-2m", [1000, 989, 977, 966, 953, 940, 926, 912, 896, 880, 863, 845, 825, 805, 783, 759, 734, 707, 677, 646, 612, 575, 534, 490, 441, 387, 326, 259, 184, 98]),
        ("krea2", 30, 0, "unipc", [999, 989, 977, 966, 953, 940, 926, 912, 896, 880, 863, 845, 825, 805, 783, 759, 734, 707, 677, 646, 612, 575, 534, 490, 441, 387, 326, 259, 184, 98]),
        ("krea2", 30, 3, "dpmpp-2m", [966, 953, 940, 926, 912, 896, 880, 863, 845, 825, 805, 783, 759, 734, 707, 677, 646, 612, 575, 534, 490, 441, 387, 326, 259, 184, 98]),
        ("krea2", 30, 3, "unipc", [966, 953, 940, 926, 912, 896, 880, 863, 845, 825, 805, 783, 759, 734, 707, 677, 646, 612, 575, 534, 490, 441, 387, 326, 259, 184, 98]),
        ("krea2", 50, 0, "dpmpp-2m", [1000, 993, 986, 980, 973, 966, 958, 950, 943, 935, 926, 918, 909, 899, 890, 880, 870, 859, 848, 837, 825, 813, 800, 787, 773, 759, 744, 729, 712, 695, 677, 659, 639, 619, 597, 575, 551, 525, 499, 471, 441, 409, 375, 339, 301, 259, 215, 167, 116, 60]),
        ("krea2", 50, 0, "unipc", [999, 993, 986, 980, 973, 966, 958, 950, 943, 935, 926, 918, 909, 899, 890, 880, 870, 859, 848, 837, 825, 813, 800, 787, 773, 759, 744, 729, 712, 695, 677, 659, 639, 619, 597, 575, 551, 525, 499, 471, 441, 409, 375, 339, 301, 259, 215, 167, 116, 60]),
        ("krea2", 50, 3, "dpmpp-2m", [980, 973, 966, 958, 950, 943, 935, 926, 918, 909, 899, 890, 880, 870, 859, 848, 837, 825, 813, 800, 787, 773, 759, 744, 729, 712, 695, 677, 659, 639, 619, 597, 575, 551, 525, 499, 471, 441, 409, 375, 339, 301, 259, 215, 167, 116, 60]),
        ("krea2", 50, 3, "unipc", [980, 973, 966, 958, 950, 943, 935, 926, 918, 909, 899, 890, 880, 870, 859, 848, 837, 825, 813, 800, 787, 773, 759, 744, 729, 712, 695, 677, 659, 639, 619, 597, 575, 551, 525, 499, 471, 441, 409, 375, 339, 301, 259, 215, 167, 116, 60]),
        ("qwen", 0, 0, "dpmpp-2m", [1000, 978, 954, 926, 895, 845, 704, 414]),
        ("qwen", 0, 0, "unipc", [999, 978, 954, 926, 895, 845, 704, 414]),
        ("qwen", 0, 3, "dpmpp-2m", [926, 895, 845, 704, 414]),
        ("qwen", 0, 3, "unipc", [926, 895, 845, 704, 414]),
        ]
        for (family, steps, start, name, want) in rows {
            let full: [Float] = family == "qwen" ? qwen : family == "anima" ? AnimaDiT.sigmas(steps: steps)
                : family == "krea2" ? Krea2DiT.sigmas(steps: steps) : FlowMatchSchedule(steps: steps).sigmas
            let solver = Solver(rawValue: name)!
            let got = (start..<(full.count - 1)).map { solver.integerTimestep(full[$0], first: $0 == start)! }
            XCTAssertEqual(got, want, "\(family) \(steps) steps from \(start), \(name)")
        }
        XCTAssertNil(Solver.euler.integerTimestep(1, first: true))
        XCTAssertNil(Solver.eulerAncestral.integerTimestep(0.5, first: false))
    }

    /// Handed `k/1000`, a DiT that computes `(σ·1000)/1000` (Qwen-Image-2.1, Anima, Krea 2) finds
    /// diffusers' `fp32(k)/1000` for every `k` — although `k/1000 · 1000` is not `k` for ten of them.
    func testToldSigmaRoundTripsEveryIntegerTimestep() {
        var notK = 0
        for k in 0...1000 {
            let q = Float(k) / 1000
            XCTAssertEqual((q * 1000) / 1000, q, "k \(k)")
            if q * 1000 != Float(k) { notK += 1 }
        }
        XCTAssertEqual(notK, 10)
        XCTAssertEqual(Solver.unipc.toldSigma(1, first: true), 0.999)
        XCTAssertEqual(Solver.dpmpp2m.toldSigma(1, first: true), 1)
        XCTAssertEqual(Solver.euler.toldSigma(0.954545, first: true), 0.954545)
    }
}
