import XCTest
@testable import Siliconed

/// The VAEs' plans: pure walks of the VAEs' own code, without a device or a weight.
final class VAEMemoryTests: XCTestCase {
    /// The arena: offsets from the lifetimes, the largest first — a slot dead before another is born
    /// shares its bytes, and the heap is the largest occupancy at one moment.
    func testTheLayoutIsPlacedFromTheLifetimes() {
        var layout = VAELayout()
        let page = VAELayout.alignment
        let a = layout.allocate(3 * page), b = layout.allocate(1)
        layout.free(a)
        let c = layout.allocate(2 * page)
        layout.free(b); layout.free(c)
        layout.place()
        XCTAssertEqual(layout.peak, 4 * page)                   // a + b, then b + c
        XCTAssertEqual(layout.registers, layout.peak)
        XCTAssertEqual(layout.offsets[a], 0)
        XCTAssertEqual(layout.offsets[c], 0)                     // a's bytes, a being dead
        XCTAssertEqual(layout.offsets[b], 3 * page)              // beside both
        layout.graph(largest: 1000)
        XCTAssertEqual(layout.workspace, page)
        XCTAssertTrue(layout.empty)
    }

    /// **The heap is the occupancy**: every VAE's placement reaches the largest sum of slots alive at
    /// once, at three formats — nothing lost to holes (the first fit at run time left ~60 MB).
    func testEveryPlanIsAsTightAsItsPeak() {
        let qwen = QwenImage21VAE.Config()
        for (h, w) in [(512, 512), (1024, 1024), (1536, 1024)] {
            let plans: [(String, VAELayout)] = [
                ("Flux decoder", VAE.plan(latentHeight: h / 8, latentWidth: w / 8)),
                ("Flux encoder", VAEEncoder.plan(family: .flux, height: h, width: w)),
                ("Qwen decoder", QwenImage21VAEDecoder.plan(config: qwen, latentHeight: h / 16, latentWidth: w / 16)),
                ("Qwen encoder", QwenImage21VAEEncoder.plan(config: qwen, height: h, width: w)),
                ("Anima decoder", AnimaVAE.plan(latentHeight: h / 8, latentWidth: w / 8)),
            ]
            for (name, plan) in plans {
                XCTAssertTrue(plan.empty, "\(name) \(w)×\(h)")
                XCTAssertEqual(plan.registers, plan.peak, "\(name) \(w)×\(h)")
            }
        }
    }

    /// The one graph MPSGraph keeps allocating beside our heaps: its SDPA, four `S·C` copies.
    func testTheAttentionIsTheOnlyOutside() {
        XCTAssertEqual(QwenImage21VAEDecoder.plan(config: QwenImage21VAE.Config(), latentHeight: 64, latentWidth: 64).outside,
                       VAELayout.sdpaOutside(tokens: 4096, channels: 1152))
        XCTAssertEqual(VAE.plan(latentHeight: 128, latentWidth: 128).outside, VAELayout.sdpaOutside(tokens: 16384, channels: 512))
    }

    /// MPS asks its workspace in whole 2 MiB: the SDPA's `S·C` queries at Qwen-Image-2.1's decoder at
    /// 512² (4.72 MB) ask 6 MiB, and the plan's workspace holds it (it held 4.5 MiB, the ask went beside).
    func testTheSDPAAskIsWholeMebibytePairs() {
        XCTAssertEqual(4 * VAELayout.sdpaAsk(tokens: 1024, channels: 1152), 6 << 20)
        XCTAssertEqual(4 * VAELayout.sdpaAsk(tokens: 4096, channels: 512), 8 << 20)
        XCTAssertEqual(4 * VAELayout.sdpaAsk(tokens: 4096, channels: 1152), 18 << 20)
        XCTAssertGreaterThanOrEqual(QwenImage21VAEDecoder.plan(config: QwenImage21VAE.Config(), latentHeight: 32,
                                                                latentWidth: 32).workspace, 6 << 20)
    }

    /// Every plan ends with nothing live, and grows with the image.
    func testThePlansGrowWithTheImage() {
        let qwen = QwenImage21VAE.Config()
        let plans: [(String, (Int, Int) -> VAELayout)] = [
            ("Flux decoder", { VAE.plan(latentHeight: $0 / 8, latentWidth: $1 / 8) }),
            ("Flux encoder", { VAEEncoder.plan(family: .flux, height: $0, width: $1) }),
            ("Qwen decoder", { QwenImage21VAEDecoder.plan(config: qwen, latentHeight: $0 / 16, latentWidth: $1 / 16) }),
            ("Qwen encoder", { QwenImage21VAEEncoder.plan(config: qwen, height: $0, width: $1) }),
            ("Anima decoder", { AnimaVAE.plan(latentHeight: $0 / 8, latentWidth: $1 / 8) }),
        ]
        for (name, plan) in plans {
            let sizes = [plan(512, 512), plan(1024, 1024), plan(1536, 1024)].map(\.bytes)
            XCTAssertEqual(sizes, sizes.sorted(), name)
            XCTAssertLessThan(sizes[0], sizes[1], name)
        }
    }

    /// The Flux decoder at 1024²: its registers hold at least the moment `bandedMoments` counts at its
    /// largest (the doubled 256 channels and `h₁` at full resolution), and the workspace is one band.
    func testTheFluxDecoderPlanHoldsItsLargestMoment() {
        let plan = VAE.plan(latentHeight: 128, latentWidth: 128)
        let moments = VAE.bandedMoments(latentHeight: 128, latentWidth: 128, latentChannels: 16)
        // `bandedMoments` keeps the latent to the end and counts the returned copy; the plan frees the one
        // and charges the other as host bytes.
        XCTAssertGreaterThanOrEqual(plan.registers + plan.host, moments.map(\.bytes).max()! - 16 * 128 * 128 * 4 - 3 * 1024 * 1024 * 4)
        XCTAssertLessThanOrEqual(plan.workspace, VAELayout.rounded(4 * (VAE.bandElements + VAE.bandElements / 4)))
    }

    /// What a Wan resnet reads: the shortcut (if any) first, then norm1, conv1, norm2, conv2.
    func testTheWeightLists() {
        XCTAssertEqual(WeightList.resnetQwen(96, 96), [96, 96 * 96 * 9, 96, 96, 96 * 96 * 9, 96])
        XCTAssertEqual(WeightList.resnetQwen(96, 192).prefix(2), [192 * 96, 192])
    }
}
