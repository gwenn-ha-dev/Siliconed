import XCTest
@testable import Siliconed

/// The FLUX.2 VAE's pipeline part and its plan: pure, without a device or a weight.
final class Flux2VAETests: XCTestCase {
    /// `reshape(C, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)`: channel `4c + 2py + px` is pixel `(2y + py, 2x + px)`.
    func testUnpatchifyOrder() {
        // One channel, a 1×2 grid: the 4 patch channels at each cell hold distinct values.
        let (h, w) = (1, 2)
        var z = [Float](repeating: 0, count: 4 * h * w)
        for k in 0..<4 { for x in 0..<w { z[k * w + x] = Float(10 * x + k) } }
        let out = Flux2VAE.unpatchify(z, channels: 1, patch: 2, height: h, width: w)
        // Row 0 of the image: (py 0) px 0, px 1 of cell 0, then cell 1; row 1: py 1.
        XCTAssertEqual(out, [0, 1, 10, 11, 2, 3, 12, 13])
    }

    /// Two channels: each keeps its own four patch channels (`c` is the slow index).
    func testUnpatchifyKeepsChannelsApart() {
        let z = (0..<8).map(Float.init)                       // [8, 1, 1]
        let out = Flux2VAE.unpatchify(z, channels: 2, patch: 2, height: 1, width: 1)
        XCTAssertEqual(out, [0, 1, 2, 3, 4, 5, 6, 7])
        // Against the reference's permutation, written the long way on a 3×2 grid.
        let (h, w, C) = (3, 2, 3)
        let big = (0..<(C * 4 * h * w)).map(Float.init)
        let got = Flux2VAE.unpatchify(big, channels: C, patch: 2, height: h, width: w)
        for c in 0..<C { for y in 0..<h { for x in 0..<w { for py in 0..<2 { for px in 0..<2 {
            let source = ((((c * 2 + py) * 2 + px) * h) + y) * w + x
            XCTAssertEqual(got[(c * 2 * h + 2 * y + py) * 2 * w + 2 * x + px], big[source])
        } } } } }
    }

    /// `z · √(var + ε) + mean`, per channel of the patched latent, ε = 1e-4.
    func testDenormalize() {
        let scale = Flux2VAE.scale(variance: [4 - 1e-4, 1 - 1e-4], epsilon: 1e-4)
        XCTAssertEqual(scale[0], 2, accuracy: 1e-6)
        XCTAssertEqual(scale[1], 1, accuracy: 1e-6)
        let out = Flux2VAE.denormalize([1, 2, 3, 4], scale: scale, shift: [10, -1], plane: 2)
        XCTAssertEqual(out, [12, 14, 2, 3])
    }

    /// The decoder is 384/384/192/96 (the encoder's 128…512 is not it) and its plan stays under B at
    /// the three formats: B = ¾ · 16 GiB − 2 GiB.
    func testThePlanHoldsUnderBAtThreeFormats() {
        let budget = 3 * (16 << 30) / 4 - (2 << 30)
        var previous = 0
        for (h, w) in [(64, 64), (128, 128), (192, 128)] {
            let plan = Flux2VAE.plan(latentHeight: h, latentWidth: w)
            let need = plan.bytes + plan.outside
            XCTAssertGreaterThan(need, previous)
            XCTAssertLessThan(need, budget / 2, "\(h)x\(w)")
            XCTAssertGreaterThanOrEqual(plan.host, 4 * 3 * 64 * h * w)       // the returned image
            previous = need
        }
        // Narrower than Flux.1's decoder (512 channels) at the same grid.
        XCTAssertLessThan(Flux2VAE.plan(latentHeight: 128, latentWidth: 128).registers,
                          VAE.plan(latentHeight: 128, latentWidth: 128).registers)
    }
}
