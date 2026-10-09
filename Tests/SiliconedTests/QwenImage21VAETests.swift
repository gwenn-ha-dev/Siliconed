import XCTest
@testable import Siliconed

/// **Qwen-Image-2.1's VAE, without a byte of weights**: the widths the config gives, the latent layout
/// (no 2×2 packing), the normalization, and the index arithmetic of the two parameter-free shortcuts —
/// the part of this VAE a port by analogy gets plausibly wrong.
final class QwenImage21VAETests: XCTestCase {
    let config = QwenImage21VAE.Config()

    func testTheWidthsAreThePublishedOnes() {
        XCTAssertEqual(config.factor, 16)
        XCTAssertEqual(config.encoderDims, [96, 96, 192, 384, 768, 768])
        XCTAssertEqual(config.decoderDims, [1152, 1152, 1152, 576, 288, 144])
        // Encoder: block 0 spatial only, 1–3 temporal too, 4 neither.
        XCTAssertEqual((0..<5).map { config.down($0).temporal }, [1, 2, 2, 2, 1])
        XCTAssertEqual((0..<5).map { config.down($0).spatial }, [2, 2, 2, 2, 1])
        // Decoder: `temperal_downsample` reversed — temporal in blocks 0–2, not 3; block 4 no upsampling.
        XCTAssertEqual((0..<5).map { config.up($0).temporal }, [2, 2, 2, 1, 1])
        XCTAssertEqual((0..<5).map { config.up($0).spatial }, [2, 2, 2, 2, 1])
        XCTAssertEqual(config.mean.count, 64)
        XCTAssertEqual(config.std.count, 64)
    }

    func testTheConfigReadsThePublishedJSONAndRefusesWhatItCannotBuild() throws {
        let json = #"{"z_dim": 64, "base_dim": 96, "decoder_base_dim": 144, "dim_mult": [1, 2, 4, 8, 8],"#
            + #" "num_res_blocks": 2, "temperal_downsample": [false, true, true, true], "attn_scales": [],"#
            + #" "is_residual": true, "in_channels": 4, "out_channels": 4, "patch_size": null,"#
            + #" "latents_mean": [\#((0..<64).map { _ in "0.5" }.joined(separator: ","))],"#
            + #" "latents_std": [\#((0..<64).map { _ in "2" }.joined(separator: ","))]}"#
        let read = try QwenImage21VAE.Config(json: Data(json.utf8))
        XCTAssertEqual(read.decoderDims, config.decoderDims)
        XCTAssertEqual(read.mean, [Float](repeating: 0.5, count: 64))
        let flat = json.replacingOccurrences(of: #""is_residual": true"#, with: #""is_residual": false"#)
        XCTAssertThrowsError(try QwenImage21VAE.Config(json: Data(flat.utf8)))
        let patched = json.replacingOccurrences(of: #""patch_size": null"#, with: #""patch_size": 2"#)
        XCTAssertThrowsError(try QwenImage21VAE.Config(json: Data(patched.utf8)))
    }

    /// `_pack_latents` is a plain flatten: token `y·w + x` carries channel `c` at column `c`.
    func testPackingIsATransposeNotA2x2Patch() {
        let (c, h, w) = (3, 2, 4)
        let planar = (0..<(c * h * w)).map(Float.init)
        let tokens = QwenImage21VAE.pack(planar, channels: c, height: h, width: w)
        XCTAssertEqual(tokens[(1 * w + 2) * c + 2], planar[(2 * h + 1) * w + 2])
        XCTAssertEqual(Array(tokens[0..<3]), [0, 8, 16])
        XCTAssertEqual(QwenImage21VAE.unpack(tokens, channels: c, height: h, width: w), planar)
    }

    /// Per channel, mean and std of the published config; `normalize` inverts `denormalize` to an ulp.
    func testNormalizationIsPerChannel() {
        let plane = 6
        let z = (0..<(64 * plane)).map { Float($0 % 7) - 3 }
        let up = QwenImage21VAE.denormalize(z, config: config)
        XCTAssertEqual(up[0], z[0] * 3.2001 + 0.5126)
        XCTAssertEqual(up[63 * plane], z[63 * plane] * 3.8161 + 0.0304)
        let back = QwenImage21VAE.normalize(up[...], config: config)
        for i in 0..<z.count { XCTAssertEqual(back[i], z[i], accuracy: 1e-6) }
    }

    /// **`DupUp3D` with `first_chunk`**: the last time slot. Blocks 0–1 copy their own channel; block 2
    /// (1152 → 576, temporal) reads `2o + 1` — not `2o`; block 3 (576 → 288, spatial only) reads `2o + i`,
    /// the row parity choosing between two channels. Never the column: the graph relies on it.
    func testDupUpReadsTheLastTimeSlot() {
        func source(_ i: Int, _ o: Int, _ di: Int, _ dj: Int) -> Int {
            let b = config.up(i)
            return QwenImage21VAE.dupUpSources(input: b.input, output: b.output, temporal: b.temporal, spatial: 2)[(o * 2 + di) * 2 + dj]
        }
        for o in [0, 1, 7, 1151] { for di in 0..<2 { for dj in 0..<2 {
            XCTAssertEqual(source(0, o, di, dj), o); XCTAssertEqual(source(1, o, di, dj), o)
        } } }
        for o in [0, 1, 5, 575] { for di in 0..<2 { for dj in 0..<2 { XCTAssertEqual(source(2, o, di, dj), 2 * o + 1) } } }
        for o in [0, 1, 5, 287] { for di in 0..<2 { for dj in 0..<2 { XCTAssertEqual(source(3, o, di, dj), 2 * o + di) } } }
    }

    /// **`AvgDown3D` on one frame**: the zero frame is padded in FRONT. In the temporal blocks the even
    /// output channels are exactly zero (their whole group is padding) and the odd ones the 2×2 mean of
    /// input channel `(o − 1)/2`; block 0 is a 2×2 mean per channel; block 4 the identity.
    func testAvgDownPadsTheZeroFrameInFront() {
        func groups(_ i: Int) -> [[(c: Int, i: Int, j: Int)?]] {
            let b = config.down(i)
            return QwenImage21VAE.avgDownGroups(input: b.input, output: b.output, temporal: b.temporal, spatial: b.spatial)
        }
        let cell = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (o, group) in groups(0).enumerated() {
            XCTAssertEqual(group.map { $0!.c }, [o, o, o, o]); XCTAssertTrue(zip(group, cell).allSatisfy { $0!.i == $1.0 && $0!.j == $1.1 })
        }
        for i in 1...3 {
            for (o, group) in groups(i).enumerated() {
                XCTAssertEqual(group.count, 4)
                if o % 2 == 0 { XCTAssertTrue(group.allSatisfy { $0 == nil }, "block \(i), channel \(o)") }
                else { XCTAssertEqual(group.map { $0?.c }, [(o - 1) / 2, (o - 1) / 2, (o - 1) / 2, (o - 1) / 2]) }
            }
        }
        XCTAssertTrue(groups(4).enumerated().allSatisfy { $1.count == 1 && $1[0]!.c == $0 })
    }

    /// The encoding module's RGBA: an opaque alpha, +1 after `[-1, 1]` (PIL's `convert("RGBA")`: 255).
    func testTheReferenceBecomesOpaqueRGBA() {
        let image = ImageRGB(pixels: [Float](repeating: -0.5, count: 3 * 2 * 3), height: 2, width: 3)
        let rgba = QwenImage21EncodingModule.rgba(image)
        XCTAssertEqual(rgba.count, 4 * 6)
        XCTAssertEqual(Array(rgba[0..<18]), image.pixels)
        XCTAssertEqual(Array(rgba[18...]), [Float](repeating: 1, count: 6))
    }
}
