import Foundation
import XCTest
@testable import Siliconed

/// **The catalogue of a render's computations** (`CapacityClasses.swift`): the lists the five families publish, the half grid, the
/// versions' formats, the merge across families and the names of the classes. No GPU, no weights: the families' `Config` defaults.
final class CapacityClassesTests: XCTestCase {
    typealias CC = CapacityClasses

    /// **The classes follow the tensors, not the name of the version**: our 4-bit Light (SQ, int8 products) is the Compact's
    /// classes, the GGUF Light's Q4_K stays its own, and nothing given means the version's.
    func testTheClassesFollowTheInstalledFormatNotTheVersionsName() {
        for family in [Family.zImage, .qwenImage21, .anima, .krea2] {
            func ids(_ variant: Variant, _ format: CC.WeightFormat?) -> [String] {
                CC.classes(for: .init(family: family, variant: variant, includesEncoder: true, ditFormat: format))!.map(\.id)
            }
            XCTAssertEqual(ids(.light, .int8), ids(.compact, nil), "\(family)")
            XCTAssertEqual(ids(.light, .q4k), ids(.light, nil), "\(family)")
            XCTAssertEqual(ids(.compact, .int8), ids(.compact, nil), "\(family)")
            XCTAssertEqual(ids(.standard, .bf16), ids(.standard, nil), "\(family)")
            XCTAssertNotEqual(ids(.light, .int8), ids(.light, nil), "\(family)")
        }
    }

    // MARK: helpers

    private func classes(_ family: Family, _ variant: Variant = .standard, w: Int = 1024, h: Int = 1024, faster: Bool = true,
                         encoder: Bool = false, prompt: Int = 11) -> [CC.Class] {
        CC.classes(for: .init(family: family, variant: variant, width: w, height: h, fasterRendering: faster,
                              promptTokens: prompt, includesEncoder: encoder))!
    }

    private func products(_ cs: [CC.Class], phase: CC.Phase = .denoising) -> [CC.Class] {
        cs.filter { if case .product = $0.shape { return $0.phase == phase } else { return false } }
    }

    private func executions(_ cs: [CC.Class], _ family: Family, k: Int, n: Int, rows: Int) -> Int {
        cs.filter { if case let .product(kk, nn, r, _) = $0.shape { return kk == k && nn == n && r == rows } else { return false } }
            .reduce(0) { $0 + ($1.executions[family] ?? 0) }
    }

    private func attentions(_ cs: [CC.Class]) -> [CC.Class] {
        cs.filter { if case .attention = $0.shape { return true } else { return false } }
    }

    private func rows(_ cs: [CC.Class]) -> Set<Int> {
        Set(products(cs).compactMap { if case let .product(_, _, r, _) = $0.shape { return r } else { return nil } })
    }

    // MARK: the published lists

    func testZImageAtFullSizeHasTheInventorysProducts() {
        let cs = classes(.zImage, faster: false)
        // 128 projections + 64 + 32 SwiGLU + 14 context refiners + 3 embedders, per evaluation, 8 evaluations.
        XCTAssertEqual(products(cs).reduce(0) { $0 + ($1.executions[.zImage] ?? 0) }, 241 * 8)
        XCTAssertEqual(rows(cs), [4128, 4096, 32])
        XCTAssertEqual(executions(cs, .zImage, k: 3840, n: 3840, rows: 4128), 120 * 8)
        XCTAssertEqual(executions(cs, .zImage, k: 3840, n: 3840, rows: 4096), 8 * 8)
        XCTAssertEqual(executions(cs, .zImage, k: 3840, n: 3840, rows: 32), 8 * 8)
        XCTAssertEqual(executions(cs, .zImage, k: 3840, n: 10240, rows: 4128), 60 * 8)
        XCTAssertEqual(executions(cs, .zImage, k: 10240, n: 3840, rows: 4128), 30 * 8)
        XCTAssertEqual(executions(cs, .zImage, k: 64, n: 3840, rows: 4096), 8)
        XCTAssertEqual(executions(cs, .zImage, k: 2560, n: 3840, rows: 32), 8)
        // The attentions: 30 layers over the joint sequence, 2 image refiners; the context refiners' 32 rows are no class.
        let a = attentions(cs)
        XCTAssertEqual(a.count, 2)
        XCTAssertEqual(a.first { $0.shape == .attention(sequence: 4128, keys: 4128, heads: 30, kvHeads: 30, headDim: 128, cross: false) }?.executions[.zImage], 30 * 8)
        XCTAssertEqual(a.first { $0.shape == .attention(sequence: 4096, keys: 4096, heads: 30, kvHeads: 30, headDim: 128, cross: false) }?.executions[.zImage], 2 * 8)
    }

    func testFasterRenderingAddsTheHalfGrid() {
        // Z-Image 1024²: 3 evaluations at 1024 + 32 rows, 5 at 4096 + 32.
        let z = classes(.zImage)
        XCTAssertEqual(rows(z), [1056, 1024, 4128, 4096, 32])
        XCTAssertEqual(executions(z, .zImage, k: 3840, n: 3840, rows: 1056), 120 * 3)
        XCTAssertEqual(executions(z, .zImage, k: 3840, n: 3840, rows: 4128), 120 * 5)
        XCTAssertEqual(executions(z, .zImage, k: 3840, n: 3840, rows: 32), 8 * 8, "the context refiners run at 32 rows at every evaluation")
        // 512²: no half grid (the 512-px floor); 1024×1536: the half grid is 512×768, 1 536 image rows.
        XCTAssertEqual(rows(classes(.zImage, w: 512, h: 512)), [1056, 1024, 32])
        XCTAssertEqual(rows(classes(.zImage, w: 1024, h: 1536)), [1568, 1536, 6176, 6144, 32])
        // The text rows and counts of the other families, from the families' configs.
        XCTAssertEqual(rows(classes(.krea2)).intersection([1040, 4112]), [1040, 4112])
        XCTAssertEqual(executions(classes(.krea2), .krea2, k: 6144, n: 6144, rows: 1040), 3 * 28 * 4, "Krea 2: 4 half-size evaluations")
        XCTAssertEqual(executions(classes(.krea2), .krea2, k: 6144, n: 6144, rows: 4112), 3 * 28 * 4)
        XCTAssertEqual(executions(classes(.mossImage), .mossImage, k: 3072, n: 3072, rows: 1045), 4 * 42, "MOSS-Image: one half-size evaluation, q k v o")
        XCTAssertEqual(executions(classes(.mossImage), .mossImage, k: 3072, n: 3072, rows: 4117), 4 * 42 * 7)
        XCTAssertEqual(executions(classes(.qwenImage21), .qwenImage21, k: 4096, n: 4096, rows: 1024), 4 * 32 * 3)
        XCTAssertEqual(executions(classes(.anima), .anima, k: 2048, n: 2048, rows: 1024), 6 * 28 * 3)
        // Not at 6 144+ rows of Krea 2 and MOSS at 512²: nothing under the floor.
        XCTAssertEqual(rows(classes(.mossImage, w: 512, h: 512)).intersection([1045]), [1045])
        XCTAssertEqual(executions(classes(.mossImage, w: 512, h: 512), .mossImage, k: 3072, n: 3072, rows: 1045), 4 * 42 * 8)
        XCTAssertFalse(classes(.zImage, faster: false).contains { if case .product(_, _, 1056, _) = $0.shape { return true } else { return false } })
    }

    func testQwenMLPRunsByChunksAndAnimaHasItsCrossAttention() {
        // 1024×1536 full grid: 6 144 rows → two MLP chunks of 3 072; one attention over 6 144 query rows and 6 144 + 19 keys.
        let q = classes(.qwenImage21, w: 1024, h: 1536)
        XCTAssertEqual(executions(q, .qwenImage21, k: 4096, n: 12288, rows: 3072), 2 * 2 * 32 * 5, "gate and proj, two chunks, 5 full evaluations")
        XCTAssertEqual(executions(q, .qwenImage21, k: 12288, n: 4096, rows: 3072), 2 * 32 * 5)
        XCTAssertEqual(executions(q, .qwenImage21, k: 4096, n: 12288, rows: 4096) + executions(q, .qwenImage21, k: 4096, n: 12288, rows: 6144), 0)
        XCTAssertNotNil(q.first { $0.shape == .attention(sequence: 6144, keys: 6144 + 19, heads: 32, kvHeads: 32, headDim: 128, cross: false) })
        // The prefill: 19 rows through every layer, the last one only for its keys and values — once per pass (the half grid and the full
        // grid each have their DiT), txt_in's two products beside them (bf16).
        XCTAssertEqual(executions(q, .qwenImage21, k: 4096, n: 4096, rows: 19), 2 * (4 * 31 + 2) + 2 * 2)
        XCTAssertEqual(executions(classes(.qwenImage21, w: 1024, h: 1536, faster: false), .qwenImage21, k: 4096, n: 4096, rows: 19), 4 * 31 + 2 + 2)
        // Anima: cross-attention keys are the 512 padded text rows, its k/v products run once per render at 512 rows.
        let a = classes(.anima, faster: false)
        XCTAssertEqual(executions(a, .anima, k: 1024, n: 2048, rows: 512), 56)
        XCTAssertEqual(executions(classes(.anima), .anima, k: 1024, n: 2048, rows: 512), 2 * 56, "the cross k/v of each of the two grids' DiT")
        let cross = a.first { $0.shape == .attention(sequence: 4096, keys: 512, heads: 16, kvHeads: 16, headDim: 128, cross: true) }
        XCTAssertEqual(cross?.executions[.anima], 28 * 8)
        // Krea 2: GQA 48/12.
        XCTAssertNotNil(classes(.krea2, faster: false).first { $0.shape == .attention(sequence: 4112, keys: 4112, heads: 48, kvHeads: 12, headDim: 128, cross: false) })
        XCTAssertEqual(executions(classes(.krea2, faster: false), .krea2, k: 6144, n: 1536, rows: 4112), 2 * 28 * 8)
        // MOSS-Image: q, k, v, o apart in the layers (the fused QKV is the two text blocks'), gate and up fused, the text blocks at 21 rows
        // with their up and gate apart.
        let m = classes(.mossImage, faster: false)
        XCTAssertEqual(executions(m, .mossImage, k: 3072, n: 9216, rows: 4117), 0)
        XCTAssertEqual(executions(m, .mossImage, k: 3072, n: 3072, rows: 4117), 4 * 42 * 8)
        XCTAssertEqual(executions(m, .mossImage, k: 3072, n: 16384, rows: 4117), 42 * 8)
        XCTAssertEqual(executions(m, .mossImage, k: 3072, n: 9216, rows: 21), 2 * 8)
        XCTAssertEqual(executions(m, .mossImage, k: 3072, n: 8192, rows: 21), 2 * 2 * 8)
        XCTAssertEqual(executions(m, .mossImage, k: 4096, n: 3072, rows: 21), 8)
    }

    /// **A LoRA's products are classes**: MOSS-Image's DMD (rank 128) rides every layer projection, two products each, `x · down` at
    /// `k → 128` and `mid · up` at `128 → n`; 2.46 TFLOP an evaluation at 1024² (the figure the plan's lot was decided on). A
    /// request's LoRAs add their rank to the stack; Qwen-Image-2.1's merged stack is not carried.
    func testMossImageCarriesItsDMDLoRAAsTwoProductsPerProjection() {
        func lora(_ cs: [CC.Class], _ format: CC.WeightFormat, k: Int, n: Int, rows: Int) -> Int {
            cs.filter { if case let .product(kk, nn, r, f) = $0.shape { return kk == k && nn == n && r == rows && f == format } else { return false } }
                .reduce(0) { $0 + ($1.executions[.mossImage] ?? 0) }
        }
        let m = classes(.mossImage, faster: false)
        // q, k, v, o and the fused gate+up read the 3 072 columns of x: five `loraDown` of 3 072 → 128 per layer; the down projection reads 8 192.
        XCTAssertEqual(lora(m, .loraDown, k: 3072, n: 128, rows: 4117), 5 * 42 * 8)
        XCTAssertEqual(lora(m, .loraDown, k: 8192, n: 128, rows: 4117), 42 * 8)
        XCTAssertEqual(lora(m, .loraUp, k: 128, n: 3072, rows: 4117), 5 * 42 * 8, "q, k, v, o and the down projection")
        XCTAssertEqual(lora(m, .loraUp, k: 128, n: 16384, rows: 4117), 42 * 8)
        XCTAssertEqual(lora(m, .loraDown, k: 3072, n: 128, rows: 21), 0, "the text blocks are bf16 products the stack does not reach")
        XCTAssertNotNil(m.first { $0.id == "product:3072x128@4117~loraDown" })
        XCTAssertNotNil(m.first { $0.id == "product:128x16384@4117~loraUp" })
        let flop = m.filter { if case let .product(_, _, rows, f) = $0.shape { return f.isLoRA && rows == 4117 } else { return false } }
            .reduce(0.0) { $0 + $1.flop * Double($1.executions[.mossImage] ?? 0) } / 8
        XCTAssertEqual(flop / 1e12, 2.45, accuracy: 0.02, "the DMD LoRA's FLOP per evaluation at 1024²")
        // The half grid's evaluations carry it at their own rows.
        XCTAssertGreaterThan(lora(classes(.mossImage), .loraUp, k: 128, n: 3072, rows: 1045), 0)
        // A request's LoRA adds its rank to the stack's.
        let more = CC.classes(for: .init(family: .mossImage, fasterRendering: false, includesEncoder: false, loraRank: 32))!
        XCTAssertEqual(lora(more, .loraDown, k: 3072, n: 160, rows: 4117), 5 * 42 * 8)
        XCTAssertEqual(lora(more, .loraDown, k: 3072, n: 128, rows: 4117), 0)
        XCTAssertEqual(CC.stackRank(.init(family: .mossImage, loraRank: 32)), 160)
    }

    func testARequestsLoRAReachesTheLayerProjectionsOfTheOtherFamilies() {
        func loras(_ family: Family, rank: Int) -> [CC.Class] {
            (CC.classes(for: .init(family: family, fasterRendering: false, includesEncoder: false, loraRank: rank)) ?? [])
                .filter { if case let .product(_, _, _, f) = $0.shape { return f.isLoRA } else { return false } }
        }
        for family in [Family.zImage, .anima, .krea2] {
            XCTAssertTrue(loras(family, rank: 0).isEmpty, "\(family): no LoRA, no class")
            let cs = loras(family, rank: 32)
            XCTAssertFalse(cs.isEmpty, "\(family)")
            for c in cs {
                guard case let .product(k, n, _, f) = c.shape else { continue }
                XCTAssertEqual(f == .loraDown ? n : k, 32, "\(family) \(c.id)")
            }
        }
        XCTAssertTrue(loras(.qwenImage21, rank: 32).isEmpty, "Qwen-Image-2.1's stack is merged where that is cheaper: not carried")
        // The Compact's formats do not touch the LoRA's operand: it is the stack's fp32.
        XCTAssertFalse((CC.classes(for: .init(family: .krea2, variant: .compact, includesEncoder: false, loraRank: 32)) ?? [])
            .filter { if case let .product(_, _, _, f) = $0.shape { return f.isLoRA } else { return false } }.isEmpty)
    }

    func testTheEncoderAtSmallM() {
        let z = CC.classes(for: .init(family: .zImage, includesEncoder: true))!
        let enc = z.filter { $0.phase == .encoding }
        XCTAssertEqual(Set(enc.compactMap { c -> Int? in if case let .product(_, _, r, _) = c.shape { return r } else { return nil } }), [19])
        XCTAssertEqual(enc.reduce(0) { $0 + ($1.executions[.zImage] ?? 0) }, 35 * 7)
        XCTAssertEqual(executions(enc, .zImage, k: 2560, n: 9728, rows: 19), 2 * 35)
        let moss = CC.classes(for: .init(family: .mossImage, includesEncoder: true))!.filter { $0.phase == .encoding }
        XCTAssertEqual(Set(moss.compactMap { c -> Int? in if case let .product(_, _, r, _) = c.shape { return r } else { return nil } }), [21])
        XCTAssertEqual(executions(CC.classes(for: .init(family: .qwenImage21))!, .qwenImage21, k: 4096, n: 12288, rows: 33), 2 * 36)
        XCTAssertEqual(executions(CC.classes(for: .init(family: .krea2))!, .krea2, k: 2560, n: 9728, rows: 50), 2 * 36)
        XCTAssertEqual(executions(CC.classes(for: .init(family: .anima))!, .anima, k: 1024, n: 3072, rows: 11), 2 * 28)
        XCTAssertTrue(classes(.zImage).allSatisfy { $0.phase == .denoising }, "an « image suivante » does not run the encoder")
    }

    func testVersionsAndTheirFormats() {
        XCTAssertEqual(CC.weightFormat(.zImage, .standard), .bf16)
        XCTAssertEqual(CC.weightFormat(.zImage, .compact), .int8)
        XCTAssertEqual(CC.weightFormat(.anima, .compact), .int8)
        XCTAssertEqual(CC.weightFormat(.krea2, .compact), .int8)
        XCTAssertEqual(CC.weightFormat(.qwenImage21, .light), .q4k)
        XCTAssertNil(CC.weightFormat(.mossImage, .light))
        XCTAssertNil(CC.classes(for: .init(family: .mossImage, variant: .light)))
        XCTAssertNil(CC.WeightFormat.q4k.kind)
        XCTAssertEqual(CC.encoderFormat(.anima, .compact), .bf16)
        XCTAssertEqual(CC.encoderFormat(.zImage, .compact), .int8)
        // The format is in the shape: the same k×n of two versions are two classes — for the blocks the version quantizes (Z-Image's layers
        // and refiners); the weights it leaves as published (its embedders) are the Standard's own classes.
        let std = classes(.zImage), cmp = classes(.zImage, .compact)
        func format(_ c: CC.Class) -> CC.WeightFormat? { if case let .product(_, _, _, f) = c.shape { return f } else { return nil } }
        XCTAssertEqual(std.count, cmp.count)
        let quantized = cmp.filter { format($0) == .int8 }, plain = cmp.filter { format($0) == .bf16 }
        XCTAssertFalse(quantized.isEmpty)
        XCTAssertTrue(Set(std.map(\.shape)).isDisjoint(with: Set(quantized.map(\.shape))))
        XCTAssertTrue(plain.allSatisfy { p in std.contains { $0.shape == p.shape } })
        XCTAssertEqual(quantized.count + plain.count, cmp.filter { format($0) != nil }.count)
    }

    // MARK: fusion

    func testIdenticalClassesMergeAcrossFamiliesAndKeepTheirCountsApart() {
        // Z-Image's encoder at 50 rows (prompt 42) and Krea 2's (prompt 11): the same Qwen3 text layers.
        let z = CC.classes(for: .init(family: .zImage, promptTokens: 42))!
        let k = CC.classes(for: .init(family: .krea2, promptTokens: 11))!
        let merged = CC.merge([z, k])
        XCTAssertLessThan(merged.count, z.count + k.count)
        let both = merged.first { c in
            if case .product(2560, 9728, 50, .bf16) = c.shape, c.phase == .encoding { return true } else { return false }
        }
        XCTAssertEqual(both?.executions, [.zImage: 2 * 35, .krea2: 2 * 36])
        // Merging changes no family's list of executions.
        for (family, own) in [(Family.zImage, z), (.krea2, k)] {
            for c in own {
                XCTAssertEqual(merged.first { $0.phase == c.phase && $0.shape == c.shape }?.executions[family], c.executions[family])
            }
        }
        // Merging a family with itself adds its executions.
        let twice = CC.merge([z, z])
        XCTAssertEqual(twice.count, z.count)
        XCTAssertEqual(twice.map { $0.executions[.zImage]! }.reduce(0, +), 2 * z.map { $0.executions[.zImage]! }.reduce(0, +))
        // Deterministic order.
        XCTAssertEqual(CC.merge([k, z]).map(\.shape), merged.map(\.shape))
    }

    // MARK: names

    /// A class is named by what the ledger can tell: the k×n at its height and the weight's form; an attention by S, heads and head dim.
    func testClassNamesAreTheLedgers() {
        XCTAssertEqual(CC.productID(k: 4096, n: 4096, m: 1024, format: .bf16), "product:4096x4096@1024")
        XCTAssertEqual(CC.productID(k: 4096, n: 4096, m: 1024, format: .int8), "product:4096x4096@1024~int8")
        XCTAssertEqual(CC.productID(k: 3840, n: 10240, m: 4128, format: .q4k), "product:3840x10240@4128~q4k")
        XCTAssertEqual(CC.attentionID(sequence: 4096, heads: 16, headDim: 128), "attention:4096/16x128")
        let anima = classes(.anima, faster: false)
        let self1 = anima.first { $0.shape == .attention(sequence: 4096, keys: 4096, heads: 16, kvHeads: 16, headDim: 128, cross: false) }
        let cross = anima.first { $0.shape == .attention(sequence: 4096, keys: 512, heads: 16, kvHeads: 16, headDim: 128, cross: true) }
        XCTAssertEqual(self1?.id, cross?.id, "a self- and a cross-attention of one S share a name: the ledger cannot tell them apart")
        // Z-Image's Compact (Comfy-Org's int8 convrot) quantizes the 30 layers (rows 4 128 and 1 056) and the refiners (rows 4 096, 1 024 and
        // the 32 of the context); the embedders stay as published, and the engine's record names them so.
        let compact = classes(.zImage, .compact).filter { $0.phase == .denoising }
        let blocks = compact.filter { ["product:3840x3840@", "product:3840x10240@", "product:10240x3840@"].contains(where: $0.id.hasPrefix) }
        XCTAssertEqual(Set(blocks.compactMap { c -> Int? in if case let .product(_, _, r, _) = c.shape { r } else { nil } }), [4128, 1056, 4096, 1024, 32])
        XCTAssertTrue(blocks.allSatisfy { $0.id.hasSuffix("~int8") })
        XCTAssertTrue(compact.filter { $0.id.contains("x64@") || $0.id.hasPrefix("product:64x") || $0.id.hasPrefix("product:2560x") }
            .allSatisfy { !$0.id.contains("~") })
    }
}
