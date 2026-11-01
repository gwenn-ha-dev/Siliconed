import XCTest
@testable import Siliconed

/// **The two-scale Size table** (`SizeScales`): every cell computed, not trusted — the proportion (exact, or how far), the
/// area, the multiple of 32, each family's limits — and the notation `silicontrol` reads. Pure: no weights, no machine.
final class SizeScalesTests: XCTestCase {
    private let families: [Family] = [.zImage, .anima, .krea2, .qwenImage21, .mossImage]

    private func cell(_ r: SizeRatio, _ q: SizeResolution) -> SizeCell { SizeScales.cell(r, q)! }

    /// Every ratio × resolution has a cell; every side is a multiple of 32 (Qwen-Image-2.1's rule, so all families'), at least 512,
    /// and within the area each scale names: 1K about 1 Mpx (0.9 to 1.1), 2K about 4 Mpx (3.6 to 4.2).
    func testEveryCellIsMultipleOf32AndNearItsArea() {
        for ratio in SizeScales.ratios {
            for q in SizeResolution.allCases {
                let c = cell(ratio, q)
                XCTAssertEqual(c.width % 32, 0, "\(ratio) \(q)")
                XCTAssertEqual(c.height % 32, 0, "\(ratio) \(q)")
                XCTAssertGreaterThanOrEqual(min(c.width, c.height), Format.minimumSide)
                let mpx = Double(c.area) / 1_000_000
                if q == .oneK { XCTAssertTrue((0.9...1.1).contains(mpx), "\(ratio) 1K \(mpx)") }
                else { XCTAssertTrue((3.6...4.2).contains(mpx), "\(ratio) 2K \(mpx)") }
                XCTAssertEqual(c.ratio.orientation, c.format.orientation, "\(ratio)")
            }
        }
    }

    /// The proportions the table claims, checked by arithmetic: exact everywhere but 9:16 and 16:9 at 1K, which are a 4:7
    /// (1.6 % off) and shown « ≈ ». At 2K, 1440 × 2560 is exactly 9:16.
    func testProportionsAreExactExceptTheOneKNineSixteenth() {
        for ratio in SizeScales.ratios {
            for q in SizeResolution.allCases {
                let c = cell(ratio, q)
                let exact = c.width * ratio.height == c.height * ratio.width
                let nineSixteenthAtOneK = (ratio == SizeRatio(9, 16) || ratio == SizeRatio(16, 9)) && q == .oneK
                XCTAssertEqual(exact, !nineSixteenthAtOneK, "\(ratio) \(q)")
                XCTAssertEqual(c.isApproximate, nineSixteenthAtOneK, "\(ratio) \(q)")
                XCTAssertEqual(c.deviation == 0, exact)
            }
        }
        let tall = cell(SizeRatio(9, 16), .oneK)
        XCTAssertEqual([tall.width, tall.height], [768, 1344])
        XCTAssertEqual(tall.actualRatio, SizeRatio(4, 7))
        XCTAssertEqual(tall.deviation, 0.0159, accuracy: 0.0001)
        let wide = cell(SizeRatio(16, 9), .oneK)
        XCTAssertEqual([wide.width, wide.height], [1344, 768])
        let exact = cell(SizeRatio(9, 16), .twoK)
        XCTAssertEqual([exact.width, exact.height], [1440, 2560])
        XCTAssertFalse(exact.isApproximate)
    }

    /// The table, as published (the issue's, with 9:16 at 2K made exact): the pixels of each portrait, the landscapes their turn.
    func testTheTable() {
        let portraits: [(SizeRatio, (Int, Int), (Int, Int))] = [
            (SizeRatio(1, 1), (1024, 1024), (2048, 2048)),
            (SizeRatio(4, 5), (896, 1120), (1792, 2240)),
            (SizeRatio(3, 4), (864, 1152), (1728, 2304)),
            (SizeRatio(2, 3), (832, 1248), (1664, 2496)),
            (SizeRatio(9, 16), (768, 1344), (1440, 2560)),
        ]
        for (r, one, two) in portraits {
            XCTAssertEqual([cell(r, .oneK).width, cell(r, .oneK).height], [one.0, one.1], "\(r)")
            XCTAssertEqual([cell(r, .twoK).width, cell(r, .twoK).height], [two.0, two.1], "\(r)")
            let turned = SizeRatio(r.height, r.width)
            XCTAssertEqual([cell(turned, .oneK).width, cell(turned, .oneK).height], [one.1, one.0], "\(turned)")
            XCTAssertEqual([cell(turned, .twoK).width, cell(turned, .twoK).height], [two.1, two.0], "\(turned)")
        }
        XCTAssertNil(SizeScales.cell(SizeRatio(21, 9), .oneK))
    }

    /// Per family, a cell is either rendered by `Format.check` or refused for the family's own limit — never for the shared
    /// rules (multiples, floor); only Anima refuses, and only past 2048 px a side: its 2K is the square alone.
    func testFamilyLimits() {
        for family in families {
            for ratio in SizeScales.ratios {
                for q in SizeResolution.allCases {
                    let c = cell(ratio, q)
                    let refusal = SizeScales.refusal(c, family: family)
                    let limit = Format.maximumSide(for: family)
                    if max(c.width, c.height) <= limit { XCTAssertNil(refusal, "\(family) \(ratio) \(q)") }
                    else {
                        guard case .formatRefused(_, _, .sideTooLong, let bound)? = refusal else { return XCTFail("\(family) \(ratio) \(q)") }
                        XCTAssertEqual(bound, limit)
                    }
                }
            }
        }
        for ratio in SizeScales.ratios {
            let c = cell(ratio, .twoK)
            XCTAssertEqual(SizeScales.refusal(c, family: .anima) == nil, ratio == SizeRatio(1, 1), "\(ratio)")
            XCTAssertNil(SizeScales.refusal(cell(ratio, .oneK), family: .anima))
            XCTAssertNil(SizeScales.refusal(c, family: .zImage))
            XCTAssertNil(SizeScales.refusal(c, family: .qwenImage21))
        }
    }

    /// The table finds its own cells back, and nothing else: the default menu's 896 × 1152 (7:9) is no cell.
    func testMatchingAFormat() {
        for ratio in SizeScales.ratios {
            for q in SizeResolution.allCases {
                let c = cell(ratio, q)
                XCTAssertEqual(SizeScales.cell(matching: c.format), c)
            }
        }
        XCTAssertNil(SizeScales.cell(matching: RecommendedFormat(896, 1152)))
        XCTAssertNil(SizeScales.cell(matching: RecommendedFormat(832, 1216)))
    }

    /// `3:4@2K` gives the table's pixels, case aside; the refusals say which part is wrong.
    func testSilicontrolNotation() throws {
        for ratio in SizeScales.ratios {
            for q in SizeResolution.allCases {
                XCTAssertEqual(try SizeScales.parse("\(ratio)@\(q.rawValue)"), cell(ratio, q))
                XCTAssertEqual(try SizeScales.parse("\(ratio)@\(q.rawValue.lowercased())"), cell(ratio, q))
            }
        }
        let c = try SizeScales.parse("3:4@2K")
        XCTAssertEqual([c.width, c.height], [1728, 2304])
        XCTAssertThrowsError(try SizeScales.parse("21:9@1K")) { XCTAssertEqual($0 as? SizeScaleError, .unknownRatio("21:9")) }
        XCTAssertThrowsError(try SizeScales.parse("3:4@4K")) { XCTAssertEqual($0 as? SizeScaleError, .unknownResolution("4K")) }
        for bad in ["3:4", "3@2K", "a:b@2K", "3:4@", "0:4@1K", "3:4@2K@1K"] {
            XCTAssertThrowsError(try SizeScales.parse(bad), bad) {
                if case .unreadable = $0 as! SizeScaleError {} else if bad == "3:4@" {} else { XCTFail(bad) }
            }
        }
        XCTAssertTrue(SizeScales.looksLikeScale("3:4@2K"))
        XCTAssertFalse(SizeScales.looksLikeScale("832x1216"))
        XCTAssertFalse(SizeScales.looksLikeScale("1024"))
    }

    /// The default menu is not touched: its list is the one the check has always judged.
    func testDefaultMenuUnchanged() {
        XCTAssertEqual(ModelCard.recommendedFormats.count, 10)
        XCTAssertEqual(ModelCard.recommendedFormats[3], RecommendedFormat(896, 1152))
    }
}
